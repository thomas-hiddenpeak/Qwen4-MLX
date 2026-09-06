// Diagnostic-only instrumentation for the pinned MLX Metal command buffers.
// Included once by a COPY of Apple's device.cpp; original sources stay intact.
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>
#include <mach/mach_time.h>
#include <json.hpp>

extern "C" __attribute__((visibility("default"))) int anemlx_timing_start(const char* output_path);
extern "C" __attribute__((visibility("default"))) int anemlx_timing_stop(void);
extern "C" __attribute__((visibility("default"))) const char* anemlx_timing_version(void);

namespace anemlx_timing {
constexpr size_t capacity = 262144;
constexpr uint64_t clock_tolerance_ns = 1000;
constexpr const char* version = "anemlx-command-timing-v1";

inline uint64_t now_ns() {
  static const mach_timebase_info_data_t base = [] {
    mach_timebase_info_data_t value{};
    if (mach_timebase_info(&value) != KERN_SUCCESS || value.denom == 0) {
      throw std::runtime_error("mach timebase unavailable");
    }
    return value;
  }();
  return static_cast<uint64_t>((static_cast<__uint128_t>(mach_absolute_time()) * base.numer) / base.denom);
}

inline uint64_t seconds_ns(double value) noexcept {
  const double ns = value * 1e9;
  if (!std::isfinite(ns) || ns <= 0 || ns >= static_cast<double>(std::numeric_limits<uint64_t>::max())) return 0;
  return static_cast<uint64_t>(std::round(ns));
}

struct Record {
  uint64_t sequence = 0, cpu_commit_ns = 0, completion_ns = 0;
  uint64_t gpu_start_ns = 0, gpu_end_ns = 0, kernel_start_ns = 0, kernel_end_ns = 0;
  uint64_t buffer_sizes_elements = 0;
  double gpu_start_seconds = 0, gpu_end_seconds = 0;
  int buffer_ops = 0, status = 0;
  bool gpu_timestamp_valid = false, host_clock_bracket_valid = false;
};

struct Slot {
  Record record;
  std::atomic<uint64_t> commit_ns{0};
  std::atomic<bool> ready{false};
};

struct Session {
  std::unique_ptr<Slot[]> slots{new Slot[capacity]};
  std::atomic<uint64_t> attempted{0}, dropped{0}, pending{0}, reservers{0};
  int fd = -1;
  bool open = false;
  uint64_t start_ns = 0;
};

// Deliberately retained until process exit: a late callback after a timed-out
// stop must never dereference freed storage. start/stop require quiescent MLX.
static Session* retained_session = nullptr;
static std::atomic<Session*> active_session{nullptr};
static std::mutex api_mutex;
static bool exit_handler_registered = false;

struct Ticket { Session* session = nullptr; size_t index = 0; };

inline Ticket reserve(int ops, size_t elements) noexcept {
  auto* s = active_session.load(std::memory_order_acquire);
  if (!s) return {};
  s->reservers.fetch_add(1, std::memory_order_acq_rel);
  if (active_session.load(std::memory_order_acquire) != s) {
    s->reservers.fetch_sub(1, std::memory_order_release);
    return {};
  }
  const auto index = s->attempted.fetch_add(1, std::memory_order_relaxed);
  if (index >= capacity) {
    s->dropped.fetch_add(1, std::memory_order_relaxed);
    s->reservers.fetch_sub(1, std::memory_order_release);
    return {};
  }
  auto& slot = s->slots[index];
  slot.record.sequence = index + 1;
  slot.record.buffer_ops = ops;
  slot.record.buffer_sizes_elements = elements;
  s->pending.fetch_add(1, std::memory_order_release);
  s->reservers.fetch_sub(1, std::memory_order_release);
  return {s, static_cast<size_t>(index)};
}

inline void before_commit(Ticket ticket) noexcept {
  if (ticket.session) ticket.session->slots[ticket.index].commit_ns.store(now_ns(), std::memory_order_release);
}

inline void completed(Ticket ticket, MTL::CommandBuffer* buffer) noexcept {
  if (!ticket.session) return;
  auto& slot = ticket.session->slots[ticket.index];
  auto& r = slot.record;
  r.completion_ns = now_ns();
  r.cpu_commit_ns = slot.commit_ns.load(std::memory_order_acquire);
  r.gpu_start_seconds = buffer->GPUStartTime();
  r.gpu_end_seconds = buffer->GPUEndTime();
  r.gpu_start_ns = seconds_ns(r.gpu_start_seconds);
  r.gpu_end_ns = seconds_ns(r.gpu_end_seconds);
  // These are CPU-driver scheduling times, not GPU shader-kernel timings.
  r.kernel_start_ns = seconds_ns(buffer->kernelStartTime());
  r.kernel_end_ns = seconds_ns(buffer->kernelEndTime());
  r.status = static_cast<int>(buffer->status());
  r.gpu_timestamp_valid = r.gpu_start_ns > 0 && r.gpu_end_ns >= r.gpu_start_ns;
  const bool starts_after_commit = r.gpu_start_ns >= r.cpu_commit_ns || r.cpu_commit_ns - r.gpu_start_ns <= clock_tolerance_ns;
  const bool ends_before_callback = r.gpu_end_ns <= r.completion_ns || r.gpu_end_ns - r.completion_ns <= clock_tolerance_ns;
  r.host_clock_bracket_valid = r.gpu_timestamp_valid && r.cpu_commit_ns > 0 && starts_after_commit && ends_before_callback;
  // Publish the complete POD record before announcing callback completion.
  slot.ready.store(true, std::memory_order_release);
  ticket.session->pending.fetch_sub(1, std::memory_order_release);
}

inline nlohmann::json nullable_ns(uint64_t value) { return value ? nlohmann::json(value) : nlohmann::json(nullptr); }
inline nlohmann::json finite_value(double value) { return std::isfinite(value) ? nlohmann::json(value) : nlohmann::json(nullptr); }

inline nlohmann::json record_json(const Record& r) {
  return {{"sequence", r.sequence}, {"cpu_commit_ns", r.cpu_commit_ns}, {"completion_ns", r.completion_ns},
          {"gpu_start_ns", nullable_ns(r.gpu_start_ns)}, {"gpu_end_ns", nullable_ns(r.gpu_end_ns)},
          {"gpu_start_seconds", finite_value(r.gpu_start_seconds)}, {"gpu_end_seconds", finite_value(r.gpu_end_seconds)},
          {"kernel_start_ns", nullable_ns(r.kernel_start_ns)}, {"kernel_end_ns", nullable_ns(r.kernel_end_ns)},
          {"status", r.status}, {"buffer_ops", r.buffer_ops}, {"buffer_sizes_elements", r.buffer_sizes_elements},
          {"gpu_timestamp_valid", r.gpu_timestamp_valid}, {"host_clock_bracket_valid", r.host_clock_bracket_valid}};
}

inline void exit_flush() { if (retained_session && retained_session->open) (void)anemlx_timing_stop(); }
} // namespace anemlx_timing

extern "C" __attribute__((visibility("default"))) const char* anemlx_timing_version(void) {
  return anemlx_timing::version;
}

extern "C" __attribute__((visibility("default"))) int anemlx_timing_start(const char* output_path) {
  using namespace anemlx_timing;
  std::lock_guard<std::mutex> lock(api_mutex);
  try {
    if (!output_path || !*output_path || active_session.load(std::memory_order_acquire)) return 1;
    if (!retained_session) {
      retained_session = new Session;
    }
    if (!exit_handler_registered) {
      if (std::atexit(exit_flush) != 0) return 2;
      exit_handler_registered = true;
    }
    auto* s = retained_session;
    if (s->open || s->pending.load(std::memory_order_acquire) || s->reservers.load(std::memory_order_acquire)) return 3;
    // O_EXCL reserves this exact inode; stop writes the same descriptor.
    const int fd = ::open(output_path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) return 4;
    s->fd = fd;
    for (size_t i = 0; i < capacity; ++i) {
      s->slots[i].record = {};
      s->slots[i].commit_ns.store(0, std::memory_order_relaxed);
      s->slots[i].ready.store(false, std::memory_order_relaxed);
    }
    s->attempted.store(0, std::memory_order_relaxed);
    s->dropped.store(0, std::memory_order_relaxed);
    s->start_ns = now_ns(); // Initializes the clock before any hot callback.
    s->open = true;
    active_session.store(s, std::memory_order_release);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "anemlx timing start failed: %s\n", e.what());
    if (retained_session && retained_session->fd >= 0 && !retained_session->open) {
      ::close(retained_session->fd); retained_session->fd = -1;
    }
    return 5;
  }
}

extern "C" __attribute__((visibility("default"))) int anemlx_timing_stop(void) {
  using namespace anemlx_timing;
  std::lock_guard<std::mutex> lock(api_mutex);
  auto* s = retained_session;
  if (!s || !s->open) return 1;
  active_session.store(nullptr, std::memory_order_release);
  try {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while ((s->pending.load(std::memory_order_acquire) || s->reservers.load(std::memory_order_acquire)) &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const auto end_ns = now_ns();
    const auto attempted = s->attempted.load(std::memory_order_acquire);
    const auto recorded = std::min<uint64_t>(attempted, capacity);
    const auto dropped = s->dropped.load(std::memory_order_acquire);
    std::vector<size_t> ready;
    ready.reserve(recorded);
    uint64_t failed_status = 0, missing_timestamps = 0, clock_mismatches = 0;
    for (size_t i = 0; i < recorded; ++i) {
      if (!s->slots[i].ready.load(std::memory_order_acquire)) continue;
      ready.push_back(i);
      const auto& r = s->slots[i].record;
      failed_status += r.status != 4;
      missing_timestamps += r.buffer_ops > 0 && !r.gpu_timestamp_valid;
      clock_mismatches += r.gpu_timestamp_valid && !r.host_clock_bracket_valid;
    }
    std::vector<std::string> errors;
    if (ready.size() != recorded || s->reservers.load(std::memory_order_acquire)) errors.push_back("Pending callbacks or reservations did not complete within the bounded stop window");
    if (dropped) errors.push_back("Fixed record capacity exceeded; dropped buffers are not zero-duration work");
    if (failed_status) errors.push_back("One or more command buffers did not report Completed status");
    if (missing_timestamps) errors.push_back("Nonempty command buffers have unavailable GPU timestamps");
    if (clock_mismatches) errors.push_back("GPU host timestamps violate the CPU commit/completion bracket");
    const bool complete = errors.empty();
    nlohmann::json metadata = {
        {"schema_version", 1}, {"version", version}, {"process_id", ::getpid()}, {"clock", "mach_absolute_time_nanoseconds"},
        {"gpu_clock", "MTLCommandBuffer host time seconds converted to nanoseconds"},
        {"clock_bracket_tolerance_ns", clock_tolerance_ns}, {"start_ns", s->start_ns}, {"end_ns", end_ns},
        {"complete", complete}, {"errors", errors}, {"capacity", capacity}, {"attempted_buffers", attempted},
        {"recorded_buffers", recorded}, {"completed_buffers", ready.size()}, {"pending_buffers", recorded - ready.size()},
        {"dropped_buffers", dropped}, {"failed_status_buffers", failed_status},
        {"missing_gpu_timestamp_buffers", missing_timestamps}, {"clock_mismatch_buffers", clock_mismatches},
        {"physical_dram_bytes", nullptr}, {"physical_dram_bandwidth_gbps", nullptr},
        {"notes", {"GPU intervals are whole command-buffer spans, not shader-active time or DRAM bandwidth.",
                   "kernel_start_ns/kernel_end_ns describe CPU-driver scheduling, not GPU shader kernels.",
                   "buffer_sizes_elements is MLX array registration in elements, not bytes or physical traffic.",
                   "Start/stop require the caller to synchronize MLX and prevent concurrent submission.",
                   "No callback file I/O, dynamic allocation, added GPU kernel, or additional GPU wait is introduced.",
                   "Explicit stop is preferred; process-exit flush is best effort and cannot survive SIGKILL."}}};
    FILE* file = ::fdopen(s->fd, "w");
    if (!file) { ::close(s->fd); s->fd = -1; s->open = false; return 6; }
    std::unique_ptr<FILE, decltype(&std::fclose)> owned_file(file, &std::fclose);
    s->fd = -1;
    std::string prefix = metadata.dump();
    prefix.pop_back();
    bool io_ok = std::fputs(prefix.c_str(), file) >= 0 && std::fputs(",\"records\":[", file) >= 0;
    bool first = true;
    for (const auto index : ready) {
      const auto row = record_json(s->slots[index].record).dump();
      if (!first) io_ok = (std::fputc(',', file) != EOF) && io_ok;
      first = false;
      io_ok = (std::fputs(row.c_str(), file) >= 0) && io_ok;
    }
    io_ok = (std::fputs("]}\n", file) >= 0) && io_ok;
    io_ok = (std::fclose(owned_file.release()) == 0) && io_ok;
    s->open = false;
    return !io_ok ? 7 : complete ? 0 : 8;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "anemlx timing stop failed: %s\n", e.what());
    if (s->fd >= 0) ::close(s->fd);
    s->fd = -1; s->open = false;
    return 9;
  }
}
