#include "paged_kv_pool.h"
#include "paged_sdpa_reader.h"
#include "immutable_page_pool.h"
#include <mlx/allocator.h>
#include <mlx/backend/metal/device.h>
#include <mlx/primitives.h>
#include <algorithm>
#include <atomic>
#include <limits>
#include <stdexcept>
#include <string>

#ifndef ANEMLX_PAGED_METALLIB
#error "Provide ANEMLX_PAGED_METALLIB"
#endif
#ifndef ANEMLX_PAGED_LIBRARY_ID
#error "Provide ANEMLX_PAGED_LIBRARY_ID"
#endif

namespace anemlx::pool {
namespace mx = mlx::core;
namespace meta = anern::paged;
namespace {
void require(bool condition, const char* message) {
  if (!condition) throw std::invalid_argument(message);
}
int check_rows(const mx::array& k, const mx::array& v) {
  require(k.ndim() == 4 && k.shape(0) == 1 && k.shape(1) == 2 &&
          k.shape(2) > 0 && k.shape(2) <= 131072 && k.shape(3) == 256 &&
          v.shape() == k.shape() && k.dtype() == mx::bfloat16 &&
          v.dtype() == mx::bfloat16, "Expected BF16 K/V [1,2,S,256], S in 1...131072");
  require(!k.is_tracer() && !v.is_tracer(), "Paged mutable arena does not support traced inputs");
  return k.shape(2);
}
void check_strides(const mx::array& a) {
  require(a.strides(3) == 1 && a.strides(1) > 0 && a.strides(2) > 0,
          "Paged KV import/append requires unit D stride and positive row/head strides");
}
mx::array allocate_arena(int pages) {
  const auto shape = mx::Shape{pages, 2, 32, 256};
  return mx::array(mx::allocator::malloc(uint64_t(pages) * 32768), shape, mx::bfloat16);
}
MTL::ComputePipelineState* kernel(mx::Stream stream, const char* name) {
  auto& device = mx::metal::device(stream.device);
  auto* library = device.get_library(ANEMLX_PAGED_LIBRARY_ID, ANEMLX_PAGED_METALLIB);
  auto* result = device.get_kernel(name, library);
  require(result->maxTotalThreadsPerThreadgroup() >= 256,
          "Paged KV copy/write requires 256 threads per group");
  return result;
}
}  // namespace

struct PoolImpl {
  explicit PoolImpl(int count, mx::Stream s)
      : physical_pages(count), stream(s), metadata(count),
        keys(allocate_arena(count)), values(allocate_arena(count)) {}
  const int physical_pages;
  const mx::Stream stream;
  meta::PagePool metadata;
  // Private mutable physical allocations; never exported as logical KV arrays.
  mx::array keys, values;
  std::atomic<uint64_t> encoded_writes{0}, encoded_reads{0}, encoded_materializations{0};
  std::atomic<uint64_t> copied_tail_bytes{0}, written_row_bytes{0}, materialized_bytes{0};
  std::atomic<uint64_t> in_flight_operations{0}, completed_operations{0}, failed_operations{0};
  void require_healthy() const {
    if (failed_operations.load(std::memory_order_acquire) != 0)
      throw std::runtime_error("Paged KV pool observed a Metal command failure; discard pool and recover runtime");
  }
  void submitted() noexcept { in_flight_operations.fetch_add(1, std::memory_order_relaxed); }
  void observe_status(bool success) noexcept {
    if (!success) failed_operations.fetch_add(1, std::memory_order_release);
  }
  void completed(bool) noexcept {
    completed_operations.fetch_add(1, std::memory_order_relaxed);
    in_flight_operations.fetch_sub(1, std::memory_order_relaxed);
  }
};

struct StateImpl {
  StateImpl(std::shared_ptr<PoolImpl> p, meta::PageList list, mx::array ready)
      : pool(std::move(p)), pages(std::move(list)),
        table(pages.pageIDs(), pool->physical_pages, int(pages.tokenCount())),
        ready(std::move(ready)) {}
  std::shared_ptr<PoolImpl> pool;
  meta::PageList pages;
  paged::PageTable table;
  mx::array ready;
};

namespace {
// The completion callback captures a separate pin before any dispatch. Primitive
// retention covers lazy graphs; callback retention covers detached in-flight work.
void retain_for_completion(mx::metal::CommandEncoder& encoder,
    const std::shared_ptr<PoolImpl>& pool, std::shared_ptr<const void> lease) {
  encoder.get_command_buffer()->addCompletedHandler(
      [pool, lease = std::move(lease)](MTL::CommandBuffer* buffer) mutable {
        const bool success = buffer->status() != MTL::CommandBufferStatusError;
        pool->observe_status(success);
        lease.reset();
        pool->completed(success);
      });
  pool->submitted();
}

class Write final : public mx::UnaryPrimitive {
 public:
  Write(std::shared_ptr<PoolImpl> pool, std::shared_ptr<const void> pin,
        std::vector<meta::CopySegment> copies, int old_tokens, int rows)
      : UnaryPrimitive(pool->stream), pool_(std::move(pool)), pin_(std::move(pin)),
        copies_(std::move(copies)), old_tokens_(old_tokens), rows_(rows) {}
  const char* name() const override { return "ANEMLXImmutablePageAppend32"; }
  void eval_cpu(const std::vector<mx::array>&, mx::array&) override {
    throw std::runtime_error("Paged KV writer is GPU-only");
  }
  void eval_gpu(const std::vector<mx::array>& inputs, mx::array& output) override {
    pool_->require_healthy();
    check_strides(inputs[2]); check_strides(inputs[3]);
    require(inputs[4].flags().row_contiguous, "Page IDs must be contiguous");
    auto* copy = copies_.empty() ? nullptr : kernel(stream(), "anemlx_kv_pool_copy_tail");
    auto* write = kernel(stream(), "anemlx_kv_pool_write_rows");
    output.set_data(mx::allocator::malloc(output.nbytes()));
    auto& encoder = mx::metal::get_command_encoder(stream());
    retain_for_completion(encoder, pool_, pin_);
    // Mutating only fresh slots in the arena. Register actual shared buffers as
    // outputs (not temporaries) so MLX retains its resource barrier tracking.
    auto arena_k = inputs[0], arena_v = inputs[1];
    for (const auto& segment : copies_) {
      const int source = segment.source.slot, destination = segment.destination.slot;
      const int rows = int(segment.rows);
      require(rows > 0 && rows < 32 && source != destination,
              "Tail COW must copy 1...31 rows to a distinct fresh slot");
      encoder.set_compute_pipeline_state(copy);
      encoder.set_output_array(arena_k, 0); encoder.set_output_array(arena_v, 1);
      encoder.set_bytes(source, 2); encoder.set_bytes(destination, 3); encoder.set_bytes(rows, 4);
      encoder.dispatch_threads(MTL::Size(2 * rows * 256, 1, 1), MTL::Size(256, 1, 1));
      pool_->copied_tail_bytes.fetch_add(uint64_t(rows) * meta::kRowPairBytes,
                                         std::memory_order_relaxed);
    }
    encoder.set_compute_pipeline_state(write);
    encoder.set_input_array(inputs[2], 0); encoder.set_input_array(inputs[3], 1);
    encoder.set_output_array(arena_k, 2); encoder.set_output_array(arena_v, 3);
    encoder.set_input_array(inputs[4], 4);
    encoder.set_bytes(old_tokens_, 5); encoder.set_bytes(rows_, 6);
    const uint64_t kh = inputs[2].strides(1), kr = inputs[2].strides(2);
    const uint64_t vh = inputs[3].strides(1), vr = inputs[3].strides(2);
    encoder.set_bytes(kh, 7); encoder.set_bytes(kr, 8);
    encoder.set_bytes(vh, 9); encoder.set_bytes(vr, 10);
    encoder.set_output_array(output, 11);
    encoder.dispatch_threads(MTL::Size(2 * rows_ * 256, 1, 1), MTL::Size(256, 1, 1));
    pool_->written_row_bytes.fetch_add(uint64_t(rows_) * meta::kRowPairBytes,
                                       std::memory_order_relaxed);
    pool_->encoded_writes.fetch_add(1, std::memory_order_relaxed);
  }
 private:
  std::shared_ptr<PoolImpl> pool_;
  std::shared_ptr<const void> pin_;
  std::vector<meta::CopySegment> copies_;
  int old_tokens_, rows_;
};

class Materialize final : public mx::Primitive {
 public:
  explicit Materialize(std::shared_ptr<const StateImpl> state)
      : Primitive(state->pool->stream), state_(std::move(state)) {}
  const char* name() const override { return "ANEMLXMaterializeImmutablePages32"; }
  void eval_cpu(const std::vector<mx::array>&, std::vector<mx::array>&) override {
    throw std::runtime_error("Paged KV materialization is GPU-only");
  }
  void eval_gpu(const std::vector<mx::array>& inputs, std::vector<mx::array>& outputs) override {
    state_->pool->require_healthy();
    auto* gather = kernel(stream(), "anemlx_kv_pool_materialize");
    require(inputs[2].flags().row_contiguous, "Page IDs must be contiguous");
    for (auto& output : outputs) output.set_data(mx::allocator::malloc(output.nbytes()));
    auto& encoder = mx::metal::get_command_encoder(stream());
    retain_for_completion(encoder, state_->pool, state_);
    encoder.set_compute_pipeline_state(gather);
    encoder.set_input_array(inputs[0], 0); encoder.set_input_array(inputs[1], 1);
    encoder.set_input_array(inputs[2], 2);
    encoder.set_output_array(outputs[0], 3); encoder.set_output_array(outputs[1], 4);
    const int tokens = int(state_->pages.tokenCount());
    encoder.set_bytes(tokens, 5);
    encoder.dispatch_threads(MTL::Size(2 * tokens * 256, 1, 1), MTL::Size(256, 1, 1));
    state_->pool->materialized_bytes.fetch_add(uint64_t(tokens) * meta::kRowPairBytes,
                                              std::memory_order_relaxed);
    state_->pool->encoded_materializations.fetch_add(1, std::memory_order_relaxed);
  }
 private:
  std::shared_ptr<const StateImpl> state_;
};

std::shared_ptr<const StateImpl> append_impl(const std::shared_ptr<PoolImpl>& pool,
    const meta::PageList& source, const std::optional<mx::array>& previous_ready,
    const mx::array& k, const mx::array& v) {
  pool->require_healthy();
  const int rows = check_rows(k, v), old_tokens = int(source.tokenCount());
  require(old_tokens <= 131072 - rows, "Paged KV logical length exceeds 131072");
  pool->metadata.validate(source);
  auto transaction = pool->metadata.planAppend(source, size_t(rows));
  // Copy the lease pin before publication; it owns both COW source and fresh
  // destination even if the resulting State is immediately discarded.
  auto pin = transaction.lifetimePin();
  const auto& ids = transaction.destinationPageIDs();
  mx::array device_ids(ids.begin(), mx::Shape{int(ids.size())}, mx::int32);
  std::vector<mx::array> inputs{pool->keys, pool->values, k, v, device_ids};
  if (previous_ready) inputs.push_back(*previous_ready);
  auto ticket = mx::array({1}, mx::uint32,
      std::make_shared<Write>(pool, std::move(pin), transaction.copies(), old_tokens, rows),
      std::move(inputs));
  auto list = transaction.commit();
  return std::make_shared<StateImpl>(pool, std::move(list), std::move(ticket));
}
}  // namespace

std::shared_ptr<Pool> Pool::create(int physical_pages, mx::Stream stream) {
  require(physical_pages > 0 && physical_pages <= 4096, "Physical pages must be in 1...4096");
  require(stream.device.type == mx::Device::gpu, "Paged KV pool requires explicit GPU stream");
  return std::shared_ptr<Pool>(new Pool(std::make_shared<PoolImpl>(physical_pages, stream)));
}
State Pool::import_kv(const mx::array& k, const mx::array& v) {
  return State(append_impl(impl_, impl_->metadata.empty(), std::nullopt, k, v));
}
Stats Pool::stats() const {
  const auto pages = impl_->metadata.stats();
  Stats s;
  s.physical_pages = impl_->physical_pages;
  s.arena_logical_bytes = uint64_t(impl_->physical_pages) * meta::kPagePairBytes;
  s.arena_allocated_bytes = impl_->keys.buffer_size() + impl_->values.buffer_size();
  s.live_pages = pages.uniqueLivePages; s.free_pages = pages.freePages;
  s.high_water_pages = pages.peakUniqueLivePages;
  s.encoded_writes = impl_->encoded_writes.load(std::memory_order_relaxed);
  s.encoded_reads = impl_->encoded_reads.load(std::memory_order_relaxed);
  s.encoded_materializations = impl_->encoded_materializations.load(std::memory_order_relaxed);
  s.copied_tail_bytes = impl_->copied_tail_bytes.load(std::memory_order_relaxed);
  s.written_row_bytes = impl_->written_row_bytes.load(std::memory_order_relaxed);
  s.materialized_bytes = impl_->materialized_bytes.load(std::memory_order_relaxed);
  s.in_flight_operations = impl_->in_flight_operations.load(std::memory_order_relaxed);
  s.completed_operations = impl_->completed_operations.load(std::memory_order_relaxed);
  s.failed_operations = impl_->failed_operations.load(std::memory_order_relaxed);
  s.key_buffer_identity = reinterpret_cast<uintptr_t>(impl_->keys.buffer().ptr());
  s.value_buffer_identity = reinterpret_cast<uintptr_t>(impl_->values.buffer().ptr());
  return s;
}
int State::logical_tokens() const { return int(impl_->pages.tokenCount()); }
std::vector<int32_t> State::page_ids() const { return impl_->pages.pageIDs(); }
State State::fork() const { return State(impl_); }
mx::array State::ready() const {
  impl_->pool->require_healthy();
  return impl_->ready;
}
State State::append(const mx::array& k, const mx::array& v) const {
  return State(append_impl(impl_->pool, impl_->pages, impl_->ready, k, v));
}
mx::array State::read(const mx::array& q, const std::optional<mx::array>& mask) const {
  impl_->pool->require_healthy();
  require(!q.is_tracer() && (!mask || !mask->is_tracer()), "Paged KV reader does not support traced inputs");
  impl_->pool->metadata.validate(impl_->pages);
  paged::ReaderLifetime life;
  life.owner = impl_; life.dependency = impl_->ready;
  auto pool = impl_->pool;
  life.validate = [pool] { pool->require_healthy(); };
  life.submitted = [pool] { pool->submitted(); };
  life.encoded = [pool] { pool->encoded_reads.fetch_add(1, std::memory_order_relaxed); };
  life.observed_status = [pool](bool success) { pool->observe_status(success); };
  life.completed = [pool](bool success) { pool->completed(success); };
  return paged::read(q, pool->keys, pool->values, impl_->table,
      paged::StorageKind::PageMajor, mask, pool->stream, std::move(life));
}
std::pair<mx::array, mx::array> State::materialize() const {
  impl_->pool->require_healthy();
  impl_->pool->metadata.validate(impl_->pages);
  const auto& ids = impl_->pages.pageIDs();
  mx::array device_ids(ids.begin(), mx::Shape{int(ids.size())}, mx::int32);
  const auto shape = mx::Shape{1, 2, logical_tokens(), 256};
  auto outputs = mx::array::make_arrays({shape, shape}, {mx::bfloat16, mx::bfloat16},
      std::make_shared<Materialize>(impl_),
      {impl_->pool->keys, impl_->pool->values, device_ids, impl_->ready});
  return {std::move(outputs[0]), std::move(outputs[1])};
}
}  // namespace anemlx::pool
