#include "paged_kv_pool_bridge.h"
#include "paged_kv_pool.h"
#include "mlx/c/private/array.h"
#include "mlx/c/private/stream.h"
#include <algorithm>
#include <cstdio>
#include <exception>
#include <iterator>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

#define ANEMLX_EXPORT extern "C" __attribute__((visibility("default")))
namespace {
namespace pool = anemlx::pool;
using PoolHandle = std::shared_ptr<pool::Pool>;
using State = pool::State;
thread_local char last_error[1024]{};
template <class Body> int32_t checked(Body&& body) noexcept {
  try {
    body(); last_error[0] = '\0'; return 0;
  } catch (const std::exception& e) {
    std::snprintf(last_error, sizeof(last_error), "%s", e.what());
  } catch (...) {
    std::snprintf(last_error, sizeof(last_error), "%s", "unknown physical KV pool error");
  }
  return 1;
}
void empty_output(void** output) {
  if (!output || *output) throw std::invalid_argument("requires an empty output handle");
}
const State& state_value(const void* state) {
  if (!state) throw std::invalid_argument("missing physical KV state");
  return *static_cast<const State*>(state);
}
const PoolHandle& pool_value(const void* value) {
  if (!value) throw std::invalid_argument("missing physical KV pool");
  return *static_cast<const PoolHandle*>(value);
}
}  // namespace

ANEMLX_EXPORT int32_t anemlx_paged_kv_pool_version(void) { return 1; }
ANEMLX_EXPORT const char* anemlx_paged_kv_pool_last_error(void) { return last_error; }
ANEMLX_EXPORT int32_t anemlx_paged_kv_pool_create(void** output, int32_t pages, mlx_stream stream) {
  return checked([&] {
    empty_output(output);
    if (!stream.ctx || pages < 1 || pages > 4096)
      throw std::invalid_argument("physical KV pool requires GPU stream and 1...4096 pages");
    auto value = std::make_unique<PoolHandle>(pool::Pool::create(pages, mlx_stream_get_(stream)));
    *output = value.release();
  });
}
ANEMLX_EXPORT void anemlx_paged_kv_pool_free(void* value) { delete static_cast<PoolHandle*>(value); }
ANEMLX_EXPORT void anemlx_paged_kv_pool_state_free(void* value) { delete static_cast<State*>(value); }
ANEMLX_EXPORT int32_t anemlx_paged_kv_pool_import(void** output, const void* value, mlx_array k, mlx_array v) {
  return checked([&] {
    empty_output(output);
    if (!k.ctx || !v.ctx) throw std::invalid_argument("missing import tensors");
    auto state = std::make_unique<State>(pool_value(value)->import_kv(mlx_array_get_(k), mlx_array_get_(v)));
    *output = state.release();
  });
}
ANEMLX_EXPORT int32_t anemlx_paged_kv_pool_fork(void** output, const void* source) {
  return checked([&] {
    empty_output(output);
    auto state = std::make_unique<State>(state_value(source).fork());
    *output = state.release();
  });
}
ANEMLX_EXPORT int32_t anemlx_paged_kv_pool_append(void** output, const void* source, mlx_array k, mlx_array v) {
  return checked([&] {
    empty_output(output);
    if (!k.ctx || !v.ctx) throw std::invalid_argument("missing append tensors");
    auto state = std::make_unique<State>(state_value(source).append(mlx_array_get_(k), mlx_array_get_(v)));
    *output = state.release();
  });
}
ANEMLX_EXPORT int32_t anemlx_paged_kv_pool_state_info(const void* state, uint64_t* out, size_t count) {
  return checked([&] {
    if (!out || count != 2) throw std::invalid_argument("state info requires two scalar outputs");
    const auto& value = state_value(state);
    out[0] = uint64_t(value.logical_tokens());
    out[1] = uint64_t((value.logical_tokens() + 31) / 32);
  });
}
ANEMLX_EXPORT int32_t anemlx_paged_kv_pool_page_ids(const void* state, int32_t* out, size_t count) {
  return checked([&] {
    const auto pages = state_value(state).page_ids();
    if (count != pages.size() || (count && !out)) throw std::invalid_argument("invalid page-ID output capacity");
    std::copy(pages.begin(), pages.end(), out);
  });
}
ANEMLX_EXPORT int32_t anemlx_paged_kv_pool_ready(mlx_array* out, const void* state) {
  return checked([&] {
    if (!out) throw std::invalid_argument("missing ready output");
    mlx_array_set_(*out, state_value(state).ready());
  });
}
ANEMLX_EXPORT int32_t anemlx_paged_kv_pool_read(mlx_array* out, const void* state, mlx_array q, mlx_array mask) {
  return checked([&] {
    if (!out || !q.ctx) throw std::invalid_argument("missing read output/query");
    std::optional<mlx::core::array> visibility;
    if (mask.ctx) visibility.emplace(mlx_array_get_(mask));
    mlx_array_set_(*out, state_value(state).read(mlx_array_get_(q), visibility));
  });
}
ANEMLX_EXPORT int32_t anemlx_paged_kv_pool_materialize(mlx_array* keys, mlx_array* values, const void* state) {
  return checked([&] {
    if (!keys || !values || keys == values) throw std::invalid_argument("requires distinct materialization outputs");
    auto result = state_value(state).materialize();
    mlx_array_set_(*keys, std::move(result.first));
    mlx_array_set_(*values, std::move(result.second));
  });
}

ANEMLX_EXPORT int32_t anemlx_paged_kv_pool_statistics(const void* value, uint64_t* out, size_t count) {
  return checked([&] {
    if (!out || count != 17) throw std::invalid_argument("statistics require 17 scalar outputs");
    const auto stats = pool_value(value)->stats();
    const uint64_t fields[] = {
      stats.physical_pages, stats.arena_logical_bytes, stats.arena_allocated_bytes,
      stats.live_pages, stats.free_pages, stats.high_water_pages,
      stats.encoded_writes, stats.encoded_reads, stats.encoded_materializations,
      stats.copied_tail_bytes, stats.written_row_bytes, stats.materialized_bytes,
      stats.in_flight_operations, stats.completed_operations, stats.failed_operations,
      stats.key_buffer_identity, stats.value_buffer_identity};
    std::copy(std::begin(fields), std::end(fields), out);
  });
}
