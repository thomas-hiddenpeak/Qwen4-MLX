// Thin bridge to the separately reviewed pinned-MLX paged vector reader.
// Compile alongside ../native/paged_sdpa_reader.cpp, never the standalone main.
#include "paged_sdpa_bridge.h"
#include "paged_sdpa_reader.h"
#include "mlx/c/private/array.h"
#include "mlx/c/private/stream.h"
#include <cstdio>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

#define ANEMLX_EXPORT extern "C" __attribute__((visibility("default")))
namespace {
namespace paged = anemlx::paged;
thread_local char last_error[1024]{};
using Context = paged::IdentityPageTable;
template <class Function> int32_t checked(Function&& body) noexcept {
  try {
    body();
    last_error[0] = '\0';
    return 0;
  } catch (const std::exception& error) {
    std::snprintf(last_error, sizeof(last_error), "%s", error.what());
  } catch (...) {
    std::snprintf(last_error, sizeof(last_error), "%s", "unknown paged SDPA bridge error");
  }
  return 1;
}
}  // namespace

ANEMLX_EXPORT int32_t anemlx_paged_sdpa_version(void) { return 1; }
ANEMLX_EXPORT const char* anemlx_paged_sdpa_last_error(void) { return last_error; }
ANEMLX_EXPORT int32_t anemlx_paged_sdpa_create(void** context, int32_t maximum_tokens) {
  return checked([&] {
    if (!context || *context) throw std::invalid_argument("create requires an empty output context");
    if (maximum_tokens < 1 || maximum_tokens > 131072)
      throw std::invalid_argument("paged SDPA maximum_tokens must be 1...131072");
    auto value = std::make_unique<Context>(maximum_tokens);
    *context = value.release();
  });
}
ANEMLX_EXPORT void anemlx_paged_sdpa_free(void* context) {
  delete static_cast<Context*>(context);
}
ANEMLX_EXPORT uint64_t anemlx_paged_sdpa_metadata_bytes(const void* context) {
  return context ? static_cast<const Context*>(context)->metadata_bytes() : 0;
}
ANEMLX_EXPORT uint64_t anemlx_paged_sdpa_encoded_reads(void) { return paged::encoded_reads(); }

ANEMLX_EXPORT int32_t anemlx_paged_sdpa_read(mlx_array* output, const void* context,
    mlx_array queries, mlx_array keys, mlx_array values, mlx_array mask, mlx_stream stream) {
  return checked([&] {
    if (!output || !context || !queries.ctx || !keys.ctx || !values.ctx || !stream.ctx)
      throw std::invalid_argument("paged SDPA read received a missing handle");
    std::optional<mlx::core::array> visibility;
    if (mask.ctx) visibility.emplace(mlx_array_get_(mask));
    auto result = paged::read_identity(mlx_array_get_(queries), mlx_array_get_(keys),
        mlx_array_get_(values), *static_cast<const Context*>(context), visibility,
        mlx_stream_get_(stream));
    mlx_array_set_(*output, std::move(result));
  });
}
ANEMLX_EXPORT int32_t anemlx_paged_sdpa_dispatch_info(int32_t tokens, mlx_stream stream,
    uint64_t* scalars, size_t count) {
  return checked([&] {
    if (!scalars || count != 3 || !stream.ctx)
      throw std::invalid_argument("dispatch_info requires a stream and three output scalars");
    const auto plan = paged::dispatch_info(tokens, mlx_stream_get_(stream));
    scalars[0] = plan.two_pass ? 1 : 0;
    scalars[1] = static_cast<uint64_t>(plan.blocks);
    scalars[2] = plan.logical_scratch_bytes;
  });
}
