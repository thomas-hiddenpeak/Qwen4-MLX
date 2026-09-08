// Pinned MLX-C private getter returns a borrowed reference, never an owner.
// The exported ABI only contains the public mlx_array and integer scalars.
#include "kv_allocation_diagnostics.h"
#include "mlx/c/private/array.h"
#include <cstdio>
#include <exception>
#include <stdexcept>

#define ANEMLX_EXPORT extern "C" __attribute__((visibility("default")))
namespace { thread_local char last_error[1024]{}; }

ANEMLX_EXPORT int32_t anemlx_kv_allocation_diagnostics_version(void) { return 1; }
ANEMLX_EXPORT const char* anemlx_kv_allocation_diagnostics_last_error(void) {
  return last_error;
}
ANEMLX_EXPORT int32_t anemlx_kv_allocation_snapshot(
    mlx_array input, uint64_t* scalars, size_t count) {
  try {
    if (!scalars || count != 8) throw std::invalid_argument("expected eight output scalars");
    const auto& array = mlx_array_get_(input);
    if (!array.is_available()) throw std::invalid_argument("array must be available before observation");
    if (!array.buffer().ptr() || array.offset() < 0)
      throw std::invalid_argument("array has no bounded allocation");
    // For the pinned Metal allocator, Buffer::ptr() is the actual MTL::Buffer*.
    // Do not replace this with C handle/ArrayDesc identity or copy the array.
    const uint64_t values[8] = {
      static_cast<uint64_t>(reinterpret_cast<uintptr_t>(array.buffer().ptr())),
      static_cast<uint64_t>(array.offset()),
      static_cast<uint64_t>(array.buffer_size()),
      static_cast<uint64_t>(array.nbytes()),
      static_cast<uint64_t>(array.data_size()),
      static_cast<uint64_t>(array.flags().contiguous),
      static_cast<uint64_t>(array.flags().row_contiguous),
      static_cast<uint64_t>(array.is_donatable())};
    for (size_t i = 0; i < 8; ++i) scalars[i] = values[i];
    last_error[0] = '\0';
    return 0;
  } catch (const std::exception& error) {
    std::snprintf(last_error, sizeof(last_error), "%s", error.what());
    return 1;
  } catch (...) {
    std::snprintf(last_error, sizeof(last_error), "%s", "unknown native allocation diagnostic error");
    return 1;
  }
}
