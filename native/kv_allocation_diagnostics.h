#pragma once
#include <stddef.h>
#include <stdint.h>
#include "mlx/c/array.h"

#ifdef __cplusplus
extern "C" {
#endif
int32_t anemlx_kv_allocation_diagnostics_version(void);
const char* anemlx_kv_allocation_diagnostics_last_error(void);
// Eight scalars: MTL::Buffer pointer, byte offset, allocation bytes, logical
// bytes, data extent in elements, contiguous, row-contiguous, donatable now.
// Input must already be available. No eval, synchronization, readback, array
// copy, shared_ptr copy, MTL retain or ownership transfer occurs here.
int32_t anemlx_kv_allocation_snapshot(mlx_array input, uint64_t* scalars, size_t count);
#ifdef __cplusplus
}
#endif
