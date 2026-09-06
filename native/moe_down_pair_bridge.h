// Copyright © 2026 ANERunner contributors.
// MLX array ownership follows Apple's mlx/c/private headers (MIT).
#pragma once
#include <stdint.h>
#include "mlx/c/array.h"
#include "mlx/c/stream.h"
#include "mlx/c/vector.h"

#ifdef __cplusplus
extern "C" {
#endif
int anemlx_moe_down_pair_version(void);
const char* anemlx_moe_down_pair_last_error(void);
// Number of successfully encoded pairs, not GPU completion or kernel launches.
uint64_t anemlx_moe_down_pair_count(int serial_control);

// Internal experiment only. The caller must supply the original unmodified
// GPUMoEFused downReduce CustomKernel recipe (not its reshape/diagnostic output)
// and MLX's 2D BF16 Matmul recipe. Both must be unscheduled single-output GPU
// nodes on stream. The scalar K/N contents must be 640/2560; IDs must come from
// the model router in [0,512). These contents are not read back by the bridge.
// Returns two lazy siblings with the recipe shapes/dtypes, without evaluating
// or mutating either recipe. The caller must pin this dylib for their lifetime.
// serial_control=1 inserts a GPU barrier between the original child eval_gpu
// calls; 0 predeclares all input hazards and leaves independent dispatches free
// to overlap. No nested eval, synchronization, queue or command-buffer commit.
int anemlx_moe_down_pair(
    mlx_vector_array* results, mlx_array routed_recipe, mlx_array shared_recipe,
    int serial_control, mlx_stream stream);
#ifdef __cplusplus
}
#endif
