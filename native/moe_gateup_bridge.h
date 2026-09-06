// Copyright © 2026 ANERunner contributors.
// MLX array ownership follows Apple's mlx/c/private/array.h (MIT).
#pragma once

#include <stdint.h>
#include "mlx/c/array.h"
#include "mlx/c/stream.h"

#ifdef __cplusplus
extern "C" {
#endif

int anemlx_moe_gateup_version(void);
const char* anemlx_moe_gateup_last_error(void);
uint64_t anemlx_moe_gateup_dispatch_count(int variant);
uint64_t anemlx_moe_expert_plan_dispatch_count(int bm);
uint64_t anemlx_moe_grouped_down_dispatch_count(int bm);

// GPU-only, lazy output [M,1,640] BF16. Inputs must evaluate row-contiguous.
// indices must be sorted, in [0,512), and have one entry per input row.
// The caller must pin this dylib until every returned lazy array is destroyed.
// Entry errors return nonzero with thread-local last_error; deferred evaluation
// errors use the existing MLX evaluation error channel. No errors call abort.
int anemlx_moe_gateup(
    mlx_array* result,
    mlx_array x,
    mlx_array gate_w,
    mlx_array gate_scale,
    mlx_array gate_bias,
    mlx_array up_w,
    mlx_array up_scale,
    mlx_array up_bias,
    mlx_array indices,
    mlx_array sigmoid_lut,
    int variant);

// ABI version 2 additions. Only GPU streams are accepted. All results are
// lazy and all evaluated inputs must be row-contiguous. The same plan must be
// used with the same sorted indices, row count and BM that constructed it.
// Plan shape is [ceil(M/BM)+512,4] int32, including row 0 as the header:
// row0.x = valid descriptor count; row1+t = {expert,start,count,0}.
int anemlx_moe_expert_plan(
    mlx_array* result, mlx_array indices, int bm, mlx_stream stream);

// variant 2 = BM32/BN32/WM2/WN1; variant 3 = BM16/BN32/WM1/WN1.
// The old indices argument is validated but kernels read the shared plan.
int anemlx_moe_gateup_planned(
    mlx_array* result,
    mlx_array x,
    mlx_array gate_w,
    mlx_array gate_scale,
    mlx_array gate_bias,
    mlx_array up_w,
    mlx_array up_scale,
    mlx_array up_bias,
    mlx_array indices,
    mlx_array sigmoid_lut,
    mlx_array plan,
    int variant,
    mlx_stream stream);

// Original Q4 down: BF16 activation [M,1,640] -> BF16 [M,1,2560].
int anemlx_moe_grouped_down(
    mlx_array* result,
    mlx_array activation,
    mlx_array down_w,
    mlx_array down_scale,
    mlx_array down_bias,
    mlx_array plan,
    int bm,
    mlx_stream stream);

#ifdef __cplusplus
}
#endif
