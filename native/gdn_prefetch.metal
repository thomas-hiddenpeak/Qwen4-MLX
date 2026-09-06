// Experimental QKV GEMV, derived from MLX GEMVKernel's thread mapping and
// accumulation loop. Original algorithm: Copyright © 2023-2024 Apple Inc.
// See UPSTREAM-NOTICE and LICENSE-APACHE-2.0 for the MLX attribution/license.
#include <metal_simdgroup>
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

// The dispatcher requires unbatched, contiguous BF16 QKV, K=2560, N=10240,
// matrix_ld=2560, with no axpby. BM8/BN1/SM1/SN32/TM4/TN4 is unchanged.
// Source-level lookahead only: the compiler decides final load scheduling.
template <bool VectorLoads>
[[kernel, max_total_threads_per_threadgroup(256)]] void gdn_prefetch4(
    const device bfloat16_t* mat [[buffer(0)]],
    const device bfloat16_t* in_vec [[buffer(1)]],
    const device bfloat16_t* bias [[buffer(2)]],
    device bfloat16_t* out_vec [[buffer(3)]],
    const constant int& in_vec_size [[buffer(4)]],
    const constant int& out_vec_size [[buffer(5)]],
    const constant int& matrix_ld [[buffer(6)]],
    const constant float& alpha [[buffer(7)]],
    const constant float& beta [[buffer(8)]],
    const constant int& batch_ndim [[buffer(9)]],
    const constant int* batch_shape [[buffer(10)]],
    const constant int64_t* vector_batch_stride [[buffer(11)]],
    const constant int64_t* matrix_batch_stride [[buffer(12)]],
    const constant int64_t* bias_batch_stride [[buffer(13)]],
    const constant int& bias_stride [[buffer(14)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  const int out_row = int(tid.x) * 32 + int(simd_gid) * 4;
  const int bn = int(simd_lid) * 4;
  mat += size_t(out_row) * matrix_ld;

  float result[4] = {0};
  // Four consecutive 128-wide K tiles. Individual threads retain the same
  // four columns per tile and the same running sum for each output row.
  for (int base = 0; base < 2560; base += 4 * 128) {
    float prefetched_vec[4][4];
    bfloat16_t prefetched_mat[4][4][4];
    MLX_MTL_PRAGMA_UNROLL
    for (int tile = 0; tile < 4; ++tile) {
      const int column = base + tile * 128 + bn;
      if constexpr (VectorLoads) {
        const auto values = *reinterpret_cast<const device vec<bfloat16_t, 4>*>(
            in_vec + column);
        MLX_MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < 4; ++tn) {
          prefetched_vec[tile][tn] = float(values[tn]);
        }
      } else {
        MLX_MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < 4; ++tn) {
          prefetched_vec[tile][tn] = float(in_vec[column + tn]);
        }
      }
      MLX_MTL_PRAGMA_UNROLL
      for (int tm = 0; tm < 4; ++tm) {
        const device bfloat16_t* row = mat + tm * matrix_ld + column;
        if constexpr (VectorLoads) {
          const auto values =
              *reinterpret_cast<const device vec<bfloat16_t, 4>*>(row);
          MLX_MTL_PRAGMA_UNROLL
          for (int tn = 0; tn < 4; ++tn) {
            prefetched_mat[tile][tm][tn] = values[tn];
          }
        } else {
          MLX_MTL_PRAGMA_UNROLL
          for (int tn = 0; tn < 4; ++tn) {
            prefetched_mat[tile][tm][tn] = row[tn];
          }
        }
      }
    }

    // Match the original tile -> row -> scalar order. No dot(), split-K,
    // extra partial sums, or vector arithmetic changes the reduction tree.
    MLX_MTL_PRAGMA_UNROLL
    for (int tile = 0; tile < 4; ++tile) {
      MLX_MTL_PRAGMA_UNROLL
      for (int tm = 0; tm < 4; ++tm) {
        MLX_MTL_PRAGMA_UNROLL
        for (int tn = 0; tn < 4; ++tn) {
          result[tm] += prefetched_mat[tile][tm][tn] * prefetched_vec[tile][tn];
        }
      }
    }
  }

  MLX_MTL_PRAGMA_UNROLL
  for (int tm = 0; tm < 4; ++tm) {
    MLX_MTL_PRAGMA_UNROLL
    for (ushort sn = 16; sn >= 1; sn >>= 1) {
      result[tm] += simd_shuffle_down(result[tm], sn);
    }
  }
  if (simd_lid == 0) {
    MLX_MTL_PRAGMA_UNROLL
    for (int tm = 0; tm < 4; ++tm) {
      out_vec[out_row + tm] = bfloat16_t(result[tm]);
    }
  }
}

instantiate_kernel(
    "gdn_prefetch4_bfloat16_bm8_bn1_sm1_sn32_tm4_tn4_nc0_axpby0",
    gdn_prefetch4, false)
instantiate_kernel(
    "gdn_prefetch4_vector_bfloat16_bm8_bn1_sm1_sn32_tm4_tn4_nc0_axpby0",
    gdn_prefetch4, true)
