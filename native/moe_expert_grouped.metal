// Expert-boundary tile planning and affine-Q4 NAX projections for ANERunner.
// Adapted from Apple's QuantizedBlockLoader / affine_gather_qmm_rhs_nax,
// MLX commit 1f8e74e3f12f31365464a6867c6579f0e9b29d85.
// Copyright © 2023-2024 Apple Inc.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "mlx/backend/metal/kernels/steel/gemm/loader.h"
#include "mlx/backend/metal/kernels/quantized_nax.h"

// Caller contracts shared by all kernels:
// * M>1, every array is row-contiguous, IDs are sorted U32 and in [0,512).
// * The plan must belong to these exact sorted IDs, M, and BM. It cannot be
//   reused across layers/routes. Gate/up and down may share it for one route.
// * Gate/up use K=2560,N=640; down uses K=640,N=2560. Both use the original
//   U32 affine-Q4 banks [512,N,K/8] and BF16 scale/bias [512,N,K/64].
// * These fixed shapes have N divisible by BN32 and K divisible by BK64.
// * There are no function constants. M masking is local to each descriptor.

METAL_FUNC int anemlx_expert_lower_bound(
    const device uint32_t* ids, int count, uint32_t value) {
  int lo = 0, hi = count;
  while (lo < hi) {
    const int mid = lo + (hi - lo) / 2;
    if (ids[mid] < value) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

// Exactly one threadgroup of 512 threads. No host readback is required.
// plan is I32[ceil(M/BM)+512,4]. Row 0={totalValidTiles,0,0,0}; row 1+t is
// {expertID, originalSortedRowStart, validRows,0}. Unused rows are zeroed.
// The descriptor bound sum(ceil(count_e/BM)) <= ceil(M/BM)+511 holds for
// at most 512 nonempty experts. Rows and expert assignments are never dropped.
template <int BM>
[[kernel]] void anemlx_moe_expert_plan(
    const device uint32_t* sorted_ids [[buffer(0)]],
    device int4* plan [[buffer(1)]],
    const constant int& M [[buffer(2)]],
    uint expert [[thread_index_in_threadgroup]]) {
  static_assert(BM == 16 || BM == 32);
  threadgroup int cumulative_tiles[512];
  const int capacity = (M + BM - 1) / BM + 511;
  for (int row = int(expert); row <= capacity; row += 512) {
    plan[row] = int4(0);
  }
  const int begin = anemlx_expert_lower_bound(sorted_ids, M, expert);
  const int end = anemlx_expert_lower_bound(sorted_ids, M, expert + 1);
  const int count = end - begin;
  const int own_tiles = (count + BM - 1) / BM;
  cumulative_tiles[expert] = own_tiles;
  // Every descriptor's initialization must finish before any lane fills it.
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  for (int distance = 1; distance < 512; distance *= 2) {
    // All lanes read the old scan generation before any lane overwrites it.
    const int previous = expert >= uint(distance)
        ? cumulative_tiles[expert - uint(distance)] : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    cumulative_tiles[expert] += previous;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const int first_tile = cumulative_tiles[expert] - own_tiles;
  for (int tile = 0; tile < own_tiles; ++tile) {
    const int row = begin + tile * BM;
    plan[1 + first_tile + tile] = int4(int(expert), row, min(BM, end - row), 0);
  }
  if (expert == 0) {
    plan[0] = int4(cumulative_tiles[511], 0, 0, 0);
  }
}

// BF16 gate/up+SwiGLU. Buffer ABI matches moe_gateup_fused.metal except buffer
// 7 is the plan instead of per-row IDs. Use grid (N/32,capacity,1) threadgroups,
// each with (32,1,WM) threads. Fixed-capacity excess groups return uniformly.
template <typename T, int BM, int BN, int BK, int WM, int WN>
[[kernel]] void anemlx_moe_gateup_grouped(
    const device T* x [[buffer(0)]],
    const device uint32_t* gate_w [[buffer(1)]],
    const device T* gate_scales [[buffer(2)]],
    const device T* gate_biases [[buffer(3)]],
    const device uint32_t* up_w [[buffer(4)]],
    const device T* up_scales [[buffer(5)]],
    const device T* up_biases [[buffer(6)]],
    const device int4* plan [[buffer(7)]],
    const device T* sigmoid_lut [[buffer(8)]],
    device T* activation [[buffer(9)]],
    const constant int& M [[buffer(10)]],
    const constant int& N [[buffer(11)]],
    const constant int& K [[buffer(12)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  constexpr short SM = BM / WM, SN = BN / WN, SK = 32;
  constexpr short TM = SM / 16, TN = SN / 16, TK = SK / 16;
  constexpr int BK_padded = BK + 16 / sizeof(T);
  static_assert((BM == 16 || BM == 32) && BM % WM == 0 && BN % WN == 0);
  static_assert(SM == 16 && SN == 32 && TM == 1 && TN == 2 && TK == 2,
                "Both variants retain the supported 16x32 NAX MMA fragment");
  static_assert(BN == 32 && BK == 64 && WN == 1);
  using loader_t = QuantizedBlockLoader<T, BN, BK, BK_padded, true,
                                         WM * WN * SIMD_SIZE, 64, 4>;
  using accumulator_t = NAXTile<float, TM, TN>;
  threadgroup T gate_ws[BN * BK_padded];
  threadgroup T up_ws[BN * BK_padded];

  const int total_tiles = plan[0].x;
  if (total_tiles <= 0 || tid.y >= uint(total_tiles)) return;
  const int4 descriptor = plan[1 + tid.y];
  const int expert = descriptor.x, start = descriptor.y, count = descriptor.z;
  // Uniform pointer-safety guards for malformed externally supplied plans.
  // The native caller still owns the sorted-ID / plan identity contract.
  if (expert < 0 || expert >= 512 || start < 0 || count <= 0 ||
      count > BM || count > M || start > M - count) return;
  const int y_col = int(tid.x) * BN;
  if (y_col >= N) return;
  const short tm = SM * (simd_group_id / WN);
  const short tn = SN * (simd_group_id % WN);
  const short valid_rows = short(min(int(SM), max(0, count - int(tm))));
  const bool active = valid_rows > 0;
  const int K_w = K / 2, K_g = K / 64;
  const size_t weight_offset = (size_t(expert) * size_t(N) + size_t(y_col)) * size_t(K_w);
  const size_t scale_offset = (size_t(expert) * size_t(N) + size_t(y_col)) * size_t(K_g);
  const device uint8_t* gate_bytes =
      reinterpret_cast<const device uint8_t*>(gate_w) + weight_offset;
  const device uint8_t* up_bytes =
      reinterpret_cast<const device uint8_t*>(up_w) + weight_offset;
  const size_t row_base = size_t(start) + size_t(active ? tm : 0);
  const device T* xn = x + row_base * size_t(K);
  device T* yn = activation + row_base * size_t(N) + size_t(y_col + tn);
  loader_t gate_loader(gate_bytes, gate_scales + scale_offset, gate_biases + scale_offset,
                       K, gate_ws, simd_group_id, simd_lane_id);
  loader_t up_loader(up_bytes, up_scales + scale_offset, up_biases + scale_offset,
                     K, up_ws, simd_group_id, simd_lane_id);
  accumulator_t gate_acc, up_acc;
  gate_acc.clear();
  up_acc.clear();

  // Same BK64 -> SK32 -> TK2 increasing-K order as the original NAX QMM.
  for (int k = 0; k < K / BK; ++k) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    gate_loader.load_unsafe();
    up_loader.load_unsafe();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    STEEL_PRAGMA_NO_UNROLL
    for (int kk1 = 0; kk1 < BK; kk1 += SK) {
      if (active) {
        NAXTile<T, TM, TK> a_tile;
        NAXTile<T, TN, TK> gate_b, up_b;
        volatile int compiler_barrier;
        if (valid_rows == SM) a_tile.load(xn + kk1, K);
        else a_tile.load_safe(xn + kk1, K, short2(SK, valid_rows));
        gate_b.template load<T, BK_padded, 1>(gate_ws + tn * BK_padded + kk1);
        up_b.template load<T, BK_padded, 1>(up_ws + tn * BK_padded + kk1);
        tile_matmad_nax(gate_acc, a_tile, metal::bool_constant<false>{},
                        gate_b, metal::bool_constant<true>{});
        tile_matmad_nax(up_acc, a_tile, metal::bool_constant<false>{},
                        up_b, metal::bool_constant<true>{});
        (void)compiler_barrier;
      }
    }
    xn += BK;
    gate_loader.next();
    up_loader.next();
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (active) {
    thread float* g = gate_acc.elems();
    const thread float* u = up_acc.elems();
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < accumulator_t::kElemsPerTile; ++i) {
      const T gate_value = T(g[i]);
      const T up_value = T(u[i]);
      const T sigmoid = sigmoid_lut[as_type<ushort>(gate_value)];
      const T activated = T(float(gate_value) * float(sigmoid));
      const T result = T(float(activated) * float(up_value));
      g[i] = float(result);
    }
    if (valid_rows == SM) gate_acc.store(yn, N);
    else gate_acc.store_slice(yn, N, short2(0, 0), short2(SN, valid_rows));
  }
}

// Standalone down projection. Input and output both retain original sorted
// row positions, so the exact same plan can be reused after grouped gate/up.
// Buffers match the stock QMM except buffer 4 holds the descriptor plan.
template <typename T, int BM, int BN, int BK, int WM, int WN>
[[kernel]] void anemlx_moe_down_grouped(
    const device T* x [[buffer(0)]],
    const device uint32_t* weight [[buffer(1)]],
    const device T* scales [[buffer(2)]],
    const device T* biases [[buffer(3)]],
    const device int4* plan [[buffer(4)]],
    device T* output [[buffer(5)]],
    const constant int& M [[buffer(6)]],
    const constant int& N [[buffer(7)]],
    const constant int& K [[buffer(8)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  constexpr short SM = BM / WM, SN = BN / WN, SK = 32;
  constexpr short TM = SM / 16, TN = SN / 16, TK = SK / 16;
  constexpr int BK_padded = BK + 16 / sizeof(T);
  static_assert((BM == 16 || BM == 32) && BM % WM == 0 && BN % WN == 0);
  static_assert(SM == 16 && SN == 32 && TM == 1 && TN == 2 && TK == 2,
                "Both variants retain the supported 16x32 NAX MMA fragment");
  static_assert(BN == 32 && BK == 64 && WN == 1);
  using loader_t = QuantizedBlockLoader<T, BN, BK, BK_padded, true,
                                         WM * WN * SIMD_SIZE, 64, 4>;
  using accumulator_t = NAXTile<float, TM, TN>;
  threadgroup T ws[BN * BK_padded];

  const int total_tiles = plan[0].x;
  if (total_tiles <= 0 || tid.y >= uint(total_tiles)) return;
  const int4 descriptor = plan[1 + tid.y];
  const int expert = descriptor.x, start = descriptor.y, count = descriptor.z;
  if (expert < 0 || expert >= 512 || start < 0 || count <= 0 ||
      count > BM || count > M || start > M - count) return;
  const int y_col = int(tid.x) * BN;
  if (y_col >= N) return;
  const short tm = SM * (simd_group_id / WN);
  const short tn = SN * (simd_group_id % WN);
  const short valid_rows = short(min(int(SM), max(0, count - int(tm))));
  const bool active = valid_rows > 0;
  const int K_w = K / 2, K_g = K / 64;
  const size_t weight_offset = (size_t(expert) * size_t(N) + size_t(y_col)) * size_t(K_w);
  const size_t scale_offset = (size_t(expert) * size_t(N) + size_t(y_col)) * size_t(K_g);
  const device uint8_t* bytes =
      reinterpret_cast<const device uint8_t*>(weight) + weight_offset;
  const size_t row_base = size_t(start) + size_t(active ? tm : 0);
  const device T* xn = x + row_base * size_t(K);
  device T* yn = output + row_base * size_t(N) + size_t(y_col + tn);
  loader_t loader(bytes, scales + scale_offset, biases + scale_offset,
                  K, ws, simd_group_id, simd_lane_id);
  accumulator_t accumulator;
  accumulator.clear();
  for (int k = 0; k < K / BK; ++k) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    loader.load_unsafe();
    threadgroup_barrier(mem_flags::mem_threadgroup);
    STEEL_PRAGMA_NO_UNROLL
    for (int kk1 = 0; kk1 < BK; kk1 += SK) {
      if (active) {
        NAXTile<T, TM, TK> a_tile;
        NAXTile<T, TN, TK> b_tile;
        volatile int compiler_barrier;
        if (valid_rows == SM) a_tile.load(xn + kk1, K);
        else a_tile.load_safe(xn + kk1, K, short2(SK, valid_rows));
        b_tile.template load<T, BK_padded, 1>(ws + tn * BK_padded + kk1);
        tile_matmad_nax(accumulator, a_tile, metal::bool_constant<false>{},
                        b_tile, metal::bool_constant<true>{});
        (void)compiler_barrier;
      }
    }
    xn += BK;
    loader.next();
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (active) {
    if (valid_rows == SM) accumulator.store(yn, N);
    else accumulator.store_slice(yn, N, short2(0, 0), short2(SN, valid_rows));
  }
}

instantiate_kernel("anemlx_moe_expert_plan_bm32", anemlx_moe_expert_plan, 32);
instantiate_kernel("anemlx_moe_expert_plan_bm16", anemlx_moe_expert_plan, 16);
instantiate_kernel(
    "anemlx_moe_gateup_grouped_bf16_q4_g64_bm32_bn32_bk64_wm2_wn1",
    anemlx_moe_gateup_grouped, bfloat16_t, 32, 32, 64, 2, 1);
instantiate_kernel(
    "anemlx_moe_gateup_grouped_bf16_q4_g64_bm16_bn32_bk64_wm1_wn1",
    anemlx_moe_gateup_grouped, bfloat16_t, 16, 32, 64, 1, 1);
instantiate_kernel(
    "anemlx_moe_down_grouped_bf16_q4_g64_bm32_bn32_bk64_wm2_wn1",
    anemlx_moe_down_grouped, bfloat16_t, 32, 32, 64, 2, 1);
instantiate_kernel(
    "anemlx_moe_down_grouped_bf16_q4_g64_bm16_bn32_bk64_wm1_wn1",
    anemlx_moe_down_grouped, bfloat16_t, 16, 32, 64, 1, 1);
