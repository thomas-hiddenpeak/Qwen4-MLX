// Adapted from Apple's affine_gather_qmm_rhs_nax in quantized_nax.h,
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

// Native caller contract:
// * x is contiguous BF16 [M, 1, K], sorted_ids is contiguous U32 [M], M > 1.
// * sorted_ids is nondecreasing, all IDs are in [0, 512).
// * Each original affine-Q4 bank is [512, N, K/8] U32; its scales/biases are
//   [512, N, K/64] BF16. Here N=640 and K=2560; no bank is concatenated.
// * sigmoid_lut contains all 65,536 public MLX BF16 sigmoid values, indexed by
//   the exact BF16 gate bits, including sign/exponent bits.
// * Function constants 200/201/202 are the original align_M/N/K constants.
// * Dispatch threadgroups (ceil(N/BN), ceil(M/BM), 1), threads (32, WN, WM).
//
// This preserves the stock global BM tiles and their expert-segment loop.
// Two independent FP32 accumulator tiles follow the same increasing K order.
// Only A is shared between the gate/up MMAs. BF16 dequantization still uses
// the original QuantizedBlockLoader and original scale/bias arrays.
template <typename T, int BM, int BN, int BK, int WM, int WN>
[[kernel]] void anemlx_moe_gateup_fused(
    const device T* x [[buffer(0)]],
    const device uint32_t* gate_w [[buffer(1)]],
    const device T* gate_scales [[buffer(2)]],
    const device T* gate_biases [[buffer(3)]],
    const device uint32_t* up_w [[buffer(4)]],
    const device T* up_scales [[buffer(5)]],
    const device T* up_biases [[buffer(6)]],
    const device uint32_t* sorted_ids [[buffer(7)]],
    const device T* sigmoid_lut [[buffer(8)]],
    device T* activation [[buffer(9)]],
    const constant int& M [[buffer(10)]],
    const constant int& N [[buffer(11)]],
    const constant int& K [[buffer(12)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  constexpr int group_size = 64;
  constexpr int bits = 4;
  constexpr int pack_factor = get_pack_factor<bits, 8>();
  constexpr int bytes_per_pack = get_bytes_per_pack<bits>();
  constexpr int BK_padded = BK + 16 / sizeof(T);
  constexpr short SM = BM / WM;
  constexpr short SN = BN / WN;
  constexpr short SK = 32;
  constexpr short TM = SM / 16;
  constexpr short TN = SN / 16;
  constexpr short TK = SK / 16;
  static_assert(BM % WM == 0 && BN % WN == 0);
  static_assert(SM % 16 == 0 && SN % 16 == 0 && TM > 0 && TN > 0);
  static_assert((TN == 1 && TM % 2 == 0) || TN % 2 == 0,
                "Pinned NAX MMA must have a supported paired fragment");
  static_assert(BK % SK == 0 && BK <= group_size && group_size % BK == 0);

  using loader_t = QuantizedBlockLoader<
      T, BN, BK, BK_padded, true, WM * WN * SIMD_SIZE, group_size, bits>;
  using accumulator_t = NAXTile<float, TM, TN>;
  threadgroup T gate_ws[BN * BK_padded];
  threadgroup T up_ws[BN * BK_padded];

  const int K_w = K * bytes_per_pack / pack_factor;
  const int K_g = K / group_size;
  const int K_it = K / BK;
  const size_t stride_w = size_t(N) * size_t(K_w);
  const size_t stride_s = size_t(N) * size_t(K_g);
  const int y_row = tid.y * BM;
  const int y_col = tid.x * BN;
  const short tgp_bm = align_M ? BM : short(min(BM, M - y_row));
  const short tgp_bn = align_N ? BN : short(min(BN, N - y_col));
  const short tm = SM * (simd_group_id / WN);
  const short tn = SN * (simd_group_id % WN);
  const short sgp_sm =
      align_M ? SM : min(SM, short(max(0, M - (y_row + tm))));
  const short sgp_sn =
      align_N ? SN : min(SN, short(max(0, N - (y_col + tn))));
  const bool is_unaligned_sm = align_M ? false : (sgp_sm != SM);
  const bool is_unaligned_bn = align_N ? false : (tgp_bn != BN);
  const int k_remain = K - K_it * BK;
  const short2 tile_w = short2(k_remain, tgp_bn);

  x += size_t(y_row) * size_t(K);
  activation += size_t(y_row) * size_t(N) + size_t(y_col);
  const device uint8_t* gate_bytes =
      reinterpret_cast<const device uint8_t*>(gate_w) + size_t(y_col) * size_t(K_w);
  const device uint8_t* up_bytes =
      reinterpret_cast<const device uint8_t*>(up_w) + size_t(y_col) * size_t(K_w);
  gate_scales += size_t(y_col) * size_t(K_g);
  gate_biases += size_t(y_col) * size_t(K_g);
  up_scales += size_t(y_col) * size_t(K_g);
  up_biases += size_t(y_col) * size_t(K_g);

  uint32_t index;
  short offset;
  uint32_t index_next = sorted_ids[y_row];
  short offset_next = 0;
  int n = 0;
  while (n < tgp_bm) {
    n++;
    offset = offset_next;
    index = index_next;
    offset_next = tgp_bm;
    for (; n < tgp_bm; n++) {
      if (sorted_ids[y_row + n] != index) {
        offset_next = n;
        index_next = sorted_ids[y_row + n];
        break;
      }
    }
    threadgroup_barrier(mem_flags::mem_none);
    // All threads see the same segment ID. Invalid caller-supplied IDs must
    // never form a bank pointer; mark that segment nonfinite rather than
    // returning uninitialized output or silently substituting another expert.
    if (index >= 512) {
      const int lane = int(simd_group_id) * 32 + int(simd_lane_id);
      const int count = (offset_next - offset) * tgp_bn;
      for (int i = lane; i < count; i += WM * WN * 32) {
        const size_t row = size_t(offset + i / tgp_bn);
        const size_t col = size_t(i % tgp_bn);
        activation[row * size_t(N) + col] = T(as_type<float>(uint(0x7fc00000)));
      }
      continue;
    }
    const short m_lo_lim = min(int(sgp_sm), max(0, offset - tm));
    const short m_hi_lim = min(int(sgp_sm), max(0, offset_next - tm));
    const bool sg_active = m_hi_lim > m_lo_lim;
    accumulator_t gate_acc, up_acc;
    gate_acc.clear();
    up_acc.clear();
    const device T* xn = x + tm * K;

    loader_t gate_loader(
        gate_bytes + size_t(index) * stride_w,
        gate_scales + size_t(index) * stride_s,
        gate_biases + size_t(index) * stride_s,
        K, gate_ws, simd_group_id, simd_lane_id);
    loader_t up_loader(
        up_bytes + size_t(index) * stride_w,
        up_scales + size_t(index) * stride_s,
        up_biases + size_t(index) * stride_s,
        K, up_ws, simd_group_id, simd_lane_id);

    dispatch_bool(align_M || !is_unaligned_sm, [&](auto kAlignedM) {
      dispatch_bool(align_N || !is_unaligned_bn, [&](auto kAlignedN) {
        for (int k = 0; k < K_it; k++) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if constexpr (kAlignedN.value) {
            gate_loader.load_unsafe();
            up_loader.load_unsafe();
          } else {
            gate_loader.load_safe(short2(BK, tgp_bn));
            up_loader.load_safe(short2(BK, tgp_bn));
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          STEEL_PRAGMA_NO_UNROLL
          for (int kk1 = 0; kk1 < BK; kk1 += SK) {
            if (sg_active) {
              NAXTile<T, TM, TK> a_tile;
              NAXTile<T, TN, TK> gate_b, up_b;
              volatile int compiler_barrier;
              if constexpr (kAlignedM.value) {
                a_tile.load(xn + kk1, K);
              } else {
                a_tile.load_safe(xn + kk1, K, short2(SK, sgp_sm));
              }
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

        // K=2560 is aligned for both instantiated geometries. Retain the
        // upstream remainder path; the native API restricts K to this model.
        if (!align_K) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          gate_loader.load_safe(tile_w);
          up_loader.load_safe(tile_w);
          threadgroup_barrier(mem_flags::mem_threadgroup);
          STEEL_PRAGMA_NO_UNROLL
          for (int kk1 = 0; kk1 < BK; kk1 += SK) {
            if (sg_active) {
              NAXTile<T, TM, TK> a_tile;
              NAXTile<T, TN, TK> gate_b, up_b;
              volatile int compiler_barrier;
              const short psk = min(int(SK), max(0, BK - kk1));
              a_tile.load_safe(xn + kk1, K, short2(psk, sgp_sm));
              gate_b.template load<T, BK_padded, 1>(gate_ws + tn * BK_padded + kk1);
              up_b.template load<T, BK_padded, 1>(up_ws + tn * BK_padded + kk1);
              tile_matmad_nax(gate_acc, a_tile, metal::bool_constant<false>{},
                              gate_b, metal::bool_constant<true>{});
              tile_matmad_nax(up_acc, a_tile, metal::bool_constant<false>{},
                              up_b, metal::bool_constant<true>{});
              (void)compiler_barrier;
            }
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sg_active) {
          // The original two QMMs store BF16 projections before SwiGLU.
          // Preserve those two rounds, the BF16 sigmoid lookup, and both
          // elementwise BF16 products. Do not algebraically reassociate them.
          thread float* g = gate_acc.elems();
          const thread float* u = up_acc.elems();
          STEEL_PRAGMA_UNROLL
          for (short i = 0; i < accumulator_t::kElemsPerTile; ++i) {
            const T gate_value = T(g[i]);
            const T up_value = T(u[i]);
            const T sigmoid = sigmoid_lut[as_type<ushort>(gate_value)];
            const T activated = T(float(gate_value) * float(sigmoid));
            const T result = T(float(activated) * float(up_value));
            // Widen only after all BF16 rounds; the final store is idempotent.
            g[i] = float(result);
          }
          if constexpr (kAlignedN.value) {
            if (m_lo_lim == 0 && m_hi_lim == SM) {
              gate_acc.store(activation + tm * N + tn, N);
            } else {
              gate_acc.store_slice(activation + tm * N + tn, N,
                                   short2(0, m_lo_lim), short2(SN, m_hi_lim));
            }
          } else {
            gate_acc.store_slice(activation + tm * N + tn, N,
                                 short2(0, m_lo_lim), short2(sgp_sn, m_hi_lim));
          }
        }
      });
    });
  }
}

instantiate_kernel(
    "anemlx_moe_gateup_fused_bf16_q4_g64_bm32_bn64_bk64_wm2_wn2",
    anemlx_moe_gateup_fused, bfloat16_t, 32, 64, 64, 2, 2);
instantiate_kernel(
    "anemlx_moe_gateup_fused_bf16_q4_g64_bm32_bn32_bk64_wm2_wn1",
    anemlx_moe_gateup_fused, bfloat16_t, 32, 32, 64, 2, 1);
