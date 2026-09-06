// Adapted from Apple MLX GEMVKernel in mlx/backend/metal/kernels/gemv.h
// and its dispatch in mlx/backend/metal/matmul.cpp, as vendored by
// garnermccloud/mlx-serve 7dbcba04c98e4fd3bcc533c63e645547f13cc3b1.
// This experiment adds independent token accumulators while retaining the
// scalar GEMV lane assignment, accumulation order and final BF16 conversion.
//
// MIT License
// Copyright © 2023 Apple Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

import CMLX
import Foundation

/// Experimental S2...5 dense projection with the pinned scalar GEMV reduction.
///
/// `weight` follows the model's convention: a transpose of row-major [N,K]
/// BF16 source weights, exposed as [K,N]. Transposing it back is a view for
/// those model weights; the public custom-kernel API ensures contiguous inputs
/// at evaluation. No expanded weights, host readback, or model weights are
/// retained here. The caller owns evaluation and serial access to this instance.
/// Bitwise parity is a test gate, not a guarantee about other Metal compilers
/// or separately tuned scalar GEMV implementations.
public final class GPUVerificationLinear {
    private let kernel: mlx_fast_metal_kernel
    private var configurations: [[Int]: Configuration] = [:]

    public init() throws {
        let inputs = mlx_vector_string_new(), outputs = mlx_vector_string_new()
        defer { _ = mlx_vector_string_free(inputs); _ = mlx_vector_string_free(outputs) }
        for name in ["x", "w"] {
            try MX.check(mlx_vector_string_append_value(inputs, name), "Verification linear input")
        }
        try MX.check(mlx_vector_string_append_value(outputs, "y"), "Verification linear output")
        let created = mlx_fast_metal_kernel_new(
            "ane_runner_verification_linear_scalar_order_v1", inputs, outputs,
            Self.source, "", true, false)
        guard created.ctx != nil else { throw GPUError.invalid("Could not create verification linear kernel") }
        kernel = created
    }

    deinit { mlx_fast_metal_kernel_free(kernel) }

    public func apply(_ x: Tensor, weight: Tensor) throws -> Tensor {
        let shape = x.shape, ws = weight.shape
        guard shape.count == 3, shape[0] == 1, (2...5).contains(shape[1]),
              ws.count == 2, shape[2] == ws[0],
              x.dtype == MLX_BFLOAT16, weight.dtype == MLX_BFLOAT16 else {
            throw GPUError.invalid("Verification linear requires BF16 [1,S,K], S2...5, and transposed [K,N] weights")
        }
        let key = [shape[1], ws[0], ws[1]]
        let configuration: Configuration
        if let cached = configurations[key] { configuration = cached }
        else {
            let parameters = try Self.parameters(inputSize: ws[0], outputSize: ws[1])
            configuration = try Configuration(tokens: shape[1], inputSize: ws[0],
                                              outputSize: ws[1], parameters: parameters)
            configurations[key] = configuration
        }
        let rows = try MX.transpose(weight, [1, 0])
        let inputs = mlx_vector_array_new_data([x.handle, rows.handle], 2)
        var outputs = mlx_vector_array_new()
        defer { _ = mlx_vector_array_free(inputs); _ = mlx_vector_array_free(outputs) }
        try MX.check(mlx_fast_metal_kernel_apply(&outputs, kernel, inputs, configuration.handle, MX.stream),
                     "Verification linear")
        guard mlx_vector_array_size(outputs) == 1 else {
            throw GPUError.invalid("Verification linear output count mismatch")
        }
        return try MX.output("Verification linear result") { mlx_vector_array_get(&$0, outputs, 0) }
    }

    /// Supported source projections only. N=1 is deliberately excluded: the
    /// scalar reference uses dot_product, whose reduction differs from GEMV.
    /// Small-K, row-major [K,N], quantized and arbitrary-shape paths are also
    /// outside this experiment. K and N below are in elements, not bytes.
    private static let sourceShapes: Set<[Int]> = [
            [10240, 320], [320, 10240], [10240, 4], // HC
            [2560, 10240], [2560, 6144], [2560, 48], [6144, 2560], // GDN / PLE
            [2560, 12288], [2560, 512], [2560, 640], // attention / router
            [2560, 2560], [640, 2560], // PLE / shared expert
            [2560, 248320], // main BF16 vocabulary head
    ]

    static func parameters(inputSize k: Int, outputSize n: Int) throws -> Parameters {
        guard sourceShapes.contains([k, n]) else {
            throw GPUError.invalid("Unsupported verification linear source shape [K,N]=[\(k),\(n)]")
        }
        // matmul.cpp::gemv_axbpy, non-transposed matrix arm. All source shapes
        // have K>64 and N>=4, hence SM1/SN32/TM4/TN4. Narrow projections split
        // K among eight SIMD groups; the others tile independent output rows.
        return k >= 16 * n
            ? Parameters(bm: 1, bn: 8, sm: 1, sn: 32, tm: 4, tn: 4)
            : Parameters(bm: n >= 4096 ? 8 : 4, bn: 1, sm: 1, sn: 32, tm: 4, tn: 4)
    }

    struct Parameters: Equatable {
        let bm, bn, sm, sn, tm, tn: Int
        var blockM: Int { bm * sm * tm }
    }

    private final class Configuration {
        let handle: mlx_fast_metal_kernel_config
        init(tokens: Int, inputSize: Int, outputSize: Int, parameters p: Parameters) throws {
            let config = mlx_fast_metal_kernel_config_new()
            let shape: [Int32] = [1, Int32(tokens), Int32(outputSize)]
            let groups = (outputSize + p.blockM - 1) / p.blockM
            do {
                try MX.check(mlx_fast_metal_kernel_config_add_output_arg(config, shape, shape.count, MLX_BFLOAT16),
                             "Verification linear output shape")
                try MX.check(mlx_fast_metal_kernel_config_set_grid(config, Int32(groups * 32), Int32(p.bn), Int32(p.bm)),
                             "Verification linear grid")
                try MX.check(mlx_fast_metal_kernel_config_set_thread_group(config, 32, Int32(p.bn), Int32(p.bm)),
                             "Verification linear threadgroup")
                for (name, value) in [("S", tokens), ("K", inputSize), ("N", outputSize),
                                      ("BM", p.bm), ("BN", p.bn), ("SM", p.sm),
                                      ("SN", p.sn), ("TM", p.tm), ("TN", p.tn)] {
                    try MX.check(mlx_fast_metal_kernel_config_add_template_arg_int(config, name, Int32(value)),
                                 "Verification linear template")
                }
            } catch {
                mlx_fast_metal_kernel_config_free(config)
                throw error
            }
            handle = config
        }
        deinit { mlx_fast_metal_kernel_config_free(handle) }
    }

    private static let source = """
    constexpr int BLOCK_M = BM * SM * TM;
    constexpr int BLOCK_N = BN * SN * TN;
    constexpr int PARTIAL_STRIDE = BLOCK_M + TM;
    threadgroup float partials[BN > 1 ? S * BN * PARTIAL_STRIDE : 1];
    const int lane = int(thread_index_in_simdgroup);
    const int simd = int(simdgroup_index_in_threadgroup);
    const int thr_m = SN != 32 ? lane / SN : 0;
    const int thr_n = SN != 32 ? lane % SN : lane;
    const int sg_n = BN != 1 ? simd % BN : 0;
    const int simd_m = BN != 1 ? SM * (simd / BN) : SM * simd;
    const int simd_n = BN != 1 ? SN * (simd % BN) : 0;
    const int bm = (simd_m + thr_m) * TM;
    int bn = (simd_n + thr_n) * TN;
    int out_row = int(threadgroup_position_in_grid.x) * BLOCK_M + bm;
    if (out_row >= N) return;
    out_row = out_row + TM <= N ? out_row : N - TM;

    float result[S][TM] = {};
    float coeff[S][TN];
    bfloat16_t inter[TN];
    const int iterations = K / BLOCK_N;
    for (int iteration = 0; iteration < iterations; ++iteration) {
      #pragma clang loop unroll(full)
      for (int token = 0; token < S; ++token) {
        #pragma clang loop unroll(full)
        for (int tn = 0; tn < TN; ++tn) coeff[token][tn] = float(x[token * K + bn + tn]);
      }
      #pragma clang loop unroll(full)
      for (int tm = 0; tm < TM; ++tm) {
        #pragma clang loop unroll(full)
        for (int tn = 0; tn < TN; ++tn) inter[tn] = w[(out_row + tm) * K + bn + tn];
        // One weight load feeds every token. Accumulate each component into
        // the persistent scalar result: no dot(float4) or chunk subtotal.
        #pragma clang loop unroll(full)
        for (int token = 0; token < S; ++token) {
          #pragma clang loop unroll(full)
          for (int tn = 0; tn < TN; ++tn) result[token][tm] += inter[tn] * coeff[token][tn];
        }
      }
      bn += BLOCK_N;
    }
    if constexpr (K % BLOCK_N != 0) {
      #pragma clang loop unroll(full)
      for (int token = 0; token < S; ++token) {
        #pragma clang loop unroll(full)
        for (int tn = 0; tn < TN; ++tn) coeff[token][tn] = bn + tn < K ? float(x[token * K + bn + tn]) : 0.0f;
      }
      #pragma clang loop unroll(full)
      for (int tm = 0; tm < TM; ++tm) {
        #pragma clang loop unroll(full)
        for (int tn = 0; tn < TN; ++tn) inter[tn] = bn + tn < K ? w[(out_row + tm) * K + bn + tn] : bfloat16_t(0);
        #pragma clang loop unroll(full)
        for (int token = 0; token < S; ++token) {
          #pragma clang loop unroll(full)
          for (int tn = 0; tn < TN; ++tn) result[token][tm] += inter[tn] * coeff[token][tn];
        }
      }
    }
    #pragma clang loop unroll(full)
    for (int token = 0; token < S; ++token) {
      #pragma clang loop unroll(full)
      for (int tm = 0; tm < TM; ++tm) {
        #pragma clang loop unroll(full)
        for (ushort sn = SN / 2; sn >= 1; sn >>= 1) result[token][tm] += simd_shuffle_down(result[token][tm], sn);
      }
    }
    if constexpr (BN > 1) {
      if (thr_n == 0) {
        #pragma clang loop unroll(full)
        for (int token = 0; token < S; ++token) {
          #pragma clang loop unroll(full)
          for (int tm = 0; tm < TM; ++tm) partials[(token * BN + sg_n) * PARTIAL_STRIDE + bm + tm] = result[token][tm];
        }
      }
      // Every thread reaches this barrier. For supported BN8 shapes BM=1 and
      // N is divisible by TM, so no SIMD group can return ahead of it.
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (sg_n == 0 && thr_n == 0) {
        #pragma clang loop unroll(full)
        for (int sgn = 1; sgn < BN; ++sgn) {
          #pragma clang loop unroll(full)
          for (int token = 0; token < S; ++token) {
            #pragma clang loop unroll(full)
            for (int tm = 0; tm < TM; ++tm) result[token][tm] += partials[(token * BN + sgn) * PARTIAL_STRIDE + bm + tm];
          }
        }
      }
    }
    if (simd_n == 0 && thr_n == 0) {
      #pragma clang loop unroll(full)
      for (int token = 0; token < S; ++token) {
        #pragma clang loop unroll(full)
        for (int tm = 0; tm < TM; ++tm) y[token * N + out_row + tm] = bfloat16_t(result[token][tm]);
      }
    }
    """
}
