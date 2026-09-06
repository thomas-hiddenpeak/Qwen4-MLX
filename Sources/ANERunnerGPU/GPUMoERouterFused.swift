// Softmax router is adapted from mlx-serve/src/transformer.zig
// (moeRouterSource(.softmax), moeRouterTopK, moeRouterThreadGroup),
// upstream garnermccloud/mlx-serve commit 7dbcba04c98e4fd3bcc533c63e645547f13cc3b1.
// The public Metal kernel preserves the upstream reduction tree, lowest-index
// tie-break, and BF16 probability and denominator rounding boundaries.
//
// Upstream attribution and permission:
// MIT License
// Copyright (c) 2026 David Dalcu
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

import CMLX
import Foundation

/// One GPU dispatch for the upstream Qwen softmax/top-k routing operation.
///
/// This accepts BF16 logits with one to four dimensions, preserving every
/// leading dimension. The caller owns evaluation. No tensor is copied to the
/// host. As with the model itself, call each instance from one execution lane.
/// Finite logits are expected, as in the upstream model's router.
public final class GPUMoERouterFused {
    public struct Routing {
        public let indices: Tensor
        public let weights: Tensor
    }

    public let expertCount: Int
    public let topK: Int
    private let kernel: mlx_fast_metal_kernel
    // Full shapes are required: [1,2,E] and [2,1,E] share a row count but
    // must receive differently shaped outputs.
    private var configurations: [[Int]: Configuration] = [:]

    public init(expertCount: Int = 512, topK: Int = 10) throws {
        guard (1...2048).contains(expertCount), (1...32).contains(topK), topK <= expertCount else {
            throw GPUError.invalid("Fused router requires 1...2048 experts and 1...min(32, experts) selected experts")
        }
        let namesIn = mlx_vector_string_new(), namesOut = mlx_vector_string_new()
        defer { _ = mlx_vector_string_free(namesIn); _ = mlx_vector_string_free(namesOut) }
        try MX.check(mlx_vector_string_append_value(namesIn, "logits"), "Router input name")
        for name in ["inds", "scores"] {
            try MX.check(mlx_vector_string_append_value(namesOut, name), "Router output name")
        }
        let created = mlx_fast_metal_kernel_new(
            "ane_runner_moe_router_softmax", namesIn, namesOut, Self.source, "", true, false)
        guard created.ctx != nil else { throw GPUError.invalid("Failed to create fused router Metal kernel") }
        self.expertCount = expertCount; self.topK = topK; self.kernel = created
    }

    deinit { mlx_fast_metal_kernel_free(kernel) }

    public func route(_ logits: Tensor) throws -> Routing {
        let shape = logits.shape
        guard (1...4).contains(shape.count), shape.last == expertCount,
              shape.allSatisfy({ $0 > 0 && $0 <= Int(Int32.max) }), logits.dtype == MLX_BFLOAT16 else {
            throw GPUError.invalid("Fused router expects nonempty rank 1...4 BF16 logits ending in \(expertCount)")
        }
        var rows = 1
        for dimension in shape.dropLast() {
            let (value, overflow) = rows.multipliedReportingOverflow(by: dimension)
            guard !overflow, value <= Int(Int32.max) else { throw GPUError.invalid("Fused router row count overflow") }
            rows = value
        }
        let configuration: Configuration
        if let cached = configurations[shape] { configuration = cached }
        else {
            configuration = try Configuration(shape: shape, rows: rows, expertCount: expertCount, topK: topK)
            configurations[shape] = configuration
        }
        let inputs = mlx_vector_array_new_data([logits.handle], 1)
        var outputs = mlx_vector_array_new()
        defer { _ = mlx_vector_array_free(inputs); _ = mlx_vector_array_free(outputs) }
        try MX.check(mlx_fast_metal_kernel_apply(&outputs, kernel, inputs, configuration.handle, MX.stream), "Fused MoE router")
        guard mlx_vector_array_size(outputs) == 2 else { throw GPUError.invalid("Fused router output count mismatch") }
        let indices = try MX.output("Router selected indices") { mlx_vector_array_get(&$0, outputs, 0) }
        let weights = try MX.output("Router normalized weights") { mlx_vector_array_get(&$0, outputs, 1) }
        return Routing(indices: indices, weights: weights)
    }

    private final class Configuration {
        let handle: mlx_fast_metal_kernel_config
        init(shape: [Int], rows: Int, expertCount: Int, topK: Int) throws {
            let config = mlx_fast_metal_kernel_config_new()
            let threadGroup = max(32, min(256, ((expertCount + 31) / 32) * 32))
            let outputShape = (Array(shape.dropLast()) + [topK]).map(Int32.init)
            do {
                for dtype in [MLX_UINT32, MLX_BFLOAT16] {
                    try MX.check(mlx_fast_metal_kernel_config_add_output_arg(config, outputShape, outputShape.count, dtype), "Router output shape")
                }
                try MX.check(mlx_fast_metal_kernel_config_set_grid(config, Int32(threadGroup), Int32(rows), 1), "Router grid")
                try MX.check(mlx_fast_metal_kernel_config_set_thread_group(config, Int32(threadGroup), 1, 1), "Router threadgroup")
                try MX.check(mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "TOUT", MLX_BFLOAT16), "Router output dtype")
                for (name, value) in [("NE", expertCount), ("NK", topK), ("TG", threadGroup), ("NORM", 1)] {
                    try MX.check(mlx_fast_metal_kernel_config_add_template_arg_int(config, name, Int32(value)), "Router template")
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
    uint row = thread_position_in_grid.y;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;

    threadgroup float rk[NE];
    threadgroup uint sel[NK];

    size_t rbase = (size_t)row * (size_t)NE;
    for (uint e = tid; e < NE; e += TG) {
      rk[e] = float(logits[rbase + e]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // The first SIMD group emulates MLX's softmax_single_row reduction tree,
    // including four consecutive values per virtual thread and fast::exp.
    // Compute this before top-k selection masks the staged logits.
    if (tid < 32) {
      constexpr int VT = (NE + 3) / 4;
      constexpr int SIMDS = (VT + 31) / 32;
      float lane_max = -INFINITY;
      for (int g = 0; g < SIMDS; ++g) {
        int vt = g * 32 + int(lane);
        float m = -INFINITY;
        for (int i = 0; i < 4; ++i) {
          int e = vt * 4 + i;
          if (e < NE) m = metal::max(m, rk[e]);
        }
        float mg = simd_max(m);
        if (int(lane) == g) lane_max = mg;
      }
      float smax = simd_max(lane_max);
      float lane_sum = 0.0f;
      for (int g = 0; g < SIMDS; ++g) {
        int vt = g * 32 + int(lane);
        float acc = 0.0f;
        for (int i = 0; i < 4; ++i) {
          int e = vt * 4 + i;
          acc += (e < NE) ? fast::exp(rk[e] - smax) : 0.0f;
        }
        float pg = simd_sum(acc);
        if (int(lane) == g) lane_sum = pg;
      }
      float snorm = 1.0f / simd_sum(lane_sum);

      // Largest raw logit first; ties always choose the lowest expert ID.
      for (uint r = 0; r < NK; ++r) {
        float best = -INFINITY;
        uint bidx = 0xFFFFFFFFu;
        for (uint e = lane; e < NE; e += 32) {
          float v = rk[e];
          if (v > best) { best = v; bidx = e; }
        }
        float gmax = simd_max(best);
        uint cand = (best == gmax) ? bidx : 0xFFFFFFFFu;
        uint gidx = metal::min(simd_min(cand), uint(NE - 1));
        if (lane == 0) { sel[r] = gidx; rk[gidx] = -INFINITY; }
        simdgroup_barrier(mem_flags::mem_threadgroup);
      }

      float w = 0.0f;
      uint myidx = 0;
      if (lane < NK) {
        myidx = sel[lane];
        w = float(TOUT(fast::exp(float(logits[rbase + myidx]) - smax) * snorm));
      }
      // Round each probability and every ascending denominator addition to
      // BF16. A FP32 or butterfly sum would change the model's routing weights.
      float tot = 0.0f;
      for (int j = 0; j < NK; ++j) tot = float(TOUT(simd_shuffle(w, ushort(j)) + tot));
      w = w / float(TOUT(tot));
      if (lane < NK) {
        inds[(size_t)row * (size_t)NK + lane] = myidx;
        scores[(size_t)row * (size_t)NK + lane] = TOUT(w);
      }
    }
    """
}
