// Reduction order adapted from Apple MLX at pinned mlx=1f8e74e3f12f:
// backend/metal/reduce.cpp::strided_reduce_small and
// kernels/reduction/reduce_col.h::col_reduce_small / ops.h::Sum.
// Copyright © 2023-2024 Apple Inc.
//
// MIT License
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

import CMLX

/// Fuses the sorted expert-output gather, BF16 score products, and reference
/// top-10 sum for the model's H=2560 prefill path. No FP32 accumulation mode.
///
/// The stock contiguous [1,S,10,2560] product takes col_reduce_small: x=32,
/// y=min(8, maxThreads/32, 10)=8 when that pipeline supports >=256 threads.
/// Its BF16 partials are (0,8), (1,9), 2,3,4,5,6,7, folded in that order.
/// This candidate preserves those boundaries, independently of its own
/// threadgroup size. Real-input bitwise validation must confirm the stock
/// pipeline uses this geometry; this is not a general replacement for sum.
///
/// Like the surrounding inference objects, this cache is single-executor and
/// not Sendable. Kernel compilation/evaluation remains lazy on MX.stream.
public final class GPUMoEPrefillReduction {
    public let threadgroupSize: Int
    private static let hidden = 2560
    private static let topK = 10
    private var programs: [Int: Program] = [:]

    public init(threadgroupSize: Int) throws {
        guard [128, 256, 512].contains(threadgroupSize) else {
            throw GPUError.invalid("Prefill reduction threadgroup must be 128, 256, or 512")
        }
        self.threadgroupSize = threadgroupSize
    }

    /// `inverseOrder` must be the original U32 argsort inversion over all
    /// token/expert assignments. Its values are not copied to the CPU or
    /// revalidated here. The kernel's row-contiguous input setting handles
    /// strided inputs by MLX's normal copy path when necessary.
    public func reduce(projected: Tensor, inverseOrder: Tensor, scores: Tensor,
                       tokens: Int) throws -> Tensor {
        guard tokens > 0, tokens <= Int(Int32.max) / (Self.topK * Self.hidden) else {
            throw GPUError.invalid("Prefill reduction token count is empty or exceeds indexed geometry")
        }
        let assignments = tokens * Self.topK
        guard projected.shape == [assignments, 1, Self.hidden], projected.dtype == MLX_BFLOAT16,
              inverseOrder.shape == [assignments], inverseOrder.dtype == MLX_UINT32,
              scores.shape == [1, tokens, Self.topK], scores.dtype == MLX_BFLOAT16 else {
            throw GPUError.invalid("Prefill reduction requires BF16 [S*10,1,2560], U32 [S*10], and BF16 [1,S,10]")
        }
        let program: Program
        if let cached = programs[tokens] { program = cached }
        else {
            program = try Program(tokens: tokens, threadgroupSize: threadgroupSize)
            programs[tokens] = program
        }
        return try program.apply([projected, inverseOrder, scores])
    }

    private final class Program {
        private let kernel: mlx_fast_metal_kernel
        private let config: mlx_fast_metal_kernel_config

        init(tokens: Int, threadgroupSize: Int) throws {
            let namesIn = mlx_vector_string_new(), namesOut = mlx_vector_string_new()
            defer { _ = mlx_vector_string_free(namesIn); _ = mlx_vector_string_free(namesOut) }
            for name in ["projected", "inverse_order", "scores"] {
                try MX.check(mlx_vector_string_append_value(namesIn, name), "Prefill reduction input name")
            }
            try MX.check(mlx_vector_string_append_value(namesOut, "y"), "Prefill reduction output name")
            let created = mlx_fast_metal_kernel_new("ane_runner_prefill_unsort_bf16_reduce8",
                namesIn, namesOut, GPUMoEPrefillReduction.source, "", true, false)
            guard created.ctx != nil else { throw GPUError.invalid("Failed to create prefill reduction kernel") }
            let configuration = mlx_fast_metal_kernel_config_new()
            do {
                let shape = [Int32(1), Int32(tokens), Int32(GPUMoEPrefillReduction.hidden)]
                try MX.check(mlx_fast_metal_kernel_config_add_output_arg(
                    configuration, shape, shape.count, MLX_BFLOAT16), "Prefill reduction output")
                let count = tokens * GPUMoEPrefillReduction.hidden
                let grid = (count + threadgroupSize - 1) / threadgroupSize * threadgroupSize
                try MX.check(mlx_fast_metal_kernel_config_set_grid(configuration, Int32(grid), 1, 1), "Prefill reduction grid")
                try MX.check(mlx_fast_metal_kernel_config_set_thread_group(
                    configuration, Int32(threadgroupSize), 1, 1), "Prefill reduction threadgroup")
                try MX.check(mlx_fast_metal_kernel_config_add_template_arg_dtype(configuration, "T", MLX_BFLOAT16), "Prefill reduction dtype")
                for (name, value) in [("H", GPUMoEPrefillReduction.hidden),
                                      ("TOPK", GPUMoEPrefillReduction.topK), ("COUNT", count)] {
                    try MX.check(mlx_fast_metal_kernel_config_add_template_arg_int(
                        configuration, name, Int32(value)), "Prefill reduction template")
                }
            } catch {
                mlx_fast_metal_kernel_free(created)
                mlx_fast_metal_kernel_config_free(configuration)
                throw error
            }
            kernel = created; config = configuration
        }

        deinit { mlx_fast_metal_kernel_free(kernel); mlx_fast_metal_kernel_config_free(config) }

        func apply(_ inputs: [Tensor]) throws -> Tensor {
            let inputVector = mlx_vector_array_new_data(inputs.map(\.handle), inputs.count)
            var outputVector = mlx_vector_array_new()
            defer { _ = mlx_vector_array_free(inputVector); _ = mlx_vector_array_free(outputVector) }
            try MX.check(mlx_fast_metal_kernel_apply(&outputVector, kernel, inputVector, config, MX.stream),
                         "Prefill fused unsort/product/reduction")
            guard mlx_vector_array_size(outputVector) == 1 else {
                throw GPUError.invalid("Prefill reduction output count mismatch")
            }
            return try MX.output("Prefill reduction output") { mlx_vector_array_get(&$0, outputVector, 0) }
        }
    }

    private static let source = """
    uint i = thread_position_in_grid.x;
    if (i >= uint(COUNT)) return;
    uint token = i / uint(H);
    uint feature = i % uint(H);
    T partials[8];
    // Match each stock col_reduce_small lid.y, including its initial +0.
    for (uint lane = 0; lane < 8; ++lane) {
        T partial = T(0.0f);
        for (uint slot = lane; slot < uint(TOPK); slot += 8) {
            size_t assignment = (size_t)token * uint(TOPK) + slot;
            size_t sorted_row = (size_t)inverse_order[assignment];
            T product = T(float(projected[sorted_row * uint(H) + feature])
                          * float(scores[assignment]));
            partial = T(float(product) + float(partial));
        }
        partials[lane] = partial;
    }
    T total = partials[0];
    for (uint lane = 1; lane < 8; ++lane) {
        total = T(float(partials[lane]) + float(total));
    }
    y[i] = total;
    """
}
