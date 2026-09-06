// Adapted from garnermccloud/mlx-serve src/transformer.zig, fixed commit
// 7dbcba04c98e4fd3bcc533c63e645547f13cc3b1. The upstream recurrence credits
// mlx-lm (Copyright (c) 2023-2026 Apple Inc.). Swift composed-op adaptation.
//
// MIT License — Copyright (c) 2026 David Dalcu
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
import Foundation

/// Port of the author's scalar GDN recurrence kernel, used by the fused
/// recurrence path for decode and, by default, prefill.
public final class GPUGatedDeltaNetFused {
    private let kernel: mlx_fast_metal_kernel
    private var roundedStateKernel: mlx_fast_metal_kernel?
    private var captureKernels: [Bool: mlx_fast_metal_kernel] = [:]

    public init() throws {
        let inputs = mlx_vector_string_new(), outputs = mlx_vector_string_new()
        defer { _ = mlx_vector_string_free(inputs); _ = mlx_vector_string_free(outputs) }
        for name in ["q","k","v","g","beta","state_in","T"] {
            try MX.check(name.withCString { mlx_vector_string_append_value(inputs,$0) },"GDN input name")
        }
        for name in ["y","state_out"] {
            try MX.check(name.withCString { mlx_vector_string_append_value(outputs,$0) },"GDN output name")
        }
        kernel = Self.source.withCString {
            mlx_fast_metal_kernel_new("ane_runner_qwen38_gdn_scalar_v1",inputs,outputs,$0,"",true,false)
        }
        guard kernel.ctx != nil else { throw GPUAttentionError.invalid("Could not create fused GDN kernel") }
    }
    deinit {
        mlx_fast_metal_kernel_free(kernel)
        if let roundedStateKernel { mlx_fast_metal_kernel_free(roundedStateKernel) }
        for kernel in captureKernels.values { mlx_fast_metal_kernel_free(kernel) }
    }

    /// Kept separate so ordinary prefill/decode retain the original kernel
    /// name and source. Kernel objects are confined to the inference session.
    private func getKernel(roundStateEachToken: Bool) throws -> mlx_fast_metal_kernel {
        guard roundStateEachToken else { return kernel }
        if let roundedStateKernel { return roundedStateKernel }
        let inputs = mlx_vector_string_new(), outputs = mlx_vector_string_new()
        defer { _ = mlx_vector_string_free(inputs); _ = mlx_vector_string_free(outputs) }
        for name in ["q","k","v","g","beta","state_in","T"] {
            try MX.check(name.withCString { mlx_vector_string_append_value(inputs,$0) },"GDN verify input name")
        }
        for name in ["y","state_out"] {
            try MX.check(name.withCString { mlx_vector_string_append_value(outputs,$0) },"GDN verify output name")
        }
        let candidate = Self.roundedStateSource.withCString {
            mlx_fast_metal_kernel_new("ane_runner_qwen38_gdn_scalar_bf16_boundaries_v1",inputs,outputs,$0,"",true,false)
        }
        guard candidate.ctx != nil else { throw GPUAttentionError.invalid("Could not create GDN verify kernel") }
        roundedStateKernel = candidate
        return candidate
    }

    private func getCaptureKernel(roundStateEachToken: Bool) throws -> mlx_fast_metal_kernel {
        if let kernel = captureKernels[roundStateEachToken] { return kernel }
        let inputs = mlx_vector_string_new(), outputs = mlx_vector_string_new()
        defer { _ = mlx_vector_string_free(inputs); _ = mlx_vector_string_free(outputs) }
        for name in ["q","k","v","g","beta","state_in","T"] {
            try MX.check(name.withCString { mlx_vector_string_append_value(inputs,$0) },"GDN capture input name")
        }
        for name in ["y","state_out","state_seq"] {
            try MX.check(name.withCString { mlx_vector_string_append_value(outputs,$0) },"GDN capture output name")
        }
        let source = "constexpr bool ROUND_STATE = \(roundStateEachToken ? "true" : "false");\n" + Self.captureSource
        let name = "ane_runner_qwen38_gdn_capture_\(roundStateEachToken ? "rounded" : "raw")_v1"
        let candidate = source.withCString { source in
            name.withCString { mlx_fast_metal_kernel_new($0,inputs,outputs,source,"",true,false) }
        }
        guard candidate.ctx != nil else { throw GPUAttentionError.invalid("Could not create GDN capture kernel") }
        captureKernels[roundStateEachToken] = candidate
        return candidate
    }

    public func apply(q: Tensor, k: Tensor, v: Tensor, decay: Tensor, beta: Tensor,
                      state: Tensor, roundStateEachToken: Bool = false) throws -> (y: Tensor, state: Tensor) {
        guard q.shape.count == 4, q.shape[0] == 1, q.shape[1] > 0,
              q.shape[2...] == [16,128], k.shape == q.shape,
              v.shape == [1,q.shape[1],48,128],
              decay.shape == [1,q.shape[1],48], beta.shape == decay.shape,
              state.shape == [1,48,128,128],
              [q,k,v,decay,beta,state].allSatisfy({ $0.dtype == MLX_BFLOAT16 }) else {
            throw GPUAttentionError.invalid("Fused GDN requires BF16 Q/K [1,S,16,128], V [1,S,48,128], gates [1,S,48], state [1,48,128,128]")
        }
        let sequence = q.shape[1]
        let t = try MX.scalar(Float(sequence),MLX_INT32)
        let arguments = [q,k,v,decay,beta,state,t]
        let inputs = mlx_vector_array_new_data(arguments.map(\.handle),arguments.count)
        var outputs = mlx_vector_array_new()
        let configuration = mlx_fast_metal_kernel_config_new()
        defer {
            _ = mlx_vector_array_free(inputs)
            _ = mlx_vector_array_free(outputs)
            mlx_fast_metal_kernel_config_free(configuration)
        }
        let outputShape: [Int32] = [1,Int32(sequence),48,128], stateShape: [Int32] = [1,48,128,128]
        try MX.check(mlx_fast_metal_kernel_config_add_output_arg(configuration,outputShape,4,MLX_BFLOAT16),"GDN output shape")
        try MX.check(mlx_fast_metal_kernel_config_add_output_arg(configuration,stateShape,4,MLX_BFLOAT16),"GDN state shape")
        try MX.check(mlx_fast_metal_kernel_config_set_grid(configuration,32,128,48),"GDN grid")
        try MX.check(mlx_fast_metal_kernel_config_set_thread_group(configuration,32,4,1),"GDN threadgroup")
        let selectedKernel = try getKernel(roundStateEachToken: roundStateEachToken)
        try MX.check(mlx_fast_metal_kernel_apply(&outputs,selectedKernel,inputs,configuration,MX.stream),"fused GDN apply")
        guard mlx_vector_array_size(outputs) == 2 else { throw GPUAttentionError.invalid("Wrong fused GDN output count") }
        let y = try MX.output("GDN y") { mlx_vector_array_get(&$0,outputs,0) }
        let next = try MX.output("GDN state") { mlx_vector_array_get(&$0,outputs,1) }
        return (y,next)
    }

    /// Per-position BF16 persistent snapshots for committing an accepted
    /// verification prefix. Capture is bounded to S1...5; ordinary apply stays
    /// on its existing two-output kernel without allocating this state bank.
    public func applyCapturing(q: Tensor, k: Tensor, v: Tensor, decay: Tensor, beta: Tensor,
                               state: Tensor, roundStateEachToken: Bool = false) throws
        -> (y: Tensor, state: Tensor, states: Tensor) {
        guard q.shape.count == 4, q.shape[0] == 1, (1...5).contains(q.shape[1]),
              q.shape[2...] == [16,128], k.shape == q.shape,
              v.shape == [1,q.shape[1],48,128],
              decay.shape == [1,q.shape[1],48], beta.shape == decay.shape,
              state.shape == [1,48,128,128],
              [q,k,v,decay,beta,state].allSatisfy({ $0.dtype == MLX_BFLOAT16 }) else {
            throw GPUAttentionError.invalid("GDN capture requires S1...5, BF16 Q/K [1,S,16,128], V [1,S,48,128], gates [1,S,48], state [1,48,128,128]")
        }
        let sequence = q.shape[1]
        let t = try MX.array([Int32(sequence)],shape: [])
        let arguments = [q,k,v,decay,beta,state,t]
        let inputs = mlx_vector_array_new_data(arguments.map(\.handle),arguments.count)
        var outputs = mlx_vector_array_new()
        let configuration = mlx_fast_metal_kernel_config_new()
        defer {
            _ = mlx_vector_array_free(inputs)
            _ = mlx_vector_array_free(outputs)
            mlx_fast_metal_kernel_config_free(configuration)
        }
        let outputShape: [Int32] = [1,Int32(sequence),48,128]
        let stateShape: [Int32] = [1,48,128,128]
        let captureShape: [Int32] = [Int32(sequence),1,48,128,128]
        try MX.check(mlx_fast_metal_kernel_config_add_output_arg(configuration,outputShape,4,MLX_BFLOAT16),"GDN capture output shape")
        try MX.check(mlx_fast_metal_kernel_config_add_output_arg(configuration,stateShape,4,MLX_BFLOAT16),"GDN capture final state shape")
        try MX.check(mlx_fast_metal_kernel_config_add_output_arg(configuration,captureShape,5,MLX_BFLOAT16),"GDN capture sequence shape")
        try MX.check(mlx_fast_metal_kernel_config_set_grid(configuration,32,128,48),"GDN capture grid")
        try MX.check(mlx_fast_metal_kernel_config_set_thread_group(configuration,32,4,1),"GDN capture threadgroup")
        let kernel = try getCaptureKernel(roundStateEachToken: roundStateEachToken)
        try MX.check(mlx_fast_metal_kernel_apply(&outputs,kernel,inputs,configuration,MX.stream),"GDN capture apply")
        guard mlx_vector_array_size(outputs) == 3 else { throw GPUAttentionError.invalid("Wrong GDN capture output count") }
        let y = try MX.output("GDN captured y") { mlx_vector_array_get(&$0,outputs,0) }
        let next = try MX.output("GDN captured final state") { mlx_vector_array_get(&$0,outputs,1) }
        let states = try MX.output("GDN captured states") { mlx_vector_array_get(&$0,outputs,2) }
        return (y,next,states)
    }

    // y is computed from the original FP32 updated state, exactly as in S1.
    // Only after that output do we emulate storing/reloading BF16 persistent
    // state at the scalar call boundary. The final BF16 store is unchanged.
    private static let roundedStateSource = source.replacingOccurrences(
        of: "  q_ += 16 * 128;",
        with: #"""
  for (int i = 0; i < n_per_t; ++i) {
    state[i] = static_cast<float>(static_cast<bfloat16_t>(state[i]));
  }
  q_ += 16 * 128;
"""#)

    private static let captureSource = source.replacingOccurrences(
        of: "  q_ += 16 * 128;",
        with: #"""
  for (int i = 0; i < n_per_t; ++i) {
    const bfloat16_t snapshot = static_cast<bfloat16_t>(state[i]);
    const size_t index = (((size_t)t * 48 + n) * 128 + dv_idx) * 128
        + n_per_t * dk_idx + i;
    state_seq[index] = snapshot;
    if constexpr (ROUND_STATE) state[i] = static_cast<float>(snapshot);
  }
  q_ += 16 * 128;
"""#)

    private static let source = #"""
auto n = thread_position_in_grid.z;
auto hv_idx = n % 48;
auto hk_idx = hv_idx / 3;
constexpr int n_per_t = 4;
auto q_ = q + hk_idx * 128;
auto k_ = k + hk_idx * 128;
auto v_ = v + hv_idx * 128;
y += hv_idx * 128;
auto dk_idx = thread_position_in_threadgroup.x;
auto dv_idx = thread_position_in_grid.y;
auto i_state = state_in + (n * 128 + dv_idx) * 128;
auto o_state = state_out + (n * 128 + dv_idx) * 128;
float state[n_per_t];
for (int i = 0; i < n_per_t; ++i) {
  state[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
}
auto g_ = g;
auto beta_ = beta;
for (int t = 0; t < T; ++t) {
  float kv_mem = 0.0f;
  for (int i = 0; i < n_per_t; ++i) {
    auto s_idx = n_per_t * dk_idx + i;
    state[i] = state[i] * g_[hv_idx];
    kv_mem += state[i] * k_[s_idx];
  }
  kv_mem = simd_sum(kv_mem);
  auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];
  float out = 0.0f;
  for (int i = 0; i < n_per_t; ++i) {
    auto s_idx = n_per_t * dk_idx + i;
    state[i] = state[i] + k_[s_idx] * delta;
    out += state[i] * q_[s_idx];
  }
  out = simd_sum(out);
  if (thread_index_in_simdgroup == 0) {
    y[dv_idx] = static_cast<bfloat16_t>(out);
  }
  q_ += 16 * 128;
  k_ += 16 * 128;
  v_ += 48 * 128;
  y += 48 * 128;
  g_ += 48;
  beta_ += 48;
}
for (int i = 0; i < n_per_t; ++i) {
  o_state[n_per_t * dk_idx + i] = static_cast<bfloat16_t>(state[i]);
}
"""#
}
