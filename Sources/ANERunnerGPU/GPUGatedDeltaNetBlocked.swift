// Adapted from garnermccloud/mlx-serve src/transformer.zig,
// GDN_KERNEL_BLOCKED_BODY, fixed commit
// 7dbcba04c98e4fd3bcc533c63e645547f13cc3b1.
//
// Metal kernel ancestry: oMLX by jundot, Copyright oMLX contributors,
// custom_kernels/qwen35_prefill/gdn.py, gated_delta_blocked_seq.
// Licensed under the Apache License, Version 2.0. You may not use that
// portion except in compliance with the License. Obtain a copy at
// https://www.apache.org/licenses/LICENSE-2.0 or see LICENSE-APACHE-2.0.
// Distributed on an AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied. See the License for its permissions and
// limitations. Full attribution chain is retained in UPSTREAM-NOTICE.
//
// Changes here: Swift ownership/shape validation; fixed BF16 input/state/output,
// Dk128/Dv128/Hk16/Hv48/B1/TB32. The upstream Metal body remains unchanged.
// The mlx-serve wrapper adaptation is covered by the MIT terms below.
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

/// Fixed-shape blocked prefill recurrence; one inference session owns each instance.
/// FP32 register state is rounded to BF16 only at the end of this call.
/// Its eight-lane reduction differs from the scalar kernel's summation order;
/// compare both y and the final state before enabling this experimental path.
public final class GPUGatedDeltaNetBlocked {
    private let kernel: mlx_fast_metal_kernel

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
            mlx_fast_metal_kernel_new("ane_runner_qwen38_gdn_blocked_bf16_tb32_v1",inputs,outputs,$0,"",true,false)
        }
        guard kernel.ctx != nil else { throw GPUAttentionError.invalid("Could not create blocked GDN kernel") }
    }
    deinit { mlx_fast_metal_kernel_free(kernel) }

    public func apply(q: Tensor, k: Tensor, v: Tensor, decay: Tensor, beta: Tensor,
                      state: Tensor) throws -> (y: Tensor, state: Tensor) {
        guard q.shape.count == 4, q.shape[0] == 1, q.shape[1] >= 64, q.shape[1] <= Int(Int32.max),
              q.shape[2...] == [16,128], k.shape == q.shape,
              v.shape == [1,q.shape[1],48,128],
              decay.shape == [1,q.shape[1],48], beta.shape == decay.shape,
              state.shape == [1,48,128,128],
              [q,k,v,decay,beta,state].allSatisfy({ $0.dtype == MLX_BFLOAT16 }) else {
            throw GPUAttentionError.invalid("Blocked GDN requires S>=64 and BF16 Q/K [1,S,16,128], V [1,S,48,128], gates [1,S,48], state [1,48,128,128]")
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
        let outputShape: [Int32] = [1,Int32(sequence),48,128], stateShape: [Int32] = [1,48,128,128]
        try MX.check(mlx_fast_metal_kernel_config_add_output_arg(configuration,outputShape,4,MLX_BFLOAT16),"GDN output shape")
        try MX.check(mlx_fast_metal_kernel_config_add_output_arg(configuration,stateShape,4,MLX_BFLOAT16),"GDN state shape")
        try MX.check(mlx_fast_metal_kernel_config_set_grid(configuration,1024,48,1),"GDN grid")
        try MX.check(mlx_fast_metal_kernel_config_set_thread_group(configuration,256,1,1),"GDN threadgroup")
        try MX.check(mlx_fast_metal_kernel_apply(&outputs,kernel,inputs,configuration,MX.stream),"blocked GDN apply")
        guard mlx_vector_array_size(outputs) == 2 else { throw GPUAttentionError.invalid("Wrong blocked GDN output count") }
        let y = try MX.output("GDN y") { mlx_vector_array_get(&$0,outputs,0) }
        let next = try MX.output("GDN state") { mlx_vector_array_get(&$0,outputs,1) }
        return (y,next)
    }

    private static let source = #"""
constexpr int TB = 32;
constexpr int Dk = 128;
constexpr int Dv = 128;
constexpr int Hk = 16;
constexpr int Hv = 48;
using InT = bfloat16_t;
using StT = bfloat16_t;
using OutT = bfloat16_t;
// 20,224 bytes of threadgroup staging; the final token block may be partial.
constexpr int DB = 32;                             // dv rows per threadgroup
const int tid = thread_position_in_threadgroup.x;  // 0..255
const int blk = threadgroup_position_in_grid.x;    // Dv/DB block
const int hv  = threadgroup_position_in_grid.y;
const int b   = threadgroup_position_in_grid.z;
const int hk  = hv / (Hv / Hk);
const int dv0 = blk * DB;

// thread -> (dv row, 16-wide d segment); 8 threads per dv row, all in
// the same simdgroup (lane = (dvr%4)*8 + seg).
const int dvr = tid / 8;            // 0..31
const int seg = tid % 8;            // 0..7
const int d0  = seg * 16;

threadgroup InT k_s[TB][Dk + 8];
threadgroup InT q_s[TB][Dk + 8];
threadgroup InT v_s[TB][DB + 8];
threadgroup float g_s[TB];
threadgroup float b_s[TB];

auto k_base = k + ((size_t)b * T * Hk + hk) * Dk;
auto q_base = q + ((size_t)b * T * Hk + hk) * Dk;
auto v_base = v + ((size_t)b * T * Hv + hv) * Dv + dv0;
const size_t krow = (size_t)Hk * Dk;

// state fragment in registers: [dv0+dvr][d0..d0+16]
float4 st[4];
{
    const device vec<StT,4>* S_in = (const device vec<StT,4>*)(
        state_in + (((size_t)b * Hv + hv) * Dv + dv0 + dvr) * Dk + d0);
    for (int i = 0; i < 4; ++i) st[i] = float4(S_in[i]);
}

device OutT* y_base = y + ((size_t)b * T * Hv + hv) * Dv + dv0;

for (int t0 = 0; t0 < T; t0 += TB) {
    const int tt = min(TB, T - t0);
    // cooperative staging (coalesced): k/q rows, v slice, g/beta
    for (int p = tid; p < tt * Dk; p += 256) {
        const int r = p / Dk, d = p % Dk;
        k_s[r][d] = static_cast<InT>(k_base[(size_t)(t0 + r) * krow + d]);
        q_s[r][d] = static_cast<InT>(q_base[(size_t)(t0 + r) * krow + d]);
    }
    for (int p = tid; p < tt * DB; p += 256) {
        const int r = p / DB, d = p % DB;
        v_s[r][d] = static_cast<InT>(v_base[(size_t)(t0 + r) * Hv * Dv + d]);
    }
    for (int p = tid; p < tt; p += 256) {
        g_s[p] = (float)g[((size_t)b * T + t0 + p) * Hv + hv];
        b_s[p] = (float)beta[((size_t)b * T + t0 + p) * Hv + hv];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int t = 0; t < tt; ++t) {
        const float gt = g_s[t];
        const float bt = b_s[t];
        const threadgroup vec<InT,4>* k4 =
            (const threadgroup vec<InT,4>*)&k_s[t][d0];
        const threadgroup vec<InT,4>* q4 =
            (const threadgroup vec<InT,4>*)&q_s[t][d0];
        float4 kf[4];
        for (int i = 0; i < 4; ++i) kf[i] = float4(k4[i]);
        // kv_mem = (g*state) . k ; decay applied to state first
        float4 p4 = 0.0f;
        for (int i = 0; i < 4; ++i) {
            st[i] *= gt;
            p4 += st[i] * kf[i];
        }
        float part = p4.x + p4.y + p4.z + p4.w;
        // reduce across the 8 segment-threads of this dv row
        part += simd_shuffle_down(part, 4);
        part += simd_shuffle_down(part, 2);
        part += simd_shuffle_down(part, 1);
        const float kv_mem = simd_shuffle(part, (tid % 32) / 8 * 8);
        const float delta = ((float)v_s[t][dvr] - kv_mem) * bt;

        float4 o4 = 0.0f;
        for (int i = 0; i < 4; ++i) {
            st[i] += kf[i] * delta;
            o4 += st[i] * float4(q4[i]);
        }
        float out = o4.x + o4.y + o4.z + o4.w;
        out += simd_shuffle_down(out, 4);
        out += simd_shuffle_down(out, 2);
        out += simd_shuffle_down(out, 1);
        if (seg == 0) {
            y_base[(size_t)(t0 + t) * Hv * Dv + dvr] = static_cast<OutT>(out);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

{
    device vec<StT,4>* S_out = (device vec<StT,4>*)(
        state_out + (((size_t)b * Hv + hv) * Dv + dv0 + dvr) * Dk + d0);
    for (int i = 0; i < 4; ++i) S_out[i] = vec<StT,4>(st[i]);
}
"""#
}
