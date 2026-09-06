// Adapted from garnermccloud/mlx-serve src/transformer.zig, fixed commit
// 7dbcba04c98e4fd3bcc533c63e645547f13cc3b1, GDN_PREWORK_SOURCE, GDN_NORMGATE_SOURCE and GDN_KERNEL_HEADER.
// The Metal source bodies below are retained verbatim; Swift owns handles/configs.
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


// Prework ancestry: Layr-Labs/mlxfast-challenge,
// Copyright (c) 2026 Layr Labs, Inc.; see UPSTREAM-NOTICE.
import CMLX
import Foundation

/// The pinned author's decode-width prework and sigmoid norm-gate kernels.
/// Confined to one inference thread. Shape-specific configs cache within this
/// instance; no kernel is evaluated during initialization.
public final class GPUGatedDeltaNetPrework {
    public struct Output {
        public let q, k, v, history, decay, beta: Tensor
    }
    private let convolution, aLog, dtBias, normWeight: Tensor
    private let qScale, kScale, epsilon: Tensor
    private let preworkKernel: GDNPreworkMetalKernel
    private let normGateKernel: GDNPreworkMetalKernel

    public init(convolution: Tensor, aLog: Tensor, dtBias: Tensor, normWeight: Tensor) throws {
        guard convolution.shape == [10240,4,1], aLog.shape == [48], dtBias.shape == [48], normWeight.shape == [128],
              [convolution,aLog,dtBias,normWeight].allSatisfy({ $0.dtype == MLX_BFLOAT16 }) else {
            throw GPUAttentionError.invalid("GDN prework weight shape/dtype mismatch")
        }
        self.convolution = convolution; self.aLog = aLog; self.dtBias = dtBias; self.normWeight = normWeight
        qScale = try MX.scalar(1.0/128,MLX_BFLOAT16)
        kScale = try MX.scalar(1.0/sqrt(128.0),MLX_BFLOAT16)
        epsilon = try MX.scalar(1e-6,MLX_FLOAT32)
        preworkKernel = try GDNPreworkMetalKernel(name: "ane_runner_gdn_prework_v1",
            inputs: ["qkv","conv_state","conv_w","q_scale","k_scale","b_in","a_in","A_log","dt_bias"],
            outputs: ["q_out","k_out","v_out","conv_out","g_out","beta_out"],source: Self.preworkSource,header: Self.header)
        normGateKernel = try GDNPreworkMetalKernel(name: "ane_runner_gdn_normgate_v1",
            inputs: ["y","z","norm_w","eps"],outputs: ["out"],source: Self.normGateSource,header: "")
    }

    public func apply(qkv: Tensor, a: Tensor, b: Tensor, history: Tensor) throws -> Output {
        guard qkv.shape.count == 3, qkv.shape[0] == 1, (1...9).contains(qkv.shape[1]), qkv.shape[2] == 10240,
              a.shape == [1,qkv.shape[1],48], b.shape == a.shape, history.shape == [1,3,10240],
              [qkv,a,b,history].allSatisfy({ $0.dtype == MLX_BFLOAT16 }) else {
            throw GPUAttentionError.invalid("GDN prework requires BF16 QKV [1,S,10240], gates [1,S,48], history [1,3,10240], S1...9")
        }
        let sequence = qkv.shape[1]
        let qShape = [1,sequence,16,128], vShape = [1,sequence,48,128], gShape = [1,sequence,48]
        let constants: [(String,Int)] = [("HK",16),("HV",48),("DK",128),("DV",128),("NKEEP",3),("C",10240),
            ("S",sequence),("QSTRIDE",10240),("QOFF",0),("BSTRIDE",48),("BOFF",0),("ASTRIDE",48),("AOFF",0)]
        let result = try preworkKernel.apply(key: sequence,inputs: [qkv,history,convolution,qScale,kScale,b,a,aLog,dtBias],
            outputShapes: [qShape,qShape,vShape,[1,3,10240],gShape,gShape],grid: (32,sequence,80),constants: constants)
        return Output(q: result[0],k: result[1],v: result[2],history: result[3],decay: result[4],beta: result[5])
    }

    /// y [1,S,48,128], z [1,S,6144] -> flattened gated output [1,S,6144].
    public func normGate(y: Tensor, z: Tensor) throws -> Tensor {
        guard y.shape.count == 4, y.shape[0] == 1, (1...9).contains(y.shape[1]), y.shape[2...] == [48,128],
              z.shape == [1,y.shape[1],6144], y.dtype == MLX_BFLOAT16, z.dtype == MLX_BFLOAT16 else {
            throw GPUAttentionError.invalid("GDN norm-gate requires BF16 Y [1,S,48,128], Z [1,S,6144], S1...9")
        }
        let sequence = y.shape[1]
        return try normGateKernel.apply(key: sequence,inputs: [y,z,normWeight,epsilon],outputShapes: [[1,sequence,6144]],
            grid: (32,sequence,48),constants: [("HV",48),("DV",128),("ZSTRIDE",6144),("ZOFF",0),("SWISH",0)])[0]
    }

    private static let header = #"""
inline float msv_log1p(float x) {
    float xp1 = 1.0f + x;
    if (xp1 == metal::numeric_limits<float>::max()) { return metal::numeric_limits<float>::max(); }
    if (xp1 == 1.0f) { return x; }
    return x * (metal::log(xp1) / (xp1 - 1.0f));
}
"""#

    private static let preworkSource = #"""
uint lane = thread_position_in_threadgroup.x;
// Grid row = b*S + r over the batch: q/k/v/g/beta/b/a are [B,S,..] flat
// so `row` indexes them directly; the conv taps + next state are per batch.
uint row = threadgroup_position_in_grid.y;
uint b = row / uint(S);
uint r = row - b * uint(S);
uint logical_head = threadgroup_position_in_grid.z;
constexpr uint q_heads = uint(HK);
constexpr uint k_head_base = uint(HK);
constexpr uint v_head_base = 2 * uint(HK);
bool is_q = logical_head < q_heads;
bool is_k = logical_head >= k_head_base && logical_head < v_head_base;
uint head = is_q ? logical_head
           : (is_k ? logical_head - k_head_base : logical_head - v_head_base);
uint channel_base = is_q ? head * uint(DK)
                   : (is_k ? uint(HK) * uint(DK) + head * uint(DK)
                           : 2 * uint(HK) * uint(DK) + head * uint(DV));
T activated[4];
float sumsq = 0.0f;
for (uint i = 0; i < 4; ++i) {
    uint channel = channel_base + lane * 4 + i;
    float acc = 0.0f;
    for (uint tap = 0; tap < 4; ++tap) {
        uint input_row = r + tap;
        const T xv = input_row < uint(NKEEP)
            ? conv_state[(b * uint(NKEEP) + input_row) * uint(C) + channel]
            : qkv[(row + tap - uint(NKEEP)) * uint(QSTRIDE) + uint(QOFF) + channel];
        acc += float(xv) * float(conv_w[channel * 4 + tap]);
    }
    const T conv = T(acc);
    // MLX's unary Sigmoid formula (unary_ops.h), verbatim, in the tensor dtype.
    T sy = T(1) / (T(1) + metal::exp(metal::abs(conv)));
    const T act = conv * ((conv < T(0)) ? sy : T(1) - sy);
    activated[i] = act;
    float value = float(act);
    sumsq += value * value;
}
if (is_q || is_k) {
    sumsq = simd_sum(sumsq);
    float inv = metal::precise::rsqrt(sumsq / float(DK) + 1e-6f);
    const T scale = is_q ? q_scale : k_scale;
    uint out_base = (row * uint(HK) + head) * uint(DK) + lane * 4;
    for (uint i = 0; i < 4; ++i) {
        // ones-weight rms_norm rounding (T(x*inv)), then the separate
        // scalar multiply's rounding — the composed chain's two casts.
        const T rms = T(1) * T(float(activated[i]) * inv);
        const T value = scale * rms;
        if (is_q) {
            q_out[out_base + i] = value;
        } else {
            k_out[out_base + i] = value;
        }
    }
} else {
    uint out_base = (row * uint(HV) + head) * uint(DV) + lane * 4;
    for (uint i = 0; i < 4; ++i) {
        v_out[out_base + i] = activated[i];
    }
    if (lane == 0) {
        // beta = sigmoid(b) (MLX unary formula); g = exp(-exp(A_log) *
        // softplus(a + dt_bias)) with the compiled chain's own casts:
        // bf16 add, f32 precise exp / log1p / exp, bf16 store.
        const T bv = b_in[row * uint(BSTRIDE) + uint(BOFF) + head];
        T by = T(1) / (T(1) + metal::exp(metal::abs(bv)));
        beta_out[row * uint(HV) + head] = (bv < T(0)) ? by : T(1) - by;
        const T apd = T(float(a_in[row * uint(ASTRIDE) + uint(AOFF) + head]) + float(dt_bias[head]));
        float sp = msv_log1p(metal::precise::exp(float(apd)));
        float ea = metal::precise::exp(float(A_log[head]));
        g_out[row * uint(HV) + head] = T(metal::precise::exp(-(ea * sp)));
    }
}
// Next conv state = rows [S, S+NKEEP) of concat(conv_state, qkv).
if (r + uint(NKEEP) >= uint(S)) {
    uint state_row = r + uint(NKEEP) - uint(S);
    uint raw_base = row * uint(QSTRIDE) + uint(QOFF) + channel_base + lane * 4;
    uint state_base = (b * uint(NKEEP) + state_row) * uint(C) + channel_base + lane * 4;
    for (uint i = 0; i < 4; ++i) {
        conv_out[state_base + i] = qkv[raw_base + i];
    }
}
if (r == 0) {
    for (uint rr = 0; rr + uint(S) < uint(NKEEP); ++rr) {
        uint src_base = (b * uint(NKEEP) + rr + uint(S)) * uint(C) + channel_base + lane * 4;
        uint dst_base = (b * uint(NKEEP) + rr) * uint(C) + channel_base + lane * 4;
        for (uint i = 0; i < 4; ++i) {
            conv_out[dst_base + i] = conv_state[src_base + i];
        }
    }
}
"""#

    private static let normGateSource = #"""
uint lane = thread_position_in_threadgroup.x;
uint row = threadgroup_position_in_grid.y;
uint head = threadgroup_position_in_grid.z;
uint base = (row * uint(HV) + head) * uint(DV) + lane * 4;
float xs[4];
float sumsq = 0.0f;
for (uint i = 0; i < 4; ++i) {
    xs[i] = float(y[base + i]);
    sumsq += xs[i] * xs[i];
}
sumsq = simd_sum(sumsq);
float inv = metal::precise::rsqrt(sumsq / float(DV) + eps);
uint zbase = row * uint(ZSTRIDE) + uint(ZOFF) + head * uint(DV) + lane * 4;
for (uint i = 0; i < 4; ++i) {
    const T normed = norm_w[lane * 4 + i] * T(xs[i] * inv);
    const T zv = z[zbase + i];
    T sy = T(1) / (T(1) + metal::exp(metal::abs(zv)));
    const T sig = (zv < T(0)) ? sy : T(1) - sy;
    // swish gate: silu(z) * normed (qwen3.5); sigmoid gate: normed * sigmoid(z) (qwen4_exp, KDA)
    out[base + i] = SWISH ? (zv * sig) * normed : normed * sig;
}
"""#
}

private final class GDNPreworkMetalKernel {
    private let kernel: mlx_fast_metal_kernel
    private var configurations: [Int: GDNPreworkConfiguration] = [:]
    private let outputCount: Int

    init(name: String, inputs: [String], outputs: [String], source: String, header: String) throws {
        let inputNames = mlx_vector_string_new(), outputNames = mlx_vector_string_new()
        defer { _ = mlx_vector_string_free(inputNames); _ = mlx_vector_string_free(outputNames) }
        for item in inputs { try MX.check(mlx_vector_string_append_value(inputNames,item),"GDN prework input name") }
        for item in outputs { try MX.check(mlx_vector_string_append_value(outputNames,item),"GDN prework output name") }
        kernel = mlx_fast_metal_kernel_new(name,inputNames,outputNames,source,header,true,false)
        outputCount = outputs.count
        guard kernel.ctx != nil else { throw GPUAttentionError.invalid("Could not create GDN prework Metal kernel") }
    }
    deinit { mlx_fast_metal_kernel_free(kernel) }

    func apply(key: Int, inputs: [Tensor], outputShapes: [[Int]], grid: (Int,Int,Int),
               constants: [(String,Int)]) throws -> [Tensor] {
        let configuration: GDNPreworkConfiguration
        if let cached = configurations[key] { configuration = cached }
        else {
            configuration = try GDNPreworkConfiguration(outputShapes: outputShapes,grid: grid,constants: constants)
            configurations[key] = configuration
        }
        let arguments = mlx_vector_array_new_data(inputs.map(\.handle),inputs.count)
        var result = mlx_vector_array_new()
        defer { _ = mlx_vector_array_free(arguments); _ = mlx_vector_array_free(result) }
        try MX.check(mlx_fast_metal_kernel_apply(&result,kernel,arguments,configuration.handle,MX.stream),"GDN fused prework apply")
        guard mlx_vector_array_size(result) == outputCount else { throw GPUAttentionError.invalid("GDN prework output count mismatch") }
        return try (0..<outputCount).map { index in
            try MX.output("GDN prework output") { mlx_vector_array_get(&$0,result,index) }
        }
    }
}

private final class GDNPreworkConfiguration {
    let handle: mlx_fast_metal_kernel_config
    init(outputShapes: [[Int]], grid: (Int,Int,Int), constants: [(String,Int)]) throws {
        let configuration = mlx_fast_metal_kernel_config_new()
        do {
            for shape in outputShapes {
                let dimensions = shape.map(Int32.init)
                try MX.check(mlx_fast_metal_kernel_config_add_output_arg(configuration,dimensions,dimensions.count,MLX_BFLOAT16),"GDN prework output shape")
            }
            try MX.check(mlx_fast_metal_kernel_config_set_grid(configuration,Int32(grid.0),Int32(grid.1),Int32(grid.2)),"GDN prework grid")
            try MX.check(mlx_fast_metal_kernel_config_set_thread_group(configuration,32,1,1),"GDN prework threadgroup")
            try MX.check(mlx_fast_metal_kernel_config_add_template_arg_dtype(configuration,"T",MLX_BFLOAT16),"GDN prework dtype")
            for (name,value) in constants {
                try MX.check(mlx_fast_metal_kernel_config_add_template_arg_int(configuration,name,Int32(value)),"GDN prework constant")
            }
            handle = configuration
        } catch {
            mlx_fast_metal_kernel_config_free(configuration)
            throw error
        }
    }
    deinit { mlx_fast_metal_kernel_config_free(handle) }
}
