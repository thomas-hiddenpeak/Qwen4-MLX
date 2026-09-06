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

import Foundation
import CMLX

/// Qwen3.8's 16-key/48-value-head GatedDeltaNet, including its causal convolution
/// and sigmoid output gate. It accepts the already normalized HC read [1,S,2560].
/// Composed scan is the default numerical baseline. The optional fused scalar
/// recurrence and decode prework/tails are separately gated by
/// probe-gpu-sequence. Blocked prefill is an opt-in experiment.
public final class GPUGatedDeltaNet {
    public enum Recurrence: String { case composed, fused }
    public struct VerificationCapture {
        public let initialOffset: Int
        /// [S,1,48,128,128], BF16 persistent state after each verified token.
        public let recurrentStates: Tensor
        /// [1,S+3,10240], old three-token convolution history followed by QKV.
        public let convInputs: Tensor
    }
    public struct State {
        public var convHistory: Tensor?
        /// Matches the pinned runtime: BF16 between forward calls, FP32 inside
        /// a chunk. The checkpoint's mamba_ssm_dtype field is not implemented
        /// as FP32 persistent storage by that reference runtime.
        public var recurrent: Tensor?
        public var offset: Int
        public var verificationCapture: VerificationCapture?

        public init(convHistory: Tensor? = nil, recurrent: Tensor? = nil, offset: Int = 0,
                    verificationCapture: VerificationCapture? = nil) {
            self.convHistory = convHistory
            self.recurrent = recurrent
            self.offset = offset
            self.verificationCapture = verificationCapture
        }

        public mutating func reset() { self = State() }
        public var tensors: [Tensor] {
            [convHistory, recurrent].compactMap { $0 }
                + (verificationCapture.map { [$0.recurrentStates, $0.convInputs] } ?? [])
        }

        /// Commit a nonempty accepted prefix without re-running projections or
        /// recurrence. Full acceptance reuses the final state and drops capture.
        public func committingPrefix(count: Int) throws -> State {
            guard let capture = verificationCapture else {
                throw GPUAttentionError.invalid("Missing GDN verification capture")
            }
            let shape = capture.recurrentStates.shape
            guard shape.count == 5, (1...5).contains(shape[0]), shape[1...] == [1,48,128,128],
                  capture.recurrentStates.dtype == MLX_BFLOAT16,
                  capture.convInputs.shape == [1,shape[0]+3,10240], capture.convInputs.dtype == MLX_BFLOAT16,
                  capture.initialOffset >= 0, capture.initialOffset <= 262144 - shape[0],
                  offset == capture.initialOffset + shape[0], (1...shape[0]).contains(count),
                  recurrent?.shape == [1,48,128,128], recurrent?.dtype == MLX_BFLOAT16,
                  convHistory?.shape == [1,3,10240], convHistory?.dtype == MLX_BFLOAT16 else {
                throw GPUAttentionError.invalid("Invalid GDN verification capture or accepted prefix")
            }
            if count == shape[0] {
                var committed = self
                committed.verificationCapture = nil
                return committed
            }
            let row = try MX.slice(capture.recurrentStates, starts: [count-1,0,0,0,0],
                                   ends: [count,1,48,128,128])
            let recurrent = try GPUVerificationCopy.tensor(MX.reshape(row, [1,48,128,128]))
            let convolution = try MX.slice(capture.convInputs, starts: [0,count,0], ends: [1,count+3,10240])
            return try State(convHistory: GPUVerificationCopy.tensor(convolution), recurrent: recurrent,
                             offset: capture.initialOffset + count)
        }
    }

    public let layer: Int
    public let recurrence: Recurrence
    public let fusedPrework: Bool
    public let fuseDecodeProjections: Bool
    /// Additional resident BF16 copies only; original source-weight ledger is unchanged.
    public let additionalProjectionBufferBytes: UInt64
    private let decodeProjectionBuffers: GDNDecodeProjectionBuffers?
    private let preworkKernel: GPUGatedDeltaNetPrework?
    private let fusedKernel: GPUGatedDeltaNetFused?
    private let blockedKernel: GPUGatedDeltaNetBlocked?
    private let qkvWeight: Tensor
    private let zWeight: Tensor
    private let aWeight: Tensor
    private let bWeight: Tensor
    private let outWeight: Tensor
    private let convolution: Tensor
    private let dtBias: Tensor
    private let expA: Tensor
    private let norm: Tensor
    private let qScale: Tensor
    private let kScale: Tensor
    private let normOnes: Tensor

    public init(layer: Int, weights: GPUWeights, recurrence: Recurrence = .composed, fusedPrework: Bool = false,
                fuseDecodeProjections: Bool = false) throws {
        guard (0..<48).contains(layer), layer % 4 != 3 else {
            throw GPUAttentionError.invalid("Layer \(layer) is not a Qwen3.8 GDN layer")
        }
        self.layer = layer
        self.recurrence = recurrence
        self.fusedPrework = fusedPrework
        self.fuseDecodeProjections = fuseDecodeProjections
        fusedKernel = recurrence == .fused ? try GPUGatedDeltaNetFused() : nil
        blockedKernel = recurrence == .fused && ProcessInfo.processInfo.environment["ANERUNNER_BLOCKED_GDN"] == "1"
            ? try GPUGatedDeltaNetBlocked() : nil
        let prefix = "language_model.model.layers.\(layer).linear_attn."
        func load(_ suffix: String, _ shape: [Int]) throws -> Tensor {
            let tensor = try weights.tensor(prefix + suffix)
            guard tensor.shape == shape, tensor.dtype == MLX_BFLOAT16 else {
                throw GPUAttentionError.invalid("Unexpected GDN weight \(prefix + suffix), expected BF16 \(shape)")
            }
            return tensor
        }
        qkvWeight = try MX.transpose(load("in_proj_qkv.weight", [10240,2560]), [1,0])
        zWeight = try MX.transpose(load("in_proj_z.weight", [6144,2560]), [1,0])
        aWeight = try MX.transpose(load("in_proj_a.weight", [48,2560]), [1,0])
        bWeight = try MX.transpose(load("in_proj_b.weight", [48,2560]), [1,0])
        outWeight = try MX.transpose(load("out_proj.weight", [2560,6144]), [1,0])
        convolution = try load("conv1d.weight", [10240,4,1])
        dtBias = try load("dt_bias", [48])
        let rawALog = try load("A_log", [48])
        expA = try MX.exp(MX.cast(rawALog, MLX_FLOAT32))
        // Conversion already folds (1 + w); never add one again.
        norm = try load("norm.weight", [128])
        normOnes = try MX.ones([128], MLX_BFLOAT16)
        qScale = try MX.scalar(1.0 / 128.0, MLX_BFLOAT16)
        kScale = try MX.scalar(1.0 / sqrt(128.0), MLX_BFLOAT16)
        preworkKernel = fusedPrework ? try GPUGatedDeltaNetPrework(convolution: convolution,aLog: rawALog,dtBias: dtBias,normWeight: norm) : nil
        let packed = fuseDecodeProjections ? try GDNDecodeProjectionBuffers(qkv: qkvWeight,z: zWeight,a: aWeight,b: bWeight) : nil
        decodeProjectionBuffers = packed
        additionalProjectionBufferBytes = packed?.byteCount ?? 0
    }

    public func forward(_ x: Tensor, state: inout State, useFusedProjections: Bool? = nil,
                        verifyScalarBoundaries: Bool = false, captureVerification: Bool = false,
                        verificationLinear: GPUVerificationLinear? = nil) throws -> Tensor {
        let usePacked = try DecodeProjectionSelection.use(requested: useFusedProjections,prepared: fuseDecodeProjections)
        guard x.shape.count == 3, x.shape[0] == 1, x.shape[1] > 0,
              x.shape[2] == 2560, x.dtype == MLX_BFLOAT16,
              state.offset >= 0, state.offset <= 262144 - x.shape[1] else {
            throw GPUAttentionError.invalid("GDN requires BF16 [1,S,2560] within the 262144-position context")
        }
        guard (state.convHistory == nil) == (state.recurrent == nil),
              state.offset == 0 || state.recurrent != nil else {
            throw GPUAttentionError.invalid("Incomplete GDN state")
        }
        let sequence = x.shape[1]
        let roundStateEachToken = verifyScalarBoundaries && sequence <= 5
        guard !roundStateEachToken || fusedKernel != nil else {
            throw GPUAttentionError.invalid("GDN scalar-boundary verification requires the fused recurrence")
        }
        guard !captureVerification || (sequence <= 5 && fusedKernel != nil) else {
            throw GPUAttentionError.invalid("GDN verification capture requires the fused recurrence and S1...5")
        }
        let history = try state.convHistory ?? MX.zeros([1,3,10240], MLX_BFLOAT16)
        let storedState = try state.recurrent ?? MX.zeros([1,48,128,128], MLX_BFLOAT16)
        guard history.shape == [1,3,10240], history.dtype == MLX_BFLOAT16,
              storedState.shape == [1,48,128,128], storedState.dtype == MLX_BFLOAT16 else {
            throw GPUAttentionError.invalid("GDN state shape/dtype mismatch")
        }
        let qkv: Tensor, z: Tensor, a: Tensor, b: Tensor
        if usePacked, sequence == 1, let decodeProjectionBuffers {
            let projections = try decodeProjectionBuffers.project(x)
            qkv = projections.qkv
            z = try MX.reshape(projections.z,[1,1,48,128])
            a = projections.a; b = projections.b
        } else {
            qkv = try MX.linear(x, qkvWeight, verification: verificationLinear)
            z = try MX.reshape(MX.linear(x, zWeight, verification: verificationLinear), [1,sequence,48,128])
            a = try MX.linear(x, aWeight, verification: verificationLinear)
            b = try MX.linear(x, bWeight, verification: verificationLinear)
        }
        let captureInputs = captureVerification ? try MX.concat([history,qkv], axis: 1) : nil
        // Match the source's initialized-state and width eligibility. Cold
        // first-prefix work and longer chunks retain the composed prework.
        let usePrework = preworkKernel != nil && sequence <= 9 && state.convHistory != nil
        let q: Tensor, k: Tensor, v: Tensor, nextHistory: Tensor, decay: Tensor, beta: Tensor
        if usePrework, let preworkKernel {
            let result = try preworkKernel.apply(qkv: qkv,a: a,b: b,history: history)
            q = result.q; k = result.k; v = result.v
            nextHistory = result.history; decay = result.decay; beta = result.beta
        } else {
            let convInput = try captureInputs ?? MX.concat([history,qkv], axis: 1)
            // Keep the final three-token history. MX.copy may share its backing
            // buffer; verification prefix extraction uses GPUVerificationCopy.
            nextHistory = try MX.copy(MX.slice(convInput, starts: [0,sequence,0], ends: [1,sequence+3,10240]))
            let conv = try MX.silu(MX.conv1d(convInput, weight: convolution, groups: 10240))
            func part(_ start: Int, _ end: Int, _ heads: Int) throws -> Tensor {
                try MX.reshape(MX.slice(conv, starts: [0,0,start], ends: [1,sequence,end]), [1,sequence,heads,128])
            }
            q = try MX.mul(MX.rmsNorm(part(0,2048,16), weight: normOnes, epsilon: 1e-6), qScale)
            k = try MX.mul(MX.rmsNorm(part(2048,4096,16), weight: normOnes, epsilon: 1e-6), kScale)
            v = try part(4096,10240,48)
            // a + dt_bias rounds in BF16 before the FP32 softplus chain, and the
            // resulting decay and beta round to BF16 before recurrence arithmetic.
            let a32 = try MX.cast(MX.add(a,dtBias), MLX_FLOAT32)
            let softplus = try AttentionOps.log1p(MX.exp(a32))
            decay = try MX.cast(MX.exp(AttentionOps.negative(MX.mul(expA,softplus))), MLX_BFLOAT16)
            beta = try MX.sigmoid(b)
        }
        let y: Tensor, nextRecurrent: Tensor
        var capturedStates: Tensor?
        if captureVerification, let fusedKernel {
            let captured = try fusedKernel.applyCapturing(q: q,k: k,v: v,decay: decay,beta: beta,state: storedState,
                                                          roundStateEachToken: roundStateEachToken)
            y = captured.y; nextRecurrent = captured.state; capturedStates = captured.states
        } else if let blockedKernel, sequence >= 64 {
            (y,nextRecurrent) = try blockedKernel.apply(q: q,k: k,v: v,decay: decay,beta: beta,state: storedState)
        } else if let fusedKernel {
            (y,nextRecurrent) = try fusedKernel.apply(q: q,k: k,v: v,decay: decay,beta: beta,state: storedState,
                                                    roundStateEachToken: roundStateEachToken)
        } else {
            // Key heads are repeated contiguously: hv -> floor(hv / 3).
            func repeatKeys(_ tensor: Tensor) throws -> Tensor {
                let grouped = try MX.reshape(tensor, [1,sequence,16,1,128])
                return try MX.reshape(AttentionOps.broadcast(grouped, [1,sequence,16,3,128]), [1,sequence,48,128])
            }
            let q32 = try MX.cast(repeatKeys(q), MLX_FLOAT32)
            let k32 = try MX.cast(repeatKeys(k), MLX_FLOAT32)
            let v32 = try MX.cast(v, MLX_FLOAT32)
            let decay32 = try MX.cast(decay, MLX_FLOAT32)
            let beta32 = try MX.cast(beta, MLX_FLOAT32)
            var accumulator = try MX.cast(storedState, MLX_FLOAT32)
            var outputs = [Tensor]()
            outputs.reserveCapacity(sequence)
            for position in 0..<sequence {
                func vector(_ tensor: Tensor) throws -> Tensor {
                    try MX.reshape(MX.slice(tensor, starts: [0,position,0,0], ends: [1,position+1,48,128]), [1,48,128])
                }
                func gate(_ tensor: Tensor) throws -> Tensor {
                    try MX.reshape(MX.slice(tensor, starts: [0,position,0], ends: [1,position+1,48]), [1,48,1])
                }
                let qt = try MX.reshape(vector(q32), [1,48,1,128])
                let kt = try MX.reshape(vector(k32), [1,48,1,128])
                let vt = try vector(v32)
                let gt = try MX.reshape(gate(decay32), [1,48,1,1])
                let bt = try gate(beta32)
                // S' = g*S; delta = beta*(v - S'k); S = S' + delta outer k;
                // y = Sq. State layout is [batch,valueHead,valueDim,keyDim].
                let decayed = try MX.mul(accumulator,gt)
                let memory = try MX.sum(MX.mul(decayed,kt), axis: -1, keepDims: false)
                let delta = try MX.mul(MX.sub(vt,memory),bt)
                accumulator = try MX.add(decayed,MX.mul(MX.reshape(delta,[1,48,128,1]),kt))
                let y = try MX.cast(MX.sum(MX.mul(accumulator,qt), axis: -1, keepDims: false), MLX_BFLOAT16)
                outputs.append(try MX.reshape(y,[1,1,48,128]))
            }
            y = try MX.concat(outputs,axis: 1)
            nextRecurrent = try MX.cast(accumulator,MLX_BFLOAT16)
        }
        let outInput: Tensor
        if usePrework, let preworkKernel {
            outInput = try preworkKernel.normGate(y: y,z: MX.reshape(z,[1,sequence,6144]))
        } else {
            let gated = try MX.mul(MX.rmsNorm(y,weight: norm,epsilon: 1e-6),MX.sigmoid(z))
            outInput = try MX.reshape(gated,[1,sequence,6144])
        }
        let result = try MX.linear(outInput,outWeight, verification: verificationLinear)
        let nextCapture: VerificationCapture?
        if let capturedStates, let captureInputs {
            nextCapture = VerificationCapture(initialOffset: state.offset, recurrentStates: capturedStates, convInputs: captureInputs)
        } else { nextCapture = nil }
        // Commit only after successfully constructing the entire result graph.
        state.convHistory = nextHistory
        state.recurrent = nextRecurrent
        state.verificationCapture = nextCapture
        state.offset += sequence
        return result
    }
}

/// Only schedules existing BF16 row dot products differently. In the pinned
/// MLX GEMV selector, QKV/Z/16384 rows all use BM8/BN1; A/B/96 rows all use
/// BM1/BN8. Combining all four would change the gate reduction topology.
/// Original matrices remain retained for prefill and same-residency A/B tests.
struct GDNDecodeProjectionBuffers {
    struct Output { let qkv, z, a, b: Tensor }
    private let qkvz, ab: Tensor
    let byteCount: UInt64

    /// Inputs have the transposed matmul view layout [input, output]. Packing
    /// is performed once on the original contiguous row layout, never per token.
    init(qkv: Tensor, z: Tensor, a: Tensor, b: Tensor) throws {
        guard qkv.shape == [2560,10240], z.shape == [2560,6144],
              a.shape == [2560,48], b.shape == [2560,48],
              [qkv,z,a,b].allSatisfy({ $0.dtype == MLX_BFLOAT16 }) else {
            throw GPUError.invalid("GDN decode projection packing requires the original BF16 matrix shapes")
        }
        let qkvzRows = try MX.concat([MX.transpose(qkv,[1,0]),MX.transpose(z,[1,0])],axis: 0)
        let abRows = try MX.concat([MX.transpose(a,[1,0]),MX.transpose(b,[1,0])],axis: 0)
        qkvz = try MX.transpose(qkvzRows,[1,0])
        ab = try MX.transpose(abRows,[1,0])
        // Charge materialization to loading, not the first selected decode step.
        try MX.eval([qkvz,ab])
        byteCount = UInt64(qkvz.nbytes) + UInt64(ab.nbytes)
    }

    func project(_ x: Tensor) throws -> Output {
        guard x.shape == [1,1,2560], x.dtype == MLX_BFLOAT16 else {
            throw GPUError.invalid("Packed GDN projections support BF16 single-token decode only")
        }
        let qkvzOutput = try MX.matmul(x,qkvz)
        let abOutput = try MX.matmul(x,ab)
        return try Output(
            qkv: MX.slice(qkvzOutput,starts: [0,0,0],ends: [1,1,10240]),
            z: MX.slice(qkvzOutput,starts: [0,0,10240],ends: [1,1,16384]),
            a: MX.slice(abOutput,starts: [0,0,0],ends: [1,1,48]),
            b: MX.slice(abOutput,starts: [0,0,48],ends: [1,1,96]))
    }
}

/// A prepared model can switch reference/candidate paths without reloading.
/// Enabling a candidate on an unprepared model is an explicit usage error.
enum DecodeProjectionSelection {
    static func use(requested: Bool?, prepared: Bool) throws -> Bool {
        let enabled = requested ?? prepared
        guard !enabled || prepared else {
            throw GPUError.invalid("Fused decode projections were not prepared; initialize with fuseDecodeProjections: true")
        }
        return enabled
    }
}
