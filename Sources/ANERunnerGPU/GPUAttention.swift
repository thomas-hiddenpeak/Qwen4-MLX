// Adapted from garnermccloud/mlx-serve src/transformer.zig, fixed commit
// 7dbcba04c98e4fd3bcc533c63e645547f13cc3b1, gatedFullAttnWith/qsaMaskFromQk.
// Text-only Swift/MLX C adaptation; no M-RoPE image-position implementation.
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

public enum GPUAttentionError: Error, CustomStringConvertible {
    case invalid(String)
    public var description: String {
        switch self { case .invalid(let message): return message }
    }
}

/// Qwen3.8 full-attention layers: per-channel query gate, QK RMSNorm, 64/256
/// partial RoPE, GQA, and the actual QSA selection mask after its budget.
/// KV storage remains complete; sparse visibility does not imply that stock
/// MLX SDPA physically avoids reading every unselected KV block.
public final class GPUAttention {
    public enum PrefillMode: String, Codable, CaseIterable, Sendable {
        /// Existing causal-fused / QSA-unfused routing.
        case reference
        /// Explicit experiment: force the existing Metal SDPA for QSA chunks.
        /// This can change softmax/projection rounding and requires validation.
        case fusedQSA

        func validate(phase: QwenExecutionPhase) throws {
            guard self == .reference || phase == .prefill else {
                throw GPUAttentionError.invalid("Prefill attention policy cannot be applied to decode or verification")
            }
        }
    }
    public struct State {
        public var keys: Tensor?
        public var values: Tensor?
        public var rawIndexerKeys: Tensor?
        public var pooledIndexerKeys: Tensor?
        public var offset: Int
        // Extents of runner-owned allocations, not only current view shapes.
        // Caller-supplied aliases may already retain storage beyond these sizes.
        fileprivate var retainedRowCount: Int
        fileprivate var retainedPooledBlockCount: Int
        /// Diagnostic allocation bookkeeping; values, not buffer identities.
        public var diagnosticRetainedStorage: [Int] { [retainedRowCount, retainedPooledBlockCount] }

        public init(keys: Tensor? = nil, values: Tensor? = nil,
                    rawIndexerKeys: Tensor? = nil, pooledIndexerKeys: Tensor? = nil,
                    offset: Int = 0) {
            self.keys = keys
            self.values = values
            self.rawIndexerKeys = rawIndexerKeys
            self.pooledIndexerKeys = pooledIndexerKeys
            self.offset = offset
            retainedRowCount = offset
            retainedPooledBlockCount = pooledIndexerKeys.map { $0.shape.count > 1 ? $0.shape[1] : 0 } ?? 0
        }

        public mutating func reset() { self = State() }
        public var tensors: [Tensor] {
            [keys,values,rawIndexerKeys,pooledIndexerKeys].compactMap { $0 }
        }
    }

    public let layer: Int
    public static let indexerBudget = 2048
    public static let indexerCompressionRatio = 4
    private let fusedPrefill: Bool
    /// The tested M5 prefill path is the default; 0 restores explicit-mask SDPA.
    public static var fusedPrefillEnabled: Bool {
        (ProcessInfo.processInfo.environment["ANERUNNER_FUSED_PREFILL"] ?? "1") == "1"
    }
    private let qWeight: Tensor
    private let kWeight: Tensor
    private let vWeight: Tensor
    private let outWeight: Tensor
    private let qNorm: Tensor
    private let kNorm: Tensor
    private let indexerWeight: Tensor
    private let indexerQNorm: Tensor
    private let indexerKNorm: Tensor

    public init(layer: Int, weights: GPUWeights, weightPrefix: String? = nil) throws {
        guard (0..<48).contains(layer), weightPrefix != nil || layer % 4 == 3 else {
            throw GPUAttentionError.invalid("Layer \(layer) is not a Qwen3.8 full-attention layer")
        }
        self.layer = layer
        fusedPrefill = Self.fusedPrefillEnabled
        let prefix = (weightPrefix ?? "language_model.model.layers.\(layer).self_attn") + "."
        func load(_ suffix: String, _ shape: [Int]) throws -> Tensor {
            let tensor = try weights.tensor(prefix + suffix)
            guard tensor.shape == shape, tensor.dtype == MLX_BFLOAT16 else {
                throw GPUAttentionError.invalid("Unexpected attention weight \(prefix + suffix), expected BF16 \(shape)")
            }
            return tensor
        }
        qWeight = try MX.transpose(load("q_proj.weight",[12288,2560]),[1,0])
        kWeight = try MX.transpose(load("k_proj.weight",[512,2560]),[1,0])
        vWeight = try MX.transpose(load("v_proj.weight",[512,2560]),[1,0])
        outWeight = try MX.transpose(load("o_proj.weight",[2560,6144]),[1,0])
        // The converter has already folded norm offsets into these weights.
        qNorm = try load("q_norm.weight",[256])
        kNorm = try load("k_norm.weight",[256])
        indexerWeight = try MX.transpose(load("indexer.index_qk_proj.weight",[640,2560]),[1,0])
        indexerQNorm = try load("indexer.q_layernorm.weight",[128])
        indexerKNorm = try load("indexer.k_layernorm.weight",[128])
    }

    /// `positionBase` is the absolute position of KV row zero. The trunk uses
    /// zero; MTP starts at one while retaining a separate, zero-based row count.
    /// A detail profiler must replace, not nest inside, an outer attention
    /// stage. Its synchronized mode materializes every stage's live outputs.
    public func forward(_ x: Tensor, state: inout State, positionBase: Int = 0,
                        verificationLinear: GPUVerificationLinear? = nil,
                        prefillMode: PrefillMode = .reference,
                        profiler: GPUProfiler? = nil) throws -> Tensor {
        guard x.shape.count == 3, x.shape[0] == 1, x.shape[1] > 0,
              x.shape[2] == 2560, x.dtype == MLX_BFLOAT16,
              state.offset >= 0, state.offset <= 262144 - x.shape[1],
              positionBase >= 0, positionBase <= 262144 - state.offset - x.shape[1] else {
            throw GPUAttentionError.invalid("Attention requires text BF16 [1,S,2560] within the 262144-position context")
        }
        guard prefillMode == .reference || verificationLinear == nil else {
            throw GPUAttentionError.invalid("QSA prefill fusion cannot select verification kernels")
        }
        try validate(state)
        let sequence = x.shape[1], offset = state.offset
        let ropeOffset = positionBase + offset
        let qkv: (queries: Tensor,gate: Tensor,newKeys: Tensor,newValues: Tensor) = try measure(
                              "attention.qkv_projection",profiler: profiler,sequence: sequence,
                              outputs: { [$0.queries,$0.gate,$0.newKeys,$0.newValues] }) {
            let queryGate = try MX.reshape(MX.linear(x,qWeight, verification: verificationLinear),[1,sequence,24,512])
            let query = try MX.slice(queryGate,starts: [0,0,0,0],ends: [1,sequence,24,256])
            let gate = try MX.slice(queryGate,starts: [0,0,0,256],ends: [1,sequence,24,512])
            let key = try MX.reshape(MX.linear(x,kWeight, verification: verificationLinear),[1,sequence,2,256])
            let value = try MX.reshape(MX.linear(x,vWeight, verification: verificationLinear),[1,sequence,2,256])
            let queries = try MX.rope(MX.transpose(MX.rmsNorm(query,weight: qNorm,epsilon: 1e-6),[0,2,1,3]),
                                      dimensions: 64,base: 10_000_000,offset: ropeOffset)
            let newKeys = try MX.rope(MX.transpose(MX.rmsNorm(key,weight: kNorm,epsilon: 1e-6),[0,2,1,3]),
                                      dimensions: 64,base: 10_000_000,offset: ropeOffset)
            let newValues = try MX.transpose(value,[0,2,1,3])
            return (queries: queries,gate: gate,newKeys: newKeys,newValues: newValues)
        }
        let queries = qkv.queries, gate = qkv.gate
        let kv: (keys: Tensor,values: Tensor) = try measure("attention.kv_append",profiler: profiler,sequence: sequence,
                             outputs: { [$0.keys,$0.values] }) {
            let keys = try state.keys.map { try MX.concat([$0,qkv.newKeys],axis: 2) } ?? qkv.newKeys
            let values = try state.values.map { try MX.concat([$0,qkv.newValues],axis: 2) } ?? qkv.newValues
            return (keys: keys,values: values)
        }
        let keys = kv.keys, values = kv.values
        var next = state
        let projectedIndex = try measure("attention.index_projection",profiler: profiler,sequence: sequence,
                                        outputs: { [$0] }) {
            try MX.linear(x,indexerWeight, verification: verificationLinear)
        }
        let sparseMask = try qsaMask(projectedIndex,state: &next,sequence: sequence,
                                     positionBase: positionBase,profiler: profiler)
        let attention = try measure("attention.sdpa",profiler: profiler,sequence: sequence,outputs: { [$0] }) {
            if prefillMode == .fusedQSA, let sparseMask, sequence > 8 {
                // Preserve QSA's exact boolean visibility, including causal tails.
                // The pinned MLX has a fused D256 kernel with an array mask, but
                // its default performance heuristic selects the unfused path.
                return try MX.sdpa(queries,keys,values,scale: 1.0 / 16.0,
                                   mask: sparseMask,forceFused: true)
            } else if fusedPrefill, sparseMask == nil, sequence > 8, queries.shape[3] == 256 {
                // MLX aligns causal queries at kL-qL, matching the existing KV prefix.
                return try MX.sdpa(queries,keys,values,scale: 1.0 / 16.0,causal: true,forceFused: true)
            } else {
                let mask: Tensor?
                if let sparseMask { mask = sparseMask }
                else if sequence > 1 { mask = try AttentionOps.causalMask(offset: offset,sequence: sequence) }
                else { mask = nil }
                if verificationLinear != nil, sequence > 2 {
                    // GQA=12 and head_dim=256: pinned MLX vector SDPA requires
                    // qL*GQA<=32. Match the author's small-query split so depth>=2
                    // verification retains the decode kernel and its reduction.
                    var pieces: [Tensor] = []
                    for start in stride(from: 0, to: sequence, by: 2) {
                        let end = min(start + 2, sequence)
                        let q = try MX.slice(queries, starts: [0,0,start,0], ends: [1,24,end,256])
                        let m = try mask.map { try MX.slice($0, starts: [0,0,start,0], ends: [1,1,end,offset+sequence]) }
                        pieces.append(try MX.sdpa(q,keys,values,scale: 1.0 / 16.0,mask: m))
                    }
                    return try MX.concat(pieces, axis: 2)
                } else {
                    return try MX.sdpa(queries,keys,values,scale: 1.0 / 16.0,mask: mask)
                }
            }
        }
        let result = try measure("attention.output",profiler: profiler,sequence: sequence,outputs: { [$0] }) {
            let heads = try MX.transpose(attention,[0,2,1,3])
            let gated = try MX.mul(heads,MX.sigmoid(gate))
            return try MX.linear(MX.reshape(gated,[1,sequence,6144]),outWeight, verification: verificationLinear)
        }
        next.keys = keys
        next.values = values
        next.offset = offset + sequence
        next.retainedRowCount = next.offset // KV and raw indexer concat allocate anew.
        state = next
        return result
    }

    /// Retain the first `count` KV rows (a total row count, not a number of
    /// newly accepted tokens). MTP positionBase does not change this row count.
    /// Small suffixes retain views: at most four KV/raw rows and one pooled
    /// block beyond the logical prefix in runner-owned buffers. Extents carry
    /// through repeated trims, so repeated small cuts cannot accumulate a large
    /// retained tail. Larger cuts allocate copies. The next KV/raw concat copies
    /// only logical rows; pooled storage refreshes when a complete block appends.
    /// Fresh verification concat buffers retain at most 9,472 extra bytes per
    /// layer (four BF16 K/V+raw rows, one pooled row). Pre-existing projection
    /// or caller-supplied aliases can also retain their original column padding.
    /// Joint evaluation detaches old graph inputs in ordinary non-traced MLX
    /// execution; the view retains its one backing buffer, not the entire graph.
    public func prefixState(_ state: State, count: Int) throws -> State {
        guard count >= 0, count <= state.offset else {
            throw GPUAttentionError.invalid("Attention prefix count is outside the existing cache")
        }
        try validate(state)
        if count == state.offset { return state }
        if count == 0 { return State() }
        guard let keys = state.keys, let values = state.values, let raw = state.rawIndexerKeys else {
            throw GPUAttentionError.invalid("Missing attention state for prefix commit")
        }
        let prefixKeys = try Self.retainedPrefix(keys, axis: 2, count: count, extent: state.retainedRowCount, maximumTail: 4)
        let prefixValues = try Self.retainedPrefix(values, axis: 2, count: count, extent: state.retainedRowCount, maximumTail: 4)
        let prefixRaw = try Self.retainedPrefix(raw, axis: 1, count: count, extent: state.retainedRowCount, maximumTail: 4)
        var prefixPooled: Tensor?
        var retainedBlocks = 0
        // QSA begins only after 2051 rows. A rejected suffix may have crossed
        // that boundary, so restore nil exactly for a shorter accepted prefix.
        if count > 2051, let pooled = state.pooledIndexerKeys {
            let blocks = min(pooled.shape[1], count / Self.indexerCompressionRatio)
            if blocks > 0 {
                let prefix = try Self.retainedPrefix(pooled, axis: 1, count: blocks,
                    extent: state.retainedPooledBlockCount, maximumTail: state.retainedRowCount - count <= 4 ? 1 : 0)
                prefixPooled = prefix.tensor; retainedBlocks = prefix.extent
            }
        }
        var result = State(keys: prefixKeys.tensor, values: prefixValues.tensor, rawIndexerKeys: prefixRaw.tensor,
                           pooledIndexerKeys: prefixPooled, offset: count)
        result.retainedRowCount = prefixKeys.extent
        result.retainedPooledBlockCount = retainedBlocks
        return result
    }

    /// Shared by prefixState and its small real-SDPA test; no model load needed.
    static func retainedPrefix(_ input: Tensor, axis: Int, count: Int, extent: Int,
                               maximumTail: Int) throws -> (tensor: Tensor, extent: Int) {
        guard axis >= 0, axis < input.shape.count, count > 0, count <= input.shape[axis],
              extent >= input.shape[axis], maximumTail >= 0 else {
            throw GPUAttentionError.invalid("Invalid attention prefix retention extent")
        }
        var ends = input.shape
        ends[axis] = count
        let view = try MX.slice(input, starts: [Int](repeating: 0, count: ends.count), ends: ends)
        if extent - count <= maximumTail { return (view, extent) }
        return (try GPUVerificationCopy.tensor(view), count)
    }

    private func validate(_ state: State) throws {
        if state.offset == 0 {
            guard state.keys == nil, state.values == nil, state.rawIndexerKeys == nil,
                  state.pooledIndexerKeys == nil else {
                throw GPUAttentionError.invalid("Zero-offset attention state must be empty; use reset()")
            }
            return
        }
        guard let keys = state.keys, let values = state.values, let raw = state.rawIndexerKeys,
              keys.shape == [1,2,state.offset,256], values.shape == keys.shape,
              keys.dtype == MLX_BFLOAT16, values.dtype == MLX_BFLOAT16,
              raw.shape == [1,state.offset,128], raw.dtype == MLX_BFLOAT16 else {
            throw GPUAttentionError.invalid("Attention KV/indexer state shape or dtype mismatch")
        }
        if let pooled = state.pooledIndexerKeys {
            guard pooled.shape.count == 3, pooled.shape[0] == 1,
                  pooled.shape[1] <= state.offset / 4, pooled.shape[1] > 0,
                  pooled.shape[2] == 128, pooled.dtype == MLX_BFLOAT16 else {
                throw GPUAttentionError.invalid("Invalid QSA pooled key state")
            }
        }
    }

    /// No profiling work (including output collection) is performed by the
    /// normal nil path. Callers are responsible for avoiding an outer stage.
    private func measure<T>(_ name: String, profiler: GPUProfiler?, sequence: Int,
                            outputs: (T) -> [Tensor], _ body: () throws -> T) throws -> T {
        guard let profiler else { return try body() }
        return try profiler.measure(name,layer: layer,tokenCount: sequence,outputs: outputs,body)
    }

    private func qsaMask(_ qk: Tensor, state: inout State, sequence: Int, positionBase: Int,
                         profiler: GPUProfiler?) throws -> Tensor? {
        let offset = state.offset, total = offset + sequence, ratio = 4, blockTopK = 512
        let blocks = total / ratio
        let historyPool: (history: Tensor,pooled: Tensor?,queryRope: Tensor?,keyTranspose: Tensor?) = try measure(
            "qsa.history_pool",profiler: profiler,sequence: sequence,
            outputs: { value in
                [value.history] + [value.pooled,value.queryRope,value.keyTranspose].compactMap { $0 }
            }) {
                let raw = try MX.slice(qk,starts: [0,0,512],ends: [1,sequence,640])
                let history = try state.rawIndexerKeys.map { try MX.concat([$0,raw],axis: 1) } ?? raw
                state.rawIndexerKeys = history
                // Up through 2051 tokens, every complete block fits the 2048-token
                // budget and the incomplete tail contains at most three more tokens.
                guard total > 2051 else { return (history,state.pooledIndexerKeys,nil,nil) }
                let query = try MX.reshape(MX.slice(qk,starts: [0,0,0],ends: [1,sequence,512]),[1,sequence,4,128])
                let queryNorm = try MX.rmsNorm(query,weight: indexerQNorm,epsilon: 1e-6)
                let queryRope = try MX.rope(MX.transpose(queryNorm,[0,2,1,3]),dimensions: 64,base: 10_000_000,offset: positionBase + offset)
                let cachedBlocks = state.pooledIndexerKeys?.shape[1] ?? 0
                if cachedBlocks < blocks {
                    let count = blocks - cachedBlocks
                    let flat = try MX.slice(history,starts: [0,cachedBlocks*ratio,0],ends: [1,blocks*ratio,128])
                    let grouped = try MX.reshape(flat,[1,count,ratio,128])
                    // Pool in FP32, then BF16 before RMSNorm/RoPE, as in the source.
                    let pooled = try MX.cast(MX.mean(MX.cast(grouped,MLX_FLOAT32),axis: 2,keepDims: false),MLX_BFLOAT16)
                    let normed = try MX.rmsNorm(pooled,weight: indexerKNorm,epsilon: 1e-6)
                    let roped = try AttentionOps.ropeAtPositions(MX.reshape(normed,[1,1,count,128]),
                                                                dimensions: 64,base: Float(positionBase + cachedBlocks*ratio),step: Float(ratio))
                    let newPooled = try MX.reshape(roped,[1,count,128])
                    state.pooledIndexerKeys = try state.pooledIndexerKeys.map { try MX.concat([$0,newPooled],axis: 1) } ?? newPooled
                    state.retainedPooledBlockCount = blocks
                }
                guard let pooled = state.pooledIndexerKeys else {
                    throw GPUAttentionError.invalid("Missing QSA pooled keys")
                }
                let keyRope = try MX.reshape(pooled,[1,1,blocks,128])
                let keyTranspose = try MX.transpose(keyRope,[0,1,3,2])
                return (history,pooled,queryRope,keyTranspose)
            }
        guard total > 2051 else { return nil }
        guard let queryRope = historyPool.queryRope, let keyTranspose = historyPool.keyTranspose else {
            throw GPUAttentionError.invalid("Missing QSA score inputs")
        }
        let scores = try measure("qsa.score",profiler: profiler,sequence: sequence,outputs: { [$0] }) {
            let products = try MX.matmul(MX.cast(queryRope,MLX_FLOAT32),MX.cast(keyTranspose,MLX_FLOAT32))
            let relu = try AttentionOps.maximum(products,MX.scalar(0))
            return try MX.sum(relu,axis: 1,keepDims: false)
        }
        return try measure("qsa.select_mask",profiler: profiler,sequence: sequence,outputs: { [$0] }) {
            // The positive 1/sqrt(128) factor is omitted by the original selector.
            let positions = try MX.reshape(AttentionOps.arange(offset,offset+sequence),[sequence,1])
            let blockEnds = try MX.reshape(AttentionOps.arange(3,blocks*ratio,step: ratio),[1,blocks])
            let visible = try MX.reshape(AttentionOps.lessEqual(blockEnds,positions),[1,sequence,blocks])
            let indices = try AttentionOps.arange(0,blocks,dtype: MLX_FLOAT32)
            let bias = try MX.mul(indices,MX.scalar(1e-7))
            let biased = try MX.sub(scores,bias)
            let masked = try AttentionOps.select(visible,biased,MX.scalar(-Float.infinity))
            let partition = try MX.output("QSA argpartition") {
                mlx_argpartition_axis(&$0,masked.handle,Int32(blocks-blockTopK),-1,MX.stream)
            }
            let selectedIndices = try MX.slice(partition,starts: [0,0,blocks-blockTopK],ends: [1,sequence,blocks])
            let unselected = try MX.zeros([1,sequence,blocks],MLX_BOOL)
            let truth = try MX.scalar(1,MLX_BOOL)
            let picked = try MX.output("QSA block selection") {
                mlx_put_along_axis(&$0,unselected.handle,selectedIndices.handle,truth.handle,-1,MX.stream)
            }
            let selected = try AttentionOps.and(picked,visible)
            let expanded = try AttentionOps.broadcast(MX.reshape(selected,[1,sequence,blocks,1]),[1,sequence,blocks,ratio])
            let selectedTokens = try MX.reshape(expanded,[1,sequence,blocks*ratio])
            let fullSelection: Tensor
            if total % ratio != 0 {
                fullSelection = try MX.concat([selectedTokens,MX.zeros([1,sequence,total%ratio],MLX_BOOL)],axis: 2)
            } else { fullSelection = selectedTokens }
            let keyIndices = try MX.reshape(AttentionOps.arange(0,total),[1,total])
            let one = try MX.scalar(1,MLX_INT32)
            let four = try MX.scalar(4,MLX_INT32)
            let nextPosition = try MX.add(positions,one)
            let completeBlockCount = try MX.output("QSA tail division") {
                mlx_floor_divide(&$0,nextPosition.handle,four.handle,MX.stream)
            }
            let tailStart = try MX.mul(completeBlockCount,four)
            let tail = try AttentionOps.greaterEqual(keyIndices,tailStart)
            let selectedOrTail = try AttentionOps.or(fullSelection,tail)
            let causal = try AttentionOps.lessEqual(keyIndices,positions)
            return try MX.reshape(AttentionOps.and(selectedOrTail,causal),[1,1,sequence,total])
        }
    }
}

/// MLX 0.32.2 Copy::eval shares its input buffer; Contiguous also permits up to
/// 16 KiB of excess storage. A batch-axis gather instead allocates out.nbytes()
/// in backend/metal/indexing.cpp::Gather::eval_gpu and copies values unchanged.
/// The lazy output must be evaluated before its source dependency is released.
enum GPUVerificationCopy {
    static func tensor(_ input: Tensor) throws -> Tensor {
        guard input.shape.first == 1 else {
            throw GPUAttentionError.invalid("Verification state copy requires batch size one")
        }
        let batch = try MX.array([Int32(0)], shape: [1])
        return try MX.take(input, batch, axis: 0)
    }
}

/// Small MLX C operations local to the attention implementation. All tensor
/// computation remains on MX.stream; no router/selection readback to the host.
enum AttentionOps {
    static func arange(_ start: Int, _ end: Int, step: Int = 1, dtype: mlx_dtype = MLX_INT32) throws -> Tensor {
        try MX.output("attention arange") { mlx_arange(&$0,Double(start),Double(end),Double(step),dtype,MX.stream) }
    }
    static func broadcast(_ x: Tensor, _ shape: [Int]) throws -> Tensor {
        let dimensions = shape.map(Int32.init)
        return try MX.output("attention broadcast") { output in
            dimensions.withUnsafeBufferPointer {
                mlx_broadcast_to(&output,x.handle,$0.baseAddress,$0.count,MX.stream)
            }
        }
    }
    static func log1p(_ x: Tensor) throws -> Tensor {
        try MX.output("attention log1p") { mlx_log1p(&$0,x.handle,MX.stream) }
    }
    static func negative(_ x: Tensor) throws -> Tensor {
        try MX.output("attention negative") { mlx_negative(&$0,x.handle,MX.stream) }
    }
    static func maximum(_ x: Tensor, _ y: Tensor) throws -> Tensor {
        try MX.output("attention maximum") { mlx_maximum(&$0,x.handle,y.handle,MX.stream) }
    }
    static func lessEqual(_ x: Tensor, _ y: Tensor) throws -> Tensor {
        try MX.output("attention <=") { mlx_less_equal(&$0,x.handle,y.handle,MX.stream) }
    }
    static func greaterEqual(_ x: Tensor, _ y: Tensor) throws -> Tensor {
        try MX.output("attention >=") { mlx_greater_equal(&$0,x.handle,y.handle,MX.stream) }
    }
    static func and(_ x: Tensor, _ y: Tensor) throws -> Tensor {
        try MX.output("attention logical and") { mlx_logical_and(&$0,x.handle,y.handle,MX.stream) }
    }
    static func or(_ x: Tensor, _ y: Tensor) throws -> Tensor {
        try MX.output("attention logical or") { mlx_logical_or(&$0,x.handle,y.handle,MX.stream) }
    }
    static func select(_ condition: Tensor, _ x: Tensor, _ y: Tensor) throws -> Tensor {
        try MX.output("attention where") { mlx_where(&$0,condition.handle,x.handle,y.handle,MX.stream) }
    }
    static func causalMask(offset: Int, sequence: Int) throws -> Tensor {
        let queries = try MX.reshape(arange(offset,offset+sequence),[sequence,1])
        let keys = try MX.reshape(arange(0,offset+sequence),[1,offset+sequence])
        return try MX.reshape(lessEqual(keys,queries),[1,1,sequence,offset+sequence])
    }

    /// Source ropeAtPositions uses BF16 cos/sin and separate BF16 products for
    /// pooled keys, unlike substituting scaled positions into fast_rope.
    static func ropeAtPositions(_ x: Tensor, dimensions: Int, base: Float, step: Float) throws -> Tensor {
        let sequence = x.shape[2], half = dimensions / 2
        let indices = try arange(0,half,dtype: MLX_FLOAT32)
        let coefficient = try MX.scalar(-2 * log(Float(10_000_000)) / Float(dimensions))
        let frequencies = try MX.exp(MX.mul(indices,coefficient))
        let positions = try MX.output("pooled RoPE positions") {
            mlx_arange(&$0,Double(base),Double(base + step * Float(sequence) - step * 0.5),Double(step),MLX_FLOAT32,MX.stream)
        }
        let angles = try MX.mul(MX.reshape(positions,[sequence,1]),frequencies)
        let doubled = try MX.concat([angles,angles],axis: 1)
        let cos32 = try MX.output("pooled RoPE cos") { mlx_cos(&$0,doubled.handle,MX.stream) }
        let sin32 = try MX.output("pooled RoPE sin") { mlx_sin(&$0,doubled.handle,MX.stream) }
        let cos = try MX.cast(cos32,x.dtype), sin = try MX.cast(sin32,x.dtype)
        let rotary = try MX.slice(x,starts: [0,0,0,0],ends: [1,1,sequence,dimensions])
        let pass = try MX.slice(x,starts: [0,0,0,dimensions],ends: [1,1,sequence,x.shape[3]])
        let first = try MX.slice(rotary,starts: [0,0,0,0],ends: [1,1,sequence,half])
        let second = try MX.slice(rotary,starts: [0,0,0,half],ends: [1,1,sequence,dimensions])
        let rotation = try MX.concat([negative(second),first],axis: -1)
        let result = try MX.add(MX.mul(rotary,cos),MX.mul(rotation,sin))
        return try MX.concat([result,pass],axis: -1)
    }
}
