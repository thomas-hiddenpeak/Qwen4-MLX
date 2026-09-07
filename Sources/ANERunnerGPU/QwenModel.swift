import ANERunnerCore
import CMLX
import Dispatch
import Foundation

/// Independent Swift text model. MLX supplies device array primitives; no
/// author server, Python process, or Core ML conversion is used to infer.
public final class QwenModel {
    public struct State {
        public fileprivate(set) var offset = 0
        public fileprivate(set) var valid = true
        fileprivate let sessionIdentity = UUID()
        fileprivate let owner: UUID
        fileprivate var gdn: [GPUGatedDeltaNet.State]
        fileprivate var attention: [GPUAttention.State]
        fileprivate var ple: [GPUPLE.State]
        fileprivate init(layers: Int, owner: UUID) {
            self.owner = owner
            gdn = (0..<layers).map { _ in .init() }; attention = (0..<layers).map { _ in .init() }
            ple = (0..<layers).map { _ in .init() }
        }
        public mutating func reset() { self = State(layers: gdn.count, owner: owner) }
        /// Diagnostic host values excluded from namedTensors. No tensor handles
        /// or mutable session identity cross this snapshot.
        public struct DiagnosticHostValues: Codable, Equatable {
            public let offset: Int
            public let valid: Bool
            public let gdnOffsets, attentionOffsets: [Int]
            public let attentionRetainedStorage: [[Int]]
            public let pleHistory: [[UInt32]]
            public let gdnCapturePresent, pleCapturePresent: [Bool]
        }
        public var diagnosticHostValues: DiagnosticHostValues {
            DiagnosticHostValues(offset: offset, valid: valid,
                gdnOffsets: gdn.map(\.offset), attentionOffsets: attention.map(\.offset),
                attentionRetainedStorage: attention.map(\.diagnosticRetainedStorage),
                pleHistory: ple.map(\.history),
                gdnCapturePresent: gdn.map { $0.verificationCapture != nil },
                pleCapturePresent: ple.map { $0.verificationCapture != nil })
        }
        public var tensors: [Tensor] { gdn.flatMap(\.tensors) + attention.flatMap(\.tensors) + ple.flatMap(\.tensors) }
        public var qsaActiveLayers: Int { attention.filter { $0.pooledIndexerKeys != nil }.count }
        /// Diagnostic names only; graph handles remain read-only to callers.
        public var namedTensors: [String: Tensor] {
            var values: [String: Tensor] = [:]
            for i in gdn.indices {
                values["layer.\(i).gdn.recurrent"] = gdn[i].recurrent
                values["layer.\(i).gdn.conv"] = gdn[i].convHistory
                values["layer.\(i).attention.keys"] = attention[i].keys
                values["layer.\(i).attention.values"] = attention[i].values
                values["layer.\(i).attention.raw_index"] = attention[i].rawIndexerKeys
                values["layer.\(i).attention.pooled_index"] = attention[i].pooledIndexerKeys
                values["layer.\(i).ple.conv"] = ple[i].convolution
            }
            return values
        }
    }
    public struct Output {
        public let logits: Tensor?
        public let stream: Tensor
        public let trace: [String: Tensor]
        public let ssdWaitSeconds: Double
        public let ssdLogicalBytes: Int
    }
    private struct Layer {
        let attnHC, mlpHC: GPUHyperConnection
        let moe: GPUMoE
        let gdn: GPUGatedDeltaNet?
        let attention: GPUAttention?
        let ple: GPUPLE?
    }
    public let configuration: QwenConfiguration
    public let weights: GPUWeights
    public let layerCount: Int
    public let profiler: GPUProfiler
    private let decodeAsyncSchedule: QwenDecodeAsyncSchedule
    /// Experimental host submission policy, captured once before model loading.
    public var experimentalDecodeAsyncEveryLayers: Int { decodeAsyncSchedule.everyLayers }
    /// Successful asyncEval calls made by this experiment, not GPU dispatches.
    public private(set) var experimentalDecodeAsyncSubmissions = 0
    /// Optional synchronous diagnostic observer on the single inference
    /// executor: (layer, absolute prompt offset, actual MoE input). Callers
    /// must clear it after capture and must not reenter model generation.
    /// It is never invoked for decode or verification. Materializing an input
    /// here perturbs scheduling; captured runs are not throughput benchmarks.
    public var prefillMoEObserver: ((Int, Int, Tensor) throws -> Void)?
    /// Host graph constructions, not physical GPU or bandwidth counters.
    public private(set) var prefillMoEReductionCalls = 0
    public private(set) var prefillMoEGateUpCalls = 0
    public private(set) var prefillMoEGroupedDownCalls = 0
    public let prefillAccumulation: GPUMoE.PrefillAccumulation
    private let embedding: Tensor
    private let layers: [Layer]
    private let mixer: GPUHyperConnection?
    private let head: Tensor?
    private let outputMask: Tensor
    private let identity = UUID()
    private let generationGate = QwenGenerationGate()
    private var generationFailure: String?
    private var verificationLinearKernel: GPUVerificationLinear?
    private let preparedFusedProjections: Bool
    private let preparedSharedElementwise: Bool

    /// Read-only capability check before a stage creates any device work.
    public func supportsDecodeMode(_ mode: GPUDecodeMode) -> Bool {
        (!mode.fusesProjections || preparedFusedProjections) &&
        (!mode.fusesSharedElementwise || preparedSharedElementwise)
    }

    /// All high-level generators sharing this model use the same admission
    /// guard. Low-level forward remains confined to its owning inference thread.
    public func withExclusiveGeneration<T>(_ body: () throws -> T) throws -> T {
        try generationGate.withExclusiveAccess {
            if let generationFailure { throw QwenGenerationError.unavailable("Reload model after failed device recovery: \(generationFailure)") }
            return try body()
        }
    }
    // Called only while the generation gate is held. All generators sharing
    // this model observe the failure; constructing another wrapper cannot hide it.
    func failGenerationRecovery(_ reason: String) { generationFailure = reason }

    public init(modelDirectory: URL, layerLimit: Int? = nil, profiler: GPUProfiler? = nil,
                reservedOutputIDs: Set<Int32>? = nil, ssdWorkers: Int = 1,
                prefillAccumulation: GPUMoE.PrefillAccumulation = .reference,
                decodeModes: [GPUDecodeMode] = [.reference], progress: ((Int, Int) -> Void)? = nil) throws {
        decodeAsyncSchedule = try QwenDecodeAsyncSchedule(
            environmentValue: ProcessInfo.processInfo.environment[QwenDecodeAsyncSchedule.environmentVariable])
        let c = try QwenConfiguration(modelDirectory: modelDirectory)
        guard c.hiddenSize == 2560, c.layerCount == 48, c.hcCount == 4,
              c.pleLayerIndices == [1], c.hcLowRank == 320,
              c.attentionHeads == 24, c.keyValueHeads == 2, c.headDimension == 256 else {
            throw GPUError.invalid("This runner supports the downloaded Qwen3.8 Flash-Next architecture only")
        }
        let count = layerLimit ?? c.layerCount
        guard (1...c.layerCount).contains(count) else { throw GPUError.invalid("Invalid layer limit") }
        configuration = c; layerCount = count
        self.profiler = try profiler ?? GPUProfiler()
        self.prefillAccumulation = prefillAccumulation
        let prepareProjections = decodeModes.contains { $0.fusesProjections }
        let prepareSharedElementwise = decodeModes.contains { $0.fusesSharedElementwise }
        preparedFusedProjections = prepareProjections
        preparedSharedElementwise = prepareSharedElementwise
        // Match tokenizer.zig's template-aware reserved-output suppression.
        // All supported chat markers remain available; these reserved tokens
        // and unsupported audio placeholders cannot be generated as text.
        var mask = [Float](repeating: 0, count: c.vocabularySize)
        let suppressed = try reservedOutputIDs ?? QwenTokenizer(modelDirectory: modelDirectory).reservedOutputTokenIDs
        for id in suppressed {
            guard id >= 0, id < c.vocabularySize else { throw GPUError.invalid("Invalid reserved output token") }
            mask[Int(id)] = 1
        }
        outputMask = try MX.array(mask, shape: [c.vocabularySize], dtype: MLX_BOOL)
        weights = try GPUWeights(modelDirectory: modelDirectory)
        embedding = try weights.tensor("language_model.model.embed_tokens.weight")
        guard embedding.shape == [c.vocabularySize, c.hiddenSize], embedding.dtype == MLX_BFLOAT16 else { throw GPUError.invalid("Unsupported embedding") }
        var loaded: [Layer] = []
        let hcFused = try GPUHyperConnectionFused()
        for i in 0..<count {
            let prefix = "language_model.model.layers.\(i)"
            let hcA = try GPUHyperConnection(weights: weights, prefix: prefix + ".attn_hyper_connection", fused: hcFused, fuseDecodeProjections: prepareProjections)
            let hcM = try GPUHyperConnection(weights: weights, prefix: prefix + ".mlp_hyper_connection", fused: hcFused, fuseDecodeProjections: prepareProjections)
            let gdn = c.layerTypes[i] == "linear_attention" ? try GPUGatedDeltaNet(layer: i, weights: weights, recurrence: .fused, fusedPrework: true, fuseDecodeProjections: prepareProjections) : nil
            let attn = c.layerTypes[i] == "full_attention" ? try GPUAttention(layer: i, weights: weights) : nil
            let ple = c.pleLayerIndices.contains(i) ? try GPUPLE(weights: weights, configuration: c, layer: i, ordinal: 0, workers: ssdWorkers) : nil
            let moe = try GPUMoE(weights: weights, layer: i, prefillAccumulation: prefillAccumulation, fuseSharedElementwise: prepareSharedElementwise)
            loaded.append(Layer(attnHC: hcA, mlpHC: hcM, moe: moe, gdn: gdn, attention: attn, ple: ple))
            progress?(i + 1, count)
        }
        layers = loaded
        if count == c.layerCount {
            mixer = try GPUHyperConnection(weights: weights, prefix: "language_model.model.hyper_connection_mixer", withInjection: false, fused: hcFused, fuseDecodeProjections: prepareProjections)
            head = try MX.transpose(weights.tensor("language_model.lm_head.weight"), [1, 0])
        } else { mixer = nil; head = nil }
    }
    public func makeState() -> State { State(layers: layerCount, owner: identity) }

    /// Immutable MLX handles retain the entire recurrent/PLE/KV/QSA state.
    /// Evaluate before publishing a checkpoint so delayed device errors cannot
    /// turn a speculative state into a reusable committed state.
    public func checkpoint(state: inout State) throws -> State {
        try evaluate([], state: &state)
        return state
    }

    /// Isolated diagnostics only. checkpoint()/State assignment retain shallow
    /// immutable handles; this method instead gathers every persistent tensor
    /// into new storage and evaluates the copies before returning. Never used
    /// by ordinary generation, MTP rollback or the experimental forward itself.
    public func diagnosticPrivateStateCopy(_ source: State) throws -> State {
        guard source.owner == identity, source.valid,
              source.gdn.allSatisfy({ $0.verificationCapture == nil }),
              source.ple.allSatisfy({ $0.verificationCapture == nil }) else {
            throw GPUError.invalid("Diagnostic copy requires this model's valid AR state without captures")
        }
        var ready = source
        _ = try checkpoint(state: &ready)
        var result = makeState()
        result.offset = source.offset
        func clone(_ value: Tensor?) throws -> Tensor? {
            try value.map { try GPUVerificationCopy.tensor($0) }
        }
        for i in layers.indices {
            result.gdn[i] = GPUGatedDeltaNet.State(convHistory: try clone(source.gdn[i].convHistory),
                recurrent: try clone(source.gdn[i].recurrent), offset: source.gdn[i].offset)
            let attention = source.attention[i]
            // Gather creates compact allocations; initializer derives retained
            // extents from the copied logical offset/pooled shape.
            result.attention[i] = GPUAttention.State(keys: try clone(attention.keys),
                values: try clone(attention.values), rawIndexerKeys: try clone(attention.rawIndexerKeys),
                pooledIndexerKeys: try clone(attention.pooledIndexerKeys), offset: attention.offset)
            result.ple[i].history = source.ple[i].history
            result.ple[i].convolution = try clone(source.ple[i].convolution)
        }
        try evaluate([], state: &result)
        return result
    }

    public func restore(_ checkpoint: State, state: inout State) throws {
        guard state.owner == identity, checkpoint.owner == identity,
              state.valid, checkpoint.valid else {
            throw GPUError.invalid("Checkpoint belongs to another model or a failed session")
        }
        state = checkpoint
    }

    /// Commit a prefix captured by one small verification forward. Recurrent
    /// states are captured at each position; attention and PLE histories are
    /// cropped to the same boundary. No rejected input survives in the result.
    public func commitVerificationPrefix(_ verified: State, from checkpoint: State,
                                         tokens: [Int32], count: Int) throws -> State {
        guard verified.owner == identity, checkpoint.owner == identity,
              verified.sessionIdentity == checkpoint.sessionIdentity,
              verified.valid, checkpoint.valid, (1...5).contains(tokens.count),
              (1...tokens.count).contains(count), verified.offset == checkpoint.offset + tokens.count else {
            throw GPUError.invalid("Invalid verification checkpoint or accepted prefix")
        }
        var result = verified
        for i in layers.indices {
            if layers[i].gdn != nil { result.gdn[i] = try verified.gdn[i].committingPrefix(count: count) }
            if let attention = layers[i].attention {
                result.attention[i] = try attention.prefixState(verified.attention[i], count: checkpoint.offset + count)
            }
            if let ple = layers[i].ple {
                result.ple[i] = try ple.committingPrefix(verified.ple[i], count: count, tokens: tokens,
                                                       previousHistory: checkpoint.ple[i].history)
            }
        }
        result.offset = checkpoint.offset + count
        try evaluate([], state: &result)
        return result
    }

    /// One known prompt chunk ahead, on a request-local serial CPU queue.
    /// Only token histories and decoded embedding data are planned ahead;
    /// recurrent and convolution state still advance in forward order.
    public final class PrefillPrefetch {
        private let owner: ObjectIdentifier
        private let tokens: [Int32]
        private let chunk: Int, initialOffset: Int
        private let pleLayers: [(Int, GPUPLE)]
        private let queue = DispatchQueue(label: "ane-runner.prompt-ssd", qos: .userInitiated)
        private var histories: [Int: [UInt32]]
        private var cursor = 0
        private var pending: (offset: Int, values: [Int: GPUPLE.PreparedInput])?

        fileprivate init(model: QwenModel, tokens: [Int32], chunk: Int, state: State) throws {
            owner = ObjectIdentifier(model)
            self.tokens = tokens; self.chunk = chunk; initialOffset = state.offset
            pleLayers = model.layers.enumerated().compactMap { i, layer in layer.ple.map { (i, $0) } }
            histories = Dictionary(uniqueKeysWithValues: pleLayers.map { ($0.0, state.ple[$0.0].history) })
            do { try scheduleNext() }
            catch { queue.sync {}; throw error }
        }

        private func scheduleNext() throws {
            guard cursor < tokens.count else { return }
            let end = cursor < tokens.count - 1 ? min(tokens.count - 1, cursor + chunk) : tokens.count
            let part = Array(tokens[cursor..<end])
            var values: [Int: GPUPLE.PreparedInput] = [:]
            for (i, ple) in pleLayers {
                let prepared = try ple.prefetch(tokens: part, history: histories[i]!, queue: queue)
                histories[i] = prepared.historyAfter
                values[i] = prepared
            }
            pending = (cursor, values)
            cursor = end
        }

        fileprivate func take(model: QwenModel, tokens: [Int32], state: inout State) throws -> [Int: PLEReadTask] {
            guard owner == ObjectIdentifier(model), let current = pending,
                  current.offset + initialOffset == state.offset else {
                throw GPUError.invalid("Prefill prefetch does not match this model/session position")
            }
            var work: [Int: PLEReadTask] = [:]
            var transferred = false
            defer { if !transferred { for task in work.values { task.drain() } } }
            for (i, ple) in pleLayers {
                guard let prepared = current.values[i] else { throw GPUError.invalid("Missing prefetched PLE input") }
                work[i] = try ple.consume(prepared: prepared, tokens: tokens, state: &state.ple[i])
            }
            pending = nil
            // The current task was queued first. At most it and the next
            // chunk are outstanding, sharing the configured CPU worker bound.
            try scheduleNext()
            transferred = true
            return work
        }

        /// Join outstanding CPU reads on success or error before the next request.
        public func finish() { queue.sync {} }
        deinit { finish() }
    }

    public func makePrefillPrefetch(tokens: [Int32], chunk: Int, state: State) throws -> PrefillPrefetch {
        guard state.owner == identity, state.valid, state.ple.count == layerCount, !tokens.isEmpty, (1...512).contains(chunk),
              state.offset <= configuration.maximumPositions - tokens.count,
              tokens.allSatisfy({ $0 >= 0 && $0 < configuration.vocabularySize }) else {
            throw GPUError.invalid("Invalid prompt for SSD prefetch")
        }
        return try PrefillPrefetch(model: self, tokens: tokens, chunk: chunk, state: state)
    }
    public var additionalProjectionBufferBytes: UInt64 {
        layers.reduce(UInt64(0)) { total, layer in
            total + layer.attnHC.additionalProjectionBufferBytes + layer.mlpHC.additionalProjectionBufferBytes
                + (layer.gdn?.additionalProjectionBufferBytes ?? 0)
        } + (mixer?.additionalProjectionBufferBytes ?? 0)
    }
    /// Device failures can happen after a lazy graph was successfully built.
    /// Invalidate the session in that case; reset is required before reuse.
    public func evaluate(_ outputs: [Tensor], state: inout State) throws {
        guard state.owner == identity, state.valid else { throw GPUError.invalid("Invalid model owner or failed session; reset before evaluation") }
        do { try MX.eval(outputs + state.tensors) }
        catch { state.valid = false; throw error }
    }
    /// Per single-token decode, count each selected expert once, each dense
    /// tensor once, and one embedding row. This is a logical footprint only;
    /// caches, refetches, and intermediate/state traffic are not measured.
    public var logicalDecodeWeightBytes: UInt64 {
        weights.ledger.reduce(UInt64(0)) { total, entry in
            if entry.name.contains(".mtp.") { return total }
            if entry.name == "language_model.model.embed_tokens.weight" { return total + UInt64(configuration.hiddenSize * 2) }
            if entry.name.contains(".switch_mlp.") { return total + entry.sourceBytes / UInt64(configuration.expertCount) * UInt64(configuration.expertsPerToken) }
            return total + entry.sourceBytes
        }
    }
    public func greedyToken(_ logits: Tensor) throws -> Tensor {
        try MX.argmax(MX.whereSelect(outputMask, MX.scalar(-.infinity, logits.dtype), logits))
    }

    /// Pass `phase` explicitly for new callers. A nil phase retains the old
    /// shape-based fallback (S1 decode, longer input prefill). The evaluation
    /// interval is local to this call, allowing independent stage policies.
    /// Experimental asynchronous submission requires an explicit .decode phase.
    /// MTP callers must pass allowExperimentalDecodeAsync: false, including S1.
    public func forward(tokens: [Int32], state: inout State, lastLogitOnly: Bool = true,
                        captureTrace: Bool = false, evaluateEveryLayers: Int = 4,
                        decodeMode: GPUDecodeMode = .reference, prefillPrefetch: PrefillPrefetch? = nil,
                        verifyScalarBoundaries: Bool = false, captureVerification: Bool = false,
                        verifyScalarMoE: Bool = false, verifyScalarLinear: Bool = false,
                        verifyTokenMoE: Bool = false,
                        phase: QwenExecutionPhase? = nil,
                        allowExperimentalDecodeAsync: Bool = true,
                        prefillAttention: GPUAttention.PrefillMode = .reference,
                        profileLogits: Bool = true,
                        prefillMoEConfiguration: GPUMoEPrefillConfiguration? = nil) throws -> Output {
        guard state.owner == identity, state.valid, state.gdn.count == layerCount, !tokens.isEmpty, evaluateEveryLayers > 0,
              !captureVerification || tokens.count <= 5,
              !verifyScalarLinear || (tokens.count <= 5 && decodeMode == .reference),
              !verifyTokenMoE || verifyScalarLinear,
              state.offset <= configuration.maximumPositions - tokens.count,
              tokens.allSatisfy({ $0 >= 0 && $0 < configuration.vocabularySize }) else {
            throw GPUError.invalid("Invalid input/session; reset a failed session before reuse")
        }
        let executionPhase = try QwenExecutionPhase.resolve(phase, tokenCount: tokens.count)
        try prefillAttention.validate(phase: executionPhase)
        if let prefillMoEConfiguration {
            try prefillMoEConfiguration.validated(phase: executionPhase)
            guard executionPhase == .prefill, !verifyScalarMoE, !verifyScalarLinear else {
                throw GPUError.invalid("Prefill MoE configuration cannot select decode or verification kernels")
            }
        }
        if profiler.isRecording { try profiler.setForwardContext(phase: executionPhase, position: state.offset) }
        defer { profiler.clearForwardContext() }
        // This prototype accepts text tokens. Vision/audio feature insertion is
        // a separate model path and must not silently become plain embeddings.
        let unsupported = Set<Int32>([248053, 248054, 248055, 248056, 248057, 248070, 248071, 248076])
        guard unsupported.isDisjoint(with: tokens) else { throw GPUError.invalid("Multimodal inputs are not supported") }
        do {
            let n = tokens.count
            if verifyScalarLinear, verificationLinearKernel == nil {
                verificationLinearKernel = try GPUVerificationLinear()
            }
            let linear = verifyScalarLinear ? verificationLinearKernel : nil
            var work: [Int: PLEReadTask] = [:]
            defer { for task in work.values { task.drain() } }
            if let prefillPrefetch {
                work = try prefillPrefetch.take(model: self, tokens: tokens, state: &state)
            } else {
                for i in layers.indices {
                    if let ple = layers[i].ple { work[i] = try ple.prepare(tokens: tokens, state: &state.ple[i]) }
                }
            }
            let embeddingPair = try profiler.measure("embedding", tokenCount: n, outputs: { [$0.0, $0.1] }) {
                let ids = try MX.array(tokens, shape: [1, n])
                let emb = try MX.take(embedding, ids, axis: 0)
                return (emb, try MX.tile(emb, [1, 1, configuration.hcCount]))
            }
            let emb = embeddingPair.0
            var h = embeddingPair.1
            var trace: [String: Tensor] = [:]
            var waitTime: Double = 0, logicalBytes = 0
            if captureTrace { trace["embedding"] = emb; trace["initial_stream"] = h }
            for i in layers.indices {
                let layer = layers[i]
                if let ple = layer.ple, let pending = work[i] {
                    // Start device work from the preceding block while SSD
                    // demand reads finish. Only the CPU payload crosses here.
                    try MX.asyncEval([h])
                    let start = DispatchTime.now().uptimeNanoseconds
                    let values = try pending.wait()
                    waitTime += Double(DispatchTime.now().uptimeNanoseconds - start) * 1e-9
                    logicalBytes += pending.logicalBytes
                    let e = try MX.array(values, shape: [1, n, configuration.pleEmbeddingDimension], dtype: MLX_BFLOAT16)
                    let pair = try profiler.measure("ple", layer: i, tokenCount: n, outputs: { [$0.0] + $0.1 }) {
                        let y = try ple.forward(h, embedding: e, state: &state.ple[i], captureVerification: captureVerification,
                                                verificationLinear: linear)
                        return (y, state.ple[i].tensors)
                    }
                    let add = pair.0
                    h = try MX.add(h, add)
                    if captureTrace { trace["ple_emb"] = e; trace["ple_out"] = add }
                    work.removeValue(forKey: i)
                }
                let pre = try profiler.measure("hc_attn_read", layer: i, tokenCount: n, outputs: { [$0.mixed] + [$0.injection].compactMap { $0 } }) { try layer.attnHC.read(h, useFusedProjections: decodeMode.fusesProjections, verificationLinear: linear) }
                let attnOut: Tensor
                if let gdn = layer.gdn {
                    attnOut = try profiler.measure("gdn", layer: i, tokenCount: n, outputs: { [$0.0] + $0.1 }) {
                        let y = try gdn.forward(pre.mixed, state: &state.gdn[i], useFusedProjections: decodeMode.fusesProjections,
                                                verifyScalarBoundaries: verifyScalarBoundaries,
                                                captureVerification: captureVerification,
                                                verificationLinear: linear); return (y, state.gdn[i].tensors)
                    }.0
                } else if let attention = layer.attention {
                    if profiler.isRecording, profiler.attentionBreakdown, executionPhase == .prefill {
                        // Flat child stages replace the inclusive parent; never
                        // double-count attention or nest synchronized measures.
                        attnOut = try attention.forward(pre.mixed, state: &state.attention[i],
                            verificationLinear: linear, prefillMode: prefillAttention, profiler: profiler)
                    } else {
                        attnOut = try profiler.measure("attention", layer: i, tokenCount: n, outputs: { [$0.0] + $0.1 }) {
                            let y = try attention.forward(pre.mixed, state: &state.attention[i],
                                verificationLinear: linear, prefillMode: prefillAttention); return (y, state.attention[i].tensors)
                        }.0
                    }
                }
                else { throw GPUError.invalid("Missing layer attention") }
                h = try profiler.measure("hc_attn_write", layer: i, tokenCount: n, outputs: { [$0] }) { try layer.attnHC.write(h, output: attnOut, injection: pre.injection!) }
                let pre2 = try profiler.measure("hc_mlp_read", layer: i, tokenCount: n, outputs: { [$0.mixed] + [$0.injection].compactMap { $0 } }) { try layer.mlpHC.read(h, useFusedProjections: decodeMode.fusesProjections, verificationLinear: linear) }
                if executionPhase == .prefill, let observer = prefillMoEObserver {
                    try observer(i, state.offset, pre2.mixed)
                }
                let moeOut: Tensor
                let reductionThreads = prefillMoEConfiguration?.threadgroupSize(tokenCount: n)
                let gateUpVariant = prefillMoEConfiguration?.effectiveGateUpVariant(tokenCount: n)
                let groupedDown = prefillMoEConfiguration?.usesGroupedDown(tokenCount: n) ?? false
                if profiler.isRecording, profiler.moeBreakdown, executionPhase == .prefill,
                   n > 1, linear == nil, !verifyScalarMoE {
                    moeOut = try layer.moe.forward(pre2.mixed,
                        useFusedSharedElementwise: decodeMode.fusesSharedElementwise, profiler: profiler,
                        prefillReductionThreadgroup: reductionThreads, prefillGateUpVariant: gateUpVariant,
                        groupedDown: groupedDown).y
                } else {
                    moeOut = try profiler.measure("moe", layer: i, tokenCount: n, outputs: { [$0] }) {
                        if let linear, n > 1 {
                            return try layer.moe.forward(pre2.mixed,
                                useFusedSharedElementwise: decodeMode.fusesSharedElementwise,
                                verificationLinear: linear, verificationTokenAxis: verifyTokenMoE).y
                        }
                        if verifyScalarMoE, n > 1 {
                            let outputs = try (0..<n).map { row in
                                let x = try MX.slice(pre2.mixed, starts: [0,row,0], ends: [1,row+1,configuration.hiddenSize])
                                return try layer.moe.forward(x, useFusedSharedElementwise: decodeMode.fusesSharedElementwise).y
                            }
                            return try MX.concat(outputs, axis: 1)
                        }
                        return try layer.moe.forward(pre2.mixed, useFusedSharedElementwise: decodeMode.fusesSharedElementwise,
                            prefillReductionThreadgroup: reductionThreads, prefillGateUpVariant: gateUpVariant,
                            groupedDown: groupedDown).y
                    }
                }
                if reductionThreads != nil { prefillMoEReductionCalls += 1 }
                if gateUpVariant != nil { prefillMoEGateUpCalls += 1 }
                if groupedDown { prefillMoEGroupedDownCalls += 1 }
                if captureTrace {
                    trace["layer.\(i).attn_input"] = pre.mixed
                    trace["layer.\(i).attn_out"] = attnOut
                    trace["layer.\(i).moe_input"] = pre2.mixed
                    trace["layer.\(i).moe_out"] = moeOut
                    trace["layer.\(i).before_moe_write"] = h
                    trace["layer.\(i).mlp_injection"] = pre2.injection!
                }
                h = try profiler.measure("hc_mlp_write", layer: i, tokenCount: n, outputs: { [$0] }) { try layer.mlpHC.write(h, output: moeOut, injection: pre2.injection!) }
                if captureTrace { trace["layer.\(i).stream"] = h }
                // Submit completed AR blocks while later blocks are still being
                // built. Final selected/state evaluation remains the caller's join.
                // MTP's scalar verify/replay and target-only calls opt out explicitly.
                if decodeAsyncSchedule.shouldSubmit(phase: phase, tokenCount: n,
                    completedLayers: i + 1, layerCount: layerCount, allowed: allowExperimentalDecodeAsync) {
                    try MX.asyncEval([h] + state.tensors)
                    experimentalDecodeAsyncSubmissions += 1
                }
                // Each business phase owns its evaluation interval. Defaults
                // preserve the previous multi-token schedule and S1 behavior.
                if executionPhase.shouldEvaluate(completedLayers: i + 1, tokenCount: n, every: evaluateEveryLayers) {
                    try MX.eval([h] + state.tensors)
                }
            }
            let logits: Tensor?
            if let mixer, let head {
                let finalInput = lastLogitOnly ? try MX.slice(h, starts: [0, n - 1, 0], ends: [1, n, configuration.hcCount * configuration.hiddenSize]) : h
                let projectHead = { () throws -> Tensor in
                    let final = try mixer.read(finalInput, useFusedProjections: decodeMode.fusesProjections, verificationLinear: linear).mixed
                    return try MX.linear(final, head, verification: linear)
                }
                // Intermediate prefill logits are a lazy, unused graph. Do not
                // make profiling execute a vocabulary projection absent from
                // normal work; the final prompt row still records it.
                if profileLogits {
                    logits = try profiler.measure("mixer_and_head", tokenCount: n, outputs: { [$0] }, projectHead)
                } else { logits = try projectHead() }
            } else { logits = nil }
            state.offset += n
            return Output(logits: logits, stream: h, trace: trace, ssdWaitSeconds: waitTime, ssdLogicalBytes: logicalBytes)
        } catch { state.valid = false; throw error }
    }
}

/// Pure CPU policy: only the literal values 0 and 8 are supported. The
/// environment is read by QwenModel.init, never again during a forward.
struct QwenDecodeAsyncSchedule {
    static let environmentVariable = "ANERUNNER_EXPERIMENTAL_DECODE_ASYNC_LAYERS"
    let everyLayers: Int

    init(environmentValue: String?) throws {
        switch environmentValue {
        case nil, .some("0"): everyLayers = 0
        case .some("8"): everyLayers = 8
        default:
            throw GPUError.invalid("\(Self.environmentVariable) requires exactly 0 or 8")
        }
    }

    func shouldSubmit(phase: QwenExecutionPhase?, tokenCount: Int,
                      completedLayers: Int, layerCount: Int, allowed: Bool) -> Bool {
        allowed && everyLayers == 8 && phase == .decode && tokenCount == 1 &&
            completedLayers > 0 && completedLayers < layerCount && completedLayers % everyLayers == 0
    }
}
