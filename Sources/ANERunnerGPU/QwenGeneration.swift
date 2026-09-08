import ANERunnerCore
import Dispatch
import Foundation
import Synchronization

public enum QwenGenerationError: Error, LocalizedError, Equatable {
    case invalidRequest(String)
    case busy
    case resourceLimit(String)
    case cancelled
    case unavailable(String)

    public var errorDescription: String? {
        switch self {
        case .invalidRequest(let reason): return "Invalid generation request: \(reason)"
        case .resourceLimit(let reason): return "State capacity unavailable: \(reason)"
        case .busy: return "This model is already processing a generation request"
        case .cancelled: return "Generation was cancelled"
        case .unavailable(let reason): return "Generation is unavailable: \(reason)"
        }
    }
}

/// Token-only, greedy text generation. Validation never touches the GPU.
public struct QwenGenerationRequest: Sendable {
    public let tokens: [Int32]
    public let maxTokens: Int
    public let contextLimit: Int
    public let prefillChunk: Int
    public let mtpDepth: Int
    public let verification: QwenMTPDecoder.Verification
    /// Optional prompt-history cap for the draft head only. Target context is complete.
    public let draftHistoryTokens: Int?
    public let prefillEvaluateEveryLayers: Int
    public let verificationEvaluateEveryLayers: Int
    /// Applies to decode/verification projections and elementwise kernels.
    /// Prefill attention has an independent request policy.
    public let decodeMode: GPUDecodeMode
    /// Request-local trunk prefill policy; draft history and decode keep their own kernels.
    public let prefillAttention: GPUAttention.PrefillMode
    public let prefillMoEConfiguration: GPUMoEPrefillConfiguration?
    /// Trusted reusable prefix of the complete encoded prompt. Nil/zero opts
    /// out; the runtime rounds down to an original prefill chunk boundary.
    public let prefixCacheMaxTokens: Int?
    /// Complete-conversation lookup with bounded original-grid checkpoints.
    /// Mutually exclusive with the legacy prefixCacheMaxTokens hint.
    public let prefixCachePlan: QwenConversationPrefixPlan?

    public init(tokens: [Int32], maxTokens: Int = 128, contextLimit: Int = 16_384,
                prefillChunk: Int = 416, mtpDepth: Int = 0,
                verification: QwenMTPDecoder.Verification = .scalar, draftHistoryTokens: Int? = nil,
                prefillEvaluateEveryLayers: Int = 4, verificationEvaluateEveryLayers: Int = 4,
                decodeMode: GPUDecodeMode = .reference,
                prefillAttention: GPUAttention.PrefillMode = .reference,
                prefillMoEConfiguration: GPUMoEPrefillConfiguration? = nil,
                prefixCacheMaxTokens: Int? = nil,
                prefixCachePlan: QwenConversationPrefixPlan? = nil) {
        self.tokens = tokens; self.maxTokens = maxTokens
        self.contextLimit = contextLimit; self.prefillChunk = prefillChunk
        self.mtpDepth = mtpDepth; self.verification = verification
        self.draftHistoryTokens = draftHistoryTokens
        self.prefillEvaluateEveryLayers = prefillEvaluateEveryLayers
        self.verificationEvaluateEveryLayers = verificationEvaluateEveryLayers
        self.decodeMode = decodeMode
        self.prefillAttention = prefillAttention
        self.prefillMoEConfiguration = prefillMoEConfiguration
        self.prefixCacheMaxTokens = prefixCacheMaxTokens
        self.prefixCachePlan = prefixCachePlan
    }

    public func validate(configuration: QwenConfiguration) throws {
        try prefillMoEConfiguration?.validated()
        guard !tokens.isEmpty else { throw QwenGenerationError.invalidRequest("prompt tokens are empty") }
        guard tokens.allSatisfy({ $0 >= 0 && Int($0) < configuration.vocabularySize }) else {
            throw QwenGenerationError.invalidRequest("prompt contains an out-of-vocabulary token")
        }
        // Keep this text-only boundary identical to QwenModel.forward.
        let multimodal = Set<Int32>([248053, 248054, 248055, 248056, 248057, 248070, 248071, 248076])
        guard multimodal.isDisjoint(with: tokens) else {
            throw QwenGenerationError.invalidRequest("multimodal inputs are unsupported")
        }
        guard maxTokens > 0 else { throw QwenGenerationError.invalidRequest("maxTokens must be positive") }
        try validatePrefixCachePolicy()
        guard contextLimit > 0, contextLimit <= configuration.maximumPositions else {
            throw QwenGenerationError.invalidRequest("contextLimit exceeds the model's supported range")
        }
        guard (1...512).contains(prefillChunk) else {
            throw QwenGenerationError.invalidRequest("prefillChunk must be between 1 and 512")
        }
        guard (0...4).contains(mtpDepth) else {
            throw QwenGenerationError.invalidRequest("mtpDepth must be between 0 and 4")
        }
        guard mtpDepth == 0 || !verification.usesScalarLinear || decodeMode == .reference else {
            throw QwenGenerationError.invalidRequest("\(verification.rawValue) verification requires reference decode kernels")
        }
        if let draftHistoryTokens, !(1...configuration.maximumPositions).contains(draftHistoryTokens) {
            throw QwenGenerationError.invalidRequest("draftHistoryTokens must be positive and within the model context limit")
        }
        guard (1...configuration.layerCount).contains(prefillEvaluateEveryLayers),
              (1...configuration.layerCount).contains(verificationEvaluateEveryLayers) else {
            throw QwenGenerationError.invalidRequest("phase evaluation intervals must be within model layer count")
        }
        let (budget, overflow) = tokens.count.addingReportingOverflow(maxTokens)
        guard !overflow, budget <= contextLimit else {
            throw QwenGenerationError.invalidRequest("prompt plus requested output exceeds contextLimit")
        }
    }
}

/// Cooperative cancellation may be requested from another thread. An active
/// device evaluation or SSD read finishes before cancellation can be observed.
public final class QwenCancellation: Sendable {
    private let value = Mutex(false)
    public init() {}
    public func cancel() { value.withLock { $0 = true } }
    public var isCancelled: Bool { value.withLock { $0 } }
    public func check() throws {
        if isCancelled { throw QwenGenerationError.cancelled }
    }
}

/// CPU-testable admission primitive. The owning model holds one gate, shared
/// by every generator using that model. No inference object becomes Sendable.
final class QwenGenerationGate: @unchecked Sendable {
    private let lock = NSLock()
    func withExclusiveAccess<T>(_ body: () throws -> T) throws -> T {
        guard lock.try() else { throw QwenGenerationError.busy }
        defer { lock.unlock() }
        return try body()
    }
}

public enum QwenGenerationFinishReason: String, Codable, Sendable {
    case eos
    case length
}

public struct QwenGenerationStatistics: Codable, Sendable {
    public let promptTokenCount: Int
    public let generatedTokenCount: Int
    /// Number of target decode calls/rounds, not the number of draft tokens.
    public let decodeRounds: Int
    /// Generated tokens after the first token; includes EOS when emitted.
    public let decodedTokenCount: Int
    public let prefillChunkCount: Int
    /// Consumed trunk position before discarding this request. MTP verification
    /// may have evaluated past a stopping EOS; this state is never reused.
    public let finalStateOffset: Int
    public let ssdWaitSeconds: Double
    /// Logical requested row bytes, not measured physical SSD/DRAM traffic.
    public let ssdLogicalBytes: Int
    public let callbackSeconds: Double
    public let prefillAccumulation: String
    public let mtpDepth: Int
    public let mtpVerification: String?
    public let mtp: QwenMTPDecoder.Statistics?
}

public struct QwenGenerationResult: Codable, Sendable {
    /// Complete committed output IDs, including a generated EOS token.
    public let tokens: [Int32]
    public let finishReason: QwenGenerationFinishReason
    /// First-use MTP head construction only. Lazy GPU work remains in generation.
    public let preparationSeconds: Double
    /// Prompt processing plus handoff waiting through first-token publication
    /// readiness. Use phases.prefill for producer-only performance.
    public let timeToFirstTokenSeconds: Double
    /// Sum of decode rounds, including draft/verify/replay, excluding callbacks.
    public let decodeSeconds: Double
    /// Request execution after preparation, including callbacks and final checks.
    public let totalSeconds: Double
    public let statistics: QwenGenerationStatistics
    /// Separate producer, handoff and consumer costs; nil in historical reports.
    public let phases: QwenGenerationPhases?
    /// Derived from completed CPU counters only; nil for AR or historical reports.
    public let mtpCostSummary: QwenMTPCostSummary?

    init(tokens: [Int32], finishReason: QwenGenerationFinishReason,
         preparationSeconds: Double, timeToFirstTokenSeconds: Double,
         decodeSeconds: Double, totalSeconds: Double,
         statistics: QwenGenerationStatistics, phases: QwenGenerationPhases?) {
        self.tokens = tokens; self.finishReason = finishReason
        self.preparationSeconds = preparationSeconds
        self.timeToFirstTokenSeconds = timeToFirstTokenSeconds
        self.decodeSeconds = decodeSeconds; self.totalSeconds = totalSeconds
        self.statistics = statistics; self.phases = phases
        mtpCostSummary = statistics.mtp.map {
            QwenMTPCostSummary(statistics: $0, decodeSteps: statistics.decodeRounds,
                committedDecodeTokens: statistics.decodedTokenCount, decodeSeconds: decodeSeconds)
        }
    }

    public var decodeTokensPerSecond: Double? {
        decodeSeconds > 0 ? Double(statistics.decodedTokenCount) / decodeSeconds : nil
    }
    /// Mean time per committed token after the first; MTP burst size is not a token count.
    public var decodeSecondsPerToken: Double? {
        statistics.decodedTokenCount > 0 ? decodeSeconds / Double(statistics.decodedTokenCount) : nil
    }
    /// Consumer throughput including callbacks, excluding prefill and handoff wait.
    public var decodeServiceTokensPerSecond: Double? {
        guard let phases, phases.decodeServiceSeconds > 0 else { return nil }
        return Double(statistics.decodedTokenCount) / phases.decodeServiceSeconds
    }
}

/// Timings for the independently completed prefill job. These are wall times,
/// not GPU-only timers. Target work includes final prompt logits/first selection.
public struct QwenPrefillStatistics: Codable, Sendable {
    public let promptTokenCount: Int
    public let chunkCount: Int
    public let targetSeconds: Double
    public let draftHistorySeconds: Double
    /// Active producer execution, excluding head construction and pauses.
    public let totalSeconds: Double
    public let ssdWaitSeconds: Double
    public let ssdLogicalBytes: Int
    public let evaluateEveryLayers: Int
    /// Requested prefill attention policy; nil in historical reports.
    public var attentionMode: String? = nil
    /// Time outside admitted prefill slices; nil in historical reports.
    public var suspensionSeconds: Double? = nil
    /// Full prompt usage is unchanged; rates count only tokens actually computed.
    public var cachedTokenCount: Int? = nil
    public var computedTokenCount: Int? = nil
    /// Trunk prefill positions actually forwarded, including any discarded work.
    /// Nil in historical reports. The current cursor never discards prefix work.
    public var actualForwardTokenCount: Int? = nil
    /// Actual forward positions beyond the unique adopted computed prefix.
    public var recomputedTokenCount: Int? = nil
    public var cacheLookupSeconds: Double? = nil
    public var cacheRestoreSeconds: Double? = nil
    public var cacheSaveSeconds: Double? = nil
    public var cacheSource: String? = nil
    public var cacheWaitSeconds: Double? = nil
    public var targetTokensPerSecond: Double? {
        targetSeconds > 0 ? Double(actualForwardTokenCount ?? computedTokenCount ?? promptTokenCount) / targetSeconds : nil
    }
    public var readyTokensPerSecond: Double? {
        totalSeconds > 0 ? Double(actualForwardTokenCount ?? computedTokenCount ?? promptTokenCount) / totalSeconds : nil
    }
}

public struct QwenGenerationPhases: Codable, Sendable {
    public let prefill: QwenPrefillStatistics
    /// Time between producer readiness and consumer admission. Not compute time.
    public let handoffWaitSeconds: Double
    /// Same-process handle consumption only; no network/serialization is implied.
    public let handoffConsumeSeconds: Double
    /// Active consumer execution including callbacks, excluding handoff and pauses.
    public let decodeServiceSeconds: Double
    public let decodeSSDWaitSeconds: Double
    public let decodeSSDLogicalBytes: Int
    public let verificationEvaluateEveryLayers: Int
    public let decodeKernelMode: String
    /// Time outside admitted decode slices; nil in historical reports.
    public var decodeSuspensionSeconds: Double? = nil
}

/// A single-use payload, confined to the model's inference executor. An invalid
/// owner check happens before take, and a taken payload cannot be retried after
/// a partial decode. This type contains no GPU operations and is CPU-testable.
final class QwenSingleUseHandoff<Value> {
    private var value: Value?
    init(_ value: Value) { self.value = value }
    var isReady: Bool { value != nil }
    func take() throws -> Value {
        guard let value else { throw QwenGenerationError.invalidRequest("prefill handoff was already consumed or discarded") }
        self.value = nil
        return value
    }
    func discard() { value = nil }
}

/// Complete, evaluated request state transferred between two jobs on the SAME
/// model and inference executor. Not Sendable or serializable. No GPU/SSD work
/// remains in flight. The initial token is selected but has not been published.
/// The MTP preparation profile is fixed before prefill because its history needs
/// real trunk hidden states; changing it requires a different prefill job.
public final class QwenPrefillResult {
    public let statistics: QwenPrefillStatistics
    public let preparationSeconds: Double
    public let firstToken: Int32
    public var isReady: Bool { payload.isReady }
    fileprivate let model: QwenModel
    fileprivate let request: QwenGenerationRequest
    fileprivate let readyAt: UInt64
    fileprivate let requestStartedAt: UInt64
    fileprivate let payload: QwenSingleUseHandoff<QwenPrefillPayload>
    fileprivate init(model: QwenModel, request: QwenGenerationRequest, state: QwenModel.State,
                     decoder: QwenMTPDecoder?, requestLease: QwenStateBudget.Lease,
                     firstToken: Int32, statistics: QwenPrefillStatistics,
                     preparationSeconds: Double, requestStartedAt: UInt64) {
        self.model = model; self.request = request; self.firstToken = firstToken
        self.statistics = statistics; self.preparationSeconds = preparationSeconds
        readyAt = DispatchTime.now().uptimeNanoseconds
        self.requestStartedAt = requestStartedAt
        payload = QwenSingleUseHandoff(QwenPrefillPayload(requestLease: requestLease, state: state, decoder: decoder))
    }
    /// Release an unused, completed job on the same inference executor.
    public func discard() { payload.discard() }
}

fileprivate struct QwenPrefillPayload {
    let requestLease: QwenStateBudget.Lease
    var state: QwenModel.State
    let decoder: QwenMTPDecoder?
}

/// A request-local prefill cursor on the model's inference executor. One step
/// consumes one original prompt chunk (including the separate final token).
/// Completed lookahead reads remain owned here. A waiting prefix restore may
/// retain host-only SSD I/O across a yield; no GPU work remains in flight.
/// This class deliberately does not conform to Sendable.
public final class QwenPrefillSession {
    fileprivate let model: QwenModel
    fileprivate let request: QwenGenerationRequest
    fileprivate let cancellation: QwenCancellation?
    fileprivate var progress: QwenPrefillProgress?
    public private(set) var processedTokenCount = 0
    public fileprivate(set) var isActive = false
    public var isFinished: Bool { progress == nil }
    public var isWaitingForPrefixCache: Bool { progress?.cacheResolved == false }
    fileprivate init(model: QwenModel, request: QwenGenerationRequest,
                     cancellation: QwenCancellation?, progress: QwenPrefillProgress) {
        self.model = model; self.request = request; self.cancellation = cancellation
        self.progress = progress
    }
    /// Discard only at a yielded boundary. Active-step/callback mutation is rejected.
    public func discard() throws {
        guard !isActive else { throw QwenGenerationError.busy }
        invalidate()
    }
    fileprivate func invalidate() {
        progress?.prefetch?.finish()
        progress = nil
    }
    fileprivate func recordProcessed(_ count: Int) { processedTokenCount = count }
}

fileprivate final class QwenPrefillProgress {
    let requestLease: QwenStateBudget.Lease
    var state: QwenModel.State
    let decoder: QwenMTPDecoder?
    let preparationSeconds: Double
    let startedAt: UInt64
    var lastYieldAt: UInt64
    var prefetch: QwenModel.PrefillPrefetch?
    var offset = 0, chunks = 0, ssdBytes = 0
    var ssdWait = 0.0, targetSeconds = 0.0, activeSeconds = 0.0, suspensionSeconds = 0.0
    var cacheBoundaries: [Int] = []
    var cachedTokens = 0, actualForwardTokens = 0
    var cacheFlight: QwenPrefixCacheFlight?
    var cacheResolved = true
    var cacheSource = "cold"
    var cacheWaitSeconds = 0.0
    var cacheLookupSeconds = 0.0, cacheRestoreSeconds = 0.0, cacheSaveSeconds = 0.0
    init(state: QwenModel.State, decoder: QwenMTPDecoder?, requestLease: QwenStateBudget.Lease, preparationSeconds: Double, now: UInt64) {
        self.requestLease = requestLease
        self.state = state; self.decoder = decoder; self.preparationSeconds = preparationSeconds
        startedAt = now; lastYieldAt = now
    }
}

/// Same-model, same-executor decode cursor. The first step publishes the
/// prepared first token; subsequent steps finish one entire AR/MTP round.
/// All tokens committed by a round are published before yielding. A callback
/// error or cancellation after admission invalidates the whole cursor.
public final class QwenDecodeSession {
    fileprivate let model: QwenModel
    fileprivate let request: QwenGenerationRequest
    fileprivate let cancellation: QwenCancellation?
    fileprivate var progress: QwenDecodeProgress?
    public private(set) var generatedTokenCount = 0
    public fileprivate(set) var isActive = false
    public var isFinished: Bool { progress == nil }
    fileprivate init(model: QwenModel, request: QwenGenerationRequest,
                     cancellation: QwenCancellation?, progress: QwenDecodeProgress) {
        self.model = model; self.request = request; self.cancellation = cancellation
        self.progress = progress
    }
    public func discard() throws {
        guard !isActive else { throw QwenGenerationError.busy }
        progress = nil
    }
    fileprivate func recordGenerated(_ count: Int) { generatedTokenCount = count }
}

fileprivate final class QwenDecodeProgress {
    let requestLease: QwenStateBudget.Lease
    var state: QwenModel.State
    let decoder: QwenMTPDecoder?
    let prepared: QwenPrefillResult
    let handoffWaitSeconds, handoffConsumeSeconds: Double
    var lastYieldAt: UInt64
    var next: Int32
    var generated: [Int32] = []
    var firstTokenReadySeconds: Double?
    var callbackSeconds = 0.0, decodeSeconds = 0.0, activeSeconds = 0.0, suspensionSeconds = 0.0
    var decodeRounds = 0, ssdBytes = 0
    var ssdWait = 0.0
    init(payload: QwenPrefillPayload, prepared: QwenPrefillResult,
         handoffWaitSeconds: Double, handoffConsumeSeconds: Double, now: UInt64) {
        requestLease = payload.requestLease
        state = payload.state; decoder = payload.decoder; self.prepared = prepared
        self.handoffWaitSeconds = handoffWaitSeconds; self.handoffConsumeSeconds = handoffConsumeSeconds
        next = prepared.firstToken; lastYieldAt = now
    }
}

/// Two independently admitted jobs on one inference executor. The convenience
/// generate API holds admission across both; prefill/decode release it between
/// jobs so another request may run while a completed handoff is waiting.
/// This is a library boundary, not a cross-process transport or HTTP service.
public final class QwenGenerator {
    public let model: QwenModel
    public let eosTokenIDs: Set<Int32>
    private var mtpHead: QwenMTP?
    private let prefixCache: QwenPrefixCache?
    private let memoryPressure: QwenMemoryPressurePolicy?
    /// Diagnostics only: readback here perturbs execution and is excluded from
    /// production timing trials. Do not mutate tensors or reenter generation.
    public var prefixStateObserver: ((String, QwenModel.State) throws -> Void)?
    /// Read only on the owning inference executor, like other model statistics.
    public var prefixCacheStatistics: QwenPrefixCacheStatistics? { prefixCache?.statistics }
    public var prefixDiskStatistics: QwenPrefixDiskStatistics? { prefixCache?.disk?.statistics }
    public var stateBudgetStatistics: QwenStateBudget.Statistics { model.stateBudget.statistics }
    public func clearPrefixCache(resetStatistics: Bool = false, includingDisk: Bool = false) throws {
        try model.withExclusiveGeneration { prefixCache?.clear(resetStatistics: resetStatistics, includingDisk: includingDisk) }
    }
    public func flushPrefixCacheWrites() throws {
        try model.withExclusiveGeneration { prefixCache?.disk?.flush() }
    }
    public func closePrefixCache(drain: Bool = true) throws {
        try model.withExclusiveGeneration { prefixCache?.disk?.close(drain: drain) }
    }
    @discardableResult
    public func closePrefixCache(drain: Bool = true, timeout: TimeInterval) throws -> QwenPrefixDiskCloseResult? {
        try model.withExclusiveGeneration { prefixCache?.disk?.close(drain: drain, timeout: timeout) }
    }
    @discardableResult
    public func trimPrefixCacheMemory(maxEntries: Int = Int.max) throws -> Int {
        try model.withExclusiveGeneration { prefixCache?.trimMemory(maxEntries: maxEntries) ?? 0 }
    }

    public init(model: QwenModel, prefixCacheLimits: QwenPrefixCacheLimits? = nil,
                prefixDiskStore: QwenPrefixDiskStore? = nil,
                memoryPressurePolicy: QwenMemoryPressurePolicy? = nil) throws {
        self.model = model
        memoryPressure = memoryPressurePolicy
        guard prefixDiskStore == nil || prefixCacheLimits != nil else {
            throw QwenGenerationError.invalidRequest("SSD prefix cache requires prefix cache limits")
        }
        prefixCache = try prefixCacheLimits.map {
            try QwenPrefixCache(limits: $0, disk: prefixDiskStore, model: model, memoryPressure: memoryPressurePolicy)
        }
        let stops = try QwenTokenizer(modelDirectory: model.configuration.modelDirectory).eosTokenIDs
        guard !stops.isEmpty, stops.allSatisfy({ $0 >= 0 && Int($0) < model.configuration.vocabularySize }) else {
            throw QwenGenerationError.unavailable("tokenizer has no valid EOS policy")
        }
        eosTokenIDs = stops
        if let prefixCache {
            try model.withExclusiveGeneration { model.registerPrefixCache(prefixCache) }
        }
    }
    private func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
    private func elapsed(_ start: UInt64) -> Double { Double(now() - start) * 1e-9 }
    private func validate(_ request: QwenGenerationRequest, cancellation: QwenCancellation?) throws {
        try cancellation?.check()
        try validateRequest(request)
    }
    /// CPU-only admission validation for local queues, before reserving state
    /// capacity. Device health and exclusive access are checked at stage entry.
    public func validateRequest(_ request: QwenGenerationRequest) throws {
        try request.validate(configuration: model.configuration)
        guard model.layerCount == model.configuration.layerCount else {
            throw QwenGenerationError.unavailable("generation requires all model layers and the output head")
        }
        guard model.supportsDecodeMode(request.decodeMode) else {
            throw QwenGenerationError.invalidRequest("model was not prepared for the requested decode kernel mode")
        }
        _ = try requestStateReservation(request)
    }

    private func requestStateReservation(_ request: QwenGenerationRequest) throws -> Int {
        let base = try model.estimatedPrefixStateBytes(at: request.tokens.count + request.maxTokens)
        // Includes a second state-sized allowance for old/new arrays during
        // functional updates. MTP additionally retains rollback and draft state.
        // Weights, general activations and the MLX allocator are separate metrics.
        let multiplier = request.mtpDepth == 0 ? 2 : request.mtpDepth + 5
        let (bytes, overflow) = base.multipliedReportingOverflow(by: multiplier)
        guard !overflow, bytes > 0, bytes <= model.stateBudget.maxBytes else {
            throw QwenGenerationError.resourceLimit("request state exceeds the configured joint byte budget")
        }
        return bytes
    }
    private func recover() {
        do { try MX.synchronize() }
        catch { model.failGenerationRecovery(String(describing: error)) }
    }

    /// Begin without executing a prompt chunk. Cache restoration and MTP
    /// preparation happen here, on the admitted inference executor.
    /// No cursor is returned on a failed admission/preparation.
    public func beginPrefill(_ request: QwenGenerationRequest,
                             cancellation: QwenCancellation? = nil) throws -> QwenPrefillSession {
        try model.withExclusiveGeneration {
            try validate(request, cancellation: cancellation)
            return try makePrefillSession(request, cancellation: cancellation)
        }
    }

    /// Nil means a prompt chunk completed, or cache I/O/a shared producer is
    /// pending (isWaitingForPrefixCache is true). Admission errors
    /// (including cancellation observed before entering the step) preserve the
    /// cursor; errors after a step starts invalidate it. Caller may discard a
    /// preserved cursor when choosing not to retry admission.
    public func stepPrefill(_ session: QwenPrefillSession,
                            cancellation: QwenCancellation? = nil) throws -> QwenPrefillResult? {
        try model.withExclusiveGeneration {
            try validatePrefillSession(session, cancellation: cancellation)
            return try prefillStep(session, cancellation: cancellation)
        }
    }

    /// Claims a ready handoff exactly once, after CPU validation/admission.
    /// The first output token is published by the first stepDecode, not here.
    public func beginDecode(_ prepared: QwenPrefillResult,
                            cancellation: QwenCancellation? = nil) throws -> QwenDecodeSession {
        try model.withExclusiveGeneration {
            try validatePrepared(prepared, cancellation: cancellation)
            return try makeDecodeSession(prepared, cancellation: cancellation)
        }
    }

    /// One first-token publication or one complete AR/MTP round. A round may
    /// publish depth+1 tokens; it is never interrupted into resumable half-rounds.
    public func stepDecode(_ session: QwenDecodeSession, cancellation: QwenCancellation? = nil,
                           onToken: ((Int32) throws -> Void)? = nil) throws -> QwenGenerationResult? {
        try model.withExclusiveGeneration {
            try validateDecodeSession(session, cancellation: cancellation)
            return try decodeStep(session, cancellation: cancellation, onToken: onToken)
        }
    }

    /// Complete-stage compatibility API, using the same chunk state machine
    /// while retaining model admission for the complete prefill stage.
    public func prefill(_ request: QwenGenerationRequest,
                        cancellation: QwenCancellation? = nil) throws -> QwenPrefillResult {
        try model.withExclusiveGeneration {
            try validate(request, cancellation: cancellation)
            return try runPrefill(request, cancellation: cancellation)
        }
    }

    public func decode(_ prepared: QwenPrefillResult, cancellation: QwenCancellation? = nil,
                       onToken: ((Int32) throws -> Void)? = nil) throws -> QwenGenerationResult {
        try model.withExclusiveGeneration {
            try validatePrepared(prepared, cancellation: cancellation)
            return try runDecode(prepared, cancellation: cancellation, onToken: onToken)
        }
    }

    /// The convenience API retains admission across all internal slices, so
    /// introducing cursors does not make an existing generate call interleave.
    public func generate(_ request: QwenGenerationRequest, cancellation: QwenCancellation? = nil,
                         onToken: ((Int32) throws -> Void)? = nil) throws -> QwenGenerationResult {
        try model.withExclusiveGeneration {
            try validate(request, cancellation: cancellation)
            let start = now()
            let prepared = try runPrefill(request, cancellation: cancellation)
            let result = try runDecode(prepared, cancellation: cancellation, onToken: onToken)
            return QwenGenerationResult(tokens: result.tokens, finishReason: result.finishReason,
                preparationSeconds: result.preparationSeconds,
                timeToFirstTokenSeconds: result.timeToFirstTokenSeconds,
                decodeSeconds: result.decodeSeconds,
                totalSeconds: max(0, elapsed(start) - result.preparationSeconds),
                statistics: result.statistics, phases: result.phases)
        }
    }

    private func checkCancellation(_ original: QwenCancellation?, _ supplied: QwenCancellation?) throws {
        try original?.check(); try supplied?.check()
    }
    private func validatePrepared(_ prepared: QwenPrefillResult, cancellation: QwenCancellation?) throws {
        guard prepared.model === model else {
            throw QwenGenerationError.invalidRequest("prefill handoff belongs to another model instance")
        }
        try validate(prepared.request, cancellation: cancellation)
    }
    private func validatePrefillSession(_ session: QwenPrefillSession, cancellation: QwenCancellation?) throws {
        guard session.model === model else {
            throw QwenGenerationError.invalidRequest("prefill cursor belongs to another model instance")
        }
        guard !session.isActive else { throw QwenGenerationError.busy }
        guard !session.isFinished else { throw QwenGenerationError.invalidRequest("prefill cursor is finished or discarded") }
        try checkCancellation(session.cancellation, cancellation)
        // The immutable request was fully validated when the cursor was made.
    }
    private func validateDecodeSession(_ session: QwenDecodeSession, cancellation: QwenCancellation?) throws {
        guard session.model === model else {
            throw QwenGenerationError.invalidRequest("decode cursor belongs to another model instance")
        }
        guard !session.isActive else { throw QwenGenerationError.busy }
        guard !session.isFinished else { throw QwenGenerationError.invalidRequest("decode cursor is finished or discarded") }
        try checkCancellation(session.cancellation, cancellation)
        // The immutable request was fully validated before claiming the handoff.
    }

    private func makePrefillSession(_ request: QwenGenerationRequest,
                                    cancellation: QwenCancellation?, allowPrefixWait: Bool = true) throws -> QwenPrefillSession {
        // A queued request can reach its first allocation after pressure has
        // escalated. Recheck here, not just at HTTP enqueue. Existing cursors
        // retain their reservation and continue through warning/critical.
        guard memoryPressure?.checkNewRequestAdmission() ?? true else {
            throw QwenGenerationError.resourceLimit("system memory pressure temporarily prevents a new request")
        }
        let bytes = try requestStateReservation(request)
        let reservation = prefixCache.map { $0.reserve(bytes: bytes, kind: .request, model: model) }
            ?? model.stateBudget.reserve(bytes: bytes, kind: .request)
        guard let requestLease = reservation else {
            throw QwenGenerationError.resourceLimit("active requests and cache workspace exhaust the joint byte budget")
        }
        var beganDeviceWork = false
        do {
            var preparationSeconds = 0.0
            var decoder: QwenMTPDecoder?
            if request.mtpDepth > 0 {
                if mtpHead == nil {
                    let start = now(); beganDeviceWork = true
                    mtpHead = try QwenMTP(weights: model.weights, configuration: model.configuration)
                    preparationSeconds = elapsed(start)
                }
                decoder = QwenMTPDecoder(model: model, head: mtpHead!, verification: request.verification,
                    draftHistoryTokens: request.draftHistoryTokens,
                    verificationEvaluateEveryLayers: request.verificationEvaluateEveryLayers)
            }
            try cancellation?.check()
            let p = QwenPrefillProgress(state: model.makeState(), decoder: decoder,
                requestLease: requestLease, preparationSeconds: preparationSeconds, now: now())
            let policy = model.profiler.isRecording ? nil : request.prefixCachePolicy
            // Keep diagnostic coldBoundary callbacks even without a cache
            // instance, so the independent cold oracle sees the same grid.
            p.cacheBoundaries = policy?.publicationBoundaries ?? []
            if let policy, let prefixCache {
                p.cacheFlight = prefixCache.begin(request, policy: policy, model: model,
                    allowWaitingForLeader: allowPrefixWait)
                p.cacheResolved = false
                beganDeviceWork = true
                try resolvePrefix(p, checkCancellation: { try cancellation?.check() })
            }
            try cancellation?.check()
            p.activeSeconds = elapsed(p.startedAt)
            p.lastYieldAt = now()
            let session = QwenPrefillSession(model: model, request: request, cancellation: cancellation, progress: p)
            session.recordProcessed(p.offset)
            return session
        } catch {
            if beganDeviceWork { recover() }
            throw error
        }
    }

    private func resolvePrefix(_ p: QwenPrefillProgress, checkCancellation: () throws -> Void) throws {
        guard !p.cacheResolved, let prefixCache, let flight = p.cacheFlight,
              let result = try prefixCache.resolve(flight, model: model, checkCancellation: checkCancellation) else { return }
        p.cacheResolved = true; p.cacheSource = result.source
        p.cacheLookupSeconds += result.lookupSeconds; p.cacheRestoreSeconds += result.restoreSeconds
        p.cacheWaitSeconds = max(0, Double(now() - flight.startedAt) * 1e-9 - result.lookupSeconds - result.restoreSeconds)
        if let state = result.state {
            p.state = state; p.offset = state.offset; p.cachedTokens = state.offset
            try prefixStateObserver?("restore", state)
        }
    }

    private func prefillStep(_ session: QwenPrefillSession, cancellation: QwenCancellation?,
                             accountSuspension: Bool = true) throws -> QwenPrefillResult? {
        guard let p = session.progress else { throw QwenGenerationError.invalidRequest("prefill cursor is finished") }
        let request = session.request, start = now()
        session.isActive = true
        defer { session.isActive = false }
        if accountSuspension { p.suspensionSeconds += Double(start - p.lastYieldAt) * 1e-9 }
        var beganDeviceWork = false
        do {
            try checkCancellation(session.cancellation, cancellation)
            if !p.cacheResolved {
                beganDeviceWork = true
                try resolvePrefix(p, checkCancellation: { try self.checkCancellation(session.cancellation, cancellation) })
                session.recordProcessed(p.offset)
                guard p.cacheResolved else {
                    p.activeSeconds += elapsed(start); p.lastYieldAt = now()
                    return nil
                }
            }
            if p.prefetch == nil {
                p.prefetch = try model.makePrefillPrefetch(tokens: Array(request.tokens[p.offset...]),
                    chunk: request.prefillChunk, state: p.state)
            }
            let end = p.offset < request.tokens.count - 1
                ? min(request.tokens.count - 1, p.offset + request.prefillChunk) : request.tokens.count
            let targetStart = now()
            beganDeviceWork = true
            let out = try model.forward(tokens: Array(request.tokens[p.offset..<end]), state: &p.state,
                evaluateEveryLayers: request.prefillEvaluateEveryLayers,
                prefillPrefetch: p.prefetch, phase: .prefill,
                prefillAttention: request.prefillAttention,
                profileLogits: end == request.tokens.count,
                prefillMoEConfiguration: request.prefillMoEConfiguration)
            var next: Int32?
            if end == request.tokens.count {
                guard let logits = out.logits else { throw QwenGenerationError.unavailable("missing target logits") }
                let selected = try model.greedyToken(logits)
                try model.evaluate([selected, out.stream], state: &p.state)
                next = try selected.uint32TokenID()
            } else {
                try model.evaluate([out.stream], state: &p.state)
            }
            p.targetSeconds += elapsed(targetStart)
            try checkCancellation(session.cancellation, cancellation)
            try p.decoder?.consumePrompt(stream: out.stream, prompt: request.tokens, offset: p.offset)
            p.ssdWait += out.ssdWaitSeconds; p.ssdBytes += out.ssdLogicalBytes
            p.actualForwardTokens += end - p.offset
            p.offset = end; p.chunks += 1
            session.recordProcessed(end)
            // Preserve the request-local pending PreparedInput and its already
            // computed history. Join only; never rebuild/re-read the lookahead.
            // The complete-stage wrapper retains admission, so its private
            // loop keeps original lookahead overlap until the final chunk.
            if accountSuspension || end == request.tokens.count { p.prefetch?.finish() }
            if end == request.tokens.count {
                try p.decoder?.finishPrompt(expectedTokenCount: request.tokens.count)
            }
            if p.cacheBoundaries.contains(end) {
                // Complete SSD lookahead before publishing. It belongs to the
                // active cursor and is never part of the shared snapshot.
                p.prefetch?.finish()
                try checkCancellation(session.cancellation, cancellation)
                try prefixStateObserver?("coldBoundary", p.state)
                if let prefixCache, let flight = p.cacheFlight {
                    let saveStart = now()
                    try prefixCache.publish(flight, at: end,
                        state: p.state, model: model,
                        checkCancellation: { try self.checkCancellation(session.cancellation, cancellation) },
                        observer: prefixStateObserver)
                    p.cacheSaveSeconds += elapsed(saveStart)
                }
            }
            try checkCancellation(session.cancellation, cancellation)
            p.activeSeconds += elapsed(start)
            p.lastYieldAt = now()
            guard let next else { return nil }
            var stats = QwenPrefillStatistics(promptTokenCount: request.tokens.count, chunkCount: p.chunks,
                targetSeconds: p.targetSeconds, draftHistorySeconds: p.decoder?.statistics.prefillHistorySeconds ?? 0,
                totalSeconds: p.activeSeconds, ssdWaitSeconds: p.ssdWait, ssdLogicalBytes: p.ssdBytes,
                evaluateEveryLayers: request.prefillEvaluateEveryLayers)
            stats.suspensionSeconds = p.suspensionSeconds
            stats.attentionMode = request.prefillAttention.rawValue
            stats.actualForwardTokenCount = p.actualForwardTokens
            stats.recomputedTokenCount = max(0, p.actualForwardTokens - (request.tokens.count - p.cachedTokens))
            if request.prefixCacheMaxTokens != nil || request.prefixCachePlan != nil {
                stats.cachedTokenCount = p.cachedTokens
                stats.computedTokenCount = request.tokens.count - p.cachedTokens
                stats.cacheLookupSeconds = p.cacheLookupSeconds
                stats.cacheRestoreSeconds = p.cacheRestoreSeconds
                stats.cacheSaveSeconds = p.cacheSaveSeconds
                stats.cacheSource = p.cacheSource
                stats.cacheWaitSeconds = p.cacheWaitSeconds
            }
            let result = QwenPrefillResult(model: model, request: request, state: p.state, decoder: p.decoder,
                requestLease: p.requestLease, firstToken: next, statistics: stats, preparationSeconds: p.preparationSeconds,
                requestStartedAt: p.startedAt)
            session.invalidate()
            return result
        } catch {
            p.prefetch?.finish()
            if beganDeviceWork { recover() }
            session.invalidate()
            throw error
        }
    }

    private func runPrefill(_ request: QwenGenerationRequest,
                            cancellation: QwenCancellation?) throws -> QwenPrefillResult {
        let session = try makePrefillSession(request, cancellation: cancellation, allowPrefixWait: false)
        let startupSeconds = session.progress?.activeSeconds ?? 0
        // No gate re-entry. Loop overhead is active service, not a suspension.
        let start = now()
        while true {
            if let result = try prefillStep(session, cancellation: cancellation, accountSuspension: false) {
                var stats = result.statistics
                // The complete-stage API retains its original wall-time scope.
                stats = QwenPrefillStatistics(promptTokenCount: stats.promptTokenCount, chunkCount: stats.chunkCount,
                    targetSeconds: stats.targetSeconds, draftHistorySeconds: stats.draftHistorySeconds,
                    totalSeconds: startupSeconds + elapsed(start), ssdWaitSeconds: stats.ssdWaitSeconds,
                    ssdLogicalBytes: stats.ssdLogicalBytes, evaluateEveryLayers: stats.evaluateEveryLayers,
                    attentionMode: stats.attentionMode, suspensionSeconds: 0,
                    cachedTokenCount: stats.cachedTokenCount, computedTokenCount: stats.computedTokenCount,
                    actualForwardTokenCount: stats.actualForwardTokenCount, recomputedTokenCount: stats.recomputedTokenCount,
                    cacheLookupSeconds: stats.cacheLookupSeconds, cacheRestoreSeconds: stats.cacheRestoreSeconds,
                    cacheSaveSeconds: stats.cacheSaveSeconds, cacheSource: stats.cacheSource, cacheWaitSeconds: stats.cacheWaitSeconds)
                let payload = try result.payload.take()
                return QwenPrefillResult(model: model, request: request, state: payload.state, decoder: payload.decoder,
                    requestLease: payload.requestLease, firstToken: result.firstToken, statistics: stats, preparationSeconds: result.preparationSeconds,
                    requestStartedAt: result.requestStartedAt)
            }
            if session.isWaitingForPrefixCache { Thread.sleep(forTimeInterval: 0.001) }
        }
    }

    private func makeDecodeSession(_ prepared: QwenPrefillResult,
                                   cancellation: QwenCancellation?) throws -> QwenDecodeSession {
        let start = now()
        let wait = Double(start - prepared.readyAt) * 1e-9
        // Every subsequent failure burns this payload; callback error cases do
        // not distinguish pre-admission busy from a busy error thrown by user code.
        let payload = try prepared.payload.take()
        let consumed = elapsed(start)
        return QwenDecodeSession(model: model, request: prepared.request, cancellation: cancellation,
            progress: QwenDecodeProgress(payload: payload, prepared: prepared,
                handoffWaitSeconds: wait, handoffConsumeSeconds: consumed, now: now()))
    }

    private func decodeStep(_ session: QwenDecodeSession, cancellation: QwenCancellation?,
                            onToken: ((Int32) throws -> Void)?,
                            accountSuspension: Bool = true) throws -> QwenGenerationResult? {
        guard let p = session.progress else { throw QwenGenerationError.invalidRequest("decode cursor is finished") }
        let request = session.request, start = now()
        session.isActive = true
        defer { session.isActive = false }
        if accountSuspension { p.suspensionSeconds += Double(start - p.lastYieldAt) * 1e-9 }
        var beganDeviceWork = false
        do {
            try checkCancellation(session.cancellation, cancellation)
            func publish(_ token: Int32) throws {
                try checkCancellation(session.cancellation, cancellation)
                p.generated.append(token); p.next = token
                session.recordGenerated(p.generated.count)
                if let onToken {
                    let callbackStart = now(); try onToken(token)
                    p.callbackSeconds += elapsed(callbackStart)
                }
                try checkCancellation(session.cancellation, cancellation)
            }
            if p.generated.isEmpty {
                p.firstTokenReadySeconds = Double(now() - p.prepared.requestStartedAt) * 1e-9
                try publish(p.next)
            } else {
                let roundStart = now()
                let tokens: [Int32]
                beganDeviceWork = true
                if let decoder = p.decoder {
                    let round = try decoder.next(pending: p.next, state: &p.state, depth: request.mtpDepth,
                        remaining: request.maxTokens - p.generated.count, eos: eosTokenIDs,
                        decodeMode: request.decodeMode,
                        checkCancellation: { try self.checkCancellation(session.cancellation, cancellation) })
                    tokens = round.tokens; p.ssdWait += round.ssdWaitSeconds; p.ssdBytes += round.ssdLogicalBytes
                } else {
                    let out = try model.forward(tokens: [p.next], state: &p.state,
                        decodeMode: request.decodeMode, phase: .decode)
                    guard let logits = out.logits else { throw QwenGenerationError.unavailable("missing target logits") }
                    let selected = try model.greedyToken(logits)
                    try model.evaluate([selected], state: &p.state)
                    tokens = [try selected.uint32TokenID()]
                    p.ssdWait += out.ssdWaitSeconds; p.ssdBytes += out.ssdLogicalBytes
                }
                p.decodeSeconds += elapsed(roundStart); p.decodeRounds += 1
                guard !tokens.isEmpty, tokens.count <= request.maxTokens - p.generated.count else {
                    throw QwenGenerationError.unavailable("decoder returned an invalid output budget")
                }
                for token in tokens {
                    try publish(token)
                    if eosTokenIDs.contains(token) { break }
                }
            }
            try checkCancellation(session.cancellation, cancellation)
            p.activeSeconds += elapsed(start)
            p.lastYieldAt = now()
            guard p.generated.count >= request.maxTokens || eosTokenIDs.contains(p.next) else { return nil }
            let prefill = p.prepared.statistics
            var phases = QwenGenerationPhases(prefill: prefill, handoffWaitSeconds: p.handoffWaitSeconds,
                handoffConsumeSeconds: p.handoffConsumeSeconds, decodeServiceSeconds: p.activeSeconds,
                decodeSSDWaitSeconds: p.ssdWait, decodeSSDLogicalBytes: p.ssdBytes,
                verificationEvaluateEveryLayers: request.verificationEvaluateEveryLayers,
                decodeKernelMode: request.decodeMode.rawValue)
            phases.decodeSuspensionSeconds = p.suspensionSeconds
            let result = QwenGenerationResult(tokens: p.generated,
                finishReason: eosTokenIDs.contains(p.next) ? .eos : .length,
                preparationSeconds: p.prepared.preparationSeconds,
                timeToFirstTokenSeconds: p.firstTokenReadySeconds!, decodeSeconds: p.decodeSeconds,
                totalSeconds: Double(now() - p.prepared.requestStartedAt) * 1e-9,
                statistics: QwenGenerationStatistics(promptTokenCount: request.tokens.count,
                    generatedTokenCount: p.generated.count, decodeRounds: p.decodeRounds,
                    decodedTokenCount: max(0, p.generated.count - 1), prefillChunkCount: prefill.chunkCount,
                    finalStateOffset: p.state.offset, ssdWaitSeconds: prefill.ssdWaitSeconds + p.ssdWait,
                    ssdLogicalBytes: prefill.ssdLogicalBytes + p.ssdBytes, callbackSeconds: p.callbackSeconds,
                    prefillAccumulation: model.prefillAccumulation.rawValue, mtpDepth: request.mtpDepth,
                    mtpVerification: p.decoder?.verification.rawValue, mtp: p.decoder?.statistics), phases: phases)
            session.progress = nil
            return result
        } catch {
            if beganDeviceWork { recover() }
            session.progress = nil
            throw error
        }
    }

    private func runDecode(_ prepared: QwenPrefillResult, cancellation: QwenCancellation?,
                           onToken: ((Int32) throws -> Void)?) throws -> QwenGenerationResult {
        let session = try makeDecodeSession(prepared, cancellation: cancellation)
        let start = now()
        while true {
            if let result = try decodeStep(session, cancellation: cancellation, onToken: onToken,
                                           accountSuspension: false) {
                var phases = result.phases!
                phases = QwenGenerationPhases(prefill: phases.prefill,
                    handoffWaitSeconds: phases.handoffWaitSeconds,
                    handoffConsumeSeconds: phases.handoffConsumeSeconds,
                    decodeServiceSeconds: elapsed(start), decodeSSDWaitSeconds: phases.decodeSSDWaitSeconds,
                    decodeSSDLogicalBytes: phases.decodeSSDLogicalBytes,
                    verificationEvaluateEveryLayers: phases.verificationEvaluateEveryLayers,
                    decodeKernelMode: phases.decodeKernelMode, decodeSuspensionSeconds: 0)
                return QwenGenerationResult(tokens: result.tokens, finishReason: result.finishReason,
                    preparationSeconds: result.preparationSeconds,
                    timeToFirstTokenSeconds: result.timeToFirstTokenSeconds,
                    decodeSeconds: result.decodeSeconds, totalSeconds: result.totalSeconds,
                    statistics: result.statistics, phases: phases)
            }
        }
    }
}
