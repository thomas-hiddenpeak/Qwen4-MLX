#if canImport(CoreAI)
import Foundation

/// CPU callers keep token/n-gram history and next-token logits beside this
/// checkpoint. The underlying model retains ownership and tensor validation.
@available(macOS 27.0, *)
public enum CoreAITextSnapshot: Sendable {
    case token(CoreAINativeSnapshot)
    case phase(CoreAIPhaseSnapshot)

    public var offset: Int {
        switch self {
        case .token(let snapshot): return snapshot.offset
        case .phase(let snapshot): return snapshot.offset
        }
    }

    public var logicalByteCount: Int {
        switch self {
        case .token(let snapshot): return snapshot.logicalByteCount
        case .phase(let snapshot): return snapshot.logicalByteCount
        }
    }
}

/// Selects exactly one CoreAI text execution path. A phase manifest loads only
/// the phase model, avoiding simultaneous copies of native and phase weights.
/// Model state remains single-owner and non-Sendable, like both implementations.
@available(macOS 27.0, *)
public final class CoreAITextRuntime {
    private enum Backend {
        case token(CoreAINativeModel)
        case phase(CoreAIPhaseModel)
    }
    private let backend: Backend

    public init(attentionManifest: URL, denseManifest: URL, moeManifest: URL,
                pdManifest: URL? = nil, computeUnits: CoreAIComputeUnits = .gpu) async throws {
        if let pdManifest {
            backend = .phase(try await CoreAIPhaseModel(manifestURL: pdManifest, computeUnits: computeUnits))
        } else {
            backend = .token(try await CoreAINativeModel(attentionManifest: attentionManifest,
                denseManifest: denseManifest, moeManifest: moeManifest, computeUnits: computeUnits))
        }
    }

    public var capacity: Int {
        switch backend {
        case .token(let model): return model.capacity
        case .phase(let model): return model.capacity
        }
    }

    public var vocabularySize: Int {
        switch backend {
        case .token(let model): return model.vocabularySize
        case .phase(let model): return model.vocabularySize
        }
    }

    public var manifestModelDirectory: URL {
        switch backend {
        case .token(let model): return model.manifestModelDirectory
        case .phase(let model): return model.manifestModelDirectory
        }
    }

    public var sourceConfigSHA256: String {
        switch backend {
        case .token(let model): return model.sourceConfigSHA256
        case .phase(let model): return model.sourceConfigSHA256
        }
    }

    public var modelLoadMilliseconds: Double {
        switch backend {
        case .token(let model): return model.modelLoadMilliseconds
        case .phase(let model): return model.modelLoadMilliseconds
        }
    }

    public var offset: Int {
        switch backend {
        case .token(let model): return model.offset
        case .phase(let model): return model.offset
        }
    }

    public var valid: Bool {
        switch backend {
        case .token(let model): return model.valid
        case .phase(let model): return model.valid
        }
    }

    public var successfulCalls: Int {
        switch backend {
        case .token(let model): return model.successfulCalls
        case .phase(let model): return model.successfulCalls
        }
    }

    public var callCounts: [String: Int] {
        switch backend {
        case .token(let model): return model.callCounts
        case .phase(let model): return model.callCounts
        }
    }

    public var predictionMillisecondsByGroup: [String: Double] {
        switch backend {
        case .token(let model): return model.predictionMillisecondsByGroup
        case .phase(let model): return model.predictionMillisecondsByGroup
        }
    }

    /// Fused-layer attribution is available for the phase backend. These times
    /// overlap the group totals and must not be added to them.
    public var predictionMillisecondsByLayer: [String: Double] {
        switch backend {
        case .token: return [:]
        case .phase(let model): return model.predictionMillisecondsByLayer
        }
    }

    public var prefillChunkSize: Int {
        switch backend {
        case .token: return 1
        case .phase(let model): return model.tokenChunk
        }
    }

    /// Available functions in descending order; every backend includes S1.
    public var supportedPrefillChunks: [Int] {
        switch backend {
        case .token: return [1]
        case .phase(let model): return model.supportedPrefillChunks
        }
    }

    /// Zero selects the primary chunk. Any exported size may be the chunk limit;
    /// smaller exported functions handle the remaining unpadded tail.
    public func resolvedPrefillChunkSize(requested: Int) throws -> Int {
        if requested == 0 { return prefillChunkSize }
        guard supportedPrefillChunks.contains(requested) else {
            throw CoreAIBlockRunnerError.invalidFixture(
                "prefill-chunk must be 0 (automatic) or an exported chunk from \(supportedPrefillChunks)")
        }
        return requested
    }

    /// Selects an existing function without padding or crossing a caller-owned
    /// boundary, such as the end of a system prefix that must be checkpointed.
    public func nextPrefillChunkSize(remaining: Int, limit: Int, boundary: Int? = nil) throws -> Int {
        let available = min(remaining, min(limit, boundary ?? remaining))
        guard remaining > 0, limit > 0, available > 0,
              let count = supportedPrefillChunks.first(where: { $0 <= available }) else {
            throw CoreAIBlockRunnerError.invalidFixture("No nonempty prefill chunk fits the remaining tokens and boundary")
        }
        return count
    }

    public var usesIndependentPhases: Bool {
        if case .phase = backend { return true }
        return false
    }

    /// Always executes the S1/decode function, including a final prefill tail.
    public func forward(token: Int32, pleEmbedding: [Float]) async throws -> [Float] {
        switch backend {
        case .token(let model): return try await model.forward(token: token, pleEmbedding: pleEmbedding)
        case .phase(let model): return try await model.forward(tokens: [token], pleEmbedding: pleEmbedding)
        }
    }

    /// Exactly one exported prefill chunk. PLE rows are token-major and pass
    /// through unchanged. The caller owns tail scheduling; no padding or hidden
    /// token loop changes the model offset or the reported phase call counts.
    public func prefill(tokens: [Int32], pleEmbedding: [Float]) async throws -> [Float] {
        guard supportedPrefillChunks.contains(tokens.count) else {
            throw CoreAIBlockRunnerError.invalidFixture(
                "CoreAI prefill requires one exported chunk from \(supportedPrefillChunks)")
        }
        switch backend {
        case .token(let model): return try await model.forward(token: tokens[0], pleEmbedding: pleEmbedding)
        case .phase(let model): return try await model.forward(tokens: tokens, pleEmbedding: pleEmbedding)
        }
    }

    public func checkpoint() throws -> CoreAITextSnapshot {
        switch backend {
        case .token(let model): return .token(try model.checkpoint())
        case .phase(let model): return .phase(try model.checkpoint())
        }
    }

    public func restore(_ snapshot: CoreAITextSnapshot) throws {
        switch (backend, snapshot) {
        case (.token(let model), .token(let saved)): try model.restore(saved)
        case (.phase(let model), .phase(let saved)): try model.restore(saved)
        default:
            throw CoreAIBlockRunnerError.invalidFixture(
                "Cannot restore a checkpoint from a different CoreAI execution path")
        }
    }

    public func reset() throws {
        switch backend {
        case .token(let model): try model.reset()
        case .phase(let model): try model.reset()
        }
    }

    public func stateByteCount() throws -> Int {
        switch backend {
        case .token(let model): return try model.stateByteCount()
        case .phase(let model): return try model.stateByteCount()
        }
    }
}
#endif
