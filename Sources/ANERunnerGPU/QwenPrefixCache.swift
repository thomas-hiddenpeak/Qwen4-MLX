import ANERunnerCore
import Foundation

/// Bounds retained compact snapshot payload; request-private copies and MLX's
/// allocator pool are not included. Zero disables caching at the service layer.
public struct QwenPrefixCacheLimits: Sendable {
    public let maxEntries: Int
    public let maxBytes: Int
    public let maxKeyTokens: Int
    public init(maxEntries: Int = 8, maxBytes: Int = 536_870_912,
                maxKeyTokens: Int = 1_048_576) {
        self.maxEntries = maxEntries; self.maxBytes = maxBytes
        self.maxKeyTokens = maxKeyTokens
    }
}

/// Confined to the model inference executor. Entries never escape to requests;
/// every restore produces fully evaluated private storage and a new identity.
final class QwenPrefixCache {
    private struct Snapshot { let state: QwenModel.State }
    private let index: QwenPrefixCacheIndex<Snapshot>
    private var published = 0, skippedOversize = 0, restoreFailures = 0
    init(limits: QwenPrefixCacheLimits) throws {
        index = try QwenPrefixCacheIndex(maxEntries: limits.maxEntries,
            maxBytes: limits.maxBytes, maxKeyTokens: limits.maxKeyTokens)
    }
    var statistics: QwenPrefixCacheStatistics {
        var result = index.statistics
        result.published = published; result.skippedOversize = skippedOversize
        result.restoreFailures = restoreFailures
        return result
    }
    func clear(resetStatistics: Bool = false) {
        index.clear(resetStatistics: resetStatistics)
        if resetStatistics { published = 0; skippedOversize = 0; restoreFailures = 0 }
    }
    /// Callback separates the CPU lookup timer from the evaluated copy timer.
    func restore(tokens: [Int32], namespace: String, maximum: Int, model: QwenModel,
                 didLookup: () -> Void) throws -> QwenModel.State? {
        let match = index.lookup(tokens: tokens, namespace: namespace, maxPrefixTokens: maximum)
        didLookup()
        guard let match else { return nil }
        do { return try model.privatePrefixStateCopy(match.value.state) }
        catch { restoreFailures += 1; index.clear(); throw error }
    }
    @discardableResult
    func publish(tokens: [Int32], namespace: String, state: QwenModel.State, model: QwenModel,
                 checkCancellation: () throws -> Void,
                 observer: ((String, QwenModel.State) throws -> Void)?) throws -> Bool {
        let bytes = try model.prefixStatePayloadBytes(state)
        guard bytes <= index.maxBytes, tokens.count <= index.maxKeyTokens else {
            skippedOversize += 1
            return false
        }
        let saved = try model.privatePrefixStateCopy(state)
        try observer?("publish", saved)
        try checkCancellation()
        let inserted = index.insert(tokens: tokens, namespace: namespace,
            value: Snapshot(state: saved), logicalPayloadBytes: bytes)
        if inserted { published += 1 }
        return inserted
    }
}

extension QwenGenerationRequest {
    /// Preserve the exact original chunk schedule; never synthesize a state
    /// at an internal radix node or prefill the final prompt token from cache.
    var prefixCacheBoundary: Int {
        guard mtpDepth == 0, let maximum = prefixCacheMaxTokens, maximum > 0 else { return 0 }
        return min(maximum, tokens.count - 1) / prefillChunk * prefillChunk
    }
    func prefixCacheNamespace(accumulation: String, fusedPrefill: Bool = true) -> String {
        let c = prefillMoEConfiguration
        let groups = c?.threadgroups.keys.sorted().map { "\($0):\(c!.threadgroups[$0]!)" }.joined(separator: ",") ?? "nil"
        return "ar-prefix-v1|chunk=\(prefillChunk)|eval=\(prefillEvaluateEveryLayers)|attention=\(prefillAttention.rawValue)|fused=\(fusedPrefill)|accumulation=\(accumulation)|moe=\(c?.version ?? 0)|groups=\(groups)|gateup=\(c?.gateUpVariant ?? -1)|down=\(c?.groupedDown == true)"
    }
}
