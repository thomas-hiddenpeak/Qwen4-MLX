import ANERunnerCore
import CryptoKit
import Foundation

/// Logical snapshot retention, separately bounded by the model's joint ledger.
public struct QwenPrefixCacheLimits: Sendable {
    public let maxEntries, maxBytes, maxKeyTokens: Int
    public let ttlSeconds: TimeInterval?
    public init(maxEntries: Int = 8, maxBytes: Int = 536_870_912,
                maxKeyTokens: Int = 1_048_576, ttlSeconds: TimeInterval? = nil) {
        self.maxEntries = maxEntries; self.maxBytes = maxBytes
        self.maxKeyTokens = maxKeyTokens; self.ttlSeconds = ttlSeconds
    }
}

/// Only host Data and the thread-safe budget lease cross the I/O boundary.
final class QwenPrefixDiskRead: @unchecked Sendable {
    private let lock = NSLock()
    private var ready = false
    private var value: QwenPrefixDiskMatch?
    let lease: QwenStateBudget.Lease
    let startedAt = DispatchTime.now().uptimeNanoseconds
    init(lease: QwenStateBudget.Lease) { self.lease = lease }
    func complete(_ value: QwenPrefixDiskMatch?) {
        lock.lock(); defer { lock.unlock() }
        self.value = value; ready = true
    }
    var isReady: Bool { lock.lock(); defer { lock.unlock() }; return ready }
    func take() -> QwenPrefixDiskMatch? {
        lock.lock(); defer { lock.unlock() }
        let result = value; value = nil
        return result
    }
}

/// Completion has no reference to Tensor/cache/index objects.
private final class QwenPrefixDiskPublication: @unchecked Sendable {
    private let lock = NSLock()
    private var complete = false
    var isComplete: Bool { lock.lock(); defer { lock.unlock() }; return complete }
    func finish() { lock.lock(); complete = true; lock.unlock() }
}

/// A leader owns no other request. Dropping/cancelling it releases the flight;
/// followers may then take over at the next scheduler slice.
final class QwenPrefixCacheFlight {
    weak var cache: QwenPrefixCache?
    let key: String, namespace: String, tokens: [Int32], identity = UUID(), epoch: UInt64
    let allowWaitingForLeader: Bool
    var leader = false
    var waiting = false
    var read: QwenPrefixDiskRead?
    var resolved = false
    let startedAt = DispatchTime.now().uptimeNanoseconds
    init(cache: QwenPrefixCache, key: String, namespace: String, tokens: [Int32], epoch: UInt64, allowWaitingForLeader: Bool) {
        self.allowWaitingForLeader = allowWaitingForLeader
        self.cache = cache; self.key = key; self.namespace = namespace; self.tokens = tokens; self.epoch = epoch
    }
    deinit { cache?.releaseFlight(key: key, identity: identity) }
}

/// All Tensor/index operations remain on the model inference executor. The
/// optional disk store exclusively handles immutable host archives.
final class QwenPrefixCache {
    private struct Snapshot {
        let state: QwenModel.State
        let lease: QwenStateBudget.Lease
        let createdAt: TimeInterval
    }
    private let index: QwenPrefixCacheIndex<Snapshot>
    private let ttlSeconds: TimeInterval?
    let disk: QwenPrefixDiskStore?
    let diskIdentity: String
    private var published = 0, skippedOversize = 0, restoreFailures = 0
    private var restoredHits = 0, diskHits = 0, diskFallbacks = 0
    private var pressureEvictions = 0, budgetSkipped = 0, duplicateSkipped = 0, flightWaits = 0, expired = 0
    private var flights: [String: UUID] = [:]
    private var publications: [String: QwenPrefixDiskPublication] = [:]
    private var epoch: UInt64 = 0

    init(limits: QwenPrefixCacheLimits, disk: QwenPrefixDiskStore?, model: QwenModel) throws {
        if let ttl = limits.ttlSeconds, !ttl.isFinite || ttl <= 0 { throw GPUError.invalid("Invalid prefix cache TTL") }
        index = try QwenPrefixCacheIndex(maxEntries: limits.maxEntries,
            maxBytes: limits.maxBytes, maxKeyTokens: limits.maxKeyTokens)
        ttlSeconds = limits.ttlSeconds; self.disk = disk
        diskIdentity = try disk == nil ? "memory" : QwenPrefixCacheIdentity.fingerprint(modelDirectory: model.configuration.modelDirectory)
    }
    var statistics: QwenPrefixCacheStatistics {
        var r = index.statistics
        r.published = published; r.skippedOversize = skippedOversize; r.restoreFailures = restoreFailures
        r.restoredHits = restoredHits; r.diskHits = diskHits; r.diskFallbacks = diskFallbacks
        r.pressureEvictions = pressureEvictions; r.budgetSkipped = budgetSkipped
        r.duplicateSkipped = duplicateSkipped; r.flightWaits = flightWaits; r.liveFlights = flights.count; r.expired = expired
        return r
    }
    func clear(resetStatistics: Bool = false, includingDisk: Bool = false) {
        epoch &+= 1; flights.removeAll(); publications.removeAll(); index.clear(resetStatistics: resetStatistics)
        if includingDisk { disk?.clear(resetStatistics: resetStatistics) }
        if resetStatistics {
            published = 0; skippedOversize = 0; restoreFailures = 0; restoredHits = 0; diskHits = 0
            diskFallbacks = 0; pressureEvictions = 0; budgetSkipped = 0; duplicateSkipped = 0; flightWaits = 0; expired = 0
        }
    }
    /// Evict retained snapshots before denying optional workspace or a request.
    func reserve(bytes: Int, kind: QwenStateBudget.Kind, model: QwenModel) -> QwenStateBudget.Lease? {
        while true {
            if let lease = model.stateBudget.reserve(bytes: bytes, kind: kind) { return lease }
            guard index.evictLeastRecentlyUsed() else { budgetSkipped += 1; return nil }
            pressureEvictions += 1
        }
    }
    func trimMemory() {
        while index.evictLeastRecentlyUsed() { pressureEvictions += 1 }
    }
    private func namespace(_ request: QwenGenerationRequest, model: QwenModel) -> String {
        diskIdentity + "|" + request.prefixCacheNamespace(accumulation: model.prefillAccumulation.rawValue,
            fusedPrefill: model.fusedPrefillEnabled)
    }
    func begin(_ request: QwenGenerationRequest, maximum: Int, model: QwenModel, allowWaitingForLeader: Bool = true) -> QwenPrefixCacheFlight {
        publications = publications.filter { !$0.value.isComplete }
        let tokens = Array(request.tokens.prefix(maximum)), ns = namespace(request, model: model)
        var h = SHA256(); h.update(data: Data(ns.utf8))
        tokens.withUnsafeBytes { h.update(bufferPointer: $0) }
        let key = h.finalize().map { String(format: "%02x", $0) }.joined()
        return QwenPrefixCacheFlight(cache: self, key: key, namespace: ns, tokens: tokens, epoch: epoch, allowWaitingForLeader: allowWaitingForLeader)
    }
    func releaseFlight(key: String, identity: UUID) {
        if flights[key] == identity { flights.removeValue(forKey: key) }
    }
    private func current(_ f: QwenPrefixCacheFlight) -> Bool { f.epoch == epoch }
    private func memoryMatch(_ f: QwenPrefixCacheFlight) -> QwenPrefixCacheMatch<Snapshot>? {
        while let match = index.peek(tokens: f.tokens, namespace: f.namespace) {
            if let ttlSeconds, Date().timeIntervalSince1970 - match.value.createdAt >= ttlSeconds {
                index.remove(tokens: Array(f.tokens.prefix(match.prefixTokenCount)), namespace: f.namespace)
                expired += 1
                continue
            }
            break
        }
        return index.lookup(tokens: f.tokens, namespace: f.namespace)
    }
    private func recoverOptionalFailure(model: QwenModel, error: Error) throws {
        if error as? QwenGenerationError == .cancelled { throw error }
        do { try MX.synchronize() }
        catch {
            model.failGenerationRecovery(String(describing: error))
            throw QwenGenerationError.unavailable("Prefix device recovery failed")
        }
    }

    struct Resolution {
        let state: QwenModel.State?
        let source: String
        let lookupSeconds, restoreSeconds: Double
    }
    /// Nil means asynchronous I/O or another prefix leader is still pending.
    /// No GPU work is submitted for a waiting follower.
    func resolve(_ f: QwenPrefixCacheFlight, model: QwenModel,
                 checkCancellation: () throws -> Void) throws -> Resolution? {
        let started = DispatchTime.now().uptimeNanoseconds
        func elapsed(_ start: UInt64) -> Double { Double(DispatchTime.now().uptimeNanoseconds - start) * 1e-9 }
        try checkCancellation()
        guard current(f) else { f.read = nil; f.resolved = true; return .init(state: nil, source: "cold", lookupSeconds: 0, restoreSeconds: 0) }
        if let read = f.read {
            guard read.isReady else { return nil }
            let match = read.take()
            f.read = nil
            defer { read.lease.release() }
            if let match {
                do {
                    let archive = QwenPrefixStateArchive(metadata: match.metadata, payload: match.payload,
                        logicalPayloadBytes: try model.estimatedPrefixStateBytes(at: match.prefixTokenCount))
                    let restored = try model.importPrefixState(archive, expectedOffset: match.prefixTokenCount,
                        checkCancellation: checkCancellation)
                    try checkCancellation()
                    // The imported arrays belong to the request. Promotion uses
                    // a separate evaluated copy and its own retained lease.
                    _ = try saveMemory(tokens: Array(f.tokens.prefix(match.prefixTokenCount)), namespace: f.namespace,
                        state: restored, model: model, checkCancellation: checkCancellation, observer: nil)
                    diskHits += 1; restoredHits += 1; f.resolved = true
                    if match.prefixTokenCount == f.tokens.count { releaseFlight(key: f.key, identity: f.identity) }
                    return .init(state: restored, source: "disk", lookupSeconds: 0, restoreSeconds: elapsed(started))
                } catch {
                    try recoverOptionalFailure(model: model, error: error)
                    _ = disk?.invalidateAsync(tokens: Array(f.tokens.prefix(match.prefixTokenCount)), namespace: f.namespace)
                    restoreFailures += 1
                }
            }
            diskFallbacks += 1; f.resolved = true
            return .init(state: nil, source: "cold", lookupSeconds: 0, restoreSeconds: elapsed(started))
        }
        if let owner = flights[f.key], owner != f.identity {
            // Complete-stage calls retain the inference gate and cannot wait
            // for an externally paused producer. Their safe fallback is cold.
            if !f.allowWaitingForLeader {
                f.resolved = true
                return .init(state: nil, source: "cold", lookupSeconds: elapsed(started), restoreSeconds: 0)
            }
            if !f.waiting { flightWaits += 1; f.waiting = true }
            return nil
        }
        if let publication = publications[f.key] {
            if publication.isComplete { publications.removeValue(forKey: f.key) }
            else if f.allowWaitingForLeader,
                    index.peek(tokens: f.tokens, namespace: f.namespace)?.prefixTokenCount != f.tokens.count {
                if !f.waiting { flightWaits += 1; f.waiting = true }
                return nil
            }
        }
        if let match = memoryMatch(f) {
            let lookup = elapsed(started), restoreStart = DispatchTime.now().uptimeNanoseconds
            do {
                let restored = try model.privatePrefixStateCopy(match.value.state)
                try checkCancellation()
                restoredHits += 1; f.resolved = true
                // A shorter hit still leads creation of the requested boundary.
                if match.prefixTokenCount < f.tokens.count { f.leader = true; flights[f.key] = f.identity }
                return .init(state: restored, source: "memory", lookupSeconds: lookup, restoreSeconds: elapsed(restoreStart))
            } catch {
                try recoverOptionalFailure(model: model, error: error)
                index.remove(tokens: Array(f.tokens.prefix(match.prefixTokenCount)), namespace: f.namespace)
                restoreFailures += 1
            }
        }
        f.leader = true; flights[f.key] = f.identity
        if let disk, let summary = disk.peek(tokens: f.tokens, namespace: f.namespace),
           summary.metadataBytes >= 0, summary.payloadBytes <= (Int.max - summary.metadataBytes) / 2,
           let lease = reserve(bytes: summary.payloadBytes * 2 + summary.metadataBytes, kind: .workspace, model: model) {
            let read = QwenPrefixDiskRead(lease: lease)
            if disk.lookupAsync(tokens: f.tokens, namespace: f.namespace, completion: { read.complete($0) }) {
                f.read = read
                return nil
            }
            lease.release(); diskFallbacks += 1
        }
        f.resolved = true
        return .init(state: nil, source: "cold", lookupSeconds: elapsed(started), restoreSeconds: 0)
    }

    private func saveMemory(tokens: [Int32], namespace: String, state: QwenModel.State, model: QwenModel,
                            checkCancellation: () throws -> Void,
                            observer: ((String, QwenModel.State) throws -> Void)?) throws -> Bool {
        if let existing = index.peek(tokens: tokens, namespace: namespace), existing.prefixTokenCount == tokens.count {
            duplicateSkipped += 1
            return false
        }
        let bytes = try model.prefixStatePayloadBytes(state)
        guard bytes <= index.maxBytes, tokens.count <= index.maxKeyTokens else { skippedOversize += 1; return false }
        guard let lease = reserve(bytes: bytes, kind: .cache, model: model) else { return false }
        let saved: QwenModel.State
        do { saved = try model.privatePrefixStateCopy(state) }
        catch { lease.release(); try recoverOptionalFailure(model: model, error: error); budgetSkipped += 1; return false }
        try observer?("publish", saved)
        try checkCancellation()
        let inserted = index.insert(tokens: tokens, namespace: namespace,
            value: Snapshot(state: saved, lease: lease, createdAt: Date().timeIntervalSince1970), logicalPayloadBytes: bytes)
        if inserted { published += 1 }
        return inserted
    }
    func publish(_ f: QwenPrefixCacheFlight, state: QwenModel.State, model: QwenModel,
                 checkCancellation: () throws -> Void,
                 observer: ((String, QwenModel.State) throws -> Void)?) throws {
        defer { releaseFlight(key: f.key, identity: f.identity) }
        guard current(f) else { return }
        _ = try saveMemory(tokens: f.tokens, namespace: f.namespace, state: state, model: model,
            checkCancellation: checkCancellation, observer: observer)
        guard let disk, disk.peek(tokens: f.tokens, namespace: f.namespace)?.prefixTokenCount != f.tokens.count else { return }
        let bytes = try model.prefixStatePayloadBytes(state)
        guard bytes <= disk.limits.maxPendingBytes, bytes <= (Int.max - QwenPrefixStateArchiveDescriptor.maximumMetadataBytes) / 2,
              let lease = reserve(bytes: bytes * 2 + QwenPrefixStateArchiveDescriptor.maximumMetadataBytes, kind: .workspace, model: model) else { return }
        do {
            let archive = try model.exportPrefixState(state, maxPayloadBytes: disk.limits.maxPendingBytes,
                checkCancellation: checkCancellation)
            try checkCancellation()
            guard current(f) else { lease.release(); return }
            let publication = QwenPrefixDiskPublication()
            publications = publications.filter { !$0.value.isComplete }
            if disk.enqueue(tokens: f.tokens, namespace: f.namespace, metadata: archive.metadata,
                payload: archive.payload, completion: { _ in lease.release(); publication.finish() }) {
                publications[f.key] = publication
            } else { lease.release() }
        } catch {
            lease.release()
            try recoverOptionalFailure(model: model, error: error)
            diskFallbacks += 1
        }
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
