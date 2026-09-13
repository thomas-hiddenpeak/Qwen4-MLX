import ANERunnerCore
import CryptoKit
import Foundation

/// Logical snapshot retention, separately bounded by the model's joint ledger.
public struct QwenPrefixCacheLimits: Sendable {
    public let maxEntries, maxBytes, maxKeyTokens: Int
    public let ttlSeconds: TimeInterval?
    /// Maximum time a request waits for an unfinished SSD read or publication.
    /// The I/O owner retains workspace until actual completion after a timeout.
    public let diskRestoreTimeoutSeconds: TimeInterval
    public init(maxEntries: Int = 8, maxBytes: Int = 536_870_912,
                maxKeyTokens: Int = 1_048_576, ttlSeconds: TimeInterval? = nil,
                diskRestoreTimeoutSeconds: TimeInterval = 5) {
        self.maxEntries = maxEntries; self.maxBytes = maxBytes
        self.maxKeyTokens = maxKeyTokens; self.ttlSeconds = ttlSeconds
        self.diskRestoreTimeoutSeconds = diskRestoreTimeoutSeconds
    }
}

/// Only host Data and the thread-safe budget lease cross the I/O boundary.
final class QwenPrefixDiskRead: @unchecked Sendable {
    private let lock = NSLock()
    private var ready = false
    private var value: QwenPrefixDiskMatch?
    let lease: QwenStateBudget.Lease
    let startedAt: UInt64
    init(lease: QwenStateBudget.Lease, startedAt: UInt64 = DispatchTime.now().uptimeNanoseconds) {
        self.lease = lease; self.startedAt = startedAt
    }
    func complete(_ value: QwenPrefixDiskMatch?) {
        lock.lock(); defer { lock.unlock() }
        guard !ready else { return }
        self.value = value; ready = true
    }
    var isReady: Bool { lock.lock(); defer { lock.unlock() }; return ready }
    func hasTimedOut(after seconds: TimeInterval, now: UInt64 = DispatchTime.now().uptimeNanoseconds) -> Bool {
        lock.lock(); defer { lock.unlock() }
        // A completion that arrived while the executor was busy remains usable.
        // Uptime never decreases; accepting an earlier value in a CPU test must
        // not underflow and manufacture an expired transfer.
        return !ready && Self.waitHasTimedOut(startedAt: startedAt, after: seconds, now: now)
    }
    /// One timestamp spans metadata admission, an identical read's fence and
    /// accepted IO. A ready result is handled separately and remains usable.
    static func waitHasTimedOut(startedAt: UInt64, after seconds: TimeInterval,
                                now: UInt64 = DispatchTime.now().uptimeNanoseconds) -> Bool {
        now >= startedAt && Double(now - startedAt) * 1e-9 >= seconds
    }
    /// Convert the already validated duration to the same attempt's absolute
    /// uptime deadline. Round positive sub-nanosecond waits up; finite durations
    /// beyond the representable clock saturate instead of trapping or wrapping.
    static func waitDeadline(startedAt: UInt64, after seconds: TimeInterval) -> UInt64? {
        guard seconds.isFinite, seconds > 0 else { return nil }
        let nanoseconds = (seconds * 1_000_000_000).rounded(.up)
        let remaining = UInt64.max - startedAt
        guard nanoseconds < Double(remaining) else { return UInt64.max }
        return startedAt + UInt64(nanoseconds)
    }
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

/// The read job and its request consumer finish independently. This fence
/// owns no Data/Tensor/lease: it only prevents another identical read while
/// the first import/promotion or an abandoned POSIX operation is unfinished.
final class QwenPrefixDiskReadFence: @unchecked Sendable {
    private let lock = NSLock()
    private var ioFinished = false, consumerFinished = false
    var isComplete: Bool {
        lock.lock(); defer { lock.unlock() }
        return ioFinished && consumerFinished
    }
    func finishIO() { lock.lock(); ioFinished = true; lock.unlock() }
    func finishConsumer() { lock.lock(); consumerFinished = true; lock.unlock() }
}

/// Executor-confined ownership. An old request cannot release a new owner
/// after clear, cancellation takeover, or a different checkpoint publication.
final class QwenPrefixProducerRegistry {
    private var owners: [String: UUID] = [:]
    var count: Int { owners.count }
    func owner(for key: String) -> UUID? { owners[key] }
    @discardableResult
    func claim(key: String, identity: UUID) -> Bool {
        if let existing = owners[key] { return existing == identity }
        owners[key] = identity
        return true
    }
    func release(key: String, identity: UUID) {
        if owners[key] == identity { owners.removeValue(forKey: key) }
    }
    func removeAll() { owners.removeAll() }
}

struct QwenPrefixCheckpoint: Equatable {
    let key: String, boundary: Int
    init(tokens: [Int32], namespace: String, boundary: Int) {
        self.boundary = boundary
        var h = SHA256(); h.update(data: Data(namespace.utf8))
        Array(tokens.prefix(boundary)).withUnsafeBytes { h.update(bufferPointer: $0) }
        key = h.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

/// Immutable cache provenance and the independently admitted attachment
/// metadata. A request retains this same wrapper through a paused prefill or
/// handoff, so evicting its dense cache entry cannot release admission early.
final class QwenPrefixPagedAttachment {
    let prefix: QwenPagedKVPrefix
    let namespace, executionNamespace: String
    let checkpoint: QwenPrefixCheckpoint
    let metadataBytes: Int
    private let metadataLease: QwenStateBudget.Lease

    fileprivate static func metadataCharge(offset: Int, attentionLayers: Int,
                                           namespace: String, executionNamespace: String) throws -> Int {
        // Token keys belong to the bounded radix index; this wrapper stores
        // only their digest. Account its strings separately from page metadata.
        let namespaceBytes = namespace.utf8.count, executionBytes = executionNamespace.utf8.count
        guard namespaceBytes > 0, namespaceBytes <= 65_536,
              executionBytes > 0, executionBytes <= 65_536 else {
            throw GPUError.invalid("Paged cache attachment namespace exceeds its binding allowance")
        }
        return try QwenPagedKVPrefix.estimatedMetadataBytes(offset: offset, attentionLayerCount: attentionLayers)
            + namespaceBytes + executionBytes + 64 + 4_096
    }

    fileprivate init(prefix: QwenPagedKVPrefix, metadataLease: QwenStateBudget.Lease,
                     namespace: String, executionNamespace: String, checkpoint: QwenPrefixCheckpoint) throws {
        let bytes = try Self.metadataCharge(offset: prefix.offset, attentionLayers: prefix.attentionStates.count,
            namespace: namespace, executionNamespace: executionNamespace)
        guard checkpoint.boundary == prefix.offset, checkpoint.key.utf8.count == 64,
              metadataLease.kind == .cache, !metadataLease.isReleased, metadataLease.bytes == bytes else {
            throw GPUError.invalid("Paged cache attachment has inconsistent binding or metadata admission")
        }
        self.prefix = prefix; self.metadataLease = metadataLease; self.metadataBytes = bytes
        self.namespace = namespace; self.executionNamespace = executionNamespace; self.checkpoint = checkpoint
    }

    /// Host-only identity check. A generator supplies its execution namespace
    /// and context; a cache additionally supplies its complete disk namespace.
    /// Invalid optional attachment metadata is ignored without evicting dense KV.
    func matches(tokens: [Int32], namespace expectedNamespace: String? = nil,
                 executionNamespace expectedExecution: String, context: QwenPagedKVContext) -> Bool {
        guard prefix.context === context, executionNamespace == expectedExecution,
              expectedNamespace == nil || namespace == expectedNamespace,
              prefix.offset > 0, prefix.offset <= tokens.count, checkpoint.boundary == prefix.offset,
              !metadataLease.isReleased, metadataLease.bytes == metadataBytes,
              checkpoint == QwenPrefixCheckpoint(tokens: tokens, namespace: namespace, boundary: prefix.offset) else {
            return false
        }
        do {
            try prefix.validate(modelOwner: context.modelOwner, context: context, layerIndices: context.pools.keys.sorted())
            return true
        } catch { return false }
    }
}

/// An optional RAM attachment failure must not be interpreted as a corrupt
/// SSD archive by the surrounding restore fallback. Capacity is preflighted;
/// unexpected writer/device errors propagate after synchronization.
private struct QwenPrefixPagedAttachmentFailure: Error, CustomStringConvertible {
    let cause: String
    var description: String { "Paged prefix attachment failed: " + cause }
}

/// One request lookup context survives both checkpoint publications. Only
/// ownedProducer is a computation flight; full lookup tokens are never its key.
final class QwenPrefixCacheFlight {
    weak var cache: QwenPrefixCache?
    let namespace: String, tokens: [Int32], identity = UUID(), epoch: UInt64
    let checkpoints: [QwenPrefixCheckpoint]
    let prefillChunk: Int
    let executionNamespace: String
    let systemProducerBoundary: Int?
    let allowWaitingForLeader: Bool
    var ownedProducer: QwenPrefixCheckpoint?
    var waiting = false
    var read: QwenPrefixDiskRead?
    var readCheckpoint: QwenPrefixCheckpoint?
    var readFence: QwenPrefixDiskReadFence?
    var readAdmissionIntent: QwenPrefixDiskReadIntent?
    var readWaitStartedAt: UInt64?
    var waitedForRead = false
    var skipDisk = false
    var publicationWaitKey: String?
    var publicationWaitStartedAt: UInt64?
    var timedOutPublications: Set<String> = []
    var resolved = false
    var lookupSeconds = 0.0, restoreSeconds = 0.0
    let startedAt = DispatchTime.now().uptimeNanoseconds
    init(cache: QwenPrefixCache, namespace: String, tokens: [Int32],
         checkpoints: [QwenPrefixCheckpoint], prefillChunk: Int, systemProducerBoundary: Int?,
         epoch: UInt64, allowWaitingForLeader: Bool, executionNamespace: String = "") {
        self.allowWaitingForLeader = allowWaitingForLeader
        self.cache = cache; self.namespace = namespace; self.tokens = tokens; self.epoch = epoch
        self.checkpoints = checkpoints; self.prefillChunk = prefillChunk
        self.systemProducerBoundary = systemProducerBoundary
        self.executionNamespace = executionNamespace
    }
    func detachRead() {
        readFence?.finishConsumer()
        readFence = nil; read = nil; readCheckpoint = nil
    }
    deinit {
        readAdmissionIntent?.release()
        cache?.releaseReadAdmission(identity: identity)
        // The callback still owns read Data/workspace after an abandonment.
        detachRead()
        if let ownedProducer { cache?.releaseFlight(key: ownedProducer.key, identity: identity) }
    }
}

/// All Tensor/index operations remain on the model inference executor. The
/// optional disk store exclusively handles immutable host archives.
final class QwenPrefixCache {
    private struct Snapshot {
        let state: QwenModel.State
        let lease: QwenStateBudget.Lease
        let createdAt: TimeInterval
        let pagedAttachment: QwenPrefixPagedAttachment?
    }
    private let index: QwenPrefixCacheIndex<Snapshot>
    private let ttlSeconds: TimeInterval?
    private let diskRestoreTimeoutSeconds: TimeInterval
    private let memoryPressure: QwenMemoryPressurePolicy?
    private let pagedKVContext: QwenPagedKVContext?
    let disk: QwenPrefixDiskStore?
    let diskIdentity: String
    private var published = 0, skippedOversize = 0, restoreFailures = 0
    private var restoredHits = 0, diskHits = 0, diskFallbacks = 0
    private var diskReadTimeouts = 0
    private var diskPublicationTimeouts = 0
    private var pressureEvictions = 0, budgetSkipped = 0, duplicateSkipped = 0, flightWaits = 0, expired = 0
    private var retainedSystemAnchorSkips = 0, restoreWaits = 0
    private let flights = QwenPrefixProducerRegistry()
    private var publications: [String: QwenPrefixDiskPublication] = [:]
    private var reads: [String: QwenPrefixDiskReadFence] = [:]
    // Strong metadata handles let RAM-only clear revoke this cache's intents
    // even if their paused request cursors remain alive elsewhere.
    private var readAdmissionIntents: [UUID: QwenPrefixDiskReadIntent] = [:]
    private var epoch: UInt64 = 0

    init(limits: QwenPrefixCacheLimits, disk: QwenPrefixDiskStore?, model: QwenModel,
         memoryPressure: QwenMemoryPressurePolicy? = nil,
         pagedKVContext: QwenPagedKVContext? = nil) throws {
        if let ttl = limits.ttlSeconds, !ttl.isFinite || ttl <= 0 { throw GPUError.invalid("Invalid prefix cache TTL") }
        guard limits.diskRestoreTimeoutSeconds.isFinite, limits.diskRestoreTimeoutSeconds > 0 else {
            throw GPUError.invalid("Invalid prefix SSD restore timeout")
        }
        index = try QwenPrefixCacheIndex(maxEntries: limits.maxEntries,
            maxBytes: limits.maxBytes, maxKeyTokens: limits.maxKeyTokens)
        ttlSeconds = limits.ttlSeconds; self.disk = disk
        self.pagedKVContext = pagedKVContext
        if let pagedKVContext { try model.validatePagedKVContext(pagedKVContext) }
        diskRestoreTimeoutSeconds = limits.diskRestoreTimeoutSeconds; self.memoryPressure = memoryPressure
        diskIdentity = try disk == nil ? "memory" : QwenPrefixCacheIdentity.fingerprint(modelDirectory: model.configuration.modelDirectory)
    }
    deinit { for intent in readAdmissionIntents.values { intent.release() } }
    var statistics: QwenPrefixCacheStatistics {
        var r = index.statistics
        r.published = published; r.skippedOversize = skippedOversize; r.restoreFailures = restoreFailures
        r.restoredHits = restoredHits; r.diskHits = diskHits; r.diskFallbacks = diskFallbacks
        r.diskReadTimeouts = diskReadTimeouts
        r.diskPublicationTimeouts = diskPublicationTimeouts
        r.pressureEvictions = pressureEvictions; r.budgetSkipped = budgetSkipped
        r.duplicateSkipped = duplicateSkipped; r.flightWaits = flightWaits; r.liveFlights = flights.count; r.expired = expired
        r.retainedSystemAnchorSkips = retainedSystemAnchorSkips; r.restoreWaits = restoreWaits
        return r
    }
    func clear(resetStatistics: Bool = false, includingDisk: Bool = false) {
        for intent in readAdmissionIntents.values { intent.release() }
        readAdmissionIntents.removeAll()
        epoch &+= 1; flights.removeAll(); publications.removeAll(); reads.removeAll()
        index.clear(resetStatistics: resetStatistics)
        if includingDisk { disk?.clear(resetStatistics: resetStatistics) }
        if resetStatistics {
            published = 0; skippedOversize = 0; restoreFailures = 0; restoredHits = 0; diskHits = 0
            diskFallbacks = 0; pressureEvictions = 0; budgetSkipped = 0; duplicateSkipped = 0; flightWaits = 0; expired = 0
            diskReadTimeouts = 0
            diskPublicationTimeouts = 0
            retainedSystemAnchorSkips = 0; restoreWaits = 0
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
    @discardableResult
    func trimMemory(maxEntries: Int = Int.max) -> Int {
        var removed = 0
        while removed < maxEntries, index.evictLeastRecentlyUsed() { pressureEvictions += 1; removed += 1 }
        return removed
    }
    private func namespace(_ request: QwenGenerationRequest, model: QwenModel) -> String {
        diskIdentity + "|" + request.prefixCacheNamespace(accumulation: model.prefillAccumulation.rawValue,
            fusedPrefill: model.fusedPrefillEnabled)
    }
    func begin(_ request: QwenGenerationRequest, policy: QwenPrefixCachePolicy,
               model: QwenModel, allowWaitingForLeader: Bool = true) -> QwenPrefixCacheFlight {
        publications = publications.filter { !$0.value.isComplete }
        reads = reads.filter { !$0.value.isComplete }
        let tokens = Array(request.tokens.prefix(policy.lookupMaximum)), ns = namespace(request, model: model)
        let checkpoints = policy.publicationBoundaries.map {
            QwenPrefixCheckpoint(tokens: tokens, namespace: ns, boundary: $0)
        }
        return QwenPrefixCacheFlight(cache: self, namespace: ns, tokens: tokens,
            checkpoints: checkpoints, prefillChunk: request.prefillChunk,
            systemProducerBoundary: policy.systemProducerBoundary,
            epoch: epoch, allowWaitingForLeader: allowWaitingForLeader,
            executionNamespace: request.prefixCacheNamespace(accumulation: model.prefillAccumulation.rawValue,
                fusedPrefill: model.fusedPrefillEnabled))
    }
    func releaseFlight(key: String, identity: UUID) {
        flights.release(key: key, identity: identity)
    }
    func releaseReadAdmission(identity: UUID) {
        readAdmissionIntents.removeValue(forKey: identity)?.release()
    }
    private func releaseReadAdmission(_ f: QwenPrefixCacheFlight) {
        f.readAdmissionIntent?.release(); f.readAdmissionIntent = nil
        releaseReadAdmission(identity: f.identity)
    }
    private func current(_ f: QwenPrefixCacheFlight) -> Bool { f.epoch == epoch }
    private func releaseProducer(_ f: QwenPrefixCacheFlight, at boundary: Int? = nil) {
        guard let owned = f.ownedProducer, boundary == nil || boundary == owned.boundary else { return }
        releaseFlight(key: owned.key, identity: f.identity)
        f.ownedProducer = nil
    }
    private func markWaiting(_ f: QwenPrefixCacheFlight) {
        if !f.waiting { flightWaits += 1; f.waiting = true }
    }
    /// The caller either has not forwarded anything yet, or forbids waiting.
    /// A lower fallback always releases a future ticket before acquiring another.
    private func prepareProducer(_ f: QwenPrefixCacheFlight, after offset: Int, mayWait: Bool) -> Bool {
        guard current(f) else { releaseProducer(f); return true }
        let next = f.checkpoints.first { $0.boundary > offset }
        if next != f.ownedProducer { releaseProducer(f) }
        guard let next else { return true }
        if let owner = flights.owner(for: next.key), owner != f.identity {
            if mayWait { markWaiting(f); return false }
            return true // Continue privately; never replace the other owner.
        }
        if let publication = publications[next.key] {
            if publication.isComplete { publications.removeValue(forKey: next.key) }
            else if mayWait && !f.timedOutPublications.contains(next.key) {
                let now = DispatchTime.now().uptimeNanoseconds
                if f.publicationWaitKey != next.key {
                    f.publicationWaitKey = next.key; f.publicationWaitStartedAt = now
                }
                if let started = f.publicationWaitStartedAt,
                   Double(now - started) * 1e-9 >= diskRestoreTimeoutSeconds {
                    f.timedOutPublications.insert(next.key)
                    diskPublicationTimeouts += 1; diskFallbacks += 1
                } else { markWaiting(f); return false }
            }
        }
        if flights.claim(key: next.key, identity: f.identity) { f.ownedProducer = next }
        return true
    }
    /// Peek does not count a successful restore or touch LRU. The selected
    /// snapshot is looked up again immediately before private copying.
    private func memoryCandidate(_ f: QwenPrefixCacheFlight) -> QwenPrefixCacheMatch<Snapshot>? {
        while let match = index.peek(tokens: f.tokens, namespace: f.namespace) {
            if let ttlSeconds, Date().timeIntervalSince1970 - match.value.createdAt >= ttlSeconds {
                index.remove(tokens: Array(f.tokens.prefix(match.prefixTokenCount)), namespace: f.namespace)
                expired += 1
                continue
            }
            guard match.prefixTokenCount > 0, match.prefixTokenCount % f.prefillChunk == 0,
                  match.value.state.offset == match.prefixTokenCount else {
                index.remove(tokens: Array(f.tokens.prefix(match.prefixTokenCount)), namespace: f.namespace)
                restoreFailures += 1
                continue
            }
            return match
        }
        return nil
    }
    private func recoverOptionalFailure(model: QwenModel, error: Error) throws {
        // Cancellation cannot return admissions before submitted work joins.
        // A failed join takes precedence over the original cancellation.
        do { try MX.synchronize() }
        catch {
            model.failGenerationRecovery(String(describing: error))
            throw QwenGenerationError.unavailable("Prefix device recovery failed")
        }
        if error as? QwenGenerationError == .cancelled { throw error }
    }

    struct Resolution {
        let state: QwenModel.State?
        let source: String
        let lookupSeconds, restoreSeconds: Double
        let pagedAttachment: QwenPrefixPagedAttachment?
    }
    /// Nil means asynchronous I/O or another prefix leader is still pending.
    /// No GPU work is submitted for a waiting follower.
    func resolve(_ f: QwenPrefixCacheFlight, model: QwenModel,
                 checkCancellation: () throws -> Void) throws -> Resolution? {
        let started = DispatchTime.now().uptimeNanoseconds
        func elapsed(_ start: UInt64) -> Double { Double(DispatchTime.now().uptimeNanoseconds - start) * 1e-9 }
        var restoreWork = 0.0
        defer {
            f.lookupSeconds += max(0, elapsed(started) - restoreWork)
            f.restoreSeconds += restoreWork
        }
        func restoring<T>(_ body: () throws -> T) rethrows -> T {
            let start = DispatchTime.now().uptimeNanoseconds
            defer { restoreWork += elapsed(start) }
            return try body()
        }
        func resolved(_ state: QwenModel.State?, source: String,
                      pagedAttachment: QwenPrefixPagedAttachment? = nil) -> Resolution {
            releaseReadAdmission(f)
            f.resolved = true
            return .init(state: state, source: source,
                lookupSeconds: f.lookupSeconds + max(0, elapsed(started) - restoreWork),
                restoreSeconds: f.restoreSeconds + restoreWork, pagedAttachment: pagedAttachment)
        }
        try checkCancellation()
        guard current(f) else {
            f.detachRead(); releaseProducer(f)
            return resolved(nil, source: "cold")
        }
        if let read = f.read {
            if read.hasTimedOut(after: diskRestoreTimeoutSeconds) {
                // Only detach this request. lookupAsync's completion still owns
                // the ticket/lease until actual read + callback completion. Do
                // not release its workspace or start a replacement read here.
                f.detachRead(); releaseProducer(f); f.skipDisk = true
                diskReadTimeouts += 1; diskFallbacks += 1
            } else {
                guard read.isReady else { return nil }
                // Retain the final ticket through import/recovery. The callback
                // may still own host Data after marking ready; no early release.
                defer { f.detachRead(); withExtendedLifetime(read) {} }
                let expected = f.readCheckpoint?.boundary
                if let match = read.take(), match.prefixTokenCount == expected {
                    do {
                        var promotedAttachment: QwenPrefixPagedAttachment?
                        let state = try restoring {
                            let archive = QwenPrefixStateArchive(metadata: match.metadata, payload: match.payload,
                                logicalPayloadBytes: try model.estimatedPrefixStateBytes(at: match.prefixTokenCount))
                            let restored = try model.importPrefixState(archive, expectedOffset: match.prefixTokenCount,
                                checkCancellation: checkCancellation)
                            try checkCancellation()
                            promotedAttachment = try saveMemory(tokens: Array(f.tokens.prefix(match.prefixTokenCount)),
                                namespace: f.namespace, executionNamespace: f.executionNamespace,
                                state: restored, model: model, retainingSystemPrefix: f.systemProducerBoundary,
                                checkCancellation: checkCancellation, observer: nil)
                            return restored
                        }
                        diskHits += 1; restoredHits += 1
                        return resolved(state, source: "disk", pagedAttachment: promotedAttachment)
                    } catch {
                        if error is QwenPrefixPagedAttachmentFailure { throw error }
                        try recoverOptionalFailure(model: model, error: error)
                        _ = disk?.invalidateAsync(tokens: Array(f.tokens.prefix(match.prefixTokenCount)), namespace: f.namespace)
                        restoreFailures += 1
                    }
                }
                // A shorter result after corruption/expiry is not the selected
                // checkpoint. Replan from RAM/cold without a second SSD read.
                f.skipDisk = true; releaseProducer(f); diskFallbacks += 1
            }
        }

        // No forward work has started. A failed deeper restoration can safely
        // replan, after releasing its future producer, from the actual fallback.
        while true {
            try checkCancellation()
            // A normal compute producer may run much longer than five seconds.
            // This timestamp exists only after producer permission allowed an
            // actual SSD attempt; polling and eventual submission never reset it.
            if !f.skipDisk, let readStarted = f.readWaitStartedAt,
               QwenPrefixDiskRead.waitHasTimedOut(startedAt: readStarted, after: diskRestoreTimeoutSeconds) {
                releaseReadAdmission(f); releaseProducer(f); f.skipDisk = true
                diskReadTimeouts += 1; diskFallbacks += 1
            }
            let memoryDepth = memoryCandidate(f)?.prefixTokenCount ?? 0
            let summary = f.skipDisk ? nil : disk?.peek(tokens: f.tokens, namespace: f.namespace)
            let useDisk = summary.map {
                $0.prefixTokenCount > memoryDepth && $0.prefixTokenCount <= f.tokens.count &&
                $0.prefixTokenCount % f.prefillChunk == 0
            } ?? false
            let depth = useDisk ? summary!.prefixTokenCount : memoryDepth
            if !useDisk { releaseReadAdmission(f) }
            guard prepareProducer(f, after: depth, mayWait: f.allowWaitingForLeader) else {
                releaseReadAdmission(f); return nil
            }

            if useDisk, let summary, let disk {
                if f.readWaitStartedAt == nil { f.readWaitStartedAt = DispatchTime.now().uptimeNanoseconds }
                guard let readDeadline = QwenPrefixDiskRead.waitDeadline(startedAt: f.readWaitStartedAt!,
                    after: diskRestoreTimeoutSeconds) else {
                    throw QwenGenerationError.unavailable("Invalid SSD read admission deadline")
                }
                let checkpoint = QwenPrefixCheckpoint(tokens: f.tokens, namespace: f.namespace, boundary: depth)
                if let pending = reads[checkpoint.key] {
                    if pending.isComplete { reads.removeValue(forKey: checkpoint.key) }
                    else {
                        releaseReadAdmission(f)
                        if f.allowWaitingForLeader,
                           !QwenPrefixDiskRead.waitHasTimedOut(startedAt: f.readWaitStartedAt!, after: diskRestoreTimeoutSeconds) {
                            if !f.waitedForRead { restoreWaits += 1; f.waitedForRead = true }
                            markWaiting(f); return nil
                        }
                        if f.allowWaitingForLeader { diskReadTimeouts += 1 }
                        f.skipDisk = true; releaseProducer(f); diskFallbacks += 1
                        continue
                    }
                }
                if f.readAdmissionIntent == nil {
                    switch disk.acquireReadIntent(deadlineUptimeNanoseconds: readDeadline) {
                    case .acquired(let intent):
                        f.readAdmissionIntent = intent; readAdmissionIntents[f.identity] = intent
                    case .busy:
                        if f.allowWaitingForLeader { markWaiting(f); return nil }
                        f.skipDisk = true; releaseProducer(f); diskFallbacks += 1; continue
                    case .expired:
                        f.skipDisk = true; releaseProducer(f); diskReadTimeouts += 1; diskFallbacks += 1; continue
                    case .closed, .unavailable:
                        f.skipDisk = true; releaseProducer(f); diskFallbacks += 1; continue
                    }
                }
                guard let intent = f.readAdmissionIntent else {
                    throw QwenGenerationError.unavailable("Missing SSD read admission intent")
                }
                switch intent.state {
                case .ready: break
                case .busy:
                    // Whole-stage generation cannot wait for another paused
                    // request to advance while it holds the model admission.
                    if f.allowWaitingForLeader { markWaiting(f); return nil }
                    releaseReadAdmission(f); f.skipDisk = true; releaseProducer(f); diskFallbacks += 1; continue
                case .closed, .unavailable, .invalidated:
                    // Only a deadline in the still-live store epoch is a
                    // timeout. Clear/close and foreign invalidation are not.
                    if intent.hasExpired { diskReadTimeouts += 1 }
                    releaseReadAdmission(f); f.skipDisk = true; releaseProducer(f); diskFallbacks += 1; continue
                }
                // A write admitted BEFORE this intent may have replaced the
                // file between the first peek and the transition to ready.
                // Replan from fresh sizes before reserving any host workspace.
                guard let currentSummary = disk.peek(tokens: f.tokens, namespace: f.namespace),
                      currentSummary.prefixTokenCount == summary.prefixTokenCount,
                      currentSummary.metadataBytes == summary.metadataBytes,
                      currentSummary.payloadBytes == summary.payloadBytes,
                      currentSummary.diskBytes == summary.diskBytes else {
                    releaseProducer(f); continue
                }
                guard summary.metadataBytes >= 0, summary.payloadBytes >= 0,
                      summary.payloadBytes <= (Int.max - summary.metadataBytes) / 2,
                      let lease = reserve(bytes: summary.payloadBytes * 2 + summary.metadataBytes,
                                          kind: .workspace, model: model) else {
                    // Core readiness is not a promise that the joint model
                    // ledger has room. Keep the existing safe cold fallback.
                    releaseReadAdmission(f); f.skipDisk = true; releaseProducer(f); diskFallbacks += 1
                    continue
                }
                if QwenPrefixDiskRead.waitHasTimedOut(startedAt: f.readWaitStartedAt!, after: diskRestoreTimeoutSeconds) {
                    lease.release(); releaseReadAdmission(f); releaseProducer(f); f.skipDisk = true
                    diskReadTimeouts += 1; diskFallbacks += 1; continue
                }
                do { try checkCancellation() }
                catch { lease.release(); throw error }
                let read = QwenPrefixDiskRead(lease: lease, startedAt: f.readWaitStartedAt!), fence = QwenPrefixDiskReadFence()
                // Restrict the actual read to the depth whose workspace was
                // reserved, even if a deeper archive publishes after peek.
                let submitted = disk.lookupAsync(tokens: f.tokens, namespace: f.namespace, maxPrefixTokens: depth,
                    readIntent: intent,
                    completion: { [read, fence] result in
                        defer { fence.finishIO(); withExtendedLifetime(read) {} }
                        read.complete(result)
                    })
                if submitted == .accepted {
                    releaseReadAdmission(f) // Core consumed it atomically at IO admission.
                    reads = reads.filter { !$0.value.isComplete }
                    reads[checkpoint.key] = fence
                    f.read = read; f.readCheckpoint = checkpoint; f.readFence = fence
                    f.skipDisk = true // This request schedules at most one read.
                    return nil
                }
                lease.release() // No job/host archive was accepted.
                if submitted == .busy, f.allowWaitingForLeader {
                    // No fence was installed and no callback accepted. Keep
                    // only the metadata intent, bounded by the same deadline.
                    markWaiting(f); return nil
                }
                if submitted == .invalidated, intent.hasExpired {
                    diskReadTimeouts += 1
                }
                releaseReadAdmission(f); f.skipDisk = true; releaseProducer(f); diskFallbacks += 1
                continue
            }
            if let match = memoryCandidate(f) {
                // A TTL can expire between the initial peek and this use.
                // Never keep a future tail owner while starting from less.
                guard match.prefixTokenCount == memoryDepth else {
                    releaseProducer(f); continue
                }
                do {
                    let state = try restoring {
                        let copy = try model.privatePrefixStateCopy(match.value.state)
                        try checkCancellation()
                        return copy
                    }
                    _ = index.lookup(tokens: f.tokens, namespace: f.namespace)
                    restoredHits += 1
                    let attachment = compatibleAttachment(match.value.pagedAttachment, tokens: f.tokens,
                        namespace: f.namespace, executionNamespace: f.executionNamespace)
                    return resolved(state, source: "memory", pagedAttachment: attachment)
                } catch {
                    try recoverOptionalFailure(model: model, error: error)
                    index.remove(tokens: Array(f.tokens.prefix(match.prefixTokenCount)), namespace: f.namespace)
                    restoreFailures += 1; releaseProducer(f)
                    continue
                }
            }
            if memoryDepth > 0 { releaseProducer(f); continue }
            _ = index.lookup(tokens: f.tokens, namespace: f.namespace, touch: false)
            return resolved(nil, source: "cold")
        }
    }

    private func compatibleAttachment(_ attachment: QwenPrefixPagedAttachment?, tokens: [Int32],
                                      namespace: String, executionNamespace: String) -> QwenPrefixPagedAttachment? {
        guard let attachment, let pagedKVContext,
              attachment.matches(tokens: tokens, namespace: namespace,
                executionNamespace: executionNamespace, context: pagedKVContext) else { return nil }
        return attachment
    }

    /// The caller already inserted a valid dense snapshot. Every capacity
    /// denial below preserves that entry and does not evict another snapshot.
    private func makeOptionalAttachment(tokens: [Int32], namespace: String, executionNamespace: String,
                                         state: QwenModel.State, denseBytes: Int, model: QwenModel,
                                         retainingPrefix: Int?, extending: QwenPrefixPagedAttachment?,
                                         checkCancellation: () throws -> Void) throws -> QwenPrefixPagedAttachment? {
        guard let context = pagedKVContext, state.offset == tokens.count,
              state.offset > 0, state.offset <= context.maximumTokens,
              memoryPressure?.checkOptionalCacheAdmission() ?? true else { return nil }
        let ancestor = compatibleAttachment(extending, tokens: tokens, namespace: namespace,
            executionNamespace: executionNamespace)
        // Binding bounds are host-only optional metadata validation. A rejected
        // binding does not invalidate an otherwise useful dense snapshot.
        guard let charge = try? QwenPrefixPagedAttachment.metadataCharge(offset: state.offset,
            attentionLayers: context.pools.count, namespace: namespace, executionNamespace: executionNamespace) else { return nil }
        guard denseBytes <= index.maxBytes, charge <= index.maxBytes - denseBytes,
              charge <= index.maxBytes - index.statistics.logicalPayloadBytes else { return nil }
        let combinedBytes = denseBytes + charge
        if let retainingPrefix,
           !index.canInsertAlongsidePrefix(tokens: tokens, namespace: namespace,
                logicalPayloadBytes: combinedBytes, prefixTokenCount: retainingPrefix) { return nil }
        let oldOffset = ancestor?.prefix.offset ?? 0
        let rows = state.offset - oldOffset
        let requiredPages = rows == 0 ? 0 : (oldOffset % 32 + rows + 31) / 32
        do {
            let statistics = try context.layerStatistics
            guard statistics.values.allSatisfy({ $0.failedOperations == 0 }) else {
                model.failGenerationRecovery("Paged cache attachment arena has failed device operations")
                throw GPUError.invalid("Cannot publish an attachment from a failed physical KV arena")
            }
            // Future request page claims are accounted by the context helper;
            // optional cache pages must never consume a promised decode slot.
            guard try context.canReserveOptionalPages(pagesPerLayer: requiredPages) else { return nil }
        } catch {
            throw QwenPrefixPagedAttachmentFailure(cause: String(describing: error))
        }
        // Deliberately avoid reserve(...model:), whose LRU eviction is useful
        // for required dense state but wrong for an optional metadata upgrade.
        guard let metadataLease = model.stateBudget.reserve(bytes: charge, kind: .cache) else { return nil }
        defer { withExtendedLifetime(metadataLease) {} }
        do {
            try checkCancellation()
            let prefix = try model.makePagedKVPrefix(state: state, context: context, extending: ancestor?.prefix)
            // Completion pins can outlive a Swift State. Join before an error
            // or cancellation could drop the optional metadata reservation.
            try MX.synchronize()
            try checkCancellation()
            return try QwenPrefixPagedAttachment(prefix: prefix, metadataLease: metadataLease,
                namespace: namespace, executionNamespace: executionNamespace,
                checkpoint: QwenPrefixCheckpoint(tokens: tokens, namespace: namespace, boundary: state.offset))
        } catch {
            do { try MX.synchronize() }
            catch {
                model.failGenerationRecovery(String(describing: error))
                throw QwenPrefixPagedAttachmentFailure(cause: "device recovery failed: \(error)")
            }
            if error as? QwenGenerationError == .cancelled { throw error }
            if let stats = try? context.layerStatistics, stats.values.contains(where: { $0.failedOperations > 0 }) {
                model.failGenerationRecovery("Paged cache attachment observed a failed device operation")
            }
            throw QwenPrefixPagedAttachmentFailure(cause: String(describing: error))
        }
    }

    private func saveMemory(tokens: [Int32], namespace: String, executionNamespace: String,
                            state: QwenModel.State, model: QwenModel, retainingSystemPrefix: Int?,
                            checkCancellation: () throws -> Void,
                            observer: ((String, QwenModel.State) throws -> Void)?,
                            extending: QwenPrefixPagedAttachment? = nil) throws -> QwenPrefixPagedAttachment? {
        guard memoryPressure?.checkOptionalCacheAdmission() ?? true else { return nil }
        if let existing = index.peek(tokens: tokens, namespace: namespace), existing.prefixTokenCount == tokens.count {
            if let ttlSeconds, Date().timeIntervalSince1970 - existing.value.createdAt >= ttlSeconds {
                index.remove(tokens: tokens, namespace: namespace); expired += 1
            } else {
                duplicateSkipped += 1
                return compatibleAttachment(existing.value.pagedAttachment, tokens: tokens,
                    namespace: namespace, executionNamespace: executionNamespace)
            }
        }
        let bytes = try model.prefixStatePayloadBytes(state)
        guard bytes <= index.maxBytes, tokens.count <= index.maxKeyTokens else { skippedOversize += 1; return nil }
        var retainedPrefix = retainingSystemPrefix
        if let boundary = retainedPrefix, boundary < tokens.count,
           let anchor = index.peek(tokens: tokens, namespace: namespace, maxPrefixTokens: boundary),
           anchor.prefixTokenCount == boundary,
           let ttlSeconds, Date().timeIntervalSince1970 - anchor.value.createdAt >= ttlSeconds {
            index.remove(tokens: Array(tokens.prefix(boundary)), namespace: namespace)
            expired += 1; retainedPrefix = nil
        }
        if let boundary = retainedPrefix,
           !index.canInsertAlongsidePrefix(tokens: tokens, namespace: namespace,
                                           logicalPayloadBytes: bytes, prefixTokenCount: boundary) {
            retainedSystemAnchorSkips += 1
            return nil
        }
        guard let lease = reserve(bytes: bytes, kind: .cache, model: model) else { return nil }
        let saved: QwenModel.State
        do { saved = try model.privatePrefixStateCopy(state) }
        catch {
            // Failed lazy copies may still be retained by native completion.
            // Keep admission through recovery, including a throwing join.
            defer { withExtendedLifetime(lease) {} }
            try recoverOptionalFailure(model: model, error: error)
            lease.release()
            budgetSkipped += 1
            return nil
        }
        try observer?("publish", saved)
        try checkCancellation()
        guard memoryPressure?.checkOptionalCacheAdmission() ?? true else { return nil }
        let createdAt = Date().timeIntervalSince1970
        let inserted = index.insert(tokens: tokens, namespace: namespace,
            value: Snapshot(state: saved, lease: lease, createdAt: createdAt, pagedAttachment: nil),
            logicalPayloadBytes: bytes, retainingPrefixTokens: retainedPrefix)
        guard inserted else { return nil }
        published += 1
        guard let attachment = try makeOptionalAttachment(tokens: tokens, namespace: namespace,
            executionNamespace: executionNamespace, state: saved, denseBytes: bytes, model: model,
            retainingPrefix: retainedPrefix, extending: extending, checkCancellation: checkCancellation) else { return nil }
        // Preflighted metadata fits alongside all current entries. Replacing
        // this exact key retains its dense handles/lease and charges the wrapper.
        let upgraded = index.insert(tokens: tokens, namespace: namespace,
            value: Snapshot(state: saved, lease: lease, createdAt: createdAt, pagedAttachment: attachment),
            logicalPayloadBytes: bytes + attachment.metadataBytes, retainingPrefixTokens: retainedPrefix)
        return upgraded ? attachment : nil
    }
    @discardableResult
    func publish(_ f: QwenPrefixCacheFlight, at boundary: Int, state: QwenModel.State, model: QwenModel,
                 checkCancellation: () throws -> Void,
                 observer: ((String, QwenModel.State) throws -> Void)?,
                 extending: QwenPrefixPagedAttachment? = nil) throws -> QwenPrefixPagedAttachment? {
        // Only this exact checkpoint's computation ownership is released.
        // Retain the request context for the next publication opportunity.
        defer {
            releaseProducer(f, at: boundary)
            if current(f) { _ = prepareProducer(f, after: boundary, mayWait: false) }
        }
        guard current(f) else { return nil }
        guard let checkpoint = f.checkpoints.first(where: { $0.boundary == boundary }),
              state.offset == boundary, boundary > 0, boundary <= f.tokens.count,
              boundary % f.prefillChunk == 0 else {
            throw QwenGenerationError.unavailable("prefix publication does not match a complete planned checkpoint")
        }
        let tokens = Array(f.tokens.prefix(boundary)), key = checkpoint.key
        guard memoryPressure?.checkOptionalCacheAdmission() ?? true else { return nil }
        let attachment = try saveMemory(tokens: tokens, namespace: f.namespace, executionNamespace: f.executionNamespace,
            state: state, model: model,
            retainingSystemPrefix: f.systemProducerBoundary,
            checkCancellation: checkCancellation, observer: observer, extending: extending)
        guard let disk, disk.peek(tokens: tokens, namespace: f.namespace)?.prefixTokenCount != tokens.count,
              memoryPressure?.checkOptionalCacheAdmission() ?? true else { return attachment }
        // A waiter may have timed out and recomputed while this key's original
        // write is still in progress. Retain its ownership; never enqueue a
        // second full archive or replace the pending publication record.
        if let pending = publications[key], !pending.isComplete { duplicateSkipped += 1; return attachment }
        // Avoid a known-low-value full host export while a foreground request
        // waits for SSD admission. This hint owns nothing; enqueue still makes
        // the atomic decision if a reader arrives after this check.
        guard (disk.statistics.foregroundReadIntents ?? 0) == 0 else { return attachment }
        let bytes = try model.prefixStatePayloadBytes(state)
        guard bytes <= disk.limits.maxPendingBytes, bytes <= (Int.max - QwenPrefixStateArchiveDescriptor.maximumMetadataBytes) / 2,
              let lease = reserve(bytes: bytes * 2 + QwenPrefixStateArchiveDescriptor.maximumMetadataBytes, kind: .workspace, model: model) else { return attachment }
        // The submitting executor also owns the exported host archive until
        // this call returns, even if a small write completes immediately.
        defer { withExtendedLifetime(lease) {} }
        do {
            let archive = try model.exportPrefixState(state, maxPayloadBytes: disk.limits.maxPendingBytes,
                checkCancellation: checkCancellation)
            try checkCancellation()
            guard current(f), memoryPressure?.checkOptionalCacheAdmission() ?? true else { return attachment }
            let publication = QwenPrefixDiskPublication()
            publications = publications.filter { !$0.value.isComplete }
            if disk.enqueue(tokens: tokens, namespace: f.namespace, metadata: archive.metadata,
                payload: archive.payload, completion: { [lease] _ in
                    // The store and callback each retain this completion.
                    // Release by final ownership, not by callback timing.
                    defer { withExtendedLifetime(lease) {} }
                    publication.finish()
                }) {
                publications[key] = publication
            } else { lease.release() }
        } catch {
            // The outer lifetime guard retains the lease through this join;
            // explicit release must not return export workspace before it.
            try recoverOptionalFailure(model: model, error: error)
            lease.release()
            diskFallbacks += 1
        }
        return attachment
    }
}

struct QwenPrefixCachePolicy: Equatable {
    let lookupMaximum: Int
    let publicationBoundaries: [Int]
    let systemProducerBoundary: Int?
}

extension QwenGenerationRequest {
    /// Independently CPU-testable; the full request validator calls this too.
    func validatePrefixCachePolicy() throws {
        if prefixCachePlan != nil && prefixCacheMaxTokens != nil {
            throw QwenGenerationError.invalidRequest("prefixCachePlan and prefixCacheMaxTokens are mutually exclusive")
        }
        if let prefixCacheMaxTokens, !(0..<tokens.count).contains(prefixCacheMaxTokens) {
            throw QwenGenerationError.invalidRequest("prefixCacheMaxTokens must be nonnegative and shorter than the prompt")
        }
        if let plan = prefixCachePlan {
            guard plan.promptTokenCount == tokens.count else {
                throw QwenGenerationError.invalidRequest("prefixCachePlan promptTokenCount differs from the complete prompt")
            }
            guard plan.prefillChunk == prefillChunk else {
                throw QwenGenerationError.invalidRequest("prefixCachePlan prefillChunk differs from the execution grid")
            }
        }
    }
    var prefixCachePolicy: QwenPrefixCachePolicy? {
        guard mtpDepth == 0, tokens.count > 1, prefillChunk > 0 else { return nil }
        if let plan = prefixCachePlan {
            guard !plan.publicationTokenCounts.isEmpty else { return nil }
            return .init(lookupMaximum: plan.lookupMaxTokens,
                publicationBoundaries: plan.publicationTokenCounts,
                systemProducerBoundary: plan.systemProducerTokenCount)
        }
        let boundary = prefixCacheBoundary
        guard boundary > 0 else { return nil }
        return .init(lookupMaximum: boundary, publicationBoundaries: [boundary], systemProducerBoundary: boundary)
    }
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
