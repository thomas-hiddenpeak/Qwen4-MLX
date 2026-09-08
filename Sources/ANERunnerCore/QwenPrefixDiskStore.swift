import Foundation
import CryptoKit
import Darwin

/// Bounds for the optional CPU/SSD tier. File bytes include the container and
/// key, unlike the GPU tier's logical tensor bytes. Pending bytes account for
/// payload + opaque metadata retained by queued/active writes, or one maximum
/// read reservation. They do not include Data returned to a lookup caller.
public struct QwenPrefixDiskLimits: Codable, Equatable, Sendable {
    public let maxEntries: Int
    public let maxBytes: Int
    public let maxKeyTokens: Int
    public let maxPendingJobs: Int
    public let maxPendingBytes: Int
    public let maxMetadataBytes: Int
    /// Leave this much space available to the current user on the cache volume
    /// after allocating a complete temporary archive. Zero disables sampling.
    public let minAvailableBytes: Int

    public init(maxEntries: Int = 32, maxBytes: Int = 8_589_934_592,
                maxKeyTokens: Int = 1_048_576, maxPendingJobs: Int = 2,
                maxPendingBytes: Int = 536_870_912, maxMetadataBytes: Int = 1_048_576,
                minAvailableBytes: Int = 1_073_741_824) {
        self.maxEntries = maxEntries; self.maxBytes = maxBytes
        self.maxKeyTokens = maxKeyTokens; self.maxPendingJobs = maxPendingJobs
        self.maxPendingBytes = maxPendingBytes; self.maxMetadataBytes = maxMetadataBytes
        self.minAvailableBytes = minAvailableBytes
    }

    private enum CodingKeys: String, CodingKey {
        case maxEntries, maxBytes, maxKeyTokens, maxPendingJobs, maxPendingBytes, maxMetadataBytes
        case minAvailableBytes
    }

    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        self.init(maxEntries: try values.decode(Int.self, forKey: .maxEntries),
                  maxBytes: try values.decode(Int.self, forKey: .maxBytes),
                  maxKeyTokens: try values.decode(Int.self, forKey: .maxKeyTokens),
                  maxPendingJobs: try values.decode(Int.self, forKey: .maxPendingJobs),
                  maxPendingBytes: try values.decode(Int.self, forKey: .maxPendingBytes),
                  maxMetadataBytes: try values.decode(Int.self, forKey: .maxMetadataBytes),
                  minAvailableBytes: try values.decodeIfPresent(Int.self, forKey: .minAvailableBytes)
                    ?? 1_073_741_824)
    }
}

public struct QwenPrefixDiskStatistics: Codable, Equatable, Sendable {
    public var hits = 0
    public var misses = 0
    public var published = 0
    public var evictions = 0
    public var expired = 0
    public var corruptions = 0
    public var writeFailures = 0
    public var rejected = 0
    public var entries = 0
    /// Greater of file length and allocated blocks, summed over published files.
    public var diskBytes = 0
    public var keyTokens = 0
    public var pendingJobs = 0
    public var pendingBytes = 0
    public var bytesRead = 0
    public var bytesWritten = 0
    public var recoveredEntries = 0
    public var spaceChecks = 0
    /// Optional writes skipped because the latest sample cannot preserve the
    /// configured floor. These also contribute to `rejected`.
    public var spaceRejections = 0
    /// Query failures also skip the optional write and contribute to `rejected`.
    public var spaceQueryFailures = 0
    /// Transitions from a failed space check to a sufficient sample; a later
    /// publication may still fail for an unrelated reason.
    public var spaceRecoveries = 0
    public var availableSpaceBytes: UInt64? = nil
    /// Last check failed (low space or unknown space). Reads remain enabled;
    /// subsequent writes retry the check without requiring clear or restart.
    public var spaceConstrained = false
    /// An owned file could not be removed, so new SSD work is disabled rather
    /// than allowing unaccounted files to accumulate. Successful clear resets it.
    public var storageUnavailable = false
    /// Metadata-only foreground ownership, not admitted IO or payload bytes.
    /// Optional for compatibility with earlier diagnostic reports.
    public var foregroundReadIntents: Int? = nil
    public var foregroundReadIntentAcquisitions: Int? = nil
    public var optionalWritePriorityRejections: Int? = nil
}

public enum QwenPrefixDiskReadAdmissionState: Equatable, Sendable {
    case ready, busy, closed, unavailable, invalidated
}

public enum QwenPrefixDiskReadSubmission: Equatable, Sendable {
    case accepted, busy, closed, unavailable, invalidated
}

public enum QwenPrefixDiskReadIntentAdmission: Sendable {
    case acquired(QwenPrefixDiskReadIntent)
    case busy, closed, unavailable
}

/// One bounded, revocable priority intention. It owns no FD, result Data,
/// pending IO charge or callback. Only lookupAsync accepting IO transfers
/// archive ownership. Dropping an old handle cannot release a newer epoch.
public final class QwenPrefixDiskReadIntent: @unchecked Sendable {
    fileprivate weak var store: QwenPrefixDiskStore?
    fileprivate let identity = UUID()
    fileprivate let epoch: UInt64
    fileprivate init(store: QwenPrefixDiskStore, epoch: UInt64) {
        self.store = store; self.epoch = epoch
    }
    public var state: QwenPrefixDiskReadAdmissionState {
        store?.readAdmissionState(self) ?? .closed
    }
    public func release() { store?.releaseReadIntent(self) }
    deinit { release() }
}

public struct QwenPrefixDiskSummary: Sendable {
    public let prefixTokenCount: Int
    public let metadataBytes: Int
    public let payloadBytes: Int
    public let diskBytes: Int
}

public struct QwenPrefixDiskMatch: Sendable {
    public let prefixTokenCount: Int
    public let metadata: Data
    public let payload: Data
    public let diskBytes: Int
}

/// A snapshot of an already requested close. Incomplete means background IO
/// or callback delivery still owns resources; it does not reopen admissions.
public struct QwenPrefixDiskCloseResult: Equatable, Sendable {
    /// All admitted IO returned and the IO queue closed its directory/lock FDs.
    public let ioCompleted: Bool
    /// All callbacks admitted before close returned and released their owners.
    public let callbacksCompleted: Bool
    public var completed: Bool { ioCompleted && callbacksCompleted }

    public init(ioCompleted: Bool, callbacksCompleted: Bool) {
        self.ioCompleted = ioCompleted
        self.callbacksCompleted = callbacksCompleted
    }
}

/// An exclusively owned, namespace-isolated, checksummed snapshot directory.
///
/// All filesystem/index work runs on one CPU queue. `enqueue` is nonblocking:
/// false means no ownership was accepted; true is admission, not durability.
/// Disk failures drop cache work and are visible in statistics. Lookups return
/// nil on any invalid/expired/missing file, then try a shorter saved prefix.
/// No Tensor, model object, or generation executor is used on this queue.
///
/// `clear` invalidates admitted work before waiting for active IO. `close`
/// rejects new jobs and optionally drains admitted writes; close(drain:false)
/// invalidates them. The original synchronous APIs wait without a deadline;
/// use close(drain:timeout:) to bound the caller's wait, including callbacks.
/// Waiting operations must not be called from an asynchronous completion.
/// Cache callbacks are dispatched separately; non-waiting completion calls
/// may use the store.
/// The directory is private (0700), files are 0600, and a process lock prevents
/// two store instances from managing it concurrently. Only exact owned names
/// are removed, and path operations use directory FDs without following links.
public final class QwenPrefixDiskStore: @unchecked Sendable {
    public enum StoreError: Error, Equatable {
        case invalidLimits
        case unsafeDirectory
        case directoryInUse
        case io(operation: String, code: Int32)
        case corruptFile
    }

    public let directory: URL
    public let limits: QwenPrefixDiskLimits
    private let ttlSeconds: TimeInterval?
    private let clock: @Sendable () -> TimeInterval
    private let availableSpace: @Sendable (Int32) throws -> UInt64
    private let publicationDirectorySync: @Sendable (Int32) throws -> Void
    private let queue = DispatchQueue(label: "qwen.prefix.ssd", qos: .utility)
    private let callbackQueue = DispatchQueue(label: "qwen.prefix.ssd.callback", qos: .utility)
    private let queueKey = DispatchSpecificKey<Bool>()
    private let admission = NSLock()
    private let closeGroup = DispatchGroup()
    private let callbackCloseGroup = DispatchGroup()
    private var epoch: UInt64 = 0
    private var closed = false
    private var foregroundReadIntent: (identity: UUID, epoch: UInt64)?
    private var stats = QwenPrefixDiskStatistics()
    private var summaries: [String: SummaryRecord] = [:]
    private struct SummaryRecord {
        let namespace: String
        let tokens: [Int32]
        let createdAt: TimeInterval
        let expiresAt: TimeInterval?
        let summary: QwenPrefixDiskSummary
        let revision: UUID
    }
    private var directoryFD: Int32 = -1
    private var lockFD: Int32 = -1
    // Below are owned exclusively by queue (including startup before exposure).
    private let index: QwenPrefixCacheIndex<String>
    private var records: [String: Record] = [:]
    // A clear invalidates public metadata immediately, before its queued file
    // removal runs. Old queue work must not refill that metadata meanwhile.
    private var recordsEpoch: UInt64 = 0
    private var order: UInt64 = 0
    private var fileAllocationUnit = 4096
    private static let prefix = "qwen-prefix-v1-"
    private static let lockName = ".qwen-prefix-store-v1.lock"
    private static let headerBytes = 92
    private static let maxNamespaceBytes = 16_384
    private static let maxManifestBytes = 32 * 1_048_576
    private static let magic = Data("QWPCSSD1".utf8)

    private struct Manifest: Codable {
        let version: Int
        let namespace: String
        let tokens: [Int32]
        let metadata: Data
        let metadataSHA256: String
        let createdAt: TimeInterval
        let expiresAt: TimeInterval?
    }
    private struct Record {
        let manifest: Manifest
        let bytes: Int
        let payloadBytes: Int
        let revision = UUID()
        var order: UInt64
    }
    private struct Decoded {
        let manifest: Manifest
        let payload: Data
        let bytes: Int
        let modified: TimeInterval
        let payloadBytes: Int
    }

    public convenience init(directory: URL, limits: QwenPrefixDiskLimits = .init(),
                ttlSeconds: TimeInterval? = nil,
                now: @escaping @Sendable () -> TimeInterval = { Date().timeIntervalSince1970 },
                availableSpace: (@Sendable (Int32) throws -> UInt64)? = nil) throws {
        try self.init(directory: directory, limits: limits, ttlSeconds: ttlSeconds,
                      now: now, availableSpace: availableSpace,
                      publicationDirectorySync: { fd in
                          guard fsync(fd) == 0 else { throw Self.io("fsync directory") }
                      })
    }

    // Module-internal fault injection for the publication commit boundary.
    // The callback borrows the directory FD, runs without admission held, and
    // must neither close the FD nor call a waiting store operation.
    init(directory: URL, limits: QwenPrefixDiskLimits = .init(),
         ttlSeconds: TimeInterval? = nil,
         now: @escaping @Sendable () -> TimeInterval = { Date().timeIntervalSince1970 },
         availableSpace: (@Sendable (Int32) throws -> UInt64)? = nil,
         publicationDirectorySync: @escaping @Sendable (Int32) throws -> Void) throws {
        guard limits.maxEntries > 0, limits.maxBytes > 0,
              limits.maxKeyTokens > 0, limits.maxPendingJobs > 0,
              limits.maxPendingBytes > 0, limits.maxMetadataBytes > 0,
              limits.minAvailableBytes >= 0,
              limits.maxMetadataBytes <= Self.maxManifestBytes / 2,
              ttlSeconds.map({ $0.isFinite && $0 > 0 }) ?? true,
              directory.isFileURL else { throw StoreError.invalidLimits }
        self.directory = directory; self.limits = limits
        self.ttlSeconds = ttlSeconds; self.clock = now
        // The optional sampler is an injection point for deterministic CPU
        // faults. It runs only on the IO queue, borrows the open directory FD,
        // and must not retain or close that FD or call back into this store.
        self.availableSpace = availableSpace ?? { try Self.sampleAvailableSpace(directoryFD: $0) }
        self.publicationDirectorySync = publicationDirectorySync
        self.index = try QwenPrefixCacheIndex(maxEntries: limits.maxEntries,
                                             maxBytes: limits.maxBytes,
                                             maxKeyTokens: limits.maxKeyTokens)
        queue.setSpecific(key: queueKey, value: true)
        do {
            directoryFD = try Self.openPrivateDirectory(directory)
            var fs = statvfs()
            // APFS reports a 1 MiB preferred IO transfer size in f_bsize;
            // f_frsize is its actual allocation fragment (typically 4 KiB).
            if fstatvfs(directoryFD, &fs) == 0, fs.f_frsize > 0,
               let fragment = Int(exactly: fs.f_frsize) {
                fileAllocationUnit = max(512, fragment)
            }
            lockFD = openat(directoryFD, Self.lockName,
                            O_RDWR | O_CREAT | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC, mode_t(0o600))
            guard lockFD >= 0 else { throw Self.io("open lock") }
            var info = stat()
            guard fstat(lockFD, &info) == 0, Self.isPrivateRegular(info) else {
                throw StoreError.unsafeDirectory
            }
            guard flock(lockFD, LOCK_EX | LOCK_NB) == 0 else {
                throw errno == EWOULDBLOCK ? StoreError.directoryInUse : Self.io("lock")
            }
            try recoverDirectory()
        } catch {
            if lockFD >= 0 { Darwin.close(lockFD); lockFD = -1 }
            if directoryFD >= 0 { Darwin.close(directoryFD); directoryFD = -1 }
            throw error
        }
    }

    deinit {
        // Admitted jobs retain self, so no job can still need these FDs here.
        // Do not enqueue a self-retaining closure from deinit.
        if DispatchQueue.getSpecific(key: queueKey) == true { closeDescriptors() }
        else { queue.sync { closeDescriptors() } }
    }

    public var statistics: QwenPrefixDiskStatistics {
        admission.lock(); defer { admission.unlock() }
        var value = stats
        value.foregroundReadIntents = foregroundReadIntent == nil ? 0 : 1
        value.foregroundReadIntentAcquisitions = stats.foregroundReadIntentAcquisitions ?? 0
        value.optionalWritePriorityRejections = stats.optionalWritePriorityRejections ?? 0
        return value
    }

    /// This is metadata admission only. A cooperative caller may wait without
    /// allocating workspace while an old write/read releases its real charge.
    public func acquireReadIntent() -> QwenPrefixDiskReadIntentAdmission {
        admission.lock(); defer { admission.unlock() }
        guard !closed else { return .closed }
        guard !stats.storageUnavailable else { return .unavailable }
        guard foregroundReadIntent == nil else { return .busy }
        let intent = QwenPrefixDiskReadIntent(store: self, epoch: epoch)
        foregroundReadIntent = (intent.identity, epoch)
        stats.foregroundReadIntentAcquisitions = (stats.foregroundReadIntentAcquisitions ?? 0) + 1
        return .acquired(intent)
    }

    fileprivate func releaseReadIntent(_ intent: QwenPrefixDiskReadIntent) {
        admission.lock(); defer { admission.unlock() }
        if ownsReadIntentLocked(intent) { foregroundReadIntent = nil }
    }

    private func ownsReadIntentLocked(_ intent: QwenPrefixDiskReadIntent) -> Bool {
        intent.store === self && intent.epoch == epoch &&
            foregroundReadIntent?.identity == intent.identity && foregroundReadIntent?.epoch == intent.epoch
    }

    fileprivate func readAdmissionState(_ intent: QwenPrefixDiskReadIntent) -> QwenPrefixDiskReadAdmissionState {
        admission.lock(); defer { admission.unlock() }
        return readAdmissionStateLocked(intent)
    }

    private func readAdmissionStateLocked(_ intent: QwenPrefixDiskReadIntent?) -> QwenPrefixDiskReadAdmissionState {
        guard !closed else { return .closed }
        guard !stats.storageUnavailable else { return .unavailable }
        if let intent {
            guard ownsReadIntentLocked(intent) else { return .invalidated }
        } else if foregroundReadIntent != nil { return .busy }
        // A read still charges the FULL configured maximum. No charge is
        // borrowed from an unfinished write, callback or another result.
        return stats.pendingJobs < limits.maxPendingJobs && stats.pendingBytes == 0 ? .ready : .busy
    }

    @discardableResult
    public func enqueue(tokens: [Int32], namespace: String,
                        metadata: Data, payload: Data,
                        completion: (@Sendable (Bool) -> Void)? = nil) -> Bool {
        let (charge, overflow) = metadata.count.addingReportingOverflow(payload.count)
        guard !overflow, !tokens.isEmpty, tokens.count <= limits.maxKeyTokens,
              !namespace.isEmpty, namespace.utf8.count <= Self.maxNamespaceBytes,
              metadata.count <= limits.maxMetadataBytes,
              payload.count > 0, charge <= limits.maxPendingBytes,
              charge < limits.maxBytes else { reject(); return false }
        admission.lock()
        guard foregroundReadIntent == nil else {
            stats.rejected += 1
            stats.optionalWritePriorityRejections = (stats.optionalWritePriorityRejections ?? 0) + 1
            admission.unlock(); return false
        }
        guard admitLocked(bytes: charge) else { admission.unlock(); return false }
        let submittedEpoch = epoch
        queue.async { [self] in
            // A callback can start on its separate queue before this IO block
            // returns. Keep its captured transfer owner through our own final
            // use of the archive, independently of callback completion timing.
            defer { withExtendedLifetime(completion) {} }
            let success = isCurrent(submittedEpoch) && write(
                tokens: tokens, namespace: namespace, metadata: metadata,
                payload: payload, submittedEpoch: submittedEpoch)
            finishJob(bytes: charge)
            if let completion { callbackQueue.async { completion(success) } }
        }
        admission.unlock()
        return true
    }

    /// Inspects published metadata without scheduling IO or changing LRU.
    /// A subsequent lookup still verifies the complete file before returning it.
    public func peek(tokens: [Int32], namespace: String,
                     maxPrefixTokens: Int? = nil) -> QwenPrefixDiskSummary? {
        let cap = min(tokens.count, maxPrefixTokens ?? tokens.count)
        guard cap > 0 else { return nil }
        let now = clock()
        admission.lock(); defer { admission.unlock() }
        guard !closed, !stats.storageUnavailable else { return nil }
        return summaries.values.filter { record in
            record.namespace == namespace && record.tokens.count <= cap &&
            now.isFinite && record.createdAt <= now + 60 &&
            !(record.expiresAt.map { now >= $0 } ?? false) &&
            !(ttlSeconds.map { now - record.createdAt >= $0 } ?? false) &&
            tokens.starts(with: record.tokens)
        }.max(by: { $0.tokens.count < $1.tokens.count })?.summary
    }

    /// Removes an exact opaque archive rejected by the model-specific decoder.
    public func invalidate(tokens: [Int32], namespace: String) {
        admission.lock()
        guard !closed else { admission.unlock(); return }
        let semaphore = DispatchSemaphore(value: 0)
        queue.async { [self] in
            removeRecord(Self.fileName(tokens: tokens, namespace: namespace))
            _ = fsync(directoryFD)
            semaphore.signal()
        }
        admission.unlock(); semaphore.wait()
    }

    /// Invalidates only the version published when this call was admitted.
    /// A replacement already queued ahead of this operation, or submitted
    /// afterward, survives. This metadata-only operation consumes one pending
    /// job; it retains no opaque Data and therefore charges zero payload bytes.
    @discardableResult
    public func invalidateAsync(tokens: [Int32], namespace: String) -> Bool {
        guard !tokens.isEmpty, tokens.count <= limits.maxKeyTokens,
              namespace.utf8.count <= Self.maxNamespaceBytes else { reject(); return false }
        let name = Self.fileName(tokens: tokens, namespace: namespace)
        admission.lock()
        guard let revision = summaries[name]?.revision, admitLocked(bytes: 0) else {
            admission.unlock(); return false
        }
        let submittedEpoch = epoch
        queue.async { [self] in
            defer { finishJob(bytes: 0) }
            guard isCurrent(submittedEpoch), records[name]?.revision == revision else { return }
            _ = removeRecord(name)
            if fsync(directoryFD) != 0 { update { $0.writeFailures += 1 } }
        }
        admission.unlock()
        return true
    }

    /// Synchronous, bounded read. It may wait for previously admitted writes;
    /// schedulers should use lookupAsync to keep their inference loop running.
    public func lookup(tokens: [Int32], namespace: String,
                       maxPrefixTokens: Int? = nil) -> QwenPrefixDiskMatch? {
        admission.lock()
        guard !closed else { stats.misses += 1; admission.unlock(); return nil }
        let submittedEpoch = epoch
        admission.unlock()
        return queue.sync { lookupOnQueue(tokens: tokens, namespace: namespace,
                                          maxPrefixTokens: maxPrefixTokens,
                                          submittedEpoch: submittedEpoch) }
    }

    /// A read reserves maxPendingBytes until its callback returns. This bounds
    /// retained result Data even when a callback is temporarily slow.
    @discardableResult
    public func lookupAsync(tokens: [Int32], namespace: String,
                            maxPrefixTokens: Int? = nil,
                            completion: @escaping @Sendable (QwenPrefixDiskMatch?) -> Void) -> Bool {
        submitLookup(tokens: tokens, namespace: namespace, maxPrefixTokens: maxPrefixTokens,
            intent: nil, completion: completion) == .accepted
    }

    /// A ready hint can race clear/close/control admission. Only `.accepted`
    /// transfers the callback and pending byte ownership; other results do not
    /// invoke completion and the caller must release untransferred workspace.
    @discardableResult
    public func lookupAsync(tokens: [Int32], namespace: String, maxPrefixTokens: Int? = nil,
                            readIntent: QwenPrefixDiskReadIntent,
                            completion: @escaping @Sendable (QwenPrefixDiskMatch?) -> Void) -> QwenPrefixDiskReadSubmission {
        submitLookup(tokens: tokens, namespace: namespace, maxPrefixTokens: maxPrefixTokens,
            intent: readIntent, completion: completion)
    }

    private func submitLookup(tokens: [Int32], namespace: String, maxPrefixTokens: Int?,
                              intent: QwenPrefixDiskReadIntent?,
                              completion: @escaping @Sendable (QwenPrefixDiskMatch?) -> Void) -> QwenPrefixDiskReadSubmission {
        admission.lock()
        let state = readAdmissionStateLocked(intent)
        guard state == .ready else {
            stats.rejected += 1; admission.unlock()
            switch state {
            case .busy: return .busy
            case .closed: return .closed
            case .unavailable: return .unavailable
            case .invalidated: return .invalidated
            case .ready: preconditionFailure("ready admission entered rejection")
            }
        }
        let charge = limits.maxPendingBytes
        guard admitLocked(bytes: charge) else { admission.unlock(); return .busy }
        if intent != nil { foregroundReadIntent = nil }
        let submittedEpoch = epoch
        queue.async { [self] in
            let result = lookupOnQueue(tokens: tokens, namespace: namespace,
                                       maxPrefixTokens: maxPrefixTokens,
                                       submittedEpoch: submittedEpoch)
            callbackQueue.async { [self] in
                completion(isCurrent(submittedEpoch) ? result : nil)
                finishJob(bytes: charge)
            }
        }
        admission.unlock()
        return .accepted
    }

    /// Waits for previously submitted IO, without closing or invalidating it.
    /// Async callback delivery may follow drain, and remains charged until done.
    public func drain() { queue.sync {} }

    /// Waits for IO admitted before this call and its callback delivery. New
    /// concurrent admissions may remain pending when flush returns.
    public func flush() {
        queue.sync {}
        callbackQueue.sync {}
    }

    public func clear(resetStatistics: Bool = false) {
        admission.lock()
        guard !closed else { admission.unlock(); return }
        epoch &+= 1
        foregroundReadIntent = nil
        let submittedEpoch = epoch
        summaries.removeAll(keepingCapacity: false)
        let semaphore = DispatchSemaphore(value: 0)
        queue.async { [self] in
            var removedAll = true
            for name in Array(records.keys) {
                if !removeRecord(name) { removedAll = false }
            }
            if !removeOwnedTemporaryFiles() { removedAll = false }
            if fsync(directoryFD) != 0 { removedAll = false }
            recordsEpoch = submittedEpoch
            admission.lock()
            stats.storageUnavailable = !removedAll
            if resetStatistics {
                let pendingJobs = stats.pendingJobs, pendingBytes = stats.pendingBytes
                let unavailable = stats.storageUnavailable
                let available = stats.availableSpaceBytes, constrained = stats.spaceConstrained
                let entries = stats.entries, bytes = stats.diskBytes, keys = stats.keyTokens
                stats = .init(); stats.pendingJobs = pendingJobs; stats.pendingBytes = pendingBytes
                stats.entries = entries; stats.diskBytes = bytes; stats.keyTokens = keys
                stats.storageUnavailable = unavailable
                stats.availableSpaceBytes = available; stats.spaceConstrained = constrained
            }
            admission.unlock()
            semaphore.signal()
        }
        admission.unlock()
        semaphore.wait()
    }

    public func close(drain: Bool = true) {
        beginClose(drain: drain)
        closeGroup.wait()
        if drain { callbackCloseGroup.wait() }
    }

    /// Rejects new work immediately, then waits at most `timeout` seconds for
    /// the already admitted IO, descriptor closure and callback delivery. Both
    /// phases share one monotonic deadline. Nonpositive or nonfinite timeouts
    /// poll without waiting; they never request an unlimited wait.
    ///
    /// A timeout only detaches this waiter. It cannot cancel an OS syscall,
    /// close an FD still in use, or release a pending Data/lease owner. Queued
    /// work keeps the store alive until actual completion. A later call may
    /// wait again; it neither schedules another close nor admits new work.
    /// `drain: false` invalidates pending writes, but completion here still
    /// requires their cleanup and callbacks to finish.
    @discardableResult
    public func close(drain: Bool = true, timeout: TimeInterval) -> QwenPrefixDiskCloseResult {
        let deadline = Self.closeDeadline(timeout)
        beginClose(drain: drain)
        let ioCompleted = closeGroup.wait(timeout: deadline) == .success
        let callbacksCompleted = ioCompleted && callbackCloseGroup.wait(timeout: deadline) == .success
        return .init(ioCompleted: ioCompleted, callbacksCompleted: callbacksCompleted)
    }

    private static func closeDeadline(_ timeout: TimeInterval) -> DispatchTime {
        let now = DispatchTime.now().uptimeNanoseconds
        guard timeout.isFinite, timeout > 0 else { return .init(uptimeNanoseconds: now) }
        // Saturate below DispatchTime.distantFuture without trapping on large
        // finite TimeIntervals or overflowing the absolute monotonic clock.
        let nanoseconds = (timeout * 1_000_000_000).rounded(.up)
        let maximum = UInt64.max - 1
        let remaining = maximum - min(now, maximum)
        guard nanoseconds < Double(remaining) else { return .init(uptimeNanoseconds: maximum) }
        return .init(uptimeNanoseconds: now + UInt64(nanoseconds))
    }

    private func beginClose(drain: Bool) {
        admission.lock()
        if !closed {
            closed = true
            foregroundReadIntent = nil
            closeGroup.enter()
            callbackCloseGroup.enter()
            if !drain { epoch &+= 1 }
            queue.async { [self] in
                closeDescriptors()
                closeGroup.leave()
                // Every admitted IO block has already enqueued its callback.
                // This marker therefore follows all callback owners, including
                // reads that remain charged until their callback returns.
                callbackQueue.async { [self] in callbackCloseGroup.leave() }
            }
        } else if !drain {
            // A concurrent immediate close may cancel jobs a draining close
            // has not processed yet; all callers still wait for the same FDs.
            epoch &+= 1
        }
        admission.unlock()
    }

    private func admitLocked(bytes: Int) -> Bool {
        guard !closed, !stats.storageUnavailable, stats.pendingJobs < limits.maxPendingJobs,
              bytes <= limits.maxPendingBytes - stats.pendingBytes else {
            stats.rejected += 1; return false
        }
        stats.pendingJobs += 1; stats.pendingBytes += bytes
        return true
    }
    private func finishJob(bytes: Int) {
        admission.lock(); defer { admission.unlock() }
        stats.pendingJobs -= 1; stats.pendingBytes -= bytes
    }
    private func reject() { update { $0.rejected += 1 } }
    private func update(_ body: (inout QwenPrefixDiskStatistics) -> Void) {
        admission.lock(); defer { admission.unlock() }; body(&stats)
    }
    private func isCurrent(_ submittedEpoch: UInt64) -> Bool {
        admission.lock(); defer { admission.unlock() }; return epoch == submittedEpoch
    }
    @discardableResult
    private func refreshAccounting(publishing submittedEpoch: UInt64? = nil,
                                   writtenBytes: Int = 0) -> Bool {
        let entries = records.count
        let bytes = records.values.reduce(0) { $0 + $1.bytes }
        let tokens = records.values.reduce(0) { $0 + $1.manifest.tokens.count }
        let snapshot = records.mapValues { record in
            SummaryRecord(namespace: record.manifest.namespace, tokens: record.manifest.tokens,
                          createdAt: record.manifest.createdAt, expiresAt: record.manifest.expiresAt,
                          summary: .init(prefixTokenCount: record.manifest.tokens.count,
                                         metadataBytes: record.manifest.metadata.count,
                                         payloadBytes: record.payloadBytes, diskBytes: record.bytes),
                          revision: record.revision)
        }
        admission.lock()
        stats.entries = entries; stats.diskBytes = bytes; stats.keyTokens = tokens
        let current = recordsEpoch == epoch && (submittedEpoch.map { $0 == epoch } ?? true)
        if current { summaries = snapshot }
        else { summaries.removeAll(keepingCapacity: false) }
        if submittedEpoch != nil, current {
            stats.published += 1; stats.bytesWritten += writtenBytes
        }
        admission.unlock()
        return current
    }

    private func write(tokens: [Int32], namespace: String, metadata: Data,
                       payload: Data, submittedEpoch: UInt64) -> Bool {
        let created = clock()
        guard created.isFinite else { update { $0.writeFailures += 1 }; return false }
        let manifest = Manifest(version: 1, namespace: namespace, tokens: tokens,
                                metadata: metadata, metadataSHA256: Self.digest(metadata),
                                createdAt: created, expiresAt: ttlSeconds.map { created + $0 })
        let name = Self.fileName(tokens: tokens, namespace: namespace)
        let temporary = Self.prefix + UUID().uuidString.lowercased() + ".tmp"
        var fd: Int32 = -1
        var installedCandidate = false
        var published = false
        defer {
            // A successful rename is not yet a published cache entry. Clear
            // or immediate close may invalidate it while directory IO runs.
            // The serial queue prevents a replacement from racing this cleanup.
            if installedCandidate && !published {
                _ = removeRecord(name)
                if fsync(directoryFD) != 0 { update { $0.writeFailures += 1 } }
            }
            if fd >= 0 { Darwin.close(fd) }
            if unlinkat(directoryFD, temporary, 0) != 0, errno != ENOENT {
                update { $0.storageUnavailable = true; $0.writeFailures += 1 }
            }
        }
        do {
            guard !statistics.storageUnavailable else { return false }
            let encoded = try JSONEncoder().encode(manifest)
            guard encoded.count <= Self.maxManifestBytes else { reject(); return false }
            let total = try Self.sum(Self.headerBytes, encoded.count, payload.count)
            let reserved = try roundedAllocation(total)
            guard reserved <= limits.maxBytes else { reject(); return false }
            // This queue has a single physical writer. Queued jobs hold RAM
            // reservations, but cannot allocate a temporary file until their
            // own fresh space check. Existing published bytes are already
            // excluded by f_bavail; do not subtract them again, or credit a
            // same-key file that rename has not replaced yet. Check before LRU
            // eviction so an optional low-space write cannot evict useful data.
            guard hasAvailableSpace(forAdditionalBytes: reserved) else { return false }
            guard isCurrent(submittedEpoch) else { return false }
            removeExpired()
            // Reserve room for a temporary complete file before creating it.
            // A replacement keeps its old file until this reservation requires
            // eviction; a failed write is allowed to lose cache entries only.
            guard makeRoom(bytes: reserved, tokens: tokens.count, entries: 1) else { return false }
            guard isCurrent(submittedEpoch) else { return false }
            fd = openat(directoryFD, temporary,
                        O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, mode_t(0o600))
            guard fd >= 0 else { throw Self.io("create snapshot") }
            var header = Self.magic
            Self.appendLE(UInt32(1), to: &header)
            Self.appendLE(UInt64(encoded.count), to: &header)
            Self.appendLE(UInt64(payload.count), to: &header)
            header.append(contentsOf: SHA256.hash(data: encoded))
            header.append(contentsOf: SHA256.hash(data: payload))
            try Self.writeAll(header, fd: fd)
            try Self.writeAll(encoded, fd: fd)
            try Self.writeAll(payload, fd: fd)
            guard fsync(fd) == 0 else { throw Self.io("fsync snapshot") }
            var info = stat()
            guard fstat(fd, &info) == 0, Self.isPrivateRegular(info) else {
                throw StoreError.corruptFile
            }
            let cost = try Self.diskCost(info)
            guard cost <= limits.maxBytes else { reject(); return false }
            // Another process can consume the volume after the first sample.
            // The temporary file is now allocated, so check only the floor;
            // charging its size again would double-count those bytes. Failure
            // drops the temporary archive. Actual ENOSPC still follows the IO
            // failure path; a sampled floor is not an OS reservation guarantee.
            guard hasAvailableSpace(forAdditionalBytes: 0) else { return false }
            guard makeRoom(bytes: cost, tokens: tokens.count, entries: 1) else { return false }
            admission.lock()
            guard epoch == submittedEpoch else { admission.unlock(); return false }
            // A replacement can have a different payload size. Do not let a
            // new peek size its restore from the old file while rename/sync
            // replaces that file. Already admitted reads stay queue ordered.
            summaries.removeValue(forKey: name)
            admission.unlock()
            // Physical IO never holds admission: even a delayed directory sync
            // must leave metadata, new admissions and clear's epoch responsive.
            guard renameat(directoryFD, temporary, directoryFD, name) == 0 else {
                throw Self.io("rename snapshot")
            }
            installedCandidate = true
            // If a same-key file survived reservation, rename replaced it.
            if let old = records.removeValue(forKey: name) {
                _ = index.remove(tokens: old.manifest.tokens, namespace: old.manifest.namespace)
            }
            // Track the candidate for cleanup/accounting, but keep it out of
            // public summaries until both durable IO and the epoch check pass.
            stageRecord(name: name, manifest: manifest, bytes: cost, payloadBytes: payload.count)
            try publicationDirectorySync(directoryFD)
            guard refreshAccounting(publishing: submittedEpoch, writtenBytes: total) else { return false }
            published = true
            return true
        } catch {
            // If rename failed, the prior archive still exists and may regain
            // its summary only if clear has not invalidated this record epoch.
            if !installedCandidate { refreshAccounting() }
            update { $0.writeFailures += 1 }
            return false
        }
    }

    private func hasAvailableSpace(forAdditionalBytes bytes: Int) -> Bool {
        guard limits.minAvailableBytes > 0 else { return true }
        do {
            let available = try availableSpace(directoryFD)
            let floor = UInt64(limits.minAvailableBytes)
            // Subtraction avoids overflowing floor + archive reservation.
            let sufficient = available >= floor && available - floor >= UInt64(bytes)
            update {
                $0.spaceChecks += 1; $0.availableSpaceBytes = available
                if sufficient {
                    if $0.spaceConstrained { $0.spaceRecoveries += 1 }
                    $0.spaceConstrained = false
                } else {
                    $0.spaceConstrained = true
                    $0.spaceRejections += 1; $0.rejected += 1
                }
            }
            return sufficient
        } catch {
            update {
                $0.spaceChecks += 1; $0.availableSpaceBytes = nil
                $0.spaceConstrained = true
                $0.spaceQueryFailures += 1; $0.rejected += 1
            }
            return false
        }
    }

    private static func sampleAvailableSpace(directoryFD: Int32) throws -> UInt64 {
        var fs = statvfs()
        guard fstatvfs(directoryFD, &fs) == 0 else { throw io("sample available snapshot space") }
        guard let blocks = UInt64(exactly: fs.f_bavail),
              let fragment = UInt64(exactly: fs.f_frsize) else {
            throw StoreError.io(operation: "sample available snapshot space", code: EOVERFLOW)
        }
        return try availableSpaceByteCount(blocks: blocks, fragmentBytes: fragment)
    }

    /// Kept internal so arithmetic edge cases can be checked without changing
    /// filesystem state or requiring an unusually large physical volume.
    static func availableSpaceByteCount(blocks: UInt64, fragmentBytes: UInt64) throws -> UInt64 {
        guard fragmentBytes > 0 else {
            throw StoreError.io(operation: "sample available snapshot space", code: EINVAL)
        }
        let (bytes, overflow) = blocks.multipliedReportingOverflow(by: fragmentBytes)
        guard !overflow else {
            throw StoreError.io(operation: "sample available snapshot space", code: EOVERFLOW)
        }
        return bytes
    }

    private func lookupOnQueue(tokens: [Int32], namespace: String,
                               maxPrefixTokens: Int?, submittedEpoch: UInt64) -> QwenPrefixDiskMatch? {
        guard isCurrent(submittedEpoch), directoryFD >= 0 else {
            update { $0.misses += 1 }; return nil
        }
        removeExpired()
        while let match = index.peek(tokens: tokens, namespace: namespace,
                                     maxPrefixTokens: maxPrefixTokens) {
            let name = match.value
            do {
                let decoded = try decodeFile(name, retainPayload: true)
                guard decoded.manifest.namespace == namespace,
                      decoded.manifest.tokens == Array(tokens.prefix(match.prefixTokenCount)),
                      !isExpired(decoded.manifest), isCurrent(submittedEpoch) else {
                    throw StoreError.corruptFile
                }
                order &+= 1; records[name]?.order = order
                _ = index.lookup(tokens: tokens, namespace: namespace,
                                 maxPrefixTokens: maxPrefixTokens)
                touchFile(name)
                update { $0.hits += 1; $0.bytesRead += decoded.bytes }
                return .init(prefixTokenCount: match.prefixTokenCount,
                             metadata: decoded.manifest.metadata,
                             payload: decoded.payload, diskBytes: decoded.bytes)
            } catch {
                removeRecord(name)
                update { $0.corruptions += 1 }
                guard isCurrent(submittedEpoch) else { break }
            }
        }
        update { $0.misses += 1 }; return nil
    }

    private func recoverDirectory() throws {
        var valid: [(String, Decoded)] = []
        // Startup verifies payloads in bounded chunks; payload Data is not kept.
        // Only bounded metadata records survive each file's verification.
        try forEachOwnedName { name in
            if Self.isTemporaryName(name) { try removeOwnedFile(name); return }
            let decoded: Decoded
            do { decoded = try decodeFile(name, retainPayload: false) }
            catch {
                try removeOwnedFile(name); update { $0.corruptions += 1 }; return
            }
            if isExpired(decoded.manifest) {
                try removeOwnedFile(name); update { $0.expired += 1 }; return
            }
            valid.append((name, decoded))
            // Bound retained startup metadata by every configured quota.
            while valid.count > limits.maxEntries ||
                    Self.exceeds(valid.map { $0.1.bytes }, limit: limits.maxBytes) ||
                    Self.exceeds(valid.map { $0.1.manifest.tokens.count }, limit: limits.maxKeyTokens) {
                valid.sort { $0.1.modified > $1.1.modified }
                let discarded = valid.removeLast()
                try removeOwnedFile(discarded.0)
                update { $0.evictions += 1 }
            }
        }
        for (name, value) in valid.sorted(by: { $0.1.modified < $1.1.modified }) {
            guard makeRoom(bytes: value.bytes, tokens: value.manifest.tokens.count, entries: 1) else {
                throw Self.io("remove over-capacity snapshot")
            }
            insertRecord(name: name, manifest: value.manifest, bytes: value.bytes, payloadBytes: value.payloadBytes)
            update { $0.recoveredEntries += 1 }
        }
        guard fsync(directoryFD) == 0 else { throw Self.io("fsync recovered directory") }
    }

    private func decodeFile(_ name: String, retainPayload: Bool) throws -> Decoded {
        guard Self.isEntryName(name) else { throw StoreError.corruptFile }
        let fd = openat(directoryFD, name, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
        guard fd >= 0 else { throw Self.io("open snapshot") }
        defer { Darwin.close(fd) }
        var info = stat()
        guard fstat(fd, &info) == 0, Self.isPrivateRegular(info),
              info.st_size >= Self.headerBytes else { throw StoreError.corruptFile }
        let cost = try Self.diskCost(info)
        guard cost <= limits.maxBytes else { throw StoreError.corruptFile }
        let header = try Self.readExact(Self.headerBytes, fd: fd)
        guard header.prefix(8) == Self.magic, Self.u32(header, at: 8) == 1,
              let manifestLength = Int(exactly: Self.u64(header, at: 12)),
              let payloadLength = Int(exactly: Self.u64(header, at: 20)),
              manifestLength > 0, manifestLength <= Self.maxManifestBytes,
              payloadLength > 0, payloadLength <= limits.maxPendingBytes,
              try Self.sum(Self.headerBytes, manifestLength, payloadLength) == Int(info.st_size) else {
            throw StoreError.corruptFile
        }
        let encoded = try Self.readExact(manifestLength, fd: fd)
        guard Data(SHA256.hash(data: encoded)) == header.subdata(in: 28..<60) else {
            throw StoreError.corruptFile
        }
        let manifest = try JSONDecoder().decode(Manifest.self, from: encoded)
        guard manifest.version == 1, !manifest.namespace.isEmpty,
              manifest.namespace.utf8.count <= Self.maxNamespaceBytes,
              !manifest.tokens.isEmpty, manifest.tokens.count <= limits.maxKeyTokens,
              manifest.metadata.count <= limits.maxMetadataBytes,
              manifest.metadata.count <= limits.maxPendingBytes - payloadLength,
              manifest.metadataSHA256 == Self.digest(manifest.metadata),
              manifest.createdAt.isFinite,
              manifest.expiresAt.map({ $0.isFinite && $0 >= manifest.createdAt }) ?? true,
              Self.fileName(tokens: manifest.tokens, namespace: manifest.namespace) == name else {
            throw StoreError.corruptFile
        }
        var hasher = SHA256()
        var remaining = payloadLength
        var payload = Data()
        if retainPayload { payload.reserveCapacity(payloadLength) }
        while remaining > 0 {
            let chunk = try Self.readExact(min(1_048_576, remaining), fd: fd)
            hasher.update(data: chunk)
            if retainPayload { payload.append(chunk) }
            remaining -= chunk.count
        }
        guard Data(hasher.finalize()) == header.subdata(in: 60..<92) else {
            throw StoreError.corruptFile
        }
        // A writer outside the process lock may truncate/replace concurrently.
        // Verify the opened inode's final size/identity and reject extra data.
        var final = stat()
        guard fstat(fd, &final) == 0, final.st_size == info.st_size,
              final.st_ino == info.st_ino, final.st_mtimespec.tv_sec == info.st_mtimespec.tv_sec,
              final.st_mtimespec.tv_nsec == info.st_mtimespec.tv_nsec else {
            throw StoreError.corruptFile
        }
        return Decoded(manifest: manifest, payload: payload, bytes: cost,
                       modified: Double(info.st_mtimespec.tv_sec) + Double(info.st_mtimespec.tv_nsec) / 1e9,
                       payloadBytes: payloadLength)
    }

    private func insertRecord(name: String, manifest: Manifest, bytes: Int, payloadBytes: Int) {
        stageRecord(name: name, manifest: manifest, bytes: bytes, payloadBytes: payloadBytes)
        refreshAccounting()
    }
    private func stageRecord(name: String, manifest: Manifest, bytes: Int, payloadBytes: Int) {
        order &+= 1
        records[name] = Record(manifest: manifest, bytes: bytes, payloadBytes: payloadBytes, order: order)
        let accepted = index.insert(tokens: manifest.tokens, namespace: manifest.namespace,
                                    value: name, logicalPayloadBytes: bytes)
        precondition(accepted, "validated SSD cache record exceeded limits")
    }
    @discardableResult
    private func makeRoom(bytes: Int, tokens: Int, entries: Int) -> Bool {
        while records.count > limits.maxEntries - entries ||
                records.values.reduce(0, { $0 + $1.bytes }) > limits.maxBytes - bytes ||
                records.values.reduce(0, { $0 + $1.manifest.tokens.count }) > limits.maxKeyTokens - tokens {
            guard let victim = records.min(by: { $0.value.order < $1.value.order })?.key else { return false }
            guard removeRecord(victim) else { return false }
            update { $0.evictions += 1 }
        }
        return true
    }
    @discardableResult
    private func removeRecord(_ name: String) -> Bool {
        if let record = records[name] {
            _ = index.remove(tokens: record.manifest.tokens, namespace: record.manifest.namespace)
        }
        guard unlinkat(directoryFD, name, 0) == 0 || errno == ENOENT else {
            // Keep the unreachable file's accounting and stop new admissions.
            update { $0.storageUnavailable = true; $0.writeFailures += 1 }
            refreshAccounting()
            return false
        }
        records.removeValue(forKey: name)
        refreshAccounting()
        return true
    }
    private func isExpired(_ manifest: Manifest) -> Bool {
        let now = clock()
        guard now.isFinite, manifest.createdAt <= now + 60 else { return true }
        return (manifest.expiresAt.map { now >= $0 } ?? false) ||
            (ttlSeconds.map { now - manifest.createdAt >= $0 } ?? false)
    }
    private func removeExpired() {
        for (name, record) in records where isExpired(record.manifest) {
            removeRecord(name); update { $0.expired += 1 }
        }
    }
    private func touchFile(_ name: String) {
        let fd = openat(directoryFD, name, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
        guard fd >= 0 else { return }
        defer { Darwin.close(fd) }
        var info = stat()
        guard fstat(fd, &info) == 0, Self.isPrivateRegular(info) else { return }
        _ = futimens(fd, nil)
    }
    private func roundedAllocation(_ bytes: Int) throws -> Int {
        let padding = fileAllocationUnit - 1
        let sum = try Self.sum(bytes, padding)
        return sum / fileAllocationUnit * fileAllocationUnit
    }
    private func removeOwnedTemporaryFiles() -> Bool {
        do {
            try forEachOwnedName { name in
                if Self.isTemporaryName(name) { try removeOwnedFile(name) }
            }
            return true
        } catch { return false }
    }
    private func removeOwnedFile(_ name: String) throws {
        guard unlinkat(directoryFD, name, 0) == 0 || errno == ENOENT else {
            throw Self.io("remove owned snapshot")
        }
    }
    private func forEachOwnedName(_ body: (String) throws -> Void) throws {
        let copy = dup(directoryFD)
        guard copy >= 0 else { throw Self.io("duplicate directory") }
        guard let stream = fdopendir(copy) else { Darwin.close(copy); throw Self.io("read directory") }
        defer { closedir(stream) }
        rewinddir(stream)
        while true {
            errno = 0
            guard let entry = readdir(stream) else {
                if errno != 0 { throw Self.io("enumerate cache directory") }
                break
            }
            let name = withUnsafePointer(to: &entry.pointee.d_name) {
                $0.withMemoryRebound(to: CChar.self, capacity: Int(entry.pointee.d_namlen) + 1) {
                    String(cString: $0)
                }
            }
            if Self.isEntryName(name) || Self.isTemporaryName(name) { try body(name) }
        }
    }
    private func closeDescriptors() {
        if lockFD >= 0 { _ = flock(lockFD, LOCK_UN); Darwin.close(lockFD); lockFD = -1 }
        if directoryFD >= 0 { Darwin.close(directoryFD); directoryFD = -1 }
    }
    private static func openPrivateDirectory(_ url: URL) throws -> Int32 {
        let path = url.path
        guard path.hasPrefix("/"), !path.utf8.contains(0) else { throw StoreError.unsafeDirectory }
        let components = path.split(separator: "/", omittingEmptySubsequences: true).map(String.init)
        guard !components.isEmpty, components.allSatisfy({ $0 != "." && $0 != ".." }) else {
            throw StoreError.unsafeDirectory
        }
        var fd = Darwin.open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC)
        guard fd >= 0 else { throw io("open filesystem root") }
        do {
            for (i, component) in components.enumerated() {
                var next = openat(fd, component, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
                if next < 0, errno == ENOENT, i == components.count - 1 {
                    guard mkdirat(fd, component, mode_t(0o700)) == 0 || errno == EEXIST else {
                        throw io("create cache directory")
                    }
                    next = openat(fd, component, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
                }
                guard next >= 0 else { throw StoreError.unsafeDirectory }
                Darwin.close(fd); fd = next
            }
            var info = stat()
            guard fstat(fd, &info) == 0, info.st_uid == geteuid(),
                  info.st_mode & mode_t(0o077) == 0 else { throw StoreError.unsafeDirectory }
            return fd
        } catch { Darwin.close(fd); throw error }
    }
    private static func isPrivateRegular(_ info: stat) -> Bool {
        (info.st_mode & S_IFMT) == S_IFREG && info.st_uid == geteuid() &&
            (info.st_mode & mode_t(0o077)) == 0 && info.st_nlink == 1
    }
    private static func isEntryName(_ name: String) -> Bool {
        guard name.hasPrefix(prefix), name.hasSuffix(".qpc") else { return false }
        let digest = name.dropFirst(prefix.count).dropLast(4)
        return digest.count == 64 && digest.allSatisfy { "0123456789abcdef".contains($0) }
    }
    private static func isTemporaryName(_ name: String) -> Bool {
        guard name.hasPrefix(prefix), name.hasSuffix(".tmp") else { return false }
        let uuid = String(name.dropFirst(prefix.count).dropLast(4))
        return uuid.count == 36 && UUID(uuidString: uuid) != nil
    }
    private static func fileName(tokens: [Int32], namespace: String) -> String {
        var hasher = SHA256()
        var length = Data(); appendLE(UInt64(namespace.utf8.count), to: &length)
        hasher.update(data: length); hasher.update(data: Data(namespace.utf8))
        var tokenData = Data(capacity: tokens.count * 4)
        for token in tokens { appendLE(UInt32(bitPattern: token), to: &tokenData) }
        hasher.update(data: tokenData)
        return prefix + hasher.finalize().map { String(format: "%02x", $0) }.joined() + ".qpc"
    }
    private static func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
    private static func appendLE<T: FixedWidthInteger>(_ value: T, to data: inout Data) {
        var little = value.littleEndian
        withUnsafeBytes(of: &little) { data.append(contentsOf: $0) }
    }
    private static func u32(_ data: Data, at offset: Int) -> UInt32 {
        (0..<4).reduce(0) { $0 | UInt32(data[offset + $1]) << (8 * $1) }
    }
    private static func u64(_ data: Data, at offset: Int) -> UInt64 {
        (0..<8).reduce(0) { $0 | UInt64(data[offset + $1]) << (8 * $1) }
    }
    private static func exceeds(_ values: [Int], limit: Int) -> Bool {
        var remaining = limit
        for value in values {
            if value > remaining { return true }
            remaining -= value
        }
        return false
    }
    private static func sum(_ values: Int...) throws -> Int {
        var total = 0
        for value in values {
            let (next, overflow) = total.addingReportingOverflow(value)
            guard value >= 0, !overflow else { throw StoreError.corruptFile }; total = next
        }
        return total
    }
    private static func diskCost(_ info: stat) throws -> Int {
        guard info.st_size >= 0, info.st_blocks >= 0,
              let size = Int(exactly: info.st_size), let blocks = Int(exactly: info.st_blocks) else {
            throw StoreError.corruptFile
        }
        let (allocated, overflow) = blocks.multipliedReportingOverflow(by: 512)
        guard !overflow else { throw StoreError.corruptFile }
        return max(size, allocated)
    }
    private static func writeAll(_ data: Data, fd: Int32) throws {
        try data.withUnsafeBytes { buffer in
            var offset = 0
            while offset < buffer.count {
                let count = Darwin.write(fd, buffer.baseAddress!.advanced(by: offset), buffer.count - offset)
                if count < 0, errno == EINTR { continue }
                guard count > 0 else { throw io("write snapshot") }; offset += count
            }
        }
    }
    private static func readExact(_ count: Int, fd: Int32) throws -> Data {
        var data = Data(count: count)
        try data.withUnsafeMutableBytes { buffer in
            var offset = 0
            while offset < count {
                let n = Darwin.read(fd, buffer.baseAddress!.advanced(by: offset), count - offset)
                if n < 0, errno == EINTR { continue }
                guard n > 0 else { throw StoreError.corruptFile }; offset += n
            }
        }
        return data
    }
    private static func io(_ operation: String) -> StoreError {
        .io(operation: operation, code: errno)
    }
}
