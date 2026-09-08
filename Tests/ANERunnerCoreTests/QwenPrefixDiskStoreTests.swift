import XCTest
import Foundation
import CryptoKit
import Darwin
@testable import ANERunnerCore

final class QwenPrefixDiskStoreTests: XCTestCase {
    private final class Clock: @unchecked Sendable {
        private let lock = NSLock()
        private var value: TimeInterval = 1_000
        func now() -> TimeInterval { lock.lock(); defer { lock.unlock() }; return value }
        func advance(_ seconds: TimeInterval) { lock.lock(); value += seconds; lock.unlock() }
    }
    private final class CompletionBox: @unchecked Sendable {
        let lock = NSLock()
        var values: [Bool] = []
        func append(_ value: Bool) { lock.lock(); values.append(value); lock.unlock() }
        func read() -> [Bool] { lock.lock(); defer { lock.unlock() }; return values }
    }
    private final class BlockingClock: @unchecked Sendable {
        let entered = DispatchSemaphore(value: 0), resume = DispatchSemaphore(value: 0)
        private let lock = NSLock()
        private var shouldBlock = false
        func blockNext() { lock.lock(); shouldBlock = true; lock.unlock() }
        func now() -> TimeInterval {
            lock.lock(); let block = shouldBlock; shouldBlock = false; lock.unlock()
            if block { entered.signal(); resume.wait() }
            return 1_000
        }
    }
    private final class SpaceProbe: @unchecked Sendable {
        let entered = DispatchSemaphore(value: 0), resume = DispatchSemaphore(value: 0)
        private let lock = NSLock()
        private var samples: [UInt64?]
        private var calls = 0
        private var shouldBlock = false
        init(_ samples: [UInt64?]) { precondition(!samples.isEmpty); self.samples = samples }
        func set(_ samples: [UInt64?]) {
            precondition(!samples.isEmpty)
            lock.lock(); self.samples = samples; lock.unlock()
        }
        func blockNext() { lock.lock(); shouldBlock = true; lock.unlock() }
        func count() -> Int { lock.lock(); defer { lock.unlock() }; return calls }
        func sample(_ fd: Int32) throws -> UInt64 {
            // Verify the injection point borrows a live directory descriptor.
            var info = stat()
            guard fstat(fd, &info) == 0, info.st_mode & S_IFMT == S_IFDIR else {
                throw QwenPrefixDiskStore.StoreError.unsafeDirectory
            }
            lock.lock()
            calls += 1
            let value = samples.count > 1 ? samples.removeFirst() : samples[0]
            let block = shouldBlock; shouldBlock = false
            lock.unlock()
            if block { entered.signal(); resume.wait() }
            guard let value else {
                throw QwenPrefixDiskStore.StoreError.io(operation: "injected space sample", code: EIO)
            }
            return value
        }
    }
    private final class PublicationSyncProbe: @unchecked Sendable {
        let entered = DispatchSemaphore(value: 0), resume = DispatchSemaphore(value: 0)
        private let lock = NSLock()
        private var shouldBlock = false
        private var shouldFail = false
        func blockNext(fail: Bool = false) {
            lock.lock(); shouldBlock = true; shouldFail = fail; lock.unlock()
        }
        func sync(_ fd: Int32) throws {
            lock.lock()
            let block = shouldBlock, fail = shouldFail
            shouldBlock = false; shouldFail = false
            lock.unlock()
            if block {
                entered.signal()
                guard resume.wait(timeout: .now() + 10) == .success else {
                    throw QwenPrefixDiskStore.StoreError.io(operation: "publication test watchdog", code: ETIMEDOUT)
                }
            }
            if fail {
                throw QwenPrefixDiskStore.StoreError.io(operation: "injected publication sync", code: EIO)
            }
            guard fsync(fd) == 0 else {
                throw QwenPrefixDiskStore.StoreError.io(operation: "test directory sync", code: errno)
            }
        }
    }
    private final class CallbackPause: @unchecked Sendable {
        let entered = DispatchSemaphore(value: 0), resume = DispatchSemaphore(value: 0)
        func hold() -> Bool {
            entered.signal()
            return resume.wait(timeout: .now() + 10) == .success
        }
    }
    private func directory() throws -> URL {
        // Foundation resolvingSymlinksInPath canonicalizes /private/var back
        // to the /var alias on macOS. Use the POSIX path without resolving it
        // again so the production O_NOFOLLOW traversal sees real directories.
        let resolved = try XCTUnwrap(Darwin.realpath(FileManager.default.temporaryDirectory.path, nil))
        defer { free(resolved) }
        let physicalTemporaryPath = String(cString: resolved)
        let name = "qwen-prefix-disk-tests-" + UUID().uuidString
        let root = URL(fileURLWithPath: physicalTemporaryPath, isDirectory: true)
            .appendingPathComponent(name, isDirectory: true)
        XCTAssertEqual(root.path, physicalTemporaryPath + "/" + name)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false,
                                               attributes: [.posixPermissions: 0o700])
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        return root
    }
    private func limits(entries: Int = 8, bytes: Int = 1_048_576,
                        tokens: Int = 1_000, pending: Int = 524_288,
                        minimumFree: Int = 0) -> QwenPrefixDiskLimits {
        .init(maxEntries: entries, maxBytes: bytes, maxKeyTokens: tokens,
              maxPendingJobs: 2, maxPendingBytes: pending, maxMetadataBytes: 1_024,
              minAvailableBytes: minimumFree)
    }
    private func store(_ directory: URL, entries: Int = 8, bytes: Int = 1_048_576,
                       tokens: Int = 1_000) throws -> QwenPrefixDiskStore {
        let store = try QwenPrefixDiskStore(directory: directory,
                                            limits: limits(entries: entries, bytes: bytes, tokens: tokens))
        addTeardownBlock { store.close() }
        return store
    }
    @discardableResult
    private func put(_ store: QwenPrefixDiskStore, _ tokens: [Int32],
                     namespace: String = "model/runtime/tenant-a", byte: UInt8 = 7,
                     bytes: Int = 200) -> Bool {
        let admitted = store.enqueue(tokens: tokens, namespace: namespace,
                                     metadata: Data("manifest".utf8), payload: Data(repeating: byte, count: bytes))
        store.flush()
        return admitted
    }
    private func files(_ directory: URL) throws -> [URL] {
        try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "qpc" }
    }
    private func lookup(_ store: QwenPrefixDiskStore, _ tokens: [Int32],
                        namespace: String = "model/runtime/tenant-a", cap: Int? = nil) -> QwenPrefixDiskMatch? {
        store.lookup(tokens: tokens, namespace: namespace, maxPrefixTokens: cap)
    }

    func testPersistentRoundTripExactPrefixNamespaceAndPeek() throws {
        let dir = try directory()
        let first = try store(dir)
        XCTAssertTrue(put(first, [1, 2], byte: 11))
        XCTAssertTrue(put(first, [1, 2, 3, 4], byte: 22))
        XCTAssertTrue(put(first, [1, 2], namespace: "tenant-b", byte: 33))
        let before = first.statistics
        let summary = try XCTUnwrap(first.peek(tokens: [1, 2, 3, 4, 5], namespace: "model/runtime/tenant-a"))
        XCTAssertEqual(summary.prefixTokenCount, 4)
        XCTAssertEqual(summary.payloadBytes, 200)
        XCTAssertGreaterThan(summary.diskBytes, summary.payloadBytes)
        XCTAssertEqual(first.statistics, before)
        XCTAssertEqual(lookup(first, [1, 2, 3, 4, 5])?.payload, Data(repeating: 22, count: 200))
        XCTAssertEqual(lookup(first, [1, 2, 3, 4, 5], cap: 3)?.prefixTokenCount, 2)
        XCTAssertNil(lookup(first, [1, 2], namespace: "tenant-c"))
        first.close()
        let reopened = try store(dir)
        XCTAssertEqual(reopened.statistics.recoveredEntries, 3)
        XCTAssertEqual(lookup(reopened, [1, 2], namespace: "tenant-b")?.payload.first, 33)
        XCTAssertEqual(lookup(reopened, [1, 2, 3, 4])?.metadata, Data("manifest".utf8))
        XCTAssertEqual(reopened.statistics.entries, 3)
    }

    func testLRUEntryTokenAndDiskCapacity() throws {
        let first = try store(directory(), entries: 2)
        put(first, [1]); put(first, [2])
        XCTAssertNotNil(lookup(first, [1]))
        put(first, [3])
        XCTAssertNotNil(lookup(first, [1]))
        XCTAssertNil(lookup(first, [2]))
        XCTAssertEqual(first.statistics.entries, 2)
        XCTAssertEqual(first.statistics.evictions, 1)

        let keys = try store(directory(), tokens: 3)
        put(keys, [1, 2]); put(keys, [3, 4])
        XCTAssertNil(lookup(keys, [1, 2]))
        XCTAssertEqual(keys.statistics.keyTokens, 2)

        let disk = try store(directory(), bytes: 32_768)
        put(disk, [1], bytes: 12_000); put(disk, [2], bytes: 12_000); put(disk, [3], bytes: 12_000)
        XCTAssertLessThanOrEqual(disk.statistics.diskBytes, 32_768)
        XCTAssertGreaterThan(disk.statistics.evictions, 0)
        XCTAssertEqual(lookup(disk, [3])?.payload.count, 12_000)
    }

    func testBadPayloadHashFallsBackToShorterPrefixAndDeletesOnlyBadEntry() throws {
        let dir = try directory(), cache = try store(dir)
        put(cache, [1, 2], byte: 9); put(cache, [1, 2, 3, 4], byte: 10)
        let entries = try files(dir)
        for entry in entries {
            var data = try Data(contentsOf: entry)
            if data.last == 10 { data[data.count - 1] ^= 1; try data.write(to: entry) }
        }
        let hit = try XCTUnwrap(lookup(cache, [1, 2, 3, 4, 5]))
        XCTAssertEqual(hit.prefixTokenCount, 2)
        XCTAssertEqual(hit.payload.first, 9)
        XCTAssertEqual(cache.statistics.corruptions, 1)
        XCTAssertEqual(try files(dir).count, 1)
    }

    func testManifestChecksumShortFileAndVersionFailClosed() throws {
        for corruption in 0..<3 {
            let dir = try directory(), cache = try store(dir)
            put(cache, [1])
            let file = try XCTUnwrap(files(dir).first)
            var data = try Data(contentsOf: file)
            if corruption == 0 { data[100] ^= 1 }
            else if corruption == 1 { data = data.prefix(30) }
            else { data[8] = 99 }
            try data.write(to: file)
            XCTAssertNil(lookup(cache, [1]))
            XCTAssertEqual(cache.statistics.entries, 0)
            XCTAssertEqual(cache.statistics.corruptions, 1)
        }
    }

    func testSwappedValidContainerCannotMasqueradeAsAnotherTokenKey() throws {
        let dir = try directory(), cache = try store(dir)
        put(cache, [1], byte: 1); put(cache, [2], byte: 2)
        let names = try files(dir)
        let first = try Data(contentsOf: names[0]), second = try Data(contentsOf: names[1])
        try second.write(to: names[0]); try first.write(to: names[1])
        XCTAssertNil(lookup(cache, [1])); XCTAssertNil(lookup(cache, [2]))
        XCTAssertEqual(cache.statistics.corruptions, 2)
    }

    func testSymlinkEntryAndHardlinkFailClosedWithoutTouchingTarget() throws {
        let dir = try directory(), cache = try store(dir)
        let outside = dir.appendingPathComponent("user-valuable-file")
        let valuable = Data("do not change".utf8)
        try valuable.write(to: outside)
        XCTAssertEqual(chmod(outside.path, mode_t(0o600)), 0)
        put(cache, [1])
        let entry = try XCTUnwrap(files(dir).first)
        try FileManager.default.removeItem(at: entry)
        try FileManager.default.createSymbolicLink(at: entry, withDestinationURL: outside)
        XCTAssertNil(lookup(cache, [1]))
        XCTAssertEqual(try Data(contentsOf: outside), valuable)
        put(cache, [2])
        let original = try XCTUnwrap(files(dir).first)
        let hardlink = dir.appendingPathComponent("user-hardlink")
        XCTAssertEqual(link(original.path, hardlink.path), 0)
        XCTAssertNil(lookup(cache, [2]))
        XCTAssertTrue(FileManager.default.fileExists(atPath: hardlink.path))
    }

    func testUnsafeDirectoryAndExclusiveProcessLock() throws {
        let dir = try directory()
        let first = try store(dir)
        XCTAssertThrowsError(try QwenPrefixDiskStore(directory: dir, limits: limits())) {
            XCTAssertEqual($0 as? QwenPrefixDiskStore.StoreError, .directoryInUse)
        }
        first.close()
        let linked = dir.appendingPathComponent("linked")
        let real = dir.appendingPathComponent("real")
        try FileManager.default.createDirectory(at: real, withIntermediateDirectories: false,
                                               attributes: [.posixPermissions: 0o700])
        try FileManager.default.createSymbolicLink(at: linked, withDestinationURL: real)
        XCTAssertThrowsError(try QwenPrefixDiskStore(directory: linked, limits: limits()))
        XCTAssertEqual(chmod(real.path, mode_t(0o755)), 0)
        XCTAssertThrowsError(try QwenPrefixDiskStore(directory: real, limits: limits()))
    }

    func testTTLExpirationAndRestartRespectNewShorterTTL() throws {
        let dir = try directory(), clock = Clock()
        let cache = try QwenPrefixDiskStore(directory: dir, limits: limits(), ttlSeconds: 30, now: { clock.now() })
        put(cache, [1]); clock.advance(20)
        XCTAssertNotNil(lookup(cache, [1]))
        cache.close()
        let reopened = try QwenPrefixDiskStore(directory: dir, limits: limits(), ttlSeconds: 10, now: { clock.now() })
        defer { reopened.close() }
        XCTAssertEqual(reopened.statistics.entries, 0)
        XCTAssertEqual(reopened.statistics.expired, 1)
        put(reopened, [2]); clock.advance(11)
        XCTAssertNil(reopened.peek(tokens: [2], namespace: "model/runtime/tenant-a"))
        XCTAssertNil(lookup(reopened, [2]))
        XCTAssertEqual(reopened.statistics.expired, 2)
    }

    func testStartupRemovesOnlyOwnedResidueAndCorruptFiles() throws {
        let dir = try directory(), cache = try store(dir)
        put(cache, [1]); cache.close()
        let temporary = dir.appendingPathComponent("qwen-prefix-v1-" + UUID().uuidString.lowercased() + ".tmp")
        let unrelated = dir.appendingPathComponent("qwen-prefix-v1-user-notes.tmp")
        let corrupt = dir.appendingPathComponent("qwen-prefix-v1-" + String(repeating: "f", count: 64) + ".qpc")
        try Data([1]).write(to: temporary); try Data([2]).write(to: unrelated); try Data([3]).write(to: corrupt)
        let reopened = try store(dir)
        XCTAssertFalse(FileManager.default.fileExists(atPath: temporary.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: corrupt.path))
        XCTAssertEqual(try Data(contentsOf: unrelated), Data([2]))
        XCTAssertNotNil(lookup(reopened, [1]))
        reopened.clear()
        XCTAssertEqual(try Data(contentsOf: unrelated), Data([2]))
        XCTAssertTrue(try files(dir).isEmpty)
    }

    func testOversizeAdmissionDoesNotEvictAndRejectedWriteHasNoCompletion() throws {
        let dir = try directory(), cache = try store(dir, bytes: 32_768)
        put(cache, [1])
        let box = CompletionBox()
        XCTAssertFalse(cache.enqueue(tokens: [2], namespace: "a", metadata: Data(),
                                     payload: Data(repeating: 0, count: 40_000),
                                     completion: { box.append($0) }))
        cache.flush()
        XCTAssertTrue(box.read().isEmpty)
        XCTAssertNotNil(lookup(cache, [1]))
        XCTAssertEqual(cache.statistics.evictions, 0)
        XCTAssertEqual(cache.statistics.pendingBytes, 0)
        XCTAssertEqual(cache.statistics.rejected, 1)
    }

    func testAcceptedWriteCompletesOnceAndInvalidationIsExact() throws {
        let dir = try directory(), cache = try store(dir), box = CompletionBox()
        XCTAssertTrue(cache.enqueue(tokens: [1, 2], namespace: "a", metadata: Data([1]),
                                    payload: Data([2]), completion: { box.append($0) }))
        cache.flush()
        XCTAssertEqual(box.read(), [true])
        cache.invalidate(tokens: [1, 2, 3], namespace: "a")
        XCTAssertNotNil(lookup(cache, [1, 2], namespace: "a"))
        cache.invalidate(tokens: [1, 2], namespace: "a")
        XCTAssertNil(lookup(cache, [1, 2], namespace: "a"))
        XCTAssertEqual(cache.statistics.pendingJobs, 0)
    }

    func testClearEpochNeverRepublishesPendingOldWrites() throws {
        let cache = try store(directory()), box = CompletionBox()
        for i: Int32 in 1...2 {
            _ = cache.enqueue(tokens: [i], namespace: "a", metadata: Data(),
                              payload: Data(repeating: 1, count: 200_000), completion: { box.append($0) })
        }
        cache.clear(resetStatistics: true)
        cache.flush()
        XCTAssertEqual(cache.statistics.entries, 0)
        XCTAssertEqual(cache.statistics.pendingJobs, 0)
        XCTAssertEqual(cache.statistics.pendingBytes, 0)
        XCTAssertNil(lookup(cache, [1], namespace: "a"))
        XCTAssertTrue(put(cache, [3]))
        XCTAssertNotNil(lookup(cache, [3]))
    }

    func testAsyncReadAdmissionRemainsChargedUntilCallbackReturns() throws {
        let cache = try store(directory())
        put(cache, [1])
        let entered = DispatchSemaphore(value: 0), release = DispatchSemaphore(value: 0)
        let box = CompletionBox()
        XCTAssertTrue(cache.lookupAsync(tokens: [1], namespace: "model/runtime/tenant-a") { match in
            box.append(match?.payload.count == 200)
            entered.signal(); release.wait()
        })
        XCTAssertEqual(entered.wait(timeout: .now() + 5), .success)
        XCTAssertEqual(cache.statistics.pendingBytes, cache.limits.maxPendingBytes)
        XCTAssertFalse(cache.enqueue(tokens: [2], namespace: "a", metadata: Data(), payload: Data([1])))
        XCTAssertFalse(cache.lookupAsync(tokens: [1], namespace: "a") { _ in })
        release.signal(); cache.flush()
        XCTAssertEqual(box.read(), [true])
        XCTAssertEqual(cache.statistics.pendingBytes, 0)
    }

    func testCloseDrainsDurablyAndRejectsNewJobs() throws {
        let dir = try directory(), cache = try store(dir), box = CompletionBox()
        XCTAssertTrue(cache.enqueue(tokens: [1], namespace: "a", metadata: Data(),
                                    payload: Data([42]), completion: { box.append($0) }))
        cache.close(drain: true); cache.flush()
        XCTAssertEqual(box.read(), [true])
        XCTAssertFalse(cache.enqueue(tokens: [2], namespace: "a", metadata: Data(), payload: Data([1])))
        let reopened = try store(dir)
        XCTAssertEqual(lookup(reopened, [1], namespace: "a")?.payload, Data([42]))
    }

    func testRestartWithSmallerLimitsPreservesNewestWithinEveryQuota() throws {
        let dir = try directory(), first = try store(dir)
        for i: Int32 in 1...4 { put(first, [i, i]) }
        first.close()
        let second = try store(dir, entries: 2, tokens: 3)
        XCTAssertLessThanOrEqual(second.statistics.entries, 2)
        XCTAssertLessThanOrEqual(second.statistics.keyTokens, 3)
        XCTAssertLessThanOrEqual(second.statistics.diskBytes, second.limits.maxBytes)
        XCTAssertEqual(second.statistics.entries, try files(dir).count)
    }
    func testWriteAndRemovalFailuresDisableAccumulationAndClearCanRecover() throws {
        if geteuid() == 0 { throw XCTSkip("permission fault requires an unprivileged test process") }
        let dir = try directory(), cache = try store(dir), box = CompletionBox()
        put(cache, [1])
        XCTAssertEqual(chmod(dir.path, mode_t(0o500)), 0)
        defer { _ = chmod(dir.path, mode_t(0o700)) }
        XCTAssertTrue(cache.enqueue(tokens: [2], namespace: "a", metadata: Data(),
                                    payload: Data([2]), completion: { box.append($0) }))
        cache.flush()
        XCTAssertEqual(box.read(), [false])
        XCTAssertGreaterThan(cache.statistics.writeFailures, 0)
        cache.invalidate(tokens: [1], namespace: "model/runtime/tenant-a")
        XCTAssertTrue(cache.statistics.storageUnavailable)
        XCTAssertGreaterThan(cache.statistics.diskBytes, 0)
        XCTAssertFalse(cache.enqueue(tokens: [3], namespace: "a", metadata: Data(), payload: Data([3])))
        XCTAssertEqual(chmod(dir.path, mode_t(0o700)), 0)
        cache.clear()
        XCTAssertFalse(cache.statistics.storageUnavailable)
        XCTAssertEqual(cache.statistics.entries, 0)
        XCTAssertEqual(cache.statistics.diskBytes, 0)
        XCTAssertTrue(put(cache, [4]))
        XCTAssertNotNil(lookup(cache, [4]))
    }

    func testConcurrentAdmissionLookupAndClearKeepAccountingBounded() throws {
        let cache = try store(directory(), entries: 4), completions = CompletionBox()
        let admitted = CompletionBox(), group = DispatchGroup()
        for worker in 0..<4 {
            group.enter()
            DispatchQueue.global().async {
                for iteration in 0..<25 {
                    let key = Int32(worker * 100 + iteration)
                    if cache.enqueue(tokens: [key], namespace: "a", metadata: Data([1]),
                                     payload: Data(repeating: 2, count: 100),
                                     completion: { completions.append($0) }) {
                        admitted.append(true)
                    }
                    _ = cache.lookup(tokens: [key], namespace: "a")
                    if iteration % 7 == 0 { cache.clear() }
                }
                group.leave()
            }
        }
        XCTAssertEqual(group.wait(timeout: .now() + 30), .success)
        cache.flush()
        XCTAssertEqual(completions.read().count, admitted.read().count)
        XCTAssertLessThanOrEqual(cache.statistics.entries, cache.limits.maxEntries)
        XCTAssertLessThanOrEqual(cache.statistics.diskBytes, cache.limits.maxBytes)
        XCTAssertEqual(cache.statistics.pendingJobs, 0)
        cache.clear(); cache.flush()
        XCTAssertEqual(cache.statistics.entries, 0)
        XCTAssertEqual(cache.statistics.pendingBytes, 0)
    }

    func testDeinitializationWithoutExplicitCloseReleasesDirectoryLock() throws {
        let dir = try directory()
        weak var released: QwenPrefixDiskStore?
        do {
            let cache = try QwenPrefixDiskStore(directory: dir, limits: limits())
            released = cache
            put(cache, [1])
        }
        XCTAssertNil(released)
        let reopened = try store(dir)
        XCTAssertNotNil(lookup(reopened, [1]))
    }

    func testConcurrentCloseWaitsForSameDescriptorRelease() throws {
        let dir = try directory(), cache = try store(dir), group = DispatchGroup()
        XCTAssertTrue(cache.enqueue(tokens: [1], namespace: "a", metadata: Data(),
                                    payload: Data(repeating: 3, count: 200_000)))
        for _ in 0..<8 {
            group.enter()
            DispatchQueue.global().async { cache.close(drain: true); group.leave() }
        }
        XCTAssertEqual(group.wait(timeout: .now() + 10), .success)
        XCTAssertEqual(cache.statistics.pendingJobs, 0)
        let reopened = try store(dir)
        XCTAssertEqual(lookup(reopened, [1], namespace: "a")?.payload.count, 200_000)
    }

    func testAsyncInvalidationPreservesReplacementQueuedBeforeIt() throws {
        let dir = try directory(), clock = BlockingClock()
        let cache = try QwenPrefixDiskStore(directory: dir, limits: limits(), now: { clock.now() })
        defer { cache.close() }
        put(cache, [1], byte: 7)
        clock.blockNext()
        XCTAssertTrue(cache.enqueue(tokens: [1], namespace: "model/runtime/tenant-a",
                                    metadata: Data(), payload: Data([8])))
        XCTAssertEqual(clock.entered.wait(timeout: .now() + 5), .success)
        XCTAssertTrue(cache.invalidateAsync(tokens: [1], namespace: "model/runtime/tenant-a"))
        XCTAssertLessThanOrEqual(cache.statistics.pendingJobs, cache.limits.maxPendingJobs)
        // The replacement publishes before queued invalidation executes. The
        // invalidation's old revision must not delete this newly published file.
        clock.resume.signal(); cache.flush()
        XCTAssertEqual(lookup(cache, [1])?.payload, Data([8]))
        XCTAssertTrue(cache.invalidateAsync(tokens: [1], namespace: "model/runtime/tenant-a"))
        cache.flush()
        XCTAssertNil(lookup(cache, [1]))
    }

    func testAsyncInvalidationDoesNotDeleteLaterPublication() throws {
        let cache = try store(directory())
        put(cache, [1], byte: 7)
        XCTAssertTrue(cache.invalidateAsync(tokens: [1], namespace: "model/runtime/tenant-a"))
        XCTAssertTrue(cache.enqueue(tokens: [1], namespace: "model/runtime/tenant-a",
                                    metadata: Data(), payload: Data([9])))
        cache.flush()
        XCTAssertEqual(lookup(cache, [1])?.payload, Data([9]))
        XCTAssertFalse(cache.invalidateAsync(tokens: [999], namespace: "model/runtime/tenant-a"))
        XCTAssertEqual(cache.statistics.pendingJobs, 0)
    }

    func testSpaceFloorDefaultsLimitsValidationAndOldCodableCompatibility() throws {
        XCTAssertEqual(QwenPrefixDiskLimits().minAvailableBytes, 1_073_741_824)
        let old = Data("""
        {"maxEntries":8,"maxBytes":1048576,"maxKeyTokens":1000,
         "maxPendingJobs":2,"maxPendingBytes":524288,"maxMetadataBytes":1024}
        """.utf8)
        XCTAssertEqual(try JSONDecoder().decode(QwenPrefixDiskLimits.self, from: old).minAvailableBytes,
                       1_073_741_824)
        let disabled = limits(minimumFree: 0)
        XCTAssertEqual(try JSONDecoder().decode(QwenPrefixDiskLimits.self,
                       from: JSONEncoder().encode(disabled)), disabled)
        XCTAssertThrowsError(try QwenPrefixDiskStore(directory: directory(), limits: limits(minimumFree: -1))) {
            XCTAssertEqual($0 as? QwenPrefixDiskStore.StoreError, .invalidLimits)
        }
    }

    func testAvailableSpaceByteCountRejectsOverflowAndInvalidFragments() throws {
        XCTAssertEqual(try QwenPrefixDiskStore.availableSpaceByteCount(blocks: 0, fragmentBytes: 4096), 0)
        XCTAssertEqual(try QwenPrefixDiskStore.availableSpaceByteCount(blocks: 7, fragmentBytes: 4096), 28_672)
        XCTAssertEqual(try QwenPrefixDiskStore.availableSpaceByteCount(blocks: UInt64.max, fragmentBytes: 1),
                       UInt64.max)
        XCTAssertThrowsError(try QwenPrefixDiskStore.availableSpaceByteCount(blocks: 1, fragmentBytes: 0))
        XCTAssertThrowsError(try QwenPrefixDiskStore.availableSpaceByteCount(blocks: UInt64.max, fragmentBytes: 4096))
    }

    func testLowSpaceSkipsBeforeEvictionKeepsReadsAndAutomaticallyRecovers() throws {
        let dir = try directory(), samples = SpaceProbe([UInt64.max]), box = CompletionBox()
        let cache = try QwenPrefixDiskStore(directory: dir, limits: limits(entries: 1, minimumFree: 1024),
                                           availableSpace: { try samples.sample($0) })
        defer { cache.close() }
        put(cache, [1])
        samples.set([0])
        XCTAssertTrue(cache.enqueue(tokens: [2], namespace: "a", metadata: Data(), payload: Data([2]),
                                    completion: { box.append($0) }))
        cache.flush()
        XCTAssertEqual(box.read(), [false])
        XCTAssertTrue(cache.statistics.spaceConstrained)
        XCTAssertEqual(cache.statistics.spaceRejections, 1)
        XCTAssertEqual(cache.statistics.rejected, 1)
        XCTAssertEqual(cache.statistics.availableSpaceBytes, 0)
        XCTAssertEqual(cache.statistics.evictions, 0)
        XCTAssertFalse(cache.statistics.storageUnavailable)
        XCTAssertNotNil(lookup(cache, [1]))
        let reads = CompletionBox()
        XCTAssertTrue(cache.lookupAsync(tokens: [1], namespace: "model/runtime/tenant-a") {
            reads.append($0?.payload.count == 200)
        })
        cache.flush()
        XCTAssertEqual(reads.read(), [true])
        XCTAssertEqual(samples.count(), 3, "reads must not query or be denied by free-space policy")
        samples.set([UInt64.max])
        put(cache, [2])
        XCTAssertNotNil(lookup(cache, [2]))
        XCTAssertFalse(cache.statistics.spaceConstrained)
        XCTAssertEqual(cache.statistics.spaceRecoveries, 1)
        XCTAssertEqual(cache.statistics.pendingJobs, 0)
        XCTAssertEqual(cache.statistics.pendingBytes, 0)
    }

    func testSpaceQueryFailureIsObservableAndLaterWritesRetry() throws {
        let samples = SpaceProbe([nil]), box = CompletionBox()
        let cache = try QwenPrefixDiskStore(directory: directory(), limits: limits(minimumFree: 1),
                                           availableSpace: { try samples.sample($0) })
        defer { cache.close() }
        XCTAssertTrue(cache.enqueue(tokens: [1], namespace: "a", metadata: Data(), payload: Data([1]),
                                    completion: { box.append($0) }))
        cache.flush()
        XCTAssertEqual(box.read(), [false])
        XCTAssertEqual(cache.statistics.spaceQueryFailures, 1)
        XCTAssertEqual(cache.statistics.spaceRejections, 0)
        XCTAssertEqual(cache.statistics.writeFailures, 0)
        XCTAssertNil(cache.statistics.availableSpaceBytes)
        XCTAssertTrue(cache.statistics.spaceConstrained)
        XCTAssertFalse(cache.statistics.storageUnavailable)
        samples.set([UInt64.max])
        put(cache, [1])
        XCTAssertNotNil(lookup(cache, [1]))
        XCTAssertEqual(cache.statistics.spaceRecoveries, 1)
    }

    func testZeroFloorDisablesSpaceSamplerAndKeepsLogicalQuota() throws {
        let samples = SpaceProbe([nil])
        let cache = try QwenPrefixDiskStore(directory: directory(), limits: limits(entries: 1, minimumFree: 0),
                                           availableSpace: { try samples.sample($0) })
        defer { cache.close() }
        put(cache, [1]); put(cache, [2])
        XCTAssertNotNil(lookup(cache, [2]))
        XCTAssertNil(lookup(cache, [1]))
        XCTAssertEqual(cache.statistics.entries, 1)
        XCTAssertEqual(cache.statistics.evictions, 1)
        XCTAssertEqual(samples.count(), 0)
        XCTAssertEqual(cache.statistics.spaceChecks, 0)
        XCTAssertNil(cache.statistics.availableSpaceBytes)
    }

    func testTemporaryReplacementNeedsFullSpaceAndExactThresholdPasses() throws {
        let dir = try directory(), samples = SpaceProbe([UInt64.max]), box = CompletionBox()
        let minimum = 1024
        let cache = try QwenPrefixDiskStore(directory: dir, limits: limits(minimumFree: minimum),
                                           now: { 1000 }, availableSpace: { try samples.sample($0) })
        defer { cache.close() }
        put(cache, [1], byte: 7)
        var fs = statvfs()
        let directoryFD = open(dir.path, O_RDONLY | O_DIRECTORY | O_CLOEXEC)
        XCTAssertGreaterThanOrEqual(directoryFD, 0)
        defer { if directoryFD >= 0 { Darwin.close(directoryFD) } }
        XCTAssertEqual(fstatvfs(directoryFD, &fs), 0)
        let fragment = max(512, Int(fs.f_frsize))
        let file = try XCTUnwrap(files(dir).first)
        let fileBytes = try Data(contentsOf: file).count
        let reserved = (fileBytes + fragment - 1) / fragment * fragment
        samples.set([UInt64(minimum + reserved - 1)])
        XCTAssertTrue(cache.enqueue(tokens: [1], namespace: "model/runtime/tenant-a",
                                    metadata: Data("manifest".utf8), payload: Data(repeating: 8, count: 200),
                                    completion: { box.append($0) }))
        cache.flush()
        XCTAssertEqual(box.read(), [false])
        XCTAssertEqual(lookup(cache, [1])?.payload.first, 7, "old replacement bytes must not be pre-credited")
        // The second sample is after the temporary file exists. Only the floor
        // remains due then, not the archive allocation a second time.
        samples.set([UInt64(minimum + reserved), UInt64(minimum)])
        XCTAssertTrue(cache.enqueue(tokens: [1], namespace: "model/runtime/tenant-a",
                                    metadata: Data("manifest".utf8), payload: Data(repeating: 8, count: 200),
                                    completion: { box.append($0) }))
        cache.flush()
        XCTAssertEqual(box.read(), [false, true])
        XCTAssertEqual(lookup(cache, [1])?.payload.first, 8)
        XCTAssertEqual(try files(dir).count, 1)
        XCTAssertEqual(cache.statistics.spaceRecoveries, 1)
    }

    func testExternalSpaceDropBeforePublishRemovesTemporaryAndPreservesOldValue() throws {
        let dir = try directory(), samples = SpaceProbe([UInt64.max]), box = CompletionBox()
        let cache = try QwenPrefixDiskStore(directory: dir, limits: limits(minimumFree: 1024),
                                           availableSpace: { try samples.sample($0) })
        defer { cache.close() }
        put(cache, [1], byte: 7)
        let previous = cache.statistics
        samples.set([UInt64.max, 1023])
        XCTAssertTrue(cache.enqueue(tokens: [1], namespace: "model/runtime/tenant-a", metadata: Data(),
                                    payload: Data([8]), completion: { box.append($0) }))
        cache.flush()
        XCTAssertEqual(box.read(), [false])
        XCTAssertEqual(lookup(cache, [1])?.payload.first, 7)
        XCTAssertEqual(cache.statistics.diskBytes, previous.diskBytes)
        XCTAssertEqual(cache.statistics.published, previous.published)
        XCTAssertEqual(cache.statistics.spaceRejections, 1)
        XCTAssertEqual(cache.statistics.pendingBytes, 0)
        let names = try FileManager.default.contentsOfDirectory(atPath: dir.path)
        XCTAssertFalse(names.contains { $0.hasSuffix(".tmp") })
    }

    func testQueuedWritesResampleSpaceAndCompleteExactlyOnce() throws {
        let samples = SpaceProbe([UInt64.max, UInt64.max, 0]), box = CompletionBox()
        let cache = try QwenPrefixDiskStore(directory: directory(), limits: limits(minimumFree: 1),
                                           availableSpace: { try samples.sample($0) })
        defer { cache.close() }
        for key: Int32 in 1...2 {
            XCTAssertTrue(cache.enqueue(tokens: [key], namespace: "a", metadata: Data(), payload: Data([1]),
                                        completion: { box.append($0) }))
        }
        cache.flush()
        XCTAssertEqual(box.read(), [true, false])
        XCTAssertEqual(samples.count(), 3)
        XCTAssertEqual(cache.statistics.entries, 1)
        XCTAssertEqual(cache.statistics.pendingJobs, 0)
        XCTAssertEqual(cache.statistics.pendingBytes, 0)
    }

    func testPostWriteSpaceQueryFailureDropsTemporaryAndClearKeepsLastCondition() throws {
        let dir = try directory(), samples = SpaceProbe([UInt64.max, nil]), box = CompletionBox()
        let cache = try QwenPrefixDiskStore(directory: dir, limits: limits(minimumFree: 1024),
                                           availableSpace: { try samples.sample($0) })
        defer { cache.close() }
        XCTAssertTrue(cache.enqueue(tokens: [1], namespace: "a", metadata: Data(), payload: Data([1]),
                                    completion: { box.append($0) }))
        cache.flush()
        XCTAssertEqual(box.read(), [false])
        XCTAssertEqual(cache.statistics.entries, 0)
        XCTAssertEqual(cache.statistics.spaceQueryFailures, 1)
        XCTAssertEqual(cache.statistics.spaceChecks, 2)
        XCTAssertTrue(cache.statistics.spaceConstrained)
        XCTAssertTrue(try files(dir).isEmpty)
        XCTAssertFalse(try FileManager.default.contentsOfDirectory(atPath: dir.path).contains {
            $0.hasSuffix(".tmp")
        })
        cache.clear(resetStatistics: true)
        XCTAssertEqual(cache.statistics.spaceQueryFailures, 0)
        XCTAssertTrue(cache.statistics.spaceConstrained, "clear is not a new volume-space sample")
        samples.set([UInt64.max])
        put(cache, [2])
        XCTAssertNotNil(lookup(cache, [2]))
        XCTAssertEqual(cache.statistics.spaceRecoveries, 1)
    }

    func testSpaceComparisonDoesNotOverflowAtLargestConfiguredFloor() throws {
        let samples = SpaceProbe([UInt64.max])
        let cache = try QwenPrefixDiskStore(directory: directory(), limits: limits(minimumFree: Int.max),
                                           availableSpace: { try samples.sample($0) })
        defer { cache.close() }
        put(cache, [1])
        XCTAssertNotNil(lookup(cache, [1]))
        XCTAssertEqual(cache.statistics.availableSpaceBytes, UInt64.max)
        samples.set([UInt64(Int.max)])
        put(cache, [2])
        XCTAssertNil(lookup(cache, [2]))
        XCTAssertEqual(cache.statistics.spaceRejections, 1)
    }

    func testClearDuringLateSpaceSampleNeverRepublishesAndReleasesAdmission() throws {
        let samples = SpaceProbe([UInt64.max]), box = CompletionBox()
        let cache = try QwenPrefixDiskStore(directory: directory(), limits: limits(minimumFree: 1),
                                           availableSpace: { try samples.sample($0) })
        defer { cache.close() }
        put(cache, [1])
        samples.blockNext()
        XCTAssertTrue(cache.enqueue(tokens: [2], namespace: "a", metadata: Data(), payload: Data([2]),
                                    completion: { box.append($0) }))
        XCTAssertEqual(samples.entered.wait(timeout: .now() + 5), .success)
        defer { samples.resume.signal() }
        XCTAssertEqual(cache.statistics.pendingJobs, 1)
        let cleared = DispatchSemaphore(value: 0)
        DispatchQueue.global().async { cache.clear(); cleared.signal() }
        let deadline = Date().addingTimeInterval(5)
        while cache.peek(tokens: [1], namespace: "model/runtime/tenant-a") != nil, Date() < deadline {
            Thread.sleep(forTimeInterval: 0.001)
        }
        XCTAssertNil(cache.peek(tokens: [1], namespace: "model/runtime/tenant-a"))
        XCTAssertEqual(cache.statistics.pendingJobs, 1, "late sampler still owns its pending admission")
        samples.resume.signal()
        XCTAssertEqual(cleared.wait(timeout: .now() + 5), .success)
        cache.flush()
        XCTAssertEqual(box.read(), [false])
        XCTAssertEqual(cache.statistics.entries, 0)
        XCTAssertEqual(cache.statistics.pendingJobs, 0)
        XCTAssertEqual(cache.statistics.pendingBytes, 0)
    }

    func testPublicationDirectorySyncDoesNotBlockMetadataOrWriteAdmission() throws {
        let gate = PublicationSyncProbe(), box = CompletionBox(), observations = CompletionBox()
        let cache = try QwenPrefixDiskStore(directory: directory(), limits: limits(),
                                           publicationDirectorySync: { try gate.sync($0) })
        defer { gate.resume.signal(); cache.close() }
        put(cache, [1], namespace: "a")
        gate.blockNext()
        XCTAssertTrue(cache.enqueue(tokens: [1], namespace: "a", metadata: Data(), payload: Data([8]),
                                    completion: { box.append($0) }))
        XCTAssertEqual(gate.entered.wait(timeout: .now() + 3), .success)
        let inspected = DispatchSemaphore(value: 0)
        DispatchQueue.global().async {
            let stats = cache.statistics
            observations.append(stats.pendingJobs == 1 && stats.published == 1)
            // The candidate has been renamed but is not public until sync;
            // the prior payload size must not be offered for the replacement.
            observations.append(cache.peek(tokens: [1], namespace: "a") == nil)
            observations.append(cache.enqueue(tokens: [2], namespace: "a", metadata: Data(), payload: Data([2])))
            inspected.signal()
        }
        XCTAssertEqual(inspected.wait(timeout: .now() + 3), .success,
                       "metadata and admission must finish while the directory sync is still blocked")
        XCTAssertEqual(observations.read(), [true, true, true])
        gate.resume.signal(); cache.flush()
        XCTAssertEqual(box.read(), [true])
        XCTAssertEqual(lookup(cache, [1], namespace: "a")?.payload, Data([8]))
        XCTAssertEqual(lookup(cache, [2], namespace: "a")?.payload, Data([2]))
        XCTAssertEqual(cache.statistics.published, 3)
        XCTAssertEqual(cache.statistics.writeFailures, 0)
        XCTAssertEqual(cache.statistics.pendingJobs, 0)
        XCTAssertEqual(cache.statistics.pendingBytes, 0)
    }

    func testClearInvalidatesSyncingPublicationAndAllowsNewEpochAdmission() throws {
        let dir = try directory(), gate = PublicationSyncProbe(), box = CompletionBox()
        let observations = CompletionBox()
        let cache = try QwenPrefixDiskStore(directory: dir, limits: limits(),
                                           publicationDirectorySync: { try gate.sync($0) })
        defer { gate.resume.signal(); cache.close() }
        put(cache, [1], namespace: "a"); put(cache, [2], namespace: "a")
        gate.blockNext()
        XCTAssertTrue(cache.enqueue(tokens: [3], namespace: "a", metadata: Data(), payload: Data([3]),
                                    completion: { box.append($0) }))
        XCTAssertEqual(gate.entered.wait(timeout: .now() + 3), .success)
        let cleared = DispatchSemaphore(value: 0), inspected = DispatchSemaphore(value: 0)
        DispatchQueue.global().async { cache.clear(); cleared.signal() }
        DispatchQueue.global().async {
            let deadline = Date().addingTimeInterval(3)
            while cache.peek(tokens: [1], namespace: "a") != nil, Date() < deadline {
                Thread.sleep(forTimeInterval: 0.001)
            }
            observations.append(cache.peek(tokens: [1], namespace: "a") == nil &&
                                cache.peek(tokens: [2], namespace: "a") == nil)
            observations.append(cache.statistics.pendingJobs == 1)
            observations.append(cache.enqueue(tokens: [4], namespace: "a", metadata: Data(),
                                               payload: Data([4]), completion: { box.append($0) }))
            inspected.signal()
        }
        XCTAssertEqual(inspected.wait(timeout: .now() + 4), .success)
        XCTAssertEqual(observations.read(), [true, true, true])
        XCTAssertEqual(cleared.wait(timeout: .now()), .timedOut,
                       "clear invalidates immediately but still owns its queued physical cleanup")
        gate.resume.signal()
        XCTAssertEqual(cleared.wait(timeout: .now() + 3), .success)
        cache.flush()
        XCTAssertEqual(box.read(), [false, true])
        XCTAssertEqual(cache.statistics.published, 3, "the cancelled candidate is not counted as published")
        XCTAssertEqual(cache.statistics.entries, 1)
        XCTAssertEqual(cache.statistics.pendingJobs, 0)
        XCTAssertEqual(cache.statistics.pendingBytes, 0)
        for key: Int32 in [1, 2, 3] { XCTAssertNil(lookup(cache, [key], namespace: "a")) }
        XCTAssertEqual(lookup(cache, [4], namespace: "a")?.payload, Data([4]))
        cache.close()
        let reopened = try store(dir)
        XCTAssertEqual(reopened.statistics.recoveredEntries, 1)
        XCTAssertNil(lookup(reopened, [3], namespace: "a"))
        XCTAssertEqual(lookup(reopened, [4], namespace: "a")?.payload, Data([4]))
    }

    func testImmediateCloseDuringDirectorySyncRemovesUnpublishedCandidate() throws {
        let dir = try directory(), gate = PublicationSyncProbe(), box = CompletionBox()
        let observations = CompletionBox()
        let cache = try QwenPrefixDiskStore(directory: dir, limits: limits(),
                                           publicationDirectorySync: { try gate.sync($0) })
        defer { gate.resume.signal(); cache.close() }
        put(cache, [1], namespace: "a")
        gate.blockNext()
        XCTAssertTrue(cache.enqueue(tokens: [2], namespace: "a", metadata: Data(), payload: Data([2]),
                                    completion: { box.append($0) }))
        XCTAssertEqual(gate.entered.wait(timeout: .now() + 3), .success)
        let closed = DispatchSemaphore(value: 0), inspected = DispatchSemaphore(value: 0)
        DispatchQueue.global().async { cache.close(drain: false); closed.signal() }
        DispatchQueue.global().async {
            let deadline = Date().addingTimeInterval(3)
            while cache.peek(tokens: [1], namespace: "a") != nil, Date() < deadline {
                Thread.sleep(forTimeInterval: 0.001)
            }
            observations.append(cache.peek(tokens: [1], namespace: "a") == nil)
            observations.append(cache.statistics.pendingJobs == 1)
            observations.append(!cache.enqueue(tokens: [3], namespace: "a", metadata: Data(), payload: Data([3])))
            inspected.signal()
        }
        XCTAssertEqual(inspected.wait(timeout: .now() + 4), .success)
        XCTAssertEqual(observations.read(), [true, true, true])
        XCTAssertEqual(closed.wait(timeout: .now()), .timedOut)
        gate.resume.signal()
        XCTAssertEqual(closed.wait(timeout: .now() + 3), .success)
        cache.flush()
        XCTAssertEqual(box.read(), [false])
        XCTAssertEqual(cache.statistics.published, 1)
        XCTAssertEqual(cache.statistics.pendingJobs, 0)
        XCTAssertEqual(cache.statistics.pendingBytes, 0)
        let reopened = try store(dir)
        XCTAssertEqual(reopened.statistics.recoveredEntries, 1)
        XCTAssertNotNil(lookup(reopened, [1], namespace: "a"))
        XCTAssertNil(lookup(reopened, [2], namespace: "a"))
    }

    func testPublicationDirectorySyncFailureCleansRenamedArchiveBeforeRestart() throws {
        let dir = try directory(), gate = PublicationSyncProbe(), box = CompletionBox()
        let cache = try QwenPrefixDiskStore(directory: dir, limits: limits(),
                                           publicationDirectorySync: { try gate.sync($0) })
        defer { gate.resume.signal(); cache.close() }
        gate.blockNext(fail: true)
        XCTAssertTrue(cache.enqueue(tokens: [1], namespace: "a", metadata: Data(), payload: Data([1]),
                                    completion: { box.append($0) }))
        XCTAssertEqual(gate.entered.wait(timeout: .now() + 3), .success)
        // The file really passed atomic rename, so this exercises commit
        // failure cleanup rather than a pre-write rejection.
        XCTAssertEqual(try files(dir).count, 1)
        gate.resume.signal(); cache.flush()
        XCTAssertEqual(box.read(), [false])
        XCTAssertEqual(cache.statistics.published, 0)
        XCTAssertEqual(cache.statistics.writeFailures, 1)
        XCTAssertEqual(cache.statistics.entries, 0)
        XCTAssertEqual(cache.statistics.diskBytes, 0)
        XCTAssertEqual(cache.statistics.pendingJobs, 0)
        XCTAssertEqual(cache.statistics.pendingBytes, 0)
        XCTAssertTrue(try files(dir).isEmpty)
        cache.close()
        let reopened = try store(dir)
        XCTAssertEqual(reopened.statistics.recoveredEntries, 0)
        XCTAssertNil(lookup(reopened, [1], namespace: "a"))
    }

    func testBoundedCloseCompletesAdmittedWriteAndCallbacksAndCanBePolledAgain() throws {
        let dir = try directory(), cache = try store(dir), box = CompletionBox()
        XCTAssertTrue(cache.enqueue(tokens: [1], namespace: "a", metadata: Data(), payload: Data([1]),
                                    completion: { box.append($0) }))
        let result = cache.close(drain: true, timeout: 3)
        XCTAssertTrue(result.completed)
        XCTAssertTrue(result.ioCompleted)
        XCTAssertTrue(result.callbacksCompleted)
        XCTAssertEqual(box.read(), [true])
        XCTAssertEqual(cache.statistics.pendingJobs, 0)
        XCTAssertEqual(cache.statistics.pendingBytes, 0)
        XCTAssertTrue(cache.close(drain: true, timeout: 0).completed)
        XCTAssertTrue(cache.close(drain: true, timeout: .greatestFiniteMagnitude).completed)
        XCTAssertTrue(cache.close(drain: false, timeout: 0).completed)
        XCTAssertFalse(cache.enqueue(tokens: [2], namespace: "a", metadata: Data(), payload: Data([2])))
        let reopened = try store(dir)
        XCTAssertEqual(lookup(reopened, [1], namespace: "a")?.payload, Data([1]))
    }

    func testBoundedCloseDuringIORetainsLeaseAndFDThenAllowsCancellationAndRetry() throws {
        let dir = try directory(), gate = PublicationSyncProbe(), box = CompletionBox()
        let budget = try QwenStateBudget(maxBytes: 128)
        let cache = try QwenPrefixDiskStore(directory: dir, limits: limits(),
                                           publicationDirectorySync: { try gate.sync($0) })
        defer { gate.resume.signal(); cache.close() }
        weak var weakLease: QwenStateBudget.Lease?
        gate.blockNext()
        do {
            let lease = try XCTUnwrap(budget.reserve(bytes: 64, kind: .workspace))
            weakLease = lease
            XCTAssertTrue(cache.enqueue(tokens: [1], namespace: "a", metadata: Data([1]),
                                        payload: Data(repeating: 2, count: 32), completion: { [lease] success in
                defer { withExtendedLifetime(lease) {} }
                box.append(success)
            }))
        }
        XCTAssertEqual(gate.entered.wait(timeout: .now() + 3), .success)
        let started = DispatchTime.now().uptimeNanoseconds
        let first = cache.close(drain: true, timeout: 0.02)
        XCTAssertLessThan(Double(DispatchTime.now().uptimeNanoseconds - started) / 1e9, 1)
        XCTAssertFalse(first.completed)
        XCTAssertFalse(first.ioCompleted)
        XCTAssertFalse(first.callbacksCompleted)
        XCTAssertEqual(cache.statistics.pendingJobs, 1)
        XCTAssertEqual(cache.statistics.pendingBytes, 33)
        XCTAssertEqual(budget.statistics.workspaceBytes, 64)
        XCTAssertNotNil(weakLease)
        XCTAssertTrue(box.read().isEmpty)
        XCTAssertFalse(cache.enqueue(tokens: [2], namespace: "a", metadata: Data(), payload: Data([2])))
        XCTAssertThrowsError(try QwenPrefixDiskStore(directory: dir, limits: limits())) {
            XCTAssertEqual($0 as? QwenPrefixDiskStore.StoreError, .directoryInUse)
        }
        // A later non-draining close can cancel an earlier draining close;
        // another draining waiter must not revive the invalidated epoch.
        XCTAssertFalse(cache.close(drain: false, timeout: 0).completed)
        XCTAssertFalse(cache.close(drain: true, timeout: 0).completed)
        XCTAssertEqual(cache.statistics.pendingJobs, 1)
        XCTAssertEqual(budget.statistics.workspaceBytes, 64)
        gate.resume.signal()
        XCTAssertTrue(cache.close(drain: true, timeout: 3).completed)
        XCTAssertEqual(box.read(), [false])
        XCTAssertEqual(cache.statistics.pendingJobs, 0)
        XCTAssertEqual(cache.statistics.pendingBytes, 0)
        XCTAssertEqual(budget.statistics.workspaceBytes, 0)
        XCTAssertNil(weakLease)
        let reopened = try store(dir)
        XCTAssertEqual(reopened.statistics.recoveredEntries, 0)
        XCTAssertNil(lookup(reopened, [1], namespace: "a"))
    }

    func testBoundedCloseWaitsForReadCallbackOwnerAfterDirectoryHasClosed() throws {
        let dir = try directory(), cache = try store(dir), pause = CallbackPause(), box = CompletionBox()
        let budget = try QwenStateBudget(maxBytes: 128)
        defer { pause.resume.signal(); cache.close() }
        put(cache, [1], namespace: "a", byte: 9)
        weak var weakLease: QwenStateBudget.Lease?
        do {
            let lease = try XCTUnwrap(budget.reserve(bytes: 64, kind: .workspace))
            weakLease = lease
            XCTAssertTrue(cache.lookupAsync(tokens: [1], namespace: "a", completion: { [lease] match in
                defer { withExtendedLifetime((lease, match)) {} }
                box.append(pause.hold() && match?.payload == Data(repeating: 9, count: 200))
            }))
        }
        XCTAssertEqual(pause.entered.wait(timeout: .now() + 3), .success)
        let first = cache.close(drain: true, timeout: 0.02)
        XCTAssertTrue(first.ioCompleted)
        XCTAssertFalse(first.callbacksCompleted)
        XCTAssertFalse(first.completed)
        XCTAssertEqual(cache.statistics.pendingJobs, 1)
        XCTAssertEqual(cache.statistics.pendingBytes, cache.limits.maxPendingBytes)
        XCTAssertEqual(budget.statistics.workspaceBytes, 64)
        XCTAssertNotNil(weakLease)
        // Invalid durations are immediate polls, never unbounded waits.
        let started = DispatchTime.now().uptimeNanoseconds
        for timeout in [0, -1, TimeInterval.nan, TimeInterval.infinity, -TimeInterval.infinity] {
            let retry = cache.close(drain: true, timeout: timeout)
            XCTAssertTrue(retry.ioCompleted)
            XCTAssertFalse(retry.callbacksCompleted)
        }
        XCTAssertLessThan(Double(DispatchTime.now().uptimeNanoseconds - started) / 1e9, 1)
        // Descriptor ownership may end before the independent callback's Data
        // ownership. A new store can acquire the directory without stealing it.
        let reopened = try store(dir)
        XCTAssertNotNil(lookup(reopened, [1], namespace: "a"))
        pause.resume.signal()
        XCTAssertTrue(cache.close(drain: true, timeout: 3).completed)
        XCTAssertEqual(box.read(), [true])
        XCTAssertEqual(cache.statistics.pendingJobs, 0)
        XCTAssertEqual(cache.statistics.pendingBytes, 0)
        XCTAssertEqual(budget.statistics.workspaceBytes, 0)
        XCTAssertNil(weakLease)
    }

    func testBoundedCloseSharesOneDeadlineAcrossIOAndCallbackWaits() throws {
        let gate = PublicationSyncProbe(), pause = CallbackPause()
        let cache = try QwenPrefixDiskStore(directory: directory(), limits: limits(),
                                           publicationDirectorySync: { try gate.sync($0) })
        defer { gate.resume.signal(); pause.resume.signal(); cache.close() }
        gate.blockNext()
        XCTAssertTrue(cache.enqueue(tokens: [1], namespace: "a", metadata: Data(), payload: Data([1]),
                                    completion: { _ in _ = pause.hold() }))
        XCTAssertEqual(gate.entered.wait(timeout: .now() + 3), .success)
        DispatchQueue.global().asyncAfter(deadline: .now() + 0.25) { gate.resume.signal() }
        let started = DispatchTime.now().uptimeNanoseconds
        let result = cache.close(drain: true, timeout: 0.4)
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - started) / 1e9
        XCTAssertTrue(result.ioCompleted)
        XCTAssertFalse(result.callbacksCompleted)
        XCTAssertFalse(result.completed)
        XCTAssertLessThan(elapsed, 0.6,
                          "callback waiting must consume the remaining deadline, not a fresh 0.4 seconds")
        XCTAssertEqual(pause.entered.wait(timeout: .now() + 1), .success)
        pause.resume.signal()
        XCTAssertTrue(cache.close(drain: true, timeout: 3).completed)
    }

    func testBoundedCloseKeepsLastStoreOwnerAliveUntilLateIOAndCallbacksFinish() throws {
        let dir = try directory(), gate = PublicationSyncProbe(), box = CompletionBox()
        var cache: QwenPrefixDiskStore? = try QwenPrefixDiskStore(directory: dir, limits: limits(),
                                                                 publicationDirectorySync: { try gate.sync($0) })
        weak var retained = cache
        defer { gate.resume.signal(); cache?.close() }
        gate.blockNext()
        XCTAssertTrue(cache!.enqueue(tokens: [1], namespace: "a", metadata: Data(), payload: Data([1]),
                                     completion: { box.append($0) }))
        XCTAssertEqual(gate.entered.wait(timeout: .now() + 3), .success)
        XCTAssertFalse(cache!.close(drain: true, timeout: 0).completed)
        cache = nil
        XCTAssertNotNil(retained, "the IO closure owns the store after the caller stops waiting")
        XCTAssertThrowsError(try QwenPrefixDiskStore(directory: dir, limits: limits())) {
            XCTAssertEqual($0 as? QwenPrefixDiskStore.StoreError, .directoryInUse)
        }
        gate.resume.signal()
        let deadline = Date().addingTimeInterval(3)
        while retained != nil, Date() < deadline { Thread.sleep(forTimeInterval: 0.001) }
        XCTAssertNil(retained)
        XCTAssertEqual(box.read(), [true])
        let reopened = try store(dir)
        XCTAssertEqual(reopened.statistics.recoveredEntries, 1)
        XCTAssertEqual(lookup(reopened, [1], namespace: "a")?.payload, Data([1]))
    }

}
