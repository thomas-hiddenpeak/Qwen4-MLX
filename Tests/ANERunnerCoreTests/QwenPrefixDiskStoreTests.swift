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
                        tokens: Int = 1_000, pending: Int = 524_288) -> QwenPrefixDiskLimits {
        .init(maxEntries: entries, maxBytes: bytes, maxKeyTokens: tokens,
              maxPendingJobs: 2, maxPendingBytes: pending, maxMetadataBytes: 1_024)
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

}
