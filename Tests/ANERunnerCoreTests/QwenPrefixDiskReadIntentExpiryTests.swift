import XCTest
import Foundation
import Darwin
@testable import ANERunnerCore

/// No model, GPU work, or sleeps. Logical uptime advances explicitly; the
/// callback gate only proves that expiry cannot release already accepted IO.
final class QwenPrefixDiskReadIntentExpiryTests: XCTestCase {
    private final class Clock: @unchecked Sendable {
        private let lock = NSLock()
        private var value: UInt64 = 1_000
        func now() -> UInt64 { lock.lock(); defer { lock.unlock() }; return value }
        func set(_ next: UInt64) { lock.lock(); value = next; lock.unlock() }
    }
    private final class Flag: @unchecked Sendable {
        private let lock = NSLock()
        private var value = false
        func set(_ next: Bool) { lock.lock(); value = next; lock.unlock() }
        func get() -> Bool { lock.lock(); defer { lock.unlock() }; return value }
    }
    private func makeStore(_ clock: Clock) throws -> QwenPrefixDiskStore {
        let resolved = try XCTUnwrap(Darwin.realpath(FileManager.default.temporaryDirectory.path, nil))
        defer { free(resolved) }
        let path = URL(fileURLWithPath: String(cString: resolved), isDirectory: true)
            .appendingPathComponent("qwen-intent-expiry-" + UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: path, withIntermediateDirectories: false,
            attributes: [.posixPermissions: 0o700])
        addTeardownBlock { try? FileManager.default.removeItem(at: path) }
        return try QwenPrefixDiskStore(directory: path,
            limits: .init(maxEntries: 8, maxBytes: 1_048_576, maxKeyTokens: 1000,
                maxPendingJobs: 2, maxPendingBytes: 4096, maxMetadataBytes: 1024, minAvailableBytes: 0),
            availableSpace: { _ in UInt64.max },
            publicationDirectorySync: { fd in
                guard fsync(fd) == 0 else {
                    throw QwenPrefixDiskStore.StoreError.io(operation: "expiry test fsync", code: errno)
                }
            }, uptimeNanoseconds: { clock.now() })
    }
    private func acquire(_ store: QwenPrefixDiskStore, deadline: UInt64? = 2_000) throws -> QwenPrefixDiskReadIntent {
        guard case .acquired(let value) = store.acquireReadIntent(deadlineUptimeNanoseconds: deadline) else {
            XCTFail("Expected metadata intent")
            throw NSError(domain: "QwenPrefixDiskReadIntentExpiryTests", code: 1)
        }
        return value
    }
    private func write(_ store: QwenPrefixDiskStore) -> Bool {
        store.enqueue(tokens: [1], namespace: "a", metadata: Data([9]), payload: Data([7]))
    }

    func testEnqueueExpiresPriorityWhileCallerRetainsUnpolledHandle() throws {
        let clock = Clock(), store = try makeStore(clock)
        defer { store.close() }
        let retained = try acquire(store)
        clock.set(1_999)
        XCTAssertFalse(write(store))
        clock.set(2_000)
        // No state/statistics/discard call helps enqueue expire the owner.
        XCTAssertTrue(write(store))
        store.flush()
        XCTAssertEqual(retained.state, .invalidated)
        XCTAssertTrue(retained.hasExpired)
        XCTAssertEqual(store.statistics.published, 1)
        XCTAssertEqual(store.statistics.optionalWritePriorityRejections, 1)
    }

    func testStatisticsExpiryAndStaleReleaseCannotRemoveNewOwner() throws {
        let clock = Clock(), store = try makeStore(clock)
        defer { store.close() }
        let old = try acquire(store)
        clock.set(2_000)
        let expired = store.statistics
        XCTAssertEqual(expired.foregroundReadIntents, 0)
        XCTAssertEqual(expired.pendingJobs, 0)
        XCTAssertEqual(expired.pendingBytes, 0)
        let current = try acquire(store, deadline: 3_000)
        old.release()
        XCTAssertEqual(old.state, .invalidated)
        XCTAssertEqual(current.state, .ready)
        XCTAssertEqual(store.statistics.foregroundReadIntents, 1)
        current.release()
    }

    func testAcquireExpiresPreviousOwnerWithoutItsCooperation() throws {
        let clock = Clock(), store = try makeStore(clock)
        defer { store.close() }
        let old = try acquire(store)
        clock.set(2_000)
        let current = try acquire(store, deadline: 3_000)
        XCTAssertEqual(store.statistics.foregroundReadIntentAcquisitions, 2)
        old.release()
        XCTAssertEqual(current.state, .ready)
        current.release()
    }

    func testStateQueryExpiresMetadataAtExactDeadline() throws {
        let clock = Clock(), store = try makeStore(clock)
        defer { store.close() }
        let intent = try acquire(store)
        clock.set(1_999)
        XCTAssertEqual(intent.state, .ready)
        XCTAssertFalse(intent.hasExpired)
        clock.set(2_000)
        XCTAssertEqual(intent.state, .invalidated)
        XCTAssertTrue(intent.hasExpired)
        XCTAssertEqual(store.statistics.foregroundReadIntents, 0)
    }

    func testSubmissionAfterExpiryHasNoCallbackOrReadCharge() throws {
        let clock = Clock(), store = try makeStore(clock), called = Flag()
        defer { store.close() }
        XCTAssertTrue(write(store)); store.flush()
        let intent = try acquire(store)
        clock.set(2_000)
        XCTAssertEqual(store.lookupAsync(tokens: [1], namespace: "a", readIntent: intent) { _ in called.set(true) }, .invalidated)
        store.flush()
        XCTAssertFalse(called.get())
        XCTAssertTrue(intent.hasExpired)
        XCTAssertEqual(store.statistics.bytesRead, 0)
        XCTAssertEqual(store.statistics.pendingJobs, 0)
        XCTAssertEqual(store.statistics.pendingBytes, 0)
    }

    func testNilDeadlinePreservesManualReleaseAPI() throws {
        let clock = Clock(), store = try makeStore(clock)
        defer { store.close() }
        let intent = try acquire(store, deadline: nil)
        clock.set(UInt64.max)
        XCTAssertEqual(intent.state, .ready)
        XCTAssertFalse(intent.hasExpired)
        XCTAssertFalse(write(store))
        intent.release()
        XCTAssertTrue(write(store)); store.flush()
    }

    func testPastOrZeroDeadlineIsRejectedBeforeAcquiringPriority() throws {
        let clock = Clock(), store = try makeStore(clock)
        defer { store.close() }
        for deadline in [UInt64(0), 999, 1_000] {
            guard case .expired = store.acquireReadIntent(deadlineUptimeNanoseconds: deadline) else {
                return XCTFail("An elapsed deadline must not acquire priority")
            }
        }
        XCTAssertEqual(store.statistics.foregroundReadIntentAcquisitions, 0)
        XCTAssertEqual(store.statistics.foregroundReadIntents, 0)
        XCTAssertTrue(write(store)); store.flush()
    }

    func testClearAndCloseInvalidationDoNotMasqueradeAsTimeout() throws {
        let clock = Clock(), store = try makeStore(clock)
        let old = try acquire(store)
        store.clear()
        clock.set(2_000)
        XCTAssertEqual(old.state, .invalidated)
        XCTAssertFalse(old.hasExpired)
        let current = try acquire(store, deadline: 3_000)
        old.release()
        XCTAssertEqual(current.state, .ready)
        store.close()
        clock.set(3_000)
        XCTAssertEqual(current.state, .closed)
        XCTAssertFalse(current.hasExpired)
        XCTAssertFalse(old.hasExpired)
    }

    func testExpiryNeverReleasesAcceptedReadCallbackOwnership() throws {
        let clock = Clock(), store = try makeStore(clock), observed = Flag()
        let entered = DispatchSemaphore(value: 0), resume = DispatchSemaphore(value: 0)
        defer { resume.signal(); store.close() }
        XCTAssertTrue(write(store)); store.flush()
        let intent = try acquire(store)
        XCTAssertEqual(store.lookupAsync(tokens: [1], namespace: "a", readIntent: intent) { result in
            observed.set(result?.payload == Data([7]))
            entered.signal()
            _ = resume.wait(timeout: .now() + 5)
        }, .accepted)
        XCTAssertEqual(entered.wait(timeout: .now() + 3), .success)
        clock.set(2_000)
        XCTAssertEqual(store.statistics.foregroundReadIntents, 0)
        XCTAssertEqual(store.statistics.pendingJobs, 1)
        XCTAssertEqual(store.statistics.pendingBytes, store.limits.maxPendingBytes)
        intent.release()
        XCTAssertFalse(write(store))
        XCTAssertEqual(store.statistics.pendingBytes, store.limits.maxPendingBytes)
        resume.signal(); store.flush()
        XCTAssertTrue(observed.get())
        XCTAssertEqual(store.statistics.pendingBytes, 0)
        XCTAssertEqual(store.statistics.pendingJobs, 0)
    }
}
