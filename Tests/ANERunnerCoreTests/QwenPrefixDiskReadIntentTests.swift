import XCTest
import Foundation
import Darwin
@testable import ANERunnerCore

/// Metadata admission and existing IO lifetime only; no model or GPU work.
final class QwenPrefixDiskReadIntentTests: XCTestCase {
    private final class Box: @unchecked Sendable {
        private let lock = NSLock()
        private var values = [Bool]()
        func append(_ value: Bool) { lock.lock(); values.append(value); lock.unlock() }
        func read() -> [Bool] { lock.lock(); defer { lock.unlock() }; return values }
    }
    private final class Gate: @unchecked Sendable {
        let entered = DispatchSemaphore(value: 0), resume = DispatchSemaphore(value: 0)
        private let lock = NSLock()
        private var armed = false
        func arm() { lock.lock(); armed = true; lock.unlock() }
        func sample(_ fd: Int32) throws -> UInt64 {
            var info = stat()
            guard fstat(fd, &info) == 0, info.st_mode & S_IFMT == S_IFDIR else {
                throw QwenPrefixDiskStore.StoreError.unsafeDirectory
            }
            lock.lock(); let block = armed; armed = false; lock.unlock()
            if block {
                entered.signal()
                guard resume.wait(timeout: .now() + 10) == .success else {
                    throw QwenPrefixDiskStore.StoreError.io(operation: "intent test gate", code: ETIMEDOUT)
                }
            }
            return UInt64.max
        }
    }
    private func directory() throws -> URL {
        let resolved = try XCTUnwrap(Darwin.realpath(FileManager.default.temporaryDirectory.path, nil))
        defer { free(resolved) }
        let path = URL(fileURLWithPath: String(cString: resolved), isDirectory: true)
            .appendingPathComponent("qwen-read-intent-tests-" + UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: path, withIntermediateDirectories: false,
            attributes: [.posixPermissions: 0o700])
        addTeardownBlock { try? FileManager.default.removeItem(at: path) }
        return path
    }
    private func limits() -> QwenPrefixDiskLimits {
        .init(maxEntries: 8, maxBytes: 1_048_576, maxKeyTokens: 1000, maxPendingJobs: 2,
            maxPendingBytes: 4096, maxMetadataBytes: 1024, minAvailableBytes: 1)
    }
    private func acquire(_ store: QwenPrefixDiskStore) throws -> QwenPrefixDiskReadIntent {
        guard case .acquired(let intent) = store.acquireReadIntent() else {
            XCTFail("Expected one available metadata intent")
            throw NSError(domain: "QwenPrefixDiskReadIntentTests", code: 1)
        }
        return intent
    }
    private func seed(_ store: QwenPrefixDiskStore) {
        XCTAssertTrue(store.enqueue(tokens: [1], namespace: "a", metadata: Data([9]),
            payload: Data(repeating: 7, count: 64)))
        store.flush()
    }

    func testBusyWriteCanHoldOneIntentWithoutAReadChargeOrCallback() throws {
        let gate = Gate(), store = try QwenPrefixDiskStore(directory: directory(), limits: limits(), availableSpace: { try gate.sample($0) })
        defer { gate.resume.signal(); store.close() }
        seed(store)
        gate.arm()
        XCTAssertTrue(store.enqueue(tokens: [2], namespace: "dummy", metadata: Data(), payload: Data([1])))
        XCTAssertEqual(gate.entered.wait(timeout: .now() + 3), .success)
        let before = store.statistics
        let intent = try acquire(store), callbacks = Box()
        defer { intent.release() }
        XCTAssertEqual(intent.state, .busy)
        XCTAssertEqual(store.statistics.foregroundReadIntents, 1)
        XCTAssertEqual(store.statistics.pendingJobs, before.pendingJobs)
        XCTAssertEqual(store.statistics.pendingBytes, before.pendingBytes)
        XCTAssertEqual(store.statistics.pendingBytes, 1)
        guard case .busy = store.acquireReadIntent() else { return XCTFail("Second metadata intent must be bounded") }
        XCTAssertFalse(store.enqueue(tokens: [3], namespace: "dummy", metadata: Data(), payload: Data([2])))
        XCTAssertFalse(store.lookupAsync(tokens: [1], namespace: "a") { _ in callbacks.append(false) })
        XCTAssertEqual(store.lookupAsync(tokens: [1], namespace: "a", readIntent: intent) { _ in callbacks.append(false) }, .busy)
        XCTAssertTrue(callbacks.read().isEmpty)
        XCTAssertEqual(store.statistics.pendingBytes, 1)
        gate.resume.signal(); store.flush()
        XCTAssertEqual(intent.state, .ready)
        XCTAssertEqual(store.statistics.pendingBytes, 0)
        XCTAssertEqual(store.lookupAsync(tokens: [1], namespace: "a", readIntent: intent) { value in
            callbacks.append(value?.payload == Data(repeating: 7, count: 64))
        }, .accepted)
        store.flush()
        XCTAssertEqual(callbacks.read(), [true])
        XCTAssertEqual(store.statistics.foregroundReadIntents, 0)
        XCTAssertEqual(store.statistics.pendingBytes, 0)
        XCTAssertEqual(store.statistics.pendingJobs, 0)
    }

    func testAcceptedReadStillChargesUntilCallbackAndCloseCompletes() throws {
        let store = try QwenPrefixDiskStore(directory: directory(), limits: limits())
        let entered = DispatchSemaphore(value: 0), resume = DispatchSemaphore(value: 0), observed = Box()
        defer { resume.signal(); store.close() }
        seed(store)
        let intent = try acquire(store)
        XCTAssertEqual(intent.state, .ready)
        XCTAssertEqual(store.lookupAsync(tokens: [1], namespace: "a", readIntent: intent) { value in
            observed.append(value?.payload == Data(repeating: 7, count: 64))
            entered.signal(); observed.append(resume.wait(timeout: .now() + 10) == .success)
        }, .accepted)
        XCTAssertEqual(entered.wait(timeout: .now() + 3), .success)
        XCTAssertEqual(intent.state, .invalidated)
        intent.release() // Consumed metadata handle cannot cancel admitted IO.
        XCTAssertEqual(store.statistics.pendingBytes, store.limits.maxPendingBytes)
        XCTAssertEqual(store.statistics.pendingJobs, 1)
        let close = store.close(drain: true, timeout: 0.02)
        XCTAssertTrue(close.ioCompleted)
        XCTAssertFalse(close.callbacksCompleted)
        XCTAssertEqual(store.statistics.pendingBytes, store.limits.maxPendingBytes)
        resume.signal()
        XCTAssertTrue(store.close(drain: true, timeout: 3).completed)
        XCTAssertEqual(observed.read(), [true, true])
        XCTAssertEqual(store.statistics.pendingBytes, 0)
        XCTAssertEqual(store.statistics.pendingJobs, 0)
    }

    func testClearInvalidatesOldIntentWithoutReleasingReplacement() throws {
        let store = try QwenPrefixDiskStore(directory: directory(), limits: limits())
        defer { store.close() }
        seed(store)
        let old = try acquire(store)
        store.clear(resetStatistics: true)
        XCTAssertEqual(old.state, .invalidated)
        let replacement = try acquire(store), callbacks = Box()
        old.release()
        XCTAssertEqual(replacement.state, .ready)
        XCTAssertEqual(store.statistics.foregroundReadIntents, 1)
        XCTAssertEqual(store.lookupAsync(tokens: [1], namespace: "a", readIntent: old) { _ in callbacks.append(false) }, .invalidated)
        XCTAssertTrue(callbacks.read().isEmpty)
        XCTAssertEqual(store.statistics.pendingJobs, 0)
        replacement.release()
        XCTAssertEqual(store.statistics.foregroundReadIntents, 0)
        seed(store)
    }

    func testCloseRevokesMetadataOnlyIntentWithoutWaitingForItsOwner() throws {
        let store = try QwenPrefixDiskStore(directory: directory(), limits: limits())
        let intent = try acquire(store), callbacks = Box()
        XCTAssertTrue(store.close(drain: true, timeout: 3).completed)
        XCTAssertEqual(intent.state, .closed)
        guard case .closed = store.acquireReadIntent() else { return XCTFail("Closed is not a busy admission") }
        XCTAssertEqual(store.lookupAsync(tokens: [1], namespace: "a", readIntent: intent) { _ in callbacks.append(false) }, .closed)
        XCTAssertTrue(callbacks.read().isEmpty)
        XCTAssertEqual(store.statistics.foregroundReadIntents, 0)
        XCTAssertEqual(store.statistics.pendingJobs, 0)
        XCTAssertEqual(store.statistics.pendingBytes, 0)
    }

    func testDroppingUnsubmittedIntentReopensOptionalWriteAdmission() throws {
        let store = try QwenPrefixDiskStore(directory: directory(), limits: limits())
        defer { store.close() }
        var intent: QwenPrefixDiskReadIntent? = try acquire(store)
        weak var observed = intent
        XCTAssertFalse(store.enqueue(tokens: [1], namespace: "a", metadata: Data(), payload: Data([1])))
        XCTAssertEqual(store.statistics.optionalWritePriorityRejections, 1)
        intent = nil
        XCTAssertNil(observed)
        XCTAssertEqual(store.statistics.foregroundReadIntents, 0)
        seed(store)
    }

    func testIntentDoesNotKeepStoreDescriptorsOrProcessLockAlive() throws {
        let path = try directory()
        var store: QwenPrefixDiskStore? = try QwenPrefixDiskStore(directory: path, limits: limits())
        let intent = try acquire(try XCTUnwrap(store))
        weak var observed = store
        store = nil
        XCTAssertNil(observed)
        XCTAssertEqual(intent.state, .closed)
        let replacement = try QwenPrefixDiskStore(directory: path, limits: limits())
        defer { replacement.close() }
        XCTAssertEqual(replacement.statistics.foregroundReadIntents, 0)
    }

    func testIntentFromAnotherStoreCannotReleaseOrSubmitCurrentIntent() throws {
        let left = try QwenPrefixDiskStore(directory: directory(), limits: limits())
        let right = try QwenPrefixDiskStore(directory: directory(), limits: limits())
        defer { left.close(); right.close() }
        let foreign = try acquire(left), current = try acquire(right), callbacks = Box()
        XCTAssertEqual(right.lookupAsync(tokens: [1], namespace: "a", readIntent: foreign) { _ in callbacks.append(false) }, .invalidated)
        foreign.release()
        XCTAssertEqual(current.state, .ready)
        XCTAssertEqual(right.statistics.foregroundReadIntents, 1)
        XCTAssertTrue(callbacks.read().isEmpty)
        current.release()
    }

    func testStorageUnavailableIsDistinctFromBusyAndClearRecovers() throws {
        if geteuid() == 0 { throw XCTSkip("permission fault requires an unprivileged test process") }
        let path = try directory(), store = try QwenPrefixDiskStore(directory: path, limits: limits())
        defer { _ = chmod(path.path, mode_t(0o700)); store.close() }
        seed(store)
        let intent = try acquire(store), callbacks = Box()
        XCTAssertEqual(chmod(path.path, mode_t(0o500)), 0)
        store.invalidate(tokens: [1], namespace: "a")
        XCTAssertTrue(store.statistics.storageUnavailable)
        XCTAssertEqual(intent.state, .unavailable)
        guard case .unavailable = store.acquireReadIntent() else { return XCTFail("Unavailable is not a busy admission") }
        XCTAssertEqual(store.lookupAsync(tokens: [1], namespace: "a", readIntent: intent) { _ in callbacks.append(false) }, .unavailable)
        XCTAssertTrue(callbacks.read().isEmpty)
        intent.release()
        XCTAssertEqual(store.statistics.foregroundReadIntents, 0)
        XCTAssertEqual(chmod(path.path, mode_t(0o700)), 0)
        store.clear()
        XCTAssertFalse(store.statistics.storageUnavailable)
        let recovered = try acquire(store)
        XCTAssertEqual(recovered.state, .ready)
        recovered.release()
    }

    func testClearDuringBusyWriteRevokesIntentBeforePhysicalCleanup() throws {
        let gate = Gate(), store = try QwenPrefixDiskStore(directory: directory(), limits: limits(), availableSpace: { try gate.sample($0) })
        defer { gate.resume.signal(); store.close() }
        seed(store); gate.arm()
        XCTAssertTrue(store.enqueue(tokens: [2], namespace: "dummy", metadata: Data(), payload: Data([1])))
        XCTAssertEqual(gate.entered.wait(timeout: .now() + 3), .success)
        let old = try acquire(store), cleared = DispatchSemaphore(value: 0)
        DispatchQueue.global().async { store.clear(resetStatistics: true); cleared.signal() }
        let deadline = Date().addingTimeInterval(3)
        while old.state != .invalidated && Date() < deadline { Thread.sleep(forTimeInterval: 0.001) }
        XCTAssertEqual(old.state, .invalidated)
        XCTAssertEqual(cleared.wait(timeout: .now()), .timedOut)
        let replacement = try acquire(store)
        old.release()
        XCTAssertEqual(replacement.state, .busy)
        XCTAssertEqual(store.statistics.foregroundReadIntents, 1)
        XCTAssertEqual(store.statistics.pendingBytes, 1)
        gate.resume.signal()
        XCTAssertEqual(cleared.wait(timeout: .now() + 3), .success)
        store.flush()
        XCTAssertEqual(replacement.state, .ready)
        XCTAssertEqual(store.statistics.foregroundReadIntents, 1)
        replacement.release()
        XCTAssertEqual(store.statistics.pendingBytes, 0)
        XCTAssertEqual(store.statistics.pendingJobs, 0)
    }

    func testExistingReplacementCanChangeSummaryBeforeIntentBecomesReady() throws {
        let gate = Gate(), store = try QwenPrefixDiskStore(directory: directory(), limits: limits(), availableSpace: { try gate.sample($0) })
        defer { gate.resume.signal(); store.close() }
        seed(store); gate.arm()
        XCTAssertTrue(store.enqueue(tokens: [1], namespace: "a", metadata: Data([8, 9]),
            payload: Data(repeating: 3, count: 128)))
        XCTAssertEqual(gate.entered.wait(timeout: .now() + 3), .success)
        let stale = try XCTUnwrap(store.peek(tokens: [1], namespace: "a")), intent = try acquire(store)
        XCTAssertEqual(intent.state, .busy)
        XCTAssertEqual(stale.payloadBytes, 64)
        gate.resume.signal(); store.flush()
        XCTAssertEqual(intent.state, .ready)
        let fresh = try XCTUnwrap(store.peek(tokens: [1], namespace: "a")), callbacks = Box()
        XCTAssertEqual(fresh.payloadBytes, 128)
        XCTAssertEqual(fresh.metadataBytes, 2)
        XCTAssertNotEqual(stale.payloadBytes, fresh.payloadBytes)
        XCTAssertEqual(store.lookupAsync(tokens: [1], namespace: "a", readIntent: intent) { value in
            callbacks.append(value?.payload == Data(repeating: 3, count: 128))
        }, .accepted)
        store.flush()
        XCTAssertEqual(callbacks.read(), [true])
        XCTAssertEqual(store.statistics.foregroundReadIntents, 0)
        XCTAssertEqual(store.statistics.pendingBytes, 0)
    }
}
