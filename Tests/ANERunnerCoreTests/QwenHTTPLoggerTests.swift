import Darwin
import Dispatch
import Foundation
import Synchronization
import XCTest
@testable import ANERunnerCore

/// These exercise the real fixed writer/queue with an owned blocked CPU sink.
/// The sink is released in defer even if an assertion fails. No GPU or server.
final class QwenHTTPLoggerTests: XCTestCase {
    private typealias Logger = QwenHTTPLogger

    private final class Sink: @unchecked Sendable {
        let entered = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        let records = Mutex<[Data]>([])
        let error: Int32?
        init(error: Int32? = nil) { self.error = error }
        func write(_ data: Data) -> Int32? {
            entered.signal()
            release.wait()
            records.withLock { $0.append(data) }
            return error
        }
    }

    private func awaitSnapshot(_ logger: Logger, _ matches: (Logger.Snapshot) -> Bool) -> Logger.Snapshot {
        let until = Date().addingTimeInterval(2)
        var state = logger.snapshot()
        while !matches(state) && Date() < until {
            Thread.sleep(forTimeInterval: 0.001)
            state = logger.snapshot()
        }
        XCTAssertTrue(matches(state))
        return state
    }

    func testInvalidLimitsAndOversizeRecordsDoNotReachWriter() throws {
        for limits in [Logger.Limits(maxBytes: 0), .init(maxEvents: 0), .init(maxEventBytes: 0),
                       .init(maxBytes: 2, maxEventBytes: 3)] {
            XCTAssertThrowsError(try Logger(limits: limits, sink: { _ in nil }))
        }
        let sink = Sink()
        let logger = try Logger(limits: .init(maxBytes: 8, maxEvents: 2, maxEventBytes: 4), sink: { sink.write($0) })
        defer { logger.stop(); sink.release.signal() }
        XCTAssertFalse(logger.enqueue(Data(repeating: 65, count: 5)))
        XCTAssertFalse(logger.enqueue(Data()))
        XCTAssertEqual(logger.snapshot().droppedEvents, 2)
        XCTAssertEqual(logger.snapshot().droppedBytes, 5)
        XCTAssertEqual(logger.snapshot().enqueuedEvents, 0)
        XCTAssertEqual(logger.snapshot().bufferedBytes, 0)
        logger.stop()
        _ = awaitSnapshot(logger) { $0.writerExited }
        XCTAssertTrue(sink.records.withLock { $0.isEmpty })
    }

    func testBlockedSinkIncludesInFlightQuotaAndDoesNotBlockProducersOrStop() throws {
        let sink = Sink()
        let logger = try Logger(limits: .init(maxBytes: 12, maxEvents: 3, maxEventBytes: 8), sink: { sink.write($0) })
        defer { logger.stop(); sink.release.signal() }
        XCTAssertTrue(logger.enqueue(Data(repeating: 65, count: 6)))
        guard sink.entered.wait(timeout: .now() + 2) == .success else { return XCTFail("writer did not enter sink") }
        XCTAssertTrue(logger.enqueue(Data(repeating: 66, count: 6)))
        let producerDone = DispatchSemaphore(value: 0)
        let producerAccepted = Mutex(0)
        DispatchQueue.global().async {
            for _ in 0..<1000 {
                if logger.enqueue(Data([67])) { producerAccepted.withLock { $0 += 1 } }
            }
            producerDone.signal()
        }
        XCTAssertEqual(producerDone.wait(timeout: .now() + 2), .success)
        XCTAssertEqual(producerAccepted.withLock { $0 }, 0)
        var state = logger.snapshot()
        XCTAssertEqual(state.bufferedBytes, 12)
        XCTAssertEqual(state.bufferedEvents, 2)
        XCTAssertEqual(state.inFlightBytes, 6)
        XCTAssertEqual(state.queuedEvents, 1)
        XCTAssertEqual(state.droppedEvents, 1000)
        XCTAssertEqual(state.writtenEvents, 0)
        let stopDone = DispatchSemaphore(value: 0)
        DispatchQueue.global().async { logger.stop(); stopDone.signal() }
        XCTAssertEqual(stopDone.wait(timeout: .now() + 2), .success)
        state = logger.snapshot()
        XCTAssertFalse(state.accepting)
        XCTAssertFalse(state.writerExited)
        XCTAssertEqual(state.bufferedBytes, 6)
        XCTAssertEqual(state.bufferedEvents, 1)
        XCTAssertEqual(state.queuedEvents, 0)
        XCTAssertEqual(state.droppedEvents, 1001)
        logger.stop() // Idempotent: queued discard is not counted twice.
        XCTAssertEqual(logger.snapshot().droppedEvents, 1001)
        XCTAssertFalse(logger.enqueue(Data([68])))
        sink.release.signal()
        state = awaitSnapshot(logger) { $0.writerExited }
        XCTAssertEqual(state.bufferedBytes, 0)
        XCTAssertEqual(state.writtenEvents, 1)
        XCTAssertEqual(sink.records.withLock { $0 }, [Data(repeating: 65, count: 6)])
    }

    func testEventLimitCountsBlockedRecordEvenWithByteSpace() throws {
        let sink = Sink()
        let logger = try Logger(limits: .init(maxBytes: 128, maxEvents: 2, maxEventBytes: 64), sink: { sink.write($0) })
        defer { logger.stop(); sink.release.signal() }
        XCTAssertTrue(logger.enqueue(Data([65])))
        guard sink.entered.wait(timeout: .now() + 2) == .success else { return XCTFail("writer did not enter sink") }
        XCTAssertTrue(logger.enqueue(Data([66])))
        XCTAssertFalse(logger.enqueue(Data([67])))
        XCTAssertEqual(logger.snapshot().bufferedBytes, 2)
        XCTAssertEqual(logger.snapshot().bufferedEvents, 2)
        XCTAssertEqual(logger.snapshot().droppedEvents, 1)
        logger.stop()
        sink.release.signal()
        _ = awaitSnapshot(logger) { $0.writerExited }
    }

    func testWriteFailureIsObservableDiscardsQueueAndDoesNotRetry() throws {
        let sink = Sink(error: EPIPE)
        let logger = try Logger(limits: .init(maxBytes: 16, maxEvents: 4, maxEventBytes: 8), sink: { sink.write($0) })
        defer { logger.stop(); sink.release.signal() }
        XCTAssertTrue(logger.enqueue(Data([65, 66])))
        guard sink.entered.wait(timeout: .now() + 2) == .success else { return XCTFail("writer did not enter sink") }
        XCTAssertTrue(logger.enqueue(Data([67, 68, 69])))
        sink.release.signal()
        var state = awaitSnapshot(logger) { $0.writerExited }
        XCTAssertEqual(state.writeFailures, 1)
        XCTAssertEqual(state.lastWriteErrno, EPIPE)
        XCTAssertFalse(state.accepting)
        XCTAssertEqual(state.bufferedBytes, 0)
        XCTAssertEqual(state.droppedEvents, 2)
        XCTAssertEqual(state.droppedBytes, 5)
        XCTAssertEqual(state.writtenEvents, 0)
        XCTAssertFalse(logger.enqueue(Data([70])))
        state = logger.snapshot()
        XCTAssertEqual(state.droppedEvents, 3)
        XCTAssertEqual(state.writeFailures, 1)
        XCTAssertEqual(sink.records.withLock { $0.count }, 1)
    }

    func testDefaultSinkRequiresExplicitIgnoredProcessPolicyWithoutChangingIt() throws {
        var policy = sigaction(), previous = sigaction()
        policy.__sigaction_u.__sa_handler = SIG_DFL
        sigemptyset(&policy.sa_mask)
        XCTAssertEqual(sigaction(SIGPIPE, &policy, &previous), 0)
        defer { _ = sigaction(SIGPIPE, &previous, nil) }
        XCTAssertThrowsError(try Logger()) { error in
            guard case Logger.SignalPolicyError.requiresIgnoredSIGPIPE = error else {
                return XCTFail("unexpected error: \(error)")
            }
        }
        var after = sigaction()
        XCTAssertEqual(sigaction(SIGPIPE, nil, &after), 0)
        XCTAssertEqual(unsafeBitCast(after.__sigaction_u.__sa_handler, to: UnsafeRawPointer?.self),
                       unsafeBitCast(SIG_DFL, to: UnsafeRawPointer?.self))
    }

    func testClosedOwnedPipeGetsEPIPEOnWriterWithoutChangingCallerSignalMask() throws {
        // Match the explicit serve-gpu process contract in this owned test
        // process, then restore only after the real descriptor writer exits.
        var policy = sigaction(), previous = sigaction()
        policy.__sigaction_u.__sa_handler = SIG_IGN
        sigemptyset(&policy.sa_mask)
        XCTAssertEqual(sigaction(SIGPIPE, &policy, &previous), 0)
        var writerExited = true // Remains true if construction throws before starting a writer.
        defer {
            // On a failed timeout, retain the safe policy until test-process
            // exit rather than restoring while a writer could still run.
            if writerExited { _ = sigaction(SIGPIPE, &previous, nil) }
        }
        var before = sigset_t()
        XCTAssertEqual(pthread_sigmask(SIG_BLOCK, nil, &before), 0)
        var descriptors: [Int32] = [-1, -1]
        XCTAssertEqual(pipe(&descriptors), 0)
        let readFD = descriptors[0], writeFD = descriptors[1]
        guard readFD >= 0 && writeFD >= 0 else { return XCTFail("owned pipe creation failed") }
        close(readFD)
        defer { if writerExited { close(writeFD) } }
        let originalFlags = fcntl(writeFD, F_GETFL)
        let originalNoSignal = fcntl(writeFD, F_GETNOSIGPIPE)
        let logger = try Logger(limits: .init(maxBytes: 8, maxEvents: 2, maxEventBytes: 4), fileDescriptor: writeFD)
        writerExited = false
        defer { logger.stop() }
        XCTAssertTrue(logger.enqueue(Data([65])))
        let state = awaitSnapshot(logger) { $0.writerExited }
        writerExited = state.writerExited
        XCTAssertTrue(state.writerExited)
        XCTAssertEqual(state.writeFailures, 1)
        XCTAssertEqual(state.lastWriteErrno, EPIPE)
        XCTAssertEqual(state.bufferedBytes, 0)
        XCTAssertEqual(fcntl(writeFD, F_GETFL), originalFlags)
        XCTAssertEqual(fcntl(writeFD, F_GETNOSIGPIPE), originalNoSignal)
        var after = sigset_t()
        XCTAssertEqual(pthread_sigmask(SIG_BLOCK, nil, &after), 0)
        XCTAssertEqual(sigismember(&before, SIGPIPE), sigismember(&after, SIGPIPE))
    }

    func testOneWriterPreservesAcceptedRecordOrderAndReleasesQuota() throws {
        let records = Mutex<[Data]>([])
        let threads = Mutex<Set<UInt64>>([])
        let logger = try Logger(limits: .init(maxBytes: 64, maxEvents: 16, maxEventBytes: 4), sink: { data in
            var tid: UInt64 = 0
            pthread_threadid_np(nil, &tid)
            threads.withLock { _ = $0.insert(tid) }
            records.withLock { $0.append(data) }
            return nil
        })
        defer { logger.stop() }
        let wanted = (0..<16).map { Data([UInt8($0)]) }
        for data in wanted { XCTAssertTrue(logger.enqueue(data)) }
        let state = awaitSnapshot(logger) { $0.writtenEvents == 16 }
        XCTAssertEqual(state.bufferedBytes, 0)
        XCTAssertEqual(state.bufferedEvents, 0)
        XCTAssertEqual(state.droppedEvents, 0)
        XCTAssertEqual(records.withLock { $0 }, wanted)
        XCTAssertEqual(threads.withLock { $0.count }, 1)
    }
}
