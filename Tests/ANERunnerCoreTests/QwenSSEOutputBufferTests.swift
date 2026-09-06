import Dispatch
import Foundation
import Synchronization
import XCTest
@testable import ANERunnerCore

/// No model, MLX, socket, sleeps or external fixtures. These exercise the actual
/// producer/writer boundary, including callbacks arriving after cancellation.
final class QwenSSEOutputBufferTests: XCTestCase {
    private typealias Buffer = QwenSSEOutputBuffer
    private func bytes(_ count: Int, _ value: UInt8 = 65) -> Data { Data(repeating: value, count: count) }
    private func buffer(maxBytes: Int = 20, maxEvents: Int = 4, reserve: Int = 4) throws -> Buffer {
        try Buffer(limits: .init(maxBytes: maxBytes, maxEvents: maxEvents, terminalReserveBytes: reserve),
                   overflowFrame: bytes(reserve, 69))
    }
    private func drain(_ buffer: Buffer) -> [Buffer.Send] {
        var frames: [Buffer.Send] = []
        while let frame = buffer.beginSend() {
            frames.append(frame)
            _ = buffer.acknowledgeSend(frame.id, succeeded: true)
        }
        return frames
    }

    func testConfigurationReservesARealTerminalSlot() throws {
        for limits in [Buffer.Limits(maxBytes: 4, terminalReserveBytes: 4),
                       .init(maxBytes: 0), .init(maxEvents: 1),
                       .init(terminalReserveBytes: 0), .init(terminalReserveBytes: -1)] {
            XCTAssertThrowsError(try Buffer(limits: limits, overflowFrame: bytes(1))) {
                XCTAssertEqual($0 as? Buffer.ConfigurationError, .invalidLimits)
            }
        }
        for frame in [Data(), bytes(5)] {
            XCTAssertThrowsError(try Buffer(limits: .init(maxBytes: 20, terminalReserveBytes: 4), overflowFrame: frame)) {
                XCTAssertEqual($0 as? Buffer.ConfigurationError, .invalidOverflowFrame)
            }
        }
        let huge = try buffer(maxBytes: Int.max, maxEvents: Int.max)
        XCTAssertEqual(huge.enqueue(bytes(1)).status, .accepted)
    }

    func testByteLimitIncludesInFlightAndReservedTerminal() throws {
        let b = try buffer()
        XCTAssertEqual(b.enqueue(bytes(6, 65)).actions, .init(scheduleSend: true, cancelProducer: false))
        let first = try XCTUnwrap(b.beginSend())
        XCTAssertNil(b.beginSend())
        XCTAssertEqual(b.enqueue(bytes(6, 66)).actions, .none)
        XCTAssertEqual(b.enqueue(bytes(4, 67)).status, .accepted)
        XCTAssertEqual(b.snapshot().bufferedBytes, 16)
        XCTAssertEqual(b.snapshot().bufferedEvents, 3)
        XCTAssertEqual(b.snapshot().inFlightBytes, 6)
        XCTAssertEqual(b.finish(.completed, frame: bytes(4, 68)).status, .accepted)
        XCTAssertEqual(b.snapshot().bufferedBytes, 20)
        XCTAssertEqual(b.snapshot().bufferedEvents, 4)
        XCTAssertEqual(b.enqueue(bytes(1)).status, .closed)
        XCTAssertEqual(b.acknowledgeSend(first.id, succeeded: true), .init(scheduleSend: true, cancelProducer: false))
        let rest = drain(b)
        XCTAssertEqual([first.data] + rest.map(\.data), [bytes(6, 65), bytes(6, 66), bytes(4, 67), bytes(4, 68)])
        XCTAssertEqual(rest.map(\.isTerminal), [false, false, true])
        XCTAssertTrue(b.snapshot().isDrained)
        XCTAssertEqual(b.snapshot().bufferedBytes, 0)
        XCTAssertEqual(b.snapshot().bufferedEvents, 0)
    }

    func testOverflowPreservesAcceptedPrefixAndCancelsOnlyOnce() throws {
        let b = try buffer(maxBytes: 16)
        _ = b.enqueue(bytes(6, 65))
        let first = try XCTUnwrap(b.beginSend())
        _ = b.enqueue(bytes(6, 66))
        let overflow = b.enqueue(bytes(1, 67))
        XCTAssertEqual(overflow.status, .overflow)
        XCTAssertEqual(overflow.actions, .init(scheduleSend: false, cancelProducer: true))
        XCTAssertEqual(b.snapshot().bufferedBytes, 16)
        XCTAssertEqual(b.snapshot().bufferedEvents, 3)
        XCTAssertEqual(b.snapshot().outcome, .slowConsumer)
        XCTAssertFalse(b.snapshot().producerFinished)
        XCTAssertEqual(b.enqueue(bytes(1)).status, .closed)
        XCTAssertEqual(b.enqueue(bytes(1)).actions, .none)
        XCTAssertEqual(b.finish(.cancelled, frame: bytes(4, 70)).status, .alreadyTerminal)
        XCTAssertTrue(b.snapshot().producerFinished)
        XCTAssertEqual(b.snapshot().outcome, .slowConsumer)
        _ = b.acknowledgeSend(first.id, succeeded: true)
        let rest = drain(b)
        XCTAssertEqual(rest.map(\.data), [bytes(6, 66), bytes(4, 69)])
        XCTAssertEqual(rest.filter(\.isTerminal).count, 1)
        XCTAssertEqual(b.disconnect(), .none)
    }

    func testEventLimitIsIndependentOfByteLimitAndOverflowCanWakeIdleWriter() throws {
        let b = try buffer(maxBytes: 100, maxEvents: 3)
        XCTAssertEqual(b.enqueue(bytes(1)).status, .accepted)
        XCTAssertEqual(b.enqueue(bytes(1)).status, .accepted)
        XCTAssertEqual(b.enqueue(bytes(1)).status, .overflow)
        XCTAssertEqual(b.snapshot().bufferedEvents, 3)
        XCTAssertEqual(b.snapshot().bufferedBytes, 6)
        XCTAssertEqual(drain(b).count, 3)
        let oversized = try buffer()
        let overflow = oversized.enqueue(bytes(17))
        XCTAssertEqual(overflow.status, .overflow)
        XCTAssertEqual(overflow.actions, .init(scheduleSend: true, cancelProducer: true))
        XCTAssertEqual(drain(oversized).map(\.isTerminal), [true])
    }

    func testTerminalSelectionIsExactlyOnceAndInvalidFrameAllowsBoundedFallback() throws {
        let b = try buffer()
        XCTAssertEqual(b.finish(.completed, frame: Data()).status, .invalidFrame)
        XCTAssertEqual(b.finish(.completed, frame: bytes(5)).status, .invalidFrame)
        XCTAssertNil(b.snapshot().outcome)
        XCTAssertFalse(b.snapshot().producerFinished)
        XCTAssertEqual(b.finish(.failed, frame: bytes(3)).actions, .init(scheduleSend: true, cancelProducer: false))
        XCTAssertEqual(b.finish(.completed, frame: bytes(4)).status, .alreadyTerminal)
        XCTAssertEqual(b.snapshot().outcome, .failed)
        XCTAssertEqual(drain(b).filter(\.isTerminal).count, 1)
        XCTAssertEqual(b.disconnect(), .none)
        XCTAssertEqual(b.snapshot().outcome, .failed)
        XCTAssertFalse(b.snapshot().cancellationRequested)
    }

    func testQueuedDisconnectDropsFramesAndKeepsLateCompletionFromReopening() throws {
        let b = try buffer()
        _ = b.enqueue(bytes(5)); _ = b.enqueue(bytes(5))
        XCTAssertEqual(b.disconnect(), .init(scheduleSend: false, cancelProducer: true))
        XCTAssertEqual(b.disconnect(), .none)
        XCTAssertTrue(b.snapshot().isDrained)
        XCTAssertEqual(b.snapshot().bufferedBytes, 0)
        XCTAssertNil(b.beginSend())
        XCTAssertEqual(b.finish(.completed, frame: bytes(4)).status, .alreadyTerminal)
        XCTAssertEqual(b.snapshot().outcome, .disconnected)
        XCTAssertTrue(b.snapshot().producerFinished)
        XCTAssertEqual(b.enqueue(bytes(1)).status, .closed)
    }

    func testDisconnectRetainsInFlightQuotaUntilLateCallback() throws {
        let b = try buffer()
        _ = b.enqueue(bytes(6)); _ = b.enqueue(bytes(5))
        let send = try XCTUnwrap(b.beginSend())
        XCTAssertTrue(b.disconnect().cancelProducer)
        XCTAssertEqual(b.snapshot().bufferedBytes, 6)
        XCTAssertEqual(b.snapshot().bufferedEvents, 1)
        XCTAssertEqual(b.snapshot().queuedEvents, 0)
        XCTAssertTrue(b.snapshot().hasInFlight)
        XCTAssertFalse(b.snapshot().isDrained)
        XCTAssertNil(b.beginSend())
        XCTAssertEqual(b.acknowledgeSend(send.id, succeeded: true), .none)
        XCTAssertEqual(b.snapshot().bufferedBytes, 0)
        XCTAssertTrue(b.snapshot().isDrained)
        XCTAssertEqual(b.acknowledgeSend(send.id, succeeded: false), .none)
    }

    func testStaleSendFailureCannotCancelCurrentSendAndCurrentFailureCleansUp() throws {
        let b = try buffer()
        _ = b.enqueue(bytes(5)); _ = b.enqueue(bytes(5))
        let a = try XCTUnwrap(b.beginSend())
        _ = b.acknowledgeSend(a.id, succeeded: true)
        let current = try XCTUnwrap(b.beginSend())
        XCTAssertNotEqual(a.id, current.id)
        XCTAssertEqual(b.acknowledgeSend(a.id, succeeded: false), .none)
        XCTAssertFalse(b.snapshot().transportClosed)
        XCTAssertEqual(b.snapshot().inFlightBytes, 5)
        _ = b.enqueue(bytes(1))
        XCTAssertEqual(b.acknowledgeSend(current.id, succeeded: false),
                       .init(scheduleSend: false, cancelProducer: true))
        XCTAssertEqual(b.snapshot().outcome, .disconnected)
        XCTAssertTrue(b.snapshot().transportClosed)
        XCTAssertTrue(b.snapshot().isDrained)
        XCTAssertEqual(b.snapshot().bufferedEvents, 0)
        XCTAssertEqual(b.disconnect(), .none)
    }

    func testConcurrentOffersHaveOneOverflowAndNeverOvercommit() throws {
        let b = try buffer(maxBytes: 100, maxEvents: 9)
        let results = Mutex([Buffer.EnqueueStatus]())
        let cancellations = Mutex(0)
        DispatchQueue.concurrentPerform(iterations: 64) { _ in
            let offered = b.enqueue(Data(repeating: 65, count: 1))
            results.withLock { $0.append(offered.status) }
            if offered.actions.cancelProducer { cancellations.withLock { $0 += 1 } }
            let snapshot = b.snapshot()
            XCTAssertLessThanOrEqual(snapshot.bufferedBytes, 100)
            XCTAssertLessThanOrEqual(snapshot.bufferedEvents, 9)
        }
        let observed = results.withLock { $0 }
        XCTAssertEqual(observed.filter { $0 == .accepted }.count, 8)
        XCTAssertEqual(observed.filter { $0 == .overflow }.count, 1)
        XCTAssertEqual(observed.filter { $0 == .closed }.count, 55)
        XCTAssertEqual(cancellations.withLock { $0 }, 1)
        XCTAssertEqual(b.snapshot().outcome, .slowConsumer)
        XCTAssertEqual(drain(b).filter(\.isTerminal).count, 1)
    }

    func testConcurrentTerminalsAndDisconnectDoNotDuplicateCancellationOrReopen() throws {
        for _ in 0..<20 {
            let b = try buffer(maxBytes: 100, maxEvents: 12)
            _ = b.enqueue(bytes(3))
            let active = try XCTUnwrap(b.beginSend())
            let cancellations = Mutex(0)
            let selected = Mutex(0)
            DispatchQueue.concurrentPerform(iterations: 48) { index in
                let actions: Buffer.Actions
                switch index % 4 {
                case 0:
                    let terminal = b.finish(.completed, frame: Data(repeating: 68, count: 4))
                    if terminal.status == .accepted { selected.withLock { $0 += 1 } }
                    actions = terminal.actions
                case 1: actions = b.enqueue(Data(repeating: 65, count: 1)).actions
                case 2: actions = b.disconnect()
                default: actions = b.acknowledgeSend(active.id, succeeded: false)
                }
                if actions.cancelProducer { cancellations.withLock { $0 += 1 } }
                let snapshot = b.snapshot()
                XCTAssertLessThanOrEqual(snapshot.bufferedBytes, 100)
                XCTAssertLessThanOrEqual(snapshot.bufferedEvents, 12)
                XCTAssertGreaterThanOrEqual(snapshot.bufferedBytes, 0)
                XCTAssertGreaterThanOrEqual(snapshot.bufferedEvents, 0)
            }
            XCTAssertLessThanOrEqual(selected.withLock { $0 }, 1)
            XCTAssertLessThanOrEqual(cancellations.withLock { $0 }, 1)
            XCTAssertTrue(b.snapshot().transportClosed)
            XCTAssertTrue(b.snapshot().producerFinished)
            XCTAssertTrue(b.snapshot().isDrained)
            XCTAssertEqual(b.snapshot().bufferedBytes, 0)
            XCTAssertEqual(b.snapshot().bufferedEvents, 0)
            let outcome = b.snapshot().outcome
            XCTAssertNotNil(outcome)
            XCTAssertEqual(b.finish(.failed, frame: bytes(4)).status, .alreadyTerminal)
            XCTAssertEqual(b.snapshot().outcome, outcome)
            XCTAssertEqual(b.enqueue(bytes(1)).status, .closed)
            XCTAssertNil(b.beginSend())
        }
    }
}
