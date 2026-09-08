@testable import ANERunnerCore
import Foundation
import XCTest
@testable import ANERunnerGPU

/// CPU-only ownership tests. No model, MLX tensor, GPU stream or disk read.
final class QwenPrefixDiskReadTests: XCTestCase {
    func testPendingReadDeadlineDoesNotUnderflowOrExpireReadyCompletion() throws {
        let budget = try QwenStateBudget(maxBytes: 64)
        let ticket = QwenPrefixDiskRead(lease: try XCTUnwrap(budget.reserve(bytes: 32, kind: .workspace)),
                                        startedAt: 1_000_000_000)
        XCTAssertFalse(ticket.hasTimedOut(after: 2, now: 0))
        XCTAssertFalse(ticket.hasTimedOut(after: 2, now: 2_999_999_999))
        XCTAssertTrue(ticket.hasTimedOut(after: 2, now: 3_000_000_000))
        ticket.complete(nil)
        XCTAssertTrue(ticket.isReady)
        XCTAssertFalse(ticket.hasTimedOut(after: 2, now: UInt64.max))
        XCTAssertEqual(budget.statistics.workspaceBytes, 32)
        ticket.lease.release()
        XCTAssertEqual(budget.statistics.workspaceBytes, 0)
    }

    func testDetachedRequestKeepsIOWorkspaceUntilLateCallbackIsReleased() throws {
        let budget = try QwenStateBudget(maxBytes: 64)
        var requestTicket: QwenPrefixDiskRead? = QwenPrefixDiskRead(
            lease: try XCTUnwrap(budget.reserve(bytes: 32, kind: .workspace)), startedAt: 0)
        weak var observedLease = requestTicket?.lease
        var callback: (@Sendable () -> Void)? = { [owned = requestTicket!] in owned.complete(nil) }
        XCTAssertTrue(requestTicket!.hasTimedOut(after: 1, now: 1_000_000_000))
        requestTicket = nil // Request timeout/cancellation detaches its reference.
        XCTAssertEqual(budget.statistics.workspaceBytes, 32)
        XCTAssertNotNil(observedLease)
        XCTAssertNil(budget.reserve(bytes: 33, kind: .workspace))
        callback?() // POSIX I/O finally finished; a cancelled request is not revived.
        callback = nil
        XCTAssertNil(observedLease)
        XCTAssertEqual(budget.statistics.workspaceBytes, 0)
        XCTAssertEqual(budget.statistics.currentLeases, 0)
    }

    func testTakingCompletedDataConsumesItOnceAndKeepsReservationUntilImportFinishes() throws {
        let budget = try QwenStateBudget(maxBytes: 64)
        var ticket: QwenPrefixDiskRead? = QwenPrefixDiskRead(
            lease: try XCTUnwrap(budget.reserve(bytes: 32, kind: .workspace)))
        var callback: (@Sendable () -> Void)? = { [owned = ticket!] in
            owned.complete(.init(prefixTokenCount: 416, metadata: Data([1]), payload: Data([2, 3]), diskBytes: 4096))
        }
        callback?()
        ticket!.complete(nil) // A duplicate completion cannot replace the first result.
        XCTAssertEqual(ticket!.take()?.payload, Data([2, 3]))
        XCTAssertNil(ticket!.take())
        XCTAssertEqual(budget.statistics.workspaceBytes, 32)
        ticket = nil // Import completed, but the callback still owns host lifetime.
        XCTAssertEqual(budget.statistics.workspaceBytes, 32)
        callback = nil
        XCTAssertEqual(budget.statistics.workspaceBytes, 0)
    }

    func testFinalReadyOwnerKeepsWorkspaceForConsumerAfterRequestDetaches() throws {
        let budget = try QwenStateBudget(maxBytes: 64)
        var request: QwenPrefixDiskRead? = QwenPrefixDiskRead(
            lease: try XCTUnwrap(budget.reserve(bytes: 32, kind: .workspace)))
        request!.complete(.init(prefixTokenCount: 416, metadata: Data([1]), payload: Data([2]), diskBytes: 4096))
        func consume() throws {
            let owner = try XCTUnwrap(request)
            defer { withExtendedLifetime(owner) {} }
            let archive = try XCTUnwrap(owner.take())
            request = nil // Callback and request no longer retain this result.
            XCTAssertEqual(archive.payload, Data([2]))
            XCTAssertEqual(budget.statistics.workspaceBytes, 32)
            XCTAssertNil(budget.reserve(bytes: 33, kind: .workspace))
        }
        try consume()
        XCTAssertEqual(budget.statistics.workspaceBytes, 0)
        XCTAssertEqual(budget.statistics.currentLeases, 0)
    }
}
