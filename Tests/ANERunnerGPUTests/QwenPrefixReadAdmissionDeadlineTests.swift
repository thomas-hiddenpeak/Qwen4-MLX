@testable import ANERunnerCore
@testable import ANERunnerGPU
import XCTest

/// CPU-only deadline/ownership checks; no model or GPU stream is created.
final class QwenPrefixReadAdmissionDeadlineTests: XCTestCase {
    func testAdmissionAndAcceptedReadShareOneWaitStartWithoutRenewal() throws {
        let budget = try QwenStateBudget(maxBytes: 64)
        let attemptStarted: UInt64 = 20_000_000_000 // Earlier producer work is outside this clock.
        XCTAssertFalse(QwenPrefixDiskRead.waitHasTimedOut(startedAt: attemptStarted, after: 5, now: 23_000_000_000))
        let read = QwenPrefixDiskRead(lease: try XCTUnwrap(budget.reserve(bytes: 32, kind: .workspace)),
            startedAt: attemptStarted) // Submission after three seconds must retain the original start.
        XCTAssertFalse(read.hasTimedOut(after: 5, now: 24_999_999_999))
        XCTAssertTrue(read.hasTimedOut(after: 5, now: 25_000_000_000))
        read.lease.release()
        XCTAssertEqual(budget.statistics.workspaceBytes, 0)
    }

    func testBackwardClockCannotManufactureTimeoutAndReadyResultRemainsUsable() throws {
        let budget = try QwenStateBudget(maxBytes: 64)
        XCTAssertFalse(QwenPrefixDiskRead.waitHasTimedOut(startedAt: 20, after: 5, now: 10))
        let read = QwenPrefixDiskRead(lease: try XCTUnwrap(budget.reserve(bytes: 32, kind: .workspace)), startedAt: 20)
        read.complete(nil)
        XCTAssertFalse(read.hasTimedOut(after: 5, now: UInt64.max))
        XCTAssertEqual(budget.statistics.workspaceBytes, 32)
        read.lease.release()
        XCTAssertEqual(budget.statistics.currentLeases, 0)
    }
}
