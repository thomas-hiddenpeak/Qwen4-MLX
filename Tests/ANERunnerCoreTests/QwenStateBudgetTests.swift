import Dispatch
import Foundation
import XCTest
@testable import ANERunnerCore

final class QwenStateBudgetTests: XCTestCase {
    func testConfigurationRequiresPositiveCapacity() throws {
        for limit in [Int.min, -1, 0] {
            XCTAssertThrowsError(try QwenStateBudget(maxBytes: limit)) {
                XCTAssertEqual($0 as? QwenStateBudget.ConfigurationError, .invalidMaxBytes)
            }
        }
        XCTAssertEqual(try QwenStateBudget(maxBytes: Int.max).statistics.maxBytes, Int.max)
    }

    func testAllKindsShareOneCapacityAndFailedAdmissionIsAtomic() throws {
        let budget = try QwenStateBudget(maxBytes: 100)
        let request = try XCTUnwrap(budget.reserve(bytes: 60, kind: .request))
        let cache = try XCTUnwrap(budget.reserve(bytes: 30, kind: .cache))
        XCTAssertNil(budget.reserve(bytes: 11, kind: .workspace))
        let workspace = try XCTUnwrap(budget.reserve(bytes: 10, kind: .workspace))
        let full = budget.statistics
        XCTAssertEqual(full.requestBytes, 60)
        XCTAssertEqual(full.cacheBytes, 30)
        XCTAssertEqual(full.workspaceBytes, 10)
        XCTAssertEqual(full.totalBytes, 100)
        XCTAssertEqual(full.peakBytes, 100)
        XCTAssertEqual(full.currentLeases, 3)
        XCTAssertEqual(full.rejections, 1)
        XCTAssertEqual(request.kind, .request)
        XCTAssertEqual(cache.kind, .cache)
        XCTAssertEqual(workspace.kind, .workspace)

        cache.release()
        let second = try XCTUnwrap(budget.reserve(bytes: 30, kind: .request))
        XCTAssertEqual(budget.statistics.requestBytes, 90)
        XCTAssertEqual(budget.statistics.cacheBytes, 0)
        XCTAssertEqual(budget.statistics.totalBytes, 100)
        request.release(); workspace.release(); second.release()
        XCTAssertEqual(budget.statistics.currentLeases, 0)
        XCTAssertEqual(budget.statistics.totalBytes, 0)
        XCTAssertEqual(budget.statistics.peakBytes, 100)
    }

    func testResizeChargesOnlyDeltaAndFailedGrowthPreservesLease() throws {
        let budget = try QwenStateBudget(maxBytes: 100)
        let request = try XCTUnwrap(budget.reserve(bytes: 60, kind: .request))
        let cache = try XCTUnwrap(budget.reserve(bytes: 30, kind: .cache))
        XCTAssertTrue(request.resize(to: 70))
        XCTAssertEqual(request.bytes, 70)
        XCTAssertEqual(budget.statistics.totalBytes, 100)
        XCTAssertFalse(cache.resize(to: 31))
        XCTAssertEqual(cache.bytes, 30)
        XCTAssertEqual(budget.statistics.cacheBytes, 30)
        XCTAssertEqual(budget.statistics.totalBytes, 100)
        XCTAssertTrue(request.resize(to: 20))
        XCTAssertEqual(budget.statistics.totalBytes, 50)
        XCTAssertTrue(cache.resize(to: 80))
        XCTAssertTrue(cache.resize(to: 80))
        XCTAssertEqual(budget.statistics.currentLeases, 2)
        XCTAssertEqual(budget.statistics.peakBytes, 100)
        XCTAssertEqual(budget.statistics.rejections, 1)
        request.release(); cache.release()
        XCTAssertEqual(budget.statistics.totalBytes, 0)
    }

    func testInvalidSizesAndReleasedLeaseCannotAlterOrReviveReservation() throws {
        let budget = try QwenStateBudget(maxBytes: 10)
        for bytes in [Int.min, -1, 0, 11, Int.max] {
            XCTAssertNil(budget.reserve(bytes: bytes, kind: .request))
        }
        let lease = try XCTUnwrap(budget.reserve(bytes: 5, kind: .cache))
        for bytes in [Int.min, -1, 0, 11, Int.max] {
            XCTAssertFalse(lease.resize(to: bytes))
            XCTAssertEqual(lease.bytes, 5)
        }
        XCTAssertFalse(lease.isReleased)
        lease.release(); lease.release()
        XCTAssertTrue(lease.isReleased)
        XCTAssertEqual(lease.bytes, 0)
        XCTAssertFalse(lease.resize(to: 1))
        XCTAssertEqual(budget.statistics.totalBytes, 0)
        XCTAssertEqual(budget.statistics.currentLeases, 0)
        XCTAssertEqual(budget.statistics.rejections, 11)
    }

    func testIntMaxReservationAndResizeDoNotOverflow() throws {
        let budget = try QwenStateBudget(maxBytes: Int.max)
        let lease = try XCTUnwrap(budget.reserve(bytes: Int.max - 1, kind: .request))
        let tail = try XCTUnwrap(budget.reserve(bytes: 1, kind: .workspace))
        XCTAssertEqual(budget.statistics.totalBytes, Int.max)
        XCTAssertNil(budget.reserve(bytes: Int.max, kind: .cache))
        XCTAssertFalse(lease.resize(to: Int.max))
        XCTAssertTrue(lease.resize(to: 1))
        XCTAssertEqual(budget.statistics.totalBytes, 2)
        XCTAssertTrue(lease.resize(to: Int.max - 1))
        tail.release()
        XCTAssertTrue(lease.resize(to: Int.max))
        XCTAssertEqual(budget.statistics.totalBytes, Int.max)
        lease.release()
        XCTAssertEqual(budget.statistics.totalBytes, 0)
        XCTAssertEqual(budget.statistics.peakBytes, Int.max)
    }

    func testDeinitReturnsCapacityOnNormalAndThrowingPaths() throws {
        let budget = try QwenStateBudget(maxBytes: 10)
        var lease: QwenStateBudget.Lease? = try XCTUnwrap(budget.reserve(bytes: 4, kind: .request))
        weak var weakLease = lease
        XCTAssertEqual(budget.statistics.totalBytes, 4)
        lease = nil
        XCTAssertNil(weakLease)
        XCTAssertEqual(budget.statistics.totalBytes, 0)

        enum Expected: Error { case failure }
        func failedOperation() throws {
            let workspace = try XCTUnwrap(budget.reserve(bytes: 10, kind: .workspace))
            try withExtendedLifetime(workspace) { throw Expected.failure }
        }
        XCTAssertThrowsError(try failedOperation())
        XCTAssertEqual(budget.statistics.totalBytes, 0)
        XCTAssertEqual(budget.statistics.currentLeases, 0)
        XCTAssertEqual(budget.statistics.workspaceBytes, 0)
    }

    func testLeaseKeepsBudgetAliveWithoutLedgerRetainingLease() throws {
        var budget: QwenStateBudget? = try QwenStateBudget(maxBytes: 10)
        weak var weakBudget = budget
        var lease: QwenStateBudget.Lease? = try XCTUnwrap(budget?.reserve(bytes: 5, kind: .request))
        weak var weakLease = lease
        budget = nil
        XCTAssertNotNil(weakBudget)
        XCTAssertTrue(lease!.resize(to: 10))
        lease = nil
        XCTAssertNil(weakLease)
        XCTAssertNil(weakBudget)
    }

    func testStatisticsAreCodableAndRetainHistoricalPeak() throws {
        let budget = try QwenStateBudget(maxBytes: 10)
        let lease = try XCTUnwrap(budget.reserve(bytes: 7, kind: .cache))
        XCTAssertNil(budget.reserve(bytes: 4, kind: .workspace))
        lease.release()
        let stats = budget.statistics
        let roundTrip = try JSONDecoder().decode(QwenStateBudget.Statistics.self,
            from: JSONEncoder().encode(stats))
        XCTAssertEqual(roundTrip, stats)
        XCTAssertEqual(roundTrip.totalBytes, 0)
        XCTAssertEqual(roundTrip.peakBytes, 7)
        XCTAssertEqual(roundTrip.rejections, 1)
    }

    func testConcurrentAdmissionResizeAndReleasePreserveAccounting() throws {
        let budget = try QwenStateBudget(maxBytes: 64)
        // All assertions use a single atomic snapshot, not several independent
        // reads which can legitimately observe different concurrent instants.
        DispatchQueue.concurrentPerform(iterations: 2_000) { iteration in
            let kind = QwenStateBudget.Kind.allCases[iteration % 3]
            if let lease = budget.reserve(bytes: 1 + iteration % 7, kind: kind) {
                _ = lease.resize(to: 1 + iteration % 11)
                let stats = budget.statistics
                XCTAssertGreaterThanOrEqual(stats.totalBytes, 0)
                XCTAssertLessThanOrEqual(stats.totalBytes, stats.maxBytes)
                XCTAssertEqual(stats.requestBytes + stats.cacheBytes + stats.workspaceBytes, stats.totalBytes)
                XCTAssertGreaterThanOrEqual(stats.currentLeases, 1)
                lease.release(); lease.release()
            }
        }
        let stats = budget.statistics
        XCTAssertEqual(stats.totalBytes, 0)
        XCTAssertEqual(stats.currentLeases, 0)
        XCTAssertEqual(stats.requestBytes + stats.cacheBytes + stats.workspaceBytes, 0)
        XCTAssertLessThanOrEqual(stats.peakBytes, stats.maxBytes)
    }

    func testConcurrentReleaseAndResizeOfTheSameLeaseCannotResurrectIt() throws {
        let budget = try QwenStateBudget(maxBytes: 64)
        let lease = try XCTUnwrap(budget.reserve(bytes: 8, kind: .workspace))
        DispatchQueue.concurrentPerform(iterations: 1_000) { iteration in
            if iteration % 5 == 0 { lease.release() }
            else { _ = lease.resize(to: 1 + iteration % 32) }
        }
        XCTAssertTrue(lease.isReleased)
        XCTAssertEqual(lease.bytes, 0)
        XCTAssertEqual(budget.statistics.totalBytes, 0)
        XCTAssertEqual(budget.statistics.currentLeases, 0)
        XCTAssertLessThanOrEqual(budget.statistics.peakBytes, 64)
        XCTAssertFalse(lease.resize(to: 1))
    }
}
