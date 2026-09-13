import XCTest
@testable import ANERunnerCore

final class QwenPagedPageAdmissionTests: XCTestCase {
    func testFullFutureRangeIncludesImportSuffixAndOneCOWSlot() throws {
        XCTAssertEqual(try QwenPagedPageAdmission.requiredDecodePages(
            promptTokens: 11057, maximumOutputTokens: 16, reusedPrefixTokens: 10816), 9)
        XCTAssertEqual(try QwenPagedPageAdmission.requiredDecodePages(
            promptTokens: 11057, maximumOutputTokens: 16, reusedPrefixTokens: 0), 347)
        XCTAssertEqual(try QwenPagedPageAdmission.requiredDecodePages(
            promptTokens: 31, maximumOutputTokens: 34, reusedPrefixTokens: 31), 3)
        XCTAssertEqual(try QwenPagedPageAdmission.requiredDecodePages(
            promptTokens: 32, maximumOutputTokens: 33, reusedPrefixTokens: 32), 2)
        XCTAssertEqual(try QwenPagedPageAdmission.requiredDecodePages(
            promptTokens: 33, maximumOutputTokens: 32, reusedPrefixTokens: 31), 3)
    }

    func testInvalidAndOverflowingTokenRangesReject() throws {
        for (prompt, output, prefix) in [(0, 1, 0), (1, 0, 0), (1, 2, -1), (2, 2, 3), (Int.max, 2, 0)] {
            XCTAssertThrowsError(try QwenPagedPageAdmission.requiredDecodePages(
                promptTokens: prompt, maximumOutputTokens: output, reusedPrefixTokens: prefix))
        }
        XCTAssertEqual(try QwenPagedPageAdmission.requiredDecodePages(
            promptTokens: Int.max, maximumOutputTokens: 1, reusedPrefixTokens: 0), Int.max / 32 + 2)
    }

    func testConstantClaimsProtectLaterDecodeFromOtherRequestsAndCache() throws {
        let ledger = try QwenPagedPageAdmission(maximumPagesPerLayer: 32)
        let a = try XCTUnwrap(ledger.reserveDecodePages(pagesPerLayer: 12, minimumFreePages: 24))
        // A has materialized eight pages; its complete original claim remains.
        XCTAssertEqual(ledger.statistics.claimedPages, 12)
        XCTAssertFalse(try ledger.canReserveOptionalPages(pagesPerLayer: 5, minimumFreePages: 16))
        XCTAssertNil(try ledger.reserveDecodePages(pagesPerLayer: 5, minimumFreePages: 16))
        let b = try XCTUnwrap(ledger.reserveDecodePages(pagesPerLayer: 4, minimumFreePages: 16))
        XCTAssertEqual(ledger.statistics.claimedPages, 16)
        XCTAssertEqual(ledger.statistics.activeClaims, 2)
        XCTAssertFalse(try ledger.canReserveOptionalPages(pagesPerLayer: 1, minimumFreePages: 16))
        b.release(); b.release()
        XCTAssertEqual(ledger.statistics.claimedPages, 12)
        a.release()
        XCTAssertEqual(ledger.statistics.claimedPages, 0)
        XCTAssertEqual(ledger.statistics.activeClaims, 0)
        XCTAssertEqual(ledger.statistics.deniedClaims, 1)
    }

    func testZeroAllocationForkAllowedWhileClaimsConservativelyExceedFree() throws {
        let ledger = try QwenPagedPageAdmission(maximumPagesPerLayer: 32)
        let held = try XCTUnwrap(ledger.reserveDecodePages(pagesPerLayer: 20, minimumFreePages: 20))
        XCTAssertTrue(try ledger.canReserveOptionalPages(pagesPerLayer: 0, minimumFreePages: 0))
        XCTAssertFalse(try ledger.canReserveOptionalPages(pagesPerLayer: 1, minimumFreePages: 0))
        XCTAssertNil(try ledger.reserveDecodePages(pagesPerLayer: 1, minimumFreePages: 0))
        withExtendedLifetime(held) {}
    }

    func testReleasedClaimCannotHideNativePagesStillPinnedByCacheOrGraph() throws {
        let ledger = try QwenPagedPageAdmission(maximumPagesPerLayer: 16)
        var held: QwenPagedPageAdmission.Lease? = try XCTUnwrap(
            ledger.reserveDecodePages(pagesPerLayer: 8, minimumFreePages: 12))
        weak var witness = held
        held = nil
        XCTAssertNil(witness)
        XCTAssertEqual(ledger.statistics.claimedPages, 0)
        // Four actual free pages excludes live cache/state/native graph pins.
        XCTAssertNil(try ledger.reserveDecodePages(pagesPerLayer: 5, minimumFreePages: 4))
        XCTAssertTrue(try ledger.canReserveOptionalPages(pagesPerLayer: 4, minimumFreePages: 4))
    }

    func testInvalidCountsAndFreeSnapshotRejectWithoutChangingClaims() throws {
        for capacity in [-1, 0, 4097, Int.max] {
            XCTAssertThrowsError(try QwenPagedPageAdmission(maximumPagesPerLayer: capacity))
        }
        let ledger = try QwenPagedPageAdmission(maximumPagesPerLayer: 4)
        for (pages, free) in [(0, 4), (-1, 4), (1, -1), (1, 5)] {
            XCTAssertThrowsError(try ledger.reserveDecodePages(pagesPerLayer: pages, minimumFreePages: free))
        }
        XCTAssertThrowsError(try ledger.canReserveOptionalPages(pagesPerLayer: -1, minimumFreePages: 4))
        XCTAssertNil(try ledger.reserveDecodePages(pagesPerLayer: Int.max, minimumFreePages: 4))
        XCTAssertFalse(try ledger.canReserveOptionalPages(pagesPerLayer: Int.max, minimumFreePages: 4))
        XCTAssertEqual(ledger.statistics.claimedPages, 0)
        XCTAssertEqual(ledger.statistics.activeClaims, 0)
    }
}
