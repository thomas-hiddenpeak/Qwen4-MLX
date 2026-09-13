import ANERunnerCore
import Darwin
import XCTest
@testable import ANERunnerGPU

/// Host-only admission paths: no successful DSO load, MLX stream or model.
final class GPUPagedKVPoolBudgetTests: XCTestCase {
    func testFixedReservationIncludesAllocatorAndMetadataAllowance() throws {
        let page = Int(getpagesize())
        for pages in [1, 2, 384, 4096] {
            let oneArena = pages * 32_768
            let rounded = ((oneArena + page - 1) / page) * page
            let reserved = try GPUPagedKVPool.reservedBytes(maximumPages: pages)
            XCTAssertEqual(reserved, 2 * (rounded + 2 * page) + 65_536 + pages * 128)
            XCTAssertGreaterThan(reserved, pages * 65_536)
        }
        for pages in [Int.min, -1, 0, 4097, Int.max] {
            XCTAssertThrowsError(try GPUPagedKVPool.reservedBytes(maximumPages: pages))
        }
    }

    func testRejectedArenaDoesNotLoadNativeCodeOrAlterExistingLease() throws {
        let required = try GPUPagedKVPool.reservedBytes(maximumPages: 384)
        let budget = try QwenStateBudget(maxBytes: required)
        let request = try XCTUnwrap(budget.reserve(bytes: 1, kind: .request))
        XCTAssertThrowsError(try GPUPagedKVPool(libraryPath: "/missing/paged-budget-test.dylib",
            maximumPages: 384, stateBudget: budget)) { error in
            XCTAssertTrue(String(describing: error).contains("state budget denied"))
        }
        XCTAssertEqual(budget.statistics.totalBytes, 1)
        XCTAssertEqual(budget.statistics.workspaceBytes, 0)
        XCTAssertEqual(budget.statistics.currentLeases, 1)
        XCTAssertEqual(budget.statistics.rejections, 1)
        withExtendedLifetime(request) {}
    }

    func testLoaderFailureReturnsPreAdmittedArenaReservation() throws {
        let required = try GPUPagedKVPool.reservedBytes(maximumPages: 2)
        let budget = try QwenStateBudget(maxBytes: required)
        XCTAssertThrowsError(try GPUPagedKVPool(libraryPath: "/missing/paged-budget-test.dylib",
            maximumPages: 2, stateBudget: budget)) { error in
            XCTAssertTrue(String(describing: error).contains("Cannot load physical KV pool"))
        }
        XCTAssertEqual(budget.statistics.totalBytes, 0)
        XCTAssertEqual(budget.statistics.currentLeases, 0)
        XCTAssertEqual(budget.statistics.peakBytes, required)
        XCTAssertEqual(budget.statistics.rejections, 0)
    }

    func testInvalidConfigurationDoesNotReserveBytes() throws {
        let budget = try QwenStateBudget(maxBytes: Int.max)
        for pages in [0, 4097, Int.max] {
            XCTAssertThrowsError(try GPUPagedKVPool(libraryPath: "/missing/paged-budget-test.dylib",
                maximumPages: pages, stateBudget: budget))
        }
        XCTAssertThrowsError(try GPUPagedKVPool(libraryPath: "relative.dylib",
            maximumPages: 1, stateBudget: budget))
        XCTAssertEqual(budget.statistics.peakBytes, 0)
        XCTAssertEqual(budget.statistics.currentLeases, 0)
        XCTAssertEqual(budget.statistics.rejections, 0)
    }
}
