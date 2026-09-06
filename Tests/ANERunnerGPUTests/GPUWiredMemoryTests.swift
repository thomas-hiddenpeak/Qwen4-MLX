import XCTest
@testable import ANERunnerGPU

/// Arithmetic only: these tests do not initialize MLX or run GPU work.
final class GPUWiredMemoryTests: XCTestCase {
    private let mib = 1024 * 1024

    func testFitAddsSlackWhenThereIsRoom() throws {
        XCTAssertEqual(
            try GPUWiredMemory.fitTarget(activeBytes: 1024 * mib, maximumRecommendedWorkingSetBytes: 4096 * mib),
            1280 * mib
        )
    }

    func testFitReducesSlackToPreserveRecommendedReserve() throws {
        XCTAssertEqual(
            try GPUWiredMemory.fitTarget(activeBytes: 900 * mib, maximumRecommendedWorkingSetBytes: 1280 * mib),
            1024 * mib
        )
        XCTAssertEqual(
            try GPUWiredMemory.fitTarget(activeBytes: 1024 * mib, maximumRecommendedWorkingSetBytes: 1280 * mib),
            1024 * mib
        )
    }

    func testRejectsMissingLiveWeights() {
        for active in [-1, 0] {
            XCTAssertThrowsError(try GPUWiredMemory.fitTarget(activeBytes: active, maximumRecommendedWorkingSetBytes: 4096 * mib))
        }
    }

    func testRejectsMissingMaximumOrInsufficientReserve() {
        for maximum in [-1, 0, 256 * mib] {
            XCTAssertThrowsError(try GPUWiredMemory.fitTarget(activeBytes: 1, maximumRecommendedWorkingSetBytes: maximum))
        }
    }

    func testRejectsCapSmallerThanLiveWeights() {
        XCTAssertThrowsError(try GPUWiredMemory.fitTarget(activeBytes: 1024 * mib + 1, maximumRecommendedWorkingSetBytes: 1280 * mib))
        XCTAssertThrowsError(try GPUWiredMemory.fitTarget(activeBytes: Int.max, maximumRecommendedWorkingSetBytes: Int.max))
    }

    func testLargestValidInputDoesNotOverflow() throws {
        let active = Int.max - 256 * mib
        XCTAssertEqual(
            try GPUWiredMemory.fitTarget(activeBytes: active, maximumRecommendedWorkingSetBytes: Int.max),
            active
        )
    }
}
