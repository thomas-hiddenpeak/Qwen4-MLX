import XCTest
@testable import ANERunnerCore

final class CoreAIQSAWorkingSetTests: XCTestCase {
    private func entry(_ count: Int, _ limit: Int) -> CoreAIQSAWorkingSet {
        .init(tokenCount: count, kvLimit: limit, function: "prefill_s\(count)_kv\(limit)")
    }

    func testPartialHistoryAndRestoredOffsetsSelectSafeView() throws {
        let entries = [entry(2048, 8192), entry(2048, 2048), entry(2048, 4096), entry(512, 4096)]
        try CoreAIQSAWorkingSet.validate(entries, counts: [2048, 512, 1], capacity: 16384)
        XCTAssertEqual(CoreAIQSAWorkingSet.select(entries, count: 2048, endOffset: 2048)?.kvLimit, 2048)
        XCTAssertEqual(CoreAIQSAWorkingSet.select(entries, count: 2048, endOffset: 2049)?.kvLimit, 4096)
        XCTAssertEqual(CoreAIQSAWorkingSet.select(entries, count: 2048, endOffset: 4096)?.kvLimit, 4096)
        XCTAssertEqual(CoreAIQSAWorkingSet.select(entries, count: 2048, endOffset: 6144)?.kvLimit, 8192)
        XCTAssertEqual(CoreAIQSAWorkingSet.select(entries, count: 512, endOffset: 3073)?.kvLimit, 4096)
        XCTAssertNil(CoreAIQSAWorkingSet.select(entries, count: 2048, endOffset: 8193))
        XCTAssertNil(CoreAIQSAWorkingSet.select(entries, count: 1, endOffset: 1024))
        XCTAssertNil(CoreAIQSAWorkingSet.select(entries, count: 256, endOffset: 4096))
    }

    func testRejectInvalidMetadataBeforeLoadingAssets() {
        let invalid: [[CoreAIQSAWorkingSet]] = [
            [entry(1, 2048)], [entry(2048, 1024)], [entry(2048, 2050)],
            [entry(2048, 16384)], [entry(4096, 8192)],
            [entry(2048, 4096), entry(2048, 4096)],
            [.init(tokenCount: 2048, kvLimit: 4096, function: "prefill")]
        ]
        for entries in invalid {
            XCTAssertThrowsError(try CoreAIQSAWorkingSet.validate(entries, counts: [2048, 1], capacity: 16384))
        }
    }
}
