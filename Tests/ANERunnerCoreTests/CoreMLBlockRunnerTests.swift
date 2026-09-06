import XCTest
@testable import ANERunnerCore

final class CoreMLBlockRunnerTests: XCTestCase {
    func testTensorValidationRejectsLossyIntegerAndOverflow() {
        XCTAssertThrowsError(try CoreMLTensor(shape: [1], dtype: .int32, values: [1.5]).validate(name: "x"))
        XCTAssertThrowsError(try CoreMLTensor(shape: [1], dtype: .int32, values: [Double(Int32.max) + 1]).validate(name: "x"))
        XCTAssertThrowsError(try CoreMLTensor(shape: [1], dtype: .float16, values: [100_000]).validate(name: "x"))
        XCTAssertNoThrow(try CoreMLTensor(shape: [2], dtype: .int32, values: [Double(Int32.min), Double(Int32.max)]).validate(name: "x"))
    }

    func testTensorValidationRejectsInconsistentShape() {
        XCTAssertThrowsError(try CoreMLTensor(shape: [2, 2], dtype: .float32, values: [1, 2]).validate(name: "x"))
        XCTAssertThrowsError(try CoreMLTensor(shape: [Int.max, 2], dtype: .float32, values: []).validate(name: "x"))
        XCTAssertThrowsError(try CoreMLTensor(shape: [0], dtype: .float32, values: []).validate(name: "x"))
    }

    func testLogicalOffsetsHandlePaddedAndSingletonDimensions() {
        let padded = LogicalOffsets(shape: [2, 3], strides: [8, 1])
        XCTAssertEqual((0..<6).map(padded.offset), [0, 1, 2, 8, 9, 10])
        let singleton = LogicalOffsets(shape: [1, 3, 1, 2], strides: [100, 2, 99, 1])
        XCTAssertTrue(singleton.contiguous)
        XCTAssertEqual((0..<6).map(singleton.offset), Array(0..<6))
    }
}
