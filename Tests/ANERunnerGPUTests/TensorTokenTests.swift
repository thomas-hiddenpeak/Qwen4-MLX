import CMLX
import Foundation
import XCTest
@testable import ANERunnerGPU

final class TensorTokenTests: XCTestCase {
    private func unsigned(_ values: [UInt32], shape: [Int]? = nil) throws -> Tensor {
        try MX.array(data: values.withUnsafeBytes { Data($0) },
            shape: shape ?? [values.count], dtype: MLX_UINT32)
    }

    func testReadsUnsignedTokenWithoutSignedGPUConversion() throws {
        for value in [UInt32(0), 248_044, UInt32(Int32.max)] {
            let tensor = try unsigned([value], shape: [1, 1])
            try MX.eval([tensor])
            XCTAssertEqual(try tensor.uint32TokenID(), Int32(value))
            XCTAssertEqual(tensor.dtype, MLX_UINT32)
        }
        XCTAssertEqual(try unsigned([42], shape: []).uint32TokenID(), 42)
    }

    func testReadsLazyAndAlreadyEvaluatedArgmax() throws {
        let logits = try MX.array([Float(-4), 3, 2], shape: [1, 1, 3])
        let lazy = try MX.argmax(logits)
        XCTAssertEqual(lazy.dtype, MLX_UINT32)
        XCTAssertEqual(try lazy.uint32TokenID(), 1)
        let evaluated = try MX.argmax(logits)
        try MX.eval([evaluated])
        XCTAssertEqual(try evaluated.uint32TokenID(), 1)
        XCTAssertEqual(try evaluated.uint32TokenID(), 1)
    }

    func testRejectsOverflowWrongDtypeAndNonScalar() throws {
        for value in [UInt32(Int32.max) + 1, UInt32.max] {
            XCTAssertThrowsError(try unsigned([value]).uint32TokenID())
        }
        XCTAssertThrowsError(try MX.array([Int32(7)], shape: [1]).uint32TokenID())
        XCTAssertThrowsError(try unsigned([0, 1]).uint32TokenID())
        XCTAssertThrowsError(try unsigned([]).uint32TokenID())
    }
}
