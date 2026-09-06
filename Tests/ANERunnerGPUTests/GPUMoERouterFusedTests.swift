import CMLX
import Foundation
import XCTest
@testable import ANERunnerGPU

final class GPUMoERouterFusedTests: XCTestCase {
    private struct Fixture: Decodable {
        struct Row: Decodable {
            let name: String
            let shape: [Int]
            let logits: [Float]
            let indices: [Int32]
            let weights: [Float]
        }
        let cases: [Row]
    }

    private func fixture() throws -> Fixture {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return try JSONDecoder().decode(Fixture.self, from: Data(contentsOf:
            root.appendingPathComponent("fixtures/gpu-router-reference.json")))
    }

    func testActualDecodeAndPrefillAreBitExactWithAuthorRouter() throws {
        let router = try GPUMoERouterFused()
        for row in try fixture().cases {
            let logits = try MX.array(row.logits, shape: row.shape, dtype: MLX_BFLOAT16)
            let output = try router.route(logits)
            XCTAssertEqual(output.indices.shape, Array(row.shape.dropLast()) + [10])
            XCTAssertEqual(output.weights.shape, output.indices.shape)
            XCTAssertEqual(output.indices.dtype, MLX_UINT32)
            XCTAssertEqual(output.weights.dtype, MLX_BFLOAT16)
            XCTAssertEqual(try output.indices.ints(), row.indices, row.name)
            XCTAssertEqual(try output.weights.floats().map(\.bitPattern), row.weights.map(\.bitPattern), row.name)
        }
    }

    func testEqualCountLayoutsRetainTheirFullShapesWhenCached() throws {
        let fixture = try fixture()
        let row = try XCTUnwrap(fixture.cases.first { $0.name == "prefill" })
        let router = try GPUMoERouterFused()
        for shape in [[1, 2, 512], [2, 1, 512], [1, 1, 2, 512], [1, 2, 512]] {
            let output = try router.route(MX.array(row.logits, shape: shape, dtype: MLX_BFLOAT16))
            XCTAssertEqual(output.indices.shape, Array(shape.dropLast()) + [10])
            XCTAssertEqual(output.weights.shape, output.indices.shape)
            XCTAssertEqual(try output.indices.ints(), row.indices)
            XCTAssertEqual(try output.weights.floats().map(\.bitPattern), row.weights.map(\.bitPattern))
        }
    }

    func testTiesChooseLowestExpertAcrossSIMDGroupsAndTail() throws {
        let router = try GPUMoERouterFused()
        let tiedHigh = [1, 33, 64, 257, 511]
        var second = [Float](repeating: -2, count: 512)
        for index in tiedHigh { second[index] = 2 }
        let values = [Float](repeating: 0, count: 512) + second
        let logits = try MX.array(values, shape: [2, 512], dtype: MLX_BFLOAT16)
        let output = try router.route(logits)
        let expectedIDs = (0..<10).map(Int32.init) + (tiedHigh + [0, 2, 3, 4, 5]).map(Int32.init)
        XCTAssertEqual(try output.indices.ints(), expectedIDs)

        // Reference probabilities use public MLX softmax; IDs are independently
        // specified above, so this checks weights without assuming argsort ties.
        let selected = try MX.takeAlong(MX.softmax(logits, axis: -1, precise: true),
            MX.array(expectedIDs, shape: [2, 10]), axis: -1)
        var denominator = try MX.slice(selected, starts: [0, 0], ends: [2, 1])
        for slot in 1..<10 {
            denominator = try MX.cast(MX.add(denominator,
                MX.slice(selected, starts: [0, slot], ends: [2, slot + 1])), MLX_BFLOAT16)
        }
        let expectedWeights = try MX.cast(MX.div(selected, denominator), MLX_BFLOAT16)
        XCTAssertEqual(try output.weights.floats().map(\.bitPattern), try expectedWeights.floats().map(\.bitPattern))
        XCTAssertEqual(Array(try output.weights.floats().prefix(10)), [Float](repeating: 0.10009765625, count: 10))
    }

    func testRankOneNonPowerOfTwoAndTopKOne() throws {
        let router = try GPUMoERouterFused(expertCount: 33, topK: 1)
        var values = [Float](repeating: -1, count: 33)
        values[32] = 1
        let output = try router.route(MX.array(values, shape: [33], dtype: MLX_BFLOAT16))
        XCTAssertEqual(output.indices.shape, [1])
        XCTAssertEqual(try output.indices.ints(), [32])
        XCTAssertEqual(try output.weights.floats(), [1])
    }

    func testRejectsUnsupportedGeometryAndDtype() throws {
        for (experts, topK) in [(0, 1), (2049, 10), (512, 0), (512, 33), (8, 9)] {
            XCTAssertThrowsError(try GPUMoERouterFused(expertCount: experts, topK: topK))
        }
        let router = try GPUMoERouterFused(expertCount: 8, topK: 2)
        XCTAssertThrowsError(try router.route(MX.array([Float](repeating: 0, count: 8), shape: [8])))
        XCTAssertThrowsError(try router.route(MX.array([Float](repeating: 0, count: 7), shape: [7], dtype: MLX_BFLOAT16)))
    }
}
