import XCTest
@testable import ANERunnerCore

final class ExpertRouterTests: XCTestCase {
    // Goldens generated independently using PyTorch matmul, stable argsort,
    // softmax over every expert, gather, and normalization. The BF16 oracle
    // uses native bfloat16 tensors for each selected-probability addition.
    // No Swift projection, rounding helper, or softmax implementation generated
    // these constants. All matrix entries below are exactly representable.
    private let matrix: [Float] = [1, 0, -1, 0.5, -1, 2, -1, 2, 0.25, 0.25, 0.5, -0.5]
    private let tokens: [Float] = [0.5, -1, 2, -2, 0.25, 1]
    private let logits: [Float] = [-1.5, 5.25, -2, -1.375, -3, 0.75, 2.75, -0.875]
    private let expertOutputs: [Float] = [2, -3, 0.25, 1, 4, -2, -1, 5, 2, 6, -0.5, 3]

    func testFloat32MatchesIndependentTorchGolden() throws {
        let router = try ExpertRouter(weights: matrix, expertCount: 4, hiddenSize: 3, topK: 2)
        let routed = try router.route(tokens: tokens, tokenCount: 2)
        XCTAssertEqual(routed.expertIDs, [1, 3, 2, 1])
        XCTAssertEqual(routed.logits, logits)
        assertClose(routed.weights, [0.9986749291419983, 0.001325022429227829, 0.8807970881462097, 0.11920291930437088])
        let merged = try ExpertRouter.weightedMerge(expertOutputs: expertOutputs, routing: routed, outputSize: 3)
        assertClose(merged, [1.9986748695373535, -2.990724802017212, 0.2470186948776245, -0.16557957231998444, 4.34438419342041, 2.1192028522491455])
    }

    func testBFloat16BoundariesMatchIndependentTorchGolden() throws {
        let router = try ExpertRouter(weights: matrix, expertCount: 4, hiddenSize: 3, topK: 2, precision: .bfloat16Boundaries)
        let routed = try router.route(tokens: tokens, tokenCount: 2)
        XCTAssertEqual(routed.expertIDs, [1, 3, 2, 1])
        XCTAssertEqual(routed.logits, logits)
        XCTAssertEqual(routed.weights, [1, 0.0013275146484375, 0.87890625, 0.119140625])
        // Deliberately NOT forced to sum to exactly one after BF16 rounding.
        XCTAssertNotEqual(routed.weights[0] + routed.weights[1], 1)
        let merged = try ExpertRouter.weightedMerge(expertOutputs: expertOutputs, routing: routed, outputSize: 3)
        XCTAssertEqual(merged, [2.0013275146484375, -2.99468994140625, 0.247344970703125, -0.1640625, 4.3349609375, 2.115234375])
    }

    func testDynamicTopTenOf512PreservesTokenAndSlotOrder() throws {
        let weights = (0..<512).flatMap { expert -> [Float] in [Float(expert - 256), 1] }
        let router = try ExpertRouter(weights: weights, expertCount: 512, hiddenSize: 2, topK: 10)
        let routed = try router.route(tokens: [1, 0, -1, 0], tokenCount: 2)
        XCTAssertEqual(Array(routed.expertIDs[0..<10]), Array((502...511).reversed()))
        XCTAssertEqual(Array(routed.expertIDs[10..<20]), Array(0..<10))
        XCTAssertEqual(routed.weights.count, 20)
        XCTAssertEqual(routed.logits.count, 1024)
        XCTAssertTrue(routed.weights.allSatisfy { $0.isFinite && $0 >= 0 })
        XCTAssertEqual(routed.weights[0..<10].reduce(0, +), 1, accuracy: 1e-6)
        XCTAssertEqual(routed.weights[10..<20].reduce(0, +), 1, accuracy: 1e-6)
    }

    func testExactTiesChooseLowerExpertID() throws {
        let router = try ExpertRouter(weights: [2, 2, 2, 2, 1], expertCount: 5, hiddenSize: 1, topK: 3)
        let routed = try router.route(tokens: [1], tokenCount: 1)
        XCTAssertEqual(routed.expertIDs, [0, 1, 2])
        assertClose(routed.weights, [1.0 / 3, 1.0 / 3, 1.0 / 3])
        XCTAssertEqual(try router.route(tokens: [1], tokenCount: 1).expertIDs, routed.expertIDs)
    }

    func testExtremeLogitsAndTopKOneRemainFinite() throws {
        let router = try ExpertRouter(weights: [10_000, 9_999, -10_000], expertCount: 3, hiddenSize: 1, topK: 2)
        let routed = try router.route(tokens: [1], tokenCount: 1)
        XCTAssertEqual(routed.expertIDs, [0, 1])
        // Analytic two-way logistic values, independent of the implementation.
        assertClose(routed.weights, [0.7310585786300049, 0.2689414213699951])
        let one = try ExpertRouter(weights: [-10_000, 10_000], expertCount: 2, hiddenSize: 1, topK: 1)
        XCTAssertEqual(try one.route(tokens: [1], tokenCount: 1).weights, [1])
    }

    func testWeightedMergeUsesSelectedSlotsNotGlobalExpertIDs() throws {
        let router = try ExpertRouter(weights: [0, 0, 0, 4, 3], expertCount: 5, hiddenSize: 1, topK: 2)
        let routed = try router.route(tokens: [1, -1], tokenCount: 2)
        XCTAssertEqual(routed.expertIDs, [3, 4, 0, 1])
        let values: [Float] = [1, 7, 20, -2, -4, 10, 6, 2]
        let actual = try ExpertRouter.weightedMerge(expertOutputs: values, routing: routed, outputSize: 2)
        // Reference walks output dimensions first and accumulates in Double;
        // production uses a row-major transposed Float32 BLAS GEMV per token.
        var expected: [Float] = []
        for token in 0..<2 {
            for dimension in 0..<2 {
                let sum = (0..<2).reduce(0.0) { value, slot in
                    value + Double(routed.weights[token * 2 + slot]) * Double(values[(token * 2 + slot) * 2 + dimension])
                }
                expected.append(Float(sum))
            }
        }
        assertClose(actual, expected)
    }

    func testInvalidGeometryNonfiniteValuesAndOverflowAreRejected() throws {
        XCTAssertThrowsError(try ExpertRouter(weights: [], expertCount: 0, hiddenSize: 1, topK: 1))
        XCTAssertThrowsError(try ExpertRouter(weights: [1], expertCount: 1, hiddenSize: 1, topK: 2))
        XCTAssertThrowsError(try ExpertRouter(weights: [1], expertCount: 2, hiddenSize: 1, topK: 1))
        XCTAssertThrowsError(try ExpertRouter(weights: [.nan], expertCount: 1, hiddenSize: 1, topK: 1))
        XCTAssertThrowsError(try ExpertRouter(weights: [], expertCount: Int.max, hiddenSize: 2, topK: 1))
        let router = try ExpertRouter(weights: [1], expertCount: 1, hiddenSize: 1, topK: 1)
        XCTAssertThrowsError(try router.route(tokens: [], tokenCount: 0))
        XCTAssertThrowsError(try router.route(tokens: [], tokenCount: 1))
        XCTAssertThrowsError(try router.route(tokens: [.infinity], tokenCount: 1))
        let overflow = try ExpertRouter(weights: [.greatestFiniteMagnitude], expertCount: 1, hiddenSize: 1, topK: 1)
        XCTAssertThrowsError(try overflow.route(tokens: [2], tokenCount: 1))
        let routed = try router.route(tokens: [1], tokenCount: 1)
        XCTAssertThrowsError(try ExpertRouter.weightedMerge(expertOutputs: [], routing: routed, outputSize: 1))
        XCTAssertThrowsError(try ExpertRouter.weightedMerge(expertOutputs: [.nan], routing: routed, outputSize: 1))
    }

    private func assertClose(_ actual: [Float], _ expected: [Float], accuracy: Float = 1e-6, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(actual.count, expected.count, file: file, line: line)
        for (left, right) in zip(actual, expected) {
            XCTAssertEqual(left, right, accuracy: accuracy, file: file, line: line)
        }
    }
}
