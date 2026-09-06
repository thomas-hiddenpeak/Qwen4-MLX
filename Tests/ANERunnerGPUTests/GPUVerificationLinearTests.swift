import CMLX
import Foundation
import XCTest
@testable import ANERunnerGPU

/// Synthetic BF16 source matrices at real projection dimensions; no checkpoint
/// is opened. The oracle is the existing S1 MLX matmul, not a second copy of
/// this kernel's loop. GPU execution is intentionally left to the test runner.
final class GPUVerificationLinearTests: XCTestCase {
    private func randomTensor(_ shape: [Int], seed: UInt64) throws -> Tensor {
        var random = seed
        var words = [UInt16]()
        words.reserveCapacity(shape.reduce(1, *))
        for _ in 0..<shape.reduce(1, *) {
            random = random &* 6364136223846793005 &+ 1442695040888963407
            let sign = UInt16((random >> 63) & 1) << 15
            // Mixed signs/exponents stress cancellation and accumulator
            // rounding. All values are finite normal BF16 numbers.
            let exponent = UInt16(118 + ((random >> 48) % 15)) << 7
            let fraction = UInt16((random >> 32) & 127)
            words.append((sign | exponent | fraction).littleEndian)
        }
        return try MX.array(data: words.withUnsafeBytes { Data($0) }, shape: shape, dtype: MLX_BFLOAT16)
    }

    func testMatchesScalarMatmulAcrossTokensShapesAndSeeds() throws {
        let kernel = try GPUVerificationLinear()
        // S, K, N: BN8 with/without a K tail; BM4; BM8 with a K tail.
        // The largest source matrix below is 31.5 MB, not the vocabulary head.
        let cases = [(2, 10240, 4), (3, 2560, 48), (4, 10240, 320),
                     (5, 2560, 512), (2, 2560, 640), (3, 320, 10240),
                     (4, 6144, 2560)]
        for seed in [UInt64(1009), UInt64(2029)] {
            for (s, k, n) in cases {
                let label = "S=\(s), K=\(k), N=\(n), seed=\(seed)"
                let x = try randomTensor([1, s, k], seed: seed)
                let rows = try randomTensor([n, k], seed: seed &+ 71)
                let weight = try MX.transpose(rows, [1, 0])
                var references: [Tensor] = []
                for token in 0..<s {
                    let row = try MX.slice(x, starts: [0, token, 0], ends: [1, token + 1, k])
                    references.append(try MX.matmul(row, weight))
                }
                let expected = try MX.concat(references, axis: 1)
                let actual = try kernel.apply(x, weight: weight)
                try MX.eval([expected, actual])
                XCTAssertEqual(actual.shape, [1, s, n], label)
                XCTAssertEqual(actual.dtype, MLX_BFLOAT16, label)
                let gold = try expected.floats(), output = try actual.floats()
                XCTAssertTrue(gold.allSatisfy(\.isFinite), label)
                XCTAssertTrue(output.allSatisfy(\.isFinite), label)
                XCTAssertTrue(gold.contains { $0 != 0 }, "Nontrivial oracle: \(label)")
                let differences = zip(output, gold).enumerated().filter {
                    $0.element.0.bitPattern != $0.element.1.bitPattern
                }
                XCTAssertEqual(differences.count, 0,
                    "\(label), first mismatch \(differences.first?.offset.description ?? "none")")
            }
        }
    }

    func testDispatchMatchesPinnedScalarSourceIncludingVocabularyWithoutAllocatingIt() throws {
        let p = GPUVerificationLinear.parameters
        XCTAssertEqual(try p(10240, 4), .init(bm: 1, bn: 8, sm: 1, sn: 32, tm: 4, tn: 4))
        XCTAssertEqual(try p(10240, 320), .init(bm: 1, bn: 8, sm: 1, sn: 32, tm: 4, tn: 4))
        XCTAssertEqual(try p(2560, 48), .init(bm: 1, bn: 8, sm: 1, sn: 32, tm: 4, tn: 4))
        XCTAssertEqual(try p(2560, 512), .init(bm: 4, bn: 1, sm: 1, sn: 32, tm: 4, tn: 4))
        XCTAssertEqual(try p(2560, 6144), .init(bm: 8, bn: 1, sm: 1, sn: 32, tm: 4, tn: 4))
        XCTAssertEqual(try p(2560, 248320), .init(bm: 8, bn: 1, sm: 1, sn: 32, tm: 4, tn: 4))
        XCTAssertThrowsError(try p(2560, 1)) // Scalar N1 uses dot_product.
        XCTAssertThrowsError(try p(64, 512)) // Unported small-K dispatch.
        XCTAssertThrowsError(try p(128, 128)) // Not a supported model projection.
        XCTAssertThrowsError(try p(Int.max, 4))
    }

    func testRejectsUnsupportedInputContractsBeforeEvaluation() throws {
        let kernel = try GPUVerificationLinear()
        let rows = try MX.zeros([4, 10240], MLX_BFLOAT16)
        let weight = try MX.transpose(rows, [1, 0])
        for shape in [[1, 1, 10240], [1, 6, 10240], [2, 2, 10240], [2, 10240], [1, 2, 512]] {
            XCTAssertThrowsError(try kernel.apply(MX.zeros(shape, MLX_BFLOAT16), weight: weight))
        }
        XCTAssertThrowsError(try kernel.apply(MX.zeros([1, 2, 10240], MLX_FLOAT32), weight: weight))
        XCTAssertThrowsError(try kernel.apply(MX.zeros([1, 2, 10240], MLX_BFLOAT16), weight: rows))
        XCTAssertThrowsError(try kernel.apply(MX.zeros([1, 2, 10240], MLX_BFLOAT16),
                                              weight: MX.zeros([10240, 4], MLX_FLOAT32)))
        XCTAssertThrowsError(try kernel.apply(MX.zeros([1, 2, 2560], MLX_BFLOAT16),
                                              weight: MX.zeros([2560, 1], MLX_BFLOAT16)))
    }
}
