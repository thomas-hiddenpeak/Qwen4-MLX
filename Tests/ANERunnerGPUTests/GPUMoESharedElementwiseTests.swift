import CMLX
import Foundation
import XCTest
@testable import ANERunnerGPU

final class GPUMoESharedElementwiseTests: XCTestCase {
    private func tensor(_ bits: [UInt16], shape: [Int]) throws -> Tensor {
        try MX.array(data: bits.withUnsafeBytes { Data($0) }, shape: shape, dtype: MLX_BFLOAT16)
    }

    private func assertSame(_ actual: Tensor, _ expected: Tensor,
                            file: StaticString = #filePath, line: UInt = #line) throws {
        XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
        XCTAssertEqual(actual.dtype, MLX_BFLOAT16, file: file, line: line)
        let a = try actual.floats(), e = try expected.floats()
        XCTAssertEqual(a.count, e.count, file: file, line: line)
        for (i, pair) in zip(a, e).enumerated() {
            // IEEE NaN payloads are not a numerical equality requirement.
            if pair.1.isNaN { XCTAssertTrue(pair.0.isNaN, "element \(i)", file: file, line: line) }
            else { XCTAssertEqual(pair.0.bitPattern, pair.1.bitPattern,
                "element \(i): \(pair.0) vs \(pair.1)", file: file, line: line) }
        }
    }

    func testActivationMatchesUncompiledBF16ReferenceAtBoundaries() throws {
        let fused = try GPUMoEFused(hidden: 260, intermediate: 8, topK: 2, groupSize: 64, bits: 4)
        // Signed zeros, subnormal/normal transitions, adjacent BF16 values,
        // saturated sigmoid, finite limits, infinities and NaN.
        let gates: [UInt16] = [0, 0x8000, 1, 0x007f, 0x0080, 0x8080, 0x3c80,
            0x3f7f, 0x3f80, 0x3f81, 0xbf80, 0x4180, 0xc180, 0x7f7f, 0xff7f,
            0x7f80, 0xff80, 0x7fc1]
        let ups: [UInt16] = [0x3f81, 0xbf81, 0x3f7f, 0x3f80, 0, 0x0080, 0x4380]
        // Exercise a partial threadgroup, actual shared width, and cache reuse.
        for width in [19, 640, 19] {
            let gate = try tensor((0..<width).map { gates[$0 % gates.count] }, shape: [1, 1, width])
            let up = try tensor((0..<width).map { ups[($0 / gates.count + $0) % ups.count] }, shape: [1, 1, width])
            let reference = try MX.mul(MX.mul(gate, MX.sigmoid(gate)), up)
            try assertSame(fused.sharedActivation(gate, up: up), reference)
        }
    }

    func testOutputMatchesReferenceAndDiagnosticsDoNotChangeValues() throws {
        let hidden = 260 // Non-multiple of 256 also exercises output tail bounds.
        let fused = try GPUMoEFused(hidden: hidden, intermediate: 8, topK: 2, groupSize: 64, bits: 4)
        let values: [UInt16] = [0, 0x8000, 1, 0x007f, 0x0080, 0x3f81, 0xbf81,
            0x3f7f, 0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7fc1]
        let routed = try tensor((0..<hidden).map { values[$0 % values.count] }, shape: [1, 1, hidden])
        let down = try tensor((0..<hidden).map { values[($0 * 3 + 1) % values.count] }, shape: [1, 1, hidden])
        for gateBits in [UInt16(0), 0x8000, 0x3f80, 0xbf80, 0x4180, 0xc180, 0x7f80, 0xff80, 0x7fc1] {
            let logits = try tensor([gateBits], shape: [1, 1, 1])
            let gate = try MX.sigmoid(logits)
            let gated = try MX.mul(down, gate)
            let reference = try MX.add(routed, gated)
            let regular = try fused.sharedOutput(routed: routed, down: down, gateLogits: logits, diagnostics: false)
            let diagnostic = try fused.sharedOutput(routed: routed, down: down, gateLogits: logits, diagnostics: true)
            XCTAssertNil(regular.gate); XCTAssertNil(regular.gated)
            try assertSame(regular.y, reference)
            try assertSame(diagnostic.y, reference)
            try assertSame(XCTUnwrap(diagnostic.gate), gate)
            try assertSame(XCTUnwrap(diagnostic.gated), gated)
        }
    }

    func testProductMustRoundBeforeOutputAddition() throws {
        let fused = try GPUMoEFused(hidden: 4, intermediate: 8, topK: 1, groupSize: 64, bits: 4)
        let routed = try MX.array([Float](repeating: -0.734375, count: 4), shape: [1, 1, 4], dtype: MLX_BFLOAT16)
        let down = try MX.array([Float](repeating: 1.0078125, count: 4), shape: [1, 1, 4], dtype: MLX_BFLOAT16)
        let gate = try MX.array([Float(1)], shape: [1, 1, 1], dtype: MLX_BFLOAT16)
        let result = try fused.sharedOutput(routed: routed, down: down, gateLogits: gate, diagnostics: true)
        // BF16 sigmoid(1)=0.73046875. Its product with 1.0078125 rounds
        // to 0.734375 before addition. FP32/FMA would leave ~0.00180054.
        XCTAssertEqual(try XCTUnwrap(result.gate).floats(), [0.73046875])
        XCTAssertEqual(try XCTUnwrap(result.gated).floats(), [Float](repeating: 0.734375, count: 4))
        XCTAssertEqual(try result.y.floats(), [Float](repeating: 0, count: 4))
    }

    func testRejectsMultiTokenOrWrongDtypeAndShape() throws {
        let fused = try GPUMoEFused(hidden: 4, intermediate: 8, topK: 1, groupSize: 64, bits: 4)
        let single = try MX.zeros([1, 1, 4], MLX_BFLOAT16)
        let multi = try MX.zeros([1, 2, 4], MLX_BFLOAT16)
        let float = try MX.zeros([1, 1, 4], MLX_FLOAT32)
        let gate = try MX.zeros([1, 1, 1], MLX_BFLOAT16)
        XCTAssertThrowsError(try fused.sharedActivation(multi, up: multi))
        XCTAssertThrowsError(try fused.sharedActivation(single, up: float))
        XCTAssertThrowsError(try fused.sharedOutput(routed: multi, down: single, gateLogits: gate, diagnostics: false))
        XCTAssertThrowsError(try fused.sharedOutput(routed: single, down: single, gateLogits: single, diagnostics: false))
    }
}
