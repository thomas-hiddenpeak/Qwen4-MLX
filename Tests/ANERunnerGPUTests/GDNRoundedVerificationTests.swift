import CMLX
import Foundation
import XCTest
@testable import ANERunnerGPU

/// Actual recurrence kernels at the model's fixed head dimensions, with tiny
/// deterministic synthetic inputs. No checkpoint or complete model is loaded.
final class GDNRoundedVerificationTests: XCTestCase {
    private func tensor(shape: [Int], seed: UInt64, exponent: UInt16) throws -> Tensor {
        var random = seed
        let bits: [UInt16] = (0..<shape.reduce(1, *)).map { _ in
            random = random &* 6364136223846793005 &+ 1442695040888963407
            let sign = UInt16((random >> 63) & 1) << 15
            let exp = (exponent + UInt16((random >> 56) & 1)) << 7
            let fraction = UInt16((random >> 32) & 127)
            return (sign | exp | fraction).littleEndian
        }
        return try MX.array(data: bits.withUnsafeBytes { Data($0) }, shape: shape, dtype: MLX_BFLOAT16)
    }

    private func gates(_ bits: [UInt16]) throws -> Tensor {
        let words = (0..<(3 * 48)).map { bits[($0 / 48 + $0 % 48) % bits.count].littleEndian }
        return try MX.array(data: words.withUnsafeBytes { Data($0) }, shape: [1,3,48], dtype: MLX_BFLOAT16)
    }

    private func row(_ tensor: Tensor, at position: Int) throws -> Tensor {
        var starts = [Int](repeating: 0, count: tensor.shape.count)
        var ends = tensor.shape
        starts[1] = position; ends[1] = position + 1
        return try MX.slice(tensor, starts: starts, ends: ends)
    }

    private func bitPatterns(_ tensor: Tensor) throws -> [UInt32] {
        XCTAssertEqual(tensor.dtype, MLX_BFLOAT16)
        let values = try tensor.floats()
        XCTAssertTrue(values.allSatisfy(\.isFinite))
        return values.map(\.bitPattern)
    }

    private func assertSame(_ actual: [UInt32], _ expected: [UInt32], _ label: String,
                            file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(actual.count, expected.count, label, file: file, line: line)
        let mismatches = zip(actual, expected).enumerated().filter { $0.element.0 != $0.element.1 }
        XCTAssertEqual(mismatches.count, 0,
                       "\(label): first mismatching index \(mismatches.first?.offset.description ?? "none")",
                       file: file, line: line)
    }

    func testRoundedS3MatchesThreeScalarCallsAndDiffersFromDefaultS3() throws {
        let kernel = try GPUGatedDeltaNetFused()
        let q = try tensor(shape: [1,3,16,128], seed: 101, exponent: 120)
        let k = try tensor(shape: [1,3,16,128], seed: 211, exponent: 121)
        let v = try tensor(shape: [1,3,48,128], seed: 307, exponent: 125)
        let initial = try tensor(shape: [1,48,128,128], seed: 401, exponent: 118)
        let decay = try gates([0x3f60, 0x3f50, 0x3f40]) // .875, .8125, .75
        let beta = try gates([0x3f10, 0x3f20, 0x3f30]) // .5625, .625, .6875
        let baseline = try kernel.apply(q: q, k: k, v: v, decay: decay, beta: beta, state: initial)
        try MX.eval([baseline.y, baseline.state])

        var scalarState = initial
        var scalarOutputs: [Tensor] = []
        var scalarStates: [Tensor] = []
        for position in 0..<3 {
            // The unchanged default S1 kernel is the independent execution
            // oracle. Its output uses FP32 state, then persistent state is BF16.
            let step = try kernel.apply(q: row(q, at: position), k: row(k, at: position),
                v: row(v, at: position), decay: row(decay, at: position),
                beta: row(beta, at: position), state: scalarState)
            try MX.eval([step.y, step.state])
            scalarOutputs.append(step.y)
            scalarStates.append(step.state)
            scalarState = step.state
        }
        let expectedY = try MX.concat(scalarOutputs, axis: 1)
        let rounded = try kernel.apply(q: q, k: k, v: v, decay: decay, beta: beta,
                                       state: initial, roundStateEachToken: true)
        try MX.eval([rounded.y, rounded.state])
        XCTAssertEqual(rounded.y.shape, [1,3,48,128])
        XCTAssertEqual(rounded.state.shape, [1,48,128,128])
        try assertSame(bitPatterns(rounded.y), bitPatterns(expectedY), "all three token outputs")
        let roundedState = try bitPatterns(rounded.state)
        try assertSame(roundedState, bitPatterns(scalarState), "final persistent state")
        let baselineState = try bitPatterns(baseline.state)
        XCTAssertTrue(zip(baselineState, roundedState).contains { $0.0 != $0.1 },
                      "Fixture must detect omitted intermediate BF16 state stores")

        // Selecting the candidate must not change the cached default path.
        let explicitDefault = try kernel.apply(q: q, k: k, v: v, decay: decay, beta: beta,
                                               state: initial, roundStateEachToken: false)
        try assertSame(bitPatterns(explicitDefault.y), bitPatterns(baseline.y), "default outputs unchanged")
        try assertSame(bitPatterns(explicitDefault.state), baselineState, "default state unchanged")

        // Capture must preserve each mode's existing arithmetic. Rounded
        // snapshots match independent S1 calls; raw snapshots match complete
        // prefix calls which round only at that prefix's final store.
        let convInputs = try tensor(shape: [1,6,10240], seed: 503, exponent: 121)
        let convBits = try bitPatterns(convInputs)
        let finalConv = try MX.slice(convInputs, starts: [0,3,0], ends: [1,6,10240])
        for roundEachToken in [false, true] {
            let captured = try kernel.applyCapturing(q: q, k: k, v: v, decay: decay, beta: beta,
                state: initial, roundStateEachToken: roundEachToken)
            let verification = GPUGatedDeltaNet.VerificationCapture(initialOffset: 17,
                recurrentStates: captured.states, convInputs: convInputs)
            let complete = GPUGatedDeltaNet.State(convHistory: finalConv, recurrent: captured.state,
                offset: 20, verificationCapture: verification)
            XCTAssertEqual(captured.states.shape, [3,1,48,128,128])
            XCTAssertTrue(complete.tensors.contains { $0 === captured.states })
            XCTAssertTrue(complete.tensors.contains { $0 === convInputs })
            try MX.eval([captured.y] + complete.tensors)
            try assertSame(bitPatterns(captured.y), bitPatterns(roundEachToken ? rounded.y : baseline.y),
                           "capture preserves outputs, rounded=\(roundEachToken)")
            try assertSame(bitPatterns(captured.state), roundEachToken ? roundedState : baselineState,
                           "capture preserves final state, rounded=\(roundEachToken)")
            for count in 1...3 {
                let expectedState: Tensor
                if roundEachToken { expectedState = scalarStates[count-1] }
                else {
                    func prefix(_ input: Tensor) throws -> Tensor {
                        var ends = input.shape
                        ends[1] = count
                        return try MX.slice(input, starts: [Int](repeating: 0, count: ends.count), ends: ends)
                    }
                    expectedState = try kernel.apply(q: prefix(q), k: prefix(k), v: prefix(v),
                        decay: prefix(decay), beta: prefix(beta), state: initial).state
                }
                let snapshot = try MX.reshape(MX.slice(captured.states, starts: [count-1,0,0,0,0],
                    ends: [count,1,48,128,128]), [1,48,128,128])
                try assertSame(bitPatterns(snapshot), bitPatterns(expectedState),
                               "snapshot \(count), rounded=\(roundEachToken)")
                let committed = try complete.committingPrefix(count: count)
                XCTAssertEqual(committed.offset, 17 + count)
                XCTAssertNil(committed.verificationCapture)
                XCTAssertEqual(committed.tensors.count, 2)
                try MX.eval(committed.tensors)
                try assertSame(bitPatterns(XCTUnwrap(committed.recurrent)), bitPatterns(expectedState),
                               "committed recurrence \(count), rounded=\(roundEachToken)")
                try assertSame(bitPatterns(XCTUnwrap(committed.convHistory)),
                               Array(convBits[(count * 10240)..<((count + 3) * 10240)]),
                               "committed convolution \(count)")
                if count == 3 {
                    XCTAssertTrue(committed.recurrent === captured.state)
                    XCTAssertTrue(committed.convHistory === finalConv)
                }
            }
            XCTAssertThrowsError(try complete.committingPrefix(count: 0))
            XCTAssertThrowsError(try complete.committingPrefix(count: 4))
            // Committing a returned value must leave the snapshot reusable.
            XCTAssertNotNil(complete.verificationCapture)
            XCTAssertEqual(complete.offset, 20)
        }
        XCTAssertThrowsError(try GPUGatedDeltaNet.State().committingPrefix(count: 1))
        let tooWide = try MX.zeros([1,6,16,128], MLX_BFLOAT16)
        XCTAssertThrowsError(try kernel.applyCapturing(q: tooWide, k: k, v: v,
                                                       decay: decay, beta: beta, state: initial))
    }
}
