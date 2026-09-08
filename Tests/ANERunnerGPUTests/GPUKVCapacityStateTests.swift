import CMLX
import Foundation
import XCTest
@testable import ANERunnerGPU

/// Real MLX storage tests, no model/weights/tokenizer/native diagnostic dylib.
/// Opt in only inside the root controller's exclusive GPU window. These do
/// not exercise QSA scoring or the full forward; synthetic pooled rows are
/// immutable fixtures, and selected appends never add a complete pooled block.
final class GPUKVCapacityStateTests: XCTestCase {
    private func requireGPUWindow() throws {
        guard ProcessInfo.processInfo.environment["ANERUNNER_TEST_KV_CAPACITY_GPU"] == "1" else {
            throw XCTSkip("Requires the controlled GPU window and ANERUNNER_TEST_KV_CAPACITY_GPU=1")
        }
    }

    private func pattern(rows: Int, width: Int, heads: Int? = nil,
                         start: Int = 0, salt: Int) throws -> Tensor {
        let headCount = heads ?? 1
        var values = [Float](repeating: 0, count: headCount * rows * width)
        for head in 0..<headCount {
            for row in 0..<rows {
                for dimension in 0..<width {
                    let value = ((start + row) * 43 + head * 71 + dimension * 19 + salt) % 509 - 254
                    values[(head * rows + row) * width + dimension] = Float(value) / 16
                }
            }
        }
        let shape = heads.map { [1,$0,rows,width] } ?? [1,rows,width]
        return try MX.array(values, shape: shape, dtype: MLX_BFLOAT16)
    }
    private func compactState(rows: Int) throws -> GPUAttention.State {
        let pooled = rows > 2051 ? try pattern(rows: rows / 4, width: 128, salt: 211) : nil
        let state = try GPUAttention.State(
            keys: pattern(rows: rows, width: 256, heads: 2, salt: 11),
            values: pattern(rows: rows, width: 256, heads: 2, salt: 137),
            rawIndexerKeys: pattern(rows: rows, width: 128, salt: 53),
            pooledIndexerKeys: pooled, offset: rows)
        try ready(state)
        return state
    }
    private func ready(_ state: GPUAttention.State) throws {
        try MX.eval(state.tensors); try MX.synchronize()
    }
    private func bits(_ tensor: Tensor) throws -> Data {
        var data = Data()
        try QwenPrefixStateArchiveBytes.append(tensor, to: &data)
        return data
    }
    private func bits(_ state: GPUAttention.State) throws -> [String: Data] {
        var result: [String: Data] = [:]
        for (name,tensor) in [("keys",state.keys),("values",state.values),
                              ("raw",state.rawIndexerKeys),("pooled",state.pooledIndexerKeys)] {
            if let tensor { result[name] = try bits(tensor) }
        }
        return result
    }
    private func headStride(_ tensor: Tensor) throws -> Int {
        Int(try XCTUnwrap(mlx_array_strides(tensor.handle))[1])
    }
    private func append(_ state: inout GPUAttention.State, rowLimit: Int = 2400) throws {
        let offset = state.offset
        guard offset / 4 == (offset + 1) / 4 else {
            throw GPUError.invalid("State fixture must not simulate a new QSA pooled block")
        }
        let raw = try XCTUnwrap(state.rawIndexerKeys)
        let keys = try pattern(rows: 1, width: 256, heads: 2, start: offset, salt: 11)
        let values = try pattern(rows: 1, width: 256, heads: 2, start: offset, salt: 137)
        let rawRow = try pattern(rows: 1, width: 128, start: offset, salt: 53)
        _ = try state.appendCapacity(keys: keys, values: values, rowLimit: rowLimit)
        // The same raw concat and final metadata transition as forward, with
        // supplied raw fixtures instead of running projection/QSA math.
        state.rawIndexerKeys = try MX.concat([raw,rawRow], axis: 1)
        state.finishForward(offset: offset + 1)
        try ready(state)
    }
    private func assertCanonical(_ state: GPUAttention.State, rows: Int,
                                 file: StaticString = #filePath, line: UInt = #line) throws {
        let expected = try compactState(rows: rows)
        XCTAssertEqual(state.offset, rows, file: file, line: line)
        XCTAssertEqual(try bits(state), try bits(expected), file: file, line: line)
        XCTAssertEqual(state.keys?.shape, [1,2,rows,256], file: file, line: line)
        XCTAssertEqual(state.values?.shape, [1,2,rows,256], file: file, line: line)
        XCTAssertEqual(state.rawIndexerKeys?.shape, [1,rows,128], file: file, line: line)
    }

    func testPublicKeysAndValuesReplacementUseActualNewContentsAndKeepOldAliases() throws {
        try requireGPUWindow()
        for replaceKeys in [true,false] {
            var state = try compactState(rows: 2052)
            try append(&state) // logical 2053, KV capacity 2304, raw extent 2053.
            XCTAssertEqual(try state.kvCapacityWorkspaceBytesForNextRow(rowLimit: 2400), 0)
            let oldState = state
            let oldKeys = try XCTUnwrap(state.keys), oldValues = try XCTUnwrap(state.values)
            let oldStateBits = try bits(oldState)
            let oldKeyBits = try bits(oldKeys), oldValueBits = try bits(oldValues)
            // Reassigning the identical wrapper must preserve the valid owner.
            state.keys = oldKeys; state.values = oldValues
            XCTAssertEqual(try state.kvCapacityWorkspaceBytesForNextRow(rowLimit: 2400), 0)

            let replacement = try pattern(rows: 2053, width: 256, heads: 2, salt: 307)
            try replacement.eval()
            let replacementBits = try bits(replacement)
            if replaceKeys { state.keys = replacement }
            else { state.values = replacement }
            XCTAssertGreaterThan(try state.kvCapacityWorkspaceBytesForNextRow(rowLimit: 2400), 0)
            let keyRow = try pattern(rows: 1, width: 256, heads: 2, start: 2053, salt: 11)
            let valueRow = try pattern(rows: 1, width: 256, heads: 2, start: 2053, salt: 137)
            let expectedKeys = try MX.concat([replaceKeys ? replacement : oldKeys,keyRow], axis: 2)
            let expectedValues = try MX.concat([replaceKeys ? oldValues : replacement,valueRow], axis: 2)
            try MX.eval([expectedKeys,expectedValues]); try MX.synchronize()

            try withExtendedLifetime((oldState,oldKeys,oldValues,replacement)) {
                try append(&state)
                XCTAssertEqual(state.offset, 2054)
                XCTAssertEqual(try bits(XCTUnwrap(state.keys)), try bits(expectedKeys))
                XCTAssertEqual(try bits(XCTUnwrap(state.values)), try bits(expectedValues))
                XCTAssertEqual(try bits(oldState), oldStateBits)
                XCTAssertEqual(try bits(oldKeys), oldKeyBits)
                XCTAssertEqual(try bits(oldValues), oldValueBits)
                XCTAssertEqual(try bits(replacement), replacementBits)
            }
            XCTAssertEqual(try state.kvCapacityWorkspaceBytesForNextRow(rowLimit: 2400), 0)
        }
    }

    func testNilReplacementCannotResurrectTheOldCapacityOwner() throws {
        try requireGPUWindow()
        for removeKeys in [true,false] {
            var state = try compactState(rows: 2052)
            try append(&state)
            let oldState = state, oldBits = try bits(state)
            let savedKeys = try XCTUnwrap(state.keys), savedValues = try XCTUnwrap(state.values)
            if removeKeys { state.keys = nil } else { state.values = nil }
            XCTAssertGreaterThan(try state.kvCapacityWorkspaceBytesForNextRow(rowLimit: 2400), 0)
            let row = try pattern(rows: 1, width: 256, heads: 2, start: 2053, salt: 31)
            XCTAssertThrowsError(try state.appendCapacity(keys: row, values: row, rowLimit: 2400))
            XCTAssertEqual(state.offset, 2053)
            XCTAssertEqual(try bits(oldState), oldBits)
            if removeKeys { XCTAssertNil(state.keys); state.keys = savedKeys }
            else { XCTAssertNil(state.values); state.values = savedValues }
            // Restoring the old wrapper does not silently revive its old helper.
            XCTAssertGreaterThan(try state.kvCapacityWorkspaceBytesForNextRow(rowLimit: 2400), 0)
            try append(&state)
            try assertCanonical(state, rows: 2054)
            XCTAssertEqual(try bits(oldState), oldBits)
        }
    }

    func testValidCapacityAppendKeepsOldStateAndLogicalViewsImmutable() throws {
        try requireGPUWindow()
        var state = try compactState(rows: 2052)
        try append(&state)
        let oldState = state, before = try bits(state)
        let oldKeys = try XCTUnwrap(state.keys), oldValues = try XCTUnwrap(state.values)
        // Alias COW consumes persistent old/new allowance, not a growth permit.
        XCTAssertEqual(try state.kvCapacityWorkspaceBytesForNextRow(rowLimit: 2400), 0)
        try withExtendedLifetime((oldState,oldKeys,oldValues)) {
            try append(&state)
            try assertCanonical(state, rows: 2054)
            XCTAssertEqual(try bits(oldState), before)
            XCTAssertEqual(try bits(oldKeys), before["keys"])
            XCTAssertEqual(try bits(oldValues), before["values"])
        }
    }

    func testLargeKVPaddingCompactsWithoutTrimmingRawAndDropsPoolBelowQSA() throws {
        try requireGPUWindow()
        var state = try compactState(rows: 2052)
        try append(&state)
        let oldKeys = try XCTUnwrap(state.keys), oldValues = try XCTUnwrap(state.values)
        let originalBits = try bits(state)
        XCTAssertEqual(try headStride(oldKeys), 2304 * 256)
        let noLogicalCut = try GPUAttention.retainedState(state, count: 2053)
        let belowQSA = try GPUAttention.retainedState(state, count: 2051)
        try ready(noLogicalCut); try ready(belowQSA)
        try assertCanonical(noLogicalCut, rows: 2053)
        try assertCanonical(belowQSA, rows: 2051)
        XCTAssertEqual(try headStride(XCTUnwrap(noLogicalCut.keys)), 2053 * 256)
        XCTAssertEqual(try headStride(XCTUnwrap(noLogicalCut.values)), 2053 * 256)
        XCTAssertEqual(try headStride(XCTUnwrap(belowQSA.keys)), 2051 * 256)
        XCTAssertEqual(try headStride(XCTUnwrap(belowQSA.values)), 2051 * 256)
        XCTAssertEqual(noLogicalCut.diagnosticRetainedStorage, [2053,513])
        XCTAssertEqual(belowQSA.diagnosticRetainedStorage, [2053,0])
        XCTAssertNil(belowQSA.pooledIndexerKeys)
        XCTAssertGreaterThan(try noLogicalCut.kvCapacityWorkspaceBytesForNextRow(rowLimit: 2400), 0)
        XCTAssertEqual(try bits(state), originalBits)
        XCTAssertEqual(try bits(oldKeys), originalBits["keys"])
        XCTAssertEqual(try bits(oldValues), originalBits["values"])
    }

    func testRepeatedSmallCutsCarryIndependentKVAndRawExtents() throws {
        try requireGPUWindow()
        var state = try compactState(rows: 2300)
        try append(&state) // logical/raw 2301; KV capacity 2304.
        let originalBits = try bits(state)
        let originalKeys = try XCTUnwrap(state.keys), originalValues = try XCTUnwrap(state.values)
        let first = try GPUAttention.retainedState(state, count: 2300)
        let second = try GPUAttention.retainedState(first, count: 2299)
        let third = try GPUAttention.retainedState(second, count: 2296)
        for (trimmed,rows) in [(first,2300),(second,2299),(third,2296)] {
            try ready(trimmed); try assertCanonical(trimmed, rows: rows)
        }
        // K/V: tail4 remains a capacity view; tail5 compacts; the subsequent
        // tail3 can retain that smaller 2299-row allocation.
        for tensor in [try XCTUnwrap(first.keys),try XCTUnwrap(first.values)] {
            XCTAssertEqual(try headStride(tensor), 2304 * 256)
        }
        for tensor in [try XCTUnwrap(second.keys),try XCTUnwrap(second.values),
                       try XCTUnwrap(third.keys),try XCTUnwrap(third.values)] {
            XCTAssertEqual(try headStride(tensor), 2299 * 256)
        }
        // Raw's own retained tail reaches5 only at the third cut. The original
        // pooled rule also compacts its one extra block when raw takes this cut.
        XCTAssertEqual(first.diagnosticRetainedStorage, [2301,575])
        XCTAssertEqual(second.diagnosticRetainedStorage, [2301,575])
        XCTAssertEqual(third.diagnosticRetainedStorage, [2296,574])
        XCTAssertEqual(try bits(state), originalBits)
        XCTAssertEqual(try bits(originalKeys), originalBits["keys"])
        XCTAssertEqual(try bits(originalValues), originalBits["values"])
    }
}
