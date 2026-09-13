import CMLX
import Foundation
import XCTest
@testable import ANERunnerGPU

/// State-boundary coverage with real MLX storage but no model weights. This
/// does not claim learned QSA or whole-model forward equivalence; the controller
/// must also run the model oracle before enabling the experimental policy.
final class GPUPagedAttentionStateTests: XCTestCase {
    private func library() throws -> String {
        let environment = ProcessInfo.processInfo.environment
        guard environment["ANERUNNER_TEST_PAGED_ATTENTION_GPU"] == "1",
              let path = environment["ANERUNNER_TEST_PAGED_KV_LIBRARY"], path.hasPrefix("/") else {
            throw XCTSkip("Requires controlled GPU window, ANERUNNER_TEST_PAGED_ATTENTION_GPU=1 and absolute ANERUNNER_TEST_PAGED_KV_LIBRARY")
        }
        return path
    }

    private func pattern(rows: Int, width: Int, heads: Int? = nil, salt: Int) throws -> Tensor {
        let headCount = heads ?? 1
        var values = [Float](repeating: 0, count: headCount * rows * width)
        for head in 0..<headCount {
            for row in 0..<rows {
                for column in 0..<width {
                    values[(head * rows + row) * width + column] =
                        Float((row * 43 + head * 71 + column * 19 + salt) % 509 - 254) / 128
                }
            }
        }
        return try MX.array(values, shape: heads.map { [1,$0,rows,width] } ?? [1,rows,width], dtype: MLX_BFLOAT16)
    }

    private func compact(rows: Int) throws -> GPUAttention.State {
        let pooled = rows > 2051 ? try pattern(rows: rows / 4, width: 128, salt: 211) : nil
        let state = try GPUAttention.State(
            keys: pattern(rows: rows, width: 256, heads: 2, salt: 11),
            values: pattern(rows: rows, width: 256, heads: 2, salt: 137),
            rawIndexerKeys: pattern(rows: rows, width: 128, salt: 53),
            pooledIndexerKeys: pooled, offset: rows)
        try evaluate(state)
        return state
    }

    private func evaluate(_ state: GPUAttention.State) throws {
        try MX.eval(state.evaluationTensors)
        try MX.synchronize()
    }

    private func bytes(_ tensor: Tensor) throws -> Data {
        var data = Data()
        try QwenPrefixStateArchiveBytes.append(tensor, to: &data)
        return data
    }

    private func assertKV(_ state: GPUAttention.State, equals expected: GPUAttention.State,
                          file: StaticString = #filePath, line: UInt = #line) throws {
        let actual = try XCTUnwrap(state.materializedKV())
        let oracle = try XCTUnwrap(expected.materializedKV())
        try MX.eval([actual.keys,actual.values,oracle.keys,oracle.values]); try MX.synchronize()
        XCTAssertEqual(try bytes(actual.keys), try bytes(oracle.keys), file: file, line: line)
        XCTAssertEqual(try bytes(actual.values), try bytes(oracle.values), file: file, line: line)
        XCTAssertEqual(state.offset, expected.offset, file: file, line: line)
        XCTAssertEqual(try bytes(XCTUnwrap(state.rawIndexerKeys)),
                       try bytes(XCTUnwrap(expected.rawIndexerKeys)), file: file, line: line)
        XCTAssertEqual(state.pooledIndexerKeys?.shape, expected.pooledIndexerKeys?.shape, file: file, line: line)
        if let actualPool = state.pooledIndexerKeys, let expectedPool = expected.pooledIndexerKeys {
            XCTAssertEqual(try bytes(actualPool), try bytes(expectedPool), file: file, line: line)
        }
    }

    func testConversionEvaluationAndFullPrefixNeverExportKV() throws {
        let libraryPath = try library()
        for rows in [31,32,33,2051,2052] {
            let pool = try GPUPagedKVPool(libraryPath: libraryPath, maximumPages: (rows + 31) / 32 + 2)
            let dense = try compact(rows: rows)
            var state = dense
            try state.usePagedKV(pool: pool)
            XCTAssertNil(state.keys); XCTAssertNil(state.values)
            XCTAssertThrowsError(try state.tensors)
            XCTAssertEqual(try state.evaluationTensors.count, rows > 2051 ? 3 : 2)
            try evaluate(state)
            let pageState = try XCTUnwrap(state.pagedKV)
            XCTAssertEqual(pageState.logicalTokens, rows)
            XCTAssertEqual(try pool.statistics.encodedWrites, 1)
            XCTAssertEqual(try pool.statistics.encodedMaterializations, 0)
            let retained = try GPUAttention.retainedState(state, count: rows)
            XCTAssertTrue(retained.pagedKV === pageState)
            try evaluate(retained)
            XCTAssertEqual(try pool.statistics.encodedMaterializations, 0)
            try assertKV(state, equals: dense)
            XCTAssertEqual(try pool.statistics.encodedMaterializations, 1)

            let trimmed = try GPUAttention.retainedState(state, count: rows - 1)
            let denseTrimmed = try GPUAttention.retainedState(dense, count: rows - 1)
            XCTAssertNil(trimmed.pagedKV)
            try evaluate(trimmed); try evaluate(denseTrimmed)
            try assertKV(trimmed, equals: denseTrimmed)
            if rows == 2052 { XCTAssertNil(trimmed.pooledIndexerKeys) }
            let empty = try GPUAttention.retainedState(state, count: 0)
            XCTAssertNil(empty.pagedKV); XCTAssertNil(try empty.materializedKV())
        }
    }

    func testConversionFailureKeepsOriginalDenseOrPagedState() throws {
        let libraryPath = try library()
        let pool = try GPUPagedKVPool(libraryPath: libraryPath, maximumPages: 2)
        var empty = GPUAttention.State()
        XCTAssertThrowsError(try empty.usePagedKV(pool: pool))
        XCTAssertNil(empty.pagedKV); XCTAssertEqual(try pool.statistics.livePages, 0)

        var tooLarge = try compact(rows: 65)
        let oldKeys = try XCTUnwrap(tooLarge.keys), oldValues = try XCTUnwrap(tooLarge.values)
        XCTAssertThrowsError(try tooLarge.usePagedKV(pool: pool))
        XCTAssertTrue(tooLarge.keys === oldKeys); XCTAssertTrue(tooLarge.values === oldValues)
        XCTAssertNil(tooLarge.pagedKV); XCTAssertEqual(try pool.statistics.livePages, 0)

        var valid = try compact(rows: 33)
        try valid.usePagedKV(pool: pool); try evaluate(valid)
        let pageState = try XCTUnwrap(valid.pagedKV), statistics = try pool.statistics
        XCTAssertThrowsError(try valid.usePagedKV(pool: pool))
        XCTAssertTrue(valid.pagedKV === pageState)
        XCTAssertEqual(try pool.statistics, statistics)
        XCTAssertThrowsError(try valid.kvCapacityWorkspaceBytesForNextRow(rowLimit: 64))
        let row = try pattern(rows: 1, width: 256, heads: 2, salt: 17)
        XCTAssertThrowsError(try valid.appendCapacity(keys: row, values: row, rowLimit: 64))
        XCTAssertTrue(valid.pagedKV === pageState)
    }

    func testPublicSettersDropPagedOwnerAndRejectIncompleteDensePair() throws {
        let libraryPath = try library()
        for replaceKeys in [true,false] {
            let pool = try GPUPagedKVPool(libraryPath: libraryPath, maximumPages: 4)
            let dense = try compact(rows: 33)
            var state = dense
            try state.usePagedKV(pool: pool); try evaluate(state)
            let immutableSource = state
            if replaceKeys { state.keys = dense.keys } else { state.values = dense.values }
            XCTAssertNil(state.pagedKV)
            XCTAssertThrowsError(try state.evaluationTensors)
            XCTAssertThrowsError(try state.materializedKV())
            if replaceKeys { state.values = dense.values } else { state.keys = dense.keys }
            try evaluate(state)
            try assertKV(state, equals: dense)
            try assertKV(immutableSource, equals: dense)

            var cleared = immutableSource
            if replaceKeys { cleared.keys = nil } else { cleared.values = nil }
            XCTAssertNil(cleared.pagedKV)
            XCTAssertThrowsError(try cleared.evaluationTensors)
            cleared.reset()
            XCTAssertEqual(try cleared.evaluationTensors.count, 0)
        }
    }

    func testForkInstallationValidatesQSAAndOffsetWithoutExport() throws {
        let libraryPath = try library()
        let pool = try GPUPagedKVPool(libraryPath: libraryPath, maximumPages: 67)
        var source = try compact(rows: 2052)
        try source.usePagedKV(pool: pool); try evaluate(source)
        let pageState = try XCTUnwrap(source.pagedKV), fork = try pageState.fork()
        let before = try pool.statistics
        var installed = GPUAttention.State(rawIndexerKeys: source.rawIndexerKeys,
            pooledIndexerKeys: source.pooledIndexerKeys, offset: source.offset)
        try installed.installPagedKV(fork); try evaluate(installed)
        XCTAssertTrue(installed.pagedKV === fork)
        XCTAssertEqual(try fork.pageIDs, try pageState.pageIDs)
        XCTAssertEqual(try pool.statistics, before)

        var wrongOffset = GPUAttention.State(rawIndexerKeys: source.rawIndexerKeys,
            pooledIndexerKeys: source.pooledIndexerKeys, offset: source.offset - 1)
        XCTAssertThrowsError(try wrongOffset.installPagedKV(fork))
        XCTAssertNil(wrongOffset.pagedKV)
        var missingPool = GPUAttention.State(rawIndexerKeys: source.rawIndexerKeys, offset: source.offset)
        XCTAssertThrowsError(try missingPool.installPagedKV(fork))
        XCTAssertNil(missingPool.pagedKV)
        XCTAssertThrowsError(try installed.installPagedKV(fork))
        XCTAssertEqual(try pool.statistics, before)
    }
}
