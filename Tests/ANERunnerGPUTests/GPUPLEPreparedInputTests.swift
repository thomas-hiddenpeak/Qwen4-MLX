import ANERunnerCore
import Dispatch
import Foundation
import XCTest
@testable import ANERunnerGPU

/// These tests only hash tokens and pread a tiny temporary file; no model or
/// MLX tensor is constructed. Existing NGramTests cover the hash/FP8 oracles.
final class GPUPLEPreparedInputTests: XCTestCase {
    private func fixture() throws -> (directory: URL, table: NGramTable, hash: NGramHash) {
        let hash = try NGramHash(unigramVocabularySize: 100, headsPerNGram: 1,
                                vocabularyBase: 17, vocabularyDivisor: 1, eosTokenID: 9)
        let directory = try GPUFixtureLocation.temporaryDirectory()
        let object: [String: Any] = [
            "__metadata__": ["format": "mlx-serve-ngram-fp8", "scale": "1"],
            "weight": ["dtype": "F8_E4M3", "shape": [hash.totalRows, 1],
                       "data_offsets": [0, hash.totalRows]],
        ]
        var header = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        header.append(Data(repeating: 32, count: (8 - header.count % 8) % 8))
        var count = UInt64(header.count).littleEndian
        var contents = withUnsafeBytes(of: &count) { Data($0) }
        contents.append(header)
        contents.append(contentsOf: (0..<hash.totalRows).map { UInt8(0x20 + $0) })
        let url = directory.appendingPathComponent("table.bin")
        try contents.write(to: url)
        return (directory, try NGramTable(url: url), hash)
    }

    func testLookaheadPreservesStateAndMatchesDirectReadsAcrossEOS() throws {
        let fixture = try fixture()
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        let allTokens: [Int32] = [5, 7, 9, 2, 11]
        let expectedRows = try fixture.hash.rowIDs(previousTokens: fixture.hash.initialHistory,
                                                   tokens: allTokens.map(UInt32.init))
        let expected = try fixture.table.readRows(expectedRows)
        for workers in [1, 4] {
            let queue = DispatchQueue(label: "ple-test.lookahead.\(workers)")
            var history: [UInt32] = []
            let first = try GPUPLE.PreparedInput(tokens: [5, 7, 9], history: history,
                hash: fixture.hash, table: fixture.table, workers: workers, queue: queue)
            defer { first.readTask.drain() }
            let next = try GPUPLE.PreparedInput(tokens: [2, 11], history: first.historyAfter,
                hash: fixture.hash, table: fixture.table, workers: workers, queue: queue)
            defer { next.readTask.drain() }
            XCTAssertEqual(history, [])
            XCTAssertEqual(first.historyBefore, [9, 9])
            XCTAssertEqual(first.historyAfter, [7, 9])
            XCTAssertEqual(next.historyBefore, [7, 9])
            XCTAssertEqual(next.historyAfter, [2, 11])
            // Reading lookahead does not advance the request's history.
            _ = try next.readTask.wait()
            XCTAssertEqual(history, [])
            XCTAssertThrowsError(try next.consume(tokens: [2, 11], history: &history, table: fixture.table))
            XCTAssertEqual(history, [])
            let left = try first.consume(tokens: [5, 7, 9], history: &history, table: fixture.table)
            XCTAssertTrue(left === first.readTask)
            XCTAssertEqual(history, [7, 9])
            let right = try next.consume(tokens: [2, 11], history: &history, table: fixture.table)
            XCTAssertEqual(history, [2, 11])
            XCTAssertEqual(try (left.wait() + right.wait()).map(\.bitPattern), expected.map(\.bitPattern))
            XCTAssertEqual(left.logicalBytes + right.logicalBytes, allTokens.count * fixture.hash.headCount)
        }
    }

    func testMismatchesPreserveHistoryAndValidConsumptionStillWorks() throws {
        let fixture = try fixture()
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        let prepared = try GPUPLE.PreparedInput(tokens: [5], history: [], hash: fixture.hash,
                                               table: fixture.table, workers: 1)
        defer { prepared.readTask.drain() }
        var history: [UInt32] = []
        XCTAssertThrowsError(try prepared.consume(tokens: [6], history: &history, table: fixture.table))
        XCTAssertEqual(history, [])
        history = [1, 2]
        XCTAssertThrowsError(try prepared.consume(tokens: [5], history: &history, table: fixture.table))
        XCTAssertEqual(history, [1, 2])
        let anotherTable = try NGramTable(url: fixture.directory.appendingPathComponent("table.bin"))
        history = []
        XCTAssertThrowsError(try prepared.consume(tokens: [5], history: &history, table: anotherTable))
        XCTAssertEqual(history, [])
        // Empty state and explicit initial EOS history are equivalent.
        history = fixture.hash.initialHistory
        _ = try prepared.consume(tokens: [5], history: &history, table: fixture.table)
        XCTAssertEqual(history, [9, 5])
    }

    func testDrainJoinsQueuedReadFailureAndLaterWorkStillCompletes() throws {
        let fixture = try fixture()
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        let queue = DispatchQueue(label: "ple-test.failure")
        queue.suspend()
        var resumed = false
        defer { if !resumed { queue.resume() } }
        let writer = try FileHandle(forWritingTo: fixture.directory.appendingPathComponent("table.bin"))
        defer { try? writer.close() }
        let failed = PLEReadTask(table: fixture.table, rows: [1], workers: 4, queue: queue)
        let next = PLEReadTask(table: fixture.table, rows: [0], workers: 4, queue: queue)
        defer {
            if !resumed { queue.resume(); resumed = true }
            failed.drain()
            next.drain()
        }
        // Both reads must wait for the supplied request queue. Make row 1
        // unavailable before resuming it, leaving row 0 readable.
        try writer.truncate(atOffset: UInt64(fixture.table.dataOffset + 1))
        queue.resume(); resumed = true
        // Cancellation discards the speculative result but still joins it.
        failed.drain()
        next.drain()
        XCTAssertThrowsError(try failed.wait()) {
            XCTAssertEqual($0 as? NGramTableError, .invalidFileLength)
        }
        XCTAssertEqual(try next.wait(), [0.125]) // E4M3FN 0x20
        // The same failure remains observable after drain; no zero substitute.
        XCTAssertThrowsError(try failed.wait())
    }

    func testInvalidInputsThrowAndEmptyChunkKeepsNormalizedHistory() throws {
        let fixture = try fixture()
        defer { try? FileManager.default.removeItem(at: fixture.directory) }
        XCTAssertThrowsError(try GPUPLE.PreparedInput(tokens: [-1], history: [], hash: fixture.hash,
                                                     table: fixture.table, workers: 1))
        XCTAssertThrowsError(try GPUPLE.PreparedInput(tokens: [5], history: [9], hash: fixture.hash,
                                                     table: fixture.table, workers: 1))
        XCTAssertThrowsError(try GPUPLE.PreparedInput(tokens: [], history: [], hash: fixture.hash,
                                                     table: fixture.table, workers: 0))
        let empty = try GPUPLE.PreparedInput(tokens: [], history: [], hash: fixture.hash,
                                             table: fixture.table, workers: 1)
        defer { empty.readTask.drain() }
        var history: [UInt32] = []
        let task = try empty.consume(tokens: [], history: &history, table: fixture.table)
        XCTAssertEqual(history, [9, 9])
        XCTAssertEqual(task.logicalBytes, 0)
        XCTAssertEqual(try task.wait(), [])
    }
}
