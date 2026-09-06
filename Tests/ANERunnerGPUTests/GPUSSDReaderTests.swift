import ANERunnerCore
import Foundation
import XCTest
@testable import ANERunnerGPU

final class GPUSSDReaderTests: XCTestCase {
    private let payload: [UInt8] = [
        0x00, 0x80, 0x38, 0xb8, // 0, -0, 1, -1
        0x40, 0xc0, 0x30, 0xb0, // 2, -2, 0.5, -0.5
        0x01, 0x81, 0x7e, 0xfe, // +/- 1/512, +/- 448
        0x48, 0xc8, 0x08, 0x88, // +/- 4, +/- 1/64
    ]
    // Independently known E4M3FN values; compare bit patterns to retain -0.
    private let decodedRows: [[Float]] = [
        [0, -Float.zero, 1, -1], [2, -2, 0.5, -0.5],
        [1 / 512, -1 / 512, 448, -448], [4, -4, 1 / 64, -1 / 64],
    ]

    private func fixture(_ bytes: [UInt8]) throws -> URL {
        let directory = try GPUFixtureLocation.temporaryDirectory()
        let object: [String: Any] = [
            "__metadata__": ["format": "mlx-serve-ngram-fp8", "scale": "1"],
            "weight": ["dtype": "F8_E4M3", "shape": [4, 4], "data_offsets": [0, 16]],
        ]
        var header = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        header.append(Data(repeating: 32, count: (8 - header.count % 8) % 8))
        var count = UInt64(header.count).littleEndian
        var contents = withUnsafeBytes(of: &count) { Data($0) }
        contents.append(header); contents.append(contentsOf: bytes)
        try contents.write(to: directory.appendingPathComponent("table.bin"))
        return directory
    }

    func testOrderDuplicatesUnevenPartitionsAndSmallBatches() throws {
        let directory = try fixture(payload)
        defer { try? FileManager.default.removeItem(at: directory) }
        let table = try NGramTable(url: directory.appendingPathComponent("table.bin"))
        let requests = [[Int](), [3], [2, 0], [1, 0, 3], [3, 0, 2, 3, 1, 0, 2, 2, 1]]
        for workers in 1...GPUSSDReader.maximumWorkers {
            for rows in requests {
                let actual = try GPUSSDReader.readRows(table: table, rows: rows, workers: workers)
                let expected = rows.flatMap { decodedRows[$0] }
                XCTAssertEqual(actual.map(\.bitPattern), expected.map(\.bitPattern), "workers=\(workers), rows=\(rows)")
            }
        }
    }

    func testInvalidWorkerCountsAreRejectedIncludingEmptyInput() throws {
        let directory = try fixture(payload)
        defer { try? FileManager.default.removeItem(at: directory) }
        let table = try NGramTable(url: directory.appendingPathComponent("table.bin"))
        for count in [-1, 0, GPUSSDReader.maximumWorkers + 1, Int.max] {
            XCTAssertThrowsError(try GPUSSDReader.readRows(table: table, rows: [], workers: count)) {
                XCTAssertEqual($0 as? GPUSSDReaderError, .invalidWorkerCount(count))
            }
        }
    }

    func testWholeRequestValidationPrecedesAnyNonfiniteRead() throws {
        var bytes = payload; bytes[0] = 0x7f
        let directory = try fixture(bytes)
        defer { try? FileManager.default.removeItem(at: directory) }
        let table = try NGramTable(url: directory.appendingPathComponent("table.bin"))
        for workers in 1...GPUSSDReader.maximumWorkers {
            XCTAssertThrowsError(try GPUSSDReader.readRows(table: table, rows: [0, 1, 4, -1], workers: workers)) {
                XCTAssertEqual($0 as? NGramTableError, .rowOutOfBounds(4))
            }
            XCTAssertThrowsError(try GPUSSDReader.readRows(table: table, rows: [0, -1, 4], workers: workers)) {
                XCTAssertEqual($0 as? NGramTableError, .rowOutOfBounds(-1))
            }
        }
    }

    func testParallelFailuresUseInputOrderAndDoNotPoisonLaterCalls() throws {
        var bytes = payload; bytes[4] = 0x7f; bytes[14] = 0xff
        let directory = try fixture(bytes)
        defer { try? FileManager.default.removeItem(at: directory) }
        let table = try NGramTable(url: directory.appendingPathComponent("table.bin"))
        for _ in 0..<20 {
            // Two worker failures: row 3 appears first in the request even
            // though row 1 has the lower physical offset and column number.
            XCTAssertThrowsError(try GPUSSDReader.readRows(table: table, rows: [2, 3, 0, 1], workers: 4)) {
                XCTAssertEqual($0 as? NGramTableError, .nonFiniteFP8(row: 3, column: 2))
            }
        }
        let actual = try GPUSSDReader.readRows(table: table, rows: [2, 0, 2], workers: 4)
        XCTAssertEqual(actual.map(\.bitPattern), (decodedRows[2] + decodedRows[0] + decodedRows[2]).map(\.bitPattern))
    }

    func testTruncationAfterOpeningSurfacesOriginalIOError() throws {
        let directory = try fixture(payload)
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("table.bin")
        let table = try NGramTable(url: file)
        let writer = try FileHandle(forWritingTo: file)
        try writer.truncate(atOffset: UInt64(table.dataOffset + 8))
        try writer.close()
        XCTAssertThrowsError(try GPUSSDReader.readRows(table: table, rows: [0, 1, 2, 3], workers: 4)) {
            XCTAssertEqual($0 as? NGramTableError, .invalidFileLength)
        }
        XCTAssertEqual(try GPUSSDReader.readRows(table: table, rows: [], workers: 4), [])
    }
}
