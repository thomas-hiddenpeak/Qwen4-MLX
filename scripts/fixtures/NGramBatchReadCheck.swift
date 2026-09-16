// Host-only regression for the production NGramTable batch reader.
// swiftc -O -swift-version 6 -parse-as-library Sources/ANERunnerCore/NGramTable.swift \
//   scripts/fixtures/NGramBatchReadCheck.swift -o /tmp/ngram-batch-check
// /tmp/ngram-batch-check [report.json]
// Uses only temporary ~82 KiB tables; never opens the real PLE table.
import Foundation

private struct CheckFailure: Error, CustomStringConvertible {
    let description: String
}

private func require(_ condition: @autoclosure () throws -> Bool, _ message: String) throws {
    if try !condition() { throw CheckFailure(description: message) }
}

private func expect(_ expected: NGramTableError, _ body: () throws -> Void) throws {
    do { try body(); throw CheckFailure(description: "expected \(expected), but read succeeded") }
    catch let actual as NGramTableError {
        try require(actual == expected, "expected \(expected), got \(actual)")
    }
}

private struct Fixture {
    let url: URL
    let original: Data
    let table: NGramTable

    init(directory: URL, name: String, invalidRows: [Int: Int] = [:]) throws {
        url = directory.appendingPathComponent(name)
        let rows = 512, dimension = 160
        let header: [String: Any] = [
            "__metadata__": ["format": "mlx-serve-ngram-fp8", "scale": "1.00390625"],
            "weight": ["dtype": "F8_E4M3", "shape": [rows, dimension],
                       "data_offsets": [0, rows * dimension]]
        ]
        var headerBytes = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        while headerBytes.count % 8 != 0 { headerBytes.append(32) }
        var data = Data((0..<8).map { UInt8(truncatingIfNeeded: UInt64(headerBytes.count) >> (8 * $0)) })
        data.append(headerBytes)
        for row in 0..<rows {
            for column in 0..<dimension {
                var code = UInt8(truncatingIfNeeded: row * 67 + column * 13)
                if code & 0x7f == 0x7f { code = 0x7e }
                if invalidRows[row] == column { code = row.isMultiple(of: 2) ? 0x7f : 0xff }
                data.append(code)
            }
        }
        original = data
        try data.write(to: url)
        table = try NGramTable(url: url)
    }

    func truncate(afterRows: Int) throws {
        let writer = try FileHandle(forWritingTo: url)
        defer { try? writer.close() }
        try writer.truncate(atOffset: UInt64(table.dataOffset + afterRows * table.dimension))
    }

    func restoreFile() throws {
        // Modify the same inode: the reader intentionally retains its open fd.
        let writer = try FileHandle(forWritingTo: url)
        defer { try? writer.close() }
        try writer.seek(toOffset: 0)
        try writer.write(contentsOf: original)
        try writer.truncate(atOffset: UInt64(original.count))
    }
}

@main private struct NGramBatchReadCheck {
    static func main() throws {
        guard CommandLine.arguments.count <= 2 else {
            throw CheckFailure(description: "usage: ngram-batch-check [report.json]")
        }
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("ngram-batch-check-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let valid = try Fixture(directory: directory, name: "valid.bin")
        let invalid = try Fixture(directory: directory, name: "invalid.bin", invalidRows: [17: 4, 443: 2])
        var results: [[String: Any]] = []
        func check(_ name: String, _ body: () throws -> Void) {
            do { try body(); results.append(["name": name, "passed": true]) }
            catch { results.append(["name": name, "passed": false, "error": String(describing: error)]) }
        }
        // Multiplication by 197 permutes all 512 rows. Extra entries include
        // distant duplicate positions, exercising reconstruction after dedup.
        let unordered = (0..<512).map { ($0 * 197) % 512 }
        let duplicateRows = unordered + [511, 0, 317, 0, 511] + Array(unordered.reversed())
        for rounded in [false, true] {
            // Each reference read is a tiny request through the unchanged S1
            // path, so neither partitioning nor dedup is repeated in the oracle.
            var reference: [[UInt32]] = []
            for row in 0..<512 {
                reference.append(try valid.table.readRows([row], roundToBFloat16: rounded).map(\.bitPattern))
            }
            check("unordered_unique_exact_rounded_\(rounded)") {
                let expected = unordered.flatMap { reference[$0] }
                let actual = try valid.table.readRows(unordered, roundToBFloat16: rounded).map(\.bitPattern)
                try require(actual == expected, "unordered rows differ from individual reads")
            }
            check("duplicates_exact_rounded_\(rounded)") {
                let expected = duplicateRows.flatMap { reference[$0] }
                let actual = try valid.table.readRows(duplicateRows, roundToBFloat16: rounded).map(\.bitPattern)
                try require(actual == expected, "duplicate/request ordering or float bit patterns changed")
            }
            check("one_unique_row_large_request_rounded_\(rounded)") {
                let actual = try valid.table.readRows(Array(repeating: 0, count: 1024),
                    roundToBFloat16: rounded).map(\.bitPattern)
                try require(actual == (0..<1024).flatMap { _ in reference[0] }, "deduplicated single row differs")
            }
            check("dispatch_boundary_rounded_\(rounded)") {
                for count in [0, 1, 16, 255, 256, 257] {
                    let requested = Array(unordered.prefix(count))
                    let actual = try valid.table.readRows(requested, roundToBFloat16: rounded).map(\.bitPattern)
                    try require(actual == requested.flatMap { reference[$0] }, "request of \(count) rows differs")
                }
            }
        }
        check("first_requested_nan_wins_across_workers") {
            let requested = [443] + unordered.filter { $0 != 443 }
            for _ in 0..<5 {
                try expect(.nonFiniteFP8(row: 443, column: 2)) { _ = try invalid.table.readRows(requested) }
            }
        }
        check("duplicate_nan_first_occurrence") {
            let requested = [17, 443, 17] + unordered
            try expect(.nonFiniteFP8(row: 17, column: 4)) { _ = try invalid.table.readRows(requested) }
        }
        check("all_ids_validated_before_nan_io") {
            try expect(.rowOutOfBounds(512)) { _ = try invalid.table.readRows([17] + unordered + [512, -1]) }
            try expect(.rowOutOfBounds(-1)) { _ = try invalid.table.readRows([443] + unordered + [-1, 512]) }
        }
        check("valid_rows_recover_after_failed_batch") {
            let requested = (0..<1024).map { $0.isMultiple(of: 2) ? 4 : 8 }
            let expected = try requested.flatMap { try invalid.table.readRows([$0]).map(\.bitPattern) }
            try require(try invalid.table.readRows(requested).map(\.bitPattern) == expected,
                        "error left reader unusable or returned a partial result")
        }
        check("eof_after_open_propagates_and_recovers") {
            let truncated = try Fixture(directory: directory, name: "truncated.bin")
            try truncated.truncate(afterRows: 256)
            try expect(.invalidFileLength) { _ = try truncated.table.readRows(unordered) }
            try require(try truncated.table.readRows([]).isEmpty, "empty read should not touch truncated file")
            try truncated.restoreFile()
            let actual = try truncated.table.readRows(duplicateRows).map(\.bitPattern)
            let expected = try valid.table.readRows(duplicateRows).map(\.bitPattern)
            try require(actual == expected, "reader failed after restoring truncated inode")
        }
        check("nan_before_eof_preserves_request_error_order") {
            let both = try Fixture(directory: directory, name: "nan-and-eof.bin", invalidRows: [17: 4])
            try both.truncate(afterRows: 256)
            try expect(.nonFiniteFP8(row: 17, column: 4)) {
                _ = try both.table.readRows([17, 443] + unordered)
            }
            try expect(.invalidFileLength) {
                _ = try both.table.readRows([443, 17] + unordered)
            }
        }
        let passed = results.allSatisfy { $0["passed"] as? Bool == true }
        let report: [String: Any] = ["passed": passed, "case_count": results.count, "cases": results,
            "fixture_rows": 512, "fixture_dimension": 160,
            "scope": "Tiny temporary files only; exact bit-pattern comparisons and ordered error propagation. No production-table I/O, CoreAI, MLX or GPU execution; no performance acceptance."]
        let output = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        if CommandLine.arguments.count == 2 {
            try output.write(to: URL(fileURLWithPath: CommandLine.arguments[1]))
        }
        print(String(decoding: output, as: UTF8.self))
        if !passed { throw CheckFailure(description: "NGram batch reader checks failed") }
    }
}
