import Foundation
import XCTest
@testable import ANERunnerCore

final class NGramTests: XCTestCase {
    // Golden values from src/qwen4_exp.zig in garnermccloud's mlx-serve fork,
    // commit 7dbcba04c98e4fd3bcc533c63e645547f13cc3b1. Upstream obtained these
    // independently from modeling_qwen4_exp.py. License notice: NGramHash.swift.
    private let referenceRows = [
        15389869, 39778609, 55713969, 62213332, 88817728, 118483999, 133731511, 155458159, 179763390, 197956758, 205378969, 220499474, 242466248, 265658744, 293662119, 315720898,
        12441580, 26378836, 53347667, 75104214, 99467174, 114254887, 126436461, 156012011, 169119442, 187827161, 214803956, 239809754, 242938905, 266427765, 294337448, 314484167,
        10204458, 27984170, 41283776, 68842151, 85621153, 118821647, 129504214, 158727320, 176298516, 181690702, 206665473, 238343128, 252151767, 267018740, 285543023, 319927855,
        18043673, 37626835, 51159316, 78294604, 94015356, 106720349, 136526052, 144330141, 176817901, 186368539, 203707490, 230017629, 247662678, 266533413, 293096193, 307951937,
        10041117, 28960672, 48420531, 71664411, 83016360, 106800418, 122476460, 150044571, 163654473, 184259024, 206781966, 224776026, 248853488, 273290488, 294849492, 303242927
    ]

    func testHashConfigurationMatchesUpstreamGoldenValues() throws {
        let hash = try NGramHash()
        XCTAssertEqual(hash.multipliers, [23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071])
        XCTAssertEqual(hash.vocabularies.first, 20_000_003)
        XCTAssertEqual(hash.vocabularies.last, 20_000_171)
        XCTAssertEqual(hash.offsets.last, 300_001_275)
        XCTAssertEqual(hash.totalRows, 320_001_536)
        XCTAssertEqual(hash.headCount, 16)
    }

    func testHashEOSAndChunkBoundariesMatchAllGoldenRows() throws {
        let hash = try NGramHash()
        let tokens: [UInt32] = [5, 7, 248044, 9, 11]
        XCTAssertEqual(try hash.rowIDs(previousTokens: hash.initialHistory, tokens: tokens), referenceRows)
        // Every possible split covers both sides of EOS and one-token chunks.
        for split in 0...tokens.count {
            let left = Array(tokens.prefix(split))
            let right = Array(tokens.dropFirst(split))
            let history = try hash.history(after: left, previousTokens: hash.initialHistory)
            let rows = try hash.rowIDs(previousTokens: hash.initialHistory, tokens: left)
                + hash.rowIDs(previousTokens: history, tokens: right)
            XCTAssertEqual(rows, referenceRows, "split=\(split)")
        }
        let fresh = try hash.rowIDs(previousTokens: hash.initialHistory, tokens: [9, 11])
        XCTAssertEqual(fresh, Array(referenceRows.suffix(32)))
    }

    func testWrappingSeedAndSignedModulusGoldenValues() throws {
        // Independently generated with Python integer arithmetic, explicit
        // modulo 2^64, signed reinterpretation, then nonnegative modulus.
        let hash = try NGramHash(unigramVocabularySize: 1, headsPerNGram: 2,
                                vocabularyBase: 17, vocabularyDivisor: 8,
                                seed: UInt64.max - 3, pleLayerIndex: 1,
                                eosTokenID: UInt32.max)
        XCTAssertEqual(hash.multipliers, [5_303_008_705_920_180_149, 3_888_210_797_208_733_891, 636_132_078_218_310_635])
        XCTAssertEqual(hash.vocabularies, [31, 37, 41, 43])
        XCTAssertEqual(hash.totalRows, 152)
        XCTAssertEqual(try hash.rowIDs(previousTokens: hash.initialHistory,
                                      tokens: [UInt32.max - 1, 2, UInt32.max, UInt32.max - 2]),
                       [17, 57, 106, 114, 25, 55, 108, 140, 0, 36, 99, 135, 11, 43, 75, 136])
    }

    func testInvalidHashInputsThrow() throws {
        XCTAssertThrowsError(try NGramHash(ngramSize: 1))
        XCTAssertThrowsError(try NGramHash(headsPerNGram: 17))
        XCTAssertThrowsError(try NGramHash(vocabularyBase: 0))
        XCTAssertThrowsError(try NGramHash(vocabularyDivisor: 0))
        XCTAssertThrowsError(try NGramHash(pleLayerIndex: UInt32.max))
        let hash = try NGramHash()
        XCTAssertThrowsError(try hash.rowIDs(previousTokens: [], tokens: [5]))
        XCTAssertThrowsError(try hash.history(after: [5], previousTokens: []))
        XCTAssertEqual(try hash.rowIDs(previousTokens: hash.initialHistory, tokens: []), [])
    }

    func testTableReadsRowsInRequestedOrderWithDuplicatesAndSignedZero() throws {
        try withFile(payload: [0x00, 0x38, 0x40, 0xc4, 0x01, 0x7e, 0xfe, 0x80]) { url in
            let table = try NGramTable(url: url)
            XCTAssertEqual(table.rowCount, 2)
            XCTAssertEqual(table.dimension, 4)
            XCTAssertEqual(table.scale, 0.5)
            XCTAssertEqual(table.dataOffset, 264)
            let expected: [Float] = [1 / 1024, 224, -224, -0.0, 0, 0.5, 1, -1.5, 1 / 1024, 224, -224, -0.0]
            XCTAssertEqual(try table.readRows([1, 0, 1]).map(\.bitPattern), expected.map(\.bitPattern))
            XCTAssertEqual(try table.readRows([]), [])
            XCTAssertThrowsError(try table.readRows([-1]))
            XCTAssertThrowsError(try table.readRows([2]))
        }
    }

    func testAllFP8CodesAgainstExponentFormula() throws {
        for integer in 0...255 {
            let code = UInt8(integer)
            let decoded = NGramTable.decodeFP8(code)
            let exponent = (integer >> 3) & 15
            let mantissa = integer & 7
            if exponent == 15 && mantissa == 7 {
                XCTAssertTrue(decoded.isNaN)
                continue
            }
            let magnitude = exponent == 0
                ? Double(mantissa) * pow(2.0, -9)
                : Double(8 + mantissa) * pow(2.0, Double(exponent - 10))
            let expected = Float(integer & 128 == 0 ? magnitude : -magnitude)
            XCTAssertEqual(decoded.bitPattern, expected.bitPattern, "FP8 code \(integer)")
        }
    }

    func testBF16TiesSubnormalInfinityAndNaN() {
        let cases: [(UInt32, UInt32)] = [
            (0x3f808000, 0x3f800000), // halfway, even lower BF16
            (0x3f818000, 0x3f820000), // halfway, even upper BF16
            (0xbf808000, 0xbf800000),
            (0xbf818000, 0xbf820000),
            (0x00008000, 0x00000000),
            (0x00008001, 0x00010000),
            (0x80000000, 0x80000000),
            (0x7f800000, 0x7f800000),
            (0xff800000, 0xff800000),
            (0x7f800001, 0x7fc00000)
        ]
        for (input, expected) in cases {
            XCTAssertEqual(NGramTable.roundBFloat16(Float(bitPattern: input)).bitPattern, expected)
        }
    }

    func testTableAppliesScaleBeforeOptionalBF16Rounding() throws {
        let header = makeHeader(rows: 1, dimension: 1, scale: "1.00390625")
        try withFile(header: header, payload: [0x38]) { url in
            let table = try NGramTable(url: url)
            XCTAssertEqual(try table.readRows([0], roundToBFloat16: false), [1.00390625])
            XCTAssertEqual(try table.readRows([0]), [1.0])
        }
    }

    func testBothFP8NaNCodesAreRejectedOnlyWhenRead() throws {
        try withFile(header: makeHeader(rows: 3, dimension: 1), payload: [0x38, 0x7f, 0xff]) { url in
            let table = try NGramTable(url: url)
            XCTAssertEqual(try table.readRows([0]), [0.5])
            XCTAssertThrowsError(try table.readRows([1]))
            XCTAssertThrowsError(try table.readRows([2], roundToBFloat16: false))
        }
    }

    func testMalformedHeadersAreRejected() throws {
        let valid = makeHeader()
        let invalid = [
            valid.replacingOccurrences(of: "F8_E4M3", with: "BF16"),
            valid.replacingOccurrences(of: "mlx-serve-ngram-fp8", with: "unknown"),
            valid.replacingOccurrences(of: "\"0.5\"", with: "\"nan\""),
            valid.replacingOccurrences(of: "\"0.5\"", with: "\"0\""),
            valid.replacingOccurrences(of: "[2,4]", with: "[true,4]"),
            valid.replacingOccurrences(of: "[2,4]", with: "[2.5,4]"),
            valid.replacingOccurrences(of: "[2,4]", with: "[0,4]"),
            valid.replacingOccurrences(of: "[2,4]", with: "[9223372036854775807,4]"),
            valid.replacingOccurrences(of: "[0,8]", with: "[1,9]"),
            valid.replacingOccurrences(of: "[0,8]", with: "[0,7]")
        ]
        for header in invalid {
            try withFile(header: header, payload: Array(repeating: 0, count: 8)) { url in
                XCTAssertThrowsError(try NGramTable(url: url), header)
            }
        }
    }

    func testTruncatedAndTrailingPayloadsAreRejected() throws {
        for count in [0, 7, 9] {
            try withFile(payload: Array(repeating: 0, count: count)) { url in
                XCTAssertThrowsError(try NGramTable(url: url))
            }
        }
    }

    func testTruncationAfterOpenIsReportedByPread() throws {
        try withFile(payload: Array(repeating: 0, count: 8)) { url in
            let table = try NGramTable(url: url)
            let writer = try FileHandle(forWritingTo: url)
            defer { try? writer.close() }
            try writer.truncate(atOffset: UInt64(table.dataOffset + 4))
            XCTAssertThrowsError(try table.readRows([1]))
        }
    }

    private func makeHeader(rows: Int = 2, dimension: Int = 4, scale: String = "0.5") -> String {
        "{\"__metadata__\":{\"format\":\"mlx-serve-ngram-fp8\",\"scale\":\"\(scale)\"},\"weight\":{\"dtype\":\"F8_E4M3\",\"shape\":[\(rows),\(dimension)],\"data_offsets\":[0,\(rows * dimension)]}}"
    }

    private func withFile<T>(header: String? = nil, payload: [UInt8], _ body: (URL) throws -> T) throws -> T {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("ane-ngram-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("table.bin")
        var headerBytes = Array((header ?? makeHeader()).utf8)
        let headerLength = max(256, (headerBytes.count + 7) / 8 * 8)
        headerBytes += Array(repeating: UInt8(32), count: headerLength - headerBytes.count)
        var data = Data((0..<8).map { UInt8(truncatingIfNeeded: UInt64(headerLength) >> (8 * $0)) })
        data.append(contentsOf: headerBytes)
        data.append(contentsOf: payload)
        try data.write(to: url)
        return try body(url)
    }
}
