import Foundation
import XCTest
@testable import ANERunnerCore

final class IncrementalUTF8DecoderTests: XCTestCase {
    func testChineseAndEmojiAreEmittedOnlyWhenTheirScalarCompletes() {
        var decoder = IncrementalUTF8Decoder()
        XCTAssertEqual(decoder.append([0x41, 0xE4]), "A")
        XCTAssertEqual(decoder.pendingByteCount, 1)
        XCTAssertEqual(decoder.append(Data([0xB8])), "")
        XCTAssertEqual(decoder.pendingByteCount, 2)
        XCTAssertEqual(decoder.append([0xAD, 0xF0]), "中")
        XCTAssertEqual(decoder.append([0x9F, 0x98]), "")
        XCTAssertEqual(decoder.pendingByteCount, 3)
        XCTAssertEqual(decoder.append(Data([0x80, 0x21])), "😀!")
        XCTAssertEqual(decoder.pendingByteCount, 0)
        XCTAssertEqual(decoder.finish(), "")
    }

    func testEmptyInputAndRepeatedFinishHaveAnExplicitResetContract() {
        var decoder = IncrementalUTF8Decoder()
        XCTAssertEqual(decoder.append([UInt8]()), "")
        XCTAssertEqual(decoder.append(Data()), "")
        XCTAssertEqual(decoder.finish(), "")
        XCTAssertEqual(decoder.append([0xF0, 0x9F, 0x98]), "")
        XCTAssertEqual(decoder.append(Data()), "")
        XCTAssertEqual(decoder.pendingByteCount, 3)
        XCTAssertEqual(decoder.finish(), "\u{FFFD}")
        XCTAssertEqual(decoder.pendingByteCount, 0)
        XCTAssertEqual(decoder.finish(), "")
        // finish is a segment boundary, so this cannot complete the old emoji.
        XCTAssertEqual(decoder.append([0x80, 0x41]), "\u{FFFD}A")
        XCTAssertEqual(decoder.append(Array("好".utf8)), "好")
        XCTAssertEqual(decoder.finish(), "")
    }

    func testInvalidPrefixesAreNotBufferedAsPotentialScalars() {
        let invalid: [[UInt8]] = [
            [0x80], [0xBF], [0xC0], [0xC1], [0xF5], [0xFF],
            [0xE0, 0x9F], [0xED, 0xA0], [0xF0, 0x8F], [0xF4, 0x90],
            [0xE1, 0x80, 0x41], [0xF1, 0x80, 0x7F],
        ]
        for bytes in invalid {
            var decoder = IncrementalUTF8Decoder()
            let text = decoder.append(bytes)
            XCTAssertEqual(Array(text.utf8), Array(String(decoding: bytes, as: UTF8.self).utf8))
            XCTAssertEqual(decoder.pendingByteCount, 0, "Invalid suffix: \(bytes)")
            XCTAssertEqual(decoder.finish(), "")
        }
    }

    func testAllUTF8ScalarRangeEdgesAndTheirTruncatedPrefixes() {
        // ASCII/NUL, two-byte, overlong/surrogate boundaries, supplementary
        // plane edges and the highest Unicode scalar. Swift is the oracle.
        let scalars: [UInt32] = [0, 0x7F, 0x80, 0x7FF, 0x800, 0xD7FF,
                                 0xE000, 0xFFFF, 0x10000, 0x3FFFF, 0x40000, 0x10FFFF]
        for value in scalars {
            let bytes = Array(String(Unicode.Scalar(value)!).utf8)
            for length in 0...bytes.count {
                let prefix = Array(bytes.prefix(length))
                assertAllPartitions(prefix)
            }
        }
    }

    func testEveryPartitionOfMixedValidInvalidAndTruncatedSequences() {
        let cases: [[UInt8]] = [
            [0x00, 0x41, 0xC2, 0xA2, 0xE4, 0xB8, 0xAD],
            [0xF0, 0x9F, 0x98, 0x80, 0xF0, 0x9F, 0x98],
            [0xE1, 0x80, 0x41, 0xE2, 0x82, 0x42, 0xC2],
            [0xED, 0xA0, 0x80, 0xE0, 0x80, 0xAF],
            [0xF4, 0x90, 0x80, 0x80, 0xF0, 0x80, 0x80, 0xAF],
            [0xFF, 0xFE, 0xF5, 0x80, 0xBF, 0xC0, 0xAF],
            [0xE4, 0xB8, 0xC2, 0xA2, 0x80, 0xF1, 0x80],
        ]
        for bytes in cases { assertAllPartitions(bytes) }
    }

    func testEveryTwoByteInputSplitMatchesNativeSwiftRepair() {
        // Exhaustive short malformed cases catch error grouping across calls,
        // independently of our suffix classification logic.
        for first in 0...255 {
            for second in 0...255 {
                let bytes = [UInt8(first), UInt8(second)]
                var decoder = IncrementalUTF8Decoder()
                let text = decoder.append([bytes[0]]) + decoder.append([bytes[1]]) + decoder.finish()
                let expected = String(decoding: bytes, as: UTF8.self)
                if !text.utf8.elementsEqual(expected.utf8) {
                    XCTFail("Native repair mismatch for \(bytes)")
                    return
                }
                if decoder.pendingByteCount != 0 {
                    XCTFail("finish retained bytes for \(bytes)")
                    return
                }
            }
        }
    }

    func testDataSlicesAndEveryTwoCutSplitPreserveExactUTF8() {
        let text = "汉字👩‍💻 e\u{301} é\u{FEFF}\u{0}尾"
        let bytes = Array(text.utf8)
        // Data slices can retain nonzero indices; append must not assume zero.
        let padded = Data([0xFF] + bytes + [0xFF])
        let slice = padded.dropFirst().dropLast()
        var direct = IncrementalUTF8Decoder()
        XCTAssertEqual(Array((direct.append(slice) + direct.finish()).utf8), bytes)
        for first in 0...bytes.count {
            for second in first...bytes.count {
                let pieces = [Array(bytes[..<first]), Array(bytes[first..<second]), Array(bytes[second...])]
                assertDecodes(pieces, expected: bytes)
            }
        }
    }

    func testCopiedDecodersDoNotSharePendingState() {
        var original = IncrementalUTF8Decoder()
        XCTAssertEqual(original.append([0xE4, 0xB8]), "")
        var copy = original
        XCTAssertEqual(original.append([0xAD]), "中")
        XCTAssertEqual(copy.append([0x8A]), "上")
        XCTAssertEqual(original.finish(), "")
        XCTAssertEqual(copy.finish(), "")
    }

    private func assertAllPartitions(_ bytes: [UInt8], file: StaticString = #filePath, line: UInt = #line) {
        let expected = Array(String(decoding: bytes, as: UTF8.self).utf8)
        guard !bytes.isEmpty else {
            assertDecodes([[], []], expected: expected, file: file, line: line)
            return
        }
        for mask in 0..<(1 << (bytes.count - 1)) {
            var chunks: [[UInt8]] = [], start = 0
            for boundary in 1..<bytes.count where mask & (1 << (boundary - 1)) != 0 {
                chunks.append(Array(bytes[start..<boundary])); start = boundary
            }
            chunks.append(Array(bytes[start...]))
            assertDecodes(chunks, expected: expected, file: file, line: line)
        }
    }

    private func assertDecodes(_ chunks: [[UInt8]], expected: [UInt8],
                               file: StaticString = #filePath, line: UInt = #line) {
        var decoder = IncrementalUTF8Decoder(), text = ""
        for (index, chunk) in chunks.enumerated() {
            text += index.isMultiple(of: 2) ? decoder.append(chunk) : decoder.append(Data(chunk))
            XCTAssertLessThanOrEqual(decoder.pendingByteCount, 3, file: file, line: line)
            XCTAssertEqual(decoder.append(Data()), "", file: file, line: line)
        }
        text += decoder.finish()
        // String equality is canonically equivalent; UTF-8 equality additionally
        // ensures no normalization or combining scalar was silently changed.
        XCTAssertEqual(Array(text.utf8), expected, "Fragments: \(chunks)", file: file, line: line)
        XCTAssertEqual(decoder.pendingByteCount, 0, file: file, line: line)
        XCTAssertEqual(decoder.finish(), "", file: file, line: line)
    }
}
