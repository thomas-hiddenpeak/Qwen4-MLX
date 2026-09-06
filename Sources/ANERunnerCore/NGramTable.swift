// Ported from garnermccloud/mlx-serve, src/qwen4_exp.zig.
// Source: https://github.com/garnermccloud/mlx-serve/blob/7dbcba04c98e4fd3bcc533c63e645547f13cc3b1/src/qwen4_exp.zig
// MIT License, Copyright (c) 2026 David Dalcu.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

import Foundation
import CoreFoundation
import Darwin

public enum NGramTableError: Error, Equatable, Sendable {
    case invalidFormat(String)
    case invalidFileLength
    case fileIO(operation: String, code: Int32)
    case rowOutOfBounds(Int)
    case nonFiniteFP8(row: Int, column: Int)
    case sizeOverflow
}

/// Read-only, demand-read FP8 table. Initialization reads at most a 1 MiB
/// header. `readRows` uses pread, so it never loads or copies the full table.
/// Calls preserve requested order and duplicates and can safely read in parallel.
public final class NGramTable: Sendable {
    public let rowCount: Int
    public let dimension: Int
    public let scale: Float
    /// Byte offset of the first table element, including the safetensors header.
    public let dataOffset: Int
    private let descriptor: Int32

    public init(url: URL) throws {
        guard url.isFileURL else { throw NGramTableError.invalidFormat("Expected a file URL") }
        let descriptor = url.withUnsafeFileSystemRepresentation { path in
            path.map { Darwin.open($0, O_RDONLY | O_CLOEXEC) } ?? -1
        }
        guard descriptor >= 0 else { throw NGramTableError.fileIO(operation: "open", code: errno) }
        var keepOpen = false
        defer { if !keepOpen { _ = Darwin.close(descriptor) } }
        var info = stat()
        guard fstat(descriptor, &info) == 0 else { throw NGramTableError.fileIO(operation: "fstat", code: errno) }
        guard info.st_mode & S_IFMT == S_IFREG, info.st_size >= 8,
              info.st_size <= Int.max else { throw NGramTableError.invalidFileLength }
        let prefix = try Self.readExactly(descriptor, offset: 0, count: 8)
        let headerLength = prefix.enumerated().reduce(UInt64(0)) { $0 | (UInt64($1.element) << (8 * $1.offset)) }
        guard headerLength > 0, headerLength <= 1_048_576,
              headerLength <= UInt64(info.st_size - 8) else { throw NGramTableError.invalidFileLength }
        let bytes = try Self.readExactly(descriptor, offset: 8, count: Int(headerLength))
        let object: Any
        do { object = try JSONSerialization.jsonObject(with: Data(bytes)) }
        catch { throw NGramTableError.invalidFormat("Invalid JSON header") }
        guard let header = object as? [String: Any], Set(header.keys) == ["__metadata__", "weight"],
              let metadata = header["__metadata__"] as? [String: Any],
              metadata["format"] as? String == "mlx-serve-ngram-fp8",
              let scaleText = metadata["scale"] as? String,
              let scale = Float(scaleText), scale.isFinite, scale > 0,
              let weight = header["weight"] as? [String: Any],
              weight["dtype"] as? String == "F8_E4M3",
              let shape = weight["shape"] as? [Any], shape.count == 2,
              let offsets = weight["data_offsets"] as? [Any], offsets.count == 2,
              let rows = Self.integer(shape[0]), rows > 0,
              let dimension = Self.integer(shape[1]), dimension > 0,
              let begin = Self.integer(offsets[0]), begin == 0,
              let end = Self.integer(offsets[1]), end >= 0 else {
            throw NGramTableError.invalidFormat("Expected one F8_E4M3 weight tensor, integer shape/offsets, and positive finite string scale")
        }
        let (payloadBytes, payloadOverflow) = rows.multipliedReportingOverflow(by: dimension)
        let dataOffset = 8 + Int(headerLength)
        let (expectedFileLength, fileOverflow) = dataOffset.addingReportingOverflow(end)
        // Single-tensor safetensors have neither gaps nor trailing data. Reject
        // truncation, overlaps, and a shape that disagrees with the payload.
        guard !payloadOverflow, !fileOverflow, payloadBytes == end,
              expectedFileLength == info.st_size else { throw NGramTableError.invalidFileLength }
        self.rowCount = rows
        self.dimension = dimension
        self.scale = scale
        self.dataOffset = dataOffset
        self.descriptor = descriptor
        keepOpen = true
    }

    deinit { _ = Darwin.close(descriptor) }

    public func readRows(_ rowIDs: [Int], roundToBFloat16: Bool = true) throws -> [Float] {
        let (outputCount, overflow) = rowIDs.count.multipliedReportingOverflow(by: dimension)
        guard !overflow, outputCount <= Int.max / MemoryLayout<Float>.stride else {
            throw NGramTableError.sizeOverflow
        }
        for row in rowIDs where row < 0 || row >= rowCount { throw NGramTableError.rowOutOfBounds(row) }
        var output: [Float] = []
        output.reserveCapacity(outputCount)
        for row in rowIDs {
            let bytes = try Self.readExactly(descriptor, offset: dataOffset + row * dimension, count: dimension)
            for (column, code) in bytes.enumerated() {
                guard code & 0x7f != 0x7f else { throw NGramTableError.nonFiniteFP8(row: row, column: column) }
                let scaled = Self.decodeFP8(code) * scale
                output.append(roundToBFloat16 ? Self.roundBFloat16(scaled) : scaled)
            }
        }
        return output
    }

    // Internal numeric primitives remain available to @testable import.
    static func decodeFP8(_ code: UInt8) -> Float {
        let mantissa = Int(code & 7)
        let exponent = Int((code >> 3) & 15)
        let magnitude: Float
        if exponent == 0 { magnitude = Float(mantissa) / 512 }
        else if exponent == 15 && mantissa == 7 { magnitude = Float(bitPattern: 0x7fc00000) }
        else {
            let powers: [Float] = [0, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256]
            magnitude = (1 + Float(mantissa) * 0.125) * powers[exponent]
        }
        return code & 0x80 == 0 ? magnitude : -magnitude
    }

    static func roundBFloat16(_ value: Float) -> Float {
        let bits = value.bitPattern
        if bits & 0x7f800000 == 0x7f800000 {
            let top = bits >> 16
            let rounded = bits & 0x007fffff == 0 ? top : top | 0x0040
            return Float(bitPattern: rounded << 16)
        }
        let rounded = bits &+ 0x7fff &+ ((bits >> 16) & 1)
        return Float(bitPattern: rounded & 0xffff0000)
    }

    private static func integer(_ value: Any) -> Int? {
        guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID() else { return nil }
        // Reject nonintegral numbers and values that overflow Int.
        let type = String(cString: number.objCType)
        guard ["c", "s", "i", "l", "q", "C", "S", "I", "L", "Q"].contains(type) else { return nil }
        return Int(number.stringValue)
    }

    private static func readExactly(_ descriptor: Int32, offset: Int, count: Int) throws -> [UInt8] {
        var bytes = [UInt8](repeating: 0, count: count)
        try bytes.withUnsafeMutableBytes { buffer in
            var completed = 0
            while completed < count {
                let amount = Darwin.pread(descriptor, buffer.baseAddress!.advanced(by: completed), count - completed, off_t(offset + completed))
                if amount < 0 {
                    if errno == EINTR { continue }
                    throw NGramTableError.fileIO(operation: "pread", code: errno)
                }
                guard amount > 0 else { throw NGramTableError.invalidFileLength }
                completed += amount
            }
        }
        return bytes
    }
}
