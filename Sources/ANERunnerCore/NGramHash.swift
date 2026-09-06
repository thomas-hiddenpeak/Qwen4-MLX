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

public enum NGramHashError: Error, Equatable, Sendable {
    case invalidConfiguration(String)
    case invalidHistoryLength(expected: Int, actual: Int)
    case sizeOverflow
}

/// Qwen4's CPU hash. `pleLayerIndex` is the PLE table ordinal (zero for the
/// downloaded model), not the decoder-layer index where the PLE is inserted.
public struct NGramHash: Sendable {
    public let ngramSize: Int
    public let headsPerNGram: Int
    public let headCount: Int
    public let eosTokenID: UInt32
    public let multipliers: [Int64]
    public let vocabularies: [Int64]
    public let offsets: [Int64]
    public let totalRows: Int

    public var initialHistory: [UInt32] {
        Array(repeating: eosTokenID, count: ngramSize - 1)
    }

    public init(
        unigramVocabularySize: UInt32 = 248_320,
        ngramSize: Int = 3,
        headsPerNGram: Int = 8,
        vocabularyBase: UInt64 = 20_000_000,
        vocabularyDivisor: UInt64 = 128,
        seed: UInt64 = 1234,
        pleLayerIndex: UInt32 = 0,
        eosTokenID: UInt32 = 248_044
    ) throws {
        guard (2...8).contains(ngramSize), headsPerNGram > 0,
              headsPerNGram <= 32 / (ngramSize - 1), vocabularyBase > 0,
              vocabularyBase <= UInt64(UInt32.max), vocabularyDivisor > 0 else {
            throw NGramHashError.invalidConfiguration("Require 2...8 n-grams, at most 32 heads, a positive UInt32 vocabulary base, and a positive divisor")
        }
        let heads = (ngramSize - 1) * headsPerNGram
        let firstPrimeRank = UInt64(pleLayerIndex) * UInt64(heads)
        // This CPU reference implementation uses trial division, as upstream
        // does. Bound configuration work; Qwen's real configuration needs 16.
        guard firstPrimeRank + UInt64(heads) <= 4096 else {
            throw NGramHashError.invalidConfiguration("At most 4096 prime ranks are supported")
        }
        self.ngramSize = ngramSize
        self.headsPerNGram = headsPerNGram
        self.headCount = heads
        self.eosTokenID = eosTokenID
        let halfBound = max(UInt64(1), UInt64(Int64.max) / UInt64(max(unigramVocabularySize, 1)) / 2)
        let baseSeed = seed &+ (10_007 &* UInt64(pleLayerIndex))
        self.multipliers = (0..<ngramSize).map { index in
            let input = baseSeed &+ (0x9E3779B97F4A7C15 &* UInt64(index + 1))
            return Int64(2 * (Self.splitMix64(input) % halfBound) + 1)
        }
        var vocabularies: [Int64] = []
        var offsets: [Int64] = []
        var prime = vocabularyBase - 1
        var total: UInt64 = 0
        for rank in 0..<(Int(firstPrimeRank) + heads) {
            repeat { prime += 1 } while !Self.isPrime(prime)
            if rank < firstPrimeRank { continue }
            guard prime <= UInt64(Int64.max), total <= UInt64(Int64.max) - prime else {
                throw NGramHashError.sizeOverflow
            }
            offsets.append(Int64(total))
            vocabularies.append(Int64(prime))
            total += prime
        }
        let remainder = total % vocabularyDivisor
        if remainder != 0 {
            let increment = vocabularyDivisor - remainder
            guard increment <= UInt64(Int.max) - total else { throw NGramHashError.sizeOverflow }
            total += increment
        }
        guard total <= UInt64(Int.max) else { throw NGramHashError.sizeOverflow }
        self.vocabularies = vocabularies
        self.offsets = offsets
        self.totalRows = Int(total)
    }

    /// Returns `[tokens.count, headCount]` as a flat row-major array. History
    /// must contain exactly `ngramSize - 1` preceding tokens, oldest first.
    /// The EOS token itself still sees its left context; tokens AFTER it reset.
    public func rowIDs(previousTokens: [UInt32], tokens: [UInt32]) throws -> [Int] {
        let context = ngramSize - 1
        guard previousTokens.count == context else {
            throw NGramHashError.invalidHistoryLength(expected: context, actual: previousTokens.count)
        }
        let (count, overflow) = tokens.count.multipliedReportingOverflow(by: headCount)
        guard !overflow, tokens.count <= Int.max - context else { throw NGramHashError.sizeOverflow }
        var output = [Int](repeating: 0, count: count)
        var lastEOS = -1
        func token(at index: Int) -> UInt32 {
            index < context ? previousTokens[index] : tokens[index - context]
        }
        for position in 0..<(context + tokens.count) {
            let current = token(at: position)
            if position >= context {
                let segmentPosition = position - (lastEOS + 1)
                var mixed = Int64(current) &* multipliers[0]
                for size in 2...ngramSize {
                    let shift = size - 1
                    let shifted = segmentPosition >= shift && position >= shift
                        ? token(at: position - shift) : eosTokenID
                    mixed ^= Int64(shifted) &* multipliers[shift]
                    let firstHead = (size - 2) * headsPerNGram
                    for head in firstHead..<(firstHead + headsPerNGram) {
                        // Swift's % is a signed remainder; Zig @mod is a
                        // nonnegative modulus for positive vocabulary sizes.
                        let remainder = mixed % vocabularies[head]
                        let modulus = remainder < 0 ? remainder + vocabularies[head] : remainder
                        output[(position - context) * headCount + head] = Int(modulus + offsets[head])
                    }
                }
            }
            if current == eosTokenID { lastEOS = position }
        }
        return output
    }

    /// History to carry into the next chunk; EOS reset is applied by rowIDs.
    public func history(after tokens: [UInt32], previousTokens: [UInt32]) throws -> [UInt32] {
        guard previousTokens.count == ngramSize - 1 else {
            throw NGramHashError.invalidHistoryLength(expected: ngramSize - 1, actual: previousTokens.count)
        }
        return Array((previousTokens + tokens).suffix(ngramSize - 1))
    }

    private static func splitMix64(_ input: UInt64) -> UInt64 {
        var value = input &+ 0x9E3779B97F4A7C15
        value = (value ^ (value >> 30)) &* 0xBF58476D1CE4E5B9
        value = (value ^ (value >> 27)) &* 0x94D049BB133111EB
        return value ^ (value >> 31)
    }

    private static func isPrime(_ value: UInt64) -> Bool {
        if value < 2 { return false }
        if value % 2 == 0 { return value == 2 }
        var divisor: UInt64 = 3
        while divisor <= value / divisor {
            if value % divisor == 0 { return false }
            divisor += 2
        }
        return true
    }
}
