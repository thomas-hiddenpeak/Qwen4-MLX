import Foundation

/// Decodes byte fragments without repairing a scalar merely because a token
/// or transport chunk ended in its middle. Holds at most three pending bytes.
///
/// Concatenating append results and finish() equals Swift's
/// String(decoding: allBytes, as: UTF8.self), including invalid-byte repair.
/// Output boundaries are Unicode scalar boundaries, not grapheme boundaries.
/// Each request owns its value; no tokenizer, model, or network state is kept.
public struct IncrementalUTF8Decoder: Sendable {
    private var pending: [UInt8] = []

    public init() {}

    /// Bytes that can still become one valid UTF-8 scalar; always in 0...3.
    public var pendingByteCount: Int { pending.count }

    /// Returns all text whose decoding cannot change with later bytes.
    /// Empty input returns an empty string and preserves any pending suffix.
    public mutating func append(_ bytes: [UInt8]) -> String {
        appendBytes(bytes)
    }

    public mutating func append(_ bytes: Data) -> String {
        appendBytes(bytes)
    }

    /// Repairs any truncated final scalar using Swift's native decoder and
    /// clears pending bytes. Repeated finish calls return an empty string.
    /// Appending afterwards begins a new independent byte segment; a scalar
    /// cannot be continued across finish(), even if later bytes would match.
    public mutating func finish() -> String {
        let text = String(decoding: pending, as: UTF8.self)
        pending.removeAll(keepingCapacity: false)
        return text
    }

    private mutating func appendBytes<Bytes: BidirectionalCollection>(_ bytes: Bytes) -> String
        where Bytes.Element == UInt8 {
        guard !bytes.isEmpty else { return "" }
        if pending.isEmpty { return consume(bytes) }
        var joined = pending
        joined.append(contentsOf: bytes)
        return consume(joined)
    }

    private mutating func consume<Bytes: BidirectionalCollection>(_ bytes: Bytes) -> String
        where Bytes.Element == UInt8 {
        let count = Self.incompleteSuffixLength(bytes)
        let boundary = bytes.index(bytes.endIndex, offsetBy: -count)
        let text = String(decoding: bytes[..<boundary], as: UTF8.self)
        // Copy into a small fresh array rather than retain a slice's storage.
        pending.removeAll(keepingCapacity: false)
        if count > 0 {
            pending.reserveCapacity(3)
            pending.append(contentsOf: bytes[boundary...])
        }
        return text
    }

    /// UTF-8 byte ranges follow RFC 3629 section 4. Only a prefix that could
    /// still become a valid scalar may cross calls. In particular E0/ED/F0/F4
    /// constrain the second byte: overlong encodings, surrogates and values
    /// above U+10FFFF are already invalid and must be repaired immediately.
    /// https://www.rfc-editor.org/rfc/rfc3629#section-4
    private static func incompleteSuffixLength<Bytes: BidirectionalCollection>(_ bytes: Bytes) -> Int
        where Bytes.Element == UInt8 {
        var start = bytes.endIndex
        var length = 0
        while start != bytes.startIndex && length < 3 {
            start = bytes.index(before: start)
            length += 1
            let lead = bytes[start]
            let required: Int
            switch lead {
            case 0xC2...0xDF: required = 2
            case 0xE0...0xEF: required = 3
            case 0xF0...0xF4: required = 4
            default: continue
            }
            guard length < required else { continue }
            if length >= 2 {
                let second = bytes[bytes.index(after: start)]
                let allowed: ClosedRange<UInt8>
                switch lead {
                case 0xE0: allowed = 0xA0...0xBF
                case 0xED: allowed = 0x80...0x9F
                case 0xF0: allowed = 0x90...0xBF
                case 0xF4: allowed = 0x80...0x8F
                default: allowed = 0x80...0xBF
                }
                guard allowed.contains(second) else { continue }
            }
            if length == 3 {
                let third = bytes[bytes.index(start, offsetBy: 2)]
                guard (0x80...0xBF).contains(third) else { continue }
            }
            return length
        }
        return 0
    }
}
