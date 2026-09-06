import Foundation
import CoreFoundation

public struct ChatMessage: Codable {
    public let role: String
    public let content: String
    public let reasoningContent: String?
    public init(role: String, content: String, reasoningContent: String? = nil) {
        self.role = role; self.content = content; self.reasoningContent = reasoningContent
    }
}

/// Native byte-level BPE for the shipped Qwen tokenizer. No Python/runtime bridge.
public final class QwenTokenizer {
    public enum Normalization {
        /// The author runtime does not apply the tokenizer.json NFC normalizer.
        case authorCompatible
        case modelNFC
    }
    private struct Merge { let rank: Int; let token: Int32 }
    private struct Added { let text: String; let id: Int32; let special: Bool }
    private struct Candidate {
        let rank: Int
        let left: Int
        let right: Int
        let leftVersion: Int
        let rightVersion: Int
        let token: Int32
        func precedes(_ other: Candidate) -> Bool { rank < other.rank || (rank == other.rank && left < other.left) }
    }
    private let normalization: Normalization
    private let split: NSRegularExpression
    private let merges: [UInt64: Merge]
    private let added: [Added]
    private let specialIDs: Set<Int32>
    private let bytesToIDs: [Int32]
    private let tokenBytes: [Int32: [UInt8]]
    public let vocabularyCount: Int
    public let eosTokenIDs: Set<Int32>
    /// Author policy: special=true tokens, excluding EOS, template-emitted
    /// markers, and the permitted output-only <|constrain|> marker.
    public let reservedOutputTokenIDs: Set<Int32>

    public init(modelDirectory: URL, normalization: Normalization = .authorCompatible) throws {
        self.normalization = normalization
        guard let root = try JSONSerialization.jsonObject(with: Data(contentsOf: modelDirectory.appendingPathComponent("tokenizer.json"))) as? [String: Any],
              let model = root["model"] as? [String: Any], model["type"] as? String == "BPE",
              let rawVocab = model["vocab"] as? [String: Any], let rawMerges = model["merges"] as? [Any],
              let addedRaw = root["added_tokens"] as? [[String: Any]],
              let pre = root["pre_tokenizer"] as? [String: Any], pre["type"] as? String == "Sequence",
              let stages = pre["pretokenizers"] as? [[String: Any]], stages.count == 2,
              stages[0]["type"] as? String == "Split", stages[0]["behavior"] as? String == "Isolated",
              stages[0]["invert"] as? Bool == false,
              let pattern = stages[0]["pattern"] as? [String: String], let regex = pattern["Regex"],
              stages[1]["type"] as? String == "ByteLevel", stages[1]["add_prefix_space"] as? Bool == false,
              stages[1]["use_regex"] as? Bool == false,
              let normalizer = root["normalizer"] as? [String: Any], normalizer["type"] as? String == "NFC" else {
            throw GPUWeightError.invalid("Unsupported Qwen tokenizer structure")
        }
        guard model["dropout"] == nil || model["dropout"] is NSNull,
              model["byte_fallback"] as? Bool == false, model["ignore_merges"] as? Bool == false,
              model["continuing_subword_prefix"] as? String == "", model["end_of_word_suffix"] as? String == "" else {
            throw GPUWeightError.invalid("Unsupported BPE modifiers")
        }
        split = try NSRegularExpression(pattern: regex)
        var vocab: [String: Int32] = [:]
        var ids: Set<Int32> = []
        vocab.reserveCapacity(rawVocab.count + addedRaw.count)
        for (token, rawID) in rawVocab {
            let id = try Self.tokenID(rawID)
            guard ids.insert(id).inserted else { throw GPUWeightError.invalid("Duplicate BPE token ID") }
            vocab[token] = id
        }
        // GPT-2's byte-to-Unicode bijection, over Unicode SCALARS, not Swift
        // grapheme clusters (combining marks must remain separate byte symbols).
        let visible = Array(33...126) + Array(161...172) + Array(174...255)
        let visibleSet = Set(visible)
        var byteScalars = [UInt32](repeating: 0, count: 256)
        var next: UInt32 = 256
        for byte in 0..<256 {
            if visibleSet.contains(byte) { byteScalars[byte] = UInt32(byte) }
            else { byteScalars[byte] = next; next += 1 }
        }
        var inverse: [UInt32: UInt8] = [:]
        var byteIDs: [Int32] = []
        for (byte, scalar) in byteScalars.enumerated() {
            inverse[scalar] = UInt8(byte)
            guard let unicode = Unicode.Scalar(scalar), let id = vocab[String(unicode)] else { throw GPUWeightError.invalid("Incomplete BPE byte alphabet") }
            byteIDs.append(id)
        }
        bytesToIDs = byteIDs
        var allAdded: [Added] = []
        for entry in addedRaw {
            guard let token = entry["content"] as? String, !token.isEmpty,
                  let rawID = entry["id"], let special = entry["special"] as? Bool,
                  entry["lstrip"] as? Bool == false, entry["rstrip"] as? Bool == false,
                  entry["single_word"] as? Bool == false, entry["normalized"] as? Bool == false else {
                throw GPUWeightError.invalid("Unsupported added-token matching flags")
            }
            let id = try Self.tokenID(rawID)
            if let old = vocab[token], old != id { throw GPUWeightError.invalid("Conflicting added token") }
            if vocab[token] == nil && !ids.insert(id).inserted { throw GPUWeightError.invalid("Duplicate added-token ID") }
            vocab[token] = id
            allAdded.append(Added(text: token, id: id, special: special))
        }
        added = allAdded.sorted { $0.text.utf8.count > $1.text.utf8.count }
        specialIDs = Set(allAdded.filter(\.special).map(\.id))
        let stopIDs = Set(["<|im_end|>", "<|endoftext|>"].compactMap { vocab[$0] })
        eosTokenIDs = stopIDs
        let templateURL = modelDirectory.appendingPathComponent("chat_template.jinja")
        let templateSource: String
        if FileManager.default.fileExists(atPath: templateURL.path) {
            templateSource = try String(contentsOf: templateURL, encoding: .utf8)
        } else {
            let configURL = modelDirectory.appendingPathComponent("tokenizer_config.json")
            if FileManager.default.fileExists(atPath: configURL.path),
               let config = try JSONSerialization.jsonObject(with: Data(contentsOf: configURL)) as? [String: Any] {
                templateSource = config["chat_template"] as? String ?? ""
            } else { templateSource = "" }
        }
        reservedOutputTokenIDs = templateSource.isEmpty ? [] : Set(allAdded.filter {
            $0.special && !stopIDs.contains($0.id) && !templateSource.contains($0.text) && $0.text != "<|constrain|>"
        }.map(\.id))
        vocabularyCount = vocab.count
        var decoded: [Int32: [UInt8]] = [:]
        let addedIDs = Set(allAdded.map(\.id))
        for (token, id) in vocab {
            if addedIDs.contains(id) { decoded[id] = Array(token.utf8) }
            else {
                var bytes: [UInt8] = []
                for scalar in token.unicodeScalars {
                    guard let byte = inverse[scalar.value] else { throw GPUWeightError.invalid("Invalid byte-level BPE vocabulary symbol") }
                    bytes.append(byte)
                }
                decoded[id] = bytes
            }
        }
        tokenBytes = decoded
        var rules: [UInt64: Merge] = [:]
        rules.reserveCapacity(rawMerges.count)
        for (rank, raw) in rawMerges.enumerated() {
            let parts: [String]
            if let string = raw as? String { parts = string.split(separator: " ", maxSplits: 1, omittingEmptySubsequences: false).map(String.init) }
            else if let pair = raw as? [String] { parts = pair }
            else { throw GPUWeightError.invalid("Invalid BPE merge") }
            guard parts.count == 2, let left = vocab[parts[0]], let right = vocab[parts[1]], let result = vocab[parts[0] + parts[1]] else {
                throw GPUWeightError.invalid("BPE merge references missing vocabulary")
            }
            let key = Self.pair(left, right)
            guard rules[key] == nil else { throw GPUWeightError.invalid("Duplicate BPE merge pair") }
            rules[key] = Merge(rank: rank, token: result)
        }
        merges = rules
    }

    public func encode(_ text: String) throws -> [Int32] {
        var result: [Int32] = []
        var cursor = text.startIndex
        while cursor < text.endIndex {
            var best: (Range<String.Index>, Added)?
            for token in added {
                if let range = text.range(of: token.text, options: .literal, range: cursor..<text.endIndex),
                   best == nil || range.lowerBound < best!.0.lowerBound {
                    best = (range, token)
                }
            }
            if let (range, token) = best {
                result.append(contentsOf: try encodeOrdinary(String(text[cursor..<range.lowerBound])))
                result.append(token.id); cursor = range.upperBound
            } else {
                result.append(contentsOf: try encodeOrdinary(String(text[cursor...])))
                break
            }
        }
        return result
    }

    public func decodeBytes(_ ids: [Int32], skipSpecialTokens: Bool = false) throws -> [UInt8] {
        var bytes: [UInt8] = []
        for id in ids {
            if skipSpecialTokens && specialIDs.contains(id) { continue }
            guard let token = tokenBytes[id] else { throw GPUWeightError.invalid("Unknown token ID \(id)") }
            bytes.append(contentsOf: token)
        }
        return bytes
    }
    /// Decode a complete token sequence together; decoding individual tokens
    /// can split a UTF-8 scalar and introduce replacement characters.
    public func decode(_ ids: [Int32], skipSpecialTokens: Bool = false) throws -> String {
        String(decoding: try decodeBytes(ids, skipSpecialTokens: skipSpecialTokens), as: UTF8.self)
    }

    /// The shipped Jinja template's text-only, no-tools, thinking-disabled arm.
    /// Historical assistant reasoning is preserved as the template defaults to.
    public func renderChat(messages: [ChatMessage], addGenerationPrompt: Bool = true) throws -> String {
        guard !messages.isEmpty, messages.contains(where: { $0.role == "user" && !(Self.trim($0.content).hasPrefix("<tool_response>") && Self.trim($0.content).hasSuffix("</tool_response>")) }) else {
            throw GPUWeightError.invalid("Chat requires a user query")
        }
        var rendered = ""
        for (index, message) in messages.enumerated() {
            let content = Self.trim(message.content)
            switch message.role {
            case "system":
                guard index == 0 else { throw GPUWeightError.invalid("System message must be first") }
                if !content.isEmpty { rendered += "<|im_start|>system\n" + content + "<|im_end|>\n" }
            case "user": rendered += "<|im_start|>user\n" + content + "<|im_end|>\n"
            case "assistant":
                rendered += "<|im_start|>assistant\n<think>\n" + Self.trim(message.reasoningContent ?? "") + "\n</think>\n\n" + content + "<|im_end|>\n"
            default: throw GPUWeightError.invalid("Text runner supports system/user/assistant only; unsupported role \(message.role)")
            }
        }
        if addGenerationPrompt { rendered += "<|im_start|>assistant\n<think>\n\n</think>\n\n" }
        return rendered
    }

    private func encodeOrdinary(_ input: String) throws -> [Int32] {
        let text = normalization == .modelNFC ? input.precomposedStringWithCanonicalMapping : input
        let ns = text as NSString
        let matches = split.matches(in: text, range: NSRange(location: 0, length: ns.length))
        var result: [Int32] = []
        var end = 0
        for match in matches {
            // Split(Isolated) also retains any unmatched spans.
            if match.range.location > end { result.append(contentsOf: bpe(Array(ns.substring(with: NSRange(location: end, length: match.range.location - end)).utf8))) }
            result.append(contentsOf: bpe(Array(ns.substring(with: match.range).utf8)))
            end = match.range.location + match.range.length
        }
        if end < ns.length { result.append(contentsOf: bpe(Array(ns.substring(from: end).utf8))) }
        return result
    }

    private func bpe(_ bytes: [UInt8]) -> [Int32] {
        guard !bytes.isEmpty else { return [] }
        var tokens = bytes.map { bytesToIDs[Int($0)] }
        if tokens.count == 1 { return tokens }
        var previous = Array(-1..<(tokens.count - 1))
        var next = Array(1...tokens.count); next[tokens.count - 1] = -1
        var version = [Int](repeating: 0, count: tokens.count)
        var live = [Bool](repeating: true, count: tokens.count)
        var heap: [Candidate] = []
        func push(_ item: Candidate) {
            heap.append(item)
            var index = heap.count - 1
            while index > 0 {
                let parent = (index - 1) / 2
                if !heap[index].precedes(heap[parent]) { break }
                heap.swapAt(index, parent); index = parent
            }
        }
        func add(_ left: Int) {
            guard left >= 0, live[left], next[left] >= 0 else { return }
            let right = next[left]
            if let merge = merges[Self.pair(tokens[left], tokens[right])] {
                push(Candidate(rank: merge.rank, left: left, right: right, leftVersion: version[left], rightVersion: version[right], token: merge.token))
            }
        }
        func pop() -> Candidate? {
            guard !heap.isEmpty else { return nil }
            let first = heap[0], tail = heap.removeLast()
            if !heap.isEmpty {
                heap[0] = tail
                var index = 0
                while 2 * index + 1 < heap.count {
                    var child = 2 * index + 1
                    if child + 1 < heap.count && heap[child + 1].precedes(heap[child]) { child += 1 }
                    if !heap[child].precedes(heap[index]) { break }
                    heap.swapAt(index, child); index = child
                }
            }
            return first
        }
        for index in 0..<(tokens.count - 1) { add(index) }
        while let candidate = pop() {
            let left = candidate.left, right = candidate.right
            guard live[left], live[right], next[left] == right,
                  version[left] == candidate.leftVersion, version[right] == candidate.rightVersion else { continue }
            tokens[left] = candidate.token; version[left] += 1
            live[right] = false; next[left] = next[right]
            if next[right] >= 0 { previous[next[right]] = left }
            add(previous[left]); add(left)
        }
        var result: [Int32] = []
        var index = 0
        while index >= 0 { result.append(tokens[index]); index = next[index] }
        return result
    }

    private static func pair(_ left: Int32, _ right: Int32) -> UInt64 { (UInt64(UInt32(bitPattern: left)) << 32) | UInt64(UInt32(bitPattern: right)) }
    private static func trim(_ text: String) -> String { text.trimmingCharacters(in: .whitespacesAndNewlines) }
    private static func tokenID(_ raw: Any) throws -> Int32 {
        guard let n = raw as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID(), let id = Int32(n.stringValue), id >= 0 else {
            throw GPUWeightError.invalid("Invalid tokenizer ID")
        }
        return id
    }
}
