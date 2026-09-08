import CoreFoundation
import Foundation

public struct QwenHTTPProtocolError: Error, Equatable, LocalizedError, Sendable {
    public let statusCode: Int
    public let message: String
    public var errorDescription: String? { message }
    public init(statusCode: Int, message: String) {
        self.statusCode = statusCode; self.message = message
    }
}

public struct QwenHTTPRequest: Sendable {
    public let method: String
    /// Original origin-form target; no percent decoding or path normalization.
    public let path: String
    public let body: Data
}

/// One HTTP/1.1 request per connection, with a fixed 16 KiB header budget.
/// feed returns a request exactly once. Any later nonempty feed is rejected;
/// adapters must continue checking received bytes after dispatching a request.
/// No parser can reject a future TCP packet before the transport supplies it.
public struct QwenHTTPRequestParser: Sendable {
    private struct Header: Sendable { let method, path: String; let length: Int }
    private enum State: Sendable { case headers, body(Header), complete, failed }
    private var state: State = .headers
    private var buffer = Data()
    private let maxBodyBytes: Int
    public static let maxHeaderBytes = 16 * 1024

    public init(maxBodyBytes: Int = 1_048_576) throws {
        guard maxBodyBytes > 0, maxBodyBytes <= Int.max - Self.maxHeaderBytes else {
            throw QwenHTTPProtocolError(statusCode: 500, message: "Invalid request body limit")
        }
        self.maxBodyBytes = maxBodyBytes
    }

    public mutating func feed(_ data: Data) throws -> QwenHTTPRequest? {
        do {
            switch state {
            case .failed: throw failure("Request parser is already closed")
            case .complete:
                guard data.isEmpty else { throw failure("Pipelined or excess request bytes are unsupported") }
                return nil
            case .body(let header): return try appendBody(data, header: header)
            case .headers:
                // Inspect at most the remaining header budget before copying a
                // possibly large body. Data fragments may have nonzero indices.
                let count = min(data.count, Self.maxHeaderBytes - buffer.count)
                buffer.append(contentsOf: data.prefix(count))
                guard let delimiter = buffer.range(of: Data([13,10,13,10])) else {
                    guard buffer.count < Self.maxHeaderBytes else {
                        throw QwenHTTPProtocolError(statusCode: 431, message: "Request headers exceed 16 KiB")
                    }
                    return nil
                }
                let header = try parseHeader(Data(buffer[..<delimiter.lowerBound]))
                buffer = Data(buffer[delimiter.upperBound...])
                guard buffer.count <= header.length else { throw failure("Pipelined or excess request bytes are unsupported") }
                state = .body(header)
                return try appendBody(data.dropFirst(count), header: header)
            }
        } catch {
            state = .failed; buffer.removeAll(keepingCapacity: false)
            throw error
        }
    }

    /// Call when the peer ends its request input. A truncated request is an
    /// error; EOF after a parsed request is harmless. Failed parsers stay failed.
    public mutating func finish() throws {
        if case .complete = state { return }
        state = .failed; buffer.removeAll(keepingCapacity: false)
        throw failure("Incomplete HTTP request")
    }

    private mutating func appendBody(_ bytes: Data, header: Header) throws -> QwenHTTPRequest? {
        guard bytes.count <= header.length - buffer.count else {
            throw failure("Pipelined or excess request bytes are unsupported")
        }
        buffer.append(bytes)
        guard buffer.count == header.length else { return nil }
        let request = QwenHTTPRequest(method: header.method, path: header.path, body: buffer)
        buffer = Data(); state = .complete
        return request
    }

    private func parseHeader(_ bytes: Data) throws -> Header {
        guard bytes.allSatisfy({ $0 == 9 || $0 == 10 || $0 == 13 || (32...126).contains($0) }),
              let text = String(data: bytes, encoding: .ascii) else {
            throw failure("Request headers must contain ASCII HTTP syntax")
        }
        let lines = text.components(separatedBy: "\r\n")
        guard let first = lines.first else { throw failure("Missing request line") }
        let request = first.split(separator: " ", omittingEmptySubsequences: false)
        guard request.count == 3, !request[0].isEmpty,
              request[0].utf8.allSatisfy(Self.isTokenByte), request[2] == "HTTP/1.1",
              request[1].hasPrefix("/"), !request[1].hasPrefix("//"),
              !request[1].contains("#"), request[1].utf8.allSatisfy({ (33...126).contains($0) }) else {
            throw failure("Require an HTTP/1.1 origin-form request line")
        }
        var host: String?, contentLength: String?, contentType: String?
        for line in lines.dropFirst() {
            guard !line.isEmpty, !line.hasPrefix(" "), !line.hasPrefix("\t"),
                  let colon = line.firstIndex(of: ":") else {
                throw failure("Invalid or folded HTTP header")
            }
            let name = line[..<colon]
            let rawValue = line[line.index(after: colon)...]
            guard !name.isEmpty, name.utf8.allSatisfy(Self.isTokenByte),
                  rawValue.utf8.allSatisfy({ $0 == 9 || (32...126).contains($0) }) else {
                throw failure("Invalid HTTP header name or value")
            }
            let value = rawValue.trimmingCharacters(in: CharacterSet(charactersIn: " \t"))
            switch name.lowercased() {
            case "host":
                guard host == nil, !value.isEmpty,
                      value.utf8.allSatisfy({ (33...126).contains($0) && ![44,47,92,64,35,63].contains($0) }) else {
                    throw failure("Require one unambiguous Host header")
                }
                host = value
            case "content-length":
                guard contentLength == nil else { throw failure("Duplicate Content-Length is unsupported") }
                contentLength = value
            case "transfer-encoding": throw failure("Transfer-Encoding requests are unsupported")
            case "content-type":
                guard contentType == nil else { throw failure("Duplicate Content-Type is unsupported") }
                contentType = value
            case "expect":
                throw QwenHTTPProtocolError(statusCode: 417, message: "Expect negotiation is unsupported")
            default: break
            }
        }
        guard host != nil else { throw failure("HTTP/1.1 requires Host") }
        let method = String(request[0])
        if method == "POST", contentLength == nil {
            throw QwenHTTPProtocolError(statusCode: 411, message: "POST requires Content-Length")
        }
        var length = 0
        if let contentLength {
            guard !contentLength.isEmpty, contentLength.utf8.allSatisfy({ (48...57).contains($0) }) else {
                throw failure("Content-Length must be a nonnegative decimal integer")
            }
            for byte in contentLength.utf8 {
                let digit = Int(byte - 48)
                guard length < maxBodyBytes / 10 ||
                        (length == maxBodyBytes / 10 && digit <= maxBodyBytes % 10) else {
                    throw QwenHTTPProtocolError(statusCode: 413, message: "Request body exceeds the configured limit")
                }
                length = length * 10 + digit
            }
        }
        if method != "POST", length != 0 { throw failure("Only POST requests may carry a body") }
        if method == "POST", let contentType {
            let media = contentType.components(separatedBy: ";").first?
                .trimmingCharacters(in: .whitespaces).lowercased()
            guard media == "application/json" else {
                throw QwenHTTPProtocolError(statusCode: 415, message: "POST body must be application/json")
            }
        }
        return Header(method: method, path: String(request[1]), length: length)
    }

    private func failure(_ message: String) -> QwenHTTPProtocolError {
        QwenHTTPProtocolError(statusCode: 400, message: message)
    }
    private static func isTokenByte(_ byte: UInt8) -> Bool {
        (48...57).contains(byte) || (65...90).contains(byte) || (97...122).contains(byte) ||
            [33,35,36,37,38,39,42,43,45,46,94,95,96,124,126].contains(byte)
    }
}

public struct QwenHTTPChatRequest: Sendable {
    public struct Message: Equatable, Sendable {
        public let role: String
        public let content: String
        public let toolCalls: [QwenToolCall]
        public let toolCallID: String?
    }
    public let model: String
    public let messages: [Message]
    public let maxTokens: Int
    public let stream: Bool
    public let mtpDepth: Int
    public let tools: [QwenToolDefinition]
    public let toolChoice: String
    public var parsesTools: Bool { !tools.isEmpty || messages.contains { !$0.toolCalls.isEmpty || $0.role == "tool" } }
    public var activeTools: [QwenToolDefinition] { toolChoice == "none" ? [] : tools }

    public static func decode(_ body: Data, expectedModel: String) throws -> Self {
        func invalid(_ message: String) -> QwenHTTPProtocolError {
            QwenHTTPProtocolError(statusCode: 400, message: message)
        }
        guard !expectedModel.isEmpty else { throw invalid("Server model identity is empty") }
        // Foundation can auto-detect UTF-16/32 JSON. Reject raw NUL and invalid
        // UTF-8 explicitly so this network DTO accepts only UTF-8 JSON bytes.
        guard !body.contains(0), String(data: body, encoding: .utf8) != nil else {
            throw invalid("JSON body must use UTF-8")
        }
        try QwenHTTPJSONKeys.validate(body)
        let value: Any
        do { value = try JSONSerialization.jsonObject(with: body) }
        catch { throw invalid("Malformed JSON body") }
        guard let object = value as? [String: Any] else { throw invalid("JSON body must be an object") }
        let allowed: Set<String> = ["model", "messages", "max_tokens", "stream", "temperature", "mtp_depth", "tools", "tool_choice"]
        guard Set(object.keys).isSubset(of: allowed) else { throw invalid("Unsupported chat request field") }
        guard let model = object["model"] as? String, model == expectedModel else {
            throw invalid("model must exactly match the served model")
        }
        guard let rawMessages = object["messages"] as? [Any], !rawMessages.isEmpty else {
            throw invalid("messages must be a nonempty array")
        }
        var tools: [QwenToolDefinition] = []
        if let raw = object["tools"] {
            guard let definitions = raw as? [Any], definitions.count <= 64 else { throw invalid("tools must be an array of at most 64 functions") }
            tools = try definitions.map(QwenToolDefinition.decode)
            guard Set(tools.map(\.name)).count == tools.count else { throw invalid("Tool names must be unique") }
        }
        let choice = object["tool_choice"] as? String ?? (object["tool_choice"] == nil ? (tools.isEmpty ? "none" : "auto") : "")
        guard ["none", "auto"].contains(choice), choice != "auto" || !tools.isEmpty else {
            throw invalid("tool_choice supports none or auto with tools; required and named choices are unsupported")
        }
        var messages: [Message] = []
        var pending: [String] = [], seen: Set<String> = []
        for (index, raw) in rawMessages.enumerated() {
            guard let item = raw as? [String: Any], let role = item["role"] as? String,
                  ["system", "user", "assistant", "tool"].contains(role), role != "system" || index == 0 else {
                throw invalid("Unsupported message role; system may appear only first")
            }
            let allowedMessage: Set<String> = role == "assistant" ? ["role", "content", "tool_calls"] : role == "tool" ? ["role", "content", "tool_call_id"] : ["role", "content"]
            guard Set(item.keys).isSubset(of: allowedMessage) else { throw invalid("Unsupported message field") }
            var calls: [QwenToolCall] = []
            if let rawCalls = item["tool_calls"] {
                guard role == "assistant", let list = rawCalls as? [Any], !list.isEmpty, list.count <= 16 else {
                    throw invalid("assistant tool_calls must contain 1...16 calls")
                }
                calls = try list.map(QwenToolCall.decode)
            }
            let content: String
            if let text = item["content"] as? String { content = text }
            else if !calls.isEmpty && (item["content"] == nil || item["content"] is NSNull) { content = "" }
            else { throw invalid("Messages require plain string content; tool-call assistants may use null") }
            let callID = item["tool_call_id"] as? String
            if role == "tool" {
                guard let callID, pending.first == callID else { throw invalid("tool_call_id must match pending assistant calls exactly once, in call order") }
                pending.removeFirst()
            } else {
                guard pending.isEmpty else { throw invalid("All pending calls need tool results before the next message") }
                for call in calls {
                    guard seen.insert(call.id).inserted else { throw invalid("Tool call ids must be unique in history") }
                    pending.append(call.id)
                }
            }
            messages.append(Message(role: role, content: content, toolCalls: calls, toolCallID: callID))
        }
        guard pending.isEmpty else { throw invalid("History ends with unanswered tool calls") }
        guard messages.contains(where: { $0.role == "user" }) else { throw invalid("messages must include a user message") }
        func number(_ value: Any, field: String) throws -> Double {
            guard let value = value as? NSNumber, CFGetTypeID(value) != CFBooleanGetTypeID(),
                  value.doubleValue.isFinite else { throw invalid("\(field) must be a finite number") }
            return value.doubleValue
        }
        func integer(_ value: Any, field: String, range: ClosedRange<Int>) throws -> Int {
            let value = try number(value, field: field)
            guard let result = Int(exactly: value), range.contains(result) else {
                throw invalid("\(field) must be an integer in \(range.lowerBound)...\(range.upperBound)")
            }
            return result
        }
        let maxTokens = try object["max_tokens"].map { try integer($0, field: "max_tokens", range: 1...4096) } ?? 128
        let mtpDepth = try object["mtp_depth"].map { try integer($0, field: "mtp_depth", range: 0...2) } ?? 0
        guard mtpDepth == 0 || mtpDepth == 2 else { throw invalid("mtp_depth must be 0 or 2") }
        guard mtpDepth == 0 || (tools.isEmpty && !messages.contains { !$0.toolCalls.isEmpty || $0.role == "tool" }) else {
            throw invalid("Tool requests currently require AR mtp_depth=0")
        }
        // The experimental MTP candidate has only been validated with a 1024
        // token draft history and output budgets up to 256. A larger budget
        // can cross the MTP head's QSA threshold and needs a separate regression.
        guard mtpDepth != 2 || maxTokens <= 256 else {
            throw invalid("Experimental mtp_depth=2 requires max_tokens <= 256")
        }
        if let temperature = object["temperature"], try number(temperature, field: "temperature") != 0 {
            throw invalid("Only greedy temperature=0 is supported")
        }
        let stream: Bool
        if let value = object["stream"] {
            guard let flag = value as? NSNumber, CFGetTypeID(flag) == CFBooleanGetTypeID() else {
                throw invalid("stream must be a JSON boolean")
            }
            stream = flag.boolValue
        } else { stream = false }
        return Self(model: model, messages: messages, maxTokens: maxTokens, stream: stream, mtpDepth: mtpDepth, tools: tools, toolChoice: choice)
    }
}

/// A bounded lexical preflight for duplicate (including escaped) object keys.
/// Full JSON syntax/value decoding remains Foundation's responsibility.
enum QwenHTTPJSONKeys {
    private struct Object { var keys: Set<String> = []; var expectsKey = true }
    static func validate(_ data: Data) throws {
        let bytes = Array(data)
        var stack: [Object?] = [], index = 0
        func invalid() -> QwenHTTPProtocolError {
            QwenHTTPProtocolError(statusCode: 400, message: "Duplicate keys, malformed JSON, or excessive JSON nesting")
        }
        while index < bytes.count {
            switch bytes[index] {
            case 123, 91:
                guard stack.count < 16 else { throw invalid() }
                stack.append(bytes[index] == 123 ? Object() : nil)
            case 125, 93:
                guard let frame = stack.last, (bytes[index] == 125) == (frame != nil) else { throw invalid() }
                stack.removeLast()
            case 44:
                if let last = stack.indices.last, stack[last] != nil { stack[last]!.expectsKey = true }
            case 34:
                let start = index
                index += 1
                while index < bytes.count && bytes[index] != 34 {
                    if bytes[index] == 92 { index += 1 }
                    index += 1
                }
                guard index < bytes.count else { throw invalid() }
                if let last = stack.indices.last, stack[last]?.expectsKey == true {
                    let key: String
                    do {
                        guard let value = try JSONSerialization.jsonObject(
                            with: Data(bytes[start...index]), options: .fragmentsAllowed) as? String else { throw invalid() }
                        key = value
                    } catch { throw invalid() }
                    guard stack[last]!.keys.insert(key).inserted else { throw invalid() }
                    stack[last]!.expectsKey = false
                }
            default: break
            }
            index += 1
        }
        guard stack.isEmpty else { throw invalid() }
    }
}
