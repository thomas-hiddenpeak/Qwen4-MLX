import CoreFoundation
import Foundation

/// JSON kept as value types so request history never carries Foundation objects
/// across the network/inference thread boundary.
public indirect enum QwenToolJSON: Codable, Equatable, Sendable {
    case object([String: QwenToolJSON]), array([QwenToolJSON]), string(String), number(Decimal), bool(Bool), null

    public init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { self = .null }
        else if let v = try? c.decode(Bool.self) { self = .bool(v) }
        else if let v = try? c.decode(String.self) { self = .string(v) }
        else if let v = try? c.decode([String: Self].self) { self = .object(v) }
        else if let v = try? c.decode([Self].self) { self = .array(v) }
        else { self = .number(try c.decode(Decimal.self)) }
    }
    public func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .object(let v): try c.encode(v)
        case .array(let v): try c.encode(v)
        case .string(let v): try c.encode(v)
        case .number(let v): try c.encode(v)
        case .bool(let v): try c.encode(v)
        case .null: try c.encodeNil()
        }
    }
    public var object: [String: Self]? { if case .object(let v) = self { return v }; return nil }
    public var string: String? { if case .string(let v) = self { return v }; return nil }
    public var array: [Self]? { if case .array(let v) = self { return v }; return nil }
    public var foundation: Any {
        switch self {
        case .object(let v): return v.mapValues(\.foundation)
        case .array(let v): return v.map(\.foundation)
        case .string(let v): return v
        case .number(let v): return NSDecimalNumber(decimal: v)
        case .bool(let v): return v
        case .null: return NSNull()
        }
    }
    public static func parse(_ data: Data) throws -> Self {
        try QwenHTTPJSONKeys.validate(data)
        return try JSONDecoder().decode(Self.self, from: data)
    }
    public static func from(_ object: Any) throws -> Self {
        try parse(JSONSerialization.data(withJSONObject: object, options: [.fragmentsAllowed, .sortedKeys]))
    }
    public func json() throws -> String {
        String(decoding: try JSONSerialization.data(withJSONObject: foundation,
            options: [.fragmentsAllowed, .sortedKeys, .withoutEscapingSlashes]), as: UTF8.self)
    }
    /// Canonical JSON: sorted keys, ASCII/HTML-safe escapes and spaced
    /// separators. Equivalent values, not byte-identical upstream rendering.
    public func templateJSON() -> String {
        func quote(_ value: String) -> String {
            var result = "\""
            for u in value.utf16 {
                switch u {
                case 34: result += "\\\""
                case 92: result += "\\\\"
                case 8: result += "\\b"
                case 9: result += "\\t"
                case 10: result += "\\n"
                case 12: result += "\\f"
                case 13: result += "\\r"
                case 0..<32, 127...65535, 38, 39, 60, 62:
                    result += String(format: "\\u%04x", Int(u))
                default: result += String(Unicode.Scalar(u)!)
                }
            }
            return result + "\""
        }
        switch self {
        case .object(let v): return "{" + v.keys.sorted().map { quote($0) + ": " + v[$0]!.templateJSON() }.joined(separator: ", ") + "}"
        case .array(let v): return "[" + v.map { $0.templateJSON() }.joined(separator: ", ") + "]"
        case .string(let v): return quote(v)
        case .number(let v): return NSDecimalNumber(decimal: v).stringValue
        case .bool(let v): return v ? "true" : "false"
        case .null: return "null"
        }
    }
}

public struct QwenToolDefinition: Codable, Equatable, Sendable {
    public let name: String
    public let function: QwenToolJSON
    public var parameters: [String: QwenToolJSON] { function.object?["parameters"]?.object ?? [:] }
    public var wire: QwenToolJSON { .object(["type": .string("function"), "function": function]) }
    public init(name: String, function: QwenToolJSON) { self.name = name; self.function = function }

    public static func decode(_ raw: Any) throws -> Self {
        guard let wrapper = raw as? [String: Any], Set(wrapper.keys) == ["type", "function"],
              wrapper["type"] as? String == "function", let f = wrapper["function"] as? [String: Any],
              Set(f.keys).isSubset(of: ["name", "description", "parameters", "strict"]),
              let name = f["name"] as? String, validName(name, limit: 64),
              f["description"] == nil || f["description"] is String,
              let p = f["parameters"] as? [String: Any], p["type"] as? String == "object" else {
            throw invalid("tools must contain function definitions with names and object parameter schemas")
        }
        // Strict constrained decoding is not implemented. Never silently claim it.
        if let strict = f["strict"] {
            guard let value = strict as? NSNumber, CFGetTypeID(value) == CFBooleanGetTypeID(), !value.boolValue else {
                throw invalid("strict tools are unsupported; use strict=false or omit it")
            }
        }
        if let properties = p["properties"] {
            guard let fields = properties as? [String: Any], fields.allSatisfy({ validName($0.key, limit: 128) && $0.value is [String: Any] }) else {
                throw invalid("Tool properties must have XML-safe names and object schemas")
            }
        }
        if let required = p["required"] {
            guard let names = required as? [String], Set(names).count == names.count,
                  names.allSatisfy({ (p["properties"] as? [String: Any])?[$0] != nil }) else {
                throw invalid("Tool required names must refer to declared properties")
            }
        }
        return Self(name: name, function: try .from(f))
    }
    static func validName(_ value: String, limit: Int) -> Bool {
        !value.isEmpty && value.utf8.count <= limit && value.utf8.allSatisfy {
            (48...57).contains($0) || (65...90).contains($0) || (97...122).contains($0) || [45, 46, 95].contains($0)
        }
    }
    static func invalid(_ message: String) -> QwenHTTPProtocolError { .init(statusCode: 400, message: message) }
}

public struct QwenToolCall: Codable, Equatable, Sendable {
    public let id: String
    public let name: String
    public let arguments: QwenToolJSON
    public init(id: String, name: String, arguments: QwenToolJSON) { self.id = id; self.name = name; self.arguments = arguments }
    public func wire(index: Int? = nil) throws -> [String: Any] {
        var result: [String: Any] = ["id": id, "type": "function", "function": ["name": name, "arguments": try arguments.json()]]
        if let index { result["index"] = index }
        return result
    }
    public static func decode(_ raw: Any) throws -> Self {
        guard let call = raw as? [String: Any], Set(call.keys) == ["id", "type", "function"],
              call["type"] as? String == "function", let id = call["id"] as? String,
              QwenToolDefinition.validName(id, limit: 128), let f = call["function"] as? [String: Any],
              Set(f.keys) == ["name", "arguments"], let name = f["name"] as? String,
              QwenToolDefinition.validName(name, limit: 64), let rawArguments = f["arguments"] as? String,
              let arguments = try? QwenToolJSON.parse(Data(rawArguments.utf8)), let fields = arguments.object,
              fields.keys.allSatisfy({ QwenToolDefinition.validName($0, limit: 128) }) else {
            throw QwenToolDefinition.invalid("assistant tool_calls need unique ids, function names and JSON object argument strings")
        }
        // The original XML dialect has no escaping for these delimiters.
        guard !fields.values.contains(where: { value in
            value.string.map { QwenToolStreamParser.containsControl($0) } ?? false
        }) else { throw QwenToolDefinition.invalid("Tool string arguments contain ambiguous XML control delimiters") }
        return Self(id: id, name: name, arguments: arguments)
    }
}

/// Parses only the shipped model's function/parameter XML dialect. A complete
/// call is validated before emission; fragments never become ordinary content.
/// All retained call payloads together, and each pending fragment, are bounded.
public struct QwenToolStreamParser: Sendable {
    public enum Event: Equatable, Sendable { case content(String), call(QwenToolCall) }
    public enum Failure: Error, Equatable, Sendable { case invalidCall, outputLimit }
    private let tools: [QwenToolDefinition]
    private let idPrefix: String
    private let maxBytes: Int
    private var pending = ""
    private var inside = false, ended = false
    private var callBytes = 0
    public private(set) var calls: [QwenToolCall] = []
    static let controls = ["<tool_call>", "</tool_call>", "<function=", "</function>", "<parameter=", "</parameter>", "<tool_response>", "</tool_response>"]
    private static let stems = ["<tool_call", "</tool_call", "<function", "</function", "<parameter", "</parameter", "<tool_response", "</tool_response"]
    public init(tools: [QwenToolDefinition], idPrefix: String, maxBytes: Int) {
        self.tools = tools; self.idPrefix = idPrefix; self.maxBytes = max(0, maxBytes)
    }
    static func containsControl(_ value: String) -> Bool { stems.contains { value.contains($0) } }
    public mutating func append(_ text: String) throws -> [Event] {
        guard !ended else { throw Failure.invalidCall }
        guard text.utf8.count <= maxBytes - pending.utf8.count else { throw Failure.outputLimit }
        pending += text
        var events: [Event] = []
        while !pending.isEmpty {
            if inside {
                guard let close = pending.range(of: "</tool_call>") else { break }
                let body = String(pending[..<close.lowerBound])
                let call = try parse(body)
                let bytes = try call.arguments.json().utf8.count + call.name.utf8.count + call.id.utf8.count
                guard calls.count < 16, bytes <= maxBytes - callBytes else { throw Failure.outputLimit }
                callBytes += bytes; calls.append(call); events.append(.call(call))
                pending = String(pending[close.upperBound...]); inside = false
                continue
            }
            if pending.hasPrefix("<tool_call>") {
                pending.removeFirst("<tool_call>".count); inside = true; continue
            }
            if Self.controls.contains(where: { $0.hasPrefix(pending) }) { break }
            if Self.stems.contains(where: { pending.hasPrefix($0) }) { throw Failure.invalidCall }
            let first = pending.removeFirst()
            if calls.isEmpty {
                if case .content(let previous) = events.last { events[events.count - 1] = .content(previous + String(first)) }
                else { events.append(.content(String(first))) }
            } else if !first.isWhitespace { throw Failure.invalidCall }
        }
        return events
    }
    public mutating func finish() throws -> [Event] {
        guard !ended else { throw Failure.invalidCall }; ended = true
        guard !inside else { throw Failure.invalidCall }
        guard !Self.controls.contains(where: { $0.hasPrefix(pending) }) || pending.isEmpty else { throw Failure.invalidCall }
        defer { pending = "" }
        if calls.isEmpty { return pending.isEmpty ? [] : [.content(pending)] }
        guard pending.allSatisfy(\.isWhitespace) else { throw Failure.invalidCall }
        return []
    }
    private func parse(_ body: String) throws -> QwenToolCall {
        var rest = body.trimmingCharacters(in: .whitespacesAndNewlines)
        guard rest.hasPrefix("<function="), let endName = rest.firstIndex(of: ">") else { throw Failure.invalidCall }
        let name = String(rest[rest.index(rest.startIndex, offsetBy: 10)..<endName])
        guard let tool = tools.first(where: { $0.name == name }) else { throw Failure.invalidCall }
        rest = String(rest[rest.index(after: endName)...])
        var fields: [String: QwenToolJSON] = [:]
        while true {
            rest = rest.trimmingCharacters(in: .whitespacesAndNewlines)
            if rest == "</function>" { break }
            guard rest.hasPrefix("<parameter="), let end = rest.firstIndex(of: ">") else { throw Failure.invalidCall }
            let key = String(rest[rest.index(rest.startIndex, offsetBy: 11)..<end])
            guard QwenToolDefinition.validName(key, limit: 128), fields[key] == nil else { throw Failure.invalidCall }
            rest = String(rest[rest.index(after: end)...])
            guard let close = rest.range(of: "</parameter>") else { throw Failure.invalidCall }
            var raw = String(rest[..<close.lowerBound])
            if raw.hasPrefix("\n") { raw.removeFirst() }
            if raw.hasSuffix("\n") { raw.removeLast() }
            guard !Self.containsControl(raw) else { throw Failure.invalidCall }
            let schema = tool.parameters["properties"]?.object?[key]?.object ?? [:]
            let value: QwenToolJSON
            if schema["type"]?.string == "string" { value = .string(raw) }
            else { value = (try? .parse(Data(raw.utf8))) ?? .string(raw) }
            try Self.validate(value, schema: schema)
            fields[key] = value; rest = String(rest[close.upperBound...])
        }
        try Self.validate(.object(fields), schema: tool.parameters)
        return QwenToolCall(id: idPrefix + "_\(calls.count)", name: name, arguments: .object(fields))
    }
    // The adapter checks declared JSON types, required keys and enum. It does
    // not advertise strict JSON Schema constrained decoding or execute tools.
    private static func validate(_ value: QwenToolJSON, schema: [String: QwenToolJSON]) throws {
        if let types = schema["type"] {
            let accepted = types.string.map { [$0] } ?? types.array?.compactMap(\.string) ?? []
            let type: String
            switch value {
            case .object: type = "object"
            case .array: type = "array"
            case .string: type = "string"
            case .number(let n):
                var input = n, rounded = Decimal()
                NSDecimalRound(&rounded, &input, 0, .plain)
                type = rounded == n ? "integer" : "number"
            case .bool: type = "boolean"
            case .null: type = "null"
            }
            guard accepted.contains(type) || (type == "integer" && accepted.contains("number")) else { throw Failure.invalidCall }
        }
        if let allowed = schema["enum"]?.array, !allowed.contains(value) { throw Failure.invalidCall }
        if case .object(let fields) = value {
            let required = schema["required"]?.array?.compactMap(\.string) ?? []
            guard required.allSatisfy({ fields[$0] != nil }) else { throw Failure.invalidCall }
            let properties = schema["properties"]?.object ?? [:]
            if schema["additionalProperties"] == .bool(false), !Set(fields.keys).isSubset(of: Set(properties.keys)) { throw Failure.invalidCall }
            for (key, field) in fields { try validate(field, schema: properties[key]?.object ?? [:]) }
        }
        if case .array(let fields) = value, let item = schema["items"]?.object {
            for field in fields { try validate(field, schema: item) }
        }
    }
}
