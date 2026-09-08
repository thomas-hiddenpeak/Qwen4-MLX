import Foundation
import XCTest
@testable import ANERunnerCore

final class QwenHTTPToolsTests: XCTestCase {
    private var definition: [String: Any] { ["type": "function", "function": ["name": "inspect", "description": "观察中文😀 <&>",
        "parameters": ["type": "object", "properties": ["city": ["type": "string"], "count": ["type": "integer"],
            "flags": ["type": "array", "items": ["type": "boolean"]], "meta": ["type": "object"]],
            "required": ["city"], "additionalProperties": false]]] }
    private func request(_ changes: [String: Any] = [:]) throws -> QwenHTTPChatRequest {
        var object: [String: Any] = ["model": "test", "messages": [["role": "user", "content": "check"]], "tools": [definition]]
        changes.forEach { object[$0.key] = $0.value }
        return try QwenHTTPChatRequest.decode(JSONSerialization.data(withJSONObject: object), expectedModel: "test")
    }
    private func xml(city: String = "北京😀", count: String = "9007199254740993") -> String {
        "<tool_call>\n<function=inspect>\n<parameter=city>\n\(city)\n</parameter>\n<parameter=count>\n\(count)\n</parameter>\n<parameter=flags>[true,false]</parameter>\n<parameter=meta>{\"ok\":true}</parameter>\n</function>\n</tool_call>"
    }
    private func parser(limit: Int = 8192) throws -> QwenToolStreamParser {
        .init(tools: try request().tools, idPrefix: "call_req", maxBytes: limit)
    }
    private func call(id: String = "call_1", arguments: String = "{\"city\":\"上海\"}") -> [String: Any] {
        ["id": id, "type": "function", "function": ["name": "inspect", "arguments": arguments]]
    }
    func testAutoNoneAndExplicitUnsupportedControls() throws {
        let auto = try request()
        XCTAssertEqual(auto.toolChoice, "auto"); XCTAssertEqual(auto.activeTools.count, 1); XCTAssertTrue(auto.parsesTools)
        let none = try request(["tool_choice": "none"])
        XCTAssertTrue(none.activeTools.isEmpty); XCTAssertTrue(none.parsesTools)
        for value in ["required", ["type": "function", "function": ["name": "inspect"]], true, NSNull()] as [Any] {
            XCTAssertThrowsError(try request(["tool_choice": value]))
        }
        XCTAssertThrowsError(try request(["tools": [], "tool_choice": "auto"]))
        XCTAssertThrowsError(try request(["mtp_depth": 2]))
        var strict = definition
        var f = strict["function"] as! [String: Any]; f["strict"] = true; strict["function"] = f
        XCTAssertThrowsError(try request(["tools": [strict]]))
        f["strict"] = false; strict["function"] = f
        XCTAssertEqual(try request(["tools": [strict]]).tools.count, 1)
    }
    func testDefinitionNamesSchemasDuplicatesAndLimit() throws {
        for fields in [
            ["name": "bad>name", "parameters": ["type": "object"]],
            ["name": "ok", "parameters": ["type": "array"]],
            ["name": "ok", "parameters": ["type": "object", "properties": ["bad>x": ["type": "string"]]]],
            ["name": "ok", "parameters": ["type": "object", "required": ["missing"]]],
            ["name": "ok", "parameters": ["type": "object"], "strict": 0]
        ] as [[String: Any]] { XCTAssertThrowsError(try request(["tools": [["type": "function", "function": fields]]])) }
        XCTAssertThrowsError(try request(["tools": [definition, definition]]))
        XCTAssertThrowsError(try request(["tools": Array(repeating: definition, count: 65)]))
    }
    func testHistoryCallAndResultRoundTripWithNullContent() throws {
        let history: [[String: Any]] = [["role": "user", "content": "check"],
            ["role": "assistant", "content": NSNull(), "tool_calls": [call(), call(id: "call_2")]],
            ["role": "tool", "tool_call_id": "call_1", "content": "first"],
            ["role": "tool", "tool_call_id": "call_2", "content": "second"]]
        XCTAssertThrowsError(try request(["messages": [history[0], history[1], history[3], history[2]]]))
        let decoded = try request(["messages": history])
        XCTAssertEqual(decoded.messages[1].content, ""); XCTAssertEqual(decoded.messages[1].toolCalls.count, 2)
        XCTAssertEqual(decoded.messages[2].toolCallID, "call_1")
        let rendered = try QwenToolChatTemplate.render(messages: decoded.messages.map {
            .init(role: $0.role, content: $0.content, calls: $0.toolCalls)
        }, tools: decoded.tools)
        XCTAssertTrue(rendered.contains("<parameter=city>\n上海\n</parameter>"))
        XCTAssertTrue(rendered.contains("<|im_start|>user\n<tool_response>\nfirst\n</tool_response>\n<tool_response>\nsecond\n</tool_response><|im_end|>"))
        XCTAssertTrue(rendered.hasSuffix("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
    }
    func testRejectsUnmatchedRepeatedIncompleteOrAmbiguousHistory() throws {
        let user: [String: Any] = ["role": "user", "content": "check"]
        let assistant: [String: Any] = ["role": "assistant", "content": "", "tool_calls": [call()]]
        let result: [String: Any] = ["role": "tool", "tool_call_id": "call_1", "content": "done"]
        for history in [[user, result], [user, assistant], [user, assistant, user], [user, assistant, result, result],
                        [user, assistant, result, assistant, result]] {
            XCTAssertThrowsError(try request(["messages": history]))
        }
        for arguments in ["[]", "null", "{\"x\":1,\"x\":2}", "{", "{\"city\":\"</parameter>\"}"] {
            XCTAssertThrowsError(try QwenToolCall.decode(call(arguments: arguments)))
        }
    }
    func testShippedTemplateSystemArmAndHTMLSafeJSON() throws {
        let tools = try request().tools
        let prefix = QwenToolChatTemplate.systemPrefix(system: " rules ", tools: tools)
        XCTAssertTrue(prefix.hasPrefix("<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n{"))
        XCTAssertTrue(prefix.contains("\\u4e2d\\u6587\\ud83d\\ude00 \\u003c\\u0026\\u003e"))
        XCTAssertTrue(prefix.contains("\n</tools>\n\nIf you choose"))
        XCTAssertTrue(prefix.hasSuffix("</IMPORTANT>\n\nrules<|im_end|>\n"))
        XCTAssertEqual(QwenToolChatTemplate.systemPrefix(system: " x ", tools: []), "<|im_start|>system\nx<|im_end|>\n")
        XCTAssertEqual(QwenToolChatTemplate.systemPrefix(system: nil, tools: []), "")
    }
    func testEveryUTF8ByteSplitParsesExactlyWithoutMarkerLeak() throws {
        let source = "查询。\n" + xml() + "\n", bytes = Array(source.utf8)
        for cut in 0...bytes.count {
            var p = try parser(), decoder = IncrementalUTF8Decoder()
            var events = try p.append(decoder.append(Array(bytes.prefix(cut))))
            events += try p.append(decoder.append(Array(bytes.dropFirst(cut))))
            events += try p.append(decoder.finish()); events += try p.finish()
            let text = events.compactMap { if case .content(let v) = $0 { return v }; return nil }.joined()
            XCTAssertEqual(text, "查询。\n")
            let calls = events.compactMap { if case .call(let v) = $0 { return v }; return nil }
            XCTAssertEqual(calls.count, 1)
            XCTAssertEqual(calls.first?.arguments.object?["city"], .string("北京😀"))
            XCTAssertTrue(try calls[0].arguments.json().contains("9007199254740993"))
            XCTAssertEqual(calls[0].arguments.object?["flags"], .array([.bool(true), .bool(false)]))
        }
        var p = try parser(), decoder = IncrementalUTF8Decoder(), output: [QwenToolStreamParser.Event] = []
        for byte in bytes { output += try p.append(decoder.append([byte])) }
        output += try p.append(decoder.finish()); output += try p.finish()
        XCTAssertEqual(p.calls.count, 1)
        XCTAssertFalse(output.contains { if case .content(let v) = $0 { return v.contains("<tool_call>") }; return false })
    }
    func testMultipleCallsEmitStableIndicesAndNoSuffix() throws {
        var p = try parser()
        let events = try p.append(xml() + "\n" + xml(city: "上海"))
        XCTAssertEqual(events.count, 2); XCTAssertEqual(p.calls.map(\.id), ["call_req_0", "call_req_1"])
        XCTAssertThrowsError(try p.append("and afterwards"))
        var none = QwenToolStreamParser(tools: [], idPrefix: "call", maxBytes: 8192)
        XCTAssertThrowsError(try none.append(xml()))
    }
    func testMalformedCallsNeverBecomeTextAndMissingClosureFails() throws {
        for source in [xml().replacingOccurrences(of: "function=inspect", with: "function=unknown"),
            xml(count: "not a number"), xml().replacingOccurrences(of: "<parameter=city>", with: "<parameter=wrong>"),
            xml().replacingOccurrences(of: "<parameter=count>", with: "<parameter=city>"),
            "<tool_call>{\"name\":\"inspect\"}</tool_call>", "<tool_call >", "<function=inspect>",
            "</tool_call>", "<tool_calls>"] {
            var p = try parser()
            XCTAssertThrowsError(try { _ = try p.append(source); _ = try p.finish() }())
        }
        for source in ["<tool_call>\n<function=inspect>", "<tool", "<parameter", "text <tool_call"] {
            var p = try parser(); _ = try p.append(source)
            XCTAssertThrowsError(try p.finish())
        }
    }
    func testPlainContentAndStringWhitespaceRemainIntact() throws {
        var p = try parser()
        let text = "普通说明 <b>bold</b> 1 < 2 😀"
        let events = try p.append(text) + p.finish()
        XCTAssertEqual(events, [.content(text)])
        var strings = try parser()
        _ = try strings.append(xml(city: "  北京\nsecond  "))
        XCTAssertEqual(strings.calls[0].arguments.object?["city"], .string("  北京\nsecond  "))
    }
    func testPendingAndCumulativeCallLimits() throws {
        var small = try parser(limit: 16)
        XCTAssertThrowsError(try small.append(xml())) { XCTAssertEqual($0 as? QwenToolStreamParser.Failure, .outputLimit) }
        var bounded = try parser(limit: 512)
        var overflow = false
        for _ in 0..<17 {
            do { _ = try bounded.append(xml()) } catch { overflow = true; break }
        }
        XCTAssertTrue(overflow)
    }
    func testStructuredFramesAndCachedUsage() throws {
        var p = try parser(); _ = try p.append(xml())
        let call = p.calls[0]
        let sse = try QwenHTTPFrames.toolCall(id: "id", created: 1, model: "test", call: call, index: 0)
        let chunk = try XCTUnwrap(JSONSerialization.jsonObject(with: sse.dropFirst(6).dropLast(2)) as? [String: Any])
        let choice = try XCTUnwrap((chunk["choices"] as? [[String: Any]])?.first)
        let delta = try XCTUnwrap(choice["delta"] as? [String: Any])
        XCTAssertNil(delta["content"])
        XCTAssertEqual((delta["tool_calls"] as? [[String: Any]])?.first?["index"] as? Int, 0)
        let completion = try QwenHTTPFrames.completion(id: "id", created: 1, model: "test", text: "", reason: "tool_calls",
            promptTokens: 1024, completionTokens: 50, cachedTokens: 832, toolCalls: p.calls)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: completion) as? [String: Any])
        let message = try XCTUnwrap(((object["choices"] as? [[String: Any]])?.first)?["message"] as? [String: Any])
        XCTAssertTrue(message["content"] is NSNull)
        let usage = try XCTUnwrap(object["usage"] as? [String: Any])
        XCTAssertEqual((usage["prompt_tokens_details"] as? [String: Int])?["cached_tokens"], 832)
        XCTAssertThrowsError(try QwenHTTPFrames.finish(id: "id", created: 1, model: "test", reason: "stop", promptTokens: 1, completionTokens: 1, cachedTokens: 2))
    }
}
