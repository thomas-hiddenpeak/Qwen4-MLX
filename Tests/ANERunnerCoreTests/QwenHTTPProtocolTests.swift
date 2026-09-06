import Foundation
import XCTest
@testable import ANERunnerCore

final class QwenHTTPProtocolTests: XCTestCase {
    private let model = "Qwen-local"
    private func body(_ edits: [String: Any] = [:]) throws -> Data {
        var object: [String: Any] = ["model": model, "messages": [["role": "user", "content": "你好😀"]]]
        for (key, value) in edits { object[key] = value }
        return try JSONSerialization.data(withJSONObject: object, options: .sortedKeys)
    }
    private func request(_ body: Data) -> Data {
        var bytes = Data("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost:8000\r\nContent-Type: application/json; charset=utf-8\r\nContent-Length: \(body.count)\r\n\r\n".utf8)
        bytes.append(body)
        return bytes
    }
    private func assertHTTPError(_ bytes: Data, status: Int = 400, limit: Int = 1_048_576,
                                 file: StaticString = #filePath, line: UInt = #line) throws {
        var parser = try QwenHTTPRequestParser(maxBodyBytes: limit)
        XCTAssertThrowsError(try parser.feed(bytes), file: file, line: line) {
            XCTAssertEqual(($0 as? QwenHTTPProtocolError)?.statusCode, status, file: file, line: line)
        }
        XCTAssertThrowsError(try parser.feed(Data()), file: file, line: line)
    }

    func testEveryRequestSplitAndByteFragmentsPreserveBodyExactly() throws {
        let payload = try body(), bytes = request(payload)
        for cut in 0...bytes.count {
            var parser = try QwenHTTPRequestParser()
            let first = try parser.feed(bytes.prefix(cut))
            let second = try parser.feed(bytes.dropFirst(cut))
            XCTAssertEqual([first, second].compactMap { $0 }.count, 1)
            let result = try XCTUnwrap(first ?? second)
            XCTAssertEqual(result.method, "POST"); XCTAssertEqual(result.path, "/v1/chat/completions")
            XCTAssertEqual(result.body, payload)
            try parser.finish(); try parser.finish()
            XCTAssertNil(try parser.feed(Data()))
        }
        var parser = try QwenHTTPRequestParser(), results: [QwenHTTPRequest] = []
        for byte in bytes {
            if let result = try parser.feed(Data([byte])) { results.append(result) }
        }
        XCTAssertEqual(results.count, 1); XCTAssertEqual(results.first?.body, payload)
        try parser.finish()
    }

    func testGETRoutesAndNonzeroDataIndices() throws {
        for route in ["/health", "/v1/models"] {
            var parser = try QwenHTTPRequestParser()
            let bytes = Data(("!GET \(route) HTTP/1.1\r\nHost: [::1]:8000\r\nContent-Length: 0\r\n\r\n!").utf8)
                .dropFirst().dropLast()
            let parsed = try XCTUnwrap(parser.feed(bytes))
            XCTAssertEqual(parsed.method, "GET"); XCTAssertEqual(parsed.path, route)
            XCTAssertTrue(parsed.body.isEmpty)
        }
    }

    func testRejectsSmugglingHeadersAndAmbiguousRequestLines() throws {
        let invalid = [
            "GET /health HTTP/1.1\r\n\r\n",
            "GET /health HTTP/1.1\r\nHost: a\r\nHOST: a\r\n\r\n",
            "POST /v1/chat/completions HTTP/1.1\r\nHost: a\r\nContent-Length: 0\r\ncontent-length: 0\r\n\r\n",
            "POST /v1/chat/completions HTTP/1.1\r\nHost: a\r\nContent-Length: 0\r\nTransfer-Encoding: chunked\r\n\r\n",
            "GET /health HTTP/1.1\r\nHost: a\r\nTransfer-Encoding: identity\r\n\r\n",
            "GET /health HTTP/1.1\r\nHost : a\r\n\r\n",
            "GET /health HTTP/1.1\r\nHost: a\r\n X-Fold: yes\r\n\r\n",
            "GET /health HTTP/1.1\r\nHost: a\nX: y\r\n\r\n",
            "GET /health HTTP/1.0\r\nHost: a\r\n\r\n",
            "GET  /health HTTP/1.1\r\nHost: a\r\n\r\n",
            "GET http://localhost/health HTTP/1.1\r\nHost: a\r\n\r\n",
            "GET /health#fragment HTTP/1.1\r\nHost: a\r\n\r\n",
            "GET /health HTTP/1.1\r\nHost: a,b\r\n\r\n",
            "GET /health HTTP/1.1\r\nHost: a\r\nX: \u{0}\r\n\r\n",
        ]
        for text in invalid { try assertHTTPError(Data(text.utf8)) }
        for value in ["", "+1", "-1", "1,1", "1 1", "1.0", "1e1"] {
            try assertHTTPError(Data("POST /v1/chat/completions HTTP/1.1\r\nHost: a\r\nContent-Length: \(value)\r\n\r\n".utf8))
        }
        try assertHTTPError(Data("POST /v1/chat/completions HTTP/1.1\r\nHost: a\r\n\r\n".utf8), status: 411)
        try assertHTTPError(Data("POST /v1/chat/completions HTTP/1.1\r\nHost: a\r\nContent-Length: 0\r\nContent-Type: text/plain\r\n\r\n".utf8), status: 415)
        try assertHTTPError(Data("POST /v1/chat/completions HTTP/1.1\r\nHost: a\r\nContent-Length: 1\r\nExpect: 100-continue\r\n\r\n".utf8), status: 417)
    }

    func testHeaderAndBodyLimitsAndTruncatedOrPipelinedRequests() throws {
        let prefix = "GET /health HTTP/1.1\r\nHost: a\r\nX-Pad: ", suffix = "\r\n\r\n"
        let padding = QwenHTTPRequestParser.maxHeaderBytes - prefix.utf8.count - suffix.utf8.count
        var exact = try QwenHTTPRequestParser()
        XCTAssertNotNil(try exact.feed(Data((prefix + String(repeating: "a", count: padding) + suffix).utf8)))
        try assertHTTPError(Data((prefix + String(repeating: "a", count: padding + 1) + suffix).utf8), status: 431)
        try assertHTTPError(Data(repeating: 65, count: 16 * 1024), status: 431)
        var small = try QwenHTTPRequestParser(maxBodyBytes: 8)
        XCTAssertEqual(try small.feed(request(Data(repeating: 65, count: 8)))?.body.count, 8)
        try assertHTTPError(request(Data(repeating: 65, count: 9)), status: 413, limit: 8)
        try assertHTTPError(Data("POST / HTTP/1.1\r\nHost: a\r\nContent-Length: 999999999999999999999999999999\r\n\r\n".utf8), status: 413)
        let complete = Data("GET /health HTTP/1.1\r\nHost: a\r\n\r\n".utf8)
        try assertHTTPError(complete + complete)
        var later = try QwenHTTPRequestParser()
        XCTAssertNotNil(try later.feed(complete))
        XCTAssertThrowsError(try later.feed(Data([13])))
        for partial in [Data(), Data("GET /health HTTP/1.1\r\n".utf8), request(Data([1,2,3])).dropLast()] {
            var parser = try QwenHTTPRequestParser()
            XCTAssertNil(try parser.feed(partial))
            XCTAssertThrowsError(try parser.finish())
        }
        XCTAssertThrowsError(try QwenHTTPRequestParser(maxBodyBytes: 0))
        XCTAssertThrowsError(try QwenHTTPRequestParser(maxBodyBytes: Int.max))
    }

    func testChatDefaultsAndSupportedControls() throws {
        let defaults = try QwenHTTPChatRequest.decode(body(), expectedModel: model)
        XCTAssertEqual(defaults.model, model); XCTAssertEqual(defaults.maxTokens, 128)
        XCTAssertFalse(defaults.stream); XCTAssertEqual(defaults.mtpDepth, 0)
        XCTAssertEqual(defaults.messages.first?.content, "你好😀")
        let messages = [["role": "system", "content": "规则"], ["role": "user", "content": ""],
                        ["role": "assistant", "content": "先前回答"], ["role": "user", "content": "继续"]]
        let supported = try QwenHTTPChatRequest.decode(body(["messages": messages, "stream": true,
            "max_tokens": 256, "mtp_depth": 2, "temperature": 0]), expectedModel: model)
        XCTAssertEqual(supported.messages.count, 4); XCTAssertTrue(supported.stream)
        XCTAssertEqual(supported.maxTokens, 256); XCTAssertEqual(supported.mtpDepth, 2)
    }

    func testExperimentalMTPOutputBudgetStaysWithinValidatedRange() throws {
        for budget in [1, 256] {
            let request = try QwenHTTPChatRequest.decode(body(["mtp_depth": 2, "max_tokens": budget]), expectedModel: model)
            XCTAssertEqual(request.maxTokens, budget); XCTAssertEqual(request.mtpDepth, 2)
        }
        for budget in [257, 4096] {
            XCTAssertThrowsError(try QwenHTTPChatRequest.decode(body(["mtp_depth": 2, "max_tokens": budget]), expectedModel: model)) {
                XCTAssertEqual(($0 as? QwenHTTPProtocolError)?.statusCode, 400)
            }
        }
        let mtpDefault = try QwenHTTPChatRequest.decode(body(["mtp_depth": 2]), expectedModel: model)
        XCTAssertEqual(mtpDefault.maxTokens, 128)
        let arFields: [[String: Any]] = [["max_tokens": 4096], ["max_tokens": 4096, "mtp_depth": 0]]
        for fields in arFields {
            let ar = try QwenHTTPChatRequest.decode(body(fields), expectedModel: model)
            XCTAssertEqual(ar.maxTokens, 4096); XCTAssertEqual(ar.mtpDepth, 0)
        }
    }

    func testChatRejectsUnsupportedFieldsRolesAndNonScalarContent() throws {
        for field in ["tools", "tool_choice", "top_p", "seed", "n", "stream_options", "extra_body"] {
            XCTAssertThrowsError(try QwenHTTPChatRequest.decode(body([field: NSNull()]), expectedModel: model))
        }
        let invalidMessages: [Any] = [[], [["role": "assistant", "content": "only assistant"]],
            [["role": "tool", "content": "tool"]], [["role": "user", "content": ["text": "image"]]],
            [["role": "user", "content": "ok", "name": "x"]],
            [["role": "user", "content": "x"], ["role": "system", "content": "late"]],
            [["role": "system", "content": "a"], ["role": "system", "content": "b"], ["role": "user", "content": "x"]]]
        for messages in invalidMessages {
            XCTAssertThrowsError(try QwenHTTPChatRequest.decode(body(["messages": messages]), expectedModel: model))
        }
        XCTAssertThrowsError(try QwenHTTPChatRequest.decode(body(), expectedModel: "another-model"))
        XCTAssertThrowsError(try QwenHTTPChatRequest.decode(Data("[]".utf8), expectedModel: model))
    }

    func testChatNumbersAreNotBooleansStringsFractionsOrOutOfRange() throws {
        let invalid: [String: [Any]] = ["max_tokens": [true, "128", 0, 4097, 1.5, NSNull()],
            "mtp_depth": [true, "2", 1, 3, -1], "temperature": [true, "0", 0.01, NSNull()],
            "stream": [0, 1, "true", NSNull()]]
        for (field, values) in invalid {
            for value in values {
                XCTAssertThrowsError(try QwenHTTPChatRequest.decode(body([field: value]), expectedModel: model), "\(field)=\(value)")
            }
        }
    }

    func testDuplicateEscapedKeysBadUTF8AndDeepJSONAreRejected() throws {
        let raw = [
            #"{"model":"Qwen-local","model":"Qwen-local","messages":[{"role":"user","content":"x"}]}"#,
            #"{"model":"Qwen-local","\u006dodel":"Qwen-local","messages":[{"role":"user","content":"x"}]}"#,
            #"{"model":"Qwen-local","messages":[{"role":"user","role":"assistant","content":"x"}]}"#,
            "{\"x\":" + String(repeating: "[", count: 17) + "0" + String(repeating: "]", count: 17) + "}",
            "", "{", "{\"model\":NaN}",
        ]
        for text in raw { XCTAssertThrowsError(try QwenHTTPChatRequest.decode(Data(text.utf8), expectedModel: model)) }
        XCTAssertThrowsError(try QwenHTTPChatRequest.decode(Data([0xFF]), expectedModel: model))
        let utf16 = try XCTUnwrap(String(data: body(), encoding: .utf8)?.data(using: .utf16LittleEndian))
        XCTAssertThrowsError(try QwenHTTPChatRequest.decode(utf16, expectedModel: model))
        // Escaping alone is valid: the preflight must not reject braces/commas
        // inside content or treat a content string as an object key.
        let text = "{\"role\":\"user\"},\\\"中文"
        XCTAssertEqual(try QwenHTTPChatRequest.decode(body(["messages": [["role": "user", "content": text]]]), expectedModel: model)
            .messages[0].content, text)
    }

    func testSSEFramesEscapeContentKeepIdentityAndCarryActualUsage() throws {
        let text = "中文😀\n\ndata: [DONE]\n\"\\"
        let values = [try QwenHTTPFrames.role(id: "chat-1", created: 123, model: model),
            try QwenHTTPFrames.content(id: "chat-1", created: 123, model: model, text: text),
            try QwenHTTPFrames.finish(id: "chat-1", created: 123, model: model, reason: "length", promptTokens: 11_123, completionTokens: 7)]
        var objects: [[String: Any]] = []
        for frame in values {
            let wire = try XCTUnwrap(String(data: frame, encoding: .utf8))
            XCTAssertTrue(wire.hasPrefix("data: ")); XCTAssertTrue(wire.hasSuffix("\n\n"))
            XCTAssertEqual(wire.components(separatedBy: "\n\n").count, 2)
            let json = Data(frame.dropFirst(6).dropLast(2))
            let object = try XCTUnwrap(JSONSerialization.jsonObject(with: json) as? [String: Any])
            XCTAssertEqual(object["id"] as? String, "chat-1"); XCTAssertEqual(object["created"] as? Int, 123)
            XCTAssertEqual(object["model"] as? String, model); objects.append(object)
        }
        let contentChoice = try XCTUnwrap((objects[1]["choices"] as? [[String: Any]])?.first)
        XCTAssertEqual((contentChoice["delta"] as? [String: String])?["content"], text)
        XCTAssertEqual(objects[2]["usage"] as? [String: Int], ["prompt_tokens": 11_123, "completion_tokens": 7, "total_tokens": 11_130])
        XCTAssertEqual(QwenHTTPFrames.done(), Data("data: [DONE]\n\n".utf8))
        let complete = try QwenHTTPFrames.completion(id: "chat-1", created: 123, model: model, text: text,
            reason: "stop", promptTokens: 11_123, completionTokens: 7)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: complete) as? [String: Any])
        XCTAssertEqual(object["object"] as? String, "chat.completion")
        XCTAssertEqual(object["usage"] as? [String: Int], objects[2]["usage"] as? [String: Int])
        XCTAssertThrowsError(try QwenHTTPFrames.finish(id: "x", created: 0, model: model, reason: "stop", promptTokens: -1, completionTokens: 1))
        XCTAssertThrowsError(try QwenHTTPFrames.finish(id: "x", created: 0, model: model, reason: "stop", promptTokens: Int.max, completionTokens: 1))
    }

    func testHTTPResponseByteFramingAndHeaderInjectionRejection() throws {
        let payload = Data("中文😀".utf8)
        let result = try QwenHTTPFrames.response(status: 200, contentType: "application/json", body: payload)
        let boundary = try XCTUnwrap(result.range(of: Data([13,10,13,10])))
        let header = try XCTUnwrap(String(data: result[..<boundary.lowerBound], encoding: .utf8))
        XCTAssertTrue(header.contains("Content-Length: \(payload.count)\r\n"))
        XCTAssertTrue(header.hasSuffix("Connection: close"))
        XCTAssertEqual(Data(result[boundary.upperBound...]), payload)
        XCTAssertThrowsError(try QwenHTTPFrames.response(status: 200, contentType: "application/json\r\nX: injected", body: payload))
        let stream = String(decoding: QwenHTTPFrames.streamHeader(), as: UTF8.self)
        XCTAssertTrue(stream.contains("Connection: close\r\n"))
        XCTAssertFalse(stream.contains("Content-Length")); XCTAssertFalse(stream.contains("Transfer-Encoding"))
        let models = try XCTUnwrap(JSONSerialization.jsonObject(with: QwenHTTPFrames.models(model: model, created: 123)) as? [String: Any])
        XCTAssertEqual((models["data"] as? [[String: Any]])?.first?["id"] as? String, model)
        let error = try XCTUnwrap(JSONSerialization.jsonObject(with: QwenHTTPFrames.error(message: "bad\ninput", code: "invalid_request")) as? [String: Any])
        XCTAssertEqual((error["error"] as? [String: Any])?["code"] as? String, "invalid_request")
    }
}
