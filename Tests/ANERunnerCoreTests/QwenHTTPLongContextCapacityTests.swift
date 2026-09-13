import Foundation
import XCTest
@testable import ANERunnerCore

final class QwenHTTPLongContextCapacityTests: XCTestCase {
    func testExistingDefaultsAndExplicitLongProfile() throws {
        let old = try QwenHTTPServiceCapacity()
        XCTAssertEqual(old.contextLimit, 16_384)
        XCTAssertEqual(old.maxReservedTokens, 32_768)
        XCTAssertEqual(old.maxResidentSequences, 2)
        XCTAssertEqual(old.maxBodyBytes, 262_144)
        XCTAssertEqual(old.connectionDeadlineSeconds, 300)
        let long = try QwenHTTPServiceCapacity(contextLimit: 262_144,
            maxReservedTokens: 262_144, maxResidentSequences: 1,
            maxBodyBytes: 8 * 1024 * 1024, connectionDeadlineSeconds: 1800)
        try long.validateModelMaximumPositions(262_144)
        XCTAssertThrowsError(try long.validateModelMaximumPositions(131_072))
        // A larger announced context cannot silently retain the old32K
        // scheduler ceiling and reject a legal full request as queue overload.
        XCTAssertThrowsError(try QwenHTTPServiceCapacity(contextLimit: 262_144))
        XCTAssertThrowsError(try QwenHTTPServiceCapacity(contextLimit: 262_145, maxReservedTokens: 262_145))
        XCTAssertThrowsError(try QwenHTTPServiceCapacity(maxBodyBytes: Int.max))
        XCTAssertThrowsError(try QwenHTTPServiceCapacity(connectionDeadlineSeconds: 0))
    }

    func testLargeFragmentedHTTPBodyAndChatDecoderUseTheExplicitBound() throws {
        let content = String(repeating: "a", count: 1_121_736)
        let body = try JSONSerialization.data(withJSONObject: ["model": "Qwen-local",
            "messages": [["role": "user", "content": content]], "max_tokens": 2])
        let header = Data("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: \(body.count)\r\n\r\n".utf8)
        let long = try QwenHTTPServiceCapacity(contextLimit: 262_144, maxReservedTokens: 262_144,
            maxResidentSequences: 1, maxBodyBytes: 8 * 1024 * 1024, connectionDeadlineSeconds: 1800)
        var parser = try QwenHTTPRequestParser(maxBodyBytes: long.maxBodyBytes)
        XCTAssertNil(try parser.feed(header))
        var parsed: QwenHTTPRequest?
        for start in stride(from: 0, to: body.count, by: 16_384) {
            let end = min(body.count, start + 16_384)
            let value = try parser.feed(body[start..<end])
            if let value { XCTAssertNil(parsed); parsed = value }
        }
        try parser.finish()
        let request = try XCTUnwrap(parsed)
        XCTAssertEqual(request.body, body)
        let chat = try QwenHTTPChatRequest.decode(request.body, expectedModel: "Qwen-local")
        XCTAssertEqual(chat.messages.first?.content, content)
        XCTAssertEqual(chat.maxTokens, 2)
        // Old/default and exact-minus-one byte limits still reject in the
        // header, before copying a large body or tokenizing it.
        for maximum in [262_144, body.count - 1] {
            var bounded = try QwenHTTPRequestParser(maxBodyBytes: maximum)
            XCTAssertThrowsError(try bounded.feed(header)) {
                XCTAssertEqual(($0 as? QwenHTTPProtocolError)?.statusCode, 413)
            }
        }
        var exact = try QwenHTTPRequestParser(maxBodyBytes: body.count)
        XCTAssertNil(try exact.feed(header))
        XCTAssertEqual(try exact.feed(body)?.body.count, body.count)
    }
}
