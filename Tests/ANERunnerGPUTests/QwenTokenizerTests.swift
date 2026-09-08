import ANERunnerCore
import Foundation
import XCTest
@testable import ANERunnerGPU

final class QwenTokenizerTests: XCTestCase {
    /// Goldens were captured from the pinned author's /tokenize endpoint,
    /// independently of this Swift BPE. The report retains source hashes.
    func testAuthorTokenizationGoldensAndRoundTrips() throws {
        let tokenizer = try QwenTokenizer(modelDirectory: GPUFixtureLocation.model())
        let reportURL = GPUFixtureLocation.package.appendingPathComponent("results/gpu-tokenizer-validation.json")
        guard FileManager.default.fileExists(atPath: reportURL.path) else { throw XCTSkip("Author tokenization goldens unavailable") }
        let report = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: reportURL)) as? [String: Any])
        let rows = try XCTUnwrap(report["rows"] as? [[String: Any]])
        XCTAssertEqual(rows.count, 10)
        for (index, row) in rows.enumerated() {
            let text = try XCTUnwrap(row["text"] as? String)
            let expected = try XCTUnwrap(row["author_ids"] as? [Int]).map(Int32.init)
            let actual = try tokenizer.encode(text)
            XCTAssertEqual(actual, expected, "Author golden \(index)")
            XCTAssertEqual(try tokenizer.decode(actual), text, "Byte roundtrip \(index)")
        }
    }

    func testActualNoThinkingTemplateAndCapturedTokens() throws {
        let tokenizer = try QwenTokenizer(modelDirectory: GPUFixtureLocation.model())
        let text = "请用一句中文解释，为什么 SSD 上的 n-gram 嵌入能够按需读取。"
        let rendered = try tokenizer.renderChat(messages: [ChatMessage(role: "user", content: " \n" + text + "\n ")])
        let expected = "<|im_start|>user\n" + text + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        XCTAssertEqual(rendered, expected)
        XCTAssertEqual(try tokenizer.encode(rendered), [248045, 846, 198, 139054, 104213, 99986, 98682, 3709, 98832, 35160, 220, 97015, 307, 12, 1466, 220, 105264, 96936, 133713, 111388, 1710, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271])
        XCTAssertThrowsError(try tokenizer.renderChat(messages: []))
        XCTAssertThrowsError(try tokenizer.renderChat(messages: [ChatMessage(role: "user", content: "x"), ChatMessage(role: "system", content: "x")]))
        XCTAssertThrowsError(try tokenizer.renderChat(messages: [ChatMessage(role: "user", content: "x"), ChatMessage(role: "tool", content: "x")]))
    }

    func testNFCIsExplicitAndSpecialFilteringUsesDeclaredFlags() throws {
        let model = try GPUFixtureLocation.model()
        let compatible = try QwenTokenizer(modelDirectory: model)
        let nfc = try QwenTokenizer(modelDirectory: model, normalization: .modelNFC)
        XCTAssertEqual(compatible.reservedOutputTokenIDs, Set<Int32>([248047, 248048, 248049, 248050, 248051, 248052, 248055, 248070, 248071, 248072, 248073, 248074, 248075, 248076]))
        XCTAssertTrue(compatible.reservedOutputTokenIDs.isDisjoint(with: compatible.eosTokenIDs))
        XCTAssertNotEqual(try compatible.encode("e\u{301}"), try compatible.encode("é"))
        XCTAssertEqual(try nfc.encode("e\u{301}"), try nfc.encode("é"))
        // <think> is an added token with special=false, unlike im_start/end.
        XCTAssertEqual(try compatible.decode([248045, 248068, 248046], skipSpecialTokens: true), "<think>")
        XCTAssertThrowsError(try compatible.decode([Int32.max]))
    }
    func testToolTemplateAndSystemPrefixUseExactCompleteTokenPrefix() throws {
        let tokenizer = try QwenTokenizer(modelDirectory: GPUFixtureLocation.model())
        let tool = try QwenToolDefinition.decode(["type": "function", "function": ["name": "weather",
            "parameters": ["type": "object", "properties": ["city": ["type": "string"]], "required": ["city"]]]])
        let history = [ChatMessage(role: "system", content: String(repeating: "相同系统规则。", count: 180)),
                       ChatMessage(role: "user", content: "北京天气")]
        let rendered = try tokenizer.renderChat(messages: history, tools: [tool])
        let tokens = try tokenizer.encode(rendered)
        let count = try tokenizer.systemPrefixTokenCount(messages: history, tools: [tool], fullTokens: tokens)
        XCTAssertGreaterThan(count, 416); XCTAssertLessThan(count, tokens.count)
        XCTAssertEqual(Array(tokens.prefix(count)), Array(try tokenizer.encode(
            QwenToolChatTemplate.systemPrefix(system: history[0].content, tools: [tool])).prefix(count)))
        XCTAssertEqual(try tokenizer.renderChat(messages: history, tools: []), try tokenizer.renderChat(messages: history))
        XCTAssertEqual(try tokenizer.systemPrefixTokenCount(messages: [.init(role: "user", content: "x")],
            tools: [], fullTokens: tokens), 0)
        let altered = try tokenizer.encode("different " + rendered)
        XCTAssertEqual(try tokenizer.systemPrefixTokenCount(messages: history, tools: [tool], fullTokens: altered), 0)
    }

}
