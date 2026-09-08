import ANERunnerCore
import Foundation
import XCTest
@testable import ANERunnerGPU

/// CPU-only tokenization and metadata tests. Real-tokenizer cases read its
/// JSON/template files; they do not load weights, create tensors or run MLX.
final class QwenConversationPrefixPolicyTests: XCTestCase {
    func testLookupPublicationAndSystemProducerAreIndependent() throws {
        let plan = try QwenConversationPrefixPlan(promptTokenCount: 11_057,
            exactSystemPrefixTokenCount: 10_000)
        XCTAssertEqual(plan.lookupMaxTokens, 11_056)
        XCTAssertEqual(plan.publicationTokenCounts, [9_984, 10_816])
        XCTAssertEqual(plan.systemProducerTokenCount, 9_984)
        XCTAssertEqual(plan.prefillChunk, 416)
        // A different tail changes its continuation opportunity, not the
        // shared system producer's token boundary.
        let extended = try QwenConversationPrefixPlan(promptTokenCount: 12_305,
            exactSystemPrefixTokenCount: 10_000)
        XCTAssertNotEqual(extended.lookupMaxTokens, plan.lookupMaxTokens)
        XCTAssertNotEqual(extended.publicationTokenCounts.last, plan.publicationTokenCounts.last)
        XCTAssertEqual(extended.systemProducerTokenCount, plan.systemProducerTokenCount)
    }

    func testFinalTokenAndShortPromptsDoNotCreateSyntheticCheckpoints() throws {
        let short = try QwenConversationPrefixPlan(promptTokenCount: 416,
            exactSystemPrefixTokenCount: 415)
        XCTAssertEqual(short.lookupMaxTokens, 415)
        XCTAssertTrue(short.publicationTokenCounts.isEmpty)
        XCTAssertNil(short.systemProducerTokenCount)
        let aligned = try QwenConversationPrefixPlan(promptTokenCount: 417,
            exactSystemPrefixTokenCount: 416)
        XCTAssertEqual(aligned.publicationTokenCounts, [416])
        let single = try QwenConversationPrefixPlan(promptTokenCount: 1,
            exactSystemPrefixTokenCount: 0)
        XCTAssertEqual(single.lookupMaxTokens, 0)
        XCTAssertTrue(single.publicationTokenCounts.isEmpty)
        let otherGrid = try QwenConversationPrefixPlan(promptTokenCount: 1_537,
            exactSystemPrefixTokenCount: 1_024, prefillChunk: 512)
        XCTAssertEqual(otherGrid.publicationTokenCounts, [1_024, 1_536])
    }

    func testInvalidCountsFailWithoutArithmeticOverflow() throws {
        for (prompt, system, chunk) in [(0, 0, 416), (-1, 0, 416), (4, -1, 416),
                                        (4, 5, 416), (4, 0, 0), (4, 0, 513)] {
            XCTAssertThrowsError(try QwenConversationPrefixPlan(promptTokenCount: prompt,
                exactSystemPrefixTokenCount: system, prefillChunk: chunk))
        }
        let huge = try QwenConversationPrefixPlan(promptTokenCount: Int.max,
            exactSystemPrefixTokenCount: Int.max, prefillChunk: 512)
        XCTAssertEqual(huge.lookupMaxTokens, Int.max - 1)
        XCTAssertEqual(huge.publicationTokenCounts.count, 1)
        XCTAssertLessThan(try XCTUnwrap(huge.publicationTokenCounts.first), Int.max)
    }

    func testRealDifferentTailsKeepExactlyTheSameSystemProducerTokens() throws {
        let tokenizer = try QwenTokenizer(modelDirectory: GPUFixtureLocation.model())
        let system = ChatMessage(role: "system", content: String(repeating: "stable system rule.\n", count: 600))
        let a = try tokenizer.encodeConversation(messages: [system,
            .init(role: "user", content: String(repeating: "branch alpha data.\n", count: 40))])
        let b = try tokenizer.encodeConversation(messages: [system,
            .init(role: "user", content: String(repeating: "branch beta measurement.\n", count: 350))])
        let anchor = try XCTUnwrap(a.prefixPlan.systemProducerTokenCount)
        XCTAssertGreaterThan(anchor, 832)
        XCTAssertEqual(b.prefixPlan.systemProducerTokenCount, anchor)
        XCTAssertEqual(Array(a.tokens.prefix(anchor)), Array(b.tokens.prefix(anchor)))
        XCTAssertNotEqual(a.prefixPlan.publicationTokenCounts.last, b.prefixPlan.publicationTokenCounts.last)
        // Equal lengths alone would not identify a producer. These are the
        // exact bytes that the runtime must hash alongside its state namespace.
        XCTAssertNotEqual(a.tokens, b.tokens)
    }

    func testRealToolConversationExtendsReuseBeyondSystemAndRejectsEditedHistory() throws {
        let tokenizer = try QwenTokenizer(modelDirectory: GPUFixtureLocation.model())
        let tool = try QwenToolDefinition.decode(["type": "function", "function": ["name": "measure",
            "parameters": ["type": "object", "properties": ["city": ["type": "string"]], "required": ["city"]]]])
        let call = QwenToolCall(id: "call_1", name: "measure", arguments: .object(["city": .string("Beijing")]))
        let history: [ChatMessage] = [
            .init(role: "system", content: String(repeating: "Use the actual tool result.\n", count: 100)),
            .init(role: "user", content: String(repeating: "Historical document with measured data.\n", count: 220)),
            .init(role: "assistant", content: "", toolCalls: [call]),
            .init(role: "tool", content: String(repeating: "The measured value is 42.\n", count: 160), toolCallID: call.id)
        ]
        let before = try tokenizer.encodeConversation(messages: history, tools: [tool])
        let anchor = try XCTUnwrap(before.prefixPlan.systemProducerTokenCount)
        let continuation = try XCTUnwrap(before.prefixPlan.publicationTokenCounts.last)
        XCTAssertGreaterThan(continuation, anchor + 416)
        let index = try QwenPrefixCacheIndex<Int>(maxEntries: 4, maxBytes: 4)
        for boundary in before.prefixPlan.publicationTokenCounts {
            XCTAssertTrue(index.insert(tokens: Array(before.tokens.prefix(boundary)), namespace: "fixture-state-policy",
                value: boundary, logicalPayloadBytes: 1))
        }
        for query in ["Summarize the measurements.", "Compare the measurements with the report."] {
            let next = try tokenizer.encodeConversation(messages: history + [
                .init(role: "assistant", content: "The report is ready."), .init(role: "user", content: query)
            ], tools: [tool])
            let hit = index.peek(tokens: next.tokens, namespace: "fixture-state-policy",
                maxPrefixTokens: next.prefixPlan.lookupMaxTokens)
            XCTAssertEqual(hit?.prefixTokenCount, continuation)
        }
        var edited = history
        edited[1] = .init(role: "user", content: "A corrected document replaces the earlier request.")
        let changed = try tokenizer.encodeConversation(messages: edited, tools: [tool])
        XCTAssertEqual(index.peek(tokens: changed.tokens, namespace: "fixture-state-policy",
            maxPrefixTokens: changed.prefixPlan.lookupMaxTokens)?.prefixTokenCount, anchor)
        let renamed = try QwenToolDefinition.decode(["type": "function", "function": ["name": "new_measure",
            "parameters": ["type": "object", "properties": [:]]]])
        let changedTools = try tokenizer.encodeConversation(messages: Array(history.prefix(2)), tools: [renamed])
        XCTAssertNil(index.peek(tokens: changedTools.tokens, namespace: "fixture-state-policy",
            maxPrefixTokens: changedTools.prefixPlan.lookupMaxTokens))
    }

    func testRealDocumentWithoutSystemStillGetsBoundedContinuationCandidate() throws {
        let tokenizer = try QwenTokenizer(modelDirectory: GPUFixtureLocation.model())
        let result = try tokenizer.encodeConversation(messages: [
            .init(role: "user", content: String(repeating: "The repeated document contains a measurement.\n", count: 350))
        ])
        XCTAssertNil(result.prefixPlan.systemProducerTokenCount)
        XCTAssertEqual(result.prefixPlan.publicationTokenCounts.count, 1)
        XCTAssertGreaterThan(try XCTUnwrap(result.prefixPlan.publicationTokenCounts.first), 416)
        XCTAssertEqual(result.prefixPlan.lookupMaxTokens, result.tokens.count - 1)
    }

    func testRealBPEUsesCompleteEncodingInsteadOfConcatenatedTextFragments() throws {
        let tokenizer = try QwenTokenizer(modelDirectory: GPUFixtureLocation.model())
        let messages = [ChatMessage(role: "user", content: "hello")]
        let rendered = try tokenizer.renderChat(messages: messages)
        let word = try XCTUnwrap(rendered.range(of: "hello"))
        let split = rendered.index(word.lowerBound, offsetBy: 2)
        let independentlyEncoded = try tokenizer.encode(String(rendered[..<split])) +
            tokenizer.encode(String(rendered[split...]))
        let actual = try tokenizer.encodeConversation(messages: messages)
        XCTAssertNotEqual(actual.tokens, independentlyEncoded)
        XCTAssertEqual(actual.tokens, try tokenizer.encode(rendered))
        XCTAssertEqual(try tokenizer.decode(actual.tokens), rendered)
    }
}
