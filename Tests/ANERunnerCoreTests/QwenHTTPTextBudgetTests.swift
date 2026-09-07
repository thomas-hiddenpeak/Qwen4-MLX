import Foundation
import XCTest
@testable import ANERunnerCore

/// Actual CPU adapter/buffer contracts; no GPU, model, socket or timing hooks.
final class QwenHTTPTextBudgetTests: XCTestCase {
    func testTextLimitKeepsAcceptedBytesAndSpecificFailureAcrossSchedulerFailure() {
        var budget = QwenHTTPTextBudget(maxBytes: 6)
        XCTAssertTrue(budget.accept(byteCount: 3))
        XCTAssertTrue(budget.accept(byteCount: 3))
        XCTAssertFalse(budget.accept(byteCount: 1))
        XCTAssertEqual(budget.acceptedBytes, 6)
        XCTAssertEqual(budget.failure, .textLimit)
        // The scheduler's failed event no longer needs the original Error.
        XCTAssertEqual(budget.errorResponse(cancelled: false).code, "output_limit")
        XCTAssertFalse(budget.accept(byteCount: 0))
        XCTAssertEqual(budget.acceptedBytes, 6)
    }

    func testUnrelatedFailureAndCancellationRetainTheirCodes() {
        let budget = QwenHTTPTextBudget(maxBytes: 6144)
        XCTAssertNil(budget.failure)
        XCTAssertEqual(budget.errorResponse(cancelled: false).code, "generation_failed")
        XCTAssertEqual(budget.errorResponse(cancelled: true).code, "cancelled")
    }

    func testResponseLimitAndEncodingFailureAreDistinctAndFirstLocalFailureWins() {
        var limit = QwenHTTPTextBudget(maxBytes: 6144)
        limit.record(.responseLimit)
        limit.record(.encodingFailed)
        XCTAssertEqual(limit.failure, .responseLimit)
        XCTAssertEqual(limit.errorResponse(cancelled: false).code, "output_limit")
        var encoding = QwenHTTPTextBudget(maxBytes: 6144)
        encoding.record(.encodingFailed)
        encoding.record(.responseLimit)
        XCTAssertEqual(encoding.failure, .encodingFailed)
        XCTAssertEqual(encoding.errorResponse(cancelled: false).code, "encoding_failed")
    }

    func testBudgetAdditionCannotOverflowAndZeroLimitIsExact() {
        var huge = QwenHTTPTextBudget(maxBytes: Int.max)
        XCTAssertTrue(huge.accept(byteCount: Int.max))
        XCTAssertFalse(huge.accept(byteCount: 1))
        XCTAssertEqual(huge.acceptedBytes, Int.max)
        var zero = QwenHTTPTextBudget(maxBytes: 0)
        XCTAssertTrue(zero.accept(byteCount: 0))
        XCTAssertFalse(zero.accept(byteCount: 1))
        XCTAssertEqual(zero.acceptedBytes, 0)
    }

    func testRealJSONEscapingCanExceedFinalResponseLimitWithoutExceedingTextLimit() throws {
        let text = String(repeating: "\"", count: 4096)
        var budget = QwenHTTPTextBudget(maxBytes: 6144)
        XCTAssertTrue(budget.accept(byteCount: text.utf8.count))
        let body = try QwenHTTPFrames.completion(id: "chatcmpl-fixed", created: 1, model: "model", text: text,
            reason: "length", promptTokens: 10, completionTokens: 4096)
        let response = try QwenHTTPFrames.response(status: 200, contentType: "application/json", body: body)
        XCTAssertGreaterThan(response.count, 8192)
        let overflow = try QwenHTTPFrames.response(status: 500, contentType: "application/json",
            body: QwenHTTPFrames.error(message: "Output exceeds configured byte limit", code: "output_limit"))
        let buffer = try QwenSSEOutputBuffer(limits: .init(maxBytes: 8193, maxEvents: 2, terminalReserveBytes: 8192),
            overflowFrame: overflow)
        XCTAssertEqual(buffer.finish(.completed, frame: response).status, .invalidFrame)
        XCTAssertNil(buffer.snapshot().outcome)
        budget.record(.responseLimit)
        let failure = budget.errorResponse(cancelled: false)
        let fallback = try QwenHTTPFrames.response(status: 500, contentType: "application/json",
            body: QwenHTTPFrames.error(message: failure.message, code: failure.code))
        XCTAssertEqual(buffer.finish(.failed, frame: fallback).status, .accepted)
        let sent = try XCTUnwrap(buffer.beginSend())
        XCTAssertEqual(sent.data, fallback)
        XCTAssertTrue(sent.isTerminal)
        _ = buffer.acknowledgeSend(sent.id, succeeded: true)
        XCTAssertTrue(buffer.snapshot().isDrained)
    }

    func testSSEOverflowWinsOverLateAdapterFailureAndDisconnectRetainsLease() throws {
        let overflow = try QwenHTTPFrames.sseError(message: "Client output buffer limit exceeded", code: "slow_consumer")
            + QwenHTTPFrames.done()
        let buffer = try QwenSSEOutputBuffer(limits: .init(maxBytes: 512, maxEvents: 4, terminalReserveBytes: 256),
            overflowFrame: overflow)
        XCTAssertEqual(buffer.enqueue(Data(repeating: 65, count: 256)).status, .accepted)
        let lease = try XCTUnwrap(buffer.beginSend())
        XCTAssertEqual(buffer.enqueue(Data([66])).status, .overflow)
        var budget = QwenHTTPTextBudget(maxBytes: 6144)
        budget.record(.responseLimit)
        let failure = budget.errorResponse(cancelled: false)
        let late = try QwenHTTPFrames.sseError(message: failure.message, code: failure.code) + QwenHTTPFrames.done()
        XCTAssertEqual(buffer.finish(.failed, frame: late).status, .alreadyTerminal)
        XCTAssertEqual(buffer.snapshot().outcome, .slowConsumer)
        _ = buffer.disconnect()
        XCTAssertEqual(buffer.snapshot().outcome, .slowConsumer)
        XCTAssertEqual(buffer.snapshot().inFlightBytes, 256)
        _ = buffer.acknowledgeSend(lease.id, succeeded: false)
        XCTAssertEqual(buffer.snapshot().bufferedBytes, 0)
        XCTAssertTrue(buffer.snapshot().isDrained)
    }

    func testDisconnectedBufferCannotBeRewrittenByOutputLimit() throws {
        let error = try QwenHTTPFrames.error(message: "Output exceeds configured byte limit", code: "output_limit")
        let buffer = try QwenSSEOutputBuffer(limits: .init(maxBytes: 1024, terminalReserveBytes: 512), overflowFrame: error)
        _ = buffer.disconnect()
        var budget = QwenHTTPTextBudget(maxBytes: 0)
        XCTAssertFalse(budget.accept(byteCount: 1))
        XCTAssertEqual(budget.errorResponse(cancelled: false).code, "output_limit")
        XCTAssertEqual(buffer.finish(.failed, frame: error).status, .alreadyTerminal)
        XCTAssertEqual(buffer.snapshot().outcome, .disconnected)
        XCTAssertNil(buffer.beginSend())
    }
}
