import Dispatch
import Foundation
import XCTest
@testable import ANERunnerGPU

/// Pure CPU contract checks. No QwenModel, MLX tensor or GPU context is created.
final class QwenGenerationTests: XCTestCase {
    private func configuration() throws -> QwenConfiguration {
        try QwenConfiguration(modelDirectory: GPUFixtureLocation.model())
    }

    func testDefaultsAndEverySupportedDraftDepthValidate() throws {
        let c = try configuration()
        let request = QwenGenerationRequest(tokens: [42])
        XCTAssertEqual(request.maxTokens, 128)
        XCTAssertEqual(request.contextLimit, 16_384)
        XCTAssertEqual(request.prefillChunk, 416)
        XCTAssertEqual(request.mtpDepth, 0)
        XCTAssertEqual(request.verification, .scalar)
        XCTAssertNil(request.draftHistoryTokens)
        XCTAssertEqual(request.prefillEvaluateEveryLayers, 4)
        XCTAssertEqual(request.verificationEvaluateEveryLayers, 4)
        XCTAssertEqual(request.decodeMode, .reference)
        try request.validate(configuration: c)
        for depth in 0...4 {
            try QwenGenerationRequest(tokens: [0, Int32(c.vocabularySize - 1)], mtpDepth: depth,
                                      verification: .scalar).validate(configuration: c)
        }
    }

    func testInvalidTokensAreRejectedBeforeInference() throws {
        let c = try configuration()
        let bad: [[Int32]] = [[], [-1], [Int32.max], [Int32(c.vocabularySize)]]
            + [248053, 248054, 248055, 248056, 248057, 248070, 248071, 248076].map { [Int32($0)] }
        for tokens in bad {
            XCTAssertThrowsError(try QwenGenerationRequest(tokens: tokens).validate(configuration: c)) {
                guard case QwenGenerationError.invalidRequest = $0 else { return XCTFail("Unexpected error: \($0)") }
            }
        }
    }

    func testBudgetChunkAndDepthBounds() throws {
        let c = try configuration()
        let invalid = [
            QwenGenerationRequest(tokens: [42], maxTokens: 0),
            QwenGenerationRequest(tokens: [42], maxTokens: -1),
            QwenGenerationRequest(tokens: [42], maxTokens: Int.max),
            QwenGenerationRequest(tokens: [42], contextLimit: 0),
            QwenGenerationRequest(tokens: [42], contextLimit: -1),
            QwenGenerationRequest(tokens: [42], contextLimit: c.maximumPositions + 1),
            QwenGenerationRequest(tokens: [42], prefillChunk: 0),
            QwenGenerationRequest(tokens: [42], prefillChunk: 513),
            QwenGenerationRequest(tokens: [42], mtpDepth: -1),
            QwenGenerationRequest(tokens: [42], mtpDepth: 5),
            QwenGenerationRequest(tokens: [42], draftHistoryTokens: 0),
            QwenGenerationRequest(tokens: [42], draftHistoryTokens: -1),
            QwenGenerationRequest(tokens: [42], draftHistoryTokens: Int.max),
            QwenGenerationRequest(tokens: [42], draftHistoryTokens: c.maximumPositions + 1),
            QwenGenerationRequest(tokens: [42], prefillEvaluateEveryLayers: 0),
            QwenGenerationRequest(tokens: [42], prefillEvaluateEveryLayers: 49),
            QwenGenerationRequest(tokens: [42], verificationEvaluateEveryLayers: 0),
            QwenGenerationRequest(tokens: [42], verificationEvaluateEveryLayers: 49),
            QwenGenerationRequest(tokens: [42], mtpDepth: 2,
                verification: .batchedScalarLinear, decodeMode: .elementwise),
            QwenGenerationRequest(tokens: [42], mtpDepth: 2,
                verification: .batchedTokenMoE, decodeMode: .elementwise)
        ]
        for request in invalid { XCTAssertThrowsError(try request.validate(configuration: c)) }
        try QwenGenerationRequest(tokens: [42, 43], maxTokens: 2, contextLimit: 4,
                                  prefillChunk: 1).validate(configuration: c)
        XCTAssertThrowsError(try QwenGenerationRequest(tokens: [42, 43], maxTokens: 3,
                                                       contextLimit: 4).validate(configuration: c))
        try QwenGenerationRequest(tokens: [42], maxTokens: c.maximumPositions - 1,
                                  contextLimit: c.maximumPositions, prefillChunk: 512).validate(configuration: c)
        try QwenGenerationRequest(tokens: [42], mtpDepth: 2, verification: .batchedScalarLinear,
                                  draftHistoryTokens: 1024).validate(configuration: c)
        try QwenGenerationRequest(tokens: [42], mtpDepth: 2, verification: .batchedTokenMoE,
                                  draftHistoryTokens: 1024).validate(configuration: c)
        try QwenGenerationRequest(tokens: [42], mtpDepth: 2,
            verification: .batchedScalarLinear, prefillEvaluateEveryLayers: 48,
            verificationEvaluateEveryLayers: 4).validate(configuration: c)
    }

    func testCancellationIsIdempotentAndThreadSafe() throws {
        let cancellation = QwenCancellation()
        XCTAssertFalse(cancellation.isCancelled)
        try cancellation.check()
        DispatchQueue.concurrentPerform(iterations: 32) { _ in cancellation.cancel() }
        XCTAssertTrue(cancellation.isCancelled)
        XCTAssertThrowsError(try cancellation.check()) {
            XCTAssertEqual($0 as? QwenGenerationError, .cancelled)
        }
    }

    func testAdmissionRejectsReentryAndReleasesAfterSuccess() throws {
        let gate = QwenGenerationGate()
        try gate.withExclusiveAccess {
            XCTAssertThrowsError(try gate.withExclusiveAccess {}) {
                XCTAssertEqual($0 as? QwenGenerationError, .busy)
            }
        }
        XCTAssertEqual(try gate.withExclusiveAccess { 42 }, 42)
    }

    func testAdmissionReleasesAfterCallbackOrCancellationFailure() throws {
        enum CallbackError: Error { case failed }
        let gate = QwenGenerationGate()
        XCTAssertThrowsError(try gate.withExclusiveAccess { throw CallbackError.failed })
        XCTAssertEqual(try gate.withExclusiveAccess { 1 }, 1)
        let cancellation = QwenCancellation()
        cancellation.cancel()
        XCTAssertThrowsError(try gate.withExclusiveAccess { try cancellation.check() }) {
            XCTAssertEqual($0 as? QwenGenerationError, .cancelled)
        }
        XCTAssertEqual(try gate.withExclusiveAccess { 2 }, 2)
    }

    func testEOSPolicyIncludesBothActualStopMarkers() throws {
        let tokenizer = try QwenTokenizer(modelDirectory: GPUFixtureLocation.model())
        XCTAssertTrue(tokenizer.eosTokenIDs.contains(248044))
        XCTAssertTrue(tokenizer.eosTokenIDs.contains(248046))
    }

    func testHandoffIsConsumedOnlyOnceAndDiscardIsFinal() throws {
        let pending = QwenSingleUseHandoff([Int32(42)])
        XCTAssertTrue(pending.isReady)
        XCTAssertEqual(try pending.take(), [42])
        XCTAssertFalse(pending.isReady)
        XCTAssertThrowsError(try pending.take())
        let discarded = QwenSingleUseHandoff(7)
        discarded.discard()
        discarded.discard()
        XCTAssertFalse(discarded.isReady)
        XCTAssertThrowsError(try discarded.take())
    }

    func testClaimedHandoffReleasesPayloadAfterConsumerFailure() throws {
        final class Payload {}
        enum Failure: Error { case consumer }
        var object: Payload? = Payload()
        weak let observer = object
        let handoff = QwenSingleUseHandoff(object!)
        object = nil
        XCTAssertNotNil(observer)
        func consumeThenFail() throws {
            let value = try handoff.take()
            withExtendedLifetime(value) {}
            throw Failure.consumer
        }
        XCTAssertThrowsError(try consumeThenFail())
        XCTAssertNil(observer)
        XCTAssertFalse(handoff.isReady)
    }
}
