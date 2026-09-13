import ANERunnerCore
import XCTest
@testable import ANERunnerGPU

/// The production queue engine receives the same explicit service token quota;
/// no model, Tensor or device work is constructed by these admission tests.
final class QwenLongContextSchedulerTests: XCTestCase {
    func testFullContextReservationAndCancellationUseConfiguredQuota() throws {
        let modelConfiguration = try QwenConfiguration(modelDirectory: GPUFixtureLocation.model())
        let capacity = try QwenHTTPServiceCapacity(contextLimit: 262_144,
            maxReservedTokens: 262_144, maxResidentSequences: 1,
            maxBodyBytes: 8 * 1024 * 1024, connectionDeadlineSeconds: 1800)
        var deviceCalls = 0
        enum Unexpected: Error { case deviceWork }
        let core = try QwenLocalSchedulerCore<Int>(limits: .init(
            maxResidentTokens: capacity.maxReservedTokens,
            maxResidentSequences: capacity.maxResidentSequences), backend: .init(
                validate: { try $0.validate(configuration: modelConfiguration) },
                prefill: { _, _ in deviceCalls += 1; throw Unexpected.deviceWork },
                decode: { _, _, _ in deviceCalls += 1; throw Unexpected.deviceWork },
                discard: { _ in deviceCalls += 1 }))
        let request = QwenGenerationRequest(tokens: Array(repeating: Int32(42), count: 262_142),
            maxTokens: 2, contextLimit: capacity.contextLimit)
        let id = try core.submit(request)
        XCTAssertEqual(core.snapshot().reservedTokens, 262_144)
        // Context validation runs before queue quota admission, so an actual
        // over-context request is invalid input even while quota is full.
        XCTAssertThrowsError(try core.submit(.init(tokens: Array(repeating: 42, count: 262_143),
            maxTokens: 2, contextLimit: capacity.contextLimit))) {
            guard case QwenGenerationError.invalidRequest = $0 else {
                return XCTFail("Expected context rejection before token-quota rejection: \($0)")
            }
        }
        XCTAssertThrowsError(try core.submit(.init(tokens: [42], maxTokens: 1,
            contextLimit: capacity.contextLimit))) {
            XCTAssertEqual($0 as? QwenLocalScheduler.Error, .overBudget)
        }
        XCTAssertEqual(core.snapshot().reservedTokens, 262_144)
        XCTAssertEqual(try core.cancel(id)?.kind, .cancelled)
        XCTAssertEqual(core.snapshot().reservedTokens, 0)
        XCTAssertEqual(core.snapshot().residentSequences, 0)
        _ = try core.submit(request)
        _ = try core.discardAll()
        XCTAssertEqual(core.snapshot().reservedTokens, 0)
        XCTAssertEqual(deviceCalls, 0)
    }

    func testFullContextConversationCheckpointKeepsOriginal416Grid() throws {
        let plan = try QwenConversationPrefixPlan(promptTokenCount: 262_142,
            exactSystemPrefixTokenCount: 11_056, prefillChunk: 416)
        XCTAssertEqual(plan.lookupMaxTokens, 262_141)
        XCTAssertEqual(plan.publicationTokenCounts, [10_816, 262_080])
        XCTAssertEqual(plan.systemProducerTokenCount, 10_816)
        XCTAssertEqual(262_142 - plan.publicationTokenCounts.last!, 62)
    }
}
