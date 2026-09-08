import Foundation
import XCTest
@testable import ANERunnerCore
@testable import ANERunnerGPU

/// CPU policy, ownership, and logical-capacity tests. No model, weights,
/// Tensor, GPU stream, tokenizer fixture, or network service is loaded.
final class QwenConversationCacheRuntimeTests: XCTestCase {
    func testRequestPlanValidationAndLegacyNormalization() throws {
        let tokens = Array(repeating: Int32(42), count: 11_057)
        let plan = try QwenConversationPrefixPlan(promptTokenCount: tokens.count, exactSystemPrefixTokenCount: 10_000)
        let request = QwenGenerationRequest(tokens: tokens, prefixCachePlan: plan)
        try request.validatePrefixCachePolicy()
        XCTAssertEqual(request.prefixCachePolicy?.lookupMaximum, 11_056)
        XCTAssertEqual(request.prefixCachePolicy?.publicationBoundaries, [9_984, 10_816])
        XCTAssertEqual(request.prefixCachePolicy?.systemProducerBoundary, 9_984)

        let legacy = QwenGenerationRequest(tokens: tokens, prefixCacheMaxTokens: 10_000)
        try legacy.validatePrefixCachePolicy()
        XCTAssertEqual(legacy.prefixCachePolicy?.lookupMaximum, 9_984)
        XCTAssertEqual(legacy.prefixCachePolicy?.publicationBoundaries, [9_984])
        XCTAssertEqual(legacy.prefixCacheNamespace(accumulation: "reference"),
                       request.prefixCacheNamespace(accumulation: "reference"))
        XCTAssertThrowsError(try QwenGenerationRequest(tokens: tokens,
            prefixCacheMaxTokens: 0, prefixCachePlan: plan).validatePrefixCachePolicy())
        XCTAssertThrowsError(try QwenGenerationRequest(tokens: tokens,
            prefillChunk: 512, prefixCachePlan: plan).validatePrefixCachePolicy())
        XCTAssertThrowsError(try QwenGenerationRequest(tokens: Array(tokens.dropLast()),
            prefixCachePlan: plan).validatePrefixCachePolicy())

        let mtp = QwenGenerationRequest(tokens: tokens, mtpDepth: 1, prefixCachePlan: plan)
        XCTAssertNoThrow(try mtp.validatePrefixCachePolicy())
        XCTAssertNil(mtp.prefixCachePolicy)
        let shortPlan = try QwenConversationPrefixPlan(promptTokenCount: 416, exactSystemPrefixTokenCount: 400)
        XCTAssertNil(QwenGenerationRequest(tokens: Array(tokens.prefix(416)), prefixCachePlan: shortPlan).prefixCachePolicy)
    }

    func testCheckpointIdentityUsesCompleteTokensAndNotTheRequestTail() {
        let system = Array(repeating: Int32(42), count: 9_984)
        let a = system + Array(repeating: Int32(100), count: 1_000)
        let b = system + Array(repeating: Int32(101), count: 1_500)
        let anchorA = QwenPrefixCheckpoint(tokens: a, namespace: "same", boundary: 9_984)
        let anchorB = QwenPrefixCheckpoint(tokens: b, namespace: "same", boundary: 9_984)
        XCTAssertEqual(anchorA, anchorB)
        let tailA = QwenPrefixCheckpoint(tokens: a, namespace: "same", boundary: 10_816)
        let tailB = QwenPrefixCheckpoint(tokens: b, namespace: "same", boundary: 10_816)
        XCTAssertNotEqual(tailA.key, tailB.key)
        XCTAssertNotEqual(anchorA.key, tailA.key)
        XCTAssertNotEqual(anchorA.key, QwenPrefixCheckpoint(tokens: a, namespace: "different", boundary: 9_984).key)
        XCTAssertEqual(tailA.key,
            QwenPrefixCheckpoint(tokens: a + [200, 201], namespace: "same", boundary: 10_816).key)
    }

    func testProducerTakeoverAndClearRejectLateOwnerRelease() {
        let registry = QwenPrefixProducerRegistry(), a = UUID(), b = UUID(), c = UUID()
        XCTAssertTrue(registry.claim(key: "system", identity: a))
        XCTAssertFalse(registry.claim(key: "system", identity: b))
        registry.release(key: "system", identity: b)
        XCTAssertEqual(registry.owner(for: "system"), a)
        registry.release(key: "system", identity: a)
        XCTAssertTrue(registry.claim(key: "system", identity: b))
        registry.release(key: "system", identity: a)
        XCTAssertEqual(registry.owner(for: "system"), b)
        registry.removeAll()
        XCTAssertTrue(registry.claim(key: "system", identity: c))
        registry.release(key: "system", identity: b)
        XCTAssertEqual(registry.owner(for: "system"), c)
        XCTAssertEqual(registry.count, 1)
    }

    func testSystemPublicationReleaseDoesNotReleaseSubsequentTail() {
        let registry = QwenPrefixProducerRegistry(), a = UUID(), b = UUID()
        XCTAssertTrue(registry.claim(key: "system", identity: a))
        registry.release(key: "system", identity: a)
        XCTAssertTrue(registry.claim(key: "tail-a", identity: a))
        XCTAssertTrue(registry.claim(key: "tail-b", identity: b))
        registry.release(key: "system", identity: a)
        XCTAssertEqual(registry.owner(for: "tail-a"), a)
        XCTAssertEqual(registry.owner(for: "tail-b"), b)
        registry.release(key: "tail-a", identity: a)
        XCTAssertEqual(registry.owner(for: "tail-b"), b)
    }

    func testReadFenceRequiresImportCompletionAfterIO() {
        let fence = QwenPrefixDiskReadFence()
        fence.finishIO()
        XCTAssertFalse(fence.isComplete)
        fence.finishIO() // Idempotent completion does not release the consumer.
        XCTAssertFalse(fence.isComplete)
        fence.finishConsumer()
        XCTAssertTrue(fence.isComplete)
    }

    func testAbandonedReadFenceKeepsJobWorkspaceUntilLateCallbackOwnerFinishes() throws {
        let budget = try QwenStateBudget(maxBytes: 64), fence = QwenPrefixDiskReadFence()
        var request: QwenPrefixDiskRead? = QwenPrefixDiskRead(
            lease: try XCTUnwrap(budget.reserve(bytes: 32, kind: .workspace)))
        var completion: (@Sendable () -> Void)? = { [read = request!, fence] in
            defer { fence.finishIO(); withExtendedLifetime(read) {} }
            read.complete(nil)
        }
        fence.finishConsumer() // Request cancelled; the OS operation is independent.
        request = nil
        XCTAssertFalse(fence.isComplete)
        XCTAssertEqual(budget.statistics.workspaceBytes, 32)
        completion?()
        XCTAssertTrue(fence.isComplete)
        XCTAssertEqual(budget.statistics.workspaceBytes, 32)
        completion = nil
        XCTAssertEqual(budget.statistics.workspaceBytes, 0)
        XCTAssertEqual(budget.statistics.currentLeases, 0)
    }

    func testDefaultRAMRetainsSharedSystemWhenLongTailCannotCoexist() throws {
        let mib = 1_048_576
        let index = try QwenPrefixCacheIndex<String>(maxEntries: 8, maxBytes: 512 * mib)
        let system = Array(repeating: Int32(42), count: 9_984), tail = system + Array(repeating: Int32(90), count: 832)
        XCTAssertTrue(index.insert(tokens: system, namespace: "model", value: "system", logicalPayloadBytes: 350 * mib))
        let before = index.statistics
        XCTAssertFalse(index.canInsertAlongsidePrefix(tokens: tail, namespace: "model",
            logicalPayloadBytes: 350 * mib, prefixTokenCount: system.count))
        XCTAssertFalse(index.insert(tokens: tail, namespace: "model", value: "tail",
            logicalPayloadBytes: 350 * mib, retainingPrefixTokens: system.count))
        XCTAssertEqual(index.statistics, before)
        let otherUser = system + Array(repeating: Int32(91), count: 1_248)
        XCTAssertEqual(index.lookup(tokens: otherUser, namespace: "model")?.value, "system")
        XCTAssertEqual(index.statistics.logicalPayloadBytes, 350 * mib)
    }

    func testSoftRetentionEvictsOtherEntriesAndRemainsExplicitlyEvictable() throws {
        let index = try QwenPrefixCacheIndex<String>(maxEntries: 3, maxBytes: 100)
        XCTAssertTrue(index.insert(tokens: [1, 2], namespace: "model", value: "system", logicalPayloadBytes: 30))
        XCTAssertTrue(index.insert(tokens: [9], namespace: "other", value: "other", logicalPayloadBytes: 30))
        XCTAssertTrue(index.canInsertAlongsidePrefix(tokens: [1, 2, 3, 4], namespace: "model",
            logicalPayloadBytes: 50, prefixTokenCount: 2))
        XCTAssertTrue(index.insert(tokens: [1, 2, 3, 4], namespace: "model", value: "tail",
            logicalPayloadBytes: 50, retainingPrefixTokens: 2))
        XCTAssertNotNil(index.peek(tokens: [1, 2], namespace: "model"))
        XCTAssertNil(index.peek(tokens: [9], namespace: "other"))
        XCTAssertEqual(index.statistics.logicalPayloadBytes, 80)
        XCTAssertEqual(index.statistics.evictions, 1)
        XCTAssertTrue(index.evictLeastRecentlyUsed()) // Soft preference is not a pin.
        XCTAssertNil(index.peek(tokens: [1, 2], namespace: "model"))
        XCTAssertEqual(index.peek(tokens: [1, 2, 3, 4], namespace: "model")?.value, "tail")
    }

    func testRetainedPairAdmissionIncludesEntryAndFullKeyBudgets() throws {
        for index in [try QwenPrefixCacheIndex<Int>(maxEntries: 1, maxBytes: 100),
                      try QwenPrefixCacheIndex<Int>(maxEntries: 4, maxBytes: 100, maxKeyTokens: 5)] {
            XCTAssertTrue(index.insert(tokens: [1, 2], namespace: "model", value: 1, logicalPayloadBytes: 10))
            XCTAssertFalse(index.canInsertAlongsidePrefix(tokens: [1, 2, 3, 4], namespace: "model",
                logicalPayloadBytes: 10, prefixTokenCount: 2))
            XCTAssertFalse(index.insert(tokens: [1, 2, 3, 4], namespace: "model", value: 2,
                logicalPayloadBytes: 10, retainingPrefixTokens: 2))
            XCTAssertEqual(index.statistics.entries, 1)
        }
    }

    func testInternalRadixBranchIsNotRetainedAsACompleteCheckpoint() throws {
        let index = try QwenPrefixCacheIndex<Int>(maxEntries: 2, maxBytes: 20)
        XCTAssertTrue(index.insert(tokens: [1, 2, 3], namespace: "model", value: 1, logicalPayloadBytes: 10))
        XCTAssertTrue(index.insert(tokens: [1, 2, 4], namespace: "model", value: 2, logicalPayloadBytes: 10))
        XCTAssertTrue(index.canInsertAlongsidePrefix(tokens: [1, 2, 5], namespace: "model",
            logicalPayloadBytes: 10, prefixTokenCount: 2))
        XCTAssertTrue(index.insert(tokens: [1, 2, 5], namespace: "model", value: 3,
            logicalPayloadBytes: 10, retainingPrefixTokens: 2))
        XCTAssertNil(index.peek(tokens: [1, 2], namespace: "model"))
        XCTAssertEqual(index.statistics.entries, 2)
        XCTAssertEqual(index.statistics.evictions, 1)
    }

    func testNewCountersPreserveHistoricalJSONAndRoundTripActualWork() throws {
        let json = """
        {"promptTokenCount":1000,"chunkCount":1,"targetSeconds":2,
         "draftHistorySeconds":0,"totalSeconds":3,"ssdWaitSeconds":0,
         "ssdLogicalBytes":0,"evaluateEveryLayers":4}
        """
        var stats = try JSONDecoder().decode(QwenPrefillStatistics.self, from: Data(json.utf8))
        XCTAssertNil(stats.actualForwardTokenCount)
        XCTAssertNil(stats.recomputedTokenCount)
        stats.cachedTokenCount = 832; stats.computedTokenCount = 168
        stats.actualForwardTokenCount = 168; stats.recomputedTokenCount = 0
        let result = try JSONDecoder().decode(QwenPrefillStatistics.self, from: JSONEncoder().encode(stats))
        XCTAssertEqual(result.actualForwardTokenCount, 168)
        XCTAssertEqual(result.recomputedTokenCount, 0)
        let cached = try XCTUnwrap(result.cachedTokenCount), computed = try XCTUnwrap(result.computedTokenCount)
        XCTAssertEqual(result.promptTokenCount, cached + computed)
        stats.actualForwardTokenCount = 200; stats.recomputedTokenCount = 32
        XCTAssertEqual(stats.targetTokensPerSecond, 100)
        XCTAssertEqual(stats.readyTokensPerSecond, 200.0 / 3.0)
        let index = try QwenPrefixCacheIndex<Int>(maxEntries: 1, maxBytes: 1)
        var cache = index.statistics
        cache.retainedSystemAnchorSkips = 2; cache.restoreWaits = 3
        let decoded = try JSONDecoder().decode(QwenPrefixCacheStatistics.self, from: JSONEncoder().encode(cache))
        XCTAssertEqual(decoded.retainedSystemAnchorSkips, 2)
        XCTAssertEqual(decoded.restoreWaits, 3)
    }
}
