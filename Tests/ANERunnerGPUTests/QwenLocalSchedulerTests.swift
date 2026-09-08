import XCTest
@testable import ANERunnerGPU

/// The production queue engine with CPU-only handoffs. No tensors, GPU work,
/// filesystem access or model construction are required by these tests.
final class QwenLocalSchedulerTests: XCTestCase {
    private typealias Core = QwenLocalSchedulerCore<Ready>
    private typealias Limits = QwenLocalScheduler.Limits
    private final class Ready {
        let token: Int32
        var claimed = false
        var discards = 0
        init(_ token: Int32) { self.token = token }
    }
    private final class Harness {
        var tick: UInt64 = 0
        var calls: [String] = []
        var payloads: [Ready] = []
        var prefillError: Swift.Error?
        var decodeError: Swift.Error?
        var healthError: Swift.Error?
        var cancelDuringPrefill = false
        func advance(_ seconds: UInt64) { tick += seconds * 1_000_000_000 }
        func make(_ limits: Limits = Limits()) throws -> Core {
            try Core(limits: limits, backend: .init(
                validate: { request in
                    if request.tokens.first == -1 { throw QwenGenerationError.invalidRequest("fixture rejection") }
                },
                prefill: { request, cancellation in
                    self.calls.append("P\(request.tokens[0])")
                    if let error = self.prefillError { throw error }
                    self.advance(3)
                    let payload = Ready(request.tokens[0]); self.payloads.append(payload)
                    if self.cancelDuringPrefill { cancellation.cancel() }
                    return .init(value: payload, statistics: .init(
                        promptTokenCount: request.tokens.count, chunkCount: 1,
                        targetSeconds: 2, draftHistorySeconds: 0, totalSeconds: 3,
                        ssdWaitSeconds: 0, ssdLogicalBytes: 0, evaluateEveryLayers: 4))
                },
                decode: { ready, _, onToken in
                    self.calls.append("D\(ready.token)")
                    // A pre-admission failure can leave the payload unclaimed.
                    if let error = self.decodeError { throw error }
                    XCTAssertFalse(ready.claimed)
                    ready.claimed = true
                    try onToken?(ready.token)
                    self.advance(2)
                    return QwenLocalSchedulerTests.result(ready.token)
                },
                discard: { $0.discards += 1 },
                checkHealth: { if let error = self.healthError { throw error } }),
                now: { self.tick })
        }
    }
    private func request(_ token: Int32, maxTokens: Int = 3) -> QwenGenerationRequest {
        .init(tokens: [token], maxTokens: maxTokens)
    }
    private static func result(_ token: Int32) -> QwenGenerationResult {
        .init(tokens: [token], finishReason: .length, preparationSeconds: 0,
              timeToFirstTokenSeconds: 3, decodeSeconds: 2, totalSeconds: 5,
              statistics: .init(promptTokenCount: 1, generatedTokenCount: 1,
                decodeRounds: 0, decodedTokenCount: 0, prefillChunkCount: 1,
                finalStateOffset: 1, ssdWaitSeconds: 0, ssdLogicalBytes: 0,
                callbackSeconds: 0, prefillAccumulation: "reference", mtpDepth: 0,
                mtpVerification: nil, mtp: nil), phases: nil)
    }

    func testAdmissionIsAtomicAndChecked() throws {
        let h = Harness()
        XCTAssertThrowsError(try h.make(.init(maxQueuedPrefills: 0)))
        XCTAssertThrowsError(try h.make(.init(maxReadyDecodes: 0)))
        XCTAssertThrowsError(try h.make(.init(maxResidentTokens: 0)))
        XCTAssertThrowsError(try h.make(.init(maxConsecutivePrefills: 0)))
        let core = try h.make(.init(maxQueuedPrefills: 2, maxResidentTokens: 8))
        for request in [request(-1), request(1, maxTokens: Int.max), request(1, maxTokens: 0)] {
            XCTAssertThrowsError(try core.submit(request))
            XCTAssertTrue(core.snapshot().isIdle)
        }
        let cancelled = QwenCancellation(); cancelled.cancel()
        XCTAssertThrowsError(try core.submit(request(1), cancellation: cancelled))
        let a = try core.submit(request(1))
        XCTAssertThrowsError(try core.submit(request(2, maxTokens: 4))) {
            XCTAssertEqual($0 as? QwenLocalScheduler.Error, .overBudget)
        }
        let b = try core.submit(request(2))
        XCTAssertThrowsError(try core.submit(request(3))) {
            XCTAssertEqual($0 as? QwenLocalScheduler.Error, .queueFull)
        }
        XCTAssertEqual(core.snapshot().queuedPrefillIDs, [a, b])
        XCTAssertEqual(core.snapshot().reservedTokens, 8)
        XCTAssertEqual(h.calls, [])
    }

    func testFIFOStageFairnessAndReadyCapacity() throws {
        for (consecutive, readyLimit, expected) in [
            (2, 2, ["P1", "P2", "D1", "D2"]),
            (1, 2, ["P1", "D1", "P2", "D2"]),
            (4, 1, ["P1", "D1", "P2", "D2"])
        ] {
            let h = Harness()
            let core = try h.make(.init(maxReadyDecodes: readyLimit, maxConsecutivePrefills: consecutive))
            try core.submit(request(1)); try core.submit(request(2))
            for _ in 0..<4 { XCTAssertNotNil(try core.runNext()) }
            XCTAssertEqual(h.calls, expected)
            XCTAssertEqual(h.payloads.map(\.discards), [1, 1])
            XCTAssertTrue(core.snapshot().isIdle)
            XCTAssertEqual(core.snapshot().reservedTokens, 0)
            XCTAssertNil(try core.runNext())
        }
    }

    func testQueueWaitIsSeparateFromStageTime() throws {
        let h = Harness(), core = try h.make()
        let id = try core.submit(request(1))
        h.advance(2)
        let prefill = try XCTUnwrap(core.runNext())
        XCTAssertEqual(prefill.jobID, id)
        XCTAssertEqual(prefill.kind, .prefillReady)
        XCTAssertEqual(prefill.timing.prefillQueueWaitSeconds!, 2, accuracy: 1e-12)
        XCTAssertEqual(prefill.timing.prefillStageSeconds!, 3, accuracy: 1e-12)
        XCTAssertEqual(core.snapshot().reservedTokens, 4)
        h.advance(4)
        let completed = try XCTUnwrap(core.runNext())
        XCTAssertEqual(completed.timing.readyQueueWaitSeconds!, 4, accuracy: 1e-12)
        XCTAssertEqual(completed.timing.decodeStageSeconds!, 2, accuracy: 1e-12)
        XCTAssertEqual(completed.timing.totalQueueWaitSeconds, 6, accuracy: 1e-12)
        XCTAssertEqual(completed.timing.elapsedSeconds, 11, accuracy: 1e-12)
        XCTAssertEqual(core.snapshot().reservedTokens, 0)
    }

    func testQueuedReadyAndInStageCancellationReleaseExactlyOnce() throws {
        let h = Harness(), core = try h.make()
        let queued = try core.submit(request(1))
        XCTAssertEqual(try core.cancel(queued)?.kind, .cancelled)
        XCTAssertEqual(h.calls, [])
        let ready = try core.submit(request(2))
        _ = try core.runNext()
        XCTAssertEqual(try core.cancel(ready)?.kind, .cancelled)
        XCTAssertEqual(h.payloads.map(\.discards), [1])
        XCTAssertNil(try core.cancel(ready))
        let cancellation = QwenCancellation()
        try core.submit(request(3), cancellation: cancellation)
        cancellation.cancel()
        XCTAssertEqual(try core.runNext()?.kind, .cancelled)
        XCTAssertEqual(h.calls, ["P2"])
        h.cancelDuringPrefill = true
        try core.submit(request(4))
        XCTAssertEqual(try core.runNext()?.kind, .cancelled)
        XCTAssertEqual(h.payloads.map(\.discards), [1, 1])
        XCTAssertTrue(core.snapshot().isIdle)
        XCTAssertEqual(core.snapshot().reservedTokens, 0)
    }

    func testCallbackReentryRejectedButSubmitAndSnapshotAreSafe() throws {
        let h = Harness(), core = try h.make()
        var activeID: UUID?
        var childID: UUID?
        activeID = try core.submit(request(1), onToken: { _ in
            XCTAssertEqual(core.snapshot().runningJob, activeID)
            XCTAssertEqual(core.snapshot().runningStage, .decode)
            XCTAssertThrowsError(try core.runNext()) { XCTAssertEqual($0 as? QwenLocalScheduler.Error, .busy) }
            XCTAssertThrowsError(try core.cancel(activeID!)) { XCTAssertEqual($0 as? QwenLocalScheduler.Error, .busy) }
            XCTAssertThrowsError(try core.discardAll()) { XCTAssertEqual($0 as? QwenLocalScheduler.Error, .busy) }
            childID = try core.submit(self.request(2))
        })
        _ = try core.runNext()
        XCTAssertEqual(try core.runNext()?.kind, .completed)
        XCTAssertEqual(core.snapshot().queuedPrefillIDs, [try XCTUnwrap(childID)])
        XCTAssertEqual(core.snapshot().reservedTokens, 4)
        let flushed = try core.discardAll()
        XCTAssertEqual(flushed.map(\.jobID), [childID!])
        XCTAssertTrue(core.snapshot().isIdle)
    }

    func testCallbackBusyIsTerminalAndHealthyUnavailableDoesNotPoison() throws {
        let h = Harness(), core = try h.make()
        for error in [QwenGenerationError.busy, .unavailable("callback error"), .cancelled] {
            try core.submit(request(1), onToken: { _ in throw error })
            _ = try core.runNext()
            XCTAssertEqual(try core.runNext()?.kind, error == .cancelled ? .cancelled : .failed)
            XCTAssertTrue(h.payloads.last!.claimed)
            XCTAssertEqual(h.payloads.last!.discards, 1)
            XCTAssertTrue(core.snapshot().acceptingJobs)
            XCTAssertTrue(core.snapshot().isIdle)
        }
        try core.submit(request(2)); _ = try core.runNext()
        XCTAssertEqual(try core.runNext()?.kind, .completed)
        XCTAssertEqual(h.calls, ["P1", "D1", "P1", "D1", "P1", "D1", "P2", "D2"])
    }

    func testFailuresBeforeClaimAndTransientHealthBusyStillCleanUp() throws {
        let h = Harness(), core = try h.make()
        h.prefillError = QwenGenerationError.busy
        h.healthError = QwenGenerationError.busy
        try core.submit(request(1))
        XCTAssertEqual(try core.runNext()?.kind, .failed)
        XCTAssertEqual(core.snapshot().reservedTokens, 0)
        h.prefillError = nil
        h.decodeError = QwenGenerationError.cancelled
        try core.submit(request(2)); _ = try core.runNext()
        XCTAssertEqual(try core.runNext()?.kind, .cancelled)
        XCTAssertFalse(h.payloads[0].claimed)
        XCTAssertEqual(h.payloads[0].discards, 1)
        XCTAssertTrue(core.snapshot().isIdle)
        XCTAssertTrue(core.snapshot().acceptingJobs)
        h.decodeError = nil
        try core.submit(request(3)); _ = try core.runNext()
        XCTAssertEqual(try core.runNext()?.kind, .completed)
    }

    func testPrefillResourceLimitIsClassifiedAndReleasesAdmissionWithoutPoisoning() throws {
        let h = Harness()
        let core = try h.make(.init(maxQueuedPrefills: 1, maxResidentTokens: 4))
        h.prefillError = QwenGenerationError.resourceLimit("fixture state bytes")
        let refused = try core.submit(request(1))
        let failure = try XCTUnwrap(core.runNext())
        XCTAssertEqual(failure.jobID, refused)
        XCTAssertEqual(failure.kind, .failed)
        XCTAssertEqual(failure.stage, .prefill)
        XCTAssertEqual(failure.errorCode, "resource_limit")
        XCTAssertTrue(failure.errorDescription?.contains("fixture state bytes") == true)
        XCTAssertEqual(core.snapshot().reservedTokens, 0)
        XCTAssertEqual(core.snapshot().residentSequences, 0)
        XCTAssertEqual(core.snapshot().queuedPrefills, 0)
        XCTAssertTrue(core.snapshot().isIdle)
        XCTAssertTrue(core.snapshot().acceptingJobs)
        XCTAssertNil(core.snapshot().unavailableReason)
        XCTAssertTrue(h.payloads.isEmpty)

        h.prefillError = nil
        let next = try core.submit(request(2))
        XCTAssertEqual(try core.runNext()?.kind, .prefillReady)
        let completed = try XCTUnwrap(core.runNext())
        XCTAssertEqual(completed.jobID, next)
        XCTAssertEqual(completed.kind, .completed)
        XCTAssertNil(completed.errorCode)
        XCTAssertEqual(completed.result?.tokens, [2])
        XCTAssertEqual(h.payloads.map(\.discards), [1])
        XCTAssertEqual(core.snapshot().reservedTokens, 0)
        XCTAssertEqual(core.snapshot().residentSequences, 0)
        XCTAssertTrue(core.snapshot().isIdle)
    }

    func testPoisonAfterCancellationFlushesReadyAndQueuedWithTerminalEvents() throws {
        let h = Harness()
        let core = try h.make(.init(maxReadyDecodes: 2, maxConsecutivePrefills: 2))
        let a = try core.submit(request(1)), b = try core.submit(request(2)), c = try core.submit(request(3))
        _ = try core.runNext(); _ = try core.runNext()
        XCTAssertEqual(core.snapshot().readyDecodeIDs, [a, b])
        h.decodeError = QwenGenerationError.cancelled
        h.healthError = QwenGenerationError.unavailable("fake failed recovery")
        let failed = try XCTUnwrap(core.runNext())
        XCTAssertEqual(failed.jobID, a); XCTAssertEqual(failed.kind, .cancelled)
        XCTAssertEqual(core.snapshot().reservedTokens, 0)
        XCTAssertEqual(core.snapshot().readyDecodes, 0)
        XCTAssertEqual(core.snapshot().queuedPrefills, 0)
        XCTAssertEqual(core.snapshot().pendingEvents, 2)
        XCTAssertFalse(core.snapshot().acceptingJobs)
        XCTAssertFalse(core.snapshot().isIdle)
        XCTAssertEqual(h.payloads.map(\.discards), [1, 1])
        XCTAssertThrowsError(try core.submit(request(4))) {
            guard let error = $0 as? QwenLocalScheduler.Error,
                  case .closed = error else { return XCTFail("Expected closed, got \($0)") }
        }
        let pending = [try XCTUnwrap(core.runNext()), try XCTUnwrap(core.runNext())]
        XCTAssertEqual(Set(pending.map(\.jobID)), Set([b, c]))
        XCTAssertTrue(pending.allSatisfy { $0.kind == .failed })
        XCTAssertTrue(core.snapshot().isIdle)
        XCTAssertEqual(try core.discardAll().count, 0)
        XCTAssertEqual(h.calls, ["P1", "P2", "D1"])
    }
}
