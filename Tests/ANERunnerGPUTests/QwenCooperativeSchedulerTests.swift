import Foundation
import XCTest
@testable import ANERunnerGPU

/// Exercise the production queue engine with CPU state and a deterministic
/// clock. These tests neither construct tensors nor load model configuration.
final class QwenCooperativeSchedulerTests: XCTestCase {
    private typealias Core = QwenLocalSchedulerCore<Session>
    private typealias Event = QwenLocalScheduler.Event
    private typealias Limits = QwenLocalScheduler.Limits

    private enum FixtureError: Swift.Error { case unexpectedWholeStage, decodeFailure, stalled }
    private final class Session {
        let request: QwenGenerationRequest
        var prompt = 0, generated = 0, prefillCalls = 0, decodeCalls = 0, discards = 0
        init(_ request: QwenGenerationRequest) { self.request = request }
        var token: Int32 { request.tokens[0] }
        var statistics: QwenPrefillStatistics {
            .init(promptTokenCount: request.tokens.count, chunkCount: prefillCalls,
                  targetSeconds: Double(prefillCalls * 3), draftHistorySeconds: 0,
                  totalSeconds: Double(prefillCalls * 3), ssdWaitSeconds: 0,
                  ssdLogicalBytes: 0, evaluateEveryLayers: 4)
        }
        var result: QwenGenerationResult {
            .init(tokens: Array(repeating: token, count: generated), finishReason: .length,
                  preparationSeconds: 0, timeToFirstTokenSeconds: statistics.totalSeconds,
                  decodeSeconds: Double(decodeCalls * 2),
                  totalSeconds: statistics.totalSeconds + Double(decodeCalls * 2),
                  statistics: .init(promptTokenCount: request.tokens.count,
                    generatedTokenCount: generated, decodeRounds: decodeCalls,
                    decodedTokenCount: max(0, generated - 1), prefillChunkCount: prefillCalls,
                    finalStateOffset: prompt + max(0, generated - 1), ssdWaitSeconds: 0,
                    ssdLogicalBytes: 0, callbackSeconds: 0, prefillAccumulation: "reference",
                    mtpDepth: 0, mtpVerification: nil, mtp: nil), phases: nil)
        }
    }
    private final class Harness {
        var tick: UInt64 = 0
        var calls: [String] = []
        var sessions: [Session] = []
        var decodeError: Swift.Error?
        var observePrefill: ((Session) -> Void)?
        func advance(_ seconds: UInt64) { tick += seconds * 1_000_000_000 }
        func make(_ limits: Limits) throws -> Core {
            try Core(limits: limits, backend: .init(
                validate: { request in
                    guard !request.tokens.isEmpty, request.maxTokens > 0 else {
                        throw QwenGenerationError.invalidRequest("invalid CPU fixture")
                    }
                },
                prefill: { _, _ in throw FixtureError.unexpectedWholeStage },
                decode: { _, _, _ in throw FixtureError.unexpectedWholeStage },
                discard: { $0.discards += 1 },
                checkHealth: {},
                prefillSlice: { request, cancellation, previous in
                    try cancellation.check()
                    let session: Session
                    if let previous {
                        session = previous
                        XCTAssertEqual(session.request.tokens, request.tokens)
                    } else {
                        session = Session(request)
                        self.sessions.append(session)
                    }
                    self.calls.append("P\(session.token)")
                    self.observePrefill?(session)
                    self.advance(3)
                    // Preserve the separate final prompt token in the fixture,
                    // so a resumed cursor cannot silently skip its last slice.
                    session.prompt = session.prompt < request.tokens.count - 1
                        ? min(request.tokens.count - 1, session.prompt + request.prefillChunk)
                        : request.tokens.count
                    session.prefillCalls += 1
                    let complete = session.prompt == request.tokens.count
                    return .init(value: session, statistics: complete ? session.statistics : nil,
                                 complete: complete)
                },
                decodeSlice: { session, cancellation, onToken in
                    try cancellation.check()
                    XCTAssertEqual(session.prompt, session.request.tokens.count)
                    self.calls.append("D\(session.token)")
                    self.advance(2)
                    if let error = self.decodeError { throw error }
                    session.decodeCalls += 1
                    session.generated += 1
                    try onToken?(session.token)
                    try cancellation.check()
                    return session.generated == session.request.maxTokens ? session.result : nil
                },
                progress: { ($0.prompt, $0.generated) }), now: { self.tick })
        }
    }

    private func limits(ready: Int = 2, residents: Int = 2, consecutive: Int = 1) -> Limits {
        .init(maxQueuedPrefills: 8, maxReadyDecodes: ready, maxResidentTokens: 256,
              maxConsecutivePrefills: consecutive, executionMode: .cooperative,
              decodeBurst: 4, maxResidentSequences: residents)
    }
    private func request(_ token: Int32, prompt: Int = 1, generated: Int = 2) -> QwenGenerationRequest {
        .init(tokens: Array(repeating: token, count: prompt), maxTokens: generated, prefillChunk: 2)
    }
    private func drain(_ core: Core, observe: ((Event) -> Void)? = nil) throws -> [Event] {
        var events: [Event] = []
        for _ in 0..<128 {
            guard let event = try core.runNext() else {
                XCTAssertTrue(core.snapshot().isIdle)
                return events
            }
            events.append(event); observe?(event)
        }
        XCTFail("Cooperative scheduler failed to finish a bounded fixture")
        throw FixtureError.stalled
    }
    private func assertReleased(_ core: Core, _ harness: Harness,
                                file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertTrue(core.snapshot().isIdle, file: file, line: line)
        XCTAssertEqual(core.snapshot().reservedTokens, 0, file: file, line: line)
        XCTAssertEqual(core.snapshot().residentSequences, 0, file: file, line: line)
        XCTAssertTrue(harness.sessions.allSatisfy { $0.discards == 1 }, file: file, line: line)
    }

    func testShortDecodeProgressBeforeLongPrefillReadyAndFourSliceBurst() throws {
        let h = Harness(), core = try h.make(limits())
        let long = try core.submit(request(1, prompt: 11, generated: 2))
        let short = try core.submit(request(2, generated: 8))
        let events = try drain(core) { _ in
            XCTAssertLessThanOrEqual(core.snapshot().residentSequences, 2)
        }
        let shortProgress = try XCTUnwrap(events.firstIndex { $0.jobID == short && $0.kind == .decodeProgress })
        let longReady = try XCTUnwrap(events.firstIndex { $0.jobID == long && $0.kind == .prefillReady })
        XCTAssertLessThan(shortProgress, longReady)
        XCTAssertEqual(events[shortProgress].generatedTokenCount, 1)
        let firstLongProgress = try XCTUnwrap(events.first { $0.jobID == long && $0.kind == .prefillProgress })
        XCTAssertEqual(firstLongProgress.processedPromptTokens, 2)
        XCTAssertEqual(Array(h.calls.prefix(7)), ["P1", "P2", "D2", "D2", "D2", "D2", "P1"])
        XCTAssertEqual(events.filter { $0.kind == .completed }.count, 2)
        XCTAssertEqual(h.sessions.first { $0.token == 1 }?.prefillCalls, 6)
        assertReleased(core, h)
    }

    func testRequeueWaitAccumulatesWithoutInflatingActiveStageTime() throws {
        let h = Harness(), core = try h.make(limits())
        try core.submit(request(1, prompt: 3, generated: 2))
        h.advance(1)
        let p1 = try XCTUnwrap(core.runNext())
        XCTAssertEqual(p1.kind, .prefillProgress)
        XCTAssertEqual(p1.timing.prefillStageSeconds!, 3, accuracy: 1e-12)
        h.advance(5)
        let p2 = try XCTUnwrap(core.runNext())
        XCTAssertEqual(p2.kind, .prefillReady)
        XCTAssertEqual(p2.timing.prefillStageSeconds!, 6, accuracy: 1e-12)
        h.advance(7)
        XCTAssertEqual(try core.runNext()?.kind, .decodeProgress)
        h.advance(11)
        let done = try XCTUnwrap(core.runNext())
        XCTAssertEqual(done.kind, .completed)
        XCTAssertEqual(done.timing.prefillQueueWaitSeconds!, 6, accuracy: 1e-12)
        XCTAssertEqual(done.timing.initialPrefillQueueWaitSeconds!, 1, accuracy: 1e-12)
        XCTAssertEqual(done.timing.prefillResumeWaitSeconds!, 5, accuracy: 1e-12)
        XCTAssertEqual(done.timing.readyQueueWaitSeconds!, 18, accuracy: 1e-12)
        XCTAssertEqual(done.timing.prefillStageSeconds!, 6, accuracy: 1e-12)
        XCTAssertEqual(done.timing.decodeStageSeconds!, 4, accuracy: 1e-12)
        XCTAssertEqual(done.timing.totalQueueWaitSeconds, 24, accuracy: 1e-12)
        XCTAssertEqual(done.timing.elapsedSeconds, 34, accuracy: 1e-12)
        assertReleased(core, h)
    }

    func testResidentLimitCountsPartialAndRunningButNotUnstartedJobs() throws {
        let h = Harness(), core = try h.make(limits(residents: 2))
        let a = try core.submit(request(1, prompt: 5, generated: 1))
        let b = try core.submit(request(2, prompt: 5, generated: 1))
        let c = try core.submit(request(3, prompt: 5, generated: 1))
        XCTAssertEqual(core.snapshot().residentSequences, 0)
        h.observePrefill = { _ in
            // The active slice has reserved its resident slot before entering
            // the backend, including the very first slice of a request.
            XCTAssertEqual(core.snapshot().residentSequences, h.sessions.filter { $0.discards == 0 }.count)
            XCTAssertLessThanOrEqual(core.snapshot().residentSequences, 2)
        }
        defer { h.observePrefill = nil }
        XCTAssertEqual(try core.runNext()?.jobID, a)
        XCTAssertEqual(core.snapshot().residentSequences, 1)
        XCTAssertEqual(try core.runNext()?.jobID, b)
        XCTAssertEqual(core.snapshot().residentSequences, 2)
        var firstCompletion: Event?
        for _ in 0..<32 {
            let event = try XCTUnwrap(core.runNext())
            XCTAssertNotEqual(event.jobID, c, "Unstarted third request exceeded resident capacity")
            XCTAssertLessThanOrEqual(core.snapshot().residentSequences, 2)
            if event.kind == .completed { firstCompletion = event; break }
        }
        XCTAssertNotNil(firstCompletion)
        XCTAssertEqual(h.sessions.map(\.token), [1, 2])
        let rest = try drain(core)
        XCTAssertTrue(rest.contains { $0.jobID == c && $0.kind == .completed })
        assertReleased(core, h)
    }

    func testReadyCapacityRemainsBoundedAcrossDecodeYields() throws {
        let h = Harness(), core = try h.make(limits(ready: 1, residents: 2, consecutive: 8))
        let a = try core.submit(request(1, generated: 3))
        let b = try core.submit(request(2, prompt: 3, generated: 1))
        let events = try drain(core) { _ in
            XCTAssertLessThanOrEqual(core.snapshot().readyDecodes, 1)
            XCTAssertLessThanOrEqual(core.snapshot().residentSequences, 2)
        }
        let aCompleted = try XCTUnwrap(events.firstIndex { $0.jobID == a && $0.kind == .completed })
        let bReady = try XCTUnwrap(events.firstIndex { $0.jobID == b && $0.kind == .prefillReady })
        XCTAssertLessThan(aCompleted, bReady)
        XCTAssertEqual(events.filter { $0.jobID == a && $0.kind == .decodeProgress }.count, 2)
        assertReleased(core, h)
    }

    func testCancelYieldedPrefillAndDecodeDiscardsAndReleasesExactlyOnce() throws {
        let h = Harness(), core = try h.make(limits())
        let partial = try core.submit(request(1, prompt: 5, generated: 3))
        XCTAssertEqual(try core.runNext()?.kind, .prefillProgress)
        let cancelledPrefill = try XCTUnwrap(core.cancel(partial))
        XCTAssertEqual(cancelledPrefill.kind, .cancelled)
        XCTAssertEqual(cancelledPrefill.stage, .prefill)
        XCTAssertNil(try core.cancel(partial))
        assertReleased(core, h)
        let cancellation = QwenCancellation()
        var delivered: [Int32] = []
        let decoding = try core.submit(request(2, generated: 3), cancellation: cancellation) { delivered.append($0) }
        XCTAssertEqual(try core.runNext()?.kind, .prefillReady)
        XCTAssertEqual(try core.runNext()?.kind, .decodeProgress)
        cancellation.cancel()
        let cancelledDecode = try XCTUnwrap(core.runNext())
        XCTAssertEqual(cancelledDecode.jobID, decoding)
        XCTAssertEqual(cancelledDecode.kind, .cancelled)
        XCTAssertEqual(cancelledDecode.stage, .decode)
        XCTAssertEqual(delivered, [2])
        XCTAssertEqual(h.calls, ["P1", "P2", "D2"])
        assertReleased(core, h)
        try core.submit(request(3, generated: 1))
        XCTAssertEqual(try drain(core).last?.kind, .completed)
        assertReleased(core, h)
    }

    func testFailureAfterProgressAndCallbackBusyNeverReplayConsumedSlice() throws {
        let h = Harness(), core = try h.make(limits())
        var firstDelivered: [Int32] = []
        let first = try core.submit(request(1, generated: 3)) { firstDelivered.append($0) }
        _ = try core.runNext()
        XCTAssertEqual(try core.runNext()?.kind, .decodeProgress)
        h.decodeError = FixtureError.decodeFailure
        let failure = try XCTUnwrap(core.runNext())
        XCTAssertEqual(failure.jobID, first)
        XCTAssertEqual(failure.kind, .failed)
        XCTAssertEqual(firstDelivered, [1])
        XCTAssertNil(try core.runNext())
        assertReleased(core, h)
        h.decodeError = nil
        var busyDelivered: [Int32] = []
        let callbackFailure = try core.submit(request(2, generated: 3)) { token in
            busyDelivered.append(token)
            throw QwenGenerationError.busy
        }
        _ = try core.runNext()
        let busy = try XCTUnwrap(core.runNext())
        XCTAssertEqual(busy.jobID, callbackFailure)
        XCTAssertEqual(busy.kind, .failed)
        XCTAssertEqual(busyDelivered, [2])
        XCTAssertNil(try core.runNext())
        XCTAssertTrue(core.snapshot().acceptingJobs)
        assertReleased(core, h)
        try core.submit(request(3, generated: 1))
        XCTAssertEqual(try drain(core).last?.kind, .completed)
        XCTAssertEqual(h.calls, ["P1", "D1", "D1", "P2", "D2", "P3", "D3"])
        assertReleased(core, h)
    }
}
