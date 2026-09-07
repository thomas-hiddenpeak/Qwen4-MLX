import Foundation
import XCTest
@testable import ANERunnerGPU

/// CPU-only filtering, timestamps and report compatibility; no GPU tensors.
final class GPUProfilerTests: XCTestCase {
    private enum ExpectedFailure: Error { case body }

    func testUnmatchedSynchronizedPhasesCallOnlyBody() throws {
        let profiler = try GPUProfiler(mode: .synchronizedStages, allocatorSnapshots: true,
                                       phaseFilter: .verification)
        var calls = 0
        let phases: [QwenExecutionPhase?] = [nil, .prefill, .decode]
        for phase in phases {
            if let phase { try profiler.setForwardContext(phase: phase, position: 12) }
            else { profiler.clearForwardContext() }
            XCTAssertTrue(profiler.isRecording, "Forward context must remain settable by the model")
            // Invalid stage metadata is ignored by the same early-return path
            // as disabled recording; no synchronization or collection occurs.
            let value = try profiler.measure("", tokenCount: 0, outputs: { (_: Int) -> [Tensor] in
                XCTFail("Filtered profiler must not inspect outputs"); return []
            }) { calls += 1; return 42 }
            XCTAssertEqual(value, 42)
        }
        XCTAssertEqual(calls, 3)
        XCTAssertTrue(profiler.stages.isEmpty)
        XCTAssertEqual(profiler.droppedRecords, 0)
    }

    func testContextSelectsPhaseAndClearingStopsRecording() throws {
        let profiler = try GPUProfiler(mode: .hostBodyOnly, phaseFilter: .verification)
        XCTAssertTrue(profiler.isRecording)
        try profiler.setForwardContext(phase: .verification, position: 128)
        let value = try profiler.measure("verify", tokenCount: 3, outputs: { (_: Int) -> [Tensor] in
            XCTFail("Host-only profiling must not inspect outputs"); return []
        }) { 7 }
        XCTAssertEqual(value, 7)
        XCTAssertEqual(profiler.stages.count, 1)
        let stage = try XCTUnwrap(profiler.stages.first)
        XCTAssertEqual(stage.phase, .verification)
        XCTAssertEqual(stage.position, 128)
        try assertInterval(stage)
        profiler.clearForwardContext()
        _ = try profiler.measure("outside", tokenCount: 1, outputs: { (_: Int) in [] }) { 9 }
        XCTAssertEqual(profiler.stages.count, 1)
        XCTAssertTrue(profiler.isRecording)
        XCTAssertEqual(profiler.report.phaseFilter, .verification)
        try profiler.reset()
        XCTAssertTrue(profiler.stages.isEmpty)
        XCTAssertEqual(profiler.report.phaseFilter, .verification)
    }

    func testFilteredBodyErrorPropagatesWithoutRecording() throws {
        let profiler = try GPUProfiler(mode: .synchronizedStages, allocatorSnapshots: true,
                                       phaseFilter: .decode)
        try profiler.setForwardContext(phase: .prefill, position: 0)
        XCTAssertThrowsError(try profiler.measure("prefill", tokenCount: 2,
            outputs: { (_: Int) -> [Tensor] in XCTFail("Filtered error must not collect"); return [] }) {
                throw ExpectedFailure.body
            }) { error in
                XCTAssertTrue(error is ExpectedFailure)
            }
        XCTAssertTrue(profiler.stages.isEmpty)
        // The filtered path never marked a stage active.
        try profiler.setRecordingEnabled(false)
        try profiler.setForwardContext(phase: .decode, position: 2)
        _ = try profiler.measure("paused", tokenCount: 1,
            outputs: { (_: Int) -> [Tensor] in XCTFail("Paused profiler must not collect"); return [] }) { 1 }
        XCTAssertTrue(profiler.stages.isEmpty)
    }

    func testFailedHostStageRetainsDiagnosticInterval() throws {
        let profiler = try GPUProfiler(mode: .hostBodyOnly, phaseFilter: .decode)
        try profiler.setForwardContext(phase: .decode, position: 9)
        XCTAssertThrowsError(try profiler.measure("decode", tokenCount: 1,
            outputs: { (_: Int) in [] }) { throw ExpectedFailure.body })
        let stage = try XCTUnwrap(profiler.stages.first)
        XCTAssertFalse(stage.succeeded)
        XCTAssertNotNil(stage.error)
        XCTAssertNil(stage.evaluationWaitMilliseconds)
        try assertInterval(stage)
    }

    func testOptionalPhaseAndTimestampFieldsDecodeHistoricalReport() throws {
        let profiler = try GPUProfiler(mode: .hostBodyOnly, phaseFilter: .prefill)
        try profiler.setForwardContext(phase: .prefill, position: 0)
        _ = try profiler.measure("prompt", tokenCount: 1, outputs: { (_: Int) in [] }) { 1 }
        let encoder = JSONEncoder(), decoder = JSONDecoder()
        let currentData = try encoder.encode(profiler.report)
        let current = try decoder.decode(GPUProfiler.Report.self, from: currentData)
        XCTAssertEqual(current.phaseFilter, .prefill)
        try assertInterval(try XCTUnwrap(current.stages.first))

        var legacy = try XCTUnwrap(JSONSerialization.jsonObject(with: currentData) as? [String: Any])
        legacy.removeValue(forKey: "phaseFilter")
        var stages = try XCTUnwrap(legacy["stages"] as? [[String: Any]])
        stages[0].removeValue(forKey: "startedUptimeNanoseconds")
        stages[0].removeValue(forKey: "endedUptimeNanoseconds")
        legacy["stages"] = stages
        let decoded = try decoder.decode(GPUProfiler.Report.self, from: JSONSerialization.data(withJSONObject: legacy))
        XCTAssertNil(decoded.phaseFilter)
        let oldStage = try XCTUnwrap(decoded.stages.first)
        XCTAssertNil(oldStage.startedUptimeNanoseconds)
        XCTAssertNil(oldStage.endedUptimeNanoseconds)
        XCTAssertEqual(oldStage.name, "prompt")
        XCTAssertEqual(oldStage.elapsedMilliseconds, current.stages[0].elapsedMilliseconds)
    }

    private func assertInterval(_ stage: GPUProfiler.Stage,
                                file: StaticString = #filePath, line: UInt = #line) throws {
        let start = try XCTUnwrap(stage.startedUptimeNanoseconds, file: file, line: line)
        let end = try XCTUnwrap(stage.endedUptimeNanoseconds, file: file, line: line)
        XCTAssertGreaterThanOrEqual(end, start, file: file, line: line)
        XCTAssertEqual(stage.elapsedMilliseconds, Double(end - start) * 1e-6,
                       accuracy: 1e-12, file: file, line: line)
    }
}
