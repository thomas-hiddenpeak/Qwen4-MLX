import XCTest
@testable import ANERunnerGPU

/// Pure CPU checks: no arrays, model weights or GPU synchronization.
final class GPUProfilerContractTests: XCTestCase {
    func testPausedSynchronizedProfilerDoesNotCollectOrSynchronize() throws {
        let p = try GPUProfiler(mode: .synchronizedStages, attentionBreakdown: true)
        try p.setRecordingEnabled(false)
        XCTAssertFalse(p.isRecording)
        let value = try p.measure("paused", tokenCount: 1, outputs: { (_: Int) -> [Tensor] in
            XCTFail("Paused profiler must not inspect outputs"); return []
        }) { 42 }
        XCTAssertEqual(value, 42)
        XCTAssertTrue(p.report.stages.isEmpty)
    }

    func testExplicitPhaseAndPositionSurviveSingleTokenPrefill() throws {
        let p = try GPUProfiler(mode: .hostBodyOnly)
        try p.setForwardContext(phase: .prefill, position: 11056)
        _ = try p.measure("last_prompt", tokenCount: 1, outputs: { (_: Int) in [] }) { 42 }
        XCTAssertEqual(p.report.stages.first?.phase, .prefill)
        XCTAssertEqual(p.report.stages.first?.position, 11056)
        p.clearForwardContext()
        _ = try p.measure("outside_forward", tokenCount: 1, outputs: { (_: Int) in [] }) { 42 }
        XCTAssertNil(p.report.stages.last?.phase)
        XCTAssertNil(p.report.stages.last?.position)
        try p.reset()
        XCTAssertTrue(p.report.stages.isEmpty)
    }

    func testRecordingCannotChangeInsideAStageAndRecoversAfterError() throws {
        let p = try GPUProfiler(mode: .hostBodyOnly)
        XCTAssertThrowsError(try p.measure("invalid_mutation", tokenCount: 1, outputs: { (_: Int) in [] }) {
            try p.setRecordingEnabled(false)
            return 42
        })
        XCTAssertEqual(p.report.stages.first?.succeeded, false)
        try p.setRecordingEnabled(false)
        try p.reset()
        XCTAssertTrue(p.report.stages.isEmpty)
    }
}
