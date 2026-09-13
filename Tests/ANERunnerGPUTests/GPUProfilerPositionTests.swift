import Foundation
import XCTest
@testable import ANERunnerGPU

/// Pure CPU filtering/report checks; no model or Tensor is constructed.
final class GPUProfilerPositionTests: XCTestCase {
    private enum Failure: Error { case expected }

    func testEarlierSynchronizedPositionsRunOnlyBodyAndKeepContextSettable() throws {
        let p = try GPUProfiler(mode: .synchronizedStages, allocatorSnapshots: true,
            attentionBreakdown: true, phaseFilter: .prefill, minimumPosition: 32032)
        var bodies = 0
        let positions: [Int?] = [nil, 0, 31616]
        for position in positions {
            if let position { try p.setForwardContext(phase: .prefill, position: position) }
            else { p.clearForwardContext() }
            XCTAssertTrue(p.isRecording)
            let value = try p.measure("", tokenCount: 0, outputs: { (_: Int) -> [Tensor] in
                XCTFail("Filtered stage must not collect outputs"); return []
            }) { bodies += 1; return 42 }
            XCTAssertEqual(value, 42)
        }
        XCTAssertEqual(bodies, 3)
        XCTAssertTrue(p.stages.isEmpty)
        XCTAssertEqual(p.droppedRecords, 0)
    }

    func testInclusiveWindowRecordsActualBodyAndFinalS1Offsets() throws {
        let p = try GPUProfiler(mode: .hostBodyOnly, phaseFilter: .prefill, minimumPosition: 32032)
        for (position, count) in [(31616, 416), (32032, 416), (32448, 317), (32765, 1)] {
            try p.setForwardContext(phase: .prefill, position: position)
            _ = try p.measure("attention", tokenCount: count, outputs: { (_: Int) in [] }) { 7 }
        }
        try p.setForwardContext(phase: .decode, position: 32766)
        _ = try p.measure("decode", tokenCount: 1, outputs: { (_: Int) in [] }) { 7 }
        XCTAssertEqual(p.stages.map(\.position), [32032, 32448, 32765])
        XCTAssertEqual(p.stages.map(\.tokenCount), [416, 317, 1])
        XCTAssertTrue(p.stages.allSatisfy { $0.phase == .prefill })
        XCTAssertEqual(p.report.minimumPosition, 32032)
        try p.reset()
        XCTAssertTrue(p.isRecording)
        XCTAssertEqual(p.report.minimumPosition, 32032)
    }

    func testFilteredErrorsPropagateAndPausedRecordingStillWins() throws {
        let p = try GPUProfiler(mode: .synchronizedStages, minimumPosition: 32032)
        try p.setForwardContext(phase: .prefill, position: 0)
        XCTAssertThrowsError(try p.measure("ignored", tokenCount: 1,
            outputs: { (_: Int) in [] }) { throw Failure.expected }) {
                XCTAssertTrue($0 is Failure)
            }
        try p.setForwardContext(phase: .prefill, position: 32032)
        try p.setRecordingEnabled(false)
        let value = try p.measure("paused", tokenCount: 1, outputs: { (_: Int) -> [Tensor] in
            XCTFail("Paused stage must not collect outputs"); return []
        }) { 3 }
        XCTAssertEqual(value, 3)
        XCTAssertTrue(p.stages.isEmpty)
    }

    func testDefaultWindowAndHistoricalReportRemainCompatible() throws {
        let p = try GPUProfiler(mode: .hostBodyOnly)
        _ = try p.measure("without_context", tokenCount: 1, outputs: { (_: Int) in [] }) { 1 }
        XCTAssertEqual(p.stages.count, 1)
        XCTAssertNil(p.report.minimumPosition)
        let scoped = try GPUProfiler(mode: .hostBodyOnly, minimumPosition: 7)
        let encoded = try JSONEncoder().encode(scoped.report)
        XCTAssertEqual(try JSONDecoder().decode(GPUProfiler.Report.self, from: encoded).minimumPosition, 7)
        var legacy = try XCTUnwrap(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        legacy.removeValue(forKey: "minimumPosition")
        let old = try JSONDecoder().decode(GPUProfiler.Report.self,
            from: JSONSerialization.data(withJSONObject: legacy))
        XCTAssertNil(old.minimumPosition)
    }

    func testNegativeWindowRejected() throws {
        XCTAssertThrowsError(try GPUProfiler(minimumPosition: -1))
    }
}
