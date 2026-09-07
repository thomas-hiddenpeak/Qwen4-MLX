import XCTest
@testable import ANERunnerGPU

/// Pure CPU policy checks; no environment mutation, model or MLX arrays.
final class QwenDecodeAsyncScheduleTests: XCTestCase {
    func testLiteralEnvironmentContract() throws {
        XCTAssertEqual(try QwenDecodeAsyncSchedule(environmentValue: nil).everyLayers, 0)
        XCTAssertEqual(try QwenDecodeAsyncSchedule(environmentValue: "0").everyLayers, 0)
        XCTAssertEqual(try QwenDecodeAsyncSchedule(environmentValue: "8").everyLayers, 8)
        for value in ["", " ", " 8", "8 ", "08", "+8", "-8", "1", "4", "16", "true", "8.0"] {
            XCTAssertThrowsError(try QwenDecodeAsyncSchedule(environmentValue: value), value)
        }
    }

    func testSubmissionBoundariesAndFinalJoin() throws {
        let enabled = try QwenDecodeAsyncSchedule(environmentValue: "8")
        let disabled = try QwenDecodeAsyncSchedule(environmentValue: nil)
        let selected = (0...48).filter {
            enabled.shouldSubmit(phase: .decode, tokenCount: 1, completedLayers: $0, layerCount: 48, allowed: true)
        }
        XCTAssertEqual(selected, [8, 16, 24, 32, 40])
        for layer in 0...48 {
            XCTAssertFalse(disabled.shouldSubmit(phase: .decode, tokenCount: 1,
                completedLayers: layer, layerCount: 48, allowed: true))
            // MTP scalar verification and target-only use .decode/S1 too.
            XCTAssertFalse(enabled.shouldSubmit(phase: .decode, tokenCount: 1,
                completedLayers: layer, layerCount: 48, allowed: false))
        }
        XCTAssertFalse(enabled.shouldSubmit(phase: .decode, tokenCount: 1,
            completedLayers: 8, layerCount: 8, allowed: true))
    }

    func testPrefillAndVerificationCannotSelectSubmission() throws {
        let enabled = try QwenDecodeAsyncSchedule(environmentValue: "8")
        for phase in [nil, QwenExecutionPhase.prefill, .verification] as [QwenExecutionPhase?] {
            for tokens in [1, 2, 3, 5, 416] {
                XCTAssertFalse(enabled.shouldSubmit(phase: phase, tokenCount: tokens,
                    completedLayers: 8, layerCount: 48, allowed: true))
            }
        }
        XCTAssertFalse(enabled.shouldSubmit(phase: .decode, tokenCount: 2,
            completedLayers: 8, layerCount: 48, allowed: true))
    }
}
