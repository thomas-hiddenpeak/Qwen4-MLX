import Foundation
import XCTest
@testable import ANERunnerGPU

/// Counter and serialization oracles only: no model, MLX array or GPU work.
final class QwenMTPCostSummaryTests: XCTestCase {
    private func ordinaryStatistics() -> QwenMTPDecoder.Statistics {
        var value = QwenMTPDecoder.Statistics()
        value.rounds = 2; value.draftedTokens = 4; value.acceptedDraftTokens = 2
        value.verifiedTokens = 6; value.emittedTokens = 4
        value.acceptanceHistogram = [1, 0, 1, 0, 0]
        value.draftSeconds = 0.125; value.verifySeconds = 0.5
        value.rollbackSeconds = 0.0625; value.historySeconds = 0.0625
        // This belongs to prefill and must never enter the decode cost sum.
        value.prefillHistorySeconds = 99; value.prefillHistoryTokens = 1024
        return value
    }

    private func summary(_ value: QwenMTPDecoder.Statistics, steps: Int = 2,
                         tokens: Int = 4, seconds: Double = 1) -> QwenMTPCostSummary {
        QwenMTPCostSummary(statistics: value, decodeSteps: steps,
                          committedDecodeTokens: tokens, decodeSeconds: seconds)
    }

    func testCompletedRequestCostsExcludePromptHistoryAndUseActualOutput() throws {
        let value = summary(ordinaryStatistics())
        XCTAssertTrue(value.countersConsistent)
        XCTAssertTrue(value.acceptanceHistogramConsistent)
        XCTAssertEqual(value.targetOnlyDecodeSteps, 0)
        XCTAssertEqual(value.unacceptedDraftTokens, 2)
        XCTAssertEqual(value.draftAcceptanceRate, 0.5)
        XCTAssertEqual(value.meanProposedDraftsPerSpeculativeRound, 2)
        XCTAssertEqual(value.meanAcceptedDraftsPerSpeculativeRound, 1)
        XCTAssertEqual(value.meanCommittedTokensPerDecodeStep, 2)
        XCTAssertEqual(value.zeroAcceptanceRoundFraction, 0.5)
        XCTAssertEqual(value.targetEvaluationsPerCommittedToken, 1.5)
        XCTAssertEqual(value.componentSeconds, 0.75)
        XCTAssertEqual(value.componentsFitDecodeWindow, true)
        XCTAssertEqual(value.decodeSecondsOutsideComponents, 0.25)
        XCTAssertEqual(value.effectiveDecodeTokensPerSecond, 4)
        XCTAssertEqual(value.decodeSecondsPerCommittedToken, 0.25)
        XCTAssertEqual(try JSONDecoder().decode(QwenMTPCostSummary.self,
                                              from: JSONEncoder().encode(value)), value)
    }

    func testFirstTokenEOSOrBudgetOneHasNoDecodeRateOrInventedRounds() throws {
        var statistics = QwenMTPDecoder.Statistics()
        statistics.prefillHistorySeconds = 0.5
        let value = summary(statistics, steps: 0, tokens: 0, seconds: 0)
        XCTAssertTrue(value.countersConsistent)
        XCTAssertTrue(value.acceptanceHistogramConsistent)
        XCTAssertEqual(value.targetOnlyDecodeSteps, 0)
        XCTAssertEqual(value.componentSeconds, 0)
        XCTAssertEqual(value.decodeSecondsOutsideComponents, 0)
        XCTAssertNil(value.draftAcceptanceRate)
        XCTAssertNil(value.meanAcceptedDraftsPerSpeculativeRound)
        XCTAssertNil(value.meanCommittedTokensPerDecodeStep)
        XCTAssertNil(value.zeroAcceptanceRoundFraction)
        XCTAssertNil(value.targetEvaluationsPerCommittedToken)
        XCTAssertNil(value.effectiveDecodeTokensPerSecond)
        XCTAssertNil(value.decodeSecondsPerCommittedToken)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(value)) as? [String: Any])
        XCTAssertTrue(json["draftAcceptanceRate"] is NSNull)
        XCTAssertTrue(json["effectiveDecodeTokensPerSecond"] is NSNull)
        XCTAssertTrue(json["decodeSecondsPerCommittedToken"] is NSNull)
        XCTAssertNil(summary(statistics, steps: 0, tokens: 0, seconds: 1).effectiveDecodeTokensPerSecond)
    }

    func testAcceptedDraftEOSDoesNotInventAnEmittedBonus() throws {
        let plan = try MTPCommitPlan(drafts: [10, 99], targetIds: [10, 99, 123], remaining: 3, eos: [99])
        XCTAssertEqual(plan.accepted, 2)
        XCTAssertEqual(plan.emitted, [10, 99])
        var statistics = QwenMTPDecoder.Statistics()
        statistics.rounds = 1; statistics.draftedTokens = 2
        statistics.acceptedDraftTokens = plan.accepted; statistics.verifiedTokens = 3
        statistics.emittedTokens = plan.emitted.count
        statistics.acceptanceHistogram = [0, 0, 1, 0, 0]
        statistics.verifySeconds = 0.5
        let value = summary(statistics, steps: 1, tokens: plan.emitted.count, seconds: 1)
        XCTAssertTrue(value.countersConsistent)
        XCTAssertEqual(value.draftAcceptanceRate, 1)
        XCTAssertEqual(value.meanAcceptedDraftsPerSpeculativeRound, 2)
        XCTAssertEqual(value.meanCommittedTokensPerDecodeStep, 2)
        XCTAssertEqual(value.effectiveDecodeTokensPerSecond, 2)
        XCTAssertEqual(value.decodeSecondsPerCommittedToken, 0.5)
        XCTAssertNotEqual(value.meanCommittedTokensPerDecodeStep, 1 + Double(plan.accepted))
    }

    func testBudgetShortensProposalThenTargetOnlyTailIsCountedSeparately() throws {
        // With two tokens left, depth2 proposes just one. Rejection emits one
        // correction, then the final remaining token takes the no-draft branch.
        let first = try MTPCommitPlan(drafts: [10], targetIds: [20, 30], remaining: 2, eos: [])
        let tail = try MTPCommitPlan(drafts: [], targetIds: [40], remaining: 1, eos: [])
        var statistics = QwenMTPDecoder.Statistics()
        statistics.rounds = 1; statistics.draftedTokens = 1
        statistics.acceptedDraftTokens = first.accepted
        statistics.verifiedTokens = 3
        statistics.emittedTokens = first.emitted.count + tail.emitted.count
        statistics.acceptanceHistogram = [1, 0, 0, 0, 0]
        statistics.verifySeconds = 0.75
        let value = summary(statistics, steps: 2, tokens: statistics.emittedTokens)
        XCTAssertTrue(value.countersConsistent)
        XCTAssertEqual(value.targetOnlyDecodeSteps, 1)
        XCTAssertEqual(value.meanProposedDraftsPerSpeculativeRound, 1)
        XCTAssertEqual(value.meanCommittedTokensPerDecodeStep, 1)
        XCTAssertEqual(value.zeroAcceptanceRoundFraction, 1)
        XCTAssertEqual(value.targetEvaluationsPerCommittedToken, 1.5)
        XCTAssertEqual(value.effectiveDecodeTokensPerSecond, 2)

        var targetOnly = QwenMTPDecoder.Statistics()
        targetOnly.verifiedTokens = 1; targetOnly.emittedTokens = 1
        targetOnly.verifySeconds = 0.125
        let single = summary(targetOnly, steps: 1, tokens: 1, seconds: 0.25)
        XCTAssertTrue(single.countersConsistent)
        XCTAssertEqual(single.targetOnlyDecodeSteps, 1)
        XCTAssertEqual(single.effectiveDecodeTokensPerSecond, 4)
        XCTAssertNil(single.draftAcceptanceRate)
        XCTAssertNil(single.zeroAcceptanceRoundFraction)
    }

    func testInvalidOrEmptyTimeDenominatorsRemainUnknownAndJSONSafe() throws {
        for duration in [0.0, -1, Double.infinity, -Double.infinity, Double.nan] {
            let value = summary(ordinaryStatistics(), seconds: duration)
            XCTAssertNil(value.effectiveDecodeTokensPerSecond)
            XCTAssertNil(value.decodeSecondsPerCommittedToken)
            XCTAssertNoThrow(try JSONEncoder().encode(value))
            if duration == 0 {
                XCTAssertEqual(value.decodeSeconds, 0)
                XCTAssertEqual(value.componentsFitDecodeWindow, false)
            } else {
                XCTAssertNil(value.decodeSeconds)
                XCTAssertNil(value.componentsFitDecodeWindow)
            }
            XCTAssertNil(value.decodeSecondsOutsideComponents)
        }
        let underflow = summary(ordinaryStatistics(), seconds: Double.leastNonzeroMagnitude)
        XCTAssertNil(underflow.effectiveDecodeTokensPerSecond)
        XCTAssertNil(underflow.decodeSecondsPerCommittedToken)
    }

    func testTimerMismatchIsExposedWithoutCallingResidualGPUOrCPUTime() throws {
        var statistics = ordinaryStatistics()
        statistics.verifySeconds = 2
        let overlap = summary(statistics)
        XCTAssertEqual(overlap.componentSeconds, 2.25)
        XCTAssertEqual(overlap.componentsFitDecodeWindow, false)
        XCTAssertNil(overlap.decodeSecondsOutsideComponents)
        // The measured outer window remains independently usable.
        XCTAssertEqual(overlap.effectiveDecodeTokensPerSecond, 4)

        for invalid in [Double.nan, Double.infinity, -1] {
            statistics = ordinaryStatistics(); statistics.historySeconds = invalid
            let value = summary(statistics)
            XCTAssertNil(value.historySeconds)
            XCTAssertNil(value.componentSeconds)
            XCTAssertNil(value.componentsFitDecodeWindow)
            XCTAssertNil(value.decodeSecondsOutsideComponents)
            XCTAssertNoThrow(try JSONEncoder().encode(value))
        }
        statistics = ordinaryStatistics()
        statistics.draftSeconds = Double.greatestFiniteMagnitude
        statistics.verifySeconds = Double.greatestFiniteMagnitude
        XCTAssertNil(summary(statistics).componentSeconds)
    }

    func testCounterOrHistogramMismatchNeverFabricatesDerivedRatios() {
        var statistics = ordinaryStatistics()
        statistics.emittedTokens = 5
        let mismatch = summary(statistics)
        XCTAssertFalse(mismatch.countersConsistent)
        XCTAssertNil(mismatch.targetOnlyDecodeSteps)
        XCTAssertNil(mismatch.effectiveDecodeTokensPerSecond)
        XCTAssertNil(mismatch.draftAcceptanceRate)

        for histogram in [[0, 0, 2, 0, 0], [], [1, -1, 1, 0, 0], [0, 0, Int.max, 0, 0]] {
            statistics = ordinaryStatistics(); statistics.acceptanceHistogram = histogram
            let value = summary(statistics)
            XCTAssertFalse(value.acceptanceHistogramConsistent)
            XCTAssertNil(value.zeroAcceptanceRoundFraction)
            // Aggregate accepted/proposed counts remain separately checkable.
            XCTAssertEqual(value.draftAcceptanceRate, 0.5)
        }
        statistics = ordinaryStatistics(); statistics.rounds = Int.max
        XCTAssertFalse(summary(statistics).countersConsistent)
        statistics = ordinaryStatistics(); statistics.replayedTokens = Int.max
        XCTAssertFalse(summary(statistics).countersConsistent)
        XCTAssertFalse(summary(ordinaryStatistics(), steps: -1).countersConsistent)
    }

    private func result(mtp: QwenMTPDecoder.Statistics?) -> QwenGenerationResult {
        QwenGenerationResult(tokens: [7, 8, 9, 10, 11], finishReason: .length,
            preparationSeconds: 10, timeToFirstTokenSeconds: 20, decodeSeconds: 1, totalSeconds: 21,
            statistics: QwenGenerationStatistics(promptTokenCount: 11_057, generatedTokenCount: 5,
                decodeRounds: 2, decodedTokenCount: 4, prefillChunkCount: 28, finalStateOffset: 11_061,
                ssdWaitSeconds: 0, ssdLogicalBytes: 0, callbackSeconds: 0,
                prefillAccumulation: "reference", mtpDepth: mtp == nil ? 0 : 2,
                mtpVerification: mtp == nil ? nil : "batchedScalarLinear", mtp: mtp), phases: nil)
    }

    func testLibraryResultsAddSummaryAndStillDecodeHistoricalReports() throws {
        let generated = result(mtp: ordinaryStatistics())
        XCTAssertEqual(generated.mtpCostSummary, summary(ordinaryStatistics()))
        let encoded = try JSONEncoder().encode(generated)
        var json = try XCTUnwrap(JSONSerialization.jsonObject(with: encoded) as? [String: Any])
        XCTAssertNotNil(json["mtpCostSummary"])
        let rawStats = try XCTUnwrap(json["statistics"] as? [String: Any])
        let rawMTP = try XCTUnwrap(rawStats["mtp"] as? [String: Any])
        XCTAssertEqual(rawMTP["rounds"] as? Int, 2)
        XCTAssertEqual(rawMTP["emittedTokens"] as? Int, 4)
        XCTAssertEqual(rawMTP["prefillHistorySeconds"] as? Double, 99)
        let roundTrip = try JSONDecoder().decode(QwenGenerationResult.self, from: encoded)
        XCTAssertEqual(roundTrip.mtpCostSummary, generated.mtpCostSummary)

        json.removeValue(forKey: "mtpCostSummary")
        let historical = try JSONDecoder().decode(QwenGenerationResult.self,
            from: JSONSerialization.data(withJSONObject: json))
        XCTAssertEqual(historical.tokens, generated.tokens)
        XCTAssertNil(historical.mtpCostSummary)
        XCTAssertNil(result(mtp: nil).mtpCostSummary)
    }
}
