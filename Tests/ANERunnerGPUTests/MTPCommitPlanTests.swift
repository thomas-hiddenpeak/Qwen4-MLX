import XCTest
@testable import ANERunnerGPU

/// Pure token/index oracles: no model, tokenizer, MLX array or GPU is created.
final class MTPCommitPlanTests: XCTestCase {
    private struct Case {
        let depth: Int
        let accepted: Int
        let drafts: [Int32]
        let targets: [Int32]
        let correction: Int32
        var output: [Int32] { Array(drafts.prefix(accepted)) + [correction] }
    }

    private var cases: [Case] {
        (1...4).flatMap { depth in
            (0...depth).map { accepted in
                let drafts = Array([Int32(10), 20, 30, 40].prefix(depth))
                var targets = drafts + [Int32(900)]
                // Later rows deliberately match their drafts. The first
                // mismatch must still end acceptance; no suffix may escape.
                if accepted < depth { targets[accepted] = 999 }
                return Case(depth: depth, accepted: accepted, drafts: drafts,
                            targets: targets, correction: accepted < depth ? 999 : 900)
            }
        }
    }

    func testEveryAcceptanceLengthAndBudgetBoundary() throws {
        XCTAssertEqual(cases.count, 14)
        for item in cases {
            for remaining in [item.depth + 1, item.depth + 2] {
                let plan = try MTPCommitPlan(drafts: item.drafts, targetIds: item.targets,
                                             remaining: remaining, eos: [])
                let label = "depth=\(item.depth), accepted=\(item.accepted), remaining=\(remaining)"
                XCTAssertEqual(plan.accepted, item.accepted, label)
                XCTAssertEqual(plan.trunkConsumed, item.accepted + 1, label)
                XCTAssertEqual(plan.headAppend, item.accepted, label)
                XCTAssertEqual(plan.nextHiddenRow, item.accepted, label)
                XCTAssertEqual(plan.correction, item.correction, label)
                XCTAssertEqual(plan.emitted, item.output, label)
                XCTAssertEqual(plan.terminal, item.accepted == item.depth && remaining == item.depth + 1, label)
                XCTAssertLessThanOrEqual(plan.emitted.count, remaining, label)
            }
        }
    }

    func testEOSAtEveryAcceptedDraftPositionDoesNotRedefineStateBoundary() throws {
        for item in cases where item.accepted > 0 {
            for position in 0..<item.accepted {
                let stop = item.drafts[position]
                let plan = try MTPCommitPlan(drafts: item.drafts, targetIds: item.targets,
                                             remaining: item.depth + 2, eos: [stop])
                let label = "depth=\(item.depth), accepted=\(item.accepted), EOS position=\(position)"
                XCTAssertEqual(plan.emitted, Array(item.drafts.prefix(position + 1)), label)
                XCTAssertTrue(plan.terminal, label)
                XCTAssertEqual(plan.accepted, item.accepted, label)
                XCTAssertEqual(plan.trunkConsumed, item.accepted + 1, label)
                XCTAssertEqual(plan.headAppend, item.accepted, label)
                XCTAssertEqual(plan.nextHiddenRow, item.accepted, label)
                XCTAssertEqual(plan.correction, item.correction, label)
            }
        }
    }

    func testCorrectionAndFullAcceptanceBonusEOS() throws {
        for item in cases {
            let plan = try MTPCommitPlan(drafts: item.drafts, targetIds: item.targets,
                                         remaining: item.depth + 2, eos: [item.correction])
            XCTAssertEqual(plan.emitted, item.output)
            XCTAssertTrue(plan.terminal)
            XCTAssertEqual(plan.accepted, item.accepted)
        }
    }

    func testUnacceptedDraftEOSCannotBePublishedOrStopGeneration() throws {
        for item in cases where item.accepted < item.depth {
            let rejectedEOS = Set(item.drafts.dropFirst(item.accepted))
            let plan = try MTPCommitPlan(drafts: item.drafts, targetIds: item.targets,
                                         remaining: item.depth + 2, eos: rejectedEOS)
            XCTAssertEqual(plan.emitted, item.output)
            XCTAssertFalse(plan.terminal)
            XCTAssertTrue(rejectedEOS.isDisjoint(with: plan.emitted))
        }
    }

    func testEarliestEOSWinsWhenSeveralValidatedOutputsAreStops() throws {
        let plan = try MTPCommitPlan(drafts: [10, 20, 30, 40], targetIds: [10, 20, 30, 40, 900],
                                     remaining: 6, eos: [20, 40, 900])
        XCTAssertEqual(plan.emitted, [10, 20])
        XCTAssertTrue(plan.terminal)
        XCTAssertEqual(plan.accepted, 4)
        XCTAssertEqual(plan.trunkConsumed, 5)
        XCTAssertEqual(plan.headAppend, 4)
        XCTAssertEqual(plan.nextHiddenRow, 4)
    }

    func testSingleRemainingTokenUsesOnlyItsTargetPrediction() throws {
        let eosSets: [Set<Int32>] = [[], [42]]
        for eos in eosSets {
            let plan = try MTPCommitPlan(drafts: [], targetIds: [42], remaining: 1, eos: eos)
            XCTAssertEqual(plan.emitted, [42])
            XCTAssertTrue(plan.terminal)
            XCTAssertEqual(plan.accepted, 0)
            XCTAssertEqual(plan.trunkConsumed, 1)
            XCTAssertEqual(plan.headAppend, 0)
            XCTAssertEqual(plan.nextHiddenRow, 0)
            XCTAssertEqual(plan.correction, 42)
        }
        let targetOnly = try MTPCommitPlan(drafts: [], targetIds: [42], remaining: 2, eos: [])
        XCTAssertFalse(targetOnly.terminal)
        XCTAssertEqual(targetOnly.emitted, [42])
    }

    func testRejectsIncompleteVerificationEvenAfterAnEarlyMismatch() {
        let incomplete: [[Int32]] = [[], [999], [999, 20], [999, 20, 30, 900, 901]]
        for targets in incomplete {
            XCTAssertThrowsError(try MTPCommitPlan(drafts: [10, 20, 30], targetIds: targets,
                                                   remaining: 4, eos: [])) {
                XCTAssertEqual($0 as? MTPCommitPlan.Failure, .invalidTargetCount)
            }
        }
    }

    func testRejectsInvalidBudgetsDraftWidthsAndTokenIDs() {
        for remaining in [Int.min, -1, 0, 1, 2] {
            XCTAssertThrowsError(try MTPCommitPlan(drafts: [10, 20], targetIds: [10, 20, 900],
                                                   remaining: remaining, eos: [])) {
                XCTAssertEqual($0 as? MTPCommitPlan.Failure, .invalidBudget)
            }
        }
        XCTAssertThrowsError(try MTPCommitPlan(drafts: [1, 2, 3, 4, 5], targetIds: [1, 2, 3, 4, 5, 6],
                                               remaining: 6, eos: [])) {
            XCTAssertEqual($0 as? MTPCommitPlan.Failure, .invalidDraftCount)
        }
        let invalid: [(drafts: [Int32], targets: [Int32], eos: Set<Int32>)] = [
            ([-1], [42, 43], []), ([42], [-1, 43], []), ([42], [42, -1], []), ([42], [42, 43], [-1])
        ]
        for item in invalid {
            XCTAssertThrowsError(try MTPCommitPlan(drafts: item.drafts, targetIds: item.targets,
                                                   remaining: 2, eos: item.eos)) {
                XCTAssertEqual($0 as? MTPCommitPlan.Failure, .invalidToken)
            }
        }
    }
}
