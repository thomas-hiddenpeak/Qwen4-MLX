/// Pure commit decisions for a COMPLETE, already evaluated target verification.
/// Target row i predicts the token after verification input row i; the input
/// rows are [pending] + drafts. This must not receive an unfinished scalar
/// verification: even an early mismatch requires targetIds.count == m + 1.
struct MTPCommitPlan: Equatable, Sendable {
    enum Failure: Error, Equatable {
        case invalidDraftCount
        case invalidTargetCount
        case invalidBudget
        case invalidToken
    }

    /// Consecutive drafts matched by the target, before any EOS truncation.
    let accepted: Int
    /// Target input rows to retain: the pending token plus accepted drafts.
    let trunkConsumed: Int
    /// History pairs to append after the pending pair has been committed.
    let headAppend: Int
    /// Zero-based target hidden row for the next pending token's draft step.
    let nextHiddenRow: Int
    /// Target prediction at the first mismatch, or the full-accept bonus row.
    let correction: Int32
    /// Validated output prefix, stopping at its first designated EOS token.
    let emitted: [Int32]
    let terminal: Bool

    init(drafts: [Int32], targetIds: [Int32], remaining: Int, eos: Set<Int32>) throws {
        guard drafts.count <= 4 else { throw Failure.invalidDraftCount }
        guard targetIds.count == drafts.count + 1 else { throw Failure.invalidTargetCount }
        // Draft width was bounded before verification by remaining - 1.
        // Refuse an oversized verification instead of publishing or silently
        // truncating proposals that were not planned within the output budget.
        guard remaining > 0, drafts.count < remaining else { throw Failure.invalidBudget }
        guard drafts.allSatisfy({ $0 >= 0 }), targetIds.allSatisfy({ $0 >= 0 }),
              eos.allSatisfy({ $0 >= 0 }) else { throw Failure.invalidToken }

        var matched = 0
        while matched < drafts.count && targetIds[matched] == drafts[matched] { matched += 1 }
        accepted = matched
        trunkConsumed = matched + 1
        headAppend = matched
        nextHiddenRow = matched
        correction = targetIds[matched]

        let validated = Array(drafts.prefix(matched)) + [correction]
        if let stop = validated.firstIndex(where: { eos.contains($0) }) {
            emitted = Array(validated[...stop])
            terminal = true
        } else {
            emitted = validated
            terminal = validated.count == remaining
        }
        // EOS may occur inside the accepted prefix. Retain the verification
        // boundary above: the terminal decoder discards its state, which may
        // already have consumed EOS. Emission does not redefine cache offsets.
    }
}
