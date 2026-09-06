import Foundation

/// CPU-only accounting over a completed request. No tensor access, evaluation,
/// synchronization or additional timing is performed here. Raw decoder counters
/// remain authoritative; undefined or inconsistent derived values encode as null.
public struct QwenMTPCostSummary: Codable, Equatable, Sendable {
    public let schemaVersion: Int
    /// Actual published output after the prefill-selected first token, including EOS.
    public let committedDecodeTokens: Int
    /// Includes the terminal one-token target-only step, unlike speculativeRounds.
    public let decodeSteps: Int
    public let speculativeRounds: Int
    public let countersConsistent: Bool
    public let acceptanceHistogramConsistent: Bool
    public let targetOnlyDecodeSteps: Int?
    public let unacceptedDraftTokens: Int?
    public let draftAcceptanceRate: Double?
    public let meanProposedDraftsPerSpeculativeRound: Double?
    public let meanAcceptedDraftsPerSpeculativeRound: Double?
    /// Uses actual output, not accepted drafts plus an assumed bonus per round.
    public let meanCommittedTokensPerDecodeStep: Double?
    public let zeroAcceptanceRoundFraction: Double?
    /// Target input rows evaluated by verification and replay per published token.
    /// This is a work ratio, not a speedup or a count of GPU launches.
    public let targetEvaluationsPerCommittedToken: Double?
    public let decodeSeconds: Double?
    public let draftSeconds: Double?
    public let verificationSeconds: Double?
    public let commitOrReplaySeconds: Double?
    public let historySeconds: Double?
    /// Sum of the four existing sequential decode component timers. Prompt
    /// history, load, prefill, queue waits and callbacks are not added to it.
    public let componentSeconds: Double?
    public let componentsFitDecodeWindow: Bool?
    /// An accounting residual, not a measurement of CPU time or GPU idle time.
    /// Null when the component sum exceeds the enclosing decode window.
    public let decodeSecondsOutsideComponents: Double?
    public let effectiveDecodeTokensPerSecond: Double?
    public let decodeSecondsPerCommittedToken: Double?

    public init(statistics: QwenMTPDecoder.Statistics, decodeSteps: Int,
                committedDecodeTokens: Int, decodeSeconds: Double) {
        schemaVersion = 1
        self.committedDecodeTokens = committedDecodeTokens
        self.decodeSteps = decodeSteps
        speculativeRounds = statistics.rounds

        let counts = [decodeSteps, committedDecodeTokens, statistics.rounds,
                      statistics.draftedTokens, statistics.acceptedDraftTokens,
                      statistics.verifiedTokens, statistics.replayedTokens, statistics.emittedTokens]
        let maxDrafts = statistics.rounds.multipliedReportingOverflow(by: 4)
        let targetEvaluations = Self.checkedSum([statistics.verifiedTokens, statistics.replayedTokens])
        // Every productive step emits at least one token. The no-draft branch
        // can occur only once, at the final single-token budget boundary.
        let countsValid = counts.allSatisfy { $0 >= 0 }
            && !maxDrafts.overflow
            && statistics.rounds <= decodeSteps
            && decodeSteps - statistics.rounds <= 1
            && ((decodeSteps == 0) == (committedDecodeTokens == 0))
            && committedDecodeTokens >= decodeSteps
            && statistics.emittedTokens == committedDecodeTokens
            && statistics.draftedTokens >= statistics.rounds
            && statistics.draftedTokens <= maxDrafts.partialValue
            && statistics.acceptedDraftTokens <= statistics.draftedTokens
            && statistics.verifiedTokens >= decodeSteps
            && targetEvaluations != nil
        countersConsistent = countsValid

        let histogram = statistics.acceptanceHistogram
        let histogramRounds = Self.checkedSum(histogram)
        let weighted = histogram.enumerated().map { index, count -> Int? in
            let result = index.multipliedReportingOverflow(by: count)
            return count >= 0 && !result.overflow ? result.partialValue : nil
        }
        let histogramAccepted = weighted.allSatisfy { $0 != nil }
            ? Self.checkedSum(weighted.compactMap { $0 }) : nil
        let histogramValid = histogram.count == 5 && statistics.rounds >= 0
            && statistics.acceptedDraftTokens >= 0
            && histogramRounds == statistics.rounds
            && histogramAccepted == statistics.acceptedDraftTokens
        acceptanceHistogramConsistent = histogramValid

        targetOnlyDecodeSteps = countsValid ? decodeSteps - statistics.rounds : nil
        unacceptedDraftTokens = countsValid ? statistics.draftedTokens - statistics.acceptedDraftTokens : nil
        draftAcceptanceRate = countsValid ? Self.ratio(statistics.acceptedDraftTokens, statistics.draftedTokens) : nil
        meanProposedDraftsPerSpeculativeRound = countsValid ? Self.ratio(statistics.draftedTokens, statistics.rounds) : nil
        meanAcceptedDraftsPerSpeculativeRound = countsValid ? Self.ratio(statistics.acceptedDraftTokens, statistics.rounds) : nil
        meanCommittedTokensPerDecodeStep = countsValid ? Self.ratio(committedDecodeTokens, decodeSteps) : nil
        zeroAcceptanceRoundFraction = countsValid && histogramValid
            ? Self.ratio(histogram[0], statistics.rounds) : nil
        targetEvaluationsPerCommittedToken = countsValid
            ? Self.ratio(targetEvaluations!, committedDecodeTokens) : nil

        self.decodeSeconds = Self.nonnegativeFinite(decodeSeconds)
        draftSeconds = Self.nonnegativeFinite(statistics.draftSeconds)
        verificationSeconds = Self.nonnegativeFinite(statistics.verifySeconds)
        commitOrReplaySeconds = Self.nonnegativeFinite(statistics.rollbackSeconds)
        historySeconds = Self.nonnegativeFinite(statistics.historySeconds)
        let components = [draftSeconds, verificationSeconds, commitOrReplaySeconds, historySeconds]
        componentSeconds = components.allSatisfy { $0 != nil }
            ? Self.nonnegativeFinite(components.compactMap { $0 }.reduce(0, +)) : nil
        if let measured = self.decodeSeconds, let sum = componentSeconds {
            componentsFitDecodeWindow = sum <= measured
            decodeSecondsOutsideComponents = sum <= measured ? measured - sum : nil
        } else {
            componentsFitDecodeWindow = nil
            decodeSecondsOutsideComponents = nil
        }
        // A zero-duration or zero-output window cannot establish throughput or
        // TPOT. Do not publish infinity or an apparent zero-latency result.
        if countsValid, committedDecodeTokens > 0, let measured = self.decodeSeconds, measured > 0 {
            effectiveDecodeTokensPerSecond = Self.nonnegativeFinite(Double(committedDecodeTokens) / measured)
            let perToken = measured / Double(committedDecodeTokens)
            decodeSecondsPerCommittedToken = perToken > 0 ? Self.nonnegativeFinite(perToken) : nil
        } else {
            effectiveDecodeTokensPerSecond = nil
            decodeSecondsPerCommittedToken = nil
        }
    }

    private static func nonnegativeFinite(_ value: Double) -> Double? {
        value.isFinite && value >= 0 ? value : nil
    }

    private static func ratio(_ numerator: Int, _ denominator: Int) -> Double? {
        guard numerator >= 0, denominator > 0 else { return nil }
        return nonnegativeFinite(Double(numerator) / Double(denominator))
    }

    private static func checkedSum(_ values: [Int]) -> Int? {
        var sum = 0
        for value in values {
            guard value >= 0 else { return nil }
            let addition = sum.addingReportingOverflow(value)
            guard !addition.overflow else { return nil }
            sum = addition.partialValue
        }
        return sum
    }

    private enum CodingKeys: String, CodingKey {
        case schemaVersion, committedDecodeTokens, decodeSteps, speculativeRounds
        case countersConsistent, acceptanceHistogramConsistent, targetOnlyDecodeSteps, unacceptedDraftTokens
        case draftAcceptanceRate, meanProposedDraftsPerSpeculativeRound, meanAcceptedDraftsPerSpeculativeRound
        case meanCommittedTokensPerDecodeStep, zeroAcceptanceRoundFraction, targetEvaluationsPerCommittedToken
        case decodeSeconds, draftSeconds, verificationSeconds, commitOrReplaySeconds, historySeconds
        case componentSeconds, componentsFitDecodeWindow, decodeSecondsOutsideComponents
        case effectiveDecodeTokensPerSecond, decodeSecondsPerCommittedToken
    }

    /// Explicit nulls distinguish unknown ratios from real zeros in JSON tools.
    public func encode(to encoder: any Encoder) throws {
        var values = encoder.container(keyedBy: CodingKeys.self)
        try values.encode(schemaVersion, forKey: .schemaVersion)
        try values.encode(committedDecodeTokens, forKey: .committedDecodeTokens)
        try values.encode(decodeSteps, forKey: .decodeSteps)
        try values.encode(speculativeRounds, forKey: .speculativeRounds)
        try values.encode(countersConsistent, forKey: .countersConsistent)
        try values.encode(acceptanceHistogramConsistent, forKey: .acceptanceHistogramConsistent)
        try values.encode(targetOnlyDecodeSteps, forKey: .targetOnlyDecodeSteps)
        try values.encode(unacceptedDraftTokens, forKey: .unacceptedDraftTokens)
        try values.encode(draftAcceptanceRate, forKey: .draftAcceptanceRate)
        try values.encode(meanProposedDraftsPerSpeculativeRound, forKey: .meanProposedDraftsPerSpeculativeRound)
        try values.encode(meanAcceptedDraftsPerSpeculativeRound, forKey: .meanAcceptedDraftsPerSpeculativeRound)
        try values.encode(meanCommittedTokensPerDecodeStep, forKey: .meanCommittedTokensPerDecodeStep)
        try values.encode(zeroAcceptanceRoundFraction, forKey: .zeroAcceptanceRoundFraction)
        try values.encode(targetEvaluationsPerCommittedToken, forKey: .targetEvaluationsPerCommittedToken)
        try values.encode(decodeSeconds, forKey: .decodeSeconds)
        try values.encode(draftSeconds, forKey: .draftSeconds)
        try values.encode(verificationSeconds, forKey: .verificationSeconds)
        try values.encode(commitOrReplaySeconds, forKey: .commitOrReplaySeconds)
        try values.encode(historySeconds, forKey: .historySeconds)
        try values.encode(componentSeconds, forKey: .componentSeconds)
        try values.encode(componentsFitDecodeWindow, forKey: .componentsFitDecodeWindow)
        try values.encode(decodeSecondsOutsideComponents, forKey: .decodeSecondsOutsideComponents)
        try values.encode(effectiveDecodeTokensPerSecond, forKey: .effectiveDecodeTokensPerSecond)
        try values.encode(decodeSecondsPerCommittedToken, forKey: .decodeSecondsPerCommittedToken)
    }
}
