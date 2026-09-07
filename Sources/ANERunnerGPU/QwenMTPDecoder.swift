// Native Qwen4 draft/history contract follows garnermccloud/mlx-serve
// generate.zig / transformer.zig at 7dbcba04 (MIT; see UPSTREAM-LICENSE).
import Foundation

/// Request-local, greedy native MTP. No head or speculative state is shared
/// between requests. Batched verification may differ from repeated S1 forwards
/// in floating point rounding; scalar mode is the correctness oracle.
public final class QwenMTPDecoder {
    public enum Verification: String, Sendable {
        case batched, scalar, batchedRounded, batchedCaptured, batchedScalarMoE, batchedScalarLinear
        /// Same S1 arithmetic as batchedScalarLinear, with routed MoE tokens
        /// dispatched together. This remains an explicit experimental policy.
        case batchedTokenMoE
        public var usesScalarLinear: Bool { self == .batchedScalarLinear || self == .batchedTokenMoE }
        var captures: Bool { self == .batchedCaptured || self == .batchedScalarMoE || usesScalarLinear }
        var roundsState: Bool { self == .batchedRounded || captures }
    }
    public struct Statistics: Codable, Sendable {
        public var rounds = 0, draftedTokens = 0, acceptedDraftTokens = 0
        public var verifiedTokens = 0, replayedTokens = 0, emittedTokens = 0
        public var draftSeconds = 0.0, verifySeconds = 0.0
        public var rollbackSeconds = 0.0, historySeconds = 0.0
        public var prefillHistorySeconds = 0.0
        public var prefillHistoryTokens = 0
        /// Absolute RoPE position of the first retained MTP cache row.
        public var historyStartPosition = 1
        public var acceptanceHistogram = [0, 0, 0, 0, 0]
    }
    public struct Round {
        /// Newly committed output tokens, including the target bonus/correction.
        public let tokens: [Int32]
        public let ssdWaitSeconds: Double
        public let ssdLogicalBytes: Int
    }
    private let model: QwenModel
    private let head: QwenMTP
    private var headState: QwenMTP.State
    private var previousStream: Tensor?
    private var promptRows = 0
    private let draftHistoryTokens: Int?
    private var promptHistoryStart = 0
    private var valid = true
    public private(set) var statistics = Statistics()
    public let verification: Verification
    public let verificationEvaluateEveryLayers: Int

    public init(model: QwenModel, head: QwenMTP, verification: Verification = .scalar,
                draftHistoryTokens: Int? = nil, verificationEvaluateEveryLayers: Int = 4) {
        self.model = model; self.head = head; self.verification = verification
        self.draftHistoryTokens = draftHistoryTokens
        self.verificationEvaluateEveryLayers = verificationEvaluateEveryLayers
        headState = head.makeState()
    }
    private func seconds(_ start: UInt64) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - start) * 1e-9
    }
    private func rows(_ stream: Tensor, _ start: Int, _ end: Int) throws -> Tensor {
        try MX.slice(stream, starts: [0,start,0], ends: [1,end,10240])
    }

    /// Called in prompt order. Each row pairs trunk h[p] with known token p+1.
    /// The caller supplies the complete prompt so pairs can cross chunk edges.
    /// An optional initial history cap affects only the draft head. Its retained
    /// pair rows start at a multiple of four, preserving QSA block boundaries;
    /// alignment may retain up to three more rows than the requested cap.
    /// Subsequent generated history grows normally; this is not a sliding window.
    public func consumePrompt(stream: Tensor, prompt: [Int32], offset: Int) throws {
        try QwenExecutionPhase.validateVerificationInterval(verificationEvaluateEveryLayers)
        guard valid, offset == promptRows, stream.shape.count == 3,
              stream.shape[0] == 1, stream.shape[2] == 10240,
              stream.shape[1] > 0, offset + stream.shape[1] <= prompt.count else {
            throw GPUError.invalid("Invalid MTP prompt history order or shape")
        }
        do {
            let start = DispatchTime.now().uptimeNanoseconds
            if promptRows == 0 {
                if let cap = draftHistoryTokens {
                    guard cap > 0, cap <= model.configuration.maximumPositions else {
                        throw GPUError.invalid("MTP draft history cap must be within 1...\(model.configuration.maximumPositions)")
                    }
                    promptHistoryStart = 4 * (max(0, prompt.count - 1 - cap) / 4)
                    headState = head.makeState(positionBase: promptHistoryStart + 1)
                }
                statistics.historyStartPosition = headState.positionBase
            }
            let end = offset + stream.shape[1]
            let pairStart = max(offset, promptHistoryStart)
            let pairEnd = min(end, prompt.count - 1)
            if pairEnd > pairStart {
                let out = try head.forward(hidden: rows(stream, pairStart - offset, pairEnd - offset),
                    tokens: Array(prompt[(pairStart + 1)...pairEnd]), state: &headState, wantLogits: false)
                try MX.eval([out.stream] + headState.tensors)
                statistics.prefillHistoryTokens += pairEnd - pairStart
            }
            // Always retain the real trunk's final row, even when this whole
            // chunk was skipped by the draft history cap. The first draft must
            // pair h[P-1] with the pending target token, not an MTP output row.
            previousStream = try rows(stream, stream.shape[1] - 1, stream.shape[1])
            promptRows = end
            statistics.prefillHistorySeconds += seconds(start)
        } catch { valid = false; throw error }
    }

    /// Seal prompt preparation before publishing a same-process handoff. The
    /// last hidden-row slice is lazy even when its source stream was evaluated.
    /// Finish that slice and every head state tensor while still in prefill.
    public func finishPrompt(expectedTokenCount: Int) throws {
        let start = DispatchTime.now().uptimeNanoseconds
        defer { statistics.prefillHistorySeconds += seconds(start) }
        do {
            guard valid, expectedTokenCount > 0, promptRows == expectedTokenCount,
                  let previousStream,
                  headState.positionBase + headState.offset == expectedTokenCount else {
                throw GPUError.invalid("MTP prompt history is not ready for handoff")
            }
            try MX.eval([previousStream] + headState.tensors)
        } catch { valid = false; throw error }
    }

    /// The pending token was already emitted, but is not yet in trunk state.
    /// Each successful nonterminal round leaves the next pending token in that
    /// form. A terminal accepted draft EOS may already be consumed by verification;
    /// the decoder then refuses reuse and the generation API discards its state.
    public func next(pending: Int32, state callerState: inout QwenModel.State, depth: Int,
                     remaining: Int, eos: Set<Int32>, decodeMode: GPUDecodeMode = .reference,
                     checkCancellation: () throws -> Void = {}) throws -> Round {
        try QwenExecutionPhase.validateVerificationInterval(verificationEvaluateEveryLayers)
        var state = callerState
        guard valid, let previousStream, (1...4).contains(depth), remaining > 0,
              state.valid, headState.positionBase + headState.offset == state.offset else {
            throw GPUError.invalid("Invalid MTP generation position or budget")
        }
        do {
            try checkCancellation()
            var ssdWait = 0.0, ssdBytes = 0
            func account(_ out: QwenModel.Output) {
                ssdWait += out.ssdWaitSeconds; ssdBytes += out.ssdLogicalBytes
            }
            // Do not speculate beyond the output budget. A final single-token
            // request uses the ordinary target path, with no unnecessary draft.
            let draftCount = min(depth, remaining - 1)
            if draftCount == 0 {
                let verifyStart = DispatchTime.now().uptimeNanoseconds
                let out = try model.forward(tokens: [pending], state: &state, decodeMode: decodeMode, phase: .decode,
                                            allowExperimentalDecodeAsync: false)
                guard let logits = out.logits else { throw GPUError.invalid("Missing target logits") }
                let selected = try model.greedyToken(logits)
                try model.evaluate([selected], state: &state)
                account(out)
                let token = try selected.ints()[0]
                // This branch exhausts the budget; the decoder cannot resume.
                valid = false
                statistics.emittedTokens += 1
                statistics.verifiedTokens += 1
                statistics.verifySeconds += seconds(verifyStart)
                try checkCancellation()
                callerState = state
                return Round(tokens: [token], ssdWaitSeconds: ssdWait, ssdLogicalBytes: ssdBytes)
            }

            let draftStart = DispatchTime.now().uptimeNanoseconds
            var temporary = headState, firstCommitted: QwenMTP.State?
            var hidden = previousStream, token = pending, drafts: [Int32] = []
            for _ in 0..<draftCount {
                try checkCancellation()
                let out = try head.forward(hidden: hidden, tokens: [token], state: &temporary)
                guard let logits = out.logits else { throw GPUError.invalid("Missing MTP logits") }
                let selected = try model.greedyToken(logits)
                try MX.eval([selected, out.stream] + temporary.tensors)
                token = try selected.ints()[0]
                drafts.append(token); hidden = out.stream
                if firstCommitted == nil { firstCommitted = temporary }
                if eos.contains(token) { break }
            }
            statistics.draftSeconds += seconds(draftStart)
            statistics.draftedTokens += drafts.count
            try checkCancellation()

            // Every returned stream belongs to a real target forward. Capture
            // modes commit an evaluated prefix; legacy modes replay on rejection.
            var accepted = 0, correction: Int32 = 0, committedStreams: [Tensor] = []
            var verificationPlan: MTPCommitPlan?
            func scalarVerify(_ limit: Int, replay: Bool) throws {
                var input = pending
                accepted = 0; committedStreams = []
                for i in 0...limit {
                    try checkCancellation()
                    let out = try model.forward(tokens: [input], state: &state, decodeMode: decodeMode, phase: .decode,
                                            allowExperimentalDecodeAsync: false)
                    guard let logits = out.logits else { throw GPUError.invalid("Missing target logits") }
                    let selected = try model.greedyToken(logits)
                    try model.evaluate([selected, out.stream], state: &state)
                    account(out); committedStreams.append(out.stream)
                    correction = try selected.ints()[0]
                    if replay { statistics.replayedTokens += 1 }
                    else { statistics.verifiedTokens += 1 }
                    if i == limit || correction != drafts[i] { break }
                    accepted += 1; input = drafts[i]
                }
            }
            let verifyStart = DispatchTime.now().uptimeNanoseconds
            if verification == .scalar {
                try scalarVerify(drafts.count, replay: false)
                statistics.verifySeconds += seconds(verifyStart)
            } else {
                let snapshot = try model.checkpoint(state: &state)
                let out = try model.forward(tokens: [pending] + drafts, state: &state,
                                            lastLogitOnly: false, evaluateEveryLayers: verificationEvaluateEveryLayers,
                                            decodeMode: decodeMode,
                                            verifyScalarBoundaries: verification.roundsState,
                                            captureVerification: verification.captures,
                                            verifyScalarMoE: verification == .batchedScalarMoE || verification.usesScalarLinear,
                                            verifyScalarLinear: verification.usesScalarLinear,
                                            verifyTokenMoE: verification == .batchedTokenMoE,
                                            phase: .verification, allowExperimentalDecodeAsync: false)
                guard let logits = out.logits else { throw GPUError.invalid("Missing verification logits") }
                let selected = try model.greedyToken(logits)
                try model.evaluate([selected, out.stream], state: &state)
                let targets = try selected.ints()
                let plan = try MTPCommitPlan(drafts: drafts, targetIds: targets, remaining: remaining, eos: eos)
                verificationPlan = plan
                account(out); statistics.verifiedTokens += targets.count
                accepted = plan.accepted
                statistics.verifySeconds += seconds(verifyStart)
                if verification.captures {
                    let commitStart = DispatchTime.now().uptimeNanoseconds
                    state = try model.commitVerificationPrefix(state, from: snapshot,
                                                               tokens: [pending] + drafts, count: plan.trunkConsumed)
                    statistics.rollbackSeconds += seconds(commitStart)
                    correction = plan.correction
                    committedStreams = try (0..<plan.trunkConsumed).map { try rows(out.stream, $0, $0 + 1) }
                } else if accepted < drafts.count {
                    let rollbackStart = DispatchTime.now().uptimeNanoseconds
                    try model.restore(snapshot, state: &state)
                    // Scalar replay can change the decision, so its outputs
                    // replace the batch plan completely.
                    verificationPlan = nil
                    try scalarVerify(accepted, replay: true)
                    statistics.rollbackSeconds += seconds(rollbackStart)
                } else {
                    correction = plan.correction
                    committedStreams = try (0..<plan.trunkConsumed).map { try rows(out.stream, $0, $0 + 1) }
                }
            }
            try checkCancellation()
            let historyStart = DispatchTime.now().uptimeNanoseconds
            // The first draft input was already a true (hidden, pending) pair.
            // Reuse that state, then append only the accepted target hiddens.
            headState = firstCommitted!
            let appendCount = verificationPlan?.headAppend ?? accepted
            if appendCount > 0 {
                let committed = try MX.concat(Array(committedStreams.prefix(appendCount)), axis: 1)
                let out = try head.forward(hidden: committed, tokens: Array(drafts.prefix(appendCount)),
                                           state: &headState, wantLogits: false)
                try MX.eval([out.stream] + headState.tensors)
            }
            self.previousStream = committedStreams[verificationPlan?.nextHiddenRow ?? accepted]
            statistics.historySeconds += seconds(historyStart)
            statistics.rounds += 1; statistics.acceptedDraftTokens += accepted
            statistics.acceptanceHistogram[accepted] += 1
            var emitted: [Int32]
            if let verificationPlan {
                emitted = verificationPlan.emitted
                if verificationPlan.terminal { valid = false }
            } else {
                emitted = Array(drafts.prefix(accepted)) + [correction]
                if let stop = emitted.firstIndex(where: { eos.contains($0) }) {
                    emitted = Array(emitted[...stop]); valid = false
                }
            }
            statistics.emittedTokens += emitted.count
            if emitted.count == remaining { valid = false }
            try checkCancellation()
            callerState = state
            return Round(tokens: emitted, ssdWaitSeconds: ssdWait, ssdLogicalBytes: ssdBytes)
        } catch { valid = false; throw error }
    }
}
