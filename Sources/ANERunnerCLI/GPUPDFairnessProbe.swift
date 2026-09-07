import ANERunnerGPU
import CMLX
import CryptoKit
import Foundation

extension RunnerCLI {
    /// Fixed AR consumer; arrival is marked in callback 8 and admitted after runNext returns.
    /// This probe only selects an existing burst limit; it never changes scheduler policy.
    static func probeGPUPDFairness(_ args: Arguments) throws {
        try args.validate(["--model-dir", "--tokens-file", "--output", "--scenario", "--frozen-reference"])
        let output = try args.require("--output")
        guard !FileManager.default.fileExists(atPath: output) else {
            throw CLIError.usage("PD fairness probe --output must be a new file")
        }
        var report: [String: Any] = ["schema": "qwen38-pd-fairness-probe-v1", "complete": false,
            "passed": false, "outcome": "incomplete", "clock": "mach_absolute_time_nanoseconds",
            "full_model_instances": 1, "burst_order": [4, 8, 8, 4], "arrival_callback_ordinal": 8,
            "short_max_tokens": 64, "long_max_tokens": 128, "prefill_chunk": 416, "mtp_depth": 0,
            "decode_mode": "reference", "prefill_attention": "reference", "prefill_eval_layers": 4,
            "notes": ["Actual EOS handling is unchanged. Short budget 64 may finish by length; no minimum length is forced.",
                "Independent short and long AR references precede all four groups; no measured group is excluded as warmup.",
                "Both job IDs, callbacks, step boundaries and snapshots are retained. Arrival precedes submission at the next safe boundary.",
                "Long TTFT gate uses arrival to first callback; submission TTFT is separately recorded.",
                "Remaining completion ends at the short terminal runNext return, not its last callback.",
                "Callback gaps include all adjacent callbacks after arrival. Prefill interruption gaps are a separately identified subset.",
                "Local callbacks are not network delivery. One serial executor, no batching, preemption or concurrent GPU execution.",
                "State checks cover AR final offset, full output IDs, cleanup and a fresh run after cancellation; no hidden tensor equality is claimed.",
                "Screening thresholds do not establish production SLOs; a failed or inconclusive screen leaves burst 4 unchanged."]]
        var references = [String: Any](), groups = [[String: Any]](), checks = [String: Bool]()
        func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
        func seconds(_ start: UInt64, _ end: UInt64) -> Double { Double(end - start) * 1e-9 }
        func object<T: Encodable>(_ value: T) throws -> Any {
            try JSONSerialization.jsonObject(with: JSONEncoder().encode(value))
        }
        func sha(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
        func write(_ complete: Bool = false) throws {
            report["complete"] = complete
            report["checks"] = checks
            report["references"] = references; report["groups"] = groups
            try emit(report, to: output)
        }
        func clean(_ scheduler: QwenLocalScheduler) -> Bool {
            let s = scheduler.snapshot()
            return s.isIdle && s.acceptingJobs && s.runningJob == nil && s.runningStage == nil &&
                s.queuedPrefills == 0 && s.readyDecodes == 0 && s.residentSequences == 0 &&
                s.reservedTokens == 0 && s.pendingEvents == 0 && s.unavailableReason == nil
        }
        func summary(_ values: [Double]) -> [String: Any] {
            let ordered = values.sorted()
            func quantile(_ q: Double) -> Any {
                guard !ordered.isEmpty else { return NSNull() }
                let x = Double(ordered.count - 1) * q, a = Int(x), b = min(Int(x) + 1, ordered.count - 1)
                return ordered[a] + (ordered[b] - ordered[a]) * (x - Double(a))
            }
            return ["values_seconds": values, "count": values.count, "p50_seconds": quantile(0.5),
                "p95_seconds": quantile(0.95), "max_seconds": ordered.last.map { $0 as Any } ?? NSNull()]
        }
        do {
            let directory = URL(fileURLWithPath: try args.require("--model-dir")).standardizedFileURL.resolvingSymlinksInPath()
            let frozenPath = try args.require("--frozen-reference")
            let frozenData = try Data(contentsOf: URL(fileURLWithPath: frozenPath))
            let frozen = try JSONDecoder().decode(PDFairnessFrozen.self, from: frozenData)
            let inputPath = try args.require("--tokens-file")
            let inputData = try Data(contentsOf: URL(fileURLWithPath: inputPath))
            let longTokens = try JSONDecoder().decode([Int32].self, from: inputData)
            guard frozen.schema == "qwen38-pd-fairness-frozen-reference-v1", longTokens == frozen.long_tokens,
                  longTokens.count == 11_057, frozen.short_tokens.count == 26,
                  frozen.short_output_tokens.count == 64, frozen.long_output_tokens.count == 128,
                  frozen.short_finish_reason == "length", frozen.long_finish_reason == "length" else {
                throw CLIError.usage("Unexpected frozen PD fairness inputs or historical outputs")
            }
            report["model_directory"] = directory.path
            report["frozen_reference"] = ["path": frozenPath, "sha256": sha(frozenData)]
            report["long_input"] = ["path": inputPath, "sha256": sha(inputData), "token_ids": longTokens]
            report["short_input"] = ["token_ids": frozen.short_tokens, "text": frozen.short_text, "thinking": false]
            let configuration = try QwenConfiguration(modelDirectory: directory)
            let short = QwenGenerationRequest(tokens: frozen.short_tokens, maxTokens: 64, contextLimit: 4096,
                prefillChunk: 416, mtpDepth: 0, decodeMode: .reference, prefillAttention: .reference)
            let long = QwenGenerationRequest(tokens: longTokens, maxTokens: 128, contextLimit: 16_384,
                prefillChunk: 416, mtpDepth: 0, decodeMode: .reference, prefillAttention: .reference)
            try short.validate(configuration: configuration); try long.validate(configuration: configuration)
            let reservation = short.tokens.count + short.maxTokens + long.tokens.count + long.maxTokens
            var previousCache = 0
            try MX.check(mlx_set_cache_limit(&previousCache, 256 * 1024 * 1024), "PD fairness allocation cache")
            defer { var ignored = 0; _ = mlx_set_cache_limit(&ignored, previousCache) }
            let loadStart = now()
            let model = try QwenModel(modelDirectory: directory)
            report["model_construction_seconds"] = seconds(loadStart, now())
            report["loaded_memory"] = try MX.memory()
            let generator = try QwenGenerator(model: model)
            func valid(_ result: QwenGenerationResult, _ request: QwenGenerationRequest) -> Bool {
                guard let last = result.tokens.last, result.phases != nil else { return false }
                let n = result.tokens.count, stats = result.statistics
                return n <= request.maxTokens && stats.promptTokenCount == request.tokens.count &&
                    stats.generatedTokenCount == n && stats.decodedTokenCount == n - 1 && stats.decodeRounds == n - 1 &&
                    stats.finalStateOffset == request.tokens.count + n - 1 && stats.mtpDepth == 0 &&
                    !result.tokens.dropLast().contains(where: generator.eosTokenIDs.contains) &&
                    (result.finishReason == .eos ? generator.eosTokenIDs.contains(last)
                        : n == request.maxTokens && !generator.eosTokenIDs.contains(last))
            }
            func reference(_ name: String, _ request: QwenGenerationRequest) throws -> QwenGenerationResult {
                var callbacks = [PDFairnessCallback](); callbacks.reserveCapacity(request.maxTokens)
                let start = now()
                let result = try generator.generate(request) { token in
                    callbacks.append(.init(index: callbacks.count, tokenID: token, timestampNS: now()))
                }
                references[name] = ["result": try object(result), "start_ns": start, "end_ns": now(),
                    "callbacks": try object(callbacks)]
                checks[name + "_valid"] = valid(result, request) && callbacks.map(\.tokenID) == result.tokens
                return result
            }
            // Short first makes the minimum-length check a cheap preflight, before any long prefill.
            let shortReference = try reference("short_ar", short)
            checks["short_frozen_exact"] = shortReference.tokens == frozen.short_output_tokens &&
                shortReference.finishReason.rawValue == frozen.short_finish_reason
            checks["short_actual_at_least_64"] = shortReference.tokens.count >= 64
            guard checks["short_ar_valid"] == true, checks["short_frozen_exact"] == true,
                  checks["short_actual_at_least_64"] == true else {
                report["outcome"] = "inconclusive_short_reference"
                try write(true); return
            }
            let longReference = try reference("long_ar", long)
            checks["long_frozen_exact"] = longReference.tokens == frozen.long_output_tokens &&
                longReference.finishReason.rawValue == frozen.long_finish_reason
            guard checks["long_ar_valid"] == true, checks["long_frozen_exact"] == true else {
                report["outcome"] = "inconclusive_long_reference"
                try write(true); return
            }
            try write()
            func phaseAccounting(_ event: QwenLocalScheduler.Event) -> Bool {
                guard let r = event.result, let p = r.phases,
                      let pw = event.timing.prefillQueueWaitSeconds, let dw = event.timing.readyQueueWaitSeconds,
                      let pa = event.timing.prefillStageSeconds, let da = event.timing.decodeStageSeconds else { return false }
                return [pw, dw, pa, da, event.timing.totalQueueWaitSeconds, event.timing.elapsedSeconds,
                    p.prefill.targetSeconds, p.prefill.draftHistorySeconds, p.prefill.totalSeconds,
                    p.decodeServiceSeconds, r.decodeSeconds, r.statistics.callbackSeconds].allSatisfy { $0.isFinite && $0 >= 0 } &&
                    abs(event.timing.totalQueueWaitSeconds - pw - dw) <= 1e-5 &&
                    event.timing.elapsedSeconds + 1e-3 >= pa + da + pw + dw &&
                    p.prefill.totalSeconds + 1e-5 >= p.prefill.targetSeconds + p.prefill.draftHistorySeconds &&
                    p.prefill.draftHistorySeconds == 0 &&
                    p.decodeServiceSeconds + 1e-5 >= r.decodeSeconds + r.statistics.callbackSeconds
            }
            // Every captured value is executor-local. The callback never calls scheduler methods.
            func pair(_ burst: Int, index: Int, cancellation: Bool = false) throws -> PDFairnessMetrics? {
                let limits = QwenLocalScheduler.Limits(maxQueuedPrefills: 2, maxReadyDecodes: 2,
                    maxResidentTokens: reservation, maxConsecutivePrefills: 1,
                    executionMode: .cooperative, decodeBurst: burst, maxResidentSequences: 2)
                let scheduler = try QwenLocalScheduler(generator: generator, limits: limits)
                defer { _ = try? scheduler.discardAll() }
                var shortCallbacks = [PDFairnessCallback](), longCallbacks = [PDFairnessCallback]()
                shortCallbacks.reserveCapacity(64); longCallbacks.reserveCapacity(128)
                var steps = [PDFairnessStep](); steps.reserveCapacity(512)
                var arrival: UInt64?, longSubmitStart: UInt64?, longSubmitEnd: UInt64?, arrivalStep: Int?
                var longID: UUID?, admittedSnapshot: QwenLocalScheduler.Snapshot?, cancellationBefore: QwenLocalScheduler.Snapshot?
                var callbackCountAtSubmit = 0, exhausted = true, cancellationTriggered = false
                let start = now(), shortSubmitted = now()
                let shortID = try scheduler.submit(short) { token in
                    let timestamp = now()
                    shortCallbacks.append(.init(index: shortCallbacks.count, tokenID: token, timestampNS: timestamp))
                    if shortCallbacks.count == 8 { arrival = timestamp }
                }
                let initial = scheduler.snapshot()
                var pumpError: Error?
                do {
                    for _ in 0..<1024 {
                        let stepStart = now()
                        guard let event = try scheduler.runNext() else { exhausted = false; break }
                        let stepEnd = now()
                        steps.append(.init(index: steps.count, startNS: stepStart, endNS: stepEnd,
                            event: event, snapshot: scheduler.snapshot()))
                        if arrival != nil && longID == nil {
                            arrivalStep = steps.count - 1; callbackCountAtSubmit = shortCallbacks.count
                            longSubmitStart = now()
                            longID = try scheduler.submit(long) { token in
                                longCallbacks.append(.init(index: longCallbacks.count, tokenID: token, timestampNS: now()))
                            }
                            longSubmitEnd = now(); admittedSnapshot = scheduler.snapshot()
                        }
                        if cancellation, event.jobID == longID, event.kind == .prefillProgress {
                            cancellationTriggered = true
                            cancellationBefore = scheduler.snapshot()
                            for id in [longID!, shortID] {
                                let cancelStart = now()
                                if let cancelled = try scheduler.cancel(id) {
                                    steps.append(.init(index: steps.count, startNS: cancelStart, endNS: now(),
                                        event: cancelled, snapshot: scheduler.snapshot()))
                                }
                            }
                            guard clean(scheduler) else { throw CLIError.usage("Cancellation did not release all jobs") }
                            exhausted = false; break
                        }
                    }
                } catch { pumpError = error }
                let end = now()
                let terminal = steps.filter { [.completed, .cancelled, .failed].contains($0.event.kind) }
                let shortDone = terminal.first { $0.event.jobID == shortID }
                let longDone = terminal.first { $0.event.jobID == longID }
                let terminalKind: QwenLocalScheduler.EventKind = cancellation ? .cancelled : .completed
                let completed = !exhausted && pumpError == nil && terminal.count == 2 &&
                    shortDone?.event.kind == terminalKind && longDone?.event.kind == terminalKind
                let bounded = initial.reservedTokens == short.tokens.count + short.maxTokens &&
                    admittedSnapshot?.reservedTokens == reservation && steps.allSatisfy {
                        (0...2).contains($0.snapshot.queuedPrefills) && (0...2).contains($0.snapshot.readyDecodes) &&
                        (0...2).contains($0.snapshot.residentSequences) && (0...reservation).contains($0.snapshot.reservedTokens)
                    }
                let arrivalValid = arrival.flatMap { a in arrivalStep.map { i in
                    steps[i].event.jobID == shortID && steps[i].event.kind == .decodeProgress &&
                    steps[i].startNS <= a && a <= steps[i].endNS && callbackCountAtSubmit == 8 &&
                    longSubmitStart.map { $0 >= steps[i].endNS } == true && longSubmitEnd.map { $0 >= longSubmitStart! } == true
                } } == true
                let interleaved = arrival.flatMap { a in shortDone.map { d in
                    steps.contains { $0.event.jobID == longID && $0.event.kind == .prefillProgress &&
                        ($0.event.processedPromptTokens ?? 0) > 0 && $0.startNS >= a && $0.endNS < d.endNS }
                } } == true
                let prefixExact = shortCallbacks.map(\.tokenID) == Array(shortReference.tokens.prefix(shortCallbacks.count)) &&
                    longCallbacks.map(\.tokenID) == Array(longReference.tokens.prefix(longCallbacks.count))
                let outputExact: Bool
                if cancellation {
                    outputExact = cancellationTriggered && cancellationBefore?.residentSequences == 2 &&
                        cancellationBefore?.reservedTokens == reservation && shortCallbacks.count == 8 && longCallbacks.isEmpty && prefixExact
                } else {
                    outputExact = shortDone?.event.result.map { valid($0, short) && $0.tokens == shortReference.tokens &&
                        $0.finishReason == shortReference.finishReason } == true &&
                        longDone?.event.result.map { valid($0, long) && $0.tokens == longReference.tokens &&
                            $0.finishReason == longReference.finishReason } == true &&
                        shortCallbacks.map(\.tokenID) == shortReference.tokens && longCallbacks.map(\.tokenID) == longReference.tokens
                }
                func callbackClocks(_ callbacks: [PDFairnessCallback], id: UUID?) -> Bool {
                    guard let id else { return false }
                    return zip(callbacks, callbacks.dropFirst()).allSatisfy { $0.0.timestampNS <= $0.1.timestampNS } &&
                        callbacks.enumerated().allSatisfy { item in
                            let index = item.offset, c = item.element
                            return c.index == index && steps.filter { $0.startNS <= c.timestampNS && c.timestampNS <= $0.endNS &&
                                $0.event.jobID == id && $0.event.stage == .decode }.count == 1
                        }
                }
                let clockValid = zip(steps, steps.dropFirst()).allSatisfy { $0.0.endNS <= $0.1.startNS } &&
                    steps.allSatisfy { $0.startNS <= $0.endNS } &&
                    callbackClocks(shortCallbacks, id: shortID) && callbackClocks(longCallbacks, id: longID)
                let phaseValid = cancellation || (completed && terminal.allSatisfy { phaseAccounting($0.event) })
                let rowChecks = ["terminal_exact": completed, "outputs_and_offsets_exact": outputExact,
                    "arrival_at_callback_8_and_submit_after_slice": arrivalValid, "actual_long_prefill_during_short": interleaved,
                    "bounded_and_released": bounded && clean(scheduler), "phase_accounting": phaseValid,
                    "clock_and_callback_ownership": clockValid]
                var row: [String: Any] = ["mode": "cooperative_arrival_burst\(burst)_group\(index)", "index": index,
                    "decode_burst": burst, "cancellation": cancellation, "limits": try object(limits),
                    "start_ns": start, "end_ns": end, "wall_seconds": seconds(start, end), "short_submit_ns": shortSubmitted,
                    "short_job_id": shortID.uuidString, "long_job_id": longID?.uuidString as Any? ?? NSNull(),
                    "arrival_ns": arrival as Any? ?? NSNull(), "long_submit_start_ns": longSubmitStart as Any? ?? NSNull(),
                    "long_submit_end_ns": longSubmitEnd as Any? ?? NSNull(), "arrival_step_index": arrivalStep as Any? ?? NSNull(),
                    "short_callback_count_at_submit": callbackCountAtSubmit, "submitted_snapshot": try object(initial),
                    "admitted_snapshot": try admittedSnapshot.map { try object($0) } ?? NSNull(),
                    "steps": try object(steps), "short_callbacks": try object(shortCallbacks), "long_callbacks": try object(longCallbacks),
                    "final_snapshot": try object(scheduler.snapshot()), "checks": rowChecks, "pump_exhausted_bound": exhausted]
                if let cancellationBefore { row["before_cancel"] = try object(cancellationBefore) }
                if let pumpError { row["pump_error"] = String(describing: pumpError) }
                var metric: PDFairnessMetrics?
                if !cancellation, let a = arrival, let s = shortDone, let l = longDone,
                   let firstLong = longCallbacks.first, let submit = longSubmitStart, let lastShort = shortCallbacks.last,
                   shortCallbacks.count >= 64 {
                    let gaps = zip(shortCallbacks.dropFirst(7), shortCallbacks.dropFirst(8)).map { seconds($0.0.timestampNS, $0.1.timestampNS) }
                    let interrupted = zip(shortCallbacks.dropFirst(7), shortCallbacks.dropFirst(8)).filter { pair in
                        let before = pair.0, after = pair.1
                        return steps.contains { $0.event.jobID == longID && $0.event.stage == .prefill &&
                            $0.startNS >= before.timestampNS && $0.endNS <= after.timestampNS }
                    }.map { seconds($0.0.timestampNS, $0.1.timestampNS) }
                    row["short_callback_metrics"] = ["submit_ns": shortSubmitted]
                    row["long_callback_metrics"] = ["submit_ns": submit]
                    row["short_after_arrival_callback_gaps"] = summary(gaps)
                    // AR emits exactly one token per decode slice. Every post-arrival callback gap
                    // is also an inter-slice output burst gap, including ordinary decode intervals.
                    row["short_after_arrival_burst_gaps"] = summary(gaps)
                    row["short_prefill_interruption_gaps"] = summary(interrupted)
                    row["short_last_callback_remaining_seconds"] = seconds(a, lastShort.timestampNS)
                    row["short_remaining_seconds"] = seconds(a, s.endNS)
                    row["long_arrival_to_first_callback_seconds"] = seconds(a, firstLong.timestampNS)
                    row["long_submit_to_first_callback_seconds"] = seconds(submit, firstLong.timestampNS)
                    row["long_arrival_to_terminal_seconds"] = seconds(a, l.endNS)
                    row["short_phase_timing"] = try object(s.event.timing)
                    row["long_phase_timing"] = try object(l.event.timing)
                    if let result = s.event.result {
                        row["short_compute_decode_seconds"] = result.decodeSeconds
                        row["short_prefill_phases"] = try result.phases.map { try object($0.prefill) } ?? NSNull()
                    }
                    if let result = l.event.result {
                        row["long_compute_decode_seconds"] = result.decodeSeconds
                        row["long_prefill_phases"] = try result.phases.map { try object($0.prefill) } ?? NSNull()
                    }
                    // Terminal events retain full phase accounting separately from wall callback gaps.
                    metric = .init(remaining: seconds(a, s.endNS), wall: seconds(start, end),
                        longTTFT: seconds(a, firstLong.timestampNS), maxGap: gaps.max() ?? 0)
                }
                if cancellation { report["cancellation"] = row } else { groups.append(row) }
                checks[cancellation ? "arrival_cancellation_lifecycle" : "group_\(index)_correctness"] = rowChecks.values.allSatisfy { $0 }
                try write()
                guard rowChecks.values.allSatisfy({ $0 }) else { throw CLIError.usage("PD fairness group \(index) correctness failed") }
                return metric
            }
            var metrics = [PDFairnessMetrics]()
            for (index, burst) in [4, 8, 8, 4].enumerated() {
                guard let value = try pair(burst, index: index) else { throw CLIError.usage("Missing fairness metrics") }
                metrics.append(value)
            }
            var comparisons = [[String: Any]](), pairPasses = [Bool](), maxGapRegressions = [Bool]()
            for (baselineIndex, candidateIndex) in [(0, 1), (3, 2)] {
                let b = metrics[baselineIndex], c = metrics[candidateIndex]
                guard [b.remaining, b.wall, b.longTTFT, b.maxGap, c.remaining, c.wall, c.longTTFT, c.maxGap]
                    .allSatisfy({ $0.isFinite && $0 > 0 }) else { throw CLIError.usage("Invalid comparison time") }
                let gain = 1 - c.remaining / b.remaining, wall = c.wall / b.wall, ttft = c.longTTFT / b.longTTFT, gap = c.maxGap / b.maxGap
                let passed = gain >= 0.15 && wall <= 1.03 && ttft <= 1.10
                pairPasses.append(passed); maxGapRegressions.append(gap > 1.05)
                comparisons.append(["baseline_group": baselineIndex, "candidate_group": candidateIndex,
                    "short_remaining_improvement_fraction": gain, "group_wall_ratio": wall, "long_arrival_ttft_ratio": ttft,
                    "short_max_gap_ratio": gap, "remaining_wall_ttft_passed": passed, "max_gap_regression_over_5pct": gap > 1.05])
            }
            report["comparisons"] = comparisons
            report["outer_burst4_observed_ratios"] = ["group3_over_group0_remaining": metrics[3].remaining / metrics[0].remaining,
                "group3_over_group0_wall": metrics[3].wall / metrics[0].wall,
                "group3_over_group0_long_ttft": metrics[3].longTTFT / metrics[0].longTTFT]
            report["outer_baseline_drift_note"] = "Raw repeated-burst4 ratios are observations, not a cause or a new drift cutoff. Unstable timing can still make this single screen inconclusive."
            let repeatedGapRegression = maxGapRegressions.allSatisfy { $0 }
            report["max_gap_repeated_regression_over_5pct"] = repeatedGapRegression
            report["performance_screen_passed"] = pairPasses.allSatisfy { $0 } && !repeatedGapRegression
            _ = try pair(4, index: 4, cancellation: true)
            let fresh = try reference("fresh_ar_after_cancellation", short)
            checks["fresh_after_cancellation_exact"] = valid(fresh, short) && fresh.tokens == shortReference.tokens &&
                fresh.finishReason == shortReference.finishReason
            let correctness = checks.values.allSatisfy { $0 }
            report["correctness_passed"] = correctness
            report["final_memory"] = try MX.memory()
            report["outcome"] = correctness ? (pairPasses.allSatisfy { $0 } && !repeatedGapRegression ? "screen_passed" : "screen_not_passed") : "correctness_failed"
            report["passed"] = correctness
            try write(true)
            guard correctness else { throw CLIError.usage("PD fairness lifecycle gate failed") }
        } catch {
            report["fatal_error"] = String(describing: error)
            report["passed"] = false
            try write()
            throw error
        }
    }
}

private struct PDFairnessFrozen: Decodable {
    let schema, short_text, short_finish_reason, long_finish_reason: String
    let short_tokens, long_tokens, short_output_tokens, long_output_tokens: [Int32]
}
private struct PDFairnessCallback: Codable {
    let index: Int
    let tokenID: Int32
    let timestampNS: UInt64
}
private struct PDFairnessStep: Encodable {
    let index: Int
    let startNS, endNS: UInt64
    let event: QwenLocalScheduler.Event
    let snapshot: QwenLocalScheduler.Snapshot
}
private struct PDFairnessMetrics {
    let remaining, wall, longTTFT, maxGap: Double
}
