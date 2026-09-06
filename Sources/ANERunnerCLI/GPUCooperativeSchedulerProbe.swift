import ANERunnerGPU
import CMLX
import CryptoKit
import Foundation

extension RunnerCLI {
    /// Fixed long/short requests compare whole-stage and cooperative scheduling.
    /// GPU execution remains serial; callback clocks measure observed latency.
    static func probeGPUCooperativeScheduler(_ args: Arguments) throws {
        try args.validate(["--model-dir", "--tokens-file", "--output", "--golden-report", "--decode-burst"])
        guard let decodeBurst = Int(args["--decode-burst"] ?? "4"), (1...64).contains(decodeBurst) else {
            throw CLIError.usage("Cooperative scheduler probe --decode-burst must be in 1...64")
        }
        let output = try args.require("--output")
        guard !FileManager.default.fileExists(atPath: output) else {
            throw CLIError.usage("Cooperative scheduler probe --output must be a new file")
        }
        var required = ["independent_ar_references", "short_mtp_reference_exact_and_exercised",
            "wholeStages_outputs_exact", "wholeStages_phase_accounting", "wholeStages_bounds_and_release",
            "wholeStages_no_chunk_interleaving", "cooperative_outputs_exact", "cooperative_phase_accounting",
            "cooperative_bounds_and_release", "cooperative_actual_interleaving",
            "cooperative_long_mtp_outputs_exact", "cooperative_long_mtp_phase_accounting",
            "cooperative_long_mtp_bounds_and_release", "cooperative_long_mtp_actual_interleaving",
            "cancel_yielded_prefill", "cancel_yielded_decode", "fresh_after_cancellations"]
        if args["--golden-report"] != nil { required.append("golden_long_exact") }
        var checks = Dictionary(uniqueKeysWithValues: required.map { ($0, false) })
        var report: [String: Any] = [
            "schema": "qwen38-cooperative-scheduler-probe-v1", "complete": false, "passed": false,
            "full_model_instances": 1, "mode_order": ["wholeStages", "cooperative", "cooperative_long_mtp"],
            "clock": "mach_absolute_time_nanoseconds", "long_max_tokens": 128, "short_max_tokens": 64,
            "decode_burst": decodeBurst,
            "callback_gap_quantile_method": "linear interpolation at q*(n-1), including zero gaps",
            "notes": [
                "One mixed long-AR/short-MTP pair per scheduling mode, then a cooperative long-MTP/short-MTP pair. Long is submitted first. This is a functional and observed latency gate, not a statistical kernel speedup benchmark.",
                "Standalone long AR, short AR and short MTP references run before both modes; short MTP initialization is therefore warmed for both.",
                "All raw callback timestamps are recorded with committed token IDs. Mean compute TPOT differs from wall callback gaps under cooperative scheduling.",
                "Callback p50/p95/max include every adjacent committed-token gap, including near-zero gaps within MTP rounds. They measure local delivery, not HTTP/SSE latency.",
                "Submission-to-terminal observation ends when the terminal runNext returns; it is separate from the last callback and active decode compute time.",
                "Callbacks only append a timestamped value. Event/JSON serialization and file writes occur after each mixed group, outside its pump loop.",
                "Cancellation checks call cancel only after a yielded slice. No further runNext is used to complete the cancelled job; cancel latency may include SSD cleanup.",
                "Logical token reservations are not physical memory bytes. No concurrent GPU execution, HTTP service, process transfer or kernel preemption is tested.",
                "Every required check starts false. Missing progress events, callbacks or references never count as covered."
            ]
        ]
        var references = [String: Any](), groups = [[String: Any]](), cancellations = [[String: Any]]()
        func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
        func seconds(_ start: UInt64, _ end: UInt64) -> Double { Double(end - start) * 1e-9 }
        func object<T: Encodable>(_ value: T) throws -> Any {
            try JSONSerialization.jsonObject(with: JSONEncoder().encode(value))
        }
        func record(_ name: String, _ passed: Bool) {
            checks[name] = passed
            FileHandle.standardError.write(Data("Cooperative scheduler probe: \(name): \(passed ? "passed" : "failed")\n".utf8))
        }
        func write(_ complete: Bool = false) throws {
            report["complete"] = complete
            report["passed"] = complete && required.allSatisfy { checks[$0] == true }
            report["checks"] = required.map { ["id": $0, "passed": checks[$0] == true] as [String: Any] }
            report["references"] = references; report["groups"] = groups; report["cancellations"] = cancellations
            try emit(report, to: output)
        }
        do {
            let directory = URL(fileURLWithPath: try args.require("--model-dir"))
                .standardizedFileURL.resolvingSymlinksInPath()
            let inputURL = URL(fileURLWithPath: try args.require("--tokens-file")).standardizedFileURL
            let inputData = try Data(contentsOf: inputURL)
            let longTokens = try JSONDecoder().decode([Int32].self, from: inputData)
            guard longTokens.count == 11_057 else {
                throw CLIError.usage("This fixed cooperative probe requires the real 11057-token agent prompt")
            }
            let tokenizer = try QwenTokenizer(modelDirectory: directory)
            let shortText = "请用三句话介绍太阳为什么发光，以及阳光如何到达地球。"
            let shortTokens = try tokenizer.encode(tokenizer.renderChat(messages: [ChatMessage(role: "user", content: shortText)]))
            let configuration = try QwenConfiguration(modelDirectory: directory)
            let longRequest = QwenGenerationRequest(tokens: longTokens, maxTokens: 128, contextLimit: 16_384)
            let longMTP = QwenGenerationRequest(tokens: longTokens, maxTokens: 128, contextLimit: 16_384,
                mtpDepth: 2, verification: .batchedScalarLinear, draftHistoryTokens: 1024)
            let shortAR = QwenGenerationRequest(tokens: shortTokens, maxTokens: 64, contextLimit: 4096)
            let shortMTP = QwenGenerationRequest(tokens: shortTokens, maxTokens: 64, contextLimit: 4096,
                mtpDepth: 2, verification: .batchedScalarLinear, draftHistoryTokens: 1024)
            for request in [longRequest, longMTP, shortAR, shortMTP] { try request.validate(configuration: configuration) }
            let reservation = longTokens.count + 128 + shortTokens.count + 64
            report["model_directory"] = directory.path
            report["long_input"] = ["path": inputURL.path, "token_ids": longTokens,
                "sha256": SHA256.hash(data: inputData).map { String(format: "%02x", $0) }.joined()]
            report["short_input"] = ["text": shortText, "token_ids": shortTokens, "thinking": false]
            report["prefill_chunk"] = 416
            report["short_mtp"] = ["depth": 2, "verification": "batchedScalarLinear", "draft_history_limit": 1024]
            var previousCache = 0
            try MX.check(mlx_set_cache_limit(&previousCache, 256 * 1024 * 1024), "cooperative probe allocation cache")
            defer { var ignored = 0; _ = mlx_set_cache_limit(&ignored, previousCache) }
            let modelStart = now()
            let model = try QwenModel(modelDirectory: directory) { current, total in
                if current % 8 == 0 || current == total {
                    FileHandle.standardError.write(Data("Cooperative scheduler probe: loaded \(current)/\(total)\n".utf8))
                }
            }
            report["model_construction_seconds"] = seconds(modelStart, now())
            report["loaded_memory"] = try MX.memory()
            let generator = try QwenGenerator(model: model)
            func valid(_ result: QwenGenerationResult, _ request: QwenGenerationRequest) -> Bool {
                guard let last = result.tokens.last, result.phases != nil else { return false }
                return result.tokens.count <= request.maxTokens &&
                    result.statistics.promptTokenCount == request.tokens.count &&
                    result.statistics.generatedTokenCount == result.tokens.count &&
                    result.statistics.decodedTokenCount == result.tokens.count - 1 &&
                    result.statistics.mtpDepth == request.mtpDepth &&
                    !result.tokens.dropLast().contains(where: generator.eosTokenIDs.contains) &&
                    (result.finishReason == .eos ? generator.eosTokenIDs.contains(last)
                        : result.tokens.count == request.maxTokens && !generator.eosTokenIDs.contains(last))
            }
            func reference(_ name: String, _ request: QwenGenerationRequest) throws -> QwenGenerationResult {
                var callbacks = [GPUCooperativeCallback]()
                callbacks.reserveCapacity(request.maxTokens)
                let start = now()
                let result = try generator.generate(request) { token in
                    callbacks.append(.init(index: callbacks.count, tokenID: token, timestampNS: now()))
                }
                let end = now()
                references[name] = ["result": try object(result), "start_ns": start, "end_ns": end,
                    "wall_seconds": seconds(start, end), "callbacks": try object(callbacks)]
                guard valid(result, request), callbacks.map(\.tokenID) == result.tokens else {
                    throw CLIError.usage("Invalid independent reference: \(name)")
                }
                return result
            }
            let longReference = try reference("long_ar", longRequest)
            let shortReference = try reference("short_ar", shortAR)
            record("independent_ar_references", longReference.tokens.count == 128 && shortReference.tokens.count > 2)
            if let path = args["--golden-report"] {
                let data = try Data(contentsOf: URL(fileURLWithPath: path))
                let source = try JSONSerialization.jsonObject(with: data) as? [String: Any]
                let trial = (source?["trials"] as? [[String: Any]])?.first
                let golden = (trial?["generated_token_ids"] as? [NSNumber])?.map { $0.int32Value }
                let prompt = (trial?["prompt_tokens"] as? [NSNumber])?.map { $0.int32Value }
                let matches = golden == longReference.tokens && prompt == longTokens && source?["max_tokens"] as? Int == 128
                report["golden"] = ["path": path, "passed": matches,
                    "sha256": SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()]
                record("golden_long_exact", matches)
                guard matches else { throw CLIError.usage("Long standalone reference does not match the frozen golden") }
            }
            let warmMTP = try reference("short_mtp_warm", shortMTP)
            let warmMatches = warmMTP.tokens == shortReference.tokens && warmMTP.finishReason == shortReference.finishReason &&
                (warmMTP.statistics.mtp?.draftedTokens ?? 0) > 0
            record("short_mtp_reference_exact_and_exercised", warmMatches)
            try write()
            guard checks["independent_ar_references"] == true, warmMatches else {
                throw CLIError.usage("Independent output gate failed before mixed scheduling")
            }
            func clean(_ scheduler: QwenLocalScheduler) -> Bool {
                let s = scheduler.snapshot()
                return s.isIdle && s.queuedPrefills == 0 && s.readyDecodes == 0 &&
                    s.reservedTokens == 0 && s.residentSequences == 0 && s.pendingEvents == 0
            }
            func phaseAccounting(_ event: QwenLocalScheduler.Event) -> Bool {
                guard let result = event.result, let phases = result.phases,
                      let prefillWait = event.timing.prefillQueueWaitSeconds,
                      let decodeWait = event.timing.readyQueueWaitSeconds,
                      let prefillActive = event.timing.prefillStageSeconds,
                      let decodeActive = event.timing.decodeStageSeconds else { return false }
                let values = [prefillWait, decodeWait, prefillActive, decodeActive,
                    event.timing.totalQueueWaitSeconds, event.timing.elapsedSeconds,
                    phases.prefill.totalSeconds, phases.prefill.targetSeconds, phases.prefill.draftHistorySeconds,
                    phases.decodeServiceSeconds, result.decodeSeconds, result.statistics.callbackSeconds]
                return values.allSatisfy { $0.isFinite && $0 >= 0 } &&
                    abs(event.timing.totalQueueWaitSeconds - prefillWait - decodeWait) <= 1e-5 &&
                    event.timing.elapsedSeconds + 1e-3 >= prefillActive + decodeActive + event.timing.totalQueueWaitSeconds &&
                    phases.prefill.totalSeconds + 1e-5 >= phases.prefill.targetSeconds + phases.prefill.draftHistorySeconds &&
                    phases.decodeServiceSeconds + 1e-5 >= result.decodeSeconds + result.statistics.callbackSeconds
            }
            var shortTTFT = [String: Double]()
            for (name, cooperative, activeLong) in [
                ("wholeStages", false, longRequest), ("cooperative", true, longRequest),
                ("cooperative_long_mtp", true, longMTP)
            ] {
                let limits = QwenLocalScheduler.Limits(maxQueuedPrefills: 2, maxReadyDecodes: 2,
                    maxResidentTokens: reservation, maxConsecutivePrefills: 1,
                    executionMode: cooperative ? .cooperative : .wholeStages, decodeBurst: decodeBurst, maxResidentSequences: 2)
                let scheduler = try QwenLocalScheduler(generator: generator, limits: limits)
                defer { _ = try? scheduler.discardAll() }
                var longCallbacks = [GPUCooperativeCallback](), shortCallbacks = [GPUCooperativeCallback]()
                longCallbacks.reserveCapacity(128); shortCallbacks.reserveCapacity(64)
                var steps = [GPUCooperativeStep](); steps.reserveCapacity(512)
                let start = now()
                let longSubmitted = now()
                let longID = try scheduler.submit(activeLong) { token in
                    longCallbacks.append(.init(index: longCallbacks.count, tokenID: token, timestampNS: now()))
                }
                let shortSubmitted = now()
                let shortID = try scheduler.submit(shortMTP) { token in
                    shortCallbacks.append(.init(index: shortCallbacks.count, tokenID: token, timestampNS: now()))
                }
                let submittedSnapshot = scheduler.snapshot()
                var pumpError: Error?, exhausted = true
                do {
                    for index in 0..<1024 {
                        let stepStart = now()
                        guard let event = try scheduler.runNext() else { exhausted = false; break }
                        let stepEnd = now()
                        steps.append(.init(index: index, startNS: stepStart, endNS: stepEnd,
                                           event: event, snapshot: scheduler.snapshot()))
                    }
                } catch { pumpError = error }
                let end = now()
                let terminals = steps.filter { [.completed, .cancelled, .failed].contains($0.event.kind) }
                let longDone = terminals.first { $0.event.jobID == longID }
                let shortDone = terminals.first { $0.event.jobID == shortID }
                let longReady = steps.first { $0.event.jobID == longID && $0.event.kind == .prefillReady }
                let shortReady = steps.first { $0.event.jobID == shortID && $0.event.kind == .prefillReady }
                let complete = !exhausted && pumpError == nil && terminals.count == 2 &&
                    longDone?.event.kind == .completed && shortDone?.event.kind == .completed
                let longExact = longDone?.event.result.map { valid($0, activeLong) &&
                    $0.tokens == longReference.tokens && $0.finishReason == longReference.finishReason &&
                    (activeLong.mtpDepth == 0 || ($0.statistics.mtp?.draftedTokens ?? 0) > 0) } == true
                let shortExact = shortDone?.event.result.map { valid($0, shortMTP) && $0.tokens == shortReference.tokens &&
                    $0.finishReason == shortReference.finishReason && ($0.statistics.mtp?.draftedTokens ?? 0) > 0 } == true
                let exact = complete && longExact && shortExact &&
                    longCallbacks.map(\.tokenID) == longReference.tokens && shortCallbacks.map(\.tokenID) == shortReference.tokens
                let bounded = submittedSnapshot.reservedTokens == reservation && steps.allSatisfy {
                    $0.snapshot.queuedPrefills <= 2 && $0.snapshot.readyDecodes <= 2 &&
                    (0...2).contains($0.snapshot.residentSequences) && (0...reservation).contains($0.snapshot.reservedTokens)
                }
                record(name + "_outputs_exact", exact)
                record(name + "_bounds_and_release", complete && bounded && clean(scheduler))
                record(name + "_phase_accounting", complete && terminals.allSatisfy { phaseAccounting($0.event) })
                if cooperative {
                    let shortSecond = shortCallbacks.count > 1 ? shortCallbacks[1].timestampNS : nil
                    let crossed = longReady.flatMap { l in shortReady.map { s in
                        s.index < l.index && shortSecond.map { $0 < l.endNS } == true &&
                        steps.contains { $0.event.jobID == shortID && $0.event.kind == .decodeProgress && $0.index < l.index } &&
                        steps.contains { $0.event.jobID == longID && $0.event.kind == .prefillProgress && $0.index < s.index }
                    } } ?? false
                    record(name + "_actual_interleaving", crossed)
                } else {
                    record("wholeStages_no_chunk_interleaving", longReady != nil && shortReady != nil &&
                        longDone.map { l in shortReady.map { $0.index > l.index } == true } == true &&
                        !steps.contains { [.prefillProgress, .decodeProgress].contains($0.event.kind) })
                }
                func callbackMetrics(_ callbacks: [GPUCooperativeCallback], submitted: UInt64,
                                     terminalObserved: UInt64?) -> [String: Any] {
                    var metrics: [String: Any] = ["submit_ns": submitted, "count": callbacks.count]
                    if let first = callbacks.first {
                        metrics["submission_to_first_callback_seconds"] = seconds(submitted, first.timestampNS)
                    }
                    if let last = callbacks.last { metrics["submission_to_last_callback_seconds"] = seconds(submitted, last.timestampNS) }
                    let gaps = zip(callbacks, callbacks.dropFirst()).map { pair in
                        seconds(pair.0.timestampNS, pair.1.timestampNS)
                    }
                    metrics["callback_gap_seconds"] = gaps
                    let sorted = gaps.sorted()
                    func quantile(_ q: Double) -> Any {
                        guard !sorted.isEmpty else { return NSNull() }
                        let rank = q * Double(sorted.count - 1)
                        let lower = Int(rank.rounded(.down)), upper = Int(rank.rounded(.up))
                        return sorted[lower] + (sorted[upper] - sorted[lower]) * (rank - Double(lower))
                    }
                    metrics["callback_gap_count"] = gaps.count
                    metrics["callback_gap_p50_seconds"] = quantile(0.5)
                    metrics["callback_gap_p95_seconds"] = quantile(0.95)
                    metrics["callback_gap_max_seconds"] = sorted.last.map { $0 as Any } ?? NSNull()
                    metrics["terminal_observed_ns"] = terminalObserved.map { $0 as Any } ?? NSNull()
                    metrics["submission_to_terminal_observation_seconds"] = terminalObserved.map {
                        seconds(submitted, $0) as Any
                    } ?? NSNull()
                    return metrics
                }
                if let first = shortCallbacks.first { shortTTFT[name] = seconds(shortSubmitted, first.timestampNS) }
                var group: [String: Any] = ["mode": name, "long_mtp_depth": activeLong.mtpDepth,
                    "limits": try object(limits), "start_ns": start, "end_ns": end,
                    "wall_seconds": seconds(start, end), "long_job_id": longID.uuidString, "short_job_id": shortID.uuidString,
                    "submitted_snapshot": try object(submittedSnapshot), "steps": try object(steps),
                    "long_callbacks": try object(longCallbacks), "short_callbacks": try object(shortCallbacks),
                    "long_callback_metrics": callbackMetrics(longCallbacks, submitted: longSubmitted,
                        terminalObserved: longDone?.endNS),
                    "short_callback_metrics": callbackMetrics(shortCallbacks, submitted: shortSubmitted,
                        terminalObserved: shortDone?.endNS),
                    "final_snapshot": try object(scheduler.snapshot()), "memory_after": try MX.memory(),
                    "pump_exhausted_bound": exhausted]
                if let pumpError { group["error"] = String(describing: pumpError) }
                groups.append(group); try write()
                if let pumpError { throw pumpError }
                guard complete else { throw CLIError.usage("Mixed \(name) group did not complete both jobs") }
            }
            if let whole = shortTTFT["wholeStages"], let coop = shortTTFT["cooperative"] {
                report["short_latency_observation"] = ["whole_stage_ttft_seconds": whole,
                    "cooperative_ttft_seconds": coop, "saved_seconds": whole - coop,
                    "cooperative_to_whole_ratio": whole > 0 ? coop / whole as Any : NSNull(),
                    "fixed_percentage_release_gate": false]
            }

            // Short inputs exercise cancellation of live cursors without paying
            // for an additional 11k prefill. The decode case keeps reference math.
            for cancellingPrefill in [true, false] {
                let name = cancellingPrefill ? "cancel_yielded_prefill" : "cancel_yielded_decode"
                let request = QwenGenerationRequest(tokens: shortTokens, maxTokens: 64, contextLimit: 4096,
                    prefillChunk: cancellingPrefill ? 8 : 416)
                let scheduler = try QwenLocalScheduler(generator: generator, limits: .init(
                    maxQueuedPrefills: 1, maxReadyDecodes: 1, maxResidentTokens: shortTokens.count + 64,
                    maxConsecutivePrefills: 1, executionMode: .cooperative, decodeBurst: 4, maxResidentSequences: 1))
                defer { _ = try? scheduler.discardAll() }
                var callbacks = [GPUCooperativeCallback](), steps = [GPUCooperativeStep]()
                callbacks.reserveCapacity(64)
                let id = try scheduler.submit(request) { token in
                    callbacks.append(.init(index: callbacks.count, tokenID: token, timestampNS: now()))
                }
                var yielded = false
                for index in 0..<64 {
                    let start = now()
                    guard let event = try scheduler.runNext() else { break }
                    let end = now()
                    steps.append(.init(index: index, startNS: start, endNS: end, event: event, snapshot: scheduler.snapshot()))
                    if cancellingPrefill && event.kind == .prefillProgress { yielded = true; break }
                    if !cancellingPrefill && event.kind == .decodeProgress && callbacks.count >= 2 { yielded = true; break }
                    if [.completed, .cancelled, .failed].contains(event.kind) { break }
                }
                let before = scheduler.snapshot(), callbackCount = callbacks.count
                let cancelStart = now()
                let cancelled = try scheduler.cancel(id)
                let cancelEnd = now()
                let secondCancel = try scheduler.cancel(id)
                let after = scheduler.snapshot()
                let callbackGate = cancellingPrefill ? callbacks.isEmpty :
                    callbacks.map(\.tokenID) == Array(shortReference.tokens.prefix(callbackCount)) && callbackCount >= 2
                record(name, yielded && before.residentSequences == 1 && before.reservedTokens == shortTokens.count + 64 &&
                    cancelled?.jobID == id && cancelled?.kind == .cancelled && secondCancel == nil &&
                    callbackCount == callbacks.count && callbackGate && clean(scheduler))
                var row: [String: Any] = ["name": name, "yield_reached": yielded, "job_id": id.uuidString,
                    "prefill_chunk": request.prefillChunk, "steps": try object(steps), "callbacks": try object(callbacks),
                    "before_cancel": try object(before), "after_cancel": try object(after),
                    "cancel_start_ns": cancelStart, "cancel_end_ns": cancelEnd,
                    "cancel_wall_seconds": seconds(cancelStart, cancelEnd), "run_next_calls_after_cancel": 0]
                if let cancelled { row["cancel_event"] = try object(cancelled) }
                cancellations.append(row); try write()
            }
            let fresh = try reference("fresh_ar_after_cancellations", shortAR)
            record("fresh_after_cancellations", fresh.tokens == shortReference.tokens && fresh.finishReason == shortReference.finishReason)
            report["final_memory"] = try MX.memory()
            report["loaded_source_weight_bytes_at_report"] = model.weights.cachedSourceBytes
            try write(true)
            guard required.allSatisfy({ checks[$0] == true }) else {
                throw CLIError.usage("Cooperative scheduler probe checks failed; see \(output)")
            }
        } catch {
            report["fatal_error"] = String(describing: error)
            try write()
            throw error
        }
    }
}

private struct GPUCooperativeCallback: Codable {
    let index: Int
    let tokenID: Int32
    let timestampNS: UInt64
}

private struct GPUCooperativeStep: Encodable {
    let index: Int
    let startNS: UInt64
    let endNS: UInt64
    let event: QwenLocalScheduler.Event
    let snapshot: QwenLocalScheduler.Snapshot
}
