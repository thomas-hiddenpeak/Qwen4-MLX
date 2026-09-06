import ANERunnerGPU
import CMLX
import CryptoKit
import Foundation

extension RunnerCLI {
    /// Real-model acceptance for the cooperative local scheduler. Stage order
    /// is deliberate; queue delay is not interpreted as kernel performance.
    static func probeGPULocalScheduler(_ args: Arguments) throws {
        try args.validate(["--model-dir", "--output", "--tokens-file", "--max-tokens", "--suite", "--golden-report"])
        let output = try args.require("--output")
        let suite = args["--suite"] ?? "all"
        guard ["all", "mixed"].contains(suite),
              let maximum = Int(args["--max-tokens"] ?? "64"), (3...256).contains(maximum),
              !FileManager.default.fileExists(atPath: output) else {
            throw CLIError.usage("Scheduler probe requires a new output, --suite all|mixed, --max-tokens 3...256")
        }
        let required = ["independent_references", "two_ready_states", "stage_fifo_order",
                        "ar_exact", "mtp_exact_and_exercised", "all_reservations_released", "phase_accounting"]
            + (suite == "all" ? ["queue_backpressure", "token_backpressure", "invalid_request_rejected",
                "queued_cancellation", "ready_cancellation", "capacity_reusable", "ready_queue_bound",
                "active_cancellation", "callback_busy_is_terminal", "callback_reentry_rejected",
                "submit_during_callback", "fresh_after_failures", "discard_all"] : [])
        var checks = Dictionary(uniqueKeysWithValues: required.map { ($0, false) })
        var report: [String: Any] = ["schema": "qwen38-local-scheduler-probe-v1", "suite": suite,
            "complete": false, "passed": false, "full_model_instances": 1,
            "max_output_tokens": maximum,
            "notes": ["Shared model, same inference executor, non-preemptive stages. No HTTP or parallel GPU execution.",
                      "Token reservations include queued, ready and running requests; they are not physical byte measurements.",
                      "Two prefills intentionally run before decode to validate state isolation; queue delay is reported separately.",
                      "Timings from this functional gate are observations, not a stable performance release gate."]]
        var records = [[String: Any]]()
        func object<T: Encodable>(_ value: T) throws -> Any {
            try JSONSerialization.jsonObject(with: JSONEncoder().encode(value))
        }
        func record(_ name: String, _ passed: Bool) {
            checks[name] = passed
            FileHandle.standardError.write(Data("Local scheduler probe: \(name): \(passed ? "passed" : "failed")\n".utf8))
        }
        func write(_ complete: Bool = false) throws {
            report["complete"] = complete
            report["passed"] = complete && required.allSatisfy { checks[$0] == true }
            report["checks"] = required.map { ["id": $0, "passed": checks[$0] == true] as [String: Any] }
            report["events"] = records
            try emit(report, to: output)
        }
        func save(_ event: QwenLocalScheduler.Event, _ scheduler: QwenLocalScheduler, _ group: String) throws {
            records.append(["group": group, "event": try object(event), "snapshot": try object(scheduler.snapshot())])
        }
        func step(_ scheduler: QwenLocalScheduler, _ group: String) throws -> QwenLocalScheduler.Event? {
            let event = try scheduler.runNext()
            if let event { try save(event, scheduler, group) }
            return event
        }
        func drain(_ scheduler: QwenLocalScheduler, _ group: String) throws -> [QwenLocalScheduler.Event] {
            var events = [QwenLocalScheduler.Event]()
            for _ in 0..<64 {
                guard let event = try step(scheduler, group) else {
                    guard scheduler.snapshot().isIdle else { throw CLIError.usage("Scheduler stalled with pending work") }
                    return events
                }
                events.append(event)
            }
            throw CLIError.usage("Scheduler exceeded bounded probe event count")
        }
        func clean(_ scheduler: QwenLocalScheduler) -> Bool {
            let s = scheduler.snapshot()
            return s.isIdle && s.queuedPrefills == 0 && s.readyDecodes == 0 && s.reservedTokens == 0 && s.pendingEvents == 0
        }

        do {
            let directory = URL(fileURLWithPath: try args.require("--model-dir"))
                .standardizedFileURL.resolvingSymlinksInPath()
            let tokenizer = try QwenTokenizer(modelDirectory: directory)
            func chat(_ text: String) throws -> [Int32] {
                try tokenizer.encode(tokenizer.renderChat(messages: [ChatMessage(role: "user", content: text)]))
            }
            let tokensA: [Int32], tokensB: [Int32]
            if let path = args["--tokens-file"] {
                let data = try Data(contentsOf: URL(fileURLWithPath: path))
                tokensA = try JSONDecoder().decode([Int32].self, from: data)
                tokensB = tokensA
                report["input"] = ["path": path, "sha256": SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined(),
                    "kind": "same_real_document_prompt_ar_and_mtp"]
            } else {
                tokensA = try chat("请用三句话介绍太阳为什么发光，以及阳光如何到达地球。")
                tokensB = try chat("请按顺序列出水循环的四个阶段，每个阶段用一句话解释。")
                report["input"] = ["kind": "two_distinct_chat_prompts"]
            }
            let configuration = try QwenConfiguration(modelDirectory: directory)
            let context = min(configuration.maximumPositions, max(4096, max(tokensA.count, tokensB.count) + maximum))
            let ar = QwenGenerationRequest(tokens: tokensA, maxTokens: maximum, contextLimit: context)
            let arB = QwenGenerationRequest(tokens: tokensB, maxTokens: maximum, contextLimit: context)
            let mtp = QwenGenerationRequest(tokens: tokensB, maxTokens: maximum, contextLimit: context,
                mtpDepth: 2, verification: .batchedScalarLinear, draftHistoryTokens: 1024)
            let one = QwenGenerationRequest(tokens: tokensA, maxTokens: 1, contextLimit: context)
            for request in [ar, arB, mtp, one] { try request.validate(configuration: configuration) }
            let reservation = tokensA.count + tokensB.count + 2 * maximum
            report["model_directory"] = directory.path
            report["prompt_token_ids_a"] = tokensA; report["prompt_token_ids_b"] = tokensB
            report["context_limit"] = context
            var previousCacheLimit = 0
            try MX.check(mlx_set_cache_limit(&previousCacheLimit, 256 * 1024 * 1024), "scheduler probe allocation cache")
            defer { var ignored = 0; _ = mlx_set_cache_limit(&ignored, previousCacheLimit) }
            let model = try QwenModel(modelDirectory: directory) { current, total in
                if current % 8 == 0 || current == total {
                    FileHandle.standardError.write(Data("Local scheduler probe: loaded layer \(current)/\(total)\n".utf8))
                }
            }
            let generator = try QwenGenerator(model: model)
            report["loaded_memory"] = try MX.memory()
            let referenceA = try generator.generate(ar)
            let referenceB = tokensA == tokensB ? referenceA : try generator.generate(arB)
            report["reference_a"] = try object(referenceA); report["reference_b"] = try object(referenceB)
            report["reference_text_a"] = try tokenizer.decode(referenceA.tokens, skipSpecialTokens: true)
            report["reference_text_b"] = try tokenizer.decode(referenceB.tokens, skipSpecialTokens: true)
            var goldenExact = true
            if let path = args["--golden-report"] {
                let data = try Data(contentsOf: URL(fileURLWithPath: path))
                let source = try JSONSerialization.jsonObject(with: data) as? [String: Any]
                let trials = source?["trials"] as? [[String: Any]]
                let golden = (trials?.first?["generated_token_ids"] as? [NSNumber])?.map { $0.int32Value }
                let prompt = (trials?.first?["prompt_tokens"] as? [NSNumber])?.map { $0.int32Value }
                goldenExact = golden == referenceA.tokens && prompt == tokensA && source?["max_tokens"] as? Int == maximum
                report["golden_report"] = path; report["golden_exact"] = goldenExact
                report["golden_sha256"] = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            }
            record("independent_references", goldenExact && !referenceA.tokens.isEmpty && !referenceB.tokens.isEmpty)
            try write()

            let scheduler = try QwenLocalScheduler(generator: generator, limits: .init(maxQueuedPrefills: 2,
                maxReadyDecodes: 2, maxResidentTokens: reservation, maxConsecutivePrefills: 2))
            var callbackA = [Int32](), callbackB = [Int32]()
            let a = try scheduler.submit(ar) { callbackA.append($0) }
            let b = try scheduler.submit(mtp) { callbackB.append($0) }
            let first = try step(scheduler, "mixed"), second = try step(scheduler, "mixed")
            let peak = scheduler.snapshot()
            report["two_ready_snapshot"] = try object(peak)
            report["two_ready_memory"] = try MX.memory()
            record("two_ready_states", first?.kind == .prefillReady && second?.kind == .prefillReady &&
                   first?.jobID == a && second?.jobID == b && peak.readyDecodes == 2 && peak.reservedTokens == reservation)
            let completed = try drain(scheduler, "mixed")
            record("stage_fifo_order", completed.count == 2 && completed[0].jobID == a && completed[1].jobID == b &&
                   completed.allSatisfy { $0.kind == .completed })
            let outputA = completed.first { $0.jobID == a }?.result
            let outputB = completed.first { $0.jobID == b }?.result
            record("ar_exact", outputA?.tokens == referenceA.tokens && outputA?.finishReason == referenceA.finishReason && callbackA == referenceA.tokens)
            record("mtp_exact_and_exercised", outputB?.tokens == referenceB.tokens && outputB?.finishReason == referenceB.finishReason &&
                   callbackB == referenceB.tokens && (outputB?.statistics.mtp?.draftedTokens ?? 0) > 0)
            record("all_reservations_released", clean(scheduler))
            report["after_mixed_memory"] = try MX.memory()
            record("phase_accounting", completed.count == 2 && completed.allSatisfy { event in
                guard let result = event.result, let phases = result.phases,
                      let wait = event.timing.readyQueueWaitSeconds else { return false }
                return result.decodeSeconds >= 0 && phases.prefill.targetSeconds >= 0 &&
                    phases.prefill.totalSeconds + 1e-5 >= phases.prefill.targetSeconds + phases.prefill.draftHistorySeconds &&
                    phases.decodeServiceSeconds + 1e-5 >= result.decodeSeconds + result.statistics.callbackSeconds &&
                    wait >= 0 && event.timing.totalQueueWaitSeconds >= wait &&
                    result.timeToFirstTokenSeconds >= phases.prefill.totalSeconds + phases.handoffWaitSeconds
            })
            try write()

            if suite == "all" {
                let small = try QwenLocalScheduler(generator: generator, limits: .init(maxQueuedPrefills: 1,
                    maxReadyDecodes: 1, maxResidentTokens: 2 * (tokensA.count + 1), maxConsecutivePrefills: 8))
                let queued = try small.submit(one)
                var queueRejected = false
                do { _ = try small.submit(one) } catch QwenLocalScheduler.Error.queueFull { queueRejected = true }
                record("queue_backpressure", queueRejected && small.snapshot().queuedPrefillIDs == [queued])
                let cancelled = try small.cancel(queued)
                if let cancelled { try save(cancelled, small, "queued_cancel") }
                record("queued_cancellation", cancelled?.kind == .cancelled && clean(small))
                let ready = try small.submit(one)
                _ = try step(small, "ready_cancel")
                let readyCancelled = try small.cancel(ready)
                if let readyCancelled { try save(readyCancelled, small, "ready_cancel") }
                record("ready_cancellation", readyCancelled?.kind == .cancelled && clean(small))
                let replacement = try small.submit(one)
                let reusable = try drain(small, "reusable")
                record("capacity_reusable", reusable.last?.jobID == replacement && reusable.last?.result?.tokens == Array(referenceA.tokens.prefix(1)) && clean(small))

                let budget = try QwenLocalScheduler(generator: generator, limits: .init(maxQueuedPrefills: 2,
                    maxReadyDecodes: 1, maxResidentTokens: tokensA.count + 1))
                let kept = try budget.submit(one)
                var budgetRejected = false, invalidRejected = false
                do { _ = try budget.submit(one) } catch QwenLocalScheduler.Error.overBudget { budgetRejected = true }
                record("token_backpressure", budgetRejected && budget.snapshot().queuedPrefillIDs == [kept] && budget.snapshot().reservedTokens == tokensA.count + 1)
                do { _ = try budget.submit(QwenGenerationRequest(tokens: [-1])) }
                catch QwenGenerationError.invalidRequest { invalidRejected = true }
                record("invalid_request_rejected", invalidRejected && budget.snapshot().queuedPrefillIDs == [kept])
                _ = try budget.discardAll()

                let bound = try QwenLocalScheduler(generator: generator, limits: .init(maxQueuedPrefills: 2,
                    maxReadyDecodes: 1, maxResidentTokens: reservation, maxConsecutivePrefills: 8))
                let boundA = try bound.submit(one), boundB = try bound.submit(one)
                let boundEvents = try drain(bound, "ready_bound")
                record("ready_queue_bound", boundEvents.map(\.kind) == [.prefillReady, .completed, .prefillReady, .completed] &&
                    boundEvents.map(\.jobID) == [boundA, boundA, boundB, boundB] && clean(bound))

                let failure = try QwenLocalScheduler(generator: generator, limits: .init(maxQueuedPrefills: 3,
                    maxReadyDecodes: 2, maxResidentTokens: reservation * 2, maxConsecutivePrefills: 2))
                let cancellation = QwenCancellation()
                var cancelledIDs = [Int32](), busyIDs = [Int32]()
                let cancelID = try failure.submit(ar, cancellation: cancellation) { token in
                    cancelledIDs.append(token)
                    if cancelledIDs.count == 2 { cancellation.cancel() }
                }
                let busyID = try failure.submit(one) { token in
                    busyIDs.append(token); throw QwenGenerationError.busy
                }
                _ = try step(failure, "callback_failures"); _ = try step(failure, "callback_failures")
                let failures = try drain(failure, "callback_failures")
                record("active_cancellation", failures.filter { $0.jobID == cancelID }.count == 1 &&
                    failures.first { $0.jobID == cancelID }?.kind == .cancelled && cancelledIDs == Array(referenceA.tokens.prefix(2)))
                record("callback_busy_is_terminal", failures.filter { $0.jobID == busyID }.count == 1 &&
                    failures.first { $0.jobID == busyID }?.kind == .failed && busyIDs == Array(referenceA.tokens.prefix(1)) && clean(failure))
                var reentryRejected = false, nestedID: UUID?
                let fresh = try failure.submit(one) { _ in
                    do { _ = try failure.runNext() } catch QwenLocalScheduler.Error.busy { reentryRejected = true }
                    nestedID = try failure.submit(one)
                }
                let freshEvents = try drain(failure, "fresh")
                record("callback_reentry_rejected", reentryRejected)
                record("submit_during_callback", nestedID != nil && freshEvents.contains { $0.jobID == nestedID && $0.kind == .completed })
                record("fresh_after_failures", freshEvents.first { $0.jobID == fresh && $0.kind == .completed }?.result?.tokens == Array(referenceA.tokens.prefix(1)) && clean(failure))
                _ = try failure.submit(one); _ = try step(failure, "discard_all"); _ = try failure.submit(one)
                let discarded = try failure.discardAll()
                for event in discarded { try save(event, failure, "discard_all") }
                record("discard_all", discarded.count == 2 && discarded.allSatisfy { $0.kind == .cancelled } && clean(failure))
            }
            try write(true)
            guard required.allSatisfy({ checks[$0] == true }) else { throw CLIError.usage("Local scheduler probe checks failed; see \(output)") }
        } catch {
            report["fatal_error"] = String(describing: error)
            try write()
            throw error
        }
    }
}
