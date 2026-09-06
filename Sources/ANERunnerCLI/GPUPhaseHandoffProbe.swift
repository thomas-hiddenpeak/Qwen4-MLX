import ANERunnerGPU
import CMLX
import CryptoKit
import Foundation

extension RunnerCLI {
    /// One model and two generators exercise the same-process producer/consumer
    /// contract. This is a real-state gate, not a throughput benchmark.
    static func probeGPUPhaseHandoff(_ args: Arguments) throws {
        try args.validate(["--model-dir", "--tokens-file", "--output"])
        let output = try args.require("--output")
        guard !FileManager.default.fileExists(atPath: output) else {
            throw CLIError.usage("Phase handoff probe --output must be a new file")
        }
        let checkOrder = [
            "ar_reference", "prefill_ready", "ready_releases_model_admission",
            "busy_nested_decode_preserves_ready", "cancel_before_consumption_preserves_ready",
            "cross_generator_split_exact", "consumed_handoff_rejected",
            "cancel_first_callback_burns_handoff", "cancelled_handoff_rejected",
            "fresh_after_decode_cancel", "throw_first_callback_burns_handoff",
            "errored_handoff_rejected", "fresh_after_decode_error",
            "discarded_handoff_rejected", "max_tokens_one",
            "mtp_prefill_ready", "mtp_split_exact_and_exercised"
        ]
        var checks: [String: [String: Any]] = Dictionary(uniqueKeysWithValues: checkOrder.map {
            ($0, ["id": $0, "status": "not_run", "passed": false, "required": true])
        })
        var results = [String: Any](), prefills = [String: Any]()
        var report: [String: Any] = [
            "schema": "qwen38-phase-handoff-probe-v1", "complete": false, "passed": false,
            "full_model_instances": 1, "generator_instances": 2,
            "max_output_tokens": 64,
            "notes": [
                "One short real prompt, one model, same executor; no cross-process transport, serialization or concurrent GPU execution is tested.",
                "Full token IDs and finish reasons are compared with a fresh AR generate reference; this probe does not compare every internal tensor.",
                "A ready handoff deliberately waits while another request runs. That wait is reported as handoffWaitSeconds, not GPU or decode compute time.",
                "Prefill, handoff and decode metrics are preserved separately. First-use/JIT/SSD cache conditions differ; these timings are not a speedup benchmark.",
                "Every required check starts as not_run. An exception or unreached callback never becomes a passing branch.",
                "The optional token file should be a short text prompt. If it immediately generates EOS, the required MTP draft branch fails rather than being assumed covered.",
                "MTP uses depth 2, batchedScalarLinear, initial draft-history limit 1024. Terminal target offsets can differ from AR after a verified draft EOS."
            ]
        ]
        func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
        func elapsed(_ start: UInt64) -> Double { Double(now() - start) * 1e-9 }
        func object<T: Encodable>(_ value: T) throws -> Any {
            try JSONSerialization.jsonObject(with: JSONEncoder().encode(value))
        }
        func describe(_ error: Error?) -> Any {
            if let error { return String(describing: error) }
            return NSNull()
        }
        func record(_ id: String, _ passed: Bool, _ details: [String: Any] = [:]) {
            var row = details
            row["id"] = id; row["required"] = true; row["passed"] = passed
            row["status"] = passed ? "passed" : "failed"
            checks[id] = row
            FileHandle.standardError.write(Data("Phase handoff probe: \(id): \(passed ? "passed" : "failed")\n".utf8))
        }
        func writeReport(complete: Bool = false) throws {
            report["checks"] = checkOrder.compactMap { checks[$0] }
            report["results"] = results; report["prefills"] = prefills
            report["complete"] = complete
            report["passed"] = complete && checkOrder.allSatisfy { checks[$0]?["passed"] as? Bool == true }
            try emit(report, to: output)
        }
        func saveResult(_ name: String, _ result: QwenGenerationResult,
                        wallSeconds: Double, callbackIDs: [Int32]) throws {
            results[name] = [
                "result": try object(result), "wall_seconds": wallSeconds,
                "callback_token_ids": callbackIDs,
                "decode_tokens_per_second": result.decodeTokensPerSecond.map { $0 as Any } ?? NSNull(),
                "decode_average_tpot_seconds": result.statistics.decodedTokenCount > 0
                    ? result.decodeSeconds / Double(result.statistics.decodedTokenCount) as Any : NSNull()
            ]
        }
        func savePrefill(_ name: String, _ prepared: QwenPrefillResult, wallSeconds: Double) throws {
            prefills[name] = [
                "statistics": try object(prepared.statistics), "wall_seconds": wallSeconds,
                "preparation_seconds": prepared.preparationSeconds,
                "first_token": prepared.firstToken, "is_ready_at_return": prepared.isReady,
                "target_tokens_per_second": prepared.statistics.targetTokensPerSecond.map { $0 as Any } ?? NSNull(),
                "ready_tokens_per_second": prepared.statistics.readyTokensPerSecond.map { $0 as Any } ?? NSNull()
            ]
        }

        var fatalError: Error?
        do {
            let directory = URL(fileURLWithPath: try args.require("--model-dir"), isDirectory: true)
                .standardizedFileURL.resolvingSymlinksInPath()
            report["model_directory"] = directory.path
            let tokenizer = try QwenTokenizer(modelDirectory: directory)
            let tokens: [Int32]
            if let path = args["--tokens-file"] {
                let url = URL(fileURLWithPath: path).standardizedFileURL
                let data = try Data(contentsOf: url)
                tokens = try JSONDecoder().decode([Int32].self, from: data)
                report["input"] = ["kind": "token_file", "path": url.path,
                    "sha256": SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()]
            } else {
                let prompt = "请用三句话介绍太阳为什么发光，以及阳光如何到达地球。"
                tokens = try tokenizer.encode(tokenizer.renderChat(messages: [ChatMessage(role: "user", content: prompt)]))
                report["input"] = ["kind": "tokenized_user_text", "text": prompt,
                                   "chat_template": true, "thinking": false]
            }
            let configuration = try QwenConfiguration(modelDirectory: directory)
            let (budget, overflow) = tokens.count.addingReportingOverflow(64)
            guard !overflow, budget <= configuration.maximumPositions else {
                throw CLIError.usage("Phase handoff probe prompt plus 64 output tokens exceeds model context")
            }
            let context = min(configuration.maximumPositions, max(4096, budget))
            let request = QwenGenerationRequest(tokens: tokens, maxTokens: 64, contextLimit: context)
            let oneRequest = QwenGenerationRequest(tokens: tokens, maxTokens: 1, contextLimit: context)
            let mtpRequest = QwenGenerationRequest(tokens: tokens, maxTokens: 64,
                contextLimit: context, mtpDepth: 2, verification: .batchedScalarLinear, draftHistoryTokens: 1024)
            try request.validate(configuration: configuration)
            try mtpRequest.validate(configuration: configuration)
            report["prompt_token_ids"] = tokens; report["context_limit"] = context
            report["mtp_configuration"] = ["depth": 2, "verification": "batchedScalarLinear", "draft_history_limit": 1024]
            var previousCacheLimit = 0
            try MX.check(mlx_set_cache_limit(&previousCacheLimit, 256 * 1024 * 1024), "phase probe allocation cache")
            defer { var ignored = 0; _ = mlx_set_cache_limit(&ignored, previousCacheLimit) }
            let modelStart = now()
            let model = try QwenModel(modelDirectory: directory) { current, total in
                if current % 8 == 0 || current == total {
                    FileHandle.standardError.write(Data("Phase handoff probe: loaded layer \(current)/\(total)\n".utf8))
                }
            }
            report["model_construction_seconds"] = elapsed(modelStart)
            let producer = try QwenGenerator(model: model)
            let consumer = try QwenGenerator(model: model)

            func valid(_ result: QwenGenerationResult, maximum: Int = 64) -> Bool {
                guard let last = result.tokens.last, let phases = result.phases else { return false }
                let count = result.tokens.count
                let timing = [result.preparationSeconds, result.timeToFirstTokenSeconds,
                    result.decodeSeconds, result.totalSeconds, phases.prefill.targetSeconds,
                    phases.prefill.draftHistorySeconds, phases.prefill.totalSeconds,
                    phases.handoffWaitSeconds, phases.handoffConsumeSeconds, phases.decodeServiceSeconds]
                return count <= maximum && result.statistics.generatedTokenCount == count &&
                    result.statistics.decodedTokenCount == count - 1 &&
                    result.statistics.promptTokenCount == tokens.count &&
                    phases.prefill.promptTokenCount == tokens.count &&
                    timing.allSatisfy { $0.isFinite && $0 >= 0 } &&
                    !result.tokens.dropLast().contains(where: producer.eosTokenIDs.contains) &&
                    (result.finishReason == .eos
                        ? producer.eosTokenIDs.contains(last)
                        : count == maximum && !producer.eosTokenIDs.contains(last))
            }
            let reference: QwenGenerationResult
            do {
                var ids = [Int32]()
                let start = now()
                reference = try producer.generate(request) { ids.append($0) }
                try saveResult("ar_reference", reference, wallSeconds: elapsed(start), callbackIDs: ids)
                record("ar_reference", valid(reference) && ids == reference.tokens)
            } catch {
                record("ar_reference", false, ["error": describe(error)])
                throw error
            }
            func exact(_ result: QwenGenerationResult) -> Bool {
                valid(result) && result.tokens == reference.tokens && result.finishReason == reference.finishReason
            }
            func invalidRequest(_ error: Error?) -> Bool {
                guard let error = error as? QwenGenerationError else { return false }
                if case .invalidRequest = error { return true }
                return false
            }
            func rejectBurned(_ name: String, _ prepared: QwenPrefillResult) {
                var callbacks = 0, returned = false
                var caught: Error?
                do { _ = try consumer.decode(prepared) { _ in callbacks += 1 }; returned = true }
                catch { caught = error }
                record(name, !returned && invalidRequest(caught) && callbacks == 0 && !prepared.isReady,
                       ["error": describe(caught), "callbacks": callbacks, "is_ready": prepared.isReady])
            }
            func freshRetry(_ name: String) {
                var ids = [Int32]()
                do {
                    let start = now()
                    let result = try producer.generate(request) { ids.append($0) }
                    try saveResult(name, result, wallSeconds: elapsed(start), callbackIDs: ids)
                    record(name, exact(result) && ids == result.tokens)
                } catch { record(name, false, ["error": describe(error), "callback_token_ids": ids]) }
            }
            try writeReport()

            // One live ready handle exercises admission release, busy, pre-cancel
            // and cross-generator transfer before it is successfully consumed.
            do {
                let start = now()
                let prepared = try producer.prefill(request)
                defer { prepared.discard() }
                try savePrefill("ar_ready", prepared, wallSeconds: elapsed(start))
                record("prefill_ready", prepared.isReady && prepared.firstToken == reference.tokens.first)
                var attempted = false, nestedReturned = false, readyAfterBusy = false
                var nestedError: Error?, outerIDs = [Int32]()
                do {
                    let outerStart = now()
                    let outer = try consumer.generate(oneRequest) { token in
                        outerIDs.append(token)
                        guard !attempted else { return }
                        attempted = true
                        do { _ = try producer.decode(prepared); nestedReturned = true }
                        catch { nestedError = error }
                        readyAfterBusy = prepared.isReady
                    }
                    try saveResult("request_while_handoff_ready", outer, wallSeconds: elapsed(outerStart), callbackIDs: outerIDs)
                    record("ready_releases_model_admission", valid(outer, maximum: 1) &&
                           outer.tokens == Array(reference.tokens.prefix(1)) && outerIDs == outer.tokens && prepared.isReady)
                } catch {
                    record("ready_releases_model_admission", false, ["error": describe(error), "callback_token_ids": outerIDs])
                }
                record("busy_nested_decode_preserves_ready", attempted && !nestedReturned &&
                       nestedError as? QwenGenerationError == .busy && readyAfterBusy,
                       ["attempted": attempted, "error": describe(nestedError), "ready_after_busy": readyAfterBusy])

                let cancellation = QwenCancellation(); cancellation.cancel()
                var cancelError: Error?, cancelCallbacks = 0, cancelReturned = false
                do { _ = try consumer.decode(prepared, cancellation: cancellation) { _ in cancelCallbacks += 1 }; cancelReturned = true }
                catch { cancelError = error }
                record("cancel_before_consumption_preserves_ready", !cancelReturned &&
                       cancelError as? QwenGenerationError == .cancelled && cancelCallbacks == 0 && prepared.isReady,
                       ["error": describe(cancelError), "callbacks": cancelCallbacks, "is_ready": prepared.isReady])
                var ids = [Int32]()
                do {
                    let decodeStart = now()
                    let result = try consumer.decode(prepared) { ids.append($0) }
                    try saveResult("cross_generator_split", result, wallSeconds: elapsed(decodeStart), callbackIDs: ids)
                    record("cross_generator_split_exact", exact(result) && ids == result.tokens && !prepared.isReady,
                           ["is_ready_after_decode": prepared.isReady])
                } catch { record("cross_generator_split_exact", false, ["error": describe(error), "callback_token_ids": ids]) }
                rejectBurned("consumed_handoff_rejected", prepared)
            } catch { record("prefill_ready", false, ["error": describe(error)]) }
            try writeReport()

            for throwFromCallback in [false, true] {
                let name = throwFromCallback ? "throw_first_callback_burns_handoff" : "cancel_first_callback_burns_handoff"
                do {
                    let start = now()
                    let prepared = try producer.prefill(request)
                    defer { prepared.discard() }
                    try savePrefill(name, prepared, wallSeconds: elapsed(start))
                    let cancellation = QwenCancellation()
                    var ids = [Int32](), caught: Error?, returned = false
                    let decodeStart = now()
                    do {
                        let result = try consumer.decode(prepared, cancellation: cancellation) { token in
                            ids.append(token)
                            if throwFromCallback { throw GPUPhaseHandoffCallbackError.deliberate }
                            cancellation.cancel()
                        }
                        returned = true
                        try saveResult(name + "_unexpected_result", result, wallSeconds: elapsed(decodeStart), callbackIDs: ids)
                    } catch { caught = error }
                    let expectedError = throwFromCallback
                        ? caught as? GPUPhaseHandoffCallbackError == .deliberate
                        : caught as? QwenGenerationError == .cancelled
                    record(name, !returned && expectedError && ids == Array(reference.tokens.prefix(1)) && !prepared.isReady,
                           ["error": describe(caught), "callback_token_ids": ids, "decode_wall_seconds": elapsed(decodeStart),
                            "is_ready_after_error": prepared.isReady])
                    rejectBurned(throwFromCallback ? "errored_handoff_rejected" : "cancelled_handoff_rejected", prepared)
                } catch { record(name, false, ["error": describe(error)]) }
                freshRetry(throwFromCallback ? "fresh_after_decode_error" : "fresh_after_decode_cancel")
                try writeReport()
            }

            do {
                let start = now()
                let prepared = try producer.prefill(oneRequest)
                try savePrefill("discard", prepared, wallSeconds: elapsed(start))
                prepared.discard()
                rejectBurned("discarded_handoff_rejected", prepared)
            } catch { record("discarded_handoff_rejected", false, ["error": describe(error)]) }
            do {
                let start = now()
                let prepared = try producer.prefill(oneRequest)
                defer { prepared.discard() }
                try savePrefill("max_tokens_one", prepared, wallSeconds: elapsed(start))
                var ids = [Int32]()
                let decodeStart = now()
                let result = try consumer.decode(prepared) { ids.append($0) }
                try saveResult("max_tokens_one", result, wallSeconds: elapsed(decodeStart), callbackIDs: ids)
                record("max_tokens_one", valid(result, maximum: 1) && result.tokens == [prepared.firstToken] &&
                       result.tokens == Array(reference.tokens.prefix(1)) && ids == result.tokens &&
                       result.statistics.decodeRounds == 0 && result.statistics.decodedTokenCount == 0 &&
                       result.decodeSeconds == 0 && !prepared.isReady)
            } catch { record("max_tokens_one", false, ["error": describe(error)]) }
            try writeReport()

            do {
                let start = now()
                let prepared = try producer.prefill(mtpRequest)
                defer { prepared.discard() }
                try savePrefill("mtp_ready", prepared, wallSeconds: elapsed(start))
                record("mtp_prefill_ready", prepared.isReady && prepared.firstToken == reference.tokens.first)
                var ids = [Int32]()
                let decodeStart = now()
                let result = try consumer.decode(prepared) { ids.append($0) }
                try saveResult("mtp_split", result, wallSeconds: elapsed(decodeStart), callbackIDs: ids)
                let drafted = result.statistics.mtp?.draftedTokens ?? 0
                record("mtp_split_exact_and_exercised", exact(result) && ids == result.tokens && drafted > 0 &&
                       result.statistics.mtpDepth == 2 && result.statistics.mtpVerification == "batchedScalarLinear" && !prepared.isReady,
                       ["drafted_tokens": drafted, "accepted_draft_tokens": result.statistics.mtp?.acceptedDraftTokens ?? 0,
                        "tokens_exact": result.tokens == reference.tokens, "finish_exact": result.finishReason == reference.finishReason])
            } catch { record("mtp_split_exact_and_exercised", false, ["error": describe(error)]) }
            report["loaded_source_weight_bytes_at_report"] = model.weights.cachedSourceBytes
            report["final_mlx_memory"] = try MX.memory()
        } catch {
            fatalError = error
            report["fatal_error"] = describe(error)
        }
        try writeReport(complete: fatalError == nil)
        if let fatalError { throw fatalError }
        guard report["passed"] as? Bool == true else {
            throw CLIError.usage("Phase handoff probe has failed or unexercised required checks; see \(output)")
        }
    }
}

private enum GPUPhaseHandoffCallbackError: Error, Equatable { case deliberate }
