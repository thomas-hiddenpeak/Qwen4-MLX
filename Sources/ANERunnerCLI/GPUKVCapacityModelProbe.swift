import ANERunnerCore
import ANERunnerGPU
import CMLX
import Foundation

extension RunnerCLI {
    /// Correctness only. Native BF16 readback inside the observer perturbs both
    /// allocation lifetime and round timing. Use the separate generator-based
    /// benchmark with nil observers for performance evidence.
    static func probeGPUKVCapacityModel(_ args: Arguments) throws {
        try args.validate(["--model-dir", "--tokens-file", "--output"])
        let output = try args.require("--output")
        guard !FileManager.default.fileExists(atPath: output) else {
            throw CLIError.usage("KV capacity model probe requires a new --output")
        }
        var report: [String: Any] = [
            "schema": "qwen38-kv-capacity-model-v1", "complete": false, "passed": false,
            "scope": "Single-model ordinary AR correctness and ownership. Diagnostic readback is not a performance trial.",
            "notes": [
                "Observer records contain native BF16 tensor hashes and integer/host state; no Tensor or State is retained after a callback.",
                "CacheReliabilityAnchor.matches compares logical state; retained storage extents are reported separately and may differ.",
                "Maximum output budgets are 16 and 128; actual complete output includes EOS and may be shorter.",
                "Warm restore uses the actual RAM compact-copy path. SSD restart/import and forced alias COW are separate validations.",
                "Workspace exhaustion is a ledger-only reservation; it neither allocates corresponding RAM nor simulates OS pressure.",
                "Capacity step counters prove the requested runtime branch, not physical allocation identity, donation, bandwidth or ANE residency.",
            ],
        ]
        var checks = [String: Bool](), trials = [[String: Any]](), states = [[String: Any]]()
        var budgets = [[String: Any]]()
        func object<T: Encodable>(_ value: T) throws -> Any {
            try JSONSerialization.jsonObject(with: JSONEncoder().encode(value))
        }
        func write(complete: Bool = false) throws {
            report["complete"] = complete
            report["checks"] = checks; report["trials"] = trials
            report["state_checks"] = states; report["budgets"] = budgets
            report["state_event_count"] = states.count
            report["logical_tensor_observations"] = states.reduce(0) { $0 + ($1["tensor_count"] as? Int ?? 0) }
            report["passed"] = complete && !checks.isEmpty && !trials.isEmpty && !states.isEmpty &&
                checks.values.allSatisfy { $0 } && states.allSatisfy { $0["passed"] as? Bool == true }
            try emit(report, to: output)
        }
        func require(_ name: String, _ condition: Bool) throws {
            checks[name] = condition
            guard condition else { throw CLIError.usage("KV capacity model probe failed: \(name)") }
        }
        try write()
        do {
            let directory = URL(fileURLWithPath: try args.require("--model-dir")).standardizedFileURL.resolvingSymlinksInPath()
            let tokenURL = URL(fileURLWithPath: try args.require("--tokens-file")).standardizedFileURL
            let input = try Data(contentsOf: tokenURL)
            let prompt = try JSONDecoder().decode([Int32].self, from: input)
            guard (10_000...12_288).contains(prompt.count), prompt.count + 128 <= 16_384 else {
                throw CLIError.usage("Use a real 10k+ agent token fixture within the 16k context")
            }
            let boundary = ((prompt.count - 1) / 416) * 416
            let executable = Bundle.main.executableURL ?? URL(fileURLWithPath: CommandLine.arguments[0])
            var metadata = [String: String]()
            for name in ["config.json", "model.safetensors.index.json", "tokenizer.json", "chat_template.jinja"] {
                metadata[name] = try MoETilingBytes.hash(directory.appendingPathComponent(name))
            }
            report["provenance"] = [
                "command": CommandLine.arguments,
                "captured_utc": Date().ISO8601Format(),
                "os": ProcessInfo.processInfo.operatingSystemVersionString,
                "pid": ProcessInfo.processInfo.processIdentifier,
                "model_directory": directory.path,
                "model_metadata_sha256": metadata,
                "executable_sha256": try MoETilingBytes.hash(executable),
                "tokens_path": tokenURL.path, "tokens_sha256": MoETilingBytes.digest(input),
                "hash_scope": "Executable, fixture and small model metadata; controller binds complete checkpoint payloads and loaded native libraries.",
            ]
            report["prompt_token_ids"] = prompt
            report["request_policy"] = ["prompt_tokens": prompt.count, "prefill_chunk": 416,
                "prefill_evaluate_every_layers": 4, "mtp_depth": 0, "prefix_boundary": boundary,
                "joint_state_budget_bytes": 4 * 1024 * 1024 * 1024,
                "candidate_ram_cache_bytes": 512 * 1024 * 1024]
            var previousCache = 0
            try MX.check(mlx_set_cache_limit(&previousCache, 256 * 1024 * 1024), "capacity probe allocator cache")
            defer { var ignored = 0; _ = mlx_set_cache_limit(&ignored, previousCache) }
            let tokenizer = try QwenTokenizer(modelDirectory: directory)
            let model = try QwenModel(modelDirectory: directory,
                reservedOutputIDs: tokenizer.reservedOutputTokenIDs, stateBudgetBytes: 4 * 1024 * 1024 * 1024) { current, total in
                if current % 8 == 0 || current == total {
                    FileHandle.standardError.write(Data("KV capacity probe loaded \(current)/\(total)\n".utf8))
                }
            }
            defer { try? MX.synchronize() }
            let reference = try QwenGenerator(model: model)
            let candidate = try QwenGenerator(model: model,
                prefixCacheLimits: .init(maxEntries: 2, maxBytes: 512 * 1024 * 1024))
            defer {
                reference.prefixStateObserver = nil; reference.decodeStateObserver = nil
                candidate.prefixStateObserver = nil; candidate.decodeStateObserver = nil
                try? candidate.clearPrefixCache()
            }
            func request(_ maximum: Int, _ mode: GPUAttention.KVAppendMode) -> QwenGenerationRequest {
                QwenGenerationRequest(tokens: prompt, maxTokens: maximum, contextLimit: 16_384,
                    prefillChunk: 416, mtpDepth: 0, prefillEvaluateEveryLayers: 4,
                    prefixCacheMaxTokens: boundary, kvAppendMode: mode)
            }
            func budget(_ label: String, empty: Bool) throws {
                let value = model.stateBudget.statistics
                budgets.append(["label": label, "value": try object(value)])
                try require(label + "_request_workspace_released", value.requestBytes == 0 && value.workspaceBytes == 0)
                try require(label + "_within_limit", value.totalBytes <= value.maxBytes)
                if empty { try require(label + "_all_leases_released", value.totalBytes == 0 && value.currentLeases == 0) }
            }
            func clean(_ label: String) throws {
                reference.prefixStateObserver = nil; reference.decodeStateObserver = nil
                candidate.prefixStateObserver = nil; candidate.decodeStateObserver = nil
                try candidate.clearPrefixCache()
                try budget(label, empty: true)
            }
            var referencePrefix: CacheReliabilityAnchor?
            var referenceSteps = [CacheReliabilityAnchor]()
            var observedSteps = [String: Int]()
            func saveState(_ label: String, event: String, ordinal: Int,
                           state: QwenModel.State, expected: CacheReliabilityAnchor?) throws -> CacheReliabilityAnchor {
                let actual = try CacheReliabilityAnchor(state)
                let countOK = actual.tensors.count == 121
                let same = expected.map { actual.matches($0) } ?? true
                let passed = actual.valid && countOK && same
                let mismatches = expected.map { golden in
                    Set(actual.tensors.keys).union(golden.tensors.keys).filter {
                        actual.tensors[$0] != golden.tensors[$0]
                    }.sorted()
                } ?? []
                states.append(["label": label, "event": event, "ordinal": ordinal,
                    "offset": actual.host.offset, "tensor_count": actual.tensors.count,
                    "compared_to_reference": expected != nil, "passed": passed,
                    "mismatched_tensor_names": mismatches,
                    "anchor": try object(actual)])
                guard passed else { throw CLIError.usage("KV state mismatch: \(label) / \(event) / \(ordinal)") }
                return actual
            }
            func observe(_ generator: QwenGenerator, label: String, establishReference: Bool = false) {
                observedSteps[label] = 0
                generator.prefixStateObserver = { event, state in
                    guard state.offset == boundary, ["coldBoundary", "restore", "publish"].contains(event) else { return }
                    let expected = establishReference ? nil : referencePrefix
                    if !establishReference && expected == nil { throw CLIError.usage("Missing independent cold prefix anchor") }
                    let value = try saveState(label, event: event, ordinal: 0, state: state, expected: expected)
                    if establishReference { referencePrefix = value }
                }
                generator.decodeStateObserver = { event, state in
                    let ordinal = observedSteps[label, default: 0]
                    guard ordinal < 16, event == (ordinal == 0 ? "firstToken" : "decode"),
                          state.offset == prompt.count + ordinal else {
                        throw CLIError.usage("Unbounded or inconsistent decode observer sequence")
                    }
                    let expected: CacheReliabilityAnchor?
                    if establishReference { expected = nil }
                    else {
                        guard ordinal < referenceSteps.count else { throw CLIError.usage("Observer exceeded independent reference") }
                        expected = referenceSteps[ordinal]
                    }
                    let value = try saveState(label, event: event, ordinal: ordinal, state: state, expected: expected)
                    if establishReference { referenceSteps.append(value) }
                    observedSteps[label] = ordinal + 1
                }
            }
            func record(_ label: String, result: QwenGenerationResult, maximum: Int,
                        mode: GPUAttention.KVAppendMode, oracle: QwenGenerationResult?,
                        source: String = "cold", expectFallback: Bool = false, observed: Bool = false) throws {
                guard let phases = result.phases else { throw CLIError.usage("Missing generation phases") }
                let decoded = result.statistics.decodedTokenCount
                trials.append(["label": label, "maximum_output_tokens": maximum,
                    "actual_output_tokens": result.tokens.count, "actual_decoded_tokens": decoded,
                    "generated_token_ids": result.tokens, "finish_reason": result.finishReason.rawValue,
                    "observer_enabled": observed, "result": try object(result)])
                try require(label + "_nonempty_bounded_output", !result.tokens.isEmpty && result.tokens.count <= maximum)
                try require(label + "_ar_counts", result.statistics.mtpDepth == 0 && decoded == result.tokens.count - 1 &&
                    result.statistics.decodeRounds == decoded && result.statistics.finalStateOffset == prompt.count + decoded)
                let endedByEOS = result.tokens.last.map { reference.eosTokenIDs.contains($0) } ?? false
                try require(label + "_finish_contract", result.finishReason == .eos ? endedByEOS :
                    (result.finishReason == .length && result.tokens.count == maximum && !endedByEOS))
                try require(label + "_phase_policy", phases.kvAppendMode == mode.rawValue && phases.prefill.evaluateEveryLayers == 4)
                try require(label + "_prefill_source", phases.prefill.cacheSource == source &&
                    phases.prefill.cachedTokenCount == (source == "memory" ? boundary : 0))
                if let oracle {
                    try require(label + "_complete_ids_finish_exact", result.tokens == oracle.tokens && result.finishReason == oracle.finishReason)
                }
                if observed { try require(label + "_one_state_per_output", observedSteps[label] == result.tokens.count) }
                if mode == .reference {
                    try require(label + "_reference_counts", phases.kvCapacityTokenSteps == 0 &&
                        phases.kvCapacityWorkspaceFallbacks == 0 && phases.kvCapacityWorkspacePeakBytes == 0)
                } else if expectFallback {
                    try require(label + "_fallback_exercised", decoded > 0 && phases.kvCapacityTokenSteps == 0 &&
                        phases.kvCapacityWorkspaceFallbacks == decoded && phases.kvCapacityWorkspacePeakBytes == 0)
                } else {
                    try require(label + "_capacity_exercised_without_fallback", decoded > 0 &&
                        phases.kvCapacityTokenSteps == decoded && phases.kvCapacityWorkspaceFallbacks == 0 &&
                        (phases.kvCapacityWorkspacePeakBytes ?? 0) > 0)
                }
                try budget(label, empty: mode == .reference)
                try write()
            }

            try clean("initial")
            observe(reference, label: "reference_16", establishReference: true)
            let oracle16 = try reference.generate(request(16, .reference))
            try record("reference_16", result: oracle16, maximum: 16, mode: .reference, oracle: nil, observed: true)
            try require("reference_has_decode_and_prefix", oracle16.tokens.count > 1 && referencePrefix != nil)
            reference.prefixStateObserver = nil; reference.decodeStateObserver = nil
            observe(candidate, label: "capacity_cold_16")
            let capacity16 = try candidate.generate(request(16, .capacity256))
            try record("capacity_cold_16", result: capacity16, maximum: 16, mode: .capacity256, oracle: oracle16, observed: true)
            observe(candidate, label: "capacity_warm_16")
            let warm16 = try candidate.generate(request(16, .capacity256))
            try record("capacity_warm_16", result: warm16, maximum: 16, mode: .capacity256,
                oracle: oracle16, source: "memory", observed: true)
            try clean("after_cold_warm_16")

            let oracle128 = try reference.generate(request(128, .reference))
            try record("reference_128", result: oracle128, maximum: 128, mode: .reference, oracle: nil)
            let capacity128 = try candidate.generate(request(128, .capacity256))
            try record("capacity_cold_128", result: capacity128, maximum: 128, mode: .capacity256, oracle: oracle128)
            try clean("after_cold_128")

            // The complete first token comes from prefill. Fill the ledger only
            // after request admission; no actual payload/allocation backs this
            // deliberate blocker. Defers release every owner on any failure.
            observe(candidate, label: "capacity_budget_fallback_16")
            func forcedFallback() throws -> QwenGenerationResult {
                let prepared = try candidate.prefill(request(16, .capacity256))
                defer { prepared.discard() }
                let session = try candidate.beginDecode(prepared)
                defer { try? session.discard() }
                let first = try candidate.stepDecode(session)
                try require("fallback_first_token_nonterminal", first == nil && session.generatedTokenCount == 1)
                try candidate.clearPrefixCache()
                let before = model.stateBudget.statistics
                try require("fallback_before_has_only_request", before.requestBytes > 0 && before.cacheBytes == 0 && before.workspaceBytes == 0)
                guard let blocker = model.stateBudget.reserve(bytes: before.maxBytes - before.totalBytes, kind: .workspace) else {
                    throw CLIError.usage("Unable to create the logical workspace blocker")
                }
                defer { blocker.release() }
                report["logical_workspace_blocker"] = ["bytes": blocker.bytes, "allocates_host_payload": false,
                    "before": try object(before), "held": try object(model.stateBudget.statistics)]
                for _ in 0..<16 {
                    if let result = try candidate.stepDecode(session) {
                        try require("fallback_reservation_rejections_observed", model.stateBudget.statistics.rejections > before.rejections)
                        return result
                    }
                }
                throw CLIError.usage("Fallback exceeded the bounded AR output steps")
            }
            let fallback = try forcedFallback()
            try record("capacity_budget_fallback_16", result: fallback, maximum: 16, mode: .capacity256,
                oracle: oracle16, expectFallback: true, observed: true)
            try clean("after_budget_fallback")

            observe(candidate, label: "capacity_callback_cancel")
            let cancellation = QwenCancellation()
            var cancelledIDs = [Int32](), sawCancellation = false
            var atCancel: QwenStateBudget.Statistics?
            do {
                _ = try candidate.generate(request(16, .capacity256), cancellation: cancellation) { token in
                    cancelledIDs.append(token)
                    if cancelledIDs.count == 2 {
                        atCancel = model.stateBudget.statistics
                        cancellation.cancel()
                    }
                }
            } catch QwenGenerationError.cancelled { sawCancellation = true }
            report["callback_cancellation"] = ["caught_cancelled": sawCancellation,
                "published_ids_before_cancel": cancelledIDs,
                "budget_during_callback": try atCancel.map { try object($0) } ?? NSNull()]
            try require("callback_cancel_after_capacity_step", sawCancellation && cancelledIDs.count == 2 &&
                cancelledIDs == Array(oracle16.tokens.prefix(2)) && observedSteps["capacity_callback_cancel"] == 2 &&
                (atCancel?.workspaceBytes ?? 0) > 0)
            try budget("after_callback_cancel", empty: false)
            try clean("final")
            try require("no_mtp_payload_loaded", !model.weights.ledger.contains { $0.name.contains(".mtp.") })
            report["observer_steps_by_case"] = observedSteps
            report["actual_completed_generations"] = trials.count
            report["cancelled_partial_generations"] = 1
            try write(complete: true)
        } catch {
            report["error"] = String(describing: error)
            report["passed"] = false
            try write()
            throw error
        }
    }
}
