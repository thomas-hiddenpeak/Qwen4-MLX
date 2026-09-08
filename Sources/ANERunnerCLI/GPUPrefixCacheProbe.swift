import ANERunnerGPU
import CMLX
import Foundation

/// Host readback is only for the explicit cache correctness probe. It is never
/// enabled in a throughput trial or the HTTP serving path.
private struct PrefixProbeSavedState {
    struct Value {
        let shape: [Int]
        let dtype: Int
        let bytes: Data
        let finite: Bool
    }
    let tensors: [String: Value]
    let host: QwenModel.State.DiagnosticHostValues

    init(_ state: QwenModel.State) throws {
        host = state.diagnosticHostValues
        var values = [String: Value]()
        for (name, tensor) in state.namedTensors {
            let floating = [MLX_BFLOAT16, MLX_FLOAT16, MLX_FLOAT32, MLX_FLOAT64].contains(tensor.dtype)
            let finite = floating ? try tensor.floats().allSatisfy(\.isFinite) : true
            values[name] = Value(shape: tensor.shape, dtype: Int(tensor.dtype.rawValue),
                bytes: try MoETilingBytes.bytes(tensor), finite: finite)
        }
        tensors = values
    }

    func compare(_ state: QwenModel.State, label: String) throws -> [String: Any] {
        let other = try Self(state)
        var rows = [[String: Any]]()
        var exact = !tensors.isEmpty
        for name in Set(tensors.keys).union(other.tensors.keys).sorted() {
            guard let a = tensors[name], let b = other.tensors[name] else {
                rows.append(["name": name, "exact": false, "nil_mismatch": true])
                exact = false
                continue
            }
            let equal = a.shape == b.shape && a.dtype == b.dtype && a.bytes == b.bytes && a.finite && b.finite
            exact = exact && equal
            rows.append(["name": name, "exact": equal, "shape_a": a.shape, "shape_b": b.shape,
                "dtype_a": a.dtype, "dtype_b": b.dtype, "byte_count_a": a.bytes.count,
                "byte_count_b": b.bytes.count, "sha256_a": MoETilingBytes.digest(a.bytes),
                "sha256_b": MoETilingBytes.digest(b.bytes), "all_finite": a.finite && b.finite])
        }
        // A compact copy can normalize backing-allocation extents. All logical
        // values, including UInt32 n-gram histories, must remain identical.
        let b = other.host
        let hostExact = host.offset == b.offset && host.valid == b.valid &&
            host.gdnOffsets == b.gdnOffsets && host.attentionOffsets == b.attentionOffsets &&
            host.pleHistory == b.pleHistory && host.gdnCapturePresent == b.gdnCapturePresent &&
            host.pleCapturePresent == b.pleCapturePresent
        let noCaptures = !b.gdnCapturePresent.contains(true) && !b.pleCapturePresent.contains(true)
        return ["label": label, "offset": host.offset, "tensor_count": tensors.count,
            "all_tensor_bytes_exact": exact, "logical_host_exact": hostExact,
            "retained_storage_exact": host.attentionRetainedStorage == b.attentionRetainedStorage,
            "no_verification_capture": noCaptures,
            "host_a": try Self.object(host), "host_b": try Self.object(b),
            "tensors": rows, "passed": exact && hostExact && noCaptures]
    }

    static func object<T: Encodable>(_ value: T) throws -> Any {
        try JSONSerialization.jsonObject(with: JSONEncoder().encode(value))
    }
}

extension RunnerCLI {
    /// Real full-model cold/hit comparisons. The readback switch separates
    /// complete state diagnostics from ordinary generation timing.
    static func probeGPUPrefixCache(_ args: Arguments) throws {
        try args.validate(["--model-dir", "--tokens-file", "--output", "--max-tokens", "--suite", "--state-readback"])
        let output = try args.require("--output")
        let suite = args["--suite"] ?? "all"
        let readbackText = args["--state-readback"] ?? "true"
        guard !FileManager.default.fileExists(atPath: output),
              ["long", "boundaries", "lifecycle", "all"].contains(suite),
              ["true", "false"].contains(readbackText),
              let maximum = Int(args["--max-tokens"] ?? "128"), (8...128).contains(maximum) else {
            throw CLIError.usage("Prefix probe requires new --output, --suite long|boundaries|lifecycle|all, --max-tokens 8...128 and --state-readback true|false")
        }
        let readback = readbackText == "true"
        var report: [String: Any] = ["schema": "qwen38-prefix-cache-probe-v1", "complete": false,
            "passed": false, "suite": suite, "state_readback": readback, "max_tokens": maximum,
            "full_model_instances": 1,
            "scope": readback ? "Numerical and lifecycle diagnostics; host tensor readback perturbs timings."
                : "Uninstrumented generation timings and complete output IDs; no tensor-state readback.",
            "notes": ["Prompts are actual model-token fixtures. B changes one suffix token; it is a controlled token fork, not a separately rendered chat.",
                "The first cold result for each prompt is its oracle; complete output IDs and finish reason are compared, never only rendered text.",
                "State comparison reads native tensor bytes and all logical host state including PLE UInt32 history. Compact copies may normalize retained allocation extents.",
                "The publish observer means an evaluated snapshot is ready for insertion; only the cache published counter proves successful insertion.",
                "Reported logical cache payload bytes are not physical memory usage; MLX active/peak/cache are recorded separately.",
                "MTP requests bypass the AR-only prefix cache while retaining the explicitly requested MTP mode. This is not an MTP performance gate."]]
        var trials = [[String: Any]](), stateChecks = [[String: Any]](), checks = [String: Bool]()
        func object<T: Encodable>(_ value: T) throws -> Any { try PrefixProbeSavedState.object(value) }
        func write(complete: Bool = false) throws {
            report["complete"] = complete
            report["trials"] = trials; report["state_checks"] = stateChecks; report["checks"] = checks
            report["passed"] = complete && !trials.isEmpty && !checks.isEmpty && checks.values.allSatisfy { $0 } &&
                trials.allSatisfy { $0["passed"] as? Bool == true } && stateChecks.allSatisfy { $0["passed"] as? Bool == true }
            try emit(report, to: output)
        }
        try write()
        do {
            let directory = URL(fileURLWithPath: try args.require("--model-dir")).standardizedFileURL.resolvingSymlinksInPath()
            let inputURL = URL(fileURLWithPath: try args.require("--tokens-file")).standardizedFileURL
            let inputData = try Data(contentsOf: inputURL)
            let prompt = try JSONDecoder().decode([Int32].self, from: inputData)
            guard (10_000...12_000).contains(prompt.count) else {
                throw CLIError.usage("Prefix probe requires the real 10k...12k token fixture")
            }
            let configuration = try QwenConfiguration(modelDirectory: directory)
            let executable = Bundle.main.executableURL ?? URL(fileURLWithPath: CommandLine.arguments[0])
            report["provenance"] = ["command": CommandLine.arguments, "executable": executable.path,
                "executable_sha256": try MoETilingBytes.hash(executable), "model_directory": directory.path,
                "config_sha256": try MoETilingBytes.hash(directory.appendingPathComponent("config.json")),
                "tokenizer_sha256": try MoETilingBytes.hash(directory.appendingPathComponent("tokenizer.json")),
                "input_path": inputURL.path, "input_sha256": MoETilingBytes.digest(inputData)]
            report["source_prompt_token_ids"] = prompt
            var previousCache = 0
            try MX.check(mlx_set_cache_limit(&previousCache, 256 * 1024 * 1024), "prefix probe allocator cache limit")
            defer { var ignored = 0; _ = mlx_set_cache_limit(&ignored, previousCache) }
            let model = try QwenModel(modelDirectory: directory) { current, total in
                if current % 8 == 0 || current == total {
                    FileHandle.standardError.write(Data("Prefix cache probe: loaded \(current)/\(total) layers\n".utf8))
                }
            }
            defer { try? MX.synchronize() }
            report["memory_after_load"] = try MX.memory()
            let cold = try QwenGenerator(model: model)
            let cached = try QwenGenerator(model: model,
                prefixCacheLimits: .init(maxEntries: 8, maxBytes: 1024 * 1024 * 1024))
            var anchors = [String: [Int: PrefixProbeSavedState]]()
            var oracles = [String: QwenGenerationResult]()
            func request(_ tokens: [Int32], prefix: Int, maxTokens: Int, mtp: Int = 0,
                         chunk: Int = 416) -> QwenGenerationRequest {
                QwenGenerationRequest(tokens: tokens, maxTokens: maxTokens, contextLimit: 16_384,
                    prefillChunk: chunk, mtpDepth: mtp,
                    verification: .batchedScalarLinear, draftHistoryTokens: mtp > 0 ? 1024 : nil,
                    prefixCacheMaxTokens: prefix)
            }
            func run(_ label: String, promptKey: String, stateKey: String, tokens: [Int32], prefix: Int,
                     generator: QwenGenerator, expectedCached: Int, maxTokens: Int, mtp: Int = 0,
                     establishOracle: Bool = false) throws {
                FileHandle.standardError.write(Data("Prefix cache probe: \(label)\n".utf8))
                var events = [String](), callbacks = [Int32]()
                if readback {
                    generator.prefixStateObserver = { event, state in
                        events.append(event)
                        guard state.offset > 0, state.offset <= prefix, state.offset % 416 == 0,
                              event != "restore" || state.offset == expectedCached else {
                            throw CLIError.usage("Prefix diagnostic observed unexpected offset")
                        }
                        if anchors[stateKey]?[state.offset] == nil {
                            guard event == "coldBoundary" else { throw CLIError.usage("No independent cold state anchor for \(label)") }
                            anchors[stateKey, default: [:]][state.offset] = try PrefixProbeSavedState(state)
                        }
                        let comparison = try anchors[stateKey]![state.offset]!.compare(state, label: label + ":" + event)
                        stateChecks.append(comparison)
                        guard comparison["passed"] as? Bool == true else { throw CLIError.usage("Prefix tensor/host mismatch at \(label):\(event)") }
                    }
                }
                defer { generator.prefixStateObserver = nil }
                let req = request(tokens, prefix: prefix, maxTokens: maxTokens, mtp: mtp)
                try req.validate(configuration: configuration)
                let memoryBefore = try MX.memory()
                let result = try generator.generate(req, onToken: { callbacks.append($0) })
                if establishOracle { oracles[promptKey] = result }
                guard let expected = oracles[promptKey], let prefill = result.phases?.prefill else {
                    throw CLIError.usage("Missing independent cold oracle or phase statistics")
                }
                let cachedCount = prefill.cachedTokenCount ?? 0
                let computedCount = prefill.computedTokenCount ?? tokens.count
                let outputExact = result.tokens == expected.tokens && result.finishReason == expected.finishReason
                let stats = result.statistics
                let offsetExact = mtp == 0 ? stats.finalStateOffset == tokens.count + result.tokens.count - 1
                    : stats.finalStateOffset >= tokens.count + result.tokens.count - 1 &&
                        stats.finalStateOffset <= tokens.count + result.tokens.count - 1 + mtp
                let countsExact = stats.promptTokenCount == tokens.count && stats.generatedTokenCount == result.tokens.count &&
                    cachedCount == expectedCached && computedCount == tokens.count - expectedCached &&
                    stats.prefillChunkCount == (tokens.count - expectedCached == 1 ? 1 : (tokens.count - expectedCached - 2) / 416 + 2)
                let mtpExact = stats.mtpDepth == mtp && (mtp == 0 ? stats.mtp == nil : (stats.mtp?.rounds ?? 0) > 0)
                let stateObserved = !readback || (mtp > 0 ? events.isEmpty : expectedCached > 0 ? events.contains("restore") : events.contains("coldBoundary"))
                let passed = outputExact && callbacks == result.tokens && offsetExact && countsExact && mtpExact && stateObserved
                var row: [String: Any] = ["label": label, "prompt_key": promptKey, "is_cold_oracle": establishOracle,
                    "prompt_token_ids": tokens, "prefix_token_count": prefix, "max_tokens": maxTokens,
                    "generated_token_ids": result.tokens, "finish_reason": result.finishReason.rawValue,
                    "expected_cached_tokens": expectedCached, "actual_cached_tokens": cachedCount,
                    "computed_tokens": computedCount, "mtp_depth": mtp, "state_events": events,
                    "output_exact": outputExact, "callback_exact": callbacks == result.tokens,
                    "offset_exact": offsetExact, "counts_exact": countsExact, "mtp_mode_exact": mtpExact,
                    "state_observed": stateObserved, "passed": passed, "result": try object(result),
                    "memory_before": memoryBefore, "memory_after": try MX.memory()]
                if let cacheStatistics = generator.prefixCacheStatistics {
                    row["cache_statistics"] = try object(cacheStatistics)
                } else { row["cache_statistics"] = NSNull() }
                trials.append(row)
                try write()
                guard passed else { throw CLIError.usage("Prefix cache full generation checks failed: \(label)") }
            }
            func modified(_ tokens: [Int32], at index: Int) -> [Int32] {
                var result = tokens
                // Both alternatives are ordinary ASCII digit token IDs in this
                // fixed tokenizer, avoiding unsupported multimodal/reserved IDs.
                result[index] = result[index] == 16 ? 17 : 16
                return result
            }
            if suite == "long" || suite == "all" {
                let prefix = 9984, b = modified(prompt, at: min(10_000, prompt.count - 1))
                try run("cold_a", promptKey: "long_a", stateKey: "long", tokens: prompt, prefix: prefix,
                    generator: cold, expectedCached: 0, maxTokens: maximum, establishOracle: true)
                try run("cold_b", promptKey: "long_b", stateKey: "long", tokens: b, prefix: prefix,
                    generator: cold, expectedCached: 0, maxTokens: maximum, establishOracle: true)
                try run("populate_a", promptKey: "long_a", stateKey: "long", tokens: prompt, prefix: prefix,
                    generator: cached, expectedCached: 0, maxTokens: maximum)
                try run("hit_b", promptKey: "long_b", stateKey: "long", tokens: b, prefix: prefix,
                    generator: cached, expectedCached: prefix, maxTokens: maximum)
                try run("hit_a", promptKey: "long_a", stateKey: "long", tokens: prompt, prefix: prefix,
                    generator: cached, expectedCached: prefix, maxTokens: maximum)
                checks["long_cache_hit_count"] = cached.prefixCacheStatistics?.hits == 2
                try cached.clearPrefixCache(resetStatistics: true)
                anchors.removeAll()
            }
            if suite == "boundaries" || suite == "all" {
                let prefix = 1664, a = Array(prompt.prefix(2051)), b = Array(prompt.prefix(2053))
                let count = min(maximum, 16)
                try run("qsa_cold_2051", promptKey: "qsa_a", stateKey: "qsa", tokens: a, prefix: prefix,
                    generator: cold, expectedCached: 0, maxTokens: count, establishOracle: true)
                try run("qsa_cold_2053", promptKey: "qsa_b", stateKey: "qsa", tokens: b, prefix: prefix,
                    generator: cold, expectedCached: 0, maxTokens: count, establishOracle: true)
                try run("qsa_populate_2051", promptKey: "qsa_a", stateKey: "qsa", tokens: a, prefix: prefix,
                    generator: cached, expectedCached: 0, maxTokens: count)
                try run("qsa_hit_2053", promptKey: "qsa_b", stateKey: "qsa", tokens: b, prefix: prefix,
                    generator: cached, expectedCached: prefix, maxTokens: count)
                try run("qsa_hit_2051", promptKey: "qsa_a", stateKey: "qsa", tokens: a, prefix: prefix,
                    generator: cached, expectedCached: prefix, maxTokens: count)
                checks["qsa_cache_hit_count"] = cached.prefixCacheStatistics?.hits == 2
                try cached.clearPrefixCache(resetStatistics: true)
                anchors.removeAll()
            }
            if suite == "lifecycle" || suite == "all" {
                let prefix = 416, a = Array(prompt.prefix(833)), b = modified(a, at: 0)
                let count = min(maximum, 16)
                let single = try QwenGenerator(model: model,
                    prefixCacheLimits: .init(maxEntries: 1, maxBytes: 512 * 1024 * 1024))
                try run("lifecycle_cold_a", promptKey: "life_a", stateKey: "life_a", tokens: a, prefix: prefix,
                    generator: cold, expectedCached: 0, maxTokens: count, establishOracle: true)
                try run("lifecycle_cold_b", promptKey: "life_b", stateKey: "life_b", tokens: b, prefix: prefix,
                    generator: cold, expectedCached: 0, maxTokens: count, establishOracle: true)
                try run("lifecycle_populate_a", promptKey: "life_a", stateKey: "life_a", tokens: a, prefix: prefix,
                    generator: single, expectedCached: 0, maxTokens: count)
                let mismatch = try single.beginPrefill(request(a, prefix: prefix, maxTokens: count, chunk: 208))
                checks["profile_mismatch_cold_miss"] = mismatch.processedTokenCount == 0
                try mismatch.discard()
                try run("lifecycle_evict_a_with_b", promptKey: "life_b", stateKey: "life_b", tokens: b, prefix: prefix,
                    generator: single, expectedCached: 0, maxTokens: count)
                checks["one_entry_budget_evicts"] = single.prefixCacheStatistics?.entries == 1 &&
                    (single.prefixCacheStatistics?.evictions ?? 0) >= 1
                let evicted = try single.beginPrefill(request(a, prefix: prefix, maxTokens: count))
                checks["evicted_prefix_and_token_mismatch_cold_miss"] = evicted.processedTokenCount == 0
                try evicted.discard()
                let cancellation = QwenCancellation()
                let cancelled = try single.beginPrefill(request(b, prefix: prefix, maxTokens: count), cancellation: cancellation)
                checks["cancelled_request_restored_private_prefix"] = cancelled.processedTokenCount == prefix
                // Advance one real suffix chunk before cancellation. The entry
                // must remain unchanged after this private state has mutated.
                let prematureReady = try single.stepPrefill(cancelled)
                checks["cancelled_copy_advanced_suffix"] = prematureReady == nil && cancelled.processedTokenCount == 832
                cancellation.cancel()
                var cancelledCorrectly = false
                do { _ = try single.stepPrefill(cancelled) }
                catch { cancelledCorrectly = error as? QwenGenerationError == .cancelled }
                try cancelled.discard()
                checks["cancel_after_suffix_releases_cursor"] = cancelledCorrectly && cancelled.isFinished
                try run("lifecycle_hit_b_after_cancel", promptKey: "life_b", stateKey: "life_b", tokens: b, prefix: prefix,
                    generator: single, expectedCached: prefix, maxTokens: count)
                let tiny = try QwenGenerator(model: model, prefixCacheLimits: .init(maxEntries: 1, maxBytes: 1))
                try run("lifecycle_payload_too_large", promptKey: "life_a", stateKey: "life_a", tokens: a, prefix: prefix,
                    generator: tiny, expectedCached: 0, maxTokens: count)
                checks["one_byte_budget_keeps_no_payload"] = tiny.prefixCacheStatistics?.entries == 0 &&
                    tiny.prefixCacheStatistics?.logicalPayloadBytes == 0
                try run("lifecycle_mtp_cold_bypass", promptKey: "life_b", stateKey: "life_b", tokens: b, prefix: prefix,
                    generator: single, expectedCached: 0, maxTokens: count, mtp: 2)
                try single.clearPrefixCache()
                checks["clear_releases_all_payload"] = single.prefixCacheStatistics?.entries == 0 &&
                    single.prefixCacheStatistics?.logicalPayloadBytes == 0
                let publicationCancellation = QwenCancellation()
                let publication = try QwenGenerator(model: model,
                    prefixCacheLimits: .init(maxEntries: 2, maxBytes: 512 * 1024 * 1024))
                var pendingPublishEvents = 0, publicationError: Error?
                publication.prefixStateObserver = { event, _ in
                    if event == "publish" {
                        // The copy is fully evaluated, but insertion has not
                        // happened. Cancellation must prevent its publication.
                        pendingPublishEvents += 1
                        publicationCancellation.cancel()
                    }
                }
                do {
                    _ = try publication.generate(request(a, prefix: prefix, maxTokens: count),
                        cancellation: publicationCancellation)
                } catch { publicationError = error }
                publication.prefixStateObserver = nil
                checks["cancel_between_snapshot_and_insert"] = pendingPublishEvents == 1 &&
                    publicationError as? QwenGenerationError == .cancelled
                checks["cancelled_snapshot_never_published"] = publication.prefixCacheStatistics?.entries == 0 &&
                    publication.prefixCacheStatistics?.logicalPayloadBytes == 0 &&
                    publication.prefixCacheStatistics?.published == 0
                report["publication_cancellation"] = ["pending_publish_observer_events": pendingPublishEvents,
                    "error": publicationError.map { String(describing: $0) } ?? NSNull(),
                    "cache_statistics": try object(publication.prefixCacheStatistics)]
                try run("lifecycle_populate_after_publish_cancel", promptKey: "life_a", stateKey: "life_a", tokens: a, prefix: prefix,
                    generator: publication, expectedCached: 0, maxTokens: count)
                try run("lifecycle_hit_after_publish_cancel", promptKey: "life_a", stateKey: "life_a", tokens: a, prefix: prefix,
                    generator: publication, expectedCached: prefix, maxTokens: count)
                checks["publication_cancel_cache_recovers"] = publication.prefixCacheStatistics?.entries == 1 &&
                    publication.prefixCacheStatistics?.published == 1 && publication.prefixCacheStatistics?.hits == 1
                try publication.clearPrefixCache()
                // Two complete recurrent checkpoints on the same token path.
                // K=P-1 is legal and the deepest hit executes only the final
                // prompt token, with no extra prefix replay or empty chunk.
                try run("lifecycle_cold_anchor_832", promptKey: "life_a", stateKey: "life_a", tokens: a, prefix: 832,
                    generator: cold, expectedCached: 0, maxTokens: count)
                let tree = try QwenGenerator(model: model,
                    prefixCacheLimits: .init(maxEntries: 2, maxBytes: 512 * 1024 * 1024))
                try run("lifecycle_tree_populate_416", promptKey: "life_a", stateKey: "life_a", tokens: a, prefix: 416,
                    generator: tree, expectedCached: 0, maxTokens: count)
                try run("lifecycle_tree_hit_416_publish_832", promptKey: "life_a", stateKey: "life_a", tokens: a, prefix: 832,
                    generator: tree, expectedCached: 416, maxTokens: count)
                try run("lifecycle_tree_longest_832", promptKey: "life_a", stateKey: "life_a", tokens: a, prefix: 832,
                    generator: tree, expectedCached: 832, maxTokens: count)
                checks["radix_longest_complete_checkpoint"] = tree.prefixCacheStatistics?.entries == 2 &&
                    tree.prefixCacheStatistics?.hits == 2 && tree.prefixCacheStatistics?.evictions == 0
                try tree.clearPrefixCache()
                anchors.removeAll()
            }
            checks["full_model_48_layers"] = model.layerCount == 48
            checks["readback_coverage"] = !readback || !stateChecks.isEmpty
            report["memory_at_completion"] = try MX.memory()
            try MX.synchronize()
            try write(complete: true)
            guard report["passed"] as? Bool == true else { throw CLIError.usage("Prefix cache probe did not pass every required check") }
        } catch {
            report["error"] = String(describing: error)
            report["passed"] = false
            try write()
            throw error
        }
    }
}
