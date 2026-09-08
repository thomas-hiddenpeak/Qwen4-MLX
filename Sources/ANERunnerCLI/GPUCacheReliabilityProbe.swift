import ANERunnerCore
import ANERunnerGPU
import CMLX
import Foundation

private struct CacheReliabilityAnchor: Codable, Equatable {
    struct Value: Codable, Equatable {
        let shape: [Int]
        let dtype, byteCount: Int
        let sha256: String
        let finite: Bool
    }
    let tensors: [String: Value]
    let host: QwenModel.State.DiagnosticHostValues

    init(_ state: QwenModel.State) throws {
        host = state.diagnosticHostValues
        var tensors = [String: Value]()
        for (name, tensor) in state.namedTensors {
            let bytes = try MoETilingBytes.bytes(tensor)
            guard tensor.dtype == MLX_BFLOAT16, bytes.count % 2 == 0 else {
                throw CLIError.usage("Reliability anchor expected native BF16 state")
            }
            // Inspect BF16 exponent bits directly, preserving the original
            // dtype and avoiding another Float32 tensor/readback allocation.
            let finite = bytes.withUnsafeBytes { (p: UnsafeRawBufferPointer) -> Bool in
                for i in stride(from: 0, to: p.count, by: 2) {
                    if p[i + 1] & 0x7f == 0x7f && p[i] & 0x80 == 0x80 { return false }
                }
                return true
            }
            tensors[name] = Value(shape: tensor.shape, dtype: Int(tensor.dtype.rawValue),
                byteCount: bytes.count, sha256: MoETilingBytes.digest(bytes), finite: finite)
        }
        self.tensors = tensors
    }

    var valid: Bool {
        host.valid && host.offset > 0 && !tensors.isEmpty && tensors.values.allSatisfy(\.finite) &&
            !host.gdnCapturePresent.contains(true) && !host.pleCapturePresent.contains(true)
    }

    func matches(_ other: Self) -> Bool {
        // Compact copies normalize storage extents. Logical state must agree;
        // the current entry and restored copy are compared separately below.
        valid && other.valid && tensors == other.tensors && host.offset == other.host.offset &&
            host.gdnOffsets == other.host.gdnOffsets && host.attentionOffsets == other.host.attentionOffsets &&
            host.pleHistory == other.host.pleHistory && host.gdnCapturePresent == other.host.gdnCapturePresent &&
            host.pleCapturePresent == other.host.pleCapturePresent
    }
}

extension RunnerCLI {
    /// Separate populate/restore invocations make process-restart reuse a real
    /// test. Diagnostic tensor readback means these are not throughput trials.
    static func probeGPUCacheReliability(_ args: Arguments) throws {
        try args.validate(["--model-dir", "--tokens-file", "--cache-directory", "--output", "--mode", "--oracle-report"])
        let output = try args.require("--output"), mode = try args.require("--mode")
        guard !FileManager.default.fileExists(atPath: output),
              ["populate", "restore", "corrupt", "lifecycle"].contains(mode),
              !["restore", "corrupt"].contains(mode) || args["--oracle-report"] != nil else {
            throw CLIError.usage("Cache reliability probe needs new --output, --mode populate|restore|corrupt|lifecycle; restore/corrupt also need --oracle-report")
        }
        var report: [String: Any] = ["schema": "qwen38-cache-reliability-v1", "mode": mode,
            "complete": false, "passed": false,
            "scope": "Real AR output/state correctness and cache lifetime diagnostics; tensor readback perturbs timings.",
            "notes": ["Every native BF16 tensor is hashed; host state includes offsets and PLE token history.",
                "Logical state leases exclude model weights, transient activations and MLX allocator retention.",
                "Concurrent cursors are cooperatively interleaved on the same inference executor."]]
        var checks = [String: Bool](), trials = [[String: Any]](), stateChecks = [[String: Any]]()
        func object<T: Encodable>(_ value: T) throws -> Any {
            try JSONSerialization.jsonObject(with: JSONEncoder().encode(value))
        }
        func write(complete: Bool = false) throws {
            report["complete"] = complete; report["checks"] = checks
            report["trials"] = trials; report["state_checks"] = stateChecks
            report["passed"] = complete && !checks.isEmpty && !trials.isEmpty &&
                checks.values.allSatisfy { $0 } && trials.allSatisfy { $0["passed"] as? Bool == true } &&
                stateChecks.allSatisfy { $0["passed"] as? Bool == true }
            try emit(report, to: output)
        }
        func require(_ label: String, _ condition: Bool) throws {
            checks[label] = condition
            try write()
            guard condition else { throw CLIError.usage("Cache reliability check failed: \(label)") }
        }
        try write()
        do {
            let directory = URL(fileURLWithPath: try args.require("--model-dir")).standardizedFileURL.resolvingSymlinksInPath()
            let inputURL = URL(fileURLWithPath: try args.require("--tokens-file")).standardizedFileURL
            let input = try Data(contentsOf: inputURL)
            let prompt = try JSONDecoder().decode([Int32].self, from: input)
            guard prompt.count == 11_057 else { throw CLIError.usage("Expected the real 11057-token agent fixture") }
            let executable = Bundle.main.executableURL ?? URL(fileURLWithPath: CommandLine.arguments[0])
            let provenance = ["executable_sha256": try MoETilingBytes.hash(executable),
                "config_sha256": try MoETilingBytes.hash(directory.appendingPathComponent("config.json")),
                "tokenizer_sha256": try MoETilingBytes.hash(directory.appendingPathComponent("tokenizer.json")),
                "input_sha256": MoETilingBytes.digest(input)]
            report["provenance"] = provenance
            report["command"] = CommandLine.arguments
            report["model_directory"] = directory.path
            report["source_prompt_token_ids"] = prompt
            let disk = try QwenPrefixDiskStore(directory: URL(fileURLWithPath: try args.require("--cache-directory")),
                limits: .init(maxEntries: 8, maxBytes: 2 * 1024 * 1024 * 1024,
                    maxPendingJobs: 2, maxPendingBytes: 1024 * 1024 * 1024))
            defer { disk.close(drain: true) }
            report["disk_at_start"] = try object(disk.statistics)
            if mode != "restore" {
                try require("dedicated_store_empty", disk.statistics.entries == 0)
            }
            var previousCache = 0
            try MX.check(mlx_set_cache_limit(&previousCache, 256 * 1024 * 1024), "reliability allocator cache limit")
            defer { var ignored = 0; _ = mlx_set_cache_limit(&ignored, previousCache) }
            let model = try QwenModel(modelDirectory: directory) { current, total in
                if current % 8 == 0 || current == total {
                    FileHandle.standardError.write(Data("Cache reliability: loaded \(current)/\(total) layers\n".utf8))
                }
            }
            defer { try? MX.synchronize() }
            let generator = try QwenGenerator(model: model,
                prefixCacheLimits: .init(maxEntries: 8, maxBytes: 512 * 1024 * 1024), prefixDiskStore: disk)
            func request(_ tokens: [Int32], _ prefix: Int, output: Int = 32) -> QwenGenerationRequest {
                QwenGenerationRequest(tokens: tokens, maxTokens: output, contextLimit: 16_384,
                    prefillChunk: 416, mtpDepth: 0, prefixCacheMaxTokens: prefix)
            }
            func clean(_ label: String, requireEmptyCache: Bool = false) throws {
                try generator.flushPrefixCacheWrites()
                let budget = generator.stateBudgetStatistics
                report["budget_" + label] = try object(budget)
                try require(label + "_request_workspace_released", budget.requestBytes == 0 && budget.workspaceBytes == 0)
                try require(label + "_within_budget", budget.totalBytes <= budget.maxBytes)
                if requireEmptyCache {
                    try require(label + "_cache_released", budget.cacheBytes == 0 && budget.totalBytes == 0)
                }
                let stats = disk.statistics
                try require(label + "_disk_queue_drained", stats.pendingJobs == 0 && stats.pendingBytes == 0)
            }
            func record(_ label: String, _ result: QwenGenerationResult,
                        tokens: [Int32], finish: String, cached: Int? = nil, source: String? = nil) throws {
                let prefill = result.phases?.prefill
                let exact = result.tokens == tokens && result.finishReason.rawValue == finish
                let cacheExact = cached.map { $0 == prefill?.cachedTokenCount } ?? true
                let sourceExact = source.map { $0 == prefill?.cacheSource } ?? true
                let offsets = result.statistics.finalStateOffset == result.statistics.promptTokenCount + result.tokens.count - 1
                let passed = exact && cacheExact && sourceExact && offsets && result.statistics.mtpDepth == 0
                trials.append(["label": label, "passed": passed, "output_exact": exact,
                    "cache_count_exact": cacheExact, "source_exact": sourceExact, "offset_exact": offsets,
                    "generated_token_ids": result.tokens, "finish_reason": result.finishReason.rawValue,
                    "result": try object(result), "disk_statistics": try object(disk.statistics),
                    "budget": try object(generator.stateBudgetStatistics)])
                try write()
                guard passed else { throw CLIError.usage("Cache reliability output mismatch: \(label)") }
            }
            func finishPrefill(_ session: QwenPrefillSession) throws -> QwenPrefillResult {
                let deadline = Date().addingTimeInterval(600)
                while Date() < deadline {
                    let before = session.processedTokenCount
                    if let ready = try generator.stepPrefill(session) { return ready }
                    if session.processedTokenCount == before { Thread.sleep(forTimeInterval: 0.001) }
                }
                throw CLIError.usage("Timed out waiting for prefix cache/prefill progress")
            }
            func advanceSuffix(_ session: QwenPrefillSession, prefix: Int) throws {
                let deadline = Date().addingTimeInterval(600)
                while session.processedTokenCount <= prefix && Date() < deadline {
                    let before = session.processedTokenCount
                    if let ready = try generator.stepPrefill(session) {
                        ready.discard()
                        throw CLIError.usage("Expected a paused prefill suffix before final handoff")
                    }
                    if session.processedTokenCount == before { Thread.sleep(forTimeInterval: 0.001) }
                }
                guard session.processedTokenCount > prefix else { throw CLIError.usage("No suffix progress") }
            }
            func decode(_ prepared: QwenPrefillResult) throws -> QwenGenerationResult {
                defer { prepared.discard() }
                return try generator.decode(prepared)
            }
            func cancellation(_ session: QwenPrefillSession, token: QwenCancellation, label: String) throws {
                token.cancel()
                var cancelled = false
                do { _ = try generator.stepPrefill(session) }
                catch { cancelled = error as? QwenGenerationError == .cancelled }
                try session.discard()
                try require(label, cancelled && session.isFinished)
            }

            if mode == "populate" || mode == "restore" || mode == "corrupt" {
                let prefix = 9984, req = request(prompt, prefix)
                var anchor: CacheReliabilityAnchor?
                var expectedTokens = [Int32](), expectedFinish = ""
                if mode == "restore" || mode == "corrupt" {
                    let data = try Data(contentsOf: URL(fileURLWithPath: try args.require("--oracle-report")))
                    guard let oracle = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                          oracle["schema"] as? String == "qwen38-cache-reliability-v1",
                          oracle["mode"] as? String == "populate", oracle["complete"] as? Bool == true,
                          oracle["passed"] as? Bool == true, let priorProvenance = oracle["provenance"] as? [String: String],
                          priorProvenance == provenance, let anchorObject = oracle["anchor"],
                          let ids = oracle["oracle_token_ids"] as? [Int],
                          let finish = oracle["oracle_finish_reason"] as? String else {
                        throw CLIError.usage("Invalid or incompatible populate oracle")
                    }
                    anchor = try JSONDecoder().decode(CacheReliabilityAnchor.self,
                        from: JSONSerialization.data(withJSONObject: anchorObject))
                    expectedTokens = try ids.map {
                        guard let id = Int32(exactly: $0) else { throw CLIError.usage("Invalid oracle token") }
                        return id
                    }
                    expectedFinish = finish
                    if mode == "restore" {
                        try require("prior_process_entry_recovered", disk.statistics.recoveredEntries > 0)
                    } else {
                        try require("corrupt_archive_rejected_at_recovery", disk.statistics.corruptions > 0 &&
                            disk.statistics.recoveredEntries == 0 && disk.statistics.entries == 0)
                    }
                    report["oracle_report"] = args["--oracle-report"]
                }
                var eventCounts = [String: Int]()
                generator.prefixStateObserver = { event, state in
                    eventCounts[event, default: 0] += 1
                    guard state.offset == prefix else { throw CLIError.usage("Unexpected archive anchor offset") }
                    let observed = try CacheReliabilityAnchor(state)
                    if anchor == nil {
                        guard mode == "populate", event == "coldBoundary" else {
                            throw CLIError.usage("Archive anchor was not established by cold prefill")
                        }
                        anchor = observed
                        report["anchor"] = try object(observed)
                    }
                    let exact = anchor?.matches(observed) == true
                    stateChecks.append(["event": event, "offset": state.offset, "passed": exact,
                        "tensor_count": observed.tensors.count, "all_finite": observed.valid,
                        "observed": try object(observed)])
                    guard exact else { throw CLIError.usage("Archive native state differs from cold anchor") }
                }
                defer { generator.prefixStateObserver = nil }
                let result = try generator.generate(req)
                if mode == "populate" {
                    expectedTokens = result.tokens; expectedFinish = result.finishReason.rawValue
                    report["oracle_token_ids"] = expectedTokens; report["oracle_finish_reason"] = expectedFinish
                    try require("nontrivial_32_token_oracle", expectedTokens.count == 32)
                    try record("cold_populate", result, tokens: expectedTokens, finish: expectedFinish, cached: 0, source: "cold")
                    try require("cold_and_publish_anchors_observed", eventCounts["coldBoundary"] == 1 && eventCounts["publish"] == 1)
                    try clean("populate")
                    try require("archive_durable_after_flush", disk.statistics.published == 1 && disk.statistics.entries == 1 &&
                        disk.statistics.writeFailures == 0 && disk.statistics.bytesWritten > 0)
                    try generator.clearPrefixCache()
                    try clean("populate_ram_clear", requireEmptyCache: true)
                    try require("ram_clear_preserves_disk", disk.statistics.entries == 1)
                } else if mode == "corrupt" {
                    try record("corrupt_archive_cold_fallback", result, tokens: expectedTokens,
                        finish: expectedFinish, cached: 0, source: "cold")
                    try require("fallback_cold_and_publish_anchors_observed", eventCounts["coldBoundary"] == 1 &&
                        eventCounts["publish"] == 1 && eventCounts["restore"] == nil)
                    try clean("corrupt_fallback")
                    try require("corrupt_fallback_replaced_with_valid_archive", disk.statistics.corruptions > 0 &&
                        disk.statistics.recoveredEntries == 0 && disk.statistics.published == 1 && disk.statistics.entries == 1)
                    try generator.clearPrefixCache()
                    try clean("corrupt_fallback_ram_clear", requireEmptyCache: true)
                } else {
                    try record("new_process_disk_restore", result, tokens: expectedTokens,
                        finish: expectedFinish, cached: prefix, source: "disk")
                    try require("first_restore_anchor_observed", eventCounts["restore"] == 1)
                    try clean("first_restore")
                    try generator.clearPrefixCache()
                    let cancelled = QwenCancellation()
                    let left = try generator.beginPrefill(req, cancellation: cancelled)
                    defer { try? left.discard() }
                    try advanceSuffix(left, prefix: prefix)
                    let right = try generator.beginPrefill(req)
                    defer { try? right.discard() }
                    try advanceSuffix(right, prefix: prefix)
                    let before = generator.stateBudgetStatistics
                    try require("paused_cursors_retain_private_state", before.requestBytes > 0)
                    try generator.clearPrefixCache(includingDisk: true)
                    try require("cache_clear_keeps_live_state_reserved", generator.stateBudgetStatistics.requestBytes == before.requestBytes)
                    try require("cache_clear_removes_both_tiers", generator.prefixCacheStatistics?.entries == 0 && disk.statistics.entries == 0)
                    try cancellation(left, token: cancelled, label: "cancelled_private_restore_released")
                    let survivor = try decode(finishPrefill(right))
                    try record("survivor_after_peer_cancel_and_cache_clear", survivor,
                        tokens: expectedTokens, finish: expectedFinish, cached: prefix)
                    try require("three_independent_restores_observed", eventCounts["restore"] == 3)
                    try clean("private_lifetime", requireEmptyCache: true)
                }
                report["state_event_counts"] = eventCounts
            } else {
                let cold = try QwenGenerator(model: model)
                for (length, prefix) in [(833, 416), (2053, 1664)] {
                    let label = "p\(length)_k\(prefix)", tokens = Array(prompt.prefix(length))
                    let req = request(tokens, prefix, output: 16)
                    let oracle = try cold.generate(req)
                    try record(label + "_cold_oracle", oracle, tokens: oracle.tokens, finish: oracle.finishReason.rawValue)
                    try generator.clearPrefixCache(includingDisk: true)
                    let leader = try generator.beginPrefill(req), follower = try generator.beginPrefill(req)
                    defer { try? leader.discard(); try? follower.discard() }
                    let before = follower.processedTokenCount
                    let waiting = try generator.stepPrefill(follower)
                    try require(label + "_follower_waits_without_duplicate_prefill", waiting == nil &&
                        before == 0 && follower.processedTokenCount == before)
                    var leaderReady: QwenPrefillResult?, followerReady: QwenPrefillResult?
                    defer { leaderReady?.discard(); followerReady?.discard() }
                    let deadline = Date().addingTimeInterval(600)
                    while (leaderReady == nil || followerReady == nil) && Date() < deadline {
                        let leaderBefore = leader.processedTokenCount, followerBefore = follower.processedTokenCount
                        if leaderReady == nil { leaderReady = try generator.stepPrefill(leader) }
                        if followerReady == nil { followerReady = try generator.stepPrefill(follower) }
                        if leader.processedTokenCount == leaderBefore && follower.processedTokenCount == followerBefore &&
                            (leaderReady == nil || followerReady == nil) { Thread.sleep(forTimeInterval: 0.001) }
                    }
                    guard let a = leaderReady, let b = followerReady else { throw CLIError.usage("Single-flight cursors stalled") }
                    let leaderResult = try decode(a), followerResult = try decode(b)
                    try record(label + "_single_flight_leader", leaderResult,
                        tokens: oracle.tokens, finish: oracle.finishReason.rawValue, cached: 0, source: "cold")
                    try record(label + "_single_flight_follower", followerResult,
                        tokens: oracle.tokens, finish: oracle.finishReason.rawValue, cached: prefix, source: "memory")
                    try clean(label + "_single_flight")

                    try generator.clearPrefixCache(includingDisk: true)
                    let cancel = QwenCancellation()
                    let abandoned = try generator.beginPrefill(req, cancellation: cancel)
                    let successor = try generator.beginPrefill(req)
                    defer { try? abandoned.discard(); try? successor.discard() }
                    if prefix > 416 { _ = try generator.stepPrefill(abandoned) }
                    let successorBefore = successor.processedTokenCount
                    let stillWaiting = try generator.stepPrefill(successor)
                    try require(label + "_replacement_waits_for_leader", stillWaiting == nil &&
                        successorBefore == 0 && successor.processedTokenCount == 0)
                    try cancellation(abandoned, token: cancel, label: label + "_leader_cancelled")
                    let takeover = try decode(finishPrefill(successor))
                    try record(label + "_leader_cancel_takeover", takeover,
                        tokens: oracle.tokens, finish: oracle.finishReason.rawValue, cached: 0, source: "cold")
                    try clean(label + "_takeover")

                    // Admission may evict retained RAM entries before retrying
                    // a reservation. Remove that reclaimable capacity first so
                    // this check tests a guaranteed denial, not eviction policy.
                    try generator.clearPrefixCache()
                    let pre = generator.stateBudgetStatistics
                    let available = pre.maxBytes - pre.totalBytes
                    guard available > 0, let pressure = model.stateBudget.reserve(bytes: available, kind: .workspace) else {
                        throw CLIError.usage("Could not create bounded state budget pressure")
                    }
                    var admissionError: String?
                    do {
                        let unwanted = try generator.beginPrefill(req)
                        try unwanted.discard()
                    } catch { admissionError = String(describing: error) }
                    report[label + "_admission_error"] = admissionError ?? NSNull()
                    pressure.release()
                    try require(label + "_budget_rejects_before_state_allocation", admissionError != nil &&
                        generator.stateBudgetStatistics.rejections > pre.rejections)
                    try clean(label + "_pressure_released")
                    let recovered = try generator.generate(req)
                    try record(label + "_admission_recovers_after_pressure", recovered,
                        tokens: oracle.tokens, finish: oracle.finishReason.rawValue)
                    try generator.clearPrefixCache(includingDisk: true)
                    try clean(label + "_complete", requireEmptyCache: true)
                }
            }
            report["disk_at_completion"] = try object(disk.statistics)
            report["state_budget_at_completion"] = try object(generator.stateBudgetStatistics)
            report["mlx_memory_at_completion"] = try MX.memory()
            try require("full_model", model.layerCount == 48)
            try MX.synchronize()
            try write(complete: true)
            guard report["passed"] as? Bool == true else { throw CLIError.usage("Cache reliability probe did not pass") }
        } catch {
            report["error"] = String(describing: error)
            try write()
            throw error
        }
    }
}
