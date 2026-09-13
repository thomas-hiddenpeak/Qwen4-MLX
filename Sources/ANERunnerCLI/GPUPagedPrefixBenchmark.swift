import ANERunnerCore
import ANERunnerGPU
import CMLX
import Foundation

extension RunnerCLI {
    /// Real warm-cache requests; no observers, tensor hashes or retained states.
    static func benchmarkGPUPagedPrefixCache(_ args: Arguments) throws {
        try args.validate(["--model-dir", "--tokens-file", "--library", "--order", "--output"])
        let output = try args.require("--output"), order = args["--order"] ?? "both"
        guard !FileManager.default.fileExists(atPath: output), ["both", "abba", "baab"].contains(order) else {
            throw CLIError.usage("Paged prefix benchmark requires new --output and --order both|abba|baab")
        }
        let directory = URL(fileURLWithPath: try args.require("--model-dir")).standardizedFileURL.resolvingSymlinksInPath()
        let library = URL(fileURLWithPath: try args.require("--library")).standardizedFileURL.resolvingSymlinksInPath()
        let tokenURL = URL(fileURLWithPath: try args.require("--tokens-file")).standardizedFileURL
        let input = try Data(contentsOf: tokenURL), prompt = try JSONDecoder().decode([Int32].self, from: input)
        guard prompt.count == 11_057 else { throw CLIError.usage("Use the fixed 11057-token agent fixture") }
        let boundary = 10_816, suffix = 241, pageCount = 512, cacheBytes = 512 * 1024 * 1024
        let stock = "stock_capacity256", paged = "physical_paged32"
        let orders = order == "both" ? ["abba", "baab"] : [order]
        var checks = [String: Bool](), populate = [[String: Any]](), trials = [[String: Any]]()
        var measured = [(order: String, mode: String, result: QwenGenerationResult)]()
        var report: [String: Any] = ["schema": "qwen-paged-prefix-benchmark-v1", "complete": false, "passed": false,
            "configuration": ["prompt_tokens": prompt.count, "prefix_tokens": boundary, "computed_warm_tokens": suffix,
                "context": 16_384, "prefill_chunk": 416, "prefill_evaluate_every_layers": 4, "mtp_depth": 0,
                "cold_output_tokens": 16, "warm_output_tokens": 128, "orders": orders,
                "maximum_pages_per_layer": pageCount, "cache_entries_per_mode": 2,
                "cache_max_bytes_per_mode": cacheBytes, "state_budget_bytes": 8 * 1024 * 1024 * 1024,
                "state_observers": false],
            "notes": [
                "A uses stock SDPA/capacity256 plus RAM cache; B uses paged32 plus an independent RAM cache. One fixed physical context remains allocated in both modes.",
                "Both cold populate requests are recorded separately and excluded from every warm performance summary.",
                "The paged cache retains the dense mixed-state snapshot plus its immutable paged KV attachment. This is not a dense-cache removal or an RSS savings measurement.",
                "Warm prefill rates use the 241 actual forwarded tokens. The complete 11057-token prompt remains API usage, not the warm compute-rate numerator.",
                "The generator's suffix import, admission, per-step page checks and cleanup remain in its decode timing. Additional whole-request snapshots below are outside that timing.",
                "Passed means output/accounting/lifetime validation, not a required speedup. Compare both order groups and their per-trial rates."]]
        func object<T: Encodable>(_ value: T) throws -> Any { try JSONSerialization.jsonObject(with: JSONEncoder().encode(value)) }
        func save(complete: Bool = false) throws {
            report["checks"] = checks; report["populate_trials"] = populate; report["warm_trials"] = trials
            report["complete"] = complete
            report["passed"] = complete && populate.count == 2 && trials.count == orders.count * 4 &&
                !checks.isEmpty && checks.values.allSatisfy { $0 }
            try emit(report, to: output)
        }
        func require(_ label: String, _ passed: Bool) throws {
            checks[label] = passed
            guard passed else { throw CLIError.usage("Paged prefix benchmark failed: \(label)") }
        }
        try save()
        do {
            let executable = Bundle.main.executableURL ?? URL(fileURLWithPath: CommandLine.arguments[0])
            report["provenance"] = ["command": CommandLine.arguments, "captured_utc": Date().ISO8601Format(),
                "os": ProcessInfo.processInfo.operatingSystemVersionString, "model_directory": directory.path,
                "executable_sha256": try MoETilingBytes.hash(executable), "pool_library_path": library.path,
                "pool_library_sha256": try MoETilingBytes.hash(library), "tokens_path": tokenURL.path,
                "tokens_sha256": MoETilingBytes.digest(input), "model_cache_identity": try QwenPrefixCacheIdentity.fingerprint(modelDirectory: directory),
                "config_sha256": try MoETilingBytes.hash(directory.appendingPathComponent("config.json")),
                "tokenizer_sha256": try MoETilingBytes.hash(directory.appendingPathComponent("tokenizer.json")),
                "base_mlx": try MoEGateUpProbeSupport.baseMLX()]
            var oldCache = 0
            try MX.check(mlx_set_cache_limit(&oldCache, 256 * 1024 * 1024), "Paged prefix benchmark allocator cache")
            defer { var ignored = 0; _ = mlx_set_cache_limit(&ignored, oldCache) }
            let tokenizer = try QwenTokenizer(modelDirectory: directory)
            let model = try QwenModel(modelDirectory: directory, reservedOutputIDs: tokenizer.reservedOutputTokenIDs,
                stateBudgetBytes: 8 * 1024 * 1024 * 1024) { count, total in
                if count % 8 == 0 || count == total { FileHandle.standardError.write(Data("Paged prefix benchmark loaded \(count)/\(total)\n".utf8)) }
            }
            defer { try? MX.synchronize() }
            try require("full_model_no_mtp_or_async_experiment", model.layerCount == 48 && model.experimentalDecodeAsyncEveryLayers == 0 &&
                !model.weights.ledger.contains { $0.name.contains(".mtp.") })
            let context = try model.makePagedKVContext(libraryPath: library.path, maximumPagesPerLayer: pageCount)
            try MX.synchronize()
            let initialPools = try context.layerStatistics, arenaBytes = context.reservedArenaBytes
            report["fixed_context"] = ["reserved_bytes": arenaBytes, "initial_layers": try object(initialPools),
                "present_during_both_modes": true, "fused_prefill": model.fusedPrefillEnabled]
            try require("initial_twelve_empty_arenas", initialPools.count == 12 && initialPools.values.allSatisfy { $0.livePages == 0 })
            let baseline = try QwenGenerator(model: model, prefixCacheLimits: .init(maxEntries: 2, maxBytes: cacheBytes))
            let candidate = try QwenGenerator(model: model, prefixCacheLimits: .init(maxEntries: 2, maxBytes: cacheBytes), pagedKVContext: context)
            defer { try? baseline.clearPrefixCache(); try? candidate.clearPrefixCache(); try? MX.synchronize() }
            var pagedPopulated = false
            func cacheSnapshot() throws -> (raw: Any, bytes: Int) {
                guard let a = baseline.prefixCacheStatistics, let b = candidate.prefixCacheStatistics else { throw CLIError.usage("Missing RAM cache statistics") }
                try require("cache_bounds", a.entries <= 2 && b.entries <= 2 && a.logicalPayloadBytes <= cacheBytes && b.logicalPayloadBytes <= cacheBytes)
                return ([stock: try object(a), paged: try object(b)], a.logicalPayloadBytes + b.logicalPayloadBytes)
            }
            func run(_ mode: String, label: String, cold: Bool, orderName: String = "populate") throws -> QwenGenerationResult {
                try MX.synchronize()
                let beforePools = try context.layerStatistics, beforeBudget = model.stateBudget.statistics
                let beforeCache = try cacheSnapshot(), beforeMemory = try MX.memory(), beforeClaims = context.pageAdmissionStatistics
                try require(label + "_admission_clean_before", beforeBudget.requestBytes == 0 && beforeBudget.workspaceBytes == arenaBytes &&
                    beforeBudget.cacheBytes == beforeCache.bytes && beforeClaims.activeClaims == 0 && beforeClaims.claimedPages == 0)
                let isPaged = mode == paged, maximum = cold ? 16 : 128
                FileHandle.standardError.write(Data("Paged prefix benchmark \(label): \(mode) O\(maximum)\n".utf8))
                let result = try (isPaged ? candidate : baseline).generate(.init(tokens: prompt, maxTokens: maximum,
                    contextLimit: 16_384, prefillChunk: 416, mtpDepth: 0, prefillEvaluateEveryLayers: 4,
                    prefixCacheMaxTokens: boundary, kvAppendMode: isPaged ? .reference : .capacity256))
                try MX.synchronize()
                let afterPools = try context.layerStatistics, afterBudget = model.stateBudget.statistics
                let afterCache = try cacheSnapshot(), afterClaims = context.pageAdmissionStatistics
                if isPaged { pagedPopulated = true }
                guard let phases = result.phases else { throw CLIError.usage("Missing phase statistics") }
                let steps = result.statistics.decodedTokenCount, computed = cold ? prompt.count : suffix
                try require(label + "_output_contract", result.tokens.count == maximum && result.finishReason == .length &&
                    steps == maximum - 1 && result.statistics.decodeRounds == steps && result.statistics.mtpDepth == 0 &&
                    result.statistics.finalStateOffset == prompt.count + steps && result.decodeSeconds.isFinite && result.decodeSeconds > 0 &&
                    phases.decodeServiceSeconds.isFinite && phases.decodeServiceSeconds > 0 && result.timeToFirstTokenSeconds.isFinite && result.timeToFirstTokenSeconds > 0)
                try require(label + "_prefill_contract", phases.prefill.promptTokenCount == prompt.count && phases.prefill.evaluateEveryLayers == 4 &&
                    phases.prefill.cachedTokenCount == (cold ? 0 : boundary) && phases.prefill.computedTokenCount == computed &&
                    phases.prefill.actualForwardTokenCount == computed && phases.prefill.cacheSource == (cold ? "cold" : "memory") &&
                    phases.prefill.targetSeconds.isFinite && phases.prefill.targetSeconds > 0 && phases.prefill.totalSeconds.isFinite && phases.prefill.totalSeconds > 0)
                if isPaged {
                    try require(label + "_paged_phase", phases.kvAppendMode == "paged32" && phases.pagedKVTokenSteps == steps &&
                        phases.pagedKVReusedPrefixTokens == boundary && phases.pagedKVImportedSuffixRows == suffix &&
                        phases.pagedKVCapacityFallbacks == 0 && (phases.pagedKVImportSeconds ?? 0) > 0 &&
                        (phases.pagedKVImportSeconds ?? .infinity) <= result.decodeSeconds && phases.kvCapacityTokenSteps == 0)
                } else {
                    try require(label + "_stock_phase", phases.kvAppendMode == "capacity256" && phases.kvCapacityTokenSteps == steps &&
                        phases.kvCapacityWorkspaceFallbacks == 0 && phases.pagedKVTokenSteps == nil)
                }
                var deltas = [[String: Any]]()
                for layer in beforePools.keys.sorted() {
                    guard let a = beforePools[layer], let b = afterPools[layer] else { throw CLIError.usage("Arena layer set changed") }
                    try require(label + "_layer\(layer)_counters_monotonic", b.encodedWrites >= a.encodedWrites && b.encodedReads >= a.encodedReads &&
                        b.encodedMaterializations >= a.encodedMaterializations && b.writtenRowBytes >= a.writtenRowBytes && b.copiedTailBytes >= a.copiedTailBytes)
                    let writes = b.encodedWrites - a.encodedWrites, reads = b.encodedReads - a.encodedReads
                    let rowBytes = b.writtenRowBytes - a.writtenRowBytes, tailBytes = b.copiedTailBytes - a.copiedTailBytes
                    let expectedWrites = isPaged ? steps + 1 + (cold ? 1 : 0) : 0
                    let expectedRows = isPaged ? suffix + steps + (cold ? boundary : 0) : 0
                    let expectedTail = isPaged ? (0..<steps).reduce(0) { $0 + (prompt.count + $1) % 32 } : 0
                    try require(label + "_layer\(layer)_direct_ops", writes == UInt64(expectedWrites) && reads == UInt64(isPaged ? steps : 0) &&
                        b.encodedMaterializations == a.encodedMaterializations && b.materializedBytes == a.materializedBytes &&
                        rowBytes == UInt64(expectedRows * 2_048) && tailBytes == UInt64(expectedTail * 2_048))
                    try require(label + "_layer\(layer)_only_cached_prefix", b.livePages == UInt64(pagedPopulated ? boundary / 32 : 0) &&
                        b.physicalPages == UInt64(pageCount) && b.freePages + b.livePages == b.physicalPages && b.inFlightOperations == 0 && b.failedOperations == 0 &&
                        b.completedOperations == b.encodedWrites + b.encodedReads + b.encodedMaterializations && b.arenaAllocatedBytes == a.arenaAllocatedBytes &&
                        b.keyBufferIdentity == a.keyBufferIdentity && b.valueBufferIdentity == a.valueBufferIdentity)
                    if !isPaged { try require(label + "_layer\(layer)_stock_no_pool_activity", a == b) }
                    deltas.append(["layer": layer, "writes": writes, "reads": reads, "materializations": b.encodedMaterializations - a.encodedMaterializations,
                        "written_row_bytes": rowBytes, "copied_tail_bytes": tailBytes, "expected_written_rows": expectedRows])
                }
                try require(label + "_request_released", afterBudget.requestBytes == 0 && afterBudget.workspaceBytes == arenaBytes &&
                    afterBudget.cacheBytes == afterCache.bytes && afterBudget.totalBytes == arenaBytes + afterCache.bytes &&
                    afterClaims.activeClaims == 0 && afterClaims.claimedPages == 0 && afterClaims.deniedClaims == beforeClaims.deniedClaims)
                let row: [String: Any] = ["label": label, "mode": mode, "order": orderName, "generated_token_ids": result.tokens, "result": try object(result),
                    "actual_prefill_tokens": computed, "actual_prefill_tokens_per_second": Double(computed) / phases.prefill.targetSeconds,
                    "prefill_ready_tokens_per_second": Double(computed) / phases.prefill.totalSeconds, "time_to_first_token_seconds": result.timeToFirstTokenSeconds,
                    "pool_before": try object(beforePools), "pool_after": try object(afterPools), "pool_deltas": deltas,
                    "budget_before": try object(beforeBudget), "budget_after": try object(afterBudget), "cache_before": beforeCache.raw, "cache_after": afterCache.raw,
                    "page_claims_before": try object(beforeClaims), "page_claims_after": try object(afterClaims),
                    "mlx_memory_before": beforeMemory, "mlx_memory_after": try MX.memory()]
                if cold { populate.append(row) } else { trials.append(row); measured.append((orderName,mode,result)) }
                try save(); return result
            }
            let coldA = try run(stock, label: "populate_a", cold: true)
            let coldB = try run(paged, label: "populate_b", cold: true)
            try require("cold_modes_output_exact", coldA.tokens == coldB.tokens && coldA.finishReason == coldB.finishReason)
            var fullOracle: QwenGenerationResult?, modeOracles = [String: QwenGenerationResult]()
            for group in orders {
                let sequence = group == "abba" ? [stock,paged,paged,stock] : [paged,stock,stock,paged]
                for (ordinal, mode) in sequence.enumerated() {
                    let label = group + "_\(ordinal)_" + mode
                    let result = try run(mode, label: label, cold: false, orderName: group)
                    try require(label + "_cold_oracle_prefix_exact", Array(result.tokens.prefix(coldA.tokens.count)) == coldA.tokens)
                    if let fullOracle { try require(label + "_complete_cross_mode_exact", result.tokens == fullOracle.tokens && result.finishReason == fullOracle.finishReason) }
                    else { fullOracle = result }
                    if let previous = modeOracles[mode] { try require(label + "_complete_same_mode_exact", result.tokens == previous.tokens && result.finishReason == previous.finishReason) }
                    else { modeOracles[mode] = result }
                    try save()
                }
            }
            var summaries = [[String: Any]]()
            for group in orders + ["all"] {
                for mode in [stock,paged] {
                    let values = measured.filter { $0.mode == mode && (group == "all" || $0.order == group) }.map(\.result)
                    let decoded = values.reduce(0) { $0 + $1.statistics.decodedTokenCount }, seconds = values.reduce(0.0) { $0 + $1.decodeSeconds }
                    let service = values.reduce(0.0) { $0 + ($1.phases?.decodeServiceSeconds ?? 0) }
                    summaries.append(["order": group, "mode": mode, "trials": values.count, "actual_decoded_tokens": decoded,
                        "decode_seconds_import_checks_cleanup_included": seconds, "aggregate_decode_tokens_per_second": Double(decoded) / seconds,
                        "aggregate_decode_service_tokens_per_second": Double(decoded) / service, "decode_tokens_per_second_by_trial": values.compactMap(\.decodeTokensPerSecond),
                        "time_to_first_token_seconds_by_trial": values.map(\.timeToFirstTokenSeconds),
                        "prefill_ready_seconds_by_trial": values.compactMap { $0.phases?.prefill.totalSeconds },
                        "actual_prefill_tokens_per_trial": suffix, "prefill_target_tokens_per_second_by_trial": values.compactMap { $0.phases?.prefill.targetTokensPerSecond },
                        "prefill_ready_tokens_per_second_by_trial": values.compactMap { $0.phases?.prefill.readyTokensPerSecond },
                        "paged_import_seconds_by_trial": values.compactMap { $0.phases?.pagedKVImportSeconds }])
                }
            }
            report["warm_performance_summary"] = summaries
            try baseline.clearPrefixCache(); try candidate.clearPrefixCache(); try MX.synchronize()
            let finalPools = try context.layerStatistics, finalBudget = model.stateBudget.statistics
            let finalCaches = try cacheSnapshot(), finalClaims = context.pageAdmissionStatistics
            try require("final_cache_pages_and_claims_released", finalCaches.bytes == 0 && finalClaims.activeClaims == 0 && finalClaims.claimedPages == 0 &&
                finalPools.values.allSatisfy { $0.livePages == 0 && $0.inFlightOperations == 0 && $0.failedOperations == 0 })
            try require("final_only_twelve_fixed_arenas_admitted", finalBudget.requestBytes == 0 && finalBudget.cacheBytes == 0 &&
                finalBudget.workspaceBytes == arenaBytes && finalBudget.totalBytes == arenaBytes && finalBudget.currentLeases == 12)
            report["final_cleanup"] = ["pools": try object(finalPools), "budget": try object(finalBudget), "caches": finalCaches.raw,
                "page_claims": try object(finalClaims), "fixed_context_retained": true]
            try save(complete: true)
        } catch { report["error"] = String(describing: error); try? save(); throw error }
    }
}
