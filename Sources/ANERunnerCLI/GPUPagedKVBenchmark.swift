import ANERunnerCore
import ANERunnerGPU
import CMLX
import Foundation

extension RunnerCLI {
    /// Real generator requests without state observers. A fixed physical arena
    /// remains allocated in both modes; the paged import stays in decode timing.
    static func benchmarkGPUPagedKV(_ args: Arguments) throws {
        try args.validate(["--model-dir", "--tokens-file", "--max-tokens", "--context",
            "--order", "--warmup", "--library", "--maximum-pages-per-layer", "--output"])
        let output = try args.require("--output")
        guard !FileManager.default.fileExists(atPath: output),
              let maximum = Int(args["--max-tokens"] ?? "128"), (16...512).contains(maximum),
              let contextLimit = Int(args["--context"] ?? "16384"), (1...131_072).contains(contextLimit),
              let maximumPages = Int(args["--maximum-pages-per-layer"] ?? "512"),
              (2...4096).contains(maximumPages) else {
            throw CLIError.usage("Paged KV benchmark needs new --output, --max-tokens 16...512, --context 1...131072 and --maximum-pages-per-layer 2...4096")
        }
        let orderName = args["--order"] ?? "abba", warmup = args["--warmup"] ?? "true"
        guard ["abba", "baab"].contains(orderName), ["true", "false"].contains(warmup) else {
            throw CLIError.usage("Use --order abba|baab and --warmup true|false")
        }
        let directory = URL(fileURLWithPath: try args.require("--model-dir")).standardizedFileURL.resolvingSymlinksInPath()
        let inputURL = URL(fileURLWithPath: try args.require("--tokens-file")).standardizedFileURL.resolvingSymlinksInPath()
        let input = try Data(contentsOf: inputURL)
        let tokens = try JSONDecoder().decode([Int32].self, from: input)
        guard tokens.count >= 10_000, tokens.count <= contextLimit - maximum,
              (tokens.count + maximum - 1 + 31) / 32 < maximumPages else {
            throw CLIError.usage("Use a 10k+ prompt within context and leave one physical page for immutable-tail COW")
        }
        let library = URL(fileURLWithPath: try args.require("--library")).standardizedFileURL.resolvingSymlinksInPath()
        let stock = "stock_capacity256", paged = "physical_paged32"
        let order = orderName == "abba" ? [stock,paged,paged,stock] : [paged,stock,stock,paged]
        var trials = [[String: Any]](), warmups = [[String: Any]](), checks = [String: Bool]()
        var report: [String: Any] = [
            "schema": "qwen-paged-kv-benchmark-v1", "complete": false, "passed": false,
            "configuration": ["prompt_tokens": tokens.count, "max_tokens": maximum,
                "context": contextLimit, "maximum_pages_per_layer": maximumPages,
                "prefill_chunk": 416, "prefill_evaluate_every_layers": 4, "mtp_depth": 0,
                "order": orderName, "warmup": warmup == "true", "warmup_output_tokens": 16,
                "state_observers": false, "prefix_cache": false],
            "notes": [
                "A=stock SDPA with capacity256; B=physical shared-page SDPA with paged32. Requests are sequential with independent states on one model.",
                "The same physical context remains allocated during both modes. DSO loading and fixed-arena construction are outside request timing; B's compact-to-page import is included in decodeSeconds and reported separately.",
                "Each request performs a complete cold prefill. Prefill target/ready, decode rounds and decode service are reported separately; the first output token belongs to prefill.",
                "No state observers, tensor hashes or per-token pool statistics are collected. Native statistics and synchronization are sampled outside request timing.",
                "Fixed arena allocations and admission reservations are separate from MLX allocator observations. This benchmark does not measure RSS savings or physical DRAM bandwidth.",
                "Passed means output, operation-accounting and resource-lifetime checks passed. Performance acceptance requires comparing measured rates, including order drift."]]
        func object<T: Encodable>(_ value: T) throws -> Any {
            try JSONSerialization.jsonObject(with: JSONEncoder().encode(value))
        }
        func save(complete: Bool = false) throws {
            report["trials"] = trials; report["warmups"] = warmups; report["checks"] = checks
            report["complete"] = complete
            report["passed"] = complete && trials.count == 4 && !checks.isEmpty && checks.values.allSatisfy { $0 }
            try emit(report, to: output)
        }
        func require(_ label: String, _ passed: Bool) throws {
            checks[label] = passed
            guard passed else { throw CLIError.usage("Paged KV benchmark check failed: \(label)") }
        }
        try save()
        do {
            let executable = Bundle.main.executableURL ?? URL(fileURLWithPath: CommandLine.arguments[0])
            report["provenance"] = ["command": CommandLine.arguments, "captured_utc": Date().ISO8601Format(),
                "os": ProcessInfo.processInfo.operatingSystemVersionString,
                "executable_sha256": try MoETilingBytes.hash(executable),
                "pool_library_path": library.path, "pool_library_sha256": try MoETilingBytes.hash(library),
                "input_path": inputURL.path, "input_sha256": MoETilingBytes.digest(input),
                "model_directory": directory.path,
                "config_sha256": try MoETilingBytes.hash(directory.appendingPathComponent("config.json")),
                "tokenizer_sha256": try MoETilingBytes.hash(directory.appendingPathComponent("tokenizer.json")),
                "model_cache_identity": try QwenPrefixCacheIdentity.fingerprint(modelDirectory: directory),
                "base_mlx": try MoEGateUpProbeSupport.baseMLX()]
            var oldCache = 0
            try MX.check(mlx_set_cache_limit(&oldCache, 256 * 1024 * 1024), "Paged KV benchmark allocator cache")
            defer { var ignored = 0; _ = mlx_set_cache_limit(&ignored, oldCache) }
            let tokenizer = try QwenTokenizer(modelDirectory: directory)
            let model = try QwenModel(modelDirectory: directory, reservedOutputIDs: tokenizer.reservedOutputTokenIDs) { count, total in
                if count % 8 == 0 || count == total {
                    FileHandle.standardError.write(Data("Paged KV benchmark loaded \(count)/\(total) layers\n".utf8))
                }
            }
            defer { try? MX.synchronize() }
            try require("full_model_without_mtp", model.layerCount == 48 &&
                !model.weights.ledger.contains { $0.name.contains(".mtp.") })
            report["experimental_decode_async_every_layers"] = model.experimentalDecodeAsyncEveryLayers
            let setupStart = DispatchTime.now().uptimeNanoseconds
            let context = try model.makePagedKVContext(libraryPath: library.path, maximumPagesPerLayer: maximumPages)
            try MX.synchronize()
            report["fixed_context_setup_seconds_excluded"] = Double(DispatchTime.now().uptimeNanoseconds - setupStart) * 1e-9
            let initialPools = try context.layerStatistics
            try require("twelve_attention_arenas", initialPools.count == 12)
            let arenaBytes = context.reservedArenaBytes
            report["fixed_arenas_present_in_both_modes"] = [
                "reserved_bytes": arenaBytes,
                "logical_bytes": initialPools.values.reduce(UInt64(0)) { $0 + $1.arenaLogicalBytes },
                "allocated_bytes": initialPools.values.reduce(UInt64(0)) { $0 + $1.arenaAllocatedBytes },
                "layers": try object(initialPools)]
            let baseline = try QwenGenerator(model: model)
            let candidate = try QwenGenerator(model: model, pagedKVContext: context)
            var measured = [(mode: String, result: QwenGenerationResult)]()

            func run(_ mode: String, outputCount: Int, label: String,
                     warmupRun: Bool) throws -> QwenGenerationResult {
                try MX.synchronize()
                let beforePools = try context.layerStatistics
                let beforeMemory = try MX.memory(), beforeBudget = model.stateBudget.statistics
                try require(label + "_starts_without_request_leases", beforeBudget.requestBytes == 0 &&
                    beforeBudget.cacheBytes == 0 && beforeBudget.totalBytes == arenaBytes && beforeBudget.currentLeases == 12)
                FileHandle.standardError.write(Data("Paged KV benchmark \(label): \(mode) O\(outputCount)\n".utf8))
                let result = try (mode == stock ? baseline : candidate).generate(.init(tokens: tokens,
                    maxTokens: outputCount, contextLimit: contextLimit, prefillChunk: 416, mtpDepth: 0,
                    prefillEvaluateEveryLayers: 4, prefixCacheMaxTokens: 0,
                    kvAppendMode: mode == stock ? .capacity256 : .reference))
                // Native completion pins and lease release are checked after the
                // generator's own wall timings have already been recorded.
                try MX.synchronize()
                let afterPools = try context.layerStatistics
                let afterBudget = model.stateBudget.statistics
                let steps = result.statistics.decodedTokenCount
                let isPaged = mode == paged
                var deltas = [[String: Any]]()
                for layer in beforePools.keys.sorted() {
                    guard let before = beforePools[layer], let after = afterPools[layer] else {
                        throw CLIError.usage("Physical arena layer set changed")
                    }
                    try require(label + "_layer\(layer)_monotonic_counters",
                        after.encodedWrites >= before.encodedWrites && after.encodedReads >= before.encodedReads &&
                        after.encodedMaterializations >= before.encodedMaterializations &&
                        after.writtenRowBytes >= before.writtenRowBytes && after.copiedTailBytes >= before.copiedTailBytes &&
                        after.materializedBytes >= before.materializedBytes)
                    let writes = after.encodedWrites - before.encodedWrites
                    let reads = after.encodedReads - before.encodedReads
                    let exports = after.encodedMaterializations - before.encodedMaterializations
                    let rowBytes = after.writtenRowBytes - before.writtenRowBytes
                    let tailBytes = after.copiedTailBytes - before.copiedTailBytes
                    let expectedRows = isPaged && steps > 0 ? (tokens.count + steps) * 2_048 : 0
                    let expectedTail = isPaged ? (0..<steps).reduce(0) { $0 + ((tokens.count + $1) % 32) * 2_048 } : 0
                    let expectedWrites = isPaged && steps > 0 ? UInt64(steps + 1) : 0
                    try require(label + "_layer\(layer)_direct_ops", writes == expectedWrites &&
                        reads == UInt64(isPaged ? steps : 0) && exports == 0 &&
                        after.materializedBytes == before.materializedBytes &&
                        rowBytes == UInt64(expectedRows) && tailBytes == UInt64(expectedTail))
                    try require(label + "_layer\(layer)_fixed_arena_drained", after.livePages == 0 &&
                        after.freePages == after.physicalPages && after.physicalPages == UInt64(maximumPages) &&
                        after.inFlightOperations == 0 && after.failedOperations == 0 &&
                        after.completedOperations == after.encodedWrites + after.encodedReads + after.encodedMaterializations &&
                        after.arenaAllocatedBytes == before.arenaAllocatedBytes &&
                        after.keyBufferIdentity == before.keyBufferIdentity && after.valueBufferIdentity == before.valueBufferIdentity)
                    if !isPaged { try require(label + "_layer\(layer)_baseline_no_pool_activity", before == after) }
                    deltas.append(["layer": layer, "writes": writes, "reads": reads, "materializations": exports,
                        "written_row_bytes": rowBytes, "copied_tail_bytes": tailBytes,
                        "materialized_bytes": after.materializedBytes - before.materializedBytes])
                }
                try require(label + "_usable_decode", result.tokens.count >= 2 && result.tokens.count <= outputCount &&
                    steps == result.tokens.count - 1 && result.statistics.decodeRounds == steps &&
                    result.statistics.finalStateOffset == tokens.count + steps && result.statistics.mtpDepth == 0 &&
                    result.decodeSeconds.isFinite && result.decodeSeconds > 0)
                let endedByEOS = result.tokens.last.map { baseline.eosTokenIDs.contains($0) } ?? false
                try require(label + "_finish_contract", result.finishReason == .eos ? endedByEOS :
                    (result.finishReason == .length && result.tokens.count == outputCount && !endedByEOS))
                guard let phases = result.phases else { throw CLIError.usage("Missing generation phase statistics") }
                try require(label + "_cold_prefill", phases.prefill.cachedTokenCount == 0 &&
                    phases.prefill.computedTokenCount == tokens.count && phases.prefill.actualForwardTokenCount == tokens.count &&
                    phases.prefill.targetSeconds.isFinite && phases.prefill.targetSeconds > 0 &&
                    phases.prefill.totalSeconds.isFinite && phases.prefill.totalSeconds > 0)
                if isPaged {
                    let importSeconds = phases.pagedKVImportSeconds ?? -1
                    try require(label + "_mode_accounting", phases.kvAppendMode == "paged32" &&
                        phases.pagedKVTokenSteps == steps && importSeconds.isFinite && importSeconds > 0 &&
                        importSeconds <= result.decodeSeconds && phases.kvCapacityTokenSteps == 0 &&
                        phases.kvCapacityWorkspaceFallbacks == 0)
                } else {
                    try require(label + "_mode_accounting", phases.kvAppendMode == "capacity256" &&
                        phases.kvCapacityWorkspaceFallbacks == 0 && phases.kvCapacityTokenSteps == steps &&
                        (phases.kvCapacityWorkspacePeakBytes ?? 0) > 0 &&
                        phases.pagedKVImportSeconds == nil && phases.pagedKVTokenSteps == nil)
                }
                try require(label + "_only_arena_leases_remain", afterBudget.requestBytes == 0 &&
                    afterBudget.cacheBytes == 0 && afterBudget.workspaceBytes == arenaBytes &&
                    afterBudget.totalBytes == arenaBytes && afterBudget.currentLeases == 12)
                let row: [String: Any] = ["label": label, "mode": mode,
                    "generated_token_ids": result.tokens, "result": try object(result),
                    "decode_tokens_per_second_import_included": result.decodeTokensPerSecond ?? 0,
                    "decode_service_tokens_per_second": result.decodeServiceTokensPerSecond ?? 0,
                    "prefill_target_tokens_per_second": phases.prefill.targetTokensPerSecond ?? 0,
                    "prefill_ready_tokens_per_second": phases.prefill.readyTokensPerSecond ?? 0,
                    "pool_operation_deltas": deltas, "pool_statistics_after": try object(afterPools),
                    "mlx_memory_before": beforeMemory, "mlx_memory_after": try MX.memory(),
                    "state_budget_before": try object(beforeBudget), "state_budget_after": try object(afterBudget)]
                if warmupRun { warmups.append(row) } else { trials.append(row); measured.append((mode,result)) }
                try save()
                return result
            }

            if warmup == "true" {
                let first = try run(stock, outputCount: 16, label: "warmup_a", warmupRun: true)
                let second = try run(paged, outputCount: 16, label: "warmup_b", warmupRun: true)
                try require("warmup_output_exact", first.tokens == second.tokens && first.finishReason == second.finishReason)
            }
            var oracle: QwenGenerationResult?
            for (index, mode) in order.enumerated() {
                let result = try run(mode, outputCount: maximum, label: "trial\(index)", warmupRun: false)
                if let oracle {
                    try require("trial\(index)_output_exact", result.tokens == oracle.tokens && result.finishReason == oracle.finishReason)
                } else { oracle = result; try require("trial\(index)_output_exact", true) }
                try save()
            }
            var summaries = [[String: Any]]()
            for mode in [stock,paged] {
                let values = measured.filter { $0.mode == mode }.map(\.result)
                let totalDecoded = values.reduce(0) { $0 + $1.statistics.decodedTokenCount }
                let totalDecodeSeconds = values.reduce(0.0) { $0 + $1.decodeSeconds }
                let rates = values.compactMap(\.decodeTokensPerSecond)
                let low = rates.min() ?? 0, high = rates.max() ?? 0
                summaries.append(["mode": mode, "trials": values.count,
                    "total_decoded_tokens": totalDecoded, "total_decode_seconds_import_included": totalDecodeSeconds,
                    "aggregate_decode_tokens_per_second_import_included": totalDecodeSeconds > 0 ? Double(totalDecoded) / totalDecodeSeconds : 0,
                    "decode_tokens_per_second_by_trial": rates,
                    "decode_rate_range_percent_of_minimum": low > 0 ? (high / low - 1) * 100 : 0,
                    "prefill_target_tokens_per_second_by_trial": values.compactMap { $0.phases?.prefill.targetTokensPerSecond },
                    "prefill_ready_tokens_per_second_by_trial": values.compactMap { $0.phases?.prefill.readyTokensPerSecond },
                    "paged_import_seconds_by_trial": values.compactMap { $0.phases?.pagedKVImportSeconds }])
            }
            report["performance_summary"] = summaries
            report["final_pool_statistics"] = try object(context.layerStatistics)
            report["final_state_budget_context_retained"] = try object(model.stateBudget.statistics)
            try save(complete: true)
        } catch {
            report["error"] = String(describing: error)
            try? save()
            throw error
        }
    }
}
