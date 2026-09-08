import ANERunnerCore
import ANERunnerGPU
import CMLX
import Foundation

extension RunnerCLI {
    /// Uses the actual generator admission/permit/evaluation path. No state
    /// observer or per-token readback is installed during timing trials.
    static func benchmarkGPUKVCapacity(_ args: Arguments) throws {
        try args.validate(["--model-dir", "--tokens-file", "--max-tokens", "--context",
            "--order", "--warmup", "--output"])
        let output = try args.require("--output")
        guard !FileManager.default.fileExists(atPath: output) else {
            throw CLIError.usage("KV benchmark requires a new output path")
        }
        guard let maxTokens = Int(args["--max-tokens"] ?? "16"), (2...4096).contains(maxTokens),
              let context = Int(args["--context"] ?? "16384"), (1...262144).contains(context) else {
            throw CLIError.usage("Invalid KV benchmark output/context limit")
        }
        let orderName = args["--order"] ?? "abba", warmupValue = args["--warmup"] ?? "true"
        guard ["abba", "baab"].contains(orderName), ["true", "false"].contains(warmupValue) else {
            throw CLIError.usage("Use --order abba|baab and --warmup true|false")
        }
        let directory = URL(fileURLWithPath: try args.require("--model-dir")).standardizedFileURL.resolvingSymlinksInPath()
        let inputURL = URL(fileURLWithPath: try args.require("--tokens-file"))
        let input = try Data(contentsOf: inputURL)
        let tokens = try JSONDecoder().decode([Int32].self, from: input)
        guard tokens.count >= 10_000, tokens.count <= context - maxTokens else {
            throw CLIError.usage("KV benchmark requires 10k+ prompt tokens within the context budget")
        }
        var trials = [[String: Any]](), warmups = [[String: Any]](), checks = [String: Bool]()
        var report: [String: Any] = [
            "schema": "qwen-kv-capacity-benchmark-v1", "complete": false, "passed": false,
            "configuration": ["prompt_tokens": tokens.count, "max_tokens": maxTokens,
                "context": context, "prefill_chunk": 416, "prefill_eval_layers": 4,
                "mtp_depth": 0, "order": orderName, "warmup": warmupValue == "true"],
            "notes": [
                "Four sequential generator requests use the same model and independent private states; prefix caching is disabled.",
                "A=reference concat, B=capacity256. Initialization/growth/fallback costs remain inside generator decode timing.",
                "Prefill and decode are separate. Generation phases and actual decoded counts are retained; first output is a prefill token.",
                "Memory snapshots are MLX/global allocator observations, not physical RSS or an isolated request peak.",
                "A successful run proves numerical/accounting parity for these requests; performance acceptance requires independent comparison and repeatability."]]
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
            checks[label] = passed; try save()
            guard passed else { throw CLIError.usage("KV benchmark check failed: \(label)") }
        }
        try save()
        do {
            let executable = Bundle.main.executableURL ?? URL(fileURLWithPath: CommandLine.arguments[0])
            report["provenance"] = ["executable_sha256": try MoETilingBytes.hash(executable),
                "input_sha256": MoETilingBytes.digest(input), "input_path": inputURL.path,
                "config_sha256": try MoETilingBytes.hash(directory.appendingPathComponent("config.json")),
                "tokenizer_sha256": try MoETilingBytes.hash(directory.appendingPathComponent("tokenizer.json")),
                "model_cache_identity": try QwenPrefixCacheIdentity.fingerprint(modelDirectory: directory)]
            var oldCache = 0
            try MX.check(mlx_set_cache_limit(&oldCache, 256 * 1024 * 1024), "KV benchmark allocation cache")
            defer { var ignored = 0; _ = mlx_set_cache_limit(&ignored, oldCache) }
            let tokenizer = try QwenTokenizer(modelDirectory: directory)
            let model = try QwenModel(modelDirectory: directory, reservedOutputIDs: tokenizer.reservedOutputTokenIDs) { count, total in
                if count % 8 == 0 || count == total {
                    FileHandle.standardError.write(Data("KV benchmark loaded \(count)/\(total) layers\n".utf8))
                }
            }
            defer { try? MX.synchronize() }
            let generator = try QwenGenerator(model: model)
            let reference: GPUAttention.KVAppendMode = .reference, capacity: GPUAttention.KVAppendMode = .capacity256
            let order = orderName == "abba" ? [reference, capacity, capacity, reference] : [capacity, reference, reference, capacity]
            func run(_ mode: GPUAttention.KVAppendMode, outputCount: Int, label: String) throws -> QwenGenerationResult {
                FileHandle.standardError.write(Data("KV benchmark \(label): \(mode.rawValue) O\(outputCount)\n".utf8))
                return try generator.generate(.init(tokens: tokens, maxTokens: outputCount, contextLimit: context,
                    prefillChunk: 416, mtpDepth: 0, prefillEvaluateEveryLayers: 4,
                    prefixCacheMaxTokens: 0, kvAppendMode: mode))
            }
            if warmupValue == "true" {
                let warmupTokens = min(16, maxTokens)
                var previous: QwenGenerationResult?
                for mode in [reference, capacity] {
                    let result = try run(mode, outputCount: warmupTokens, label: "warmup")
                    warmups.append(["kv_append_mode": mode.rawValue, "result": try object(result)])
                    try save()
                    if let previous {
                        try require("warmup_output_exact", previous.tokens == result.tokens && previous.finishReason == result.finishReason)
                    }
                    previous = result
                }
            }
            var oracleIDs: [Int32]?, oracleFinish: QwenGenerationFinishReason?
            for (index, mode) in order.enumerated() {
                let beforeMemory = try MX.memory(), beforeBudget = model.stateBudget.statistics
                let result = try run(mode, outputCount: maxTokens, label: "trial\(index)")
                let afterBudget = model.stateBudget.statistics
                trials.append(["index": index, "kv_append_mode": mode.rawValue,
                    "generated_token_ids": result.tokens, "result": try object(result),
                    "mlx_memory_before": beforeMemory, "mlx_memory_after": try MX.memory(),
                    "state_budget_before": try object(beforeBudget), "state_budget_after": try object(afterBudget)])
                if oracleIDs == nil { oracleIDs = result.tokens; oracleFinish = result.finishReason }
                try require("trial\(index)_output_exact", result.tokens == oracleIDs && result.finishReason == oracleFinish)
                try require("trial\(index)_usable_decode", result.tokens.count >= 2 &&
                    result.tokens.count <= maxTokens &&
                    result.statistics.decodedTokenCount == result.tokens.count - 1 &&
                    result.statistics.decodeRounds == result.statistics.decodedTokenCount &&
                    result.statistics.finalStateOffset == tokens.count + result.statistics.decodedTokenCount &&
                    result.statistics.mtpDepth == 0 && result.decodeSeconds.isFinite && result.decodeSeconds > 0)
                let endedByEOS = result.tokens.last.map { generator.eosTokenIDs.contains($0) } ?? false
                try require("trial\(index)_finish_contract", result.finishReason == .eos ? endedByEOS :
                    (result.finishReason == .length && result.tokens.count == maxTokens && !endedByEOS))
                let phases = result.phases
                try require("trial\(index)_cold_prefill", phases?.prefill.cachedTokenCount == 0 &&
                    phases?.prefill.computedTokenCount == tokens.count && phases?.prefill.actualForwardTokenCount == tokens.count)
                try require("trial\(index)_mode_accounting", phases?.kvAppendMode == mode.rawValue &&
                    phases?.kvCapacityWorkspaceFallbacks == 0 &&
                    phases?.kvCapacityTokenSteps == (mode == capacity ? result.statistics.decodedTokenCount : 0) &&
                    (mode == capacity ? (phases?.kvCapacityWorkspacePeakBytes ?? 0) > 0 :
                        phases?.kvCapacityWorkspacePeakBytes == 0))
                try require("trial\(index)_leases_released", afterBudget.currentLeases == 0 && afterBudget.totalBytes == 0)
            }
            try save(complete: true)
        } catch {
            report["error"] = String(describing: error)
            try? save()
            throw error
        }
    }
}
