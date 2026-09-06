import ANERunnerGPU
import CMLX
import CryptoKit
import Foundation

extension RunnerCLI {
    /// Seven fixed scenarios, one model, fresh state for every request. This
    /// records compatibility and request latency without layer diagnostics.
    static func probeGPUMTPRelease(_ args: Arguments) throws {
        try args.validate(["--model-dir", "--long-tokens-file", "--output", "--depth", "--verification", "--mtp-draft-history"])
        let draftHistoryTokens = try mtpDraftHistory(args)
        let output = try args.require("--output")
        guard !FileManager.default.fileExists(atPath: output) else {
            throw CLIError.usage("MTP release probe --output must be a new file")
        }
        var report: [String: Any] = [
            "schema": "qwen38-mtp-seven-scenarios-v1", "passed": false, "complete": false,
            "case_order": ["bilingual", "arithmetic", "strict_json", "function_code", "qsa_prefix_2051", "qsa_prefix_2053", "agent_11057"],
            "request_order": ["ar_before", "candidate_first", "candidate_second", "ar_after"],
            "notes": [
                "Only seven fixed scenarios; passing is not comprehensive release or quality certification.",
                "Each generate call creates fresh state on one shared model/generator; model weights and OS caches remain warm.",
                "QSA boundary inputs are literal token prefixes of real project text; they may end inside a chat message and are not complete conversations.",
                "No stage tracing or numerical readback is enabled. Memory queries and JSON serialization are outside request wall timing.",
                "wall_seconds includes lazy head preparation. result.totalSeconds excludes result.preparationSeconds; MTP head prompt-history computation is included in request execution.",
                "First-use/JIT/SSD cache effects remain visible; four sequential runs per case are a bounded comparison, not a statistical performance claim.",
                "Memory is MLX allocator bytes: active/cache are snapshots, peak is cumulative since process start, not per-request peak or physical process memory.",
                "EOS and output stopping belong to QwenGenerator. Terminal state offsets can differ when verification already consumed an accepted draft EOS.",
                "The assertions check token/finish compatibility and result bookkeeping; they do not execute or grade the generated code/JSON/task answer."
            ]
        ]
        var fatalError: Error?
        do {
            let directory = URL(fileURLWithPath: try args.require("--model-dir"), isDirectory: true)
                .standardizedFileURL.resolvingSymlinksInPath()
            let longURL = URL(fileURLWithPath: try args.require("--long-tokens-file")).standardizedFileURL
            guard let depth = Int(args["--depth"] ?? "1"), (1...2).contains(depth),
                  let verification = QwenMTPDecoder.Verification(rawValue: args["--verification"] ?? "batchedScalarLinear") else {
                throw CLIError.usage("Use --depth 1 or 2 and a supported --verification mode")
            }
            report["model_directory"] = directory.path
            report["depth"] = depth; report["verification"] = verification.rawValue
            report["draft_history_limit"] = draftHistoryTokens.map { $0 as Any } ?? NSNull()
            let data = try Data(contentsOf: longURL)
            let longTokens = try JSONDecoder().decode([Int32].self, from: data)
            guard longTokens.count == 11_057 else {
                throw CLIError.usage("This frozen probe requires the original 11057-token agent JSON array")
            }
            report["long_input"] = ["path": longURL.path, "token_count": longTokens.count,
                "sha256": SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()]
            let tokenizer = try QwenTokenizer(modelDirectory: directory)
            let configuration = try QwenConfiguration(modelDirectory: directory)
            let short: [(String, String)] = [
                ("bilingual", "先用一句中文解释太阳为什么发光，再用一句英文表达同一个意思。"),
                ("arithmetic", "计算 17 × 23，只输出十进制整数答案，不要解释。"),
                ("strict_json", "仅输出一个合法 JSON 对象，不要 Markdown 或解释。对象必须恰好包含 name 和 count 两个键，name 的值为字符串 sun，count 的值为整数 3。"),
                ("function_code", "Write a Python function named sum_even(values) that returns the sum of the even integers in values. Output only the function code, without Markdown fences or explanation.")
            ]
            var scenarios = try short.map { name, text in
                GPUMTPReleaseScenario(name: name,
                    tokens: try tokenizer.encode(tokenizer.renderChat(messages: [ChatMessage(role: "user", content: text)])),
                    maximumOutput: 64, source: ["kind": "frozen_user_prompt", "text": text, "thinking": false])
            }
            for count in [2051, 2053] {
                scenarios.append(.init(name: "qsa_prefix_\(count)", tokens: Array(longTokens.prefix(count)),
                    maximumOutput: 64, source: ["kind": "truncated_real_project_token_prefix", "prefix_token_count": count]))
            }
            scenarios.append(.init(name: "agent_11057", tokens: longTokens, maximumOutput: 128,
                                   source: ["kind": "complete_long_token_file"]))
            let context = 16_384
            for item in scenarios {
                try QwenGenerationRequest(tokens: item.tokens, maxTokens: item.maximumOutput,
                    contextLimit: context, mtpDepth: depth, verification: verification,
                    draftHistoryTokens: draftHistoryTokens).validate(configuration: configuration)
            }
            report["context_limit"] = context; report["prefill_chunk"] = 416
            report["physical_machine_memory_bytes"] = ProcessInfo.processInfo.physicalMemory
            report["fused_attention_prefill_enabled"] = GPUAttention.fusedPrefillEnabled
            var previousCache = 0
            try MX.check(mlx_set_cache_limit(&previousCache, 256 * 1024 * 1024), "release probe allocation cache")
            defer { var ignored = 0; _ = mlx_set_cache_limit(&ignored, previousCache) }
            let constructionStart = DispatchTime.now().uptimeNanoseconds
            let model = try QwenModel(modelDirectory: directory) { count, total in
                if count % 8 == 0 || count == total {
                    FileHandle.standardError.write(Data("MTP release probe: loaded \(count)/\(total)\n".utf8))
                }
            }
            report["model_construction_seconds"] = Double(DispatchTime.now().uptimeNanoseconds - constructionStart) * 1e-9
            report["loaded_memory"] = try MX.memory()
            let generator = try QwenGenerator(model: model)
            var cases = [[String: Any]]()
            var allPassed = true
            let order = ["ar_before", "candidate_first", "candidate_second", "ar_after"]
            for item in scenarios {
                var rows = [[String: Any]](), outputs = [QwenGenerationResult?](), bookkeeping = [Bool]()
                for (index, name) in order.enumerated() {
                    let candidate = index == 1 || index == 2
                    let request = QwenGenerationRequest(tokens: item.tokens, maxTokens: item.maximumOutput,
                        contextLimit: context, mtpDepth: candidate ? depth : 0, verification: verification,
                        draftHistoryTokens: draftHistoryTokens)
                    var row: [String: Any] = ["name": name, "candidate": candidate, "succeeded": false]
                    var generated: QwenGenerationResult?, callbackIDs = [Int32]()
                    var generationError: Error?, wallSeconds: Double?
                    do {
                        row["memory_before"] = try MX.memory()
                        let start = DispatchTime.now().uptimeNanoseconds
                        do { generated = try generator.generate(request) { callbackIDs.append($0) } }
                        catch { generationError = error }
                        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) * 1e-9
                        wallSeconds = elapsed
                        row["wall_seconds"] = elapsed
                        row["callback_token_ids"] = callbackIDs
                        if let generationError { row["generation_error"] = String(describing: generationError) }
                        if let result = generated {
                            row["result"] = try JSONSerialization.jsonObject(with: JSONEncoder().encode(result))
                            row["text"] = try tokenizer.decode(result.tokens.filter { !generator.eosTokenIDs.contains($0) })
                            row["end_to_end_reported_seconds"] = result.preparationSeconds + result.totalSeconds
                            row["decode_tokens_per_second"] = result.decodeTokensPerSecond.map { $0 as Any } ?? NSNull()
                            let drafted = result.statistics.mtp?.draftedTokens ?? 0
                            row["mtp_exercised"] = drafted > 0
                            if drafted > 0 {
                                row["acceptance_rate"] = Double(result.statistics.mtp?.acceptedDraftTokens ?? 0) / Double(drafted)
                            } else { row["acceptance_rate"] = NSNull() }
                        }
                        // A memory-report failure is retained without dropping a
                        // simultaneous generation error or its delivered tokens.
                        row["memory_after"] = try MX.memory()
                        row["succeeded"] = generated != nil && generationError == nil
                    } catch {
                        row["error"] = String(describing: error)
                        row["callback_token_ids"] = callbackIDs
                        if let generationError { row["generation_error"] = String(describing: generationError) }
                        if let wallSeconds { row["wall_seconds"] = wallSeconds }
                    }
                    let valid = generated.map { result in
                        !result.tokens.isEmpty && result.tokens.count <= item.maximumOutput &&
                        result.tokens == callbackIDs && result.statistics.generatedTokenCount == result.tokens.count &&
                        result.statistics.promptTokenCount == item.tokens.count &&
                        result.statistics.mtpDepth == (candidate ? depth : 0) &&
                        (!candidate || result.statistics.mtpVerification == verification.rawValue) &&
                        (result.finishReason == .length
                            ? result.tokens.count == item.maximumOutput && !result.tokens.contains(where: generator.eosTokenIDs.contains)
                            : result.tokens.last.map(generator.eosTokenIDs.contains) == true &&
                              !result.tokens.dropLast().contains(where: generator.eosTokenIDs.contains))
                    } ?? false
                    row["bookkeeping_passed"] = valid
                    rows.append(row); outputs.append(generated)
                    bookkeeping.append(valid && row["succeeded"] as? Bool == true)
                    FileHandle.standardError.write(Data("MTP release probe: \(item.name) / \(name): \(row["succeeded"] as? Bool == true ? "completed" : "failed")\n".utf8))
                }
                let exact = outputs.first.flatMap { $0 }.map { baseline in
                    outputs.allSatisfy { other in
                        other.map { $0.tokens == baseline.tokens && $0.finishReason == baseline.finishReason } ?? false
                    }
                } ?? false
                let mtpExercised = [1, 2].allSatisfy { index in
                    (outputs[index]?.statistics.mtp?.draftedTokens ?? 0) > 0
                }
                let passed = exact && mtpExercised && bookkeeping.allSatisfy { $0 }
                allPassed = allPassed && passed
                cases.append(["name": item.name, "source": item.source, "prompt_token_ids": item.tokens,
                              "max_output_tokens": item.maximumOutput, "passed": passed,
                              "all_token_ids_and_finish_reasons_exact": exact,
                              "both_candidate_runs_exercised_mtp": mtpExercised, "runs": rows])
                report["cases"] = cases
                // Preserve completed cases if a later request is interrupted.
                try emit(report, to: output)
            }
            report["loaded_source_weight_bytes_at_end"] = model.weights.cachedSourceBytes
            report["final_memory"] = try MX.memory()
            report["complete"] = cases.count == 7; report["passed"] = allPassed && cases.count == 7
        } catch {
            fatalError = error
            report["fatal_error"] = String(describing: error)
        }
        try emit(report, to: output)
        if let fatalError { throw fatalError }
        guard report["passed"] as? Bool == true else {
            throw CLIError.usage("MTP release probe failed; see \(output)")
        }
    }
}

private struct GPUMTPReleaseScenario {
    let name: String
    let tokens: [Int32]
    let maximumOutput: Int
    let source: [String: Any]
}
