import ANERunnerGPU
import CMLX
import CryptoKit
import Foundation

extension RunnerCLI {
    /// Same loaded model, actual producer/consumer handoff, and a per-request
    /// attention policy confined to trunk prefill. No extra forward is run.
    static func probeGPUPrefillAttention(_ args: Arguments) throws {
        try args.validate(["--model-dir", "--tokens-file", "--output", "--golden-report",
                           "--max-tokens", "--order", "--mtp-depth", "--mtp-draft-history"])
        let output = try args.require("--output")
        guard !FileManager.default.fileExists(atPath: output),
              let maximum = Int(args["--max-tokens"] ?? "128"), (1...256).contains(maximum),
              let depth = Int(args["--mtp-depth"] ?? "0"), [0, 2].contains(depth) else {
            throw CLIError.usage("Prefill attention probe requires new --output, --max-tokens 1...256 and --mtp-depth 0|2")
        }
        let names = (args["--order"] ?? "reference,fusedQSA,reference,fusedQSA,fusedQSA,reference")
            .split(separator: ",", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        let modes = names.compactMap(GPUAttention.PrefillMode.init(rawValue:))
        guard (2...10).contains(names.count), modes.count == names.count,
              names.first == "reference", names.contains("fusedQSA") else {
            throw CLIError.usage("--order must contain 2...10 reference|fusedQSA entries, begin with reference, and include fusedQSA")
        }
        let history: Int?
        if depth == 0 {
            guard args["--mtp-draft-history"] == nil else {
                throw CLIError.usage("--mtp-draft-history requires --mtp-depth 2")
            }
            history = nil
        } else {
            if args["--mtp-draft-history"] == nil { history = 1024 }
            else { history = try mtpDraftHistory(args) }
        }
        var trials = [[String: Any]]()
        var report: [String: Any] = [
            "schema": "qwen38-prefill-attention-probe-v1", "complete": false, "passed": false,
            "correctnessPassed": false, "full_model_instances": 1, "generator_instances": 2,
            "order": names, "max_tokens": maximum, "mtp_depth": depth,
            "mtp_verification": depth == 2 ? "batchedScalarLinear" as Any : NSNull(),
            "mtp_draft_history": history.map { $0 as Any } ?? NSNull(),
            "clock": "mach_absolute_time_nanoseconds", "logit_finiteness_checked": false,
            "notes": [
                "Every trial calls producer.prefill(request), then consumer.decode(ready) on the same model and executor.",
                "Only trunk prefill attention varies. Decode policy and the native MTP head remain unchanged.",
                "Complete IDs and finish reason are checked against the first same-process reference and the complete-budget AR golden.",
                "Public APIs expose IDs and statistics, not logits/hidden tensors. Finite-public-metrics checks do not establish internal tensor finiteness.",
                "First-use compilation, SSD caching and MTP head preparation may affect early trials. These are bounded screening observations, not a stable speedup claim.",
                "Comparison failure completes the report with passed=false. Runtime errors or nonfinite public metrics are fatal and produce nonzero exit."
            ]]
        func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
        func elapsed(_ start: UInt64) -> Double { Double(now() - start) * 1e-9 }
        func hash(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
        func object<T: Encodable>(_ value: T) throws -> Any {
            try JSONSerialization.jsonObject(with: JSONEncoder().encode(value))
        }
        func difference(_ actual: [Int32], _ expected: [Int32]) -> Any {
            let common = min(actual.count, expected.count)
            let index = (0..<common).first { actual[$0] != expected[$0] }
                ?? (actual.count == expected.count ? nil : common)
            guard let index else { return NSNull() }
            return ["index": index, "actual": index < actual.count ? actual[index] as Any : NSNull(),
                    "expected": index < expected.count ? expected[index] as Any : NSNull(),
                    "actual_count": actual.count, "expected_count": expected.count] as [String: Any]
        }
        func write(_ complete: Bool = false) throws {
            let correct = complete && trials.count == modes.count && trials.allSatisfy { $0["correctnessPassed"] as? Bool == true }
            report["complete"] = complete; report["passed"] = correct; report["correctnessPassed"] = correct
            report["trials"] = trials
            try emit(report, to: output)
        }
        try write()
        do {
            let directory = URL(fileURLWithPath: try args.require("--model-dir"))
                .standardizedFileURL.resolvingSymlinksInPath()
            let inputURL = URL(fileURLWithPath: try args.require("--tokens-file")).standardizedFileURL
            let inputData = try Data(contentsOf: inputURL)
            let tokens = try JSONDecoder().decode([Int32].self, from: inputData)
            guard (10_000...12_000).contains(tokens.count) else {
                throw CLIError.usage("Prefill attention screen expects a real 10k...12k token prompt")
            }
            let configuration = try QwenConfiguration(modelDirectory: directory)
            let tokenizer = try QwenTokenizer(modelDirectory: directory)
            let context = max(16_384, tokens.count + maximum)
            let requests = modes.map { mode in
                QwenGenerationRequest(tokens: tokens, maxTokens: maximum, contextLimit: context,
                    prefillChunk: 416, mtpDepth: depth, verification: .batchedScalarLinear,
                    draftHistoryTokens: history, prefillAttention: mode)
            }
            for request in requests { try request.validate(configuration: configuration) }
            let goldenURL = URL(fileURLWithPath: try args.require("--golden-report")).standardizedFileURL
            let goldenData = try Data(contentsOf: goldenURL)
            struct Golden: Decodable {
                struct Trial: Decodable {
                    let generated_token_ids, prompt_tokens: [Int32]
                    let mtp_depth: Int?
                    let mtp_enabled: Bool?
                    let finish_reason: String
                }
                let max_tokens: Int
                let mtp_enabled: Bool?
                let trials: [Trial]
            }
            let goldenReport = try JSONDecoder().decode(Golden.self, from: goldenData)
            guard let golden = goldenReport.trials.first, !golden.generated_token_ids.isEmpty,
                  golden.prompt_tokens == tokens, goldenReport.max_tokens == maximum,
                  (golden.mtp_depth == 0 || golden.mtp_enabled == false || goldenReport.mtp_enabled == false),
                  golden.mtp_depth == nil || golden.mtp_depth == 0,
                  golden.mtp_enabled != true, goldenReport.mtp_enabled != true,
                  ["eos", "length"].contains(golden.finish_reason) else {
                throw CLIError.usage("Golden must contain matching complete prompt/budget and an AR trial; no prefix truncation is used")
            }
            let expected = golden.generated_token_ids, goldenFinish = golden.finish_reason
            let executable = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL.resolvingSymlinksInPath()
            report["provenance"] = ["command": CommandLine.arguments, "executable": executable.path,
                "executable_sha256": hash(try Data(contentsOf: executable)),
                "config_sha256": hash(try Data(contentsOf: directory.appendingPathComponent("config.json")))]
            report["model_directory"] = directory.path
            report["input"] = ["path": inputURL.path, "sha256": hash(inputData), "token_count": tokens.count, "token_ids": tokens]
            report["golden"] = ["path": goldenURL.path, "sha256": hash(goldenData),
                "token_ids": expected, "finish_reason": goldenFinish, "max_tokens": maximum]
            report["configuration"] = ["context_limit": context, "prefill_chunk": 416,
                "prefill_evaluate_every_layers": 4, "verification_evaluate_every_layers": 4,
                "decode_mode": "reference", "prefill_accumulation": "reference",
                "scope": "trunk_prefill_only"]
            report["environment"] = Dictionary(uniqueKeysWithValues:
                ["ANERUNNER_FUSED_PREFILL", "ANERUNNER_BLOCKED_GDN", "ANERUNNER_PREFILL_EVAL_LAYERS"]
                    .map { ($0, ProcessInfo.processInfo.environment[$0] ?? "unset") })
            try write()
            var previousCache = 0
            try MX.check(mlx_set_cache_limit(&previousCache, 256 * 1024 * 1024), "prefill attention probe cache limit")
            defer { var ignored = 0; _ = mlx_set_cache_limit(&ignored, previousCache) }
            let loadStart = now()
            let model = try QwenModel(modelDirectory: directory) { current, total in
                if current % 8 == 0 || current == total {
                    FileHandle.standardError.write(Data("Prefill attention probe: loaded layer \(current)/\(total)\n".utf8))
                }
            }
            report["model_load_seconds"] = elapsed(loadStart)
            report["loaded_memory"] = try MX.memory()
            let producer = try QwenGenerator(model: model), consumer = try QwenGenerator(model: model)
            var reference: QwenGenerationResult?
            for (index, request) in requests.enumerated() {
                var trial: [String: Any] = ["index": index, "prefill_attention": names[index],
                    "correctnessPassed": false, "complete": false]
                var stage = "prefill", callbacks = [Int32]()
                do {
                    FileHandle.standardError.write(Data("Prefill attention probe: trial \(index) \(names[index])\n".utf8))
                    let prefillStart = now()
                    let ready = try producer.prefill(request)
                    let prefillWall = elapsed(prefillStart)
                    defer { ready.discard() }
                    trial["prefill"] = try object(ready.statistics)
                    trial["prefill_wall_seconds"] = prefillWall
                    trial["preparation_seconds"] = ready.preparationSeconds
                    trial["first_token"] = ready.firstToken
                    trial["ready_at_return"] = ready.isReady
                    stage = "decode"
                    let decodeStart = now()
                    let result = try consumer.decode(ready) { callbacks.append($0) }
                    let decodeWall = elapsed(decodeStart)
                    guard let phases = result.phases else { throw CLIError.usage("Missing phase statistics") }
                    let finite = [prefillWall, decodeWall, result.preparationSeconds,
                        result.timeToFirstTokenSeconds, result.decodeSeconds, result.totalSeconds,
                        phases.prefill.targetSeconds, phases.prefill.draftHistorySeconds, phases.prefill.totalSeconds,
                        phases.prefill.ssdWaitSeconds, phases.handoffWaitSeconds, phases.handoffConsumeSeconds,
                        phases.decodeServiceSeconds, phases.decodeSSDWaitSeconds, result.statistics.callbackSeconds]
                        .allSatisfy { $0.isFinite && $0 >= 0 }
                    trial["finite_public_metrics"] = finite
                    guard finite else { throw CLIError.usage("Nonfinite or negative public generation metrics") }
                    // JSONEncoder also rejects any nested nonfinite MTP statistics.
                    trial["result"] = try object(result)
                    trial["decode_wall_seconds"] = decodeWall
                    trial["handoff_consumed"] = !ready.isReady
                    trial["generated_token_ids"] = result.tokens
                    trial["callback_token_ids"] = callbacks
                    trial["finish_reason"] = result.finishReason.rawValue
                    trial["final_state_offset"] = result.statistics.finalStateOffset
                    trial["text"] = try tokenizer.decode(result.tokens, skipSpecialTokens: true)
                    if reference == nil { reference = result }
                    let baseline = reference!
                    let goldenExact = result.tokens == expected && result.finishReason.rawValue == goldenFinish
                    let referenceExact = result.tokens == baseline.tokens && result.finishReason == baseline.finishReason
                    let validTokens = !result.tokens.isEmpty && result.tokens.count <= maximum &&
                        result.tokens.allSatisfy { $0 >= 0 && Int($0) < configuration.vocabularySize } &&
                        result.statistics.generatedTokenCount == result.tokens.count &&
                        callbacks == result.tokens && !ready.isReady
                    let modeMatches = ready.statistics.attentionMode == names[index] &&
                        phases.prefill.attentionMode == names[index]
                    let referenceOffsetExact = result.statistics.finalStateOffset == baseline.statistics.finalStateOffset
                    let arOffsetExact = depth != 0 || result.statistics.finalStateOffset == tokens.count + result.tokens.count - 1
                    trial["reference_final_state_offset_matches"] = referenceOffsetExact
                    trial["ar_pending_token_offset_matches"] = depth == 0 ? arOffsetExact as Any : NSNull()
                    trial["valid_tokens_and_handoff"] = validTokens
                    guard validTokens else { throw CLIError.usage("Invalid generated token IDs, callback delivery or handoff consumption") }
                    let correct = goldenExact && referenceExact && validTokens && modeMatches && referenceOffsetExact && arOffsetExact
                    trial["golden_exact"] = goldenExact; trial["reference_exact"] = referenceExact
                    trial["first_difference_vs_golden"] = difference(result.tokens, expected)
                    trial["first_difference_vs_reference"] = difference(result.tokens, baseline.tokens)
                    trial["reference_final_state_offset"] = baseline.statistics.finalStateOffset
                    trial["valid_tokens_and_handoff"] = validTokens
                    trial["reported_attention_mode_matches"] = modeMatches
                    trial["correctnessPassed"] = correct; trial["complete"] = true
                    trial["memory"] = try MX.memory()
                    FileHandle.standardError.write(Data("Prefill attention probe: trial \(index) exact=\(correct), prefill=\(phases.prefill.targetSeconds)s, decode=\(result.decodeSeconds)s\n".utf8))
                } catch {
                    trial["failed_stage"] = stage; trial["error"] = String(describing: error)
                    trial["callback_token_ids"] = callbacks
                    trials.append(trial)
                    throw error
                }
                trials.append(trial)
                try write()
            }
            try write(true)
        } catch {
            report["fatal_error"] = String(describing: error)
            try write()
            throw error
        }
    }

}
