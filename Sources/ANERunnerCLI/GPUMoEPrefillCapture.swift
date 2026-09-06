import ANERunnerGPU
import CMLX
import CryptoKit
import Foundation

extension RunnerCLI {
    /// Capture real prefill inputs at selected MoE boundaries. Extra eval/file
    /// IO is diagnostic; the resulting phase clocks are not throughput baselines.
    static func captureGPUMoEPrefill(_ args: Arguments) throws {
        try args.validate(["--model-dir", "--tokens-file", "--golden-report", "--output", "--fixture-dir"])
        let manager = FileManager.default
        let output = URL(fileURLWithPath: try args.require("--output")).standardizedFileURL.resolvingSymlinksInPath()
        let destination = URL(fileURLWithPath: try args.require("--fixture-dir")).standardizedFileURL.resolvingSymlinksInPath()
        guard !manager.fileExists(atPath: output.path), !manager.fileExists(atPath: destination.path),
              output.path != destination.path, !output.path.hasPrefix(destination.path + "/") else {
            throw CLIError.usage("Capture needs a new manifest outside the new --fixture-dir")
        }
        let layers: Set<Int> = [0, 23, 47], offsets: Set<Int> = [0, 4_992, 9_984]
        let maximum = 128, chunk = 416
        let staging = destination.deletingLastPathComponent()
            .appendingPathComponent(".moe-prefill-staging-" + UUID().uuidString, isDirectory: true)
        var fixtures = [[String: Any]](), seen = Set<String>(), callbacks = [Int32]()
        var stage = "validation"
        var report: [String: Any] = [
            "schema": "qwen38-moe-prefill-input-capture-v1", "complete": false,
            "passed": false, "committed": false, "max_tokens": maximum, "mtp_enabled": false,
            "full_model_instances": 1, "generator_instances": 2,
            "fixture_directory": destination.path, "artifact_root": staging.path,
            "requested_layers": layers.sorted(), "requested_offsets": offsets.sorted(),
            "clock": "mach_absolute_time_nanoseconds", "logit_finiteness_checked": false,
            "notes": [
                "Real producer.prefill(request) to consumer.decode(ready), with original full prompt and reference AR generation.",
                "Only x=pre2.mixed immediately before MoE is captured; no full-model trace or expert-output diagnostics are requested.",
                "The observer is synchronous and disabled before decode. Input evaluation and file IO change overlap; clocks are not throughput baselines.",
                "Fixtures remain staged until all nine positions and complete golden IDs, finish reason, callbacks, handoff and AR final offset pass.",
                "Failed runs retain staged artifacts for inspection; only committed=true and passed=true authorize fixture reuse.",
                "BF16 input bytes are preserved. Public token/statistic checks do not establish internal tensor or logit finiteness."
            ]]
        func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
        func elapsed(_ start: UInt64) -> Double { Double(now() - start) * 1e-9 }
        func hash(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
        func object<T: Encodable>(_ value: T) throws -> Any {
            try JSONSerialization.jsonObject(with: JSONEncoder().encode(value))
        }
        func write() throws {
            report["fixtures"] = fixtures
            report["callback_token_ids"] = callbacks
            try emit(report, to: output.path)
        }
        try write()
        do {
            let directory = URL(fileURLWithPath: try args.require("--model-dir")).standardizedFileURL.resolvingSymlinksInPath()
            let inputURL = URL(fileURLWithPath: try args.require("--tokens-file")).standardizedFileURL
            let inputData = try Data(contentsOf: inputURL)
            let tokens = try JSONDecoder().decode([Int32].self, from: inputData)
            guard (10_000...12_000).contains(tokens.count), tokens.count - 1 >= 9_984 + chunk else {
                throw CLIError.usage("Capture needs a real 11k prompt containing the complete offset-9984 chunk")
            }
            let configuration = try QwenConfiguration(modelDirectory: directory)
            let request = QwenGenerationRequest(tokens: tokens, maxTokens: maximum, contextLimit: 16_384,
                prefillChunk: chunk, mtpDepth: 0, decodeMode: .reference, prefillAttention: .reference)
            try request.validate(configuration: configuration)
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
            guard let golden = goldenReport.trials.first, golden.prompt_tokens == tokens,
                  goldenReport.max_tokens == maximum, !golden.generated_token_ids.isEmpty,
                  golden.generated_token_ids.count <= maximum,
                  golden.generated_token_ids.allSatisfy({ $0 >= 0 && Int($0) < configuration.vocabularySize }),
                  (golden.mtp_depth == 0 || golden.mtp_enabled == false || goldenReport.mtp_enabled == false),
                  golden.mtp_depth == nil || golden.mtp_depth == 0,
                  golden.mtp_enabled != true, goldenReport.mtp_enabled != true,
                  ["eos", "length"].contains(golden.finish_reason),
                  golden.finish_reason != "length" || golden.generated_token_ids.count == maximum else {
                throw CLIError.usage("Golden must have matching complete prompt, budget 128 and explicit AR configuration")
            }
            let configHash = hash(try Data(contentsOf: directory.appendingPathComponent("config.json")))
            let indexHash = hash(try Data(contentsOf: directory.appendingPathComponent("model.safetensors.index.json")))
            let inputHash = hash(inputData)
            let executable = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL.resolvingSymlinksInPath()
            report["provenance"] = ["command": CommandLine.arguments, "executable": executable.path,
                "executable_sha256": hash(try Data(contentsOf: executable)), "config_sha256": configHash,
                "weight_index_sha256": indexHash]
            report["model_directory"] = directory.path
            report["input"] = ["path": inputURL.path, "sha256": inputHash, "token_count": tokens.count, "token_ids": tokens]
            report["golden"] = ["path": goldenURL.path, "sha256": hash(goldenData),
                "token_ids": golden.generated_token_ids, "finish_reason": golden.finish_reason, "max_tokens": maximum]
            report["configuration"] = ["context_limit": 16_384, "prefill_chunk": chunk, "mtp_depth": 0,
                "prefill_evaluate_every_layers": 4, "prefill_attention": "reference",
                "prefill_accumulation": "reference", "decode_mode": "reference"]
            report["environment"] = Dictionary(uniqueKeysWithValues:
                ["ANERUNNER_FUSED_PREFILL", "ANERUNNER_BLOCKED_GDN", "ANERUNNER_PREFILL_EVAL_LAYERS"]
                    .map { ($0, ProcessInfo.processInfo.environment[$0] ?? "unset") })
            try manager.createDirectory(at: staging, withIntermediateDirectories: true)
            try write()
            var previousCache = 0
            try MX.check(mlx_set_cache_limit(&previousCache, 256 * 1024 * 1024), "MoE capture cache limit")
            defer { var ignored = 0; _ = mlx_set_cache_limit(&ignored, previousCache) }
            stage = "load"
            let loadStart = now()
            let model = try QwenModel(modelDirectory: directory) { current, total in
                if current % 8 == 0 || current == total {
                    FileHandle.standardError.write(Data("MoE prefill capture: loaded layer \(current)/\(total)\n".utf8))
                }
            }
            report["model_load_seconds"] = elapsed(loadStart)
            report["loaded_memory"] = try MX.memory()
            let producer = try QwenGenerator(model: model), consumer = try QwenGenerator(model: model)
            model.prefillMoEObserver = { layer, offset, x in
                guard layers.contains(layer), offsets.contains(offset) else { return }
                let key = "layer\(layer)-offset\(offset)"
                guard seen.insert(key).inserted, x.shape == [1, chunk, 2_560], x.dtype == MLX_BFLOAT16 else {
                    throw CLIError.usage("Duplicate capture or unexpected MoE input shape/dtype at \(key)")
                }
                let captureStart = now()
                let file = staging.appendingPathComponent(key + ".safetensors")
                let temporary = staging.appendingPathComponent("." + UUID().uuidString + ".safetensors")
                guard !manager.fileExists(atPath: file.path) else { throw CLIError.usage("Capture file already exists") }
                defer { try? manager.removeItem(at: temporary) }
                // No dtype conversion and no retained full-model capture graph.
                try x.eval()
                let arrays = mlx_map_string_to_array_new(), metadata = mlx_map_string_to_string_new()
                defer { _ = mlx_map_string_to_array_free(arrays); _ = mlx_map_string_to_string_free(metadata) }
                try MX.check(mlx_map_string_to_array_insert(arrays, "x", x.handle), "insert MoE input")
                let attributes = ["schema": "qwen38-moe-prefill-input-v1", "phase": "prefill",
                    "layer": String(layer), "absolute_offset": String(offset), "boundary": "pre2.mixed before MoE.forward",
                    "config_sha256": configHash, "weight_index_sha256": indexHash, "token_file_sha256": inputHash]
                for (key, value) in attributes {
                    try MX.check(key.withCString { k in value.withCString { mlx_map_string_to_string_insert(metadata, k, $0) } }, "MoE capture metadata")
                }
                try MX.check(temporary.path.withCString { mlx_save_safetensors($0, arrays, metadata) }, "save MoE input")
                try manager.moveItem(at: temporary, to: file)
                let data = try Data(contentsOf: file)
                fixtures.append(["layer": layer, "offset": offset, "phase": "prefill", "file": file.path,
                    "filename": file.lastPathComponent,
                    "sha256": hash(data), "file_bytes": data.count, "tensor_name": "x", "shape": x.shape,
                    "dtype": "BF16", "tensor_bytes": x.nbytes, "token_ids": Array(tokens[offset..<(offset + chunk)]),
                    "capture_seconds": elapsed(captureStart)])
                try write()
            }
            defer { model.prefillMoEObserver = nil }
            stage = "prefill"
            let prefillStart = now()
            let ready = try producer.prefill(request)
            let prefillWall = elapsed(prefillStart)
            model.prefillMoEObserver = nil
            defer { ready.discard() }
            report["prefill"] = try object(ready.statistics)
            report["prefill_wall_seconds"] = prefillWall
            report["preparation_seconds"] = ready.preparationSeconds
            report["first_token"] = ready.firstToken
            report["ready_at_return"] = ready.isReady
            report["capture_count"] = fixtures.count
            guard fixtures.count == 9, seen.count == 9, ready.isReady else {
                throw CLIError.usage("Expected all nine MoE captures and a ready handoff")
            }
            try write()
            stage = "decode"
            let decodeStart = now()
            let result = try consumer.decode(ready) { callbacks.append($0) }
            let decodeWall = elapsed(decodeStart)
            report["result"] = try object(result)
            report["decode_wall_seconds"] = decodeWall
            report["generated_token_ids"] = result.tokens
            report["finish_reason"] = result.finishReason.rawValue
            report["final_state_offset"] = result.statistics.finalStateOffset
            report["handoff_consumed"] = !ready.isReady
            guard let phases = result.phases else { throw CLIError.usage("Missing phase statistics") }
            let expectedOffset = tokens.count + result.tokens.count - 1
            let finite = [prefillWall, decodeWall, result.preparationSeconds, result.timeToFirstTokenSeconds,
                result.decodeSeconds, result.totalSeconds, phases.prefill.totalSeconds, phases.prefill.targetSeconds,
                phases.prefill.ssdWaitSeconds, phases.decodeServiceSeconds, phases.decodeSSDWaitSeconds,
                phases.handoffWaitSeconds, phases.handoffConsumeSeconds, result.statistics.callbackSeconds]
                .allSatisfy { $0.isFinite && $0 >= 0 }
            let checks = ["all_nine_inputs": fixtures.count == 9 && seen.count == 9,
                "golden_ids_exact": result.tokens == golden.generated_token_ids,
                "golden_finish_exact": result.finishReason.rawValue == golden.finish_reason,
                "ar_final_offset": result.statistics.finalStateOffset == expectedOffset,
                "callback_ids_exact": callbacks == result.tokens, "handoff_consumed": !ready.isReady,
                "generated_count": result.statistics.generatedTokenCount == result.tokens.count,
                "reference_attention": ready.statistics.attentionMode == "reference" && phases.prefill.attentionMode == "reference",
                "observer_disabled_before_decode": model.prefillMoEObserver == nil, "finite_public_metrics": finite]
            report["checks"] = checks
            report["expected_final_state_offset"] = expectedOffset
            if let difference = zip(result.tokens, golden.generated_token_ids).enumerated().first(where: { $0.element.0 != $0.element.1 }) {
                report["first_difference_vs_golden"] = ["index": difference.offset, "actual": difference.element.0, "expected": difference.element.1] as [String: Any]
            } else if result.tokens.count != golden.generated_token_ids.count {
                report["first_difference_vs_golden"] = ["index": min(result.tokens.count, golden.generated_token_ids.count),
                    "actual_count": result.tokens.count, "expected_count": golden.generated_token_ids.count]
            } else { report["first_difference_vs_golden"] = NSNull() }
            guard checks.values.allSatisfy({ $0 }) else { throw CLIError.usage("MoE input capture failed full-generation checks; fixtures remain staged") }
            stage = "commit"
            report["memory"] = try MX.memory()
            try manager.moveItem(at: staging, to: destination)
            for index in fixtures.indices {
                let filename = fixtures[index]["filename"] as! String
                fixtures[index]["file"] = destination.appendingPathComponent(filename).path
            }
            report["artifact_root"] = destination.path
            report["committed"] = true; report["complete"] = true; report["passed"] = true
            try write()
        } catch {
            report["failed_stage"] = stage; report["fatal_error"] = String(describing: error)
            report["passed"] = false; report["complete"] = false
            try write()
            throw error
        }
    }
}
