import ANERunnerCore
import ANERunnerGPU
import CMLX
import Dispatch
import Foundation

extension RunnerCLI {
    static func generateCoreAIHybrid(_ args: Arguments) async throws {
        #if canImport(CoreAI)
        if #available(macOS 27.0, *) {
            try await withTaskExecutorPreference(CoreAIHybridExecutor.shared) {
                try await generateCoreAIHybridOnExecutor(args)
            }
            return
        }
        #endif
        throw CLIError.usage("generate-coreai-hybrid requires macOS 27 and a runner built with its SDK")
    }

    // Force entry onto the preferred executor before the first MLX stream is
    // created. Async CoreAI suspension then resumes on the same native thread.
    @concurrent
    private static func generateCoreAIHybridOnExecutor(_ args: Arguments) async throws {
        try args.validate(["--model-dir", "--manifest", "--prompt", "--raw-prompt", "--max-tokens",
                           "--repeat", "--compare-reference", "--output"])
        #if canImport(CoreAI)
        if #available(macOS 27.0, *) {
            func boolean(_ key: String) throws -> Bool {
                let value = args[key] ?? "false"
                guard ["true", "false"].contains(value) else { throw CLIError.usage("\(key) requires true/false") }
                return value == "true"
            }
            guard let maximum = Int(args["--max-tokens"] ?? "32"), (1...256).contains(maximum),
                  let repetitions = Int(args["--repeat"] ?? "1"), (1...3).contains(repetitions) else {
                throw CLIError.usage("max-tokens must be 1...256 and repeat 1...3")
            }
            let compare = try boolean("--compare-reference")
            let directory = URL(fileURLWithPath: try args.require("--model-dir"))
            let manifest = URL(fileURLWithPath: try args.require("--manifest"))
            let tokenizer = try QwenTokenizer(modelDirectory: directory)
            let prompt = try args.require("--prompt")
            let rendered = try boolean("--raw-prompt") ? prompt : tokenizer.renderChat(messages: [.init(role: "user", content: prompt)])
            let tokens = try tokenizer.encode(rendered)
            guard !tokens.isEmpty else { throw CLIError.usage("Prompt must contain tokens") }
            var previousCacheLimit = 0
            try MX.check(mlx_set_cache_limit(&previousCacheLimit, 256 * 1024 * 1024), "set allocation cache limit")
            defer { var ignored = 0; _ = mlx_set_cache_limit(&ignored, previousCacheLimit) }
            // The helper owns all CoreAI/MLX model objects. They are released
            // before the optional second full-model reference load.
            let result = try await runCoreAIHybrid(directory: directory, manifest: manifest,
                tokenizer: tokenizer, tokens: tokens, maximum: maximum, repetitions: repetitions,
                captureLogits: compare)
            var report = result.report
            if compare {
                try MX.synchronize()
                try MX.check(mlx_clear_cache(), "clear allocation cache before reference")
                report["reference"] = try compareCoreAIHybrid(directory: directory, tokenizer: tokenizer,
                    tokens: tokens, generated: result.generated, expected: result.logits)
            }
            try emit(report, to: args["--output"])
            return
        }
        #endif
        throw CLIError.usage("generate-coreai-hybrid requires macOS 27 and a runner built with its SDK")
    }

    #if canImport(CoreAI)
    @available(macOS 27.0, *)
    private static func runCoreAIHybrid(directory: URL, manifest: URL, tokenizer: QwenTokenizer,
        tokens: [Int32], maximum: Int, repetitions: Int, captureLogits: Bool) async throws
        -> (report: [String: Any], generated: [Int32], logits: [[Float]]) {
        func seconds(_ start: UInt64) -> Double { Double(DispatchTime.now().uptimeNanoseconds - start) * 1e-9 }
        let loadStart = DispatchTime.now().uptimeNanoseconds
        hybridProgress("Loading CoreAI attention assets")
        let backend = try await CoreAIHybridAttention(manifestURL: manifest, computeUnits: .gpu) { current, total in
            if current % 4 == 0 { hybridProgress("Loaded CoreAI attention \(current)/\(total)") }
        }
        guard tokens.count + maximum <= backend.capacity else {
            throw CLIError.usage("Prompt plus output exceeds hybrid asset capacity \(backend.capacity)")
        }
        let model = try QwenCoreAIHybridModel(modelDirectory: directory, attentionBackend: backend) { current, total in
            if current % 4 == 0 { hybridProgress("Loaded hybrid weights \(current)/\(total)") }
        }
        let loadSeconds = seconds(loadStart)
        var reports = [[String: Any]](), captured = [[Float]](), firstTokens = [Int32]()
        for repetition in 0..<repetitions {
            if repetition > 0 { try model.reset() }
            let initialCalls = backend.successfulCalls
            let initialPrediction = backend.totalPredictionMilliseconds
            let initialInput = backend.totalInputMilliseconds
            let initialOutput = backend.totalOutputMilliseconds
            let start = DispatchTime.now().uptimeNanoseconds
            var nextLogits: Tensor?
            for (index, token) in tokens.enumerated() {
                nextLogits = try await model.forward(token: token)
                if (index + 1) % 8 == 0 || index + 1 == tokens.count {
                    hybridProgress("Prefill \(index + 1)/\(tokens.count), run \(repetition + 1)")
                }
            }
            guard let ready = nextLogits else { throw CLIError.usage("No prompt logits") }
            var values = try ready.floats()
            let prefillSeconds = seconds(start)
            let prefillCoreAIMilliseconds = backend.totalPredictionMilliseconds - initialPrediction
            var generated = [Int32](), decodeDurations = [Double](), stop = "length"
            let decodeStart = DispatchTime.now().uptimeNanoseconds
            for step in 0..<maximum {
                if captureLogits, repetition == 0 { captured.append(values) }
                let selected = try hybridGreedy(values, excluded: tokenizer.reservedOutputTokenIDs)
                generated.append(selected)
                if tokenizer.eosTokenIDs.contains(selected) { stop = "eos"; break }
                hybridProgress("Generated \(generated.count): \(try tokenizer.decode(generated))")
                if step + 1 < maximum {
                    let stepStart = DispatchTime.now().uptimeNanoseconds
                    values = try await model.forward(token: selected).floats()
                    decodeDurations.append(seconds(stepStart))
                }
            }
            let decodeSeconds = seconds(decodeStart)
            if repetition == 0 { firstTokens = generated }
            let same = generated == firstTokens
            let expectedCalls = 48 * (tokens.count + generated.count - 1)
            let actualCalls = backend.successfulCalls - initialCalls
            guard actualCalls == expectedCalls else { throw CLIError.usage("CoreAI attention call count is incomplete") }
            reports.append(["run": repetition + 1, "generated_token_ids": generated,
                "text": try tokenizer.decode(generated, skipSpecialTokens: true), "stop_reason": stop,
                "reset_tokens_match": same, "prefill_tokens": tokens.count,
                "prefill_seconds": prefillSeconds, "decode_forward_steps": decodeDurations.count,
                "prefill_coreai_prediction_milliseconds": prefillCoreAIMilliseconds,
                "decode_coreai_prediction_milliseconds": backend.totalPredictionMilliseconds - initialPrediction - prefillCoreAIMilliseconds,
                "decode_forward_seconds": decodeDurations.reduce(0, +), "decode_wall_seconds": decodeSeconds,
                "decode_step_seconds": decodeDurations, "final_state_offset": model.offset,
                "expected_coreai_calls": expectedCalls, "actual_coreai_calls": actualCalls,
                "coreai_prediction_milliseconds": backend.totalPredictionMilliseconds - initialPrediction,
                "coreai_input_milliseconds": backend.totalInputMilliseconds - initialInput,
                "coreai_output_milliseconds": backend.totalOutputMilliseconds - initialOutput])
            guard same else { throw CLIError.usage("Reset replay generated different token IDs") }
        }
        return (["schema_version": 1, "backend": "coreai-attention-mlx-q4-hybrid",
            "status": "completed", "layer_count": 48, "coreai_gdn_layers": 36, "coreai_qsa_layers": 12,
            "coreai_compute_preference": "gpu", "hardware_placement_verified": false,
            "mlx_components": ["Q4 routed experts", "shared experts", "hyper connections", "SSD PLE", "embedding", "output head"],
            "prefill_policy": "tokenwise S1", "capacity": backend.capacity,
            "manifest": manifest.standardizedFileURL.path, "model_directory": directory.standardizedFileURL.path,
            "prompt_token_ids": tokens, "load_seconds": loadSeconds, "runs": reports,
            "memory": try MX.memory(), "performance_acceptance": false,
            "limitations": "Experimental short-context hybrid generation. No native CoreAI Q4 MoE, server, shared prefix cache, SSD KV archive or hardware placement validation."], firstTokens, captured)
    }

    @available(macOS 27.0, *)
    private static func compareCoreAIHybrid(directory: URL, tokenizer: QwenTokenizer,
        tokens: [Int32], generated: [Int32], expected: [[Float]]) throws -> [String: Any] {
        guard expected.count == generated.count else { throw CLIError.usage("Incomplete hybrid logit capture") }
        hybridProgress("Loading independent MLX reference after releasing hybrid model")
        let model = try QwenModel(modelDirectory: directory, reservedOutputIDs: tokenizer.reservedOutputTokenIDs) { current, _ in
            if current % 12 == 0 { hybridProgress("Loaded reference \(current)/48") }
        }
        var state = model.makeState()
        var latest: Tensor?
        for token in tokens {
            let output = try model.forward(tokens: [token], state: &state, phase: .prefill)
            latest = output.logits
            try model.evaluate([output.stream] + [latest].compactMap { $0 }, state: &state)
        }
        var comparisons = [[String: Any]](), agreements = 0
        for step in expected.indices {
            guard let tensor = latest else { throw CLIError.usage("Reference produced no logits") }
            let reference = try tensor.floats(), actual = expected[step]
            guard actual.count == reference.count, reference.allSatisfy(\.isFinite) else {
                throw CLIError.usage("Invalid reference logits")
            }
            var errorSquared = 0.0, referenceSquared = 0.0, maximumError = 0.0
            for (a, b) in zip(actual, reference) {
                let delta = Double(a) - Double(b)
                errorSquared += delta * delta; referenceSquared += Double(b) * Double(b)
                maximumError = max(maximumError, abs(delta))
            }
            let selected = try hybridGreedy(reference, excluded: tokenizer.reservedOutputTokenIDs)
            let same = selected == generated[step]
            if same { agreements += 1 }
            comparisons.append(["step": step, "hybrid_token": generated[step], "reference_token": selected,
                "top1_match": same, "maximum_absolute_error": maximumError,
                "relative_l2_error": sqrt(errorSquared / max(referenceSquared, 1e-30))])
            if step + 1 < expected.count {
                let output = try model.forward(tokens: [generated[step]], state: &state, phase: .decode)
                latest = output.logits
                try model.evaluate([output.stream] + [output.logits].compactMap { $0 }, state: &state)
            }
        }
        return ["kind": "independent MLX BF16 reference, teacher-forced hybrid token stream",
            "top1_agreements": agreements, "steps": comparisons,
            "quality_acceptance": false, "note": "Diagnostics measure cumulative source precision drift; matching a short completion does not establish model quality."]
    }

    private static func hybridGreedy(_ values: [Float], excluded: Set<Int32>) throws -> Int32 {
        guard !values.isEmpty, values.allSatisfy(\.isFinite) else { throw CLIError.usage("Nonfinite or empty logits") }
        var winner: Int32?, score = -Float.infinity
        for (index, value) in values.enumerated() where !excluded.contains(Int32(index)) {
            if value > score { score = value; winner = Int32(index) }
        }
        guard let winner else { throw CLIError.usage("No permitted output token") }
        return winner
    }

    private static func hybridProgress(_ value: String) {
        FileHandle.standardError.write(Data((value + "\n").utf8))
    }
    #endif
}
