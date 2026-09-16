import ANERunnerCore
import CryptoKit
import Dispatch
import Foundation
import MachO

private enum NativeCLIError: Error, LocalizedError {
    case invalid(String)
    var errorDescription: String? { switch self { case .invalid(let message): message } }
}

/// This executable deliberately has no ANERunnerGPU / CMLX dependency.
@main
struct CoreAIRunnerCLI {
    static func main() async {
        do {
            let arguments = Array(CommandLine.arguments.dropFirst())
            if arguments.isEmpty || arguments == ["help"] || arguments == ["--help"] {
                print("""
                coreai-runner generate --model-dir PATH --attention-manifest PATH \
                  --dense-manifest PATH --moe-manifest PATH --prompt TEXT \
                  [--max-tokens 32] [--repeat 1] [--raw-prompt false] [--output report.json]
                Complete CoreAI text inference. CPU handles tokenization, SSD rows and greedy selection.
                Requires macOS 27. The exported attention manifest sets context capacity.
                """)
                return
            }
            guard arguments.first == "generate", arguments.count % 2 == 1 else {
                throw NativeCLIError.invalid("Use generate followed by unique --option value pairs")
            }
            let allowed: Set<String> = ["--model-dir", "--attention-manifest", "--dense-manifest", "--moe-manifest",
                "--prompt", "--max-tokens", "--repeat", "--raw-prompt", "--output"]
            var options: [String: String] = [:]
            for index in stride(from: 1, to: arguments.count, by: 2) {
                let key = arguments[index]
                guard allowed.contains(key), options[key] == nil else {
                    throw NativeCLIError.invalid("Unknown or duplicate option: \(key)")
                }
                options[key] = arguments[index + 1]
            }
            #if canImport(CoreAI)
            if #available(macOS 27.0, *) { try await generate(options); return }
            #endif
            throw NativeCLIError.invalid("Native CoreAI generation requires macOS 27 and its SDK")
        } catch {
            FileHandle.standardError.write(Data("coreai-runner: \(error.localizedDescription)\n".utf8))
            exit(1)
        }
    }

    #if canImport(CoreAI)
    @available(macOS 27.0, *)
    private static func generate(_ options: [String: String]) async throws {
        func require(_ key: String) throws -> String {
            guard let value = options[key], !value.isEmpty else { throw NativeCLIError.invalid("Missing \(key)") }
            return value
        }
        guard let maximum = Int(options["--max-tokens"] ?? "32"), (1...256).contains(maximum),
              let repeats = Int(options["--repeat"] ?? "1"), (1...3).contains(repeats),
              ["true", "false"].contains(options["--raw-prompt"] ?? "false") else {
            throw NativeCLIError.invalid("max-tokens must be 1...256, repeat 1...3, raw-prompt true/false")
        }
        let directory = URL(fileURLWithPath: try require("--model-dir")).standardizedFileURL.resolvingSymlinksInPath()
        let config = try QwenConfiguration(modelDirectory: directory)
        let tokenizer = try QwenTokenizer(modelDirectory: directory)
        let prompt = try require("--prompt")
        let rendered = options["--raw-prompt"] == "true" ? prompt : try tokenizer.renderChat(messages: [.init(role: "user", content: prompt)])
        let tokens = try tokenizer.encode(rendered)
        guard !tokens.isEmpty else { throw NativeCLIError.invalid("Prompt must contain tokens") }
        let hash = try NGramHash(unigramVocabularySize: UInt32(config.vocabularySize), ngramSize: config.ngramSize,
            headsPerNGram: config.ngramHeadsPerOrder, vocabularyBase: UInt64(config.ngramVocabularyBase),
            vocabularyDivisor: UInt64(config.ngramDivisor), pleLayerIndex: 0, eosTokenID: UInt32(config.eosTokenID))
        guard !config.ngramTableFile.contains(".."), !config.ngramTableFile.hasPrefix("/") else {
            throw NativeCLIError.invalid("Invalid n-gram table path")
        }
        let table = try NGramTable(url: directory.appendingPathComponent(config.ngramTableFile))
        guard table.rowCount == hash.totalRows, table.dimension * hash.headCount == config.pleEmbeddingDimension,
              table.scale == Float(config.ngramScale), config.pleLayerIndices == [1] else {
            throw NativeCLIError.invalid("PLE table does not match model configuration")
        }
        let loadStart = DispatchTime.now().uptimeNanoseconds
        let model = try await CoreAINativeModel(
            attentionManifest: URL(fileURLWithPath: require("--attention-manifest")),
            denseManifest: URL(fileURLWithPath: require("--dense-manifest")),
            moeManifest: URL(fileURLWithPath: require("--moe-manifest")), computeUnits: .gpu) { kind, current, total in
                if current % 4 == 0 || current == total { progress("Loaded \(kind) \(current)/\(total)") }
            }
        let configSHA = SHA256.hash(data: try Data(contentsOf: directory.appendingPathComponent("config.json")))
            .map { String(format: "%02x", $0) }.joined()
        guard model.manifestModelDirectory == directory, model.sourceConfigSHA256 == configSHA,
              tokens.count + maximum <= model.capacity else {
            throw NativeCLIError.invalid("Model identity mismatch or prompt/output exceeds exported capacity")
        }
        let loadSeconds = seconds(loadStart)
        var runs = [[String: Any]](), first = [Int32]()
        for run in 0..<repeats {
            if run > 0 { try model.reset() }
            var history = hash.initialHistory, logicalBytes = 0, readSeconds = 0.0
            func embedding(_ token: Int32) throws -> [Float] {
                let rows = try hash.rowIDs(previousTokens: history, tokens: [UInt32(token)])
                let started = DispatchTime.now().uptimeNanoseconds
                let values = try table.readRows(rows)
                readSeconds += seconds(started)
                history = try hash.history(after: [UInt32(token)], previousTokens: history)
                logicalBytes += rows.count * table.dimension
                return values
            }
            let initialCalls = model.successfulCalls
            let start = DispatchTime.now().uptimeNanoseconds
            var logits = [Float]()
            for (index, token) in tokens.enumerated() {
                logits = try await model.forward(token: token, pleEmbedding: embedding(token))
                if (index + 1) % 8 == 0 || index + 1 == tokens.count { progress("Prefill \(index + 1)/\(tokens.count), run \(run + 1)") }
            }
            let prefillSeconds = seconds(start), prefillCalls = model.successfulCalls - initialCalls
            var output = [Int32](), decodeDurations = [Double](), stop = "length"
            for step in 0..<maximum {
                let selected = try greedy(logits, excluding: tokenizer.reservedOutputTokenIDs)
                output.append(selected)
                if tokenizer.eosTokenIDs.contains(selected) { stop = "eos"; break }
                progress("Generated \(output.count): \(try tokenizer.decode(output))")
                if step + 1 < maximum {
                    let started = DispatchTime.now().uptimeNanoseconds
                    logits = try await model.forward(token: selected, pleEmbedding: embedding(selected))
                    decodeDurations.append(seconds(started))
                }
            }
            if run == 0 { first = output }
            guard first == output else { throw NativeCLIError.invalid("Reset replay generated different token IDs") }
            runs.append(["run": run + 1, "text": try tokenizer.decode(output, skipSpecialTokens: true),
                "generated_token_ids": output, "stop_reason": stop, "reset_tokens_match": output == first,
                "prefill_tokens": tokens.count, "prefill_seconds": prefillSeconds, "prefill_coreai_calls": prefillCalls,
                "decode_forward_steps": decodeDurations.count, "decode_forward_seconds": decodeDurations.reduce(0, +),
                "decode_step_seconds": decodeDurations, "total_coreai_calls": model.successfulCalls - initialCalls,
                "final_state_offset": model.offset, "ssd_logical_bytes": logicalBytes, "ssd_read_seconds": readSeconds])
        }
        let images = (0..<_dyld_image_count()).compactMap { _dyld_get_image_name($0).map { String(cString: $0) } }
        let mlxImages = images.filter { URL(fileURLWithPath: $0).lastPathComponent.lowercased().contains("mlx") }
        guard mlxImages.isEmpty else { throw NativeCLIError.invalid("Unexpected MLX runtime image in native executable") }
        let report: [String: Any] = ["schema_version": 1, "backend": "native-coreai", "status": "completed",
            "coreai_compute_preference": "gpu", "hardware_placement_verified": false,
            "mlx_runtime_images": mlxImages, "capacity": model.capacity, "prompt_token_ids": tokens,
            "load_seconds": loadSeconds, "runs": runs,
            "cpu_operations": ["tokenization", "n-gram hash", "SSD row reads and FP8 unpack", "greedy token selection"],
            "model_directory": directory.path,
            "quality_acceptance": false, "performance_acceptance": false,
            "limitations": "S1 short-context integration, explicit CoreAI state; server/prefix/SSD KV cache migration remains separate."]
        var data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
        data.append(0x0a)
        if let path = options["--output"] {
            let url = URL(fileURLWithPath: path)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: url, options: .atomic)
        } else { FileHandle.standardOutput.write(data) }
    }

    private static func greedy(_ logits: [Float], excluding: Set<Int32>) throws -> Int32 {
        guard !logits.isEmpty, logits.allSatisfy(\.isFinite) else { throw NativeCLIError.invalid("Invalid final logits") }
        var selected: Int32?, maximum = -Float.infinity
        for (index, score) in logits.enumerated() where !excluding.contains(Int32(index)) {
            if score > maximum { maximum = score; selected = Int32(index) }
        }
        guard let selected else { throw NativeCLIError.invalid("No permitted output token") }
        return selected
    }
    #endif

    private static func seconds(_ start: UInt64) -> Double { Double(DispatchTime.now().uptimeNanoseconds - start) * 1e-9 }
    private static func progress(_ value: String) { FileHandle.standardError.write(Data((value + "\n").utf8)) }
}
