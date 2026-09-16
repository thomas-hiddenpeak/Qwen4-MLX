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
                  [--pd-manifest PATH] [--prefill-chunk 0|1|PRIMARY] [--compare-prefill true]
                coreai-runner serve --model-dir PATH --attention-manifest PATH \
                  --dense-manifest PATH --moe-manifest PATH [--host 127.0.0.1] [--port 11236]
                  [--prefix-cache-bytes 536870912] [--prefix-cache-entries 2]
                  [--request-timeout-seconds 1800] [--max-pending-requests 2]
                  [--pd-manifest PATH] [--prefill-chunk 0|1|PRIMARY]
                Complete CoreAI text inference. CPU handles tokenization, SSD rows and greedy selection.
                Requires macOS 27. The selected manifest sets context capacity.
                generate accepts --prompt-file instead of --prompt. Chunk 0 selects the manifest primary.
                Primary chunks: 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048. Compare runs S1/primary/primary/S1.
                """)
                return
            }
            guard let command = arguments.first, ["generate", "serve"].contains(command), arguments.count % 2 == 1 else {
                throw NativeCLIError.invalid("Use generate or serve followed by unique --option value pairs")
            }
            var allowed: Set<String> = ["--model-dir", "--attention-manifest", "--dense-manifest", "--moe-manifest", "--pd-manifest", "--prefill-chunk"]
            allowed.formUnion(command == "generate" ? ["--prompt", "--prompt-file", "--max-tokens", "--repeat", "--raw-prompt", "--output", "--compare-prefill"] :
                ["--host", "--port", "--prefix-cache-bytes", "--prefix-cache-entries", "--request-timeout-seconds",
                 "--max-pending-requests", "--max-connections", "--max-body-bytes", "--max-output-bytes"])
            var options: [String: String] = [:]
            for index in stride(from: 1, to: arguments.count, by: 2) {
                let key = arguments[index]
                guard allowed.contains(key), options[key] == nil else {
                    throw NativeCLIError.invalid("Unknown or duplicate option: \(key)")
                }
                options[key] = arguments[index + 1]
            }
            #if canImport(CoreAI)
            if #available(macOS 27.0, *) {
                if command == "serve" { try await serve(options) }
                else { try await generate(options) }
                return
            }
            #endif
            throw NativeCLIError.invalid("Native CoreAI generation requires macOS 27 and its SDK")
        } catch {
            FileHandle.standardError.write(Data("coreai-runner: \(error.localizedDescription)\n".utf8))
            exit(1)
        }
    }

    #if canImport(CoreAI)
    private static func prefillChunkOption(_ options: [String: String]) throws -> Int {
        guard let value = Int(options["--prefill-chunk"] ?? "0"),
              [0, 1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048].contains(value) else {
            throw NativeCLIError.invalid("prefill-chunk must be 0, 1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, or 2048")
        }
        guard value <= 1 || options["--pd-manifest"]?.isEmpty == false else {
            throw NativeCLIError.invalid("Chunked prefill requires a PD manifest")
        }
        return value
    }

    @available(macOS 27.0, *)
    private static func serve(_ options: [String: String]) async throws {
        let requestedChunk = try prefillChunkOption(options)
        func path(_ key: String) throws -> URL {
            guard let value = options[key], !value.isEmpty else { throw NativeCLIError.invalid("Missing \(key)") }
            return URL(fileURLWithPath: value).standardizedFileURL.resolvingSymlinksInPath()
        }
        func number(_ key: String, _ fallback: Int, _ range: ClosedRange<Int>) throws -> Int {
            guard let value = Int(options[key] ?? String(fallback)), range.contains(value) else {
                throw NativeCLIError.invalid("\(key) must be in \(range)")
            }
            return value
        }
        var configuration = try CoreAIServiceConfiguration(modelDirectory: path("--model-dir"),
            attentionManifest: path("--attention-manifest"), denseManifest: path("--dense-manifest"), moeManifest: path("--moe-manifest"),
            cacheBytes: number("--prefix-cache-bytes", 536_870_912, 0...2_147_483_648),
            cacheEntries: number("--prefix-cache-entries", 2, 0...8),
            maxPendingRequests: number("--max-pending-requests", 2, 1...8),
            maxOutputBytes: number("--max-output-bytes", 65_536, 4096...1_048_576))
        if options["--pd-manifest"] != nil { configuration.pdManifest = try path("--pd-manifest") }
        configuration.prefillChunk = requestedChunk
        let host = options["--host"] ?? "127.0.0.1"
        guard ["127.0.0.1", "0.0.0.0"].contains(host) else {
            throw NativeCLIError.invalid("--host must be 127.0.0.1 or 0.0.0.0")
        }
        let transport = try CoreAIHTTPServer.Configuration(host: host, port: UInt16(number("--port", 11236, 1024...65535)),
            maxConnections: number("--max-connections", 8, 1...32), maxBodyBytes: number("--max-body-bytes", 262_144, 1024...1_048_576),
            maxOutputBytes: configuration.maxOutputBytes,
            requestTimeoutSeconds: Double(number("--request-timeout-seconds", 1800, 1...86_400)), sendTimeoutSeconds: 15)
        let worker = CoreAIServiceWorker(configuration: configuration)
        worker.start()
        progress("CoreAI service loading; GET /health on \(host):\(transport.port) reports readiness")
        do { try await CoreAIHTTPServer.run(configuration: transport, backend: worker) }
        catch { worker.shutdown(); await worker.waitUntilStopped(); throw error }
        worker.shutdown()
        await worker.waitUntilStopped()
    }

    @available(macOS 27.0, *)
    private static func generate(_ options: [String: String]) async throws {
        func require(_ key: String) throws -> String {
            guard let value = options[key], !value.isEmpty else { throw NativeCLIError.invalid("Missing \(key)") }
            return value
        }
        guard let maximum = Int(options["--max-tokens"] ?? "32"), (1...256).contains(maximum),
              let repeats = Int(options["--repeat"] ?? "1"), (1...4).contains(repeats),
              ["true", "false"].contains(options["--raw-prompt"] ?? "false"),
              ["true", "false"].contains(options["--compare-prefill"] ?? "false") else {
            throw NativeCLIError.invalid("max-tokens must be 1...256, repeat 1...4, raw-prompt true/false")
        }
        let requestedChunk = try prefillChunkOption(options)
        guard options["--compare-prefill"] != "true" || options["--pd-manifest"]?.isEmpty == false else {
            throw NativeCLIError.invalid("compare-prefill requires a PD manifest")
        }
        let directory = URL(fileURLWithPath: try require("--model-dir")).standardizedFileURL.resolvingSymlinksInPath()
        let config = try QwenConfiguration(modelDirectory: directory)
        let tokenizer = try QwenTokenizer(modelDirectory: directory)
        guard options["--prompt"] == nil || options["--prompt-file"] == nil else {
            throw NativeCLIError.invalid("Choose prompt or prompt-file")
        }
        let prompt: String
        if let file = options["--prompt-file"] { prompt = try String(contentsOfFile: file, encoding: .utf8) }
        else { prompt = try require("--prompt") }
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
        progress("Loading CoreAI runtime")
        let model = try await CoreAITextRuntime(
            attentionManifest: URL(fileURLWithPath: require("--attention-manifest")),
            denseManifest: URL(fileURLWithPath: require("--dense-manifest")),
            moeManifest: URL(fileURLWithPath: require("--moe-manifest")),
            pdManifest: options["--pd-manifest"].map { URL(fileURLWithPath: $0).standardizedFileURL.resolvingSymlinksInPath() },
            computeUnits: .gpu)
        let configSHA = SHA256.hash(data: try Data(contentsOf: directory.appendingPathComponent("config.json")))
            .map { String(format: "%02x", $0) }.joined()
        guard model.manifestModelDirectory == directory, model.sourceConfigSHA256 == configSHA,
              tokens.count + maximum <= model.capacity else {
            throw NativeCLIError.invalid("Model identity mismatch or prompt/output exceeds exported capacity")
        }
        let loadSeconds = seconds(loadStart)
        let chunk = try model.resolvedPrefillChunkSize(requested: requestedChunk)
        let chunks = options["--compare-prefill"] == "true" ? [1, model.prefillChunkSize, model.prefillChunkSize, 1] : Array(repeating: chunk, count: repeats)
        var runs = [[String: Any]](), first = [Int32](), firstLogits = [Float]()
        for run in chunks.indices {
            if run > 0 { try model.reset() }
            var history = hash.initialHistory, logicalBytes = 0, readSeconds = 0.0
            func embedding(_ input: [Int32]) throws -> [Float] {
                let sequence = input.map(UInt32.init)
                let rows = try hash.rowIDs(previousTokens: history, tokens: sequence)
                let started = DispatchTime.now().uptimeNanoseconds
                let values = try table.readRows(rows)
                readSeconds += seconds(started)
                history = try hash.history(after: sequence, previousTokens: history)
                logicalBytes += rows.count * table.dimension
                return values
            }
            let initialCalls = model.successfulCalls
            let start = DispatchTime.now().uptimeNanoseconds
            var logits = [Float]()
            var position = 0, chunkCount = 0
            while position < tokens.count {
                let count = try model.nextPrefillChunkSize(remaining: tokens.count - position, limit: chunks[run])
                let batch = Array(tokens[position..<position + count])
                logits = try await model.prefill(tokens: batch, pleEmbedding: embedding(batch))
                position += count; chunkCount += 1
                if position % 32 == 0 || position == tokens.count { progress("Prefill \(position)/\(tokens.count), run \(run + 1), chunk \(chunks[run])") }
            }
            let prefillSeconds = seconds(start), prefillCalls = model.successfulCalls - initialCalls
            let prefillGroups = model.predictionMillisecondsByGroup
            let prefillSSDSeconds = readSeconds
            if run == 0 { firstLogits = logits }
            var errorSquared = 0.0, referenceSquared = 0.0, maximumError = 0.0
            for (a, b) in zip(logits, firstLogits) {
                let delta = Double(a) - Double(b)
                errorSquared += delta * delta; referenceSquared += Double(b) * Double(b)
                maximumError = max(maximumError, abs(delta))
            }
            var output = [Int32](), decodeDurations = [Double](), stop = "length"
            for step in 0..<maximum {
                let selected = try greedy(logits, excluding: tokenizer.reservedOutputTokenIDs)
                output.append(selected)
                if tokenizer.eosTokenIDs.contains(selected) { stop = "eos"; break }
                progress("Generated \(output.count): \(try tokenizer.decode(output))")
                if step + 1 < maximum {
                    let started = DispatchTime.now().uptimeNanoseconds
                    logits = try await model.forward(token: selected, pleEmbedding: embedding([selected]))
                    decodeDurations.append(seconds(started))
                }
            }
            if run == 0 { first = output }
            let decodeGroups = model.predictionMillisecondsByGroup.mapValues { $0 }
                .map { key, value in (key, value - (prefillGroups[key] ?? 0)) }
            runs.append(["run": run + 1, "text": try tokenizer.decode(output, skipSpecialTokens: true),
                "generated_token_ids": output, "stop_reason": stop, "reset_tokens_match": output == first,
                "prefill_tokens": tokens.count, "prefill_seconds": prefillSeconds, "prefill_coreai_calls": prefillCalls,
                "prefill_chunk_size": chunks[run], "prefill_chunks": chunkCount,
                "prefill_logits_relative_l2": sqrt(errorSquared / max(referenceSquared, 1e-30)),
                "prefill_logits_max_abs": maximumError,
                "prefill_group_milliseconds": prefillGroups,
                "decode_group_milliseconds": Dictionary(uniqueKeysWithValues: decodeGroups),
                "prefill_ssd_read_seconds": prefillSSDSeconds, "decode_ssd_read_seconds": readSeconds - prefillSSDSeconds,
                "decode_forward_steps": decodeDurations.count, "decode_forward_seconds": decodeDurations.reduce(0, +),
                "decode_step_seconds": decodeDurations, "total_coreai_calls": model.successfulCalls - initialCalls,
                "final_state_offset": model.offset, "ssd_logical_bytes": logicalBytes, "ssd_read_seconds": readSeconds])
            progress("Run \(run + 1): prefill \(prefillSeconds)s, decode \(decodeDurations.reduce(0, +))s, output matches first: \(output == first)")
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
            "prefill_chunk_size": model.prefillChunkSize, "independent_pd_functions": model.usesIndependentPhases,
            "supported_prefill_chunks": model.supportedPrefillChunks,
            "limitations": "Explicit serial phase execution; group times are awaited function wall time, not device kernel profiling. Source BF16 quality equivalence remains unverified."]
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
