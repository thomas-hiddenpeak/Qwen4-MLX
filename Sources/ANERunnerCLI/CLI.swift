import ANERunnerCore
import Foundation

enum CLIError: Error, LocalizedError {
    case usage(String)

    var errorDescription: String? {
        switch self {
        case .usage(let message): message
        }
    }
}

struct Arguments {
    let command: String
    private let options: [String: String]

    init(_ input: [String]) throws {
        command = input.first ?? "help"
        var parsed: [String: String] = [:]
        var index = 1
        while index < input.count {
            let name = input[index]
            guard name.hasPrefix("--"), index + 1 < input.count,
                !input[index + 1].hasPrefix("--"), parsed[name] == nil
            else { throw CLIError.usage("Expected unique --option value pairs; got \(name)") }
            parsed[name] = input[index + 1]
            index += 2
        }
        options = parsed
    }

    func require(_ key: String) throws -> String {
        guard let value = options[key], !value.isEmpty else {
            throw CLIError.usage("Missing \(key)")
        }
        return value
    }

    subscript(_ key: String) -> String? { options[key] }

    func validate(_ allowed: Set<String>) throws {
        let unknown = Set(options.keys).subtracting(allowed)
        guard unknown.isEmpty else {
            throw CLIError.usage("Unknown options: \(unknown.sorted().joined(separator: ", "))")
        }
    }
}

@main
struct RunnerCLI {
    static func mtpDraftHistory(_ args: Arguments) throws -> Int? {
        guard let text = args["--mtp-draft-history"], text != "full" else { return nil }
        guard let value = Int(text), (1...262144).contains(value) else {
            throw CLIError.usage("--mtp-draft-history must be full or 1...262144 (draft head only)")
        }
        return value
    }
    static func main() async {
        do {
            let arguments = try Arguments(Array(CommandLine.arguments.dropFirst()))
            switch arguments.command {
            case "help", "--help", "-h":
                print(help)
            case "inspect":
                try inspect(arguments)
            case "lookup":
                try lookup(arguments)
            case "hash":
                try hash(arguments)
            case "probe-coreml":
                try probeCoreML(arguments)
            case "probe-moe", "route-moe":
                try probeMoE(arguments)
            case "probe-gpu-moe":
                try probeGPUMoE(arguments)
            case "probe-gpu-moe-grouped-gateup":
                try probeGPUMoEGroupedGateUp(arguments)
            case "probe-gpu-moe-verification":
                try probeGPUMoEVerification(arguments)
            case "probe-gpu-sequence":
                try probeGPUSequence(arguments)
            case "probe-gpu-model":
                try probeGPUModel(arguments)
            case "probe-gpu-kv-capacity":
                try probeGPUKVCapacityMechanism(arguments)
            case "probe-gpu-kv-capacity-model":
                try probeGPUKVCapacityModel(arguments)
            case "benchmark-gpu-kv-capacity":
                try benchmarkGPUKVCapacity(arguments)
            case "benchmark-gpu-paged-attention":
                try benchmarkGPUPagedAttention(arguments)
            case "probe-gpu-paged-kv-pool":
                try probeGPUPagedKVPoolMechanism(arguments)
            case "generate-gpu":
                try generateGPU(arguments)
            case "probe-gpu-phase-handoff":
                try probeGPUPhaseHandoff(arguments)
            case "probe-gpu-prefill-attention":
                try probeGPUPrefillAttention(arguments)
            case "probe-gpu-hotspots":
                try probeGPUHotspots(arguments)
            case "probe-gpu-prefix-cache":
                try probeGPUPrefixCache(arguments)
            case "probe-gpu-cache-reliability":
                try probeGPUCacheReliability(arguments)
            case "probe-gpu-cache-timeouts":
                try probeGPUCacheTimeouts(arguments)
            case "probe-gpu-cache-read-admission":
                try probeGPUCacheReadAdmission(arguments)
            case "probe-gpu-conversation-cache":
                try probeGPUConversationCache(arguments)
            case "capture-gpu-moe-prefill":
                try captureGPUMoEPrefill(arguments)
            case "probe-gpu-moe-prefill-tiling":
                try probeGPUMoEPrefillTiling(arguments)
            case "autotune-gpu-moe-prefill":
                try autotuneGPUMoEPrefill(arguments)
            case "probe-gpu-moe-prefill-gateup":
                try probeGPUMoEPrefillGateUp(arguments)
            case "probe-gpu-moe-prefill-expert":
                try probeGPUMoEPrefillExpert(arguments)
            case "probe-gpu-local-scheduler":
                try probeGPULocalScheduler(arguments)
            case "probe-gpu-cooperative-scheduler":
                try probeGPUCooperativeScheduler(arguments)
            case "probe-moe-down-pair-gpu":
                try probeGPUMoEDownPair(arguments)
            case "serve-gpu":
                try serveGPU(arguments)
            case "probe-gpu-session":
                try probeGPUSession(arguments)
            case "probe-gpu-mtp-numerics":
                try probeGPUMTPNumerics(arguments)
            case "probe-gpu-mtp-state":
                try probeGPUMTPState(arguments)
            case "probe-gpu-mtp-release":
                try probeGPUMTPRelease(arguments)
            case "probe-gpu-command-timing":
                try probeGPUCommandTiming(arguments)
            case "probe-gpu-matvec":
                try probeGPUMatvec(arguments)
            case "probe-gpu-verification-qkv":
                try probeGPUVerificationQKV(arguments)
            case "probe-telemetry":
                try probeTelemetry(arguments)
            case "tokenize":
                try gpuTokenize(arguments)
            default:
                throw CLIError.usage("Unknown command: \(arguments.command). Run ane-runner help.")
            }
        } catch {
            FileHandle.standardError.write(Data("ane-runner: \(error.localizedDescription)\n".utf8))
            exit(1)
        }
    }

    static func load(_ args: Arguments) throws -> (URL, ModelManifest, NGramTable) {
        let directory = URL(fileURLWithPath: try args.require("--model-dir"), isDirectory: true)
        let manifest = try ModelManifest.load(from: directory)
        let table = try NGramTable(url: manifest.tableURL(in: directory))
        guard table.scale == manifest.ngramTable.scale else {
            throw CLIError.usage("Table scale differs from config.json")
        }
        return (directory, manifest, table)
    }

    static func inspect(_ args: Arguments) throws {
        try args.validate(["--model-dir", "--output"])
        let (directory, manifest, table) = try load(args)
        let config = manifest.textConfig
        try emit([
            "model_directory": directory.standardizedFileURL.path,
            "model_type": manifest.modelType,
            "layers": config.numHiddenLayers,
            "gated_delta_layers": config.layerTypes.filter { $0 == "linear_attention" }.count,
            "full_attention_layers": config.layerTypes.filter { $0 == "full_attention" }.count,
            "experts_per_layer": config.numExperts,
            "selected_experts_per_token": config.numExpertsPerToken,
            "ngram_table": [
                "row_count": table.rowCount, "dimension": table.dimension,
                "scale": table.scale, "data_offset": table.dataOffset,
                "logical_parameter_count": Int64(table.rowCount) * Int64(table.dimension),
                "read_mode": "requested rows only via pread",
            ],
            "full_model_generation_implemented": true,
            "full_model_generation_command": "generate-gpu",
            "note": "Asset inspection only. Matrix weights were not loaded; no ANE inference is implied.",
        ], to: args["--output"])
    }

    static func lookup(_ args: Arguments) throws {
        try args.validate(["--model-dir", "--rows", "--output"])
        let (_, _, table) = try load(args)
        let pieces = try args.require("--rows").split(separator: ",", omittingEmptySubsequences: false)
        guard !pieces.isEmpty, pieces.count <= 256 else {
            throw CLIError.usage("lookup accepts 1...256 row IDs")
        }
        let rowIDs = try pieces.map { part in
            guard let id = Int(part.trimmingCharacters(in: .whitespaces)), id >= 0 else {
                throw CLIError.usage("Invalid row ID: \(part)")
            }
            return id
        }
        let values = try table.readRows(rowIDs)
        try emit([
            "row_count": table.rowCount, "dimension": table.dimension, "scale": table.scale,
            "row_ids": rowIDs, "values": values,
            "output_dtype": "bfloat16_as_float32",
            "logical_bytes_requested": rowIDs.count * table.dimension,
            "note": "CPU SSD lookup; filesystem cache and physical I/O were not measured.",
        ], to: args["--output"])
    }

    static func hash(_ args: Arguments) throws {
        try args.validate(["--model-dir", "--tokens", "--previous", "--output"])
        let (_, manifest, table) = try load(args)
        let config = manifest.textConfig
        let hashing = try NGramHash(
            unigramVocabularySize: config.vocabularySize,
            ngramSize: config.ngramSize,
            headsPerNGram: config.headsPerNGram,
            vocabularyBase: config.ngramVocabularyBase,
            vocabularyDivisor: config.vocabularyDivisor,
            seed: config.seed ?? 1234, pleLayerIndex: 0, eosTokenID: config.eosTokenID
        )
        guard hashing.totalRows == table.rowCount else {
            throw CLIError.usage("Hash configuration does not match the table row count")
        }
        func tokens(_ value: String) throws -> [UInt32] {
            let parts = value.split(separator: ",", omittingEmptySubsequences: false)
            guard !parts.isEmpty, parts.count <= 256 else {
                throw CLIError.usage("hash accepts 1...256 tokens")
            }
            return try parts.map { part in
                guard let token = UInt32(part.trimmingCharacters(in: .whitespaces)),
                    token < config.vocabularySize else {
                    throw CLIError.usage("Token outside vocabulary: \(part)")
                }
                return token
            }
        }
        let input = try tokens(args.require("--tokens"))
        let previous = try args["--previous"].map(tokens) ?? hashing.initialHistory
        let rowIDs = try hashing.rowIDs(previousTokens: previous, tokens: input)
        guard rowIDs.allSatisfy({ $0 >= 0 && $0 < table.rowCount }) else {
            throw CLIError.usage("Hash produced a row outside this model's table")
        }
        try emit([
            "token_ids": input, "previous_tokens": previous, "row_ids": rowIDs,
            "heads_per_token": config.headsPerNGram * (config.ngramSize - 1),
            "hash_seed": config.seed ?? 1234, "ple_table_index": 0,
            "note": "Token IDs must use this model's tokenizer; use tokenize for text input and generate-gpu for full-model inference.",
        ], to: args["--output"])
    }

    static func emit(_ object: Any, to path: String?) throws {
        var data = try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
        data.append(0x0A)
        if let path {
            let url = URL(fileURLWithPath: path)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: url, options: .atomic)
        } else {
            FileHandle.standardOutput.write(data)
        }
    }

    static func probeCoreML(_ args: Arguments) throws {
        try args.validate(["--model", "--fixture", "--output"])
        let runner = try CoreMLBlockRunner(
            modelURL: URL(fileURLWithPath: args.require("--model"))
        )
        let fixture = try CoreMLBlockFixture.load(
            from: URL(fileURLWithPath: args.require("--fixture"))
        )
        let report = try runner.run(fixture: fixture, warmups: 3, runs: 5)
        let data = try JSONEncoder().encode(report)
        let object = try JSONSerialization.jsonObject(with: data)
        try emit(object, to: args["--output"])
    }

    static func probeMoE(_ args: Arguments) throws {
        try args.validate(["--manifest", "--fixture", "--output", "--precision", "--compute-units", "--cache-experts", "--warmups", "--runs", "--expert-concurrency"])
        guard let precision = ExpertRouterPrecision(rawValue: args["--precision"] ?? "bfloat16Boundaries"),
            let units = MoEComputeUnits(rawValue: args["--compute-units"] ?? "cpuAndNeuralEngine"),
            let capacity = Int(args["--cache-experts"] ?? "64"),
            let concurrency = Int(args["--expert-concurrency"] ?? "1"),
            let warmups = Int(args["--warmups"] ?? "1"), let runs = Int(args["--runs"] ?? "3") else {
            throw CLIError.usage("Invalid precision, compute-units, cache-experts, warmups, or runs")
        }
        let runner = try MoEBlockRunner(manifestURL: URL(fileURLWithPath: args.require("--manifest")),
                                       precision: precision, computeUnits: units, maximumResidentExperts: capacity, expertConcurrency: concurrency)
        let fixture = try CoreMLBlockFixture.load(from: URL(fileURLWithPath: args.require("--fixture")))
        if args.command == "route-moe" {
            let route = try runner.route(fixture: fixture)
            try emit(["token_count": route.tokenCount, "expert_count": route.expertCount,
                      "top_k": route.topK, "precision": route.precision.rawValue,
                      "expert_ids": route.expertIDs, "weights": route.weights, "logits": route.logits,
                      "required_expert_ids": Set(route.expertIDs).sorted()], to: args["--output"])
        } else {
            let report = try runner.run(fixture: fixture, warmups: warmups, runs: runs)
            try emit(JSONSerialization.jsonObject(with: JSONEncoder().encode(report)), to: args["--output"])
        }
    }

    static let help = """
    Independent Swift Model Runner — Metal GPU inference and Core ML experiments

    generate-gpu --model-dir PATH --prompt TEXT [--max-tokens 32] [--output report.json]
        Generate text with all 48 layers on Metal GPU and requested n-gram rows from SSD.
        Input: --prompt TEXT or --tokens-file PATH; --raw-prompt true|false
        Options: --prefill-chunk 416 --context 4096 --repeat 1 --ssd-workers 1
        SSD prefetch: --ssd-prefetch off|nextChunk (default nextChunk); --ssd-prefetch-order CSV for interleaved trials
        Native MTP: --mtp-depth 0...4 (default 0); --mtp-order CSV; --mtp-verification scalar|batched|batchedRounded|batchedCaptured|batchedScalarMoE|batchedScalarLinear|batchedTokenMoE; --mtp-verification-order CSV
        Verification defaults to scalar (slower output oracle). All batched modes are experimental.
        Independent scheduling: --prefill-eval-layers 4 --verify-eval-layers 4; --decode-mode affects decode only.
        Prefill attention experiment: --prefill-attention reference|fusedQSA (default reference); preserves QSA visibility.
        Optional MoE kernels: --prefill-moe-config PATH; requires matching ANERUNNER_GATEUP_LIBRARY.
        Phase handoff check: probe-gpu-phase-handoff --model-dir PATH --output NEW.json
        Local queues: probe-gpu-local-scheduler --model-dir PATH --output NEW.json
            [--tokens-file PATH --max-tokens 128 --suite mixed --golden-report PATH]
        Chunk/round scheduling: probe-gpu-cooperative-scheduler --model-dir PATH
            --tokens-file AGENT_11K.json --output NEW.json [--golden-report PATH]
            [--decode-burst 4] (1...64 steps; includes first-token publication)
        MoE down scheduling probe: probe-moe-down-pair-gpu --model-dir PATH
            --pair-library ABSOLUTE_DYLIB --output NEW.json [--shared-elementwise reference|fused]
        Experimental loopback service: serve-gpu --model-dir PATH [--port 11236]
            [--kv-append-mode reference|capacity256] (default reference; AR decode only)
            [--max-connections 8 --max-body-bytes 262144 --output-buffer-bytes 65536]
            [--prefix-cache-bytes 536870912 --prefix-cache-entries 8] (bytes 0 disables)
            [--prefix-cache-ttl-seconds 86400 --state-budget-bytes 4294967296]
            [--prefix-cache-directory PRIVATE_DIRECTORY] (SSD cache is off unless configured)
            [--prefix-cache-disk-bytes 8589934592 --prefix-cache-disk-entries 32]
            SSD limits require a directory and an enabled RAM cache; parent directory must exist.
            State budget bounds logical request/cache/copy reservations, not process RSS.
            Greedy text/tool chat and SSE; AR default, explicit experimental mtp_depth=2 for text only.
            Function tools support tool_choice auto|none and ordered tool result history.
        Experimental draft prompt history: --mtp-draft-history full|1024 (default full); target context remains complete.
        Precision: --prefill-accumulation reference|float32 (default: reference)
        Fused causal prefill: enabled by default (M5, sequence > 8; QSA unchanged); ANERUNNER_FUSED_PREFILL=0 disables it
        Blocked GDN prefill experiment: ANERUNNER_BLOCKED_GDN=1 (sequence >= 64)
        Prefill synchronization experiment: ANERUNNER_PREFILL_EVAL_LAYERS=1...48 (default 4)
        Decode: --decode-mode reference|scalar|elementwise|projections|all (default: reference)
        Residency: --wired-policy disabled|fit (default: disabled; setup excluded from request timing)
        Interleaved trials: --decode-order CSV and/or --wired-order CSV (one value per repetition, at most 10)
        GDN tuning library: --gdn-gemv-mode reference|bm4|rows4|bm2|bm1|gemm|gemmSplit|prefetch4|prefetch4Vector or --gdn-gemv-order CSV
        Profiling: --profile-stages disabled|hostBodyOnly|synchronizedStages
        Optional stage filter: --profile-phase prefill|decode|verification (requires profiling; default all phases)
        Routing diagnostic: --capture-verification-routing true (D2 scalar-linear, one request, at most 128 output tokens; profiling disabled)
        System telemetry: --telemetry-dir NEW_PATH [--telemetry-interval-ms 200]
        Diagnostic MLX build only: --gpu-command-timing-output NEW_JSON_PATH
    probe-gpu-kv-capacity --diagnostics-library ABSOLUTE_DYLIB --output NEW_NDJSON [--eval sync|async]
        Model-free Swift capacity allocation/ARC diagnostic; default eval sync. Run in an exclusive GPU window.
    probe-gpu-paged-kv-pool --library ABSOLUTE_DYLIB --output NEW_NDJSON [--eval sync|async]
        Model-free physical page sharing, immutable tails, direct SDPA and GPU lifetime checks; diagnostic timing only.
    probe-gpu-kv-capacity-model --model-dir PATH --tokens-file AGENT_11K_JSON --output NEW_REPORT_JSON
        Actual generator correctness, RAM restore, workspace fallback and callback cancellation; diagnostic timing only.
    benchmark-gpu-kv-capacity --model-dir PATH --tokens-file AGENT_11K_JSON --output NEW_REPORT_JSON
        [--max-tokens 16] [--context 16384] [--order abba|baab] [--warmup true|false]
        Four generator trials with observers disabled; capacity remains an explicit experimental request policy.
    benchmark-gpu-paged-attention --library ABSOLUTE_DYLIB --model-dir PATH --tokens-file AGENT_11K_JSON --output NEW_REPORT_JSON
        [--max-tokens 16] [--context 16384] [--order abba|baab] [--warmup true|false] [--state-check false]
        Stock versus identity page-table SDPA, both capacity256. State check requires ABBA, no warmup and at most 16 outputs.
    probe-gpu-session --model-dir PATH [--tokens-file PATH] --output NEW_REPORT_JSON
        Exercise typed generation API: invalid input, cancellation, callback failure, busy rejection, fresh-request recovery and scalar MTP.
    probe-gpu-mtp-state --model-dir PATH --output NEW_REPORT_JSON [--verification POLICY]
        Exercise real-weight MTP cancellation, budget and stop branches.
    probe-gpu-mtp-numerics --model-dir PATH --reference-report PATH --positions CSV --output NEW_REPORT_JSON [--verify-scalar-linear true]
        Compare S1 and S2 from identical AR checkpoints; not a performance benchmark.
    probe-gpu-mtp-release --model-dir PATH --long-tokens-file PATH --output NEW_REPORT_JSON [--depth 1] [--verification batchedScalarLinear]
        Seven fixed short, QSA-edge and long inputs, with fresh AR/candidate/candidate/AR sessions.
    tokenize --model-dir PATH --prompt TEXT [--chat true|false] [--output report.json]
        Encode text with the model's native byte BPE tokenizer; --chat applies the no-thinking chat template.
    probe-telemetry --telemetry-dir NEW_PATH [--seconds 2] [--output report.json]
        Check the optional system sampler lifecycle while idle; no model or bandwidth benchmark.
    probe-gpu-command-timing --gpu-command-timing-output NEW_JSON_PATH [--output report.json]
        Validate the isolated native timing hook with tiny matrices; no model load.
    probe-gpu-verification-qkv --model-dir PATH --output NEW_JSON_PATH
        Four real QKV weights, synthetic S2/S3 inputs; fixed 3 warm and 12 AB/BA pairs per layer.
    probe-gpu-matvec --model-dir PATH --output NEW_JSON_PATH [--repeats 16] [--gpu-command-timing-output NEW_JSON_PATH]
        Diagnose real BF16 GDN/head matrix-vector kernels with synthetic fixed inputs; not full-model inference.
        GDN tuning library: --gdn-gemv-order reference,bm4,rows4,bm2,bm1,gemm,gemmSplit,prefetch4,prefetch4Vector
        Exploratory GEMM rounding check: --allow-rounding true (default false; records the measured error)
    probe-gpu-model --model-dir PATH --tokens-file PATH --capture-output PATH [--layers 1] [--output report.json]
        Capture real hidden states from the first few GPU layers for numerical validation.
    probe-gpu-prefill-attention --model-dir PATH --tokens-file PATH --output NEW_REPORT_JSON --golden-report PATH [--order reference,fusedQSA] [--max-tokens 128] [--mtp-depth 0|2]
    probe-gpu-hotspots --model-dir PATH --tokens-file PATH --golden-report PATH --output NEW_REPORT_JSON [--max-tokens 128] [--detail attention|moe|tiling|moe-fusion|moe-gateup|moe-expert|moe-composed] [--moe-config PATH] [--baseline-moe-config PATH] [--ab-order ABBA|BAAB]
    probe-gpu-prefix-cache --model-dir PATH --tokens-file AGENT_11K.json --output NEW.json
        [--suite long|boundaries|lifecycle|all --max-tokens 128 --state-readback true|false]
    probe-gpu-cache-reliability --model-dir PATH --tokens-file AGENT_11K.json
        --cache-directory PRIVATE_DIRECTORY --output NEW.json --mode populate|restore|corrupt|lifecycle|pressure
        [--oracle-report POPULATE_REPORT.json] (required for restore/corrupt)
    probe-gpu-cache-timeouts --model-dir PATH --tokens-file AGENT_11K.json
        --cache-directory PRIVATE_DIRECTORY --output NEW.json
    probe-gpu-cache-read-admission --model-dir PATH --tokens-file AGENT_11K.json
        --cache-directory PRIVATE_DIRECTORY --output NEW.json
    probe-gpu-conversation-cache --model-dir PATH --system-file AGENT_11K.txt
        --cache-directory PRIVATE_DIRECTORY --output NEW.json
    capture-gpu-moe-prefill --model-dir PATH --tokens-file PATH --golden-report PATH --output NEW_MANIFEST_JSON --fixture-dir NEW_DIRECTORY
    probe-gpu-moe-prefill-tiling --model-dir PATH --manifest PATH --output NEW_REPORT_JSON
    autotune-gpu-moe-prefill --model-dir PATH --manifest PATH --output NEW_REPORT_JSON --config-output NEW_CONFIG_JSON
    probe-gpu-moe-prefill-gateup --model-dir PATH --manifest PATH --output NEW_REPORT_JSON --config-output NEW_CONFIG_JSON
    probe-gpu-moe-prefill-expert --model-dir PATH --manifest PATH --output NEW_REPORT_JSON --config-output NEW_CONFIG_JSON [--suite expert|composition]

    probe-gpu-moe --model-dir PATH --fixture PATH [--output report.json]
    probe-gpu-moe-grouped-gateup --model-dir PATH --route-cases PATH --output NEW_REPORT_JSON
        Operator-only S3 forced-route comparison: scalar loop, token-axis, grouped gate/up.
    probe-gpu-moe-verification --model-dir PATH --output NEW_REPORT_JSON [--layer 0] [--warmups 3] [--runs 12]
        Validate and time the GPU router, selected experts and shared expert.
    probe-gpu-sequence --model-dir PATH --fixture PATH [--output report.json]
        Validate GPU recurrent and attention state with captured fixture inputs.
        --blocked-only true compares blocked/scalar recurrence on replayed inputs, with quick kernel timings.

    inspect --model-dir PATH [--output report.json]
        Inspect the Qwen3.8 model and SSD table without loading matrix weights.
    lookup --model-dir PATH --rows 0,1,2 [--output rows.json]
        Read up to 256 FP8 table rows and return the author's BF16-rounded values.
    hash --model-dir PATH --tokens 1,2,3 [--previous 248044,248044] [--output ids.json]
        Compute n-gram row IDs with the model's EOS and token-history semantics.
    probe-coreml --model PATH --fixture PATH [--output report.json]
        Execute a Core ML block with CPU + Neural Engine allowed; CPU fallback is possible.
    route-moe --manifest PATH --fixture PATH [--precision bfloat16Boundaries|float32] [--output report.json]
        Route real token activations across every expert without loading expert models.
    probe-moe --manifest PATH --fixture PATH [--output report.json]
        Execute dynamic top-K experts, the shared expert and weighted merge.
        Options: --compute-units cpuAndNeuralEngine|cpuOnly --cache-experts 64 --expert-concurrency 1 --warmups 1 --runs 3

    Full text generation uses generate-gpu. Core ML probes validate individual blocks only.
    """
}
