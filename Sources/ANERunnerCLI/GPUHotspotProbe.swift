import ANERunnerGPU
import CMLX
import CryptoKit
import Foundation

extension RunnerCLI {
    /// Stage diagnostics and the optional unrecorded tiling A/B are distinct
    /// modes; both use real producer/consumer handoff on one loaded model.
    static func probeGPUHotspots(_ args: Arguments) throws {
        try args.validate(["--model-dir", "--tokens-file", "--golden-report", "--output", "--max-tokens", "--detail", "--moe-config", "--ab-order"])
        let output = try args.require("--output")
        let detail = args["--detail"] ?? "attention"
        guard !FileManager.default.fileExists(atPath: output),
              ["attention", "moe", "tiling", "moe-fusion", "moe-gateup", "moe-expert"].contains(detail),
              let maximum = Int(args["--max-tokens"] ?? "128"), (1...256).contains(maximum) else {
            throw CLIError.usage("Hotspot probe requires new --output, --max-tokens 1...256 and --detail attention|moe|tiling|moe-fusion|moe-gateup|moe-expert")
        }
        let isTiling = detail == "tiling"
        let isMoEFusion = detail == "moe-fusion"
        let isExpert = detail == "moe-expert"
        let isGateUp = detail == "moe-gateup" || isExpert
        let hasMoEConfiguration = isMoEFusion || isGateUp
        guard hasMoEConfiguration == (args["--moe-config"] != nil) else {
            throw CLIError.usage("--detail moe-fusion|moe-gateup|moe-expert requires --moe-config; other details do not accept it")
        }
        let isUnrecordedAB = isTiling || hasMoEConfiguration
        let measuredOrder = args["--ab-order"] ?? "ABBA"
        guard ["ABBA", "BAAB"].contains(measuredOrder), isUnrecordedAB || args["--ab-order"] == nil else {
            throw CLIError.usage("--ab-order ABBA|BAAB is only available for unrecorded A/B probes")
        }
        let measuredModes = measuredOrder.map { $0 == "A" ? "baseline" : "candidate" }
        let order = isUnrecordedAB ? ["baseline", "candidate"] + measuredModes
            : ["baseline", "profiled", "profiled", "baseline"]
        var trials = [[String: Any]]()
        var report: [String: Any] = [
            "schema": "qwen38-gpu-hotspot-probe-v1", "complete": false, "passed": false,
            "order": order, "max_tokens": maximum, "mtp_enabled": false, "mtp_depth": 0,
            "detail": detail,
            "full_model_instances": 1, "generator_instances": 2,
            "clock": "mach_absolute_time_nanoseconds", "logit_finiteness_checked": false,
            "notes": [
                "Each trial uses producer.prefill(request), then consumer.decode(ready) on one shared model and executor.",
                hasMoEConfiguration ? "MoE A/B disables all stage recording. Each trial creates a request with nil baseline or the saved candidate prefill MoE configuration; decode remains reference."
                    : isTiling ? "Tiling A/B disables all stage recording. Prefill alone selects BM0 (baseline) or BM16 (candidate); BM0 is restored before every decode."
                    : "Only the two profiled prefills enable synchronized stage recording; every decode and both baseline prefills disable it.",
                hasMoEConfiguration ? "The first baseline/candidate pair is warmup, followed by measured \(measuredOrder). The saved configuration is decoded and validated before model loading."
                    : isTiling ? "The first baseline/candidate pair is warmup. Dispatch counts establish path selection, not physical weight reads or bandwidth."
                    : "Stage synchronization changes graph evaluation, overlap and allocation reuse. These timings are not GPU-only time or undisturbed throughput.",
                isExpert ? "Expert variants are request-local plugin calls. Nil is reference; variant1 is the prior fused path, variant2/3 add a shared expert plan and optionally grouped down. Plugin plan/gate/down and host graph counters are checked independently; all tracked decode counters must stay zero. Old native selectors and reduction fusion remain off."
                    : isGateUp ? "Gate/up variants are request-local plugin calls: nil is reference, 0/1 are candidates. Plugin and stock base MLX hashes must match; old native selectors and reduction fusion remain off. Plugin dispatches and host gate/up construction are checked separately; down has no counter and no down-dispatch count is inferred. Decode and reduction deltas must stay zero."
                    : isMoEFusion ? "The saved native configuration requires a matching library SHA. Native counters count encoded matmuls; reduction counters count host graph construction. Neither measures DRAM traffic. All decode deltas must remain zero."
                    : isTiling ? "Prefill dispatch deltas must show candidate BM16 or baseline BM32 with no BM16; all tracked counts must remain unchanged during decode."
                    : "Raw phase and position fields locate stages within the prompt. Stage durations do not measure physical DRAM or SSD traffic.",
                "Complete output IDs and finish reason must match the AR golden and first baseline. AR final offset is prompt count plus output count minus one, including EOS.",
                "Visible metric finiteness is checked; token-only public APIs do not establish internal logit or hidden-tensor finiteness.",
                "Any runtime, profiling, output, state-offset, callback or handoff check failure preserves available data and exits nonzero.",
                "First-use compilation and SSD caching can affect early trials. This bounded experiment does not by itself establish a stable speedup."
            ]]
        func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
        func elapsed(_ start: UInt64) -> Double { Double(now() - start) * 1e-9 }
        func hash(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
        func object<T: Encodable>(_ value: T) throws -> Any {
            try JSONSerialization.jsonObject(with: JSONEncoder().encode(value))
        }
        func countDelta(_ after: [String: UInt64], _ before: [String: UInt64]) throws -> [String: UInt64] {
            guard !before.isEmpty, Set(after.keys) == Set(before.keys) else {
                throw CLIError.usage("Tiling dispatch counter keys changed or are unavailable")
            }
            var delta = [String: UInt64]()
            for (key, value) in after {
                guard let initial = before[key], value >= initial else {
                    throw CLIError.usage("Tiling dispatch counter decreased: \(key)")
                }
                delta[key] = value - initial
            }
            return delta
        }
        func nativeDelta(_ after: [UInt64], _ before: [UInt64]) throws -> [UInt64] {
            guard after.count == 5, before.count == 5,
                  zip(after, before).allSatisfy({ $0.0 >= $0.1 }) else {
                throw CLIError.usage("Invalid or decreasing native autotune counters")
            }
            return zip(after, before).map { $0.0 - $0.1 }
        }
        func firstDifference(_ actual: [Int32], _ expected: [Int32]) -> Any {
            let common = min(actual.count, expected.count)
            guard let index = (0..<common).first(where: { actual[$0] != expected[$0] })
                ?? (actual.count == expected.count ? nil : common) else { return NSNull() }
            return ["index": index, "actual": index < actual.count ? actual[index] as Any : NSNull(),
                    "expected": index < expected.count ? expected[index] as Any : NSNull(),
                    "actual_count": actual.count, "expected_count": expected.count] as [String: Any]
        }
        func write(_ complete: Bool = false) throws {
            let passed = complete && trials.count == order.count && trials.allSatisfy { $0["correctnessPassed"] as? Bool == true }
            report["complete"] = complete; report["passed"] = passed; report["correctnessPassed"] = passed
            report["trials"] = trials
            try emit(report, to: output)
        }
        try write()
        do {
            let directory = URL(fileURLWithPath: try args.require("--model-dir")).standardizedFileURL.resolvingSymlinksInPath()
            let inputURL = URL(fileURLWithPath: try args.require("--tokens-file")).standardizedFileURL
            let inputData = try Data(contentsOf: inputURL)
            let tokens = try JSONDecoder().decode([Int32].self, from: inputData)
            guard (10_000...12_000).contains(tokens.count) else {
                throw CLIError.usage("Hotspot probe expects a real 10k...12k token prompt")
            }
            let configuration = try QwenConfiguration(modelDirectory: directory)
            let context = max(16_384, tokens.count + maximum)
            // Match generation's held-final-prompt-token convention exactly.
            let promptChunkLengths = stride(from: 0, to: tokens.count - 1, by: 416)
                .map { min(416, tokens.count - 1 - $0) } + [1]
            // The sorted RHS NAX path needs at least four assignments/expert.
            // Shorter remainder chunks retain the other stock QMM dispatchers.
            let expectedNativePrefillCalls = promptChunkLengths.filter { $0 * 10 / 512 >= 4 }.count * configuration.layerCount * 3
            struct NativeSelection: Decodable {
                let native_configuration: Int
                let native_library_sha256: String
            }
            struct GateUpSelection: Decodable {
                let gateup_plugin_sha256: String
                let base_mlx_sha256: String
            }
            let moeConfiguration: GPUMoEPrefillConfiguration?
            let nativeSelection: NativeSelection?
            let gateUpSelection: GateUpSelection?
            if hasMoEConfiguration {
                let configURL = URL(fileURLWithPath: try args.require("--moe-config")).standardizedFileURL.resolvingSymlinksInPath()
                let configData = try Data(contentsOf: configURL)
                let decoded = try JSONDecoder().decode(GPUMoEPrefillConfiguration.self, from: configData)
                try decoded.validated()
                if isGateUp {
                    try MoEGateUpProbeSupport.requireStockSelectors()
                    guard decoded.threadgroups.isEmpty else { throw CLIError.usage("Gate/up isolation requires an empty reduction table") }
                    if !isExpert {
                        guard decoded.gateUpVariant == nil || (0...1).contains(decoded.gateUpVariant!),
                              decoded.groupedDown != true else { throw CLIError.usage("Grouped expert configurations require --detail moe-expert") }
                    }
                    gateUpSelection = try JSONDecoder().decode(GateUpSelection.self, from: configData)
                    nativeSelection = nil
                } else {
                    guard decoded.gateUpVariant == nil, decoded.groupedDown != true else { throw CLIError.usage("Use moe-gateup or moe-expert for plugin configurations") }
                    let selection = try JSONDecoder().decode(NativeSelection.self, from: configData)
                    guard (0...4).contains(selection.native_configuration), selection.native_library_sha256.count == 64,
                          selection.native_library_sha256.allSatisfy({ $0.isHexDigit }) else {
                        throw CLIError.usage("Invalid saved native MoE configuration or library SHA256")
                    }
                    nativeSelection = selection; gateUpSelection = nil
                }
                moeConfiguration = decoded
                report["moe_configuration"] = ["path": configURL.path, "source_sha256": hash(configData),
                    "source_json": try JSONSerialization.jsonObject(with: configData), "effective_configuration": try object(decoded)]
                report["prefill_chunk_lengths"] = promptChunkLengths
                if isMoEFusion { report["expected_native_prefill_calls"] = expectedNativePrefillCalls }
            } else { moeConfiguration = nil; nativeSelection = nil; gateUpSelection = nil }
            func makeRequest(candidate: Bool) -> QwenGenerationRequest {
                QwenGenerationRequest(tokens: tokens, maxTokens: maximum,
                    contextLimit: context, prefillChunk: 416, mtpDepth: 0,
                    decodeMode: .reference, prefillAttention: .reference,
                    prefillMoEConfiguration: candidate ? moeConfiguration : nil)
            }
            try makeRequest(candidate: false).validate(configuration: configuration)
            if hasMoEConfiguration { try makeRequest(candidate: true).validate(configuration: configuration) }
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
                  golden.generated_token_ids.count <= maximum,
                  golden.generated_token_ids.allSatisfy({ $0 >= 0 && Int($0) < configuration.vocabularySize }),
                  ["eos", "length"].contains(golden.finish_reason),
                  golden.finish_reason != "length" || golden.generated_token_ids.count == maximum else {
                throw CLIError.usage("Golden must contain matching complete prompt/budget and explicit AR trial; no prefix truncation is used")
            }
            let expected = golden.generated_token_ids, expectedFinish = golden.finish_reason
            let executable = URL(fileURLWithPath: CommandLine.arguments[0]).standardizedFileURL.resolvingSymlinksInPath()
            report["provenance"] = ["command": CommandLine.arguments, "executable": executable.path,
                "executable_sha256": hash(try Data(contentsOf: executable)),
                "config_sha256": hash(try Data(contentsOf: directory.appendingPathComponent("config.json")))]
            report["model_directory"] = directory.path
            report["input"] = ["path": inputURL.path, "sha256": hash(inputData), "token_count": tokens.count, "token_ids": tokens]
            report["golden"] = ["path": goldenURL.path, "sha256": hash(goldenData), "token_ids": expected,
                "finish_reason": expectedFinish, "max_tokens": maximum]
            report["configuration"] = ["context_limit": context, "prefill_chunk": 416,
                "prefill_evaluate_every_layers": 4, "decode_mode": "reference",
                "prefill_attention": "reference", "prefill_accumulation": "reference",
                "profiler_mode": isUnrecordedAB ? "disabled" : "synchronizedStages", "attention_breakdown": detail == "attention",
                "moe_breakdown": detail == "moe", "maximum_records": 32_768,
                "tiling_warmup_trials": isTiling ? 2 : 0, "moe_fusion_warmup_trials": isMoEFusion ? 2 : 0,
                "gateup_warmup_trials": isGateUp ? 2 : 0]
            report["environment"] = Dictionary(uniqueKeysWithValues:
                ["ANERUNNER_FUSED_PREFILL", "ANERUNNER_BLOCKED_GDN", "ANERUNNER_PREFILL_EVAL_LAYERS",
                 "ANERUNNER_MOE_QMM_CONFIG", "ANERUNNER_MOE_QMM_BM", "ANERUNNER_GATEUP_LIBRARY"]
                    .map { ($0, ProcessInfo.processInfo.environment[$0] ?? "unset") })
            try write()
            var previousCache = 0
            try MX.check(mlx_set_cache_limit(&previousCache, 256 * 1024 * 1024), "hotspot probe cache limit")
            defer { var ignored = 0; _ = mlx_set_cache_limit(&ignored, previousCache) }
            let tilingRuntime: MoEPrefillTilingRuntime?
            if isTiling { tilingRuntime = try MoEPrefillTilingRuntime() }
            else { tilingRuntime = nil }
            defer { if let tilingRuntime { try? tilingRuntime.setBM(0) } }
            if let tilingRuntime {
                try tilingRuntime.setBM(0)
                report["tiling_library_path"] = tilingRuntime.libraryPath
                try write()
            }
            let autotuneRuntime: MoEPrefillAutotuneRuntime?
            if let nativeSelection {
                let runtime = try MoEPrefillAutotuneRuntime()
                guard runtime.librarySHA256.lowercased() == nativeSelection.native_library_sha256.lowercased() else {
                    throw CLIError.usage("Loaded native MoE library SHA256 differs from saved autotune configuration")
                }
                autotuneRuntime = runtime
            } else { autotuneRuntime = nil }
            defer { if let autotuneRuntime { try? autotuneRuntime.select(0) } }
            if let autotuneRuntime {
                try autotuneRuntime.select(0)
                report["native_library"] = ["path": autotuneRuntime.libraryPath, "sha256": autotuneRuntime.librarySHA256,
                    "matches_saved_configuration": true]
                try write()
            }
            let gateUpPlugin: GPUMoEPrefillGateUp?
            if let gateUpSelection {
                let plugin = try GPUMoEPrefillGateUp(variant: 0)
                let pluginHash = try MoETilingBytes.hash(URL(fileURLWithPath: plugin.libraryPath))
                if isExpert {
                    guard plugin.dispatchCounts().count == 4, plugin.groupedDispatchCounts().count == 4 else {
                        throw CLIError.usage("Expert full PD probe requires plugin API v2 counters")
                    }
                }
                let base = try MoEGateUpProbeSupport.baseMLX()
                guard pluginHash == gateUpSelection.gateup_plugin_sha256,
                      base["loaded_sha256"] == gateUpSelection.base_mlx_sha256 else {
                    throw CLIError.usage("Gate/up plugin or stock base MLX differs from the selected configuration")
                }
                gateUpPlugin = plugin
                report["gateup_plugin"] = ["path": plugin.libraryPath, "sha256": pluginHash, "matches_saved_configuration": true]
                report["base_mlx"] = base
                try write()
            } else { gateUpPlugin = nil }
            let profiler = try GPUProfiler(mode: isUnrecordedAB ? .disabled : .synchronizedStages, maximumRecords: 32_768,
                attentionBreakdown: detail == "attention", moeBreakdown: detail == "moe")
            try profiler.setRecordingEnabled(false)
            defer { try? profiler.setRecordingEnabled(false) }
            let loadStart = now()
            let model = try QwenModel(modelDirectory: directory, profiler: profiler) { current, total in
                if current % 8 == 0 || current == total {
                    FileHandle.standardError.write(Data("Hotspot probe: loaded layer \(current)/\(total)\n".utf8))
                }
            }
            report["model_load_seconds"] = elapsed(loadStart)
            report["loaded_memory"] = try MX.memory()
            let producer = try QwenGenerator(model: model), consumer = try QwenGenerator(model: model)
            var baseline: QwenGenerationResult?
            for (index, name) in order.enumerated() {
                let profiled = name == "profiled"
                let selectedBM = name == "candidate" ? 16 : 0
                let selectedNative = name == "candidate" ? nativeSelection?.native_configuration ?? 0 : 0
                let request = makeRequest(candidate: name == "candidate")
                let expectedReductionCalls = promptChunkLengths.filter {
                    request.prefillMoEConfiguration?.threadgroupSize(tokenCount: $0) != nil
                }.count * configuration.layerCount
                let selectedGateUp = request.prefillMoEConfiguration?.gateUpVariant
                let effectiveVariants = promptChunkLengths.compactMap {
                    request.prefillMoEConfiguration?.effectiveGateUpVariant(tokenCount: $0)
                }
                let expectedGateUpCalls = effectiveVariants.count * configuration.layerCount
                let expectedGroupedDownCalls = promptChunkLengths.filter {
                    request.prefillMoEConfiguration?.usesGroupedDown(tokenCount: $0) == true
                }.count * configuration.layerCount
                var expectedGroupedCounts = [UInt64](repeating: 0, count: 4)
                for length in promptChunkLengths {
                    if let variant = request.prefillMoEConfiguration?.effectiveGateUpVariant(tokenCount: length), variant >= 2 {
                        expectedGroupedCounts[variant - 2] += UInt64(configuration.layerCount)
                        if request.prefillMoEConfiguration?.usesGroupedDown(tokenCount: length) == true {
                            expectedGroupedCounts[variant] += UInt64(configuration.layerCount)
                        }
                    }
                }
                var trial: [String: Any] = ["index": index, "mode": name, "complete": false,
                    "correctnessPassed": false, "prefill_profiling_enabled": profiled, "decode_profiling_enabled": false]
                if hasMoEConfiguration {
                    trial["warmup"] = index < 2
                    trial["prefill_moe_configuration"] = name == "candidate" ? try object(moeConfiguration!) : NSNull()
                    if isMoEFusion { trial["selected_native_configuration"] = selectedNative }
                    trial["expected_prefill_reduction_calls"] = expectedReductionCalls
                }
                if isGateUp {
                    trial["selected_gateup_variant"] = selectedGateUp.map { $0 as Any } ?? NSNull()
                    trial["expected_prefill_gateup_calls"] = expectedGateUpCalls
                    if isExpert {
                        trial["selected_grouped_down"] = request.prefillMoEConfiguration?.groupedDown == true
                        trial["expected_prefill_grouped_down_calls"] = expectedGroupedDownCalls
                        trial["expected_grouped_prefill_count_deltas"] = expectedGroupedCounts
                        trial["grouped_counter_order"] = ["plan32", "plan16", "down32", "down16"]
                    }
                }
                if let tilingRuntime {
                    trial["selected_prefill_bm"] = selectedBM
                    trial["tiling_library_path"] = tilingRuntime.libraryPath
                    trial["warmup"] = index < 2
                }
                var stage = "prefill", callbacks = [Int32]()
                do {
                    try profiler.reset()
                    try profiler.setRecordingEnabled(profiled)
                    FileHandle.standardError.write(Data("Hotspot probe: trial \(index) \(name)\n".utf8))
                    if let tilingRuntime { try tilingRuntime.setBM(selectedBM) }
                    if let autotuneRuntime { try autotuneRuntime.select(selectedNative) }
                    let nativeBeforePrefill = autotuneRuntime?.counts()
                    let reductionBeforePrefill = model.prefillMoEReductionCalls
                    let gateUpBeforePrefill = gateUpPlugin?.dispatchCounts()
                    let gateUpHostBeforePrefill = model.prefillMoEGateUpCalls
                    let groupedBeforePrefill = isExpert ? gateUpPlugin?.groupedDispatchCounts() : nil
                    let groupedHostBeforePrefill = model.prefillMoEGroupedDownCalls
                    if let groupedBeforePrefill {
                        trial["grouped_prefill_counts_before"] = groupedBeforePrefill
                        trial["grouped_down_host_calls_before_prefill"] = groupedHostBeforePrefill
                    }
                    if let gateUpBeforePrefill {
                        trial["gateup_prefill_counts_before"] = gateUpBeforePrefill
                        trial["gateup_host_calls_before_prefill"] = gateUpHostBeforePrefill
                        trial["prefill_reduction_calls_before"] = reductionBeforePrefill
                    }
                    if let nativeBeforePrefill {
                        trial["native_prefill_counts_before"] = nativeBeforePrefill
                        trial["prefill_reduction_calls_before"] = reductionBeforePrefill
                    }
                    let countsBeforePrefill = tilingRuntime?.counts()
                    if let countsBeforePrefill { trial["prefill_counts_before"] = countsBeforePrefill }
                    let prefillStart = now()
                    let ready = try producer.prefill(request)
                    let prefillWall = elapsed(prefillStart)
                    defer { ready.discard() }
                    let nativeAfterPrefill = autotuneRuntime?.counts()
                    let reductionAfterPrefill = model.prefillMoEReductionCalls
                    let gateUpAfterPrefill = gateUpPlugin?.dispatchCounts()
                    let gateUpHostAfterPrefill = model.prefillMoEGateUpCalls
                    let groupedAfterPrefill = isExpert ? gateUpPlugin?.groupedDispatchCounts() : nil
                    let groupedHostAfterPrefill = model.prefillMoEGroupedDownCalls
                    if let groupedAfterPrefill {
                        trial["grouped_prefill_counts_after"] = groupedAfterPrefill
                        trial["grouped_down_host_calls_after_prefill"] = groupedHostAfterPrefill
                    }
                    if let gateUpAfterPrefill {
                        trial["gateup_prefill_counts_after"] = gateUpAfterPrefill
                        trial["gateup_host_calls_after_prefill"] = gateUpHostAfterPrefill
                        trial["prefill_reduction_calls_after"] = reductionAfterPrefill
                    }
                    if let nativeAfterPrefill {
                        trial["native_prefill_counts_after"] = nativeAfterPrefill
                        trial["prefill_reduction_calls_after"] = reductionAfterPrefill
                    }
                    if let autotuneRuntime { try autotuneRuntime.select(0) }
                    let countsAfterPrefill = tilingRuntime?.counts()
                    if let countsAfterPrefill { trial["prefill_counts_after"] = countsAfterPrefill }
                    if let tilingRuntime { try tilingRuntime.setBM(0) }
                    let profile = profiler.report
                    try profiler.setRecordingEnabled(false)
                    trial["prefill_profile"] = try object(profile)
                    trial["prefill"] = try object(ready.statistics)
                    trial["prefill_wall_seconds"] = prefillWall
                    trial["preparation_seconds"] = ready.preparationSeconds
                    trial["first_token"] = ready.firstToken
                    trial["ready_at_return"] = ready.isReady
                    let profileValid = profile.droppedRecords == 0 &&
                        profile.stages.allSatisfy { $0.succeeded && $0.phase == .prefill && $0.position != nil } &&
                        (profiled ? !profile.stages.isEmpty : profile.stages.isEmpty) &&
                        profile.stages.filter { $0.name == "mixer_and_head" }.count == (profiled ? 1 : 0)
                    trial["profile_valid"] = profileValid
                    guard profileValid, ready.isReady else { throw CLIError.usage("Invalid prefill profile or ready handoff") }
                    if let gateUpBeforePrefill, let gateUpAfterPrefill {
                        let delta = try MoEGateUpProbeSupport.delta(gateUpAfterPrefill, gateUpBeforePrefill)
                        var expected = [UInt64](repeating: 0, count: gateUpBeforePrefill.count)
                        for variant in effectiveVariants {
                            guard expected.indices.contains(variant) else { throw CLIError.usage("Selected gate/up variant lacks a counter") }
                            expected[variant] += UInt64(configuration.layerCount)
                        }
                        let hostDelta = gateUpHostAfterPrefill - gateUpHostBeforePrefill
                        let reductionDelta = reductionAfterPrefill - reductionBeforePrefill
                        let valid = delta == expected && hostDelta == expectedGateUpCalls && reductionDelta == 0
                        trial["gateup_prefill_count_deltas"] = delta
                        trial["expected_gateup_prefill_count_deltas"] = expected
                        trial["gateup_host_prefill_call_delta"] = hostDelta
                        trial["prefill_reduction_call_delta"] = reductionDelta
                        trial["gateup_prefill_dispatch_matches"] = valid
                        guard valid else { throw CLIError.usage("Gate/up prefill dispatches differ from requested variant, or reduction fusion was enabled") }
                    }
                    if let groupedBeforePrefill, let groupedAfterPrefill {
                        let delta = try MoEGateUpProbeSupport.delta(groupedAfterPrefill, groupedBeforePrefill)
                        let hostDelta = groupedHostAfterPrefill - groupedHostBeforePrefill
                        let valid = delta == expectedGroupedCounts && hostDelta == expectedGroupedDownCalls
                        trial["grouped_prefill_count_deltas"] = delta
                        trial["grouped_down_host_prefill_call_delta"] = hostDelta
                        trial["grouped_prefill_dispatch_matches"] = valid
                        guard valid else { throw CLIError.usage("Grouped plan/down prefill counters differ from the effective saved configuration") }
                    }
                    if let nativeBeforePrefill, let nativeAfterPrefill {
                        let delta = try nativeDelta(nativeAfterPrefill, nativeBeforePrefill)
                        var expectedCounts = [UInt64](repeating: 0, count: 5)
                        expectedCounts[selectedNative] = UInt64(expectedNativePrefillCalls)
                        let reductionDelta = reductionAfterPrefill - reductionBeforePrefill
                        let valid = delta == expectedCounts && reductionDelta == expectedReductionCalls
                        trial["native_prefill_count_deltas"] = delta
                        trial["expected_native_prefill_count_deltas"] = expectedCounts
                        trial["prefill_reduction_call_delta"] = reductionDelta
                        trial["moe_prefill_dispatch_matches"] = valid
                        guard valid else { throw CLIError.usage("Native or reduction prefill dispatch counts do not match saved configuration") }
                    }
                    if let countsBeforePrefill, let countsAfterPrefill {
                        let delta = try countDelta(countsAfterPrefill, countsBeforePrefill)
                        trial["prefill_count_deltas"] = delta
                        guard let bm16 = delta["bm16"], let bm32 = delta["bm32"] else {
                            throw CLIError.usage("Missing BM16/BM32 dispatch counters")
                        }
                        let selectedPath = selectedBM == 16 ? bm16 > 0 : bm16 == 0 && bm32 > 0
                        trial["prefill_tiling_dispatch_matches"] = selectedPath
                        guard selectedPath else { throw CLIError.usage("Prefill did not dispatch the requested tiling path") }
                    }
                    stage = "decode"
                    let nativeBeforeDecode = autotuneRuntime?.counts()
                    let reductionBeforeDecode = model.prefillMoEReductionCalls
                    let gateUpBeforeDecode = gateUpPlugin?.dispatchCounts()
                    let gateUpHostBeforeDecode = model.prefillMoEGateUpCalls
                    let groupedBeforeDecode = isExpert ? gateUpPlugin?.groupedDispatchCounts() : nil
                    let groupedHostBeforeDecode = model.prefillMoEGroupedDownCalls
                    if let groupedBeforeDecode {
                        trial["grouped_decode_counts_before"] = groupedBeforeDecode
                        trial["grouped_down_host_calls_before_decode"] = groupedHostBeforeDecode
                    }
                    if let gateUpBeforeDecode {
                        trial["gateup_decode_counts_before"] = gateUpBeforeDecode
                        trial["gateup_host_calls_before_decode"] = gateUpHostBeforeDecode
                        trial["decode_reduction_calls_before"] = reductionBeforeDecode
                    }
                    if let nativeBeforeDecode {
                        trial["native_decode_counts_before"] = nativeBeforeDecode
                        trial["decode_reduction_calls_before"] = reductionBeforeDecode
                    }
                    let countsBeforeDecode = tilingRuntime?.counts()
                    if let countsBeforeDecode { trial["decode_counts_before"] = countsBeforeDecode }
                    let decodeStart = now()
                    let result = try consumer.decode(ready) { callbacks.append($0) }
                    let decodeWall = elapsed(decodeStart)
                    let nativeAfterDecode = autotuneRuntime?.counts()
                    let reductionAfterDecode = model.prefillMoEReductionCalls
                    let gateUpAfterDecode = gateUpPlugin?.dispatchCounts()
                    let gateUpHostAfterDecode = model.prefillMoEGateUpCalls
                    let groupedAfterDecode = isExpert ? gateUpPlugin?.groupedDispatchCounts() : nil
                    let groupedHostAfterDecode = model.prefillMoEGroupedDownCalls
                    if let groupedAfterDecode {
                        trial["grouped_decode_counts_after"] = groupedAfterDecode
                        trial["grouped_down_host_calls_after_decode"] = groupedHostAfterDecode
                    }
                    if let gateUpAfterDecode {
                        trial["gateup_decode_counts_after"] = gateUpAfterDecode
                        trial["gateup_host_calls_after_decode"] = gateUpHostAfterDecode
                        trial["decode_reduction_calls_after"] = reductionAfterDecode
                    }
                    if let nativeAfterDecode {
                        trial["native_decode_counts_after"] = nativeAfterDecode
                        trial["decode_reduction_calls_after"] = reductionAfterDecode
                    }
                    let countsAfterDecode = tilingRuntime?.counts()
                    if let countsAfterDecode { trial["decode_counts_after"] = countsAfterDecode }
                    trial["result"] = try object(result)
                    trial["decode_wall_seconds"] = decodeWall
                    trial["generated_token_ids"] = result.tokens
                    trial["callback_token_ids"] = callbacks
                    trial["finish_reason"] = result.finishReason.rawValue
                    trial["final_state_offset"] = result.statistics.finalStateOffset
                    trial["handoff_consumed"] = !ready.isReady
                    guard let phases = result.phases else { throw CLIError.usage("Missing generation phase statistics") }
                    let finite = [prefillWall, decodeWall, result.preparationSeconds, result.timeToFirstTokenSeconds,
                        result.decodeSeconds, result.totalSeconds, phases.prefill.targetSeconds,
                        phases.prefill.draftHistorySeconds, phases.prefill.totalSeconds, phases.prefill.ssdWaitSeconds,
                        phases.handoffWaitSeconds, phases.handoffConsumeSeconds, phases.decodeServiceSeconds,
                        phases.decodeSSDWaitSeconds, result.statistics.callbackSeconds].allSatisfy { $0.isFinite && $0 >= 0 }
                    let validTokens = !result.tokens.isEmpty && result.tokens.count <= maximum &&
                        result.tokens.allSatisfy { $0 >= 0 && Int($0) < configuration.vocabularySize } &&
                        result.statistics.generatedTokenCount == result.tokens.count && callbacks == result.tokens && !ready.isReady
                    let reference = baseline ?? result
                    let goldenExact = result.tokens == expected && result.finishReason.rawValue == expectedFinish
                    let referenceExact = result.tokens == reference.tokens && result.finishReason == reference.finishReason
                    let expectedOffset = tokens.count + result.tokens.count - 1
                    let offsetExact = result.statistics.finalStateOffset == expectedOffset &&
                        result.statistics.finalStateOffset == reference.statistics.finalStateOffset
                    let modeMatches = ready.statistics.attentionMode == "reference" && phases.prefill.attentionMode == "reference"
                    let decodeUnprofiled = profiler.report.stages.count == profile.stages.count && profiler.report.droppedRecords == 0
                    var decodeCountsUnchanged = true
                    if let gateUpBeforeDecode, let gateUpAfterDecode {
                        let delta = try MoEGateUpProbeSupport.delta(gateUpAfterDecode, gateUpBeforeDecode)
                        let hostDelta = gateUpHostAfterDecode - gateUpHostBeforeDecode
                        let reductionDelta = reductionAfterDecode - reductionBeforeDecode
                        decodeCountsUnchanged = delta.allSatisfy { $0 == 0 } && hostDelta == 0 && reductionDelta == 0
                        trial["gateup_decode_count_deltas"] = delta
                        trial["gateup_host_decode_call_delta"] = hostDelta
                        trial["decode_reduction_call_delta"] = reductionDelta
                        trial["gateup_decode_counts_unchanged"] = decodeCountsUnchanged
                    }
                    if let groupedBeforeDecode, let groupedAfterDecode {
                        let delta = try MoEGateUpProbeSupport.delta(groupedAfterDecode, groupedBeforeDecode)
                        let hostDelta = groupedHostAfterDecode - groupedHostBeforeDecode
                        let valid = delta.allSatisfy { $0 == 0 } && hostDelta == 0
                        decodeCountsUnchanged = decodeCountsUnchanged && valid
                        trial["grouped_decode_count_deltas"] = delta
                        trial["grouped_down_host_decode_call_delta"] = hostDelta
                        trial["grouped_decode_counts_unchanged"] = valid
                    }
                    if let nativeBeforeDecode, let nativeAfterDecode {
                        let delta = try nativeDelta(nativeAfterDecode, nativeBeforeDecode)
                        let reductionDelta = reductionAfterDecode - reductionBeforeDecode
                        decodeCountsUnchanged = delta.allSatisfy { $0 == 0 } && reductionDelta == 0
                        trial["native_decode_count_deltas"] = delta
                        trial["decode_reduction_call_delta"] = reductionDelta
                        trial["moe_decode_counts_unchanged"] = decodeCountsUnchanged
                    }
                    if let countsBeforeDecode, let countsAfterDecode {
                        let delta = try countDelta(countsAfterDecode, countsBeforeDecode)
                        trial["decode_count_deltas"] = delta
                        decodeCountsUnchanged = delta.values.allSatisfy { $0 == 0 }
                        trial["decode_tiling_counts_unchanged"] = decodeCountsUnchanged
                    }
                    trial["finite_public_metrics"] = finite; trial["valid_tokens_and_handoff"] = validTokens
                    trial["golden_exact"] = goldenExact; trial["reference_exact"] = referenceExact
                    trial["first_difference_vs_golden"] = firstDifference(result.tokens, expected)
                    trial["first_difference_vs_reference"] = firstDifference(result.tokens, reference.tokens)
                    trial["expected_final_state_offset"] = expectedOffset
                    trial["reference_final_state_offset"] = reference.statistics.finalStateOffset
                    trial["final_state_offset_matches"] = offsetExact
                    trial["reported_attention_mode_matches"] = modeMatches
                    trial["no_decode_profile_records"] = decodeUnprofiled
                    let passed = finite && validTokens && goldenExact && referenceExact && offsetExact && modeMatches && decodeUnprofiled && decodeCountsUnchanged
                    trial["correctnessPassed"] = passed
                    guard passed else { throw CLIError.usage("Hotspot trial \(index) failed correctness/profile isolation checks; see saved report") }
                    if baseline == nil { baseline = result }
                    trial["complete"] = true
                    trial["memory"] = try MX.memory()
                    FileHandle.standardError.write(Data("Hotspot probe: trial \(index) passed, prefill=\(phases.prefill.targetSeconds)s, stages=\(profile.stages.count)\n".utf8))
                } catch {
                    if let gateUpPlugin {
                        trial["gateup_counts_on_failure"] = gateUpPlugin.dispatchCounts()
                        trial["gateup_host_calls_on_failure"] = model.prefillMoEGateUpCalls
                        if isExpert {
                            trial["grouped_counts_on_failure"] = gateUpPlugin.groupedDispatchCounts()
                            trial["grouped_down_host_calls_on_failure"] = model.prefillMoEGroupedDownCalls
                        }
                    }
                    if let autotuneRuntime {
                        trial["native_counts_on_failure"] = autotuneRuntime.counts()
                        trial["reduction_calls_on_failure"] = model.prefillMoEReductionCalls
                        do { try autotuneRuntime.select(0) }
                        catch { trial["native_reset_error"] = String(describing: error) }
                    }
                    if let tilingRuntime {
                        trial["tiling_counts_on_failure"] = tilingRuntime.counts()
                        do { try tilingRuntime.setBM(0) }
                        catch { trial["tiling_reset_error"] = String(describing: error) }
                    }
                    try? profiler.setRecordingEnabled(false)
                    if trial["prefill_profile"] == nil { trial["prefill_profile"] = try? object(profiler.report) }
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
