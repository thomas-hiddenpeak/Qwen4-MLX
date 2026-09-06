import ANERunnerGPU
import CMLX
import Foundation

private enum SSDPrefetchMode: String { case off, nextChunk }

extension RunnerCLI {
    static func gpuTokenize(_ args: Arguments) throws {
        try args.validate(["--model-dir", "--prompt", "--chat", "--output"])
        let tokenizer = try QwenTokenizer(modelDirectory: URL(fileURLWithPath: args.require("--model-dir")))
        let prompt = try args.require("--prompt")
        let rendered = args["--chat"] == "true" ? try tokenizer.renderChat(messages: [ChatMessage(role: "user", content: prompt)]) : prompt
        let tokens = try tokenizer.encode(rendered)
        try emit(["tokens": tokens, "rendered_prompt": rendered, "decoded": try tokenizer.decode(tokens)], to: args["--output"])
    }

    static func generateGPU(_ args: Arguments) throws {
        try args.validate(["--model-dir", "--prompt", "--tokens-file", "--raw-prompt", "--max-tokens", "--prefill-chunk", "--context", "--output", "--repeat", "--profile-stages", "--ssd-workers", "--ssd-prefetch", "--ssd-prefetch-order", "--prefill-accumulation", "--telemetry-dir", "--telemetry-interval-ms", "--decode-mode", "--decode-order", "--wired-policy", "--wired-order", "--gpu-command-timing-output", "--gdn-gemv-mode", "--gdn-gemv-order", "--mtp-depth", "--mtp-order", "--mtp-verification", "--mtp-verification-order", "--mtp-draft-history", "--prefill-eval-layers", "--verify-eval-layers", "--prefill-attention", "--prefill-moe-config"])
        let draftHistoryTokens = try mtpDraftHistory(args)
        let draftHistoryJSON: Any = draftHistoryTokens.map { $0 as Any } ?? NSNull()
        let directory = URL(fileURLWithPath: try args.require("--model-dir"))
        let evidence = try generationEvidence(modelDirectory: directory)
        func positive(_ key: String, _ fallback: Int, _ limit: Int) throws -> Int {
            guard let n = Int(args[key] ?? String(fallback)), (1...limit).contains(n) else { throw CLIError.usage("Invalid \(key)") }
            return n
        }
        let count = try positive("--max-tokens", 32, 4096), chunk = try positive("--prefill-chunk", 416, 512)
        let context = try positive("--context", 4096, 262144)
        guard let prefillEvalLayers = Int(args["--prefill-eval-layers"] ?? ProcessInfo.processInfo.environment["ANERUNNER_PREFILL_EVAL_LAYERS"] ?? "4"),
              (1...48).contains(prefillEvalLayers) else {
            throw CLIError.usage("ANERUNNER_PREFILL_EVAL_LAYERS must be in 1...48")
        }
        let verificationEvalLayers = try positive("--verify-eval-layers", 4, 48)
        guard let prefillAttention = GPUAttention.PrefillMode(rawValue: args["--prefill-attention"] ?? "reference") else {
            throw CLIError.usage("--prefill-attention requires reference or fusedQSA")
        }
        func order<T: RawRepresentable>(_ single: String, _ sequence: String, _ fallback: T) throws -> [T] where T.RawValue == String {
            guard args[single] == nil || args[sequence] == nil else { throw CLIError.usage("Use \(single) or \(sequence)") }
            let values = (args[sequence] ?? args[single] ?? fallback.rawValue).split(separator: ",", omittingEmptySubsequences: false)
            return try values.map { value in
                guard let result = T(rawValue: String(value)) else { throw CLIError.usage("Invalid \(single)/\(sequence) value: \(value)") }
                return result
            }
        }
        let requestedModes = try order("--decode-mode", "--decode-order", GPUDecodeMode.reference)
        let requestedWired = try order("--wired-policy", "--wired-order", GPUWiredPolicy.disabled)
        let requestedGEMV = try order("--gdn-gemv-mode", "--gdn-gemv-order", GDNGEMVMode.reference)
        let requestedPrefetch = try order("--ssd-prefetch", "--ssd-prefetch-order", SSDPrefetchMode.nextChunk)
        let requestedVerification = try order("--mtp-verification", "--mtp-verification-order", QwenMTPDecoder.Verification.scalar)
        guard args["--mtp-depth"] == nil || args["--mtp-order"] == nil else {
            throw CLIError.usage("Use --mtp-depth or --mtp-order")
        }
        let requestedDepths = try (args["--mtp-order"] ?? args["--mtp-depth"] ?? "0")
            .split(separator: ",", omittingEmptySubsequences: false).map { value -> Int in
                guard let depth = Int(value), (0...4).contains(depth) else { throw CLIError.usage("MTP depth must be 0...4") }
                return depth
            }
        let repetitions = try positive("--repeat", [requestedModes.count, requestedWired.count, requestedGEMV.count, requestedPrefetch.count, requestedDepths.count, requestedVerification.count].max()!, 10)
        guard (args["--mtp-order"] == nil || requestedDepths.count == repetitions),
              (args["--mtp-depth"] == nil || requestedDepths.count == 1) else { throw CLIError.usage("MTP order must match --repeat") }
        let mtpOrder = args["--mtp-order"] == nil ? Array(repeating: requestedDepths[0], count: repetitions) : requestedDepths
        guard !mtpOrder.contains(where: { $0 > 0 }) || args["--gpu-command-timing-output"] == nil else {
            throw CLIError.usage("GPU command graph-boundary tracing currently supports AR only; MTP has separate draft/verify/history timers")
        }
        guard (args["--decode-order"] == nil || requestedModes.count == repetitions),
              (args["--wired-order"] == nil || requestedWired.count == repetitions),
              (args["--gdn-gemv-order"] == nil || requestedGEMV.count == repetitions),
              (args["--ssd-prefetch-order"] == nil || requestedPrefetch.count == repetitions),
              (args["--mtp-verification-order"] == nil || requestedVerification.count == repetitions),
              (args["--decode-mode"] == nil || requestedModes.count == 1),
              (args["--wired-policy"] == nil || requestedWired.count == 1),
              (args["--gdn-gemv-mode"] == nil || requestedGEMV.count == 1),
              (args["--ssd-prefetch"] == nil || requestedPrefetch.count == 1),
              (args["--mtp-verification"] == nil || requestedVerification.count == 1) else {
            throw CLIError.usage("Each order must contain exactly --repeat values; a single mode/policy cannot contain commas")
        }
        let decodeOrder = args["--decode-order"] == nil ? Array(repeating: requestedModes[0], count: repetitions) : requestedModes
        let verificationOrder = args["--mtp-verification-order"] == nil ? Array(repeating: requestedVerification[0], count: repetitions) : requestedVerification
        guard !(0..<repetitions).contains(where: { mtpOrder[$0] > 0 && verificationOrder[$0].usesScalarLinear && decodeOrder[$0] != .reference }) else {
            throw CLIError.usage("Scalar-linear verification policies require reference decode kernels")
        }
        let wiredOrder = args["--wired-order"] == nil ? Array(repeating: requestedWired[0], count: repetitions) : requestedWired
        let gemvOrder = args["--gdn-gemv-order"] == nil ? Array(repeating: requestedGEMV[0], count: repetitions) : requestedGEMV
        let prefetchOrder = args["--ssd-prefetch-order"] == nil ? Array(repeating: requestedPrefetch[0], count: repetitions) : requestedPrefetch
        for gemv in requestedGEMV { try gemv.apply() }
        try GDNGEMVMode.reference.apply()
        defer { try? GDNGEMVMode.reference.apply() }
        let commandTiming = try args["--gpu-command-timing-output"].map { try GPUCommandTimingSession(path: $0) }
        if let commandTiming, let output = args["--output"], URL(fileURLWithPath: output).standardizedFileURL.path == commandTiming.path {
            throw CLIError.usage("Generation and command timing reports require different paths")
        }
        let ssdWorkers = try positive("--ssd-workers", 1, GPUSSDReader.maximumWorkers)
        let tokenizer = try QwenTokenizer(modelDirectory: directory)
        guard let mode = GPUProfiler.Mode(rawValue: args["--profile-stages"] ?? "disabled") else { throw CLIError.usage("Invalid --profile-stages mode") }
        let profiler = try GPUProfiler(mode: mode, maximumRecords: 8192)
        guard let accumulation = GPUMoE.PrefillAccumulation(rawValue: args["--prefill-accumulation"] ?? "reference") else { throw CLIError.usage("Invalid --prefill-accumulation; use reference or float32") }
        // Validate the saved selection and loaded native identities before
        // allocating any model weights. No environment selector is changed.
        let moeSelection = try args["--prefill-moe-config"].map {
            try GPUMoEPrefillSelection(path: $0, modelDirectory: directory, accumulation: accumulation)
        }
        let tokens: [Int32]
        if let path = args["--tokens-file"] {
            guard args["--prompt"] == nil else { throw CLIError.usage("Use --prompt or --tokens-file") }
            tokens = try JSONDecoder().decode([Int32].self, from: Data(contentsOf: URL(fileURLWithPath: path)))
        } else {
            let prompt = try args.require("--prompt")
            if let raw = args["--raw-prompt"], raw != "true", raw != "false" { throw CLIError.usage("--raw-prompt requires true/false") }
            let rendered = args["--raw-prompt"] == "true" ? prompt : try tokenizer.renderChat(messages: [ChatMessage(role: "user", content: prompt)])
            tokens = try tokenizer.encode(rendered)
        }
        guard !tokens.isEmpty, tokens.count + count <= context else { throw CLIError.usage("Prompt plus output exceeds configured context") }
        guard args["--telemetry-dir"] != nil || args["--telemetry-interval-ms"] == nil else {
            throw CLIError.usage("--telemetry-interval-ms requires --telemetry-dir")
        }
        let telemetry: GPUTelemetrySession?
        if let path = args["--telemetry-dir"] {
            telemetry = try GPUTelemetrySession(directory: URL(fileURLWithPath: path),
                intervalMilliseconds: positive("--telemetry-interval-ms", 200, 10_000))
        } else { telemetry = nil }
        var inferenceSucceeded = false
        defer { telemetry?.finish(succeeded: inferenceSucceeded) }
        defer { try? commandTiming?.finish() }
        // Release transient allocations promptly; the loaded weight cache is
        // intentionally separate and retains original Q4/BF16 model tensors.
        var previousLimit = 0
        try MX.check(mlx_set_cache_limit(&previousLimit, 256 * 1024 * 1024), "set allocation cache limit")
        let loadSpan = telemetry?.begin("load")
        let loadStart = DispatchTime.now().uptimeNanoseconds
        let model = try QwenModel(modelDirectory: directory, profiler: profiler, reservedOutputIDs: tokenizer.reservedOutputTokenIDs, ssdWorkers: ssdWorkers, prefillAccumulation: accumulation, decodeModes: decodeOrder) { current, total in
            if current % 4 == 0 || current == total { FileHandle.standardError.write(Data("Loaded \(current)/\(total) layers\n".utf8)) }
        }
        let mtpHead = mtpOrder.contains(where: { $0 > 0 }) ? try QwenMTP(weights: model.weights, configuration: model.configuration) : nil
        let loadSeconds = Double(DispatchTime.now().uptimeNanoseconds - loadStart) * 1e-9
        let loadedMemory = try MX.memory()
        if let loadSpan { telemetry?.end(loadSpan) }
        try commandTiming?.start()
        let expectedMoEPrefill = GPUMoEPrefillSelection.expectedPrefill(
            selection: moeSelection, promptTokens: tokens.count, chunk: chunk, layers: model.layerCount)
        var trials: [[String: Any]] = []
        for repetition in 0..<repetitions {
            let decodeMode = decodeOrder[repetition]
            let verification = verificationOrder[repetition]
            try gemvOrder[repetition].apply()
            FileHandle.standardError.write(Data("Trial \(repetition + 1)/\(repetitions), MTP: \(mtpOrder[repetition]) / \(verification.rawValue), GDN GEMV: \(gemvOrder[repetition].rawValue), SSD prefetch: \(prefetchOrder[repetition].rawValue)\n".utf8))
            // Policy setup is outside request timing and reported separately.
            let wiredReport = try GPUWiredMemory.apply(wiredOrder[repetition])
            var state = model.makeState(), next: Int32 = 0
            let mtpDecoder = mtpOrder[repetition] > 0 ? QwenMTPDecoder(model: model, head: mtpHead!, verification: verification,
                                                                    draftHistoryTokens: draftHistoryTokens,
                                                                    verificationEvaluateEveryLayers: verificationEvalLayers) : nil
            var prefill: [Double] = [], decode: [Double] = [], waitSeconds = 0.0, logicalSSD = 0
            var prefillWait: [Double] = [], prefillTargetSeconds = 0.0
            var prefetch: QwenModel.PrefillPrefetch?
            defer { prefetch?.finish() }
            let moeBeforePrefill = GPUMoEPrefillSelection.snapshot(model: model, selection: moeSelection)
            let requestStart = DispatchTime.now().uptimeNanoseconds
            let requestStartNS = telemetry == nil ? 0 : GPUTelemetrySession.now()
            var decodeStartNS: UInt64?
            var offset = 0
            while offset < tokens.count {
                // Match the reference scheduler: retain the final prompt
                // token for a decode-shaped step, including its BF16 state
                // boundary. This is ordinary autoregressive decoding, not MTP.
                let end = offset < tokens.count - 1 ? min(tokens.count - 1, offset + chunk) : tokens.count
                let span = telemetry?.begin("prefill", repetition: repetition, step: prefill.count, inputTokens: end - offset)
                let start = DispatchTime.now().uptimeNanoseconds
                let commandStart = commandTiming == nil ? 0 : GPUCommandTimingSession.now()
                // Plan only after request/prefill timing starts, so lookup
                // preparation and all unhidden IO remain part of TTFT.
                if prefetchOrder[repetition] == .nextChunk, prefetch == nil {
                    prefetch = try model.makePrefillPrefetch(tokens: tokens, chunk: chunk, state: state)
                }
                let out = try model.forward(tokens: Array(tokens[offset..<end]), state: &state,
                                            evaluateEveryLayers: prefillEvalLayers, decodeMode: .reference,
                                            prefillPrefetch: prefetch, phase: .prefill,
                                            prefillAttention: prefillAttention,
                                            prefillMoEConfiguration: moeSelection?.configuration)
                let commandForwardEnd = commandTiming == nil ? 0 : GPUCommandTimingSession.now()
                let forwardEnd = telemetry == nil ? nil : GPUTelemetrySession.now()
                guard let logits = out.logits else { throw GPUError.invalid("Full model did not return logits") }
                if end == tokens.count {
                    let selected = try model.greedyToken(logits)
                    try model.evaluate([selected], state: &state)
                    next = try decodeMode.usesNativeTokenReadback ? selected.uint32TokenID() : selected.ints()[0]
                } else { try model.evaluate([out.stream], state: &state) }
                prefillTargetSeconds += Double(DispatchTime.now().uptimeNanoseconds - start) * 1e-9
                try mtpDecoder?.consumePrompt(stream: out.stream, prompt: tokens, offset: offset)
                let commandEvaluationEnd = commandTiming == nil ? 0 : GPUCommandTimingSession.now()
                prefill.append(Double(DispatchTime.now().uptimeNanoseconds - start) * 1e-9)
                commandTiming?.step(phase: "prefill", repetition: repetition, index: prefill.count - 1,
                    inputTokens: end - offset, start: commandStart, forwardEnd: commandForwardEnd, evaluationEnd: commandEvaluationEnd)
                if let span {
                    telemetry?.end(span, outputTokens: end == tokens.count ? 1 : 0,
                        forwardEndNS: forwardEnd, evaluationEndNS: GPUTelemetrySession.now(),
                        ssdWaitSeconds: out.ssdWaitSeconds, ssdRequestedBytes: out.ssdLogicalBytes)
                }
                waitSeconds += out.ssdWaitSeconds; logicalSSD += out.ssdLogicalBytes
                prefillWait.append(out.ssdWaitSeconds)
                offset = end
                if tokens.count > chunk && end < tokens.count && end % (chunk * 4) == 0 {
                    FileHandle.standardError.write(Data("Prefilled \(end)/\(tokens.count) tokens\n".utf8))
                }
            }
            prefetch?.finish(); prefetch = nil
            try mtpDecoder?.finishPrompt(expectedTokenCount: tokens.count)
            let prefillLogicalSSD = logicalSSD
            let ttft = Double(DispatchTime.now().uptimeNanoseconds - requestStart) * 1e-9
            let prefillEndNS = telemetry == nil ? 0 : GPUTelemetrySession.now()
            let moeAfterPrefill = GPUMoEPrefillSelection.snapshot(model: model, selection: moeSelection)
            var generated: [Int32] = [next]
            var stopped = tokenizer.eosTokenIDs.contains(next)
            while generated.count < count && !stopped {
                let span = telemetry?.begin("decode", repetition: repetition, step: decode.count, inputTokens: mtpDecoder == nil ? 1 : 0)
                if decodeStartNS == nil { decodeStartNS = span?.start }
                let start = DispatchTime.now().uptimeNanoseconds
                let commandStart = commandTiming == nil ? 0 : GPUCommandTimingSession.now()
                let emitted: [Int32], stepWait: Double, stepSSD: Int, inputCount: Int
                let commandForwardEnd: UInt64, forwardEnd: UInt64?
                if let mtpDecoder {
                    let before = mtpDecoder.statistics
                    let result = try mtpDecoder.next(pending: next, state: &state, depth: mtpOrder[repetition],
                        remaining: count - generated.count, eos: tokenizer.eosTokenIDs, decodeMode: decodeMode)
                    emitted = result.tokens; stepWait = result.ssdWaitSeconds; stepSSD = result.ssdLogicalBytes
                    let after = mtpDecoder.statistics
                    inputCount = max(1, after.verifiedTokens + after.replayedTokens - before.verifiedTokens - before.replayedTokens)
                    commandForwardEnd = commandTiming == nil ? 0 : GPUCommandTimingSession.now()
                    forwardEnd = telemetry == nil ? nil : GPUTelemetrySession.now()
                } else {
                    let out = try model.forward(tokens: [next], state: &state, decodeMode: decodeMode, phase: .decode)
                    commandForwardEnd = commandTiming == nil ? 0 : GPUCommandTimingSession.now()
                    forwardEnd = telemetry == nil ? nil : GPUTelemetrySession.now()
                    guard let logits = out.logits else { throw GPUError.invalid("Missing decode logits") }
                    let selected = try model.greedyToken(logits)
                    try model.evaluate([selected], state: &state)
                    let token = try decodeMode.usesNativeTokenReadback ? selected.uint32TokenID() : selected.ints()[0]
                    emitted = [token]; stepWait = out.ssdWaitSeconds; stepSSD = out.ssdLogicalBytes; inputCount = 1
                }
                // MTP combines draft/verify/replay/history; its stage timings
                // are reported separately instead of inventing a graph boundary.
                next = emitted.last!
                let commandEvaluationEnd = commandTiming == nil ? 0 : GPUCommandTimingSession.now()
                decode.append(Double(DispatchTime.now().uptimeNanoseconds - start) * 1e-9)
                commandTiming?.step(phase: "decode", repetition: repetition, index: decode.count - 1,
                    inputTokens: inputCount, start: commandStart, forwardEnd: commandForwardEnd, evaluationEnd: commandEvaluationEnd)
                if let span {
                    telemetry?.end(span, outputTokens: emitted.count,
                        forwardEndNS: forwardEnd, evaluationEndNS: GPUTelemetrySession.now(),
                        ssdWaitSeconds: stepWait, ssdRequestedBytes: stepSSD)
                }
                waitSeconds += stepWait; logicalSSD += stepSSD
                generated.append(contentsOf: emitted)
                stopped = tokenizer.eosTokenIDs.contains(next)
                if generated.count % 16 == 0 { FileHandle.standardError.write(Data("Generated \(generated.count)/\(count) tokens\n".utf8)) }
            }
            let total = Double(DispatchTime.now().uptimeNanoseconds - requestStart) * 1e-9
            telemetry?.request(repetition: repetition, startNS: requestStartNS, prefillEndNS: prefillEndNS,
                decodeStartNS: decodeStartNS, endNS: GPUTelemetrySession.now())
            // Snapshot after completed evaluations; validation/report work is
            // outside the measured target/decode steps and request duration.
            let moeAfterDecode = GPUMoEPrefillSelection.snapshot(model: model, selection: moeSelection)
            let moePrefillCounts = try moeAfterPrefill.subtracting(moeBeforePrefill)
            let moeDecodeCounts = try moeAfterDecode.subtracting(moeAfterPrefill)
            guard moePrefillCounts == expectedMoEPrefill, moeDecodeCounts.isZero else {
                throw CLIError.usage("Prefill MoE call counts differ from selected chunks, or a prefill optimization ran during decode/MTP")
            }
            let moeCalls: [String: Any] = [
                "prefill": moePrefillCounts.report, "expected_prefill": expectedMoEPrefill.report,
                "decode": moeDecodeCounts.report, "prefill_matches_expected": true, "decode_zero": true,
                "scope": "Host graph calls and native encoded dispatches after evaluation; not GPU time or physical memory traffic. Native fields are null when no plugin was selected."
            ]
            let decodeTime = decode.reduce(0, +)
            let decodeRate: Any = decodeTime > 0 ? Double(generated.count - 1) / decodeTime : NSNull()
            FileHandle.standardError.write(Data("Trial \(repetition + 1): \(decodeRate) tokens/s\n".utf8))
            let logicalRate: Any = decodeTime > 0 && mtpDecoder == nil ? Double(model.logicalDecodeWeightBytes) * Double(decode.count) / decodeTime * 1e-9 : NSNull()
            let decodeTPOT: Any = generated.count > 1 ? decodeTime / Double(generated.count - 1) : NSNull()
            let mtpCostSummary = mtpDecoder.map {
                QwenMTPCostSummary(statistics: $0.statistics, decodeSteps: decode.count,
                    committedDecodeTokens: max(0, generated.count - 1), decodeSeconds: decodeTime)
            }
            // Keep throwing generic serialization out of the large heterogeneous
            // trial dictionary so Swift does not need to infer the whole expression.
            let mtpStatisticsJSON: Any
            if let mtpDecoder {
                mtpStatisticsJSON = try JSONSerialization.jsonObject(with: JSONEncoder().encode(mtpDecoder.statistics))
            } else { mtpStatisticsJSON = NSNull() }
            let mtpCostSummaryJSON: Any
            if let mtpCostSummary {
                mtpCostSummaryJSON = try JSONSerialization.jsonObject(with: JSONEncoder().encode(mtpCostSummary))
            } else { mtpCostSummaryJSON = NSNull() }
            let phaseMetrics: [String: Any] = [
                "prefill_target_seconds": prefillTargetSeconds,
                "prefill_target_tokens_per_second": Double(tokens.count) / prefillTargetSeconds,
                "prefill_total_seconds": ttft,
                "mtp_prompt_history_seconds": mtpDecoder?.statistics.prefillHistorySeconds ?? 0,
                "prefill_ssd_requested_row_bytes": prefillLogicalSSD,
                "decode_seconds": decodeTime, "decode_mean_seconds_per_token": decodeTPOT,
                "decode_ssd_wait_seconds": max(0, waitSeconds - prefillWait.reduce(0, +)),
                "decode_ssd_requested_row_bytes": logicalSSD - prefillLogicalSSD,
                "prefill_kernel_mode": moeSelection != nil ? "request-configured" : (prefillAttention == .reference ? "reference" : "reference-with-fusedQSA-attention"),
                "decode_kernel_mode": decodeMode.rawValue,
                "prefill_attention_mode": prefillAttention.rawValue,
                "verification_evaluate_every_layers": verificationEvalLayers,
                "handoff_transport": "inline-benchmark-no-transport"
            ]
            trials.append([
                "phase_metrics": phaseMetrics,
                "repetition": repetition, "prompt_tokens": tokens, "generated_token_ids": generated,
                "text": try tokenizer.decode(generated, skipSpecialTokens: true), "finish_reason": stopped ? "eos" : "length",
                "sampling": "greedy", "mtp_enabled": mtpDecoder != nil, "prefill_chunk": chunk,
                "mtp_depth": mtpOrder[repetition], "mtp_verification": verification.rawValue,
                "mtp_draft_history_limit": draftHistoryJSON,
                "mtp_statistics": mtpStatisticsJSON,
                "mtp_cost_summary": mtpCostSummaryJSON,
                "decode_mode": decodeMode.rawValue,
                "gdn_gemv_mode": gemvOrder[repetition].rawValue,
                "wired_memory": try JSONSerialization.jsonObject(with: JSONEncoder().encode(wiredReport)),
                "prefill_accumulation": accumulation.rawValue,
                "prefill_moe_configuration": moeSelection?.effectiveJSON ?? NSNull(),
                "prefill_moe_calls": moeCalls,
                "final_state_offset": state.offset, "qsa_active_layers": state.qsaActiveLayers,
                "ssd_workers": ssdWorkers,
                "ssd_prefetch": prefetchOrder[repetition].rawValue,
                "prefill_ssd_wait_seconds": prefillWait.reduce(0, +),
                "prefill_chunk_ssd_wait_seconds": prefillWait,
                "final_prompt_token_held_back": true,
                "time_to_first_token_seconds_excluding_load": ttft,
                "request_seconds_excluding_load": total,
                "prefill_chunk_seconds": prefill, "prefill_tokens_per_second": Double(tokens.count) / prefill.reduce(0, +),
                "decode_step_seconds": decode, "decode_steps": decode.count,
                "decode_tokens_per_second": decodeRate,
                "ssd_wait_seconds": waitSeconds, "ssd_requested_row_bytes": logicalSSD,
                "physical_dram_bytes": NSNull(), "physical_dram_bandwidth_gbps": NSNull(),
                "logical_decode_weight_bytes_per_token": model.logicalDecodeWeightBytes,
                "logical_decode_weight_footprint_rate_gbps": logicalRate,
                "memory": try MX.memory()
            ])
        }
        try commandTiming?.finish()
        inferenceSucceeded = true
        telemetry?.finish(succeeded: true)
        try emit([
            "schema_version": 1, "runner": "independent-swift-mlx-c", "backend": "Metal GPU",
            "model_directory": directory.standardizedFileURL.path, "layers": model.layerCount,
            "provenance": evidence,
            "load_seconds": loadSeconds, "loaded_memory": loadedMemory,
            "max_tokens": count, "context_limit": context, "requested_repetitions": repetitions,
            "decode_order": decodeOrder.map(\.rawValue), "wired_order": wiredOrder.map(\.rawValue),
            "gdn_gemv_order": gemvOrder.map(\.rawValue),
            "ssd_prefetch_order": prefetchOrder.map(\.rawValue),
            "fused_attention_prefill": GPUAttention.fusedPrefillEnabled,
            "experimental_fused_attention_prefill": GPUAttention.fusedPrefillEnabled,
            "experimental_blocked_gdn_prefill": ProcessInfo.processInfo.environment["ANERUNNER_BLOCKED_GDN"] == "1",
            "prefill_evaluate_every_layers": prefillEvalLayers,
            "prefill_attention_mode": prefillAttention.rawValue,
            "prefill_moe_selection": moeSelection?.provenance ?? ["enabled": false],
            "verification_evaluate_every_layers": verificationEvalLayers,
            "additional_projection_buffer_bytes": model.additionalProjectionBufferBytes,
            "scheduler_environment": ProcessInfo.processInfo.environment.filter { ["MLX_MAX_MB_PER_BUFFER", "MLX_MAX_OPS_PER_BUFFER"].contains($0.key) },
            "loaded_source_weight_bytes": model.weights.cachedSourceBytes,
            "mtp_enabled": mtpHead != nil, "mtp_order": mtpOrder, "ane_used": false,
            "mtp_verification_order": verificationOrder.map(\.rawValue),
            "mtp_weights_loaded": model.weights.ledger.contains { $0.name.contains(".mtp.") },
            "profiler": try JSONSerialization.jsonObject(with: JSONEncoder().encode(profiler.report)),
            "telemetry": telemetry?.report ?? ["enabled": false],
            "gpu_command_timing": commandTiming?.report ?? ["enabled": false],
            "bandwidth_note": "Physical DRAM traffic is unavailable. SSD row bytes are logical requested payload, not physical disk traffic.",
            "trials": trials
        ], to: args["--output"])
    }

    private static func generationEvidence(modelDirectory: URL) throws -> [String: Any] {
        var version = mlx_string_new()
        defer { _ = mlx_string_free(version) }
        try MX.check(mlx_version(&version), "MLX version")
        let mlxVersion = mlx_string_data(version).map { String(cString: $0) } ?? "unavailable"
        let executable = Bundle.main.executableURL ?? URL(fileURLWithPath: CommandLine.arguments[0])
        var hashes: [String: String] = [:]
        for name in ["config.json", "model.safetensors.index.json", "tokenizer.json", "chat_template.jinja"] {
            hashes[name] = try GPUProbeSupport.hash(modelDirectory.appendingPathComponent(name))
        }
        return [
            "captured_before_model_load_utc": Date().ISO8601Format(),
            "operating_system": ProcessInfo.processInfo.operatingSystemVersionString,
            "physical_memory_bytes": ProcessInfo.processInfo.physicalMemory,
            "process_id": ProcessInfo.processInfo.processIdentifier,
            "logical_cpu_count": ProcessInfo.processInfo.processorCount,
            "mlx_version": mlxVersion, "executable": executable.standardizedFileURL.path,
            "executable_sha256": (try? GPUProbeSupport.hash(executable)) as Any? ?? NSNull(),
            "model_metadata_sha256": hashes,
            "hash_scope": "Executable and model metadata captured before loading; complete checkpoint payloads are not rehashed by this command."
        ]
    }
}
