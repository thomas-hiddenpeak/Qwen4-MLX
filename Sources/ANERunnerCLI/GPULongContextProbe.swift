import ANERunnerCore
import ANERunnerGPU
import CMLX
import Foundation

private struct LongContextAnchor: Codable {
    let host: QwenModel.State.DiagnosticHostValues
    let tensors: [String: CacheReliabilityAnchor.Value]
    var finite: Bool { tensors.count == 121 && tensors.values.allSatisfy(\.finite) }
    func matches(_ other: Self) -> Bool {
        finite && other.finite && tensors == other.tensors && host.valid && other.host.valid &&
            host.offset == other.host.offset && host.gdnOffsets == other.host.gdnOffsets &&
            host.attentionOffsets == other.host.attentionOffsets && host.pleHistory == other.host.pleHistory &&
            host.gdnCapturePresent == other.host.gdnCapturePresent && host.pleCapturePresent == other.host.pleCapturePresent
    }
}

/// Counts only device-allocator observations, never RSS or physical bandwidth.
/// Peak is reset at explicit diagnostic boundaries to distinguish those copies
/// from the business graph's allocations, including temporary intermediates.
private final class LongContextProbeMemory {
    var businessPeaks = [String: Int](), diagnosticPeak = 0
    var rows = [[String: Any]]()
    func resetPeak() throws { try MX.check(mlx_reset_peak_memory(), "long-context peak reset") }
    func business(_ phase: String, offset: Int) throws {
        let values = try MX.memory()
        businessPeaks[phase] = max(businessPeaks[phase] ?? 0, values["peak_bytes"] ?? 0)
        rows.append(["phase": phase, "offset": offset, "memory": values])
    }
    func diagnostic() throws {
        let values = try MX.memory()
        diagnosticPeak = max(diagnosticPeak, values["peak_bytes"] ?? 0)
    }
}

extension RunnerCLI {
    /// Bounded per-context screening, then explicit full-context RAM reuse.
    /// No paged mode, state clones, MTP, service lifecycle or implicit HTTP call.
    static func probeGPULongContext(_ args: Arguments) throws {
        try args.validate(["--model-dir", "--tokens-file", "--output", "--mode", "--context",
            "--max-tokens", "--prefill-attention", "--state-budget-bytes", "--prefix-cache-bytes",
            "--rmse-atol", "--rmse-rtol"])
        let output = try args.require("--output"), mode = args["--mode"] ?? "screen"
        guard !FileManager.default.fileExists(atPath: output), ["screen", "full"].contains(mode),
              let context = Int(args["--context"] ?? "32768"), [32_768, 65_536, 262_144].contains(context),
              let maximum = Int(args["--max-tokens"] ?? "2"), [2, 8].contains(maximum),
              context < 262_144 || maximum == 2,
              mode != "screen" || context <= 65_536,
              let budgetBytes = Int(args["--state-budget-bytes"] ?? "25769803776"), budgetBytes > 0,
              let cacheBytes = Int(args["--prefix-cache-bytes"] ?? "8589934592"), cacheBytes > 0,
              let atol = Double(args["--rmse-atol"] ?? "0.02"), atol.isFinite, (0...1).contains(atol),
              let rtol = Double(args["--rmse-rtol"] ?? "0.02"), rtol.isFinite, (0...1).contains(rtol),
              let attention = GPUAttention.PrefillMode(rawValue: args["--prefill-attention"] ?? "reference") else {
            throw CLIError.usage("Long context probe: new --output; --mode screen|full; --context 32768|65536|262144 (screen <=65536); --max-tokens 2|8 (262144 requires2); explicit reference|fusedQSA; positive byte budgets and finite RMSE thresholds")
        }
        guard mode != "screen" || args["--prefill-attention"] == nil else {
            throw CLIError.usage("screen compares both attention modes; use full for one explicitly selected mode")
        }
        let directory = URL(fileURLWithPath: try args.require("--model-dir")).standardizedFileURL.resolvingSymlinksInPath()
        let inputURL = URL(fileURLWithPath: try args.require("--tokens-file")).standardizedFileURL
        let input = try Data(contentsOf: inputURL)
        let source: [Int32]
        if let plain = try? JSONDecoder().decode([Int32].self, from: input) { source = plain }
        else {
            struct TokenReport: Decodable { let tokens: [Int32] }
            source = try JSONDecoder().decode(TokenReport.self, from: input).tokens
        }
        let promptCount = context - maximum
        guard source.count >= promptCount else { throw CLIError.usage("Token fixture is shorter than context minus output budget") }
        let tokens = Array(source.prefix(promptCount)), boundary = (promptCount - 1) / 416 * 416
        let configuration = try QwenConfiguration(modelDirectory: directory)
        let tokenizer = try QwenTokenizer(modelDirectory: directory)
        var report: [String: Any] = ["schema": "qwen-long-context-probe-v1", "complete": false,
            "passed": false, "mode": mode, "synthetic_capacity_only": true, "quality_evaluation": false,
            "configuration": ["context_limit": context, "prompt_tokens": promptCount, "max_tokens": maximum,
                "prefix_checkpoint": boundary, "prefill_chunk": 416, "prefill_eval_layers": 4,
                "state_budget_bytes": budgetBytes, "prefix_cache_bytes": cacheBytes, "mtp_depth": 0],
            "tolerance": ["rmse_atol": atol, "rmse_rtol": rtol,
                "rule": "tensor RMSE <= atol + rtol * reference RMS",
                "scope": "Predeclared experimental screening at the identical complete prompt, not a model-quality guarantee; never automatically relaxed after observing a result."],
            "notes": [
                "GPU runs use existing token IDs directly; this synthetic capacity fixture is not a business-quality evaluation.",
                "Each attention mode has its own exact cold/warm121-tensor/host/output oracle. Reference versus fused is compared with declared numerical screening tolerances, not required bitwise equality.",
                "Only dense namedTensors are read. One tensor's BF16 host bytes are admitted/read/released at a time; no full private model-state or paged materialization is created for diagnostics.",
                "Diagnostic intervals are measured explicitly and excluded from reported active wall times. Raw generator phase metrics may include observers; they are retained only as diagnostic metadata.",
                "MLX business peak is sampled before diagnostics and at every public slice boundary; diagnostic peak is separate. These are allocator peaks, not RSS, DRAM bandwidth, or uninstrumented performance acceptance.",
                "O2 must actually execute at least one decode round; early EOS before that produces capacity_only_no_decode and fails the decode gate.",
                "Full-context SSD checkpoint restore is not part of this RAM-only probe."
            ],
            "correctness_scope": ["finite_state_tensors_per_anchor": 121,
                "all_logits_directly_checked": false,
                "cold_warm_state_comparison": "All 121 tensor hashes and logical host state must match exactly within each attention mode.",
                "cross_mode_comparison": "All 121 prompt-state tensors use fixed experimental RMSE bounds; complete generated IDs and finish reason must also match."]]
        var checks = [String: Bool](), states = [[String: Any]](), trials = [[String: Any]]()
        var crossRows = [[String: Any]]()
        func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
        func elapsed(_ start: UInt64) -> Double { Double(now() - start) * 1e-9 }
        func object<T: Encodable>(_ value: T) throws -> Any {
            try JSONSerialization.jsonObject(with: JSONEncoder().encode(value))
        }
        func save(complete: Bool = false) throws {
            report["complete"] = complete; report["checks"] = checks
            report["states"] = states; report["trials"] = trials; report["cross_attention_tensors"] = crossRows
            report["passed"] = complete && !checks.isEmpty && checks.values.allSatisfy { $0 } &&
                !states.isEmpty && !trials.isEmpty
            try emit(report, to: output)
        }
        func require(_ label: String, _ condition: Bool) throws {
            checks[label] = condition
            guard condition else { throw CLIError.usage("Long context check failed: \(label)") }
        }
        try save()
        do {
            let executable = Bundle.main.executableURL ?? URL(fileURLWithPath: CommandLine.arguments[0])
            report["provenance"] = ["command": CommandLine.arguments, "captured_utc": Date().ISO8601Format(),
                "executable_sha256": try MoETilingBytes.hash(executable), "model_directory": directory.path,
                "config_sha256": try MoETilingBytes.hash(directory.appendingPathComponent("config.json")),
                "tokenizer_sha256": try MoETilingBytes.hash(directory.appendingPathComponent("tokenizer.json")),
                "source_token_path": inputURL.path, "source_token_sha256": MoETilingBytes.digest(input),
                "source_token_count": source.count, "used_token_prefix_count": promptCount,
                "used_tokens_sha256": MoETilingBytes.digest(try JSONEncoder().encode(tokens))]
            let valid = QwenGenerationRequest(tokens: tokens, maxTokens: maximum, contextLimit: context,
                prefillChunk: 416, mtpDepth: 0, prefillEvaluateEveryLayers: 4,
                prefillAttention: attention, prefixCacheMaxTokens: boundary)
            try valid.validate(configuration: configuration)
            var overRejected = false
            do {
                try QwenGenerationRequest(tokens: tokens, maxTokens: maximum + 1,
                    contextLimit: context, prefillChunk: 416).validate(configuration: configuration)
            } catch QwenGenerationError.invalidRequest { overRejected = true }
            try require("cpu_prompt_plus_output_overlimit_rejected", overRejected)
            try require("prefix_checkpoint_original416_grid", boundary > 0 && boundary < promptCount && boundary % 416 == 0)

            var oldCache = 0
            try MX.check(mlx_set_cache_limit(&oldCache, 256 * 1024 * 1024), "long context allocator cache")
            defer { var ignored = 0; _ = mlx_set_cache_limit(&ignored, oldCache) }
            let model = try QwenModel(modelDirectory: directory,
                reservedOutputIDs: tokenizer.reservedOutputTokenIDs, stateBudgetBytes: budgetBytes) { current, total in
                if current % 8 == 0 || current == total {
                    FileHandle.standardError.write(Data("Long context loaded \(current)/\(total) layers\n".utf8))
                }
            }
            defer { try? MX.synchronize() }
            try require("full_model_no_mtp", model.layerCount == 48 && !model.weights.ledger.contains { $0.name.contains(".mtp.") })
            try require("no_decode_async_experiment", model.experimentalDecodeAsyncEveryLayers == 0)
            let estimatedState = try model.estimatedPrefixStateBytes(at: context)
            let checkpointBytes = try model.estimatedPrefixStateBytes(at: boundary)
            report["state_bytes_estimated"] = estimatedState
            report["checkpoint_bytes_estimated"] = checkpointBytes
            report["memory_after_model_load"] = try MX.memory()
            try require("ram_snapshot_fits_explicit_cache_budget", checkpointBytes <= cacheBytes)
            // Largest dense K/V tensor is one BF16 [1,2,T,256] array.
            // Three tensor sizes cover possible contiguous device readback,
            // host bytes and mapped/reference bytes; no full state copy.
            let maximumDiagnosticBytes = 3 * context * 2 * 256 * 2 + 65_536
            try require("request_cache_and_single_tensor_diagnostic_admitted",
                estimatedState <= (budgetBytes - checkpointBytes - maximumDiagnosticBytes) / 2)
            report["maximum_diagnostic_workspace_bytes"] = maximumDiagnosticBytes
            let scratch = URL(fileURLWithPath: output).appendingPathExtension("reference-tensors")
            if mode == "screen" {
                guard !FileManager.default.fileExists(atPath: scratch.path) else {
                    throw CLIError.usage("Reference tensor directory already exists")
                }
                try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: false)
                report["numerical_reference_directory"] = scratch.path
            }
            let modes: [GPUAttention.PrefillMode] = mode == "screen" ? [.reference, .fusedQSA] : [attention]
            var referenceResult: QwenGenerationResult?
            var numericalReference = [String: CacheReliabilityAnchor.Value]()

            func difference(_ data: Data, reference: Data) throws -> (rmse: Double, referenceRMS: Double, maxAbs: Double) {
                guard data.count == reference.count, data.count > 0, data.count % 2 == 0 else {
                    throw CLIError.usage("Numerical comparison shape/byte mismatch")
                }
                return data.withUnsafeBytes { (actual: UnsafeRawBufferPointer) in
                    reference.withUnsafeBytes { (expected: UnsafeRawBufferPointer) in
                        var squared = 0.0, norm = 0.0, largest = 0.0
                        for i in stride(from: 0, to: data.count, by: 2) {
                            let a = UInt32(actual[i]) | UInt32(actual[i + 1]) << 8
                            let b = UInt32(expected[i]) | UInt32(expected[i + 1]) << 8
                            let x = Double(Float(bitPattern: a << 16)), y = Double(Float(bitPattern: b << 16))
                            let d = x - y
                            squared += d * d; norm += y * y; largest = max(largest, abs(d))
                        }
                        let count = Double(data.count / 2)
                        return (sqrt(squared / count), sqrt(norm / count), largest)
                    }
                }
            }

            func inspect(_ state: QwenModel.State, label: String, storeReference: Bool,
                         compareReference: Bool) throws -> LongContextAnchor {
                guard !state.hasPagedKV, state.valid else { throw CLIError.usage("Long context diagnostics require committed dense state") }
                var values = [String: CacheReliabilityAnchor.Value]()
                for (name, tensor) in try state.namedTensors.sorted(by: { $0.key < $1.key }) {
                    guard tensor.dtype == MLX_BFLOAT16, tensor.nbytes > 0,
                          let lease = model.stateBudget.reserve(bytes: tensor.nbytes * 3 + 65_536, kind: .workspace) else {
                        throw CLIError.usage("Cannot admit one BF16 diagnostic tensor")
                    }
                    func oneTensor() throws -> CacheReliabilityAnchor.Value {
                        defer { lease.release() }
                        do {
                            let bytes = try MoETilingBytes.bytes(tensor)
                            let finite = bytes.withUnsafeBytes { (p: UnsafeRawBufferPointer) -> Bool in
                                for i in stride(from: 0, to: p.count, by: 2) {
                                    if p[i + 1] & 0x7f == 0x7f && p[i] & 0x80 == 0x80 { return false }
                                }
                                return true
                            }
                            let value = CacheReliabilityAnchor.Value(shape: tensor.shape, dtype: Int(tensor.dtype.rawValue),
                                byteCount: bytes.count, sha256: MoETilingBytes.digest(bytes), finite: finite)
                            let path = scratch.appendingPathComponent(name + ".bf16")
                            if storeReference {
                                try bytes.write(to: path, options: .withoutOverwriting)
                                numericalReference[name] = value
                            }
                            if compareReference {
                                guard let expected = numericalReference[name], expected.shape == value.shape,
                                      expected.dtype == value.dtype, expected.byteCount == value.byteCount else {
                                    throw CLIError.usage("Missing matching reference tensor \(name)")
                                }
                                let baseline = try Data(contentsOf: path, options: .mappedIfSafe)
                                guard MoETilingBytes.digest(baseline) == expected.sha256 else {
                                    throw CLIError.usage("Reference tensor checksum changed")
                                }
                                guard finite && expected.finite else {
                                    report["nonfinite_tensor"] = ["label": label, "tensor": name]
                                    throw CLIError.usage("Nonfinite tensor cannot enter numerical comparison")
                                }
                                let d = try difference(bytes, reference: baseline)
                                let bound = atol + rtol * d.referenceRMS
                                let accepted = finite && expected.finite && d.rmse.isFinite && d.rmse <= bound
                                crossRows.append(["label": label, "tensor": name, "offset": state.offset,
                                    "reference_sha256": expected.sha256, "actual_sha256": value.sha256,
                                    "bitwise_exact": expected == value, "rmse": d.rmse, "reference_rms": d.referenceRMS,
                                    "maximum_absolute_error": d.maxAbs, "declared_rmse_bound": bound,
                                    "screening_passed": accepted])
                            }
                            try MX.synchronize()
                            return value
                        } catch {
                            let original = error
                            try MX.synchronize()
                            throw original
                        }
                    }
                    values[name] = try oneTensor()
                }
                let host = state.diagnosticHostValues
                let result = LongContextAnchor(host: host, tensors: values)
                try require(label + "_121_state_tensors_finite", result.finite &&
                    !host.gdnCapturePresent.contains(true) && !host.pleCapturePresent.contains(true))
                return result
            }

            for selectedMode in modes {
                let generator = try QwenGenerator(model: model,
                    prefixCacheLimits: .init(maxEntries: 2, maxBytes: cacheBytes))
                defer { try? generator.clearPrefixCache() }
                let request = QwenGenerationRequest(tokens: tokens, maxTokens: maximum, contextLimit: context,
                    prefillChunk: 416, mtpDepth: 0, prefillEvaluateEveryLayers: 4,
                    prefillAttention: selectedMode, prefixCacheMaxTokens: boundary)
                var coldAnchors = [Int: LongContextAnchor]()
                var coldResult: QwenGenerationResult?
                for warm in [false, true] {
                    let label = selectedMode.rawValue + (warm ? "_warm" : "_cold")
                    FileHandle.standardError.write(Data("Long context \(context): \(label)\n".utf8))
                    let memory = LongContextProbeMemory()
                    var diagnosticSeconds = 0.0, prefillDiagnostic = 0.0, decodeDiagnostic = 0.0
                    var ordinaryDecodeDiagnostic = 0.0, activePrefill = 0.0, activeDecode = 0.0
                    var callbacks = [Int32](), observedOffsets = [Int](), phase = "prefill"
                    var roundStart: UInt64 = 0, roundBusinessSeconds = 0.0
                    var lastProgress = now()
                    func observe(_ event: String, _ state: QwenModel.State) throws {
                        let began = now()
                        if phase == "decode", event == "decode" { roundBusinessSeconds += Double(began - roundStart) * 1e-9 }
                        try memory.business(phase, offset: state.offset)
                        try memory.resetPeak()
                        let numerical = mode == "screen" && !warm && event == "firstToken"
                        let value = try inspect(state, label: label + "_" + event + "_\(state.offset)",
                            storeReference: numerical && selectedMode == .reference,
                            compareReference: numerical && selectedMode == .fusedQSA)
                        let expectedLabel = selectedMode.rawValue + "_cold_offset\(state.offset)"
                        let exact: Bool
                        if warm { exact = coldAnchors[state.offset].map { value.matches($0) } ?? false }
                        else if let old = coldAnchors[state.offset] { exact = value.matches(old) }
                        else { coldAnchors[state.offset] = value; exact = true }
                        states.append(["label": label, "event": event, "offset": state.offset,
                            "oracle_anchor_label": expectedLabel, "establishes_oracle": !warm,
                            "state_anchor_label": warm ? label + "_offset\(state.offset)" : expectedLabel,
                            "exact_cold_warm": exact, "anchor": try object(value)])
                        try require(label + "_offset\(state.offset)_exact_cold_warm", exact)
                        observedOffsets.append(state.offset)
                        try memory.diagnostic()
                        try memory.resetPeak()
                        let duration = elapsed(began)
                        diagnosticSeconds += duration
                        if phase == "prefill" { prefillDiagnostic += duration }
                        else {
                            decodeDiagnostic += duration
                            if event == "decode" { ordinaryDecodeDiagnostic += duration }
                        }
                    }
                    generator.prefixStateObserver = { event, state in
                        // The real restore is compared to coldBoundary. A
                        // duplicate publish hash would add a 7 GB read without
                        // testing another independently consumed state.
                        if event != "publish" { try observe(event, state) }
                    }
                    generator.decodeStateObserver = { event, state in try observe(event, state) }
                    defer { generator.prefixStateObserver = nil; generator.decodeStateObserver = nil }
                    try MX.synchronize(); try memory.resetPeak()
                    let prefillStart = now(), beforeBeginDiagnostic = diagnosticSeconds
                    let producer = try generator.beginPrefill(request)
                    activePrefill += elapsed(prefillStart) - (diagnosticSeconds - beforeBeginDiagnostic)
                    defer { try? producer.discard() }
                    var prepared: QwenPrefillResult?
                    while prepared == nil {
                        let start = now(), beforeDiagnostic = diagnosticSeconds
                        prepared = try generator.stepPrefill(producer)
                        activePrefill += elapsed(start) - (diagnosticSeconds - beforeDiagnostic)
                        try memory.business("prefill", offset: producer.processedTokenCount)
                        if elapsed(lastProgress) >= 15 {
                            FileHandle.standardError.write(Data("Long context \(label): prefill \(producer.processedTokenCount)/\(promptCount)\n".utf8))
                            lastProgress = now()
                        }
                    }
                    guard let ready = prepared else { throw CLIError.usage("Missing long context prefill result") }
                    defer { ready.discard() }
                    try require(label + "_prefill_accounting", ready.statistics.cachedTokenCount == (warm ? boundary : 0) &&
                        ready.statistics.actualForwardTokenCount == promptCount - (warm ? boundary : 0) &&
                        ready.statistics.attentionMode == selectedMode.rawValue && ready.statistics.evaluateEveryLayers == 4)
                    phase = "decode"
                    try memory.resetPeak()
                    let consumer = try generator.beginDecode(ready)
                    defer { try? consumer.discard() }
                    var result: QwenGenerationResult?
                    while result == nil && callbacks.count < maximum {
                        let ordinal = callbacks.count, beforeDiagnostic = diagnosticSeconds
                        roundStart = now()
                        result = try generator.stepDecode(consumer) { callbacks.append($0) }
                        if ordinal > 0 { activeDecode += elapsed(roundStart) - (diagnosticSeconds - beforeDiagnostic) }
                        try memory.business(ordinal == 0 ? "first_token_publication" : "decode", offset: promptCount + max(0, callbacks.count - 1))
                    }
                    guard let result, let phases = result.phases else { throw CLIError.usage("No bounded long context generation result") }
                    let actualDecode = result.statistics.decodeRounds >= 1 && result.statistics.decodedTokenCount >= 1
                    checks[label + "_actual_decode_evaluated"] = actualDecode
                    let outputValid = result.tokens == callbacks && result.tokens.count <= maximum &&
                        !result.tokens.isEmpty && result.tokens.allSatisfy { $0 >= 0 && Int($0) < configuration.vocabularySize } &&
                        !result.tokens.dropLast().contains(where: tokenizer.eosTokenIDs.contains) &&
                        (result.finishReason == .eos ? tokenizer.eosTokenIDs.contains(result.tokens.last!) :
                            result.finishReason == .length && result.tokens.count == maximum &&
                            !tokenizer.eosTokenIDs.contains(result.tokens.last!)) &&
                        result.statistics.finalStateOffset == promptCount + result.tokens.count - 1 &&
                        consumer.isFinished && !ready.isReady &&
                        phases.prefill.attentionMode == selectedMode.rawValue && result.statistics.mtpDepth == 0
                    try require(label + "_output_and_mode_contract", outputValid)
                    if warm {
                        try require(label + "_complete_ids_and_finish_exact", result.tokens == coldResult?.tokens &&
                            result.finishReason == coldResult?.finishReason)
                    } else { coldResult = result }
                    if selectedMode == .reference && !warm { referenceResult = result }
                    if mode == "screen", selectedMode == .fusedQSA && !warm {
                        checks["cross_attention_output_ids_exact"] = result.tokens == referenceResult?.tokens &&
                            result.finishReason == referenceResult?.finishReason
                        checks["cross_attention_declared_numerical_screen"] = crossRows.count == 121 &&
                            crossRows.allSatisfy { $0["screening_passed"] as? Bool == true }
                    }
                    let decodeWithoutDiagnostics = result.decodeSeconds - ordinaryDecodeDiagnostic
                    try require(label + "_finite_separate_phase_timings", [activePrefill, activeDecode,
                        roundBusinessSeconds, decodeWithoutDiagnostics, phases.prefill.targetSeconds].allSatisfy {
                            $0.isFinite && $0 >= -0.001
                        })
                    trials.append(["label": label, "attention_mode": selectedMode.rawValue,
                        "cache_state": warm ? "warm" : "cold", "actual_decode_evaluated": actualDecode,
                        "capacity_only_no_decode": !actualDecode, "raw_generation_result": try object(result),
                        "output_text": try tokenizer.decode(result.tokens, skipSpecialTokens: true),
                        "prefill_active_seconds_excluding_diagnostics": max(0, activePrefill),
                        "prefill_target_forward_seconds": phases.prefill.targetSeconds,
                        "decode_step_wall_seconds_excluding_diagnostics": max(0, activeDecode),
                        "decode_round_seconds_excluding_diagnostics": max(0, decodeWithoutDiagnostics),
                        "decode_entry_to_observer_seconds": roundBusinessSeconds,
                        "diagnostic_seconds": diagnosticSeconds, "prefill_diagnostic_seconds": prefillDiagnostic,
                        "decode_diagnostic_seconds": decodeDiagnostic,
                        "business_allocator_peak_bytes_by_phase": memory.businessPeaks,
                        "diagnostic_allocator_peak_bytes": memory.diagnosticPeak,
                        "memory_samples": memory.rows, "state_observation_offsets": observedOffsets,
                        "prefix_cache": try object(generator.prefixCacheStatistics),
                        "state_budget_after_request": try object(model.stateBudget.statistics)])
                    try save()
                }
                try generator.clearPrefixCache()
                try MX.synchronize()
                try require(selectedMode.rawValue + "_all_request_cache_and_workspace_leases_released",
                    model.stateBudget.statistics.totalBytes == 0 && model.stateBudget.statistics.currentLeases == 0)
            }
            report["final_state_budget"] = try object(model.stateBudget.statistics)
            report["final_mlx_memory"] = try MX.memory()
            try save(complete: true)
            if checks.values.contains(false) {
                throw CLIError.usage("Long context screening or actual-decode gate failed; complete report retained")
            }
        } catch {
            report["error"] = String(describing: error)
            try? save(complete: report["complete"] as? Bool ?? false)
            throw error
        }
    }
}
