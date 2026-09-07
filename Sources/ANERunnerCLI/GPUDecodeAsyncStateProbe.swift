import ANERunnerGPU
import CMLX
import Foundation

extension RunnerCLI {
    /// One loaded model and one known nonterminal AR position. Diagnostic
    /// readback/deep copies are outside any throughput claim.
    static func probeGPUDecodeAsyncState(_ args: Arguments) throws {
        try args.validate(["--scenario", "--model-dir", "--reference-report", "--output"])
        let output = try args.require("--output")
        guard !FileManager.default.fileExists(atPath: output) else {
            throw CLIError.usage("Decode async state probe requires a new output file")
        }
        var report: [String: Any] = ["schema": "qwen38-decode-async-one-step-v1", "complete": false,
            "passed": false, "full_model_instances": 1, "mtp_head_instances": 0,
            "scope": "Exact same-checkpoint AR state/logits comparison; not throughput or MTP validation.",
            "checkpoint_contract": "checkpoint()/State assignment are shallow. Each branch uses evaluated GPUVerificationCopy batch gathers into independent compact storage."]
        var checks = [String: Bool]()
        func write() throws {
            report["checks"] = checks
            try emit(report, to: output)
        }
        func object<T: Encodable>(_ value: T) throws -> Any {
            try JSONSerialization.jsonObject(with: JSONEncoder().encode(value))
        }
        do {
            let directory = URL(fileURLWithPath: try args.require("--model-dir")).standardizedFileURL.resolvingSymlinksInPath()
            let referenceURL = URL(fileURLWithPath: try args.require("--reference-report"))
            let referenceData = try Data(contentsOf: referenceURL)
            guard let reference = try JSONSerialization.jsonObject(with: referenceData) as? [String: Any],
                  let trial = (reference["trials"] as? [[String: Any]])?.first,
                  let promptInts = trial["prompt_tokens"] as? [Int],
                  let goldenInts = trial["generated_token_ids"] as? [Int],
                  let sourceModel = reference["model_directory"] as? String,
                  URL(fileURLWithPath: sourceModel).standardizedFileURL.resolvingSymlinksInPath() == directory,
                  let provenance = reference["provenance"] as? [String: Any],
                  let metadata = provenance["model_metadata_sha256"] as? [String: String],
                  promptInts.count == 11_057, goldenInts.count == 128 else {
                throw CLIError.usage("Require the frozen 11,057-token / AR128 reference report")
            }
            let prompt = try promptInts.map { value -> Int32 in
                guard let value = Int32(exactly: value) else { throw CLIError.usage("Invalid prompt ID") }
                return value
            }
            let golden = try goldenInts.map { value -> Int32 in
                guard let value = Int32(exactly: value) else { throw CLIError.usage("Invalid output ID") }
                return value
            }
            var metadataChecks = [String: Bool]()
            for name in ["config.json", "model.safetensors.index.json", "tokenizer.json", "chat_template.jinja"] {
                metadataChecks[name] = try GPUProbeSupport.hash(directory.appendingPathComponent(name)) == metadata[name]
            }
            checks["model_metadata_exact"] = metadataChecks.values.allSatisfy { $0 }
            guard checks["model_metadata_exact"] == true else { throw CLIError.usage("Model metadata differs from reference") }
            report["reference_report"] = ["path": referenceURL.path, "sha256": GPUProbeSupport.digest(referenceData)]
            report["model_directory"] = directory.path; report["model_metadata_sha256"] = metadata
            let executable = Bundle.main.executableURL ?? URL(fileURLWithPath: CommandLine.arguments[0])
            report["executable_sha256"] = try GPUProbeSupport.hash(executable)
            var previousCache = 0
            try MX.check(mlx_set_cache_limit(&previousCache, 256 * 1024 * 1024), "decode state probe cache")
            defer { var ignored = 0; _ = mlx_set_cache_limit(&ignored, previousCache) }
            let model = try QwenModel(modelDirectory: directory)
            var finishedDeviceWork = false
            defer { if !finishedDeviceWork { try? MX.synchronize() } }
            guard model.experimentalDecodeAsyncEveryLayers == 8 else {
                throw CLIError.usage("Decode async state probe requires ANERUNNER_EXPERIMENTAL_DECODE_ASYNC_LAYERS=8")
            }
            report["experimental_decode_async_every_layers"] = model.experimentalDecodeAsyncEveryLayers
            let tokenizer = try QwenTokenizer(modelDirectory: directory)
            guard !tokenizer.eosTokenIDs.contains(golden[0]) else { throw CLIError.usage("Reference first token must be nonterminal") }
            var state = model.makeState(), cursor = 0, pending: Int32 = 0
            let prefetch = try model.makePrefillPrefetch(tokens: prompt, chunk: 416, state: state)
            defer { prefetch.finish() }
            while cursor < prompt.count {
                let end = cursor < prompt.count - 1 ? min(prompt.count - 1, cursor + 416) : prompt.count
                let out = try model.forward(tokens: Array(prompt[cursor..<end]), state: &state,
                    prefillPrefetch: prefetch, phase: .prefill)
                if end == prompt.count {
                    guard let logits = out.logits else { throw CLIError.usage("Missing prompt logits") }
                    let selected = try model.greedyToken(logits)
                    try model.evaluate([selected], state: &state)
                    pending = try selected.uint32TokenID()
                } else { try model.evaluate([out.stream], state: &state) }
                cursor = end
            }
            prefetch.finish()
            checks["prefill_did_not_submit_experiment"] = model.experimentalDecodeAsyncSubmissions == 0
            checks["prefill_first_token_exact"] = pending == golden[0]
            guard checks.values.allSatisfy({ $0 }) else { throw CLIError.usage("AR prefill prerequisite failed") }
            let checkpoint = try model.checkpoint(state: &state)
            let checkpointHost = checkpoint.diagnosticHostValues
            let checkpointValues = try DecodeAsyncSavedTensor.snapshot(checkpoint.namedTensors)
            var baseline = try model.diagnosticPrivateStateCopy(checkpoint)
            var candidate = try model.diagnosticPrivateStateCopy(checkpoint)
            let initialBaseline = try DecodeAsyncSavedTensor.compare(checkpointValues, baseline.namedTensors)
            let initialCandidate = try DecodeAsyncSavedTensor.compare(checkpointValues, candidate.namedTensors)
            report["initial_baseline_copy"] = initialBaseline.rows
            report["initial_candidate_copy"] = initialCandidate.rows
            report["checkpoint_host"] = try object(checkpointHost)
            report["private_copy_host"] = try object(baseline.diagnosticHostValues)
            checks["deep_copies_preserve_every_tensor"] = initialBaseline.exact && initialCandidate.exact
            checks["private_copy_host_values_exact"] = baseline.diagnosticHostValues == candidate.diagnosticHostValues
            // Compact gather allocations may normalize retained extents; logical
            // host values must still equal the source checkpoint exactly.
            let host = baseline.diagnosticHostValues
            checks["private_copy_logical_host_exact"] = host.offset == checkpointHost.offset && host.valid == checkpointHost.valid &&
                host.gdnOffsets == checkpointHost.gdnOffsets && host.attentionOffsets == checkpointHost.attentionOffsets &&
                host.pleHistory == checkpointHost.pleHistory && host.gdnCapturePresent == checkpointHost.gdnCapturePresent &&
                host.pleCapturePresent == checkpointHost.pleCapturePresent
            guard checks.values.allSatisfy({ $0 }) else { throw CLIError.usage("Deep-copy prerequisite failed") }
            let baselineBefore = model.experimentalDecodeAsyncSubmissions
            let baselineOut = try model.forward(tokens: [pending], state: &baseline, phase: .decode,
                allowExperimentalDecodeAsync: false)
            guard let baselineLogits = baselineOut.logits else { throw CLIError.usage("Missing baseline logits") }
            let baselineSelected = try model.greedyToken(baselineLogits)
            try model.evaluate([baselineSelected], state: &baseline)
            let baselineToken = try baselineSelected.uint32TokenID()
            let baselineCalls = model.experimentalDecodeAsyncSubmissions - baselineBefore
            let baselineValues = try DecodeAsyncSavedTensor.snapshot(baseline.namedTensors)
            let logitValues = try DecodeAsyncSavedTensor.snapshot(["logits": baselineLogits])
            let candidateBefore = model.experimentalDecodeAsyncSubmissions
            let candidateOut = try model.forward(tokens: [pending], state: &candidate, phase: .decode,
                allowExperimentalDecodeAsync: true)
            guard let candidateLogits = candidateOut.logits else { throw CLIError.usage("Missing candidate logits") }
            let candidateSelected = try model.greedyToken(candidateLogits)
            try model.evaluate([candidateSelected], state: &candidate)
            let candidateToken = try candidateSelected.uint32TokenID()
            let candidateCalls = model.experimentalDecodeAsyncSubmissions - candidateBefore
            let logits = try DecodeAsyncSavedTensor.compare(logitValues, ["logits": candidateLogits])
            let persistent = try DecodeAsyncSavedTensor.compare(baselineValues, candidate.namedTensors)
            let unchanged = try DecodeAsyncSavedTensor.compare(checkpointValues, checkpoint.namedTensors)
            report["complete_logits"] = logits.rows; report["complete_persistent_tensors"] = persistent.rows
            report["checkpoint_after_branches"] = unchanged.rows
            report["baseline_host"] = try object(baseline.diagnosticHostValues)
            report["candidate_host"] = try object(candidate.diagnosticHostValues)
            report["tokens"] = ["pending": pending, "baseline": baselineToken, "candidate": candidateToken, "golden": golden[1]]
            report["experimental_host_calls"] = ["baseline": baselineCalls, "candidate": candidateCalls]
            checks["complete_logits_bitwise_finite"] = logits.exact
            checks["all_persistent_tensors_bitwise_finite_nil_shape_dtype"] = persistent.exact
            checks["all_host_integers_history_and_flags_exact"] = baseline.diagnosticHostValues == candidate.diagnosticHostValues
            checks["checkpoint_immutable"] = unchanged.exact && checkpoint.diagnosticHostValues == checkpointHost
            checks["one_step_tokens_and_offsets_exact"] = baselineToken == candidateToken && baselineToken == golden[1] &&
                baseline.offset == prompt.count + 1 && candidate.offset == prompt.count + 1 &&
                baseline.qsaActiveLayers == 12 && candidate.qsaActiveLayers == 12
            checks["submission_counts_exact"] = baselineCalls == 0 && candidateCalls == 5
            checks["no_mtp_weights_loaded"] = !model.weights.ledger.contains { $0.name.contains(".mtp.") }
            report["complete"] = true; report["passed"] = checks.values.allSatisfy { $0 }
            finishedDeviceWork = true
            try write()
            guard checks.values.allSatisfy({ $0 }) else { throw CLIError.usage("Decode async one-step numerical gate failed") }
        } catch {
            report["passed"] = false; report["error"] = String(describing: error)
            try write()
            throw error
        }
    }
}

private struct DecodeAsyncSavedTensor {
    let shape: [Int]
    let dtype: Int
    let bytes: Data
    let finite: Bool

    static func snapshot(_ tensors: [String: Tensor]) throws -> [String: Self] {
        var saved = [String: Self]()
        for (name, value) in tensors {
            let floating = [MLX_BFLOAT16, MLX_FLOAT16, MLX_FLOAT32, MLX_FLOAT64].contains(value.dtype)
            let finite: Bool
            if floating { finite = try value.floats().allSatisfy(\.isFinite) }
            else { finite = true }
            saved[name] = Self(shape: value.shape, dtype: Int(value.dtype.rawValue),
                bytes: try MoETilingBytes.bytes(value), finite: finite)
        }
        return saved
    }
    static func compare(_ before: [String: Self], _ tensors: [String: Tensor]) throws -> (rows: [[String: Any]], exact: Bool) {
        let after = try snapshot(tensors)
        var rows = [[String: Any]](), exact = true
        for name in Set(before.keys).union(after.keys).sorted() {
            guard let a = before[name], let b = after[name] else {
                rows.append(["name": name, "exact": false, "nil_mismatch": true]); exact = false; continue
            }
            let equal = a.shape == b.shape && a.dtype == b.dtype && a.bytes == b.bytes && a.finite && b.finite
            exact = exact && equal
            let row: [String: Any] = ["name": name, "shape_a": a.shape, "shape_b": b.shape,
                "dtype_a": a.dtype, "dtype_b": b.dtype, "byte_count_a": a.bytes.count, "byte_count_b": b.bytes.count,
                "sha256_a": GPUProbeSupport.digest(a.bytes), "sha256_b": GPUProbeSupport.digest(b.bytes),
                "all_finite": a.finite && b.finite, "exact": equal]
            rows.append(row)
        }
        return (rows, exact)
    }
}
