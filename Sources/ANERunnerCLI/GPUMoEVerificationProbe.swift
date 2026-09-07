import ANERunnerCore
import ANERunnerGPU
import CMLX
import Dispatch
import Foundation

extension RunnerCLI {
    static func probeGPUMoEVerification(_ args: Arguments) throws {
        try args.validate(["--model-dir", "--output", "--layer", "--warmups", "--runs",
                           "--prefill-fixture", "--decode-fixture", "--candidate"])
        let candidatePolicy = args["--candidate"] ?? "tokenAxis"
        guard ["tokenAxis", "sharedElementwise"].contains(candidatePolicy) else {
            throw CLIError.usage("--candidate must be tokenAxis or sharedElementwise")
        }
        let sharedElementwise = candidatePolicy == "sharedElementwise"
        let lengths = sharedElementwise ? [2, 3] : [2, 3, 5]
        let output = try args.require("--output")
        guard !FileManager.default.fileExists(atPath: output),
              let layer = Int(args["--layer"] ?? "0"), layer >= 0,
              let warmups = Int(args["--warmups"] ?? "3"), (1...10).contains(warmups),
              let runs = Int(args["--runs"] ?? "12"), (10...20).contains(runs), runs.isMultiple(of: 2) else {
            throw CLIError.usage("Use a new --output, layer >= 0, warmups 1...10, and an even runs count in 10...20")
        }
        var cases: [[String: Any]] = []
        var report: [String: Any] = [
            "schema": sharedElementwise ? "qwen38-moe-verification-shared-elementwise-v1" : "qwen38-moe-verification-token-axis-v1",
            "complete": false, "passed": false, "candidate_policy": candidatePolicy,
            "layer_index": layer, "warmups_per_variant": warmups, "samples_per_variant": runs,
            "baseline": "verificationLinear enabled; routed experts run existing per-token S1 kernels",
            "candidate": sharedElementwise
                ? "verificationLinear enabled; per-token routed experts unchanged; verificationSharedElementwise=true for S2/S3"
                : "verificationLinear enabled; verificationTokenAxis=true",
            "full_model_generation": false, "performance_threshold_applied": false,
            "timing_scope": "Resident BF16 input and original Q4 weights; Swift forward construction plus GPU execution and y.eval. No diagnostics, host output copies, input parsing, or model loading in samples.",
            "physical_dram_bandwidth_gbps": NSNull(),
            "notes": [sharedElementwise
                        ? "Four cases: S2/S3 mixed/repeated captured rows; reference initialization leaves S1 shared fusion disabled. Only the new explicit verification selector can select the candidate tails."
                        : "Six cases: S2/S3/S5 with mixed captured rows and repeated captured rows.",
                      "Default activations are actual layer-0 rows, reordered/repeated for an independent MoE operator test, not a new autoregressive trajectory.",
                      "Expert overlap counts logical routing assignments; hardware weight-cache reuse is not measured.",
                      "Passed means bitwise correctness and completed timing collection, not a whole-model speedup gate."]
        ]
        func save() throws {
            report["cases"] = cases
            try emit(report, to: output)
        }
        do {
            let modelURL = URL(fileURLWithPath: try args.require("--model-dir"))
                .standardizedFileURL.resolvingSymlinksInPath()
            let c = try QwenConfiguration(modelDirectory: modelURL)
            guard layer < c.layerCount else { throw CLIError.usage("Requested layer is outside the model") }
            let fixtureRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
                .deletingLastPathComponent().deletingLastPathComponent()
                .appendingPathComponent("fixtures/moe-real/converted")
            let fixturePaths = [
                args["--prefill-fixture"] ?? fixtureRoot.appendingPathComponent("prefill.json").path,
                args["--decode-fixture"] ?? fixtureRoot.appendingPathComponent("decode.json").path
            ]
            var rows: [[Float]] = [], rowSources: [[String: Any]] = [], sources: [[String: Any]] = []
            for path in fixturePaths {
                let url = URL(fileURLWithPath: path).standardizedFileURL.resolvingSymlinksInPath()
                let fixture = try CoreMLBlockFixture.load(from: url)
                guard fixture.inputs.count == 1, let input = fixture.inputs["x"],
                      input.shape.count == 3, input.shape[0] == 1, input.shape[1] > 0,
                      input.shape[2] == c.hiddenSize,
                      input.dtype == .float16 || input.dtype == .float32,
                      input.values.allSatisfy({ Float($0).isFinite }) else {
                    throw CLIError.usage("MoE fixture needs one finite floating x[1,T,hiddenSize]")
                }
                for row in 0..<input.shape[1] {
                    let start = row * c.hiddenSize
                    rows.append(input.values[start..<(start + c.hiddenSize)].map(Float.init))
                    rowSources.append(["path": url.path, "row": row])
                }
                sources.append(["path": url.path, "sha256": try GPUProbeSupport.hash(url), "shape": input.shape])
            }
            guard rows.count >= 3 else { throw CLIError.usage("Provide at least three actual input rows across the two fixtures") }
            report["model_directory"] = modelURL.path
            report["config_sha256"] = try GPUProbeSupport.hash(modelURL.appendingPathComponent("config.json"))
            report["weight_index_sha256"] = try GPUProbeSupport.hash(modelURL.appendingPathComponent("model.safetensors.index.json"))
            report["input_fixtures"] = sources; report["row_bank"] = rowSources
            report["hidden_size"] = c.hiddenSize; report["top_k"] = c.expertsPerToken
            report["weight_precision"] = "Original affine Q4 data with BF16 scales/biases; no requantization"
            let loadStart = DispatchTime.now().uptimeNanoseconds
            let weights = try GPUWeights(modelDirectory: modelURL)
            let moe = try GPUMoE(weights: weights, layer: layer, hiddenSize: c.hiddenSize,
                experts: c.expertCount, topK: c.expertsPerToken,
                groupSize: c.quantizationGroupSize, bits: c.quantizationBits,
                prefillAccumulation: .reference)
            let linear = try GPUVerificationLinear()
            try MX.synchronize()
            report["load_milliseconds"] = GPUProbeSupport.elapsed(loadStart)
            report["loaded_memory"] = try MX.memory()
            let prefix = "language_model.model.layers.\(layer).mlp."
            guard weights.ledger.allSatisfy({ $0.name.hasPrefix(prefix) }) else {
                throw GPUError.invalid("Probe unexpectedly loaded a tensor outside the selected MoE")
            }
            report["loaded_source_tensors"] = try weights.ledger.map { entry -> [String: Any] in
                let m = try weights.metadata(entry.name)
                return ["name": m.name, "shard": m.shard, "shape": m.shape,
                        "dtype": m.dtypeName, "source_bytes": m.byteCount]
            }
            report["loaded_source_hash_scope"] = "Config/index and fixture hashes only; tensor entries are metadata, not a full weight-content checksum."
            report["mlx_version"] = try GPUProbeSupport.mlxVersion()
            report["operating_system"] = ProcessInfo.processInfo.operatingSystemVersionString

            var firstSharedCase: (x: Tensor, reference: GPUMoEOutput)?
            for length in lengths {
                for pattern in ["mixed", "repeated"] {
                    let rowIDs = (0..<length).map { pattern == "mixed" ? $0 % rows.count : rows.count - 1 }
                    let x = try MX.array(rowIDs.flatMap { rows[$0] }, shape: [1,length,c.hiddenSize], dtype: MLX_BFLOAT16)
                    try x.eval()
                    func forward(_ candidate: Bool, diagnostics: Bool) throws -> GPUMoEOutput {
                        try moe.forward(x, diagnostics: diagnostics, verificationLinear: linear,
                                        verificationTokenAxis: candidate && !sharedElementwise,
                                        verificationSharedElementwise: candidate && sharedElementwise)
                    }
                    func diagnostics(_ candidate: Bool) throws -> GPUMoEOutput {
                        let out = try forward(candidate, diagnostics: true)
                        try MX.eval(Array(out.diagnostics.values) + [out.y])
                        return out
                    }
                    let a = try diagnostics(false), b = try diagnostics(true)
                    if sharedElementwise, firstSharedCase == nil { firstSharedCase = (x, a) }
                    var at = a.diagnostics, bt = b.diagnostics
                    at["y"] = a.y; bt["y"] = b.y
                    guard !at.isEmpty, Set(at.keys) == Set(bt.keys),
                          at["selected_experts"] != nil, at["selected_expert_outputs"] != nil else {
                        throw GPUError.invalid("Verification variants returned different or incomplete diagnostic names")
                    }
                    var comparisons: [String: Any] = [:], exact = true
                    for name in at.keys.sorted() {
                        let comparison = try GPUMoEVerificationBytes.compare(at[name]!, bt[name]!)
                        comparisons[name] = comparison
                        exact = exact && (comparison["exact"] as? Bool == true)
                    }
                    // Exercise the same diagnostics=false branch that is timed,
                    // and ensure extra diagnostic outputs did not change y.
                    let an = try forward(false, diagnostics: false); try an.y.eval()
                    let bn = try forward(true, diagnostics: false); try bn.y.eval()
                    for (name, lhs, rhs) in [("timed_y_A_vs_B", an.y, bn.y),
                                             ("A_y_diagnostics_vs_timed", a.y, an.y),
                                             ("B_y_diagnostics_vs_timed", b.y, bn.y)] {
                        let comparison = try GPUMoEVerificationBytes.compare(lhs, rhs)
                        comparisons[name] = comparison
                        exact = exact && (comparison["exact"] as? Bool == true)
                    }
                    let ids = try a.diagnostics["selected_experts"]!.ints().map(Int.init)
                    guard ids.count == length * c.expertsPerToken,
                          ids.allSatisfy({ (0..<c.expertCount).contains($0) }) else {
                        throw GPUError.invalid("Invalid routed expert IDs")
                    }
                    let perToken = (0..<length).map { token in
                        Array(ids[(token*c.expertsPerToken)..<((token+1)*c.expertsPerToken)])
                    }
                    guard perToken.allSatisfy({ Set($0).count == c.expertsPerToken }) else {
                        throw GPUError.invalid("Duplicate expert inside one top-k row")
                    }
                    var pairOverlap: [[String: Any]] = []
                    for i in 0..<length {
                        for j in (i+1)..<length {
                            pairOverlap.append(["row_a": i, "row_b": j,
                                "shared_experts": Set(perToken[i]).intersection(perToken[j]).sorted()])
                        }
                    }
                    var item: [String: Any] = [
                        "name": "s\(length)_\(pattern)", "input_shape": x.shape, "row_bank_indices": rowIDs,
                        "candidate_shared_elementwise_sequence": sharedElementwise ? length : NSNull(),
                        "input_bf16_sha256": GPUProbeSupport.digest(try GPUMoEVerificationBytes.read(x)),
                        "bitwise_passed": exact, "diagnostic_tensor_count_including_y": at.count,
                        "comparisons": comparisons, "routing_expert_ids_per_token": perToken,
                        "routing_assignment_count": ids.count, "unique_expert_count": Set(ids).count,
                        "repeated_assignment_count": ids.count - Set(ids).count,
                        "expert_overlap_between_tokens": pairOverlap
                    ]
                    if sharedElementwise {
                        let gates = try a.diagnostics["shared_gate"]!.floats()
                        item["baseline_shared_gate_values"] = gates
                        item["baseline_shared_gate_logits"] = try a.diagnostics["shared_gate_logits"]!.floats()
                        item["distinct_shared_gate_bit_patterns"] = Set(gates.map { $0.bitPattern }).count
                    }
                    guard exact else {
                        cases.append(item)
                        throw GPUError.invalid("Bitwise verification failed for s\(length)_\(pattern); timing stopped")
                    }
                    func timed(_ candidate: Bool) throws -> Double {
                        let start = DispatchTime.now().uptimeNanoseconds
                        let out = try forward(candidate, diagnostics: false)
                        try out.y.eval()
                        let ms = GPUProbeSupport.elapsed(start)
                        guard ms > 0, ms.isFinite else { throw GPUError.invalid("Invalid timing sample") }
                        return ms
                    }
                    var warm: [[String: Any]] = [], samples: [[String: Any]] = []
                    var baseline: [Double] = [], candidate: [Double] = []
                    for round in 0..<warmups {
                        for variant in (round.isMultiple(of: 2) ? [false, true] : [true, false]) {
                            warm.append(["variant": variant ? "B" : "A", "milliseconds": try timed(variant)])
                        }
                    }
                    // Alternating AB and BA pairs produce ABBA blocks and give
                    // each variant the same number of samples and positions.
                    for round in 0..<runs {
                        for variant in (round.isMultiple(of: 2) ? [false, true] : [true, false]) {
                            let ms = try timed(variant)
                            if variant { candidate.append(ms) } else { baseline.append(ms) }
                            samples.append(["index": samples.count, "variant": variant ? "B" : "A", "milliseconds": ms])
                        }
                    }
                    let am = GPUProbeSupport.percentile(baseline, 0.5), bm = GPUProbeSupport.percentile(candidate, 0.5)
                    item["warmup_samples"] = warm; item["interleaved_samples"] = samples
                    item["baseline_milliseconds"] = baseline; item["candidate_milliseconds"] = candidate
                    item["baseline_median_milliseconds"] = am; item["candidate_median_milliseconds"] = bm
                    item["baseline_over_candidate_median_ratio"] = am / bm
                    cases.append(item)
                    try save()
                }
            }
            if let firstSharedCase {
                // Return to S2 after S3 on the same object: catch cache/config
                // cross-talk without adding another benchmark or input fixture.
                let diagnostic = try moe.forward(firstSharedCase.x, diagnostics: true,
                    verificationLinear: linear, verificationSharedElementwise: true)
                let ordinary = try moe.forward(firstSharedCase.x, diagnostics: false,
                    verificationLinear: linear, verificationSharedElementwise: true)
                let comparisons = [
                    "diagnostic_y": try GPUMoEVerificationBytes.compare(firstSharedCase.reference.y, diagnostic.y),
                    "diagnostic_gate": try GPUMoEVerificationBytes.compare(firstSharedCase.reference.diagnostics["shared_gate"]!, diagnostic.diagnostics["shared_gate"]!),
                    "ordinary_y": try GPUMoEVerificationBytes.compare(firstSharedCase.reference.y, ordinary.y)
                ]
                let exact = comparisons.values.allSatisfy { $0["exact"] as? Bool == true }
                report["cache_return_s2"] = ["comparisons": comparisons, "passed": exact]
                guard exact else { throw GPUError.invalid("S2 shared-tail cache return after S3 was not bitwise exact") }
            }
            report["final_memory"] = try MX.memory()
            report["complete"] = true; report["passed"] = cases.count == lengths.count * 2
            try save()
        } catch {
            report["fatal_error"] = String(describing: error)
            report["passed"] = false
            try save()
            throw error
        }
    }
}

private enum GPUMoEVerificationBytes {
    /// Public dtype view preserves native BF16/FP32/integer bit patterns.
    /// Contiguity converts storage strides to logical row-major order only.
    static func read(_ tensor: Tensor) throws -> Data {
        let contiguous = try MX.contiguous(tensor)
        let bytes = try MX.output("MoE verification byte view") {
            mlx_view(&$0, contiguous.handle, MLX_UINT8, MX.stream)
        }
        try bytes.eval()
        guard bytes.count == tensor.nbytes, let pointer = mlx_array_data_uint8(bytes.handle) else {
            throw GPUError.invalid("Missing native tensor bytes for verification")
        }
        return Data(bytes: pointer, count: bytes.count)
    }
    static func compare(_ a: Tensor, _ b: Tensor) throws -> [String: Any] {
        let lhs = try read(a), rhs = try read(b)
        let layoutMatch = a.shape == b.shape && a.dtype == b.dtype && lhs.count == rhs.count
        let mismatchIndices = zip(lhs, rhs).enumerated().compactMap { $0.element.0 == $0.element.1 ? nil : $0.offset }
        let floating = [MLX_BFLOAT16, MLX_FLOAT16, MLX_FLOAT32, MLX_FLOAT64].contains(a.dtype)
        let finite: Bool
        if floating {
            let av = try a.floats(), bv = try b.floats()
            finite = av.allSatisfy(\.isFinite) && bv.allSatisfy(\.isFinite)
        } else { finite = true }
        return ["shape_a": a.shape, "shape_b": b.shape,
                "dtype_a": Int(a.dtype.rawValue), "dtype_b": Int(b.dtype.rawValue),
                "element_count_a": a.count, "byte_count_a": lhs.count, "byte_count_b": rhs.count,
                "sha256_a": GPUProbeSupport.digest(lhs), "sha256_b": GPUProbeSupport.digest(rhs),
                "different_byte_count": mismatchIndices.count + abs(lhs.count - rhs.count),
                "first_different_byte_offsets": Array(mismatchIndices.prefix(8)),
                "all_finite": finite, "exact": layoutMatch && lhs == rhs && finite]
    }
}
