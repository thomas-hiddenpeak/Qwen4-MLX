import ANERunnerGPU
import CMLX
import Foundation

extension RunnerCLI {
    /// A resident four-weight QKV probe. Inputs are the existing matvec LCG,
    /// not captured model activations or an autoregressive trajectory.
    static func probeGPUVerificationQKV(_ args: Arguments) throws {
        try args.validate(["--model-dir", "--output"])
        let output = try args.require("--output")
        guard !FileManager.default.fileExists(atPath: output) else {
            throw CLIError.usage("Verification QKV --output must be a new file")
        }
        let layers = [0, 4, 8, 12], sequences = [2, 3]
        let k = 2560, n = 10240, warmupPairs = 3, measuredPairs = 12
        let seed: UInt64 = 0x514D_4154_5645_4300
        var checks: [[String: Any]] = [], warmups: [[String: Any]] = []
        var samples: [[String: Any]] = [], groups: [[String: Any]] = []
        var ordinal = 0
        var report: [String: Any] = [
            "schema": "qwen38-verification-qkv-tm2-probe-v1", "complete": false, "passed": false,
            "layers": layers, "sequences": sequences, "input_size": k, "output_size": n,
            "baseline": "GPUVerificationLinear reference, scalar-order TM4",
            "candidate": "qkvS3TM2: only S3/K2560/N10240/BN1 uses TM2; S2 uses original TM4",
            "warmup_pairs_per_layer_per_sequence": warmupPairs,
            "measured_pairs_per_layer_per_sequence": measuredPairs,
            "input_kind": "Synthetic deterministic finite BF16; same GPUMatvecCase LCG; no real trajectory",
            "input_seed_hex": String(seed, radix: 16),
            "input_generator": "UInt64 LCG; multiplier 6364136223846793005, increment 1442695040888963407; sign=bit63, BF16 exponent=119+bits32..33, mantissa=bits0..6. Generate three rows continuously; S2 is its two-row prefix.",
            "timing_scope": "New verification-linear graph construction plus y.eval wall time. Weight loading, input creation, S1 oracle, native-byte readback and comparisons are outside all sample windows. Not device-only time.",
            "interleave": "For each pair round rotate the starting weight slot by round modulo four. Each layer sees AB in even rounds and BA in odd rounds; 12 rounds yield six nonoverlapping ABBA blocks per layer. All four weights remain resident.",
            "performance_threshold_applied": false, "full_model_generation": false,
            "physical_dram_bandwidth_gbps": NSNull(), "physical_dram_bytes": NSNull(),
            "notes": [
                "Passed means numerical checks and complete sample collection; it does not accept performance or enable a generation mode.",
                "S2 is an unchanged-parameter control. Its timing differences are not TM2 gains.",
                "Source-level accumulators fall from 12 to 6 floats at S3; compiled register allocation and occupancy are not measured.",
                "Initial A/B outputs and each layer/variant's final timed output are compared against an independent repeated S1 matmul oracle. Intermediate timed outputs are evaluated but not individually read back.",
                "All samples are retained, including early samples and both AB/BA orders; medians alone do not establish repeatability."]
        ]
        func save() throws {
            report["checks"] = checks; report["warmup_samples"] = warmups
            report["samples"] = samples; report["layer_groups"] = groups
            try emit(report, to: output)
        }
        func compare(_ a: Tensor, _ b: Tensor, id: String, sequence: Int, slot: Int) throws {
            var result = try GPUMoEVerificationBytes.compare(a, b)
            result["id"] = id; result["sequence"] = sequence
            result["weight_slot"] = slot; result["layer_index"] = layers[slot]
            checks.append(result)
            guard result["exact"] as? Bool == true else {
                throw GPUError.invalid("Verification QKV numerical gate failed: S\(sequence) layer\(layers[slot]) \(id)")
            }
        }
        do {
            // The oracle must retain pinned S1 mode even if a tuning library
            // happens to be loaded. Selection is outside every timing window.
            try GDNGEMVMode.reference.apply()
            defer { try? GDNGEMVMode.reference.apply() }
            let directory = URL(fileURLWithPath: try args.require("--model-dir"))
                .standardizedFileURL.resolvingSymlinksInPath()
            report["model_directory"] = directory.path
            report["config_sha256"] = try GPUProbeSupport.hash(directory.appendingPathComponent("config.json"))
            report["weight_index_sha256"] = try GPUProbeSupport.hash(directory.appendingPathComponent("model.safetensors.index.json"))
            report["mlx_version"] = try GPUProbeSupport.mlxVersion()
            report["process_id"] = ProcessInfo.processInfo.processIdentifier
            report["operating_system"] = ProcessInfo.processInfo.operatingSystemVersionString
            let loadStart = DispatchTime.now().uptimeNanoseconds
            let weights = try GPUWeights(modelDirectory: directory)
            var matrices: [Tensor] = [], descriptions: [[String: Any]] = []
            for (slot, layer) in layers.enumerated() {
                let name = "language_model.model.layers.\(layer).linear_attn.in_proj_qkv.weight"
                let matrix = try weights.tensor(name)
                guard matrix.dtype == MLX_BFLOAT16, matrix.shape == [n, k] else {
                    throw GPUError.invalid("Unexpected QKV source shape/dtype: \(name)")
                }
                let metadata = try weights.metadata(name)
                matrices.append(try MX.transpose(matrix, [1, 0]))
                descriptions.append(["weight_slot": slot, "layer_index": layer, "name": name,
                    "shard": metadata.shard, "shape": metadata.shape, "dtype": "BF16",
                    "source_bytes": metadata.byteCount])
            }
            let sourceBytes = weights.ledger.reduce(UInt64(0)) { $0 + $1.sourceBytes }
            guard matrices.count == 4, weights.ledger.count == 4, sourceBytes == 209_715_200 else {
                throw GPUError.invalid("Verification QKV loaded tensors outside the four-weight contract")
            }
            let inputBank = GPUMatvecCase.syntheticBF16(count: 3 * k, seed: seed)
            var inputs: [Int: Tensor] = [:], inputRecords: [[String: Any]] = []
            for sequence in sequences {
                let bytes = Data(inputBank.prefix(sequence * k * 2))
                let x = try MX.array(data: bytes, shape: [1, sequence, k], dtype: MLX_BFLOAT16)
                inputs[sequence] = x
                inputRecords.append(["sequence": sequence, "shape": x.shape,
                    "bf16_sha256": GPUMatvecCase.sha256(bytes), "bytes": bytes.count])
            }
            try MX.eval(matrices + Array(inputs.values))
            let baseline = try GPUVerificationLinear()
            let candidate = try GPUVerificationLinear(experiment: .qkvS3TM2)
            let kernels = [baseline, candidate]
            report["load_milliseconds"] = GPUProbeSupport.elapsed(loadStart)
            report["loaded_memory"] = try MX.memory()
            report["loaded_source_tensors"] = descriptions
            report["source_weight_bytes"] = sourceBytes
            report["source_hash_scope"] = "Config/index hashes and tensor metadata; root must separately freeze source payload identity. No additional weight buffers or repacking."
            report["inputs"] = inputRecords
            var oracles: [Int: [Tensor]] = [:]
            for sequence in sequences {
                let x = inputs[sequence]!
                let baselineTM = try baseline.rowsPerThread(tokens: sequence, inputSize: k, outputSize: n)
                let candidateTM = try candidate.rowsPerThread(tokens: sequence, inputSize: k, outputSize: n)
                guard baselineTM == 4, candidateTM == (sequence == 3 ? 2 : 4) else {
                    throw GPUError.invalid("Unexpected verification dispatch parameters")
                }
                var references: [Tensor] = []
                for slot in matrices.indices {
                    var rows: [Tensor] = []
                    for token in 0..<sequence {
                        let row = try MX.slice(x, starts: [0, token, 0], ends: [1, token + 1, k])
                        let y = try MX.matmul(row, matrices[slot]); try y.eval()
                        rows.append(y)
                    }
                    let oracle = try MX.concat(rows, axis: 1); try oracle.eval()
                    let a = try baseline.apply(x, weight: matrices[slot])
                    let b = try candidate.apply(x, weight: matrices[slot])
                    try MX.eval([a, b])
                    try compare(a, oracle, id: "initial_A_vs_S1_oracle", sequence: sequence, slot: slot)
                    try compare(b, oracle, id: "initial_B_vs_S1_oracle", sequence: sequence, slot: slot)
                    try compare(a, b, id: "initial_A_vs_B", sequence: sequence, slot: slot)
                    references.append(oracle)
                }
                oracles[sequence] = references
                var last = Array(repeating: [Tensor?](repeating: nil, count: 4), count: 2)
                func timed(_ variant: Int, slot: Int, pair: Int, warm: Bool) throws {
                    let start = DispatchTime.now().uptimeNanoseconds
                    let y = try kernels[variant].apply(x, weight: matrices[slot])
                    let forwardEnd = DispatchTime.now().uptimeNanoseconds
                    try y.eval()
                    let end = DispatchTime.now().uptimeNanoseconds
                    let milliseconds = Double(end - start) * 1e-6
                    guard milliseconds.isFinite, milliseconds > 0 else { throw GPUError.invalid("Invalid QKV timing") }
                    let row: [String: Any] = ["ordinal": ordinal, "sequence": sequence,
                        "weight_slot": slot, "layer_index": layers[slot], "pair_index": pair,
                        "variant": variant == 0 ? "A" : "B", "order": pair.isMultiple(of: 2) ? "AB" : "BA",
                        "start_ns": start, "forward_end_ns": forwardEnd, "evaluation_end_ns": end,
                        "wall_milliseconds": milliseconds,
                        "forward_milliseconds": Double(forwardEnd - start) * 1e-6,
                        "evaluation_wait_milliseconds": Double(end - forwardEnd) * 1e-6]
                    ordinal += 1
                    if warm { warmups.append(row) }
                    else { samples.append(row); last[variant][slot] = y }
                }
                for warm in [true, false] {
                    for pair in 0..<(warm ? warmupPairs : measuredPairs) {
                        for offset in matrices.indices {
                            let slot = (offset + pair) % matrices.count
                            for variant in (pair.isMultiple(of: 2) ? [0, 1] : [1, 0]) {
                                try timed(variant, slot: slot, pair: pair, warm: warm)
                            }
                        }
                    }
                }
                for slot in matrices.indices {
                    for variant in kernels.indices {
                        guard let actual = last[variant][slot] else { throw GPUError.invalid("Missing timed QKV output") }
                        try compare(actual, references[slot], id: variant == 0 ? "timed_A_vs_S1_oracle" : "timed_B_vs_S1_oracle",
                            sequence: sequence, slot: slot)
                    }
                    let rows = samples.filter { $0["sequence"] as? Int == sequence && $0["weight_slot"] as? Int == slot }
                    func values(_ variant: String, order: String? = nil) -> [Double] {
                        rows.filter { $0["variant"] as? String == variant && (order == nil || $0["order"] as? String == order) }
                            .map { $0["wall_milliseconds"] as! Double }
                    }
                    let a = values("A"), b = values("B")
                    guard a.count == measuredPairs, b.count == measuredPairs else { throw GPUError.invalid("Incomplete QKV layer samples") }
                    let am = GPUProbeSupport.percentile(a, 0.5), bm = GPUProbeSupport.percentile(b, 0.5)
                    var blockRatios: [Double] = []
                    for block in 0..<6 {
                        let blockRows = rows.filter { ($0["pair_index"] as! Int) / 2 == block }
                        let at = blockRows.filter { $0["variant"] as? String == "A" }.reduce(0.0) { $0 + ($1["wall_milliseconds"] as! Double) }
                        let bt = blockRows.filter { $0["variant"] as? String == "B" }.reduce(0.0) { $0 + ($1["wall_milliseconds"] as! Double) }
                        blockRatios.append(at / bt)
                    }
                    groups.append(["sequence": sequence, "weight_slot": slot, "layer_index": layers[slot],
                        "baseline_tm": baselineTM, "candidate_tm": candidateTM,
                        "baseline_threadgroups": n / (8 * baselineTM), "candidate_threadgroups": n / (8 * candidateTM),
                        "baseline_median_milliseconds": am, "candidate_median_milliseconds": bm,
                        "baseline_over_candidate_median_ratio": am / bm,
                        "median_difference_microseconds": (am - bm) * 1000,
                        "AB_order_median_ratio": GPUProbeSupport.percentile(values("A", order: "AB"), 0.5) / GPUProbeSupport.percentile(values("B", order: "AB"), 0.5),
                        "BA_order_median_ratio": GPUProbeSupport.percentile(values("A", order: "BA"), 0.5) / GPUProbeSupport.percentile(values("B", order: "BA"), 0.5),
                        "nonoverlapping_ABBA_sum_ratios": blockRatios,
                        "sample_ordinals": rows.map { $0["ordinal"] as! Int }])
                }
            }
            // Same candidate instance returns from S3/TM2 to the original
            // S2/TM4 configuration. No additional timing or new input.
            for (variant, kernel) in kernels.enumerated() {
                let y = try kernel.apply(inputs[2]!, weight: matrices[0])
                try compare(y, oracles[2]![0], id: variant == 0 ? "return_S2_A" : "return_S2_B", sequence: 2, slot: 0)
            }
            guard checks.count == 42, warmups.count == 48, samples.count == 192, groups.count == 8 else {
                throw GPUError.invalid("Incomplete verification QKV probe")
            }
            report["final_memory"] = try MX.memory()
            report["complete"] = true; report["passed"] = true
            try save()
        } catch {
            report["fatal_error"] = String(describing: error)
            try save()
            throw error
        }
    }
}
