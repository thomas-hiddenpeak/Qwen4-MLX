import ANERunnerGPU
import CMLX
import CryptoKit
import Foundation

extension RunnerCLI {
    /// Real BF16 projection weights with deterministic synthetic performance
    /// inputs. This is an isolated GEMV probe, not a full-model benchmark.
    static func probeGPUMatvec(_ args: Arguments) throws {
        try args.validate(["--model-dir", "--gpu-command-timing-output", "--output", "--repeats", "--gdn-gemv-order", "--allow-rounding"])
        let rounding = args["--allow-rounding"] ?? "false"
        guard rounding == "true" || rounding == "false" else {
            throw CLIError.usage("--allow-rounding requires true or false")
        }
        let allowRounding = rounding == "true"
        let relativeL2Limit = 0.01
        let selectedGate = allowRounding ? "finite_and_relative_l2_at_most_0.01" : "finite_and_bitwise_equal"
        let modes = try (args["--gdn-gemv-order"] ?? "reference")
            .split(separator: ",", omittingEmptySubsequences: false).map { name in
                guard let mode = GDNGEMVMode(rawValue: String(name)) else {
                    throw CLIError.usage("Invalid --gdn-gemv-order mode: \(name)")
                }
                return mode
            }
        guard modes.contains(where: { $0.rawValue == "reference" }),
              Set(modes.map(\.rawValue)).count == modes.count else {
            throw CLIError.usage("--gdn-gemv-order requires reference and distinct modes")
        }
        defer { try? GDNGEMVMode.reference.apply() }
        guard let repeats = Int(args["--repeats"] ?? "16"), (4...64).contains(repeats) else {
            throw CLIError.usage("probe-gpu-matvec requires --repeats in 4...64")
        }
        let directory = URL(fileURLWithPath: try args.require("--model-dir"), isDirectory: true)
            .standardizedFileURL.resolvingSymlinksInPath()
        let destination = URL(fileURLWithPath: try args.require("--output"))
            .standardizedFileURL.resolvingSymlinksInPath()
        let nativeDestination = args["--gpu-command-timing-output"].map {
            URL(fileURLWithPath: $0).standardizedFileURL.resolvingSymlinksInPath()
        }
        guard destination != nativeDestination,
              !FileManager.default.fileExists(atPath: destination.path) else {
            throw CLIError.usage("Matvec JSON must be a new file distinct from native timing output")
        }
        let timing = try nativeDestination.map { try GPUCommandTimingSession(path: $0.path) }
        defer { try? timing?.finish() }
        try timing?.start()
        let loadStart = GPUCommandTimingSession.now()
        let weights = try GPUWeights(modelDirectory: directory)
        let layers = [0, 4, 8, 12]
        let definitions: [(name: String, suffix: String, input: Int, output: Int)] = [
            ("gdn_qkv", "in_proj_qkv.weight", 2560, 10240),
            ("gdn_z", "in_proj_z.weight", 2560, 6144),
            ("gdn_out", "out_proj.weight", 6144, 2560),
            ("lm_head", "", 2560, 248320),
        ]
        var cases: [GPUMatvecCase] = []
        for (index, definition) in definitions.enumerated() {
            let names = definition.name == "lm_head" ? ["language_model.lm_head.weight"] :
                layers.map { "language_model.model.layers.\($0).linear_attn.\(definition.suffix)" }
            var matrices: [Tensor] = []
            var descriptions: [[String: Any]] = []
            var sourceBytes: UInt64 = 0
            for (slot, name) in names.enumerated() {
                let matrix = try weights.tensor(name)
                guard matrix.dtype == MLX_BFLOAT16, matrix.shape == [definition.output, definition.input] else {
                    throw CLIError.usage("Unexpected BF16 projection shape: \(name)")
                }
                let metadata = try weights.metadata(name)
                sourceBytes += metadata.byteCount
                // Keep the original row-major payload and reduction layout.
                // A transpose view is not an additional matrix allocation.
                matrices.append(try MX.transpose(matrix, [1, 0]))
                descriptions.append([
                    "slot": slot, "name": name, "shard": metadata.shard,
                    "source_shape": metadata.shape, "dtype": "BF16",
                    "source_bytes": metadata.byteCount,
                ])
            }
            let seed = UInt64(0x514D_4154_5645_4300) + UInt64(index)
            let inputData = GPUMatvecCase.syntheticBF16(count: definition.input, seed: seed)
            let input = try MX.array(data: inputData, shape: [1, 1, definition.input], dtype: MLX_BFLOAT16)
            try MX.eval(matrices + [input])
            cases.append(GPUMatvecCase(name: definition.name, input: input, matrices: matrices,
                outputWidth: definition.output, sourceBytes: sourceBytes, descriptions: descriptions,
                seed: seed, inputSHA256: GPUMatvecCase.sha256(inputData)))
        }
        let loadEnd = GPUCommandTimingSession.now()
        let sourceTotal = weights.ledger.reduce(UInt64(0)) { $0 + $1.sourceBytes }
        guard sourceTotal == 1_732_771_840, weights.ledger.count == 13 else {
            throw CLIError.usage("Unexpected matvec load set or source byte count")
        }
        let inputBytes = cases.reduce(0) { $0 + $1.input.nbytes }
        let loadedMemory = try MX.memory()
        var reports: [[String: Any]] = []
        var passed = true
        for test in cases {
            var warmIterations: [[String: Any]] = []
            var measuredIterations: [[String: Any]] = []
            var warmLast = [Tensor?](repeating: nil, count: test.matrices.count)
            var measuredLast = Array(repeating: [Tensor?](repeating: nil, count: test.matrices.count), count: modes.count)
            var measuredLastIndex = [Int](repeating: -1, count: test.matrices.count)

            func iteration(_ index: Int, modeIndex: Int, warmup: Bool) throws -> (Tensor, [String: Any]) {
                let slot = index % test.matrices.count
                let matrix = test.matrices[slot]
                let mode = modes[modeIndex]
                try mode.apply() // Selection setup is outside the timed NEW graph.
                let dispatchBefore = try mode.prefetchDispatchCount()
                let phase = test.name + "." + mode.rawValue + (warmup ? ".warmup" : "")
                let start = GPUCommandTimingSession.now()
                // Always build a NEW matmul graph; evaluating a prior result
                // would benchmark an already materialized no-op.
                let y = try MX.matmul(test.input, matrix)
                let forwardEnd = GPUCommandTimingSession.now()
                try MX.eval([y])
                let evaluationEnd = GPUCommandTimingSession.now()
                let dispatchAfter = try mode.prefetchDispatchCount()
                if let before = dispatchBefore, let after = dispatchAfter {
                    let expected: UInt64 = test.name == "gdn_qkv" ? 1 : 0
                    guard after >= before && after - before == expected else {
                        throw CLIError.usage("Unexpected GDN prefetch dispatch count for \(test.name)/\(mode.rawValue)")
                    }
                }
                timing?.step(phase: phase, repetition: modeIndex, index: index, inputTokens: 0,
                    start: start, forwardEnd: forwardEnd, evaluationEnd: evaluationEnd)
                var record: [String: Any] = [
                    "phase": phase, "index": index, "weight_slot": slot, "gdn_gemv_mode": mode.rawValue,
                    "start_ns": start, "forward_end_ns": forwardEnd,
                    "evaluation_end_ns": evaluationEnd,
                    "cpu_wall_seconds": Double(evaluationEnd - start) * 1e-9,
                    "logical_weight_bytes": test.descriptions[slot]["source_bytes"]!,
                ]
                if let before = dispatchBefore, let after = dispatchAfter {
                    record["prefetch_dispatch_count"] = after - before
                }
                return (y, record)
            }

            // Two rotations of four distinct GDN weights; the head has one
            // distinct matrix and is intentionally warmed eight times too.
            for index in 0..<8 {
                for offset in modes.indices {
                    let modeIndex = (offset + index) % modes.count
                    let (y, record) = try iteration(index, modeIndex: modeIndex, warmup: true)
                    if modes[modeIndex].rawValue == "reference" { warmLast[index % test.matrices.count] = y }
                    warmIterations.append(record)
                }
            }
            // Readbacks are outside all iteration marker windows. Store the
            // final warm output for EVERY slot, including non-multiple-of-four
            // repeat counts, so every measured slot can be compared exactly.
            var warmValues: [[Float]] = []
            for tensor in warmLast {
                guard let tensor, tensor.dtype == MLX_BFLOAT16,
                      tensor.shape == [1, 1, test.outputWidth] else {
                    throw CLIError.usage("Missing or malformed warmed matvec output")
                }
                warmValues.append(try tensor.floats())
            }
            try MX.synchronize()
            for index in 0..<repeats {
                for offset in modes.indices {
                    let modeIndex = (offset + index) % modes.count
                    let (y, record) = try iteration(index, modeIndex: modeIndex, warmup: false)
                    let slot = index % test.matrices.count
                    measuredLast[modeIndex][slot] = y
                    measuredLastIndex[slot] = index
                    measuredIterations.append(record)
                }
            }
            var comparisons: [[String: Any]] = []
            for modeIndex in modes.indices {
                for slot in test.matrices.indices {
                    guard let output = measuredLast[modeIndex][slot], output.dtype == MLX_BFLOAT16,
                          output.shape == [1, 1, test.outputWidth] else {
                        throw CLIError.usage("Missing or malformed measured matvec output")
                    }
                    let actual = try output.floats(), expected = warmValues[slot]
                    let finite = actual.allSatisfy(\.isFinite) && expected.allSatisfy(\.isFinite)
                    let exact = actual.count == expected.count && zip(actual, expected).allSatisfy {
                        $0.0.bitPattern == $0.1.bitPattern
                    }
                    var squaredError = 0.0, squaredReference = 0.0, maxAbsolute = 0.0
                    let metricsAvailable = finite && actual.count == expected.count
                    if metricsAvailable {
                        for (value, reference) in zip(actual, expected) {
                            let difference = Double(value) - Double(reference)
                            squaredError += difference * difference
                            squaredReference += Double(reference) * Double(reference)
                            maxAbsolute = max(maxAbsolute, abs(difference))
                        }
                    }
                    // A zero reference with nonzero error has undefined relative
                    // error and fails the tolerance gate; never serialize infinity.
                    let relativeL2 = !metricsAvailable ? Double.infinity :
                        squaredReference > 0 ? sqrt(squaredError / squaredReference) :
                        squaredError == 0 ? 0 : Double.infinity
                    let selectedGatePassed = metricsAvailable &&
                        (allowRounding ? relativeL2 <= relativeL2Limit : exact)
                    passed = passed && selectedGatePassed
                    comparisons.append([
                        "gdn_gemv_mode": modes[modeIndex].rawValue, "reference_mode": "reference",
                        "weight_slot": slot, "measured_iteration": measuredLastIndex[slot],
                        "output_values": actual.count, "finite": finite, "bitwise_equal": exact,
                        "relative_l2": relativeL2.isFinite ? relativeL2 : NSNull(),
                        "max_abs": metricsAvailable ? maxAbsolute : NSNull(),
                        "selected_numerical_gate": selectedGate,
                        "selected_numerical_gate_passed": selectedGatePassed,
                        "warm_float32_bits_sha256": GPUMatvecCase.outputSHA256(expected),
                        "measured_float32_bits_sha256": GPUMatvecCase.outputSHA256(actual),
                    ])
                }
            }
            func mean(_ name: String) -> Double {
                let rows = measuredIterations.filter { $0["gdn_gemv_mode"] as? String == name }
                return rows.reduce(0.0) { $0 + ($1["cpu_wall_seconds"] as! Double) } / Double(rows.count)
            }
            let referenceMean = mean("reference")
            let referenceValid = comparisons.filter { $0["gdn_gemv_mode"] as? String == "reference" }
                .allSatisfy { $0["selected_numerical_gate_passed"] as? Bool == true }
            let summaries: [[String: Any]] = modes.map { mode in
                let average = mean(mode.rawValue)
                let modeComparisons = comparisons.filter { $0["gdn_gemv_mode"] as? String == mode.rawValue }
                let exact = modeComparisons
                    .allSatisfy { $0["finite"] as? Bool == true && $0["bitwise_equal"] as? Bool == true }
                let valid = referenceValid && modeComparisons
                    .allSatisfy { $0["selected_numerical_gate_passed"] as? Bool == true }
                return ["gdn_gemv_mode": mode.rawValue, "measured_calls": repeats,
                    "mean_cpu_wall_seconds": average, "finite_and_bitwise_equal_to_reference": exact,
                    "selected_numerical_gate": selectedGate, "selected_numerical_gate_passed": valid,
                    "latency_ratio_to_reference": valid ? average / referenceMean : NSNull(),
                    "speed_ratio_to_reference": valid ? referenceMean / average : NSNull()]
            }
            reports.append([
                "name": test.name, "dtype": "BF16", "input_shape": test.input.shape,
                "output_shape": [1, 1, test.outputWidth],
                "unique_weight_slots": test.matrices.count, "weight_slots": test.descriptions,
                "source_weight_bytes": test.sourceBytes,
                "input_bytes": test.input.nbytes,
                "working_set_bytes": test.sourceBytes + UInt64(test.input.nbytes),
                "input_seed_hex": String(test.seed, radix: 16), "input_bf16_sha256": test.inputSHA256,
                "warmup_iterations": warmIterations, "measured_iterations": measuredIterations,
                "comparisons": comparisons,
                "mode_summaries": summaries,
            ])
        }
        try timing?.finish()
        let report: [String: Any] = [
            "schema": "qwen38-bf16-matvec-probe-v1", "passed": passed,
            "allow_rounding": allowRounding, "selected_numerical_gate": selectedGate,
            "relative_l2_limit": allowRounding ? relativeL2Limit : NSNull(),
            "model_directory": directory.path, "repeats": repeats,
            "gdn_gemv_order": modes.map(\.rawValue), "warmup_calls_per_mode_per_case": 8,
            "warmup_calls_per_case": 8 * modes.count,
            "measured_calls_per_mode_per_case": repeats,
            "interleave": "For each matrix slot, run every mode on the identical input; rotate starting mode each iteration.",
            "source_weight_bytes": sourceTotal, "input_bytes": inputBytes,
            "working_set_bytes": sourceTotal + UInt64(inputBytes),
            "working_set_definition": "Unique source weight and BF16 input payloads across the cases; excludes outputs, allocator cache and runtime overhead. All cases remain resident. Transposed views share the original buffers.",
            "load_start_ns": loadStart, "load_end_ns": loadEnd,
            "loaded_memory": loadedMemory,
            "input_kind": "Deterministic synthetic nonzero finite BF16 inputs, not captured model activations",
            "input_generator": "UInt64 LCG; multiplier 6364136223846793005, increment 1442695040888963407; sign=bit63, BF16 exponent=119+bits32..33, mantissa=bits0..6",
            "cpu_timing_scope": "New matmul graph construction plus MX.eval wait; output readback and comparisons occur outside each iteration window. This is wall latency, not device-only execution time.",
            "gpu_command_timing": timing?.report ?? ["enabled": false],
            "physical_dram_bytes": NSNull(), "physical_dram_bandwidth_gbps": NSNull(),
            "mtp_enabled": false, "ane_used": false,
            "provenance": ["process_id": ProcessInfo.processInfo.processIdentifier,
                           "operating_system": ProcessInfo.processInfo.operatingSystemVersionString],
            "notes": ["Every candidate is compared with the warmed original reference GEMV on identical real weights and deterministic synthetic BF16 inputs; this is not a real-activation or full-model gate.",
                      "Finite and bitwise gates cover every mode and weight slot's final measured output against the reference warm output.",
                      "Mode selection is outside wall timing; each measured call constructs a NEW graph and waits for evaluation. Ratios are descriptive and suppressed for candidates failing the local numerical gate.",
                      "When allow_rounding=true, finite relative L2 <= 0.01 permits this microbenchmark screen only. BF16 weights/inputs are unchanged; different accumulation order may change output bits or generated tokens. Bitwise equality remains separately reported.",
                      "Native command-buffer intervals require offline association with the step markers; they do not measure physical DRAM traffic.",
                      "Rotating four GDN matrices changes cache residency; head repeatedly uses one larger matrix. Neither case is automatically equivalent to full-model decode."],
            "cases": reports,
        ]
        try emit(report, to: destination.path)
        if !passed { throw CLIError.usage("Matvec selected numerical gate failed (\(selectedGate)); inspect saved report") }
    }
}

struct GPUMatvecCase {
    let name: String
    let input: Tensor
    let matrices: [Tensor]
    let outputWidth: Int
    let sourceBytes: UInt64
    let descriptions: [[String: Any]]
    let seed: UInt64
    let inputSHA256: String

    static func syntheticBF16(count: Int, seed: UInt64) -> Data {
        var state = seed
        var words: [UInt16] = []
        words.reserveCapacity(count)
        for _ in 0..<count {
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            let sign = UInt16((state >> 63) << 15)
            let exponent = UInt16(119 + ((state >> 32) & 3)) << 7
            words.append((sign | exponent | UInt16(state & 127)).littleEndian)
        }
        return words.withUnsafeBytes { Data($0) }
    }

    static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    static func outputSHA256(_ values: [Float]) -> String {
        let bits = values.map { $0.bitPattern.littleEndian }
        return bits.withUnsafeBytes { sha256(Data($0)) }
    }
}
