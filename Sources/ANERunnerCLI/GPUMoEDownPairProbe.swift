import ANERunnerCore
import ANERunnerGPU
import CMLX
import Dispatch
import Foundation

extension RunnerCLI {
    static func probeGPUMoEDownPair(_ args: Arguments) throws {
        try args.validate(["--model-dir", "--pair-library", "--output", "--warmups", "--runs",
                           "--prefill-fixture", "--decode-fixture", "--shared-elementwise"])
        let output = URL(fileURLWithPath: try args.require("--output")).standardizedFileURL
        let sharedMode = args["--shared-elementwise"] ?? "fused"
        guard !FileManager.default.fileExists(atPath: output.path),
              let warmups = Int(args["--warmups"] ?? "3"), (1...10).contains(warmups),
              let runs = Int(args["--runs"] ?? "12"), (4...40).contains(runs), runs.isMultiple(of: 2),
              ["reference", "fused"].contains(sharedMode) else {
            throw CLIError.usage("Use a new output, warmups 1...10, even runs 4...40, and shared-elementwise reference or fused")
        }
        let modes = ["baseline", "recipe_graph", "native_serial", "native_pair"]
        var cases: [[String: Any]] = [], samples: [[String: Any]] = []
        var report: [String: Any] = [
            "schema": "qwen38-moe-down-pair-probe-v1", "complete": false, "passed": false,
            "layer_index": 0, "modes": modes, "shared_elementwise": sharedMode,
            "warmups_per_row_per_mode": warmups, "timed_samples_per_row_per_mode": runs,
            "timing_scope": "Fresh complete S1 MoE graph construction and y.eval; resident input and weights; no diagnostics or host readback in timed samples",
            "timed_order": "Four modes then reverse on alternate rounds; real row order rotates each round",
            "native_counter_order": ["native_pair", "native_serial"],
            "physical_dram_bandwidth_gbps": NSNull(), "full_model_generation": false,
            "minimum_layer_speedup_ratio": 1.05,
            "notes": [
                "baseline uses unmodified GPUMoE.forward; recipe_graph uses the same two recipes without a native wrapper.",
                "native_serial inserts a Metal barrier between the original routed and shared down primitives; native_pair resolves both input hazards before launching either.",
                "Captured prefill rows are individually replayed as S1 and compared with the current S1 baseline, not their original multi-token capture outputs.",
                "Counters count successfully encoded pairs, not GPU completion, launches or bandwidth. A pair beating only the artificial serial control is insufficient.",
                "Interleaved real routes still form a hot single-layer workload; any gain requires a separate full-model decode gate before adoption."]]
        func save() throws {
            report["cases"] = cases; report["interleaved_samples"] = samples
            try emit(report, to: output.path)
        }
        try save()
        do {
            try MoEGateUpProbeSupport.requireStockSelectors()
            report["base_mlx"] = try MoEGateUpProbeSupport.baseMLX()
            let plugin = try GPUMoEDownPairProbe(libraryPath: args.require("--pair-library"))
            // Also reject a second MLX image introduced by the plugin's dependency.
            _ = try MoEGateUpProbeSupport.baseMLX()
            report["pair_library"] = plugin.libraryPath
            report["pair_library_sha256"] = try GPUProbeSupport.hash(URL(fileURLWithPath: plugin.libraryPath))
            report["pair_abi_version"] = plugin.abiVersion
            report["executable_sha256"] = try GPUProbeSupport.hash(URL(fileURLWithPath: CommandLine.arguments[0]))
            let beforeCounts = plugin.encodedPairCounts()
            let modelURL = URL(fileURLWithPath: try args.require("--model-dir"))
                .standardizedFileURL.resolvingSymlinksInPath()
            let configuration = try QwenConfiguration(modelDirectory: modelURL)
            guard configuration.hiddenSize == 2560, configuration.expertCount == 512,
                  configuration.expertsPerToken == 10, configuration.quantizationBits == 4,
                  configuration.quantizationGroupSize == 64 else {
                throw CLIError.usage("Down-pair probe requires the fixed H2560/E512/top10 affine-Q4 model")
            }
            report["model_directory"] = modelURL.path
            report["config_sha256"] = try GPUProbeSupport.hash(modelURL.appendingPathComponent("config.json"))
            report["weight_index_sha256"] = try GPUProbeSupport.hash(modelURL.appendingPathComponent("model.safetensors.index.json"))
            report["mlx_version"] = try GPUProbeSupport.mlxVersion()
            report["operating_system"] = ProcessInfo.processInfo.operatingSystemVersionString

            let fixtureRoot = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
                .deletingLastPathComponent().deletingLastPathComponent()
                .appendingPathComponent("fixtures/moe-real/converted")
            let paths = [args["--decode-fixture"] ?? fixtureRoot.appendingPathComponent("decode.json").path,
                         args["--prefill-fixture"] ?? fixtureRoot.appendingPathComponent("prefill.json").path]
            var inputs: [Tensor] = [], rowSources: [[String: Any]] = []
            for path in paths {
                let url = URL(fileURLWithPath: path).standardizedFileURL.resolvingSymlinksInPath()
                let fixture = try CoreMLBlockFixture.load(from: url)
                guard fixture.inputs.count == 1, let x = fixture.inputs["x"],
                      x.shape.count == 3, x.shape[0] == 1, (1...8).contains(x.shape[1]),
                      x.shape[2] == 2560, x.values.count == x.shape[1] * 2560,
                      x.dtype == .float16 || x.dtype == .float32,
                      x.values.allSatisfy({ Float($0).isFinite }) else {
                    throw CLIError.usage("Provide a finite floating real x[1,T,2560] fixture with T1...8")
                }
                let hash = try GPUProbeSupport.hash(url)
                for row in 0..<x.shape[1] {
                    let values = x.values[(row*2560)..<((row+1)*2560)].map(Float.init)
                    let input = try MX.array(values, shape: [1,1,2560], dtype: MLX_BFLOAT16)
                    try input.eval()
                    inputs.append(input)
                    rowSources.append(["fixture": url.path, "fixture_sha256": hash, "row": row,
                                       "input_bf16_sha256": try GPUProbeSupport.digest(MoETilingBytes.bytes(input))])
                }
            }
            guard inputs.count >= 3 else { throw CLIError.usage("Use at least three captured real rows") }
            report["row_bank"] = rowSources
            let loadStart = DispatchTime.now().uptimeNanoseconds
            let weights = try GPUWeights(modelDirectory: modelURL)
            let moe = try GPUMoE(weights: weights, layer: 0, fuseSharedElementwise: sharedMode == "fused")
            try MX.synchronize()
            report["load_milliseconds"] = GPUProbeSupport.elapsed(loadStart)
            report["loaded_source_bytes"] = weights.cachedSourceBytes
            report["loaded_memory"] = try MX.memory()
            guard weights.ledger.allSatisfy({ $0.name.hasPrefix("language_model.model.layers.0.mlp.") }) else {
                throw GPUError.invalid("Down-pair probe loaded tensors outside layer-0 MoE")
            }
            func forward(_ row: Int, _ mode: String, diagnostics: Bool) throws -> GPUMoEOutput {
                if mode == "baseline" { return try moe.forward(inputs[row], diagnostics: diagnostics) }
                return try moe.probeDownPair(inputs[row], executor: mode == "recipe_graph" ? nil : plugin,
                    serialControl: mode == "native_serial", diagnostics: diagnostics)
            }
            for row in inputs.indices {
                let reference = try forward(row, "baseline", diagnostics: true)
                try MX.eval(Array(reference.diagnostics.values) + [reference.y])
                let referencePlain = try forward(row, "baseline", diagnostics: false)
                try referencePlain.y.eval()
                var checks: [String: Any] = [:], exact = true
                let baselineCheck = try MoETilingBytes.compare(reference.y, referencePlain.y)
                checks["baseline_diagnostic_vs_plain"] = baselineCheck
                exact = baselineCheck["exact"] as? Bool == true
                for mode in modes.dropFirst() {
                    let candidate = try forward(row, mode, diagnostics: true)
                    try MX.eval(Array(candidate.diagnostics.values) + [candidate.y])
                    var comparison: [String: Any] = [:]
                    for name in candidate.diagnostics.keys.sorted() {
                        guard let a = reference.diagnostics[name], let b = candidate.diagnostics[name] else {
                            throw GPUError.invalid("Missing down-pair comparison tensor \(name)")
                        }
                        let value = try MoETilingBytes.compare(a, b)
                        comparison[name] = value; exact = exact && (value["exact"] as? Bool == true)
                    }
                    let plain = try forward(row, mode, diagnostics: false)
                    try plain.y.eval()
                    for (name, a, b) in [("plain_y", referencePlain.y, plain.y),
                                         ("diagnostic_vs_plain", candidate.y, plain.y)] {
                        let value = try MoETilingBytes.compare(a, b)
                        comparison[name] = value; exact = exact && (value["exact"] as? Bool == true)
                    }
                    checks[mode] = comparison
                }
                let ids = try reference.diagnostics["selected_experts"]!.ints()
                guard ids.count == 10, Set(ids).count == 10, ids.allSatisfy({ (0..<512).contains($0) }) else {
                    throw GPUError.invalid("Invalid real routing IDs")
                }
                cases.append(["row_index": row, "source": rowSources[row], "selected_experts": ids,
                              "bitwise_passed": exact, "comparisons": checks])
                try save()
                guard exact else { throw GPUError.invalid("Down-pair bitwise/finite gate failed at row \(row); timing stopped") }
            }
            for round in 0..<warmups {
                let order = round.isMultiple(of: 2) ? modes : Array(modes.reversed())
                for row in inputs.indices {
                    for mode in order { try forward(row, mode, diagnostics: false).y.eval() }
                }
            }
            for round in 0..<runs {
                let order = round.isMultiple(of: 2) ? modes : Array(modes.reversed())
                for offset in inputs.indices {
                    let row = (offset + round) % inputs.count
                    for mode in order {
                        let start = DispatchTime.now().uptimeNanoseconds
                        let y = try forward(row, mode, diagnostics: false).y
                        try y.eval()
                        let milliseconds = GPUProbeSupport.elapsed(start)
                        guard milliseconds.isFinite, milliseconds > 0 else { throw GPUError.invalid("Invalid timing sample") }
                        samples.append(["index": samples.count, "round": round, "row_index": row,
                                        "mode": mode, "milliseconds": milliseconds])
                    }
                }
            }
            let delta = try MoEGateUpProbeSupport.delta(plugin.encodedPairCounts(), beforeCounts)
            let expected = UInt64(inputs.count * (2 + warmups + runs))
            report["encoded_pair_delta"] = delta; report["expected_encoded_pair_delta"] = [expected, expected]
            guard delta == [expected, expected] else { throw GPUError.invalid("Down-pair encoded count differs from evaluated graphs") }
            var medians: [String: Double] = [:]
            for mode in modes {
                let values = samples.filter { $0["mode"] as? String == mode }.compactMap { $0["milliseconds"] as? Double }
                guard values.count == inputs.count * runs else { throw GPUError.invalid("Missing timing samples") }
                medians[mode] = GPUProbeSupport.percentile(values, 0.5)
            }
            for row in cases.indices {
                var rowMedians: [String: Double] = [:]
                for mode in modes {
                    let values = samples.filter { ($0["row_index"] as? Int) == row && ($0["mode"] as? String) == mode }
                        .compactMap { $0["milliseconds"] as? Double }
                    rowMedians[mode] = GPUProbeSupport.percentile(values, 0.5)
                }
                cases[row]["median_milliseconds"] = rowMedians
            }
            let ratio = medians["baseline"]! / medians["native_pair"]!
            report["median_milliseconds"] = medians
            report["baseline_over_native_pair_ratio"] = ratio
            report["native_serial_over_native_pair_ratio"] = medians["native_serial"]! / medians["native_pair"]!
            report["layer_performance_gate_passed"] = ratio >= 1.05
            report["complete"] = true; report["passed"] = true
            report["passed_scope"] = "All finite/bitwise and encoded-count gates passed; timing finished. Performance gate is a separate field."
            report["final_memory"] = try MX.memory()
            try save()
        } catch {
            report["fatal_error"] = String(describing: error); report["passed"] = false
            try save()
            throw error
        }
    }
}
