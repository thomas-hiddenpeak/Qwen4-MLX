import ANERunnerGPU
import CMLX
import Darwin
import Foundation

extension RunnerCLI {
    static func probeGPUMoEPrefillGateUp(_ args: Arguments) throws {
        try args.validate(["--model-dir", "--manifest", "--output", "--config-output"])
        let output = URL(fileURLWithPath: try args.require("--output")).standardizedFileURL.resolvingSymlinksInPath()
        let configOutput = URL(fileURLWithPath: try args.require("--config-output")).standardizedFileURL.resolvingSymlinksInPath()
        guard output != configOutput, !FileManager.default.fileExists(atPath: output.path),
              !FileManager.default.fileExists(atPath: configOutput.path) else {
            throw CLIError.usage("Gate/up probe outputs must be new and distinct")
        }
        try MoEGateUpProbeSupport.requireStockSelectors()
        let base = try MoEGateUpProbeSupport.baseMLX()
        let plugin = try GPUMoEPrefillGateUp(variant: 0)
        let zeroCounts = [UInt64](repeating: 0, count: plugin.dispatchCounts().count)
        let pluginHash = try MoETilingBytes.hash(URL(fileURLWithPath: plugin.libraryPath))
        let modelURL = URL(fileURLWithPath: try args.require("--model-dir")).standardizedFileURL.resolvingSymlinksInPath()
        let manifestURL = URL(fileURLWithPath: try args.require("--manifest")).standardizedFileURL
        let manifestData = try Data(contentsOf: manifestURL)
        struct Manifest: Decodable {
            struct Fixture: Decodable { let layer, offset: Int; let file, sha256: String }
            struct Provenance: Decodable { let config_sha256, weight_index_sha256: String }
            let passed, committed: Bool
            let model_directory: String
            let provenance: Provenance
            let fixtures: [Fixture]
        }
        let manifest = try JSONDecoder().decode(Manifest.self, from: manifestData)
        guard manifest.passed, manifest.committed, manifest.fixtures.count == 9,
              manifest.model_directory == modelURL.path,
              manifest.provenance.config_sha256 == (try MoETilingBytes.hash(modelURL.appendingPathComponent("config.json"))),
              manifest.provenance.weight_index_sha256 == (try MoETilingBytes.hash(modelURL.appendingPathComponent("model.safetensors.index.json"))) else {
            throw CLIError.usage("Require the committed golden-checked capture for this model")
        }
        var cases = [[String: Any]](), rejected = Set<Int>()
        var totals = [Int: (reference: Double, candidate: Double, count: Int)]()
        var report: [String: Any] = [
            "schema": "qwen38-prefill-gateup-probe-v1", "complete": false, "passed": false,
            "manifest": manifestURL.path, "manifest_sha256": MoETilingBytes.digest(manifestData),
            "model_directory": modelURL.path, "base_mlx": base,
            "gateup_plugin_path": plugin.libraryPath, "gateup_plugin_sha256": pluginHash,
            "executable_sha256": try MoETilingBytes.hash(URL(fileURLWithPath: CommandLine.arguments[0])),
            "reference_variant": NSNull(), "candidate_variants": [0, 1], "reduction_threadgroup": NSNull(),
            "minimum_micro_speedup_percent": 2.0, "warmups_per_variant": 3, "timed_samples_per_variant": 8,
            "timed_order": "ABBA repeated four times",
            "notes": [
                "Nil selects the original gate/up/SwiGLU chain; 0 and 1 are distinct fused candidates.",
                "Original reduction and stock base MLX remain active; old native tiling/autotune selectors must be unset or zero.",
                "Nine real S416 inputs determine the paired full-MoE score. Derived S205/S240 prefixes only check correctness.",
                "Timing includes a fresh complete MoE graph and y.eval; diagnostics, host readback, loading and warmup are excluded.",
                "Candidate activation, expert and complete outputs must be finite and bitwise equal, including diagnostic/plain cross-checks.",
                "Plugin counters count encoded dispatches after output evaluation, not physical weight reads or bandwidth.",
                "Reference may win. Saved configuration requires a separate full 11k producer/consumer PD validation."
            ]]
        func save() throws {
            report["cases"] = cases; report["rejected_variants"] = rejected.sorted()
            try emit(report, to: output.path)
        }
        try save()
        do {
            var oldCache = 0
            try MX.check(mlx_set_cache_limit(&oldCache, 128 * 1024 * 1024), "gate/up probe cache cap")
            defer { var ignored = 0; _ = mlx_set_cache_limit(&ignored, oldCache) }
            for layer in [0, 23, 47] {
                let weights = try GPUWeights(modelDirectory: modelURL)
                let moe = try GPUMoE(weights: weights, layer: layer, prefillAccumulation: .reference)
                let fixtures = manifest.fixtures.filter { $0.layer == layer }.sorted { $0.offset < $1.offset }
                guard fixtures.map(\.offset) == [0, 4_992, 9_984] else { throw CLIError.usage("Unexpected capture positions") }
                for fixture in fixtures {
                    let url = URL(fileURLWithPath: fixture.file)
                    guard try MoETilingBytes.hash(url) == fixture.sha256 else { throw CLIError.usage("Fixture checksum changed") }
                    let original = try MoETilingBytes.loadX(url)
                    guard original.shape == [1, 416, 2_560], original.dtype == MLX_BFLOAT16 else {
                        throw CLIError.usage("Expected native BF16 x[1,416,2560]")
                    }
                    for length in (layer == 0 && fixture.offset == 0 ? [416, 205, 240] : [416]) {
                        let x = length == 416 ? original : try MX.slice(original, starts: [0,0,0], ends: [1,length,2560])
                        try x.eval()
                        var item: [String: Any] = ["layer": layer, "offset": fixture.offset, "tokens": length,
                            "derived": length != 416, "fixture": fixture.file]
                        var trials = [[String: Any]]()
                        func forward(_ variant: Int?, diagnostics: Bool) throws -> GPUMoEOutput {
                            let result = try moe.forward(x, diagnostics: diagnostics,
                                prefillReductionThreadgroup: nil, prefillGateUpVariant: variant)
                            try MX.eval([result.y] + Array(result.diagnostics.values))
                            return result
                        }
                        let referenceBefore = plugin.dispatchCounts()
                        let reference = try forward(nil, diagnostics: true), referencePlain = try forward(nil, diagnostics: false)
                        guard try MoEGateUpProbeSupport.delta(plugin.dispatchCounts(), referenceBefore) == zeroCounts,
                              try MoETilingBytes.compare(reference.y, referencePlain.y)["exact"] as? Bool == true else {
                            throw CLIError.usage("Reference path used plugin or diagnostic/plain output differs")
                        }
                        for variant in [0, 1] where !rejected.contains(variant) {
                            FileHandle.standardError.write(Data("Gate/up probe: L\(layer) P\(fixture.offset) S\(length) variant\(variant)\n".utf8))
                            let before = plugin.dispatchCounts()
                            let candidate = try forward(variant, diagnostics: true), plain = try forward(variant, diagnostics: false)
                            let dispatchDelta = try MoEGateUpProbeSupport.delta(plugin.dispatchCounts(), before)
                            var expected = zeroCounts; expected[variant] = 2
                            guard dispatchDelta == expected else { throw CLIError.usage("Diagnostic/plain candidate did not encode exactly two selected gate/up dispatches") }
                            var comparisons = [String: Any](), exact = true
                            for name in ["selected_experts", "routing_weights", "prefill_activation", "selected_expert_outputs",
                                         "routed_sum", "shared_down", "shared_gate", "output"] {
                                guard let a = reference.diagnostics[name], let b = candidate.diagnostics[name] else {
                                    throw CLIError.usage("Missing gate/up diagnostic: \(name)")
                                }
                                let comparison = try MoETilingBytes.compare(a, b)
                                comparisons[name] = comparison; exact = exact && (comparison["exact"] as? Bool == true)
                            }
                            for (name, a, b) in [("plain_y", referencePlain.y, plain.y),
                                                ("candidate_diagnostic_plain", candidate.y, plain.y)] {
                                let comparison = try MoETilingBytes.compare(a, b)
                                comparisons[name] = comparison; exact = exact && (comparison["exact"] as? Bool == true)
                            }
                            var trial: [String: Any] = ["variant": variant, "exact": exact, "comparisons": comparisons,
                                "diagnostic_dispatch_delta": dispatchDelta]
                            if !exact { rejected.insert(variant); trials.append(trial); continue }
                            for _ in 0..<3 { _ = try forward(nil, diagnostics: false); _ = try forward(variant, diagnostics: false) }
                            var a = 0.0, b = 0.0, samples = [[String: Any]]()
                            let timedBefore = plugin.dispatchCounts()
                            for _ in 0..<4 {
                                for useCandidate in [false, true, true, false] {
                                    let selected: Int? = useCandidate ? variant : nil
                                    let start = DispatchTime.now().uptimeNanoseconds
                                    let y = try moe.forward(x, prefillReductionThreadgroup: nil, prefillGateUpVariant: selected).y
                                    try y.eval()
                                    let ms = Double(DispatchTime.now().uptimeNanoseconds - start) * 1e-6
                                    guard ms.isFinite, ms > 0 else { throw CLIError.usage("Invalid gate/up timing") }
                                    if useCandidate { b += ms } else { a += ms }
                                    samples.append(["candidate": useCandidate, "milliseconds": ms])
                                }
                            }
                            let timedDelta = try MoEGateUpProbeSupport.delta(plugin.dispatchCounts(), timedBefore)
                            expected[variant] = 8
                            guard timedDelta == expected else { throw CLIError.usage("Timed gate/up dispatch count mismatch") }
                            trial["samples"] = samples; trial["baseline_mean_ms"] = a / 8; trial["candidate_mean_ms"] = b / 8
                            trial["speedup_percent"] = (a / b - 1) * 100; trial["timed_dispatch_delta"] = timedDelta
                            if length == 416 {
                                let old = totals[variant] ?? (0,0,0)
                                totals[variant] = (old.reference + a / 8, old.candidate + b / 8, old.count + 1)
                            }
                            trials.append(trial)
                        }
                        item["trials"] = trials; cases.append(item); try save()
                    }
                }
            }
            let ranking = [0,1].filter { !rejected.contains($0) && totals[$0]?.count == 9 }
                .sorted {
                    let lhs = totals[$0]!, rhs = totals[$1]!
                    let a = lhs.reference / lhs.candidate, b = rhs.reference / rhs.candidate
                    return a == b ? $0 < $1 : a > b
                }
            let winner = ranking.first.flatMap { totals[$0]!.reference / totals[$0]!.candidate >= 1.02 ? $0 : nil }
            let config = GPUMoEPrefillConfiguration(threadgroups: [:], gateUpVariant: winner)
            try config.validated()
            report["ranking"] = ranking.map { variant in ["variant": variant, "baseline_sum_ms": totals[variant]!.reference,
                "candidate_sum_ms": totals[variant]!.candidate, "speedup_percent": (totals[variant]!.reference / totals[variant]!.candidate - 1) * 100] as [String: Any] }
            report["selected_variant"] = winner.map { $0 as Any } ?? NSNull()
            report["selected_reference"] = winner == nil
            report["complete"] = cases.count == 11; report["passed"] = cases.count == 11
            report["config_output"] = configOutput.path; try save()
            var saved = try JSONSerialization.jsonObject(with: JSONEncoder().encode(config)) as! [String: Any]
            saved["gateup_plugin_path"] = plugin.libraryPath; saved["gateup_plugin_sha256"] = pluginHash
            saved["base_mlx_sha256"] = base["loaded_sha256"]!
            saved["base_mlx_path"] = base["loaded_path"]!
            saved["gateup_report"] = output.path; saved["gateup_report_sha256"] = try MoETilingBytes.hash(output)
            saved["model_directory"] = modelURL.path; saved["device_target"] = "Apple M5 Max"
            saved["status"] = "micro_selected_requires_full_pd_validation"
            try emit(saved, to: configOutput.path)
        } catch {
            report["fatal_error"] = String(describing: error); report["passed"] = false
            try save(); throw error
        }
    }
}

/// Small CPU-only identity/count checks shared by the micro and full PD probes.
enum MoEGateUpProbeSupport {
    static func requireStockSelectors() throws {
        for name in ["ANERUNNER_MOE_QMM_CONFIG", "ANERUNNER_MOE_QMM_BM"] {
            guard let value = ProcessInfo.processInfo.environment[name] else { continue }
            guard value == "0" else { throw CLIError.usage("Gate/up isolation requires \(name) unset or 0") }
        }
    }
    static func baseMLX() throws -> [String: String] {
        let package = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let expected = package.appendingPathComponent("../qwen38-ssd/runtime/mlx-serve/lib/mlx/lib/libmlx.dylib")
            .standardizedFileURL.resolvingSymlinksInPath()
        let paths = (0..<_dyld_image_count()).compactMap { index -> String? in
            guard let name = _dyld_get_image_name(index) else { return nil }
            let url = URL(fileURLWithPath: String(cString: name)).standardizedFileURL.resolvingSymlinksInPath()
            return url.lastPathComponent == "libmlx.dylib" ? url.path : nil
        }
        guard Set(paths).count == 1, let path = paths.first else { throw CLIError.usage("Expected one loaded libmlx.dylib") }
        let loadedHash = try MoETilingBytes.hash(URL(fileURLWithPath: path))
        let stockHash = path == expected.path ? loadedHash : try MoETilingBytes.hash(expected)
        guard loadedHash == stockHash else { throw CLIError.usage("Gate/up probe requires Package's stock pinned MLX, not an earlier autotune overlay") }
        return ["loaded_path": path, "loaded_sha256": loadedHash, "package_default_path": expected.path, "package_default_sha256": stockHash]
    }
    static func delta(_ after: [UInt64], _ before: [UInt64]) throws -> [UInt64] {
        guard !after.isEmpty, after.count == before.count, zip(after,before).allSatisfy({ $0.0 >= $0.1 }) else {
            throw CLIError.usage("Invalid gate/up dispatch counters")
        }
        return zip(after,before).map { $0.0 - $0.1 }
    }
}
