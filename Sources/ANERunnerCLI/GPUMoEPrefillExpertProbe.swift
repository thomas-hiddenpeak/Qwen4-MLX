import ANERunnerGPU
import CMLX
import Foundation

extension RunnerCLI {
    static func probeGPUMoEPrefillExpert(_ args: Arguments) throws {
        try args.validate(["--model-dir", "--manifest", "--output", "--config-output"])
        let output = URL(fileURLWithPath: try args.require("--output")).standardizedFileURL.resolvingSymlinksInPath()
        let configOutput = URL(fileURLWithPath: try args.require("--config-output")).standardizedFileURL.resolvingSymlinksInPath()
        guard output != configOutput, !FileManager.default.fileExists(atPath: output.path),
              !FileManager.default.fileExists(atPath: configOutput.path) else {
            throw CLIError.usage("Expert probe outputs must be new and distinct")
        }
        struct Mode {
            let id: String
            let variant: Int?
            let groupedDown: Bool
            func expected(_ calls: UInt64) -> (gate: [UInt64], grouped: [UInt64]) {
                var gate = [UInt64](repeating: 0, count: 4), grouped = gate
                if let variant {
                    gate[variant] = calls
                    if variant >= 2 {
                        grouped[variant - 2] = calls
                        if groupedDown { grouped[variant] = calls }
                    }
                }
                return (gate, grouped)
            }
        }
        let referenceMode = Mode(id: "reference", variant: nil, groupedDown: false)
        let modes = [Mode(id: "prior1", variant: 1, groupedDown: false),
            Mode(id: "expert32", variant: 2, groupedDown: false),
            Mode(id: "expert16", variant: 3, groupedDown: false),
            Mode(id: "expert32_down", variant: 2, groupedDown: true),
            Mode(id: "expert16_down", variant: 3, groupedDown: true)]
        try MoEGateUpProbeSupport.requireStockSelectors()
        let base = try MoEGateUpProbeSupport.baseMLX()
        let plugin = try GPUMoEPrefillGateUp(variant: 0)
        let pluginHash = try MoETilingBytes.hash(URL(fileURLWithPath: plugin.libraryPath))
        func counters() throws -> (gate: [UInt64], grouped: [UInt64]) {
            let gate = plugin.dispatchCounts(), grouped = plugin.groupedDispatchCounts()
            guard gate.count == 4, grouped.count == 4 else { throw CLIError.usage("Expert plugin requires four gate and four grouped counters") }
            return (gate, grouped)
        }
        func delta(_ before: (gate: [UInt64], grouped: [UInt64])) throws -> (gate: [UInt64], grouped: [UInt64]) {
            let after = try counters()
            return (try MoEGateUpProbeSupport.delta(after.gate, before.gate),
                    try MoEGateUpProbeSupport.delta(after.grouped, before.grouped))
        }
        func verify(_ actual: (gate: [UInt64], grouped: [UInt64]), _ mode: Mode, calls: UInt64) throws {
            let expected = mode.expected(calls)
            guard actual.gate == expected.gate, actual.grouped == expected.grouped else {
                throw CLIError.usage("Expert dispatch counts differ for \(mode.id)")
            }
        }
        _ = try counters()
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
        var cases = [[String: Any]](), rejected = Set<String>()
        var totals = [String: (reference: Double, candidate: Double, count: Int)]()
        var report: [String: Any] = [
            "schema": "qwen38-prefill-expert-probe-v1", "complete": false, "passed": false,
            "manifest": manifestURL.path, "manifest_sha256": MoETilingBytes.digest(manifestData),
            "model_directory": modelURL.path, "base_mlx": base,
            "gateup_plugin_path": plugin.libraryPath, "gateup_plugin_sha256": pluginHash,
            "executable_sha256": try MoETilingBytes.hash(URL(fileURLWithPath: CommandLine.arguments[0])),
            "modes": ([referenceMode] + modes).map { ["id": $0.id, "variant": $0.variant.map { $0 as Any } ?? NSNull(), "grouped_down": $0.groupedDown] as [String: Any] },
            "grouped_counter_order": ["plan32", "plan16", "down32", "down16"],
            "reduction_threadgroup": NSNull(), "minimum_micro_speedup_percent": 2.0,
            "warmups_per_mode": 3, "timed_samples_per_mode": 8, "timed_order": "ABBA repeated four times per candidate",
            "notes": [
                "Each candidate is independently paired with the original complete MoE chain. Nine real S416 inputs determine selection; derived S205/S240 prefixes only check correctness.",
                "Ten finite, bitwise comparisons include activation, expert outputs, complete output, and diagnostic/plain cross-checks. Consumed activation diagnostics are compared in the same sorted assignment order.",
                "Timing builds a fresh complete MoE graph and evaluates y; diagnostics, host readback, loading and warmup are excluded. Original reduction and stock MLX stay active.",
                "Relative-to-prior1 values compare separately measured reference-normalized ratios. They are indirect, not a direct paired prior1-versus-grouped benchmark.",
                "Counters establish encoded plan/gate/down dispatches, not physical reads or bandwidth. Complete/passed means the bounded search finished without fatal error; rejected_modes and candidate_correctness_passed separately describe candidates.",
                "Reference may win. The saved configuration requires full 11k producer/consumer PD validation before use."
            ]]
        func save() throws {
            report["cases"] = cases; report["rejected_modes"] = rejected.sorted()
            try emit(report, to: output.path)
        }
        try save()
        do {
            var oldCache = 0
            try MX.check(mlx_set_cache_limit(&oldCache, 128 * 1024 * 1024), "expert probe cache cap")
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
                        func forward(_ mode: Mode, diagnostics: Bool) throws -> GPUMoEOutput {
                            let result = try moe.forward(x, diagnostics: diagnostics, prefillReductionThreadgroup: nil,
                                prefillGateUpVariant: mode.variant, groupedDown: mode.groupedDown)
                            try MX.eval([result.y] + Array(result.diagnostics.values))
                            return result
                        }
                        let referenceBefore = try counters()
                        let reference = try forward(referenceMode, diagnostics: true)
                        let referencePlain = try forward(referenceMode, diagnostics: false)
                        try verify(delta(referenceBefore), referenceMode, calls: 2)
                        let referenceCrossCheck = try MoETilingBytes.compare(reference.y, referencePlain.y)
                        guard referenceCrossCheck["exact"] as? Bool == true else { throw CLIError.usage("Reference diagnostic/plain differs") }
                        item["reference_diagnostic_plain"] = referenceCrossCheck
                        for mode in modes {
                            FileHandle.standardError.write(Data("Expert probe: L\(layer) P\(fixture.offset) S\(length) \(mode.id)\n".utf8))
                            let before = try counters()
                            let candidate = try forward(mode, diagnostics: true), plain = try forward(mode, diagnostics: false)
                            let dispatchDelta = try delta(before)
                            try verify(dispatchDelta, mode, calls: 2)
                            var comparisons = [String: Any](), exact = true
                            for name in ["selected_experts", "routing_weights", "prefill_activation", "selected_expert_outputs",
                                         "routed_sum", "shared_down", "shared_gate", "output"] {
                                guard let a = reference.diagnostics[name], let b = candidate.diagnostics[name] else {
                                    throw CLIError.usage("Missing expert diagnostic: \(name)")
                                }
                                let comparison = try MoETilingBytes.compare(a, b)
                                comparisons[name] = comparison; exact = exact && (comparison["exact"] as? Bool == true)
                            }
                            for (name, a, b) in [("plain_y", referencePlain.y, plain.y),
                                                ("candidate_diagnostic_plain", candidate.y, plain.y)] {
                                let comparison = try MoETilingBytes.compare(a, b)
                                comparisons[name] = comparison; exact = exact && (comparison["exact"] as? Bool == true)
                            }
                            var trial: [String: Any] = ["mode": mode.id, "variant": mode.variant!, "grouped_down": mode.groupedDown,
                                "exact": exact, "comparisons": comparisons, "diagnostic_dispatch_delta": dispatchDelta.gate,
                                "diagnostic_grouped_dispatch_delta": dispatchDelta.grouped]
                            if !exact { rejected.insert(mode.id) }
                            if rejected.contains(mode.id) { trials.append(trial); continue }
                            for _ in 0..<3 { _ = try forward(referenceMode, diagnostics: false); _ = try forward(mode, diagnostics: false) }
                            var a = 0.0, b = 0.0, samples = [[String: Any]]()
                            let timedBefore = try counters()
                            for _ in 0..<4 {
                                for useCandidate in [false, true, true, false] {
                                    let selected = useCandidate ? mode : referenceMode
                                    let start = DispatchTime.now().uptimeNanoseconds
                                    let y = try moe.forward(x, prefillReductionThreadgroup: nil,
                                        prefillGateUpVariant: selected.variant, groupedDown: selected.groupedDown).y
                                    try y.eval()
                                    let ms = Double(DispatchTime.now().uptimeNanoseconds - start) * 1e-6
                                    guard ms.isFinite, ms > 0 else { throw CLIError.usage("Invalid expert timing") }
                                    if useCandidate { b += ms } else { a += ms }
                                    samples.append(["candidate": useCandidate, "milliseconds": ms])
                                }
                            }
                            let timedDelta = try delta(timedBefore)
                            try verify(timedDelta, mode, calls: 8)
                            trial["samples"] = samples; trial["baseline_mean_ms"] = a / 8; trial["candidate_mean_ms"] = b / 8
                            trial["speedup_percent"] = (a / b - 1) * 100
                            trial["timed_dispatch_delta"] = timedDelta.gate; trial["timed_grouped_dispatch_delta"] = timedDelta.grouped
                            if length == 416 {
                                let old = totals[mode.id] ?? (0,0,0)
                                totals[mode.id] = (old.reference + a / 8, old.candidate + b / 8, old.count + 1)
                            }
                            trials.append(trial)
                        }
                        item["trials"] = trials; cases.append(item); try save()
                    }
                }
            }
            let ranking = modes.filter { !rejected.contains($0.id) && totals[$0.id]?.count == 9 }
                .sorted {
                    let lhs = totals[$0.id]!, rhs = totals[$1.id]!
                    let a = lhs.reference / lhs.candidate, b = rhs.reference / rhs.candidate
                    return a == b ? $0.id < $1.id : a > b
                }
            let winner = ranking.first.flatMap { totals[$0.id]!.reference / totals[$0.id]!.candidate >= 1.02 ? $0 : nil }
            let config = GPUMoEPrefillConfiguration(threadgroups: [:], gateUpVariant: winner?.variant,
                groupedDown: winner?.groupedDown == true ? true : nil)
            try config.validated()
            let prior = !rejected.contains("prior1") && totals["prior1"]?.count == 9 ? totals["prior1"] : nil
            report["ranking"] = ranking.map { mode -> [String: Any] in
                let total = totals[mode.id]!, ratio = total.reference / total.candidate
                return ["mode": mode.id, "variant": mode.variant!, "grouped_down": mode.groupedDown,
                    "baseline_sum_ms": total.reference, "candidate_sum_ms": total.candidate, "speedup_percent": (ratio - 1) * 100,
                    "indirect_normalized_gain_vs_prior1_percent": prior.map { ((ratio / ($0.reference / $0.candidate) - 1) * 100) as Any } ?? NSNull()]
            }
            report["selected_mode"] = winner?.id ?? "reference"
            report["selected_variant"] = (winner?.variant).map { $0 as Any } ?? NSNull()
            report["selected_grouped_down"] = winner?.groupedDown ?? false
            report["selected_reference"] = winner == nil
            report["complete"] = cases.count == 11; report["passed"] = cases.count == 11
            report["candidate_correctness_passed"] = rejected.isEmpty && cases.count == 11
            report["config_output"] = configOutput.path; try save()
            var saved = try JSONSerialization.jsonObject(with: JSONEncoder().encode(config)) as! [String: Any]
            saved["gateup_plugin_path"] = plugin.libraryPath; saved["gateup_plugin_sha256"] = pluginHash
            saved["base_mlx_sha256"] = base["loaded_sha256"]!; saved["base_mlx_path"] = base["loaded_path"]!
            saved["expert_report"] = output.path; saved["expert_report_sha256"] = try MoETilingBytes.hash(output)
            saved["model_directory"] = modelURL.path; saved["device_target"] = "Apple M5 Max"
            saved["status"] = "micro_selected_requires_full_pd_validation"
            try emit(saved, to: configOutput.path)
        } catch {
            report["fatal_error"] = String(describing: error); report["passed"] = false
            try save(); throw error
        }
    }
}
