import ANERunnerGPU
import CMLX
import CryptoKit
import Darwin
import Foundation

/// Isolated experimental library only; never modifies the installed runtime.
final class MoEPrefillTilingRuntime {
    let libraryPath: String
    private let handle: UnsafeMutableRawPointer
    private let counter: @convention(c) (Int32) -> UInt64
    init() throws {
        guard let h = dlopen(nil, RTLD_NOW) else { throw CLIError.usage("Cannot inspect MLX runtime") }
        guard let version = dlsym(h, "anemlx_moe_qmm_tile_version"),
              let count = dlsym(h, "anemlx_moe_qmm_dispatch_count") else {
            dlclose(h)
            throw CLIError.usage("Load the isolated MoE tiling library with DYLD_LIBRARY_PATH")
        }
        let getVersion = unsafeBitCast(version, to: (@convention(c) () -> Int32).self)
        var info = Dl_info()
        guard getVersion() == 1, dladdr(version, &info) != 0, let name = info.dli_fname else {
            dlclose(h); throw CLIError.usage("Unknown MoE tiling library")
        }
        handle = h; counter = unsafeBitCast(count, to: (@convention(c) (Int32) -> UInt64).self)
        libraryPath = URL(fileURLWithPath: String(cString: name)).standardizedFileURL.resolvingSymlinksInPath().path
    }
    deinit { dlclose(handle) }
    func setBM(_ value: Int) throws {
        guard [0,16,32].contains(value) else { throw CLIError.usage("BM must be 0, 16 or 32") }
        try MX.synchronize()
        guard setenv("ANERUNNER_MOE_QMM_BM", String(value), 1) == 0 else { throw CLIError.usage("Cannot set MoE tile mode") }
    }
    func counts() -> [String: UInt64] { ["bm16": counter(16), "bm32": counter(32), "bm64": counter(64)] }
}

extension RunnerCLI {
    static func probeGPUMoEPrefillTiling(_ args: Arguments) throws {
        try args.validate(["--model-dir", "--manifest", "--output"])
        let output = try args.require("--output")
        guard !FileManager.default.fileExists(atPath: output) else { throw CLIError.usage("Output must be new") }
        let modelURL = URL(fileURLWithPath: try args.require("--model-dir")).standardizedFileURL.resolvingSymlinksInPath()
        let manifestURL = URL(fileURLWithPath: try args.require("--manifest")).standardizedFileURL
        let data = try Data(contentsOf: manifestURL)
        struct Manifest: Decodable {
            struct Fixture: Decodable { let layer, offset: Int; let file, sha256: String; let shape: [Int] }
            struct Provenance: Decodable { let config_sha256, weight_index_sha256: String }
            let passed, committed: Bool
            let model_directory: String
            let provenance: Provenance
            let fixtures: [Fixture]
        }
        let manifest = try JSONDecoder().decode(Manifest.self, from: data)
        guard manifest.passed, manifest.committed, manifest.fixtures.count == 9,
              manifest.model_directory == modelURL.path,
              manifest.provenance.config_sha256 == (try MoETilingBytes.hash(modelURL.appendingPathComponent("config.json"))),
              manifest.provenance.weight_index_sha256 == (try MoETilingBytes.hash(modelURL.appendingPathComponent("model.safetensors.index.json"))) else {
            throw CLIError.usage("Require a committed full-golden capture for this exact model configuration")
        }
        let runtime = try MoEPrefillTilingRuntime()
        try runtime.setBM(0)
        defer { try? runtime.setBM(0) }
        var cases = [[String: Any]]()
        var report: [String: Any] = [
            "schema": "qwen38-moe-prefill-tiling-probe-v1", "complete": false, "passed": false,
            "manifest": manifestURL.path, "manifest_sha256": MoETilingBytes.digest(data),
            "model_directory": modelURL.path, "library_path": runtime.libraryPath,
            "library_sha256": try MoETilingBytes.hash(URL(fileURLWithPath: runtime.libraryPath)),
            "executable_sha256": try MoETilingBytes.hash(URL(fileURLWithPath: CommandLine.arguments[0])),
            "baseline_bm": 0, "candidate_bm": 16, "forced_bm32_control": true,
            "shared_expert": "Always evaluated, independently sigmoid-gated; not selected by top10 and not modified by this Q4 dispatcher",
            "timing_scope": "Resident full MoE forward and y.eval; routing, routed experts and shared expert included. Mode switch/drain, diagnostics, host copies and loading excluded.",
            "notes": ["Nine actual full-model inputs plus one derived 205-row prefix for nonaligned tile tail coverage.",
                "BM16/WM1/WN2 preserves 16x32 SIMD matrix fragments; BM32 already loops over expert segments. No claim that BM16 halves wasted rows.",
                "Dispatch counters count encoded target-shaped matmuls, not GPU completion or DRAM bytes. Output evaluation finishes each sample.",
                "All variants use original affine Q4/group64 expert weights and BF16 scales/biases. No requantization.",
                "Microbenchmark success does not establish full-model speedup. Shared-expert work remains identical."]]
        func save() throws { report["cases"] = cases; try emit(report, to: output) }
        try save()
        do {
            var cacheLimit = 0
            try MX.check(mlx_set_cache_limit(&cacheLimit, 128 * 1024 * 1024), "bound tiling probe cache")
            defer { var ignored = 0; _ = mlx_set_cache_limit(&ignored, cacheLimit) }
            for layer in [0,23,47] {
                let weights = try GPUWeights(modelDirectory: modelURL)
                let moe = try GPUMoE(weights: weights, layer: layer, prefillAccumulation: .reference)
                let fixtures = manifest.fixtures.filter { $0.layer == layer }.sorted { $0.offset < $1.offset }
                guard fixtures.map(\.offset) == [0,4992,9984] else { throw CLIError.usage("Unexpected capture positions") }
                for fixture in fixtures {
                    let url = URL(fileURLWithPath: fixture.file)
                    guard try MoETilingBytes.hash(url) == fixture.sha256 else { throw CLIError.usage("Fixture checksum mismatch") }
                    let original = try MoETilingBytes.loadX(url)
                    guard original.shape == fixture.shape, original.shape == [1,416,2560], original.dtype == MLX_BFLOAT16 else {
                        throw CLIError.usage("Fixture must contain native BF16 x[1,416,2560]")
                    }
                    try original.eval()
                    let lengths = layer == 0 && fixture.offset == 0 ? [416,205] : [416]
                    for length in lengths {
                        let x = length == 416 ? original : try MX.slice(original, starts: [0,0,0], ends: [1,length,2560])
                        try x.eval()
                        FileHandle.standardError.write(Data("MoE tiling: layer \(layer), offset \(fixture.offset), S\(length)\n".utf8))
                        var item: [String: Any] = ["layer": layer, "offset": fixture.offset, "shape": x.shape,
                            "derived_prefix": length != 416, "fixture": fixture.file, "passed": false]
                        func forward(_ bm: Int, diagnostics: Bool) throws -> GPUMoEOutput {
                            try runtime.setBM(bm)
                            let result = try moe.forward(x, diagnostics: diagnostics)
                            try MX.eval([result.y] + Array(result.diagnostics.values))
                            return result
                        }
                        let before = runtime.counts()
                        let reference = try forward(0, diagnostics: true)
                        let candidate = try forward(16, diagnostics: true)
                        let control = try forward(32, diagnostics: false)
                        let after = runtime.counts()
                        guard after["bm16"]! > before["bm16"]!, after["bm32"]! > before["bm32"]! else {
                            throw CLIError.usage("Expected target-shaped BM16/BM32 dispatches were not encoded")
                        }
                        var comparisons = [String: Any](), exact = true
                        for name in ["selected_experts", "routing_weights", "selected_expert_outputs", "routed_sum", "shared_down", "shared_gate", "output"] {
                            guard let a = reference.diagnostics[name], let b = candidate.diagnostics[name] else {
                                throw CLIError.usage("Missing real MoE diagnostic: \(name)")
                            }
                            let comparison = try MoETilingBytes.compare(a,b)
                            comparisons[name] = comparison; exact = exact && (comparison["exact"] as? Bool == true)
                        }
                        let controlComparison = try MoETilingBytes.compare(reference.y,control.y)
                        comparisons["forced32_vs_stock_y"] = controlComparison
                        exact = exact && (controlComparison["exact"] as? Bool == true)
                        let plainA = try forward(0,diagnostics: false), plainB = try forward(16,diagnostics: false)
                        for (name,a,b) in [("plain_y",plainA.y,plainB.y),("stock_diagnostics_vs_plain",reference.y,plainA.y),("candidate_diagnostics_vs_plain",candidate.y,plainB.y)] {
                            let c = try MoETilingBytes.compare(a,b); comparisons[name] = c; exact = exact && (c["exact"] as? Bool == true)
                        }
                        item["comparisons"] = comparisons; item["dispatches_before"] = before; item["dispatches_after_diagnostics"] = after
                        let ids = try reference.diagnostics["selected_experts"]!.ints().map(Int.init)
                        guard ids.count == length * 10, ids.allSatisfy({ (0..<512).contains($0) }) else { throw CLIError.usage("Invalid routed IDs") }
                        var counts = [Int](repeating: 0, count: 512)
                        for id in ids { counts[id] += 1 }
                        let activeCounts = counts.filter { $0 > 0 }.sorted()
                        item["routing"] = ["assignment_count": ids.count, "counts_per_expert": counts,
                            "selected_expert_ids": ids, "active_experts": activeCounts.count,
                            "mean_per_all_512": Double(ids.count)/512,
                            "mean_per_active_expert": Double(ids.count)/Double(activeCounts.count),
                            "max_per_expert": activeCounts.last!, "p50_per_active_expert": activeCounts[activeCounts.count/2]]
                        guard exact else { cases.append(item); try save(); throw CLIError.usage("MoE tiling exact output gate failed") }
                        for _ in 0..<3 { _ = try forward(0,diagnostics: false); _ = try forward(16,diagnostics: false) }
                        var samples = [[String: Any]](), totals = [0:0.0,16:0.0]
                        let timedBefore = runtime.counts()
                        for _ in 0..<4 {
                            for bm in [0,16,16,0] {
                                try runtime.setBM(bm)
                                let start = DispatchTime.now().uptimeNanoseconds
                                let result = try moe.forward(x)
                                try result.y.eval()
                                let ms = Double(DispatchTime.now().uptimeNanoseconds-start)*1e-6
                                guard ms.isFinite, ms > 0 else { throw CLIError.usage("Invalid elapsed time") }
                                totals[bm]! += ms; samples.append(["bm":bm,"milliseconds":ms])
                            }
                        }
                        item["warmups_per_variant"] = 3; item["samples_per_variant"] = 8; item["samples"] = samples
                        item["baseline_mean_ms"] = totals[0]!/8; item["candidate_mean_ms"] = totals[16]!/8
                        item["observed_speedup_percent"] = (totals[0]!/totals[16]!-1)*100
                        let timedAfter = runtime.counts()
                        let timedDelta = timedAfter.mapValues { $0 }
                            .map { ($0.key, $0.value - timedBefore[$0.key]!) }
                        let deltas = Dictionary(uniqueKeysWithValues: timedDelta)
                        item["timed_dispatches_before"] = timedBefore; item["timed_dispatches_after"] = timedAfter
                        item["timed_dispatches_delta"] = deltas
                        guard deltas["bm16"] == 24, deltas["bm32"] == 24, deltas["bm64"] == 0 else {
                            cases.append(item); try save()
                            throw CLIError.usage("Timed samples did not encode exactly three target matmuls per MoE forward")
                        }
                        item["passed"] = true; cases.append(item); try save()
                    }
                }
            }
            report["complete"] = true; report["passed"] = cases.count == 10; report["final_memory"] = try MX.memory()
            try save()
        } catch {
            report["fatal_error"] = String(describing:error); report["passed"] = false; try save(); throw error
        }
    }
}

enum MoETilingBytes {
    static func digest(_ data: Data) -> String { SHA256.hash(data:data).map { String(format:"%02x",$0) }.joined() }
    static func hash(_ url: URL) throws -> String { digest(try Data(contentsOf:url)) }
    static func loadX(_ url: URL) throws -> Tensor {
        var arrays = mlx_map_string_to_array_new(), metadata = mlx_map_string_to_string_new()
        let cpu = mlx_default_cpu_stream_new()
        defer { _ = mlx_map_string_to_array_free(arrays); _ = mlx_map_string_to_string_free(metadata); _ = mlx_stream_free(cpu) }
        try MX.check(mlx_load_safetensors(&arrays,&metadata,url.path,cpu),"load real MoE input")
        return try MX.output("MoE fixture x") { mlx_map_string_to_array_get(&$0,arrays,"x") }
    }
    static func bytes(_ tensor: Tensor) throws -> Data {
        let contiguous = try MX.contiguous(tensor)
        let b = try MX.output("tiling byte view") { mlx_view(&$0,contiguous.handle,MLX_UINT8,MX.stream) }
        try b.eval()
        guard let p = mlx_array_data_uint8(b.handle), b.count == tensor.nbytes else { throw CLIError.usage("Missing tensor bytes") }
        return Data(bytes:p,count:b.count)
    }
    static func compare(_ a: Tensor,_ b: Tensor) throws -> [String: Any] {
        let aa = try bytes(a), bb = try bytes(b)
        let av = try a.floats(), bv = try b.floats()
        let finite = av.allSatisfy(\.isFinite) && bv.allSatisfy(\.isFinite)
        if !finite {
            return ["exact":false,"shape":a.shape,"all_finite":false,
                "reference_sha256":digest(aa),"candidate_sha256":digest(bb),
                "max_abs":NSNull(),"relative_l2":NSNull()]
        }
        var maxAbs = 0.0, sum = 0.0, norm = 0.0
        if aa != bb { for (x,y) in zip(av,bv) { let d = Double(x)-Double(y); maxAbs=max(maxAbs,abs(d));sum+=d*d;norm+=Double(x)*Double(x) } }
        return ["exact": a.shape == b.shape && a.dtype == b.dtype && aa == bb && finite,
            "shape":a.shape,"all_finite":finite,"reference_sha256":digest(aa),"candidate_sha256":digest(bb),
            "max_abs":maxAbs,"relative_l2": norm > 0 ? sqrt(sum/norm) : 0]
    }
}
