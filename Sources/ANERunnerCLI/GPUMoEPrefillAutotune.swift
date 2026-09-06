import ANERunnerGPU
import CMLX
import CryptoKit
import Darwin
import Foundation

/// Only used by serial experiments. Drain before changing the native selector;
/// it is not a request-safe backend interface for cooperative scheduling.
final class MoEPrefillAutotuneRuntime {
    let libraryPath: String
    let librarySHA256: String
    private let handle: UnsafeMutableRawPointer
    private let count: @convention(c) (Int32) -> UInt64
    init() throws {
        guard let h = dlopen(nil,RTLD_NOW) else { throw CLIError.usage("Cannot inspect MLX") }
        guard let v = dlsym(h,"anemlx_moe_qmm_tune_version"),
              let c = dlsym(h,"anemlx_moe_qmm_tune_dispatch_count") else {
            dlclose(h); throw CLIError.usage("Load the isolated MoE autotune library")
        }
        var info = Dl_info()
        guard unsafeBitCast(v,to:(@convention(c) () -> Int32).self)() == 1,
              dladdr(v,&info) != 0, let path = info.dli_fname else {
            dlclose(h); throw CLIError.usage("Unknown MoE autotune library")
        }
        handle = h; count = unsafeBitCast(c,to:(@convention(c) (Int32) -> UInt64).self)
        libraryPath = URL(fileURLWithPath:String(cString:path)).resolvingSymlinksInPath().path
        librarySHA256 = try MoETilingBytes.hash(URL(fileURLWithPath:libraryPath))
    }
    deinit { dlclose(handle) }
    func select(_ id: Int) throws {
        guard (0...4).contains(id) else { throw CLIError.usage("Unknown native MoE configuration") }
        try MX.synchronize()
        guard setenv("ANERUNNER_MOE_QMM_CONFIG",String(id),1) == 0 else { throw CLIError.usage("Cannot select native MoE configuration") }
    }
    func counts() -> [UInt64] { (0...4).map { count(Int32($0)) } }
}

extension RunnerCLI {
    static func autotuneGPUMoEPrefill(_ args: Arguments) throws {
        try args.validate(["--model-dir","--manifest","--output","--config-output"])
        let output = try args.require("--output"), configOutput = try args.require("--config-output")
        guard output != configOutput, !FileManager.default.fileExists(atPath:output),
              !FileManager.default.fileExists(atPath:configOutput) else { throw CLIError.usage("Autotune outputs must be new") }
        let model = URL(fileURLWithPath:try args.require("--model-dir")).resolvingSymlinksInPath()
        let manifestPath = try args.require("--manifest"), manifestData = try Data(contentsOf:URL(fileURLWithPath:manifestPath))
        struct Manifest: Decodable {
            struct Fixture: Decodable { let layer,offset:Int; let file,sha256:String }
            struct Provenance: Decodable { let config_sha256,weight_index_sha256:String }
            let passed,committed:Bool
            let model_directory:String
            let provenance:Provenance
            let fixtures:[Fixture]
        }
        let manifest = try JSONDecoder().decode(Manifest.self,from:manifestData)
        guard manifest.passed, manifest.committed, manifest.fixtures.count == 9,
              manifest.model_directory == model.path,
              manifest.provenance.config_sha256 == (try MoETilingBytes.hash(model.appendingPathComponent("config.json"))),
              manifest.provenance.weight_index_sha256 == (try MoETilingBytes.hash(model.appendingPathComponent("model.safetensors.index.json"))) else {
            throw CLIError.usage("Autotune needs the committed golden-checked capture for this model")
        }
        let runtime = try MoEPrefillAutotuneRuntime()
        try runtime.select(0); defer { try? runtime.select(0) }
        struct Candidate {
            let native,threads:Int
            var key:String { "qmm\(native)-reduce\(threads)" }
        }
        let candidates = (0...4).flatMap { n in [0,128,256,512].map { Candidate(native:n,threads:$0) } }
        let reference = candidates[0]
        var rejected = Set<String>(), cases = [[String:Any]]()
        var totals = [String:(a:Double,b:Double,count:Int)]()
        var report:[String:Any] = ["schema":"qwen38-prefill-moe-autotune-v1","complete":false,"passed":false,
            "manifest":manifestPath,"manifest_sha256":MoETilingBytes.digest(manifestData),
            "model_directory":model.path,"device_target":"Apple M5 Max",
            "library_path":runtime.libraryPath,"library_sha256":runtime.librarySHA256,
            "executable_sha256":try MoETilingBytes.hash(URL(fileURLWithPath:CommandLine.arguments[0])),
            "search_space":candidates.map { ["key":$0.key,"native":$0.native,"reduction_threads":$0.threads] as [String:Any] },
            "timing_scope":"Full resident MoE fresh graph plus y.eval. No diagnostic outputs, host copies, mode switches or loading in timing.",
            "selection_scope":"Nine actual S416 inputs. Derived S205/S240 inputs cover tails and are excluded from selection scoring.",
            "minimum_micro_speedup_percent":2.0,
            "warmups_per_variant_per_pair":2,"timed_samples_per_variant_per_pair":4,
            "timed_order":"ABBA repeated twice",
            "notes":["Reference is eligible to win. Every non-reference candidate must pass all eleven input cases.",
                "Numerical failures reject a configuration; runtime/dispatch failures stop the experiment.",
                "Saved native configuration is for serial prefill experiments only, with stream drain and baseline reset before decode.",
                "The request-local reduction table supports measured lengths; other lengths fall back to reference.",
                "Microbenchmark selection is a candidate, not full-model or production acceptance."]]
        func save() throws { report["cases"] = cases; report["rejected"] = rejected.sorted(); try emit(report,to:output) }
        try save()
        do {
            var previousCache = 0
            try MX.check(mlx_set_cache_limit(&previousCache,128*1024*1024),"autotune cache cap")
            defer { var ignored = 0; _ = mlx_set_cache_limit(&ignored,previousCache) }
            for layer in [0,23,47] {
                let weights = try GPUWeights(modelDirectory:model), moe = try GPUMoE(weights:weights,layer:layer)
                let fixtures = manifest.fixtures.filter { $0.layer == layer }.sorted { $0.offset < $1.offset }
                guard fixtures.map(\.offset) == [0,4992,9984] else { throw CLIError.usage("Unexpected capture positions") }
                for fixture in fixtures {
                    let url = URL(fileURLWithPath:fixture.file)
                    guard try MoETilingBytes.hash(url) == fixture.sha256 else { throw CLIError.usage("Fixture changed") }
                    let original = try MoETilingBytes.loadX(url)
                    guard original.shape == [1,416,2560], original.dtype == MLX_BFLOAT16 else { throw CLIError.usage("Invalid real MoE fixture") }
                    for length in (layer == 0 && fixture.offset == 0 ? [416,205,240] : [416]) {
                        let x = length == 416 ? original : try MX.slice(original,starts:[0,0,0],ends:[1,length,2560])
                        try x.eval()
                        var trials = [[String:Any]]()
                        var item:[String:Any] = ["layer":layer,"offset":fixture.offset,"tokens":length,"derived":length != 416]
                        func forward(_ c:Candidate,_ diagnostics:Bool) throws -> GPUMoEOutput {
                            try runtime.select(c.native)
                            let result = try moe.forward(x,diagnostics:diagnostics,prefillReductionThreadgroup:c.threads == 0 ? nil : c.threads)
                            try MX.eval([result.y] + Array(result.diagnostics.values))
                            return result
                        }
                        let refDiag = try forward(reference,true), refPlain = try forward(reference,false)
                        guard try MoETilingBytes.compare(refDiag.y,refPlain.y)["exact"] as? Bool == true else {
                            throw CLIError.usage("Reference diagnostic/plain mismatch")
                        }
                        for c in candidates.dropFirst() where !rejected.contains(c.key) {
                            FileHandle.standardError.write(Data("Autotune L\(layer) P\(fixture.offset) S\(length) \(c.key)\n".utf8))
                            let before = runtime.counts(), diag = try forward(c,true), plain = try forward(c,false), after = runtime.counts()
                            let delta = zip(after,before).map { $0-$1 }
                            guard delta.enumerated().allSatisfy({ $0.element == ($0.offset == c.native ? 6 : 0) }) else {
                                throw CLIError.usage("Candidate did not encode six matmuls with its selected native config")
                            }
                            var comparisons = [String:Any](), exact = true
                            for name in ["selected_experts","routing_weights","selected_expert_outputs","routed_sum","shared_down","shared_gate","output"] {
                                guard let a = refDiag.diagnostics[name], let b = diag.diagnostics[name] else { throw CLIError.usage("Missing MoE diagnostics") }
                                let comparison = try MoETilingBytes.compare(a,b); comparisons[name] = comparison
                                exact = exact && (comparison["exact"] as? Bool == true)
                            }
                            for (name,a,b) in [("plain_y",refPlain.y,plain.y),("candidate_diagnostic_plain",diag.y,plain.y)] {
                                let comparison = try MoETilingBytes.compare(a,b); comparisons[name] = comparison
                                exact = exact && (comparison["exact"] as? Bool == true)
                            }
                            var trial:[String:Any] = ["key":c.key,"native":c.native,"reduction_threads":c.threads,
                                "exact":exact,"comparisons":comparisons,"diagnostic_dispatch_delta":delta]
                            if !exact { rejected.insert(c.key); trials.append(trial); continue }
                            for _ in 0..<2 { _ = try forward(reference,false); _ = try forward(c,false) }
                            var a = 0.0,b = 0.0,samples = [[String:Any]]()
                            let timedBefore = runtime.counts()
                            for _ in 0..<2 {
                                for candidate in [false,true,true,false] {
                                    let chosen = candidate ? c : reference
                                    try runtime.select(chosen.native)
                                    let start = DispatchTime.now().uptimeNanoseconds
                                    let y = try moe.forward(x,prefillReductionThreadgroup:chosen.threads == 0 ? nil : chosen.threads).y
                                    try y.eval()
                                    let ms = Double(DispatchTime.now().uptimeNanoseconds-start)*1e-6
                                    guard ms.isFinite,ms>0 else { throw CLIError.usage("Invalid elapsed time") }
                                    if candidate { b += ms } else { a += ms }
                                    samples.append(["candidate":candidate,"milliseconds":ms])
                                }
                            }
                            let timedDelta = zip(runtime.counts(),timedBefore).map { $0-$1 }
                            guard timedDelta.enumerated().allSatisfy({ $0.element == ($0.offset == 0 ? 12 : 0) + ($0.offset == c.native ? 12 : 0) }) else {
                                throw CLIError.usage("Timed MoE dispatches differ from selected configuration")
                            }
                            trial["samples"] = samples; trial["baseline_mean_ms"] = a/4; trial["candidate_mean_ms"] = b/4
                            trial["speedup_percent"] = (a/b-1)*100; trial["timed_dispatch_delta"] = timedDelta
                            if length == 416 {
                                let previous = totals[c.key] ?? (0,0,0)
                                totals[c.key] = (previous.a+a/4,previous.b+b/4,previous.count+1)
                            }
                            trials.append(trial)
                        }
                        item["trials"] = trials; cases.append(item); try save()
                    }
                }
            }
            let ranked = candidates.dropFirst().filter { !rejected.contains($0.key) && totals[$0.key]?.count == 9 }
                .sorted { totals[$0.key]!.a/totals[$0.key]!.b > totals[$1.key]!.a/totals[$1.key]!.b }
            let winner = ranked.first.flatMap { totals[$0.key]!.a/totals[$0.key]!.b >= 1.02 ? $0 : nil } ?? reference
            let config = GPUMoEPrefillConfiguration(threadgroups:winner.threads == 0 ? [:] : [205:winner.threads,240:winner.threads,416:winner.threads])
            try config.validated()
            report["ranking"] = ranked.map { c in ["key":c.key,"baseline_sum_ms":totals[c.key]!.a,
                "candidate_sum_ms":totals[c.key]!.b,"speedup_percent":(totals[c.key]!.a/totals[c.key]!.b-1)*100] as [String:Any] }
            report["winner"] = winner.key; report["selected_reference"] = winner.key == reference.key
            report["complete"] = cases.count == 11; report["passed"] = cases.count == 11
            report["config_output"] = configOutput; try save()
            var saved = try JSONSerialization.jsonObject(with:JSONEncoder().encode(config)) as! [String:Any]
            saved["native_configuration"] = winner.native; saved["native_library_sha256"] = runtime.librarySHA256
            saved["device_target"] = "Apple M5 Max"; saved["model_directory"] = model.path
            saved["autotune_report"] = URL(fileURLWithPath:output).standardizedFileURL.path
            saved["autotune_report_sha256"] = try MoETilingBytes.hash(URL(fileURLWithPath:output))
            saved["status"] = "micro_selected_requires_full_pd_validation"
            try emit(saved,to:configOutput)
        } catch { report["fatal_error"] = String(describing:error); report["passed"] = false; try save(); throw error }
    }
}
