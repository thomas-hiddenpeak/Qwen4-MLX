import ANERunnerGPU
import CMLX
import CryptoKit
import Foundation

extension RunnerCLI {
    /// Sequence validation, with an optional blocked-only recurrence timing probe.
    static func probeGPUSequence(_ args: Arguments) throws {
        try args.validate(["--model-dir","--fixture","--output","--max-relative-l2","--fused","--prework","--blocked-only"])
        let model = URL(fileURLWithPath: try args.require("--model-dir"),isDirectory: true)
        let manifestURL = URL(fileURLWithPath: try args.require("--fixture"))
        let manifestData = try Data(contentsOf: manifestURL)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let manifest = try decoder.decode(GPUSequenceManifest.self,from: manifestData)
        guard manifest.schema == "qwen38-gpu-sequence-reference-v1", !manifest.cases.isEmpty,
              let tolerance = Double(args["--max-relative-l2"] ?? "0.005"), tolerance >= 0, tolerance.isFinite,
              ["true","false"].contains(args["--fused"] ?? "true"),
              ["true","false"].contains(args["--prework"] ?? "false"),
              ["true","false"].contains(args["--blocked-only"] ?? "false") else {
            throw CLIError.usage("Invalid sequence reference or tolerance; --fused/--prework accept true/false")
        }
        let runFused = (args["--fused"] ?? "true") == "true"
        let runPrework = (args["--prework"] ?? "false") == "true"
        let directory = manifestURL.deletingLastPathComponent().standardizedFileURL.resolvingSymlinksInPath()
        if args["--blocked-only"] == "true" {
            guard let step = manifest.cases.first(where: { $0.kind == "gdn" })?.steps.first,
                  step.file == URL(fileURLWithPath: step.file).lastPathComponent else {
                throw CLIError.usage("Blocked GDN probe requires a GDN recurrence fixture")
            }
            let url = directory.appendingPathComponent(step.file).resolvingSymlinksInPath()
            guard url.deletingLastPathComponent() == directory, try sequenceSHA256(url) == step.sha256 else {
                throw CLIError.usage("Blocked GDN fixture path or SHA-256 mismatch")
            }
            try probeBlockedGDN(GPUSequenceTensors(url), tolerance: tolerance, output: args["--output"])
            return
        }
        let weights = try GPUWeights(modelDirectory: model)
        let fused = runFused ? try GPUGatedDeltaNetFused() : nil
        var cases = [[String: Any]](), allPassed = true
        for test in manifest.cases {
            guard !test.steps.isEmpty, ["gdn","attention"].contains(test.kind) else {
                throw CLIError.usage("Invalid sequence test \(test.id)")
            }
            let gdn = test.kind == "gdn" ? try GPUGatedDeltaNet(layer: test.layer,weights: weights) : nil
            let fusedBlock = test.kind == "gdn" && runFused ? try GPUGatedDeltaNet(layer: test.layer,weights: weights,recurrence: .fused) : nil
            let preworkBlock = test.kind == "gdn" && runPrework ? try GPUGatedDeltaNet(layer: test.layer,weights: weights,recurrence: .fused,fusedPrework: true) : nil
            let attention = test.kind == "attention" ? try GPUAttention(layer: test.layer,weights: weights) : nil
            var gdnState = GPUGatedDeltaNet.State(), fusedState = GPUGatedDeltaNet.State(), attentionState = GPUAttention.State()
            var preworkState = GPUGatedDeltaNet.State()
            var stepReports = [[String: Any]]()
            var firstInput: Tensor?, firstActual = [Float]()
            var offset = 0
            for (index, step) in test.steps.enumerated() {
                guard step.offsetBefore == offset, step.offsetAfter > offset,
                      step.file == URL(fileURLWithPath: step.file).lastPathComponent else {
                    throw CLIError.usage("Invalid sequence step/file in \(test.id)")
                }
                let url = directory.appendingPathComponent(step.file).resolvingSymlinksInPath()
                guard url.deletingLastPathComponent() == directory else {
                    throw CLIError.usage("Sequence fixture escapes its directory")
                }
                guard try sequenceSHA256(url) == step.sha256 else {
                    throw CLIError.usage("Sequence fixture SHA-256 mismatch: \(step.file)")
                }
                let fixture = try GPUSequenceTensors(url)
                let input = try fixture.require("input")
                guard input.shape.count == 3, input.shape[1] == step.offsetAfter-offset else {
                    throw CLIError.usage("Sequence input/offset mismatch: \(step.file)")
                }
                let output: Tensor, stateTensors: [String: Tensor]
                if let gdn {
                    output = try gdn.forward(input,state: &gdnState)
                    stateTensors = ["convHistory": gdnState.convHistory!, "recurrent": gdnState.recurrent!]
                    guard gdnState.offset == step.offsetAfter else { throw CLIError.usage("GDN offset mismatch") }
                } else if let attention {
                    output = try attention.forward(input,state: &attentionState)
                    var tensors = ["keys": attentionState.keys!, "values": attentionState.values!,
                                   "rawIndexerKeys": attentionState.rawIndexerKeys!]
                    if let pooled = attentionState.pooledIndexerKeys { tensors["pooledIndexerKeys"] = pooled }
                    stateTensors = tensors
                    guard attentionState.offset == step.offsetAfter,
                          step.qsaExpected == (attentionState.pooledIndexerKeys != nil) else {
                        throw CLIError.usage("QSA threshold/state mismatch")
                    }
                } else { throw CLIError.usage("No sequence backend") }
                try MX.eval([output] + Array(stateTensors.values))
                var comparisons = [[String: Any]]()
                let actual = try output.floats()
                comparisons.append(try sequenceCompare(actual,shape: output.shape,
                                                       expected: fixture.require("expected.output"),name: "output",tolerance: tolerance))
                for key in stateTensors.keys.sorted() {
                    let tensor = stateTensors[key]!
                    comparisons.append(try sequenceCompare(tensor.floats(),shape: tensor.shape,
                                                           expected: fixture.require("expected.state."+key),name: "state."+key,tolerance: tolerance))
                }
                if let fused, test.kind == "gdn" {
                    let candidate = try fused.apply(q: fixture.require("recurrence.q"),k: fixture.require("recurrence.k"),
                                                    v: fixture.require("recurrence.v"),decay: fixture.require("recurrence.decay"),
                                                    beta: fixture.require("recurrence.beta"),state: fixture.require("recurrence.stateIn"))
                    try MX.eval([candidate.y,candidate.state])
                    for (name,tensor,reference) in [("fused.y",candidate.y,"recurrence.y"),
                                                     ("fused.state",candidate.state,"recurrence.stateOut")] {
                        comparisons.append(try sequenceCompare(tensor.floats(),shape: tensor.shape,
                                                               expected: fixture.require(reference),name: name,tolerance: tolerance))
                    }
                }
                if let fusedBlock {
                    let full = try fusedBlock.forward(input,state: &fusedState)
                    try MX.eval([full] + fusedState.tensors)
                    for (name,tensor,reference) in [("fusedBlock.output",full,"expected.output"),
                                                     ("fusedBlock.convHistory",fusedState.convHistory!,"expected.state.convHistory"),
                                                     ("fusedBlock.recurrent",fusedState.recurrent!,"expected.state.recurrent")] {
                        comparisons.append(try sequenceCompare(tensor.floats(),shape: tensor.shape,
                                                               expected: fixture.require(reference),name: name,tolerance: tolerance))
                    }
                }
                if let preworkBlock {
                    let full = try preworkBlock.forward(input,state: &preworkState)
                    try MX.eval([full] + preworkState.tensors)
                    guard preworkState.offset == step.offsetAfter else { throw CLIError.usage("Prework state offset mismatch") }
                    for (name,tensor,reference) in [("preworkBlock.output",full,"expected.output"),
                                                     ("preworkBlock.convHistory",preworkState.convHistory!,"expected.state.convHistory"),
                                                     ("preworkBlock.recurrent",preworkState.recurrent!,"expected.state.recurrent")] {
                        // This is an exact optimization gate, not the looser
                        // tolerance permitted for the composed recurrence.
                        comparisons.append(try sequenceCompare(tensor.floats(),shape: tensor.shape,
                                                               expected: fixture.require(reference),name: name,tolerance: 0))
                    }
                }
                let passed = comparisons.allSatisfy { $0["passed"] as? Bool == true }
                allPassed = allPassed && passed
                stepReports.append(["phase": step.phase,"fixture": step.file,"fixture_sha256": step.sha256,
                                    "offset_before": offset,"offset_after": step.offsetAfter,
                                    "qsa_expected": step.qsaExpected,"passed": passed,"comparisons": comparisons])
                if index == 0 { firstInput = input; firstActual = actual }
                offset = step.offsetAfter
            }
            // Reset the SAME state holder after its complete trajectory, then
            // verify that the initial output is exactly reproducible.
            gdnState.reset(); attentionState.reset()
            guard gdnState.offset == 0, gdnState.tensors.isEmpty,
                  attentionState.offset == 0, attentionState.tensors.isEmpty,
                  let firstInput else { throw CLIError.usage("Sequence reset failed") }
            let resetOutput: Tensor
            if let gdn { resetOutput = try gdn.forward(firstInput,state: &gdnState) }
            else { resetOutput = try attention!.forward(firstInput,state: &attentionState) }
            try MX.eval([resetOutput] + gdnState.tensors + attentionState.tensors)
            let resetEqual = try resetOutput.floats() == firstActual
            allPassed = allPassed && resetEqual
            cases.append(["id": test.id,"kind": test.kind,"layer": test.layer,"steps": stepReports,
                          "reset_output_exactly_reproduced": resetEqual])
        }
        let report: [String: Any] = ["schema": "qwen38-swift-gpu-sequence-probe-v1","passed": allPassed,
                                   "reference_manifest_sha256": SHA256.hash(data: manifestData).map { String(format: "%02x",$0) }.joined(),
                                   "source_commit": manifest.sourceCommit,"max_relative_l2": tolerance,
                                   "fused_gdn_candidate_checked": runFused,"fused_gdn_tested_as_full_block": runFused,
                                   "prework_gdn_candidate_checked": runPrework,"prework_gdn_exact_gate": runPrework,
                                   "fused_gdn_is_default": false,
                                   "cases": cases,"reference_notes": manifest.notes,
                                   "notes": ["Numerical validation only; no performance or physical DRAM measurement.",
                                             "Inputs are captured MoE activations replayed at other boundaries, not native attention captures.",
                                             "QSA threshold trajectory repeats captured activations; it is not a real long-context prompt.",
                                             "Python MLX references do not establish whole-model generation correctness."]]
        try emit(report,to: args["--output"])
        if !allPassed { throw CLIError.usage("GPU sequence numerical/reset gate failed; see saved report") }
    }
}

private func probeBlockedGDN(_ fixture: GPUSequenceTensors, tolerance: Double, output: String?) throws {
    let scalar = try GPUGatedDeltaNetFused(), blocked = try GPUGatedDeltaNetBlocked()
    let source = try ["q","k","v","decay","beta"].map { try fixture.require("recurrence." + $0) }
    let initial = try fixture.require("recurrence.stateIn")
    func inputs(_ count: Int) throws -> [Tensor] {
        try source.map { tensor in
            let repeated = try MX.concat(Array(repeating: tensor, count: (count + tensor.shape[1] - 1) / tensor.shape[1]), axis: 1)
            var end = tensor.shape; end[1] = count
            return try MX.copy(MX.slice(repeated, starts: Array(repeating: 0, count: end.count), ends: end))
        }
    }
    var cases = [[String: Any]]()
    let one = try inputs(1)
    for count in [64,65,512] {
        let x = try inputs(count)
        let reference = try scalar.apply(q: x[0], k: x[1], v: x[2], decay: x[3], beta: x[4], state: initial)
        let candidate = try blocked.apply(q: x[0], k: x[1], v: x[2], decay: x[3], beta: x[4], state: initial)
        try MX.eval([reference.y, reference.state, candidate.y, candidate.state])
        var comparisons = [[String: Any]]()
        for (name,a,b) in [("y",candidate.y,reference.y),("state",candidate.state,reference.state)] {
            comparisons.append(try sequenceCompare(a.floats(), shape: a.shape, expected: b, name: name, tolerance: tolerance))
        }
        let referenceNext = try scalar.apply(q: one[0], k: one[1], v: one[2], decay: one[3], beta: one[4], state: reference.state)
        let candidateNext = try scalar.apply(q: one[0], k: one[1], v: one[2], decay: one[3], beta: one[4], state: candidate.state)
        try MX.eval([referenceNext.y, referenceNext.state, candidateNext.y, candidateNext.state])
        for (name,a,b) in [("decode.y",candidateNext.y,referenceNext.y),("decode.state",candidateNext.state,referenceNext.state)] {
            comparisons.append(try sequenceCompare(a.floats(), shape: a.shape, expected: b, name: name, tolerance: tolerance))
        }
        var scalarTimes = [Double](), blockedTimes = [Double]()
        for repetition in 0..<10 {
            for useBlocked in repetition % 2 == 0 ? [false,true] : [true,false] {
                let start = DispatchTime.now().uptimeNanoseconds
                let result = useBlocked
                    ? try blocked.apply(q: x[0], k: x[1], v: x[2], decay: x[3], beta: x[4], state: initial)
                    : try scalar.apply(q: x[0], k: x[1], v: x[2], decay: x[3], beta: x[4], state: initial)
                try MX.eval([result.y,result.state])
                let seconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e9
                if repetition >= 2 {
                    if useBlocked { blockedTimes.append(seconds) } else { scalarTimes.append(seconds) }
                }
            }
        }
        cases.append(["sequence": count, "comparisons": comparisons,
                      "scalar_seconds": scalarTimes, "blocked_seconds": blockedTimes,
                      "passed": comparisons.allSatisfy { $0["passed"] as? Bool == true }])
    }
    let passed = cases.allSatisfy { $0["passed"] as? Bool == true }
    try RunnerCLI.emit(["schema": "qwen38-blocked-gdn-quick-probe-v1", "passed": passed,
                        "max_relative_l2": tolerance, "cases": cases,
                        "notes": ["Repeated captured recurrence inputs; not a real long-context prompt.",
                                  "Timing includes host dispatch and synchronization; alternating order, two warmups and eight samples per variant.",
                                  "Compares blocked BF16 output/state and the following scalar decode against the existing scalar kernel."]], to: output)
    if !passed { throw CLIError.usage("Blocked GDN quick numerical check failed; see report") }
}

private struct GPUSequenceManifest: Decodable {
    let schema, sourceCommit: String
    let notes: [String]
    let cases: [Test]
    struct Test: Decodable {
        let id, kind: String
        let layer: Int
        let steps: [Step]
    }
    struct Step: Decodable {
        let file, sha256, phase: String
        let offsetBefore, offsetAfter: Int
        let qsaExpected: Bool
    }
}

private struct GPUSequenceTensors {
    let tensors: [String: Tensor]
    init(_ url: URL) throws {
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        guard let prefix = try file.read(upToCount: 8), prefix.count == 8 else { throw CLIError.usage("Short sequence fixture") }
        let count = prefix.enumerated().reduce(UInt64(0)) { $0 | UInt64($1.element) << (8*$1.offset) }
        guard count <= 1_048_576, let header = try file.read(upToCount: Int(count)), header.count == Int(count),
              let descriptor = try JSONSerialization.jsonObject(with: header) as? [String: Any] else {
            throw CLIError.usage("Invalid sequence fixture header")
        }
        var arrays = mlx_map_string_to_array_new(), metadata = mlx_map_string_to_string_new()
        defer { _ = mlx_map_string_to_array_free(arrays); _ = mlx_map_string_to_string_free(metadata) }
        // MLX's file Load primitive is CPU-only. Subsequent computation uses
        // MX.stream and reads the loaded arrays from unified memory.
        let loadStream = mlx_default_cpu_stream_new()
        defer { _ = mlx_stream_free(loadStream) }
        try MX.check(mlx_load_safetensors(&arrays,&metadata,url.path,loadStream),"sequence fixture load")
        var result = [String: Tensor]()
        for name in descriptor.keys where name != "__metadata__" {
            result[name] = try MX.output("sequence fixture \(name)") { mlx_map_string_to_array_get(&$0,arrays,name) }
        }
        tensors = result
    }
    func require(_ key: String) throws -> Tensor {
        guard let result = tensors[key] else { throw CLIError.usage("Missing sequence tensor: \(key)") }
        return result
    }
}

private func sequenceCompare(_ actual: [Float], shape: [Int], expected: Tensor,
                             name: String, tolerance: Double) throws -> [String: Any] {
    guard shape == expected.shape else { throw CLIError.usage("Sequence shape mismatch: \(name)") }
    let reference = try expected.floats()
    guard actual.count == reference.count else { throw CLIError.usage("Sequence element count mismatch") }
    var squaredDifference = 0.0, squaredReference = 0.0, maxAbsolute = 0.0, finite = true, exact = true
    for (a,b) in zip(actual,reference) {
        finite = finite && a.isFinite && b.isFinite
        exact = exact && a == b
        let difference = Double(a)-Double(b)
        squaredDifference += difference*difference
        squaredReference += Double(b)*Double(b)
        maxAbsolute = max(maxAbsolute,abs(difference))
    }
    let relative = sqrt(squaredDifference/max(squaredReference,Double.leastNormalMagnitude))
    return ["name": name,"shape": shape,"elements": actual.count,"finite": finite,"exactly_equal": exact,
            "relative_l2": relative.isFinite ? relative as Any : NSNull(),
            "max_absolute": maxAbsolute.isFinite ? maxAbsolute as Any : NSNull(),
            "passed": finite && relative.isFinite && relative <= tolerance]
}

private func sequenceSHA256(_ url: URL) throws -> String {
    let file = try FileHandle(forReadingFrom: url)
    defer { try? file.close() }
    var hash = SHA256()
    while let data = try file.read(upToCount: 1_048_576), !data.isEmpty { hash.update(data: data) }
    return hash.finalize().map { String(format: "%02x",$0) }.joined()
}
