import ANERunnerCore
import ANERunnerGPU
import CMLX
import Dispatch
import Foundation

extension RunnerCLI {
    /// Original captured routing patterns with independent synthetic activations.
    /// Complete operator timing, including the real router, routed and shared work;
    /// the forced IDs are deliberately not claimed to be this input's trajectory.
    static func probeGPUMoEGroupedGateUp(_ args: Arguments) throws {
        try args.validate(["--model-dir", "--route-cases", "--output"])
        let output = try args.require("--output")
        guard !FileManager.default.fileExists(atPath: output) else {
            throw CLIError.usage("Grouped gate/up probe requires a new --output")
        }
        var cases: [[String: Any]] = [], loads: [[String: Any]] = []
        var report: [String: Any] = [
            "schema": "qwen38-moe-grouped-gateup-probe-v1", "complete": false, "passed": false,
            "timing_scope": "Fresh full forced-routing MoE graph plus synchronous y evaluation: ordinary router, routed experts, shared branch and combine. Grouped C includes its planner. No captured model forward or generation.",
            "routing_scope": "Frozen actual S3 IDs in original token/slot order. Independent synthetic BF16 x; the ordinary router runs and its original BF16 scores are retained by slot. Scores do not correspond to the forced IDs and are not captured trajectory scores.",
            "variants": ["A": "existing per-token S1 routed loop", "B": "existing token-axis gate/up and down", "C": "membership planner + shared-load grouped gate/up + unchanged token-axis down"],
            "warmups_per_variant_per_case": 3, "samples_per_variant_per_case": 12,
            "layer_order": [0,45], "performance_gate": "Correctness/completed timing only sets passed. Before any generation integration, seek at least 5% repeatable complete-operator gain against BOTH A and B on representative overlap cases and no material regression on U30. Six patterns from one captured request are not independent workload evidence.",
            "notes": ["No tensor-bank copy, re-quantization, sorted expert reduction or down-kernel change.",
                      "Logical byte savings are not measured DRAM bandwidth; singleton owners skip nonmember arithmetic but static register demand may still regress.",
                      "Diagnostic tensors are evaluated outside timing. Ordinary timed graphs return y only."]
        ]
        func save() throws { report["cases"] = cases; report["loads"] = loads; try emit(report, to: output) }
        do {
            let routeURL = URL(fileURLWithPath: try args.require("--route-cases"))
                .standardizedFileURL.resolvingSymlinksInPath()
            let frozen = try JSONDecoder().decode(GroupedGateUpRouteCases.self, from: Data(contentsOf: routeURL))
            guard frozen.schema == "qwen38-moe-grouped-gateup-route-cases-v1",
                  frozen.cases.count == 6,
                  frozen.cases.map(\.layer) == [0,0,0,45,45,45],
                  frozen.cases.map(\.uniqueExperts) == [30,24,19,20,15,11],
                  Set(frozen.cases.map(\.name)).count == 6 else {
                throw CLIError.usage("Expected the six frozen actual route cases, ordered layer0 U30/24/19 then layer45 U20/15/11")
            }
            // Range/uniqueness checks precede any expert gather/kernel dispatch.
            // The whole frozen file hash is recorded and externally frozen by the controller.
            for item in frozen.cases {
                let flat = item.expertIDs.flatMap { $0 }
                let counts = Dictionary(grouping: flat, by: { $0 }).mapValues(\.count)
                guard item.sourceKind == "captured", item.tokenCount == 3,
                      item.expertIDs.count == 3,
                      item.expertIDs.allSatisfy({ $0.count == 10 && Set($0).count == 10 }),
                      flat.allSatisfy({ (0..<512).contains($0) }),
                      counts.count == item.uniqueExperts,
                      counts.values.filter({ $0 == 1 }).count == item.n1,
                      counts.values.filter({ $0 == 2 }).count == item.n2,
                      counts.values.filter({ $0 == 3 }).count == item.n3 else {
                    throw CLIError.usage("Malformed frozen routing case \(item.name)")
                }
            }
            report["route_cases"] = ["path":routeURL.path,"sha256":try GPUProbeSupport.hash(routeURL)]
            report["capture_source"] = ["path":frozen.source.path,"sha256":frozen.source.sha256]
            report["analysis_source"] = ["path":frozen.source.analysis.path,"sha256":frozen.source.analysis.sha256]
            let modelURL = URL(fileURLWithPath: try args.require("--model-dir"))
                .standardizedFileURL.resolvingSymlinksInPath()
            let config = try QwenConfiguration(modelDirectory: modelURL)
            guard config.layerCount == 48, config.hiddenSize == 2560, config.expertCount == 512,
                  config.expertsPerToken == 10, config.quantizationBits == 4,
                  config.quantizationGroupSize == 64 else {
                throw CLIError.usage("Grouped gate/up probe is limited to the original fixed model geometry")
            }
            report["model_directory"] = modelURL.path
            report["config_sha256"] = try GPUProbeSupport.hash(modelURL.appendingPathComponent("config.json"))
            report["weight_index_sha256"] = try GPUProbeSupport.hash(modelURL.appendingPathComponent("model.safetensors.index.json"))
            report["mlx_version"] = try GPUProbeSupport.mlxVersion()
            report["operating_system"] = ProcessInfo.processInfo.operatingSystemVersionString
            report["source_hash_scope"] = "Config/index and routing fixture hashes; weight ledger is metadata, not a weight-content checksum. Captured source paths/SHA identify provenance; the controller freezes/checks the actual route-cases file."
            try GDNGEMVMode.reference.apply()
            let seed: UInt64 = 0x514D_4154_5645_4300
            let inputBytes = GPUMatvecCase.syntheticBF16(count: 3 * 2560, seed: seed)
            let rowHashes = (0..<3).map { token in
                GPUMatvecCase.sha256(inputBytes.subdata(in:(token*5120)..<((token+1)*5120)))
            }
            guard Set(rowHashes).count == 3 else { throw GPUError.invalid("Synthetic S3 input rows must be distinct") }
            let x = try MX.array(data: inputBytes, shape: [1,3,2560], dtype: MLX_BFLOAT16)
            try x.eval()
            report["input"] = ["shape":x.shape,"dtype":"BF16","bytes":inputBytes.count,
                "sha256":GPUMatvecCase.sha256(inputBytes),"seed_hex":String(seed,radix:16),
                "kind":"synthetic","row_bf16_sha256":rowHashes,"formula":"Existing GPUMatvecCase.syntheticBF16 LCG, three consecutive distinct rows. Same independent x reused for both layer banks; not a captured normalized activation."]
            let orders = [["A","B","C"],["C","B","A"],["B","C","A"],
                          ["A","C","B"],["C","A","B"],["B","A","C"]]
            report["measurement_triplet_order"] = (0..<12).map { orders[$0 % orders.count] }
            // A scoped function releases one resident MoE before loading the next.
            // The allocator may retain freed cache pages; that is reported, not assumed absent.
            func runLayer(_ layer: Int) throws {
                let start = DispatchTime.now().uptimeNanoseconds
                let weights = try GPUWeights(modelDirectory: modelURL)
                let moe = try GPUMoE(weights: weights, layer: layer, hiddenSize: config.hiddenSize,
                    experts: config.expertCount, topK: config.expertsPerToken,
                    groupSize: config.quantizationGroupSize, bits: config.quantizationBits,
                    prefillAccumulation: .reference)
                let linear = try GPUVerificationLinear()
                try MX.synchronize()
                let prefix = "language_model.model.layers.\(layer).mlp."
                guard !weights.ledger.isEmpty, weights.ledger.allSatisfy({ $0.name.hasPrefix(prefix) }) else {
                    throw GPUError.invalid("Grouped probe loaded outside its selected MoE layer")
                }
                let entries = try weights.ledger.map { entry -> [String: Any] in
                    let metadata = try weights.metadata(entry.name)
                    return ["name":entry.name,"shard":metadata.shard,"shape":metadata.shape,
                            "dtype":metadata.dtypeName,"source_bytes":metadata.byteCount]
                }
                loads.append(["layer":layer,"load_milliseconds":GPUProbeSupport.elapsed(start),
                              "memory":try MX.memory(),"source_tensors":entries])
                for item in frozen.cases.filter({ $0.layer == layer }) {
                    let flattenedIDs = item.expertIDs.flatMap { $0 }.map(Int32.init)
                    let ids = try MX.cast(MX.array(flattenedIDs, shape:[1,3,10]),MLX_UINT32)
                    try ids.eval()
                    func mode(_ name: String) -> GPUMoE.GroupedGateUpProbeMode {
                        switch name { case "A": return .scalarLoop; case "B": return .tokenAxis; default: return .groupedGateUp }
                    }
                    func forward(_ name: String, diagnostic: Bool) throws -> GPUMoEOutput {
                        try moe.probeGroupedGateUp(x,indices:ids,mode:mode(name),linear:linear,diagnostics:diagnostic)
                    }
                    func diagnostic(_ name: String) throws -> GPUMoEOutput {
                        let value = try forward(name,diagnostic:true)
                        try MX.eval(Array(value.diagnostics.values) + [value.y]); return value
                    }
                    let a = try diagnostic("A"), b = try diagnostic("B"), c = try diagnostic("C")
                    var at = a.diagnostics, bt = b.diagnostics, ct = c.diagnostics
                    at["y"] = a.y; bt["y"] = b.y; ct["y"] = c.y
                    guard let membership = ct.removeValue(forKey:"group_membership"),
                          Set(at.keys) == Set(bt.keys), Set(at.keys) == Set(ct.keys),
                          at["verification_activation"]?.shape == [1,3,10,640],
                          at["unmodified_router_experts"] != nil else {
                        throw GPUError.invalid("Missing/different grouped verification diagnostics")
                    }
                    var checks: [String:Any] = [:], exact = true
                    func compare(_ name: String,_ lhs: Tensor,_ rhs: Tensor) throws {
                        let value = try GPUMoEVerificationBytes.compare(lhs,rhs)
                        checks[name] = value; exact = exact && (value["exact"] as? Bool == true)
                    }
                    for name in at.keys.sorted() {
                        try compare("A_B.\(name)",at[name]!,bt[name]!)
                        try compare("A_C.\(name)",at[name]!,ct[name]!)
                    }
                    let expectedPlan = try MX.array(item.membership(),shape:[30,3])
                    try compare("C_membership_cpu_oracle",membership,expectedPlan)
                    try compare("forced_IDs",at["selected_experts"]!,ids)
                    var ordinary: [String:Tensor] = [:]
                    for name in ["A","B","C"] {
                        let value = try forward(name,diagnostic:false); try value.y.eval(); ordinary[name] = value.y
                        let reference = name == "A" ? a.y : (name == "B" ? b.y : c.y)
                        try compare("\(name)_diagnostic_vs_ordinary",reference,value.y)
                    }
                    try compare("A_B_ordinary",ordinary["A"]!,ordinary["B"]!)
                    try compare("A_C_ordinary",ordinary["A"]!,ordinary["C"]!)
                    var row: [String:Any] = ["name":item.name,"layer":layer,"source_kind":item.sourceKind,
                        "capture_record_index":item.recordIndex,"capture_repetition":item.repetition,
                        "capture_position":item.position,"expert_ids":item.expertIDs,
                        "unique_experts":item.uniqueExperts,"n1":item.n1,"n2":item.n2,"n3":item.n3,
                        "routed_weight_logical_bytes":82_944_000,
                        "gate_up_logical_bytes_avoided":(30-item.uniqueExperts)*1_843_200,
                        "membership_logical_bytes":360,"bitwise_passed":exact,
                        "comparisons":checks,"comparison_count":checks.count]
                    guard exact else { cases.append(row); throw GPUError.invalid("Grouped gate/up numerical gate failed for \(item.name); no timing") }
                    var lastTimed: [String:Tensor] = [:]
                    func timed(_ name: String) throws -> [String:Any] {
                        let begin = DispatchTime.now().uptimeNanoseconds
                        let value = try forward(name,diagnostic:false)
                        let forwardEnd = DispatchTime.now().uptimeNanoseconds
                        try value.y.eval()
                        let end = DispatchTime.now().uptimeNanoseconds
                        let ms = Double(end-begin)*1e-6
                        lastTimed[name] = value.y // Retention happens outside the measured interval.
                        guard end > begin, ms.isFinite else { throw GPUError.invalid("Invalid grouped probe timing") }
                        return ["variant":name,"start_ns":begin,"forward_end_ns":forwardEnd,
                                "evaluation_end_ns":end,"milliseconds":ms]
                    }
                    var warm: [[String:Any]] = [], samples: [[String:Any]] = []
                    for round in 0..<3 {
                        for name in orders[round] {
                            var sample = try timed(name); sample["round"] = round; warm.append(sample)
                        }
                    }
                    for round in 0..<12 {
                        for (position,name) in orders[round % orders.count].enumerated() {
                            var sample = try timed(name); sample["round"] = round; sample["position"] = position
                            samples.append(sample)
                        }
                    }
                    func summarize(_ records: [[String:Any]]) -> [String:Any] {
                        let medians = Dictionary(uniqueKeysWithValues: ["A","B","C"].map { name in
                            let values = records.filter { $0["variant"] as? String == name }.map { $0["milliseconds"] as! Double }
                            return (name,GPUProbeSupport.percentile(values,0.5))
                        })
                        return ["median_milliseconds":medians,"A_over_C":medians["A"]!/medians["C"]!,
                                "B_over_C":medians["B"]!/medians["C"]!,"A_over_B":medians["A"]!/medians["B"]!]
                    }
                    row["warmup_samples"] = warm; row["interleaved_samples"] = samples
                    row["all_samples_summary"] = summarize(samples)
                    row["six_round_blocks"] = (0..<2).map { block -> [String:Any] in
                        let selected = samples.filter { ($0["round"] as! Int) / 6 == block }
                        var result = summarize(selected); result["block"] = block; result["round_start"] = block*6
                        result["round_end_inclusive"] = block*6+5; return result
                    }
                    guard warm.count == 9, samples.count == 36, lastTimed.count == 3 else {
                        throw GPUError.invalid("Incomplete grouped probe samples")
                    }
                    // Read the final outputs produced by the ACTUAL timed ordinary
                    // path, outside its timing, to catch cache/config/lifetime effects.
                    for name in ["A","B","C"] {
                        try compare("\(name)_final_timed_vs_reference",a.y,lastTimed[name]!)
                    }
                    row["comparisons"] = checks; row["comparison_count"] = checks.count
                    row["final_timed_comparison_count"] = 3; row["bitwise_passed"] = exact
                    cases.append(row); try save()
                    guard exact else { throw GPUError.invalid("Final timed output gate failed for \(item.name)") }
                }
                try MX.synchronize()
            }
            for layer in [0,45] { try runLayer(layer) }
            report["final_memory"] = try MX.memory()
            report["complete"] = true; report["passed"] = cases.count == 6
            try save()
        } catch {
            report["fatal_error"] = String(describing:error); report["passed"] = false
            try save(); throw error
        }
    }
}

private struct GroupedGateUpRouteCases: Decodable {
    struct Source: Decodable {
        struct Analysis: Decodable { let path: String; let sha256: String }
        let path: String; let sha256: String; let analysis: Analysis
    }
    struct Case: Decodable {
        let name: String; let sourceKind: String; let recordIndex: Int
        let repetition: Int; let layer: Int; let position: Int; let tokenCount: Int
        let expertIDs: [[Int]]; let n1: Int; let n2: Int; let n3: Int; let uniqueExperts: Int
        func membership() -> [Int32] {
            let ids = expertIDs.flatMap { $0 }
            var result = [Int32](repeating:-1,count:90)
            for owner in 0..<ids.count where ids.firstIndex(of:ids[owner]) == owner {
                for token in 0..<3 {
                    if let slot = expertIDs[token].firstIndex(of:ids[owner]) { result[owner*3+token] = Int32(slot) }
                }
            }
            return result
        }
    }
    let schema: String; let source: Source; let cases: [Case]
}
