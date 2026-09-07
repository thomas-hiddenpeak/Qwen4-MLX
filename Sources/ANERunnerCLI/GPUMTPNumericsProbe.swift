import ANERunnerGPU
import CMLX
import Foundation

extension RunnerCLI {
    /// Compare target computations from the SAME evaluated AR checkpoint.
    /// Fixed future inputs isolate causality from draft quality and earlier drift.
    static func probeGPUMTPNumerics(_ args: Arguments) throws {
        if args["--scenario"] == "decode-async-one-step" {
            return try probeGPUDecodeAsyncState(args)
        }
        try args.validate(["--model-dir", "--reference-report", "--positions", "--output", "--verify-scalar-linear"])
        guard [nil, "true", "false"].contains(args["--verify-scalar-linear"]) else {
            throw CLIError.usage("--verify-scalar-linear expects true or false")
        }
        let scalarLinear = args["--verify-scalar-linear"] == "true"
        let directory = URL(fileURLWithPath: try args.require("--model-dir"))
        let destination = try args.require("--output")
        guard !FileManager.default.fileExists(atPath: destination),
              let root = try JSONSerialization.jsonObject(with: Data(contentsOf: URL(fileURLWithPath: args.require("--reference-report")))) as? [String: Any],
              let trial = (root["trials"] as? [[String: Any]])?.first,
              let input = trial["prompt_tokens"] as? [Int],
              let output = trial["generated_token_ids"] as? [Int] else {
            throw CLIError.usage("Require a reference generation report and a new output file")
        }
        let prompt = try input.map { value -> Int32 in
            guard let n = Int32(exactly: value) else { throw CLIError.usage("Invalid prompt token") }; return n
        }
        let golden = try output.map { value -> Int32 in
            guard let n = Int32(exactly: value) else { throw CLIError.usage("Invalid output token") }; return n
        }
        let positions = try Set((args["--positions"] ?? "0,28,55,56").split(separator: ",").map { part -> Int in
            guard let n = Int(part), n >= 0, n < golden.count else { throw CLIError.usage("Invalid output position") }; return n
        })
        guard !prompt.isEmpty, let last = positions.max(), prompt.count + last + 3 < 262144 else {
            throw CLIError.usage("Invalid diagnostic context")
        }
        var previous = 0
        try MX.check(mlx_set_cache_limit(&previous, 256 * 1024 * 1024), "diagnostic allocation cache")
        let model = try QwenModel(modelDirectory: directory) { i, _ in
            if i % 8 == 0 { FileHandle.standardError.write(Data("Numerics: loaded \(i)/48\n".utf8)) }
        }
        let tokenizer = try QwenTokenizer(modelDirectory: directory)
        var state = model.makeState(), offset = 0
        let prefetch = try model.makePrefillPrefetch(tokens: prompt, chunk: 416, state: state)
        defer { prefetch.finish() }
        while offset < prompt.count - 1 {
            let end = min(prompt.count - 1, offset + 416)
            let out = try model.forward(tokens: Array(prompt[offset..<end]), state: &state, prefillPrefetch: prefetch)
            try model.evaluate([out.stream], state: &state)
            offset = end
        }
        var reports: [[String: Any]] = []
        func firstRow(_ x: Tensor) throws -> Tensor {
            guard x.shape.count >= 3, x.shape[0] == 1 else {
                throw GPUError.invalid("Expected [1,S,...] diagnostic tensor, got \(x.shape)")
            }
            var ends = x.shape
            ends[1] = 1
            return try MX.slice(x, starts: Array(repeating: 0, count: ends.count), ends: ends)
        }
        func top(_ values: [Float]) -> [[String: Any]] {
            var indices: [Int] = []
            for i in values.indices where !tokenizer.reservedOutputTokenIDs.contains(Int32(i)) {
                let insertion = indices.firstIndex(where: { values[i] > values[$0] }) ?? indices.count
                if insertion < 5 { indices.insert(i, at: insertion); if indices.count > 5 { indices.removeLast() } }
            }
            return indices.map { ["token": $0, "logit": values[$0], "text": (try? tokenizer.decode([Int32($0)])) ?? ""] }
        }
        func compare(_ a: [Float], _ b: [Float]) throws -> [String: Any] {
            guard a.count == b.count, !a.isEmpty, a.allSatisfy(\.isFinite), b.allSatisfy(\.isFinite) else {
                throw GPUError.invalid("Nonfinite or mismatched diagnostic arrays")
            }
            var maximum = 0.0, squared = 0.0, aa = 0.0, bb = 0.0, ab = 0.0, changed = 0
            for i in a.indices {
                let x = Double(a[i]), y = Double(b[i]), delta = x - y
                maximum = max(maximum, abs(delta)); squared += delta * delta
                aa += x*x; bb += y*y; ab += x*y
                if a[i].bitPattern != b[i].bitPattern { changed += 1 }
            }
            return ["elements": a.count, "changed": changed, "max_abs": maximum,
                    "rms": sqrt(squared / Double(a.count)), "relative_l2": sqrt(squared / max(aa, 1e-30)),
                    "cosine": ab / max(sqrt(aa * bb), 1e-30)]
        }
        for position in 0...last {
            let pending = position == 0 ? prompt.last! : golden[position - 1]
            let before = try model.checkpoint(state: &state)
            let traced = positions.contains(position)
            let ar = try model.forward(tokens: [pending], state: &state, captureTrace: traced,
                                       prefillPrefetch: position == 0 ? prefetch : nil)
            guard let arLogits = ar.logits else { throw GPUError.invalid("Missing AR logits") }
            let selected = try model.greedyToken(arLogits)
            try model.evaluate([selected, ar.stream] + Array(ar.trace.values), state: &state)
            let actual = try selected.ints()[0]
            guard actual == golden[position] else {
                throw GPUError.invalid("AR replay itself diverged at \(position): \(actual) vs \(golden[position])")
            }
            guard traced else { continue }
            FileHandle.standardError.write(Data("Numerics: comparing generated index \(position), context \(before.offset)\n".utf8))
            let arValues = try arLogits.floats()
            var batchRows: [[String: Any]] = []
            var referenceBatchValues: [Float]?
            var referenceBatchTrace: [String: [Float]] = [:]
            for future in [golden[position], golden[position] == 42 ? Int32(43) : Int32(42)] {
                var candidate = before
                let out = try model.forward(tokens: [pending, future], state: &candidate, lastLogitOnly: false,
                    captureTrace: true, verifyScalarBoundaries: true, captureVerification: true, verifyScalarMoE: true,
                    verifyScalarLinear: scalarLinear)
                guard let logits = out.logits else { throw GPUError.invalid("Missing batch logits") }
                let selectedBatch = try model.greedyToken(logits)
                try model.evaluate([selectedBatch, out.stream] + Array(out.trace.values), state: &candidate)
                candidate = try model.commitVerificationPrefix(candidate, from: before, tokens: [pending, future], count: 1)
                let batchValues = try firstRow(logits).floats()
                var row: [String: Any] = ["future_token": future, "selected_token": try selectedBatch.ints()[0],
                    "logits_vs_scalar": try compare(arValues, batchValues), "top5": top(batchValues)]
                if let referenceBatchValues { row["logits_future_invariance"] = try compare(referenceBatchValues, batchValues) }
                var layers: [[String: Any]] = []
                for name in ar.trace.keys.sorted() {
                    guard let b = out.trace[name] else { throw GPUError.invalid("Missing batch trace") }
                    let a = try firstRow(ar.trace[name]!).floats(), values = try firstRow(b).floats()
                    var layer: [String: Any] = ["name": name, "vs_scalar": try compare(a, values)]
                    if let ref = referenceBatchTrace[name] { layer["future_invariance"] = try compare(ref, values) }
                    else { referenceBatchTrace[name] = values }
                    layers.append(layer)
                }
                row["layer_trace"] = layers
                // State comparisons read all retained buffers only in diagnostics.
                // At long context this is deliberately not a timing benchmark.
                var states: [[String: Any]] = []
                for (name, a) in state.namedTensors.sorted(by: { $0.key < $1.key }) {
                    guard let b = candidate.namedTensors[name], a.shape == b.shape else {
                        throw GPUError.invalid("Committed state shape mismatch: \(name)")
                    }
                    // Full KV prefixes are unchanged handles/values except the
                    // new row; inspect only that row to bound CPU readback.
                    let av: Tensor, bv: Tensor
                    if name.hasSuffix(".keys") || name.hasSuffix(".values") {
                        av = try MX.slice(a, starts: [0,0,state.offset-1,0], ends: [1,2,state.offset,256])
                        bv = try MX.slice(b, starts: [0,0,state.offset-1,0], ends: [1,2,state.offset,256])
                    } else if name.hasSuffix(".raw_index") {
                        av = try MX.slice(a, starts: [0,state.offset-1,0], ends: [1,state.offset,128])
                        bv = try MX.slice(b, starts: [0,state.offset-1,0], ends: [1,state.offset,128])
                    } else { av = a; bv = b }
                    states.append(["name": name, "vs_scalar": try compare(av.floats(), bv.floats())])
                }
                row["state_after_one_input"] = states
                batchRows.append(row)
                if referenceBatchValues == nil { referenceBatchValues = batchValues }
            }
            reports.append(["generated_index_zero_based": position, "context_before_input": before.offset,
                            "pending": pending, "scalar_token": actual, "scalar_top5": top(arValues), "batches": batchRows])
        }
        try emit(["schema": "same-prefix-target-numerics-v1", "reference_report": args["--reference-report"]!,
                  "prompt_token_count": prompt.count, "positions": reports, "scalar_linear": scalarLinear,
                  "method": "Same evaluated AR checkpoint; S1 vs S2; only future input changed in second S2 branch. Diagnostic readback is not performance timing.",
                  "passed_ar_replay": true], to: destination)
    }
}
