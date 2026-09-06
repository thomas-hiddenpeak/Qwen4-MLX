import ANERunnerGPU
import CMLX
import Foundation

extension RunnerCLI {
    /// Isolated recurrence experiment. No model, production MTP policy, or
    /// mutable session is created. Fixture rows are explicitly replay inputs.
    static func probeGDNReplay(_ fixture: [String: Tensor], fixturePath: String,
                              fixtureSHA256: String, manifestSHA256: String,
                              repeats: Int, output: String) throws {
        guard !FileManager.default.fileExists(atPath: output), (4...64).contains(repeats) else {
            throw CLIError.usage("GDN replay output must be a new file; repeats must be in 4...64")
        }
        func require(_ name: String, _ tail: [Int]) throws -> Tensor {
            guard let tensor = fixture[name], tensor.dtype == MLX_BFLOAT16,
                  tensor.shape == tail else {
                throw CLIError.usage("Invalid BF16 GDN replay fixture tensor: \(name)")
            }
            return tensor
        }
        guard let q = fixture["recurrence.q"], q.shape.count == 4,
              q.shape[0] == 1, q.shape[1] >= 6, q.shape[2...] == [16,128] else {
            throw CLIError.usage("GDN replay fixture needs at least six recurrence input rows")
        }
        let width = q.shape[1]
        let source = try [
            require("recurrence.q", [1,width,16,128]),
            require("recurrence.k", [1,width,16,128]),
            require("recurrence.v", [1,width,48,128]),
            require("recurrence.decay", [1,width,48]),
            require("recurrence.beta", [1,width,48]),
        ]
        let cold = try require("recurrence.stateIn", [1,48,128,128])
        let warm = try require("recurrence.stateOut", [1,48,128,128])
        let convRows = try require("expected.state.convHistory", [1,3,10240])
        try MX.eval(source + [cold,warm,convRows])
        guard try warm.floats().contains(where: { $0.isFinite && $0 != 0 }) else {
            throw CLIError.usage("Warm replay checkpoint must contain nonzero finite values")
        }
        let kernel = try GPUGatedDeltaNetFused()
        let initialMemory = try MX.memory()
        var reports: [[String: Any]] = []
        var rejectedPrefixes: [[String: Any]] = []
        var passed = true

        func slice(_ value: Tensor, _ start: Int, _ end: Int) throws -> Tensor {
            var lower = [Int](repeating: 0, count: value.shape.count)
            var upper = value.shape
            lower[1] = start; upper[1] = end
            return try MX.slice(value, starts: lower, ends: upper)
        }
        // Match GPUVerificationCopy's owned, batch-axis gather. MX.copy alone
        // may share a larger backing allocation in the pinned MLX runtime.
        func owned(_ value: Tensor) throws -> Tensor {
            try MX.take(value, MX.array([Int32(0)], shape: [1]), axis: 0)
        }
        func apply(_ inputs: [Tensor], _ state: Tensor, rounded: Bool = true) throws
            -> (y: Tensor, state: Tensor) {
            try kernel.apply(q: inputs[0], k: inputs[1], v: inputs[2],
                decay: inputs[3], beta: inputs[4], state: state, roundStateEachToken: rounded)
        }
        func compare(_ actual: Tensor, _ expected: Tensor, _ name: String) throws -> [String: Any] {
            guard actual.shape == expected.shape, actual.dtype == MLX_BFLOAT16,
                  expected.dtype == MLX_BFLOAT16 else {
                throw CLIError.usage("GDN replay comparison shape/dtype mismatch: \(name)")
            }
            let a = try actual.floats(), b = try expected.floats()
            var finite = true, mismatches = 0
            for (x,y) in zip(a,b) {
                finite = finite && x.isFinite && y.isFinite
                if x.bitPattern != y.bitPattern { mismatches += 1 }
            }
            return ["name": name, "shape": actual.shape, "elements": a.count,
                    "finite": finite, "mismatching_elements": mismatches,
                    "passed": finite && a.count == b.count && mismatches == 0]
        }
        func elapsed(_ start: UInt64) -> Double {
            Double(DispatchTime.now().uptimeNanoseconds - start) * 1e-9
        }
        func distribution(_ values: [Double]) -> [String: Any] {
            let sorted = values.sorted()
            let middle = sorted.count / 2
            let median = sorted.count % 2 == 0
                ? (sorted[middle-1] + sorted[middle]) / 2 : sorted[middle]
            return ["samples": values, "mean": values.reduce(0,+) / Double(values.count),
                    "median": median, "minimum": sorted.first!, "maximum": sorted.last!]
        }

        struct Result {
            let y, finalState: Tensor
            let committed: GPUGatedDeltaNet.State
            let verifySeconds, commitSeconds, totalSeconds: Double
        }

        for (stateName, initial, initialOffset) in [("cold_fixture",cold,0), ("warm_fixture_replay",warm,width)] {
            for sequence in 1...5 {
                let inputs = try source.map { try slice($0,0,sequence) }
                // These are real fixture convolution values cycled into a
                // bank for crop/ownership validation, not a new native trace.
                let history = try stateName == "cold_fixture"
                    ? MX.zeros([1,3,10240], MLX_BFLOAT16) : convRows
                let cycled = try MX.concat(Array(repeating: convRows, count: 2), axis: 1)
                let convBank = try MX.concat([history,slice(cycled,0,sequence)], axis: 1)
                let finalConv = try owned(slice(convBank,sequence,sequence+3))
                try MX.eval(inputs + [convBank,finalConv])

                // This is the existing state API's bounds contract; creating
                // these lazy handles alone does not launch the capture kernel.
                let guardCapture = try kernel.applyCapturing(q: inputs[0], k: inputs[1], v: inputs[2],
                    decay: inputs[3], beta: inputs[4], state: initial, roundStateEachToken: true)
                let guardState = GPUGatedDeltaNet.State(convHistory: finalConv, recurrent: guardCapture.state,
                    offset: initialOffset + sequence,
                    verificationCapture: .init(initialOffset: initialOffset,
                        recurrentStates: guardCapture.states, convInputs: convBank))
                for invalidCount in [0,sequence+1] {
                    var rejected = false
                    do { _ = try guardState.committingPrefix(count: invalidCount) }
                    catch { rejected = true }
                    guard rejected else { throw CLIError.usage("GDN prefix bounds check unexpectedly accepted invalid count") }
                    rejectedPrefixes.append(["initial_state": stateName, "sequence": sequence,
                                             "count": invalidCount, "rejected": rejected])
                }

                // Independent execution oracle: unchanged S1 recurrence.
                var oracleState = initial
                var oracleStates: [Tensor] = [], oracleOutputs: [Tensor] = []
                for index in 0..<sequence {
                    let one = try source.map { try slice($0,index,index+1) }
                    let result = try apply(one,oracleState,rounded: false)
                    try MX.eval([result.y,result.state])
                    oracleStates.append(result.state); oracleOutputs.append(result.y)
                    oracleState = result.state
                }
                let oracleY = try MX.concat(oracleOutputs, axis: 1)
                try MX.eval([oracleY])

                for count in 1...sequence {
                    let acceptedDrafts = count - 1
                    let nextInputs = try source.map { try slice($0,count,count+1) }
                    try MX.eval(nextInputs)
                    func iteration(replay: Bool) throws -> Result {
                        let start = DispatchTime.now().uptimeNanoseconds
                        let y: Tensor, final: Tensor
                        let complete: GPUGatedDeltaNet.State
                        if replay {
                            let verified = try apply(inputs,initial)
                            y = verified.y; final = verified.state
                            try MX.eval([y,final])
                            complete = .init(convHistory: finalConv, recurrent: final,
                                             offset: initialOffset + sequence)
                        } else {
                            let verified = try kernel.applyCapturing(q: inputs[0], k: inputs[1], v: inputs[2],
                                decay: inputs[3], beta: inputs[4], state: initial, roundStateEachToken: true)
                            y = verified.y; final = verified.state
                            let capture = GPUGatedDeltaNet.VerificationCapture(initialOffset: initialOffset,
                                recurrentStates: verified.states, convInputs: convBank)
                            complete = .init(convHistory: finalConv, recurrent: final,
                                             offset: initialOffset + sequence, verificationCapture: capture)
                            try MX.eval([y] + complete.tensors)
                        }
                        let verifySeconds = elapsed(start)
                        let commitStart = DispatchTime.now().uptimeNanoseconds
                        let committed: GPUGatedDeltaNet.State
                        if replay && count < sequence {
                            let prefix = try inputs.map { try slice($0,0,count) }
                            let recomputed = try apply(prefix,initial)
                            let conv = try owned(slice(convBank,count,count+3))
                            committed = .init(convHistory: conv, recurrent: recomputed.state,
                                              offset: initialOffset + count)
                            // The minimal experiment computes y as well; its
                            // cost is included even though commit only needs state.
                            try MX.eval([recomputed.y] + committed.tensors)
                        } else if replay {
                            committed = complete
                            try MX.eval(committed.tensors)
                        } else {
                            committed = try complete.committingPrefix(count: count)
                            try MX.eval(committed.tensors)
                        }
                        return Result(y: y, finalState: final, committed: committed,
                            verifySeconds: verifySeconds, commitSeconds: elapsed(commitStart),
                            totalSeconds: elapsed(start))
                    }

                    var comparisons: [[String: Any]] = []
                    let expectedConv = try slice(convBank,count,count+3)
                    let oracleNext = try apply(nextInputs,oracleStates[count-1],rounded: false)
                    try MX.eval([expectedConv,oracleNext.y,oracleNext.state])
                    func validate(_ result: Result, _ variant: String) throws {
                        guard result.committed.offset == initialOffset + count,
                              result.committed.verificationCapture == nil,
                              let recurrent = result.committed.recurrent,
                              let convolution = result.committed.convHistory else {
                            throw CLIError.usage("GDN replay commit contract failed")
                        }
                        if count == sequence {
                            guard recurrent === result.finalState, convolution === finalConv else {
                                throw CLIError.usage("Full GDN acceptance must reuse final state and convolution")
                            }
                        }
                        for (a,b,name) in [
                            (result.y,oracleY,"verify_y"),
                            (result.finalState,oracleState,"verify_final_state"),
                            (recurrent,oracleStates[count-1],"accepted_state"),
                            (convolution,expectedConv,"accepted_conv"),
                        ] {
                            comparisons.append(try compare(a,b,variant + "." + name))
                        }
                        let next = try apply(nextInputs,recurrent,rounded: false)
                        try MX.eval([next.y,next.state])
                        comparisons.append(try compare(next.y,oracleNext.y,variant + ".next_y"))
                        comparisons.append(try compare(next.state,oracleNext.state,variant + ".next_state"))
                    }
                    // Correctness is checked before spending time on a bad candidate.
                    try validate(iteration(replay: false),"capture")
                    try validate(iteration(replay: true),"replay")
                    let initialPassed = comparisons.allSatisfy { $0["passed"] as? Bool == true }
                    guard initialPassed else {
                        try emit(["schema": "qwen4-gdn-replay-probe-v1", "passed": false,
                                  "initial_state": stateName, "sequence": sequence,
                                  "accepted_drafts": acceptedDrafts, "comparisons": comparisons], to: output)
                        throw CLIError.usage("GDN replay numerical gate failed before timing")
                    }
                    var verify = [[Double](),[Double]()], commit = [[Double](),[Double]()]
                    var total = [[Double](),[Double]()], warmup: [[String: Any]] = []
                    var last: [Result?] = [nil,nil]
                    for index in 0..<(repeats+4) {
                        for variant in index % 2 == 0 ? [0,1] : [1,0] {
                            let result = try iteration(replay: variant == 1)
                            if index < 4 {
                                warmup.append(["iteration": index, "variant": variant == 0 ? "capture" : "replay",
                                               "total_seconds": result.totalSeconds])
                            } else {
                                verify[variant].append(result.verifySeconds)
                                commit[variant].append(result.commitSeconds)
                                total[variant].append(result.totalSeconds)
                            }
                            last[variant] = result
                        }
                    }
                    try validate(last[0]!,"capture_last_timed")
                    try validate(last[1]!,"replay_last_timed")
                    let casePassed = comparisons.allSatisfy { $0["passed"] as? Bool == true }
                    passed = passed && casePassed
                    let inputBytes = inputs.reduce(0) { $0 + $1.nbytes }
                    let stateBytes = initial.nbytes
                    reports.append([
                        "initial_state": stateName, "sequence": sequence, "accepted_drafts": acceptedDrafts,
                        "committed_rows": count, "full_acceptance": count == sequence, "passed": casePassed,
                        "comparisons": comparisons, "warmups": warmup,
                        "capture": ["verify_seconds": distribution(verify[0]),
                                    "commit_seconds": distribution(commit[0]), "total_seconds": distribution(total[0])],
                        "replay": ["verify_seconds": distribution(verify[1]),
                                   "commit_seconds": distribution(commit[1]), "total_seconds": distribution(total[1])],
                        "logical_bytes": [
                            "capture_bank": sequence * stateBytes, "retained_replay_inputs_qkv_g_beta": inputBytes,
                            "bank_minus_replay_inputs": sequence * stateBytes - inputBytes,
                            "common_initial_state": stateBytes, "common_final_state": stateBytes,
                            "common_conv_bank": convBank.nbytes, "common_final_conv": finalConv.nbytes,
                            "partial_commit_recurrent_copy_read": count < sequence ? stateBytes : 0,
                            "partial_commit_recurrent_copy_write": count < sequence ? stateBytes : 0,
                            "partial_replay_state_read": count < sequence ? stateBytes : 0,
                            "partial_replay_state_write": count < sequence ? stateBytes : 0,
                            "unused_replay_y_output": count < sequence ? count * 48 * 128 * 2 : 0,
                        ],
                    ])
                }
            }
        }
        try emit([
            "schema": "qwen4-gdn-replay-probe-v1", "passed": passed,
            "fixture": fixturePath, "fixture_sha256": fixtureSHA256, "manifest_sha256": manifestSHA256,
            "repeats_per_variant": repeats, "warmups_per_variant": 4,
            "invalid_prefix_checks": rejectedPrefixes,
            "model_loaded": false, "production_mtp_changed": false,
            "initial_memory": initialMemory, "final_memory": try MX.memory(), "cases": reports,
            "notes": [
                "Existing layer0 recurrence fixture replays captured activations at another boundary; it is not a native full-generation recurrence trace.",
                "Warm state is the fixture's final recurrent state reused with its first rows. Convolution values are cycled from its recorded final history for crop validation.",
                "All timings include new Swift graph construction, GPU work, and evaluation wait. CPU comparisons and input/conv preparation are outside timing.",
                "Capture verify includes the extra state bank; replay verify uses the existing rounded two-output kernel. Partial replay replaces the recurrent copy and still computes unused y.",
                "Replay retains references to already materialized recurrence inputs. Logical payload differences are not allocator peak deltas or physical DRAM traffic.",
                "Current probe memory snapshots include shared fixtures and cached allocations from both variants; they do not establish per-variant peak memory.",
            ],
        ], to: output)
        if !passed { throw CLIError.usage("GDN replay numerical gate failed; see saved report") }
    }
}
