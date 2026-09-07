// GPU MoE math follows the public-MLX reference in scripts/benchmark_moe_mlx.py
// and mlx-serve/src/transformer.zig (moeMLP2, moeRoutingChain, sorted experts),
// upstream garnermccloud/mlx-serve commit 7dbcba04c98e4fd3bcc533c63e645547f13cc3b1.
// The affine decode gate/up and down/reduction kernels below are adapted from
// the upstream public Metal sources, preserving its BF16 rounding boundaries.
//
// Upstream attribution and permission:
// MIT License
// Copyright (c) 2026 David Dalcu
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

import CMLX
import Foundation

/// Public MLX Metal kernels for original affine-Q4 decode and a precise
/// prefill reduction. No weights are requantized or copied to host memory.
final class GPUMoEFused {
    struct Projection {
        let weight: Tensor
        let scales: Tensor
        let biases: Tensor
    }
    struct DecodeResult {
        let routed: Tensor
        let expertOutputs: Tensor?
        let activation: Tensor?
        let membership: Tensor?
    }
    struct SharedOutput {
        let y: Tensor
        let gate: Tensor?
        let gated: Tensor?
    }
    private let hidden: Int
    private let intermediate: Int
    private let topK: Int
    private let sigmoidTable: Tensor
    var prefillSigmoidTable: Tensor { sigmoidTable }
    private let hiddenScalar: Tensor
    private let intermediateScalar: Tensor
    private let gateUp: Program
    private let downReduce: Program
    private let downReduceDiagnostic: Program
    private var prefillPrograms: [[Int]: Program] = [:]
    private var sharedActivationPrograms: [Int: Program] = [:]
    private var sharedOutputPrograms: [Bool: Program] = [:]
    // Separate S2/S3 caches leave the original S1 programs and keys unchanged.
    private var verificationSharedActivationPrograms: [Int: Program] = [:]
    private var verificationSharedOutputPrograms: [VerificationKey: Program] = [:]
    private struct VerificationKey: Hashable {
        let sequence: Int
        let diagnostics: Bool
    }
    private struct VerificationPrograms {
        let gateUp: Program
        let downReduce: Program
    }
    private var verificationPrograms: [VerificationKey: VerificationPrograms] = [:]
    private var groupedVerificationPrograms: (plan: Program, gateUp: Program)?

    init(hidden: Int, intermediate: Int, topK: Int, groupSize: Int, bits: Int) throws {
        guard hidden > 0, hidden % 4 == 0, intermediate > 0, intermediate % 8 == 0,
              (1...32).contains(topK), groupSize == 64, bits == 4 else {
            throw GPUError.invalid("Unsupported fused affine-Q4 decode geometry")
        }
        self.hidden = hidden; self.intermediate = intermediate; self.topK = topK
        hiddenScalar = try MX.array([Int32(hidden)], shape: [])
        intermediateScalar = try MX.array([Int32(intermediate)], shape: [])
        // Same construction as upstream swigluSigTable: each possible 16-bit
        // input is widened exactly, then public MLX sigmoid produces the LUT.
        let allBits = (0..<65536).map { Float(bitPattern: UInt32($0) << 16) }
        sigmoidTable = try MX.sigmoid(MX.array(allBits, shape: [65536], dtype: MLX_BFLOAT16))
        try sigmoidTable.eval()
        let template = ["GS": groupSize, "BITS": bits]
        gateUp = try Program(
            name: "ane_runner_q4_gateup", source: Self.gateUpSource,
            inputs: ["x", "wg_q", "g_scales", "g_biases", "wu_q", "u_scales", "u_biases", "inds", "sigtab", "K_size", "N_size"],
            outputs: ["y"], outputShapes: [[topK, intermediate]],
            grid: [32, intermediate, topK], threadgroup: [32, 8, 1], template: template)
        var downTemplate = template
        downTemplate["TOPK"] = topK; downTemplate["ROWS"] = 4
        let downInputs = ["x", "w_q", "scales", "biases", "inds", "scores", "K_size", "N_size"]
        downReduce = try Program(
            name: "ane_runner_q4_down_reduce", source: Self.downReduceSource,
            inputs: downInputs, outputs: ["y"], outputShapes: [[hidden]],
            grid: [hidden / 4 * topK * 32, 1, 1], threadgroup: [topK * 32, 1, 1], template: downTemplate)
        // The diagnostic variant adds only a store of the same rounded T(acc)
        // used by the reduction. The timed path does not materialize this bank.
        let diagnosticSource = Self.downReduceSource.replacingOccurrences(
            of: "slot_vals[slot * uint(ROWS) + r] = T(acc);",
            with: "slot_vals[slot * uint(ROWS) + r] = T(acc);\n    expert_values[(size_t)slot * (size_t)N + (size_t)n] = T(acc);")
        downReduceDiagnostic = try Program(
            name: "ane_runner_q4_down_reduce_diagnostic", source: diagnosticSource,
            inputs: downInputs, outputs: ["y", "expert_values"], outputShapes: [[hidden], [topK, hidden]],
            grid: [hidden / 4 * topK * 32, 1, 1], threadgroup: [topK * 32, 1, 1], template: downTemplate)
    }

    func decode(_ x: Tensor, indices: Tensor, scores: Tensor,
                gate: Projection, up: Projection, down: Projection, diagnostics: Bool,
                activationDiagnostics: Bool = false) throws -> DecodeResult {
        guard x.count == hidden, x.dtype == MLX_BFLOAT16,
              indices.count == topK, scores.count == topK, scores.dtype == MLX_BFLOAT16 else {
            throw GPUError.invalid("Fused decode input/dtype mismatch")
        }
        let flatX = try MX.reshape(x, [hidden])
        let flatIDs = try MX.cast(MX.reshape(indices, [topK]), MLX_UINT32)
        let flatScores = try MX.reshape(scores, [topK])
        let activation = try gateUp.apply([flatX, gate.weight, gate.scales, gate.biases,
                                          up.weight, up.scales, up.biases, flatIDs,
                                          sigmoidTable, hiddenScalar, intermediateScalar])[0]
        let program = diagnostics ? downReduceDiagnostic : downReduce
        let values = try program.apply([activation, down.weight, down.scales, down.biases,
                                        flatIDs, flatScores, intermediateScalar, hiddenScalar])
        return DecodeResult(routed: try MX.reshape(values[0], [1, 1, hidden]),
                            expertOutputs: diagnostics ? try MX.reshape(values[1], [1, 1, topK, hidden]) : nil,
                            activation: activationDiagnostics ? try MX.reshape(activation, [1, 1, topK, intermediate]) : nil,
                            membership: nil)
    }

    /// Experimental recipe only. The ordinary decode above stays unchanged.
    /// Return the original single-output down primitive before its reshape so
    /// the native pair can own its input dependencies and reuse its exact kernel.
    func downPairRecipe(_ x: Tensor, indices: Tensor, scores: Tensor,
                        gate: Projection, up: Projection, down: Projection) throws -> Tensor {
        guard hidden == 2560, intermediate == 640, topK == 10,
              x.shape == [1,1,2560], x.dtype == MLX_BFLOAT16,
              indices.shape == [1,1,10], indices.dtype == MLX_UINT32,
              scores.shape == [1,1,10], scores.dtype == MLX_BFLOAT16 else {
            throw GPUError.invalid("Down-pair recipe requires the original S1 H2560/top10 affine-Q4 geometry")
        }
        let flatX = try MX.reshape(x, [hidden])
        let flatIDs = try MX.cast(MX.reshape(indices, [topK]), MLX_UINT32)
        let flatScores = try MX.reshape(scores, [topK])
        let activation = try gateUp.apply([flatX, gate.weight, gate.scales, gate.biases,
                                          up.weight, up.scales, up.biases, flatIDs,
                                          sigmoidTable, hiddenScalar, intermediateScalar])[0]
        return try downReduce.apply([activation, down.weight, down.scales, down.biases,
                                     flatIDs, flatScores, intermediateScalar, hiddenScalar])[0]
    }

    /// Two primary dispatches for S2...5 verification. Each (token, selected
    /// expert) retains the original S1 arithmetic; this adds scheduling width,
    /// not cross-token weight reuse. IDs must be valid router outputs in
    /// 0..<expertCount, as required by decode; no host readback is introduced.
    /// Program's ensureRowContiguous setting makes the flat offsets below valid.
    func verifyTokens(_ x: Tensor, indices: Tensor, scores: Tensor,
                      gate: Projection, up: Projection, down: Projection,
                      diagnostics: Bool, groupedGateUp: Bool = false,
                      activationDiagnostics: Bool = false) throws -> DecodeResult {
        guard x.shape.count == 3, x.shape[0] == 1,
              (2...5).contains(x.shape[1]), x.shape[2] == hidden,
              x.dtype == MLX_BFLOAT16,
              indices.shape == [1, x.shape[1], topK], indices.dtype == MLX_UINT32,
              scores.shape == indices.shape, scores.dtype == MLX_BFLOAT16,
              hidden % 64 == 0, intermediate % 64 == 0,
              hidden <= Int(Int32.max) / (topK * 8),
              intermediate <= Int(Int32.max), gate.weight.shape.count == 3 else {
            throw GPUError.invalid("Verification experts require BF16 [1,S,H], U32 [1,S,topK], S2...5 and affine Q4/group64 geometry")
        }
        let sequence = x.shape[1], experts = gate.weight.shape[0]
        func valid(_ projection: Projection, output: Int, input: Int) -> Bool {
            projection.weight.shape == [experts, output, input / 8] &&
            projection.weight.dtype == MLX_UINT32 &&
            projection.scales.shape == [experts, output, input / 64] &&
            projection.biases.shape == projection.scales.shape &&
            projection.scales.dtype == MLX_BFLOAT16 && projection.biases.dtype == MLX_BFLOAT16
        }
        guard experts >= topK,
              valid(gate, output: intermediate, input: hidden),
              valid(up, output: intermediate, input: hidden),
              valid(down, output: hidden, input: intermediate) else {
            throw GPUError.invalid("Verification expert packed weights/scales/biases do not match the affine Q4 bank")
        }
        guard !groupedGateUp || (sequence == 3 && hidden == 2560 && intermediate == 640 &&
                                  topK == 10 && experts == 512) else {
            throw GPUError.invalid("Grouped gate/up probe requires the fixed S3 H2560/I640/top10/E512 bank")
        }
        let key = VerificationKey(sequence: sequence, diagnostics: diagnostics)
        let programs: VerificationPrograms
        if let existing = verificationPrograms[key] { programs = existing }
        else {
            // Adapt only indexing from the scalar sources. A source edit that
            // removes/duplicates an anchor fails explicitly instead of silently
            // generating a partly converted kernel with cross-token accesses.
            let gateSource = try Self.reindex(Self.gateUpSource, [
                ("uint e = thread_position_in_grid.z;      // top-K slot",
                 "uint e = thread_position_in_grid.z;      // flattened token * TOPK + slot\n    uint token = e / uint(TOPK);"),
                ("size_t xi = (size_t)(k_base + ki);",
                 "size_t xi = (size_t)token * (size_t)K + (size_t)(k_base + ki);")
            ])
            let tokenGateUp = try Program(name: "ane_runner_q4_gateup_verify_s\(sequence)",
                source: gateSource,
                inputs: ["x", "wg_q", "g_scales", "g_biases", "wu_q", "u_scales", "u_biases", "inds", "sigtab", "K_size", "N_size"],
                outputs: ["y"], outputShapes: [[sequence, topK, intermediate]],
                grid: [32, intermediate, sequence * topK], threadgroup: [32, 8, 1],
                template: ["GS": 64, "BITS": 4, "TOPK": topK])
            var downSource = try Self.reindex(Self.downReduceSource, [
                ("uint tile = threadgroup_position_in_grid.x;   // block of ROWS output rows",
                 "uint tile = threadgroup_position_in_grid.x;   // block of ROWS output rows\n    uint token = threadgroup_position_in_grid.y;"),
                ("uint eid = inds[slot];",
                 "uint eid = inds[(size_t)token * (size_t)TOPK + (size_t)slot];"),
                ("size_t xoff = (size_t)slot * (size_t)K;",
                 "size_t xoff = ((size_t)token * (size_t)TOPK + (size_t)slot) * (size_t)K;"),
                ("scores[s2]", "scores[(size_t)token * (size_t)TOPK + (size_t)s2]"),
                ("y[(size_t)tile * (size_t)ROWS + (size_t)lane] = total;",
                 "y[(size_t)token * (size_t)N + (size_t)tile * (size_t)ROWS + (size_t)lane] = total;")
            ])
            if diagnostics {
                downSource = try Self.reindex(downSource, [
                    ("slot_vals[slot * uint(ROWS) + r] = T(acc);",
                     "slot_vals[slot * uint(ROWS) + r] = T(acc);\n    expert_values[((size_t)token * (size_t)TOPK + (size_t)slot) * (size_t)N + (size_t)n] = T(acc);")
                ])
            }
            let tokenDown = try Program(
                name: "ane_runner_q4_down_reduce_verify_s\(sequence)" + (diagnostics ? "_diagnostic" : ""),
                source: downSource,
                inputs: ["x", "w_q", "scales", "biases", "inds", "scores", "K_size", "N_size"],
                outputs: diagnostics ? ["y", "expert_values"] : ["y"],
                outputShapes: diagnostics ? [[sequence, hidden], [sequence, topK, hidden]] : [[sequence, hidden]],
                grid: [hidden / 4 * topK * 32, sequence, 1], threadgroup: [topK * 32, 1, 1],
                template: ["GS": 64, "BITS": 4, "TOPK": topK, "ROWS": 4])
            programs = VerificationPrograms(gateUp: tokenGateUp, downReduce: tokenDown)
            verificationPrograms[key] = programs
        }
        let flatX = try MX.reshape(x, [sequence, hidden])
        let flatIDs = try MX.reshape(indices, [sequence, topK])
        let flatScores = try MX.reshape(scores, [sequence, topK])
        let activation: Tensor
        var membership: Tensor?
        if groupedGateUp {
            let grouped: (plan: Program, gateUp: Program)
            if let cached = groupedVerificationPrograms { grouped = cached }
            else {
                let plan = try Program(name: "ane_runner_q4_gateup_s3_membership",
                    source: Self.groupedMembershipSource, inputs: ["inds"], outputs: ["members"],
                    outputShapes: [[30,3]], grid: [32,1,1], threadgroup: [32,1,1],
                    template: [:], outputDType: MLX_INT32)
                let gateUp = try Program(name: "ane_runner_q4_gateup_s3_grouped",
                    source: Self.groupedGateUpSource,
                    inputs: ["x", "wg_q", "g_scales", "g_biases", "wu_q", "u_scales", "u_biases",
                             "inds", "members", "sigtab", "K_size", "N_size"],
                    outputs: ["y"], outputShapes: [[3,topK,intermediate]],
                    grid: [32,intermediate,30], threadgroup: [32,8,1],
                    template: ["GS":64,"BITS":4,"TOPK":topK])
                grouped = (plan,gateUp); groupedVerificationPrograms = grouped
            }
            // The plan is a real lazy dependency. No host U/readback or weight copy.
            let plan = try grouped.plan.apply([flatIDs])[0]
            activation = try grouped.gateUp.apply([flatX,gate.weight,gate.scales,gate.biases,
                up.weight,up.scales,up.biases,flatIDs,plan,sigmoidTable,hiddenScalar,intermediateScalar])[0]
            if activationDiagnostics { membership = plan }
        } else {
            activation = try programs.gateUp.apply([flatX, gate.weight, gate.scales, gate.biases,
                up.weight, up.scales, up.biases, flatIDs, sigmoidTable, hiddenScalar, intermediateScalar])[0]
        }
        let values = try programs.downReduce.apply([activation, down.weight, down.scales, down.biases,
            flatIDs, flatScores, intermediateScalar, hiddenScalar])
        return DecodeResult(routed: try MX.reshape(values[0], [1, sequence, hidden]),
            expertOutputs: diagnostics ? try MX.reshape(values[1], [1, sequence, topK, hidden]) : nil,
            activation: activationDiagnostics ? try MX.reshape(activation, [1,sequence,topK,intermediate]) : nil,
            membership: membership)
    }

    private static func reindex(_ original: String, _ replacements: [(String, String)]) throws -> String {
        var source = original
        for (before, after) in replacements {
            guard source.components(separatedBy: before).count == 2 else {
                throw GPUError.invalid("Scalar MoE source changed; verification index adaptation needs review")
            }
            source = source.replacingOccurrences(of: before, with: after)
        }
        return source
    }

    /// Only a single token: preserve the three public MLX BF16 writes in
    /// sigmoid(gate), gate * sigmoid, and that product * up. The existing LUT
    /// supplies exactly the reference sigmoid, without another table or eval.
    func sharedActivation(_ gate: Tensor, up: Tensor) throws -> Tensor {
        let shape = gate.shape
        guard shape.count == 3, shape[0] == 1, shape[1] == 1,
              shape[2] > 0, shape[2] <= Int(Int32.max) - 255,
              up.shape == shape, gate.dtype == MLX_BFLOAT16, up.dtype == MLX_BFLOAT16 else {
            throw GPUError.invalid("Shared activation requires matching one-token BF16 tensors")
        }
        let width = shape[2]
        let program: Program
        if let cached = sharedActivationPrograms[width] { program = cached }
        else {
            program = try Program(name: "ane_runner_shared_swiglu_bf16",
                source: Self.sharedActivationSource, inputs: ["gate", "up", "sigtab"],
                outputs: ["y"], outputShapes: [shape],
                grid: [(width + 255) / 256 * 256, 1, 1], threadgroup: [256, 1, 1],
                template: ["COUNT": width])
            sharedActivationPrograms[width] = program
        }
        return try program.apply([gate, up, sigmoidTable])[0]
    }

    /// Preserve BF16 sigmoid, rounded shared product, then rounded addition.
    /// Diagnostics only add stores of those same intermediates. The timed
    /// path allocates one H-element output, with no diagnostic tensors.
    func sharedOutput(routed: Tensor, down: Tensor, gateLogits: Tensor,
                      diagnostics: Bool) throws -> SharedOutput {
        let shape = [1, 1, hidden]
        guard routed.shape == shape, down.shape == shape, gateLogits.shape == [1, 1, 1],
              routed.dtype == MLX_BFLOAT16, down.dtype == MLX_BFLOAT16,
              gateLogits.dtype == MLX_BFLOAT16 else {
            throw GPUError.invalid("Shared output requires one-token BF16 routed/down/gate tensors")
        }
        let program: Program
        if let cached = sharedOutputPrograms[diagnostics] { program = cached }
        else {
            let source = Self.sharedOutputSource.replacingOccurrences(of: "// DIAGNOSTIC_STORES", with:
                diagnostics ? "if (i == 0) gate_value[0] = sig;\n    gated_value[i] = product;" : "")
            program = try Program(
                name: diagnostics ? "ane_runner_shared_output_bf16_diagnostic" : "ane_runner_shared_output_bf16",
                source: source, inputs: ["routed", "down", "gate_logits", "sigtab"],
                outputs: diagnostics ? ["y", "gate_value", "gated_value"] : ["y"],
                outputShapes: diagnostics ? [shape, [1, 1, 1], shape] : [shape],
                grid: [(hidden + 255) / 256 * 256, 1, 1], threadgroup: [256, 1, 1],
                template: ["COUNT": hidden])
            sharedOutputPrograms[diagnostics] = program
        }
        let values = try program.apply([routed, down, gateLogits, sigmoidTable])
        return SharedOutput(y: values[0], gate: diagnostics ? values[1] : nil,
            gated: diagnostics ? values[2] : nil)
    }

    /// Explicit S2/S3 verification only. Reuse the S1 source and sigmoid LUT;
    /// flattening independent rows changes neither the BF16 operation order nor
    /// any projection, routing, or expert reduction.
    func verificationSharedActivation(_ gate: Tensor, up: Tensor) throws -> Tensor {
        let shape = gate.shape
        guard shape.count == 3, shape[0] == 1, (2...3).contains(shape[1]),
              shape[2] == intermediate, intermediate <= (Int(Int32.max) - 255) / 3,
              up.shape == shape, gate.dtype == MLX_BFLOAT16, up.dtype == MLX_BFLOAT16 else {
            throw GPUError.invalid("Verification shared activation requires matching BF16 [1,S,I], S2/S3")
        }
        let sequence = shape[1], count = sequence * intermediate
        let program: Program
        if let cached = verificationSharedActivationPrograms[sequence] { program = cached }
        else {
            program = try Program(name: "ane_runner_verify_shared_swiglu_bf16",
                source: Self.sharedActivationSource, inputs: ["gate", "up", "sigtab"],
                outputs: ["y"], outputShapes: [shape],
                grid: [(count + 255) / 256 * 256, 1, 1], threadgroup: [256, 1, 1],
                template: ["COUNT": count])
            verificationSharedActivationPrograms[sequence] = program
        }
        return try program.apply([gate, up, sigmoidTable])[0]
    }

    /// Reindex only the scalar router gate and its diagnostic store per token.
    /// Each multiply/add still rounds to BF16 at exactly the S1 boundaries.
    func verificationSharedOutput(routed: Tensor, down: Tensor, gateLogits: Tensor,
                                  diagnostics: Bool) throws -> SharedOutput {
        let shape = routed.shape
        guard shape.count == 3, shape[0] == 1, (2...3).contains(shape[1]),
              shape[2] == hidden, hidden <= (Int(Int32.max) - 255) / 3,
              down.shape == shape, gateLogits.shape == [1, shape[1], 1],
              routed.dtype == MLX_BFLOAT16, down.dtype == MLX_BFLOAT16,
              gateLogits.dtype == MLX_BFLOAT16 else {
            throw GPUError.invalid("Verification shared output requires BF16 [1,S,H]/[1,S,1], S2/S3")
        }
        let sequence = shape[1], count = sequence * hidden
        let key = VerificationKey(sequence: sequence, diagnostics: diagnostics)
        let program: Program
        if let cached = verificationSharedOutputPrograms[key] { program = cached }
        else {
            let source = try Self.reindex(Self.sharedOutputSource, [
                ("gate_logits[0]", "gate_logits[i / uint(H)]"),
                ("// DIAGNOSTIC_STORES", diagnostics
                    ? "if (i % uint(H) == 0) gate_value[i / uint(H)] = sig;\n    gated_value[i] = product;" : "")
            ])
            program = try Program(
                name: diagnostics ? "ane_runner_verify_shared_output_bf16_diagnostic" : "ane_runner_verify_shared_output_bf16",
                source: source, inputs: ["routed", "down", "gate_logits", "sigtab"],
                outputs: diagnostics ? ["y", "gate_value", "gated_value"] : ["y"],
                outputShapes: diagnostics ? [shape, [1, sequence, 1], shape] : [shape],
                grid: [(count + 255) / 256 * 256, 1, 1], threadgroup: [256, 1, 1],
                template: ["COUNT": count, "H": hidden])
            verificationSharedOutputPrograms[key] = program
        }
        let values = try program.apply([routed, down, gateLogits, sigmoidTable])
        return SharedOutput(y: values[0], gate: diagnostics ? values[1] : nil,
            gated: diagnostics ? values[2] : nil)
    }

    /// BF16 per-expert products, FP32 accumulation, then one BF16 output round.
    /// This intentionally differs from the author's low-precision prefill sum.
    func reducePrefill(_ experts: Tensor, scores: Tensor) throws -> Tensor {
        let shape = experts.shape
        guard shape.count == 4, shape[2] == topK, shape[3] == hidden,
              scores.shape == [shape[0], shape[1], topK],
              experts.dtype == MLX_BFLOAT16, scores.dtype == MLX_BFLOAT16 else {
            throw GPUError.invalid("Prefill weighted reduction shape/dtype mismatch")
        }
        let outputShape = [shape[0], shape[1], hidden]
        let count = shape[0] * shape[1] * hidden
        let program: Program
        if let cached = prefillPrograms[outputShape] { program = cached }
        else {
            program = try Program(
                name: "ane_runner_bf16_weighted_sum_fp32_accum", source: Self.prefillReduceSource,
                inputs: ["expert_values", "scores"], outputs: ["y"], outputShapes: [outputShape],
                grid: [(count + 255) / 256 * 256, 1, 1], threadgroup: [256, 1, 1],
                template: ["H": hidden, "TOPK": topK, "COUNT": count])
            prefillPrograms[outputShape] = program
        }
        return try program.apply([experts, scores])[0]
    }

    private final class Program {
        private let kernel: mlx_fast_metal_kernel
        private let config: mlx_fast_metal_kernel_config
        private let outputCount: Int

        init(name: String, source: String, inputs: [String], outputs: [String], outputShapes: [[Int]],
             grid: [Int], threadgroup: [Int], template: [String: Int],
             outputDType: mlx_dtype = MLX_BFLOAT16) throws {
            let namesIn = mlx_vector_string_new(), namesOut = mlx_vector_string_new()
            defer { _ = mlx_vector_string_free(namesIn); _ = mlx_vector_string_free(namesOut) }
            for value in inputs { try MX.check(mlx_vector_string_append_value(namesIn, value), "Metal input name") }
            for value in outputs { try MX.check(mlx_vector_string_append_value(namesOut, value), "Metal output name") }
            let created = mlx_fast_metal_kernel_new(name, namesIn, namesOut, source, "", true, false)
            guard created.ctx != nil else { throw GPUError.invalid("Failed to create Metal kernel \(name)") }
            let configuration = mlx_fast_metal_kernel_config_new()
            do {
                for shape in outputShapes {
                    let dimensions = shape.map(Int32.init)
                    try MX.check(mlx_fast_metal_kernel_config_add_output_arg(configuration, dimensions, dimensions.count, outputDType), "Metal output")
                }
                try MX.check(mlx_fast_metal_kernel_config_set_grid(configuration, Int32(grid[0]), Int32(grid[1]), Int32(grid[2])), "Metal grid")
                try MX.check(mlx_fast_metal_kernel_config_set_thread_group(configuration, Int32(threadgroup[0]), Int32(threadgroup[1]), Int32(threadgroup[2])), "Metal threadgroup")
                try MX.check(mlx_fast_metal_kernel_config_add_template_arg_dtype(configuration, "T", MLX_BFLOAT16), "Metal dtype")
                for (key, value) in template {
                    try MX.check(mlx_fast_metal_kernel_config_add_template_arg_int(configuration, key, Int32(value)), "Metal template")
                }
            } catch {
                mlx_fast_metal_kernel_free(created)
                mlx_fast_metal_kernel_config_free(configuration)
                throw error
            }
            kernel = created; config = configuration; outputCount = outputs.count
        }
        deinit { mlx_fast_metal_kernel_free(kernel); mlx_fast_metal_kernel_config_free(config) }
        func apply(_ inputs: [Tensor]) throws -> [Tensor] {
            let inputVector = mlx_vector_array_new_data(inputs.map(\.handle), inputs.count)
            var outputVector = mlx_vector_array_new()
            defer { _ = mlx_vector_array_free(inputVector); _ = mlx_vector_array_free(outputVector) }
            try MX.check(mlx_fast_metal_kernel_apply(&outputVector, kernel, inputVector, config, MX.stream), "Metal MoE kernel")
            guard mlx_vector_array_size(outputVector) == outputCount else { throw GPUError.invalid("Metal MoE output count mismatch") }
            return try (0..<outputCount).map { index in
                try MX.output("Metal MoE output") { mlx_vector_array_get(&$0, outputVector, index) }
            }
        }
    }

    // Explicit T conversions are observable BF16 rounding boundaries. In
    // particular the output kernel must not become an FP32 multiply-add/FMA.
    private static let sharedActivationSource = """
    uint i = thread_position_in_grid.x;
    if (i >= uint(COUNT)) return;
    T sig = sigtab[as_type<ushort>(gate[i])];
    T activated = T(float(gate[i]) * float(sig));
    y[i] = T(float(activated) * float(up[i]));
    """

    private static let sharedOutputSource = """
    uint i = thread_position_in_grid.x;
    if (i >= uint(COUNT)) return;
    T sig = sigtab[as_type<ushort>(gate_logits[0])];
    T product = T(float(down[i]) * float(sig));
    y[i] = T(float(routed[i]) + float(product));
    // DIAGNOSTIC_STORES
    """

    private static let prefillReduceSource = """
    uint i = thread_position_in_grid.x;
    if (i >= uint(COUNT)) return;
    uint token = i / uint(H);
    uint feature = i % uint(H);
    float total = 0.0f;
    for (uint slot = 0; slot < uint(TOPK); ++slot) {
        T product = expert_values[((size_t)token * uint(TOPK) + slot) * uint(H) + feature] * scores[(size_t)token * uint(TOPK) + slot];
        total += float(product);
    }
    y[i] = T(total);
    """

    // Explicit S3 operator probe only. Original gate/up and down source remain
    // byte-for-byte unchanged. Ordered IDs must be valid and unique per row.
    private static let groupedMembershipSource = """
    // One thread owns all three membership cells for one potential owner.
    uint e = thread_position_in_grid.x;
    if (e >= 30) return;
    uint expert = inds[e];
    bool owner = true;
    for (uint previous = 0; previous < e; ++previous) {
      if (inds[previous] == expert) { owner = false; break; }
    }
    for (int token = 0; token < 3; ++token) {
      int slot = -1;
      if (owner) {
        for (int candidate = 0; candidate < 10; ++candidate) {
          if (inds[token * 10 + candidate] == expert) { slot = candidate; break; }
        }
      }
      members[e * 3 + token] = slot;
    }
    """

    private static let groupedGateUpSource = """
    auto lane = thread_index_in_simdgroup;
    uint n = thread_position_in_grid.y;      // output row within the expert
    uint e = thread_position_in_grid.z;      // potential owner in original assignment order
    int slots[3] = {members[e * 3 + 0], members[e * 3 + 1], members[e * 3 + 2]};
    // Every lane/SIMD in this group takes the same branch, before weight loads.
    if (slots[0] < 0 && slots[1] < 0 && slots[2] < 0) return;

    int K = int(K_size);
    int N = int(N_size);
    int VPW = 32 / BITS;
    int K_by_p = K / VPW;
    int K_by_gs = K / GS;
    uint mask = (1u << BITS) - 1u;

    uint eid = inds[e];
    size_t wbase = (size_t)eid * (size_t)N * (size_t)K_by_p + (size_t)n * (size_t)K_by_p;
    size_t gbase = (size_t)eid * (size_t)N * (size_t)K_by_gs + (size_t)n * (size_t)K_by_gs;

    // Two independent 4-way accumulator sets: the gate chain and the up
    // chain never wait on each other.
    float g0[3] = {0}, g1[3] = {0}, g2[3] = {0}, g3[3] = {0};
    float u0[3] = {0}, u1[3] = {0}, u2[3] = {0}, u3[3] = {0};
    for (int pack = int(lane); pack < K_by_p; pack += 32) {
      uint32_t packed_g = wg_q[wbase + (size_t)pack];
      uint32_t packed_u = wu_q[wbase + (size_t)pack];
      int k_base = pack * VPW;
      int gi = k_base / GS;
      float sjg = float(g_scales[gbase + (size_t)gi]);
      float bjg = float(g_biases[gbase + (size_t)gi]);
      float sju = float(u_scales[gbase + (size_t)gi]);
      float bju = float(u_biases[gbase + (size_t)gi]);
      for (int ki = 0; ki < VPW; ki += 4) {
        uint32_t qg = packed_g >> (ki * BITS);
        uint32_t qu = packed_u >> (ki * BITS);
        // Only active members execute loads/arithmetic; no unconditional S3 work
        // for singleton owners. Maximum register demand remains a measured risk.
        #pragma clang loop unroll(full)
        for (int token = 0; token < 3; ++token) {
          if (slots[token] < 0) continue;
          size_t xi = (size_t)token * (size_t)K + (size_t)(k_base + ki);
        float x0 = float(x[xi + 0]);
        float x1 = float(x[xi + 1]);
        float x2 = float(x[xi + 2]);
        float x3 = float(x[xi + 3]);
        g0[token] += x0 * (float((qg >> (0 * BITS)) & mask) * sjg + bjg);
        g1[token] += x1 * (float((qg >> (1 * BITS)) & mask) * sjg + bjg);
        g2[token] += x2 * (float((qg >> (2 * BITS)) & mask) * sjg + bjg);
        g3[token] += x3 * (float((qg >> (3 * BITS)) & mask) * sjg + bjg);
        u0[token] += x0 * (float((qu >> (0 * BITS)) & mask) * sju + bju);
        u1[token] += x1 * (float((qu >> (1 * BITS)) & mask) * sju + bju);
        u2[token] += x2 * (float((qu >> (2 * BITS)) & mask) * sju + bju);
        u3[token] += x3 * (float((qu >> (3 * BITS)) & mask) * sju + bju);
        }
      }
    }
    #pragma clang loop unroll(full)
    for (int token = 0; token < 3; ++token) {
    if (slots[token] < 0) continue;
    float acc_g = simd_sum((g0[token] + g1[token]) + (g2[token] + g3[token]));
    float acc_u = simd_sum((u0[token] + u1[token]) + (u2[token] + u3[token]));
    if (lane == 0) {
      // Round exactly where the unfused path's two kernels wrote T(acc),
      // then the same table-lookup SwiGLU.
      T gt = T(acc_g);
      T ut = T(acc_u);
      T sig = sigtab[as_type<ushort>(gt)];
      y[((size_t)token * (size_t)TOPK + (size_t)slots[token]) * (size_t)N + (size_t)n] = (gt * sig) * ut;
    }
    }
    """

    private static let gateUpSource = """
    auto lane = thread_index_in_simdgroup;
    uint n = thread_position_in_grid.y;      // output row within the expert
    uint e = thread_position_in_grid.z;      // top-K slot
    
    int K = int(K_size);
    int N = int(N_size);
    int VPW = 32 / BITS;
    int K_by_p = K / VPW;
    int K_by_gs = K / GS;
    uint mask = (1u << BITS) - 1u;
    
    uint eid = inds[e];
    size_t wbase = (size_t)eid * (size_t)N * (size_t)K_by_p + (size_t)n * (size_t)K_by_p;
    size_t gbase = (size_t)eid * (size_t)N * (size_t)K_by_gs + (size_t)n * (size_t)K_by_gs;
    
    // Two independent 4-way accumulator sets: the gate chain and the up
    // chain never wait on each other.
    float g0 = 0.0f, g1 = 0.0f, g2 = 0.0f, g3 = 0.0f;
    float u0 = 0.0f, u1 = 0.0f, u2 = 0.0f, u3 = 0.0f;
    for (int pack = int(lane); pack < K_by_p; pack += 32) {
      uint32_t packed_g = wg_q[wbase + (size_t)pack];
      uint32_t packed_u = wu_q[wbase + (size_t)pack];
      int k_base = pack * VPW;
      int gi = k_base / GS;
      float sjg = float(g_scales[gbase + (size_t)gi]);
      float bjg = float(g_biases[gbase + (size_t)gi]);
      float sju = float(u_scales[gbase + (size_t)gi]);
      float bju = float(u_biases[gbase + (size_t)gi]);
      for (int ki = 0; ki < VPW; ki += 4) {
        size_t xi = (size_t)(k_base + ki);
        uint32_t qg = packed_g >> (ki * BITS);
        uint32_t qu = packed_u >> (ki * BITS);
        float x0 = float(x[xi + 0]);
        float x1 = float(x[xi + 1]);
        float x2 = float(x[xi + 2]);
        float x3 = float(x[xi + 3]);
        g0 += x0 * (float((qg >> (0 * BITS)) & mask) * sjg + bjg);
        g1 += x1 * (float((qg >> (1 * BITS)) & mask) * sjg + bjg);
        g2 += x2 * (float((qg >> (2 * BITS)) & mask) * sjg + bjg);
        g3 += x3 * (float((qg >> (3 * BITS)) & mask) * sjg + bjg);
        u0 += x0 * (float((qu >> (0 * BITS)) & mask) * sju + bju);
        u1 += x1 * (float((qu >> (1 * BITS)) & mask) * sju + bju);
        u2 += x2 * (float((qu >> (2 * BITS)) & mask) * sju + bju);
        u3 += x3 * (float((qu >> (3 * BITS)) & mask) * sju + bju);
      }
    }
    float acc_g = simd_sum((g0 + g1) + (g2 + g3));
    float acc_u = simd_sum((u0 + u1) + (u2 + u3));
    if (lane == 0) {
      // Round exactly where the unfused path's two kernels wrote T(acc),
      // then the same table-lookup SwiGLU.
      T gt = T(acc_g);
      T ut = T(acc_u);
      T sig = sigtab[as_type<ushort>(gt)];
      y[(size_t)e * (size_t)N + (size_t)n] = (gt * sig) * ut;
    }
    """

    private static let downReduceSource = """
    auto lane = thread_index_in_simdgroup;
    uint slot = simdgroup_index_in_threadgroup;   // top-K slot
    uint tile = threadgroup_position_in_grid.x;   // block of ROWS output rows
    
    int K = int(K_size);
    int N = int(N_size);
    int VPW = 32 / BITS;
    int K_by_p = K / VPW;
    int K_by_gs = K / GS;
    uint mask = (1u << BITS) - 1u;
    
    uint eid = inds[slot];
    size_t xoff = (size_t)slot * (size_t)K;
    threadgroup T slot_vals[uint(TOPK) * uint(ROWS)];
    for (uint r = 0; r < uint(ROWS); ++r) {
      uint n = tile * uint(ROWS) + r;
      size_t wbase = (size_t)eid * (size_t)N * (size_t)K_by_p + (size_t)n * (size_t)K_by_p;
      size_t gbase = (size_t)eid * (size_t)N * (size_t)K_by_gs + (size_t)n * (size_t)K_by_gs;
      float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
      for (int pack = int(lane); pack < K_by_p; pack += 32) {
        uint32_t packed = w_q[wbase + (size_t)pack];
        int k_base = pack * VPW;
        int gi = k_base / GS;
      float sj = float(scales[gbase + (size_t)gi]);
      float bj = float(biases[gbase + (size_t)gi]);
      for (int ki = 0; ki < VPW; ki += 4) {
        size_t xi = xoff + (size_t)(k_base + ki);
        uint32_t q = packed >> (ki * BITS);
        a0 += float(x[xi + 0]) * (float((q >> (0 * BITS)) & mask) * sj + bj);
        a1 += float(x[xi + 1]) * (float((q >> (1 * BITS)) & mask) * sj + bj);
        a2 += float(x[xi + 2]) * (float((q >> (2 * BITS)) & mask) * sj + bj);
        a3 += float(x[xi + 3]) * (float((q >> (3 * BITS)) & mask) * sj + bj);
      }
      }
      float acc = simd_sum((a0 + a1) + (a2 + a3));
      if (lane == 0) {
        slot_vals[slot * uint(ROWS) + r] = T(acc);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Composed-pair replica: T-rounded per-slot product, ascending T sum.
    if (slot == 0 && lane < uint(ROWS)) {
      T total = T(0.0f);
      for (uint s2 = 0; s2 < uint(TOPK); ++s2) {
        T p = slot_vals[s2 * uint(ROWS) + lane] * scores[s2];
        total = total + p;
      }
      y[(size_t)tile * (size_t)ROWS + (size_t)lane] = total;
    }
    """
}
