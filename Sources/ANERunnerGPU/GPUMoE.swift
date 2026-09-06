// GPU MoE math follows the public-MLX reference in scripts/benchmark_moe_mlx.py
// and mlx-serve/src/transformer.zig (moeMLP2, moeRoutingChain, sorted experts),
// upstream garnermccloud/mlx-serve commit 7dbcba04c98e4fd3bcc533c63e645547f13cc3b1.
// Decode uses the author's affine fused kernels in GPUMoEFused.swift.
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

public struct GPUMoEOutput {
    public let y: Tensor
    public let diagnostics: [String: Tensor]

    public init(y: Tensor, diagnostics: [String: Tensor]) {
        self.y = y
        self.diagnostics = diagnostics
    }
}

public enum GPUMoEError: Error, LocalizedError {
    case invalidConfiguration(String)
    case invalidTensor(String)

    public var errorDescription: String? {
        switch self {
        case .invalidConfiguration(let message), .invalidTensor(let message):
            return message
        }
    }
}

/// Complete routed and shared MoE on GPU, with original packed affine weights.
///
/// No selected IDs, probabilities, activations, or weights are read back to the
/// CPU. The caller owns evaluation/synchronization of the returned lazy graph.
/// It accepts already mixed/normalized [B,S,H] hidden states; the decoder
/// supplies HC/normalization and explicitly selects verification kernels.
public final class GPUMoE {
    public enum PrefillAccumulation: String, Codable, CaseIterable, Sendable {
        /// Preserve the author's BF16 products and public MLX BF16 reduction.
        case reference
        /// Keep BF16 products, accumulate in FP32, and round the result to BF16.
        /// This changes model numerics and may change generated token IDs.
        case float32
    }

    public let layer: Int
    public let hiddenSize: Int
    public let expertCount: Int
    public let topK: Int
    public let groupSize: Int
    public let bits: Int
    public let prefillAccumulation: PrefillAccumulation
    public let fuseSharedElementwise: Bool

    private struct QuantizedProjection {
        let weight: Tensor
        let scales: Tensor
        let biases: Tensor
    }

    private let intermediateSize: Int
    private let routerTransposed: Tensor
    private let sharedGateTransposed: Tensor
    private let sharedUpTransposed: Tensor
    private let sharedDownTransposed: Tensor
    private let sharedRouterTransposed: Tensor
    private let gate: QuantizedProjection
    private let up: QuantizedProjection
    private let down: QuantizedProjection
    private let fused: GPUMoEFused?
    private let routerFused: GPUMoERouterFused?
    private var prefillReductions: [Int: GPUMoEPrefillReduction] = [:]
    private var prefillGateUps: [Int: GPUMoEPrefillGateUp] = [:]

    public var decodeImplementation: String {
        fused == nil ? "public MLX gather_qmm" : "upstream affine fused gate/up/SwiGLU + down/weighted reduction"
    }
    public var routingImplementation: String {
        routerFused == nil ? "public MLX argsort/softmax with sequential BF16 normalization" :
            "upstream fused Metal softmax/top-k router; lowest expert ID tie-break; sequential BF16 normalization"
    }
    public var sharedElementwiseImplementation: String {
        fuseSharedElementwise && fused != nil ?
            "decode: two Metal kernels with explicit BF16 boundaries and source sigmoid LUT; prefill: public MLX" :
            "public MLX sigmoid/multiply/multiply and sigmoid/multiply/add"
    }
    public var prefillReductionImplementation: String {
        switch prefillAccumulation {
        case .reference:
            return "reference: BF16 per-expert products and public MLX BF16 reduction, matching upstream"
        case .float32:
            return "float32: BF16 per-expert products, FP32 accumulation, BF16 output; intentionally differs from upstream"
        }
    }

    public init(
        weights: GPUWeights,
        layer: Int,
        hiddenSize: Int = 2560,
        experts: Int = 512,
        topK: Int = 10,
        groupSize: Int = 64,
        bits: Int = 4,
        prefillAccumulation: PrefillAccumulation = .reference,
        fuseSharedElementwise: Bool = false,
        weightPrefix: String? = nil
    ) throws {
        guard layer >= 0, hiddenSize > 0, experts > 0, topK > 0,
              topK <= experts, groupSize > 0, [2, 4, 8].contains(bits),
              hiddenSize % groupSize == 0 else {
            throw GPUMoEError.invalidConfiguration("Invalid affine MoE dimensions or quantization parameters")
        }
        let packedPerWord = 32 / bits
        guard hiddenSize % packedPerWord == 0 else {
            throw GPUMoEError.invalidConfiguration("Hidden size must be divisible by the packed uint32 width")
        }
        let prefix = (weightPrefix ?? "language_model.model.layers.\(layer).mlp") + "."
        let router = try weights.tensor(prefix + "gate.weight")
        let sharedRouter = try weights.tensor(prefix + "shared_expert_gate.weight")
        let sharedGate = try weights.tensor(prefix + "shared_expert.gate_proj.weight")
        let sharedUp = try weights.tensor(prefix + "shared_expert.up_proj.weight")
        let sharedDown = try weights.tensor(prefix + "shared_expert.down_proj.weight")
        func projection(_ name: String) throws -> QuantizedProjection {
            let name = prefix + "switch_mlp." + name
            return try QuantizedProjection(
                weight: weights.tensor(name + ".weight"),
                scales: weights.tensor(name + ".scales"),
                biases: weights.tensor(name + ".biases")
            )
        }
        let gate = try projection("gate_proj")
        let up = try projection("up_proj")
        let down = try projection("down_proj")
        guard gate.weight.shape.count == 3 else {
            throw GPUMoEError.invalidTensor("Expected rank-3 packed expert gate weights")
        }
        let intermediate = gate.weight.shape[1]
        guard intermediate > 0, intermediate % groupSize == 0,
              intermediate % packedPerWord == 0 else {
            throw GPUMoEError.invalidTensor("Expert intermediate size is incompatible with packed affine groups")
        }
        func expect(_ tensor: Tensor, _ shape: [Int], _ dtype: mlx_dtype, _ name: String) throws {
            guard tensor.shape == shape, tensor.dtype == dtype else {
                throw GPUMoEError.invalidTensor(
                    "\(prefix)\(name): expected shape \(shape) and dtype \(dtype), got \(tensor.shape) / \(tensor.dtype)"
                )
            }
        }
        func expectProjection(_ projection: QuantizedProjection, output: Int, input: Int, name: String) throws {
            try expect(projection.weight, [experts, output, input / packedPerWord], MLX_UINT32, name + ".weight")
            try expect(projection.scales, [experts, output, input / groupSize], MLX_BFLOAT16, name + ".scales")
            try expect(projection.biases, [experts, output, input / groupSize], MLX_BFLOAT16, name + ".biases")
        }
        try expect(router, [experts, hiddenSize], MLX_BFLOAT16, "gate.weight")
        try expect(sharedRouter, [1, hiddenSize], MLX_BFLOAT16, "shared_expert_gate.weight")
        guard sharedGate.shape.count == 2, sharedGate.shape[0] > 0 else {
            throw GPUMoEError.invalidTensor("Expected rank-2 shared expert gate weights")
        }
        let sharedIntermediate = sharedGate.shape[0]
        try expect(sharedGate, [sharedIntermediate, hiddenSize], MLX_BFLOAT16, "shared_expert.gate_proj.weight")
        try expect(sharedUp, [sharedIntermediate, hiddenSize], MLX_BFLOAT16, "shared_expert.up_proj.weight")
        try expect(sharedDown, [hiddenSize, sharedIntermediate], MLX_BFLOAT16, "shared_expert.down_proj.weight")
        try expectProjection(gate, output: intermediate, input: hiddenSize, name: "switch_mlp.gate_proj")
        try expectProjection(up, output: intermediate, input: hiddenSize, name: "switch_mlp.up_proj")
        try expectProjection(down, output: hiddenSize, input: intermediate, name: "switch_mlp.down_proj")

        self.layer = layer
        self.hiddenSize = hiddenSize
        self.expertCount = experts
        self.topK = topK
        self.groupSize = groupSize
        self.bits = bits
        self.prefillAccumulation = prefillAccumulation
        self.fuseSharedElementwise = fuseSharedElementwise
        self.intermediateSize = intermediate
        self.gate = gate
        self.up = up
        self.down = down
        self.routerTransposed = try MX.transpose(router, [1, 0])
        self.sharedGateTransposed = try MX.transpose(sharedGate, [1, 0])
        self.sharedUpTransposed = try MX.transpose(sharedUp, [1, 0])
        self.sharedDownTransposed = try MX.transpose(sharedDown, [1, 0])
        self.sharedRouterTransposed = try MX.transpose(sharedRouter, [1, 0])
        if bits == 4, groupSize == 64, hiddenSize % 4 == 0, intermediate % 8 == 0, topK <= 32 {
            self.fused = try GPUMoEFused(hidden: hiddenSize, intermediate: intermediate,
                                         topK: topK, groupSize: groupSize, bits: bits)
        } else { self.fused = nil }
        if experts <= 2048, topK <= 32 {
            self.routerFused = try GPUMoERouterFused(expertCount: experts, topK: topK)
        } else { self.routerFused = nil }
    }

    private func project(
        _ x: Tensor, _ projection: QuantizedProjection, indices: Tensor, sorted: Bool
    ) throws -> Tensor {
        // `biases` is the affine dequantization offset, never a Linear bias.
        try MX.gatherQMM(
            x, weight: projection.weight, scales: projection.scales, biases: projection.biases,
            rhsIndices: indices, groupSize: groupSize, bits: bits, sortedIndices: sorted
        )
    }

    private func swiglu(_ gate: Tensor, _ up: Tensor) throws -> Tensor {
        // Preserve the public MLX reference's BF16 operation boundaries.
        let sigmoid = try MX.sigmoid(gate)
        let activated = try MX.mul(gate, sigmoid)
        return try MX.mul(activated, up)
    }

    /// The nil path preserves lazy execution without clocks or output collection.
    /// Detail stages replace the caller's outer MoE stage; nesting is invalid.
    private func measure<T>(_ name: String, profiler: GPUProfiler?, tokens: Int,
                            outputs: (T) -> [Tensor], _ body: () throws -> T) throws -> T {
        guard let profiler else { return try body() }
        return try profiler.measure(name,layer: layer,tokenCount: tokens,outputs: outputs,body)
    }

    /// Detail profiling is restricted to the multi-token prefill path; the
    /// caller selects the business phase and must omit its outer MoE measure.
    public func forward(_ input: Tensor, diagnostics: Bool = false,
                        useFusedSharedElementwise: Bool? = nil,
                        verificationLinear: GPUVerificationLinear? = nil,
                        verificationTokenAxis: Bool = false,
                        profiler: GPUProfiler? = nil,
                        prefillReductionThreadgroup: Int? = nil,
                        prefillGateUpVariant: Int? = nil,
                        groupedDown: Bool = false) throws -> GPUMoEOutput {
        let useSharedFusion = useFusedSharedElementwise ?? fuseSharedElementwise
        guard !useSharedFusion || fuseSharedElementwise else {
            throw GPUMoEError.invalidConfiguration("Shared fusion must be enabled at initialization before selecting it per request")
        }
        guard input.shape.count == 3, input.shape[0] > 0, input.shape[1] > 0,
              input.shape[2] == hiddenSize else {
            throw GPUMoEError.invalidTensor("MoE input must have nonempty shape [batch, sequence, \(hiddenSize)]")
        }
        let batch = input.shape[0]
        let sequence = input.shape[1]
        let (tokens, tokenOverflow) = batch.multipliedReportingOverflow(by: sequence)
        let (assignments, assignmentOverflow) = tokens.multipliedReportingOverflow(by: topK)
        if let variant = prefillGateUpVariant {
            guard (0...3).contains(variant), batch == 1,
                  ((variant >= 2 ? 205 : 2)...512).contains(tokens),
                  hiddenSize == 2560, intermediateSize == 640, expertCount == 512, topK == 10,
                  bits == 4, groupSize == 64, fused != nil,
                  verificationLinear == nil, !verificationTokenAxis else {
                throw GPUMoEError.invalidConfiguration("Fused gate/up requires the model's multi-token prefill path")
            }
        }
        guard !groupedDown || (prefillGateUpVariant.map { $0 >= 2 } ?? false) else {
            throw GPUMoEError.invalidConfiguration("Grouped down requires an expert-aligned gate/up plan")
        }
        if let threads = prefillReductionThreadgroup {
            guard [128,256,512].contains(threads), batch == 1, tokens > 1,
                  hiddenSize == 2560, topK == 10, prefillAccumulation == .reference,
                  verificationLinear == nil, !verificationTokenAxis else {
                throw GPUMoEError.invalidConfiguration("Fused prefill reduction requires the reference multi-token Qwen MoE path")
            }
        }
        guard !tokenOverflow, !assignmentOverflow else {
            throw GPUMoEError.invalidTensor("MoE token assignment count overflow")
        }
        guard profiler == nil || (tokens > 1 && verificationLinear == nil && !verificationTokenAxis) else {
            throw GPUMoEError.invalidConfiguration("Detailed MoE profiling requires multi-token prefill without verification kernels")
        }
        let verification = batch == 1 && (2...5).contains(sequence) ? verificationLinear : nil
        guard !verificationTokenAxis || verification != nil else {
            throw GPUMoEError.invalidConfiguration("Token-axis MoE requires verification linear kernels and [1,S,H] with S in 2...5")
        }
        let routing: (x: Tensor,logits: Tensor,indices: Tensor,weights: Tensor) = try measure(
            "moe.router",profiler: profiler,tokens: tokens,
            outputs: { [$0.x,$0.logits,$0.indices,$0.weights] }) {
                let x = try MX.cast(input, MLX_BFLOAT16)
                let logits = try MX.linear(x, routerTransposed, verification: verification)
                let indices: Tensor
                let routingWeights: Tensor
                if let routerFused {
                    let routing = try routerFused.route(logits)
                    indices = routing.indices
                    routingWeights = routing.weights
                } else {
                    let order = try MX.argsort(MX.negative(logits), axis: -1)
                    indices = try MX.slice(order, starts: [0, 0, 0], ends: [batch, sequence, topK])
                    let probabilities = try MX.softmax(logits, axis: -1, precise: true)
                    let selected = try MX.takeAlong(probabilities, indices, axis: -1)

                    // The upstream router rounds every denominator add to BF16.
                    // This fallback retains that order outside the fused geometry.
                    var denominator = try MX.slice(selected, starts: [0, 0, 0], ends: [batch, sequence, 1])
                    for slot in 1..<topK {
                        let score = try MX.slice(selected, starts: [0, 0, slot], ends: [batch, sequence, slot + 1])
                        denominator = try MX.cast(MX.add(denominator, score), MLX_BFLOAT16)
                    }
                    routingWeights = try MX.cast(MX.div(selected, denominator), MLX_BFLOAT16)
                }
                return (x,logits,indices,routingWeights)
            }
        let x = routing.x, logits = routing.logits
        let indices = routing.indices, routingWeights = routing.weights

        let expertOutputs: Tensor?
        let routed: Tensor
        var prefillActivation: Tensor?
        if verification != nil {
            guard let fused else {
                throw GPUMoEError.invalidConfiguration("Verification linear MoE requires the scalar affine-Q4 fused expert path")
            }
            if verificationTokenAxis {
                let result = try fused.verifyTokens(
                    x, indices: indices, scores: routingWeights,
                    gate: .init(weight: gate.weight, scales: gate.scales, biases: gate.biases),
                    up: .init(weight: up.weight, scales: up.scales, biases: up.biases),
                    down: .init(weight: down.weight, scales: down.scales, biases: down.biases),
                    diagnostics: diagnostics)
                routed = result.routed
                expertOutputs = result.expertOutputs
            } else {
                // Share the router and dense shared-expert work across tokens,
                // while keeping each routed expert's original S1 arithmetic.
                // No sorting, gather-QMM, or different expert reduction is used.
                var rows: [Tensor] = [], diagnosticRows: [Tensor] = []
                rows.reserveCapacity(sequence)
                if diagnostics { diagnosticRows.reserveCapacity(sequence) }
                for token in 0..<sequence {
                    let row = try MX.slice(x, starts: [0, token, 0], ends: [1, token + 1, hiddenSize])
                    let ids = try MX.slice(indices, starts: [0, token, 0], ends: [1, token + 1, topK])
                    let scores = try MX.slice(routingWeights, starts: [0, token, 0], ends: [1, token + 1, topK])
                    let result = try fused.decode(
                        row, indices: ids, scores: scores,
                        gate: .init(weight: gate.weight, scales: gate.scales, biases: gate.biases),
                        up: .init(weight: up.weight, scales: up.scales, biases: up.biases),
                        down: .init(weight: down.weight, scales: down.scales, biases: down.biases),
                        diagnostics: diagnostics)
                    rows.append(result.routed)
                    if diagnostics {
                        guard let experts = result.expertOutputs else {
                            throw GPUError.invalid("Missing requested scalar expert diagnostics during verification")
                        }
                        diagnosticRows.append(experts)
                    }
                }
                routed = try MX.concat(rows, axis: 1)
                expertOutputs = diagnostics ? try MX.concat(diagnosticRows, axis: 1) : nil
            }
        } else if tokens > 1 {
            // Sort and invert entirely on-device, grouping assignments by
            // expert so the original Q4 bank can be streamed efficiently.
            let sorted: (expandedX: Tensor,indices: Tensor,inverseOrder: Tensor) = try measure(
                "moe.sort_gather",profiler: profiler,tokens: tokens,
                outputs: { [$0.expandedX,$0.indices,$0.inverseOrder] }) {
                    let flatIndices = try MX.reshape(indices, [assignments])
                    let sortedOrder = try MX.argsort(flatIndices, axis: 0)
                    let inverseOrder = try MX.argsort(sortedOrder, axis: 0)
                    let sortedIndices = try MX.take(flatIndices, sortedOrder, axis: 0)
                    let tokenIndices = try MX.floorDivide(sortedOrder, by: topK)
                    let flatX = try MX.reshape(x, [tokens, hiddenSize])
                    let gatheredX = try MX.take(flatX, tokenIndices, axis: 0)
                    let expandedX = try MX.reshape(gatheredX, [assignments, 1, hiddenSize])
                    return (expandedX,sortedIndices,inverseOrder)
                }
            // One request-local GPU plan is shared by both matrix stages.
            // It is retained by the lazy nodes; no routing metadata is read on CPU.
            let gateUpKernel: GPUMoEPrefillGateUp?
            if let variant = prefillGateUpVariant {
                if prefillGateUps[variant] == nil {
                    prefillGateUps[variant] = try GPUMoEPrefillGateUp(variant: variant)
                }
                gateUpKernel = prefillGateUps[variant]
            } else { gateUpKernel = nil }
            let expertPlan = try gateUpKernel?.plan(indices: sorted.indices)
            let activation = try measure(
                "moe.gate_up_activation",profiler: profiler,tokens: tokens,outputs: { [$0] }) {
                    if let gateUpKernel {
                        return try gateUpKernel.forward(sorted.expandedX,
                            gateWeight: gate.weight, gateScales: gate.scales, gateBiases: gate.biases,
                            upWeight: up.weight, upScales: up.scales, upBiases: up.biases,
                            indices: sorted.indices, sigmoidTable: fused!.prefillSigmoidTable, plan: expertPlan)
                    }
                    let gates = try project(sorted.expandedX, gate, indices: sorted.indices, sorted: true)
                    let ups = try project(sorted.expandedX, up, indices: sorted.indices, sorted: true)
                    return try swiglu(gates, ups)
                }
            if diagnostics { prefillActivation = activation }
            let projected = try measure("moe.down",profiler: profiler,tokens: tokens,outputs: { [$0] }) {
                if groupedDown, let gateUpKernel, let expertPlan {
                    return try gateUpKernel.down(activation, weight: down.weight, scales: down.scales,
                        biases: down.biases, plan: expertPlan)
                }
                return try project(activation, down, indices: sorted.indices, sorted: true)
            }
            let reduction: (routed: Tensor,experts: Tensor?) = try measure(
                "moe.unsort_reduce",profiler: profiler,tokens: tokens,outputs: { [$0.routed] + [$0.experts].compactMap { $0 } }) {
                    if let threads = prefillReductionThreadgroup {
                        if prefillReductions[threads] == nil {
                            prefillReductions[threads] = try GPUMoEPrefillReduction(threadgroupSize: threads)
                        }
                        let routed = try prefillReductions[threads]!.reduce(projected: projected,
                            inverseOrder: sorted.inverseOrder, scores: routingWeights, tokens: tokens)
                        // Diagnostic materialization is a side branch. The routed
                        // result always comes from the candidate, including here.
                        let experts: Tensor?
                        if diagnostics {
                            experts = try MX.reshape(MX.take(MX.reshape(projected, [assignments,hiddenSize]),
                                sorted.inverseOrder, axis: 0), [batch,sequence,topK,hiddenSize])
                        } else { experts = nil }
                        return (routed,experts)
                    }
                    let flatOutputs = try MX.reshape(projected, [assignments, hiddenSize])
                    let originalOrder = try MX.take(flatOutputs, sorted.inverseOrder, axis: 0)
                    let experts = try MX.reshape(originalOrder, [batch, sequence, topK, hiddenSize])
                    let routed: Tensor
                    switch prefillAccumulation {
                    case .reference:
                        // Preserve the original public MLX reduction's BF16 products,
                        // accumulation dtype and reduction order for this layout.
                        let expandedWeights = try MX.reshape(routingWeights, [batch, sequence, topK, 1])
                        let products = try MX.mul(experts, expandedWeights)
                        routed = try MX.sum(products, axis: -2, keepDims: false)
                    case .float32:
                        if let fused {
                            routed = try fused.reducePrefill(experts, scores: routingWeights)
                        } else {
                            let expandedWeights = try MX.reshape(routingWeights, [batch, sequence, topK, 1])
                            let products = try MX.mul(experts, expandedWeights)
                            routed = try MX.cast(MX.sum(MX.cast(products, MLX_FLOAT32), axis: -2, keepDims: false), MLX_BFLOAT16)
                        }
                    }
                    return (routed,experts)
                }
            expertOutputs = reduction.experts
            routed = reduction.routed
        } else if let fused {
            let result = try fused.decode(
                x, indices: indices, scores: routingWeights,
                gate: .init(weight: gate.weight, scales: gate.scales, biases: gate.biases),
                up: .init(weight: up.weight, scales: up.scales, biases: up.biases),
                down: .init(weight: down.weight, scales: down.scales, biases: down.biases),
                diagnostics: diagnostics)
            routed = result.routed
            expertOutputs = result.expertOutputs
        } else {
            // Broadcast the single token across all selected experts. No
            // expert bank is gathered or dequantized into a CPU-side array.
            let expandedX = try MX.reshape(x, [batch, sequence, 1, 1, hiddenSize])
            let gates = try project(expandedX, gate, indices: indices, sorted: false)
            let ups = try project(expandedX, up, indices: indices, sorted: false)
            let activation = try swiglu(gates, ups)
            let projected = try project(activation, down, indices: indices, sorted: false)
            let experts = try MX.reshape(projected, [batch, sequence, topK, hiddenSize])
            expertOutputs = experts
            let expandedWeights = try MX.reshape(routingWeights, [batch, sequence, topK, 1])
            let weightedExperts = try MX.mul(experts, expandedWeights)
            routed = try MX.sum(weightedExperts, axis: -2, keepDims: false)
        }

        let shared: (gateProjection: Tensor,upProjection: Tensor,activation: Tensor,down: Tensor,
                     gateLogits: Tensor,gate: Tensor?,gated: Tensor?,y: Tensor) = try measure(
            "moe.shared_combine",profiler: profiler,tokens: tokens,
            outputs: { value in
                guard diagnostics else { return [value.y] }
                return [value.y,value.gateProjection,value.upProjection,value.activation,value.down,value.gateLogits]
                    + [value.gate,value.gated].compactMap { $0 }
            }) {
                let sharedGateProjection = try MX.linear(x, sharedGateTransposed, verification: verification)
                let sharedUpProjection = try MX.linear(x, sharedUpTransposed, verification: verification)
                // The flag only changes a one-token shared expert's elementwise tails.
                // The explicit verification policy above independently selects dense
                // scalar-order projections; no weight bank is copied or repacked.
                let sharedFusion = tokens == 1 && useSharedFusion ? fused : nil
                let sharedActivation: Tensor
                if let sharedFusion {
                    sharedActivation = try sharedFusion.sharedActivation(sharedGateProjection, up: sharedUpProjection)
                } else {
                    sharedActivation = try swiglu(sharedGateProjection, sharedUpProjection)
                }
                let sharedDown = try MX.linear(sharedActivation, sharedDownTransposed, verification: verification)
                let sharedGateLogits: Tensor
                if verification != nil {
                    // N=1 takes MLX's scalar dot-product kernel, not GEMV. Retain it
                    // per row instead of routing this matrix to verification linear.
                    var rows: [Tensor] = []
                    rows.reserveCapacity(sequence)
                    for token in 0..<sequence {
                        let row = try MX.slice(x, starts: [0, token, 0], ends: [1, token + 1, hiddenSize])
                        rows.append(try MX.matmul(row, sharedRouterTransposed))
                    }
                    sharedGateLogits = try MX.concat(rows, axis: 1)
                } else {
                    sharedGateLogits = try MX.matmul(x, sharedRouterTransposed)
                }
                let sharedGate: Tensor?, sharedGated: Tensor?, y: Tensor
                if let sharedFusion {
                    let result = try sharedFusion.sharedOutput(routed: routed, down: sharedDown,
                        gateLogits: sharedGateLogits, diagnostics: diagnostics)
                    y = result.y; sharedGate = result.gate; sharedGated = result.gated
                } else {
                    let gate = try MX.sigmoid(sharedGateLogits)
                    let gated = try MX.mul(sharedDown, gate)
                    sharedGate = gate; sharedGated = gated
                    y = try MX.add(routed, gated)
                }
                return (sharedGateProjection,sharedUpProjection,sharedActivation,sharedDown,
                        sharedGateLogits,sharedGate,sharedGated,y)
            }
        let y = shared.y
        var values: [String: Tensor] = [:]
        if diagnostics {
            guard let expertOutputs, let sharedGate = shared.gate, let sharedGated = shared.gated else {
                throw GPUError.invalid("Missing requested MoE output diagnostics")
            }
            values = [
                "x": x, "router_input": x, "expert_input": x,
                "router_logits": logits, "selected_experts": indices,
                "routing_weights": routingWeights, "selected_expert_outputs": expertOutputs,
                "routed_sum": routed, "shared_gate_projection": shared.gateProjection,
                "shared_up_projection": shared.upProjection, "shared_activation": shared.activation,
                "shared_down": shared.down, "shared_gate_logits": shared.gateLogits,
                "shared_gate": sharedGate, "shared_gated": sharedGated, "output": y,
            ]
            if let prefillActivation { values["prefill_activation"] = prefillActivation }
        }
        return GPUMoEOutput(y: y, diagnostics: values)
    }
}
