import Accelerate
import Darwin
import Foundation

/// `.bfloat16Boundaries` follows the qwen4_exp router's dtype boundaries,
/// including BF16 rounding after each selected-probability addition. It does
/// not emulate MLX's GPU GEMV reduction tree or fast exponential implementation.
public enum ExpertRouterPrecision: String, Codable, Sendable {
    case float32
    case bfloat16Boundaries
}

public enum ExpertRouterError: Error, LocalizedError, Sendable {
    case invalid(String)

    public var errorDescription: String? {
        switch self {
        case .invalid(let reason): reason
        }
    }
}

/// Token-major results. Expert slots are sorted by descending logit, with the
/// lower expert ID first for exact ties. The fused MLX router uses that tie
/// rule; its argpartition fallback may return a different order or tied set.
public struct RoutedTokens: Sendable {
    public let tokenCount: Int
    public let expertCount: Int
    public let topK: Int
    public let precision: ExpertRouterPrecision
    /// [tokenCount, topK], preserving the slot order used by `weights`.
    public let expertIDs: [Int]
    /// [tokenCount, topK]. BF16 rounding can make their sum differ from one.
    public let weights: [Float]
    /// [tokenCount, expertCount], after the requested logit precision boundary.
    public let logits: [Float]
}

/// CPU routing for a dense, bias-free [expertCount, hiddenSize] matrix.
///
/// qwen4_exp uses direct xWᵀ, selects on logits, computes softmax across ALL
/// experts, then renormalizes the selected probabilities. There is no router
/// RMSNorm, bias, sigmoid, or routed scale. Its shared-expert sigmoid is a
/// separate operation and is intentionally outside this module.
///
/// Semantics checked against garnermccloud/mlx-serve commit 7dbcba04, functions
/// moeMLP2, moeRoutingChain, and moeRouterSource in src/transformer.zig.
/// Exporters provide decoded Float values; this type never opens model files.
public struct ExpertRouter: Sendable {
    public let expertCount: Int
    public let hiddenSize: Int
    public let topK: Int
    public let precision: ExpertRouterPrecision
    private let matrix: [Float]

    public init(
        weights: [Float], expertCount: Int, hiddenSize: Int, topK: Int,
        precision: ExpertRouterPrecision = .float32
    ) throws {
        guard expertCount > 0, hiddenSize > 0, topK > 0, topK <= expertCount,
            expertCount <= Int(Int32.max), hiddenSize <= Int(Int32.max)
        else { throw ExpertRouterError.invalid("Invalid expertCount, hiddenSize, or topK") }
        let expected = try Self.checkedCount(expertCount, hiddenSize)
        guard weights.count == expected else {
            throw ExpertRouterError.invalid("Router weights must contain \(expected) row-major values")
        }
        try Self.requireFinite(weights, name: "Router weights")
        let prepared = precision == .bfloat16Boundaries ? weights.map(NGramTable.roundBFloat16) : weights
        try Self.requireFinite(prepared, name: "Rounded router weights")
        self.expertCount = expertCount
        self.hiddenSize = hiddenSize
        self.topK = topK
        self.precision = precision
        self.matrix = prepared
    }

    /// `tokens` is row-major [tokenCount, hiddenSize]. Accelerate accumulates
    /// the decoded matrix-vector products in Float32 for each token.
    public func route(tokens: [Float], tokenCount: Int) throws -> RoutedTokens {
        guard tokenCount > 0 else { throw ExpertRouterError.invalid("tokenCount must be positive") }
        guard tokens.count == (try Self.checkedCount(tokenCount, hiddenSize)) else {
            throw ExpertRouterError.invalid("Token tensor shape does not match tokenCount and hiddenSize")
        }
        try Self.requireFinite(tokens, name: "Router input")
        let input = precision == .bfloat16Boundaries ? tokens.map(NGramTable.roundBFloat16) : tokens
        try Self.requireFinite(input, name: "Rounded router input")
        var logits = [Float](repeating: 0, count: try Self.checkedCount(tokenCount, expertCount))
        matrix.withUnsafeBufferPointer { w in
            input.withUnsafeBufferPointer { x in
                logits.withUnsafeMutableBufferPointer { y in
                    for token in 0..<tokenCount {
                        cblas_sgemv(
                            CblasRowMajor, CblasNoTrans, Int32(expertCount), Int32(hiddenSize),
                            1, w.baseAddress!, Int32(hiddenSize), x.baseAddress! + token * hiddenSize,
                            1, 0, y.baseAddress! + token * expertCount, 1
                        )
                    }
                }
            }
        }
        if precision == .bfloat16Boundaries { logits = logits.map(NGramTable.roundBFloat16) }
        try Self.requireFinite(logits, name: "Router logits (matrix product overflow)")

        let slotCount = try Self.checkedCount(tokenCount, topK)
        var selectedIDs = [Int]()
        var selectedWeights = [Float]()
        selectedIDs.reserveCapacity(slotCount)
        selectedWeights.reserveCapacity(slotCount)
        var probabilities = [Float](repeating: 0, count: expertCount)
        for token in 0..<tokenCount {
            let offset = token * expertCount
            let order = (0..<expertCount).sorted { left, right in
                let lhs = logits[offset + left]
                let rhs = logits[offset + right]
                return lhs == rhs ? left < right : lhs > rhs
            }
            let maximum = logits[offset + order[0]]
            var exponentialSum: Float = 0
            for expert in 0..<expertCount {
                // Finite opposite-sign extrema may subtract to -infinity;
                // exp(-infinity)=0 remains a valid softmax contribution.
                let value = Darwin.expf(logits[offset + expert] - maximum)
                probabilities[expert] = value
                exponentialSum += value
            }
            guard exponentialSum.isFinite, exponentialSum > 0 else {
                throw ExpertRouterError.invalid("Softmax denominator is not positive and finite")
            }
            let reciprocal: Float = 1 / exponentialSum
            for expert in 0..<expertCount {
                probabilities[expert] = rounded(probabilities[expert] * reciprocal)
            }
            var selectedSum: Float = 0
            for expert in order.prefix(topK) {
                selectedSum = rounded(selectedSum + probabilities[expert])
            }
            guard selectedSum.isFinite, selectedSum > 0 else {
                throw ExpertRouterError.invalid("Selected routing probabilities cannot be normalized")
            }
            for expert in order.prefix(topK) {
                selectedIDs.append(expert)
                selectedWeights.append(rounded(probabilities[expert] / selectedSum))
            }
        }
        return RoutedTokens(
            tokenCount: tokenCount, expertCount: expertCount, topK: topK, precision: precision,
            expertIDs: selectedIDs, weights: selectedWeights, logits: logits
        )
    }

    /// Merge row-major [tokenCount, topK, outputSize] expert outputs into
    /// [tokenCount, outputSize]. Slots must match the routing result's order;
    /// expert IDs are global IDs, NOT indices into this selected-output tensor.
    ///
    /// This helper uses Float32 products/accumulation without hidden BF16 casts.
    /// It is not a bitwise emulator of MLX's fused down-projection/reduction.
    public static func weightedMerge(
        expertOutputs: [Float], routing: RoutedTokens, outputSize: Int
    ) throws -> [Float] {
        guard routing.tokenCount > 0, routing.expertCount > 0,
            routing.topK > 0, routing.topK <= routing.expertCount,
            outputSize > 0, outputSize <= Int(Int32.max), routing.topK <= Int(Int32.max)
        else { throw ExpertRouterError.invalid("Invalid weighted-merge dimensions") }
        let slots = try checkedCount(routing.tokenCount, routing.topK)
        guard routing.weights.count == slots, routing.expertIDs.count == slots,
            routing.logits.count == (try checkedCount(routing.tokenCount, routing.expertCount)),
            expertOutputs.count == (try checkedCount(slots, outputSize)),
            routing.expertIDs.allSatisfy({ $0 >= 0 && $0 < routing.expertCount }),
            routing.weights.allSatisfy({ $0.isFinite && $0 >= 0 })
        else { throw ExpertRouterError.invalid("Inconsistent routing or selected expert-output shape") }
        try requireFinite(expertOutputs, name: "Expert outputs")
        var output = [Float](repeating: 0, count: try checkedCount(routing.tokenCount, outputSize))
        expertOutputs.withUnsafeBufferPointer { experts in
            routing.weights.withUnsafeBufferPointer { weights in
                output.withUnsafeMutableBufferPointer { merged in
                    for token in 0..<routing.tokenCount {
                        cblas_sgemv(
                            CblasRowMajor, CblasTrans, Int32(routing.topK), Int32(outputSize),
                            1, experts.baseAddress! + token * routing.topK * outputSize,
                            Int32(outputSize), weights.baseAddress! + token * routing.topK,
                            1, 0, merged.baseAddress! + token * outputSize, 1
                        )
                    }
                }
            }
        }
        try requireFinite(output, name: "Weighted-merge output (accumulation overflow)")
        return output
    }

    private func rounded(_ value: Float) -> Float {
        precision == .bfloat16Boundaries ? NGramTable.roundBFloat16(value) : value
    }

    private static func checkedCount(_ dimensions: Int...) throws -> Int {
        var count = 1
        for dimension in dimensions {
            let product = count.multipliedReportingOverflow(by: dimension)
            guard dimension >= 0, !product.overflow else {
                throw ExpertRouterError.invalid("Tensor element count overflows Int")
            }
            count = product.partialValue
        }
        return count
    }

    private static func requireFinite(_ values: [Float], name: String) throws {
        if let index = values.firstIndex(where: { !$0.isFinite }) {
            throw ExpertRouterError.invalid("\(name) contains a nonfinite value at index \(index)")
        }
    }
}
