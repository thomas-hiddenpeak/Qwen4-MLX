// Adapted from garnermccloud/mlx-serve src/transformer.zig,
// loadQwen4Mtp/qwen4MtpForward, and src/generate.zig MtpHeadRef.qwen4,
// fixed commit 7dbcba04c98e4fd3bcc533c63e645547f13cc3b1.
// This is the qwen4_exp residual_linear_shared head, not generic mtp.zig.
// Changes: Swift ownership, request-local copyable state, fixed text/BF16
// geometry, original affine Q6 draft head, and reuse of the native GPU blocks.
//
// MIT License — Copyright (c) 2026 David Dalcu
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

import CMLX
import Foundation

/// Model-specific draft head. Its weights share the trunk's GPUWeights cache;
/// KV and QSA history belong exclusively to the supplied request state.
public final class QwenMTP {
    public struct State {
        /// Absolute position of cache row zero, normally one after prompt reset.
        public let positionBase: Int
        public fileprivate(set) var valid = true
        fileprivate var attention = GPUAttention.State()
        fileprivate var owner: ObjectIdentifier?

        public init(positionBase: Int = 1) { self.positionBase = positionBase }
        public var offset: Int { attention.offset }
        public var tensors: [Tensor] { attention.tensors }
        public mutating func reset(positionBase: Int = 1) { self = State(positionBase: positionBase) }
    }

    public struct Output {
        /// Post-MTP-layer, pre-mixer HC stream for the next draft depth.
        public let stream: Tensor
        /// All S rows, [1,S,vocabulary]; the controller selects its final row.
        public let logits: Tensor?
    }

    public let draftHeadBits = 6
    public let draftHeadGroupSize = 64
    public let maximumDraftDepth = 4
    /// Source bytes of the 41 MTP tensors; excludes the shared trunk embedding.
    public let sourceWeightBytes: UInt64
    private let hiddenSize: Int
    private let streams: Int
    private let vocabularySize: Int
    private let maximumPositions: Int
    private let epsilon: Float
    private let embedding, normEmbedding, normHidden, fcEmbedding, fcHidden: Tensor
    private let draftWeight, draftScales, draftBiases: Tensor
    private let attentionHC, mlpHC, mixer: GPUHyperConnection
    private let attention: GPUAttention
    private let moe: GPUMoE

    public init(weights: GPUWeights, configuration c: QwenConfiguration) throws {
        guard c.hiddenSize == 2560, c.hcCount == 4, c.hcLowRank == 320,
              c.vocabularySize == 248320, c.attentionHeads == 24,
              c.keyValueHeads == 2, c.headDimension == 256,
              c.expertCount == 512, c.expertsPerToken == 10,
              c.intermediateSize == 640, c.sharedIntermediateSize == 640,
              c.indexerBudget == 2048, c.indexerCompressRatio == 4,
              c.indexerHeadDimension == 128, c.indexerHeads == 4, c.indexerKVHeads == 1,
              c.ropeTheta == 10_000_000, c.partialRotaryFactor == 0.25,
              c.rmsNormEpsilon == 1e-6, c.maximumPositions == 262144,
              c.text["mtp_num_hidden_layers"] as? Int == 1,
              c.text["mtp_use_dedicated_embeddings"] as? Bool == false,
              let mtp = c.text["mtp"] as? [String: Any],
              mtp["num_hidden_layers"] as? Int == 1,
              mtp["layer_types"] as? [String] == ["full_attention"],
              mtp["mtp_use_hidden_state_from_layer"] is NSNull,
              mtp["rope_theta"] as? Int == 10_000_000,
              let head = c.raw["mtp_draft_head"] as? [String: Any],
              head["bits"] as? Int == 6, head["group_size"] as? Int == 64,
              head["depth"] as? Int == 4,
              weights.modelDirectory == c.modelDirectory.standardizedFileURL.resolvingSymlinksInPath() else {
            throw GPUError.invalid("MTP requires the downloaded Qwen3.8 text head: one HC/QSA/Q4 MoE layer and affine Q6/group64 draft head")
        }
        hiddenSize = c.hiddenSize; streams = c.hcCount
        vocabularySize = c.vocabularySize; maximumPositions = c.maximumPositions
        epsilon = Float(c.rmsNormEpsilon)
        let prefix = "language_model.mtp"
        let mtpNames = weights.weightMap.keys.filter { $0.hasPrefix(prefix + ".") }
        guard mtpNames.count == 41 else { throw GPUError.invalid("Expected 41 native MTP tensors") }
        sourceWeightBytes = try mtpNames.reduce(UInt64(0)) { try $0 + weights.metadata($1).byteCount }
        func load(_ name: String, _ shape: [Int], _ dtype: mlx_dtype = MLX_BFLOAT16) throws -> Tensor {
            let metadata = try weights.metadata(name)
            guard metadata.shape == shape, metadata.dtype == dtype else {
                throw GPUError.invalid("Invalid MTP tensor \(name): expected \(shape) / \(dtype)")
            }
            return try weights.tensor(name)
        }
        // GPUWeights returns the existing tensor handle when the trunk is loaded.
        embedding = try load("language_model.model.embed_tokens.weight", [c.vocabularySize, c.hiddenSize])
        normEmbedding = try load(prefix + ".pre_fc_norm_embedding.weight", [c.hiddenSize])
        normHidden = try load(prefix + ".pre_fc_norm_hidden.weight", [c.hcCount * c.hiddenSize])
        fcEmbedding = try MX.transpose(load(prefix + ".fc_embedding.weight", [c.hiddenSize, c.hiddenSize]), [1,0])
        fcHidden = try MX.transpose(load(prefix + ".fc_hidden.weight", [c.hiddenSize, c.hiddenSize]), [1,0])
        draftWeight = try load(prefix + ".draft_lm_head.weight", [c.vocabularySize, c.hiddenSize * 6 / 32], MLX_UINT32)
        draftScales = try load(prefix + ".draft_lm_head.scales", [c.vocabularySize, c.hiddenSize / 64])
        draftBiases = try load(prefix + ".draft_lm_head.biases", [c.vocabularySize, c.hiddenSize / 64])
        let layerPrefix = prefix + ".layers.0"
        let fusedHC = try GPUHyperConnectionFused()
        attentionHC = try GPUHyperConnection(weights: weights, prefix: layerPrefix + ".attn_hyper_connection", fused: fusedHC)
        mlpHC = try GPUHyperConnection(weights: weights, prefix: layerPrefix + ".mlp_hyper_connection", fused: fusedHC)
        mixer = try GPUHyperConnection(weights: weights, prefix: prefix + ".hyper_connection_mixer", withInjection: false, fused: fusedHC)
        attention = try GPUAttention(layer: 0, weights: weights, weightPrefix: layerPrefix + ".self_attn")
        moe = try GPUMoE(weights: weights, layer: 0, weightPrefix: layerPrefix + ".mlp")
    }

    public func makeState(positionBase: Int = 1) -> State {
        var state = State(positionBase: positionBase)
        state.owner = ObjectIdentifier(self)
        return state
    }

    /// Row r pairs a pre-mixer stream at position p with token p+1. Its query
    /// RoPE position is p+1 and its logits predict p+2. Full prompt history is
    /// therefore trunk streams [0..<P-1] paired with prompt tokens [1..<P].
    /// Array operations create new buffers/graphs, so copying State before a
    /// speculative step retains a rollback snapshot without global head state.
    public func forward(hidden: Tensor, tokens: [Int32], state: inout State,
                        wantLogits: Bool = true) throws -> Output {
        let n = tokens.count
        guard state.valid, state.owner == nil || state.owner == ObjectIdentifier(self),
              n > 0, n <= maximumPositions,
              state.positionBase >= 1, state.positionBase <= maximumPositions - n,
              state.offset >= 0, state.offset <= maximumPositions - state.positionBase - n,
              hidden.shape == [1,n,streams * hiddenSize], hidden.dtype == MLX_BFLOAT16,
              tokens.allSatisfy({ $0 >= 0 && $0 < vocabularySize }) else {
            throw GPUError.invalid("Invalid MTP stream/tokens/session; expected BF16 [1,S,10240] and contiguous text history")
        }
        let unsupported = Set<Int32>([248053, 248054, 248055, 248056, 248057, 248070, 248071, 248076])
        guard unsupported.isDisjoint(with: tokens) else { throw GPUError.invalid("MTP supports text inputs only") }
        do {
            var next = state
            next.owner = ObjectIdentifier(self)
            let ids = try MX.array(tokens, shape: [1,n])
            let e = try MX.take(embedding, ids, axis: 0)
            let ep = try MX.matmul(MX.rmsNorm(e, weight: normEmbedding, epsilon: epsilon), fcEmbedding)
            // ONE RMS over hc*H before splitting; per-stream RMS changes the head.
            let hn = try MX.rmsNorm(hidden, weight: normHidden, epsilon: epsilon)
            let hp = try MX.matmul(MX.reshape(hn, [1,n,streams,hiddenSize]), fcHidden)
            var h = try MX.reshape(MX.add(hp, MX.reshape(ep, [1,n,1,hiddenSize])), [1,n,streams * hiddenSize])
            let pre = try attentionHC.read(h)
            let attended = try attention.forward(pre.mixed, state: &next.attention, positionBase: next.positionBase)
            guard let injectAttention = pre.injection else { throw GPUError.invalid("Missing MTP attention HC injection") }
            h = try attentionHC.write(h, output: attended, injection: injectAttention)
            let preMLP = try mlpHC.read(h)
            let transformed = try moe.forward(preMLP.mixed).y
            guard let injectMLP = preMLP.injection else { throw GPUError.invalid("Missing MTP MLP HC injection") }
            h = try mlpHC.write(h, output: transformed, injection: injectMLP)
            let logits: Tensor?
            if wantLogits {
                let mixed = try mixer.read(h).mixed
                logits = try MX.quantizedMatmul(mixed, weight: draftWeight, scales: draftScales,
                                               biases: draftBiases, groupSize: draftHeadGroupSize, bits: draftHeadBits)
            } else { logits = nil }
            state = next
            return Output(stream: h, logits: logits)
        } catch {
            state.valid = false
            throw error
        }
    }

    /// Optional standalone evaluation; joint trunk/head evaluation may instead
    /// include State.tensors directly and discard the request on a device error.
    public func evaluate(_ outputs: [Tensor], state: inout State) throws {
        guard state.valid, state.owner == ObjectIdentifier(self) else { throw GPUError.invalid("Invalid MTP state for evaluation") }
        do { try MX.eval(outputs + state.tensors) }
        catch { state.valid = false; throw error }
    }
}
