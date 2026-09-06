import Foundation
import CoreFoundation

public struct QwenConfiguration {
    public let raw: [String: Any]
    public let text: [String: Any]
    public let modelDirectory: URL
    public let hiddenSize, layerCount, vocabularySize, attentionHeads, keyValueHeads, headDimension: Int
    public let expertCount, expertsPerToken, intermediateSize, sharedIntermediateSize: Int
    public let hcCount, hcLowRank, fullAttentionInterval: Int
    public let linearConvKernel, linearKeyHeadDimension, linearValueHeadDimension, linearKeyHeads, linearValueHeads: Int
    public let indexerBudget, indexerCompressRatio, indexerHeadDimension, indexerHeads, indexerKVHeads: Int
    public let ngramSize, ngramHeadsPerOrder, ngramVocabularyBase, ngramParts, ngramDivisor: Int
    public let pleEmbeddingDimension, pleConvKernel, maximumPositions: Int
    /// Configuration uses one-based layer IDs. pleLayerIndices is zero-based.
    public let pleLayerIDs: [Int]
    public var pleLayerIndices: [Int] { pleLayerIDs.map { $0 - 1 } }
    public let layerTypes: [String]
    public let rmsNormEpsilon, ropeTheta, partialRotaryFactor: Double
    public let quantizationBits, quantizationGroupSize: Int
    public let quantizationMode: String
    public let bosTokenID, eosTokenID: Int
    public let ngramTableFile: String
    public let ngramScale: Double

    public init(modelDirectory: URL) throws {
        self.modelDirectory = modelDirectory
        guard let root = try JSONSerialization.jsonObject(with: Data(contentsOf: modelDirectory.appendingPathComponent("config.json"))) as? [String: Any],
              root["model_type"] as? String == "qwen4_exp", let t = root["text_config"] as? [String: Any] else {
            throw GPUWeightError.invalid("Expected qwen4_exp config with text_config")
        }
        raw = root; text = t
        func integer(_ key: String) throws -> Int { try Self.integer(t, key) }
        hiddenSize = try integer("hidden_size"); layerCount = try integer("num_hidden_layers")
        vocabularySize = try integer("vocab_size"); attentionHeads = try integer("num_attention_heads")
        keyValueHeads = try integer("num_key_value_heads"); headDimension = try integer("head_dim")
        expertCount = try integer("num_experts"); expertsPerToken = try integer("num_experts_per_tok")
        intermediateSize = try integer("moe_intermediate_size"); sharedIntermediateSize = try integer("shared_expert_intermediate_size")
        hcCount = try integer("hc_count"); hcLowRank = try integer("hc_lowrank")
        fullAttentionInterval = try integer("full_attention_interval")
        linearConvKernel = try integer("linear_conv_kernel_dim"); linearKeyHeadDimension = try integer("linear_key_head_dim")
        linearValueHeadDimension = try integer("linear_value_head_dim"); linearKeyHeads = try integer("linear_num_key_heads")
        linearValueHeads = try integer("linear_num_value_heads")
        indexerBudget = try integer("indexer_budget"); indexerCompressRatio = try integer("indexer_compress_ratio")
        indexerHeadDimension = try integer("indexer_head_dim"); indexerHeads = try integer("indexer_n_heads")
        indexerKVHeads = try integer("indexer_kv_heads")
        ngramSize = try integer("ngram_size"); ngramHeadsPerOrder = try integer("heads_per_ngram")
        ngramVocabularyBase = try integer("ngram_vocab_size_base"); ngramParts = try integer("split_ngram_parts")
        ngramDivisor = try integer("make_ngram_vocab_size_divisible_by")
        pleEmbeddingDimension = try integer("ple_embed_dim"); pleConvKernel = try integer("ple_conv_kernel_size")
        maximumPositions = try integer("max_position_embeddings")
        bosTokenID = try integer("bos_token_id"); eosTokenID = try integer("eos_token_id")
        let validatedLayerCount = layerCount
        guard let ids = t["ple_layer_ids"] as? [Int], ids.allSatisfy({ $0 > 0 && $0 <= validatedLayerCount }),
              let types = t["layer_types"] as? [String], types.count == layerCount,
              types.allSatisfy({ $0 == "full_attention" || $0 == "linear_attention" }) else {
            throw GPUWeightError.invalid("Invalid Qwen layer_types/ple_layer_ids")
        }
        pleLayerIDs = ids; layerTypes = types
        rmsNormEpsilon = try Self.number(t, "rms_norm_eps")
        let rope = t["rope_parameters"] as? [String: Any] ?? [:]
        ropeTheta = try Self.number(rope, "rope_theta")
        partialRotaryFactor = try Self.number(rope, "partial_rotary_factor")
        guard let q = root["quantization"] as? [String: Any], let mode = q["mode"] as? String,
              let table = root["ngram_table"] as? [String: Any], let tableFile = table["file"] as? String,
              table["format"] as? String == "fp8_e4m3fn" else { throw GPUWeightError.invalid("Missing quantization/ngram configuration") }
        quantizationBits = try Self.integer(q, "bits"); quantizationGroupSize = try Self.integer(q, "group_size")
        quantizationMode = mode; ngramTableFile = tableFile; ngramScale = try Self.number(table, "scale")
        guard expertsPerToken <= expertCount, attentionHeads % keyValueHeads == 0,
              linearValueHeads % linearKeyHeads == 0, partialRotaryFactor <= 1,
              mode == "affine", quantizationBits == 4, quantizationGroupSize == 64,
              t["hidden_act"] as? String == "silu", t["output_gate_type"] as? String == "sigmoid" else {
            throw GPUWeightError.invalid("Unsupported Qwen architecture/quantization combination")
        }
    }

    public func integer(_ key: String) throws -> Int { try Self.integer(text, key) }
    public func number(_ key: String) throws -> Double { try Self.number(text, key) }
    private static func integer(_ object: [String: Any], _ key: String) throws -> Int {
        guard let n = object[key] as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID(),
              let value = Int(n.stringValue), value > 0 else { throw GPUWeightError.invalid("Missing/invalid positive integer \(key)") }
        return value
    }
    private static func number(_ object: [String: Any], _ key: String) throws -> Double {
        guard let n = object[key] as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID(),
              n.doubleValue.isFinite, n.doubleValue > 0 else { throw GPUWeightError.invalid("Missing/invalid positive number \(key)") }
        return n.doubleValue
    }
}
