import Foundation

/// The architecture and asset boundary shared by the experiment's runners.
/// This descriptor does not load matrix weights or create an inference session.
public struct ModelManifest: Decodable, Sendable {
    public let modelType: String
    public let textConfig: TextConfig
    public let ngramTable: TableConfig

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
        case ngramTable = "ngram_table"
    }

    public struct TextConfig: Decodable, Sendable {
        public let hiddenSize: Int
        public let numHiddenLayers: Int
        public let numExperts: Int
        public let numExpertsPerToken: Int
        public let layerTypes: [String]
        public let vocabularySize: UInt32
        public let eosTokenID: UInt32
        public let ngramSize: Int
        public let headsPerNGram: Int
        public let ngramVocabularyBase: UInt64
        public let vocabularyDivisor: UInt64
        public let seed: UInt64?

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case numHiddenLayers = "num_hidden_layers"
            case numExperts = "num_experts"
            case numExpertsPerToken = "num_experts_per_tok"
            case layerTypes = "layer_types"
            case vocabularySize = "vocab_size"
            case eosTokenID = "eos_token_id"
            case ngramSize = "ngram_size"
            case headsPerNGram = "heads_per_ngram"
            case ngramVocabularyBase = "ngram_vocab_size_base"
            case vocabularyDivisor = "make_ngram_vocab_size_divisible_by"
            case seed
        }
    }

    public struct TableConfig: Decodable, Sendable {
        public let file: String
        public let format: String
        public let scale: Float
    }

    public static func load(from directory: URL) throws -> ModelManifest {
        let manifest = try JSONDecoder().decode(
            Self.self,
            from: Data(contentsOf: directory.appendingPathComponent("config.json"))
        )
        guard manifest.modelType == "qwen4_exp" else {
            throw ManifestError.invalid("Expected qwen4_exp; got \(manifest.modelType)")
        }
        let text = manifest.textConfig
        guard text.numHiddenLayers > 0, text.hiddenSize > 0,
            text.layerTypes.count == text.numHiddenLayers,
            text.layerTypes.allSatisfy({ ["linear_attention", "full_attention"].contains($0) }),
            text.numExperts > 0, text.numExpertsPerToken > 0,
            text.numExpertsPerToken <= text.numExperts,
            text.vocabularySize > 0, text.eosTokenID < text.vocabularySize,
            text.ngramSize >= 2, text.headsPerNGram > 0,
            text.vocabularyDivisor > 0
        else {
            throw ManifestError.invalid("Inconsistent or unsupported text architecture")
        }
        guard manifest.ngramTable.format == "fp8_e4m3fn",
            manifest.ngramTable.scale.isFinite, manifest.ngramTable.scale > 0,
            !manifest.ngramTable.file.isEmpty
        else {
            throw ManifestError.invalid("Expected an FP8 E4M3FN table with positive finite scale")
        }
        _ = try manifest.tableURL(in: directory)
        return manifest
    }

    public func tableURL(in directory: URL) throws -> URL {
        let root = directory.resolvingSymlinksInPath().standardizedFileURL
        let table = directory.appendingPathComponent(ngramTable.file)
            .resolvingSymlinksInPath().standardizedFileURL
        guard table.path.hasPrefix(root.path + "/") else {
            throw ManifestError.invalid("n-gram table must be inside the model directory")
        }
        return table
    }

    public enum ManifestError: Error, LocalizedError {
        case invalid(String)

        public var errorDescription: String? {
            switch self {
            case .invalid(let reason): reason
            }
        }
    }
}
