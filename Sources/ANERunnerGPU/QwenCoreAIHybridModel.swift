#if canImport(CoreAI)
import ANERunnerCore
import CMLX
import Darwin
import CryptoKit
import Dispatch
import Foundation

/// Experimental complete text forward: every GDN/QSA sublayer runs through
/// system CoreAI, while HC, original packed-Q4 MoE, PLE, embeddings and the
/// vocabulary head retain the independent MLX implementation.
///
/// Each call processes one token through all 48 layers. Activation bridges
/// materialize host Float arrays and round at the CoreAI FP16 / MLX BF16 boundary.
/// This is a correctness-first integration, not an optimized or pure-CoreAI
/// language-model backend. Its state is deliberately separate from QwenModel,
/// prefix caches, paged KV, SSD archives and the serving scheduler.
@available(macOS 27.0, *)
public final class QwenCoreAIHybridModel {
    public struct ForwardStatistics: Codable, Sendable {
        public let offsetBefore: Int
        public let offsetAfter: Int
        public let wallSeconds: Double
        /// Includes evaluating preceding lazy MLX work, not just the host copy.
        public let attentionInputMaterializationSeconds: Double
        /// Tensor creation/submission time; GPU work may complete later.
        public let attentionOutputImportSeconds: Double
        public let coreAIPredictionSeconds: Double
        public let coreAIInputSeconds: Double
        public let coreAIOutputSeconds: Double
        public let coreAICalls: Int
        public let ssdWaitSeconds: Double
        public let ssdLogicalBytes: Int
    }

    private struct Layer {
        let attentionHC: GPUHyperConnection
        let moeHC: GPUHyperConnection
        let moe: GPUMoE
        let ple: GPUPLE?
    }

    public let configuration: QwenConfiguration
    public let weights: GPUWeights
    public let capacity: Int
    public let layerCount: Int
    public private(set) var lastForwardStatistics: ForwardStatistics?

    private let attentionBackend: CoreAIHybridAttention
    private let embedding: Tensor
    private let head: Tensor
    private let mixer: GPUHyperConnection
    private let outputMask: Tensor
    private let layers: [Layer]
    private var pleStates: [GPUPLE.State]
    private let operationGate = NSLock()
    private var operationInProgress = false
    private var currentOffset = 0
    private var sessionValid = true
    private let mlxThread = pthread_self()

    public var offset: Int { operationGate.withLock { currentOffset } }
    public var valid: Bool { operationGate.withLock { sessionValid } }

    public init(modelDirectory: URL, attentionBackend: CoreAIHybridAttention,
                progress: ((Int, Int) -> Void)? = nil) throws {
        let sourceDirectory = modelDirectory.standardizedFileURL.resolvingSymlinksInPath()
        guard sourceDirectory == attentionBackend.manifestModelDirectory else {
            throw GPUError.invalid("CoreAI attention assets and MLX weights come from different model directories")
        }
        let sourceConfig = try Data(contentsOf: sourceDirectory.appendingPathComponent("config.json"))
        let configDigest = SHA256.hash(data: sourceConfig).map { String(format: "%02x", $0) }.joined()
        guard configDigest == attentionBackend.sourceConfigSHA256 else {
            throw GPUError.invalid("CoreAI attention assets were exported from a different model configuration")
        }
        let configuration = try QwenConfiguration(modelDirectory: modelDirectory)
        guard configuration.hiddenSize == 2560, configuration.layerCount == 48,
              configuration.hcCount == 4, configuration.hcLowRank == 320,
              configuration.pleLayerIndices == [1], configuration.attentionHeads == 24,
              configuration.keyValueHeads == 2, configuration.headDimension == 256 else {
            throw GPUError.invalid("CoreAI hybrid supports the downloaded Qwen3.8 Flash-Next text architecture only")
        }
        guard attentionBackend.loadedLayerIndices == Array(0..<configuration.layerCount),
              !attentionBackend.isPoisoned, attentionBackend.capacity > 0,
              attentionBackend.capacity <= configuration.maximumPositions else {
            throw GPUError.invalid("CoreAI hybrid requires a fresh complete 48-layer attention backend")
        }
        for layer in 0..<configuration.layerCount {
            let expected: String
            switch configuration.layerTypes[layer] {
            case "linear_attention": expected = "gdn"
            case "full_attention": expected = "qsa"
            default: throw GPUError.invalid("Unsupported CoreAI hybrid attention kind at layer \(layer)")
            }
            guard attentionBackend.layerKinds[layer] == expected,
                  try attentionBackend.nextOffset(for: layer) == 0 else {
                throw GPUError.invalid("CoreAI asset layer kind or initial offset differs from source configuration at layer \(layer)")
            }
        }
        self.configuration = configuration
        self.attentionBackend = attentionBackend
        capacity = attentionBackend.capacity
        layerCount = configuration.layerCount
        let weights = try GPUWeights(modelDirectory: modelDirectory)
        self.weights = weights
        let embedding = try weights.tensor("language_model.model.embed_tokens.weight")
        guard embedding.shape == [configuration.vocabularySize, configuration.hiddenSize],
              embedding.dtype == MLX_BFLOAT16 else {
            throw GPUError.invalid("CoreAI hybrid requires the original BF16 token embedding")
        }
        self.embedding = embedding
        let hcFused = try GPUHyperConnectionFused()
        var loaded: [Layer] = []
        for layer in 0..<configuration.layerCount {
            let prefix = "language_model.model.layers.\(layer)"
            let attentionHC = try GPUHyperConnection(weights: weights,
                prefix: prefix + ".attn_hyper_connection", fused: hcFused)
            let moeHC = try GPUHyperConnection(weights: weights,
                prefix: prefix + ".mlp_hyper_connection", fused: hcFused)
            let moe = try GPUMoE(weights: weights, layer: layer, prefillAccumulation: .reference)
            let ple = configuration.pleLayerIndices.contains(layer)
                ? try GPUPLE(weights: weights, configuration: configuration, layer: layer, ordinal: 0)
                : nil
            loaded.append(Layer(attentionHC: attentionHC, moeHC: moeHC, moe: moe, ple: ple))
            progress?(layer + 1, configuration.layerCount)
        }
        layers = loaded
        pleStates = (0..<configuration.layerCount).map { _ in GPUPLE.State() }
        mixer = try GPUHyperConnection(weights: weights,
            prefix: "language_model.model.hyper_connection_mixer", withInjection: false, fused: hcFused)
        let sourceHead = try weights.tensor("language_model.lm_head.weight")
        guard sourceHead.shape == [configuration.vocabularySize, configuration.hiddenSize],
              sourceHead.dtype == MLX_BFLOAT16 else {
            throw GPUError.invalid("CoreAI hybrid requires the original BF16 vocabulary projection")
        }
        head = try MX.transpose(sourceHead, [1, 0])
        var mask = [Float](repeating: 0, count: configuration.vocabularySize)
        for token in try QwenTokenizer(modelDirectory: modelDirectory).reservedOutputTokenIDs {
            guard token >= 0, token < configuration.vocabularySize else {
                throw GPUError.invalid("Invalid reserved output token")
            }
            mask[Int(token)] = 1
        }
        outputMask = try MX.array(mask, shape: [configuration.vocabularySize], dtype: MLX_BOOL)
    }

    /// Reset both runtime domains. No cache entry from the existing MLX runner
    /// can be imported here, including after a partially completed token fails.
    public func reset() throws {
        try requireMLXThread()
        try operationGate.withLock {
            guard !operationInProgress else {
                throw GPUError.invalid("Cannot reset CoreAI hybrid during an active forward")
            }
            operationInProgress = true
        }
        defer { operationGate.withLock { operationInProgress = false } }
        do {
            try MX.synchronize()
            try attentionBackend.reset()
            pleStates = (0..<layerCount).map { _ in GPUPLE.State() }
            lastForwardStatistics = nil
            operationGate.withLock { currentOffset = 0; sessionValid = true }
        } catch {
            operationGate.withLock { sessionValid = false }
            throw error
        }
    }

    /// Match the text runner's reserved-token suppression without modifying raw
    /// logits, which remain available for teacher-forced numerical comparisons.
    public func greedyToken(_ logits: Tensor) throws -> Tensor {
        try requireMLXThread()
        return try MX.argmax(MX.whereSelect(outputMask, MX.scalar(-.infinity, logits.dtype), logits))
    }

    /// The asynchronous suspension points do not permit a second caller to
    /// mutate either domain. After any execution failure, reset is mandatory.
    public func forward(token: Int32) async throws -> Tensor {
        try requireMLXThread()
        guard token >= 0, token < configuration.vocabularySize else {
            throw GPUError.invalid("CoreAI hybrid token is outside the model vocabulary")
        }
        let unsupported: Set<Int32> = [248053, 248054, 248055, 248056, 248057, 248070, 248071, 248076]
        guard !unsupported.contains(token) else {
            throw GPUError.invalid("Multimodal inputs are not supported by CoreAI hybrid text inference")
        }
        let before = try operationGate.withLock {
            guard !operationInProgress else {
                throw GPUError.invalid("CoreAI hybrid accepts one active forward at a time")
            }
            guard sessionValid, !attentionBackend.isPoisoned else {
                throw GPUError.invalid("Reset the failed CoreAI hybrid session before reuse")
            }
            guard currentOffset < capacity else {
                throw GPUError.invalid("CoreAI hybrid fixed context capacity is exhausted")
            }
            operationInProgress = true
            return currentOffset
        }
        defer { operationGate.withLock { operationInProgress = false } }
        let started = DispatchTime.now().uptimeNanoseconds
        let predictionBefore = attentionBackend.totalPredictionMilliseconds
        let inputBefore = attentionBackend.totalInputMilliseconds
        let outputBefore = attentionBackend.totalOutputMilliseconds
        let callsBefore = attentionBackend.successfulCalls
        var pending: [Int: PLEReadTask] = [:]
        defer { for task in pending.values { task.drain() } }
        var ssdWaitSeconds = 0.0, ssdLogicalBytes = 0
        var inputMaterializationSeconds = 0.0, outputImportSeconds = 0.0
        do {
            try Task.checkCancellation()
            for layer in layers.indices {
                guard try attentionBackend.nextOffset(for: layer) == before else {
                    throw GPUError.invalid("CoreAI hybrid attention offsets disagree before token \(before)")
                }
                if let ple = layers[layer].ple {
                    pending[layer] = try ple.prepare(tokens: [token], state: &pleStates[layer])
                }
            }
            let ids = try MX.array([token], shape: [1, 1])
            let embedded = try MX.take(embedding, ids, axis: 0)
            var stream = try MX.tile(embedded, [1, 1, configuration.hcCount])
            for index in layers.indices {
                try Task.checkCancellation()
                let layer = layers[index]
                if let ple = layer.ple, let task = pending[index] {
                    try MX.asyncEval([stream])
                    let start = DispatchTime.now().uptimeNanoseconds
                    let rows = try task.wait()
                    ssdWaitSeconds += Self.seconds(since: start)
                    ssdLogicalBytes += task.logicalBytes
                    let pleEmbedding = try MX.array(rows,
                        shape: [1, 1, configuration.pleEmbeddingDimension], dtype: MLX_BFLOAT16)
                    let addition = try ple.forward(stream, embedding: pleEmbedding, state: &pleStates[index])
                    stream = try MX.add(stream, addition)
                    pending.removeValue(forKey: index)
                }
                let attentionRead = try layer.attentionHC.read(stream)
                guard let attentionInjection = attentionRead.injection else {
                    throw GPUError.invalid("Missing attention hyper-connection injection")
                }
                let inputStart = DispatchTime.now().uptimeNanoseconds
                let input = try attentionRead.mixed.floats()
                inputMaterializationSeconds += Self.seconds(since: inputStart)
                let attentionValues = try await attentionBackend.forward(layer: index, input: input, offset: before)
                try requireMLXThread()
                try Task.checkCancellation()
                guard attentionValues.count == configuration.hiddenSize,
                      attentionValues.allSatisfy(\.isFinite) else {
                    throw GPUError.invalid("CoreAI attention output is invalid at layer \(index)")
                }
                let importStart = DispatchTime.now().uptimeNanoseconds
                let attentionOutput = try MX.array(attentionValues,
                    shape: [1, 1, configuration.hiddenSize], dtype: MLX_BFLOAT16)
                outputImportSeconds += Self.seconds(since: importStart)
                stream = try layer.attentionHC.write(stream, output: attentionOutput, injection: attentionInjection)
                let moeRead = try layer.moeHC.read(stream)
                guard let moeInjection = moeRead.injection else {
                    throw GPUError.invalid("Missing MoE hyper-connection injection")
                }
                let moeOutput = try layer.moe.forward(moeRead.mixed).y
                stream = try layer.moeHC.write(stream, output: moeOutput, injection: moeInjection)
            }
            let mixed = try mixer.read(stream).mixed
            let logits = try MX.linear(mixed, head, verification: nil)
            // A successful token has materialized every lazy MLX result and PLE
            // recurrence, while CoreAI already owns its completed native states.
            try MX.eval([logits] + pleStates.flatMap(\.tensors))
            try MX.synchronize()
            try Task.checkCancellation()
            let calls = attentionBackend.successfulCalls - callsBefore
            guard calls == layerCount else {
                throw GPUError.invalid("CoreAI hybrid token did not execute all \(layerCount) attention layers")
            }
            lastForwardStatistics = ForwardStatistics(offsetBefore: before, offsetAfter: before + 1,
                wallSeconds: Self.seconds(since: started),
                attentionInputMaterializationSeconds: inputMaterializationSeconds,
                attentionOutputImportSeconds: outputImportSeconds,
                coreAIPredictionSeconds: (attentionBackend.totalPredictionMilliseconds - predictionBefore) * 1e-3,
                coreAIInputSeconds: (attentionBackend.totalInputMilliseconds - inputBefore) * 1e-3,
                coreAIOutputSeconds: (attentionBackend.totalOutputMilliseconds - outputBefore) * 1e-3,
                coreAICalls: calls, ssdWaitSeconds: ssdWaitSeconds, ssdLogicalBytes: ssdLogicalBytes)
            operationGate.withLock { currentOffset = before + 1 }
            return logits
        } catch {
            operationGate.withLock { sessionValid = false }
            lastForwardStatistics = nil
            throw error
        }
    }

    private static func seconds(since start: UInt64) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - start) * 1e-9
    }

    private func requireMLXThread() throws {
        guard pthread_equal(mlxThread, pthread_self()) != 0 else {
            throw GPUError.invalid("CoreAI hybrid MLX work must remain on its initializing native thread; use CoreAIHybridExecutor")
        }
    }
}
#endif
