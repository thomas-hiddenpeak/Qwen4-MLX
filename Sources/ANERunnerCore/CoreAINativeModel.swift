#if canImport(CoreAI)
import CoreAI
import CryptoKit
import Dispatch
import Foundation

private struct NativeZeroState: Decodable {
    let shape: [Int]
    let dtype: CoreMLTensorDataType
    let fill: Double
}

private struct NativeAsset: Decodable {
    let path: String
    let function: String
    let inputNames: [String]
    let outputNames: [String]
    let stateBindings: [String: String]?
    let initialState: [String: NativeZeroState]?
    let layerIndex: Int?
}

private struct NativeDenseManifest: Decodable {
    struct Layer: Decodable {
        let index: Int
        let attentionRead: NativeAsset
        let moeRead: NativeAsset
    }
    let version: Int
    let status: String
    let modelDirectory: String
    let configSHA256: String
    let assets: [String: NativeAsset]
    let layers: [Layer]
}

private struct NativeMoEManifest: Decodable {
    struct Layer: Decodable {
        let index: Int
        let path: String
        let function: String
        let inputName: String
        let outputName: String
        let expertCount: Int
        let topK: Int
        let completeExpertBank: Bool
    }
    let version: Int
    let status: String
    let modelDirectory: String
    let configSHA256: String
    let layers: [Layer]
}

/// Complete native model state at a finished token boundary. Tensor storage is
/// independent of the running model and is copied again on each restore.
/// CPU n-gram history, prompt IDs and next-token logits belong to the caller.
@available(macOS 27.0, *)
public struct CoreAINativeSnapshot: Sendable {
    fileprivate let owner: UUID
    fileprivate let attention: CoreAIHybridAttentionSnapshot
    fileprivate let pleStates: [String: NDArray]
    public let offset: Int
    /// Logical state bytes only; excludes weights, padding and caller metadata.
    public let logicalByteCount: Int
}

/// Complete S1 text forward using system CoreAI for every neural-network block.
/// Native NDArrays connect embedding, HC, attention, routed/shared MoE, PLE and
/// vocabulary projection. The caller supplies CPU SSD lookup rows and token IDs;
/// only final logits and two QSA integer counters are read back to the CPU.
///
/// This bounded prototype has independent state; it cannot import the MLX
/// runner's cache, archives or prefix state and is not a concurrent service.
@available(macOS 27.0, *)
public final class CoreAINativeModel {
    private struct DenseLayer {
        let attentionRead: CoreAIBlockRunner
        let moeRead: CoreAIBlockRunner
        let moe: CoreAIBlockRunner
        let moeInputName: String
        let moeOutputName: String
    }

    public let capacity: Int
    public let vocabularySize: Int
    public let manifestModelDirectory: URL
    public let sourceConfigSHA256: String
    public let modelLoadMilliseconds: Double
    public private(set) var offset = 0
    public private(set) var valid = true
    public private(set) var failureReason: String?
    public private(set) var successfulCalls = 0
    public private(set) var callCounts: [String: Int] = [:]
    public private(set) var predictionMillisecondsByGroup: [String: Double] = [:]
    public private(set) var lastForwardMilliseconds = 0.0
    public var totalPredictionMilliseconds: Double { predictionMillisecondsByGroup.values.reduce(0, +) }

    private let attention: CoreAIHybridAttention
    private let embedding: CoreAIBlockRunner
    private let hcWrite: CoreAIBlockRunner
    private let ple: CoreAIBlockRunner
    private let head: CoreAIBlockRunner
    private let layers: [DenseLayer]
    private let pleSpec: NativeAsset
    private var pleStates: [String: NDArray]
    private let snapshotOwner = UUID()
    private let gate = NSLock()
    private var operationInProgress = false

    public init(attentionManifest: URL, denseManifest: URL, moeManifest: URL,
                computeUnits: CoreAIComputeUnits = .gpu,
                progress: ((String, Int, Int) -> Void)? = nil) async throws {
        let started = DispatchTime.now().uptimeNanoseconds
        guard attentionManifest.isFileURL, denseManifest.isFileURL, moeManifest.isFileURL else {
            throw CoreAIBlockRunnerError.invalidModel("CoreAI native manifests must be local files")
        }
        let dense = try JSONDecoder().decode(NativeDenseManifest.self, from: Data(contentsOf: denseManifest))
        let moe = try JSONDecoder().decode(NativeMoEManifest.self, from: Data(contentsOf: moeManifest))
        guard dense.version == 1, moe.version == 1, dense.status == "complete", moe.status == "complete",
              dense.layers.count == 48, moe.layers.count == 48,
              Set(dense.layers.map(\.index)) == Set(0..<48), Set(moe.layers.map(\.index)) == Set(0..<48),
              moe.layers.allSatisfy({ $0.expertCount == 512 && $0.topK == 10 && $0.completeExpertBank }),
              Set(dense.assets.keys) == Set(["embedding", "hcWrite", "ple", "head"]) else {
            throw CoreAIBlockRunnerError.invalidModel("CoreAI native generation requires completed dense and MoE assets for all 48 layers")
        }
        let moeBase = moeManifest.standardizedFileURL.deletingLastPathComponent()
        let moeAssetPaths = moe.layers.map {
            URL(fileURLWithPath: $0.path, relativeTo: moeBase).standardizedFileURL.resolvingSymlinksInPath().path
        }
        guard Set(moeAssetPaths).count == 48 else {
            throw CoreAIBlockRunnerError.invalidModel("Each native MoE layer must have a distinct complete model asset")
        }
        let source = URL(fileURLWithPath: dense.modelDirectory).standardizedFileURL.resolvingSymlinksInPath()
        guard dense.modelDirectory.hasPrefix("/"), moe.modelDirectory.hasPrefix("/"),
              URL(fileURLWithPath: moe.modelDirectory).standardizedFileURL.resolvingSymlinksInPath() == source,
              dense.configSHA256 == moe.configSHA256 else {
            throw CoreAIBlockRunnerError.invalidModel("Dense and MoE asset source identities disagree")
        }
        let config = try Data(contentsOf: source.appendingPathComponent("config.json"))
        let digest = SHA256.hash(data: config).map { String(format: "%02x", $0) }.joined()
        guard digest == dense.configSHA256 else {
            throw CoreAIBlockRunnerError.invalidModel("CoreAI native assets do not match the source model configuration")
        }
        let attention = try await CoreAIHybridAttention(manifestURL: attentionManifest, computeUnits: computeUnits) {
            progress?("attention", $0, $1)
        }
        guard attention.manifestModelDirectory == source, attention.sourceConfigSHA256 == digest else {
            throw CoreAIBlockRunnerError.invalidModel("Attention assets have a different source identity")
        }
        let denseBase = denseManifest.standardizedFileURL.deletingLastPathComponent()
        var denseLoaded = 0
        func loadDense(_ spec: NativeAsset) async throws -> CoreAIBlockRunner {
            try Task.checkCancellation()
            let runner = try await CoreAIBlockRunner(modelURL: URL(fileURLWithPath: spec.path, relativeTo: denseBase).standardizedFileURL,
                functionName: spec.function, computeUnits: computeUnits)
            let descriptor = runner.function.descriptor
            guard Set(spec.inputNames) == Set(descriptor.inputNames), Set(spec.outputNames) == Set(descriptor.outputNames),
                  spec.inputNames.count == descriptor.inputNames.count, spec.outputNames.count == descriptor.outputNames.count else {
                throw CoreAIBlockRunnerError.invalidModel("Dense asset descriptor differs from manifest: \(spec.path)")
            }
            try Self.validateTensorFeatures(runner)
            denseLoaded += 1
            progress?("dense", denseLoaded, 100)
            return runner
        }
        let embedding = try await loadDense(dense.assets["embedding"]!)
        let hcWrite = try await loadDense(dense.assets["hcWrite"]!)
        let pleSpec = dense.assets["ple"]!
        let ple = try await loadDense(pleSpec)
        let head = try await loadDense(dense.assets["head"]!)
        guard pleSpec.layerIndex == 1, let bindings = pleSpec.stateBindings,
              bindings == ["conv_state": "next_conv_state"],
              pleSpec.initialState?["conv_state"]?.shape == [1, 9, 10240],
              pleSpec.initialState?["conv_state"]?.dtype == .float16 else {
            throw CoreAIBlockRunnerError.invalidModel("CoreAI native PLE requires layer 1 with an explicit FP16 convolution state")
        }
        try Self.requireDescriptor(embedding, input: "token", shape: [1], dtype: .int32)
        try Self.requireDescriptor(embedding, output: "stream", shape: [1, 1, 10240], dtype: .float16)
        try Self.requireDescriptor(hcWrite, input: "stream", shape: [1, 1, 10240], dtype: .float16)
        try Self.requireDescriptor(hcWrite, input: "output", shape: [1, 1, 2560], dtype: .float16)
        try Self.requireDescriptor(hcWrite, input: "injection", shape: [1, 1, 4, 1], dtype: .float16)
        try Self.requireDescriptor(hcWrite, output: "stream_out", shape: [1, 1, 10240], dtype: .float16)
        try Self.requireDescriptor(ple, input: "stream", shape: [1, 1, 10240], dtype: .float16)
        try Self.requireDescriptor(ple, input: "embedding", shape: [1, 1, 2560], dtype: .float16)
        try Self.requireDescriptor(ple, output: "stream_out", shape: [1, 1, 10240], dtype: .float16)
        try Self.requireDescriptor(head, output: "logits", shape: [1, 1, 248320], dtype: .float32)
        var loaded: [DenseLayer] = []
        let moeByIndex = Dictionary(uniqueKeysWithValues: moe.layers.map { ($0.index, $0) })
        for spec in dense.layers.sorted(by: { $0.index < $1.index }) {
            let attentionRead = try await loadDense(spec.attentionRead)
            let moeRead = try await loadDense(spec.moeRead)
            for runner in [attentionRead, moeRead] {
                try Self.requireDescriptor(runner, input: "stream", shape: [1, 1, 10240], dtype: .float16)
                try Self.requireDescriptor(runner, output: "mixed", shape: [1, 1, 2560], dtype: .float16)
                try Self.requireDescriptor(runner, output: "injection", shape: [1, 1, 4, 1], dtype: .float16)
            }
            let moeSpec = moeByIndex[spec.index]!
            guard moeSpec.inputName == "x", moeSpec.outputName == "output" else {
                throw CoreAIBlockRunnerError.invalidModel("Layer \(spec.index): unsupported MoE activation names")
            }
            let moeRunner = try await CoreAIBlockRunner(modelURL: URL(fileURLWithPath: moeSpec.path, relativeTo: moeBase).standardizedFileURL,
                functionName: moeSpec.function, computeUnits: computeUnits)
            try Self.validateTensorFeatures(moeRunner)
            guard Set(moeRunner.function.descriptor.inputNames) == Set([moeSpec.inputName]) else {
                throw CoreAIBlockRunnerError.invalidModel("Layer \(spec.index): MoE routing must be inside the CoreAI function")
            }
            try Self.requireDescriptor(moeRunner, input: moeSpec.inputName, shape: [1, 1, 2560], dtype: .float16)
            try Self.requireDescriptor(moeRunner, output: moeSpec.outputName, shape: [1, 1, 2560], dtype: .float16)
            loaded.append(DenseLayer(attentionRead: attentionRead, moeRead: moeRead, moe: moeRunner,
                moeInputName: moeSpec.inputName, moeOutputName: moeSpec.outputName))
            progress?("moe", loaded.count, 48)
        }
        self.attention = attention
        self.embedding = embedding
        self.hcWrite = hcWrite
        self.ple = ple
        self.head = head
        self.layers = loaded
        self.pleSpec = pleSpec
        self.pleStates = try Self.zeroStates(spec: pleSpec, runner: ple)
        self.capacity = attention.capacity
        self.vocabularySize = 248320
        self.manifestModelDirectory = source
        self.sourceConfigSHA256 = digest
        self.modelLoadMilliseconds = CoreAIBlockRunner.milliseconds(since: started)
    }

    /// The CPU supplies the 16 fetched PLE rows in their original order (2560
    /// values). All projections, convolution, mixing and expert routing stay in CoreAI.
    public func forward(token: Int32, pleEmbedding: [Float]) async throws -> [Float] {
        try beginOperation()
        defer { endOperation() }
        guard valid, !attention.isPoisoned else {
            throw CoreAIBlockRunnerError.invalidFixture("CoreAI native model requires reset after its previous failed token")
        }
        let started = DispatchTime.now().uptimeNanoseconds
        let callsBefore = successfulCalls
        do {
            guard token >= 0, Int(token) < vocabularySize,
                  !Set<Int32>([248053, 248054, 248055, 248056, 248057, 248070, 248071, 248076]).contains(token),
                  offset < capacity else {
                throw CoreAIBlockRunnerError.invalidFixture("Token is unsupported or CoreAI native context capacity is exhausted")
            }
            guard pleEmbedding.count == 2560, pleEmbedding.allSatisfy({ $0.isFinite && Float16($0).isFinite }) else {
                throw CoreAIBlockRunnerError.invalidFixture("PLE lookup must supply 2560 finite FP16-representable values")
            }
            try Task.checkCancellation()
            for layer in 0..<48 {
                guard try attention.nextOffset(for: layer) == offset else {
                    throw CoreAIBlockRunnerError.invalidFixture("CoreAI native attention offsets disagree before token \(offset)")
                }
            }
            let ids = try CoreAIBlockRunner.makeArray(CoreMLTensor(shape: [1], dtype: .int32, values: [Double(token)]), name: "token")
            let rows = try CoreAIBlockRunner.makeArray(CoreMLTensor(shape: [1, 1, 2560], dtype: .float16,
                values: pleEmbedding.map(Double.init)), name: "embedding")
            var stream = try Self.required(await call(embedding, inputs: ["token": ids], group: "embedding"), "stream")
            for (index, layer) in layers.enumerated() {
                try Task.checkCancellation()
                if index == pleSpec.layerIndex {
                    var inputs = pleStates
                    inputs["stream"] = stream
                    inputs["embedding"] = rows
                    let result = try await call(ple, inputs: inputs, group: "ple")
                    stream = try Self.required(result, "stream_out")
                    var nextStates: [String: NDArray] = [:]
                    for (name, outputName) in pleSpec.stateBindings! {
                        let state = try Self.required(result, outputName)
                        guard case .ndArray(let descriptor) = ple.function.descriptor.inputDescriptor(of: name) else {
                            throw CoreAIBlockRunnerError.invalidModel("Missing PLE state input \(name)")
                        }
                        try CoreAIBlockRunner.validate(state, descriptor: descriptor, name: name)
                        nextStates[name] = state
                    }
                    pleStates = nextStates
                }
                let preAttention = try await call(layer.attentionRead, inputs: ["stream": stream], group: "hc_read")
                let attentionOutput = try await attention.forwardArray(layer: index,
                    input: Self.required(preAttention, "mixed"), offset: offset)
                record(group: "attention", milliseconds: attention.lastPredictionMilliseconds)
                stream = try Self.required(await call(hcWrite, inputs: ["stream": stream, "output": attentionOutput,
                    "injection": Self.required(preAttention, "injection")], group: "hc_write"), "stream_out")
                let preMoE = try await call(layer.moeRead, inputs: ["stream": stream], group: "hc_read")
                let moeOutput = try await call(layer.moe, inputs: [layer.moeInputName: Self.required(preMoE, "mixed")], group: "moe")
                stream = try Self.required(await call(hcWrite, inputs: ["stream": stream,
                    "output": Self.required(moeOutput, layer.moeOutputName),
                    "injection": Self.required(preMoE, "injection")], group: "hc_write"), "stream_out")
            }
            let prediction = try await call(head, inputs: ["stream": stream], group: "head")
            let logits = try CoreAIBlockRunner.read(Self.required(prediction, "logits"), name: "logits")
            guard logits.shape == [1, 1, vocabularySize], logits.dtype == .float32,
                  successfulCalls - callsBefore == 291 else {
                throw CoreAIBlockRunnerError.invalidFixture("CoreAI native token returned invalid logits or did not complete every model block")
            }
            try Task.checkCancellation()
            offset += 1
            lastForwardMilliseconds = CoreAIBlockRunner.milliseconds(since: started)
            return logits.values.map(Float.init)
        } catch {
            valid = false
            failureReason = "Token at offset \(offset): \(error)"
            throw error
        }
    }

    /// Reads state shapes/dtypes only. Callers can reserve a cache budget before
    /// checkpoint() allocates independent copies of all state tensors.
    public func stateByteCount() throws -> Int {
        try beginOperation()
        defer { endOperation() }
        return try CoreAITensorCopy.addingByteCounts(attention.stateByteCount(),
                                                    CoreAITensorCopy.logicalByteCount(pleStates))
    }

    public func checkpoint() throws -> CoreAINativeSnapshot {
        try beginOperation()
        defer { endOperation() }
        guard valid, !attention.isPoisoned, offset >= 0, offset <= capacity else {
            throw CoreAIBlockRunnerError.invalidFixture("Cannot checkpoint a failed or incomplete CoreAI native token")
        }
        try validatePLECheckpoint(pleStates)
        let attentionState = try attention.checkpoint()
        guard attentionState.offset == offset else {
            throw CoreAIBlockRunnerError.invalidFixture("Native model and attention checkpoint offsets disagree")
        }
        let bytes = try CoreAITensorCopy.addingByteCounts(attentionState.logicalByteCount,
                                                        CoreAITensorCopy.logicalByteCount(pleStates))
        let copiedPLE = try CoreAITensorCopy.deepCopy(pleStates)
        return CoreAINativeSnapshot(owner: snapshotOwner, attention: attentionState, pleStates: copiedPLE,
                                   offset: offset, logicalByteCount: bytes)
    }

    /// Foreign or rejected snapshots do not alter the live model. A successful
    /// restore may recover a failed session, and clears all timing/call counters.
    public func restore(_ snapshot: CoreAINativeSnapshot) throws {
        try beginOperation()
        defer { endOperation() }
        guard snapshot.owner == snapshotOwner else {
            throw CoreAIBlockRunnerError.invalidFixture("Cannot restore a native checkpoint from another model instance")
        }
        guard snapshot.offset >= 0, snapshot.offset <= capacity,
              snapshot.attention.offset == snapshot.offset else {
            throw CoreAIBlockRunnerError.invalidFixture("Native checkpoint offsets are invalid or inconsistent")
        }
        try validatePLECheckpoint(snapshot.pleStates)
        let bytes = try CoreAITensorCopy.addingByteCounts(snapshot.attention.logicalByteCount,
                                                        CoreAITensorCopy.logicalByteCount(snapshot.pleStates))
        guard bytes == snapshot.logicalByteCount else {
            throw CoreAIBlockRunnerError.invalidFixture("Native checkpoint logical size does not match its state tensors")
        }
        // Finish every potentially throwing PLE operation before attention's
        // atomic restore. Nothing after its successful return can throw.
        let restoredPLE = try CoreAITensorCopy.deepCopy(snapshot.pleStates)
        try attention.restore(snapshot.attention)
        pleStates = restoredPLE
        offset = snapshot.offset
        valid = true
        failureReason = nil
        successfulCalls = 0
        callCounts = [:]
        predictionMillisecondsByGroup = [:]
        lastForwardMilliseconds = 0
    }

    private func validatePLECheckpoint(_ states: [String: NDArray]) throws {
        guard let bindings = pleSpec.stateBindings, Set(states.keys) == Set(bindings.keys) else {
            throw CoreAIBlockRunnerError.invalidFixture("Native checkpoint PLE state names differ from the model")
        }
        for (name, state) in states {
            guard case .ndArray(let descriptor) = ple.function.descriptor.inputDescriptor(of: name) else {
                throw CoreAIBlockRunnerError.invalidFixture("Native checkpoint PLE input \(name) is missing")
            }
            try CoreAIBlockRunner.validate(state, descriptor: descriptor, name: name)
        }
    }

    public func reset() throws {
        try beginOperation()
        defer { endOperation() }
        do {
            let states = try Self.zeroStates(spec: pleSpec, runner: ple)
            try attention.reset()
            pleStates = states
            offset = 0
            valid = true
            failureReason = nil
            successfulCalls = 0
            callCounts = [:]
            predictionMillisecondsByGroup = [:]
            lastForwardMilliseconds = 0
        } catch {
            valid = false
            failureReason = "Reset failed: \(error)"
            throw error
        }
    }

    private func call(_ runner: CoreAIBlockRunner, inputs: [String: NDArray], group: String) async throws -> [String: NDArray] {
        try Task.checkCancellation()
        let validated = try runner.makeInputs([:], retainedInputs: inputs)
        let start = DispatchTime.now().uptimeNanoseconds
        var prediction = try await runner.function.run(inputs: validated)
        let duration = CoreAIBlockRunner.milliseconds(since: start)
        let descriptor = runner.function.descriptor
        guard Set(prediction.names) == Set(descriptor.outputNames) else {
            throw CoreAIBlockRunnerError.invalidFixture("\(group): returned outputs differ from function descriptor")
        }
        var result: [String: NDArray] = [:]
        for name in descriptor.outputNames {
            guard let array = prediction.remove(name)?.ndArray,
                  case .ndArray(let declared) = descriptor.outputDescriptor(of: name) else {
                throw CoreAIBlockRunnerError.invalidFixture("\(group): missing native tensor output \(name)")
            }
            try CoreAIBlockRunner.validate(array, descriptor: declared, name: name)
            result[name] = array
        }
        record(group: group, milliseconds: duration)
        return result
    }

    private func record(group: String, milliseconds: Double) {
        successfulCalls += 1
        callCounts[group, default: 0] += 1
        predictionMillisecondsByGroup[group, default: 0] += milliseconds
    }

    private static func required(_ outputs: [String: NDArray], _ name: String) throws -> NDArray {
        guard let array = outputs[name] else {
            throw CoreAIBlockRunnerError.invalidFixture("Required native CoreAI output \(name) is missing")
        }
        return array
    }

    private static func validateTensorFeatures(_ runner: CoreAIBlockRunner) throws {
        let descriptor = runner.function.descriptor
        guard descriptor.stateNames.isEmpty else {
            throw CoreAIBlockRunnerError.unsupportedFeature("Native model assets must expose their persistent states as tensor inputs/outputs")
        }
        for name in descriptor.inputNames {
            guard case .ndArray(let tensor) = descriptor.inputDescriptor(of: name), tensor.interleaveLayout == nil,
                  !tensor.hasDynamicShape else {
                throw CoreAIBlockRunnerError.unsupportedFeature("\(name): native S1 requires fixed non-interleaved tensor inputs")
            }
            _ = try CoreAIBlockRunner.dtype(tensor.scalarType, name: name)
        }
        for name in descriptor.outputNames {
            guard case .ndArray(let tensor) = descriptor.outputDescriptor(of: name), tensor.interleaveLayout == nil,
                  !tensor.hasDynamicShape else {
                throw CoreAIBlockRunnerError.unsupportedFeature("\(name): native S1 requires fixed non-interleaved tensor outputs")
            }
            _ = try CoreAIBlockRunner.dtype(tensor.scalarType, name: name)
        }
    }

    private static func requireDescriptor(_ runner: CoreAIBlockRunner, input: String? = nil,
                                          output: String? = nil, shape: [Int], dtype: NDArray.ScalarType) throws {
        let declared = input.map { runner.function.descriptor.inputDescriptor(of: $0) }
            ?? output.map { runner.function.descriptor.outputDescriptor(of: $0) }
        guard case .ndArray(let descriptor) = declared ?? nil,
              descriptor.shape == shape, descriptor.scalarType == dtype, descriptor.interleaveLayout == nil else {
            throw CoreAIBlockRunnerError.invalidModel("Native \(input ?? output ?? "unknown") descriptor differs from the supported architecture")
        }
    }

    private static func zeroStates(spec: NativeAsset, runner: CoreAIBlockRunner) throws -> [String: NDArray] {
        guard let initial = spec.initialState, let bindings = spec.stateBindings,
              Set(initial.keys) == Set(bindings.keys), Set(bindings.values).count == bindings.count else {
            throw CoreAIBlockRunnerError.invalidModel("Invalid native state initialization metadata")
        }
        var states: [String: NDArray] = [:]
        for (name, spec) in initial {
            guard spec.fill == 0, spec.dtype == .float16, spec.shape == [1, 9, 10240],
                  case .ndArray(let descriptor) = runner.function.descriptor.inputDescriptor(of: name),
                  descriptor.shape == spec.shape, descriptor.scalarType == .float16,
                  descriptor.interleaveLayout == nil else {
                throw CoreAIBlockRunnerError.invalidModel("Invalid PLE initial-state metadata or descriptor")
            }
            let array = try CoreAITensorCopy.zeroArray(shape: spec.shape, scalarType: descriptor.scalarType)
            try CoreAIBlockRunner.validate(array, descriptor: descriptor, name: name)
            states[name] = array
        }
        return states
    }

    private func beginOperation() throws {
        gate.lock()
        defer { gate.unlock() }
        guard !operationInProgress else {
            throw CoreAIBlockRunnerError.invalidFixture("CoreAI native model already has an operation in progress")
        }
        operationInProgress = true
    }

    private func endOperation() {
        gate.lock()
        operationInProgress = false
        gate.unlock()
    }
}
#endif
