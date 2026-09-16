#if canImport(CoreAI)
import CoreAI
import CryptoKit
import Dispatch
import Foundation

private struct PhaseZeroState: Decodable {
    let shape: [Int]
    let dtype: CoreMLTensorDataType
    let fill: Double
}

private struct PhaseAsset: Decodable {
    let path: String
    let function: String
    let prefillFunction: String
    let inputNames: [String]
    let outputNames: [String]
}

private struct PhaseManifest: Decodable {
    struct Layer: Decodable {
        let index: Int
        let kind: String
        let path: String
        let function: String
        let prefillFunction: String
        let inputNames: [String]
        let outputNames: [String]
        let inputName: String
        let outputName: String
        let expertCount: Int
        let topK: Int
        let completeExpertBank: Bool
        let stateBindings: [String: String]
        let initialState: [String: PhaseZeroState]
        let hasPLE: Bool
        var asset: PhaseAsset {
            PhaseAsset(path: path, function: function, prefillFunction: prefillFunction,
                       inputNames: inputNames, outputNames: outputNames)
        }
    }
    let version: Int
    let backend: String
    let status: String
    let completeModelLayerSet: Bool
    let capacity: Int
    let tokenChunk: Int
    let hiddenSize: Int
    let streamCount: Int
    let vocabularySize: Int
    let modelDirectory: String
    let configSHA256: String
    let assets: [String: PhaseAsset]
    let layers: [Layer]
}

/// Independent state at a completed S1/S4 boundary. Prompt IDs, CPU n-gram
/// history and the last logits remain the caller's responsibility.
@available(macOS 27.0, *)
public struct CoreAIPhaseSnapshot: Sendable {
    fileprivate struct LayerState: Sendable {
        let states: [String: NDArray]
        let offset: Int
    }
    fileprivate let owner: UUID
    fileprivate let layers: [LayerState]
    public let offset: Int
    public let logicalByteCount: Int
}

/// Complete CoreAI execution with fixed S1 and S4 functions sharing each asset's
/// AIModel owner. Both phases use the same explicit state tensors. Intermediate
/// activations remain NDArrays; only logits and QSA counters reach the CPU.
/// Operations are serial and fail closed after a partially completed forward.
@available(macOS 27.0, *)
public final class CoreAIPhaseModel {
    private struct Functions {
        let decode: CoreAIBlockRunner
        let prefill: CoreAIBlockRunner
        func runner(count: Int) -> CoreAIBlockRunner { count == 1 ? decode : prefill }
    }
    private final class LoadedLayer {
        let spec: PhaseManifest.Layer
        let functions: Functions
        var states: [String: NDArray]
        var nextOffset = 0
        init(spec: PhaseManifest.Layer, functions: Functions, states: [String: NDArray]) {
            self.spec = spec
            self.functions = functions
            self.states = states
        }
    }

    public let capacity: Int
    public let vocabularySize: Int
    public let tokenChunk: Int
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

    private let embedding: Functions
    private let head: Functions
    private let layers: [LoadedLayer]
    private let snapshotOwner = UUID()
    private let gate = NSLock()
    private var operationInProgress = false

    public init(manifestURL: URL, computeUnits: CoreAIComputeUnits = .gpu,
                progress: ((Int, Int) -> Void)? = nil) async throws {
        let started = DispatchTime.now().uptimeNanoseconds
        guard manifestURL.isFileURL else {
            throw CoreAIBlockRunnerError.invalidModel("CoreAI phase manifest must be a local file")
        }
        let manifest = try JSONDecoder().decode(PhaseManifest.self, from: Data(contentsOf: manifestURL))
        guard manifest.version == 1, manifest.backend == "native-coreai-pd", manifest.status == "complete",
              manifest.completeModelLayerSet, manifest.layers.count == 48,
              Set(manifest.layers.map(\.index)) == Set(0..<48),
              manifest.tokenChunk == 4, manifest.hiddenSize == 2560, manifest.streamCount == 4,
              manifest.vocabularySize == 248320, manifest.capacity > 0,
              manifest.capacity <= Int(Int32.max), manifest.capacity.isMultiple(of: 4),
              manifest.modelDirectory.hasPrefix("/"), manifest.configSHA256.count == 64,
              manifest.configSHA256.allSatisfy({ "0123456789abcdef".contains($0) }),
              Set(manifest.assets.keys) == Set(["embedding", "head"]) else {
            throw CoreAIBlockRunnerError.invalidModel("CoreAI phase execution requires complete S1/S4 assets for all 48 layers")
        }
        let source = URL(fileURLWithPath: manifest.modelDirectory).standardizedFileURL.resolvingSymlinksInPath()
        let configData = try Data(contentsOf: source.appendingPathComponent("config.json"))
        let digest = SHA256.hash(data: configData).map { String(format: "%02x", $0) }.joined()
        let config = try QwenConfiguration(modelDirectory: source)
        guard digest == manifest.configSHA256,
              config.layerCount == 48, config.hiddenSize == 2560, config.hcCount == 4,
              config.vocabularySize == 248320, config.expertCount == 512, config.expertsPerToken == 10,
              config.pleLayerIndices == [1], manifest.capacity <= config.maximumPositions else {
            throw CoreAIBlockRunnerError.invalidModel("CoreAI phase assets do not match the supported source configuration")
        }
        let base = manifestURL.standardizedFileURL.deletingLastPathComponent()
        func assetURL(_ spec: PhaseAsset) -> URL {
            URL(fileURLWithPath: spec.path, relativeTo: base).standardizedFileURL.resolvingSymlinksInPath()
        }
        let specs = Array(manifest.assets.values) + manifest.layers.map(\.asset)
        guard specs.allSatisfy({ !$0.path.isEmpty && $0.function == "main" && $0.prefillFunction == "prefill" }),
              Set(specs.map { assetURL($0).path }).count == 50 else {
            throw CoreAIBlockRunnerError.invalidModel("CoreAI phase assets require distinct paths and main/prefill functions")
        }
        for spec in manifest.layers {
            guard spec.kind == (config.layerTypes[spec.index] == "linear_attention" ? "gdn" : "qsa"),
                  spec.hasPLE == (spec.index == 1), spec.expertCount == 512, spec.topK == 10,
                  spec.completeExpertBank, spec.inputName == "stream", spec.outputName == "stream_out" else {
                throw CoreAIBlockRunnerError.invalidModel("Layer \(spec.index): incomplete experts or unsupported fused architecture")
            }
            try Self.validateStateMetadata(spec, capacity: manifest.capacity)
        }
        var completed = 0
        func load(_ spec: PhaseAsset) async throws -> Functions {
            try Task.checkCancellation()
            let decode = try await CoreAIBlockRunner(modelURL: assetURL(spec), functionName: spec.function,
                                                    computeUnits: computeUnits)
            let prefill = try CoreAIBlockRunner(sharing: decode, functionName: spec.prefillFunction)
            for runner in [decode, prefill] {
                try Self.validateFeatures(runner, spec: spec)
            }
            completed += 1
            progress?(completed, 50)
            return Functions(decode: decode, prefill: prefill)
        }
        let embedding = try await load(manifest.assets["embedding"]!)
        let head = try await load(manifest.assets["head"]!)
        guard manifest.assets["embedding"]!.inputNames == ["token"],
              manifest.assets["embedding"]!.outputNames == ["stream"],
              manifest.assets["head"]!.inputNames == ["stream"],
              manifest.assets["head"]!.outputNames == ["logits"] else {
            throw CoreAIBlockRunnerError.invalidModel("CoreAI phase embedding/head names do not match the runtime")
        }
        for count in [1, manifest.tokenChunk] {
            try Self.require(embedding.runner(count: count), input: "token", shape: [count], type: .int32)
            try Self.require(embedding.runner(count: count), output: "stream", shape: [1, count, 10240], type: .float16)
            try Self.require(head.runner(count: count), input: "stream", shape: [1, count, 10240], type: .float16)
            try Self.require(head.runner(count: count), output: "logits", shape: [1, 1, 248320], type: .float32)
        }
        var loaded: [LoadedLayer] = []
        for spec in manifest.layers.sorted(by: { $0.index < $1.index }) {
            let functions = try await load(spec.asset)
            for count in [1, manifest.tokenChunk] {
                let runner = functions.runner(count: count)
                try Self.require(runner, input: "stream", shape: [1, count, 10240], type: .float16)
                try Self.require(runner, output: "stream_out", shape: [1, count, 10240], type: .float16)
                if spec.hasPLE {
                    try Self.require(runner, input: "ple_embedding", shape: [1, count, 2560], type: .float16)
                }
                for (name, state) in spec.initialState {
                    let type = try Self.scalarType(state.dtype)
                    try Self.require(runner, input: name, shape: state.shape, type: type)
                    try Self.require(runner, output: spec.stateBindings[name]!, shape: state.shape, type: type)
                }
            }
            loaded.append(LoadedLayer(spec: spec, functions: functions, states: try Self.zeroStates(spec)))
        }
        self.embedding = embedding
        self.head = head
        self.layers = loaded
        self.capacity = manifest.capacity
        self.tokenChunk = manifest.tokenChunk
        self.vocabularySize = manifest.vocabularySize
        self.manifestModelDirectory = source
        self.sourceConfigSHA256 = digest
        self.modelLoadMilliseconds = CoreAIBlockRunner.milliseconds(since: started)
    }

    /// PLE rows are token-major, 2560 values per token. S4 is for prompt chunks;
    /// S1 handles prompt remainders and autoregressive decode without padding.
    public func forward(tokens: [Int32], pleEmbedding: [Float]) async throws -> [Float] {
        try beginOperation()
        defer { endOperation() }
        guard valid else {
            throw CoreAIBlockRunnerError.invalidFixture("CoreAI phase model requires reset/restore after a failed forward")
        }
        let count = tokens.count
        let unsupported = Set<Int32>([248053, 248054, 248055, 248056, 248057, 248070, 248071, 248076])
        guard count == 1 || count == tokenChunk,
              offset >= 0, offset <= capacity - count,
              tokens.allSatisfy({ $0 >= 0 && Int($0) < vocabularySize && !unsupported.contains($0) }),
              pleEmbedding.count == count * 2560,
              pleEmbedding.allSatisfy({ $0.isFinite && Float16($0).isFinite }) else {
            throw CoreAIBlockRunnerError.invalidFixture("Expected S1/S4 supported tokens, finite PLE rows and available context capacity")
        }
        try Task.checkCancellation()
        let started = DispatchTime.now().uptimeNanoseconds
        let callsBefore = successfulCalls
        let nextOffset = offset + count
        let phase = count == 1 ? "decode" : "prefill"
        do {
            for layer in layers {
                guard layer.nextOffset == offset else {
                    throw CoreAIBlockRunnerError.invalidFixture("Layer \(layer.spec.index): phase state offsets disagree")
                }
                try Self.validateStates(layer.states, layer: layer, at: offset)
            }
            let ids = try CoreAIBlockRunner.makeArray(CoreMLTensor(shape: [count], dtype: .int32,
                                                                  values: tokens.map(Double.init)), name: "token")
            let rows = try CoreAIBlockRunner.makeArray(CoreMLTensor(shape: [1, count, 2560], dtype: .float16,
                values: pleEmbedding.map(Double.init)), name: "ple_embedding")
            var stream = try Self.required(await call(embedding.runner(count: count), inputs: ["token": ids],
                                                     group: "\(phase).embedding"), "stream")
            for layer in layers {
                var inputs = layer.states
                inputs["stream"] = stream
                if layer.spec.hasPLE { inputs["ple_embedding"] = rows }
                let output = try await call(layer.functions.runner(count: count), inputs: inputs,
                                            group: "\(phase).\(layer.spec.kind)")
                var nextStates: [String: NDArray] = [:]
                for (name, outputName) in layer.spec.stateBindings {
                    nextStates[name] = try Self.required(output, outputName)
                }
                try Self.validateStates(nextStates, layer: layer, at: nextOffset)
                let nextStream = try Self.required(output, "stream_out")
                layer.states = nextStates
                layer.nextOffset = nextOffset
                stream = nextStream
            }
            let output = try await call(head.runner(count: count), inputs: ["stream": stream], group: "\(phase).head")
            let logits = try CoreAIBlockRunner.read(Self.required(output, "logits"), name: "logits")
            guard logits.shape == [1, 1, vocabularySize], logits.dtype == .float32,
                  logits.values.allSatisfy({ $0.isFinite && Float($0).isFinite }),
                  successfulCalls - callsBefore == 50 else {
                throw CoreAIBlockRunnerError.invalidFixture("CoreAI phase forward returned invalid logits or incomplete model execution")
            }
            try Task.checkCancellation()
            offset = nextOffset
            lastForwardMilliseconds = CoreAIBlockRunner.milliseconds(since: started)
            return logits.values.map(Float.init)
        } catch {
            valid = false
            failureReason = "Forward at offset \(offset), tokens \(count): \(error)"
            throw error
        }
    }

    /// Metadata-only size query; reserve a cache budget before copying storage.
    public func stateByteCount() throws -> Int {
        try beginOperation()
        defer { endOperation() }
        var bytes = 0
        for layer in layers {
            bytes = try CoreAITensorCopy.addingByteCounts(bytes, CoreAITensorCopy.logicalByteCount(layer.states))
        }
        return bytes
    }

    public func checkpoint() throws -> CoreAIPhaseSnapshot {
        try beginOperation()
        defer { endOperation() }
        guard valid, offset >= 0, offset <= capacity, layers.allSatisfy({ $0.nextOffset == offset }) else {
            throw CoreAIBlockRunnerError.invalidFixture("Cannot checkpoint an incomplete or failed CoreAI phase forward")
        }
        var copied: [CoreAIPhaseSnapshot.LayerState] = []
        var bytes = 0
        for layer in layers {
            try Self.validateStates(layer.states, layer: layer, at: offset)
            bytes = try CoreAITensorCopy.addingByteCounts(bytes, CoreAITensorCopy.logicalByteCount(layer.states))
            copied.append(.init(states: try CoreAITensorCopy.deepCopy(layer.states), offset: offset))
        }
        return CoreAIPhaseSnapshot(owner: snapshotOwner, layers: copied, offset: offset, logicalByteCount: bytes)
    }

    /// Restores independent copies after all validation/allocation succeeds.
    /// Rejected or foreign snapshots leave the current model untouched.
    public func restore(_ snapshot: CoreAIPhaseSnapshot) throws {
        try beginOperation()
        defer { endOperation() }
        guard snapshot.owner == snapshotOwner, snapshot.layers.count == layers.count,
              snapshot.offset >= 0, snapshot.offset <= capacity,
              snapshot.layers.allSatisfy({ $0.offset == snapshot.offset }) else {
            throw CoreAIBlockRunnerError.invalidFixture("Foreign or inconsistent CoreAI phase checkpoint")
        }
        var restored: [[String: NDArray]] = []
        var bytes = 0
        for (layer, saved) in zip(layers, snapshot.layers) {
            try Self.validateStates(saved.states, layer: layer, at: snapshot.offset)
            bytes = try CoreAITensorCopy.addingByteCounts(bytes, CoreAITensorCopy.logicalByteCount(saved.states))
            restored.append(try CoreAITensorCopy.deepCopy(saved.states))
        }
        guard bytes == snapshot.logicalByteCount else {
            throw CoreAIBlockRunnerError.invalidFixture("CoreAI phase checkpoint byte count differs from its state")
        }
        for (layer, states) in zip(layers, restored) {
            layer.states = states
            layer.nextOffset = snapshot.offset
        }
        offset = snapshot.offset
        clearStatus()
    }

    /// Recreates all zero state without reloading either phase's weights.
    public func reset() throws {
        try beginOperation()
        defer { endOperation() }
        let zeros = try layers.map { try Self.zeroStates($0.spec) }
        for (layer, states) in zip(layers, zeros) {
            layer.states = states
            layer.nextOffset = 0
        }
        offset = 0
        clearStatus()
    }

    private func clearStatus() {
        valid = true
        failureReason = nil
        successfulCalls = 0
        callCounts = [:]
        predictionMillisecondsByGroup = [:]
        lastForwardMilliseconds = 0
    }

    private func call(_ runner: CoreAIBlockRunner, inputs: [String: NDArray], group: String) async throws -> [String: NDArray] {
        try Task.checkCancellation()
        let validated = try runner.makeInputs([:], retainedInputs: inputs)
        let started = DispatchTime.now().uptimeNanoseconds
        var prediction = try await runner.function.run(inputs: validated)
        let duration = CoreAIBlockRunner.milliseconds(since: started)
        let descriptor = runner.function.descriptor
        guard Set(prediction.names) == Set(descriptor.outputNames) else {
            throw CoreAIBlockRunnerError.invalidFixture("\(group): output names differ from the phase function descriptor")
        }
        var output: [String: NDArray] = [:]
        for name in descriptor.outputNames {
            guard let array = prediction.remove(name)?.ndArray,
                  case .ndArray(let declared) = descriptor.outputDescriptor(of: name) else {
                throw CoreAIBlockRunnerError.invalidFixture("\(group): missing tensor output \(name)")
            }
            try CoreAIBlockRunner.validate(array, descriptor: declared, name: name)
            output[name] = array
        }
        successfulCalls += 1
        callCounts[group, default: 0] += 1
        predictionMillisecondsByGroup[group, default: 0] += duration
        return output
    }

    private static func required(_ outputs: [String: NDArray], _ name: String) throws -> NDArray {
        guard let array = outputs[name] else {
            throw CoreAIBlockRunnerError.invalidFixture("Required CoreAI phase output \(name) is missing")
        }
        return array
    }

    private static func scalarType(_ dtype: CoreMLTensorDataType) throws -> NDArray.ScalarType {
        switch dtype {
        case .float16: return .float16
        case .float32: return .float32
        case .int32: return .int32
        default: throw CoreAIBlockRunnerError.invalidModel("Unsupported CoreAI phase state dtype \(dtype)")
        }
    }

    private static func validateFeatures(_ runner: CoreAIBlockRunner, spec: PhaseAsset) throws {
        let descriptor = runner.function.descriptor
        guard descriptor.stateNames.isEmpty,
              Set(spec.inputNames).count == spec.inputNames.count,
              Set(spec.outputNames).count == spec.outputNames.count,
              Set(descriptor.inputNames) == Set(spec.inputNames),
              Set(descriptor.outputNames) == Set(spec.outputNames) else {
            throw CoreAIBlockRunnerError.invalidModel("\(spec.path): phase tensor interface differs from manifest")
        }
        for name in descriptor.inputNames {
            guard case .ndArray(let tensor) = descriptor.inputDescriptor(of: name),
                  tensor.interleaveLayout == nil, !tensor.hasDynamicShape else {
                throw CoreAIBlockRunnerError.invalidModel("\(name): phase inputs require fixed non-interleaved tensors")
            }
            _ = try CoreAIBlockRunner.dtype(tensor.scalarType, name: name)
        }
        for name in descriptor.outputNames {
            guard case .ndArray(let tensor) = descriptor.outputDescriptor(of: name),
                  tensor.interleaveLayout == nil, !tensor.hasDynamicShape else {
                throw CoreAIBlockRunnerError.invalidModel("\(name): phase outputs require fixed non-interleaved tensors")
            }
            _ = try CoreAIBlockRunner.dtype(tensor.scalarType, name: name)
        }
    }

    private static func require(_ runner: CoreAIBlockRunner, input: String? = nil, output: String? = nil,
                                shape: [Int], type: NDArray.ScalarType) throws {
        let declared = input.map { runner.function.descriptor.inputDescriptor(of: $0) }
            ?? output.map { runner.function.descriptor.outputDescriptor(of: $0) }
        guard case .ndArray(let tensor) = declared ?? nil, tensor.shape == shape,
              tensor.scalarType == type, tensor.interleaveLayout == nil, !tensor.hasDynamicShape else {
            throw CoreAIBlockRunnerError.invalidModel("\(input ?? output ?? "unknown"): phase descriptor differs from supported architecture")
        }
    }

    private static func validateStateMetadata(_ spec: PhaseManifest.Layer, capacity: Int) throws {
        var expected: [String: ([Int], CoreMLTensorDataType, String)]
        if spec.kind == "gdn" {
            expected = ["conv_history": ([1, 3, 10240], .float16, "next_conv_history"),
                        "recurrent_state": ([1, 48, 128, 128], .float32, "next_recurrent_state")]
        } else {
            expected = ["key_cache": ([1, 2, capacity, 256], .float16, "key_cache_out"),
                        "value_cache": ([1, 2, capacity, 256], .float16, "value_cache_out"),
                        "raw_cache": ([1, capacity, 128], .float16, "raw_cache_out"),
                        "pooled_cache": ([1, capacity / 4, 128], .float16, "pooled_cache_out"),
                        "offset": ([1], .int32, "offset_out"),
                        "pooled_count": ([1], .int32, "pooled_count_out")]
        }
        if spec.hasPLE { expected["ple_state"] = ([1, 9, 10240], .float16, "next_ple_state") }
        let activationNames: Set<String> = spec.hasPLE ? ["stream", "ple_embedding"] : ["stream"]
        guard Set(spec.initialState.keys) == Set(expected.keys),
              Set(spec.stateBindings.keys) == Set(expected.keys),
              Set(spec.stateBindings.values).count == expected.count,
              Set(spec.inputNames) == Set(expected.keys).union(activationNames),
              Set(spec.outputNames) == Set(expected.values.map { $0.2 }).union(["stream_out"]) else {
            throw CoreAIBlockRunnerError.invalidModel("Layer \(spec.index): incomplete explicit phase state interface")
        }
        for (name, (shape, dtype, output)) in expected {
            guard let actual = spec.initialState[name], actual.shape == shape, actual.dtype == dtype,
                  actual.fill == 0, spec.stateBindings[name] == output else {
                throw CoreAIBlockRunnerError.invalidModel("Layer \(spec.index)/\(name): phase zero-state metadata differs from architecture")
            }
        }
    }

    private static func zeroStates(_ spec: PhaseManifest.Layer) throws -> [String: NDArray] {
        try spec.initialState.mapValues { state in
            try CoreAITensorCopy.zeroArray(shape: state.shape, scalarType: scalarType(state.dtype))
        }
    }

    private static func validateStates(_ states: [String: NDArray], layer: LoadedLayer, at offset: Int) throws {
        guard Set(states.keys) == Set(layer.spec.stateBindings.keys) else {
            throw CoreAIBlockRunnerError.invalidFixture("Layer \(layer.spec.index): incomplete phase state")
        }
        for (name, state) in states {
            guard case .ndArray(let descriptor) = layer.functions.decode.function.descriptor.inputDescriptor(of: name),
                  state.interleaveLayout == nil else {
                throw CoreAIBlockRunnerError.invalidFixture("Layer \(layer.spec.index)/\(name): unsupported phase state")
            }
            try CoreAIBlockRunner.validate(state, descriptor: descriptor, name: name)
        }
        if layer.spec.kind == "qsa" {
            guard try integerState(states, name: "offset") == offset,
                  try integerState(states, name: "pooled_count") == offset / 4 else {
                throw CoreAIBlockRunnerError.invalidFixture("Layer \(layer.spec.index): QSA counters differ from phase offset \(offset)")
            }
        }
    }

    private static func integerState(_ states: [String: NDArray], name: String) throws -> Int {
        guard let state = states[name], state.shape == [1], state.scalarType == .int32,
              state.interleaveLayout == nil else {
            throw CoreAIBlockRunnerError.invalidFixture("QSA \(name) must be Int32[1]")
        }
        return state.view(as: Int32.self).withUnsafePointer { pointer, _, _ in Int(pointer[0]) }
    }

    private func beginOperation() throws {
        gate.lock()
        defer { gate.unlock() }
        guard !operationInProgress else {
            throw CoreAIBlockRunnerError.invalidFixture("CoreAI phase model already has an operation in progress")
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
