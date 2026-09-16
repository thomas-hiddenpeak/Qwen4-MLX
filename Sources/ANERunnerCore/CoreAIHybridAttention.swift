#if canImport(CoreAI)
import CoreAI
import Dispatch
import Foundation

private struct HybridAttentionManifest: Decodable {
    struct InitialState: Decodable {
        let shape: [Int]
        let dtype: CoreMLTensorDataType
        let fill: Double
    }
    struct Layer: Decodable {
        let index: Int
        let kind: String
        let path: String
        let function: String
        let inputName: String
        let outputName: String
        let stateBindings: [String: String]
        let initialState: [String: InitialState]
    }
    let version: Int
    let status: String
    let capacity: Int
    let tokenChunk: Int
    let layerCount: Int
    let hiddenSize: Int
    let modelDirectory: String
    let configSHA256: String
    let sourceLayerTypes: [String]
    let layers: [Layer]
}

/// An immutable, independently owned RAM checkpoint of all 48 attention layers.
/// It can only be restored into the backend instance that created it.
@available(macOS 27.0, *)
public struct CoreAIHybridAttentionSnapshot: Sendable {
    fileprivate struct LayerState: Sendable {
        let states: [String: NDArray]
        let offset: Int
    }
    fileprivate let owner: UUID
    fileprivate let layers: [Int: LayerState]
    public let offset: Int
    public let logicalByteCount: Int
}

/// Serial S1 attention backend for the Qwen3.8 hybrid prototype. It retains native
/// CoreAI state and reads only the 2560-element activation and QSA integer counters.
/// MoE, normalization, PLE and sampling remain the caller's responsibility.
/// This is not a concurrent serving or persistent-cache interface.
@available(macOS 27.0, *)
public final class CoreAIHybridAttention {
    private final class LoadedLayer {
        let spec: HybridAttentionManifest.Layer
        let runner: CoreAIBlockRunner
        var states: [String: NDArray]
        var nextOffset = 0

        init(spec: HybridAttentionManifest.Layer, runner: CoreAIBlockRunner, states: [String: NDArray]) {
            self.spec = spec
            self.runner = runner
            self.states = states
        }
    }

    public let capacity: Int
    public let layerKinds: [Int: String]
    public let loadedLayerIndices: [Int]
    public let computeUnits: CoreAIComputeUnits
    public let manifestModelDirectory: URL
    public let sourceConfigSHA256: String
    public let modelLoadMilliseconds: Double
    public private(set) var successfulCalls = 0
    public private(set) var totalPredictionMilliseconds = 0.0
    public private(set) var totalInputMilliseconds = 0.0
    public private(set) var totalOutputMilliseconds = 0.0
    public private(set) var lastPredictionMilliseconds = 0.0
    public private(set) var isPoisoned = false
    public private(set) var poisonReason: String?

    private let layers: [Int: LoadedLayer]
    private let snapshotOwner = UUID()
    private let gate = NSLock()
    private var operationInProgress = false

    /// Full-model mode requires all 48 completed layer assets. Partial manifests
    /// are accepted only by the explicitly selected bounded smoke-test mode.
    public init(manifestURL: URL, computeUnits: CoreAIComputeUnits = .gpu,
                requiresFullModel: Bool = true, progress: ((Int, Int) -> Void)? = nil) async throws {
        let start = DispatchTime.now().uptimeNanoseconds
        guard manifestURL.isFileURL else {
            throw CoreAIBlockRunnerError.invalidModel("Hybrid attention manifest must be a local file")
        }
        let manifest = try JSONDecoder().decode(HybridAttentionManifest.self, from: Data(contentsOf: manifestURL))
        let indices = Set(manifest.layers.map(\.index))
        guard manifest.version == 1, manifest.tokenChunk == 1, manifest.layerCount == 48, manifest.hiddenSize == 2560,
              manifest.modelDirectory.hasPrefix("/"), manifest.configSHA256.count == 64,
              manifest.configSHA256.allSatisfy({ "0123456789abcdef".contains($0) }),
              manifest.sourceLayerTypes.count == 48,
              manifest.sourceLayerTypes.allSatisfy({ ["linear_attention", "full_attention"].contains($0) }),
              manifest.capacity > 0, manifest.capacity <= Int(Int32.max), manifest.capacity.isMultiple(of: 4),
              !manifest.layers.isEmpty, indices.count == manifest.layers.count,
              indices.isSubset(of: Set(0..<48)),
              ["complete", "exporting"].contains(manifest.status) else {
            throw CoreAIBlockRunnerError.invalidModel("Invalid S1 hybrid attention manifest or capacity")
        }
        guard !requiresFullModel || (manifest.status == "complete" && indices == Set(0..<48)) else {
            throw CoreAIBlockRunnerError.invalidModel("Full hybrid attention requires a completed manifest containing all 48 unique layers")
        }
        let base = manifestURL.standardizedFileURL.deletingLastPathComponent()
        var urls: [Int: URL] = [:]
        for spec in manifest.layers {
            guard ["gdn", "qsa"].contains(spec.kind), !spec.path.isEmpty, !spec.function.isEmpty,
                  manifest.sourceLayerTypes[spec.index] == (spec.kind == "gdn" ? "linear_attention" : "full_attention"),
                  spec.inputName == (spec.kind == "gdn" ? "hidden" : "x"),
                  spec.outputName == (spec.kind == "gdn" ? "output" : "y"),
                  !spec.stateBindings.isEmpty,
                  Set(spec.stateBindings.values).count == spec.stateBindings.count,
                  Set(spec.stateBindings.keys) == Set(spec.initialState.keys),
                  spec.stateBindings[spec.inputName] == nil,
                  !spec.stateBindings.values.contains(spec.outputName) else {
                throw CoreAIBlockRunnerError.invalidModel("Layer \(spec.index): invalid kind, activation names or state bindings")
            }
            let url = URL(fileURLWithPath: spec.path, relativeTo: base).standardizedFileURL
            guard AIModelAsset.isValid(at: url) else {
                throw CoreAIBlockRunnerError.invalidModel("Layer \(spec.index): missing or invalid CoreAI model asset at \(url.path)")
            }
            urls[spec.index] = url
        }
        // Reject aliases that silently repeat one exported layer in a full manifest.
        guard Set(urls.values.map { $0.resolvingSymlinksInPath().path }).count == urls.count else {
            throw CoreAIBlockRunnerError.invalidModel("Each hybrid layer must have a distinct model asset")
        }
        var loaded: [Int: LoadedLayer] = [:]
        for spec in manifest.layers.sorted(by: { $0.index < $1.index }) {
            try Task.checkCancellation()
            let runner = try await CoreAIBlockRunner(modelURL: urls[spec.index]!, functionName: spec.function, computeUnits: computeUnits)
            try Self.validate(spec, runner: runner, capacity: manifest.capacity)
            let states = try Self.zeroStates(spec, runner: runner)
            loaded[spec.index] = LoadedLayer(spec: spec, runner: runner, states: states)
            progress?(loaded.count, manifest.layers.count)
        }
        self.capacity = manifest.capacity
        self.layers = loaded
        self.layerKinds = Dictionary(uniqueKeysWithValues: manifest.layers.map { ($0.index, $0.kind) })
        self.loadedLayerIndices = indices.sorted()
        self.computeUnits = computeUnits
        self.manifestModelDirectory = URL(fileURLWithPath: manifest.modelDirectory).standardizedFileURL.resolvingSymlinksInPath()
        self.sourceConfigSHA256 = manifest.configSHA256
        self.modelLoadMilliseconds = CoreAIBlockRunner.milliseconds(since: start)
    }

    public func nextOffset(for layer: Int) throws -> Int {
        try beginOperation()
        defer { endOperation() }
        guard let loaded = layers[layer] else {
            throw CoreAIBlockRunnerError.invalidFixture("Hybrid attention layer \(layer) is not loaded")
        }
        return loaded.nextOffset
    }

    /// Failures poison the session because earlier layers of the same token may
    /// already have advanced. Call reset() and replay the prompt before continuing.
    public func forward(layer: Int, input: [Float], offset: Int) async throws -> [Float] {
        let result = try await execute(layer: layer, input: .host(input), offset: offset)
        return result.host!
    }

    /// Native activation handoff for a complete CoreAI graph chain. It does not
    /// materialize intermediate activations or persistent state on the CPU.
    public func forwardArray(layer: Int, input: NDArray, offset: Int) async throws -> NDArray {
        try await execute(layer: layer, input: .native(input), offset: offset).array
    }

    private enum ActivationInput {
        case host([Float])
        case native(NDArray)
    }

    private func execute(layer: Int, input: ActivationInput, offset: Int) async throws -> (array: NDArray, host: [Float]?) {
        try beginOperation()
        defer { endOperation() }
        guard !isPoisoned else {
            throw CoreAIBlockRunnerError.invalidFixture("Hybrid attention is poisoned; reset and replay are required: \(poisonReason ?? "previous failure")")
        }
        do {
            guard let loaded = layers[layer] else {
                throw CoreAIBlockRunnerError.invalidFixture("Hybrid attention layer \(layer) is not loaded")
            }
            guard offset >= 0, offset < capacity, offset == loaded.nextOffset else {
                throw CoreAIBlockRunnerError.invalidFixture("Layer \(layer): offset \(offset) must equal \(loaded.nextOffset) and be below capacity \(capacity)")
            }
            let descriptor = loaded.runner.function.descriptor
            let prepareStart = DispatchTime.now().uptimeNanoseconds
            let inputs: [String: NDArray]
            let materializeActivation: Bool
            switch input {
            case .host(let values):
                guard values.count == 2560, values.allSatisfy({ $0.isFinite && Float16($0).isFinite }) else {
                    throw CoreAIBlockRunnerError.invalidFixture("Layer \(layer): input must contain 2560 finite FP16-representable values")
                }
                let tensor = CoreMLTensor(shape: [1, 1, 2560], dtype: .float16, values: values.map(Double.init))
                inputs = try loaded.runner.makeInputs([loaded.spec.inputName: tensor], retainedInputs: loaded.states)
                materializeActivation = true
            case .native(let array):
                var retained = loaded.states
                retained[loaded.spec.inputName] = array
                inputs = try loaded.runner.makeInputs([:], retainedInputs: retained)
                materializeActivation = false
            }
            let prepareTime = CoreAIBlockRunner.milliseconds(since: prepareStart)
            try Task.checkCancellation()
            let runStart = DispatchTime.now().uptimeNanoseconds
            var prediction = try await loaded.runner.function.run(inputs: inputs)
            let predictionTime = CoreAIBlockRunner.milliseconds(since: runStart)
            let readStart = DispatchTime.now().uptimeNanoseconds
            guard Set(prediction.names) == Set(descriptor.outputNames),
                  let output = prediction.remove(loaded.spec.outputName)?.ndArray,
                  case .ndArray(let outputDescriptor) = descriptor.outputDescriptor(of: loaded.spec.outputName) else {
                throw CoreAIBlockRunnerError.invalidFixture("Layer \(layer): output names or activation output differ from descriptor")
            }
            try CoreAIBlockRunner.validate(output, descriptor: outputDescriptor, name: loaded.spec.outputName)
            guard output.shape == [1, 1, 2560], output.scalarType == .float16 else {
                throw CoreAIBlockRunnerError.invalidFixture("Layer \(layer): activation output must be FP16 [1,1,2560]")
            }
            let host: [Float]?
            if materializeActivation {
                host = try CoreAIBlockRunner.read(output, name: loaded.spec.outputName).values.map(Float.init)
            } else {
                host = nil
            }
            var nextStates: [String: NDArray] = [:]
            for (inputName, outputName) in loaded.spec.stateBindings {
                guard let state = prediction.remove(outputName)?.ndArray,
                      case .ndArray(let declaredOutput) = descriptor.outputDescriptor(of: outputName),
                      case .ndArray(let declaredInput) = descriptor.inputDescriptor(of: inputName) else {
                    throw CoreAIBlockRunnerError.invalidFixture("Layer \(layer): missing tensor state output \(outputName)")
                }
                try CoreAIBlockRunner.validate(state, descriptor: declaredOutput, name: outputName)
                try CoreAIBlockRunner.validate(state, descriptor: declaredInput, name: inputName)
                nextStates[inputName] = state
            }
            if loaded.spec.kind == "qsa" {
                guard try Self.integerState(nextStates, name: "offset") == offset + 1,
                      try Self.integerState(nextStates, name: "pooled_count") == (offset + 1) / 4 else {
                    throw CoreAIBlockRunnerError.invalidFixture("Layer \(layer): QSA state counters did not advance consistently")
                }
            }
            // Auxiliary outputs, such as the diagnostic attention mask, are
            // discarded with prediction. No KV/GDN state is materialized on host.
            try Task.checkCancellation()
            let readTime = CoreAIBlockRunner.milliseconds(since: readStart)
            let nextCount = successfulCalls.addingReportingOverflow(1)
            guard !nextCount.overflow else {
                throw CoreAIBlockRunnerError.invalidFixture("Hybrid attention call count overflows Int")
            }
            loaded.states = nextStates
            loaded.nextOffset = offset + 1
            successfulCalls = nextCount.partialValue
            totalInputMilliseconds += prepareTime
            totalPredictionMilliseconds += predictionTime
            totalOutputMilliseconds += readTime
            lastPredictionMilliseconds = predictionTime
            return (output, host)
        } catch {
            isPoisoned = true
            poisonReason = "Layer \(layer), offset \(offset): \(error)"
            throw error
        }
    }

    /// Metadata-only size for reserving a checkpoint budget before allocating it.
    /// Does not include weights, allocator padding or Swift object overhead.
    public func stateByteCount() throws -> Int {
        try beginOperation()
        defer { endOperation() }
        var bytes = 0
        for layer in layers.values {
            bytes = try CoreAITensorCopy.addingByteCounts(bytes, CoreAITensorCopy.logicalByteCount(layer.states))
        }
        return bytes
    }

    public func checkpoint() throws -> CoreAIHybridAttentionSnapshot {
        try beginOperation()
        defer { endOperation() }
        guard !isPoisoned, loadedLayerIndices == Array(0..<48),
              let commonOffset = layers[0]?.nextOffset,
              commonOffset >= 0, commonOffset <= capacity,
              layers.values.allSatisfy({ $0.nextOffset == commonOffset }) else {
            throw CoreAIBlockRunnerError.invalidFixture("Attention checkpoint requires a valid, complete 48-layer token boundary")
        }
        var copied: [Int: CoreAIHybridAttentionSnapshot.LayerState] = [:]
        var bytes = 0
        for index in loadedLayerIndices {
            let layer = layers[index]!
            try Self.validateCheckpointStates(layer.states, layer: layer, offset: commonOffset)
            bytes = try CoreAITensorCopy.addingByteCounts(bytes, CoreAITensorCopy.logicalByteCount(layer.states))
            copied[index] = .init(states: try CoreAITensorCopy.deepCopy(layer.states), offset: commonOffset)
        }
        return CoreAIHybridAttentionSnapshot(owner: snapshotOwner, layers: copied,
                                            offset: commonOffset, logicalByteCount: bytes)
    }

    /// Validates and copies every layer before changing any live state. Rejected
    /// snapshots leave the current state, poison flags and counters unchanged.
    public func restore(_ snapshot: CoreAIHybridAttentionSnapshot) throws {
        try beginOperation()
        defer { endOperation() }
        guard snapshot.owner == snapshotOwner else {
            throw CoreAIBlockRunnerError.invalidFixture("Cannot restore an attention checkpoint from another backend instance")
        }
        guard loadedLayerIndices == Array(0..<48), Set(snapshot.layers.keys) == Set(loadedLayerIndices),
              snapshot.offset >= 0, snapshot.offset <= capacity else {
            throw CoreAIBlockRunnerError.invalidFixture("Attention checkpoint does not contain a complete valid layer set")
        }
        var restored: [Int: [String: NDArray]] = [:]
        var bytes = 0
        for index in loadedLayerIndices {
            let saved = snapshot.layers[index]!
            guard saved.offset == snapshot.offset else {
                throw CoreAIBlockRunnerError.invalidFixture("Attention checkpoint layer offsets disagree")
            }
            try Self.validateCheckpointStates(saved.states, layer: layers[index]!, offset: saved.offset)
            bytes = try CoreAITensorCopy.addingByteCounts(bytes, CoreAITensorCopy.logicalByteCount(saved.states))
            restored[index] = try CoreAITensorCopy.deepCopy(saved.states)
        }
        guard bytes == snapshot.logicalByteCount else {
            throw CoreAIBlockRunnerError.invalidFixture("Attention checkpoint logical size does not match its state tensors")
        }
        for index in loadedLayerIndices {
            layers[index]!.states = restored[index]!
            layers[index]!.nextOffset = snapshot.layers[index]!.offset
        }
        successfulCalls = 0
        totalInputMilliseconds = 0
        totalPredictionMilliseconds = 0
        totalOutputMilliseconds = 0
        lastPredictionMilliseconds = 0
        isPoisoned = false
        poisonReason = nil
    }

    private static func validateCheckpointStates(_ states: [String: NDArray], layer: LoadedLayer, offset: Int) throws {
        guard Set(states.keys) == Set(layer.spec.stateBindings.keys) else {
            throw CoreAIBlockRunnerError.invalidFixture("Layer \(layer.spec.index): checkpoint state names differ from the model")
        }
        for (name, state) in states {
            guard case .ndArray(let descriptor) = layer.runner.function.descriptor.inputDescriptor(of: name) else {
                throw CoreAIBlockRunnerError.invalidFixture("Layer \(layer.spec.index): checkpoint state input \(name) is missing")
            }
            try CoreAIBlockRunner.validate(state, descriptor: descriptor, name: name)
        }
        if layer.spec.kind == "qsa" {
            guard try integerState(states, name: "offset") == offset,
                  try integerState(states, name: "pooled_count") == offset / 4 else {
                throw CoreAIBlockRunnerError.invalidFixture("Layer \(layer.spec.index): checkpoint QSA counters do not match its offset")
            }
        }
    }

    /// Recreates zero state without reloading weights. Replaces all layers only
    /// after every new state allocation succeeds, and clears counters and poison.
    public func reset() throws {
        try beginOperation()
        defer { endOperation() }
        do {
            var restored: [Int: [String: NDArray]] = [:]
            for index in loadedLayerIndices {
                let layer = layers[index]!
                restored[index] = try Self.zeroStates(layer.spec, runner: layer.runner)
            }
            for index in loadedLayerIndices {
                layers[index]!.states = restored[index]!
                layers[index]!.nextOffset = 0
            }
            successfulCalls = 0
            totalInputMilliseconds = 0
            totalPredictionMilliseconds = 0
            totalOutputMilliseconds = 0
            lastPredictionMilliseconds = 0
            isPoisoned = false
            poisonReason = nil
        } catch {
            isPoisoned = true
            poisonReason = "Reset failed: \(error)"
            throw error
        }
    }

    private static func validate(_ spec: HybridAttentionManifest.Layer, runner: CoreAIBlockRunner, capacity: Int) throws {
        let descriptor = runner.function.descriptor
        guard descriptor.stateNames.isEmpty,
              Set(descriptor.inputNames) == Set(spec.stateBindings.keys).union([spec.inputName]),
              Set(spec.stateBindings.values).union([spec.outputName]).isSubset(of: descriptor.outputNames),
              case .ndArray(let input) = descriptor.inputDescriptor(of: spec.inputName),
              case .ndArray(let output) = descriptor.outputDescriptor(of: spec.outputName),
              input.scalarType == .float16, input.shape == [1, 1, 2560], input.interleaveLayout == nil,
              output.scalarType == .float16, output.shape == [1, 1, 2560], output.interleaveLayout == nil else {
            throw CoreAIBlockRunnerError.invalidModel("Layer \(spec.index): model does not expose the expected stateless S1 tensor interface")
        }
        for (name, state) in spec.initialState {
            guard state.fill == 0, !state.shape.isEmpty, state.shape.allSatisfy({ $0 > 0 }),
                  case .ndArray(let inputDescriptor) = descriptor.inputDescriptor(of: name),
                  let outputName = spec.stateBindings[name],
                  case .ndArray(let outputDescriptor) = descriptor.outputDescriptor(of: outputName),
                  inputDescriptor.shape == state.shape, outputDescriptor.shape == state.shape,
                  inputDescriptor.scalarType == outputDescriptor.scalarType,
                  inputDescriptor.interleaveLayout == nil, outputDescriptor.interleaveLayout == nil,
                  try CoreAIBlockRunner.dtype(inputDescriptor.scalarType, name: name) == state.dtype else {
                throw CoreAIBlockRunnerError.invalidModel("Layer \(spec.index)/\(name): initial zero-state metadata or state descriptors differ")
            }
            _ = try elementCount(state.shape)
        }
        if spec.kind == "qsa" {
            guard spec.initialState["offset"]?.shape == [1], spec.initialState["offset"]?.dtype == .int32,
                  spec.initialState["pooled_count"]?.shape == [1], spec.initialState["pooled_count"]?.dtype == .int32,
                  spec.initialState["key_cache"]?.shape == [1, 2, capacity, 256],
                  spec.initialState["value_cache"]?.shape == [1, 2, capacity, 256],
                  spec.initialState["raw_cache"]?.shape == [1, capacity, 128],
                  spec.initialState["pooled_cache"]?.shape == [1, capacity / 4, 128] else {
                throw CoreAIBlockRunnerError.invalidModel("Layer \(spec.index): QSA state shapes do not match this model and manifest capacity")
            }
        }
    }

    private static func zeroStates(_ spec: HybridAttentionManifest.Layer, runner: CoreAIBlockRunner) throws -> [String: NDArray] {
        var result: [String: NDArray] = [:]
        for (name, state) in spec.initialState {
            guard case .ndArray(let descriptor) = runner.function.descriptor.inputDescriptor(of: name) else {
                throw CoreAIBlockRunnerError.invalidModel("Layer \(spec.index): missing zero-state descriptor \(name)")
            }
            guard state.fill == 0, descriptor.shape == state.shape,
                  descriptor.interleaveLayout == nil,
                  try CoreAIBlockRunner.dtype(descriptor.scalarType, name: name) == state.dtype else {
                throw CoreAIBlockRunnerError.invalidModel("Invalid attention zero-state metadata or descriptor")
            }
            let array = try CoreAITensorCopy.zeroArray(shape: state.shape, scalarType: descriptor.scalarType)
            try CoreAIBlockRunner.validate(array, descriptor: descriptor, name: name)
            result[name] = array
        }
        return result
    }

    private static func integerState(_ states: [String: NDArray], name: String) throws -> Int {
        guard let array = states[name], array.scalarType == .int32, array.shape == [1], array.interleaveLayout == nil else {
            throw CoreAIBlockRunnerError.invalidFixture("QSA \(name) must be an Int32[1] state")
        }
        return array.view(as: Int32.self).withUnsafePointer { pointer, _, _ in Int(pointer[0]) }
    }

    private static func elementCount(_ shape: [Int]) throws -> Int {
        var count = 1
        for dimension in shape {
            let next = count.multipliedReportingOverflow(by: dimension)
            guard dimension > 0, !next.overflow, next.partialValue <= Int.max / MemoryLayout<Double>.stride else {
                throw CoreAIBlockRunnerError.invalidModel("Hybrid attention state shape overflows supported allocation range")
            }
            count = next.partialValue
        }
        return count
    }

    private func beginOperation() throws {
        gate.lock()
        defer { gate.unlock() }
        guard !operationInProgress else {
            throw CoreAIBlockRunnerError.invalidFixture("Hybrid attention already has an operation in progress")
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
