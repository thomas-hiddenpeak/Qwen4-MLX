#if canImport(CoreAI)
import CoreAI
import Dispatch
import Foundation
import MachO

@available(macOS 27.0, *)
public enum CoreAIComputeUnits: String, Codable, Sendable {
    case cpuOnly, `default`, gpu, neuralEngine

    fileprivate var options: SpecializationOptions {
        switch self {
        case .cpuOnly: .cpuOnly
        case .default: .default
        case .gpu: .init(preferredComputeUnitKind: .gpu)
        case .neuralEngine: .init(preferredComputeUnitKind: .neuralEngine)
        }
    }
}

@available(macOS 27.0, *)
public struct CoreAIBlockMetadata: Codable, Sendable {
    public let operatingSystem: String
    public let deviceArchitecture: String
    public let loadedCoreAIImages: [String]
    public let availableComputeUnits: [String]
    public let allowedComputeUnits: [String]
    public let preferredComputeUnit: String?
}

@available(macOS 27.0, *)
public struct CoreAIBlockReport: Codable, Sendable {
    public let modelPath: String
    public let functionName: String
    public let computeUnits: CoreAIComputeUnits
    public let hardwareEvidence: String
    public let metadata: CoreAIBlockMetadata
    /// Includes specialization and function loading; a cached model may load faster.
    public let modelLoadMilliseconds: Double
    public let inputPreparationMilliseconds: Double
    /// Measured function runs only, excluding warmups and output serialization.
    public let predictionMilliseconds: [Double]
    public let medianPredictionMilliseconds: Double
    public let outputMaterializationMilliseconds: Double
    public let totalRunMilliseconds: Double
    public let warmups: Int
    public let runs: Int
    public let outputs: [String: CoreMLTensor]
    public let comparisons: [String: CoreMLTensorComparison]?
}

@available(macOS 27.0, *)
public enum CoreAIBlockRunnerError: LocalizedError, Sendable {
    case invalidModel(String)
    case invalidFixture(String)
    case unsupportedFeature(String)

    public var errorDescription: String? {
        switch self {
        case .invalidModel(let message), .invalidFixture(let message), .unsupportedFeature(let message): message
        }
    }
}

/// Executes one stateless CoreAI function using the existing JSON tensor fixtures.
/// GPU/Neural Engine preferences allow fallback and are not placement evidence.
@available(macOS 27.0, *)
public final class CoreAIBlockRunner {
    private let model: AIModel
    let function: InferenceFunction
    private let modelURL: URL
    private let computeUnits: CoreAIComputeUnits
    private let options: SpecializationOptions
    private let modelLoadMilliseconds: Double

    public init(modelURL: URL, functionName: String = "main", computeUnits: CoreAIComputeUnits = .default) async throws {
        guard modelURL.isFileURL, FileManager.default.fileExists(atPath: modelURL.path),
              AIModelAsset.isValid(at: modelURL) else {
            throw CoreAIBlockRunnerError.invalidModel("Model must be an existing local CoreAI model asset")
        }
        guard !functionName.isEmpty else {
            throw CoreAIBlockRunnerError.invalidModel("Function name must not be empty")
        }
        self.modelURL = modelURL.standardizedFileURL
        self.computeUnits = computeUnits
        self.options = computeUnits.options
        let start = DispatchTime.now().uptimeNanoseconds
        let model = try await AIModel(contentsOf: modelURL, options: options)
        guard let descriptor = model.functionDescriptor(for: functionName) else {
            throw CoreAIBlockRunnerError.invalidModel("Function '\(functionName)' not found; available: \(model.functionNames.sorted())")
        }
        guard descriptor.stateNames.isEmpty else {
            throw CoreAIBlockRunnerError.unsupportedFeature("Stateful functions are unsupported: \(descriptor.stateNames.sorted())")
        }
        guard let function = try model.loadFunction(named: functionName) else {
            throw CoreAIBlockRunnerError.invalidModel("Could not load function '\(functionName)'")
        }
        self.model = model
        self.function = function
        self.modelLoadMilliseconds = Self.milliseconds(since: start)
    }

    /// Load another entrypoint from the same runtime model and weight owner.
    /// This does not construct another AIModel or reload its asset.
    public init(sharing owner: CoreAIBlockRunner, functionName: String) throws {
        let started = DispatchTime.now().uptimeNanoseconds
        guard !functionName.isEmpty,
              let descriptor = owner.model.functionDescriptor(for: functionName),
              descriptor.stateNames.isEmpty,
              let function = try owner.model.loadFunction(named: functionName) else {
            throw CoreAIBlockRunnerError.invalidModel("Missing or unsupported shared entrypoint '\(functionName)'")
        }
        self.model = owner.model
        self.function = function
        self.modelURL = owner.modelURL
        self.computeUnits = owner.computeUnits
        self.options = owner.options
        self.modelLoadMilliseconds = Self.milliseconds(since: started)
    }

    public static func validateIterations(warmups: Int, runs: Int) throws {
        guard (0...1000).contains(warmups), (1...10000).contains(runs) else {
            throw CoreAIBlockRunnerError.invalidFixture("warmups must be 0...1000 and runs must be 1...10000")
        }
    }

    public func run(fixture: CoreMLBlockFixture, warmups: Int = 3, runs: Int = 5) async throws -> CoreAIBlockReport {
        try Self.validateIterations(warmups: warmups, runs: runs)
        let totalStart = DispatchTime.now().uptimeNanoseconds
        let preparationStart = DispatchTime.now().uptimeNanoseconds
        let inputs = try makeInputs(fixture.inputs)
        for (name, tensor) in fixture.expectedOutputs ?? [:] { try tensor.validate(name: name) }
        let preparationTime = Self.milliseconds(since: preparationStart)
        for _ in 0..<warmups { _ = try await function.run(inputs: inputs) }
        var durations = [Double]()
        durations.reserveCapacity(runs)
        var outputs = [String: CoreMLTensor]()
        var materializationTime = 0.0
        for iteration in 0..<runs {
            let start = DispatchTime.now().uptimeNanoseconds
            var prediction = try await function.run(inputs: inputs)
            durations.append(Self.milliseconds(since: start))
            if iteration == runs - 1 {
                let materializationStart = DispatchTime.now().uptimeNanoseconds
                for name in function.descriptor.outputNames {
                    guard let array = prediction.remove(name)?.ndArray else {
                        throw CoreAIBlockRunnerError.unsupportedFeature("\(name): only tensor outputs are supported")
                    }
                    outputs[name] = try Self.read(array, name: name)
                }
                materializationTime = Self.milliseconds(since: materializationStart)
            }
        }
        var comparisons: [String: CoreMLTensorComparison]?
        if let expected = fixture.expectedOutputs {
            comparisons = [:]
            for (name, tensor) in expected {
                guard let actual = outputs[name], actual.shape == tensor.shape else {
                    throw CoreAIBlockRunnerError.invalidFixture("\(name): expected output is missing or has a different shape")
                }
                comparisons?[name] = try Self.compare(actual.values, tensor.values, name: name)
            }
        }
        let sorted = durations.sorted()
        let middle = sorted.count / 2
        let median = sorted.count.isMultiple(of: 2) ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle]
        return CoreAIBlockReport(
            modelPath: modelURL.path, functionName: function.descriptor.name, computeUnits: computeUnits,
            hardwareEvidence: "Public system CoreAI execution completed. Preferred compute units are scheduling preferences; fallback remains possible. No hardware execution trace or ANE residency claim is included.",
            metadata: metadata(), modelLoadMilliseconds: modelLoadMilliseconds,
            inputPreparationMilliseconds: preparationTime, predictionMilliseconds: durations,
            medianPredictionMilliseconds: median, outputMaterializationMilliseconds: materializationTime,
            totalRunMilliseconds: Self.milliseconds(since: totalStart), warmups: warmups, runs: runs,
            outputs: outputs, comparisons: comparisons)
    }

    func makeInputs(_ tensors: [String: CoreMLTensor], retainedInputs: [String: NDArray] = [:]) throws -> [String: NDArray] {
        guard Set(tensors.keys).isDisjoint(with: retainedInputs.keys) else {
            throw CoreAIBlockRunnerError.invalidFixture("Fixture inputs must not replace retained state")
        }
        guard Set(tensors.keys).union(retainedInputs.keys) == Set(function.descriptor.inputNames) else {
            throw CoreAIBlockRunnerError.invalidFixture("Input names must match \(function.descriptor.inputNames.sorted())")
        }
        var inputs = retainedInputs
        for (name, array) in retainedInputs {
            guard case .ndArray(let descriptor) = function.descriptor.inputDescriptor(of: name) else {
                throw CoreAIBlockRunnerError.unsupportedFeature("\(name): only tensor inputs are supported")
            }
            try Self.validate(array, descriptor: descriptor, name: name)
        }
        for (name, tensor) in tensors {
            guard case .ndArray(let descriptor) = function.descriptor.inputDescriptor(of: name) else {
                throw CoreAIBlockRunnerError.unsupportedFeature("\(name): only tensor inputs are supported")
            }
            inputs[name] = try Self.makeArray(tensor, descriptor: descriptor, name: name)
        }
        return inputs
    }

    static func makeArray(_ tensor: CoreMLTensor, descriptor: NDArrayDescriptor? = nil, name: String) throws -> NDArray {
        try tensor.validate(name: name)
        var array: NDArray
        if let descriptor {
            guard try Self.dtype(descriptor.scalarType, name: name) == tensor.dtype else {
                throw CoreAIBlockRunnerError.invalidFixture("\(name): fixture dtype does not match model dtype")
            }
            guard descriptor.rank == tensor.shape.count,
                  zip(descriptor.shape, tensor.shape).allSatisfy({ $0 <= 0 || $0 == $1 }) else {
                throw CoreAIBlockRunnerError.invalidFixture("\(name): shape \(tensor.shape) does not match declared shape \(descriptor.shape)")
            }
            let resolved = descriptor.hasDynamicShape ? descriptor.resolvingDynamicDimensions(tensor.shape) : descriptor
            guard !resolved.hasDynamicShape, resolved.shape == tensor.shape else {
                throw CoreAIBlockRunnerError.invalidFixture("\(name): model input shape could not be resolved")
            }
            guard resolved.interleaveLayout == nil else {
                throw CoreAIBlockRunnerError.unsupportedFeature("\(name): interleaved tensor inputs are unsupported")
            }
            array = NDArray(descriptor: resolved)
        } else {
            let scalarType: NDArray.ScalarType
            switch tensor.dtype {
            case .float16: scalarType = .float16
            case .float32: scalarType = .float32
            case .float64: scalarType = .float64
            case .int32: scalarType = .int32
            }
            array = NDArray(shape: tensor.shape, scalarType: scalarType)
        }
        switch tensor.dtype {
        case .float16:
            var view = array.mutableView(as: Float16.self)
            view.copyElements(fromContentsOf: tensor.values.map(Float16.init))
        case .float32:
            var view = array.mutableView(as: Float.self)
            view.copyElements(fromContentsOf: tensor.values.map(Float.init))
        case .float64:
            var view = array.mutableView(as: Double.self)
            view.copyElements(fromContentsOf: tensor.values)
        case .int32:
            var view = array.mutableView(as: Int32.self)
            view.copyElements(fromContentsOf: tensor.values.map(Int32.init))
        }
        return array
    }

    static func validate(_ array: NDArray, descriptor: NDArrayDescriptor, name: String) throws {
        guard array.scalarType == descriptor.scalarType,
              array.shape.count == descriptor.rank,
              !array.shape.isEmpty, array.shape.allSatisfy({ $0 > 0 }),
              zip(descriptor.shape, array.shape).allSatisfy({ $0 < 0 || $0 == $1 }) else {
            throw CoreAIBlockRunnerError.invalidFixture("\(name): retained tensor dtype or shape does not match model descriptor")
        }
        guard array.interleaveLayout == nil, descriptor.interleaveLayout == nil else {
            throw CoreAIBlockRunnerError.unsupportedFeature("\(name): interleaved tensors are unsupported")
        }
        let resolved = descriptor.hasDynamicShape ? descriptor.resolvingDynamicDimensions(array.shape) : descriptor
        guard !resolved.hasDynamicShape, resolved.shape == array.shape else {
            throw CoreAIBlockRunnerError.invalidFixture("\(name): retained tensor shape could not be resolved")
        }
        _ = try dtype(array.scalarType, name: name)
    }

    static func dtype(_ type: NDArray.ScalarType, name: String) throws -> CoreMLTensorDataType {
        switch type {
        case .float16: .float16
        case .float32: .float32
        case .float64: .float64
        case .int32: .int32
        default: throw CoreAIBlockRunnerError.unsupportedFeature("\(name): unsupported tensor dtype \(type)")
        }
    }

    static func read(_ array: NDArray, name: String) throws -> CoreMLTensor {
        let type = try dtype(array.scalarType, name: name)
        guard array.interleaveLayout == nil else {
            throw CoreAIBlockRunnerError.unsupportedFeature("\(name): interleaved tensor outputs are unsupported")
        }
        var count = 1
        guard !array.shape.isEmpty, array.shape.allSatisfy({ $0 > 0 }), array.shape.count == array.strides.count else {
            throw CoreAIBlockRunnerError.unsupportedFeature("\(name): unsupported output shape or strides")
        }
        for dimension in array.shape {
            let product = count.multipliedReportingOverflow(by: dimension)
            guard !product.overflow else { throw CoreAIBlockRunnerError.unsupportedFeature("\(name): output shape overflows Int") }
            count = product.partialValue
        }
        let offsets = LogicalOffsets(shape: array.shape, strides: array.strides)
        func values<T: BitwiseCopyable>(as: T.Type, convert: (T) -> Double) -> [Double] {
            array.view(as: T.self).withUnsafePointer { pointer, _, _ in
                (0..<count).map { convert(pointer[offsets.offset($0)]) }
            }
        }
        let data: [Double]
        switch type {
        case .float16: data = values(as: Float16.self, convert: Double.init)
        case .float32: data = values(as: Float.self, convert: Double.init)
        case .float64: data = values(as: Double.self, convert: { $0 })
        case .int32: data = values(as: Int32.self, convert: Double.init)
        }
        let tensor = CoreMLTensor(shape: array.shape, dtype: type, values: data)
        try tensor.validate(name: name)
        return tensor
    }

    private static func compare(_ actual: [Double], _ expected: [Double], name: String) throws -> CoreMLTensorComparison {
        var maximum = 0.0, deltaSquared = 0.0, referenceSquared = 0.0
        for (a, b) in zip(actual, expected) {
            let difference = a - b
            maximum = max(maximum, abs(difference))
            deltaSquared += difference * difference
            referenceSquared += b * b
        }
        guard maximum.isFinite, deltaSquared.isFinite, referenceSquared.isFinite else {
            throw CoreAIBlockRunnerError.invalidFixture("\(name): comparison exceeds finite Double range")
        }
        return CoreMLTensorComparison(maxAbsoluteError: maximum,
                                      rootMeanSquareError: sqrt(deltaSquared / Double(actual.count)),
                                      relativeL2Error: referenceSquared > 0 ? sqrt(deltaSquared / referenceSquared) : nil,
                                      exactMatch: maximum == 0)
    }

    private func metadata() -> CoreAIBlockMetadata {
        var images = [String]()
        for index in 0..<_dyld_image_count() {
            guard let pointer = _dyld_get_image_name(index) else { continue }
            let path = String(cString: pointer)
            if path.contains("CoreAI") { images.append(path) }
        }
        return CoreAIBlockMetadata(
            operatingSystem: ProcessInfo.processInfo.operatingSystemVersionString,
            deviceArchitecture: AIModel.deviceArchitectureName, loadedCoreAIImages: images.sorted(),
            availableComputeUnits: ComputeUnitKind.availableKinds.map { String(describing: $0) }.sorted(),
            allowedComputeUnits: options.allowedComputeUnitKinds.map { String(describing: $0) }.sorted(),
            preferredComputeUnit: options.preferredComputeUnitKind.map { String(describing: $0) })
    }

    static func milliseconds(since start: UInt64) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
    }
}
#endif
