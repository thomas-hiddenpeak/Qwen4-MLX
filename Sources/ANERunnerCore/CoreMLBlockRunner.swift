import CoreML
import Dispatch
import Foundation

public enum CoreMLTensorDataType: String, Codable, Sendable {
    case float16
    case float32
    case float64
    case int32

    var mlDataType: MLMultiArrayDataType {
        switch self {
        case .float16: .float16
        case .float32: .float32
        case .float64: .double
        case .int32: .int32
        }
    }

    init(mlDataType: MLMultiArrayDataType) throws {
        switch mlDataType {
        case .float16: self = .float16
        case .float32: self = .float32
        case .double: self = .float64
        case .int32: self = .int32
        case .int8:
            throw CoreMLBlockRunnerError.unsupportedFeature("int8 tensors are not supported by this fixed-subgraph JSON executor")
        @unknown default:
            throw CoreMLBlockRunnerError.unsupportedFeature("Unsupported MLMultiArray dtype: \(mlDataType.rawValue)")
        }
    }
}

/// Values use logical row-major order, independent of Core ML's storage strides.
public struct CoreMLTensor: Codable, Sendable {
    public let shape: [Int]
    public let dtype: CoreMLTensorDataType
    public let values: [Double]

    public init(shape: [Int], dtype: CoreMLTensorDataType, values: [Double]) {
        self.shape = shape
        self.dtype = dtype
        self.values = values
    }

    func validate(name: String) throws {
        guard !shape.isEmpty, shape.allSatisfy({ $0 > 0 }) else {
            throw CoreMLBlockRunnerError.invalidFixture("\(name): shape must contain positive dimensions")
        }
        var count = 1
        for dimension in shape {
            let product = count.multipliedReportingOverflow(by: dimension)
            guard !product.overflow else {
                throw CoreMLBlockRunnerError.invalidFixture("\(name): shape element count overflows Int")
            }
            count = product.partialValue
        }
        guard count == values.count else {
            throw CoreMLBlockRunnerError.invalidFixture("\(name): shape requires \(count) values, found \(values.count)")
        }
        for value in values {
            guard value.isFinite else {
                throw CoreMLBlockRunnerError.invalidFixture("\(name): non-finite JSON tensor value")
            }
            let representable: Bool
            switch dtype {
            case .float16: representable = Float16(value).isFinite
            case .float32: representable = Float(value).isFinite
            case .float64: representable = true
            case .int32:
                representable = value.rounded(.towardZero) == value
                    && value >= Double(Int32.min) && value <= Double(Int32.max)
            }
            guard representable else {
                throw CoreMLBlockRunnerError.invalidFixture("\(name): value \(value) cannot be represented as \(dtype.rawValue)")
            }
        }
    }
}

public struct CoreMLBlockFixture: Codable, Sendable {
    public let inputs: [String: CoreMLTensor]
    public let expectedOutputs: [String: CoreMLTensor]?

    public init(inputs: [String: CoreMLTensor], expectedOutputs: [String: CoreMLTensor]? = nil) {
        self.inputs = inputs
        self.expectedOutputs = expectedOutputs
    }

    public static func load(from url: URL) throws -> Self {
        let fixture = try JSONDecoder().decode(Self.self, from: Data(contentsOf: url))
        for (name, tensor) in fixture.inputs { try tensor.validate(name: name) }
        for (name, tensor) in fixture.expectedOutputs ?? [:] { try tensor.validate(name: name) }
        return fixture
    }
}

public struct CoreMLTensorComparison: Codable, Sendable {
    public let maxAbsoluteError: Double
    public let rootMeanSquareError: Double
    public let relativeL2Error: Double?
    public let exactMatch: Bool
}

public struct CoreMLBlockReport: Codable, Sendable {
    public let modelPath: String
    public let compiledModelPath: String
    public let computeUnits: String
    public let hardwareEvidence: String
    public let compileMilliseconds: Double
    public let modelLoadMilliseconds: Double
    public let inputPreparationMilliseconds: Double
    public let predictionMilliseconds: [Double]
    public let medianPredictionMilliseconds: Double
    public let outputMaterializationMilliseconds: Double
    public let totalRunMilliseconds: Double
    public let warmups: Int
    public let runs: Int
    public let outputs: [String: CoreMLTensor]
    public let comparisons: [String: CoreMLTensorComparison]?
}

public enum CoreMLBlockRunnerError: LocalizedError, Sendable {
    case invalidModelURL(String)
    case invalidFixture(String)
    case unsupportedFeature(String)

    public var errorDescription: String? {
        switch self {
        case .invalidModelURL(let message), .invalidFixture(let message), .unsupportedFeature(let message): message
        }
    }
}

/// A synchronous executor for fixed-shape Core ML tensor subgraphs.
/// CPU_AND_NE allows CPU fallback. This class does not claim or trace ANE placement.
/// Each call is stateless: any history must be supplied explicitly as an input.
public final class CoreMLBlockRunner {
    private let model: MLModel
    private let modelURL: URL
    private let compiledURL: URL
    private let compileMilliseconds: Double
    private let modelLoadMilliseconds: Double

    public init(modelURL: URL) throws {
        guard modelURL.isFileURL, FileManager.default.fileExists(atPath: modelURL.path) else {
            throw CoreMLBlockRunnerError.invalidModelURL("Model must be an existing local .mlmodelc, .mlpackage, or .mlmodel")
        }
        self.modelURL = modelURL.standardizedFileURL
        let compileStart = DispatchTime.now().uptimeNanoseconds
        switch modelURL.pathExtension.lowercased() {
        case "mlmodelc":
            self.compiledURL = modelURL
            self.compileMilliseconds = 0
        case "mlpackage", "mlmodel":
            // Core ML owns the compiled artifact location returned by this API.
            self.compiledURL = try MLModel.compileModel(at: modelURL)
            self.compileMilliseconds = Self.milliseconds(since: compileStart)
        default:
            throw CoreMLBlockRunnerError.invalidModelURL("Unsupported model extension: \(modelURL.pathExtension)")
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        let loadStart = DispatchTime.now().uptimeNanoseconds
        self.model = try MLModel(contentsOf: compiledURL, configuration: configuration)
        self.modelLoadMilliseconds = Self.milliseconds(since: loadStart)
    }

    public func run(fixture: CoreMLBlockFixture, warmups: Int = 3, runs: Int = 5) throws -> CoreMLBlockReport {
        guard warmups >= 0, runs > 0 else {
            throw CoreMLBlockRunnerError.invalidFixture("warmups must be nonnegative and runs must be positive")
        }
        let totalStart = DispatchTime.now().uptimeNanoseconds
        let preparationStart = DispatchTime.now().uptimeNanoseconds
        let provider = try makeProvider(inputs: fixture.inputs)
        let preparationTime = Self.milliseconds(since: preparationStart)
        for _ in 0..<warmups { _ = try model.prediction(from: provider) }
        var durations = [Double]()
        var lastPrediction: (any MLFeatureProvider)?
        for _ in 0..<runs {
            let start = DispatchTime.now().uptimeNanoseconds
            lastPrediction = try model.prediction(from: provider)
            durations.append(Self.milliseconds(since: start))
        }
        guard let prediction = lastPrediction else {
            throw CoreMLBlockRunnerError.invalidFixture("No prediction was produced")
        }
        let materializationStart = DispatchTime.now().uptimeNanoseconds
        var outputs = [String: CoreMLTensor]()
        for name in prediction.featureNames.sorted() {
            guard let array = prediction.featureValue(for: name)?.multiArrayValue else {
                throw CoreMLBlockRunnerError.unsupportedFeature("\(name): only tensor outputs are supported")
            }
            outputs[name] = try Self.read(array: array, name: name)
        }
        let materializationTime = Self.milliseconds(since: materializationStart)
        var comparisons: [String: CoreMLTensorComparison]?
        if let expected = fixture.expectedOutputs {
            comparisons = [:]
            for (name, tensor) in expected {
                try tensor.validate(name: name)
                guard let actual = outputs[name], actual.shape == tensor.shape else {
                    throw CoreMLBlockRunnerError.invalidFixture("\(name): expected output is missing or has a different shape")
                }
                comparisons?[name] = Self.compare(actual.values, tensor.values)
            }
        }
        let sorted = durations.sorted()
        let middle = sorted.count / 2
        let median = sorted.count.isMultiple(of: 2) ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle]
        return CoreMLBlockReport(
            modelPath: modelURL.path, compiledModelPath: compiledURL.path,
            computeUnits: "CPU_AND_NE",
            hardwareEvidence: "Public Core ML prediction completed with CPU_AND_NE. CPU fallback remains possible; this report contains neither a compute plan nor a hardware execution trace.",
            compileMilliseconds: compileMilliseconds, modelLoadMilliseconds: modelLoadMilliseconds,
            inputPreparationMilliseconds: preparationTime, predictionMilliseconds: durations,
            medianPredictionMilliseconds: median, outputMaterializationMilliseconds: materializationTime,
            totalRunMilliseconds: Self.milliseconds(since: totalStart), warmups: warmups, runs: runs,
            outputs: outputs, comparisons: comparisons)
    }

    private func makeProvider(inputs: [String: CoreMLTensor]) throws -> MLDictionaryFeatureProvider {
        let descriptions = model.modelDescription.inputDescriptionsByName
        let unknown = Set(inputs.keys).subtracting(descriptions.keys)
        guard unknown.isEmpty else {
            throw CoreMLBlockRunnerError.invalidFixture("Unknown model inputs: \(unknown.sorted().joined(separator: ", "))")
        }
        let missing = descriptions.filter { !$0.value.isOptional && inputs[$0.key] == nil }.map(\.key)
        guard missing.isEmpty else {
            throw CoreMLBlockRunnerError.invalidFixture("Missing model inputs: \(missing.sorted().joined(separator: ", "))")
        }
        var features = [String: MLFeatureValue]()
        for (name, tensor) in inputs {
            try tensor.validate(name: name)
            guard let constraint = descriptions[name]?.multiArrayConstraint else {
                throw CoreMLBlockRunnerError.unsupportedFeature("\(name): only tensor inputs are supported")
            }
            guard tensor.dtype.mlDataType == constraint.dataType else {
                throw CoreMLBlockRunnerError.invalidFixture("\(name): fixture dtype does not match model dtype")
            }
            // This executor deliberately uses the declared fixed/default shape.
            guard tensor.shape == constraint.shape.map(\.intValue) else {
                throw CoreMLBlockRunnerError.invalidFixture("\(name): shape \(tensor.shape) does not match declared shape \(constraint.shape)")
            }
            let array = try MLMultiArray(shape: tensor.shape.map(NSNumber.init(value:)), dataType: tensor.dtype.mlDataType)
            let offsets = LogicalOffsets(shape: tensor.shape, strides: array.strides.map(\.intValue))
            for (index, value) in tensor.values.enumerated() {
                let offset = offsets.offset(index)
                switch tensor.dtype {
                case .float16: array.dataPointer.assumingMemoryBound(to: Float16.self)[offset] = Float16(value)
                case .float32: array.dataPointer.assumingMemoryBound(to: Float.self)[offset] = Float(value)
                case .float64: array.dataPointer.assumingMemoryBound(to: Double.self)[offset] = value
                case .int32: array.dataPointer.assumingMemoryBound(to: Int32.self)[offset] = Int32(value)
                }
            }
            features[name] = MLFeatureValue(multiArray: array)
        }
        return try MLDictionaryFeatureProvider(dictionary: features)
    }

    private static func read(array: MLMultiArray, name: String) throws -> CoreMLTensor {
        let shape = array.shape.map(\.intValue)
        let dtype = try CoreMLTensorDataType(mlDataType: array.dataType)
        let offsets = LogicalOffsets(shape: shape, strides: array.strides.map(\.intValue))
        var values = [Double]()
        values.reserveCapacity(array.count)
        for index in 0..<array.count {
            let offset = offsets.offset(index)
            let value: Double
            switch dtype {
            case .float16: value = Double(array.dataPointer.assumingMemoryBound(to: Float16.self)[offset])
            case .float32: value = Double(array.dataPointer.assumingMemoryBound(to: Float.self)[offset])
            case .float64: value = array.dataPointer.assumingMemoryBound(to: Double.self)[offset]
            case .int32: value = Double(array.dataPointer.assumingMemoryBound(to: Int32.self)[offset])
            }
            guard value.isFinite else {
                throw CoreMLBlockRunnerError.unsupportedFeature("\(name): non-finite output cannot be encoded as a JSON number")
            }
            values.append(value)
        }
        return CoreMLTensor(shape: shape, dtype: dtype, values: values)
    }

    private static func compare(_ actual: [Double], _ expected: [Double]) -> CoreMLTensorComparison {
        var maximum = 0.0, deltaSquared = 0.0, referenceSquared = 0.0
        for (a, b) in zip(actual, expected) {
            let difference = a - b
            maximum = max(maximum, abs(difference))
            deltaSquared += difference * difference
            referenceSquared += b * b
        }
        return CoreMLTensorComparison(maxAbsoluteError: maximum,
                                      rootMeanSquareError: sqrt(deltaSquared / Double(actual.count)),
                                      relativeL2Error: referenceSquared > 0 ? sqrt(deltaSquared / referenceSquared) : nil,
                                      exactMatch: maximum == 0)
    }

    private static func milliseconds(since start: UInt64) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
    }
}

/// Core ML outputs may be padded or strided, even when their logical shape is fixed.
struct LogicalOffsets {
    let shape: [Int]
    let strides: [Int]
    let contiguous: Bool

    init(shape: [Int], strides: [Int]) {
        self.shape = shape
        self.strides = strides
        var expected = 1
        var contiguous = true
        for axis in shape.indices.reversed() {
            if shape[axis] > 1 && strides[axis] != expected { contiguous = false }
            expected *= shape[axis]
        }
        self.contiguous = contiguous
    }

    func offset(_ logicalIndex: Int) -> Int {
        if contiguous { return logicalIndex }
        var remainder = logicalIndex
        var offset = 0
        for axis in shape.indices.reversed() {
            offset += (remainder % shape[axis]) * strides[axis]
            remainder /= shape[axis]
        }
        return offset
    }
}
