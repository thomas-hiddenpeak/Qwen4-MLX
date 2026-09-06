import CoreML
import Dispatch
import Foundation
import Synchronization

/// A checked-Sendable handle with exclusive ownership of all mutable Core ML
/// state. Neither the model nor its reusable input/provider escapes the mutex.
/// Distinct handles may predict concurrently; calls on one handle serialize.
final class ConcurrentExpertKernel: Sendable {
    private let state: Mutex<State>

    init(url: URL, manifest: MoEManifest, units: MoEComputeUnits) throws {
        // State is constructed here, then transferred into the mutex. No alias
        // to its non-Sendable Core ML objects is retained by the caller.
        state = Mutex(try State(url: url, manifest: manifest, units: units))
    }

    func predict(tokens: [Float], tokenIndices: [Int]) throws -> ([Float], Double) {
        try state.withLock { state in
            try state.predict(tokens: tokens, tokenIndices: tokenIndices)
        }
    }

    private final class State {
        let model: MLModel
        let input: MLMultiArray
        let provider: MLDictionaryFeatureProvider
        let outputName: String
        let hiddenSize: Int
        let capacity: Int
        let inputOffsets: LogicalOffsets

        init(url: URL, manifest: MoEManifest, units: MoEComputeUnits) throws {
            let configuration = MLModelConfiguration()
            configuration.computeUnits = units.coreML
            model = try MLModel(contentsOf: url, configuration: configuration)
            hiddenSize = manifest.hiddenSize
            capacity = manifest.tokenCapacity
            outputName = manifest.outputName
            let shape = [1, hiddenSize, 1, capacity]
            guard model.modelDescription.inputDescriptionsByName.count == 1,
                let feature = model.modelDescription.inputDescriptionsByName[manifest.inputName]?.multiArrayConstraint,
                feature.shape.map(\.intValue) == shape, feature.dataType == .float16 else {
                throw ExpertRouterError.invalid("Expert input must be FP16 [1,H,1,capacity]")
            }
            input = try MLMultiArray(shape: shape.map(NSNumber.init), dataType: .float16)
            inputOffsets = LogicalOffsets(shape: shape, strides: input.strides.map(\.intValue))
            provider = try MLDictionaryFeatureProvider(dictionary: [manifest.inputName: MLFeatureValue(multiArray: input)])
        }

        func predict(tokens: [Float], tokenIndices: [Int]) throws -> ([Float], Double) {
            guard !tokenIndices.isEmpty, tokenIndices.count <= capacity,
                tokens.count.isMultiple(of: hiddenSize),
                tokenIndices.allSatisfy({ $0 >= 0 && $0 < tokens.count / hiddenSize }) else {
                throw ExpertRouterError.invalid("Invalid expert token chunk")
            }
            let pointer = input.dataPointer.bindMemory(to: Float16.self, capacity: input.count)
            for channel in 0..<hiddenSize {
                for slot in 0..<capacity {
                    if slot < tokenIndices.count {
                        let value = Float16(tokens[tokenIndices[slot] * hiddenSize + channel])
                        guard value.isFinite else { throw ExpertRouterError.invalid("Expert input exceeds finite FP16") }
                        pointer[inputOffsets.offset(channel * capacity + slot)] = value
                    } else {
                        pointer[inputOffsets.offset(channel * capacity + slot)] = 0
                    }
                }
            }
            let start = DispatchTime.now().uptimeNanoseconds
            let result = try model.prediction(from: provider)
            let milliseconds = Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
            guard let array = result.featureValue(for: outputName)?.multiArrayValue,
                array.shape.map(\.intValue) == [1, hiddenSize, 1, capacity], array.dataType == .float16 else {
                throw ExpertRouterError.invalid("Expert output must be FP16 [1,H,1,capacity]")
            }
            let offsets = LogicalOffsets(shape: [1, hiddenSize, 1, capacity], strides: array.strides.map(\.intValue))
            let outputPointer = array.dataPointer.bindMemory(to: Float16.self, capacity: array.count)
            var output = [Float](repeating: 0, count: tokenIndices.count * hiddenSize)
            for slot in tokenIndices.indices {
                for channel in 0..<hiddenSize {
                    output[slot * hiddenSize + channel] = Float(outputPointer[offsets.offset(channel * capacity + slot)])
                }
            }
            guard output.allSatisfy(\.isFinite) else { throw ExpertRouterError.invalid("Nonfinite Core ML expert output") }
            return (output, milliseconds)
        }
    }
}
