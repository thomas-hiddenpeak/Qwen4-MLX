#if canImport(CoreAI)
import CoreAI
import Dispatch
import Foundation

/// An opaque, independent checkpoint belonging to one state session.
/// Restoring a checkpoint copies its buffers again, keeping the checkpoint reusable.
@available(macOS 27.0, *)
public struct CoreAIStateSnapshot {
    fileprivate let owner: UUID
    fileprivate let states: [String: NDArray]
    fileprivate let stepCount: Int
}

/// Diagnostic execution of explicit-state CoreAI subgraphs across prefill/decode.
/// State outputs stay in native NDArrays between steps. Returned host tensors are
/// diagnostic copies and are never used to reconstruct the next state.
///
/// This session is not Sendable. Each operation rejects reentry, including while an
/// inference is suspended. It does not provide production cache or scheduling APIs.
@available(macOS 27.0, *)
public final class CoreAIStateSession {
    private let models: [String: CoreAIBlockRunner]
    private let stateBindings: [String: String]
    private let initialState: [String: NDArray]
    private let owner = UUID()
    private var states: [String: NDArray]
    private let operationGate = NSLock()
    private var operationInProgress = false

    public private(set) var stepCount = 0
    /// Function execution only, from the last successful step; zero after restore/reset.
    public private(set) var lastPredictionMilliseconds = 0.0
    /// Output validation/materialization only; separate from device prediction time.
    public private(set) var lastOutputReadMilliseconds = 0.0

    public init(models: [String: CoreAIBlockRunner], stateBindings: [String: String],
                initialState: [String: CoreMLTensor]) throws {
        guard !models.isEmpty, models.keys.allSatisfy({ !$0.isEmpty }) else {
            throw CoreAIBlockRunnerError.invalidFixture("State session requires named models")
        }
        guard !stateBindings.isEmpty,
              stateBindings.keys.allSatisfy({ !$0.isEmpty }),
              stateBindings.values.allSatisfy({ !$0.isEmpty }),
              Set(stateBindings.values).count == stateBindings.count,
              Set(initialState.keys) == Set(stateBindings.keys) else {
            throw CoreAIBlockRunnerError.invalidFixture("State bindings require unique output names and exactly matching initial state inputs")
        }
        for (modelName, runner) in models {
            let descriptor = runner.function.descriptor
            guard descriptor.stateNames.isEmpty else {
                throw CoreAIBlockRunnerError.unsupportedFeature("\(modelName): native mutable states are unsupported by the explicit-state session")
            }
            guard Set(stateBindings.keys).isSubset(of: descriptor.inputNames),
                  Set(stateBindings.values).isSubset(of: descriptor.outputNames) else {
                throw CoreAIBlockRunnerError.invalidFixture("\(modelName): state bindings are missing from model inputs or outputs")
            }
            // Validate the supported feature kinds up front. Concrete state shapes
            // are checked when selecting a model, because cache length may change.
            for name in descriptor.inputNames {
                guard case .ndArray(let tensor) = descriptor.inputDescriptor(of: name),
                      tensor.interleaveLayout == nil else {
                    throw CoreAIBlockRunnerError.unsupportedFeature("\(modelName)/\(name): only non-interleaved tensor inputs are supported")
                }
                _ = try CoreAIBlockRunner.dtype(tensor.scalarType, name: name)
            }
            for name in descriptor.outputNames {
                guard case .ndArray(let tensor) = descriptor.outputDescriptor(of: name),
                      tensor.interleaveLayout == nil else {
                    throw CoreAIBlockRunnerError.unsupportedFeature("\(modelName)/\(name): only non-interleaved tensor outputs are supported")
                }
                _ = try CoreAIBlockRunner.dtype(tensor.scalarType, name: name)
            }
        }
        var prepared: [String: NDArray] = [:]
        for (name, tensor) in initialState {
            prepared[name] = try CoreAIBlockRunner.makeArray(tensor, name: name)
        }
        self.models = models
        self.stateBindings = stateBindings
        self.initialState = try CoreAITensorCopy.deepCopy(prepared)
        self.states = prepared
    }

    /// Executes one model and commits all returned state only after every output
    /// passes descriptor and finite-value validation. Failure leaves state/count intact.
    public func step(model: String, inputs: [String: CoreMLTensor]) async throws -> [String: CoreMLTensor] {
        try beginOperation()
        defer { endOperation() }
        guard let runner = models[model] else {
            throw CoreAIBlockRunnerError.invalidFixture("Unknown state-session model '\(model)'")
        }
        let nextCount = stepCount.addingReportingOverflow(1)
        guard !nextCount.overflow else {
            throw CoreAIBlockRunnerError.invalidFixture("State-session step count overflows Int")
        }
        let preparedInputs = try runner.makeInputs(inputs, retainedInputs: states)
        try Task.checkCancellation()
        let start = DispatchTime.now().uptimeNanoseconds
        var prediction = try await runner.function.run(inputs: preparedInputs)
        let predictionTime = CoreAIBlockRunner.milliseconds(since: start)
        let readStart = DispatchTime.now().uptimeNanoseconds
        let descriptor = runner.function.descriptor
        guard Set(prediction.names) == Set(descriptor.outputNames) else {
            throw CoreAIBlockRunnerError.invalidFixture("\(model): returned output names do not match the function descriptor")
        }
        var nativeOutputs: [String: NDArray] = [:]
        var hostOutputs: [String: CoreMLTensor] = [:]
        for name in descriptor.outputNames {
            guard let array = prediction.remove(name)?.ndArray,
                  case .ndArray(let tensorDescriptor) = descriptor.outputDescriptor(of: name) else {
                throw CoreAIBlockRunnerError.unsupportedFeature("\(model)/\(name): expected tensor output is missing")
            }
            try CoreAIBlockRunner.validate(array, descriptor: tensorDescriptor, name: name)
            hostOutputs[name] = try CoreAIBlockRunner.read(array, name: name)
            nativeOutputs[name] = array
        }
        var nextState: [String: NDArray] = [:]
        for (inputName, outputName) in stateBindings {
            guard let array = nativeOutputs[outputName] else {
                throw CoreAIBlockRunnerError.invalidFixture("\(model): missing state output '\(outputName)'")
            }
            nextState[inputName] = array
        }
        try Task.checkCancellation()
        let readTime = CoreAIBlockRunner.milliseconds(since: readStart)
        // Native output ownership passes directly to state. Host values above are
        // intentionally not involved in this transition.
        states = nextState
        stepCount = nextCount.partialValue
        lastPredictionMilliseconds = predictionTime
        lastOutputReadMilliseconds = readTime
        return hostOutputs
    }

    public func checkpoint() throws -> CoreAIStateSnapshot {
        try beginOperation()
        defer { endOperation() }
        return CoreAIStateSnapshot(owner: owner, states: try CoreAITensorCopy.deepCopy(states), stepCount: stepCount)
    }

    public func restore(_ snapshot: CoreAIStateSnapshot) throws {
        try beginOperation()
        defer { endOperation() }
        guard snapshot.owner == owner else {
            throw CoreAIBlockRunnerError.invalidFixture("Cannot restore a checkpoint created by another CoreAI state session")
        }
        let restored = try CoreAITensorCopy.deepCopy(snapshot.states)
        states = restored
        stepCount = snapshot.stepCount
        lastPredictionMilliseconds = 0
        lastOutputReadMilliseconds = 0
    }

    public func reset() throws {
        try beginOperation()
        defer { endOperation() }
        let restored = try CoreAITensorCopy.deepCopy(initialState)
        states = restored
        stepCount = 0
        lastPredictionMilliseconds = 0
        lastOutputReadMilliseconds = 0
    }

    private func beginOperation() throws {
        operationGate.lock()
        defer { operationGate.unlock() }
        guard !operationInProgress else {
            throw CoreAIBlockRunnerError.invalidFixture("CoreAI state session already has an operation in progress")
        }
        operationInProgress = true
    }

    private func endOperation() {
        operationGate.lock()
        operationInProgress = false
        operationGate.unlock()
    }
}
#endif
