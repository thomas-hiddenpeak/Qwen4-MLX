import ANERunnerCore
import CryptoKit
import Dispatch
import Foundation

private struct CoreAISequencePlan: Decodable {
    struct Model: Decodable { let path: String; let function: String? }
    struct Step: Decodable { let name: String; let phase: String; let model: String; let fixture: String }
    struct Tolerances: Decodable {
        let maximumAbsoluteError: Double
        let relativeL2Error: Double
        let zeroAbsoluteError: Double?
    }
    let version: Int
    let models: [String: Model]
    let stateBindings: [String: String]
    let initialState: String
    let steps: [Step]
    let tolerances: Tolerances

    func validate() throws {
        guard version == 1, (1...16).contains(models.count), (1...512).contains(steps.count),
              !stateBindings.isEmpty, !initialState.isEmpty,
              Set(steps.map(\.name)).count == steps.count,
              steps.allSatisfy({ !$0.name.isEmpty && !$0.fixture.isEmpty && models[$0.model] != nil && ["prefill", "decode"].contains($0.phase) }),
              models.allSatisfy({ !$0.key.isEmpty && !$0.value.path.isEmpty }),
              [tolerances.maximumAbsoluteError, tolerances.relativeL2Error, tolerances.zeroAbsoluteError ?? 1e-6]
                .allSatisfy({ $0.isFinite && $0 >= 0 }) else {
            throw CLIError.usage("Invalid version-1 CoreAI sequence plan, phases, models, or tolerances")
        }
    }
}

extension RunnerCLI {
    static func probeCoreAISequence(_ args: Arguments) async throws {
        try args.validate(["--sequence", "--compute-units", "--replays", "--checkpoint-step", "--output"])
        #if canImport(CoreAI)
        if #available(macOS 27.0, *) {
            try await runCoreAISequence(args)
            return
        }
        #endif
        throw CLIError.usage("probe-coreai-sequence requires macOS 27 and a runner built with its SDK")
    }

    #if canImport(CoreAI)
    @available(macOS 27.0, *)
    private static func runCoreAISequence(_ args: Arguments) async throws {
        guard let units = CoreAIComputeUnits(rawValue: args["--compute-units"] ?? "gpu"),
              let replays = Int(args["--replays"] ?? "2"), (1...4).contains(replays) else {
            throw CLIError.usage("Invalid compute units or replay count (1...4)")
        }
        let path = URL(fileURLWithPath: try args.require("--sequence")).standardizedFileURL
        let base = path.deletingLastPathComponent()
        func resolve(_ value: String) -> URL { URL(fileURLWithPath: value, relativeTo: base).standardizedFileURL }
        let plan = try JSONDecoder().decode(CoreAISequencePlan.self, from: Data(contentsOf: path))
        try plan.validate()
        let defaultCheckpoint = plan.steps.count > 1 ? 1 : 0
        guard let checkpointStep = Int(args["--checkpoint-step"] ?? String(defaultCheckpoint)),
              checkpointStep >= 0, checkpointStep < plan.steps.count else {
            throw CLIError.usage("checkpoint-step must be 0 (disabled) or a step count smaller than the sequence length")
        }
        let initial = try JSONDecoder().decode([String: CoreMLTensor].self, from: Data(contentsOf: resolve(plan.initialState)))
        let fixtures = try plan.steps.map { try CoreMLBlockFixture.load(from: resolve($0.fixture)) }
        guard fixtures.allSatisfy({ fixture in
            fixture.expectedOutputs?.isEmpty == false && Set(fixture.inputs.keys).isDisjoint(with: plan.stateBindings.keys)
        }) else {
            throw CLIError.usage("Each step needs expected outputs and must not supply reference state as ordinary inputs")
        }
        let loadStart = DispatchTime.now().uptimeNanoseconds
        var models = [String: CoreAIBlockRunner]()
        for key in plan.models.keys.sorted() {
            let spec = plan.models[key]!
            models[key] = try await CoreAIBlockRunner(modelURL: resolve(spec.path), functionName: spec.function ?? "main", computeUnits: units)
        }
        let loadMilliseconds = Double(DispatchTime.now().uptimeNanoseconds - loadStart) / 1e6
        let session = try CoreAIStateSession(models: models, stateBindings: plan.stateBindings, initialState: initial)
        var records = [[String: Any]]()
        var baselineDigests = [Int: [String: String]]()
        var checkpoint: CoreAIStateSnapshot?
        var failure: String?
        var allPassed = true
        var checkpointRestores = 0

        func execute(_ index: Int, pass: String, baseline: Bool) async throws {
            let step = plan.steps[index]
            let fixture = fixtures[index]
            let output = try await session.step(model: step.model, inputs: fixture.inputs)
            guard let expected = fixture.expectedOutputs, Set(output.keys) == Set(expected.keys) else {
                throw CLIError.usage("\(step.name): reference must cover every model output, including all states")
            }
            var checks = [String: Any]()
            var digests = [String: String]()
            var stepPassed = session.stepCount == index + 1
            for name in output.keys.sorted() {
                let actual = output[name]!, reference = expected[name]!
                guard actual.shape == reference.shape, actual.values.count == reference.values.count else {
                    throw CLIError.usage("\(step.name)/\(name): output shape differs from reference")
                }
                var maxError = 0.0, errorSquared = 0.0, referenceSquared = 0.0
                for (a, b) in zip(actual.values, reference.values) {
                    let delta = a - b
                    maxError = max(maxError, abs(delta))
                    errorSquared += delta * delta
                    referenceSquared += b * b
                }
                guard maxError.isFinite, errorSquared.isFinite, referenceSquared.isFinite else {
                    throw CLIError.usage("\(step.name)/\(name): nonfinite comparison")
                }
                let relative = referenceSquared > 0 ? sqrt(errorSquared / referenceSquared) : nil
                let passed: Bool
                if actual.dtype == .int32 || reference.dtype == .int32 {
                    passed = actual.dtype == .int32 && reference.dtype == .int32 && maxError == 0
                } else if let relative {
                    passed = maxError <= plan.tolerances.maximumAbsoluteError && relative <= plan.tolerances.relativeL2Error
                } else {
                    passed = maxError <= (plan.tolerances.zeroAbsoluteError ?? 1e-6)
                }
                let digest = logicalTensorDigest(actual)
                digests[name] = digest
                let replayMatches = baseline || baselineDigests[index]?[name] == digest
                stepPassed = stepPassed && passed && replayMatches
                var check: [String: Any] = [
                    "shape": actual.shape, "dtype": actual.dtype.rawValue, "elements": actual.values.count,
                    "maxAbsoluteError": maxError, "rootMeanSquareError": sqrt(errorSquared / Double(actual.values.count)),
                    "exactReferenceMatch": maxError == 0, "referencePassed": passed,
                    "logicalDoubleSHA256": digest, "replayMatches": replayMatches,
                ]
                if let relative { check["relativeL2Error"] = relative }
                checks[name] = check
            }
            if baseline { baselineDigests[index] = digests }
            allPassed = allPassed && stepPassed
            records.append([
                "pass": pass, "index": index, "name": step.name, "phase": step.phase,
                "stepCount": session.stepCount, "passed": stepPassed, "outputs": checks,
                "predictionMilliseconds": session.lastPredictionMilliseconds,
                "outputReadMilliseconds": session.lastOutputReadMilliseconds,
            ])
        }
        do {
            for replay in 0..<replays {
                if replay > 0 { try session.reset() }
                for index in plan.steps.indices {
                    try await execute(index, pass: replay == 0 ? "initial" : "reset-\(replay)", baseline: replay == 0)
                    if replay == 0 && index + 1 == checkpointStep { checkpoint = try session.checkpoint() }
                }
            }
            // Restore the same checkpoint twice: its contents must survive all later steps.
            if let checkpoint {
                for round in 1...2 {
                    try session.restore(checkpoint)
                    checkpointRestores += 1
                    for index in checkpointStep..<plan.steps.count {
                        try await execute(index, pass: "checkpoint-\(round)", baseline: false)
                    }
                }
            }
        } catch {
            failure = String(describing: error)
            allPassed = false
        }
        var report: [String: Any] = [
            "schema": "coreai-sequence-report-v1", "sequence": path.path,
            "operatingSystem": ProcessInfo.processInfo.operatingSystemVersionString,
            "computeUnits": units.rawValue, "modelLoadMilliseconds": loadMilliseconds,
            "replays": replays, "checkpointAfterStep": checkpointStep, "checkpointRestores": checkpointRestores,
            "stateBindings": plan.stateBindings, "steps": records, "passed": allPassed,
            "stateTransfer": "Actual CoreAI NDArray outputs are retained across steps; CPU reference states are never fed back. Checkpoints/reset use independent typed copies.",
            "timingScope": "Function calls and diagnostic output reads are separate. This is not a full-model prefill/decode or throughput benchmark.",
            "hardwareEvidence": "Compute unit preferences permit fallback. No device execution trace is included.",
        ]
        if let failure { report["error"] = failure }
        try emit(report, to: args["--output"])
        guard allPassed else { throw CLIError.usage("CoreAI sequence validation failed; see report\(failure.map { ": " + $0 } ?? "")") }
    }

    /// Digest logical values and metadata, not physical device storage or padding.
    @available(macOS 27.0, *)
    private static func logicalTensorDigest(_ tensor: CoreMLTensor) -> String {
        var hash = SHA256()
        hash.update(data: Data(tensor.dtype.rawValue.utf8))
        tensor.shape.withUnsafeBytes { hash.update(bufferPointer: $0) }
        tensor.values.withUnsafeBytes { hash.update(bufferPointer: $0) }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }
    #endif
}
