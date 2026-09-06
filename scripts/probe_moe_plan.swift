import CoreML
import Foundation

// Standalone public-API plan helper. Compile separately from the Swift package.
// Plans are anticipated placement; they are not hardware execution traces.
@main struct ProbeMoEPlan {
    static func main() async throws {
        let arguments = CommandLine.arguments
        guard arguments.count == 4 else {
            throw NSError(domain: "Usage: probe-moe-plan model.mlpackage default|fast output.json", code: 1)
        }
        let modelURL = URL(fileURLWithPath: arguments[1])
        let compiled = try await MLModel.compileModel(at: modelURL)
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        if arguments[2] == "fast" {
            configuration.optimizationHints.specializationStrategy = .fastPrediction
        } else if arguments[2] != "default" {
            throw NSError(domain: "Strategy must be default or fast", code: 2)
        }
        let plan = try await MLComputePlan.load(contentsOf: compiled, configuration: configuration)
        var rows = [[String: Any]]()
        func visit(_ block: MLModelStructure.Program.Block) {
            for operation in block.operations {
                if operation.operatorName != "const" {
                    let usage = plan.deviceUsage(for: operation)
                    rows.append(["op": operation.operatorName,
                                 "preferred": usage.map { String(describing: $0.preferred) } ?? "none",
                                 "supported": usage?.supported.map { String(describing: $0) } ?? []])
                }
                for child in operation.blocks { visit(child) }
            }
        }
        if case .program(let program) = plan.modelStructure {
            for function in program.functions.values { visit(function.block) }
        }
        let result: [String: Any] = ["model": modelURL.path, "specialization_strategy": arguments[2],
                                     "compute_units": "CPU_AND_NE", "operations": rows,
                                     "hardware_trace": false]
        try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
            .write(to: URL(fileURLWithPath: arguments[3]))
    }
}
