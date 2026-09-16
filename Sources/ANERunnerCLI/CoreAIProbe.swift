import ANERunnerCore
import Foundation

extension RunnerCLI {
    static func probeCoreAI(_ args: Arguments) async throws {
        try args.validate(["--model", "--fixture", "--function", "--compute-units", "--warmups", "--runs", "--output"])
        #if canImport(CoreAI)
        if #available(macOS 27.0, *) {
            guard let computeUnits = CoreAIComputeUnits(rawValue: args["--compute-units"] ?? "default"),
                  let warmups = Int(args["--warmups"] ?? "3"), let runs = Int(args["--runs"] ?? "5") else {
                throw CLIError.usage("Invalid compute-units, warmups, or runs; see ane-runner help")
            }
            try CoreAIBlockRunner.validateIterations(warmups: warmups, runs: runs)
            let fixture = try CoreMLBlockFixture.load(from: URL(fileURLWithPath: args.require("--fixture")))
            let runner = try await CoreAIBlockRunner(
                modelURL: URL(fileURLWithPath: args.require("--model")),
                functionName: args["--function"] ?? "main", computeUnits: computeUnits)
            let report = try await runner.run(fixture: fixture, warmups: warmups, runs: runs)
            try emit(JSONSerialization.jsonObject(with: JSONEncoder().encode(report)), to: args["--output"])
            return
        }
        #endif
        throw CLIError.usage("probe-coreai requires macOS 27 or newer and a runner built with the macOS 27 SDK")
    }
}
