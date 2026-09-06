import ANERunnerGPU
import Darwin
import Foundation

/// Optional symbols exist only in the isolated instrumented MLX library.
/// Normal builds have no dependency on a patched MLX ABI.
final class GPUCommandTimingSession {
    private typealias Start = @convention(c) (UnsafePointer<CChar>?) -> Int32
    private typealias Stop = @convention(c) () -> Int32
    private typealias Version = @convention(c) () -> UnsafePointer<CChar>?
    private let library: UnsafeMutableRawPointer
    private let begin: Start
    private let end: Stop
    let path: String
    let version: String
    private var started = false
    private var finished = false
    private var steps: [[String: Any]] = []

    init(path: String) throws {
        let url = URL(fileURLWithPath: path).standardizedFileURL
        guard !FileManager.default.fileExists(atPath: url.path) else {
            throw CLIError.usage("GPU command timing output must be a new file")
        }
        guard let library = dlopen(nil, RTLD_NOW) else { throw CLIError.usage("Cannot inspect loaded MLX library") }
        guard let start = dlsym(library, "anemlx_timing_start"),
              let stop = dlsym(library, "anemlx_timing_stop"),
              let versionSymbol = dlsym(library, "anemlx_timing_version") else {
            dlclose(library)
            throw CLIError.usage("GPU command timing requires the isolated instrumented MLX library; see GPU_BOTTLENECK.md")
        }
        let getVersion = unsafeBitCast(versionSymbol, to: Version.self)
        guard let text = getVersion() else { dlclose(library); throw CLIError.usage("Missing timing hook version") }
        self.library = library
        begin = unsafeBitCast(start, to: Start.self)
        end = unsafeBitCast(stop, to: Stop.self)
        version = String(cString: text)
        self.path = url.path
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    }
    deinit { dlclose(library) }

    func start() throws {
        try MX.synchronize()
        let status = path.withCString { begin($0) }
        guard status == 0 else { throw CLIError.usage("GPU command timing start failed (status \(status))") }
        started = true
    }

    func finish() throws {
        guard started, !finished else { return }
        try MX.synchronize()
        let status = end()
        finished = true
        guard status == 0 else { throw CLIError.usage("GPU command timing flush failed (status \(status)); inspect native report") }
    }

    func step(phase: String, repetition: Int, index: Int, inputTokens: Int,
              start: UInt64, forwardEnd: UInt64, evaluationEnd: UInt64) {
        steps.append(["phase": phase, "repetition": repetition, "step": index,
                      "input_tokens": inputTokens, "start_ns": start,
                      "forward_end_ns": forwardEnd, "evaluation_end_ns": evaluationEnd])
    }

    var report: [String: Any] {
        ["enabled": true, "hook_version": version, "output": path,
         "started": started, "finished": finished, "clock": "mach_absolute_time nanoseconds since boot",
         "steps": steps,
         "scope": "CPU step markers plus actual MLX command-buffer GPU intervals. Buffer spans can contain stalls; they are not shader utilization or DRAM bandwidth. Instrumentation overhead is not assumed zero."]
    }

    private static let nanosecondsPerTick: Double = {
        var info = mach_timebase_info_data_t()
        mach_timebase_info(&info)
        return Double(info.numer) / Double(info.denom)
    }()
    static func now() -> UInt64 { UInt64(Double(mach_absolute_time()) * nanosecondsPerTick) }
}

extension RunnerCLI {
    static func probeGPUCommandTiming(_ args: Arguments) throws {
        try args.validate(["--gpu-command-timing-output", "--output"])
        let timing = try GPUCommandTimingSession(path: args.require("--gpu-command-timing-output"))
        if let output = args["--output"], URL(fileURLWithPath: output).standardizedFileURL.path == timing.path {
            throw CLIError.usage("Probe and native timing reports require different paths")
        }
        defer { try? timing.finish() }
        try timing.start()
        var outputs: [[Float]] = []
        for i in 1...3 {
            let x = try MX.array([Float](repeating: Float(i), count: 4), shape: [1,4])
            let weight = try MX.ones([4,4], x.dtype)
            let start = GPUCommandTimingSession.now()
            let y = try MX.matmul(x, weight)
            let forward = GPUCommandTimingSession.now()
            let values = try y.floats()
            let end = GPUCommandTimingSession.now()
            guard values == [Float](repeating: Float(4*i), count: 4) else {
                throw CLIError.usage("Command timing probe changed matrix results")
            }
            outputs.append(values)
            timing.step(phase: "probe", repetition: 0, index: i-1, inputTokens: 0,
                        start: start, forwardEnd: forward, evaluationEnd: end)
        }
        try timing.finish()
        try emit(["scope": "Small matrix capability check; not full-model inference or a throughput benchmark",
                  "passed": true, "outputs": outputs,
                  "provenance": ["process_id": ProcessInfo.processInfo.processIdentifier],
                  "gpu_command_timing": timing.report], to: args["--output"])
    }
}
