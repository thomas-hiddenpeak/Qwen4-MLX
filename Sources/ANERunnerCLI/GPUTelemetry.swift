import Darwin
import Foundation
import os

/// Optional diagnostics. The private system APIs run in a separate process;
/// the inference process only records timestamps and owns its child's lifetime.
final class GPUTelemetrySession {
    struct Span {
        let id: UInt64
        let phase: String
        let repetition: Int?
        let step: Int?
        let inputTokens: Int
        let start: UInt64
    }
    let directory: URL
    let runID = UUID().uuidString
    let intervalMilliseconds: Int
    private let process = Process()
    private let log = OSLog(subsystem: "org.ane-runner.telemetry", category: "inference")
    private let errors: FileHandle
    private var events: [[String: Any]] = []
    private var requests: [[String: Any]] = []
    private var active: [UInt64: Span] = [:]
    private var nextID: UInt64 = 1
    private var warnings: [String] = []
    private var droppedEvents = 0
    private var finished = false
    private var collectorStatus = "starting"
    private let maximumEvents = 100_000
    private let startedNS: UInt64
    private static let timebase: mach_timebase_info_data_t = {
        var value = mach_timebase_info_data_t()
        mach_timebase_info(&value)
        return value
    }()

    static func now() -> UInt64 {
        let ticks = mach_absolute_time()
        let n = UInt64(timebase.numer), d = UInt64(timebase.denom)
        return (ticks / d) * n + ((ticks % d) * n) / d
    }

    init(directory: URL, intervalMilliseconds: Int) throws {
        guard (50...10_000).contains(intervalMilliseconds) else {
            throw CLIError.usage("--telemetry-interval-ms must be in 50...10000")
        }
        let fm = FileManager.default
        let location = directory.standardizedFileURL
        guard !fm.fileExists(atPath: location.path) else {
            throw CLIError.usage("--telemetry-dir must be a new directory; existing evidence is never overwritten")
        }
        let executable = (Bundle.main.executableURL ?? URL(fileURLWithPath: CommandLine.arguments[0]))
            .deletingLastPathComponent().appendingPathComponent("ane-telemetry")
        guard fm.isExecutableFile(atPath: executable.path) else {
            throw CLIError.usage("Missing sibling ane-telemetry executable; build all Swift package products")
        }
        try fm.createDirectory(at: location.deletingLastPathComponent(), withIntermediateDirectories: true)
        // The initial existence check is only a friendly preflight. mkdir
        // claims the final directory atomically, including concurrent callers.
        guard location.path.withCString({ mkdir($0, S_IRWXU) }) == 0 else {
            let reason = String(cString: strerror(errno))
            throw CLIError.usage("Cannot claim new telemetry directory: \(reason)")
        }
        let stderrURL = location.appendingPathComponent("collector.stderr.log")
        guard fm.createFile(atPath: stderrURL.path, contents: nil) else {
            throw CLIError.usage("Cannot create telemetry collector log")
        }
        self.directory = location
        self.intervalMilliseconds = intervalMilliseconds
        self.errors = try FileHandle(forWritingTo: stderrURL)
        self.startedNS = Self.now()
        process.executableURL = executable
        process.arguments = ["--output", location.appendingPathComponent("hardware.jsonl").path,
                             "--interval-ms", String(intervalMilliseconds), "--max-samples", "100000",
                             "--pid", String(ProcessInfo.processInfo.processIdentifier)]
        process.standardInput = FileHandle.nullDevice
        process.standardOutput = FileHandle.nullDevice
        process.standardError = errors
        do { try process.run() }
        catch { try? errors.close(); throw error }
        // A complete baseline precedes model loading. Failure is explicit, while
        // subsequent sampling failures never invalidate model state.
        let deadline = Self.now() + 3_000_000_000
        let hardware = location.appendingPathComponent("hardware.jsonl")
        var ready = false
        while process.isRunning && Self.now() < deadline {
            if let bytes = try? Data(contentsOf: hardware), let firstEnd = bytes.firstIndex(of: 10),
               let secondEnd = bytes[(firstEnd + 1)...].firstIndex(of: 10),
               let row = try? JSONSerialization.jsonObject(with: bytes.subdata(in: (firstEnd + 1)..<secondEnd)),
               let baseline = row as? [String: Any], baseline["type"] as? String == "baseline" {
                ready = true
                break
            }
            Thread.sleep(forTimeInterval: 0.01)
        }
        collectorStatus = ready ? "started" : "startup_baseline_unavailable"
        if !ready { warnings.append("Collector did not publish a complete baseline before model loading; initial hardware coverage may be missing. Check hardware.jsonl and stderr.") }
        anchor()
    }

    private func anchor() {
        let ns = Self.now()
        os_signpost(.event, log: log, name: "TelemetryAnchor", "run=%{public}@ uptime_ns=%{public}llu", runID, ns)
    }

    func begin(_ phase: String, repetition: Int? = nil, step: Int? = nil, inputTokens: Int = 0) -> Span {
        let id = nextID; nextID += 1
        let span = Span(id: id, phase: phase, repetition: repetition, step: step,
                        inputTokens: inputTokens, start: Self.now())
        active[id] = span
        os_signpost(.begin, log: log, name: "InferencePhase", signpostID: OSSignpostID(id),
                    "run=%{public}@ phase=%{public}@ repetition=%d step=%d uptime_ns=%{public}llu",
                    runID, phase, repetition ?? -1, step ?? -1, span.start)
        return span
    }

    func end(_ span: Span, outputTokens: Int = 0, forwardEndNS: UInt64? = nil,
             evaluationEndNS: UInt64? = nil, ssdWaitSeconds: Double? = nil,
             ssdRequestedBytes: Int? = nil, succeeded: Bool = true) {
        guard active.removeValue(forKey: span.id) != nil else { return }
        let end = Self.now()
        os_signpost(.end, log: log, name: "InferencePhase", signpostID: OSSignpostID(span.id),
                    "run=%{public}@ phase=%{public}@ uptime_ns=%{public}llu", runID, span.phase, end)
        let event: [String: Any] = [
            "type": "interval", "phase": span.phase, "start_ns": span.start, "end_ns": end,
            "repetition": span.repetition as Any? ?? NSNull(), "step_index": span.step as Any? ?? NSNull(),
            "input_tokens": span.inputTokens, "output_tokens": outputTokens, "succeeded": succeeded,
            "forward_end_ns": forwardEndNS as Any? ?? NSNull(),
            "evaluation_end_ns": evaluationEndNS as Any? ?? NSNull(),
            "ssd_wait_seconds": ssdWaitSeconds as Any? ?? NSNull(),
            "ssd_requested_row_bytes": ssdRequestedBytes as Any? ?? NSNull()
        ]
        if events.count < maximumEvents { events.append(event) } else { droppedEvents += 1 }
    }

    func request(repetition: Int, startNS: UInt64, prefillEndNS: UInt64,
                 decodeStartNS: UInt64?, endNS: UInt64) {
        requests.append(["repetition": repetition, "start_ns": startNS, "prefill_end_ns": prefillEndNS,
                         "decode_start_ns": decodeStartNS as Any? ?? NSNull(), "end_ns": endNS])
    }

    func finish(succeeded: Bool) {
        guard !finished else { return }
        for span in Array(active.values) { end(span, succeeded: false) }
        anchor()
        if process.isRunning {
            process.terminate()
            let deadline = Self.now() + 2_000_000_000
            while process.isRunning && Self.now() < deadline { Thread.sleep(forTimeInterval: 0.01) }
            if process.isRunning {
                // This Process owns the still-running child; never signal a
                // PID taken from a previous run's report.
                _ = kill(process.processIdentifier, SIGKILL)
                warnings.append("Collector did not exit after SIGTERM and was killed; its final sample may be missing.")
            }
        }
        collectorStatus = process.isRunning ? "termination_pending" : (process.terminationStatus == 0 ? "completed" : "collector_error")
        if collectorStatus != "completed" { warnings.append("Telemetry collector did not complete cleanly; inference results are independent of collector success.") }
        finished = true
        try? errors.close()
        do {
            var data = try JSONSerialization.data(withJSONObject: [
                "type": "metadata", "schema_version": 1, "run_id": runID,
                "clock": "mach_absolute_time_nanoseconds", "target_pid": ProcessInfo.processInfo.processIdentifier,
                "mach_timebase_numer": Self.timebase.numer, "mach_timebase_denom": Self.timebase.denom,
                "started_ns": startedNS, "sampling_interval_ms": intervalMilliseconds,
                "inference_succeeded": succeeded, "collector_status": collectorStatus
            ], options: [.sortedKeys])
            data.append(10)
            for event in events {
                data.append(try JSONSerialization.data(withJSONObject: event, options: [.sortedKeys])); data.append(10)
            }
            try data.write(to: directory.appendingPathComponent("phases.jsonl"), options: .atomic)
            try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
                .write(to: directory.appendingPathComponent("session.json"), options: .atomic)
        } catch {
            warnings.append("Could not save telemetry phase files: \(error.localizedDescription)")
            FileHandle.standardError.write(Data("Telemetry output error: \(error.localizedDescription)\n".utf8))
        }
    }

    var report: [String: Any] {
        ["schema_version": 1, "enabled": true, "run_id": runID, "clock": "mach_absolute_time_nanoseconds",
         "started_ns": startedNS, "target_pid": ProcessInfo.processInfo.processIdentifier,
         "sidecar_pid": process.processIdentifier, "collector_status": collectorStatus,
         "sidecar_file": directory.appendingPathComponent("hardware.jsonl").path,
         "phase_file": directory.appendingPathComponent("phases.jsonl").path,
         "sampling_interval_ms": intervalMilliseconds, "events": events, "request_windows": requests,
         "dropped_events": droppedEvents, "warnings": warnings,
         "physical_dram_bytes": NSNull(), "physical_dram_bandwidth_gbps": NSNull(),
         "notes": ["All timestamps use mach_absolute_time converted with the host timebase.",
                   "GPU counters require a separately configured Instruments capture; hardware.jsonl alone does not supply them.",
                   "IOReport histograms are coarse system signals; target-process disk counters and device disk totals have different scopes.",
                   "Forward wall time includes graph building, any explicit SSD wait and early GPU submission. Evaluation wait is not pure GPU time.",
                   "Phases are recorded in memory and saved after sampling stops; signposts are emitted live. External termination may prevent phase-file persistence.",
                   "Profiler/collector overhead must be evaluated against an uninstrumented run."]]
    }
}

extension RunnerCLI {
    /// A short idle diagnostic validates sampler startup, clocks and shutdown
    /// without allocating model weights or presenting idle values as inference.
    static func probeTelemetry(_ args: Arguments) throws {
        try args.validate(["--telemetry-dir", "--telemetry-interval-ms", "--seconds", "--output"])
        guard let interval = Int(args["--telemetry-interval-ms"] ?? "200"),
              let seconds = Double(args["--seconds"] ?? "2"), seconds.isFinite,
              (0.2...10).contains(seconds) else { throw CLIError.usage("Invalid telemetry probe duration or interval") }
        let session = try GPUTelemetrySession(directory: URL(fileURLWithPath: args.require("--telemetry-dir")),
                                              intervalMilliseconds: interval)
        defer { session.finish(succeeded: false) }
        let span = session.begin("diagnostic_idle")
        Thread.sleep(forTimeInterval: seconds)
        session.end(span)
        session.finish(succeeded: true)
        try emit(["scope": "Idle sampler lifecycle diagnostic; no model inference or bandwidth benchmark",
                  "telemetry": session.report], to: args["--output"])
    }
}
