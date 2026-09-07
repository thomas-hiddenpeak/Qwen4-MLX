import CMLX
import Foundation

/// Explicit diagnostics for a single inference thread. This is deliberately
/// not Sendable: neither stage nesting nor concurrent use is supported.
/// Synchronizing every stage changes scheduling, reuse and overlap; those
/// results are not an undisturbed end-to-end throughput benchmark.
public final class GPUProfiler {
    public enum Mode: String, Codable {
        case disabled
        case hostBodyOnly
        case synchronizedStages
    }
    public struct Stage: Codable {
        public let name: String
        public let layer: Int?
        public let tokenCount: Int
        /// Explicit business phase and absolute trunk offset, independent of S.
        /// Absent in historical reports and probes outside a model forward.
        public var phase: QwenExecutionPhase? = nil
        public var position: Int? = nil
        /// DispatchTime uptime interval, excluding the preceding stream drain.
        /// Missing in historical reports. Failed intervals are diagnostic only.
        public var startedUptimeNanoseconds: UInt64? = nil
        public var endedUptimeNanoseconds: UInt64? = nil
        public let hostBodyMilliseconds: Double
        public let outputCollectionMilliseconds: Double
        public let evaluationWaitMilliseconds: Double?
        public let elapsedMilliseconds: Double
        public let precedingStreamDrainMilliseconds: Double
        public let evaluatedTensorCount: Int
        public let logicalWeightBytes: UInt64?
        public let allocatorBefore: [String: Int]?
        public let allocatorAfter: [String: Int]?
        public let succeeded: Bool
        public let error: String?
    }
    public struct Report: Codable {
        public let mode: Mode
        public var phaseFilter: QwenExecutionPhase? = nil
        public let attentionBreakdown: Bool?
        public let moeBreakdown: Bool?
        public let stages: [Stage]
        public let droppedRecords: Int
        public let actualDRAMBytesAvailable: Bool
        public let deviceOnlyTimeAvailable: Bool
        public let notes: [String]
    }
    public let mode: Mode
    public let phaseFilter: QwenExecutionPhase?
    public let attentionBreakdown: Bool
    public let moeBreakdown: Bool
    public var isRecording: Bool { mode != .disabled && recordingEnabled }
    public private(set) var stages = [Stage]()
    public private(set) var droppedRecords = 0
    private let allocatorSnapshots: Bool
    private let maximumRecords: Int
    private var active = false
    private var recordingEnabled = true
    private var phase: QwenExecutionPhase?
    private var position: Int?

    public init(mode: Mode = .disabled, allocatorSnapshots: Bool = false, maximumRecords: Int = 4096,
                attentionBreakdown: Bool = false, moeBreakdown: Bool = false,
                phaseFilter: QwenExecutionPhase? = nil) throws {
        guard maximumRecords > 0 else { throw GPUError.invalid("Profiler record limit must be positive") }
        self.mode = mode
        self.phaseFilter = phaseFilter
        self.allocatorSnapshots = allocatorSnapshots
        self.maximumRecords = maximumRecords
        self.attentionBreakdown = attentionBreakdown
        self.moeBreakdown = moeBreakdown
    }

    /// Select baseline/diagnostic work on one inference executor, never inside
    /// an active stage. Disabled recording performs no clocks/eval/collection.
    public func setRecordingEnabled(_ enabled: Bool) throws {
        guard !active else { throw GPUError.invalid("Cannot change recording inside an active profiler stage") }
        recordingEnabled = enabled
    }

    func setForwardContext(phase: QwenExecutionPhase, position: Int) throws {
        guard !active, position >= 0 else { throw GPUError.invalid("Invalid profiler forward context") }
        self.phase = phase; self.position = position
    }

    func clearForwardContext() { phase = nil; position = nil }

    public func reset() throws {
        guard !active else { throw GPUError.invalid("Cannot reset an active profiler stage") }
        stages.removeAll(keepingCapacity: true)
        droppedRecords = 0
        clearForwardContext()
    }

    /// `outputs` must include every newly written persistent state, not only y.
    /// Disabled or unmatched phases call only `body`, with no clocks, output
    /// extraction, synchronization or allocator queries. A phase filter skips
    /// work without forward context. No tensor readback occurs here.
    public func measure<T>(_ name: String, layer: Int? = nil, tokenCount: Int,
                           logicalWeightBytes: UInt64? = nil,
                           outputs: (T) -> [Tensor], _ body: () throws -> T) throws -> T {
        if !isRecording || (phaseFilter != nil && phase != phaseFilter) { return try body() }
        guard !active, !name.isEmpty, tokenCount > 0 else {
            throw GPUError.invalid("Invalid or nested profiler stage")
        }
        active = true
        defer { active = false }
        var drain = 0.0
        if mode == .synchronizedStages {
            let before = Self.now()
            try MX.synchronize()
            drain = Self.milliseconds(since: before)
        }
        let memoryBefore = allocatorSnapshots ? try MX.memory() : nil
        let start = Self.now()
        var bodyMilliseconds = 0.0, collectMilliseconds = 0.0
        var waitMilliseconds: Double?, count = 0
        do {
            let value = try body()
            bodyMilliseconds = Self.milliseconds(since: start)
            if mode == .synchronizedStages {
                let collect = Self.now()
                let tensors = outputs(value)
                count = tensors.count
                collectMilliseconds = Self.milliseconds(since: collect)
                guard !tensors.isEmpty else { throw GPUError.invalid("A synchronized stage needs result/state tensors") }
                let evaluate = Self.now()
                try MX.eval(tensors)
                try MX.synchronize()
                waitMilliseconds = Self.milliseconds(since: evaluate)
            }
            let end = Self.now()
            let elapsed = Double(end - start) * 1e-6
            let after = allocatorSnapshots ? try MX.memory() : nil
            append(Stage(name: name,layer: layer,tokenCount: tokenCount,
                         phase: phase,position: position,
                         startedUptimeNanoseconds: start,endedUptimeNanoseconds: end,
                         hostBodyMilliseconds: bodyMilliseconds,outputCollectionMilliseconds: collectMilliseconds,
                         evaluationWaitMilliseconds: waitMilliseconds,elapsedMilliseconds: elapsed,
                         precedingStreamDrainMilliseconds: drain,evaluatedTensorCount: count,
                         logicalWeightBytes: logicalWeightBytes,allocatorBefore: memoryBefore,allocatorAfter: after,
                         succeeded: true,error: nil))
            return value
        } catch {
            // Best-effort drain on failure; preserve the original error. A
            // failed record has no valid stage timing split or bandwidth.
            if mode == .synchronizedStages { try? MX.synchronize() }
            let end = Self.now()
            append(Stage(name: name,layer: layer,tokenCount: tokenCount,
                         phase: phase,position: position,
                         startedUptimeNanoseconds: start,endedUptimeNanoseconds: end,
                         hostBodyMilliseconds: bodyMilliseconds,outputCollectionMilliseconds: collectMilliseconds,
                         evaluationWaitMilliseconds: nil,elapsedMilliseconds: Double(end - start) * 1e-6,
                         precedingStreamDrainMilliseconds: drain,evaluatedTensorCount: count,
                         logicalWeightBytes: logicalWeightBytes,allocatorBefore: memoryBefore,allocatorAfter: nil,
                         succeeded: false,error: String(describing: error)))
            throw error
        }
    }

    public var report: Report {
        Report(mode: mode,phaseFilter: phaseFilter,attentionBreakdown: attentionBreakdown,moeBreakdown: moeBreakdown,stages: stages,droppedRecords: droppedRecords,
               actualDRAMBytesAvailable: false,deviceOnlyTimeAvailable: false,
               notes: [
                "hostBodyMilliseconds is host closure wall time: usually lazy graph construction, but includes any explicit I/O/evaluation inside the body.",
                "evaluationWaitMilliseconds includes lazy graph traversal, compilation/encoding, scheduling, device work and synchronization. It is not GPU-only time.",
                "The preceding drain waits only already submitted work. Unmaterialized dependencies still run with the stage that evaluates them.",
                "Synchronized stages remove normal overlap and may change allocator reuse. Their totals are not undisturbed prefill or decode latency.",
                "phase and position identify the enclosing trunk forward, including a final S1 prefill. Detailed attention or MoE stages replace the corresponding outer stage and are not added to an inclusive parent.",
                "When phaseFilter is set, unmatched or absent forward context runs without profiler clocks, synchronization, output collection or allocator snapshots.",
                "startedUptimeNanoseconds and endedUptimeNanoseconds use DispatchTime uptime, matching command timing. They bound elapsedMilliseconds after the preceding drain and before the final allocator snapshot; failed stages are excluded from performance analysis.",
                "Allocator active/cache/peak are capacity counters, not DRAM traffic. logicalWeightBytes is a caller-supplied estimate, not hardware bytes.",
                "No DRAM read/write or physical SSD byte counter is exposed by this helper. No bandwidth is inferred from allocations or model sizes."
               ])
    }

    private func append(_ stage: Stage) {
        if stages.count < maximumRecords { stages.append(stage) }
        else { droppedRecords += 1 }
    }
    private static func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
    private static func milliseconds(since start: UInt64) -> Double { Double(now()-start)*1e-6 }

    /// Capture through the pinned public MLX C API. Call only for an explicitly
    /// requested trace; it captures this Metal device, not an isolated layer.
    /// Output/state evaluation is inside the capture so lazy graphs are included.
    public static func capture<T>(at url: URL, outputs: (T) -> [Tensor], _ body: () throws -> T) throws -> T {
        guard url.isFileURL, !FileManager.default.fileExists(atPath: url.path) else {
            throw GPUError.invalid("Metal capture needs a new local output path")
        }
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(),withIntermediateDirectories: true)
        try MX.synchronize()
        try MX.check(mlx_metal_start_capture(url.path),"start Metal capture")
        do {
            let value = try body()
            let tensors = outputs(value)
            guard !tensors.isEmpty else { throw GPUError.invalid("Metal capture needs result/state tensors") }
            try MX.eval(tensors)
            try MX.synchronize()
            try MX.check(mlx_metal_stop_capture(),"stop Metal capture")
            return value
        } catch {
            try? MX.synchronize()
            _ = mlx_metal_stop_capture()
            throw error
        }
    }
}
