import CMLX
import Foundation

/// Opt-in routing diagnostics owned by one inference thread. Append only retains
/// indices; finish must run after all measured requests and their evaluations.
/// Keeping these references can change graph/buffer lifetimes, so captured runs
/// are diagnostics, not undisturbed throughput or physical bandwidth evidence.
public final class GPUVerificationRoutingCapture {
    public struct Record: Codable, Sendable {
        public let repetition: Int
        public let phase: QwenExecutionPhase
        public let position: Int
        public let tokenCount: Int
        public let layer: Int
        /// Original routing slot order, without sorting or deduplication.
        public let expertIDs: [[Int32]]
    }

    public struct Report: Codable, Sendable {
        public let enabled: Bool
        public let finished: Bool
        public let maximumRecords: Int
        public let droppedRecords: Int
        /// Deferred ints() conversion/evaluation/readback and validation wall time.
        /// This is neither device-only time nor part of measured generation.
        public let readbackMilliseconds: Double
        public let expertCount: Int
        public let topK: Int
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let bits: Int
        public let groupSize: Int
        public let records: [Record]
        public let notes: [String]
    }

    private struct Pending {
        let repetition: Int
        let phase: QwenExecutionPhase
        let position: Int
        let tokenCount: Int
        let layer: Int
        let indices: Tensor
    }

    public let maximumRecords: Int
    public private(set) var droppedRecords = 0
    public private(set) var finished = false
    private var pending = [Pending]()

    public init(maximumRecords: Int = 8192) throws {
        guard maximumRecords > 0 else {
            throw GPUError.invalid("Routing capture record limit must be positive")
        }
        self.maximumRecords = maximumRecords
    }

    /// Shape/dtype metadata only. No evaluation, readback, copy or clock here.
    public func append(repetition: Int, phase: QwenExecutionPhase, position: Int,
                       tokenCount: Int, layer: Int, indices: Tensor) throws {
        guard !finished else { throw GPUError.invalid("Routing capture already finished") }
        try Self.validateMetadata(repetition: repetition, phase: phase, position: position,
                                  tokenCount: tokenCount, layer: layer,
                                  shape: indices.shape, dtype: indices.dtype)
        guard pending.count < maximumRecords else {
            droppedRecords += 1
            return
        }
        pending.append(Pending(repetition: repetition, phase: phase, position: position,
                               tokenCount: tokenCount, layer: layer, indices: indices))
    }

    /// Consumes the capture once, also on readback/validation failure. All held
    /// tensor references are released before return or throw; no retry is allowed.
    public func finish() throws -> Report {
        guard !finished else { throw GPUError.invalid("Routing capture already finished") }
        finished = true
        defer { pending.removeAll(keepingCapacity: false) }
        let start = DispatchTime.now().uptimeNanoseconds
        var records = [Record]()
        records.reserveCapacity(pending.count)
        for entry in pending {
            let ids = try Self.validatedExpertIDs(entry.indices.ints(), tokenCount: entry.tokenCount)
            records.append(Record(repetition: entry.repetition, phase: entry.phase,
                                  position: entry.position, tokenCount: entry.tokenCount,
                                  layer: entry.layer, expertIDs: ids))
        }
        let elapsed = Double(DispatchTime.now().uptimeNanoseconds - start) * 1e-6
        return Report(enabled: true, finished: true, maximumRecords: maximumRecords,
                      droppedRecords: droppedRecords, readbackMilliseconds: elapsed,
                      expertCount: 512, topK: 10, hiddenSize: 2560, intermediateSize: 640,
                      bits: 4, groupSize: 64, records: records,
                      notes: [
                        "Readback runs in finish(), which the caller must place after all request/phase timing. Tensor.ints() can create cast/contiguous graphs and evaluate them; readbackMilliseconds includes that deferred work and validation, not CPU-only or device-only time.",
                        "Append retains original routing tensors without evaluation or readback. Retention can change graph/buffer lifetimes; captured generation timings are diagnostic and do not establish undisturbed performance or physical bandwidth."
                      ])
    }

    // Internal pure validation helpers allow CPU tests without constructing MLX
    // tensors. The append/finish paths use exactly these metadata and ID checks.
    static func validateMetadata(repetition: Int, phase: QwenExecutionPhase, position: Int,
                                 tokenCount: Int, layer: Int, shape: [Int], dtype: mlx_dtype) throws {
        guard repetition >= 0, position >= 0, phase == .verification,
              tokenCount == 2 || tokenCount == 3, (0..<48).contains(layer),
              shape == [1, tokenCount, 10], dtype == MLX_UINT32 || dtype == MLX_INT32 else {
            throw GPUError.invalid("Routing capture requires verification S2/S3 Int32/UInt32 [1,S,10] and a valid layer/position")
        }
    }

    static func validatedExpertIDs(_ ids: [Int32], tokenCount: Int) throws -> [[Int32]] {
        guard (tokenCount == 2 || tokenCount == 3), ids.count == tokenCount * 10 else {
            throw GPUError.invalid("Routing capture expert count does not match S2/S3 topK10")
        }
        var rows = [[Int32]]()
        rows.reserveCapacity(tokenCount)
        for row in 0..<tokenCount {
            let values = Array(ids[(row * 10)..<((row + 1) * 10)])
            guard values.allSatisfy({ $0 >= 0 && $0 < 512 }), Set(values).count == 10 else {
                throw GPUError.invalid("Routing capture needs ten unique expert IDs in 0..<512 per row")
            }
            rows.append(values)
        }
        return rows
    }
}
