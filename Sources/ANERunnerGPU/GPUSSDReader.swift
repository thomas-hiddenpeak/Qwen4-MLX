import ANERunnerCore
import Dispatch
import Synchronization

public enum GPUSSDReaderError: Error, Equatable, Sendable {
    case invalidWorkerCount(Int)
    case missingWorkerResult(Int)
    case unexpectedResultSize(worker: Int, expected: Int, actual: Int)
}

/// CPU demand reads only. The worker bound applies to each call; callers own
/// admission of independent requests. No MLX handles cross worker boundaries.
public enum GPUSSDReader {
    public static let maximumWorkers = 4

    /// Reads consecutive slices of the *request sequence*, not consecutive
    /// physical rows. Order, duplicates, original NGramTable errors, and the
    /// table's FP8 -> scaled BF16-as-Float values are preserved.
    ///
    /// Each worker owns its decoded chunk. Joining and ordered concatenation
    /// require at most roughly two copies of the requested output, never the
    /// full table. There is no persistent row cache or speculative IO.
    public static func readRows(table: NGramTable, rows: [Int], workers: Int = 1) throws -> [Float] {
        guard (1...maximumWorkers).contains(workers) else {
            throw GPUSSDReaderError.invalidWorkerCount(workers)
        }
        let degree = min(workers, rows.count)
        // Preserve the existing direct path, including its validation/error
        // semantics, for serial calls and requests containing fewer than two rows.
        guard degree > 1 else { return try table.readRows(rows) }

        // NGramTable validates the complete request before touching the file.
        // Do the same before splitting, so an invalid ID anywhere wins over a
        // nonfinite value or IO failure in an otherwise earlier chunk.
        let (outputCount, overflow) = rows.count.multipliedReportingOverflow(by: table.dimension)
        guard !overflow, outputCount <= Int.max / MemoryLayout<Float>.stride else {
            throw NGramTableError.sizeOverflow
        }
        for row in rows where row < 0 || row >= table.rowCount {
            throw NGramTableError.rowOutOfBounds(row)
        }
        let baseCount = rows.count / degree, remainder = rows.count % degree
        let ranges: [Range<Int>] = (0..<degree).map { worker in
            let start = worker * baseCount + min(worker, remainder)
            return start..<(start + baseCount + (worker < remainder ? 1 : 0))
        }
        let storage = ChunkStorage(count: degree)
        DispatchQueue.concurrentPerform(iterations: degree) { worker in
            let requestedRows = Array(rows[ranges[worker]])
            let outcome = Result { try table.readRows(requestedRows) }
            storage.values.withLock { $0[worker] = outcome }
        }
        // concurrentPerform joins every worker, including on failed reads.
        // No task can keep using the table or buffers after this method throws.
        // Choose errors by input chunk order, not nondeterministic finish order.
        let outcomes = storage.values.withLock { $0 }
        var orderedChunks: [[Float]] = []
        orderedChunks.reserveCapacity(degree)
        for (worker, outcome) in outcomes.enumerated() {
            guard let outcome else { throw GPUSSDReaderError.missingWorkerResult(worker) }
            let values = try outcome.get()
            let expected = ranges[worker].count * table.dimension
            guard values.count == expected else {
                throw GPUSSDReaderError.unexpectedResultSize(worker: worker, expected: expected, actual: values.count)
            }
            orderedChunks.append(values)
        }
        // Do not allocate the merged output at all when any worker failed.
        var result: [Float] = []
        result.reserveCapacity(outputCount)
        for values in orderedChunks { result.append(contentsOf: values) }
        return result
    }

    private final class ChunkStorage: Sendable {
        let values: Mutex<[Result<[Float], any Error>?]>
        init(count: Int) { values = Mutex(Array(repeating: nil, count: count)) }
    }
}
