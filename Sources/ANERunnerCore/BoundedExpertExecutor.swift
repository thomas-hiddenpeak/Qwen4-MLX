import Dispatch
import Foundation
import Synchronization

struct ExpertPredictionChunk: Sendable {
    /// Flattened [token, topK] slots to which the returned rows belong.
    let slots: [Int]
    let tokenIndices: [Int]
}

struct ExpertPredictionJob: Sendable {
    let expertID: Int
    let kernel: ConcurrentExpertKernel
    /// All chunks for this expert are predicted sequentially by one worker.
    let chunks: [ExpertPredictionChunk]
}

struct ExpertPredictionChunkResult: Sendable {
    let slots: [Int]
    /// Row-major [slots.count, hiddenSize]; padded rows have been removed.
    let values: [Float]
    let predictionMilliseconds: Double
}

struct ExpertPredictionResult: Sendable {
    let expertID: Int
    let chunks: [ExpertPredictionChunkResult]
}

struct ExpertPredictionBatchResult: Sendable {
    /// Results always preserve the input job order, regardless of completion.
    let results: [ExpertPredictionResult]
    /// Dispatch + prediction + result collection for this batch, excluding load.
    let wallMilliseconds: Double
    let concurrency: Int
    let callCount: Int
    /// Sum of individual call durations; can exceed wallMilliseconds.
    let predictionMilliseconds: Double
}

/// Executes independent expert jobs without accessing the runner or its cache.
///
/// The caller must preload at most min(requested degree, resident capacity)
/// distinct kernels per wave, then release job references before changing the
/// cache for the next wave. A resident capacity of one therefore runs serially.
/// Shared-expert execution and the final reduction remain the caller's concern.
enum BoundedExpertExecutor {
    static func predict(jobs: [ExpertPredictionJob], tokens: [Float], maximumConcurrency: Int) throws -> ExpertPredictionBatchResult {
        guard maximumConcurrency > 0 else {
            throw ExpertRouterError.invalid("Expert prediction concurrency must be positive")
        }
        guard Set(jobs.map(\.expertID)).count == jobs.count,
            Set(jobs.map { ObjectIdentifier($0.kernel) }).count == jobs.count,
            jobs.allSatisfy({ !$0.chunks.isEmpty && $0.chunks.allSatisfy {
                !$0.slots.isEmpty && $0.slots.count == $0.tokenIndices.count && $0.slots.allSatisfy { $0 >= 0 }
            } }) else {
            throw ExpertRouterError.invalid("A prediction batch requires distinct experts/kernels and valid chunks")
        }
        let degree = min(maximumConcurrency, jobs.count)
        guard degree > 0 else {
            return ExpertPredictionBatchResult(results: [], wallMilliseconds: 0, concurrency: 0,
                                               callCount: 0, predictionMilliseconds: 0)
        }
        let storage = ResultStorage(count: jobs.count)
        let start = DispatchTime.now().uptimeNanoseconds
        let work: @Sendable (Int) -> Void = { worker in
            for index in stride(from: worker, to: jobs.count, by: degree) {
                let outcome: Outcome = autoreleasepool {
                    do {
                        let job = jobs[index]
                        var chunks = [ExpertPredictionChunkResult]()
                        chunks.reserveCapacity(job.chunks.count)
                        for chunk in job.chunks {
                            let (values, milliseconds) = try job.kernel.predict(tokens: tokens, tokenIndices: chunk.tokenIndices)
                            chunks.append(ExpertPredictionChunkResult(slots: chunk.slots, values: values,
                                                                       predictionMilliseconds: milliseconds))
                        }
                        return .success(ExpertPredictionResult(expertID: job.expertID, chunks: chunks))
                    } catch {
                        return .failure("Expert \(jobs[index].expertID) prediction failed: \(error.localizedDescription)")
                    }
                }
                storage.values.withLock { $0[index] = outcome }
            }
        }
        if degree == 1 { work(0) }
        else { DispatchQueue.concurrentPerform(iterations: degree, execute: work) }
        // concurrentPerform is a barrier: even failures are surfaced only after
        // every worker stops using its model/buffer, so cache cleanup is safe.
        let outcomes = storage.values.withLock { $0 }
        var results = [ExpertPredictionResult]()
        results.reserveCapacity(jobs.count)
        for outcome in outcomes {
            switch outcome {
            case .success(let result): results.append(result)
            case .failure(let message): throw ExpertRouterError.invalid(message)
            case nil: throw ExpertRouterError.invalid("Expert worker produced no result")
            }
        }
        let calls = results.flatMap(\.chunks)
        return ExpertPredictionBatchResult(
            results: results, wallMilliseconds: Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000,
            concurrency: degree, callCount: calls.count,
            predictionMilliseconds: calls.reduce(0) { $0 + $1.predictionMilliseconds })
    }

    private enum Outcome: Sendable {
        case success(ExpertPredictionResult)
        case failure(String)
    }

    private final class ResultStorage: Sendable {
        let values: Mutex<[Outcome?]>
        init(count: Int) { values = Mutex(Array(repeating: nil, count: count)) }
    }
}
