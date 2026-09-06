import CoreML
import Foundation
import XCTest
@testable import ANERunnerCore

final class BoundedExpertExecutorTests: XCTestCase {
    private var ownedCompiledURLs = [URL]()

    override func tearDownWithError() throws {
        for url in ownedCompiledURLs { try FileManager.default.removeItem(at: url) }
        ownedCompiledURLs.removeAll()
        try super.tearDownWithError()
    }

    func testNonpositiveConcurrencyRejectedEvenForEmptyBatch() {
        for degree in [0, -1, Int.min] {
            XCTAssertThrowsError(try BoundedExpertExecutor.predict(jobs: [], tokens: [], maximumConcurrency: degree)) { error in
                XCTAssertTrue(error.localizedDescription.contains("concurrency must be positive"))
            }
        }
    }

    func testEmptyBatchHasNoCallsWorkersOrElapsedPrediction() throws {
        for degree in [1, 4, Int.max] {
            let result = try BoundedExpertExecutor.predict(jobs: [], tokens: [], maximumConcurrency: degree)
            XCTAssertTrue(result.results.isEmpty)
            XCTAssertEqual(result.concurrency, 0)
            XCTAssertEqual(result.callCount, 0)
            XCTAssertEqual(result.wallMilliseconds, 0)
            XCTAssertEqual(result.predictionMilliseconds, 0)
        }
    }

    func testRunnerConcurrencyBoundsRejectedBeforeReadingManifest() {
        let unusedURL = URL(fileURLWithPath: "/unused-concurrency-test-manifest.json")
        for degree in [0, -1, 33] {
            XCTAssertThrowsError(try MoEBlockRunner(manifestURL: unusedURL, expertConcurrency: degree)) { error in
                XCTAssertTrue(error.localizedDescription.contains("Expert concurrency must be in 1...32"))
            }
        }
    }

    func testDuplicateExpertIDsRejectedWithDistinctKernels() throws {
        let kernels = try tinyKernels(count: 2)
        let jobs = [job(id: 7, kernel: kernels[0], slot: 0), job(id: 7, kernel: kernels[1], slot: 1)]
        assertInvalidBatch(jobs)
    }

    func testDuplicateKernelRejectedAcrossDistinctExpertIDs() throws {
        let kernel = try tinyKernels(count: 1)[0]
        assertInvalidBatch([job(id: 7, kernel: kernel, slot: 0), job(id: 8, kernel: kernel, slot: 1)])
    }

    func testInvalidChunksRejectedBeforePrediction() throws {
        let kernel = try tinyKernels(count: 1)[0]
        let invalid: [[ExpertPredictionChunk]] = [
            [],
            [ExpertPredictionChunk(slots: [], tokenIndices: [])],
            [ExpertPredictionChunk(slots: [0], tokenIndices: [])],
            [ExpertPredictionChunk(slots: [-1], tokenIndices: [0])],
        ]
        for chunks in invalid {
            assertInvalidBatch([ExpertPredictionJob(expertID: 7, kernel: kernel, chunks: chunks)])
        }
    }

    func testWorkerFailureHasExpertContextAndKernelsRemainReusable() throws {
        let kernels = try tinyKernels(count: 2)
        let tokens: [Float] = [1, -0.5, 0.25, 2]
        let bad = ExpertPredictionJob(expertID: 7, kernel: kernels[0], chunks: [
            ExpertPredictionChunk(slots: [0], tokenIndices: [1]), // Only token zero exists.
        ])
        let good = job(id: 8, kernel: kernels[1], slot: 1)
        XCTAssertThrowsError(try BoundedExpertExecutor.predict(jobs: [bad, good], tokens: tokens, maximumConcurrency: 2)) { error in
            XCTAssertTrue(error.localizedDescription.contains("Expert 7 prediction failed"))
            XCTAssertTrue(error.localizedDescription.contains("Invalid expert token chunk"))
        }
        let recovered = try BoundedExpertExecutor.predict(jobs: [
            job(id: 7, kernel: kernels[0], slot: 0), good,
        ], tokens: tokens, maximumConcurrency: 2)
        XCTAssertEqual(recovered.callCount, 2)
        XCTAssertEqual(recovered.results.map(\.expertID), [7, 8])
        XCTAssertEqual(recovered.results[0].chunks[0].values, recovered.results[1].chunks[0].values)
        XCTAssertTrue(recovered.results[0].chunks[0].values.allSatisfy(\.isFinite))
        // This checks error attribution and reuse. It does not independently
        // observe worker completion: joining is guaranteed structurally by the
        // synchronous concurrentPerform call before outcomes are inspected.
    }

    func testConcurrentResultsPreserveJobAndChunkOrderExactly() throws {
        let kernels = try tinyKernels(count: 2)
        let tokens: [Float] = [1, -0.5, 0.25, 2, -1, 0.75, 2, 0.5, 0.125, 1, -2, 0.25]
        // Deliberately non-sorted expert IDs and slot order; the executor must
        // return the submitted order and leave token/slot mapping untouched.
        let jobs = [
            ExpertPredictionJob(expertID: 9, kernel: kernels[0], chunks: [
                ExpertPredictionChunk(slots: [7], tokenIndices: [2]),
                ExpertPredictionChunk(slots: [3, 1], tokenIndices: [0, 1]),
            ]),
            ExpertPredictionJob(expertID: 2, kernel: kernels[1], chunks: [
                ExpertPredictionChunk(slots: [5], tokenIndices: [1]),
            ]),
        ]
        let serial = try BoundedExpertExecutor.predict(jobs: jobs, tokens: tokens, maximumConcurrency: 1)
        let parallel = try BoundedExpertExecutor.predict(jobs: jobs, tokens: tokens, maximumConcurrency: 4)
        XCTAssertEqual(serial.callCount, 3)
        XCTAssertEqual(parallel.callCount, 3)
        XCTAssertEqual(parallel.concurrency, 2)
        XCTAssertEqual(parallel.results.map(\.expertID), [9, 2])
        XCTAssertEqual(parallel.results.flatMap(\.chunks).map(\.slots), [[7], [3, 1], [5]])
        XCTAssertEqual(parallel.results.flatMap(\.chunks).map(\.values), serial.results.flatMap(\.chunks).map(\.values))
        XCTAssertEqual(parallel.results.flatMap(\.chunks).map { $0.values.count }, [4, 8, 4])
    }

    private func job(id: Int, kernel: ConcurrentExpertKernel, slot: Int) -> ExpertPredictionJob {
        ExpertPredictionJob(expertID: id, kernel: kernel, chunks: [ExpertPredictionChunk(slots: [slot], tokenIndices: [0])])
    }

    private func assertInvalidBatch(_ jobs: [ExpertPredictionJob], file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertThrowsError(try BoundedExpertExecutor.predict(jobs: jobs, tokens: [1, 2, 3, 4], maximumConcurrency: 2), file: file, line: line) { error in
            XCTAssertTrue(error.localizedDescription.contains("distinct experts/kernels and valid chunks"), file: file, line: line)
        }
    }

    private func tinyKernels(count: Int) throws -> [ConcurrentExpertKernel] {
        // The independent Python scheduler gate owns this synthetic H4/I8 bank.
        // CPU-only loading here does not depend on full model weights or ANE.
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        let bank = root.appendingPathComponent("results/moe-scheduler-test")
        let manifestURL = bank.appendingPathComponent("manifest.json")
        let packageURL = bank.appendingPathComponent("expert_0.mlpackage")
        guard FileManager.default.fileExists(atPath: manifestURL.path),
            FileManager.default.fileExists(atPath: packageURL.path) else {
            throw XCTSkip("Run scripts/verify_moe_scheduler.py to create the independent tiny CPU fixture")
        }
        let manifest = try MoEManifest.load(from: manifestURL)
        guard manifest.hiddenSize == 4, manifest.tokenCapacity == 2 else {
            throw XCTSkip("Expected the synthetic H4/capacity2 scheduler bank")
        }
        let compiledURL = try MLModel.compileModel(at: packageURL)
        ownedCompiledURLs.append(compiledURL)
        return try (0..<count).map { _ in
            try ConcurrentExpertKernel(url: compiledURL, manifest: manifest, units: .cpuOnly)
        }
    }
}
