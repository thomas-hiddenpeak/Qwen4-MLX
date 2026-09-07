import CMLX
import Foundation
import XCTest
@testable import ANERunnerGPU

/// CPU-only lifecycle/schema and shape/ID validation. No tensor is constructed;
/// retained-reference limits and real readback need the explicit GPU diagnostic.
final class GPUVerificationRoutingCaptureTests: XCTestCase {
    func testEmptyCaptureFinishesOnceAndRoundTripsReport() throws {
        XCTAssertThrowsError(try GPUVerificationRoutingCapture(maximumRecords: 0))
        XCTAssertThrowsError(try GPUVerificationRoutingCapture(maximumRecords: -1))
        let capture = try GPUVerificationRoutingCapture()
        XCTAssertFalse(capture.finished)
        let report = try capture.finish()
        XCTAssertTrue(capture.finished)
        XCTAssertThrowsError(try capture.finish())
        let data = try JSONEncoder().encode(report)
        let decoded = try JSONDecoder().decode(GPUVerificationRoutingCapture.Report.self, from: data)
        XCTAssertTrue(decoded.enabled)
        XCTAssertTrue(decoded.finished)
        XCTAssertEqual(decoded.maximumRecords, 8192)
        XCTAssertEqual(decoded.droppedRecords, 0)
        XCTAssertTrue(decoded.records.isEmpty)
        XCTAssertTrue(decoded.readbackMilliseconds.isFinite)
        XCTAssertGreaterThanOrEqual(decoded.readbackMilliseconds, 0)
        XCTAssertEqual(decoded.expertCount, 512)
        XCTAssertEqual(decoded.topK, 10)
        XCTAssertEqual(decoded.hiddenSize, 2560)
        XCTAssertEqual(decoded.intermediateSize, 640)
        XCTAssertEqual(decoded.bits, 4)
        XCTAssertEqual(decoded.groupSize, 64)
        XCTAssertEqual(decoded.notes.count, 2)
        let fields = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(Set(fields.keys), Set(["enabled", "finished", "maximumRecords", "droppedRecords",
            "readbackMilliseconds", "expertCount", "topK", "hiddenSize", "intermediateSize", "bits",
            "groupSize", "records", "notes"]))
    }

    func testMetadataAcceptsOnlyVerificationS2S3IntegerRouting() throws {
        func validate(repetition: Int = 0, phase: QwenExecutionPhase = .verification,
                      position: Int = 11216, tokenCount: Int = 2, layer: Int = 0,
                      shape: [Int] = [1, 2, 10], dtype: mlx_dtype = MLX_UINT32) throws {
            try GPUVerificationRoutingCapture.validateMetadata(repetition: repetition, phase: phase,
                position: position, tokenCount: tokenCount, layer: layer, shape: shape, dtype: dtype)
        }
        try validate()
        try validate(tokenCount: 3, layer: 47, shape: [1, 3, 10], dtype: MLX_INT32)
        XCTAssertThrowsError(try validate(phase: .prefill))
        XCTAssertThrowsError(try validate(phase: .decode))
        XCTAssertThrowsError(try validate(tokenCount: 1, shape: [1, 1, 10]))
        XCTAssertThrowsError(try validate(tokenCount: 4, shape: [1, 4, 10]))
        XCTAssertThrowsError(try validate(shape: [2, 2, 10]))
        XCTAssertThrowsError(try validate(shape: [1, 3, 10]))
        XCTAssertThrowsError(try validate(shape: [1, 2, 9]))
        XCTAssertThrowsError(try validate(dtype: MLX_FLOAT32))
        XCTAssertThrowsError(try validate(layer: -1))
        XCTAssertThrowsError(try validate(layer: 48))
        XCTAssertThrowsError(try validate(position: -1))
        XCTAssertThrowsError(try validate(repetition: -1))
    }

    func testExpertRowsPreserveSlotOrderAndAllowReuseAcrossRows() throws {
        let first: [Int32] = [511, 0, 128, 7, 3, 9, 42, 50, 2, 1]
        let second = Array(first.reversed())
        let rows = try GPUVerificationRoutingCapture.validatedExpertIDs(first + second + first, tokenCount: 3)
        XCTAssertEqual(rows, [first, second, first])
        let record = GPUVerificationRoutingCapture.Record(repetition: 2, phase: .verification,
            position: 11216, tokenCount: 3, layer: 47, expertIDs: rows)
        let data = try JSONEncoder().encode(record)
        let decoded = try JSONDecoder().decode(GPUVerificationRoutingCapture.Record.self, from: data)
        XCTAssertEqual(decoded.expertIDs, rows)
        XCTAssertEqual(decoded.repetition, 2)
        XCTAssertEqual(decoded.phase, .verification)
        XCTAssertEqual(decoded.position, 11216)
        XCTAssertEqual(decoded.tokenCount, 3)
        XCTAssertEqual(decoded.layer, 47)
        let fields = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(Set(fields.keys), Set(["repetition", "phase", "position", "tokenCount", "layer", "expertIDs"]))
    }

    func testExpertValidationRejectsBadLengthRangeAndWithinRowDuplicates() throws {
        let valid = (0..<20).map(Int32.init)
        XCTAssertEqual(try GPUVerificationRoutingCapture.validatedExpertIDs(valid, tokenCount: 2).count, 2)
        for invalidValue in [Int32(-1), Int32(512)] {
            var bad = valid
            bad[19] = invalidValue
            XCTAssertThrowsError(try GPUVerificationRoutingCapture.validatedExpertIDs(bad, tokenCount: 2))
        }
        var duplicate = valid
        duplicate[19] = duplicate[10]
        XCTAssertThrowsError(try GPUVerificationRoutingCapture.validatedExpertIDs(duplicate, tokenCount: 2))
        XCTAssertThrowsError(try GPUVerificationRoutingCapture.validatedExpertIDs(Array(valid.dropLast()), tokenCount: 2))
        XCTAssertThrowsError(try GPUVerificationRoutingCapture.validatedExpertIDs(valid, tokenCount: 1))
        XCTAssertThrowsError(try GPUVerificationRoutingCapture.validatedExpertIDs(valid, tokenCount: 4))
    }
}
