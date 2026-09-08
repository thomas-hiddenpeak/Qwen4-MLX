import ANERunnerCore
import Foundation
import XCTest
@testable import ANERunnerGPU

/// CPU-only permit ownership and accounting. No model, tensor or GPU is made.
final class QwenKVCapacityAppendPermitTests: XCTestCase {
    func testReferenceDefaultAndCapacityMTPExclusionAreCPUValidated() throws {
        let configuration = try QwenConfiguration(modelDirectory: GPUFixtureLocation.model())
        XCTAssertEqual(QwenGenerationRequest(tokens: [42]).kvAppendMode, .reference)
        try QwenGenerationRequest(tokens: [42], kvAppendMode: .capacity256).validate(configuration: configuration)
        for depth in 1...4 {
            XCTAssertThrowsError(try QwenGenerationRequest(tokens: [42], mtpDepth: depth,
                kvAppendMode: .capacity256).validate(configuration: configuration)) {
                guard case QwenGenerationError.invalidRequest(let message) = $0 else {
                    return XCTFail("Unexpected error: \($0)")
                }
                XCTAssertTrue(message.contains("MTP"))
            }
        }
    }

    func testZeroWorkspaceStillRequiresSingleUseOwnerAndPosition() throws {
        let owner = UUID(), session = UUID()
        let permit = QwenKVCapacityAppendPermit(modelOwner: owner, sessionIdentity: session,
            offset: 256, rowLimit: 272, workspaceBytes: 0, lease: nil)
        XCTAssertThrowsError(try permit.consume(modelOwner: UUID(), sessionIdentity: session,
            offset: 256, maximumRowLimit: 16_384))
        XCTAssertThrowsError(try permit.consume(modelOwner: owner, sessionIdentity: UUID(),
            offset: 256, maximumRowLimit: 16_384))
        XCTAssertThrowsError(try permit.consume(modelOwner: owner, sessionIdentity: session,
            offset: 257, maximumRowLimit: 16_384))
        XCTAssertEqual(try permit.consume(modelOwner: owner, sessionIdentity: session,
            offset: 256, maximumRowLimit: 16_384), 272)
        XCTAssertThrowsError(try permit.consume(modelOwner: owner, sessionIdentity: session,
            offset: 256, maximumRowLimit: 16_384))
        permit.releaseAfterCompletion()
    }

    func testConsumedPermitKeepsWorkspaceUntilExplicitCompletion() throws {
        let budget = try QwenStateBudget(maxBytes: 100)
        let owner = UUID(), session = UUID()
        let lease = try XCTUnwrap(budget.reserve(bytes: 80, kind: .workspace))
        let permit = QwenKVCapacityAppendPermit(modelOwner: owner, sessionIdentity: session,
            offset: 255, rowLimit: 271, workspaceBytes: 80, lease: lease)
        XCTAssertEqual(try permit.consume(modelOwner: owner, sessionIdentity: session,
            offset: 255, maximumRowLimit: 16_384), 271)
        XCTAssertEqual(budget.statistics.workspaceBytes, 80)
        XCTAssertNil(budget.reserve(bytes: 21, kind: .workspace))
        permit.releaseAfterCompletion()
        permit.releaseAfterCompletion()
        XCTAssertEqual(budget.statistics.totalBytes, 0)
        XCTAssertEqual(budget.statistics.currentLeases, 0)
        XCTAssertTrue(lease.isReleased)
    }

    func testReleaseAndInvalidBoundsPreventConsume() throws {
        let owner = UUID(), session = UUID()
        for (offset, limit, maximum) in [(-1, 1, 100), (10, 10, 100), (10, 20, 19)] {
            let permit = QwenKVCapacityAppendPermit(modelOwner: owner, sessionIdentity: session,
                offset: offset, rowLimit: limit, workspaceBytes: 0, lease: nil)
            XCTAssertThrowsError(try permit.consume(modelOwner: owner, sessionIdentity: session,
                offset: offset, maximumRowLimit: maximum))
        }
        let released = QwenKVCapacityAppendPermit(modelOwner: owner, sessionIdentity: session,
            offset: 1, rowLimit: 2, workspaceBytes: 0, lease: nil)
        released.releaseAfterCompletion()
        XCTAssertThrowsError(try released.consume(modelOwner: owner, sessionIdentity: session,
            offset: 1, maximumRowLimit: 2))
    }

    func testMissingWrongOrReleasedWorkspaceCannotAuthorizeGrowth() throws {
        let budget = try QwenStateBudget(maxBytes: 100)
        let owner = UUID(), session = UUID()
        let wrongKind = try XCTUnwrap(budget.reserve(bytes: 10, kind: .request))
        let wrongBytes = try XCTUnwrap(budget.reserve(bytes: 20, kind: .workspace))
        let released = try XCTUnwrap(budget.reserve(bytes: 10, kind: .workspace))
        released.release()
        let cases: [(Int, QwenStateBudget.Lease?)] = [(-1, nil), (10, nil),
            (10, wrongKind), (10, wrongBytes), (10, released), (0, wrongBytes)]
        for (bytes, lease) in cases {
            let permit = QwenKVCapacityAppendPermit(modelOwner: owner, sessionIdentity: session,
                offset: 1, rowLimit: 2, workspaceBytes: bytes, lease: lease)
            XCTAssertThrowsError(try permit.consume(modelOwner: owner, sessionIdentity: session,
                offset: 1, maximumRowLimit: 2))
        }
        XCTAssertEqual(budget.statistics.totalBytes, 0)
    }

    func testAbandonedPermitReleasesOnlyItsLease() throws {
        let budget = try QwenStateBudget(maxBytes: 100)
        let request = try XCTUnwrap(budget.reserve(bytes: 30, kind: .request))
        var permit: QwenKVCapacityAppendPermit? = QwenKVCapacityAppendPermit(
            modelOwner: UUID(), sessionIdentity: UUID(), offset: 1, rowLimit: 2,
            workspaceBytes: 60, lease: try XCTUnwrap(budget.reserve(bytes: 60, kind: .workspace)))
        XCTAssertEqual(permit?.workspaceBytes, 60)
        XCTAssertEqual(budget.statistics.totalBytes, 90)
        permit = nil
        XCTAssertEqual(budget.statistics.workspaceBytes, 0)
        XCTAssertEqual(budget.statistics.requestBytes, 30)
        request.release()
    }
}
