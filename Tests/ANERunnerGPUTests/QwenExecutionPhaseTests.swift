import XCTest
@testable import ANERunnerGPU

/// Pure CPU contracts; no MLX arrays or model weights are created.
final class QwenExecutionPhaseTests: XCTestCase {
    func testGateUpSelectionPreservesLegacyConfigAndPhaseBoundary() throws {
        let legacy = Data(#"{"version":1,"threadgroups":{"416":256}}"#.utf8)
        let restored = try JSONDecoder().decode(GPUMoEPrefillConfiguration.self, from: legacy)
        XCTAssertNil(restored.gateUpVariant)
        XCTAssertNil(restored.groupedDown)
        XCTAssertNoThrow(try restored.validated())
        for variant in [0, 1, 2, 3] {
            let candidate = GPUMoEPrefillConfiguration(threadgroups: [:], gateUpVariant: variant)
            XCTAssertEqual(try JSONDecoder().decode(GPUMoEPrefillConfiguration.self,
                from: JSONEncoder().encode(candidate)), candidate)
            XCTAssertNoThrow(try candidate.validated(phase: .prefill))
            XCTAssertThrowsError(try candidate.validated(phase: .decode))
            XCTAssertThrowsError(try candidate.validated(phase: .verification))
        }
        for variant in [-1, 4] {
            XCTAssertThrowsError(try GPUMoEPrefillConfiguration(threadgroups: [:],
                gateUpVariant: variant).validated())
        }
    }
    func testExpertAlignedPolicyFallsBackForShortTails() throws {
        for variant in [2, 3] {
            let selected = GPUMoEPrefillConfiguration(threadgroups: [:], gateUpVariant: variant, groupedDown: true)
            XCTAssertNoThrow(try selected.validated())
            XCTAssertEqual(try JSONDecoder().decode(GPUMoEPrefillConfiguration.self,
                from: JSONEncoder().encode(selected)), selected)
            for length in [1, 2, 64, 204, 513] {
                XCTAssertNil(selected.effectiveGateUpVariant(tokenCount: length))
                XCTAssertFalse(selected.usesGroupedDown(tokenCount: length))
            }
            for length in [205, 240, 416, 512] {
                XCTAssertEqual(selected.effectiveGateUpVariant(tokenCount: length), variant)
                XCTAssertTrue(selected.usesGroupedDown(tokenCount: length))
            }
            XCTAssertThrowsError(try selected.validated(phase: .decode))
            XCTAssertThrowsError(try selected.validated(phase: .verification))
        }
        for variant in [nil, 0, 1] as [Int?] {
            XCTAssertThrowsError(try GPUMoEPrefillConfiguration(threadgroups: [:],
                gateUpVariant: variant, groupedDown: true).validated())
        }
    }
    func testTunedMoEReductionIsRequestLocalAndPrefillOnly() throws {
        let selected = GPUMoEPrefillConfiguration(threadgroups: [416:256,240:128])
        let restored = try JSONDecoder().decode(GPUMoEPrefillConfiguration.self, from: JSONEncoder().encode(selected))
        XCTAssertEqual(restored, selected)
        XCTAssertNoThrow(try restored.validated())
        XCTAssertThrowsError(try restored.validated(phase: .decode))
        XCTAssertThrowsError(try restored.validated(phase: .verification))
        XCTAssertNil(restored.threadgroupSize(tokenCount: 1))
        XCTAssertNil(restored.threadgroupSize(tokenCount: 205))
        XCTAssertEqual(restored.threadgroupSize(tokenCount: 416), 256)
        XCTAssertNil(QwenGenerationRequest(tokens: [42]).prefillMoEConfiguration)
        XCTAssertEqual(QwenGenerationRequest(tokens: [42], prefillMoEConfiguration: restored).prefillMoEConfiguration, selected)
        for invalid in [[1:256],[416:64],[513:256]] {
            XCTAssertThrowsError(try GPUMoEPrefillConfiguration(threadgroups: invalid).validated())
        }
    }
    func testPrefillAttentionPolicyIsConfinedToPrefill() throws {
        for phase in QwenExecutionPhase.allCases {
            XCTAssertNoThrow(try GPUAttention.PrefillMode.reference.validate(phase: phase))
        }
        XCTAssertNoThrow(try GPUAttention.PrefillMode.fusedQSA.validate(phase: .prefill))
        XCTAssertThrowsError(try GPUAttention.PrefillMode.fusedQSA.validate(phase: .decode))
        XCTAssertThrowsError(try GPUAttention.PrefillMode.fusedQSA.validate(phase: .verification))
        XCTAssertEqual(QwenGenerationRequest(tokens: [42]).prefillAttention, .reference)
    }
    func testExplicitPhaseIsNotInferredFromShape() throws {
        XCTAssertEqual(try QwenExecutionPhase.resolve(.prefill, tokenCount: 1), .prefill)
        XCTAssertEqual(try QwenExecutionPhase.resolve(.decode, tokenCount: 1), .decode)
        XCTAssertEqual(try QwenExecutionPhase.resolve(.verification, tokenCount: 3), .verification)
        XCTAssertEqual(try QwenExecutionPhase.resolve(nil, tokenCount: 1), .decode)
        XCTAssertEqual(try QwenExecutionPhase.resolve(nil, tokenCount: 3), .prefill)
        XCTAssertThrowsError(try QwenExecutionPhase.resolve(.decode, tokenCount: 2))
        XCTAssertThrowsError(try QwenExecutionPhase.resolve(.verification, tokenCount: 6))
        XCTAssertThrowsError(try QwenExecutionPhase.resolve(.prefill, tokenCount: 0))
    }

    func testDefaultScheduleAndIndependentVerificationInterval() throws {
        func points(_ phase: QwenExecutionPhase, tokens: Int, interval: Int) -> [Int] {
            (1...48).filter { phase.shouldEvaluate(completedLayers: $0, tokenCount: tokens, every: interval) }
        }
        let original = Array(stride(from: 4, through: 48, by: 4))
        XCTAssertEqual(points(.prefill, tokens: 512, interval: 4), original)
        XCTAssertEqual(points(.verification, tokens: 3, interval: 4), original)
        XCTAssertEqual(points(.verification, tokens: 3, interval: 12), [12, 24, 36, 48])
        XCTAssertEqual(points(.prefill, tokens: 512, interval: 4), original)
        for phase in QwenExecutionPhase.allCases {
            XCTAssertEqual(points(phase, tokens: 1, interval: 4), [])
        }
        XCTAssertEqual(points(.decode, tokens: 1, interval: 1), [])
        // Older forward callers could choose any positive interval, including
        // one beyond the layer count to defer all materialization to the caller.
        XCTAssertEqual(points(.prefill, tokens: 3, interval: 64), [])
    }

    func testVerificationIntervalBoundsWithoutLoadingDecoder() throws {
        for interval in [1, 4, 12, 48] {
            XCTAssertNoThrow(try QwenExecutionPhase.validateVerificationInterval(interval))
        }
        for interval in [Int.min, -1, 0, 49, Int.max] {
            XCTAssertThrowsError(try QwenExecutionPhase.validateVerificationInterval(interval))
        }
    }
}
