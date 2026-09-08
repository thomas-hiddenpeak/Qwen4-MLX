import Dispatch
import Foundation
import XCTest
@testable import ANERunnerCore

final class QwenMemoryPressureTests: XCTestCase {
    private final class Clock: @unchecked Sendable {
        private let lock = NSLock()
        private var time: TimeInterval = 100
        func now() -> TimeInterval {
            lock.lock(); defer { lock.unlock() }; return time
        }
        func set(_ value: TimeInterval) {
            lock.lock(); defer { lock.unlock() }; time = value
        }
    }

    private func makePolicy(_ clock: Clock, stable: TimeInterval = 5) throws -> QwenMemoryPressurePolicy {
        try .init(recoveryStableSeconds: stable, clock: { clock.now() })
    }

    func testRecoveryConfigurationRejectsNegativeAndNonFiniteDurations() throws {
        for value in [-1, .infinity, -.infinity, .nan] as [TimeInterval] {
            XCTAssertThrowsError(try QwenMemoryPressurePolicy(recoveryStableSeconds: value)) {
                XCTAssertEqual($0 as? QwenMemoryPressurePolicy.ConfigurationError,
                               .invalidRecoveryStableSeconds)
            }
        }
        XCTAssertEqual(try QwenMemoryPressurePolicy(recoveryStableSeconds: 0)
            .snapshot.recoveryStableSeconds, 0)
    }

    func testInitialUnknownIsExplicitAndDoesNotPretendToObserveOperatingSystem() throws {
        let clock = Clock(), policy = try makePolicy(clock)
        let snapshot = policy.snapshot
        XCTAssertEqual(snapshot.observedLevel, .unknown)
        XCTAssertEqual(snapshot.effectiveLevel, .unknown)
        XCTAssertEqual(snapshot.reason, .notObserved)
        XCTAssertNil(snapshot.lastEventSource)
        XCTAssertNil(snapshot.lastEventAgeSeconds)
        XCTAssertNil(snapshot.recoveryRemainingSeconds)
        XCTAssertEqual(snapshot.events, 0)
        XCTAssertEqual(snapshot.operatingSystemEvents, 0)
        XCTAssertTrue(policy.checkNewRequestAdmission())
        XCTAssertTrue(policy.checkOptionalCacheAdmission())
        XCTAssertNil(policy.takeTrimRequest())
        clock.set(10_000)
        XCTAssertEqual(policy.snapshot.effectiveLevel, .unknown)
    }

    func testWarningStopsOptionalWorkButAllowsNewRequestsAndTrimIsConsumedOnce() throws {
        let clock = Clock(), policy = try makePolicy(clock)
        policy.observe(.warning)
        let snapshot = policy.snapshot
        XCTAssertEqual(snapshot.observedLevel, .warning)
        XCTAssertEqual(snapshot.effectiveLevel, .warning)
        XCTAssertEqual(snapshot.reason, .warning)
        XCTAssertTrue(snapshot.allowsNewRequests)
        XCTAssertFalse(snapshot.allowsOptionalCache)
        XCTAssertEqual(policy.takeTrimRequest(), 1)
        XCTAssertNil(policy.takeTrimRequest())
        XCTAssertTrue(policy.checkNewRequestAdmission())
        XCTAssertFalse(policy.checkOptionalCacheAdmission())
        policy.observe(.warning)
        XCTAssertNil(policy.takeTrimRequest())
        XCTAssertEqual(policy.snapshot.warningEvents, 2)
        XCTAssertEqual(policy.snapshot.transitions, 1)
        XCTAssertEqual(policy.snapshot.optionalCacheDenials, 1)
    }

    func testCriticalDeniesOnlyNewAdmissionsAndEscalationCreatesNewTrimGeneration() throws {
        let clock = Clock(), policy = try makePolicy(clock)
        policy.observe(.warning)
        XCTAssertEqual(policy.takeTrimRequest(), 1)
        policy.observe(.critical)
        XCTAssertEqual(policy.takeTrimRequest(), 2)
        XCTAssertFalse(policy.checkNewRequestAdmission())
        XCTAssertFalse(policy.checkOptionalCacheAdmission())
        // Health polling is observational and must not count as new requests.
        for _ in 0..<10 { XCTAssertFalse(policy.snapshot.allowsNewRequests) }
        XCTAssertEqual(policy.snapshot.newRequestDenials, 1)
        XCTAssertEqual(policy.snapshot.optionalCacheDenials, 1)
        XCTAssertEqual(policy.snapshot.reason, .critical)
    }

    func testNormalRecoveryHoldsCriticalUntilStabilityWindowAndRepeatedNormalDoesNotPostpone() throws {
        let clock = Clock(), policy = try makePolicy(clock)
        policy.observe(.critical)
        clock.set(101); policy.observe(.normal)
        XCTAssertEqual(policy.snapshot.observedLevel, .normal)
        XCTAssertEqual(policy.snapshot.effectiveLevel, .critical)
        XCTAssertEqual(policy.snapshot.reason, .criticalRecoveryHold)
        XCTAssertEqual(policy.snapshot.recoveryTarget, .normal)
        XCTAssertEqual(policy.snapshot.recoveryRemainingSeconds, 5)
        clock.set(104); policy.observe(.normal)
        XCTAssertEqual(policy.snapshot.recoveryRemainingSeconds, 2)
        XCTAssertFalse(policy.checkNewRequestAdmission())
        clock.set(105.999)
        XCTAssertFalse(policy.checkNewRequestAdmission())
        clock.set(106)
        XCTAssertTrue(policy.checkNewRequestAdmission())
        XCTAssertTrue(policy.checkOptionalCacheAdmission())
        XCTAssertEqual(policy.snapshot.effectiveLevel, .normal)
        XCTAssertNil(policy.snapshot.recoveryRemainingSeconds)
        XCTAssertNil(policy.snapshot.recoveryTarget)
        XCTAssertEqual(policy.snapshot.recoveries, 1)
        XCTAssertEqual(policy.snapshot.lastEventAgeSeconds, 2)
        XCTAssertNil(policy.takeTrimRequest())
    }

    func testCriticalCanRecoverToWarningWithoutResumingOptionalCache() throws {
        let clock = Clock(), policy = try makePolicy(clock)
        policy.observe(.critical)
        clock.set(101); policy.observe(.warning)
        XCTAssertFalse(policy.checkNewRequestAdmission())
        clock.set(106)
        XCTAssertTrue(policy.checkNewRequestAdmission())
        XCTAssertFalse(policy.checkOptionalCacheAdmission())
        XCTAssertEqual(policy.snapshot.effectiveLevel, .warning)
        XCTAssertEqual(policy.snapshot.recoveries, 1)
        clock.set(107); policy.observe(.normal)
        XCTAssertEqual(policy.snapshot.reason, .warningRecoveryHold)
        clock.set(112)
        XCTAssertTrue(policy.checkOptionalCacheAdmission())
        XCTAssertEqual(policy.snapshot.recoveries, 2)
    }

    func testPressureFlappingCancelsRecoveryAndNewTargetRequiresFreshStableWindow() throws {
        let clock = Clock(), policy = try makePolicy(clock)
        policy.observe(.critical)
        clock.set(101); policy.observe(.normal)
        clock.set(104); policy.observe(.critical)
        XCTAssertNil(policy.snapshot.recoveryTarget)
        XCTAssertNil(policy.snapshot.recoveryRemainingSeconds)
        clock.set(105); policy.observe(.warning)
        clock.set(108); policy.observe(.normal)
        XCTAssertEqual(policy.snapshot.recoveryRemainingSeconds, 5)
        clock.set(110)
        XCTAssertFalse(policy.checkNewRequestAdmission())
        clock.set(113)
        XCTAssertTrue(policy.checkNewRequestAdmission())
        XCTAssertEqual(policy.snapshot.recoveries, 1)
    }

    func testNoNormalEventMeansElapsedTimeAloneCannotClearPressure() throws {
        let clock = Clock(), policy = try makePolicy(clock)
        policy.observe(.critical)
        clock.set(100_000)
        XCTAssertFalse(policy.checkNewRequestAdmission())
        XCTAssertEqual(policy.snapshot.effectiveLevel, .critical)
        XCTAssertNil(policy.snapshot.recoveryRemainingSeconds)
        XCTAssertEqual(policy.snapshot.lastEventAgeSeconds, 99_900)
    }

    func testZeroWindowRecoversImmediatelyButInitialStateStillUnknown() throws {
        let clock = Clock(), policy = try makePolicy(clock, stable: 0)
        XCTAssertEqual(policy.snapshot.effectiveLevel, .unknown)
        policy.observe(.critical)
        policy.observe(.normal)
        XCTAssertTrue(policy.checkNewRequestAdmission())
        XCTAssertTrue(policy.checkOptionalCacheAdmission())
        XCTAssertEqual(policy.snapshot.recoveries, 1)
    }

    func testCountersSeparateInjectedObservationsAndCodableSnapshot() throws {
        let clock = Clock(), policy = try makePolicy(clock)
        policy.observe(.normal, source: .operatingSystem)
        policy.observe(.warning)
        policy.observe(.critical, source: .operatingSystem)
        policy.observe(.normal)
        let snapshot = policy.snapshot
        XCTAssertEqual(snapshot.events, 4)
        XCTAssertEqual(snapshot.operatingSystemEvents, 2)
        XCTAssertEqual(snapshot.injectedEvents, 2)
        XCTAssertEqual(snapshot.normalEvents, 2)
        XCTAssertEqual(snapshot.warningEvents, 1)
        XCTAssertEqual(snapshot.criticalEvents, 1)
        XCTAssertEqual(snapshot.lastEventSource, .injected)
        XCTAssertEqual(snapshot.transitions, 3)
        let decoded = try JSONDecoder().decode(QwenMemoryPressurePolicy.Snapshot.self,
            from: JSONEncoder().encode(snapshot))
        XCTAssertEqual(decoded, snapshot)
    }

    func testBackwardOrNonFiniteClockDoesNotPrematurelyRecoverOrPoisonSnapshot() throws {
        let clock = Clock(), policy = try makePolicy(clock)
        policy.observe(.critical)
        clock.set(101); policy.observe(.normal)
        clock.set(104)
        XCTAssertEqual(policy.snapshot.recoveryRemainingSeconds, 2)
        for invalid in [20, .nan, .infinity, -.infinity] as [TimeInterval] {
            clock.set(invalid)
            XCTAssertEqual(policy.snapshot.recoveryRemainingSeconds, 2)
            XCTAssertFalse(policy.checkNewRequestAdmission())
            XCTAssertNoThrow(try JSONEncoder().encode(policy.snapshot))
        }
        clock.set(106)
        XCTAssertTrue(policy.checkNewRequestAdmission())
    }

    func testEventFlagDecoderIgnoresEmptyAndPicksMostSevereCoalescedSignal() {
        XCTAssertNil(QwenMemoryPressureMonitor.event(for: []))
        XCTAssertEqual(QwenMemoryPressureMonitor.event(for: [.normal]), .normal)
        XCTAssertEqual(QwenMemoryPressureMonitor.event(for: [.normal, .warning]), .warning)
        XCTAssertEqual(QwenMemoryPressureMonitor.event(for: [.normal, .warning, .critical]), .critical)
    }

    func testActualMonitorStartsStopsIdempotentlyAndDoesNotRetainItself() throws {
        let clock = Clock(), policy = try makePolicy(clock)
        var monitor: QwenMemoryPressureMonitor? = .init(policy: policy)
        weak var weakMonitor = monitor
        XCTAssertTrue(monitor!.isRunning)
        monitor!.stop()
        XCTAssertFalse(monitor!.isRunning)
        monitor!.stop()
        let stopped = policy.snapshot
        monitor = nil
        XCTAssertNil(weakMonitor)
        // The host may deliver a real event during this test; do not infer
        // normal pressure or assert that a monitored system emitted no events.
        XCTAssertEqual(policy.snapshot.events, stopped.events)
    }

    func testConcurrentPolicyOperationsPreserveCountersAndConsistentDecisions() throws {
        let clock = Clock(), policy = try makePolicy(clock)
        DispatchQueue.concurrentPerform(iterations: 2_000) { iteration in
            policy.observe(QwenMemoryPressurePolicy.Event.allCases[iteration % 3])
            _ = policy.checkNewRequestAdmission()
            _ = policy.checkOptionalCacheAdmission()
            _ = policy.takeTrimRequest()
            let snapshot = policy.snapshot
            XCTAssertEqual(snapshot.events,
                           snapshot.normalEvents + snapshot.warningEvents + snapshot.criticalEvents)
            XCTAssertEqual(snapshot.events, snapshot.operatingSystemEvents + snapshot.injectedEvents)
            XCTAssertEqual(snapshot.allowsNewRequests, snapshot.effectiveLevel != .critical)
            if snapshot.allowsOptionalCache { XCTAssertTrue(snapshot.allowsNewRequests) }
        }
        XCTAssertEqual(policy.snapshot.events, 2_000)
        XCTAssertEqual(policy.snapshot.injectedEvents, 2_000)
        XCTAssertEqual(policy.snapshot.operatingSystemEvents, 0)
        XCTAssertEqual(policy.snapshot.effectiveLevel, .critical)
    }
}
