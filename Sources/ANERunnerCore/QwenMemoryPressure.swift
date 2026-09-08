import Dispatch
import Foundation

/// Admission advice from macOS memory-pressure notifications. This is separate
/// from QwenStateBudget: neither a normal signal nor this policy reserves bytes
/// or measures process/MLX allocations. Existing requests are never revoked.
///
/// The initial state is unknown because Dispatch reports pressure changes, not
/// a guaranteed initial sample. Unknown keeps the existing logical-budget
/// policy usable, including optional caching; it must not be reported as an
/// observed normal system. A monitored service should retain the monitor.
public final class QwenMemoryPressurePolicy: @unchecked Sendable {
    public enum Level: String, Codable, CaseIterable, Sendable {
        case unknown, normal, warning, critical

        fileprivate var severity: Int {
            switch self {
            case .unknown, .normal: 0
            case .warning: 1
            case .critical: 2
            }
        }
    }

    /// Only actual pressure conditions can be recorded. Unknown is a lack of
    /// observations, not an event that can clear a critical state.
    public enum Event: String, Codable, CaseIterable, Sendable {
        case normal, warning, critical
        fileprivate var level: Level {
            switch self {
            case .normal: .normal
            case .warning: .warning
            case .critical: .critical
            }
        }
    }

    public enum Source: String, Codable, Sendable {
        case operatingSystem, injected
    }

    public enum Reason: String, Codable, Sendable {
        case notObserved, normal, warning, critical
        case warningRecoveryHold, criticalRecoveryHold
    }

    public enum ConfigurationError: Error, Equatable {
        case invalidRecoveryStableSeconds
    }

    public struct Snapshot: Codable, Equatable, Sendable {
        /// Most recent observation; effectiveLevel can remain more restrictive
        /// while the recovery stability window is still running.
        public let observedLevel, effectiveLevel: Level
        public let lastEventSource: Source?
        public let reason: Reason
        public let allowsNewRequests, allowsOptionalCache: Bool
        public let recoveryTarget: Level?
        public let recoveryRemainingSeconds, lastEventAgeSeconds: TimeInterval?
        public let recoveryStableSeconds: TimeInterval
        /// Delivered notifications, not a count of every OS transition: the
        /// Dispatch source may coalesce changes before invoking its handler.
        public let events, operatingSystemEvents, injectedEvents: UInt64
        public let normalEvents, warningEvents, criticalEvents: UInt64
        public let transitions, recoveries: UInt64
        public let newRequestDenials, optionalCacheDenials: UInt64
        public let trimGeneration: UInt64
        public let trimRequested: Bool
    }

    public let recoveryStableSeconds: TimeInterval
    private let clock: @Sendable () -> TimeInterval
    private let lock = NSLock()
    private var observedLevel: Level = .unknown
    private var effectiveLevel: Level = .unknown
    private var lastEventSource: Source?
    private var lastEventAt: TimeInterval?
    private var lastClock: TimeInterval = 0
    private var recoveryTarget: Level?
    private var recoveryStartedAt: TimeInterval?
    private var events: UInt64 = 0, operatingSystemEvents: UInt64 = 0, injectedEvents: UInt64 = 0
    private var normalEvents: UInt64 = 0, warningEvents: UInt64 = 0, criticalEvents: UInt64 = 0
    private var transitions: UInt64 = 0, recoveries: UInt64 = 0
    private var newRequestDenials: UInt64 = 0, optionalCacheDenials: UInt64 = 0
    private var trimGeneration: UInt64 = 0
    private var trimPending = false

    /// clock is monotonic seconds. Injection is for deterministic CPU tests;
    /// it must be fast and must not call back into this policy while sampled.
    public init(recoveryStableSeconds: TimeInterval = 5,
                clock: @escaping @Sendable () -> TimeInterval = {
                    Double(DispatchTime.now().uptimeNanoseconds) / 1_000_000_000
                }) throws {
        guard recoveryStableSeconds.isFinite, recoveryStableSeconds >= 0 else {
            throw ConfigurationError.invalidRecoveryStableSeconds
        }
        self.recoveryStableSeconds = recoveryStableSeconds
        self.clock = clock
    }

    /// Source defaults to injected so tests/debug callers cannot accidentally
    /// label their synthetic observation as an actual operating-system signal.
    public func observe(_ event: Event, source: Source = .injected) {
        withLock {
            let now = sampleClock()
            finishRecoveryIfReady(at: now)
            observedLevel = event.level
            lastEventSource = source
            lastEventAt = now
            increment(&events)
            switch source {
            case .operatingSystem: increment(&operatingSystemEvents)
            case .injected: increment(&injectedEvents)
            }
            switch event {
            case .normal: increment(&normalEvents)
            case .warning: increment(&warningEvents)
            case .critical: increment(&criticalEvents)
            }

            let level = event.level
            if level.severity >= effectiveLevel.severity {
                recoveryTarget = nil
                recoveryStartedAt = nil
                setEffectiveLevel(level)
            } else {
                // Repeated observations of the same improving condition do not
                // postpone recovery forever. A different target starts a fresh
                // window; any escalation immediately cancels the prior window.
                if recoveryTarget != level {
                    recoveryTarget = level
                    recoveryStartedAt = now
                }
                finishRecoveryIfReady(at: now)
            }
        }
    }

    /// Snapshot reads never count as admission attempts. They do advance a
    /// completed recovery window; no timer or background GPU work is needed.
    public var snapshot: Snapshot {
        withLock {
            let now = sampleClock()
            finishRecoveryIfReady(at: now)
            let remaining = recoveryStartedAt.map {
                max(0, recoveryStableSeconds - elapsed(since: $0, now: now))
            }
            let reason: Reason
            switch effectiveLevel {
            case .unknown: reason = .notObserved
            case .normal: reason = .normal
            case .warning: reason = recoveryTarget == nil ? .warning : .warningRecoveryHold
            case .critical: reason = recoveryTarget == nil ? .critical : .criticalRecoveryHold
            }
            return .init(
                observedLevel: observedLevel, effectiveLevel: effectiveLevel,
                lastEventSource: lastEventSource, reason: reason,
                allowsNewRequests: effectiveLevel != .critical,
                allowsOptionalCache: effectiveLevel.severity == 0,
                recoveryTarget: recoveryTarget, recoveryRemainingSeconds: remaining,
                lastEventAgeSeconds: lastEventAt.map { elapsed(since: $0, now: now) },
                recoveryStableSeconds: recoveryStableSeconds,
                events: events, operatingSystemEvents: operatingSystemEvents, injectedEvents: injectedEvents,
                normalEvents: normalEvents, warningEvents: warningEvents, criticalEvents: criticalEvents,
                transitions: transitions, recoveries: recoveries,
                newRequestDenials: newRequestDenials, optionalCacheDenials: optionalCacheDenials,
                trimGeneration: trimGeneration, trimRequested: trimPending)
        }
    }

    /// Apply at the new-request admission boundary, together with the logical
    /// byte budget. A successful check is advice at this instant, not a lease.
    public func checkNewRequestAdmission() -> Bool {
        withLock {
            finishRecoveryIfReady(at: sampleClock())
            let allowed = effectiveLevel != .critical
            if !allowed { increment(&newRequestDenials) }
            return allowed
        }
    }

    /// Applies to optional cache fill/promotion, prefetch and writeback. It must
    /// not be used to discard already admitted decode state or completed I/O.
    public func checkOptionalCacheAdmission() -> Bool {
        withLock {
            finishRecoveryIfReady(at: sampleClock())
            let allowed = effectiveLevel.severity == 0
            if !allowed { increment(&optionalCacheDenials) }
            return allowed
        }
    }

    /// One consumer (the model executor) can take a coalesced trim suggestion.
    /// Never walk tensors, synchronize a GPU or free model state from the
    /// Dispatch callback. SDK guidance discourages scanning old caches during
    /// pressure: use already-resident metadata for bounded safe reclamation,
    /// and favor reducing future allocations. Active owners remain pinned.
    public func takeTrimRequest() -> UInt64? {
        withLock {
            finishRecoveryIfReady(at: sampleClock())
            guard trimPending else { return nil }
            trimPending = false
            return trimGeneration
        }
    }

    private func setEffectiveLevel(_ level: Level) {
        guard effectiveLevel != level else { return }
        let previous = effectiveLevel
        effectiveLevel = level
        increment(&transitions)
        if level.severity > previous.severity {
            increment(&trimGeneration)
            trimPending = true
        } else if level.severity == 0 {
            // A stale, unconsumed pressure suggestion should not evict cache
            // after the system has already completed its recovery window.
            trimPending = false
        }
    }

    private func finishRecoveryIfReady(at now: TimeInterval) {
        guard let target = recoveryTarget, let started = recoveryStartedAt,
              elapsed(since: started, now: now) >= recoveryStableSeconds else { return }
        recoveryTarget = nil
        recoveryStartedAt = nil
        setEffectiveLevel(target)
        increment(&recoveries)
    }

    private func sampleClock() -> TimeInterval {
        let candidate = clock()
        // A broken injected clock cannot spuriously recover pressure or create
        // NaN/Infinity health fields. Production Dispatch uptime is monotonic.
        if candidate.isFinite, candidate >= lastClock { lastClock = candidate }
        return lastClock
    }

    private func elapsed(since then: TimeInterval, now: TimeInterval) -> TimeInterval {
        max(0, now - then)
    }

    private func increment(_ counter: inout UInt64) {
        if counter != .max { counter += 1 }
    }

    private func withLock<T>(_ body: () -> T) -> T {
        lock.lock()
        defer { lock.unlock() }
        return body()
    }
}

/// Real macOS change notification monitor, verified against the local SDK's
/// dispatch/source.h (MEMORYPRESSURE source, NORMAL/WARN/CRITICAL flags).
/// This does not manufacture an initial normal observation, poll RSS, perform
/// a sysctl-based approximation, or generate pressure. Events can be coalesced;
/// when multiple flags arrive together the most severe condition wins.
public final class QwenMemoryPressureMonitor: @unchecked Sendable {
    public let policy: QwenMemoryPressurePolicy
    private let lock = NSLock()
    private var source: DispatchSourceMemoryPressure?
    private var running = true

    public init(policy: QwenMemoryPressurePolicy) {
        self.policy = policy
        let queue = DispatchQueue(label: "ane-runner.memory-pressure", qos: .utility)
        let source = DispatchSource.makeMemoryPressureSource(
            eventMask: [.normal, .warning, .critical], queue: queue)
        self.source = source
        source.setEventHandler { [weak self] in self?.receiveEvent() }
        source.activate()
    }

    public var isRunning: Bool {
        lock.lock()
        defer { lock.unlock() }
        return running
    }

    /// Idempotent and safe from any thread. Once this returns, this monitor
    /// cannot make another policy observation. Source cancellation itself is
    /// asynchronous, but no pending callback can pass the running guard.
    public func stop() {
        lock.lock()
        defer { lock.unlock() }
        guard running else { return }
        running = false
        source?.cancel()
        source = nil
    }

    deinit { stop() }

    /// Pure decoding shared with CPU tests. Empty/unknown flags do not imply
    /// normal pressure and are ignored.
    static func event(for flags: DispatchSource.MemoryPressureEvent) -> QwenMemoryPressurePolicy.Event? {
        if flags.contains(.critical) { return .critical }
        if flags.contains(.warning) { return .warning }
        if flags.contains(.normal) { return .normal }
        return nil
    }

    private func receiveEvent() {
        lock.lock()
        defer { lock.unlock() }
        guard running, let source, let event = Self.event(for: source.data) else { return }
        // This short policy-only operation is protected against stop(). No
        // user callback or cache/GPU work occurs under either lock.
        policy.observe(event, source: .operatingSystem)
    }
}
