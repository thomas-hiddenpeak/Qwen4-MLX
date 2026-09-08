import Dispatch
import Foundation

/// Cooperatively pumped prefill/decode queues on ONE inference
/// executor and ONE generator/model. This object and its private GPU handoffs
/// are deliberately not Sendable. There is no background worker, parallel GPU
/// execution, cross-process transfer, or HTTP service. The default completes
/// whole stages; opt-in cooperative mode yields at chunk/round boundaries.
public final class QwenLocalScheduler {
    public enum ExecutionMode: String, Codable, Sendable { case wholeStages, cooperative }
    public struct Limits: Codable, Sendable {
        public let maxQueuedPrefills, maxReadyDecodes, maxResidentTokens, maxConsecutivePrefills: Int
        public let executionMode: ExecutionMode
        public let decodeBurst, maxResidentSequences: Int
        public init(maxQueuedPrefills: Int = 8, maxReadyDecodes: Int = 2,
                    maxResidentTokens: Int = 32768, maxConsecutivePrefills: Int = 1,
                    executionMode: ExecutionMode = .wholeStages, decodeBurst: Int = 4,
                    maxResidentSequences: Int = 2) {
            self.maxQueuedPrefills = maxQueuedPrefills; self.maxReadyDecodes = maxReadyDecodes
            self.maxResidentTokens = maxResidentTokens; self.maxConsecutivePrefills = maxConsecutivePrefills
            self.executionMode = executionMode; self.decodeBurst = decodeBurst
            self.maxResidentSequences = maxResidentSequences
        }
    }
    public enum Error: Swift.Error, LocalizedError, Equatable {
        case queueFull, overBudget, invalidLimits(String), busy, closed(String)
        public var errorDescription: String? {
            switch self {
            case .queueFull: return "The local prefill queue is full"
            case .overBudget: return "The local scheduler token reservation budget is exhausted"
            case .invalidLimits(let message): return "Invalid scheduler limits: \(message)"
            case .busy: return "Scheduler pumping or discard is not reentrant; use the active cancellation token"
            case .closed(let message): return "The local scheduler is closed: \(message)"
            }
        }
    }
    public enum Stage: String, Codable, Sendable { case prefill, decode }
    public enum EventKind: String, Codable, Sendable {
        case prefillProgress, prefillReady, decodeProgress, completed, failed, cancelled
    }
    public struct Timing: Codable, Sendable {
        public let prefillQueueWaitSeconds, readyQueueWaitSeconds: Double?
        /// Accumulated active scheduler stage wall time, excluding resume waits.
        /// Prefill includes first-use head
        /// preparation; decode includes callbacks. Kernel/inference timings
        /// remain separately available in prefill/result statistics.
        public let prefillStageSeconds, decodeStageSeconds: Double?
        public let totalQueueWaitSeconds, elapsedSeconds: Double
        /// Split initial admission wait from later prefill resumptions. The
        /// generation result's TTFT already includes its prefill suspensions.
        public var initialPrefillQueueWaitSeconds: Double? = nil
        public var prefillResumeWaitSeconds: Double? = nil
    }
    public struct Event: Codable, Sendable {
        public let jobID: UUID
        public let kind: EventKind
        public let stage: Stage
        public let prefill: QwenPrefillStatistics?
        public let result: QwenGenerationResult?
        public let timing: Timing
        public let errorDescription: String?
        public let processedPromptTokens: Int?
        public let generatedTokenCount: Int?
        /// Stable machine-readable classification; older reports omit it.
        public var errorCode: String? = nil
    }
    public struct Snapshot: Codable, Sendable {
        public let queuedPrefills, readyDecodes, reservedTokens: Int
        public let runningJob: UUID?
        public let runningStage: Stage?
        public let pendingEvents, consecutivePrefills: Int
        public let isIdle, acceptingJobs: Bool
        public let unavailableReason: String?
        public let queuedPrefillIDs, readyDecodeIDs: [UUID]
        public let residentSequences: Int
        /// Producers explicitly waiting for SSD IO or another prefix leader.
        /// Nil in older reports or backends without wait-state observation.
        public var waitingPrefixSequences: Int? = nil
    }

    private let core: QwenLocalSchedulerCore<QwenLocalWork>

    public init(generator: QwenGenerator, limits: Limits = Limits()) throws {
        core = try QwenLocalSchedulerCore(limits: limits, backend: .init(
            validate: { try generator.validateRequest($0) },
            prefill: { request, cancellation in
                let ready = try generator.prefill(request, cancellation: cancellation)
                let work = QwenLocalWork(); work.ready = ready
                work.processed = request.tokens.count
                return .init(value: work, statistics: ready.statistics)
            },
            decode: { work, cancellation, onToken in
                guard let ready = work.ready else { throw QwenGenerationError.unavailable("Missing ready state") }
                let result = try generator.decode(ready, cancellation: cancellation, onToken: onToken)
                work.generated = result.tokens.count
                return result
            },
            discard: { $0.discard() },
            checkHealth: { try generator.model.withExclusiveGeneration {} },
            prefillSlice: { request, cancellation, previous in
                let work = previous ?? QwenLocalWork()
                if work.producer == nil {
                    work.producer = try generator.beginPrefill(request, cancellation: cancellation)
                }
                guard let session = work.producer else { throw QwenGenerationError.unavailable("Missing prefill cursor") }
                defer { work.processed = session.processedTokenCount }
                let ready = try generator.stepPrefill(session, cancellation: cancellation)
                if let ready { work.ready = ready; work.producer = nil }
                return .init(value: work, statistics: ready?.statistics, complete: ready != nil)
            },
            decodeSlice: { work, cancellation, onToken in
                if work.consumer == nil {
                    guard let ready = work.ready else { throw QwenGenerationError.unavailable("Missing ready state") }
                    work.consumer = try generator.beginDecode(ready, cancellation: cancellation)
                    work.ready = nil
                }
                guard let session = work.consumer else { throw QwenGenerationError.unavailable("Missing decode cursor") }
                defer { work.generated = session.generatedTokenCount }
                let result = try generator.stepDecode(session, cancellation: cancellation, onToken: onToken)
                return result
            },
            progress: { ($0.processed, $0.generated) },
            isWaitingForPrefixCache: { $0.producer?.isWaitingForPrefixCache ?? false }))
    }

    /// Admission reserves prompt.count + maxTokens immediately, including
    /// queued work, until completion/failure/cancellation. This conservative
    /// logical quota is NOT a physical-memory byte limit: fixed GDN states,
    /// MTP state, allocator caches and verification temporaries also consume
    /// memory. Queue/ready limits additionally bound concurrent request states.
    /// Submission and snapshots are safe from a callback on this executor.
    @discardableResult
    public func submit(_ request: QwenGenerationRequest, cancellation: QwenCancellation? = nil,
                       onToken: ((Int32) throws -> Void)? = nil) throws -> UUID {
        try core.submit(request, cancellation: cancellation, onToken: onToken)
    }

    /// One stage/slice or one pending terminal event. Reentrant pumping throws busy.
    /// Stage failures become events, never retries: even a callback can throw
    /// `busy` after consuming a handoff. A poisoned model closes this scheduler
    /// and generates a terminal event for every flushed job.
    public func runNext() throws -> Event? { try core.runNext() }

    /// Remove a queued/ready job without GPU execution. During a running stage,
    /// request cancellation through the QwenCancellation supplied at submit;
    /// direct queue discard is rejected until the stage boundary.
    public func cancel(_ jobID: UUID) throws -> Event? { try core.cancel(jobID) }

    /// Cancel all waiting work and return every pending terminal event. Rejected
    /// during a callback/stage; does not reopen a scheduler closed by model failure.
    public func discardAll() throws -> [Event] { try core.discardAll() }
    public func snapshot() -> Snapshot { core.snapshot() }
}

/// Only the owning scheduler executor can create, advance or release this work.
private final class QwenLocalWork {
    var producer: QwenPrefillSession?
    var ready: QwenPrefillResult?
    var consumer: QwenDecodeSession?
    var processed = 0, generated = 0
    func discard() {
        // All backend calls have returned; no cursor is active here.
        try? producer?.discard(); producer = nil
        ready?.discard(); ready = nil
        try? consumer?.discard(); consumer = nil
    }
}

/// The actual queue engine, parameterized only over the non-Sendable prepared
/// payload. CPU tests inject stage functions into this same lifecycle/policy.
final class QwenLocalSchedulerCore<Prepared> {
    typealias Limits = QwenLocalScheduler.Limits
    typealias Event = QwenLocalScheduler.Event
    typealias Stage = QwenLocalScheduler.Stage
    typealias Kind = QwenLocalScheduler.EventKind
    typealias SchedulerError = QwenLocalScheduler.Error
    struct Prefilled { let value: Prepared; let statistics: QwenPrefillStatistics }
    struct PrefillSlice { let value: Prepared; let statistics: QwenPrefillStatistics?; let complete: Bool }
    struct Backend {
        let validate: (QwenGenerationRequest) throws -> Void
        let prefill: (QwenGenerationRequest, QwenCancellation) throws -> Prefilled
        let decode: (Prepared, QwenCancellation, ((Int32) throws -> Void)?) throws -> QwenGenerationResult
        let discard: (Prepared) -> Void
        var checkHealth: (() throws -> Void)? = nil
        var prefillSlice: ((QwenGenerationRequest, QwenCancellation, Prepared?) throws -> PrefillSlice)? = nil
        var decodeSlice: ((Prepared, QwenCancellation, ((Int32) throws -> Void)?) throws -> QwenGenerationResult?)? = nil
        var progress: ((Prepared) -> (prompt: Int, generated: Int))? = nil
        var isWaitingForPrefixCache: ((Prepared) -> Bool)? = nil
    }
    private final class Job {
        let id = UUID()
        let request: QwenGenerationRequest
        let cancellation: QwenCancellation
        let onToken: ((Int32) throws -> Void)?
        let reservation: Int
        let submittedAt: UInt64
        var prefillStartedAt, prefillEndedAt, decodeStartedAt, decodeEndedAt: UInt64?
        var prefillEnqueuedAt, decodeEnqueuedAt: UInt64?
        var prefillWait = 0.0, decodeWait = 0.0
        var prefillActive = 0.0, decodeActive = 0.0
        var prefillComplete = false
        var prepared: Prepared?
        var prefillStatistics: QwenPrefillStatistics?
        init(request: QwenGenerationRequest, cancellation: QwenCancellation,
             onToken: ((Int32) throws -> Void)?, reservation: Int, now: UInt64) {
            self.request = request; self.cancellation = cancellation; self.onToken = onToken
            self.reservation = reservation; submittedAt = now; prefillEnqueuedAt = now
        }
    }
    private let limits: Limits
    private let backend: Backend
    private let now: () -> UInt64
    private var jobs: [UUID: Job] = [:]
    private var prefills: [UUID] = [], ready: [UUID] = []
    private var pendingEvents: [Event] = []
    private var reservedTokens = 0, consecutivePrefills = 0, consecutiveDecodes = 0
    private var active: UUID?
    private var activeStage: Stage?
    private var unavailableReason: String?

    init(limits: Limits, backend: Backend,
         now: @escaping () -> UInt64 = { DispatchTime.now().uptimeNanoseconds }) throws {
        guard limits.maxQueuedPrefills > 0, limits.maxReadyDecodes > 0,
              limits.maxResidentTokens > 0, limits.maxConsecutivePrefills > 0,
              limits.decodeBurst > 0, limits.maxResidentSequences > 0 else {
            throw SchedulerError.invalidLimits("queue, ready, token and fairness limits must be positive")
        }
        if limits.executionMode == .cooperative && (backend.prefillSlice == nil || backend.decodeSlice == nil) {
            throw SchedulerError.invalidLimits("cooperative mode requires resumable stage backends")
        }
        self.limits = limits; self.backend = backend; self.now = now
    }
    deinit {
        for job in jobs.values { if let prepared = job.prepared { backend.discard(prepared) } }
    }

    @discardableResult
    func submit(_ request: QwenGenerationRequest, cancellation: QwenCancellation? = nil,
                onToken: ((Int32) throws -> Void)? = nil) throws -> UUID {
        if let unavailableReason { throw SchedulerError.closed(unavailableReason) }
        let cancellation = cancellation ?? QwenCancellation()
        try cancellation.check()
        try backend.validate(request)
        let (reservation, overflow) = request.tokens.count.addingReportingOverflow(request.maxTokens)
        guard !request.tokens.isEmpty, request.maxTokens > 0, !overflow, reservation > 0 else {
            throw QwenGenerationError.invalidRequest("invalid scheduler token reservation")
        }
        // Keep a slot for a running cooperative producer until it either
        // completes or returns to the prefill queue. Partial producers count.
        let waiting = prefills.count + (limits.executionMode == .cooperative && activeStage == .prefill ? 1 : 0)
        guard waiting < limits.maxQueuedPrefills else { throw SchedulerError.queueFull }
        guard reservation <= limits.maxResidentTokens - reservedTokens else { throw SchedulerError.overBudget }
        let job = Job(request: request, cancellation: cancellation, onToken: onToken,
                      reservation: reservation, now: now())
        jobs[job.id] = job; prefills.append(job.id); reservedTokens += reservation
        return job.id
    }

    func snapshot() -> QwenLocalScheduler.Snapshot {
        .init(queuedPrefills: prefills.count, readyDecodes: ready.count, reservedTokens: reservedTokens,
              runningJob: active, runningStage: activeStage, pendingEvents: pendingEvents.count,
              consecutivePrefills: consecutivePrefills,
              isIdle: jobs.isEmpty && active == nil && pendingEvents.isEmpty,
              acceptingJobs: unavailableReason == nil, unavailableReason: unavailableReason,
              queuedPrefillIDs: prefills, readyDecodeIDs: ready, residentSequences: residentCount,
              waitingPrefixSequences: backend.isWaitingForPrefixCache.map { waiting in
                  prefills.reduce(0) { count, id in
                      guard let prepared = jobs[id]?.prepared else { return count }
                      return count + (waiting(prepared) ? 1 : 0)
                  }
              })
    }

    private var residentCount: Int { jobs.values.filter { $0.prefillStartedAt != nil }.count }
    private func seconds(_ start: UInt64, _ end: UInt64) -> Double { Double(end - start) * 1e-9 }

    func runNext() throws -> Event? {
        guard active == nil else { throw SchedulerError.busy }
        if !pendingEvents.isEmpty { return pendingEvents.removeFirst() }
        // External cancellation can be observed without executing either stage.
        if let id = (prefills + ready).first(where: { jobs[$0]?.cancellation.isCancelled == true }),
           let job = jobs[id] {
            return terminate(job, kind: .cancelled, stage: job.prefillComplete ? .decode : .prefill,
                             at: now(), message: QwenGenerationError.cancelled.localizedDescription)
        }
        let cooperative = limits.executionMode == .cooperative
        // A partial producer already owns a state slot. With the state limit
        // reached, skip new arrivals to let those existing producers progress.
        let prefillIndex = ready.count < limits.maxReadyDecodes ? prefills.firstIndex(where: { id in
            !cooperative || jobs[id]?.prefillStartedAt != nil || residentCount < limits.maxResidentSequences
        }) : nil
        let stage: Stage
        if !ready.isEmpty && (prefillIndex == nil || consecutivePrefills >= limits.maxConsecutivePrefills ||
                              (cooperative && consecutiveDecodes > 0 && consecutiveDecodes < limits.decodeBurst)) {
            stage = .decode
        } else if prefillIndex != nil {
            stage = .prefill
        } else if !ready.isEmpty { stage = .decode }
        else { return nil }
        let id = stage == .prefill ? prefills.remove(at: prefillIndex!) : ready.removeFirst()
        guard let job = jobs[id] else { preconditionFailure("Scheduler queue lost its job") }
        active = id; activeStage = stage
        defer { active = nil; activeStage = nil }
        let started = now()
        if stage == .prefill {
            if let enqueued = job.prefillEnqueuedAt { job.prefillWait += seconds(enqueued, started) }
            job.prefillEnqueuedAt = nil
            if job.prefillStartedAt == nil { job.prefillStartedAt = started }
            consecutiveDecodes = 0
        } else {
            if let enqueued = job.decodeEnqueuedAt { job.decodeWait += seconds(enqueued, started) }
            job.decodeEnqueuedAt = nil
            if job.decodeStartedAt == nil { job.decodeStartedAt = started }
            consecutivePrefills = 0
            if consecutiveDecodes < limits.decodeBurst { consecutiveDecodes += 1 }
        }
        var accounted = false
        func account(_ end: UInt64) {
            guard !accounted else { return }
            if stage == .prefill { job.prefillActive += seconds(started, end) }
            else { job.decodeActive += seconds(started, end) }
            accounted = true
        }
        do {
            try job.cancellation.check()
            switch stage {
            case .prefill:
                let complete: Bool
                if cooperative, let slice = backend.prefillSlice {
                    let output = try slice(job.request, job.cancellation, job.prepared)
                    job.prepared = output.value; job.prefillStatistics = output.statistics
                    complete = output.complete
                } else {
                    let output = try backend.prefill(job.request, job.cancellation)
                    job.prepared = output.value; job.prefillStatistics = output.statistics
                    complete = true
                }
                let ended = now(); account(ended)
                try job.cancellation.check()
                if consecutivePrefills < limits.maxConsecutivePrefills { consecutivePrefills += 1 }
                if complete {
                    job.prefillComplete = true; job.prefillEndedAt = ended
                    job.decodeEnqueuedAt = ended; ready.append(id)
                    return event(job, kind: .prefillReady, stage: .prefill, at: ended)
                }
                job.prefillEnqueuedAt = ended; prefills.append(id)
                return event(job, kind: .prefillProgress, stage: .prefill, at: ended)
            case .decode:
                guard let prepared = job.prepared else { preconditionFailure("Ready job lost its handoff") }
                let result: QwenGenerationResult?
                if cooperative, let slice = backend.decodeSlice {
                    result = try slice(prepared, job.cancellation, job.onToken)
                } else { result = try backend.decode(prepared, job.cancellation, job.onToken) }
                let ended = now(); account(ended)
                try job.cancellation.check()
                if let result {
                    job.decodeEndedAt = ended
                    return terminate(job, kind: .completed, stage: .decode, at: ended, result: result)
                }
                job.decodeEnqueuedAt = ended; ready.append(id)
                return event(job, kind: .decodeProgress, stage: .decode, at: ended)
            }
        } catch {
            let ended = now()
            account(ended)
            if stage == .prefill { job.prefillEndedAt = ended }
            else { job.decodeEndedAt = ended }
            let kind: Kind = (error as? QwenGenerationError) == .cancelled ? .cancelled : .failed
            let errorCode: String?
            if case QwenGenerationError.resourceLimit(_) = error { errorCode = "resource_limit" }
            else { errorCode = nil }
            let failed = terminate(job, kind: kind, stage: stage, at: ended,
                message: error.localizedDescription, errorCode: errorCode)
            // The generator has released model admission before throwing here.
            // An original cancellation/busy error does not prove recovery was
            // healthy. Only an explicit unavailable health result poisons us;
            // transient busy is not treated as permanent model failure.
            var poison: String?
            if let checkHealth = backend.checkHealth {
                do { try checkHealth() }
                catch QwenGenerationError.unavailable(let reason) { poison = reason }
                catch { /* Preserve this job's failure; do not retry it. */ }
            } else if case QwenGenerationError.unavailable(let reason) = error { poison = reason }
            if let poison { closePending(reason: poison, at: now()) }
            return failed
        }
    }

    func cancel(_ id: UUID) throws -> Event? {
        guard active == nil else { throw SchedulerError.busy }
        guard let job = jobs[id] else { return nil }
        return terminate(job, kind: .cancelled, stage: job.prefillComplete ? .decode : .prefill,
                         at: now(), message: QwenGenerationError.cancelled.localizedDescription)
    }
    func discardAll() throws -> [Event] {
        guard active == nil else { throw SchedulerError.busy }
        var events = pendingEvents; pendingEvents.removeAll()
        for id in prefills + ready {
            if let job = jobs[id] {
                events.append(terminate(job, kind: .cancelled, stage: job.prefillComplete ? .decode : .prefill,
                                        at: now(), message: QwenGenerationError.cancelled.localizedDescription))
            }
        }
        consecutivePrefills = 0; consecutiveDecodes = 0
        return events
    }
    private func closePending(reason: String, at time: UInt64) {
        unavailableReason = reason
        for id in prefills + ready {
            if let job = jobs[id] {
                pendingEvents.append(terminate(job, kind: .failed, stage: job.prefillComplete ? .decode : .prefill,
                                               at: time, message: "Model unavailable: \(reason)"))
            }
        }
        consecutivePrefills = 0; consecutiveDecodes = 0
    }
    private func terminate(_ job: Job, kind: Kind, stage: Stage, at time: UInt64,
                           result: QwenGenerationResult? = nil, message: String? = nil,
                           errorCode: String? = nil) -> Event {
        let output = event(job, kind: kind, stage: stage, at: time,
            result: result, message: message, errorCode: errorCode)
        // Release even when decode was rejected before claiming its payload.
        // No failed job is retried, regardless of the error's enum case.
        if let prepared = job.prepared { job.prepared = nil; backend.discard(prepared) }
        precondition(jobs.removeValue(forKey: job.id) != nil, "Scheduler released a job twice")
        prefills.removeAll { $0 == job.id }; ready.removeAll { $0 == job.id }
        reservedTokens -= job.reservation
        return output
    }
    private func event(_ job: Job, kind: Kind, stage: Stage, at time: UInt64,
                       result: QwenGenerationResult? = nil, message: String? = nil,
                       errorCode: String? = nil) -> Event {
        let prefillWait = job.prefillWait + (job.prefillEnqueuedAt.map { seconds($0, time) } ?? 0)
        let initialWait = seconds(job.submittedAt, job.prefillStartedAt ?? time)
        let readyWait = job.prefillComplete
            ? job.decodeWait + (job.decodeEnqueuedAt.map { seconds($0, time) } ?? 0) : nil
        let prefillTime: Double? = job.prefillStartedAt == nil ? nil : job.prefillActive
        let decodeTime: Double? = job.decodeStartedAt == nil ? nil : job.decodeActive
        let progress = job.prepared.flatMap { backend.progress?($0) }
        let timing = QwenLocalScheduler.Timing(
            prefillQueueWaitSeconds: prefillWait, readyQueueWaitSeconds: readyWait,
            prefillStageSeconds: prefillTime, decodeStageSeconds: decodeTime,
            totalQueueWaitSeconds: prefillWait + (readyWait ?? 0),
            elapsedSeconds: seconds(job.submittedAt, time),
            initialPrefillQueueWaitSeconds: initialWait,
            prefillResumeWaitSeconds: max(0, prefillWait - initialWait))
        return .init(jobID: job.id, kind: kind, stage: stage, prefill: job.prefillStatistics,
                     result: result, timing: timing, errorDescription: message,
                     processedPromptTokens: progress?.prompt, generatedTokenCount: progress?.generated,
                     errorCode: errorCode)
    }
}
