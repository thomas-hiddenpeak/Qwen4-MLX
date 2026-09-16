import Foundation
import Synchronization

/// Bounded, removable FIFO for a single inference consumer. The active request
/// keeps its reservation until finishActive, including after stopAndDrain.
/// This type never invokes job code, cancellation callbacks, or wakeup code.
final class CoreAIRequestQueue<Job: Sendable>: Sendable {
    enum AdmissionError: Error, Equatable, Sendable {
        case full
        case stopped
        case duplicateID
    }

    struct Entry: Sendable {
        let id: UUID
        let job: Job
    }

    struct Snapshot: Sendable, Equatable {
        let limit: Int
        let activeID: UUID?
        let queuedCount: Int
        let stopped: Bool
        var activeCount: Int { activeID == nil ? 0 : 1 }
        var reservedCount: Int { activeCount + queuedCount }
    }

    private struct State {
        var queued: [Entry] = []
        var activeID: UUID?
        var stopped = false
    }

    let limit: Int
    private let state = Mutex(State())

    init(limit: Int) {
        precondition(limit > 0, "Request queue limit must be positive")
        self.limit = limit
    }

    /// Caller publishes its wakeup only after this method succeeds.
    func enqueue(id: UUID, job: Job) throws {
        try state.withLock { value in
            guard !value.stopped else { throw AdmissionError.stopped }
            guard value.activeID != id, !value.queued.contains(where: { $0.id == id }) else {
                throw AdmissionError.duplicateID
            }
            let reserved = value.queued.count + (value.activeID == nil ? 0 : 1)
            guard reserved < limit else { throw AdmissionError.full }
            value.queued.append(Entry(id: id, job: job))
        }
    }

    /// Atomically transfers the first queued job to the sole active reservation.
    /// An empty queue, stopped admission, or an existing active job returns nil.
    func takeNext() -> Entry? {
        state.withLock { value in
            guard !value.stopped, value.activeID == nil, !value.queued.isEmpty else { return nil }
            let entry = value.queued.removeFirst()
            value.activeID = entry.id
            return entry
        }
    }

    /// Removes a queued reservation immediately. Cancellation of active work
    /// belongs to the consumer and must never release that work's reservation.
    @discardableResult
    func cancelQueued(id: UUID) -> Entry? {
        state.withLock { value in
            guard let index = value.queued.firstIndex(where: { $0.id == id }) else { return nil }
            return value.queued.remove(at: index)
        }
    }

    /// A stale completion cannot release a different request's reservation.
    @discardableResult
    func finishActive(id: UUID) -> Bool {
        state.withLock { value in
            guard value.activeID == id else { return false }
            value.activeID = nil
            return true
        }
    }

    /// Idempotent shutdown. Returned jobs are owned by the caller, which may
    /// cancel or notify them after this method has released the queue's lock.
    func stopAndDrain() -> [Entry] {
        state.withLock { value in
            value.stopped = true
            let entries = value.queued
            value.queued.removeAll(keepingCapacity: false)
            return entries
        }
    }

    var snapshot: Snapshot {
        state.withLock { value in
            Snapshot(limit: limit, activeID: value.activeID,
                     queuedCount: value.queued.count, stopped: value.stopped)
        }
    }
}
