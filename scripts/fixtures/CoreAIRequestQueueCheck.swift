import Foundation
import Synchronization

private struct CheckFailure: Error, CustomStringConvertible {
    let description: String
}
private func require(_ condition: @autoclosure () -> Bool, _ message: String) throws {
    if !condition() { throw CheckFailure(description: message) }
}
private typealias Queue = CoreAIRequestQueue<Int>

private final class Ledger: Sendable {
    struct State {
        var admitted = Set<Int>(), cancelled = Set<Int>(), finished = Set<Int>(), drained = Set<Int>()
        var errors: [String] = []
        var producersRemaining: Int
        var threadsRemaining: Int
    }
    let state: Mutex<State>
    init(producers: Int, threads: Int) {
        state = Mutex(State(producersRemaining: producers, threadsRemaining: threads))
    }
    func record(_ key: WritableKeyPath<State, Set<Int>>, _ job: Int) {
        state.withLock { value in
            if !value[keyPath: key].insert(job).inserted { value.errors.append("duplicate accounting for \(job)") }
        }
    }
    func error(_ message: String) { state.withLock { $0.errors.append(message) } }
    func inspect(_ queue: Queue) {
        let s = queue.snapshot
        if s.reservedCount > s.limit || s.activeCount > 1 || s.queuedCount < 0 {
            error("invalid atomic snapshot")
        }
    }
    func wait() throws {
        let deadline = ProcessInfo.processInfo.systemUptime + 10
        while state.withLock({ $0.threadsRemaining > 0 }) {
            try require(ProcessInfo.processInfo.systemUptime < deadline, "concurrent check timed out")
            Thread.sleep(forTimeInterval: 0.001)
        }
    }
    func verify() throws -> Int {
        let s = state.withLock { $0 }
        try require(s.errors.isEmpty, s.errors.joined(separator: "; "))
        try require(s.cancelled.isDisjoint(with: s.finished) && s.cancelled.isDisjoint(with: s.drained)
                    && s.finished.isDisjoint(with: s.drained), "request accounted for twice")
        try require(s.admitted == s.cancelled.union(s.finished).union(s.drained), "lost or invented request")
        return s.admitted.count
    }
}

@main private struct QueueCheck {
    static func capacity() throws -> [String: Any] {
        let q = Queue(limit: 2), first = UUID(), second = UUID(), third = UUID()
        try q.enqueue(id: first, job: 1)
        try require(q.takeNext()?.id == first, "first request was not activated")
        try q.enqueue(id: second, job: 2)
        do { try q.enqueue(id: third, job: 3); throw CheckFailure(description: "active reservation was not charged") }
        catch Queue.AdmissionError.full { }
        try require(q.snapshot.activeCount == 1 && q.snapshot.queuedCount == 1, "incorrect counts")
        try require(q.takeNext() == nil, "two jobs became active")
        try require(q.cancelQueued(id: first) == nil, "queued cancellation removed active request")
        try require(!q.finishActive(id: third), "stale finish released active reservation")
        try require(q.cancelQueued(id: second)?.job == 2, "queued cancellation did not return job")
        try q.enqueue(id: third, job: 3)
        try require(q.finishActive(id: first), "active completion failed")
        try require(q.takeNext()?.job == 3, "released slot was not reusable")
        try require(q.finishActive(id: third) && q.snapshot.reservedCount == 0, "queue not empty")
        return ["limit": 2]
    }

    static func duplicates() throws -> [String: Any] {
        let q = Queue(limit: 3), id = UUID()
        try q.enqueue(id: id, job: 1)
        for active in [false, true] {
            if active { _ = q.takeNext() }
            do { try q.enqueue(id: id, job: 9); throw CheckFailure(description: "duplicate request id admitted") }
            catch Queue.AdmissionError.duplicateID { }
        }
        try require(q.finishActive(id: id), "active completion failed")
        try q.enqueue(id: id, job: 2)
        try require(q.cancelQueued(id: id)?.job == 2, "completed ID could not be reused")
        return [:]
    }

    static func cancelRepeatedly() throws -> [String: Any] {
        let q = Queue(limit: 2), active = UUID()
        try q.enqueue(id: active, job: -1)
        _ = q.takeNext()
        for n in 0..<5000 {
            let id = UUID()
            try q.enqueue(id: id, job: n)
            try require(q.cancelQueued(id: id)?.job == n, "cancelled queued job lost")
            try require(q.cancelQueued(id: id) == nil, "duplicate cancellation returned a job")
            try require(q.snapshot.reservedCount == 1, "cancelled job retained its slot")
        }
        try require(q.finishActive(id: active), "active reservation was altered")
        return ["cancel_enqueue_cycles": 5000]
    }

    static func fifo() throws -> [String: Any] {
        let q = Queue(limit: 100)
        var entries: [(UUID, Int)] = []
        for n in 0..<100 {
            let id = UUID()
            entries.append((id, n))
            try q.enqueue(id: id, job: n)
        }
        for (id, n) in entries where n % 3 == 1 { _ = q.cancelQueued(id: id) }
        for (id, n) in entries where n % 3 != 1 {
            let next = q.takeNext()
            try require(next?.id == id && next?.job == n, "FIFO changed after interior removal")
            try require(q.finishActive(id: id), "FIFO completion failed")
        }
        try require(q.takeNext() == nil && q.snapshot.reservedCount == 0, "FIFO did not drain")
        return ["enqueued": 100, "cancelled": 33]
    }

    static func stop() throws -> [String: Any] {
        let q = Queue(limit: 4), active = UUID(), queued = [UUID(), UUID(), UUID()]
        try q.enqueue(id: active, job: 0)
        _ = q.takeNext()
        for (n, id) in queued.enumerated() { try q.enqueue(id: id, job: n + 1) }
        try require(q.stopAndDrain().map(\.id) == queued, "stop did not return queued jobs in order")
        try require(q.snapshot.stopped && q.snapshot.reservedCount == 1, "stop released active reservation")
        try require(q.stopAndDrain().isEmpty && q.takeNext() == nil, "stop was not idempotent")
        do { try q.enqueue(id: UUID(), job: 5); throw CheckFailure(description: "admission continued after stop") }
        catch Queue.AdmissionError.stopped { }
        try require(q.cancelQueued(id: active) == nil, "stop made active cancellation removable")
        try require(q.finishActive(id: active) && q.snapshot.reservedCount == 0, "active cannot finish after stop")
        return [:]
    }

    static func concurrent() throws -> [String: Any] {
        let q = Queue(limit: 8), ledger = Ledger(producers: 6, threads: 8)
        for producer in 0..<6 {
            Thread.detachNewThread {
                defer { ledger.state.withLock { $0.producersRemaining -= 1; $0.threadsRemaining -= 1 } }
                for n in 0..<150 {
                    let id = UUID(), job = producer * 150 + n
                    var admitted = false
                    for _ in 0..<100_000 {
                        do { try q.enqueue(id: id, job: job); admitted = true; break }
                        catch Queue.AdmissionError.full { Thread.sleep(forTimeInterval: 0.00001) }
                        catch { ledger.error("unexpected admission error: \(error)"); return }
                    }
                    guard admitted else { ledger.error("producer admission timed out"); return }
                    ledger.record(\.admitted, job)
                    if n % 2 == 0, let removed = q.cancelQueued(id: id) { ledger.record(\.cancelled, removed.job) }
                    ledger.inspect(q)
                }
            }
        }
        for _ in 0..<2 {
            Thread.detachNewThread {
                defer { ledger.state.withLock { $0.threadsRemaining -= 1 } }
                for _ in 0..<100_000 {
                    if let entry = q.takeNext() {
                        Thread.sleep(forTimeInterval: 0.00001)
                        if !q.finishActive(id: entry.id) { ledger.error("active reservation stolen") }
                        ledger.record(\.finished, entry.job)
                    } else if ledger.state.withLock({ $0.producersRemaining == 0 }) && q.snapshot.reservedCount == 0 {
                        return
                    } else { Thread.sleep(forTimeInterval: 0.00001) }
                    ledger.inspect(q)
                }
                ledger.error("consumer timed out")
            }
        }
        try ledger.wait()
        let count = try ledger.verify()
        try require(count == 900 && q.snapshot.reservedCount == 0, "concurrent queue did not drain")
        return ["jobs": count, "producer_threads": 6, "consumer_threads": 2]
    }

    static func concurrentStop() throws -> [String: Any] {
        let q = Queue(limit: 16), ledger = Ledger(producers: 4, threads: 5), active = UUID()
        try q.enqueue(id: active, job: -1)
        _ = q.takeNext()
        ledger.record(\.admitted, -1)
        for producer in 0..<4 {
            Thread.detachNewThread {
                defer { ledger.state.withLock { $0.producersRemaining -= 1; $0.threadsRemaining -= 1 } }
                for n in 0..<100_000 {
                    let id = UUID(), job = producer * 100_000 + n
                    do {
                        try q.enqueue(id: id, job: job)
                        ledger.record(\.admitted, job)
                        if n % 2 == 0, let removed = q.cancelQueued(id: id) { ledger.record(\.cancelled, removed.job) }
                    } catch Queue.AdmissionError.full { Thread.sleep(forTimeInterval: 0.00001) }
                    catch Queue.AdmissionError.stopped { return }
                    catch { ledger.error("unexpected stop admission error"); return }
                }
                ledger.error("stop did not stop producer")
            }
        }
        Thread.detachNewThread {
            defer { ledger.state.withLock { $0.threadsRemaining -= 1 } }
            Thread.sleep(forTimeInterval: 0.002)
            for entry in q.stopAndDrain() { ledger.record(\.drained, entry.job) }
            if !q.finishActive(id: active) { ledger.error("stop lost active job") }
            ledger.record(\.finished, -1)
        }
        try ledger.wait()
        let count = try ledger.verify()
        try require(q.snapshot.stopped && q.snapshot.reservedCount == 0, "concurrent stop did not drain")
        return ["accounted_jobs": count, "producer_threads": 4]
    }

    static func main() throws {
        let tests: [(String, () throws -> [String: Any])] = [
            ("active_plus_queued_limit", capacity), ("duplicate_ids", duplicates),
            ("cancelled_queue_slot_reused_immediately", cancelRepeatedly),
            ("fifo_with_interior_cancellations", fifo), ("stop_and_active_completion", stop),
            ("concurrent_producers_consumers_cancellation", concurrent), ("concurrent_stop", concurrentStop)]
        var results: [[String: Any]] = []
        var passed = true
        for (name, test) in tests {
            let start = ProcessInfo.processInfo.systemUptime
            do {
                var result = try test()
                result["name"] = name; result["passed"] = true
                result["seconds"] = ProcessInfo.processInfo.systemUptime - start
                results.append(result)
            } catch {
                passed = false
                results.append(["name": name, "passed": false, "error": String(describing: error)])
            }
        }
        let report: [String: Any] = ["passed": passed, "cases": results,
            "scope": "Foundation/Synchronization request mailbox only; no model, network or CoreAI runtime"]
        print(String(decoding: try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys]), as: UTF8.self))
        if !passed { throw CheckFailure(description: "request queue checks failed") }
    }
}
