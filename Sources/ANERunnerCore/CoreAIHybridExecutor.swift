import Foundation

/// A process-lifetime executor backed by one physical Foundation Thread.
///
/// Serial DispatchQueues do not guarantee thread affinity. The pinned MLX C API
/// stores stream registrations per thread, so hybrid-model construction, calls,
/// resets, and destruction must run inside the same executor preference scope.
/// This executor preserves that thread across ordinary async suspensions.
///
/// Call a non-actor-isolated async helper from withTaskExecutorPreference(shared).
/// An actor-isolated operation still uses its actor executor: this preference
/// does not override MainActor or another actor's isolation. The model remains
/// non-Sendable and must never escape the helper that creates it.
@available(macOS 15.0, *)
public final class CoreAIHybridExecutor: TaskExecutor {
    public static let shared = CoreAIHybridExecutor()

    /// The sole unchecked conformance covers only the locked job mailbox.
    /// No model, MLX tensor, runtime state, or user closure is declared Sendable.
    /// enqueue/take are the only accessors and always hold the same condition.
    private final class Mailbox: @unchecked Sendable {
        private let condition = NSCondition()
        private var jobs: [UnownedJob] = []

        func enqueue(_ job: UnownedJob) {
            condition.lock()
            jobs.append(job)
            condition.signal()
            condition.unlock()
        }

        func take() -> UnownedJob {
            condition.lock()
            defer { condition.unlock() }
            while jobs.isEmpty { condition.wait() }
            return jobs.removeFirst()
        }
    }

    private let mailbox = Mailbox()

    /// The singleton and its thread deliberately live until process exit.
    /// There is no shutdown race in which a suspended task can later enqueue
    /// onto a stopped worker, and no destructor moving model work elsewhere.
    private init() {
        let worker = Thread { [self] in runLoop() }
        worker.name = "ANERunner.CoreAIHybrid"
        worker.qualityOfService = .userInitiated
        worker.start()
    }

    public func enqueue(_ job: consuming ExecutorJob) {
        mailbox.enqueue(UnownedJob(job))
    }

    private func runLoop() {
        while true {
            let job = mailbox.take()
            autoreleasepool {
                job.runSynchronously(on: asUnownedTaskExecutor())
            }
        }
    }
}
