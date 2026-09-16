import ANERunnerCore
import Foundation
import Synchronization

struct CoreAIServiceResult: Sendable {
    let text: String
    let finishReason: String
    let promptTokens: Int
    let completionTokens: Int
    let cachedTokens: Int
    let prefillSeconds: Double
    let decodeSeconds: Double
}

enum CoreAIServiceEvent: Sendable {
    /// Input validation and resource admission have completed.
    case started
    /// Prefill liveness for streaming clients; carries no model output.
    case heartbeat
    case delta(String)
    case completed(CoreAIServiceResult)
    case failed(status: Int, message: String)
}

final class CoreAIRequestCancellation: Sendable {
    private let cancelled = Mutex(false)
    func cancel() { cancelled.withLock { $0 = true } }
    var isCancelled: Bool { cancelled.withLock { $0 } }
    func check() throws {
        if isCancelled || Task.isCancelled { throw CancellationError() }
    }
}

protocol CoreAIServiceBackend: Sendable {
    var modelID: String { get }
    func health() -> Data
    func submit(_ request: QwenHTTPChatRequest,
                onEvent: @escaping @Sendable (CoreAIServiceEvent) async throws -> Void) throws -> CoreAIRequestCancellation
    func shutdown()
}
