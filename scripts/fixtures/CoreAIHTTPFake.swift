import ANERunnerCore
import Foundation
import Synchronization

/// CPU-only service fixture. The real HTTP transport and protocol DTOs are
/// compiled alongside this file; no model runtime is imported or constructed.
final class FakeBackend: CoreAIServiceBackend {
    let modelID = "cpu-transport-only"
    let probeID: String
    init(probeID: String) { self.probeID = probeID }
    struct State {
        var active: [UUID: CoreAIRequestCancellation] = [:]
        var admitted = 0, completed = 0, cancelled = 0, errors = 0, heartbeats = 0
        var stopping = false
    }
    private let state = Mutex(State())
    func health() -> Data {
        let values = state.withLock { s in ["active": s.active.count, "admitted": s.admitted,
            "completed": s.completed, "cancelled": s.cancelled, "errors": s.errors,
            "heartbeats": s.heartbeats, "stopping": s.stopping ? 1 : 0] }
        var object = values.mapValues { $0 as Any }
        object["probe_id"] = probeID
        return try! JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    }
    func submit(_ request: QwenHTTPChatRequest,
                onEvent: @escaping @Sendable (CoreAIServiceEvent) async throws -> Void) throws -> CoreAIRequestCancellation {
        let id = UUID(), cancellation = CoreAIRequestCancellation()
        state.withLock { $0.active[id] = cancellation; $0.admitted += 1 }
        Task.detached { [self] in
            defer { _ = state.withLock { $0.active.removeValue(forKey: id) } }
            do {
                try await onEvent(.started)
                let mode = request.messages.last!.content
                if mode == "slow" {
                    for _ in 0..<1000 {
                        try await Task.sleep(for: .milliseconds(20))
                        try cancellation.check()
                        try await onEvent(.heartbeat)
                        state.withLock { $0.heartbeats += 1 }
                    }
                } else if mode == "flood" {
                    for _ in 0..<10000 {
                        try cancellation.check()
                        try await onEvent(.delta(String(repeating: "a", count: 4096)))
                    }
                } else if mode == "overflow" {
                    try await onEvent(.delta(String(repeating: "x", count: 1_048_577)))
                } else {
                    try await onEvent(.heartbeat)
                    try await onEvent(.delta("你好"))
                }
                try await onEvent(.completed(.init(text: "你好", finishReason: "stop", promptTokens: 3,
                    completionTokens: 2, cachedTokens: 0, prefillSeconds: 0, decodeSeconds: 0)))
                state.withLock { $0.completed += 1 }
            } catch {
                state.withLock { value in
                    if cancellation.isCancelled || error is CancellationError { value.cancelled += 1 }
                    else { value.errors += 1 }
                }
                try? await onEvent(.failed(status: 500, message: "Fake producer stopped"))
            }
        }
        return cancellation
    }
    func shutdown() {
        state.withLock { value in
            value.stopping = true
            for token in value.active.values { token.cancel() }
        }
    }
    func drain() async -> Bool {
        for _ in 0..<100 {
            if state.withLock({ $0.active.isEmpty }) { return true }
            try? await Task.sleep(for: .milliseconds(20))
        }
        return false
    }
}

@main struct FakeMain {
    static func main() async throws {
        guard CommandLine.arguments.count == 4, let port = UInt16(CommandLine.arguments[2]), port > 0 else {
            throw NSError(domain: "CPUProbeArguments", code: 1)
        }
        let backend = FakeBackend(probeID: CommandLine.arguments[3])
        var configuration = CoreAIHTTPServer.Configuration()
        configuration.port = port
        configuration.maxConnections = 3
        configuration.maxBodyBytes = 1024
        configuration.maxOutputBytes = 1_048_576
        configuration.receiveTimeoutSeconds = 0.5
        configuration.requestTimeoutSeconds = 1.2
        configuration.sendTimeoutSeconds = 0.15
        try await CoreAIHTTPServer.run(configuration: configuration, backend: backend)
        let drained = await backend.drain()
        let report: [String: Any] = ["drained": drained,
            "state": try JSONSerialization.jsonObject(with: backend.health())]
        try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys, .prettyPrinted])
            .write(to: URL(fileURLWithPath: CommandLine.arguments[1]))
        guard drained else { throw NSError(domain: "CPUProbe", code: 1) }
    }
}
