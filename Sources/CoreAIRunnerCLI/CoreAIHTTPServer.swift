import ANERunnerCore
import Darwin
import Dispatch
import Foundation
@preconcurrency import Network
import Synchronization

/// HTTP/1.1 transport for the independent CoreAI worker. It does not own or
/// access any model, tensor, tokenizer, or inference executor.
enum CoreAIHTTPServer {
    struct Configuration: Sendable {
        var host: String = "127.0.0.1"
        var port: UInt16 = 11236
        var maxConnections: Int = 32
        var maxBodyBytes: Int = 1_048_576
        /// Total encoded response body budget, including SSE framing.
        var maxOutputBytes: Int = 1_048_576
        var requestTimeoutSeconds: Double = 1800
        var sendTimeoutSeconds: Double = 15
        var receiveTimeoutSeconds: Double = 15

        fileprivate func validate() throws {
            guard ["127.0.0.1", "0.0.0.0"].contains(host), port > 0,
                  (1...256).contains(maxConnections),
                  (1...16_777_216).contains(maxBodyBytes),
                  (4096...16_777_216).contains(maxOutputBytes),
                  requestTimeoutSeconds.isFinite, (1...86_400).contains(requestTimeoutSeconds),
                  sendTimeoutSeconds.isFinite, (0.1...300).contains(sendTimeoutSeconds),
                  receiveTimeoutSeconds.isFinite, (0.1...300).contains(receiveTimeoutSeconds) else {
                throw QwenHTTPProtocolError(statusCode: 500, message: "Invalid CoreAI HTTP transport configuration")
            }
        }
    }

    /// A CLI process runs one listener. Cancellation and SIGINT/SIGTERM stop
    /// admission, cancel requests, release send waiters, and stop the backend.
    /// backend.shutdown() must be nonblocking; its owner may separately await
    /// inference drain after this method returns.
    static func run(configuration: Configuration = .init(), backend: any CoreAIServiceBackend) async throws {
        try configuration.validate()
        let server = try CoreAIHTTPTransport(configuration: configuration, backend: backend)
        let previousINT = Darwin.signal(SIGINT, SIG_IGN)
        let previousTERM = Darwin.signal(SIGTERM, SIG_IGN)
        defer {
            Darwin.signal(SIGINT, previousINT)
            Darwin.signal(SIGTERM, previousTERM)
        }
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
                server.network.async { server.start(continuation) }
            }
        } onCancel: {
            server.network.async { server.stop(error: CancellationError()) }
        }
    }
}

/// A broken producer cannot queue arbitrarily many callbacks while a socket is
/// stalled. The supported worker awaits each event before producing the next.
private final class CoreAIHTTPEventGate: Sendable {
    private let occupied = Mutex(false)
    func enter() -> Bool {
        occupied.withLock { value in
            guard !value else { return false }
            value = true
            return true
        }
    }
    func leave() { occupied.withLock { $0 = false } }
}

private enum CoreAIHTTPTransportError: Error {
    case closed, concurrentProducer, overlappingSend
}

/// Every mutable property is used only on CoreAIHTTPTransport.network. The
/// connection callbacks run on that same queue; inference sees only the UUID.
private final class CoreAIHTTPClient {
    struct Send {
        let id = UUID()
        let startedAt = DispatchTime.now().uptimeNanoseconds
        let continuation: CheckedContinuation<Void, any Error>?
        let terminal: Bool
    }
    let id = UUID()
    let connection: NWConnection
    let openedAt = DispatchTime.now().uptimeNanoseconds
    let completionID = "chatcmpl-" + UUID().uuidString.lowercased()
    let created = Int(Date().timeIntervalSince1970)
    var parser: QwenHTTPRequestParser
    var parsed = false
    var closed = false
    var responseStarted = false
    var producerStarted = false
    var terminal = false
    var streaming = false
    var cancellation: CoreAIRequestCancellation?
    var pendingSend: Send?
    var bodyBytes = 0
    var textBytes = 0

    init(connection: NWConnection, maxBodyBytes: Int) throws {
        self.connection = connection
        parser = try QwenHTTPRequestParser(maxBodyBytes: maxBodyBytes)
    }
}

/// @unchecked applies only to this queue-confined networking coordinator, never
/// to inference state. All outside access schedules onto `network`; immutable
/// backend calls use its Sendable, internally synchronized service interface.
private final class CoreAIHTTPTransport: @unchecked Sendable {
    let network = DispatchQueue(label: "coreai-runner.http.network", autoreleaseFrequency: .workItem)
    private let configuration: CoreAIHTTPServer.Configuration
    private let backend: any CoreAIServiceBackend
    private let listener: NWListener
    private let created = Int(Date().timeIntervalSince1970)
    private var clients: [UUID: CoreAIHTTPClient] = [:]
    private var signalSources: [DispatchSourceSignal] = []
    private var timer: DispatchSourceTimer?
    private var completion: CheckedContinuation<Void, any Error>?
    private var stopped = false
    private var stopError: (any Error)?
    /// Preserve space inside the total body limit for one terminal SSE event.
    private let terminalReserve = 2048

    init(configuration: CoreAIHTTPServer.Configuration, backend: any CoreAIServiceBackend) throws {
        self.configuration = configuration
        self.backend = backend
        let parameters = NWParameters.tcp
        parameters.requiredLocalEndpoint = .hostPort(host: NWEndpoint.Host(configuration.host),
            port: NWEndpoint.Port(rawValue: configuration.port)!)
        parameters.allowLocalEndpointReuse = true
        listener = try NWListener(using: parameters)
    }

    func start(_ continuation: CheckedContinuation<Void, any Error>) {
        dispatchPrecondition(condition: .onQueue(network))
        if stopped {
            continuation.resume(throwing: stopError ?? CancellationError())
            return
        }
        completion = continuation
        for number in [SIGINT, SIGTERM] {
            let source = DispatchSource.makeSignalSource(signal: number, queue: network)
            source.setEventHandler { [weak self] in self?.stop() }
            source.resume()
            signalSources.append(source)
        }
        let timer = DispatchSource.makeTimerSource(queue: network)
        timer.schedule(deadline: .now() + .milliseconds(100), repeating: .milliseconds(100))
        timer.setEventHandler { [weak self] in self?.checkDeadlines() }
        timer.resume()
        self.timer = timer
        listener.newConnectionHandler = { [weak self] connection in
            guard let self else { connection.cancel(); return }
            self.accept(connection)
        }
        listener.stateUpdateHandler = { [weak self] state in
            if case .failed(let error) = state { self?.stop(error: error) }
        }
        listener.start(queue: network)
    }

    private func accept(_ connection: NWConnection) {
        guard !stopped, clients.count < configuration.maxConnections else {
            // Refuse before allocating a parser, receive buffer or send waiter.
            connection.cancel()
            return
        }
        do {
            let client = try CoreAIHTTPClient(connection: connection, maxBodyBytes: configuration.maxBodyBytes)
            clients[client.id] = client
            let id = client.id
            connection.stateUpdateHandler = { [weak self] state in
                guard let self, let client = self.clients[id] else { return }
                switch state {
                case .ready: self.receive(client)
                case .failed, .cancelled: self.close(client)
                default: break
                }
            }
            connection.start(queue: network)
        } catch {
            connection.cancel()
        }
    }

    private func receive(_ client: CoreAIHTTPClient) {
        guard !client.closed else { return }
        let id = client.id
        client.connection.receive(minimumIncompleteLength: 1, maximumLength: 16_384) { [weak self] data, _, eof, error in
            guard let self, let client = self.clients[id], !client.closed else { return }
            if error != nil { self.close(client); return }
            do {
                if let data, !data.isEmpty, let request = try client.parser.feed(data) {
                    client.parsed = true
                    try self.route(request, client: client)
                }
                if eof {
                    try client.parser.finish()
                    // TCP write half-close is legal after the complete request.
                    // Send failure/state/deadlines detect an unavailable reader.
                    return
                }
                self.receive(client)
            } catch let error as QwenHTTPProtocolError {
                self.reject(client, status: error.statusCode, message: error.message)
            } catch {
                self.reject(client, status: 400, message: "Invalid HTTP request")
            }
        }
    }

    private func route(_ request: QwenHTTPRequest, client: CoreAIHTTPClient) throws {
        switch (request.method, request.path) {
        case ("GET", "/health"):
            // This snapshot must not enter the model's actor/executor.
            try simple(client, status: 200, body: backend.health())
        case ("GET", "/v1/models"):
            try simple(client, status: 200, body: QwenHTTPFrames.models(model: backend.modelID, created: created))
        case ("POST", "/v1/chat/completions"):
            let request = try QwenHTTPChatRequest.decode(request.body, expectedModel: backend.modelID)
            client.streaming = request.stream
            let id = client.id
            let gate = CoreAIHTTPEventGate()
            do {
                client.cancellation = try backend.submit(request) { [weak self] event in
                    guard let self else { throw CoreAIHTTPTransportError.closed }
                    guard gate.enter() else { throw CoreAIHTTPTransportError.concurrentProducer }
                    defer { gate.leave() }
                    try await self.deliver(event, clientID: id)
                }
            } catch let error as QwenHTTPProtocolError {
                reject(client, status: error.statusCode, message: error.message)
            } catch {
                reject(client, status: 500, message: "CoreAI request admission failed")
            }
        case (_, "/health"), (_, "/v1/models"), (_, "/v1/chat/completions"):
            reject(client, status: 405, message: "Method is unsupported for this endpoint")
        default:
            reject(client, status: 404, message: "Endpoint not found")
        }
    }

    private func deliver(_ event: CoreAIServiceEvent, clientID: UUID) async throws {
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, any Error>) in
                network.async {
                    guard let client = self.clients[clientID], !client.closed, !client.terminal else {
                        continuation.resume(throwing: CoreAIHTTPTransportError.closed)
                        return
                    }
                    self.handle(event, client: client, continuation: continuation)
                }
            }
        } onCancel: {
            self.network.async {
                if let client = self.clients[clientID] { self.close(client) }
            }
        }
    }

    private func handle(_ event: CoreAIServiceEvent, client: CoreAIHTTPClient,
                        continuation: CheckedContinuation<Void, any Error>) {
        do {
            switch event {
            case .started:
                guard !client.producerStarted else {
                    throw QwenHTTPProtocolError(statusCode: 500, message: "Duplicate generation admission")
                }
                client.producerStarted = true
                if client.streaming {
                    let role = try QwenHTTPFrames.role(id: client.completionID, created: client.created, model: backend.modelID)
                    try charge(role.count, client: client, terminal: false)
                    var data = QwenHTTPFrames.streamHeader()
                    data.append(role)
                    client.responseStarted = true
                    try send(data, client: client, terminal: false, continuation: continuation)
                } else { continuation.resume() }
            case .heartbeat:
                try requireStarted(client)
                if client.streaming {
                    // The worker emits this between bounded prefill chunks.
                    // Await the ordinary send path so disconnect detection has
                    // backpressure and cannot overlap a token/terminal event.
                    let data = Data(": keep-alive\n\n".utf8)
                    try charge(data.count, client: client, terminal: false)
                    try send(data, client: client, terminal: false, continuation: continuation)
                } else { continuation.resume() }
            case .delta(let text):
                try requireStarted(client)
                let size = text.utf8.count
                guard size <= configuration.maxOutputBytes - client.textBytes else {
                    throw QwenHTTPProtocolError(statusCode: 500, message: "Output exceeds configured byte limit")
                }
                client.textBytes += size
                if client.streaming, !text.isEmpty {
                    let data = try QwenHTTPFrames.content(id: client.completionID, created: client.created,
                        model: backend.modelID, text: text)
                    try charge(data.count, client: client, terminal: false)
                    try send(data, client: client, terminal: false, continuation: continuation)
                } else { continuation.resume() }
            case .completed(let result):
                try requireStarted(client)
                guard result.text.utf8.count <= configuration.maxOutputBytes else {
                    throw QwenHTTPProtocolError(statusCode: 500, message: "Output exceeds configured byte limit")
                }
                let body: Data
                if client.streaming {
                    var data = try QwenHTTPFrames.finish(id: client.completionID, created: client.created,
                        model: backend.modelID, reason: result.finishReason, promptTokens: result.promptTokens,
                        completionTokens: result.completionTokens, cachedTokens: result.cachedTokens)
                    data.append(QwenHTTPFrames.done())
                    body = data
                } else {
                    body = try QwenHTTPFrames.completion(id: client.completionID, created: client.created,
                        model: backend.modelID, text: result.text, reason: result.finishReason,
                        promptTokens: result.promptTokens, completionTokens: result.completionTokens,
                        cachedTokens: result.cachedTokens)
                }
                try charge(body.count, client: client, terminal: true)
                client.terminal = true
                client.responseStarted = true
                let data = client.streaming ? body : try QwenHTTPFrames.response(status: 200,
                    contentType: "application/json; charset=utf-8", body: body)
                try send(data, client: client, terminal: true, continuation: continuation)
            case .failed(let status, let message):
                try failure(client, status: status, message: message, continuation: continuation)
            }
        } catch {
            continuation.resume(throwing: error)
            reject(client, status: (error as? QwenHTTPProtocolError)?.statusCode ?? 500,
                message: (error as? QwenHTTPProtocolError)?.message ?? "Response could not be encoded")
        }
    }

    private func requireStarted(_ client: CoreAIHTTPClient) throws {
        guard client.producerStarted else {
            throw QwenHTTPProtocolError(statusCode: 500, message: "Generation output preceded admission")
        }
    }

    private func charge(_ size: Int, client: CoreAIHTTPClient, terminal: Bool) throws {
        let limit = configuration.maxOutputBytes - (terminal ? 0 : terminalReserve)
        guard client.bodyBytes <= limit, size <= limit - client.bodyBytes else {
            throw QwenHTTPProtocolError(statusCode: 500, message: "Output exceeds configured byte limit")
        }
        client.bodyBytes += size
    }

    private func simple(_ client: CoreAIHTTPClient, status: Int, body: Data) throws {
        try charge(body.count, client: client, terminal: true)
        let data = try QwenHTTPFrames.response(status: status, contentType: "application/json; charset=utf-8", body: body)
        client.responseStarted = true
        client.terminal = true
        try send(data, client: client, terminal: true, continuation: nil)
    }

    private func reject(_ client: CoreAIHTTPClient, status: Int, message: String) {
        guard !client.closed else { return }
        client.cancellation?.cancel()
        // We cannot replace bytes already in flight. Closing also resumes the
        // producer's pending continuation; NWConnection cancellation alone is
        // insufficient because its callback may arrive late.
        guard client.pendingSend == nil, !client.terminal else { close(client); return }
        do { try failure(client, status: status, message: message, continuation: nil) }
        catch { close(client) }
    }

    private func failure(_ client: CoreAIHTTPClient, status: Int, message: String,
                         continuation: CheckedContinuation<Void, any Error>?) throws {
        let status = (400...599).contains(status) ? status : 500
        let code = status == 408 ? "request_timeout" : status == 429 ? "rate_limit_exceeded" :
            status >= 500 ? "generation_failed" : "invalid_request_error"
        let type = status == 429 ? "rate_limit_error" : status >= 500 ? "server_error" : "invalid_request_error"
        let message = String(message.prefix(256))
        let data: Data
        if client.responseStarted {
            guard client.streaming else { throw CoreAIHTTPTransportError.closed }
            var body = try QwenHTTPFrames.sseError(message: message, code: code, type: type)
            body.append(QwenHTTPFrames.done())
            try charge(body.count, client: client, terminal: true)
            data = body
        } else {
            let body = try QwenHTTPFrames.error(message: message, code: code, type: type)
            try charge(body.count, client: client, terminal: true)
            data = try QwenHTTPFrames.response(status: status, contentType: "application/json; charset=utf-8", body: body)
        }
        client.terminal = true
        client.responseStarted = true
        try send(data, client: client, terminal: true, continuation: continuation)
    }

    /// At most one Data payload and one continuation are in flight per client.
    /// No unbounded AsyncStream/event queue or fire-and-forget socket writes.
    private func send(_ data: Data, client: CoreAIHTTPClient, terminal: Bool,
                      continuation: CheckedContinuation<Void, any Error>?) throws {
        guard !client.closed else { throw CoreAIHTTPTransportError.closed }
        guard client.pendingSend == nil else { throw CoreAIHTTPTransportError.overlappingSend }
        let pending = CoreAIHTTPClient.Send(continuation: continuation, terminal: terminal)
        client.pendingSend = pending
        let clientID = client.id
        let sendID = pending.id
        client.connection.send(content: data, completion: .contentProcessed { [weak self] error in
            guard let self, let client = self.clients[clientID],
                  let current = client.pendingSend, current.id == sendID else { return }
            client.pendingSend = nil
            if let error {
                current.continuation?.resume(throwing: error)
                self.close(client)
            } else {
                current.continuation?.resume()
                if current.terminal { self.close(client, cancelRequest: false) }
            }
        })
    }

    private func close(_ client: CoreAIHTTPClient, cancelRequest: Bool = true) {
        guard !client.closed else { return }
        client.closed = true
        if cancelRequest { client.cancellation?.cancel() }
        let pending = client.pendingSend
        client.pendingSend = nil
        pending?.continuation?.resume(throwing: CoreAIHTTPTransportError.closed)
        client.connection.stateUpdateHandler = nil
        client.connection.cancel()
        clients.removeValue(forKey: client.id)
    }

    private func checkDeadlines() {
        let now = DispatchTime.now().uptimeNanoseconds
        for client in Array(clients.values) {
            if let send = client.pendingSend,
               Double(now - send.startedAt) * 1e-9 >= configuration.sendTimeoutSeconds {
                close(client)
                continue
            }
            guard !client.terminal else { continue }
            let age = Double(now - client.openedAt) * 1e-9
            if age >= configuration.requestTimeoutSeconds {
                reject(client, status: 408, message: "Request deadline exceeded")
            } else if !client.parsed && age >= configuration.receiveTimeoutSeconds {
                reject(client, status: 408, message: "Request receive deadline exceeded")
            }
        }
    }

    func stop(error: (any Error)? = nil) {
        dispatchPrecondition(condition: .onQueue(network))
        guard !stopped else { return }
        stopped = true
        stopError = error
        listener.newConnectionHandler = nil
        listener.stateUpdateHandler = nil
        listener.cancel()
        timer?.cancel()
        timer = nil
        for source in signalSources { source.cancel() }
        signalSources.removeAll()
        for client in Array(clients.values) { close(client) }
        backend.shutdown()
        let continuation = completion
        completion = nil
        if let error { continuation?.resume(throwing: error) }
        else { continuation?.resume() }
    }
}
