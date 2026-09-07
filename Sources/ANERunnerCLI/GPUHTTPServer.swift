import ANERunnerCore
import ANERunnerGPU
import CMLX
import Darwin
import Dispatch
import Foundation
@preconcurrency import Network
import Synchronization

extension RunnerCLI {
    static func serveGPU(_ args: Arguments) throws {
        try args.validate(["--model-dir", "--port", "--max-connections", "--max-body-bytes", "--output-buffer-bytes"])
        func number(_ key: String, _ fallback: Int, _ range: ClosedRange<Int>) throws -> Int {
            guard let n = Int(args[key] ?? String(fallback)), range.contains(n) else {
                throw CLIError.usage("\(key) must be in \(range)")
            }
            return n
        }
        let config = GPUHTTPConfiguration(
            modelDirectory: URL(fileURLWithPath: try args.require("--model-dir"))
                .standardizedFileURL.resolvingSymlinksInPath(),
            port: try number("--port", 11236, 1024...65535),
            maxConnections: try number("--max-connections", 8, 1...32),
            maxBodyBytes: try number("--max-body-bytes", 262_144, 1024...1_048_576),
            outputBytes: try number("--output-buffer-bytes", 65_536, 8192...1_048_576))
        try GPUHTTPServer(configuration: config).run()
    }
}

private struct GPUHTTPConfiguration: Sendable {
    let modelDirectory: URL
    let port, maxConnections, maxBodyBytes, outputBytes: Int
    var modelID: String { modelDirectory.lastPathComponent }
}

private struct GPUHTTPWork: Sendable {
    let connectionID: UUID
    let id: String
    let created: Int
    let chat: QwenHTTPChatRequest
    let cancellation: QwenCancellation
    let output: QwenSSEOutputBuffer
}

/// The condition protects only Sendable commands. Model objects never enter it.
private final class GPUHTTPInbox: @unchecked Sendable {
    private let condition = NSCondition()
    private var requests: [GPUHTTPWork] = []
    private var stopping = false
    func offer(_ work: GPUHTTPWork) -> Bool {
        condition.lock(); defer { condition.unlock() }
        guard !stopping, requests.count < 8 else { return false }
        requests.append(work); condition.signal(); return true
    }
    func take(wait: Bool) -> GPUHTTPWork? {
        condition.lock(); defer { condition.unlock() }
        while wait && requests.isEmpty && !stopping { condition.wait() }
        guard !stopping, !requests.isEmpty else { return nil }
        return requests.removeFirst()
    }
    func stop() {
        condition.lock()
        stopping = true
        let discarded = requests; requests.removeAll(keepingCapacity: false)
        condition.broadcast(); condition.unlock()
        for work in discarded { work.cancellation.cancel() }
    }
    var isStopping: Bool { condition.lock(); defer { condition.unlock() }; return stopping }
    var count: Int { condition.lock(); defer { condition.unlock() }; return requests.count }
}

private struct GPUHTTPHealth: Sendable {
    var state = "loading", detail: String?
    var runningJob: String?
    var idle = false, active = 0, jobs = 0
    var prefills = 0, ready = 0, resident = 0, reserved = 0
}

/// Mutable connection state is confined to GPUHTTPServer.network. The only
/// fields passed to the worker are the immutable Sendable GPUHTTPWork value.
private final class GPUHTTPClient: @unchecked Sendable {
    let id: UUID
    let connection: NWConnection
    let openedAt = DispatchTime.now().uptimeNanoseconds
    var parser: QwenHTTPRequestParser
    var parsed = false, closed = false, responseStarted = false
    var headerInFlight = false, headerProcessed = false
    var sendingSimple = false
    var sendStartedAt: UInt64?
    var sendID: UInt64?
    var rejectionStatus: Int?
    var rejectionCode: String?
    var work: GPUHTTPWork?
    init(_ connection: NWConnection, id: UUID, maxBodyBytes: Int) throws {
        self.id = id; self.connection = connection
        parser = try QwenHTTPRequestParser(maxBodyBytes: maxBodyBytes)
    }
}

/// Owns CPU text state only and is used exclusively on the fixed inference thread.
private final class GPUHTTPActive {
    let work: GPUHTTPWork
    var utf8 = IncrementalUTF8Decoder()
    var text = ""
    var textBudget: QwenHTTPTextBudget
    var schedulerID: UUID?
    init(_ work: GPUHTTPWork, maxTextBytes: Int) {
        self.work = work; textBudget = QwenHTTPTextBudget(maxBytes: maxTextBytes)
    }
}

private enum GPUHTTPOutputError: Error { case tooLarge }

private enum GPUHTTPCloseReason: String {
    case connectionLimit = "connection_limit", serverStopping = "server_stopping"
    case clientInitializationFailed = "client_initialization_failed"
    case transportFailed = "transport_failed", transportCancelled = "transport_cancelled"
    case receiveFailed = "receive_failed", responseConflict = "response_conflict"
    case simpleSent = "simple_sent", simpleSendFailed = "simple_send_failed"
    case rejectedAfterResponse = "rejected_after_response", errorEncodingFailed = "error_encoding_failed"
    case headerSendFailed = "header_send_failed", sendFailed = "send_failed", terminalSent = "terminal_sent"
    case sendDeadline = "send_deadline", connectionDeadline = "connection_deadline"
    case shutdown, outputEncodingFailed = "output_encoding_failed"
}

private enum GPUHTTPLogEvent: String {
    case modelTerminal = "model_terminal", outputTerminal = "output_terminal", connectionClose = "connection_close"
    case closedSendReleased = "closed_send_released"
}

/// Network objects are queue-confined; inbox/health are synchronized. The worker
/// creates and destroys all non-Sendable inference objects as thread-local vars.
/// No model/generator/scheduler is stored in this unchecked-Sendable coordinator.
private final class GPUHTTPServer: @unchecked Sendable {
    let configuration: GPUHTTPConfiguration
    private let network = DispatchQueue(label: "ane-runner.http.network", autoreleaseFrequency: .workItem)
    private let inbox = GPUHTTPInbox()
    private let health = Mutex(GPUHTTPHealth())
    private let finished = DispatchSemaphore(value: 0)
    private let failure = Mutex<String?>(nil)
    private let logLock = Mutex(())
    private let listener: NWListener
    private var clients: [UUID: GPUHTTPClient] = [:]
    private var signals: [DispatchSourceSignal] = []
    private var timer: DispatchSourceTimer?
    private var stopping = false, workerStarted = false, completionSignalled = false
    private let created = Int(Date().timeIntervalSince1970)

    init(configuration: GPUHTTPConfiguration) throws {
        self.configuration = configuration
        let parameters = NWParameters.tcp
        parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1",
            port: NWEndpoint.Port(rawValue: UInt16(configuration.port))!)
        parameters.allowLocalEndpointReuse = true
        listener = try NWListener(using: parameters)
    }

    func run() throws {
        let previousINT = Darwin.signal(SIGINT, SIG_IGN)
        let previousTERM = Darwin.signal(SIGTERM, SIG_IGN)
        defer { Darwin.signal(SIGINT, previousINT); Darwin.signal(SIGTERM, previousTERM) }
        network.async { self.startNetwork() }
        finished.wait()
        if let message = failure.withLock({ $0 }) { throw CLIError.usage(message) }
    }

    private func startNetwork() {
        dispatchPrecondition(condition: .onQueue(network))
        for number in [SIGINT, SIGTERM] {
            let signal = DispatchSource.makeSignalSource(signal: number, queue: network)
            signal.setEventHandler { [weak self] in self?.shutdown() }
            signal.resume(); signals.append(signal)
        }
        let timer = DispatchSource.makeTimerSource(queue: network)
        timer.schedule(deadline: .now() + .seconds(1), repeating: .seconds(1))
        timer.setEventHandler { [weak self] in self?.checkDeadlines() }
        timer.resume(); self.timer = timer
        listener.newConnectionHandler = { [weak self] connection in self?.accept(connection) }
        listener.stateUpdateHandler = { [weak self] state in
            guard let self else { return }
            switch state {
            case .ready:
                guard !self.stopping, !self.workerStarted else { return }
                self.workerStarted = true
                self.log("experimental HTTP pid=\(getpid()) address=127.0.0.1:\(self.configuration.port) model=\(self.configuration.modelID) state=loading connections=\(self.configuration.maxConnections) body_bytes=\(self.configuration.maxBodyBytes) output_bytes=\(self.configuration.outputBytes)")
                let thread = Thread { self.inferenceMain() }
                thread.name = "ane-runner.http.inference"
                thread.start()
            case .failed: self.shutdown(error: "HTTP listener failed")
            default: break
            }
        }
        listener.start(queue: network)
    }

    private func accept(_ connection: NWConnection) {
        let id = UUID()
        guard !stopping, clients.count < configuration.maxConnections else {
            // Admission before receive prevents unbounded parsers/send callbacks.
            logLifecycle(.connectionClose, connectionID: id,
                reason: (stopping ? GPUHTTPCloseReason.serverStopping : .connectionLimit).rawValue)
            connection.cancel(); return
        }
        do {
            let client = try GPUHTTPClient(connection, id: id, maxBodyBytes: configuration.maxBodyBytes)
            clients[client.id] = client
            connection.stateUpdateHandler = { [weak self, weak client] state in
                guard let self, let client else { return }
                switch state {
                case .ready: self.receive(client)
                case .failed: self.close(client, reason: .transportFailed)
                case .cancelled: self.close(client, reason: .transportCancelled)
                default: break
                }
            }
            connection.start(queue: network)
        } catch {
            logLifecycle(.connectionClose, connectionID: id, reason: GPUHTTPCloseReason.clientInitializationFailed.rawValue)
            connection.cancel()
        }
    }

    private func receive(_ client: GPUHTTPClient) {
        guard !client.closed else { return }
        client.connection.receive(minimumIncompleteLength: 1, maximumLength: 16_384) { [weak self, weak client] data, _, eof, error in
            guard let self, let client, !client.closed else { return }
            if error != nil { self.close(client, reason: .receiveFailed); return }
            do {
                if let data, !data.isEmpty, let request = try client.parser.feed(data) {
                    client.parsed = true
                    try self.route(request, client: client)
                }
                if eof {
                    try client.parser.finish()
                    // EOF after a complete request can be a legal TCP write
                    // half-close. Failed sends/state or deadlines detect a dead
                    // reader; EOF alone is not proof the peer cannot receive.
                    return
                }
                if !client.closed { self.receive(client) }
            } catch let error as QwenHTTPProtocolError {
                self.reject(client, status: error.statusCode, message: error.message, code: "invalid_request_error")
            } catch {
                self.reject(client, status: 400, message: "Invalid HTTP request", code: "invalid_request_error")
            }
        }
    }

    private func route(_ request: QwenHTTPRequest, client: GPUHTTPClient) throws {
        switch (request.method, request.path) {
        case ("GET", "/health"):
            let h = health.withLock { $0 }
            let body = try JSONSerialization.data(withJSONObject: [
                "status": h.state, "ready": h.state == "ready", "model": configuration.modelID,
                "pid": Int(getpid()), "experimental": true,
                "idle": h.idle && inbox.count == 0, "active": h.active, "active_jobs": h.jobs,
                "queued_prefills": h.prefills, "ready_decodes": h.ready,
                "resident_sequences": h.resident, "reserved_tokens": h.reserved,
                "pending_requests": inbox.count, "connections": clients.count,
                "running_job": h.runningJob as Any? ?? NSNull(),
                "running_job_known": h.active == 0 || h.runningJob != nil,
                "detail": h.detail.map { String($0.prefix(512)) } as Any? ?? NSNull()
            ])
            simple(client, data: try QwenHTTPFrames.response(status: h.state == "ready" ? 200 : 503,
                contentType: "application/json", body: body))
        case ("GET", "/v1/models"):
            simple(client, data: try QwenHTTPFrames.response(status: 200, contentType: "application/json",
                body: QwenHTTPFrames.models(model: configuration.modelID, created: created)))
        case ("POST", "/v1/chat/completions"):
            let chat = try QwenHTTPChatRequest.decode(request.body, expectedModel: configuration.modelID)
            guard health.withLock({ $0.state == "ready" }) else {
                reject(client, status: 503, message: "Model is not ready", code: "model_unavailable"); return
            }
            let overflow: Data
            let limits: QwenSSEOutputBuffer.Limits
            if chat.stream {
                overflow = try QwenHTTPFrames.sseError(message: "Client output buffer limit exceeded", code: "slow_consumer", type: "server_error") + QwenHTTPFrames.done()
                limits = .init(maxBytes: configuration.outputBytes, maxEvents: 256, terminalReserveBytes: 4096)
            } else {
                overflow = try QwenHTTPFrames.response(status: 500, contentType: "application/json",
                    body: QwenHTTPFrames.error(message: "Output exceeds configured byte limit", code: "output_limit", type: "server_error"))
                // Nonstream emits one bounded final HTTP response, with no
                // intermediate frame queue or separate unbounded send path.
                limits = .init(maxBytes: configuration.outputBytes + 1, maxEvents: 2,
                    terminalReserveBytes: configuration.outputBytes)
            }
            let work = GPUHTTPWork(connectionID: client.id, id: "chatcmpl-" + UUID().uuidString.lowercased(),
                created: Int(Date().timeIntervalSince1970), chat: chat, cancellation: QwenCancellation(),
                output: try QwenSSEOutputBuffer(limits: limits, overflowFrame: overflow))
            client.work = work
            guard inbox.offer(work) else {
                reject(client, status: 429, message: "Pending request queue is full", code: "queue_full"); return
            }
        default:
            reject(client, status: ["/health", "/v1/models", "/v1/chat/completions"].contains(request.path) ? 405 : 404,
                message: "Unsupported endpoint or method", code: "invalid_request_error")
        }
    }

    private func simple(_ client: GPUHTTPClient, data: Data) {
        guard !client.closed, !client.sendingSimple, !client.responseStarted else { close(client, reason: .responseConflict); return }
        client.sendingSimple = true; client.responseStarted = true
        client.sendStartedAt = DispatchTime.now().uptimeNanoseconds
        client.connection.send(content: data, completion: .contentProcessed { [weak self, weak client] error in
            guard let self, let client else { return }
            self.close(client, reason: error == nil ? .simpleSent : .simpleSendFailed)
        })
    }

    private func reject(_ client: GPUHTTPClient, status: Int, message: String, code: String) {
        client.rejectionStatus = status; client.rejectionCode = code
        if let work = client.work {
            work.cancellation.cancel()
            _ = work.output.disconnect()
        }
        guard !client.responseStarted else { close(client, reason: .rejectedAfterResponse); return }
        do {
            simple(client, data: try QwenHTTPFrames.response(status: status, contentType: "application/json",
                body: QwenHTTPFrames.error(message: String(message.prefix(512)), code: code,
                    type: status >= 500 ? "server_error" : (status == 429 ? "rate_limit_error" : "invalid_request_error"))))
        } catch { close(client, reason: .errorEncodingFailed) }
    }

    /// Called only after scheduler admission, before role/content wakeups.
    private func admitted(_ work: GPUHTTPWork) {
        network.async {
            guard let client = self.clients[work.connectionID], !client.closed else { work.cancellation.cancel(); return }
            if !work.chat.stream { return }
            guard !client.responseStarted else { return }
            client.responseStarted = true; client.headerInFlight = true
            client.sendStartedAt = DispatchTime.now().uptimeNanoseconds
            client.connection.send(content: QwenHTTPFrames.streamHeader(), completion: .contentProcessed { [weak self, weak client] error in
                guard let self, let client, !client.closed else { return }
                if error != nil { self.close(client, reason: .headerSendFailed); return }
                client.headerInFlight = false; client.sendStartedAt = nil
                client.headerProcessed = true
                self.sendNext(client)
            })
        }
    }

    private func actions(_ value: QwenSSEOutputBuffer.Actions, work: GPUHTTPWork) {
        if value.cancelProducer { work.cancellation.cancel() }
        if value.scheduleSend {
            network.async { if let client = self.clients[work.connectionID] { self.sendNext(client) } }
        }
    }

    private func sendNext(_ client: GPUHTTPClient) {
        guard !client.closed, !client.sendingSimple, let work = client.work,
              !work.chat.stream || client.headerProcessed,
              let send = work.output.beginSend() else { return }
        client.responseStarted = true
        client.sendID = send.id; client.sendStartedAt = DispatchTime.now().uptimeNanoseconds
        client.connection.send(content: send.data, completion: .contentProcessed { [weak self, client] error in
            let next = work.output.acknowledgeSend(send.id, succeeded: error == nil)
            if next.cancelProducer { work.cancellation.cancel() }
            if let self, client.closed {
                // At most one outstanding send remains after close. Its actual
                // callback releases retained Data; close itself cannot do that.
                self.logLifecycle(.closedSendReleased, connectionID: client.id, work: work,
                    reason: error == nil ? "send_processed" : "send_failed", fields: ["lease_id": send.id])
            }
            guard let self, !client.closed, client.sendID == send.id else { return }
            if error != nil { self.close(client, reason: .sendFailed); return }
            if send.isTerminal { self.close(client, reason: .terminalSent); return }
            client.sendID = nil; client.sendStartedAt = nil
            if next.scheduleSend { self.sendNext(client) }
        })
    }

    private func close(_ client: GPUHTTPClient, reason: GPUHTTPCloseReason) {
        guard !client.closed else { return }
        client.closed = true
        let now = DispatchTime.now().uptimeNanoseconds
        let priorOutcome = client.work?.output.snapshot().outcome?.rawValue
        if let work = client.work {
            let effect = work.output.disconnect()
            if effect.cancelProducer { work.cancellation.cancel() }
        }
        logLifecycle(.connectionClose, connectionID: client.id, work: client.work, reason: reason.rawValue,
            fields: ["connection_age_seconds": Double(now - client.openedAt) * 1e-9,
                     "send_elapsed_seconds": client.sendStartedAt.map { Double(now - $0) * 1e-9 } as Any? ?? NSNull(),
                     "lease_id": client.sendID as Any? ?? NSNull(),
                     "send_kind": client.headerInFlight ? "header" : (client.sendingSimple ? "simple" : (client.sendID == nil ? "none" : "body")),
                     "rejection_status": client.rejectionStatus as Any? ?? NSNull(),
                     "rejection_code": client.rejectionCode as Any? ?? NSNull(),
                     "output_outcome_before_close": priorOutcome as Any? ?? NSNull()])
        client.connection.stateUpdateHandler = nil
        client.connection.cancel()
        clients.removeValue(forKey: client.id)
    }

    private func checkDeadlines() {
        let now = DispatchTime.now().uptimeNanoseconds
        for client in Array(clients.values) {
            let age = Double(now - client.openedAt) * 1e-9
            // One periodic observer reads the current lease's start, rather
            // than scheduling uncancellable timers for past sends/requests.
            if let start = client.sendStartedAt, Double(now - start) * 1e-9 >= 15 { close(client, reason: .sendDeadline) }
            else if age >= 300 { close(client, reason: .connectionDeadline) }
            else if !client.parsed && age >= 15 {
                reject(client, status: 408, message: "Request receive deadline exceeded", code: "request_timeout")
            }
        }
    }

    private func shutdown(error: String? = nil) {
        guard !stopping else { return }
        stopping = true
        if let error { failure.withLock { $0 = error } }
        health.withLock { $0.state = error == nil ? "stopping" : "failed"; $0.detail = error; $0.idle = false }
        listener.newConnectionHandler = nil; listener.cancel()
        timer?.cancel(); timer = nil
        for client in Array(clients.values) { close(client, reason: .shutdown) }
        inbox.stop()
        if !workerStarted { inferenceStopped() }
    }

    private func inferenceStopped(error: String? = nil) {
        if let error { failure.withLock { $0 = error }; log("HTTP inference failed") }
        shutdown(error: error)
        for signal in signals { signal.cancel() }; signals.removeAll()
        listener.stateUpdateHandler = nil
        health.withLock { $0.state = error == nil ? "stopping" : "failed"; $0.active = 0; $0.jobs = 0; $0.idle = true }
        guard !completionSignalled else { return }
        completionSignalled = true; finished.signal()
    }

    private func inferenceMain() {
        var message: String?
        do { try autoreleasepool { try inferenceLoop() } }
        catch { message = "HTTP inference failed" }
        // All local MLX/model values were released on this OS thread above.
        let result = message
        network.async { self.inferenceStopped(error: result) }
    }

    private func inferenceLoop() throws {
        let tokenizer = try QwenTokenizer(modelDirectory: configuration.modelDirectory)
        if inbox.isStopping { return }
        var previousCache = 0
        try MX.check(mlx_set_cache_limit(&previousCache, 256 * 1024 * 1024), "HTTP allocator cache")
        defer { var ignored = 0; _ = mlx_set_cache_limit(&ignored, previousCache) }
        let model = try QwenModel(modelDirectory: configuration.modelDirectory,
            reservedOutputIDs: tokenizer.reservedOutputTokenIDs) { count, total in
                if count % 8 == 0 || count == total { self.log("HTTP model loaded \(count)/\(total)") }
            }
        let generator = try QwenGenerator(model: model)
        let scheduler = try QwenLocalScheduler(generator: generator, limits: .init(executionMode: .cooperative))
        var jobs: [UUID: GPUHTTPActive] = [:]
        defer {
            for job in jobs.values { job.work.cancellation.cancel() }
            _ = try? scheduler.discardAll()
            jobs.removeAll()
            try? MX.synchronize()
        }
        health.withLock { $0.state = inbox.isStopping ? "stopping" : "ready"; $0.idle = true }
        log("HTTP model state=ready default=AR experimental_mtp_depth=2 context=16384 chunk=416")
        func snapshot(active: Int = 0) {
            let s = scheduler.snapshot()
            health.withLock {
                $0.active = active; $0.jobs = jobs.count; $0.prefills = s.queuedPrefills
                $0.runningJob = s.runningJob?.uuidString
                $0.ready = s.readyDecodes; $0.resident = s.residentSequences; $0.reserved = s.reservedTokens
                $0.idle = s.isIdle && active == 0
                if !s.acceptingJobs { $0.state = "failed"; $0.detail = s.unavailableReason }
            }
        }
        while !inbox.isStopping {
            // Drain transient Foundation/Objective-C objects after each bounded
            // admission + inference slice, rather than only when the service exits.
            let keepRunning = try autoreleasepool { () throws -> Bool in
            // At most one bounded tokenization/admission between GPU slices.
            if let work = inbox.take(wait: scheduler.snapshot().isIdle) {
                snapshot(active: 1)
                do {
                    try work.cancellation.check()
                    let messages = work.chat.messages.map { ChatMessage(role: $0.role, content: $0.content) }
                    let tokens = try tokenizer.encode(tokenizer.renderChat(messages: messages))
                    try work.cancellation.check()
                    let request = QwenGenerationRequest(tokens: tokens, maxTokens: work.chat.maxTokens,
                        contextLimit: 16_384, prefillChunk: 416, mtpDepth: work.chat.mtpDepth,
                        verification: work.chat.mtpDepth == 2 ? .batchedScalarLinear : .scalar,
                        draftHistoryTokens: work.chat.mtpDepth == 2 ? 1024 : nil)
                    let active = GPUHTTPActive(work, maxTextBytes: configuration.outputBytes - 2048)
                    let id = try scheduler.submit(request, cancellation: work.cancellation) { token in
                        self.health.withLock { $0.runningJob = active.schedulerID?.uuidString }
                        let text = active.utf8.append(try tokenizer.decodeBytes([token], skipSpecialTokens: true))
                        try self.publish(text, active: active)
                    }
                    active.schedulerID = id; jobs[id] = active
                    admitted(work)
                    if work.chat.stream {
                        let offered = work.output.enqueue(try QwenHTTPFrames.role(id: work.id, created: work.created, model: work.chat.model))
                        if offered.status == .overflow { logOutput(work, reason: "slow_consumer") }
                        actions(offered.actions, work: work)
                        if offered.status != .accepted { work.cancellation.cancel() }
                    }
                } catch {
                    work.cancellation.cancel()
                    let status: Int
                    switch error {
                    case QwenLocalScheduler.Error.queueFull, QwenLocalScheduler.Error.overBudget: status = 429
                    case QwenLocalScheduler.Error.closed(_), QwenGenerationError.unavailable(_): status = 503
                    default: status = 400
                    }
                    let text = String(error.localizedDescription.prefix(512))
                    network.async {
                        if let client = self.clients[work.connectionID] {
                            self.reject(client, status: status, message: text, code: status == 429 ? "queue_full" : "invalid_request_error")
                        }
                    }
                }
            }
            if inbox.isStopping { return false }
            snapshot(active: scheduler.snapshot().isIdle ? 0 : 1)
            if let event = try scheduler.runNext(), [.completed, .failed, .cancelled].contains(event.kind),
               let active = jobs.removeValue(forKey: event.jobID) {
                complete(event, active: active)
            }
            snapshot()
            return true
            }
            if !keepRunning { break }
        }
    }

    private func publish(_ text: String, active: GPUHTTPActive) throws {
        guard !text.isEmpty else { return }
        let work = active.work
        try work.cancellation.check()
        if work.chat.stream {
            let frame: Data
            do { frame = try QwenHTTPFrames.content(id: work.id, created: work.created, model: work.chat.model, text: text) }
            catch { active.textBudget.record(.encodingFailed); throw error }
            let offered = work.output.enqueue(frame)
            if offered.status == .overflow { logOutput(work, reason: "slow_consumer") }
            actions(offered.actions, work: work)
            guard offered.status == .accepted else { work.cancellation.cancel(); throw QwenGenerationError.cancelled }
        } else {
            // Leave JSON/header headroom; check the actual final encoding too.
            guard active.textBudget.accept(byteCount: text.utf8.count) else { throw GPUHTTPOutputError.tooLarge }
            active.text += text
        }
    }

    private func complete(_ event: QwenLocalScheduler.Event, active: GPUHTTPActive) {
        let work = active.work
        var modelFields: [String: Any] = ["model_kind": event.kind.rawValue, "stage": event.stage.rawValue,
            "scheduler_elapsed_seconds": finite(event.timing.elapsedSeconds)]
        if let result = event.result {
            modelFields["model_finish_reason"] = result.finishReason.rawValue
            modelFields["prompt_tokens"] = result.statistics.promptTokenCount
            modelFields["completion_tokens"] = result.tokens.count
            modelFields["prefill_seconds"] = (result.phases?.prefill.targetSeconds).map(finite) ?? NSNull()
            modelFields["decode_seconds"] = finite(result.decodeSeconds)
        }
        logLifecycle(.modelTerminal, connectionID: work.connectionID, work: work,
            reason: active.textBudget.failure?.rawValue, fields: modelFields)
        do {
            let completion: QwenSSEOutputBuffer.Completion
            let outputReason: String
            let frame: Data
            if event.kind == .completed, let result = event.result {
                log("HTTP request id=\(work.id) mtp_depth=\(work.chat.mtpDepth) finish=\(result.finishReason.rawValue) prompt_tokens=\(result.statistics.promptTokenCount) completion_tokens=\(result.tokens.count) prefill_seconds=\(result.phases?.prefill.targetSeconds ?? 0) decode_seconds=\(result.decodeSeconds) scheduler_elapsed_seconds=\(event.timing.elapsedSeconds) event=model_terminal")
                try publish(active.utf8.finish(), active: active)
                let reason = result.finishReason == .eos ? "stop" : "length"
                if work.chat.stream {
                    frame = try QwenHTTPFrames.finish(id: work.id, created: work.created, model: work.chat.model,
                        reason: reason, promptTokens: result.statistics.promptTokenCount, completionTokens: result.tokens.count)
                        + QwenHTTPFrames.done()
                } else {
                    let body = try QwenHTTPFrames.completion(id: work.id, created: work.created, model: work.chat.model,
                        text: active.text, reason: reason, promptTokens: result.statistics.promptTokenCount, completionTokens: result.tokens.count)
                    frame = try QwenHTTPFrames.response(status: 200, contentType: "application/json", body: body)
                }
                completion = .completed
                outputReason = "completed"
            } else {
                log("HTTP request id=\(work.id) terminal=\(event.kind.rawValue) stage=\(event.stage.rawValue) scheduler_elapsed_seconds=\(event.timing.elapsedSeconds) event=model_terminal")
                completion = event.kind == .cancelled ? .cancelled : .failed
                let failure = active.textBudget.errorResponse(cancelled: event.kind == .cancelled)
                frame = try failureFrame(work, message: failure.message, code: failure.code)
                outputReason = active.textBudget.failure?.rawValue ?? failure.code
            }
            let finished = work.output.finish(completion, frame: frame)
            if finished.status == .invalidFrame {
                active.textBudget.record(.responseLimit)
                throw GPUHTTPOutputError.tooLarge
            }
            if finished.status == .accepted { logOutput(work, reason: outputReason, budget: active.textBudget) }
            actions(finished.actions, work: work)
        } catch {
            work.cancellation.cancel()
            let cancelled = (error as? QwenGenerationError) == .cancelled
            if !cancelled && active.textBudget.failure == nil { active.textBudget.record(.encodingFailed) }
            let failure = active.textBudget.errorResponse(cancelled: cancelled)
            if let frame = try? failureFrame(work, message: failure.message, code: failure.code) {
                let finished = work.output.finish(cancelled ? .cancelled : .failed, frame: frame)
                if finished.status == .accepted {
                    logOutput(work, reason: active.textBudget.failure?.rawValue ?? failure.code, budget: active.textBudget)
                }
                actions(finished.actions, work: work)
                if finished.status == .invalidFrame {
                    network.async { if let client = self.clients[work.connectionID] { self.close(client, reason: .outputEncodingFailed) } }
                }
            } else {
                network.async { if let client = self.clients[work.connectionID] { self.close(client, reason: .outputEncodingFailed) } }
            }
        }
    }

    private func failureFrame(_ work: GPUHTTPWork, message: String, code: String) throws -> Data {
        if work.chat.stream { return try QwenHTTPFrames.sseError(message: message, code: code, type: "server_error") + QwenHTTPFrames.done() }
        return try QwenHTTPFrames.response(status: 500, contentType: "application/json",
            body: QwenHTTPFrames.error(message: message, code: code, type: "server_error"))
    }

    private func finite(_ value: Double) -> Any { value.isFinite ? value as Any : NSNull() }

    private func logOutput(_ work: GPUHTTPWork, reason: String, budget: QwenHTTPTextBudget? = nil) {
        // Only the operation that selected the buffer terminal logs this event.
        // A late scheduler finish returns alreadyTerminal and never rewrites it.
        let code = budget?.failure?.code ?? (reason == "completed" ? nil : reason)
        // Streaming retains local encoding failures but never charges this
        // cumulative text budget. These fields are inapplicable, not zero.
        let textBudget = work.chat.stream ? nil : budget
        logLifecycle(.outputTerminal, connectionID: work.connectionID, work: work, reason: reason,
            fields: ["text_bytes": textBudget?.acceptedBytes as Any? ?? NSNull(),
                     "text_limit_bytes": textBudget?.maxBytes as Any? ?? NSNull(),
                     "error_code": code as Any? ?? NSNull()])
    }

    private func logLifecycle(_ event: GPUHTTPLogEvent, connectionID: UUID, work: GPUHTTPWork? = nil,
                              reason: String? = nil, fields: [String: Any] = [:]) {
        // Call sites supply finite reason codes and numeric metadata only.
        // Never pass prompts, content, token IDs, or an Error's description.
        var record = fields
        record["schema"] = "qwen-http-lifecycle-v1"
        record["pid"] = Int(getpid())
        record["event"] = event.rawValue
        record["connection_id"] = connectionID.uuidString.lowercased()
        record["request_id"] = work?.id as Any? ?? NSNull()
        record["reason"] = reason as Any? ?? NSNull()
        record["uptime_seconds"] = Double(DispatchTime.now().uptimeNanoseconds) * 1e-9
        if let work {
            let state = work.output.snapshot()
            record["stream"] = work.chat.stream
            record["mtp_depth"] = work.chat.mtpDepth
            record["output_outcome"] = state.outcome?.rawValue as Any? ?? NSNull()
            record["buffered_bytes"] = state.bufferedBytes
            record["buffered_events"] = state.bufferedEvents
            record["queued_events"] = state.queuedEvents
            record["in_flight_bytes"] = state.inFlightBytes
            record["has_in_flight"] = state.hasInFlight
            record["producer_finished"] = state.producerFinished
            record["transport_closed"] = state.transportClosed
            record["cancellation_requested"] = state.cancellationRequested
            record["output_drained"] = state.isDrained
        }
        guard let data = try? JSONSerialization.data(withJSONObject: record, options: .sortedKeys),
              let line = String(data: data, encoding: .utf8) else {
            log("HTTP lifecycle record encoding failed")
            return
        }
        log(line)
    }

    private func log(_ text: String) {
        logLock.withLock { _ in FileHandle.standardError.write(Data((text + "\n").utf8)) }
    }
}
