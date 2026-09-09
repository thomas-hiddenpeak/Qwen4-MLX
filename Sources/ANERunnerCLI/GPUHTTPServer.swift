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
        // serve-gpu is one command in its own CLI process. Darwin pipe EPIPE
        // can signal the whole process, so this explicit service policy lasts
        // until process exit. It does not alter the parent process or any fd's
        // shared flags. Core's stderr logger verifies this precondition.
        var pipePolicy = sigaction()
        pipePolicy.__sigaction_u.__sa_handler = SIG_IGN
        sigemptyset(&pipePolicy.sa_mask)
        guard sigaction(SIGPIPE, &pipePolicy, nil) == 0 else {
            throw CLIError.usage("Cannot install HTTP SIGPIPE policy")
        }
        try args.validate(["--model-dir", "--port", "--max-connections", "--max-body-bytes", "--output-buffer-bytes",
            "--prefix-cache-bytes", "--prefix-cache-entries", "--prefix-cache-directory",
            "--prefix-cache-disk-bytes", "--prefix-cache-disk-entries", "--prefix-cache-ttl-seconds", "--state-budget-bytes",
            "--prefix-cache-min-free-bytes", "--prefix-cache-restore-timeout-seconds",
            "--prefix-cache-shutdown-timeout-seconds", "--kv-append-mode"])
        func number(_ key: String, _ fallback: Int, _ range: ClosedRange<Int>) throws -> Int {
            guard let n = Int(args[key] ?? String(fallback)), range.contains(n) else {
                throw CLIError.usage("\(key) must be in \(range)")
            }
            return n
        }
        guard let kvAppendMode = GPUAttention.KVAppendMode(rawValue: args["--kv-append-mode"] ?? "reference") else {
            throw CLIError.usage("--kv-append-mode must be reference or capacity256")
        }
        let prefixCacheBytes = try number("--prefix-cache-bytes", 536_870_912, 0...8_589_934_592)
        let prefixCacheDirectory: URL?
        if args["--prefix-cache-directory"] != nil {
            guard prefixCacheBytes > 0 else {
                throw CLIError.usage("--prefix-cache-directory requires --prefix-cache-bytes greater than zero")
            }
            prefixCacheDirectory = URL(fileURLWithPath: try args.require("--prefix-cache-directory"))
                .standardizedFileURL.resolvingSymlinksInPath()
        } else {
            guard args["--prefix-cache-disk-bytes"] == nil, args["--prefix-cache-disk-entries"] == nil,
                  args["--prefix-cache-min-free-bytes"] == nil, args["--prefix-cache-restore-timeout-seconds"] == nil,
                  args["--prefix-cache-shutdown-timeout-seconds"] == nil else {
                throw CLIError.usage("SSD cache limits require --prefix-cache-directory")
            }
            prefixCacheDirectory = nil
        }
        let config = GPUHTTPConfiguration(
            modelDirectory: URL(fileURLWithPath: try args.require("--model-dir"))
                .standardizedFileURL.resolvingSymlinksInPath(),
            port: try number("--port", 11236, 1024...65535),
            maxConnections: try number("--max-connections", 8, 1...32),
            maxBodyBytes: try number("--max-body-bytes", 262_144, 1024...1_048_576),
            outputBytes: try number("--output-buffer-bytes", 65_536, 8192...1_048_576),
            prefixCacheBytes: prefixCacheBytes,
            prefixCacheEntries: try number("--prefix-cache-entries", 8, 1...256),
            prefixCacheDirectory: prefixCacheDirectory,
            prefixCacheDiskBytes: try number("--prefix-cache-disk-bytes", 8_589_934_592, 1...Int.max),
            prefixCacheDiskEntries: try number("--prefix-cache-disk-entries", 32, 1...4096),
            prefixCacheTTLSeconds: try number("--prefix-cache-ttl-seconds", 86_400, 1...Int.max),
            prefixCacheMinFreeBytes: try number("--prefix-cache-min-free-bytes", 1_073_741_824, 0...Int.max),
            prefixCacheRestoreTimeoutSeconds: try number("--prefix-cache-restore-timeout-seconds", 5, 1...300),
            prefixCacheShutdownTimeoutSeconds: try number("--prefix-cache-shutdown-timeout-seconds", 30, 1...300),
            stateBudgetBytes: try number("--state-budget-bytes", 4_294_967_296, 1...Int.max),
            kvAppendMode: kvAppendMode)
        let server = try GPUHTTPServer(configuration: config)
        // A running service has its own bounded logger. Do not re-enter the
        // general CLI catch's synchronous stderr write after a sink failure.
        guard server.run() else { Darwin.exit(EXIT_FAILURE) }
    }
}

private struct GPUHTTPConfiguration: Sendable {
    let modelDirectory: URL
    let port, maxConnections, maxBodyBytes, outputBytes, prefixCacheBytes, prefixCacheEntries: Int
    let prefixCacheDirectory: URL?
    let prefixCacheDiskBytes, prefixCacheDiskEntries, prefixCacheTTLSeconds: Int
    let prefixCacheMinFreeBytes, prefixCacheRestoreTimeoutSeconds: Int
    let prefixCacheShutdownTimeoutSeconds: Int
    let stateBudgetBytes: Int
    let kvAppendMode: GPUAttention.KVAppendMode
    var prefixDiskLimits: QwenPrefixDiskLimits {
        .init(maxEntries: prefixCacheDiskEntries, maxBytes: prefixCacheDiskBytes,
              minAvailableBytes: prefixCacheMinFreeBytes)
    }
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
    func take(wait: Bool, timeout: TimeInterval? = nil) -> GPUHTTPWork? {
        condition.lock(); defer { condition.unlock() }
        let deadline = timeout.map { Date().addingTimeInterval($0) }
        while wait && requests.isEmpty && !stopping {
            if let deadline {
                if !condition.wait(until: deadline) { break }
            } else { condition.wait() }
        }
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
    var prefills = 0, ready = 0, resident = 0, reserved = 0, waitingPrefix = 0
    var prefixCacheJSON: Data?, prefixDiskJSON: Data?, stateBudgetJSON: Data?
    var mlxMemory: [String: Int]?
    var pressureMonitorRunning = false
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
    var toolParser: QwenToolStreamParser?
    var invalidToolCall = false
    init(_ work: GPUHTTPWork, maxTextBytes: Int) {
        self.work = work; textBudget = QwenHTTPTextBudget(maxBytes: maxTextBytes)
        if work.chat.parsesTools {
            toolParser = QwenToolStreamParser(tools: work.chat.activeTools,
                idPrefix: "call_" + work.id, maxBytes: maxTextBytes)
        }
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
    private let memoryPressure: QwenMemoryPressurePolicy
    private let finished = DispatchSemaphore(value: 0)
    private let failure = Mutex<String?>(nil)
    private let logger: QwenHTTPLogger
    private let listener: NWListener
    private var clients: [UUID: GPUHTTPClient] = [:]
    private var signals: [DispatchSourceSignal] = []
    private var timer: DispatchSourceTimer?
    private var stopping = false, workerStarted = false, completionSignalled = false
    private let created = Int(Date().timeIntervalSince1970)

    init(configuration: GPUHTTPConfiguration) throws {
        self.configuration = configuration
        memoryPressure = try QwenMemoryPressurePolicy()
        let parameters = NWParameters.tcp
        parameters.requiredLocalEndpoint = .hostPort(host: "127.0.0.1",
            port: NWEndpoint.Port(rawValue: UInt16(configuration.port))!)
        parameters.allowLocalEndpointReuse = true
        listener = try NWListener(using: parameters)
        logger = try QwenHTTPLogger()
    }

    func run() -> Bool {
        let previousINT = Darwin.signal(SIGINT, SIG_IGN)
        let previousTERM = Darwin.signal(SIGTERM, SIG_IGN)
        defer { Darwin.signal(SIGINT, previousINT); Darwin.signal(SIGTERM, previousTERM) }
        network.async { self.startNetwork() }
        finished.wait()
        let succeeded = failure.withLock { $0 == nil }
        if !succeeded { log("HTTP server failed") }
        logger.stop()
        return succeeded
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
        case ("GET", "/metrics"):
            simple(client, data: try QwenHTTPFrames.response(status: 200,
                contentType: "text/plain; version=0.0.4; charset=utf-8", body: prometheusMetrics()))
        case ("GET", "/health"):
            let h = health.withLock { $0 }
            let logs = logger.snapshot()
            let body = try JSONSerialization.data(withJSONObject: [
                "status": h.state, "ready": h.state == "ready", "model": configuration.modelID,
                "pid": Int(getpid()), "experimental": true,
                "logging": [
                    "accepting": logs.accepting, "writer_exited": logs.writerExited,
                    "max_bytes": logs.maxBytes, "max_events": logs.maxEvents, "max_event_bytes": logs.maxEventBytes,
                    "buffered_bytes": logs.bufferedBytes, "buffered_events": logs.bufferedEvents,
                    "queued_events": logs.queuedEvents, "in_flight_bytes": logs.inFlightBytes,
                    "enqueued_events": logs.enqueuedEvents, "written_events": logs.writtenEvents,
                    "dropped_events": logs.droppedEvents, "dropped_bytes": logs.droppedBytes,
                    "write_failures": logs.writeFailures,
                    "last_write_errno": logs.lastWriteErrno as Any? ?? NSNull()
                ],
                "idle": h.idle && inbox.count == 0, "active": h.active, "active_jobs": h.jobs,
                "queued_prefills": h.prefills, "ready_decodes": h.ready,
                "waiting_prefix_sequences": h.waitingPrefix,
                "resident_sequences": h.resident, "reserved_tokens": h.reserved,
                "pending_requests": inbox.count, "connections": clients.count,
                "running_job": h.runningJob as Any? ?? NSNull(),
                "running_job_known": h.active == 0 || h.runningJob != nil,
                "detail": h.detail.map { String($0.prefix(512)) } as Any? ?? NSNull(),
                "prefix_cache": h.prefixCacheJSON.flatMap { try? JSONSerialization.jsonObject(with: $0) } ?? NSNull(),
                "prefix_cache_limits": configuration.prefixCacheBytes == 0 ? NSNull() : [
                    "maxEntries": configuration.prefixCacheEntries, "maxBytes": configuration.prefixCacheBytes,
                    "maxKeyTokens": 1_048_576, "ttlSeconds": configuration.prefixCacheTTLSeconds,
                    "diskRestoreTimeoutSeconds": configuration.prefixCacheRestoreTimeoutSeconds] as Any,
                "prefix_disk_cache": h.prefixDiskJSON.flatMap { try? JSONSerialization.jsonObject(with: $0) } ?? NSNull(),
                "prefix_disk_cache_limits": try prefixDiskLimitsJSON(),
                "prefix_cache_policy": configuration.prefixCacheBytes == 0 ? NSNull() : [
                    "lookup": "complete_canonical_prompt", "maximum_checkpoints_per_request": 2,
                    "checkpoint_grid_tokens": 416, "mtp_enabled": false] as Any,
                "kv_append_policy": [
                    "kv_append_mode": configuration.kvAppendMode.rawValue,
                    "scope": "autoregressive_decode_only", "prefill_uses_capacity": false,
                    "mtp_kv_append_mode": "reference"],
                "prefix_cache_shutdown_timeout_seconds": configuration.prefixCacheShutdownTimeoutSeconds,
                "state_budget": h.stateBudgetJSON.flatMap { try? JSONSerialization.jsonObject(with: $0) } ?? NSNull(),
                "mlx_memory": h.mlxMemory as Any? ?? NSNull(),
                "memory_pressure": try JSONSerialization.jsonObject(with: JSONEncoder().encode(memoryPressure.snapshot)),
                "memory_pressure_monitor_running": h.pressureMonitorRunning
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
            guard memoryPressure.checkNewRequestAdmission() else {
                reject(client, status: 429, message: "System memory pressure temporarily prevents a new request",
                       code: "resource_limit"); return
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

    /// Reads the same copied counters as health. No device calls, token keys,
    /// request IDs or unbounded labels enter the monitoring endpoint.
    private func prometheusMetrics() -> Data {
        let h = health.withLock { $0 }
        let pressure = memoryPressure.snapshot
        var lines = [String]()
        func emit<T: BinaryInteger>(_ name: String, _ type: String, _ help: String, _ value: T) {
            lines.append("# HELP qwen_\(name) \(help)")
            lines.append("# TYPE qwen_\(name) \(type)")
            lines.append("qwen_\(name) \(value)")
        }
        func decoded(_ bytes: Data?) -> [String: Any] {
            guard let bytes, let value = try? JSONSerialization.jsonObject(with: bytes) as? [String: Any] else { return [:] }
            return value
        }
        func counters(_ bytes: Data?, _ fields: [(String, String, String, String)]) {
            let values = decoded(bytes)
            for (field, name, type, help) in fields {
                // Absent observations stay absent, including during model load.
                guard let number = values[field] as? NSNumber else { continue }
                emit(name, type, help, number.int64Value)
            }
        }
        emit("ready", "gauge", "Model ready to serve requests.", h.state == "ready" ? 1 : 0)
        emit("queued_prefills", "gauge", "Queued prefill jobs including prefix waiters.", h.prefills)
        emit("ready_decodes", "gauge", "Decode jobs ready on the local scheduler.", h.ready)
        emit("waiting_prefix_sequences", "gauge", "Requests waiting for a producer or SSD prefix.", h.waitingPrefix)
        emit("resident_sequences", "gauge", "Sequences retained by the local scheduler.", h.resident)
        emit("pending_requests", "gauge", "HTTP requests waiting for tokenization and scheduler admission.", inbox.count)
        emit("memory_pressure_level", "gauge", "Effective policy level: unknown=-1 normal=0 warning=1 critical=2.",
             [QwenMemoryPressurePolicy.Level.unknown: -1, .normal: 0, .warning: 1, .critical: 2][pressure.effectiveLevel]!)
        emit("memory_pressure_monitor_running", "gauge", "Whether the OS notification monitor is running.", h.pressureMonitorRunning ? 1 : 0)
        emit("memory_pressure_os_events_total", "counter", "Delivered OS pressure notifications; excludes injected policy events.", pressure.operatingSystemEvents)
        emit("memory_pressure_injected_events_total", "counter", "Explicitly injected pressure policy observations.", pressure.injectedEvents)
        emit("memory_pressure_request_denials_total", "counter", "New request admission checks denied by pressure policy.", pressure.newRequestDenials)
        emit("memory_pressure_optional_denials_total", "counter", "Optional cache admission checks denied by pressure policy.", pressure.optionalCacheDenials)
        counters(h.stateBudgetJSON, [
            ("requestBytes", "state_request_bytes", "gauge", "Logical request state reservation; not RSS."),
            ("cacheBytes", "state_cache_bytes", "gauge", "Logical retained RAM snapshot reservation; not RSS."),
            ("workspaceBytes", "state_workspace_bytes", "gauge", "Logical workspace reservation, including unfinished transfers."),
            ("maxBytes", "state_limit_bytes", "gauge", "Joint logical state reservation limit; not a physical memory limit."),
            ("currentLeases", "state_leases", "gauge", "Live logical state reservations."),
            ("rejections", "state_reservation_rejections_total", "counter", "Rejected logical reservation attempts.")])
        counters(h.prefixCacheJSON, [
            ("restoredHits", "prefix_restores_total", "counter", "Complete prefix states actually restored from RAM or SSD."),
            ("diskHits", "prefix_ssd_restores_total", "counter", "Complete prefix states actually restored from SSD."),
            ("entries", "prefix_ram_entries", "gauge", "Retained complete RAM prefix snapshots."),
            ("evictions", "prefix_ram_evictions_total", "counter", "RAM index evictions."),
            ("liveFlights", "prefix_live_flights", "gauge", "Live prefix producer and transfer coordination records."),
            ("diskReadTimeouts", "prefix_read_timeouts_total", "counter", "Requests detached from unfinished SSD admission, read or shared-read waits after their deadline."),
            ("diskPublicationTimeouts", "prefix_publication_timeouts_total", "counter", "Requests detached from pending SSD publication waits."),
            ("restoreFailures", "prefix_restore_failures_total", "counter", "Optional state restoration failures."),
            ("retainedSystemAnchorSkips", "prefix_system_anchor_preservation_skips_total", "counter", "RAM candidates skipped to retain a shared system checkpoint.")])
        counters(h.prefixDiskJSON, [
            ("diskBytes", "ssd_archive_bytes", "gauge", "Accounted cache file bytes; excludes model n-gram storage."),
            ("entries", "ssd_archive_entries", "gauge", "Accounted cache archives."),
            ("pendingJobs", "ssd_pending_jobs", "gauge", "Admitted unfinished SSD cache jobs."),
            ("pendingBytes", "ssd_pending_bytes", "gauge", "Admitted SSD job byte reservations."),
            ("foregroundReadIntents", "ssd_foreground_read_intents", "gauge", "Metadata-only read priority ownership; not admitted IO or payload bytes."),
            ("foregroundReadIntentAcquisitions", "ssd_foreground_read_intent_acquisitions_total", "counter", "Read priority intentions acquired; not successful restorations."),
            ("optionalWritePriorityRejections", "ssd_optional_write_priority_rejections_total", "counter", "Submitted optional writes rejected to preserve foreground read priority."),
            ("bytesRead", "ssd_archive_read_bytes_total", "counter", "Cache archive file bytes read; not physical device IO."),
            ("bytesWritten", "ssd_archive_written_bytes_total", "counter", "Successfully published archive bytes; not physical device IO."),
            ("evictions", "ssd_archive_evictions_total", "counter", "SSD cache archive evictions."),
            ("writeFailures", "ssd_write_failures_total", "counter", "SSD cache write failures."),
            ("corruptions", "ssd_corruptions_total", "counter", "Rejected corrupt cache archives."),
            ("spaceRejections", "ssd_space_rejections_total", "counter", "Optional writes rejected by available space checks."),
            ("spaceQueryFailures", "ssd_space_query_failures_total", "counter", "Failed available space queries."),
            ("availableSpaceBytes", "ssd_available_space_bytes", "gauge", "Last successful available space sample for the cache filesystem.")])
        if let memory = h.mlxMemory {
            for (field, name) in [("active_bytes", "mlx_active_bytes"), ("cache_bytes", "mlx_cache_bytes"), ("peak_bytes", "mlx_peak_bytes")] {
                if let value = memory[field] { emit(name, "gauge", "MLX allocator observation; not process RSS.", value) }
            }
        }
        return Data((lines.joined(separator: "\n") + "\n").utf8)
    }

    private func prefixDiskLimitsJSON() throws -> Any {
        guard configuration.prefixCacheDirectory != nil else { return NSNull() }
        var limits = try JSONSerialization.jsonObject(with: JSONEncoder().encode(configuration.prefixDiskLimits)) as? [String: Any] ?? [:]
        limits["ttlSeconds"] = configuration.prefixCacheTTLSeconds
        return limits
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
        let pressureMonitor = QwenMemoryPressureMonitor(policy: memoryPressure)
        health.withLock { $0.pressureMonitorRunning = true }
        defer { pressureMonitor.stop(); health.withLock { $0.pressureMonitorRunning = false } }
        let diskStore = try configuration.prefixCacheDirectory.map {
            try QwenPrefixDiskStore(directory: $0, limits: configuration.prefixDiskLimits,
                ttlSeconds: TimeInterval(configuration.prefixCacheTTLSeconds))
        }
        // One finite close path covers normal shutdown and construction errors.
        // Pending host-only IO retains its own FD/Data/lease owners if this
        // deadline expires. Do not add a later unbounded flush/close fallback.
        defer {
            if let diskStore {
                let result = diskStore.close(drain: true,
                    timeout: TimeInterval(configuration.prefixCacheShutdownTimeoutSeconds))
                let state = diskStore.statistics
                log("HTTP prefix cache shutdown completed=\(result.completed) io_completed=\(result.ioCompleted) callbacks_completed=\(result.callbacksCompleted) pending_jobs=\(state.pendingJobs) pending_bytes=\(state.pendingBytes)")
            }
        }
        var previousCache = 0
        try MX.check(mlx_set_cache_limit(&previousCache, 256 * 1024 * 1024), "HTTP allocator cache")
        defer { var ignored = 0; _ = mlx_set_cache_limit(&ignored, previousCache) }
        let model = try QwenModel(modelDirectory: configuration.modelDirectory,
            reservedOutputIDs: tokenizer.reservedOutputTokenIDs,
            stateBudgetBytes: configuration.stateBudgetBytes) { count, total in
                if count % 8 == 0 || count == total { self.log("HTTP model loaded \(count)/\(total)") }
            }
        let generator = try QwenGenerator(model: model, prefixCacheLimits: configuration.prefixCacheBytes == 0 ? nil :
            .init(maxEntries: configuration.prefixCacheEntries, maxBytes: configuration.prefixCacheBytes,
                  ttlSeconds: TimeInterval(configuration.prefixCacheTTLSeconds),
                  diskRestoreTimeoutSeconds: TimeInterval(configuration.prefixCacheRestoreTimeoutSeconds)),
            prefixDiskStore: diskStore, memoryPressurePolicy: memoryPressure)
        let scheduler = try QwenLocalScheduler(generator: generator, limits: .init(executionMode: .cooperative))
        var jobs: [UUID: GPUHTTPActive] = [:]
        defer {
            for job in jobs.values { job.work.cancellation.cancel() }
            _ = try? scheduler.discardAll()
            jobs.removeAll()
            // Release private device state here. The single outer close owns
            // the finite SSD drain; service shutdown does not purge archives.
            try? generator.clearPrefixCache(includingDisk: false)
            try? MX.synchronize()
        }
        health.withLock { $0.state = inbox.isStopping ? "stopping" : "ready"; $0.idle = true }
        log("HTTP model state=ready default=AR experimental_mtp_depth=2 context=16384 chunk=416")
        func snapshot(active: Int = 0) {
            let s = scheduler.snapshot()
            let cacheJSON = generator.prefixCacheStatistics.flatMap { try? JSONEncoder().encode($0) }
            let diskJSON = generator.prefixDiskStatistics.flatMap { try? JSONEncoder().encode($0) }
            let budgetJSON = try? JSONEncoder().encode(generator.stateBudgetStatistics)
            // All MLX interaction stays on this inference OS thread. The
            // network queue sees only the copied numeric values below.
            let memory = try? MX.memory()
            health.withLock {
                $0.prefixCacheJSON = cacheJSON
                $0.prefixDiskJSON = diskJSON; $0.stateBudgetJSON = budgetJSON; $0.mlxMemory = memory
                $0.active = active; $0.jobs = jobs.count; $0.prefills = s.queuedPrefills
                $0.runningJob = s.runningJob?.uuidString
                $0.ready = s.readyDecodes; $0.resident = s.residentSequences; $0.reserved = s.reservedTokens
                $0.waitingPrefix = s.waitingPrefixSequences ?? 0
                $0.idle = s.isIdle && active == 0
                if !s.acceptingJobs { $0.state = "failed"; $0.detail = s.unavailableReason }
            }
        }
        snapshot()
        while !inbox.isStopping {
            // Drain transient Foundation/Objective-C objects after each bounded
            // admission + inference slice, rather than only when the service exits.
            let keepRunning = try autoreleasepool { () throws -> Bool in
            // Pressure callbacks never touch MLX or walk old tensor contents.
            // At an executor boundary release at most one retained snapshot;
            // private active states and unfinished I/O keep their own leases.
            _ = memoryPressure.takeTrimRequest()
            if !memoryPressure.snapshot.allowsOptionalCache {
                _ = try generator.trimPrefixCacheMemory(maxEntries: 1)
            }
            // At most one bounded tokenization/admission between GPU slices.
            // SSD completion can release workspace after the last GPU slice.
            // A short idle deadline refreshes health even without another
            // request, while condition.signal still wakes admission instantly.
            if let work = inbox.take(wait: scheduler.snapshot().isIdle,
                                     timeout: 0.1) {
                snapshot(active: 1)
                do {
                    try work.cancellation.check()
                    let messages = work.chat.messages.map { ChatMessage(role: $0.role, content: $0.content,
                        toolCalls: $0.toolCalls, toolCallID: $0.toolCallID) }
                    let tokens: [Int32]
                    let prefixPlan: QwenConversationPrefixPlan?
                    if configuration.prefixCacheBytes > 0 && work.chat.mtpDepth == 0 {
                        let conversation = try tokenizer.encodeConversation(messages: messages, tools: work.chat.activeTools,
                                                                            prefillChunk: 416)
                        tokens = conversation.tokens; prefixPlan = conversation.prefixPlan
                    } else {
                        tokens = try tokenizer.encode(tokenizer.renderChat(messages: messages, tools: work.chat.activeTools))
                        prefixPlan = nil
                    }
                    try work.cancellation.check()
                    let request = QwenGenerationRequest(tokens: tokens, maxTokens: work.chat.maxTokens,
                        contextLimit: 16_384, prefillChunk: 416, mtpDepth: work.chat.mtpDepth,
                        verification: work.chat.mtpDepth == 2 ? .batchedScalarLinear : .scalar,
                        draftHistoryTokens: work.chat.mtpDepth == 2 ? 1024 : nil, prefixCachePlan: prefixPlan,
                        kvAppendMode: work.chat.mtpDepth == 0 ? configuration.kvAppendMode : .reference)
                    let active = GPUHTTPActive(work, maxTextBytes: configuration.outputBytes - 2048)
                    let id = try scheduler.submit(request, cancellation: work.cancellation) { token in
                        self.health.withLock { $0.runningJob = active.schedulerID?.uuidString }
                        let text = active.utf8.append(try tokenizer.decodeBytes([token], skipSpecialTokens: true))
                        try self.acceptDecoded(text, active: active)
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
                    let status: Int, code: String
                    switch error {
                    case QwenGenerationError.resourceLimit(_): status = 429; code = "resource_limit"
                    case QwenLocalScheduler.Error.queueFull, QwenLocalScheduler.Error.overBudget: status = 429; code = "queue_full"
                    case QwenLocalScheduler.Error.closed(_), QwenGenerationError.unavailable(_): status = 503; code = "model_unavailable"
                    default: status = 400; code = "invalid_request_error"
                    }
                    let text = String(error.localizedDescription.prefix(512))
                    network.async {
                        if let client = self.clients[work.connectionID] {
                            self.reject(client, status: status, message: text, code: code)
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
            let paused = scheduler.snapshot()
            if paused.readyDecodes == 0, paused.queuedPrefills > 0,
               paused.waitingPrefixSequences == paused.queuedPrefills {
                // An SSD read/coalesced prefix can yield without device work.
                // Back off only when every queued producer is waiting and no
                // decode is runnable; do not penalize decode for one follower.
                Thread.sleep(forTimeInterval: 0.001)
            }
            return true
            }
            if !keepRunning { break }
        }
    }

    private func acceptDecoded(_ text: String, active: GPUHTTPActive, finishing: Bool = false) throws {
        guard var parser = active.toolParser else { try publish(text, active: active); return }
        do {
            var events = try parser.append(text)
            if finishing { events += try parser.finish() }
            active.toolParser = parser
            for event in events {
                switch event {
                case .content(let content): try publish(content, active: active)
                case .call(let call):
                    if active.work.chat.stream {
                        let index = parser.calls.firstIndex(where: { $0.id == call.id })!
                        let frame = try QwenHTTPFrames.toolCall(id: active.work.id, created: active.work.created,
                            model: active.work.chat.model, call: call, index: index)
                        let offered = active.work.output.enqueue(frame)
                        if offered.status == .overflow { logOutput(active.work, reason: "slow_consumer") }
                        actions(offered.actions, work: active.work)
                        guard offered.status == .accepted else { active.work.cancellation.cancel(); throw QwenGenerationError.cancelled }
                    } else {
                        guard active.textBudget.accept(byteCount: try call.arguments.json().utf8.count + call.name.utf8.count + call.id.utf8.count) else {
                            throw GPUHTTPOutputError.tooLarge
                        }
                    }
                }
            }
        } catch QwenToolStreamParser.Failure.outputLimit {
            active.textBudget.record(.textLimit); throw GPUHTTPOutputError.tooLarge
        } catch QwenToolStreamParser.Failure.invalidCall {
            active.invalidToolCall = true; throw QwenToolStreamParser.Failure.invalidCall
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
        // The completed result supplies the selected policy and evaluated step counts.
        // Failed/cancelled events without a result have unknown execution counts.
        let terminalPhases = event.result?.phases
        modelFields["kv_append_mode"] = terminalPhases?.kvAppendMode as Any? ?? NSNull()
        modelFields["kv_capacity_token_steps"] = terminalPhases?.kvCapacityTokenSteps as Any? ?? NSNull()
        modelFields["kv_capacity_workspace_fallbacks"] = terminalPhases?.kvCapacityWorkspaceFallbacks as Any? ?? NSNull()
        modelFields["kv_capacity_workspace_peak_bytes"] = terminalPhases?.kvCapacityWorkspacePeakBytes as Any? ?? NSNull()
        if let result = event.result {
            modelFields["model_finish_reason"] = result.finishReason.rawValue
            modelFields["prompt_tokens"] = result.statistics.promptTokenCount
            modelFields["completion_tokens"] = result.tokens.count
            modelFields["prefill_seconds"] = (result.phases?.prefill.targetSeconds).map(finite) ?? NSNull()
            modelFields["decode_seconds"] = finite(result.decodeSeconds)
            modelFields["decoded_tokens"] = result.statistics.decodedTokenCount
            modelFields["model_first_token_ready_seconds"] = finite(result.timeToFirstTokenSeconds)
            if let phases = result.phases {
                modelFields["decode_service_seconds"] = finite(phases.decodeServiceSeconds)
                modelFields["decode_suspension_seconds"] = phases.decodeSuspensionSeconds.map(finite) ?? NSNull()
                modelFields["handoff_wait_seconds"] = finite(phases.handoffWaitSeconds)
                modelFields["handoff_consume_seconds"] = finite(phases.handoffConsumeSeconds)
            }
            modelFields["cached_prompt_tokens"] = result.phases?.prefill.cachedTokenCount ?? 0
            if let prefill = result.phases?.prefill {
                modelFields["computed_prompt_tokens"] = prefill.computedTokenCount
                modelFields["actual_prefill_tokens"] = prefill.actualForwardTokenCount
                modelFields["recomputed_prefill_tokens"] = prefill.recomputedTokenCount
                modelFields["prefill_active_seconds"] = finite(prefill.totalSeconds)
                modelFields["prefill_suspension_seconds"] = prefill.suspensionSeconds.map(finite) ?? NSNull()
                modelFields["cache_source"] = prefill.cacheSource
                modelFields["cache_lookup_seconds"] = prefill.cacheLookupSeconds.map(finite) ?? NSNull()
                modelFields["cache_restore_seconds"] = prefill.cacheRestoreSeconds.map(finite) ?? NSNull()
                modelFields["cache_save_seconds"] = prefill.cacheSaveSeconds.map(finite) ?? NSNull()
                modelFields["cache_wait_seconds"] = prefill.cacheWaitSeconds.map(finite) ?? NSNull()
            }
        }
        logLifecycle(.modelTerminal, connectionID: work.connectionID, work: work,
            reason: active.textBudget.failure?.rawValue, fields: modelFields)
        do {
            let completion: QwenSSEOutputBuffer.Completion
            let outputReason: String
            let frame: Data
            if event.kind == .completed, let result = event.result {
                log("HTTP request id=\(work.id) mtp_depth=\(work.chat.mtpDepth) finish=\(result.finishReason.rawValue) prompt_tokens=\(result.statistics.promptTokenCount) completion_tokens=\(result.tokens.count) prefill_seconds=\(result.phases?.prefill.targetSeconds ?? 0) decode_seconds=\(result.decodeSeconds) scheduler_elapsed_seconds=\(event.timing.elapsedSeconds) event=model_terminal")
                try acceptDecoded(active.utf8.finish(), active: active, finishing: true)
                let calls = active.toolParser?.calls ?? []
                let reason = calls.isEmpty ? (result.finishReason == .eos ? "stop" : "length") : "tool_calls"
                let cachedTokens = result.phases?.prefill.cachedTokenCount ?? 0
                if work.chat.stream {
                    frame = try QwenHTTPFrames.finish(id: work.id, created: work.created, model: work.chat.model,
                        reason: reason, promptTokens: result.statistics.promptTokenCount, completionTokens: result.tokens.count, cachedTokens: cachedTokens)
                        + QwenHTTPFrames.done()
                } else {
                    let body = try QwenHTTPFrames.completion(id: work.id, created: work.created, model: work.chat.model,
                        text: active.text, reason: reason, promptTokens: result.statistics.promptTokenCount, completionTokens: result.tokens.count,
                        cachedTokens: cachedTokens, toolCalls: calls)
                    frame = try QwenHTTPFrames.response(status: 200, contentType: "application/json", body: body)
                }
                completion = .completed
                outputReason = "completed"
            } else {
                log("HTTP request id=\(work.id) terminal=\(event.kind.rawValue) stage=\(event.stage.rawValue) scheduler_elapsed_seconds=\(event.timing.elapsedSeconds) event=model_terminal")
                completion = event.kind == .cancelled ? .cancelled : .failed
                let resourceLimited = event.errorCode == "resource_limit"
                let failure = resourceLimited ? (code: "resource_limit", message: String((event.errorDescription ?? "State reservation budget exhausted").prefix(512))) :
                    active.invalidToolCall ? (code: "invalid_tool_call", message: "Model emitted an invalid or incomplete tool call") : active.textBudget.errorResponse(cancelled: event.kind == .cancelled)
                frame = try failureFrame(work, message: failure.message, code: failure.code, status: resourceLimited ? 429 : 500)
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
            if !cancelled && !active.invalidToolCall && active.textBudget.failure == nil { active.textBudget.record(.encodingFailed) }
            let failure = active.invalidToolCall ? (code: "invalid_tool_call", message: "Model emitted an invalid or incomplete tool call") : active.textBudget.errorResponse(cancelled: cancelled)
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

    private func failureFrame(_ work: GPUHTTPWork, message: String, code: String, status: Int = 500) throws -> Data {
        let type = status == 429 ? "rate_limit_error" : "server_error"
        if work.chat.stream { return try QwenHTTPFrames.sseError(message: message, code: code, type: type) + QwenHTTPFrames.done() }
        return try QwenHTTPFrames.response(status: status, contentType: "application/json",
            body: QwenHTTPFrames.error(message: message, code: code, type: type))
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
        logger.enqueue(Data((text + "\n").utf8))
    }
}
