#if canImport(CoreAI)
import ANERunnerCore
import CryptoKit
import Dispatch
import Foundation
import MachO
import Synchronization

struct CoreAIServiceConfiguration: Sendable {
    let modelDirectory: URL
    let attentionManifest: URL
    let denseManifest: URL
    let moeManifest: URL
    var pdManifest: URL? = nil
    /// Zero selects the loaded model's chunk, one forces tokenwise prefill.
    var prefillChunk = 0
    var cacheBytes = 536_870_912
    var cacheEntries = 2
    var maxPendingRequests = 2
    var maxOutputBytes = 65_536
    var modelID: String { modelDirectory.lastPathComponent + "-CoreAI" }
}

/// The single consumer task owns every non-Sendable CoreAI tensor and model.
/// Network callbacks exchange only Sendable requests/events and locked counters.
@available(macOS 27.0, *)
final class CoreAIServiceWorker: CoreAIServiceBackend {
    private struct Job: Sendable {
        let id: UUID
        let request: QwenHTTPChatRequest
        let cancellation: CoreAIRequestCancellation
        let emit: @Sendable (CoreAIServiceEvent) async throws -> Void
    }
    private struct Status: Codable {
        var ready = false
        var stopping = false
        var phase = "loading"
        var capacity = 0
        var completed = 0
        var cancelled = 0
        var failed = 0
        var cachedTokens = 0
        var cacheBytes = 0
        var cacheEntries = 0
        var cacheHits = 0
        var prefillTokensProcessed = 0
        var prefillSeconds = 0.0
        var decodeSeconds = 0.0
        var prefillGroupMilliseconds: [String: Double] = [:]
        var decodeGroupMilliseconds: [String: Double] = [:]
        var independentPDFunctions = false
        var prefillChunkSize = 1
        var progressCompleted = 0
        var progressTotal = 0
        var lastError: String?
    }
    private struct Shared {
        var status = Status()
        var jobs: [UUID: CoreAIRequestCancellation] = [:]
    }
    let modelID: String
    private let configuration: CoreAIServiceConfiguration
    private let state = Mutex(Shared())
    private let task = Mutex<Task<Void, Never>?>(nil)
    private let queue: CoreAIRequestQueue<Job>
    // Coalesced wakeups carry no request payload; queued jobs can be removed.
    private let stream: AsyncStream<Void>
    private let continuation: AsyncStream<Void>.Continuation

    init(configuration: CoreAIServiceConfiguration) {
        self.configuration = configuration
        modelID = configuration.modelID
        queue = CoreAIRequestQueue(limit: configuration.maxPendingRequests)
        (stream, continuation) = AsyncStream.makeStream(bufferingPolicy: .bufferingNewest(1))
    }

    func start() {
        task.withLock { current in
            guard current == nil else { return }
            current = Task.detached { [self] in await consume() }
        }
    }

    func health() -> Data {
        let (status, count, queueState) = state.withLock { ($0.status, $0.jobs.count, queue.snapshot) }
        var data = (try? JSONSerialization.jsonObject(with: JSONEncoder().encode(status))) as? [String: Any] ?? [:]
        data["backend"] = "native-coreai"
        data["model"] = modelID
        data["requests_in_flight"] = count
        data["active_requests"] = queueState.activeCount
        data["queued_requests"] = queueState.queuedCount
        data["max_pending_requests"] = queueState.limit
        data["prefix_cache_enabled"] = configuration.cacheBytes > 0 && configuration.cacheEntries > 0
        data["prefix_cache_limit_bytes"] = configuration.cacheBytes
        data["prefill_policy"] = status.prefillChunkSize == 1 ? "tokenwise" : "chunked"
        data["independent_pd_functions"] = status.independentPDFunctions
        data["prefill_chunk_size"] = status.prefillChunkSize
        data["pd_scheduling"] = "serial"
        data["prefill_group_milliseconds"] = status.prefillGroupMilliseconds
        data["decode_group_milliseconds"] = status.decodeGroupMilliseconds
        data["quality_acceptance"] = false
        data["hardware_placement_verified"] = false
        data["compute_preference"] = "gpu"
        data["mlx_runtime_loaded"] = false
        return (try? JSONSerialization.data(withJSONObject: data, options: [.sortedKeys])) ?? Data("{}".utf8)
    }

    func submit(_ request: QwenHTTPChatRequest,
                onEvent: @escaping @Sendable (CoreAIServiceEvent) async throws -> Void) throws -> CoreAIRequestCancellation {
        guard request.mtpDepth == 0, !request.parsesTools, request.tools.isEmpty else {
            throw QwenHTTPProtocolError(statusCode: 400, message: "CoreAI currently supports text AR requests; tools and MTP are unavailable")
        }
        let id = UUID()
        let cancellation = CoreAIRequestCancellation { [weak self] in self?.cancelQueued(id) }
        let job = Job(id: id, request: request, cancellation: cancellation, emit: onEvent)
        try state.withLock { value in
            guard value.status.ready, !value.status.stopping else {
                throw QwenHTTPProtocolError(statusCode: 503, message: "CoreAI model is loading, unavailable or stopping")
            }
            do { try queue.enqueue(id: id, job: job) }
            catch CoreAIRequestQueue<Job>.AdmissionError.full {
                throw QwenHTTPProtocolError(statusCode: 429, message: "CoreAI request queue is full")
            } catch {
                throw QwenHTTPProtocolError(statusCode: 503, message: "CoreAI worker is unavailable")
            }
            value.jobs[id] = cancellation
        }
        switch continuation.yield(()) {
        case .enqueued, .dropped:
            // A replaced wakeup is harmless: the consumer drains the FIFO.
            return cancellation
        case .terminated:
            cancellation.cancel()
            throw QwenHTTPProtocolError(statusCode: 503, message: "CoreAI worker is unavailable")
        @unknown default:
            cancellation.cancel()
            throw QwenHTTPProtocolError(statusCode: 503, message: "CoreAI worker could not admit this request")
        }
    }

    private func cancelQueued(_ id: UUID) {
        state.withLock { value in
            // An active job retains its slot until its model state is reset.
            guard queue.cancelQueued(id: id) != nil else { return }
            value.jobs.removeValue(forKey: id)
            value.status.cancelled += 1
        }
    }

    func shutdown() {
        let cancellations = state.withLock { value in
            value.status.stopping = true
            value.status.ready = false
            return Array(value.jobs.values)
        }
        // Cancellation callbacks remove queued jobs and acquire state themselves.
        for cancellation in cancellations { cancellation.cancel() }
        continuation.finish()
        task.withLock { $0?.cancel() }
    }

    func waitUntilStopped() async {
        let current = task.withLock { $0 }
        await current?.value
    }

    private func consume() async {
        defer {
            continuation.finish()
            state.withLock { value in
                value.status.ready = false
                value.status.phase = "stopped"
                value.jobs.removeAll()
                _ = queue.stopAndDrain()
                if let active = queue.snapshot.activeID { _ = queue.finishActive(id: active) }
            }
        }
        do {
            let session = try await CoreAIServiceSession(configuration: configuration)
            try Task.checkCancellation()
            state.withLock {
                guard !$0.status.stopping else { return }
                $0.status.ready = true; $0.status.capacity = session.model.capacity; $0.status.phase = "idle"
                $0.status.independentPDFunctions = session.model.usesIndependentPhases
                $0.status.prefillChunkSize = session.prefillChunk
            }
            for await _ in stream {
                while !Task.isCancelled, let entry = queue.takeNext() {
                    let job = entry.job
                    do {
                        try job.cancellation.check()
                        let result = try await session.generate(job.request, cancellation: job.cancellation, emit: job.emit) { [self] phase, completed, total in
                            state.withLock {
                                $0.status.phase = phase
                                $0.status.progressCompleted = completed
                                $0.status.progressTotal = total
                            }
                        }
                        try await job.emit(.completed(result))
                        state.withLock { value in
                            value.status.completed += 1
                            value.status.cachedTokens += result.cachedTokens
                            if result.cachedTokens > 0 { value.status.cacheHits += 1 }
                            value.status.prefillTokensProcessed += result.promptTokens - result.cachedTokens
                            value.status.prefillSeconds += result.prefillSeconds
                            value.status.decodeSeconds += result.decodeSeconds
                            for (group, milliseconds) in result.prefillGroupMilliseconds {
                                value.status.prefillGroupMilliseconds[group, default: 0] += milliseconds
                            }
                            for (group, milliseconds) in result.decodeGroupMilliseconds {
                                value.status.decodeGroupMilliseconds[group, default: 0] += milliseconds
                            }
                        }
                    } catch {
                        let cancelled = job.cancellation.isCancelled || error is CancellationError || Task.isCancelled
                        state.withLock { value in
                            if cancelled { value.status.cancelled += 1 }
                            else { value.status.failed += 1; value.status.lastError = error.localizedDescription }
                        }
                        let status = (error as? QwenHTTPProtocolError)?.statusCode ?? (cancelled ? 408 : 500)
                        try? await job.emit(.failed(status: status, message: cancelled ? "Request cancelled" : error.localizedDescription))
                    }
                    // Reset even after partial/failed tokens before accepting the next request.
                    try session.reset()
                    state.withLock { value in
                        _ = queue.finishActive(id: job.id)
                        value.jobs.removeValue(forKey: job.id)
                        value.status.cacheEntries = session.cacheCount
                        value.status.cacheBytes = session.cacheBytes
                        value.status.phase = "idle"
                        value.status.progressCompleted = 0
                        value.status.progressTotal = 0
                    }
                }
            }
        } catch {
            state.withLock {
                $0.status.ready = false
                $0.status.lastError = error.localizedDescription; $0.status.failed += 1
            }
            // Atomically close admission, then notify the bounded remaining queue.
            continuation.finish()
            for entry in queue.stopAndDrain() {
                try? await entry.job.emit(.failed(status: 503, message: "CoreAI model is unavailable"))
            }
        }
    }
}

@available(macOS 27.0, *)
private final class CoreAIServiceSession {
    private struct Entry {
        let tokens: [Int32]
        let history: [UInt32]
        let logits: [Float]
        let snapshot: CoreAITextSnapshot
        var bytes: Int { snapshot.logicalByteCount + tokens.count * 4 + history.count * 4 + logits.count * 4 }
    }
    /// A handoff transfers the live state only within this serial session.
    /// Its owner, epoch, exact offset and single-use identity must all match.
    private struct PrefillHandoff {
        let id: UUID
        let owner: UUID
        let epoch: UInt64
        let offset: Int
        let promptTokens: Int
        let cachedTokens: Int
        let history: [UInt32]
        let logits: [Float]
        let seconds: Double
        let groupMilliseconds: [String: Double]
    }
    let model: CoreAITextRuntime
    let prefillChunk: Int
    private let handoffOwner = UUID()
    private var generationEpoch: UInt64 = 0
    private var generationActive = false
    private var pendingHandoffID: UUID?
    private let tokenizer: QwenTokenizer
    private let hash: NGramHash
    private let table: NGramTable
    private let configuration: CoreAIServiceConfiguration
    private var entries: [Entry] = [] // Most recently used first.
    var cacheCount: Int { entries.count }
    var cacheBytes: Int { entries.reduce(0) { $0 + $1.bytes } }

    init(configuration: CoreAIServiceConfiguration) async throws {
        self.configuration = configuration
        let directory = configuration.modelDirectory
        let config = try QwenConfiguration(modelDirectory: directory)
        tokenizer = try QwenTokenizer(modelDirectory: directory)
        hash = try NGramHash(unigramVocabularySize: UInt32(config.vocabularySize), ngramSize: config.ngramSize,
            headsPerNGram: config.ngramHeadsPerOrder, vocabularyBase: UInt64(config.ngramVocabularyBase),
            vocabularyDivisor: UInt64(config.ngramDivisor), pleLayerIndex: 0, eosTokenID: UInt32(config.eosTokenID))
        guard !config.ngramTableFile.contains(".."), !config.ngramTableFile.hasPrefix("/") else {
            throw QwenGenerationError.invalidRequest("Invalid n-gram table path")
        }
        table = try NGramTable(url: directory.appendingPathComponent(config.ngramTableFile))
        guard table.rowCount == hash.totalRows, table.dimension * hash.headCount == config.pleEmbeddingDimension,
              table.scale == Float(config.ngramScale), config.pleLayerIndices == [1] else {
            throw QwenGenerationError.invalidRequest("PLE table does not match model configuration")
        }
        model = try await CoreAITextRuntime(attentionManifest: configuration.attentionManifest,
            denseManifest: configuration.denseManifest, moeManifest: configuration.moeManifest,
            pdManifest: configuration.pdManifest)
        if configuration.prefillChunk == 0 {
            prefillChunk = model.prefillChunkSize
        } else if configuration.prefillChunk == 1 || configuration.prefillChunk == model.prefillChunkSize {
            prefillChunk = configuration.prefillChunk
        } else {
            throw QwenGenerationError.invalidRequest(
                "prefillChunk must be 0 (automatic), 1, or the loaded model chunk \(model.prefillChunkSize)")
        }
        let configSHA = SHA256.hash(data: try Data(contentsOf: directory.appendingPathComponent("config.json")))
            .map { String(format: "%02x", $0) }.joined()
        guard model.manifestModelDirectory == directory, model.sourceConfigSHA256 == configSHA else {
            throw QwenGenerationError.invalidRequest("CoreAI assets belong to a different model")
        }
        let images = (0..<_dyld_image_count()).compactMap { _dyld_get_image_name($0).map { String(cString: $0) } }
        guard !images.contains(where: { URL(fileURLWithPath: $0).lastPathComponent.lowercased().contains("mlx") }) else {
            throw QwenGenerationError.unavailable("Unexpected MLX runtime in CoreAI worker")
        }
    }

    func generate(_ request: QwenHTTPChatRequest, cancellation: CoreAIRequestCancellation,
                  emit: @Sendable (CoreAIServiceEvent) async throws -> Void,
                  phase: (String, Int, Int) -> Void) async throws -> CoreAIServiceResult {
        guard !generationActive else { throw QwenGenerationError.unavailable("CoreAI session is already executing a request") }
        generationActive = true
        generationEpoch &+= 1
        defer {
            pendingHandoffID = nil
            generationActive = false
            generationEpoch &+= 1
        }
        let handoff = try await prefill(request, cancellation: cancellation, emit: emit, phase: phase)
        return try await decode(handoff, maxTokens: request.maxTokens,
                                cancellation: cancellation, emit: emit, phase: phase)
    }

    /// Worker calls this after every terminal request, before releasing its slot.
    func reset() throws {
        guard !generationActive else { throw QwenGenerationError.unavailable("Cannot reset an active CoreAI request") }
        pendingHandoffID = nil
        generationEpoch &+= 1
        try model.reset()
    }

    private func prefill(_ request: QwenHTTPChatRequest, cancellation: CoreAIRequestCancellation,
                         emit: @Sendable (CoreAIServiceEvent) async throws -> Void,
                         phase: (String, Int, Int) -> Void) async throws -> PrefillHandoff {
        let messages = request.messages.map { ChatMessage(role: $0.role, content: $0.content) }
        let tokens: [Int32]
        let systemCount: Int
        do {
            tokens = try tokenizer.encode(tokenizer.renderChat(messages: messages))
            systemCount = try tokenizer.systemPrefixTokenCount(messages: messages, tools: [], fullTokens: tokens)
        } catch {
            throw QwenHTTPProtocolError(statusCode: 400, message: "Invalid text conversation: \(error)")
        }
        guard !tokens.isEmpty, request.maxTokens > 0, tokens.count <= model.capacity - request.maxTokens else {
            throw QwenHTTPProtocolError(statusCode: 400, message: "Prompt plus max_tokens exceeds CoreAI context capacity \(model.capacity)")
        }
        let unsupported: Set<Int32> = [248053, 248054, 248055, 248056, 248057, 248070, 248071, 248076]
        guard tokens.allSatisfy({ $0 >= 0 && $0 < model.vocabularySize && !unsupported.contains($0) }) else {
            throw QwenHTTPProtocolError(statusCode: 400, message: "Conversation contains unsupported non-text token IDs")
        }
        guard model.valid, model.offset == 0, pendingHandoffID == nil else {
            throw QwenGenerationError.unavailable("CoreAI request requires a reset model before prefill")
        }
        try cancellation.check()
        phase("prefill", 0, tokens.count)
        let prefillStart = DispatchTime.now().uptimeNanoseconds
        var cached = 0, history = hash.initialHistory, logits: [Float] = []
        if let index = entries.indices.filter({ tokens.starts(with: entries[$0].tokens) })
            .max(by: { entries[$0].tokens.count < entries[$1].tokens.count }) {
            let entry = entries.remove(at: index)
            try model.restore(entry.snapshot)
            entries.insert(entry, at: 0)
            cached = entry.tokens.count; history = entry.history; logits = entry.logits
        }
        // restore clears model counters. Establish the baseline only after the
        // optional restore, so a prefix hit cannot create negative deltas.
        let groupsBefore = model.predictionMillisecondsByGroup
        guard model.offset == cached else { throw QwenGenerationError.unavailable("Restored prefix offset disagrees with token history") }
        // No SSE success header is sent until all request validation is complete.
        try await emit(.started)
        phase("prefill", cached, tokens.count)
        var position = cached, heartbeatTokens = 0
        while position < tokens.count {
            try cancellation.check()
            let remaining = tokens.count - position
            let beforeSystemBoundary = systemCount > position ? systemCount - position : remaining
            let count = remaining >= prefillChunk && beforeSystemBoundary >= prefillChunk ? prefillChunk : 1
            let chunk = Array(tokens[position..<position + count])
            let lookup = try pleRows(tokens: chunk, history: history)
            try cancellation.check()
            logits = try await model.prefill(tokens: chunk, pleEmbedding: lookup.values)
            history = lookup.history
            position += count
            guard model.offset == position else { throw QwenGenerationError.unavailable("Prefill model offset disagrees with token history") }
            try cancellation.check()
            phase("prefill", position, tokens.count)
            if position == systemCount {
                try publish(tokens: tokens, count: systemCount, history: history, logits: logits, cancellation: cancellation)
            }
            heartbeatTokens += count
            if heartbeatTokens >= 8 {
                try await emit(.heartbeat)
                heartbeatTokens %= 8
            }
        }
        try cancellation.check()
        if cached != tokens.count {
            try publish(tokens: tokens, count: tokens.count, history: history, logits: logits, cancellation: cancellation)
        }
        let handoff = PrefillHandoff(id: UUID(), owner: handoffOwner, epoch: generationEpoch,
            offset: model.offset, promptTokens: tokens.count, cachedTokens: cached,
            history: history, logits: logits, seconds: seconds(prefillStart),
            groupMilliseconds: groupDelta(since: groupsBefore))
        pendingHandoffID = handoff.id
        return handoff
    }

    private func decode(_ handoff: PrefillHandoff, maxTokens: Int,
                        cancellation: CoreAIRequestCancellation,
                        emit: @Sendable (CoreAIServiceEvent) async throws -> Void,
                        phase: (String, Int, Int) -> Void) async throws -> CoreAIServiceResult {
        guard generationActive, pendingHandoffID == handoff.id, handoff.owner == handoffOwner,
              handoff.epoch == generationEpoch, handoff.offset == model.offset,
              handoff.offset == handoff.promptTokens, model.valid else {
            throw QwenGenerationError.unavailable("Prefill handoff is stale, consumed, foreign or at a different model offset")
        }
        // Consume before the first await. Copies of this value cannot start a
        // second decode, even if the model has not yet advanced by one token.
        pendingHandoffID = nil
        try cancellation.check()
        let groupsBefore = model.predictionMillisecondsByGroup
        var history = handoff.history, logits = handoff.logits
        phase("decode", 0, maxTokens)
        var decoder = IncrementalUTF8Decoder(), text = "", outputCount = 0, reason = "length", decodeSeconds = 0.0
        for position in 0..<maxTokens {
            try cancellation.check()
            guard logits.count == model.vocabularySize, logits.allSatisfy(\.isFinite) else {
                throw QwenGenerationError.unavailable("Invalid or nonfinite model logits")
            }
            var token: Int32?, best = -Float.infinity
            for (index, score) in logits.enumerated() where !tokenizer.reservedOutputTokenIDs.contains(Int32(index)) {
                if score > best { best = score; token = Int32(index) }
            }
            guard let selected = token else { throw QwenGenerationError.unavailable("No permitted output token") }
            outputCount += 1
            phase("decode", outputCount, maxTokens)
            if tokenizer.eosTokenIDs.contains(selected) { reason = "stop"; break }
            let delta = decoder.append(try tokenizer.decodeBytes([selected], skipSpecialTokens: true))
            guard delta.utf8.count <= configuration.maxOutputBytes - text.utf8.count else {
                throw QwenHTTPProtocolError(statusCode: 500, message: "Generated text exceeds the configured output limit")
            }
            text += delta
            if !delta.isEmpty { try await emit(.delta(delta)) }
            if position + 1 < maxTokens {
                let start = DispatchTime.now().uptimeNanoseconds
                let lookup = try pleRows(tokens: [selected], history: history)
                try cancellation.check()
                logits = try await model.forward(token: selected, pleEmbedding: lookup.values)
                history = lookup.history
                try cancellation.check()
                decodeSeconds += seconds(start)
            }
        }
        let tail = decoder.finish()
        guard tail.utf8.count <= configuration.maxOutputBytes - text.utf8.count else {
            throw QwenHTTPProtocolError(statusCode: 500, message: "Generated text exceeds the configured output limit")
        }
        text += tail
        if !tail.isEmpty { try await emit(.delta(tail)) }
        try cancellation.check()
        return CoreAIServiceResult(text: text, finishReason: reason, promptTokens: handoff.promptTokens,
            completionTokens: outputCount, cachedTokens: handoff.cachedTokens,
            prefillSeconds: handoff.seconds, decodeSeconds: decodeSeconds,
            prefillGroupMilliseconds: handoff.groupMilliseconds,
            decodeGroupMilliseconds: groupDelta(since: groupsBefore))
    }

    /// NGramHash returns token-major rows for the whole chunk, including EOS
    /// history resets. The table reads that batch once; no padding is introduced.
    private func pleRows(tokens: [Int32], history: [UInt32]) throws -> (values: [Float], history: [UInt32]) {
        let ids = tokens.map(UInt32.init)
        let rows = try hash.rowIDs(previousTokens: history, tokens: ids)
        let values = try table.readRows(rows)
        return (values, try hash.history(after: ids, previousTokens: history))
    }

    private func publish(tokens: [Int32], count: Int, history: [UInt32], logits: [Float],
                         cancellation: CoreAIRequestCancellation) throws {
        guard configuration.cacheBytes > 0, configuration.cacheEntries > 0 else { return }
        let prefix = Array(tokens.prefix(count))
        guard !entries.contains(where: { $0.tokens == prefix }) else { return }
        guard model.offset == count else { throw QwenGenerationError.unavailable("Prefix checkpoint is not at its token boundary") }
        let required = try model.stateByteCount() + prefix.count * 4 + history.count * 4 + logits.count * 4
        guard required <= configuration.cacheBytes else { return }
        while !entries.isEmpty && (entries.count >= configuration.cacheEntries || cacheBytes > configuration.cacheBytes - required) {
            entries.removeLast()
        }
        let snapshot = try model.checkpoint()
        try cancellation.check()
        entries.insert(Entry(tokens: prefix, history: history, logits: logits, snapshot: snapshot), at: 0)
    }

    /// Outer result dictionaries describe the business phase. Inner group names
    /// preserve the kernel path (for example decode.gdn for an S1 prefill tail).
    private func groupDelta(since baseline: [String: Double]) -> [String: Double] {
        model.predictionMillisecondsByGroup.reduce(into: [:]) { result, item in
            let delta = item.value - (baseline[item.key] ?? 0)
            if delta > 0 { result[item.key] = delta }
        }
    }

    private func seconds(_ start: UInt64) -> Double { Double(DispatchTime.now().uptimeNanoseconds - start) * 1e-9 }
}
#endif
