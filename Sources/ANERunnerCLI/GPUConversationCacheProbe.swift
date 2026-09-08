import ANERunnerCore
import ANERunnerGPU
import CMLX
import Foundation

private struct ConversationCacheFixture {
    let id: String
    let messages: [ChatMessage]
    let tools: [QwenToolDefinition]
    let conversation: QwenTokenizedConversation
    let rendered: String
    var tokens: [Int32] { conversation.tokens }
    var plan: QwenConversationPrefixPlan { conversation.prefixPlan }
    var system: Int { plan.systemProducerTokenCount ?? 0 }
    var tail: Int { plan.publicationTokenCounts.last ?? 0 }

    func wire() throws -> [String: Any] {
        let messages = try messages.map { message -> [String: Any] in
            var value: [String: Any] = ["role": message.role, "content": message.content]
            if let reasoning = message.reasoningContent { value["reasoning_content"] = reasoning }
            if let calls = message.toolCalls { value["tool_calls"] = try calls.map { try $0.wire() } }
            if let id = message.toolCallID { value["tool_call_id"] = id }
            return value
        }
        return ["id": id, "messages": messages, "tools": tools.map { $0.wire.foundation },
            "token_ids": tokens, "prompt_tokens": tokens.count,
            "rendered_sha256": MoETilingBytes.digest(Data(rendered.utf8)),
            "token_ids_json_sha256": MoETilingBytes.digest(try JSONEncoder().encode(tokens)),
            "plan": ["prefill_chunk": plan.prefillChunk, "lookup_max_tokens": plan.lookupMaxTokens,
                "publication_token_counts": plan.publicationTokenCounts, "system_producer_tokens": system,
                "tail_tokens": tail]]
    }
}

extension RunnerCLI {
    /// Six independent cold references and twelve cache/lifetime comparisons.
    /// All model work is serial; cursor interleaving exercises ownership only.
    static func probeGPUConversationCache(_ args: Arguments) throws {
        try args.validate(["--model-dir", "--system-file", "--cache-directory", "--output"])
        let output = try args.require("--output")
        guard !FileManager.default.fileExists(atPath: output) else {
            throw CLIError.usage("Conversation cache probe requires a new --output")
        }
        var report: [String: Any] = ["schema": "qwen38-conversation-cache-v1", "complete": false,
            "passed": false, "command": CommandLine.arguments, "http_transport_tested": false,
            "scope": "Real structured multi-turn/tool history, native mixed-state checkpoints and AR output correctness; diagnostic readback is not a performance measurement.",
            "notes": ["Synthetic tool results are input fixtures; no external tool is executed.",
                "One model, one executor, cache-disabled references and cooperative cached cursors.",
                "State anchors cover checkpoints, not all final decode tensors.",
                "Logical state leases exclude model weights, transient activations and allocator retention."]]
        var checks = [String: Bool](), trials = [[String: Any]](), stateChecks = [[String: Any]]()
        var eventCounts = [String: Int]()
        func object<T: Encodable>(_ value: T) throws -> Any {
            try JSONSerialization.jsonObject(with: JSONEncoder().encode(value))
        }
        func write(complete: Bool = false) throws {
            report["complete"] = complete; report["checks"] = checks
            report["trials"] = trials; report["state_checks"] = stateChecks; report["state_event_counts"] = eventCounts
            report["passed"] = complete && trials.count == 12 && !stateChecks.isEmpty &&
                checks.values.allSatisfy { $0 } && trials.allSatisfy { $0["passed"] as? Bool == true } &&
                stateChecks.allSatisfy { $0["passed"] as? Bool == true }
            try emit(report, to: output)
        }
        func require(_ label: String, _ condition: Bool) throws {
            checks[label] = condition
            try write()
            guard condition else { throw CLIError.usage("Conversation cache check failed: \(label)") }
        }
        try write()
        do {
            let directory = URL(fileURLWithPath: try args.require("--model-dir")).standardizedFileURL.resolvingSymlinksInPath()
            let sourceURL = URL(fileURLWithPath: try args.require("--system-file")).standardizedFileURL
            let source = try Data(contentsOf: sourceURL)
            guard let systemText = String(data: source, encoding: .utf8), !systemText.isEmpty else {
                throw CLIError.usage("Expected a nonempty UTF-8 system fixture")
            }
            let executable = Bundle.main.executableURL ?? URL(fileURLWithPath: CommandLine.arguments[0])
            report["provenance"] = ["executable_sha256": try MoETilingBytes.hash(executable),
                "config_sha256": try MoETilingBytes.hash(directory.appendingPathComponent("config.json")),
                "tokenizer_sha256": try MoETilingBytes.hash(directory.appendingPathComponent("tokenizer.json")),
                "system_file": sourceURL.path, "system_sha256": MoETilingBytes.digest(source)]
            report["model_directory"] = directory.path
            let tokenizer = try QwenTokenizer(modelDirectory: directory)
            let chunk = 416, maximumOutput = 16, contextLimit = 16_384
            let tool = try QwenToolDefinition.decode(["type": "function", "function": [
                "name": "measure", "description": "Read the recorded weather measurement for a city.",
                "parameters": ["type": "object", "properties": ["city": ["type": "string"]], "required": ["city"]]]])
            let changedTool = try QwenToolDefinition.decode(["type": "function", "function": [
                "name": "measure", "description": "Return the archived weather measurement for a city.",
                "parameters": ["type": "object", "properties": ["city": ["type": "string"]], "required": ["city"]]]])
            let call = QwenToolCall(id: "call_measure", name: "measure", arguments: .object(["city": .string("Beijing")]))
            func repeated(_ line: String, _ count: Int) -> String { String(repeating: line, count: count) }
            func makeFixtures(document: Int, result: Int, question: Int) throws -> [ConversationCacheFixture] {
                let documentText = "Original station document.\n" + repeated("Station records are fixed and verified.\n", document)
                let correctedText = "Corrected station document.\n" + repeated("Station records are fixed and verified.\n", document)
                func history(_ document: String, _ result: String) -> [ChatMessage] {
                    [.init(role: "system", content: systemText),
                     .init(role: "user", content: document + "Read the Beijing measurement, then briefly summarize it."),
                     .init(role: "assistant", content: "", toolCalls: [call]),
                     .init(role: "tool", content: result, toolCallID: call.id)]
                }
                let resultA = "Beijing measurement: dry.\n" + repeated("The recorded station value is stable.\n", result)
                let resultB = "Beijing measurement: wet.\n" + repeated("The recorded station value is revised.\n", result)
                let a = history(documentText, resultA)
                let completed = ChatMessage(role: "assistant", content: "The recorded Beijing measurement is dry and stable.")
                let alpha = ChatMessage(role: "user", content: "Alpha review request.\n" +
                    repeated("Explain the fixed record in this review.\n", question) + "Give one brief final sentence.")
                let beta = ChatMessage(role: "user", content: "Beta review request.\n" +
                    repeated("Explain the revised record in this review.\n", question) + "Give one brief final sentence.")
                let raw: [(String, [ChatMessage], [QwenToolDefinition])] = [
                    ("A", a, [tool]), ("B", history(documentText, resultB), [tool]),
                    ("N1", a + [completed, alpha], [tool]), ("N2", a + [completed, beta], [tool]),
                    ("E", history(correctedText, resultA), [tool]), ("T", a, [changedTool])]
                return try raw.map { id, messages, tools in
                    ConversationCacheFixture(id: id, messages: messages, tools: tools,
                        conversation: try tokenizer.encodeConversation(messages: messages, tools: tools, prefillChunk: chunk),
                        rendered: try tokenizer.renderChat(messages: messages, tools: tools))
                }
            }
            func lcp(_ a: [Int32], _ b: [Int32]) -> Int {
                var count = 0
                for (left, right) in zip(a, b) { if left != right { break }; count += 1 }
                return count
            }
            var fixtures = [ConversationCacheFixture](), cpuAttempts = [[String: Any]]()
            var documentRepeats = 64, resultRepeats = 64, questionRepeats = 64
            var geometryValid = false
            for attempt in 1...12 {
                fixtures = try makeFixtures(document: documentRepeats, result: resultRepeats, question: questionRepeats)
                let a = fixtures[0], b = fixtures[1], n1 = fixtures[2], n2 = fixtures[3], e = fixtures[4], t = fixtures[5]
                let longest = fixtures.map { $0.tokens.count }.max() ?? 0
                let pairs = ["A_B": lcp(a.tokens, b.tokens), "A_N1": lcp(a.tokens, n1.tokens),
                    "A_N2": lcp(a.tokens, n2.tokens), "N1_N2": lcp(n1.tokens, n2.tokens),
                    "A_E": lcp(a.tokens, e.tokens), "A_T": lcp(a.tokens, t.tokens)]
                let grid = fixtures.allSatisfy { f in
                    f.plan.lookupMaxTokens == f.tokens.count - 1 && !f.plan.publicationTokenCounts.isEmpty &&
                    f.plan.publicationTokenCounts.allSatisfy { $0 > 0 && $0 % chunk == 0 && $0 < f.tokens.count }
                }
                let sameSystem = fixtures.prefix(5).allSatisfy { $0.system == a.system &&
                    Array($0.tokens.prefix(a.system)) == Array(a.tokens.prefix(a.system)) }
                geometryValid = longest + maximumOutput <= contextLimit && a.system >= 10_000 && grid && sameSystem &&
                    a.tail > a.system + chunk && a.tail < a.tokens.count - 1 &&
                    pairs["A_B"]! < min(a.tail, b.tail) && pairs["A_N1"]! >= a.tail && pairs["A_N2"]! >= a.tail &&
                    pairs["N1_N2"]! < min(n1.tail, n2.tail) && pairs["A_E"]! >= a.system &&
                    pairs["A_E"]! < a.tail && pairs["A_T"]! < a.system
                let actual: [String: Any] = ["attempt": attempt, "document_repeats": documentRepeats,
                    "result_repeats": resultRepeats, "question_repeats": questionRepeats, "lcp": pairs,
                    "geometry_valid": geometryValid, "grid_valid": grid, "same_system": sameSystem,
                    "lengths": fixtures.map { ["id": $0.id, "prompt": $0.tokens.count, "system": $0.system, "tail": $0.tail] }]
                cpuAttempts.append(actual); report["cpu_fixture_attempts"] = cpuAttempts
                report["fixtures"] = try fixtures.map { try $0.wire() }
                try write()
                let summary = fixtures.map { "\($0.id):P\($0.tokens.count)/S\($0.system)/K\($0.tail)" }.joined(separator: " ")
                FileHandle.standardError.write(Data("Conversation fixture \(attempt): \(summary); LCP \(pairs)\n".utf8))
                if geometryValid { break }
                if longest + maximumOutput > contextLimit {
                    documentRepeats = max(24, documentRepeats * 4 / 5)
                    resultRepeats = max(24, resultRepeats * 4 / 5)
                    questionRepeats = max(24, questionRepeats * 4 / 5)
                } else {
                    if a.tail <= a.system + chunk { documentRepeats += 16; resultRepeats += 16 }
                    if pairs["A_B"]! >= min(a.tail, b.tail) { resultRepeats += 16 }
                    if pairs["N1_N2"]! >= min(n1.tail, n2.tail) { questionRepeats += 16 }
                    if pairs["A_N1"]! < a.tail || pairs["A_N2"]! < a.tail || a.tail == a.tokens.count - 1 {
                        resultRepeats += 1
                    }
                }
            }
            try require("cpu_fixture_geometry", geometryValid)
            let byID = Dictionary(uniqueKeysWithValues: fixtures.map { ($0.id, $0) })
            let a = byID["A"]!, b = byID["B"]!, system = a.system, tail = a.tail
            let ramBytes = 1024 * 1024 * 1024, diskBytes = 4 * 1024 * 1024 * 1024, stateBytes = 4 * 1024 * 1024 * 1024
            let disk = try QwenPrefixDiskStore(directory: URL(fileURLWithPath: try args.require("--cache-directory")),
                limits: .init(maxEntries: 16, maxBytes: diskBytes, maxPendingJobs: 2, maxPendingBytes: ramBytes))
            defer { disk.close(drain: true) }
            report["disk_at_start"] = try object(disk.statistics)
            try require("dedicated_store_empty", disk.statistics.entries == 0)
            var previousCache = 0
            try MX.check(mlx_set_cache_limit(&previousCache, 256 * 1024 * 1024), "conversation allocator cache limit")
            defer { var ignored = 0; _ = mlx_set_cache_limit(&ignored, previousCache) }
            let model = try QwenModel(modelDirectory: directory, reservedOutputIDs: tokenizer.reservedOutputTokenIDs,
                stateBudgetBytes: stateBytes) { current, total in
                if current % 8 == 0 || current == total {
                    FileHandle.standardError.write(Data("Conversation cache: loaded \(current)/\(total) layers\n".utf8))
                }
            }
            defer { try? MX.synchronize() }
            let stateAtSystem = try model.estimatedPrefixStateBytes(at: system)
            let stateAtTail = try model.estimatedPrefixStateBytes(at: tail)
            let largestRequest = try fixtures.map { try model.estimatedPrefixStateBytes(at: $0.tokens.count + maximumOutput) * 2 }.max() ?? 0
            var uniqueCheckpoints = [String: Int]()
            for fixture in fixtures {
                for boundary in fixture.plan.publicationTokenCounts {
                    let key = MoETilingBytes.digest(try JSONEncoder().encode(Array(fixture.tokens.prefix(boundary))))
                    uniqueCheckpoints[key] = try model.estimatedPrefixStateBytes(at: boundary)
                }
            }
            report["capacity_preflight"] = ["ram_limit_bytes": ramBytes, "disk_limit_bytes": diskBytes,
                "state_limit_bytes": stateBytes, "system_snapshot_bytes": stateAtSystem,
                "root_tail_snapshot_bytes": stateAtTail, "largest_request_lease_bytes": largestRequest,
                "unique_checkpoint_count": uniqueCheckpoints.count, "all_checkpoint_payload_bytes": uniqueCheckpoints.values.reduce(0, +)]
            try require("capacity_preconditions", stateAtSystem + stateAtTail <= ramBytes && largestRequest * 3 <= stateBytes &&
                largestRequest * 2 + ramBytes + stateAtTail * 2 <= stateBytes && uniqueCheckpoints.count <= 16 &&
                uniqueCheckpoints.values.reduce(0, +) < diskBytes)
            var generator = try QwenGenerator(model: model, prefixCacheLimits: .init(maxEntries: 8, maxBytes: ramBytes), prefixDiskStore: disk)
            defer { generator.prefixStateObserver = nil; try? generator.closePrefixCache(drain: true) }
            let cold = try QwenGenerator(model: model)
            defer { cold.prefixStateObserver = nil }
            var scheduler = try QwenLocalScheduler(generator: generator, limits: .init(executionMode: .cooperative))
            defer { _ = try? scheduler.discardAll() }
            func request(_ id: String, legacy: Int? = nil) -> QwenGenerationRequest {
                let fixture = byID[id]!
                return QwenGenerationRequest(tokens: fixture.tokens, maxTokens: maximumOutput, contextLimit: contextLimit,
                    prefillChunk: chunk, mtpDepth: 0, prefixCacheMaxTokens: legacy, prefixCachePlan: legacy == nil ? fixture.plan : nil)
            }
            var anchors = [String: [Int: CacheReliabilityAnchor]](), oracles = [String: QwenGenerationResult]()
            var oracleReports = [[String: Any]]()
            for fixture in fixtures {
                report["phase"] = "oracle_" + fixture.id; try write()
                var observed = [Int: CacheReliabilityAnchor](), unexpected = [String]()
                cold.prefixStateObserver = { event, state in
                    if event != "coldBoundary" || observed[state.offset] != nil { unexpected.append("\(event):\(state.offset)") }
                    observed[state.offset] = try CacheReliabilityAnchor(state)
                }
                let oracle = try cold.generate(request(fixture.id))
                cold.prefixStateObserver = nil
                let endsInEOS = oracle.tokens.last.map { tokenizer.eosTokenIDs.contains($0) } ?? false
                let noEarlierEOS = oracle.tokens.dropLast().allSatisfy { !tokenizer.eosTokenIDs.contains($0) }
                let nativeStop = oracle.finishReason == .eos ? endsInEOS : oracle.tokens.count == maximumOutput && !endsInEOS
                let validOutput = !oracle.tokens.isEmpty && oracle.tokens.count <= maximumOutput && noEarlierEOS && nativeStop
                let validAnchors = Set(observed.keys) == Set(fixture.plan.publicationTokenCounts) && unexpected.isEmpty &&
                    observed.allSatisfy { $0.key == $0.value.host.offset && $0.value.valid }
                let validOffset = oracle.statistics.finalStateOffset == fixture.tokens.count + oracle.tokens.count - 1
                oracleReports.append(["input_id": fixture.id, "generated_token_ids": oracle.tokens,
                    "finish_reason": oracle.finishReason.rawValue, "text": try tokenizer.decode(oracle.tokens, skipSpecialTokens: true),
                    "result": try object(oracle), "anchors": try object(observed), "anchor_valid": validAnchors,
                    "output_valid": validOutput, "offset_valid": validOffset, "unexpected_events": unexpected,
                    "actual_generated_tokens": oracle.tokens.count, "max_generated_tokens": maximumOutput,
                    "ends_in_eos": endsInEOS, "no_earlier_eos": noEarlierEOS, "native_stop_valid": nativeStop])
                report["cold_oracles"] = oracleReports
                try require("oracle_\(fixture.id)_valid", validAnchors && validOutput && validOffset)
                anchors[fixture.id] = observed; oracles[fixture.id] = oracle
                try require("oracle_\(fixture.id)_leases_released", model.stateBudget.statistics.requestBytes == 0 &&
                    model.stateBudget.statistics.workspaceBytes == 0)
            }
            try require("independent_shared_system_anchors_agree", fixtures.prefix(5).allSatisfy {
                anchors["A"]![system]!.matches(anchors[$0.id]![system]!)
            })
            var phase = "", currentID = "A"
            func select(_ nextPhase: String, _ id: String) {
                phase = nextPhase; currentID = id; report["phase"] = nextPhase; report["current_input_id"] = id
            }
            let stateObserver: (String, QwenModel.State) throws -> Void = { event, state in
                let observed = try CacheReliabilityAnchor(state), fixture = byID[currentID]!
                let candidates = [currentID] + fixtures.map(\.id).filter { $0 != currentID }
                let referenceID = candidates.first { id in
                    anchors[id]?[state.offset] != nil && state.offset <= fixture.tokens.count &&
                    state.offset <= byID[id]!.tokens.count &&
                    Array(fixture.tokens.prefix(state.offset)) == Array(byID[id]!.tokens.prefix(state.offset))
                }
                let exact = referenceID.flatMap { anchors[$0]?[state.offset] }?.matches(observed) ?? false
                let key = "\(phase):\(currentID):\(event):\(state.offset)"
                eventCounts[key, default: 0] += 1
                stateChecks.append(["phase": phase, "input_id": currentID, "event": event, "offset": state.offset,
                    "passed": exact, "reference_input_id": referenceID as Any? ?? NSNull(),
                    "reference_kind": "independent_cache_disabled_prefill", "observed": try object(observed)])
                try write()
                guard exact else { throw CLIError.usage("Conversation native state mismatch at \(key)") }
            }
            generator.prefixStateObserver = stateObserver
            func count(_ phase: String, _ id: String, _ event: String, _ offset: Int) -> Int {
                eventCounts["\(phase):\(id):\(event):\(offset)", default: 0]
            }
            func snapshot(_ label: String) throws {
                report["snapshot_" + label] = ["budget": try object(generator.stateBudgetStatistics),
                    "ram": try object(generator.prefixCacheStatistics), "disk": try object(disk.statistics),
                    "scheduler": try object(scheduler.snapshot())]
                try write()
            }
            func clean(_ label: String, empty: Bool = false) throws {
                try generator.flushPrefixCacheWrites()
                try snapshot(label)
                let budget = generator.stateBudgetStatistics, diskStats = disk.statistics, state = scheduler.snapshot()
                try require(label + "_requests_released", budget.requestBytes == 0 && budget.workspaceBytes == 0 && budget.totalBytes <= budget.maxBytes)
                try require(label + "_queues_drained", diskStats.pendingJobs == 0 && diskStats.pendingBytes == 0 &&
                    generator.prefixCacheStatistics?.liveFlights == 0 && state.isIdle && state.acceptingJobs && state.reservedTokens == 0)
                if empty { try require(label + "_empty", budget.totalBytes == 0 && budget.cacheBytes == 0 &&
                    generator.prefixCacheStatistics?.entries == 0 && diskStats.entries == 0) }
            }
            func record(_ label: String, _ id: String, _ result: QwenGenerationResult, cached: Int, source: String? = nil) throws {
                let reference = oracles[id]!, p = result.phases?.prefill
                let outputExact = result.tokens == reference.tokens && result.finishReason == reference.finishReason
                let cacheExact = p?.cachedTokenCount == cached && p?.computedTokenCount == byID[id]!.tokens.count - cached
                let sourceExact = source.map { p?.cacheSource == $0 } ?? ["memory", "disk"].contains(p?.cacheSource ?? "")
                let accounting = p?.actualForwardTokenCount == p?.computedTokenCount && p?.recomputedTokenCount == 0
                let offsetExact = result.statistics.finalStateOffset == byID[id]!.tokens.count + result.tokens.count - 1
                let passed = outputExact && cacheExact && sourceExact && accounting && offsetExact && result.statistics.mtpDepth == 0
                trials.append(["label": label, "input_id": id, "passed": passed, "expected_cached_tokens": cached,
                    "expected_source": source ?? "memory_or_disk", "output_exact": outputExact, "cache_exact": cacheExact,
                    "source_exact": sourceExact, "accounting_exact": accounting, "offset_exact": offsetExact,
                    "generated_token_ids": result.tokens, "finish_reason": result.finishReason.rawValue,
                    "text": try tokenizer.decode(result.tokens, skipSpecialTokens: true), "result": try object(result)])
                try write()
                guard passed else { throw CLIError.usage("Conversation cache result mismatch: \(label)") }
            }
            func start(_ label: String, _ id: String, cancellation: QwenCancellation? = nil) throws -> QwenPrefillSession {
                select(label, id); return try generator.beginPrefill(request(id), cancellation: cancellation)
            }
            func step(_ label: String, _ id: String, _ session: QwenPrefillSession) throws -> QwenPrefillResult? {
                select(label, id); return try generator.stepPrefill(session)
            }
            func finish(_ label: String, _ id: String, _ session: QwenPrefillSession) throws -> QwenPrefillResult {
                let deadline = Date().addingTimeInterval(600)
                while Date() < deadline {
                    let before = session.processedTokenCount
                    if let ready = try step(label, id, session) { return ready }
                    if session.processedTokenCount == before { Thread.sleep(forTimeInterval: 0.001) }
                }
                throw CLIError.usage("Conversation prefill timed out: \(label)/\(id)")
            }
            func decode(_ label: String, _ id: String, _ ready: QwenPrefillResult) throws -> QwenGenerationResult {
                select(label, id); defer { ready.discard() }; return try generator.decode(ready)
            }
            func cancel(_ label: String, _ id: String, _ session: QwenPrefillSession, _ token: QwenCancellation) throws {
                select(label, id); token.cancel(); var cancelled = false
                do { _ = try generator.stepPrefill(session) } catch { cancelled = error as? QwenGenerationError == .cancelled }
                try session.discard()
                report["cancel_" + label + "_" + id] = ["cancelled": cancelled, "finished": session.isFinished,
                    "processed_tokens": session.processedTokenCount]
                try require(label + "_" + id + "_cancelled", cancelled && session.isFinished)
            }
            func scheduled(_ label: String, _ id: String, legacy: Int? = nil) throws -> QwenGenerationResult {
                select(label, id)
                let job = try scheduler.submit(request(id, legacy: legacy)), deadline = Date().addingTimeInterval(600)
                while Date() < deadline {
                    if let event = try scheduler.runNext() {
                        guard event.jobID == job else { throw CLIError.usage("Unexpected conversation scheduler job") }
                        if let result = event.result { return result }
                        if event.kind == .failed || event.kind == .cancelled {
                            report["scheduler_failure"] = try object(event); try write()
                            throw CLIError.usage("Conversation scheduler failed: \(event.errorDescription ?? event.kind.rawValue)")
                        }
                    } else { Thread.sleep(forTimeInterval: 0.001) }
                }
                throw CLIError.usage("Conversation scheduled request timed out: \(label)")
            }
            func waiting(_ label: String, _ id: String, _ session: QwenPrefillSession) throws {
                let ready = try step(label, id, session)
                defer { ready?.discard() }
                report["waiting_" + label + "_" + id] = ["processed_tokens": session.processedTokenCount,
                    "waiting": session.isWaitingForPrefixCache, "produced_handoff": ready != nil]
                try require(label + "_" + id + "_waits_without_compute", ready == nil && session.processedTokenCount == 0 && session.isWaitingForPrefixCache)
            }

            // C1: old-epoch destruction must not remove the replacement owner.
            try generator.clearPrefixCache(includingDisk: true)
            let oldCancelA = QwenCancellation(), oldCancelB = QwenCancellation()
            let oldA = try start("C1_old", "A", cancellation: oldCancelA)
            defer { try? oldA.discard() }
            let oldB = try start("C1_old", "B", cancellation: oldCancelB)
            defer { try? oldB.discard() }
            try waiting("C1_old", "B", oldB)
            try generator.clearPrefixCache(includingDisk: true)
            let newA = try start("C1", "A")
            defer { try? newA.discard() }
            try snapshot("C1_three_leases_before_old_cancel")
            try cancel("C1_old", "A", oldA, oldCancelA)
            try cancel("C1_old", "B", oldB, oldCancelB)
            let newB = try start("C1", "B")
            defer { try? newB.discard() }
            try waiting("C1", "B", newB)
            var readyA: QwenPrefillResult?, readyB: QwenPrefillResult?
            defer { readyA?.discard(); readyB?.discard() }
            let pairDeadline = Date().addingTimeInterval(600)
            while (readyA == nil || readyB == nil) && Date() < pairDeadline {
                let before = newA.processedTokenCount + newB.processedTokenCount
                if readyA == nil { readyA = try step("C1", "A", newA) }
                if readyB == nil { readyB = try step("C1", "B", newB) }
                if newA.processedTokenCount + newB.processedTokenCount == before { Thread.sleep(forTimeInterval: 0.001) }
            }
            try require("C1_pair_handoff", readyA != nil && readyB != nil)
            try record("C1_new_epoch_producer", "A", decode("C1", "A", readyA!), cached: 0, source: "cold")
            try record("C1_shared_system_waiter", "B", decode("C1", "B", readyB!), cached: system, source: "memory")
            try require("C1_one_system_two_tail_publications", count("C1", "A", "coldBoundary", system) == 1 &&
                count("C1", "B", "coldBoundary", system) == 0 && count("C1", "A", "publish", system) == 1 &&
                count("C1", "B", "restore", system) == 1 && count("C1", "A", "publish", tail) == 1 &&
                count("C1", "B", "publish", b.tail) == 1)
            try clean("C1")

            try record("C2_root_history_replay", "A", scheduled("C2", "A"), cached: tail)
            try clean("C2")

            // C5 runs before new branches so the deep A archive has a simple
            // retention precondition. The legacy request bounds lookup at S.
            try generator.clearPrefixCache()
            try require("C5_ram_cleared_disk_retained", generator.prefixCacheStatistics?.entries == 0 && disk.statistics.entries >= 3)
            try record("C5_shallow_disk_restore", "A", scheduled("C5_shallow", "A", legacy: system), cached: system, source: "disk")
            try clean("C5_shallow")
            try require("C5_only_shallow_ram_entry", generator.prefixCacheStatistics?.entries == 1 && count("C5_shallow", "A", "restore", system) == 1)
            try record("C5_deep_ssd_beats_shallow_ram", "A", scheduled("C5_deep", "A"), cached: tail, source: "disk")
            try clean("C5_deep")

            try record("C3_complete_next_turn", "N1", scheduled("C3", "N1"), cached: tail)
            try clean("C3_N1")
            try record("C3_sibling_branch", "N2", scheduled("C3", "N2"), cached: tail)
            try clean("C3_N2")
            try record("C4_early_history_edit", "E", scheduled("C4", "E"), cached: system)
            try clean("C4_E")
            try record("C4_tool_schema_edit", "T", scheduled("C4", "T"), cached: 0, source: "cold")
            try clean("C4_T")

            // C6: restored states belong to requests. Pause left after import;
            // right owns a decode cursor before cache clear and peer cancel.
            let leftCancel = QwenCancellation(), left = try start("C6_left", "A", cancellation: leftCancel)
            defer { try? left.discard() }
            let restoreDeadline = Date().addingTimeInterval(600)
            while left.processedTokenCount < tail && Date() < restoreDeadline {
                let before = left.processedTokenCount
                let unexpected = try step("C6_left", "A", left)
                defer { unexpected?.discard() }
                try require("C6_left_remains_prefill", unexpected == nil)
                if left.processedTokenCount == before { Thread.sleep(forTimeInterval: 0.001) }
            }
            try require("C6_left_adopted_deep_checkpoint", left.processedTokenCount >= tail && count("C6_left", "A", "restore", tail) == 1)
            let right = try start("C6_right", "A")
            defer { try? right.discard() }
            let rightReady = try finish("C6_right", "A", right)
            defer { rightReady.discard() }
            let rightDecode = try generator.beginDecode(rightReady)
            defer { try? rightDecode.discard() }
            try require("C6_two_private_restores", count("C6_right", "A", "restore", tail) == 1 && rightDecode.generatedTokenCount == 0)
            let beforeClear = generator.stateBudgetStatistics
            try snapshot("C6_before_clear")
            try generator.clearPrefixCache(includingDisk: true)
            try require("C6_clear_keeps_request_leases", generator.stateBudgetStatistics.requestBytes == beforeClear.requestBytes && beforeClear.requestBytes > 0)
            try cancel("C6_left", "A", left, leftCancel)
            try require("C6_cancel_releases_only_left", generator.stateBudgetStatistics.requestBytes > 0 &&
                generator.stateBudgetStatistics.requestBytes < beforeClear.requestBytes)
            select("C6_right", "A")
            var survivor: QwenGenerationResult?
            let decodeDeadline = Date().addingTimeInterval(120)
            while survivor == nil && Date() < decodeDeadline { survivor = try generator.stepDecode(rightDecode) }
            try require("C6_survivor_completed", survivor != nil)
            try record("C6_private_survivor_after_clear_and_cancel", "A", survivor!, cached: tail)
            try clean("C6", empty: true)

            // Keep the same twelve complete comparisons, but use the shipping
            // RAM capacity for C7/C8. No old request/cache lease remains here.
            // The shared disk remains open; only the empty in-memory cache and
            // idle local scheduler are replaced. Pending SSD capacity stays
            // at the explicitly controlled 1 GiB used throughout this probe.
            let defaultRAMBytes = 512 * 1024 * 1024
            report["default_capacity_phase"] = ["starts_at": "C7", "ram_limit_bytes": defaultRAMBytes,
                "disk_max_pending_bytes": ramBytes, "system_snapshot_bytes": stateAtSystem,
                "root_tail_snapshot_bytes": stateAtTail,
                "shared_system_and_tail_exceed_ram": stateAtSystem + stateAtTail > defaultRAMBytes]
            try require("default_capacity_exercises_anchor_retention", stateAtSystem <= defaultRAMBytes &&
                stateAtTail <= defaultRAMBytes && stateAtSystem + stateAtTail > defaultRAMBytes)
            generator.prefixStateObserver = nil
            generator = try QwenGenerator(model: model,
                prefixCacheLimits: .init(maxEntries: 8, maxBytes: defaultRAMBytes), prefixDiskStore: disk)
            scheduler = try QwenLocalScheduler(generator: generator, limits: .init(executionMode: .cooperative))
            generator.prefixStateObserver = stateObserver

            // C7: cancel a valid producer before its shared system checkpoint.
            let beforeSystemCancel = QwenCancellation(), beforeSystem = try start("C7_producer", "A", cancellation: beforeSystemCancel)
            defer { try? beforeSystem.discard() }
            let takeover = try start("C7_waiter", "B")
            defer { try? takeover.discard() }
            try waiting("C7_waiter", "B", takeover)
            let earlyReady = try step("C7_producer", "A", beforeSystem)
            defer { earlyReady?.discard() }
            try require("C7_cancel_before_system", earlyReady == nil && beforeSystem.processedTokenCount == chunk && chunk < system)
            try cancel("C7_producer", "A", beforeSystem, beforeSystemCancel)
            try record("C7_waiter_takes_over_cancelled_producer", "B", decode("C7_waiter", "B", finish("C7_waiter", "B", takeover)), cached: 0, source: "cold")
            try require("C7_waiter_computed_system", count("C7_waiter", "B", "coldBoundary", system) == 1 && count("C7_waiter", "B", "publish", system) == 1)
            try clean("C7")
            let defaultReplayCancel = QwenCancellation()
            let defaultReplay = try start("C7_default_other_tail", "A", cancellation: defaultReplayCancel)
            defer { try? defaultReplay.discard() }
            report["C7_default_other_tail_actual"] = ["processed_tokens": defaultReplay.processedTokenCount,
                "system_tokens": system, "system_restore_events": count("C7_default_other_tail", "A", "restore", system),
                "ram": try object(generator.prefixCacheStatistics)]
            try require("C7_default_capacity_keeps_system_for_other_tail", defaultReplay.processedTokenCount == system &&
                count("C7_default_other_tail", "A", "restore", system) == 1 &&
                (generator.prefixCacheStatistics?.retainedSystemAnchorSkips ?? 0) > 0)
            try cancel("C7_default_other_tail", "A", defaultReplay, defaultReplayCancel)
            try clean("C7_default_other_tail")

            // C8: completed S survives cancellation; the unfinished A tail
            // cannot be published by destruction or borrowed by B.
            try generator.clearPrefixCache(includingDisk: true)
            let afterSystemCancel = QwenCancellation(), afterSystem = try start("C8_producer", "A", cancellation: afterSystemCancel)
            defer { try? afterSystem.discard() }
            let adopting = try start("C8_waiter", "B")
            defer { try? adopting.discard() }
            try waiting("C8_waiter", "B", adopting)
            let systemDeadline = Date().addingTimeInterval(600)
            while afterSystem.processedTokenCount < system && Date() < systemDeadline {
                let unexpected = try step("C8_producer", "A", afterSystem)
                defer { unexpected?.discard() }
                guard unexpected == nil else { throw CLIError.usage("System checkpoint unexpectedly finished the prompt") }
            }
            report["C8_cancel_point"] = ["processed_tokens": afterSystem.processedTokenCount, "system_tokens": system,
                "tail_tokens": tail, "system_publications": count("C8_producer", "A", "publish", system)]
            try require("C8_system_completed_before_cancel", afterSystem.processedTokenCount == system && count("C8_producer", "A", "publish", system) == 1)
            try cancel("C8_producer", "A", afterSystem, afterSystemCancel)
            try record("C8_waiter_keeps_completed_system", "B", decode("C8_waiter", "B", finish("C8_waiter", "B", adopting)), cached: system, source: "memory")
            try require("C8_no_unfinished_tail_publication", count("C8_producer", "A", "publish", tail) == 0 &&
                count("C8_waiter", "B", "restore", system) == 1 && count("C8_waiter", "B", "coldBoundary", system) == 0)
            try clean("C8")
            try generator.clearPrefixCache(includingDisk: true)
            try clean("final", empty: true)
            report["actual_complete_reference_requests"] = oracles.count
            report["actual_complete_cached_requests"] = trials.count
            report["mlx_memory_at_end"] = try object(MX.memory())
            try require("request_set_complete", oracles.count == 6 && trials.count == 12)
            try write(complete: true)
        } catch {
            report["error"] = String(describing: error)
            try? write()
            throw error
        }
    }
}
