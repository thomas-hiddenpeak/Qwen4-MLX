import ANERunnerCore
import ANERunnerGPU
import CMLX
import Darwin
import Foundation

/// A one-shot delay of the existing space sampler, not a blocked physical
/// device. It always releases after its watchdog, and never touches MLX.
private final class CacheTimeoutSpaceGate: @unchecked Sendable {
    struct Snapshot: Codable {
        let armed, entered, released, watchdogFired: Bool
        let sampleCalls: Int
        let enteredAt, releasedAt: UInt64?
        let maximumWaitSeconds: TimeInterval
    }
    private let lock = NSLock()
    private let enteredSignal = DispatchSemaphore(value: 0)
    private let releaseSignal = DispatchSemaphore(value: 0)
    private let maximumWaitSeconds: TimeInterval
    private var armed = false, entered = false, released = false, watchdogFired = false
    private var sampleCalls = 0
    private var enteredAt: UInt64?, releasedAt: UInt64?

    init(maximumWaitSeconds: TimeInterval) { self.maximumWaitSeconds = maximumWaitSeconds }

    func arm() throws {
        lock.lock(); defer { lock.unlock() }
        guard !armed, !entered, !released else { throw CLIError.usage("Space delay gate can only be armed once") }
        armed = true
    }

    func release() {
        lock.lock()
        guard !released else { lock.unlock(); return }
        released = true; releasedAt = DispatchTime.now().uptimeNanoseconds
        lock.unlock()
        releaseSignal.signal()
    }

    func waitUntilEntered(seconds: TimeInterval) -> Bool {
        enteredSignal.wait(timeout: .now() + seconds) == .success
    }

    var snapshot: Snapshot {
        lock.lock(); defer { lock.unlock() }
        return .init(armed: armed, entered: entered, released: released, watchdogFired: watchdogFired,
                     sampleCalls: sampleCalls, enteredAt: enteredAt, releasedAt: releasedAt,
                     maximumWaitSeconds: maximumWaitSeconds)
    }

    func sample(_ directoryFD: Int32) throws -> UInt64 {
        lock.lock()
        sampleCalls += 1
        let delay = armed && !entered && !released
        if delay { armed = false; entered = true; enteredAt = DispatchTime.now().uptimeNanoseconds }
        lock.unlock()
        if delay {
            enteredSignal.signal()
            if releaseSignal.wait(timeout: .now() + maximumWaitSeconds) == .timedOut {
                lock.lock()
                watchdogFired = true; released = true
                releasedAt = DispatchTime.now().uptimeNanoseconds
                lock.unlock()
                throw CLIError.usage("Space delay gate watchdog fired")
            }
        }
        // Keep real volume-space protection after the controlled delay. The
        // hook neither reports infinite space nor creates fake disk pressure.
        var fs = statvfs()
        guard fstatvfs(directoryFD, &fs) == 0,
              let blocks = UInt64(exactly: fs.f_bavail),
              let fragment = UInt64(exactly: fs.f_frsize), fragment > 0 else {
            throw CLIError.usage("Space delay probe could not sample its cache volume")
        }
        let (bytes, overflow) = blocks.multipliedReportingOverflow(by: fragment)
        guard !overflow else { throw CLIError.usage("Cache volume-space sample overflow") }
        return bytes
    }
}

extension RunnerCLI {
    /// Six AR requests on the first successful read-timeout race; at most eight
    /// if the callback wins early. Native state readback perturbs all timings.
    static func probeGPUCacheTimeouts(_ args: Arguments) throws {
        try args.validate(["--model-dir", "--tokens-file", "--cache-directory", "--output"])
        let output = try args.require("--output")
        guard !FileManager.default.fileExists(atPath: output) else {
            throw CLIError.usage("Cache timeout probe requires a new output path")
        }
        let prefix = 9_984, outputTokens = 16
        let readTimeout = 0.000_001, normalTimeout = 5.0, gateTimeout = 60.0
        let gate = CacheTimeoutSpaceGate(maximumWaitSeconds: gateTimeout)
        var report: [String: Any] = [
            "schema": "qwen38-cache-timeouts-v1", "complete": false, "passed": false,
            "scope": "Real-model AR state/output regression with a real read-timeout race and controlled publication delay.",
            "notes": [
                "The available-space hook delays one CPU write task; it does not block or fill a real disk.",
                "A read reserves the full pending byte allowance; blocking a write cannot queue that read behind it.",
                "Only diskReadTimeouts proves a real read timeout. Early ready callbacks are correct but do not cover it.",
                "CPU ticket tests establish deterministic lifetime boundaries; independent ledger samples are not atomic together.",
                "State hashing changes latency. This is not a throughput benchmark, real bad-disk test, or long-duration gate."]]
        var checks = [String: Bool](), trials = [[String: Any]](), stateChecks = [[String: Any]]()
        var completedAll = false, readTimeoutCovered = false
        var oracleAnchor: CacheReliabilityAnchor?, oracleIDs = [Int32](), oracleFinish = ""
        func object<T: Encodable>(_ value: T) throws -> Any {
            try JSONSerialization.jsonObject(with: JSONEncoder().encode(value))
        }
        func save(complete: Bool = false) throws {
            report["complete"] = complete
            report["read_timeout_covered"] = readTimeoutCovered
            report["checks"] = checks; report["trials"] = trials; report["state_checks"] = stateChecks
            report["gate"] = try object(gate.snapshot)
            report["passed"] = complete && readTimeoutCovered && !checks.isEmpty && !trials.isEmpty &&
                checks.values.allSatisfy { $0 } && trials.allSatisfy { $0["passed"] as? Bool == true } &&
                !stateChecks.isEmpty && stateChecks.allSatisfy { $0["passed"] as? Bool == true }
            try emit(report, to: output)
        }
        func require(_ label: String, _ value: Bool) throws {
            checks[label] = value
            try save()
            guard value else { throw CLIError.usage("Cache timeout check failed: \(label)") }
        }
        func log(_ message: String) {
            FileHandle.standardError.write(Data("Cache timeout probe: \(message)\n".utf8))
        }
        try save()
        do {
            let modelDirectory = URL(fileURLWithPath: try args.require("--model-dir"))
                .standardizedFileURL.resolvingSymlinksInPath()
            let inputURL = URL(fileURLWithPath: try args.require("--tokens-file")).standardizedFileURL
            let input = try Data(contentsOf: inputURL)
            let prompt = try JSONDecoder().decode([Int32].self, from: input)
            try require("long_agent_fixture", prompt.count == 11_057)
            let executable = Bundle.main.executableURL ?? URL(fileURLWithPath: CommandLine.arguments[0])
            report["provenance"] = [
                "executable_sha256": try MoETilingBytes.hash(executable),
                "config_sha256": try MoETilingBytes.hash(modelDirectory.appendingPathComponent("config.json")),
                "tokenizer_sha256": try MoETilingBytes.hash(modelDirectory.appendingPathComponent("tokenizer.json")),
                "input_sha256": MoETilingBytes.digest(input),
                "model_cache_identity": try QwenPrefixCacheIdentity.fingerprint(modelDirectory: modelDirectory)]
            report["command"] = CommandLine.arguments
            report["model_directory"] = modelDirectory.path
            report["source_prompt_token_ids"] = prompt
            report["configuration"] = [
                "prefix_tokens": prefix, "output_tokens": outputTokens, "prefill_chunk": 416,
                "context_limit": 16_384, "mtp_depth": 0, "ram_cache_bytes": 1,
                "read_timeout_seconds": readTimeout, "normal_timeout_seconds": normalTimeout,
                "maximum_read_attempts": 3, "request_deadline_seconds": 180,
                "space_gate_watchdog_seconds": gateTimeout] as [String: Any]
            let disk = try QwenPrefixDiskStore(
                directory: URL(fileURLWithPath: try args.require("--cache-directory")),
                limits: .init(maxEntries: 8, maxBytes: 2 * 1024 * 1024 * 1024,
                              maxPendingJobs: 2, maxPendingBytes: 512 * 1024 * 1024),
                availableSpace: { try gate.sample($0) })
            // Registered before any request: all throw paths release the
            // artificial gate before a synchronous drain/close can wait on it.
            defer { gate.release(); disk.close(drain: true) }
            report["disk_at_start"] = try object(disk.statistics)
            report["disk_limits"] = try object(disk.limits)
            try require("dedicated_store_empty", disk.statistics.entries == 0)
            var previousCache = 0
            try MX.check(mlx_set_cache_limit(&previousCache, 256 * 1024 * 1024), "timeout probe allocator cache")
            defer { var ignored = 0; _ = mlx_set_cache_limit(&ignored, previousCache) }
            let model = try QwenModel(modelDirectory: modelDirectory) { count, total in
                if count % 8 == 0 || count == total { log("loaded \(count)/\(total) layers") }
            }
            defer { try? MX.synchronize() }
            let normal = try QwenGenerator(model: model, prefixCacheLimits:
                .init(maxEntries: 8, maxBytes: 1, diskRestoreTimeoutSeconds: normalTimeout), prefixDiskStore: disk)
            let shortDeadline = try QwenGenerator(model: model, prefixCacheLimits:
                .init(maxEntries: 8, maxBytes: 1, diskRestoreTimeoutSeconds: readTimeout), prefixDiskStore: disk)
            let request = QwenGenerationRequest(tokens: prompt, maxTokens: outputTokens,
                contextLimit: 16_384, prefillChunk: 416, mtpDepth: 0, prefixCacheMaxTokens: prefix)
            report["state_budget_limit"] = model.stateBudget.maxBytes
            try require("full_model", model.layerCount == 48)

            func cacheStats(_ generator: QwenGenerator) throws -> QwenPrefixCacheStatistics {
                guard let stats = generator.prefixCacheStatistics else {
                    throw CLIError.usage("Timeout probe requires cache statistics")
                }
                return stats
            }
            func stableSnapshot(_ label: String) throws {
                report["snapshot_" + label] = [
                    "uptime_ns": DispatchTime.now().uptimeNanoseconds,
                    "budget": try object(model.stateBudget.statistics),
                    "disk": try object(disk.statistics),
                    "normal_cache": try object(cacheStats(normal)),
                    "short_deadline_cache": try object(cacheStats(shortDeadline)),
                    "gate": try object(gate.snapshot)]
                try save()
            }
            func drained(_ label: String) throws {
                disk.flush()
                try stableSnapshot(label)
                let budget = model.stateBudget.statistics, store = disk.statistics
                try require(label + "_request_workspace_cache_released",
                            budget.requestBytes == 0 && budget.workspaceBytes == 0 && budget.cacheBytes == 0 &&
                            budget.totalBytes == 0 && budget.currentLeases == 0)
                try require(label + "_pending_released", store.pendingJobs == 0 && store.pendingBytes == 0)
                try require(label + "_no_store_failure", store.writeFailures == 0 && store.corruptions == 0 &&
                            !store.storageUnavailable && store.spaceRejections == 0 && store.spaceQueryFailures == 0)
                try require(label + "_no_live_leader", try cacheStats(normal).liveFlights == 0 &&
                            cacheStats(shortDeadline).liveFlights == 0)
            }
            func run(_ label: String, generator: QwenGenerator,
                     requirePublicationWait: Bool = false) throws -> QwenGenerationResult {
                log("starting \(label)")
                let stateCountBefore = stateChecks.count
                generator.prefixStateObserver = { event, state in
                    guard state.offset == prefix else { throw CLIError.usage("Unexpected timeout-probe state boundary") }
                    let observed = try CacheReliabilityAnchor(state)
                    if oracleAnchor == nil {
                        guard label == "R1_cold_oracle", event == "coldBoundary", observed.valid else {
                            throw CLIError.usage("Timeout oracle must come from the first cold boundary")
                        }
                        oracleAnchor = observed
                        report["oracle_anchor"] = try object(observed)
                    }
                    let exact = oracleAnchor?.matches(observed) == true
                    stateChecks.append(["trial": label, "event": event, "passed": exact,
                        "offset": state.offset, "tensor_count": observed.tensors.count,
                        "observed": try object(observed)])
                    try save()
                    guard exact else { throw CLIError.usage("Timeout path native state differs from cold oracle") }
                }
                defer { generator.prefixStateObserver = nil }
                let cancellation = QwenCancellation()
                let started = DispatchTime.now().uptimeNanoseconds
                let session = try generator.beginPrefill(request, cancellation: cancellation)
                var prepared: QwenPrefillResult?, decodeSession: QwenDecodeSession?
                var completedRequest = false
                defer {
                    // A failed request must not leave the fixture's writer
                    // held while any cursor cleanup below joins async work.
                    if !completedRequest { gate.release() }
                    cancellation.cancel()
                    try? decodeSession?.discard()
                    prepared?.discard()
                    try? session.discard()
                }
                // Only lightweight thread-safe sampling before the immediate
                // first step; JSON/state hashing would unnecessarily let the
                // real read callback win the short-deadline race.
                let waitingAtBegin = session.isWaitingForPrefixCache
                let processedAtBegin = session.processedTokenCount
                let beginBudget = model.stateBudget.statistics, beginDisk = disk.statistics
                prepared = try generator.stepPrefill(session)
                report["begin_" + label] = [
                    "waiting_prefix": waitingAtBegin, "processed_tokens": processedAtBegin,
                    "budget": try object(beginBudget), "disk": try object(beginDisk),
                    "first_step_processed_tokens": session.processedTokenCount]
                if requirePublicationWait {
                    try require(label + "_entered_publication_wait", waitingAtBegin && processedAtBegin == 0)
                    try require(label + "_initial_owner_charged", beginBudget.workspaceBytes > 0 &&
                                beginBudget.requestBytes > 0 && beginDisk.pendingJobs == 1 && beginDisk.pendingBytes > 0)
                }
                func checkDeadline() throws {
                    let elapsed = Double(DispatchTime.now().uptimeNanoseconds - started) * 1e-9
                    guard elapsed < 180 else { throw CLIError.usage("Timeout probe request exceeded 180 seconds: \(label)") }
                    guard !gate.snapshot.watchdogFired else { throw CLIError.usage("Publication delay watchdog fired") }
                }
                while prepared == nil {
                    try checkDeadline()
                    let before = session.processedTokenCount
                    prepared = try generator.stepPrefill(session)
                    if prepared == nil, session.processedTokenCount == before { Thread.sleep(forTimeInterval: 0.001) }
                }
                decodeSession = try generator.beginDecode(prepared!, cancellation: cancellation)
                while true {
                    try checkDeadline()
                    if let result = try generator.stepDecode(decodeSession!) {
                        try require(label + "_one_exact_state_boundary", stateChecks.count == stateCountBefore + 1)
                        log("finished \(label) source=\(result.phases?.prefill.cacheSource ?? "unknown")")
                        completedRequest = true
                        return result
                    }
                }
            }
            func record(_ label: String, _ result: QwenGenerationResult, source: String, cached: Int) throws {
                if oracleIDs.isEmpty {
                    guard label == "R1_cold_oracle" else { throw CLIError.usage("Missing timeout output oracle") }
                    oracleIDs = result.tokens; oracleFinish = result.finishReason.rawValue
                    report["oracle_token_ids"] = oracleIDs; report["oracle_finish_reason"] = oracleFinish
                    try require("nontrivial_oracle", oracleIDs.count == outputTokens)
                }
                let prefill = result.phases?.prefill
                let exact = result.tokens == oracleIDs && result.finishReason.rawValue == oracleFinish
                let stateOffset = result.statistics.finalStateOffset == prompt.count + result.tokens.count - 1
                let cacheExact = prefill?.cacheSource == source && prefill?.cachedTokenCount == cached &&
                    prefill?.computedTokenCount == prompt.count - cached
                let passed = exact && stateOffset && cacheExact && result.statistics.mtpDepth == 0
                trials.append(["label": label, "passed": passed, "output_exact": exact,
                    "cache_exact": cacheExact, "offset_exact": stateOffset,
                    "generated_token_ids": result.tokens, "result": try object(result),
                    "budget_after_return": try object(model.stateBudget.statistics),
                    "disk_after_return": try object(disk.statistics)])
                try save()
                guard passed else { throw CLIError.usage("Timeout trial failed: \(label)") }
            }

            let seed = try run("R1_cold_oracle", generator: normal)
            try record("R1_cold_oracle", seed, source: "cold", cached: 0)
            try drained("R1")
            try require("R1_archive_durable", disk.statistics.entries == 1 && disk.statistics.published == 1 &&
                        disk.statistics.diskBytes > 64 * 1024 * 1024)

            for attempt in 1...3 {
                let label = "R2_read_attempt_\(attempt)"
                let before = try cacheStats(shortDeadline)
                let result = try run(label, generator: shortDeadline)
                let after = try cacheStats(shortDeadline)
                let timeoutDelta = after.diskReadTimeouts - before.diskReadTimeouts
                let hitDelta = after.diskHits - before.diskHits
                report["race_" + label] = ["read_timeout_delta": timeoutDelta, "disk_hit_delta": hitDelta,
                    "publication_timeout_delta": after.diskPublicationTimeouts - before.diskPublicationTimeouts,
                    "coverage": timeoutDelta == 1 ? "read_timeout" : "ready_callback_won"]
                try require(label + "_unambiguous_read_path", (timeoutDelta == 1 && hitDelta == 0) ||
                            (timeoutDelta == 0 && hitDelta == 1))
                try require(label + "_not_publication_timeout", after.diskPublicationTimeouts == before.diskPublicationTimeouts)
                try record(label, result, source: timeoutDelta == 1 ? "cold" : "disk", cached: timeoutDelta == 1 ? 0 : prefix)
                try drained(label)
                if timeoutDelta == 1 { readTimeoutCovered = true; break }
            }
            checks["real_read_timeout_covered"] = readTimeoutCovered
            try save()
            let beforeHit = try cacheStats(normal).diskHits
            let restored = try run("R3_normal_read", generator: normal)
            try record("R3_normal_read", restored, source: "disk", cached: prefix)
            try require("R3_effective_disk_hit", try cacheStats(normal).diskHits == beforeHit + 1)
            try drained("R3")

            try normal.clearPrefixCache(includingDisk: true)
            try drained("publication_setup")
            try require("publication_store_empty", disk.statistics.entries == 0)
            let publishedBefore = disk.statistics.published
            try gate.arm()
            let producer = try run("R4_publication_producer", generator: normal)
            try record("R4_publication_producer", producer, source: "cold", cached: 0)
            try require("R4_space_hook_entered", gate.waitUntilEntered(seconds: 5))
            let heldBudget = model.stateBudget.statistics, heldDisk = disk.statistics
            try stableSnapshot("R4_original_publication_pending")
            try require("R4_request_finished_but_publication_owned", heldBudget.requestBytes == 0 &&
                        heldBudget.cacheBytes == 0 && heldBudget.workspaceBytes > 0 &&
                        heldDisk.pendingJobs == 1 && heldDisk.pendingBytes > 0 && heldDisk.entries == 0 &&
                        heldDisk.published == publishedBefore && !gate.snapshot.released)
            let beforeWait = try cacheStats(normal)
            let follower = try run("R5_publication_timeout", generator: normal, requirePublicationWait: true)
            try record("R5_publication_timeout", follower, source: "cold", cached: 0)
            let afterWait = try cacheStats(normal)
            let stillHeld = model.stateBudget.statistics, stillPending = disk.statistics
            try stableSnapshot("R5_before_gate_release")
            try require("R5_publication_timeout_only", afterWait.diskPublicationTimeouts == beforeWait.diskPublicationTimeouts + 1 &&
                        afterWait.diskReadTimeouts == beforeWait.diskReadTimeouts)
            try require("R5_original_owner_still_charged", !gate.snapshot.released && !gate.snapshot.watchdogFired &&
                        stillHeld.requestBytes == 0 && stillHeld.cacheBytes == 0 &&
                        stillHeld.workspaceBytes == heldBudget.workspaceBytes &&
                        stillHeld.currentLeases == heldBudget.currentLeases &&
                        stillPending.pendingJobs == 1 && stillPending.pendingBytes == heldDisk.pendingBytes)
            try require("R5_no_duplicate_write_attempt", stillPending.published == heldDisk.published &&
                        stillPending.rejected == heldDisk.rejected && stillPending.entries == 0 &&
                        afterWait.duplicateSkipped == beforeWait.duplicateSkipped + 1)
            gate.release()
            try drained("publication_release")
            try require("original_publication_completed_once", disk.statistics.published == publishedBefore + 1 &&
                        disk.statistics.entries == 1 && !gate.snapshot.watchdogFired)
            let beforeFinalHit = try cacheStats(normal).diskHits
            let final = try run("R6_after_publication", generator: normal)
            try record("R6_after_publication", final, source: "disk", cached: prefix)
            try require("R6_effective_disk_hit", try cacheStats(normal).diskHits == beforeFinalHit + 1)
            try drained("R6_final")
            try require("bounded_gpu_request_count", (6...8).contains(trials.count))
            completedAll = true
            try save(complete: true)
            guard readTimeoutCovered else {
                throw CLIError.usage("Read callback won all three races; real read-timeout coverage remains unverified")
            }
        } catch {
            // This is also safe if model setup failed before the gate was armed.
            gate.release()
            report["error"] = String(describing: error)
            try? save(complete: completedAll)
            throw error
        }
    }
}
