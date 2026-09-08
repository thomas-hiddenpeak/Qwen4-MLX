import ANERunnerCore
import ANERunnerGPU
import CMLX
import Darwin
import Foundation

/// Controlled CPU delay, never an actual blocked device. A hold owns no FD.
private final class ReadAdmissionProbeHold: @unchecked Sendable {
    struct Snapshot: Codable {
        let label: String
        let entered, released, watchdogFired, callbackFinished, writeSucceeded: Bool
        let enteredAt, releasedAt: UInt64?
    }
    let label: String
    private let lock = NSLock()
    private let enteredSignal = DispatchSemaphore(value: 0)
    private let releaseSignal = DispatchSemaphore(value: 0)
    private var entered = false, released = false, watchdogFired = false
    private var callbackFinished = false, writeSucceeded = false
    private var enteredAt: UInt64?, releasedAt: UInt64?
    init(label: String) { self.label = label }
    func pauseOnce() throws {
        lock.lock()
        let shouldPause = !entered && !released
        if shouldPause { entered = true; enteredAt = DispatchTime.now().uptimeNanoseconds }
        lock.unlock()
        guard shouldPause else { return }
        enteredSignal.signal()
        if releaseSignal.wait(timeout: .now() + 60) == .timedOut {
            lock.lock(); watchdogFired = true; lock.unlock()
            release()
            throw CLIError.usage("Read admission space gate watchdog fired: \(label)")
        }
    }
    func waitUntilEntered() -> Bool { enteredSignal.wait(timeout: .now() + 5) == .success }
    func release() {
        lock.lock()
        guard !released else { lock.unlock(); return }
        released = true; releasedAt = DispatchTime.now().uptimeNanoseconds
        lock.unlock(); releaseSignal.signal()
    }
    func completed(_ success: Bool) {
        lock.lock(); callbackFinished = true; writeSucceeded = success; lock.unlock()
    }
    var snapshot: Snapshot {
        lock.lock(); defer { lock.unlock() }
        return .init(label: label, entered: entered, released: released,
            watchdogFired: watchdogFired, callbackFinished: callbackFinished,
            writeSucceeded: writeSucceeded, enteredAt: enteredAt, releasedAt: releasedAt)
    }
}

private final class ReadAdmissionProbeGate: @unchecked Sendable {
    private let lock = NSLock()
    private var current: ReadAdmissionProbeHold?
    func arm(_ label: String) throws -> ReadAdmissionProbeHold {
        lock.lock(); defer { lock.unlock() }
        guard current.map({ $0.snapshot.callbackFinished }) ?? true else {
            throw CLIError.usage("Previous read admission hold did not finish")
        }
        let hold = ReadAdmissionProbeHold(label: label); current = hold
        return hold
    }
    func release() {
        lock.lock(); let hold = current; lock.unlock()
        hold?.release()
    }
    func sample(_ fd: Int32) throws -> UInt64 {
        lock.lock(); let hold = current; lock.unlock()
        try hold?.pauseOnce()
        var fs = statvfs()
        guard fstatvfs(fd, &fs) == 0,
              let blocks = UInt64(exactly: fs.f_bavail),
              let fragment = UInt64(exactly: fs.f_frsize), fragment > 0 else {
            throw CLIError.usage("Read admission probe could not sample cache space")
        }
        let (bytes, overflow) = blocks.multipliedReportingOverflow(by: fragment)
        guard !overflow else { throw CLIError.usage("Read admission space sample overflow") }
        return bytes
    }
}

extension RunnerCLI {
    /// Real-model validation for bounded foreground SSD read admission.
    static func probeGPUCacheReadAdmission(_ args: Arguments) throws {
        try args.validate(["--model-dir", "--tokens-file", "--cache-directory", "--output"])
        let output = try args.require("--output")
        guard !FileManager.default.fileExists(atPath: output) else {
            throw CLIError.usage("Read admission probe requires a new output path")
        }
        // 2080 is an original 416 boundary above QSA's 2051 threshold.
        // P=2081 preserves the separate final prompt token and all 121 tensors.
        let promptCount = 2081, prefix = 2080, outputTokens = 16
        let gate = ReadAdmissionProbeGate()
        var holds: [ReadAdmissionProbeHold] = []
        var checks = [String: Bool](), trials = [[String: Any]](), stateChecks = [[String: Any]]()
        var report: [String: Any] = [
            "schema": "qwen38-cache-read-admission-v1", "complete": false, "passed": false,
            "scope": "Real-model AR restoration after bounded foreground read admission; controlled unrelated tiny writes.",
            "notes": [
                "The source token fixture is truncated to 2081 valid IDs; this is a causal-state probe, not a complete chat-template assessment.",
                "A different namespace two-byte metadata/payload write occupies IO admission; it is not a full model publication workspace.",
                "The available-space gate delays a CPU write and has a watchdog; no physical SSD stall or OS pressure is claimed.",
                "All native state comparisons use an independent cache-disabled oracle and preserve BF16 bytes.",
                "State hashing and report writes perturb latency. This is neither a performance benchmark nor long-duration acceptance.",
                "Ready-but-insufficient shared workspace and producer-wait timing are separate CPU/integration cases, not claimed here."]]
        func object<T: Encodable>(_ value: T) throws -> Any {
            try JSONSerialization.jsonObject(with: JSONEncoder().encode(value))
        }
        func save(complete: Bool = false) throws {
            report["complete"] = complete
            report["checks"] = checks; report["trials"] = trials; report["state_checks"] = stateChecks
            report["holds"] = try holds.map { try object($0.snapshot) }
            report["passed"] = complete && !checks.isEmpty && checks.values.allSatisfy { $0 } &&
                trials.count == 7 && trials.allSatisfy { $0["passed"] as? Bool == true } &&
                stateChecks.count == 7 && stateChecks.allSatisfy { $0["passed"] as? Bool == true }
            try emit(report, to: output)
        }
        func require(_ label: String, _ value: Bool) throws {
            checks[label] = value
            try save()
            guard value else { throw CLIError.usage("Read admission check failed: \(label)") }
        }
        func log(_ message: String) {
            FileHandle.standardError.write(Data("Cache read admission: \(message)\n".utf8))
        }
        func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
        func seconds(_ started: UInt64) -> Double { Double(now() - started) * 1e-9 }
        try save()
        do {
            let directory = URL(fileURLWithPath: try args.require("--model-dir")).standardizedFileURL.resolvingSymlinksInPath()
            let inputURL = URL(fileURLWithPath: try args.require("--tokens-file")).standardizedFileURL
            let input = try Data(contentsOf: inputURL)
            let source = try JSONDecoder().decode([Int32].self, from: input)
            try require("source_has_2081_tokens", source.count >= promptCount)
            let prompt = Array(source.prefix(promptCount))
            let executable = Bundle.main.executableURL ?? URL(fileURLWithPath: CommandLine.arguments[0])
            report["provenance"] = [
                "executable_sha256": try MoETilingBytes.hash(executable),
                "config_sha256": try MoETilingBytes.hash(directory.appendingPathComponent("config.json")),
                "tokenizer_sha256": try MoETilingBytes.hash(directory.appendingPathComponent("tokenizer.json")),
                "source_input_sha256": MoETilingBytes.digest(input),
                "source_input_path": inputURL.path,
                "model_cache_identity": try QwenPrefixCacheIdentity.fingerprint(modelDirectory: directory)]
            report["prompt_token_ids"] = prompt
            report["command"] = CommandLine.arguments
            report["configuration"] = ["prompt_tokens": promptCount, "prefix_tokens": prefix,
                "output_tokens": outputTokens, "prefill_chunk": 416, "mtp_depth": 0,
                "ram_cache_bytes": 1, "read_timeout_seconds": 5, "gate_watchdog_seconds": 60]
            let disk = try QwenPrefixDiskStore(directory: URL(fileURLWithPath: try args.require("--cache-directory")),
                limits: .init(maxEntries: 16, maxBytes: 1024 * 1024 * 1024,
                    maxPendingJobs: 2, maxPendingBytes: 512 * 1024 * 1024),
                availableSpace: { try gate.sample($0) })
            defer { gate.release(); _ = disk.close(drain: true, timeout: 5) }
            try require("dedicated_store_empty", disk.statistics.entries == 0)
            var previousCache = 0
            try MX.check(mlx_set_cache_limit(&previousCache, 256 * 1024 * 1024), "read admission allocator cache")
            defer { var ignored = 0; _ = mlx_set_cache_limit(&ignored, previousCache) }
            let model = try QwenModel(modelDirectory: directory) { count, total in
                if count % 8 == 0 || count == total { log("loaded \(count)/\(total) layers") }
            }
            defer { try? MX.synchronize() }
            let generator = try QwenGenerator(model: model,
                prefixCacheLimits: .init(maxEntries: 8, maxBytes: 1, diskRestoreTimeoutSeconds: 5),
                prefixDiskStore: disk)
            let cold = try QwenGenerator(model: model)
            defer { generator.prefixStateObserver = nil; cold.prefixStateObserver = nil }
            let request = QwenGenerationRequest(tokens: prompt, maxTokens: outputTokens,
                contextLimit: 4096, prefillChunk: 416, mtpDepth: 0, prefixCacheMaxTokens: prefix)
            var oracle: CacheReliabilityAnchor?, oracleIDs: [Int32] = [], oracleFinish = ""
            var currentLabel = ""
            func stats() throws -> QwenPrefixCacheStatistics {
                guard let stats = generator.prefixCacheStatistics else { throw CLIError.usage("Missing prefix cache statistics") }
                return stats
            }
            let observer: (String, QwenModel.State) throws -> Void = { event, state in
                guard state.offset == prefix else { throw CLIError.usage("Unexpected read admission state boundary") }
                let observed = try CacheReliabilityAnchor(state)
                if currentLabel == "R0_oracle" {
                    guard event == "coldBoundary", oracle == nil else { throw CLIError.usage("Oracle boundary was not independently cold") }
                    oracle = observed; report["oracle_anchor"] = try object(observed)
                }
                let exact = observed.tensors.count == 121 && oracle?.matches(observed) == true
                stateChecks.append(["trial": currentLabel, "event": event, "passed": exact,
                    "observed": try object(observed)])
                try save()
                guard exact else { throw CLIError.usage("Read admission native state mismatch: \(currentLabel)") }
            }
            generator.prefixStateObserver = observer; cold.prefixStateObserver = observer
            func select(_ label: String) { currentLabel = label; log("starting \(label)") }
            func snapshot(_ label: String) throws {
                report["snapshot_" + label] = ["uptime_ns": now(), "disk": try object(disk.statistics),
                    "cache": try object(stats()), "budget": try object(model.stateBudget.statistics)]
                try save()
            }
            func drained(_ label: String) throws {
                let started = now()
                while true {
                    let d = disk.statistics, b = model.stateBudget.statistics
                    if d.pendingJobs == 0 && d.pendingBytes == 0 && b.workspaceBytes == 0 &&
                        holds.allSatisfy({ $0.snapshot.callbackFinished }) { break }
                    guard seconds(started) < 15 else { throw CLIError.usage("Read admission drain timed out: \(label)") }
                    Thread.sleep(forTimeInterval: 0.002)
                }
                try snapshot(label)
                let d = disk.statistics, b = model.stateBudget.statistics
                try require(label + "_all_owners_released", d.foregroundReadIntents == 0 &&
                    b.requestBytes == 0 && b.cacheBytes == 0 && b.workspaceBytes == 0 &&
                    b.currentLeases == 0 && b.totalBytes == 0 && (try stats()).liveFlights == 0)
                try require(label + "_no_store_failure", d.writeFailures == 0 && d.corruptions == 0 &&
                    !d.storageUnavailable && holds.allSatisfy { !$0.snapshot.watchdogFired && $0.snapshot.writeSucceeded })
            }
            func finish(_ label: String, _ session: QwenPrefillSession) throws -> QwenGenerationResult {
                currentLabel = label
                let started = now()
                var prepared: QwenPrefillResult?, decoding: QwenDecodeSession?
                defer { prepared?.discard(); try? decoding?.discard(); try? session.discard() }
                while prepared == nil {
                    guard seconds(started) < 180 else { throw CLIError.usage("Read admission prefill deadline: \(label)") }
                    let before = session.processedTokenCount
                    prepared = try generator.stepPrefill(session)
                    if prepared == nil && session.processedTokenCount == before { Thread.sleep(forTimeInterval: 0.001) }
                }
                decoding = try generator.beginDecode(prepared!)
                while true {
                    guard seconds(started) < 180 else { throw CLIError.usage("Read admission decode deadline: \(label)") }
                    if let result = try generator.stepDecode(decoding!) { return result }
                }
            }
            func record(_ label: String, _ result: QwenGenerationResult, source: String, cached: Int) throws {
                if label == "R0_oracle" { oracleIDs = result.tokens; oracleFinish = result.finishReason.rawValue }
                let p = result.phases?.prefill
                let exact = !oracleIDs.isEmpty && result.tokens == oracleIDs && result.finishReason.rawValue == oracleFinish
                let counts = (p?.cachedTokenCount ?? 0) == cached &&
                    (p?.computedTokenCount ?? promptCount) == promptCount - cached &&
                    (p?.actualForwardTokenCount ?? promptCount - cached) == promptCount - cached &&
                    (p?.recomputedTokenCount ?? 0) == 0
                let boundary = stateChecks.filter { $0["trial"] as? String == label }.count == 1
                let passed = exact && counts && boundary && (p?.cacheSource ?? "cold") == source &&
                    result.statistics.finalStateOffset == promptCount + result.tokens.count - 1 && result.statistics.mtpDepth == 0
                trials.append(["label": label, "passed": passed, "generated_token_ids": result.tokens,
                    "output_exact": exact, "token_accounting_exact": counts, "one_state_boundary": boundary,
                    "result": try object(result)])
                try save()
                guard passed else { throw CLIError.usage("Read admission result mismatch: \(label)") }
            }
            func block(_ label: String) throws -> ReadAdmissionProbeHold {
                try drained(label + "_before")
                let hold = try gate.arm(label); holds.append(hold)
                let accepted = disk.enqueue(tokens: [17], namespace: "probe.unrelated.write." + label,
                    metadata: Data([1]), payload: Data([2]), completion: { hold.completed($0) })
                guard accepted else { hold.release(); throw CLIError.usage("Could not admit unrelated blocker") }
                try require(label + "_write_gate_entered", hold.waitUntilEntered())
                let d = disk.statistics
                try require(label + "_small_write_owns_io", d.pendingJobs == 1 && d.pendingBytes == 2 && !hold.snapshot.released)
                return hold
            }
            func waiting(_ label: String, cancellation: QwenCancellation? = nil) throws -> QwenPrefillSession {
                select(label)
                let before = disk.statistics
                let session = try generator.beginPrefill(request, cancellation: cancellation)
                do {
                    let d = disk.statistics, b = model.stateBudget.statistics
                    try require(label + "_metadata_only_wait", session.isWaitingForPrefixCache &&
                        session.processedTokenCount == 0 && d.foregroundReadIntents == 1 &&
                        d.pendingJobs == before.pendingJobs && d.pendingBytes == before.pendingBytes &&
                        d.bytesRead == before.bytesRead && b.workspaceBytes == 0 && b.requestBytes > 0)
                    return session
                } catch { try? session.discard(); throw error }
            }

            select("R0_oracle")
            let reference = try cold.generate(request)
            try record("R0_oracle", reference, source: "cold", cached: 0)
            try require("oracle_covers_qsa_and_all_state", oracle?.tensors.count == 121)
            try drained("R0")
            select("R1_seed")
            let seed = try generator.beginPrefill(request)
            try record("R1_seed", finish("R1_seed", seed), source: "cold", cached: 0)
            try drained("R1")
            try require("real_archive_published", disk.statistics.entries == 1 && disk.statistics.published == 1 &&
                disk.statistics.diskBytes > 32 * 1024 * 1024)

            // R2: accepted priority owns no workspace while the old tiny write
            // still has its real charge. Releasing it permits the actual archive read.
            let released = try block("R2_release")
            let beforeRelease = try stats()
            let releaseReader = try waiting("R2_release")
            defer { try? releaseReader.discard() }
            let priorityBefore = disk.statistics.optionalWritePriorityRejections ?? 0
            try require("R2_new_optional_write_rejected", !disk.enqueue(tokens: [18], namespace: "probe.rejected.optional",
                metadata: Data([1]), payload: Data([2])) &&
                disk.statistics.optionalWritePriorityRejections == priorityBefore + 1)
            try snapshot("R2_before_release")
            released.release()
            try record("R2_release", finish("R2_release", releaseReader), source: "disk", cached: prefix)
            try require("R2_one_real_restore", try stats().diskHits == beforeRelease.diskHits + 1)
            try drained("R2")

            // R3: keep the write held until the full 5-second admission wait
            // expires. The cold request must not erase the unrelated IO owner.
            let timed = try block("R3_timeout")
            let beforeTimeout = try stats()
            let beforeTimeoutDisk = disk.statistics, beforeTimeoutBudget = model.stateBudget.statistics
            let timeoutStarted = now()
            let timeoutReader = try waiting("R3_timeout")
            defer { try? timeoutReader.discard() }
            let timeoutResult = try finish("R3_timeout", timeoutReader)
            try record("R3_timeout", timeoutResult, source: "cold", cached: 0)
            let afterTimeout = try stats(), stillHeld = disk.statistics
            let timing = timeoutResult.phases?.prefill
            let resolutionTimings = [timing?.cacheWaitSeconds ?? 0,
                timing?.cacheLookupSeconds ?? 0, timing?.cacheRestoreSeconds ?? 0]
            // In this case the held unrelated writer prevents any restore.
            // The admission deadline includes synchronous lookup work as well
            // as suspended wait; cacheWaitSeconds alone excludes that work.
            let resolutionSeconds = resolutionTimings.reduce(0, +)
            report["R3_timeout_evidence"] = [
                "before_cache": try object(beforeTimeout), "after_cache": try object(afterTimeout),
                "before_disk": try object(beforeTimeoutDisk), "after_disk": try object(stillHeld),
                "before_budget": try object(beforeTimeoutBudget), "after_budget": try object(model.stateBudget.statistics),
                "gate": try object(timed.snapshot), "elapsed_through_generation_seconds": seconds(timeoutStarted),
                "cache_wait_seconds": resolutionTimings[0], "cache_lookup_seconds": resolutionTimings[1],
                "cache_restore_seconds": resolutionTimings[2], "cache_resolution_seconds": resolutionSeconds,
                "timeout_delta": afterTimeout.diskReadTimeouts - beforeTimeout.diskReadTimeouts,
                "disk_hit_delta": afterTimeout.diskHits - beforeTimeout.diskHits,
                "bytes_read_delta": stillHeld.bytesRead - beforeTimeoutDisk.bytesRead]
            try save()
            try require("R3_one_admission_timeout", afterTimeout.diskReadTimeouts == beforeTimeout.diskReadTimeouts + 1)
            try require("R3_no_archive_read", afterTimeout.diskHits == beforeTimeout.diskHits &&
                stillHeld.bytesRead == beforeTimeoutDisk.bytesRead)
            try require("R3_5s_admission_resolution", resolutionTimings.allSatisfy { $0.isFinite && $0 >= 0 } &&
                resolutionSeconds >= 4.999)
            try require("R3_old_write_still_owned", !timed.snapshot.released && !timed.snapshot.watchdogFired &&
                stillHeld.foregroundReadIntents == 0 && stillHeld.pendingJobs == 1 && stillHeld.pendingBytes == 2 &&
                model.stateBudget.statistics.workspaceBytes == 0)
            timed.release(); try drained("R3")

            // R4: cancellation before read acceptance releases metadata only;
            // a new waiter may immediately hold the store priority.
            let cancelledHold = try block("R4_cancel")
            let cancellation = QwenCancellation()
            let cancelled = try waiting("R4_cancelled_waiter", cancellation: cancellation)
            cancellation.cancel()
            var sawCancellation = false
            do { _ = try generator.stepPrefill(cancelled) }
            catch { sawCancellation = error as? QwenGenerationError == .cancelled }
            try cancelled.discard()
            try require("R4_cancel_released_intent", sawCancellation && disk.statistics.foregroundReadIntents == 0 &&
                model.stateBudget.statistics.workspaceBytes == 0 && model.stateBudget.statistics.requestBytes == 0)
            let successor = try waiting("R4_successor")
            defer { try? successor.discard() }
            try cancelled.discard()
            try require("R4_old_discard_keeps_new_intent", disk.statistics.foregroundReadIntents == 1)
            cancelledHold.release()
            try record("R4_successor", finish("R4_successor", successor), source: "disk", cached: prefix)
            try drained("R4")

            // R5: RAM-only clear must revoke this cache's metadata intent even
            // though it deliberately leaves the underlying archive and write.
            let clearedHold = try block("R5_clear")
            let stale = try waiting("R5_stale_waiter")
            defer { try? stale.discard() }
            try generator.clearPrefixCache(includingDisk: false)
            try require("R5_clear_revoked_old_intent", disk.statistics.foregroundReadIntents == 0 &&
                disk.statistics.pendingJobs == 1 && disk.statistics.pendingBytes == 2)
            let fresh = try waiting("R5_fresh")
            defer { try? fresh.discard() }
            try stale.discard()
            try require("R5_stale_owner_cannot_clear_fresh_intent", disk.statistics.foregroundReadIntents == 1)
            clearedHold.release()
            try record("R5_fresh", finish("R5_fresh", fresh), source: "disk", cached: prefix)
            try drained("R5")

            // R6: the compatibility whole-stage call must choose cold without
            // synchronously waiting for the IO gate. Its actual prefill still runs.
            let wholeHold = try block("R6_whole")
            select("R6_whole")
            let beforeWhole = try stats()
            let prepared = try generator.prefill(request)
            defer { prepared.discard() }
            try require("R6_prefill_completed_while_write_held", !wholeHold.snapshot.released &&
                !wholeHold.snapshot.watchdogFired && disk.statistics.foregroundReadIntents == 0)
            let whole = try generator.decode(prepared)
            try record("R6_whole", whole, source: "cold", cached: 0)
            try require("R6_no_read_or_timeout", try stats().diskHits == beforeWhole.diskHits &&
                stats().diskReadTimeouts == beforeWhole.diskReadTimeouts)
            wholeHold.release(); try drained("R6")
            let closed = disk.close(drain: true, timeout: 10)
            report["close"] = ["io_completed": closed.ioCompleted, "callbacks_completed": closed.callbacksCompleted]
            try require("finite_close_completed", closed.completed)
            try require("finite_request_and_state_counts", trials.count == 7 && stateChecks.count == 7)
            try save(complete: true)
        } catch {
            gate.release()
            report["error"] = String(describing: error)
            try? save()
            throw error
        }
    }
}
