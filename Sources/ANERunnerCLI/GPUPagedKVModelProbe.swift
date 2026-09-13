import ANERunnerCore
import ANERunnerGPU
import CMLX
import Foundation

extension RunnerCLI {
    /// Full-model correctness and lifetime gate. Every state observation is an
    /// explicit compact diagnostic export; its cost is not a throughput result.
    static func probeGPUPagedKVModel(_ args: Arguments) throws {
        try args.validate(["--model-dir", "--tokens-file", "--library", "--max-tokens",
            "--maximum-pages-per-layer", "--output"])
        let output = try args.require("--output")
        guard !FileManager.default.fileExists(atPath: output),
              let maximum = Int(args["--max-tokens"] ?? "16"), (2...16).contains(maximum),
              let maximumPages = Int(args["--maximum-pages-per-layer"] ?? "1024"),
              (1...4096).contains(maximumPages) else {
            throw CLIError.usage("Paged KV model probe requires new --output, --max-tokens 2...16 and --maximum-pages-per-layer 1...4096")
        }
        var report: [String: Any] = [
            "schema": "qwen-paged-kv-model-v1", "complete": false, "passed": false,
            "scope": "One full model, bounded ordinary AR branches, mixed-state/archive equality and physical page lifetime. Diagnostic exports perturb execution and are not a performance trial.",
            "notes": [
                "The dense and paged paths start from one evaluated dense prefill; this does not compare independent cold-prefill implementations.",
                "All 121 logical BF16 tensors and semantic host fields are checked. Retained storage extents may differ.",
                "Ordinary decode materialization deltas are sampled before diagnostic exports. Byte counters describe encoded payload, not physical DRAM bandwidth.",
                "Archive roundtrip uses the real host archive encoder/decoder in this process; SSD persistence and process restart are separate tests.",
                "A bounded conservative request/workspace ledger reservation covers simultaneous diagnostic copies; each physical arena has its own native-lifetime reservation.",
            ],
        ]
        var checks = [String: Bool](), observations = [[String: Any]]()
        var generatorOracleIDs = [Int32]()
        var operations = [[String: Any]](), poolSnapshots = [[String: Any]]()
        func object<T: Encodable>(_ value: T) throws -> Any {
            try JSONSerialization.jsonObject(with: JSONEncoder().encode(value))
        }
        func save(complete: Bool = false) throws {
            report["complete"] = complete; report["checks"] = checks
            report["state_checks"] = observations; report["ordinary_decode"] = operations
            report["pool_snapshots"] = poolSnapshots
            report["tensor_comparisons"] = observations.reduce(0) { $0 + ($1["tensor_count"] as? Int ?? 0) }
            report["passed"] = complete && !checks.isEmpty && !observations.isEmpty &&
                checks.values.allSatisfy { $0 } && observations.allSatisfy { $0["passed"] as? Bool == true }
            try emit(report, to: output)
        }
        func require(_ label: String, _ condition: Bool) throws {
            checks[label] = condition
            guard condition else { throw CLIError.usage("Paged KV model check failed: \(label)") }
        }
        try save()
        do {
            let directory = URL(fileURLWithPath: try args.require("--model-dir")).standardizedFileURL.resolvingSymlinksInPath()
            let tokenURL = URL(fileURLWithPath: try args.require("--tokens-file")).standardizedFileURL
            let input = try Data(contentsOf: tokenURL)
            let prompt = try JSONDecoder().decode([Int32].self, from: input)
            guard (10_000...12_288).contains(prompt.count),
                  maximumPages >= 2 * ((prompt.count + 32 + 31) / 32) + 8 else {
                throw CLIError.usage("Use a 10k+ agent fixture and enough pages for the shared seed plus a separately imported archive")
            }
            let library = URL(fileURLWithPath: try args.require("--library")).standardizedFileURL.resolvingSymlinksInPath()
            let executable = Bundle.main.executableURL ?? URL(fileURLWithPath: CommandLine.arguments[0])
            report["provenance"] = [
                "command": CommandLine.arguments, "captured_utc": Date().ISO8601Format(),
                "os": ProcessInfo.processInfo.operatingSystemVersionString,
                "model_directory": directory.path,
                "executable_sha256": try MoETilingBytes.hash(executable),
                "pool_library_path": library.path, "pool_library_sha256": try MoETilingBytes.hash(library),
                "tokens_path": tokenURL.path, "tokens_sha256": MoETilingBytes.digest(input),
                "config_sha256": try MoETilingBytes.hash(directory.appendingPathComponent("config.json")),
                "tokenizer_sha256": try MoETilingBytes.hash(directory.appendingPathComponent("tokenizer.json")),
                "base_mlx": try MoEGateUpProbeSupport.baseMLX(),
            ]
            report["configuration"] = ["prompt_tokens": prompt.count, "maximum_output_tokens": maximum,
                "prefill_chunk": 416, "prefill_evaluate_every_layers": 4,
                "maximum_pages_per_layer": maximumPages, "mtp_depth": 0,
                "joint_state_budget_bytes": 8 * 1024 * 1024 * 1024]
            var oldCache = 0
            try MX.check(mlx_set_cache_limit(&oldCache, 256 * 1024 * 1024), "Paged KV model probe allocator cache")
            defer { var ignored = 0; _ = mlx_set_cache_limit(&ignored, oldCache) }
            let tokenizer = try QwenTokenizer(modelDirectory: directory)
            let model = try QwenModel(modelDirectory: directory,
                reservedOutputIDs: tokenizer.reservedOutputTokenIDs,
                stateBudgetBytes: 8 * 1024 * 1024 * 1024) { count, total in
                if count % 8 == 0 || count == total {
                    FileHandle.standardError.write(Data("Paged KV model probe loaded \(count)/\(total) layers\n".utf8))
                }
            }
            defer { try? MX.synchronize() }
            try require("full_model_without_mtp", model.layerCount == 48 &&
                !model.weights.ledger.contains { $0.name.contains(".mtp.") })
            try require("no_decode_async_experiment", model.experimentalDecodeAsyncEveryLayers == 0)

            func snapshot(_ context: QwenPagedKVContext, label: String) throws -> [Int: GPUPagedKVPool.Statistics] {
                let stats = try context.layerStatistics
                poolSnapshots.append(["label": label, "layers": try object(stats),
                    "state_budget": try object(model.stateBudget.statistics)])
                try require(label + "_all_attention_layers", stats.count == 12)
                try require(label + "_arena_consistent", stats.values.allSatisfy {
                    $0.physicalPages == UInt64(maximumPages) && $0.failedOperations == 0 &&
                    $0.livePages + $0.freePages == $0.physicalPages
                })
                return stats
            }
            @discardableResult
            func observe(_ label: String, _ state: QwenModel.State,
                         expected: CacheReliabilityAnchor) throws -> CacheReliabilityAnchor {
                let actual = try CacheReliabilityAnchor(state)
                let mismatches = Set(actual.tensors.keys).union(expected.tensors.keys).filter {
                    actual.tensors[$0] != expected.tensors[$0]
                }.sorted()
                let passed = actual.valid && actual.tensors.count == 121 && actual.matches(expected)
                observations.append(["label": label, "offset": actual.host.offset,
                    "tensor_count": actual.tensors.count, "passed": passed,
                    "mismatched_tensor_names": mismatches, "anchor": try object(actual),
                    "expected_anchor": try object(expected)])
                try require(label + "_all_mixed_state_exact", passed)
                return actual
            }
            func denseStep(_ input: Int32, state: inout QwenModel.State) throws -> Int32 {
                let result = try model.forward(tokens: [input], state: &state, phase: .decode,
                    allowExperimentalDecodeAsync: false)
                guard let logits = result.logits else { throw CLIError.usage("Missing dense decode logits") }
                let selected = try model.greedyToken(logits)
                try model.evaluate([selected, result.stream], state: &state)
                return try selected.uint32TokenID()
            }
            func pagedStep(_ input: Int32, label: String, state: inout QwenModel.State,
                           context: QwenPagedKVContext) throws -> Int32 {
                let before = try context.layerStatistics
                let result = try model.forward(tokens: [input], state: &state, phase: .decode,
                    allowExperimentalDecodeAsync: false, allowPagedKV: true)
                guard let logits = result.logits else { throw CLIError.usage("Missing paged decode logits") }
                let selected = try model.greedyToken(logits)
                try model.evaluate([selected, result.stream], state: &state)
                let token = try selected.uint32TokenID()
                let after = try context.layerStatistics
                var layers = [[String: Any]]()
                for layer in before.keys.sorted() {
                    guard let a = before[layer], let b = after[layer] else {
                        throw CLIError.usage("Paged layer set changed during decode")
                    }
                    let passed = b.encodedWrites == a.encodedWrites + 1 &&
                        b.encodedReads == a.encodedReads + 1 &&
                        b.encodedMaterializations == a.encodedMaterializations &&
                        b.materializedBytes == a.materializedBytes && b.failedOperations == 0 &&
                        b.keyBufferIdentity == a.keyBufferIdentity && b.valueBufferIdentity == a.valueBufferIdentity
                    layers.append(["layer": layer, "passed": passed,
                        "writes_delta": b.encodedWrites - a.encodedWrites,
                        "reads_delta": b.encodedReads - a.encodedReads,
                        "materializations_delta": b.encodedMaterializations - a.encodedMaterializations,
                        "materialized_bytes_delta": b.materializedBytes - a.materializedBytes,
                        "copied_tail_bytes_delta": b.copiedTailBytes - a.copiedTailBytes,
                        "written_row_bytes_delta": b.writtenRowBytes - a.writtenRowBytes])
                    try require(label + "_layer\(layer)_direct_append_read_without_export", passed)
                }
                operations.append(["label": label, "input_token": input, "selected_token": token,
                    "offset": state.offset, "layers": layers])
                return token
            }

            // The nested scope releases all model states, archives and temporary
            // materializations before the outer pool lifetime assertions.
            func exercise(_ context: QwenPagedKVContext) throws {
                let estimate = try model.estimatedPrefixStateBytes(at: prompt.count + 32)
                guard let requests = model.stateBudget.reserve(bytes: estimate * 12, kind: .request) else {
                    throw CLIError.usage("Cannot reserve bounded diagnostic state copies")
                }
                defer { requests.release() }
                guard let workspace = model.stateBudget.reserve(bytes: estimate * 4, kind: .workspace) else {
                    throw CLIError.usage("Cannot reserve bounded diagnostic export workspace")
                }
                defer { workspace.release() }
                report["diagnostic_reservations"] = ["estimated_state_bytes": estimate,
                    "request_bytes": requests.bytes, "workspace_bytes": workspace.bytes]
                func runStates() throws {
                    var denseSeed = model.makeState(), cursor = 0, first: Int32 = 0
                    let prefetch = try model.makePrefillPrefetch(tokens: prompt, chunk: 416, state: denseSeed)
                    defer { prefetch.finish() }
                    while cursor < prompt.count {
                        let end = cursor < prompt.count - 1 ? min(prompt.count - 1, cursor + 416) : prompt.count
                        let result = try model.forward(tokens: Array(prompt[cursor..<end]), state: &denseSeed,
                            evaluateEveryLayers: 4, prefillPrefetch: prefetch, phase: .prefill,
                            profileLogits: end == prompt.count)
                        if end == prompt.count {
                            guard let logits = result.logits else { throw CLIError.usage("Missing prompt logits") }
                            let selected = try model.greedyToken(logits)
                            try model.evaluate([selected, result.stream], state: &denseSeed)
                            first = try selected.uint32TokenID()
                        } else { try model.evaluate([result.stream], state: &denseSeed) }
                        cursor = end
                    }
                    prefetch.finish()
                    _ = try model.checkpoint(state: &denseSeed)
                    let seedAnchor = try CacheReliabilityAnchor(denseSeed)
                    try require("dense_seed_complete", seedAnchor.valid && seedAnchor.tensors.count == 121 &&
                        denseSeed.offset == prompt.count && denseSeed.qsaActiveLayers == 12)
                    try require("first_output_nonterminal", !tokenizer.eosTokenIDs.contains(first))
                    var pagedSeed = try model.diagnosticPrivateStateCopy(denseSeed)
                    try model.usePagedKV(state: &pagedSeed, context: context)
                    try model.evaluate([], state: &pagedSeed)
                    try observe("imported_seed", pagedSeed, expected: seedAnchor)
                    let seedPages = try pagedSeed.diagnosticPageIDs
                    try MX.synchronize()
                    let beforeFork = try snapshot(context, label: "before_fork")
                    var denseA = try model.diagnosticPrivateStateCopy(denseSeed)
                    var pagedA = try model.forkPrefixState(pagedSeed)
                    var denseB = try model.diagnosticPrivateStateCopy(denseSeed)
                    var pagedB = try model.forkPrefixState(pagedSeed)
                    try MX.synchronize()
                    let afterFork = try snapshot(context, label: "after_fork")
                    let branchAPages = try pagedA.diagnosticPageIDs
                    let branchBPages = try pagedB.diagnosticPageIDs
                    try require("forks_share_identical_page_ids", seedPages.count == 12 &&
                        branchAPages == seedPages && branchBPages == seedPages)
                    try require("forks_share_kv_without_writes_or_exports", beforeFork.keys.allSatisfy { layer in
                        guard let a = beforeFork[layer], let b = afterFork[layer] else { return false }
                        return a.livePages == b.livePages && a.encodedWrites == b.encodedWrites &&
                            a.encodedMaterializations == b.encodedMaterializations &&
                            a.keyBufferIdentity == b.keyBufferIdentity && a.valueBufferIdentity == b.valueBufferIdentity
                    })
                    try observe("branch_a_initial", pagedA, expected: seedAnchor)
                    try observe("branch_b_initial", pagedB, expected: seedAnchor)
                    var denseIDs = [first], pagedIDs = [first], pending = first
                    for ordinal in 1..<maximum {
                        let reference = try denseStep(pending, state: &denseA)
                        let candidate = try pagedStep(pending, label: "branch_a_\(ordinal)", state: &pagedA, context: context)
                        denseIDs.append(reference); pagedIDs.append(candidate)
                        try require("branch_a_\(ordinal)_token_exact", reference == candidate)
                        let expected = try CacheReliabilityAnchor(denseA)
                        try observe("branch_a_\(ordinal)", pagedA, expected: expected)
                        pending = reference
                        try save()
                        if tokenizer.eosTokenIDs.contains(reference) { break }
                    }
                    report["branch_a"] = ["dense_token_ids": denseIDs, "paged_token_ids": pagedIDs,
                        "finish_reason": tokenizer.eosTokenIDs.contains(pending) ? "eos" : "length"]
                    generatorOracleIDs = Array(denseIDs.prefix(4))
                    try require("branch_a_nonempty_decode_exact", denseIDs == pagedIDs && denseIDs.count >= 2 &&
                        denseA.offset == prompt.count + denseIDs.count - 1 && pagedA.offset == denseA.offset)
                    try observe("held_seed_after_branch_a", pagedSeed, expected: seedAnchor)

                    // Forced inputs guarantee a different branch even when greedy
                    // continuations would coincide. Outputs still compare exactly.
                    let branchInputs: [Int32] = [first == 42 ? 43 : 42, 314, 2718]
                    var branchBIDs = [Int32](), branchBDenseIDs = [Int32]()
                    for (ordinal, input) in branchInputs.enumerated() {
                        let reference = try denseStep(input, state: &denseB)
                        let candidate = try pagedStep(input, label: "branch_b_\(ordinal + 1)", state: &pagedB, context: context)
                        branchBIDs.append(candidate)
                        branchBDenseIDs.append(reference)
                        try require("branch_b_\(ordinal + 1)_token_exact", reference == candidate)
                        try observe("branch_b_\(ordinal + 1)", pagedB, expected: CacheReliabilityAnchor(denseB))
                        try save()
                    }
                    report["branch_b"] = ["forced_input_token_ids": branchInputs,
                        "selected_token_ids": branchBIDs, "dense_token_ids": branchBDenseIDs]
                    try require("branches_use_distinct_first_inputs", branchInputs[0] != denseIDs[0])
                    try observe("held_seed_after_branch_b", pagedSeed, expected: seedAnchor)
                    let finalA = try CacheReliabilityAnchor(denseA)
                    try observe("branch_a_unchanged_after_branch_b", pagedA, expected: finalA)
                    let finalAPages = try pagedA.diagnosticPageIDs
                    let finalBPages = try pagedB.diagnosticPageIDs
                    let unchangedSeedPages = try pagedSeed.diagnosticPageIDs
                    try require("full_prefix_shared_and_partial_tail_private", seedPages.keys.allSatisfy { layer in
                        guard let seed = seedPages[layer], let a = finalAPages[layer], let b = finalBPages[layer],
                              seed == unchangedSeedPages[layer] else { return false }
                        let fullPrefixPages = prompt.count / 32
                        guard seed.count >= fullPrefixPages, a.count >= fullPrefixPages, b.count >= fullPrefixPages,
                              prompt.count % 32 == 0 ||
                                (seed.count > fullPrefixPages && a.count > fullPrefixPages && b.count > fullPrefixPages) else {
                            return false
                        }
                        let fullPrefixShared = a.prefix(fullPrefixPages).elementsEqual(seed.prefix(fullPrefixPages)) &&
                            b.prefix(fullPrefixPages).elementsEqual(seed.prefix(fullPrefixPages))
                        let distinctTail = prompt.count % 32 == 0 ||
                            (a[fullPrefixPages] != seed[fullPrefixPages] &&
                             b[fullPrefixPages] != seed[fullPrefixPages] && a[fullPrefixPages] != b[fullPrefixPages])
                        return fullPrefixShared && distinctTail
                    })
                    _ = try snapshot(context, label: "after_independent_branches")

                    let denseArchive = try model.exportPrefixState(denseA)
                    let pagedArchive = try model.exportPrefixState(pagedA)
                    try require("archive_payload_and_metadata_exact", denseArchive.metadata == pagedArchive.metadata &&
                        denseArchive.payload == pagedArchive.payload &&
                        denseArchive.logicalPayloadBytes == pagedArchive.logicalPayloadBytes)
                    report["archive"] = ["metadata_bytes": pagedArchive.metadata.count,
                        "payload_bytes": pagedArchive.payload.count,
                        "logical_payload_bytes": pagedArchive.logicalPayloadBytes,
                        "metadata_sha256": MoETilingBytes.digest(pagedArchive.metadata),
                        "payload_sha256": MoETilingBytes.digest(pagedArchive.payload),
                        "dense_metadata_bytes": denseArchive.metadata.count,
                        "dense_payload_bytes": denseArchive.payload.count,
                        "dense_logical_payload_bytes": denseArchive.logicalPayloadBytes,
                        "dense_metadata_sha256": MoETilingBytes.digest(denseArchive.metadata),
                        "dense_payload_sha256": MoETilingBytes.digest(denseArchive.payload)]
                    var restored = try model.importPrefixState(pagedArchive, expectedOffset: pagedA.offset)
                    try observe("archive_import_dense", restored, expected: finalA)
                    try model.usePagedKV(state: &restored, context: context)
                    try model.evaluate([], state: &restored)
                    try observe("archive_import_repage", restored, expected: finalA)
                    // This continuation is forced even if A ended at EOS: the probe
                    // validates state restoration, not a serving finish contract.
                    let restoredInput: Int32 = tokenizer.eosTokenIDs.contains(pending) ? 42 : pending
                    let reference = try denseStep(restoredInput, state: &denseA)
                    let candidate = try pagedStep(restoredInput, label: "archive_continuation", state: &restored, context: context)
                    try require("archive_continuation_token_exact", reference == candidate)
                    try observe("archive_continuation", restored, expected: CacheReliabilityAnchor(denseA))
                    try observe("held_seed_after_archive", pagedSeed, expected: seedAnchor)
                    try observe("original_branch_a_unchanged_after_archive", pagedA, expected: finalA)
                    _ = try snapshot(context, label: "before_state_release")
                    // Do not rely on optimizer lifetime shortening for this check.
                    denseSeed.reset(); pagedSeed.reset(); denseA.reset(); pagedA.reset()
                    denseB.reset(); pagedB.reset(); restored.reset()
                    try MX.synchronize()
                }
                do { try runStates() }
                catch {
                    // State wrappers leave scope before the join; conservative
                    // diagnostic leases remain held through GPU completion.
                    try? MX.synchronize()
                    throw error
                }
            }

            func generatorGate(_ context: QwenPagedKVContext) throws {
                let baseline = try QwenGenerator(model: model)
                let candidate = try QwenGenerator(model: model, pagedKVContext: context)
                let outputCount = generatorOracleIDs.count
                try require("generator_has_bounded_dense_oracle", (2...4).contains(outputCount))
                func request(mtp: Int = 0, append: GPUAttention.KVAppendMode = .reference) -> QwenGenerationRequest {
                    QwenGenerationRequest(tokens: prompt, maxTokens: outputCount, contextLimit: 16_384,
                        prefillChunk: 416, mtpDepth: mtp, prefillEvaluateEveryLayers: 4,
                        prefixCacheMaxTokens: 0, kvAppendMode: append)
                }
                let policyBefore = try context.layerStatistics
                let policyBudget = model.stateBudget.statistics
                for (name, invalid) in [("mtp", request(mtp: 1)), ("capacity256", request(append: .capacity256))] {
                    var rejected = false
                    do { try candidate.validateRequest(invalid) }
                    catch QwenGenerationError.invalidRequest { rejected = true }
                    try require("generator_rejects_" + name + "_policy", rejected)
                }
                try require("generator_policy_rejection_has_no_device_or_budget_effect", policyBefore == context.layerStatistics &&
                    policyBudget == model.stateBudget.statistics)

                FileHandle.standardError.write(Data("Paged KV model probe generator handoff P\(prompt.count)/O\(outputCount)\n".utf8))
                let prefillSession = try baseline.beginPrefill(request())
                defer { try? prefillSession.discard() }
                func prefillHost() -> [String: Int] {
                    ["processed_tokens": prefillSession.processedTokenCount,
                     "is_active": prefillSession.isActive ? 1 : 0,
                     "is_finished": prefillSession.isFinished ? 1 : 0,
                     "is_waiting_for_prefix_cache": prefillSession.isWaitingForPrefixCache ? 1 : 0,
                     "prefill_moe_reduction_calls": model.prefillMoEReductionCalls,
                     "prefill_moe_gate_up_calls": model.prefillMoEGateUpCalls,
                     "prefill_moe_grouped_down_calls": model.prefillMoEGroupedDownCalls,
                     "experimental_decode_async_submissions": model.experimentalDecodeAsyncSubmissions]
                }
                let prefillBefore = prefillHost()
                let prefillBudgetBefore = model.stateBudget.statistics
                let prefillLayersBefore = try context.layerStatistics
                var prefillForeignRejected = false
                do { _ = try candidate.stepPrefill(prefillSession) }
                catch QwenGenerationError.invalidRequest { prefillForeignRejected = true }
                let prefillAfter = prefillHost()
                let prefillBudgetAfter = model.stateBudget.statistics
                let prefillLayersAfter = try context.layerStatistics
                var prefillOwner: [String: Any] = ["foreign_rejected": prefillForeignRejected,
                    "host_before": prefillBefore, "host_after": prefillAfter,
                    "budget_before": try object(prefillBudgetBefore), "budget_after": try object(prefillBudgetAfter),
                    "layers_before": try object(prefillLayersBefore), "layers_after": try object(prefillLayersAfter)]
                report["prefill_cursor_owner"] = prefillOwner
                try require("generator_foreign_prefill_rejected_without_progress", prefillForeignRejected &&
                    prefillBefore["processed_tokens"] == 0 && prefillBefore["is_finished"] == 0 &&
                    prefillBefore["is_active"] == 0 && prefillBefore == prefillAfter &&
                    prefillBudgetBefore == prefillBudgetAfter && prefillLayersBefore == prefillLayersAfter)
                var preparedResult: QwenPrefillResult?
                var prefillOffsets = [Int](), expectedPrefillOffsets = [Int](), expectedOffset = 0
                let maximumPrefillSteps = (prompt.count - 1 + 415) / 416 + 1
                for ordinal in 0..<maximumPrefillSteps {
                    expectedOffset = expectedOffset < prompt.count - 1
                        ? min(prompt.count - 1, expectedOffset + 416) : prompt.count
                    let result = try baseline.stepPrefill(prefillSession)
                    expectedPrefillOffsets.append(expectedOffset)
                    prefillOffsets.append(prefillSession.processedTokenCount)
                    try require("generator_prefill_chunk\(ordinal)_original_grid", prefillSession.processedTokenCount == expectedOffset &&
                        (result != nil) == (expectedOffset == prompt.count))
                    if let result { preparedResult = result; break }
                }
                guard let prepared = preparedResult else {
                    throw CLIError.usage("Originating generator did not finish the bounded prefill cursor")
                }
                defer { prepared.discard() }
                prefillOwner["step_offsets"] = prefillOffsets
                prefillOwner["expected_step_offsets"] = expectedPrefillOffsets
                prefillOwner["originating_cursor_finished"] = prefillSession.isFinished
                prefillOwner["completed_handoff_ready"] = prepared.isReady
                prefillOwner["completed_handoff_first_token"] = prepared.firstToken
                report["prefill_cursor_owner"] = prefillOwner
                try require("generator_original_prefill_cursor_completed", prefillSession.isFinished &&
                    prefillOffsets == expectedPrefillOffsets && prefillOffsets.last == prompt.count)
                try require("generator_prefill_first_token_exact", prepared.isReady && prepared.firstToken == generatorOracleIDs[0])
                let session = try candidate.beginDecode(prepared)
                defer { try? session.discard() }
                try require("generator_cross_owner_prefill_handoff_consumed", !prepared.isReady &&
                    session.generatedTokenCount == 0 && !session.isFinished)
                let beforeForeign = try context.layerStatistics
                let foreignBudget = model.stateBudget.statistics
                var foreignRejected = false
                do { _ = try baseline.stepDecode(session) }
                catch QwenGenerationError.invalidRequest { foreignRejected = true }
                try require("generator_foreign_decode_rejected_without_consumption", foreignRejected &&
                    session.generatedTokenCount == 0 && !session.isFinished && !session.isActive &&
                    beforeForeign == context.layerStatistics && foreignBudget == model.stateBudget.statistics)
                var final: QwenGenerationResult?
                var stepRows = [[String: Any]]()
                for ordinal in 0..<outputCount {
                    let before = try context.layerStatistics
                    let result = try candidate.stepDecode(session)
                    let after = try context.layerStatistics
                    let expectedWrites: UInt64 = ordinal == 0 ? 0 : (ordinal == 1 ? 2 : 1)
                    let expectedReads: UInt64 = ordinal == 0 ? 0 : 1
                    let direct = before.keys.allSatisfy { layer in
                        guard let a = before[layer], let b = after[layer] else { return false }
                        return b.encodedMaterializations == a.encodedMaterializations &&
                            b.materializedBytes == a.materializedBytes &&
                            b.encodedWrites == a.encodedWrites + expectedWrites &&
                            b.encodedReads == a.encodedReads + expectedReads && b.failedOperations == 0
                    }
                    try require("generator_step\(ordinal)_direct_paged_path", direct)
                    try require("generator_step\(ordinal)_bounded_progress", session.generatedTokenCount == ordinal + 1)
                    stepRows.append(["ordinal": ordinal, "generated_tokens": session.generatedTokenCount,
                        "expected_writes_per_layer": expectedWrites, "expected_reads_per_layer": expectedReads,
                        "materializations_delta": 0, "completed": result != nil,
                        "layers_before": try object(before), "layers_after": try object(after)])
                    if let result { final = result; break }
                }
                guard let final, let phases = final.phases else {
                    throw CLIError.usage("Paged generator handoff did not finish within its bounded output steps")
                }
                let decoded = outputCount - 1
                try require("generator_complete_output_and_finish_exact", final.tokens == generatorOracleIDs &&
                    final.finishReason == (tokenizer.eosTokenIDs.contains(generatorOracleIDs.last!) ? .eos : .length) &&
                    session.isFinished && final.statistics.decodedTokenCount == decoded &&
                    final.statistics.decodeRounds == decoded && final.statistics.finalStateOffset == prompt.count + decoded)
                try require("generator_paged_phase_accounting", phases.kvAppendMode == "paged32" &&
                    phases.pagedKVTokenSteps == decoded && (phases.pagedKVImportSeconds ?? 0) > 0 &&
                    phases.kvCapacityTokenSteps == 0 && phases.kvCapacityWorkspaceFallbacks == 0 &&
                    phases.prefill.promptTokenCount == prompt.count && phases.prefill.evaluateEveryLayers == 4 &&
                    final.statistics.mtpDepth == 0)
                try MX.synchronize()
                let after = try snapshot(context, label: "generator_completed")
                let budget = model.stateBudget.statistics
                try require("generator_request_and_pages_released", budget.requestBytes == 0 &&
                    budget.totalBytes == context.reservedArenaBytes && after.values.allSatisfy {
                        $0.livePages == 0 && $0.inFlightOperations == 0 && $0.failedOperations == 0
                    })
                report["generator_gate"] = ["source_generator": "dense", "decode_generator": "physical_paged",
                    "oracle_token_ids": generatorOracleIDs, "foreign_decode_rejected": foreignRejected,
                    "step_checks": stepRows, "result": try object(final), "state_budget_after": try object(budget)]
                try save()
            }

            func ownedContext() throws {
                let context = try model.makePagedKVContext(libraryPath: library.path,
                    maximumPagesPerLayer: maximumPages)
                _ = try MoEGateUpProbeSupport.baseMLX()
                _ = try snapshot(context, label: "initial")
                try exercise(context)
                try generatorGate(context)
                try MX.synchronize()
                let released = try snapshot(context, label: "all_states_released")
                try require("all_pages_and_operations_released", released.values.allSatisfy {
                    $0.livePages == 0 && $0.freePages == $0.physicalPages &&
                    $0.inFlightOperations == 0 && $0.failedOperations == 0 &&
                    $0.completedOperations == $0.encodedWrites + $0.encodedReads + $0.encodedMaterializations
                })
                let budget = model.stateBudget.statistics
                try require("only_arena_lease_remains", budget.requestBytes == 0 &&
                    budget.totalBytes == context.reservedArenaBytes)
            }
            try ownedContext()
            try MX.synchronize()
            let finalBudget = model.stateBudget.statistics
            report["final_state_budget"] = try object(finalBudget)
            try require("arena_native_lifetime_lease_released", finalBudget.totalBytes == 0 && finalBudget.currentLeases == 0)
            try require("ordinary_decode_exercised", operations.count >= 5)
            try save(complete: true)
        } catch {
            report["error"] = String(describing: error)
            try save()
            throw error
        }
    }
}
