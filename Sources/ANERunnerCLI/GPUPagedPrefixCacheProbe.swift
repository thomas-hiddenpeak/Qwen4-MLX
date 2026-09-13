import ANERunnerCore
import ANERunnerGPU
import CMLX
import Foundation

/// Only cursors retain device state. Diagnostics retain hashes and scalar page
/// IDs, never a State/Tensor alias that could pin an unadmitted historical tail.
private final class PagedPrefixProbeRun {
    let label, family: String
    let generator: QwenGenerator
    let context: QwenPagedKVContext?
    let session: QwenDecodeSession
    let expectedCached, expectedReused: Int
    let fallback, oracle, alternate: Bool
    var tokens = [Int32]()
    var result: QwenGenerationResult?
    var pages = [Int: [Int32]]()
    var offset = 0
    init(label: String, family: String, generator: QwenGenerator,
         context: QwenPagedKVContext?, session: QwenDecodeSession,
         expectedCached: Int, expectedReused: Int, fallback: Bool, oracle: Bool, alternate: Bool) {
        self.label = label; self.family = family; self.generator = generator
        self.context = context; self.session = session
        self.expectedCached = expectedCached; self.expectedReused = expectedReused
        self.fallback = fallback; self.oracle = oracle; self.alternate = alternate
    }
}

extension RunnerCLI {
    /// Public generator/cache lifecycle gate. Hash exports perturb execution;
    /// native counters are sampled before those explicit diagnostic exports.
    static func probeGPUPagedPrefixCache(_ args: Arguments) throws {
        try args.validate(["--model-dir", "--tokens-file", "--library", "--max-tokens",
            "--maximum-pages-per-layer", "--suite", "--output"])
        let output = try args.require("--output"), suite = args["--suite"] ?? "all"
        guard !FileManager.default.fileExists(atPath: output),
              ["basic", "all"].contains(suite),
              let maximum = Int(args["--max-tokens"] ?? "8"), (8...16).contains(maximum),
              let maximumPages = Int(args["--maximum-pages-per-layer"] ?? "512"),
              [512, 1024].contains(maximumPages) else {
            throw CLIError.usage("Paged prefix cache probe requires new --output, --suite basic|all, --max-tokens 8...16, and --maximum-pages-per-layer 512|1024")
        }
        let boundary = 10_816, pageRows = 32, rowBytes = 2_048, attentionLayers = 12
        var report: [String: Any] = ["schema": "qwen-paged-prefix-cache-v1",
            "complete": false, "passed": false, "suite": suite,
            "scope": "Full model public cached generator, exact ordinary AR output/mixed state, shared physical pages, admission and lifetime. Diagnostic execution is not a throughput measurement.",
            "notes": [
                "The oracle runs an independent complete cold prefill with no cache or paged context.",
                "All 121 native BF16 state tensors and logical host values are hashed; only host hashes and page ID arrays are retained.",
                "Business pool counters are sampled before explicit diagnostic exports; diagnostic materializations are reported separately.",
                "Native byte counters describe encoded payload. MLX active/cache/peak, arena allocation, live slots, logical state admission and page union are distinct; none is process RSS.",
                "Prefix page IDs are first observed in the cold producer's actual decoder, then compared with warm and held request readers.",
            ]]
        var checks = [String: Bool](), states = [[String: Any]](), steps = [[String: Any]]()
        var trials = [[String: Any]](), snapshots = [[String: Any]]()
        func object<T: Encodable>(_ value: T) throws -> Any {
            try JSONSerialization.jsonObject(with: JSONEncoder().encode(value))
        }
        func save(complete: Bool = false) throws {
            report["complete"] = complete; report["checks"] = checks
            report["states"] = states; report["steps"] = steps
            report["trials"] = trials; report["snapshots"] = snapshots
            report["passed"] = complete && !checks.isEmpty && !states.isEmpty &&
                checks.values.allSatisfy { $0 } && states.allSatisfy { $0["passed"] as? Bool == true }
            try emit(report, to: output)
        }
        func require(_ label: String, _ value: Bool) throws {
            checks[label] = value
            guard value else { throw CLIError.usage("Paged prefix cache check failed: \(label)") }
        }
        try save()
        do {
            let directory = URL(fileURLWithPath: try args.require("--model-dir")).standardizedFileURL.resolvingSymlinksInPath()
            let tokenURL = URL(fileURLWithPath: try args.require("--tokens-file")).standardizedFileURL
            let input = try Data(contentsOf: tokenURL)
            let prompt = try JSONDecoder().decode([Int32].self, from: input)
            guard prompt.count == 11_057 else { throw CLIError.usage("Expected the real 11057-token agent fixture") }
            let library = URL(fileURLWithPath: try args.require("--library")).standardizedFileURL.resolvingSymlinksInPath()
            let executable = Bundle.main.executableURL ?? URL(fileURLWithPath: CommandLine.arguments[0])
            report["provenance"] = ["command": CommandLine.arguments,
                "captured_utc": Date().ISO8601Format(), "os": ProcessInfo.processInfo.operatingSystemVersionString,
                "model_directory": directory.path, "tokens_path": tokenURL.path,
                "tokens_sha256": MoETilingBytes.digest(input), "executable_sha256": try MoETilingBytes.hash(executable),
                "pool_library_path": library.path, "pool_library_sha256": try MoETilingBytes.hash(library),
                "config_sha256": try MoETilingBytes.hash(directory.appendingPathComponent("config.json")),
                "tokenizer_sha256": try MoETilingBytes.hash(directory.appendingPathComponent("tokenizer.json")),
                "base_mlx": try MoEGateUpProbeSupport.baseMLX()]
            report["configuration"] = ["prompt_tokens": prompt.count, "prefix_tokens": boundary,
                "shared_full_pages_per_layer": boundary / pageRows, "suffix_rows": prompt.count - boundary,
                "maximum_output_tokens": maximum, "prefill_chunk": 416, "prefill_evaluate_every_layers": 4,
                "context_limit": 16_384, "maximum_pages_per_layer": maximumPages,
                "mtp_depth": 0, "joint_state_budget_bytes": 8 * 1024 * 1024 * 1024]
            var oldCache = 0
            try MX.check(mlx_set_cache_limit(&oldCache, 256 * 1024 * 1024), "Paged prefix probe allocator cache")
            defer { var ignored = 0; _ = mlx_set_cache_limit(&ignored, oldCache) }
            let tokenizer = try QwenTokenizer(modelDirectory: directory)
            let model = try QwenModel(modelDirectory: directory,
                reservedOutputIDs: tokenizer.reservedOutputTokenIDs,
                stateBudgetBytes: 8 * 1024 * 1024 * 1024) { current, total in
                if current % 8 == 0 || current == total {
                    FileHandle.standardError.write(Data("Paged prefix probe loaded \(current)/\(total) layers\n".utf8))
                }
            }
            defer { try? MX.synchronize() }
            try require("full_model_without_mtp", model.layerCount == 48 &&
                !model.weights.ledger.contains { $0.name.contains(".mtp.") })
            try require("no_decode_async_experiment", model.experimentalDecodeAsyncEveryLayers == 0)
            let request = QwenGenerationRequest(tokens: prompt, maxTokens: maximum, contextLimit: 16_384,
                prefillChunk: 416, mtpDepth: 0, prefillEvaluateEveryLayers: 4,
                prefixCacheMaxTokens: boundary, kvAppendMode: .reference)
            var variant = prompt
            let changedIndex = boundary + 17
            guard let replacement = prompt.first(where: {
                $0 > 0 && $0 < 100_000 && $0 != prompt[changedIndex] && !tokenizer.eosTokenIDs.contains($0)
            }) else { throw CLIError.usage("Fixture has no distinct ordinary token for a controlled suffix fork") }
            variant[changedIndex] = replacement
            let variantRequest = QwenGenerationRequest(tokens: variant, maxTokens: maximum, contextLimit: 16_384,
                prefillChunk: 416, mtpDepth: 0, prefillEvaluateEveryLayers: 4,
                prefixCacheMaxTokens: boundary, kvAppendMode: .reference)
            report["suffix_variant"] = ["changed_index": changedIndex,
                "original_token_id": prompt[changedIndex], "replacement_token_id": replacement,
                "replacement_origin": "A distinct ordinary-range token already present in this real fixture"]
            let exportBytes = try model.estimatedPrefixStateBytes(at: prompt.count + maximum) * 2
            report["per_observation_export_workspace_bytes"] = exportBytes
            var oracleAnchors = [Int: CacheReliabilityAnchor]()
            var oracleResult: QwenGenerationResult?
            var variantAnchors = [Int: CacheReliabilityAnchor]()
            var variantResult: QwenGenerationResult?
            var seeds = [String: [Int: [Int32]]]()

            func anchor(_ state: QwenModel.State, label: String, establish: Bool, alternate: Bool = false) throws {
                guard let lease = model.stateBudget.reserve(bytes: exportBytes, kind: .workspace) else {
                    throw CLIError.usage("Cannot admit explicit probe export workspace")
                }
                defer { lease.release() }
                let value: CacheReliabilityAnchor
                do {
                    value = try CacheReliabilityAnchor(state)
                    try MX.synchronize()
                } catch {
                    // Keep the export allowance until pending diagnostic work
                    // is joined; propagate a failed join rather than hiding it.
                    let original = error
                    try MX.synchronize()
                    throw original
                }
                var passed = value.valid && value.tensors.count == 121
                var mismatches = [String]()
                let expected = alternate ? variantAnchors[state.offset] : oracleAnchors[state.offset]
                if establish {
                    if let expected { passed = passed && value.matches(expected) }
                    else if alternate { variantAnchors[state.offset] = value }
                    else { oracleAnchors[state.offset] = value }
                } else if let expected {
                    passed = passed && value.matches(expected)
                    mismatches = Set(value.tensors.keys).union(expected.tensors.keys).filter {
                        value.tensors[$0] != expected.tensors[$0]
                    }.sorted()
                } else { passed = false }
                let oracleLabel = alternate ? "oracle_changed_suffix" : "oracle_cold"
                let expectedLabel = state.offset == boundary ? oracleLabel + "_prefix_coldBoundary"
                    : oracleLabel + "_step\(state.offset - prompt.count)"
                states.append(["label": label, "prompt_variant": alternate ? "changed_suffix" : "base",
                    "oracle_anchor_label": expectedLabel, "establishes_oracle": establish && expected == nil,
                    "offset": state.offset, "tensor_count": value.tensors.count,
                    "passed": passed, "mismatched_tensors": mismatches, "anchor": try object(value)])
                try require(label + "_121_tensor_and_host_exact", passed)
            }

            @discardableResult
            func snapshot(_ label: String, _ context: QwenPagedKVContext,
                          runs: [PagedPrefixProbeRun] = []) throws -> [Int: GPUPagedKVPool.Statistics] {
                try MX.synchronize()
                let pools = try context.layerStatistics
                try require(label + "_healthy_complete_pools", pools.count == attentionLayers && pools.values.allSatisfy {
                    $0.physicalPages == UInt64(context.maximumPagesPerLayer) &&
                    $0.livePages + $0.freePages == $0.physicalPages &&
                    $0.failedOperations == 0 && $0.inFlightOperations == 0
                })
                var unionRows = [[String: Any]]()
                for layer in pools.keys.sorted() {
                    let unique = Set(runs.flatMap { $0.pages[layer] ?? [] })
                    unionRows.append(["layer": layer, "unique_page_ids": unique.sorted(),
                        "unique_page_bytes": unique.count * 65_536,
                        "logical_kv_payload_bytes": runs.reduce(0) { $0 + $1.offset * rowBytes },
                        "logical_page_references": runs.reduce(0) { $0 + ($1.pages[layer]?.count ?? 0) }])
                }
                snapshots.append(["label": label, "pool_layers": try object(pools),
                    "page_admission": try object(context.pageAdmissionStatistics),
                    "state_budget": try object(model.stateBudget.statistics),
                    "mlx_memory": try MX.memory(), "arena_reserved_bytes": context.reservedArenaBytes,
                    "held_readers": try runs.map { ["label": $0.label, "offset": $0.offset,
                        "prompt_variant": $0.alternate ? "changed_suffix" : "base",
                        "page_ids": try object($0.pages)] as [String: Any] },
                    "held_reader_page_union": unionRows])
                return pools
            }

            func prepare(_ generator: QwenGenerator, label: String, cached: Int,
                         establish: Bool = false, alternate: Bool = false) throws -> QwenPrefillResult {
                FileHandle.standardError.write(Data("Paged prefix probe: \(label) prefill\n".utf8))
                var events = [String]()
                generator.prefixStateObserver = { event, state in
                    events.append(event)
                    try require(label + "_prefix_offset_\(event)", state.offset == boundary && !state.hasPagedKV)
                    try anchor(state, label: label + "_prefix_" + event, establish: establish, alternate: alternate)
                }
                defer { generator.prefixStateObserver = nil }
                let prepared = try generator.prefill(alternate ? variantRequest : request)
                try require(label + "_prefill_accounting", prepared.statistics.cachedTokenCount == cached &&
                    prepared.statistics.computedTokenCount == prompt.count - cached &&
                    prepared.statistics.actualForwardTokenCount == prompt.count - cached &&
                    prepared.statistics.evaluateEveryLayers == 4)
                try require(label + "_prefix_observed", events.contains(cached > 0 ? "restore" : "coldBoundary"))
                report[label + "_prefill"] = try object(prepared.statistics)
                return prepared
            }

            func begin(_ label: String, family: String, generator: QwenGenerator,
                       context: QwenPagedKVContext?, cached: Int, reused: Int,
                       fallback: Bool = false, oracle: Bool = false, alternate: Bool = false,
                       handoff: QwenPrefillResult? = nil,
                       cancellation: QwenCancellation? = nil) throws -> PagedPrefixProbeRun {
                let prepared = try handoff ?? prepare(generator, label: label, cached: cached,
                    establish: oracle, alternate: alternate)
                defer { prepared.discard() }
                let session = try generator.beginDecode(prepared, cancellation: cancellation)
                return PagedPrefixProbeRun(label: label, family: family, generator: generator,
                    context: context, session: session, expectedCached: cached, expectedReused: reused,
                    fallback: fallback, oracle: oracle, alternate: alternate)
            }

            func step(_ run: PagedPrefixProbeRun) throws {
                let ordinal = run.tokens.count
                let before = try run.context?.layerStatistics
                var observations = 0
                run.generator.decodeStateObserver = { event, state in
                    observations += 1
                    let label = run.label + "_step\(ordinal)"
                    let afterBusiness = try run.context?.layerStatistics
                    let physical = run.context != nil && !run.fallback && ordinal > 0
                    // Preserve raw evidence even when an assertion below
                    // throws before the diagnostic export or token callback.
                    let observationIndex = steps.count
                    var raw: [String: Any] = ["label": label, "event": event,
                        "offset": state.offset, "expected_physical": physical,
                        "actual_physical": state.hasPagedKV]
                    if let before { raw["pool_before_step"] = try object(before) }
                    if let afterBusiness { raw["pool_after_business_before_diagnostic"] = try object(afterBusiness) }
                    steps.append(raw)
                    try require(label + "_state_mode", state.hasPagedKV == physical &&
                        state.offset == prompt.count + ordinal && event == (ordinal == 0 ? "firstToken" : "decode"))
                    run.offset = state.offset
                    run.pages = try state.diagnosticPageIDs
                    steps[observationIndex]["page_ids"] = try object(run.pages)
                    var layerRows = [[String: Any]]()
                    if let before, let afterBusiness {
                        for layer in before.keys.sorted() {
                            guard let a = before[layer], let b = afterBusiness[layer] else {
                                throw CLIError.usage("Pool layer set changed during a decode step")
                            }
                            let writes = physical ? (ordinal == 1 ? 2 : 1) : 0
                            let rows = physical ? (ordinal == 1 ? prompt.count - run.expectedReused + 1 : 1) : 0
                            let valid = b.encodedWrites == a.encodedWrites + UInt64(writes) &&
                                b.encodedReads == a.encodedReads + (physical ? 1 : 0) &&
                                b.writtenRowBytes == a.writtenRowBytes + UInt64(rows * rowBytes) &&
                                b.encodedMaterializations == a.encodedMaterializations &&
                                b.materializedBytes == a.materializedBytes && b.failedOperations == 0 &&
                                b.keyBufferIdentity == a.keyBufferIdentity && b.valueBufferIdentity == a.valueBufferIdentity
                            layerRows.append(["layer": layer, "passed": valid,
                                "writes_delta": Int64(b.encodedWrites) - Int64(a.encodedWrites),
                                "reads_delta": Int64(b.encodedReads) - Int64(a.encodedReads),
                                "written_row_bytes_delta": Int64(b.writtenRowBytes) - Int64(a.writtenRowBytes),
                                "copied_tail_bytes_delta": Int64(b.copiedTailBytes) - Int64(a.copiedTailBytes),
                                "materializations_delta": Int64(b.encodedMaterializations) - Int64(a.encodedMaterializations),
                                "expected_imported_suffix_rows": ordinal == 1 && physical ? prompt.count - run.expectedReused : 0])
                            steps[observationIndex]["business_layer_deltas"] = layerRows
                            try require(label + "_layer\(layer)_business_counts", valid)
                        }
                        if physical && run.expectedReused > 0 {
                            let full = run.expectedReused / pageRows
                            let currentPrefix = run.pages.mapValues { Array($0.prefix(full)) }
                            if let seed = seeds[run.family] {
                                try require(label + "_shared_\(full)_page_ids", currentPrefix == seed)
                            } else {
                                seeds[run.family] = currentPrefix
                                try require(label + "_seed_\(full)_pages", currentPrefix.count == attentionLayers &&
                                    currentPrefix.values.allSatisfy { $0.count == full })
                            }
                        }
                    }
                    try anchor(state, label: label, establish: run.oracle, alternate: run.alternate)
                    let afterDiagnostic = try run.context?.layerStatistics
                    if let afterDiagnostic {
                        steps[observationIndex]["pool_after_explicit_diagnostic"] = try object(afterDiagnostic)
                    }
                    if let afterBusiness, let afterDiagnostic {
                        try require(label + "_explicit_export_accounted", afterBusiness.keys.allSatisfy { layer in
                            guard let a = afterBusiness[layer], let b = afterDiagnostic[layer] else { return false }
                            return b.encodedMaterializations == a.encodedMaterializations + (physical ? 1 : 0) &&
                                b.materializedBytes == a.materializedBytes + UInt64(physical ? state.offset * rowBytes : 0) &&
                                b.encodedWrites == a.encodedWrites && b.encodedReads == a.encodedReads
                        })
                    }
                    if let context = run.context {
                        steps[observationIndex]["page_admission"] = try object(context.pageAdmissionStatistics)
                    }
                }
                defer { run.generator.decodeStateObserver = nil }
                run.result = try run.generator.stepDecode(run.session) { run.tokens.append($0) }
                try require(run.label + "_step\(ordinal)_one_publication", observations == 1 &&
                    run.tokens.count == ordinal + 1 && run.session.generatedTokenCount == run.tokens.count)
                if let expected = run.alternate ? variantResult : oracleResult, !run.oracle {
                    try require(run.label + "_step\(ordinal)_output_prefix_exact",
                        run.tokens == Array(expected.tokens.prefix(run.tokens.count)))
                }
            }

            func finish(_ run: PagedPrefixProbeRun) throws {
                while run.result == nil && run.tokens.count < maximum { try step(run) }
                guard let result = run.result, let phases = result.phases else {
                    throw CLIError.usage("\(run.label) did not finish within the output budget")
                }
                try require(run.label + "_finished", run.session.isFinished && result.tokens == run.tokens &&
                    result.tokens.count >= 3 && result.statistics.decodedTokenCount == result.tokens.count - 1 &&
                    result.statistics.decodeRounds == result.tokens.count - 1 &&
                    result.statistics.finalStateOffset == prompt.count + result.tokens.count - 1 &&
                    result.statistics.mtpDepth == 0)
                if run.oracle {
                    if run.alternate { variantResult = result }
                    else { oracleResult = result }
                }
                else {
                    let expected = run.alternate ? variantResult : oracleResult
                    try require(run.label + "_complete_ids_and_finish_exact", result.tokens == expected?.tokens &&
                        result.finishReason == expected?.finishReason)
                }
                try require(run.label + "_final_prefill_accounting", phases.prefill.cachedTokenCount == run.expectedCached &&
                    phases.prefill.actualForwardTokenCount == prompt.count - run.expectedCached)
                if run.context != nil {
                    let physical = !run.fallback
                    try require(run.label + "_paged_phase_accounting", phases.kvAppendMode == (physical ? "paged32" : "reference") &&
                        phases.pagedKVTokenSteps == (physical ? result.tokens.count - 1 : 0) &&
                        phases.pagedKVReusedPrefixTokens == (physical ? run.expectedReused : 0) &&
                        phases.pagedKVImportedSuffixRows == (physical ? prompt.count - run.expectedReused : 0) &&
                        phases.pagedKVCapacityFallbacks == (physical ? 0 : 1) &&
                        (physical ? (phases.pagedKVReservedPagesPerLayer ?? 0) > 0 : phases.pagedKVReservedPagesPerLayer == 0))
                }
                trials.append(["label": run.label, "passed": true, "result": try object(result)])
                try save()
            }

            func cachedGenerator(_ context: QwenPagedKVContext) throws -> QwenGenerator {
                try QwenGenerator(model: model,
                    prefixCacheLimits: .init(maxEntries: 8, maxBytes: 1024 * 1024 * 1024),
                    pagedKVContext: context)
            }

            func empty(_ label: String, context: QwenPagedKVContext) throws {
                let pools = try snapshot(label, context)
                try require(label + "_all_pages_claims_and_metadata_released", pools.values.allSatisfy {
                    $0.livePages == 0 && $0.freePages == $0.physicalPages &&
                    $0.completedOperations == $0.encodedWrites + $0.encodedReads + $0.encodedMaterializations
                } && context.outstandingDecodeClaimPages == 0 && context.pageAdmissionStatistics.activeClaims == 0 &&
                    model.stateBudget.statistics.cacheBytes == 0 && model.stateBudget.statistics.requestBytes == 0 &&
                    model.stateBudget.statistics.totalBytes == context.reservedArenaBytes &&
                    model.stateBudget.statistics.currentLeases == attentionLayers)
            }

            // An independent cold request supplies the whole sequence and the
            // original-grid prefix anchor; candidate prefills never supply it.
            do {
                let generator = try QwenGenerator(model: model)
                let run = try begin("oracle_cold", family: "oracle", generator: generator,
                    context: nil, cached: 0, reused: 0, oracle: true)
                defer { try? run.session.discard() }
                try finish(run)
            }
            try MX.synchronize()
            try require("oracle_released_all_state", model.stateBudget.statistics.totalBytes == 0)
            if suite == "all" {
                do {
                    let generator = try QwenGenerator(model: model)
                    let run = try begin("oracle_changed_suffix", family: "oracle_variant", generator: generator,
                        context: nil, cached: 0, reused: 0, oracle: true, alternate: true)
                    defer { try? run.session.discard() }
                    try finish(run)
                }
                guard let base = oracleAnchors[prompt.count], let changed = variantAnchors[prompt.count],
                      let basePrefix = oracleAnchors[boundary], let changedPrefix = variantAnchors[boundary] else {
                    throw CLIError.usage("Missing independent suffix-fork oracle states")
                }
                try require("variant_same_checkpoint_different_dense_suffix_state", basePrefix.matches(changedPrefix) &&
                    base.tensors.keys.contains { base.tensors[$0] != changed.tensors[$0] })
                try MX.synchronize()
                try require("variant_oracle_released_all_state", model.stateBudget.statistics.totalBytes == 0)
            }

            func basic() throws {
                let context = try model.makePagedKVContext(libraryPath: library.path, maximumPagesPerLayer: maximumPages)
                let generator = try cachedGenerator(context)
                defer { try? generator.clearPrefixCache() }
                _ = try snapshot("main_initial", context)
                do {
                    let cold = try begin("paged_cold_populate", family: "main", generator: generator,
                        context: context, cached: 0, reused: boundary)
                    defer { try? cold.session.discard() }
                    let populated = try snapshot("cold_completed_prefill", context)
                    try require("cold_completed_prefill_imported_only_checkpoint", populated.values.allSatisfy {
                        $0.livePages == UInt64(boundary / pageRows) && $0.encodedWrites == 1 &&
                        $0.writtenRowBytes == UInt64(boundary * rowBytes) &&
                        $0.encodedReads == 0 && $0.encodedMaterializations == 0
                    })
                    try finish(cold)
                }
                do {
                    let warm = try begin("paged_warm_hit", family: "main", generator: generator,
                        context: context, cached: boundary, reused: boundary)
                    defer { try? warm.session.discard() }
                    try finish(warm)
                }
                if suite == "all" {
                    let changed = try begin("paged_warm_changed_suffix", family: "main", generator: generator,
                        context: context, cached: boundary, reused: boundary, alternate: true)
                    defer { try? changed.session.discard() }
                    try finish(changed)
                }
                do {
                    let a = try begin("held_a", family: "main", generator: generator,
                        context: context, cached: boundary, reused: boundary)
                    defer { try? a.session.discard() }
                    let b = try begin("held_b", family: "main", generator: generator,
                        context: context, cached: boundary, reused: boundary, alternate: suite == "all")
                    defer { try? b.session.discard() }
                    try step(a); try step(b); try step(a); try step(b)
                    try require("two_live_paged_claims", context.pageAdmissionStatistics.activeClaims == 2)
                    let beforeClear = try snapshot("two_paged_cursors_before_clear", context, runs: [a, b])
                    let cachedBytesBeforeClear = model.stateBudget.statistics.cacheBytes
                    try require("held_private_suffix_shared_prefix", a.pages.count == attentionLayers &&
                        a.pages.keys.allSatisfy { layer in
                            guard let x = a.pages[layer], let y = b.pages[layer] else { return false }
                            return Array(x.prefix(boundary / pageRows)) == Array(y.prefix(boundary / pageRows)) &&
                                Set(x.dropFirst(boundary / pageRows)).isDisjoint(with: y.dropFirst(boundary / pageRows))
                        })
                    try generator.clearPrefixCache()
                    let afterClear = try snapshot("two_paged_cursors_after_clear", context, runs: [a, b])
                    try require("clear_preserves_held_physical_pages_and_metadata", beforeClear.keys.allSatisfy { layer in
                        beforeClear[layer]?.livePages == afterClear[layer]?.livePages
                    } && context.pageAdmissionStatistics.activeClaims == 2 &&
                        model.stateBudget.statistics.cacheBytes > 0 &&
                        model.stateBudget.statistics.cacheBytes < cachedBytesBeforeClear)
                    while a.result == nil || b.result == nil {
                        if a.result == nil { try step(a) }
                        if b.result == nil { try step(b) }
                    }
                    try finish(a); try finish(b)
                }
                try empty("held_cursors_finished_after_clear", context: context)
                do {
                    let repopulate = try begin("repopulate_after_clear", family: "after_clear", generator: generator,
                        context: context, cached: 0, reused: boundary)
                    defer { try? repopulate.session.discard() }
                    try finish(repopulate)
                }
                do {
                    let cancellation = QwenCancellation()
                    let cancelled = try begin("cancelled_warm", family: "after_clear", generator: generator,
                        context: context, cached: boundary, reused: boundary, cancellation: cancellation)
                    defer { try? cancelled.session.discard() }
                    try step(cancelled); try step(cancelled)
                    let before = try snapshot("cancel_before_next_step", context, runs: [cancelled])
                    cancellation.cancel()
                    var rejected = false
                    do { _ = try generator.stepDecode(cancelled.session) }
                    catch QwenGenerationError.cancelled { rejected = true }
                    let after = try context.layerStatistics
                    try require("cancel_before_admission_has_no_native_work", rejected && before == after)
                    try cancelled.session.discard()
                    try MX.synchronize()
                    try require("cancel_discard_releases_claim", context.pageAdmissionStatistics.activeClaims == 0)
                    trials.append(["label": "cancelled_warm", "passed": true,
                        "committed_token_ids": cancelled.tokens, "cancelled_before_next_step": rejected])
                }
                do {
                    let recovered = try begin("warm_after_cancel", family: "after_clear", generator: generator,
                        context: context, cached: boundary, reused: boundary)
                    defer { try? recovered.session.discard() }
                    try finish(recovered)
                }
                try generator.clearPrefixCache()
                try empty("main_final_only_fixed_arena", context: context)
            }
            try basic()
            try MX.synchronize()
            try require("main_context_destruction_releases_arena", model.stateBudget.statistics.totalBytes == 0 &&
                model.stateBudget.statistics.currentLeases == 0)

            if suite == "all" {
                func fallback() throws {
                    // A real first cursor occupies the short suffix. Its
                    // constant future claim denies the second cursor before
                    // any import; no direct/private admission APIs are used.
                    let finalPages = (prompt.count + maximum - 1 + pageRows - 1) / pageRows
                    let smallPages = finalPages + 1
                    let context = try model.makePagedKVContext(libraryPath: library.path, maximumPagesPerLayer: smallPages)
                    let generator = try cachedGenerator(context)
                    defer { try? generator.clearPrefixCache() }
                    report["fallback_maximum_pages_per_layer"] = smallPages
                    do {
                        let a = try begin("small_pool_held_paged", family: "small", generator: generator,
                            context: context, cached: 0, reused: boundary)
                        defer { try? a.session.discard() }
                        try step(a); try step(a)
                        let b = try begin("small_pool_dense_fallback", family: "small", generator: generator,
                            context: context, cached: boundary, reused: 0, fallback: true)
                        defer { try? b.session.discard() }
                        _ = try snapshot("small_pool_before_second_decision", context, runs: [a])
                        try step(b); try step(b)
                        try require("small_pool_one_claim_one_denial", context.pageAdmissionStatistics.activeClaims == 1 &&
                            context.pageAdmissionStatistics.deniedClaims == 1)
                        // Release the competing claim while B is still live:
                        // B must keep its original dense decision afterward.
                        try finish(a)
                        try require("fallback_remains_dense_after_capacity_returns", context.pageAdmissionStatistics.activeClaims == 0)
                        try finish(b)
                    }
                    try generator.clearPrefixCache()
                    try empty("small_pool_final_only_fixed_arena", context: context)
                }
                try fallback()
                try MX.synchronize()
                try require("small_context_destruction_releases_arena", model.stateBudget.statistics.totalBytes == 0 &&
                    model.stateBudget.statistics.currentLeases == 0)

                func foreignHandoff() throws {
                    let source = try model.makePagedKVContext(libraryPath: library.path, maximumPagesPerLayer: maximumPages)
                    let producer = try cachedGenerator(source)
                    defer { try? producer.clearPrefixCache() }
                    do {
                        let cold = try begin("foreign_source_populate", family: "foreign_source", generator: producer,
                            context: source, cached: 0, reused: boundary)
                        defer { try? cold.session.discard() }
                        try finish(cold)
                    }
                    func consumeElsewhere() throws {
                        let destination = try model.makePagedKVContext(libraryPath: library.path,
                            maximumPagesPerLayer: maximumPages)
                        let consumer = try QwenGenerator(model: model, pagedKVContext: destination)
                        let prepared = try prepare(producer, label: "foreign_completed_handoff", cached: boundary)
                        defer { prepared.discard() }
                        let before = try snapshot("foreign_source_before_consume", source)
                        let run = try begin("foreign_context_full_import", family: "foreign_destination",
                            generator: consumer, context: destination, cached: boundary, reused: 0, handoff: prepared)
                        defer { try? run.session.discard() }
                        try finish(run)
                        let after = try snapshot("foreign_source_after_consume", source)
                        try require("foreign_attachment_ignored_without_touching_source", before == after)
                        let dest = try snapshot("foreign_destination_finished", destination)
                        try require("foreign_destination_all_pages_and_claims_released", dest.values.allSatisfy {
                            $0.livePages == 0 && $0.inFlightOperations == 0
                        } && destination.pageAdmissionStatistics.activeClaims == 0)
                    }
                    try consumeElsewhere()
                    try MX.synchronize()
                    do {
                        let warm = try begin("foreign_source_still_warm", family: "foreign_source", generator: producer,
                            context: source, cached: boundary, reused: boundary)
                        defer { try? warm.session.discard() }
                        try finish(warm)
                    }
                    try producer.clearPrefixCache()
                    try empty("foreign_final_only_source_arena", context: source)
                }
                try foreignHandoff()
                try MX.synchronize()
                try require("foreign_contexts_destruction_releases_arena", model.stateBudget.statistics.totalBytes == 0 &&
                    model.stateBudget.statistics.currentLeases == 0)
            }
            report["final_state_budget"] = try object(model.stateBudget.statistics)
            report["final_mlx_memory"] = try MX.memory()
            try save(complete: true)
        } catch {
            report["error"] = String(describing: error)
            try? save()
            throw error
        }
    }
}
