import CMLX
import Foundation

/// Isolated candidate diagnostic; no model, server, cache or default policy.
/// Emits bounded NDJSON events. The caller must serialize GPU access and use a
/// finite external timeout. Both modes finish each token before its successor.
public enum GPUKVCapacityMechanismProbe {
    public static func run(libraryPath: String, asynchronous: Bool,
                           onEvent: @escaping (Data) throws -> Void) throws {
        let runner = try CapacityMechanismRun(libraryPath: libraryPath,
            asynchronous: asynchronous, onEvent: onEvent)
        try runner.run()
    }
}

private final class CapacityMechanismRun {
    private typealias Pair = GPUKVCapacityStorage.LogicalView
    private let diagnostics: GPUKVAllocationDiagnostics
    private let asynchronous: Bool
    private let onEvent: (Data) throws -> Void
    private var checks = 0, appends = 0, growths = 0, cows = 0, reuses = 0
    private var comparedBF16Elements = 0

    init(libraryPath: String, asynchronous: Bool,
         onEvent: @escaping (Data) throws -> Void) throws {
        diagnostics = try GPUKVAllocationDiagnostics(libraryPath: libraryPath)
        self.asynchronous = asynchronous; self.onEvent = onEvent
    }
    private func emit(_ fields: [String: Any]) throws {
        var data = try JSONSerialization.data(withJSONObject: fields, options: [.sortedKeys])
        data.append(0x0a); try onEvent(data)
    }
    private func require(_ condition: Bool, _ label: String) throws {
        checks += 1
        guard condition else { throw GPUError.invalid("Swift capacity probe: \(label)") }
    }
    private func elapsed(_ start: UInt64) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - start) * 1e-9
    }
    private func trackedPeak(before: [String: Int], after: [String: Int]) -> Int {
        // This pinned allocator resets peak to zero. With no new allocation,
        // raw peak stays zero despite resident data; preserve all three values.
        max(before["active_bytes"] ?? 0, max(after["peak_bytes"] ?? 0, after["active_bytes"] ?? 0))
    }
    private func ready(_ pair: Pair) throws {
        if asynchronous { try MX.asyncEval(pair.tensors) }
        else { try MX.eval(pair.tensors) }
        try MX.synchronize()
    }
    /// The logical roots mimic current State.tensors; no full-buffer root is
    /// added here just to make donation pass. This local view dies on return.
    private func ready(_ state: GPUKVCapacityStorage) throws {
        guard let view = try state.logicalView() else { throw GPUError.invalid("Missing logical capacity state") }
        try ready(view)
    }
    private func snapshot(_ state: GPUKVCapacityStorage) throws -> GPUKVAllocationPair {
        guard let result = try state.allocationSnapshot(using: diagnostics) else {
            throw GPUError.invalid("Missing full capacity allocation")
        }
        return result
    }
    private func snapshot(_ pair: Pair) throws -> GPUKVAllocationPair {
        try GPUKVAllocationPair(keys: diagnostics.snapshot(pair.keys), values: diagnostics.snapshot(pair.values))
    }
    private func pattern(start: Int, count: Int, salt: Int) throws -> Tensor {
        var host = [Float](repeating: 0, count: 2 * count * 256)
        for head in 0..<2 {
            for row in 0..<count {
                for dimension in 0..<256 {
                    let value = ((start + row) * 43 + head * 71 + dimension * 19 + salt) % 509 - 254
                    host[(head * count + row) * 256 + dimension] = Float(value) / 16
                }
            }
        }
        return try MX.array(host, shape: [1,2,count,256], dtype: MLX_BFLOAT16)
    }
    private func patterns(start: Int = 0, count: Int) throws -> Pair {
        try Pair(keys: pattern(start: start, count: count, salt: 11),
            values: pattern(start: start, count: count, salt: 137))
    }
    private func bytes(_ pair: Pair) throws -> Data {
        var payload = Data()
        payload.reserveCapacity(pair.keys.nbytes + pair.values.nbytes)
        try QwenPrefixStateArchiveBytes.append(pair.keys, to: &payload)
        try QwenPrefixStateArchiveBytes.append(pair.values, to: &payload)
        return payload
    }
    private func bytes(_ state: GPUKVCapacityStorage) throws -> Data {
        guard let view = try state.logicalView() else { throw GPUError.invalid("Missing state for raw readback") }
        return try bytes(view)
    }
    private func verify(_ state: GPUKVCapacityStorage, oracle: Pair, label: String) throws {
        guard let view = try state.logicalView() else { throw GPUError.invalid("Missing capacity logical view") }
        try ready(view)
        let backing = try snapshot(state), logical = try snapshot(view)
        try require(view.keys.shape == [1,2,state.logicalRows,256] && view.values.shape == view.keys.shape,
            label + ": logical shape")
        try require(view.keys.dtype == MLX_BFLOAT16 && view.values.dtype == MLX_BFLOAT16,
            label + ": logical dtype")
        try require(oracle.keys.shape == view.keys.shape && oracle.values.shape == view.values.shape,
            label + ": concat shape")
        try require(logical.keys.metalBuffer == backing.keys.metalBuffer &&
            logical.values.metalBuffer == backing.values.metalBuffer &&
            logical.keys.offsetBytes == backing.keys.offsetBytes &&
            logical.values.offsetBytes == backing.values.offsetBytes, label + ": logical views alias full backing")
        let visibleBytes = UInt64(state.logicalRows * 2 * 256 * 2)
        let fullBytes = UInt64(state.capacityRows * 2 * 256 * 2)
        try require(logical.keys.logicalBytes == visibleBytes && logical.values.logicalBytes == visibleBytes &&
            backing.keys.logicalBytes == fullBytes && backing.values.logicalBytes == fullBytes,
            label + ": separate logical and capacity bytes")
        try require(backing.keys.contiguous && backing.values.contiguous &&
            backing.keys.rowContiguous && backing.values.rowContiguous &&
            backing.keys.allocationBytes >= fullBytes && backing.values.allocationBytes >= fullBytes,
            label + ": complete contiguous backing")
        let actual = try bytes(view), expected = try bytes(oracle)
        comparedBF16Elements += actual.count / 2
        try require(actual == expected, label + ": complete BF16 bytes versus independent concat")
        try require(actual.count == state.logicalPayloadBytes &&
            state.backingShapeBytes - state.paddingShapeBytes == actual.count,
            label + ": payload contains logical rows only")
    }
    private func initialize(oracle: Pair, rows: Int, limit: Int, label: String) throws -> GPUKVCapacityStorage {
        try ready(oracle)
        let source = try snapshot(oracle)
        try MX.check(mlx_reset_peak_memory(), "capacity initial peak reset")
        let before = try MX.memory(), start = DispatchTime.now().uptimeNanoseconds
        let state = try GPUKVCapacityStorage(compactKeys: oracle.keys, compactValues: oracle.values, rowLimit: limit)
        try ready(state)
        let seconds = elapsed(start), target = try snapshot(state), after = try MX.memory()
        try emit(["event": "initial", "label": label, "rows": rows, "capacity": state.capacityRows,
            "row_limit": limit, "source": source.json, "backing": target.json,
            "initialize_and_eval_seconds": seconds, "memory_before": before, "memory_after": after,
            "raw_peak_bytes": after["peak_bytes"] ?? 0,
            "tracked_active_high_water_bytes": trackedPeak(before: before, after: after)])
        try require(state.logicalRows == rows && state.capacityRows <= limit, label + ": initial extent")
        try require(source.keys.metalBuffer != target.keys.metalBuffer &&
            source.values.metalBuffer != target.values.metalBuffer, label + ": independent initial backing")
        try verify(state, oracle: oracle, label: label + " initial")
        return state
    }
    private func append(_ state: inout GPUKVCapacityStorage, oracle: inout Pair,
                        label: String, retainedAlias: Bool = false) throws {
        let row = try patterns(start: state.logicalRows, count: 1)
        // Finish the independent reference first. Its old graph never refers
        // to a capacity buffer and cannot create a hidden capacity owner.
        oracle = try Pair(keys: MX.concat([oracle.keys,row.keys], axis: 2),
            values: MX.concat([oracle.values,row.values], axis: 2))
        try ready(oracle)
        let old = try snapshot(state), oldRows = state.logicalRows, oldCapacity = state.capacityRows
        let growing = oldRows + 1 > oldCapacity
        try MX.check(mlx_reset_peak_memory(), "capacity append peak reset")
        let before = try MX.memory(), start = DispatchTime.now().uptimeNanoseconds
        // Discard the returned visible handles before evaluation. ready(state)
        // obtains its own short-lived logical roots after append has returned.
        _ = try state.append(keys: row.keys, values: row.values)
        try ready(state)
        let seconds = elapsed(start), next = try snapshot(state), after = try MX.memory()
        appends += 1
        if growing { growths += 1 }
        else if retainedAlias { cows += 1 }
        else { reuses += 1 }
        try emit(["event": "append", "label": label, "old_rows": oldRows, "rows": state.logicalRows,
            "old_capacity": oldCapacity, "capacity": state.capacityRows, "growth": growing,
            "retained_alias": retainedAlias, "old": old.json, "new": next.json,
            "append_and_eval_seconds": seconds, "memory_before": before, "memory_after": after,
            "raw_peak_bytes": after["peak_bytes"] ?? 0,
            "tracked_active_high_water_bytes": trackedPeak(before: before, after: after)])
        try require(state.logicalRows == oldRows + 1 && state.capacityRows <= state.rowLimit,
            label + ": append extent")
        if growing || retainedAlias {
            try require(old.keys.metalBuffer != next.keys.metalBuffer &&
                old.values.metalBuffer != next.values.metalBuffer, label + ": independent growth/COW allocation")
        } else {
            try require(old.sameAllocation(as: next), label + ": actual allocation reuse")
        }
        try verify(state, oracle: oracle, label: label)
    }
    private func sequence(start: Int, end: Int, limit: Int, label: String) throws {
        var oracle = try patterns(count: start)
        var state = try initialize(oracle: oracle, rows: start, limit: limit, label: label)
        for _ in start..<end { try append(&state, oracle: &oracle, label: label) }
    }
    private func keepFullAlias(_ state: inout GPUKVCapacityStorage, oracle: inout Pair) throws {
        // Swift copies share Tensor wrappers, so the diagnostic is_donatable
        // may still be true before constructing the next graph. Actual COW and
        // immutable old bytes are the acceptance checks, not that early flag.
        let alias = state, old = try snapshot(alias), oldBytes = try bytes(alias)
        try withExtendedLifetime(alias) {
            try append(&state, oracle: &oracle, label: "full_helper_alias", retainedAlias: true)
            try require(try snapshot(alias).sameAllocation(as: old), "full helper alias kept old allocation")
            try require(try bytes(alias) == oldBytes, "full helper alias kept all old logical BF16 bytes")
        }
    }
    private func keepLogicalAlias(_ state: inout GPUKVCapacityStorage, oracle: inout Pair) throws {
        guard let alias = try state.logicalView() else { throw GPUError.invalid("Missing retained logical alias") }
        try ready(alias)
        let old = try snapshot(alias), oldBytes = try bytes(alias)
        try withExtendedLifetime(alias) {
            try append(&state, oracle: &oracle, label: "logical_view_alias", retainedAlias: true)
            try require(try snapshot(alias).sameAllocation(as: old), "logical alias kept old allocation")
            try require(try bytes(alias) == oldBytes, "logical alias kept all old logical BF16 bytes")
        }
    }
    private func aliasCase(logical: Bool) throws {
        var oracle = try patterns(count: 2051)
        var state = try initialize(oracle: oracle, rows: 2051, limit: 2304,
            label: logical ? "logical_alias" : "helper_alias")
        if logical { try keepLogicalAlias(&state, oracle: &oracle) }
        else { try keepFullAlias(&state, oracle: &oracle) }
        // Separate helper function ensures the intentionally retained owner is
        // out of scope before constructing this graph, including debug builds.
        try append(&state, oracle: &oracle, label: logical ? "after_logical_alias_release" : "after_helper_alias_release")
    }
    private func clampAndReset() throws {
        var oracle = try patterns(count: 255)
        var state = try initialize(oracle: oracle, rows: 255, limit: 257, label: "clamp_257")
        try append(&state, oracle: &oracle, label: "clamp_257")
        try append(&state, oracle: &oracle, label: "clamp_257")
        try require(state.capacityRows == 257, "capacity clamped exactly at 257")
        let before = try snapshot(state), row = try patterns(start: 257, count: 1)
        var refused = false
        do { _ = try state.append(keys: row.keys, values: row.values) }
        catch { refused = true }
        try require(refused && state.logicalRows == 257 && state.capacityRows == 257 &&
            (try snapshot(state)).sameAllocation(as: before), "bound rejection kept state/allocation")
        try verify(state, oracle: oracle, label: "rejected_258")
        state.reset()
        try require(state.logicalRows == 0 && state.capacityRows == 0 && state.logicalPayloadBytes == 0 &&
            state.backingShapeBytes == 0, "reset extents")
        try require(try state.logicalView() == nil, "reset releases logical view")
        try require(try state.allocationSnapshot(using: diagnostics) == nil, "reset releases backing owner")
        try MX.synchronize()
        try emit(["event": "reset", "memory": try MX.memory(),
            "scope": "Own capacity handles released; oracle/row and allocator cache may still be live"])
        // The same reset helper can accept a fresh first row. This adds one
        // append to the fixed 17-case sequence, not a donation assertion.
        oracle = try patterns(count: 1)
        try ready(oracle)
        _ = try state.append(keys: oracle.keys, values: oracle.values)
        try ready(state); appends += 1
        try emit(["event": "append_from_empty", "rows": state.logicalRows,
            "capacity": state.capacityRows, "backing": try snapshot(state).json])
        try verify(state, oracle: oracle, label: "fresh_after_reset")
    }
    private func firstAppendAtExactCapacity(stageInitialization: Bool) throws {
        // Real first decode can construct conversion and append before a joint
        // eval. At P=256 shape planning uses C0=256 then C1=512; pinned MLX
        // eliminates the full-region C0 slice_update and can alias compact.
        // Record the actual staged identity and measure the whole window;
        // do not infer two allocations just from the two capacity shapes.
        let compact = try patterns(count: 256), row = try patterns(start: 256, count: 1)
        var oracle = try Pair(keys: MX.concat([compact.keys,row.keys], axis: 2),
            values: MX.concat([compact.values,row.values], axis: 2))
        try ready(compact); try ready(oracle)
        let source = try snapshot(compact), beforeBytes = try bytes(compact)
        try MX.check(mlx_reset_peak_memory(), "capacity first decode peak reset")
        let before = try MX.memory(), start = DispatchTime.now().uptimeNanoseconds
        var state = try GPUKVCapacityStorage(compactKeys: compact.keys, compactValues: compact.values, rowLimit: 512)
        let initialBacking: GPUKVAllocationPair?
        if stageInitialization { try ready(state) }
        initialBacking = stageInitialization ? try snapshot(state) : nil
        _ = try state.append(keys: row.keys, values: row.values)
        try ready(state)
        let seconds = elapsed(start), target = try snapshot(state), after = try MX.memory()
        let label = stageInitialization ? "P256_staged_init_append" : "P256_joint_init_append"
        let initialBackingJSON: Any
        if let initialBacking { initialBackingJSON = initialBacking.json }
        else { initialBackingJSON = NSNull() }
        try emit(["event": "first_decode_boundary", "label": label,
            "initialization_evaluated_before_append": stageInitialization,
            "rows": state.logicalRows, "capacity": state.capacityRows,
            "source": source.json, "backing": target.json,
            "staged_initial_backing": initialBackingJSON,
            "initialize_append_eval_seconds": seconds, "memory_before": before, "memory_after": after,
            "raw_peak_bytes": after["peak_bytes"] ?? 0,
            "tracked_active_high_water_bytes": trackedPeak(before: before, after: after),
            "extra_tracked_active_peak_bytes": max(0, trackedPeak(before: before, after: after) - (before["active_bytes"] ?? 0))])
        try require(state.logicalRows == 257 && state.capacityRows == 512, label + ": exact initial boundary")
        try require(source.keys.metalBuffer != target.keys.metalBuffer &&
            source.values.metalBuffer != target.values.metalBuffer, label + ": independent converted backing")
        try require(try bytes(compact) == beforeBytes, label + ": compact source immutable")
        try verify(state, oracle: oracle, label: label)
        let subsequentRow = try patterns(start: 257, count: 1)
        oracle = try Pair(keys: MX.concat([oracle.keys,subsequentRow.keys], axis: 2),
            values: MX.concat([oracle.values,subsequentRow.values], axis: 2))
        try ready(oracle)
        let steadyBefore = try snapshot(state)
        _ = try state.append(keys: subsequentRow.keys, values: subsequentRow.values)
        try ready(state)
        let steadyAfter = try snapshot(state)
        try emit(["event": "first_decode_followup", "label": label,
            "rows": state.logicalRows, "capacity": state.capacityRows,
            "old": steadyBefore.json, "new": steadyAfter.json])
        try require(steadyBefore.sameAllocation(as: steadyAfter), label + ": next row reuses allocation")
        try verify(state, oracle: oracle, label: label + " followup")
    }
    func run() throws {
        try emit(["event": "start", "schema": "qwen-kv-capacity-swift-mechanism-v1",
            "shape": ["1","2","T","256"], "dtype": "BF16", "asynchronous_eval": asynchronous,
            "evaluation_roots": "logical K/V views", "native_abi": 1,
            "native_library": diagnostics.libraryPath,
            "model_executed": false, "sdpa_executed": false, "qsa_indexer_executed": false,
            "physical_dram_bytes_measured": false,
            "memory_scope": "MLX allocator counters include independent oracle; not a request-budget proof"])
        do {
            try sequence(start: 254, end: 258, limit: 512, label: "255_256_257_growth")
            try sequence(start: 2050, end: 2053, limit: 2304, label: "2051_2052")
            try sequence(start: 11230, end: 11233, limit: 11520, label: "11232")
            try aliasCase(logical: false); try aliasCase(logical: true); try clampAndReset()
            try firstAppendAtExactCapacity(stageInitialization: false)
            try firstAppendAtExactCapacity(stageInitialization: true)
            try MX.synchronize()
            try require(appends == 17 && growths == 2 && cows == 2 && reuses == 12,
                "finite coverage: 17 appends = 12 reuse + 2 growth + 2 COW + 1 empty")
            try emit(["event": "summary", "passed": true, "checks": checks, "appends": appends,
                "growths": growths, "cow_appends": cows, "reused_appends": reuses,
                "empty_appends": 1, "compared_bf16_elements": comparedBF16Elements,
                "extra_first_decode_boundary_cases": 2,
                "extra_first_decode_append_operations": 4, "total_append_operations": appends + 4,
                "memory": try MX.memory()])
        } catch {
            try? emit(["event": "summary", "passed": false, "checks": checks, "appends": appends,
                "growths": growths, "cow_appends": cows, "reused_appends": reuses,
                "compared_bf16_elements": comparedBF16Elements, "error": String(describing: error)])
            throw error
        }
    }
}
