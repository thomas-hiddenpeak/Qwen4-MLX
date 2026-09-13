import ANERunnerCore
import CMLX
import Darwin
import Foundation

/// Model-free, single-layer physical-sharing and immutable-state diagnostic.
/// The caller owns the exclusive GPU window. Readback is never throughput evidence.
public enum GPUPagedKVPoolMechanismProbe {
    public static func run(libraryPath: String, asynchronous: Bool,
                           onEvent: @escaping (Data) throws -> Void) throws {
        try PagePoolRun(libraryPath: libraryPath, asynchronous: asynchronous, onEvent: onEvent).run()
    }
}

private final class PagePoolRun {
    typealias Pair = (keys: Tensor, values: Tensor)
    typealias State = GPUPagedKVPool.State
    let library: String, asynchronous: Bool
    let onEvent: (Data) throws -> Void
    var checks = 0, comparedElements = 0
    init(libraryPath: String, asynchronous: Bool, onEvent: @escaping (Data) throws -> Void) {
        library = libraryPath; self.asynchronous = asynchronous; self.onEvent = onEvent
    }
    func emit(_ object: [String: Any]) throws {
        var data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        data.append(10); try onEvent(data)
    }
    func object<T: Encodable>(_ value: T) throws -> Any {
        try JSONSerialization.jsonObject(with: JSONEncoder().encode(value))
    }
    func require(_ passed: Bool, _ label: String) throws {
        checks += 1
        guard passed else { throw GPUError.invalid("Physical page pool probe: " + label) }
    }
    func ready(_ tensors: [Tensor]) throws {
        if asynchronous { try MX.asyncEval(tensors) } else { try MX.eval(tensors) }
        try MX.synchronize()
    }
    func pattern(start: Int, count: Int, salt: Int, heads: Int = 2) throws -> Tensor {
        var values = [Float](repeating: 0, count: heads * count * 256)
        for h in 0..<heads { for t in 0..<count { for d in 0..<256 {
            values[(h * count + t) * 256 + d] = Float(((start+t)*43+h*71+d*19+salt)%509-254)/128
        } } }
        return try MX.array(values, shape: [1,heads,count,256], dtype: MLX_BFLOAT16)
    }
    func pair(start: Int = 0, count: Int, branch: Int = 0) throws -> Pair {
        try (pattern(start: start, count: count, salt: 11+branch*37),
             pattern(start: start, count: count, salt: 137+branch*53))
    }
    func concatenate(_ a: Pair, _ b: Pair) throws -> Pair {
        try (MX.concat([a.keys,b.keys], axis: 2), MX.concat([a.values,b.values], axis: 2))
    }
    func bytes(_ value: Tensor) throws -> Data {
        var data = Data(); try QwenPrefixStateArchiveBytes.append(value, to: &data); return data
    }
    func equal(_ a: Tensor, _ b: Tensor, _ label: String) throws {
        try ready([a,b]); try require(a.shape == b.shape && a.dtype == b.dtype, label + " shape/dtype")
        let lhs = try bytes(a), rhs = try bytes(b); comparedElements += lhs.count/2
        try require(lhs == rhs, label + " BF16 bytes")
    }
    func mask(_ n: Int) throws -> Tensor {
        var visible = Data(repeating: 0, count: n)
        // Independent representative visibility; this does not run learned QSA.
        for t in 0..<n where (t/4)%5 == 0 || t >= n-4 { visible[t] = 1 }
        return try MX.array(data: visible, shape: [1,1,1,n], dtype: MLX_BOOL)
    }
    func verifyRead(_ state: State, _ expected: Pair, _ query: Tensor) throws {
        let visibility = try mask(state.logicalTokens)
        let actual = try state.read(queries: query, mask: visibility)
        let oracle = try MX.sdpa(query, expected.keys, expected.values, scale: 1/16, mask: visibility)
        try equal(actual, oracle, "direct page-table SDPA")
    }
    func verifyExport(_ state: State, _ expected: Pair) throws {
        let actual = try state.materialize()
        try equal(actual.keys, expected.keys, "explicit K export")
        try equal(actual.values, expected.values, "explicit V export")
    }
    func drained(_ pool: GPUPagedKVPool, _ label: String) throws {
        try MX.synchronize()
        let s = try pool.statistics
        try require(s.livePages == 0 && s.freePages == s.physicalPages && s.inFlightOperations == 0,
                    label + " all page leases returned")
        try require(s.failedOperations == 0 && s.completedOperations ==
            s.encodedWrites+s.encodedReads+s.encodedMaterializations, label + " completion accounting")
    }
    func boundaryWork(_ pool: GPUPagedKVPool, _ n: Int, _ query: Tensor) throws {
        let source = try pair(count: n), rowA = try pair(start: n, count: 1, branch: 1)
        let rowB = try pair(start: n, count: 1, branch: 2)
        let seed = try pool.importState(keys: source.keys, values: source.values)
        let before = try pool.statistics, ids = try seed.pageIDs
        let fork = try seed.fork()
        try require(try fork.pageIDs == ids && pool.statistics == before, "fork is metadata-only")
        let a = try seed.append(keys: rowA.keys, values: rowA.values)
        let b = try fork.append(keys: rowB.keys, values: rowB.values)
        // Evaluate both branches together: no CPU wait was inserted after import.
        try ready([try a.ready(), try b.ready()])
        let end = try pool.statistics
        try require(end.encodedWrites == 3 && end.copiedTailBytes == UInt64(2*(n%32)*2048), "bounded tail copy")
        try require(Array(try a.pageIDs.prefix(n/32)) == Array(ids.prefix(n/32)) &&
                    Array(try b.pageIDs.prefix(n/32)) == Array(ids.prefix(n/32)), "full prefix pages shared")
        try verifyExport(seed, source)
        try verifyExport(a, concatenate(source,rowA)); try verifyExport(b, concatenate(source,rowB))
        try verifyRead(a, concatenate(source,rowA), query); try verifyRead(b, concatenate(source,rowB), query)
    }
    func boundaries(_ query: Tensor) throws {
        for n in [31,32,33,2051,2052] {
            let pool = try GPUPagedKVPool(libraryPath: library, maximumPages: (n+31)/32+4)
            try boundaryWork(pool,n,query); try drained(pool,"boundary \(n)")
            try emit(["event":"boundary","tokens":n,"passed":true,"statistics":try object(pool.statistics)])
        }
    }
    func longWork(_ pool: GPUPagedKVPool, _ query: Tensor) throws {
        let n = 11057, suffix = 64, branchCount = 4
        let source = try pair(count: n)
        let seed = try pool.importState(keys: source.keys, values: source.values)
        try ready([try seed.ready()]); try verifyRead(seed, source, query)
        let initial = try pool.statistics, prefixIDs = try seed.pageIDs
        var states = try (0..<branchCount).map { _ in try seed.fork() }
        var expected = [Pair](repeating: source, count: branchCount)
        try require(try pool.statistics.livePages == initial.livePages, "four forks allocate no physical pages")
        var appendReadSeconds = 0.0
        for step in 0..<suffix { for branch in 0..<branchCount {
            let row = try pair(start: n+step, count: 1, branch: branch+1)
            expected[branch] = try concatenate(expected[branch],row)
            try ready([expected[branch].keys,expected[branch].values]) // Oracle is outside candidate interval.
            let visibility = try mask(n+step+1), before = try pool.statistics
            let start = DispatchTime.now().uptimeNanoseconds
            states[branch] = try states[branch].append(keys: row.keys, values: row.values)
            let actual = try states[branch].read(queries: query, mask: visibility)
            try ready([actual])
            appendReadSeconds += Double(DispatchTime.now().uptimeNanoseconds-start)*1e-9
            let after = try pool.statistics
            try require(after.keyBufferIdentity == initial.keyBufferIdentity &&
                after.valueBufferIdentity == initial.valueBufferIdentity &&
                after.arenaAllocatedBytes == initial.arenaAllocatedBytes, "fixed arena reused")
            try require(after.encodedWrites == before.encodedWrites+1 && after.encodedReads == before.encodedReads+1 &&
                after.encodedMaterializations == before.encodedMaterializations, "direct decode has no materialize")
            try require(after.copiedTailBytes-before.copiedTailBytes == UInt64((n+step)%32*2048), "exact encoded COW bytes")
            let oracle = try MX.sdpa(query,expected[branch].keys,expected[branch].values,scale:1/16,mask:visibility)
            try equal(actual,oracle,"11k branched decode")
        } }
        for branch in 0..<branchCount {
            try require(Array(try states[branch].pageIDs.prefix(n/32)) == Array(prefixIDs.prefix(n/32)), "long shared prefix IDs")
            try verifyExport(states[branch],expected[branch])
        }
        try verifyExport(seed,source)
        var unique = Set(prefixIDs)
        for state in states { unique.formUnion(try state.pageIDs) }
        let s = try pool.statistics
        try require(s.livePages == UInt64(unique.count), "unique physical pages counted once")
        let copiedRows = (0..<suffix).reduce(0) { $0+(n+$1)%32 }*branchCount
        try require(s.copiedTailBytes == UInt64(copiedRows*2048) && s.writtenRowBytes == UInt64((n+suffix*branchCount)*2048), "complete GPU copy/write totals")
        let privateLogicalBytes = branchCount*(n+suffix)*2048
        try require(s.arenaAllocatedBytes < UInt64(privateLogicalBytes), "fixed arena smaller than four independent logical KVs")
        try emit(["event":"long_four_branches","passed":true,"prompt_tokens":n,"branches":branchCount,
            "suffix_tokens_each":suffix,"shared_full_prefix_pages":n/32,"unique_pages":unique.count,
            "independent_four_KV_logical_bytes":privateLogicalBytes,"statistics":try object(s),
            "live_unique_page_bytes":s.liveUniquePageBytes,"diagnostic_append_read_seconds":appendReadSeconds,
            "model_loaded":false,"rss_savings_measured":false,"throughput_benchmark":false])
    }
    func oomWork(_ pool: GPUPagedKVPool, _ query: Tensor) throws {
        let source = try pair(count:31), row = try pair(start:31,count:1), row2 = try pair(start:32,count:1)
        var seed: State? = try pool.importState(keys:source.keys,values:source.values)
        try ready([try seed!.ready()])
        let oldIDs = try seed!.pageIDs
        var heldRead: Tensor? = try seed!.read(queries:query)
        let child = try seed!.append(keys:row.keys,values:row.values)
        try ready([try child.ready()]); seed = nil
        try require(try pool.statistics.livePages == 2, "lazy read pins old page after Swift state release")
        let before = try pool.statistics
        var refused = false
        do { _ = try child.append(keys:row2.keys,values:row2.values) }
        catch { refused = String(describing: error).contains("insufficient free KV page slots") }
        try require(refused && (try pool.statistics) == before, "OOM leaves pages and GPU counters unchanged")
        // Preserve and evaluate the old read before allowing its page to be reused.
        let expectedRead = try MX.sdpa(query,source.keys,source.values,scale:1/16)
        try equal(heldRead!,expectedRead,"read retained through pending COW and OOM")
        heldRead = nil; try MX.synchronize()
        let grandchild = try child.append(keys:row2.keys,values:row2.values)
        try ready([try grandchild.ready()])
        try require(try grandchild.pageIDs.last == oldIDs.first, "released old slot reused")
        try verifyExport(grandchild,concatenate(concatenate(source,row),row2))
    }
    func discardedWork(_ pool: GPUPagedKVPool, _ query: Tensor) throws {
        let source = try pair(count:33), row = try pair(start:33,count:1)
        let state = try pool.importState(keys:source.keys,values:source.values)
        let next = try state.append(keys:row.keys,values:row.values)
        _ = try next.read(queries:query)
        _ = try next.materialize()
        try require(try pool.statistics.encodedWrites == 0, "construction remains lazy")
    }
    final class WeakPool { weak var value: GPUPagedKVPool? }
    func deferredRead(_ query: Tensor, _ witness: WeakPool) throws -> Tensor {
        let pool = try GPUPagedKVPool(libraryPath:library,maximumPages:4); witness.value = pool
        let source = try pair(count:33), row = try pair(start:33,count:1)
        let seed = try pool.importState(keys:source.keys,values:source.values)
        let next = try seed.append(keys:row.keys,values:row.values)
        return try next.read(queries:query)
    }
    func deferredBudgetedRead(_ query: Tensor, _ witness: WeakPool,
                              _ budget: QwenStateBudget) throws -> Tensor {
        let pool = try GPUPagedKVPool(libraryPath:library,maximumPages:4,stateBudget:budget)
        witness.value = pool
        try require(pool.admittedBytes == GPUPagedKVPool.reservedBytes(maximumPages:4),
                    "native pool reports exactly the admitted arena bytes")
        let source = try pair(count:33), row = try pair(start:33,count:1)
        let seed = try pool.importState(keys:source.keys,values:source.values)
        return try seed.append(keys:row.keys,values:row.values).read(queries:query)
    }
    func completedBudgetedWork(_ pool: GPUPagedKVPool, _ query: Tensor) throws {
        let source = try pair(count:33)
        let state = try pool.importState(keys:source.keys,values:source.values)
        try ready([try state.read(queries:query)])
    }
    func emptyBudgetedPool(_ query: Tensor, _ budget: QwenStateBudget,
                           _ expectedCharge: Int) throws -> StatisticsSnapshot {
        let pool = try GPUPagedKVPool(libraryPath:library,maximumPages:4,stateBudget:budget)
        return try withExtendedLifetime(pool) {
            try completedBudgetedWork(pool,query)
            try drained(pool,"budgeted completed state")
            try require(budget.statistics.workspaceBytes == expectedCharge,
                        "held empty pool keeps fixed arena charged")
            return StatisticsSnapshot(pool: try pool.statistics, budget: budget.statistics)
        }
    }
    struct StatisticsSnapshot: Encodable {
        let pool: GPUPagedKVPool.Statistics
        let budget: QwenStateBudget.Statistics
    }
    func budgetLifetime(_ query: Tensor) throws {
        let admitted = try GPUPagedKVPool.reservedBytes(maximumPages:4)
        // Two compact states conservatively cover this small diagnostic's
        // dynamic lists/tables/source rows; the pool lease covers arenas only.
        let dynamicBytes = 2 * 34 * 2048
        let budget = try QwenStateBudget(maxBytes:admitted+dynamicBytes)
        guard let dynamic = budget.reserve(bytes:dynamicBytes,kind:.workspace) else {
            throw GPUError.invalid("Physical page pool probe could not reserve diagnostic graph workspace")
        }
        let witness = WeakPool()
        var output: Tensor? = try deferredBudgetedRead(query,witness,budget)
        try require(witness.value == nil,"budgeted Swift pool/state wrappers gone before evaluation")
        try require(budget.statistics.workspaceBytes == admitted+dynamicBytes &&
                    budget.statistics.currentLeases == 2,
                    "lazy native graph retains arena reservation after Swift wrappers disappear")
        let lazy = budget.statistics
        let source = try pair(count:33), row = try pair(start:33,count:1)
        let expected = try concatenate(source,row)
        let oracle = try MX.sdpa(query,expected.keys,expected.values,scale:1/16)
        try equal(output!,oracle,"budgeted deferred graph remains valid")
        output = nil; try MX.synchronize()
        try require(budget.statistics.workspaceBytes == dynamicBytes &&
                    budget.statistics.currentLeases == 1,
                    "last native graph/completion releases arena reservation")
        let releasedGraph = budget.statistics
        let empty = try emptyBudgetedPool(query,budget,admitted+dynamicBytes)
        try require(budget.statistics.workspaceBytes == dynamicBytes,
                    "empty pool destruction returns arena reservation")
        dynamic.release()
        try require(budget.statistics.totalBytes == 0 && budget.statistics.currentLeases == 0,
                    "completed diagnostic returns all state reservations")
        try emit(["event":"budget_lifetime","passed":true,
            "fixed_arena_admitted_bytes":admitted,"dynamic_graph_workspace_bytes":dynamicBytes,
            "lazy_without_swift_wrappers":try object(lazy),"graph_released":try object(releasedGraph),
            "held_empty_pool":try object(empty),"final_budget":try object(budget.statistics),
            "rss_limit_verified":false])
    }
    final class ReleaseWitness { var calls = 0 }
    func ownedFailurePaths() throws {
        typealias Release = @convention(c) (UnsafeMutableRawPointer?) -> Void
        typealias CreateOwned = @convention(c) (UnsafeMutablePointer<UnsafeMutableRawPointer?>?, Int32,
            mlx_stream, UnsafeMutableRawPointer?, Release?) -> Int32
        guard let handle = dlopen(library,RTLD_NOW|RTLD_LOCAL) else {
            throw GPUError.invalid("Cannot reopen pool ABI for owner failure-path probe")
        }
        defer { dlclose(handle) } // The successful pool initializer already pins this DSO.
        guard let symbol = dlsym(handle,"anemlx_paged_kv_pool_create_owned") else {
            throw GPUError.invalid("Missing owned pool ABI in failure-path probe")
        }
        let create = unsafeBitCast(symbol,to:CreateOwned.self)
        for invalidOutput in [false,true] {
            let witness = ReleaseWitness()
            // Keep a separate observable object alive while native consumes its
            // retained reference, including the rejected output-pointer path.
            let retained = Unmanaged.passRetained(witness).toOpaque()
            var output: UnsafeMutableRawPointer?
            let release: Release = { value in
                if let value { Unmanaged<ReleaseWitness>.fromOpaque(value).takeRetainedValue().calls += 1 }
            }
            let status: Int32
            if invalidOutput { status = create(nil,4,mlx_stream(ctx:nil),retained,release) }
            else { status = create(&output,4,mlx_stream(ctx:nil),retained,release) }
            try require(status != 0 && output == nil && witness.calls == 1,
                        "rejected native pool consumes owner exactly once")
        }
        try emit(["event":"owned_create_failures","passed":true,"cases":2,"gpu_work_submitted":false])
    }
    func run() throws {
        // First callback occurs after the DSO loads, before any GPU dispatch,
        // so the CLI can verify that it did not load a second MLX runtime.
        let large = try GPUPagedKVPool(libraryPath:library,maximumPages:384)
        try emit(["event":"start","async_eval":asynchronous,"model_loaded":false,"page_tokens":32])
        try ownedFailurePaths()
        let query = try pattern(start:0,count:1,salt:7,heads:24)
        try budgetLifetime(query)
        try boundaries(query)
        try longWork(large,query); try drained(large,"four branches released")
        try emit(["event":"long_drained","statistics":try object(large.statistics)])
        let small = try GPUPagedKVPool(libraryPath:library,maximumPages:2)
        try oomWork(small,query); try drained(small,"OOM/reuse")
        try emit(["event":"oom_and_reuse","passed":true,"statistics":try object(small.statistics)])
        let unused = try GPUPagedKVPool(libraryPath:library,maximumPages:4)
        try discardedWork(unused,query); try drained(unused,"discarded lazy graphs")
        try require(try unused.statistics.encodedWrites == 0 && unused.statistics.completedOperations == 0, "discarded graphs never encoded")
        let witness = WeakPool(), output = try deferredRead(query,witness)
        try require(witness.value == nil,"Swift pool/state wrappers released before evaluation")
        let source = try pair(count:33), row = try pair(start:33,count:1)
        let expected = try concatenate(source,row)
        let oracle = try MX.sdpa(query,expected.keys,expected.values,scale:1/16)
        try equal(output,oracle,"native graph outlives all Swift pool/state wrappers")
        try emit(["event":"summary","passed":true,"checks":checks,"compared_bf16_elements":comparedElements,
                  "model_correctness_verified":false,"learned_qsa_indexer_executed":false])
    }
}
