import CMLX

/// Candidate storage for this model's BF16 B1/Hkv2/D256 ordinary AR decode.
///
/// Value fields own complete buffers; logical views are returned separately.
/// Copies of this struct retain immutable MLX handles. A live old state/view
/// must force MLX's functional copy fallback instead of mutating that alias.
/// No pointer writes, synchronization, model hooks or cache leases occur here.
struct GPUKVCapacityStorage {
    struct LogicalView {
        let keys: Tensor
        let values: Tensor
        var tensors: [Tensor] { [keys, values] }
    }

    static let growthRows = 256
    static let maximumRows = 262_144
    private static let bytesPerRow = 2 * 2 * 256 * 2 // K/V × heads × dimensions × BF16

    let rowLimit: Int
    private(set) var logicalRows = 0
    private(set) var capacityRows = 0
    private var backing: LogicalView?

    // Shape accounting only. MLX allocation size, actual copies/donation and
    // extra aliases are separate observations, not inferred from these values.
    var logicalPayloadBytes: Int { logicalRows * Self.bytesPerRow }
    var backingShapeBytes: Int { capacityRows * Self.bytesPerRow }
    var paddingShapeBytes: Int { backingShapeBytes - logicalPayloadBytes }

    init(rowLimit: Int) throws {
        guard rowLimit > 0, rowLimit <= Self.maximumRows else {
            throw GPUError.invalid("KV capacity needs a positive admitted row limit within context")
        }
        self.rowLimit = rowLimit
    }

    /// Places a compact/logical prefix in capacity-shaped functional buffers.
    /// At rows == capacity, pinned MLX slice_update returns the update itself;
    /// initialization can alias the source instead of allocating. A later
    /// append then grows (or is refused at rowLimit), preserving that source.
    /// With padding, conversion is lazy and included in decode timing.
    init(compactKeys: Tensor, compactValues: Tensor, rowLimit: Int) throws {
        try self.init(rowLimit: rowLimit)
        let rows = compactKeys.shape.count == 4 ? compactKeys.shape[2] : 0
        let compact = LogicalView(keys: compactKeys, values: compactValues)
        try Self.validate(compact, rows: rows)
        let capacity = try Self.capacity(requiredRows: rows, rowLimit: rowLimit)
        let full = try Self.copyIntoNewBacking(compact, rows: rows, capacity: capacity)
        backing = full; logicalRows = rows; capacityRows = capacity
    }

    /// Host-only arithmetic. The allocation cannot exceed the request's bound,
    /// including a final partial growth quantum such as 256 -> 257.
    static func capacity(requiredRows: Int, rowLimit: Int) throws -> Int {
        guard rowLimit > 0, rowLimit <= maximumRows,
              requiredRows > 0, requiredRows <= rowLimit else {
            throw GPUError.invalid("KV capacity exceeds the admitted row limit")
        }
        // Validate first; bounded integers cannot overflow the rounded value.
        return min(((requiredRows - 1) / growthRows + 1) * growthRows, rowLimit)
    }

    /// Returns logical shapes only. The returned Tensor handles must leave
    /// scope before the next append if donation is desired. Keeping them is
    /// correct but may require copying the complete backing allocation.
    func logicalView() throws -> LogicalView? {
        guard let backing else { return nil }
        return try Self.view(backing, rows: logicalRows)
    }

    /// Candidate probe hook only: observe existing full buffers without making
    /// another array/view or letting a backing owner escape this method.
    func allocationSnapshot(using diagnostics: GPUKVAllocationDiagnostics) throws -> GPUKVAllocationPair? {
        guard let backing else { return nil }
        return try GPUKVAllocationPair(keys: diagnostics.snapshot(backing.keys),
            values: diagnostics.snapshot(backing.values))
    }

    /// Appends exactly one AR row. All old local handles leave this function
    /// before its caller can evaluate the returned graph. No unaliased reuse
    /// is claimed until the native allocation identity is observed after eval.
    @discardableResult
    mutating func append(keys: Tensor, values: Tensor) throws -> LogicalView {
        let row = LogicalView(keys: keys, values: values)
        try Self.validate(row, rows: 1)
        guard logicalRows < rowLimit else {
            throw GPUError.invalid("KV append exceeds the admitted row limit")
        }
        let nextRows = logicalRows + 1
        let nextCapacity: Int
        if nextRows > capacityRows {
            nextCapacity = try Self.capacity(requiredRows: nextRows, rowLimit: rowLimit)
        } else {
            nextCapacity = capacityRows
        }
        let base: LogicalView
        if let backing {
            if nextCapacity > capacityRows {
                let oldLogical = try Self.view(backing, rows: logicalRows)
                base = try Self.copyIntoNewBacking(oldLogical, rows: logicalRows, capacity: nextCapacity)
            } else {
                base = backing
            }
        } else {
            base = try Self.zeros(capacity: nextCapacity)
        }
        let next = LogicalView(
            keys: try MX.sliceUpdate(base.keys, update: row.keys,
                starts: [0,0,logicalRows,0], ends: [1,2,nextRows,256]),
            values: try MX.sliceUpdate(base.values, update: row.values,
                starts: [0,0,logicalRows,0], ends: [1,2,nextRows,256]))
        let visible = try Self.view(next, rows: nextRows)
        // Commit only after both lazy update/view graphs were constructed.
        // Existing aliases retain base independently and stay immutable.
        backing = next; logicalRows = nextRows; capacityRows = nextCapacity
        return visible
    }

    /// Releases this owner's handles only; pending graphs/external aliases keep
    /// their own storage alive. Request leases must follow the existing device
    /// completion rules outside this helper, not this host method's return.
    mutating func reset() { backing = nil; logicalRows = 0; capacityRows = 0 }

    private static func validate(_ pair: LogicalView, rows: Int) throws {
        let shape = [1,2,rows,256]
        guard rows > 0, pair.keys.shape == shape, pair.values.shape == shape,
              pair.keys.dtype == MLX_BFLOAT16, pair.values.dtype == MLX_BFLOAT16 else {
            throw GPUError.invalid("KV capacity requires matching BF16 [1,2,T,256] tensors")
        }
    }
    private static func zeros(capacity: Int) throws -> LogicalView {
        LogicalView(keys: try MX.zeros([1,2,capacity,256], MLX_BFLOAT16),
            values: try MX.zeros([1,2,capacity,256], MLX_BFLOAT16))
    }
    private static func copyIntoNewBacking(_ source: LogicalView, rows: Int,
                                           capacity: Int) throws -> LogicalView {
        let empty = try zeros(capacity: capacity)
        return LogicalView(
            keys: try MX.sliceUpdate(empty.keys, update: source.keys,
                starts: [0,0,0,0], ends: [1,2,rows,256]),
            values: try MX.sliceUpdate(empty.values, update: source.values,
                starts: [0,0,0,0], ends: [1,2,rows,256]))
    }
    private static func view(_ source: LogicalView, rows: Int) throws -> LogicalView {
        LogicalView(keys: try MX.slice(source.keys, starts: [0,0,0,0], ends: [1,2,rows,256]),
            values: try MX.slice(source.values, starts: [0,0,0,0], ends: [1,2,rows,256]))
    }
}
