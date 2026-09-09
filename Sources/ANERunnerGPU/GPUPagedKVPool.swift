import CMLX
import Darwin
import Foundation

/// Bounded physical K/V page arena experiment. All calls and State handles are
/// confined to the owning inference executor. Native completion handlers retain
/// the pool and page leases independently of these Swift wrapper lifetimes.
/// This does not change QwenModel.State, prefix archives, or serving defaults.
public final class GPUPagedKVPool {
    public let libraryPath: String
    public let abiVersion: Int
    public let maximumPages: Int
    public var maximumTokens: Int { maximumPages * 32 }

    public struct Statistics: Codable, Equatable, Sendable {
        public let physicalPages, arenaLogicalBytes, arenaAllocatedBytes: UInt64
        /// Occupied slots include graph and GPU completion pins. They are not
        /// the complete arena allocation, which remains fixed while the pool lives.
        public let livePages, freePages, highWaterPages: UInt64
        public let encodedWrites, encodedReads, encodedMaterializations: UInt64
        public let copiedTailBytes, writtenRowBytes, materializedBytes: UInt64
        public let inFlightOperations, completedOperations, failedOperations: UInt64
        /// Diagnostic stable arena identities, not dereferenceable CPU pointers.
        public let keyBufferIdentity, valueBufferIdentity: UInt64
        public var liveUniquePageBytes: UInt64 { livePages * 65_536 }
    }

    private typealias Create = @convention(c) (UnsafeMutablePointer<UnsafeMutableRawPointer?>?, Int32, mlx_stream) -> Int32
    private typealias Destroy = @convention(c) (UnsafeMutableRawPointer?) -> Void
    private typealias Import = @convention(c) (UnsafeMutablePointer<UnsafeMutableRawPointer?>?, UnsafeRawPointer?, mlx_array, mlx_array) -> Int32
    private typealias Fork = @convention(c) (UnsafeMutablePointer<UnsafeMutableRawPointer?>?, UnsafeRawPointer?) -> Int32
    private typealias Scalars = @convention(c) (UnsafeRawPointer?, UnsafeMutablePointer<UInt64>?, Int) -> Int32
    private typealias PageIDs = @convention(c) (UnsafeRawPointer?, UnsafeMutablePointer<Int32>?, Int) -> Int32
    private typealias Ready = @convention(c) (UnsafeMutablePointer<mlx_array>?, UnsafeRawPointer?) -> Int32
    private typealias Read = @convention(c) (UnsafeMutablePointer<mlx_array>?, UnsafeRawPointer?, mlx_array, mlx_array) -> Int32
    private typealias Materialize = @convention(c) (UnsafeMutablePointer<mlx_array>?, UnsafeMutablePointer<mlx_array>?, UnsafeRawPointer?) -> Int32
    private let context: UnsafeMutableRawPointer
    private let destroyNative, destroyStateNative: Destroy
    private let importNative, appendNative: Import
    private let forkNative: Fork
    private let infoNative, statisticsNative: Scalars
    private let pageIDsNative: PageIDs
    private let readyNative: Ready
    private let readNative: Read
    private let materializeNative: Materialize
    private let errorNative: @convention(c) () -> UnsafePointer<CChar>?

    public init(libraryPath: String, maximumPages: Int) throws {
        guard libraryPath.hasPrefix("/"), (1...4096).contains(maximumPages) else {
            throw GPUError.invalid("Physical KV pool requires absolute library path and maximumPages 1...4096")
        }
        let path = URL(fileURLWithPath: libraryPath).resolvingSymlinksInPath().path
        guard let handle = dlopen(path, RTLD_NOW | RTLD_LOCAL) else {
            throw GPUError.invalid("Cannot load physical KV pool: \(dlerror().map { String(cString: $0) } ?? "loader error")")
        }
        let names = ["version", "create", "free", "import", "fork", "append", "state_free",
                     "state_info", "page_ids", "ready", "read", "materialize", "statistics", "last_error"]
        var symbols: [UnsafeMutableRawPointer] = []
        for name in names {
            guard let symbol = dlsym(handle, "anemlx_paged_kv_pool_" + name) else {
                dlclose(handle)
                throw GPUError.invalid("Incomplete physical KV pool ABI: \(name)")
            }
            symbols.append(symbol)
        }
        let version = Int(unsafeBitCast(symbols[0], to: (@convention(c) () -> Int32).self)())
        guard version == 1 else {
            dlclose(handle)
            throw GPUError.invalid("Physical KV pool requires ABI 1")
        }
        // Pin the successful DSO before constructing any native graph. Even a
        // constructor failure may leave a submitted graph using its vtables.
        let create = unsafeBitCast(symbols[1], to: Create.self)
        let destroy = unsafeBitCast(symbols[2], to: Destroy.self)
        let error = unsafeBitCast(symbols[13], to: (@convention(c) () -> UnsafePointer<CChar>?).self)
        var native: UnsafeMutableRawPointer?
        let status = create(&native, Int32(maximumPages), MX.stream)
        guard status == 0, let native else {
            if let native { destroy(native) }
            throw GPUError.invalid("Physical KV pool initialization: \(error().map { String(cString: $0) } ?? "empty pool (\(status))")")
        }
        self.libraryPath = path; self.maximumPages = maximumPages; abiVersion = version
        context = native; destroyNative = destroy
        importNative = unsafeBitCast(symbols[3], to: Import.self)
        forkNative = unsafeBitCast(symbols[4], to: Fork.self)
        appendNative = unsafeBitCast(symbols[5], to: Import.self)
        destroyStateNative = unsafeBitCast(symbols[6], to: Destroy.self)
        infoNative = unsafeBitCast(symbols[7], to: Scalars.self)
        pageIDsNative = unsafeBitCast(symbols[8], to: PageIDs.self)
        readyNative = unsafeBitCast(symbols[9], to: Ready.self)
        readNative = unsafeBitCast(symbols[10], to: Read.self)
        materializeNative = unsafeBitCast(symbols[11], to: Materialize.self)
        statisticsNative = unsafeBitCast(symbols[12], to: Scalars.self)
        errorNative = error
    }

    deinit { destroyNative(context) } // No dlclose; graphs may outlive the pool wrapper.

    /// Snapshot only. Encoding counters are not completion counters. Arena
    /// allocation remains charged even when every slot is available for reuse.
    public var statistics: Statistics {
        get throws {
            var v = [UInt64](repeating: 0, count: 17)
            let status = v.withUnsafeMutableBufferPointer { statisticsNative(UnsafeRawPointer(context), $0.baseAddress, $0.count) }
            guard status == 0 else { throw nativeError(status, operation: "statistics") }
            guard v[0] == UInt64(maximumPages), v[1] == v[0] * 65_536,
                  v[2] >= v[1], v[3] <= v[0], v[4] == v[0] - v[3],
                  v[5] >= v[3], v[5] <= v[0], v[15] != 0, v[16] != 0 else {
                throw GPUError.invalid("Physical KV pool returned inconsistent statistics")
            }
            return Statistics(physicalPages: v[0], arenaLogicalBytes: v[1], arenaAllocatedBytes: v[2],
                livePages: v[3], freePages: v[4], highWaterPages: v[5],
                encodedWrites: v[6], encodedReads: v[7], encodedMaterializations: v[8],
                copiedTailBytes: v[9], writtenRowBytes: v[10], materializedBytes: v[11],
                inFlightOperations: v[12], completedOperations: v[13], failedOperations: v[14],
                keyBufferIdentity: v[15], valueBufferIdentity: v[16])
        }
    }

    public func importState(keys: Tensor, values: Tensor) throws -> State {
        let shape = keys.shape
        guard shape.count == 4, shape[0] == 1, shape[1] == 2, shape[3] == 256,
              (1...maximumTokens).contains(shape[2]), values.shape == shape,
              keys.dtype == MLX_BFLOAT16, values.dtype == MLX_BFLOAT16 else {
            throw GPUError.invalid("Physical KV import requires bounded BF16 [1,2,T,256]")
        }
        return try createState(operation: "import") { importNative(&$0, UnsafeRawPointer(context), keys.handle, values.handle) }
    }

    private func createState(operation: String, _ body: (inout UnsafeMutableRawPointer?) -> Int32) throws -> State {
        var result: UnsafeMutableRawPointer?
        let status = body(&result)
        guard status == 0, let result else {
            if let result { destroyStateNative(result) }
            throw nativeError(status, operation: operation)
        }
        var fields = [UInt64](repeating: 0, count: 2)
        let infoStatus = fields.withUnsafeMutableBufferPointer { infoNative(UnsafeRawPointer(result), $0.baseAddress, $0.count) }
        guard infoStatus == 0, fields[0] <= UInt64(maximumTokens), fields[0] > 0,
              fields[1] == (fields[0] + 31) / 32 else {
            destroyStateNative(result)
            throw nativeError(infoStatus, operation: "state metadata")
        }
        return State(pool: self, context: result, logicalTokens: Int(fields[0]), pageCount: Int(fields[1]))
    }

    private func nativeError(_ status: Int32, operation: String) -> GPUError {
        let message = errorNative().map { String(cString: $0) } ?? "native status \(status)"
        return GPUError.invalid("Physical KV \(operation): \(message)")
    }

    /// Immutable native page-list handle. Assignment shares the wrapper; fork()
    /// creates an independently releasable handle without copying K/V or page IDs.
    public final class State {
        public let logicalTokens: Int
        public let pageCount: Int
        public var logicalKVBytes: Int { logicalTokens * 2_048 }
        private let pool: GPUPagedKVPool
        private let context: UnsafeMutableRawPointer
        fileprivate init(pool: GPUPagedKVPool, context: UnsafeMutableRawPointer, logicalTokens: Int, pageCount: Int) {
            self.pool = pool; self.context = context; self.logicalTokens = logicalTokens; self.pageCount = pageCount
        }
        deinit { pool.destroyStateNative(context) }

        /// Diagnostic host metadata only; allocating this array is intentionally
        /// excluded from the O(1) fork and steady-state reader measurement.
        public var pageIDs: [Int32] {
            get throws {
                var pages = [Int32](repeating: 0, count: pageCount)
                let status = pages.withUnsafeMutableBufferPointer { pool.pageIDsNative(UnsafeRawPointer(context), $0.baseAddress, $0.count) }
                guard status == 0 else { throw pool.nativeError(status, operation: "page IDs") }
                guard pages.allSatisfy({ $0 >= 0 && Int($0) < pool.maximumPages }) else {
                    throw GPUError.invalid("Physical KV returned an out-of-pool page ID")
                }
                return pages
            }
        }

        public func fork() throws -> State {
            try pool.createState(operation: "fork") { pool.forkNative(&$0, UnsafeRawPointer(context)) }
        }

        public func append(keys: Tensor, values: Tensor) throws -> State {
            guard logicalTokens < pool.maximumTokens, keys.shape == [1,2,1,256], values.shape == keys.shape,
                  keys.dtype == MLX_BFLOAT16, values.dtype == MLX_BFLOAT16 else {
                throw GPUError.invalid("Physical KV append requires BF16 [1,2,1,256] within the initialized context")
            }
            return try pool.createState(operation: "append") { pool.appendNative(&$0, UnsafeRawPointer(context), keys.handle, values.handle) }
        }

        /// Returns the import/write graph dependency. Evaluate it to join those
        /// writes without adding an SDPA read; returning the Tensor does not wait.
        public func ready() throws -> Tensor {
            try output(operation: "ready") { pool.readyNative(&$0, UnsafeRawPointer(context)) }
        }

        public func read(queries: Tensor, mask: Tensor? = nil) throws -> Tensor {
            guard queries.shape == [1,24,1,256], queries.dtype == MLX_BFLOAT16,
                  mask == nil || (mask!.shape == [1,1,1,logicalTokens] && mask!.dtype == MLX_BOOL) else {
                throw GPUError.invalid("Physical KV read requires BF16 Q[1,24,1,256] and logical boolean QSA mask")
            }
            let result = try output(operation: "read") {
                pool.readNative(&$0, UnsafeRawPointer(context), queries.handle, mask?.handle ?? MX.null)
            }
            guard result.shape == queries.shape, result.dtype == MLX_BFLOAT16 else {
                throw GPUError.invalid("Physical KV reader returned unexpected shape/dtype")
            }
            return result
        }

        /// Explicit compact export for diagnostics/archive adaptation. This
        /// allocates a full logical K/V copy when evaluated; never call per token.
        public func materialize() throws -> (keys: Tensor, values: Tensor) {
            var keys = mlx_array_new(), values = mlx_array_new()
            let status = pool.materializeNative(&keys, &values, UnsafeRawPointer(context))
            guard status == 0, keys.ctx != nil, values.ctx != nil else {
                _ = mlx_array_free(keys); _ = mlx_array_free(values)
                throw pool.nativeError(status, operation: "materialize")
            }
            let k = Tensor(taking: keys), v = Tensor(taking: values)
            guard k.shape == [1,2,logicalTokens,256], v.shape == k.shape,
                  k.dtype == MLX_BFLOAT16, v.dtype == MLX_BFLOAT16 else {
                throw GPUError.invalid("Physical KV materialization returned unexpected shape/dtype")
            }
            return (k, v)
        }

        private func output(operation: String, _ body: (inout mlx_array) -> Int32) throws -> Tensor {
            var result = mlx_array_new()
            let status = body(&result)
            guard status == 0, result.ctx != nil else {
                _ = mlx_array_free(result)
                throw pool.nativeError(status, operation: operation)
            }
            return Tensor(taking: result)
        }
    }
}
