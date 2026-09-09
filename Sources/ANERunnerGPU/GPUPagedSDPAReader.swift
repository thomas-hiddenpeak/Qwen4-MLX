import CMLX
import Darwin
import Foundation

/// Explicit single-token AR experiment. The owning inference executor must
/// serialize calls, as for Tensor/model state; this class is not Sendable.
/// Successful ABI handles stay pinned because lazy arrays retain C++ vtables.
/// The identity page table is initialized once. apply performs no evaluation,
/// synchronization, page-table rebuild, K/V pack, or K/V gather.
public final class GPUPagedSDPAReader {
    public let libraryPath: String
    public let abiVersion: Int
    public let maximumTokens: Int
    /// Int32 identity payload only, at most 16384 bytes; not RSS or allocation
    /// high water. The native constructor also uses a temporary vector of IDs.
    public let identityMetadataLogicalBytes: UInt64

    public struct DispatchPlan: Equatable, Sendable {
        public let twoPass: Bool
        public let blocks: Int
        /// Partial BF16 output plus Float32 sums/maxs. Excludes output (12288
        /// bytes), allocator rounding/cache, metadata and overlapping graphs.
        public let logicalScratchBytes: UInt64
    }

    private typealias Create = @convention(c) (UnsafeMutablePointer<UnsafeMutableRawPointer?>?, Int32) -> Int32
    private typealias Destroy = @convention(c) (UnsafeMutableRawPointer?) -> Void
    private typealias MetadataBytes = @convention(c) (UnsafeRawPointer?) -> UInt64
    private typealias Apply = @convention(c) (UnsafeMutablePointer<mlx_array>?, UnsafeRawPointer?,
        mlx_array, mlx_array, mlx_array, mlx_array, mlx_stream) -> Int32
    private typealias Plan = @convention(c) (Int32, mlx_stream, UnsafeMutablePointer<UInt64>?, Int) -> Int32
    private let context: UnsafeMutableRawPointer
    private let destroyNative: Destroy
    private let applyNative: Apply
    private let planNative: Plan
    private let countNative: @convention(c) () -> UInt64
    private let lastError: @convention(c) () -> UnsafePointer<CChar>?

    public init(libraryPath: String, maxTokens: Int = 131_072) throws {
        let maximumTokens = maxTokens
        guard libraryPath.hasPrefix("/"), (1...131_072).contains(maximumTokens) else {
            throw GPUError.invalid("Paged SDPA requires an absolute library path and maximumTokens 1...131072")
        }
        let path = URL(fileURLWithPath: libraryPath).resolvingSymlinksInPath().path
        guard let handle = dlopen(path, RTLD_NOW | RTLD_LOCAL) else {
            let reason = dlerror().map { String(cString: $0) } ?? "unknown loader error"
            throw GPUError.invalid("Cannot load paged SDPA reader: \(reason)")
        }
        guard let version = dlsym(handle, "anemlx_paged_sdpa_version"),
              let create = dlsym(handle, "anemlx_paged_sdpa_create"),
              let destroy = dlsym(handle, "anemlx_paged_sdpa_free"),
              let metadata = dlsym(handle, "anemlx_paged_sdpa_metadata_bytes"),
              let apply = dlsym(handle, "anemlx_paged_sdpa_read"),
              let plan = dlsym(handle, "anemlx_paged_sdpa_dispatch_info"),
              let count = dlsym(handle, "anemlx_paged_sdpa_encoded_reads"),
              let error = dlsym(handle, "anemlx_paged_sdpa_last_error") else {
            dlclose(handle)
            throw GPUError.invalid("Incomplete paged SDPA reader ABI")
        }
        let abi = Int(unsafeBitCast(version, to: (@convention(c) () -> Int32).self)())
        guard abi == 1 else {
            dlclose(handle)
            throw GPUError.invalid("Paged SDPA reader requires ABI 1")
        }
        // Pin every successfully checked DSO, including a later constructor
        // failure. Only the small per-instance identity table is destroyed.
        let createNative = unsafeBitCast(create, to: Create.self)
        let destroyNative = unsafeBitCast(destroy, to: Destroy.self)
        let errorNative = unsafeBitCast(error, to: (@convention(c) () -> UnsafePointer<CChar>?).self)
        var nativeContext: UnsafeMutableRawPointer?
        let status = createNative(&nativeContext, Int32(maximumTokens))
        guard status == 0, let nativeContext else {
            if let nativeContext { destroyNative(nativeContext) }
            let reason = errorNative().map { String(cString: $0) } ?? "empty native context (\(status))"
            throw GPUError.invalid("Paged SDPA identity initialization: \(reason)")
        }
        let metadataBytes = unsafeBitCast(metadata, to: MetadataBytes.self)(UnsafeRawPointer(nativeContext))
        guard metadataBytes == UInt64((maximumTokens + 31) / 32) * 4 else {
            destroyNative(nativeContext)
            throw GPUError.invalid("Paged SDPA reader returned invalid identity metadata size")
        }
        self.libraryPath = path; abiVersion = abi
        self.maximumTokens = maximumTokens; identityMetadataLogicalBytes = metadataBytes
        context = nativeContext; self.destroyNative = destroyNative
        applyNative = unsafeBitCast(apply, to: Apply.self)
        planNative = unsafeBitCast(plan, to: Plan.self)
        countNative = unsafeBitCast(count, to: (@convention(c) () -> UInt64).self)
        lastError = errorNative
    }

    deinit { destroyNative(context) } // Intentionally no dlclose.

    /// Monotonic DSO-wide successful reader encodings, not GPU completion.
    /// Take before/after deltas under exclusive model ownership; other reader
    /// instances and generic page-major calls in the same DSO also count.
    public var encodedReads: UInt64 { countNative() }

    /// Diagnostic policy query only. Callers must not mutate MLX_SDPA_BLOCKS
    /// between this observation and evaluation of already-constructed graphs.
    public func dispatchPlan(tokens: Int) throws -> DispatchPlan {
        guard (1...maximumTokens).contains(tokens) else {
            throw GPUError.invalid("Paged SDPA dispatch length exceeds initialized metadata bound")
        }
        var fields = [UInt64](repeating: 0, count: 3)
        let status = fields.withUnsafeMutableBufferPointer {
            planNative(Int32(tokens), MX.stream, $0.baseAddress, $0.count)
        }
        guard status == 0 else { throw nativeError(status, operation: "dispatch policy") }
        guard fields[0] <= 1, fields[1] <= 4096,
              fields[0] == 1 ? (fields[1] > 0 && fields[1] % 32 == 0 && fields[2] == 24 * fields[1] * 520)
                            : (fields[1] == 0 && fields[2] == 0) else {
            throw GPUError.invalid("Paged SDPA returned an invalid dispatch plan")
        }
        return DispatchPlan(twoPass: fields[0] == 1, blocks: Int(fields[1]), logicalScratchBytes: fields[2])
    }

    func apply(queries: Tensor, keys: Tensor, values: Tensor, mask: Tensor? = nil) throws -> Tensor {
        let keyShape = keys.shape
        guard queries.shape == [1,24,1,256], queries.dtype == MLX_BFLOAT16,
              keyShape.count == 4, keyShape[0] == 1, keyShape[1] == 2, keyShape[3] == 256,
              (1...maximumTokens).contains(keyShape[2]), values.shape == keyShape,
              keys.dtype == MLX_BFLOAT16, values.dtype == MLX_BFLOAT16,
              mask == nil || (mask!.shape == [1,1,1,keyShape[2]] && mask!.dtype == MLX_BOOL) else {
            throw GPUError.invalid("Paged SDPA requires BF16 S1/GQA24:2/D256 and an optional logical boolean QSA mask")
        }
        // Layout is checked after lazy inputs materialize by the native reader.
        // In particular head stride may be padded capacity C*256, not T*256.
        var output = mlx_array_new()
        let status = applyNative(&output, UnsafeRawPointer(context), queries.handle, keys.handle,
                                 values.handle, mask?.handle ?? MX.null, MX.stream)
        guard status == 0, output.ctx != nil else {
            _ = mlx_array_free(output)
            throw nativeError(status, operation: "read graph")
        }
        let result = Tensor(taking: output)
        guard result.shape == queries.shape, result.dtype == MLX_BFLOAT16 else {
            throw GPUError.invalid("Paged SDPA reader returned invalid output shape/dtype")
        }
        return result
    }

    private func nativeError(_ status: Int32, operation: String) -> GPUError {
        let reason = lastError().map { String(cString: $0) } ?? "native status \(status)"
        return GPUError.invalid("Paged SDPA \(operation): \(reason)")
    }
}
