import CMLX
import Darwin
import Foundation

/// Optional S1 MoE scheduling experiment, never selected by model defaults.
/// Successfully checked plugin handles remain pinned because lazy arrays own
/// C++ primitive vtables. This wrapper performs no evaluation or synchronization.
public final class GPUMoEDownPairProbe {
    public let libraryPath: String
    public let abiVersion: Int
    private typealias Apply = @convention(c) (UnsafeMutablePointer<mlx_vector_array>?,
        mlx_array, mlx_array, Int32, mlx_stream) -> Int32
    private typealias Count = @convention(c) (Int32) -> UInt64
    private let applyNative: Apply
    private let countNative: Count
    private let lastError: @convention(c) () -> UnsafePointer<CChar>?

    public init(libraryPath: String) throws {
        guard libraryPath.hasPrefix("/") else {
            throw GPUError.invalid("MoE down-pair probe requires an absolute library path")
        }
        let url = URL(fileURLWithPath: libraryPath).resolvingSymlinksInPath()
        guard let handle = dlopen(url.path, RTLD_NOW | RTLD_LOCAL) else {
            let reason = dlerror().map { String(cString: $0) } ?? "unknown loader error"
            throw GPUError.invalid("Cannot load MoE down-pair probe: \(reason)")
        }
        guard let version = dlsym(handle, "anemlx_moe_down_pair_version"),
              let apply = dlsym(handle, "anemlx_moe_down_pair"),
              let count = dlsym(handle, "anemlx_moe_down_pair_count"),
              let error = dlsym(handle, "anemlx_moe_down_pair_last_error") else {
            dlclose(handle)
            throw GPUError.invalid("Incomplete MoE down-pair ABI")
        }
        let abi = Int(unsafeBitCast(version, to: (@convention(c) () -> Int32).self)())
        guard abi == 1 else {
            dlclose(handle)
            throw GPUError.invalid("MoE down-pair probe requires ABI 1")
        }
        self.libraryPath = url.path; abiVersion = abi
        applyNative = unsafeBitCast(apply, to: Apply.self)
        countNative = unsafeBitCast(count, to: Count.self)
        lastError = unsafeBitCast(error, to: (@convention(c) () -> UnsafePointer<CChar>?).self)
        // Intentionally no dlclose after successful validation.
    }

    /// Successfully encoded pairs in [overlap, serial-control] order.
    /// This does not report GPU completion, dispatch count, or physical bytes.
    public func encodedPairCounts() -> [UInt64] { [countNative(0), countNative(1)] }

    func apply(routedRecipe: Tensor, sharedRecipe: Tensor,
               serialControl: Bool) throws -> (routed: Tensor, shared: Tensor) {
        guard routedRecipe.shape == [2560], sharedRecipe.shape == [1,2560],
              routedRecipe.dtype == MLX_BFLOAT16, sharedRecipe.dtype == MLX_BFLOAT16 else {
            throw GPUError.invalid("MoE down-pair recipes have invalid shape/dtype")
        }
        var outputs = mlx_vector_array_new()
        defer { _ = mlx_vector_array_free(outputs) }
        let status = applyNative(&outputs, routedRecipe.handle, sharedRecipe.handle,
                                 serialControl ? 1 : 0, MX.stream)
        guard status == 0 else {
            let reason = lastError().map { String(cString: $0) } ?? "native status \(status)"
            throw GPUError.invalid("MoE down-pair probe: \(reason)")
        }
        guard mlx_vector_array_size(outputs) == 2 else {
            throw GPUError.invalid("MoE down-pair must return two siblings")
        }
        let routed = try MX.output("MoE paired routed down") { mlx_vector_array_get(&$0, outputs, 0) }
        let shared = try MX.output("MoE paired shared down") { mlx_vector_array_get(&$0, outputs, 1) }
        guard routed.shape == routedRecipe.shape, shared.shape == sharedRecipe.shape,
              routed.dtype == MLX_BFLOAT16, shared.dtype == MLX_BFLOAT16 else {
            throw GPUError.invalid("MoE down-pair returned invalid outputs")
        }
        return (routed, shared)
    }
}
