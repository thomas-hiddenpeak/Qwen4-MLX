import CMLX
import Dispatch
import Foundation

public enum GPUWiredPolicy: String, Codable, Sendable {
    case disabled
    case fit
}

/// A successful process-local MLX residency-cap change, not a measurement of
/// physical DRAM traffic, resident bytes, or the allocator's memory limit.
public struct GPUWiredMemoryReport: Codable, Sendable {
    public let policy: GPUWiredPolicy
    public let activeBytes: Int
    public let maximumRecommendedWorkingSetBytes: Int
    public let previousLimitBytes: Int
    public let targetLimitBytes: Int
    public let requestedSlackBytes: Int
    public let recommendedReserveBytes: Int
    public let actualHeadroomBytes: Int?
    public let cacheCleared: Bool
    public let setupDurationMilliseconds: Double
    public let residencyScope: String
}

/// Call from the sole inference owner after weight evaluation and before the
/// measured request. Synchronizes MX's default stream; callers must not have
/// concurrent MLX work on other streams or mutate residency from other threads.
///
/// The fit-policy sequence follows mlx-serve src/mlx.zig:588-645; this runner
/// requests 256 MiB of slack. Upstream attribution: Copyright (c) 2026 David
/// Dalcu, MIT; see ../qwen38-ssd/runtime/mlx-serve/LICENSE from the package root.
public enum GPUWiredMemory {
    public static let fitSlackBytes = 256 * 1024 * 1024
    public static let recommendedReserveBytes = 256 * 1024 * 1024

    /// Pure CPU validation/calculation. Rejects an empty live set or a cap that
    /// cannot hold it; a smaller positive slack is allowed near the maximum.
    public static func fitTarget(
        activeBytes: Int,
        maximumRecommendedWorkingSetBytes: Int
    ) throws -> Int {
        guard activeBytes > 0 else {
            throw GPUError.invalid("Wired fit requires a nonempty evaluated live set")
        }
        guard maximumRecommendedWorkingSetBytes > recommendedReserveBytes else {
            throw GPUError.invalid("Recommended working set leaves no room after the 256 MiB reserve")
        }
        let cap = maximumRecommendedWorkingSetBytes - recommendedReserveBytes
        guard activeBytes <= cap else {
            throw GPUError.invalid("Active MLX memory exceeds the wired-fit cap after the 256 MiB reserve")
        }
        let (requested, overflow) = activeBytes.addingReportingOverflow(fitSlackBytes)
        guard !overflow else {
            throw GPUError.invalid("Wired-fit active memory plus slack overflows Int")
        }
        return min(requested, cap)
    }

    public static func apply(_ policy: GPUWiredPolicy) throws -> GPUWiredMemoryReport {
        let start = DispatchTime.now().uptimeNanoseconds
        // check() installs the nonterminating error handler without accessing a
        // device. Do this BEFORE any C call whose failure invokes mlx_error.
        try MX.check(0, "install MLX error handler for wired policy")
        try MX.synchronize()
        if policy == .fit {
            try MX.check(mlx_clear_cache(), "clear cached buffers before wired fit")
        }

        let maximum = try maximumRecommendedWorkingSet()
        var active = 0
        try MX.check(mlx_get_active_memory(&active), "query active memory for wired policy")
        guard active >= 0 else {
            throw GPUError.invalid("MLX returned a negative active-memory size")
        }
        let target: Int
        if policy == .fit {
            target = try fitTarget(activeBytes: active, maximumRecommendedWorkingSetBytes: maximum)
        } else {
            target = 0
        }

        var previous = 0
        try MX.check(mlx_set_wired_limit(&previous, 0), "disable previous wired residency cap")
        if policy == .fit {
            var intermediateLimit = 0
            do {
                // Shrink then grow re-walks existing live allocations, including
                // weights loaded since a previous policy application.
                try MX.check(mlx_set_wired_limit(&intermediateLimit, target), "apply wired-fit residency cap")
            } catch {
                let originalError = error.localizedDescription
                var rollbackPrevious = 0
                do {
                    try MX.check(mlx_set_wired_limit(&rollbackPrevious, previous), "restore previous wired cap after failure")
                } catch {
                    throw GPUError.invalid("Wired fit failed: \(originalError). Restoring previous cap also failed: \(error.localizedDescription)")
                }
                throw GPUError.invalid("Wired fit failed: \(originalError). Previous residency cap was restored")
            }
        }

        return GPUWiredMemoryReport(
            policy: policy,
            activeBytes: active,
            maximumRecommendedWorkingSetBytes: maximum,
            previousLimitBytes: previous,
            targetLimitBytes: target,
            requestedSlackBytes: policy == .fit ? fitSlackBytes : 0,
            recommendedReserveBytes: policy == .fit ? recommendedReserveBytes : 0,
            actualHeadroomBytes: policy == .fit ? target - active : nil,
            cacheCleared: policy == .fit,
            setupDurationMilliseconds: Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000,
            residencyScope: "Process-local MLX Metal residency-set capacity; not physical DRAM traffic or measured resident bytes. Allocator memory limit is unchanged."
        )
    }

    private static func maximumRecommendedWorkingSet() throws -> Int {
        let device = mlx_device_new_type(MLX_GPU, 0)
        guard device.ctx != nil else {
            throw GPUError.invalid("MLX did not create the GPU device handle for working-set query")
        }
        defer { _ = mlx_device_free(device) }
        // new() deliberately creates an empty handle; get() populates it.
        var info = mlx_device_info_new()
        defer { _ = mlx_device_info_free(info) }
        try MX.check(mlx_device_info_get(&info, device), "query GPU device info")
        var maximum = 0
        try MX.check(
            mlx_device_info_get_size(&maximum, info, "max_recommended_working_set_size"),
            "query maximum recommended GPU working set"
        )
        guard maximum > 0 else {
            throw GPUError.invalid("GPU maximum recommended working set is unavailable or zero")
        }
        return maximum
    }
}
