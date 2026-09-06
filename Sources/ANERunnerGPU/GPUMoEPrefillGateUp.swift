import CMLX
import Darwin
import Foundation

/// Optional model-specific primitives selected only for prefill. Successful
/// plugin handles stay pinned while lazy graphs may retain their C++ vtables.
public final class GPUMoEPrefillGateUp {
    public let variant: Int
    public let libraryPath: String
    public let abiVersion: Int
    private typealias Apply = @convention(c) (UnsafeMutablePointer<mlx_array>?,
        mlx_array, mlx_array, mlx_array, mlx_array, mlx_array, mlx_array,
        mlx_array, mlx_array, mlx_array, Int32) -> Int32
    private typealias Plan = @convention(c) (UnsafeMutablePointer<mlx_array>?, mlx_array, Int32, mlx_stream) -> Int32
    private typealias Planned = @convention(c) (UnsafeMutablePointer<mlx_array>?,
        mlx_array, mlx_array, mlx_array, mlx_array, mlx_array, mlx_array,
        mlx_array, mlx_array, mlx_array, mlx_array, Int32, mlx_stream) -> Int32
    private typealias Down = @convention(c) (UnsafeMutablePointer<mlx_array>?,
        mlx_array, mlx_array, mlx_array, mlx_array, mlx_array, Int32, mlx_stream) -> Int32
    private typealias Count = @convention(c) (Int32) -> UInt64
    private struct Grouped {
        let plan: Plan, gateUp: Planned, down: Down
        let planCount: Count, downCount: Count
    }
    private let applyNative: Apply
    private let lastError: @convention(c) () -> UnsafePointer<CChar>?
    private let countNative: Count
    private let grouped: Grouped?
    private var blockSize: Int { variant == 2 ? 32 : 16 }

    public init(variant: Int) throws {
        guard (0...3).contains(variant),
              let path = ProcessInfo.processInfo.environment["ANERUNNER_GATEUP_LIBRARY"],
              path.hasPrefix("/") else {
            throw GPUError.invalid("Gate/up fusion requires variant 0...3 and an absolute ANERUNNER_GATEUP_LIBRARY path")
        }
        let url = URL(fileURLWithPath: path).resolvingSymlinksInPath()
        guard let handle = dlopen(url.path, RTLD_NOW | RTLD_LOCAL) else {
            let reason = dlerror().map { String(cString: $0) } ?? "unknown loader error"
            throw GPUError.invalid("Cannot load gate/up primitive: \(reason)")
        }
        guard let version = dlsym(handle, "anemlx_moe_gateup_version"),
              let apply = dlsym(handle, "anemlx_moe_gateup"),
              let error = dlsym(handle, "anemlx_moe_gateup_last_error"),
              let count = dlsym(handle, "anemlx_moe_gateup_dispatch_count") else {
            dlclose(handle); throw GPUError.invalid("Missing gate/up primitive ABI")
        }
        let abi = Int(unsafeBitCast(version, to: (@convention(c) () -> Int32).self)())
        guard [1, 2].contains(abi), variant < 2 || abi == 2 else {
            dlclose(handle); throw GPUError.invalid("Expert-aligned gate/up requires primitive ABI 2")
        }
        if abi == 2 {
            guard let plan = dlsym(handle, "anemlx_moe_expert_plan"),
                  let planned = dlsym(handle, "anemlx_moe_gateup_planned"),
                  let down = dlsym(handle, "anemlx_moe_grouped_down"),
                  let pc = dlsym(handle, "anemlx_moe_expert_plan_dispatch_count"),
                  let dc = dlsym(handle, "anemlx_moe_grouped_down_dispatch_count") else {
                dlclose(handle); throw GPUError.invalid("Incomplete expert-aligned primitive ABI")
            }
            grouped = Grouped(plan: unsafeBitCast(plan, to: Plan.self),
                gateUp: unsafeBitCast(planned, to: Planned.self), down: unsafeBitCast(down, to: Down.self),
                planCount: unsafeBitCast(pc, to: Count.self), downCount: unsafeBitCast(dc, to: Count.self))
        } else { grouped = nil }
        self.variant = variant; libraryPath = url.path; abiVersion = abi
        applyNative = unsafeBitCast(apply, to: Apply.self)
        lastError = unsafeBitCast(error, to: (@convention(c) () -> UnsafePointer<CChar>?).self)
        countNative = unsafeBitCast(count, to: Count.self)
        // Intentionally no dlclose after a successful ABI check.
    }

    public func dispatchCounts() -> [UInt64] {
        (0..<(abiVersion == 2 ? 4 : 2)).map { countNative(Int32($0)) }
    }

    /// Encoded plan32, plan16, down32, down16; not completion or memory bytes.
    public func groupedDispatchCounts() -> [UInt64] {
        guard let grouped else { return [0, 0, 0, 0] }
        return [grouped.planCount(32), grouped.planCount(16), grouped.downCount(32), grouped.downCount(16)]
    }

    private func take(_ status: Int32, _ output: mlx_array, shape: [Int], dtype: mlx_dtype) throws -> Tensor {
        guard status == 0, output.ctx != nil else {
            let reason = lastError().map { String(cString: $0) } ?? "empty native result (\(status))"
            _ = mlx_array_free(output)
            throw GPUError.invalid("Gate/up primitive: \(reason)")
        }
        let result = Tensor(taking: output)
        guard result.shape == shape, result.dtype == dtype else {
            throw GPUError.invalid("Gate/up primitive returned an invalid shape/dtype")
        }
        return result
    }

    func plan(indices: Tensor) throws -> Tensor? {
        guard variant >= 2 else { return nil }
        guard let grouped, indices.shape.count == 1 else { throw GPUError.invalid("Missing expert planner") }
        var output = mlx_array_new()
        let status = grouped.plan(&output, indices.handle, Int32(blockSize), MX.stream)
        let rows = (indices.shape[0] + blockSize - 1) / blockSize + 512
        return try take(status, output, shape: [rows, 4], dtype: MLX_INT32)
    }

    func forward(_ x: Tensor, gateWeight: Tensor, gateScales: Tensor, gateBiases: Tensor,
                 upWeight: Tensor, upScales: Tensor, upBiases: Tensor,
                 indices: Tensor, sigmoidTable: Tensor, plan: Tensor? = nil) throws -> Tensor {
        _ = MX.stream
        var output = mlx_array_new()
        let status: Int32
        if variant >= 2 {
            guard let grouped, let plan else {
                _ = mlx_array_free(output); throw GPUError.invalid("Expert-aligned gate/up requires a plan")
            }
            status = grouped.gateUp(&output, x.handle, gateWeight.handle, gateScales.handle, gateBiases.handle,
                upWeight.handle, upScales.handle, upBiases.handle, indices.handle, sigmoidTable.handle,
                plan.handle, Int32(variant), MX.stream)
        } else {
            status = applyNative(&output, x.handle, gateWeight.handle, gateScales.handle, gateBiases.handle,
                upWeight.handle, upScales.handle, upBiases.handle, indices.handle, sigmoidTable.handle, Int32(variant))
        }
        return try take(status, output, shape: [x.shape[0], 1, 640], dtype: MLX_BFLOAT16)
    }

    func down(_ activation: Tensor, weight: Tensor, scales: Tensor, biases: Tensor, plan: Tensor) throws -> Tensor {
        guard variant >= 2, let grouped else { throw GPUError.invalid("Grouped down requires a plan-capable primitive") }
        var output = mlx_array_new()
        let status = grouped.down(&output, activation.handle, weight.handle, scales.handle, biases.handle,
            plan.handle, Int32(blockSize), MX.stream)
        return try take(status, output, shape: [activation.shape[0], 1, 2560], dtype: MLX_BFLOAT16)
    }
}
