import CMLX
import Foundation

public enum GPUError: Error, LocalizedError {
    case invalid(String)
    public var errorDescription: String? { switch self { case .invalid(let s): s } }
}

/// Owns one MLX C handle. Graph edges retain their device buffers independently
/// of this Swift wrapper. Intentionally single-session, not unchecked-Sendable.
public final class Tensor {
    public let handle: mlx_array
    public init(taking handle: mlx_array) { self.handle = handle }
    deinit { _ = mlx_array_free(handle) }
    public var shape: [Int] { (0..<Int(mlx_array_ndim(handle))).map { Int(mlx_array_dim(handle, Int32($0))) } }
    public var dtype: mlx_dtype { mlx_array_dtype(handle) }
    public var count: Int { Int(mlx_array_size(handle)) }
    public var nbytes: Int { Int(mlx_array_nbytes(handle)) }
    public func eval() throws { try MX.check(mlx_array_eval(handle), "eval") }
    public func floats() throws -> [Float] {
        let array = try MX.contiguous(MX.cast(self, MLX_FLOAT32))
        try array.eval()
        guard let p = mlx_array_data_float32(array.handle) else { throw GPUError.invalid("Missing evaluated float data") }
        return Array(UnsafeBufferPointer(start: p, count: array.count))
    }
    public func ints() throws -> [Int32] {
        let array = try MX.contiguous(MX.cast(self, MLX_INT32))
        try array.eval()
        guard let p = mlx_array_data_int32(array.handle) else { throw GPUError.invalid("Missing evaluated integer data") }
        return Array(UnsafeBufferPointer(start: p, count: array.count))
    }

    /// Read an argmax token without creating a UInt32 -> Int32 GPU cast graph.
    /// The scalar item API evaluates this array if needed; after the caller's
    /// joint token/state evaluation it only reads the existing scalar result.
    public func uint32TokenID() throws -> Int32 {
        guard dtype == MLX_UINT32, count == 1 else {
            throw GPUError.invalid("Token scalar must contain exactly one UInt32 value")
        }
        var value: UInt32 = 0
        try MX.check(mlx_array_item_uint32(&value, handle), "UInt32 token scalar")
        guard let token = Int32(exactly: value) else {
            throw GPUError.invalid("UInt32 token scalar exceeds Int32.max")
        }
        return token
    }
}

/// Thin checked wrappers over the pinned public MLX C API. Hot-path arrays stay
/// on the GPU; host extraction is explicit and used only at output/diagnostics.
public enum MX {
    private static let installHandler: Void = {
        mlx_set_error_handler({ message, _ in
            Thread.current.threadDictionary["ANERunner.MLXError"] = message.map { String(cString: $0) } ?? "Unknown MLX error"
        }, nil, nil)
    }()
    // This is an immutable default-stream handle, used by synchronous sessions.
    // The foreign C struct cannot express Sendable; no model state is global.
    nonisolated(unsafe) public static let stream: mlx_stream = {
        _ = installHandler
        return mlx_default_gpu_stream_new()
    }()
    nonisolated(unsafe) public static let null = mlx_array(ctx: nil)

    public static func check(_ status: Int32, _ operation: String) throws {
        _ = installHandler
        guard status == 0 else {
            let message = Thread.current.threadDictionary["ANERunner.MLXError"] as? String ?? "status \(status)"
            Thread.current.threadDictionary.removeObject(forKey: "ANERunner.MLXError")
            throw GPUError.invalid("MLX \(operation): \(message)")
        }
    }
    public static func check(status: Int32, operation: String) throws { try check(status, operation) }
    public static func output(_ operation: String, _ body: (inout mlx_array) -> Int32) throws -> Tensor {
        _ = installHandler
        var value = mlx_array_new()
        do {
            try check(body(&value), operation)
            guard value.ctx != nil else { throw GPUError.invalid("MLX \(operation): empty result") }
            return Tensor(taking: value)
        } catch { _ = mlx_array_free(value); throw error }
    }
    public static func array(data: Data, shape: [Int], dtype: mlx_dtype) throws -> Tensor {
        _ = installHandler
        var size = 1
        for d in shape {
            guard d >= 0, d <= Int(Int32.max) else { throw GPUError.invalid("Invalid tensor dimension") }
            let next = size.multipliedReportingOverflow(by: d)
            guard !next.overflow else { throw GPUError.invalid("Tensor size overflow") }
            size = next.partialValue
        }
        let bytes = size.multipliedReportingOverflow(by: Int(mlx_dtype_size(dtype)))
        guard !bytes.overflow, bytes.partialValue == data.count else { throw GPUError.invalid("Tensor data/shape mismatch") }
        let dims = shape.map(Int32.init)
        let value = data.withUnsafeBytes { p in mlx_array_new_data(p.baseAddress, dims, Int32(dims.count), dtype) }
        guard value.ctx != nil else { throw GPUError.invalid("Could not create MLX array") }
        return Tensor(taking: value)
    }
    public static func array(_ values: [Float], shape: [Int], dtype: mlx_dtype = MLX_FLOAT32) throws -> Tensor {
        let data = values.withUnsafeBytes { Data($0) }
        return try cast(array(data: data, shape: shape, dtype: MLX_FLOAT32), dtype)
    }
    public static func array(_ values: [Int32], shape: [Int]) throws -> Tensor {
        try array(data: values.withUnsafeBytes { Data($0) }, shape: shape, dtype: MLX_INT32)
    }
    public static func scalar(_ value: Float, _ dtype: mlx_dtype = MLX_FLOAT32) throws -> Tensor {
        try cast(Tensor(taking: mlx_array_new_float32(value)), dtype)
    }
    public static func scalar(_ value: Float, dtype: mlx_dtype) throws -> Tensor { try scalar(value, dtype) }
    public static func zeros(_ shape: [Int], _ dtype: mlx_dtype) throws -> Tensor {
        let s = shape.map(Int32.init)
        return try output("zeros") { mlx_zeros(&$0, s, s.count, dtype, stream) }
    }
    public static func ones(_ shape: [Int], _ dtype: mlx_dtype) throws -> Tensor {
        let s = shape.map(Int32.init)
        return try output("ones") { mlx_ones(&$0, s, s.count, dtype, stream) }
    }
    public static func cast(_ x: Tensor, _ dtype: mlx_dtype) throws -> Tensor {
        if x.dtype == dtype { return x }
        return try output("cast") { mlx_astype(&$0, x.handle, dtype, stream) }
    }
    public static func contiguous(_ x: Tensor) throws -> Tensor {
        try output("contiguous") { mlx_contiguous(&$0, x.handle, false, stream) }
    }
    public static func copy(_ x: Tensor) throws -> Tensor {
        try output("copy") { mlx_copy(&$0, x.handle, stream) }
    }
    public static func reshape(_ x: Tensor, _ shape: [Int]) throws -> Tensor {
        let s = shape.map(Int32.init)
        return try output("reshape") { mlx_reshape(&$0, x.handle, s, s.count, stream) }
    }
    public static func transpose(_ x: Tensor, _ axes: [Int]) throws -> Tensor {
        let a = axes.map(Int32.init)
        return try output("transpose") { mlx_transpose_axes(&$0, x.handle, a, a.count, stream) }
    }
    public static func slice(_ x: Tensor, starts: [Int], ends: [Int], strides: [Int]? = nil) throws -> Tensor {
        let a = starts.map(Int32.init), b = ends.map(Int32.init), c = (strides ?? Array(repeating: 1, count: starts.count)).map(Int32.init)
        guard a.count == x.shape.count, b.count == a.count, c.count == a.count else { throw GPUError.invalid("Slice rank mismatch") }
        return try output("slice") { mlx_slice(&$0, x.handle, a, a.count, b, b.count, c, c.count, stream) }
    }
    public static func concat(_ xs: [Tensor], axis: Int) throws -> Tensor {
        guard !xs.isEmpty else { throw GPUError.invalid("Empty concatenate") }
        let v = mlx_vector_array_new_data(xs.map(\.handle), xs.count)
        defer { _ = mlx_vector_array_free(v) }
        return try output("concat") { mlx_concatenate_axis(&$0, v, Int32(axis), stream) }
    }
    public static func take(_ x: Tensor, _ indices: Tensor, axis: Int) throws -> Tensor {
        try output("take") { mlx_take_axis(&$0, x.handle, indices.handle, Int32(axis), stream) }
    }
    public static func takeAlong(_ x: Tensor, _ indices: Tensor, axis: Int) throws -> Tensor {
        try output("takeAlong") { mlx_take_along_axis(&$0, x.handle, indices.handle, Int32(axis), stream) }
    }
    public static func broadcast(_ x: Tensor, _ shape: [Int]) throws -> Tensor {
        let s = shape.map(Int32.init)
        return try output("broadcast") { mlx_broadcast_to(&$0, x.handle, s, s.count, stream) }
    }
    public static func tile(_ x: Tensor, _ repetitions: [Int]) throws -> Tensor {
        let r = repetitions.map(Int32.init)
        return try output("tile") { mlx_tile(&$0, x.handle, r, r.count, stream) }
    }
    public static func add(_ a: Tensor, _ b: Tensor) throws -> Tensor { try output("add") { mlx_add(&$0, a.handle, b.handle, stream) } }
    public static func sub(_ a: Tensor, _ b: Tensor) throws -> Tensor { try output("sub") { mlx_subtract(&$0, a.handle, b.handle, stream) } }
    public static func mul(_ a: Tensor, _ b: Tensor) throws -> Tensor { try output("mul") { mlx_multiply(&$0, a.handle, b.handle, stream) } }
    public static func div(_ a: Tensor, _ b: Tensor) throws -> Tensor { try output("div") { mlx_divide(&$0, a.handle, b.handle, stream) } }
    public static func matmul(_ a: Tensor, _ b: Tensor) throws -> Tensor { try output("matmul") { mlx_matmul(&$0, a.handle, b.handle, stream) } }
    /// Explicit per-forward verification policy; ordinary AR/prefill and the
    /// draft head keep the pinned MLX path. No process-global numeric mode.
    static func linear(_ a: Tensor, _ b: Tensor, verification: GPUVerificationLinear?) throws -> Tensor {
        if let verification, a.shape.count == 3, a.shape[1] > 1 {
            return try verification.apply(a, weight: b)
        }
        return try matmul(a, b)
    }
    public static func maximum(_ a: Tensor, _ b: Tensor) throws -> Tensor { try output("maximum") { mlx_maximum(&$0, a.handle, b.handle, stream) } }
    public static func minimum(_ a: Tensor, _ b: Tensor) throws -> Tensor { try output("minimum") { mlx_minimum(&$0, a.handle, b.handle, stream) } }
    public static func negative(_ x: Tensor) throws -> Tensor { try output("negative") { mlx_negative(&$0, x.handle, stream) } }
    public static func sigmoid(_ x: Tensor) throws -> Tensor { try output("sigmoid") { mlx_sigmoid(&$0, x.handle, stream) } }
    public static func exp(_ x: Tensor) throws -> Tensor { try output("exp") { mlx_exp(&$0, x.handle, stream) } }
    public static func log1p(_ x: Tensor) throws -> Tensor { try output("log1p") { mlx_log1p(&$0, x.handle, stream) } }
    public static func sqrt(_ x: Tensor) throws -> Tensor { try output("sqrt") { mlx_sqrt(&$0, x.handle, stream) } }
    public static func rsqrt(_ x: Tensor) throws -> Tensor { try output("rsqrt") { mlx_rsqrt(&$0, x.handle, stream) } }
    public static func abs(_ x: Tensor) throws -> Tensor { try output("abs") { mlx_abs(&$0, x.handle, stream) } }
    public static func sign(_ x: Tensor) throws -> Tensor { try output("sign") { mlx_sign(&$0, x.handle, stream) } }
    public static func sin(_ x: Tensor) throws -> Tensor { try output("sin") { mlx_sin(&$0, x.handle, stream) } }
    public static func cos(_ x: Tensor) throws -> Tensor { try output("cos") { mlx_cos(&$0, x.handle, stream) } }
    public static func silu(_ x: Tensor) throws -> Tensor { try mul(x, sigmoid(x)) }
    public static func floorDivide(_ x: Tensor, by value: Int) throws -> Tensor {
        let s = try scalar(Float(value), x.dtype)
        return try output("floorDivide") { mlx_floor_divide(&$0, x.handle, s.handle, stream) }
    }
    public static func sum(_ x: Tensor, axis: Int, keepDims: Bool = false) throws -> Tensor {
        try output("sum") { mlx_sum_axis(&$0, x.handle, Int32(axis), keepDims, stream) }
    }
    public static func mean(_ x: Tensor, axis: Int, keepDims: Bool = false) throws -> Tensor {
        try output("mean") { mlx_mean_axis(&$0, x.handle, Int32(axis), keepDims, stream) }
    }
    public static func softmax(_ x: Tensor, axis: Int = -1, precise: Bool = true) throws -> Tensor {
        try output("softmax") { mlx_softmax_axis(&$0, x.handle, Int32(axis), precise, stream) }
    }
    public static func argsort(_ x: Tensor, axis: Int = -1) throws -> Tensor {
        try output("argsort") { mlx_argsort_axis(&$0, x.handle, Int32(axis), stream) }
    }
    public static func argmax(_ x: Tensor, axis: Int = -1, keepDims: Bool = false) throws -> Tensor {
        try output("argmax") { mlx_argmax_axis(&$0, x.handle, Int32(axis), keepDims, stream) }
    }
    public static func arange(_ start: Float, _ stop: Float, step: Float = 1, dtype: mlx_dtype = MLX_FLOAT32) throws -> Tensor {
        try output("arange") { mlx_arange(&$0, Double(start), Double(stop), Double(step), dtype, stream) }
    }
    public static func lessEqual(_ a: Tensor, _ b: Tensor) throws -> Tensor { try output("lessEqual") { mlx_less_equal(&$0, a.handle, b.handle, stream) } }
    public static func greaterEqual(_ a: Tensor, _ b: Tensor) throws -> Tensor { try output("greaterEqual") { mlx_greater_equal(&$0, a.handle, b.handle, stream) } }
    public static func logicalAnd(_ a: Tensor, _ b: Tensor) throws -> Tensor { try output("and") { mlx_logical_and(&$0, a.handle, b.handle, stream) } }
    public static func logicalOr(_ a: Tensor, _ b: Tensor) throws -> Tensor { try output("or") { mlx_logical_or(&$0, a.handle, b.handle, stream) } }
    public static func whereSelect(_ condition: Tensor, _ a: Tensor, _ b: Tensor) throws -> Tensor { try output("where") { mlx_where(&$0, condition.handle, a.handle, b.handle, stream) } }
    public static func rmsNorm(_ x: Tensor, weight: Tensor?, epsilon: Float) throws -> Tensor {
        try output("rmsNorm") { mlx_fast_rms_norm(&$0, x.handle, weight?.handle ?? null, epsilon, stream) }
    }
    public static func conv1d(_ x: Tensor, weight: Tensor, stride: Int = 1, padding: Int = 0, dilation: Int = 1, groups: Int = 1) throws -> Tensor {
        try output("conv1d") { mlx_conv1d(&$0, x.handle, weight.handle, Int32(stride), Int32(padding), Int32(dilation), Int32(groups), stream) }
    }
    public static func rope(_ x: Tensor, dimensions: Int, base: Float, offset: Int, traditional: Bool = false) throws -> Tensor {
        try output("rope") { mlx_fast_rope(&$0, x.handle, Int32(dimensions), traditional,
            mlx_optional_float(value: base, has_value: true), 1, Int32(offset), null, stream) }
    }
    public static func sdpa(_ q: Tensor, _ k: Tensor, _ v: Tensor, scale: Float, mask: Tensor? = nil, causal: Bool = false, forceFused: Bool = false) throws -> Tensor {
        try output("sdpa") { mlx_fast_scaled_dot_product_attention(&$0, q.handle, k.handle, v.handle, scale,
            mask != nil ? "array" : (causal ? "causal" : ""), mask?.handle ?? null, null, forceFused, stream) }
    }
    public static func gatherQMM(_ x: Tensor, weight: Tensor, scales: Tensor, biases: Tensor, rhsIndices: Tensor, groupSize: Int = 64, bits: Int = 4, sortedIndices: Bool = false) throws -> Tensor {
        try output("gatherQMM") { mlx_gather_qmm(&$0, x.handle, weight.handle, scales.handle, biases.handle, null,
            rhsIndices.handle, true, mlx_optional_int(value: Int32(groupSize), has_value: true),
            mlx_optional_int(value: Int32(bits), has_value: true), "affine", sortedIndices, stream) }
    }
    public static func quantizedMatmul(_ x: Tensor, weight: Tensor, scales: Tensor, biases: Tensor, groupSize: Int = 64, bits: Int = 4) throws -> Tensor {
        try output("quantizedMatmul") { mlx_quantized_matmul(&$0, x.handle, weight.handle, scales.handle, biases.handle, true,
            mlx_optional_int(value: Int32(groupSize), has_value: true), mlx_optional_int(value: Int32(bits), has_value: true), "affine", stream) }
    }
    public static func eval(_ xs: [Tensor]) throws {
        let v = mlx_vector_array_new_data(xs.map(\.handle), xs.count)
        defer { _ = mlx_vector_array_free(v) }
        try check(mlx_eval(v), "eval multiple")
    }
    public static func asyncEval(_ xs: [Tensor]) throws {
        let v = mlx_vector_array_new_data(xs.map(\.handle), xs.count)
        defer { _ = mlx_vector_array_free(v) }
        try check(mlx_async_eval(v), "async eval")
    }
    public static func synchronize() throws { try check(mlx_synchronize(stream), "synchronize") }
    public static func memory() throws -> [String: Int] {
        var active = 0, peak = 0, cache = 0, limit = 0
        try check(mlx_get_active_memory(&active), "active memory")
        try check(mlx_get_peak_memory(&peak), "peak memory")
        try check(mlx_get_cache_memory(&cache), "cache memory")
        try check(mlx_get_memory_limit(&limit), "memory limit")
        return ["active_bytes": active, "peak_bytes": peak, "cache_bytes": cache, "limit_bytes": limit]
    }
}
