// Qwen4 hyper connections follow garnermccloud/mlx-serve transformer.zig,
// commit 7dbcba04c98e4fd3bcc533c63e645547f13cc3b1 (MIT; see UPSTREAM-LICENSE).
import CMLX
import Foundation

public final class GPUHyperConnection {
    public struct Read { public let mixed: Tensor; public let injection: Tensor? }
    private let down, up, norm: Tensor
    private let inject: Tensor?
    private let hidden, streams: Int
    private let epsilon: Float
    private let fused: GPUHyperConnectionFused?
    public let fuseDecodeProjections: Bool
    /// Additional resident BF16 copy only; zero for disabled mode or the mixer.
    public let additionalProjectionBufferBytes: UInt64
    private let decodeProjectionBuffers: HCDecodeProjectionBuffers?

    public init(weights: GPUWeights, prefix: String, hidden: Int = 2560, streams: Int = 4,
                epsilon: Float = 1e-6, withInjection: Bool = true, fused: GPUHyperConnectionFused? = nil,
                fuseDecodeProjections: Bool = false) throws {
        self.hidden = hidden; self.streams = streams; self.epsilon = epsilon
        self.fused = fused
        self.fuseDecodeProjections = fuseDecodeProjections
        let scale = try MX.scalar(1 / Float(streams), MLX_BFLOAT16)
        let dw = try weights.tensor(prefix + ".input_mix_weight_down.weight")
        let uw = try weights.tensor(prefix + ".input_mix_weight_up.weight")
        guard dw.dtype == MLX_BFLOAT16, uw.dtype == MLX_BFLOAT16,
              dw.shape.count == 2, dw.shape[1] == hidden * streams,
              uw.shape == [hidden * streams, dw.shape[0]] else { throw GPUError.invalid("Invalid HC projections: \(prefix)") }
        down = try MX.transpose(MX.mul(dw, scale), [1, 0])
        up = try MX.transpose(uw, [1, 0])
        norm = try MX.reshape(weights.tensor(prefix + ".hc_norm.weight"), [streams, hidden])
        if withInjection {
            let iw = try weights.tensor(prefix + ".block_inject_weight.weight")
            guard iw.shape == [streams, streams * hidden], iw.dtype == MLX_BFLOAT16 else { throw GPUError.invalid("Invalid HC injection") }
            inject = try MX.transpose(MX.mul(iw, scale), [1, 0])
        } else { inject = nil }
        try MX.eval([down, up, norm] + [inject].compactMap { $0 })
        let packed = fuseDecodeProjections ? try HCDecodeProjectionBuffers(down: down,injection: inject) : nil
        decodeProjectionBuffers = packed
        additionalProjectionBufferBytes = packed?.byteCount ?? 0
    }

    /// RMS normalization rounds to BF16 before the separate learned multiply.
    public static func groupNorm(_ x: Tensor, weight: Tensor, streams: Int = 4,
                                 hidden: Int = 2560, epsilon: Float = 1e-6) throws -> Tensor {
        let shape = x.shape
        guard shape.count == 3, shape[2] == streams * hidden else { throw GPUError.invalid("Invalid HC stream shape") }
        let x4 = try MX.reshape(x, [shape[0], shape[1], streams, hidden])
        return try MX.mul(MX.rmsNorm(x4, weight: nil, epsilon: epsilon), weight)
    }
    public func read(_ stream: Tensor, useFusedProjections: Bool? = nil,
                     verificationLinear: GPUVerificationLinear? = nil) throws -> Read {
        let usePacked = try DecodeProjectionSelection.use(requested: useFusedProjections,prepared: fuseDecodeProjections)
        let shape = stream.shape, n4 = try Self.groupNorm(stream, weight: norm, streams: streams, hidden: hidden, epsilon: epsilon)
        let flat = try MX.reshape(n4, shape)
        let projected: Tensor
        let preparedInjection: Tensor?
        if usePacked, shape == [1,1,10240], let decodeProjectionBuffers {
            let projections = try decodeProjectionBuffers.project(flat)
            projected = projections.down; preparedInjection = projections.injection
        } else {
            projected = try MX.linear(flat, down, verification: verificationLinear)
            preparedInjection = nil
        }
        let activated = try fused?.silu(projected) ?? MX.silu(projected)
        let up4 = try MX.reshape(MX.linear(activated, up, verification: verificationLinear), [shape[0], shape[1], streams, hidden])
        let mixed = try fused?.readMix(normalized: n4, upOutput: up4) ?? MX.mean(MX.mul(n4, MX.sigmoid(up4)), axis: 2)
        let injection: Tensor?
        if let inject {
            let logits = try preparedInjection ?? MX.linear(flat, inject, verification: verificationLinear)
            injection = try fused?.injection(logits) ?? MX.reshape(MX.mul(MX.sigmoid(logits), MX.scalar(2, MLX_BFLOAT16)), [shape[0], shape[1], streams, 1])
        } else { injection = nil }
        return Read(mixed: mixed, injection: injection)
    }
    public func write(_ stream: Tensor, output: Tensor, injection: Tensor) throws -> Tensor {
        if let fused { return try fused.write(stream: stream, output: output, injection: injection) }
        let s = stream.shape
        let x4 = try MX.reshape(stream, [s[0], s[1], streams, hidden])
        let out4 = try MX.reshape(output, [s[0], s[1], 1, hidden])
        return try MX.reshape(MX.add(x4, MX.mul(out4, injection)), s)
    }
}

/// The 320 down rows and four injection rows share the same 10240-wide input.
/// All three shapes select the pinned MLX GEMV BM1/BN8 reduction. Inputs have
/// already undergone the original BF16 1/4 scaling; no arithmetic is folded.
struct HCDecodeProjectionBuffers {
    struct Output { let down, injection: Tensor }
    private let packed: Tensor
    let byteCount: UInt64

    /// A mixer has no injection and deliberately retains its original path.
    init?(down: Tensor, injection: Tensor?) throws {
        guard let injection else { return nil }
        guard down.shape == [10240,320], injection.shape == [10240,4],
              down.dtype == MLX_BFLOAT16, injection.dtype == MLX_BFLOAT16 else {
            throw GPUError.invalid("HC decode projection packing requires BF16 down/injection matrices")
        }
        let rows = try MX.concat([MX.transpose(down,[1,0]),MX.transpose(injection,[1,0])],axis: 0)
        packed = try MX.transpose(rows,[1,0])
        try packed.eval()
        byteCount = UInt64(packed.nbytes)
    }

    func project(_ x: Tensor) throws -> Output {
        guard x.shape == [1,1,10240], x.dtype == MLX_BFLOAT16 else {
            throw GPUError.invalid("Packed HC projections support BF16 single-token decode only")
        }
        let output = try MX.matmul(x,packed)
        return try Output(
            down: MX.slice(output,starts: [0,0,0],ends: [1,1,320]),
            injection: MX.slice(output,starts: [0,0,320],ends: [1,1,324]))
    }
}
