import ANERunnerCore
import CMLX
import Foundation

/// An immutable host-only archive. This is the boundary passed to disk I/O;
/// Tensor and model State remain confined to the inference executor.
public struct QwenPrefixStateArchive: Sendable {
    public let metadata: Data
    public let payload: Data
    public let logicalPayloadBytes: Int

    public init(metadata: Data, payload: Data, logicalPayloadBytes: Int) {
        self.metadata = metadata; self.payload = payload
        self.logicalPayloadBytes = logicalPayloadBytes
    }
}

enum QwenPrefixStateArchiveBytes {
    /// Preserve BF16 bits, including their exact rounding. Casting to Float
    /// would create a different payload and hide dtype/copy mistakes.
    static func append(_ tensor: Tensor, to payload: inout Data) throws {
        let contiguous = try MX.contiguous(tensor)
        let bytes = try MX.output("prefix archive byte view") {
            mlx_view(&$0, contiguous.handle, MLX_UINT8, MX.stream)
        }
        try bytes.eval()
        guard bytes.count == tensor.nbytes,
              let pointer = mlx_array_data_uint8(bytes.handle) else {
            throw GPUError.invalid("Missing evaluated prefix tensor bytes")
        }
        payload.append(pointer, count: bytes.count)
    }
}
