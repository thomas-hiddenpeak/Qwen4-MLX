import CMLX

// Candidate only. This file is not part of the runner target until reviewed.
extension MX {
    /// Exact-shape, same-dtype replacement of one nonempty slice. Broadcasting,
    /// casts and strided writes are deliberately excluded from this KV helper.
    /// This creates a functional MLX graph; it does not promise buffer donation.
    static func sliceUpdate(_ source: Tensor, update: Tensor,
                            starts: [Int], ends: [Int]) throws -> Tensor {
        let shape = source.shape, updateShape = update.shape
        guard !shape.isEmpty, starts.count == shape.count, ends.count == shape.count,
              updateShape.count == shape.count, source.dtype == update.dtype else {
            throw GPUError.invalid("Slice update rank or dtype mismatch")
        }
        for axis in shape.indices {
            guard starts[axis] >= 0, ends[axis] > starts[axis], ends[axis] <= shape[axis],
                  updateShape[axis] == ends[axis] - starts[axis], ends[axis] <= Int(Int32.max) else {
                throw GPUError.invalid("Slice update must match a bounded, nonempty region exactly")
            }
        }
        let start = starts.map(Int32.init), end = ends.map(Int32.init)
        let strides = Array(repeating: Int32(1), count: shape.count)
        return try output("slice update") {
            mlx_slice_update(&$0, source.handle, update.handle,
                start, start.count, end, end.count, strides, strides.count, stream)
        }
    }
}
