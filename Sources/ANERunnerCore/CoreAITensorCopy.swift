#if canImport(CoreAI)
import CoreAI
import Foundation

/// Independent storage for sequence checkpoints. It deliberately makes no assumption
/// about the copy-on-write or shared-buffer semantics of NDArray's value copy.
@available(macOS 27.0, *)
enum CoreAITensorCopy {
    enum CopyError: LocalizedError {
        case unsupported(String)
        var errorDescription: String? {
            switch self { case .unsupported(let message): message }
        }
    }

    /// Copies logical values without serializing them, changing dtype, or assuming
    /// that the source is contiguous. The returned storage is contiguous row-major.
    static func deepCopy(_ source: NDArray) throws -> NDArray {
        let shape = source.shape, strides = source.strides
        guard source.interleaveLayout == nil else {
            throw CopyError.unsupported("Interleaved CoreAI state cannot be copied by this probe")
        }
        guard !shape.isEmpty, shape.count == strides.count,
              shape.allSatisfy({ $0 > 0 }), strides.allSatisfy({ $0 >= 0 }) else {
            throw CopyError.unsupported("CoreAI state requires positive dimensions and nonnegative strides")
        }
        var count = 1, largestOffset = 0
        for (dimension, stride) in zip(shape, strides) {
            let product = count.multipliedReportingOverflow(by: dimension)
            let span = (dimension - 1).multipliedReportingOverflow(by: stride)
            let sum = largestOffset.addingReportingOverflow(span.partialValue)
            guard !product.overflow, !span.overflow, !sum.overflow else {
                throw CopyError.unsupported("CoreAI state shape or stride overflows Int")
            }
            count = product.partialValue
            largestOffset = sum.partialValue
        }
        func transfer<T: BitwiseCopyable>(_ type: T.Type) throws -> NDArray {
            guard count <= Int.max / MemoryLayout<T>.stride,
                  largestOffset < Int.max / MemoryLayout<T>.stride else {
                throw CopyError.unsupported("CoreAI state byte extent overflows Int")
            }
            // The public initializer guarantees contiguous row-major strides.
            var destination = NDArray(shape: shape, scalarType: source.scalarType)
            source.view(as: T.self).withUnsafePointer { sourcePointer, _, _ in
                let destinationView = destination.mutableView(as: T.self)
                destinationView.withUnsafeMutablePointer { destinationPointer, _, _ in
                    for linear in 0..<count {
                        var remaining = linear, sourceOffset = 0
                        for dimension in shape.indices.reversed() {
                            sourceOffset += (remaining % shape[dimension]) * strides[dimension]
                            remaining /= shape[dimension]
                        }
                        destinationPointer[linear] = sourcePointer[sourceOffset]
                    }
                }
            }
            return destination
        }
        switch source.scalarType {
        case .float16: return try transfer(Float16.self)
        case .float32: return try transfer(Float.self)
        case .float64: return try transfer(Double.self)
        case .int32: return try transfer(Int32.self)
        default: throw CopyError.unsupported("Unsupported CoreAI state dtype: \(source.scalarType)")
        }
    }

    static func deepCopy(_ source: [String: NDArray]) throws -> [String: NDArray] {
        try source.mapValues { try deepCopy($0) }
    }
}
#endif
