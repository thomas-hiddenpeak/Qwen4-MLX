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

    /// Allocates independent, contiguous storage and clears its logical bytes.
    /// IEEE floating-point positive zero and Int32 zero all have zero bit patterns.
    /// Never apply this flat fill to arbitrary, padded or interleaved model views.
    static func zeroArray(shape: [Int], scalarType: NDArray.ScalarType) throws -> NDArray {
        guard !shape.isEmpty, shape.allSatisfy({ $0 > 0 }) else {
            throw CopyError.unsupported("CoreAI zero state requires positive dimensions")
        }
        let elementBytes: Int
        switch scalarType {
        case .float16: elementBytes = MemoryLayout<Float16>.stride
        case .float32: elementBytes = MemoryLayout<Float>.stride
        case .float64: elementBytes = MemoryLayout<Double>.stride
        case .int32: elementBytes = MemoryLayout<Int32>.stride
        default: throw CopyError.unsupported("Unsupported CoreAI zero-state dtype: \(scalarType)")
        }
        var count = 1
        for dimension in shape {
            let product = count.multipliedReportingOverflow(by: dimension)
            guard !product.overflow else { throw CopyError.unsupported("CoreAI zero-state shape overflows Int") }
            count = product.partialValue
        }
        let extent = count.multipliedReportingOverflow(by: elementBytes)
        guard !extent.overflow else { throw CopyError.unsupported("CoreAI zero-state byte count overflows Int") }
        // This initializer creates contiguous row-major storage. Check the layout
        // before taking a flat pointer rather than relying on model strides.
        var array = NDArray(shape: shape, scalarType: scalarType)
        let strides = array.strides
        guard array.interleaveLayout == nil, strides.count == shape.count else {
            throw CopyError.unsupported("CoreAI zero allocation has an unsupported layout")
        }
        var expectedStride = 1
        for dimension in shape.indices.reversed() {
            guard shape[dimension] == 1 || strides[dimension] == expectedStride else {
                throw CopyError.unsupported("CoreAI zero allocation is not contiguous")
            }
            expectedStride *= shape[dimension]
        }
        func clear<T: BitwiseCopyable>(_ zero: T) {
            let view = array.mutableView(as: T.self)
            view.withUnsafeMutablePointer { pointer, _, _ in
                pointer.update(repeating: zero, count: count)
            }
        }
        switch scalarType {
        case .float16: clear(Float16.zero)
        case .float32: clear(Float.zero)
        case .float64: clear(Double.zero)
        case .int32: clear(Int32.zero)
        default: throw CopyError.unsupported("Unsupported CoreAI zero-state dtype: \(scalarType)")
        }
        return array
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
        var contiguous = true, expectedStride = 1
        for dimension in shape.indices.reversed() {
            if shape[dimension] > 1 && strides[dimension] != expectedStride { contiguous = false }
            expectedStride *= shape[dimension]
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
                    if contiguous {
                        destinationPointer.update(from: sourcePointer, count: count)
                        return
                    }
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

    /// Logical tensor storage only; excludes allocator padding and object overhead.
    /// Reads tensor metadata without materializing any values or allocating a copy.
    static func logicalByteCount(_ source: NDArray) throws -> Int {
        guard source.interleaveLayout == nil, !source.shape.isEmpty,
              source.shape.allSatisfy({ $0 > 0 }) else {
            throw CopyError.unsupported("Cannot size an empty or interleaved CoreAI state")
        }
        let elementBytes: Int
        switch source.scalarType {
        case .float16: elementBytes = MemoryLayout<Float16>.stride
        case .float32: elementBytes = MemoryLayout<Float>.stride
        case .float64: elementBytes = MemoryLayout<Double>.stride
        case .int32: elementBytes = MemoryLayout<Int32>.stride
        default: throw CopyError.unsupported("Unsupported CoreAI state dtype: \(source.scalarType)")
        }
        var bytes = elementBytes
        for dimension in source.shape {
            let product = bytes.multipliedReportingOverflow(by: dimension)
            guard !product.overflow else { throw CopyError.unsupported("CoreAI state logical size overflows Int") }
            bytes = product.partialValue
        }
        return bytes
    }

    static func logicalByteCount(_ source: [String: NDArray]) throws -> Int {
        var bytes = 0
        for array in source.values {
            bytes = try addingByteCounts(bytes, logicalByteCount(array))
        }
        return bytes
    }

    static func addingByteCounts(_ first: Int, _ second: Int) throws -> Int {
        let total = first.addingReportingOverflow(second)
        guard first >= 0, second >= 0, !total.overflow else {
            throw CopyError.unsupported("CoreAI state logical size overflows Int")
        }
        return total.partialValue
    }
}
#endif
