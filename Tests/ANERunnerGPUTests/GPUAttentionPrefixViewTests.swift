import CMLX
import XCTest
@testable import ANERunnerGPU

final class GPUAttentionPrefixViewTests: XCTestCase {
    /// Tiny real BF16/GQA SDPA, sentinel rejected rows, and repeated trimming.
    /// This uses no weights, full attention module, tokenizer or model load.
    func testBoundedViewsMatchCopiesAndExcludeRejectedRows() throws {
        let length = 9, keep = 6, width = 256
        var keyData = [Float](), valueData = [Float]()
        for head in 0..<2 {
            for row in 0..<length {
                let key: Float = row < keep ? Float(row - 4) / 32 + Float(head) / 16 : 8
                let value: Float = row < keep ? Float(head * 32 + row) : 2048
                keyData += [Float](repeating: key, count: width)
                valueData += [Float](repeating: value, count: width)
            }
        }
        let keys = try MX.array(keyData, shape: [1,2,length,width], dtype: MLX_BFLOAT16)
        let values = try MX.array(valueData, shape: [1,2,length,width], dtype: MLX_BFLOAT16)
        let query = try MX.array([Float](repeating: 1 / 32, count: 24 * width),
                                 shape: [1,24,1,width], dtype: MLX_BFLOAT16)
        let keyPrefix = try GPUAttention.retainedPrefix(keys, axis: 2, count: keep, extent: length, maximumTail: 4)
        let valuePrefix = try GPUAttention.retainedPrefix(values, axis: 2, count: keep, extent: length, maximumTail: 4)
        let copiedKeys = try GPUVerificationCopy.tensor(keyPrefix.tensor)
        let copiedValues = try GPUVerificationCopy.tensor(valuePrefix.tensor)
        try MX.eval([keys, values, keyPrefix.tensor, valuePrefix.tensor, copiedKeys, copiedValues])
        XCTAssertEqual(keyPrefix.extent, length)
        XCTAssertEqual(valuePrefix.extent, length)
        XCTAssertEqual(Int(try XCTUnwrap(mlx_array_strides(keyPrefix.tensor.handle))[1]), length * width)
        XCTAssertEqual(Int(try XCTUnwrap(mlx_array_strides(copiedKeys.handle))[1]), keep * width)

        let viewed = try MX.sdpa(query, keyPrefix.tensor, valuePrefix.tensor, scale: 1 / 16)
        let copied = try MX.sdpa(query, copiedKeys, copiedValues, scale: 1 / 16)
        let untrimmed = try MX.sdpa(query, keys, values, scale: 1 / 16)
        try MX.eval([viewed, copied, untrimmed])
        let viewBits = try viewed.floats().map(\.bitPattern)
        let copyBits = try copied.floats().map(\.bitPattern)
        XCTAssertEqual(zip(viewBits, copyBits).filter { $0.0 != $0.1 }.count, 0,
                       "Noncontiguous KV prefix must produce identical SDPA values")
        let futureBits = try untrimmed.floats().map(\.bitPattern)
        XCTAssertTrue(zip(viewBits, futureBits).contains { $0.0 != $0.1 },
                      "Rejected future rows must actually affect the control fixture")

        let newValues = try MX.array([Float](repeating: 256, count: 2 * width),
                                     shape: [1,2,1,width], dtype: MLX_BFLOAT16)
        let concatenated = try MX.concat([valuePrefix.tensor, newValues], axis: 2)
        var expected = [Float]()
        for head in 0..<2 {
            for row in 0..<keep { expected += [Float](repeating: Float(head * 32 + row), count: width) }
            expected += [Float](repeating: 256, count: width)
        }
        XCTAssertEqual(concatenated.shape, [1,2,keep+1,width])
        XCTAssertEqual(try concatenated.floats(), expected,
                       "Concat must append after logical rows, not the retained future suffix")
        XCTAssertEqual(Int(try XCTUnwrap(mlx_array_strides(concatenated.handle))[1]), (keep + 1) * width)

        // Each logical cut removes one row, but the third view would retain
        // five rows from the original buffer; its extent must reset on copy.
        let second = try GPUAttention.retainedPrefix(valuePrefix.tensor, axis: 2, count: 5,
                                                     extent: valuePrefix.extent, maximumTail: 4)
        let third = try GPUAttention.retainedPrefix(second.tensor, axis: 2, count: 4,
                                                    extent: second.extent, maximumTail: 4)
        try MX.eval([second.tensor, third.tensor])
        XCTAssertEqual(second.extent, 9)
        XCTAssertEqual(third.extent, 4)
        XCTAssertEqual(Int(try XCTUnwrap(mlx_array_strides(third.tensor.handle))[1]), 4 * width)
        XCTAssertEqual(try third.tensor.floats(), (0..<2).flatMap { head in
            (0..<4).flatMap { row in [Float](repeating: Float(head * 32 + row), count: width) }
        })

        // Pooled storage can survive a forward with no new complete block.
        // Its independently carried block extent must also bound repeated cuts.
        let pooled = try MX.zeros([1,4,128], MLX_BFLOAT16)
        let pooledFirst = try GPUAttention.retainedPrefix(pooled, axis: 1, count: 3, extent: 4, maximumTail: 1)
        let pooledSecond = try GPUAttention.retainedPrefix(pooledFirst.tensor, axis: 1, count: 2,
                                                         extent: pooledFirst.extent, maximumTail: 1)
        XCTAssertEqual(pooledFirst.extent, 4)
        XCTAssertEqual(pooledSecond.extent, 2)
        XCTAssertThrowsError(try GPUAttention.retainedPrefix(values, axis: 2, count: 6, extent: 5, maximumTail: 4))
    }
}
