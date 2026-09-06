import CMLX
import Foundation
import XCTest
@testable import ANERunnerGPU

/// Synthetic deterministic BF16 data at the checkpoint's real matrix shapes.
/// These are projection correctness gates, not real-prompt or throughput claims.
/// The parent validation process owns when GPU tests are run.
final class GPUDecodeProjectionTests: XCTestCase {
    private func values(_ shape: [Int], seed: UInt64) throws -> Tensor {
        let count = shape.reduce(1, *)
        var data = Data(count: count * MemoryLayout<UInt16>.stride)
        data.withUnsafeMutableBytes { raw in
            let words = raw.bindMemory(to: UInt16.self)
            var state = seed
            for index in 0..<count {
                state = state &* 6364136223846793005 &+ 1442695040888963407
                let sign = UInt16((state >> 63) & 1) << 15
                let exponent = UInt16(118 + ((state >> 56) & 3)) << 7
                let fraction = UInt16((state >> 32) & 127)
                words[index] = (sign | exponent | fraction).littleEndian
            }
        }
        return try MX.array(data: data,shape: shape,dtype: MLX_BFLOAT16)
    }

    private func assertBits(_ actual: Tensor, _ expected: Tensor, _ label: String,
                            file: StaticString = #filePath, line: UInt = #line) throws {
        XCTAssertEqual(actual.shape,expected.shape,label,file: file,line: line)
        XCTAssertEqual(actual.dtype,MLX_BFLOAT16,label,file: file,line: line)
        let lhs = try actual.floats(), rhs = try expected.floats()
        XCTAssertEqual(lhs.count,rhs.count,label,file: file,line: line)
        XCTAssertTrue(lhs.allSatisfy(\.isFinite),label,file: file,line: line)
        let first = zip(lhs,rhs).enumerated().first { $0.element.0.bitPattern != $0.element.1.bitPattern }
        if let first {
            XCTFail("\(label): first BF16 mismatch at \(first.offset): \(first.element.0) versus \(first.element.1)",file: file,line: line)
        }
    }

    func testGDNProjectionPairsMatchAllFourOriginalMatmulsBitwise() throws {
        let qkv = try MX.transpose(values([10240,2560],seed: 0x2026090501),[1,0])
        let z = try MX.transpose(values([6144,2560],seed: 0x2026090502),[1,0])
        let a = try MX.transpose(values([48,2560],seed: 0x2026090503),[1,0])
        let b = try MX.transpose(values([48,2560],seed: 0x2026090504),[1,0])
        let packed = try GDNDecodeProjectionBuffers(qkv: qkv,z: z,a: a,b: b)
        XCTAssertEqual(packed.byteCount,84_377_600)
        for seed in [UInt64(7),19,271] {
            let x = try values([1,1,2560],seed: seed)
            let actual = try packed.project(x)
            try assertBits(actual.qkv,MX.matmul(x,qkv),"QKV seed \(seed)")
            try assertBits(actual.z,MX.matmul(x,z),"Z seed \(seed)")
            try assertBits(actual.a,MX.matmul(x,a),"A seed \(seed)")
            try assertBits(actual.b,MX.matmul(x,b),"B seed \(seed)")
        }
        XCTAssertThrowsError(try packed.project(MX.zeros([1,2,2560],MLX_BFLOAT16)))
        XCTAssertThrowsError(try packed.project(MX.zeros([2,1,2560],MLX_BFLOAT16)))
        XCTAssertThrowsError(try packed.project(MX.zeros([1,1,2560],MLX_FLOAT32)))
    }

    func testHCDownAndInjectionMatchOriginalScaledWeightsBitwise() throws {
        let quarter = try MX.scalar(0.25,MLX_BFLOAT16)
        // Preserve the module's original BF16 weight scaling before packing.
        let down = try MX.transpose(MX.mul(values([320,10240],seed: 0x2026090511),quarter),[1,0])
        let injection = try MX.transpose(MX.mul(values([4,10240],seed: 0x2026090512),quarter),[1,0])
        let packed = try XCTUnwrap(HCDecodeProjectionBuffers(down: down,injection: injection))
        XCTAssertEqual(packed.byteCount,6_635_520)
        for seed in [UInt64(42),88,391] {
            let normalized = try values([1,1,10240],seed: seed)
            let actual = try packed.project(normalized)
            try assertBits(actual.down,MX.matmul(normalized,down),"HC down seed \(seed)")
            try assertBits(actual.injection,MX.matmul(normalized,injection),"HC injection seed \(seed)")
        }
        XCTAssertThrowsError(try packed.project(MX.zeros([1,2,10240],MLX_BFLOAT16)))
        XCTAssertThrowsError(try packed.project(MX.zeros([1,1,10240],MLX_FLOAT32)))
    }

    func testRuntimeSelectionRequiresPreparedBuffersAndAllowsReferenceAB() throws {
        XCTAssertFalse(try DecodeProjectionSelection.use(requested: nil,prepared: false))
        XCTAssertTrue(try DecodeProjectionSelection.use(requested: nil,prepared: true))
        XCTAssertFalse(try DecodeProjectionSelection.use(requested: false,prepared: true))
        XCTAssertFalse(try DecodeProjectionSelection.use(requested: false,prepared: false))
        XCTAssertTrue(try DecodeProjectionSelection.use(requested: true,prepared: true))
        XCTAssertThrowsError(try DecodeProjectionSelection.use(requested: true,prepared: false))
    }

    func testMixerDoesNotPrepareAnyExtraProjectionBuffer() throws {
        XCTAssertNil(try HCDecodeProjectionBuffers(down: MX.zeros([10240,320],MLX_BFLOAT16),injection: nil))
    }

    func testPackingRejectsNonModelShapesAndDtypes() throws {
        let wrong = try MX.zeros([1,1],MLX_BFLOAT16)
        XCTAssertThrowsError(try GDNDecodeProjectionBuffers(qkv: wrong,z: wrong,a: wrong,b: wrong))
        XCTAssertThrowsError(try HCDecodeProjectionBuffers(down: wrong,injection: wrong))
    }
}
