import Foundation
import XCTest
@testable import ANERunnerCore

final class QwenPrefixStateArchiveDescriptorTests: XCTestCase {
    private typealias Descriptor = QwenPrefixStateArchiveDescriptor
    private let layout = Descriptor.Layout(
        layerTypes: (0..<48).map { $0 % 4 == 3 ? "full_attention" : "linear_attention" },
        pleLayerIndices: [1], hiddenSize: 2560, hcCount: 4, ngramSize: 3,
        pleConvKernel: 4, vocabularySize: 248_320, maximumPositions: 262_144)

    private func valid(_ offset: Int = 9984) throws -> Descriptor {
        let tensors = try Descriptor.expectedTensors(layout: layout, offset: offset)
        let last = try XCTUnwrap(tensors.last)
        var history = [[UInt32]](repeating: [], count: 48); history[1] = [17, 42]
        return Descriptor(layout: layout, offset: offset,
            gdnOffsets: layout.layerTypes.map { $0 == "linear_attention" ? offset : 0 },
            attentionOffsets: layout.layerTypes.map { $0 == "full_attention" ? offset : 0 },
            pleHistory: history, tensors: tensors,
            tensorPayloadBytes: last.byteOffset + last.byteCount,
            logicalPayloadBytes: last.byteOffset + last.byteCount + 8)
    }

    private func validate(_ d: Descriptor, expectedOffset: Int? = nil,
                          bytes: Int? = nil, cap: Int = Descriptor.defaultMaximumPayloadBytes) throws {
        try d.validate(expectedLayout: layout, expectedOffset: expectedOffset ?? d.offset,
            actualPayloadBytes: bytes ?? d.tensorPayloadBytes, maxPayloadBytes: cap)
    }

    func testCanonicalFullStateRoundTripsWithoutDeviceWork() throws {
        let d = try valid(), metadata = try d.encoded()
        let decoded = try Descriptor.decodeAndValidate(metadata, expectedLayout: layout,
            expectedOffset: d.offset, actualPayloadBytes: d.tensorPayloadBytes)
        XCTAssertEqual(decoded, d)
        XCTAssertLessThan(metadata.count, Descriptor.maximumMetadataBytes)
        XCTAssertEqual(d.tensors.count, 121)
        XCTAssertEqual(d.tensors.filter { $0.name.hasSuffix("gdn.recurrent") }.count, 36)
        XCTAssertEqual(d.tensors.filter { $0.name.hasSuffix("attention.keys") }.count, 12)
        XCTAssertEqual(d.logicalPayloadBytes,
            try Descriptor.estimatedLogicalPayloadBytes(layout: layout, offset: d.offset))
    }

    func testExactQSAActivationAndPartialPoolBoundaries() throws {
        for offset in [1, 416, 1664, 2051, 2052, 2053, 2055, 2056, 9984] {
            let d = try valid(offset); try validate(d)
            let pools = d.tensors.filter { $0.name.hasSuffix("pooled_index") }
            XCTAssertEqual(pools.count, offset > 2051 ? 12 : 0)
            XCTAssertTrue(pools.allSatisfy { $0.shape == [1, offset / 4, 128] })
            XCTAssertEqual(d.tensors.count, offset > 2051 ? 121 : 109)
        }
    }

    func testOffsetVersionLayoutAndLimitsFailClosed() throws {
        let d = try valid()
        XCTAssertThrowsError(try validate(d, expectedOffset: d.offset + 1))
        for version in [-1, 0, 2, Int.max] {
            var bad = d; bad.version = version
            XCTAssertThrowsError(try validate(bad))
        }
        for cap in [-1, 0, d.logicalPayloadBytes - 1, Descriptor.absoluteMaximumPayloadBytes + 1, Int.max] {
            XCTAssertThrowsError(try validate(d, cap: cap))
        }
        try validate(d, cap: d.logicalPayloadBytes)
        for offset in [-1, 0, 262_145, Int.max] {
            XCTAssertThrowsError(try Descriptor.expectedTensors(layout: layout, offset: offset))
        }
        var bad = d
        bad.layout = .init(layerTypes: layout.layerTypes, pleLayerIndices: [1], hiddenSize: 2560,
            hcCount: 4, ngramSize: 3, pleConvKernel: 4, vocabularySize: 100, maximumPositions: 262_144)
        XCTAssertThrowsError(try validate(bad))
    }

    func testWrongTensorSetsAndOrderFailBeforeShapeAllocation() throws {
        let d = try valid()
        var bad = d; bad.tensors.removeLast()
        XCTAssertThrowsError(try validate(bad))
        bad = d; bad.tensors.append(d.tensors[0])
        XCTAssertThrowsError(try validate(bad))
        bad = d; bad.tensors[1] = d.tensors[0]
        XCTAssertThrowsError(try validate(bad))
        bad = d; bad.tensors.swapAt(0, 1)
        XCTAssertThrowsError(try validate(bad))
        let first = d.tensors[0]
        let variants: [Descriptor.TensorDescriptor] = [
            .init(name: "unknown", shape: first.shape, byteOffset: first.byteOffset, byteCount: first.byteCount),
            .init(name: first.name, shape: [Int.max, Int.max], byteOffset: 0, byteCount: first.byteCount),
            .init(name: first.name, shape: [-1, 48, 128, 128], byteOffset: 0, byteCount: first.byteCount),
            .init(name: first.name, shape: [1, 48, 128, 0], byteOffset: 0, byteCount: first.byteCount),
            .init(name: first.name, shape: first.shape, dtype: "float32-le", byteOffset: 0, byteCount: first.byteCount),
            .init(name: first.name, shape: first.shape, byteOffset: -1, byteCount: first.byteCount),
            .init(name: first.name, shape: first.shape, byteOffset: 1, byteCount: first.byteCount),
            .init(name: first.name, shape: first.shape, byteOffset: 0, byteCount: Int.max),
            .init(name: first.name, shape: first.shape, byteOffset: 0, byteCount: -1),
        ]
        for tensor in variants {
            bad = d; bad.tensors[0] = tensor
            XCTAssertThrowsError(try validate(bad))
        }
    }

    func testPayloadTruncationGapsOverlapAndTrailingBytesFail() throws {
        let d = try valid()
        for bytes in [-1, 0, d.tensorPayloadBytes - 1, d.tensorPayloadBytes + 1, Int.max] {
            XCTAssertThrowsError(try validate(d, bytes: bytes))
        }
        for adjustment in [-1, 1] {
            var bad = d
            let tensor = bad.tensors[1]
            bad.tensors[1] = .init(name: tensor.name, shape: tensor.shape,
                byteOffset: tensor.byteOffset + adjustment, byteCount: tensor.byteCount)
            XCTAssertThrowsError(try validate(bad))
        }
        var bad = d; bad.tensorPayloadBytes += 1
        XCTAssertThrowsError(try validate(bad))
        bad = d; bad.logicalPayloadBytes = d.tensorPayloadBytes
        XCTAssertThrowsError(try validate(bad))
    }

    func testMissingOrUnexpectedQSAStateFails() throws {
        var beyond = try valid(2052)
        beyond.tensors.removeAll { $0.name.hasSuffix("pooled_index") }
        XCTAssertThrowsError(try validate(beyond))
        var before = try valid(2051)
        before.tensors.append(.init(name: "layer.3.attention.pooled_index", shape: [1, 512, 128],
            byteOffset: before.tensorPayloadBytes, byteCount: 131_072))
        XCTAssertThrowsError(try validate(before))
        beyond = try valid(2055)
        let index = try XCTUnwrap(beyond.tensors.firstIndex { $0.name.hasSuffix("pooled_index") })
        let tensor = beyond.tensors[index]
        beyond.tensors[index] = .init(name: tensor.name, shape: [1, 512, 128],
            byteOffset: tensor.byteOffset, byteCount: tensor.byteCount)
        XCTAssertThrowsError(try validate(beyond))
    }

    func testLayerOffsetsAndPLEHistoryAreCompleteAndInRange() throws {
        let d = try valid()
        var bad = d; bad.gdnOffsets.removeLast()
        XCTAssertThrowsError(try validate(bad))
        bad = d; bad.gdnOffsets[0] -= 1
        XCTAssertThrowsError(try validate(bad))
        bad = d; bad.gdnOffsets[3] = d.offset
        XCTAssertThrowsError(try validate(bad))
        bad = d; bad.attentionOffsets[3] -= 1
        XCTAssertThrowsError(try validate(bad))
        bad = d; bad.attentionOffsets[0] = d.offset
        XCTAssertThrowsError(try validate(bad))
        bad = d; bad.pleHistory.removeLast()
        XCTAssertThrowsError(try validate(bad))
        let invalidHistories: [[UInt32]] = [[], [1], [1, 2, 3], [248_320, 1], [UInt32.max, 1]]
        for history in invalidHistories {
            bad = d; bad.pleHistory[1] = history
            XCTAssertThrowsError(try validate(bad))
        }
        bad = d; bad.pleHistory[0] = [1, 2]
        XCTAssertThrowsError(try validate(bad))
    }

    func testMalformedAndOversizedMetadataFailBeforeJSONDecode() throws {
        let d = try valid()
        for metadata in [Data(), Data("not json".utf8),
                         Data(repeating: 32, count: Descriptor.maximumMetadataBytes + 1)] {
            XCTAssertThrowsError(try Descriptor.decodeAndValidate(metadata, expectedLayout: layout,
                expectedOffset: d.offset, actualPayloadBytes: d.tensorPayloadBytes))
        }
        var huge = d
        huge.tensors = Array(repeating: d.tensors[0], count: 1024)
        XCTAssertThrowsError(try huge.encoded())
    }

    func testFullContextEstimateDoesNotAllocateOrOverflowButArchiveCapRejectsIt() throws {
        let d = try valid(262_144)
        XCTAssertGreaterThan(d.logicalPayloadBytes, Descriptor.absoluteMaximumPayloadBytes)
        XCTAssertEqual(d.logicalPayloadBytes,
            try Descriptor.estimatedLogicalPayloadBytes(layout: layout, offset: 262_144))
        XCTAssertThrowsError(try validate(d, cap: Descriptor.absoluteMaximumPayloadBytes))
    }
}
