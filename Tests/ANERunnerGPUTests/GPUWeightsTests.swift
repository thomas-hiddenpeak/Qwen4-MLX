import Foundation
import XCTest
@testable import ANERunnerGPU

final class GPUWeightsTests: XCTestCase {
    private func fixture(descriptors: [String: Any], payload: Data = Data(repeating: 0, count: 8)) throws -> URL {
        let directory = try GPUFixtureLocation.temporaryDirectory()
        var header = try JSONSerialization.data(withJSONObject: descriptors, options: [.sortedKeys])
        header.append(Data(repeating: 32, count: (8 - header.count % 8) % 8))
        var size = UInt64(header.count).littleEndian
        var data = withUnsafeBytes(of: &size) { Data($0) }
        data.append(header); data.append(payload)
        try data.write(to: directory.appendingPathComponent("model.safetensors"))
        let index = ["weight_map": ["weight": "model.safetensors"]]
        try JSONSerialization.data(withJSONObject: index).write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        return directory
    }

    func testPackedWeightMetadataAndTinyNativeRead() throws {
        // The exact unsigned bit patterns exercise values above Int32.max.
        let payload = Data([0x00, 0x00, 0x80, 0x3f, 0xff, 0xff, 0xff, 0xff])
        let directory = try fixture(descriptors: ["weight": ["dtype": "U32", "shape": [2], "data_offsets": [0, 8]]], payload: payload)
        defer { try? FileManager.default.removeItem(at: directory) }
        let weights = try GPUWeights(modelDirectory: directory)
        let info = try weights.metadata("weight")
        XCTAssertEqual(info.dtypeName, "U32"); XCTAssertEqual(info.shape, [2]); XCTAssertEqual(info.byteCount, 8)
        XCTAssertEqual(weights.cachedSourceBytes, 0)
        let tensor = try weights.tensor("weight")
        XCTAssertEqual(tensor.nbytes, 8)
        XCTAssertTrue(tensor === (try weights.tensor("weight")))
        XCTAssertEqual(try tensor.ints(), [0x3f800000, -1])
        XCTAssertEqual(weights.ledger.first?.loadCount, 1)
        weights.release(prefix: "weight")
        XCTAssertEqual(weights.cachedSourceBytes, 0)
        // Caller ownership survives loader eviction.
        XCTAssertEqual(try tensor.ints(), [0x3f800000, -1])
        XCTAssertThrowsError(try weights.tensor("missing"))
    }

    func testRejectsBoundsDtypesOverflowAndOverlapBeforeLoadingPayload() throws {
        let cases: [[String: Any]] = [
            ["weight": ["dtype": "U32", "shape": [2], "data_offsets": [0, 12]]],
            ["weight": ["dtype": "U32", "shape": [2], "data_offsets": [4, 12]]],
            ["weight": ["dtype": "U32", "shape": [2], "data_offsets": [8, 0]]],
            ["weight": ["dtype": "F8_NOT_SUPPORTED", "shape": [8], "data_offsets": [0, 8]]],
            ["weight": ["dtype": "U32", "shape": [2.5], "data_offsets": [0, 8]]],
            ["weight": ["dtype": "U32", "shape": [true], "data_offsets": [0, 8]]],
            ["weight": ["dtype": "U32", "shape": [Int32.max, Int32.max, Int32.max], "data_offsets": [0, 8]]],
            ["weight": ["dtype": "U32", "shape": [2], "data_offsets": [0, 8]], "overlap": ["dtype": "U32", "shape": [1], "data_offsets": [4, 8]]],
        ]
        for (index, descriptor) in cases.enumerated() {
            let directory = try fixture(descriptors: descriptor)
            defer { try? FileManager.default.removeItem(at: directory) }
            let weights = try GPUWeights(modelDirectory: directory)
            XCTAssertThrowsError(try weights.metadata("weight"), "Bad descriptor \(index)")
            XCTAssertEqual(weights.cachedSourceBytes, 0)
        }
    }

    func testRejectsIndexTraversalAndTruncatedHeader() throws {
        let directory = try GPUFixtureLocation.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        try JSONSerialization.data(withJSONObject: ["weight_map": ["weight": "../model.safetensors"]]).write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        XCTAssertThrowsError(try GPUWeights(modelDirectory: directory))
        try JSONSerialization.data(withJSONObject: ["weight_map": ["weight": "model.safetensors"]]).write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        try Data([1, 2, 3]).write(to: directory.appendingPathComponent("model.safetensors"))
        let weights = try GPUWeights(modelDirectory: directory)
        XCTAssertThrowsError(try weights.metadata("weight"))
    }
}
