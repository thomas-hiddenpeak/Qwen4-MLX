import Foundation
import XCTest
@testable import ANERunnerGPU

enum GPUFixtureLocation {
    static var package: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    }
    static func model() throws -> URL {
        let path = ProcessInfo.processInfo.environment["ANERUNNER_TEST_MODEL_DIR"]
        let url = path.map { URL(fileURLWithPath: $0) } ?? package.appendingPathComponent("../qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream").standardizedFileURL
        guard FileManager.default.fileExists(atPath: url.appendingPathComponent("config.json").path) else {
            throw XCTSkip("Local Qwen fixture unavailable; set ANERUNNER_TEST_MODEL_DIR")
        }
        return url
    }
    static func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("ANERunnerGPU-test-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}

final class QwenConfigurationTests: XCTestCase {
    func testRealArchitectureAndLayerNumbering() throws {
        let config = try QwenConfiguration(modelDirectory: GPUFixtureLocation.model())
        XCTAssertEqual(config.hiddenSize, 2560)
        XCTAssertEqual(config.layerCount, 48)
        XCTAssertEqual(config.layerTypes.enumerated().filter { $0.element == "full_attention" }.map(\.offset), Array(stride(from: 3, through: 47, by: 4)))
        XCTAssertEqual(config.pleLayerIDs, [2])
        XCTAssertEqual(config.pleLayerIndices, [1])
        XCTAssertEqual(config.expertsPerToken, 10)
        XCTAssertEqual(config.expertCount, 512)
        XCTAssertEqual(config.hcCount, 4)
        XCTAssertEqual(config.ngramScale, 0.00019931793212890625)
        XCTAssertEqual(config.quantizationBits, 4)
        XCTAssertEqual(config.quantizationGroupSize, 64)
    }

    func testRejectsInvalidArchitectureAndFractionalDimensions() throws {
        let model = try GPUFixtureLocation.model()
        let data = try Data(contentsOf: model.appendingPathComponent("config.json"))
        let original = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let temporary = try GPUFixtureLocation.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: temporary) }
        let badValues: [String: Any] = ["hidden_size": 2560.5, "num_experts_per_tok": 513, "hidden_act": "gelu", "layer_types": ["full_attention"], "num_key_value_heads": 0]
        for (key, value) in badValues {
            var root = original
            var text = try XCTUnwrap(root["text_config"] as? [String: Any])
            text[key] = value; root["text_config"] = text
            try JSONSerialization.data(withJSONObject: root).write(to: temporary.appendingPathComponent("config.json"))
            XCTAssertThrowsError(try QwenConfiguration(modelDirectory: temporary), "Should reject \(key)")
        }
    }
}
