import Foundation
import XCTest
@testable import ANERunnerCore

final class MoEManifestTests: XCTestCase {
    func testManifestRejectsUnsupportedGeometryAndPrecision() throws {
        let invalid: [(String, Any)] = [
            ("schema_version", 2), ("hidden_size", 0), ("expert_count", 0),
            ("top_k", 5), ("token_capacity", 0), ("token_capacity", 4097),
            ("dtype", "float32"), ("experts", ["4": "expert.mlpackage"])
        ]
        for (key, value) in invalid {
            try withExport { directory, manifest in
                var object = manifest; object[key] = value
                let url = try writeManifest(object, to: directory)
                XCTAssertThrowsError(try MoEManifest.load(from: url), key)
            }
        }
    }

    func testRelativeAssetsCannotEscapeThroughParentOrSymlink() throws {
        try withExport { directory, object in
            let manifest = try MoEManifest.load(from: writeManifest(object, to: directory))
            XCTAssertThrowsError(try manifest.resolve("../outside.bin", relativeTo: directory))
            XCTAssertThrowsError(try manifest.resolve("/tmp/outside.bin", relativeTo: directory))
            let link = directory.appendingPathComponent("escape")
            try FileManager.default.createSymbolicLink(at: link, withDestinationURL: directory.deletingLastPathComponent())
            XCTAssertThrowsError(try manifest.resolve("escape", relativeTo: directory))
        }
    }

    func testMissingDynamicallySelectedExpertFailsBeforeCoreMLCompilation() throws {
        try withExport { directory, manifest in
            let runner = try MoEBlockRunner(manifestURL: writeManifest(manifest, to: directory), precision: .float32)
            let fixture = CoreMLBlockFixture(inputs: ["x": CoreMLTensor(shape: [1,1,2], dtype: .float32, values: [1,0])])
            let routed = try runner.route(fixture: fixture)
            XCTAssertEqual(routed.expertIDs, [0,3])
            // The manifest advertises expert 0 with a nonexistent package.
            // Missing expert 3 must be rejected before touching that package.
            XCTAssertThrowsError(try runner.run(fixture: fixture, warmups: 0, runs: 1)) { error in
                XCTAssertTrue(error.localizedDescription.contains("missing experts: 3"), error.localizedDescription)
                XCTAssertTrue(error.localizedDescription.contains("no assignments were dropped"))
            }
        }
    }

    func testRouterFileLengthAndFiniteValuesAreCheckedBeforeModelsLoad() throws {
        try withExport { directory, manifest in
            let path = try writeManifest(manifest, to: directory)
            let router = directory.appendingPathComponent("router.f32")
            try Data([0,0,0]).write(to: router)
            XCTAssertThrowsError(try MoEBlockRunner(manifestURL: path))
            try writeFloats([Float.nan,0,0,0,0,0,0,0], to: router)
            XCTAssertThrowsError(try MoEBlockRunner(manifestURL: path))
        }
    }

    func testMalformedExpectedRoutingFailsBeforeExpertLoading() throws {
        try withExport { directory, manifest in
            let runner = try MoEBlockRunner(manifestURL: writeManifest(manifest, to: directory))
            let x = CoreMLTensor(shape: [1,1,2], dtype: .float32, values: [1,0])
            let ids = CoreMLTensor(shape: [1,1,2], dtype: .int32, values: [0,3])
            let weights = CoreMLTensor(shape: [1,1,2], dtype: .float32, values: [0.9,0.1])
            for expected in [
                ["routing_weights": weights], ["selected_experts": ids],
                ["selected_experts": ids, "routing_weights": CoreMLTensor(shape: [1,1,2], dtype: .float32, values: [0.9])]
            ] {
                XCTAssertThrowsError(try runner.run(fixture: CoreMLBlockFixture(inputs: ["x": x], expectedOutputs: expected))) { error in
                    XCTAssertFalse(error.localizedDescription.contains("missing experts:"), error.localizedDescription)
                }
            }
        }
    }

    func testBFloat16RoundingCannotOverflowExpertFP16Input() throws {
        try withExport { directory, manifest in
            let runner = try MoEBlockRunner(manifestURL: writeManifest(manifest, to: directory))
            let x = CoreMLTensor(shape: [1,1,2], dtype: .float32, values: [65504,0])
            XCTAssertThrowsError(try runner.route(fixture: CoreMLBlockFixture(inputs: ["x": x]))) { error in
                XCTAssertTrue(error.localizedDescription.contains("after requested BF16 rounding"))
            }
        }
    }

    private func withExport(_ body: (URL, [String: Any]) throws -> Void) throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("moe-manifest-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try writeFloats([4,0, 0,3, -1,0, 1,0], to: directory.appendingPathComponent("router.f32"))
        try writeFloats([0,0], to: directory.appendingPathComponent("shared.f32"))
        try body(directory, [
            "schema_version": 1, "layer_index": 0, "hidden_size": 2,
            "expert_count": 4, "top_k": 2, "token_capacity": 2,
            "input_name": "x", "output_name": "y", "dtype": "float16",
            "routing": ["weights_file": "router.f32", "shared_gate_file": "shared.f32", "dtype": "float32_le"],
            "experts": ["0": "missing-expert0.mlpackage"],
            "shared_expert": "missing-shared.mlpackage", "weight_mode": "synthetic manifest validation only"
        ])
    }

    private func writeManifest(_ object: [String: Any], to directory: URL) throws -> URL {
        let path = directory.appendingPathComponent("manifest.json")
        try JSONSerialization.data(withJSONObject: object).write(to: path)
        return path
    }

    private func writeFloats(_ values: [Float], to url: URL) throws {
        var bytes = Data()
        for value in values {
            var word = value.bitPattern.littleEndian
            withUnsafeBytes(of: &word) { bytes.append(contentsOf: $0) }
        }
        try bytes.write(to: url)
    }
}
