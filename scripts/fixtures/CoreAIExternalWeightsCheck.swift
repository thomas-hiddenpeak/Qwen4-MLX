// Standalone host-access checks against the production weight loader.
// Compile from experiments/ane-runner (macOS 27 SDK):
// swiftc -parse-as-library Sources/ANERunnerCore/CoreAIExternalWeights.swift \
//   scripts/fixtures/CoreAIExternalWeightsCheck.swift -o /tmp/coreai-weights-check
// Run: /tmp/coreai-weights-check [report.json]
// Creates shared Metal buffers, but never a command queue, model or GPU command.
import CoreAI
import CryptoKit
import Foundation
import Metal

// The loader's error type normally lives in CoreAIBlockRunner.swift. Compiling
// this fixture with only the loader avoids loading a model or its dependencies.
enum CoreAIBlockRunnerError: Error, CustomStringConvertible {
    case invalidModel(String)
    var description: String {
        switch self { case .invalidModel(let message): return message }
    }
}

private struct CheckFailure: Error, CustomStringConvertible {
    let description: String
}

private func require(_ condition: @autoclosure () -> Bool, _ message: String) throws {
    if !condition() { throw CheckFailure(description: message) }
}

private func digest(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

private func bytes<T>(_ values: [T]) -> Data {
    values.withUnsafeBufferPointer { Data(buffer: $0) }
}

private struct Fixture {
    let base: URL
    let file: URL
    let data: Data
    let slices: [String: Data]
    let document: [String: Any]

    init(in base: URL) throws {
        self.base = base
        file = base.appendingPathComponent("weights.bin")
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        // Four aligned slices exercise offset handling as well as exact integer
        // and floating-point bit patterns, including negative zero and extrema.
        let entries: [(String, String, [Int], Data)] = [
            ("w_half", "float16", [2, 3], bytes([Float16(-0.0), 0.5, -2, 3.25, 65504, -0.125])),
            ("w_float", "float32", [2, 2], bytes([Float(-0.0), 1.25, -3.5, 65537])),
            ("w_short", "int16", [1, 5], bytes([Int16.min, -1, 0, 257, Int16.max])),
            ("w_int", "int32", [2, 3], bytes([Int32.min, -1, 0, 65537, 16_777_217, Int32.max]))
        ]
        let alignment = 16_384
        var data = Data(repeating: 0xA5, count: alignment * entries.count)
        var records: [[String: Any]] = []
        var slices: [String: Data] = [:]
        for (index, entry) in entries.enumerated() {
            let offset = index * alignment
            data.replaceSubrange(offset..<(offset + entry.3.count), with: entry.3)
            slices[entry.0] = entry.3
            records.append(["inputName": entry.0, "bufferName": "module.\(entry.0)",
                            "dtype": entry.1, "shape": entry.2, "byteOffset": offset,
                            "byteLength": entry.3.count, "sha256": digest(entry.3)])
        }
        self.data = data
        self.slices = slices
        document = ["path": "weights.bin", "alignment": alignment, "byteLength": data.count,
                    "sha256": digest(data), "byteOrder": "little", "buffers": records]
        try data.write(to: file)
    }

    func spec(_ change: (inout [String: Any]) -> Void = { _ in }) throws -> CoreAIExternalWeightsSpec {
        var modified = document
        change(&modified)
        return try JSONDecoder().decode(CoreAIExternalWeightsSpec.self,
            from: JSONSerialization.data(withJSONObject: modified, options: [.sortedKeys]))
    }
}

private func editSlice(_ document: inout [String: Any], _ index: Int = 0,
                       _ change: (inout [String: Any]) -> Void) {
    var entries = document["buffers"] as! [[String: Any]]
    change(&entries[index])
    document["buffers"] = entries
}

@available(macOS 27.0, *)
private final class WeakOwner {
    weak var value: CoreAIExternalWeights?
}

@available(macOS 27.0, *)
@main private struct ExternalWeightsCheck {
    private static func checkViews(_ values: [String: InferenceFunction.AsyncValue],
                                   fixture: Fixture) async throws {
        let spec = try fixture.spec()
        try require(Set(values.keys) == Set(fixture.slices.keys), "weight input names changed")
        for entry in spec.buffers {
            guard let value = values[entry.inputName], let array = try await value.ndArray else {
                throw CheckFailure(description: "missing NDArray for \(entry.inputName)")
            }
            try require(array.shape == entry.shape, "shape changed for \(entry.inputName)")
            let actual: Data
            switch entry.dtype {
            case "float16":
                try require(array.scalarType == .float16, "float16 dtype changed")
                actual = array.view(as: Float16.self).withUnsafePointer { pointer, _, _ in
                    Data(bytes: pointer, count: entry.byteLength)
                }
            case "float32":
                try require(array.scalarType == .float32, "float32 dtype changed")
                actual = array.view(as: Float.self).withUnsafePointer { pointer, _, _ in
                    Data(bytes: pointer, count: entry.byteLength)
                }
            case "int16":
                try require(array.scalarType == .int16, "int16 dtype changed")
                actual = array.view(as: Int16.self).withUnsafePointer { pointer, _, _ in
                    Data(bytes: pointer, count: entry.byteLength)
                }
            case "int32":
                try require(array.scalarType == .int32, "int32 dtype changed")
                actual = array.view(as: Int32.self).withUnsafePointer { pointer, _, _ in
                    Data(bytes: pointer, count: entry.byteLength)
                }
            default: throw CheckFailure(description: "unexpected fixture dtype")
            }
            try require(actual == fixture.slices[entry.inputName], "bytes changed for \(entry.inputName)")
        }
    }

    // A separate scope prevents the optimizer extending a caller's strong
    // reference across the lifetime check. AsyncValues must retain their views.
    @inline(never)
    private static func detachedViews(fixture: Fixture, device: any MTLDevice,
                                      storage: CoreAIExternalWeightStorage,
                                      witness: WeakOwner) throws -> [String: InferenceFunction.AsyncValue] {
        let owner = try CoreAIExternalWeights(spec: fixture.spec(), baseURL: fixture.base,
            device: device, storage: storage, verifyIntegrity: true)
        witness.value = owner
        return owner.values
    }

    static func main() async throws {
        guard CommandLine.arguments.count <= 2 else {
            throw CheckFailure(description: "usage: coreai-weights-check [report.json]")
        }
        // Device creation allocates host-visible storage only. No inference or
        // command submission is needed for these file/view boundary checks.
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw CheckFailure(description: "Metal device unavailable; checks did not run")
        }
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("coreai-weights-check-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let fixture = try Fixture(in: root.appendingPathComponent("base"))
        var results: [[String: Any]] = []

        func record(_ name: String, _ body: () throws -> Void) {
            do { try body(); results.append(["name": name, "passed": true]) }
            catch { results.append(["name": name, "passed": false, "error": String(describing: error)]) }
        }

        func rejects(_ name: String, _ message: String, fixture: Fixture,
                     integrity: Bool = false, storage: CoreAIExternalWeightStorage = .resident,
                     _ change: (inout [String: Any]) -> Void) {
            record(name) {
                do {
                    _ = try CoreAIExternalWeights(spec: fixture.spec(change), baseURL: fixture.base,
                        device: device, storage: storage, verifyIntegrity: integrity)
                    throw CheckFailure(description: "invalid input was accepted")
                } catch CoreAIBlockRunnerError.invalidModel(let actual) {
                    try require(actual.contains(message), "unexpected rejection: \(actual)")
                }
            }
        }

        for storage in [CoreAIExternalWeightStorage.resident, .residentFile, .mapped] {
            let name = "\(storage.rawValue)_typed_content_and_metadata"
            do {
                let owner: CoreAIExternalWeights
                if storage == .resident {
                    // Omit storage deliberately: per-tensor resident is default.
                    owner = try CoreAIExternalWeights(spec: fixture.spec(), baseURL: fixture.base,
                        device: device, verifyIntegrity: true)
                } else {
                    owner = try CoreAIExternalWeights(spec: fixture.spec(), baseURL: fixture.base,
                        device: device, storage: storage, verifyIntegrity: true)
                }
                try require(owner.storage == storage, "storage policy differs")
                try require(owner.byteLength == fixture.data.count, "file byte count differs")
                try require(owner.logicalByteCount == fixture.slices.values.reduce(0) { $0 + $1.count },
                            "logical byte count includes padding or drops a slice")
                try require(owner.inputNames == Set(fixture.slices.keys), "input names differ")
                try require(owner.fileURL == fixture.file.standardizedFileURL.resolvingSymlinksInPath(),
                            "canonical file URL differs")
                try await checkViews(owner.values, fixture: fixture)
                results.append(["name": name, "passed": true])
            } catch { results.append(["name": name, "passed": false, "error": String(describing: error)]) }

            let lifetimeName = "\(storage.rawValue)_views_outlive_owner_and_unlinked_file"
            do {
                let separate = try Fixture(in: root.appendingPathComponent("lifetime-\(storage.rawValue)"))
                let witness = WeakOwner()
                let values = try detachedViews(fixture: separate, device: device, storage: storage, witness: witness)
                try require(witness.value == nil, "test still retains the loader owner")
                // Unlink instead of truncating: a live mmap may safely retain
                // the unlinked inode; truncating a live mapping can SIGBUS.
                try FileManager.default.removeItem(at: separate.file)
                try await checkViews(values, fixture: separate)
                // A second materialization also checks reusable immutable views.
                try await checkViews(values, fixture: separate)
                results.append(["name": lifetimeName, "passed": true])
            } catch { results.append(["name": lifetimeName, "passed": false,
                                       "error": String(describing: error)]) }
        }

        rejects("zero_dimension", "slice metadata", fixture: fixture) {
            editSlice(&$0) { $0["shape"] = [0, 3] }
        }
        rejects("negative_dimension", "slice metadata", fixture: fixture) {
            editSlice(&$0) { $0["shape"] = [-2, 3] }
        }
        rejects("empty_shape", "slice metadata", fixture: fixture) {
            editSlice(&$0) { $0["shape"] = [Int]() }
        }
        rejects("shape_byte_count_overflow", "shape byte count overflows", fixture: fixture) {
            editSlice(&$0) { $0["shape"] = [Int.max, 2] }
        }
        rejects("slice_length_mismatch", "incorrectly sized", fixture: fixture) {
            editSlice(&$0) { $0["byteLength"] = 10 }
        }
        rejects("offset_plus_length_overflow", "out-of-bounds", fixture: fixture) {
            editSlice(&$0, 3) {
                $0["byteOffset"] = Int.max - (Int.max % 16_384)
                $0["shape"] = [8192]; $0["byteLength"] = 32_768
            }
        }
        rejects("slice_past_end", "out-of-bounds", fixture: fixture) {
            editSlice(&$0, 3) { $0["byteOffset"] = fixture.data.count }
        }
        rejects("overlapping_slices", "overlapping", fixture: fixture) {
            editSlice(&$0, 1) { $0["byteOffset"] = 0 }
        }
        rejects("duplicate_input_name", "duplicate names", fixture: fixture) {
            editSlice(&$0, 1) { $0["inputName"] = "w_half" }
        }
        rejects("duplicate_buffer_name", "duplicate names", fixture: fixture) {
            editSlice(&$0, 1) { $0["bufferName"] = "module.w_half" }
        }
        rejects("empty_buffer_set", "Invalid external-weight", fixture: fixture) { $0["buffers"] = [[String: Any]]() }
        rejects("empty_input_name", "slice metadata", fixture: fixture) { editSlice(&$0) { $0["inputName"] = "" } }
        rejects("nul_input_name", "slice metadata", fixture: fixture) { editSlice(&$0) { $0["inputName"] = "w\0half" } }
        rejects("unsupported_dtype", "Unsupported external-weight dtype", fixture: fixture) {
            editSlice(&$0) { $0["dtype"] = "float64" }
        }
        rejects("invalid_alignment", "alignment", fixture: fixture) { $0["alignment"] = 3 }
        rejects("misaligned_slice_offset", "slice metadata", fixture: fixture) {
            editSlice(&$0, 1) { $0["byteOffset"] = 16_386 }
        }
        rejects("negative_slice_offset", "slice metadata", fixture: fixture) {
            editSlice(&$0) { $0["byteOffset"] = -16_384 }
        }
        rejects("misaligned_file_length", "alignment", fixture: fixture) { $0["byteLength"] = fixture.data.count - 1 }
        rejects("wrong_byte_order", "byte order", fixture: fixture) { $0["byteOrder"] = "big" }
        rejects("invalid_file_hash_syntax", "hashes", fixture: fixture) { $0["sha256"] = String(repeating: "g", count: 64) }
        rejects("invalid_slice_hash_syntax", "slice metadata", fixture: fixture) {
            editSlice(&$0) { $0["sha256"] = "abc" }
        }
        rejects("absolute_path", "relative local path", fixture: fixture) { $0["path"] = fixture.file.path }
        rejects("nul_path", "relative local path", fixture: fixture) { $0["path"] = "weights.bin\0suffix" }
        rejects("parent_directory_escape", "escapes", fixture: fixture) { $0["path"] = "../outside.bin" }
        rejects("base_directory_itself", "escapes", fixture: fixture) { $0["path"] = "." }

        let outside = root.appendingPathComponent("outside.bin")
        try fixture.data.write(to: outside)
        try FileManager.default.createSymbolicLink(at: fixture.base.appendingPathComponent("outside-link.bin"),
                                                   withDestinationURL: outside)
        rejects("symlink_escape", "escapes", fixture: fixture) { $0["path"] = "outside-link.bin" }
        let truncated = try Fixture(in: root.appendingPathComponent("truncated"))
        try Data(truncated.data.dropLast()).write(to: truncated.file)
        rejects("truncated_file", "match its declared byteLength", fixture: truncated) { _ in }
        let oversized = try Fixture(in: root.appendingPathComponent("oversized"))
        var oversizedData = oversized.data
        oversizedData.append(0)
        try oversizedData.write(to: oversized.file)
        rejects("oversized_file", "match its declared byteLength", fixture: oversized) { _ in }

        rejects("file_hash_mismatch", "file SHA256 differs", fixture: fixture, integrity: true) {
            $0["sha256"] = String(repeating: "0", count: 64)
        }
        rejects("slice_hash_mismatch", "slice SHA256 differs", fixture: fixture, integrity: true) {
            editSlice(&$0) { $0["sha256"] = String(repeating: "0", count: 64) }
        }
        let padding = try Fixture(in: root.appendingPathComponent("padding-corruption"))
        var changedPadding = padding.data
        changedPadding[128] ^= 1 // Outside every logical tensor slice.
        try changedPadding.write(to: padding.file)
        for storage in [CoreAIExternalWeightStorage.resident, .residentFile, .mapped] {
            rejects("\(storage.rawValue)_whole_file_hash_includes_padding", "file SHA256 differs",
                    fixture: padding, integrity: true, storage: storage) { _ in }
        }
        record("integrity_is_explicitly_opt_in") {
            let owner = try CoreAIExternalWeights(spec: fixture.spec { $0["sha256"] = String(repeating: "0", count: 64) },
                baseURL: fixture.base, device: device)
            try require(owner.byteLength == fixture.data.count, "default integrity policy changed")
        }
        record("uppercase_hash_and_omitted_byte_order") {
            _ = try CoreAIExternalWeights(spec: fixture.spec {
                $0["sha256"] = digest(fixture.data).uppercased()
                $0.removeValue(forKey: "byteOrder")
                editSlice(&$0) { $0["sha256"] = ($0["sha256"] as! String).uppercased() }
            }, baseURL: fixture.base, device: device, verifyIntegrity: true)
        }

        let passed = results.allSatisfy { $0["passed"] as? Bool == true }
        let report: [String: Any] = ["passed": passed, "case_count": results.count, "cases": results,
            "fixture_file_bytes": fixture.data.count, "device": device.name,
            "scope": "Production external-weight loader; shared Metal allocation and host NDArray reads only. No command queue, AIModel, inference or GPU commands. Descriptor merging is not covered."]
        let output = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        if CommandLine.arguments.count == 2 {
            try output.write(to: URL(fileURLWithPath: CommandLine.arguments[1]))
        }
        print(String(decoding: output, as: UTF8.self))
        if !passed { throw CheckFailure(description: "external-weight loader checks failed") }
    }
}
