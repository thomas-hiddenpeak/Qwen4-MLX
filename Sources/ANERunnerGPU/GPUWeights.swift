import Foundation
import CoreFoundation
import CMLX

public enum GPUWeightError: Error, CustomStringConvertible {
    case invalid(String)
    public var description: String { switch self { case .invalid(let message): return message } }
}

/// Confined to the model's inference thread. MLX owns tensor storage; this class
/// never expands BF16 or packed U32 weights into a second host copy.
public final class GPUWeights {
    public struct TensorMetadata {
        public let name: String
        public let shard: String
        public let shape: [Int]
        public let dtype: mlx_dtype
        public let dtypeName: String
        public let byteOffset: UInt64
        public let byteCount: UInt64
    }
    public struct LedgerEntry: Codable {
        public let name: String
        public let shard: String
        public let sourceBytes: UInt64
        public var cached: Bool
        public var loadCount: UInt64
    }
    public let modelDirectory: URL
    public let weightMap: [String: String]
    private var headers: [String: [String: TensorMetadata]] = [:]
    private var cache: [String: Tensor] = [:]
    private var entries: [String: LedgerEntry] = [:]
    public var ledger: [LedgerEntry] { entries.values.sorted { $0.name < $1.name } }
    /// Bytes retained by this loader, excluding external references and MLX's allocator cache.
    public var cachedSourceBytes: UInt64 { entries.values.filter(\.cached).reduce(0) { $0 + $1.sourceBytes } }

    public init(modelDirectory: URL) throws {
        self.modelDirectory = modelDirectory.standardizedFileURL.resolvingSymlinksInPath()
        let index = self.modelDirectory.appendingPathComponent("model.safetensors.index.json")
        guard let root = try JSONSerialization.jsonObject(with: Data(contentsOf: index)) as? [String: Any],
              let map = root["weight_map"] as? [String: String], !map.isEmpty else {
            throw GPUWeightError.invalid("Missing or invalid safetensors weight_map")
        }
        for shard in Set(map.values) {
            guard shard == URL(fileURLWithPath: shard).lastPathComponent,
                  shard.hasSuffix(".safetensors"), !shard.hasPrefix(".") else {
                throw GPUWeightError.invalid("Unsafe safetensors shard path: \(shard)")
            }
        }
        self.weightMap = map
    }

    public func contains(_ name: String) -> Bool { weightMap[name] != nil }

    public func metadata(_ name: String) throws -> TensorMetadata {
        guard let shard = weightMap[name] else { throw GPUWeightError.invalid("Missing weight: \(name)") }
        if headers[shard] == nil { headers[shard] = try readHeader(shard) }
        guard let result = headers[shard]?[name] else {
            throw GPUWeightError.invalid("Index tensor \(name) absent from \(shard)")
        }
        return result
    }

    public func tensor(_ name: String) throws -> Tensor {
        if let existing = cache[name] { return existing }
        let info = try metadata(name)
        // MLX safetensors loading creates lazy Load nodes, not materialized
        // shard data. Only the selected handle is evaluated; the map is freed
        // before returning, leaving unrelated MTP/vision tensors unloaded.
        var arrays = mlx_map_string_to_array_new()
        var strings = mlx_map_string_to_string_new()
        defer { _ = mlx_map_string_to_array_free(arrays); _ = mlx_map_string_to_string_free(strings) }
        // Load is a CPU IO primitive in the pinned MLX build (eval_gpu is not
        // implemented). Unified-memory buffers feed subsequent GPU ops without
        // expanding or copying their numeric representation in Swift.
        try MX.check(0, "initialize IO error handling")
        let stream = mlx_default_cpu_stream_new()
        defer { _ = mlx_stream_free(stream) }
        let path = modelDirectory.appendingPathComponent(info.shard).path
        try MX.check(path.withCString { mlx_load_safetensors(&arrays, &strings, $0, stream) }, "load safetensors \(info.shard)")
        let result = try MX.output("load weight \(name)") { out in
            name.withCString { mlx_map_string_to_array_get(&out, arrays, $0) }
        }
        guard result.shape == info.shape, result.dtype == info.dtype else {
            throw GPUWeightError.invalid("MLX/header type or shape mismatch for \(name)")
        }
        try result.eval()
        cache[name] = result
        var entry = entries[name] ?? LedgerEntry(name: name, shard: info.shard, sourceBytes: info.byteCount, cached: false, loadCount: 0)
        entry.cached = true
        entry.loadCount += 1
        entries[name] = entry
        return result
    }

    /// Releases only this loader's ownership. Existing layer references remain valid.
    public func release(prefix: String) {
        for name in cache.keys.filter({ $0.hasPrefix(prefix) }) {
            cache.removeValue(forKey: name)
            entries[name]?.cached = false
        }
    }
    public func releaseAll() {
        cache.removeAll()
        for name in Array(entries.keys) { entries[name]?.cached = false }
    }

    private func readHeader(_ shard: String) throws -> [String: TensorMetadata] {
        let url = modelDirectory.appendingPathComponent(shard).resolvingSymlinksInPath()
        guard url.deletingLastPathComponent() == modelDirectory else { throw GPUWeightError.invalid("Shard symlink escapes model directory") }
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        let size = try file.seekToEnd()
        try file.seek(toOffset: 0)
        guard let first = try file.read(upToCount: 8), first.count == 8 else { throw GPUWeightError.invalid("Truncated safetensors: \(shard)") }
        let headerSize = first.enumerated().reduce(UInt64(0)) { $0 | (UInt64($1.element) << (8 * $1.offset)) }
        guard headerSize <= 64 * 1024 * 1024, size >= 8, headerSize <= size - 8,
              let data = try file.read(upToCount: Int(headerSize)), data.count == Int(headerSize),
              let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw GPUWeightError.invalid("Invalid safetensors header: \(shard)")
        }
        let payload = headerSize + 8
        var result: [String: TensorMetadata] = [:]
        var intervals: [(UInt64, UInt64)] = []
        for (name, raw) in json where name != "__metadata__" {
            guard let object = raw as? [String: Any], let dtypeName = object["dtype"] as? String,
                  let shapeRaw = object["shape"] as? [Any], let offsets = object["data_offsets"] as? [Any], offsets.count == 2 else {
                throw GPUWeightError.invalid("Invalid tensor descriptor: \(name)")
            }
            let (dtype, width) = try Self.dtype(dtypeName)
            let shape64 = try shapeRaw.map(Self.uint)
            guard shape64.count <= 32, shape64.allSatisfy({ $0 <= UInt64(Int32.max) }) else { throw GPUWeightError.invalid("Unsupported tensor rank/dimension: \(name)") }
            var bytes = width
            for dimension in shape64 {
                let (product, overflow) = bytes.multipliedReportingOverflow(by: dimension)
                guard !overflow else { throw GPUWeightError.invalid("Tensor byte count overflow: \(name)") }
                bytes = product
            }
            let lower = try Self.uint(offsets[0]), upper = try Self.uint(offsets[1])
            guard upper >= lower, upper - lower == bytes, upper <= size - payload else { throw GPUWeightError.invalid("Invalid tensor bounds: \(name)") }
            result[name] = TensorMetadata(name: name, shard: shard, shape: shape64.map(Int.init), dtype: dtype, dtypeName: dtypeName, byteOffset: payload + lower, byteCount: bytes)
            if upper > lower { intervals.append((lower, upper)) }
        }
        intervals.sort { $0.0 < $1.0 }
        var end: UInt64 = 0
        for interval in intervals {
            guard interval.0 >= end else { throw GPUWeightError.invalid("Overlapping safetensors payload: \(shard)") }
            end = interval.1
        }
        return result
    }

    private static func uint(_ value: Any) throws -> UInt64 {
        guard let n = value as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID(),
              let value = UInt64(n.stringValue) else { throw GPUWeightError.invalid("Expected nonnegative JSON integer") }
        return value
    }
    private static func dtype(_ name: String) throws -> (mlx_dtype, UInt64) {
        switch name {
        case "BOOL": return (MLX_BOOL, 1)
        case "U8": return (MLX_UINT8, 1)
        case "U16": return (MLX_UINT16, 2)
        case "U32": return (MLX_UINT32, 4)
        case "U64": return (MLX_UINT64, 8)
        case "I8": return (MLX_INT8, 1)
        case "I16": return (MLX_INT16, 2)
        case "I32": return (MLX_INT32, 4)
        case "I64": return (MLX_INT64, 8)
        case "F16": return (MLX_FLOAT16, 2)
        case "BF16": return (MLX_BFLOAT16, 2)
        case "F32": return (MLX_FLOAT32, 4)
        case "F64": return (MLX_FLOAT64, 8)
        default: throw GPUWeightError.invalid("Unsupported safetensors dtype \(name)")
        }
    }
}
