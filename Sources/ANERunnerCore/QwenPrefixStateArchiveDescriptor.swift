import Foundation

/// CPU-only schema for the dedicated Qwen3.8 AR cache format. Validation must
/// finish before any archive-controlled shape reaches a device allocator.
public struct QwenPrefixStateArchiveDescriptor: Codable, Equatable, Sendable {
    public static let currentVersion = 1
    public static let maximumMetadataBytes = 65_536
    public static let defaultMaximumPayloadBytes = 1_073_741_824
    public static let absoluteMaximumPayloadBytes = 2_147_483_648

    public enum ValidationError: Error, LocalizedError, Equatable {
        case invalid(String)
        /// Only archive-controlled decoding/structure failures after the local
        /// import layout, offset and payload allowance have been validated.
        case invalidArchive(String)
        public var errorDescription: String? {
            switch self {
            case .invalid(let message), .invalidArchive(let message): return "Prefix archive: " + message
            }
        }
    }

    /// Weight identity and numerical execution policy are checked by the cache
    /// namespace. This layout independently protects state structure and shape.
    public struct Layout: Codable, Equatable, Sendable {
        public let layerTypes: [String]
        public let pleLayerIndices: [Int]
        public let hiddenSize, hcCount, ngramSize, pleConvKernel: Int
        public let vocabularySize, maximumPositions: Int

        public init(layerTypes: [String], pleLayerIndices: [Int], hiddenSize: Int,
                    hcCount: Int, ngramSize: Int, pleConvKernel: Int,
                    vocabularySize: Int, maximumPositions: Int) {
            self.layerTypes = layerTypes; self.pleLayerIndices = pleLayerIndices
            self.hiddenSize = hiddenSize; self.hcCount = hcCount
            self.ngramSize = ngramSize; self.pleConvKernel = pleConvKernel
            self.vocabularySize = vocabularySize; self.maximumPositions = maximumPositions
        }

        public func validate() throws {
            guard layerTypes == (0..<48).map({ $0 % 4 == 3 ? "full_attention" : "linear_attention" }),
                  pleLayerIndices == [1], hiddenSize == 2560, hcCount == 4,
                  ngramSize == 3, pleConvKernel == 4, vocabularySize == 248_320,
                  maximumPositions == 262_144 else {
                throw ValidationError.invalid("unsupported model layout")
            }
        }
    }

    public struct TensorDescriptor: Codable, Equatable, Sendable {
        public let name: String
        public let shape: [Int]
        public let dtype: String
        public let byteOffset, byteCount: Int

        public init(name: String, shape: [Int], dtype: String = "bfloat16-le",
                    byteOffset: Int, byteCount: Int) {
            self.name = name; self.shape = shape; self.dtype = dtype
            self.byteOffset = byteOffset; self.byteCount = byteCount
        }
    }

    public var version: Int
    public var layout: Layout
    public var offset: Int
    public var gdnOffsets, attentionOffsets: [Int]
    public var pleHistory: [[UInt32]]
    public var tensors: [TensorDescriptor]
    public var tensorPayloadBytes, logicalPayloadBytes: Int

    public init(layout: Layout, offset: Int, gdnOffsets: [Int], attentionOffsets: [Int],
                pleHistory: [[UInt32]], tensors: [TensorDescriptor], tensorPayloadBytes: Int,
                logicalPayloadBytes: Int, version: Int = QwenPrefixStateArchiveDescriptor.currentVersion) {
        self.version = version; self.layout = layout; self.offset = offset
        self.gdnOffsets = gdnOffsets; self.attentionOffsets = attentionOffsets
        self.pleHistory = pleHistory; self.tensors = tensors
        self.tensorPayloadBytes = tensorPayloadBytes; self.logicalPayloadBytes = logicalPayloadBytes
    }

    /// The canonical order rules out unknown, duplicate, missing, overlapping,
    /// gapped and oversized tensors without multiplying untrusted dimensions.
    public static func expectedTensors(layout: Layout, offset: Int) throws -> [TensorDescriptor] {
        try layout.validate()
        guard (1...layout.maximumPositions).contains(offset) else {
            throw ValidationError.invalid("offset outside model context")
        }
        var tensors: [TensorDescriptor] = []
        var cursor = 0
        func append(_ name: String, _ shape: [Int]) throws {
            var bytes = 2
            for dimension in shape {
                let next = bytes.multipliedReportingOverflow(by: dimension)
                guard !next.overflow, dimension > 0 else {
                    throw ValidationError.invalid("tensor size overflow")
                }
                bytes = next.partialValue
            }
            let end = cursor.addingReportingOverflow(bytes)
            guard !end.overflow else { throw ValidationError.invalid("payload size overflow") }
            tensors.append(TensorDescriptor(name: name, shape: shape, byteOffset: cursor, byteCount: bytes))
            cursor = end.partialValue
        }
        for layer in 0..<48 {
            let prefix = "layer.\(layer)."
            if layout.layerTypes[layer] == "linear_attention" {
                try append(prefix + "gdn.recurrent", [1, 48, 128, 128])
                try append(prefix + "gdn.conv", [1, 3, 10_240])
            } else {
                try append(prefix + "attention.keys", [1, 2, offset, 256])
                try append(prefix + "attention.values", [1, 2, offset, 256])
                try append(prefix + "attention.raw_index", [1, offset, 128])
                if offset > 2051 {
                    try append(prefix + "attention.pooled_index", [1, offset / 4, 128])
                }
            }
            if layout.pleLayerIndices.contains(layer) {
                try append(prefix + "ple.conv", [1, 9, 10_240])
            }
        }
        return tensors
    }

    public static func estimatedLogicalPayloadBytes(layout: Layout, offset: Int) throws -> Int {
        let tensors = try expectedTensors(layout: layout, offset: offset)
        guard let final = tensors.last else { throw ValidationError.invalid("empty tensor layout") }
        // Layout validation bounds the layer/history counts before arithmetic.
        let historyBytes = layout.pleLayerIndices.count * (layout.ngramSize - 1) * MemoryLayout<UInt32>.stride
        return final.byteOffset + final.byteCount + historyBytes
    }

    public func validate(expectedLayout: Layout, expectedOffset: Int, actualPayloadBytes: Int,
                         maxPayloadBytes: Int = QwenPrefixStateArchiveDescriptor.defaultMaximumPayloadBytes) throws {
        try expectedLayout.validate()
        guard (1...Self.absoluteMaximumPayloadBytes).contains(maxPayloadBytes),
              version == Self.currentVersion, layout == expectedLayout,
              offset == expectedOffset, (1...layout.maximumPositions).contains(offset),
              actualPayloadBytes > 0, actualPayloadBytes <= maxPayloadBytes,
              tensorPayloadBytes == actualPayloadBytes else {
            throw ValidationError.invalid("version, layout, offset or payload limit mismatch")
        }
        let expected = try Self.expectedTensors(layout: expectedLayout, offset: expectedOffset)
        guard tensors == expected, let final = expected.last,
              final.byteOffset + final.byteCount == tensorPayloadBytes,
              gdnOffsets == layout.layerTypes.map({ $0 == "linear_attention" ? offset : 0 }),
              attentionOffsets == layout.layerTypes.map({ $0 == "full_attention" ? offset : 0 }),
              pleHistory.count == 48 else {
            throw ValidationError.invalid("incomplete or inconsistent state descriptors")
        }
        var historyBytes = 0
        for layer in 0..<48 {
            let history = pleHistory[layer]
            let active = layout.pleLayerIndices.contains(layer)
            guard history.count == (active ? layout.ngramSize - 1 : 0),
                  history.allSatisfy({ $0 < UInt32(layout.vocabularySize) }) else {
                throw ValidationError.invalid("invalid PLE token history")
            }
            historyBytes += history.count * MemoryLayout<UInt32>.stride
        }
        let total = tensorPayloadBytes.addingReportingOverflow(historyBytes)
        guard !total.overflow, logicalPayloadBytes == total.partialValue,
              logicalPayloadBytes <= maxPayloadBytes else {
            throw ValidationError.invalid("logical payload accounting mismatch or limit exceeded")
        }
    }

    public func encoded() throws -> Data {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(self)
        guard data.count <= Self.maximumMetadataBytes else {
            throw ValidationError.invalid("metadata limit exceeded")
        }
        return data
    }

    public static func decodeAndValidate(_ metadata: Data, expectedLayout: Layout,
                                         expectedOffset: Int, actualPayloadBytes: Int,
                                         maxPayloadBytes: Int = QwenPrefixStateArchiveDescriptor.defaultMaximumPayloadBytes) throws -> Self {
        // A valid archive rejected by a caller's layout, offset or allowance
        // is not evidence of disk corruption. Derive the required size from
        // the model before inspecting any archive-controlled fields.
        let requiredBytes = try estimatedLogicalPayloadBytes(layout: expectedLayout, offset: expectedOffset)
        guard (1...absoluteMaximumPayloadBytes).contains(maxPayloadBytes),
              requiredBytes <= maxPayloadBytes, actualPayloadBytes >= 0 else {
            throw ValidationError.invalid("invalid import allowance or payload byte count")
        }
        guard !metadata.isEmpty, metadata.count <= maximumMetadataBytes else {
            throw ValidationError.invalidArchive("metadata limit exceeded")
        }
        let descriptor: Self
        do { descriptor = try JSONDecoder().decode(Self.self, from: metadata) }
        catch let error as DecodingError {
            throw ValidationError.invalidArchive("metadata decoding failed: \(error)")
        }
        do {
            try descriptor.validate(expectedLayout: expectedLayout, expectedOffset: expectedOffset,
                                    actualPayloadBytes: actualPayloadBytes, maxPayloadBytes: maxPayloadBytes)
        } catch ValidationError.invalid(let message) {
            // validate() keeps its existing local/export contract. Here its
            // local arguments are already checked, leaving only file content.
            throw ValidationError.invalidArchive(message)
        }
        return descriptor
    }

    /// Shared CPU-only disk invalidation policy. Arbitrary device, admission,
    /// cancellation or later promotion errors do not establish a bad archive.
    public static func isInvalidArchive(_ error: Error) -> Bool {
        guard let error = error as? ValidationError else { return false }
        if case .invalidArchive = error { return true }
        return false
    }
}
