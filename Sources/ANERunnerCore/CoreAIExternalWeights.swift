#if canImport(CoreAI)
import CoreAI
import CryptoKit
import Darwin
import Foundation
import Metal

/// The `weights` object in an external-weight manifest. Shapes describe dense,
/// contiguous, little-endian tensors; offsets refer to one aligned raw file.
struct CoreAIExternalWeightsSpec: Decodable {
    struct Buffer: Decodable {
        let inputName: String
        let bufferName: String
        let dtype: String
        let shape: [Int]
        let byteOffset: Int
        let byteLength: Int
        let sha256: String
    }
    let path: String
    let alignment: Int
    let byteLength: Int
    let sha256: String
    let byteOrder: String?
    let buffers: [Buffer]
}

enum CoreAIExternalWeightStorage: String {
    /// Reads the file once directly into a shared Metal allocation.
    case resident
    /// Experimental file mapping; performance can differ from resident storage.
    case mapped
}

/// Retains one weight allocation and immutable inference-input views into it.
/// Callers must retain the owner for their model lifetime and must not expose
/// these weight inputs as mutable model state or outputs.
@available(macOS 27.0, *)
final class CoreAIExternalWeights {
    let fileURL: URL
    let storage: CoreAIExternalWeightStorage
    let byteLength: Int
    let logicalByteCount: Int
    let inputNames: Set<String>
    let values: [String: InferenceFunction.AsyncValue]

    private let spec: CoreAIExternalWeightsSpec
    private let buffer: any MTLBuffer

    /// Integrity verification is opt-in. It hashes the resident/mapped bytes,
    /// including each declared slice, without rereading a second file copy.
    init(spec: CoreAIExternalWeightsSpec, baseURL: URL, device: any MTLDevice,
         storage: CoreAIExternalWeightStorage = .resident, verifyIntegrity: Bool = false) throws {
        let logicalBytes = try Self.validateMetadata(spec)
        guard baseURL.isFileURL, !spec.path.isEmpty, !spec.path.hasPrefix("/"),
              !spec.path.utf8.contains(0) else {
            throw Self.invalid("Weight file must be a relative local path")
        }
        let base = baseURL.standardizedFileURL.resolvingSymlinksInPath()
        let url = base.appendingPathComponent(spec.path).standardizedFileURL.resolvingSymlinksInPath()
        let prefix = base.path.hasSuffix("/") ? base.path : base.path + "/"
        guard url.path.hasPrefix(prefix), url.path != base.path else {
            throw Self.invalid("Weight file escapes the manifest directory")
        }
        let descriptor = Darwin.open(url.path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW)
        guard descriptor >= 0 else { throw Self.fileError("open", path: url.path) }
        defer { Darwin.close(descriptor) }
        var information = stat()
        guard fstat(descriptor, &information) == 0 else { throw Self.fileError("fstat", path: url.path) }
        guard information.st_mode & mode_t(S_IFMT) == mode_t(S_IFREG),
              information.st_size >= 0, information.st_size == off_t(spec.byteLength) else {
            throw Self.invalid("Weight file must be regular and match its declared byteLength")
        }

        let allocation: any MTLBuffer
        switch storage {
        case .resident:
            guard spec.byteLength <= device.maxBufferLength,
                  let created = device.makeBuffer(length: spec.byteLength, options: .storageModeShared) else {
                throw Self.invalid("Cannot allocate the resident external-weight buffer")
            }
            try Self.readExactly(descriptor, into: created.contents(), count: spec.byteLength, path: url.path)
            allocation = created
        case .mapped:
            let page = Int(getpagesize())
            guard page > 0 else { throw Self.invalid("Invalid virtual-memory page size") }
            let padded = spec.byteLength.addingReportingOverflow(page - 1)
            guard !padded.overflow else { throw Self.invalid("Mapped weight length overflows Int") }
            let mappingLength = (padded.partialValue / page) * page
            guard mappingLength <= device.maxBufferLength else {
                throw Self.invalid("Mapped weights exceed the Metal buffer length limit")
            }
            let address = mmap(nil, mappingLength, PROT_READ | PROT_WRITE, MAP_PRIVATE, descriptor, 0)
            guard let address, address != MAP_FAILED else { throw Self.fileError("mmap", path: url.path) }
            guard let created = device.makeBuffer(bytesNoCopy: address, length: mappingLength,
                options: .storageModeShared, deallocator: { pointer, length in
                    _ = munmap(pointer, length)
                }) else {
                _ = munmap(address, mappingLength)
                throw Self.invalid("Cannot wrap the mapped external-weight file in a Metal buffer")
            }
            allocation = created
        }
        if verifyIntegrity {
            guard try Self.digest(allocation.contents(), count: spec.byteLength) == spec.sha256.lowercased() else {
                throw Self.invalid("External-weight file SHA256 differs from the manifest")
            }
            for entry in spec.buffers {
                guard try Self.digest(allocation.contents().advanced(by: entry.byteOffset), count: entry.byteLength)
                        == entry.sha256.lowercased() else {
                    throw Self.invalid("\(entry.inputName): external-weight slice SHA256 differs from the manifest")
                }
            }
        }
        var views: [String: InferenceFunction.AsyncValue] = [:]
        for entry in spec.buffers {
            let type = try Self.scalarType(entry.dtype)
            views[entry.inputName] = InferenceFunction.AsyncValue(unsafeBuffer: allocation,
                byteOffset: entry.byteOffset, scalarType: type, shape: entry.shape)
        }
        self.spec = spec
        self.fileURL = url
        self.storage = storage
        self.byteLength = spec.byteLength
        self.logicalByteCount = logicalBytes
        self.inputNames = Set(views.keys)
        self.buffer = allocation
        self.values = views
    }

    /// Checks only declared weight inputs; activation inputs are supplied later.
    func validateInputs(for descriptor: InferenceFunctionDescriptor) throws {
        guard descriptor.stateNames.isEmpty else {
            throw Self.invalid("External-weight functions require explicit tensor state inputs")
        }
        for entry in spec.buffers {
            guard case .ndArray(let declared) = descriptor.inputDescriptor(of: entry.inputName) else {
                throw Self.invalid("\(entry.inputName): missing tensor weight input")
            }
            try Self.validate(shape: entry.shape, type: Self.scalarType(entry.dtype), declared: declared,
                              name: entry.inputName)
        }
    }

    /// Activations/states remain NDArrays until their descriptors have been
    /// checked. No host materialization or weight tensor copy is performed.
    func merging(activations: [String: NDArray], for descriptor: InferenceFunctionDescriptor) throws
        -> [String: InferenceFunction.AsyncValue] {
        try validateInputs(for: descriptor)
        guard inputNames.isDisjoint(with: activations.keys),
              Set(descriptor.inputNames) == inputNames.union(activations.keys) else {
            throw Self.invalid("Weight and activation input names overlap or leave an incomplete function input set")
        }
        var merged = values
        for (name, array) in activations {
            guard array.interleaveLayout == nil,
                  case .ndArray(let declared) = descriptor.inputDescriptor(of: name) else {
                throw Self.invalid("\(name): unsupported activation tensor input")
            }
            try Self.validate(shape: array.shape, type: array.scalarType, declared: declared, name: name)
            merged[name] = InferenceFunction.AsyncValue(array)
        }
        return merged
    }

    private static func validateMetadata(_ spec: CoreAIExternalWeightsSpec) throws -> Int {
        guard spec.alignment >= 4, spec.alignment & (spec.alignment - 1) == 0,
              spec.byteLength > 0, spec.byteLength.isMultiple(of: spec.alignment),
              validHash(spec.sha256), spec.byteOrder == nil || spec.byteOrder == "little",
              !spec.buffers.isEmpty, Set(spec.buffers.map(\.inputName)).count == spec.buffers.count,
              Set(spec.buffers.map(\.bufferName)).count == spec.buffers.count else {
            throw invalid("Invalid external-weight alignment, length, byte order, hashes or duplicate names")
        }
        var logicalBytes = 0
        var previousEnd = 0
        for entry in spec.buffers.sorted(by: { $0.byteOffset < $1.byteOffset }) {
            let type = try scalarType(entry.dtype)
            let width = type == .float16 || type == .int16 ? 2 : 4
            guard !entry.inputName.isEmpty, !entry.bufferName.isEmpty,
                  !entry.inputName.utf8.contains(0), !entry.bufferName.utf8.contains(0),
                  validHash(entry.sha256), !entry.shape.isEmpty,
                  entry.shape.allSatisfy({ $0 > 0 }), entry.byteOffset >= 0,
                  entry.byteOffset.isMultiple(of: spec.alignment), entry.byteLength > 0 else {
                throw invalid("\(entry.inputName): invalid external-weight slice metadata")
            }
            var expectedBytes = width
            for dimension in entry.shape {
                let next = expectedBytes.multipliedReportingOverflow(by: dimension)
                guard !next.overflow else { throw invalid("\(entry.inputName): shape byte count overflows Int") }
                expectedBytes = next.partialValue
            }
            let end = entry.byteOffset.addingReportingOverflow(entry.byteLength)
            let total = logicalBytes.addingReportingOverflow(entry.byteLength)
            guard expectedBytes == entry.byteLength, !end.overflow, !total.overflow,
                  end.partialValue <= spec.byteLength, entry.byteOffset >= previousEnd else {
                throw invalid("\(entry.inputName): overlapping, out-of-bounds or incorrectly sized weight slice")
            }
            previousEnd = end.partialValue
            logicalBytes = total.partialValue
        }
        return logicalBytes
    }

    private static func validate(shape: [Int], type: NDArray.ScalarType,
                                 declared: NDArrayDescriptor, name: String) throws {
        guard !shape.isEmpty, shape.allSatisfy({ $0 > 0 }), !declared.hasDynamicShape,
              declared.interleaveLayout == nil, declared.scalarType == type,
              declared.rank == shape.count, declared.shape == shape else {
            throw invalid("\(name): tensor shape/dtype differs from the fixed external-weight function descriptor")
        }
    }

    private static func scalarType(_ value: String) throws -> NDArray.ScalarType {
        switch value {
        case "float16": return .float16
        case "float32": return .float32
        case "int16": return .int16
        case "int32": return .int32
        default: throw invalid("Unsupported external-weight dtype: \(value)")
        }
    }

    private static func readExactly(_ descriptor: Int32, into pointer: UnsafeMutableRawPointer,
                                    count: Int, path: String) throws {
        var completed = 0
        while completed < count {
            try Task.checkCancellation()
            let amount = Darwin.pread(descriptor, pointer.advanced(by: completed),
                                      min(16 * 1024 * 1024, count - completed), off_t(completed))
            if amount < 0 {
                if errno == EINTR { continue }
                throw fileError("pread", path: path)
            }
            guard amount > 0 else { throw invalid("External-weight file ended during resident loading") }
            completed += amount
        }
    }

    private static func digest(_ pointer: UnsafeRawPointer, count: Int) throws -> String {
        var hash = SHA256()
        var offset = 0
        while offset < count {
            try Task.checkCancellation()
            let length = min(16 * 1024 * 1024, count - offset)
            hash.update(bufferPointer: UnsafeRawBufferPointer(start: pointer.advanced(by: offset), count: length))
            offset += length
        }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private static func validHash(_ value: String) -> Bool {
        value.utf8.count == 64 && value.utf8.allSatisfy {
            (48...57).contains($0) || (65...70).contains($0) || (97...102).contains($0)
        }
    }

    private static func invalid(_ message: String) -> CoreAIBlockRunnerError {
        .invalidModel(message)
    }

    private static func fileError(_ operation: String, path: String) -> CoreAIBlockRunnerError {
        let code = errno
        return invalid("\(operation) external weights \(path): \(String(cString: strerror(code))) (errno \(code))")
    }
}
#endif
