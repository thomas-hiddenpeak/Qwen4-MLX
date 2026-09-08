import CMLX
import Darwin
import Foundation

/// Probe-only scalar observations. No Tensor/native/Metal owner is returned.
struct GPUKVAllocationSnapshot: Equatable {
    let metalBuffer, offsetBytes, allocationBytes, logicalBytes, dataElements: UInt64
    let contiguous, rowContiguous, donatableAtObservation: Bool
    func sameAllocation(as other: Self) -> Bool {
        metalBuffer == other.metalBuffer && offsetBytes == other.offsetBytes &&
            allocationBytes == other.allocationBytes
    }
    var json: [String: Any] {
        ["metal_buffer": "0x" + String(metalBuffer, radix: 16), "offset_bytes": offsetBytes,
         "allocation_bytes": allocationBytes, "logical_bytes": logicalBytes,
         "data_elements": dataElements, "contiguous": contiguous,
         "row_contiguous": rowContiguous, "donatable_at_observation": donatableAtObservation]
    }
}
struct GPUKVAllocationPair {
    let keys, values: GPUKVAllocationSnapshot
    func sameAllocation(as other: Self) -> Bool {
        keys.sameAllocation(as: other.keys) && values.sameAllocation(as: other.values)
    }
    var json: [String: Any] { ["keys": keys.json, "values": values.json] }
}

final class GPUKVAllocationDiagnostics {
    let libraryPath: String
    private typealias Snapshot = @convention(c) (mlx_array, UnsafeMutablePointer<UInt64>?, Int) -> Int32
    private let handle: UnsafeMutableRawPointer
    private let snapshotNative: Snapshot
    private let lastError: @convention(c) () -> UnsafePointer<CChar>?

    init(libraryPath: String) throws {
        guard libraryPath.hasPrefix("/") else { throw GPUError.invalid("Allocation diagnostics needs an absolute library path") }
        let path = URL(fileURLWithPath: libraryPath).resolvingSymlinksInPath().path
        guard let handle = dlopen(path, RTLD_NOW | RTLD_LOCAL) else {
            let reason = dlerror().map { String(cString: $0) } ?? "unknown loader error"
            throw GPUError.invalid("Cannot load allocation diagnostics: \(reason)")
        }
        guard let version = dlsym(handle, "anemlx_kv_allocation_diagnostics_version"),
              let snapshot = dlsym(handle, "anemlx_kv_allocation_snapshot"),
              let error = dlsym(handle, "anemlx_kv_allocation_diagnostics_last_error") else {
            dlclose(handle); throw GPUError.invalid("Incomplete allocation diagnostics ABI")
        }
        guard unsafeBitCast(version, to: (@convention(c) () -> Int32).self)() == 1 else {
            dlclose(handle); throw GPUError.invalid("Allocation diagnostics requires ABI 1")
        }
        self.handle = handle; self.libraryPath = path
        snapshotNative = unsafeBitCast(snapshot, to: Snapshot.self)
        lastError = unsafeBitCast(error, to: (@convention(c) () -> UnsafePointer<CChar>?).self)
    }
    // The bridge creates no graph/primitives and has no asynchronous callbacks.
    // Consequently no lazy array vtable depends on keeping this dylib loaded.
    deinit { dlclose(handle) }

    func snapshot(_ tensor: Tensor) throws -> GPUKVAllocationSnapshot {
        var fields = [UInt64](repeating: 0, count: 8)
        let status = fields.withUnsafeMutableBufferPointer {
            snapshotNative(tensor.handle, $0.baseAddress, $0.count)
        }
        guard status == 0 else {
            let reason = lastError().map { String(cString: $0) } ?? "status \(status)"
            throw GPUError.invalid("Allocation diagnostic: \(reason)")
        }
        guard fields[0] != 0, fields[5] <= 1, fields[6] <= 1, fields[7] <= 1 else {
            throw GPUError.invalid("Malformed allocation diagnostics result")
        }
        return GPUKVAllocationSnapshot(metalBuffer: fields[0], offsetBytes: fields[1],
            allocationBytes: fields[2], logicalBytes: fields[3], dataElements: fields[4],
            contiguous: fields[5] == 1, rowContiguous: fields[6] == 1,
            donatableAtObservation: fields[7] == 1)
    }
}
