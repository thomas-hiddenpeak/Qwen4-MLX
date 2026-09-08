import CryptoKit
import Darwin
import Foundation

/// Local immutable-checkpoint identity. Small model/config/runtime files are
/// hashed; large weights and n-gram data bind canonical path, device/inode,
/// size and nanosecond mtime/ctime. Replaced or modified files therefore miss.
/// This is a local cache identity, not a portable cryptographic weight digest.
public enum QwenPrefixCacheIdentity {
    public static func fingerprint(modelDirectory: URL) throws -> String {
        let directory = modelDirectory.standardizedFileURL.resolvingSymlinksInPath()
        let files = try FileManager.default.contentsOfDirectory(at: directory,
            includingPropertiesForKeys: nil).filter {
                ["safetensors", "json", "jinja", "bin"].contains($0.pathExtension)
            }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else { throw GPUError.invalid("No model files for prefix cache identity") }
        var hash = SHA256()
        func field(_ text: String) { hash.update(data: Data((text + "\n").utf8)) }
        field("qwen-prefix-archive-local-v1"); field(directory.path)
        for file in files {
            let resolved = file.resolvingSymlinksInPath()
            var info = stat()
            guard resolved.path.withCString({ Darwin.lstat($0, &info) }) == 0,
                  (info.st_mode & S_IFMT) == S_IFREG else {
                throw GPUError.invalid("Cannot identify prefix cache model file")
            }
            field("\(file.lastPathComponent)|\(resolved.path)|\(info.st_dev)|\(info.st_ino)|\(info.st_size)|\(info.st_mtimespec.tv_sec):\(info.st_mtimespec.tv_nsec)|\(info.st_ctimespec.tv_sec):\(info.st_ctimespec.tv_nsec)")
            if ["json", "jinja"].contains(file.pathExtension) {
                hash.update(data: try Data(contentsOf: resolved))
            }
        }
        guard let executable = Bundle.main.executableURL else { throw GPUError.invalid("Missing runtime identity") }
        hash.update(data: try Data(contentsOf: executable))
        var libraries = Set<String>()
        for i in 0..<_dyld_image_count() {
            if let name = _dyld_get_image_name(i) {
                let path = String(cString: name)
                if path.contains("mlx") && path.hasSuffix(".dylib") { libraries.insert(path) }
            }
        }
        for key in ["ANERUNNER_GATEUP_LIBRARY", "ANERUNNER_MOE_DOWN_PAIR_LIBRARY"] {
            if let path = ProcessInfo.processInfo.environment[key], !path.isEmpty { libraries.insert(path) }
        }
        for path in libraries.sorted() { field(path); hash.update(data: try Data(contentsOf: URL(fileURLWithPath: path))) }
        for key in ["ANERUNNER_BLOCKED_GDN", "ANERUNNER_FUSED_PREFILL", "ANERUNNER_EXPERIMENTAL_DECODE_ASYNC_LAYERS",
                    "MLX_MAX_MB_PER_BUFFER", "MLX_MAX_OPS_PER_BUFFER"] {
            field("\(key)=\(ProcessInfo.processInfo.environment[key] ?? "default")")
        }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
