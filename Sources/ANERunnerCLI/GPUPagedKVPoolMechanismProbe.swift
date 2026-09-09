import ANERunnerGPU
import Darwin
import Foundation

extension RunnerCLI {
    /// Isolated, model-free physical page sharing and lifetime validation.
    static func probeGPUPagedKVPoolMechanism(_ args: Arguments) throws {
        try args.validate(["--library", "--eval", "--output"])
        let mode = args["--eval"] ?? "sync"
        guard ["sync", "async"].contains(mode) else { throw CLIError.usage("Use --eval sync or async") }
        let output = URL(fileURLWithPath: try args.require("--output")).standardizedFileURL
        // Create exclusively, including a no-follow leaf. No existing result
        // is silently replaced if an external controller reuses its filename.
        let descriptor = open(output.path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, mode_t(0o600))
        guard descriptor >= 0 else { throw CLIError.usage("Use a new writable output path for page pool NDJSON") }
        let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
        defer { try? handle.close() }
        let before = try MoEGateUpProbeSupport.baseMLX()
        var provenance = try JSONSerialization.data(withJSONObject: ["event": "cli_provenance",
            "base_mlx": before, "executable_sha256": try GPUProbeSupport.hash(
                URL(fileURLWithPath: CommandLine.arguments[0])),
            "pool_library_sha256": try GPUProbeSupport.hash(
                URL(fileURLWithPath: args.require("--library"))),
            "operating_system": ProcessInfo.processInfo.operatingSystemVersionString], options: [.sortedKeys])
        provenance.append(0x0a)
        var checkedPluginImage = false
        try GPUPagedKVPoolMechanismProbe.run(libraryPath: args.require("--library"),
            asynchronous: mode == "async") { data in
                // First event occurs after dlopen, before any GPU dispatch.
                // Check this again so a plugin cannot silently load a second MLX.
                if !checkedPluginImage {
                    _ = try MoEGateUpProbeSupport.baseMLX()
                    try handle.write(contentsOf: provenance)
                    checkedPluginImage = true
                }
                try handle.write(contentsOf: data)
            }
    }
}
