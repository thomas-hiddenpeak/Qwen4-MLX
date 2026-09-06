import ANERunnerGPU
import Darwin
import Foundation

/// A saved, explicitly selected prefill configuration and its checked native
/// dependencies. Loading this selection never changes process environment.
struct GPUMoEPrefillSelection {
    let configuration: GPUMoEPrefillConfiguration
    let plugin: GPUMoEPrefillGateUp
    let effectiveJSON: Any
    let provenance: [String: Any]

    init(path: String, modelDirectory: URL, accumulation: GPUMoE.PrefillAccumulation) throws {
        guard accumulation == .reference else {
            throw CLIError.usage("--prefill-moe-config requires --prefill-accumulation reference")
        }
        let configURL = URL(fileURLWithPath: path).standardizedFileURL.resolvingSymlinksInPath()
        let data = try Data(contentsOf: configURL)
        struct Saved: Decodable {
            let model_directory: String
            let gateup_plugin_sha256: String
            let base_mlx_sha256: String
            let native_configuration: Int?
            let status: String?
        }
        let saved = try JSONDecoder().decode(Saved.self, from: data)
        let decoded = try JSONDecoder().decode(GPUMoEPrefillConfiguration.self, from: data)
        try decoded.validated()
        guard saved.native_configuration == nil || saved.native_configuration == 0 else {
            throw CLIError.usage("--prefill-moe-config cannot apply legacy nonzero native_configuration; this entry selects request-local prefill kernels only")
        }
        let modelURL = modelDirectory.standardizedFileURL.resolvingSymlinksInPath()
        guard saved.model_directory.hasPrefix("/"),
              URL(fileURLWithPath: saved.model_directory).standardizedFileURL.resolvingSymlinksInPath() == modelURL else {
            throw CLIError.usage("Saved prefill MoE configuration belongs to a different model directory")
        }
        for hash in [saved.gateup_plugin_sha256, saved.base_mlx_sha256] {
            guard hash.count == 64, hash.allSatisfy({ $0.isHexDigit && $0.isASCII }) else {
                throw CLIError.usage("Saved prefill MoE configuration requires valid plugin and base MLX SHA256 values")
            }
        }
        try MoEGateUpProbeSupport.requireStockSelectors()
        guard let library = ProcessInfo.processInfo.environment["ANERUNNER_GATEUP_LIBRARY"], library.hasPrefix("/") else {
            throw CLIError.usage("--prefill-moe-config requires absolute ANERUNNER_GATEUP_LIBRARY; the command does not set it automatically")
        }
        let pluginURL = URL(fileURLWithPath: library).standardizedFileURL.resolvingSymlinksInPath()
        let pluginHash = try MoETilingBytes.hash(pluginURL)
        guard pluginHash.lowercased() == saved.gateup_plugin_sha256.lowercased() else {
            throw CLIError.usage("ANERUNNER_GATEUP_LIBRARY SHA256 differs from the saved prefill configuration")
        }
        let baseBeforeLoad = try MoEGateUpProbeSupport.baseMLX()
        guard baseBeforeLoad["loaded_sha256"]?.lowercased() == saved.base_mlx_sha256.lowercased() else {
            throw CLIError.usage("Loaded stock MLX SHA256 differs from the saved prefill configuration")
        }
        let loaded = try GPUMoEPrefillGateUp(variant: decoded.gateUpVariant ?? 0)
        guard URL(fileURLWithPath: loaded.libraryPath).standardizedFileURL.resolvingSymlinksInPath() == pluginURL else {
            throw CLIError.usage("Loaded gate/up plugin path differs from the checked library")
        }
        // Verify actual symbol ownership, not just the path requested by dlopen.
        guard let handle = dlopen(pluginURL.path, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD) else {
            throw CLIError.usage("Checked gate/up plugin is not present in the loaded process")
        }
        defer { dlclose(handle) } // The GPU wrapper retains its separate pinned reference.
        var symbols = ["anemlx_moe_gateup_version", "anemlx_moe_gateup", "anemlx_moe_gateup_last_error", "anemlx_moe_gateup_dispatch_count"]
        if loaded.abiVersion == 2 {
            symbols += ["anemlx_moe_expert_plan", "anemlx_moe_gateup_planned", "anemlx_moe_grouped_down",
                        "anemlx_moe_expert_plan_dispatch_count", "anemlx_moe_grouped_down_dispatch_count"]
        }
        for name in symbols {
            var info = Dl_info()
            guard let symbol = dlsym(handle, name), dladdr(symbol, &info) != 0, let owner = info.dli_fname,
                  URL(fileURLWithPath: String(cString: owner)).standardizedFileURL.resolvingSymlinksInPath() == pluginURL else {
                throw CLIError.usage("Gate/up symbol \(name) does not belong to the checked plugin")
            }
        }
        let base = try MoEGateUpProbeSupport.baseMLX()
        guard base["loaded_path"] == baseBeforeLoad["loaded_path"],
              base["loaded_sha256"] == baseBeforeLoad["loaded_sha256"],
              try MoETilingBytes.hash(pluginURL) == pluginHash else {
            throw CLIError.usage("Native library identity changed while loading the prefill selection")
        }
        let decodedJSON = try JSONSerialization.jsonObject(with: JSONEncoder().encode(decoded))
        configuration = decoded; plugin = loaded
        effectiveJSON = decodedJSON
        provenance = [
            "enabled": true, "configuration_path": configURL.path,
            "configuration_sha256": MoETilingBytes.digest(data), "effective_configuration": decodedJSON,
            "model_directory": modelURL.path,
            "saved_status": saved.status.map { $0 as Any } ?? NSNull(),
            "plugin": ["path": pluginURL.path, "sha256": pluginHash, "abi_version": loaded.abiVersion,
                       "symbol_ownership_verified": true, "matches_saved_configuration": true],
            "base_mlx": base, "environment_mutated": false,
            "scope": "Request-local prefill only. Grouped variants apply to chunks of 205...512 tokens; other chunk lengths retain their configured reference paths. Saved status is provenance, not a new validation claim."
        ]
    }

    struct Counts: Equatable {
        var host: [Int] // reduction, gate/up, grouped down
        var gateUp: [UInt64] // ABI 1: variants 0/1; ABI 2: variants 0...3
        var grouped: [UInt64] // plan32, plan16, down32, down16

        func subtracting(_ before: Counts) throws -> Counts {
            guard host.count == before.host.count, gateUp.count == before.gateUp.count,
                  grouped.count == before.grouped.count,
                  zip(host, before.host).allSatisfy({ $0.0 >= $0.1 }),
                  zip(gateUp, before.gateUp).allSatisfy({ $0.0 >= $0.1 }),
                  zip(grouped, before.grouped).allSatisfy({ $0.0 >= $0.1 }) else {
                throw CLIError.usage("Prefill MoE counters changed shape or decreased during a request")
            }
            return Counts(host: zip(host, before.host).map { $0.0 - $0.1 },
                          gateUp: zip(gateUp, before.gateUp).map { $0.0 - $0.1 },
                          grouped: zip(grouped, before.grouped).map { $0.0 - $0.1 })
        }

        var isZero: Bool { host.allSatisfy { $0 == 0 } && gateUp.allSatisfy { $0 == 0 } && grouped.allSatisfy { $0 == 0 } }
        var report: [String: Any] {
            ["host": ["reduction": host[0], "gate_up": host[1], "grouped_down": host[2]],
             "native_gate_up_by_variant": gateUp.isEmpty ? NSNull() : gateUp as Any,
             "native_grouped_plan32_plan16_down32_down16": grouped.isEmpty ? NSNull() : grouped as Any]
        }
    }

    static func snapshot(model: QwenModel, selection: Self?) -> Counts {
        Counts(host: [model.prefillMoEReductionCalls, model.prefillMoEGateUpCalls, model.prefillMoEGroupedDownCalls],
               gateUp: selection?.plugin.dispatchCounts() ?? [], grouped: selection?.plugin.groupedDispatchCounts() ?? [])
    }

    static func expectedPrefill(selection: Self?, promptTokens: Int, chunk: Int, layers: Int) -> Counts {
        var counts = Counts(host: [0, 0, 0],
                            gateUp: selection.map { Array(repeating: 0, count: $0.plugin.abiVersion == 2 ? 4 : 2) } ?? [],
                            grouped: selection == nil ? [] : [0, 0, 0, 0])
        guard let configuration = selection?.configuration else { return counts }
        var offset = 0
        while offset < promptTokens {
            let end = offset < promptTokens - 1 ? min(promptTokens - 1, offset + chunk) : promptTokens
            let length = end - offset
            if configuration.threadgroupSize(tokenCount: length) != nil { counts.host[0] += layers }
            if let variant = configuration.effectiveGateUpVariant(tokenCount: length) {
                counts.host[1] += layers; counts.gateUp[variant] += UInt64(layers)
                if variant >= 2 { counts.grouped[variant - 2] += UInt64(layers) }
                if configuration.usesGroupedDown(tokenCount: length) {
                    counts.host[2] += layers; counts.grouped[variant] += UInt64(layers)
                }
            }
            offset = end
        }
        return counts
    }
}
