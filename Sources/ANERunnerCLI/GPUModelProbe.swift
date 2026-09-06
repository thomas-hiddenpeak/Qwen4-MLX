import ANERunnerGPU
import CMLX
import CryptoKit
import Dispatch
import Foundation

extension RunnerCLI {
    /// Captures actual boundaries of a limited real decoder, without sampling.
    static func probeGPUModel(_ args: Arguments) throws {
        try args.validate(["--model-dir", "--tokens-file", "--decode-tokens-file", "--layers", "--prefill-chunk", "--capture-output", "--output", "--decode-mode"])
        guard let decodeMode = GPUDecodeMode(rawValue: args["--decode-mode"] ?? "reference") else { throw CLIError.usage("Invalid --decode-mode") }
        guard let layerLimit = Int(args["--layers"] ?? "1"), (1...4).contains(layerLimit),
              let chunkSize = Int(args["--prefill-chunk"] ?? "64"), (1...512).contains(chunkSize) else {
            throw CLIError.usage("probe-gpu-model requires layers in 1...4 and prefill-chunk in 1...512")
        }
        let directory = URL(fileURLWithPath: try args.require("--model-dir"), isDirectory: true).standardizedFileURL.resolvingSymlinksInPath()
        let promptURL = URL(fileURLWithPath: try args.require("--tokens-file")).standardizedFileURL
        let decodeURL = args["--decode-tokens-file"].map { URL(fileURLWithPath: $0).standardizedFileURL }
        let destination = URL(fileURLWithPath: try args.require("--capture-output")).standardizedFileURL
        guard destination.pathExtension == "safetensors", !FileManager.default.fileExists(atPath: destination.path) else {
            throw CLIError.usage("--capture-output must be a new .safetensors file")
        }
        if let summaryPath = args["--output"], URL(fileURLWithPath: summaryPath).standardizedFileURL == destination {
            throw CLIError.usage("Summary JSON and capture safetensors need different paths")
        }
        let promptData = try Data(contentsOf: promptURL)
        let decodeData = try decodeURL.map { try Data(contentsOf: $0) }
        let tokens = try JSONDecoder().decode([Int32].self, from: promptData)
        let decode = try decodeData.map { try JSONDecoder().decode([Int32].self, from: $0) } ?? []
        let configuration = try QwenConfiguration(modelDirectory: directory)
        guard !tokens.isEmpty, tokens.count <= 4096, decode.count <= 4096 - tokens.count,
              (tokens + decode).allSatisfy({ $0 >= 0 && $0 < configuration.vocabularySize }) else {
            throw CLIError.usage("Capture requires 1...4096 valid total token IDs (prefill plus fixed decode suffix)")
        }
        // Diagnostic traces keep several full HC streams per layer. Reject an
        // oversized capture before allocating it; this is not a long-context bench.
        let captureLimit = 512 * 1024 * 1024
        let perTokenEstimate = configuration.hiddenSize * 2 * (8 + layerLimit * 18)
        guard tokens.count + decode.count <= captureLimit / perTokenEstimate else {
            throw CLIError.usage("Requested trace exceeds the 512 MiB diagnostic budget; shorten token files")
        }
        try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
        let temporary = destination.deletingLastPathComponent().appendingPathComponent(".gpu-capture-" + UUID().uuidString + ".safetensors")
        defer { try? FileManager.default.removeItem(at: temporary) }
        try MX.check(0, "initialize probe")
        var previousCacheLimit = 0
        try MX.check(mlx_set_cache_limit(&previousCacheLimit, 128 * 1024 * 1024), "bound probe allocation cache")
        defer { var ignored = 0; _ = mlx_set_cache_limit(&ignored, previousCacheLimit) }
        let beforeLoad = try MX.memory()
        let loadStart = DispatchTime.now().uptimeNanoseconds
        let model = try QwenModel(modelDirectory: directory, layerLimit: layerLimit, decodeModes: [decodeMode])
        let loadMilliseconds = GPUModelCapture.elapsed(loadStart)
        let afterLoad = try MX.memory()
        var state = model.makeState()
        let arrays = mlx_map_string_to_array_new()
        let metadata = mlx_map_string_to_string_new()
        defer { _ = mlx_map_string_to_array_free(arrays); _ = mlx_map_string_to_string_free(metadata) }
        var logicalCaptureBytes = 0
        var tensorDescriptions: [String: [String: Any]] = [:]
        var steps: [[String: Any]] = []

        func record(_ name: String, _ tensor: Tensor) throws {
            guard tensorDescriptions[name] == nil, tensor.nbytes <= captureLimit - logicalCaptureBytes else {
                throw GPUError.invalid("Duplicate tensor or exceeded capture byte budget: \(name)")
            }
            try MX.check(name.withCString { mlx_map_string_to_array_insert(arrays, $0, tensor.handle) }, "insert capture \(name)")
            logicalCaptureBytes += tensor.nbytes
            tensorDescriptions[name] = ["shape": tensor.shape, "dtype": GPUModelCapture.dtype(tensor.dtype), "bytes": tensor.nbytes]
        }
        func step(_ ids: [Int32], phase: String, index: Int) throws {
            let prefix = "\(phase).\(index)", before = state.offset
            let start = DispatchTime.now().uptimeNanoseconds
            let output = try model.forward(tokens: ids, state: &state, captureTrace: true, decodeMode: decodeMode)
            // Evaluation is mandatory before advancing state or saving handles:
            // trace tensors and recurrent state must describe the same forward.
            try MX.eval([output.stream] + state.tensors + Array(output.trace.values))
            let milliseconds = GPUModelCapture.elapsed(start)
            guard state.offset == before + ids.count, state.valid, output.logits == nil else {
                throw GPUError.invalid("Unexpected state/logits from layer-limited capture")
            }
            for name in output.trace.keys.sorted() { try record(prefix + "." + name, output.trace[name]!) }
            try record(prefix + ".token_ids", MX.array(ids, shape: [1, ids.count]))
            try record(prefix + ".sequence_offset", MX.scalar(Float(before), MLX_INT32))
            try record(prefix + ".sequence_offset_after", MX.scalar(Float(state.offset), MLX_INT32))
            steps.append([
                "prefix": prefix, "phase": phase, "token_ids": ids, "sequence_offset": before,
                "sequence_offset_after": state.offset, "forward_and_eval_milliseconds": milliseconds,
                "ssd_wait_seconds": output.ssdWaitSeconds, "ssd_logical_row_bytes": output.ssdLogicalBytes,
                "trace_names": output.trace.keys.sorted(), "state_tensor_count": state.tensors.count,
            ])
        }
        var offset = 0, chunkIndex = 0
        while offset < tokens.count {
            let end = min(tokens.count, offset + chunkSize)
            try step(Array(tokens[offset..<end]), phase: "prefill", index: chunkIndex)
            offset = end; chunkIndex += 1
        }
        for (index, id) in decode.enumerated() { try step([id], phase: "decode", index: index) }
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let indexData = try Data(contentsOf: directory.appendingPathComponent("model.safetensors.index.json"))
        let provenance: [String: String] = [
            "schema": "independent-qwen-gpu-boundaries-v1", "model_directory": directory.path,
            "layer_count": String(layerLimit), "mtp_enabled": "false", "sampling": "none; fixed token replay",
            "decode_mode": decodeMode.rawValue,
            "config_sha256": GPUModelCapture.digest(configData), "weight_index_sha256": GPUModelCapture.digest(indexData),
            "prefill_token_file_sha256": GPUModelCapture.digest(promptData),
            "decode_token_file_sha256": decodeData.map(GPUModelCapture.digest) ?? "absent",
            "state_semantics": "sequence_offset is before this chunk; decode inputs are supplied tokens, not generated tokens",
        ]
        for (key, value) in provenance {
            try MX.check(key.withCString { k in value.withCString { mlx_map_string_to_string_insert(metadata, k, $0) } }, "capture metadata")
        }
        let saveStart = DispatchTime.now().uptimeNanoseconds
        // The C save API is CPU file IO and has no stream parameter. All graph
        // outputs were evaluated above; BF16 and integer dtypes remain native.
        try MX.check(temporary.path.withCString { mlx_save_safetensors($0, arrays, metadata) }, "save capture safetensors")
        try FileManager.default.moveItem(at: temporary, to: destination)
        let saveMilliseconds = GPUModelCapture.elapsed(saveStart)
        let ledger = try JSONSerialization.jsonObject(with: JSONEncoder().encode(model.weights.ledger))
        try emit([
            "schema_version": 1, "runner": "independent-swift-mlx-c", "mode": "layer-limited fixed-token boundary capture",
            "model_directory": directory.path, "layers": layerLimit, "prefill_chunk": chunkSize,
            "decode_mode": decodeMode.rawValue, "additional_projection_buffer_bytes": model.additionalProjectionBufferBytes,
            "mtp_enabled": false, "full_model_generation": false, "final_state_offset": state.offset,
            "prefill_token_ids": tokens, "decode_token_ids": decode, "steps": steps,
            "capture_file": destination.path, "capture_logical_bytes": logicalCaptureBytes,
            "capture_limit_bytes": captureLimit, "tensors": tensorDescriptions, "provenance": provenance,
            "load_milliseconds": loadMilliseconds, "save_milliseconds": saveMilliseconds,
            "loaded_weight_source_bytes": model.weights.cachedSourceBytes, "weight_ledger": ledger,
            "memory": ["before_load": beforeLoad, "after_load": afterLoad, "after_capture": try MX.memory()],
            "limitation": "Only actual captured layer boundaries. Times include diagnostic graph materialization and are not generation benchmarks. No reference comparison or correctness pass is implied by capture completion.",
        ], to: args["--output"])
    }
}

private enum GPUModelCapture {
    static func elapsed(_ start: UInt64) -> Double { Double(DispatchTime.now().uptimeNanoseconds - start) * 1e-6 }
    static func digest(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
    static func dtype(_ dtype: mlx_dtype) -> String {
        switch dtype {
        case MLX_BFLOAT16: return "BF16"
        case MLX_FLOAT16: return "F16"
        case MLX_FLOAT32: return "F32"
        case MLX_UINT32: return "U32"
        case MLX_INT32: return "I32"
        case MLX_BOOL: return "BOOL"
        default: return "MLX_\(dtype.rawValue)"
        }
    }
}
