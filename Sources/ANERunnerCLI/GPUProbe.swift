import ANERunnerCore
import ANERunnerGPU
import CMLX
import CryptoKit
import Dispatch
import Foundation

extension RunnerCLI {
    static func probeGPUMoE(_ args: Arguments) throws {
        try args.validate(["--model-dir", "--fixture", "--layer", "--warmups", "--runs", "--output", "--prefill-accumulation"])
        guard let layer = Int(args["--layer"] ?? "0"), layer >= 0,
              let warmups = Int(args["--warmups"] ?? "3"), (0...100).contains(warmups),
              let runs = Int(args["--runs"] ?? "10"), (1...100).contains(runs) else {
            throw CLIError.usage("Require layer>=0, warmups in 0...100, and runs in 1...100")
        }
        guard let prefillAccumulation = GPUMoE.PrefillAccumulation(rawValue: args["--prefill-accumulation"] ?? "reference") else {
            throw CLIError.usage("--prefill-accumulation must be reference or float32")
        }
        let modelURL = URL(fileURLWithPath: try args.require("--model-dir"), isDirectory: true)
            .standardizedFileURL.resolvingSymlinksInPath()
        let fixtureURL = URL(fileURLWithPath: try args.require("--fixture"))
            .standardizedFileURL.resolvingSymlinksInPath()
        let configuration = try QwenConfiguration(modelDirectory: modelURL)
        guard layer < configuration.layerCount else { throw CLIError.usage("Requested layer is outside this model") }
        let fixtureStart = DispatchTime.now().uptimeNanoseconds
        let fixture = try CoreMLBlockFixture.load(from: fixtureURL)
        guard fixture.inputs.count == 1, let input = fixture.inputs["x"],
              input.shape.count == 3, input.shape[0] == 1, input.shape[1] > 0,
              input.shape[2] == configuration.hiddenSize,
              input.dtype == .float16 || input.dtype == .float32,
              input.values.allSatisfy({ Float($0).isFinite }) else {
            throw CLIError.usage("GPU MoE fixture must contain one finite floating x tensor with shape [1,T,hiddenSize]")
        }
        let fixtureLoadMilliseconds = GPUProbeSupport.elapsed(fixtureStart)
        let memoryBeforeLoading = try MX.memory()
        let loadStart = DispatchTime.now().uptimeNanoseconds
        let weights = try GPUWeights(modelDirectory: modelURL)
        let moe = try GPUMoE(
            weights: weights, layer: layer, hiddenSize: configuration.hiddenSize,
            experts: configuration.expertCount, topK: configuration.expertsPerToken,
            groupSize: configuration.quantizationGroupSize, bits: configuration.quantizationBits,
            prefillAccumulation: prefillAccumulation
        )
        try MX.synchronize()
        let modelLoadMilliseconds = GPUProbeSupport.elapsed(loadStart)
        let memoryAfterLoading = try MX.memory()
        let preparationStart = DispatchTime.now().uptimeNanoseconds
        let x = try MX.array(input.values.map(Float.init), shape: input.shape, dtype: MLX_BFLOAT16)
        try x.eval()
        let inputPreparationMilliseconds = GPUProbeSupport.elapsed(preparationStart)

        // Each sample includes Swift graph construction, dispatch, and waiting
        // for the final GPU output. No output tensor is copied to the CPU here.
        func iteration() throws -> Double {
            let start = DispatchTime.now().uptimeNanoseconds
            let result = try moe.forward(x, diagnostics: false)
            try result.y.eval()
            return GPUProbeSupport.elapsed(start)
        }
        let firstMilliseconds = try iteration()
        var warmupMilliseconds: [Double] = []
        for _ in 0..<warmups { warmupMilliseconds.append(try iteration()) }
        var samples: [Double] = []
        for _ in 0..<runs { samples.append(try iteration()) }
        let memoryAfterTiming = try MX.memory()

        // Diagnostics are an independent forward after the timed series.
        let diagnosticStart = DispatchTime.now().uptimeNanoseconds
        let diagnostic = try moe.forward(x, diagnostics: true)
        try MX.eval(Array(diagnostic.diagnostics.values))
        let diagnosticPredictionMilliseconds = GPUProbeSupport.elapsed(diagnosticStart)
        let hostStart = DispatchTime.now().uptimeNanoseconds
        var tensors = diagnostic.diagnostics
        tensors["y"] = diagnostic.y
        var outputs: [String: CoreMLTensor] = [:]
        for name in tensors.keys.sorted() {
            let tensor = tensors[name]!
            if name == "selected_experts" {
                outputs[name] = CoreMLTensor(shape: tensor.shape, dtype: .int32,
                                            values: try tensor.ints().map(Double.init))
            } else {
                let values = try tensor.floats().map(Double.init)
                guard values.allSatisfy(\.isFinite) else {
                    throw GPUError.invalid("Nonfinite GPU diagnostic output: \(name)")
                }
                outputs[name] = CoreMLTensor(shape: tensor.shape, dtype: .float32, values: values)
            }
        }
        let hostMaterializationMilliseconds = GPUProbeSupport.elapsed(hostStart)
        let memoryAfterDiagnostics = try MX.memory()
        guard let actualIDs = outputs["selected_experts"], let actualWeights = outputs["routing_weights"],
              actualIDs.shape == [1, input.shape[1], configuration.expertsPerToken] else {
            throw GPUError.invalid("MoE returned invalid routing diagnostic dimensions")
        }
        let ids = actualIDs.values.map(Int.init)
        guard ids.allSatisfy({ (0..<configuration.expertCount).contains($0) }) else {
            throw GPUError.invalid("MoE selected an expert outside the full bank")
        }
        let uniqueIDs = Set(ids).sorted()
        let routeComparison = try GPUProbeSupport.compareRouting(
            actualIDs: actualIDs, actualWeights: actualWeights,
            expectedIDs: fixture.expectedOutputs?["selected_experts"],
            expectedWeights: fixture.expectedOutputs?["routing_weights"]
        )
        var comparisons: [String: Any] = [:]
        for (name, expected) in fixture.expectedOutputs ?? [:] {
            if let actual = outputs[name] {
                comparisons[name] = try GPUProbeSupport.comparison(actual, expected)
            }
        }
        let median = GPUProbeSupport.percentile(samples, 0.5)
        let byteLedger = try GPUProbeSupport.logicalBytes(
            weights: weights, layer: layer, experts: configuration.expertCount,
            topK: configuration.expertsPerToken, tokenCount: input.shape[1],
            actualIDs: ids, medianMilliseconds: median
        )

        // Read and hash only this layer's loaded tensor slices, after all timing
        // and diagnostic work. No unrelated decoder, vision, MTP or n-gram data
        // is hashed. This is source-file provenance, not a GPU-buffer checksum.
        let sourceHashStart = DispatchTime.now().uptimeNanoseconds
        var sourceSlices: [[String: Any]] = []
        for entry in weights.ledger {
            let metadata = try weights.metadata(entry.name)
            let url = modelURL.appendingPathComponent(metadata.shard)
            sourceSlices.append([
                "name": metadata.name, "shard": metadata.shard, "path": url.path,
                "shape": metadata.shape, "dtype": metadata.dtypeName,
                "byteOffset": metadata.byteOffset, "byteCount": metadata.byteCount,
                "sha256": try GPUProbeSupport.hash(url, offset: metadata.byteOffset, length: metadata.byteCount),
                "loadCount": entry.loadCount, "cachedByWeightLoader": entry.cached,
            ])
        }
        let configURL = modelURL.appendingPathComponent("config.json")
        let indexURL = modelURL.appendingPathComponent("model.safetensors.index.json")
        let source: [String: Any] = [
            "modelDirectory": modelURL.path, "fixturePath": fixtureURL.path,
            "fixtureSHA256": try GPUProbeSupport.hash(fixtureURL),
            "configSHA256": try GPUProbeSupport.hash(configURL),
            "weightIndexSHA256": try GPUProbeSupport.hash(indexURL),
            "loadedTensorSlices": sourceSlices,
            "loadedTensorSourceDigestSHA256": GPUProbeSupport.digest(
                try JSONSerialization.data(withJSONObject: sourceSlices, options: [.sortedKeys])
            ),
            "hashScope": "Current source file bytes for this layer's loaded tensors, plus config/index/fixture. Collected after timed inference; not a full-model hash or GPU-memory checksum.",
        ]
        let sourceHashingMilliseconds = GPUProbeSupport.elapsed(sourceHashStart)
        let outputObject = try JSONSerialization.jsonObject(with: JSONEncoder().encode(outputs))
        let report: [String: Any] = [
            "schemaVersion": 1, "backend": "swift-mlx-c-gpu-affine-moe",
            "layerIndex": layer, "tokenCount": input.shape[1], "hiddenSize": configuration.hiddenSize,
            "expertCount": configuration.expertCount, "topK": configuration.expertsPerToken,
            "mtpEnabled": false, "fullModelGeneration": false,
            "activationPrecision": "bfloat16; JSON outputs expanded to float32",
            "weightPrecision": "original packed affine Q4; BF16 scales/biases and BF16 router/shared weights",
            "decodeImplementation": moe.decodeImplementation,
            "routingImplementation": moe.routingImplementation,
            "prefillAccumulation": moe.prefillAccumulation.rawValue,
            "prefillReductionImplementation": moe.prefillReductionImplementation,
            "accuracyGateSemantics": "Source-capture agreement and the separate FP32 relative-L2 threshold are independent checks. Selecting reference does not assert that the FP32 threshold passes; selecting float32 intentionally changes prefill accumulation and may change generated tokens.",
            "groupSize": configuration.quantizationGroupSize, "bits": configuration.quantizationBits,
            "selectedExpertIDs": uniqueIDs,
            "routingExpertSetsMatch": routeComparison.setsMatch as Any? ?? NSNull(),
            "routingWeightComparisonByExpert": routeComparison.alignedWeights as Any? ?? NSNull(),
            "outputs": outputObject, "comparisons": comparisons,
            "modelLoadMilliseconds": modelLoadMilliseconds,
            "fixtureLoadMilliseconds": fixtureLoadMilliseconds,
            "inputPreparationMilliseconds": inputPreparationMilliseconds,
            "firstIterationMilliseconds": firstMilliseconds,
            "warmups": warmups, "warmupMilliseconds": warmupMilliseconds,
            "runs": runs, "predictionMilliseconds": samples,
            "medianMilliseconds": median,
            "p10Milliseconds": GPUProbeSupport.percentile(samples, 0.1),
            "p90Milliseconds": GPUProbeSupport.percentile(samples, 0.9),
            "diagnosticPredictionMilliseconds": diagnosticPredictionMilliseconds,
            "outputMaterializationMilliseconds": hostMaterializationMilliseconds,
            "sourceHashingMilliseconds": sourceHashingMilliseconds,
            "predictionTimingSemantics": "Resident BF16 input and loaded packed weights; Swift graph construction + MLX GPU dispatch + final y.eval synchronization. No tensor host materialization, source hashing, fixture parsing or model loading in samples. Diagnostic forward and host copies are separate.",
            "mlxMemory": ["beforeLoading": memoryBeforeLoading, "afterLoading": memoryAfterLoading,
                          "afterTiming": memoryAfterTiming, "afterDiagnostics": memoryAfterDiagnostics],
            "loadedWeightSourceBytes": weights.cachedSourceBytes,
            "logicalByteLedger": byteLedger,
            "physicalDRAMReadBytes": NSNull(), "physicalDRAMBandwidthGBps": NSNull(),
            "physicalBandwidthEvidence": "No GPU DRAM transaction counter was collected. Logical source bytes divided by wall time are not physical memory bandwidth; cache reuse and actual transfer traffic are unknown.",
            "source": source, "mlxVersion": try GPUProbeSupport.mlxVersion(),
            "operatingSystem": ProcessInfo.processInfo.operatingSystemVersionString,
            "hardwareEvidence": "Actual MLX calls use the public GPU stream. No Metal performance-counter trace or ANE execution is claimed.",
            "limitations": "One complete MoE block over supplied real token rows. Routing and affine-Q4 decode use upstream fused Metal kernels when geometry permits. Prefill retains public gather_qmm; its explicit accumulation mode is reported separately. Local numerical and latency results do not establish whole-model quality or generation speed.",
        ]
        try emit(report, to: args["--output"])
    }
}

enum GPUProbeSupport {
    static func elapsed(_ start: UInt64) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000
    }

    static func percentile(_ values: [Double], _ fraction: Double) -> Double {
        let sorted = values.sorted()
        let position = Double(sorted.count - 1) * fraction
        let lower = Int(position.rounded(.down)), upper = Int(position.rounded(.up))
        return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - Double(lower))
    }

    static func digest(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    static func hash(_ url: URL, offset: UInt64 = 0, length: UInt64? = nil) throws -> String {
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        let size = try file.seekToEnd()
        guard offset <= size else { throw GPUError.invalid("Source hash offset exceeds file: \(url.path)") }
        var remaining = length ?? (size - offset)
        guard remaining <= size - offset else { throw GPUError.invalid("Source hash length exceeds file: \(url.path)") }
        try file.seek(toOffset: offset)
        var hash = SHA256()
        while remaining > 0 {
            let amount = Int(min(remaining, 8 * 1024 * 1024))
            guard let data = try file.read(upToCount: amount), data.count == amount else {
                throw GPUError.invalid("Truncated source during hashing: \(url.path)")
            }
            hash.update(data: data)
            remaining -= UInt64(data.count)
        }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }

    static func mlxVersion() throws -> String {
        var value = mlx_string_new()
        defer { _ = mlx_string_free(value) }
        try MX.check(mlx_version(&value), "MLX version")
        guard let text = mlx_string_data(value) else { throw GPUError.invalid("Missing MLX version string") }
        return String(cString: text)
    }

    static func comparison(_ actual: CoreMLTensor, _ expected: CoreMLTensor) throws -> [String: Any] {
        guard actual.shape == expected.shape, actual.values.count == expected.values.count else {
            throw GPUError.invalid("Comparison shape mismatch: \(actual.shape) versus \(expected.shape)")
        }
        var maximum = 0.0, squared = 0.0, expectedSquared = 0.0
        for (a, b) in zip(actual.values, expected.values) {
            let delta = a - b
            maximum = max(maximum, abs(delta))
            squared += delta * delta
            expectedSquared += b * b
        }
        return ["shape": actual.shape, "maxAbsoluteError": maximum,
                "rootMeanSquareError": sqrt(squared / Double(actual.values.count)),
                "relativeL2Error": expectedSquared > 0 ? sqrt(squared / expectedSquared) as Any : NSNull(),
                "exactMatch": actual.values == expected.values]
    }

    static func compareRouting(
        actualIDs: CoreMLTensor, actualWeights: CoreMLTensor,
        expectedIDs: CoreMLTensor?, expectedWeights: CoreMLTensor?
    ) throws -> (setsMatch: Bool?, alignedWeights: [String: Any]?) {
        guard let expectedIDs else { return (nil, nil) }
        guard actualIDs.shape == expectedIDs.shape, actualIDs.shape.count == 3,
              actualWeights.shape == actualIDs.shape else {
            throw GPUError.invalid("Routing reference dimensions do not match GPU diagnostics")
        }
        let topK = actualIDs.shape[2]
        var matches = true
        var alignedActual: [Double] = [], alignedExpected: [Double] = []
        if let expectedWeights, expectedWeights.shape != expectedIDs.shape {
            throw GPUError.invalid("Expected routing weights do not match expected IDs")
        }
        for start in stride(from: 0, to: actualIDs.values.count, by: topK) {
            let actual = Array(actualIDs.values[start..<(start + topK)])
            let expected = Array(expectedIDs.values[start..<(start + topK)])
            let rowMatches = Set(actual).count == topK && Set(expected).count == topK && Set(actual) == Set(expected)
            matches = matches && rowMatches
            if rowMatches, let expectedWeights {
                for slot in 0..<topK {
                    let actualSlot = actual.firstIndex(of: expected[slot])!
                    alignedActual.append(actualWeights.values[start + actualSlot])
                    alignedExpected.append(expectedWeights.values[start + slot])
                }
            }
        }
        guard matches, expectedWeights != nil else { return (matches, nil) }
        let shape = [alignedActual.count]
        return (true, try comparison(CoreMLTensor(shape: shape, dtype: .float32, values: alignedActual),
                                     CoreMLTensor(shape: shape, dtype: .float32, values: alignedExpected)))
    }

    static func logicalBytes(
        weights: GPUWeights, layer: Int, experts: Int, topK: Int,
        tokenCount: Int, actualIDs: [Int], medianMilliseconds: Double
    ) throws -> [String: Any] {
        let uniqueIDs = Set(actualIDs).sorted()
        let prefix = "language_model.model.layers.\(layer).mlp."
        var rows: [[String: Any]] = []
        var routedPacked: UInt64 = 0, routedScales: UInt64 = 0, routedBiases: UInt64 = 0
        var denseBytes: UInt64 = 0
        for entry in weights.ledger {
            let metadata = try weights.metadata(entry.name)
            guard metadata.name.hasPrefix(prefix) else { throw GPUError.invalid("Unexpected non-MoE weight in probe ledger") }
            let local = String(metadata.name.dropFirst(prefix.count))
            let routed = local.hasPrefix("switch_mlp.")
            let perExpert: UInt64
            let counted: UInt64
            if routed {
                guard metadata.shape.first == experts, metadata.byteCount % UInt64(experts) == 0 else {
                    throw GPUError.invalid("Cannot divide packed expert tensor into equal expert slices")
                }
                perExpert = metadata.byteCount / UInt64(experts)
                counted = perExpert * UInt64(uniqueIDs.count)
                if local.hasSuffix(".weight") { routedPacked += perExpert }
                else if local.hasSuffix(".scales") { routedScales += perExpert }
                else if local.hasSuffix(".biases") { routedBiases += perExpert }
            } else {
                perExpert = 0
                counted = metadata.byteCount
                denseBytes += counted
            }
            rows.append(["name": metadata.name, "shape": metadata.shape, "dtype": metadata.dtypeName,
                         "fullSourceTensorBytes": metadata.byteCount,
                         "bytesPerExpert": routed ? perExpert as Any : NSNull(),
                         "expertsCounted": routed ? uniqueIDs.count : 0,
                         "logicalBytesPerWindow": counted,
                         "countingRule": routed ? "Each unique selected expert once across all tokens" : "Dense router/shared tensor once across the token window"])
        }
        let expertBytes = routedPacked + routedScales + routedBiases
        let uniqueExpertBytes = expertBytes * UInt64(uniqueIDs.count)
        let logicalBytes = uniqueExpertBytes + denseBytes
        let independentTokenBytes = (expertBytes * UInt64(topK) + denseBytes) * UInt64(tokenCount)
        return [
            "definition": "Logical weight footprint of one invocation: each selected expert once across the complete token window; router, shared expert and shared gate once. S2 reuses an expert in this accounting when both tokens select it. Cache behavior, physical fetch count and reuse across calls are unknown.",
            "tokenCount": tokenCount, "topK": topK, "assignmentCount": actualIDs.count,
            "uniqueSelectedExpertIDs": uniqueIDs, "uniqueSelectedExpertCount": uniqueIDs.count,
            "routedPackedQ4BytesPerExpert": routedPacked,
            "routedBF16ScaleBytesPerExpert": routedScales,
            "routedBF16BiasBytesPerExpert": routedBiases,
            "routedWeightBytesPerExpert": expertBytes,
            "selectedExpertLogicalWeightBytesPerWindow": uniqueExpertBytes,
            "routerAndSharedLogicalWeightBytesPerWindow": denseBytes,
            "logicalWeightBytesPerWindow": logicalBytes,
            "logicalWeightBytesPerTokenAverage": Double(logicalBytes) / Double(tokenCount),
            "logicalWeightBytesIfCountedSeparatelyForEveryToken": independentTokenBytes,
            "logicalWeightBytesPerWallSecondGBps": Double(logicalBytes) / (medianMilliseconds * 1_000_000),
            "logicalRateMeaning": "Decimal GB/s from logical weight bytes divided by measured wall time; not a measured DRAM bandwidth or hardware utilization percentage.",
            "excludedTraffic": "Activation/intermediate reads and writes, sorting/index buffers, instructions, allocator effects and implementation-specific bank metadata traffic are not included.",
            "weightTensors": rows,
        ]
    }
}
