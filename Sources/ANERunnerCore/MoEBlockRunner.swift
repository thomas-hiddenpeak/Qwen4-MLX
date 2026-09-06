import Accelerate
import CoreML
import Darwin
import Dispatch
import Foundation

/// Exported expert packages are a bank, not a fixed routing decision. Every
/// token still routes across the full router matrix; missing experts are errors.
public struct MoEManifest: Decodable, Sendable {
    public struct Routing: Decodable, Sendable {
        let weightsFile: String
        let sharedGateFile: String
        let dtype: String
    }
    public let schemaVersion: Int
    public let layerIndex: Int
    public let hiddenSize: Int
    public let expertCount: Int
    public let topK: Int
    public let tokenCapacity: Int
    public let inputName: String
    public let outputName: String
    public let dtype: String
    public let routing: Routing
    public let experts: [String: String]
    public let sharedExpert: String
    public let weightMode: String

    public static func load(from url: URL) throws -> Self {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let value = try decoder.decode(Self.self, from: Data(contentsOf: url))
        guard value.schemaVersion == 1, value.layerIndex >= 0,
            value.hiddenSize > 0, value.hiddenSize <= Int(Int32.max),
            value.expertCount > 0, value.topK > 0, value.topK <= value.expertCount,
            value.tokenCapacity > 0, value.tokenCapacity <= 4096,
            value.dtype == "float16", value.routing.dtype == "float32_le",
            !value.inputName.isEmpty, !value.outputName.isEmpty,
            value.experts.keys.allSatisfy({ Int($0).map { $0 >= 0 && $0 < value.expertCount } ?? false })
        else { throw ExpertRouterError.invalid("Invalid or unsupported MoE manifest") }
        return value
    }

    func resolve(_ path: String, relativeTo directory: URL) throws -> URL {
        guard !path.isEmpty, !path.hasPrefix("/"),
            !path.split(separator: "/").contains("..") else {
            throw ExpertRouterError.invalid("Manifest paths must be relative and stay inside the export directory")
        }
        let root = directory.resolvingSymlinksInPath().standardizedFileURL
        let result = directory.appendingPathComponent(path).resolvingSymlinksInPath().standardizedFileURL
        guard result.path.hasPrefix(root.path + "/"), FileManager.default.fileExists(atPath: result.path) else {
            throw ExpertRouterError.invalid("Missing or escaping manifest asset: \(path)")
        }
        return result
    }

    func readFloats(_ path: String, count: Int, directory: URL) throws -> [Float] {
        let url = try resolve(path, relativeTo: directory)
        let bytes = count.multipliedReportingOverflow(by: 4)
        guard !bytes.overflow else { throw ExpertRouterError.invalid("Weight byte count overflow") }
        let data = try Data(contentsOf: url)
        guard data.count == bytes.partialValue else {
            throw ExpertRouterError.invalid("\(path): expected \(bytes.partialValue) bytes, found \(data.count)")
        }
        let values: [Float] = data.withUnsafeBytes { buffer in
            (0..<count).map { Float(bitPattern: UInt32(littleEndian: buffer.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self))) }
        }
        guard values.allSatisfy(\.isFinite) else { throw ExpertRouterError.invalid("Nonfinite weights in \(path)") }
        return values
    }
}

public enum MoEComputeUnits: String, Codable, Sendable {
    case cpuAndNeuralEngine, cpuOnly
    var coreML: MLComputeUnits { self == .cpuOnly ? .cpuOnly : .cpuAndNeuralEngine }
}

public struct MoEIterationTiming: Codable, Sendable {
    public var totalMilliseconds = 0.0
    public var routingMilliseconds = 0.0
    public var expertMilliseconds = 0.0
    public var sharedMilliseconds = 0.0
    public var mergeMilliseconds = 0.0
    public var compilationMilliseconds = 0.0
    public var loadingMilliseconds = 0.0
    public var predictionMilliseconds = 0.0
    public var expertPredictionWallMilliseconds = 0.0
    public var maximumScheduledConcurrency = 0
    public var expertCalls = 0
    public var sharedCalls = 0
    public var cacheMisses = 0
}

public struct MoEBlockReport: Codable, Sendable {
    public let manifestPath: String
    public let layerIndex: Int
    public let tokenCount: Int
    public let tokenCapacity: Int
    public let computeUnits: MoEComputeUnits
    public let routingPrecision: ExpertRouterPrecision
    public let mergePrecision: String
    public let hardwareEvidence: String
    public let selectedExpertIDs: [Int]
    public let exportedExpertCount: Int
    public let maximumResidentExperts: Int
    public let requestedExpertConcurrency: Int
    public let predictionTimingSemantics: String
    public let memoryBeforeExecution: MoEProcessMemory?
    public let memoryAfterExecution: MoEProcessMemory?
    public let firstIteration: MoEIterationTiming
    public let warmups: Int
    public let iterations: [MoEIterationTiming]
    public let medianMilliseconds: Double
    public let outputs: [String: CoreMLTensor]
    public let comparisons: [String: CoreMLTensorComparison]
    public let routingExpertSetsMatch: Bool?
    public let routingWeightComparisonByExpert: CoreMLTensorComparison?
}

public struct MoEProcessMemory: Codable, Sendable {
    public let physicalFootprintBytes: UInt64
    public let residentBytes: UInt64
    public let peakResidentBytes: UInt64

    static func current() -> Self? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let status = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard status == KERN_SUCCESS else { return nil }
        return Self(physicalFootprintBytes: info.phys_footprint, residentBytes: info.resident_size,
                    peakResidentBytes: info.resident_size_peak)
    }
}

/// The CPU owns routing and scheduling. Each selected expert executes its
/// entire SwiGLU gate/up/down graph through Core ML. A bounded LRU controls
/// loaded MLModel objects; this is NOT an OS-level physical memory guarantee.
/// Instances are synchronous, single-request executors and are not Sendable.
public final class MoEBlockRunner {
    public let manifest: MoEManifest
    private let manifestURL: URL
    private let directory: URL
    private let router: ExpertRouter
    private let sharedGateWeights: [Float]
    private let computeUnits: MoEComputeUnits
    private let maxResident: Int
    private let expertConcurrency: Int
    private var kernels: [Int: ConcurrentExpertKernel] = [:]
    private var lru: [Int] = []
    private var compiled: [Int: URL] = [:]
    private var ownedCompiledURLs: [URL] = []
    private var sharedKernel: ConcurrentExpertKernel?

    public init(manifestURL: URL, precision: ExpertRouterPrecision = .bfloat16Boundaries,
                computeUnits: MoEComputeUnits = .cpuAndNeuralEngine, maximumResidentExperts: Int = 64,
                expertConcurrency: Int = 1) throws {
        guard maximumResidentExperts > 0 else { throw ExpertRouterError.invalid("Expert cache capacity must be positive") }
        guard (1...32).contains(expertConcurrency) else { throw ExpertRouterError.invalid("Expert concurrency must be in 1...32") }
        self.manifestURL = manifestURL.standardizedFileURL
        directory = manifestURL.deletingLastPathComponent()
        manifest = try MoEManifest.load(from: manifestURL)
        let matrixCount = manifest.expertCount.multipliedReportingOverflow(by: manifest.hiddenSize)
        guard !matrixCount.overflow else { throw ExpertRouterError.invalid("Router shape overflow") }
        router = try ExpertRouter(
            weights: manifest.readFloats(manifest.routing.weightsFile, count: matrixCount.partialValue, directory: directory),
            expertCount: manifest.expertCount, hiddenSize: manifest.hiddenSize, topK: manifest.topK, precision: precision)
        let gateWeights = try manifest.readFloats(manifest.routing.sharedGateFile, count: manifest.hiddenSize, directory: directory)
        sharedGateWeights = precision == .bfloat16Boundaries ? gateWeights.map(NGramTable.roundBFloat16) : gateWeights
        self.computeUnits = computeUnits
        maxResident = maximumResidentExperts
        self.expertConcurrency = expertConcurrency
    }

    deinit {
        kernels.removeAll()
        sharedKernel = nil
        for url in ownedCompiledURLs { try? FileManager.default.removeItem(at: url) }
    }

    private func input(_ fixture: CoreMLBlockFixture) throws -> (values: [Float], count: Int) {
        guard fixture.inputs.count == 1, let x = fixture.inputs["x"] else {
            throw ExpertRouterError.invalid("MoE fixture must have exactly one input named x")
        }
        try x.validate(name: "x")
        guard x.shape.count == 3, x.shape[0] == 1, x.shape[2] == manifest.hiddenSize,
            x.dtype == .float32 || x.dtype == .float16, x.values.allSatisfy({ Float16($0).isFinite }) else {
            throw ExpertRouterError.invalid("MoE input must be finite FP16-representable [1,T,hiddenSize] values")
        }
        let values = x.values.map(Float.init)
        let prepared = router.precision == .bfloat16Boundaries ? values.map(NGramTable.roundBFloat16) : values
        guard prepared.allSatisfy({ Float16($0).isFinite }) else {
            throw ExpertRouterError.invalid("MoE input exceeds FP16 after requested BF16 rounding")
        }
        return (prepared, x.shape[1])
    }

    /// Route without loading any Core ML expert package, useful before export.
    public func route(fixture: CoreMLBlockFixture) throws -> RoutedTokens {
        let x = try input(fixture)
        return try router.route(tokens: x.values, tokenCount: x.count)
    }

    public func run(fixture: CoreMLBlockFixture, warmups: Int = 1, runs: Int = 3) throws -> MoEBlockReport {
        guard warmups >= 0, runs > 0, runs <= 100 else { throw ExpertRouterError.invalid("Invalid warmup/run count") }
        let expected = fixture.expectedOutputs ?? [:]
        for (name, tensor) in expected { try tensor.validate(name: name) }
        guard (expected["selected_experts"] == nil) == (expected["routing_weights"] == nil) else {
            throw ExpertRouterError.invalid("Expected selected_experts and routing_weights must be provided together")
        }
        let x = try input(fixture)
        let required = Set(try router.route(tokens: x.values, tokenCount: x.count).expertIDs)
        let missing = required.filter { manifest.experts[String($0)] == nil }.sorted()
        guard missing.isEmpty else {
            throw ExpertRouterError.invalid("Dynamic routing requires missing experts: \(missing.map(String.init).joined(separator: ",")). Export them before running; no assignments were dropped.")
        }
        let memoryBefore = MoEProcessMemory.current()
        var last = try iteration(x: x.values, count: x.count)
        let first = last.timing
        for _ in 0..<warmups { last = try iteration(x: x.values, count: x.count) }
        var iterations = [MoEIterationTiming]()
        for _ in 0..<runs {
            last = try iteration(x: x.values, count: x.count)
            iterations.append(last.timing)
        }
        var comparisons = [String: CoreMLTensorComparison]()
        for (name, reference) in expected where name != "selected_experts" && name != "routing_weights" {
            try reference.validate(name: name)
            guard let actual = last.outputs[name], actual.shape == reference.shape else {
                throw ExpertRouterError.invalid("Missing or mismatched expected output \(name)")
            }
            comparisons[name] = Self.compare(actual.values, reference.values)
        }
        var setMatch: Bool?
        var weightComparison: CoreMLTensorComparison?
        if let referenceIDs = expected["selected_experts"] {
            try referenceIDs.validate(name: "selected_experts")
            guard referenceIDs.shape == [1, x.count, manifest.topK], referenceIDs.dtype == .int32 else {
                throw ExpertRouterError.invalid("Expected expert IDs must be int32 [1,T,topK]")
            }
            let ids = referenceIDs.values.map(Int.init)
            var alignedActual = [Double](), alignedReference = [Double]()
            setMatch = true
            if let weights = expected["routing_weights"], weights.shape != referenceIDs.shape {
                throw ExpertRouterError.invalid("Expected routing weight shape differs from IDs")
            }
            for token in 0..<x.count {
                let slots = token * manifest.topK..<(token + 1) * manifest.topK
                guard Set(ids[slots]).count == manifest.topK,
                    ids[slots].allSatisfy({ $0 >= 0 && $0 < manifest.expertCount }) else {
                    throw ExpertRouterError.invalid("Expected routing has duplicate or invalid expert IDs")
                }
                if Set(ids[slots]) != Set(last.routing.expertIDs[slots]) { setMatch = false }
                if let weights = expected["routing_weights"] {
                    for slot in slots {
                        if let actualSlot = slots.first(where: { last.routing.expertIDs[$0] == ids[slot] }) {
                            alignedActual.append(Double(last.routing.weights[actualSlot]))
                            alignedReference.append(weights.values[slot])
                        }
                    }
                }
            }
            if setMatch == true, !alignedActual.isEmpty { weightComparison = Self.compare(alignedActual, alignedReference) }
        }
        let times = iterations.map(\.totalMilliseconds).sorted()
        let mid = times.count / 2
        let median = times.count.isMultiple(of: 2) ? (times[mid-1] + times[mid]) / 2 : times[mid]
        return MoEBlockReport(
            manifestPath: manifestURL.path, layerIndex: manifest.layerIndex, tokenCount: x.count,
            tokenCapacity: manifest.tokenCapacity, computeUnits: computeUnits, routingPrecision: router.precision,
            mergePrecision: "Float32 weighted expert sum; shared sigmoid and final add rounded to BF16 only in bfloat16Boundaries mode. Core ML expert outputs are FP16; this does not emulate every MLX fused-reduction boundary.",
            hardwareEvidence: "CPU routing and scheduling; Core ML experts use the requested compute units. CPU_AND_NE permits CPU fallback. This run contains no hardware execution trace and is not full-model generation.",
            selectedExpertIDs: required.sorted(), exportedExpertCount: manifest.experts.count,
            maximumResidentExperts: maxResident, requestedExpertConcurrency: expertConcurrency,
            predictionTimingSemantics: "predictionMilliseconds sums individual calls and may exceed elapsed time with concurrency. expertPredictionWallMilliseconds measures worker dispatch, input/output materialization and prediction wall time, excluding loading. CPU routing, cache mutation and final reduction are serial.",
            memoryBeforeExecution: memoryBefore,
            memoryAfterExecution: MoEProcessMemory.current(), firstIteration: first, warmups: warmups,
            iterations: iterations, medianMilliseconds: median, outputs: last.outputs, comparisons: comparisons,
            routingExpertSetsMatch: setMatch, routingWeightComparisonByExpert: weightComparison)
    }

    private func iteration(x: [Float], count: Int) throws -> (outputs: [String: CoreMLTensor], routing: RoutedTokens, timing: MoEIterationTiming) {
        let start = DispatchTime.now().uptimeNanoseconds
        var timing = MoEIterationTiming()
        let routeStart = DispatchTime.now().uptimeNanoseconds
        let routing = try router.route(tokens: x, tokenCount: count)
        timing.routingMilliseconds = Self.ms(routeStart)
        let h = manifest.hiddenSize, k = manifest.topK, capacity = manifest.tokenCapacity
        var groups = [Int: [Int]]()
        for slot in routing.expertIDs.indices { groups[routing.expertIDs[slot], default: []].append(slot) }
        var expertOutputs = [Float](repeating: 0, count: count * k * h)
        let expertStart = DispatchTime.now().uptimeNanoseconds
        let experts = groups.keys.sorted()
        let waveSize = min(expertConcurrency, maxResident)
        for waveStart in stride(from: 0, to: experts.count, by: waveSize) {
            // Worker jobs retain models only until this scope returns. Cache
            // mutations occur between joined waves, never on worker threads.
            let batch = try autoreleasepool {
                let jobs = try experts[waveStart..<min(waveStart + waveSize, experts.count)].map { expert in
                    let slots = groups[expert]!
                    let chunks = stride(from: 0, to: slots.count, by: capacity).map { offset in
                        let chunk = Array(slots[offset..<min(offset + capacity, slots.count)])
                        return ExpertPredictionChunk(slots: chunk, tokenIndices: chunk.map { $0 / k })
                    }
                    return ExpertPredictionJob(expertID: expert, kernel: try getKernel(expert, timing: &timing), chunks: chunks)
                }
                return try BoundedExpertExecutor.predict(jobs: jobs, tokens: x, maximumConcurrency: expertConcurrency)
            }
            timing.predictionMilliseconds += batch.predictionMilliseconds
            timing.expertPredictionWallMilliseconds += batch.wallMilliseconds
            timing.maximumScheduledConcurrency = max(timing.maximumScheduledConcurrency, batch.concurrency)
            timing.expertCalls += batch.callCount
            for result in batch.results {
                for chunk in result.chunks {
                    for (local, slot) in chunk.slots.enumerated() {
                        expertOutputs.replaceSubrange(slot*h..<(slot+1)*h, with: chunk.values[local*h..<(local+1)*h])
                    }
                }
            }
        }
        timing.expertMilliseconds = Self.ms(expertStart)
        let sharedStart = DispatchTime.now().uptimeNanoseconds
        let shared = try getKernel(-1, timing: &timing)
        var sharedDown = [Float](repeating: 0, count: count*h)
        for offset in stride(from: 0, to: count, by: capacity) {
            let chunk = Array(offset..<min(offset + capacity, count))
            let (y, milliseconds) = try shared.predict(tokens: x, tokenIndices: chunk)
            timing.predictionMilliseconds += milliseconds
            timing.sharedCalls += 1
            sharedDown.replaceSubrange(offset*h..<(offset+chunk.count)*h, with: y)
        }
        var gateLogits = [Float](), gates = [Float](), sharedGated = sharedDown
        for token in 0..<count {
            let logit = x.withUnsafeBufferPointer { input in
                sharedGateWeights.withUnsafeBufferPointer { weights in
                    cblas_sdot(Int32(h), input.baseAddress! + token*h, 1, weights.baseAddress!, 1)
                }
            }
            let roundedLogit = round(logit)
            let gate = round(1 / (1 + exp(-roundedLogit)))
            gateLogits.append(roundedLogit)
            gates.append(gate)
            for channel in 0..<h { sharedGated[token*h+channel] = round(sharedDown[token*h+channel] * gate) }
        }
        timing.sharedMilliseconds = Self.ms(sharedStart)
        let mergeStart = DispatchTime.now().uptimeNanoseconds
        let routed = try ExpertRouter.weightedMerge(expertOutputs: expertOutputs, routing: routing, outputSize: h)
        let output = zip(routed, sharedGated).map { round($0 + $1) }
        guard output.allSatisfy(\.isFinite), gateLogits.allSatisfy(\.isFinite) else { throw ExpertRouterError.invalid("MoE output overflow") }
        timing.mergeMilliseconds = Self.ms(mergeStart)
        timing.totalMilliseconds = Self.ms(start)
        func tensor(_ values: [Float], _ width: Int) -> CoreMLTensor {
            CoreMLTensor(shape: [1, count, width], dtype: .float32, values: values.map(Double.init))
        }
        return ([
            "router_logits": tensor(routing.logits, manifest.expertCount),
            "selected_experts": CoreMLTensor(shape: [1,count,k], dtype: .int32, values: routing.expertIDs.map(Double.init)),
            "routing_weights": tensor(routing.weights, k), "routed_sum": tensor(routed,h),
            "shared_down": tensor(sharedDown,h), "shared_gate_logits": tensor(gateLogits,1),
            "shared_gate": tensor(gates,1), "shared_gated": tensor(sharedGated,h), "y": tensor(output,h),
        ], routing, timing)
    }

    private func round(_ value: Float) -> Float {
        router.precision == .bfloat16Boundaries ? NGramTable.roundBFloat16(value) : value
    }

    private func getKernel(_ id: Int, timing: inout MoEIterationTiming) throws -> ConcurrentExpertKernel {
        if id == -1, let sharedKernel { return sharedKernel }
        if let kernel = kernels[id] {
            lru.removeAll { $0 == id }; lru.append(id)
            return kernel
        }
        timing.cacheMisses += 1
        guard let path = id == -1 ? manifest.sharedExpert : manifest.experts[String(id)] else {
            throw ExpertRouterError.invalid("Missing dynamically selected expert \(id)")
        }
        // Evict before load; the shared expert has its own single resident slot.
        if id != -1, kernels.count >= maxResident, let oldest = lru.first {
            kernels.removeValue(forKey: oldest); lru.removeFirst()
        }
        let url: URL
        if let existing = compiled[id] { url = existing }
        else {
            let source = try manifest.resolve(path, relativeTo: directory)
            if source.pathExtension == "mlmodelc" { url = source }
            else {
                let start = DispatchTime.now().uptimeNanoseconds
                url = try MLModel.compileModel(at: source)
                timing.compilationMilliseconds += Self.ms(start)
                ownedCompiledURLs.append(url)
            }
            compiled[id] = url
        }
        let start = DispatchTime.now().uptimeNanoseconds
        let kernel = try ConcurrentExpertKernel(url: url, manifest: manifest, units: computeUnits)
        timing.loadingMilliseconds += Self.ms(start)
        if id == -1 { sharedKernel = kernel }
        else { kernels[id] = kernel; lru.append(id) }
        return kernel
    }

    private static func ms(_ start: UInt64) -> Double { Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000 }

    private static func compare(_ a: [Double], _ b: [Double]) -> CoreMLTensorComparison {
        var maximum = 0.0, error = 0.0, norm = 0.0
        for (x,y) in zip(a,b) { maximum = max(maximum, abs(x-y)); error += (x-y)*(x-y); norm += y*y }
        return CoreMLTensorComparison(maxAbsoluteError: maximum, rootMeanSquareError: sqrt(error/Double(a.count)),
                                      relativeL2Error: norm > 0 ? sqrt(error/norm) : nil, exactMatch: maximum == 0)
    }
}

