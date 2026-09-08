import Foundation
import XCTest
@testable import ANERunnerGPU

/// Policy/JSON tests only. The temporary configuration has no weight files;
/// no model, MLX tensor, device stream, or external model fixture is loaded.
final class QwenPrefixCachePolicyTests: XCTestCase {
    private func configuration() throws -> QwenConfiguration {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("prefix-cache-policy-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let positiveKeys = [
            "hidden_size", "num_hidden_layers", "vocab_size", "num_attention_heads", "num_key_value_heads",
            "head_dim", "num_experts", "num_experts_per_tok", "moe_intermediate_size",
            "shared_expert_intermediate_size", "hc_count", "hc_lowrank", "full_attention_interval",
            "linear_conv_kernel_dim", "linear_key_head_dim", "linear_value_head_dim",
            "linear_num_key_heads", "linear_num_value_heads", "indexer_budget", "indexer_compress_ratio",
            "indexer_head_dim", "indexer_n_heads", "indexer_kv_heads", "ngram_size", "heads_per_ngram",
            "ngram_vocab_size_base", "split_ngram_parts", "make_ngram_vocab_size_divisible_by",
            "ple_embed_dim", "ple_conv_kernel_size", "max_position_embeddings", "bos_token_id", "eos_token_id"
        ]
        var text = Dictionary(uniqueKeysWithValues: positiveKeys.map { ($0, 1 as Any) })
        text["num_hidden_layers"] = 48
        text["vocab_size"] = 248_320
        text["max_position_embeddings"] = 262_144
        text["eos_token_id"] = 2
        text["ple_layer_ids"] = [2]
        text["layer_types"] = Array(repeating: "linear_attention", count: 48)
        text["rms_norm_eps"] = 1e-6
        text["rope_parameters"] = ["rope_theta": 10_000_000.0, "partial_rotary_factor": 0.25]
        text["hidden_act"] = "silu"
        text["output_gate_type"] = "sigmoid"
        let root: [String: Any] = [
            "model_type": "qwen4_exp", "text_config": text,
            "quantization": ["mode": "affine", "bits": 4, "group_size": 64],
            "ngram_table": ["file": "never-opened.bin", "format": "fp8_e4m3fn", "scale": 1.0]
        ]
        try JSONSerialization.data(withJSONObject: root).write(to: directory.appendingPathComponent("config.json"))
        return try QwenConfiguration(modelDirectory: directory)
    }

    private func request(count: Int = 1_000, chunk: Int = 416, depth: Int = 0,
                         hint: Int? = 900, evaluation: Int = 4,
                         attention: GPUAttention.PrefillMode = .reference,
                         moe: GPUMoEPrefillConfiguration? = nil) -> QwenGenerationRequest {
        QwenGenerationRequest(tokens: Array(repeating: 42, count: count),
            prefillChunk: chunk, mtpDepth: depth,
            prefillEvaluateEveryLayers: evaluation, prefillAttention: attention,
            prefillMoEConfiguration: moe, prefixCacheMaxTokens: hint)
    }

    func testHintValidationAllowsOptOutAndRequiresPrefixShorterThanPrompt() throws {
        let c = try configuration()
        for hint: Int? in [nil, 0, 1, 415, 416, 999] {
            XCTAssertNoThrow(try request(hint: hint).validate(configuration: c), "hint=\(String(describing: hint))")
        }
        for hint in [Int.min, -1, 1_000, 1_001, Int.max] {
            XCTAssertThrowsError(try request(hint: hint).validate(configuration: c)) {
                guard case QwenGenerationError.invalidRequest(let message) = $0 else {
                    return XCTFail("Unexpected error: \($0)")
                }
                XCTAssertTrue(message.contains("prefixCacheMaxTokens"))
            }
        }
        XCTAssertNoThrow(try request(count: 1, hint: 0).validate(configuration: c))
        XCTAssertNoThrow(try request(count: 1, hint: nil).validate(configuration: c))
        XCTAssertThrowsError(try request(count: 1, hint: 1).validate(configuration: c))
    }

    func testBoundaryRoundsDownWithoutChangingOriginalChunkSchedule() throws {
        let c = try configuration()
        let cases: [(hint: Int?, boundary: Int)] = [
            (nil, 0), (0, 0), (1, 0), (415, 0), (416, 416),
            (831, 416), (832, 832), (10_000, 9_984), (11_056, 10_816)
        ]
        for test in cases {
            let r = request(count: 11_057, hint: test.hint)
            try r.validate(configuration: c)
            XCTAssertEqual(r.prefixCacheBoundary, test.boundary)
            XCTAssertEqual(r.prefixCacheBoundary % r.prefillChunk, 0)
            XCTAssertLessThan(r.prefixCacheBoundary, r.tokens.count)
            XCTAssertLessThanOrEqual(r.prefixCacheBoundary, test.hint ?? 0)
        }
        // These boundaries leave the final prompt token for its original
        // one-token forward and initial output-token selection.
        XCTAssertEqual(request(count: 417, hint: 416).prefixCacheBoundary, 416)
        XCTAssertEqual(request(count: 416, hint: 415).prefixCacheBoundary, 0)
        XCTAssertEqual(request(count: 1, hint: 0).prefixCacheBoundary, 0)
        XCTAssertEqual(request(count: 1_025, chunk: 512, hint: 1_024).prefixCacheBoundary, 1_024)
        XCTAssertEqual(request(count: 5, chunk: 1, hint: 4).prefixCacheBoundary, 4)
    }

    func testMTPBypassesCacheWithoutTurningValidHintIntoRequestFailure() throws {
        let c = try configuration()
        for depth in 1...4 {
            let r = request(depth: depth)
            XCTAssertNoThrow(try r.validate(configuration: c))
            XCTAssertEqual(r.prefixCacheBoundary, 0)
        }
        XCTAssertGreaterThan(request(depth: 0).prefixCacheBoundary, 0)
    }

    func testNamespaceSeparatesEveryPrefillNumericalPolicy() {
        let baseline = request().prefixCacheNamespace(accumulation: "reference", fusedPrefill: true)
        let variants = [
            request(chunk: 256).prefixCacheNamespace(accumulation: "reference", fusedPrefill: true),
            request(evaluation: 8).prefixCacheNamespace(accumulation: "reference", fusedPrefill: true),
            request(attention: .fusedQSA).prefixCacheNamespace(accumulation: "reference", fusedPrefill: true),
            request().prefixCacheNamespace(accumulation: "fp32", fusedPrefill: true),
            request().prefixCacheNamespace(accumulation: "reference", fusedPrefill: false),
            request(moe: .init(threadgroups: [416: 256]))
                .prefixCacheNamespace(accumulation: "reference", fusedPrefill: true)
        ]
        for variant in variants { XCTAssertNotEqual(variant, baseline) }
        XCTAssertEqual(Set(variants + [baseline]).count, variants.count + 1)
    }

    func testNamespaceSeparatesMoEGeometryAndSortsDictionaryKeys() {
        func key(_ moe: GPUMoEPrefillConfiguration) -> String {
            request(moe: moe).prefixCacheNamespace(accumulation: "reference", fusedPrefill: true)
        }
        var forward: [Int: Int] = [:], reverse: [Int: Int] = [:]
        let pairs = [(205, 512), (240, 128), (416, 256)]
        for (tokens, threads) in pairs { forward[tokens] = threads }
        for (tokens, threads) in pairs.reversed() { reverse[tokens] = threads }
        XCTAssertEqual(key(.init(threadgroups: forward, gateUpVariant: 2, groupedDown: true)),
                       key(.init(threadgroups: reverse, gateUpVariant: 2, groupedDown: true)))

        let baseline = key(.init(threadgroups: [416: 256], gateUpVariant: 2, groupedDown: true))
        let variants = [
            key(.init(threadgroups: [240: 256], gateUpVariant: 2, groupedDown: true)),
            key(.init(threadgroups: [416: 128], gateUpVariant: 2, groupedDown: true)),
            key(.init(threadgroups: [416: 256], gateUpVariant: 3, groupedDown: true)),
            key(.init(threadgroups: [416: 256], gateUpVariant: 2, groupedDown: false)),
            key(.init(threadgroups: [416: 256]))
        ]
        for variant in variants { XCTAssertNotEqual(variant, baseline) }
        XCTAssertEqual(Set(variants + [baseline]).count, variants.count + 1)
    }

    func testCacheBoundaryHintDoesNotFragmentIdenticalExecutionNamespace() {
        let a = request(count: 1_000, hint: 832)
        let b = request(count: 1_200, hint: 900)
        XCTAssertEqual(a.prefixCacheNamespace(accumulation: "reference", fusedPrefill: true),
                       b.prefixCacheNamespace(accumulation: "reference", fusedPrefill: true))
    }

    private let legacyStatistics = """
        {"promptTokenCount":1000,"chunkCount":4,"targetSeconds":2.0,
         "draftHistorySeconds":0.0,"totalSeconds":2.5,"ssdWaitSeconds":0.1,
         "ssdLogicalBytes":120,"evaluateEveryLayers":4}
        """

    func testHistoricalStatisticsDecodeWithNilCacheFieldsAndOriginalRates() throws {
        let stats = try JSONDecoder().decode(QwenPrefillStatistics.self, from: Data(legacyStatistics.utf8))
        XCTAssertNil(stats.attentionMode)
        XCTAssertNil(stats.suspensionSeconds)
        XCTAssertNil(stats.cachedTokenCount)
        XCTAssertNil(stats.computedTokenCount)
        XCTAssertNil(stats.cacheLookupSeconds)
        XCTAssertNil(stats.cacheRestoreSeconds)
        XCTAssertNil(stats.cacheSaveSeconds)
        XCTAssertEqual(stats.promptTokenCount, 1_000)
        XCTAssertEqual(stats.targetTokensPerSecond, 500)
        XCTAssertEqual(stats.readyTokensPerSecond, 400)
    }

    func testHitRatesCountOnlyComputedTokensAndRoundTripCacheCosts() throws {
        var stats = try JSONDecoder().decode(QwenPrefillStatistics.self, from: Data(legacyStatistics.utf8))
        stats.cachedTokenCount = 900
        stats.computedTokenCount = 100
        stats.cacheLookupSeconds = 0.001
        stats.cacheRestoreSeconds = 0.2
        stats.cacheSaveSeconds = 0.3
        XCTAssertEqual(stats.promptTokenCount, 1_000)
        XCTAssertEqual(stats.targetTokensPerSecond, 50)
        XCTAssertEqual(stats.readyTokensPerSecond, 40)
        let roundTrip = try JSONDecoder().decode(QwenPrefillStatistics.self, from: JSONEncoder().encode(stats))
        XCTAssertEqual(roundTrip.cachedTokenCount, 900)
        XCTAssertEqual(roundTrip.computedTokenCount, 100)
        XCTAssertEqual(roundTrip.cacheLookupSeconds, 0.001)
        XCTAssertEqual(roundTrip.cacheRestoreSeconds, 0.2)
        XCTAssertEqual(roundTrip.cacheSaveSeconds, 0.3)
        XCTAssertEqual(roundTrip.targetTokensPerSecond, 50)
        XCTAssertEqual(roundTrip.readyTokensPerSecond, 40)
    }

    func testZeroDurationRatesRemainUnavailableForCachedRequests() throws {
        let json = legacyStatistics.replacingOccurrences(of: "\"targetSeconds\":2.0", with: "\"targetSeconds\":0.0")
            .replacingOccurrences(of: "\"totalSeconds\":2.5", with: "\"totalSeconds\":0.0")
        var stats = try JSONDecoder().decode(QwenPrefillStatistics.self, from: Data(json.utf8))
        stats.cachedTokenCount = 900; stats.computedTokenCount = 100
        XCTAssertNil(stats.targetTokensPerSecond)
        XCTAssertNil(stats.readyTokensPerSecond)
    }
}
