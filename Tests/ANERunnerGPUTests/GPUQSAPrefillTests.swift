import CMLX
import Dispatch
import Foundation
import XCTest
@testable import ANERunnerGPU

/// Real GPU operator probes, without model weights. Cross-kernel BF16 error is
/// measured, not required to be bitwise zero. Passing establishes finite output,
/// shape and mask isolation only; the full-model golden remains a separate gate.
final class GPUQSAPrefillTests: XCTestCase {
    func testQSAChunkAcrossSparseThreshold() throws { try exercise(total: 2080) }
    func testQSALongContextPrefillChunk() throws { try exercise(total: 11056) }

    private func exercise(total: Int) throws {
        let sequence = 416, heads = 24, kvHeads = 2, width = 256
        let queryShape = [1,heads,sequence,width], kvShape = [1,kvHeads,total,width]
        let qData = values(count: heads * sequence * width, seed: 0x1234)
        let kData = values(count: kvHeads * total * width, seed: 0x2345)
        let vData = values(count: kvHeads * total * width, seed: 0x3456)
        let query = try MX.array(qData, shape: queryShape, dtype: MLX_BFLOAT16)
        let keys = try MX.array(kData, shape: kvShape, dtype: MLX_BFLOAT16)
        let vals = try MX.array(vData, shape: kvShape, dtype: MLX_BFLOAT16)
        let selection = mask(total: total, sequence: sequence)
        XCTAssertGreaterThan(selection.visibleCounts.min()!, 0)
        XCTAssertLessThanOrEqual(selection.visibleCounts.max()!, 512 * 4 + 3)
        let boolMask = try MX.array(data: Data(selection.bytes), shape: [1,1,sequence,total], dtype: MLX_BOOL)
        try MX.eval([query, keys, vals, boolMask])

        func run(_ fused: Bool, keys k: Tensor, values v: Tensor) throws -> Tensor {
            let output = try MX.sdpa(query, k, v, scale: 1 / 16, mask: boolMask, forceFused: fused)
            try output.eval()
            guard output.shape == queryShape, output.dtype == MLX_BFLOAT16 else {
                throw GPUError.invalid("QSA prefill SDPA returned an unexpected shape or dtype")
            }
            return output
        }
        let reference = try run(false, keys: keys, values: vals).floats()
        let fused = try run(true, keys: keys, values: vals).floats()
        guard reference.allSatisfy(\.isFinite), fused.allSatisfy(\.isFinite) else {
            throw GPUError.invalid("Nonfinite QSA prefill SDPA output")
        }
        let error = comparison(reference, fused)

        // Two discarded warmups per dispatch, then one ABBA block (two samples
        // each). Timers include graph submission and output evaluation only.
        for mode in [false, true] {
            for _ in 0..<2 { _ = try run(mode, keys: keys, values: vals) }
        }
        var samples: [[String: Any]] = [], aTimes: [Double] = [], bTimes: [Double] = []
        for mode in [false, true, true, false] {
            let start = DispatchTime.now().uptimeNanoseconds
            _ = try run(mode, keys: keys, values: vals)
            let milliseconds = Double(DispatchTime.now().uptimeNanoseconds - start) * 1e-6
            if mode { bTimes.append(milliseconds) } else { aTimes.append(milliseconds) }
            samples.append(["variant": mode ? "force_fused" : "default_dispatch", "milliseconds": milliseconds])
        }

        var isolation: [[String: Any]] = []
        for mutation in ["globally_unselected_block", "future_last_row"] {
            var changedKeys = kData, changedValues = vData
            for head in 0..<kvHeads {
                let changedRows = mutation == "globally_unselected_block" ? Array(0..<4) : [total - 1]
                for row in changedRows {
                    for column in 0..<width {
                        let at = (head * total + row) * width + column
                        if mutation == "globally_unselected_block" {
                            changedKeys[at] += 8; changedValues[at] -= 6
                        } else {
                            // Align the altered key with the final query in
                            // one GQA head per KV head, ensuring a strong visible
                            // positive control at that final query, without NaN.
                            let queryAt = ((head * 12) * sequence + sequence - 1) * width + column
                            changedKeys[at] = qData[queryAt] * 8
                            changedValues[at] = 6
                        }
                    }
                }
            }
            let alteredK = try MX.array(changedKeys, shape: kvShape, dtype: MLX_BFLOAT16)
            let alteredV = try MX.array(changedValues, shape: kvShape, dtype: MLX_BFLOAT16)
            try MX.eval([alteredK, alteredV])
            for mode in [false, true] {
                let original = mode ? fused : reference
                let changed = try run(mode, keys: alteredK, values: alteredV).floats()
                XCTAssertTrue(changed.allSatisfy(\.isFinite))
                let protectedRows = mutation == "globally_unselected_block" ? sequence : sequence - 1
                let protectedChanges = differences(original, changed, sequence: sequence, rows: 0..<protectedRows)
                XCTAssertEqual(protectedChanges, 0, "\(mutation) leaked into masked queries; forceFused=\(mode), N=\(total)")
                let visibleChanges = differences(original, changed, sequence: sequence, rows: (sequence-1)..<sequence)
                if mutation == "future_last_row" {
                    XCTAssertGreaterThan(visibleChanges, 0, "The deliberately visible last row must affect the last query")
                }
                isolation.append(["mutation": mutation, "force_fused": mode,
                    "protected_query_rows_per_head": protectedRows,
                    "protected_different_elements": protectedChanges, "last_query_different_elements": visibleChanges])
            }
        }
        let report: [String: Any] = [
            "query_shape": queryShape, "kv_shape": kvShape, "mask_shape": [1,1,sequence,total],
            "input_dtype": "bfloat16", "input_distribution": "deterministic finite uniform approximately [-2,2]",
            "mask": "At most 512 visible complete four-token blocks plus the causal incomplete tail; block 0 excluded globally; most recent complete block always selected",
            "minimum_visible_tokens": selection.visibleCounts.min()!,
            "maximum_visible_tokens": selection.visibleCounts.max()!,
            "reference": "MX.sdpa default dispatch, forceFused=false; not an FP32 oracle",
            "candidate": "Same q/k/v/bool mask with forceFused=true",
            "cross_kernel_error": error, "mask_isolation": isolation,
            "warmups_per_variant": 2, "abba_samples": samples,
            "default_median_milliseconds": aTimes.reduce(0,+) / 2,
            "force_fused_median_milliseconds": bTimes.reduce(0,+) / 2,
            "timing_scope": "Resident tensors; host graph submission and synchronous output eval, excluding host copies and fixture creation",
            "accuracy_threshold_applied": false
        ]
        let data = try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys])
        print("QSA_PREFILL_PROBE " + String(decoding: data, as: UTF8.self))
    }

    private func values(count: Int, seed: UInt64) -> [Float] {
        var state = seed
        return (0..<count).map { _ in
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Float(Int((state >> 32) & 0xffff) - 32768) / 16384
        }
    }

    /// Mirrors QSA's block visibility and incomplete causal tail. Selection
    /// scores are synthetic; this tests masked SDPA, not the learned indexer.
    private func mask(total: Int, sequence: Int) -> (bytes: [UInt8], visibleCounts: [Int]) {
        var bytes = [UInt8](repeating: 0, count: sequence * total), counts = [Int]()
        for row in 0..<sequence {
            let position = total - sequence + row, complete = (position + 1) / 4
            let selectedCount = min(512, complete - 1)
            var picked = [complete - 1] // Keep the latest complete block; never block zero.
            let olderCount = complete - 2
            for i in 0..<(selectedCount - 1) { picked.append(1 + (row * 17 + i) % olderCount) }
            for block in picked {
                for column in (block*4)..<(block*4+4) { bytes[row*total+column] = 1 }
            }
            for column in (complete*4)..<(position+1) { bytes[row*total+column] = 1 }
            counts.append(selectedCount * 4 + (position + 1) % 4)
        }
        return (bytes, counts)
    }

    private func comparison(_ a: [Float], _ b: [Float]) -> [String: Any] {
        var maximum = 0.0, squared = 0.0, norm = 0.0, changed = 0
        for (x, y) in zip(a, b) {
            let delta = Double(y) - Double(x)
            maximum = max(maximum, abs(delta)); squared += delta * delta; norm += Double(x) * Double(x)
            if x.bitPattern != y.bitPattern { changed += 1 }
        }
        return ["max_absolute_error": maximum, "relative_l2": norm > 0 ? sqrt(squared/norm) as Any : NSNull(),
                "different_elements": changed, "element_count": a.count, "all_finite": true]
    }
    private func differences(_ a: [Float], _ b: [Float], sequence: Int, rows: Range<Int>) -> Int {
        var count = 0
        for head in 0..<24 {
            for row in rows {
                for column in 0..<256 {
                    let at = (head * sequence + row) * 256 + column
                    if a[at].bitPattern != b[at].bitPattern { count += 1 }
                }
            }
        }
        return count
    }
}
