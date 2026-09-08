#!/usr/bin/env python3
"""Recompute the bounded prefix-cache probe's output/state/metric contracts.

This reads real GPU probe reports; its CPU tests use deliberately small parser
fixtures and do not establish model correctness or performance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


LABELS = {
    "long": ["cold_a", "cold_b", "populate_a", "hit_b", "hit_a"],
    "boundaries": ["qsa_cold_2051", "qsa_cold_2053", "qsa_populate_2051", "qsa_hit_2053", "qsa_hit_2051"],
    "lifecycle": ["lifecycle_cold_a", "lifecycle_cold_b", "lifecycle_populate_a", "lifecycle_evict_a_with_b",
                  "lifecycle_hit_b_after_cancel", "lifecycle_payload_too_large", "lifecycle_mtp_cold_bypass",
                  "lifecycle_populate_after_publish_cancel", "lifecycle_hit_after_publish_cancel",
                  "lifecycle_cold_anchor_832", "lifecycle_tree_populate_416", "lifecycle_tree_hit_416_publish_832",
                  "lifecycle_tree_longest_832"],
}
HOST_FIELDS = ("offset", "valid", "gdnOffsets", "attentionOffsets", "pleHistory", "gdnCapturePresent", "pleCapturePresent")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(report):
    require(report.get("schema") == "qwen38-prefix-cache-probe-v1", "unknown report schema")
    require(report.get("complete") is True and report.get("passed") is True, "incomplete or failed GPU probe")
    suite = report.get("suite")
    require(suite in {*LABELS, "all"}, "unknown suite")
    suites = list(LABELS) if suite == "all" else [suite]
    trials = report.get("trials", [])
    require([row.get("label") for row in trials] == sum((LABELS[s] for s in suites), []), "missing, extra or reordered trials")
    checks = report.get("checks", {})
    require(bool(checks) and all(value is True for value in checks.values()), "lifecycle or coverage check failed")
    if "lifecycle" in suites:
        cancellation = report.get("publication_cancellation", {})
        require(cancellation.get("pending_publish_observer_events") == 1, "publication cancellation was not exercised")
        statistics = cancellation.get("cache_statistics", {})
        require(all(statistics.get(field) == 0 for field in ("entries", "logicalPayloadBytes", "published")),
                "cancelled snapshot contaminated cache")
    oracles = {}
    total_ids = 0
    hits = []
    for row in trials:
        label = row["label"]
        require(row.get("passed") is True, f"{label}: trial failed")
        for field in ("callback_exact", "offset_exact", "counts_exact", "mtp_mode_exact", "state_observed", "output_exact"):
            require(row.get(field) is True, f"{label}: {field} failed")
        ids = row.get("generated_token_ids", [])
        require(ids and all(type(value) is int and value >= 0 for value in ids), f"{label}: missing/invalid output IDs")
        maximum = row["max_tokens"]
        require(len(ids) <= maximum, f"{label}: output exceeds budget")
        require(row["finish_reason"] in ("length", "eos"), f"{label}: invalid finish reason")
        require(row["finish_reason"] != "length" or len(ids) == maximum, f"{label}: truncated length output")
        key = row["prompt_key"]
        if row["is_cold_oracle"]:
            require(key not in oracles and row["expected_cached_tokens"] == 0 and row["mtp_depth"] == 0,
                    f"{label}: duplicate or non-cold oracle")
            oracles[key] = (ids, row["finish_reason"], row["prompt_token_ids"])
        require(key in oracles, f"{label}: missing preceding cold oracle")
        require((ids, row["finish_reason"], row["prompt_token_ids"]) == oracles[key], f"{label}: output/prompt differs from cold oracle")
        prompt_count = len(row["prompt_token_ids"])
        cached = row["actual_cached_tokens"]
        require(type(cached) is int and cached == row["expected_cached_tokens"] and 0 <= cached < prompt_count,
                f"{label}: incorrect hit length")
        require(row["computed_tokens"] == prompt_count - cached, f"{label}: skipped tokens counted as compute")
        result = row["result"]
        require(result["tokens"] == ids and result["finishReason"] == row["finish_reason"], f"{label}: raw result changed")
        stats = result["statistics"]
        require(stats["promptTokenCount"] == prompt_count and stats["generatedTokenCount"] == len(ids), f"{label}: usage changed")
        require(stats["mtpDepth"] == row["mtp_depth"], f"{label}: MTP mode changed")
        phase = result["phases"]["prefill"]
        require(phase.get("cachedTokenCount", 0) == cached and phase.get("computedTokenCount", prompt_count) == prompt_count - cached,
                f"{label}: raw prefill counts disagree")
        require(phase["promptTokenCount"] == prompt_count, f"{label}: phase usage lost full prompt")
        if cached:
            hits.append({"label": label, "cached_tokens": cached, "computed_tokens": prompt_count - cached,
                         "ttft_seconds": result["timeToFirstTokenSeconds"], "prefill_compute_seconds": phase["targetSeconds"],
                         "decode_seconds": result["decodeSeconds"], "cache_restore_seconds": phase.get("cacheRestoreSeconds")})
        total_ids += len(ids)
    states = report.get("state_checks", [])
    require(not report["state_readback"] or bool(states), "state readback requested but absent")
    tensor_comparisons = 0
    for state in states:
        require(state.get("passed") is True and state.get("all_tensor_bytes_exact") is True and
                state.get("logical_host_exact") is True and state.get("no_verification_capture") is True,
                "state comparison failed")
        a, b = state["host_a"], state["host_b"]
        require(all(a[field] == b[field] for field in HOST_FIELDS), "host state or PLE history mismatch")
        require(not any(b["gdnCapturePresent"]) and not any(b["pleCapturePresent"]), "verification capture entered cache")
        tensors = state.get("tensors", [])
        require(len(tensors) == state["tensor_count"] > 0, "incomplete state tensor inventory")
        require(len({row["name"] for row in tensors}) == len(tensors), "duplicate tensor comparison")
        for row in tensors:
            require(row.get("exact") is True and row.get("all_finite") is True, "non-exact/nonfinite tensor")
            require(all(row[field + "_a"] == row[field + "_b"] for field in ("shape", "dtype", "byte_count", "sha256")),
                    "tensor bytes, dtype or shape mismatch")
            require(row["byte_count_a"] > 0 and len(row["sha256_a"]) == 64, "invalid tensor digest")
        tensor_comparisons += len(tensors)
    return {"passed": True, "suite": suite, "generation_trials": len(trials), "complete_output_ids": total_ids,
            "state_comparisons": len(states), "tensor_comparisons": tensor_comparisons,
            "timings_include_state_readback": report["state_readback"], "hits": hits,
            "note": "Timings need separate cold/hit interpretation; this validator does not declare a stable speedup."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    try:
        summary = validate(json.loads(args.report.read_text()))
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(1, f"prefix cache report validation failed: {error}\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
