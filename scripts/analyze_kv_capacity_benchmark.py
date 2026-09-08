#!/usr/bin/env python3
"""Validate complete KV benchmark blocks and summarize every measured sample.

Usage: python3 analyze-capacity-benchmark.py ABBA.json BAAB.json --output NEW.json
       python3 analyze-capacity-benchmark.py --self-test

CPU JSON analysis only. Exit 2 on any invalid input; invalid blocks are never
silently omitted to produce a performance conclusion. Warmups remain in the
raw reports and are validated separately, outside the four measured trials.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import unittest


SCHEMA = "qwen-kv-capacity-benchmark-v1"
FORMAT_SOURCE_SHA256 = "38f79a9a89f14b321c81788110c9304668b4817408b9a8a7d8958d49a6575d98"
ORDERS = {"abba": ("reference", "capacity256", "capacity256", "reference"),
          "baab": ("capacity256", "reference", "reference", "capacity256")}
CHECKS = ("output_exact", "usable_decode", "finish_contract", "cold_prefill",
          "mode_accounting", "leases_released")
HASH_FIELDS = ("executable_sha256", "input_sha256", "config_sha256",
               "tokenizer_sha256", "model_cache_identity")
METRICS = ("prefill_target", "prefill_active", "decode_round", "decode_service")
MAX_INPUT_BYTES = 16 * 1024 * 1024


class Invalid(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise Invalid(message)


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, f"{label}: expected integer >= {minimum}")
    return value


def seconds(value, label, positive=False):
    require(type(value) in (int, float) and math.isfinite(value) and
            (value > 0 if positive else value >= 0), f"{label}: invalid finite seconds")
    return float(value)


def numeric_tree(value, label):
    """Also reject negative/nonfinite numeric fields unknown to this revision."""
    if type(value) in (int, float):
        require(math.isfinite(value) and value >= 0, f"{label}: invalid numeric field")
    elif isinstance(value, dict):
        for key, item in value.items():
            numeric_tree(item, f"{label}.{key}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            numeric_tree(item, f"{label}[{i}]")


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def finite_float(text):
    value = float(text)
    require(math.isfinite(value), "nonfinite JSON float")
    return value


def load_json(data):
    def bad_constant(text):
        raise Invalid(f"nonstandard JSON constant: {text}")
    return json.loads(data, object_pairs_hook=no_duplicates, parse_float=finite_float,
                      parse_constant=bad_constant)


def ids(value, label):
    require(isinstance(value, list) and value, f"{label}: missing complete token IDs")
    for index, token in enumerate(value):
        integer(token, f"{label}[{index}]")
        require(token <= 2**31 - 1, f"{label}: token outside Int32 range")
    return value


def no_greater(a, b, label):
    # Source clocks have nanosecond resolution. This allows rounding, not a
    # performance threshold, and never merges the underlying timer scopes.
    require(a <= b + max(1e-6, abs(b) * 1e-8), f"{label}: inconsistent timer scopes")


def budget(value, label):
    require(isinstance(value, dict), f"{label}: missing budget snapshot")
    for key in ("maxBytes", "requestBytes", "cacheBytes", "workspaceBytes",
                "totalBytes", "peakBytes", "rejections", "currentLeases"):
        integer(value.get(key), f"{label}.{key}")
    require(value["maxBytes"] > 0, f"{label}: zero budget limit")
    require(sum(value[key] for key in ("requestBytes", "cacheBytes", "workspaceBytes")) ==
            value["totalBytes"], f"{label}: budget categories do not sum")
    require(value["totalBytes"] <= value["peakBytes"] <= value["maxBytes"],
            f"{label}: exceeded/invalid budget")
    require(value["totalBytes"] == value["currentLeases"] == 0,
            f"{label}: request workspace/lease did not drain")


def memory(value, label):
    require(isinstance(value, dict), f"{label}: missing MLX observation")
    for key in ("active_bytes", "peak_bytes", "cache_bytes", "limit_bytes"):
        integer(value.get(key), f"{label}.{key}")
    # reset_peak may leave raw peak at zero until an allocation: no peak>=active
    # assertion, and no inference from these global observations to process RSS.


def validate_result(result, mode, config, max_tokens, label):
    require(isinstance(result, dict), f"{label}: missing result")
    output = ids(result.get("tokens"), f"{label}.tokens")
    finish = result.get("finishReason")
    require(finish in ("eos", "length"), f"{label}: invalid finishReason")
    require(2 <= len(output) <= max_tokens, f"{label}: unusable output count")
    require(finish != "length" or len(output) == max_tokens,
            f"{label}: length finish before max_tokens")
    stats, phases = result.get("statistics"), result.get("phases")
    require(isinstance(stats, dict) and isinstance(phases, dict), f"{label}: missing statistics/phases")
    numeric_tree(result, label)
    prefill = phases.get("prefill")
    require(isinstance(prefill, dict), f"{label}: missing prefill phases")
    prompt, decoded = config["prompt_tokens"], len(output) - 1
    exact_stats = {"promptTokenCount": prompt, "generatedTokenCount": len(output),
                   "decodedTokenCount": decoded, "decodeRounds": decoded,
                   "finalStateOffset": prompt + decoded, "mtpDepth": 0}
    for key, expected in exact_stats.items():
        require(integer(stats.get(key), f"{label}.statistics.{key}") == expected,
                f"{label}: mismatched {key}")
    require(stats.get("mtp") is None and stats.get("mtpVerification") is None and
            result.get("mtpCostSummary") is None, f"{label}: unexpected MTP state")
    for key, expected in {"promptTokenCount": prompt, "cachedTokenCount": 0,
                          "computedTokenCount": prompt, "actualForwardTokenCount": prompt,
                          "recomputedTokenCount": 0,
                          "evaluateEveryLayers": config["prefill_eval_layers"]}.items():
        require(integer(prefill.get(key), f"{label}.prefill.{key}") == expected,
                f"{label}: cold prefill mismatch for {key}")
    require(prefill.get("cacheSource") == "cold", f"{label}: unexpected cache source")
    require(integer(prefill.get("chunkCount"), f"{label}.chunkCount", 1) ==
            integer(stats.get("prefillChunkCount"), f"{label}.prefillChunkCount", 1),
            f"{label}: prefill chunk counts disagree")
    require(phases.get("kvAppendMode") == mode, f"{label}: result policy mismatch")
    require(integer(phases.get("kvCapacityTokenSteps"), f"{label}.capacity steps") ==
            (decoded if mode == "capacity256" else 0), f"{label}: successful capacity step mismatch")
    require(integer(phases.get("kvCapacityWorkspaceFallbacks"), f"{label}.fallbacks") == 0,
            f"{label}: benchmark unexpectedly fell back")
    peak = integer(phases.get("kvCapacityWorkspacePeakBytes"), f"{label}.workspace peak")
    require(peak > 0 if mode == "capacity256" else peak == 0, f"{label}: workspace accounting mismatch")
    for key in ("decodeKernelMode",):
        require(isinstance(phases.get(key), str) and phases[key], f"{label}: missing {key}")
    for container, key in ((prefill, "attentionMode"), (stats, "prefillAccumulation")):
        require(isinstance(container.get(key), str) and container[key], f"{label}: missing {key}")
    integer(phases.get("verificationEvaluateEveryLayers"), f"{label}.verify cadence", 1)
    timing = {"prefill_target": seconds(prefill.get("targetSeconds"), label + ".prefill target", True),
              "prefill_active": seconds(prefill.get("totalSeconds"), label + ".prefill active", True),
              "decode_round": seconds(result.get("decodeSeconds"), label + ".decode round", True),
              "decode_service": seconds(phases.get("decodeServiceSeconds"), label + ".decode service", True)}
    total = seconds(result.get("totalSeconds"), label + ".total", True)
    ttft = seconds(result.get("timeToFirstTokenSeconds"), label + ".TTFT", True)
    seconds(result.get("preparationSeconds"), label + ".preparation")
    for key in ("handoffWaitSeconds", "handoffConsumeSeconds", "decodeSSDWaitSeconds", "decodeSuspensionSeconds"):
        seconds(phases.get(key), f"{label}.{key}")
    for key in ("draftHistorySeconds", "suspensionSeconds", "ssdWaitSeconds", "cacheLookupSeconds",
                "cacheRestoreSeconds", "cacheSaveSeconds", "cacheWaitSeconds"):
        seconds(prefill.get(key), f"{label}.prefill.{key}")
    for key in ("ssdWaitSeconds", "callbackSeconds"):
        seconds(stats.get(key), f"{label}.statistics.{key}")
    require(prefill["draftHistorySeconds"] == 0 and prefill["suspensionSeconds"] == 0 and
            phases["decodeSuspensionSeconds"] == 0, f"{label}: unexpected draft/suspension in direct AR benchmark")
    pre_bytes = integer(prefill.get("ssdLogicalBytes"), label + ".prefill SSD bytes")
    dec_bytes = integer(phases.get("decodeSSDLogicalBytes"), label + ".decode SSD bytes")
    require(integer(stats.get("ssdLogicalBytes"), label + ".total SSD bytes") == pre_bytes + dec_bytes,
            f"{label}: SSD logical byte conservation failed")
    require(math.isclose(stats["ssdWaitSeconds"], prefill["ssdWaitSeconds"] + phases["decodeSSDWaitSeconds"],
                         rel_tol=1e-8, abs_tol=1e-6), f"{label}: SSD wait conservation failed")
    no_greater(timing["prefill_target"], timing["prefill_active"], label + ".prefill target/active")
    no_greater(timing["decode_round"], timing["decode_service"], label + ".round/service")
    for key, value in timing.items():
        no_greater(value, total, label + "." + key + "/total")
    no_greater(timing["prefill_target"] + timing["decode_round"], total,
               label + ".sequential prefill target plus decode rounds/total")
    no_greater(ttft, total, label + ".TTFT/total")
    return {"mode": mode, "finish_reason": finish, "generated_tokens": len(output),
            "decoded_tokens": decoded, "actual_prefill_tokens": prefill["actualForwardTokenCount"],
            "seconds": timing, "ttft_seconds_diagnostic_only": ttft,
            "rates_tokens_per_second": {key: (prompt if key.startswith("prefill") else decoded) / value
                                        for key, value in timing.items()},
            "runtime_policy": {"decode_kernel_mode": phases["decodeKernelMode"],
                               "prefill_attention": prefill["attentionMode"],
                               "prefill_accumulation": stats["prefillAccumulation"],
                               "verification_eval_layers": phases["verificationEvaluateEveryLayers"]}}


def validate_report(report, source):
    require(isinstance(report, dict) and report.get("schema") == SCHEMA, "unsupported report schema")
    require(report.get("complete") is True and report.get("passed") is True,
            "benchmark incomplete or not passed")
    require("error" not in report, "benchmark has an error field")
    config, provenance = report.get("configuration"), report.get("provenance")
    require(isinstance(config, dict) and isinstance(provenance, dict), "missing configuration/provenance")
    for field in HASH_FIELDS:
        require(isinstance(provenance.get(field), str) and
                re.fullmatch(r"[0-9a-f]{64}", provenance[field]), f"invalid provenance {field}")
    prompt = integer(config.get("prompt_tokens"), "prompt_tokens", 10_000)
    max_tokens = integer(config.get("max_tokens"), "max_tokens", 2)
    context = integer(config.get("context"), "context", 1)
    require(max_tokens <= 4096 and prompt + max_tokens <= context <= 262144, "context/output limit mismatch")
    require(config.get("order") in ORDERS and type(config.get("warmup")) is bool, "invalid order/warmup")
    require(integer(config.get("prefill_chunk"), "prefill_chunk") == 416 and
            integer(config.get("prefill_eval_layers"), "prefill_eval_layers") == 4 and
            integer(config.get("mtp_depth"), "mtp_depth") == 0, "unexpected benchmark configuration")
    checks = report.get("checks")
    require(isinstance(checks, dict) and checks and all(value is True for value in checks.values()),
            "checks missing, false, or not boolean true")
    required = {f"trial{index}_{suffix}" for index in range(4) for suffix in CHECKS}
    if config["warmup"]:
        required.add("warmup_output_exact")
    require(required <= checks.keys(), "required benchmark checks missing: " + ", ".join(sorted(required - checks.keys())))
    trials, warmups = report.get("trials"), report.get("warmups")
    require(isinstance(trials, list) and len(trials) == 4, "exactly four measured trials required")
    require(isinstance(warmups, list) and len(warmups) == (2 if config["warmup"] else 0), "warmup count mismatch")
    normalized, oracle = [], None
    for index, (trial, mode) in enumerate(zip(trials, ORDERS[config["order"]])):
        require(isinstance(trial, dict), f"trial{index}: missing record")
        require(integer(trial.get("index"), f"trial{index}.index") == index and
                trial.get("kv_append_mode") == mode, f"trial{index}: index/order mismatch")
        sample = validate_result(trial.get("result"), mode, config, max_tokens, f"trial{index}")
        output = ids(trial.get("generated_token_ids"), f"trial{index}.generated_token_ids")
        require(output == trial["result"]["tokens"], f"trial{index}: duplicate ID arrays disagree")
        identity = (output, sample["finish_reason"])
        if oracle is None:
            oracle = identity
        require(identity == oracle, f"trial{index}: complete IDs/finish differ from first measured trial")
        for stage in ("before", "after"):
            budget(trial.get("state_budget_" + stage), f"trial{index}.budget_{stage}")
            memory(trial.get("mlx_memory_" + stage), f"trial{index}.mlx_memory_{stage}")
        sample.update(source=source, trial_index=index, order=config["order"], max_tokens=max_tokens)
        normalized.append(sample)
    require(all(s["runtime_policy"] == normalized[0]["runtime_policy"] for s in normalized),
            "runtime policy changed between timed trials")
    warm_oracle = None
    for index, (warmup, mode) in enumerate(zip(warmups, ("reference", "capacity256"))):
        require(isinstance(warmup, dict) and warmup.get("kv_append_mode") == mode, "warmup order mismatch")
        sample = validate_result(warmup.get("result"), mode, config, min(16, max_tokens), f"warmup{index}")
        output = warmup["result"]["tokens"]
        identity = (output, sample["finish_reason"])
        if warm_oracle is None:
            warm_oracle = identity
        require(identity == warm_oracle, "complete warmup IDs/finish differ")
        require(output == oracle[0][:len(output)], "warmup IDs differ from measured output prefix")
        require(sample["finish_reason"] != "eos" or identity == oracle, "warmup EOS differs from measured finish")
        require(sample["runtime_policy"] == normalized[0]["runtime_policy"], "warmup runtime policy differs")
    cohort = {"configuration": {key: value for key, value in config.items() if key not in ("order", "max_tokens")},
              "provenance": {key: provenance[key] for key in HASH_FIELDS},
              "runtime_policy": normalized[0]["runtime_policy"]}
    cohort_key = json.dumps(cohort, sort_keys=True, separators=(",", ":"))
    return {"source": source, "cohort": cohort, "cohort_key": cohort_key,
            "order": config["order"], "max_tokens": max_tokens, "samples": normalized,
            "tokens": oracle[0], "finish": oracle[1]}


def summary(samples):
    out = {"samples": len(samples), "actual_prefill_tokens": sum(s["actual_prefill_tokens"] for s in samples),
           "actual_decoded_tokens": sum(s["decoded_tokens"] for s in samples), "phases": {}}
    for metric in METRICS:
        elapsed = math.fsum(s["seconds"][metric] for s in samples)
        token_count = out["actual_prefill_tokens" if metric.startswith("prefill") else "actual_decoded_tokens"]
        out["phases"][metric] = {"tokens": token_count, "seconds": elapsed,
                                 "weighted_tokens_per_second": token_count / elapsed,
                                 "mean_request_seconds": elapsed / len(samples)}
    return out


def drift(first, last):
    return {"first_trial_index": first["trial_index"], "last_trial_index": last["trial_index"],
            "phases": {key: {"first_seconds": first["seconds"][key], "last_seconds": last["seconds"][key],
                              "duration_change_percent": 100 * (last["seconds"][key] / first["seconds"][key] - 1),
                              "first_tokens_per_second": first["rates_tokens_per_second"][key],
                              "last_tokens_per_second": last["rates_tokens_per_second"][key],
                              "rate_change_percent": 100 * (last["rates_tokens_per_second"][key] /
                                                             first["rates_tokens_per_second"][key] - 1)}
                       for key in METRICS}}


def analyze(inputs):
    output = {"schema": "qwen-kv-capacity-benchmark-analysis-v1", "valid": False,
              "errors": [], "input_reports": inputs, "groups": [],
              "format_reference": {"file": "GPUKVCapacityBenchmark.swift", "sha256": FORMAT_SOURCE_SHA256,
                                   "scope": "Format source inspected; each actual producer executable hash is retained separately."},
              "metric_definitions": {
                  "prefill_target": "sum(actualForwardTokenCount) / sum(phases.prefill.targetSeconds)",
                  "prefill_active": "sum(actualForwardTokenCount) / sum(phases.prefill.totalSeconds)",
                  "decode_round": "sum(decodedTokenCount) / sum(result.decodeSeconds)",
                  "decode_service": "sum(decodedTokenCount) / sum(phases.decodeServiceSeconds)",
                  "drift": "second occurrence versus first occurrence of the same mode within each four-trial block"},
              "limitations": [
                  "Finite sequential blocks only; no acceptance threshold, default-mode decision, or production-stability conclusion.",
                  "TTFT includes other stages and is preserved only as a raw diagnostic; no TTFT-based decode or prefill rate.",
                  "Raw reports retain every measured trial, full output IDs, checks, warmups, and memory/budget observations. Warmups are not timed-trial samples.",
                  "EOS token membership is attested by the producer finish_contract check; the JSON does not include the model EOS-ID set for an independent membership recheck.",
                  "Timing is wall time with the producer's stated scopes, not GPU-only timing. Workspace is admission accounting; MLX snapshots are global and are not RSS.",
                  "Reports do not capture hardware/thermal/background-load or all environment overrides. Equal recorded provenance cannot establish equal external conditions.",
                  "Grouping keeps order, max_tokens, warmup, recorded configuration, runtime policy, and executable/model/input provenance separate."]}
    parsed, seen_hashes, seen_sources = [], set(), set()
    for item in inputs:
        source = item["source"]
        try:
            require(source not in seen_sources, "duplicate input path")
            seen_sources.add(source)
            if item.get("sha256") is not None:
                require(item["sha256"] not in seen_hashes, "duplicate report bytes would double-count samples")
                seen_hashes.add(item["sha256"])
            require("load_error" not in item, item.get("load_error", "load failure"))
            parsed.append(validate_report(item.get("report"), source))
        except (Invalid, KeyError, TypeError, OverflowError) as error:
            output["errors"].append({"source": source, "message": str(error)})
    # Compare complete outputs across every compatible block, including ABBA vs
    # BAAB; different max_tokens must have identical overlapping output IDs.
    for index, left in enumerate(parsed):
        for right in parsed[index + 1:]:
            if left["cohort_key"] != right["cohort_key"]:
                continue
            a, b = left["tokens"], right["tokens"]
            same = a[:min(len(a), len(b))] == b[:min(len(a), len(b))]
            if left["max_tokens"] == right["max_tokens"]:
                same = same and a == b and left["finish"] == right["finish"]
            if left["finish"] == "eos" or right["finish"] == "eos":
                eos_case, other = (left, right) if left["finish"] == "eos" else (right, left)
                if other["max_tokens"] >= len(eos_case["tokens"]):
                    same = same and other["tokens"] == eos_case["tokens"] and other["finish"] == "eos"
            if not same:
                output["errors"].append({"source": [left["source"], right["source"]],
                                         "message": "compatible reports disagree on full IDs/finish or cross-length output prefix"})
    if output["errors"] or not parsed:
        if not inputs:
            output["errors"].append({"source": None, "message": "no input reports"})
        return output  # No selectively filtered performance summaries.
    groups = {}
    for block in parsed:
        key = (block["cohort_key"], block["order"], block["max_tokens"])
        groups.setdefault(key, []).append(block)
    for (_, order, max_tokens), blocks in sorted(groups.items()):
        samples = [sample for block in blocks for sample in block["samples"]]
        modes = {mode: summary([sample for sample in samples if sample["mode"] == mode])
                 for mode in ("reference", "capacity256")}
        output["groups"].append({"order": order, "max_tokens": max_tokens, "cohort": blocks[0]["cohort"],
                                 "blocks": len(blocks), "samples": samples, "by_mode": modes,
                                 "observed_capacity_over_reference_rate_ratio": {
                                     metric: modes["capacity256"]["phases"][metric]["weighted_tokens_per_second"] /
                                     modes["reference"]["phases"][metric]["weighted_tokens_per_second"] for metric in METRICS},
                                 "within_block_same_mode_drift": [
                                     {"source": block["source"], "mode": mode,
                                      **drift(*[sample for sample in block["samples"] if sample["mode"] == mode])}
                                     for block in blocks for mode in ("reference", "capacity256")]})
    output["valid"] = True
    output["measured_samples"] = sum(len(block["samples"]) for block in parsed)
    return output


def fake_report(order="abba", max_tokens=16, warmup=True):
    """Synthetic CPU control only; never a measured performance record."""
    prompt = 10_400
    config = {"prompt_tokens": prompt, "max_tokens": max_tokens, "context": 16_384,
              "prefill_chunk": 416, "prefill_eval_layers": 4, "mtp_depth": 0,
              "order": order, "warmup": warmup}
    def result(mode, count):
        return {"tokens": list(range(100, 100 + count)), "finishReason": "length",
                "preparationSeconds": 0, "timeToFirstTokenSeconds": 10.2,
                "decodeSeconds": 0.5, "totalSeconds": 11,
                "statistics": {"promptTokenCount": prompt, "generatedTokenCount": count,
                               "decodeRounds": count - 1, "decodedTokenCount": count - 1,
                               "prefillChunkCount": 25, "finalStateOffset": prompt + count - 1,
                               "ssdWaitSeconds": 0.1, "ssdLogicalBytes": 100,
                               "callbackSeconds": 0, "prefillAccumulation": "reference", "mtpDepth": 0},
                "phases": {"prefill": {"promptTokenCount": prompt, "chunkCount": 25,
                                       "targetSeconds": 10, "draftHistorySeconds": 0,
                                       "totalSeconds": 10.1, "ssdWaitSeconds": 0.08,
                                       "ssdLogicalBytes": 80, "evaluateEveryLayers": 4,
                                       "attentionMode": "reference", "suspensionSeconds": 0,
                                       "cachedTokenCount": 0, "computedTokenCount": prompt,
                                       "actualForwardTokenCount": prompt, "recomputedTokenCount": 0,
                                       "cacheLookupSeconds": 0, "cacheRestoreSeconds": 0,
                                       "cacheSaveSeconds": 0, "cacheWaitSeconds": 0, "cacheSource": "cold"},
                           "handoffWaitSeconds": 0.001, "handoffConsumeSeconds": 0.001,
                           "decodeServiceSeconds": 0.6, "decodeSSDWaitSeconds": 0.02,
                           "decodeSSDLogicalBytes": 20, "verificationEvaluateEveryLayers": 4,
                           "decodeKernelMode": "reference", "decodeSuspensionSeconds": 0,
                           "kvAppendMode": mode, "kvCapacityTokenSteps": count - 1 if mode == "capacity256" else 0,
                           "kvCapacityWorkspaceFallbacks": 0,
                           "kvCapacityWorkspacePeakBytes": 4096 if mode == "capacity256" else 0}}
    empty_budget = {"maxBytes": 10000, "requestBytes": 0, "cacheBytes": 0, "workspaceBytes": 0,
                    "totalBytes": 0, "peakBytes": 8000, "rejections": 0, "currentLeases": 0}
    mlx = {"active_bytes": 100, "peak_bytes": 0, "cache_bytes": 10, "limit_bytes": 1000}
    trials = []
    for index, mode in enumerate(ORDERS[order]):
        value = result(mode, max_tokens)
        trials.append({"index": index, "kv_append_mode": mode, "generated_token_ids": value["tokens"][:],
                       "result": value, "state_budget_before": copy.deepcopy(empty_budget),
                       "state_budget_after": copy.deepcopy(empty_budget), "mlx_memory_before": mlx.copy(),
                       "mlx_memory_after": mlx.copy()})
    return {"schema": SCHEMA, "complete": True, "passed": True, "configuration": config,
            "provenance": {key: str(index) * 64 for index, key in enumerate(HASH_FIELDS)},
            "checks": {**{f"trial{i}_{suffix}": True for i in range(4) for suffix in CHECKS},
                       **({"warmup_output_exact": True} if warmup else {})}, "trials": trials,
            "warmups": [{"kv_append_mode": mode, "result": result(mode, min(16, max_tokens))}
                        for mode in ("reference", "capacity256")] if warmup else []}


class SelfTests(unittest.TestCase):
    def run_report(self, report):
        return analyze([{"source": "synthetic-control.json", "report": report}])

    def test_orders_and_lengths_separate(self):
        reports = [{"source": f"synthetic-{order}-{n}", "report": fake_report(order, n)}
                   for order in ORDERS for n in (16, 128, 512)]
        result = analyze(reports)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(len(result["groups"]), 6)
        self.assertEqual(result["measured_samples"], 24)

    def test_weighted_rate_is_ratio_of_sums(self):
        report = fake_report(warmup=False)
        report["trials"][0]["result"]["decodeSeconds"] = 0.1
        report["trials"][3]["result"]["decodeSeconds"] = 0.5
        result = self.run_report(report)
        self.assertTrue(result["valid"], result["errors"])
        group = result["groups"][0]
        self.assertAlmostEqual(group["by_mode"]["reference"]["phases"]["decode_round"]["weighted_tokens_per_second"], 30 / 0.6)
        self.assertEqual(group["within_block_same_mode_drift"][0]["phases"]["decode_round"]["duration_change_percent"], 400)
        self.assertEqual(len(result["input_reports"][0]["report"]["trials"]), 4)

    def test_multiple_same_order_blocks_keep_all_samples(self):
        second = fake_report()
        second["trials"][0]["result"]["decodeSeconds"] = 0.4
        result = analyze([{"source": "one", "report": fake_report()}, {"source": "two", "report": second}])
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(len(result["groups"]), 1)
        self.assertEqual(result["groups"][0]["blocks"], 2)
        self.assertEqual(len(result["groups"][0]["samples"]), 8)
        self.assertEqual(len(result["groups"][0]["within_block_same_mode_drift"]), 4)

    def test_matching_duplicate_id_arrays_still_compare_all_trials(self):
        report = fake_report()
        report["trials"][3]["result"]["tokens"][7] = 900
        report["trials"][3]["generated_token_ids"][7] = 900
        result = self.run_report(report)
        self.assertFalse(result["valid"])
        self.assertIn("complete IDs/finish", result["errors"][0]["message"])

    def test_corruptions_are_never_filtered(self):
        mutations = [
            ("complete", lambda r: r.update(complete=False)),
            ("passed", lambda r: r.update(passed=False)),
            ("failed check", lambda r: r["checks"].update(trial2_output_exact=False)),
            ("numeric check", lambda r: r["checks"].update(trial2_output_exact=1)),
            ("missing check", lambda r: r["checks"].pop("trial2_usable_decode")),
            ("missing trial", lambda r: r["trials"].pop()),
            ("wrong order", lambda r: r["trials"][0].update(kv_append_mode="capacity256")),
            ("wrong index", lambda r: r["trials"][0].update(index=True)),
            ("ID duplicate", lambda r: r["trials"][0]["generated_token_ids"].__setitem__(1, 900)),
            ("ID bool", lambda r: r["trials"][0]["result"]["tokens"].__setitem__(1, True)),
            ("ID fractional", lambda r: r["trials"][0]["result"]["tokens"].__setitem__(1, 1.5)),
            ("finish", lambda r: r["trials"][0]["result"].update(finishReason="stop")),
            ("short length", lambda r: r["trials"][0]["result"]["tokens"].pop()),
            ("decode count", lambda r: r["trials"][0]["result"]["statistics"].update(decodedTokenCount=16)),
            ("round count", lambda r: r["trials"][0]["result"]["statistics"].update(decodeRounds=16)),
            ("offset", lambda r: r["trials"][0]["result"]["statistics"].update(finalStateOffset=1)),
            ("cache", lambda r: r["trials"][0]["result"]["phases"]["prefill"].update(cachedTokenCount=416)),
            ("forward count", lambda r: r["trials"][0]["result"]["phases"]["prefill"].update(actualForwardTokenCount=1)),
            ("nan", lambda r: r["trials"][0]["result"]["phases"].update(decodeServiceSeconds=float("nan"))),
            ("negative", lambda r: r["trials"][0]["result"]["phases"].update(handoffWaitSeconds=-1)),
            ("zero round", lambda r: r["trials"][0]["result"].update(decodeSeconds=0)),
            ("scope", lambda r: r["trials"][0]["result"]["phases"].update(decodeServiceSeconds=0.01)),
            ("missing phases", lambda r: r["trials"][0]["result"].pop("phases")),
            ("mode", lambda r: r["trials"][0]["result"]["phases"].update(kvAppendMode="capacity256")),
            ("steps", lambda r: r["trials"][1]["result"]["phases"].update(kvCapacityTokenSteps=1)),
            ("fallback", lambda r: r["trials"][1]["result"]["phases"].update(kvCapacityWorkspaceFallbacks=1)),
            ("workspace", lambda r: r["trials"][1]["result"]["phases"].update(kvCapacityWorkspacePeakBytes=0)),
            ("lease", lambda r: r["trials"][0]["state_budget_after"].update(currentLeases=1)),
            ("provenance", lambda r: r["provenance"].update(input_sha256="missing")),
            ("warmup", lambda r: r["warmups"][1]["result"]["tokens"].__setitem__(0, 900)),
        ]
        for label, mutate in mutations:
            with self.subTest(label=label):
                report = fake_report()
                mutate(report)
                result = self.run_report(report)
                self.assertFalse(result["valid"])
                self.assertEqual(result["groups"], [])
                self.assertTrue(result["errors"])

    def test_cross_report_and_cross_length_ids(self):
        for length in (16, 128):
            with self.subTest(length=length):
                wrong = fake_report("baab", length, False)
                for trial in wrong["trials"]:
                    trial["result"]["tokens"][3] = 900
                    trial["generated_token_ids"][3] = 900
                result = analyze([{"source": "one", "report": fake_report(warmup=False)},
                                  {"source": "two", "report": wrong}])
                self.assertFalse(result["valid"])
                self.assertEqual(result["groups"], [])

    def test_eos_finish_and_short_outputs(self):
        report = fake_report(max_tokens=128, warmup=False)
        for trial in report["trials"]:
            value = trial["result"]
            value["tokens"] = value["tokens"][:12]
            value["finishReason"] = "eos"
            value["statistics"].update(generatedTokenCount=12, decodedTokenCount=11,
                                       decodeRounds=11, finalStateOffset=10_411)
            value["phases"]["kvCapacityTokenSteps"] = 11 if trial["kv_append_mode"] == "capacity256" else 0
            trial["generated_token_ids"] = value["tokens"][:]
        result = self.run_report(report)
        self.assertTrue(result["valid"], result["errors"])
        self.assertEqual(result["groups"][0]["by_mode"]["reference"]["actual_decoded_tokens"], 22)

    def test_duplicate_input_and_json_rejection(self):
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'):
            with self.subTest(text=text), self.assertRaises(Invalid):
                load_json(text)
        item = {"source": "one", "sha256": "same", "report": fake_report()}
        self.assertFalse(analyze([item, dict(item, source="two")])["valid"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument("--output", type=Path, help="New JSON output path; existing files are not overwritten")
    parser.add_argument("--self-test", action="store_true", help="Run synthetic CPU controls only")
    args = parser.parse_args()
    if args.self_test:
        require(not args.files and args.output is None, "--self-test cannot consume reports or write an analysis")
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SelfTests))
        return 0 if result.wasSuccessful() else 2
    if not args.files:
        parser.error("provide at least one benchmark report, or --self-test")
    if len(args.files) > 64:
        parser.error("at most 64 reports per analysis")
    inputs = []
    for path in args.files:
        item = {"source": str(path.resolve())}
        try:
            require(path.stat().st_size <= MAX_INPUT_BYTES, "input exceeds 16 MiB bound")
            data = path.read_bytes()
            require(len(data) <= MAX_INPUT_BYTES, "input grew beyond 16 MiB bound")
            item.update(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data), report=load_json(data))
        except (OSError, ValueError) as error:
            item["load_error"] = str(error)
        inputs.append(item)
    result = analyze(inputs)
    rendered = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(rendered)
    else:
        sys.stdout.write(rendered)
    if not result["valid"]:
        print(f"Invalid input: {len(result['errors'])} issue(s); no aggregate performance conclusion.", file=sys.stderr)
        return 2
    print(f"Validated {len(inputs)} block(s), {result['measured_samples']} measured samples; "
          f"{len(result['groups'])} separate order/output/cohort group(s).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Invalid, OSError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
