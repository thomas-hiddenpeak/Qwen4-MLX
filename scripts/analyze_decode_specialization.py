#!/usr/bin/env python3
"""Offline analysis of one generate-gpu interleave report; no device access.

Rates use sum(decode_steps) / sum(decode_step_seconds), excluding TTFT/load/
wired setup. Logical weight-footprint rates are NOT physical DRAM bandwidth.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


MODES = {"reference", "scalar", "elementwise", "projections", "all"}
POLICIES = {"disabled", "fit"}
MEMORY_FIELDS = ("active_bytes", "cache_bytes", "peak_bytes", "limit_bytes")


def required(obj, key, where):
    if not isinstance(obj, dict) or key not in obj:
        raise ValueError(f"{where}: missing required field {key!r}; a current interleave report is required")
    return obj[key]


def integer(value, where, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{where}: expected integer >= {minimum}")
    return value


def number(value, where, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f"{where}: expected finite {'positive' if positive else 'nonnegative'} number")
    return value


def token_ids(value, where):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{where}: expected complete nonempty token-ID list")
    for i, token in enumerate(value):
        integer(token, f"{where}[{i}]")
    return value


def memory(value, where):
    return {key: integer(required(value, key, where), f"{where}.{key}") for key in MEMORY_FIELDS}


def group_key(row):
    return row["decode_mode"], row["wired_policy"]


def label(key):
    return {"decode_mode": key[0], "wired_policy": key[1]}


def summarize(rows, additional_bytes):
    steps = sum(r["decode_steps"] for r in rows)
    seconds = math.fsum(r["decode_seconds"] for r in rows)
    logical_bytes = sum(r["logical_weight_bytes"] for r in rows)
    return {
        **label(group_key(rows[0])),
        "trial_indices": [r["trial_index"] for r in rows],
        "repetitions": [r["repetition"] for r in rows],
        "trial_count": len(rows),
        "tokens_per_second": [r["tokens_per_second"] for r in rows],
        "median_tokens_per_second": statistics.median(r["tokens_per_second"] for r in rows),
        "aggregate_decode_steps": steps,
        "aggregate_decode_seconds": seconds,
        "aggregate_tokens_per_second": steps / seconds,
        "median_ttft_seconds": statistics.median(r["ttft_seconds"] for r in rows),
        "memory": {key: {"values": [r["memory"][key] for r in rows],
                         "median": statistics.median(r["memory"][key] for r in rows),
                         "maximum": max(r["memory"][key] for r in rows)} for key in MEMORY_FIELDS},
        "wired_setup_milliseconds": [r["wired_setup_milliseconds"] for r in rows],
        "wired_target_limit_bytes": [r["wired_target_limit_bytes"] for r in rows],
        "additional_projection_buffer_bytes_in_process": additional_bytes,
        "aggregate_logical_weight_footprint_bytes": logical_bytes,
        "aggregate_logical_weight_footprint_gbps": logical_bytes / seconds / 1e9,
        "physical_dram_bandwidth_gbps": None,
    }


def paired_windows(rows, additional_bytes, eligible, baseline_mode, baseline_wired):
    pairs = []
    for start in range(max(0, len(rows) - 3)):
        window = rows[start:start + 4]
        keys = [group_key(r) for r in window]
        if keys[0] != keys[3] or keys[1] != keys[2] or keys[0] == keys[1]:
            continue
        changed = [i for i in (0, 1) if keys[0][i] != keys[1][i]]
        item = {"trial_indices": [r["trial_index"] for r in window],
                "sequence": [label(k) for k in keys],
                "rate_ratio": None, "percent_delta": None}
        if len(changed) != 1:
            pairs.append({**item, "status": "rejected_two_dimensions_changed"})
            continue
        dimension = changed[0]
        baseline_value = baseline_mode if dimension == 0 else baseline_wired
        baseline = next((k for k in (keys[0], keys[1]) if k[dimension] == baseline_value), None)
        groups = [summarize([r for r in window if group_key(r) == k], additional_bytes)
                  for k in (keys[0], keys[1])]
        item.update({"changed_dimension": "decode_mode" if dimension == 0 else "wired_policy", "groups": groups})
        if baseline is None:
            item["status"] = "requested_baseline_absent"
        else:
            candidate = keys[1] if baseline == keys[0] else keys[0]
            item.update({"baseline": label(baseline), "candidate": label(candidate),
                         "pattern": "ABBA" if keys[0] == baseline else "BAAB"})
            if eligible:
                base = next(g for g in groups if (g["decode_mode"], g["wired_policy"]) == baseline)
                cand = next(g for g in groups if (g["decode_mode"], g["wired_policy"]) == candidate)
                ratio = cand["aggregate_tokens_per_second"] / base["aggregate_tokens_per_second"]
                item.update({"status": "descriptive_comparison", "rate_ratio": ratio, "percent_delta": (ratio - 1) * 100})
            else:
                item["status"] = "ineligible_for_unprofiled_comparison"
        pairs.append(item)
    return pairs


def analyze(data, skip_first=0, baseline_mode="reference", baseline_wired="disabled"):
    integer(skip_first, "skip_first")
    if baseline_mode not in {"reference", "scalar"} or baseline_wired not in POLICIES:
        raise ValueError("Unsupported baseline mode or wired policy")
    trials = required(data, "trials", "report")
    if not isinstance(trials, list) or not trials or skip_first >= len(trials):
        raise ValueError("At least one trial must remain after --skip-first")
    context = integer(required(data, "context_limit", "report"), "context_limit", 1)
    maximum_tokens = integer(required(data, "max_tokens", "report"), "max_tokens", 1)
    count = integer(required(data, "requested_repetitions", "report"), "requested_repetitions", 1)
    if count != len(trials):
        raise ValueError("requested_repetitions does not match the complete trial list")
    decode_order = required(data, "decode_order", "report")
    wired_order = required(data, "wired_order", "report")
    additional = integer(required(data, "additional_projection_buffer_bytes", "report"), "additional_projection_buffer_bytes")
    loaded_memory = memory(required(data, "loaded_memory", "report"), "loaded_memory")
    profiler = required(data, "profiler", "report")
    telemetry = required(data, "telemetry", "report")
    required(data, "provenance", "report")
    failures = []

    def check(condition, message):
        if not condition:
            failures.append(message)

    check(required(data, "mtp_enabled", "report") is False, "Root MTP must be explicitly false")
    check(required(data, "mtp_weights_loaded", "report") is False, "MTP weights must be explicitly unloaded")
    check(required(profiler, "mode", "profiler") == "disabled", "Profiler must be disabled")
    check(required(profiler, "stages", "profiler") == [], "Disabled profiler must have no stage records")
    check(required(profiler, "droppedRecords", "profiler") == 0, "Profiler must have no dropped records")
    check(required(telemetry, "enabled", "telemetry") is False, "Telemetry must be explicitly false")
    # Reports predating command timing have no such field. When the producer
    # includes it, only an explicitly disabled timer is an unprofiled run.
    if "gpu_command_timing" in data:
        check(required(data["gpu_command_timing"], "enabled", "gpu_command_timing") is False,
              "GPU command timing must be disabled for unprofiled comparisons")
    first_prompt = token_ids(required(trials[0], "prompt_tokens", "trial[0]"), "trial[0].prompt_tokens")
    first_generated = token_ids(required(trials[0], "generated_token_ids", "trial[0]"), "trial[0].generated_token_ids")
    prompts_equal, generated_equal, lengths_equal = True, True, True
    rows = []
    settings = []
    for i, trial in enumerate(trials):
        where = f"trial[{i}]"
        mode = required(trial, "decode_mode", where)
        wired = required(trial, "wired_memory", where)
        policy = required(wired, "policy", where + ".wired_memory")
        if mode not in MODES or policy not in POLICIES:
            raise ValueError(f"{where}: unknown decode mode or wired policy")
        repetition = integer(required(trial, "repetition", where), where + ".repetition")
        if repetition != i:
            raise ValueError(f"{where}: expected original, ordered repetition {i}")
        prompt = token_ids(required(trial, "prompt_tokens", where), where + ".prompt_tokens")
        generated = token_ids(required(trial, "generated_token_ids", where), where + ".generated_token_ids")
        prompts_equal &= prompt == first_prompt
        generated_equal &= generated == first_generated
        lengths_equal &= len(generated) == len(first_generated)
        check(len(prompt) + maximum_tokens <= context, f"{where}: prompt/max_tokens exceed context_limit")
        check(len(generated) <= maximum_tokens, f"{where}: generated length exceeds max_tokens")
        check(required(trial, "mtp_enabled", where) is False, f"{where}: MTP must be explicitly false")
        check(required(trial, "prefill_accumulation", where) == "reference", f"{where}: prefill accumulation must be reference")
        check(required(trial, "sampling", where) == "greedy", f"{where}: sampling must be greedy")
        check(required(trial, "final_prompt_token_held_back", where) is True, f"{where}: held-back token convention must be true")
        check(trial.get("context_limit", context) == context, f"{where}: context_limit changed")
        check(trial.get("max_tokens", maximum_tokens) == maximum_tokens, f"{where}: max_tokens changed")
        chunk = integer(required(trial, "prefill_chunk", where), where + ".prefill_chunk", 1)
        workers = integer(required(trial, "ssd_workers", where), where + ".ssd_workers", 1)
        settings.append((chunk, workers, required(trial, "finish_reason", where)))
        steps = integer(required(trial, "decode_steps", where), where + ".decode_steps", 1)
        durations = required(trial, "decode_step_seconds", where)
        if not isinstance(durations, list) or len(durations) != steps or steps != len(generated) - 1:
            raise ValueError(f"{where}: complete decode durations and generated IDs must satisfy steps = generated length - 1")
        seconds = math.fsum(number(v, where + ".decode_step_seconds", True) for v in durations)
        rate = steps / seconds
        reported_rate = number(required(trial, "decode_tokens_per_second", where), where + ".decode_tokens_per_second", True)
        if not math.isclose(rate, reported_rate, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"{where}: reported decode rate disagrees with steps / summed durations")
        logical = integer(required(trial, "logical_decode_weight_bytes_per_token", where), where + ".logical_decode_weight_bytes_per_token", 1)
        reported_logical = number(required(trial, "logical_decode_weight_footprint_rate_gbps", where), where + ".logical_decode_weight_footprint_rate_gbps", True)
        if not math.isclose(logical * rate / 1e9, reported_logical, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"{where}: logical rate disagrees with the footprint ledger")
        check(required(trial, "final_state_offset", where) == len(prompt) + steps, f"{where}: final state offset differs from the prompt/decode window")
        rows.append({"trial_index": i, "repetition": repetition, "decode_mode": mode,
                     "wired_policy": policy, "decode_steps": steps, "decode_seconds": seconds,
                     "tokens_per_second": rate,
                     "ttft_seconds": number(required(trial, "time_to_first_token_seconds_excluding_load", where), where + ".ttft", True),
                     "memory": memory(required(trial, "memory", where), where + ".memory"),
                     "wired_setup_milliseconds": number(required(wired, "setupDurationMilliseconds", where), where + ".wired_setup"),
                     "wired_target_limit_bytes": integer(required(wired, "targetLimitBytes", where), where + ".wired_target"),
                     "logical_weight_bytes": logical * steps})
    if decode_order != [r["decode_mode"] for r in rows] or wired_order != [r["wired_policy"] for r in rows]:
        raise ValueError("Root decode_order/wired_order do not match the complete ordered trials")
    check(prompts_equal, "Prompt token IDs differ across trials")
    check(lengths_equal, "Generated lengths differ across trials")
    check(generated_equal, "Complete generated token IDs differ across trials, including skipped warmups")
    check(all(s == settings[0] for s in settings), "Prefill chunk, SSD workers, or finish reason changed")
    eligible = not failures
    retained = rows[skip_first:]
    grouped = {}
    for row in retained:
        grouped.setdefault(group_key(row), []).append(row)
    groups = [summarize(group, additional) for group in grouped.values()]
    comparisons = []
    for group in groups:
        baseline = next((g for g in groups if g["decode_mode"] == baseline_mode and g["wired_policy"] == group["wired_policy"]), None)
        result = {**label((group["decode_mode"], group["wired_policy"])), "baseline_mode": baseline_mode,
                  "rate_ratio": None, "percent_delta": None}
        if baseline is None:
            result["status"] = "baseline_absent_for_this_wired_policy"
        elif not eligible:
            result["status"] = "ineligible_for_unprofiled_comparison"
        else:
            ratio = group["aggregate_tokens_per_second"] / baseline["aggregate_tokens_per_second"]
            result.update({"status": "baseline" if group is baseline else "descriptive_comparison",
                           "rate_ratio": ratio, "percent_delta": (ratio - 1) * 100})
        comparisons.append(result)
    return {"schema_version": 1, "eligible_for_unprofiled_comparison": eligible,
            "validation_failures": failures, "performance_pass": None,
            "skip_first": skip_first, "excluded_trial_indices": list(range(skip_first)),
            "baseline_mode": baseline_mode, "baseline_wired_policy": baseline_wired,
            "prompt_token_ids": first_prompt, "prompt_ids_equal": prompts_equal,
            "generated_token_ids_from_first_trial": first_generated,
            "generated_lengths_equal": lengths_equal, "all_exact_generated_token_equality": generated_equal,
            "context_limit": context, "max_tokens": maximum_tokens,
            "loaded_memory": loaded_memory, "additional_projection_buffer_bytes_in_process": additional,
            "provenance": data["provenance"], "scheduler_environment": required(data, "scheduler_environment", "report"),
            "groups": groups, "baseline_comparisons": comparisons,
            "adjacent_four_trial_pairs": paired_windows(retained, additional, eligible, baseline_mode, baseline_wired),
            "physical_dram_bandwidth_gbps": None,
            "notes": ["Descriptive rates only; no automatic performance acceptance decision or significance claim.",
                      "Decode timing is sum(decode_step_seconds); load, TTFT and wired setup are excluded.",
                      "Logical GB/s is weight-footprint bytes / decode seconds / 1e9, not physical DRAM traffic or measured bandwidth.",
                      "Allocator peak memory is process-cumulative; projection buffers are allocated once for the process and shared by all trial modes.",
                      "Adjacent four-trial windows slide by one and can overlap; their gains are not independent observations.",
                      "ABBA/BAAB pairs change only decode mode OR wired policy. A denotes the requested baseline for that dimension.",
                      "Exact token checks include skipped warmups. Single-process scope relies on the producer's one-report contract; no independent process trace is inferred."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-first", type=int, default=0)
    parser.add_argument("--baseline-mode", choices=("reference", "scalar"), default="reference")
    parser.add_argument("--baseline-wired-policy", choices=("disabled", "fit"), default="disabled")
    args = parser.parse_args()
    try:
        if args.output and args.output.resolve() == args.input.resolve():
            raise ValueError("Output must not overwrite the input report")
        raw = args.input.read_bytes()
        result = analyze(json.loads(raw), args.skip_first, args.baseline_mode, args.baseline_wired_policy)
        result["source"] = {"path": str(args.input.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}
        rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered)
        else:
            sys.stdout.write(rendered)
    except (ValueError, TypeError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
