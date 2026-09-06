#!/usr/bin/env python3
"""CPU-only MTP window audit from an explicit frozen plan and generation reports.

No discovery of replacement reports, inference, controller, or plan mutation.
--plan may be repeated to compare the two separately recorded windows.
--validate-plan-only also checks the current source/library/payload snapshot;
normal historical analysis checks report provenance, not today's executable.
This tests the stated decode performance gate, never the whole release/soak gate.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

from check_agent_workload import check_text, fixture, strict_json
from analyze_gpu_telemetry import build_phases


GATE = {"minimum_decode_ratio": 1.10, "maximum_ar_decode_drift": 0.05,
        "groups_per_case": 3, "windows_required": 2}
ORDERS = [[0, 2, 0, 2, 2, 0, 0, 2, 2, 0], [0, 2, 0, 2, 2, 0]]


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return strict_json(Path(path).read_text())


def utc(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None, "UTC timestamp must include timezone")
    return parsed


def integer(value):
    return type(value) is int and value >= 0


def ids(value):
    return type(value) is list and bool(value) and all(integer(x) for x in value)


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def field(value, dotted):
    for key in dotted.split("."):
        value = value[key]
    return value


def same(actual, expected):
    # bool is not an integer field, even though Python's == normally permits it.
    return type(actual) is type(expected) and actual == expected


def file_spec(spec):
    path = Path(spec["path"])
    require(sha(path) == spec["sha256"], f"Changed frozen file: {path}")
    return path


def close(a, b):
    return number(a) and number(b) and math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-8)


def validate_plan(plan, current=False):
    require(plan["schema"] == "mtp-release-window-plan-v1", "Unknown plan schema")
    require(plan["status"] in ("draft", "frozen"), "Plan must be draft or frozen")
    require(plan["gate"] == GATE, "Release thresholds/group counts were changed")
    require(plan["window_id"] in ("a", "b"), "Expected window a or b")
    if plan["status"] == "frozen":
        utc(plan["freeze_utc"])
        require(plan.get("reference_ledger_confirmed") is True,
                "Frozen plan needs root-confirmed predecessor ledger")
    require(len(plan["cases"]) == 6, "Expected all three tasks and both budgets")
    require({(c["task"], c["budget"]) for c in plan["cases"]} ==
            {(t, b) for t in ("original", "tools", "facts") for b in (128, 256)},
            "Missing or duplicate task/budget")
    require([c["id"] for c in plan["cases"]] == plan["case_order"], "Case order changed")
    process_ids, report_paths = [], []
    for case in plan["cases"]:
        require(len(case["processes"]) == 2, "Each case requires ten and six trial processes")
        for part, process in enumerate(case["processes"]):
            order = ORDERS[part]
            require(process["mtp_order"] == order, "Changed trial order")
            require(process["warmup_indices"] == [0, 1], "Changed warmup indices")
            wanted = [[2, 3, 4, 5], [6, 7, 8, 9]] if part == 0 else [[2, 3, 4, 5]]
            require(process["groups"] == wanted, "Changed group partition")
            require(process["group_numbers"] == ([1, 2] if part == 0 else [3]),
                    "Changed group numbers")
            command = process["command"]
            for option, value in {"--mtp-order": ",".join(map(str, order)),
                                  "--max-tokens": str(case["budget"]),
                                  "--output": process["report"],
                                  "--tokens-file": case["input"]["path"]}.items():
                require(command.count(option) == 1 and command[command.index(option)+1] == value,
                        f"Plan argv mismatch: {option}")
            process_ids.append(process["id"])
            report_paths.append(str(Path(process["report"]).resolve()))
        require(case["functional"]["kind"] ==
                ("historical_text_regression" if case["task"] == "original" else "frozen_json"),
                "Functional contract changed")
    require(len(set(process_ids)) == 12 and len(set(report_paths)) == 12,
            "Duplicate process ID or report path")
    if current:
        for spec in plan["identity"]["files"]:
            file_spec(spec)
        for spec in plan["identity"]["payload_stat_inventory"]:
            actual = Path(spec["path"]).stat()
            require(actual.st_size == spec["bytes"] and actual.st_mtime_ns == spec["mtime_ns"],
                    f"Model payload stat changed: {spec['path']}")


def frozen_case(case, identity):
    input_ids = load(file_spec(case["input"]))
    require(ids(input_ids) and len(input_ids) == case["input_tokens"], "Invalid frozen input")
    golden = load(file_spec(case["golden"]))
    require(golden["model_directory"] == identity["model_directory"], "Golden model differs")
    require(golden["provenance"]["model_metadata_sha256"] ==
            identity["report_provenance"]["model_metadata_sha256"], "Golden model metadata differs")
    trial = golden["trials"][case["golden"]["trial_index"]]
    require(trial["mtp_depth"] == 0 and trial["prompt_tokens"] == input_ids,
            "Golden must be AR on the frozen input")
    require(golden["max_tokens"] == case["budget"] and ids(trial["generated_token_ids"]),
            "Golden budget/output differs")
    functional = case["functional"]
    if functional["kind"] == "frozen_json":
        file_spec(functional["manifest"])
        _, _, _, fixture_ids, expected, schema = fixture(Path(functional["fixture_directory"]))
        require(fixture_ids == input_ids, "Fixture input differs")
        check_text(trial["text"], expected, schema)
    else:
        expected, schema = None, None
    return input_ids, trial, expected, schema


def check_trial(trial, index, depth, case, frozen, config, eos):
    input_ids, golden, expected, schema = frozen
    checks, errors = {}, []

    def check(name, ok):
        checks[name] = bool(ok)
        if not ok:
            errors.append(name)

    check("trial_order", same(trial.get("repetition"), index) and same(trial.get("mtp_depth"), depth))
    check("configuration", all(same(field(trial, k), v) for k, v in config.items()) and
          same(trial["mtp_enabled"], depth > 0))
    check("input_ids", ids(trial.get("prompt_tokens")) and trial["prompt_tokens"] == input_ids)
    output = trial.get("generated_token_ids")
    valid_ids = ids(output)
    check("output_id_types", valid_ids)
    check("exact_ids_vs_frozen_ar", valid_ids and output == golden["generated_token_ids"])
    count = len(output) if valid_ids else 0
    finish = trial.get("finish_reason")
    eos_ok = valid_ids and 0 < count <= case["budget"] and not set(output[:-1]).intersection(eos)
    eos_ok = eos_ok and ((finish == "eos" and output[-1] in eos) or
                        (finish == "length" and count == case["budget"] and output[-1] not in eos))
    check("finish_contract", eos_ok)
    check("finish_vs_frozen_ar", finish == golden["finish_reason"])
    offset = trial.get("final_state_offset")
    delta = offset - (len(input_ids) + count - 1) if integer(offset) and valid_ids else None
    check("terminal_offset", delta in ([0, 1] if depth and finish == "eos" else [0]))
    functional_error = None
    try:
        if expected is not None:
            check_text(trial["text"], expected, schema)
        else:
            require(trial["text"] == golden["text"], "Historical prose text differs")
    except (ValueError, TypeError, KeyError) as exc:
        functional_error = str(exc)
    check("functional", functional_error is None)
    raw_steps = trial.get("decode_step_seconds")
    timings_ok = type(raw_steps) is list and all(number(t) and t > 0 for t in raw_steps)
    seconds = math.fsum(raw_steps) if timings_ok else None
    decode_count = max(0, count - 1)
    rate = decode_count / seconds if timings_ok and seconds > 0 and decode_count else None
    phase = trial["phase_metrics"]
    phase_ok = all(number(phase.get(k)) for k in
                   ("prefill_target_seconds", "prefill_target_tokens_per_second", "prefill_total_seconds",
                    "mtp_prompt_history_seconds", "decode_seconds", "decode_mean_seconds_per_token"))
    phase_ok = phase_ok and phase["prefill_target_seconds"] > 0 and close(
        phase["prefill_target_tokens_per_second"], len(input_ids) / phase["prefill_target_seconds"])
    phase_ok = phase_ok and phase["prefill_total_seconds"] + 1e-8 >= (
        phase["prefill_target_seconds"] + phase["mtp_prompt_history_seconds"])
    phase_ok = phase_ok and close(phase["decode_seconds"], seconds)
    if rate is not None:
        phase_ok = phase_ok and close(trial["decode_tokens_per_second"], rate) and close(
            phase["decode_mean_seconds_per_token"], 1 / rate)
    check("timing_consistency", timings_ok and phase_ok and same(trial["decode_steps"], len(raw_steps)))
    cost = trial.get("mtp_cost_summary")
    if depth:
        cost_ok = type(cost) is dict and all(cost.get(k) is True for k in
            ("countersConsistent", "acceptanceHistogramConsistent", "componentsFitDecodeWindow"))
        if cost_ok:
            components = [cost.get(k) for k in ("draftSeconds", "verificationSeconds",
                          "commitOrReplaySeconds", "historySeconds", "decodeSecondsOutsideComponents")]
            cost_ok = all(number(v) for v in components) and close(math.fsum(components), seconds)
            cost_ok = cost_ok and same(cost["committedDecodeTokens"], decode_count)
            cost_ok = cost_ok and same(cost["decodeSteps"], len(raw_steps)) and close(cost["decodeSeconds"], seconds)
        check("mtp_decode_cost", cost_ok)
    else:
        check("ar_has_no_mtp_cost", cost is None and phase["mtp_prompt_history_seconds"] == 0)
    memory = trial["memory"]
    check("memory_counter_types", all(integer(memory.get(k)) for k in ("active_bytes", "peak_bytes")))
    return {"repetition": index, "mtp_depth": depth, "checks": checks, "passed": all(checks.values()),
            "errors": errors, "functional_error": functional_error,
            "functional_scope": case["functional"]["kind"],
            "output_tokens_including_eos": count, "decode_token_count_excluding_first": decode_count,
            "generated_token_ids": output, "finish_reason": finish,
            "terminal_offset": offset, "terminal_offset_delta_vs_last_pending": delta,
            "decode_seconds": seconds, "decode_tokens_per_second": rate,
            "phase_metrics": phase, "mtp_cost_summary": cost, "memory": memory,
            "time_to_first_token_seconds_excluding_load": trial["time_to_first_token_seconds_excluding_load"],
            "request_seconds_excluding_load": trial["request_seconds_excluding_load"]}


def summarize_group(trials, group_number, process_valid):
    ar, mtp = [trials[i] for i in (0, 3)], [trials[i] for i in (1, 2)]

    def aggregate(rows):
        count = sum(t["decode_token_count_excluding_first"] for t in rows)
        good = all(number(t["decode_seconds"]) and t["decode_seconds"] > 0 for t in rows)
        elapsed = sum(t["decode_seconds"] for t in rows) if good else None
        rate = count / elapsed if good and count else None
        return {"actual_decode_tokens": count, "decode_seconds": elapsed,
                "token_s": rate, "mean_tpot_seconds": 1 / rate if rate else None,
                "prefill_target_seconds": [t["phase_metrics"]["prefill_target_seconds"] for t in rows],
                "mtp_prompt_history_seconds": [t["phase_metrics"]["mtp_prompt_history_seconds"] for t in rows],
                "prefill_total_seconds": [t["phase_metrics"]["prefill_total_seconds"] for t in rows],
                "decode_head_history_seconds": [(t["mtp_cost_summary"] or {}).get("historySeconds", 0) for t in rows]}

    a, m = aggregate(ar), aggregate(mtp)
    first, last = ar[0]["decode_tokens_per_second"], ar[1]["decode_tokens_per_second"]
    drift = abs(last - first) / first if first and last else None
    ratio = m["token_s"] / a["token_s"] if a["token_s"] and m["token_s"] else None
    prefill = [t["phase_metrics"]["prefill_target_tokens_per_second"] for t in ar]
    prefill_drift = abs(prefill[-1] - prefill[0]) / prefill[0] if all(prefill) else None
    correct = process_valid and all(t["passed"] for t in trials)
    if not correct:
        status = "invalid_correctness_or_identity"
    elif drift is None or ratio is None:
        status = "indeterminate_no_decode"
    elif drift > GATE["maximum_ar_decode_drift"]:
        status = "indeterminate_drift"
    elif ratio < GATE["minimum_decode_ratio"]:
        status = "failed_decode_ratio"
    else:
        status = "passed_group_decode_gate"
    return {"group": group_number, "repetitions": [t["repetition"] for t in trials],
            "status": status, "ar": a, "mtp": m, "mtp_over_ar": ratio,
            "ar_decode_drift_fraction": drift, "ar_prefill_target_drift_fraction": prefill_drift,
            "prefill_drift_does_not_decide_decode": True}


def validated_phases(generation):
    phases, _ = build_phases(generation)
    trials = generation["trials"]
    expected = [("load", None)] + [(name, i) for i in range(len(trials)) for name in ("prefill", "decode")]
    require([(p["phase"], p["repetition"]) for p in phases] == expected,
            "Missing/extra/reordered request phase envelopes")
    for phase in phases[1:]:
        trial, name = trials[phase["repetition"]], phase["phase"]
        events = phase["events"]
        count = len(trial["prefill_chunk_seconds"]) if name == "prefill" else trial["decode_steps"]
        require([e["step_index"] for e in events] == list(range(count)), "Phase leaf steps differ from trial")
        require(all(integer(e["input_tokens"]) and integer(e["output_tokens"]) for e in events),
                "Invalid phase token counters")
        if name == "prefill":
            require(sum(e["input_tokens"] for e in events) == len(trial["prompt_tokens"]) and
                    sum(e["output_tokens"] for e in events) == 1, "Prefill phase token counts differ")
        else:
            require(sum(e["output_tokens"] for e in events) == len(trial["generated_token_ids"]) - 1,
                    "Decode phase token count differs from emitted tokens")
    return phases


def analyze_window(path):
    plan = load(path)
    validate_plan(plan)
    result = {"plan": str(path.resolve()), "plan_sha256": sha(path), "window_id": plan["window_id"],
              "plan_status": plan["status"], "identity": plan["identity"], "case_order": plan["case_order"],
              "cases": [], "errors": [], "complete": True, "all_correct": True}
    seen_pids, captures, monotonic_windows = set(), [], []
    for case in plan["cases"]:
        c = {"id": case["id"], "task": case["task"], "budget": case["budget"],
             "input_tokens": case["input_tokens"], "functional_scope": case["functional"],
             "processes": [], "groups": [], "errors": []}
        result["cases"].append(c)
        try:
            frozen = frozen_case(case, plan["identity"])
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            c["errors"].append(f"Frozen input/golden/functional contract: {exc}")
            c["all_three_groups_passed"] = False
            result["complete"] = result["all_correct"] = False
            for process in case["processes"]:
                p = Path(process["report"])
                entry = {"id": process["id"], "path": str(p), "errors": ["invalid_frozen_contract"]}
                c["processes"].append(entry)
                try:
                    raw = load(p)
                    entry["sha256"] = sha(p)
                    entry["raw_trials_after_invalid_report"] = raw.get("trials", [])
                except (OSError, ValueError, TypeError, AttributeError) as error:
                    entry["errors"].append(str(error))
            continue
        for process in case["processes"]:
            p = Path(process["report"])
            entry = {"id": process["id"], "path": str(p), "errors": [], "trials": []}
            c["processes"].append(entry)
            if not p.is_file():
                entry["errors"].append("missing_report")
                result["complete"] = result["all_correct"] = False
                continue
            entry["sha256"] = sha(p)
            g = None
            try:
                g = load(p)
                require(g["model_directory"] == plan["identity"]["model_directory"], "Wrong model directory")
                for key, expected in plan["identity"]["report_provenance"].items():
                    require(same(g["provenance"][key], expected), f"Provenance mismatch: {key}")
                for key, expected in plan["config"]["report_fields"].items():
                    require(same(field(g, key), expected), f"Report config mismatch: {key}")
                require(g["max_tokens"] == case["budget"] and g["mtp_order"] == process["mtp_order"], "Budget/order differs")
                pid = g["provenance"]["process_id"]
                require(integer(pid) and pid not in seen_pids, "Duplicated/invalid process PID")
                seen_pids.add(pid)
                captured = utc(g["provenance"]["captured_before_model_load_utc"])
                captures.append(captured)
                if plan["status"] == "frozen":
                    require(utc(plan["freeze_utc"]) <= captured, "Plan frozen after run began")
                require(len(g["trials"]) == len(process["mtp_order"]) == g["requested_repetitions"], "Incomplete/extra trials")
                phases = validated_phases(g)
                envelope = [g["telemetry"]["started_ns"], max(x["end_ns"] for x in phases)]
                require(all(integer(x) for x in envelope) and envelope[0] < envelope[1], "Invalid process phase envelope")
                require(g["telemetry"]["target_pid"] == pid and
                        g["telemetry"]["clock"] == "mach_absolute_time_nanoseconds", "Telemetry identity differs")
                if monotonic_windows:
                    require(monotonic_windows[-1][1] < envelope[0], "Processes overlap or differ from planned order")
                monotonic_windows.append(envelope)
                entry["phase_envelope_ns"] = envelope
                for i, (trial, depth) in enumerate(zip(g["trials"], process["mtp_order"])):
                    checked = check_trial(trial, i, depth, case, frozen, plan["config"]["trial_fields"], set(plan["eos_token_ids"]))
                    checked["warmup"] = i in process["warmup_indices"]
                    checked["exact_ids_vs_process_first_ar"] = trial["generated_token_ids"] == g["trials"][0]["generated_token_ids"]
                    checked["passed"] &= checked["exact_ids_vs_process_first_ar"]
                    entry["trials"].append(checked)
                entry["loaded_memory"] = g["loaded_memory"]
                entry["provenance"] = g["provenance"]
                entry["telemetry"] = {k: v for k, v in g["telemetry"].items() if k not in ("events", "request_windows")}
                entry["correct"] = all(t["passed"] for t in entry["trials"])
                for indices, group in zip(process["groups"], process["group_numbers"]):
                    c["groups"].append(summarize_group([entry["trials"][i] for i in indices], group, entry["correct"]))
                result["all_correct"] &= entry["correct"]
            except (ValueError, KeyError, TypeError, IndexError, ZeroDivisionError) as exc:
                entry["errors"].append(str(exc))
                # A provenance/config/malformed-row failure must not make the
                # rest of that process disappear. Preserve every raw trial;
                # none of its groups are eligible for a performance pass.
                if type(g) is dict and type(g.get("trials")) is list:
                    entry["raw_trials_after_invalid_report"] = g["trials"]
                result["complete"] = result["all_correct"] = False
        c["all_three_groups_passed"] = len(c["groups"]) == 3 and all(
            g["status"] == "passed_group_decode_gate" for g in c["groups"])
    result["capture_utc_range"] = [min(captures).isoformat(), max(captures).isoformat()] if captures else None
    result["phase_envelope_ns"] = [monotonic_windows[0][0], monotonic_windows[-1][1]] if monotonic_windows else None
    result["window_decode_gate_passed"] = plan["status"] == "frozen" and result["complete"] and result["all_correct"] and all(
        c["all_three_groups_passed"] for c in result["cases"])
    result["memory_scope"] = "Per-trial allocator active/cache/peak reported only; request states may still be alive. Not a soak or physical bandwidth claim."
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-plan-only", action="store_true")
    args = parser.parse_args()
    report = {"schema": "mtp-release-window-analysis-v1", "gate": GATE,
              "full_release_passed": False, "two_window_decode_gate_passed": False,
              "scope": "Decode performance and frozen regression checks only; historical prose text identity is not a fresh semantic oracle. No automatic default change.",
              "errors": [], "windows": []}
    try:
        require(len(set(p.resolve() for p in args.plan)) == len(args.plan), "Duplicate plan path")
        if args.validate_plan_only:
            for p in args.plan:
                plan = load(p)
                validate_plan(plan, current=True)
                for case in plan["cases"]:
                    frozen_case(case, plan["identity"])
                report["windows"].append({"plan": str(p.resolve()), "status": plan["status"], "valid": True})
            report["plan_validation_passed"] = True
        else:
            # Keep already analyzed windows if a later plan cannot be opened
            # or validated. A top-level error must not erase prior evidence.
            for p in args.plan:
                report["windows"].append(analyze_window(p))
            windows = report["windows"]
            if len(windows) == 2:
                a, b = sorted(windows, key=lambda w: w["window_id"])
                require([a["window_id"], b["window_id"]] == ["a", "b"], "Two distinct window IDs required")
                require(a["identity"] == b["identity"], "Window source/model identities differ")
                plans = [load(p) for p in args.plan]
                require(plans[0]["config"] == plans[1]["config"] and
                        plans[0]["eos_token_ids"] == plans[1]["eos_token_ids"], "Window config/EOS differs")
                frozen_inputs = lambda p: {c["id"]: {k: c[k] for k in
                    ("task", "budget", "input", "input_tokens", "golden", "functional")} for c in p["cases"]}
                require(frozen_inputs(plans[0]) == frozen_inputs(plans[1]), "Window frozen tasks/goldens differ")
                require(a["case_order"] == list(reversed(b["case_order"])), "Window b must reverse case order")
                if a["capture_utc_range"] and b["capture_utc_range"]:
                    require(utc(a["capture_utc_range"][-1]) < utc(b["capture_utc_range"][0]), "Window capture times overlap/reverse")
                if a["phase_envelope_ns"] and b["phase_envelope_ns"]:
                    require(a["phase_envelope_ns"][-1] < b["phase_envelope_ns"][0], "Window phase envelopes overlap/reverse")
                report["two_window_decode_gate_passed"] = all(w["window_decode_gate_passed"] for w in windows)
            else:
                require(len(windows) == 1, "Exactly one or two window plans supported")
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        report["errors"].append(str(exc))
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(encoded)
        print(json.dumps({"output": str(args.output.resolve()), "errors": report["errors"],
                          "plan_validation_passed": report.get("plan_validation_passed"),
                          "two_window_decode_gate_passed": report["two_window_decode_gate_passed"],
                          "windows": [{k: w[k] for k in ("window_id", "complete", "all_correct", "window_decode_gate_passed") if k in w} for w in report["windows"]]}, indent=2))
    else:
        print(encoded, end="")
    if args.validate_plan_only:
        return 0 if report.get("plan_validation_passed") and not report["errors"] else 2
    return 0 if report["two_window_decode_gate_passed"] and not report["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
