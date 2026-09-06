#!/usr/bin/env python3
"""Compare bounded expert concurrency using the same real MoE fixtures and calls.

Runs degrees 1/2/4/8 in forward then reverse order, with separate cold-load
records and warmed measurements. Requires an exclusive performance window.
Also runs the independent tiny CPU scheduler gate at degree 4/cache capacity 1.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_outputs(actual: dict, baseline: dict) -> dict:
    comparisons = {}
    for name in sorted(set(actual) | set(baseline)):
        a, b = actual.get(name), baseline.get(name)
        if a is None or b is None or a["shape"] != b["shape"] or a["dtype"] != b["dtype"]:
            comparisons[name] = {"exact": False, "reason": "missing tensor, shape or dtype mismatch"}
            continue
        av, bv = np.asarray(a["values"], dtype=np.float64), np.asarray(b["values"], dtype=np.float64)
        if av.shape != bv.shape or not np.all(np.isfinite(av)) or not np.all(np.isfinite(bv)):
            comparisons[name] = {"exact": False, "reason": "nonfinite or unequal value counts"}
            continue
        delta, norm = av - bv, np.linalg.norm(bv)
        comparisons[name] = {"exact": bool(np.array_equal(av, bv)), "max_abs": float(np.max(np.abs(delta))),
                             "relative_l2": float(np.linalg.norm(delta) / norm) if norm else None}
    return {"exact": all(value["exact"] for value in comparisons.values()), "outputs": comparisons}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, default=ROOT / ".build/release/ane-runner")
    parser.add_argument("--manifest", type=Path, default=ROOT / "results/moe-export/layer_0_fp16_linear_s32/manifest.json")
    parser.add_argument("--fixture-dir", type=Path, default=ROOT / "fixtures/moe-real/converted")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/moe-concurrency")
    parser.add_argument("--degrees", default="1,2,4,8")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--cache-experts", type=int, default=64)
    parser.add_argument("--compute-units", choices=("cpuOnly", "cpuAndNeuralEngine"), default="cpuAndNeuralEngine")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--skip-scheduler-regression", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    report = {"schema": "moe-concurrency-benchmark-v1", "status": "running", "groups": [],
              "scope": "Same layer-0 real token fixtures and complete dynamic routing. Warm block latency only, not full-model token throughput. CPU_AND_NE permits fallback; this benchmark provides no hardware execution trace.",
              "timing_semantics": "predictionMilliseconds sums per-call durations and may exceed wall time when calls overlap. First iteration includes compilation/loading and is excluded from warmed medians."}
    try:
        runner, manifest_path = args.runner.resolve(), args.manifest.resolve()
        degrees = list(dict.fromkeys(int(value) for value in args.degrees.split(",")))
        if 1 not in degrees or degrees[0] != 1 or min(degrees) < 1 or args.rounds < 1 or args.runs < 1 or args.warmups < 0:
            raise ValueError("Degrees must start with 1 and be positive; rounds/runs positive, warmups nonnegative")
        manifest = json.loads(manifest_path.read_text())
        fixtures = {phase: (args.fixture_dir / f"{phase}.json").resolve() for phase in ("decode", "prefill")}
        coverage = {}
        for phase, path in fixtures.items():
            fixture = json.loads(path.read_text())
            count = fixture["inputs"]["x"]["shape"][1]
            ids = fixture["expectedOutputs"]["selected_experts"]["values"]
            expert_counts = Counter(map(int, ids))
            coverage[phase] = {"tokens": count, "assignments": len(ids), "expert_ids": sorted(expert_counts),
                               "expert_calls": sum(math.ceil(n / manifest["token_capacity"]) for n in expert_counts.values()),
                               "shared_calls": math.ceil(count / manifest["token_capacity"]),
                               "fixture": str(path), "fixture_sha256": sha256(path)}
        report["source"] = {"runner": str(runner), "runner_sha256": sha256(runner),
                            "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path),
                            "script_sha256": sha256(Path(__file__)), "fixtures": coverage}
        report["configuration"] = {"degrees": degrees, "rounds": args.rounds, "warmups": args.warmups,
                                   "runs": args.runs, "cache_experts": args.cache_experts,
                                   "compute_units": args.compute_units}
        if not args.skip_scheduler_regression:
            directory = output / "scheduler-degree4"
            command = [sys.executable, str(ROOT / "scripts/verify_moe_scheduler.py"), "--runner", str(runner),
                       "--output-dir", str(directory), "--expert-concurrency", "4", "--timeout", str(args.timeout)]
            process = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
            (output / "scheduler-degree4.log").write_text(process.stdout + process.stderr)
            scheduler = json.loads((directory / "report.json").read_text()) if (directory / "report.json").exists() else {}
            report["scheduler_regression"] = {"pass": process.returncode == 0 and scheduler.get("status") == "pass",
                                               "command": command, "report": str(directory / "report.json")}
            if not report["scheduler_regression"]["pass"]:
                raise RuntimeError("Independent CPU scheduler regression failed; performance run skipped")
        baselines = {}
        samples = {phase: {degree: [] for degree in degrees} for phase in fixtures}
        all_gates = True
        for round_index in range(args.rounds):
            order = degrees if round_index % 2 == 0 else list(reversed(degrees))
            for degree in order:
                for phase, fixture_path in fixtures.items():
                    name = f"round{round_index + 1}-{phase}-degree{degree}"
                    result_path = output / f"{name}.json"
                    result_path.unlink(missing_ok=True)
                    command = [str(runner), "probe-moe", "--manifest", str(manifest_path), "--fixture", str(fixture_path),
                               "--precision", "bfloat16Boundaries", "--compute-units", args.compute_units,
                               "--cache-experts", str(args.cache_experts), "--expert-concurrency", str(degree),
                               "--warmups", str(args.warmups), "--runs", str(args.runs), "--output", str(result_path)]
                    started = time.monotonic()
                    process = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
                    (output / f"{name}.stdout.log").write_text(process.stdout)
                    (output / f"{name}.stderr.log").write_text(process.stderr)
                    if process.returncode != 0 or not result_path.exists():
                        raise RuntimeError(f"{name} failed ({process.returncode}): {process.stderr[-2000:]}")
                    actual = json.loads(result_path.read_text())
                    if phase not in baselines:
                        baselines[phase] = actual["outputs"]
                    comparison = compare_outputs(actual["outputs"], baselines[phase])
                    calls = [{key: t[key] for key in ("expertCalls", "sharedCalls", "cacheMisses")}
                             for t in [actual["firstIteration"], *actual["iterations"]]]
                    calls_match = all(t["expertCalls"] == coverage[phase]["expert_calls"] and
                                      t["sharedCalls"] == coverage[phase]["shared_calls"] for t in calls)
                    gate = comparison["exact"] and calls_match and actual.get("routingExpertSetsMatch") is True
                    all_gates &= gate
                    milliseconds = [t["totalMilliseconds"] for t in actual["iterations"]]
                    samples[phase][degree].extend(milliseconds)
                    record = {"round": round_index + 1, "phase": phase, "requested_degree": degree, "pass": gate,
                              "command": command, "report": str(result_path), "process_seconds": time.monotonic() - started,
                              "median_milliseconds": statistics.median(milliseconds), "samples_milliseconds": milliseconds,
                              "outputs_vs_serial": comparison, "calls_match": calls_match, "calls": calls,
                              "first_iteration": actual["firstIteration"], "iterations": actual["iterations"],
                              "native_y_comparison": actual.get("comparisons", {}).get("y"),
                              "concurrency_fields": {k: v for k, v in actual.items() if "concurr" in k.lower()}}
                    report["groups"].append(record)
                    write_json(report_path, report)
                    print(json.dumps({"group": name, "pass": gate, "median_ms": record["median_milliseconds"]}), flush=True)
        summary = {}
        for phase, degree_samples in samples.items():
            serial = statistics.median(degree_samples[1])
            summary[phase] = {str(degree): {"median_milliseconds": statistics.median(values),
                                           "minimum_milliseconds": min(values), "maximum_milliseconds": max(values),
                                           "sample_count": len(values), "speedup_vs_serial": serial / statistics.median(values),
                                           "round_medians": [g["median_milliseconds"] for g in report["groups"]
                                                             if g["phase"] == phase and g["requested_degree"] == degree]}
                              for degree, values in degree_samples.items()}
        report["summary"] = summary
        report["status"] = "pass" if all_gates else "fail"
        report["all_outputs_exact_vs_serial"] = all(g["outputs_vs_serial"]["exact"] for g in report["groups"])
        report["same_call_counts"] = all(g["calls_match"] for g in report["groups"])
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.TimeoutExpired) as error:
        report["status"], report["error"] = "fail", str(error)
    write_json(report_path, report)
    print(json.dumps({"status": report["status"], "report": str(report_path),
                      **({"summary": report["summary"]} if "summary" in report else {}),
                      **({"error": report["error"]} if "error" in report else {})}, indent=2), flush=True)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
