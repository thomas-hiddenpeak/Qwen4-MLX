#!/usr/bin/env python3
"""Summarize recorded cooperative-scheduler callbacks without loading a model.

Only integer monotonic timestamps from the report define wall-clock metrics.
Missing or invalid data stays null; compute timings are never a TTFT fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def seconds(start: Any, end: Any) -> float | None:
    return (end - start) / 1e9 if integer(start) and integer(end) and end >= start else None


def rate(count: Any, duration: Any) -> float | None:
    return count / duration if integer(count) and number(duration) and duration > 0 else None


def distribution(values: list[float] | None) -> dict[str, Any]:
    """Linear interpolation at q * (n - 1), including genuine zero gaps."""
    if values is None:
        return {"count": None, "p50_seconds": None, "p95_seconds": None, "max_seconds": None}
    ordered = sorted(values)

    def quantile(q: float) -> float | None:
        if not ordered:
            return None
        rank = q * (len(ordered) - 1)
        lo, hi = math.floor(rank), math.ceil(rank)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)

    return {"count": len(values), "p50_seconds": quantile(0.5),
            "p95_seconds": quantile(0.95), "max_seconds": max(values) if values else None}


def summarize_job(group: dict[str, Any], role: str) -> dict[str, Any]:
    issues: list[str] = []
    callbacks = group.get(f"{role}_callbacks")
    raw_valid = isinstance(callbacks, list) and all(
        isinstance(cb, dict) and integer(cb.get("timestampNS")) for cb in callbacks)
    stamps = [cb["timestampNS"] for cb in callbacks] if raw_valid else None
    if stamps is not None and any(b < a for a, b in zip(stamps, stamps[1:])):
        stamps = None
        issues.append("callback timestamps are not monotonic in recorded delivery order")
    if not raw_valid:
        issues.append("raw callback timestamps are missing or invalid")
    metrics = group.get(f"{role}_callback_metrics") or {}
    submitted = metrics.get("submit_ns")
    if not integer(submitted):
        issues.append("raw submission timestamp is missing or invalid")
    if stamps and integer(submitted) and stamps[0] < submitted:
        stamps = None
        issues.append("callback precedes submission")
    gaps = [seconds(a, b) for a, b in zip(stamps, stamps[1:])] if stamps is not None else None
    job_id = group.get(f"{role}_job_id")
    steps = group.get("steps")
    steps = steps if isinstance(steps, list) else []
    job_steps = [step for step in steps if isinstance(step, dict) and
                 isinstance(step.get("event"), dict) and job_id is not None and
                 step["event"].get("jobID") == job_id]
    terminals = [step for step in job_steps if step["event"].get("kind") in
                 ("completed", "cancelled", "failed")]
    terminal = terminals[0] if len(terminals) == 1 else None
    if terminal is None:
        issues.append("exactly one terminal event was not recorded")
    event = terminal["event"] if terminal else {}
    result = event.get("result") or {}
    stats = result.get("statistics") or {}
    phases = result.get("phases") or {}
    prefill = phases.get("prefill") or {}
    completion = seconds(submitted, terminal.get("endNS")) if terminal else None
    ttft = seconds(submitted, stamps[0]) if stamps else None
    last_output = seconds(submitted, stamps[-1]) if stamps else None
    delivery_span = seconds(stamps[0], stamps[-1]) if stamps else None
    token_ids = result.get("tokens")
    matched = ([cb.get("tokenID") for cb in callbacks] == token_ids
               if raw_valid and isinstance(token_ids, list) else None)
    decoded = stats.get("decodedTokenCount")
    decoded = decoded if integer(decoded) else None
    compute_seconds = result.get("decodeSeconds")
    compute_seconds = compute_seconds if number(compute_seconds) else None
    service_seconds = phases.get("decodeServiceSeconds")
    service_seconds = service_seconds if number(service_seconds) else None

    # Only a cooperative step corresponds to one first-token delivery or one
    # AR/MTP round. Whole-stage steps can contain many rounds: do not guess.
    burst_gaps = None
    burst_counts = None
    if (group.get("limits") or {}).get("executionMode") == "cooperative" and stamps is not None:
        bursts: list[tuple[int, int, int]] = []
        mapped = 0
        windows_valid = all(seconds(step.get("startNS"), step.get("endNS")) is not None for step in job_steps)
        prior_end = None
        if windows_valid:
            for step in job_steps:
                if prior_end is not None and step["startNS"] <= prior_end:
                    windows_valid = False
                    break
                prior_end = step["endNS"]
                selected = [stamp for stamp in stamps if step["startNS"] <= stamp <= step["endNS"]]
                if selected:
                    bursts.append((selected[0], selected[-1], len(selected)))
                    mapped += len(selected)
        if windows_valid and mapped == len(stamps):
            burst_gaps = [seconds(a[1], b[0]) for a, b in zip(bursts, bursts[1:])]
            burst_counts = [burst[2] for burst in bursts]
        else:
            issues.append("callbacks could not be mapped uniquely to cooperative steps")

    return {
        "role": role, "job_id": job_id, "terminal_kind": event.get("kind"),
        "callback_count": len(callbacks) if isinstance(callbacks, list) else None,
        "callback_ids_match_result": matched,
        "mtp_depth": stats.get("mtpDepth"), "decode_rounds": stats.get("decodeRounds"),
        "submission_to_first_callback_seconds": ttft,
        "submission_to_last_callback_seconds": last_output,
        "submission_to_terminal_observation_seconds": completion,
        "scheduler_elapsed_seconds": (event.get("timing") or {}).get("elapsedSeconds"),
        "callback_gaps": distribution(gaps),
        "cooperative_burst_gaps": distribution(burst_gaps),
        "cooperative_burst_output_counts": burst_counts,
        "wall_delivery_decode_tokens_per_second": rate(max(len(stamps) - 1, 0), delivery_span) if stamps is not None else None,
        "completed_request_tokens_per_second": rate(len(token_ids), completion) if
            event.get("kind") == "completed" and isinstance(token_ids, list) else None,
        "compute_decode_token_count": decoded, "compute_decode_seconds": compute_seconds,
        "compute_decode_tokens_per_second": rate(decoded, compute_seconds),
        "active_decode_service_seconds": service_seconds,
        "active_decode_service_tokens_per_second": rate(decoded, service_seconds),
        "prefill_target_seconds": prefill.get("targetSeconds"),
        "prefill_total_active_seconds": prefill.get("totalSeconds"),
        "queue_timing": event.get("timing"), "issues": issues,
    }


def summarize_report(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    report = json.loads(raw)
    if not isinstance(report.get("groups"), list):
        raise ValueError(f"{path}: expected a cooperative scheduler probe report with groups")
    groups = []
    for group in report["groups"]:
        jobs = [summarize_job(group, role) for role in ("long", "short")]
        duration = seconds(group.get("start_ns"), group.get("end_ns"))
        count = sum(job["callback_count"] for job in jobs) if all(
            integer(job["callback_count"]) and job["terminal_kind"] == "completed" and
            job["callback_ids_match_result"] is True for job in jobs) else None
        groups.append({"mode": group.get("mode"), "limits": group.get("limits"),
                       "group_wall_seconds": duration, "completed_output_token_count": count,
                       "group_wall_tokens_per_second": rate(count, duration), "jobs": jobs})
    return {"source": str(path.resolve()), "source_sha256": hashlib.sha256(raw).hexdigest(),
            "source_schema": report.get("schema"), "source_complete": report.get("complete"),
            "source_passed": report.get("passed"), "clock": report.get("clock"), "groups": groups}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New JSON path; existing files are not overwritten")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must be a new file")
    reports = [summarize_report(path) for path in args.reports]
    summary = {
        "schema": "qwen38-scheduler-latency-analysis-v1",
        "quantile_method": "linear interpolation at q*(n-1); includes zero callback gaps",
        "notes": [
            "These are local library callbacks, not HTTP/SSE or rendered-text delivery measurements.",
            "TTFT includes submit admission, queueing, active prefill and suspension through the first callback; excludes model load before submit.",
            "Request completion is the observed terminal runNext return, not the last output callback.",
            "The first output was selected by prefill; decode numerator excludes it. Emitted EOS is included. Uncommitted drafts never count.",
            "MTP may publish several callbacks in one round; near-zero within-round gaps are retained. Burst gaps require cooperative step records.",
            "Compute decode excludes callback time and suspended queue time; active decode service includes callbacks but excludes suspension.",
            "Wall delivery throughput uses (callback_count-1)/(last_callback-first_callback); it includes scheduling delays and differs from compute throughput.",
            "Missing or invalid raw timing remains null; old aggregate compute/TTFT fields are never substituted.",
            "A single functional run is a latency observation, not a stable performance win or a populated-workload p95.",
        ], "reports": reports,
    }
    with args.output.open("x") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
    print("source | mode | role | TTFT s | terminal s | gap p50/p95/max ms | compute tok/s")
    def display(value: Any, scale: float = 1) -> str:
        return f"{value * scale:.3f}" if number(value) else "unknown"
    for report in reports:
        for group in report["groups"]:
            for job in group["jobs"]:
                gaps = job["callback_gaps"]
                gap_text = "/".join(display(gaps[key], 1000) for key in ("p50_seconds", "p95_seconds", "max_seconds"))
                print(f"{Path(report['source']).name} | {group['mode']} | {job['role']} | "
                      f"{display(job['submission_to_first_callback_seconds'])} | "
                      f"{display(job['submission_to_terminal_observation_seconds'])} | {gap_text} | "
                      f"{display(job['compute_decode_tokens_per_second'])}")


if __name__ == "__main__":
    main()
