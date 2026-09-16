#!/usr/bin/env python3
"""Audit saved CoreAI health responses without starting or contacting a service.

Use --artifacts with a check_coreai_service.py artifacts directory, or supply a
bare /health JSON response through --health-json. This checks observed API state
and timing accounting, not GPU execution, CLI rejection, or handoff correctness.
"""
import argparse
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def validate_health(value, *, expected_chunk=4, independent=True):
    require(isinstance(value, dict), "health must be an object")
    require(value.get("backend") == "native-coreai", "unexpected backend")
    require(value.get("pd_scheduling") == "serial", "scheduler must explicitly report serial")
    require(type(value.get("independent_pd_functions")) is bool, "independent_pd_functions must be boolean")
    chunk = value.get("prefill_chunk_size")
    require(type(chunk) is int and chunk in (1, 4), "invalid prefill chunk size")
    require(value.get("prefill_policy") == ("tokenwise" if chunk == 1 else "chunked"), "prefill policy contradicts chunk size")
    if value.get("ready") is True:
        require(chunk == expected_chunk, "ready service selected a different chunk size")
        require(value["independent_pd_functions"] == independent, "ready service selected a different P/D path")
    counts = {}
    for key in ("active_requests", "queued_requests", "requests_in_flight", "max_pending_requests"):
        item = value.get(key)
        require(type(item) is int and item >= 0, f"invalid {key}")
        counts[key] = item
    require(counts["active_requests"] <= 1, "serial service reports multiple active requests")
    require(counts["max_pending_requests"] > 0, "admission limit must be positive")
    require(counts["active_requests"] + counts["queued_requests"] == counts["requests_in_flight"], "request reservation counts disagree")
    require(counts["requests_in_flight"] <= counts["max_pending_requests"], "admitted requests exceed the limit")
    for key in ("prefill_group_milliseconds", "decode_group_milliseconds"):
        groups = value.get(key)
        require(isinstance(groups, dict), f"missing {key}")
        for group, duration in groups.items():
            require(isinstance(group, str) and bool(group), f"invalid group in {key}")
            require(type(duration) in (int, float) and math.isfinite(duration) and duration >= 0,
                    f"invalid duration for {key}.{group}")
    return value


def load_samples(health_paths, artifact_paths):
    for path in health_paths:
        yield str(path), json.loads(path.read_text())
    for directory in artifact_paths:
        paths = sorted(directory.glob("*health.ndjson"))
        require(bool(paths), f"no health records in {directory}")
        for path in paths:
            for number, line in enumerate(path.read_text().splitlines(), start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                # Connection failures are not health snapshots; their original
                # acceptance report determines whether transport checks passed.
                if record.get("status") == 200 and "raw_utf8" in record:
                    yield f"{path}:{number}", json.loads(record["raw_utf8"])


def audit(samples, *, expected_chunk=4, independent=True, require_queued=False, require_work=False):
    ready = queued = prefill_work = decode_work = count = 0
    for source, value in samples:
        try:
            validate_health(value, expected_chunk=expected_chunk, independent=independent)
        except (AssertionError, TypeError) as error:
            raise AssertionError(f"{source}: {error}") from error
        count += 1
        ready += value.get("ready") is True
        queued += value["active_requests"] == 1 and value["queued_requests"] > 0
        prefill_work += any(value["prefill_group_milliseconds"].values())
        decode_work += any(value["decode_group_milliseconds"].values())
    require(count > 0 and ready > 0, "no ready health sample established the selected P/D configuration")
    require(not require_queued or queued > 0, "no active-plus-queued observation")
    require(not require_work or (prefill_work > 0 and decode_work > 0), "missing positive timing evidence for both business phases")
    return {"passed": True, "samples": count, "ready_samples": ready, "queued_samples": queued,
            "prefill_work_samples": prefill_work, "decode_work_samples": decode_work,
            "expected_chunk": expected_chunk, "independent_pd_functions": independent,
            "scope": "Saved health API invariants only; not device, CLI, handoff or scheduler concurrency proof."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--health-json", action="append", type=Path, default=[])
    parser.add_argument("--artifacts", action="append", type=Path, default=[])
    parser.add_argument("--expected-chunk", type=int, choices=(1, 4), default=4)
    parser.add_argument("--legacy", action="store_true", help="Expect the non-PD model path")
    parser.add_argument("--require-queued", action="store_true")
    parser.add_argument("--require-work", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.health_json and not args.artifacts:
        parser.error("provide --health-json or --artifacts")
    try:
        result = audit(load_samples(args.health_json, args.artifacts), expected_chunk=args.expected_chunk,
                       independent=not args.legacy, require_queued=args.require_queued, require_work=args.require_work)
    except Exception as error:
        result = {"passed": False, "error": str(error)}
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
