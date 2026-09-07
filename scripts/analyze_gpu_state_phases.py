#!/usr/bin/env python3
"""Attribute saved system GPU state deltas to complete request phase envelopes.

Read-only: no sampler, process signals, model loading, or frequency inference.
Raw weights include OFF states. Thermal observations are counts, not durations.
"""
import argparse
import hashlib
import json
from pathlib import Path

from analyze_gpu_telemetry import CLOCK, classify_interval, duration
from analyze_mtp_release_window import validated_phases
from check_agent_workload import strict_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def bounds(start, end):
    require(type(start) is int and type(end) is int and 0 <= start <= end,
            "Invalid source read bounds")
    return {"start_ns": start, "end_ns": end}


def read_bounds(source, collection):
    span = bounds(source["read_start_ns"], source["read_end_ns"])
    require(collection["start_ns"] <= span["start_ns"] <= span["end_ns"] <= collection["end_ns"],
            "Source read is outside its collection sweep")
    return span


def attribute(generation, records):
    telemetry = generation["telemetry"]
    require(telemetry["collector_status"] == "completed" and telemetry["dropped_events"] == 0,
            "Generation collector is incomplete")
    metadata = [r for r in records if r.get("type") == "metadata"]
    summaries = [r for r in records if r.get("type") == "summary"]
    require(len(metadata) == len(summaries) == 1, "Expected one metadata and completed summary")
    meta, summary = metadata[0], summaries[0]
    require(telemetry["clock"] == meta["clock"] == summary["clock"] == CLOCK, "Clock mismatch")
    pid = generation["provenance"]["process_id"]
    require(all(type(p) is int and p > 0 and p == pid for p in
                (pid, telemetry["target_pid"], meta["target_pid"], summary["target_pid"])), "PID mismatch")
    require(summary["target_signalled_by_sampler"] is False, "Sampler signalled target")
    samples = [r for r in records if r.get("type") == "sample"]
    require(type(summary["samples_written"]) is int and summary["samples_written"] >= 0,
            "Sample count is not a nonnegative integer")
    baseline = [r for r in records if r.get("type") == "baseline"]
    require(len(baseline) == 1 and type(baseline[0]["sample_id"]) is int and baseline[0]["sample_id"] == 0,
            "Expected one baseline with sample_id=0")
    require([r for r in records if r.get("type") in ("sample", "baseline")] == baseline + samples,
            "Baseline must precede samples")
    require(all(type(r["sample_id"]) is int for r in samples), "Sample IDs must be integers")
    require([r["sample_id"] for r in samples] == list(range(1, summary["samples_written"] + 1)),
            "Missing, duplicate or reordered sample IDs")
    require(records[0] == meta and records[-1] == summary, "Metadata/summary ordering mismatch")
    require([t["repetition"] for t in generation["trials"]] == list(range(len(generation["trials"]))),
            "Trial repetition order mismatch")
    phases = validated_phases(generation)
    trials = {t["repetition"]: t for t in generation["trials"]}
    require(len(trials) == len(generation["trials"]), "Duplicate trial repetition")
    output = {}
    for p in phases:
        result = {k: p[k] for k in ("id", "phase", "repetition", "start_ns", "end_ns", "envelope_source")}
        result.update(thermal_observation_counts={}, channels={})
        if p["repetition"] is not None:
            t = trials[p["repetition"]]
            result.update(mtp_depth=t["mtp_depth"], output_tokens=len(t["generated_token_ids"]),
                          decode_token_s=t["decode_tokens_per_second"], phase_metrics=t["phase_metrics"])
        output[p["id"]] = result
    excluded = {"incomplete_channel_deltas": 0, "crossing_or_outside_channel_deltas": 0,
                "crossing_or_outside_thermal_observations": 0, "records_without_gpu_states": 0}
    previous = {}
    for record in records:
        if record.get("type") not in ("sample", "baseline"):
            continue
        require(record["clock"] == CLOCK and type(record["target_pid"]) is int and record["target_pid"] == pid,
                "Sample clock/PID mismatch")
        collection = bounds(record["collection_start_ns"], record["end_ns"])
        states = record.get("gpu_states")
        if not states:
            previous.clear()
            excluded["records_without_gpu_states"] += 1
            continue
        thermal = states.get("thermal")
        if thermal:
            owner = classify_interval(read_bounds(thermal, collection), phases)
            if owner["status"] == "contained":
                counts = output[owner["phase_id"]]["thermal_observation_counts"]
                name = thermal["state"]
                require(isinstance(name, str), "Invalid thermal category")
                counts[name] = counts.get(name, 0) + 1
            else:
                excluded["crossing_or_outside_thermal_observations"] += 1
        seen = set()
        for source in states.get("state_channels", []):
            key = (source["group"], source["subgroup"])
            require(key not in seen, "Duplicate state source in a sample")
            seen.add(key)
            old = previous.get(key)
            previous[key] = source
            read_bounds(source, collection)
            if source["complete_delta"] is not True:
                excluded["incomplete_channel_deltas"] += 1
                continue
            require(old is not None and
                    (source["previous_read_start_ns"], source["previous_read_end_ns"]) ==
                    (old["read_start_ns"], old["read_end_ns"]), "Broken state endpoint chain")
            require(source["previous_read_end_ns"] <= source["read_start_ns"], "Reversed state endpoints")
            envelope = bounds(source["previous_read_start_ns"], source["read_end_ns"])
            owner = classify_interval(envelope, phases)
            if owner["status"] != "contained":
                excluded["crossing_or_outside_channel_deltas"] += 1
                continue
            target = output[owner["phase_id"]]["channels"]
            require(source["channels"], "Complete delta has no channels")
            channel_names = set()
            for channel in source["channels"]:
                name, unit = channel["name"], channel["unit"]
                require(isinstance(name, str) and name not in channel_names, "Invalid/duplicate channel name")
                channel_names.add(name)
                names = [s["name"] for s in channel["states"]]
                values = [s["residency_delta_raw"] for s in channel["states"]]
                require(names and all(isinstance(n, str) for n in names) and len(set(names)) == len(names),
                        "Missing/duplicate state names")
                require(all(type(v) is int and 0 <= v < 2**63 for v in values), "Invalid raw state delta")
                channel_key = key + (name,)
                entry = target.setdefault(channel_key, {"group": key[0], "subgroup": key[1], "name": name,
                    "unit": unit, "state_names": names, "raw_counts": [0] * len(names),
                    "complete_deltas": 0, "endpoint_envelopes": []})
                require(entry["unit"] == unit and entry["state_names"] == names, "Channel layout changed")
                entry["raw_counts"] = [a + b for a, b in zip(entry["raw_counts"], values)]
                entry["complete_deltas"] += 1
                entry["endpoint_envelopes"].append([envelope["start_ns"], envelope["end_ns"]])
        for missing in previous.keys() - seen:
            del previous[missing]
    for phase in output.values():
        for channel in phase["channels"].values():
            covered = duration(channel["endpoint_envelopes"])
            window = phase["end_ns"] - phase["start_ns"]
            channel.update(endpoint_envelope_union_ns=covered, phase_window_ns=window,
                           envelope_coverage_fraction=covered / window)
            total = sum(channel["raw_counts"])
            channel["raw_weight_fraction"] = ({n: v / total for n, v in zip(channel["state_names"], channel["raw_counts"])}
                                               if total else None)
        phase["channels"] = list(phase["channels"].values())
    return {"target_pid": meta["target_pid"], "metadata_gpu_states": meta.get("gpu_states"),
            "summary": summary, "excluded": excluded, "phases": list(output.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Refuse to overwrite an analysis")
    result = {"schema": "gpu-state-phase-attribution-v1", "analysis_source": fingerprint(Path(__file__)),
        "helpers": [fingerprint(Path(__file__).with_name(name)) for name in
                    ("analyze_gpu_telemetry.py", "analyze_mtp_release_window.py", "check_agent_workload.py")],
        "scope": "System-wide raw residency weights, including OFF, not MHz, target utilization, physical DRAM or a causal throttling claim. Only complete source read endpoint envelopes wholly inside a phase are attributed. Thermal categories are observation counts, not time fractions. Phase envelopes include host gaps/history as defined by request windows.",
        "reports": []}
    for path in args.report:
        entry = {"path": str(path.resolve()), "error": None}
        result["reports"].append(entry)
        try:
            entry["generation_source"] = fingerprint(path)
            generation = strict_json(path.read_text())
            sidecar = Path(generation["telemetry"]["sidecar_file"])
            entry["hardware_source"] = fingerprint(sidecar)
            records = [strict_json(line) for line in sidecar.read_text().splitlines()]
            entry.update(attribute(generation, records))
        except (OSError, ValueError, TypeError, KeyError) as error:
            entry["error"] = f"{type(error).__name__}: {error}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "reports": len(result["reports"]),
                      "errors": [x["error"] for x in result["reports"] if x["error"]]}))
    return int(any(x["error"] for x in result["reports"]))


if __name__ == "__main__":
    raise SystemExit(main())
