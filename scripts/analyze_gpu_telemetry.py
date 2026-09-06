#!/usr/bin/env python3
"""Join saved generation, hardware and Instruments records; never sample a device.

All absolute intervals use mach_absolute_time converted to nanoseconds, and
half-open [start, end) boundaries. A counter delta is assigned only when its
*entire* interval belongs to one phase envelope. Crossing records retain their
raw values and overlap durations, but are never prorated into bytes or counts.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


CLOCK = "mach_absolute_time_nanoseconds"
DISK_FIELDS = (
    "process_disk_read_bytes_delta", "process_disk_write_bytes_delta",
    "system_disk_read_bytes_delta", "system_disk_write_bytes_delta",
)


def interval(record):
    start, end = record.get("start_ns"), record.get("end_ns")
    if type(start) is not int or type(end) is not int or start < 0 or end < start:
        raise ValueError("Interval requires nonnegative integer start_ns <= end_ns")
    return start, end


def union_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("Reversed interval")
        if start == end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def duration(intervals):
    return sum(end - start for start, end in union_intervals(intervals))


def overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def classify_interval(sample, phases):
    """Return one whole-interval owner or partial overlaps, without prorating."""
    start, end = interval(sample)
    partial = []
    for phase in phases:
        low, high = phase["start_ns"], phase["end_ns"]
        # A point observation belongs to the phase starting at that point,
        # never the phase whose exclusive end equals that point.
        if low <= start and (end <= high if end > start else start < high):
            return {"status": "contained", "phase_id": phase["id"], "overlaps": []}
        amount = overlap((start, end), (low, high))
        if amount:
            partial.append({"phase_id": phase["id"], "overlap_ns": amount})
    return {"status": "partial" if partial else "outside", "phase_id": None, "overlaps": partial}


def is_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def optional_sum(records, key):
    values = [r[key] for r in records if is_number(r.get(key))]
    return sum(values) if values else None


def clock_is_absolute(clock):
    if clock == CLOCK:
        return True
    return isinstance(clock, dict) and (
        clock.get("kind") == CLOCK or clock.get("name") in (CLOCK, "mach_absolute_time")
    ) and clock.get("unit", "ns") == "ns"


def absolute_record(record, clock):
    absolute = clock_is_absolute(clock)
    clock_data = clock if isinstance(clock, dict) else {}
    anchor = clock_data.get("absolute_anchor_ns")
    verified = clock_data.get("alignment_status") in ("verified", "aligned", "exact")
    if (type(record.get("start_absolute_ns")) is int and type(record.get("end_absolute_ns")) is int
            and (absolute or verified or record.get("alignment_status") in ("verified", "aligned", "exact"))):
        result = {**record, "start_ns": record["start_absolute_ns"], "end_ns": record["end_absolute_ns"]}
    elif absolute:
        result = record
    elif verified and type(anchor) is int:
        start, end = interval(record)
        result = {**record, "start_ns": start + anchor, "end_ns": end + anchor}
    else:
        return None
    interval(result)
    return result


def assign_absolute(record, clock, phases):
    normalized = absolute_record(record, clock)
    attribution = classify_interval(normalized, phases) if normalized else {"status": "unaligned", "phase_id": None, "overlaps": []}
    uncertainty = clock.get("uncertainty_ns", 0) if isinstance(clock, dict) else 0
    if normalized and type(uncertainty) is int and uncertainty > 0 and attribution["status"] == "contained":
        owner = next(p for p in phases if p["id"] == attribution["phase_id"])
        if normalized["start_ns"] - uncertainty < owner["start_ns"] or normalized["end_ns"] + uncertainty > owner["end_ns"]:
            attribution = {"status": "alignment_boundary_ambiguous", "phase_id": None,
                           "overlaps": [{"phase_id": owner["id"], "overlap_ns": normalized["end_ns"] - normalized["start_ns"]}],
                           "uncertainty_ns": uncertainty}
    return normalized, attribution


def build_phases(generation):
    telemetry = generation.get("telemetry") or {}
    events = telemetry.get("events", [])
    if not events:
        return [], ["No absolute telemetry events: legacy trial durations cannot establish phase alignment."]
    if not clock_is_absolute(telemetry.get("clock")):
        raise ValueError("Generation telemetry clock is not mach_absolute_time nanoseconds")
    groups = {}
    all_intervals = []
    for event in events:
        bounds = interval(event)
        phase = event.get("phase")
        if phase not in ("load", "prefill", "decode") or bounds[0] == bounds[1]:
            raise ValueError("Generation events require load/prefill/decode and positive duration")
        repetition = event.get("repetition")
        if phase != "load" and (type(repetition) is not int or repetition < 0):
            raise ValueError("Request events require a nonnegative repetition")
        groups.setdefault((phase, repetition), []).append(event)
        all_intervals.append((bounds, (phase, repetition)))
    # Leaf events must not overlap; otherwise token/SSD counts double count.
    ordered = sorted(all_intervals)
    for previous, current in zip(ordered, ordered[1:]):
        if current[0][0] < previous[0][1]:
            raise ValueError("Overlapping generation leaf events")
    requests = {}
    for request in telemetry.get("request_windows", []):
        rep = request.get("repetition")
        if rep in requests:
            raise ValueError("Duplicate repetition in request_windows")
        interval(request)
        requests[rep] = request
    phases = []
    for (name, rep), leaves in groups.items():
        leaves.sort(key=lambda e: e["start_ns"])
        start, end = leaves[0]["start_ns"], leaves[-1]["end_ns"]
        envelope_source = "first_to_last_leaf_including_loop_gaps"
        request = requests.get(rep)
        if request and name == "prefill":
            start, end = request["start_ns"], request["prefill_end_ns"]
            envelope_source = "request_windows"
        elif request and name == "decode" and request.get("decode_start_ns") is not None:
            start, end = request["decode_start_ns"], request.get("decode_end_ns", request["end_ns"])
            envelope_source = "request_windows"
        interval({"start_ns": start, "end_ns": end})
        if any(e["start_ns"] < start or e["end_ns"] > end for e in leaves):
            raise ValueError("Phase envelope does not contain its leaf events")
        for bounds, key in all_intervals:
            if key != (name, rep) and overlap((start, end), bounds):
                raise ValueError("Phase envelope crosses a different phase/repetition")
        phases.append({"id": f"{name}:{rep}" if rep is not None else name,
                       "phase": name, "repetition": rep, "start_ns": start, "end_ns": end,
                       "envelope_source": envelope_source, "events": leaves})
    phases.sort(key=lambda p: p["start_ns"])
    for a, b in zip(phases, phases[1:]):
        if a["end_ns"] > b["start_ns"]:
            raise ValueError("Overlapping phase envelopes")
    return phases, []


def coverage(intervals, phase, partial_intervals=()):
    window = phase["end_ns"] - phase["start_ns"]
    full = duration(intervals)
    clips = [(max(a, phase["start_ns"]), min(b, phase["end_ns"]))
             for a, b in partial_intervals if overlap((a, b), (phase["start_ns"], phase["end_ns"]))]
    return {"phase_window_ns": window, "fully_attributed_ns": full,
            "fully_attributed_fraction": full / window if window else None,
            "partial_overlap_ns": duration(clips),
            "any_observation_overlap_ns": duration(list(intervals) + clips)}


def histogram_samples(record):
    report = record.get("ioreport") or {}
    if report.get("kind") != "bandwidth_residency_histogram":
        return []
    result = []
    for channel in report.get("channels") or []:
        values = channel.get("residency_delta_raw")
        if not isinstance(values, list) or not values or not all(is_number(v) and v >= 0 for v in values):
            continue
        result.append({"channel_id": channel.get("id"), "name": channel.get("name"),
                       "source": report.get("source"), "scope": "system; requester mapping is unavailable",
                       "kind": report["kind"], "estimated": report.get("estimated"),
                       "unit": channel.get("unit", report.get("unit")),
                       "state_names": channel.get("state_names", []), "bins": values,
                       "start_ns": record["start_ns"], "end_ns": record["end_ns"],
                       "sample_id": record.get("sample_id")})
    return result


def source_interval(record, previous, source):
    """Conservative endpoint uncertainty: previous read start to current end.

    A proc/IOKit counter is observed somewhere inside its read call, not at
    the end of the whole sampler sweep. Widening boundary checks avoids a
    falsely exact assignment. The raw sweep interval remains separately visible.
    """
    current = record.get(source) or {}
    earlier = (previous or {}).get(source) or {}
    start, end = earlier.get("read_start_ns"), current.get("read_end_ns")
    if type(start) is int and type(end) is int and 0 <= start <= end:
        return {"start_ns": start, "end_ns": end, "interval_source": "conservative_source_read_envelope"}
    start, end = current.get("delta_start_ns"), current.get("delta_end_ns")
    if type(start) is int and type(end) is int and 0 <= start <= end:
        return {"start_ns": start, "end_ns": end, "interval_source": "source_delta_endpoints"}
    return {"start_ns": record["start_ns"], "end_ns": record["end_ns"], "interval_source": "sampler_sweep_fallback"}


def process_resources(records, phases, target_pid, aligned):
    counters, snapshots, previous = [], [], None
    for record in records:
        if record.get("type") not in ("sample", "baseline"):
            continue
        process = record.get("process") or {}
        usage = process.get("rusage") or {}
        pid_ok = target_pid is None or record.get("target_pid") == target_pid
        if aligned and pid_ok and usage:
            if type(process.get("read_start_ns")) is int and type(process.get("read_end_ns")) is int:
                own = {"start_ns": process["read_start_ns"], "end_ns": process["read_end_ns"]}
                snapshots.append({"assignment": classify_interval(own, phases), "sample_id": record.get("sample_id"), **own, "rusage": usage})
            if previous and record.get("type") == "sample":
                older = (previous.get("process") or {}).get("rusage") or {}
                identity_ok = (older.get("start_abstime") is not None and older.get("start_abstime") == usage.get("start_abstime")
                               and previous.get("target_pid") == record.get("target_pid"))
                values = []
                for key in ("cpu_user_time_ns_cumulative", "cpu_system_time_ns_cumulative"):
                    before, after = older.get(key), usage.get(key)
                    values.append(after - before if identity_ok and type(before) is int and type(after) is int and 0 <= before <= after else None)
                bounds = source_interval(record, previous, "process")
                counters.append({"sample_id": record.get("sample_id"), **bounds,
                                 "assignment": classify_interval(bounds, phases),
                                 "cpu_user_ns": values[0], "cpu_system_ns": values[1]})
        previous = record
    result = {}
    for phase in phases:
        full = [r for r in counters if r["assignment"]["phase_id"] == phase["id"]]
        valid = [r for r in full if r["cpu_user_ns"] is not None and r["cpu_system_ns"] is not None]
        partial = [r for r in counters if any(p["phase_id"] == phase["id"] for p in r["assignment"]["overlaps"])]
        covered = duration([interval(r) for r in valid])
        user = optional_sum(valid, "cpu_user_ns")
        system = optional_sum(valid, "cpu_system_ns")
        total = user + system if user is not None and system is not None else None
        observed = [r for r in snapshots if r["assignment"]["phase_id"] == phase["id"]]
        peaks = {}
        for field in ("physical_footprint_bytes", "resident_bytes"):
            available = [r["rusage"][field] for r in observed if type(r["rusage"].get(field)) is int and r["rusage"][field] >= 0]
            peaks[field] = {"sample_peak": max(available) if available else None, "valid_snapshots": len(available), "unit": "bytes"}
        result[phase["id"]] = {
            "source": "libproc proc_pid_rusage cumulative CPU times converted by sampler to ns",
            "cpu_user_ns": user, "cpu_system_ns": system, "cpu_total_ns": total,
            "covered_ns": covered, "one_core_fraction": total / covered if total is not None and covered else None,
            "valid_delta_intervals": len(valid), "invalid_delta_intervals": len(full) - len(valid),
            "coverage": coverage([interval(r) for r in valid], phase, [interval(r) for r in partial]),
            "sampled_memory": peaks,
            "notes": ["CPU total / covered wall time is a one-core fraction and may exceed 1; it is not whole-machine utilization.",
                      "Memory peaks are maxima of phase-contained snapshots, not guaranteed instantaneous high-water marks."]}
    return {"phases": result, "cpu_delta_records": counters, "memory_snapshots": snapshots}


def hardware_summary(records, phases, target_pid=None):
    metadata = next((r for r in records if r.get("type") == "metadata"), {})
    aligned = clock_is_absolute(metadata.get("clock"))
    samples = [r for r in records if r.get("type") == "sample"]
    result = {"alignment": "absolute_clock" if aligned else "unknown",
              "metadata": metadata, "sample_count": len(samples),
              "raw_records": records, "assignments": [], "source_assignments": [], "phases": {}}
    contained = {p["id"]: [] for p in phases}
    partial = {p["id"]: [] for p in phases}
    previous = None
    metric_records = {key: [] for key in DISK_FIELDS}
    histograms = []
    for record in records:
        if record.get("type") not in ("sample", "baseline"):
            continue
        if record.get("type") == "sample":
            for source in ("process", "system_disk", "ioreport"):
                bounds = source_interval(record, previous, source)
                assignment = classify_interval(bounds, phases) if aligned else {"status": "unaligned", "phase_id": None, "overlaps": []}
                result["source_assignments"].append({"sample_id": record.get("sample_id"), "source": source, **bounds, **assignment})
                for field in DISK_FIELDS:
                    if field.startswith("process_") and source == "process" or field.startswith("system_") and source == "system_disk":
                        metric_records[field].append({"sample": record, "bounds": bounds, "assignment": assignment})
                if source == "ioreport":
                    histograms.extend({**h, **bounds, "assignment": assignment} for h in histogram_samples(record))
        previous = record
    previous_end = None
    for sample in samples:
        start, end = interval(sample)
        if previous_end is not None and start < previous_end:
            raise ValueError("Overlapping hardware delta intervals; cannot safely sum")
        previous_end = end
        assignment = classify_interval(sample, phases) if aligned else {"status": "unaligned", "phase_id": None, "overlaps": []}
        result["assignments"].append({"sample_id": sample.get("sample_id"), **assignment})
        if assignment["status"] == "contained":
            contained[assignment["phase_id"]].append(sample)
        for hit in assignment["overlaps"]:
            partial[hit["phase_id"]].append(sample)
    for phase in phases:
        full, edges = contained[phase["id"]], partial[phase["id"]]
        counters = {}
        for field in DISK_FIELDS:
            owned = [r for r in metric_records[field] if r["assignment"]["phase_id"] == phase["id"]]
            boundary = [r for r in metric_records[field] if any(h["phase_id"] == phase["id"] for h in r["assignment"]["overlaps"])]
            valid = [r for r in owned if is_number(r["sample"].get(field)) and r["sample"][field] >= 0
                     and (not field.startswith("process_") or target_pid is None or r["sample"].get("target_pid") == target_pid)]
            value = sum(r["sample"][field] for r in valid) if valid else None
            counters[field] = {"sum": value, "unit": "bytes", "valid_intervals": len(valid),
                               "missing_or_invalid_intervals": len(owned) - len(valid),
                               "partial_sample_ids": [r["sample"].get("sample_id") for r in boundary],
                               "coverage": coverage([interval(r["bounds"]) for r in valid], phase,
                                                    [interval(r["bounds"]) for r in boundary if is_number(r["sample"].get(field))])}
        result["phases"][phase["id"]] = {
            "fully_attributed_samples": len(full), "partial_samples": len(edges),
            "coverage": coverage([interval(r) for r in full], phase, [interval(r) for r in edges]),
            "disk_counters": counters,
            "pmp_histogram_trend": [h for h in histograms if h["assignment"]["phase_id"] == phase["id"]],
            "partial_sample_ids": [r.get("sample_id") for r in edges],
            "target_pids": sorted({r["target_pid"] for r in full if type(r.get("target_pid")) is int}),
            "expected_target_pid": target_pid,
            "target_pid_mismatch_samples": sum(target_pid is not None and r.get("target_pid") != target_pid for r in full),
            "physical_dram_bytes": None, "physical_dram_bandwidth_gbps": None}
    result["process_resources"] = process_resources(records, phases, target_pid, aligned)
    return result


def instruments_summary(report, phases):
    clock = report.get("clock") or {}
    anchor = clock.get("absolute_anchor_ns") if isinstance(clock, dict) else None
    absolute = clock_is_absolute(clock)
    # Unknown/approximate anchors cannot establish phase membership.
    anchor_valid = (type(anchor) is int and isinstance(clock, dict)
                    and clock.get("alignment_status") in ("verified", "aligned", "exact"))
    def counter_key(record, field="id"):
        key = str(record.get(field))
        return key + ":" + str(record["group_index"]) if "group_index" in record else key
    counters = {counter_key(c): c for c in report.get("counters", [])}
    result = {"source": report.get("source"), "clock": clock, "raw_report": report,
              "assignments": [], "phases": {p["id"]: {"counter_samples": [], "counter_statistics": {}} for p in phases}}
    for index, sample in enumerate(report.get("samples", [])):
        normalized, attribution = assign_absolute(sample, clock, phases)
        result["assignments"].append({"sample_index": index, **attribution})
        if attribution["status"] == "contained":
            definition = counters.get(counter_key(sample, "counter_id"), {})
            result["phases"][attribution["phase_id"]]["counter_samples"].append({
                "sample_index": index, "definition": definition,
                "absolute_interval": {"start_ns": normalized["start_ns"], "end_ns": normalized["end_ns"]},
                "raw_sample": sample})
    for phase in result["phases"].values():
        groups = {}
        for record in phase["counter_samples"]:
            groups.setdefault(counter_key(record["raw_sample"], "counter_id"), []).append(record)
        for key, samples in groups.items():
            values = [s["raw_sample"]["value"] for s in samples if is_number(s["raw_sample"].get("value"))]
            phase["counter_statistics"][key] = {
                "definition": counters.get(key, {}), "numeric_sample_count": len(values),
                "min": min(values) if values else None, "max": max(values) if values else None,
                "unweighted_sample_mean": statistics.mean(values) if values else None,
                "interpretation": "Raw source units. No integration into bytes, physical DRAM estimate, or GPU duration inferred."}
    result["alignment"] = "absolute_clock" if absolute else "verified_anchor" if anchor_valid else "unknown"
    return result


def idle_intervals(active, start, end):
    result, cursor = [], start
    for low, high in union_intervals(active):
        if low > cursor:
            result.append((cursor, low))
        cursor = max(cursor, high)
    if cursor < end:
        result.append((cursor, end))
    return result


def gpu_timeline_summary(report, phases, target_pid):
    """PID-attributed execution union, distinct from CPU submission intervals."""
    clock = report.get("clock") or {}
    expected = target_pid if type(target_pid) is int else report.get("target_pid")
    matching_report = type(expected) is int and report.get("target_pid") == expected
    coverage_data = report.get("coverage") or {}
    has_bounds = (type(coverage_data.get("start_ns")) is int and type(coverage_data.get("end_ns")) is int
                  or type(coverage_data.get("start_absolute_ns")) is int and type(coverage_data.get("end_absolute_ns")) is int)
    trace_window = absolute_record(coverage_data, clock) if has_bounds else None
    uncertainty = clock.get("uncertainty_ns", 0) if isinstance(clock, dict) else 0
    uncertainty = uncertainty if type(uncertainty) is int and uncertainty >= 0 else 0
    duration_rounding = coverage_data.get("duration_rounding_uncertainty_ns", 0)
    if type(duration_rounding) is int and duration_rounding >= 0:
        uncertainty += duration_rounding
    assignments = {"gpu_intervals": [], "command_buffer_submissions": []}
    for name in assignments:
        for index, raw in enumerate(report.get(name, [])):
            if not matching_report or raw.get("pid") != expected:
                attribution, normalized = {"status": "pid_mismatch", "phase_id": None, "overlaps": []}, None
            elif name == "gpu_intervals" and raw.get("state") != "Active":
                attribution, normalized = {"status": "not_active", "phase_id": None, "overlaps": []}, None
            else:
                normalized, attribution = assign_absolute(raw, clock, phases)
            assignments[name].append({"record_index": index, "absolute_interval": None if normalized is None else {"start_ns": normalized["start_ns"], "end_ns": normalized["end_ns"]}, **attribution})
    result = {"source": report.get("source"), "target_pid": expected,
              "target_pid_matches_generation": matching_report,
              "clock": clock, "coverage": coverage_data, "raw_report": report,
              "status": report.get("status", "unavailable"), "assignments": assignments, "phases": {}}
    for phase in phases:
        start, end = phase["start_ns"], phase["end_ns"]
        selected = [r for r in assignments["gpu_intervals"] if r["phase_id"] == phase["id"]]
        partial = [r for r in assignments["gpu_intervals"] if any(h["phase_id"] == phase["id"] for h in r["overlaps"])]
        active = union_intervals([interval(r["absolute_interval"]) for r in selected])
        active_ns = duration(active)
        complete = (matching_report and coverage_data.get("complete") is True and not coverage_data.get("errors")
                    and coverage_data.get("target_interval_table_available") is not False
                    and trace_window is not None and trace_window["start_ns"] + uncertainty <= start
                    and end <= trace_window["end_ns"] - uncertainty
                    and not any(r["status"] == "unaligned" for r in assignments["gpu_intervals"]))
        exact = complete and not partial
        idle = idle_intervals(active, start, end) if exact else []
        observed_gaps = [(a[1], b[0]) for a, b in zip(active, active[1:]) if b[0] > a[1]]
        submissions = [r for r in assignments["command_buffer_submissions"] if r["phase_id"] == phase["id"]]
        submission_union = union_intervals([interval(r["absolute_interval"]) for r in submissions])
        covered_ns = overlap((start, end), interval(trace_window)) if trace_window is not None else None
        result["phases"][phase["id"]] = {
            "status": "complete" if exact else "partial_execution_intervals" if partial else "observed_intervals_only" if active else "no_target_intervals_or_unavailable",
            "complete_trace_window_coverage": complete, "trace_overlap_ns": covered_ns,
            "coverage_boundary_uncertainty_ns": uncertainty,
            "trace_window_coverage_fraction": covered_ns / (end - start) if covered_ns is not None else None,
            "fully_attributed_active_records": len(selected), "partial_active_records": len(partial),
            "active_union_intervals_ns": active,
            "observed_active_union_seconds": active_ns / 1e9 if active or exact else None,
            "gpu_active_seconds": active_ns / 1e9 if exact else None,
            "gpu_active_fraction": active_ns / (end - start) if exact else None,
            "gpu_idle_seconds": duration(idle) / 1e9 if exact else None,
            "largest_idle_gap_seconds": max((b - a for a, b in idle), default=0) / 1e9 if exact else None,
            "idle_gaps_ns": idle,
            "between_observed_interval_gaps_ns": observed_gaps,
            "largest_between_observed_interval_gap_seconds": max((b - a for a, b in observed_gaps), default=0) / 1e9 if len(active) > 1 else None,
            "cpu_command_buffer_submission_records": len(submissions),
            "cpu_command_buffer_submission_union_seconds": duration(submission_union) / 1e9 if submissions else None,
            "partial_record_indices": [r["record_index"] for r in partial]}
    result["notes"] = [
        "GPU Active execution intervals require exact target PID attribution; overlapping channels/nested events are unioned, never summed.",
        "No target intervals does not mean zero utilization unless an independently verified complete trace window covers the phase.",
        "Execution intervals crossing phase/clock-uncertainty boundaries are retained as partial, not clipped or prorated into phase active time.",
        "Without complete coverage, gaps between observed intervals are not proven GPU idle gaps.",
        "CPU command-buffer submission intervals are reported separately and never counted as GPU execution."
    ]
    return result


def phase_summary(phase):
    events = phase["events"]
    good = [e for e in events if e.get("succeeded") is True]
    step_ns = duration([interval(e) for e in good])
    outputs = optional_sum(good, "output_tokens")
    inputs = optional_sum(good, "input_tokens")
    wait = optional_sum(good, "ssd_wait_seconds")
    rows = optional_sum(good, "ssd_requested_row_bytes")
    per_token = []
    for event in events:
        n = event.get("input_tokens")
        forward_end, evaluation_end = event.get("forward_end_ns"), event.get("evaluation_end_ns")
        valid_boundaries = (type(forward_end) is int and type(evaluation_end) is int
                            and event["start_ns"] <= forward_end <= evaluation_end <= event["end_ns"])
        per_token.append({"step_index": event.get("step_index"), "start_ns": event["start_ns"], "end_ns": event["end_ns"],
                          "succeeded": event.get("succeeded"), "input_tokens": n,
                          "output_tokens": event.get("output_tokens"),
                          "forward_end_ns": forward_end, "evaluation_end_ns": evaluation_end,
                          "host_forward_seconds": (forward_end - event["start_ns"]) / 1e9 if valid_boundaries else None,
                          "evaluation_and_readback_seconds": (evaluation_end - forward_end) / 1e9 if valid_boundaries else None,
                          "residual_ssd_wait_seconds": event.get("ssd_wait_seconds"),
                          "logical_requested_row_bytes": event.get("ssd_requested_row_bytes"),
                          "residual_ssd_wait_seconds_per_input_token": event.get("ssd_wait_seconds") / n if type(n) is int and n > 0 and is_number(event.get("ssd_wait_seconds")) else None,
                          "logical_requested_row_bytes_per_input_token": event.get("ssd_requested_row_bytes") / n if type(n) is int and n > 0 and is_number(event.get("ssd_requested_row_bytes")) else None})
    envelope_ns = phase["end_ns"] - phase["start_ns"]
    return {k: v for k, v in phase.items() if k != "events"} | {
        "phase_window_seconds": envelope_ns / 1e9, "successful_step_seconds": step_ns / 1e9,
        "all_step_seconds": duration([interval(e) for e in events]) / 1e9,
        "event_count": len(events), "successful_events": len(good), "failed_or_unknown_events": len(events) - len(good),
        "input_tokens": inputs, "output_tokens": outputs,
        "output_tokens_per_successful_step_second": outputs / (step_ns / 1e9) if outputs is not None and step_ns else None,
        "output_tokens_per_phase_window_second": outputs / (envelope_ns / 1e9) if outputs is not None and envelope_ns else None,
        "residual_ssd_wait_seconds": wait, "ssd_wait_events_with_values": sum(is_number(e.get("ssd_wait_seconds")) for e in good),
        "logical_requested_row_bytes": rows, "ssd_byte_events_with_values": sum(is_number(e.get("ssd_requested_row_bytes")) for e in good),
        "per_step": per_token, "physical_dram_bytes": None, "physical_dram_bandwidth_gbps": None,
        "gpu_device_seconds": None}


def analyze(generation, hardware=(), instruments=None, warm_repetitions=None, gpu_intervals=None):
    phases, warnings = build_phases(generation)
    summaries = [phase_summary(p) for p in phases]
    selected = {p["repetition"] for p in phases if p["phase"] == "decode" and p["repetition"] > 0} if warm_repetitions is None else set(warm_repetitions)
    warm = [p for p in summaries if p["phase"] == "decode" and p["repetition"] in selected]
    warm_steps = [s for p in warm for s in p["per_step"]]
    latency_seconds = sorted((s["end_ns"] - s["start_ns"]) / 1e9 for s in warm_steps if s["succeeded"] is True)
    success_seconds = sum(p["successful_step_seconds"] for p in warm)
    window_seconds = sum(p["phase_window_seconds"] for p in warm)
    tokens = optional_sum(warm, "output_tokens")
    telemetry = generation.get("telemetry") or {}
    hardware_result = hardware_summary(hardware, phases, telemetry.get("target_pid"))
    instrument_result = instruments_summary(instruments or {}, phases)
    gpu_timeline = gpu_timeline_summary(gpu_intervals or {}, phases, telemetry.get("target_pid"))
    for summary in summaries:
        timeline = gpu_timeline["phases"][summary["id"]]
        summary["gpu_device_seconds"] = timeline["gpu_active_seconds"]
        summary["gpu_active_fraction"] = timeline["gpu_active_fraction"]
        summary["gpu_timeline_coverage_status"] = timeline["status"]
    warm_gpu = [gpu_timeline["phases"][p["id"]] for p in warm]
    complete_gpu = bool(warm_gpu) and all(p["gpu_active_seconds"] is not None for p in warm_gpu)
    gpu_seconds = sum(p["gpu_active_seconds"] for p in warm_gpu) if complete_gpu else None
    warm_process = [hardware_result["process_resources"]["phases"][p["id"]] for p in warm]
    cpu_ns = optional_sum(warm_process, "cpu_total_ns")
    cpu_covered = sum(p["covered_ns"] for p in warm_process)
    memory_peaks = {}
    for field in ("physical_footprint_bytes", "resident_bytes"):
        values = [p["sampled_memory"][field]["sample_peak"] for p in warm_process if p["sampled_memory"][field]["sample_peak"] is not None]
        memory_peaks[field] = max(values) if values else None
    warm_disk = {}
    for field in DISK_FIELDS:
        entries = [hardware_result["phases"][p["id"]]["disk_counters"][field] for p in warm]
        attributed_ns = sum(e["coverage"]["fully_attributed_ns"] for e in entries)
        warm_disk[field] = {"sum": optional_sum(entries, "sum"), "unit": "bytes",
                            "fully_attributed_ns": attributed_ns,
                            "warm_window_coverage_fraction": attributed_ns / (window_seconds * 1e9) if window_seconds else None,
                            "partial_overlap_ns": sum(e["coverage"]["partial_overlap_ns"] for e in entries)}
    if not warm:
        warnings.append("No selected warm decode phase; no warm throughput is inferred from a lone first trial.")
    profiler = generation.get("profiler") or {}
    if profiler.get("mode") not in (None, "disabled"):
        warnings.append("GPUProfiler stage durations lack absolute phase anchors here; synchronized mode perturbs execution and is not device-only time.")
    if instrument_result["alignment"] == "unknown" and instruments:
        warnings.append("Instruments clock has no verified absolute anchor; its raw counters are retained without phase attribution.")
    if telemetry.get("dropped_events", 0):
        warnings.append("Generation telemetry dropped events; step/token/SSD aggregates are incomplete.")
    warnings.extend(telemetry.get("warnings", []))
    return {"schema_version": 1, "clock": CLOCK, "phases": summaries,
            "warm_decode": {"selection": "repetition > 0" if warm_repetitions is None else "explicit repetitions",
                            "repetitions": sorted(selected), "phase_ids": [p["id"] for p in warm],
                            "successful_step_seconds": success_seconds, "phase_window_seconds": window_seconds,
                            "successful_steps": len(latency_seconds),
                            "step_latency_median_seconds": statistics.median(latency_seconds) if latency_seconds else None,
                            "step_latency_p95_nearest_rank_seconds": latency_seconds[math.ceil(len(latency_seconds) * 0.95) - 1] if latency_seconds else None,
                            "output_tokens": tokens,
                            "tokens_per_successful_step_second": tokens / success_seconds if tokens is not None and success_seconds else None,
                            "tokens_per_phase_window_second": tokens / window_seconds if tokens is not None and window_seconds else None,
                            "residual_ssd_wait_seconds": optional_sum(warm, "residual_ssd_wait_seconds"),
                            "logical_requested_row_bytes": optional_sum(warm, "logical_requested_row_bytes"),
                            "ssd_wait_steps_with_values": sum(p["ssd_wait_events_with_values"] for p in warm),
                            "ssd_byte_steps_with_values": sum(p["ssd_byte_events_with_values"] for p in warm),
                            "per_step": warm_steps,
                            "disk_counters": warm_disk,
                            "pmp_histogram_trend": [h for p in warm for h in hardware_result["phases"][p["id"]]["pmp_histogram_trend"]],
                            "process_cpu": {"total_ns": cpu_ns, "covered_ns": cpu_covered,
                                            "one_core_fraction": cpu_ns / cpu_covered if cpu_ns is not None and cpu_covered else None,
                                            "warm_window_coverage_fraction": cpu_covered / (window_seconds * 1e9) if window_seconds else None},
                            "process_sampled_memory_peaks_bytes": memory_peaks,
                            "gpu_timeline": {"all_phase_windows_complete": complete_gpu, "gpu_active_seconds": gpu_seconds,
                                             "gpu_active_fraction": gpu_seconds / window_seconds if gpu_seconds is not None and window_seconds else None,
                                             "observed_active_union_seconds": optional_sum(warm_gpu, "observed_active_union_seconds"),
                                             "partial_active_records": sum(p["partial_active_records"] for p in warm_gpu),
                                             "largest_idle_gap_seconds": max((p["largest_idle_gap_seconds"] for p in warm_gpu), default=None) if complete_gpu else None},
                            "physical_dram_bytes": None, "physical_dram_bandwidth_gbps": None, "gpu_device_seconds": gpu_seconds},
            "hardware": hardware_result, "instruments": instrument_result,
            "gpu_timeline": gpu_timeline,
            "gpu_profiler_unaligned_raw": profiler,
            "generation_provenance": generation.get("provenance"), "collector_status": telemetry.get("collector_status"),
            "telemetry_run_id": telemetry.get("run_id"), "telemetry_dropped_events": telemetry.get("dropped_events", 0),
            "warnings": warnings,
            "notes": [
                "Warm selection is a repetition convention, not proof that OS pages, kernels, or weights are cache-resident.",
                "Phase envelopes include token-loop gaps; successful_step_seconds sums completed model/sampling leaves. Both denominators are retained.",
                "Prefill includes the held-back final prompt token and first sampled output. Decode output counts may include EOS.",
                "Residual SSD wait is overlapped host wait, not disk service time. Requested row bytes are logical payload, not physical IO.",
                "Hardware deltas spanning a phase boundary are partial and excluded from attributed sums. No time-proportional byte or histogram allocation is performed.",
                "Source counter attribution prefers the conservative previous-read-start/current-read-end envelope; its coverage measures observation windows, not exact instantaneous counter endpoints.",
                "Disk counters remain per-process or system-wide as labeled. PMP bins are system residency observations, not physical DRAM bytes or bandwidth.",
                "Instruments source definitions, raw intervals and values are preserved. GPU/cache/interface counters are not relabeled as DRAM traffic.",
                "Allocator peak is process-lifetime cumulative. GPUProfiler wait includes graph/JIT/encoding/synchronization and is not GPU-only execution."
            ]}


def load_json(path):
    return json.loads(Path(path).read_text())


def load_phase_file(path):
    """Accept session.json, a generation wrapper, or metadata + interval JSONL."""
    text = Path(path).read_text()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
        value = dict(next((r for r in records if r.get("type") == "metadata"), {}))
        value["events"] = [r for r in records if r.get("type") == "interval"]
        value["request_windows"] = []
        value.setdefault("warnings", []).append("Phase JSONL has no request envelopes; first-to-last leaf envelopes include interior loop gaps only.")
    return value.get("telemetry", value)


def file_evidence(path):
    file = Path(path)
    return {"path": str(file.resolve()), "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", required=True, type=Path)
    parser.add_argument("--phase-file", type=Path, help="Optional session.json or phases.jsonl overriding embedded telemetry")
    parser.add_argument("--hardware", type=Path, help="Hardware sampler JSONL")
    parser.add_argument("--instruments", type=Path, help="Exported Instruments counters JSON")
    parser.add_argument("--gpu-intervals", type=Path, help="Exported exact-PID GPU execution and CPU submission intervals JSON")
    parser.add_argument("--warm-repetitions", help="Explicit comma-separated repetition IDs; default: repetitions > 0")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    generation = load_json(args.generation)
    inputs = {"generation": file_evidence(args.generation)}
    if args.phase_file:
        generation["telemetry"] = load_phase_file(args.phase_file)
        inputs["phase_file"] = file_evidence(args.phase_file)
    hardware = []
    if args.hardware:
        hardware = [json.loads(line) for line in args.hardware.read_text().splitlines() if line.strip()]
        inputs["hardware"] = file_evidence(args.hardware)
    instruments = load_json(args.instruments) if args.instruments else None
    if args.instruments:
        inputs["instruments"] = file_evidence(args.instruments)
    gpu_intervals = load_json(args.gpu_intervals) if args.gpu_intervals else None
    if args.gpu_intervals:
        inputs["gpu_intervals"] = file_evidence(args.gpu_intervals)
    repetitions = None
    if args.warm_repetitions is not None:
        repetitions = [int(value) for value in args.warm_repetitions.split(",") if value.strip()]
        if any(value < 0 for value in repetitions):
            parser.error("Warm repetition IDs must be nonnegative")
    report = analyze(generation, hardware, instruments, repetitions, gpu_intervals)
    report["input_files"] = inputs
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "phases": len(report["phases"]),
                      "warm_decode_tokens_per_second": report["warm_decode"]["tokens_per_successful_step_second"],
                      "warnings": report["warnings"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
