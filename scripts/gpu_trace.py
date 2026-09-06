#!/usr/bin/env python3
"""Record/export Apple Instruments GPU counters without claiming physical DRAM.

Uses the installed xctrace executable and its exported XML schemas. No private
framework calls, privileged changes, or runtime model process management.
"""
from __future__ import annotations

import argparse
import bisect
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import plistlib
import subprocess
import sqlite3
import xml.etree.ElementTree as ET

DEFAULT_DEVELOPER = "/Applications/Xcode.app/Contents/Developer"
SCHEMAS = ("time-info", "device-gpu-info", "gpu-counter-info", "gpu-counter-value",
           "metal-gpu-counter-intervals", "metal-gpu-counter-profile",
           "metal-gpu-intervals", "metal-gpu-state-intervals",
           "metal-application-command-buffer-submissions", "os-signpost")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_xctrace(args, developer):
    env = os.environ.copy()
    env["DEVELOPER_DIR"] = developer
    result = subprocess.run(["/usr/bin/xcrun", "xctrace", *args], env=env,
                            capture_output=True, text=True)
    return {"command": ["xcrun", "xctrace", *args], "exit_code": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr}


class XMLTable:
    """Resolve xctrace's shared id/ref nodes before interpreting values."""
    def __init__(self, path):
        self.root = ET.parse(path).getroot()
        self.ids = {node.get("id"): node for node in self.root.iter() if node.get("id")}

    def resolve(self, node):
        seen = set()
        while node.get("ref"):
            ref = node.get("ref")
            if ref in seen or ref not in self.ids:
                raise ValueError(f"Missing or cyclic xctrace ref {ref}")
            seen.add(ref)
            node = self.ids[ref]
        return node

    def value(self, node):
        node = self.resolve(node)
        if list(node):
            return [self.value(child) for child in node]
        return (node.text or "").strip()

    def rows(self):
        for node in self.root.findall("node"):
            schema = node.find("schema")
            if schema is None:
                continue
            columns = [col.findtext("mnemonic") for col in schema.findall("col")]
            for row in node.findall("row"):
                if len(row) != len(columns):
                    raise ValueError("xctrace row/schema column mismatch")
                yield {name: self.value(value) for name, value in zip(columns, row)}, {
                    name: self.resolve(value) for name, value in zip(columns, row)}


def read_rows(path):
    if not Path(path).exists():
        return []
    return list(XMLTable(path).rows())


def time_mapping(rows):
    mappings = []
    for row, _ in rows:
        numerator, denominator = map(int, row["timebase-info"])
        if numerator <= 0 or denominator <= 0:
            raise ValueError("Invalid mach timebase")
        epoch = int(row["mabs-epoch"])
        update = int(row["update-time"])
        # mabs-epoch is the run epoch (trace relative zero), NOT update time.
        # update-time marks when this calibration was emitted, not its origin.
        anchor = epoch * numerator // denominator
        mappings.append({"effective_from_ns": update, "absolute_anchor_ns": anchor,
                         "mabs_epoch_ticks": epoch, "numerator": numerator,
                         "denominator": denominator})
    mappings.sort(key=lambda m: m["effective_from_ns"])
    if not mappings:
        return {"kind": "trace_relative_ns", "absolute_anchor_ns": None,
                "alignment_status": "unknown", "uncertainty_ns": None,
                "reason": "Instruments time-info mapping unavailable", "mappings": []}
    anchors = {m["absolute_anchor_ns"] for m in mappings}
    return {"kind": "trace_relative_ns", "absolute_anchor_ns": mappings[0]["absolute_anchor_ns"] if len(anchors) == 1 else None,
            "alignment_status": "exact", "uncertainty_ns": 1,
            "absolute_clock": "mach_absolute_time_nanoseconds",
            "source": "xctrace time-info: Run Epoch times mach Timebase plus trace-relative ns",
            "rounding": "integer floor; <= 1 ns difference between separate rational conversions",
            "mappings": mappings}


def absolute_time(relative_ns, clock):
    mappings = clock["mappings"]
    if not mappings:
        return None
    index = bisect.bisect_right([m["effective_from_ns"] for m in mappings], relative_ns) - 1
    if index < 0:
        return None
    return relative_ns + mappings[index]["absolute_anchor_ns"]


def load_phases(path):
    if path is None:
        return None
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return {"path": str(Path(path).resolve()), "sha256": digest(path), "events": rows,
            "join_policy": "Raw intervals only; phase aggregation must reject boundary/uncertainty overlaps, never prorate."}


def gpu_intervals(directory, clock, target_pid):
    """Strict PID filtering; retain queue submissions separately from execution."""
    result = {"schema_version": 1, "source": "Apple Instruments Metal System Trace",
              "clock": clock, "target_pid": target_pid, "gpu_intervals": [], "command_buffer_submissions": [],
              "status": "target_pid_required" if target_pid is None else "no_target_intervals",
              "attribution": "process.pid from exported row, exact integer match; system GPU-state rows excluded",
              "duration_policy": "Union overlapping Active GPU execution intervals; never sum overlapping channels/nesting. Submissions are CPU-side intervals."}
    for schema, output_key in (("metal-gpu-intervals", "gpu_intervals"),
                               ("metal-application-command-buffer-submissions", "command_buffer_submissions")):
        rows = read_rows(Path(directory) / (schema + ".xml"))
        for row, nodes in rows:
            process = row.get("process")
            if not isinstance(process, list) or not process or not str(process[0]).isdigit():
                continue
            pid = int(process[0])
            if target_pid is None or pid != target_pid:
                continue
            start = int(row["start"])
            end = start + int(row["duration"])
            item = {"start_ns": start, "end_ns": end, "pid": pid,
                    "process": nodes["process"].get("fmt"), "raw": row}
            for key in ("state", "channel-name", "event-depth", "cmdbuffer-id", "encoder-id", "gpu-submission-id", "start-latency", "gpu"):
                if key in row:
                    item[key.replace("-", "_")] = row[key]
            for name, relative in (("start_absolute_ns", start), ("end_absolute_ns", end)):
                absolute = absolute_time(relative, clock)
                if absolute is not None:
                    item[name] = absolute
            result[output_key].append(item)
    if result["gpu_intervals"]:
        result["status"] = "target_gpu_intervals_collected"
    result["interval_count"] = len(result["gpu_intervals"])
    return result


def trace_coverage(trace, directory, clock):
    """Trace duration is coverage evidence; event min/max is never coverage.

    Completeness is deliberately conservative: any recorded run issue or
    unreadable issue store prevents a complete-coverage claim.
    """
    trace, directory = Path(trace).resolve(), Path(directory)
    result = {"start_ns": None, "end_ns": None, "start_absolute_ns": None,
              "end_absolute_ns": None, "complete": False,
              "source": "xctrace run/1/info/summary/duration; run issues retained from trace archive",
              "errors": [], "run_issues": [], "source_status": "unknown"}
    toc_path = directory / "toc.xml"
    if not toc_path.exists():
        result["errors"].append("Trace TOC unavailable")
        return result
    toc = ET.parse(toc_path)
    summary = toc.find(".//run[@number='1']/info/summary")
    if summary is None or not summary.findtext("duration"):
        result["errors"].append("Trace duration unavailable")
        return result
    result["start_ns"] = 0
    result["end_ns"] = int(Decimal(summary.findtext("duration")) * 1_000_000_000)
    result["start_absolute_ns"] = absolute_time(0, clock)
    result["end_absolute_ns"] = absolute_time(result["end_ns"], clock)
    result["end_reason"] = summary.findtext("end-reason")
    issue_store = trace / "Trace1.run/RunIssues.storedata"
    try:
        connection = sqlite3.connect(issue_store.as_uri() + "?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        result["run_issues"] = [dict(row) for row in connection.execute(
            "SELECT ZTYPE AS type_code,ZSUBTYPE AS subtype_code,ZMESSAGE AS message,ZCOUNT AS count,ZRELATIVETIMESTAMP AS relative_timestamp FROM ZISSUE")]
        connection.close()
    except (ValueError, sqlite3.Error) as error:
        result["errors"].append(f"Run issue inspection unavailable: {error}")
    if result["run_issues"]:
        result["errors"].append("Trace has recorded run issues; completeness is not asserted")
    gpu_table = directory / "metal-gpu-intervals.xml"
    result["target_interval_table_available"] = gpu_table.exists()
    if not gpu_table.exists():
        result["errors"].append("GPU interval schema export unavailable")
    if not (trace / "form.template").exists():
        result["errors"].append("Finalized trace archive marker unavailable")
    if not result["end_reason"]:
        result["errors"].append("Trace end reason unavailable")
    result["complete"] = not result["errors"]
    result["source_status"] = "closed_trace_no_recorded_issues" if result["complete"] else "coverage_window_known_completeness_unverified"
    result["duration_rounding_uncertainty_ns"] = 1000
    result["duration_rounding_note"] = "TOC duration is exported in seconds (observed six decimals); boundary comparisons allow 1 microsecond in addition to clock rounding."
    return result


def mark_incomplete_source(coverage, source_trace):
    """Successful import cannot repair evidential loss in its source capture."""
    source_trace = Path(source_trace).resolve()
    raw = source_trace / "Trace1.run/Attachments/trace-data.atrc"
    coverage["complete"] = False
    coverage["source_status"] = "recovered_from_incomplete_source_trace"
    coverage["errors"].append("Original source trace was incomplete; successful offline import does not establish capture completeness or absence of dropped events")
    coverage["incomplete_source_trace"] = {
        "path": str(source_trace), "exists": source_trace.exists(),
        "finalized_archive_marker_present": (source_trace / "form.template").exists(),
        "raw_attachment": str(raw) if raw.exists() else None,
        "raw_attachment_bytes": raw.stat().st_size if raw.exists() else None,
        "declared_by": "--incomplete-source-trace CLI argument"}


def build_report(directory, trace, phase_path=None, target=None):
    directory = Path(directory)
    clock = time_mapping(read_rows(directory / "time-info.xml"))
    info = {}
    for row, _ in read_rows(directory / "gpu-counter-info.xml"):
        key = (int(row["counter-id"]), int(row["group-index"]))
        info[key] = {"id": key[0], "group_index": key[1], "name": row["name"],
                     "unit": "%" if row["type"] == "Percentage" else None,
                     "type": row["type"], "description": row["description"],
                     "max_value": int(row["max-value"]), "accelerator_id": int(row["accelerator-id"]),
                     "is_percentage": row["type"] == "Percentage", "raw": row,
                     "scope": "GPU device counter; no per-process attribution established",
                     "source": "Apple Instruments gpu-counter-info and metal-gpu-counter-intervals"}
    samples = []
    for row, nodes in read_rows(directory / "metal-gpu-counter-intervals.xml"):
        key = (int(row["counter-id"]), int(row["group-index"]))
        counter = info.setdefault(key, {"id": key[0], "group_index": key[1], "name": nodes["name"].get("fmt", row["name"]), "unit": None})
        # Label supplies the unit actually emitted for THIS trace, avoiding the
        # known GB/s vs GiB/s ambiguity in the generic UI resource catalog.
        label = row["label"]
        if isinstance(label, list) and len(label) >= 2 and isinstance(label[1], str):
            unit = label[1].strip()
            if counter["unit"] is not None and counter["unit"] != unit:
                raise ValueError(f"Conflicting units for counter {key}")
            counter["unit"] = unit
            counter["unit_source"] = "exported formatted-label unit"
        start = int(row["start"])
        end = start + int(row["duration"])
        if end < start:
            raise ValueError("Negative GPU counter interval")
        sample = {"counter_id": key[0], "group_index": key[1], "start_ns": start, "end_ns": end,
                  "value": float(row["value"]), "gpu": row["gpu"],
                  "ring_buffer_index": int(row["ring-buffer-index"]),
                  "is_percentage": row["is-percentage"] == "1"}
        start_abs, end_abs = absolute_time(start, clock), absolute_time(end, clock)
        if start_abs is not None and end_abs is not None:
            sample.update(start_absolute_ns=start_abs, end_absolute_ns=end_abs)
        samples.append(sample)
    counters = list(info.values())
    for counter in counters:
        if "bandwidth" in counter["name"].lower():
            counter["scope"] = "GPU external-memory interface; may include system-level cache; not isolated physical DRAM"
    bandwidth_ids = {(c["id"], c["group_index"]) for c in counters if "bandwidth" in c["name"].lower() and not c.get("is_percentage")}
    bandwidth_samples = sum((s["counter_id"], s["group_index"]) in bandwidth_ids for s in samples)
    return {"schema_version": 1, "source": "Instruments Metal System Trace", "trace": str(Path(trace).resolve()),
            "target": target, "clock": clock, "counters": counters, "samples": samples,
            "sample_count": len(samples), "bandwidth_sample_count": bandwidth_samples,
            "gpu_read_write_bandwidth_available": bool(bandwidth_samples),
            "physical_dram_bytes": None, "physical_dram_bandwidth_bytes_per_second": None,
            "attribution": "Counter values describe the GPU, including concurrent system GPU work. Target PID is trace selection, not proof of byte attribution.",
            "capability_status": "bandwidth_samples_collected" if bandwidth_samples else "bandwidth_unavailable_in_this_trace",
            "phases": load_phases(phase_path),
            "raw_tables": {p.name: {"path": str(p.resolve()), "sha256": digest(p)} for p in directory.glob("*.xml")},
            "notes": ["Counter percentages are not bytes or DRAM utilization.",
                      "No physical DRAM total is inferred from logical weights, SSD traffic, elapsed time, or GPU peak bandwidth.",
                      "GPU activity/state tables are retained as raw XML; overlapping execution intervals must be unioned before duration aggregation."]}


def export_trace(trace, output, developer, phases=None, target=None, incomplete_source_trace=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "counters.json").exists():
        raise FileExistsError("Refusing to overwrite existing counters.json")
    results = [run_xctrace(["export", "--input", str(trace), "--toc", "--output", str(output / "toc.xml")], developer)]
    if results[0]["exit_code"] != 0:
        raise RuntimeError(results[0])
    toc = ET.parse(output / "toc.xml")
    available = {t.get("schema") for t in toc.findall(".//table")}
    if not (target or {}).get("pid"):
        process = toc.find(".//run[@number='1']/info/target/process")
        if process is not None and process.get("pid", "").isdigit():
            target = {**(target or {}), "pid": int(process.get("pid")), "pid_source": "xctrace run/1/info/target/process"}
    for schema in SCHEMAS:
        if schema not in available:
            continue
        xpath = f'/trace-toc/run[@number="1"]/data/table[@schema="{schema}"]'
        results.append(run_xctrace(["export", "--input", str(trace), "--xpath", xpath,
                                   "--output", str(output / (schema + ".xml"))], developer))
    (output / "export-metadata.json").write_text(json.dumps({"developer_dir": developer, "exports": results}, indent=2))
    report = build_report(output, trace, phases, target)
    intervals = gpu_intervals(output, report["clock"], (target or {}).get("pid"))
    coverage = trace_coverage(Path(trace).resolve(), output, report["clock"])
    if incomplete_source_trace is not None:
        mark_incomplete_source(coverage, incomplete_source_trace)
    intervals["coverage"] = coverage
    report["coverage"] = coverage
    (output / "gpu_intervals.json").write_text(json.dumps(intervals, indent=2, allow_nan=False))
    report["gpu_intervals_file"] = str((output / "gpu_intervals.json").resolve())
    report["export_errors"] = [r for r in results if r["exit_code"] != 0]
    (output / "counters.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def make_template(source, destination, profile):
    """Copy the installed archived template, changing only its profile fields.

    Numeric profile IDs are device/Xcode-dependent. A successful copy does not
    establish hardware support: inspect record warnings and exported counters.
    """
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    with open(source, "rb") as stream:
        template = plistlib.load(stream)
    objects = template["$objects"]
    changes = []
    for obj in objects[:]:
        if not isinstance(obj, dict) or "NS.keys" not in obj:
            continue
        for index, key in enumerate(obj["NS.keys"]):
            name = objects[key.data]
            if name in ("counterprofile", "counterprofileinternal", "counterscounterprofile"):
                obj["NS.objects"][index] = plistlib.UID(len(objects))
                objects.append(profile)
                changes.append(name)
    if set(changes) != {"counterprofile", "counterprofileinternal", "counterscounterprofile"}:
        raise ValueError("Installed template profile structure differs; refusing unverified edit")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        plistlib.dump(template, stream, fmt=plistlib.FMT_BINARY, sort_keys=False)
    return {"source": str(Path(source).resolve()), "source_sha256": digest(source),
            "output": str(destination.resolve()), "output_sha256": digest(destination),
            "profile": profile, "changed_fields": changes, "hardware_support": "unverified"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--developer-dir", default=DEFAULT_DEVELOPER)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--trace", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    export.add_argument("--phases", type=Path)
    export.add_argument("--target-pid", type=int, help="Required for PID-attributed GPU execution intervals")
    export.add_argument("--incomplete-source-trace", type=Path,
                        help="Original incomplete trace used for offline recovery; prevents completeness claims")
    template = commands.add_parser("template")
    template.add_argument("--source", required=True, type=Path)
    template.add_argument("--output", required=True, type=Path)
    template.add_argument("--profile", type=int, required=True)
    record = commands.add_parser("record")
    record.add_argument("--template", default="Metal System Trace")
    record.add_argument("--output", required=True, type=Path)
    record.add_argument("--time-limit", required=True)
    record.add_argument("--phases", type=Path)
    record.add_argument("--notify-tracing-started")
    target = record.add_mutually_exclusive_group(required=True)
    target.add_argument("--attach", type=int)
    target.add_argument("--launch", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == "template":
        result = make_template(args.source, args.output, args.profile)
        args.output.with_suffix(".provenance.json").write_text(json.dumps(result, indent=2))
    elif args.command == "export":
        result = export_trace(args.trace, args.output, args.developer_dir, args.phases,
                              {"pid": args.target_pid} if args.target_pid is not None else None,
                              args.incomplete_source_trace)
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        trace = args.output / "recording.trace"
        command = ["record", "--template", args.template, "--output", str(trace), "--time-limit", args.time_limit, "--no-prompt"]
        if args.notify_tracing_started:
            command += ["--notify-tracing-started", args.notify_tracing_started]
        if args.attach:
            command += ["--attach", str(args.attach)]
        else:
            if not args.launch or not Path(args.launch[0]).is_absolute():
                raise ValueError("--launch requires an absolute executable path")
            command += ["--launch", "--", *args.launch]
        recording = run_xctrace(command, args.developer_dir)
        (args.output / "record-metadata.json").write_text(json.dumps(recording, indent=2))
        if not trace.exists():
            raise RuntimeError(recording)
        result = export_trace(trace, args.output, args.developer_dir, args.phases,
                              {"pid": args.attach, "launch_argv": args.launch})
    print(json.dumps({key: result.get(key) for key in ("output", "capability_status", "sample_count", "bandwidth_sample_count", "clock") if key in result}, indent=2))


if __name__ == "__main__":
    main()
