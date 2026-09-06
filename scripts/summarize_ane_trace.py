#!/usr/bin/env python3
"""Summarize ANE hardware intervals exported by Apple's Core ML Instruments template.

The XML uses ID/ref cells. Resolve them before counting named model predictions.
This records actual ANE events for the selected experts, not CPU routing or a
claim that the entire decoder executes on ANE. Traced timings are not benchmarks.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics
import xml.etree.ElementTree as ET


def summarize(path):
    root = ET.parse(path).getroot()
    ids = {element.attrib["id"]: element for element in root.iter() if "id" in element.attrib}
    def resolve(element):
        while "ref" in element.attrib:
            element = ids[element.attrib["ref"]]
        return element
    predictions = defaultdict(list)
    prediction_labels = defaultdict(set)
    labels = defaultdict(int)
    for row in root.findall(".//row"):
        cells = [resolve(cell) for cell in row]
        label = next((cell.attrib.get("fmt", "") for cell in cells if cell.tag == "formatted-label"), "")
        labels[label] += 1
        # Core ML can append a compile UUID to an otherwise unchanged copied
        # package. Normalize only that strict suffix, preserving raw labels as
        # evidence; unexpected extra models/calls still fail the exact gate.
        match = re.fullmatch(
            r"(expert_\d+|shared)(?:_[0-9A-Fa-f]{8}(?:-[0-9A-Fa-f]{4}){3}-[0-9A-Fa-f]{12})?"
            r"(_main__Op\d+_AneInference)\s+Prediction", label)
        if match:
            duration = next(float(cell.text) / 1_000_000 for cell in cells if cell.tag == "duration")
            name = match.group(1) + match.group(2)
            predictions[name].append(duration)
            prediction_labels[name].add(label)
    return {"xml": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "total_ane_intervals": len(root.findall(".//row")),
            "matched_prediction_count": sum(map(len, predictions.values())),
            "models": {name: {"predictions": len(times), "raw_prediction_labels": sorted(prediction_labels[name]),
                               "median_hardware_interval_ms": statistics.median(times),
                               "sum_hardware_interval_ms": sum(times)} for name, times in sorted(predictions.items())},
            "labels": dict(labels)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ane-xml", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cpu-control-xml", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    ane = summarize(args.ane_xml)
    iterations = 1 + report["warmups"] + len(report["iterations"])
    expected = {f"expert_{expert:04d}_main__Op0_AneInference": iterations for expert in report["selectedExpertIDs"]}
    expected["shared_main__Op0_AneInference"] = iterations
    actual = {name: row["predictions"] for name, row in ane["models"].items()}
    # This gate deliberately targets the one-token, one-call-per-expert decode.
    passed = report["tokenCount"] == 1 and actual == expected
    control = summarize(args.cpu_control_xml) if args.cpu_control_xml else None
    if control is not None:
        passed = passed and control["matched_prediction_count"] == 0
    result = {"ane": ane, "cpu_only_control": control, "expected_predictions_by_model": expected,
              "traced_swift_report": str(args.report.resolve()),
              "traced_swift_report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
              "passed": passed,
              "evidence": "Named ANE hardware prediction intervals from Instruments match every selected expert plus shared expert and every decode iteration. The CPU_ONLY control must contain none of those model predictions when supplied.",
              "limits": "One real layer-0 decode input. CPU routing, merging, and I/O remain on CPU. Tracing affects timing; this is not an end-to-end generation, energy, or memory measurement."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": passed, "prediction_count": ane["matched_prediction_count"], "models": actual,
                      "cpu_control_predictions": control["matched_prediction_count"] if control else None}, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
