#!/usr/bin/env python3
"""Fuse a fixed expert bank into three wide projections with dynamic routing.

Concatenate gate/up rows and down columns. Apply per-token routing coefficients
before the final projection: Wd_cat @ concat(p_i * SiLU(Wg_i x) * Wu_i x).
This preserves real arithmetic but changes FP16 reduction boundaries. It computes
all bank experts, including zero-coefficient ones; coverage and wasted work are
reported. A partial bank is never presented as a complete 512-expert layer.
"""
import argparse
from collections import Counter
import gc
import hashlib
import traceback
from pathlib import Path
import json
import time

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
import numpy as np
import torch
import torch.nn.functional as F

import export_moe as e


def make_program(weights, capacity):
    count = len(weights)
    gate = np.concatenate([w["gate_proj"]["dense16"] for w in weights])
    up = np.concatenate([w["up_proj"]["dense16"] for w in weights])
    down = np.concatenate([w["down_proj"]["dense16"] for w in weights], axis=1)

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, capacity, e.HIDDEN), dtype=types.fp16),
                            mb.TensorSpec(shape=(1, capacity, count, 1), dtype=types.fp16)],
                opset_version=ct.target.macOS15)
    def program(x, coefficients):
        g = mb.linear(x=x, weight=gate, name="gate_projection")
        u = mb.linear(x=x, weight=up, name="up_projection")
        denominator = mb.add(x=mb.exp(x=mb.mul(x=g, y=np.float16(-1))), y=np.float16(1))
        middle = mb.mul(x=mb.real_div(x=g, y=denominator), y=u)
        grouped = mb.reshape(x=middle, shape=[1, capacity, count, e.INTERMEDIATE])
        weighted = mb.mul(x=grouped, y=coefficients)
        combined = mb.reshape(x=weighted, shape=[1, capacity, count * e.INTERMEDIATE])
        return mb.linear(x=combined, weight=down, name="y")
    return program


def reference(weights, x, coefficients):
    result = torch.zeros(x.shape, dtype=torch.float32)
    tx = torch.from_numpy(x.astype(np.float32))
    with torch.inference_mode():
        for slot, w in enumerate(weights):
            matrix = {name: torch.from_numpy(value["dense32"].copy()) for name,value in w.items()}
            middle = F.silu(F.linear(tx, matrix["gate_proj"])) * F.linear(tx, matrix["up_proj"])
            result += F.linear(middle, matrix["down_proj"]) * torch.from_numpy(coefficients[:,:,slot].astype(np.float32))
    return result.numpy()


def evaluate_capacity(args, ids, weights, cases, capacity):
    name = f"bank{len(ids)}_shared{int(args.shared)}_s{capacity}_scale{args.coefficient_scale:g}"
    path = args.output / f"{name}.mlpackage"
    start = time.perf_counter()
    row = {"name": name, "experts": ids, "shared": args.shared, "capacity": capacity,
           "path": str(path), "coefficient_scale": args.coefficient_scale, "cases": {}}
    model = ct.convert(make_program(weights, capacity), convert_to="mlprogram",
                       minimum_deployment_target=ct.target.macOS15,
                       compute_precision=ct.precision.FLOAT16, compute_units=ct.ComputeUnit.CPU_AND_NE)
    model.save(str(path))
    plan = e.compute_plan(model)
    row.update({"package_bytes": e.package_bytes(path), "export_load_seconds": time.perf_counter() - start,
                "compute_plan": plan, "preferred_counts": dict(Counter(p["preferred"] for p in plan))})
    lookup = {expert: slot for slot, expert in enumerate(ids)}
    for phase, case in cases.items():
        for begin in range(0, case["x"].shape[1], capacity):
            end = min(begin + capacity, case["x"].shape[1])
            n = end - begin
            case_name = phase if begin == 0 and end == case["x"].shape[1] else f"{phase}_{begin}_{end}"
            x = np.zeros((1, capacity, e.HIDDEN), np.float16)
            x[:, :n] = case["x"][:, begin:end]
            coefficients = np.zeros((1, capacity, len(weights), 1), np.float16)
            missing = set()
            for token in range(n):
                for expert, weight in zip(case["selected_experts"][0, begin+token], case["routing_weights"][0, begin+token]):
                    if int(expert) in lookup:
                        coefficients[0, token, lookup[int(expert)], 0] = weight
                    else:
                        missing.add(int(expert))
            if args.shared:
                coefficients[0, :n, -1, 0] = case["shared_gate"][0, begin:end, 0]
            complete = not missing and args.shared
            if args.require_complete and not complete:
                raise ValueError(f"Full-MoE gate requires every selected expert and shared branch: {case_name}, missing={sorted(missing)}")
            expected = reference(weights, x, coefficients)
            scaled_coefficients = (coefficients.astype(np.float32) * args.coefficient_scale).astype(np.float16)
            if not np.isfinite(scaled_coefficients).all():
                raise ValueError("Scaled coefficient overflow")
            ready_inputs = {"x": x, "coefficients": scaled_coefficients}
            def predict_ready():
                return model.predict(ready_inputs)["y"].astype(np.float32) / args.coefficient_scale
            def predict_host_boundary():
                prepared = (coefficients.astype(np.float32) * args.coefficient_scale).astype(np.float16)
                return model.predict({"x": x.copy(), "coefficients": prepared})["y"].astype(np.float32) / args.coefficient_scale
            y = predict_ready()
            valid_y = y[:, :n]
            record = {"phase": phase, "positions": case["positions"][begin:end].tolist(),
                      "token_ids": case["token_ids"][:, begin:end].tolist(),
                      "selected_experts": case["selected_experts"][:, begin:end].tolist(),
                      "valid_token_count": n, "missing_experts": sorted(missing),
                      "complete_moe_for_these_tokens": complete,
                      "error_vs_fp32_selected_bank": e.errors(valid_y, expected[:, :n]),
                      "padded_outputs_zero": bool(np.count_nonzero(y[:, n:]) == 0),
                      "finite": bool(np.isfinite(y).all())}
            if complete:
                record["error_vs_captured_full_moe"] = e.errors(valid_y, case["y"][:, begin:end])
            # A routing-control test, not another claimed real context: keep x
            # fixed, change one actually nonzero runtime expert coefficient.
            changed = scaled_coefficients.copy()
            first = int(case["selected_experts"][0, begin, 0])
            if first in lookup:
                changed[0, 0, lookup[first], 0] *= np.float16(0.5)
                changed_y = model.predict({"x": x, "coefficients": changed})["y"].astype(np.float32) / args.coefficient_scale
                delta = float(np.max(np.abs(changed_y - y)))
                record["dynamic_coefficient_check"] = {"same_x": True, "changed_expert": first,
                                                       "coefficient_multiplier": 0.5, "max_output_change": delta,
                                                       "output_changed": delta > 0}
                if not delta > 0:
                    raise AssertionError("Changing a nonzero runtime coefficient did not change output")
            if args.runs:
                for _ in range(3):
                    predict_host_boundary()
                ready_times, native_times, host_times = [], [], []
                for _ in range(args.runs):
                    t = time.perf_counter_ns()
                    predict_ready()
                    ready_times.append((time.perf_counter_ns() - t) / 1e6)
                    native_times.append(model.last_predict_duration_in_nano_seconds / 1e6)
                for _ in range(args.runs):
                    t = time.perf_counter_ns()
                    predict_host_boundary()
                    host_times.append((time.perf_counter_ns() - t) / 1e6)
                record.update({"ready_prediction_ms": ready_times, "coreml_prediction_from_features_ms": native_times,
                               "prediction_ms": host_times, "median_ready_prediction_ms": float(np.median(ready_times)),
                               "median_coreml_prediction_ms": float(np.median(native_times)),
                               "median_prediction_ms": float(np.median(host_times)),
                               "timing_boundary": "Host x copy, coefficient scaling/copy, Core ML prediction, output FP32 conversion/unscale; CPU 512-way router and shared-gate projection are outside this timed graph."})
            record["eligible_for_integration_gate"] = bool(complete and record["finite"] and
                record["error_vs_fp32_selected_bank"]["relative_l2"] is not None and
                record["error_vs_fp32_selected_bank"]["relative_l2"] <= 0.005 and
                record.get("median_prediction_ms", float("inf")) < 1.8)
            np.savez(args.output / f"{name}_{case_name}.npz", x=x, coefficients=scaled_coefficients,
                     unscaled_coefficients=coefficients, y=y, fp32=expected)
            row["cases"][case_name] = record
    row["status"] = "completed"
    row["all_real_tokens_covered"] = sum(c["valid_token_count"] for c in row["cases"].values()) == 3
    row["all_cases_pass_integration_gate"] = row["all_real_tokens_covered"] and all(
        c["eligible_for_integration_gate"] for c in row["cases"].values())
    del model
    gc.collect()
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experts", default="88,90")
    parser.add_argument("--shared", action="store_true")
    parser.add_argument("--capacities", default="1,32")
    parser.add_argument("--runs", type=int, default=0)
    parser.add_argument("--coefficient-scale", type=float, default=1)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--output", type=Path, default=e.ROOT / "ane-runner/results/moe-grouped-v2")
    args = parser.parse_args()
    ids = sorted(set(int(i) for i in args.experts.split(",")))
    capacities = sorted(set(int(i) for i in args.capacities.split(",")))
    if not ids or any(i < 0 or i >= e.EXPERTS for i in ids) or any(c < 1 for c in capacities) or args.runs < 0:
        parser.error("Invalid expert IDs, token capacities or runs")
    if args.coefficient_scale <= 0 or not np.isfinite(args.coefficient_scale):
        parser.error("Coefficient scale must be positive and finite")
    torch.set_num_threads(4)
    cases = {phase: dict(np.load(e.ROOT / f"ane-runner/fixtures/moe-real/converted/{phase}.npz")) for phase in ["prefill", "decode"]}
    actual_ids = set(int(v) for case in cases.values() for v in case["selected_experts"].ravel())
    missing = sorted(actual_ids - set(ids))
    if args.require_complete and (missing or not args.shared):
        parser.error(f"Full real-token coverage required: missing={missing}, shared={args.shared}")
    args.output.mkdir(parents=True, exist_ok=True)
    source = e.Source(0)
    weights = [source.expert(i) for i in ids]
    if args.shared:
        weights.append(source.shared())
    suffix = f"bank{len(ids)}_shared{int(args.shared)}_scale{args.coefficient_scale:g}"
    e.write_json(args.output / f"provenance_{suffix}.json", {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_records": source.records, "actual_expert_union": sorted(actual_ids),
        "missing_from_bank": missing, "require_complete": args.require_complete,
        "fixture_sha256": {phase: hashlib.sha256((e.ROOT / f"ane-runner/fixtures/moe-real/converted/{phase}.npz").read_bytes()).hexdigest() for phase in cases},
        "limitations": ["All bank experts are computed densely, including zero-coefficient experts.",
                        "This is a fixed 29-expert coverage probe, not a general sparse 512-expert implementation.",
                        "CPU full-router and shared-gate coefficients come from the complete native capture, outside timed graph.",
                        "ComputePlan preference is not runtime hardware trace."]})
    rows = []
    for capacity in capacities:
        try:
            row = evaluate_capacity(args, ids, weights, cases, capacity)
        except Exception as exc:
            row = {"capacity": capacity, "status": "failed", "error_type": type(exc).__name__,
                   "error": str(exc), "traceback": traceback.format_exc(),
                   "all_cases_pass_integration_gate": False}
            gc.collect()
        rows.append(row)
        e.write_json(args.output / f"report_{suffix}.json", rows)
        print(json.dumps({k: v for k, v in row.items() if k != "compute_plan"}), flush=True)


if __name__ == "__main__":
    main()
