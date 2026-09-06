#!/usr/bin/env python3
"""Pad one real expert's intermediate channels with zeros and inspect Core ML.

Default operation exports I1024/I1280, S1 and reads compute plans only. Prediction
requires --predict and a coordinated device window. Optional --coefficient-input
adds one runtime multiplier after SwiGLU; --middle-scales 1,64 then evaluates
host division by the same multiplier as separate numerical cases.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import time

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
import numpy as np
import torch
import torch.nn.functional as F

import export_moe as source_export


ROOT = Path(__file__).resolve().parents[1]
HIDDEN, ORIGINAL_INTERMEDIATE = 2560, 640


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def padded_weights(original: dict, intermediate: int) -> dict:
    result = {
        "gate_proj": np.zeros((intermediate, HIDDEN), np.float16),
        "up_proj": np.zeros((intermediate, HIDDEN), np.float16),
        "down_proj": np.zeros((HIDDEN, intermediate), np.float16),
    }
    for name in ("gate_proj", "up_proj"):
        result[name][:ORIGINAL_INTERMEDIATE] = original[name]
        assert np.array_equal(result[name][:ORIGINAL_INTERMEDIATE], original[name])
        assert np.count_nonzero(result[name][ORIGINAL_INTERMEDIATE:]) == 0
    result["down_proj"][:, :ORIGINAL_INTERMEDIATE] = original["down_proj"]
    assert np.array_equal(result["down_proj"][:, :ORIGINAL_INTERMEDIATE], original["down_proj"])
    assert np.count_nonzero(result["down_proj"][:, ORIGINAL_INTERMEDIATE:]) == 0
    return result


def make_program(weights: dict, capacity: int, coefficient_input: bool):
    def graph(x, coefficient=None):
        value = mb.reshape(x=mb.transpose(x=x, perm=[0, 3, 2, 1]), shape=[1, capacity, HIDDEN])
        gate = mb.linear(x=value, weight=weights["gate_proj"], name="gate_projection")
        up = mb.linear(x=value, weight=weights["up_proj"], name="up_projection")
        denominator = mb.add(x=mb.exp(x=mb.mul(x=gate, y=np.float16(-1))), y=np.float16(1))
        activated = mb.real_div(x=gate, y=denominator, name="silu_explicit")
        middle = mb.mul(x=activated, y=up, name="gated_up")
        if coefficient is not None:
            middle = mb.mul(x=middle, y=coefficient, name="runtime_middle_scale")
        down = mb.linear(x=middle, weight=weights["down_proj"], name="down_projection")
        return mb.transpose(x=mb.reshape(x=down, shape=[1, capacity, 1, HIDDEN]), perm=[0, 3, 2, 1], name="y")

    specs = [mb.TensorSpec(shape=(1, HIDDEN, 1, capacity), dtype=types.fp16)]
    if coefficient_input:
        specs.append(mb.TensorSpec(shape=(1, capacity, 1), dtype=types.fp16))
        @mb.program(input_specs=specs, opset_version=ct.target.macOS15)
        def program(x, coefficient):
            return graph(x, coefficient)
    else:
        @mb.program(input_specs=specs, opset_version=ct.target.macOS15)
        def program(x):
            return graph(x)
    return program


def torch_reference(original: dict, x: np.ndarray) -> np.ndarray:
    # Original, unpadded matrices and independent Torch SwiGLU in FP32.
    weights = {name: torch.from_numpy(value.astype(np.float32)) for name, value in original.items()}
    with torch.inference_mode():
        value = torch.from_numpy(x.astype(np.float32).transpose(0, 3, 2, 1).reshape(1, x.shape[-1], HIDDEN))
        gate, up = F.linear(value, weights["gate_proj"]), F.linear(value, weights["up_proj"])
        y = F.linear(F.silu(gate) * up, weights["down_proj"])
    return y.numpy().transpose(0, 2, 1)[:, :, None, :]


def input_cases(fixture_dir: Path, capacity: int) -> list:
    cases = []
    for phase in ("decode", "prefill"):
        path = fixture_dir / f"{phase}.npz"
        with np.load(path) as data:
            x, positions = data["x"], data["absolute_positions"].tolist()
            tokens = data["token_ids"].reshape(-1).tolist()
            routed = data["selected_experts"][0].tolist()
        for begin in range(0, x.shape[1], capacity):
            count = min(capacity, x.shape[1] - begin)
            value = np.zeros((1, HIDDEN, 1, capacity), np.float16)
            value[0, :, 0, :count] = x[0, begin:begin + count].T.astype(np.float16)
            cases.append({"name": f"{phase}_{begin}", "x": value, "valid_tokens": count,
                          "fixture": str(path), "fixture_sha256": digest(path),
                          "absolute_positions": positions[begin:begin + count], "token_ids": tokens[begin:begin + count],
                          "native_selected_experts": routed[begin:begin + count]})
    cases.append({"name": "zero", "x": np.zeros((1, HIDDEN, 1, capacity), np.float16), "valid_tokens": capacity})
    return cases


def paired_benchmark(model, baseline, original, fixture_dir, capacity, coefficient_input, scale):
    """Same real decode row, prepared inputs, equal valid output conversion."""
    case = input_cases(fixture_dir, capacity)[0]
    x = case["x"]
    baseline_x = np.zeros((1, HIDDEN, 1, 32), np.float16)
    baseline_x[..., 0] = x[..., 0]
    ready = {"x": x}
    if coefficient_input:
        ready["coefficient"] = np.full((1, capacity, 1), scale, dtype=np.float16)
    def padded_call():
        return model.predict(ready)["y"][..., :1].astype(np.float32) / scale
    def baseline_call():
        return baseline.predict({"x": baseline_x})["y"][..., :1].astype(np.float32)
    functions = {"padded": padded_call, "original_s32": baseline_call}
    expected = torch_reference(original, x)[..., :1]
    numerical = {name: source_export.errors(fn(), expected) for name, fn in functions.items()}
    groups = []
    samples = {name: [] for name in functions}
    for round_index, order in enumerate((list(functions), list(reversed(functions))), start=1):
        for name in order:
            fn = functions[name]
            for _ in range(3):
                fn()
            values = []
            for _ in range(10):
                started = time.perf_counter_ns()
                fn()
                values.append((time.perf_counter_ns() - started) / 1e6)
            samples[name].extend(values)
            groups.append({"round": round_index, "kind": name, "milliseconds": values,
                           "median_ms": statistics.median(values), "calls": 10})
    medians = {name: statistics.median(values) for name, values in samples.items()}
    return {"middle_scale": scale, "groups": groups, "medians_ms": medians,
            "speedup_vs_original_s32": medians["original_s32"] / medians["padded"],
            "errors_vs_original_fp32_fp16_weights": numerical,
            "numerical_gate_0_5_percent": numerical["padded"]["relative_l2"] <= 0.005,
            "speed_gate": medians["padded"] < medians["original_s32"],
            "boundary": "Single expert with prepared FP16 x; Core ML Python predict plus valid-row FP32 conversion and optional host unscale. Input padding/materialization, router and full MoE scheduling excluded. No ANE trace."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert", type=int, default=88)
    parser.add_argument("--intermediates", default="1024,1280")
    parser.add_argument("--capacities", default="1")
    parser.add_argument("--coefficient-input", action="store_true")
    parser.add_argument("--middle-scales", default="1")
    parser.add_argument("--fixture-dir", type=Path, default=ROOT / "fixtures/moe-real/converted")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/moe-padding-v2")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--benchmark", action="store_true", help="Exclusive-window paired decode comparison against original S32 expert, 2 rounds x10 calls")
    args = parser.parse_args()
    intermediates = list(dict.fromkeys(int(v) for v in args.intermediates.split(",")))
    capacities = list(dict.fromkeys(int(v) for v in args.capacities.split(",")))
    scales = list(dict.fromkeys(float(v) for v in args.middle_scales.split(",")))
    if any(v < ORIGINAL_INTERMEDIATE or v > 1280 for v in intermediates) or any(v not in (1, 8) for v in capacities):
        parser.error("Bounded probe supports 640..1280 intermediate channels and S1/S8 only")
    if any(v not in (1.0, 64.0) for v in scales) or (not args.coefficient_input and scales != [1.0]):
        parser.error("Scales are 1 or 64; non-unit scale requires --coefficient-input")
    if not 0 <= args.expert < 512:
        parser.error("Expert must be in 0..511")
    if args.benchmark and not args.predict:
        parser.error("--benchmark requires --predict")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / ("report-prediction.json" if args.predict else "report-plan.json")
    report = {"schema": "moe-zero-padding-v2", "phase": "predict" if args.predict else "plan_only",
              "expert": args.expert, "records": [],
              "scope": "One real expert, not a whole MoE or decoder. Compute-plan preference does not establish hardware residency.",
              "input_scope": "All nonzero inputs are captured real HC activations. Only rows listing the tested expert in native_selected_experts were actually sent to this expert by native routing; other rows are sensitivity checks.",
              "math": "FP16-rounded source Q4 affine weights retained unchanged. Append zero gate/up rows and matching zero down columns; no real channel is discarded. Native FP16 arithmetic can still differ from the unpadded reduction.",
              "coefficient_input": args.coefficient_input,
              "scale_semantics": "If enabled, runtime scalar multiplies SwiGLU middle; host converts output to FP32 and divides by same scalar. Numerical cases are kept separate."}
    try:
        torch.set_num_threads(1)
        source = source_export.Source(0)
        extracted = source.expert(args.expert)
        original = {name: values["dense16"] for name, values in extracted.items()}
        weight_hashes = {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in original.items()}
        report["source"] = {"model_directory": str(source.directory), "prior_verification": str(source.verification_path),
                            "prior_verification_sha256": digest(source.verification_path), "tensor_slices": source.records,
                            "dense16_sha256": weight_hashes, "script_sha256": digest(Path(__file__))}
        baseline = None
        if args.benchmark:
            baseline_path = ROOT / f"results/moe-export/layer_0_fp16_linear_s32/expert_{args.expert:04d}.mlpackage"
            baseline = ct.models.MLModel(str(baseline_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
            report["baseline"] = {"package": str(baseline_path), "capacity": 32}
        for intermediate in intermediates:
            weights = padded_weights(original, intermediate)
            for capacity in capacities:
                tag = f"expert{args.expert}_i{intermediate}_s{capacity}" + ("_coefficient" if args.coefficient_input else "")
                package = args.output_dir / f"{tag}.mlpackage"
                metadata_path = args.output_dir / f"{tag}-export.json"
                signature = {"expert": args.expert, "intermediate": intermediate, "capacity": capacity,
                             "coefficient_input": args.coefficient_input, "dense16_sha256": weight_hashes,
                             "script_sha256": digest(Path(__file__))}
                reused = package.exists() and metadata_path.exists() and json.loads(metadata_path.read_text()) == signature
                if reused:
                    model = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_AND_NE)
                else:
                    model = ct.convert(make_program(weights, capacity, args.coefficient_input), convert_to="mlprogram",
                                       minimum_deployment_target=ct.target.macOS15, compute_precision=ct.precision.FLOAT16,
                                       compute_units=ct.ComputeUnit.CPU_AND_NE)
                    model.save(str(package))
                    write_json(metadata_path, signature)
                plan = source_export.compute_plan(model)
                row = {"intermediate": intermediate, "capacity": capacity, "package": str(package.resolve()),
                       "reused_package": reused, "package_bytes": source_export.package_bytes(package),
                       "parameter_expansion": intermediate / ORIGINAL_INTERMEDIATE,
                       "compute_plan": plan, "preferred_counts": dict(Counter(op["preferred"] for op in plan)),
                       "all_ops_prefer_ane": bool(plan) and all(op["preferred"] == "MLNeuralEngineComputeDevice" for op in plan),
                       "prediction_cases": []}
                if args.predict:
                    for scale in scales:
                        for case in input_cases(args.fixture_dir, capacity):
                            x = case["x"]
                            inputs = {"x": x}
                            if args.coefficient_input:
                                inputs["coefficient"] = np.full((1, capacity, 1), scale, dtype=np.float16)
                            expected = torch_reference(original, x)
                            started = time.perf_counter_ns()
                            raw = model.predict(inputs)["y"]
                            elapsed = (time.perf_counter_ns() - started) / 1e6
                            y = raw.astype(np.float32) / scale
                            count = case["valid_tokens"]
                            output_path = args.output_dir / f"{tag}-scale{scale:g}-{case['name']}.npz"
                            np.savez(output_path, x=x, raw_y=raw, y=y, reference_fp32=expected)
                            error = source_export.errors(y[..., :count], expected[..., :count])
                            row["prediction_cases"].append({
                                **{k: v for k, v in case.items() if k != "x"}, "middle_scale": scale,
                                "finite": bool(np.isfinite(y).all()), "error_vs_original_fp32_fp16_weights": error,
                                "padded_rows_exact_zero": bool(np.count_nonzero(y[..., count:]) == 0),
                                "zero_exact": bool(np.count_nonzero(y) == 0) if case["name"] == "zero" else None,
                                "single_prediction_ms_not_benchmark": elapsed, "output": str(output_path.resolve()),
                            })
                    if args.benchmark:
                        row["paired_benchmarks"] = [paired_benchmark(model, baseline, original, args.fixture_dir,
                                                    capacity, args.coefficient_input, scale) for scale in scales]
                report["records"].append(row)
                write_json(report_path, report)
                print(json.dumps({k: v for k, v in row.items() if k != "compute_plan"}), flush=True)
                del model
        report["status"] = "complete"
    except Exception as error:
        report["status"], report["error"] = "failed", str(error)
        write_json(report_path, report)
        raise
    write_json(report_path, report)
    print(json.dumps({"status": report["status"], "report": str(report_path.resolve())}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
