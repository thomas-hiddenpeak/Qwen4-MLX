#!/usr/bin/env python3
"""Bounded public Core ML layout probes for the same complete real expert."""
import argparse
from collections import Counter
import gc
from pathlib import Path
import json
import time

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
import numpy as np
import torch
import export_moe as e


def program(weights, capacity, kind):
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, e.HIDDEN, 1, capacity), dtype=types.fp16)],
                opset_version=ct.target.macOS15)
    def graph(x):
        if kind == "linear":
            value = mb.reshape(x=mb.transpose(x=x, perm=[0, 3, 2, 1]), shape=[1, capacity, e.HIDDEN])
            gate = mb.linear(x=value, weight=weights["gate_proj"]["dense16"])
            up = mb.linear(x=value, weight=weights["up_proj"]["dense16"])
        else:
            combined = np.concatenate((weights["gate_proj"]["dense16"], weights["up_proj"]["dense16"]))[:, :, None, None]
            both = mb.conv(x=x, weight=combined)
            gate, up = mb.split(x=both, num_splits=2, axis=1)
        denominator = mb.add(x=mb.exp(x=mb.mul(x=gate, y=np.float16(-1))), y=np.float16(1))
        middle = mb.mul(x=mb.real_div(x=gate, y=denominator), y=up)
        if kind == "linear":
            y = mb.linear(x=middle, weight=weights["down_proj"]["dense16"])
            return mb.transpose(x=mb.reshape(x=y, shape=[1, capacity, 1, e.HIDDEN]), perm=[0, 3, 2, 1], name="y")
        return mb.conv(x=middle, weight=weights["down_proj"]["dense16"][:, :, None, None], name="y")
    return graph


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=e.DEFAULT_OUTPUT / "layout-probe")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    weights = e.Source(0).expert(0)
    rows = []
    for kind, capacity in (("linear", 1), ("linear", 32), ("fused_gate_up", 1), ("fused_gate_up", 32), ("fused_gate_up", 128)):
        model = ct.convert(program(weights, capacity, kind), convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                           compute_units=ct.ComputeUnit.CPU_AND_NE, compute_precision=ct.precision.FLOAT16)
        path = args.output / f"expert0_{kind}_s{capacity}.mlpackage"
        model.save(str(path))
        x = np.random.default_rng(20260905).standard_normal((1, e.HIDDEN, 1, capacity)).astype(np.float16)
        y = model.predict({"x": x})["y"]
        reference = e.reference(weights, x, "fp32_fp16_weights")
        plan = e.compute_plan(model)
        row = {"kind": kind, "capacity": capacity, "model": str(path),
               "error_vs_fp32_fp16_weights": e.errors(y, reference), "compute_plan": plan,
               "plan_preferred_counts": dict(Counter(p["preferred"] for p in plan)),
               "zero_exact": bool(np.count_nonzero(model.predict({"x": np.zeros_like(x)})["y"]) == 0)}
        for _ in range(3): model.predict({"x": x})
        durations = []
        for _ in range(5):
            start = time.perf_counter_ns(); model.predict({"x": x}); durations.append((time.perf_counter_ns() - start) / 1e6)
        row["prediction_median_ms"] = float(np.median(durations))
        rows.append(row)
        e.write_json(args.output / "report.json", rows)
        print(json.dumps({k: v for k, v in row.items() if k != "compute_plan"}), flush=True)
        del model
        gc.collect()


if __name__ == "__main__":
    main()
