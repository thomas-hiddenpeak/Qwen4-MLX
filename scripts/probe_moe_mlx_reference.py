#!/usr/bin/env python3
"""Small MLX QMM/decoded-weight diagnostic; run in the existing MLX environment."""
import argparse
import json
from pathlib import Path
import mlx.core as mx
import numpy as np


def error(actual, expected):
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    return {"relative_l2": float(np.linalg.norm(delta) / np.linalg.norm(expected.astype(np.float64))),
            "max_abs": float(np.max(np.abs(delta)))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    source = np.load(args.source)
    fixture = json.loads(args.fixture.read_text())["inputs"]["x"]
    bc1s = np.array(fixture["values"], np.float32).reshape(fixture["shape"])
    x32 = bc1s[:, :, 0].transpose(0, 2, 1)
    result = {}
    for label, dtype in (("fp32", mx.float32), ("fp16", mx.float16), ("bf16", mx.bfloat16)):
        x = mx.array(x32).astype(dtype)
        params = {}
        for name in ("gate_proj", "up_proj", "down_proj"):
            params[name] = (mx.array(source[name + "_weight"]),
                            mx.array(source[name + "_scales"]).astype(dtype),
                            mx.array(source[name + "_biases"]).astype(dtype))
        def linear(v, name):
            w, s, b = params[name]
            return mx.quantized_matmul(v, w, s, b, transpose=True, group_size=64, bits=4)
        gate, up = linear(x, "gate_proj"), linear(x, "up_proj")
        middle = (gate * mx.sigmoid(gate)) * up
        y = linear(middle, "down_proj")
        # Independent affine FP32 weights, evaluated against the same rounded input.
        dense = {}
        for name in params:
            w, s, b = params[name]
            dense[name] = mx.dequantize(w, s.astype(mx.float32), b.astype(mx.float32), group_size=64, bits=4, dtype=mx.float32)
        xr = x.astype(mx.float32)
        gr, ur = xr @ dense["gate_proj"].T, xr @ dense["up_proj"].T
        yr = ((gr * mx.sigmoid(gr)) * ur) @ dense["down_proj"].T
        mx.eval(gate, up, y, gr, ur, yr)
        arrays = {"gate": np.array(gate.astype(mx.float32)), "up": np.array(up.astype(mx.float32)),
                  "y": np.array(y.astype(mx.float32)), "gate_reference": np.array(gr),
                  "up_reference": np.array(ur), "y_reference": np.array(yr)}
        result[label] = {"qmm_gate_vs_affine_fp32": error(arrays["gate"], arrays["gate_reference"]),
                         "qmm_up_vs_affine_fp32": error(arrays["up"], arrays["up_reference"]),
                         "qmm_y_vs_affine_fp32": error(arrays["y"], arrays["y_reference"])}
        np.savez(args.output.with_name(args.output.stem + "_" + label + ".npz"), **arrays)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
