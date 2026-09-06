# /// script
# requires-python = ">=3.12"
# dependencies = ["mlx==0.32.2", "numpy>=2,<3"]
# ///
"""One real layer-0 MoE bank, using public Python MLX APIs; never a full model.

Run with the existing cached environment:
  uv run --offline --no-project scripts/benchmark_moe_mlx.py
The native capture remains the numerical oracle; this is NOT a timing replica
of the author's custom fused Metal router / gather-QMV implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import struct
import time

import mlx.core as mx
import numpy as np


class TensorReader:
    def __init__(self):
        self.headers = {}

    def header(self, path):
        if path not in self.headers:
            with path.open("rb") as handle:
                size = struct.unpack("<Q", handle.read(8))[0]
                if size > 16 * 1024 * 1024:
                    raise ValueError(f"Excessive tensor header: {path}")
                self.headers[path] = (8 + size, json.loads(handle.read(size)))
        return self.headers[path]

    def read(self, path, key):
        offset, header = self.header(path)
        entry = header[key]
        start, end = entry["data_offsets"]
        with path.open("rb") as handle:
            handle.seek(offset + start)
            data = handle.read(end - start)
        if len(data) != end - start:
            raise ValueError(f"Truncated tensor {path}:{key}")
        dtype = {"U32": "<u4", "I32": "<i4", "BF16": "<u2", "F32": "<f4"}[entry["dtype"]]
        values = np.frombuffer(data, dtype=dtype).reshape(entry["shape"])
        if entry["dtype"] == "BF16":
            values = (values.astype(np.uint32) << 16).view(np.float32)
        return values, {"source_file": str(path), "key": key, "dtype": entry["dtype"],
                        "shape": entry["shape"], "bytes": len(data),
                        "tensor_sha256": hashlib.sha256(data).hexdigest()}


class MoE:
    def __init__(self, model):
        reader = TensorReader()
        mapping = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
        self.weights = {}
        self.evidence = []
        prefix = "language_model.model.layers.0.mlp."
        keys = ["gate.weight", "shared_expert_gate.weight"]
        keys += [f"switch_mlp.{projection}.{part}" for projection in ("gate_proj", "up_proj", "down_proj")
                 for part in ("weight", "scales", "biases")]
        keys += [f"shared_expert.{projection}.weight" for projection in ("gate_proj", "up_proj", "down_proj")]
        for key in keys:
            full = prefix + key
            host, evidence = reader.read(model / mapping[full], full)
            dtype = mx.uint32 if evidence["dtype"] == "U32" else mx.bfloat16
            array = mx.array(host, dtype=dtype)
            mx.eval(array)
            self.weights[key] = array
            self.evidence.append(evidence)
            del host
        assert self.weights["gate.weight"].shape == (512, 2560)
        assert self.weights["switch_mlp.gate_proj.weight"].shape == (512, 640, 320)
        assert self.weights["switch_mlp.down_proj.weight"].shape == (512, 2560, 80)

    def dense(self, x, name):
        return x @ self.weights[name].T

    def expert(self, x, projection, indices, sorted_indices):
        base = f"switch_mlp.{projection}"
        return mx.gather_qmm(x, self.weights[base + ".weight"],
                             self.weights[base + ".scales"], self.weights[base + ".biases"],
                             rhs_indices=indices, transpose=True, group_size=64, bits=4,
                             mode="affine", sorted_indices=sorted_indices)

    @staticmethod
    def swiglu(gate, up):
        return (gate * mx.sigmoid(gate)) * up

    def __call__(self, x):
        tokens, hidden, topk = x.shape[1], 2560, 10
        logits = self.dense(x, "gate.weight")
        # Public argsort, rather than the author's fused router. Actual ID
        # order/set agreement is measured against the saved native capture.
        ids = mx.argsort(-logits, axis=-1)[..., :topk]
        probabilities = mx.softmax(logits, axis=-1, precise=True)
        selected = mx.take_along_axis(probabilities, ids, axis=-1)
        denominator = mx.zeros(selected.shape[:-1], dtype=mx.bfloat16)
        for slot in range(topk):
            denominator = (denominator + selected[..., slot]).astype(mx.bfloat16)
        weights = (selected / denominator[..., None]).astype(mx.bfloat16)
        if tokens > 1:
            flat = ids.reshape(-1)
            order = mx.argsort(flat)
            inverse = mx.argsort(order)
            sorted_ids = mx.take(flat, order)
            gathered = mx.take(x.reshape(tokens, hidden), order // topk, axis=0)[:, None, :]
            gate = self.expert(gathered, "gate_proj", sorted_ids, True).squeeze(-2)
            up = self.expert(gathered, "up_proj", sorted_ids, True).squeeze(-2)
            act = self.swiglu(gate, up)
            down = self.expert(act[:, None, :], "down_proj", sorted_ids, True).squeeze(-2)
            experts = mx.take(down, inverse, axis=0).reshape(1, tokens, topk, hidden)
        else:
            expanded = x.reshape(1, 1, 1, 1, hidden)
            gate = self.expert(expanded, "gate_proj", ids, False).squeeze(-2)
            up = self.expert(expanded, "up_proj", ids, False).squeeze(-2)
            act = self.swiglu(gate, up)
            experts = self.expert(act[..., None, :], "down_proj", ids, False).squeeze(-2)
        routed = mx.sum(experts * weights[..., None], axis=-2)
        shared = self.dense(self.swiglu(self.dense(x, "shared_expert.gate_proj.weight"),
                                       self.dense(x, "shared_expert.up_proj.weight")),
                            "shared_expert.down_proj.weight")
        shared_logits = self.dense(x, "shared_expert_gate.weight")
        shared_gate = mx.sigmoid(shared_logits)
        shared_gated = shared * shared_gate
        output = routed + shared_gated
        return {"router_logits": logits, "selected_experts": ids, "routing_weights": weights,
                "routed_sum": routed, "shared_down": shared, "shared_gate_logits": shared_logits,
                "shared_gate": shared_gate, "shared_gated": shared_gated, "output": output}


def host(array):
    if array.dtype in (mx.uint32, mx.int32):
        return np.array(array)
    return np.array(array.astype(mx.float32))


def comparison(actual, expected):
    a, b = actual.astype(np.float64), expected.astype(np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch {a.shape} != {b.shape}")
    difference = a - b
    norm = np.linalg.norm(b.reshape(-1))
    return {"shape": list(a.shape), "finite": bool(np.all(np.isfinite(a))),
            "exact": bool(np.array_equal(a, b)), "max_absolute_error": float(np.max(np.abs(difference))),
            "relative_l2_error": float(np.linalg.norm(difference.reshape(-1)) / norm) if norm else None}


def timing_summary(values):
    return {"samples_ms": values, "median_ms": float(np.median(values)),
            "p10_ms": float(np.percentile(values, 10)), "p90_ms": float(np.percentile(values, 90))}


def main():
    base = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=base.parent / "qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream")
    parser.add_argument("--capture", type=Path, default=base / "fixtures/moe-real")
    parser.add_argument("--results", type=Path, default=base / "results/moe-mlx")
    parser.add_argument("--prefill-indices", type=int, nargs="+", default=[14, 25])
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()
    if args.warmups < 0 or not 1 <= args.runs <= 100:
        parser.error("Require warmups>=0 and 1<=runs<=100")
    mx.set_default_device(mx.gpu)
    args.results.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print("Loading only the layer-0 packed Q4 MoE bank and router/shared weights", flush=True)
    moe = MoE(args.model)
    load_seconds = time.perf_counter() - started
    reader = TensorReader()
    results = {}
    for phase, positions in (("prefill", args.prefill_indices), ("decode", [0])):
        path = args.capture / f"{phase}.safetensors"
        _, header = reader.header(path)
        if header.get("__metadata__", {}).get("schema") != "qwen4-layer0-moe-real-v1":
            raise ValueError("Expected native real-request capture")
        capture = {key: reader.read(path, key)[0] for key in header if key != "__metadata__"}
        if not positions or len(set(positions)) != len(positions) or any(p < 0 or p >= capture["x"].shape[1] for p in positions):
            raise ValueError(f"Invalid real token positions for {phase}: {positions}")
        x_host = np.ascontiguousarray(capture["x"][:, positions, :])
        x = mx.array(x_host, dtype=mx.bfloat16)
        mx.eval(x)
        for _ in range(args.warmups):
            warmed = moe(x)
            mx.eval(warmed["output"])
        samples = []
        for _ in range(args.runs):
            start = time.perf_counter_ns()
            actual = moe(x)
            mx.eval(actual["output"])
            samples.append((time.perf_counter_ns() - start) / 1e6)
        # Extra host boundary series permits a clearer comparison with the
        # Swift/Core ML executor; both series include Python graph construction.
        roundtrip = []
        for _ in range(args.runs):
            start = time.perf_counter_ns()
            new_x = mx.array(x_host, dtype=mx.bfloat16)
            y = moe(new_x)["output"]
            mx.eval(y)
            materialized = host(y)
            roundtrip.append((time.perf_counter_ns() - start) / 1e6)
        mx.eval(*actual.values())
        actual_host = {key: host(value) for key, value in actual.items()}
        reference = {key: capture[key][:, positions, ...] for key in actual}
        comparisons = {key: comparison(value, reference[key]) for key, value in actual_host.items()}
        ids, oracle = actual_host["selected_experts"], reference["selected_experts"]
        set_match = np.array_equal(np.sort(ids, axis=-1), np.sort(oracle, axis=-1))
        aligned_a, aligned_b = [], []
        if set_match:
            for token in range(len(positions)):
                for slot, expert_id in enumerate(oracle[0, token]):
                    found = int(np.flatnonzero(ids[0, token] == expert_id)[0])
                    aligned_a.append(actual_host["routing_weights"][0, token, found])
                    aligned_b.append(reference["routing_weights"][0, token, slot])
        result = {
            "capture_file": str(path), "capture_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "positions_within_capture": positions, "sequence_offset": int(capture["sequence_offset"]),
            "token_ids": capture["token_ids"][:, positions].tolist(),
            "selected_expert_ids": ids.tolist(), "routing_sets_match": bool(set_match),
            "comparisons": comparisons,
            "routing_weights_aligned_by_expert": comparison(np.array(aligned_a), np.array(aligned_b)) if set_match else None,
            "resident_input_dispatch_and_sync": timing_summary(samples),
            "host_input_output_roundtrip": timing_summary(roundtrip),
        }
        results[phase] = result
        np.savez(args.results / f"{phase}-outputs.npz", **actual_host)
        print(json.dumps({"phase": phase, "resident_median_ms": np.median(samples),
                          "host_median_ms": np.median(roundtrip), "routing_sets_match": bool(set_match),
                          "output_relative_l2": comparisons["output"]["relative_l2_error"]}), flush=True)
    report = {
        "schema": "real-layer0-python-mlx-moe-benchmark-v1", "mlx_version": importlib.metadata.version("mlx"),
        "numpy_version": np.__version__, "python": platform.python_version(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model": str(args.model), "weight_load_seconds": load_seconds,
        "weight_source_bytes": sum(w["bytes"] for w in moe.evidence), "weights": moe.evidence,
        "warmups": args.warmups, "runs_per_series": args.runs,
        "mlx_active_bytes": mx.get_active_memory(), "mlx_peak_bytes": mx.get_peak_memory(),
        "cases": results,
        "notes": [
            "Only the real layer-0 packed expert bank, router and shared expert were loaded. No full-model server was launched.",
            "Dynamic routing considers all 512 experts; every selected top-10 assignment is computed without dropping tokens.",
            "Public Python MLX argsort/gather_qmm/SwiGLU differs from the author's custom fused router/gather-QMV implementation. Timings are not native-fork kernel timings.",
            "Original BF16 activations and coefficients are retained. Arithmetic/reduction differences from native fused kernels are measured, not assumed absent.",
            "Resident series includes Python graph construction, dispatch and mx.eval(final output), with GPU-resident input/weights. It is not pure GPU kernel time.",
            "Host roundtrip additionally uploads BF16 input and materializes the returned output as Float32 NumPy. No JSON, model compilation, or weight loading is timed.",
            "These are selected real MoE token rows, not an independent end-to-end model quality or generation-speed test.",
        ],
    }
    report["passed_numerical_smoke"] = all(case["routing_sets_match"] and all(c["finite"] for c in case["comparisons"].values()) for case in results.values())
    (args.results / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not report["passed_numerical_smoke"]:
        raise SystemExit("Finite/routing smoke check failed; inspect report. This is not a comprehensive quality test.")


if __name__ == "__main__":
    main()
