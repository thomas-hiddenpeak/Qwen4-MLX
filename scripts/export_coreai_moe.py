#!/usr/bin/env python3
"""Export one real Qwen shared/routed SwiGLU to fixed-shape FP16 CoreAI assets.

CPU authoring only: this script never invokes CoreAI/Core ML/MLX inference.
The system CoreAI runtime requires macOS 27; export success is not runtime or
hardware-placement evidence. Dependencies: numpy, torch (validated 2.9.0),
coreai-core (1.0.0b2), and coreai-torch (0.4.1). coremltools
is also required because the reused export_moe module imports it; none of its
conversion or prediction functions are called here.

Requires the existing sibling qwen38-ssd source manifest and verified shards.
Only one shared expert or ONE routed expert is read, never all 512 experts.
Routed Q4 weights are expanded to FP16, not preserved as compressed constants.
--fp32-projections promotes only the three matrix products; weights, I/O,
SiLU, and intermediate rounding remain FP16. Use a fresh --output for this
variant. Its CPU numerical benefit does not establish Neural Engine accuracy.

Example (from this repository, using a Python environment with dependencies):
  python scripts/export_coreai_moe.py --output results/coreai-moe --kind shared
  python scripts/export_coreai_moe.py --output results/coreai-moe --kind routed --expert 333
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
import torch.nn.functional as F

from export_moe import EXPERTS, PROJECTIONS, SOURCE_ROOT, Source, sha256_file, write_json


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCRIPT = Path(__file__).with_name("export_moe.py")
REFERENCE_NAMES = ("fp16", "fp32_fp16_weights", "fp32_source_weights")


class SwiGLU(torch.nn.Module):
    def __init__(self, weights, key="dense16", dtype=torch.float16, fp32_projections=False):
        super().__init__()
        self.fp32_projections = fp32_projections
        for name in PROJECTIONS:
            self.register_buffer(name, torch.from_numpy(weights[name][key].copy()).to(dtype=dtype, device="cpu"))

    def linear(self, x, weight):
        if self.fp32_projections:
            return F.linear(x.float(), weight.float()).to(dtype=x.dtype)
        return F.linear(x, weight)

    def forward(self, x):
        gate = self.linear(x, self.gate_proj)
        up = self.linear(x, self.up_proj)
        return self.linear(F.silu(gate) * up, self.down_proj)


def errors(actual, expected):
    if actual.shape != expected.shape:
        raise ValueError(f"Comparison shape mismatch: {actual.shape} != {expected.shape}")
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError("Nonfinite tensor in numerical comparison")
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    norm = np.linalg.norm(expected.astype(np.float64))
    return {"max_abs": float(np.abs(delta).max()), "rms": float(np.sqrt(np.mean(delta ** 2))),
            "relative_l2": float(np.linalg.norm(delta) / norm) if norm else None,
            "all_finite": True}


def tensor_json(value):
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "values": value.reshape(-1).astype(float).tolist()}


def fixture_inputs(fixture_dir, capacity, hidden, weight_layer):
    """Retain captured token order and identify cross-layer replay explicitly."""
    phase = "decode" if capacity == 1 else "prefill"
    path = fixture_dir / f"{phase}.json"
    provenance_path = fixture_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    phase_metadata = provenance["phases"][phase]
    captured_layer = int(phase_metadata["source_metadata"]["layer_index"])
    digest = sha256_file(path)
    if phase_metadata["fixture_sha256"] != digest:
        raise ValueError(f"Fixture hash does not match capture provenance: {path}")
    fixture = json.loads(path.read_text())
    stored = fixture["inputs"]["x"]
    value = np.asarray(stored["values"], dtype=np.float32).reshape(stored["shape"])
    if value.ndim != 3 or value.shape[0] != 1 or value.shape[1] < 1 or value.shape[2] != hidden:
        raise ValueError(f"Expected nonempty [1,S,{hidden}] fixture, got {value.shape}")
    value = value.astype(np.float16)
    if not np.isfinite(value).all():
        raise ValueError("Captured input is nonfinite after FP16 conversion")
    valid_tokens = min(capacity, value.shape[1])
    x = np.zeros((1, capacity, hidden), dtype=np.float16)
    x[:, :valid_tokens] = value[:, :valid_tokens]
    rng = np.random.default_rng(20260917 + capacity)
    cases = {"actual": x, "zero": np.zeros_like(x),
             "normal": (0.5 * rng.standard_normal(x.shape)).astype(np.float16)}
    evidence = {"path": str(path), "sha256": digest,
                "provenance": str(provenance_path), "provenance_sha256": sha256_file(provenance_path),
                "input_capture_layer": captured_layer, "matches_weight_layer": captured_layer == weight_layer,
                "captured_tokens": value.shape[1], "valid_tokens": valid_tokens,
                "padding_tokens": capacity - valid_tokens,
                "input_scope": "captured layer input" if captured_layer == weight_layer else "cross-layer activation replay"}
    return cases, fixture, evidence


def reference_models(weights):
    return {"fp16": SwiGLU(weights).eval(),
            "fp32_fp16_weights": SwiGLU(weights, dtype=torch.float32).eval(),
            "fp32_source_weights": SwiGLU(weights, key="dense32", dtype=torch.float32).eval()}


def cpu_references(models, x):
    with torch.inference_mode():
        refs = {name: models[name](torch.from_numpy(x.copy()).to(
                    device="cpu", dtype=torch.float16 if name == "fp16" else torch.float32)).float().numpy()
                for name in REFERENCE_NAMES}
    if not all(np.isfinite(value).all() for value in refs.values()):
        raise ValueError("Nonfinite CPU reference")
    return refs


def save_cases(directory, inputs, models, fixture, evidence, kind):
    cases = []
    for case, x in inputs.items():
        refs = cpu_references(models, x)
        path = directory / f"{case}.npz"
        np.savez(path, x=x, **{f"y_{name}": value for name, value in refs.items()})
        json_path = directory / f"{case}.json"
        write_json(json_path, {"inputs": {"x": tensor_json(x)},
                               "expectedOutputs": {"y": tensor_json(refs["fp32_fp16_weights"])}})
        row = {"case": case, "npz": str(path), "npz_sha256": sha256_file(path),
               "json": str(json_path), "json_sha256": sha256_file(json_path),
               "input_shape": list(x.shape), "valid_tokens": evidence["valid_tokens"] if case == "actual" else x.shape[1],
               "fp16_vs_fp32_fp16_weights": errors(refs["fp16"], refs["fp32_fp16_weights"]),
               "fp32_fp16_weights_vs_source": errors(refs["fp32_fp16_weights"], refs["fp32_source_weights"])}
        if case == "actual":
            row["fixture"] = evidence
            captured = fixture.get("expectedOutputs", {}).get("shared_down")
            if kind == "shared" and evidence["matches_weight_layer"] and captured is not None:
                count = evidence["valid_tokens"]
                expected = np.asarray(captured["values"], dtype=np.float32).reshape(captured["shape"])[:, :count]
                row["fp32_source_vs_captured_bf16_shared_down"] = errors(refs["fp32_source_weights"][:, :count], expected)
        cases.append(row)
    return cases


def export_asset(module, x, path):
    # Lazy import keeps small CPU reference tests independent of the authoring SDK.
    import coreai_torch

    example = torch.from_numpy(x.copy())
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    converter.add_pytorch_module(
        module, input_names=("x",), output_names=("y",),
        export_fn=lambda m: torch.export.export(m, args=(example,)).run_decompositions(coreai_torch.get_decomp_table()))
    program = converter.to_coreai()
    program.optimize()
    program.save_asset(path)
    return [{"path": str(p.relative_to(path)), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
            for p in sorted(path.rglob("*")) if p.is_file()]


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/coreai-moe-export", help="Output root; a layer/expert subdirectory is created")
    parser.add_argument("--kind", choices=("shared", "routed"), default="shared")
    parser.add_argument("--expert", type=int, default=333, help="One routed expert ID; used only with --kind routed")
    parser.add_argument("--layer", type=int, default=0, help="Weight layer; default captures are from layer 0, other layers are labeled as replay")
    parser.add_argument("--capacities", type=int, choices=(1, 32), nargs="+", default=[1, 32])
    parser.add_argument("--fp32-projections", action="store_true",
                        help="Compute only three linear projections in FP32, retaining FP16 weights/I/O/SiLU; use a fresh --output")
    parser.add_argument("--fixture-dir", type=Path, default=ROOT / "fixtures/moe-real/converted", help="decode.json, prefill.json and capture provenance.json")
    parser.add_argument("--overwrite", action="store_true", help="Replace generated files for this layer/expert under --output")
    args = parser.parse_args(argv)
    if args.layer < 0:
        parser.error("--layer must be nonnegative")
    if not 0 <= args.expert < EXPERTS:
        parser.error(f"--expert must be between 0 and {EXPERTS - 1}")
    if len(set(args.capacities)) != len(args.capacities):
        parser.error("--capacities must not contain duplicates")
    return args


def main(argv=None):
    args = arguments(argv)
    torch.set_num_threads(2)
    kind = "shared" if args.kind == "shared" else f"expert{args.expert}"
    output = args.output.resolve() / f"layer{args.layer}-{kind}"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} already exists; choose another --output or use --overwrite")
    projection_dtype = "float32" if args.fp32_projections else "float16"
    previous_manifest = output / "manifest.json"
    if previous_manifest.exists():
        previous_dtype = json.loads(previous_manifest.read_text()).get("projection_dtype", "float16")
        if previous_dtype != projection_dtype:
            raise ValueError("Changing --fp32-projections requires a fresh --output; precision variants must not be mixed")
    source = Source(args.layer)
    weights = source.shared() if args.kind == "shared" else source.expert(args.expert)
    models = reference_models(weights)
    export_model = SwiGLU(weights, fp32_projections=True).eval() if args.fp32_projections else models["fp16"]
    fixtures = {s: fixture_inputs(args.fixture_dir.resolve(), s, weights["gate_proj"]["dense16"].shape[1], args.layer)
                for s in args.capacities}
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "authoring_in_progress", "schema": "coreai-moe-export-v1", "layer": args.layer, "kind": kind,
              "input_name": "x", "output_name": "y", "layout": "BSH", "dtype": "float16", "function": "main",
              "projection_dtype": projection_dtype, "weight_storage_dtype": "float16",
              "activation_dtype": "float16", "projection_output_dtype": "float16",
              "operation": "down_proj(silu(gate_proj(x)) * up_proj(x))",
              "exporter": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
              "source_script": {"path": str(SOURCE_SCRIPT), "sha256": sha256_file(SOURCE_SCRIPT)},
              "source_records": source.records,
              "source_manifest": {"path": str(SOURCE_ROOT / "source-manifest.json"), "sha256": sha256_file(SOURCE_ROOT / "source-manifest.json")},
              "source_download_verification": {"path": str(source.verification_path), "sha256": sha256_file(source.verification_path)},
              "source_verification_scope": "Current slice hashes and file sizes; prior whole-shard hashes are not recomputed.",
              "weight_shapes": {k: list(v["dense16"].shape) for k, v in weights.items()},
              "weight_fp32_to_fp16_errors": {k: errors(v["dense16"].astype(np.float32), v["dense32"]) for k, v in weights.items()},
              "versions": {k: importlib.metadata.version(k) for k in ("torch", "numpy", "coreai-core", "coreai-torch", "coremltools")},
              "os": platform.platform(), "system_runtime_minimum": "macOS 27",
              "limitations": ["One SwiGLU subgraph, excluding routing, shared gating, residuals and full-model generation.",
                              "FP16 dense constants; source compression and native INT4 arithmetic are not preserved.",
                              "CPU authoring/references only; runtime correctness and device placement are not established.",
                              "Captured input layer, valid tokens and zero padding are recorded per fixture; S32 is not a captured 32-token workload.",
                              "BF16 captured shared_down is diagnostic only and compared only when capture and weight layers match."],
              "suggested_runtime_gate": {"all_finite": True, "correct_shape_dtype": True,
                                         "relative_l2_fp32_fp16_weights_max": 0.005, "max_abs_fp32_fp16_weights_max": 0.02,
                                         "zero_max_abs_max": 1e-6, "scope": "Initial smoke thresholds, not full-model acceptance."},
              "exports": []}
    write_json(output / "manifest.json", report)
    for capacity in args.capacities:
        directory = output / f"s{capacity}"
        directory.mkdir(exist_ok=True)
        inputs, fixture, evidence = fixtures[capacity]
        cases = save_cases(directory, inputs, models, fixture, evidence, args.kind)
        precision_suffix = "fp16-gemm32" if args.fp32_projections else "fp16"
        path = directory / f"layer{args.layer}-{kind}-s{capacity}-{precision_suffix}.aimodel"
        started = time.perf_counter()
        files = export_asset(export_model, inputs["actual"], path)
        row = {"capacity": capacity, "model": str(path), "authoring_seconds": time.perf_counter() - started,
               "model_files": files, "model_bytes": sum(f["bytes"] for f in files), "cases": cases}
        report["exports"].append(row)
        write_json(output / "manifest.json", report)
        print(f"Exported {path} ({row['model_bytes']} bytes)", flush=True)
    report["status"] = "authoring_complete_runtime_not_executed"
    write_json(output / "manifest.json", report)
    print(f"Manifest: {output / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
