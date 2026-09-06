#!/usr/bin/env python3
"""Export bounded real Qwen layer MoE experts for the independent Swift runner.

Only requested expert slices are read. No model server or full decoder is loaded.
The Q4 LUT variant preserves round_fp16(q * scale_bf16 + bias_bf16) exactly;
it does not claim native ANE INT4 arithmetic or compressed runtime residency.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import struct
import time

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "qwen38-ssd/results"
DEFAULT_OUTPUT = ROOT / "ane-runner/results/moe-export"
HIDDEN, INTERMEDIATE, EXPERTS, TOP_K, GROUP = 2560, 640, 512, 10, 64
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class Source:
    """Read exact tensor/expert byte ranges, retaining digest evidence per slice."""
    def __init__(self, layer):
        self.manifest = json.loads((SOURCE_ROOT / "source-manifest.json").read_text())
        self.directory = Path(self.manifest["model_directory"])
        self.index = json.loads((SOURCE_ROOT / "remote-model.safetensors.index.json").read_text())["weight_map"]
        self.metadata = {f["Path"]: f for f in self.manifest["files"]}
        self.verification_path = SOURCE_ROOT / "download-verification.json"
        verification = json.loads(self.verification_path.read_text())
        if verification["phase"] != "complete" or Path(verification["model_directory"]) != self.directory:
            raise ValueError("Completed source verification for this model directory is required")
        self.verified = {f["path"]: f for f in verification["files"]}
        self.headers = {}
        self.records = []
        self.prefix = f"language_model.model.layers.{layer}.mlp."

    def read(self, suffix, expert=None):
        key = self.prefix + suffix
        filename = self.index[key]
        path = self.directory / filename
        expected = self.metadata[filename]
        if (not path.is_file() or path.stat().st_size != expected["Size"]
                or self.verified[filename]["sha256"] != expected["Sha256"]):
            raise ValueError(f"Completed, previously hash-verified source shard required: {path}")
        with path.open("rb") as stream:
            if filename not in self.headers:
                header_size = struct.unpack("<Q", stream.read(8))[0]
                if not 0 < header_size < 1024 * 1024:
                    raise ValueError("Invalid safetensors header length")
                self.headers[filename] = (8 + header_size, json.loads(stream.read(header_size)))
            base, header = self.headers[filename]
            metadata = header[key]
            shape = metadata["shape"]
            start, end = metadata["data_offsets"]
            if metadata["dtype"] not in ("BF16", "U32"):
                raise ValueError(f"Unsupported source dtype {metadata['dtype']}")
            if expert is not None:
                if shape[0] != EXPERTS or not 0 <= expert < EXPERTS:
                    raise ValueError("Invalid routed expert slice")
                slice_bytes = (end - start) // EXPERTS
                start += expert * slice_bytes
                end = start + slice_bytes
                shape = shape[1:]
            stream.seek(base + start)
            raw = stream.read(end - start)
            if len(raw) != end - start:
                raise ValueError("Truncated tensor data")
        record = {"key": key, "file": filename, "expert": expert,
                  "source_dtype": metadata["dtype"], "shape": shape,
                  "file_byte_offset": base + start, "byte_length": len(raw),
                  "slice_sha256": hashlib.sha256(raw).hexdigest(),
                  "previous_whole_shard_sha256": expected["Sha256"]}
        self.records.append(record)
        if metadata["dtype"] == "BF16":
            return (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32).reshape(shape)
        return np.frombuffer(raw, dtype="<u4").reshape(shape)

    def expert(self, index):
        result = {}
        for name in PROJECTIONS:
            prefix = f"switch_mlp.{name}."
            packed = self.read(prefix + "weight", index)
            scale = self.read(prefix + "scales", index)
            bias = self.read(prefix + "biases", index)
            codes = ((packed[..., None] >> (4 * np.arange(8, dtype=np.uint32))) & 15).astype(np.uint8)
            codes = codes.reshape(packed.shape[0], -1)
            if codes.shape != (scale.shape[0], scale.shape[1] * GROUP) or scale.shape != bias.shape:
                raise ValueError("Q4 group64 source shape mismatch")
            # Each BF16 scale/bias is exactly represented as FP32. No requantization.
            palette32 = scale[..., None] * np.arange(16, dtype=np.float32) + bias[..., None]
            dense32 = np.take_along_axis(palette32, codes.reshape(*scale.shape, GROUP), axis=-1).reshape(codes.shape)
            palette16 = palette32.astype(np.float16)
            dense16 = np.take_along_axis(palette16, codes.reshape(*scale.shape, GROUP), axis=-1).reshape(codes.shape)
            if not np.array_equal(dense16, dense32.astype(np.float16)):
                raise AssertionError("LUT does not exactly preserve FP16-rounded affine Q4 values")
            result[name] = {"codes": codes, "scale32": scale, "bias32": bias,
                            "palette16": palette16, "dense32": dense32, "dense16": dense16}
        return result

    def shared(self):
        return {name: {"dense32": (weight := self.read(f"shared_expert.{name}.weight")),
                       "dense16": weight.astype(np.float16)} for name in PROJECTIONS}


def make_program(weights, capacity, mode, diagnostic=False, layout="conv"):
    def constant(name):
        value = weights[name]
        if mode == "q4_lut":
            codes = value["codes"] if layout == "linear" else value["codes"][:, :, None, None]
            codes = codes.astype(types.nptype_from_builtin(types.uint4))
            palette = value["palette16"][..., None] if layout == "linear" else value["palette16"][:, :, None, None, :, None]
            return mb.constexpr_lut_to_dense(indices=codes, lut=palette, name=name + "_lut")
        weight = value["dense16"] if layout == "linear" else value["dense16"][:, :, None, None]
        return mb.const(val=weight, name=name + "_weight")

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, HIDDEN, 1, capacity), dtype=types.fp16)],
                opset_version=ct.target.macOS15)
    def program(x):
        value = x
        projection = mb.conv
        if layout == "linear":
            value = mb.reshape(x=mb.transpose(x=x, perm=[0, 3, 2, 1]), shape=[1, capacity, HIDDEN])
            projection = mb.linear
        gate = projection(x=value, weight=constant("gate_proj"), name="gate_projection")
        up = projection(x=value, weight=constant("up_proj"), name="up_projection")
        # Equivalent SiLU form avoids the fused SiLU zero offset observed on this host.
        neg_gate = mb.mul(x=gate, y=np.float16(-1), name="negative_gate")
        denominator = mb.add(x=mb.exp(x=neg_gate), y=np.float16(1))
        activated = mb.real_div(x=gate, y=denominator, name="silu_explicit")
        middle = mb.mul(x=activated, y=up, name="gated_up")
        y = projection(x=middle, weight=constant("down_proj"), name="down_projection" if layout == "linear" else "y")
        if layout == "linear":
            y = mb.transpose(x=mb.reshape(x=y, shape=[1, capacity, 1, HIDDEN]), perm=[0, 3, 2, 1], name="y")
        return (y, gate, up, activated, middle) if diagnostic else y
    return program


def export(weights, capacity, mode, path, layout="conv"):
    program = make_program(weights, capacity, mode, layout=layout)
    model = ct.convert(program, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                       compute_units=ct.ComputeUnit.CPU_AND_NE,
                       compute_precision=ct.precision.FLOAT16)
    model.save(str(path))
    return model


def compute_plan(model):
    plan = ct.models.compute_plan.MLComputePlan.load_from_path(
        model.get_compiled_model_path(), compute_units=ct.ComputeUnit.CPU_AND_NE)
    result = []
    def visit(block):
        for op in block.operations:
            if op.operator_name != "const":
                usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
                result.append({"op": op.operator_name,
                               "preferred": type(usage.preferred_compute_device).__name__ if usage else None,
                               "supported": [type(d).__name__ for d in usage.supported_compute_devices] if usage else []})
            for child in op.blocks:
                visit(child)
    for function in plan.model_structure.program.functions.values():
        visit(function.block)
    return result


def reference(weights, x, mode):
    dtype = torch.float16 if mode == "fp16" else torch.float32
    weight_key = "dense32" if mode == "fp32_affine" else "dense16"
    w = {name: torch.from_numpy(value[weight_key]).to(dtype)[:, :, None, None] for name, value in weights.items()}
    with torch.inference_mode():
        value = torch.from_numpy(x.copy()).to(dtype)
        gate = F.conv2d(value, w["gate_proj"])
        up = F.conv2d(value, w["up_proj"])
        y = F.conv2d(F.silu(gate) * up, w["down_proj"])
    return y.float().numpy()


def errors(actual, expected):
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    norm = np.linalg.norm(expected.astype(np.float64))
    return {"max_abs": float(np.max(np.abs(delta))),
            "rms": float(np.sqrt(np.mean(delta * delta))),
            "relative_l2": float(np.linalg.norm(delta) / norm) if norm else None,
            "exact": bool(np.array_equal(actual, expected))}


def tensor(value):
    return {"shape": list(value.shape), "dtype": str(value.dtype), "values": value.astype(np.float64).ravel().tolist()}


def package_bytes(path):
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def probe(model, weights, capacity, directory, expert, mode, fixture_path=None):
    rng = np.random.default_rng(20260905)
    cases = {"normal": rng.standard_normal((1, HIDDEN, 1, capacity)).astype(np.float16),
             "zero": np.zeros((1, HIDDEN, 1, capacity), np.float16)}
    if fixture_path is not None:
        source = np.load(fixture_path)
        key = next((k for k in ("expert_input", "x", "router_input") if k in source), None)
        if key is None:
            raise ValueError("NPZ requires expert_input, x, or router_input in BSH layout")
        value = source[key]
        if value.ndim != 3 or value.shape[0] != 1 or value.shape[2] != HIDDEN or value.shape[1] < 1:
            raise ValueError("Fixture requires nonempty BSH activations")
        valid_tokens = min(value.shape[1], capacity)
        padded = np.zeros((1, capacity, HIDDEN), np.float16)
        padded[:, :valid_tokens] = value[:, :valid_tokens].astype(np.float16)
        cases["actual"] = padded.transpose(0, 2, 1)[:, :, None]
    rows = []
    for name, x in cases.items():
        refs = {key: reference(weights, x, key) for key in ("fp32_affine", "fp32_fp16_weights", "fp16")}
        y = model.predict({"x": x})["y"]
        row = {"case": name, "input_rms": float(np.sqrt(np.mean(x.astype(np.float32) ** 2))),
               "finite": bool(np.isfinite(y).all()), "errors": {key: errors(y, ref) for key, ref in refs.items()}}
        if name == "actual":
            row["valid_tokens"] = valid_tokens
            row["fixture"] = str(fixture_path)
            row["fixture_sha256"] = sha256_file(fixture_path)
            row["padded_outputs_exact_zero"] = bool(np.count_nonzero(y[..., valid_tokens:]) == 0)
        if name != "zero":
            for _ in range(3):
                model.predict({"x": x})
            times = []
            for _ in range(5):
                start = time.perf_counter_ns()
                model.predict({"x": x})
                times.append((time.perf_counter_ns() - start) / 1e6)
            row["prediction_ms"] = times
            row["prediction_median_ms"] = float(np.median(times))
            fixture = {"inputs": {"x": tensor(x)}, "expectedOutputs": {"y": tensor(y)}}
            write_json(directory / f"expert_{expert:04d}_{name}_fixture.json", fixture)
        rows.append(row)
    return {"expert_id": expert, "mode": mode, "capacity": capacity,
            "compute_plan": compute_plan(model), "cases": rows,
            "hardware_evidence": "Public CPU_AND_NE prediction completed. Plan preferences are not a hardware trace; CPU fallback remains possible."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--experts", default="0", help="Comma-separated actual expert IDs; only these are exported")
    parser.add_argument("--all-experts", action="store_true", help="Explicit opt-in to export all 512 experts sequentially")
    parser.add_argument("--capacities", default="1,32")
    parser.add_argument("--modes", default="fp16,q4_lut")
    parser.add_argument("--layout", choices=("conv", "linear"), default="conv")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fixture", type=Path, help="Optional real captured MoE input NPZ in BSH layout")
    parser.add_argument("--probe", action="store_true", help="Probe the first selected expert for each capacity/mode")
    args = parser.parse_args()
    expert_ids = list(range(EXPERTS)) if args.all_experts else sorted(set(map(int, args.experts.split(","))))
    capacities = sorted(set(map(int, args.capacities.split(","))))
    modes = args.modes.split(",")
    if not expert_ids or any(not 0 <= e < EXPERTS for e in expert_ids):
        parser.error("Expert IDs must be in [0,511]")
    if any(c < 1 for c in capacities) or any(m not in ("fp16", "q4_lut") for m in modes):
        parser.error("Capacities must be positive; supported modes: fp16,q4_lut")
    torch.set_num_threads(4)
    source = Source(args.layer)
    args.output.mkdir(parents=True, exist_ok=True)
    router = source.read("gate.weight")
    shared_gate = source.read("shared_expert_gate.weight")
    shared = source.shared()
    report_path = args.output / "report.json"
    previous = json.loads(report_path.read_text()) if report_path.exists() else {}
    reports = previous.get("exports", [])
    previous_slices = previous.get("source_slices", [])
    for capacity in capacities:
        for mode in modes:
            layout_suffix = "_linear" if args.layout == "linear" else ""
            directory = args.output / f"layer_{args.layer}_{mode}{layout_suffix}_s{capacity}"
            directory.mkdir(parents=True, exist_ok=True)
            router.astype("<f4").tofile(directory / "router.f32.bin")
            shared_gate.astype("<f4").tofile(directory / "shared_gate.f32.bin")
            manifest_path = directory / "manifest.json"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
                if manifest["weight_mode"] != mode or manifest["token_capacity"] != capacity or manifest["layer_index"] != args.layer:
                    raise ValueError("Existing bank does not match requested export")
            else:
                manifest = {"schema_version": 1, "layer_index": args.layer, "hidden_size": HIDDEN,
                            "intermediate_size": INTERMEDIATE, "expert_count": EXPERTS, "top_k": TOP_K,
                            "token_capacity": capacity, "input_name": "x", "output_name": "y", "dtype": "float16",
                            "routing": {"weights_file": "router.f32.bin", "shared_gate_file": "shared_gate.f32.bin", "dtype": "float32_le"},
                            "experts": {}, "shared_expert": "shared.mlpackage", "weight_mode": mode, "graph_layout": args.layout,
                            "source": {"model_directory": str(source.directory), "modelscope_commit": source.manifest["modelscope_latest_commit"],
                                       "prior_verification": str(source.verification_path), "prior_verification_sha256": sha256_file(source.verification_path)},
                            "weight_semantics": "Q4 codes and BF16 affine scales/biases are decoded in FP32, then rounded to FP16. q4_lut stores those exact FP16 values using original 4-bit codes and 16-entry LUTs per output-row/input-group64. Shared expert remains FP16.",
                            "scope": "Only listed experts are available. Missing dynamically selected experts must be an error. No full-decoder support or ANE-only guarantee."}
            shared_path = directory / manifest["shared_expert"]
            if not shared_path.exists():
                shared_model = export(shared, capacity, "fp16", shared_path, args.layout)
                del shared_model
            for expert in expert_ids:
                path = directory / f"expert_{expert:04d}.mlpackage"
                weights = source.expert(expert)
                if not path.exists():
                    start = time.perf_counter()
                    model = export(weights, capacity, mode, path, args.layout)
                    seconds = time.perf_counter() - start
                else:
                    model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
                    seconds = None
                manifest["experts"][str(expert)] = path.name
                write_json(manifest_path, manifest)
                row = {"expert": expert, "mode": mode, "layout": args.layout, "capacity": capacity, "package": str(path),
                       "package_bytes": package_bytes(path), "export_seconds": seconds,
                       "dense_fp16_weight_bytes": sum(w["dense16"].nbytes for w in weights.values()),
                       "decoded_weight_equality": "q4_lut reconstruction equals the FP16-rounded affine Q4 weights exactly, checked before export"}
                if args.probe and expert == expert_ids[0]:
                    row["probe"] = probe(model, weights, capacity, directory, expert, mode, args.fixture)
                reports.append(row)
                write_json(args.output / "report.json", {"macOS": platform.mac_ver()[0], "coremltools": ct.__version__,
                           "torch": torch.__version__, "exports": reports, "source_slices": previous_slices + source.records,
                           "limitations": ["File compression does not prove native low-bit ANE arithmetic or compressed runtime weight residency.",
                                           "Source whole-shard hashes were verified by the completed download workflow; this exporter rechecks file sizes and hashes only bytes it actually reads.",
                                           "FP32 affine-weight and FP16-weight/compute references are separate; neither is a claim of bitwise equivalence to MLX quantized_matmul."]})
                print(json.dumps({k: v for k, v in row.items() if k != "probe"}), flush=True)
                del model, weights
                gc.collect()


if __name__ == "__main__":
    main()
