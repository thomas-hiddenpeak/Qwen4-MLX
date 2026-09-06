#!/usr/bin/env python3
"""Slice real native MoE captures into CoreMLBlockFixture JSON without recomputing outputs.

Uses the workspace Python environment (numpy, torch, safetensors). Raw captures
are read only. Default prefill positions are 14 and last: in the current captured
request these cover a Chinese content token and the final chat-template token.
The separate decode fixture retains the first actual decode token.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from safetensors import safe_open
import torch


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "qwen4-layer0-moe-real-v1"
HIDDEN, EXPERTS, TOP_K = 2560, 512, 10
OUTPUT_WIDTHS = {
    "router_logits": EXPERTS,
    "selected_experts": TOP_K,
    "routing_weights": TOP_K,
    "routed_sum": HIDDEN,
    "shared_down": HIDDEN,
    "shared_gated": HIDDEN,
    "output": HIDDEN,
    "shared_gate_logits": 1,
    "shared_gate": 1,
}
REQUIRED = {"x", "selected_experts", "routing_weights", "output", "token_ids", "sequence_offset"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def integer_values(tensor: torch.Tensor, name: str) -> np.ndarray:
    if tensor.dtype not in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64, torch.uint32, torch.uint64):
        raise ValueError(f"{name} must have an integer source dtype, got {tensor.dtype}")
    values = tensor.to(torch.int64).numpy()
    if np.any(values < 0):
        raise ValueError(f"{name} must contain nonnegative integers")
    return values


def floats(tensor: torch.Tensor, name: str) -> np.ndarray:
    if not tensor.is_floating_point():
        raise ValueError(f"{name} must have a floating source dtype, got {tensor.dtype}")
    # Explicit BF16 -> FP32 is lossless; never round capture values through FP16.
    values = tensor.to(torch.float32).numpy()
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains nonfinite values")
    return values


def coreml_tensor(values: np.ndarray, dtype: str = "float32") -> dict:
    return {"shape": list(values.shape), "dtype": dtype, "values": values.reshape(-1).tolist()}


def load_capture(path: Path, phase: str, requested_positions: list[str] | None) -> dict:
    before = path.stat()
    with safe_open(path, framework="pt", device="cpu") as source:
        metadata = source.metadata() or {}
        if metadata.get("schema") != SCHEMA or metadata.get("phase") != phase:
            raise ValueError(f"{path}: expected {SCHEMA} {phase} metadata, got {metadata}")
        if metadata.get("layer_index") != "0" or not metadata.get("source_commit"):
            raise ValueError(f"{path}: missing layer-0 source provenance")
        keys = set(source.keys())
        if missing := REQUIRED - keys:
            raise ValueError(f"{path}: missing required tensors {sorted(missing)}")
        raw_x = source.get_tensor("x")
        if raw_x.ndim != 3 or raw_x.shape[0] != 1 or raw_x.shape[2] != HIDDEN:
            raise ValueError(f"{path}: x must have shape [1,S,{HIDDEN}], got {list(raw_x.shape)}")
        count = raw_x.shape[1]
        offsets = integer_values(source.get_tensor("sequence_offset"), "sequence_offset")
        if offsets.size != 1:
            raise ValueError("sequence_offset must be a scalar")
        offset = int(offsets.reshape(-1)[0])
        if phase == "prefill" and not (1 < count <= 512 and offset == 0):
            raise ValueError("prefill must contain 2..512 real positions at offset 0")
        if phase == "decode" and not (count == 1 and offset > 0):
            raise ValueError("decode must contain one real token at a positive offset")
        positions = [count - 1 if p == "last" else int(p) for p in requested_positions] if requested_positions is not None else [0]
        if not positions or len(set(positions)) != len(positions) or any(p < 0 or p >= count for p in positions):
            raise ValueError(f"Invalid {phase} positions {positions} for {count} captured tokens")
        tokens = integer_values(source.get_tensor("token_ids"), "token_ids")
        if tokens.shape != (1, count):
            raise ValueError(f"token_ids shape {tokens.shape} disagrees with x")
        x = floats(raw_x[:, positions, :], "x")
        arrays = {"x": x}
        descriptors = {"x": {"shape": list(raw_x.shape), "dtype": str(raw_x.dtype)}}
        expected = {}
        for name, width in OUTPUT_WIDTHS.items():
            if name not in keys:
                continue
            raw = source.get_tensor(name)
            if tuple(raw.shape) != (1, count, width):
                raise ValueError(f"{name}: expected [1,{count},{width}], got {list(raw.shape)}")
            descriptors[name] = {"shape": list(raw.shape), "dtype": str(raw.dtype)}
            if name == "selected_experts":
                # MLX stores these as UInt32; Torch cannot advanced-index UInt32.
                values = integer_values(raw, name)[:, positions, :]
                if np.any(values >= EXPERTS) or any(len(set(row)) != TOP_K for row in values.reshape(-1, TOP_K)):
                    raise ValueError("selected_experts contains out-of-range or repeated IDs")
                values = values.astype(np.int32)
            else:
                values = floats(raw[:, positions, :], name)
            output_name = "y" if name == "output" else name
            arrays[output_name] = values
            expected[output_name] = coreml_tensor(values, "int32" if name == "selected_experts" else "float32")
        weights = arrays["routing_weights"]
        if np.any(weights < 0) or np.any(np.sum(weights, axis=-1) <= 0):
            raise ValueError("routing_weights must be nonnegative with positive row sums")
        # Keep this optional native per-expert oracle in NPZ, outside block output names.
        if "selected_expert_outputs" in keys:
            raw = source.get_tensor("selected_expert_outputs")
            if tuple(raw.shape) != (1, count, TOP_K, HIDDEN):
                raise ValueError("selected_expert_outputs shape disagrees with x")
            arrays["selected_expert_outputs"] = floats(raw[:, positions, :, :], "selected_expert_outputs")
            descriptors["selected_expert_outputs"] = {"shape": list(raw.shape), "dtype": str(raw.dtype)}
    source_sha = sha256(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError(f"Capture changed while reading: {path}; wait for the native save to finish")
    arrays["token_ids"] = tokens[:, positions].copy()
    arrays["positions"] = np.asarray(positions, dtype=np.int64)
    arrays["absolute_positions"] = np.asarray([offset + p for p in positions], dtype=np.int64)
    return {
        "fixture": {"inputs": {"x": coreml_tensor(x)}, "expectedOutputs": expected},
        "arrays": arrays,
        "provenance": {
            "phase": phase, "raw_capture": str(path), "raw_capture_bytes": after.st_size,
            "raw_capture_sha256": source_sha, "source_metadata": metadata,
            "captured_token_count": count, "sequence_offset": offset,
            "captured_token_ids": tokens.reshape(-1).tolist(),
            "positions": positions, "absolute_positions": arrays["absolute_positions"].tolist(),
            "selected_token_ids": tokens[:, positions].reshape(-1).tolist(),
            "selected_experts": arrays["selected_experts"].reshape(-1, TOP_K).tolist(),
            "routing_weights": weights.reshape(-1, TOP_K).tolist(),
            "routing_weight_sums": np.sum(weights.astype(np.float64), axis=-1).reshape(-1).tolist(),
            "retained_tensor_sources": descriptors, "raw_tensor_names": sorted(keys),
            "omitted_optional_outputs": sorted(set(OUTPUT_WIDTHS) - keys),
        },
    }


def atomic_bytes(path: Path, data: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as output:
        temporary = Path(output.name)
        try:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
            output.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def write_json(path: Path, data: dict) -> None:
    atomic_bytes(path, (json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, default=ROOT / "fixtures/moe-real")
    parser.add_argument("--output-dir", type=Path, help="Default: CAPTURE_DIR/converted")
    parser.add_argument("--prefill-positions", default="14,last", help="Distinct zero-based positions, comma separated; 'last' allowed (default: 14,last for current captured prompt)")
    parser.add_argument("--build-report", type=Path, default=ROOT / "results/moe-capture/build-report.json")
    args = parser.parse_args()
    try:
        capture_dir = args.capture_dir.resolve()
        output_dir = (args.output_dir or capture_dir / "converted").resolve()
        positions = [v.strip() for v in args.prefill_positions.split(",")]
        # Validate both inputs before creating any converted outputs.
        bundles = {
            "prefill": load_capture(capture_dir / "prefill.safetensors", "prefill", positions),
            "decode": load_capture(capture_dir / "decode.safetensors", "decode", None),
        }
        if bundles["decode"]["provenance"]["sequence_offset"] != bundles["prefill"]["provenance"]["captured_token_count"]:
            raise ValueError("decode offset is not the first position after this prefill")
        commits = {b["provenance"]["source_metadata"]["source_commit"] for b in bundles.values()}
        if len(commits) != 1:
            raise ValueError("prefill and decode have different source commits")
        build = None
        if args.build_report.exists():
            build = {"path": str(args.build_report.resolve()), "sha256": sha256(args.build_report),
                     "content": json.loads(args.build_report.read_text())}
            if build["content"].get("source_commit") not in commits:
                raise ValueError("capture build report source commit disagrees with captures")
        output_dir.mkdir(parents=True, exist_ok=True)
        experts = sorted({int(e) for bundle in bundles.values() for e in bundle["arrays"]["selected_experts"].reshape(-1)})
        provenance = {
            "schema": "ane-runner-moe-fixture-v1", "capture_schema": SCHEMA,
            "source_commit": next(iter(commits)), "capture_build_report": build,
            "converter": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
            "conversion": "Select actual token rows in original order; expand native floating tensors to float32; output renamed y. No routing, weights, or expected values recomputed.",
            "scope": "Two prefill positions by default and first decode token from layer 0, not an end-to-end decoder or performance validation. All original raw capture files retained.",
            "actual_expert_ids": experts, "phases": {},
        }
        for phase, bundle in bundles.items():
            fixture_path, npz_path = output_dir / f"{phase}.json", output_dir / f"{phase}.npz"
            write_json(fixture_path, bundle["fixture"])
            with tempfile.TemporaryFile() as temporary:
                np.savez(temporary, **bundle["arrays"])
                temporary.seek(0)
                atomic_bytes(npz_path, temporary.read())
            provenance["phases"][phase] = {
                **bundle["provenance"], "fixture": str(fixture_path), "fixture_sha256": sha256(fixture_path),
                "numpy_companion": str(npz_path), "numpy_companion_sha256": sha256(npz_path),
            }
        expert_path = output_dir / "actual-expert-ids.txt"
        atomic_bytes(expert_path, (",".join(map(str, experts)) + "\n").encode())
        provenance_path = output_dir / "provenance.json"
        write_json(provenance_path, provenance)
        print(json.dumps({"status": "complete", "provenance": str(provenance_path),
                          "fixtures": {p: str(output_dir / f"{p}.json") for p in bundles},
                          "actual_expert_ids": experts, "expert_count": len(experts),
                          "expert_ids_file": str(expert_path)}, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
