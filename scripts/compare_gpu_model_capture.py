#!/usr/bin/env python3
"""Compare actual Swift layer-0 boundaries with the original saved MLX capture.

No inference, model loading, or synthetic activations. Token IDs and absolute
positions must agree before any numerical comparison is admitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from safetensors import safe_open


MAPPING = {
    "layer.0.moe_input": "x",
    "layer.0.moe_out": "output",
    "layer.0.before_moe_write": "hc_stream_before",
    "layer.0.mlp_injection": "hc_inject",
    "layer.0.stream": "hc_stream_after",
}


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def metrics(actual: np.ndarray, expected: np.ndarray) -> dict:
    if actual.shape != expected.shape:
        return {"status": "shape_mismatch", "actual_shape": list(actual.shape), "expected_shape": list(expected.shape)}
    a, b = actual.astype(np.float64), expected.astype(np.float64)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return {"status": "nonfinite", "actual_nonfinite": int((~np.isfinite(a)).sum()), "reference_nonfinite": int((~np.isfinite(b)).sum())}
    difference = a - b
    actual_norm, reference_norm = float(np.linalg.norm(a.ravel())), float(np.linalg.norm(b.ravel()))
    error_norm = float(np.linalg.norm(difference.ravel()))
    denominator = actual_norm * reference_norm
    return {
        "status": "compared", "shape": list(a.shape), "elements": int(a.size),
        "exact": bool(np.array_equal(actual, expected)),
        "relative_l2": error_norm / reference_norm if reference_norm else (0.0 if error_norm == 0 else None),
        "reference_zero": reference_norm == 0,
        "max_absolute": float(np.max(np.abs(difference))),
        "rms_absolute": float(np.sqrt(np.mean(difference * difference))),
        "reference_rms": float(np.sqrt(np.mean(b * b))),
        "cosine": float(np.dot(a.ravel(), b.ravel()) / denominator) if denominator else (1.0 if error_norm == 0 else 0.0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--reference-dir", type=Path, default=Path(__file__).resolve().parents[1] / "fixtures/moe-real")
    parser.add_argument("--relative-l2-limit", type=float, default=0.01,
                        help="Local diagnostic threshold, default 1%%; not a model quality criterion")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not np.isfinite(args.relative_l2_limit) or args.relative_l2_limit < 0:
        parser.error("relative-l2-limit must be finite and nonnegative")

    reference_rows: dict[int, tuple[int, dict[str, np.ndarray], str]] = {}
    sources = []
    for phase in ("prefill", "decode"):
        path = args.reference_dir / f"{phase}.safetensors"
        if not path.exists():
            continue
        with safe_open(path, framework="pt", device="cpu") as file:
            meta = file.metadata() or {}
            if meta.get("layer_index") != "0":
                raise ValueError(f"Reference does not declare layer 0: {path}")
            token_tensor = file.get_tensor("token_ids")
            if token_tensor.ndim != 2 or token_tensor.shape[0] != 1:
                raise ValueError(f"Invalid reference token shape: {path}")
            ids = token_tensor.flatten().tolist()
            start = int(file.get_tensor("sequence_offset").item())
            tensors = {name: file.get_tensor(name).float().numpy() for name in MAPPING.values() if name in file.keys()}
            for name, tensor in tensors.items():
                if tensor.ndim < 2 or tensor.shape[:2] != (1, len(ids)):
                    raise ValueError(f"Reference token axis differs: {path}:{name}")
            for index, token in enumerate(ids):
                position = start + index
                if position in reference_rows:
                    raise ValueError(f"Overlapping reference positions: {position}")
                reference_rows[position] = (int(token), {name: tensor[:, index:index + 1] for name, tensor in tensors.items()}, phase)
            sources.append({"path": str(path.resolve()), "sha256": digest(path), "metadata": meta,
                            "sequence_offset": start, "token_ids": ids})
    if not sources:
        raise ValueError("No original prefill/decode safetensors reference available")

    steps = []
    compared_names = set()
    capture_history: dict[int, int] = {}
    with safe_open(args.capture, framework="pt", device="cpu") as file:
        capture_meta = file.metadata() or {}
        if capture_meta.get("schema") != "independent-qwen-gpu-boundaries-v1":
            raise ValueError("Unexpected Swift capture schema")
        prefixes = {name.rsplit(".", 1)[0] for name in file.keys() if re.match(r"^(prefill|decode)\.\d+\.token_ids$", name)}
        records = []
        for prefix in prefixes:
            token_tensor = file.get_tensor(prefix + ".token_ids")
            if token_tensor.ndim != 2 or token_tensor.shape[0] != 1:
                raise ValueError(f"Invalid capture token axis: {prefix}")
            ids = [int(value) for value in token_tensor.flatten().tolist()]
            start = int(file.get_tensor(prefix + ".sequence_offset").item())
            end = int(file.get_tensor(prefix + ".sequence_offset_after").item())
            if end != start + len(ids) or start < 0:
                raise ValueError(f"Incoherent capture state offset: {prefix}")
            for index, token in enumerate(ids):
                if start + index in capture_history:
                    raise ValueError("Overlapping Swift capture steps")
                capture_history[start + index] = token
            records.append((start, prefix, ids))
        if not records:
            raise ValueError("Capture has no named token steps")
        for start, prefix, ids in sorted(records):
            step = {"prefix": prefix, "sequence_offset": start, "token_ids": ids, "comparisons": {}}
            # Identical current IDs are insufficient for a recurrent model:
            # verify the complete captured prefix leading into the step too.
            history_matches = all(position in reference_rows and capture_history.get(position) == reference_rows[position][0]
                                  for position in range(start + len(ids)))
            step["complete_history_and_tokens_match"] = history_matches
            if not history_matches:
                step["status"] = "unmatched_tokens_or_missing_reference_history"
                steps.append(step)
                continue
            step["reference_spans"] = sorted({reference_rows[position][2] for position in range(start, start + len(ids))})
            for own_name, reference_name in MAPPING.items():
                key = prefix + "." + own_name
                if key not in file.keys():
                    step["comparisons"][own_name] = {"status": "capture_field_missing"}
                    continue
                if any(reference_name not in reference_rows[position][1] for position in range(start, start + len(ids))):
                    step["comparisons"][own_name] = {"status": "reference_field_missing"}
                    continue
                expected = np.concatenate([reference_rows[position][1][reference_name] for position in range(start, start + len(ids))], axis=1)
                actual = file.get_tensor(key).float().numpy()
                result = metrics(actual, expected)
                result["reference_field"] = reference_name
                relative = result.get("relative_l2")
                result["within_limit"] = result["status"] == "compared" and relative is not None and relative <= args.relative_l2_limit
                step["comparisons"][own_name] = result
                compared_names.add(key)
            step["status"] = "compared"
            steps.append(step)
        uncompared = [name for name in file.keys() if name not in compared_names and not name.endswith((".token_ids", ".sequence_offset", ".sequence_offset_after"))]

    comparisons = [value for step in steps for value in step["comparisons"].values()]
    passed = bool(comparisons) and all(step["complete_history_and_tokens_match"] for step in steps) and all(value.get("within_limit", False) for value in comparisons)
    report = {
        "schema_version": 1, "passed_captured_layer0_boundaries": passed,
        "relative_l2_limit": args.relative_l2_limit,
        "capture": {"path": str(args.capture.resolve()), "sha256": digest(args.capture), "metadata": capture_meta},
        "reference_sources": sources, "mapping": MAPPING, "steps": steps,
        "uncompared_capture_tensors": sorted(uncompared),
        "scope": "Only saved layer-0 MoE input/output and HC boundaries with matching full token history. No inference was run; later layers and uncaptured internal operations are not validated.",
        "reference_decode_note": "The existing offset-26 token 271 is the author's held-back final prompt newline, not the first generated content token.",
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded)
    print(encoded, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
