#!/usr/bin/env python3
"""Export the existing S1 PLE fixture and Python Core ML outputs for Swift QA."""
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import coremltools as ct
import numpy as np


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parents[1]
PROBE = EXPERIMENTS / "qwen38-ssd/results/coreml-ple-probe"
SOURCE = PROBE / "fixture_s1_actual_rows.npz"
MODEL = PROBE / "ple_scale64_minimal_s1.mlpackage"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024**2):
            digest.update(block)
    return digest.hexdigest()


def tensor(value):
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "values": value.astype(np.float64).ravel().tolist()}


def main():
    source = dict(np.load(SOURCE))
    inputs = {
        "embeddings": (source["embeddings"].transpose(0, 2, 1)[:, :, None] * 64).astype(np.float16),
        "residual": source["residual"].transpose(0, 2, 1)[:, :, None].astype(np.float16),
        "history": source["history"][:, :, None].astype(np.float16),
        "mask": source["mask"].transpose(0, 2, 1)[:, :, None].astype(np.float16),
    }
    model = ct.models.MLModel(str(MODEL), compute_units=ct.ComputeUnit.CPU_AND_NE)
    # coremltools 9.0 predict replaces float16 values in its input dictionary
    # with float32 arrays. Preserve the declared float16 fixture boundary.
    outputs = model.predict(dict(inputs))
    destination = HERE / "ple-s1-actual-rows.json"
    destination.write_text(json.dumps({"inputs": {k: tensor(v) for k, v in inputs.items()},
                                        "expectedOutputs": {k: tensor(v) for k, v in outputs.items()}},
                                       separators=(",", ":"), allow_nan=False) + "\n")
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_fixture": str(SOURCE), "source_fixture_sha256": sha256(SOURCE),
        "exported_fixture": str(destination), "exported_fixture_sha256": sha256(destination),
        "model_package": str(MODEL),
        "package_file_sha256": {str(path.relative_to(MODEL)): sha256(path) for path in MODEL.rglob("*") if path.is_file()},
        "python_coremltools": ct.__version__, "macOS": platform.mac_ver()[0],
        "python_input_bridge": "coremltools 9.0 replaces float16 arrays with float32 arrays in the dictionary passed to predict. A shallow dictionary copy preserves the exported float16 inputs; widening preserves every FP16 input value exactly.",
        "compute_units": "CPU_AND_NE", "python_prediction_calls": 1,
        "conversion": "Source BSH embedding/residual/mask are transposed to BC1S. History BCH becomes BC1H. Embedding is multiplied by 64 in float32 before casting to float16. Values serialize in logical row-major order.",
        "reference": "expectedOutputs are fresh Python Core ML predictions of the same three-output package, not the FP32 mathematical oracle.",
        "scope": "16 actual sampled n-gram rows form the embedding; residual and history are synthetic. No full embedding table or decoder is loaded.",
        "hardware_evidence": "Prediction configured for CPU_AND_NE; no hardware trace collected.",
        "previous_source_shard_verification": json.loads((PROBE / "report.json").read_text())["verified_source_shards"],
    }
    (HERE / "ple-s1-actual-rows.provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"fixture": str(destination), "bytes": destination.stat().st_size,
                      "inputs": {k: list(v.shape) for k, v in inputs.items()},
                      "outputs": {k: list(v.shape) for k, v in outputs.items()}}, indent=2))


if __name__ == "__main__":
    main()
