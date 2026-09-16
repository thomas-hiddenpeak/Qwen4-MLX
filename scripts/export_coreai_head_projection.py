#!/usr/bin/env python3
"""CPU-only isolation of real head GEMV from HC mixer and input suffix selection.

Author separate old FP32-linear and new FP32-output Metal projection assets,
both consuming exactly the same saved FP16 mixed input. Do not execute CoreAI.

Example (paths relative to the runner repo):
  python scripts/export_coreai_head_projection.py \
    --head-fixtures results/coreai-prefill-1k/head-metal-fp32-real \
    --output results/coreai-prefill-1k/head-projection-fp32-real

Pass each generated *-spec.json to the standalone Swift device probe in a
separate process. Compare its device-*/logits.bin with expected-logits.bin and
lane-oracle-logits.bin as little-endian FP32. Device execution is deliberately
separate from this CPU authoring script.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import torch
import torch.nn.functional as F

from coreai_head_metal import MetalHeadProjection, get_head_kernel
from export_coreai_pd import export_shared
from export_coreai_q4_moe import tensor_json
from export_moe import Source, sha256_file, write_json


class FloatHeadProjection(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.register_buffer("weight", weight)

    def forward(self, x):
        return F.linear(x.float(), self.weight.float())


def metrics(actual, expected):
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    return {"exact": bool(np.array_equal(actual, expected)),
            "maxAbsoluteError": float(np.abs(delta).max()),
            "relativeL2Error": float(np.linalg.norm(delta) / max(np.linalg.norm(expected), 1e-30))}


def audit_fixtures(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    rows = []
    final_bytes = None
    for spec_path in sorted(directory.glob("*-spec.json")):
        spec = json.loads(spec_path.read_text())
        entry = spec["inputs"]["stream"]
        if (entry["dtype"] != "float16" or len(entry["shape"]) != 3 or
                entry["shape"][0] != 1 or min(entry["shape"]) <= 0 or entry["offset"] < 0):
            raise ValueError("Expected FP16 single-batch stream fixture")
        row_bytes = entry["shape"][-1] * 2
        if entry["bytes"] != int(np.prod(entry["shape"])) * 2:
            raise ValueError("Incorrect byte count in stream fixture")
        file = directory / entry["file"]
        with file.open("rb") as handle:
            handle.seek(entry["offset"] + entry["bytes"] - row_bytes)
            last = handle.read(row_bytes)
        if len(last) != row_bytes:
            raise ValueError("Truncated last row in stream fixture")
        if final_bytes is None:
            final_bytes = last
        if last != final_bytes:
            raise ValueError(f"Different final input row in {spec_path}")
        expected_count = manifest["phaseSizes"][spec["function"]]
        if entry["shape"][1] != expected_count:
            raise ValueError("Phase size does not match fixture")
        rows.append({"spec": spec_path.name, "function": spec["function"],
                     "shape": entry["shape"], "offset": entry["offset"],
                     "lastRowSHA256": hashlib.sha256(last).hexdigest()})
    if final_bytes is None:
        raise ValueError("No fixture specs found")
    fixture = json.loads((directory / "actual-s1.json").read_text())["inputs"]["stream"]
    values = np.asarray(fixture["values"], dtype="<f2")
    if fixture["shape"] != [1, 1, len(final_bytes)//2] or values.tobytes() != final_bytes:
        raise ValueError("JSON S1 input differs from device spec final row")
    return {"allLastRowsExact": True, "jsonS1Matches": True, "specs": rows}


def lane_oracle(x, weight, block=512):
    # FP16*FP16 products are exactly representable in FP32; each lane's FP32
    # accumulation is then followed by a balanced 32-lane sum. Actual Metal
    # simd_sum ordering may differ, so this is an error-scale diagnostic.
    x = x.numpy().astype(np.float32)[0, 0]
    output = np.empty(weight.shape[0], dtype=np.float32)
    for row in range(0, weight.shape[0], block):
        w = weight[row:row+block].numpy().astype(np.float32)
        accum = np.zeros((len(w), 32), dtype=np.float32)
        for start in range(0, len(x), 32):
            count = min(32, len(x) - start)
            accum[:, :count] += w[:, start:start+count] * x[None, start:start+count]
        while accum.shape[-1] > 1:
            accum = accum[:, ::2] + accum[:, 1::2]
        output[row:row+len(w)] = accum[:, 0]
    return output.reshape(1, 1, -1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.output.exists():
        raise ValueError("Use a fresh output directory")
    if shutil.disk_usage(args.output.parent).free < 4_000_000_000:
        raise ValueError("At least 4 GB free required")
    args.output.mkdir(parents=True)
    original = json.loads((args.head_fixtures / "manifest.json").read_text())
    audit = audit_fixtures(args.head_fixtures)
    write_json(args.output / "fixture-audit.json", audit)
    source = Source(0)
    source.prefix = ""
    weight = torch.from_numpy(source.read("language_model.lm_head.weight")).half()
    prior_slice = next(row for row in original["sourceSlices"] if row["key"] == "language_model.lm_head.weight")
    if source.records[-1] != prior_slice:
        raise ValueError("Projection source weight differs from full-head candidate source")
    gc.collect()
    x = torch.from_numpy(np.fromfile(args.head_fixtures / "mixed.bin", dtype="<f2").reshape(1, 1, weight.shape[1]))
    if not torch.isfinite(x).all() or not torch.isfinite(weight).all():
        raise ValueError("Finite FP16 input and weights required")
    x.numpy().tofile(args.output / "mixed.bin")
    with torch.inference_mode():
        cpu = torch.cat([F.linear(x.float(), weight[row:row+1024].float())
                         for row in range(0, len(weight), 1024)], dim=-1).numpy()
    prior = np.fromfile(args.head_fixtures / "expected-logits.bin", dtype="<f4").reshape(cpu.shape)
    if not np.array_equal(cpu, prior):
        raise ValueError("Same mixed input/source weights do not reproduce previous CPU logits")
    cpu.astype("<f4").tofile(args.output / "expected-logits.bin")
    print("Computing FP32 lane oracle for the real weight/input", flush=True)
    lane = lane_oracle(x, weight)
    lane.tofile(args.output / "lane-oracle-logits.bin")
    report = {"version": 1, "status": "authoring", "input": {"path":"mixed.bin", "shape":list(x.shape), "dtype":"float16", "sha256":sha256_file(args.output / "mixed.bin")},
              "sourceWeight": source.records[-1], "priorCPUReferenceExact": True,
              "cpuLaneVsLinear": metrics(lane, cpu), "tolerances":{"maximumAbsoluteError":0.0005,"relativeL2Error":0.00001},
              "originalFullHeadToleranceUnchanged": original["tolerances"],
              "hypothesis": "Projection-only tests isolate GEMV reduction/compiler error from HC compiler differences; no attribution until device results.",
              "models": {}}
    write_json(args.output / "manifest.json", report)
    for name, module, kernels in (("baseline", FloatHeadProjection(weight), ()),
                                   ("candidate", MetalHeadProjection(weight), (get_head_kernel(),))):
        print(f"Authoring projection-only {name}", flush=True)
        asset = export_shared(module, {"main":{"x":x}}, ("logits",), args.output / f"{name}.aimodel", kernels)
        asset.pop("prefillFunction", None)  # This bounded probe has only main.
        report["models"][name] = asset
        spec = {"asset":f"{name}.aimodel", "function":"main", "inputs":{"x":{"file":"mixed.bin", "offset":0, "bytes":x.numel()*2, "shape":list(x.shape), "dtype":"float16"}},
                "output":f"device-{name}", "repeats":6, "mapped":False}
        write_json(args.output / f"{name}-spec.json", spec)
    write_json(args.output / "actual.json", {"inputs":{"x":tensor_json(x)}, "expectedOutputs":{"logits":tensor_json(torch.from_numpy(cpu))}})
    report["status"] = "cpu-authored-device-unvalidated"
    write_json(args.output / "manifest.json", report)
    (args.output / "README.md").write_text(
        "# Projection-only head diagnostic\n\n"
        "CPU-authored assets; run baseline-spec.json and candidate-spec.json in separate device processes. "
        "Both use the exact same FP16 mixed.bin input and original FP16 vocabulary weights. "
        "Their FP32 logits.bin outputs can be compared with expected-logits.bin and lane-oracle-logits.bin.\n\n"
        "The lane oracle models independent FP32 per-lane accumulation and a balanced reduction; "
        "it is an error-scale reference, not a specification of Metal simd_sum ordering. "
        "A passing isolated projection does not make full HC+head numerics or generation quality pass.\n")
    print(json.dumps({k:v for k,v in report.items() if k != "models"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
