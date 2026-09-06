#!/usr/bin/env python3
"""Compare native MoE output to captured MLX and an independent FP32 reference.

The FP32 reference uses the Swift route (reported separately against the actual
capture), and exact affine-decoded source Q4 weights. No expert assignment is
dropped. Thresholds are local block gates, not a full-model quality guarantee.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from export_moe import Source, errors, write_json


def read_tensor(container, name):
    tensor = container[name]
    return np.array(tensor["values"], dtype=np.float32).reshape(tensor["shape"])


def mlp(x, weights):
    matrix = {name: torch.from_numpy(value["dense32"].copy()) for name, value in weights.items()}
    with torch.inference_mode():
        gate = F.linear(x, matrix["gate_proj"])
        up = F.linear(x, matrix["up_proj"])
        return F.linear(F.silu(gate) * up, matrix["down_proj"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-capture-relative-l2", type=float, default=0.02)
    parser.add_argument("--max-fp32-relative-l2", type=float, default=0.005)
    args = parser.parse_args()
    torch.set_num_threads(4)
    fixture = json.loads(args.fixture.read_text())
    report = json.loads(args.report.read_text())
    x = torch.from_numpy(read_tensor(fixture["inputs"], "x")[0])
    outputs = report["outputs"]
    ids = read_tensor(outputs, "selected_experts")[0].astype(np.int32)
    routing = read_tensor(outputs, "routing_weights")[0]
    source = Source(report["layerIndex"])
    routed = torch.zeros_like(x)
    for expert in sorted(set(ids.ravel().tolist())):
        token_indices, slot_indices = np.where(ids == expert)
        projected = mlp(x[token_indices], source.expert(expert))
        routed[token_indices] += projected * torch.from_numpy(routing[token_indices, slot_indices, None])
    shared = mlp(x, source.shared())
    shared_gate = torch.from_numpy(source.read("shared_expert_gate.weight").copy())
    gate = torch.sigmoid(F.linear(x, shared_gate))
    y = (routed + shared * gate).numpy()[None, :, :]
    native = read_tensor(outputs, "y")
    captured = read_tensor(fixture["expectedOutputs"], "y")
    versus_fp32 = errors(native, y)
    versus_capture = errors(native, captured)
    fp32_versus_capture = errors(y, captured)
    gates = {
        "actual_routing_expert_sets_match": report["routingExpertSetsMatch"] is True,
        "versus_fp32_local_block_tolerance": versus_fp32["relative_l2"] is not None and versus_fp32["relative_l2"] <= args.max_fp32_relative_l2,
        "versus_capture_local_block_tolerance": versus_capture["relative_l2"] is not None and versus_capture["relative_l2"] <= args.max_capture_relative_l2,
        "all_values_finite": bool(np.isfinite(native).all()),
    }
    result = {
        "fixture": str(args.fixture.resolve()), "fixture_sha256": hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        "swift_report": str(args.report.resolve()), "swift_report_sha256": hashlib.sha256(args.report.read_bytes()).hexdigest(),
        "token_count": x.shape[0], "expert_ids": sorted(set(ids.ravel().tolist())),
        "reference": "CPU PyTorch FP32 exact affine-decoded Q4 and original BF16 shared weights; SwiGLU; Swift-selected routing weights; FP32 shared sigmoid and accumulation",
        "versus_fp32": versus_fp32, "versus_actual_mlx_capture": versus_capture,
        "fp32_reference_versus_actual_mlx_capture": fp32_versus_capture,
        "tolerances": {"capture_relative_l2": args.max_capture_relative_l2, "fp32_relative_l2": args.max_fp32_relative_l2},
        "gates": gates, "passed": all(gates.values()),
        "limits": "One layer on selected real token rows; local tolerances do not establish full-model quality, ANE hardware execution, or a performance win.",
        "source_slices": source.records,
    }
    write_json(args.output, result)
    print(json.dumps({k: result[k] for k in ("token_count", "versus_fp32", "versus_actual_mlx_capture", "gates", "passed")}, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
