#!/usr/bin/env python3
"""CPU-only real-layer chunk GDN check against the unchanged CoreAI S1 math.

This reads one verified layer and replays captured MoE activations at its GDN
boundary. It does not instantiate the language model, export assets, or run a
CoreAI/MLX device. This is chunk equivalence, not BF16 source-model acceptance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from export_coreai_gdn import (GDN, GDNConfig, GDNPrefill, OUTPUT_NAMES, ROOT,
                               capture_input, finite_outputs)
from export_coreai_moe import errors
from export_moe import Source, sha256_file, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--chunks", default="4", help="CPU-only comparison lengths, e.g. 4,8,16")
    parser.add_argument("--capture", type=Path, default=ROOT / "fixtures/moe-real/prefill.safetensors")
    parser.add_argument("--output", type=Path, default=ROOT / "results/coreai-prefill/gdn-cpu.json")
    args = parser.parse_args(argv)
    chunks = [int(part) for part in args.chunks.split(",")]
    if not chunks or len(set(chunks)) != len(chunks) or any(chunk not in (4, 8, 16) for chunk in chunks):
        raise ValueError("--chunks requires unique sizes from 4,8,16")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    source = Source(args.layer)
    config_path = source.directory / "config.json"
    raw_config = json.loads(config_path.read_text())["text_config"]
    if not 0 <= args.layer < len(raw_config["layer_types"]) or raw_config["layer_types"][args.layer] != "linear_attention":
        raise ValueError("--layer must select a GDN layer from the real config")
    config = GDNConfig.from_model(raw_config)
    source.prefix = f"language_model.model.layers.{args.layer}.linear_attn."
    weights = {name: source.read(name) for name in config.weight_shapes}
    model = GDN(config, weights).eval()
    del weights
    candidates = [("original_projections", GDNPrefill.from_gdn(model).eval()),
                  ("fused_input_projection", GDNPrefill.from_gdn(model, fuse_input_projections=True).eval())]
    warmup, decode = 2, 3
    hidden, evidence = capture_input(args.capture.resolve(), warmup + max(chunks) + decode, config.hidden)
    history = torch.zeros(1, config.kernel - 1, config.channels, dtype=torch.float16)
    state = torch.zeros(1, config.value_heads, config.value_dim, config.key_dim, dtype=torch.float32)
    report = {"version": 1, "layer": args.layer, "passed": False, "runtime_executed": False,
              "scope": "Real-weight CPU chunk/state equivalence against existing FP16/FP32 S1; no device speed or BF16 quality claim",
              "configSHA256": sha256_file(config_path), "input_provenance": evidence,
              "source_records": source.records, "warmup_s1_steps": warmup, "decode_s1_steps": decode,
              "tolerances": {"relativeL2Error": 0.005, "maximumAbsoluteError": 0.02}, "checks": []}
    with torch.inference_mode():
        for position in range(warmup):
            _, history, state = model(hidden[:, position:position+1], history, state)
        if not bool(torch.count_nonzero(history)) or not bool(torch.count_nonzero(state)):
            raise ValueError("The warmup failed to create nonzero state")
        saved_history, saved_state = history.clone(), state.clone()
        for chunk in chunks:
            end = warmup + chunk + decode
            outputs = []
            scalar_history, scalar_state = history, state
            for position in range(warmup, end):
                values = model(hidden[:, position:position+1], scalar_history, scalar_state)
                finite_outputs(values)
                outputs.append(values[0])
                scalar_history, scalar_state = values[1:]
            expected = (torch.cat(outputs, dim=1), scalar_history, scalar_state)
            for name, candidate in candidates:
                values = candidate(hidden[:, warmup:warmup+chunk], history, state)
                finite_outputs(values)
                pieces = [values[0]]
                next_history, next_state = values[1:]
                for position in range(warmup + chunk, end):
                    values = model(hidden[:, position:position+1], next_history, next_state)
                    finite_outputs(values)
                    pieces.append(values[0])
                    next_history, next_state = values[1:]
                actual = (torch.cat(pieces, dim=1), next_history, next_state)
                comparisons = {key: errors(value.float().numpy(), target.float().numpy())
                               for key, value, target in zip(OUTPUT_NAMES, actual, expected)}
                passed = all(row["relative_l2"] <= 0.005 and row["max_abs"] <= 0.02
                             for row in comparisons.values())
                report["checks"].append({"chunk": chunk, "mode": name, "passed": passed,
                    "state_dtype": str(next_state.dtype), "comparisons": comparisons})
                print(f"S{chunk} {name}: {comparisons}", flush=True)
                if not torch.equal(history, saved_history) or not torch.equal(state, saved_state):
                    raise ValueError("Chunk execution mutated its input state")
    report["passed"] = all(row["passed"] for row in report["checks"])
    write_json(args.output.resolve(), report)
    if not report["passed"]:
        raise AssertionError(f"CPU chunk equivalence failed: {args.output}")
    print(f"CPU chunk equivalence passed: {args.output}", flush=True)


if __name__ == "__main__":
    main()
