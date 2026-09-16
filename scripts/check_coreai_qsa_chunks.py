#!/usr/bin/env python3
"""CPU-only QSA prefill/decode state handoff checks using one real weight layer.

Compares S4/S8/S16 followed by S1 with the same tokens processed entirely as S1.
Only one layer is loaded. Optional --export-small authors a small synthetic
multi-function archive for device probing; it never runs CoreAI/MLX/Core ML.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from export_moe import Source, sha256_file, write_json
from export_coreai_qsa import (BINDINGS, FIXTURES, INPUT_NAMES, OUTPUT_NAMES, QwenQSA, ROOT,
                               compare, export_chunk_asset, initial_state, read_fixture, save_sequence)

WEIGHT_NAMES = ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight", "q_norm.weight", "k_norm.weight",
                "indexer.index_qk_proj.weight", "indexer.q_layernorm.weight", "indexer.k_layernorm.weight")
# Declared before running. Integer cursors and every per-query mask must be exact.
CPU_TOLERANCES = {"maximumAbsoluteError": 0.02, "relativeL2Error": 0.002}


def execute(module, tokens, state, first_chunk):
    current = state
    outputs, masks = [], []
    position = 0
    with torch.inference_mode():
        while position < tokens.shape[1]:
            count = first_chunk if position == 0 else 1
            values = {"x": tokens[:, position:position + count].half(), **current}
            result = dict(zip(OUTPUT_NAMES, module(*(values[name] for name in INPUT_NAMES))))
            outputs.append(result["y"])
            masks.append(result["attention_mask"])
            current = {name: result[out] for name, out in BINDINGS.items()}
            position += count
    return torch.cat(outputs, dim=1), torch.cat(masks, dim=2), current


def metric(actual, expected):
    result = compare(actual, expected)
    result["passed"] = (result["finite"] and result["max_abs"] <= CPU_TOLERANCES["maximumAbsoluteError"]
                        and (result["relative_l2"] <= CPU_TOLERANCES["relativeL2Error"]
                             if result["relative_l2"] is not None else result["max_abs"] == 0))
    return result


def check_case(module, tokens, state, count, name, references=()):
    offset = int(state["offset"].item())
    end = offset + tokens.shape[1]
    if end > module.capacity:
        raise ValueError("CPU sequence exceeds cache capacity")
    single_y, single_mask, single_state = execute(module, tokens, state, 1)
    chunk_y, chunk_mask, chunk_state = execute(module, tokens, state, count)
    checks = {"output": metric(chunk_y, single_y), "mask": compare(chunk_mask, single_mask)}
    checks["mask"]["passed"] = checks["mask"]["exact"]
    for name_in in BINDINGS:
        checks[name_in] = metric(chunk_state[name_in], single_state[name_in])
        if not chunk_state[name_in].is_floating_point():
            checks[name_in]["passed"] = checks[name_in]["exact"]
    for key, axis, start, stop in (("key_cache", 2, offset, end), ("value_cache", 2, offset, end),
                                  ("raw_cache", 1, offset, end),
                                  ("pooled_cache", 1, int(state["pooled_count"].item()), end // module.ratio)):
        region = [slice(None)] * chunk_state[key].ndim
        region[axis] = slice(start, stop)
        checks[key + "_updated_rows"] = metric(chunk_state[key][tuple(region)], single_state[key][tuple(region)])
        prefix, suffix = region.copy(), region.copy()
        prefix[axis], suffix[axis] = slice(0, start), slice(stop, None)
        checks[key + "_unchanged_rows"] = {"passed": bool(
            torch.equal(chunk_state[key][tuple(prefix)], state[key][tuple(prefix)])
            and torch.equal(chunk_state[key][tuple(suffix)], state[key][tuple(suffix)]))}
    source_metrics = {str(step): compare(chunk_y[:, step:step + 1], fixture["expected.output"])
                      for step, fixture in references}
    return {"name": name, "chunk": count, "decodeSteps": tokens.shape[1] - count,
            "offsetBefore": offset, "offsetAfter": end,
            "pooledCountBefore": int(state["pooled_count"].item()),
            "pooledCountAfter": int(chunk_state["pooled_count"].item()),
            "visibleTokensPerQuery": chunk_mask.sum(-1).reshape(-1).tolist(),
            "checks": checks, "sourceBF16ComparisonNotParityAcceptance": source_metrics,
            "passed": all(check["passed"] for check in checks.values())}


def small_module(capacity=32):
    config = {"hidden_size": 8, "num_attention_heads": 2, "num_key_value_heads": 1,
              "head_dim": 8, "indexer_n_heads": 2, "indexer_head_dim": 8,
              "indexer_compress_ratio": 4, "indexer_budget": 8, "partial_rotary_factor": 0.5,
              "rms_norm_eps": 1e-6, "rope_parameters": {"rope_theta": 10000000}}
    rng = np.random.default_rng(123)
    shapes = {"q_proj.weight": (32, 8), "k_proj.weight": (8, 8), "v_proj.weight": (8, 8),
              "o_proj.weight": (8, 16), "indexer.index_qk_proj.weight": (24, 8)}
    weights = {name: (0.1 * rng.standard_normal(shape)).astype(np.float32) for name, shape in shapes.items()}
    for name in ("q_norm.weight", "k_norm.weight", "indexer.q_layernorm.weight", "indexer.k_layernorm.weight"):
        weights[name] = np.ones(8, dtype=np.float32)
    return QwenQSA(config, weights, capacity).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/coreai-pd/qsa-chunks")
    parser.add_argument("--chunk-sizes", nargs="+", type=int, choices=(4, 8, 16), default=[4, 8, 16])
    parser.add_argument("--export-small", action="store_true")
    args = parser.parse_args()
    if len(set(args.chunk_sizes)) != len(args.chunk_sizes):
        parser.error("--chunk-sizes must be unique")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Choose a fresh output directory")
    output.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    source = Source(3)
    config_path = source.directory / "config.json"
    config = json.loads(config_path.read_text())["text_config"]
    source.prefix = "language_model.model.layers.3.self_attn."
    weights = {name: source.read(name) for name in WEIGHT_NAMES}
    capacity = 2072
    module = QwenQSA(config, weights, capacity).eval()
    del weights
    fixture_manifest = json.loads((FIXTURES / "manifest.json").read_text())
    pre, _ = read_fixture(FIXTURES / "attention-continuous-0-prefill.safetensors", fixture_manifest)
    seeded, _ = read_fixture(FIXTURES / "qsa-threshold-0-prefill.safetensors", fixture_manifest)
    d1, _ = read_fixture(FIXTURES / "qsa-threshold-1-decode.safetensors", fixture_manifest)
    d2, _ = read_fixture(FIXTURES / "qsa-threshold-2-decode.safetensors", fixture_manifest)
    continuation = torch.cat((d1["input"], d2["input"], pre["input"]), dim=1)
    cases = []
    for count in sorted(args.chunk_sizes):
        cases.append(check_case(module, pre["input"][:, :count + 3].half(), initial_state(module), count, f"short-s{count}"))
        for offset in (2047, 2049, 2051):
            state = initial_state(module, seeded)
            state["offset"][:] = offset
            for key, axis in (("key_cache", 2), ("value_cache", 2), ("raw_cache", 1)):
                tail = [slice(None)] * state[key].ndim
                tail[axis] = slice(offset, None)
                state[key][tuple(tail)] = 0
            tokens = torch.cat((seeded["input"][:, offset:], continuation), dim=1)[:, :count + 3].half()
            refs = [(2051 - offset, d1), (2052 - offset, d2)]
            refs = [(step, fixture) for step, fixture in refs if step < tokens.shape[1]]
            cases.append(check_case(module, tokens, state, count, f"offset{offset}-s{count}", refs))
        # Existing independent BF16 state already has 513 pooled keys. Preserve
        # those keys exactly while chunking across the next complete block.
        state = initial_state(module, d1)
        old_pool = d1["expected.state.pooledIndexerKeys"].half()
        state["pooled_cache"][:, :old_pool.shape[1]] = old_pool
        state["pooled_count"][:] = old_pool.shape[1]
        cases.append(check_case(module, continuation[:, 1:count + 4].half(), state, count,
                                f"existing-pooled-offset2052-s{count}", [(0, d2)]))
    report = {"schema": "qwen-coreai-qsa-chunk-cpu-v1", "passed": all(case["passed"] for case in cases),
              "layer": 3, "capacity": capacity, "stateBindings": BINDINGS, "cases": cases,
              "cpuTolerances": CPU_TOLERANCES, "sourceRecords": source.records,
              "configSHA256": sha256_file(config_path), "fixtureManifestSHA256": sha256_file(FIXTURES / "manifest.json"),
              "authoringSources": [{"path": str(p), "sha256": sha256_file(p)} for p in
                                    (Path(__file__), Path(__file__).with_name("export_coreai_qsa.py"))],
              "limitations": ["CPU-only functional verification; no CoreAI device performance or placement evidence.",
                              "S1 and chunk share FP16 equations; BF16 source comparisons are separate and are not quality acceptance.",
                              "Full fixed-capacity pooled recomputation remains; this is not an optimized cache update kernel."]}
    if args.export_small:
        small = small_module()
        asset = export_chunk_asset(small, [1, *args.chunk_sizes], output / "qsa-small.aimodel")
        asset["path"] = "qsa-small.aimodel"
        report["smallAsset"] = asset
        tokens = torch.from_numpy(np.random.default_rng(83).standard_normal((1, 19, 8)).astype(np.float16))
        models = {f"s{count}": {"path": asset["path"], "function": name} for name, count in asset["functions"].items()}
        report["smallSequences"] = [save_sequence(output, f"small-s{count}", small,
            [tokens[:, :count], tokens[:, count:count + 1], tokens[:, count + 1:count + 2]], models=models)
            for count in args.chunk_sizes]
    write_json(output / "report.json", report)
    print(json.dumps({"passed": report["passed"], "cases": len(cases), "report": str(output / "report.json")}), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
