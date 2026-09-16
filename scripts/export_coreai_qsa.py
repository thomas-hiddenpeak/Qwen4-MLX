#!/usr/bin/env python3
"""Export bounded real layer-3 attention/QSA with explicit persistent tensor I/O.

CPU authoring only: never calls CoreAI, Core ML, MLX, GPU, or ANE inference.
Input activations replay the captured layer-0 MoE input at the attention boundary;
they are NOT a native layer-3 activation capture. The sparse case starts from an
existing independently generated MLX 2051-token state, avoiding full-model loading.
FP16 I/O/weights with FP32 projection, normalization, scores and SDPA intermediates
is an accuracy-first backend candidate, not BF16 bit parity or an optimized kernel.
The fixed cache and full pooled-key recomputation are validation scaffolding.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import struct
import time

import numpy as np
import torch
import torch.nn.functional as F

from export_moe import Source, sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures/gpu-sequence-reference"
INPUT_NAMES = ("x", "key_cache", "value_cache", "raw_cache", "pooled_cache", "offset", "pooled_count")
OUTPUT_NAMES = ("y", "key_cache_out", "value_cache_out", "raw_cache_out", "pooled_cache_out", "offset_out", "pooled_count_out", "attention_mask")
BINDINGS = dict(zip(INPUT_NAMES[1:], OUTPUT_NAMES[1:7]))


def tensor_json(value):
    a = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    return {"shape": list(a.shape), "dtype": str(a.dtype), "values": a.reshape(-1).astype(float).tolist()}


def read_fixture(path, manifest):
    evidence = next(s for c in manifest["cases"] for s in c["steps"] if s["file"] == path.name)
    if sha256_file(path) != evidence["sha256"]:
        raise ValueError(f"Fixture hash differs: {path}")
    tensors = {}
    with path.open("rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
        base = size + 8
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            start, end = entry["data_offsets"]
            f.seek(base + start)
            raw = f.read(end - start)
            if entry["dtype"] == "BF16":
                a = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
            elif entry["dtype"] == "BOOL":
                a = np.frombuffer(raw, dtype=np.bool_)
            else:
                raise ValueError(f"Unsupported fixture dtype {entry['dtype']}")
            tensors[name] = torch.from_numpy(a.copy().reshape(entry["shape"]))
    return tensors, evidence


def compare(actual, expected):
    a, b = actual.detach().float().numpy(), expected.detach().float().numpy()
    delta = a.astype(np.float64) - b.astype(np.float64)
    norm = np.linalg.norm(b.astype(np.float64))
    return {"relative_l2": float(np.linalg.norm(delta) / norm) if norm else None,
            "max_abs": float(np.abs(delta).max()), "exact": bool(np.array_equal(a, b)),
            "finite": bool(np.isfinite(a).all() and np.isfinite(b).all())}


class QwenQSA(torch.nn.Module):
    """Fixed capacity functional state update, with real 24:2 GQA and top512 QSA."""
    def __init__(self, config, weights, capacity):
        super().__init__()
        if capacity <= 0 or capacity % config["indexer_compress_ratio"]:
            raise ValueError("Cache capacity must be positive and divisible by the compression ratio")
        self.capacity = capacity
        self.hidden = config["hidden_size"]
        self.heads = config["num_attention_heads"]
        self.kv_heads = config["num_key_value_heads"]
        self.head_dim = config["head_dim"]
        self.idx_heads = config["indexer_n_heads"]
        self.idx_dim = config["indexer_head_dim"]
        self.ratio = config["indexer_compress_ratio"]
        self.budget = config["indexer_budget"]
        self.topk = self.budget // self.ratio
        self.blocks = capacity // self.ratio
        self.rope_dim = int(config["head_dim"] * config["partial_rotary_factor"])
        self.eps = config["rms_norm_eps"]
        if (config.get("indexer_kv_heads", 1) != 1 or config.get("output_gate_type", "sigmoid") != "sigmoid"
                or self.heads % self.kv_heads or self.budget % self.ratio
                or not 0 < self.rope_dim <= min(self.head_dim, self.idx_dim) or self.rope_dim % 2):
            raise ValueError("Unsupported Qwen QSA configuration")
        expected_shapes = {"q_proj.weight": (self.heads * self.head_dim * 2, self.hidden),
            "k_proj.weight": (self.kv_heads * self.head_dim, self.hidden),
            "v_proj.weight": (self.kv_heads * self.head_dim, self.hidden),
            "o_proj.weight": (self.hidden, self.heads * self.head_dim),
            "q_norm.weight": (self.head_dim,), "k_norm.weight": (self.head_dim,),
            "indexer.index_qk_proj.weight": ((self.idx_heads + 1) * self.idx_dim, self.hidden),
            "indexer.q_layernorm.weight": (self.idx_dim,), "indexer.k_layernorm.weight": (self.idx_dim,)}
        if set(weights) != set(expected_shapes):
            raise ValueError("Attention weight set differs from configured projections and norms")
        for name, value in weights.items():
            if value.shape != expected_shapes[name] or not np.isfinite(value.astype(np.float16)).all():
                raise ValueError(f"Invalid shape or FP16 weight range: {name}")
            self.register_buffer(name.replace(".", "_"), torch.from_numpy(value.copy()).half())
        self.register_buffer("cache_positions", torch.arange(capacity, dtype=torch.int32))
        self.register_buffer("block_positions", torch.arange(self.blocks, dtype=torch.int32) * self.ratio)
        theta = config["rope_parameters"]["rope_theta"]
        freq = torch.exp(torch.arange(self.rope_dim // 2, dtype=torch.float32) * (-2 * np.log(theta) / self.rope_dim))
        # Positions beyond this fixed cache are not used by these bounded fixtures.
        angles = torch.arange(capacity, dtype=torch.float32)[:, None] * freq
        self.register_buffer("cosine", torch.cos(angles))
        self.register_buffer("sine", torch.sin(angles))
        from coreai_torch.composite_ops import SDPA
        self.sdpa = SDPA(scale=self.head_dim ** -0.5, is_causal=False)

    def linear(self, x, name):
        return F.linear(x.float(), getattr(self, name.replace(".", "_")).float()).half()

    def norm(self, x, name):
        weight = getattr(self, name.replace(".", "_")).float()
        xx = x.float()
        return (xx * torch.rsqrt(xx.square().mean(-1, keepdim=True) + self.eps) * weight).half()

    def rope(self, x, positions, pooled=False):
        cos = torch.index_select(self.cosine, 0, positions.long())[None, None]
        sin = torch.index_select(self.sine, 0, positions.long())[None, None]
        a, b = x[..., :self.rope_dim // 2], x[..., self.rope_dim // 2:self.rope_dim]
        if pooled:
            # Source pooled RoPE rounds cos/sin and each multiply before addition.
            cos, sin = cos.half(), sin.half()
            first = (a * cos).half() - (b * sin).half()
            second = (b * cos).half() + (a * sin).half()
        else:
            first = (a.float() * cos - b.float() * sin).half()
            second = (b.float() * cos + a.float() * sin).half()
        return torch.cat((first, second, x[..., self.rope_dim:]), dim=-1)

    def forward(self, x, key_cache, value_cache, raw_cache, pooled_cache, offset, pooled_count):
        count = x.shape[1]
        positions = offset + torch.arange(count, dtype=torch.int32)
        end = offset + count
        query_gate = self.linear(x, "q_proj.weight").reshape(1, count, self.heads, self.head_dim * 2)
        query, gate = query_gate.split(self.head_dim, dim=-1)
        key = self.linear(x, "k_proj.weight").reshape(1, count, self.kv_heads, self.head_dim)
        value = self.linear(x, "v_proj.weight").reshape(1, count, self.kv_heads, self.head_dim).transpose(1, 2)
        query = self.rope(self.norm(query, "q_norm.weight").transpose(1, 2), positions)
        key = self.rope(self.norm(key, "k_norm.weight").transpose(1, 2), positions)
        kv_indices = positions.long()[None, None, :, None].expand(1, self.kv_heads, count, self.head_dim)
        keys = key_cache.scatter(2, kv_indices, key)
        values = value_cache.scatter(2, kv_indices, value)
        index = self.linear(x, "indexer.index_qk_proj.weight")
        raw = index[..., self.idx_heads * self.idx_dim:]
        raw_indices = positions.long()[None, :, None].expand(1, count, self.idx_dim)
        raw_out = raw_cache.scatter(1, raw_indices, raw)
        pooled = raw_out.reshape(1, self.blocks, self.ratio, self.idx_dim).float().mean(2).half()
        pooled = self.norm(pooled, "indexer.k_layernorm.weight")
        pooled = self.rope(pooled[:, None], self.block_positions, pooled=True)[:, 0]
        full_blocks = torch.div(end, self.ratio, rounding_mode="floor")
        update = ((self.block_positions // self.ratio >= pooled_count) &
                  (self.block_positions // self.ratio < full_blocks))[None, :, None]
        pooled_out = torch.where(update, pooled, pooled_cache)
        causal = self.cache_positions[None, :] <= positions[:, None]
        if self.blocks > self.topk:
            iq = self.norm(index[..., :self.idx_heads * self.idx_dim].reshape(1, count, self.idx_heads, self.idx_dim),
                           "indexer.q_layernorm.weight").transpose(1, 2)
            iq = self.rope(iq, positions)
            scores = torch.relu(iq.float() @ pooled_out[:, None].float().transpose(-1, -2)).sum(1)
            visible = self.block_positions[None, :] + self.ratio - 1 <= positions[:, None]
            scores = scores - (self.block_positions.float() / self.ratio) * 1e-7
            scores = torch.where(visible[None], scores, float("-inf"))
            chosen = scores.topk(self.topk, dim=-1, sorted=False).indices
            selected = torch.zeros_like(scores, dtype=torch.int32).scatter(-1, chosen, 1).bool() & visible[None]
            selected_tokens = selected.repeat_interleave(self.ratio, dim=-1)
            tail_start = torch.div(positions + 1, self.ratio, rounding_mode="floor") * self.ratio
            tail = self.cache_positions[None, :] >= tail_start[:, None]
            sparse = (selected_tokens | tail[None]) & causal[None]
            mask = torch.where(end > self.budget + self.ratio - 1, sparse, causal[None])[:, None]
        else:
            mask = causal[None, None]
        # FP32 is deliberate for candidate accuracy, not a throughput claim.
        attention = self.sdpa(query.float(), keys.float(), values.float(), attn_mask=mask).half()
        gated = (attention.transpose(1, 2) * torch.sigmoid(gate)).half().reshape(1, count, self.heads * self.head_dim)
        y = self.linear(gated, "o_proj.weight")
        return y, keys, values, raw_out, pooled_out, end, full_blocks, mask.to(torch.int32)


def initial_state(module, seeded=None):
    c, d, h = module.capacity, module.head_dim, module.kv_heads
    state = {"key_cache": torch.zeros(1, h, c, d, dtype=torch.float16),
             "value_cache": torch.zeros(1, h, c, d, dtype=torch.float16),
             "raw_cache": torch.zeros(1, c, module.idx_dim, dtype=torch.float16),
             "pooled_cache": torch.zeros(1, module.blocks, module.idx_dim, dtype=torch.float16),
             "offset": torch.zeros(1, dtype=torch.int32), "pooled_count": torch.zeros(1, dtype=torch.int32)}
    if seeded is not None:
        count = seeded["expected.state.keys"].shape[2]
        state["key_cache"][:, :, :count] = seeded["expected.state.keys"].half()
        state["value_cache"][:, :, :count] = seeded["expected.state.values"].half()
        state["raw_cache"][:, :count] = seeded["expected.state.rawIndexerKeys"].half()
        state["offset"][:] = count
    return state


def export_asset(module, count, destination):
    import coreai_torch
    from coreai_torch.composite_ops import SDPA
    state = initial_state(module)
    example = {"x": torch.zeros(1, count, module.hidden, dtype=torch.float16), **state}
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    converter.add_pytorch_module(module, input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
        externalize_modules=[coreai_torch.ExternalizeSpec(target_class=SDPA, composite_op_name="scaled_dot_product_attention",
                                                        composite_attrs=["scale", "is_causal", "window_size"])],
        export_fn=lambda m: torch.export.export(m, args=tuple(example[n] for n in INPUT_NAMES)).run_decompositions(coreai_torch.get_decomp_table()))
    program = converter.to_coreai()
    program.optimize()
    program.save_asset(destination)
    return {"path": str(destination), "sequence": count, "files": [
        {"path": str(p.relative_to(destination)), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
        for p in sorted(destination.rglob("*")) if p.is_file()]}


def save_sequence(directory, name, module, chunks, seeded=None, references=None):
    state = initial_state(module, seeded)
    initial = {k: tensor_json(v) for k, v in state.items()}
    steps, evidence = [], []
    for step, x in enumerate(chunks):
        count = x.shape[1]
        values = {"x": x.half(), **state}
        start = int(state["offset"].item())
        if count < 1 or start < 0 or start + count > module.capacity:
            raise ValueError("Sequence step exceeds fixed cache capacity")
        with torch.inference_mode():
            result = dict(zip(OUTPUT_NAMES, module(*(values[n] for n in INPUT_NAMES))))
        if not all(torch.isfinite(t).all() for t in result.values()):
            raise ValueError("Nonfinite CPU attention reference")
        path = directory / f"{name}-{step}.json"
        write_json(path, {"inputs": {"x": tensor_json(x.half())}, "expectedOutputs": {k: tensor_json(v) for k, v in result.items()}})
        steps.append({"name": f"{name}-{step}", "model": f"s{count}",
                      "fixture": path.name, "phase": "decode" if count == 1 else "prefill"})
        row = {"step": step, "offset_before": start, "offset_after": start + count,
               "qsa_active": start + count > module.budget + module.ratio - 1,
               "visible_tokens": result["attention_mask"].sum(-1).reshape(-1).tolist()}
        if references and step in references:
            ref = references[step]
            row["candidate_vs_existing_bf16_reference"] = compare(result["y"], ref["expected.output"])
            if "reference.qsa_mask" in ref:
                row["qsa_mask_vs_existing_bf16_reference"] = compare(result["attention_mask"][..., :start + count], ref["reference.qsa_mask"])
        evidence.append(row)
        state = {inp: result[out] for inp, out in BINDINGS.items()}
    initial_path = directory / f"{name}-initial-state.json"
    write_json(initial_path, initial)
    sequence = {"version": 1, "initialState": initial_path.name, "stateBindings": BINDINGS, "steps": steps,
                "models": {f"s{count}": {"path": f"attention-c{module.capacity}-s{count}.aimodel", "function": "main"}
                           for count in sorted({x.shape[1] for x in chunks})},
                "tolerances": {"maximumAbsoluteError": 0.02, "relativeL2Error": 0.005}}
    write_json(directory / f"{name}-sequence.json", sequence)
    return {"name": name, "sequence": str(directory / f"{name}-sequence.json"), "steps": evidence}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/coreai-stateful/qsa")
    parser.add_argument("--layer", type=int, choices=(3,), default=3,
                        help="Boundary-state fixture is specific to layer 3")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--cases", nargs="+", choices=("short", "sparse"), default=["short", "sparse"])
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "manifest.json").exists():
        raise FileExistsError("Choose a fresh output directory")
    source = Source(args.layer)
    config_path = source.directory / "config.json"
    config = json.loads(config_path.read_text())["text_config"]
    if config["layer_types"][args.layer] != "full_attention":
        raise ValueError("Selected layer is not full attention")
    source.prefix = f"language_model.model.layers.{args.layer}.self_attn."
    names = ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight", "q_norm.weight", "k_norm.weight",
             "indexer.index_qk_proj.weight", "indexer.q_layernorm.weight", "indexer.k_layernorm.weight")
    weights = {name: source.read(name) for name in names}
    fixture_manifest = json.loads((FIXTURES / "manifest.json").read_text())
    pre, _ = read_fixture(FIXTURES / "attention-continuous-0-prefill.safetensors", fixture_manifest)
    dec, _ = read_fixture(FIXTURES / "attention-continuous-1-decode.safetensors", fixture_manifest)
    sequences, assets = [], []
    for case in args.cases:
        capacity = 32 if case == "short" else 2056
        module = QwenQSA(config, weights, capacity).eval()
        if case == "short":
            chunks = [pre["input"][:, :4], pre["input"][:, 4:5], pre["input"][:, 5:6], pre["input"][:, 6:7]]
            seeded, refs = None, None
        else:
            seeded, _ = read_fixture(FIXTURES / "qsa-threshold-0-prefill.safetensors", fixture_manifest)
            d1, _ = read_fixture(FIXTURES / "qsa-threshold-1-decode.safetensors", fixture_manifest)
            d2, _ = read_fixture(FIXTURES / "qsa-threshold-2-decode.safetensors", fixture_manifest)
            chunks = [d1["input"], d2["input"], pre["input"][:, 1:2], pre["input"][:, 2:3], pre["input"][:, 3:4]]
            refs = {0: d1, 1: d2}
        sequences.append(save_sequence(args.output, case, module, chunks, seeded, refs))
        if not args.skip_export:
            for count in sorted({x.shape[1] for x in chunks}):
                path = args.output / f"attention-c{capacity}-s{count}.aimodel"
                began = time.perf_counter()
                asset = export_asset(module, count, path)
                asset["export_seconds"] = time.perf_counter() - began
                assets.append(asset)
                print(f"Exported {path.name}", flush=True)
    report = {"schema": "qwen-coreai-attention-sequence-v1", "layer": args.layer,
              "weights": source.records, "config_sha256": sha256_file(config_path),
              "fixture_manifest_sha256": sha256_file(FIXTURES / "manifest.json"),
              "fixture_source_commit": fixture_manifest["source_commit"],
              "script_sha256": sha256_file(Path(__file__)), "torch": torch.__version__,
              "coreai_torch": importlib.metadata.version("coreai-torch"),
              "state_bindings": BINDINGS, "assets": assets, "sequences": sequences,
              "limitations": ["CPU authoring/reference only; runtime, numerical acceptance, and hardware placement require separate validation.",
                  "Real layer weights; layer0 MoE activation replay, not native layer3 capture.",
                  "FP16 rounded state/projections differ from baseline BF16; reference comparisons are reported.",
                  "Fixed cache and full pooled recomputation are correctness candidates, not optimized production cache management.",
                  "Caller must enforce offset >= 0 and offset + input sequence <= cache capacity before execution.",
                  "Sparse case begins with existing independently computed MLX BF16 state at 2051 tokens, converted losslessly to FP16 where representable.",
                  "Text-only sublayer: no decoder residual, input layernorm, GDN, MoE, PLE, tokenizer, or generation API."]}
    write_json(args.output / "manifest.json", report)
    print(json.dumps({"manifest": str(args.output / "manifest.json"), "sequences": sequences}, indent=2), flush=True)


if __name__ == "__main__":
    main()
