#!/usr/bin/env python3
"""CPU author exact-shape stage probes for the current large-chunk QSA candidate.

Separate model.predict timings locate work; their sum is not an end-to-end
latency estimate because stage boundaries materialize tensors and add dispatches.
No GPU, CoreAI inference, Core ML inference, or MLX is called by this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from coreai_qsa_chunk import (QwenQSAChunk, get_pool_kernel, replay_state,
                              repeat_activations, write_tensor_json)
from export_coreai_qsa import (BINDINGS, FIXTURES, INPUT_NAMES, OUTPUT_NAMES,
                               QwenQSA, compare, read_fixture)
from export_moe import Source, sha256_file, write_json


PROJECTION_OUTPUTS = ("query", "gate", "keys", "values", "raw_out", "pooled_out", "index", "end", "full_blocks")


class ProjectionCache(torch.nn.Module):
    def __init__(self, candidate):
        super().__init__()
        self.candidate = candidate

    def forward(self, x, key_cache, value_cache, raw_cache, pooled_cache, offset, pooled_count):
        c, s = self.candidate, self.candidate.source
        count = x.shape[1]
        positions = offset + torch.arange(count, dtype=torch.int32)
        end = offset + count
        query_gate = c.linear(x, "q_proj.weight").reshape(1, count, s.heads, s.head_dim * 2)
        query, gate = query_gate.split(s.head_dim, dim=-1)
        key = c.linear(x, "k_proj.weight").reshape(1, count, s.kv_heads, s.head_dim)
        value = c.linear(x, "v_proj.weight").reshape(1, count, s.kv_heads, s.head_dim).transpose(1, 2)
        query = s.rope(s.norm(query, "q_norm.weight").transpose(1, 2), positions)
        key = s.rope(s.norm(key, "k_norm.weight").transpose(1, 2), positions)
        indices = positions.long()[None, None, :, None].expand(1, s.kv_heads, count, s.head_dim)
        keys = key_cache.scatter(2, indices, key)
        values = value_cache.scatter(2, indices, value)
        index = c.linear(x, "indexer.index_qk_proj.weight")
        raw = index[..., s.idx_heads * s.idx_dim:]
        raw_indices = positions.long()[None, :, None].expand(1, count, s.idx_dim)
        raw_out = raw_cache.scatter(1, raw_indices, raw)
        full_blocks = torch.div(end, s.ratio, rounding_mode="floor")
        pooled_out = get_pool_kernel(s.ratio, s.rope_dim, s.eps)(
            raw_out, pooled_cache, s.indexer_k_layernorm_weight, s.cosine, s.sine, pooled_count, full_blocks,
            threads_per_grid=(s.blocks * 32, 1, 1), threads_per_thread_group=(32, 1, 1),
            result_shapes=[[1, s.blocks, s.idx_dim]])
        return query, gate, keys, values, raw_out, pooled_out, index, end, full_blocks


class IndexScores(torch.nn.Module):
    def __init__(self, source):
        super().__init__()
        self.source = source

    def forward(self, index, pooled, offset):
        s = self.source
        count = index.shape[1]
        positions = offset + torch.arange(count, dtype=torch.int32)
        iq = s.norm(index[..., :s.idx_heads * s.idx_dim].reshape(1, count, s.idx_heads, s.idx_dim),
                    "indexer.q_layernorm.weight").transpose(1, 2)
        iq = s.rope(iq, positions)
        scores = torch.relu(iq.float() @ pooled[:, None].float().transpose(-1, -2)).sum(1)
        # Keep outputs finite for the existing JSON fixture reader. Visibility
        # and -inf masking remain exactly as before inside the next stage.
        return scores - (s.block_positions.float() / s.ratio) * 1e-7


class TopKMask(torch.nn.Module):
    def __init__(self, source):
        super().__init__()
        self.source = source

    def forward(self, scores, offset):
        s = self.source
        count = scores.shape[1]
        positions = offset + torch.arange(count, dtype=torch.int32)
        causal = s.cache_positions[None, :] <= positions[:, None]
        if s.blocks > s.topk:
            visible = s.block_positions[None, :] + s.ratio - 1 <= positions[:, None]
            scores = torch.where(visible[None], scores, float("-inf"))
            chosen = scores.topk(s.topk, dim=-1, sorted=False).indices
            selected = torch.zeros_like(scores, dtype=torch.int32).scatter(-1, chosen, 1).bool() & visible[None]
            selected_tokens = selected.repeat_interleave(s.ratio, dim=-1)
            tail_start = torch.div(positions + 1, s.ratio, rounding_mode="floor") * s.ratio
            tail = s.cache_positions[None, :] >= tail_start[:, None]
            sparse = (selected_tokens | tail[None]) & causal[None]
            mask = torch.where(offset + count > s.budget + s.ratio - 1, sparse, causal[None])[:, None]
        else:
            mask = causal[None, None]
        return mask.to(torch.int32)


class Indexer(torch.nn.Module):
    def __init__(self, source):
        super().__init__()
        self.scores = IndexScores(source)
        self.mask = TopKMask(source)

    def forward(self, index, pooled, offset):
        return self.mask(self.scores(index, pooled, offset), offset)


class Attention(torch.nn.Module):
    def __init__(self, source):
        super().__init__()
        self.sdpa = source.sdpa

    def forward(self, query, keys, values, mask):
        # The production op consumes these same F32 shapes. Only the fixture
        # boundary is Int32 because CoreMLTensor has no boolean representation.
        return self.sdpa(query, keys, values, attn_mask=mask.bool())


class AttentionHalfIO(torch.nn.Module):
    def __init__(self, source, *, upcast):
        super().__init__()
        self.sdpa = source.sdpa
        self.upcast = upcast

    def forward(self, query, keys, values, mask):
        if self.upcast:
            query, keys, values = query.float(), keys.float(), values.float()
        return self.sdpa(query, keys, values, attn_mask=mask.bool()).half()


class OutputProjection(torch.nn.Module):
    def __init__(self, candidate):
        super().__init__()
        self.candidate = candidate

    def forward(self, attention, gate):
        s = self.candidate.source
        count = gate.shape[1]
        gated = (attention.half().transpose(1, 2) * torch.sigmoid(gate)).half().reshape(1, count, s.heads * s.head_dim)
        return self.candidate.linear(gated, "o_proj.weight")


def capture_stages(candidate, inputs):
    """Independent stage composition must reproduce every production output."""
    s = candidate.source
    with torch.inference_mode():
        projection = ProjectionCache(candidate).eval()
        parts = dict(zip(PROJECTION_OUTPUTS, projection(*(inputs[name] for name in INPUT_NAMES))))
        index_inputs = {"index": parts["index"], "pooled": parts["pooled_out"], "offset": inputs["offset"]}
        scores = IndexScores(s)(*index_inputs.values())
        mask = TopKMask(s)(scores, inputs["offset"])
        attention_inputs = {"query": parts["query"].float(), "keys": parts["keys"].float(),
                            "values": parts["values"].float(), "mask": mask}
        attention = Attention(s)(*attention_inputs.values())
        half_inputs = {"query": parts["query"], "keys": parts["keys"], "values": parts["values"], "mask": mask}
        y = OutputProjection(candidate)(attention, parts["gate"])
        stitched = (y, parts["keys"], parts["values"], parts["raw_out"], parts["pooled_out"],
                    parts["end"], parts["full_blocks"], mask)
        original = candidate(*(inputs[name] for name in INPUT_NAMES))
        checks = {name: compare(a, b) for name, a, b in zip(OUTPUT_NAMES, stitched, original)}
        if not all(value["exact"] for value in checks.values()):
            raise AssertionError(f"Profile stage composition changed source math: {checks}")
    # SDPA is intentionally first: export an unweighted probe before the
    # independently useful sorter/indexer and larger projection assets.
    return [
        ("sdpa", Attention(s), attention_inputs, {"attention": attention}),
        ("sdpa_fp32_halfio", AttentionHalfIO(s, upcast=True), half_inputs, {"attention": attention.half()}),
        ("sdpa_fp16", AttentionHalfIO(s, upcast=False), half_inputs, {"attention": attention.half()}),
        ("indexer", Indexer(s), index_inputs, {"mask": mask}),
        ("index_scores", IndexScores(s), index_inputs, {"scores": scores}),
        ("topk_mask", TopKMask(s), {"scores": scores, "offset": inputs["offset"]}, {"mask": mask}),
        ("projection_cache", projection, inputs, parts),
        ("output_projection", OutputProjection(candidate), {"attention": attention, "gate": parts["gate"]}, {"y": y}),
    ], checks


def export_stage(name, model, inputs, expected, path, custom_kernels):
    import coreai_torch
    from coreai_torch.composite_ops import SDPA
    began = time.perf_counter()
    path.mkdir(parents=True, exist_ok=False)
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    converter.register_custom_kernels(custom_kernels)
    args = tuple(inputs.values())
    converter.add_pytorch_module(model.eval(), input_names=tuple(inputs), output_names=tuple(expected),
        externalize_modules=[coreai_torch.ExternalizeSpec(target_class=SDPA,
            composite_op_name="scaled_dot_product_attention", composite_attrs=["scale", "is_causal", "window_size"])],
        export_fn=lambda m: torch.export.export(m, args=args).run_decompositions(coreai_torch.get_decomp_table()))
    program = converter.to_coreai()
    program.optimize()
    (path / "coreai-main-after.txt").write_text(str(program.get_graph("main")))
    program.save_asset(path / "stage.aimodel")
    write_tensor_json(path / "fixture.json", expected, inputs=inputs)
    with torch.inference_mode():
        actual = model(*args)
        actual = actual if isinstance(actual, tuple) else (actual,)
        cpu_comparison = {key: compare(a, b) for (key, b), a in zip(expected.items(), actual)}
    report = {"stage": name, "model": "stage.aimodel", "function": "main", "fixture": "fixture.json",
        "modelBytes": sum(p.stat().st_size for p in (path / "stage.aimodel").rglob("*") if p.is_file()),
        "inputs": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in inputs.items()},
        "outputs": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in expected.items()},
        "cpuComparison": cpu_comparison,
        "authoringSeconds": time.perf_counter() - began, "deviceValidated": False}
    write_json(path / "manifest.json", report)
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=16384)
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--offset", type=int, default=8192)
    args = parser.parse_args()
    if min(args.count, args.capacity) <= 0 or args.offset < 0 or args.offset + args.count > args.capacity:
        parser.error("Positive count/capacity and valid offset+count required")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    from check_coreai_qsa_chunks import WEIGHT_NAMES
    reader = Source(3)
    config_path = reader.directory / "config.json"
    config = json.loads(config_path.read_text())["text_config"]
    reader.prefix = "language_model.model.layers.3.self_attn."
    source = QwenQSA(config, {name: reader.read(name) for name in WEIGHT_NAMES}, args.capacity).eval()
    fixture_manifest = json.loads((FIXTURES / "manifest.json").read_text())
    captured, evidence = read_fixture(FIXTURES / "attention-continuous-0-prefill.safetensors", fixture_manifest)
    rows = captured["input"].half()
    inputs = {"x": repeat_activations(rows, args.count, args.offset), **replay_state(source, rows, args.offset)}
    candidate = QwenQSAChunk(source).eval()
    stages, checks = capture_stages(candidate, inputs)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "cpu-composition.json", checks)
    reports = []
    for name, model, values, expected in stages:
        reports.append(export_stage(name, model, values, expected, args.output / name, candidate.custom_kernels()))
    write_json(args.output / "manifest.json", {"scope": "CPU stage authoring only; device timings pending",
        "layer": 3, "capacity": args.capacity, "count": args.count, "offset": args.offset, "stages": reports,
        "modelDirectory": str(reader.directory), "configSHA256": sha256_file(config_path),
        "sourceRecords": reader.records, "capture": evidence, "scriptSHA256": sha256_file(Path(__file__)),
        "provenance": "Repeated verified layer0 MoE activation at layer3 attention; not native full-model layer3 capture",
        "limitations": ["Standalone stage timings include extra materialization/dispatch and are not additive end-to-end proof.",
            "SDPA inputs exactly match current FP32 shape; fixture Int32 mask converts to Bool inside graph.",
            "sdpa_fp16 and sdpa_fp32_halfio share identical half inputs and FP32-to-half oracle; FP16 candidate is not an accepted default.",
            "index_scores returns finite scores before visibility masking; topk_mask applies original -inf internally.",
            "Stage CPU outputs compose exactly to QwenQSAChunk; accelerator arithmetic is separately validated."]})


if __name__ == "__main__":
    main()
