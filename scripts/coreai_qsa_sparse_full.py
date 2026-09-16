#!/usr/bin/env python3
"""Experimental full QSA wrapper: sparse prefill, unchanged S1 and state I/O.

The existing indexer still selects blocks. Only prefill attention consumes its
compact IDs directly instead of expanding the selected set through native SDPA.
Mask outputs remain available for the same diagnostic/reference contract.
CPU authoring only; neither the default QSA wrapper nor exporter is changed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from coreai_qsa_chunk import CompactProbe, PROBE_OUTPUTS, QwenQSAChunk, get_pool_kernel
from coreai_qsa_sparse import SparseAttention, get_sparse_kernel
from export_coreai_qsa import BINDINGS, INPUT_NAMES, QwenQSA, initial_state
from export_moe import Source, sha256_file, write_json


class QwenQSASparsePrefill(QwenQSAChunk):
    def __init__(self, source, tile_m=32, tile_n=64):
        super().__init__(source, tile_m, tile_n)
        self.sparse_attention = SparseAttention(source.ratio)

    def custom_kernels(self):
        s = self.source
        return [*super().custom_kernels(), get_sparse_kernel(s.head_dim, s.heads // s.kv_heads, s.ratio)]

    def forward(self, x, key_cache, value_cache, raw_cache, pooled_cache, offset, pooled_count):
        if x.shape[1] == 1:
            return super().forward(x, key_cache, value_cache, raw_cache, pooled_cache, offset, pooled_count)
        s = self.source
        count = x.shape[1]
        positions = offset + torch.arange(count, dtype=torch.int32)
        end = offset + count
        query_gate = self.linear(x, "q_proj.weight").reshape(1, count, s.heads, s.head_dim * 2)
        query, gate = query_gate.split(s.head_dim, dim=-1)
        key = self.linear(x, "k_proj.weight").reshape(1, count, s.kv_heads, s.head_dim)
        value = self.linear(x, "v_proj.weight").reshape(1, count, s.kv_heads, s.head_dim).transpose(1, 2)
        query = s.rope(s.norm(query, "q_norm.weight").transpose(1, 2), positions)
        key = s.rope(s.norm(key, "k_norm.weight").transpose(1, 2), positions)
        indices = positions.long()[None, None, :, None].expand(1, s.kv_heads, count, s.head_dim)
        keys, values = key_cache.scatter(2, indices, key), value_cache.scatter(2, indices, value)
        index = self.linear(x, "indexer.index_qk_proj.weight")
        raw = index[..., s.idx_heads * s.idx_dim:]
        raw_indices = positions.long()[None, :, None].expand(1, count, s.idx_dim)
        raw_out = raw_cache.scatter(1, raw_indices, raw)
        full_blocks = torch.div(end, s.ratio, rounding_mode="floor")
        pooled_out = get_pool_kernel(s.ratio, s.rope_dim, s.eps)(
            raw_out, pooled_cache, s.indexer_k_layernorm_weight, s.cosine, s.sine, pooled_count, full_blocks,
            threads_per_grid=(s.blocks * 32, 1, 1), threads_per_thread_group=(32, 1, 1),
            result_shapes=[[1, s.blocks, s.idx_dim]])
        causal = s.cache_positions[None, :] <= positions[:, None]
        if s.blocks > s.topk:
            iq = s.norm(index[..., :s.idx_heads * s.idx_dim].reshape(1, count, s.idx_heads, s.idx_dim),
                        "indexer.q_layernorm.weight").transpose(1, 2)
            iq = s.rope(iq, positions)
            scores = torch.relu(iq.float() @ pooled_out[:, None].float().transpose(-1, -2)).sum(1)
            visible = s.block_positions[None, :] + s.ratio - 1 <= positions[:, None]
            scores = scores - (s.block_positions.float() / s.ratio) * 1e-7
            scores = torch.where(visible[None], scores, float("-inf"))
            chosen = scores.topk(s.topk, dim=-1, sorted=False).indices
            selected = torch.zeros_like(scores, dtype=torch.int32).scatter(-1, chosen, 1).bool() & visible[None]
            selected_tokens = selected.repeat_interleave(s.ratio, dim=-1)
            tail_start = torch.div(positions + 1, s.ratio, rounding_mode="floor") * s.ratio
            tail = s.cache_positions[None, :] >= tail_start[:, None]
            sparse = (selected_tokens | tail[None]) & causal[None]
            mask = torch.where(end > s.budget + s.ratio - 1, sparse, causal[None])[:, None]
            block_ids = chosen.to(torch.int32)
        else:
            mask = causal[None, None]
            block_ids = torch.arange(s.blocks, dtype=torch.int32)[None, None].expand(1, count, s.blocks)
        attention = self.sparse_attention(query, keys, values, block_ids, offset)
        gated = (attention.transpose(1, 2) * torch.sigmoid(gate)).half().reshape(1, count, s.heads * s.head_dim)
        y = self.linear(gated, "o_proj.weight")
        return y, keys, values, raw_out, pooled_out, end, full_blocks, mask.to(torch.int32)


def export_candidate(source, output, reference_sequence, *, count):
    """Reuse immutable reference fixtures; no large CPU candidate execution."""
    import coreai_torch
    from coreai_torch.composite_ops import SDPA
    sequence_path = Path(reference_sequence).resolve()
    sequence = json.loads(sequence_path.read_text())
    if sequence["stateBindings"] != BINDINGS or set(sequence["models"]) != {"s1", f"s{count}"}:
        raise ValueError("Reference must use the same state contract and selected prefill plusS1")
    if count <= 1 or count > source.capacity:
        raise ValueError("Invalid prefill shape")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model = QwenQSASparsePrefill(source).eval()
    probe = CompactProbe(model).eval()
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    kernels = model.custom_kernels()
    converter.register_custom_kernels(kernels)
    for length in (1, count):
        function = "main" if length == 1 else f"prefill_s{length}"
        values = {"x": torch.zeros(1, length, source.hidden, dtype=torch.float16), **initial_state(source)}
        args = tuple(values[name] for name in INPUT_NAMES)
        converter.add_pytorch_module(probe, entrypoint_name=function, input_names=INPUT_NAMES, output_names=PROBE_OUTPUTS,
            externalize_modules=[coreai_torch.ExternalizeSpec(target_class=SDPA,
                composite_op_name="scaled_dot_product_attention", composite_attrs=["scale", "is_causal", "window_size"])],
            export_fn=lambda m, args=args: torch.export.export(m, args=args).run_decompositions(coreai_torch.get_decomp_table()))
        sequence["models"][f"s{length}"] = {"path": "qsa-sparse.aimodel", "function": function}
    program = converter.to_coreai()
    program.optimize()
    program.save_asset(output / "qsa-sparse.aimodel")
    for function in ("main", f"prefill_s{count}"):
        (output / f"coreai-{function}-after.txt").write_text(str(program.get_graph(function)))
    for kernel in kernels:
        for identifier, code in kernel.kernel_cache.values():
            (output / (identifier + ".metal")).write_text(code)
    # Hardlinks keep fixtures self-contained without rewriting gigabytes.
    # These immutable JSON files are never modified by this exporter.
    references = []
    for relative in (sequence["initialState"], *(step["fixture"] for step in sequence["steps"])):
        old = (sequence_path.parent / relative).resolve()
        destination = output / Path(relative).name
        destination.hardlink_to(old)
        references.append({"path": destination.name, "source": str(old), "sha256": sha256_file(old)})
    sequence["initialState"] = Path(sequence["initialState"]).name
    for step in sequence["steps"]:
        step["fixture"] = Path(step["fixture"]).name
    write_json(output / "sequence.json", sequence)
    report = {"status": "experimental-full-layer-cpu-authored-device-unvalidated", "count": count,
        "capacity": source.capacity, "sequence": "sequence.json", "model": "qsa-sparse.aimodel",
        "modelBytes": sum(p.stat().st_size for p in (output / "qsa-sparse.aimodel").rglob("*") if p.is_file()),
        "referenceSequence": str(sequence_path), "referenceSequenceSHA256": sha256_file(sequence_path),
        "referenceFixtures": references, "candidateCPUExecuted": False,
        "prefill": "Original indexer top-k IDs feed direct sparse attention; state and diagnostic mask unchanged",
        "decode": "Inherited S1 source projections and FP32 SDPA; incremental pooled state retained",
        "authoringSources": [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in
            (Path(__file__), Path(__file__).with_name("coreai_qsa_sparse.py"), Path(__file__).with_name("coreai_qsa_chunk.py"))]}
    write_json(output / "manifest.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-sequence", type=Path, required=True)
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--capacity", type=int, default=16384)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    from check_coreai_qsa_chunks import WEIGHT_NAMES
    reader = Source(3)
    config_path = reader.directory / "config.json"
    config = json.loads(config_path.read_text())["text_config"]
    reference_manifest = json.loads((args.reference_sequence.parent / "manifest.json").read_text())
    if (reference_manifest["capacity"] != args.capacity or
        reference_manifest["provenance"]["configSHA256"] != sha256_file(config_path)):
        raise ValueError("Reference cache capacity/source config identity differ")
    reader.prefix = "language_model.model.layers.3.self_attn."
    source = QwenQSA(config, {name: reader.read(name) for name in WEIGHT_NAMES}, args.capacity).eval()
    report = export_candidate(source, args.output, args.reference_sequence, count=args.count)
    report["sourceRecords"] = reader.records
    report["configSHA256"] = sha256_file(config_path)
    write_json(args.output / "manifest.json", report)
    print(json.dumps({key: report[key] for key in ("status", "count", "capacity", "modelBytes", "sequence")}), flush=True)


if __name__ == "__main__":
    main()
