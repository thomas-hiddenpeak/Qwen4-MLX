#!/usr/bin/env python3
"""Independent large-chunk QSA candidate with TensorOps GEMM and incremental pool.

The attention, cache, routing, and FP16 boundaries match QwenQSA. Chunked prefill
uses five TensorOps linear helpers; S1 retains the source linear implementation.
Both phases use the custom incremental pooled-cache update.
SDPA still receives full-capacity K/V and an explicit mask: this candidate
does not yet claim that sparse selection reduces attention device arithmetic.
CPU authoring only; no CoreAI/MLX/Core ML inference or accelerator API is used.
"""
from __future__ import annotations

import argparse
from functools import cache
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from coreai_tensor_matmul import get_tensor_kernel, tensor_linear
from export_coreai_qsa import (BINDINGS, INPUT_NAMES, OUTPUT_NAMES, QwenQSA,
                               FIXTURES, compare, initial_state, read_fixture)
from export_moe import Source, sha256_file, write_json


POOL_METAL_BODY = r"""
// One SIMD group owns one block. Input tensors are immutable; every output
// element is initialized even when no new block completes in this call.
const uint block = group.x;
const uint width = raw.get_extent(0);
if (block >= pooled.get_extent(1)) return;
const int begin = pooled_count[0];
const int end = full_blocks[0];
if (int(block) < begin || int(block) >= end) {
    for (uint d = lane; d < width; d += 32u) {
        output[d, block, 0] = pooled[d, block, 0];
    }
    return;
}
const uint start = block * RATIO;
float local_squares = 0.0f;
for (uint d = lane; d < width; d += 32u) {
    float sum = 0.0f;
    for (uint row = 0u; row < RATIO; ++row) sum += float(raw[d, start + row, 0]);
    const half mean = half(sum / float(RATIO));
    local_squares += float(mean) * float(mean);
}
const float inverse = rsqrt(simd_sum(local_squares) / float(width) + EPSILON);
for (uint d = lane; d < width; d += 32u) {
    float sum = 0.0f;
    for (uint row = 0u; row < RATIO; ++row) sum += float(raw[d, start + row, 0]);
    const half mean = half(sum / float(RATIO));
    const half value = half(float(mean) * inverse * float(norm[d]));
    if (d >= 2u * ROPE_HALF) {
        output[d, block, 0] = value;
    } else {
        const uint other_d = d < ROPE_HALF ? d + ROPE_HALF : d - ROPE_HALF;
        float other_sum = 0.0f;
        for (uint row = 0u; row < RATIO; ++row) other_sum += float(raw[other_d, start + row, 0]);
        const half other_mean = half(other_sum / float(RATIO));
        const half other = half(float(other_mean) * inverse * float(norm[other_d]));
        const half c = half(cosine[d % ROPE_HALF, start]);
        const half s = half(sine[d % ROPE_HALF, start]);
        // Keep the source's two rounded half products before the half sum.
        const half left = half(float(value) * float(c));
        const half right = half(float(other) * float(s));
        output[d, block, 0] = d < ROPE_HALF
            ? half(float(left) - float(right)) : half(float(left) + float(right));
    }
}
"""


def pool_reference(raw, pooled, norm, cosine, sine, pooled_count, full_blocks,
                   *, ratio, rope_dim, epsilon):
    """Exact eager oracle, using the established full-pool arithmetic.

    Eager/fake inference deliberately computes a full candidate and masks it;
    only the Metal body conditionally computes new blocks. This avoids turning
    runtime tensor counters into Python shape integers while exporting.
    """
    capacity, width = raw.shape[1:]
    means = raw.reshape(1, capacity // ratio, ratio, width).float().mean(2).half()
    xx = means.float()
    normalized = (xx * torch.rsqrt(xx.square().mean(-1, keepdim=True) + epsilon) * norm.float()).half()
    positions = torch.arange(capacity // ratio, dtype=torch.int32) * ratio
    cos = cosine.index_select(0, positions.long()).half()[None]
    sin = sine.index_select(0, positions.long()).half()[None]
    half = rope_dim // 2
    a, b = normalized[..., :half], normalized[..., half:rope_dim]
    rotated = torch.cat(((a * cos).half() - (b * sin).half(),
                         (b * cos).half() + (a * sin).half(), normalized[..., rope_dim:]), -1)
    blocks = positions // ratio
    update = ((blocks >= pooled_count) & (blocks < full_blocks))[None, :, None]
    return torch.where(update, rotated, pooled)


@cache
def get_pool_kernel(ratio=4, rope_dim=64, epsilon=1e-6):
    from coreai.authoring import MetalParameter
    from coreai_torch import TorchMetalKernel
    if ratio < 1 or rope_dim < 2 or rope_dim % 2 or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Expected positive ratio/epsilon and positive even rotary dimension")

    def reference(raw: torch.Tensor, pooled: torch.Tensor, norm: torch.Tensor,
                  cosine: torch.Tensor, sine: torch.Tensor, pooled_count: torch.Tensor,
                  full_blocks: torch.Tensor) -> torch.Tensor:
        return pool_reference(raw, pooled, norm, cosine, sine, pooled_count, full_blocks,
                              ratio=ratio, rope_dim=rope_dim, epsilon=epsilon)

    epsilon_id = hashlib.sha256(repr(float(epsilon)).encode()).hexdigest()[:8]
    body = POOL_METAL_BODY.replace("RATIO", str(ratio) + "u")
    body = body.replace("ROPE_HALF", str(rope_dim // 2) + "u").replace("EPSILON", f"{epsilon:.9g}f")
    return TorchMetalKernel(
        f"qwen_qsa_incremental_pool_r{ratio}_p{rope_dim}_e{epsilon_id}_v1",
        input_names=["raw", "pooled", "norm", "cosine", "sine", "pooled_count", "full_blocks"],
        result_names=["output"], src=body, torch_defn=reference,
        metal_params=[MetalParameter("group", "uint3", "threadgroup_position_in_grid"),
                      MetalParameter("lane", "uint", "thread_index_in_simdgroup")],
    )


class QwenQSAChunk(torch.nn.Module):
    """Wrap one QwenQSA without copying weights or changing six-state ownership.

    Register custom_kernels() before conversion. S1 and any positive static
    chunk use the same weights/state contract, including lagging pooled_count.
    Callers retain the existing 0<=offset<=capacity-count validation.
    """
    def __init__(self, source: QwenQSA, tile_m=32, tile_n=64, *, prefill_sdpa_fp16=False):
        super().__init__()
        if source.q_proj_weight.dtype != torch.float16:
            raise ValueError("This candidate expects the existing FP16 QSA weights")
        get_tensor_kernel(tile_m, tile_n)
        self.source = source
        self.tile_m, self.tile_n = tile_m, tile_n
        self.prefill_sdpa_fp16 = prefill_sdpa_fp16
        for name in ("capacity", "hidden", "head_dim", "kv_heads", "idx_dim", "blocks", "ratio", "budget"):
            setattr(self, name, getattr(source, name))

    def custom_kernels(self):
        s = self.source
        return [get_tensor_kernel(self.tile_m, self.tile_n), get_pool_kernel(s.ratio, s.rope_dim, s.eps)]

    def linear(self, x, name):
        # Chunk TensorOps are a prefill optimization. Retain the established
        # decode projection implementation, as the other P/D layers do.
        if x.shape[1] == 1:
            return self.source.linear(x, name)
        return tensor_linear(x, getattr(self.source, name.replace(".", "_")), self.tile_m, self.tile_n)

    def forward(self, x, key_cache, value_cache, raw_cache, pooled_cache, offset, pooled_count):
        # Keep this arithmetic in lockstep with QwenQSA.forward; the changed
        # boundaries are linear() above and the opaque incremental pool below.
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
        kv_indices = positions.long()[None, None, :, None].expand(1, s.kv_heads, count, s.head_dim)
        keys = key_cache.scatter(2, kv_indices, key)
        values = value_cache.scatter(2, kv_indices, value)
        index = self.linear(x, "indexer.index_qk_proj.weight")
        raw = index[..., s.idx_heads * s.idx_dim:]
        raw_indices = positions.long()[None, :, None].expand(1, count, s.idx_dim)
        raw_out = raw_cache.scatter(1, raw_indices, raw)
        full_blocks = torch.div(end, s.ratio, rounding_mode="floor")
        pooled_out = get_pool_kernel(s.ratio, s.rope_dim, s.eps)(
            raw_out, pooled_cache, s.indexer_k_layernorm_weight, s.cosine, s.sine, pooled_count, full_blocks,
            threads_per_grid=(s.blocks * 32, 1, 1), threads_per_thread_group=(32, 1, 1),
            result_shapes=[[1, s.blocks, s.idx_dim]],
        )
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
        else:
            mask = causal[None, None]
        if self.prefill_sdpa_fp16 and count > 1:
            attention = s.sdpa(query, keys, values, attn_mask=mask).half()
        else:
            attention = s.sdpa(query.float(), keys.float(), values.float(), attn_mask=mask).half()
        gated = (attention.transpose(1, 2) * torch.sigmoid(gate)).half().reshape(1, count, s.heads * s.head_dim)
        y = self.linear(gated, "o_proj.weight")
        return y, keys, values, raw_out, pooled_out, end, full_blocks, mask.to(torch.int32)


class CompactProbe(torch.nn.Module):
    """Same neural/state math, with per-query visibility counts instead of a huge mask output."""
    def __init__(self, source):
        super().__init__()
        self.source = source

    def forward(self, *args):
        result = self.source(*args)
        return (*result[:7], result[-1].sum(-1).reshape(-1).to(torch.int32))


PROBE_OUTPUTS = (*OUTPUT_NAMES[:7], "visible_counts")


def make_tiny(capacity=16384, budget=2048, seed=618):
    config = {"hidden_size": 8, "num_attention_heads": 2, "num_key_value_heads": 1,
              "head_dim": 8, "indexer_n_heads": 2, "indexer_head_dim": 8,
              "indexer_compress_ratio": 4, "indexer_budget": budget, "partial_rotary_factor": 0.5,
              "rms_norm_eps": 1e-6, "rope_parameters": {"rope_theta": 10000000}}
    rng = np.random.default_rng(seed)
    shapes = {"q_proj.weight": (32, 8), "k_proj.weight": (8, 8), "v_proj.weight": (8, 8),
              "o_proj.weight": (8, 16), "indexer.index_qk_proj.weight": (24, 8)}
    weights = {name: (rng.standard_normal(shape) * 0.1).astype(np.float32) for name, shape in shapes.items()}
    for name in ("q_norm.weight", "k_norm.weight", "indexer.q_layernorm.weight", "indexer.k_layernorm.weight"):
        weights[name] = rng.uniform(0.8, 1.2, 8).astype(np.float32)
    return QwenQSA(config, weights, capacity).eval()


def seeded_state(source, offset, *, seed=619, pooled_count=None):
    """Deterministic nonzero synthetic state; zero-length state remains zero."""
    if not 0 <= offset <= source.capacity:
        raise ValueError("Seed offset exceeds capacity")
    state = initial_state(source)
    if offset == 0:
        return state
    generator = torch.Generator().manual_seed(seed)
    for name in ("key_cache", "value_cache", "raw_cache", "pooled_cache"):
        state[name] = (torch.randn(state[name].shape, generator=generator) * 0.1).half()
    state["offset"][:] = offset
    completed = offset // source.ratio if pooled_count is None else pooled_count
    if not 0 <= completed <= offset // source.ratio:
        raise ValueError("Invalid seeded pooled_count")
    state["pooled_cache"] = pool_reference(state["raw_cache"], state["pooled_cache"],
        source.indexer_k_layernorm_weight, source.cosine, source.sine, state["pooled_count"],
        torch.tensor([completed], dtype=torch.int32), ratio=source.ratio, rope_dim=source.rope_dim, epsilon=source.eps)
    state["pooled_count"][:] = completed
    return state


def repeat_activations(rows, count, offset=0):
    """Replay capture rows at absolute positions, without calling a model."""
    if rows.ndim != 3 or rows.shape[0] != 1 or rows.shape[1] == 0:
        raise ValueError("Expected nonempty [1, sequence, hidden] capture")
    indices = torch.arange(offset, offset + count) % rows.shape[1]
    return rows.index_select(1, indices).half().contiguous()


def replay_state(source, rows, offset, projection_chunk=128):
    """Build real layer-local state from captured inputs, without attention.

    Future K/V/index state depends only on that layer's input activations, not
    its attention outputs. Computing these projections in chunks therefore
    prepares valid state without quadratically attending an 8192-token prefix.
    These are replayed inputs, not a complete model's autoregressive trajectory.
    """
    if not 0 <= offset <= source.capacity or projection_chunk < 1 or rows.shape[-1] != source.hidden:
        raise ValueError("Invalid replay prefix or input width")
    state = initial_state(source)
    with torch.inference_mode():
        for start in range(0, offset, projection_chunk):
            count = min(projection_chunk, offset - start)
            x = repeat_activations(rows, count, start)
            positions = torch.arange(start, start + count, dtype=torch.int32)
            key = source.linear(x, "k_proj.weight").reshape(1, count, source.kv_heads, source.head_dim)
            key = source.rope(source.norm(key, "k_norm.weight").transpose(1, 2), positions)
            value = source.linear(x, "v_proj.weight").reshape(1, count, source.kv_heads, source.head_dim).transpose(1, 2)
            index = source.linear(x, "indexer.index_qk_proj.weight")
            state["key_cache"][:, :, start:start + count] = key
            state["value_cache"][:, :, start:start + count] = value
            state["raw_cache"][:, start:start + count] = index[..., source.idx_heads * source.idx_dim:]
        full_blocks = torch.tensor([offset // source.ratio], dtype=torch.int32)
        state["pooled_cache"] = pool_reference(state["raw_cache"], state["pooled_cache"],
            source.indexer_k_layernorm_weight, source.cosine, source.sine, state["pooled_count"],
            full_blocks, ratio=source.ratio, rope_dim=source.rope_dim, epsilon=source.eps)
        state["offset"][:] = offset
        state["pooled_count"][:] = full_blocks
    return state


def write_tensor_json(path, tensors, *, inputs=None):
    """Stream large CoreMLTensor JSON, bounding Python float-list allocation.

    ``inputs=None`` writes a bare state map; otherwise writes a normal fixture.
    Values retain the existing JSON schema and full FP16-representable precision.
    """
    def tensor_map(stream, values):
        stream.write("{")
        for number, (name, tensor) in enumerate(values.items()):
            if number:
                stream.write(",")
            array = tensor.detach().cpu().numpy()
            stream.write(json.dumps(name) + ":{\"shape\":" + json.dumps(list(array.shape)))
            stream.write(",\"dtype\":" + json.dumps(str(array.dtype)) + ",\"values\":[")
            flat = array.reshape(-1)
            for start in range(0, flat.size, 16384):
                if start:
                    stream.write(",")
                encoded = json.dumps(flat[start:start + 16384].astype(float).tolist(),
                                     separators=(",", ":"), allow_nan=False)
                stream.write(encoded[1:-1])
            stream.write("]}")
        stream.write("}")
    with Path(path).open("w") as stream:
        if inputs is not None:
            stream.write('{"inputs":')
            tensor_map(stream, inputs)
            stream.write(',"expectedOutputs":')
        tensor_map(stream, tensors)
        if inputs is not None:
            stream.write("}")
        stream.write("\n")


def export_probe(source, output, *, chunks=(128, 256), tile_m=32, tile_n=64, provenance=None,
                 offsets=None, activation_rows=None, prefill_sdpa_fp16=False, decode_steps=4,
                 verify_candidate_cpu=True):
    """Author one archive plus replayable state sequences, entirely on CPU."""
    import coreai_torch
    from coreai_torch.composite_ops import SDPA
    chunks = tuple(chunks)
    if not chunks or len(set(chunks)) != len(chunks) or any(c not in (128, 256, 512, 1024, 2048) for c in chunks):
        raise ValueError("Choose unique S128/S256/S512/S1024/S2048 chunks")
    if decode_steps not in (1, 4):
        raise ValueError("Choose one or four continuation steps")
    if source.capacity < max(chunks) + decode_steps:
        raise ValueError("Capacity must fit the chunk and decode steps")
    if offsets is not None and (not offsets or any(o < 0 or o + max(chunks) + decode_steps > source.capacity for o in offsets)):
        raise ValueError("Every offset must fit every chunk plus decode steps")
    if activation_rows is not None and activation_rows.shape[-1] != source.hidden:
        raise ValueError("Capture width differs from source hidden width")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    candidate = QwenQSAChunk(source, tile_m, tile_n, prefill_sdpa_fp16=prefill_sdpa_fp16).eval()
    probe = CompactProbe(candidate).eval()
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    kernels = candidate.custom_kernels()
    converter.register_custom_kernels(kernels)
    models, shapes = {}, {}
    for count in (1, *chunks):
        function = "main" if count == 1 else f"prefill_s{count}"
        example = {"x": torch.zeros(1, count, source.hidden, dtype=torch.float16), **initial_state(source)}
        args = tuple(example[name] for name in INPUT_NAMES)
        graph = torch.export.export(probe, args)
        (output / f"torch-{function}.txt").write_text(str(graph.graph))
        targets = [str(node.target) for node in graph.graph.nodes if node.op == "call_function"]
        converter.add_pytorch_module(probe, entrypoint_name=function, input_names=INPUT_NAMES, output_names=PROBE_OUTPUTS,
            externalize_modules=[coreai_torch.ExternalizeSpec(target_class=SDPA,
                composite_op_name="scaled_dot_product_attention", composite_attrs=["scale", "is_causal", "window_size"])],
            export_fn=lambda m, args=args: torch.export.export(m, args=args).run_decompositions(coreai_torch.get_decomp_table()))
        models[f"s{count}"] = {"path": "qsa-chunk.aimodel", "function": function}
        shapes[function] = {"queries": [1, source.heads, count, source.head_dim],
            "keysAndValues": [1, source.kv_heads, source.capacity, source.head_dim],
            "mask": [1, 1, count, source.capacity],
            "operandDType": "float16" if prefill_sdpa_fp16 and count > 1 else "float32",
            "tensorProjectionCalls": sum("qwen_mpp_fp16_gemm_" in op for op in targets),
            "poolCalls": sum("qwen_qsa_incremental_pool_" in op for op in targets)}
    program = converter.to_coreai()
    for function in shapes:
        (output / f"coreai-{function}-before.txt").write_text(str(program.get_graph(function)))
    program.optimize()
    for function in shapes:
        (output / f"coreai-{function}-after.txt").write_text(str(program.get_graph(function)))
    asset_path = output / "qsa-chunk.aimodel"
    program.save_asset(asset_path)
    metal_files = []
    for kernel in kernels:
        for kernel_id, body in kernel.kernel_cache.values():
            path = output / (kernel_id + ".metal")
            path.write_text(body)
            metal_files.append(path.name)
    cases = [(f"cold-s{chunks[0]}", chunks[0], 0, None),
             (f"near-end-s{chunks[-1]}", chunks[-1], source.capacity - chunks[-1] - decode_steps, None)]
    if source.capacity >= 2051 + chunks[0] + decode_steps:
        cases.insert(1, (f"sparse-s{chunks[0]}", chunks[0], 2051, 512))
    if offsets is not None:
        cases = [(f"offset{offset}-s{chunk}", chunk, offset, None) for offset in offsets for chunk in chunks]
    sequences, cpu_checks = [], []
    generator = torch.Generator().manual_seed(620)
    replayed_states = {}
    for name, chunk, offset, pool_count in cases:
        print(f"Preparing CPU fixture {name}", flush=True)
        if activation_rows is None:
            state = seeded_state(source, offset, pooled_count=pool_count)
        else:
            if offset not in replayed_states:
                replayed_states[offset] = replay_state(source, activation_rows, offset)
            state = {key: value.clone() for key, value in replayed_states[offset].items()}
        cpu_state = {key: value.clone() for key, value in state.items()}
        state_file = name + "-initial-state.json"
        write_tensor_json(output / state_file, state)
        tokens = ((torch.randn(1, chunk + decode_steps, source.hidden, generator=generator) * 0.2).half()
                  if activation_rows is None else repeat_activations(activation_rows, chunk + decode_steps, offset))
        steps, position = [], 0
        for number, count in enumerate((chunk, *((1,) * decode_steps))):
            x = tokens[:, position:position + count].contiguous()
            values = {"x": x, **state}
            baseline_values = {"x": x, **cpu_state}
            with torch.inference_mode():
                actual = candidate(*(values[key] for key in INPUT_NAMES)) if verify_candidate_cpu else None
                expected = source(*(baseline_values[key] for key in INPUT_NAMES))
            checks = None
            if actual is not None:
                checks = {key: compare(a, b) for key, a, b in zip(OUTPUT_NAMES, actual, expected)}
                valid = all(item["exact"] for key, item in checks.items() if key != "y")
                y = checks["y"]
                if prefill_sdpa_fp16 and count > 1:
                    valid = valid and y["finite"] and y["max_abs"] <= 0.02 and (y["relative_l2"] or 0.0) <= 0.005
                else:
                    valid = valid and y["exact"]
                if not valid:
                    raise AssertionError(f"CPU source comparison failed in {name}/{number}: {checks}")
            reference_finite = all(bool(torch.isfinite(value).all()) for value in expected)
            if not reference_finite:
                raise AssertionError(f"Nonfinite reference outputs in {name}/{number}")
            cpu_checks.append({"case": name, "step": number, "candidateCPUExecuted": actual is not None,
                               "referenceOutputsFinite": reference_finite, "checks": checks})
            compact = (*expected[:7], expected[-1].sum(-1).reshape(-1).to(torch.int32))
            fixture = f"{name}-{number}.json"
            write_tensor_json(output / fixture, dict(zip(PROBE_OUTPUTS, compact)), inputs={"x": x})
            steps.append({"name": f"{name}-{number}", "phase": "prefill" if count > 1 else "decode",
                          "model": f"s{count}", "fixture": fixture})
            reference_or_actual = expected if actual is None else actual
            state = {key: reference_or_actual[OUTPUT_NAMES.index(out)] for key, out in BINDINGS.items()}
            cpu_state = {key: expected[OUTPUT_NAMES.index(out)] for key, out in BINDINGS.items()}
            position += count
        sequence = name + "-sequence.json"
        used_models = {step["model"]: models[step["model"]] for step in steps}
        write_json(output / sequence, {"version": 1, "models": used_models, "stateBindings": BINDINGS,
            "initialState": state_file, "steps": steps,
            "tolerances": {"maximumAbsoluteError": 0.02, "relativeL2Error": 0.005}})
        sequences.append(sequence)
    write_json(output / "cpu-checks.json", cpu_checks)
    files = [{"path": str(path.relative_to(asset_path)), "bytes": path.stat().st_size,
              "sha256": sha256_file(path)} for path in sorted(asset_path.rglob("*")) if path.is_file()]
    report = {"version": 1, "status": "cpu-authored-device-unvalidated", "capacity": source.capacity,
        "chunks": list(chunks), "models": models, "stateBindings": BINDINGS, "sequences": sequences,
        "prefillSDPAFP16": prefill_sdpa_fp16, "decodeSDPADType": "float32", "decodeSteps": decode_steps,
        "candidateCPUExecuted": verify_candidate_cpu,
        "cpuAcceptance": {"yMaximumAbsoluteError": 0.02 if prefill_sdpa_fp16 else 0,
                          "yRelativeL2Error": 0.005 if prefill_sdpa_fp16 else 0,
                          "allStatesAndMask": "exact", "decodeY": "exact"},
        "fixtureInputs": "repeated captured rows at absolute positions" if activation_rows is not None else "seeded random",
        "initialStateMethod": ("K/V/index projections + K norm/RoPE + pool of repeated captured inputs; "
                               "128-token projection chunks; future tail zero") if activation_rows is not None else "seeded random",
        "provenance": provenance or "Tiny synthetic weights and nonzero synthetic states; not real model performance",
        "outputNames": list(PROBE_OUTPUTS), "sdpaShapeAudit": shapes, "sourceFiles": metal_files,
        "modelBytes": sum(row["bytes"] for row in files), "files": files,
        "authoringSeconds": time.perf_counter() - started,
        "limitations": ["No device execution or Metal compilation here; both custom kernels need device validation.",
            "Chunk projections call TensorOps; S1 retains source projections. CPU oracle uses FP32 BLAS then FP16.",
            "Pool Metal computes only newly committed blocks; its eager reference intentionally computes full pooled candidates.",
            "Full-capacity SDPA, sparse score/top-k, and functional scatter remain. FP16 SDPA is an optional prefill candidate; S1 stays FP32.",
            "Reference-only authoring uses original QwenQSA outputs and states; it does not establish candidate numerical parity.",
            "Visible counts are compact diagnostics, not a substitute for exact CPU mask or device output comparison.",
            "IR SDPA shape/call audit cannot show actual hardware efficiency or sparse-skip behavior."],
        "authoringSources": [{"path": str(path), "sha256": sha256_file(path)} for path in
            (Path(__file__), Path(__file__).with_name("export_coreai_qsa.py"), Path(__file__).with_name("coreai_tensor_matmul.py"))]}
    write_json(output / "manifest.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=16384)
    parser.add_argument("--chunks", type=int, nargs="+", choices=(128, 256, 512, 1024, 2048), default=[128, 256])
    parser.add_argument("--prefill-sdpa-fp16", action="store_true", help="Candidate FP16 SDPA for chunks only; S1 remains FP32")
    parser.add_argument("--decode-steps", type=int, choices=(1, 4), default=4)
    parser.add_argument("--reference-only", action="store_true", help="Author original FP32-reference fixtures without eager candidate execution")
    parser.add_argument("--offsets", type=int, nargs="+", help="Test every chosen chunk at these starting offsets")
    parser.add_argument("--capture-replay", action="store_true", help="Repeat verified captured activation rows; requires --real-layer")
    parser.add_argument("--real-layer", type=int, choices=(3,), help="Read only real layer3; omitted means tiny synthetic weights")
    args = parser.parse_args()
    if args.capture_replay and args.real_layer is None:
        parser.error("--capture-replay requires --real-layer")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    provenance = None
    if args.real_layer is None:
        source = make_tiny(args.capacity)
    else:
        from check_coreai_qsa_chunks import WEIGHT_NAMES
        reader = Source(args.real_layer)
        config_path = reader.directory / "config.json"
        config = json.loads(config_path.read_text())["text_config"]
        reader.prefix = f"language_model.model.layers.{args.real_layer}.self_attn."
        source = QwenQSA(config, {name: reader.read(name) for name in WEIGHT_NAMES}, args.capacity).eval()
        provenance = {"layer": args.real_layer, "modelDirectory": str(reader.directory),
                      "configSHA256": sha256_file(config_path), "sourceRecords": reader.records,
                      "activations": "Synthetic; not a full-model capture"}
    activation_rows = None
    if args.capture_replay:
        fixture_manifest = json.loads((FIXTURES / "manifest.json").read_text())
        captured, evidence = read_fixture(FIXTURES / "attention-continuous-0-prefill.safetensors", fixture_manifest)
        activation_rows = captured["input"].half()
        provenance["activations"] = "Verified layer0 MoE boundary input replayed at layer3 attention; NOT native layer3 full-model capture"
        provenance["inputCapture"] = evidence
        provenance["fixtureManifestSHA256"] = sha256_file(FIXTURES / "manifest.json")
    report = export_probe(source, args.output, chunks=tuple(args.chunks), provenance=provenance,
                          offsets=args.offsets, activation_rows=activation_rows,
                          prefill_sdpa_fp16=args.prefill_sdpa_fp16, decode_steps=args.decode_steps,
                          verify_candidate_cpu=not args.reference_only)
    print(json.dumps({key: report[key] for key in ("status", "modelBytes", "sequences", "authoringSeconds")}), flush=True)


if __name__ == "__main__":
    main()
