# /// script
# requires-python = ">=3.12"
# dependencies = ["mlx==0.32.2", "numpy>=2,<3"]
# ///
"""Prepare isolated GDN/QSA numerical references, never a full model.

Run only in an agreed GPU validation window:
  uv run --offline --no-project export_gpu_sequence_reference.py

Inputs replay the captured layer-0 MoE activation at DIFFERENT module boundaries.
They are not native attention activations or the original request's attention
states. QSA coverage repeats that activation to cross the budget, not a real
long-context conversation. Outputs come from an independent Python MLX program;
GDN recurrence follows the pinned author's Metal scalar kernel exactly.

Source: garnermccloud/mlx-serve transformer.zig, commit
7dbcba04c98e4fd3bcc533c63e645547f13cc3b1. MIT, Copyright (c) 2026 David Dalcu;
GDN recurrence originally mlx-lm, Copyright (c) 2023-2026 Apple Inc.
Full retained permission notice is in Sources/ANERunnerGPU/GPUGatedDeltaNet.swift.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import mlx.core as mx
import numpy as np
from benchmark_moe_mlx import TensorReader

SOURCE_COMMIT = "7dbcba04c98e4fd3bcc533c63e645547f13cc3b1"
GDN_SOURCE = r"""
auto n = thread_position_in_grid.z;
auto hv_idx = n % 48;
auto hk_idx = hv_idx / 3;
constexpr int n_per_t = 4;
auto q_ = q + hk_idx * 128;
auto k_ = k + hk_idx * 128;
auto v_ = v + hv_idx * 128;
y += hv_idx * 128;
auto dk_idx = thread_position_in_threadgroup.x;
auto dv_idx = thread_position_in_grid.y;
auto i_state = state_in + (n * 128 + dv_idx) * 128;
auto o_state = state_out + (n * 128 + dv_idx) * 128;
float state[n_per_t];
for (int i = 0; i < n_per_t; ++i) {
  state[i] = static_cast<float>(i_state[n_per_t * dk_idx + i]);
}
auto g_ = g;
auto beta_ = beta;
for (int t = 0; t < T; ++t) {
  float kv_mem = 0.0f;
  for (int i = 0; i < n_per_t; ++i) {
    auto s_idx = n_per_t * dk_idx + i;
    state[i] = state[i] * g_[hv_idx];
    kv_mem += state[i] * k_[s_idx];
  }
  kv_mem = simd_sum(kv_mem);
  auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];
  float out = 0.0f;
  for (int i = 0; i < n_per_t; ++i) {
    auto s_idx = n_per_t * dk_idx + i;
    state[i] = state[i] + k_[s_idx] * delta;
    out += state[i] * q_[s_idx];
  }
  out = simd_sum(out);
  if (thread_index_in_simdgroup == 0) {
    y[dv_idx] = static_cast<bfloat16_t>(out);
  }
  q_ += 16 * 128;
  k_ += 16 * 128;
  v_ += 48 * 128;
  y += 48 * 128;
  g_ += 48;
  beta_ += 48;
}
for (int i = 0; i < n_per_t; ++i) {
  o_state[n_per_t * dk_idx + i] = static_cast<bfloat16_t>(state[i]);
}
"""

class Weights:
    def __init__(self, path):
        self.path = path
        self.mapping = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
        self.reader, self.evidence = TensorReader(), []

    def load(self, name):
        host, evidence = self.reader.read(self.path / self.mapping[name], name)
        if evidence["dtype"] != "BF16":
            raise ValueError(f"Expected original BF16 projection: {name}")
        result = mx.array(host, dtype=mx.bfloat16)
        mx.eval(result)
        self.evidence.append(evidence)
        return result


def norm(x, weight):
    # This converted checkpoint already folds norm offsets into the weights.
    return mx.fast.rms_norm(x, weight, 1e-6)


class GDN:
    def __init__(self, source):
        p = "language_model.model.layers.0.linear_attn."
        self.w = {s: source.load(p + s) for s in (
            "in_proj_qkv.weight", "in_proj_z.weight", "in_proj_a.weight", "in_proj_b.weight",
            "out_proj.weight", "conv1d.weight", "A_log", "dt_bias", "norm.weight")}
        self.kernel = mx.fast.metal_kernel(
            name="sequence_reference_qwen38_gdn", input_names=["q", "k", "v", "g", "beta", "state_in", "T"],
            output_names=["y", "state_out"], source=GDN_SOURCE, ensure_row_contiguous=True)
        self.history, self.state = None, None

    def __call__(self, x):
        count = x.shape[1]
        w = self.w
        qkv = x @ w["in_proj_qkv.weight"].T
        z = (x @ w["in_proj_z.weight"].T).reshape(1, count, 48, 128)
        a, b = x @ w["in_proj_a.weight"].T, x @ w["in_proj_b.weight"].T
        old_history = self.history if self.history is not None else mx.zeros((1, 3, 10240), dtype=mx.bfloat16)
        conv_input = mx.concatenate([old_history, qkv], axis=1)
        history = conv_input[:, -3:, :]
        raw = mx.conv1d(conv_input, w["conv1d.weight"], groups=10240)
        conv = raw * mx.sigmoid(raw)
        ones = mx.ones((128,), dtype=mx.bfloat16)
        q = norm(conv[..., :2048].reshape(1, count, 16, 128), ones) * mx.array(1/128, dtype=mx.bfloat16)
        k = norm(conv[..., 2048:4096].reshape(1, count, 16, 128), ones) * mx.array(1/np.sqrt(128), dtype=mx.bfloat16)
        v = conv[..., 4096:].reshape(1, count, 48, 128)
        decay = mx.exp(-mx.exp(w["A_log"].astype(mx.float32)) *
                       mx.log1p(mx.exp((a + w["dt_bias"]).astype(mx.float32)))).astype(mx.bfloat16)
        beta = mx.sigmoid(b)
        old_state = self.state if self.state is not None else mx.zeros((1, 48, 128, 128), dtype=mx.bfloat16)
        y, state = self.kernel(inputs=[q, k, v, decay, beta, old_state, mx.array(count, dtype=mx.int32)],
                               output_shapes=[(1, count, 48, 128), (1, 48, 128, 128)],
                               output_dtypes=[mx.bfloat16, mx.bfloat16],
                               grid=(32, 128, 48), threadgroup=(32, 4, 1))
        result = (norm(y, w["norm.weight"]) * mx.sigmoid(z)).reshape(1, count, 6144) @ w["out_proj.weight"].T
        self.history, self.state = history, state
        return {"expected.output": result, "expected.state.convHistory": history,
                "expected.state.recurrent": state, "recurrence.q": q, "recurrence.k": k,
                "recurrence.v": v, "recurrence.decay": decay, "recurrence.beta": beta,
                "recurrence.stateIn": old_state, "recurrence.y": y, "recurrence.stateOut": state}


def pooled_rope(x, offset):
    dims = 64
    frequency = mx.exp(mx.arange(dims//2, dtype=mx.float32) * mx.array(-2*np.log(np.float32(10_000_000))/dims, dtype=mx.float32))
    positions = mx.arange(offset, offset+4*x.shape[2], 4, dtype=mx.float32)
    angles = positions[:, None] * frequency
    angles = mx.concatenate([angles, angles], axis=-1)
    cosine, sine = mx.cos(angles).astype(x.dtype), mx.sin(angles).astype(x.dtype)
    rotary, rest = x[..., :dims], x[..., dims:]
    rotation = mx.concatenate([-rotary[..., dims//2:], rotary[..., :dims//2]], axis=-1)
    return mx.concatenate([rotary*cosine + rotation*sine, rest], axis=-1)


class Attention:
    def __init__(self, source):
        p = "language_model.model.layers.3.self_attn."
        self.w = {s: source.load(p+s) for s in (
            "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight", "q_norm.weight", "k_norm.weight",
            "indexer.index_qk_proj.weight", "indexer.q_layernorm.weight", "indexer.k_layernorm.weight")}
        self.reset()

    def reset(self):
        self.keys = self.values = self.raw = self.pooled = None
        self.offset = 0

    def __call__(self, x):
        w, count, offset = self.w, x.shape[1], self.offset
        total = offset+count
        query, gate = mx.split((x @ w["q_proj.weight"].T).reshape(1, count, 24, 512), 2, axis=-1)
        key = (x @ w["k_proj.weight"].T).reshape(1, count, 2, 256)
        value = (x @ w["v_proj.weight"].T).reshape(1, count, 2, 256).transpose(0, 2, 1, 3)
        query = mx.fast.rope(norm(query, w["q_norm.weight"]).transpose(0, 2, 1, 3), dims=64, traditional=False, base=10_000_000, scale=1, offset=offset)
        key = mx.fast.rope(norm(key, w["k_norm.weight"]).transpose(0, 2, 1, 3), dims=64, traditional=False, base=10_000_000, scale=1, offset=offset)
        self.keys = key if self.keys is None else mx.concatenate([self.keys, key], axis=2)
        self.values = value if self.values is None else mx.concatenate([self.values, value], axis=2)
        projected = x @ w["indexer.index_qk_proj.weight"].T
        raw = projected[..., 512:]
        self.raw = raw if self.raw is None else mx.concatenate([self.raw, raw], axis=1)
        query_positions = mx.arange(offset, total, dtype=mx.int32)[:, None]
        key_positions = mx.arange(total, dtype=mx.int32)[None, :]
        causal = key_positions <= query_positions
        mask = causal[None, None, :, :] if count > 1 else None
        evidence = {}
        if total > 2051:
            blocks = total//4
            iq = norm(projected[..., :512].reshape(1, count, 4, 128), w["indexer.q_layernorm.weight"]).transpose(0, 2, 1, 3)
            iq = mx.fast.rope(iq, dims=64, traditional=False, base=10_000_000, scale=1, offset=offset)
            cached = 0 if self.pooled is None else self.pooled.shape[1]
            if cached < blocks:
                flat = self.raw[:, cached*4:blocks*4, :].reshape(1, blocks-cached, 4, 128)
                average = flat.astype(mx.float32).mean(axis=2).astype(mx.bfloat16)
                pk = norm(average, w["indexer.k_layernorm.weight"])
                pk = pooled_rope(pk[:, None, :, :], cached*4).reshape(1, blocks-cached, 128)
                self.pooled = pk if self.pooled is None else mx.concatenate([self.pooled, pk], axis=1)
            ik = self.pooled[:, None, :, :].swapaxes(-1, -2)
            scores = mx.maximum(iq.astype(mx.float32) @ ik.astype(mx.float32), 0).sum(axis=1)
            visible = (mx.arange(3, blocks*4, 4, dtype=mx.int32)[None, :] <= query_positions)[None, :, :]
            scores = mx.where(visible, scores - mx.arange(blocks, dtype=mx.float32)*mx.array(1e-7), -mx.inf)
            selected = mx.argpartition(scores, kth=blocks-512, axis=-1)[..., -512:]
            chosen = mx.put_along_axis(mx.zeros((1, count, blocks), dtype=mx.bool_), selected,
                                       mx.array(True), axis=-1) & visible
            token_mask = mx.repeat(chosen, 4, axis=-1)
            if total % 4:
                token_mask = mx.concatenate([token_mask, mx.zeros((1, count, total%4), dtype=mx.bool_)], axis=-1)
            tail = key_positions >= ((query_positions+1)//4)*4
            mask = ((token_mask | tail) & causal)[:, None, :, :]
            evidence["reference.qsa_mask"] = mask
        y = mx.fast.scaled_dot_product_attention(query, self.keys, self.values, scale=1/16, mask=mask)
        result = (y.transpose(0, 2, 1, 3)*mx.sigmoid(gate)).reshape(1, count, 6144) @ w["o_proj.weight"].T
        self.offset = total
        evidence.update({"expected.output": result, "expected.state.keys": self.keys,
                         "expected.state.values": self.values, "expected.state.rawIndexerKeys": self.raw})
        if self.pooled is not None:
            evidence["expected.state.pooledIndexerKeys"] = self.pooled
        return evidence


def sha(path):
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=root.parent/"qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream")
    parser.add_argument("--capture-dir", type=Path, default=root/"fixtures/moe-real")
    parser.add_argument("--output-dir", type=Path, default=root/"fixtures/gpu-sequence-reference")
    parser.add_argument("--skip-qsa", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir/"manifest.json"
    if manifest.exists():
        raise FileExistsError(f"Refusing to overwrite completed reference {manifest}")
    reader = TensorReader()
    host, pre_evidence = reader.read(args.capture_dir/"prefill.safetensors", "x")
    decode_host, dec_evidence = reader.read(args.capture_dir/"decode.safetensors", "x")
    prefill = mx.array(host, dtype=mx.bfloat16)
    decode = mx.array(decode_host, dtype=mx.bfloat16)
    source = Weights(args.model_dir)
    cases = []

    def write_case(name, kind, layer, module, chunks):
        case = {"id": name, "kind": kind, "layer": layer, "steps": []}
        offset = 0
        for number, (phase, x) in enumerate(chunks):
            values = module(x)
            mx.eval(list(values.values()))
            values["input"] = x
            path = args.output_dir/f"{name}-{number}-{phase}.safetensors"
            if path.exists():
                raise FileExistsError(path)
            mx.save_safetensors(str(path), values, metadata={
                "origin": "actual layer0 MoE activation replay at another boundary; not native attention capture",
                "reference": "independent Python MLX; author scalar Metal GDN recurrence",
                "source_commit": SOURCE_COMMIT})
            count = x.shape[1]
            case["steps"].append({"file": path.name, "sha256": sha(path), "phase": phase,
                                   "offset_before": offset, "offset_after": offset+count,
                                   "qsa_expected": kind == "attention" and offset+count > 2051})
            offset += count
            print(f"Saved {path.name} ({path.stat().st_size} bytes)", flush=True)
        cases.append(case)

    write_case("gdn-continuous", "gdn", 0, GDN(source), [("prefill", prefill), ("decode", decode)])
    attention = Attention(source)
    write_case("attention-continuous", "attention", 3, attention, [("prefill", prefill), ("decode", decode)])
    if not args.skip_qsa:
        attention.reset()
        replay = mx.tile(prefill, (1, (2051+prefill.shape[1]-1)//prefill.shape[1], 1))[:, :2051, :]
        write_case("qsa-threshold", "attention", 3, attention,
                   [("prefill", replay), ("decode", decode), ("decode", prefill[:, :1, :])])
    report = {"schema": "qwen38-gpu-sequence-reference-v1", "source_commit": SOURCE_COMMIT,
              "reference_script_sha256": sha(Path(__file__)), "mlx_version": importlib.metadata.version("mlx"),
              "input_provenance": [pre_evidence, dec_evidence], "weight_provenance": source.evidence,
              "notes": ["MoE activation replay, not native attention-layer input/state capture.",
                        "QSA long input repeats the capture and is only a mask/state coverage test.",
                        "No performance benchmark or whole-model correctness claim.",
                        "GDN uses BF16 inter-call state and FP32 intra-chunk accumulation like the pinned runtime."],
              "cases": cases}
    manifest.write_text(json.dumps(report, indent=2)+"\n")
    print(manifest, flush=True)


if __name__ == "__main__":
    main()
