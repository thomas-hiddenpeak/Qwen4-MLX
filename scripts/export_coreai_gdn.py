#!/usr/bin/env python3
"""Export a real Qwen3.8 GDN sublayer and a continuing CPU reference sequence.

CPU authoring only. This never runs MLX, Core ML, CoreAI inference or a server.
The input is captured MoE activation replay at the GDN boundary, not a native
GDN activation capture. Only one layer's weights are read. Projection and
convolution accumulation and persistent recurrence use FP32; weight storage,
activation boundaries and convolution history use FP16. This intentionally
does not reproduce the existing BF16 inter-call state rounding.

Equations follow GPUGatedDeltaNet.swift and the pinned MIT-licensed original
garnermccloud/mlx-serve transformer.zig (7dbcba04c98e4fd3bcc533c63e645547f13cc3b1).
The retained full license is in Sources/ANERunnerGPU/GPUGatedDeltaNet.swift.
Unlike the generic Qwen3.5 primitive, this checkpoint needs a sigmoid output
gate, folded norm weights, RMS epsilon semantics and contiguous key-head
repetition. Explicit state I/O avoids hidden state-reset behavior.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from export_moe import Source, sha256_file, write_json
from export_coreai_moe import errors, tensor_json

ROOT = Path(__file__).resolve().parents[1]
INPUT_NAMES = ("hidden", "conv_history", "recurrent_state")
OUTPUT_NAMES = ("output", "next_conv_history", "next_recurrent_state")
STATE_BINDINGS = {"conv_history": "next_conv_history", "recurrent_state": "next_recurrent_state"}


@dataclass(frozen=True)
class GDNConfig:
    hidden: int
    key_heads: int
    value_heads: int
    key_dim: int
    value_dim: int
    kernel: int
    epsilon: float

    @classmethod
    def from_model(cls, value):
        return cls(value["hidden_size"], value["linear_num_key_heads"],
                   value["linear_num_value_heads"], value["linear_key_head_dim"],
                   value["linear_value_head_dim"], value["linear_conv_kernel_dim"],
                   value["rms_norm_eps"])

    @property
    def channels(self):
        return 2 * self.key_heads * self.key_dim + self.value_heads * self.value_dim

    @property
    def weight_shapes(self):
        return {"in_proj_qkv.weight": (self.channels, self.hidden),
                "in_proj_z.weight": (self.value_heads * self.value_dim, self.hidden),
                "in_proj_a.weight": (self.value_heads, self.hidden),
                "in_proj_b.weight": (self.value_heads, self.hidden),
                "out_proj.weight": (self.hidden, self.value_heads * self.value_dim),
                "conv1d.weight": (self.channels, self.kernel, 1),
                "A_log": (self.value_heads,), "dt_bias": (self.value_heads,),
                "norm.weight": (self.value_dim,)}


class GDN(torch.nn.Module):
    def __init__(self, config, weights, dtype=torch.float16):
        super().__init__()
        self.config = config
        if config.value_heads % config.key_heads or config.kernel < 2:
            raise ValueError("Expected grouped GDN heads and a nonempty conv history")
        for name, shape in config.weight_shapes.items():
            value = np.asarray(weights[name])
            if value.shape != shape:
                raise ValueError(f"{name}: {value.shape} != {shape}")
            tensor = torch.from_numpy(value.copy()).to(device="cpu", dtype=dtype)
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Nonfinite converted weight: {name}")
            self.register_buffer(name.replace(".", "_"), tensor)

    @staticmethod
    def linear(x, weight):
        # FP16 CPU GEMM on the first SDK did not reliably accumulate accurately.
        return F.linear(x.float(), weight.float()).to(x.dtype)

    def norm(self, x, weight=None):
        result = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.config.epsilon)
        if weight is not None:
            result = result * weight.float()
        return result.to(x.dtype)

    def project_inputs(self, hidden):
        """Original four projections; the default S1 and chunk graphs are unchanged."""
        c = self.config
        sequence = hidden.shape[1]
        qkv = self.linear(hidden, self.in_proj_qkv_weight)
        z = self.linear(hidden, self.in_proj_z_weight).reshape(1, sequence, c.value_heads, c.value_dim)
        a = self.linear(hidden, self.in_proj_a_weight)
        b = self.linear(hidden, self.in_proj_b_weight)
        return qkv, z, a, b

    def forward(self, hidden, conv_history, recurrent_state):
        c = self.config
        sequence = hidden.shape[1]
        qkv, z, a, b = self.project_inputs(hidden)
        conv_input = torch.cat((conv_history, qkv), dim=1)
        next_history = conv_input[:, sequence:sequence + c.kernel - 1, :]
        # The exact causal depthwise convolution as a fixed sum of shifted
        # products avoids a 10240-group convolution backend requirement.
        convolved = conv_input[:, :sequence, :].float() * self.conv1d_weight[:, 0, 0].float()
        for tap in range(1, c.kernel):
            convolved = convolved + conv_input[:, tap:tap + sequence, :].float() * self.conv1d_weight[:, tap, 0].float()
        convolved = F.silu(convolved.to(hidden.dtype))
        key_width = c.key_heads * c.key_dim
        q = self.norm(convolved[..., :key_width].reshape(1, sequence, c.key_heads, c.key_dim))
        k = self.norm(convolved[..., key_width:2 * key_width].reshape(1, sequence, c.key_heads, c.key_dim))
        q = q * torch.tensor(1.0 / c.key_dim, dtype=hidden.dtype)
        k = k * torch.tensor(c.key_dim ** -0.5, dtype=hidden.dtype)
        repeats = c.value_heads // c.key_heads
        q = q.unsqueeze(3).expand(1, sequence, c.key_heads, repeats, c.key_dim).reshape(1, sequence, c.value_heads, c.key_dim).float()
        k = k.unsqueeze(3).expand(1, sequence, c.key_heads, repeats, c.key_dim).reshape(1, sequence, c.value_heads, c.key_dim).float()
        v = convolved[..., 2 * key_width:].reshape(1, sequence, c.value_heads, c.value_dim).float()
        decay = torch.exp(-self.A_log.float().exp() * F.softplus((a + self.dt_bias).float())).to(hidden.dtype).float()
        beta = b.sigmoid().float()
        state = recurrent_state.float()
        outputs = []
        for position in range(sequence):
            kt = k[:, position].unsqueeze(-2)
            state = state * decay[:, position, :, None, None]
            memory = (state * kt).sum(-1)
            delta = (v[:, position] - memory) * beta[:, position, :, None]
            state = state + delta.unsqueeze(-1) * kt
            outputs.append((state * q[:, position].unsqueeze(-2)).sum(-1).to(hidden.dtype))
        y = torch.stack(outputs, dim=1)
        gated = self.norm(y, self.norm_weight) * z.sigmoid()
        output = self.linear(gated.reshape(1, sequence, c.value_heads * c.value_dim), self.out_proj_weight)
        return output, next_history, state


class GDNPrefill(GDN):
    """Chunk prefill with an optional fused input projection, identical state I/O.

    The default shares the original GDN buffers and equations. It is suitable
    for separate S1/S4 functions whose weights can be deduplicated. Explicitly
    enabling fusion concatenates qkv/z/a/b rows into one constant and reduces
    the five projection GEMMs (four input, one output) to two per chunk. That
    packed constant is extra storage if the original decode module is retained;
    this class does not claim cross-function deduplication or a speedup.

    Projection, convolution, normalization and output projection cover all S
    tokens. Only the FP32 recurrent update is sequential; final state remains
    [batch,value_head,value_dim,key_dim], independent of chunk length.
    """

    INPUT_PROJECTION_NAMES = ("in_proj_qkv_weight", "in_proj_z_weight",
                              "in_proj_a_weight", "in_proj_b_weight")

    def __init__(self, source: GDN, *, fuse_input_projections=False):
        # Reuse immutable buffers instead of reading/converting a layer twice.
        # Never retain source as a submodule: that would duplicate named weights.
        torch.nn.Module.__init__(self)
        self.config = source.config
        self.fuse_input_projections = bool(fuse_input_projections)
        missing = [name for name in self.INPUT_PROJECTION_NAMES if not hasattr(source, name)]
        if missing:
            raise ValueError(f"GDNPrefill requires the original GDN projection buffers: {missing}")
        for name, value in source.named_buffers(recurse=False):
            if not self.fuse_input_projections or name not in self.INPUT_PROJECTION_NAMES:
                self.register_buffer(name, value)
        if self.fuse_input_projections:
            self.register_buffer("in_proj_combined_weight", torch.cat(
                [getattr(source, name) for name in self.INPUT_PROJECTION_NAMES], dim=0).contiguous())
        self.train(source.training)

    @classmethod
    def from_gdn(cls, source: GDN, *, fuse_input_projections=False):
        return cls(source, fuse_input_projections=fuse_input_projections)

    def project_inputs(self, hidden):
        if not self.fuse_input_projections:
            return super().project_inputs(hidden)
        c = self.config
        projected = self.linear(hidden, self.in_proj_combined_weight)
        qkv, z, a, b = torch.split(projected,
            (c.channels, c.value_heads * c.value_dim, c.value_heads, c.value_heads), dim=-1)
        return qkv, z.reshape(1, hidden.shape[1], c.value_heads, c.value_dim), a, b


def as_json(tensor):
    return tensor_json(tensor.detach().cpu().numpy())


def finite_outputs(outputs):
    if not all(torch.isfinite(value).all() for value in outputs):
        raise ValueError("Nonfinite GDN CPU output/state")


def export_asset(model, example, path):
    import coreai_torch
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    converter.add_pytorch_module(model, input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
        export_fn=lambda m: torch.export.export(m, args=example).run_decompositions(coreai_torch.get_decomp_table()))
    program = converter.to_coreai()
    program.optimize()
    program.save_asset(path)
    return [{"path": str(p.relative_to(path)), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
            for p in sorted(path.rglob("*")) if p.is_file()]


def capture_input(path, count, hidden):
    with safe_open(path, framework="pt", device="cpu") as source:
        metadata = source.metadata()
        x = source.get_tensor("x").float()
    if list(x.shape[:1]) != [1] or x.shape[1] < count or x.shape[-1] != hidden:
        raise ValueError(f"Capture does not contain {count} rows of [1,S,{hidden}]")
    x = x[:, :count].to(torch.float16).contiguous()
    if not torch.isfinite(x).all():
        raise ValueError("Nonfinite activation replay")
    return x, {"path": str(path), "sha256": sha256_file(path), "metadata": metadata,
               "selection": f"First {count} captured MoE input tokens in their original order",
               "scope": "MoE activation replay at a different GDN module boundary, not native GDN capture"}


def source_bf16_diagnostics(model, directory):
    """Compare a separate 26+1 replay against the existing immutable MLX oracle.

    This diagnostic is not the CoreAI compilation gate: dtype and persistent
    state rounding differ intentionally. It catches gross equation/weight
    mistakes while making that remaining model-fidelity boundary measurable.
    """
    c = model.config
    manifest_path = directory / "manifest.json"
    oracle_manifest = json.loads(manifest_path.read_text())
    case = next(row for row in oracle_manifest["cases"] if row["id"] == "gdn-continuous")
    history = torch.zeros(1, c.kernel - 1, c.channels, dtype=torch.float16)
    state = torch.zeros(1, c.value_heads, c.value_dim, c.key_dim, dtype=torch.float32)
    rows = []
    for step in case["steps"]:
        path = directory / step["file"]
        if sha256_file(path) != step["sha256"]:
            raise ValueError(f"Oracle file hash mismatch: {path}")
        with safe_open(path, framework="pt", device="cpu") as source:
            hidden = source.get_tensor("input").to(torch.float16)
            expected = [source.get_tensor(name).float().numpy() for name in (
                "expected.output", "expected.state.convHistory", "expected.state.recurrent")]
            metadata = source.metadata()
        with torch.inference_mode():
            actual = model(hidden, history, state)
        finite_outputs(actual)
        rows.append({"phase": step["phase"], "path": str(path), "sha256": step["sha256"],
                     "metadata": metadata,
                     "comparisons": {name: errors(value.float().numpy(), target)
                                     for name, value, target in zip(OUTPUT_NAMES, actual, expected)}})
        history, state = actual[1:]
    return {"scope": "Diagnostic FP16/FP32 candidate versus existing BF16 MLX sublayer replay; not bitwise equivalence or a runtime gate",
            "manifest": str(manifest_path), "manifest_sha256": sha256_file(manifest_path), "steps": rows}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/coreai-stateful/gdn")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--prefill", type=int, default=4, choices=range(1, 17), metavar="1..16")
    parser.add_argument("--prefill-fused-projections", action="store_true",
                        help="Use one packed qkv/z/a/b projection for the prefill asset only; S1 decode remains original")
    parser.add_argument("--decode-steps", type=int, default=3, choices=range(1, 9), metavar="1..8")
    parser.add_argument("--capture", type=Path, default=ROOT / "fixtures/moe-real/prefill.safetensors")
    parser.add_argument("--source-oracle", type=Path, default=ROOT / "fixtures/gpu-sequence-reference")
    parser.add_argument("--reference-only", action="store_true", help="Author CPU fixtures without exporting assets")
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    if args.prefill_fused_projections and args.prefill == 1:
        raise ValueError("Fused prefill requires S>1 so its asset cannot replace the original S1 decode function")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output}")
    source = Source(args.layer)
    config_path = source.directory / "config.json"
    config_json = json.loads(config_path.read_text())["text_config"]
    if not 0 <= args.layer < len(config_json["layer_types"]) or config_json["layer_types"][args.layer] != "linear_attention":
        raise ValueError("--layer must name a GDN layer")
    config = GDNConfig.from_model(config_json)
    source.prefix = f"language_model.model.layers.{args.layer}.linear_attn."
    weights = {name: source.read(name) for name in config.weight_shapes}
    model = GDN(config, weights).eval()
    prefill_model = GDNPrefill.from_gdn(model, fuse_input_projections=args.prefill_fused_projections).eval()
    input_all, evidence = capture_input(args.capture.resolve(), args.prefill + args.decode_steps, config.hidden)
    history = torch.zeros(1, config.kernel - 1, config.channels, dtype=torch.float16)
    state = torch.zeros(1, config.value_heads, config.value_dim, config.key_dim, dtype=torch.float32)
    initial_history, initial_state = history.clone(), state.clone()
    output.mkdir(parents=True)
    write_json(output / "initial-state.json", {"conv_history": as_json(initial_history),
                                                "recurrent_state": as_json(initial_state)})
    sequence = {"version": 1,
                "models": {"prefill": {"path": f"layer{args.layer}-gdn-s{args.prefill}.aimodel", "function": "main"},
                           "decode": {"path": f"layer{args.layer}-gdn-s1.aimodel", "function": "main"}},
                "stateBindings": STATE_BINDINGS, "initialState": "initial-state.json", "steps": [],
                "tolerances": {"maximumAbsoluteError": 0.02, "relativeL2Error": 0.005}}
    manifest = {"schema": "coreai-gdn-export-v1", "status": "authoring_in_progress",
                "layer": args.layer, "config": vars(config), "function": "main",
                "input_names": INPUT_NAMES, "output_names": OUTPUT_NAMES, "state_bindings": STATE_BINDINGS,
                "recurrence_state_layout": "batch,value_head,value_dim,key_dim", "recurrence_state_dtype": "float32",
                "activation_and_weight_storage_dtype": "float16", "projection_and_conv_accumulation_dtype": "float32",
                "prefill_fused_input_projections": args.prefill_fused_projections,
                "prefill_projection_gemms_per_chunk": 2 if args.prefill_fused_projections else 5,
                "input_provenance": evidence, "source_records": source.records,
                "config_file": {"path": str(config_path), "sha256": sha256_file(config_path)},
                "exporter": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__))},
                "versions": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "coreai-core", "coreai-torch")},
                "limitations": ["A full real-weight GDN sublayer, not a full decoder layer or language-model generation.",
                                "MoE activation replay is not native GDN activation capture.",
                                "FP16 activation/weight conversion and FP32 persistent state differ from the existing BF16 runner.",
                                "CPU authoring only; device execution, hardware placement and performance are not established."],
                "suggested_runtime_gate": {"relative_l2_max": 0.005, "max_absolute_error_max": 0.02,
                                           "scope": "Initial sublayer smoke gate; not full-model production acceptance"},
                "exports": [], "steps": []}
    write_json(output / "manifest.json", manifest)
    examples = {}
    offset = 0
    with torch.inference_mode():
        for index, length in enumerate([args.prefill] + [1] * args.decode_steps):
            hidden = input_all[:, offset:offset + length]
            inputs = (hidden, history, state)
            examples.setdefault(length, tuple(value.clone() for value in inputs))
            values = (prefill_model if index == 0 else model)(*inputs)
            finite_outputs(values)
            fixture_path = output / f"step-{index}.json"
            write_json(fixture_path, {"inputs": {"hidden": as_json(hidden)},
                                     "expectedOutputs": dict(zip(OUTPUT_NAMES, map(as_json, values)))})
            phase = "prefill" if index == 0 else "decode"
            sequence["steps"].append({"name": f"{phase}-{index}", "phase": phase,
                                       "model": phase, "fixture": fixture_path.name})
            cold_values = model(hidden, initial_history, initial_state) if index else values
            state, history = values[2], values[1]
            step = {"index": index, "phase": "prefill" if index == 0 else "decode", "length": length,
                    "offset_before": offset, "offset_after": offset + length,
                    "fixture": str(fixture_path), "fixture_sha256": sha256_file(fixture_path),
                    "output_max_abs": float(values[0].float().abs().max()),
                    "state_max_abs": float(state.abs().max()),
                    "continued_vs_cold_output": errors(values[0].float().numpy(), cold_values[0].float().numpy())}
            manifest["steps"].append(step)
            offset += length
            write_json(output / "manifest.json", manifest)
    write_json(output / "sequence.json", sequence)
    if args.layer == 0:
        manifest["source_bf16_diagnostics"] = source_bf16_diagnostics(model, args.source_oracle.resolve())
        write_json(output / "manifest.json", manifest)
    for length, example in examples.items():
        path = output / f"layer{args.layer}-gdn-s{length}.aimodel"
        started = time.perf_counter()
        asset_model = prefill_model if length == args.prefill and length > 1 else model
        files = [] if args.reference_only else export_asset(asset_model, example, path)
        manifest["exports"].append({"length": length, "model": str(path), "model_files": files,
                                    "model_bytes": sum(row["bytes"] for row in files),
                                    "authoring_seconds": time.perf_counter() - started})
        write_json(output / "manifest.json", manifest)
        print(f"Prepared GDN S{length}: {path}", flush=True)
    manifest["status"] = "cpu_reference_complete" if args.reference_only else "authoring_complete_runtime_not_executed"
    write_json(output / "manifest.json", manifest)
    print(f"Manifest: {output / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
