#!/usr/bin/env python3
"""Export every non-attention/non-MoE neural block for a pure CoreAI text runner.

CPU authoring only. No MLX, Core ML, CoreAI runtime, GPU or ANE execution occurs.
Equations follow GPUHyperConnection.swift, GPUPLE.swift and QwenModel.swift;
source attribution: garnermccloud/mlx-serve transformer.zig, MIT, upstream commit
7dbcba04c98e4fd3bcc533c63e645547f13cc3b1 (see UPSTREAM-LICENSE).

All weights are read as bounded real BF16 tensors from the verified local model,
then stored as FP16. Neural intermediates round to FP16; matrix products,
convolution accumulation and reductions use FP32 where specified. This candidate
is not bit-equivalent to the BF16 runner. SSD row lookup/hash and text handling
remain CPU responsibilities; all PLE neural math is inside the exported graph.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from export_moe import Source, sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DenseConfig:
    hidden: int
    streams: int
    low_rank: int
    epsilon: float
    vocabulary: int
    ple_dim: int
    ple_kernel: int
    ple_dilation: int

    @classmethod
    def from_model(cls, config):
        return cls(config["hidden_size"], config["hc_count"], config["hc_lowrank"],
                   config["rms_norm_eps"], config["vocab_size"], config["ple_embed_dim"],
                   config["ple_conv_kernel_size"], config["ngram_size"])

    @property
    def width(self):
        return self.hidden * self.streams

    @property
    def ple_history(self):
        return (self.ple_kernel - 1) * self.ple_dilation


def half_weight(value, shape, name):
    value = np.asarray(value)
    if value.shape != shape:
        raise ValueError(f"{name} shape {value.shape} != {shape}")
    result = torch.from_numpy(value.copy()).half()
    if not torch.isfinite(result).all():
        raise ValueError(f"Nonfinite FP16 weight {name}")
    return result


def linear(x, weight):
    return F.linear(x.float(), weight.float()).half()


def silu(x):
    # Preserve separate activation/product rounding, as in the original MLX path.
    return (x * torch.sigmoid(x)).half()


def group_norm(x, weight, config):
    grouped = x.reshape(1, x.shape[1], config.streams, config.hidden)
    normalized = (grouped.float() * torch.rsqrt(grouped.float().square().mean(-1, keepdim=True) + config.epsilon)).half()
    return (normalized * weight.reshape(config.streams, config.hidden)).half()


class HCRead(torch.nn.Module):
    linear = staticmethod(linear)
    def __init__(self, config, weights, with_injection=True):
        super().__init__()
        self.config, self.with_injection = config, with_injection
        self.register_buffer("down", (half_weight(weights["input_mix_weight_down.weight"],
            (config.low_rank, config.width), "HC down") / config.streams).half())
        self.register_buffer("up", half_weight(weights["input_mix_weight_up.weight"],
            (config.width, config.low_rank), "HC up"))
        self.register_buffer("norm", half_weight(weights["hc_norm.weight"], (config.width,), "HC norm"))
        if with_injection:
            self.register_buffer("inject", (half_weight(weights["block_inject_weight.weight"],
                (config.streams, config.width), "HC injection") / config.streams).half())

    def forward(self, stream):
        c = self.config
        count = stream.shape[1]
        normalized = group_norm(stream, self.norm, c)
        flat = normalized.reshape(1, count, c.width)
        activated = silu(self.linear(flat, self.down))
        logits = self.linear(activated, self.up).reshape(1, count, c.streams, c.hidden)
        weighted = (normalized * torch.sigmoid(logits)).half()
        mixed = weighted.float().mean(2).half()
        if not self.with_injection:
            return mixed
        injection = (torch.sigmoid(self.linear(flat, self.inject)) * 2).half().reshape(1, count, c.streams, 1)
        return mixed, injection


class HCWrite(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, stream, output, injection):
        c = self.config
        count = stream.shape[1]
        update = (output.reshape(1, count, 1, c.hidden) * injection).half()
        return (stream.reshape(1, count, c.streams, c.hidden) + update).half().reshape(1, count, c.width)


class PLE(torch.nn.Module):
    linear = staticmethod(linear)
    def __init__(self, config, weights):
        super().__init__()
        self.config = config
        shapes = {"key_proj.weight": (config.width, config.ple_dim),
                  "value_proj.weight": (config.hidden, config.ple_dim),
                  "norm_key.weight": (config.width,), "norm_query.weight": (config.width,),
                  "norm_conv.weight": (config.width,), "conv1d.weight": (config.width, config.ple_kernel, 1)}
        for name, shape in shapes.items():
            self.register_buffer(name.replace(".", "_"), half_weight(weights[name], shape, name))
        self.register_buffer("gate_scale", torch.tensor(config.hidden ** -0.5, dtype=torch.float16))

    def forward(self, stream, embedding, conv_state):
        c = self.config
        count = stream.shape[1]
        key = group_norm(self.linear(embedding, self.key_proj_weight), self.norm_key_weight, c)
        value = self.linear(embedding, self.value_proj_weight)
        query = group_norm(stream, self.norm_query_weight, c)
        products = (key * query).half()
        gate = (products.float().sum(-1, keepdim=True).half() * self.gate_scale).half()
        signed_root = (torch.sqrt(torch.clamp(torch.abs(gate), min=1e-6)).half() * torch.sign(gate)).half()
        gated = (torch.sigmoid(signed_root) * value.reshape(1, count, 1, c.hidden)).half().reshape(1, count, c.width)
        normalized = group_norm(gated, self.norm_conv_weight, c).reshape(1, count, c.width)
        joined = torch.cat((conv_state, normalized), dim=1)
        convolution = joined[:, :count].float() * self.conv1d_weight[:, 0, 0].float()
        for tap in range(1, c.ple_kernel):
            index = tap * c.ple_dilation
            convolution = convolution + joined[:, index:index + count].float() * self.conv1d_weight[:, tap, 0].float()
        addition = (gated + silu(convolution.half())).half()
        stream_out = (stream + addition).half()
        next_conv = joined[:, count:count + c.ple_history]
        return stream_out, next_conv


class Embedding(torch.nn.Module):
    def __init__(self, config, weight):
        super().__init__()
        self.config = config
        self.register_buffer("weight", half_weight(weight, (config.vocabulary, config.hidden), "embedding"))

    def forward(self, token):
        return F.embedding(token.long(), self.weight).reshape(1, token.shape[0], self.config.hidden).repeat(1, 1, self.config.streams)


class Head(torch.nn.Module):
    def __init__(self, config, weights, weight):
        super().__init__()
        self.mixer = HCRead(config, weights, with_injection=False)
        self.register_buffer("weight", half_weight(weight, (config.vocabulary, config.hidden), "head"))

    def forward(self, stream):
        mixed = self.mixer(stream)
        # Keep logits FP32 for sampling and comparisons; avoid an unnecessary
        # vocabulary-sized FP16 rounding at the final boundary.
        return F.linear(mixed.float(), self.weight.float())


def read_hc(source, prefix, with_injection=True):
    names = ["input_mix_weight_down.weight", "input_mix_weight_up.weight", "hc_norm.weight"]
    if with_injection:
        names.append("block_inject_weight.weight")
    source.prefix = prefix + "."
    return {name: source.read(name) for name in names}


def tensor_spec(tensor):
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype).removeprefix("torch.")}


def export_asset(module, examples, output_names, path):
    import coreai_torch
    module.eval()
    inputs = tuple(examples)
    with torch.inference_mode():
        outputs = module(*(examples[name] for name in inputs))
    outputs = outputs if isinstance(outputs, tuple) else (outputs,)
    if len(outputs) != len(output_names) or not all(torch.isfinite(value).all() for value in outputs):
        raise ValueError("Invalid dense-block CPU output")
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    converter.add_pytorch_module(module, input_names=inputs, output_names=output_names,
        export_fn=lambda model: torch.export.export(model, args=tuple(examples[name] for name in inputs)).run_decompositions(coreai_torch.get_decomp_table()))
    program = converter.to_coreai()
    program.optimize()
    program.save_asset(path)
    files = [{"path": str(p.relative_to(path)), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
             for p in sorted(path.rglob("*")) if p.is_file()]
    return {"path": path.name, "function": "main", "inputNames": list(inputs), "outputNames": list(output_names),
            "inputSpecs": {name: tensor_spec(examples[name]) for name in inputs},
            "outputSpecs": {name: tensor_spec(value) for name, value in zip(output_names, outputs)},
            "modelBytes": sum(row["bytes"] for row in files), "files": files}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/coreai-native/dense")
    parser.add_argument("--components", nargs="+", choices=("embedding", "hc", "write", "ple", "head"),
                        default=["embedding", "hc", "write", "ple", "head"])
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(48)))
    args = parser.parse_args(argv)
    if len(set(args.layers)) != len(args.layers) or any(not 0 <= layer < 48 for layer in args.layers):
        parser.error("--layers requires unique indices in 0..<48")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output}")
    source = Source(0)
    config_path = source.directory / "config.json"
    config_json = json.loads(config_path.read_text())["text_config"]
    config = DenseConfig.from_model(config_json)
    if (config.hidden, config.streams, config.low_rank, config_json["num_hidden_layers"], config_json["ple_layer_ids"]) != (2560, 4, 320, 48, [2]):
        raise ValueError("This exporter is for the downloaded Qwen3.8 Flash-Next architecture")
    output.mkdir(parents=True)
    manifest_path = output / "manifest.json"
    manifest = {"version": 1, "status": "exporting", "modelDirectory": str(source.directory.resolve()),
                "configSHA256": sha256_file(config_path), "layerCount": 48,
                "layerIndices": sorted(args.layers), "config": vars(config), "tokenChunk": 1,
                "assets": {}, "layers": [], "sourceRecords": source.records,
                "exporterSHA256": sha256_file(Path(__file__)),
                "versions": {name: importlib.metadata.version(name) for name in ("torch", "numpy", "coreai-core", "coreai-torch")},
                "limitations": ["CPU authoring only: runtime, full-model correctness and performance need separate validation.",
                                "FP16 neural path and FP32 logits differ numerically from the BF16 reference runner.",
                                "S1 text-only; SSD/hash/tokenizer are host work, neural PLE math is in the model."]}
    write_json(manifest_path, manifest)
    stream = torch.zeros(1, 1, config.width, dtype=torch.float16)
    began = time.perf_counter()

    def save(module, inputs, names, filename):
        started = time.perf_counter()
        asset = export_asset(module, inputs, names, output / filename)
        asset["authoringSeconds"] = time.perf_counter() - started
        print(f"Exported {filename}: {asset['modelBytes']} bytes, {asset['authoringSeconds']:.2f}s", flush=True)
        return asset

    try:
        if "write" in args.components:
            manifest["assets"]["hcWrite"] = save(HCWrite(config),
                {"stream": stream, "output": torch.zeros(1, 1, config.hidden, dtype=torch.float16),
                 "injection": torch.zeros(1, 1, config.streams, 1, dtype=torch.float16)}, ("stream_out",), "hc-write.aimodel")
            write_json(manifest_path, manifest)
        if "hc" in args.components:
            for layer in sorted(args.layers):
                row = {"index": layer}
                for kind, field in (("attn", "attentionRead"), ("mlp", "moeRead")):
                    weights = read_hc(source, f"language_model.model.layers.{layer}.{kind}_hyper_connection")
                    module = HCRead(config, weights)
                    row[field] = save(module, {"stream": stream}, ("mixed", "injection"), f"layer{layer:02d}-hc-{kind}.aimodel")
                    del module, weights
                    gc.collect()
                manifest["layers"].append(row)
                write_json(manifest_path, manifest)
        if "ple" in args.components:
            source.prefix = "language_model.model.layers.1.ple."
            weights = {name: source.read(name) for name in ("key_proj.weight", "value_proj.weight", "norm_key.weight", "norm_query.weight", "norm_conv.weight", "conv1d.weight")}
            module = PLE(config, weights)
            asset = save(module, {"stream": stream, "embedding": torch.zeros(1, 1, config.ple_dim, dtype=torch.float16),
                "conv_state": torch.zeros(1, config.ple_history, config.width, dtype=torch.float16)},
                ("stream_out", "next_conv_state"), "layer01-ple.aimodel")
            asset.update({"layerIndex": 1, "stateBindings": {"conv_state": "next_conv_state"},
                          "initialState": {"conv_state": {"shape": [1, config.ple_history, config.width], "dtype": "float16", "fill": 0}}})
            manifest["assets"]["ple"] = asset
            write_json(manifest_path, manifest)
            del module, weights
            gc.collect()
        if "embedding" in args.components:
            source.prefix = ""
            weight = source.read("language_model.model.embed_tokens.weight")
            module = Embedding(config, weight)
            del weight
            manifest["assets"]["embedding"] = save(module, {"token": torch.zeros(1, dtype=torch.int32)}, ("stream",), "embedding.aimodel")
            write_json(manifest_path, manifest)
            del module
            gc.collect()
        if "head" in args.components:
            weights = read_hc(source, "language_model.model.hyper_connection_mixer", with_injection=False)
            source.prefix = ""
            weight = source.read("language_model.lm_head.weight")
            module = Head(config, weights, weight)
            del weights, weight
            manifest["assets"]["head"] = save(module, {"stream": stream}, ("logits",), "head.aimodel")
            write_json(manifest_path, manifest)
            del module
            gc.collect()
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        write_json(manifest_path, manifest)
        raise
    manifest["status"] = "complete"
    manifest["completeModelLayerSet"] = (sorted(args.layers) == list(range(48)) and
        set(args.components) == {"embedding", "hc", "write", "ple", "head"})
    manifest["authoringSeconds"] = time.perf_counter() - began
    assets = list(manifest["assets"].values()) + [row[name] for row in manifest["layers"] for name in ("attentionRead", "moeRead")]
    manifest["modelBytes"] = sum(asset["modelBytes"] for asset in assets)
    write_json(manifest_path, manifest)
    print(f"Complete manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
