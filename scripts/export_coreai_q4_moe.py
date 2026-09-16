#!/usr/bin/env python3
"""CPU-author a complete S1 MoE using original packed affine-Q4 expert banks.

The CoreAI graph performs routing, selected-bank gather, integer nibble unpack,
three expert projections, shared expert and weighted output reduction. Packed
512-expert banks are never expanded to dense weights. Only runtime-selected
experts are unpacked. This is a functional candidate, not a fused Q4 kernel.

--experts 12 creates a real-weight subset for operation smoke testing. Omit it
for the full 512-expert bank; a subset must never be loaded as a full model.
No CoreAI/Core ML/MLX runtime or device inference is executed by this script.
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F

from export_moe import Source, sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
OUTPUT_NAMES = ("output", "selected_ids", "selected_scores")
DIAGNOSTIC_NAMES = ("expert_gate", "expert_up", "expert_active", "expert_down", "routed", "shared_gate",
                    "shared_up", "shared_active", "shared_down", "shared_score")


class PackedQ4(torch.nn.Module):
    """Gather packed expert rows before producing any dense expert tensor."""
    def __init__(self, packed, scales, biases, group_size=64):
        super().__init__()
        if packed.dtype != np.uint32 or packed.ndim != 3 or scales.shape != biases.shape:
            raise ValueError("Expected U32 expert banks and matching affine parameters")
        experts, outputs, words = packed.shape
        if scales.shape != (experts, outputs, words * 8 // group_size) or words * 8 % group_size:
            raise ValueError("Packed affine group layout mismatch")
        self.group_size = group_size
        self.input_size = words * 8
        # CoreAI 27 beta's native gather of large I32 values loses low bits as if
        # converted through FP32. Reinterpret each source word as two signed I16
        # lanes: identical bytes/compression, and every lane is exactly FP32-safe.
        self.register_buffer("packed", torch.from_numpy(packed.view(np.int16).copy()))
        self.register_buffer("scales", torch.from_numpy(scales.copy()).half())
        self.register_buffer("biases", torch.from_numpy(biases.copy()).half())
        powers = np.left_shift(np.uint32(1), np.arange(0, 16, 4, dtype=np.uint32))
        masks = powers * np.uint32(15)
        self.register_buffer("powers", torch.from_numpy(powers.astype(np.int32)))
        self.register_buffer("masks", torch.from_numpy(masks.view(np.int32).copy()))
        self.register_buffer("nibble_mask", torch.tensor(15, dtype=torch.int32))
        if not torch.isfinite(self.scales).all() or not torch.isfinite(self.biases).all():
            raise ValueError("Affine metadata does not fit FP16")

    def selected_dense(self, ids):
        # These gathers must stay before all unpack/dequantization operations.
        packed = torch.index_select(self.packed, 0, ids.long()).to(torch.int32)
        scales = torch.index_select(self.scales, 0, ids.long())
        biases = torch.index_select(self.biases, 0, ids.long())
        # TorchConverter currently lacks right_shift lowering. Integer masking
        # plus power-of-two floor division exactly extracts signed I16 bitpatterns.
        # Masking first also limits each dividend to four significant bits.
        masked = torch.bitwise_and(packed.unsqueeze(-1), self.masks)
        codes = torch.bitwise_and(torch.div(masked, self.powers, rounding_mode="floor"), self.nibble_mask)
        codes = codes.reshape(packed.shape[0], packed.shape[1], -1, self.group_size).float()
        dense = (codes * scales.float().unsqueeze(-1) + biases.float().unsqueeze(-1)).half()
        return dense.reshape(packed.shape[0], packed.shape[1], self.input_size)

    def forward(self, x, ids):
        dense = self.selected_dense(ids)
        return torch.matmul(x.float(), dense.float().transpose(-1, -2)).half()


class Q4MoE(torch.nn.Module):
    def __init__(self, router, shared_router, shared, quantized, top_k=10, diagnostics=False):
        super().__init__()
        self.experts, self.hidden = router.shape
        self.top_k = top_k
        self.diagnostics = diagnostics
        if not 1 <= top_k <= self.experts:
            raise ValueError("Invalid routed expert count")
        self.register_buffer("router", torch.from_numpy(router.copy()).half())
        self.register_buffer("shared_router", torch.from_numpy(shared_router.copy()).half())
        self.register_buffer("expert_ids", torch.arange(self.experts, dtype=torch.int32))
        for name in PROJECTIONS:
            self.register_buffer("shared_" + name, torch.from_numpy(shared[name].copy()).half())
            setattr(self, name, PackedQ4(*quantized[name]))

    @staticmethod
    def linear(x, weight):
        return F.linear(x.float(), weight.float()).half()

    def routing(self, x):
        logits = self.linear(x, self.router).reshape(self.experts)
        remaining = logits.float()
        selected = []
        # Explicit lowest-ID ties, matching the source router independently of
        # a backend's topk/argsort tie policy.
        for _ in range(self.top_k):
            maximum = remaining.amax(dim=0)
            chosen = torch.where(remaining == maximum, self.expert_ids, self.experts).amin(dim=0)
            selected.append(chosen)
            remaining = torch.where(self.expert_ids == chosen, float("-inf"), remaining)
        ids = torch.stack(selected)
        probabilities = logits.float().softmax(-1).half()
        scores = torch.index_select(probabilities, 0, ids.long())
        denominator = scores[0]
        for slot in range(1, self.top_k):
            denominator = (denominator + scores[slot]).half()
        return ids, (scores / denominator).half()

    def forward(self, x):
        if x.shape[1] != 1:
            return self.prefill(x)
        ids, scores = self.routing(x)
        expanded = x.reshape(1, 1, self.hidden).expand(self.top_k, 1, self.hidden)
        gate = self.gate_proj(expanded, ids)
        up = self.up_proj(expanded, ids)
        active = ((gate * gate.sigmoid()).half() * up).half()
        down = self.down_proj(active, ids).reshape(self.top_k, self.hidden)
        products = (down * scores[:, None]).half()
        routed = products.float().sum(0).half().reshape(1, 1, self.hidden)
        shared_gate = self.linear(x, self.shared_gate_proj)
        shared_up = self.linear(x, self.shared_up_proj)
        shared_active = ((shared_gate * shared_gate.sigmoid()).half() * shared_up).half()
        shared_down = self.linear(shared_active, self.shared_down_proj)
        shared_score = self.linear(x, self.shared_router).sigmoid()
        output = (routed + (shared_down * shared_score).half()).half()
        result = (output, ids.reshape(1, 1, self.top_k), scores.reshape(1, 1, self.top_k))
        if self.diagnostics:
            return result + (gate, up, active, down, routed, shared_gate, shared_up, shared_active, shared_down, shared_score)
        return result

    def prefill(self, x):
        """Vectorized token routing and selected expert projections, bounded by S.

        Each token retains its own ordered top-k and FP16 score normalization.
        Only S*top_k expert matrices are materialized, never the full bank.
        """
        count = x.shape[1]
        logits = self.linear(x, self.router).reshape(count, self.experts)
        remaining = logits.float()
        selected = []
        for _ in range(self.top_k):
            maximum = remaining.amax(dim=-1, keepdim=True)
            chosen = torch.where(remaining == maximum, self.expert_ids, self.experts).amin(dim=-1)
            selected.append(chosen)
            remaining = torch.where(self.expert_ids == chosen[:, None], float("-inf"), remaining)
        ids = torch.stack(selected, dim=-1)
        scores = torch.gather(logits.float().softmax(-1).half(), -1, ids.long())
        denominator = scores[:, 0]
        for slot in range(1, self.top_k):
            denominator = (denominator + scores[:, slot]).half()
        scores = (scores / denominator[:, None]).half()
        expanded = x.reshape(count, 1, self.hidden).expand(count, self.top_k, self.hidden)
        expanded = expanded.reshape(count * self.top_k, 1, self.hidden)
        flat_ids = ids.reshape(-1)
        gate = self.gate_proj(expanded, flat_ids)
        up = self.up_proj(expanded, flat_ids)
        active = ((gate * gate.sigmoid()).half() * up).half()
        down = self.down_proj(active, flat_ids).reshape(count, self.top_k, self.hidden)
        routed = (down * scores[:, :, None]).half().float().sum(1).half().reshape(1, count, self.hidden)
        shared_gate = self.linear(x, self.shared_gate_proj)
        shared_up = self.linear(x, self.shared_up_proj)
        shared_active = ((shared_gate * shared_gate.sigmoid()).half() * shared_up).half()
        shared_down = self.linear(shared_active, self.shared_down_proj)
        shared_score = self.linear(x, self.shared_router).sigmoid()
        output = (routed + (shared_down * shared_score).half()).half()
        result = (output, ids.reshape(1, count, self.top_k), scores.reshape(1, count, self.top_k))
        if self.diagnostics:
            return result + (gate, up, active, down, routed, shared_gate, shared_up, shared_active, shared_down, shared_score)
        return result


def load_layer(source, expert_count, top_k, diagnostics=False):
    router = source.read("gate.weight")[:expert_count]
    shared_router = source.read("shared_expert_gate.weight")
    shared = {name: source.read("shared_expert." + name + ".weight") for name in PROJECTIONS}
    quantized = {}
    for name in PROJECTIONS:
        prefix = "switch_mlp." + name + "."
        if expert_count == 512:
            quantized[name] = tuple(source.read(prefix + part) for part in ("weight", "scales", "biases"))
        else:
            quantized[name] = tuple(np.stack([source.read(prefix + part, expert) for expert in range(expert_count)])
                                    for part in ("weight", "scales", "biases"))
    return Q4MoE(router, shared_router, shared, quantized, top_k, diagnostics=diagnostics).eval()


def tensor_json(value):
    value = value.detach().cpu().numpy()
    return {"shape": list(value.shape), "dtype": str(value.dtype), "values": value.reshape(-1).astype(float).tolist()}


def export_asset(model, path, x):
    import coreai_torch
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    names = OUTPUT_NAMES + (DIAGNOSTIC_NAMES if model.diagnostics else ())
    converter.add_pytorch_module(model, input_names=("x",), output_names=names,
        export_fn=lambda m: torch.export.export(m, args=(x,)).run_decompositions(coreai_torch.get_decomp_table()))
    program = converter.to_coreai()
    program.optimize()
    program.save_asset(path)
    return [{"path": str(p.relative_to(path)), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
            for p in sorted(path.rglob("*")) if p.is_file()]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/coreai-q4-moe")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--layer", type=int, help="One layer, default 0")
    selection.add_argument("--layers", help="all or comma-separated layer indices")
    parser.add_argument("--experts", type=int, default=512)
    parser.add_argument("--diagnostics", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    if args.output.exists():
        raise FileExistsError("Choose a fresh output directory")
    source = Source(0)
    config_path = source.directory / "config.json"
    config = json.loads(config_path.read_text())["text_config"]
    if args.layers == "all":
        layers = list(range(config["num_hidden_layers"]))
    elif args.layers is not None:
        layers = [int(value) for value in args.layers.split(",")]
    else:
        layers = [args.layer if args.layer is not None else 0]
    if (not layers or len(set(layers)) != len(layers) or any(not 0 <= layer < config["num_hidden_layers"] for layer in layers)
            or not config["num_experts_per_tok"] <= args.experts <= config["num_experts"]):
        raise ValueError("Invalid layer or expert subset count")
    layers.sort()
    started = time.perf_counter()
    args.output.mkdir(parents=True)
    estimated = len(layers) * (args.experts * 2_800_000 + 12_500_000)
    if shutil.disk_usage(args.output).free < estimated + 512 * 1024 * 1024:
        raise ValueError("Insufficient free disk space for packed MoE assets plus 512 MiB margin")
    fixture_path = ROOT / "fixtures/moe-real/converted/decode.json"
    capture = json.loads(fixture_path.read_text())["inputs"]["x"]
    actual = torch.tensor(capture["values"], dtype=torch.float16).reshape(capture["shape"])[:, :1].contiguous()
    cases = {"actual": actual, "zero": torch.zeros_like(actual)}
    report = {"version": 1, "status": "exporting", "modelDirectory": str(source.directory),
              "configSHA256": sha256_file(config_path), "layers": [], "selectedLayers": layers,
              "layerCount": config["num_hidden_layers"], "exportedLayerCount": 0,
              "completeModelLayerSet": False, "completeExpertBanks": args.experts == config["num_experts"],
              "cases": [], "capture": {"path": str(fixture_path), "sha256": sha256_file(fixture_path),
                                         "scope": "Captured layer-0 MoE decode input; other layers use cross-layer replay"},
              "scriptSHA256": sha256_file(Path(__file__)),
              "versions": {name: importlib.metadata.version(name) for name in ("torch", "coreai-core", "coreai-torch")},
              "packedFormat": "Original affine group64 Q4 U32 bytes reinterpreted as pairs of I16 lanes; unchanged bank bytes, gather selected experts before unpack",
              "precision": "FP16 weights/activation boundaries and affine dense selected weights; FP32 GEMM/reduction",
              "runtimeCompatibility": "macOS 27 beta GPU-preferred I32 gather/unpack graph lost packed low bits while CPUOnly was exact; this observation does not isolate the responsible internal operator. I16 lane storage preserves source bytes and avoids loss in validated GPU-preferred smoke.",
              "limitations": ["CPU references and export only; CoreAI runtime validation is separate.",
                               "FP16 operation boundaries differ from the source BF16 runner.",
                               "Subset exports restrict routing and must not be used as complete model layers.",
                               "Gather/unpack/FP32 GEMM is a functional path, not a fused packed-Q4 performance kernel."]}
    def publish():
        temporary = args.output / "manifest.json.tmp"
        write_json(temporary, report)
        temporary.replace(args.output / "manifest.json")
    publish()
    try:
        for layer in layers:
            layer_started = time.perf_counter()
            source = Source(layer)
            model = load_layer(source, args.experts, config["num_experts_per_tok"], diagnostics=args.diagnostics)
            for name, x in cases.items():
                with torch.inference_mode():
                    values = model(x)
                if not all(torch.isfinite(value).all() for value in values):
                    raise ValueError(f"Nonfinite CPU MoE reference at layer {layer}")
                prefix = f"layer-{layer:02d}-" if len(layers) > 1 else ""
                path = args.output / f"{prefix}{name}.json"
                write_json(path, {"inputs": {"x": tensor_json(x)}, "expectedOutputs": {
                    key: tensor_json(value) for key, value in zip(OUTPUT_NAMES + (DIAGNOSTIC_NAMES if args.diagnostics else ()), values)}})
                report["cases"].append({"layer": layer, "name": name, "fixture": path.name, "sha256": sha256_file(path),
                                        "selectedIDs": values[1].reshape(-1).tolist(), "maximumAbsoluteOutput": float(values[0].abs().max())})
            asset = args.output / f"layer-{layer:02d}-moe-q4-e{args.experts}.aimodel"
            files = export_asset(model, asset, actual)
            report["layers"].append({"index": layer, "path": asset.name, "function": "main", "inputName": "x", "outputName": "output",
                "expertCount": args.experts, "topK": config["num_experts_per_tok"], "completeExpertBank": args.experts == config["num_experts"],
                "files": files, "modelBytes": sum(row["bytes"] for row in files), "sourceRecords": source.records,
                "authoringSeconds": time.perf_counter() - layer_started})
            report["exportedLayerCount"] = len(report["layers"])
            publish()
            print(f"Exported MoE layer {layer:02d}: {report['layers'][-1]['modelBytes']} bytes, "
                  f"{report['layers'][-1]['authoringSeconds']:.2f}s", flush=True)
            del model
            gc.collect()
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        publish()
        raise
    report["status"] = "complete"
    report["completeModelLayerSet"] = layers == list(range(config["num_hidden_layers"]))
    report["modelBytes"] = sum(row["modelBytes"] for row in report["layers"])
    report["authoringSeconds"] = time.perf_counter() - started
    publish()
    print(f"Manifest: {args.output / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
