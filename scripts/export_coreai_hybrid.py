#!/usr/bin/env python3
"""Export every real attention sublayer for the initial CoreAI/MLX hybrid runner.

CPU authoring only: no CoreAI/Core ML/MLX inference or full-model instantiation.
One layer is resident at a time. S1 functions serve both tokenwise prefill and
decode. The runner retains the other decoder operations and existing SSD PLE.
The manifest describes zero state metadata; it never stores large state JSONs.

Default output is intentionally a fresh directory: partial exports cannot be
resumed or mistaken for a complete set. --layers 0,3 makes a two-layer smoke
export; omitting --layers exports the complete config-derived layer list.
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import torch

from export_moe import Source, sha256_file
from export_coreai_gdn import GDN, GDNConfig, STATE_BINDINGS as GDN_BINDINGS, export_asset as export_gdn
from export_coreai_qsa import QwenQSA, BINDINGS as QSA_BINDINGS, initial_state as qsa_initial_state, export_asset as export_qsa

ROOT = Path(__file__).resolve().parents[1]
QSA_WEIGHTS = ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight", "q_norm.weight", "k_norm.weight",
               "indexer.index_qk_proj.weight", "indexer.q_layernorm.weight", "indexer.k_layernorm.weight")


def atomic_manifest(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def select_layers(value, count):
    if value is None:
        return list(range(count))
    try:
        selected = [int(part) for part in value.split(",")]
    except ValueError as error:
        raise ValueError("--layers must be comma-separated integer layer indices") from error
    if not selected or len(set(selected)) != len(selected) or any(not 0 <= layer < count for layer in selected):
        raise ValueError(f"--layers must contain unique indices between 0 and {count - 1}")
    return sorted(selected)


def state_metadata(state):
    result = {}
    for name, value in state.items():
        dtype = str(value.dtype).removeprefix("torch.")
        if dtype not in ("float16", "float32", "int32") or bool(torch.count_nonzero(value)):
            raise ValueError(f"Expected supported zero initial state: {name}")
        result[name] = {"shape": list(value.shape), "dtype": dtype, "fill": 0}
    return result


def prepare_layer(source, config, layer, capacity):
    kind = config["layer_types"][layer]
    if kind == "linear_attention":
        parameters = GDNConfig.from_model(config)
        source.prefix = f"language_model.model.layers.{layer}.linear_attn."
        weights = {name: source.read(name) for name in parameters.weight_shapes}
        model = GDN(parameters, weights).eval()
        state = {"conv_history": torch.zeros(1, parameters.kernel - 1, parameters.channels, dtype=torch.float16),
                 "recurrent_state": torch.zeros(1, parameters.value_heads, parameters.value_dim, parameters.key_dim, dtype=torch.float32)}
        return "gdn", model, state, GDN_BINDINGS, "hidden", "output"
    if kind == "full_attention":
        source.prefix = f"language_model.model.layers.{layer}.self_attn."
        weights = {name: source.read(name) for name in QSA_WEIGHTS}
        model = QwenQSA(config, weights, capacity).eval()
        return "qsa", model, qsa_initial_state(model), QSA_BINDINGS, "x", "y"
    raise ValueError(f"Unsupported config layer type at {layer}: {kind}")


def cpu_smoke(model, state, bindings, hidden, layer):
    """Two continuing synthetic activations; no claim of whole-model correctness."""
    generator = torch.Generator(device="cpu").manual_seed(20260917 + layer)
    inputs = (torch.randn(1, 2, hidden, generator=generator) * 0.125).half()
    current = {name: value.clone() for name, value in state.items()}
    maximum_output = 0.0
    with torch.inference_mode():
        for position in range(2):
            results = model(inputs[:, position:position+1], *current.values())
            if not all(bool(torch.isfinite(value).all()) for value in results):
                raise ValueError(f"Layer {layer}: nonfinite CPU smoke output")
            maximum_output = max(maximum_output, float(results[0].abs().max()))
            # Both source modules return all states directly after the activation,
            # in exactly the order used by their published state bindings.
            current = dict(zip(bindings, results[1:1 + len(bindings)]))
            if "offset" in current:
                if int(current["offset"].item()) != position + 1 or int(current["pooled_count"].item()) != (position + 1) // model.ratio:
                    raise ValueError(f"Layer {layer}: CPU attention position update differs")
    return {"steps": 2, "input": "seeded synthetic normal activations scaled by 0.125",
            "finite": True, "maximumAbsoluteOutput": maximum_output,
            "scope": "CPU equation/finite smoke only; no device or full-model acceptance"}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/coreai-hybrid-attention")
    parser.add_argument("--model-dir", type=Path, help="Must match the existing verified source manifest")
    parser.add_argument("--layers", help="Comma-separated smoke layer subset; omit for all configured layers")
    parser.add_argument("--capacity", type=int, default=256, help="Fixed attention cache capacity, default 256")
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    source = Source(0)
    if args.model_dir is not None and args.model_dir.resolve() != source.directory.resolve():
        raise ValueError("--model-dir must match the previously verified source manifest")
    config_path = source.directory / "config.json"
    full_config = json.loads(config_path.read_text())
    config = full_config["text_config"]
    layer_count = config["num_hidden_layers"]
    if layer_count != len(config["layer_types"]):
        raise ValueError("Config layer count and layer_types disagree")
    ratio = config["indexer_compress_ratio"]
    if not 4 <= args.capacity <= min(config["max_position_embeddings"], 4096) or args.capacity % ratio:
        raise ValueError(f"Initial hybrid capacity must be a multiple of {ratio} between 4 and 4096")
    selected = select_layers(args.layers, layer_count)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a fresh output directory; resume is unsupported: {output}")
    output.mkdir(parents=True)
    space = shutil.disk_usage(output)
    estimated_bytes = len(selected) * 120_000_000
    if space.free < estimated_bytes + 512 * 1024 * 1024:
        raise ValueError(f"Insufficient free disk space: need about {estimated_bytes} asset bytes plus 512 MiB margin")
    manifest = {"version": 1, "status": "exporting", "capacity": args.capacity, "tokenChunk": 1,
                "layerCount": layer_count, "exportedLayerCount": 0, "selectedLayers": selected,
                "completeModelLayerSet": False, "hiddenSize": config["hidden_size"],
                "modelDirectory": str(source.directory), "configSHA256": sha256_file(config_path),
                "sourceLayerTypes": config["layer_types"], "layers": [],
                "disk": {"freeBytesBefore": space.free, "estimatedAssetBytes": estimated_bytes},
                "authoringSources": [{"path": str(path), "sha256": sha256_file(path)} for path in (
                    Path(__file__).resolve(), Path(__file__).with_name("export_coreai_gdn.py").resolve(),
                    Path(__file__).with_name("export_coreai_qsa.py").resolve(), Path(__file__).with_name("export_moe.py").resolve())],
                "versions": {name: importlib.metadata.version(name) for name in ("torch", "coreai-core", "coreai-torch")},
                "limitations": ["All attention sublayers use CoreAI; residual/HC, MoE, embeddings, SSD PLE and logits remain in the existing runner.",
                                "S1 tokenwise prefill is a functional integration path, not a prefill performance optimization.",
                                "FP16 weights/activations and FP32 GDN state differ from original BF16 runner numerics.",
                                "The fixed attention capacity is an explicit integration limit; this does not establish 262K context support.",
                                "Caller must enforce nonnegative offsets and offset + 1 <= capacity before executing QSA.",
                                "Authoring and per-layer CPU smoke only; device and full-generation checks are separate."]}
    path = output / "manifest.json"
    atomic_manifest(path, manifest)
    started = time.perf_counter()
    try:
        for layer in selected:
            layer_started = time.perf_counter()
            source = Source(layer)
            kind, model, state, bindings, input_name, output_name = prepare_layer(source, config, layer, args.capacity)
            smoke = cpu_smoke(model, state, bindings, config["hidden_size"], layer)
            destination = output / f"layer-{layer:02d}-{kind}-s1.aimodel"
            if kind == "gdn":
                example = (torch.zeros(1, 1, config["hidden_size"], dtype=torch.float16), *state.values())
                files = export_gdn(model, example, destination)
            else:
                files = export_qsa(model, 1, destination)["files"]
            manifest["layers"].append({"index": layer, "kind": kind, "path": destination.name,
                "function": "main", "inputName": input_name, "outputName": output_name,
                "stateBindings": bindings, "initialState": state_metadata(state), "files": files,
                "modelBytes": sum(item["bytes"] for item in files), "sourceRecords": source.records,
                "cpuSmoke": smoke, "authoringSeconds": time.perf_counter() - layer_started})
            manifest["exportedLayerCount"] = len(manifest["layers"])
            atomic_manifest(path, manifest)
            print(f"Exported layer {layer:02d} {kind}: {manifest['layers'][-1]['modelBytes']} bytes, "
                  f"{manifest['layers'][-1]['authoringSeconds']:.2f}s", flush=True)
            del model, state
            gc.collect()
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        atomic_manifest(path, manifest)
        raise
    manifest["status"] = "complete"
    manifest["completeModelLayerSet"] = selected == list(range(layer_count))
    manifest["authoringSeconds"] = time.perf_counter() - started
    manifest["modelBytes"] = sum(layer["modelBytes"] for layer in manifest["layers"])
    atomic_manifest(path, manifest)
    print(f"Complete manifest: {path}", flush=True)


if __name__ == "__main__":
    main()
