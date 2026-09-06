#!/usr/bin/env python3
"""Official Core ML compression APIs on one real expert and shared expert.

All compressed variants are new approximations of the existing FP16 package,
not claims of lossless preservation of the original MLX Q4 representation.
"""
from __future__ import annotations
import argparse
from collections import Counter
import gc
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/moe-compression-v2"
if (OUTPUT / "deps").exists():
    sys.path.insert(0, str(OUTPUT / "deps"))

import coremltools as ct
import numpy as np
import torch
import export_moe as e

BASE = ROOT / "results/moe-export/layer_0_fp16_linear_s32"
FIXTURES = ROOT / "fixtures/moe-real/converted"


def output_group_lut8_public(weight):
    """Official CUSTOM LUT callback: exact unique values or weighted scalar KMeans.

    Small groups would make Core ML Tools choose sklearn despite kmeans1d being
    available. This callback applies the same scalar kmeans1d algorithm directly.
    """
    import kmeans1d
    values, inverse = np.unique(weight, return_inverse=True)
    lut = np.zeros(256, dtype=weight.dtype)
    if len(values) <= 256:
        lut[:len(values)] = values
        indices = inverse
    else:
        # The public kmeans1d package takes the original samples. Repeated
        # samples supply their multiplicity without relying on Core ML Tools'
        # private weighted-kmeans fork or its small-group sklearn fallback.
        clustered = kmeans1d.cluster(weight.reshape(-1), 256)
        lut[:] = clustered.centroids
        indices = clustered.clusters
    return lut, np.asarray(indices, dtype=np.uint8).reshape(-1)


def output_group_lut8(weight):
    """Same weighted scalar objective using Core ML Tools' bundled CPU helper.

    Only clustering uses this version-pinned internal numerical dependency.
    Compression is the public CUSTOM LUT API; no private ANE APIs are called.
    """
    from coremltools._deps import _kmeans1d
    values, inverse, counts = np.unique(weight, return_inverse=True, return_counts=True)
    lut = np.zeros(256, dtype=weight.dtype)
    if len(values) <= 256:
        lut[:len(values)] = values
        indices = inverse
    else:
        clustered = _kmeans1d.cluster(values, 256, weights=counts)
        lut[:] = clustered.centroids
        indices = np.asarray(clustered.clusters, dtype=np.uint8)[inverse]
    return lut, np.asarray(indices, dtype=np.uint8).reshape(-1)


def source_affine_model(storage_bits):
    """Control that changes only source-code storage width, not group geometry."""
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
    source = e.Source(0).expert(88)
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 2560, 1, 32), dtype=types.fp16)], opset_version=ct.target.macOS15)
    def program(x):
        value = mb.reshape(x=mb.transpose(x=x, perm=[0, 3, 2, 1]), shape=[1, 32, 2560])
        def linear(v, name):
            w = source[name]
            data = w["codes"].astype(np.uint8 if storage_bits == 8 else types.nptype_from_builtin(types.uint4))
            if np.any(w["scale32"] == 0):
                raise ValueError("Zero source scale requires a separate exact representation")
            weight = mb.constexpr_blockwise_shift_scale(data=data, scale=w["scale32"].astype(np.float16),
                        offset=(-w["bias32"] / w["scale32"]).astype(np.float16), name=name + "_source_affine")
            return mb.linear(x=v, weight=weight, name=name.replace("_proj", "_projection"))
        gate, up = linear(value, "gate_proj"), linear(value, "up_proj")
        activation = mb.real_div(x=gate, y=mb.add(x=mb.exp(x=mb.mul(x=gate, y=np.float16(-1))), y=np.float16(1)))
        y = linear(mb.mul(x=activation, y=up), "down_proj")
        return mb.transpose(x=mb.reshape(x=y, shape=[1, 32, 1, 2560]), perm=[0, 3, 2, 1], name="y")
    return ct.convert(program, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                      compute_units=ct.ComputeUnit.CPU_AND_NE, compute_precision=ct.precision.FLOAT16)


def configuration(label, weight_names):
    api = ct.optimize.coreml
    if label == "int8_channel":
        op = api.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int8", granularity="per_channel")
        transform = api.linear_quantize_weights
    elif label == "int4_channel":
        op = api.OpLinearQuantizerConfig(mode="linear_symmetric", dtype="int4", granularity="per_channel")
        transform = api.linear_quantize_weights
    elif label in ("int8_linear_block64", "int8_linear_block128"):
        op = api.OpLinearQuantizerConfig(mode="linear", dtype="int8", granularity="per_block",
                                         block_size=64 if label.endswith("64") else 128)
        transform = api.linear_quantize_weights
    elif label in ("lut4_tensor", "lut8_tensor"):
        op = api.OpPalettizerConfig(mode="kmeans", nbits=4 if label == "lut4_tensor" else 8,
                                   granularity="per_tensor")
        transform = api.palettize_weights
    elif label in ("lut4_group32", "lut8_group32"):
        op = api.OpPalettizerConfig(mode="kmeans", nbits=4 if label.startswith("lut4") else 8, granularity="per_grouped_channel",
                                   group_size=32, channel_axis=0, num_kmeans_workers=1)
        transform = api.palettize_weights
    elif label in ("lut8_group1", "lut8_group4", "lut8_group16"):
        op = api.OpPalettizerConfig(mode="custom", lut_function=output_group_lut8,
                                   granularity="per_grouped_channel", group_size=int(label.split("group")[1]),
                                   channel_axis=0)
        transform = api.palettize_weights
    else:
        raise ValueError(f"Unknown candidate {label}")
    # Select the actual three weight constants, excluding generated zero biases.
    config = api.OptimizationConfig(op_name_configs={name: op for name in weight_names})
    return transform, config, str(op)


def matrix_metadata(model):
    result = {}
    for name, meta in ct.optimize.coreml.get_weights_metadata(model, weight_threshold=10000).items():
        if meta.val.ndim != 2:
            continue
        consumers = [child.name for child in meta.child_ops if child.params_name_mapping.get("weight") == name]
        if len(consumers) != 1:
            raise ValueError(f"Ambiguous matrix consumer: {name}")
        result[consumers[0]] = {"name": name, "value": meta.val}
    if set(result) != {"gate_projection", "up_projection", "down_projection"}:
        raise ValueError(f"Expected complete SwiGLU matrices, found {set(result)}")
    return result


def torch_weights(metadata):
    return {projection.replace("_projection", "_proj"): {"dense16": value["value"].astype(np.float16),
                                                          "dense32": value["value"].astype(np.float32)}
            for projection, value in metadata.items()}


def predict(model, value):
    start = time.perf_counter_ns()
    result = model.predict({"x": value})["y"]
    return result, (time.perf_counter_ns() - start) / 1e6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="int8_channel,int4_channel,lut4_tensor,lut8_tensor,lut4_group32")
    parser.add_argument("--models", default="expert88,shared")
    parser.add_argument("--prepare-only", action="store_true", help="Compress/read plans and weights without any predictions")
    args = parser.parse_args()
    if any(name not in ("expert88", "shared") for name in args.models.split(",")):
        parser.error("Supported bounded models are expert88 and shared")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    report_path = OUTPUT / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {
        "macOS": platform.mac_ver()[0], "coremltools": ct.__version__, "variants": [],
        "evidence_scope": "Public API actual predictions and ComputePlan; no hardware trace, no full-model loading, no multi-run speed qualification.",
        "weight_semantics": "Each candidate requantizes the existing FP16-decoded package. It does not preserve the original MLX Q4 codes or exact source weights.",
        "official_sources": ["https://apple.github.io/coremltools/docs-guides/source/opt-quantization-api.html",
                             "https://apple.github.io/coremltools/docs-guides/source/opt-palettization-api.html"]}
    try:
        report["isolated_kmeans1d_version"] = importlib.metadata.version("kmeans1d")
    except importlib.metadata.PackageNotFoundError:
        report["isolated_kmeans1d_version"] = None
    inputs, sources = {}, {}
    for name in ("prefill", "decode"):
        path = FIXTURES / f"{name}.npz"
        source = dict(np.load(path))
        value = np.zeros((1, 2560, 1, 32), np.float16)
        value[:, :, 0, :source["x"].shape[1]] = source["x"].transpose(0, 2, 1)
        inputs[name], sources[name] = value, source
    for model_name in args.models.split(","):
        path = BASE / ("expert_0088.mlpackage" if model_name == "expert88" else "shared.mlpackage")
        start = time.perf_counter()
        original = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
        original_load = time.perf_counter() - start
        original_metadata = matrix_metadata(original)
        original_weights = torch_weights(original_metadata)
        baseline_predictions = {} if args.prepare_only else {name: predict(original, value) for name, value in inputs.items()}
        baseline_outputs = {name: result[0] for name, result in baseline_predictions.items()}
        baseline_math = {} if args.prepare_only else {name: e.reference(original_weights, value[..., :sources[name]["x"].shape[1]], "fp32_fp16_weights")
                                                      for name, value in inputs.items()}
        baseline_plan = e.compute_plan(original)
        for candidate in args.candidates.split(","):
            if candidate.endswith("_source_affine") and model_name != "expert88":
                continue
            row = {"model": model_name, "candidate": candidate, "source": str(path),
                   "source_package_bytes": e.package_bytes(path), "source_load_seconds": original_load,
                   "source_single_prediction_ms": {name: result[1] for name, result in baseline_predictions.items()},
                   "source_plan_counts": dict(Counter(p["preferred"] for p in baseline_plan))}
            if candidate in ("lut8_group1", "lut8_group4", "lut8_group16"):
                row["custom_lut_clustering"] = "Core ML Tools 9.0 bundled weighted kmeans1d CPU helper; same objective as public kmeans1d on repeated samples"
            directory = OUTPUT / candidate
            directory.mkdir(exist_ok=True)
            destination = directory / f"{model_name}.mlpackage"
            try:
                if candidate.endswith("_source_affine"):
                    storage_bits = 8 if candidate.startswith("uint8") else 4
                    description = f"Source affine group64 with original codes, scale FP16, offset FP16; uint{storage_bits} data storage. Offset conversion is not lossless."
                    transform = lambda _model, config: source_affine_model(storage_bits)
                    config = None
                else:
                    transform, config, description = configuration(candidate, [v["name"] for v in original_metadata.values()])
                row["configuration"] = description
                if not destination.exists():
                    start = time.perf_counter()
                    compressed = transform(original, config=config)
                    row["compression_and_initial_load_seconds"] = time.perf_counter() - start
                    compressed.save(str(destination))
                    del compressed
                else:
                    row["reused_existing_package"] = True
                start = time.perf_counter()
                model = ct.models.MLModel(str(destination), compute_units=ct.ComputeUnit.CPU_AND_NE)
                row["saved_package_load_seconds"] = time.perf_counter() - start
                compile_start = time.perf_counter()
                compiled_path = ct.models.utils.compile_model(str(destination))
                row["explicit_public_compile_seconds_after_initial_load"] = time.perf_counter() - compile_start
                load_start = time.perf_counter()
                compiled_model = ct.models.CompiledMLModel(compiled_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
                row["compiled_model_load_seconds"] = time.perf_counter() - load_start
                del compiled_model
                row["package"] = str(destination)
                row["package_bytes"] = e.package_bytes(destination)
                row["source_over_compressed_size"] = row["source_package_bytes"] / row["package_bytes"]
                plan = e.compute_plan(model)
                row["compute_plan"] = plan
                row["runtime_plan_counts"] = dict(Counter(p["preferred"] for p in plan if "constexpr" not in p["op"]))
                start = time.perf_counter()
                decompressed = ct.optimize.coreml.decompress_weights(model)
                row["public_decompress_seconds"] = time.perf_counter() - start
                reconstructed = matrix_metadata(decompressed)
                row["weight_errors"] = {name: e.errors(reconstructed[name]["value"], original_metadata[name]["value"])
                                        for name in original_metadata}
                weights = torch_weights(reconstructed)
                row["cases"] = []
                for name, value in (() if args.prepare_only else inputs.items()):
                    valid = sources[name]["x"].shape[1]
                    y, first_ms = predict(model, value)
                    _, warm_ms = predict(model, value)
                    decoded_y, _ = predict(decompressed, value)
                    math = e.reference(weights, value[..., :valid], "fp32_fp16_weights")
                    result = {"name": name, "valid_tokens": valid,
                              "fixture_sha256": e.sha256_file(FIXTURES / f"{name}.npz"),
                              "first_prediction_ms": first_ms, "single_warm_prediction_ms": warm_ms,
                              "output_vs_original_fp16_package": e.errors(y[..., :valid], baseline_outputs[name][..., :valid]),
                              "output_vs_decompressed_same_weights": e.errors(y[..., :valid], decoded_y[..., :valid]),
                              "output_vs_fp32_math_of_reconstructed_weights": e.errors(y[..., :valid], math),
                              "weight_change_only_fp32_math": e.errors(math, baseline_math[name]),
                              "output_vs_original_weight_fp32_math": e.errors(y[..., :valid], baseline_math[name]),
                              "padded_outputs_exact_zero": bool(np.count_nonzero(y[..., valid:]) == 0)}
                    if model_name == "shared":
                        result["output_vs_native_shared_down"] = e.errors(y[:, :, 0, :valid].transpose(0, 2, 1), sources[name]["shared_down"])
                    row["cases"].append(result)
                    np.savez(directory / f"{model_name}_{name}.npz", original=baseline_outputs[name][..., :valid],
                             compressed=y[..., :valid], decompressed=decoded_y[..., :valid],
                             original_fp32=baseline_math[name], reconstructed_fp32=math)
                row["status"] = "prepared_without_predictions" if args.prepare_only else "predicted"
                del model, decompressed, weights, reconstructed
            except Exception as error:
                row["status"] = "failed"
                row["error"] = repr(error)
            report["variants"].append(row)
            e.write_json(report_path, report)
            print(json.dumps({k: row.get(k) for k in ("model", "candidate", "status", "package_bytes", "runtime_plan_counts", "error")}), flush=True)
            gc.collect()
        del original, original_metadata
        gc.collect()


if __name__ == "__main__":
    main()
