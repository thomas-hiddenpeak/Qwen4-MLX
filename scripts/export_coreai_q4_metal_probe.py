#!/usr/bin/env python3
"""CPU-author one real full-bank MoE with custom Q4 Metal S1/S4 entrypoints.

The default fixtures use captured layer-0 MoE input followed by three explicitly
synthetic deterministic activation variations. They are not a four-token model
capture. No device runtime is called, and no source exporter is modified.
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
from pathlib import Path
import resource
import shutil
import time

import torch

from coreai_q4_metal import MetalPackedQ4, Q4_METAL_SOURCE, get_q4_kernel
from export_coreai_q4_moe import DIAGNOSTIC_NAMES, OUTPUT_NAMES, PROJECTIONS, load_layer, tensor_json
from export_moe import Source, sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/coreai-pd/q4-metal-layer0")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    if args.output.exists():
        raise FileExistsError("Use a fresh output directory")
    args.output.mkdir(parents=True)
    if shutil.disk_usage(args.output).free < 4_000_000_000:
        raise ValueError("At least 4 GB free space is required")
    started = time.perf_counter()
    source = Source(0)
    config_path = source.directory / "config.json"
    config = json.loads(config_path.read_text())["text_config"]
    capture_path = ROOT / "fixtures/moe-real/converted/decode.json"
    captured = json.loads(capture_path.read_text())["inputs"]["x"]
    actual = torch.tensor(captured["values"], dtype=torch.float16).reshape(captured["shape"])[:, :1].contiguous()
    if actual.shape != (1, 1, config["hidden_size"]):
        raise ValueError("Unexpected captured MoE input shape")
    # Purely synthetic changes of activation; no claim of source-model sequence.
    column = torch.arange(config["hidden_size"], dtype=torch.float32).reshape(1, 1, -1)
    variations = [actual]
    for offset in range(1, 4):
        perturbation = ((column + offset * 3) % 17 - 8) * (0.00025 * offset)
        variations.append((actual.float() * (1.0 - 0.01 * offset) + perturbation).half())
    prefill = torch.cat(variations, dim=1)
    names = OUTPUT_NAMES + DIAGNOSTIC_NAMES
    report = {"version": 1, "status": "exporting", "layer": 0,
              "modelDirectory": str(source.directory), "configSHA256": sha256_file(config_path),
              "completeExpertBank": True, "expertCount": config["num_experts"],
              "topK": config["num_experts_per_tok"], "tokenChunk": 4,
              "inputName": "x", "outputNames": list(names), "diagnostics": True,
              "packedFormat": "Original affine group64 Q4 bytes reinterpreted as signed I16, no repacking or requantization",
              "numerics": "Existing Q4MoE routing/shared/reduction unchanged; fused selected projections preserve FP16 dequantized weight and FP32 dot",
              "capture": {"path": str(capture_path), "sha256": sha256_file(capture_path),
                          "scope": "First S4 row is actual layer0 MoE capture; next three rows deterministic synthetic variations"},
              "tolerances": {"maximumAbsoluteError": 0.02, "relativeL2Error": 0.005,
                             "selectedIDsExact": True},
              "sourceScripts": {path.name: sha256_file(path) for path in
                                [Path(__file__), Path(__file__).with_name("coreai_q4_metal.py"),
                                 Path(__file__).with_name("export_coreai_q4_moe.py")]},
              "versions": {name: importlib.metadata.version(name) for name in ("torch", "coreai-core", "coreai-torch")},
              "cases": [], "deviceValidated": False,
              "limitations": ["CPU export and reference only; GPU numerical/performance testing remains separate.",
                              "The source BF16 model is not exactly equivalent to these FP16 operation boundaries.",
                              "Custom Metal may introduce CoreAI layout/copy constraints; actual overhead requires device measurement."]}

    def publish():
        temporary = args.output / "manifest.json.tmp"
        write_json(temporary, report)
        temporary.replace(args.output / "manifest.json")

    publish()
    try:
        print("Loading real layer0 full512 MoE banks on CPU", flush=True)
        model = load_layer(source, config["num_experts"], config["num_experts_per_tok"], diagnostics=True)
        cases = [("actual-s1", "main", actual), ("actual-s4", "prefill", prefill),
                 ("zero-s1", "main", torch.zeros_like(actual)),
                 ("zero-s4", "prefill", torch.zeros_like(prefill))]
        cpu_references = []
        for label, function, x in cases:
            with torch.inference_mode():
                expected = model(x)
            if not all(torch.isfinite(value).all() for value in expected):
                raise ValueError("Nonfinite CPU reference")
            cpu_references.append(expected)
            fixture_path = args.output / f"{label}.json"
            write_json(fixture_path, {"inputs": {"x": tensor_json(x)},
                                     "expectedOutputs": {name: tensor_json(value) for name, value in zip(names, expected)}})
            report["cases"].append({"name": label, "function": function, "fixture": fixture_path.name,
                                    "sha256": sha256_file(fixture_path),
                                    "selectedIDs": expected[1].reshape(-1).tolist()})
            print(f"CPU baseline fixture: {label}", flush=True)
        for name in PROJECTIONS:
            original = getattr(model, name)
            replacement = MetalPackedQ4.from_packed(original)
            if any(getattr(original, field).data_ptr() != getattr(replacement, field).data_ptr()
                   for field in ("packed", "scales", "biases")):
                raise AssertionError("Custom projection unexpectedly copied/repacked a bank")
            setattr(model, name, replacement)
        del original, replacement
        for (label, function, x), expected in zip(cases, cpu_references):
            with torch.inference_mode():
                result = model(x)
            for name, actual_value, expected_value in zip(names, result, expected):
                torch.testing.assert_close(actual_value, expected_value, atol=0, rtol=0,
                                           msg=f"CPU projection replacement changed {label}/{name}")
        report["cpuReplacementMatchesOldPathExactly"] = True
        report["sourceRecords"] = source.records
        publish()
        del cpu_references, expected, result
        gc.collect()
        print("Authoring shared main S1 and prefill S4 asset", flush=True)
        import coreai_torch
        converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
        converter.register_custom_kernels([get_q4_kernel()])
        for name, x in (("main", actual), ("prefill", prefill)):
            converter.add_pytorch_module(model, entrypoint_name=name, input_names=("x",), output_names=names,
                export_fn=lambda m, x=x: torch.export.export(m, args=(x,)).run_decompositions(coreai_torch.get_decomp_table()))
        program = converter.to_coreai()
        program.optimize()
        asset = args.output / "layer-00-moe-q4-metal.aimodel"
        program.save_asset(asset)
        files = [{"path": str(p.relative_to(args.output)), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
                 for p in sorted(asset.rglob("*")) if p.is_file()]
        (args.output / "kernel.metal.txt").write_text(Q4_METAL_SOURCE)
        report.update({"status": "complete", "model": asset.name, "function": "main", "prefillFunction": "prefill",
                       "modelBytes": sum(item["bytes"] for item in files), "assets": files,
                       "authoringSeconds": time.perf_counter() - started,
                       "peakRSSBytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        publish()
        print(json.dumps({key: report[key] for key in ("status", "modelBytes", "authoringSeconds", "peakRSSBytes")}), flush=True)
    except Exception as error:
        report.update({"status": "failed", "error": repr(error)})
        publish()
        raise


if __name__ == "__main__":
    main()
