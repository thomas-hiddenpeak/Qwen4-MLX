#!/usr/bin/env python3
"""CPU-author an optional real vocabulary head with FP32-output Metal GEMV.

No model execution, GPU invocation, build, or baseline asset mutation. The
baseline supplies phase shapes; source weights are read through the existing
verified Source reader. The saved stream is a reproducible diagnostic input,
not a claim of final-layer activation capture or generation acceptance.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F

from coreai_head_metal import HEAD_METAL_SOURCE, MetalHead, get_head_kernel
from export_coreai_dense import DenseConfig, Head, read_hc
from export_coreai_pd import LastHead, export_shared
from export_coreai_q4_moe import tensor_json
from export_moe import Source, sha256_file, write_json


def export_head(output, baseline, stream_path=None, smoke=False):
    if output.exists():
        raise ValueError(f"Use a fresh output directory: {output}")
    if shutil.disk_usage(output.parent).free < (50_000_000 if smoke else 3_000_000_000):
        raise ValueError("Insufficient disk space for separate head asset")
    output.mkdir(parents=True)
    start = time.perf_counter()
    if smoke:
        c = DenseConfig(129, 4, 7, 1e-6, 17, 2, 3, 2)
        rng = np.random.default_rng(8117)
        hc = {"input_mix_weight_down.weight": rng.normal(0, .04, (c.low_rank, c.width)),
              "input_mix_weight_up.weight": rng.normal(0, .04, (c.width, c.low_rank)),
              "hc_norm.weight": np.ones(c.width)}
        old = Head(c, hc, rng.normal(0, .04, (c.vocabulary, c.hidden))).eval()
        phase_sizes = {"main": 1, "prefill": 4}
        stream = torch.from_numpy(rng.normal(0, .3, (1, 4, c.width)).astype(np.float16))
        source_records, baseline_asset = [], None
    else:
        baseline = baseline.resolve()
        manifest = json.loads((baseline / "manifest.json").read_text())
        if manifest["status"] != "complete" or "head" not in manifest["assets"]:
            raise ValueError("Completed baseline with a head asset required")
        phase_sizes = {"main": 1, "prefill": manifest["tokenChunk"]}
        phase_sizes.update({f"prefill_s{n}": n for n in manifest.get("tailChunks", [])})
        source = Source(0)
        config_path = source.directory / "config.json"
        if sha256_file(config_path) != manifest["configSHA256"]:
            raise ValueError("Baseline config differs from verified weight source")
        c = DenseConfig.from_model(json.loads(config_path.read_text())["text_config"])
        hc = read_hc(source, "language_model.model.hyper_connection_mixer", with_injection=False)
        source.prefix = ""
        old = Head(c, hc, source.read("language_model.lm_head.weight")).eval()
        source_records = source.records
        baseline_asset = baseline / manifest["assets"]["head"]["path"]
        if stream_path is None:
            rng = np.random.default_rng(8117)
            stream = torch.from_numpy(rng.normal(0, .3, (1, manifest["tokenChunk"], c.width)).astype(np.float16))
        else:
            if stream_path.stat().st_size != manifest["tokenChunk"] * c.width * 2:
                raise ValueError("Stream fixture does not match baseline main prefill shape")
            stream = torch.from_numpy(np.fromfile(stream_path, dtype="<f2").reshape(1, manifest["tokenChunk"], c.width))
    module = LastHead(MetalHead(old)).eval()
    gc.collect()
    # All phase inputs end in exactly the same row, so same-device comparison
    # can require exact equality across S1 and larger LastHead entrypoints.
    examples = {name: {"stream": torch.zeros(1, n, c.width, dtype=torch.float16)}
                for name, n in phase_sizes.items()}
    report = {"version": 1, "status": "authoring", "phaseSizes": phase_sizes,
              "baselineAsset": str(baseline_asset) if baseline_asset else None,
              "precision": {"input": "float16", "weight": "float16", "accumulation": "float32", "logits": "float32"},
              "inputProvenance": "Reused diagnostic stream; not a captured final-layer model activation" if stream_path else "Deterministic synthetic stream",
              "sourceSlices": source_records,
              "tolerances": {"maximumAbsoluteError": 0.0005, "relativeL2Error": 0.00001},
              "chunkEquivalenceTarget": "Exact same-device logits across all LastHead phases using the same final input row",
              "limitations": ["Metal lane reduction order can differ from framework FP32 GEMM.",
                              "Runtime memory, speed and whole-model quality remain device-unvalidated.",
                              "HC mixer and its rounding are unchanged."]}
    write_json(output / "manifest.json", report)
    print("Authoring shared head functions: " + ", ".join(phase_sizes), flush=True)
    asset = export_shared(module, examples, ("logits",), output / "head.aimodel", [get_head_kernel()])
    print("Asset authored; computing bounded FP32 CPU reference", flush=True)
    with torch.inference_mode():
        last = stream[:, -1:].contiguous()
        mixed = old.mixer(last)
        # Keep only one <=16 MB FP32 slice of the vocabulary weight at a time.
        expected = torch.cat([F.linear(mixed.float(), old.weight[i:i+1024].float())
                              for i in range(0, c.vocabulary, 1024)], dim=-1)
        if not torch.isfinite(expected).all():
            raise ValueError("Nonfinite CPU head reference")
        expected.numpy().astype("<f4").tofile(output / "expected-logits.bin")
        mixed.numpy().astype("<f2").tofile(output / "mixed.bin")
        write_json(output / "actual-s1.json", {"inputs": {"stream": tensor_json(last)},
            "expectedOutputs": {"logits": tensor_json(expected)}})
        write_json(output / "zero-s1.json", {"inputs": {"stream": tensor_json(torch.zeros_like(last))},
            "expectedOutputs": {"logits": tensor_json(torch.zeros_like(expected))}})
    stream.numpy().astype("<f2").tofile(output / "stream.bin")
    specs = []
    for name, count in phase_sizes.items():
        # LastHead input is a contiguous suffix of the same stored stream.
        entry = {"file": "stream.bin", "offset": (stream.shape[1]-count)*c.width*2,
                 "bytes": count*c.width*2, "shape": [1,count,c.width], "dtype": "float16"}
        spec = {"asset": "head.aimodel", "function": name, "inputs": {"stream": entry},
                "output": f"device-candidate-{name}", "repeats": 6, "mapped": False}
        file = f"candidate-{name}-spec.json"
        write_json(output / file, spec)
        specs.append(file)
        if baseline_asset is not None and name in ("main", "prefill"):
            write_json(output / f"baseline-{name}-spec.json", {**spec, "asset": str(baseline_asset),
                "output": f"device-baseline-{name}"})
    (output / "kernel.metal.txt").write_text(HEAD_METAL_SOURCE)
    report.update(status="cpu-authored-device-unvalidated", model=asset, authoringSeconds=time.perf_counter()-start,
                  expectedLogits={"path":"expected-logits.bin", "shape":[1,1,c.vocabulary], "dtype":"float32",
                                  "sha256":sha256_file(output / "expected-logits.bin")},
                  projectionWeightBytes=old.weight.numel()*old.weight.element_size(),
                  specs=specs, sourceConfigSHA256=sha256_file(config_path) if not smoke else None)
    write_json(output / "manifest.json", report)
    (output / "README.md").write_text(
        "# FP32-output Metal head diagnostic\n\n"
        "CPU-authored only. The existing baseline asset is untouched.\n\n"
        "Use the already-built `../probe-external-swift` with a candidate spec and a baseline spec in separate processes. "
        "Compare `function_loaded` graphics/physical footprint, warm run seconds, and written FP32 `logits.bin` against "
        "`expected-logits.bin`. No FP16 logits boundary is introduced.\n\n"
        "Every candidate phase reads a suffix ending in the same stream row; logits across candidate phases should be "
        "bitwise equal. `actual-s1.json` and `zero-s1.json` also support the ordinary block probe. "
        "CPU reference uses chunked FP32 PyTorch linear after the unchanged HC mixer; it is not an exact Metal reduction oracle.\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-pd", type=Path)
    parser.add_argument("--stream-bin", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not args.smoke and args.baseline_pd is None:
        parser.error("--baseline-pd is required for real weights")
    torch.set_num_threads(2)
    print(json.dumps(export_head(args.output, args.baseline_pd, args.stream_bin, args.smoke), indent=2))


if __name__ == "__main__":
    main()
