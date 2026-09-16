#!/usr/bin/env python3
"""CPU-author larger embedding/head assets beside a completed PD baseline.

Only new embedding-sN.aimodel, head-sN.aimodel and assets-sN.json paths are
written. Existing manifests/assets are untouched. Embedding keeps the original
math; head retains HC and LastHead with the optional FP32-output Metal GEMV.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import time

import torch

from coreai_head_metal import MetalHead, get_head_kernel
from export_coreai_dense import DenseConfig, Embedding, Head, read_hc
from export_coreai_pd import LastHead, export_shared
from export_moe import Source, sha256_file, write_json


TAILS = (4, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)


def export_top_group(directory, phases, c, *, suffix="", components=("embedding", "head"), metal_head=True):
    """Author requested top components; caller owns config validation/thread limit.

    ``phases`` maps entrypoint names to token counts. ``suffix='-s4096'`` gives
    embedding-s4096.aimodel/head-s4096.aimodel; an empty suffix gives the normal
    asset names. ``metal_head=False`` preserves the original FP32 F.linear head.
    Existing assets are never replaced. This helper does not write a manifest.
    """
    directory = Path(directory)
    components = tuple(components)
    if (not components or len(set(components)) != len(components) or
            any(name not in ("embedding", "head") for name in components)):
        raise ValueError("Expected distinct embedding/head components")
    if not phases or any(not isinstance(n, int) or n <= 0 for n in phases.values()):
        raise ValueError("Positive integer phase sizes required")
    if phases.get("main") != 1 or "prefill" not in phases:
        raise ValueError("Top assets require main S1 and a prefill phase")
    if any(ch in suffix for ch in ("/", "\\")):
        raise ValueError("Asset suffix must not contain a path separator")
    for component in components:
        path = directory / f"{component}{suffix}.aimodel"
        if path.exists():
            raise ValueError(f"Refusing to replace existing asset: {path}")
    directory.mkdir(parents=True, exist_ok=True)
    result = {"assets":{}, "sourceSlices":{}}
    for component in components:
        print(f"Authoring {component}{suffix} with {len(phases)} functions", flush=True)
        source = Source(0)
        if component == "embedding":
            source.prefix = ""
            module = Embedding(c, source.read("language_model.model.embed_tokens.weight")).eval()
            examples = {name:{"token":torch.zeros(n, dtype=torch.int32)} for name,n in phases.items()}
            outputs, kernels = ("stream",), ()
        else:
            hc = read_hc(source, "language_model.model.hyper_connection_mixer", with_injection=False)
            source.prefix = ""
            original = Head(c, hc, source.read("language_model.lm_head.weight")).eval()
            module = LastHead(MetalHead(original) if metal_head else original).eval()
            del original, hc
            examples = {name:{"stream":torch.zeros(1,n,c.width,dtype=torch.float16)} for name,n in phases.items()}
            outputs, kernels = ("logits",), (get_head_kernel(),) if metal_head else ()
        asset = export_shared(module, examples, outputs, directory / f"{component}{suffix}.aimodel", kernels)
        result["assets"][component] = asset
        result["sourceSlices"][component] = source.records
        print(f"{component}{suffix} complete, {asset['modelBytes']} bytes", flush=True)
        del module, examples, source
        gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-pd", type=Path, required=True)
    parser.add_argument("--chunks", type=int, nargs="+", default=[4096, 8192], choices=(4096, 8192))
    args = parser.parse_args()
    torch.set_num_threads(2)
    directory = args.baseline_pd.resolve()
    if len(set(args.chunks)) != len(args.chunks):
        raise ValueError("Duplicate chunk sizes")
    baseline = json.loads((directory / "manifest.json").read_text())
    if baseline["status"] != "complete":
        raise ValueError("Completed baseline required")
    for count in args.chunks:
        for name in (f"embedding-s{count}.aimodel", f"head-s{count}.aimodel", f"assets-s{count}.json"):
            if (directory / name).exists():
                raise ValueError(f"Refusing to replace existing output: {directory / name}")
    if shutil.disk_usage(directory).free < len(args.chunks) * 2_600_000_000 + 1_000_000_000:
        raise ValueError("Insufficient free space for separate assets")
    source = Source(0)
    config_path = source.directory / "config.json"
    config_sha = sha256_file(config_path)
    if config_sha != baseline["configSHA256"]:
        raise ValueError("Source config differs from baseline")
    c = DenseConfig.from_model(json.loads(config_path.read_text())["text_config"])
    del source
    for count in args.chunks:
        started = time.perf_counter()
        tails = [n for n in TAILS if n < count]
        phases = {"main":1, "prefill":count, **{f"prefill_s{n}":n for n in tails}}
        overlay = {"version":1, "status":"authoring", "tokenChunk":count, "tailChunks":tails,
                   "phaseSizes":phases, "configSHA256":config_sha,
                   "baselineManifestSHA256":sha256_file(directory / "manifest.json"),
                   "headProjection":"metal-fp16-weights-fp32-logits", "assets":{}, "sourceSlices":{},
                   "limitations":["CPU-authored only; device validation is separate.",
                                  "This is an asset overlay, not a complete model manifest.",
                                  "The original full-head numerical acceptance boundary remains unchanged."]}
        overlay.update(export_top_group(directory, phases, c, suffix=f"-s{count}"))
        overlay.update(status="complete", modelBytes=sum(item["modelBytes"] for item in overlay["assets"].values()),
                       authoringSeconds=time.perf_counter()-started)
        path = directory / f"assets-s{count}.json"
        write_json(path, overlay)
        print(f"READY {path}: {overlay['modelBytes']} bytes, {overlay['authoringSeconds']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
