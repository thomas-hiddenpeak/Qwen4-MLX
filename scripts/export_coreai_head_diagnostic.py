#!/usr/bin/env python3
"""CPU-author one shared-weight asset exposing HC mixed and final head logits.

The baseline and candidate functions share learned buffers and S1 input. Adding
mixed as an explicit output may constrain the compiler differently; agreement
in this diagnostic alone does not prove unchanged single-output head behavior.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil

import numpy as np
import torch
import torch.nn.functional as F

from coreai_head_metal import get_head_kernel, head_linear
from export_coreai_dense import DenseConfig, HCRead, read_hc
from export_coreai_head_projection import audit_fixtures
from export_coreai_q4_moe import tensor_json
from export_moe import Source, sha256_file, write_json


class HeadDiagnostic(torch.nn.Module):
    def __init__(self, mixer, weight, *, metal):
        super().__init__()
        self.mixer = mixer
        self.register_buffer("weight", weight)
        self.metal = metal

    def forward(self, stream):
        mixed = self.mixer(stream[:, -1:, :])
        logits = head_linear(mixed, self.weight) if self.metal else F.linear(mixed.float(), self.weight.float())
        return mixed, logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.output.exists():
        raise ValueError("Use a fresh output directory")
    if shutil.disk_usage(args.output.parent).free < 2_000_000_000:
        raise ValueError("At least 2 GB free required")
    audit = audit_fixtures(args.head_fixtures)
    original = json.loads((args.head_fixtures / "manifest.json").read_text())
    source = Source(0)
    config_path = source.directory / "config.json"
    if sha256_file(config_path) != original["sourceConfigSHA256"]:
        raise ValueError("Different source config")
    c = DenseConfig.from_model(json.loads(config_path.read_text())["text_config"])
    mixer = HCRead(c, read_hc(source, "language_model.model.hyper_connection_mixer", False), False).eval()
    source.prefix = ""
    weight = torch.from_numpy(source.read("language_model.lm_head.weight")).half()
    if source.records != original["sourceSlices"]:
        raise ValueError("Source weights differ from original full-head test")
    gc.collect()
    fixture = json.loads((args.head_fixtures / "actual-s1.json").read_text())
    record = fixture["inputs"]["stream"]
    stream = torch.tensor(record["values"], dtype=torch.float16).reshape(record["shape"])
    cpu_mixed = torch.from_numpy(np.fromfile(args.head_fixtures / "mixed.bin", dtype="<f2").reshape(1, 1, c.hidden))
    cpu_logits = torch.from_numpy(np.fromfile(args.head_fixtures / "expected-logits.bin", dtype="<f4").reshape(1, 1, c.vocabulary))
    with torch.inference_mode():
        torch.testing.assert_close(mixer(stream), cpu_mixed, atol=0, rtol=0)
    args.output.mkdir(parents=True)
    report = {"version":1, "status":"authoring", "purpose":"Expose mixed and logits without claiming identical compiler decisions to the original one-output head", "sourceSlices":source.records,
              "functions":{"baseline":"FP32 F.linear", "candidate":"FP16-weight Metal GEMV with FP32 output"},
              "inputSpecs":{"stream":{"shape":list(stream.shape), "dtype":"float16"}},
              "outputSpecs":{"mixed":{"shape":list(cpu_mixed.shape), "dtype":"float16"}, "logits":{"shape":list(cpu_logits.shape), "dtype":"float32"}},
              "tolerances":original["tolerances"], "cpuMixedMatchesOriginalFixture":True}
    write_json(args.output / "manifest.json", report)
    write_json(args.output / "fixture-audit.json", audit)
    import coreai_torch
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    converter.register_custom_kernels([get_head_kernel()])
    modules = {name:HeadDiagnostic(mixer, weight, metal=(name == "candidate")).eval()
               for name in ("baseline", "candidate")}
    for name, module in modules.items():
        converter.add_pytorch_module(module, entrypoint_name=name, input_names=("stream",), output_names=("mixed", "logits"),
            export_fn=lambda m:torch.export.export(m, (stream,)).run_decompositions(coreai_torch.get_decomp_table()))
    program = converter.to_coreai()
    program.optimize()
    asset = args.output / "head-diagnostic.aimodel"
    program.save_asset(asset)
    files = [{"path":str(p.relative_to(asset)), "bytes":p.stat().st_size, "sha256":sha256_file(p)}
             for p in sorted(asset.rglob("*")) if p.is_file()]
    stream.numpy().astype("<f2").tofile(args.output / "stream.bin")
    cpu_mixed.numpy().astype("<f2").tofile(args.output / "expected-mixed.bin")
    cpu_logits.numpy().astype("<f4").tofile(args.output / "expected-logits.bin")
    write_json(args.output / "actual.json", {"inputs":{"stream":tensor_json(stream)},
        "expectedOutputs":{"mixed":tensor_json(cpu_mixed),"logits":tensor_json(cpu_logits)}})
    for name in modules:
        write_json(args.output / f"{name}-spec.json", {"asset":asset.name, "function":name,
            "inputs":{"stream":{"file":"stream.bin", "offset":0, "bytes":stream.numel()*2, "shape":list(stream.shape), "dtype":"float16"}},
            "output":f"device-{name}", "repeats":6, "mapped":False})
    report.update(status="cpu-authored-device-unvalidated", model={"path":asset.name, "modelBytes":sum(p["bytes"] for p in files), "files":files})
    write_json(args.output / "manifest.json", report)
    (args.output / "README.md").write_text(
        "# Exposed HC boundary diagnostic\n\n"
        "Run baseline-spec.json and candidate-spec.json using the standalone Swift probe. Compare both "
        "mixed.bin (FP16) and logits.bin (FP32) against each other and expected-* files. "
        "Also compare these logits with the original one-output full-head device results.\n\n"
        "The functions share source buffers and a single asset. The extra mixed output intentionally changes "
        "graph observability. Agreement only here would narrow the issue to graph-context treatment; "
        "it would not identify a particular compiler pass or retroactively pass the original head.\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
