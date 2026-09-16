#!/usr/bin/env python3
"""Optional shape-stable FP16-weight dense GEMV for CoreAI authoring.

Each token uses the same 32-lane FP32 dot/reduction for S1 and S4. This is a
diagnostic fallback for shape-dependent GEMM rounding, not a default performance
replacement. It is not bitwise equivalent to CPU/ordinary CoreAI GEMM, whose
reduction order may differ. Device chunk equivalence still requires validation.

Integration: use dense_linear(x, weight), then register get_dense_kernel() with
TorchConverter.register_custom_kernels before adding the module. Parameters and
activations must both be FP16, x[1,S,K], weight[N,K]; output is FP16[1,S,N].
"""
from __future__ import annotations

import argparse
from functools import cache
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# Original implementation; tensor indices reverse PyTorch's dimension order.
# API reference: https://apple.github.io/coreai-torch/main/guides/custom-metal-kernels.html
DENSE_METAL_SOURCE = r"""
const uint row = group.x * 4u + simd;
const uint token = group.y;
if (row >= weight.get_extent(1)) return;
float accum = 0.0f;
for (uint k = lane; k < weight.get_extent(0); k += 32u) {
    accum += float(x[k, token, 0]) * float(weight[k, row]);
}
const float total = simd_sum(accum);
if (lane == 0u) output[row, token, 0] = half(total);
"""


def dense_reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """CPU and fake authoring reference; BLAS reduction order may differ."""
    return F.linear(x.float(), weight.float()).half()


@cache
def get_dense_kernel():
    from coreai.authoring import MetalParameter
    from coreai_torch import TorchMetalKernel
    return TorchMetalKernel(
        "qwen_fp16_dense_gemv_stable_v1",
        input_names=["x", "weight"], result_names=["output"],
        src=DENSE_METAL_SOURCE, torch_defn=dense_reference,
        metal_params=[MetalParameter("group", "uint3", "threadgroup_position_in_grid"),
                      MetalParameter("lane", "uint", "thread_index_in_simdgroup"),
                      MetalParameter("simd", "uint", "simdgroup_index_in_threadgroup")],
    )


def dense_linear(x, weight):
    """Standalone drop-in for FP16-input/weight linear helpers, without bias."""
    if (x.dtype != torch.float16 or weight.dtype != torch.float16 or x.ndim != 3 or
            weight.ndim != 2 or x.shape[0] != 1 or x.shape[-1] != weight.shape[-1] or
            min(x.shape[1], x.shape[2], weight.shape[0]) <= 0):
        raise ValueError("Expected positive FP16 x[1,S,K] and FP16 weight[N,K]")
    count, outputs = x.shape[1], weight.shape[0]
    return get_dense_kernel()(
        x, weight,
        threads_per_grid=(((outputs + 3) // 4) * 128, count, 1),
        threads_per_thread_group=(128, 1, 1), result_shapes=[[1, count, outputs]],
    )


class MetalDenseLinear(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        if weight.ndim != 2 or weight.dtype != torch.float16:
            raise ValueError("Expected FP16 matrix weight")
        self.register_buffer("weight", weight)

    def forward(self, x):
        return dense_linear(x, self.weight)


def make_smoke(input_size=129, output_size=7, seed=4107):
    if min(input_size, output_size) <= 0:
        raise ValueError("Positive dimensions required")
    rng = np.random.default_rng(seed)
    weight = torch.from_numpy(rng.normal(0, 0.04, (output_size, input_size)).astype(np.float16))
    x = torch.from_numpy(rng.normal(0, 0.3, (1, 4, input_size)).astype(np.float16))
    return MetalDenseLinear(weight).eval(), x


def export_smoke(output: Path, input_size=129, output_size=7):
    """Author S1/S4 shared weights with CPU fixtures; no CoreAI runtime call."""
    import coreai_torch
    from export_coreai_q4_moe import tensor_json
    from export_moe import sha256_file, write_json
    output.mkdir(parents=True, exist_ok=False)
    model, x = make_smoke(input_size, output_size)
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    converter.register_custom_kernels([get_dense_kernel()])
    for name, example in (("main", x[:, :1].contiguous()), ("prefill", x)):
        converter.add_pytorch_module(model, entrypoint_name=name, input_names=("x",), output_names=("output",),
            export_fn=lambda m, example=example: torch.export.export(m, args=(example,)).run_decompositions(coreai_torch.get_decomp_table()))
    program = converter.to_coreai()
    program.optimize()
    asset = output / "dense.aimodel"
    program.save_asset(asset)
    cases = []
    with torch.inference_mode():
        for name, example, function in (("actual-s1", x[:, :1].contiguous(), "main"), ("actual-s4", x, "prefill"),
                                        ("zero-s1", torch.zeros_like(x[:, :1]), "main"),
                                        ("zero-s4", torch.zeros_like(x), "prefill")):
            fixture = output / f"{name}.json"
            write_json(fixture, {"inputs": {"x": tensor_json(example)},
                                 "expectedOutputs": {"output": tensor_json(dense_reference(example, model.weight))}})
            cases.append({"name": name, "function": function, "fixture": fixture.name})
        # Independent S1 calls for all S4 rows enable exact GPU chunk comparison.
        for token in range(4):
            example = x[:, token:token+1].contiguous()
            name = f"row-{token}-s1.json"
            write_json(output / name, {"inputs": {"x": tensor_json(example)},
                "expectedOutputs": {"output": tensor_json(dense_reference(example, model.weight))}})
        np.savez(output / "cpu-reference.npz", x=x.numpy(), weight=model.weight.numpy(),
                 output=dense_reference(x, model.weight).numpy())
    (output / "kernel.metal.txt").write_text(DENSE_METAL_SOURCE)
    report = {"version": 1, "status": "cpu-authored-device-unvalidated", "model": asset.name,
              "function": "main", "prefillFunction": "prefill", "inputSize": input_size, "outputSize": output_size,
              "cases": cases, "s1RowFixtures": [f"row-{token}-s1.json" for token in range(4)],
              "tolerances": {"maximumAbsoluteError": 0.002, "relativeL2Error": 0.001},
              "chunkEquivalenceTarget": "Bitwise equal GPU S4 rows and same-input S1 calls, measured separately from CPU tolerance",
              "provenance": "Synthetic FP16 dense weights and activations",
              "limitations": ["Optional diagnostic fallback, not enabled in the main exporter.",
                              "No device or speed claim; four tokens read the same weights independently.",
                              "The CPU BLAS oracle does not reproduce Metal's lane/reduction order."],
              "assets": [{"path": str(p.relative_to(output)), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
                         for p in sorted(asset.rglob("*")) if p.is_file()]}
    write_json(output / "manifest.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-size", type=int, default=129)
    parser.add_argument("--output-size", type=int, default=7)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(export_smoke(args.output, args.input_size, args.output_size), indent=2))


if __name__ == "__main__":
    main()
