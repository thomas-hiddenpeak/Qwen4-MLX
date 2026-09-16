#!/usr/bin/env python3
"""CoreAI custom Metal selected affine-Q4 GEMV, with a CPU authoring oracle.

This experimental kernel reads the original packed I16 view directly, preserving
FP32 affine arithmetic -> FP16 weight -> FP32 dot -> FP16 output. It never
materializes a selected dense weight matrix on the device. The Python callback
does materialize a selected matrix, solely for CPU reference and fake inference.

Integration (after creating a regular Q4MoE):
    for name in PROJECTIONS:
        setattr(moe, name, MetalPackedQ4.from_packed(getattr(moe, name)))
    converter.register_custom_kernels([get_q4_kernel()])

The converter registration must precede add_pytorch_module/add_exported_program.
Running this file authors a small synthetic asset; it runs no CoreAI device API.
The Metal API is experimental; device correctness and speed need separate tests.
"""
from __future__ import annotations

import argparse
from functools import cache
import json
from pathlib import Path

import numpy as np
import torch


# Metal tensor coordinates reverse the corresponding PyTorch dimension order:
# [batch, 1, input] -> x[input, 0, batch]. One simdgroup owns one output row.
# The body is original implementation, using Apple's documented tensor API:
# https://apple.github.io/coreai-torch/main/guides/custom-metal-kernels.html
Q4_METAL_SOURCE = r"""
const uint row = group.x * 4u + simd;
const uint batch = group.y;
const uint out_size = packed.get_extent(1);
const uint words = packed.get_extent(0);
if (row >= out_size) return;
const int expert = ids[batch];
if (expert < 0 || uint(expert) >= packed.get_extent(2)) {
    if (lane == 0u) output[row, 0, batch] = half(0.0f);
    return;
}
float accum = 0.0f;
for (uint word = lane; word < words; word += 32u) {
    // Conversion to ushort preserves the bit pattern of a negative I16 lane.
    const ushort bits = ushort(packed[word, row, expert]);
    const uint affine_group = word / 16u;
    const float scale = float(scales[affine_group, row, expert]);
    const float bias = float(biases[affine_group, row, expert]);
    for (uint nibble = 0u; nibble < 4u; ++nibble) {
        const uint code = (uint(bits) >> (4u * nibble)) & 15u;
        // Deliberate half boundary matches the existing PackedQ4 path.
        const half weight = half(scale * float(code) + bias);
        accum += float(x[word * 4u + nibble, 0, batch]) * float(weight);
    }
}
const float total = simd_sum(accum);
if (lane == 0u) output[row, 0, batch] = half(total);
"""


def selected_q4_reference(x: torch.Tensor, ids: torch.Tensor, packed: torch.Tensor,
                          scales: torch.Tensor, biases: torch.Tensor) -> torch.Tensor:
    """CPU/fake implementation. Caller guarantees ids are in the expert bank."""
    chosen = packed.index_select(0, ids.long()).to(torch.int32)
    shift = torch.arange(0, 16, 4, device=packed.device, dtype=torch.int32)
    codes = torch.bitwise_and(torch.bitwise_right_shift(chosen.unsqueeze(-1), shift), 15)
    codes = codes.reshape(ids.shape[0], packed.shape[1], -1, 64).float()
    scale = scales.index_select(0, ids.long()).float().unsqueeze(-1)
    bias = biases.index_select(0, ids.long()).float().unsqueeze(-1)
    dense = (codes * scale + bias).half().flatten(-2)
    return torch.matmul(x.float(), dense.float().transpose(-1, -2)).half()


@cache
def get_q4_kernel():
    """One custom-op registration shared by all model instances and dimensions."""
    from coreai.authoring import MetalParameter
    from coreai_torch import TorchMetalKernel
    return TorchMetalKernel(
        "qwen_affine_q4_selected_gemv_i16_v1",
        input_names=["x", "ids", "packed", "scales", "biases"],
        result_names=["output"], src=Q4_METAL_SOURCE,
        torch_defn=selected_q4_reference,
        metal_params=[
            MetalParameter("group", "uint3", "threadgroup_position_in_grid"),
            MetalParameter("lane", "uint", "thread_index_in_simdgroup"),
            MetalParameter("simd", "uint", "simdgroup_index_in_threadgroup"),
        ],
    )


class SelectedQ4Matmul(torch.nn.Module):
    """All-input variant: x[B,1,K], ids[B], bank[E,N,K/4], affine[E,N,K/64]."""
    def forward(self, x, ids, packed, scales, biases):
        batch, _, input_size = x.shape
        outputs = packed.shape[1]
        if (x.dtype != torch.float16 or ids.dtype != torch.int32 or
                packed.dtype != torch.int16 or scales.dtype != torch.float16 or
                biases.dtype != torch.float16):
            raise ValueError("Expected FP16 x/affine, I32 ids and original I16 packed bytes")
        if (x.shape[1] != 1 or ids.shape != (batch,) or input_size % 64 or
                packed.shape[2] * 4 != input_size or scales.shape != biases.shape or
                scales.shape != (packed.shape[0], outputs, input_size // 64)):
            raise ValueError("Expected matching affine-Q4 group64 selected GEMV shapes")
        return get_q4_kernel()(
            x, ids, packed, scales, biases,
            threads_per_grid=(((outputs + 3) // 4) * 128, batch, 1),
            threads_per_thread_group=(128, 1, 1),
            result_shapes=[[batch, 1, outputs]],
        )


class MetalPackedQ4(torch.nn.Module):
    """Drop-in selected projection with constant banks and no duplicate weights."""
    def __init__(self, packed, scales, biases):
        super().__init__()
        if packed.dtype != torch.int16 or packed.ndim != 3:
            raise ValueError("packed must be the original bank's rank-3 I16 byte view")
        if (scales.dtype != torch.float16 or biases.dtype != torch.float16 or
                packed.shape[-1] % 16 or scales.shape != biases.shape or
                scales.shape != (*packed.shape[:2], packed.shape[-1] // 16)):
            raise ValueError("Only FP16 affine parameters with original group64 layout are supported")
        self.input_size = packed.shape[-1] * 4
        self.group_size = 64
        self.register_buffer("packed", packed)
        self.register_buffer("scales", scales)
        self.register_buffer("biases", biases)
        self.projection = SelectedQ4Matmul()

    @classmethod
    def from_packed(cls, original):
        if original.group_size != 64:
            raise ValueError("Only affine group64 is supported")
        return cls(original.packed, original.scales, original.biases)

    def forward(self, x, ids):
        return self.projection(x, ids, self.packed, self.scales, self.biases)


def make_smoke(batch=4, input_size=128, output_size=7, experts=3, seed=1907):
    """Small non-square, signed-word, distinct-affine, repeated-ID smoke case."""
    if input_size <= 0 or input_size % 64 or min(batch, output_size, experts) <= 0:
        raise ValueError("Positive dimensions and a multiple-of-64 input are required")
    rng = np.random.default_rng(seed)
    words = rng.integers(0, 2**32, (experts, output_size, input_size // 8), dtype=np.uint32)
    words.reshape(-1)[:8] = [0, 0xffffffff, 0x80000000, 0x7fffffff,
                             0x01234567, 0x89abcdef, 0xfedcba98, 0x76543210]
    scales = torch.from_numpy(rng.uniform(0.003, 0.08, (experts, output_size, input_size // 64)).astype(np.float16))
    biases = (-scales.float() * torch.from_numpy(rng.uniform(4, 10, scales.shape).astype(np.float32))).half()
    model = MetalPackedQ4(torch.from_numpy(words.view(np.int16).copy()), scales, biases).eval()
    x = torch.from_numpy(rng.normal(0, 0.2, (batch, 1, input_size)).astype(np.float16))
    ids = torch.tensor([(index * 2 + 1) % experts for index in range(batch)], dtype=torch.int32)
    return model, x, ids


def export_smoke(output: Path, batch=4, input_size=128, output_size=7, experts=3):
    """CPU authoring only. Includes fixture and the exact Metal source for audit."""
    import coreai_torch
    from export_coreai_q4_moe import tensor_json
    from export_moe import sha256_file, write_json
    output.mkdir(parents=True, exist_ok=False)
    model, x, ids = make_smoke(batch, input_size, output_size, experts)
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    converter.register_custom_kernels([get_q4_kernel()])
    converter.add_pytorch_module(model, input_names=("x", "ids"), output_names=("output",),
        export_fn=lambda m: torch.export.export(m, args=(x, ids)).run_decompositions(coreai_torch.get_decomp_table()))
    program = converter.to_coreai()
    program.optimize()
    path = output / "q4.aimodel"
    program.save_asset(path)
    with torch.inference_mode():
        for name, activation in (("actual", x), ("zero", torch.zeros_like(x))):
            write_json(output / f"{name}.json", {
                "inputs": {"x": tensor_json(activation), "ids": tensor_json(ids)},
                "expectedOutputs": {"output": tensor_json(model(activation, ids))},
            })
    (output / "kernel.metal.txt").write_text(Q4_METAL_SOURCE)
    report = {"version": 1, "status": "cpu-authored-device-unvalidated", "model": "q4.aimodel",
              "function": "main", "batch": batch, "inputSize": input_size,
              "outputSize": output_size, "experts": experts, "groupSize": 64,
              "provenance": "synthetic weights and activations, original affine-Q4 packing",
              "numerics": "FP32 affine -> FP16 dequantized weight -> FP32 dot -> FP16 output",
              "threadsPerThreadgroup": [128, 1, 1], "fixtures": ["actual.json", "zero.json"],
              "tolerances": {"maximumAbsoluteError": 0.002, "relativeL2Error": 0.001},
              "assets": [{"path": str(p.relative_to(output)), "bytes": p.stat().st_size,
                          "sha256": sha256_file(p)} for p in sorted(path.rglob("*")) if p.is_file()]}
    write_json(output / "manifest.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--output-size", type=int, default=7)
    parser.add_argument("--experts", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(export_smoke(args.output, args.batch, args.input_size, args.output_size, args.experts), indent=2))


if __name__ == "__main__":
    main()
