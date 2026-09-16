#!/usr/bin/env python3
"""CPU author a CoreAI custom dense GEMM using Metal 4 MPP matmul2d.

The device operation is TensorOps FP16 x FP16 with a cooperative FP32 result,
followed by one FP16 rounding. This is not a hand-written SIMD dot-product.
TorchMetalKernel generates the Metal tensor signature and MPP include. No GPU
API is used by this script; successful asset authoring is not device validation.

References:
https://developer.apple.com/videos/play/wwdc2026/330/
https://developer.apple.com/documentation/metal/running-inline-ml-operations-in-a-shader-with-metal-4
Local SDK: MetalPerformancePrimitives/MPPTensorOpsMatMul2d.h
"""
from __future__ import annotations

import argparse
from functools import cache
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


MPP_METAL_BODY = r"""
// PyTorch x[S,K], weight[N,K], output[S,N] become Metal tensors[K,S],
// [K,N], [N,S]. Slices keep the original tensor bounds for edge checking.
const int start_m = int(group.y) * TILE_M;
const int start_n = int(group.x) * TILE_N;
auto left = x.slice(0, start_m);
auto right = weight.slice(0, start_n);
constexpr auto descriptor = matmul2d_descriptor(
    TILE_M, TILE_N, static_cast<int>(dynamic_extent),
    false, true, false, matmul2d_descriptor::mode::multiply);
matmul2d<descriptor, execution_simdgroups<4>> operation;
auto accum = operation.get_destination_cooperative_tensor<
    decltype(left), decltype(right), float>();
operation.run(left, right, accum);

// cooperative<float>.store(halfTensor) is rejected by this SDK. Convert only
// the completed FP32 result; multiplication/accumulation remain inside MPP.
#pragma clang loop unroll(full)
for (uint16_t element = 0; element < accum.get_capacity(); ++element) {
    if (accum.is_valid_element(element)) {
        const auto coordinate = accum.get_multidimensional_index(element);
        const int column = start_n + int(coordinate[0]);
        const int row = start_m + int(coordinate[1]);
        if (column < int(output.get_extent(0)) && row < int(output.get_extent(1))) {
            output[column, row] = half(accum[element]);
        }
    }
}
"""


def matrix_reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """CPU authoring oracle; its reduction order is not a GPU accuracy claim."""
    return F.linear(x.float(), weight.float()).half()


@cache
def get_tensor_kernel(tile_m: int = 32, tile_n: int = 64):
    from coreai.authoring import MetalParameter
    from coreai_torch import TorchMetalKernel
    if tile_m not in (16, 32, 64) or tile_n not in (32, 64):
        raise ValueError("Use tile M=16/32/64 and N=32/64 for this bounded candidate")
    body = MPP_METAL_BODY.replace("TILE_M", str(tile_m)).replace("TILE_N", str(tile_n))
    return TorchMetalKernel(
        f"qwen_mpp_fp16_gemm_m{tile_m}_n{tile_n}_v2",
        input_names=["x", "weight"], result_names=["output"],
        src=body, torch_defn=matrix_reference,
        metal_params=[MetalParameter("group", "uint3", "threadgroup_position_in_grid")],
    )


def tensor_linear(x, weight, tile_m=32, tile_n=64):
    """FP16 x[1,S,K] @ weight[N,K].T -> FP16[1,S,N], no bias/padding."""
    if (x.dtype != torch.float16 or weight.dtype != torch.float16 or x.ndim != 3 or
            weight.ndim != 2 or x.shape[0] != 1 or x.shape[-1] != weight.shape[-1] or
            min(x.shape[1], x.shape[2], weight.shape[0]) <= 0):
        raise ValueError("Expected positive FP16 x[1,S,K] and weight[N,K]")
    count, outputs = x.shape[1], weight.shape[0]
    result = get_tensor_kernel(tile_m, tile_n)(
        x.reshape(count, x.shape[2]), weight,
        threads_per_grid=(((outputs + tile_n - 1) // tile_n) * 128,
                          (count + tile_m - 1) // tile_m, 1),
        threads_per_thread_group=(128, 1, 1), result_shapes=[[count, outputs]],
    )
    return result.reshape(1, count, outputs)


class TensorLinear(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, tile_m=32, tile_n=64):
        super().__init__()
        if weight.ndim != 2 or weight.dtype != torch.float16 or min(weight.shape) <= 0:
            raise ValueError("Expected a nonempty FP16 weight matrix")
        self.register_buffer("weight", weight)
        self.tile_m, self.tile_n = tile_m, tile_n

    def forward(self, x):
        return tensor_linear(x, self.weight, self.tile_m, self.tile_n)


def make_example(count, input_size, output_size, seed=927330):
    if min(count, input_size, output_size) <= 0:
        raise ValueError("Positive dimensions required")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    weight = (torch.randn(output_size, input_size, generator=generator) * 0.04).half()
    x = (torch.randn(1, count, input_size, generator=generator) * 0.3).half()
    return weight, x


def tensor_json(tensor):
    value = tensor.detach().cpu().contiguous()
    return {"shape": list(value.shape), "dtype": str(value.dtype).removeprefix("torch."),
            "values": value.flatten().float().tolist()}


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def export_smoke(output: Path, *, counts=(5, 128, 256), input_size=128,
                 output_size=67, tile_m=32, tile_n=64):
    """Export static entrypoints sharing one matrix; every oracle runs on CPU."""
    import coreai_torch
    if not counts or len(set(counts)) != len(counts) or min(counts) <= 0:
        raise ValueError("Sequence lengths must be unique positive integers")
    kernel = get_tensor_kernel(tile_m, tile_n)
    weight, longest = make_example(max(counts), input_size, output_size)
    model = TensorLinear(weight, tile_m, tile_n).eval()
    output.mkdir(parents=True, exist_ok=False)
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    converter.register_custom_kernels([kernel])
    cases = []
    with torch.inference_mode():
        for index, count in enumerate(counts):
            name = "main" if index == 0 else f"s{count}"
            example = longest[:, :count].contiguous()
            converter.add_pytorch_module(model, entrypoint_name=name, input_names=("x",), output_names=("output",),
                export_fn=lambda m, example=example: torch.export.export(m, args=(example,)).run_decompositions(coreai_torch.get_decomp_table()))
            expected = matrix_reference(example, weight)
            np.savez(output / f"reference-s{count}.npz", x=example.numpy(), weight=weight.numpy(), output=expected.numpy())
            for label, value in (("random", example), ("zero", torch.zeros_like(example))):
                fixture = f"{label}-s{count}.json"
                _write_json(output / fixture, {"inputs": {"x": tensor_json(value)},
                    "expectedOutputs": {"output": tensor_json(matrix_reference(value, weight))}})
                cases.append({"name": f"{label}-s{count}", "function": name, "fixture": fixture,
                              "shape": [count, input_size, output_size],
                              "flops": 2 * count * input_size * output_size})
    program = converter.to_coreai()
    program.optimize()
    path = output / "tensor-matmul.aimodel"
    program.save_asset(path)
    sources = []
    for kernel_id, source in kernel.kernel_cache.values():
        source_name = f"{kernel_id}.metal"
        (output / source_name).write_text(source)
        sources.append(source_name)
    report = {"version": 1, "status": "cpu-authored-device-unvalidated",
              "model": path.name, "inputSize": input_size, "outputSize": output_size,
              "sequenceLengths": list(counts), "tile": [tile_m, tile_n], "simdgroups": 4,
              "operation": "MPP tensor_ops::matmul2d FP16xFP16 with cooperative FP32 accumulator then FP16 output",
              "sourceFiles": sources, "cases": cases,
              "references": ["https://developer.apple.com/videos/play/wwdc2026/330/",
                  "https://developer.apple.com/documentation/metal/running-inline-ml-operations-in-a-shader-with-metal-4"],
              "deviceValidated": False, "hardwarePlacementVerified": False,
              "limitations": ["CPU export does not compile or execute MSL.",
                  "FP32 CPU BLAS oracle can have a different summation order.",
                  "MPP API selection is not a measured Neural Accelerator residency or throughput claim."],
              "files": [{"path": str(p.relative_to(output)), "bytes": p.stat().st_size,
                         "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                        for p in sorted(path.rglob("*")) if p.is_file()]}
    _write_json(output / "manifest.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--counts", type=int, nargs="+", default=[5, 128, 256])
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--output-size", type=int, default=67)
    parser.add_argument("--tile-m", type=int, default=32, choices=(16, 32, 64))
    parser.add_argument("--tile-n", type=int, default=64, choices=(32, 64))
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    report = export_smoke(args.output, counts=tuple(args.counts), input_size=args.input_size,
                          output_size=args.output_size, tile_m=args.tile_m, tile_n=args.tile_n)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
