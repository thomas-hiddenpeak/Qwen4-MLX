"""Optional FP16-weight, FP32-output vocabulary projection for CoreAI.

Keeps the existing Head's HC mixer and all its rounding boundaries. Only the
last linear is a custom GEMV so the graph never casts the large constant bank
to FP32. Accumulation and logits stay FP32. Device allocation/performance and
FP32 reduction-order differences require a separate measured comparison.
"""
from functools import cache

import torch
import torch.nn.functional as F


HEAD_METAL_SOURCE = r"""
const uint row = group.x * 4u + simd;
const uint token = group.y;
if (row >= weight.get_extent(1)) return;
float accum = 0.0f;
for (uint k = lane; k < weight.get_extent(0); k += 32u) {
    accum += float(x[k, token, 0]) * float(weight[k, row]);
}
const float total = simd_sum(accum);
if (lane == 0u) output[row, token, 0] = total;
"""


def head_reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Reference uses FP32 GEMM, with no vocabulary-sized FP16 rounding."""
    return F.linear(x.float(), weight.float())


@cache
def get_head_kernel():
    from coreai.authoring import MetalParameter
    from coreai_torch import TorchMetalKernel
    return TorchMetalKernel(
        "qwen_fp16_head_gemv_fp32_output_v1",
        input_names=["x", "weight"], result_names=["output"],
        src=HEAD_METAL_SOURCE, torch_defn=head_reference,
        metal_params=[MetalParameter("group", "uint3", "threadgroup_position_in_grid"),
                      MetalParameter("lane", "uint", "thread_index_in_simdgroup"),
                      MetalParameter("simd", "uint", "simdgroup_index_in_threadgroup")],
    )


def head_linear(x, weight):
    if (x.dtype != torch.float16 or weight.dtype != torch.float16 or x.ndim != 3 or
            weight.ndim != 2 or x.shape[0] != 1 or x.shape[-1] != weight.shape[-1] or
            min(x.shape[1], x.shape[2], weight.shape[0]) <= 0):
        raise ValueError("Expected positive FP16 x[1,S,K] and FP16 weight[N,K]")
    count, outputs = x.shape[1], weight.shape[0]
    return get_head_kernel()(
        x, weight,
        threads_per_grid=(((outputs + 3) // 4) * 128, count, 1),
        threads_per_thread_group=(128, 1, 1), result_shapes=[[1, count, outputs]],
    )


class MetalHead(torch.nn.Module):
    """Adopt existing Head's buffers/mixer without copying or changing HC math."""
    def __init__(self, head):
        super().__init__()
        if head.weight.dtype != torch.float16 or head.weight.ndim != 2:
            raise ValueError("Head weight must be a FP16 matrix")
        self.mixer = head.mixer
        self.register_buffer("weight", head.weight)

    def forward(self, stream):
        return head_linear(self.mixer(stream), self.weight)


class MetalHeadProjection(torch.nn.Module):
    """Projection-only probe for separating HC error from final GEMV error."""
    def __init__(self, weight):
        super().__init__()
        self.register_buffer("weight", weight)

    def forward(self, x):
        return head_linear(x, self.weight)
