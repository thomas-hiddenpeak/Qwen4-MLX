"""Experimental dequant-once MoE preserving measured native gate/up rounding.

This does not install or change a default. Per-weight affine Q4 expansion still
rounds to FP16. Gate GEMM rounds to FP16; up GEMM keeps its FP32 accumulator until
half(gate_half * up_float), then multiplies by the FP16 sigmoid. This matches the
measured packed/NAX policy, but full-K dense GEMM accumulation still needs device
comparison. The three full FP16 banks may remain allocated simultaneously.
"""
from functools import cache

import torch
import torch.nn.functional as F

from coreai_moe_chunk import grouping_permutations
from coreai_moe_dequant import (
    DequantOnceChunkMoE, DENSE_SOURCE, _fake, get_dequant_once_kernel,
)
from coreai_q4_flat import _replace_once
from coreai_q4_grouped import make_plan
from coreai_tensor_matmul import tensor_linear


def parity_source(block=32, columns=64):
    return _replace_once(DENSE_SOURCE,
        'if(n>=0 && n<BN && column+n<N && m>=0 && m<count)output[column+n,start+m]=half(accum[i]);',
        '''if(n>=0 && n<BN && column+n<N && m>=0 && m<count) {
    // Explicitly retain the measured native half boundaries. Up is deliberately
    // the FP32 GEMM accumulator, not a separately rounded FP16 projection.
    volatile thread half gate_boundary=gate[column+n,start+m];
    const float g=float(gate_boundary);
    volatile thread half sigmoid_boundary=half(1.0f/(1.0f+exp(-g)));
    volatile thread half gate_up_boundary=half(g*accum[i]);
    output[column+n,start+m]=half(float(gate_up_boundary)*float(sigmoid_boundary));
  }''').replace('BM', str(block)).replace('BN', str(columns))


def get_parity_up_kernel(experts, outputs, inputs, block=32, columns=64):
    return _parity_up_kernel(experts, outputs, inputs, block, columns)


@cache
def _parity_up_kernel(experts, outputs, inputs, block, columns):
    from coreai.authoring import MetalParameter
    from coreai_torch import TorchMetalKernel
    if min(experts, outputs, inputs) < 1 or inputs % 64 or block not in (16, 32) or columns not in (32, 64):
        raise ValueError('Expected positive group64 geometry and supported dequant GEMM tile')

    def reference(x: torch.Tensor, plan: torch.Tensor, dense: torch.Tensor,
                  gate: torch.Tensor) -> torch.Tensor:
        result = torch.empty(x.shape[0], outputs, dtype=torch.float16, device=x.device)
        if _fake(x):
            return result
        if experts * outputs * inputs > 2_000_000:
            raise ValueError('Large CPU dequant-once GEMM is deliberately disabled')
        for expert, start, count, _ in plan[1:1+int(plan[0, 0])].tolist():
            up = F.linear(x[start:start+count].float(), dense[expert*outputs:(expert+1)*outputs].float())
            g = gate[start:start+count].float()
            sigmoid = (1/(1+torch.exp(-g))).half()
            result[start:start+count] = ((g*up).half().float()*sigmoid.float()).half()
        return result

    return TorchMetalKernel(
        f'qwen_dequant_up_native_parity_e{experts}_n{outputs}_k{inputs}_m{block}_n{columns}_v1',
        input_names=['x', 'plan', 'dense', 'gate'], result_names=['output'],
        src=parity_source(block, columns), torch_defn=reference,
        metal_params=[MetalParameter('group', 'uint3', 'threadgroup_position_in_grid')])


class ParityDequantOnceChunkMoE(DequantOnceChunkMoE):
    """Independent candidate; inherited adoption preserves all learned storage."""
    def custom_kernels(self):
        return list(dict.fromkeys(super().custom_kernels()+[
            get_parity_up_kernel(*self.decode.up_proj.geometry,
                                 self.dequant_block, self.dequant_columns)]))

    def parity_up(self, x, plan, gate):
        projection = self.decode.up_proj
        experts, outputs, inputs = projection.geometry
        dense = get_dequant_once_kernel(experts, outputs, inputs)(
            projection.packed, projection.scales, projection.biases, plan, gate,
            threads_per_grid=(experts*outputs*(inputs//4), 1, 1),
            threads_per_thread_group=(256, 1, 1), result_shapes=[[experts*outputs, inputs]])
        return get_parity_up_kernel(experts, outputs, inputs, self.dequant_block, self.dequant_columns)(
            x, plan, dense, gate,
            threads_per_grid=(((outputs+self.dequant_columns-1)//self.dequant_columns)*128, plan.shape[0]-1, 1),
            threads_per_thread_group=(128, 1, 1), result_shapes=[[x.shape[0], outputs]])

    def forward(self, x):
        if x.shape[1] < self.dequant_minimum_chunk:
            return super().forward(x)
        maximum = 8192 if self.integer_grouping else 2048
        if (x.dtype != torch.float16 or x.ndim != 3 or x.shape[0] != 1 or
                x.shape[2] != self.hidden or not 1 <= x.shape[1] <= maximum):
            raise ValueError(f'Expected FP16 x[1,S,hidden] with S <= {maximum}')
        count = x.shape[1]
        ids, scores = self.routing(x)
        flat_ids = ids.reshape(-1)
        if self.integer_grouping:
            from coreai_expert_grouping import integer_grouping
            permutation, inverse, sorted_ids = integer_grouping(flat_ids, self.experts)
            permutation, inverse = permutation.long(), inverse.long()
        else:
            permutation, inverse = grouping_permutations(flat_ids, self.experts)
            sorted_ids = torch.index_select(flat_ids, 0, permutation)
        ordered_x = self.ordered_inputs(x, permutation)
        plan = make_plan(sorted_ids, self.experts, self.dequant_block)
        gate = self.dequant_projection('gate_proj', ordered_x, plan, ordered_x)
        active = self.parity_up(ordered_x, plan, gate)
        ordered_down = self.dequant_projection('down_proj', active, plan, active)
        routed = self.reduce_routed(ordered_down, inverse, scores)
        shared_gate = tensor_linear(x, self.decode.shared_gate_proj)
        shared_up = tensor_linear(x, self.decode.shared_up_proj)
        shared_active = ((shared_gate*shared_gate.sigmoid()).half()*shared_up).half()
        shared_down = tensor_linear(shared_active, self.decode.shared_down_proj)
        shared_score = tensor_linear(x, self.decode.shared_router).sigmoid()
        output = (routed+(shared_down*shared_score).half()).half()
        return output, ids.reshape(1, count, self.top_k), scores.reshape(1, count, self.top_k)
