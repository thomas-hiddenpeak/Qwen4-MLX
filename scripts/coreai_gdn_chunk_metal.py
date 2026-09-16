#!/usr/bin/env python3
"""One-dispatch GDN recurrent prefill with FP32 register-resident state rows.

K=128, one SIMD group per (head,value-row), four key columns per lane. The
time loop remains sequential *inside* Metal; large state tensors and dispatches
are no longer materialized per token. This preserves the current FP16 output /
FP32 recurrent-state policy, but changes FP32 summation/FMA order. Device
numerical and performance acceptance are separate from CPU authoring.

Input q/k/v [1,T,H,K/V] are already normalized/scaled by the existing GDN
preprocessor. Decay/beta [1,T,H] are FP32 representations of its FP16 gates.
State [1,H,V,128] is FP32. Outputs are FP16 y[1,T,H,V] and FP32 next_state.
"""
from __future__ import annotations

import argparse
from functools import cache
import json
from pathlib import Path

import numpy as np
import torch

INPUT_NAMES = ("q", "k", "v", "decay", "beta", "state")
OUTPUT_NAMES = ("y", "next_state")
GDN_RECURRENCE_METAL = r"""
const uint value_row = group.x * 4u + simd;
const uint head = group.y;
if (value_row >= state.get_extent(1)) return;
float cell[4];
for (uint part = 0u; part < 4u; ++part)
    cell[part] = state[lane + 32u * part, value_row, head, 0];
for (uint token = 0u; token < q.get_extent(2); ++token) {
    const float alpha = decay[head, token, 0];
    const float beta_t = beta[head, token, 0];
    float key[4];
    float memory_partial = 0.0f;
    for (uint part = 0u; part < 4u; ++part) {
        key[part] = float(k[lane + 32u * part, head, token, 0]);
        cell[part] *= alpha;
        memory_partial += cell[part] * key[part];
    }
    const float memory = simd_sum(memory_partial);
    const float delta = (float(v[value_row, head, token, 0]) - memory) * beta_t;
    float output_partial = 0.0f;
    for (uint part = 0u; part < 4u; ++part) {
        cell[part] += delta * key[part];
        output_partial += cell[part] * float(q[lane + 32u * part, head, token, 0]);
    }
    const float value = simd_sum(output_partial);
    if (lane == 0u) y[value_row, head, token, 0] = half(value);
}
for (uint part = 0u; part < 4u; ++part)
    next_state[lane + 32u * part, value_row, head, 0] = cell[part];
"""


def recurrence_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         decay: torch.Tensor, beta: torch.Tensor,
                         state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Exactly the existing export_coreai_gdn recurrence equations on CPU."""
    current = state.float()
    rows = []
    for token in range(q.shape[1]):
        key = k[:, token].float().unsqueeze(-2)
        current = current * decay[:, token, :, None, None]
        memory = (current * key).sum(-1)
        delta = (v[:, token].float() - memory) * beta[:, token, :, None]
        current = current + delta.unsqueeze(-1) * key
        rows.append((current * q[:, token].float().unsqueeze(-2)).sum(-1).half())
    return torch.stack(rows, dim=1), current


@cache
def get_gdn_recurrence_kernel():
    from coreai.authoring import MetalParameter
    from coreai_torch import TorchMetalKernel
    return TorchMetalKernel(
        "qwen_gdn_recurrence_k128_register_v1", input_names=list(INPUT_NAMES),
        result_names=list(OUTPUT_NAMES), src=GDN_RECURRENCE_METAL,
        torch_defn=recurrence_reference,
        metal_params=[MetalParameter("group", "uint3", "threadgroup_position_in_grid"),
                      MetalParameter("lane", "uint", "thread_index_in_simdgroup"),
                      MetalParameter("simd", "uint", "simdgroup_index_in_threadgroup")],
    )


class FusedGDNRecurrence(torch.nn.Module):
    def forward(self, q, k, v, decay, beta, state):
        if q.ndim != 4 or q.shape[0] != 1 or q.shape[-1] != 128 or min(q.shape) <= 0:
            raise ValueError("Expected positive q[1,T,H,128]")
        _, count, heads, _ = q.shape
        if (k.shape != q.shape or v.ndim != 4 or v.shape[:3] != q.shape[:3] or v.shape[-1] < 1 or
                decay.shape != q.shape[:3] or beta.shape != decay.shape or
                state.shape != (1, heads, v.shape[-1], 128)):
            raise ValueError("GDN recurrence input/state shape mismatch")
        if (any(x.dtype not in (torch.float16, torch.float32) for x in (q,k,v)) or
                any(x.dtype != torch.float32 for x in (decay,beta,state))):
            raise ValueError("Expected FP16/FP32 qkv and FP32 half-derived gates/state")
        return get_gdn_recurrence_kernel()(q, k, v, decay, beta, state,
            threads_per_grid=(((v.shape[-1]+3)//4)*128, heads, 1),
            threads_per_thread_group=(128,1,1), result_shapes=[list(v.shape), list(state.shape)])


def make_inputs(tokens=128, heads=2, value_dim=7, seed=1927):
    generator = torch.Generator().manual_seed(seed)
    q = (torch.randn(1,tokens,heads,128,generator=generator)*0.01).half().float()
    k = torch.randn(1,tokens,heads,128,generator=generator)
    k = (k / k.square().sum(-1,keepdim=True).sqrt()).half().float()
    v = (torch.randn(1,tokens,heads,value_dim,generator=generator)*0.2).half().float()
    decay = (torch.rand(1,tokens,heads,generator=generator)*0.2+0.8).half().float()
    beta = torch.randn(1,tokens,heads,generator=generator).sigmoid().half().float()
    state = torch.randn(1,heads,value_dim,128,generator=generator)*0.1
    return q,k,v,decay,beta,state


def export_smoke(output: Path, tokens=128, heads=2, value_dim=7):
    import coreai_torch
    from export_coreai_q4_moe import tensor_json
    from export_moe import sha256_file, write_json
    output.mkdir(parents=True, exist_ok=False)
    model = FusedGDNRecurrence().eval()
    inputs = make_inputs(tokens+3,heads,value_dim)
    prefill = tuple(x[:,:tokens].contiguous() for x in inputs[:-1])+(inputs[-1],)
    decode = tuple(x[:,:1].contiguous() for x in inputs[:-1])+(inputs[-1],)
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    converter.register_custom_kernels([get_gdn_recurrence_kernel()])
    for name, example in (("main",prefill),("decode",decode)):
        converter.add_pytorch_module(model,entrypoint_name=name,input_names=INPUT_NAMES,output_names=OUTPUT_NAMES,
            export_fn=lambda m,example=example:torch.export.export(m,args=example).run_decompositions(coreai_torch.get_decomp_table()))
    program=converter.to_coreai()
    program.optimize()
    asset=output/'recurrence.aimodel'
    program.save_asset(asset)
    with torch.inference_mode():
        values=model(*prefill)
        write_json(output/'actual.json',{'inputs':{n:tensor_json(v) for n,v in zip(INPUT_NAMES,prefill)},
            'expectedOutputs':{n:tensor_json(v) for n,v in zip(OUTPUT_NAMES,values)}})
        write_json(output/'initial-state.json',{'state':tensor_json(inputs[-1])})
        steps=[]
        state=inputs[-1]
        for index,(start,count,function) in enumerate([(0,tokens,'prefill')]+[(tokens+i,1,'decode') for i in range(3)]):
            data=tuple(x[:,start:start+count].contiguous() for x in inputs[:-1])
            values=model(*data,state)
            fixture=f'step-{index}.json'
            write_json(output/fixture,{'inputs':{n:tensor_json(v) for n,v in zip(INPUT_NAMES[:-1],data)},
                'expectedOutputs':{n:tensor_json(v) for n,v in zip(OUTPUT_NAMES,values)}})
            steps.append({'name':f'{function}-{index}','phase':function,'model':function,'fixture':fixture})
            state=values[1]
        write_json(output/'sequence.json',{'version':1,
            'models':{'prefill':{'path':asset.name,'function':'main'},'decode':{'path':asset.name,'function':'decode'}},
            'stateBindings':{'state':'next_state'},'initialState':'initial-state.json','steps':steps,
            'tolerances':{'maximumAbsoluteError':0.0002,'relativeL2Error':0.002}})
    (output/'kernel.metal.txt').write_text(GDN_RECURRENCE_METAL)
    report={'version':1,'status':'cpu-authored-device-unvalidated','model':asset.name,'tokens':tokens,
        'heads':heads,'keyDim':128,'valueDim':value_dim,'provenance':'Synthetic already-preprocessed qkv/gates and nonzero state',
        'policy':'FP32 state and FP16 output per token unchanged; FP32 reduction/FMA order differs',
        'sequence':'sequence.json','fixture':'actual.json',
        'tolerances':{'maximumAbsoluteError':0.0002,'relativeL2Error':0.002},
        'assets':[{'path':str(p.relative_to(output)),'bytes':p.stat().st_size,'sha256':sha256_file(p)}
                  for p in sorted(asset.rglob('*')) if p.is_file()]}
    write_json(output/'manifest.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--tokens',type=int,default=128)
    parser.add_argument('--heads',type=int,default=2)
    parser.add_argument('--value-dim',type=int,default=7)
    args=parser.parse_args()
    if min(args.tokens,args.heads,args.value_dim)<1: raise ValueError('Positive dimensions required')
    torch.set_num_threads(2)
    print(json.dumps(export_smoke(args.output,args.tokens,args.heads,args.value_dim),indent=2))


if __name__=='__main__': main()
