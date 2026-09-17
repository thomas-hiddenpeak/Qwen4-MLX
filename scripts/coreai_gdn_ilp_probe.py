#!/usr/bin/env python3
"""Experimental multi-value-row GDN SIMD recurrence; no time reordering.

Each SIMD owns2/4 independent value rows, sharing q/k/decay/beta. Per-row
four-part FP32 sums, simd_sum and token update order are unchanged in source.
Compiler scheduling/FMA/register pressure still require paired device checks.
Importing this module never modifies the production recurrence. An explicit
install_gdn_ilp call enables the optional prefill path and leaves S1 unchanged.
"""
from __future__ import annotations

import argparse
from functools import cache
import json
from pathlib import Path

import numpy as np
import torch
from torch._subclasses.fake_tensor import FakeTensor

from coreai_gdn_chunk_metal import (INPUT_NAMES,OUTPUT_NAMES,FusedGDNRecurrence,
    get_gdn_recurrence_kernel,make_inputs,recurrence_reference)
from export_coreai_q4_moe import tensor_json


SOURCE=r"""
const uint first_row=(group.x*4u+simd)*ROWS;
const uint head=group.y;
const uint value_dim=state.get_extent(1);
if(first_row>=value_dim)return;
float cell[ROWS][4];
#pragma clang loop unroll(full)
for(uint row=0u;row<ROWS;++row) {
    #pragma clang loop unroll(full)
    for(uint part=0u;part<4u;++part)
        cell[row][part]=first_row+row<value_dim ? state[lane+32u*part,first_row+row,head,0] : 0.0f;
}
for(uint token=0u;token<q.get_extent(2);++token) {
    const float alpha=decay[head,token,0];
    const float beta_t=beta[head,token,0];
    float key[4],query[4],memory_partial[ROWS],delta[ROWS],output_partial[ROWS];
    #pragma clang loop unroll(full)
    for(uint row=0u;row<ROWS;++row) {
        memory_partial[row]=0.0f;
        output_partial[row]=0.0f;
    }
    #pragma clang loop unroll(full)
    for(uint part=0u;part<4u;++part) {
        key[part]=float(k[lane+32u*part,head,token,0]);
        query[part]=float(q[lane+32u*part,head,token,0]);
        #pragma clang loop unroll(full)
        for(uint row=0u;row<ROWS;++row) {
            cell[row][part]*=alpha;
            memory_partial[row]+=cell[row][part]*key[part];
        }
    }
    #pragma clang loop unroll(full)
    for(uint row=0u;row<ROWS;++row) {
        const float memory=simd_sum(memory_partial[row]);
        const float value=first_row+row<value_dim ? float(v[first_row+row,head,token,0]) : 0.0f;
        delta[row]=(value-memory)*beta_t;
    }
    #pragma clang loop unroll(full)
    for(uint part=0u;part<4u;++part) {
        #pragma clang loop unroll(full)
        for(uint row=0u;row<ROWS;++row) {
            cell[row][part]+=delta[row]*key[part];
            output_partial[row]+=cell[row][part]*query[part];
        }
    }
    #pragma clang loop unroll(full)
    for(uint row=0u;row<ROWS;++row) {
        const float value=simd_sum(output_partial[row]);
        if(lane==0u && first_row+row<value_dim)y[first_row+row,head,token,0]=half(value);
    }
}
#pragma clang loop unroll(full)
for(uint row=0u;row<ROWS;++row) {
    #pragma clang loop unroll(full)
    for(uint part=0u;part<4u;++part)
        if(first_row+row<value_dim)next_state[lane+32u*part,first_row+row,head,0]=cell[row][part];
}
"""


def reference(q:torch.Tensor,k:torch.Tensor,v:torch.Tensor,decay:torch.Tensor,
              beta:torch.Tensor,state:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
    if isinstance(q,FakeTensor) or q.device.type=='meta':
        return torch.empty_like(v,dtype=torch.float16),torch.empty_like(state)
    return recurrence_reference(q,k,v,decay,beta,state)


@cache
def get_kernel(rows):
    from coreai.authoring import MetalParameter
    from coreai_torch import TorchMetalKernel
    if rows not in (2,4):raise ValueError('This bounded ILP probe supports2/4 rows perSIMD')
    return TorchMetalKernel(f'qwen_experimental_gdn_recurrence_k128_ilp{rows}_v1',
        input_names=list(INPUT_NAMES),result_names=list(OUTPUT_NAMES),src=SOURCE.replace('ROWS',str(rows)+'u'),
        torch_defn=reference,metal_params=[MetalParameter('group','uint3','threadgroup_position_in_grid'),
            MetalParameter('lane','uint','thread_index_in_simdgroup'),
            MetalParameter('simd','uint','simdgroup_index_in_threadgroup')])


class ILPRecurrence(torch.nn.Module):
    def __init__(self,rows):
        super().__init__()
        if rows not in (2,4):raise ValueError('Expected2/4 rows perSIMD')
        self.rows=rows

    def forward(self,q,k,v,decay,beta,state):
        if (q.ndim!=4 or q.shape[0]!=1 or q.shape[-1]!=128 or k.shape!=q.shape or
                v.ndim!=4 or v.shape[:3]!=q.shape[:3] or min(v.shape)<1 or
                decay.shape!=q.shape[:3] or beta.shape!=decay.shape or
                state.shape!=(1,q.shape[2],v.shape[-1],128)):
            raise ValueError('GDN ILP input/state geometry differs')
        if (any(value.dtype not in (torch.float16,torch.float32) for value in (q,k,v)) or
                any(value.dtype!=torch.float32 for value in (decay,beta,state))):
            raise ValueError('Expected FP16/FP32 qkv and FP32 gates/state')
        per_group=4*self.rows
        return get_kernel(self.rows)(q,k,v,decay,beta,state,
            threads_per_grid=(((v.shape[-1]+per_group-1)//per_group)*128,q.shape[2],1),
            threads_per_thread_group=(128,1,1),result_shapes=[list(v.shape),list(state.shape)])


class PhaseILPRecurrence(torch.nn.Module):
    """Opt-in prefill recurrence, retaining the existing S1 module verbatim."""

    def __init__(self, decode, rows=4):
        super().__init__()
        self.decode = decode
        self.prefill = ILPRecurrence(rows)
        self.rows = rows

    def forward(self, q, k, v, decay, beta, state):
        if q.shape[1] == 1:
            return self.decode(q, k, v, decay, beta, state)
        return self.prefill(q, k, v, decay, beta, state)


def install_gdn_ilp(module, rows=4):
    """Install optional ILP prefill in existing GDNRegisterPrefill wrappers.

    No learned or geometry buffer is added, moved, renamed or replaced. S1 uses
    each wrapper's original FusedGDNRecurrence object. Returns custom kernels to
    register with the authoring converter, including the original S1 kernel.
    Calling this helper is the only activation mechanism; imports change nothing.
    """
    from coreai_gdn_chunk import GDNRegisterPrefill
    if rows not in (2, 4):
        raise ValueError('GDN ILP supports 2 or 4 value rows per SIMD')
    targets = [child for child in module.modules() if isinstance(child, GDNRegisterPrefill)]
    if not targets:
        raise ValueError('No GDNRegisterPrefill wrapper found')
    # Validate every target before mutating any: avoid partially installed graphs.
    for target in targets:
        recurrence = target.recurrence
        if isinstance(recurrence, PhaseILPRecurrence):
            if recurrence.rows != rows:
                raise ValueError('GDN ILP is already installed with a different row count')
        elif not isinstance(recurrence, FusedGDNRecurrence):
            raise ValueError('Refusing to replace an unknown GDN recurrence')
    for target in targets:
        if not isinstance(target.recurrence, PhaseILPRecurrence):
            target.recurrence = PhaseILPRecurrence(target.recurrence, rows)
    return [get_gdn_recurrence_kernel(), get_kernel(rows)]


def export_asset(path,values):
    import coreai_torch
    converter=coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    kernels=[get_gdn_recurrence_kernel(),get_kernel(2),get_kernel(4)]
    converter.register_custom_kernels(kernels)
    for name,module in [('baseline',FusedGDNRecurrence()),('ilp2',ILPRecurrence(2)),('ilp4',ILPRecurrence(4))]:
        converter.add_pytorch_module(module,entrypoint_name=name,input_names=INPUT_NAMES,output_names=OUTPUT_NAMES,
            export_fn=lambda m:torch.export.export(m,args=values).run_decompositions(coreai_torch.get_decomp_table()))
    program=converter.to_coreai();program.optimize();program.save_asset(path)
    for kernel in kernels:
        for name,body in kernel.kernel_cache.values():(path.parent/(name+'.metal')).write_text(body)


def tiny(output):
    values=list(make_inputs(tokens=11,heads=2,value_dim=7))
    values[3][:,5]=0 # Explicit reset in the middle of the token loop.
    values[4][:,7]=0 # Zero beta keeps the decayed state without an update.
    values=tuple(values)
    expected=recurrence_reference(*values)
    export_asset(output/'tiny.aimodel',values)
    fixture={'inputs':{name:tensor_json(value) for name,value in zip(INPUT_NAMES,values)},
        'expectedOutputs':{name:tensor_json(value) for name,value in zip(OUTPUT_NAMES,expected)}}
    (output/'tiny.json').write_text(json.dumps(fixture)+'\n')
    return values


def real_inputs(input_npz):
    from export_moe import Source,sha256_file
    from export_coreai_gdn import GDN,GDNConfig
    from coreai_gdn_chunk import prepare_gdn_inputs
    source=Source(0)
    config_path=source.directory/'config.json'
    config=GDNConfig.from_model(json.loads(config_path.read_text())['text_config'])
    source.prefix='language_model.model.layers.0.linear_attn.'
    module=GDN(config,{name:source.read(name) for name in config.weight_shapes}).eval()
    with np.load(input_npz) as data:
        hidden,history,state=(torch.from_numpy(data['input.'+name].copy()) for name in
                              ('hidden','conv_history','recurrent_state'))
    with torch.inference_mode():
        q,k,v,decay,beta,_,_=prepare_gdn_inputs(module,hidden,history)
    return (q.contiguous(),k.contiguous(),v.contiguous(),decay.contiguous(),beta.contiguous(),state),{
        'inputNPZ':str(input_npz.resolve()),'inputSHA256':sha256_file(input_npz),
        'sourceConfigSHA256':sha256_file(config_path),'sourceTensorReads':source.records,
        'scope':'Real layer0 weights and existing source-faithful CPU GDN preprocessing of repeated MoE capture inputs; not device-native GDN intermediates.'}


def export(output,input_npz=None):
    output.mkdir(parents=True,exist_ok=False)
    tiny(output)
    report={'status':'cpu-authored-device-unvalidated','deviceValidated':False,'productionEnabled':False,
        'functions':['baseline','ilp2','ilp4'],'tiny':{'asset':'tiny.aimodel','fixture':'tiny.json','tokens':11,'heads':2,'valueDim':7},
        'sourcePolicy':'Per-row4part FP32 order and sequential token updates retained;2/4 independent rows share inputs.',
        'risks':['Compiler may change scheduling/FMA despite unchanged source per-row ordering.',
                 'More registers and fewer independent SIMD groups may hurt occupancy; performance is not implied.']}
    if input_npz is not None:
        values,provenance=real_inputs(input_npz)
        export_asset(output/'real-s2048.aimodel',values)
        inputs={}
        for name,value in zip(INPUT_NAMES,values):
            array=value.numpy();filename=name+'.bin';array.tofile(output/filename)
            inputs[name]={'file':filename,'offset':0,'bytes':array.nbytes,'shape':list(array.shape),'dtype':str(array.dtype)}
        for function in ('baseline','ilp2','ilp4'):
            spec={'asset':'real-s2048.aimodel','function':function,'inputs':inputs,
                'output':function+'-output','repeats':15,'mapped':False}
            (output/(function+'-spec.json')).write_text(json.dumps(spec,indent=2)+'\n')
        report['real']={'tokens':values[0].shape[1],'heads':values[0].shape[2],
            'valueDim':values[2].shape[-1],'asset':'real-s2048.aimodel','provenance':provenance,
            'fullCPURecurrenceExecuted':False,'comparison':'Compare ILP2/4 outputs with baseline on identical preprocessed inputs.'}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--input-npz',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    result=export(args.output,args.input_npz)
    print(json.dumps({key:result[key] for key in ('status','functions','tiny')},indent=2))
