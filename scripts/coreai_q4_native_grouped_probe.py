#!/usr/bin/env python3
"""Experimental grouped native uint4 projection, separate from production.

Preserves sorted expert plans and original rank-one I16/FP16 input bytes. The
arithmetic deliberately moves affine scaling outside each K64 dot product and
therefore removes production's per-weight FP16 rounding. Device validation must
compare both candidate arithmetic and original-model arithmetic.
"""
from __future__ import annotations

import argparse
from functools import cache
import json
from pathlib import Path

import torch
from torch._subclasses.fake_tensor import FakeTensor

from coreai_q4_grouped import get_plan_kernel, make_plan, grouped_reference
from coreai_q4_metal import make_smoke
from coreai_q4_native_probe import tensor_json


SOURCE = r"""
const int tile=int(group.y);
if(tile>=plan[0,0])return;
const int expert=plan[0,tile+1],start=plan[1,tile+1],count=plan[2,tile+1];
const int K=INPUT_SIZE,N=OUTPUT_SIZE,col=int(group.x)*32,t=int(thread_id);
// Format tensors require 128-byte aligned addresses and row strides.
alignas(128) threadgroup ushort right_memory[32*64];
threadgroup half left_memory[16*64];
threadgroup float row_sum[16],group_scale[32],group_bias[32];
// Only the first 16 words of each padded row are data. Initialize padding once.
for(int i=t;i<32*64;i+=128)right_memory[i]=0;
auto left=tensor<threadgroup half,extents<int,64,16>,tensor_inline>(
    left_memory,extents<int,64,16>());
using right_type=tensor<threadgroup uint4b_format,extents<int,64,32>,tensor_inline>;
auto right=right_type(reinterpret_cast<right_type::data_handle_type>(right_memory),
    extents<int,64,32>(),array<int,2>{1,256});
constexpr auto desc=matmul2d_descriptor(16,32,64,false,true,false,
    matmul2d_descriptor::mode::multiply);
matmul2d<desc,execution_simdgroups<4>> operation;
auto partial=operation.get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
auto accum=operation.get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
for(uint16_t i=0;i<accum.get_capacity();++i)if(accum.is_valid_element(i))accum[i]=0.0f;
// Ensure padding initialization cannot race the first group's packed stores.
threadgroup_barrier(mem_flags::mem_threadgroup);
for(int base=0;base<K;base+=64) {
    // Eight adjacent threads own one input row, loading each half once while
    // summing its K64 values. Each SIMD group contains four independent rows.
    const int row=t/8,lane=t%8;
    float sum=0.0f;
    for(int k=lane;k<64;k+=8) {
        const half value=row<count ? x[base+k,start+row] : half(0);
        left_memory[row*64+k]=value;
        sum+=float(value);
    }
    sum+=simd_shuffle_down(sum,4);
    sum+=simd_shuffle_down(sum,2);
    sum+=simd_shuffle_down(sum,1);
    if(lane==0)row_sum[row]=sum;
    // Copy packed words without scalar nibble extraction or FP16 dequantization.
    for(int i=t;i<32*16;i+=128) {
        const int n=i/16,word=i%16;
        right_memory[n*64+word]=col+n<N ? ushort(packed[(expert*N+col+n)*(K/4)+base/4+word]) : ushort(0);
    }
    if(t<32) {
        const int index=(expert*N+col+t)*(K/64)+base/64;
        group_scale[t]=col+t<N ? float(scales[index]) : 0.0f;
        group_bias[t]=col+t<N ? float(biases[index]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    operation.run(left,right,partial);
    for(uint16_t i=0;i<accum.get_capacity();++i) {
        if(!accum.is_valid_element(i))continue;
        const auto at=accum.get_multidimensional_index(i);
        const int n=int(at[0]),m=int(at[1]);
        accum[i]+=group_scale[n]*partial[i]+group_bias[n]*row_sum[m];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
for(uint16_t i=0;i<accum.get_capacity();++i) {
    if(!accum.is_valid_element(i))continue;
    const auto at=accum.get_multidimensional_index(i);
    const int n=int(at[0]),m=int(at[1]);
    if(n<32 && m<count && col+n<N)output[col+n,start+m]=half(accum[i]);
}
"""


def _weights(packed,scales,biases,experts,outputs,inputs):
    return (packed.reshape(experts,outputs,inputs//4),
            scales.reshape(experts,outputs,inputs//64),
            biases.reshape(experts,outputs,inputs//64))


def native_reference(x,plan,packed,scales,biases,experts,outputs,inputs):
    result=torch.empty(x.shape[0],outputs,dtype=torch.float16,device=x.device)
    if isinstance(x,FakeTensor) or x.device.type=='meta':return result
    packed,scales,biases=_weights(packed,scales,biases,experts,outputs,inputs)
    shifts=torch.arange(4,dtype=torch.int32)*4
    for expert,start,count,_ in plan[1:1+int(plan[0,0])].tolist():
        xx=x[start:start+count].float()
        accum=torch.zeros(count,outputs,dtype=torch.float32)
        for base in range(0,inputs,64):
            q=((packed[expert,:,base//4:base//4+16].int().unsqueeze(-1)>>shifts)&15).reshape(outputs,64).float()
            group=base//64
            left=xx[:,base:base+64]
            accum+=(left@q.T)*scales[expert,:,group].float()[None]+left.sum(-1,keepdim=True)*biases[expert,:,group].float()[None]
        result[start:start+count]=accum.half()
    return result


@cache
def get_kernel(experts,outputs,inputs):
    from coreai.authoring import MetalParameter
    from coreai_torch import TorchMetalKernel
    if min(experts,outputs,inputs)<1 or inputs%64:raise ValueError('Positive geometry with group64 required')

    def reference(x:torch.Tensor,plan:torch.Tensor,packed:torch.Tensor,
                  scales:torch.Tensor,biases:torch.Tensor)->torch.Tensor:
        return native_reference(x,plan,packed,scales,biases,experts,outputs,inputs)

    return TorchMetalKernel(f'qwen_experimental_grouped_native_uint4_e{experts}_n{outputs}_k{inputs}_v1',
        input_names=['x','plan','packed','scales','biases'],result_names=['output'],
        src=SOURCE.replace('INPUT_SIZE',str(inputs)).replace('OUTPUT_SIZE',str(outputs)),torch_defn=reference,
        metal_params=[MetalParameter('group','uint3','threadgroup_position_in_grid'),
                      MetalParameter('thread_id','uint','thread_index_in_threadgroup')])


class ExternalNativeProjection(torch.nn.Module):
    def __init__(self,experts=512,outputs=640,inputs=2560):
        super().__init__()
        self.experts,self.outputs,self.inputs=experts,outputs,inputs

    def forward(self,x,ids,packed,scales,biases):
        assert x.dtype==torch.float16 and x.shape[1]==self.inputs
        assert ids.dtype==torch.int32 and ids.shape==(x.shape[0],)
        assert packed.dtype==torch.int16 and packed.shape==(self.experts*self.outputs*(self.inputs//4),)
        assert scales.dtype==biases.dtype==torch.float16
        assert scales.shape==biases.shape==(self.experts*self.outputs*(self.inputs//64),)
        plan=make_plan(ids,self.experts,16)
        result=get_kernel(self.experts,self.outputs,self.inputs)(x,plan,packed,scales,biases,
            threads_per_grid=(((self.outputs+31)//32)*128,plan.shape[0]-1,1),
            threads_per_thread_group=(128,1,1),result_shapes=[[x.shape[0],self.outputs]])
        return result,plan


def export_asset(output,module,args):
    import coreai_torch
    converter=coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    kernel=get_kernel(module.experts,module.outputs,module.inputs)
    converter.register_custom_kernels([get_plan_kernel(module.experts,16),kernel])
    converter.add_pytorch_module(module,entrypoint_name='main',
        input_names=('x','ids','packed','scales','biases'),output_names=('output','plan'),
        export_fn=lambda m:torch.export.export(m,args=args).run_decompositions(coreai_torch.get_decomp_table()))
    program=converter.to_coreai();program.optimize();program._mlir_module.operation.verify()
    program.save_asset(output)
    for kernel_id,source in kernel.kernel_cache.values():
        (output.parent/f'{kernel_id}.metal').write_text(source)


def small_case():
    source,x,ids=make_smoke(23,128,37,5)
    ids,order=torch.sort(ids)
    x=x[:,0][order]
    return (x,ids,source.packed.flatten(),source.scales.flatten(),source.biases.flatten())


def export(output,source_spec):
    output.mkdir(parents=True,exist_ok=False)
    small=small_case()
    module=ExternalNativeProjection(5,37,128)
    with torch.inference_mode():
        candidate,plan=module(*small)
        original=grouped_reference(small[0],plan,*_weights(*small[2:],5,37,128))
    export_asset(output/'small.aimodel',module,small)
    fixture_inputs={name:tensor_json(value) for name,value in zip(('x','ids','packed','scales','biases'),small)}
    for name,value in [('candidate',candidate),('original-half',original)]:
        (output/f'small-{name}.json').write_text(json.dumps({'inputs':fixture_inputs,
            'expectedOutputs':{'output':tensor_json(value),'plan':tensor_json(plan)}})+'\n')
    torch.save({'inputs':small,'candidate':candidate,'original':original,'plan':plan},output/'small-cpu-reference.pt')
    report={'status':'cpu-authored-device-unvalidated','deviceValidated':False,'productionEnabled':False,
        'tile':[16,32,64],'arithmetic':'Sum K64 groups of (FP32 dot(x, uint4)*FP16 scale + FP32 sum(x)*FP16 bias); final FP16.',
        'originalMathDifference':{'maximumAbsoluteError':float((candidate.float()-original.float()).abs().max()),
            'relativeL2Error':float((candidate.float()-original.float()).norm()/original.float().norm())},
        'limitations':['The affine reordering omits per-weight FP16 rounding and is not source-model equivalent.',
            'The reused large source fixture has production geometry but synthetic weights and routing.',
            'Only the small CPU projection executes; the large bank is never dequantized during export.']}
    if source_spec is not None:
        source_spec=source_spec.resolve()
        source=json.loads(source_spec.read_text())
        names=('x','ids','packed','scales','biases')
        if tuple(source['inputs'])!=names:raise ValueError('Unexpected external input ordering')
        inputs={}
        arguments=[]
        dtypes={'float16':torch.float16,'int16':torch.int16,'int32':torch.int32}
        expected={'x':[20480,2560],'ids':[20480],'packed':[209715200],
                  'scales':[13107200],'biases':[13107200]}
        for name in names:
            spec=dict(source['inputs'][name])
            if spec['shape']!=expected[name]:raise ValueError(f'Unexpected shape for {name}')
            spec['file']=str((source_spec.parent/spec['file']).resolve())
            inputs[name]=spec
            # Shapes only. Fake/export does not read or materialize the large bank.
            arguments.append(torch.empty(spec['shape'],dtype=dtypes[spec['dtype']]))
        export_asset(output/'large.aimodel',ExternalNativeProjection(),tuple(arguments))
        spec={'asset':'large.aimodel','function':'main','inputs':inputs,
              'output':'large-output','repeats':15,'mapped':False}
        (output/'large-spec.json').write_text(json.dumps(spec,indent=2)+'\n')
        report['largeSourceSpec']=str(source_spec)
        report['largeSpec']='large-spec.json'
        report['weightBytesRewritten']=0
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--source-spec',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    print(json.dumps(export(args.output,args.source_spec),indent=2))
