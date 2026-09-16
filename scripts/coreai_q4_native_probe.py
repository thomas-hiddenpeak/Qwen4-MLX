#!/usr/bin/env python3
"""Experimental native half x uint4 K64 tile; never a production replacement.

The groupwise correction s*dot(x,q)+b*sum(x) omits the production path's
per-weight FP16 rounding. Fixtures distinguish the candidate's CPU oracle from
that original math. CPU asset authoring does not compile or run Metal.

Public MSL specification (2026-06-04): sections 2.22.2.7 and 6.25;
https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
SDK MPPTensorOpsMatMul2d.h explicitly permits half x uint4b_format -> float.
"""
from __future__ import annotations

import argparse
from functools import cache
import json
from pathlib import Path

import torch


SOURCE = r"""
// uint4 inline storage requires 128-byte base AND row-stride alignment.
// K64 occupies 32 bytes; rows are padded to 128 bytes (256 uint4 elements).
alignas(128) threadgroup ushort right_memory[32*64];
threadgroup half left_memory[16*64];
threadgroup float row_sum[16];
const int t=int(thread_id);
for(int i=t;i<32*64;i+=128) {
    const int n=i/64,word=i%64;
    right_memory[i]=word<16 ? ushort(packed[n*16+word]) : ushort(0);
}
for(int i=t;i<16*64;i+=128)left_memory[i]=x[i%64,i/64];
if(t<16) {
    float value=0.0f;
    for(int k=0;k<64;++k)value+=float(x[k,t]);
    row_sum[t]=value;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
auto left=tensor<threadgroup half,extents<int,64,16>,tensor_inline>(
    left_memory,extents<int,64,16>());
using right_type=tensor<threadgroup uint4b_format,extents<int,64,32>,tensor_inline>;
// Use the tensor's declared public constructor pointer type, rather than
// treating a format tag as an unpacked array of scalar values.
auto right=right_type(reinterpret_cast<right_type::data_handle_type>(right_memory),
    extents<int,64,32>(),array<int,2>{1,256});
constexpr auto desc=matmul2d_descriptor(16,32,64,false,true,false,
    matmul2d_descriptor::mode::multiply);
matmul2d<desc,execution_simdgroups<4>> operation;
auto accum=operation.get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
operation.run(left,right,accum);
for(uint16_t i=0;i<accum.get_capacity();++i) {
    if(!accum.is_valid_element(i))continue;
    const auto at=accum.get_multidimensional_index(i);
    const int n=int(at[0]),m=int(at[1]);
    if(n<32 && m<16) {
        raw[n,m]=accum[i];
        corrected[n,m]=float(scales[n])*accum[i]+float(biases[n])*row_sum[m];
    }
}
"""


def unpack(packed):
    shifts=torch.arange(4,device=packed.device,dtype=torch.int32)*4
    return ((packed.int().reshape(32,16,1)>>shifts)&15).reshape(32,64).float()


def reference(x: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor,
              biases: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    raw=x.float()@unpack(packed).T
    corrected=raw*scales.float()[None]+x.float().sum(-1,keepdim=True)*biases.float()[None]
    return raw,corrected


@cache
def get_kernel():
    from coreai.authoring import MetalParameter
    from coreai_torch import TorchMetalKernel
    return TorchMetalKernel('qwen_experimental_native_uint4_k64_m16_n32_v1',
        input_names=['x','packed','scales','biases'],result_names=['raw','corrected'],
        src=SOURCE,torch_defn=reference,
        metal_params=[MetalParameter('thread_id','uint','thread_index_in_threadgroup')])


class Probe(torch.nn.Module):
    def forward(self,x,packed,scales,biases):
        return get_kernel()(x,packed,scales,biases,threads_per_grid=(128,1,1),
            threads_per_thread_group=(128,1,1),result_shapes=[[16,32],[16,32]])


def example(seed=27330):
    generator=torch.Generator().manual_seed(seed)
    q=torch.randint(0,16,(32,64),generator=generator,dtype=torch.int32)
    # First rows make low/high nibble ordering observable independent of randomness.
    q[0]=torch.arange(64)%16
    q[1]=15-torch.arange(64)%16
    packed=(q.reshape(32,16,4) << (torch.arange(4)*4)).sum(-1).short().reshape(-1)
    x=(torch.randn(16,64,generator=generator)*0.3).half()
    x[0].zero_();x[0,0]=1
    x[1].zero_();x[1,1]=1
    x[2].zero_();x[2,63]=1
    scales=(torch.rand(32,generator=generator)*0.11+0.001).half()
    biases=(torch.randn(32,generator=generator)*0.4).half()
    return x,packed,scales,biases


def tensor_json(value):
    return {'shape':list(value.shape),'dtype':str(value.dtype).removeprefix('torch.'),
            'values':value.flatten().tolist()}


def export(output):
    import coreai_torch
    output.mkdir(parents=True,exist_ok=False)
    inputs=example()
    raw,corrected=reference(*inputs)
    x,packed,scales,biases=inputs
    original=x.float()@(unpack(packed)*scales.float()[:,None]+biases.float()[:,None]).half().float().T
    difference=corrected-original
    converter=coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    converter.register_custom_kernels([get_kernel()])
    converter.add_pytorch_module(Probe(),entrypoint_name='main',
        input_names=('x','packed','scales','biases'),output_names=('raw','corrected'),
        export_fn=lambda m:torch.export.export(m,args=inputs).run_decompositions(coreai_torch.get_decomp_table()))
    program=converter.to_coreai();program.optimize();program.save_asset(output/'native-q4.aimodel')
    fixture={'inputs':{name:tensor_json(value) for name,value in zip(('x','packed','scales','biases'),inputs)},
             'expectedOutputs':{'raw':tensor_json(raw),'corrected':tensor_json(corrected)}}
    (output/'fixture.json').write_text(json.dumps(fixture)+'\n')
    torch.save({'inputs':inputs,'raw':raw,'candidate':corrected,'original':original},output/'cpu-reference.pt')
    for kernel_id,source in get_kernel().kernel_cache.values():
        (output/f'{kernel_id}.metal').write_text(source)
    report={'status':'cpu-authored-device-unvalidated','model':'native-q4.aimodel','fixture':'fixture.json',
        'shape':{'M':16,'N':32,'K':64},'deviceValidated':False,'defaultEnabled':False,
        'originalMathDifference':{'maximumAbsoluteError':float(difference.abs().max()),
                                 'relativeL2Error':float(difference.norm()/original.norm())},
        'limitations':['This deliberately changes per-weight FP16 rounding.',
            'One group and one tile only; no expert routing or performance claim.',
            '128-byte inline tensor row stride means K64 packed rows require padding.',
            'Native E8M0 block32 scales cannot represent source FP16 affine group64.']}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    print(json.dumps(export(args.output),indent=2))
