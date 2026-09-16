#!/usr/bin/env python3
"""CPU-author a fused grouped affine-Q4 gate/up/SwiGLU TensorOps candidate.

Shares each input tile across two matrix products, reuses one dequantization
buffer, and stores only the half-precision SwiGLU result. No device execution.
"""
from __future__ import annotations

import argparse
from functools import cache
import json
from pathlib import Path

import torch
from torch._subclasses.fake_tensor import FakeTensor

from coreai_q4_grouped import make_plan, get_plan_kernel, grouped_reference


GATEUP_SOURCE = r"""
const int tile=int(group.y);
if(tile>=plan[0,0])return;
const int expert=plan[0,tile+1],start=plan[1,tile+1],count=plan[2,tile+1];
const int K=int(x.get_extent(0)),N=int(gate_packed.get_extent(1));
const int col=int(group.x)*BN;
threadgroup half left_memory[BM*BK];
threadgroup half right_memory[BN*BK];
auto left=tensor<threadgroup half,extents<int,BK,BM>,tensor_inline>(left_memory,extents<int,BK,BM>());
auto right=tensor<threadgroup half,extents<int,BK,BN>,tensor_inline>(right_memory,extents<int,BK,BN>());
constexpr auto desc=matmul2d_descriptor(BM,BN,BK,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<desc,execution_simdgroups<4>> operation;
auto gate_accum=operation.get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
auto up_accum=operation.get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
for(uint16_t i=0;i<gate_accum.get_capacity();++i)if(gate_accum.is_valid_element(i)) {
  gate_accum[i]=0.0f;up_accum[i]=0.0f;
}
for(int base=0;base<K;base+=BK) {
  for(int i=int(thread_id);i<BM*BK;i+=128) {
    const int row=i/BK,k=i%BK;
    left_memory[i]=(row<count && base+k<K) ? x[base+k,start+row] : half(0);
  }
  for(int i=int(thread_id);i<BN*(BK/4);i+=128) {
    const int row=i/(BK/4),word=i%(BK/4),n=col+row,k=base+word*4;
    ushort bits=0;float scale=0.0f,bias=0.0f;
    if(n<N && k<K) {
      bits=ushort(gate_packed[k/4,n,expert]);
      scale=float(gate_scales[k/64,n,expert]);bias=float(gate_biases[k/64,n,expert]);
    }
    #pragma clang loop unroll(full)
    for(int nibble=0;nibble<4;++nibble) {
      const int code=(uint(bits)>>(nibble*4))&15;
      right_memory[row*BK+word*4+nibble]=half(scale*float(code)+bias);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  operation.run(left,right,gate_accum);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(int i=int(thread_id);i<BN*(BK/4);i+=128) {
    const int row=i/(BK/4),word=i%(BK/4),n=col+row,k=base+word*4;
    ushort bits=0;float scale=0.0f,bias=0.0f;
    if(n<N && k<K) {
      bits=ushort(up_packed[k/4,n,expert]);
      scale=float(up_scales[k/64,n,expert]);bias=float(up_biases[k/64,n,expert]);
    }
    #pragma clang loop unroll(full)
    for(int nibble=0;nibble<4;++nibble) {
      const int code=(uint(bits)>>(nibble*4))&15;
      right_memory[row*BK+word*4+nibble]=half(scale*float(code)+bias);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  operation.run(left,right,up_accum);
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
for(uint16_t i=0;i<gate_accum.get_capacity();++i) {
  if(!gate_accum.is_valid_element(i))continue;
  const auto at=gate_accum.get_multidimensional_index(i);
  const int n=int(at[0]),m=int(at[1]);
  if(n>=0 && n<BN && m>=0 && m<count && col+n<N) {
    const half gate=half(gate_accum[i]),up=half(up_accum[i]);
    const half sigmoid=half(1.0f/(1.0f+exp(-float(gate))));
    const half silu=half(float(gate)*float(sigmoid));
    output[col+n,start+m]=half(float(silu)*float(up));
  }
}
"""


def gateup_reference(x: torch.Tensor, plan: torch.Tensor, gate_packed: torch.Tensor,
                     gate_scales: torch.Tensor, gate_biases: torch.Tensor,
                     up_packed: torch.Tensor, up_scales: torch.Tensor,
                     up_biases: torch.Tensor) -> torch.Tensor:
    if isinstance(x, FakeTensor) or x.device.type == 'meta':
        return torch.empty((x.shape[0],gate_packed.shape[1]),dtype=torch.float16,device=x.device)
    gate=grouped_reference(x,plan,gate_packed,gate_scales,gate_biases)
    up=grouped_reference(x,plan,up_packed,up_scales,up_biases)
    return ((gate*gate.sigmoid()).half()*up).half()


@cache
def get_gateup_kernel(block=16,columns=64,inner=128):
    from coreai.authoring import MetalParameter
    from coreai_torch import TorchMetalKernel
    if block not in (16,32) or columns not in (32,64) or inner not in (64,128):
        raise ValueError('Unsupported fused gate/up tile')
    source=GATEUP_SOURCE.replace('BM',str(block)).replace('BN',str(columns)).replace('BK',str(inner))
    return TorchMetalKernel(f'qwen_q4_grouped_gateup_m{block}_n{columns}_k{inner}_v1',
        input_names=['x','plan','gate_packed','gate_scales','gate_biases','up_packed','up_scales','up_biases'],
        result_names=['output'],src=source,torch_defn=gateup_reference,
        metal_params=[MetalParameter('group','uint3','threadgroup_position_in_grid'),
                      MetalParameter('thread_id','uint','thread_index_in_threadgroup')])


def fused_grouped_gateup(x,plan,gate_packed,gate_scales,gate_biases,
                        up_packed,up_scales,up_biases,block=16,columns=64,inner=128):
    if x.ndim!=2 or x.dtype!=torch.float16 or plan.ndim!=2 or plan.dtype!=torch.int32 or plan.shape[1]!=4:
        raise ValueError('Expected FP16 x[rows,K] and I32 expert plan[capacity,4]')
    rows,inputs=x.shape
    for packed,scales,biases in ((gate_packed,gate_scales,gate_biases),(up_packed,up_scales,up_biases)):
        if (packed.ndim!=3 or packed.dtype!=torch.int16 or inputs%64 or inputs!=packed.shape[-1]*4 or
                scales.dtype!=torch.float16 or biases.dtype!=torch.float16 or scales.shape!=biases.shape or
                scales.shape!=(*packed.shape[:2],inputs//64)):
            raise ValueError('Invalid affine-Q4 group64 gate/up bank')
    if gate_packed.shape!=up_packed.shape:
        raise ValueError('Gate and up projection shapes must match')
    outputs=gate_packed.shape[1]
    return get_gateup_kernel(block,columns,inner)(x,plan,gate_packed,gate_scales,gate_biases,
        up_packed,up_scales,up_biases,
        threads_per_grid=(((outputs+columns-1)//columns)*128,plan.shape[0]-1,1),
        threads_per_thread_group=(128,1,1),result_shapes=[[rows,outputs]])


class GroupedGateUp(torch.nn.Module):
    def __init__(self,gate,up,block=16,columns=64,inner=128):
        super().__init__()
        for prefix,source in (('gate',gate),('up',up)):
            for name in ('packed','scales','biases'):
                self.register_buffer(prefix+'_'+name,getattr(source,name))
        self.block,self.columns,self.inner=block,columns,inner

    def forward(self,x,ids):
        plan=make_plan(ids,self.gate_packed.shape[0],self.block)
        output=fused_grouped_gateup(x,plan,self.gate_packed,self.gate_scales,self.gate_biases,
            self.up_packed,self.up_scales,self.up_biases,self.block,self.columns,self.inner)
        return output,plan


def export_smoke(path,batch=79,inputs=256,outputs=67,experts=5,block=16,columns=64,inner=128):
    import coreai_torch
    from coreai_q4_metal import make_smoke
    from export_coreai_q4_moe import tensor_json
    from export_moe import write_json,sha256_file
    path.mkdir(parents=True,exist_ok=False)
    gate,x,ids=make_smoke(batch,inputs,outputs,experts,seed=1907)
    up,_,_=make_smoke(batch,inputs,outputs,experts,seed=2399)
    ids,permutation=torch.sort(ids);x=x[:,0][permutation]
    module=GroupedGateUp(gate,up,block,columns,inner).eval()
    kernels=[get_plan_kernel(experts,block),get_gateup_kernel(block,columns,inner)]
    converter=coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    converter.register_custom_kernels(kernels)
    converter.add_pytorch_module(module,input_names=('x','ids'),output_names=('output','plan'),
        export_fn=lambda m:torch.export.export(m,args=(x,ids)).run_decompositions(coreai_torch.get_decomp_table()))
    program=converter.to_coreai();program.optimize();asset=path/'gateup.aimodel';program.save_asset(asset)
    for label,value in [('actual',x),('zero',torch.zeros_like(x))]:
        with torch.inference_mode():output,plan=module(value,ids)
        write_json(path/f'{label}.json',{'inputs':{'x':tensor_json(value),'ids':tensor_json(ids)},
            'expectedOutputs':{'output':tensor_json(output),'plan':tensor_json(plan)}})
    for kernel in kernels:
        for kernel_id,source in kernel.kernel_cache.values():
            (path/(kernel_id+'.metal')).write_text(source)
    report={'version':1,'status':'cpu-authored-device-unvalidated','model':asset.name,'batch':batch,
        'inputs':inputs,'outputs':outputs,'experts':experts,'tile':[block,columns,inner],
        'threadgroupMemoryBytes':(block+columns)*inner*2,'deviceValidated':False,
        'provenance':'Synthetic original affine-Q4 packing; distinct gate/up weights and uniform repeated expert IDs',
        'numerics':'FP32 affine -> half weights -> FP32 dot -> half gate/up -> half sigmoid -> half(gate*sigmoid) -> half(*up)',
        'fixtures':['actual.json','zero.json'],
        'files':[{'path':str(p.relative_to(asset)),'bytes':p.stat().st_size,'sha256':sha256_file(p)}
                 for p in sorted(asset.rglob('*')) if p.is_file()]}
    write_json(path/'manifest.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch',type=int,default=79)
    parser.add_argument('--inputs',type=int,default=256)
    parser.add_argument('--outputs',type=int,default=67)
    parser.add_argument('--experts',type=int,default=5)
    parser.add_argument('--block',type=int,default=16)
    parser.add_argument('--columns',type=int,default=64)
    parser.add_argument('--inner',type=int,default=128)
    args=parser.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(2)
    print(json.dumps(export_smoke(args.output,args.batch,args.inputs,args.outputs,args.experts,
        args.block,args.columns,args.inner),indent=2))
