#!/usr/bin/env python3
"""Expert-grouped CoreAI affine-Q4 prefill using Metal TensorOps.

GPU builds fixed-capacity tiles from sorted expert IDs. Each projection unpacks
only BKxBN weight tiles to threadgroup memory and reuses them across BM rows.
Original I16 packed bytes and FP16 affine/activation boundaries are preserved.
CPU callbacks are independent authoring oracles, not a deployment fallback.
"""
from __future__ import annotations
import argparse
from functools import cache
import json
from pathlib import Path
import torch
from torch._subclasses.fake_tensor import FakeTensor

PLAN_SOURCE = r"""
threadgroup int prefix[EXPERT_THREADS];
const int expert = int(thread_id);
const int M = int(ids.get_extent(0));
const int rows = int(plan.get_extent(1));
for (int i=expert; i<rows*4; i+=EXPERT_THREADS) plan[i%4,i/4]=0;
int lo=0,hi=M;
while (lo<hi) { const int mid=(lo+hi)/2; if(ids[mid]<expert)lo=mid+1;else hi=mid; }
const int start=lo;
hi=M;
while (lo<hi) { const int mid=(lo+hi)/2; if(ids[mid]<expert+1)lo=mid+1;else hi=mid; }
const int count=lo-start;
const int tiles=(count+BM-1)/BM;
prefix[expert]=tiles;
threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
for (int distance=1;distance<EXPERT_THREADS;distance*=2) {
  const int old=expert>=distance ? prefix[expert-distance] : 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  prefix[expert]+=old;
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
const int first=prefix[expert]-tiles;
for (int t=0;t<tiles;++t) {
  const int row=1+first+t;
  plan[0,row]=expert;plan[1,row]=start+t*BM;
  plan[2,row]=min(BM,count-t*BM);
}
if(expert==0)plan[0,0]=prefix[EXPERT_THREADS-1];
"""

GEMM_SOURCE = r"""
const int tile=int(group.y);
if(tile>=plan[0,0])return;
const int expert=plan[0,tile+1], start=plan[1,tile+1], count=plan[2,tile+1];
const int K=int(x.get_extent(0)), N=int(packed.get_extent(1));
const int col=int(group.x)*BN;
threadgroup half left_memory[BM*BK];
threadgroup half right_memory[BN*BK];
auto left=tensor<threadgroup half,extents<int,BK,BM>,tensor_inline>(left_memory,extents<int,BK,BM>());
auto right=tensor<threadgroup half,extents<int,BK,BN>,tensor_inline>(right_memory,extents<int,BK,BN>());
constexpr auto desc=matmul2d_descriptor(BM,BN,BK,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
matmul2d<desc,execution_simdgroups<4>> operation;
auto accum=operation.get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
for(uint16_t i=0;i<accum.get_capacity();++i)if(accum.is_valid_element(i))accum[i]=0.0f;
for(int base=0;base<K;base+=BK) {
  for(int i=int(thread_id);i<BM*BK;i+=128) {
    const int r=i/BK,k=i%BK;
    left_memory[i]=(r<count && base+k<K) ? x[base+k,start+r] : half(0);
  }
  // Decode each original I16 word once, reusing its affine group across four values.
  for(int i=int(thread_id);i<BN*(BK/4);i+=128) {
    const int row=i/(BK/4), word=i%(BK/4), n=col+row, k=base+word*4;
    ushort bits=0; float scale=0.0f,bias=0.0f;
    if(n<N && k<K) {
      bits=ushort(packed[k/4,n,expert]);
      scale=float(scales[k/64,n,expert]);bias=float(biases[k/64,n,expert]);
    }
    #pragma clang loop unroll(full)
    for(int nibble=0;nibble<4;++nibble) {
      const int code=(uint(bits)>>(nibble*4))&15;
      right_memory[row*BK+word*4+nibble]=half(scale*float(code)+bias);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  operation.run(left,right,accum);
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
for(uint16_t i=0;i<accum.get_capacity();++i) {
  if(!accum.is_valid_element(i))continue;
  const auto at=accum.get_multidimensional_index(i);
  const int n=int(at[0]),m=int(at[1]);
  if(n>=0 && n<BN && m>=0 && m<count && col+n<N)output[col+n,start+m]=half(accum[i]);
}
"""


def _fake(value):
    return isinstance(value, FakeTensor) or value.device.type == 'meta'


def plan_reference(ids: torch.Tensor, experts: int, block: int) -> torch.Tensor:
    capacity=(ids.shape[0]+block-1)//block+experts
    out=torch.zeros(capacity,4,dtype=torch.int32,device=ids.device)
    if _fake(ids):return out
    if ids.dtype!=torch.int32 or ids.ndim!=1 or ids.numel()==0:
        raise ValueError('Nonempty sorted I32 IDs required')
    if not bool(((ids>=0)&(ids<experts)).all()) or not bool((ids[1:]>=ids[:-1]).all()):
        raise ValueError('IDs must be sorted and in range')
    counts=torch.bincount(ids.long(),minlength=experts).tolist()
    offset=0;tile=1
    for expert,count in enumerate(counts):
        for start in range(0,count,block):
            out[tile]=torch.tensor([expert,offset+start,min(block,count-start),0],dtype=torch.int32)
            tile+=1
        offset+=count
    out[0,0]=tile-1
    return out


@cache
def get_plan_kernel(experts=512,block=16):
    from coreai.authoring import MetalParameter
    from coreai_torch import TorchMetalKernel
    threads=max(32,1<<(experts-1).bit_length())
    if experts>512 or experts<1 or block not in (16,32):raise ValueError('Invalid expert plan')
    source=PLAN_SOURCE.replace('EXPERT_THREADS',str(threads)).replace('BM',str(block))
    return TorchMetalKernel(f'qwen_expert_plan_e{experts}_m{block}_v1',
        input_names=['ids','experts','block'],result_names=['plan'],src=source,torch_defn=plan_reference,
        metal_params=[MetalParameter('thread_id','uint','thread_index_in_threadgroup')])


def make_plan(ids,experts=512,block=16):
    threads=max(32,1<<(experts-1).bit_length())
    return get_plan_kernel(experts,block)(ids,experts,block,threads_per_grid=(threads,1,1),
        threads_per_thread_group=(threads,1,1),result_shapes=[[(ids.shape[0]+block-1)//block+experts,4]])


def grouped_reference(x: torch.Tensor, plan: torch.Tensor, packed: torch.Tensor,
                      scales: torch.Tensor, biases: torch.Tensor) -> torch.Tensor:
    result=torch.empty(x.shape[0],packed.shape[1],dtype=torch.float16,device=x.device)
    if _fake(x):return result
    shift=torch.arange(0,16,4,dtype=torch.int32)
    for expert,start,count,_ in plan[1:1+int(plan[0,0])].tolist():
        codes=((packed[expert].int().unsqueeze(-1)>>shift)&15).reshape(packed.shape[1],-1,64).float()
        weight=(codes*scales[expert].float().unsqueeze(-1)+biases[expert].float().unsqueeze(-1)).half().flatten(-2)
        result[start:start+count]=torch.nn.functional.linear(x[start:start+count].float(),weight.float()).half()
    return result


@cache
def get_grouped_kernel(block=16,columns=64,inner=128):
    from coreai.authoring import MetalParameter
    from coreai_torch import TorchMetalKernel
    if block not in (16,32) or columns not in (32,64) or inner not in (64,128):raise ValueError('Unsupported tiles')
    source=GEMM_SOURCE.replace('BM',str(block)).replace('BN',str(columns)).replace('BK',str(inner))
    return TorchMetalKernel(f'qwen_q4_grouped_m{block}_n{columns}_k{inner}_v2',
        input_names=['x','plan','packed','scales','biases'],result_names=['output'],src=source,
        torch_defn=grouped_reference,metal_params=[MetalParameter('group','uint3','threadgroup_position_in_grid'),
        MetalParameter('thread_id','uint','thread_index_in_threadgroup')])


def grouped_linear(x,plan,packed,scales,biases,block=16,columns=64,inner=128):
    rows,inputs=x.shape;outputs=packed.shape[1]
    if (x.dtype!=torch.float16 or packed.dtype!=torch.int16 or inputs%64 or
            inputs!=packed.shape[-1]*4 or scales.shape!=biases.shape or
            scales.shape!=(*packed.shape[:2],inputs//64)):
        raise ValueError('Invalid affine Q4 group64 input')
    return get_grouped_kernel(block,columns,inner)(x,plan,packed,scales,biases,
        threads_per_grid=(((outputs+columns-1)//columns)*128,plan.shape[0]-1,1),
        threads_per_thread_group=(128,1,1),result_shapes=[[rows,outputs]])


class GroupedProjection(torch.nn.Module):
    def __init__(self,original,block=16,columns=64,inner=128):
        super().__init__()
        for name in ('packed','scales','biases'):self.register_buffer(name,getattr(original,name))
        self.block,self.columns,self.inner=block,columns,inner
    def forward(self,x,ids):
        # IDs sorted by caller. Standalone projection includes plan in result for validation.
        plan=make_plan(ids,self.packed.shape[0],self.block)
        return grouped_linear(x,plan,self.packed,self.scales,self.biases,self.block,self.columns,self.inner),plan


def export_smoke(path,batch=79,inputs=256,outputs=67,experts=5,block=16,columns=64,inner=128):
    import coreai_torch
    from coreai_q4_metal import make_smoke
    from export_coreai_q4_moe import tensor_json
    path.mkdir(parents=True,exist_ok=False)
    original,x,ids=make_smoke(batch,inputs,outputs,experts)
    ids,permutation=torch.sort(ids);x=x[:,0][permutation]
    module=GroupedProjection(original,block,columns,inner).eval()
    converter=coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    converter.register_custom_kernels([get_plan_kernel(experts,block),get_grouped_kernel(block,columns,inner)])
    converter.add_pytorch_module(module,input_names=('x','ids'),output_names=('output','plan'),
        export_fn=lambda m:torch.export.export(m,args=(x,ids)).run_decompositions(coreai_torch.get_decomp_table()))
    program=converter.to_coreai();program.optimize();program.save_asset(path/'grouped.aimodel')
    for label,value in [('actual',x),('zero',torch.zeros_like(x))]:
        with torch.inference_mode():output,plan=module(value,ids)
        (path/f'{label}.json').write_text(json.dumps({'inputs':{'x':tensor_json(value),'ids':tensor_json(ids)},
            'expectedOutputs':{'output':tensor_json(output),'plan':tensor_json(plan)}}))
    (path/'manifest.json').write_text(json.dumps({'model':'grouped.aimodel','batch':batch,'inputs':inputs,
        'outputs':outputs,'experts':experts,'tile':[block,columns,inner],'deviceValidated':False},indent=2)+'\n')


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
    args=parser.parse_args();torch.set_num_threads(2)
    export_smoke(args.output,args.batch,args.inputs,args.outputs,args.experts,args.block,args.columns,args.inner)
