"""Independent padded16, stride8/4/2/1 routed-tail experiment.

FP32 products and every tree node are explicit. The kernel returns FP32 so the
original half cast and shared addition stay in the native graph. This is a
candidate reduction order, not a claim about undocumented CoreAI internals.
"""
from functools import cache

import torch
from torch._subclasses.fake_tensor import FakeTensor

from coreai_moe_transfers_float_output import FloatOutputChunkMoE


TREE_SOURCE = r'''
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
const uint K=uint(down.get_extent(0)),S=uint(scores.get_extent(1));
const uint flat=index*4u,total=S*K;
if(flat>=total)return;
const uint token=flat/K,column=flat%K;
float sums[16][4];
#pragma clang loop unroll(full)
for(uint slot=0;slot<16u;++slot) {
  #pragma clang loop unroll(full)
  for(uint j=0;j<4u;++j)sums[slot][j]=0.0f;
}
#pragma clang loop unroll(full)
for(uint slot=0;slot<10u;++slot) {
  const uint ordered=uint(inverse[token*10u+slot]);
  const float score=float(scores[slot,token,0]);
  #pragma clang loop unroll(full)
  for(uint j=0;j<4u;++j) {
    if(column+j<K)sums[slot][j]=metal::fma(float(down[column+j,ordered]),score,0.0f);
  }
}
#pragma clang loop unroll(full)
for(uint stride=8u;stride>0u;stride/=2u) {
  #pragma clang loop unroll(full)
  for(uint slot=0;slot<stride;++slot) {
    #pragma clang loop unroll(full)
    for(uint j=0;j<4u;++j)sums[slot][j]=metal::fma(1.0f,sums[slot][j],sums[slot+stride][j]);
  }
}
#pragma clang loop unroll(full)
for(uint j=0;j<4u;++j) {
  if(column+j<K)output[column+j,token,0]=sums[0][j];
}
'''


def tree_reference(down:torch.Tensor,inverse:torch.Tensor,scores:torch.Tensor)->torch.Tensor:
    count,hidden=scores.shape[1],down.shape[-1]
    if isinstance(down,FakeTensor) or down.device.type=='meta':
        return torch.empty(1,count,hidden,dtype=torch.float32,device=down.device)
    if down.numel()>2_000_000:raise ValueError('Large CPU routed tree replay disabled')
    terms=torch.zeros(count,16,hidden,dtype=torch.float32,device=down.device)
    for slot in range(10):
        terms[:,slot]=down[inverse[slot::10].long()].float()*scores[0,:,slot,None].float()
    for stride in (8,4,2,1):
        terms=terms[:,:stride]+terms[:,stride:stride*2]
    return terms[:,0].unsqueeze(0)


@cache
def get_tree_float_output_tail_kernel():
    from coreai.authoring import MetalParameter
    from coreai_torch import TorchMetalKernel
    return TorchMetalKernel('qwen_moe_inverse_weight_sum_top10_h4_tree16_floatoutput_v1',
        input_names=['down','inverse','scores'],result_names=['output'],src=TREE_SOURCE,torch_defn=tree_reference,
        metal_params=[MetalParameter('index','uint','thread_position_in_grid')])


class TreeFloatOutputChunkMoE(FloatOutputChunkMoE):
    def custom_kernels(self):
        return super().custom_kernels()+[get_tree_float_output_tail_kernel()]

    def reduce_routed(self,ordered_down,inverse,scores):
        count,hidden=scores.shape[0],ordered_down.shape[-1]
        summed=get_tree_float_output_tail_kernel()(ordered_down,inverse.int(),scores.reshape(1,count,10),
            threads_per_grid=((count*hidden+3)//4,1,1),threads_per_thread_group=(256,1,1),
            result_shapes=[[1,count,hidden]])
        return summed.half()
