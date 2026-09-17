"""Optional large-chunk MoE with sequential dequant-once expert projections.

The adopted ChunkQ4MoE's buffer names/order and smaller-chunk path are retained.
Large chunks use the existing integer grouping and all original FP16 activation,
weighting and shared-expert boundaries. Each projection expands its bank once
to a private FP16 intermediate then uses direct-device, full-K MPP GEMM.

Ordering operands make gate -> up -> down dependencies explicit in the opaque
graph. They do not prove that CoreAI aliases or releases the three large scratch
allocations. This remains an opt-in experiment until memory and whole-MoE device
comparisons pass; no full-model memory/performance claim is made here.
"""
from functools import cache

import torch
import torch.nn.functional as F
from torch._subclasses.fake_tensor import FakeTensor

from coreai_moe_chunk import ChunkQ4MoE, grouping_permutations
from coreai_q4_grouped import make_plan, get_plan_kernel
from coreai_tensor_matmul import tensor_linear


DEQUANT_SOURCE = r"""
const uint words=INPUTS/4u,total=EXPERTS*OUTPUTS*words;
if(index>=total)return;
const uint word=index%words,flat_row=index/words;
// 'after' is an explicit opaque-op scheduling operand; it does not change
// the affine formula. The graph must observe the producer before this call.
if(plan[0,0]==0) {
  for(uint j=0;j<4;++j)dense[word*4+j,flat_row]=half(0);
  return;
}
const ushort bits=ushort(packed[index]);
const float scale=float(scales[index/16u]),bias=float(biases[index/16u]);
#pragma clang loop unroll(full)
for(uint j=0;j<4;++j) {
  const uint code=(uint(bits)>>(j*4))&15u;
  dense[word*4+j,flat_row]=half(scale*float(code)+bias);
}
"""

DENSE_SOURCE = r"""
const int tile=int(group.y);
if(tile>=plan[0,0])return;
const int expert=plan[0,tile+1],start=plan[1,tile+1],count=plan[2,tile+1];
const int N=int(output.get_extent(0)),column=int(group.x)*BN;
auto left=x.slice(0,start);
auto right=dense.slice(0,expert*N+column);
constexpr auto desc=matmul2d_descriptor(BM,BN,static_cast<int>(dynamic_extent),false,true,false,matmul2d_descriptor::mode::multiply);
matmul2d<desc,execution_simdgroups<4>> operation;
auto accum=operation.get_destination_cooperative_tensor<decltype(left),decltype(right),float>();
operation.run(left,right,accum);
for(uint16_t i=0;i<accum.get_capacity();++i) {
  if(!accum.is_valid_element(i))continue;
  const auto at=accum.get_multidimensional_index(i);
  const int n=int(at[0]),m=int(at[1]);
  if(n>=0 && n<BN && column+n<N && m>=0 && m<count)output[column+n,start+m]=half(accum[i]);
}
"""


def _fake(value):
    return isinstance(value, FakeTensor) or value.device.type == 'meta'


@cache
def get_dequant_once_kernel(experts, outputs, inputs):
    from coreai.authoring import MetalParameter
    from coreai_torch import TorchMetalKernel
    if min(experts, outputs, inputs) <= 0 or inputs % 64:
        raise ValueError('Positive affine-Q4 group64 dimensions required')

    def reference(packed:torch.Tensor, scales:torch.Tensor, biases:torch.Tensor,
                  plan:torch.Tensor, after:torch.Tensor)->torch.Tensor:
        if _fake(packed):
            return torch.empty((experts*outputs,inputs),dtype=torch.float16,device=packed.device)
        if experts*outputs*inputs > 2_000_000:
            raise ValueError('Large CPU dequant-once execution is deliberately disabled')
        if int(plan[0,0]) == 0:
            return torch.zeros(experts*outputs,inputs,dtype=torch.float16)
        shift=torch.arange(0,16,4,dtype=torch.int32)
        codes=((packed.int().reshape(experts,outputs,inputs//4).unsqueeze(-1)>>shift)&15).reshape(experts,outputs,inputs//64,64).float()
        return (codes*scales.reshape(experts,outputs,inputs//64).float().unsqueeze(-1)+biases.reshape(experts,outputs,inputs//64).float().unsqueeze(-1)).half().reshape(experts*outputs,inputs)

    src=DEQUANT_SOURCE.replace('EXPERTS',str(experts)+'u').replace('OUTPUTS',str(outputs)+'u').replace('INPUTS',str(inputs)+'u')
    return TorchMetalKernel(f'qwen_q4_dequant_sequence_e{experts}_n{outputs}_k{inputs}_v1',
        input_names=['packed','scales','biases','plan','after'],result_names=['dense'],
        src=src,torch_defn=reference,
        metal_params=[MetalParameter('index','uint','thread_position_in_grid')])


@cache
def get_dequant_gemm_kernel(experts, outputs, inputs, block=32, columns=64):
    from coreai.authoring import MetalParameter
    from coreai_torch import TorchMetalKernel
    if block not in (16,32) or columns not in (32,64):
        raise ValueError('Unsupported dequant-once GEMM tile')

    def reference(x:torch.Tensor,plan:torch.Tensor,dense:torch.Tensor)->torch.Tensor:
        result=torch.empty(x.shape[0],outputs,dtype=torch.float16,device=x.device)
        if _fake(x):return result
        if experts*outputs*inputs > 2_000_000:
            raise ValueError('Large CPU dequant-once GEMM is deliberately disabled')
        for expert,start,count,_ in plan[1:1+int(plan[0,0])].tolist():
            result[start:start+count]=F.linear(x[start:start+count].float(),dense[expert*outputs:(expert+1)*outputs].float()).half()
        return result

    return TorchMetalKernel(f'qwen_dequant_gemm_e{experts}_n{outputs}_k{inputs}_m{block}_n{columns}_v1',
        input_names=['x','plan','dense'],result_names=['output'],
        src=DENSE_SOURCE.replace('BM',str(block)).replace('BN',str(columns)),torch_defn=reference,
        metal_params=[MetalParameter('group','uint3','threadgroup_position_in_grid')])


class DequantOnceChunkMoE(ChunkQ4MoE):
    def __init__(self, original, *, minimum_chunk=8192, block=32, columns=64):
        if not isinstance(original, ChunkQ4MoE) or not original.flat_weights:
            raise ValueError('Adopt an existing flat-weight ChunkQ4MoE')
        if not 2 <= minimum_chunk <= 8192:
            raise ValueError('Dequant-once threshold must be in 2...8192')
        torch.nn.Module.__init__(self)
        # Keep existing optional dispatch flags and direct children without
        # introducing a wrapper prefix or duplicating learned buffers.
        for name,value in original.__dict__.items():
            if not name.startswith('_'):setattr(self,name,value)
        for name,module in original.named_children():self.add_module(name,module)
        self.dequant_minimum_chunk=minimum_chunk
        self.dequant_block,self.dequant_columns=block,columns
        before=[(name,value.dtype,tuple(value.shape),value.data_ptr()) for name,value in original.named_buffers()]
        after=[(name,value.dtype,tuple(value.shape),value.data_ptr()) for name,value in self.named_buffers()]
        if before != after:raise AssertionError('MoE adoption changed weight names/order/storage')

    def custom_kernels(self):
        kernels=super().custom_kernels()+[get_plan_kernel(self.experts,self.dequant_block)]
        for name in ('gate_proj','up_proj','down_proj'):
            geometry=getattr(self.decode,name).geometry
            kernels += [get_dequant_once_kernel(*geometry),
                        get_dequant_gemm_kernel(*geometry,self.dequant_block,self.dequant_columns)]
        return list(dict.fromkeys(kernels))

    def dequant_projection(self,name,x,plan,after):
        projection=getattr(self.decode,name)
        experts,outputs,inputs=projection.geometry
        dense=get_dequant_once_kernel(experts,outputs,inputs)(projection.packed,projection.scales,
            projection.biases,plan,after,threads_per_grid=(experts*outputs*(inputs//4),1,1),
            threads_per_thread_group=(256,1,1),result_shapes=[[experts*outputs,inputs]])
        return get_dequant_gemm_kernel(experts,outputs,inputs,self.dequant_block,self.dequant_columns)(x,plan,dense,
            threads_per_grid=(((outputs+self.dequant_columns-1)//self.dequant_columns)*128,plan.shape[0]-1,1),
            threads_per_thread_group=(128,1,1),result_shapes=[[x.shape[0],outputs]])

    def forward(self,x):
        if x.shape[1] < self.dequant_minimum_chunk:
            return super().forward(x)
        maximum=8192 if self.integer_grouping else 2048
        if x.dtype != torch.float16 or x.ndim != 3 or x.shape[0] != 1 or x.shape[2] != self.hidden or not 1 <= x.shape[1] <= maximum:
            raise ValueError(f'Expected FP16 x[1,S,hidden] with S <= {maximum}')
        count=x.shape[1]
        ids,scores=self.routing(x)
        flat_ids=ids.reshape(-1)
        if self.integer_grouping:
            from coreai_expert_grouping import integer_grouping
            permutation,inverse,sorted_ids=integer_grouping(flat_ids,self.experts)
            permutation,inverse=permutation.long(),inverse.long()
        else:
            permutation,inverse=grouping_permutations(flat_ids,self.experts)
            sorted_ids=torch.index_select(flat_ids,0,permutation)
        ordered_x=self.ordered_inputs(x,permutation)
        plan=make_plan(sorted_ids,self.experts,self.dequant_block)
        gate=self.dequant_projection('gate_proj',ordered_x,plan,ordered_x)
        up=self.dequant_projection('up_proj',ordered_x,plan,gate)
        active=((gate*gate.sigmoid()).half()*up).half()
        ordered_down=self.dequant_projection('down_proj',active,plan,active)
        routed=self.reduce_routed(ordered_down,inverse,scores)
        shared_gate=tensor_linear(x,self.decode.shared_gate_proj)
        shared_up=tensor_linear(x,self.decode.shared_up_proj)
        shared_active=((shared_gate*shared_gate.sigmoid()).half()*shared_up).half()
        shared_down=tensor_linear(shared_active,self.decode.shared_down_proj)
        shared_score=tensor_linear(x,self.decode.shared_router).sigmoid()
        output=(routed+(shared_down*shared_score).half()).half()
        return output,ids.reshape(1,count,self.top_k),scores.reshape(1,count,self.top_k)
