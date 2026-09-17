"""Optional top10 MoE movement/reduction kernels, separate from default graphs.

Gather copies FP16 unchanged. The original tail mode rounds each weighted
product to FP16 before FP32 slot-order accumulation and final FP16 output.
An explicit float32-product mode instead compares the compiled native graph's
observed fused behavior; it does not preserve that eager per-product boundary.
The FP32 reduction order may differ from native CoreAI reduce_sum; numerical
and performance acceptance remain separate device measurements.
"""
from functools import cache

import torch
from torch._subclasses.fake_tensor import FakeTensor


GATHER_SOURCE = r"""
const uint K=uint(x.get_extent(0)),R=uint(permutation.get_extent(0));
const uint flat=index*4u,total=R*K;
if(flat>=total)return;
const uint row=flat/K,column=flat%K;
const uint token=uint(permutation[row])/10u;
#pragma clang loop unroll(full)
for(uint j=0;j<4u;++j) {
  if(column+j<K)output[column+j,row]=x[column+j,token,0];
}
"""

TAIL_SOURCE = r"""
const uint K=uint(down.get_extent(0)),S=uint(scores.get_extent(1));
const uint flat=index*4u,total=S*K;
if(flat>=total)return;
const uint token=flat/K,column=flat%K;
float sums[4]={0.0f,0.0f,0.0f,0.0f};
#pragma clang loop unroll(full)
for(uint slot=0;slot<10u;++slot) {
  const uint ordered=uint(inverse[token*10u+slot]);
  const half score=scores[slot,token,0];
  #pragma clang loop unroll(full)
  for(uint j=0;j<4u;++j) {
    if(column+j<K) {
      const half weighted=half(float(down[column+j,ordered])*float(score));
      sums[j]+=float(weighted);
    }
  }
}
#pragma clang loop unroll(full)
for(uint j=0;j<4u;++j) {
  if(column+j<K)output[column+j,token,0]=half(sums[j]);
}
"""


def _fake(x):return isinstance(x,FakeTensor) or x.device.type=='meta'


def gather_reference(x:torch.Tensor,permutation:torch.Tensor)->torch.Tensor:
    if _fake(x):return torch.empty(permutation.shape[0],x.shape[-1],dtype=torch.float16,device=x.device)
    if permutation.numel()*x.shape[-1]>2_000_000:raise ValueError('Large CPU gathered activation deliberately disabled')
    tokens=torch.div(permutation.long(),10,rounding_mode='floor')
    return torch.index_select(x.reshape(-1,x.shape[-1]),0,tokens)


def tail_reference(down:torch.Tensor,inverse:torch.Tensor,scores:torch.Tensor,*,product_precision='float16')->torch.Tensor:
    count=scores.shape[1]
    if _fake(down):return torch.empty(1,count,down.shape[-1],dtype=torch.float16,device=down.device)
    if down.numel()>2_000_000:raise ValueError('Large CPU inverse/weighted reduction deliberately disabled')
    out=torch.zeros(count,down.shape[-1],dtype=torch.float32,device=down.device)
    for slot in range(10):
        rows=inverse[slot::10].long()
        selected=torch.index_select(down,0,rows)
        products=selected.float()*scores[0,:,slot,None].float()
        if product_precision=='float16':products=products.half()
        out=out+products.float()
    return out.half().unsqueeze(0)


@cache
def get_ordered_gather_kernel():
    from coreai.authoring import MetalParameter
    from coreai_torch import TorchMetalKernel
    return TorchMetalKernel('qwen_moe_ordered_gather_top10_h4_v1',
        input_names=['x','permutation'],result_names=['output'],src=GATHER_SOURCE,torch_defn=gather_reference,
        metal_params=[MetalParameter('index','uint','thread_position_in_grid')])


@cache
def get_inverse_reduce_kernel(product_precision='float16'):
    from coreai.authoring import MetalParameter
    from coreai_torch import TorchMetalKernel
    if product_precision not in ('float16','float32'):raise ValueError('Unsupported weighted product precision')
    def reference(down:torch.Tensor,inverse:torch.Tensor,scores:torch.Tensor)->torch.Tensor:
        return tail_reference(down,inverse,scores,product_precision=product_precision)
    source=TAIL_SOURCE
    name='qwen_moe_inverse_weight_sum_top10_h4_v1'
    if product_precision=='float32':
        source=source.replace('const half weighted=half(float(down[column+j,ordered])*float(score));',
                              'const float weighted=float(down[column+j,ordered])*float(score);')
        name='qwen_moe_inverse_weight_sum_top10_h4_floatproducts_v1'
    return TorchMetalKernel(name,
        input_names=['down','inverse','scores'],result_names=['output'],src=source,torch_defn=reference,
        metal_params=[MetalParameter('index','uint','thread_position_in_grid')])


def ordered_gather(x,permutation):
    if (x.dtype!=torch.float16 or x.ndim!=3 or x.shape[0]!=1 or min(x.shape)<1 or x.shape[-1]%4 or
            permutation.dtype!=torch.int32 or permutation.ndim!=1 or permutation.shape[0]!=x.shape[1]*10):
        raise ValueError('Expected FP16 x[1,S,K] with K divisible4 and I32 permutation[S*10]')
    rows,hidden=permutation.shape[0],x.shape[-1]
    return get_ordered_gather_kernel()(x,permutation,threads_per_grid=((rows*hidden+3)//4,1,1),
        threads_per_thread_group=(256,1,1),result_shapes=[[rows,hidden]])


def inverse_weight_sum(down,inverse,scores,*,product_precision='float16'):
    if (down.dtype!=torch.float16 or down.ndim!=2 or min(down.shape)<1 or down.shape[-1]%4 or
            inverse.dtype!=torch.int32 or inverse.shape!=(down.shape[0],) or
            scores.dtype!=torch.float16 or scores.ndim!=3 or scores.shape[0]!=1 or
            scores.shape[-1]!=10 or scores.shape[1]*10!=down.shape[0]):
        raise ValueError('Expected FP16 down[S*10,K], I32 inverse[S*10], FP16 scores[1,S,10]')
    count,hidden=scores.shape[1],down.shape[-1]
    return get_inverse_reduce_kernel(product_precision)(down,inverse,scores,
        threads_per_grid=((count*hidden+3)//4,1,1),threads_per_thread_group=(256,1,1),result_shapes=[[1,count,hidden]])


def install_moe_transfers(module,*,tail_precision='float16'):
    """Opt in eligible ChunkQ4MoE children without changing names or weights.

    S1 still takes the original decode branch. Indices are internal trusted
    grouping results; these kernels do not validate permutation values on GPU.
    tail_precision=None enables only gather and retains the original tail graph.
    """
    from coreai_moe_chunk import ChunkQ4MoE
    if tail_precision not in ('float16','float32',None):raise ValueError('Unsupported tail product precision')
    children=[child for child in module.modules() if isinstance(child,ChunkQ4MoE)]
    if not children:raise ValueError('No ChunkQ4MoE found for direct transfers')
    for child in children:
        if child.top_k!=10 or child.hidden%4:
            raise ValueError('Direct transfers require top10 and hidden size divisible by4')
    for child in children:
        child.direct_transfers=True
        child.direct_transfer_tail_precision=tail_precision
    kernels=[get_ordered_gather_kernel()]
    if tail_precision is not None:kernels.append(get_inverse_reduce_kernel(tail_precision))
    return kernels
