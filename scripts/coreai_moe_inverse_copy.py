"""Optional inverse-row copy preserving native weighted-reduction graph math."""
from functools import cache

import torch
from torch._subclasses.fake_tensor import FakeTensor

from coreai_moe_chunk import ChunkQ4MoE
from coreai_moe_transfers import install_moe_transfers


INVERSE_COPY_SOURCE = r'''
const uint K=uint(down.get_extent(0)),R=uint(inverse.get_extent(0));
const uint flat=index*4u,total=R*K;
if(flat>=total)return;
const uint row=flat/K,column=flat%K,source_row=uint(inverse[row]);
#pragma clang loop unroll(full)
for(uint j=0;j<4u;++j) {
  if(column+j<K)output[column+j,row]=down[column+j,source_row];
}
'''


@cache
def get_inverse_copy_kernel():
    from coreai.authoring import MetalParameter
    from coreai_torch import TorchMetalKernel
    def reference(down:torch.Tensor,inverse:torch.Tensor)->torch.Tensor:
        if isinstance(down,FakeTensor) or down.device.type=='meta':
            return torch.empty(inverse.shape[0],down.shape[-1],dtype=torch.float16,device=down.device)
        if inverse.numel()*down.shape[-1]>2_000_000:raise ValueError('Large CPU inverse-copy replay disabled')
        return torch.index_select(down,0,inverse.long())
    return TorchMetalKernel('qwen_moe_inverse_copy_h4_v1',input_names=['down','inverse'],result_names=['output'],
        src=INVERSE_COPY_SOURCE,torch_defn=reference,
        metal_params=[MetalParameter('index','uint','thread_position_in_grid')])


def inverse_copy(down,inverse):
    """Indices are trusted internal grouping results, not unvalidated API input."""
    if (down.dtype!=torch.float16 or down.ndim!=2 or min(down.shape)<1 or down.shape[-1]%4 or
            inverse.dtype!=torch.int32 or inverse.ndim!=1 or inverse.shape[0]<1):
        raise ValueError('Expected FP16 down[R,K], K divisible4, and nonempty I32 row indices')
    rows,hidden=inverse.shape[0],down.shape[-1]
    return get_inverse_copy_kernel()(down,inverse,threads_per_grid=((rows*hidden+3)//4,1,1),
        threads_per_thread_group=(256,1,1),result_shapes=[[rows,hidden]])


class InverseCopyChunkMoE(ChunkQ4MoE):
    """Adopt packed MoE; copy-only transfers surround unchanged expert kernels."""
    def __init__(self,original):
        if type(original) is not ChunkQ4MoE:raise ValueError('Adopt a plain packed ChunkQ4MoE')
        torch.nn.Module.__init__(self)
        for name,value in original.__dict__.items():
            if not name.startswith('_'):setattr(self,name,value)
        for name,module in original.named_children():self.add_module(name,module)
        install_moe_transfers(self,tail_precision=None)

    def custom_kernels(self):
        return super().custom_kernels()+[get_inverse_copy_kernel()]

    def reduce_routed(self,ordered_down,inverse,scores):
        count=scores.shape[0]
        down=inverse_copy(ordered_down,inverse.int()).reshape(count,self.top_k,self.hidden)
        # Keep the original native expression, including every declared cast.
        return (down*scores[:,:,None]).half().float().sum(1).half().reshape(1,count,self.hidden)
