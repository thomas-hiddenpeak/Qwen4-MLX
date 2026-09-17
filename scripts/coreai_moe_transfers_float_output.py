"""Optional routed FP32 output to expose the final FP16 cast to native CoreAI.

This changes only the opaque tail's output boundary. FP32 products and slot
order match the existing float-product transfer kernel. The graph then applies
the original half cast before shared addition, so the compiler can decide how
to fuse that cast. Device equivalence remains an independent acceptance gate.
"""
from functools import cache

import torch
from torch._subclasses.fake_tensor import FakeTensor

from coreai_moe_chunk import ChunkQ4MoE
from coreai_moe_transfers import TAIL_SOURCE, install_moe_transfers


@cache
def get_float_output_tail_kernel():
    from coreai.authoring import MetalParameter
    from coreai_torch import TorchMetalKernel

    def reference(down:torch.Tensor,inverse:torch.Tensor,scores:torch.Tensor)->torch.Tensor:
        count,hidden=scores.shape[1],down.shape[-1]
        if isinstance(down,FakeTensor) or down.device.type=='meta':
            return torch.empty(1,count,hidden,dtype=torch.float32,device=down.device)
        if down.numel()>2_000_000:raise ValueError('Large CPU tail replay disabled')
        total=torch.zeros(count,hidden,dtype=torch.float32,device=down.device)
        for slot in range(10):
            selected=down[inverse[slot::10].long()].float()
            total=total+selected*scores[0,:,slot,None].float()
        return total.unsqueeze(0)

    source=TAIL_SOURCE.replace('const half weighted=half(float(down[column+j,ordered])*float(score));',
                              'const float weighted=float(down[column+j,ordered])*float(score);')
    source=source.replace('output[column+j,token,0]=half(sums[j]);','output[column+j,token,0]=sums[j];')
    return TorchMetalKernel('qwen_moe_inverse_weight_sum_top10_h4_floatoutput_v1',
        input_names=['down','inverse','scores'],result_names=['output'],src=source,torch_defn=reference,
        metal_params=[MetalParameter('index','uint','thread_position_in_grid')])


class FloatOutputChunkMoE(ChunkQ4MoE):
    """Adopt packed ChunkQ4MoE without adding a learned-buffer name prefix."""
    def __init__(self,original):
        if type(original) is not ChunkQ4MoE:
            raise ValueError('Float-output experiment adopts plain packed ChunkQ4MoE only')
        torch.nn.Module.__init__(self)
        for name,value in original.__dict__.items():
            if not name.startswith('_'):setattr(self,name,value)
        for name,module in original.named_children():self.add_module(name,module)
        install_moe_transfers(self,tail_precision='float32')

    def custom_kernels(self):
        return super().custom_kernels()+[get_float_output_tail_kernel()]

    def reduce_routed(self,ordered_down,inverse,scores):
        count,hidden=scores.shape[0],ordered_down.shape[-1]
        summed=get_float_output_tail_kernel()(ordered_down,inverse.int(),scores.reshape(1,count,10),
            threads_per_grid=((count*hidden+3)//4,1,1),threads_per_thread_group=(256,1,1),
            result_shapes=[[1,count,hidden]])
        return summed.half()
