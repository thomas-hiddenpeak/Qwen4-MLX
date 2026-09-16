"""Optional GDN prefill wrappers and CPU chunkwise-delta algebra prototypes.

The register recurrence path preserves the existing preprocessing and FP16
boundaries. The WY routine is an independent algebraic CPU oracle, not required
for the register path. No main exporter is changed by importing this module.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from coreai_gdn_chunk_metal import FusedGDNRecurrence


def prepare_gdn_inputs(model, hidden, conv_history):
    """Existing GDN equations up to recurrence, with the same rounding points."""
    c = model.config
    sequence = hidden.shape[1]
    qkv, z, a, b = model.project_inputs(hidden)
    conv_input = torch.cat((conv_history,qkv),dim=1)
    next_history = conv_input[:,sequence:sequence+c.kernel-1,:]
    convolved = conv_input[:,:sequence,:].float()*model.conv1d_weight[:,0,0].float()
    for tap in range(1,c.kernel):
        convolved = convolved+conv_input[:,tap:tap+sequence,:].float()*model.conv1d_weight[:,tap,0].float()
    convolved = F.silu(convolved.to(hidden.dtype))
    width = c.key_heads*c.key_dim
    q = model.norm(convolved[...,:width].reshape(1,sequence,c.key_heads,c.key_dim))
    k = model.norm(convolved[...,width:2*width].reshape(1,sequence,c.key_heads,c.key_dim))
    q = q*torch.tensor(1.0/c.key_dim,dtype=hidden.dtype)
    k = k*torch.tensor(c.key_dim**-0.5,dtype=hidden.dtype)
    repeats = c.value_heads//c.key_heads
    q = q.unsqueeze(3).expand(1,sequence,c.key_heads,repeats,c.key_dim).reshape(1,sequence,c.value_heads,c.key_dim).float()
    k = k.unsqueeze(3).expand(1,sequence,c.key_heads,repeats,c.key_dim).reshape(1,sequence,c.value_heads,c.key_dim).float()
    v = convolved[...,2*width:].reshape(1,sequence,c.value_heads,c.value_dim).float()
    decay = torch.exp(-model.A_log.float().exp()*F.softplus((a+model.dt_bias).float())).to(hidden.dtype).float()
    beta = b.sigmoid().float()
    return q,k,v,decay,beta,z,next_history


def finish_gdn(model, y, z):
    gated = model.norm(y,model.norm_weight)*z.sigmoid()
    return model.linear(gated.reshape(1,y.shape[1],-1),model.out_proj_weight)


class GDNRegisterPrefill(torch.nn.Module):
    """Drop-in GDN attention wrapper with one register-resident time-loop op.

    Register get_gdn_recurrence_kernel() with TorchConverter before conversion.
    Both S1 and arbitrary fixed prefill lengths use the same state and weights.
    """
    def __init__(self, source):
        super().__init__()
        if source.config.key_dim != 128 or source.in_proj_qkv_weight.dtype != torch.float16:
            raise ValueError('Register GDN requires K128 and existing FP16 projection policy')
        self.source=source
        self.config=source.config
        self.recurrence=FusedGDNRecurrence()

    def forward(self,hidden,conv_history,recurrent_state):
        q,k,v,decay,beta,z,next_history=prepare_gdn_inputs(self.source,hidden,conv_history)
        y,state=self.recurrence(q,k,v,decay,beta,recurrent_state)
        return finish_gdn(self.source,y,z),next_history,state


def _decay_matrix(decay):
    """FP32/FP64 products in log space; exact zero gates remain state resets.

    This is an explicit WY numerical-policy change from sequential FP32 products.
    Exponents are masked/clamped before exp to avoid upper-triangle overflow.
    """
    zeros=(decay==0).to(decay.dtype).cumsum(-1)
    logarithm=torch.where(decay>0,decay,torch.ones_like(decay)).log().cumsum(-1)
    index=torch.arange(decay.shape[-1],device=decay.device)
    causal=index[:,None]>=index[None,:]
    same_segment=zeros.unsqueeze(-1)==zeros.unsqueeze(-2)
    intervals=(logarithm.unsqueeze(-1)-logarithm.unsqueeze(-2)).clamp(max=0).exp()
    intervals=torch.where(causal & same_segment,intervals,torch.zeros_like(intervals))
    cumulative=torch.where(zeros==0,logarithm.exp(),torch.zeros_like(logarithm))
    return intervals,cumulative


def unit_lower_inverse(lower):
    """Blockwise inverse of I+strict_lower, using batched GEMM and log2 levels.

    Unlike a power-series doubling inverse, block merging avoids huge cancelling
    powers for strongly correlated keys. Sequence length must be a power of two.
    All groups at one level are batched; there is no loop over token updates.
    """
    size=lower.shape[-1]
    if size<1 or size & (size-1): raise ValueError('Power-of-two block required')
    inverse=torch.ones(*lower.shape[:-2],size,1,1,dtype=lower.dtype,device=lower.device)
    width=2
    while width<=size:
        half=width//2
        offdiag=torch.stack([lower[...,start+half:start+width,start:start+half]
                             for start in range(0,size,width)],dim=-3)
        left,right=inverse[...,0::2,:,:],inverse[...,1::2,:,:]
        cross=-(right@offdiag)@left
        inverse=torch.cat((torch.cat((left,torch.zeros_like(cross)),dim=-1),
                           torch.cat((cross,right),dim=-1)),dim=-2)
        width*=2
    return inverse.squeeze(-3)


def wy_recurrence(q,k,v,decay,beta,state,block_size=64,solver='triangular'):
    """Gated delta WY CPU oracle and an exportable block-inverse alternative.

    Inputs have the register kernel interface. FP16 qkv/gate boundaries are
    inherited, state algebra runs in state.dtype, each output rounds to FP16.
    A final partial block is processed as-is by triangular solve; inverse mode
    requires sequence divisible by power-of-two block_size. No state padding.
    Equations are independently written from S_t=alpha_t S_(t-1)+delta_t k_t^T.
    Reference: Yang et al. arXiv:2412.06464; FLA gated_delta_rule/naive.py.
    """
    if solver not in ('triangular','block_inverse'): raise ValueError('Unknown solver')
    dtype=state.dtype
    q,k,v=[x.transpose(1,2).to(dtype) for x in (q,k,v)]
    decay,beta=[x.transpose(1,2).to(dtype) for x in (decay,beta)]
    current=state
    outputs=[]
    for start in range(0,q.shape[-2],block_size):
        stop=min(start+block_size,q.shape[-2])
        qc,kc,vc=q[...,start:stop,:],k[...,start:stop,:],v[...,start:stop,:]
        bc=beta[...,start:stop]
        transition,cumulative=_decay_matrix(decay[...,start:stop])
        indices=torch.arange(stop-start,device=q.device)
        lower_mask=indices[:,None]>indices[None,:]
        lower=(kc@kc.transpose(-1,-2))*bc.unsqueeze(-1)*transition*lower_mask
        incoming=current.transpose(-1,-2)
        rhs=bc.unsqueeze(-1)*(vc-cumulative.unsqueeze(-1)*(kc@incoming))
        if solver=='triangular':
            eye=torch.eye(stop-start,dtype=dtype,device=q.device)
            delta=torch.linalg.solve_triangular(eye+lower,rhs,upper=False,unitriangular=True)
        else:
            delta=unit_lower_inverse(lower)@rhs
        scores=(qc@kc.transpose(-1,-2))*transition
        y=cumulative.unsqueeze(-1)*(qc@incoming)+scores@delta
        outputs.append(y.half())
        current=current*cumulative[...,-1,None,None]+delta.transpose(-1,-2)@(kc*transition[...,-1,:,None])
    return torch.cat(outputs,dim=-2).transpose(1,2),current
