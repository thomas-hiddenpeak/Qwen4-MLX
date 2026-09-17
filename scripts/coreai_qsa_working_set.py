#!/usr/bin/env python3
"""Optional QSA prefill working views; persistent state stays full capacity.

A caller must select a static kv_limit >= offset + token_count. Only attention
K/V, indexer pooled-key and mask working views are cropped. Absolute RoPE,
selection tie correction and all six state outputs keep their original meaning.
S1 bypasses this optimization. No padding, sliding window or cache eviction.
"""
from __future__ import annotations

import torch

from coreai_qsa_chunk import QwenQSAChunk, get_pool_kernel


class QwenQSAWorkingSet(QwenQSAChunk):
    def __init__(self,source,kv_limit,*,prefill_sdpa_fp16=True,tile_m=32,tile_n=64):
        super().__init__(source,tile_m,tile_n,prefill_sdpa_fp16=prefill_sdpa_fp16)
        if not isinstance(kv_limit,int) or not 0<kv_limit<=source.capacity or kv_limit%source.ratio:
            raise ValueError('Working KV limit must be positive, ratio-aligned and within capacity')
        self.kv_limit=kv_limit

    def validate_bounds(self,offset,count):
        """Host selection guard; not invoked with symbolic tensor values in graphs."""
        limit=self.capacity if count==1 else self.kv_limit
        if not isinstance(offset,int) or not isinstance(count,int) or count<1 or offset<0 or offset+count>limit:
            raise ValueError('Working KV limit does not cover every visible position')

    def forward(self,x,key_cache,value_cache,raw_cache,pooled_cache,offset,pooled_count):
        count=x.shape[1]
        if count==1:
            return super().forward(x,key_cache,value_cache,raw_cache,pooled_cache,offset,pooled_count)
        if count>self.kv_limit:
            raise ValueError('Token chunk cannot exceed working KV limit')
        s=self.source
        positions=offset+torch.arange(count,dtype=torch.int32)
        end=offset+count
        query_gate=self.linear(x,'q_proj.weight').reshape(1,count,s.heads,s.head_dim*2)
        query,gate=query_gate.split(s.head_dim,dim=-1)
        key=self.linear(x,'k_proj.weight').reshape(1,count,s.kv_heads,s.head_dim)
        value=self.linear(x,'v_proj.weight').reshape(1,count,s.kv_heads,s.head_dim).transpose(1,2)
        query=s.rope(s.norm(query,'q_norm.weight').transpose(1,2),positions)
        key=s.rope(s.norm(key,'k_norm.weight').transpose(1,2),positions)
        indices=positions.long()[None,None,:,None].expand(1,s.kv_heads,count,s.head_dim)
        keys=key_cache.scatter(2,indices,key)
        values=value_cache.scatter(2,indices,value)
        index=self.linear(x,'indexer.index_qk_proj.weight')
        raw=index[...,s.idx_heads*s.idx_dim:]
        raw_indices=positions.long()[None,:,None].expand(1,count,s.idx_dim)
        raw_out=raw_cache.scatter(1,raw_indices,raw)
        full_blocks=torch.div(end,s.ratio,rounding_mode='floor')
        pooled_out=get_pool_kernel(s.ratio,s.rope_dim,s.eps)(
            raw_out,pooled_cache,s.indexer_k_layernorm_weight,s.cosine,s.sine,pooled_count,full_blocks,
            threads_per_grid=(s.blocks*32,1,1),threads_per_thread_group=(32,1,1),
            result_shapes=[[1,s.blocks,s.idx_dim]])
        # Only work tensors change shape. State outputs above remain full size.
        work_positions=s.cache_positions[:self.kv_limit]
        work_blocks=self.kv_limit//s.ratio
        causal=work_positions[None,:]<=positions[:,None]
        if work_blocks>s.topk:
            iq=s.norm(index[...,:s.idx_heads*s.idx_dim].reshape(1,count,s.idx_heads,s.idx_dim),
                      'indexer.q_layernorm.weight').transpose(1,2)
            iq=s.rope(iq,positions)
            pooled_view=pooled_out[:,:work_blocks]
            block_positions=s.block_positions[:work_blocks]
            scores=torch.relu(iq.float()@pooled_view[:,None].float().transpose(-1,-2)).sum(1)
            visible=block_positions[None,:]+s.ratio-1<=positions[:,None]
            # Absolute block IDs and original tie correction are preserved.
            scores=scores-(block_positions.float()/s.ratio)*1e-7
            scores=torch.where(visible[None],scores,float('-inf'))
            chosen=scores.topk(s.topk,dim=-1,sorted=False).indices
            selected=torch.zeros_like(scores,dtype=torch.int32).scatter(-1,chosen,1).bool()&visible[None]
            selected_tokens=selected.repeat_interleave(s.ratio,dim=-1)
            tail_start=torch.div(positions+1,s.ratio,rounding_mode='floor')*s.ratio
            tail=work_positions[None,:]>=tail_start[:,None]
            sparse=(selected_tokens|tail[None])&causal[None]
            mask=torch.where(end>s.budget+s.ratio-1,sparse,causal[None])[:,None]
        else:
            # Bounds guarantee end<=kv_limit<=budget, so sparse routing cannot
            # affect this call. The index projection/raw/pooled state still ran.
            mask=causal[None,None]
        keys_view=keys[:,:,:self.kv_limit]
        values_view=values[:,:,:self.kv_limit]
        if self.prefill_sdpa_fp16:
            attention=s.sdpa(query,keys_view,values_view,attn_mask=mask).half()
        else:
            attention=s.sdpa(query.float(),keys_view.float(),values_view.float(),attn_mask=mask).half()
        gated=(attention.transpose(1,2)*torch.sigmoid(gate)).half().reshape(1,count,s.heads*s.head_dim)
        y=self.linear(gated,'o_proj.weight')
        # Diagnostic mask is a working view. Its omitted columns are all false
        # under validate_bounds; native DecoderLayer discards this last output.
        return y,keys,values,raw_out,pooled_out,end,full_blocks,mask.to(torch.int32)


def working_set_function(token_count,kv_limit):
    if token_count<=1 or token_count>kv_limit or kv_limit%4:
        raise ValueError('Working set entries require count>1 and a covering 4-aligned limit')
    return f'prefill_s{token_count}_kv{kv_limit}'
