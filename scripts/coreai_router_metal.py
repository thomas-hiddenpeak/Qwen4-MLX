#!/usr/bin/env python3
"""Experimental fused512-expert/top10 router; CPU authoring only.

Finite FP16 logits use an exact integer order key with lowest-ID tie breaking.
All512 FP32 softmax probabilities are rounded to FP16 before selected sequential
FP16 denominator accumulation. The softmax reduction order still needs paired
device comparison against CoreAI's original graph. No production dispatch changes.
"""
from __future__ import annotations

import argparse
from functools import cache
import json
from pathlib import Path

import numpy as np
import torch

from coreai_tensor_matmul import tensor_linear, get_tensor_kernel


ROUTER_SOURCE = r'''
// One FP16 expert logit per thread; sixteen SIMD groups share this token.
const int token=int(group.x),expert=int(thread_id);
const uint lane=thread_id%32u,simd=thread_id/32u;
const half raw=logits[expert,token];
const float value=float(raw);
threadgroup float maxima[16];
threadgroup float sums[16];
threadgroup uint keys[16];
threadgroup float maximum;
threadgroup float denominator;
threadgroup uint winner;
threadgroup half selected[10];

const float local_max=simd_max(value);
if(lane==0u)maxima[simd]=local_max;
threadgroup_barrier(mem_flags::mem_threadgroup);
if(simd==0u) {
  const float total_max=simd_max(lane<16u ? maxima[lane] : -INFINITY);
  if(lane==0u)maximum=total_max;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const float exponential=exp(value-maximum);
const float local_sum=simd_sum(exponential);
if(lane==0u)sums[simd]=local_sum;
threadgroup_barrier(mem_flags::mem_threadgroup);
if(simd==0u) {
  const float total_sum=simd_sum(lane<16u ? sums[lane] : 0.0f);
  if(lane==0u)denominator=total_sum;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const half probability=half(exponential/denominator);

// Transform finite binary16 into increasing numerical order, with both zeros
// equal. Nine low bits prefer the smallest expert ID for equal logits. A zero
// key is a removal sentinel below every finite FP16 logit, including -65504.
const uint bits=uint(as_type<ushort>(raw));
const uint ordered=value==0.0f ? 0x8000u : ((bits&0x8000u)!=0u ? ((~bits)&0xffffu) : (bits|0x8000u));
uint key=(ordered<<9u)|(511u-uint(expert));
for(int slot=0;slot<10;++slot) {
  const uint local_best=simd_max(key);
  if(lane==0u)keys[simd]=local_best;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(simd==0u) {
    const uint best=simd_max(lane<16u ? keys[lane] : 0u);
    if(lane==0u)winner=511u-(best&511u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(uint(expert)==winner) {
    ids[slot,token]=expert;
    selected[slot]=probability;
    key=0u;
  }
  // The next iteration's first barrier follows all writes of this iteration;
  // it also prevents its winner update from racing this winner read.
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if(thread_id==0u) {
  half total=selected[0];
  for(int slot=1;slot<10;++slot)total=half(float(total)+float(selected[slot]));
  // V1 with explicit float casts differed from FP32-division replay on device.
  // Force the precise FP32 operation before FP16 rounding.
  for(int slot=0;slot<10;++slot)scores[slot,token]=half(precise::divide(float(selected[slot]),float(total)));
}
'''

IDS_SOURCE = r'''
const int token=int(group.x),expert=int(thread_id);
const uint lane=thread_id%32u,simd=thread_id/32u;
const half raw=logits[expert,token];
const uint bits=uint(as_type<ushort>(raw));
const uint ordered=float(raw)==0.0f ? 0x8000u : ((bits&0x8000u)!=0u ? ((~bits)&0xffffu) : (bits|0x8000u));
uint key=(ordered<<9u)|(511u-uint(expert));
threadgroup uint keys[16];
threadgroup uint winner;
for(int slot=0;slot<10;++slot) {
  const uint local_best=simd_max(key);
  if(lane==0u)keys[simd]=local_best;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(simd==0u) {
    const uint best=simd_max(lane<16u ? keys[lane] : 0u);
    if(lane==0u)winner=511u-(best&511u);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(uint(expert)==winner) {
    ids[slot,token]=expert;
    key=0u;
  }
}
'''


def _shape(logits):
    if logits.dtype != torch.float16 or logits.ndim != 2 or logits.shape[1] != 512 or not 1 <= logits.shape[0] <= 8192:
        raise ValueError('Expected FP16 logits[S,512], S in1...8192')


def original_ids(logits):
    _shape(logits)
    remaining = logits.float()
    expert_ids = torch.arange(512, dtype=torch.int32, device=logits.device)[None]
    selected = []
    for _ in range(10):
        maximum = remaining.amax(dim=-1, keepdim=True)
        chosen = torch.where(remaining == maximum, expert_ids, 512).amin(dim=-1)
        selected.append(chosen)
        remaining = torch.where(expert_ids == chosen[:, None], float('-inf'), remaining)
    return torch.stack(selected, dim=-1)


def score_stages(logits, ids):
    probabilities = logits.float().softmax(-1).half()
    scores = torch.gather(probabilities, -1, ids.long())
    denominator = scores[:, 0]
    for slot in range(1, 10):
        denominator = (denominator + scores[:, slot]).half()
    return (scores/denominator[:, None]).half(), probabilities, denominator[:, None]


def original_routing(logits):
    """The existing ChunkQ4MoE.routing operations after tensor_linear."""
    ids = original_ids(logits)
    return ids, score_stages(logits, ids)[0]


@cache
def get_router_kernel():
    from coreai.authoring import MetalParameter
    from coreai_torch import TorchMetalKernel

    def reference(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return original_routing(logits)

    return TorchMetalKernel('qwen_router_softmax_top10_e512_f16_v2',
        input_names=['logits'], result_names=['ids', 'scores'], src=ROUTER_SOURCE,
        torch_defn=reference,
        metal_params=[MetalParameter('group', 'uint3', 'threadgroup_position_in_grid'),
                      MetalParameter('thread_id', 'uint', 'thread_index_in_threadgroup')])


def fused_routing(logits):
    _shape(logits)
    count = logits.shape[0]
    return get_router_kernel()(logits, threads_per_grid=(count*512, 1, 1),
        threads_per_thread_group=(512, 1, 1), result_shapes=[[count, 10], [count, 10]])


@cache
def get_router_ids_kernel():
    from coreai.authoring import MetalParameter
    from coreai_torch import TorchMetalKernel

    def reference(logits: torch.Tensor) -> torch.Tensor:
        return original_ids(logits)

    return TorchMetalKernel('qwen_router_top10_ids_e512_f16_v1',
        input_names=['logits'], result_names=['ids'], src=IDS_SOURCE, torch_defn=reference,
        metal_params=[MetalParameter('group', 'uint3', 'threadgroup_position_in_grid'),
                      MetalParameter('thread_id', 'uint', 'thread_index_in_threadgroup')])


def ids_only_routing(logits):
    _shape(logits)
    count = logits.shape[0]
    ids = get_router_ids_kernel()(logits, threads_per_grid=(count*512, 1, 1),
        threads_per_thread_group=(512, 1, 1), result_shapes=[[count, 10]])
    # Leave the complete native score subgraph unchanged: no softmax canceling,
    # no custom reduction/division and the same sequential FP16 additions.
    return ids, score_stages(logits, ids)[0]


@cache
def get_router_diagnostic_kernel():
    from coreai.authoring import MetalParameter
    from coreai_torch import TorchMetalKernel

    def reference(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ids = original_ids(logits)
        scores, probabilities, normalizer = score_stages(logits, ids)
        return ids, scores, probabilities, normalizer

    source = ROUTER_SOURCE.replace('const half probability=half(exponential/denominator);',
        'const half probability=half(exponential/denominator);\nprobabilities[expert,token]=probability;')
    source = source.replace('for(int slot=0;slot<10;++slot)scores[slot,token]=',
        'normalizer[0,token]=total;\n  for(int slot=0;slot<10;++slot)scores[slot,token]=')
    return TorchMetalKernel('qwen_router_softmax_top10_e512_f16_diagnostic_v2',
        input_names=['logits'], result_names=['ids', 'scores', 'probabilities', 'normalizer'],
        src=source, torch_defn=reference,
        metal_params=[MetalParameter('group', 'uint3', 'threadgroup_position_in_grid'),
                      MetalParameter('thread_id', 'uint', 'thread_index_in_threadgroup')])


class LogitsProbe(torch.nn.Module):
    def __init__(self, fused, ids_only=False):
        super().__init__()
        self.fused = fused
        self.ids_only = ids_only

    def forward(self, logits):
        if self.fused:
            return ids_only_routing(logits) if self.ids_only else fused_routing(logits)
        return original_routing(logits)


class FullRouterProbe(LogitsProbe):
    def forward(self, x, weight):
        logits = tensor_linear(x, weight).reshape(x.shape[1], 512)
        return super().forward(logits)


class DiagnosticProbe(torch.nn.Module):
    def __init__(self, fused):
        super().__init__()
        self.fused = fused

    def forward(self, logits):
        if self.fused:
            count = logits.shape[0]
            return get_router_diagnostic_kernel()(logits,
                threads_per_grid=(count*512, 1, 1), threads_per_thread_group=(512, 1, 1),
                result_shapes=[[count, 10], [count, 10], [count, 512], [count, 1]])
        ids = original_ids(logits)
        scores, probabilities, normalizer = score_stages(logits, ids)
        return ids, scores, probabilities, normalizer


def boundary_logits():
    generator = torch.Generator().manual_seed(718234)
    values = (torch.randn(32, 512, generator=generator)*2).half()
    values[0] = 0
    values[1] = -0.0
    values[1, ::2] = 0.0
    values[2] = torch.arange(512).half()/512
    values[3] = values[2].flip(0)
    values[4] = (torch.arange(512)//32).half()
    values[5] = -65504
    values[5, [511, 255, 31, 0]] = 65504
    values[6] = 65504
    values[7] = -65504
    values[8] = -30
    values[8, :10] = 0
    values[9] = (torch.arange(512)-256).float().mul(2**-24).half()
    values[10] = (torch.arange(512)%17).float().mul(2**-10).half()
    values[11] = 7.0
    values[11, [511, 479, 447, 415, 383, 351, 319, 287, 255, 223, 191]] = 8.0
    values[12] = torch.tensor([1.0, 1.0+2**-10, 1.0-2**-11, -1.0]).repeat(128).half()
    values[13] = 0.001953125
    values[13, ::2] = -0.001953125
    return values


def order_keys(logits):
    """Independent CPU check of the integer comparison, not a graph operation."""
    if not bool(torch.isfinite(logits).all()):
        raise ValueError('Finite logits required for ordered-key oracle')
    bits = logits.contiguous().view(torch.int16).int() & 0xffff
    ordered = torch.where(logits == 0, 0x8000,
        torch.where((bits & 0x8000) != 0, (~bits)&0xffff, bits|0x8000))
    return (ordered << 9) | (511-torch.arange(512, dtype=torch.int32))


def export_pair(path, examples, full=False, ids_only=False, diagnostic=False):
    import coreai_torch
    path.mkdir(parents=True, exist_ok=False)
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    if full and diagnostic:
        raise ValueError('Stage diagnostic is logits-only to isolate the scoring math')
    kernel = get_router_diagnostic_kernel() if diagnostic else get_router_ids_kernel() if ids_only else get_router_kernel()
    converter.register_custom_kernels([kernel, get_tensor_kernel()] if full else [kernel])
    for fused, name in ((False, 'baseline'), (True, 'candidate')):
        module = DiagnosticProbe(fused).eval() if diagnostic else (FullRouterProbe if full else LogitsProbe)(fused, ids_only).eval()
        converter.add_pytorch_module(module, entrypoint_name=name,
            input_names=('x', 'weight') if full else ('logits',),
            output_names=('ids', 'scores', 'probabilities', 'normalizer') if diagnostic else ('ids', 'scores'),
            export_fn=lambda model: torch.export.export(model, args=examples).run_decompositions(coreai_torch.get_decomp_table()))
    program = converter.to_coreai()
    program.optimize()
    program.save_asset(path/'router.aimodel')
    for identifier, source in kernel.kernel_cache.values():
        (path/(identifier+'.metal')).write_text(source)


def write_specs(path, inputs):
    for name in ('baseline', 'candidate'):
        (path/(name+'-spec.json')).write_text(json.dumps({'asset': 'router.aimodel',
            'function': name, 'inputs': inputs, 'output': name+'-output', 'repeats': 16,
            'mapped': False, 'oneBufferPerFile': False}, indent=2)+'\n')


def export_boundaries(path, ids_only=False, diagnostic=False):
    from export_coreai_q4_moe import tensor_json
    logits = boundary_logits()
    expected = original_routing(logits)
    assert torch.equal(torch.topk(order_keys(logits), 10, dim=-1).indices.int(), expected[0])
    assert all(torch.equal(a, b) for a, b in zip(expected, fused_routing(logits), strict=True))
    if diagnostic:
        expected = DiagnosticProbe(False)(logits)
    export_pair(path, (logits,), ids_only=ids_only, diagnostic=diagnostic)
    (path/'actual.json').write_text(json.dumps({'inputs': {'logits': tensor_json(logits)},
        'expectedOutputs': dict(zip(('ids', 'scores', 'probabilities', 'normalizer') if diagnostic else ('ids', 'scores'),
                                   map(tensor_json, expected), strict=True))})+'\n')
    logits.numpy().tofile(path/'logits.bin')
    write_specs(path, {'logits': {'file': 'logits.bin', 'offset': 0, 'bytes': logits.numel()*2,
        'shape': list(logits.shape), 'dtype': 'float16'}})
    (path/'cpu-check.json').write_text(json.dumps({'orderedKeysMatchTop10': True, 'cpuCallbackExact': True,
        'rows': 32, 'deviceValidated': False, 'finiteLogitsRequired': True, 'idsOnly': ids_only, 'diagnostic': diagnostic,
        'cases': 'Signed zeros, expert/SIMD ties, FP16 maxima/subnormals/adjacent values, softmax underflow, seeded random'}, indent=2)+'\n')


def export_full(path, manifest_path, count, ids_only=False):
    manifest = json.loads(manifest_path.read_text())
    weights = next(layer for layer in manifest['layers'] if layer['index'] == 0)['weights']
    record = next(row for row in weights['buffers'] if row['bufferName'] == 'moe.decode.router')
    assert record['dtype'] == 'float16' and record['shape'] == [512, 2560]
    original = (manifest_path.parent/weights['path']).resolve()
    with original.open('rb') as source:
        source.seek(record['byteOffset'])
        data = source.read(record['byteLength'])
    assert len(data) == record['byteLength']
    weight = torch.from_numpy(np.frombuffer(data, dtype=np.float16).copy().reshape(512, 2560))
    generator = torch.Generator().manual_seed(718234)
    x = (torch.randn(1, count, 2560, generator=generator)*.125).half()
    x[:, 0] = 0
    export_pair(path, (x, weight), full=True, ids_only=ids_only)
    x.numpy().tofile(path/'x.bin')
    with torch.inference_mode():
        logits = tensor_linear(x, weight).reshape(count, 512)
        expected = original_routing(logits)
        assert bool(torch.isfinite(logits).all())
        assert torch.equal(torch.topk(order_keys(logits), 10, dim=-1).indices.int(), expected[0])
    logits.numpy().tofile(path/'cpu-projection-logits.bin')
    for name, value in zip(('ids', 'scores'), expected, strict=True):
        value.numpy().tofile(path/('cpu-'+name+'.bin'))
    write_specs(path, {
        'x': {'file': 'x.bin', 'offset': 0, 'bytes': x.numel()*2, 'shape': list(x.shape), 'dtype': 'float16'},
        'weight': {'file': str(original), 'offset': record['byteOffset'], 'bytes': record['byteLength'],
                   'shape': record['shape'], 'dtype': record['dtype']}})
    (path/'manifest.json').write_text(json.dumps({'status': 'CPU-authored-device-unvalidated',
        'count': count, 'experts': 512, 'topK': 10, 'fullRouteIncludesMPPProjection': True, 'idsOnly': ids_only,
        'sourceWeightSHA256': record['sha256'], 'seed': 718234,
        'fixture': 'Seeded random hidden activations plus first zero row; real layer0 router weight',
        'orderedKeysCPUExact': True, 'softmax': 'All512 FP32 then FP16 probabilities; device reduction order needs comparison',
        'selectedDenominator': 'Sequential FP16 additions in ordered top10; final FP16 division'}, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--count', type=int, default=2048)
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument('--ids-only', action='store_true', help='Fuse only stable top10 IDs; use original native scoring')
    variants.add_argument('--diagnostic', action='store_true', help='Logits-only full-fused stage outputs for numerical attribution')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    if args.manifest:
        if args.diagnostic:
            raise ValueError('Diagnostic uses fixed boundary logits without a model manifest')
        if not 1 <= args.count <= 8192:
            raise ValueError('Expected count in1...8192')
        export_full(args.output, args.manifest, args.count, args.ids_only)
    else:
        export_boundaries(args.output, args.ids_only, args.diagnostic)
