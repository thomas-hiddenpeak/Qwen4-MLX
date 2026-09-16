#!/usr/bin/env python3
"""Stable integer expert grouping for optional large CoreAI MoE prefill.

Three kernels construct block histograms, exact integer exclusive offsets and
stable scatter indices. No floating-point sorting keys or atomic-order-dependent
placement are used. Outputs are permutation, inverse and sorted IDs in I32.
"""
from __future__ import annotations

import argparse
from functools import cache
import json
from pathlib import Path

import torch
from torch._subclasses.fake_tensor import FakeTensor


MAX_ASSIGNMENTS = 8192 * 10
BLOCK_ASSIGNMENTS = 256

HISTOGRAM_SOURCE = r"""
threadgroup atomic_int histogram[EXPERTS];
const int t=int(thread_id),b=int(group.x);
for(int expert=t;expert<EXPERTS;expert+=BLOCK)atomic_store_explicit(&histogram[expert],0,memory_order_relaxed);
threadgroup_barrier(mem_flags::mem_threadgroup);
const int row=b*BLOCK+t;
if(row<int(ids.get_extent(0))) {
  const int expert=ids[row];
  if(expert>=0 && expert<EXPERTS)atomic_fetch_add_explicit(&histogram[expert],1,memory_order_relaxed);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for(int expert=t;expert<EXPERTS;expert+=BLOCK)counts[expert,b]=atomic_load_explicit(&histogram[expert],memory_order_relaxed);
"""

PREFIX_SOURCE = r"""
threadgroup int total[THREADS];
const int expert=int(thread_id),blocks=int(counts.get_extent(1));
int running=0;
if(expert<EXPERTS) {
  for(int b=0;b<blocks;++b) {
    offsets[expert,b]=running;
    running+=counts[expert,b];
  }
}
total[expert]=running;
threadgroup_barrier(mem_flags::mem_threadgroup);
for(int distance=1;distance<THREADS;distance*=2) {
  const int old=expert>=distance ? total[expert-distance] : 0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  total[expert]+=old;
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
const int base=expert==0 ? 0 : total[expert-1];
if(expert<EXPERTS)for(int b=0;b<blocks;++b)offsets[expert,b]+=base;
"""

SCATTER_SOURCE = r"""
threadgroup int local_ids[BLOCK];
const int t=int(thread_id),b=int(group.x),row=b*BLOCK+t;
const int rows=int(ids.get_extent(0));
local_ids[t]=row<rows ? ids[row] : -1;
threadgroup_barrier(mem_flags::mem_threadgroup);
if(row>=rows)return;
const int expert=local_ids[t];
// The router guarantees valid IDs. Guard malformed data from indexing buffers.
if(expert<0 || expert>=EXPERTS) { inverse[row]=-1;return; }
int local_rank=0;
for(int previous=0;previous<t;++previous)local_rank+=int(local_ids[previous]==expert);
const int destination=offsets[expert,b]+local_rank;
permutation[destination]=row;
inverse[row]=destination;
sorted_ids[destination]=expert;
"""


def _fake(tensor):
    return isinstance(tensor, FakeTensor) or tensor.device.type == 'meta'


def _check(ids, experts, block):
    if ids.dtype != torch.int32 or ids.ndim != 1 or not 0 < ids.numel() <= MAX_ASSIGNMENTS:
        raise ValueError('Expected nonempty I32 expert IDs with at most81920assignments')
    if not 1 <= experts <= 512 or block != BLOCK_ASSIGNMENTS:
        raise ValueError('Expected1...512experts and256assignment blocks')
    if not _fake(ids) and not bool(((ids >= 0) & (ids < experts)).all()):
        raise ValueError('Expert IDs out of range')


def histogram_reference(ids, experts, block):
    _check(ids, experts, block)
    count = (ids.numel() + block - 1) // block
    if _fake(ids):
        return torch.empty(count, experts, dtype=torch.int32, device=ids.device)
    return torch.stack([torch.bincount(ids[start:start + block].long(), minlength=experts).int()
                        for start in range(0, ids.numel(), block)])


def prefix_reference(counts):
    if _fake(counts):
        return torch.empty_like(counts)
    totals = counts.sum(0, dtype=torch.int32)
    return counts.cumsum(0, dtype=torch.int32) - counts + (totals.cumsum(0, dtype=torch.int32) - totals)[None]


def scatter_reference(ids, offsets, experts, block):
    _check(ids, experts, block)
    if _fake(ids):
        return tuple(torch.empty_like(ids) for _ in range(3))
    source, bases = ids.tolist(), offsets.tolist()
    permutation, inverse, sorted_ids = ([-1] * len(source) for _ in range(3))
    for start in range(0, len(source), block):
        local = [0] * experts
        for row in range(start, min(start + block, len(source))):
            expert = source[row]
            destination = bases[start // block][expert] + local[expert]
            local[expert] += 1
            permutation[destination], inverse[row], sorted_ids[destination] = row, destination, expert
    return tuple(torch.tensor(values, dtype=torch.int32) for values in (permutation, inverse, sorted_ids))


@cache
def get_integer_grouping_kernels(experts=512, block=BLOCK_ASSIGNMENTS):
    from coreai.authoring import MetalParameter
    from coreai_torch import TorchMetalKernel
    if not 1 <= experts <= 512 or block != BLOCK_ASSIGNMENTS:
        raise ValueError('Unsupported integer grouping geometry')
    threads = max(32, 1 << (experts - 1).bit_length())

    def histogram(ids: torch.Tensor) -> torch.Tensor:
        return histogram_reference(ids, experts, block)

    def prefix(counts: torch.Tensor) -> torch.Tensor:
        return prefix_reference(counts)

    def scatter(ids: torch.Tensor, offsets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return scatter_reference(ids, offsets, experts, block)

    params = [MetalParameter('group', 'uint3', 'threadgroup_position_in_grid'),
              MetalParameter('thread_id', 'uint', 'thread_index_in_threadgroup')]
    histogram_kernel = TorchMetalKernel(f'qwen_group_hist_i32_e{experts}_b{block}_v1',
        input_names=['ids'], result_names=['counts'], torch_defn=histogram,
        src=HISTOGRAM_SOURCE.replace('EXPERTS', str(experts)).replace('BLOCK', str(block)), metal_params=params)
    prefix_kernel = TorchMetalKernel(f'qwen_group_prefix_i32_e{experts}_b{block}_v1',
        input_names=['counts'], result_names=['offsets'], torch_defn=prefix,
        src=PREFIX_SOURCE.replace('EXPERTS', str(experts)).replace('THREADS', str(threads)),
        metal_params=[MetalParameter('thread_id', 'uint', 'thread_index_in_threadgroup')])
    scatter_kernel = TorchMetalKernel(f'qwen_group_scatter_i32_e{experts}_b{block}_v1',
        input_names=['ids', 'offsets'], result_names=['permutation', 'inverse', 'sorted_ids'], torch_defn=scatter,
        src=SCATTER_SOURCE.replace('EXPERTS', str(experts)).replace('BLOCK', str(block)), metal_params=params)
    return histogram_kernel, prefix_kernel, scatter_kernel


def integer_grouping(ids, experts=512, block=BLOCK_ASSIGNMENTS):
    # Runtime router outputs satisfy the value-range contract; validation here
    # checks static shape/type, so export does not introduce a host graph break.
    if ids.dtype != torch.int32 or ids.ndim != 1 or not 0 < ids.numel() <= MAX_ASSIGNMENTS:
        raise ValueError('Expected nonempty I32 expert IDs with at most81920assignments')
    histogram, prefix, scatter = get_integer_grouping_kernels(experts, block)
    blocks = (ids.numel() + block - 1) // block
    counts = histogram(ids, threads_per_grid=(blocks * block, 1, 1),
        threads_per_thread_group=(block, 1, 1), result_shapes=[[blocks, experts]])
    threads = max(32, 1 << (experts - 1).bit_length())
    offsets = prefix(counts, threads_per_grid=(threads, 1, 1),
        threads_per_thread_group=(threads, 1, 1), result_shapes=[[blocks, experts]])
    return scatter(ids, offsets, threads_per_grid=(blocks * block, 1, 1),
        threads_per_thread_group=(block, 1, 1), result_shapes=[[ids.numel()]] * 3)


def enable_integer_grouping(module):
    """Enable optional exact grouping before export; return custom registrations."""
    from coreai_moe_chunk import ChunkQ4MoE
    chunks = [child for child in module.modules() if isinstance(child, ChunkQ4MoE)]
    if not chunks:
        raise ValueError('No ChunkQ4MoE found')
    kernels = []
    for moe in chunks:
        moe.integer_grouping = True
        kernels.extend(get_integer_grouping_kernels(moe.experts))
    return list(dict.fromkeys(kernels))


class GroupingProbe(torch.nn.Module):
    def __init__(self, experts=512):
        super().__init__()
        self.experts = experts

    def forward(self, ids):
        return integer_grouping(ids, self.experts)


def export_probe(path, assignments=529):
    import coreai_torch
    from export_coreai_q4_moe import tensor_json
    from export_moe import sha256_file
    if not 1 <= assignments <= MAX_ASSIGNMENTS:
        raise ValueError('Probe assignments must be in1...81920')
    path.mkdir(parents=True, exist_ok=False)
    generator = torch.Generator().manual_seed(6721)
    count, experts = assignments, 512
    mixed = torch.randint(0, 13 if count <= 529 else experts, (count,), generator=generator, dtype=torch.int32)
    mixed[::17] = 511
    cases = {'mixed_partial': mixed, 'one_expert': torch.full((count,), 511, dtype=torch.int32),
             'descending_repeated': (torch.arange(count, dtype=torch.int32).flip(0) % experts)}
    module = GroupingProbe(experts)
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    kernels = get_integer_grouping_kernels(experts)
    converter.register_custom_kernels(list(kernels))
    converter.add_pytorch_module(module, input_names=('ids',), output_names=('permutation', 'inverse', 'sorted_ids'),
        export_fn=lambda m: torch.export.export(m, args=(mixed,)).run_decompositions(coreai_torch.get_decomp_table()))
    program = converter.to_coreai()
    program.optimize()
    asset = path / 'grouping.aimodel'
    program.save_asset(asset)
    for kernel in kernels:
        for kernel_id, source in kernel.kernel_cache.values():
            (path / (kernel_id + '.metal')).write_text(source)
    for name, ids in cases.items():
        actual = module(ids)
        permutation = torch.argsort(ids, stable=True).int()
        inverse = torch.empty_like(permutation).scatter(0, permutation.long(), torch.arange(count, dtype=torch.int32))
        expected = permutation, inverse, ids[permutation.long()]
        assert all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True))
        fixture = {'inputs': {'ids': tensor_json(ids)},
                   'expectedOutputs': dict(zip(('permutation', 'inverse', 'sorted_ids'), map(tensor_json, expected)))}
        (path / (name + '.json')).write_text(json.dumps(fixture) + '\n')
    report = {'version': 1, 'status': 'cpu-authored-device-unvalidated', 'deviceValidated': False,
        'model': asset.name, 'function': 'main', 'cases': list(cases), 'experts': experts, 'assignments': count,
        'maximumAssignments': MAX_ASSIGNMENTS, 'assignmentBlock': BLOCK_ASSIGNMENTS,
        'outputDtype': 'int32', 'CPUOracle': 'Exact independent torch.argsort(ids,stable=True) over integers',
        'assets': [{'path': str(p.relative_to(path)), 'bytes': p.stat().st_size, 'sha256': sha256_file(p)}
                   for p in sorted(asset.rglob('*')) if p.is_file()],
        'limitations': ['No device compilation, correctness or performance acceptance from authoring.',
                       'Router must produce valid IDs in0...E-1; CPU fixture authoring rejects invalid IDs.',
                       'Large chunk support also needs exporter/runtime/attention limits to be widened independently.']}
    (path / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--assignments', type=int, default=529)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    print(json.dumps(export_probe(args.output, args.assignments), indent=2))
