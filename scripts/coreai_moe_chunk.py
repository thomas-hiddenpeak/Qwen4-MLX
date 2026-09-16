#!/usr/bin/env python3
"""Expert-grouped CoreAI MoE chunks with shared original affine-Q4 buffers.

S1 retains the existing direct-Q4 Metal GEMV path. Multi-token inputs use MPP
for router/shared projections and a single GPU expert-tile plan reused by the
three routed projections. Sorting changes execution order only; inverse sorting
restores each token's original ordered top-k before FP16 weighting/reduction.
This file authors CPU references/assets only; device acceptance is separate.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from coreai_q4_grouped import make_plan, grouped_linear, get_plan_kernel, get_grouped_kernel
from coreai_q4_gateup import fused_grouped_gateup, get_gateup_kernel
from coreai_q4_metal import MetalPackedQ4, get_q4_kernel
from coreai_tensor_matmul import tensor_linear, get_tensor_kernel
from export_coreai_q4_moe import Q4MoE, PROJECTIONS, load_layer, tensor_json
from export_moe import Source, sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_NAMES = ('output', 'ids', 'scores')
MAX_CHUNK = 2048


def grouping_permutations(flat_ids, experts):
    """Stable expert grouping using unique FP32-exact keys, without aten.sort.

    Routed IDs are in [0, experts). Integer multiplication is intentional: only
    the completed exact integer keys are converted to FP32 for supported topk.
    """
    rows = flat_ids.shape[0]
    if flat_ids.dtype != torch.int32 or flat_ids.ndim != 1 or rows < 1:
        raise ValueError('Grouping requires nonempty flat I32 expert IDs')
    if experts < 1 or experts * rows > 2**24:
        raise ValueError('Stable grouping keys exceed exact FP32 integer range')
    positions = torch.arange(rows, dtype=torch.int32, device=flat_ids.device)
    keys = flat_ids * rows + positions
    permutation = torch.topk(keys.float(), rows, largest=False, sorted=True).indices
    inverse = torch.zeros_like(permutation).scatter(0, permutation, positions.long())
    return permutation, inverse


def routing_statistics(ids, experts, block):
    """CPU fixture diagnostics, outside the exported model graph."""
    counts = torch.bincount(ids.reshape(-1).long(), minlength=experts)
    active = counts[counts > 0]
    tiles = int(torch.div(counts + block - 1, block, rounding_mode='floor').sum())
    return {'assignments': ids.numel(), 'activeExperts': active.numel(),
            'minimumAssignmentsPerActiveExpert': int(active.min()),
            'maximumAssignmentsPerActiveExpert': int(active.max()),
            'medianAssignmentsPerActiveExpert': float(active.float().median()),
            'plannedTiles': tiles, 'tileRowUtilization': ids.numel() / (tiles * block),
            'expertAssignmentCounts': counts.tolist()}


class ChunkQ4MoE(torch.nn.Module):
    """Adopts an existing Q4MoE; all weight buffers retain their storage.

    Construction replaces the adopted module's selected projections with the
    already supported S1 Metal wrappers. Do not reuse it as an unpack baseline.
    """
    def __init__(self, original: Q4MoE, block=16, columns=64, inner=128, fuse_gateup=False):
        super().__init__()
        self.experts, self.hidden, self.top_k = original.experts, original.hidden, original.top_k
        self.block, self.columns, self.inner = block, columns, inner
        self.fuse_gateup = fuse_gateup
        get_plan_kernel(self.experts, block)
        get_grouped_kernel(block, columns, inner)
        if fuse_gateup:
            get_gateup_kernel(block, columns, inner)
        for name in PROJECTIONS:
            projection = getattr(original, name)
            if not isinstance(projection, MetalPackedQ4):
                setattr(original, name, MetalPackedQ4.from_packed(projection))
        self.decode = original

    def custom_kernels(self):
        kernels = [get_plan_kernel(self.experts, self.block),
                get_grouped_kernel(self.block, self.columns, self.inner),
                get_tensor_kernel(), get_q4_kernel()]
        if self.fuse_gateup:
            kernels.append(get_gateup_kernel(self.block, self.columns, self.inner))
        return kernels

    def routing(self, x):
        count = x.shape[1]
        logits = tensor_linear(x, self.decode.router).reshape(count, self.experts)
        remaining = logits.float()
        selected = []
        for _ in range(self.top_k):
            maximum = remaining.amax(dim=-1, keepdim=True)
            chosen = torch.where(remaining == maximum, self.decode.expert_ids, self.experts).amin(dim=-1)
            selected.append(chosen)
            remaining = torch.where(self.decode.expert_ids == chosen[:, None], float('-inf'), remaining)
        ids = torch.stack(selected, dim=-1)
        scores = torch.gather(logits.float().softmax(-1).half(), -1, ids.long())
        denominator = scores[:, 0]
        for slot in range(1, self.top_k):
            denominator = (denominator + scores[:, slot]).half()
        return ids, (scores / denominator[:, None]).half()

    def grouped(self, name, x, plan):
        projection = getattr(self.decode, name)
        return grouped_linear(x, plan, projection.packed, projection.scales, projection.biases,
                              self.block, self.columns, self.inner)

    def forward(self, x):
        if x.dtype != torch.float16 or x.ndim != 3 or x.shape[0] != 1 or x.shape[2] != self.hidden or not 1 <= x.shape[1] <= MAX_CHUNK:
            raise ValueError('Expected FP16 x[1,S,hidden] with 1 <= S <= 2048')
        if x.shape[1] == 1:
            return self.decode(x)[:3]
        count = x.shape[1]
        ids, scores = self.routing(x)
        flat_ids = ids.reshape(-1)
        # Ties can occur between token assignments to the same expert. Stable
        # sorting preserves token order; routing ties themselves were resolved
        # by explicit lowest expert ID before this execution-only permutation.
        permutation, inverse = grouping_permutations(flat_ids, self.experts)
        sorted_ids = torch.index_select(flat_ids, 0, permutation)
        tokens = torch.div(permutation, self.top_k, rounding_mode='floor')
        ordered_x = torch.index_select(x.reshape(count, self.hidden), 0, tokens)
        plan = make_plan(sorted_ids, self.experts, self.block)
        if self.fuse_gateup:
            gate, up = self.decode.gate_proj, self.decode.up_proj
            active = fused_grouped_gateup(ordered_x, plan, gate.packed, gate.scales, gate.biases,
                up.packed, up.scales, up.biases, self.block, self.columns, self.inner)
        else:
            gate = self.grouped('gate_proj', ordered_x, plan)
            up = self.grouped('up_proj', ordered_x, plan)
            active = ((gate * gate.sigmoid()).half() * up).half()
        ordered_down = self.grouped('down_proj', active, plan)
        down = torch.index_select(ordered_down, 0, inverse).reshape(count, self.top_k, self.hidden)
        routed = (down * scores[:, :, None]).half().float().sum(1).half().reshape(1, count, self.hidden)
        shared_gate = tensor_linear(x, self.decode.shared_gate_proj)
        shared_up = tensor_linear(x, self.decode.shared_up_proj)
        shared_active = ((shared_gate * shared_gate.sigmoid()).half() * shared_up).half()
        shared_down = tensor_linear(shared_active, self.decode.shared_down_proj)
        shared_score = tensor_linear(x, self.decode.shared_router).sigmoid()
        output = (routed + (shared_down * shared_score).half()).half()
        return output, ids.reshape(1, count, self.top_k), scores.reshape(1, count, self.top_k)


def make_synthetic(seed=3719):
    """Asymmetric weights with paired/tied routes and unused expert IDs."""
    experts, hidden, intermediate, shared_intermediate, top_k = 12, 64, 128, 96, 10
    rng = np.random.default_rng(seed)
    router = np.zeros((experts, hidden), dtype=np.float32)
    router[:, 0] = (np.arange(experts) // 2) * 0.25
    router[:, 1] = (np.arange(experts) % 3) * 0.125
    shared_router = (rng.normal(size=(1, hidden)) * 0.035).astype(np.float32)
    shared, quantized = {}, {}
    for name in PROJECTIONS:
        n, k = (hidden, intermediate) if name == 'down_proj' else (intermediate, hidden)
        shape = (hidden, shared_intermediate) if name == 'down_proj' else (shared_intermediate, hidden)
        shared[name] = (rng.normal(size=shape) * 0.035).astype(np.float32)
        packed = rng.integers(0, 2**32, (experts, n, k // 8), dtype=np.uint32)
        scales = np.full((experts, n, k // 64), 0.0078125, dtype=np.float32)
        quantized[name] = packed, scales, -7 * scales
    return Q4MoE(router, shared_router, shared, quantized, top_k).eval()


def compare(actual, expected):
    a, b = actual.double(), expected.double()
    delta = a - b
    norm = torch.linalg.vector_norm(b)
    return {'maxAbsoluteError': float(delta.abs().max()),
            'relativeL2Error': float(torch.linalg.vector_norm(delta) / norm) if norm else None,
            'exact': bool(torch.equal(actual, expected))}


def export_model(output, original, *, counts, source=None, block=16, columns=64, inner=128, fuse_gateup=False):
    import coreai_torch
    if not counts or min(counts) < 1 or max(counts) > MAX_CHUNK or len(set(counts)) != len(counts):
        raise ValueError('Unique token counts in 1...2048 required')
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    module = ChunkQ4MoE(original, block, columns, inner, fuse_gateup).eval()
    generator = torch.Generator(device='cpu').manual_seed(611297)
    longest = (torch.randn(1, max(counts), module.hidden, generator=generator) * 0.125).half()
    # Include an exact all-expert tie and contrasting signed routing inputs.
    longest[:, 0] = 0
    if source is None and max(counts) > 2:
        longest[0, 1, :2] = torch.tensor([0.5, 0], dtype=torch.float16)
        longest[0, 2, :2] = torch.tensor([-0.5, 0], dtype=torch.float16)
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    converter.register_custom_kernels(module.custom_kernels())
    cases = []
    for index, count in enumerate(counts):
        function = 'main' if count == 1 or (1 not in counts and index == 0) else f's{count}'
        example = longest[:, :count].contiguous()
        converter.add_pytorch_module(module, entrypoint_name=function, input_names=('x',), output_names=OUTPUT_NAMES,
            export_fn=lambda m, example=example: torch.export.export(m,args=(example,)).run_decompositions(coreai_torch.get_decomp_table()))
        with torch.inference_mode():
            result = module(example)
            if not all(bool(torch.isfinite(t).all()) for t in result):
                raise ValueError('Nonfinite CPU MoE oracle')
            checks = []
            # Bounded independent token replay avoids the original S*topK dense
            # expert expansion, especially for the full 512-expert real bank.
            for row in sorted(set([0, min(2, count - 1), count - 1])):
                reference = module.decode(example[:, row:row + 1])[:3]
                checks.append({'row': row,
                    'output': compare(result[0][:, row:row+1], reference[0]),
                    'idsExact': bool(torch.equal(result[1][:, row:row+1], reference[1])),
                    'scores': compare(result[2][:, row:row+1], reference[2])})
            filename = f'random-s{count}.json'
            write_json(output / filename, {'inputs':{'x':tensor_json(example)},
                'expectedOutputs':dict(zip(OUTPUT_NAMES, map(tensor_json, result)))})
            cases.append({'name': f'random-s{count}', 'function': function, 'fixture': filename,
                          'cpuTokenChecks': checks,
                          'routingStatistics': routing_statistics(result[1], module.experts, module.block)})
        print(f'CPU fixture S{count} ready ({time.perf_counter()-started:.2f}s)', flush=True)
        gc.collect()
    program = converter.to_coreai()
    program.optimize()
    asset = output / 'moe-chunk.aimodel'
    program.save_asset(asset)
    sources = []
    for kernel in module.custom_kernels():
        for kernel_id, text in kernel.kernel_cache.values():
            filename = kernel_id + '.metal'
            (output / filename).write_text(text)
            sources.append(filename)
    manifest = {'version':1,'status':'cpu-authored-device-unvalidated','model':asset.name,
        'expertCount':module.experts,'topK':module.top_k,'hiddenSize':module.hidden,'counts':list(counts),
        'tile':[block,columns,inner],'fusedGateUp':fuse_gateup,
        'routedKernelCallsPerChunk':2 if fuse_gateup else 3,
        'cases':cases,'planCountPerChunk':1,'routedProjectionsPerChunk':3,'sources':sources,
        'authoringSeconds':time.perf_counter()-started,'deviceValidated':False,
        'provenance':'Real layer weights with synthetic varied CPU activations' if source is not None else 'Synthetic asymmetric weights and explicit routing ties',
        'limitations':['The CPU grouped reference is bounded but still dequantizes experts for its oracle.',
                       'No GPU compilation, numerical or throughput acceptance is implied by export.',
                       'MPP summation order can differ from S1 and CPU BLAS.',
                       'Converter lacks sort; grouping uses full topk on unique FP32-exact integer keys, bounded below 2**24.'],
        'files':[{'path':str(p.relative_to(asset)),'bytes':p.stat().st_size,'sha256':sha256_file(p)}
                 for p in sorted(asset.rglob('*')) if p.is_file()]}
    if source is not None:
        manifest.update(modelDirectory=str(source.directory),configSHA256=sha256_file(source.directory/'config.json'),
                        sourceRecords=source.records,layer=0,completeExpertBank=module.experts==512)
    write_json(output / 'manifest.json',manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--real-layer0',action='store_true')
    parser.add_argument('--counts',type=int,nargs='+',default=[1,33,128])
    parser.add_argument('--block',type=int,default=16)
    parser.add_argument('--columns',type=int,default=64)
    parser.add_argument('--inner',type=int,default=128)
    parser.add_argument('--fuse-gateup',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    source=Source(0) if args.real_layer0 else None
    original=load_layer(source,512,10) if source is not None else make_synthetic()
    report=export_model(args.output,original,counts=tuple(args.counts),source=source,
        block=args.block,columns=args.columns,inner=args.inner,fuse_gateup=args.fuse_gateup)
    print(json.dumps({'model':str(args.output/report['model']),'counts':report['counts'],
                      'authoringSeconds':report['authoringSeconds']},indent=2))


if __name__=='__main__':main()
