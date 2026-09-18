#!/usr/bin/env python3
"""Fused router v3 matching observed native GPU score accumulation policy.

This intentionally differs from source-level sequential FP16 addition: CoreAI's
native graph was measured to use FP32 selected-score sum followed by one FP16
cast. Full512 FP32 softmax still rounds each probability to FP16 first. Final
division uses precise FP32 divide then FP16. CPU callback describes this policy.
"""
from __future__ import annotations

import argparse
from functools import cache
import json
from pathlib import Path

import numpy as np
import torch

import coreai_router_metal as original
from coreai_tensor_matmul import get_tensor_kernel, tensor_linear

_OLD_SUM = 'half total=selected[0];\n  for(int slot=1;slot<10;++slot)total=half(float(total)+float(selected[slot]));'
_NEW_SUM = '''float total_fp32=float(selected[0]);
  for(int slot=1;slot<10;++slot)total_fp32+=float(selected[slot]);
  const half total=half(total_fp32);'''
assert original.ROUTER_SOURCE.count(_OLD_SUM) == 1
NATIVE_SOURCE = original.ROUTER_SOURCE.replace(_OLD_SUM, _NEW_SUM)


def native_stages(logits):
    ids = original.original_ids(logits)
    probabilities = logits.float().softmax(-1).half()
    selected = torch.gather(probabilities, -1, ids.long())
    normalizer = selected.float().sum(-1, keepdim=True).half()
    scores = (selected.float()/normalizer.float()).half()
    return ids, scores, probabilities, normalizer


def get_kernel(diagnostic=False):
    return _get_kernel(diagnostic)


@cache
def _get_kernel(diagnostic):
    from coreai.authoring import MetalParameter
    from coreai_torch import TorchMetalKernel
    source = NATIVE_SOURCE
    if diagnostic:
        source = source.replace('const half probability=half(exponential/denominator);',
            'const half probability=half(exponential/denominator);\nprobabilities[expert,token]=probability;')
        source = source.replace('for(int slot=0;slot<10;++slot)scores[slot,token]=',
            'normalizer[0,token]=total;\n  for(int slot=0;slot<10;++slot)scores[slot,token]=')

        def reference(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            return native_stages(logits)
    else:
        def reference(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return native_stages(logits)[:2]
    names = ['ids', 'scores', 'probabilities', 'normalizer'] if diagnostic else ['ids', 'scores']
    return TorchMetalKernel('qwen_router_native_parity_e512_f16'+('_diagnostic' if diagnostic else '')+'_v3',
        input_names=['logits'], result_names=names, src=source, torch_defn=reference,
        metal_params=[MetalParameter('group', 'uint3', 'threadgroup_position_in_grid'),
                      MetalParameter('thread_id', 'uint', 'thread_index_in_threadgroup')])


def route(logits, diagnostic=False):
    original._shape(logits)
    count = logits.shape[0]
    shapes = [[count, 10], [count, 10]] + ([[count, 512], [count, 1]] if diagnostic else [])
    return get_kernel(diagnostic)(logits, threads_per_grid=(count*512, 1, 1),
        threads_per_thread_group=(512, 1, 1), result_shapes=shapes)


class Probe(torch.nn.Module):
    def __init__(self, candidate, diagnostic=False):
        super().__init__()
        self.candidate, self.diagnostic = candidate, diagnostic

    def forward(self, logits):
        if self.candidate:
            return route(logits, self.diagnostic)
        if self.diagnostic:
            return original.DiagnosticProbe(False)(logits)
        return original.original_routing(logits)


class FullProbe(Probe):
    def forward(self, x, weight):
        return super().forward(tensor_linear(x, weight).reshape(x.shape[1], 512))


def export(output, examples, full=False, diagnostic=False):
    import coreai_torch
    output.mkdir(parents=True, exist_ok=False)
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    kernel = get_kernel(diagnostic)
    converter.register_custom_kernels([kernel, get_tensor_kernel()] if full else [kernel])
    names = ('ids', 'scores', 'probabilities', 'normalizer') if diagnostic else ('ids', 'scores')
    for candidate, name in ((False, 'baseline'), (True, 'candidate')):
        module = (FullProbe if full else Probe)(candidate, diagnostic).eval()
        converter.add_pytorch_module(module, entrypoint_name=name,
            input_names=('x', 'weight') if full else ('logits',), output_names=names,
            export_fn=lambda current: torch.export.export(current, args=examples).run_decompositions(coreai_torch.get_decomp_table()))
    program = converter.to_coreai()
    program.optimize()
    program.save_asset(output/'router.aimodel')
    for name, source in kernel.kernel_cache.values():
        (output/(name+'.metal')).write_text(source)


def main():
    from export_coreai_q4_moe import tensor_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--full-fixture', type=Path, help='Reuse prior full-router binary spec and inputs verbatim')
    parser.add_argument('--diagnostic', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    if args.full_fixture:
        previous = args.full_fixture.resolve()
        spec = json.loads((previous/'baseline-spec.json').read_text())
        inputs = spec['inputs']
        examples = tuple(torch.empty(inputs[name]['shape'], dtype=torch.float16) for name in ('x', 'weight'))
        export(args.output, examples, full=True, diagnostic=args.diagnostic)
        for value in inputs.values():
            value['file'] = str((previous/value['file']).resolve())
        original.write_specs(args.output, inputs)
    else:
        logits = original.boundary_logits()
        export(args.output, (logits,), diagnostic=args.diagnostic)
        logits.numpy().tofile(args.output/'logits.bin')
        original.write_specs(args.output, {'logits': {'file': 'logits.bin', 'offset': 0,
            'bytes': logits.numel()*2, 'shape': list(logits.shape), 'dtype': 'float16'}})
        expected = native_stages(logits)
        names = ('ids', 'scores', 'probabilities', 'normalizer') if args.diagnostic else ('ids', 'scores')
        (args.output/'actual.json').write_text(json.dumps({'inputs': {'logits': tensor_json(logits)},
            'expectedOutputs': {name: tensor_json(value) for name, value in zip(names, expected)}})+'\n')
        replay = route(logits, args.diagnostic)
        assert all(torch.equal(a, b) for a, b in zip(replay, expected[:len(names)], strict=True))
        assert not torch.equal(expected[3], original.score_stages(logits, expected[0])[2])
    (args.output/'manifest.json').write_text(json.dumps({'status': 'CPU-authored-device-unvalidated',
        'version': 3, 'policy': 'native-parity', 'diagnostic': args.diagnostic,
        'softmax': 'FP32 full512 softmax then each probability FP16',
        'denominator': 'FP32 sum of selected FP16 top10 probabilities then one FP16 cast',
        'division': 'precise::divide of explicit FP32 operands then FP16',
        'CPUCallback': 'native-parity policy; deliberately differs from source sequential half additions',
        'fixtureSource': str(args.full_fixture) if args.full_fixture else 'boundary_logits'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
