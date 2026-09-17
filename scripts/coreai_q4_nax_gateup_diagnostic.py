#!/usr/bin/env python3
"""Observe grouped gate/up dot products and activation stages without GPU use.

Diagnostic clones retain the original arithmetic and append stores for both
FP32 accumulators, their FP16 casts, sigmoid, and SiLU. Added stores can change
compiler optimization, so compare the diagnostic final result with the prior
non-diagnostic pair too. Separate single-projection entrypoints avoid fusion.
"""
from __future__ import annotations

import argparse
from functools import cache
import json
from pathlib import Path

import torch
from torch._subclasses.fake_tensor import FakeTensor

from coreai_q4_flat import _grouped_source, _replace_once, _reshape, get_flat_grouped_kernel
from coreai_q4_gateup import GATEUP_SOURCE
from coreai_q4_grouped import get_plan_kernel, make_plan
from coreai_q4_nax import MLX_NOTICE, get_kernel as get_single_kernel
from coreai_q4_nax_gateup import NAMES, source_text


OUTPUT_NAMES = ('output', 'gate_f32', 'up_f32', 'gate_half', 'up_half', 'sigmoid_half', 'silu_half')


def diagnostic_source(experts, outputs, inputs, candidate):
    if candidate:
        source = source_text(outputs, inputs, 1)
        position = 'n,start+row'
        target = 'yp[(start+row)*y_stride+n]=half(float(silu)*float(up));'
    else:
        source = _grouped_source(GATEUP_SOURCE, experts, outputs, inputs,
                                 16, 32, 64, ('gate_', 'up_'))
        position = 'col+n,start+m'
        target = 'output[col+n,start+m]=half(float(silu)*float(up));'
    stores = '\n'.join(f'{name}[{position}]={value};' for name, value in (
        ('gate_f32', 'gate_accum[i]'), ('up_f32', 'up_accum[i]'),
        ('gate_half', 'gate'), ('up_half', 'up'), ('sigmoid_half', 'sigmoid'), ('silu_half', 'silu')))
    return _replace_once(source, target, target+'\n'+stores)


def reference_stages(x, plan, gate_packed, gate_scales, gate_biases,
                     up_packed, up_scales, up_biases, geometry):
    experts, outputs, inputs = geometry
    shape = (x.shape[0], outputs)
    if isinstance(x, FakeTensor) or x.device.type == 'meta':
        return tuple(torch.empty(shape, device=x.device,
                     dtype=torch.float32 if name.endswith('f32') else torch.float16)
                     for name in OUTPUT_NAMES)
    accumulators = []
    shift = torch.arange(0, 16, 4, dtype=torch.int32)
    for packed, scales, biases in ((gate_packed, gate_scales, gate_biases),
                                    (up_packed, up_scales, up_biases)):
        packed, scales, biases = _reshape(packed, scales, biases, *geometry)
        result = torch.empty(shape, dtype=torch.float32)
        for expert, start, count, _ in plan[1:1+int(plan[0, 0])].tolist():
            codes = ((packed[expert].int().unsqueeze(-1)>>shift)&15).reshape(outputs, -1, 64).float()
            weight = (codes*scales[expert].float().unsqueeze(-1)+biases[expert].float().unsqueeze(-1)).half().flatten(-2)
            result[start:start+count] = torch.nn.functional.linear(x[start:start+count].float(), weight.float())
        accumulators.append(result)
    gate_f32, up_f32 = accumulators
    gate, up = gate_f32.half(), up_f32.half()
    sigmoid = (1/(1+torch.exp(-gate.float()))).half()
    silu = (gate.float()*sigmoid.float()).half()
    output = (silu.float()*up.float()).half()
    return output, gate_f32, up_f32, gate, up, sigmoid, silu


@cache
def get_kernel(experts, outputs, inputs, candidate):
    from coreai.authoring import MetalParameter
    from coreai_torch import TorchMetalKernel

    def reference(x: torch.Tensor, plan: torch.Tensor, gate_packed: torch.Tensor,
                  gate_scales: torch.Tensor, gate_biases: torch.Tensor,
                  up_packed: torch.Tensor, up_scales: torch.Tensor,
                  up_biases: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                   torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return reference_stages(x, plan, gate_packed, gate_scales, gate_biases,
                                up_packed, up_scales, up_biases, (experts, outputs, inputs))

    tag = 'nax' if candidate else 'flat'
    return TorchMetalKernel(f'qwen_q4_gateup_stages_{tag}_e{experts}_n{outputs}_k{inputs}_v1',
        input_names=['x', 'plan', *NAMES[2:]], result_names=list(OUTPUT_NAMES),
        src=diagnostic_source(experts, outputs, inputs, candidate),
        helper_src=MLX_NOTICE if candidate else '', torch_defn=reference,
        metal_params=[MetalParameter('group', 'uint3', 'threadgroup_position_in_grid'),
                      MetalParameter('thread_id', 'uint', 'thread_index_in_threadgroup')])


class Probe(torch.nn.Module):
    def __init__(self, geometry, candidate, singles=False):
        super().__init__()
        self.geometry, self.candidate, self.singles = geometry, candidate, singles

    def forward(self, x, ids, gate_packed, gate_scales, gate_biases, up_packed, up_scales, up_biases):
        experts, outputs, _ = self.geometry
        plan = make_plan(ids, experts, 16)
        threads = 32 if self.candidate else 128
        launch = dict(threads_per_grid=(((outputs+31)//32)*threads, plan.shape[0]-1, 1),
                      threads_per_thread_group=(threads, 1, 1))
        if self.singles:
            kernel = get_single_kernel(*self.geometry, 1, True) if self.candidate else get_flat_grouped_kernel(*self.geometry)
            gate = kernel(x, plan, gate_packed, gate_scales, gate_biases,
                          **launch, result_shapes=[[x.shape[0], outputs]])
            up = kernel(x, plan, up_packed, up_scales, up_biases,
                        **launch, result_shapes=[[x.shape[0], outputs]])
            return gate, up, plan
        values = get_kernel(*self.geometry, self.candidate)(x, plan,
            gate_packed, gate_scales, gate_biases, up_packed, up_scales, up_biases,
            **launch, result_shapes=[[x.shape[0], outputs]]*len(OUTPUT_NAMES))
        return (*values, plan)


def main():
    import coreai_torch
    from export_coreai_q4_moe import tensor_json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    previous = args.fixture.resolve()
    inputs = json.loads((previous/'baseline-spec.json').read_text())['inputs']
    k = inputs['x']['shape'][1]
    if k == 128:
        geometry = (5, 67, 128)
    elif k == 2560:
        geometry = (512, 640, 2560)
    else:
        raise ValueError('Expected existing tiny or real gate/up fixture')
    examples = tuple(torch.empty(inputs[name]['shape'], dtype=getattr(torch, inputs[name]['dtype'])) for name in NAMES)
    args.output.mkdir(parents=True, exist_ok=False)
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    kernels = [get_kernel(*geometry, candidate) for candidate in (False, True)]
    converter.register_custom_kernels([get_plan_kernel(geometry[0], 16),
        get_single_kernel(*geometry, 1, True), get_flat_grouped_kernel(*geometry), *kernels])
    entries = [('baseline', False, False), ('candidate', True, False),
               ('single_baseline', False, True), ('single_candidate', True, True)]
    for name, candidate, singles in entries:
        converter.add_pytorch_module(Probe(geometry, candidate, singles).eval(),
            entrypoint_name=name, input_names=NAMES,
            output_names=('gate_half', 'up_half', 'plan') if singles else (*OUTPUT_NAMES, 'plan'),
            export_fn=lambda module: torch.export.export(module, args=examples).run_decompositions(coreai_torch.get_decomp_table()))
    program = converter.to_coreai()
    program.optimize()
    program.save_asset(args.output/'diagnostic.aimodel')
    for kernel in kernels:
        for name, source in kernel.kernel_cache.values():
            (args.output/(name+'.metal')).write_text(source)
    for value in inputs.values():
        value['file'] = str((previous/value['file']).resolve())
    for name, _, _ in entries:
        (args.output/(name+'-spec.json')).write_text(json.dumps({'asset': 'diagnostic.aimodel',
            'function': name, 'inputs': inputs, 'output': name+'-output', 'repeats': 2,
            'mapped': False, 'oneBufferPerFile': False}, indent=2)+'\n')
    if geometry[0] == 5:
        import numpy as np
        examples = tuple(torch.from_numpy(np.fromfile(inputs[name]['file'],
                    dtype=inputs[name]['dtype'], count=inputs[name]['bytes']//torch.empty((), dtype=getattr(torch, inputs[name]['dtype'])).element_size(),
                    offset=inputs[name]['offset']).reshape(inputs[name]['shape']).copy()) for name in NAMES)
        with torch.inference_mode():
            expected = Probe(geometry, False)(*examples)
            candidate = Probe(geometry, True)(*examples)
        assert all(torch.equal(a, b) for a, b in zip(expected, candidate, strict=True))
        (args.output/'actual.json').write_text(json.dumps({'expectedOutputs':
            dict(zip((*OUTPUT_NAMES, 'plan'), map(tensor_json, expected), strict=True))})+'\n')
    (args.output/'manifest.json').write_text(json.dumps({'status': 'CPU-authored-device-unvalidated',
        'geometry': dict(zip(('E', 'N', 'K'), geometry)), 'fixtureSource': str(previous),
        'caution': 'Diagnostic stores can change optimization. Compare final output to prior uninstrumented pair.',
        'CPUOracle': 'FP32 PyTorch dot; same policy but not an oracle for GPU accumulation order'}, indent=2)+'\n')


if __name__ == '__main__':
    main()
