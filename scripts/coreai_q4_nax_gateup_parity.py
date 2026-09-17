#!/usr/bin/env python3
"""Optional NAX gate/up matching measured original GPU fused-output policy.

The original flat kernel was observed to reassociate its product as
half(half(gate_half * up_float_accumulator) * sigmoid_half). This reproduces
all 13,107,200 captured real outputs. Volatile thread-local half boundaries
make that measured policy explicit without extra graph outputs or buffers.
"""
from __future__ import annotations

import argparse
from functools import cache
import json
from pathlib import Path

import torch

from coreai_q4_flat import _replace_once, get_flat_gateup_kernel
from coreai_q4_grouped import get_plan_kernel, make_plan
from coreai_q4_nax import MLX_NOTICE
from coreai_q4_nax_gateup import NAMES, Probe as OriginalProbe, source_text as original_source
from coreai_q4_nax_gateup_diagnostic import reference_stages


def source_text(outputs, inputs):
    return _replace_once(original_source(outputs, inputs, 1),
        '''const half gate=half(gate_accum[i]),up=half(up_accum[i]);
    const half sigmoid=half(1.0f/(1.0f+exp(-float(gate))));
    const half silu=half(float(gate)*float(sigmoid));
    yp[(start+row)*y_stride+n]=half(float(silu)*float(up));''',
        '''// Match the measured original flat fused graph, including its
    // reassociation and retained FP32 up accumulator. These volatile local
    // stores preserve half rounding even under fast-math reassociation.
    volatile thread half gate_boundary=half(gate_accum[i]);
    const float gate=float(gate_boundary);
    volatile thread half sigmoid_boundary=half(1.0f/(1.0f+exp(-gate)));
    volatile thread half gate_up_boundary=half(gate*up_accum[i]);
    yp[(start+row)*y_stride+n]=half(float(gate_up_boundary)*float(sigmoid_boundary));''')


@cache
def get_kernel(experts, outputs, inputs):
    from coreai.authoring import MetalParameter
    from coreai_torch import TorchMetalKernel

    def reference(x: torch.Tensor, plan: torch.Tensor, gate_packed: torch.Tensor,
                  gate_scales: torch.Tensor, gate_biases: torch.Tensor,
                  up_packed: torch.Tensor, up_scales: torch.Tensor,
                  up_biases: torch.Tensor) -> torch.Tensor:
        stages = reference_stages(x, plan, gate_packed, gate_scales, gate_biases,
                                  up_packed, up_scales, up_biases, (experts, outputs, inputs))
        _, _, up_f32, gate_half, _, sigmoid_half, _ = stages
        return ((gate_half.float()*up_f32).half().float()*sigmoid_half.float()).half()

    return TorchMetalKernel(f'qwen_q4_nax_gateup_native_parity_e{experts}_n{outputs}_k{inputs}_v2',
        input_names=['x', 'plan', *NAMES[2:]], result_names=['output'],
        src=source_text(outputs, inputs), helper_src=MLX_NOTICE, torch_defn=reference,
        metal_params=[MetalParameter('group', 'uint3', 'threadgroup_position_in_grid'),
                      MetalParameter('thread_id', 'uint', 'thread_index_in_threadgroup')])


class Probe(torch.nn.Module):
    def __init__(self, geometry):
        super().__init__()
        self.geometry = geometry

    def forward(self, x, ids, gate_packed, gate_scales, gate_biases, up_packed, up_scales, up_biases):
        experts, outputs, _ = self.geometry
        plan = make_plan(ids, experts, 16)
        output = get_kernel(*self.geometry)(x, plan, gate_packed, gate_scales, gate_biases,
            up_packed, up_scales, up_biases,
            threads_per_grid=(((outputs+31)//32)*32, plan.shape[0]-1, 1),
            threads_per_thread_group=(32, 1, 1), result_shapes=[[x.shape[0], outputs]])
        return output, plan


def main():
    import coreai_torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    previous = args.fixture.resolve()
    inputs = json.loads((previous/'baseline-spec.json').read_text())['inputs']
    geometry = (5, 67, 128) if inputs['x']['shape'][1] == 128 else (512, 640, 2560)
    if inputs['x']['shape'][1] != geometry[2]:
        raise ValueError('Expected existing tiny or real gate/up fixture')
    examples = tuple(torch.empty(inputs[name]['shape'], dtype=getattr(torch, inputs[name]['dtype'])) for name in NAMES)
    args.output.mkdir(parents=True, exist_ok=False)
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    kernel = get_kernel(*geometry)
    converter.register_custom_kernels([get_plan_kernel(geometry[0], 16),
                                       get_flat_gateup_kernel(*geometry), kernel])
    for name, module in [('baseline', OriginalProbe(geometry, False)), ('candidate', Probe(geometry))]:
        converter.add_pytorch_module(module.eval(), entrypoint_name=name, input_names=NAMES,
            output_names=('output', 'plan'),
            export_fn=lambda current: torch.export.export(current, args=examples).run_decompositions(coreai_torch.get_decomp_table()))
    program = converter.to_coreai()
    program.optimize()
    program.save_asset(args.output/'gateup.aimodel')
    for name, source in kernel.kernel_cache.values():
        (args.output/(name+'.metal')).write_text(source)
    for value in inputs.values():
        value['file'] = str((previous/value['file']).resolve())
    for name in ('baseline', 'candidate'):
        (args.output/(name+'-spec.json')).write_text(json.dumps({'asset': 'gateup.aimodel',
            'function': name, 'inputs': inputs, 'output': name+'-output', 'repeats': 12,
            'mapped': False, 'oneBufferPerFile': False}, indent=2)+'\n')
    (args.output/'manifest.json').write_text(json.dumps({'status': 'CPU-authored-device-unvalidated',
        'geometry': dict(zip(('E', 'N', 'K'), geometry)), 'fixtureSource': str(previous),
        'policy': 'native-parity-v2', 'rounding': 'half(half(gate_half*up_accum_float)*sigmoid_half)',
        'CPUCallback': 'Measured native policy, intentionally different from source half-SiLU then half-up policy',
        'noExtraOutputs': True, 'noExtraBuffers': True}, indent=2)+'\n')


if __name__ == '__main__':
    main()
