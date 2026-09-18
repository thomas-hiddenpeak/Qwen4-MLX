#!/usr/bin/env python3
"""Independent SG1/2/4 BM16 gate/up probe with native-parity-v2 output math.

Only the N tile and SIMD groups per threadgroup change:32/64/128 columns.
Each SIMD retains the same M16xN32xK16 register MMA and three volatile half
boundaries. The original BM16 expert plan is shared by all variants. No default
kernel, installer, shared exporter or model module is changed by this probe.
"""
from __future__ import annotations

import argparse
from functools import cache
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from coreai_q4_flat import _replace_once, _validate
from coreai_q4_grouped import get_plan_kernel, make_plan, plan_reference
from coreai_q4_nax import MLX_NOTICE
from coreai_q4_nax_gateup import NAMES
from coreai_q4_nax_gateup_diagnostic import reference_stages
from coreai_q4_nax_gateup_parity import get_kernel as get_sg1_kernel, source_text as sg1_source
from coreai_q4_nax_gateup_m32_probe import tiny_inputs


def source_text(outputs, inputs, simdgroups):
    if simdgroups not in (1, 2, 4):
        raise ValueError('Expected1/2/4 SIMD groups')
    # The stable source computes simd=thread_id/32, maps each SIMD to its own
    #32 N columns and stages one output weight row per thread. Consequently
    #only BN and the launch size need to grow. Keep all measured parity math.
    return _replace_once(sg1_source(outputs, inputs), 'BN=32*1;', f'BN=32*{simdgroups};')


def get_kernel(experts, outputs, inputs, simdgroups):
    if simdgroups == 1:
        return get_sg1_kernel(experts, outputs, inputs)
    return _get_kernel(experts, outputs, inputs, simdgroups)


@cache
def _get_kernel(experts, outputs, inputs, simdgroups):
    from coreai.authoring import MetalParameter
    from coreai_torch import TorchMetalKernel
    if not 1 <= experts <= 512:
        raise ValueError('Expected1...512 experts')
    def reference(x: torch.Tensor, plan: torch.Tensor, gate_packed: torch.Tensor,
                  gate_scales: torch.Tensor, gate_biases: torch.Tensor,
                  up_packed: torch.Tensor, up_scales: torch.Tensor, up_biases: torch.Tensor) -> torch.Tensor:
        _, _, up_float, gate_half, _, sigmoid_half, _ = reference_stages(x, plan,
            gate_packed, gate_scales, gate_biases, up_packed, up_scales, up_biases,
            (experts, outputs, inputs))
        return ((gate_half.float() * up_float).half().float() * sigmoid_half.float()).half()
    return TorchMetalKernel(f'qwen_experimental_q4_nax_gateup_m16_sg{simdgroups}_parity_e{experts}_n{outputs}_k{inputs}_v1',
        input_names=['x', 'plan', *NAMES[2:]], result_names=['output'],
        src=source_text(outputs, inputs, simdgroups), helper_src=MLX_NOTICE, torch_defn=reference,
        metal_params=[MetalParameter('group', 'uint3', 'threadgroup_position_in_grid'),
                      MetalParameter('thread_id', 'uint', 'thread_index_in_threadgroup')])


class Projection(torch.nn.Module):
    def __init__(self, geometry, simdgroups):
        super().__init__()
        if simdgroups not in (1, 2, 4):
            raise ValueError('Expected1/2/4 SIMD groups')
        self.geometry, self.simdgroups = geometry, simdgroups

    def forward(self, x, ids, gate_packed, gate_scales, gate_biases, up_packed, up_scales, up_biases):
        for packed, scales, biases in ((gate_packed, gate_scales, gate_biases), (up_packed, up_scales, up_biases)):
            _validate(x, packed, scales, biases, *self.geometry)
        if x.ndim != 2 or ids.dtype != torch.int32 or ids.shape != (x.shape[0],):
            raise ValueError('Expected FP16 x[R,K] and sorted I32 ids[R]')
        experts, outputs, _ = self.geometry
        plan = make_plan(ids, experts, 16)
        threads = columns = self.simdgroups * 32
        output = get_kernel(*self.geometry, self.simdgroups)(x, plan,
            gate_packed, gate_scales, gate_biases, up_packed, up_scales, up_biases,
            threads_per_grid=(((outputs + columns - 1) // columns) * threads, plan.shape[0] - 1, 1),
            threads_per_thread_group=(threads, 1, 1), result_shapes=[[x.shape[0], outputs]])
        return output, plan


def export(output, fixture=None):
    import coreai_torch
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if fixture:
        fixture = fixture.resolve()
        inputs = json.loads((fixture / 'baseline-spec.json').read_text())['inputs']
        geometry = (512, 640, 2560)
        if inputs['x']['shape'] != [20480, 2560]:
            raise ValueError('Expected existing real S2048 grouped gate/up fixture')
        for entry in inputs.values():
            entry['file'] = str((fixture / entry['file']).resolve())
        examples = tuple(torch.empty(inputs[name]['shape'], dtype=getattr(torch, inputs[name]['dtype'])) for name in NAMES)
        meta = inputs['ids']
        ids = torch.from_numpy(np.fromfile(meta['file'], dtype=np.int32, count=meta['bytes'] // 4, offset=meta['offset']))
    else:
        geometry = (5, 67, 128)
        examples = tiny_inputs()
        ids = examples[1]
        inputs = {}
        for name, value in zip(NAMES, examples):
            value.numpy().tofile(output / (name + '.bin'))
            inputs[name] = {'file': name + '.bin', 'offset': 0, 'bytes': value.numel() * value.element_size(),
                            'shape': list(value.shape), 'dtype': str(value.dtype).removeprefix('torch.')}
        with torch.inference_mode():
            expected = Projection(geometry, 1)(*examples)
            for sg in (2, 4):
                for a, b in zip(Projection(geometry, sg)(*examples), expected):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            expected[0].numpy().tofile(output / 'expected-output.bin')
    kernels = [get_plan_kernel(geometry[0], 16), *[get_kernel(*geometry, sg) for sg in (1, 2, 4)]]
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    converter.register_custom_kernels(kernels)
    for sg in (1, 2, 4):
        converter.add_pytorch_module(Projection(geometry, sg), entrypoint_name=f'sg{sg}',
            input_names=NAMES, output_names=('output', 'plan'),
            export_fn=lambda current: torch.export.export(current, args=examples).run_decompositions(coreai_torch.get_decomp_table()))
    program = converter.to_coreai()
    program.optimize()
    program._mlir_module.operation.verify()
    asset = output / 'gateup.aimodel'
    program.save_asset(asset)
    for kernel in kernels:
        for name, source in kernel.kernel_cache.values():
            (output / (name + '.metal')).write_text(source)
    plan = plan_reference(ids, geometry[0], 16)
    plan.numpy().tofile(output / 'expected-plan.bin')
    for sg in (1, 2, 4):
        spec = {'asset': asset.name, 'function': f'sg{sg}', 'inputs': inputs,
                'output': f'sg{sg}-output', 'repeats': 15, 'mapped': False, 'oneBufferPerFile': False}
        (output / f'sg{sg}-spec.json').write_text(json.dumps(spec, indent=2) + '\n')
    sources = ('coreai_q4_nax_gateup', 'coreai_q4_nax_gateup_parity', 'coreai_q4_nax_gateup_diagnostic',
               'coreai_q4_nax', 'coreai_q4_nax_gateup_simd_probe')
    report = {'status': 'cpu-authored-device-unvalidated', 'productionEnabled': False,
        'geometry': dict(zip(('E', 'N', 'K'), geometry)),
        'plan': {'shape': list(plan.shape), 'activeTiles': int(plan[0, 0]), 'BM': 16, 'sameAllVariants': True},
        'variants': [{'function': f'sg{sg}', 'simdgroups': sg, 'columns': sg * 32,
                      'threads': sg * 32, 'weightThreadgroupBytes': sg * 32 * 72 * 2} for sg in (1, 2, 4)],
        'assetBytes': sum(path.stat().st_size for path in asset.rglob('*') if path.is_file()),
        'sourceSHA256': {name: hashlib.sha256(Path(__file__).with_name(name + '.py').read_bytes()).hexdigest() for name in sources},
        'fixtureSource': str(fixture) if fixture else 'tiny expert counts1/16/17/32/33, N67/K128',
        'policy': 'native-parity-v2: half(half(gate_half*up_accum_float)*sigmoid_half)',
        'CPUOutputExact': fixture is None,
        'limitations': ['GPU compilation/numerics/performance not yet validated.',
            'Larger threadgroup storage and synchronization can offset reduced group count.']}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    report = export(args.output, args.fixture)
    print(json.dumps({key: report[key] for key in ('status', 'geometry', 'plan', 'assetBytes')}, indent=2))
