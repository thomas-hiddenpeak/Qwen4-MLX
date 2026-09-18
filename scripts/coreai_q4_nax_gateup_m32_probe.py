#!/usr/bin/env python3
"""Experimental BM32 NAX gate/up with the measured native-parity-v2 policy.

Two M16 A fragments and two pairs of gate/up accumulators share each BK72
weight tile and B fragment. The per-output K16 order and volatile half output
boundaries remain those of the independently GPU-validated BM16 parity kernel.
Only this independent module is experimental; importing it installs nothing.
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
from coreai_q4_nax_gateup_parity import get_kernel as get_bm16_kernel, source_text as bm16_source
from coreai_q4_nax_m32_probe import check_fragment_mapping


def source_text(outputs, inputs):
    source = bm16_source(outputs, inputs)
    source = _replace_once(source,
        'auto a=operation.get_left_input_cooperative_tensor<half,half,float>();',
        'auto a=operation.get_left_input_cooperative_tensor<half,half,float>();\n'
        'auto a_hi=operation.get_left_input_cooperative_tensor<half,half,float>();')
    for prefix in ('gate_', 'up_'):
        declaration = f'auto {prefix}accum=operation.get_destination_cooperative_tensor<\n' + \
            '    metal::remove_addrspace_t<decltype(a)>,metal::remove_addrspace_t<decltype(b)>,float>();'
        source = _replace_once(source, declaration,
            declaration + '\n' + declaration.replace(prefix + 'accum=', prefix + 'accum_hi='))
        source = _replace_once(source, f'operation.run(a,b,{prefix}accum);',
            f'operation.run(a,b,{prefix}accum);\n    operation.run(a_hi,b,{prefix}accum_hi);')
    source = _replace_once(source, 'for(int i=0;i<16;++i) {gate_accum[i]=0.0f;up_accum[i]=0.0f;}',
        'for(int i=0;i<16;++i) {gate_accum[i]=0.0f;up_accum[i]=0.0f;gate_accum_hi[i]=0.0f;up_accum_hi[i]=0.0f;}')
    source = _replace_once(source, 'thread half cached_a[32];', 'thread half cached_a[32],cached_a_hi[32];')
    source = _replace_once(source, 'cached_a[part*8+i]=row<count ? xp[(start+row)*x_stride+k] : half(0);',
        'cached_a[part*8+i]=row<count ? xp[(start+row)*x_stride+k] : half(0);\n'
        '      cached_a_hi[part*8+i]=row+16<count ? xp[(start+row+16)*x_stride+k] : half(0);')
    if source.count('a[i]=cached_a[part*8+i];') != 2:
        raise ValueError('Expected one A load in each gate/up weight stage')
    source = source.replace('a[i]=cached_a[part*8+i];',
        'a[i]=cached_a[part*8+i];\n      a_hi[i]=cached_a_hi[part*8+i];')
    source = _replace_once(source, '#pragma clang loop unroll(full)\nfor(int i=0;i<16;++i) {\n  const int row=',
        '#pragma clang loop unroll(full)\nfor(int mblock=0;mblock<2;++mblock) {\n'
        '#pragma clang loop unroll(full)\nfor(int i=0;i<16;++i) {\n  const int row=')
    source = _replace_once(source, 'const int row=frag_row+((i%8)/4)*8;',
        'const int row=16*mblock+frag_row+((i%8)/4)*8;')
    source = _replace_once(source, 'volatile thread half gate_boundary=half(gate_accum[i]);',
        'volatile thread half gate_boundary=half(mblock==0 ? gate_accum[i] : gate_accum_hi[i]);')
    source = _replace_once(source, 'volatile thread half gate_up_boundary=half(gate*up_accum[i]);',
        'volatile thread half gate_up_boundary=half(gate*(mblock==0 ? up_accum[i] : up_accum_hi[i]));')
    return source + '\n}\n'


@cache
def get_kernel(experts, outputs, inputs):
    from coreai.authoring import MetalParameter
    from coreai_torch import TorchMetalKernel
    if not 1 <= experts <= 512:
        raise ValueError('Expected1...512 experts')
    def reference(x: torch.Tensor, plan: torch.Tensor, gate_packed: torch.Tensor,
                  gate_scales: torch.Tensor, gate_biases: torch.Tensor,
                  up_packed: torch.Tensor, up_scales: torch.Tensor, up_biases: torch.Tensor) -> torch.Tensor:
        stages = reference_stages(x, plan, gate_packed, gate_scales, gate_biases,
                                  up_packed, up_scales, up_biases, (experts, outputs, inputs))
        _, _, up_f32, gate_half, _, sigmoid_half, _ = stages
        return ((gate_half.float() * up_f32).half().float() * sigmoid_half.float()).half()
    return TorchMetalKernel(f'qwen_experimental_q4_nax_gateup_m32_parity_e{experts}_n{outputs}_k{inputs}_v1',
        input_names=['x', 'plan', *NAMES[2:]], result_names=['output'],
        src=source_text(outputs, inputs), helper_src=MLX_NOTICE, torch_defn=reference,
        metal_params=[MetalParameter('group', 'uint3', 'threadgroup_position_in_grid'),
                      MetalParameter('thread_id', 'uint', 'thread_index_in_threadgroup')])


class Projection(torch.nn.Module):
    def __init__(self, geometry, block):
        super().__init__()
        if block not in (16, 32):
            raise ValueError('Expected BM16 or BM32')
        self.geometry, self.block = geometry, block

    def forward(self, x, ids, gate_packed, gate_scales, gate_biases, up_packed, up_scales, up_biases):
        for packed, scales, biases in ((gate_packed, gate_scales, gate_biases), (up_packed, up_scales, up_biases)):
            _validate(x, packed, scales, biases, *self.geometry)
        if x.ndim != 2 or ids.dtype != torch.int32 or ids.shape != (x.shape[0],):
            raise ValueError('Expected FP16 x[R,K] and sorted I32 ids[R]')
        experts, outputs, _ = self.geometry
        plan = make_plan(ids, experts, self.block)
        kernel = get_bm16_kernel(*self.geometry) if self.block == 16 else get_kernel(*self.geometry)
        output = kernel(x, plan, gate_packed, gate_scales, gate_biases, up_packed, up_scales, up_biases,
            threads_per_grid=(((outputs + 31) // 32) * 32, plan.shape[0] - 1, 1),
            threads_per_thread_group=(32, 1, 1), result_shapes=[[x.shape[0], outputs]])
        return output, plan


def tiny_inputs():
    from coreai_q4_metal import make_smoke
    ids = torch.repeat_interleave(torch.arange(5, dtype=torch.int32), torch.tensor([1, 16, 17, 32, 33]))
    gate, x, _ = make_smoke(len(ids), 128, 67, 5, seed=29117)
    up, _, _ = make_smoke(len(ids), 128, 67, 5, seed=29118)
    return (x[:, 0].contiguous(), ids, gate.packed.flatten(), gate.scales.flatten(), gate.biases.flatten(),
            up.packed.flatten(), up.scales.flatten(), up.biases.flatten())


def export(output, fixture=None):
    import coreai_torch
    check_fragment_mapping()
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
            outputs = [Projection(geometry, block)(*examples)[0] for block in (16, 32)]
            torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
            for block, value in zip((16, 32), outputs):
                value.numpy().tofile(output / f'bm{block}-expected-output.bin')
    kernels = [get_plan_kernel(geometry[0], block) for block in (16, 32)]
    kernels += [get_bm16_kernel(*geometry), get_kernel(*geometry)]
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    converter.register_custom_kernels(kernels)
    for block in (16, 32):
        converter.add_pytorch_module(Projection(geometry, block), entrypoint_name=f'bm{block}',
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
    plans = {}
    for block in (16, 32):
        plan = plan_reference(ids, geometry[0], block)
        plan.numpy().tofile(output / f'bm{block}-expected-plan.bin')
        plans[f'bm{block}'] = {'shape': list(plan.shape), 'activeTiles': int(plan[0, 0])}
        spec = {'asset': asset.name, 'function': f'bm{block}', 'inputs': inputs,
                'output': f'bm{block}-output', 'repeats': 15, 'mapped': False, 'oneBufferPerFile': False}
        (output / f'bm{block}-spec.json').write_text(json.dumps(spec, indent=2) + '\n')
    sources = ('coreai_q4_nax_gateup', 'coreai_q4_nax_gateup_parity', 'coreai_q4_nax_gateup_diagnostic',
               'coreai_q4_nax', 'coreai_q4_nax_gateup_m32_probe')
    report = {'status': 'cpu-authored-device-unvalidated', 'productionEnabled': False,
        'geometry': dict(zip(('E', 'N', 'K'), geometry)), 'plans': plans,
        'assetBytes': sum(path.stat().st_size for path in asset.rglob('*') if path.is_file()),
        'sourceSHA256': {name: hashlib.sha256(Path(__file__).with_name(name + '.py').read_bytes()).hexdigest() for name in sources},
        'fixtureSource': str(fixture) if fixture else 'tiny expert counts1/16/17/32/33; N67; K128',
        'policy': 'native-parity-v2: half(half(gate_half*up_accum_float)*sigmoid_half)',
        'candidate': 'One SIMD; twin M16 A/cachedA and gate/up C pairs; shared single BK72 weight tile and B fragment.',
        'CPUOutputExact': fixture is None,
        'limitations': ['GPU compilation/numerics/performance not yet validated.',
            'Register pressure and padded row arithmetic increase; speedup is not assumed.',
            'Compare each plan with its own expected bytes because BM16/BM32 plans differ.']}
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
    print(json.dumps({key: report[key] for key in ('status', 'geometry', 'plans', 'assetBytes')}, indent=2))
