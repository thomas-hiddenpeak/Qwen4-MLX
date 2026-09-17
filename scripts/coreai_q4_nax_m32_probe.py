#!/usr/bin/env python3
"""Isolated BM32 down projection using two independent NAX register subtiles.

One SIMD reuses each BK64/stride72 dequantized weight tile and B fragment across
two M16xN32xK16 operations. The K order, affine FP16 dequantization and final
FP16 store for each output are unchanged from the validated BM16 NAX source.
Only the expert planner's M tile changes. This is not a production installer.
"""
from __future__ import annotations

import argparse
from functools import cache
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from coreai_q4_flat import _replace_once, _reshape, _validate
from coreai_q4_grouped import get_plan_kernel, grouped_reference, make_plan, plan_reference
from coreai_q4_nax import MLX_NOTICE, get_kernel as get_bm16_kernel, source_text as bm16_source


INPUTS = ('x', 'ids', 'packed', 'scales', 'biases')


def source_text(outputs, inputs):
    """Strict transformations fail if the independently owned BM16 source drifts."""
    source = bm16_source(outputs, inputs, 1, True)
    source = _replace_once(source,
        'auto a=operation.get_left_input_cooperative_tensor<half,half,float>();',
        'auto a=operation.get_left_input_cooperative_tensor<half,half,float>();\n'
        'auto a_hi=operation.get_left_input_cooperative_tensor<half,half,float>();')
    source = _replace_once(source,
        '#pragma clang loop unroll(full)\nfor(int i=0;i<16;++i)c[i]=0.0f;',
        'auto c_hi=operation.get_destination_cooperative_tensor<\n'
        '    metal::remove_addrspace_t<decltype(a)>,metal::remove_addrspace_t<decltype(b)>,float>();\n'
        '#pragma clang loop unroll(full)\nfor(int i=0;i<16;++i){c[i]=0.0f;c_hi[i]=0.0f;}')
    source = _replace_once(source,
        'a[i]=row<count ? xp[(start+row)*x_stride+k] : half(0);',
        'a[i]=row<count ? xp[(start+row)*x_stride+k] : half(0);\n'
        '      a_hi[i]=row+16<count ? xp[(start+row+16)*x_stride+k] : half(0);')
    source = _replace_once(source, 'operation.run(a,b,c);',
        'operation.run(a,b,c);\n    operation.run(a_hi,b,c_hi);')
    source = _replace_once(source,
        'if(row<count && n+16<N)yp[(start+row)*y_stride+n+16]=half(c[8+i]);',
        'if(row<count && n+16<N)yp[(start+row)*y_stride+n+16]=half(c[8+i]);\n'
        '  if(row+16<count && n<N)yp[(start+row+16)*y_stride+n]=half(c_hi[i]);\n'
        '  if(row+16<count && n+16<N)yp[(start+row+16)*y_stride+n+16]=half(c_hi[8+i]);')
    return source


@cache
def get_kernel(experts, outputs, inputs):
    from coreai.authoring import MetalParameter
    from coreai_torch import TorchMetalKernel
    if not 1 <= experts <= 512:
        raise ValueError('Expected 1...512 experts')
    def reference(x: torch.Tensor, plan: torch.Tensor, packed: torch.Tensor,
                  scales: torch.Tensor, biases: torch.Tensor) -> torch.Tensor:
        return grouped_reference(x, plan, *_reshape(packed, scales, biases, experts, outputs, inputs))
    return TorchMetalKernel(f'qwen_experimental_q4_nax_down_m32_e{experts}_n{outputs}_k{inputs}_v1',
        input_names=['x', 'plan', 'packed', 'scales', 'biases'], result_names=['output'],
        src=source_text(outputs, inputs), helper_src=MLX_NOTICE, torch_defn=reference,
        metal_params=[MetalParameter('group', 'uint3', 'threadgroup_position_in_grid'),
                      MetalParameter('thread_id', 'uint', 'thread_index_in_threadgroup')])


class Projection(torch.nn.Module):
    def __init__(self, geometry, block):
        super().__init__()
        if block not in (16, 32):
            raise ValueError('Expected BM16 or BM32')
        self.geometry, self.block = geometry, block

    def forward(self, x, ids, packed, scales, biases):
        _validate(x, packed, scales, biases, *self.geometry)
        if x.ndim != 2 or ids.dtype != torch.int32 or ids.shape != (x.shape[0],):
            raise ValueError('Expected FP16 x[R,K] and sorted I32 ids[R]')
        experts, outputs, _ = self.geometry
        plan = make_plan(ids, experts, self.block)
        kernel = get_bm16_kernel(*self.geometry, 1, True) if self.block == 16 else get_kernel(*self.geometry)
        output = kernel(x, plan, packed, scales, biases,
            threads_per_grid=(((outputs + 31) // 32) * 32, plan.shape[0] - 1, 1),
            threads_per_thread_group=(32, 1, 1), result_shapes=[[x.shape[0], outputs]])
        return output, plan


def check_fragment_mapping():
    coordinates = []
    for lane in range(32):
        qid = lane >> 2
        row, col = (qid & 4) | ((lane >> 1) & 3), ((qid & 2) | (lane & 1)) * 4
        for mblock in (0, 16):
            for nblock in (0, 16):
                coordinates += [(mblock + row + (i // 4) * 8, nblock + col + i % 4) for i in range(8)]
    expected = {(m, n) for m in range(32) for n in range(32)}
    if len(coordinates) != len(expected) or set(coordinates) != expected:
        raise AssertionError('BM32 register output fragment coordinates must cover every output once')


def export_pair(output, examples, geometry):
    import coreai_torch
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    kernels = [get_plan_kernel(geometry[0], block) for block in (16, 32)]
    kernels += [get_bm16_kernel(*geometry, 1, True), get_kernel(*geometry)]
    converter.register_custom_kernels(kernels)
    for block in (16, 32):
        converter.add_pytorch_module(Projection(geometry, block), entrypoint_name=f'bm{block}',
            input_names=INPUTS, output_names=('output', 'plan'),
            export_fn=lambda current: torch.export.export(current, args=examples).run_decompositions(coreai_torch.get_decomp_table()))
    program = converter.to_coreai()
    program.optimize()
    program._mlir_module.operation.verify()
    asset = output / 'down.aimodel'
    program.save_asset(asset)
    for kernel in kernels:
        for name, source in kernel.kernel_cache.values():
            (output / (name + '.metal')).write_text(source)
    return sum(path.stat().st_size for path in asset.rglob('*') if path.is_file())


def export(output, fixture=None):
    from coreai_q4_metal import make_smoke
    check_fragment_mapping()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cpu_outputs = {}
    if fixture:
        fixture = fixture.resolve()
        spec = json.loads((fixture / 'baseline-spec.json').read_text())
        inputs = spec['inputs']
        geometry = (512, 2560, 640)
        if inputs['x']['shape'] != [20480, 640] or inputs['packed']['shape'] != [512 * 2560 * 640 // 4]:
            raise ValueError('Expected the existing real down S2048 fixture geometry')
        for entry in inputs.values():
            entry['file'] = str((fixture / entry['file']).resolve())
        examples = tuple(torch.empty(inputs[name]['shape'], dtype={
            'int32': torch.int32, 'int16': torch.int16, 'float16': torch.float16}[inputs[name]['dtype']]) for name in INPUTS)
        meta = inputs['ids']
        ids = torch.from_numpy(np.fromfile(meta['file'], dtype=np.int32,
            count=meta['bytes'] // 4, offset=meta['offset']))
        input_scope = 'Exact real layer0 down projection bytes and ordered activation fixture from q4-loader-down-S2048; learned bytes are not copied.'
    else:
        geometry = (5, 67, 128)
        counts = torch.tensor([1, 16, 17, 32, 33])
        ids = torch.repeat_interleave(torch.arange(5, dtype=torch.int32), counts)
        projection, x, _ = make_smoke(len(ids), 128, 67, 5, seed=29117)
        examples = (x[:, 0].contiguous(), ids, projection.packed.flatten(),
                    projection.scales.flatten(), projection.biases.flatten())
        with torch.inference_mode():
            for block in (16, 32):
                cpu_outputs[block] = Projection(geometry, block)(*examples)
            torch.testing.assert_close(cpu_outputs[16][0], cpu_outputs[32][0], rtol=0, atol=0)
        inputs = {}
        for name, value in zip(INPUTS, examples):
            value.numpy().tofile(output / (name + '.bin'))
            inputs[name] = {'file': name + '.bin', 'offset': 0, 'bytes': value.numel() * value.element_size(),
                            'shape': list(value.shape), 'dtype': str(value.dtype).removeprefix('torch.')}
        input_scope = 'Deterministic synthetic expert counts1/16/17/32/33, N67 and K128 cover both M subtiles, partial expert tiles and multiple K blocks.'
    asset_bytes = export_pair(output, examples, geometry)
    plans = {}
    for block in (16, 32):
        plan = plan_reference(ids, geometry[0], block)
        plan.numpy().tofile(output / f'bm{block}-expected-plan.bin')
        plans[f'bm{block}'] = {'shape': list(plan.shape), 'activeTiles': int(plan[0, 0]),
                             'coveredRows': int(plan[1:1 + int(plan[0, 0]), 2].sum())}
        spec = {'asset': 'down.aimodel', 'function': f'bm{block}', 'inputs': inputs,
                'output': f'bm{block}-output', 'repeats': 15, 'mapped': False, 'oneBufferPerFile': False}
        (output / f'bm{block}-spec.json').write_text(json.dumps(spec, indent=2) + '\n')
        if cpu_outputs:
            cpu_outputs[block][0].numpy().tofile(output / f'bm{block}-expected-output.bin')
    report = {'status': 'cpu-authored-device-unvalidated', 'productionEnabled': False,
              'assetBytes': asset_bytes, 'geometry': dict(zip(('E', 'N', 'K'), geometry)),
              'baseline': 'Validated pointer-ABI NAX BM16, one SIMD and one C register fragment.',
              'candidate': 'BM32, one SIMD and two independent C fragments sharing the same B register fragment.',
              'tileN': 32, 'tileK': 64, 'weightSharedStride': 72, 'inputScope': input_scope,
              'plans': plans, 'planComparison': 'Plan shapes and tile counts intentionally differ; compare each with its own expected raw plan.',
              'cpuOutputExact': bool(cpu_outputs) if cpu_outputs else None,
              'baselineSourceSHA256': hashlib.sha256(Path(__file__).with_name('coreai_q4_nax.py').read_bytes()).hexdigest(),
              'candidateSourceSHA256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'limitations': ['No candidate GPU compilation/numerical/performance acceptance yet.',
                  'Same source accumulation order does not guarantee compiler scheduling or register occupancy.',
                  'Larger M tiles can perform more unused arithmetic in lightly populated experts.']}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    result = export(args.output, args.fixture)
    print(json.dumps({key: result[key] for key in ('status', 'geometry', 'assetBytes', 'plans')}, indent=2))
