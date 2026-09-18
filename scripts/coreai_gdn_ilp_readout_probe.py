#!/usr/bin/env python3
"""Independent ILP4 readout repair candidate, never installed implicitly.

Retain the old ILP4 state/key reuse, but perform each row's output update and
readout in the baseline lexical order: scalar accumulator, inline q load, and
no explicit part-loop unroll. CPU authoring does not establish GPU parity.
"""
from __future__ import annotations

import argparse
from functools import cache
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from coreai_gdn_chunk_metal import (
    INPUT_NAMES, OUTPUT_NAMES, FusedGDNRecurrence, get_gdn_recurrence_kernel,
    recurrence_reference,
)
from coreai_gdn_ilp_probe import SOURCE as ORIGINAL_ILP_SOURCE, reference
from export_coreai_q4_moe import tensor_json


def source_text() -> str:
    """Change only q preload and the output update/readout loop."""
    changes = (
        ('float key[4],query[4],memory_partial[ROWS],delta[ROWS],output_partial[ROWS];',
         'float key[4],memory_partial[ROWS],delta[ROWS];'),
        ('        output_partial[row]=0.0f;\n', ''),
        ('        query[part]=float(q[lane+32u*part,head,token,0]);\n', ''),
        ('''    #pragma clang loop unroll(full)
    for(uint part=0u;part<4u;++part) {
        #pragma clang loop unroll(full)
        for(uint row=0u;row<ROWS;++row) {
            cell[row][part]+=delta[row]*key[part];
            output_partial[row]+=cell[row][part]*query[part];
        }
    }
    #pragma clang loop unroll(full)
    for(uint row=0u;row<ROWS;++row) {
        const float value=simd_sum(output_partial[row]);
        if(lane==0u && first_row+row<value_dim)y[first_row+row,head,token,0]=half(value);
    }''', '''    #pragma clang loop unroll(full)
    for(uint row=0u;row<ROWS;++row) {
        float output_partial=0.0f;
        for(uint part=0u;part<4u;++part) {
            cell[row][part]+=delta[row]*key[part];
            output_partial+=cell[row][part]*float(q[lane+32u*part,head,token,0]);
        }
        const float value=simd_sum(output_partial);
        if(lane==0u && first_row+row<value_dim)y[first_row+row,head,token,0]=half(value);
    }'''),
    )
    source = ORIGINAL_ILP_SOURCE
    for before, after in changes:
        if source.count(before) != 1:
            raise ValueError('Original ILP source changed; inspect before authoring this candidate')
        source = source.replace(before, after)
    return source.replace('ROWS', '4u')


@cache
def get_kernel():
    from coreai.authoring import MetalParameter
    from coreai_torch import TorchMetalKernel
    return TorchMetalKernel(
        'qwen_experimental_gdn_recurrence_k128_ilp4_readout_v2',
        input_names=list(INPUT_NAMES), result_names=list(OUTPUT_NAMES),
        src=source_text(), torch_defn=reference,
        metal_params=[MetalParameter('group', 'uint3', 'threadgroup_position_in_grid'),
                      MetalParameter('lane', 'uint', 'thread_index_in_simdgroup'),
                      MetalParameter('simd', 'uint', 'simdgroup_index_in_threadgroup')],
    )


class ReadoutILPRecurrence(torch.nn.Module):
    def forward(self, q, k, v, decay, beta, state):
        if (q.ndim != 4 or q.shape[0] != 1 or q.shape[-1] != 128 or min(q.shape) < 1 or
                k.shape != q.shape or v.ndim != 4 or v.shape[:3] != q.shape[:3] or min(v.shape) < 1 or
                decay.shape != q.shape[:3] or beta.shape != decay.shape or
                state.shape != (1, q.shape[2], v.shape[-1], 128)):
            raise ValueError('GDN ILP readout input/state geometry differs')
        if (any(value.dtype not in (torch.float16, torch.float32) for value in (q, k, v)) or
                any(value.dtype != torch.float32 for value in (decay, beta, state))):
            raise ValueError('Expected FP16/FP32 qkv and FP32 gates/state')
        return get_kernel()(q, k, v, decay, beta, state,
                            threads_per_grid=(((v.shape[-1] + 15) // 16) * 128, q.shape[2], 1),
                            threads_per_thread_group=(128, 1, 1),
                            result_shapes=[list(v.shape), list(state.shape)])


def load_fixture(spec_path: Path):
    spec = json.loads(spec_path.read_text())
    values, descriptors = [], {}
    for name in INPUT_NAMES:
        entry = dict(spec['inputs'][name])
        path = (spec_path.parent / entry['file']).resolve()
        dtype = np.dtype(entry['dtype'])
        count = int(np.prod(entry['shape']))
        if count * dtype.itemsize != entry['bytes']:
            raise ValueError(f'{name}: inconsistent byte count')
        with path.open('rb') as handle:
            handle.seek(entry['offset'])
            data = handle.read(entry['bytes'])
        if len(data) != entry['bytes']:
            raise ValueError(f'{name}: short fixture')
        values.append(torch.from_numpy(np.frombuffer(data, dtype=dtype).copy().reshape(entry['shape'])))
        entry['file'] = str(path)
        descriptors[name] = entry
    return tuple(values), descriptors


def export_pair(output: Path, label: str, spec_path: Path):
    import coreai_torch
    values, inputs = load_fixture(spec_path)
    kernels = [get_gdn_recurrence_kernel(), get_kernel()]
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    converter.register_custom_kernels(kernels)
    for name, module in (('baseline', FusedGDNRecurrence()), ('candidate', ReadoutILPRecurrence())):
        converter.add_pytorch_module(module, entrypoint_name=name,
            input_names=INPUT_NAMES, output_names=OUTPUT_NAMES,
            export_fn=lambda m: torch.export.export(m, args=values).run_decompositions(coreai_torch.get_decomp_table()))
    program = converter.to_coreai()
    program.optimize()
    asset = output / (label + '.aimodel')
    program.save_asset(asset)
    for function in ('baseline', 'candidate'):
        spec = {'asset': asset.name, 'function': function, 'inputs': inputs,
                'output': label + '-' + function + '-output', 'repeats': 15, 'mapped': False}
        (output / (label + '-' + function + '-spec.json')).write_text(json.dumps(spec, indent=2) + '\n')
    for kernel in kernels:
        for name, body in kernel.kernel_cache.values():
            (output / (name + '.metal')).write_text(body)
    result = {'asset': asset.name, 'sourceSpec': str(spec_path.resolve()),
              'shape': {'q': list(values[0].shape), 'v': list(values[2].shape)},
              'inputsReusedWithoutChanges': True, 'fullCPURecurrenceExecuted': False}
    if label == 'tiny':
        with torch.inference_mode():
            expected = recurrence_reference(*values)
            actual = ReadoutILPRecurrence()(*values)
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        fixture = {'inputs': {name: tensor_json(value) for name, value in zip(INPUT_NAMES, values)},
                   'expectedOutputs': {name: tensor_json(value) for name, value in zip(OUTPUT_NAMES, expected)}}
        (output / 'tiny.json').write_text(json.dumps(fixture) + '\n')
        result['fullCPURecurrenceExecuted'] = True
        result['cpuCallbackMatchesReference'] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True,
                        help='Existing gdn-ilp2-4 directory; reuse its binary inputs verbatim')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'cpu-authored-device-unvalidated', 'deviceValidated': False,
              'productionEnabled': False, 'functions': ['baseline', 'candidate'],
              'candidateKernel': 'qwen_experimental_gdn_recurrence_k128_ilp4_readout_v2',
              'candidateSourceSHA256': hashlib.sha256(source_text().encode()).hexdigest(),
              'policy': 'ILP4 state/key reuse retained; each row uses a scalar output accumulator, inline q load, and baseline part-loop lexical order without explicit part unroll.',
              'acceptance': 'Require every GPU y and next_state element to match the paired baseline; CPU reference is not GPU evidence.',
              'pairs': {}}
    for label, filename in (('tiny', 'tiny-ilp4-spec.json'), ('real-s2048', 'ilp4-spec.json')):
        report['pairs'][label] = export_pair(args.output, label, args.fixtures / filename)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
