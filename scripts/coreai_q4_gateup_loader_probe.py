#!/usr/bin/env python3
"""Paired external-weight fused gate/up assets for contiguous-affine loading."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from coreai_q4_flat import get_flat_gateup_kernel, _grouped_source
from coreai_q4_gateup import GATEUP_SOURCE
from coreai_q4_grouped import get_plan_kernel, make_plan
from coreai_q4_loader_probe import verify_loader_mapping

NAMES = ('x', 'ids', 'gate_packed', 'gate_scales', 'gate_biases',
         'up_packed', 'up_scales', 'up_biases')


class GateUp(torch.nn.Module):
    def __init__(self, geometry, candidate):
        super().__init__()
        self.geometry, self.candidate = geometry, candidate

    def forward(self, x, ids, gate_packed, gate_scales, gate_biases,
                up_packed, up_scales, up_biases):
        experts, outputs, _ = self.geometry
        plan = make_plan(ids, experts, 16)
        output = get_flat_gateup_kernel(*self.geometry, 16, 32, 64, self.candidate)(
            x, plan, gate_packed, gate_scales, gate_biases, up_packed, up_scales, up_biases,
            threads_per_grid=(((outputs+31)//32)*128, plan.shape[0]-1, 1),
            threads_per_thread_group=(128, 1, 1), result_shapes=[[x.shape[0], outputs]])
        return output, plan


def export_pair(output, examples, geometry):
    import coreai_torch
    output.mkdir(parents=True, exist_ok=False)
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    converter.register_custom_kernels([get_plan_kernel(geometry[0], 16),
        get_flat_gateup_kernel(*geometry, 16, 32, 64, False),
        get_flat_gateup_kernel(*geometry, 16, 32, 64, True)])
    for candidate, name in ((False, 'baseline'), (True, 'candidate')):
        converter.add_pytorch_module(GateUp(geometry, candidate).eval(), entrypoint_name=name,
            input_names=NAMES, output_names=('output', 'plan'),
            export_fn=lambda module: torch.export.export(module, args=examples).run_decompositions(
                coreai_torch.get_decomp_table()))
    program = converter.to_coreai()
    program.optimize()
    program.save_asset(output/'gateup.aimodel')
    (output/'candidate.metalbody').write_text(_grouped_source(
        GATEUP_SOURCE, *geometry, 16, 32, 64, ('gate_', 'up_'), True))


def tiny(output):
    from coreai_q4_metal import make_smoke
    from export_coreai_q4_moe import tensor_json
    gate, x, ids = make_smoke(79, 256, 67, 5)
    up, _, _ = make_smoke(79, 256, 67, 5)
    up.packed.bitwise_xor_(0x1234)
    up.scales.mul_(0.5)
    ids, order = torch.sort(ids)
    examples = (x[:, 0][order], ids, gate.packed.flatten(), gate.scales.flatten(), gate.biases.flatten(),
                up.packed.flatten(), up.scales.flatten(), up.biases.flatten())
    with torch.inference_mode():
        baseline = GateUp((5, 67, 256), False)(*examples)
        candidate = GateUp((5, 67, 256), True)(*examples)
    assert all(torch.equal(a, b) for a, b in zip(baseline, candidate, strict=True))
    export_pair(output, examples, (5, 67, 256))
    (output/'actual.json').write_text(json.dumps({
        'inputs': dict(zip(NAMES, map(tensor_json, examples), strict=True)),
        'expectedOutputs': dict(zip(('output', 'plan'), map(tensor_json, baseline), strict=True))})+'\n')
    (output/'cpu-check.json').write_text(json.dumps({'cpuCallbackExact': True,
        'mapping': verify_loader_mapping(), 'deviceValidated': False}, indent=2)+'\n')


def real_shape(output, manifest_path, rows):
    manifest = json.loads(manifest_path.read_text())
    weights = next(layer for layer in manifest['layers'] if layer['index'] == 0)['weights']
    records = {}
    for projection in ('gate', 'up'):
        prefix = f'moe.decode.{projection}_proj.'
        for row in weights['buffers']:
            if row['bufferName'].startswith(prefix):
                records[f'{projection}_{row["bufferName"].rsplit(".", 1)[-1]}'] = row
    assert set(records) == set(NAMES[2:])
    geometry = (512, 640, 2560)
    examples = (torch.empty(rows, 2560, dtype=torch.float16), torch.empty(rows, dtype=torch.int32),
        *(torch.empty(records[name]['byteLength']//2,
            dtype=torch.int16 if name.endswith('packed') else torch.float16) for name in NAMES[2:]))
    # Fake callbacks only: learned tensors are never populated or dequantized
    # here. Both GPU functions reference the same original file slices.
    export_pair(output, examples, geometry)
    generator = np.random.default_rng(927641)
    with (output/'x.bin').open('wb') as handle:
        for start in range(0, rows, 1024):
            data = (generator.standard_normal((min(1024, rows-start), 2560), dtype=np.float32)*.25).astype(np.float16)
            handle.write(data.tobytes())
    (np.arange(rows, dtype=np.int64)*512//rows).astype(np.int32).tofile(output/'ids.bin')
    inputs = {'x': {'file': 'x.bin', 'offset': 0, 'bytes': rows*2560*2, 'shape': [rows, 2560], 'dtype': 'float16'},
              'ids': {'file': 'ids.bin', 'offset': 0, 'bytes': rows*4, 'shape': [rows], 'dtype': 'int32'}}
    for name, record in records.items():
        inputs[name] = {'file': str((manifest_path.parent/weights['path']).resolve()),
            'offset': record['byteOffset'], 'bytes': record['byteLength'],
            'shape': [record['byteLength']//2], 'dtype': record['dtype']}
    for name in ('baseline', 'candidate'):
        (output/(name+'-spec.json')).write_text(json.dumps({'asset': 'gateup.aimodel',
            'function': name, 'inputs': inputs, 'output': name+'-output', 'repeats': 12,
            'mapped': False, 'oneBufferPerFile': False}, indent=2)+'\n')
    (output/'manifest.json').write_text(json.dumps({'status': 'CPU-authored-device-unvalidated',
        'geometry': {'E': 512, 'N': 640, 'K': 2560}, 'rows': rows, 'tile': [16, 32, 64],
        'fixture': 'Seed927641 random FP16 activations, uniformly sorted expert assignments, real layer0 gate/up weights',
        'ownedInputPolicy': 'Separate MTLBuffer per input; identical bytes/files for both functions',
        'roundingAndMMA': 'Unchanged existing fused gate/up FP16 boundaries and MPP reduction'}, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--rows', type=int, default=20480)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    if args.manifest:
        if not 0 < args.rows <= 81920:
            raise ValueError('Expected positive assignment rows <=81920')
        real_shape(args.output, args.manifest, args.rows)
    else:
        tiny(args.output)
