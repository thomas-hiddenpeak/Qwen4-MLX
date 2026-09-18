#!/usr/bin/env python3
"""Experimental single-pass HCWrite, with three explicit rounding policies.

CPU authoring only; no production exporter or existing kernel is modified.
The original graph may fuse its FP16 multiply/add. Consequently the saved native
GPU result, not eager CPU half arithmetic, is the target for device parity.
"""
from __future__ import annotations

import argparse
from functools import cache
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch._subclasses.fake_tensor import FakeTensor

from export_coreai_dense import DenseConfig, HCWrite
from export_coreai_hc_probe import read_input, tensor_input


POLICIES = ('rounded', 'unrounded', 'fma')
SOURCE = r"""
const uint width=uint(stream.get_extent(0)),tokens=uint(stream.get_extent(1));
if(index>=width*tokens)return;
const uint hidden=uint(output.get_extent(0));
const uint token=index/width,channel=index%width;
const uint branch=channel/hidden,column=channel%hidden;
const float old=float(stream[channel,token,0]);
const float value=float(output[column,token,0]);
const float gate=float(injection[0,branch,token,0]);
UPDATE
"""
UPDATES = {
    'rounded': 'const half update=half(value*gate);\nstream_out[channel,token,0]=half(old+float(update));',
    'unrounded': 'const float update=value*gate;\nstream_out[channel,token,0]=half(old+update);',
    'fma': 'stream_out[channel,token,0]=half(fma(value,gate,old));',
}


def reference(stream, output, injection, *, policy):
    if policy not in POLICIES:
        raise ValueError('Unknown HCWrite numerical policy')
    if isinstance(stream, FakeTensor) or stream.device.type == 'meta':
        return torch.empty_like(stream)
    count, branches, hidden = stream.shape[1], injection.shape[2], output.shape[-1]
    old = stream.reshape(1, count, branches, hidden)
    value = output.reshape(1, count, 1, hidden)
    if policy == 'rounded':
        update = (value.float() * injection.float()).half()
        result = old.float() + update.float()
    elif policy == 'unrounded':
        result = old.float() + value.float() * injection.float()
    else:
        # Finite FP16 operands have an exact FP32 product. Double intermediate
        # models one FP32 FMA rounding before the final FP16 store, for this
        # restricted input domain; it is not a general FP32 FMA implementation.
        result = (value.double() * injection.double() + old.double()).float()
    return result.half().reshape_as(stream)


@cache
def get_kernel(policy):
    from coreai.authoring import MetalParameter
    from coreai_torch import TorchMetalKernel
    if policy not in POLICIES:
        raise ValueError('Unknown HCWrite numerical policy')
    def cpu(stream: torch.Tensor, output: torch.Tensor, injection: torch.Tensor) -> torch.Tensor:
        return reference(stream, output, injection, policy=policy)
    return TorchMetalKernel(f'qwen_experimental_hc_write_{policy}_v1',
        input_names=['stream', 'output', 'injection'], result_names=['stream_out'],
        src=SOURCE.replace('UPDATE', UPDATES[policy]), torch_defn=cpu,
        metal_params=[MetalParameter('index', 'uint', 'thread_position_in_grid')])


class DirectHCWrite(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        if policy not in POLICIES:
            raise ValueError('Unknown HCWrite numerical policy')
        self.policy = policy

    def forward(self, stream, output, injection):
        if (stream.ndim != 3 or stream.shape[0] != 1 or min(stream.shape) < 1 or
                output.ndim != 3 or output.shape[:2] != stream.shape[:2] or min(output.shape) < 1 or
                injection.ndim != 4 or injection.shape[:2] != stream.shape[:2] or
                injection.shape[-1] != 1 or injection.shape[2] < 1 or
                stream.shape[2] != output.shape[2] * injection.shape[2] or
                any(value.dtype != torch.float16 for value in (stream, output, injection))):
            raise ValueError('Expected FP16 stream[1,S,B*H], output[1,S,H], injection[1,S,B,1]')
        return get_kernel(self.policy)(stream, output, injection,
            threads_per_grid=(stream.numel(), 1, 1), threads_per_thread_group=(256, 1, 1),
            result_shapes=[list(stream.shape)])


def export_asset(path, values, *, include_baseline=False):
    import coreai_torch
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    converter.register_custom_kernels([get_kernel(policy) for policy in POLICIES])
    modules = [(policy, DirectHCWrite(policy)) for policy in POLICIES]
    if include_baseline:
        c = DenseConfig(values[1].shape[-1], values[2].shape[2], 1, 1e-6, 1, 1, 2, 1)
        modules.insert(0, ('baseline', HCWrite(c)))
    for name, module in modules:
        converter.add_pytorch_module(module, entrypoint_name=name,
            input_names=('stream', 'output', 'injection'), output_names=('stream_out',),
            export_fn=lambda current: torch.export.export(current, args=values).run_decompositions(coreai_torch.get_decomp_table()))
    program = converter.to_coreai()
    program.optimize()
    program._mlir_module.operation.verify()
    program.save_asset(path)
    for name, _ in modules:
        (path.parent / (path.stem + '-' + name + '-after.txt')).write_text(str(program.get_graph(name)))
    return sum(item.stat().st_size for item in path.rglob('*') if item.is_file())


def metrics(actual, expected):
    delta = actual.float() - expected.float()
    return {'bitwiseExact': np.array_equal(actual.numpy().view(np.uint16), expected.numpy().view(np.uint16)),
            'unequalElements': int(torch.count_nonzero(actual != expected)),
            'maximumAbsoluteError': float(delta.abs().max()),
            'relativeL2Error': float(delta.norm() / expected.float().norm().clamp_min(1e-30))}


def export(output, baseline_directory):
    output, baseline_directory = output.resolve(), baseline_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'cpu-authored-device-unvalidated', 'productionEnabled': False,
              'policies': list(POLICIES), 'cases': [], 'baselineDirectory': str(baseline_directory),
              'inputScope': 'Same deterministic synthetic stream and real layer0 HCRead outputs as the existing HC component probe.',
              'numericalTarget': 'Bitwise parity with native CoreAI HCWrite device output on identical input bytes.',
              'limitations': ['Unrounded expression may contract; explicit FMA is a separate entrypoint.',
                  'CPU oracle comparison does not prove candidate Metal compilation, parity or performance.',
                  'Standalone component timing does not establish fused whole-layer or full-model speedup.']}
    for component in ('attention', 'moe'):
        original = json.loads((baseline_directory / (component + '-write-spec.json')).read_text())
        inputs, values = {}, []
        for name in ('stream', 'output', 'injection'):
            entry = dict(original['inputs'][name])
            entry['file'] = str((baseline_directory / entry['file']).resolve())
            value, _ = read_input(entry, baseline_directory)
            if not torch.isfinite(value).all():
                raise ValueError('Nonfinite real-geometry HC fixture input')
            values.append(value)
            inputs[name] = entry
        if component == 'attention':
            report['assetBytes'] = export_asset(output / 'hc-write.aimodel', tuple(values))
        shape = list(values[0].shape)
        device_file = baseline_directory / (component + '-write-output') / 'stream_out.bin'
        expected_gpu = None
        if device_file.exists():
            raw = device_file.read_bytes()
            expected_gpu = torch.from_numpy(np.frombuffer(raw, dtype=np.float16).reshape(shape).copy())
        cases = [('baseline', dict(original, asset=str((baseline_directory / original['asset']).resolve())))]
        for policy in POLICIES:
            cases.append((policy, {'asset': 'hc-write.aimodel', 'function': policy, 'inputs': inputs}))
        with torch.inference_mode():
            comparisons = {policy: metrics(reference(*values, policy=policy), expected_gpu)
                           for policy in POLICIES} if expected_gpu is not None else None
        for policy, spec in cases:
            label = component + '-' + policy
            spec.update(inputs=inputs, output=label + '-output', repeats=15, mapped=False)
            filename = label + '-spec.json'
            (output / filename).write_text(json.dumps(spec, indent=2) + '\n')
            report['cases'].append({'name': label, 'spec': filename, 'shape': shape,
                'nativeDeviceReference': str(device_file) if expected_gpu is not None else None,
                'nativeDeviceReferenceSHA256': hashlib.sha256(device_file.read_bytes()).hexdigest() if expected_gpu is not None else None,
                'cpuPolicyVersusNativeDevice': comparisons.get(policy) if comparisons else None})

    generator = torch.Generator().manual_seed(1941)
    tiny_values = ((torch.randn(1, 3, 20, generator=generator) * .2).half(),
                   (torch.randn(1, 3, 5, generator=generator) * .4).half(),
                   (torch.rand(1, 3, 4, 1, generator=generator) * 2).half())
    tiny_values[0][0, 0, :4] = torch.tensor([2**-24, -2**-24, 2**-14, -2**-14]).half()
    tiny_values[1][0, 0, 0] = 2**-24
    tiny_values[2][0, 0, 0, 0] = .5
    report['tinyAssetBytes'] = export_asset(output / 'tiny.aimodel', tiny_values, include_baseline=True)
    tiny_inputs = {name: tensor_input(output / ('tiny-' + name + '.bin'), value)
                   for name, value in zip(('stream', 'output', 'injection'), tiny_values)}
    for policy in ('baseline', *POLICIES):
        label = 'tiny-' + policy
        spec = {'asset': 'tiny.aimodel', 'function': policy, 'inputs': tiny_inputs,
                'output': label + '-output', 'repeats': 15, 'mapped': False}
        (output / (label + '-spec.json')).write_text(json.dumps(spec, indent=2) + '\n')
    report['tiny'] = {'shape': [1, 3, 20], 'hidden': 5, 'streams': 4,
                      'scope': 'Non-vector-aligned hidden width, partial threadgroup and FP16 subnormal inputs.'}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-directory', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    result = export(args.output, args.baseline_directory)
    print(json.dumps({key: result[key] for key in ('status', 'assetBytes', 'tinyAssetBytes')}, indent=2))
