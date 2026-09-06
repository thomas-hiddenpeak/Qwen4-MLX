#!/usr/bin/env python3
"""Exact Q4 input-group LUT -> output-channel LUT grouped-convolution probe.

Reorders each source [N,G,64] matrix into [G*N,64,1,1], with G convolution
groups, and sums the G partial products. No source Q4 code is re-clustered.
The FP16 decoded matrices are preserved, but reduction precision may change.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import time
import traceback

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
import numpy as np
import torch
import torch.nn.functional as F

import export_moe as e


def reorder(weight):
    codes = weight['codes']
    n, c = codes.shape
    groups = c // 64
    if c % 64:
        raise ValueError('Source input width must be divisible by 64')
    ordered_codes = codes.reshape(n, groups, 64).transpose(1, 0, 2).copy().reshape(groups*n, 64, 1, 1)
    ordered_lut = weight['palette16'].transpose(1, 0, 2).copy().reshape(groups*n, 1, 1, 1, 16, 1)
    decoded = np.take_along_axis(ordered_lut.reshape(groups*n, 16),
                                ordered_codes.reshape(groups*n, 64), axis=1)
    restored = decoded.reshape(groups, n, 64).transpose(1, 0, 2).reshape(n, c)
    if not np.array_equal(restored, weight['dense16']):
        raise AssertionError('Reordered LUT failed bit-exact FP16 matrix preservation')
    return ordered_codes, ordered_lut, {'input_channels': c, 'output_channels': n,
        'convolution_groups': groups, 'partial_output_channels': groups*n,
        'dense_parameter_count_original': n*c, 'dense_parameter_count_reordered': ordered_codes.size,
        'uint4_index_bytes': ordered_codes.size//2, 'fp16_palette_bytes': ordered_lut.nbytes,
        'fp16_decoded_weight_exact': True,
        'decoded_weight_sha256': hashlib.sha256(restored.tobytes()).hexdigest()}


def projection(x, rearranged, capacity, name):
    codes, lut, info = rearranged
    packed = codes.astype(types.nptype_from_builtin(types.uint4))
    weight = mb.constexpr_lut_to_dense(indices=packed, lut=lut, name=name+'_output_channel_lut4')
    partial = mb.conv(x=x, weight=weight, groups=info['convolution_groups'], name=name+'_grouped_conv')
    split = mb.reshape(x=partial, shape=[1, info['convolution_groups'], info['output_channels'], capacity],
                       name=name+'_split_input_groups')
    summed = mb.reduce_sum(x=split, axes=[1], keep_dims=False, name=name+'_sum_input_groups')
    return mb.reshape(x=summed, shape=[1, info['output_channels'], 1, capacity], name=name+'_y')


def make_program(rearranged, capacity, full):
    input_channels = rearranged['gate_proj'][2]['input_channels']
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, input_channels, 1, capacity), dtype=types.fp16)],
                opset_version=ct.target.macOS15)
    def program(x):
        gate = projection(x, rearranged['gate_proj'], capacity, 'gate')
        if not full:
            return mb.identity(x=gate, name='y')
        up = projection(x, rearranged['up_proj'], capacity, 'up')
        denominator = mb.add(x=mb.exp(x=mb.mul(x=gate, y=np.float16(-1))), y=np.float16(1))
        middle = mb.mul(x=mb.real_div(x=gate, y=denominator), y=up)
        down = projection(middle, rearranged['down_proj'], capacity, 'down')
        return mb.identity(x=down, name='y')
    return program


def get_plan(model):
    plan = ct.models.compute_plan.MLComputePlan.load_from_path(model.get_compiled_model_path(),
            compute_units=ct.ComputeUnit.CPU_AND_NE)
    operations = []
    def visit(block):
        for op in block.operations:
            if op.operator_name.split('.')[-1] != 'const':
                usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
                operations.append({'operator': op.operator_name, 'outputs': [v.name for v in op.outputs],
                                   'preferred': type(usage.preferred_compute_device).__name__ if usage else None,
                                   'supported': [type(v).__name__ for v in usage.supported_compute_devices] if usage else []})
            for child in op.blocks:
                visit(child)
    for function in plan.model_structure.program.functions.values():
        visit(function.block)
    required = [op for op in operations if op['operator'].split('.')[-1] in ('conv', 'reduce_sum')]
    return {'operations': operations, 'preferred_counts': dict(Counter(o['preferred'] for o in operations)),
            'all_convolution_and_reduction_prefer_ane': bool(required) and all(
                op['preferred'] == 'MLNeuralEngineComputeDevice' for op in required)}


def predict_cases(model, weights, capacity, full, folder, runs):
    rows = []
    for phase in ('prefill', 'decode'):
        case = np.load(e.ROOT / f'ane-runner/fixtures/moe-real/converted/{phase}.npz')
        for first in range(0, case['x'].shape[1], capacity):
            last = min(first+capacity, case['x'].shape[1])
            n = last-first
            bsh = np.zeros((1, capacity, e.HIDDEN), np.float16)
            bsh[:, :n] = case['x'][:, first:last]
            x = np.ascontiguousarray(bsh.transpose(0, 2, 1)[:, :, None])
            refs = {}
            for key in ('dense32', 'dense16'):
                with torch.inference_mode():
                    tx = torch.from_numpy(x.astype(np.float32))
                    matrices = {p: torch.from_numpy(w[key].astype(np.float32))[:, :, None, None] for p,w in weights.items()}
                    gate = F.conv2d(tx, matrices['gate_proj'])
                    expected = F.conv2d(F.silu(gate)*F.conv2d(tx, matrices['up_proj']), matrices['down_proj']) if full else gate
                    refs[key] = expected.numpy()
            result = np.asarray(model.predict({'x': x})['y'])
            row = {'phase': phase, 'positions': case['positions'][first:last].tolist(), 'valid_tokens': n,
                   'finite': bool(np.isfinite(result).all()),
                   'vs_source_affine_fp32': e.errors(result[..., :n], refs['dense32'][..., :n]),
                   'vs_fp16_weights_fp32_compute': e.errors(result[..., :n], refs['dense16'][..., :n])}
            if runs:
                for _ in range(3):
                    model.predict({'x': x})
                wall, core = [], []
                for _ in range(runs):
                    t = time.perf_counter_ns()
                    model.predict({'x': x})
                    wall.append((time.perf_counter_ns()-t)/1e6)
                    core.append(model.last_predict_duration_in_nano_seconds/1e6)
                row.update({'warmups': 3, 'runs': runs, 'ready_host_prediction_ms': wall,
                            'coreml_prediction_from_features_ms': core,
                            'median_host_ms': float(np.median(wall)), 'median_coreml_ms': float(np.median(core))})
            np.savez(folder / f'{phase}_{first}_{last}.npz', x=x, y=result, **refs)
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expert', type=int, default=88)
    parser.add_argument('--capacities', type=int, nargs='+', default=[1,8])
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--output', type=Path, default=e.ROOT/'ane-runner/results/moe-grouped-quant-v2')
    args = parser.parse_args()
    if args.expert not in range(512) or min(args.capacities) < 1 or args.runs < 0:
        parser.error('Invalid expert, capacity or run count')
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    source = e.Source(0)
    weights = source.expert(args.expert)
    rearranged = {p: reorder(w) for p,w in weights.items()}
    mode = 'full' if args.full else 'gate'
    result = {'schema':'exact-q4-grouped-conv-v1', 'expert':args.expert, 'mode':mode,
              'source_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'source_records':source.records, 'transformation':{p:v[2] for p,v in rearranged.items()},
              'limitations':['Decoded FP16 weight values preserved exactly; accumulation boundaries change.',
                             'ComputePlan is not runtime hardware proof.',
                             'No whole-model, shared expert, router, or scheduling test.'], 'capacities':[]}
    for capacity in args.capacities:
        folder = args.output / f'expert{args.expert}_{mode}_s{capacity}'
        folder.mkdir(parents=True, exist_ok=True)
        try:
            t = time.perf_counter()
            model = ct.convert(make_program(rearranged,capacity,args.full), convert_to='mlprogram',
                               minimum_deployment_target=ct.target.macOS15,
                               compute_units=ct.ComputeUnit.CPU_AND_NE, compute_precision=ct.precision.FLOAT16)
            model.save(str(folder/'model.mlpackage'))
            row = {'capacity':capacity,'status':'compiled','compile_seconds':time.perf_counter()-t,
                   'package_bytes':e.package_bytes(folder/'model.mlpackage'),'plan':get_plan(model)}
            if args.predict and row['plan']['all_convolution_and_reduction_prefer_ane']:
                row['predictions'] = predict_cases(model,weights,capacity,args.full,folder,args.runs)
            elif args.predict:
                row['prediction_skipped'] = 'Grouped convolution or group reduction does not prefer ANE: stop candidate.'
            del model
        except Exception as exc:
            row = {'capacity':capacity,'status':'failed','error_type':type(exc).__name__,'error':str(exc),
                   'traceback':traceback.format_exc()}
        result['capacities'].append(row)
        e.write_json(args.output / f'expert{args.expert}_{mode}_report.json',result)
        print(json.dumps(row),flush=True)
        gc.collect()


if __name__ == '__main__':
    main()
