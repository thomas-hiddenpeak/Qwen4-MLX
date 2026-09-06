#!/usr/bin/env python3
"""Bounded public Core ML probes for runtime-selected real MoE weights.

No full model is loaded. This script deliberately preserves dynamic weights or
indices as model inputs; the selected expert combination is never baked in.
ComputePlan is placement evidence, not a runtime ANE hardware trace.
Run using the existing workspace .venv, with no package installation.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import time

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
import numpy as np
import torch

from export_moe import Source, HIDDEN, INTERMEDIATE, PROJECTIONS, errors, write_json

BASE = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = BASE / 'results/moe-dynamic-v2'
FIXTURES = BASE / 'fixtures/moe-real/converted'


def program(k, bank=None):
    specs = [mb.TensorSpec(shape=(k, 1, HIDDEN), dtype=types.fp16)]
    if bank is None:
        specs += [mb.TensorSpec(shape=(k, INTERMEDIATE, HIDDEN), dtype=types.fp16),
                  mb.TensorSpec(shape=(k, INTERMEDIATE, HIDDEN), dtype=types.fp16),
                  mb.TensorSpec(shape=(k, HIDDEN, INTERMEDIATE), dtype=types.fp16)]
    else:
        specs += [mb.TensorSpec(shape=(k,), dtype=types.int32)]

    def graph(x, gate_weights, up_weights, down_weights):
        gate = mb.matmul(x=x, y=gate_weights, transpose_y=True, name='gate_projection')
        up = mb.matmul(x=x, y=up_weights, transpose_y=True, name='up_projection')
        negative = mb.mul(x=gate, y=np.float16(-1))
        denominator = mb.add(x=mb.exp(x=negative), y=np.float16(1))
        activated = mb.real_div(x=gate, y=denominator, name='silu_explicit')
        middle = mb.mul(x=activated, y=up, name='gated_up')
        return mb.matmul(x=middle, y=down_weights, transpose_y=True, name='y')

    if bank is None:
        @mb.program(input_specs=specs, opset_version=ct.target.macOS15)
        def result(x, gate_weights, up_weights, down_weights):
            return graph(x, gate_weights, up_weights, down_weights)
    else:
        @mb.program(input_specs=specs, opset_version=ct.target.macOS15)
        def result(x, expert_indices):
            weights = [mb.gather(x=bank[p], indices=expert_indices, axis=0, name=p+'_dynamic_gather')
                       for p in PROJECTIONS]
            return graph(x, *weights)
    return result


def get_plan(model):
    plan = ct.models.compute_plan.MLComputePlan.load_from_path(
        model.get_compiled_model_path(), compute_units=ct.ComputeUnit.CPU_AND_NE)
    operations = []
    def visit(block):
        for op in block.operations:
            if op.operator_name != 'const':
                usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
                operations.append({'operator': op.operator_name,
                                   'outputs': [v.name for v in op.outputs],
                                   'preferred': type(usage.preferred_compute_device).__name__ if usage else None,
                                   'supported': [type(d).__name__ for d in usage.supported_compute_devices] if usage else []})
            for child in op.blocks:
                visit(child)
    for function in plan.model_structure.program.functions.values():
        visit(function.block)
    projections = [o for o in operations if o['operator'].split('.')[-1] in ('matmul', 'linear', 'conv')]
    return {'operations': operations, 'projection_count': len(projections),
            'all_projections_prefer_ane': bool(projections) and all(
                o['preferred'] == 'MLNeuralEngineComputeDevice' for o in projections)}


def convert(k, folder, bank=None):
    started = time.perf_counter()
    model = ct.convert(program(k, bank), convert_to='mlprogram',
                       minimum_deployment_target=ct.target.macOS15,
                       compute_units=ct.ComputeUnit.CPU_AND_NE,
                       compute_precision=ct.precision.FLOAT16)
    model.save(str(folder / 'model.mlpackage'))
    record = {'top_k': k, 'mode': 'dynamic_weights' if bank is None else 'constant_bank_dynamic_gather',
              'conversion_seconds': time.perf_counter() - started,
              'dynamic_weight_input_bytes': k * 3 * HIDDEN * INTERMEDIATE * 2 if bank is None else 0,
              'plan': get_plan(model)}
    write_json(folder / 'plan.json', record)
    return model, record


def read_case():
    raw = np.load(FIXTURES / 'decode.npz')
    print('fixture fields:', raw.files, flush=True)
    return raw


def load_bank(ids):
    source = Source(0)
    bank = {p: [] for p in PROJECTIONS}
    started = time.perf_counter()
    for i in ids:
        weights = source.expert(int(i))
        for p in PROJECTIONS:
            bank[p].append(weights[p]['dense16'])
        del weights
    result = {p: np.stack(values) for p, values in bank.items()}
    return result, {'expert_ids': list(map(int, ids)), 'source_records': source.records,
                    'load_and_affine_decode_seconds': time.perf_counter() - started,
                    'fp16_bank_bytes': sum(a.nbytes for a in result.values())}


def stats(samples):
    return {'samples_ms': samples, 'median_ms': float(np.median(samples)),
            'min_ms': min(samples), 'max_ms': max(samples)}


def measure(model, k, ids, bank, bank_ids, folder, mode, runs):
    lookup = {int(e): i for i, e in enumerate(bank_ids)}
    routes = []
    for phase in ('decode', 'prefill'):
        raw = np.load(FIXTURES / (phase + '.npz'))
        for token in range(raw['x'].shape[1]):
            selected_ids = raw['selected_experts'][0, token, :k].astype(np.int32)
            routes.append({'phase': phase, 'position': int(raw['positions'][token]),
                           'token_id': int(raw['token_ids'][0, token]), 'ids': selected_ids,
                           'indices': np.array([lookup[int(i)] for i in selected_ids], np.int32),
                           'x': np.repeat(raw['x'][:, token:token+1].astype(np.float16), k, axis=0),
                           'routing_weights': raw['routing_weights'][0, token, :k],
                           'native_routed': raw['routed_sum'][:, token:token+1]})

    # Q4 decoding is outside the warmed path. Runtime CPU gather copies exactly
    # selected matrices from a host FP16 bank, changing real captured routes.
    def prepare(route):
        if mode == 'dynamic_weights':
            return {'x': route['x'], **{p.replace('_proj', '_weights'): np.ascontiguousarray(bank[p][route['indices']])
                                      for p in PROJECTIONS}}
        return {'x': route['x'], 'expert_indices': route['indices']}

    numerical = []
    for route in routes:
        with torch.inference_mode():
            tx = torch.from_numpy(route['x'].astype(np.float32))
            w = {p: torch.from_numpy(bank[p][route['indices']].astype(np.float32)) for p in PROJECTIONS}
            gate = tx @ w['gate_proj'].transpose(-1, -2)
            up = tx @ w['up_proj'].transpose(-1, -2)
            expected = (((gate * torch.sigmoid(gate)) * up) @ w['down_proj'].transpose(-1, -2)).numpy()
        output = np.asarray(model.predict(prepare(route))['y'])
        row = {key: route[key] for key in ('phase', 'position', 'token_id')}
        row.update({'expert_ids': route['ids'].tolist(), 'errors_vs_fp32': errors(output, expected),
                    'finite': bool(np.isfinite(output).all())})
        if k == 10:
            merged = np.sum(output.astype(np.float32) * route['routing_weights'][:, None, None], axis=0, keepdims=True)
            row['host_fp32_weighted_sum_vs_native_routed'] = errors(merged, route['native_routed'])
        numerical.append(row)
        np.savez(folder / f"{route['phase']}-{route['position']}.npz", x=route['x'], output=output,
                 fp32_reference=expected, expert_ids=route['ids'])

    inputs = prepare(routes[0])
    for _ in range(3):
        model.predict(inputs)
    ready, regroup, preparation, native, bridge = [], [], [], [], []
    for _ in range(runs):
        t = time.perf_counter_ns()
        model.predict(inputs)
        elapsed = (time.perf_counter_ns() - t) / 1e6
        ready.append(elapsed)
        core_ms = model.last_predict_duration_in_nano_seconds / 1e6
        native.append(core_ms)
        bridge.append(elapsed - core_ms)
    for n in range(runs):
        route = routes[n % len(routes)]
        t = time.perf_counter_ns()
        dynamic_inputs = prepare(route)
        prepared = time.perf_counter_ns()
        model.predict(dynamic_inputs)
        regroup.append((time.perf_counter_ns() - t) / 1e6)
        preparation.append((prepared - t) / 1e6)
    changed = dict(routes[0])
    changed['indices'] = routes[0]['indices'].copy()
    replacement = next(int(i) for i in routes[1]['indices'] if i not in routes[0]['indices'])
    changed['indices'][0] = replacement
    original_output = np.asarray(model.predict(inputs)['y'])
    changed_output = np.asarray(model.predict(prepare(changed))['y'])
    change_test = {'same_x': True, 'original_expert': int(routes[0]['ids'][0]),
                   'replacement_expert': int(bank_ids[replacement]),
                   'max_output_change': float(np.max(np.abs(changed_output.astype(np.float32) - original_output.astype(np.float32)))),
                   'output_changed': not np.array_equal(original_output, changed_output)}
    if not change_test['output_changed']:
        raise AssertionError('Changing a runtime expert weight did not change output')
    fresh_cpu, fresh_total = [], []
    if mode == 'dynamic_weights':
        # Exact source ranges are re-read and Q4 unpacked anew. OS file cache is
        # uncontrolled; this is not a cold-storage latency measurement.
        for route in routes:
            t = time.perf_counter_ns()
            fresh, _ = load_bank(route['ids'])
            cpu_done = time.perf_counter_ns()
            model.predict({'x': route['x'], **{p.replace('_proj', '_weights'): fresh[p] for p in PROJECTIONS}})
            fresh_cpu.append((cpu_done - t) / 1e6)
            fresh_total.append((time.perf_counter_ns() - t) / 1e6)
            del fresh
    result = {'input_boundary': 'Ready FP16 activation and FP16-rounded affine Q4 host bank',
              'reference': 'PyTorch FP32 arithmetic over FP16-rounded source affine Q4 weights',
              'numerical': numerical, 'warmups': 3, 'runs': runs,
              'ready_host_prediction': stats(ready),
              'coreml_prediction_from_features': stats(native),
              'python_marshaling_and_result_overhead': stats(bridge),
              'pure_device_compute_time': None,
              'provider_copy_only_time': None,
              'timing_note': 'Native property times Core ML predictionFromFeatures, not pure ANE execution; wall minus native includes validation, provider construction/copies and output conversion, not copy alone.',
              'runtime_weight_replacement': change_test,
              'fresh_q4_read_decode_stack_cpu': stats(fresh_cpu) if fresh_cpu else None,
              'fresh_q4_read_decode_stack_and_prediction': stats(fresh_total) if fresh_total else None,
              'real_route_recomposition_and_prediction': stats(regroup),
              'cpu_weight_preparation_only': stats(preparation),
              'new_route_boundary': 'Cycles actual decode and both selected real prefill token routes; no SSD/Q4 decode timed'}
    write_json(folder / 'prediction.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--mode', choices=('dynamic', 'bank'), default='dynamic')
    parser.add_argument('--ks', type=int, nargs='+', default=[1, 2])
    parser.add_argument('--predict', action='store_true', help='Run only in a coordinated ANE timing window')
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if any(k < 1 or k > 10 for k in args.ks) or args.runs < 1:
        raise ValueError('K must be 1..10, runs must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    fixture = np.load(FIXTURES / 'decode.npz')
    ids = fixture['selected_experts'].reshape(-1).astype(np.int32)
    bank_ids = np.array(sorted(set(map(int, (FIXTURES / 'actual-expert-ids.txt').read_text().replace(',', ' ').split()))), dtype=np.int32)
    bank = evidence = None
    if args.mode == 'bank' or args.predict:
        needed = bank_ids
        bank, evidence = load_bank(needed)
        bank_ids = needed
    report = {'schema': 'moe-runtime-dynamic-weights-v1',
              'coremltools': ct.__version__, 'macos': platform.mac_ver()[0],
              'source_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'compute_units': 'CPU_AND_NE', 'bank': evidence, 'cases': [],
              'limitations': ['ComputePlan is not a runtime hardware trace.',
                             'Only the routed SwiGLU expert path is probed; shared expert and routing are excluded.',
                             'No full decoder, full 512-expert conversion, or model server is started.']}
    destination = args.output / ('dynamic-report.json' if args.mode == 'dynamic' else 'bank-report.json')
    for k in args.ks:
        folder = args.output / f'{args.mode}-k{k}'
        folder.mkdir(parents=True, exist_ok=True)
        try:
            model, row = convert(k, folder, bank if args.mode == 'bank' else None)
            if args.predict and row['plan']['all_projections_prefer_ane']:
                row['prediction'] = measure(model, k, ids, bank, bank_ids, folder,
                                            'dynamic_weights' if args.mode == 'dynamic' else 'bank', args.runs)
            elif args.predict:
                row['prediction_skipped'] = 'Projection plan does not prefer ANE; terminate candidate.'
            report['cases'].append(row)
            print(json.dumps({'mode': args.mode, 'k': k, 'plan': row['plan']}, indent=2), flush=True)
            del model
            gc.collect()
        except Exception as exc:
            report['cases'].append({'k': k, 'error_type': type(exc).__name__, 'error': str(exc)})
            write_json(destination, report)
            raise
        write_json(destination, report)


if __name__ == '__main__':
    main()
