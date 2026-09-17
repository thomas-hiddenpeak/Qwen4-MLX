#!/usr/bin/env python3
"""CPU-author isolated current HCRead/HCWrite probes with real layer0 weights.

The shared input residual stream is reused from the supplied external-layer spec;
its provenance is recorded, not promoted to a native model activation capture.
Current prefill TensorOps projections and original FP16 rounding boundaries are
retained. Each learned input references the original byte range without copying
the model bank. This script does not execute CoreAI or any GPU API.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from coreai_tensor_matmul import get_tensor_kernel
from export_coreai_dense import DenseConfig, HCRead, HCWrite
from export_coreai_pd import phase_linear
from export_coreai_pd_shared import ExternalModule
from export_moe import sha256_file


def tensor_input(path, value):
    array = value.detach().contiguous().numpy()
    array.tofile(path)
    return {'file': str(path.resolve()), 'offset': 0, 'bytes': array.nbytes,
            'shape': list(array.shape), 'dtype': str(array.dtype)}


def read_input(entry, base):
    file = Path(entry['file'])
    if not file.is_absolute():
        file = base / file
    dtype = np.dtype(entry['dtype'])
    elements = int(np.prod(entry['shape']))
    if dtype.itemsize * elements != entry['bytes']:
        raise ValueError('Input byte length does not match shape/dtype')
    with file.open('rb') as source:
        source.seek(entry['offset'])
        raw = source.read(entry['bytes'])
    if len(raw) != entry['bytes']:
        raise ValueError(f'Truncated tensor input: {file}')
    return torch.from_numpy(np.frombuffer(raw, dtype=dtype).reshape(entry['shape']).copy()), raw


def make_read(config):
    # Runtime arguments replace every placeholder; these zeros never stand in
    # for learned weights in the numerical oracle or the device probe spec.
    module = HCRead(config, {
        'input_mix_weight_down.weight': np.zeros((config.low_rank, config.width), np.float16),
        'input_mix_weight_up.weight': np.zeros((config.width, config.low_rank), np.float16),
        'hc_norm.weight': np.zeros(config.width, np.float16),
        'block_inject_weight.weight': np.zeros((config.streams, config.width), np.float16),
    })
    module.linear = phase_linear
    return module


def export(manifest_path, stream_spec_path, output):
    import coreai_torch

    manifest_path, stream_spec_path, output = (p.resolve() for p in
                                               (manifest_path, stream_spec_path, output))
    manifest = json.loads(manifest_path.read_text())
    if manifest['version'] != 2 or manifest['status'] != 'complete':
        raise ValueError('A complete v2 shared manifest is required')
    config_path = Path(manifest['modelDirectory']) / 'config.json'
    if sha256_file(config_path) != manifest['configSHA256']:
        raise ValueError('Source config identity differs')
    config = DenseConfig.from_model(json.loads(config_path.read_text())['text_config'])
    layer = next(item for item in manifest['layers'] if item['index'] == 0)
    weight_file = manifest_path.parent / 'weights' / layer['weights']['path']
    if not weight_file.exists():
        weight_file = manifest_path.parent / layer['weights']['path']
    entries = {item['bufferName']: item for item in layer['weights']['buffers']}
    stream_spec = json.loads(stream_spec_path.read_text())
    stream_entry = dict(stream_spec['inputs']['stream'])
    stream_entry['file'] = str((stream_spec_path.parent / stream_entry['file']).resolve())
    stream, stream_bytes = read_input(stream_entry, stream_spec_path.parent)
    if stream.shape != (1, 2048, config.width) or stream.dtype != torch.float16:
        raise ValueError('This bounded probe requires FP16 stream[1,2048,width]')
    if not torch.isfinite(stream).all():
        raise ValueError('Nonfinite input residual stream')

    output.mkdir(parents=True, exist_ok=False)
    reader, writer = make_read(config), HCWrite(config)
    named = list(reader.named_buffers())
    names = tuple(name for name, _ in named)
    input_names = ('stream', *names)
    external = ExternalModule(reader, names, 1).eval()
    read_args = (stream, *(value for _, value in named))
    write_args = (stream, torch.zeros(1, 2048, config.hidden, dtype=torch.float16),
                  torch.zeros(1, 2048, config.streams, 1, dtype=torch.float16))
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    converter.register_custom_kernels([get_tensor_kernel()])
    captured = {}
    for function, module, args, inputs, outputs in (
        ('read', external, read_args, input_names, ('mixed', 'injection')),
        ('write', writer, write_args, ('stream', 'output', 'injection'), ('stream_out',)),
    ):
        def export_fn(current, args=args, function=function):
            program = torch.export.export(current, args=args).run_decompositions(coreai_torch.get_decomp_table())
            placeholders = [node for node in program.graph.nodes if node.op == 'placeholder']
            captured[function] = [spec.target for spec, node in zip(program.graph_signature.input_specs, placeholders)
                                  if str(spec.kind).endswith('BUFFER') and len(node.users)]
            if captured[function]:
                raise ValueError('HC probe must not capture learned buffers')
            return program
        converter.add_pytorch_module(module, entrypoint_name=function,
            input_names=inputs, output_names=outputs, export_fn=export_fn)
    program = converter.to_coreai()
    program.optimize()
    program._mlir_module.operation.verify()
    asset = output / 'hc.aimodel'
    program.save_asset(asset)
    graphs = {}
    for function in ('read', 'write'):
        graph = str(program.get_graph(function))
        (output / (function + '-after.txt')).write_text(graph)
        graphs[function] = {'customGEMMInvocations': sum('coreai.metal4_kernel ' in line and 'qwen_mpp_' in line
                                                      for line in graph.splitlines())}

    cases = []
    with torch.inference_mode():
        for component in ('attention_read', 'moe_read'):
            weight_values, inputs, weight_provenance = [], {'stream': stream_entry}, []
            for name in names:
                record = entries[component + '.' + name]
                entry = {'file': str(weight_file), 'offset': record['byteOffset'],
                         'bytes': record['byteLength'], 'shape': record['shape'], 'dtype': record['dtype']}
                value, raw = read_input(entry, manifest_path.parent)
                if hashlib.sha256(raw).hexdigest() != record['sha256']:
                    raise ValueError(f'Learned weight bytes differ: {component}.{name}')
                inputs[name] = entry
                weight_values.append(value)
                weight_provenance.append(record)
            mixed, injection = external(stream, *weight_values)
            suffix = component.removesuffix('_read')
            mixed_entry = tensor_input(output / (suffix + '-mixed.bin'), mixed)
            injection_entry = tensor_input(output / (suffix + '-injection.bin'), injection)
            # Use the real HCRead result as a bounded standalone update. This
            # is not an attention/MoE output or a full-layer neural trajectory.
            written = writer(stream, mixed, injection)
            write_entry = tensor_input(output / (suffix + '-stream-out.bin'), written)
            for operation, arguments, expected in (
                ('read', inputs, {'mixed': mixed_entry, 'injection': injection_entry}),
                ('write', {'stream': stream_entry, 'output': mixed_entry, 'injection': injection_entry},
                 {'stream_out': write_entry}),
            ):
                label = suffix + '-' + operation
                spec = {'asset': asset.name, 'function': operation, 'inputs': arguments,
                        'output': label + '-output', 'repeats': 15, 'mapped': False}
                (output / (label + '-spec.json')).write_text(json.dumps(spec, indent=2) + '\n')
                cases.append({'name': label, 'spec': label + '-spec.json', 'expectedOutputs': expected,
                              'weights': weight_provenance if operation == 'read' else []})
    report = {'status': 'cpu-authored-device-unvalidated', 'deviceValidated': False,
              'productionEnabled': False, 'tokens': 2048, 'asset': asset.name,
              'assetBytes': sum(path.stat().st_size for path in asset.rglob('*') if path.is_file()),
              'capturedBuffers': captured, 'graphs': graphs,
              'manifest': str(manifest_path), 'manifestSHA256': sha256_file(manifest_path),
              'configSHA256': manifest['configSHA256'], 'inputSpec': str(stream_spec_path),
              'inputStreamSHA256': hashlib.sha256(stream_bytes).hexdigest(),
              'inputScope': 'Existing external-layer0 deterministic synthetic residual, same bytes as full-layer component timing; not a native activation capture.',
              'weightScope': 'True layer0 stored FP16 HC weights, SHA256 checked; no weight bytes rewritten.',
              'writeScope': 'Standalone HCWrite uses true HCRead mixed/injection output; no attention/MoE was run.',
              'policy': 'Current phase_linear TensorOps prefill with FP16 activation boundaries and original FP32 reductions.',
              'cpuOracle': 'Original HCRead/HCWrite functions, external weights; Metal reduction scheduling may differ.',
              'cases': cases}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--stream-spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    result = export(args.manifest, args.stream_spec, args.output)
    print(json.dumps({key: result[key] for key in ('status', 'asset', 'assetBytes', 'graphs')}, indent=2))
