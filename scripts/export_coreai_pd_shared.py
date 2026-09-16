#!/usr/bin/env python3
"""Export v2 CoreAI PD assets with shared graphs and explicit per-layer weights.

CPU authoring only. A complete v1 PD manifest pins model geometry and numerical
options. Learned module buffers become named runtime inputs; geometry buffers
remain constants. Graphs are shared by (attention kind, PLE presence), while each
layer owns a separate aligned little-endian weight file and recurrent/cache state.
Embedding/head remain unchanged constant assets copied from the v1 export.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import torch

from export_moe import Source, sha256_file
from export_coreai_dense import DenseConfig, HCRead, HCWrite, PLE, read_hc
from export_coreai_hybrid import atomic_manifest, prepare_layer, state_metadata
from export_coreai_pd import DecoderLayer, authoring_source_hashes as _pd_authoring_source_hashes, phase_linear, verify_recorded_asset
from export_coreai_q4_moe import PROJECTIONS, load_layer

ALIGNMENT = 16384
WEIGHT_DTYPES = {torch.float16, torch.float32, torch.int16, torch.int32}


def authoring_source_hashes():
    """Record the optional flat implementation alongside the original math."""
    hashes = _pd_authoring_source_hashes()
    hashes['coreai_q4_flat'] = sha256_file(Path(__file__).with_name('coreai_q4_flat.py'))
    return hashes


class ExternalModule(torch.nn.Module):
    """Generic explicit-buffer wrapper, also usable for attention/MoE probes."""

    def __init__(self, base, weight_names, input_count):
        super().__init__()
        self.base = base
        self.weight_names = tuple(weight_names)
        self.input_count = input_count

    def forward(self, *values):
        if len(values) != self.input_count + len(self.weight_names):
            raise ValueError('ExternalModule requires every explicit input and learned buffer')
        weights = dict(zip(self.weight_names, values[self.input_count:]))
        return torch.func.functional_call(self.base, weights,
            values[:self.input_count], strict=False)


def externalizable_buffers(module, geometry_names):
    """Explicit exclusions prevent geometry/RoPE tables becoming layer weights."""
    if list(module.named_parameters()):
        raise ValueError('Expected the existing buffer-only decoder modules')
    buffers = dict(module.named_buffers())
    missing = set(geometry_names) - buffers.keys()
    if missing:
        raise ValueError(f'Unknown geometry buffers: {sorted(missing)}')
    learned = [(name, value) for name, value in buffers.items() if name not in geometry_names]
    for name, value in learned:
        if value.device.type != 'cpu' or value.dtype not in WEIGHT_DTYPES or not value.is_contiguous():
            raise ValueError(f'Unsupported noncontiguous/non-CPU weight buffer: {name}')
    return learned, {name: buffers[name] for name in sorted(geometry_names)}


def tensor_bytes(value):
    if sys.byteorder != 'little':
        raise ValueError('Weight writer currently requires a little-endian host')
    return memoryview(value.detach().numpy()).cast('B')


def buffer_signature(named):
    return [{'inputName': f'weight_{index:03d}', 'bufferName': name,
             'dtype': str(value.dtype).removeprefix('torch.'), 'shape': list(value.shape)}
            for index, (name, value) in enumerate(named)]


def write_aligned_weights(path, named):
    """One durable binary per layer; no expansion of original packed I16 banks."""
    digest, offset, records = hashlib.sha256(), 0, []
    signatures = buffer_signature(named)
    with path.open('xb') as output:
        for metadata, (_, value) in zip(signatures, named, strict=True):
            padding = (-offset) % ALIGNMENT
            if padding:
                data = bytes(padding)
                output.write(data)
                digest.update(data)
                offset += padding
            view = tensor_bytes(value)
            record = {**metadata, 'byteOffset': offset, 'byteLength': len(view),
                      'sha256': hashlib.sha256(view).hexdigest()}
            output.write(view)
            digest.update(view)
            offset += len(view)
            records.append(record)
        # Permit mapping the complete binary directly as one page-aligned Metal
        # buffer. Padding is never part of a tensor's byteLength.
        padding = (-offset) % ALIGNMENT
        if padding:
            data = bytes(padding)
            output.write(data)
            digest.update(data)
            offset += padding
    return {'path': path.name, 'alignment': ALIGNMENT, 'byteLength': offset,
            'sha256': digest.hexdigest(), 'byteOrder': 'little', 'buffers': records}


def geometry_signature(geometry):
    return {name: {'shape': list(value.shape), 'dtype': str(value.dtype),
                   'sha256': hashlib.sha256(tensor_bytes(value)).hexdigest()}
            for name, value in geometry.items()}


def build_decoder_layer(source, config, capacity, *, prefill_sdpa_fp16, fuse_gateup,
                        moe_tile=(16, 32, 64), flat_q4=False):
    """Construct unchanged tensor-PD math; reusable by component diagnostics."""
    from coreai_q4_metal import MetalPackedQ4, get_q4_kernel
    from coreai_moe_chunk import ChunkQ4MoE
    from coreai_gdn_chunk import GDNRegisterPrefill
    from coreai_gdn_chunk_metal import get_gdn_recurrence_kernel
    from coreai_tensor_matmul import get_tensor_kernel
    from coreai_q4_grouped import get_plan_kernel, get_grouped_kernel
    from coreai_q4_gateup import get_gateup_kernel
    from coreai_qsa_chunk import QwenQSAChunk

    # Source starts at this layer's mlp prefix, as in the original exporter.
    prefix = source.prefix.removesuffix('.mlp.')
    layer = int(prefix.rsplit('.', 1)[-1])
    c = DenseConfig.from_model(config)
    moe = load_layer(source, config['num_experts'], config['num_experts_per_tok'])
    for name in PROJECTIONS:
        setattr(moe, name, MetalPackedQ4.from_packed(getattr(moe, name)))
    kind, attention, states, bindings, _, _ = prepare_layer(source, config, layer, capacity)
    read_attention = HCRead(c, read_hc(source, prefix + '.attn_hyper_connection'))
    read_moe = HCRead(c, read_hc(source, prefix + '.mlp_hyper_connection'))
    states, bindings = dict(states), dict(bindings)
    state_count = len(states)
    ple = None
    if layer == 1:
        source.prefix = prefix + '.ple.'
        names = ('key_proj.weight', 'value_proj.weight', 'norm_key.weight',
                 'norm_query.weight', 'norm_conv.weight', 'conv1d.weight')
        ple = PLE(c, {name: source.read(name) for name in names})
        states['ple_state'] = torch.zeros(1, c.ple_history, c.width, dtype=torch.float16)
        bindings['ple_state'] = 'next_ple_state'
        ple.linear = phase_linear
    block, columns, inner = moe_tile
    moe = ChunkQ4MoE(moe, block=block, columns=columns, inner=inner, fuse_gateup=fuse_gateup)
    kernels = [get_q4_kernel(), get_tensor_kernel(), get_gdn_recurrence_kernel(),
               get_plan_kernel(config['num_experts'], block), get_grouped_kernel(block, columns, inner)]
    if fuse_gateup:
        kernels.append(get_gateup_kernel(block, columns, inner))
    geometry = {'moe.decode.expert_ids'}
    if kind == 'gdn':
        attention.linear = phase_linear
        attention = GDNRegisterPrefill(attention)
    else:
        attention = QwenQSAChunk(attention, prefill_sdpa_fp16=prefill_sdpa_fp16)
        kernels += attention.custom_kernels()[1:]
        geometry.update('attention.source.' + name for name in
                        ('cache_positions', 'block_positions', 'cosine', 'sine'))
    if ple is not None:
        geometry.add('ple.gate_scale')
    read_attention.linear = phase_linear
    read_moe.linear = phase_linear
    module = DecoderLayer(attention, read_attention, moe, read_moe, HCWrite(c), state_count, ple).eval()
    if flat_q4:
        from coreai_q4_flat import flatten_moe_weights
        kernels += flatten_moe_weights(module)
    return module, kind, states, bindings, geometry, kernels


def export_generic(module, named, geometry_names, examples, output_names, path,
                   custom_kernels, *, metal_weight_inputs=False):
    """Export fixed phase entrypoints; verify learned buffers are never captured."""
    import coreai_torch
    from coreai_torch.composite_ops import SDPA

    standard_names = list(next(iter(examples.values())))
    metadata = buffer_signature(named)
    input_names = standard_names + [record['inputName'] for record in metadata]
    wrapper = ExternalModule(module, [name for name, _ in named], len(standard_names)).eval()
    values = tuple(value for _, value in named)
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    converter.register_custom_kernels(list(custom_kernels))
    externalize = [coreai_torch.ExternalizeSpec(target_class=SDPA,
        composite_op_name='scaled_dot_product_attention', composite_attrs=['scale', 'is_causal', 'window_size'])] \
        if any(isinstance(child, SDPA) for child in module.modules()) else None
    export_stats = {}
    for entry, inputs in examples.items():
        args = (*inputs.values(), *values)
        def export_fn(current, args=args, entry=entry):
            ep = torch.export.export(current, args=args).run_decompositions(coreai_torch.get_decomp_table())
            placeholders = [node for node in ep.graph.nodes if node.op == 'placeholder']
            captured = [spec.target for spec, node in zip(ep.graph_signature.input_specs, placeholders, strict=True)
                        if str(spec.kind).endswith('BUFFER') and len(node.users)]
            unexpected = set(captured) - {'base.' + name for name in geometry_names}
            if unexpected:
                raise ValueError(f'Learned buffers still captured by {entry}: {sorted(unexpected)}')
            export_stats[entry] = {'usedCapturedBuffers': captured,
                                   'userInputCount': len(ep.graph_signature.user_inputs)}
            return ep
        converter.add_pytorch_module(wrapper, entrypoint_name=entry, input_names=input_names,
            output_names=output_names, externalize_modules=externalize, export_fn=export_fn)
    program = converter.to_coreai()
    program.optimize()
    if metal_weight_inputs:
        from coreai.authoring import AllocationType, HardwareConstraints
        constraints = {record['inputName']: HardwareConstraints(AllocationType.MTLBuffer,
            [1] * (len(record['shape']) + 1), [1] * len(record['shape'])) for record in metadata}
        for entry in examples:
            program.set_hardware_constraints(entry, constraints)
        program.optimize()
    program.save_asset(path)
    files = [{'path': str(p.relative_to(path)), 'bytes': p.stat().st_size, 'sha256': sha256_file(p)}
             for p in sorted(path.rglob('*')) if p.is_file()]
    return {'path': path.name, 'function': 'main', 'prefillFunction': 'prefill',
            'inputNames': input_names, 'outputNames': list(output_names),
            'modelBytes': sum(record['bytes'] for record in files), 'files': files,
            'weightSignature': metadata, 'torchExport': export_stats,
            'metalWeightInputs': metal_weight_inputs}


def validate_baseline(manifest):
    if (manifest.get('version') != 1 or manifest.get('backend') != 'native-coreai-pd'
            or manifest.get('status') != 'complete' or not manifest.get('completeModelLayerSet')):
        raise ValueError('Expected a completed full v1 native-coreai-pd baseline')
    if (manifest.get('prefillKernels') != 'tensor' or manifest.get('q4Kernel') != 'metal'
            or manifest.get('stableProjections')):
        raise ValueError('Shared exporter requires the existing tensor-prefill/Metal-Q4 baseline')
    if len(manifest['layers']) != 48 or {x['index'] for x in manifest['layers']} != set(range(48)):
        raise ValueError('Baseline must contain each real model layer exactly once')
    chunk = manifest['tokenChunk']
    valid = (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
    tails = manifest.get('tailChunks', [])
    if chunk not in valid or len(set(tails)) != len(tails) or any(t not in valid or t >= chunk for t in tails):
        raise ValueError('Invalid baseline phase sizes')
    return [('main', 1), ('prefill', chunk)] + [(f'prefill_s{size}', size) for size in sorted(tails)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-pd', type=Path, required=True, help='Completed v1 PD directory; numerical options are inherited')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--layers', help='Comma-separated smoke subset; omit for all48')
    parser.add_argument('--components', nargs='+', choices=('layers', 'embedding', 'head'), default=['layers', 'embedding', 'head'])
    parser.add_argument('--metal-weight-inputs', action='store_true', help='Optional explicit MTLBuffer IO constraint; no demonstrated speed benefit')
    parser.add_argument('--flat-q4', action='store_true', help='Use rank-one external Q4 buffers and explicit kernel addressing; preserves packed bytes')
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    baseline_path = args.baseline_pd / 'manifest.json'
    baseline = json.loads(baseline_path.read_text())
    phases = validate_baseline(baseline)
    layers = list(range(48)) if args.layers is None else [int(value) for value in args.layers.split(',')]
    if not layers or len(set(layers)) != len(layers) or any(index not in range(48) for index in layers):
        raise ValueError('Invalid layer selection')
    if len(set(args.components)) != len(args.components):
        raise ValueError('Duplicate component selection')
    if args.output.exists():
        raise FileExistsError('Use a fresh shared-export directory; resume is not implemented')
    source = Source(0)
    config_path = source.directory / 'config.json'
    if str(source.directory) != baseline['modelDirectory'] or sha256_file(config_path) != baseline['configSHA256']:
        raise ValueError('Verified source/config differs from baseline')
    config = json.loads(config_path.read_text())['text_config']
    c = DenseConfig.from_model(config)
    estimated = len(layers) * 1_650_000_000 if 'layers' in args.components else 0
    estimated += sum(baseline['assets'][name]['modelBytes'] for name in args.components if name != 'layers')
    ancestor = args.output.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    if shutil.disk_usage(ancestor).free < estimated + 4_000_000_000:
        raise ValueError('Insufficient disk space for requested weight binaries/assets')
    args.output.mkdir(parents=True)
    manifest = deepcopy(baseline)
    manifest.update(version=2, backend='native-coreai-pd-shared', status='exporting', completeModelLayerSet=False,
        layers=[], assets={}, sharedAssets={}, modelBytes=0,
        requestedLayers=sorted(layers), requestedComponents=sorted(args.components),
        sourcePDDirectory=str(args.baseline_pd.resolve()), sourcePDManifestSHA256=sha256_file(baseline_path),
        exporterSHA256=sha256_file(Path(__file__)), authoringSourceSHA256=authoring_source_hashes(),
        metalWeightInputs=args.metal_weight_inputs, flatQ4=args.flat_q4,
        weightStorage={'byteOrder': 'little', 'alignment': ALIGNMENT, 'recommendedRuntimeOwner': 'residentMTLBuffer',
                       'mappingPolicy': 'Per-layer bytes are owned explicitly; graph/function objects shared by kind/PLE'},
        limitations=['New external-weight execution requires separate device/quality/performance validation.',
                     'No full-model or1000tokens/s acceptance implied by CPU export.'])
    manifest['baselineAuthoringSourceSHA256'] = baseline.get('authoringSourceSHA256', {})
    manifest['authoringChangesFromBaseline'] = sorted(name for name, digest in manifest['authoringSourceSHA256'].items()
        if manifest['baselineAuthoringSourceSHA256'].get(name) != digest)
    path = args.output / 'manifest.json'
    atomic_manifest(path, manifest)
    started = time.perf_counter()
    baseline_layers = {layer['index']: layer for layer in baseline['layers']}
    if 'layers' in args.components:
        for index in sorted(layers):
            layer_start = time.perf_counter()
            source = Source(index)
            module, kind, states, bindings, geometry_names, kernels = build_decoder_layer(source, config,
                baseline['capacity'], prefill_sdpa_fp16=baseline['prefillSDPA'] == 'float16',
                fuse_gateup=baseline['fusedGateUp'], moe_tile=tuple(baseline['moeTile']), flat_q4=args.flat_q4)
            named, geometry = externalizable_buffers(module, geometry_names)
            key = kind + ('-ple' if module.ple is not None else '')
            weights = write_aligned_weights(args.output / f'layer-{index:02d}.weights.bin', named)
            signature = buffer_signature(named)
            geo_signature = geometry_signature(geometry)
            if key not in manifest['sharedAssets']:
                examples = {}
                for entry, count in phases:
                    inputs = {'stream': torch.zeros(1, count, c.width, dtype=torch.float16)}
                    if module.ple is not None:
                        inputs['ple_embedding'] = torch.zeros(1, count, c.ple_dim, dtype=torch.float16)
                    examples[entry] = {**inputs, **states}
                asset = export_generic(module, named, geometry_names, examples, ('stream_out', *bindings.values()),
                    args.output / f'shared-{key}.aimodel', kernels, metal_weight_inputs=args.metal_weight_inputs)
                asset.update(geometrySignature=geo_signature, stateBindings=bindings,
                             initialState=state_metadata(states), kind=kind, hasPLE=module.ple is not None,
                             exampleLayer=index)
                manifest['sharedAssets'][key] = asset
                del examples
            else:
                asset = manifest['sharedAssets'][key]
                if (asset['weightSignature'] != signature or asset['geometrySignature'] != geo_signature
                        or asset['stateBindings'] != bindings or asset['initialState'] != state_metadata(states)):
                    raise ValueError(f'Layer{index} cannot safely reuse graph {key}: signature/geometry/state differs')
            original = baseline_layers[index]
            if original['kind'] != kind or original['stateBindings'] != bindings or original['initialState'] != state_metadata(states):
                raise ValueError(f'Layer{index} state contract changed from baseline')
            layer = deepcopy(original)
            layer.update({name: asset[name] for name in
                          ('path', 'function', 'prefillFunction', 'inputNames', 'outputNames', 'modelBytes', 'files')})
            layer.update(sharedAsset=key, weights=weights, sourceTensorReads=source.records)
            manifest['layers'].append(layer)
            atomic_manifest(path, manifest)
            print(f'Layer{index:02d} {key}: weights={weights["byteLength"]} bytes, graph={asset["path"]}, '
                  f'{time.perf_counter()-layer_start:.2f}s', flush=True)
            del module, named, geometry, states, source
            gc.collect()
    for name in ('embedding', 'head'):
        if name not in args.components:
            continue
        asset = baseline['assets'][name]
        verify_recorded_asset(args.baseline_pd, asset)
        shutil.copytree(args.baseline_pd / asset['path'], args.output / asset['path'])
        verify_recorded_asset(args.output, asset)
        manifest['assets'][name] = deepcopy(asset)
        atomic_manifest(path, manifest)
    manifest['status'] = 'complete'
    manifest['completeModelLayerSet'] = len(manifest['layers']) == 48 and set(manifest['assets']) == {'embedding', 'head'}
    manifest['modelBytes'] = (sum(asset['modelBytes'] for asset in manifest['sharedAssets'].values())
        + sum(asset['modelBytes'] for asset in manifest['assets'].values())
        + sum(layer['weights']['byteLength'] for layer in manifest['layers']))
    manifest['authoringSeconds'] = time.perf_counter() - started
    atomic_manifest(path, manifest)
    print(f'Shared export complete: {len(manifest["layers"])} layers, {len(manifest["sharedAssets"])} graphs, '
          f'{manifest["modelBytes"]} bytes', flush=True)


if __name__ == '__main__':
    main()
