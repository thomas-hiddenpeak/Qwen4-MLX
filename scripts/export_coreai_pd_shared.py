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
    for name in ('coreai_q4_flat', 'coreai_expert_grouping', 'coreai_head_metal', 'export_coreai_top_chunks',
                 'coreai_qsa_working_set', 'coreai_moe_transfers', 'coreai_moe_inverse_copy',
                 'coreai_gdn_ilp_probe', 'coreai_gdn_ilp_readout_probe'):
        hashes[name] = sha256_file(Path(__file__).with_name(name + '.py'))
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


def validate_contiguous_affine(*, flat_q4, moe_tile, contiguous_affine):
    """Reject unsupported optional loader combinations before reading weights."""
    if not contiguous_affine:
        return
    if not flat_q4:
        raise ValueError('--contiguous-affine requires --flat-q4')
    if (len(moe_tile) != 3 or moe_tile[0] not in (16, 32)
            or moe_tile[1] not in (32, 64) or moe_tile[2] != 64):
        raise ValueError('--contiguous-affine requires BM16/32, BN32/64 and BK=64')


def resolve_moe_transfers(enabled=False, tail_precision=None):
    """Pin the optional numerical boundary; omitted options preserve old graphs."""
    if tail_precision not in (None, 'float16', 'float32', 'native', 'native-copy'):
        raise ValueError('Unsupported MoE tail precision')
    if not enabled and tail_precision is not None:
        raise ValueError('--moe-tail-precision requires --moe-direct-transfers')
    return {'enabled': enabled, 'tailPrecision': (tail_precision or 'float32') if enabled else 'native'}


def validate_gdn_prefill_rows(rows):
    """Reject unsupported recurrence variants before reading model weights."""
    if type(rows) is not int or rows not in (1, 2, 4):
        raise ValueError('--gdn-prefill-rows must be 1, 2 or 4')


def validate_gdn_prefill_policy(rows, policy):
    validate_gdn_prefill_rows(rows)
    if policy not in ('experimental-v1', 'readout-v2'):
        raise ValueError('--gdn-prefill-policy must be experimental-v1 or readout-v2')
    if policy == 'readout-v2' and rows != 4:
        raise ValueError('--gdn-prefill-policy readout-v2 requires --gdn-prefill-rows 4')


def build_decoder_layer(source, config, capacity, *, prefill_sdpa_fp16, fuse_gateup,
                        moe_tile=(16, 32, 64), flat_q4=False, integer_grouping=False,
                        contiguous_affine=False, moe_direct_transfers=False, moe_tail_precision=None,
                        gdn_prefill_rows=1, gdn_prefill_policy='experimental-v1'):
    """Construct unchanged tensor-PD math; reusable by component diagnostics."""
    validate_gdn_prefill_policy(gdn_prefill_rows, gdn_prefill_policy)
    validate_contiguous_affine(flat_q4=flat_q4, moe_tile=moe_tile,
                               contiguous_affine=contiguous_affine)
    transfers = resolve_moe_transfers(moe_direct_transfers, moe_tail_precision)
    if transfers['enabled'] and (config.get('num_experts_per_tok') != 10 or
                                config.get('hidden_size', 0) <= 0 or config['hidden_size'] % 4):
        raise ValueError('Direct MoE transfers require top10 and hidden size divisible by4')
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
        if gdn_prefill_rows != 1:
            from coreai_gdn_ilp_probe import install_gdn_ilp
            kernels += [kernel for kernel in install_gdn_ilp(attention, rows=gdn_prefill_rows,
                                                            policy=gdn_prefill_policy)
                        if kernel not in kernels]
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
    if contiguous_affine:
        from coreai_q4_flat import install_contiguous_affine
        kernels += install_contiguous_affine(module)
    if integer_grouping:
        from coreai_expert_grouping import enable_integer_grouping
        kernels += enable_integer_grouping(module)
    if transfers['enabled']:
        from coreai_moe_transfers import install_moe_transfers
        precision = transfers['tailPrecision']
        kernels += install_moe_transfers(module, tail_precision=None if precision == 'native' else precision)
    return module, kind, states, bindings, geometry, kernels


def export_generic(module, named, geometry_names, examples, output_names, path,
                   custom_kernels, *, metal_weight_inputs=False, entry_modules=None):
    """Export fixed phase entrypoints; verify learned buffers are never captured."""
    import coreai_torch
    from coreai_torch.composite_ops import SDPA

    standard_names = list(next(iter(examples.values())))
    metadata = buffer_signature(named)
    input_names = standard_names + [record['inputName'] for record in metadata]
    entry_modules = {} if entry_modules is None else dict(entry_modules)
    if set(entry_modules) - set(examples):
        raise ValueError('Entry module override has no corresponding input example')
    _, original_geometry = externalizable_buffers(module, geometry_names)
    original_geometry_signature = geometry_signature(original_geometry)
    for entry, alternate in entry_modules.items():
        alternate_named, alternate_geometry = externalizable_buffers(alternate, geometry_names)
        if (buffer_signature(alternate_named) != metadata or
                geometry_signature(alternate_geometry) != original_geometry_signature):
            raise ValueError(f'Entry {entry} changes the shared weight or geometry signature')
    wrappers = {entry: ExternalModule(entry_modules.get(entry, module),
        [name for name, _ in named], len(standard_names)).eval() for entry in examples}
    values = tuple(value for _, value in named)
    converter = coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    converter.register_custom_kernels(list(custom_kernels))
    externalize = [coreai_torch.ExternalizeSpec(target_class=SDPA,
        composite_op_name='scaled_dot_product_attention', composite_attrs=['scale', 'is_causal', 'window_size'])] \
        if any(isinstance(child, SDPA) for wrapper in wrappers.values() for child in wrapper.modules()) else None
    export_stats = {}
    for entry, inputs in examples.items():
        if list(inputs) != standard_names:
            raise ValueError(f'Entry {entry} changes the shared input-name contract')
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
        converter.add_pytorch_module(wrappers[entry], entrypoint_name=entry, input_names=input_names,
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


RADIX4_TAILS = (48, 192, 768)


def radix4_tail_chunks(primary):
    """Bounded 3*16*4**k family; no padding or prompt-dependent shapes."""
    if type(primary) is not int or primary < 1:
        raise ValueError('Expected a positive integer primary chunk')
    return [size for size in RADIX4_TAILS if size < primary]


def resolve_phases(baseline, chunk=None, *, integer_grouping=False, radix4_tails=False):
    """Inherit a verified v1 geometry while optionally enlarging v2 prefill."""
    inherited = validate_baseline(baseline)
    if type(radix4_tails) is not bool:
        raise ValueError('radix4_tails must be boolean')
    if chunk is None and not radix4_tails:
        return inherited
    if chunk is None:
        chunk = baseline['tokenChunk']
    if chunk not in (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192):
        raise ValueError('Unsupported shared prefill chunk')
    if chunk > 2048 and not integer_grouping:
        raise ValueError('Chunks above2048 require --integer-grouping')
    if chunk > baseline['capacity']:
        raise ValueError('Chunk exceeds inherited cache capacity')
    tails = {size for _, size in inherited if 1 < size < chunk}
    # When8192 is primary,4096 is also useful as an exact unpadded tail.
    if chunk == 8192:
        tails.add(4096)
    if radix4_tails:
        tails.update(radix4_tail_chunks(chunk))
    return [('main', 1), ('prefill', chunk)] + [(f'prefill_s{size}', size) for size in sorted(tails)]


def _boolean_option(value):
    if value.lower() not in ('true', 'false'):
        raise argparse.ArgumentTypeError('Expected true or false')
    return value.lower() == 'true'


def resolve_qsa_working_sets(limits, token_count, capacity):
    """Optional static views of the primary prefill chunk, never S1 overrides."""
    from coreai_qsa_working_set import working_set_function
    if limits is None:
        return []
    if (not limits or len(set(limits)) != len(limits) or token_count <= 1 or
            any(not isinstance(limit, int) or not token_count <= limit < capacity or limit % 4 for limit in limits)):
        raise ValueError('QSA working-set limits must be unique4-aligned values covering tokenChunk and below capacity')
    return [{'tokenCount': token_count, 'kvLimit': limit, 'function': working_set_function(token_count, limit)}
            for limit in sorted(limits)]


def qsa_working_set_modules(module, entries):
    """Share every learned tensor while replacing only attention working views."""
    from coreai_qsa_chunk import QwenQSAChunk
    from coreai_qsa_working_set import QwenQSAWorkingSet
    if not entries:
        return {}
    if not isinstance(module.attention, QwenQSAChunk):
        raise ValueError('QSA working-set overrides require QwenQSAChunk attention')
    original = module.attention
    result = {}
    for entry in entries:
        if entry['function'] in result:
            raise ValueError('Duplicate QSA working-set entry')
        validated = resolve_qsa_working_sets([entry['kvLimit']], entry['tokenCount'], original.capacity)[0]
        if entry != validated:
            raise ValueError('Invalid QSA working-set function name or metadata')
        attention = QwenQSAWorkingSet(original.source, entry['kvLimit'],
            prefill_sdpa_fp16=original.prefill_sdpa_fp16, tile_m=original.tile_m, tile_n=original.tile_n)
        result[entry['function']] = DecoderLayer(attention, module.attention_read, module.moe,
            module.moe_read, module.write, module.state_count, module.ple).eval()
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-pd', type=Path, required=True, help='Completed v1 PD directory; numerical options are inherited')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--layers', help='Comma-separated smoke subset; omit for all48')
    parser.add_argument('--components', nargs='+', choices=('layers', 'embedding', 'head'), default=['layers', 'embedding', 'head'])
    parser.add_argument('--metal-weight-inputs', action='store_true', help='Optional explicit MTLBuffer IO constraint; no demonstrated speed benefit')
    parser.add_argument('--flat-q4', action='store_true', help='Use rank-one external Q4 buffers and explicit kernel addressing; preserves packed bytes')
    parser.add_argument('--contiguous-affine', action='store_true',
                        help='Optional adjacent-word affine loader for routed prefill; requires --flat-q4 and baseline BK64; decode unchanged')
    parser.add_argument('--integer-grouping', action='store_true', help='Use stable I32 expert grouping; required for chunks above2048')
    parser.add_argument('--radix4-tails', action='store_true',
                        help='Add optional unpadded S48/S192/S768 tail phases below the primary chunk; defaults unchanged')
    parser.add_argument('--gdn-prefill-rows', type=int, choices=(1, 2, 4), default=1,
                        help='GDN value rows per SIMD during prefill; default1 preserves original kernel;2/4 opt into ILP; S1 unchanged')
    parser.add_argument('--gdn-prefill-policy', choices=('experimental-v1', 'readout-v2'), default='experimental-v1',
                        help='Explicit ILP numerical policy; readout-v2 requires rows4 and restores the baseline scalar readout; default rows1 remains unchanged')
    parser.add_argument('--moe-direct-transfers', action='store_true',
                        help='Optional direct ordered gather and fused routed tail; S1 decode unchanged')
    parser.add_argument('--moe-tail-precision', choices=('float16', 'float32', 'native', 'native-copy'),
                        help='Requires --moe-direct-transfers; default float32 products, float16 rounds products, native keeps original tail graph, native-copy replaces inverse copying and preserves native weighted math')
    parser.add_argument('--chunk', type=int, choices=(4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192),
                        help='Override primary chunk while inheriting v1 capacity/geometry and smaller phases')
    parser.add_argument('--metal-head', type=_boolean_option, nargs='?', const=True, default=False,
                        help='Optional true/false FP32-output Metal head; omitted keeps original F.linear math')
    parser.add_argument('--qsa-working-sets', type=int, nargs='+',
                        help='Optional static KV limits for the primary prefill chunk;4-aligned and below capacity; states stay full capacity')
    args = parser.parse_args(argv)
    validate_gdn_prefill_policy(args.gdn_prefill_rows, args.gdn_prefill_policy)
    moe_transfers = resolve_moe_transfers(args.moe_direct_transfers, args.moe_tail_precision)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    baseline_path = args.baseline_pd / 'manifest.json'
    baseline = json.loads(baseline_path.read_text())
    inherited_phases = validate_baseline(baseline)
    phases = resolve_phases(baseline, args.chunk, integer_grouping=args.integer_grouping,
                            radix4_tails=args.radix4_tails)
    qsa_working_sets = resolve_qsa_working_sets(args.qsa_working_sets, dict(phases)['prefill'], baseline['capacity'])
    validate_contiguous_affine(flat_q4=args.flat_q4, moe_tile=baseline['moeTile'],
                               contiguous_affine=args.contiguous_affine)
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
        contiguousAffine=args.contiguous_affine,
        integerExpertGrouping=args.integer_grouping,
        radix4Tails=args.radix4_tails,
        gdnPrefillRows=args.gdn_prefill_rows,
        gdnPrefillPolicy=args.gdn_prefill_policy if args.gdn_prefill_rows > 1 else 'original',
        gdnPrefillNumerics=('Original one-value-row SIMD recurrence; S1 decode unchanged' if args.gdn_prefill_rows == 1 else
            f'GDN prefill only: {args.gdn_prefill_rows} independent value rows per SIMD; policy={args.gdn_prefill_policy}; sequential FP32 state updates; S1 uses original kernel; full-model device validation remains separate'),
        moeDirectTransfers=moe_transfers['enabled'], moeTailPrecision=moe_transfers['tailPrecision'],
        moeTransferNumerics=('Original gather and tail graph; S1 decode unchanged' if not moe_transfers['enabled'] else
            'Direct FP16 gather; S1 decode unchanged; ' + {
                'native': 'original native inverse/weight/reduce graph',
                'native-copy': 'direct inverse FP16 row copy; original native scores/products/reduction/casts/shared addition',
                'float16': 'products rounded FP16 then slot-order FP32 sum and FP16 output',
                'float32': 'FP32 products and slot-order FP32 sum then FP16 output; intentionally no eager per-product FP16 boundary'
            }[moe_transfers['tailPrecision']]),
        qsaWorkingSets=qsa_working_sets,
        tokenChunk=dict(phases)['prefill'], tailChunks=[count for name, count in phases if name.startswith('prefill_s')],
        headProjection=('metal-fp16-weights-fp32-logits' if args.metal_head
                        else 'original-linear' if phases != inherited_phases
                        else baseline.get('headProjection', 'original-linear')),
        weightStorage={'byteOrder': 'little', 'alignment': ALIGNMENT, 'recommendedRuntimeOwner': 'residentPerTensorMTLBuffer',
                       'mappingPolicy': 'Each weight tensor owns a shared buffer bound at offset zero; graphs are shared by kind/PLE'},
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
                fuse_gateup=baseline['fusedGateUp'], moe_tile=tuple(baseline['moeTile']), flat_q4=args.flat_q4,
                integer_grouping=args.integer_grouping, contiguous_affine=args.contiguous_affine,
                moe_direct_transfers=args.moe_direct_transfers, moe_tail_precision=args.moe_tail_precision,
                gdn_prefill_rows=args.gdn_prefill_rows, gdn_prefill_policy=args.gdn_prefill_policy)
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
                entry_modules = qsa_working_set_modules(module, qsa_working_sets) if kind == 'qsa' else {}
                for entry in entry_modules:
                    examples[entry] = dict(examples['prefill'])
                asset = export_generic(module, named, geometry_names, examples, ('stream_out', *bindings.values()),
                    args.output / f'shared-{key}.aimodel', kernels, metal_weight_inputs=args.metal_weight_inputs,
                    entry_modules=entry_modules)
                asset.update(geometrySignature=geo_signature, stateBindings=bindings,
                             initialState=state_metadata(states), kind=kind, hasPLE=module.ple is not None,
                             exampleLayer=index, contiguousAffine=args.contiguous_affine,
                             moeDirectTransfers=moe_transfers['enabled'], moeTailPrecision=moe_transfers['tailPrecision'],
                             gdnPrefillRows=args.gdn_prefill_rows if kind == 'gdn' else None,
                             gdnPrefillPolicy=(args.gdn_prefill_policy if args.gdn_prefill_rows > 1 else 'original') if kind == 'gdn' else None)
                manifest['sharedAssets'][key] = asset
                del examples, entry_modules
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
    top_reexport = [name for name in ('embedding', 'head') if name in args.components
                    and (phases != inherited_phases or (name == 'head' and args.metal_head))]
    if top_reexport:
        from export_coreai_top_chunks import export_top_group
        top = export_top_group(args.output, dict(phases), c, components=tuple(top_reexport), metal_head=args.metal_head)
        manifest['assets'].update(top['assets'])
        manifest['topSourceTensorReads'] = top['sourceSlices']
        atomic_manifest(path, manifest)
    for name in ('embedding', 'head'):
        if name not in args.components:
            continue
        if name in top_reexport:
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
