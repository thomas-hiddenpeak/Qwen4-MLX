#!/usr/bin/env python3
"""CPU-author optimized chunk overlays while reusing all learned files.

Reproduces the bounded optimized-v1 composition: direct gather, explicit FP32
tree tail returned to the native graph, stable integer grouping, NAX down only,
and GDN ILP4. Optional v2 uses BM32 down and measured-parity BM16 NAX gate/up;
v2-native-tail replaces only inverse row copying and retains native tail math.
This is an experimental overlay, not a quality/default promotion.
Existing S4096/S8192 embedding and Metal-head assets are verified and referenced.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import time

import torch

from coreai_moe_transfers import install_moe_transfers
from coreai_moe_transfers_tree import TreeFloatOutputChunkMoE
from coreai_moe_inverse_copy import InverseCopyChunkMoE
from coreai_q4_nax import install_nax_moe
from export_coreai_dense import DenseConfig
from export_coreai_hybrid import atomic_manifest, state_metadata
from export_coreai_pd import verify_recorded_asset
from export_coreai_pd_shared import (authoring_source_hashes, buffer_signature, build_decoder_layer,
    export_generic, externalizable_buffers, geometry_signature, resolve_qsa_working_sets,
    qsa_working_set_modules)
from export_moe import Source, sha256_file


def source_hashes():
    hashes = authoring_source_hashes()
    for name in ('coreai_moe_transfers_tree', 'coreai_moe_transfers_float_output',
                 'coreai_q4_nax', 'coreai_q4_nax_gateup', 'coreai_gdn_ilp_probe',
                 'coreai_q4_nax_m32_probe', 'coreai_q4_nax_gateup_parity',
                 'coreai_q4_nax_gateup_diagnostic', 'coreai_moe_inverse_copy'):
        hashes[name] = sha256_file(Path(__file__).with_name(name + '.py'))
    return hashes


def phase_contract(original, top, count):
    if count not in (2048, 4096, 8192) or original['capacity'] != 16384:
        raise ValueError('This bounded overlay supports S2048/S4096/S8192 at capacity16384')
    tails = sorted({*original['tailChunks'], *([2048] if count > 2048 else []),
                    *([4096] if count == 8192 else [])})
    phases = [('main', 1), ('prefill', count)] + [(f'prefill_s{size}', size) for size in tails]
    if (top.get('status') != 'complete' or top['tokenChunk'] != count or
            top['tailChunks'] != tails or top['phaseSizes'] != dict(phases) or
            top['configSHA256'] != original['configSHA256'] or
            top['headProjection'] != 'metal-fp16-weights-fp32-logits'):
        raise ValueError('Existing top assets do not match the requested phase/config contract')
    working = []
    for size in sorted({count, *tails}):
        if size >= 2048:
            working += resolve_qsa_working_sets(list(range(size, original['capacity'], 2048)), size, original['capacity'])
    return phases, working


def export(base, count, *, variant='v1'):
    if variant not in ('v1', 'v2', 'v2-native-tail'):
        raise ValueError('Expected optimized-v1, optimized-v2 or optimized-v2-native-tail variant')
    base = base.resolve()
    source_manifest = base / 'manifest-optimized-v1.json'
    original = json.loads(source_manifest.read_text())
    if (original.get('status') != 'complete' or not original.get('completeModelLayerSet') or
            original.get('integerExpertGrouping') is not True or original.get('gdnPrefillRows') != 4 or
            original.get('naxMoEProjections') != 'down' or
            original.get('moeRoutedOutputPrecision') != 'float32-tree-to-native-graph'):
        raise ValueError('Expected the complete optimized-v1 baseline and its exact numerical options')
    if len(original['layers']) != 48 or {item['index'] for item in original['layers']} != set(range(48)):
        raise ValueError('Expected all48 unique source layers')
    top_path = source_manifest if count == 2048 else base / f'assets-s{count}.json'
    top = json.loads(top_path.read_text())
    if count == 2048:
        top = {**top, 'phaseSizes': {'main': 1, 'prefill': 2048,
            **{f'prefill_s{size}': size for size in top['tailChunks']}}}
    phases, working = phase_contract(original, top, count)
    label = f'optimized-{variant}-s{count}'
    v2, native_tail = variant.startswith('v2'), variant == 'v2-native-tail'
    nax = {'projections': 'all' if v2 else 'down', 'simdgroups': 1,
           'down_block': 32 if v2 else 16, 'gateup_policy': 'native-parity-v2'}
    tail_precision = 'native-copy' if native_tail else 'float32'
    routed_precision = 'original-native-graph' if native_tail else 'float32-tree-to-native-graph'
    destination = base / f'manifest-{label}.json'
    paths = [destination, *[base / f'shared-{key}-{label}.aimodel' for key in ('gdn', 'gdn-ple', 'qsa')]]
    if any(path.exists() for path in paths):
        raise FileExistsError('Refusing to overwrite an existing optimized chunk overlay')
    for asset in top['assets'].values():
        verify_recorded_asset(base, asset)
    config_path = Source(0).directory / 'config.json'
    if str(config_path.parent) != original['modelDirectory'] or sha256_file(config_path) != original['configSHA256']:
        raise ValueError('Source config/model directory differs')
    config = json.loads(config_path.read_text())['text_config']
    c = DenseConfig.from_model(config)
    # Verify all durable records exist without rereading/hash-scanning78GB.
    # The three authoring representatives are additionally byte-hashed below.
    for layer in original['layers']:
        path = base / layer['weights']['path']
        if not path.exists():
            path = base / 'weights' / layer['weights']['path']
        if path.stat().st_size != layer['weights']['byteLength']:
            raise ValueError(f'Existing layer file size differs: {path}')
    started = time.monotonic()
    hashes = source_hashes()
    manifest = deepcopy(original)
    manifest.update(status='exporting', completeModelLayerSet=False, tokenChunk=count,
        tailChunks=[size for name, size in phases if name.startswith('prefill_s')],
        assets=deepcopy(top['assets']), qsaWorkingSets=working,
        authoringSourceSHA256=hashes, sourceSharedManifestSHA256=sha256_file(source_manifest),
        optimizedChunkSourceManifestSHA256=sha256_file(source_manifest),
        optimizedChunkOverlaySHA256=sha256_file(Path(__file__)),
        topAssetsManifest=top_path.name, topAssetsManifestSHA256=sha256_file(top_path),
        optimizedVariant=variant, naxMoEProjections=nax['projections'], naxDownBlock=nax['down_block'],
        naxGateUpPolicy=nax['gateup_policy'] if v2 else 'not-enabled',
        moeDirectTransfers=True, directMoETransfers=True, moeInverseCopy=native_tail,
        moeTailPrecision=tail_precision, directMoETailProductPrecision=tail_precision,
        moeRoutedOutputPrecision=routed_precision,
        headProjection=top['headProjection'], headProjectionSource=top_path.name,
        qsaWorkingSetSourceManifestSHA256=sha256_file(source_manifest),
        qsaWorkingSetAuthoringSourceSHA256=hashes,
        qsaWorkingSetOverlaySHA256=sha256_file(Path(__file__)),
        optimizedChunkValidation={'deviceValidated': False, 'qualityAccepted': False,
            'fullWeightFilesRehashed': False, 'weightRecordsAndFilesReused': True,
            'topAssetFilesSHA256Verified': True, 'representativeWeightSHA256Verified': []})
    atomic_manifest(destination, manifest)
    for index in (0, 1, 3):
        layer_started = time.monotonic()
        module, kind, states, bindings, geometry, kernels = build_decoder_layer(Source(index), config,
            manifest['capacity'], prefill_sdpa_fp16=True, fuse_gateup=True, moe_tile=(16, 32, 64),
            flat_q4=True, integer_grouping=True, gdn_prefill_rows=4)
        kernels += install_moe_transfers(module, tail_precision=None if native_tail else 'float32')
        module.moe = InverseCopyChunkMoE(module.moe) if native_tail else TreeFloatOutputChunkMoE(module.moe)
        kernels += install_nax_moe(module, **nax)
        kernels += module.moe.custom_kernels()
        named, geo = externalizable_buffers(module, geometry)
        old = next(layer for layer in original['layers'] if layer['index'] == index)
        signature = buffer_signature(named)
        expected = [{key: record[key] for key in ('inputName', 'bufferName', 'dtype', 'shape')}
                    for record in old['weights']['buffers']]
        if signature != expected or bindings != old['stateBindings'] or state_metadata(states) != old['initialState']:
            raise ValueError(f'Layer{index} weight/state signature changed')
        for (name, value), record in zip(named, old['weights']['buffers'], strict=True):
            if hashlib.sha256(memoryview(value.numpy()).cast('B')).hexdigest() != record['sha256']:
                raise ValueError(f'Learned weight bytes changed: layer{index}.{name}')
        manifest['optimizedChunkValidation']['representativeWeightSHA256Verified'].append(index)
        examples = {}
        entry_for_count = {}
        for entry, size in phases:
            inputs = {'stream': torch.zeros(1, size, c.width, dtype=torch.float16)}
            if module.ple is not None:
                inputs['ple_embedding'] = torch.zeros(1, size, c.ple_dim, dtype=torch.float16)
            examples[entry] = {**inputs, **states}
            entry_for_count[size] = entry
        overrides = qsa_working_set_modules(module, working) if kind == 'qsa' else {}
        for record in working if kind == 'qsa' else []:
            # A working set belongs to its own tokenCount, including tail
            # phases. Never copy the primary4096/8192 example to an S2048 view.
            examples[record['function']] = dict(examples[entry_for_count[record['tokenCount']]])
        key = kind + ('-ple' if module.ple is not None else '')
        asset = export_generic(module, named, geometry, examples, ('stream_out', *bindings.values()),
            base / f'shared-{key}-{label}.aimodel', list(dict.fromkeys(kernels)), entry_modules=overrides)
        asset.update(geometrySignature=geometry_signature(geo), stateBindings=bindings,
            initialState=state_metadata(states), kind=kind, hasPLE=module.ple is not None,
            exampleLayer=index, authoringSourceSHA256=hashes, gdnPrefillRows=4 if kind == 'gdn' else None,
            naxMoEProjections=nax['projections'], naxDownBlock=nax['down_block'],
            naxGateUpPolicy=nax['gateup_policy'] if v2 else 'not-enabled',
            moeDirectTransfers=True, moeInverseCopy=native_tail, moeTailPrecision=tail_precision,
            moeRoutedOutputPrecision=routed_precision)
        verify_recorded_asset(base, asset)
        manifest['sharedAssets'][key] = asset
        for layer in manifest['layers']:
            if layer['sharedAsset'] != key:
                continue
            current = [{name: record[name] for name in ('inputName', 'bufferName', 'dtype', 'shape')}
                       for record in layer['weights']['buffers']]
            if current != signature or layer['stateBindings'] != bindings or layer['initialState'] != state_metadata(states):
                raise ValueError(f'Layer{layer["index"]} cannot share the new {key} graph')
            layer.update({name: asset[name] for name in ('path', 'function', 'prefillFunction', 'inputNames', 'outputNames', 'modelBytes', 'files')})
        if kind == 'qsa':
            manifest['qsaWorkingSetAuthoringSeconds'] = time.monotonic() - layer_started
        atomic_manifest(destination, manifest)
        print(label, key, 'complete', round(time.monotonic() - started, 3), flush=True)
        del module, named, geo, states, examples, overrides
        gc.collect()
    if source_hashes() != hashes:
        raise ValueError('Authoring sources changed during export; retained assets require provenance review')
    if any(a['weights'] != b['weights'] for a, b in zip(manifest['layers'], original['layers'], strict=True)):
        raise AssertionError('Overlay must reuse every learned-weight record unchanged')
    manifest.update(status='complete', completeModelLayerSet=True,
        modelBytes=sum(asset['modelBytes'] for asset in manifest['sharedAssets'].values())
            + sum(asset['modelBytes'] for asset in manifest['assets'].values())
            + sum(layer['weights']['byteLength'] for layer in manifest['layers']),
        optimizedChunkAuthoringSeconds=time.monotonic() - started)
    atomic_manifest(destination, manifest)
    print(destination, flush=True)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--chunks', type=int, nargs='+', choices=(2048, 4096, 8192), default=[4096, 8192])
    parser.add_argument('--variant', choices=('v1', 'v2', 'v2-native-tail'), default='v1')
    args = parser.parse_args()
    if len(set(args.chunks)) != len(args.chunks):
        raise ValueError('Duplicate chunk selection')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    for chunk in args.chunks:
        export(args.base, chunk, variant=args.variant)
