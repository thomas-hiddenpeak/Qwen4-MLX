#!/usr/bin/env python3
"""Reauthor only GDN/GDN+PLE with repaired ILP4 readout, no radix tails.

Uses the optimized-v2-native-tail S4096 manifest and references every existing
weight file, QSA graph and top asset unchanged. S2048 and the original power-of-
two tail entrypoints remain available for a matched full-model comparison.
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

from coreai_q4_nax import install_nax_moe
from export_coreai_dense import DenseConfig
from export_coreai_hybrid import atomic_manifest, state_metadata
from export_coreai_optimized_chunks import source_hashes as optimized_source_hashes
from export_coreai_pd import verify_recorded_asset
from export_coreai_pd_shared import (
    buffer_signature, build_decoder_layer, export_generic, externalizable_buffers,
    geometry_signature,
)
from export_moe import Source, sha256_file


def source_hashes():
    hashes = optimized_source_hashes()
    for name in ('export_coreai_ilp_readout', 'export_coreai_pd_shared',
                 'coreai_gdn_ilp_probe', 'coreai_gdn_ilp_readout_probe'):
        hashes[name] = sha256_file(Path(__file__).with_name(name + '.py'))
    return hashes


def phase_contract(original):
    required = {'version': 2, 'backend': 'native-coreai-pd-shared', 'status': 'complete',
                'completeModelLayerSet': True, 'tokenChunk': 4096, 'integerExpertGrouping': True,
                'gdnPrefillRows': 4, 'naxMoEProjections': 'all', 'naxDownBlock': 32,
                'naxGateUpPolicy': 'native-parity-v2', 'moeTailPrecision': 'native-copy',
                'headProjection': 'metal-fp16-weights-fp32-logits',
                'moeRoutedOutputPrecision': 'original-native-graph'}
    if any(original.get(key) != value for key, value in required.items()):
        raise ValueError('Expected the complete optimized-v2 native-copy S4096 baseline')
    if len(original['layers']) != 48 or {item['index'] for item in original['layers']} != set(range(48)):
        raise ValueError('Expected all 48 unique layers')
    tails = original['tailChunks']
    if (original.get('radix4Tails', False) or len(tails) != len(set(tails)) or 2048 not in tails or
            any(type(size) is not int or size not in (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
                for size in tails)):
        raise ValueError('Expected existing power-of-two tails including S2048, with no radix tails')
    phases = {'main': 1, 'prefill': 4096, **{f'prefill_s{size}': size for size in tails}}
    for key in ('gdn', 'gdn-ple', 'qsa'):
        if not set(phases).issubset(original['sharedAssets'][key]['torchExport']):
            raise ValueError(f'Existing {key} graph lacks required phase functions')
    return phases


def for_primary(manifest, primary):
    if primary not in (2048, 4096):
        raise ValueError('This bounded overlay supports primary S2048/S4096')
    result = deepcopy(manifest)
    result['tokenChunk'] = primary
    result['tailChunks'] = [size for size in manifest['tailChunks'] if size < primary]
    result['qsaWorkingSets'] = [entry for entry in manifest.get('qsaWorkingSets', [])
                              if entry['tokenCount'] <= primary]
    function = 'prefill' if primary == 4096 else 'prefill_s2048'
    for section in ('assets', 'sharedAssets'):
        for asset in result[section].values():
            asset['prefillFunction'] = function
    for layer in result['layers']:
        layer['prefillFunction'] = function
    return result


def export(base):
    base = base.resolve()
    source_path = base / 'manifest-optimized-v2-native-tail-s4096.json'
    original = json.loads(source_path.read_text())
    phases = phase_contract(original)
    label = 'ilp4-readout-v2-native-tail'
    destination = base / f'manifest-{label}-s4096.json'
    paths = [destination, base / f'manifest-{label}-s2048.json',
             *[base / f'shared-{kind}-{label}.aimodel' for kind in ('gdn', 'gdn-ple')]]
    if any(path.exists() for path in paths):
        raise FileExistsError('Refusing to overwrite an existing readout-v2 overlay')
    for asset in (*original['assets'].values(), original['sharedAssets']['qsa']):
        verify_recorded_asset(base, asset)
    for layer in original['layers']:
        path = base / layer['weights']['path']
        if path.stat().st_size != layer['weights']['byteLength']:
            raise ValueError(f'Existing weight length differs: {path}')
    config_path = Source(0).directory / 'config.json'
    if str(config_path.parent) != original['modelDirectory'] or sha256_file(config_path) != original['configSHA256']:
        raise ValueError('Source model/config differs')
    config = json.loads(config_path.read_text())['text_config']
    c = DenseConfig.from_model(config)
    hashes, started = source_hashes(), time.monotonic()
    manifest = deepcopy(original)
    manifest.update(status='exporting', completeModelLayerSet=False,
        gdnPrefillRows=4, gdnPrefillPolicy='readout-v2', radix4Tails=False,
        gdnPrefillNumerics='ILP4 state/key reuse with baseline scalar per-row readout; no query preload or explicit readout part unroll; S1 original',
        authoringSourceSHA256=hashes, gdnReadoutSourceManifest=source_path.name,
        gdnReadoutSourceManifestSHA256=sha256_file(source_path),
        gdnReadoutValidation={'deviceValidated': False, 'qualityAccepted': False,
            'weightRecordsAndFilesReused': True, 'qsaAndTopAssetsUnchanged': True,
            'weightBytesRewritten': 0, 'fullWeightFilesRehashed': False,
            'representativeWeightSHA256Verified': []})
    atomic_manifest(destination, manifest)
    for index in (0, 1):
        module, kind, states, bindings, geometry, kernels = build_decoder_layer(
            Source(index), config, original['capacity'], prefill_sdpa_fp16=True,
            fuse_gateup=True, moe_tile=(16, 32, 64), flat_q4=True, integer_grouping=True,
            gdn_prefill_rows=4, gdn_prefill_policy='readout-v2',
            moe_direct_transfers=True, moe_tail_precision='native-copy')
        kernels += install_nax_moe(module, projections='all', simdgroups=1,
                                   down_block=32, gateup_policy='native-parity-v2')
        kernels += module.moe.custom_kernels()
        named, geo = externalizable_buffers(module, geometry)
        previous = next(layer for layer in original['layers'] if layer['index'] == index)
        signature = buffer_signature(named)
        expected = [{key: record[key] for key in ('inputName', 'bufferName', 'dtype', 'shape')}
                    for record in previous['weights']['buffers']]
        if (kind != 'gdn' or signature != expected or bindings != previous['stateBindings'] or
                state_metadata(states) != previous['initialState']):
            raise ValueError(f'Layer {index} learned/state contract changed')
        for (name, value), record in zip(named, previous['weights']['buffers'], strict=True):
            if hashlib.sha256(memoryview(value.numpy()).cast('B')).hexdigest() != record['sha256']:
                raise ValueError(f'Layer {index} learned bytes differ: {name}')
        examples = {}
        for entry, size in phases.items():
            inputs = {'stream': torch.zeros(1, size, c.width, dtype=torch.float16)}
            if module.ple is not None:
                inputs['ple_embedding'] = torch.zeros(1, size, c.ple_dim, dtype=torch.float16)
            examples[entry] = {**inputs, **states}
        key = kind + ('-ple' if module.ple is not None else '')
        asset = export_generic(module, named, geometry, examples, ('stream_out', *bindings.values()),
            base / f'shared-{key}-{label}.aimodel', list(dict.fromkeys(kernels)),
            metal_weight_inputs=original.get('metalWeightInputs', False))
        asset.update(geometrySignature=geometry_signature(geo), stateBindings=bindings,
            initialState=state_metadata(states), kind=kind, hasPLE=module.ple is not None,
            exampleLayer=index, authoringSourceSHA256=hashes,
            gdnPrefillRows=4, gdnPrefillPolicy='readout-v2', naxMoEProjections='all',
            naxDownBlock=32, naxGateUpPolicy='native-parity-v2', moeDirectTransfers=True,
            moeInverseCopy=True, moeTailPrecision='native-copy',
            moeRoutedOutputPrecision='original-native-graph')
        verify_recorded_asset(base, asset)
        manifest['sharedAssets'][key] = asset
        for layer in manifest['layers']:
            if layer['sharedAsset'] != key:
                continue
            if ([{name: record[name] for name in ('inputName', 'bufferName', 'dtype', 'shape')}
                    for record in layer['weights']['buffers']] != signature or
                    layer['stateBindings'] != bindings or layer['initialState'] != state_metadata(states)):
                raise ValueError(f'Layer {layer["index"]} cannot share repaired {key}')
            layer.update({name: asset[name] for name in ('path', 'function', 'prefillFunction',
                'inputNames', 'outputNames', 'modelBytes', 'files')})
        manifest['gdnReadoutValidation']['representativeWeightSHA256Verified'].append(index)
        atomic_manifest(destination, manifest)
        print(label, key, 'complete', round(time.monotonic() - started, 3), flush=True)
        del module, named, geo, states, examples
        gc.collect()
    if source_hashes() != hashes:
        raise ValueError('Sources changed during authoring; retained assets need review')
    if (manifest['assets'] != original['assets'] or manifest['sharedAssets']['qsa'] != original['sharedAssets']['qsa'] or
            manifest['qsaWorkingSets'] != original['qsaWorkingSets'] or manifest['tailChunks'] != original['tailChunks'] or
            [layer['weights'] for layer in manifest['layers']] != [layer['weights'] for layer in original['layers']]):
        raise AssertionError('Readout overlay must preserve all weights, QSA, top assets and phase boundaries')
    manifest.update(status='complete', completeModelLayerSet=True,
        gdnReadoutAuthoringSeconds=time.monotonic() - started,
        modelBytes=sum(asset['modelBytes'] for asset in manifest['assets'].values())
            + sum(asset['modelBytes'] for asset in manifest['sharedAssets'].values())
            + sum(layer['weights']['byteLength'] for layer in manifest['layers']))
    for primary in (2048, 4096):
        path = base / f'manifest-{label}-s{primary}.json'
        atomic_manifest(path, for_primary(manifest, primary))
        print('READY', path, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    export(args.base)
