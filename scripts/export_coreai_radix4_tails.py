#!/usr/bin/env python3
"""CPU-author bounded radix-4 tail overlays without duplicating decoder weights.

Adds the general 3*16*4**k family S48/S192/S768 to a verified optimized v2
native-copy S4096 baseline. The S2048 and S4096 manifests share all five graph
assets and all48 weight files. Top constant assets are reauthored once because
their public entrypoints need the new shapes; no runtime/device work is done.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time

import torch

from coreai_q4_nax import install_nax_moe
from export_coreai_dense import DenseConfig
from export_coreai_hybrid import atomic_manifest, state_metadata
from export_coreai_optimized_chunks import source_hashes as optimized_source_hashes
from export_coreai_pd import verify_recorded_asset
from export_coreai_pd_shared import (buffer_signature, build_decoder_layer,
    export_generic, externalizable_buffers, geometry_signature, radix4_tail_chunks,
    qsa_working_set_modules)
from export_coreai_top_chunks import export_top_group
from export_moe import Source, sha256_file


def source_hashes():
    result = optimized_source_hashes()
    for name in ('export_coreai_radix4_tails', 'export_coreai_pd_shared',
                 'export_coreai_optimized_chunks'):
        result[name] = sha256_file(Path(__file__).with_name(name + '.py'))
    return result


def phase_contract(original):
    required = {'version': 2, 'backend': 'native-coreai-pd-shared', 'status': 'complete',
        'completeModelLayerSet': True, 'tokenChunk': 4096, 'integerExpertGrouping': True,
        'gdnPrefillRows': 4, 'naxMoEProjections': 'all', 'naxDownBlock': 32,
        'naxGateUpPolicy': 'native-parity-v2', 'moeTailPrecision': 'native-copy',
        'headProjection': 'metal-fp16-weights-fp32-logits'}
    if any(original.get(key) != value for key, value in required.items()):
        raise ValueError('Expected the complete optimized-v2 native-copy S4096 baseline')
    if len(original['layers']) != 48 or {x['index'] for x in original['layers']} != set(range(48)):
        raise ValueError('Expected all48 unique layers')
    tails = sorted(set(original['tailChunks']) | set(radix4_tail_chunks(4096)))
    return {'main': 1, 'prefill': 4096, **{f'prefill_s{n}': n for n in tails}}


def for_primary(manifest, primary):
    """Reuse the same graph assets through their existing phase entrypoint."""
    if primary not in (2048, 4096):
        raise ValueError('This bounded overlay supports primary2048/4096')
    result = deepcopy(manifest)
    result['tokenChunk'] = primary
    result['tailChunks'] = [n for n in manifest['tailChunks'] if n < primary]
    result['qsaWorkingSets'] = [x for x in manifest.get('qsaWorkingSets', [])
                              if x['tokenCount'] <= primary]
    function = 'prefill' if primary == 4096 else 'prefill_s2048'
    for section in ('assets', 'sharedAssets'):
        for record in result[section].values(): record['prefillFunction'] = function
    for layer in result['layers']: layer['prefillFunction'] = function
    return result


def export(base):
    base = base.resolve()
    source_manifest = base / 'manifest-optimized-v2-native-tail-s4096.json'
    original = json.loads(source_manifest.read_text())
    phases = phase_contract(original)
    label = 'radix4-v2-native-tail'
    destination = base / f'manifest-{label}-s4096.json'
    paths = [destination, base / f'manifest-{label}-s2048.json',
        *[base / f'{kind}-{label}.aimodel' for kind in ('embedding', 'head')],
        *[base / f'shared-{kind}-{label}.aimodel' for kind in ('gdn', 'gdn-ple', 'qsa')]]
    if any(p.exists() for p in paths):
        raise FileExistsError('Refusing to overwrite an existing radix-4 overlay')
    if shutil.disk_usage(base).free < 4_000_000_000:
        raise ValueError('Need4GB free for new constant top assets and graph metadata')
    for asset in original['assets'].values(): verify_recorded_asset(base, asset)
    config_path = Source(0).directory / 'config.json'
    if str(config_path.parent) != original['modelDirectory'] or sha256_file(config_path) != original['configSHA256']:
        raise ValueError('Source config/model directory differs')
    config = json.loads(config_path.read_text())['text_config']
    c = DenseConfig.from_model(config)
    for layer in original['layers']:
        path = base / layer['weights']['path']
        if path.stat().st_size != layer['weights']['byteLength']:
            raise ValueError(f'Existing weight file size differs: {path}')
    hashes, started = source_hashes(), time.monotonic()
    manifest = deepcopy(original)
    manifest.update(status='exporting', completeModelLayerSet=False,
        radix4Tails=True, radix4TailFamily=radix4_tail_chunks(4096),
        tailChunks=sorted(n for n in phases.values() if 1 < n < 4096),
        authoringSourceSHA256=hashes, radix4SourceManifest=source_manifest.name,
        radix4SourceManifestSHA256=sha256_file(source_manifest),
        radix4Validation={'deviceValidated': False, 'qualityAccepted': False,
            'weightRecordsAndFilesReused': True, 'fullWeightFilesRehashed': False,
            'representativeWeightSHA256Verified': []})
    atomic_manifest(destination, manifest)
    top = export_top_group(base, phases, c, suffix='-' + label, metal_head=True)
    manifest['assets'] = top['assets']
    manifest['radix4TopSourceSlices'] = top['sourceSlices']
    atomic_manifest(destination, manifest)
    for index in (0, 1, 3):
        module, kind, states, bindings, geometry, kernels = build_decoder_layer(Source(index), config,
            manifest['capacity'], prefill_sdpa_fp16=True, fuse_gateup=True, moe_tile=(16, 32, 64),
            flat_q4=True, integer_grouping=True, gdn_prefill_rows=4,
            moe_direct_transfers=True, moe_tail_precision='native-copy')
        kernels += install_nax_moe(module, projections='all', simdgroups=1, down_block=32,
                                   gateup_policy='native-parity-v2')
        kernels += module.moe.custom_kernels()
        named, geo = externalizable_buffers(module, geometry)
        old = next(x for x in original['layers'] if x['index'] == index)
        signature = buffer_signature(named)
        expected = [{k: x[k] for k in ('inputName', 'bufferName', 'dtype', 'shape')}
                    for x in old['weights']['buffers']]
        if signature != expected or bindings != old['stateBindings'] or state_metadata(states) != old['initialState']:
            raise ValueError(f'Layer{index} weight/state contract changed')
        for (name, value), record in zip(named, old['weights']['buffers'], strict=True):
            if hashlib.sha256(memoryview(value.numpy()).cast('B')).hexdigest() != record['sha256']:
                raise ValueError(f'Learned weight bytes differ: layer{index}.{name}')
        examples = {}
        for entry, size in phases.items():
            inputs = {'stream': torch.zeros(1, size, c.width, dtype=torch.float16)}
            if module.ple is not None:
                inputs['ple_embedding'] = torch.zeros(1, size, c.ple_dim, dtype=torch.float16)
            examples[entry] = {**inputs, **states}
        working = manifest.get('qsaWorkingSets', []) if kind == 'qsa' else []
        overrides = qsa_working_set_modules(module, working) if working else {}
        by_count = {size: name for name, size in phases.items()}
        for entry in working:
            examples[entry['function']] = dict(examples[by_count[entry['tokenCount']]])
        key = kind + ('-ple' if module.ple is not None else '')
        asset = export_generic(module, named, geometry, examples, ('stream_out', *bindings.values()),
            base / f'shared-{key}-{label}.aimodel', list(dict.fromkeys(kernels)), entry_modules=overrides)
        asset.update(geometrySignature=geometry_signature(geo), stateBindings=bindings,
            initialState=state_metadata(states), kind=kind, hasPLE=module.ple is not None,
            exampleLayer=index, authoringSourceSHA256=hashes, radix4Tails=True,
            gdnPrefillRows=4 if kind == 'gdn' else None, naxMoEProjections='all', naxDownBlock=32,
            naxGateUpPolicy='native-parity-v2', moeDirectTransfers=True, moeInverseCopy=True,
            moeTailPrecision='native-copy', moeRoutedOutputPrecision='original-native-graph')
        verify_recorded_asset(base, asset)
        manifest['sharedAssets'][key] = asset
        for layer in manifest['layers']:
            if layer['sharedAsset'] != key: continue
            if [{k: x[k] for k in ('inputName', 'bufferName', 'dtype', 'shape')}
                    for x in layer['weights']['buffers']] != signature:
                raise ValueError(f'Layer{layer["index"]} cannot share graph {key}')
            layer.update({name: asset[name] for name in ('path', 'function', 'prefillFunction',
                'inputNames', 'outputNames', 'modelBytes', 'files')})
        manifest['radix4Validation']['representativeWeightSHA256Verified'].append(index)
        atomic_manifest(destination, manifest)
        print(label, key, 'complete', round(time.monotonic() - started, 3), flush=True)
        del module, named, geo, states, examples, overrides
        gc.collect()
    if source_hashes() != hashes:
        raise ValueError('Source changed during authoring; retained assets need provenance review')
    if [x['weights'] for x in manifest['layers']] != [x['weights'] for x in original['layers']]:
        raise AssertionError('Decoder weight records must remain unchanged')
    manifest.update(status='complete', completeModelLayerSet=True,
        radix4AuthoringSeconds=time.monotonic() - started,
        modelBytes=sum(x['modelBytes'] for x in manifest['assets'].values())
            + sum(x['modelBytes'] for x in manifest['sharedAssets'].values())
            + sum(x['weights']['byteLength'] for x in manifest['layers']))
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
