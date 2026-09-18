#!/usr/bin/env python3
"""CPU-author only the two GDN graphs of a radix-4 overlay with original ILP1.

QSA, top assets, every learned weight file, integer grouping, NAX and native
copy transfers remain unchanged. The two primary manifests share the same new
GDN graphs. Device quality/performance validation is deliberately separate.
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
from export_coreai_pd import verify_recorded_asset
from export_coreai_pd_shared import (buffer_signature, build_decoder_layer,
    export_generic, externalizable_buffers, geometry_signature)
from export_coreai_radix4_tails import (for_primary, phase_contract,
    source_hashes as radix4_source_hashes)
from export_moe import Source, sha256_file


def source_hashes():
    result = radix4_source_hashes()
    result['export_coreai_radix4_ilp1'] = sha256_file(Path(__file__))
    return result


def export(base):
    base = base.resolve()
    source_manifest = base / 'manifest-radix4-v2-native-tail-s4096.json'
    original = json.loads(source_manifest.read_text())
    phases = phase_contract(original)
    if not original.get('radix4Tails') or not {48, 192, 768}.issubset(original['tailChunks']):
        raise ValueError('Expected the complete radix-4 ILP4 source overlay')
    label = 'radix4-ilp1-v2-native-tail'
    destination = base / f'manifest-{label}-s4096.json'
    paths = [destination, base / f'manifest-{label}-s2048.json',
        *[base / f'shared-{kind}-{label}.aimodel' for kind in ('gdn', 'gdn-ple')]]
    if any(path.exists() for path in paths):
        raise FileExistsError('Refusing to overwrite an existing ILP1 overlay')
    # Verify the unchanged assets instead of copying/re-exporting them.
    for asset in [*original['assets'].values(), original['sharedAssets']['qsa']]:
        verify_recorded_asset(base, asset)
    config_path = Source(0).directory / 'config.json'
    if str(config_path.parent) != original['modelDirectory'] or sha256_file(config_path) != original['configSHA256']:
        raise ValueError('Source config/model directory differs')
    config = json.loads(config_path.read_text())['text_config']
    c = DenseConfig.from_model(config)
    hashes, started = source_hashes(), time.monotonic()
    manifest = deepcopy(original)
    manifest.update(status='exporting', completeModelLayerSet=False, gdnPrefillRows=1,
        gdnPrefillNumerics='Original one-value-row SIMD recurrence; token order and S1 unchanged',
        authoringSourceSHA256=hashes, radix4ILP1SourceManifest=source_manifest.name,
        radix4ILP1SourceManifestSHA256=sha256_file(source_manifest),
        radix4ILP1Validation={'deviceValidated': False, 'qualityAccepted': False,
            'weightRecordsAndFilesReused': True, 'qsaAndTopAssetsUnchanged': True,
            'fullWeightFilesRehashed': False, 'representativeWeightSHA256Verified': []})
    atomic_manifest(destination, manifest)
    for index in (0, 1):
        module, kind, states, bindings, geometry, kernels = build_decoder_layer(Source(index), config,
            manifest['capacity'], prefill_sdpa_fp16=True, fuse_gateup=True, moe_tile=(16, 32, 64),
            flat_q4=True, integer_grouping=True, gdn_prefill_rows=1,
            moe_direct_transfers=True, moe_tail_precision='native-copy')
        kernels += install_nax_moe(module, projections='all', simdgroups=1, down_block=32,
                                   gateup_policy='native-parity-v2')
        kernels += module.moe.custom_kernels()
        named, geo = externalizable_buffers(module, geometry)
        old = next(x for x in original['layers'] if x['index'] == index)
        signature = buffer_signature(named)
        expected = [{k: x[k] for k in ('inputName', 'bufferName', 'dtype', 'shape')}
                    for x in old['weights']['buffers']]
        if kind != 'gdn' or signature != expected or bindings != old['stateBindings'] or state_metadata(states) != old['initialState']:
            raise ValueError(f'Layer{index} GDN weight/state contract changed')
        for (name, value), record in zip(named, old['weights']['buffers'], strict=True):
            if hashlib.sha256(memoryview(value.numpy()).cast('B')).hexdigest() != record['sha256']:
                raise ValueError(f'Learned weight bytes differ: layer{index}.{name}')
        examples = {}
        for entry, size in phases.items():
            inputs = {'stream': torch.zeros(1, size, c.width, dtype=torch.float16)}
            if module.ple is not None:
                inputs['ple_embedding'] = torch.zeros(1, size, c.ple_dim, dtype=torch.float16)
            examples[entry] = {**inputs, **states}
        key = kind + ('-ple' if module.ple is not None else '')
        asset = export_generic(module, named, geometry, examples, ('stream_out', *bindings.values()),
            base / f'shared-{key}-{label}.aimodel', list(dict.fromkeys(kernels)))
        asset.update(geometrySignature=geometry_signature(geo), stateBindings=bindings,
            initialState=state_metadata(states), kind=kind, hasPLE=module.ple is not None,
            exampleLayer=index, authoringSourceSHA256=hashes, radix4Tails=True,
            gdnPrefillRows=1, naxMoEProjections='all', naxDownBlock=32,
            naxGateUpPolicy='native-parity-v2', moeDirectTransfers=True, moeInverseCopy=True,
            moeTailPrecision='native-copy', moeRoutedOutputPrecision='original-native-graph')
        verify_recorded_asset(base, asset)
        manifest['sharedAssets'][key] = asset
        for layer in manifest['layers']:
            if layer['sharedAsset'] != key: continue
            if ([{k: x[k] for k in ('inputName', 'bufferName', 'dtype', 'shape')}
                    for x in layer['weights']['buffers']] != signature
                    or layer['stateBindings'] != bindings or layer['initialState'] != state_metadata(states)):
                raise ValueError(f'Layer{layer["index"]} cannot share graph {key}')
            layer.update({name: asset[name] for name in ('path', 'function', 'prefillFunction',
                'inputNames', 'outputNames', 'modelBytes', 'files')})
        manifest['radix4ILP1Validation']['representativeWeightSHA256Verified'].append(index)
        atomic_manifest(destination, manifest)
        print(label, key, 'complete', round(time.monotonic() - started, 3), flush=True)
        del module, named, geo, states, examples
        gc.collect()
    if source_hashes() != hashes:
        raise ValueError('Source changed during authoring; retained assets need provenance review')
    if ([x['weights'] for x in manifest['layers']] != [x['weights'] for x in original['layers']]
            or manifest['assets'] != original['assets']
            or manifest['sharedAssets']['qsa'] != original['sharedAssets']['qsa']
            or manifest['qsaWorkingSets'] != original['qsaWorkingSets']):
        raise AssertionError('ILP1 overlay must preserve all weights and QSA/top assets')
    manifest.update(status='complete', completeModelLayerSet=True,
        radix4ILP1AuthoringSeconds=time.monotonic() - started,
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
