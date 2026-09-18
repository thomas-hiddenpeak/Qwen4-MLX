#!/usr/bin/env python3
"""CPU-author one S8192 GDN layer probe, without a full-model overlay.

The new graph has exactly one entrypoint. It uses native-copy I/O, NAX parity-v2
gate/up with BM32 down, and repaired ILP4 readout-v2. Original layer0 weight bytes
are explicit inputs reused from the full shared export. The old8K v1 comparison
has different tail/recurrence policy and is a performance/memory control only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import torch

from coreai_moe_transfers import install_moe_transfers
from coreai_q4_nax import install_nax_moe
from export_coreai_dense import DenseConfig
from export_coreai_hybrid import state_metadata
from export_coreai_optimized_chunks import source_hashes
from export_coreai_pd_shared import build_decoder_layer, buffer_signature, export_generic, externalizable_buffers
from export_moe import Source, sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base', type=Path, default=ROOT/'results/coreai-prefill-1k/full-s2048-shared-v2')
    parser.add_argument('--fixture', type=Path, default=ROOT/'results/coreai-prefill-1k/external-layer0/swift-flat-owned-per-input-spec.json')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    base, output, fixture = args.base.resolve(), args.output.resolve(), args.fixture.resolve()
    output.mkdir(parents=True, exist_ok=False)
    previous = base/'manifest-optimized-v1-s8192.json'
    manifest = json.loads(previous.read_text())
    old = next(layer for layer in manifest['layers'] if layer['index'] == 0)
    if manifest['capacity'] != 16384 or manifest['tokenChunk'] != 8192 or old['kind'] != 'gdn' or old['hasPLE']:
        raise ValueError('Expected completed old8K non-PLE GDN layer0 baseline')
    old_input = json.loads(fixture.read_text())['inputs']
    def full_path(record):
        path = Path(record['file'])
        return path if path.is_absolute() else (fixture.parent/path).resolve()
    stream_record = old_input['stream']
    if stream_record['shape'] != [1, 2048, 10240] or stream_record['offset'] != 0:
        raise ValueError('Expected existing S2048 synthetic stream fixture')
    stream_source = full_path(stream_record)
    if stream_source.stat().st_size != stream_record['bytes']:
        raise ValueError('Source stream file size differs')
    stream_path = output/'stream-s8192.bin'
    with stream_path.open('xb') as target:
        for _ in range(4):
            with stream_source.open('rb') as source:
                shutil.copyfileobj(source, target, length=1024*1024)
    inputs = {'stream': {'file': str(stream_path), 'offset': 0, 'bytes': 8192*10240*2,
                         'shape': [1, 8192, 10240], 'dtype': 'float16'}}
    for name in old['stateBindings']:
        record = dict(old_input[name])
        record['file'] = str(full_path(record))
        inputs[name] = record
    weights = base/old['weights']['path']
    if weights.stat().st_size != old['weights']['byteLength']:
        raise ValueError('Original layer0 weight file length differs')
    for record in old['weights']['buffers']:
        inputs[record['inputName']] = {'file': str(weights), 'offset': record['byteOffset'],
            'bytes': record['byteLength'], 'shape': record['shape'], 'dtype': record['dtype']}
    hashes = source_hashes()
    hashes.update({name: sha256_file(Path(__file__).with_name(name+'.py'))
                   for name in ('coreai_gdn_ilp_readout_probe', 'export_coreai_s8192_layer_probe')})
    started = time.perf_counter()
    source = Source(0)
    config_path = source.directory/'config.json'
    if str(source.directory) != manifest['modelDirectory'] or sha256_file(config_path) != manifest['configSHA256']:
        raise ValueError('Source model/config differs from old asset')
    config = json.loads(config_path.read_text())['text_config']
    c = DenseConfig.from_model(config)
    module, kind, states, bindings, geometry, kernels = build_decoder_layer(source, config, 16384,
        prefill_sdpa_fp16=True, fuse_gateup=True, moe_tile=(16, 32, 64), flat_q4=True,
        integer_grouping=True, gdn_prefill_rows=4, gdn_prefill_policy='readout-v2')
    kernels += install_moe_transfers(module, tail_precision='native-copy')
    kernels += install_nax_moe(module, projections='all', down_block=32, gateup_policy='native-parity-v2')
    kernels += module.moe.custom_kernels()
    named, _ = externalizable_buffers(module, geometry)
    signature = buffer_signature(named)
    expected = [{key: item[key] for key in ('inputName', 'bufferName', 'dtype', 'shape')}
                for item in old['weights']['buffers']]
    if signature != expected or state_metadata(states) != old['initialState'] or bindings != old['stateBindings']:
        raise ValueError('Learned buffer/state signature changed')
    for (_, value), record in zip(named, old['weights']['buffers'], strict=True):
        if hashlib.sha256(memoryview(value.numpy()).cast('B')).hexdigest() != record['sha256']:
            raise ValueError('Source learned tensor differs from existing original layer0 bytes')
    asset = export_generic(module, named, geometry,
        {'prefill': {'stream': torch.zeros(1, 8192, c.width, dtype=torch.float16), **states}},
        ('stream_out', *bindings.values()), output/'layer0-s8192.aimodel', list(dict.fromkeys(kernels)))
    # This is a one-function probe, not a complete runtime phase asset.
    asset['function'] = 'prefill'
    if set(asset['torchExport']) != {'prefill'}:
        raise ValueError('Probe must contain only the bounded S8192 entrypoint')
    for label, path in (('baseline-old8192', base/old['path']), ('candidate', output/'layer0-s8192.aimodel')):
        write_json(output/(label+'-spec.json'), {'asset': str(path), 'function': 'prefill',
            'inputs': inputs, 'output': 'device-'+label, 'repeats': 10,
            'mapped': False, 'oneBufferPerFile': False})
    current_hashes = source_hashes()
    current_hashes.update({name: sha256_file(Path(__file__).with_name(name+'.py'))
                           for name in ('coreai_gdn_ilp_readout_probe', 'export_coreai_s8192_layer_probe')})
    if hashes != current_hashes:
        raise RuntimeError('Authoring source changed during single-layer export')
    report = {'status': 'CPU-authored-device-unvalidated', 'completeModelLayerSet': False,
        'layer': 0, 'tokens': 8192, 'capacity': 16384, 'asset': asset,
        'sourceManifest': str(previous), 'sourceManifestSHA256': sha256_file(previous),
        'sourceSHA256': hashes, 'weightRecords': old['weights'],
        'learnedWeightBytesCopied': 0, 'learnedWeightTensorSHA256Verified': True,
        'weightBytes': sum(r['byteLength'] for r in old['weights']['buffers']),
        'stateBindings': bindings, 'initialState': state_metadata(states),
        'newPolicy': {'moeTail': 'native-copy', 'naxProjections': 'all', 'downBlock': 32,
                      'gateUpPolicy': 'native-parity-v2', 'gdnPrefillRows': 4, 'gdnPrefillPolicy': 'readout-v2'},
        'comparisonBoundary': 'Old8K v1 uses treeFP32 tail, NAX downBM16 and experimental ILP4. Its output is not a quality oracle for the repaired candidate; compare speed/physical footprint, and later validate candidate against ILP1/native-copy same-geometry control.',
        'inputProvenance': 'Existing S2048 synthetic residual stream repeated4times, existing conv/state fixtures; not a real model activation capture',
        'inputSourceSHA256': sha256_file(stream_source), 'streamSHA256': sha256_file(stream_path),
        'outputTensorBytes': 8192*c.width*2 + sum(v.numel()*v.element_size() for v in states.values()),
        'prospectiveBoundedFamily': {'counts': [1, 4, 16, 32, 256, 512, 2048, 8192],
            'qsaWorkingSets': [[8192, 8192], [2048, 10240], [2048, 12288], [2048, 14336]],
            'fullModelAuthoringApproved': False},
        'arenaCaution': 'Single function candidate only; external weights retained per tensor. This is not a full-model peak estimate. Compare in independent processes; no concurrent old/new asset load.',
        'authorSeconds': time.perf_counter()-started}
    write_json(output/'manifest.json', report)
    print('READY '+str(output), flush=True)


if __name__ == '__main__':
    main()
