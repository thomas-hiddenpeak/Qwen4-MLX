#!/usr/bin/env python3
"""CPU-author original whole layer versus optional HCWrite scopes, no diagnostics.

Uses real layer0 weights and existing deterministic synthetic input/state files.
No weights are copied. All functions output only stream_out and original states.
Device parity/throughput is pending and no default implementation is changed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from coreai_hc_write_layer import with_direct_hc_write
from coreai_q4_nax import install_nax_moe
from export_coreai_pd_shared import (build_decoder_layer, externalizable_buffers,
    buffer_signature, export_generic, authoring_source_hashes)
from export_moe import Source, sha256_file


def export(base, fixture, output):
    if output.exists(): raise FileExistsError('Use a fresh HCWrite probe directory')
    baseline_path = base / 'manifest-inverse-copy.json'
    baseline = json.loads(baseline_path.read_text())
    layer = next(x for x in baseline['layers'] if x['index'] == 0)
    source = Source(0)
    config_path = source.directory/'config.json'
    if sha256_file(config_path) != baseline['configSHA256']:
        raise ValueError('Model config differs')
    config = json.loads(config_path.read_text())['text_config']
    module, kind, states, bindings, geometry, kernels = build_decoder_layer(source, config,
        baseline['capacity'], prefill_sdpa_fp16=True, fuse_gateup=True, moe_tile=(16,32,64),
        flat_q4=True, integer_grouping=True, gdn_prefill_rows=1,
        moe_direct_transfers=True, moe_tail_precision='native-copy')
    kernels += install_nax_moe(module, projections='all', down_block=32, simdgroups=1,
                              gateup_policy='native-parity-v2')
    kernels += module.moe.custom_kernels()
    named, geo = externalizable_buffers(module, geometry)
    signature = buffer_signature(named)
    if signature != [{k:x[k] for k in ('inputName','bufferName','dtype','shape')}
                     for x in layer['weights']['buffers']]:
        raise ValueError('Weight signature differs')
    for (name,value),record in zip(named,layer['weights']['buffers'],strict=True):
        if hashlib.sha256(memoryview(value.numpy()).cast('B')).hexdigest() != record['sha256']:
            raise ValueError(f'Weight bytes differ: {name}')
    scopes = {'baseline':module}
    for scope in ('attention','moe','both'):
        scopes[scope], extra = with_direct_hc_write(module,scope=scope)
        kernels += extra
    examples = {name:{'stream':torch.zeros(1,2048,config['hidden_size']*config['hc_count'],dtype=torch.float16),**states}
                for name in scopes}
    original_spec = json.loads(fixture.read_text())
    inputs = {}
    for name in ('stream',*states):
        record = dict(original_spec['inputs'][name])
        record['file'] = str((fixture.parent/record['file']).resolve())
        inputs[name] = record
    for record in layer['weights']['buffers']:
        inputs[record['inputName']] = {'file':str((base/layer['weights']['path']).resolve()),
            'offset':record['byteOffset'],'bytes':record['byteLength'],
            'shape':record['shape'],'dtype':record['dtype']}
    source_hashes = authoring_source_hashes()
    for name in ('coreai_hc_write_layer','coreai_hc_write_probe','export_coreai_hc_write_layer'):
        source_hashes[name] = sha256_file(Path(__file__).with_name(name+'.py'))
    output.mkdir(parents=True)
    asset = export_generic(module,named,geometry,examples,('stream_out',*bindings.values()),
        output/'layer.aimodel',list(dict.fromkeys(kernels)),entry_modules=scopes)
    # Generic exporter metadata names main/prefill by convention; this probe
    # intentionally selects the explicit baseline/attention/moe/both entrypoints.
    asset['function'] = 'baseline'
    asset.pop('prefillFunction',None)
    for name in scopes:
        spec = {'asset':'layer.aimodel','function':name,'inputs':inputs,
            'output':'device-'+name,'repeats':12,'mapped':False}
        (output/(name+'-spec.json')).write_text(json.dumps(spec,indent=2)+'\n')
    report = {'status':'cpu-authored-device-unvalidated','deviceValidated':False,
        'asset':asset,'tokens':2048,'policy':'unrounded','cases':list(scopes),
        'inputScope':'Existing deterministic synthetic residual/state files; real layer0 weights',
        'stateBindings':bindings,'weightRecordsAndFilesReused':True,
        'outputContract':'Only original stream_out and state outputs; no intermediate diagnostics',
        'numericalBoundary':'Custom HCWrite may materialize native intermediates as FP16; whole-layer device parity required',
        'sourceHashes':source_hashes,'baselineManifestSHA256':sha256_file(baseline_path)}
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print('READY',output,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,default=Path('results/coreai-prefill-1k/full-s2048-shared-v2'))
    p.add_argument('--fixture',type=Path,default=Path('results/coreai-prefill-1k/external-layer0/swift-flat-owned-per-input-spec.json'))
    p.add_argument('--output',type=Path,default=Path('results/coreai-prefill-1k/whole-hcwrite-s2048'))
    args=p.parse_args();torch.set_num_threads(2);torch.set_num_interop_threads(2)
    export(args.base,args.fixture,args.output)
