#!/usr/bin/env python3
"""CPU-author paired full-capacity / working-view external QSA probes.

Reuse the existing v2 layer weight file without rewriting learned bytes. The
real-geometry fixtures replay captured layer0 boundary activations at layer3;
they are not full-model layer3 captures. Nonzero offset uses the exact valid
layer-local history obtained from zero inputs, not an arbitrary populated state.
Tiny nonzero/random histories are covered by test_coreai_qsa_working_set.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from coreai_qsa_chunk import CompactProbe, PROBE_OUTPUTS, QwenQSAChunk, repeat_activations
from coreai_qsa_working_set import QwenQSAWorkingSet, working_set_function
from export_coreai_pd_shared import ExternalModule
from export_coreai_qsa import FIXTURES, INPUT_NAMES, QwenQSA, initial_state, read_fixture
from check_coreai_qsa_chunks import WEIGHT_NAMES
from export_moe import sha256_file


def write_input(output,name,value):
    filename=f'{name}.bin'
    array=value.detach().contiguous().numpy()
    array.tofile(output/filename)
    return {'file':filename,'offset':0,'bytes':array.nbytes,'shape':list(array.shape),'dtype':str(array.dtype)}


def export(manifest_path,output,count,limits,offsets):
    import coreai_torch
    from coreai_torch.composite_ops import SDPA
    manifest_path=manifest_path.resolve()
    manifest=json.loads(manifest_path.read_text())
    if manifest['version']!=2 or manifest['status']!='complete':raise ValueError('Complete shared v2 manifest required')
    config_path=Path(manifest['modelDirectory'])/'config.json'
    if sha256_file(config_path)!=manifest['configSHA256']:raise ValueError('Source model identity differs')
    config=json.loads(config_path.read_text())['text_config']
    layer=next(item for item in manifest['layers'] if item['index']==3)
    if layer['kind']!='qsa':raise ValueError('Layer3 must be QSA')
    capacity=manifest['capacity']
    if count<=1 or any(not count<=limit<capacity or limit%4 for limit in limits):
        raise ValueError('Require count>1 and covering4-aligned limits below capacity')
    if any(offset<0 or offset+count>capacity for offset in offsets):raise ValueError('Invalid fixture offset')
    output.mkdir(parents=True,exist_ok=False)
    records=[record for record in layer['weights']['buffers'] if record['bufferName'].startswith('attention.source.')]
    by_name={record['bufferName'].removeprefix('attention.source.'):record for record in records}
    if set(by_name)!={name.replace('.','_') for name in WEIGHT_NAMES}:raise ValueError('Unexpected QSA weight set')
    # Real values are external inputs. Zero placeholders avoid reading/copying
    # any large learned bank merely to export shapes.
    weights={name:np.zeros(by_name[name.replace('.','_')]['shape'],dtype=np.float16) for name in WEIGHT_NAMES}
    source=QwenQSA(config,weights,capacity)
    del weights
    baseline=QwenQSAChunk(source,prefill_sdpa_fp16=True)
    named=['source.source.'+record['bufferName'].removeprefix('attention.source.') for record in records]
    weight_values=tuple(getattr(source,record['bufferName'].removeprefix('attention.source.')) for record in records)
    weight_names=tuple(record['inputName'] for record in records)
    geometry={'base.source.source.'+name for name in ('cache_positions','block_positions','cosine','sine')}
    converter=coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    converter.register_custom_kernels(baseline.custom_kernels())
    variants={'main':(baseline,1),'prefill_s'+str(count):(baseline,count)}
    for limit in limits:
        variants[working_set_function(count,limit)]=(QwenQSAWorkingSet(source,limit),count)
    stats={}
    for function,(module,length) in variants.items():
        values={'x':torch.zeros(1,length,source.hidden,dtype=torch.float16),**initial_state(source)}
        arguments=(*[values[name] for name in INPUT_NAMES],*weight_values)
        wrapper=ExternalModule(CompactProbe(module),named,len(INPUT_NAMES)).eval()
        def export_fn(current,arguments=arguments,function=function):
            ep=torch.export.export(current,args=arguments).run_decompositions(coreai_torch.get_decomp_table())
            placeholders=[node for node in ep.graph.nodes if node.op=='placeholder']
            captured=[spec.target for spec,node in zip(ep.graph_signature.input_specs,placeholders)
                      if str(spec.kind).endswith('BUFFER') and len(node.users)]
            if set(captured)-geometry:raise ValueError(f'Unexpected captured learned buffers: {captured}')
            stats[function]={'capturedBuffers':captured}
            return ep
        converter.add_pytorch_module(wrapper,entrypoint_name=function,
            input_names=(*INPUT_NAMES,*weight_names),output_names=PROBE_OUTPUTS,
            externalize_modules=[coreai_torch.ExternalizeSpec(target_class=SDPA,
                composite_op_name='scaled_dot_product_attention',composite_attrs=['scale','is_causal','window_size'])],
            export_fn=export_fn)
    program=converter.to_coreai();program.optimize();program._mlir_module.operation.verify()
    path=output/'qsa-working-set.aimodel';program.save_asset(path)
    for function in variants:
        graph=str(program.get_graph(function))
        (output/f'{function}-after.txt').write_text(graph)
        stats[function]['argsortCount']=graph.count('coreai.argsort ')
        stats[function]['sdpaCalls']=[line.strip() for line in graph.splitlines()
                                      if 'coreai.invoke ' in line and '.sdpa_' in line]
    captured,evidence=read_fixture(FIXTURES/'attention-continuous-0-prefill.safetensors',
        json.loads((FIXTURES/'manifest.json').read_text()))
    rows=captured['input'].half()
    weight_file=manifest_path.parent/'weights'/layer['weights']['path']
    if not weight_file.exists():
        weight_file=manifest_path.parent/layer['weights']['path']
    if not weight_file.exists():raise FileNotFoundError(weight_file)
    input_weights={record['inputName']:{'file':str(weight_file),'offset':record['byteOffset'],
        'bytes':record['byteLength'],'shape':record['shape'],'dtype':record['dtype']} for record in records}
    specs=[]
    # Reuse one full-capacity zero state for all offsets. Zero source activations
    # produce zero K/V/raw/pooled states with the source bias-free projections.
    empty=initial_state(source)
    state_inputs={name:write_input(output,name,value) for name,value in empty.items() if name not in ('offset','pooled_count')}
    for offset in offsets:
        values={**state_inputs,
            'x':write_input(output,f'x-o{offset}',repeat_activations(rows,count,offset)),
            'offset':write_input(output,f'offset-o{offset}',torch.tensor([offset],dtype=torch.int32)),
            'pooled_count':write_input(output,f'pooled-count-o{offset}',torch.tensor([offset//4],dtype=torch.int32)),
            **input_weights}
        for function,(_,length) in variants.items():
            if length!=count:continue
            limit=capacity if function=='prefill_s'+str(count) else int(function.rsplit('kv',1)[1])
            if offset+count>limit:continue
            label=f'{function}-o{offset}'
            spec={'asset':path.name,'function':function,'inputs':values,
                'output':label+'-output','repeats':15,'mapped':False}
            (output/(label+'-spec.json')).write_text(json.dumps(spec,indent=2)+'\n')
            specs.append({'path':label+'-spec.json','function':function,'offset':offset,'tokens':count,'kvLimit':limit})
    report={'status':'cpu-authored-device-unvalidated','deviceValidated':False,'model':path.name,
        'weightBytesRewritten':0,'capacity':capacity,'tokenCount':count,
        'qsaWorkingSets':[{'tokenCount':count,'kvLimit':limit,'function':working_set_function(count,limit)} for limit in limits],
        'functionAudit':stats,'specs':specs,'manifestSource':str(manifest_path),
        'configSHA256':manifest['configSHA256'],'sourceManifestSHA256':sha256_file(manifest_path),
        'authoringSourceSHA256':{p.name:sha256_file(p) for p in (Path(__file__),Path(__file__).with_name('coreai_qsa_working_set.py'))},
        'fixtureProvenance':{'activations':'Verified layer0 boundary capture repeated at layer3; not full-model layer3 input.',
            'sourceFile':str(FIXTURES/'attention-continuous-0-prefill.safetensors'),'sourceEvidence':evidence,
            'history':'Exact layer-local state for offset zero inputs; nonzero random history covered by CPU tests.'},
        'limitations':['Device numerical comparison and performance remain unvalidated.',
            'Static prefix cropping may produce layout copies; count of physical GPU copies is not known.',
            'All six state tensor shapes remain full capacity; no in-place aliasing or state-copy savings are claimed.']}
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--count',type=int,default=2048)
    parser.add_argument('--limits',type=int,nargs='+',default=[2048,10240])
    parser.add_argument('--offsets',type=int,nargs='+',default=[0,8192])
    args=parser.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    result=export(args.manifest,args.output,args.count,args.limits,args.offsets)
    print(json.dumps({key:result[key] for key in ('status','model','qsaWorkingSets','specs')},indent=2))
