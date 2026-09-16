#!/usr/bin/env python3
"""CPU-author real-sized GDN recurrence and full-layer TensorOps candidates.

References are NPZ (input.<name>, expected.<name>) to avoid large JSON states.
The nonzero initial state is produced by two source-weight CPU GDN steps. Longer
activations repeat 24 captured MoE rows at the GDN boundary: explicitly replay,
not a genuine S512 GDN activation capture. No device runtime is called.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from coreai_gdn_chunk import GDNRegisterPrefill,prepare_gdn_inputs
from coreai_gdn_chunk_metal import FusedGDNRecurrence,get_gdn_recurrence_kernel,recurrence_reference
from export_coreai_gdn import GDN,GDNConfig,capture_input
from export_moe import Source,sha256_file,write_json

ROOT=Path(__file__).resolve().parents[1]


def write_json_fixtures(directory, counts=None):
    """Convert bounded NPZ cases to the existing Swift CoreMLBlockFixture form."""
    path=Path(directory)/'manifest.json'
    manifest=json.loads(path.read_text())
    selected=manifest['cases'] if counts is None else [next(c for c in manifest['cases'] if c['tokens']==n) for n in counts]
    for case in selected:
        with np.load(path.parent/case['fixture']) as data:
            def tensor(value):
                return {'shape':list(value.shape),'dtype':str(value.dtype),'values':value.reshape(-1).tolist()}
            fixture={'inputs':{n:tensor(data['input.'+n]) for n in manifest['inputNames']},
                     'expectedOutputs':{n:tensor(data['expected.'+n]) for n in manifest['outputNames']}}
        destination=path.parent/f'actual-s{case["tokens"]}.json'
        write_json(destination,fixture)
        del fixture
        case['jsonFixture']=destination.name
        print(f'JSON ready: {destination}',flush=True)
        write_json(path,manifest)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=('recurrence','layer'),default='recurrence')
    parser.add_argument('--lengths',default='128,256,512')
    parser.add_argument('--json-fixtures',action='store_true')
    args=parser.parse_args()
    lengths=[int(n) for n in args.lengths.split(',')]
    if len(set(lengths))!=len(lengths) or min(lengths)<1: raise ValueError('Unique positive lengths required')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    args.output.mkdir(parents=True,exist_ok=False)
    begin=time.perf_counter()
    source=Source(0)
    config_path=source.directory/'config.json'
    config=json.loads(config_path.read_text())['text_config']
    c=GDNConfig.from_model(config)
    source.prefix='language_model.model.layers.0.linear_attn.'
    weights={name:source.read(name) for name in c.weight_shapes}
    gdn=GDN(c,weights).eval()
    del weights
    captured,evidence=capture_input(ROOT/'fixtures/moe-real/prefill.safetensors',26,c.hidden)
    history=torch.zeros(1,c.kernel-1,c.channels,dtype=torch.float16)
    state=torch.zeros(1,c.value_heads,c.value_dim,c.key_dim,dtype=torch.float32)
    with torch.inference_mode():
        for i in range(2): _,history,state=gdn(captured[:,i:i+1],history,state)
    kernels=[get_gdn_recurrence_kernel()]
    if args.mode=='layer':
        from coreai_tensor_matmul import tensor_linear,get_tensor_kernel
        model=GDNRegisterPrefill(gdn).eval()
        names=('hidden','conv_history','recurrent_state')
        outputs=('output','next_conv_history','next_recurrent_state')
        kernels.append(get_tensor_kernel())
    else:
        model=FusedGDNRecurrence().eval()
        names=('q','k','v','decay','beta','state')
        outputs=('y','next_state')
    report={'version':1,'status':'exporting','mode':args.mode,'model':'gdn.aimodel','layer':0,
        'heads':c.value_heads,'keyDim':c.key_dim,'valueDim':c.value_dim,'inputNames':list(names),'outputNames':list(outputs),
        'modelDirectory':str(source.directory),'configSHA256':sha256_file(config_path),'sourceRecords':source.records,
        'capture':evidence,'inputScope':'Two-row CPU warmup; longer hidden uses repeated remaining24 MoE capture rows at GDN boundary',
        'fixtureFormat':'npz','inputKeyPrefix':'input.','expectedKeyPrefix':'expected.','deviceExecuted':False,
        'tolerances':{'maximumAbsoluteError':0.002,'relativeL2Error':0.002},'cases':[]}
    write_json(args.output/'manifest.json',report)
    import coreai_torch
    converter=coreai_torch.TorchConverter(mode=coreai_torch.TorchConverter.Mode.RELEASE)
    converter.register_custom_kernels(kernels)
    baseline_linear=gdn.linear
    for index,count in enumerate(lengths):
        hidden=captured[:,2:].repeat(1,(count+23)//24,1)[:,:count].contiguous()
        function='main' if index==0 else f's{count}'
        with torch.inference_mode():
            if args.mode=='layer':
                gdn.linear=baseline_linear
                baseline=gdn(hidden,history,state)
                # Replace only authoring instance, never global/main exporter.
                gdn.linear=tensor_linear
                inputs=(hidden,history,state)
                actual=model(*inputs)
                for a,b in zip(actual,baseline): torch.testing.assert_close(a,b,rtol=0,atol=0)
                expected=baseline
            else:
                q,k,v,decay,beta,_,_=prepare_gdn_inputs(gdn,hidden,history)
                inputs=(q,k,v,decay,beta,state)
                expected=recurrence_reference(*inputs)
        fixture=args.output/f'actual-s{count}.npz'
        np.savez(fixture,**{**{'input.'+n:v.numpy() for n,v in zip(names,inputs)},
                            **{'expected.'+n:v.numpy() for n,v in zip(outputs,expected)}})
        converter.add_pytorch_module(model,entrypoint_name=function,input_names=names,output_names=outputs,
            export_fn=lambda m,inputs=inputs:torch.export.export(m,args=inputs).run_decompositions(coreai_torch.get_decomp_table()))
        report['cases'].append({'tokens':count,'function':function,'fixture':fixture.name,'sha256':sha256_file(fixture),
            'inputs':{n:{'shape':list(v.shape),'dtype':str(v.dtype).removeprefix('torch.')} for n,v in zip(names,inputs)},
            'outputs':{n:{'shape':list(v.shape),'dtype':str(v.dtype).removeprefix('torch.')} for n,v in zip(outputs,expected)}})
        print(f'Authored {args.mode} S{count}',flush=True)
    program=converter.to_coreai()
    program.optimize()
    asset=args.output/report['model']
    program.save_asset(asset)
    report['assets']=[{'path':str(p.relative_to(args.output)),'bytes':p.stat().st_size,'sha256':sha256_file(p)}
                      for p in sorted(asset.rglob('*')) if p.is_file()]
    report.update(status='complete',seconds=time.perf_counter()-begin)
    write_json(args.output/'manifest.json',report)
    if args.json_fixtures: write_json_fixtures(args.output)
    print(f'Complete {args.output}: {report["seconds"]:.2f}s',flush=True)


if __name__=='__main__': main()
