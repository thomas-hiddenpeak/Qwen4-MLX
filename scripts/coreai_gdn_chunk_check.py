#!/usr/bin/env python3
"""Real-weight CPU checks for optional GDN register/WY prefill paths."""
import argparse
import json
from pathlib import Path
import time

import torch

from coreai_gdn_chunk import GDNRegisterPrefill, prepare_gdn_inputs, finish_gdn, wy_recurrence
from coreai_gdn_chunk_metal import recurrence_reference
from export_coreai_gdn import GDN,GDNConfig,capture_input,OUTPUT_NAMES
from export_coreai_moe import errors
from export_moe import Source,write_json

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/coreai-prefill-1k/gdn-cpu.json')
    parser.add_argument('--lengths',default='128,256,512')
    parser.add_argument('--layer',type=int,default=0)
    args=parser.parse_args()
    lengths=[int(n) for n in args.lengths.split(',')]
    if any(n<1 or n%64 for n in lengths): raise ValueError('Lengths must be positive multiples of64')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    source=Source(args.layer)
    config=json.loads((source.directory/'config.json').read_text())['text_config']
    c=GDNConfig.from_model(config)
    source.prefix=f'language_model.model.layers.{args.layer}.linear_attn.'
    weights={name:source.read(name) for name in c.weight_shapes}
    model=GDN(c,weights).eval()
    del weights
    wrapper=GDNRegisterPrefill(model).eval()
    captured,provenance=capture_input(ROOT/'fixtures/moe-real/prefill.safetensors',26,c.hidden)
    history=torch.zeros(1,c.kernel-1,c.channels,dtype=torch.float16)
    state=torch.zeros(1,c.value_heads,c.value_dim,c.key_dim,dtype=torch.float32)
    report={'version':1,'deviceExecuted':False,'layer':args.layer,'sourceRecords':source.records,'capture':provenance,
        'inputScope':'Two captured MoE rows warm nonzero GDN state; repeat remaining24 rows for synthetic S128/256/512 GDN replay',
        'tolerances':{'relativeL2Error':0.002,'maximumAbsoluteError':0.002},'checks':[]}
    start=time.perf_counter()
    with torch.inference_mode():
        for position in range(2): _,history,state=model(captured[:,position:position+1],history,state)
        for count in lengths:
            hidden=captured[:,2:].repeat(1,(count+23)//24,1)[:,:count].contiguous()
            baseline=model(hidden,history,state)
            candidate=wrapper(hidden,history,state)
            for actual,expected in zip(candidate,baseline): torch.testing.assert_close(actual,expected,atol=0,rtol=0)
            q,k,v,decay,beta,z,next_history=prepare_gdn_inputs(model,hidden,history)
            reference=recurrence_reference(q,k,v,decay,beta,state)
            row={'tokens':count,'registerWrapperCPUExact':True,'decayMinimum':float(decay.min()),
                 'decayZeroCount':int(torch.count_nonzero(decay==0)),'wy':[]}
            for solver in ('triangular','block_inverse'):
                for block in (64,128):
                    t=time.perf_counter()
                    y,next_state=wy_recurrence(q,k,v,decay,beta,state,block_size=block,solver=solver)
                    comparisons={name:errors(a.float().numpy(),b.float().numpy()) for name,a,b in
                        [('y',y,reference[0]),('state',next_state,reference[1]),
                         ('output',finish_gdn(model,y,z),baseline[0])]}
                    passed=all(x['max_abs']<=.002 and (x['relative_l2'] or 0)<=.002 for x in comparisons.values())
                    row['wy'].append({'solver':solver,'block':block,'passed':passed,'cpuSeconds':time.perf_counter()-t,
                                      'comparisons':comparisons})
            report['checks'].append(row)
            print(json.dumps(row),flush=True)
    report['seconds']=time.perf_counter()-start
    report['passed']=all(item['passed'] for row in report['checks'] for item in row['wy'])
    write_json(args.output,report)
    if not report['passed']: raise AssertionError('WY predeclared tolerance failed; inspect report without widening')


if __name__=='__main__': main()
