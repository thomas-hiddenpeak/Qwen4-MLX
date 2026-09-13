#!/usr/bin/env python3
"""One controller-owned HTTP child and client; never touch the reference service."""
import argparse
import hashlib
import http.client
import json
import signal
import subprocess
import sys
import time
from pathlib import Path


def http_get(port, path, timeout=3):
    c = http.client.HTTPConnection('127.0.0.1', port, timeout=timeout)
    try:
        c.request('GET', path)
        r = c.getresponse(); raw = r.read(2*1024*1024+1)
        if len(raw)>2*1024*1024: raise ValueError('Monitoring response exceeds2MiB')
        return r.status, raw
    finally:
        c.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('runner','model-dir','fixture','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--port',type=int,default=11249)
    p.add_argument('--prefill-attention',choices=('reference','fusedQSA'),required=True)
    p.add_argument('--paged-kv-pool-library',type=Path)
    p.add_argument('--decode-fixture',type=Path)
    a=p.parse_args()
    if not 1024<=a.port<=65535 or a.port==11235: p.error('Use a separate non-11235 test port')
    out=a.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    fixture=json.loads((a.fixture/'fixture.json').read_text())
    context=fixture.get('context_limit')
    if fixture.get('complete') is not True or context not in (32768,65536,131072,262144): p.error('Prepared bounded fixture required')
    runner=a.runner.resolve(); script=Path(__file__).with_name('probe_http_long_context.py')
    if hashlib.sha256(runner.read_bytes()).hexdigest()!=fixture.get('runner_sha256'): p.error('Freeze one runner binary for fixture preparation and serving')
    if a.model_dir.resolve().name!=fixture.get('model_id'): p.error('Fixture and service model identifiers differ')
    for name,expected in fixture.get('model_files',{}).items():
        if name not in ('config.json','tokenizer.json','chat_template.jinja'): p.error('Unexpected model provenance filename')
        if hashlib.sha256((a.model_dir.resolve()/name).read_bytes()).hexdigest()!=expected: p.error('Fixture/service model metadata mismatch: '+name)
    if set(fixture.get('model_files',{}))!={'config.json','tokenizer.json','chat_template.jinja'}: p.error('Missing model metadata provenance')
    decode_evidence = None
    if a.decode_fixture is not None:
        from long_context_decode_fixture import load_decode_fixture
        decode_evidence = load_decode_fixture(a.decode_fixture.resolve(), a.fixture.resolve(), fixture, runner)['evidence']
    command=[str(runner),'serve-gpu','--model-dir',str(a.model_dir.resolve()),'--port',str(a.port),
        '--context-limit',str(context),'--max-reserved-tokens',str(context),'--max-resident-sequences','1',
        '--max-body-bytes','8388608','--connection-deadline-seconds','3600','--state-budget-bytes','25769803776',
        '--prefix-cache-bytes','8589934592','--prefix-cache-entries','2',
        '--prefill-attention',a.prefill_attention]
    if a.paged_kv_pool_library:
        command+=['--paged-kv-pool-library',str(a.paged_kv_pool_library.resolve()),'--paged-kv-pages-per-layer','512']
    report={'schema':'qwen-long-http-owned-server-v1','complete':False,'passed':False,'command':command,
        'runner_sha256':hashlib.sha256(runner.read_bytes()).hexdigest(),'context_limit':context,'port':a.port,
        'model_directory':str(a.model_dir.resolve()),'model_files':fixture['model_files'],
        'wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'validator_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),
        'fixture_sha256':hashlib.sha256((a.fixture/'fixture.json').read_bytes()).hexdigest(),
        'pool_library_sha256':hashlib.sha256(a.paged_kv_pool_library.read_bytes()).hexdigest() if a.paged_kv_pool_library else None,
        'notes':['Outer experiment controller owns reference-service pause/resume.',
                 'Only this wrapper child and validator child are terminated; one40s cleanup deadline.']}
    report['extended_decode_fixture'] = decode_evidence
    report['decode_fixture_helper_sha256'] = hashlib.sha256(Path(__file__).with_name('long_context_decode_fixture.py').read_bytes()).hexdigest()
    # One outer case budget includes readiness and reserves the same40s cleanup.
    case_deadline=time.monotonic()+4800
    report['timeouts_seconds']={'connection':3600,'client_request':3600,'case':4800,'ready':240,'cleanup':40}
    child=client=None
    server_log=(out/'server.log').open('x')
    def save():
        (out/'owned-server.json').write_text(json.dumps(report,indent=2)+'\n')
    def interrupted(signum,frame): raise InterruptedError('Controller interrupted wrapper')
    old={s:signal.signal(s,interrupted) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        child=subprocess.Popen(command,stdout=server_log,stderr=subprocess.STDOUT)
        report['server_pid']=child.pid; save()
        deadline=min(time.monotonic()+240,case_deadline-40)
        while time.monotonic()<deadline:
            if child.poll() is not None: raise RuntimeError('Owned server exited during load')
            try:
                status,raw=http_get(a.port,'/health')
                h=json.loads(raw)
                if h.get('pid')!=child.pid: raise RuntimeError('Port belongs to another process')
                if status==200 and h.get('ready') is True:
                    (out/'initial.health.json').write_bytes(raw); break
                if h.get('status')=='failed': raise RuntimeError('Server reported failed load')
            except (OSError,http.client.HTTPException): pass
            time.sleep(.2)
        else: raise TimeoutError('Server did not become ready')
        print(json.dumps({'event':'owned_server_ready','pid':child.pid,'port':a.port,'context_limit':context}),flush=True)
        client_command=[sys.executable,'-B',str(script),'run','--runner',str(runner),'--fixture',str(a.fixture.resolve()),
            '--server-pid',str(child.pid),'--server-log',str(out/'server.log'),'--port',str(a.port),
            '--prefill-attention',a.prefill_attention,'--paged-pages','512' if a.paged_kv_pool_library else '0',
            '--output',str(out/'client'),'--timeout','3600']
        if a.decode_fixture is not None:
            client_command += ['--decode-fixture',str(a.decode_fixture.resolve())]
        report['client_command']=client_command; save()
        # Inherit controller capture so progress events remain visible. The
        # validator separately records identical events under client/.
        client=subprocess.Popen(client_command)
        report['client_pid']=client.pid; save()
        report['client_exit_code']=client.wait(timeout=max(0,case_deadline-time.monotonic()-40))
        if report['client_exit_code']!=0: raise RuntimeError('HTTP validation failed')
        report['client_passed']=True
    except BaseException as error:
        report['error']=type(error).__name__+': '+str(error)
    finally:
        for s in old: signal.signal(s,signal.SIG_IGN)
        end=time.monotonic()+40
        try:
            if client is not None and client.poll() is None: client.terminate()
            if child is not None and child.poll() is None:
                for path,name in (('/health','final.health.json'),('/metrics','final.prometheus.txt')):
                    try:
                        status,raw=http_get(a.port,path,timeout=min(3,max(.1,end-time.monotonic())))
                        if path=='/health' and json.loads(raw).get('pid')!=child.pid: raise RuntimeError('Final health PID changed')
                        (out/name).write_bytes(raw); report[name+'_status']=status
                    except Exception as error: report[name+'_error']=str(error)
                child.terminate()
            for name,process in (('server',child),('client',client)):
                if process is None: continue
                try: process.wait(timeout=max(0,end-time.monotonic()-3))
                except subprocess.TimeoutExpired:
                    process.kill(); report[name+'_forced_kill']=True
                    try: process.wait(timeout=max(0,end-time.monotonic()))
                    except subprocess.TimeoutExpired: report[name+'_unreaped_at_deadline']=True
                report[name+'_exit_code']=process.poll()
        finally:
            server_log.close()
            report['complete']=True
            report['passed']=(report.get('client_passed') is True and report.get('server_exit_code')==0
                              and not report.get('server_forced_kill') and not report.get('error')
                              and report.get('final.health.json_status')==200 and report.get('final.prometheus.txt_status')==200)
            report['server_log_sha256']=hashlib.sha256((out/'server.log').read_bytes()).hexdigest()
            save()
            for s,handler in old.items(): signal.signal(s,handler)
    print(json.dumps({'event':'owned_server_complete','passed':report['passed'],'output':str(out)}),flush=True)
    return 0 if report['passed'] else 1

if __name__=='__main__': raise SystemExit(main())
