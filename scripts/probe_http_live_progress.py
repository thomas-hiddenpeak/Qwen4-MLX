#!/usr/bin/env python3
"""Short client-only HTTP progress gate. Root owns the already-ready service."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import json
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from probe_http_long_context import ExternalServer, idle, strict_completion
from test_http_server_edges import HTTPHarness, read_response, require_lossless_logging


def require(okay, message):
    if not okay: raise ValueError(message)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runner', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--server-pid', type=int, required=True)
    p.add_argument('--server-log', type=Path, required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--fixture', type=Path, default=ROOT / 'fixtures/gpu-agent-11k')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    require(not sys.flags.optimize and a.port != 11235 and a.server_pid > 1, 'Use non-optimized Python and an owned non-reference server')
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + 180
    def interrupted(_number, _frame): raise InterruptedError('Client stop requested')
    signal.signal(signal.SIGTERM, interrupted)
    harness = HTTPHarness(a.port, ExternalServer(a.server_pid))
    report = {'schema': 'qwen-http-live-progress-gate-v1', 'complete': False, 'passed': False,
              'server_pid': a.server_pid, 'responses': {}, 'terminals': {}, 'health': {}, 'samples': 0,
              'notes': ['Client-only three-minute smoke; no server management or performance claim.',
                        'Native CPU tokenization of existing authored 11K text as one user message.',
                        'Progress is logical prefix including cache, not actual forward work.']}
    history, disappeared, previous = {}, set(), set()
    samples = (out / 'health.ndjson').open('x')
    def dump(name, value):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    def remaining(cap=30):
        value = min(cap, deadline-time.monotonic())
        if value <= 0: raise TimeoutError('Three-minute client deadline')
        return value
    def sample():
        nonlocal previous
        remaining(); status, h = harness.health()
        require(status == 200 and h.get('ready') is True, 'Owned service unavailable')
        require_lossless_logging(h)
        require(h.get('request_progress_sample') == 'completed_scheduler_slice' and h.get('processed_prompt_tokens_includes_cache') is True, 'Progress contract missing')
        rows = h.get('request_progress')
        require(isinstance(rows, list) and len(rows) <= 10 and len(rows) == h.get('active_jobs'), 'Progress/job count mismatch')
        identities = set()
        for r in rows:
            require(set(r) == {'request_id','stage','last_event','prompt_total','processed_prompt_tokens','generated_tokens'}, 'Unexpected progress fields')
            ident = r['request_id']; total, done, generated = (r[k] for k in ('prompt_total','processed_prompt_tokens','generated_tokens'))
            require(isinstance(ident,str) and 0<len(ident)<=128 and ident not in identities and ident not in disappeared, 'Duplicate/reappearing request ID')
            require(all(type(x) is int for x in (total,done,generated)) and 0<=done<=total and total>0 and 0<=generated<=32, 'Progress count outside bounds')
            event = r['last_event']
            require((r['stage']=='prefill' and event in (None,'prefillProgress','prefillReady')) or (r['stage']=='decode' and event=='decodeProgress'), 'Invalid progress phase')
            if event is None: require(done == generated == 0, 'Unstarted request has progress')
            if r['stage']=='prefill': require(generated == 0, 'Prefill reports published outputs')
            if event in ('prefillReady','decodeProgress'): require(done==total, 'Ready/decode prefix incomplete')
            if event=='decodeProgress': require(generated>0, 'Decode slice has no committed output')
            old = history.setdefault(ident, [])
            if old: require(total==old[-1]['prompt_total'] and done>=old[-1]['processed_prompt_tokens'] and generated>=old[-1]['generated_tokens'], 'Progress decreased')
            old.append(dict(r)); identities.add(ident)
        disappeared.update(previous-identities); previous = identities
        report['samples'] += 1
        require(report['samples'] <= 2000, 'Health sample bound exceeded')
        samples.write(json.dumps({'elapsed':180-remaining(180),'health':h},allow_nan=False)+'\n'); samples.flush()
        return h
    def settled(name):
        end = time.monotonic()+remaining(20)
        while time.monotonic()<end:
            h=sample()
            if idle(h,0) and not h['request_progress'] and all(h.get('logging',{}).get(k)==0 for k in ('queued_events','in_flight_bytes','buffered_events')):
                report['health'][name]=h; return h
            time.sleep(.1)
        raise TimeoutError('Resources did not settle: '+name)
    def terminal(ident):
        end=time.monotonic()+remaining(10)
        while time.monotonic()<end:
            with a.server_log.open('rb') as log: raw=log.read(8*1024*1024+1)
            require(len(raw)<=8*1024*1024,'Server log exceeds bound')
            rows=[json.loads(x) for x in raw[:raw.rfind(b'\n')+1].splitlines() if x.startswith(b'{')]
            found=[r for r in rows if r.get('schema')=='qwen-http-lifecycle-v1' and r.get('pid')==a.server_pid and r.get('event')=='model_terminal' and r.get('request_id')==ident]
            require(len(found)<=1,'Duplicate terminal')
            if found: report['terminals'][ident]=found[0]; return found[0]
            time.sleep(.1)
        raise TimeoutError('Missing terminal '+ident)
    def tokenize(name, prompt):
        source=out/(name+'.prompt.txt'); source.write_text(prompt)
        target=out/(name+'.tokenization.json')
        command=[str(a.runner.resolve()),'tokenize','--model-dir',str(a.model_dir.resolve()),'--prompt-file',str(source),'--chat','true','--output',str(target)]
        with (out/(name+'.tokenizer.log')).open('xb') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=remaining(20),check=True)
        row=json.loads(target.read_text()); ids=row.get('tokens')
        require(isinstance(ids,list) and ids and all(type(x)is int and 0<=x<2**31 for x in ids),'Invalid native tokenization')
        require(row.get('decoded')==row.get('rendered_prompt') and isinstance(row.get('decoded'),str) and prompt.strip() in row['decoded'],'Native chat roundtrip mismatch')
        return {'model':a.model_dir.resolve().name,'messages':[{'role':'user','content':prompt}],'max_tokens':32,'stream':False,'mtp_depth':0}, len(ids)
    def completion(name, payload, prompt, expected_cached):
        dump(name+'.request.json',payload); before=set(history); started=time.monotonic()
        sock=harness.post_socket(payload)
        pool=ThreadPoolExecutor(max_workers=1)
        future=pool.submit(read_response,sock,remaining(75))
        try:
            while not future.done(): sample(); time.sleep(.1)
            status,headers,raw=future.result(); response=(status,headers,raw,time.monotonic()-started)
            (out/(name+'.response.raw')).write_bytes(raw)
            result=strict_completion(response,payload['stream'],payload['model']); ident=result['id']
            settled(name); row=terminal(ident); usage=result['usage']; n=usage.get('completion_tokens')
            require(set(history)-before=={ident},'Progress ID does not match response or was not observed')
            require(all(r['prompt_total']==prompt for r in history[ident]),'Progress total differs from native chat tokens')
            require(type(n)is int and 1<n<=32 and usage.get('prompt_tokens')==prompt and usage.get('total_tokens')==prompt+n,'Response token accounting mismatch')
            cached=usage.get('prompt_tokens_details',{}).get('cached_tokens',0)
            require(cached==expected_cached and row.get('cached_prompt_tokens')==cached and row.get('computed_prompt_tokens')==prompt-cached,'Cache accounting mismatch')
            require(row.get('model_kind')=='completed' and row.get('mtp_depth')==0 and row.get('prompt_tokens')==prompt and row.get('completion_tokens')==n and row.get('decoded_tokens')==n-1 and row.get('final_state_offset')==prompt+n-1,'Terminal does not match response')
            require(all(r['generated_tokens']<=n for r in history[ident]),'Observed output count exceeds terminal')
            if cached:
                require(row.get('cache_source')=='memory' and max(r['processed_prompt_tokens'] for r in history[ident])>=cached,'Warm logical progress omits cache')
            report['responses'][name]=result; dump('summary.json',report); return result
        finally:
            if not future.done():
                try: sock.shutdown(socket.SHUT_RDWR)
                except OSError: pass
            harness.close(sock,reset=not future.done()); pool.shutdown(wait=True,cancel_futures=True)
    try:
        initial=settled('initial')
        require(initial.get('prefix_disk_cache') is None and not initial.get('paged_kv_pool',{}).get('configured'),'Smoke requires RAM-only dense service')
        require(initial.get('prefix_cache',{}).get('entries')==0,'Cold case requires fresh empty RAM cache')
        prompt=(a.fixture/'system-prompt.txt').read_text().strip()+'\n\n'+(a.fixture/'user-prompt.txt').read_text().strip()
        normal,P=tokenize('long',prompt); cancel,CP=tokenize('cancel','Independent cancellation request.\n'+prompt)
        require(10000<=P<=32768 and CP>=10000,'Fixture must remain a long prompt')
        report['input_sha256']={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in (a.runner,Path(__file__),a.fixture/'system-prompt.txt',a.fixture/'user-prompt.txt')}
        B=(P-1)//416*416; report.update(prompt_tokens=P,cancel_prompt_tokens=CP,cached_tokens=B)
        cold=completion('cold',normal,P,0)
        require(any(0<r['processed_prompt_tokens']<P for r in history[cold['id']]),'No actual partial cold prefill sample')
        warm=completion('warm',dict(normal,stream=True),P,B)
        require(any(r['stage']=='decode' and r['generated_tokens']>0 for ident in (cold['id'],warm['id']) for r in history[ident]),'No decode progress was observed')
        require(all(cold[k]==warm[k] for k in ('text','finish')) and cold['usage']['completion_tokens']==warm['usage']['completion_tokens'],'Cold/warm output changed')
        dump('cancel.request.json',dict(cancel,stream=True)); sock=harness.post_socket(dict(cancel,stream=True)); response=http.client.HTTPResponse(sock)
        try:
            sock.settimeout(remaining(10)); response.begin(); require(response.status==200,'Cancellation did not admit SSE')
            body=bytearray()
            while not body.endswith(b'\n\n'):
                line=response.readline(8193); require(line and len(body)+len(line)<16384,'Invalid cancellation role frame'); body.extend(line)
            require(body.startswith(b'data: '),'Missing SSE role'); frame=json.loads(body[6:]); ident=frame['id']
            require(frame['choices'][0]['delta'].get('role')=='assistant' and frame['choices'][0]['finish_reason'] is None,'Invalid cancellation role')
            (out/'cancel.role.sse').write_bytes(body); end=time.monotonic()+remaining(15)
            while True:
                h=sample(); rows=[r for r in h['request_progress'] if r['request_id']==ident]
                if rows and rows[0]['stage']=='prefill' and 0<rows[0]['processed_prompt_tokens']<CP and h['resident_sequences']>0: break
                require(time.monotonic()<end,'No owned partial prefill before cancellation'); time.sleep(.1)
            require(rows[0]['prompt_total']==CP,'Cancellation prompt total mismatch')
        finally:
            sock.setsockopt(socket.SOL_SOCKET,socket.SO_LINGER,struct.pack('ii',1,0)); response.close(); harness.close(sock,reset=True)
        settled('cancel'); row=terminal(ident)
        require(row.get('model_kind')=='cancelled' and row.get('stage')=='prefill' and ident in disappeared,'Cancellation was not confirmed and removed')
        require(all(r['prompt_total']==CP for r in history[ident]),'Cancellation total differs from native chat tokens')
        fresh=completion('fresh',normal,P,B)
        require(all(fresh[k]==warm[k] for k in ('text','finish','usage')),'Post-cancel warm request changed')
        settled('final'); report.update(complete=True,passed=True,elapsed_seconds=180-remaining(180),cancel_id=ident)
    except Exception as error:
        report['error']=f'{type(error).__name__}: {error}'
    finally:
        harness.close_all(); samples.close(); dump('summary.json',report)
    print(json.dumps({'complete':report['complete'],'passed':report['passed'],'error':report.get('error')}))
    return 0 if report['passed'] else 1

if __name__ == '__main__': raise SystemExit(main())
