#!/usr/bin/env python3
"""Bounded, controller-owned live loopback checks. Starts one model server.

Run only after the experiment controller pauses the reference. The child is
always stopped in finally. No unrelated process is inspected or signalled.
"""
import argparse
import concurrent.futures
import hashlib
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runner',type=Path,required=True)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--golden-report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--port',type=int,default=11236)
    a=p.parse_args(); out=a.output.resolve()
    if out.exists():p.error('--output must be new')
    out.parent.mkdir(parents=True,exist_ok=True)
    model=a.model_dir.resolve().name
    child=None; shutdown_socket=None; report={'schema':'qwen-live-http-gate-v1','complete':False,'passed':False,'checks':[],
        'notes':['One real model; correctness and observed loopback lifecycle only, not a performance benchmark.',
                 'Paused client reads need not exhaust the OS send buffer. A live overflow is not claimed unless observed.',
                 'TCP write half-close and complete peer disappearance differ; explicit RST is used for cancellation.']}
    log=out.with_suffix('.server.log')
    def save():out.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    def check(name,condition,**data):
        report['checks'].append({'id':name,'passed':bool(condition),**data});save()
        print(name,condition,flush=True)
        if not condition:raise AssertionError(name)
    def request(method,path,body=None,timeout=300):
        c=http.client.HTTPConnection('127.0.0.1',a.port,timeout=timeout)
        start=time.monotonic(); payload=None if body is None else json.dumps(body,ensure_ascii=False).encode()
        try:
            c.request(method,path,body=payload,headers={'Content-Type':'application/json'} if body is not None else {})
            r=c.getresponse(); status=r.status;headers=dict(r.getheaders());raw=r.read(2*1024*1024)
            if len(raw)>=2*1024*1024:raise ValueError('Response exceeds harness bound')
            return status,headers,raw,time.monotonic()-start
        finally:c.close()
    def health():
        status,_,raw,_=request('GET','/health',timeout=3)
        return status,json.loads(raw)
    def until(predicate,timeout=20):
        end=time.monotonic()+timeout;last=None
        while time.monotonic()<end:
            if child and child.poll() is not None:raise RuntimeError(f'Server exited {child.returncode}')
            try:
                last=health()
                if predicate(last):return last
            except (OSError,ValueError,http.client.HTTPException):pass
            time.sleep(.1)
        raise TimeoutError(f'Health condition not reached; last={last}')
    def chat(messages,stream=False,depth=0,budget=128):
        return {'model':model,'messages':messages,'max_tokens':budget,'stream':stream,'mtp_depth':depth}
    def decode_response(response,stream):
        status,headers,raw,wall=response
        if status!=200:raise ValueError(f'HTTP {status}: {raw[:1000]!r}')
        if not stream:
            d=json.loads(raw);choice=d['choices'][0]
            assert d['object']=='chat.completion' and choice['message']['role']=='assistant'
            return {'text':choice['message']['content'],'finish':choice['finish_reason'],'usage':d['usage'],'wall':wall}
        assert 'text/event-stream' in headers.get('Content-Type',headers.get('content-type',''))
        # The concatenated byte stream is UTF-8; each complete SSE JSON frame
        # must independently decode without replacing incomplete scalars.
        events=[]
        for frame in raw.replace(b'\r\n',b'\n').split(b'\n\n'):
            if not frame:continue
            lines=frame.decode('utf-8',errors='strict').split('\n')
            data='\n'.join(x[5:].lstrip(' ') for x in lines if x.startswith('data:'))
            if data:events.append(data)
        assert events and events[-1]=='[DONE]' and events.count('[DONE]')==1
        frames=[json.loads(x) for x in events[:-1]]
        assert not any('error' in x for x in frames)
        assert frames[0]['choices'][0]['delta'].get('role')=='assistant'
        text=[];finish=[];usage=None
        for frame in frames:
            if frame.get('usage') is not None:usage=frame['usage']
            for choice in frame.get('choices',[]):
                assert choice['index']==0
                if choice['delta'].get('content') is not None:text.append(choice['delta']['content'])
                if choice.get('finish_reason') is not None:finish.append(choice['finish_reason'])
        assert len(finish)==1 and usage is not None
        return {'text':''.join(text),'finish':finish[0],'usage':usage,'wall':wall,'frames':len(frames)}
    def raw_send(body):
        payload=json.dumps(body,ensure_ascii=False).encode()
        s=socket.create_connection(('127.0.0.1',a.port),timeout=5)
        s.sendall((f'POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {len(payload)}\r\nConnection: close\r\n\r\n').encode()+payload)
        return s
    try:
        # Refuse to collide with any listener. Do not discover or stop its owner.
        test=socket.socket();test.bind(('127.0.0.1',a.port));test.close()
        golden=json.loads(a.golden_report.read_text())['trials'][0]
        messages=[{'role':'system','content':(a.fixture/'system-prompt.txt').read_text().strip()},
                  {'role':'user','content':(a.fixture/'user-prompt.txt').read_text().strip()}]
        command=[str(a.runner.resolve()),'serve-gpu','--model-dir',str(a.model_dir.resolve()),'--port',str(a.port),
                 '--max-connections','8','--output-buffer-bytes','8192']
        report['command']=command;report['runner_sha256']=hashlib.sha256(a.runner.read_bytes()).hexdigest()
        report['golden_sha256']=hashlib.sha256(a.golden_report.read_bytes()).hexdigest()
        with log.open('x') as f:
            child=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT)
        report['server_pid']=child.pid;save()
        _,ready=until(lambda x:x[0]==200 and x[1].get('status')=='ready',180)
        check('ready',ready.get('ready') is True,health=ready)
        status,_,raw,_=request('GET','/v1/models');models=json.loads(raw)
        check('model_identity',status==200 and [x['id'] for x in models['data']]==[model])
        base=chat([{'role':'user','content':'Reply with OK.'}])
        for field,value in [('model','unknown-model'),('temperature',.7),('tools',[]),('max_tokens',True),('stream','true')]:
            bad=dict(base);bad[field]=value;status,_,raw,_=request('POST','/v1/chat/completions',bad)
            check('reject_'+field,400<=status<500,status=status,error=json.loads(raw))
        ar=decode_response(request('POST','/v1/chat/completions',chat(messages,True)),True)
        check('long_ar_sse_exact',ar['text']==golden['text'] and ar['usage']['prompt_tokens']==len(golden['prompt_tokens']) and ar['usage']['completion_tokens']==len(golden['generated_token_ids']) and ar['finish']==('stop' if golden['finish_reason']=='eos' else 'length'),result=ar)
        mtp=decode_response(request('POST','/v1/chat/completions',chat(messages,False,2)),False)
        check('long_mtp_nonstream_exact',mtp['text']==ar['text'] and mtp['usage']==ar['usage'] and mtp['finish']==ar['finish'],result=mtp)
        chinese=[{'role':'user','content':'请只输出下面这行文字，不要解释：中文测试，海浪🌊。'}]
        short=decode_response(request('POST','/v1/chat/completions',chat(chinese,False,budget=64)),False)
        short_stream=decode_response(request('POST','/v1/chat/completions',chat(chinese,True,2,64)),True)
        check('chinese_stream_matches_nonstream',short_stream['text']==short['text'] and '\ufffd' not in short_stream['text'] and any(ord(c)>127 for c in short['text']),ar=short,mtp=short_stream)
        # Real concurrent requests use one scheduler and independent sockets.
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            long=pool.submit(request,'POST','/v1/chat/completions',chat(messages,True))
            brief=pool.submit(request,'POST','/v1/chat/completions',chat(chinese,True,2,64))
            lr=decode_response(long.result(),True);sr=decode_response(brief.result(),True)
        check('concurrent_long_short_exact',lr['text']==ar['text'] and sr['text']==short['text'],long_wall=lr['wall'],short_wall=sr['wall'])
        _,idle=until(lambda x:x[0]==200 and x[1].get('idle') is True)
        check('idle_after_concurrency',idle.get('reserved_tokens')==0 and idle.get('resident_sequences')==0,health=idle)
        # Two long reservations fit; a third exceeds the fixed 32768 quota.
        held=[]
        try:
            held.append(raw_send(chat(messages,True)))
            held.append(raw_send(chat(messages,True)))
            until(lambda x:x[1].get('active_jobs',0)>=2,20)
            status,_,raw,_=request('POST','/v1/chat/completions',chat(messages,True),timeout=15)
            check('token_reservation_overload_429',status==429,status=status,error=json.loads(raw))
        finally:
            for held_socket in held:
                held_socket.setsockopt(socket.SOL_SOCKET,socket.SO_LINGER,struct.pack('ii',1,0));held_socket.close()
        _,idle=until(lambda x:x[0]==200 and x[1].get('idle') is True,30)
        check('overload_cleanup',idle.get('reserved_tokens')==0 and idle.get('resident_sequences')==0,health=idle)
        # Legal request-side EOF must leave the response readable.
        half=raw_send(chat(chinese,False,budget=64))
        try:
            half.settimeout(60);half.shutdown(socket.SHUT_WR)
            response=http.client.HTTPResponse(half);response.begin()
            raw=response.read(65536);body=json.loads(raw)
            check('legal_write_half_close',response.status==200 and body['choices'][0]['message']['content']==short['text'])
        finally:half.close()
        # A request leaves with explicit RST while the GPU prefill is active.
        s=raw_send(chat(messages,True))
        try:
            until(lambda x:x[1].get('active_jobs',0)>0 or x[1].get('active',0)>0,15)
            start=time.monotonic();s.setsockopt(socket.SOL_SOCKET,socket.SO_LINGER,struct.pack('ii',1,0))
        finally:s.close()
        _,idle=until(lambda x:x[0]==200 and x[1].get('idle') is True,30)
        check('rst_cancellation_releases',idle.get('reserved_tokens')==0 and idle.get('resident_sequences')==0,elapsed=time.monotonic()-start,health=idle)
        # Pause reads without trying to infer whether the OS buffers filled.
        slow=raw_send(chat(messages,True))
        try:
            until(lambda x:x[1].get('active_jobs',0)>0,15)
            fresh=decode_response(request('POST','/v1/chat/completions',chat(chinese,True,2,64)),True)
            check('paused_reader_does_not_block_other_request',fresh['text']==short['text'],other_wall=fresh['wall'],live_overflow_observed=False)
        finally:
            slow.setsockopt(socket.SOL_SOCKET,socket.SO_LINGER,struct.pack('ii',1,0));slow.close()
        until(lambda x:x[0]==200 and x[1].get('idle') is True,30)
        fresh=decode_response(request('POST','/v1/chat/completions',chat(chinese,False,budget=64)),False)
        check('fresh_after_disconnects',fresh['text']==short['text'],result=fresh)
        shutdown_socket=raw_send(chat(messages,True))
        _,active=until(lambda x:x[1].get('active',0)>0 and x[1].get('active_jobs',0)>0 and x[1].get('resident_sequences',0)>0,20)
        check('shutdown_during_active_request_precondition',True,health=active)
        report['complete']=True;report['passed']=all(x['passed'] for x in report['checks'])
    except Exception as error:
        report['error']=f'{type(error).__name__}: {error}'
    finally:
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill();child.wait(timeout=10);report['forced_shutdown']=True
        if child is not None:
            report['server_exit_code']=child.returncode
            report['graceful_shutdown']=not report.get('forced_shutdown',False) and child.returncode==0
            report['passed']=report['passed'] and report['graceful_shutdown']
        if shutdown_socket is not None:shutdown_socket.close()
        save()
    print(json.dumps({'passed':report['passed'],'complete':report['complete'],'error':report.get('error'),'graceful_shutdown':report.get('graceful_shutdown')},indent=2))
    return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
