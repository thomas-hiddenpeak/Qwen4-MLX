#!/usr/bin/env python3
"""CPU fixture preparation and one controller-owned 262K HTTP case. Never launch a server."""
import argparse
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import signal
import socket
import struct
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from test_http_server_edges import HTTPHarness, decode_completion, read_response, require_lossless_logging

P = 262142
CHUNK = 416
CACHED = (P - 1) // CHUNK * CHUNK
ZERO = ('active', 'active_jobs', 'pending_requests', 'queued_prefills', 'ready_decodes',
        'resident_sequences', 'reserved_tokens', 'waiting_prefix_sequences')
PREFIX = 'This is a synthetic long-context cache capacity test. Ignore the filler words below.\n'
TAIL = '\nNow write the integers from 1 to 100, separated by commas. Start with 1,2,3 and do not explain.'
UNIT = ' record'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write('\n')


def prepare(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    context = args.context_limit
    P = context - 2
    CACHED = (P - 1) // CHUNK * CHUNK
    model = args.model_dir.resolve()
    runner = args.runner.resolve()
    manifest = {'schema': 'qwen-long-http-fixture-v1', 'complete': False,
                'runner_sha256': sha(runner), 'script_sha256': sha(__file__),
                'model_id': model.name, 'context_limit': context, 'unit': UNIT, 'synthetic_capacity_only': True,
                'quality_evaluation': False, 'source': 'Deterministic authored single-user chat; native Swift full-render tokenization.',
                'model_files': {name: sha(model / name) for name in ('config.json', 'tokenizer.json', 'chat_template.jinja')},
                'tokenizations': {}, 'requests': {}}

    def tokenize(name, prompt):
        prompt_path = out / (name + '.prompt.txt')
        prompt_path.write_text(prompt, encoding='utf-8')
        report_path = out / (name + '.tokenization.json')
        # Existing tokenize schema; root adds only the bounded UTF-8 file input
        # alternative before running this preparation. No model weights/GPU.
        command = [str(runner), 'tokenize', '--model-dir', str(model), '--prompt-file',
                   str(prompt_path), '--chat', 'true', '--output', str(report_path)]
        with (out / (name + '.tokenizer.log')).open('xb') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=180, check=True)
        report = json.loads(report_path.read_text())
        ids = report.get('tokens')
        require(isinstance(ids, list) and ids and all(type(t) is int and 0 <= t < 2**31 for t in ids), 'Bad native tokens: ' + name)
        rendered = report.get('rendered_prompt')
        require(isinstance(rendered, str) and rendered.count(prompt.strip()) == 1, 'Native rendered prompt does not contain exact authored content')
        require(report.get('decoded') == rendered, 'Full native token roundtrip differs from rendering: ' + name)
        tokens_path = out / (name + '.tokens.json')
        dump(tokens_path, ids)
        manifest['tokenizations'][name] = {'prompt_file': prompt_path.name, 'prompt_sha256': sha(prompt_path),
            'tokenization_file': report_path.name, 'tokenization_sha256': sha(report_path),
            'tokens_file': tokens_path.name, 'tokens_sha256': sha(tokens_path), 'prompt_tokens': len(ids), 'command': command}
        print(json.dumps({'event': 'tokenized', 'name': name, 'prompt_tokens': len(ids)}), flush=True)
        return ids

    try:
        # Three tiny whole-chat observations calibrate a one-token repeat. Two
        # complete 262K encodings then verify exact counts; no iterative large search.
        samples = {n: tokenize('calibrate-' + str(n), PREFIX + UNIT * n + TAIL) for n in (32, 33, 64)}
        overhead = len(samples[32]) - 32
        require(all(len(ids) - n == overhead for n, ids in samples.items()), 'Selected repeat is not exactly one token per unit; stop before large tokenization')
        cases = [('long', P, PREFIX + UNIT * (P - overhead) + TAIL, 2),
                 ('over', P + 1, PREFIX + UNIT * (P + 1 - overhead) + TAIL, 2),
                 ('cancel', None, 'Cancellation-only independent prefix. Ignore this filler.\n' + UNIT * 32000 + TAIL, 32),
                 ('short', None, 'Write the integers from 1 to 100 separated by commas, starting with 1,2,3. Do not explain.', 8)]
        for name, expected, prompt, output in cases:
            ids = tokenize(name, prompt)
            if expected is not None:
                require(len(ids) == expected, 'Exact whole-chat boundary failed for ' + name)
            if name == 'cancel':
                require(30000 <= len(ids) <= 33000 and len(ids)+output <= context, 'Cancellation fixture is outside the 32K envelope/context')
            if name == 'short':
                require(len(ids) < CHUNK, 'Short fixture would publish a prefix checkpoint')
            payload = {'model': model.name, 'messages': [{'role': 'user', 'content': prompt}],
                       'max_tokens': output, 'stream': name == 'cancel', 'mtp_depth': 0}
            path = out / (name + '.request.json')
            dump(path, payload)
            manifest['requests'][name] = {'file': path.name, 'sha256': sha(path),
                'prompt_tokens': len(ids), 'max_tokens': output, 'body_bytes': len(json.dumps(payload, ensure_ascii=False).encode())}
        if context == 262144:
            require(manifest['requests']['long']['body_bytes'] > 1024 * 1024, 'Long chat does not exercise the old 1MiB body ceiling')
        require(manifest['requests']['over']['prompt_tokens'] + 2 == context+1, 'Over-bound total is wrong')
        manifest.update(complete=True, expected_warm_cached_tokens=CACHED, expected_warm_computed_tokens=P-CACHED,
                        expected_final_state_offset=P+1, full_262k_tokenization_calls=2 if context == 262144 else 0)
    except BaseException as error:
        manifest['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        dump(out / 'fixture.json', manifest)


class ExternalServer:
    """Read-only liveness adapter for the existing socket harness; never signals/stops the service."""
    def __init__(self, pid):
        self.pid, self.returncode = pid, None
    def poll(self):
        try:
            os.kill(self.pid, 0)
            return None
        except ProcessLookupError:
            self.returncode = -1
            return -1


def strict_completion(response, stream, model):
    status, headers, raw, _ = response
    headers = {k.lower():v for k,v in headers.items()}
    require(status == 200, 'Completion HTTP status is not200')
    content_type = headers.get('content-type', '')
    require(content_type.split(';',1)[0].strip().lower() == ('text/event-stream' if stream else 'application/json'), 'Incorrect completion content type')
    if 'content-length' in headers:
        require(int(headers['content-length']) == len(raw), 'Content-Length differs from response body')
    if stream:
        require(raw.endswith(b'\n\n'), 'Truncated final SSE delimiter')
        parts = [p for p in raw.split(b'\n\n') if p]
        require(parts and parts[-1] == b'data: [DONE]' and parts.count(b'data: [DONE]') == 1, 'Incomplete/duplicated SSE DONE')
        require(all(p.startswith(b'data: ') for p in parts[:-1]), 'Malformed SSE data frame')
        values = [json.loads(p[6:].decode('utf-8', errors='strict')) for p in parts[:-1]]
    else:
        values = [json.loads(raw.decode('utf-8', errors='strict'))]
    require(values, 'Missing completion object')
    for v in values:
        require(isinstance(v,dict) and 'error' not in v and v.get('model') == model, 'Wrong completion model/object')
        require(v.get('object') == ('chat.completion.chunk' if stream else 'chat.completion'), 'Wrong completion object type')
        require(isinstance(v.get('id'),str) and 0<len(v['id'])<=128 and type(v.get('created')) is int and v['created']>=0, 'Invalid completion identity')
        choices = v.get('choices')
        require(isinstance(choices,list) and len(choices)==1 and isinstance(choices[0],dict), 'Completion must contain exactly one choice')
        choice=choices[0]
        require(type(choice.get('index')) is int and choice['index']==0, 'Invalid completion choice index')
        message=choice.get('delta' if stream else 'message')
        require(isinstance(message,dict) and not message.get('tool_calls'), 'Unexpected tool or malformed AR message')
        require((stream and message.get('content') is None) or isinstance(message.get('content'),str), 'Completion content must be text')
        require(choice.get('finish_reason') in ((None,'stop','length') if stream else ('stop','length')), 'Invalid AR finish reason')
    return decode_completion(response, stream)


def idle(value, pages):
    if value.get('idle') is not True or any(value.get(k) != 0 for k in ZERO):
        return False
    budget = value.get('state_budget', {})
    cache = value.get('prefix_cache', {})
    if cache.get('liveFlights') != 0 or budget.get('cacheBytes') != cache.get('logicalPayloadBytes'):
        return False
    pool = value.get('paged_kv_pool', {})
    stats = pool.get('statistics')
    if pages:
        if not isinstance(stats, dict) or stats.get('statistics_error') is not None:
            return False
        if any(stats.get(k) != 0 for k in ('claimed_pages_per_layer', 'active_decode_claims', 'in_flight_operations', 'failed_operations')):
            return False
        fixed = stats.get('arena_reserved_bytes')
        vm = stats.get('vm_page_bytes')
        if type(vm) is not int or vm <= 0:
            return False
        arena_each = 2 * (((pages * 32768 + vm - 1) // vm) * vm + 2 * vm)
        expected = 12 * (arena_each + 65536 + pages * 128)
        if type(fixed) is not int or fixed != expected or stats.get('layer_count') != 12:
            return False
        if stats.get('physical_pages') != 12 * pages or stats.get('live_pages', -1) + stats.get('free_pages', -1) != 12 * pages:
            return False
        # These fixtures publish only >=30K checkpoints, beyond512-page
        # extent, or no checkpoint at all. At idle all retained cache is dense.
        if stats.get('live_pages') != 0 or stats.get('completed_operations') != sum(stats.get(k, -1) for k in ('encoded_writes','encoded_reads','encoded_materializations')):
            return False
        allocated = stats.get('arena_allocated_bytes')
        if type(allocated) is not int or not 12 * pages * 65536 <= allocated <= 12 * arena_each:
            return False
    else:
        fixed = 0
    disk = value.get('prefix_disk_cache')
    return (budget.get('requestBytes') == 0 and budget.get('workspaceBytes') == fixed
            and (disk is None or all(disk.get(k) == 0 for k in ('pendingJobs', 'pendingBytes', 'foregroundReadIntents'))))


def run(args):
    require(args.port != 11235, 'The reference business service on 11235 is excluded')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    fixture_root = args.fixture.resolve()
    fixture = json.loads((fixture_root / 'fixture.json').read_text())
    require(fixture.get('schema') == 'qwen-long-http-fixture-v1' and fixture.get('complete') is True, 'Fixture preparation is incomplete')
    context = fixture.get('context_limit')
    require(context in (32768,65536,131072,262144), 'Unsupported bounded context profile')
    P = context-2
    CACHED = (P-1)//CHUNK*CHUNK
    require(sha(args.runner) == fixture['runner_sha256'], 'Serving/fixture runner hash differs; freeze one binary before preparation and launch')
    for record in fixture['requests'].values():
        require(sha(fixture_root / record['file']) == record['sha256'], 'Request fixture changed')
    for record in fixture['tokenizations'].values():
        for kind in ('prompt', 'tokenization', 'tokens'):
            require(sha(fixture_root / record[kind + '_file']) == record[kind + '_sha256'], 'Native tokenizer evidence changed')
    payloads = {name: json.loads((fixture_root / record['file']).read_text()) for name, record in fixture['requests'].items()}
    # Re-read actual native IDs and link the exact HTTP content to the input
    # file. The gate does not accept counts asserted only by the manifest.
    for name, request in payloads.items():
        record = fixture['tokenizations'][name]
        actual = json.loads((fixture_root / record['tokenization_file']).read_text())
        ids = json.loads((fixture_root / record['tokens_file']).read_text())
        prompt = (fixture_root / record['prompt_file']).read_text()
        require(ids == actual['tokens'] and len(ids) == fixture['requests'][name]['prompt_tokens'], 'Raw native token count mismatch')
        require(request['messages'] == [{'role':'user','content':prompt}], 'HTTP content differs from native-tokenized input')
        require(actual['decoded'] == actual['rendered_prompt'] and actual['rendered_prompt'].count(prompt.strip()) == 1, 'Native roundtrip/render evidence changed')
    require(len(json.loads((fixture_root / fixture['tokenizations']['long']['tokens_file']).read_text())) == P, 'Boundary prompt does not equal context-2')
    require(len(json.loads((fixture_root / fixture['tokenizations']['over']['tokens_file']).read_text())) == P+1, 'Over-bound prompt does not equal context-1')
    require(payloads['long']['max_tokens'] == payloads['over']['max_tokens'] == 2, 'Boundary output budget is not two')
    decode_fixture = None
    if args.decode_fixture is not None:
        from long_context_decode_fixture import load_decode_fixture
        decode_fixture = load_decode_fixture(args.decode_fixture.resolve(), fixture_root, fixture, args.runner)
    harness = HTTPHarness(args.port, ExternalServer(args.server_pid))
    report = {'schema': 'qwen-long-http-gate-v1', 'complete': False, 'passed': False,
              'server_pid': args.server_pid, 'port': args.port, 'context_limit': context, 'runner_sha256': sha(args.runner),
              'script_sha256': sha(__file__), 'fixture_sha256': sha(fixture_root/'fixture.json'),
              'edge_harness_sha256': sha(ROOT/'scripts/test_http_server_edges.py'),
              'checks': [], 'responses': {}, 'health': {}, 'terminal_records': {}, 'wire_evidence': {},
              'notes': ['Caller exclusively owns service launch and shutdown; this script never launches or signals it.',
                        'Synthetic boundary/transport/RAM-cache check; not quality, speed acceptance, RSS cap or endurance.',
                        'MLX peak snapshots are cumulative, not isolated per-phase peaks. Native counters are separate from terminal phase totals.',
                        'Cancellation is a separate approximately32K prefill; full262K cancellation and full262K SSD restore are not claimed.']}
    report['extended_decode'] = {'enabled': decode_fixture is not None,
                                 'helper_sha256': sha(Path(__file__).with_name('long_context_decode_fixture.py'))}
    if decode_fixture is not None:
        report['extended_decode'].update(decode_fixture['evidence'])
        report['notes'].append('Extended P262112/O32 uses the original prompt cache; no independent cold oracle for this new prompt and no quality claim.')
    progress = (out/'progress.ndjson').open('x')

    def event(name, **fields):
        line = json.dumps({'event': name, 'unix_seconds': time.time(), **fields}, allow_nan=False)
        progress.write(line+'\n'); progress.flush(); print(line, flush=True)
    def save():
        temp = out/'summary.pending.json'
        temp.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
        temp.replace(out/'summary.json')
    def check(name, okay, **fields):
        report['checks'].append({'id': name, 'passed': bool(okay), **fields})
        save(); event(name, passed=bool(okay))
        require(okay, name)
    def settled(name, timeout=60):
        h = harness.until(lambda item: item[0] == 200 and idle(item[1], args.paged_pages)
            and all(item[1].get('logging',{}).get(k)==0 for k in ('queued_events','in_flight_bytes','buffered_events')), timeout=timeout)
        require_lossless_logging(h)
        report['health'][name] = h; save()
        return h
    def log_records():
        raw = args.server_log.read_bytes()
        require(len(raw) <= 16*1024*1024, 'Server log exceeds this bounded gate')
        records = []
        for line in raw[:raw.rfind(b'\n')+1].splitlines():
            if not line.startswith(b'{'):
                continue
            value = json.loads(line)
            if value.get('schema') == 'qwen-http-lifecycle-v1' and value.get('pid') == args.server_pid:
                records.append(value)
        return records
    def terminal(identity):
        deadline = time.monotonic()+15
        while time.monotonic() < deadline:
            matches = [r for r in log_records() if r.get('event') == 'model_terminal' and r.get('request_id') == identity]
            require(len(matches) <= 1, 'Duplicate model terminal for '+identity)
            if matches:
                report['terminal_records'][identity] = matches[0]; save()
                return matches[0]
            time.sleep(.1)
        raise TimeoutError('Missing model terminal '+identity)
    def record_wire(name, payload, response=None):
        wire=json.dumps(payload,ensure_ascii=False).encode()
        (out/(name+'.request.json')).write_bytes(wire)
        record={'request_body_bytes':len(wire),'request_body_sha256':hashlib.sha256(wire).hexdigest()}
        if response is not None:
            record.update(response_status=response[0],response_headers=response[1],response_body_bytes=len(response[2]),
                          response_body_sha256=hashlib.sha256(response[2]).hexdigest())
        report['wire_evidence'][name]=record; save()
    def request_case(name, payload, prompt, cached, exact_output=None):
        event('request_start', name=name, prompt_tokens=prompt, max_tokens=payload['max_tokens'], stream=payload['stream'])
        response = harness.request('POST', '/v1/chat/completions', payload, timeout=args.timeout)
        (out/(name+'.response.raw')).write_bytes(response[2])
        record_wire(name,payload,response)
        result = strict_completion(response, payload['stream'], fixture['model_id'])
        report['responses'][name] = result; save()
        h = settled(name)
        row = terminal(result['id'])
        usage = result['usage']; count = usage.get('completion_tokens')
        cached_actual = usage.get('prompt_tokens_details', {}).get('cached_tokens', 0)
        require(type(count) is int and 0 < count <= payload['max_tokens'], name+' completion count invalid')
        if exact_output is not None:
            require(count == exact_output and result['finish'] == 'length', name+' did not execute the exact output budget')
        check(name+'_usage_and_terminal', usage.get('prompt_tokens') == prompt and usage.get('total_tokens') == prompt+count
              and cached_actual == cached and row.get('model_kind') == 'completed' and row.get('mtp_depth') == 0
              and row.get('prompt_tokens') == prompt and row.get('completion_tokens') == count
              and row.get('decoded_tokens') == count-1 and row.get('cached_prompt_tokens') == cached
              and row.get('computed_prompt_tokens') == prompt-cached
              and row.get('actual_prefill_tokens') == prompt-cached and row.get('recomputed_prefill_tokens') == 0
              and row.get('final_state_offset') == prompt+count-1
              and row.get('prefill_attention') == args.prefill_attention)
        for key in ('prefill_seconds', 'decode_seconds', 'prefill_active_seconds', 'decode_service_seconds',
                    'handoff_wait_seconds', 'handoff_consume_seconds', 'scheduler_elapsed_seconds'):
            value = row.get(key)
            require(type(value) in (int,float) and math.isfinite(value) and value >= 0, name+' phase missing/invalid: '+key)
        if count > 1:
            require(row['decode_seconds'] > 0 and row['decode_service_seconds'] > 0, 'A real decode requires positive measured time')
        return result, row, h
    def no_native_delta(before, after):
        if not args.paged_pages:
            return True
        a, b = before['paged_kv_pool']['statistics'], after['paged_kv_pool']['statistics']
        return all(a[k] == b[k] for k in ('encoded_writes','encoded_reads','encoded_materializations'))
    def rejected(name, payload, status=400, reason=None):
        before = settled(name+'_before')
        prior = len([r for r in log_records() if r.get('event') == 'model_terminal'])
        response = harness.request('POST', '/v1/chat/completions', payload, timeout=60)
        (out/(name+'.response.raw')).write_bytes(response[2])
        after = settled(name+'_after')
        record_wire(name,payload,response)
        body = json.loads(response[2])
        stable_admission = (all(before['state_budget'][k]==after['state_budget'][k] for k in ('peakBytes','rejections','requestBytes','cacheBytes','workspaceBytes','currentLeases'))
            and all(before['prefix_cache'][k]==after['prefix_cache'][k] for k in ('hits','misses','published','entries','restoredHits','liveFlights','flightWaits')))
        check(name, response[0] == status and isinstance(body.get('error'),dict)
              and body['error'].get('code') == 'invalid_request_error'
              and (reason is None or reason in body['error'].get('message',''))
              and len([r for r in log_records() if r.get('event') == 'model_terminal']) == prior
              and stable_admission and no_native_delta(before, after), status=response[0], response=body,
              evidence_scope='Unchanged admission ledger/cache counters, drained terminal logger and optional native counters; not global GPU tracing')

    def extended_decode_case(name, streaming, before):
        # Restore the original262080 checkpoint, including after the unrelated
        # request cancellation. No independent cold oracle for this new prompt.
        result, row, after = request_case(name, dict(decode_fixture['payload'], stream=streaming), 262112, 262080, 32)
        check(name+'_restores_original_checkpoint', row.get('cache_source') == 'memory'
              and row.get('computed_prompt_tokens') == 32 and row.get('actual_prefill_tokens') == 32
              and row.get('decoded_tokens') == 31 and row.get('final_state_offset') == 262143
              and after['state_budget']['cacheBytes'] > 0)
        if args.paged_pages:
            sa, sb = before['paged_kv_pool']['statistics'], after['paged_kv_pool']['statistics']
            check(name+'_dense_fallback_without_long_import', no_native_delta(before, after)
                  and row.get('kv_append_mode') == 'reference' and row.get('paged_kv_capacity_fallbacks') == 1
                  and all(row.get(k) == 0 for k in ('paged_kv_token_steps', 'paged_kv_reused_prefix_tokens',
                                                   'paged_kv_imported_suffix_rows', 'paged_kv_reserved_pages_per_layer'))
                  and sb['completed_capacity_fallbacks']-sa['completed_capacity_fallbacks'] == 1)
        else:
            check(name+'_dense_default', row.get('kv_append_mode') == 'reference'
                  and row.get('paged_kv_token_steps') is None and row.get('paged_kv_capacity_fallbacks') is None)
        return result

    old_term = signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(InterruptedError('Controller interrupted client')))
    try:
        start = settled('start')
        capacity = start.get('service_capacity', {})
        pool = start.get('paged_kv_pool', {})
        check('explicit_long_profile', start.get('model') == fixture['model_id'] and start.get('model_maximum_positions') >= context
              and capacity.get('contextLimit') == context and capacity.get('maxReservedTokens') == context
              and capacity.get('maxResidentSequences') == 1 and capacity.get('maxBodyBytes') >= 8*1024*1024
              and capacity.get('connectionDeadlineSeconds') >= 3600 and start.get('maximum_mtp_depth') == 0
              and start.get('prefill_attention') == args.prefill_attention
              and start['state_budget'].get('maxBytes') >= 24*1024**3
              and start['prefix_cache_limits'].get('maxBytes') >= 8*1024**3
              and start.get('prefix_disk_cache') is None and start.get('prefix_disk_cache_limits') is None
              and pool.get('configured') == bool(args.paged_pages)
              and (not args.paged_pages or pool.get('maximum_pages_per_layer') == args.paged_pages))
        check('fresh_ram_cache', start['prefix_cache'].get('entries') == 0 and start['prefix_cache'].get('restoredHits') == 0)
        rejected('over_one_token', payloads['over'], reason='prompt plus requested output exceeds contextLimit')
        mtp = dict(payloads['short'], mtp_depth=2, max_tokens=2)
        rejected('mtp_rejected', mtp, reason='mtp_depth=0')
        sock = harness.connect()
        try:
            sock.sendall(('POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: '+str(capacity['maxBodyBytes']+1)+'\r\n\r\n').encode())
            status, _, raw = read_response(sock, timeout=5)
            dump(out/'body413.response.json', json.loads(raw))
            check('declared_body_over_limit_413', status == 413)
        finally:
            harness.close(sock)
        settled('body413')
        short, _, _ = request_case('short_before', payloads['short'], fixture['requests']['short']['prompt_tokens'], 0)
        before = settled('before_cold')
        cold, cold_row, cold_h = request_case('cold_json', payloads['long'], P, 0, 2)
        warm, warm_row, warm_h = request_case('warm_sse', dict(payloads['long'], stream=True), P, CACHED, 2)
        check('cold_warm_same_output', cold['text'] == warm['text'] and cold['finish'] == warm['finish'])
        check('warm_uses_ram_checkpoint', warm_row.get('cache_source') == 'memory' and warm_h['state_budget']['cacheBytes'] > 0)
        for name, row, a, b in (('cold',cold_row,before,cold_h),('warm',warm_row,cold_h,warm_h)):
            if args.paged_pages:
                sa, sb = a['paged_kv_pool']['statistics'], b['paged_kv_pool']['statistics']
                check(name+'_dense_fallback_without_long_import', no_native_delta(a,b)
                      and row.get('kv_append_mode') == 'reference' and row.get('paged_kv_capacity_fallbacks') == 1
                      and all(row.get(k) == 0 for k in ('paged_kv_token_steps','paged_kv_reused_prefix_tokens',
                                                       'paged_kv_imported_suffix_rows','paged_kv_reserved_pages_per_layer'))
                      and sb['completed_capacity_fallbacks']-sa['completed_capacity_fallbacks'] == 1)
            else:
                check(name+'_dense_default', row.get('kv_append_mode') == 'reference'
                      and row.get('paged_kv_token_steps') is None and row.get('paged_kv_capacity_fallbacks') is None)
        decode32_first = None
        if decode_fixture is not None:
            decode32_first = extended_decode_case('decode32_json', False, warm_h)
        # A distinct32K prefix avoids an already-warm262K request completing
        # before the disconnect. Wait for resident prefill ownership, then RST.
        event('cancel_prefill_start', prompt_tokens=fixture['requests']['cancel']['prompt_tokens'])
        record_wire('cancel',payloads['cancel'])
        sock = harness.post_socket(payloads['cancel'])
        response = http.client.HTTPResponse(sock)
        try:
            response.begin()
            require(response.status == 200 and 'text/event-stream' in response.getheader('Content-Type',''), 'Cancellation SSE role missing')
            pending = bytearray()
            while not pending.endswith(b'\n\n'):
                line = response.readline(8193)
                require(line and len(pending)+len(line) < 16384, 'Invalid cancellation role frame')
                pending.extend(line)
            require(pending.startswith(b'data: '), 'Invalid SSE frame prefix')
            frame = json.loads(bytes(pending[6:]).strip())
            require(frame['choices'][0]['delta'].get('role') == 'assistant' and frame['choices'][0]['finish_reason'] is None, 'Cancellation did not begin with role')
            identity = frame['id']
            (out/'cancel.role.sse').write_bytes(pending)
            report['cancel_role_frame'] = frame
            report['wire_evidence']['cancel'].update(response_status=response.status,response_headers=dict(response.getheaders()),
                observed_body_prefix_bytes=len(pending),observed_body_prefix_sha256=hashlib.sha256(pending).hexdigest())
            resident = harness.until(lambda item: item[0] == 200 and item[1].get('resident_sequences') == 1
                and item[1].get('reserved_tokens') == fixture['requests']['cancel']['prompt_tokens']+32, timeout=60)
            report['health']['cancel_resident'] = resident; report['cancel_request_id'] = identity; save()
        finally:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii',1,0))
            response.close(); harness.close(sock, reset=True)
        settled('after_cancel', timeout=120)
        cancel_row = terminal(identity)
        check('prefill_cancelled_and_reclaimed', cancel_row.get('model_kind') == 'cancelled' and cancel_row.get('stage') == 'prefill')
        fresh, _, fresh_h = request_case('short_after_cancel', payloads['short'], fixture['requests']['short']['prompt_tokens'], 0)
        check('short_recovery_exact', short['text'] == fresh['text'] and short['usage'] == fresh['usage'] and short['finish'] == fresh['finish'])
        if decode_fixture is not None:
            decode32_after_cancel = extended_decode_case('decode32_sse_after_cancel', True, fresh_h)
            check('decode32_repeated_output_exact', decode32_first['text'] == decode32_after_cancel['text']
                  and decode32_first['finish'] == decode32_after_cancel['finish'] == 'length'
                  and decode32_first['usage'] == decode32_after_cancel['usage'])
        final = settled('final')
        _, _, metrics, _ = harness.request('GET','/metrics',timeout=5)
        (out/'final.prometheus.txt').write_bytes(metrics)
        dump(out/'lifecycle.records.json', log_records())
        report['server_log_sha256_at_finish'] = sha(args.server_log)
        report['complete'] = True
        report['passed'] = all(c['passed'] for c in report['checks'])
        event('gate_complete', passed=report['passed'])
    except BaseException as error:
        report['error'] = type(error).__name__+': '+str(error)
        report['passed'] = False
        event('gate_failed', error=report['error'])
    finally:
        harness.close_all()
        signal.signal(signal.SIGTERM, old_term)
        save(); progress.close()
    return 0 if report['passed'] else 1


def main():
    require(__debug__, 'Python optimization disables assertions in the reused HTTP helper; run without -O/PYTHONOPTIMIZE')
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--runner',type=Path,required=True); p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--context-limit',type=int,choices=(32768,65536,131072,262144),default=262144)
    r = sub.add_parser('run')
    r.add_argument('--runner',type=Path,required=True); r.add_argument('--fixture',type=Path,required=True)
    r.add_argument('--server-pid',type=int,required=True); r.add_argument('--server-log',type=Path,required=True)
    r.add_argument('--port',type=int,required=True); r.add_argument('--output',type=Path,required=True)
    r.add_argument('--paged-pages',type=int,choices=(0,512),default=0)
    r.add_argument('--prefill-attention',choices=('reference','fusedQSA'),required=True)
    r.add_argument('--timeout',type=int,default=3600)
    r.add_argument('--decode-fixture',type=Path,help='Optional separately prepared P262112/O32 extension; context262144 only')
    args=parser.parse_args()
    if args.action=='prepare':
        prepare(args); return 0
    require(1024 <= args.port <= 65535 and args.server_pid > 1 and 60 <= args.timeout <= 3600, 'Invalid client limits')
    return run(args)

if __name__=='__main__':
    raise SystemExit(main())
