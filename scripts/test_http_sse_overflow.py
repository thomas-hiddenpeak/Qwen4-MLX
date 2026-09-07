#!/usr/bin/env python3
"""One fixed real SSE paused-reader attempt; only run under the GPU controller."""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / 'Package.swift').is_file())
FIXTURE = REPO / 'fixtures/http-sse-overflow'
sys.path.insert(0, str(REPO / 'scripts'))
from test_http_server_edges import HTTPHarness, decode_completion
from test_http_terminal_logs import (NUMBERS_PROMPT, read_log, require,
                                    require_complete_logging, validate_request)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def constant(_):
        raise ValueError('Nonfinite JSON')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


class Transcript:
    """Complete received frames, with strict UTF-8 and one terminal tail."""
    def __init__(self, model):
        self.model, self.identity = model, None
        self.wires, self.frames, self.contents = [], [], []
        self.error = self.finish = None
        self.done = False
        self.bytes = 0

    def add(self, wire):
        require(not self.done, 'Frame after DONE')
        require(wire.endswith(b'\n\n') or wire.endswith(b'\r\n\r\n'), 'Incomplete SSE frame')
        self.bytes += len(wire)
        require(self.bytes <= 2 * 1024 * 1024 and len(self.wires) < 8192, 'SSE transcript bound exceeded')
        line = wire.rstrip(b'\r\n')
        require(line.startswith(b'data: ') and b'\n' not in line, 'Unexpected SSE field/line')
        payload = line[6:]
        if payload == b'[DONE]':
            require(self.error is not None or self.finish is not None, 'DONE before terminal')
            self.done = True
            self.wires.append(wire)
            return
        obj = strict_json(payload.decode('utf-8', errors='strict'))
        require(isinstance(obj, dict), 'SSE payload is not an object')
        require(self.error is None and self.finish is None, 'Duplicate terminal or frame after terminal')
        if 'error' in obj:
            require(self.identity is not None and set(obj) == {'error'}, 'Error before request identity')
            require(isinstance(obj['error'], dict) and isinstance(obj['error'].get('code'), str), 'Invalid SSE error')
            self.error = obj['error']
        else:
            require(obj.get('object') == 'chat.completion.chunk' and obj.get('model') == self.model, 'Wrong SSE object/model')
            identity = (obj.get('id'), obj.get('created'), obj.get('model'))
            require(isinstance(identity[0], str) and identity[0].startswith('chatcmpl-')
                    and type(identity[1]) is int, 'Invalid SSE identity')
            require(isinstance(obj.get('choices'), list) and len(obj['choices']) == 1, 'Invalid choices')
            choice = obj['choices'][0]
            require(choice.get('index') == 0 and isinstance(choice.get('delta'), dict), 'Invalid choice')
            delta, reason = choice['delta'], choice.get('finish_reason')
            if self.identity is None:
                require(delta == {'role': 'assistant'} and reason is None, 'First frame must declare assistant')
                self.identity = identity
            else:
                require(identity == self.identity, 'Request identity changed')
                if reason is None:
                    require(set(delta) == {'content'} and isinstance(delta['content'], str)
                            and len(delta['content']) > 0, 'Invalid/noncontent middle frame')
                    self.contents.append(delta['content'])
                else:
                    require(reason in ('stop', 'length') and delta == {} and isinstance(obj.get('usage'), dict), 'Invalid finish frame')
                    self.finish = obj
        self.frames.append(obj)
        self.wires.append(wire)

    def summary(self):
        return {'request_id': self.identity[0] if self.identity else None,
                'content_frames': len(self.contents), 'received_body_bytes': self.bytes,
                'wire_sha256': hashlib.sha256(b''.join(self.wires)).hexdigest(),
                'text_sha256': hashlib.sha256(''.join(self.contents).encode()).hexdigest(),
                'done': self.done, 'error': self.error, 'finish': self.finish}


def next_frame(response, sock, deadline):
    pending = bytearray()
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Bounded SSE read deadline')
        sock.settimeout(remaining)
        line = response.readline(8193)
        require(len(line) <= 8192, 'Oversized SSE line')
        if not line:
            require(not pending, 'EOF inside SSE frame')
            return None
        pending.extend(line)
        require(len(pending) <= 16384, 'Oversized SSE frame')
        if line in (b'\n', b'\r\n'):
            return bytes(pending)


def observation(parsed, rid):
    records = [r for r in parsed['records'] if r['request_id'] == rid]
    slow = [r for r in records if r['event'] == 'output_terminal'
            and r['reason'] == 'slow_consumer' and r['output_outcome'] == 'slowConsumer']
    require(len(slow) <= 1, 'Duplicate overflow log')
    closes = [r for r in records if r['event'] == 'connection_close']
    require(len(closes) <= 1, 'Duplicate close log')
    if slow:
        # Selection is authoritative. Cross-thread log timestamps/order can lag
        # the buffer operation; a later deadline makes delivery fail, not erase
        # an already observed overflow.
        require(slow[0]['stream'] is True and slow[0]['mtp_depth'] == 0, 'Overflow configuration mismatch')
        return 'observed', {'stop_reason': 'slow_consumer', 'output_terminal': slow[0]}
    if closes:
        # An already-selected overflow record may still be queued in the async
        # logger. Wait for its positive evidence instead of inventing a verdict.
        if closes[0]['output_outcome'] == 'slowConsumer':
            return None, None
        return 'not_triggered', {'stop_reason': closes[0]['reason'], 'close': closes[0]}
    # Model completion precedes the final UTF-8 publish; only an actual output
    # selection/close decides this observation, not an earlier model event.
    terminals = [r for r in records if r['event'] == 'output_terminal']
    if terminals:
        return 'not_triggered', {'stop_reason': 'natural_or_other_terminal', 'terminal': terminals[0]}
    return None, None


def validate_overflow(parsed, rid):
    records = [r for r in parsed['records'] if r['request_id'] == rid]
    grouped = {name: [r for r in records if r['event'] == name]
               for name in ('output_terminal', 'model_terminal', 'connection_close', 'closed_send_released')}
    require(all(len(grouped[k]) == 1 for k in ('output_terminal', 'model_terminal', 'connection_close')), 'Need unique overflow/model/close records')
    output, model, close = [grouped[k][0] for k in ('output_terminal', 'model_terminal', 'connection_close')]
    require(len({r['connection_id'] for r in records}) == 1
            and all(r['stream'] is True and r['mtp_depth'] == 0 for r in records), 'Lifecycle identity/configuration mismatch')
    require(output['reason'] == 'slow_consumer' and output['output_outcome'] == 'slowConsumer'
            and output['error_code'] == 'slow_consumer' and output['cancellation_requested'], 'Overflow selection missing')
    require(output['buffered_bytes'] <= 8192 and output['buffered_events'] <= 256,
            'Overflow snapshot exceeded configured quota')
    require(model['model_kind'] == 'cancelled' and model['stage'] == 'decode', 'Overflow did not cancel decode')
    legacy = parsed['legacy'].get(rid, [])
    require(len(legacy) == 1 and 'terminal=cancelled stage=decode ' in legacy[0], 'Legacy decode-cancel gate lost')
    require(close['reason'] == 'terminal_sent' and close['output_outcome'] == 'slowConsumer'
            and close['transport_closed'] and close['output_drained'], 'Overflow did not preserve terminal outcome through drain')
    require(close['buffered_bytes'] == close['buffered_events'] == close['in_flight_bytes'] == 0
            and close['has_in_flight'] is False and not grouped['closed_send_released'], 'Normal terminal close retained quota')
    return {'records': records, 'legacy': legacy}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('runner', 'model-dir', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--port', type=int, default=11240)
    args = parser.parse_args()
    require(1024 <= args.port <= 65535, 'Invalid port')
    out, log, wire_path = args.output.resolve(), args.output.resolve().with_suffix('.server.log'), args.output.resolve().with_suffix('.sse')
    require(not any(p.exists() for p in (out, log, wire_path)), 'All output artifacts must be new')
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = strict_json((FIXTURE / 'manifest.json').read_text())
    for name, sha in manifest['input_files_sha256'].items():
        require(digest(FIXTURE / name) == sha, 'Frozen input changed: ' + name)
    for name, sha in manifest['tokenizer_metadata_sha256'].items():
        require(digest(args.model_dir / name) == sha, 'Tokenizer metadata changed: ' + name)
    for name, sha in manifest['helper_sha256'].items():
        require(digest(REPO / 'scripts' / name) == sha, 'Reviewed HTTP helper changed: ' + name)
    messages = strict_json((FIXTURE / 'messages.json').read_text())
    ids = strict_json((FIXTURE / 'prompt-token-ids.json').read_text())
    require(all(type(x) is int for x in ids) and len(ids) == manifest['prompt_tokens'] == 4039
            and len(ids) + 4096 <= 16384, 'Frozen context budget invalid')
    model = args.model_dir.resolve().name
    report = {'schema': 'qwen-http-sse-overflow-gate-v1', 'complete': False, 'passed': False,
              'checks': [], 'requests': [], 'attempt': {'status': 'not_run'},
              'manifest_sha256': digest(FIXTURE / 'manifest.json'), 'script_sha256': digest(__file__),
              'helper_sha256': {name: digest(REPO / 'scripts' / name) for name in ('test_http_server_edges.py', 'test_http_terminal_logs.py')},
              'runner_sha256': digest(args.runner), 'frozen_prompt_tokens': len(ids),
              'scope': 'One AR4096 attempt; received-prefix/terminal/lifecycle only; no complete enqueue-prefix or performance claim.'}
    child = harness = response = sock = transcript = None
    guard, pending_signal = False, None
    expectations = []
    def save():
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    def check(name, valid, **evidence):
        report['checks'].append({'id': name, 'passed': bool(valid), **evidence})
        save()
        require(valid, name)
    def interrupted(number, _):
        nonlocal pending_signal
        if guard:
            pending_signal = number
            return
        raise InterruptedError('Fixed work deadline' if number == signal.SIGALRM else 'Harness signal ' + str(number))
    previous = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM)}
    timer = signal.setitimer(signal.ITIMER_REAL, 360)
    started = time.monotonic()
    try:
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', args.port))
        command = [str(args.runner.resolve()), 'serve-gpu', '--model-dir', str(args.model_dir.resolve()),
                   '--port', str(args.port), '--max-connections', '4', '--output-buffer-bytes', '8192']
        with log.open('xb') as sink:
            guard = True
            try:
                child = subprocess.Popen(command, cwd=REPO, stdout=sink, stderr=subprocess.STDOUT)
            finally:
                guard = False
        if pending_signal is not None:
            interrupted(pending_signal, None)
        report.update(command=command, server_pid=child.pid, harness_pgid=os.getpgrp(), server_pgid=os.getpgid(child.pid))
        require(report['harness_pgid'] == report['server_pgid'], 'Server escaped controller-owned group')
        harness = HTTPHarness(args.port, child)
        report['ready'] = harness.until(lambda h: h[0] == 200 and h[1].get('ready') is True, timeout=180)
        require_complete_logging(report['ready'])
        body = {'model': model, 'messages': messages, 'max_tokens': 4096, 'mtp_depth': 0, 'stream': True}
        payload = json.dumps(body, ensure_ascii=False).encode()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        harness.sockets.add(sock)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
        report['receive_buffer_actual_bytes'] = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        sock.settimeout(5)
        sock.connect(('127.0.0.1', args.port))
        sock.sendall(('POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n'
                      f'Content-Length: {len(payload)}\r\n\r\n').encode() + payload)
        transcript = Transcript(model)
        response = http.client.HTTPResponse(sock)
        sock.settimeout(120)
        response.begin()
        require(response.status == 200 and 'text/event-stream' in response.getheader('Content-Type', ''), 'Attempt did not obtain SSE HTTP200')
        precondition_end = time.monotonic() + 120
        while len(transcript.contents) < 2 and not transcript.done:
            wire = next_frame(response, sock, precondition_end)
            require(wire is not None, 'EOF before precondition terminal')
            transcript.add(wire)
        rid = transcript.identity[0]
        report['attempt'].update(status='not_triggered', request_id=rid, prefix=transcript.summary(), paused=False)
        before = tuple(transcript.wires)
        if len(transcript.contents) >= 2:
            paused_at = time.monotonic()
            report['attempt']['paused'] = True
            report['attempt']['pause_started_monotonic_seconds'] = paused_at
            while True:
                harness.alive()
                parsed = read_log(log, child.pid, model)
                status, evidence = observation(parsed, rid)
                if status is not None:
                    report['attempt'].update(status=status, observation=evidence)
                    break
                if time.monotonic() - paused_at >= 150:
                    report['attempt']['observation'] = {'stop_reason': 'fixed_pause_cap'}
                    break
                time.sleep(.05)
            # Resume immediately after observing the log; save only after this
            # boundary. HTTPResponse's already-buffered bytes remain intact.
            resumed_at = time.monotonic()
            report['attempt'].update(pause_seconds=resumed_at - paused_at,
                                     resume_started_monotonic_seconds=resumed_at)
        else:
            report['attempt']['observation'] = {'stop_reason': 'natural_terminal_before_two_content_frames'}
        try:
            end = time.monotonic() + 45
            while not transcript.done:
                wire = next_frame(response, sock, end)
                if wire is None:
                    break
                transcript.add(wire)
            trailing = next_frame(response, sock, end)
            require(trailing is None, 'Bytes/frames after terminal')
            report['attempt']['normal_eof'] = True
        except (OSError, TimeoutError, ValueError, http.client.HTTPException) as error:
            report['attempt']['read_error'] = type(error).__name__ + ': ' + str(error)
        finally:
            response.close(); response = None
            harness.close(sock); sock = None
            wire_path.write_bytes(b''.join(transcript.wires))
            report['attempt']['received'] = transcript.summary()
            report['attempt']['prefix_preserved'] = tuple(transcript.wires[:len(before)]) == before
            save()
        report['attempt']['idle'] = harness.idle()
        require_complete_logging(report['attempt']['idle'])
        # An honest non-trigger still tests fresh recovery once; never reattempt.
        for depth, stream in ((0, False), (2, True)):
            body = {'model': model, 'messages': NUMBERS_PROMPT, 'max_tokens': 4, 'mtp_depth': depth, 'stream': stream}
            result = decode_completion(harness.request('POST', '/v1/chat/completions', body, timeout=60), stream)
            expected = {'kind': 'normal', 'id': result['id'], 'depth': depth, 'stream': stream, 'result': result}
            expectations.append(expected); report['requests'].append(expected)
            check('fresh_' + str(depth), result['text'] == '1,2,' and result['finish'] == 'length'
                  and result['usage'] == {'prompt_tokens': 35, 'completion_tokens': 4, 'total_tokens': 39}, result=result)
        final = harness.idle()
        report['logging_health'] = require_complete_logging(final)
        end, last = time.monotonic() + 10, None
        while time.monotonic() < end:
            parsed = read_log(log, child.pid, model)
            try:
                fresh = [validate_request(parsed, expected) for expected in expectations]
                attempt_records = [r for r in parsed['records'] if r['request_id'] == rid]
                require(sum(r['event'] == 'model_terminal' for r in attempt_records) == 1
                        and sum(r['event'] == 'connection_close' for r in attempt_records) == 1, 'Attempt terminal logs pending')
                report['attempt']['terminal_records'] = attempt_records
                if report['attempt']['status'] == 'observed':
                    report['attempt']['validated_logs'] = validate_overflow(parsed, rid)
                require({r['request_id'] for r in parsed['records'] if r['request_id']} == {rid, *(x['id'] for x in expectations)}, 'Unexpected request ID')
                report['fresh_log_evidence'] = fresh
                break
            except ValueError as error:
                last = str(error)
                time.sleep(.05)
        else:
            raise TimeoutError('Terminal log validation: ' + str(last))
        report['logging_health'] = require_complete_logging(harness.idle())
        check('fresh_terminal_logs_and_idle', True)
        if report['attempt']['status'] == 'observed':
            check('real_overflow_received_prefix_and_terminal', transcript.done and transcript.finish is None
                  and transcript.error is not None and transcript.error['code'] == 'slow_consumer'
                  and transcript.error.get('type') == 'server_error'
                  and report['attempt']['prefix_preserved'] and report['attempt'].get('normal_eof') is True
                  and 'read_error' not in report['attempt'])
        # Recovery is still checked above, but a nested protocol/read failure
        # must not become a normally completed, non-triggered attempt.
        require('read_error' not in report['attempt'], 'Attempt SSE read failed; see attempt.read_error')
        report['complete'] = True
        report['passed'] = report['attempt']['status'] == 'observed' and all(c['passed'] for c in report['checks'])
    except BaseException as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
        report['passed'] = False
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for s in (signal.SIGTERM, signal.SIGINT):
            signal.signal(s, signal.SIG_IGN)
        try:
            try:
                if response is not None:
                    response.close()
                if sock is not None:
                    sock.close()
                if harness is not None:
                    harness.close_all()
                if transcript is not None:
                    wire_path.write_bytes(b''.join(transcript.wires))
                    report['partial_received'] = transcript.summary()
            except BaseException as error:
                # A response/file cleanup error cannot bypass owned-child TERM.
                report['artifact_cleanup_error'] = type(error).__name__ + ': ' + str(error)
                report['passed'] = False
            if child is not None and child.poll() is None:
                child.send_signal(signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    report['forced_shutdown'] = True
                    child.kill(); child.wait(timeout=10)
            if child is not None:
                report['server_exit_code'] = child.poll()
                report['graceful_shutdown'] = child.returncode == 0 and not report.get('forced_shutdown', False)
                report['passed'] = report['passed'] and report['graceful_shutdown']
        except BaseException as error:
            report['cleanup_error'] = type(error).__name__ + ': ' + str(error)
            report['passed'] = False
        finally:
            report['harness_seconds'] = time.monotonic() - started
            save()
            for s, previous_handler in previous.items():
                signal.signal(s, previous_handler)
            signal.setitimer(signal.ITIMER_REAL, *timer)
    print(json.dumps({key: report.get(key) for key in ('complete', 'passed', 'error', 'graceful_shutdown')}))
    if report['passed']:
        return 0
    if (report['complete'] and report['attempt']['status'] == 'not_triggered'
            and report.get('graceful_shutdown') is True
            and 'read_error' not in report['attempt']
            and not any(k.endswith('error') for k in report)):
        return 2
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
