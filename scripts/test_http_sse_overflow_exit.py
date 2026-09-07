"""Exercise actual main classification with all process/network operations mocked."""
import contextlib
import io
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

import test_http_sse_overflow as gate

RID = 'chatcmpl-11111111-1111-1111-1111-111111111111'
MODEL = 'fixture-model'


def frame(delta, reason=None):
    obj = {'id': RID, 'created': 1, 'model': MODEL, 'object': 'chat.completion.chunk',
           'choices': [{'index': 0, 'delta': delta, 'finish_reason': reason}]}
    if reason is not None:
        obj['usage'] = {'prompt_tokens': 4039, 'completion_tokens': 2, 'total_tokens': 4041}
    return b'data: '+json.dumps(obj).encode()+b'\n\n'


class MainExitControls(unittest.TestCase):
    def exercise(self, malformed):
        manifest = gate.strict_json((gate.FIXTURE/'manifest.json').read_text())
        logging = dict(accepting=True, writer_exited=False, max_bytes=65536, max_events=128,
            max_event_bytes=4096, buffered_bytes=0, buffered_events=0, queued_events=0,
            in_flight_bytes=0, enqueued_events=0, written_events=0, dropped_events=0,
            dropped_bytes=0, write_failures=0, last_write_errno=None)
        health = dict(ready=True, idle=True, logging=logging)
        # read_log/validate_request are separately parser-tested. This fixture
        # supplies their already-validated decisions to exercise main itself.
        records = [dict(request_id=RID, event='model_terminal'),
                   dict(request_id=RID, event='connection_close', reason='terminal_sent', output_outcome='completed')]

        class Child:
            pid = 321
            returncode = None
            def __init__(self): self.signals = []
            def poll(self): return self.returncode
            def send_signal(self, number):
                self.signals.append(number); self.returncode = 0
            def wait(self, timeout): return self.returncode
            def kill(self): raise AssertionError('Unexpected forced cleanup')

        class Sock:
            def __enter__(self): return self
            def __exit__(self, *_): self.close()
            def bind(self, *_): pass
            def connect(self, *_): pass
            def setsockopt(self, *_): pass
            def getsockopt(self, *_): return 1024
            def settimeout(self, *_): pass
            def sendall(self, *_): pass
            def close(self): pass

        class Response:
            status = 200
            def __init__(self, *_): pass
            def begin(self): pass
            def getheader(self, *_): return 'text/event-stream'
            def close(self): pass

        requests = []
        class Harness:
            def __init__(self, *_): self.sockets = set()
            def until(self, predicate, timeout):
                assert predicate((200, health)); return health
            def alive(self): pass
            def idle(self): return health
            def close(self, sock): self.sockets.discard(sock); sock.close()
            def close_all(self):
                for sock in list(self.sockets): self.close(sock)
            def request(self, method, path, body, timeout):
                requests.append(body)
                rid = 'chatcmpl-fresh-'+str(len(requests))
                records.append(dict(request_id=rid, event='model_terminal'))
                return dict(id=rid, text='1,2,', finish='length',
                    usage={'prompt_tokens': 35, 'completion_tokens': 4, 'total_tokens': 39})

        wires = [frame({'role': 'assistant'}), frame({'content': '山'}), frame({'content': '川'})]
        wires += [b'data: {bad-json}\n\n'] if malformed else [frame({}, 'stop'), b'data: [DONE]\n\n', None]
        original_digest = gate.digest
        with tempfile.TemporaryDirectory(prefix='cpu-main-exit-') as directory:
            out = Path(directory)/'result.json'
            model = Path(directory)/MODEL
            def digest(path):
                path = Path(path)
                if path.parent == model:
                    return manifest['tokenizer_metadata_sha256'][path.name]
                return original_digest(path)
            child = Child()
            with contextlib.ExitStack() as stack:
                replacements = {'HTTPHarness': Harness, 'digest': digest,
                    'decode_completion': lambda result, stream: result,
                    'read_log': lambda *_: {'records': records, 'legacy': {}},
                    'validate_request': lambda parsed, expected: {'request_id': expected['id']}}
                for name, value in replacements.items(): stack.enter_context(patch.object(gate, name, value))
                stack.enter_context(patch.object(gate, 'next_frame', side_effect=wires))
                stack.enter_context(patch.object(gate.socket, 'socket', side_effect=lambda *_: Sock()))
                stack.enter_context(patch.object(gate.http.client, 'HTTPResponse', Response))
                popen = stack.enter_context(patch.object(gate.subprocess, 'Popen', return_value=child))
                stack.enter_context(patch.object(gate.os, 'getpgrp', return_value=99))
                stack.enter_context(patch.object(gate.os, 'getpgid', return_value=99))
                stack.enter_context(patch.object(gate.signal, 'signal', return_value=signal.SIG_DFL))
                stack.enter_context(patch.object(gate.signal, 'setitimer', return_value=(0, 0)))
                stack.enter_context(patch.object(gate.sys, 'argv', ['cpu-test', '--runner', str(gate.HERE/'test_http_sse_overflow.py'),
                    '--model-dir', str(model), '--output', str(out)]))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                status = gate.main()
                popen.assert_called_once()
            report = json.loads(out.read_text())
            self.assertEqual(len(requests), 2, 'Fresh AR/MTP recovery was skipped')
            self.assertEqual(child.signals, [signal.SIGTERM])
            self.assertTrue(report['graceful_shutdown'])
            return status, report

    def test_actual_main_clean_not_triggered_returns_two(self):
        status, report = self.exercise(malformed=False)
        self.assertEqual(status, 2, report)
        self.assertTrue(report['complete'])
        self.assertFalse(report['passed'])
        self.assertNotIn('read_error', report['attempt'])

    def test_actual_main_nested_protocol_error_returns_failure_after_recovery(self):
        status, report = self.exercise(malformed=True)
        self.assertEqual(status, 1, report)
        self.assertEqual(report['attempt']['status'], 'not_triggered')
        self.assertIn('read_error', report['attempt'])
        self.assertIn('error', report)
        self.assertFalse(report['complete'])
        self.assertFalse(report['passed'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
