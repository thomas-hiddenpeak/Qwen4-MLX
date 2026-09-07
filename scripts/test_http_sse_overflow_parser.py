"""Small CPU-only controls; no process/server/GPU is started by this file."""
import copy
import io
import json
from pathlib import Path
import sys
import time
import unittest

sys.dont_write_bytecode = True
from test_http_sse_overflow import (Transcript, strict_json, next_frame,
                                    observation, validate_overflow, FIXTURE)
from test_http_terminal_log_validator import record, RID
from test_http_terminal_logs import parse_record

MODEL = 'fixture-model'
DONE = b'data: [DONE]\n\n'


def frame(delta, reason=None, **extras):
    obj = {'object': 'chat.completion.chunk', 'id': RID, 'model': MODEL, 'created': 1,
           'choices': [{'index': 0, 'delta': delta, 'finish_reason': reason}], **extras}
    return b'data: ' + json.dumps(obj, ensure_ascii=False).encode() + b'\n\n'


def prefix():
    t = Transcript(MODEL)
    for delta in ({'role': 'assistant'}, {'content': '山'}, {'content': '川'}):
        t.add(frame(delta))
    return t


ERROR = b'data: {"error":{"code":"slow_consumer","type":"server_error"}}\n\n'


def logs():
    output, model, close = [record(k) for k in ('output_terminal', 'model_terminal', 'connection_close')]
    for r in (output, model, close):
        r.update(mtp_depth=0, output_outcome='slowConsumer', cancellation_requested=True)
    output.update(reason='slow_consumer', error_code='slow_consumer', text_bytes=None,
                  text_limit_bytes=None, buffered_bytes=4000, buffered_events=12, queued_events=11,
                  has_in_flight=True, in_flight_bytes=250, uptime_seconds=10, producer_finished=False)
    model.update(model_kind='cancelled', stage='decode', scheduler_elapsed_seconds=10, uptime_seconds=10.1)
    # Fast network may close before the scheduler's late finish; final idle and
    # unique model terminal establish producer completion separately.
    close.update(reason='terminal_sent', transport_closed=True, output_drained=True,
                 producer_finished=False, uptime_seconds=10.2)
    for r in (output, model, close):
        parse_record(r, 123)
    return {'records': [close, output, model], 'legacy': {RID: [
        f'HTTP request id={RID} terminal=cancelled stage=decode scheduler_elapsed_seconds=10 event=model_terminal']}}


class ParserControls(unittest.TestCase):
    def test_accepted_received_prefix_then_one_error_done(self):
        t = prefix(); before = tuple(t.wires)
        t.add(frame({'content': '日月'})); t.add(ERROR); t.add(DONE)
        self.assertEqual(tuple(t.wires[:len(before)]), before)
        self.assertEqual(''.join(t.contents), '山川日月')
        self.assertTrue(t.done)
        self.assertEqual(t.error['code'], 'slow_consumer')

    def test_duplicate_or_after_terminal_rejected(self):
        for suffix in (ERROR, frame({'content': '日'})):
            t = prefix(); t.add(ERROR)
            with self.assertRaises(ValueError): t.add(suffix)
        t = prefix(); t.add(ERROR); t.add(DONE)
        with self.assertRaises(ValueError): t.add(DONE)

    def test_identity_and_order_rejected(self):
        with self.assertRaises(ValueError): Transcript(MODEL).add(frame({'content': '山'}))
        t = prefix()
        with self.assertRaises(ValueError): t.add(frame({'content': '日'}, id=RID.replace('1111', '2222')))
        with self.assertRaises(ValueError): prefix().add(DONE)

    def test_utf8_duplicate_keys_and_nonfinite_rejected(self):
        with self.assertRaises(UnicodeDecodeError): prefix().add(b'data: {"error":"\xff"}\n\n')
        for raw in ('{"a":1,"a":2}', '{"a":NaN}'):
            with self.assertRaises(ValueError): strict_json(raw)

    def test_complete_lines_and_truncated_eof(self):
        class Sock:
            def settimeout(self, _): pass
        wire = frame({'content': '山川'})
        self.assertEqual(next_frame(io.BytesIO(wire), Sock(), time.monotonic()+1), wire)
        with self.assertRaises(ValueError): next_frame(io.BytesIO(wire[:-1]), Sock(), time.monotonic()+1)

    def test_real_log_selection_and_fast_close(self):
        p = logs()
        self.assertEqual(observation(p, RID)[0], 'observed')
        validate_overflow(p, RID)
        self.assertIsNone(observation(p, RID+'x')[0])

    def test_deadline_or_natural_completion_is_not_overflow(self):
        p = logs(); p['records'][0].update(reason='send_deadline', uptime_seconds=9)
        self.assertEqual(observation(p, RID)[0], 'observed')
        with self.assertRaises(ValueError): validate_overflow(p, RID)
        p['records'] = [p['records'][0]]
        p['records'][0]['output_outcome'] = 'disconnected'
        self.assertEqual(observation(p, RID)[0], 'not_triggered')
        p = logs(); p['records'] = [p['records'][1]]
        p['records'][0].update(reason='completed', output_outcome='completed')
        self.assertEqual(observation(p, RID)[0], 'not_triggered')

    def test_bad_log_identity_duplicate_terminal_quota_and_legacy(self):
        for change in ('identity', 'duplicate', 'quota', 'legacy', 'outcome'):
            p = logs()
            if change == 'identity': p['records'][0]['connection_id'] = 'different'
            if change == 'duplicate': p['records'].append(copy.deepcopy(p['records'][1]))
            if change == 'quota': p['records'][0]['buffered_events'] = 1
            if change == 'legacy': p['legacy'][RID] *= 2
            if change == 'outcome': p['records'][0]['output_outcome'] = 'cancelled'
            with self.assertRaises(ValueError): validate_overflow(p, RID)

    def test_frozen_http_template_and_context(self):
        messages = strict_json((FIXTURE/'messages.json').read_text())
        tok = strict_json((FIXTURE/'tokenization.json').read_text())
        ids = strict_json((FIXTURE/'prompt-token-ids.json').read_text())
        rendered = '<|im_start|>user\n'+messages[0]['content'].strip()+'<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        self.assertEqual(tok['rendered_prompt'], rendered)
        self.assertEqual(tok['tokens'], ids)
        self.assertEqual(len(ids), 4039)
        self.assertLessEqual(len(ids)+4096, 16384)


if __name__ == '__main__':
    unittest.main()
