"""CPU-only parsing controls, never evidence of live model/network coverage."""
import copy
import sys
import unittest

sys.dont_write_bytecode = True
from test_http_terminal_logs import parse_record, validate_request

RID = "chatcmpl-11111111-1111-1111-1111-111111111111"
CID = "22222222-2222-2222-2222-222222222222"


def record(event):
    return {"schema": "qwen-http-lifecycle-v1", "pid": 123, "event": event, "connection_id": CID,
            "request_id": RID, "reason": None, "uptime_seconds": 1, "stream": True, "mtp_depth": 2,
            "output_outcome": None, "buffered_bytes": 0, "buffered_events": 0, "queued_events": 0,
            "in_flight_bytes": 0, "has_in_flight": False, "producer_finished": True,
            "transport_closed": False, "cancellation_requested": False, "output_drained": False}


def rst_case(in_flight=False):
    model = record("model_terminal")
    model.update(model_kind="cancelled", stage="decode", scheduler_elapsed_seconds=1)
    close = record("connection_close")
    close.update(reason="receive_failed", output_outcome="disconnected", transport_closed=True,
                 connection_age_seconds=2, send_elapsed_seconds=None, lease_id=None, send_kind="none",
                 rejection_status=None, rejection_code=None, output_outcome_before_close=None)
    if in_flight:
        close.update(has_in_flight=True, in_flight_bytes=10, buffered_bytes=10, buffered_events=1, lease_id=1, send_kind="body")
    expectation = {"id": RID, "kind": "rst", "depth": 2, "stream": True}
    old = f"HTTP request id={RID} terminal=cancelled stage=decode scheduler_elapsed_seconds=1 event=model_terminal"
    return {"records": [close, model], "legacy": {RID: [old]}}, expectation


class LogValidatorTests(unittest.TestCase):
    def testNormalUsageAndCompletedOutputAreValidatedIndependentOfOrder(self):
        model = record("model_terminal")
        model.update(model_kind="completed", stage="decode", scheduler_elapsed_seconds=1,
                     model_finish_reason="length", prompt_tokens=35, completion_tokens=4, prefill_seconds=.1, decode_seconds=.2)
        output = record("output_terminal")
        output.update(reason="completed", output_outcome="completed", error_code=None, text_bytes=None, text_limit_bytes=None)
        close = record("connection_close")
        close.update(reason="terminal_sent", output_outcome="completed", transport_closed=True)
        parsed = {"records": [close, output, model], "legacy": {RID: ["one legacy model event"]}}
        expectation = {"id": RID, "kind": "normal", "depth": 2, "stream": True,
                       "result": {"text": "1,2,", "finish": "length", "usage": {"prompt_tokens": 35, "completion_tokens": 4}}}
        for item in parsed["records"]:
            parse_record(item, 123)
        validate_request(parsed, expectation)
        model["completion_tokens"] = 3
        with self.assertRaises(ValueError):
            validate_request(parsed, expectation)

    def testDisconnectWithoutOutputTerminalAndAsynchronousRecordOrder(self):
        parsed, expectation = rst_case()
        for item in parsed["records"]:
            parse_record(item, 123)
        self.assertEqual(validate_request(parsed, expectation)["request_id"], RID)
        parsed["records"].reverse()
        self.assertEqual(validate_request(parsed, expectation)["request_id"], RID)

    def testRetainedLeaseRequiresActualMatchingRelease(self):
        parsed, expectation = rst_case(in_flight=True)
        with self.assertRaises(ValueError):
            validate_request(parsed, expectation)
        release = record("closed_send_released")
        release.update(reason="send_failed", lease_id=1, output_outcome="disconnected", transport_closed=True)
        parsed["records"].append(release)
        parse_record(release, 123)
        validate_request(parsed, expectation)
        release["lease_id"] = 2
        with self.assertRaises(ValueError):
            validate_request(parsed, expectation)

    def testFiniteSchemaRejectsBodyAndBadCountersAndSSETextBudget(self):
        output = record("output_terminal")
        output.update(reason="completed", output_outcome="completed", error_code=None, text_bytes=None, text_limit_bytes=None)
        parse_record(output, 123)
        for change in ({"content": "must not enter logs"}, {"reason": "free-form exception"},
                       {"buffered_bytes": True}, {"uptime_seconds": float("nan")},
                       {"text_bytes": 0, "text_limit_bytes": 6144}):
            bad = copy.deepcopy(output)
            bad.update(change)
            with self.assertRaises(ValueError):
                parse_record(bad, 123)

    def testDeadlineCannotMasqueradeAsRSTAndLegacyGateRemains(self):
        parsed, expectation = rst_case()
        parsed["records"][0]["reason"] = "send_deadline"
        with self.assertRaises(ValueError):
            validate_request(parsed, expectation)
        parsed, expectation = rst_case()
        parsed["legacy"][RID] *= 2
        with self.assertRaises(ValueError):
            validate_request(parsed, expectation)

    def testTextLimitDistinguishesGenerationFromFinalUTF8Flush(self):
        model = record("model_terminal")
        model.update(stream=False, mtp_depth=0, model_kind="failed", reason="text_limit", stage="decode", scheduler_elapsed_seconds=1)
        output = record("output_terminal")
        output.update(stream=False, mtp_depth=0, reason="text_limit", output_outcome="failed",
                      text_bytes=6144, text_limit_bytes=6144, error_code="output_limit")
        close = record("connection_close")
        close.update(stream=False, mtp_depth=0, reason="terminal_sent", output_outcome="failed", transport_closed=True)
        parsed = {"records": [model, output, close], "legacy": {RID: ["one legacy model event"]}}
        expectation = {"id": RID, "kind": "output_limit", "depth": 0, "stream": False}
        self.assertEqual(validate_request(parsed, expectation)["output_limit_phase"], "generation_text_limit")
        model["reason"] = None
        with self.assertRaises(ValueError):
            validate_request(parsed, expectation)
        model["model_kind"] = "completed"
        self.assertEqual(validate_request(parsed, expectation)["output_limit_phase"], "completion_utf8_flush_text_limit")


if __name__ == "__main__":
    unittest.main()
