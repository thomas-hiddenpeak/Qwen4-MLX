"""Offline checks only: no HTTP requests, service operations, or GPU calls."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("capture_gpu_author_reference.py")
SPEC = importlib.util.spec_from_file_location("capture_gpu_author_reference", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CaptureReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = SCRIPT.parents[1] / "fixtures/gpu-full-model-reference"
        cls.response = json.loads((fixture / "completion-response.json").read_text())
        cls.prompt_ids = json.loads((fixture / "prompt-token-ids.json").read_text())
        cls.expected_ids = json.loads((fixture / "generated-token-ids.json").read_text())
        tokenizer = json.loads((MODULE.DEFAULT_MODEL / "tokenizer.json").read_text())
        cls.raw, cls.inverse = MODULE.vocabulary_decoder(tokenizer)

    def recover(self, response=None, inverse=None):
        return MODULE.recover_ids(response or self.response, self.raw, inverse or self.inverse, self.prompt_ids, 64)

    def testOldReferencePreservesEveryExactId(self):
        ids, evidence, text, coverage = self.recover()
        self.assertEqual(ids, self.expected_ids)
        self.assertEqual(len(evidence), 16)
        self.assertEqual(text, self.response["choices"][0]["text"])
        self.assertEqual(coverage["usage_tokens_without_exposed_ids"], 0)

    def testUnexposedStopTokenIsRecordedButNeverInferred(self):
        response = copy.deepcopy(self.response)
        response["choices"][0]["finish_reason"] = "stop"
        response["usage"]["completion_tokens"] += 1
        response["usage"]["total_tokens"] += 1
        ids, _, _, coverage = self.recover(response)
        self.assertEqual(ids, self.expected_ids)
        self.assertEqual(coverage["usage_tokens_without_exposed_ids"], 1)
        self.assertEqual(coverage["inferred_or_appended_token_ids"], [])
        self.assertFalse(coverage["all_usage_tokens_have_exposed_ids"])

    def testLossyAmbiguousOffsetsAndCountsAreRejected(self):
        response = copy.deepcopy(self.response)
        response["choices"][0]["logprobs"]["tokens"][0] = "\ufffd"
        with self.assertRaisesRegex(RuntimeError, "Lossy"):
            self.recover(response)
        response = copy.deepcopy(self.response)
        response["choices"][0]["logprobs"]["text_offset"][1] += 1
        with self.assertRaisesRegex(RuntimeError, "offset"):
            self.recover(response)
        inverse = dict(self.inverse)
        inverse["太阳"] = [99519, 123456]
        with self.assertRaisesRegex(RuntimeError, "2 lossless inverse candidates"):
            self.recover(inverse=inverse)
        response = copy.deepcopy(self.response)
        response["usage"]["completion_tokens"] = 65
        with self.assertRaisesRegex(RuntimeError, "within the requested limit"):
            self.recover(response)

    def testOnlyNewUnambiguousServerLogTimingsAreAccepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            log = directory / "server.log"
            log.write_text("old request must not be captured\n")
            cursor = MODULE.log_cursor(log)
            segment = ("[info] POST /v1/completions (max_tokens=64)\n"
                       "[info]   <- 19+27 tokens (715ms) [prefill: 1000.0 tok/s, decode: 38.8 tok/s] [stop]\n")
            with log.open("a") as stream:
                stream.write(segment)
            usage = {"prompt_tokens": 19, "completion_tokens": 27}
            report = MODULE.capture_log(log, cursor, directory, "one", usage)
            self.assertTrue(report["unambiguous_request_window"])
            self.assertEqual(report["parsed_timing"]["decode_tokens_per_second"], 38.8)
            self.assertEqual((directory / "one-server-log.bin").read_bytes(), segment.encode())
            with log.open("a") as stream:
                stream.write("[info] POST /v1/chat/completions (other client)\n")
            report = MODULE.capture_log(log, cursor, directory, "overlap", usage)
            self.assertFalse(report["unambiguous_request_window"])
            self.assertIsNone(report["parsed_timing"])


if __name__ == "__main__":
    unittest.main()
