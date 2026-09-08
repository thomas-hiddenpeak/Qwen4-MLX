"""CPU-only parser controls; no server, model, process or GPU is started."""
import copy
import json
import sys
import unittest

sys.dont_write_bytecode = True
from probe_http_cache_reliability import parse_completion


HEADERS = {"content-type": "text/event-stream"}
DONE = b"data: [DONE]\n\n"
TEXT = "CACHE_RELIABILITY_0_OK。"


def usage():
    return {"prompt_tokens": 833, "completion_tokens": 8, "total_tokens": 841}


def nonstream(counts):
    return json.dumps({"object": "chat.completion", "id": "cache-parser-request",
        "created": 1, "model": "fixture-model", "choices": [{"index": 0,
            "message": {"role": "assistant", "content": TEXT}, "finish_reason": "length"}],
        "usage": counts}, ensure_ascii=False).encode()


def frame(delta=None, reason=None, **extras):
    value = {"object": "chat.completion.chunk", "id": "cache-parser-request", "created": 1,
        "model": "fixture-model", "choices": [{"index": 0, "delta": delta or {}, "finish_reason": reason}], **extras}
    return b"data: " + json.dumps(value, ensure_ascii=False).encode() + b"\n\n"


def stream(counts, separate_usage=False):
    first = frame({"role": "assistant"}) + frame({"content": TEXT})
    if separate_usage:
        return first + frame(reason="length") + frame(choices=[], usage=counts) + DONE
    return first + frame(reason="length", usage=counts) + DONE


class CacheReliabilityParserTests(unittest.TestCase):
    def test_cold_missing_optional_details_defaults_to_zero_without_mutation(self):
        for details in (None, {}):
            counts = usage()
            if details is not None:
                counts["prompt_tokens_details"] = details
            before = copy.deepcopy(counts)
            for streaming in (False, True):
                raw = stream(counts) if streaming else nonstream(counts)
                result = parse_completion(200, HEADERS, raw, streaming)
                self.assertEqual(result["cached_tokens"], 0)
                self.assertEqual(result["usage"], before)
                self.assertEqual(result["content"], TEXT)

    def test_warm_cache_value_and_unknown_usage_details_preserved(self):
        counts = usage()
        counts["prompt_tokens_details"] = {"cached_tokens": 416, "future_detail": 5}
        counts["completion_tokens_details"] = {"reasoning_tokens": 0}
        for streaming in (False, True):
            result = parse_completion(200, HEADERS, stream(counts) if streaming else nonstream(counts), streaming)
            self.assertEqual(result["cached_tokens"], 416)
            self.assertEqual(result["usage"], counts)

    def test_negative_complete_prefix_excess_and_noninteger_cache_counts_rejected(self):
        for cached in (-1, 833, 834, 1.5, "416", True, None):
            counts = usage(); counts["prompt_tokens_details"] = {"cached_tokens": cached}
            for streaming in (False, True):
                with self.subTest(cached=cached, streaming=streaming), self.assertRaises(ValueError):
                    parse_completion(200, HEADERS, stream(counts) if streaming else nonstream(counts), streaming)

    def test_invalid_required_counts_and_optional_container_rejected(self):
        variants = [None, {}, {**usage(), "prompt_tokens_details": None},
                    {**usage(), "prompt_tokens_details": []}, {**usage(), "prompt_tokens": True},
                    {**usage(), "total_tokens": 840}, {**usage(), "completion_tokens": 17}]
        for counts in variants:
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                parse_completion(200, HEADERS, nonstream(counts), False)

    def test_sse_and_nonstream_match_with_inline_or_separate_final_usage(self):
        for cached in (None, 416):
            counts = usage()
            if cached is not None:
                counts["prompt_tokens_details"] = {"cached_tokens": cached}
            expected = parse_completion(200, {}, nonstream(counts), False)
            for separate in (False, True):
                raw = stream(counts, separate_usage=separate)
                self.assertEqual(parse_completion(200, HEADERS, raw, True), expected)
                self.assertEqual(parse_completion(200, HEADERS, raw.replace(b"\n", b"\r\n"), True), expected)

    def test_missing_duplicate_or_post_terminal_frames_are_rejected(self):
        raw = stream(usage())
        variants = [raw[:-len(DONE)], raw + DONE, raw + frame({"content": "late"}),
                    raw[:-len(DONE)] + frame({"content": "late"}) + DONE,
                    raw[:-len(DONE)] + frame(reason="length") + DONE,
                    raw[:-len(DONE)] + frame(choices=[], usage=usage()) + DONE,
                    frame({"role": "assistant"}) + frame({"content": TEXT}, usage=usage()) + DONE]
        for bad in variants:
            with self.subTest(raw=bad[-200:]), self.assertRaises(ValueError):
                parse_completion(200, HEADERS, bad, True)

    def test_sse_identity_role_usage_and_structured_error_controls(self):
        variants = [frame({"content": TEXT}) + frame(reason="length", usage=usage()) + DONE,
                    frame({"role": "assistant"}) + frame({"content": TEXT}, id="different") + frame(reason="length", usage=usage()) + DONE,
                    frame({"role": "assistant"}) + frame(reason="length") + DONE,
                    b'data: {"error":{"code":"resource_limit"}}\n\n' + DONE]
        for bad in variants:
            with self.subTest(raw=bad), self.assertRaises(ValueError):
                parse_completion(200, HEADERS, bad, True)
        with self.assertRaises(ValueError):
            parse_completion(429, {}, b'{"error":{"code":"resource_limit"}}', False)


if __name__ == "__main__":
    unittest.main()
