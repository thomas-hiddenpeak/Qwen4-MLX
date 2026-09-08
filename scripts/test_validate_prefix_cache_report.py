"""Parser checks only: these tiny JSON objects are not GPU evidence."""
import copy
import unittest

from validate_prefix_cache_report import LABELS, validate


def fixture():
    rows = []
    for label in LABELS["long"]:
        key = "b" if label.endswith("b") else "a"
        hit = label.startswith("hit")
        cached = 416 if hit else 0
        ids = [7, 8]
        prompt = [10 if key == "a" else 11] * 833
        phase = {"promptTokenCount": 833, "cachedTokenCount": cached, "computedTokenCount": 833 - cached,
                 "targetSeconds": 1.0, "cacheRestoreSeconds": .01 if hit else 0}
        result = {"tokens": ids, "finishReason": "length", "statistics": {"promptTokenCount": 833,
                  "generatedTokenCount": 2, "mtpDepth": 0}, "phases": {"prefill": phase},
                  "timeToFirstTokenSeconds": 1.2, "decodeSeconds": .1}
        row = {"label": label, "prompt_key": key, "is_cold_oracle": label.startswith("cold"),
               "generated_token_ids": ids, "prompt_token_ids": prompt, "max_tokens": 2,
               "finish_reason": "length", "actual_cached_tokens": cached, "expected_cached_tokens": cached,
               "computed_tokens": 833 - cached, "mtp_depth": 0, "result": result}
        row.update({k: True for k in ("passed", "callback_exact", "offset_exact", "counts_exact", "mtp_mode_exact",
                                     "state_observed", "output_exact")})
        rows.append(row)
    return {"schema": "qwen38-prefix-cache-probe-v1", "suite": "long", "complete": True, "passed": True,
            "state_readback": False, "trials": rows, "state_checks": [], "checks": {"full_model": True}}


class PrefixCacheReportTests(unittest.TestCase):
    def test_complete_uninstrumented_report(self):
        summary = validate(fixture())
        self.assertEqual(summary["complete_output_ids"], 10)
        self.assertEqual(len(summary["hits"]), 2)

    def test_final_pass_flag_cannot_hide_mismatched_output(self):
        report = fixture()
        report["trials"][3] = copy.deepcopy(report["trials"][3])
        report["trials"][3]["generated_token_ids"] = [7, 9]
        with self.assertRaisesRegex(ValueError, "differs from cold oracle"):
            validate(report)

    def test_skipped_tokens_cannot_inflate_compute(self):
        report = fixture()
        report["trials"][3]["computed_tokens"] = 833
        with self.assertRaisesRegex(ValueError, "counted as compute"):
            validate(report)

    def test_missing_trial_is_rejected(self):
        report = fixture()
        report["trials"].pop()
        with self.assertRaisesRegex(ValueError, "missing, extra"):
            validate(report)

    def test_missing_state_readback_is_rejected(self):
        report = fixture()
        report["state_readback"] = True
        with self.assertRaisesRegex(ValueError, "state readback requested"):
            validate(report)

    def test_mtp_cannot_silently_change_mode(self):
        report = fixture()
        report["trials"][3]["result"]["statistics"]["mtpDepth"] = 2
        with self.assertRaisesRegex(ValueError, "MTP mode changed"):
            validate(report)


if __name__ == "__main__":
    unittest.main()
