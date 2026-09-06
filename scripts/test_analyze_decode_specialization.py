import copy
import unittest

from analyze_decode_specialization import analyze


def fixture(modes=("reference", "all", "all", "reference"), policies=None, rates=None):
    policies = list(policies or ["disabled"] * len(modes))
    rates = rates or [10, 20, 30, 10][:len(modes)]
    mem = {"active_bytes": 100, "cache_bytes": 10, "peak_bytes": 120, "limit_bytes": 1000}
    trials = []
    for i, (mode, policy, rate) in enumerate(zip(modes, policies, rates)):
        trials.append({
            "repetition": i, "prompt_tokens": [1, 2], "generated_token_ids": [7, 8, 9, 10],
            "decode_mode": mode, "wired_memory": {"policy": policy, "targetLimitBytes": 0 if policy == "disabled" else 1000,
                                                   "setupDurationMilliseconds": 5},
            "mtp_enabled": False, "prefill_accumulation": "reference", "sampling": "greedy",
            "final_prompt_token_held_back": True, "prefill_chunk": 64, "ssd_workers": 1,
            "finish_reason": "length", "decode_steps": 3, "decode_step_seconds": [1 / rate] * 3,
            "decode_tokens_per_second": rate, "logical_decode_weight_bytes_per_token": 1000,
            "logical_decode_weight_footprint_rate_gbps": 1000 * rate / 1e9,
            "final_state_offset": 5, "time_to_first_token_seconds_excluding_load": 1 + i / 10,
            "memory": mem.copy(),
        })
    return {"trials": trials, "context_limit": 128, "max_tokens": 4,
            "requested_repetitions": len(modes), "decode_order": list(modes), "wired_order": policies,
            "additional_projection_buffer_bytes": 500, "loaded_memory": mem.copy(),
            "mtp_enabled": False, "mtp_weights_loaded": False, "provenance": {"executable_sha256": "fixture-only"},
            "scheduler_environment": {}, "profiler": {"mode": "disabled", "stages": [], "droppedRecords": 0},
            "telemetry": {"enabled": False}}


class DecodeSpecializationTests(unittest.TestCase):
    def test_aggregate_uses_total_steps_over_time_and_logical_units(self):
        result = analyze(fixture())
        self.assertTrue(result["eligible_for_unprofiled_comparison"])
        candidate = next(g for g in result["groups"] if g["decode_mode"] == "all")
        self.assertEqual(candidate["trial_indices"], [1, 2])
        self.assertAlmostEqual(candidate["aggregate_tokens_per_second"], 24)
        self.assertEqual(candidate["median_tokens_per_second"], 25)
        self.assertAlmostEqual(candidate["aggregate_logical_weight_footprint_gbps"], 24000 / 1e9)
        self.assertIsNone(candidate["physical_dram_bandwidth_gbps"])
        self.assertIsNone(result["performance_pass"])
        self.assertAlmostEqual(result["adjacent_four_trial_pairs"][0]["percent_delta"], 140)

    def test_baab_and_scalar_baseline(self):
        result = analyze(fixture(("all", "scalar", "scalar", "all")), baseline_mode="scalar")
        pair = result["adjacent_four_trial_pairs"][0]
        self.assertEqual(pair["pattern"], "BAAB")
        self.assertEqual(pair["baseline"]["decode_mode"], "scalar")
        self.assertAlmostEqual(pair["rate_ratio"], 10 / 24)

    def test_same_mode_wired_pair_is_valid_but_groups_stay_separate(self):
        result = analyze(fixture(("reference",) * 4, ("disabled", "fit", "fit", "disabled")))
        self.assertEqual(len(result["groups"]), 2)
        pair = result["adjacent_four_trial_pairs"][0]
        self.assertEqual(pair["changed_dimension"], "wired_policy")
        self.assertEqual(pair["candidate"]["wired_policy"], "fit")
        self.assertAlmostEqual(pair["percent_delta"], 140)

    def test_pair_changing_two_dimensions_has_no_gain(self):
        result = analyze(fixture(policies=("disabled", "fit", "fit", "disabled")))
        pair = result["adjacent_four_trial_pairs"][0]
        self.assertEqual(pair["status"], "rejected_two_dimensions_changed")
        self.assertIsNone(pair["percent_delta"])
        candidate = next(c for c in result["baseline_comparisons"] if c["decode_mode"] == "all")
        self.assertEqual(candidate["status"], "baseline_absent_for_this_wired_policy")

    def test_skip_removes_performance_only(self):
        data = fixture(("reference", "reference", "all", "all", "reference"), rates=[1, 10, 20, 30, 10])
        result = analyze(data, skip_first=1)
        self.assertEqual(result["excluded_trial_indices"], [0])
        self.assertEqual(result["adjacent_four_trial_pairs"][0]["trial_indices"], [1, 2, 3, 4])
        data["trials"][0]["generated_token_ids"][0] = 999
        result = analyze(data, skip_first=1)
        self.assertFalse(result["all_exact_generated_token_equality"])
        self.assertFalse(result["eligible_for_unprofiled_comparison"])
        self.assertIsNone(result["adjacent_four_trial_pairs"][0]["percent_delta"])

    def test_instrumentation_mtp_precision_and_context_disable_comparisons(self):
        alterations = [
            lambda d: d["profiler"].update(mode="hostBodyOnly"),
            lambda d: d["telemetry"].update(enabled=True),
            lambda d: d.update(mtp_enabled=True),
            lambda d: d.update(mtp_weights_loaded=True),
            lambda d: d["trials"][1].update(mtp_enabled=True),
            lambda d: d["trials"][1].update(prefill_accumulation="float32"),
            lambda d: d["trials"][1].update(context_limit=129),
            lambda d: d["trials"][1].update(prompt_tokens=[2, 1]),
        ]
        for alter in alterations:
            data = fixture()
            alter(data)
            result = analyze(data)
            self.assertFalse(result["eligible_for_unprofiled_comparison"])
            self.assertTrue(result["validation_failures"])
            self.assertTrue(all(c["percent_delta"] is None for c in result["baseline_comparisons"]))

    def test_missing_current_fields_fail_explicitly(self):
        for key in ("context_limit", "max_tokens", "decode_order", "wired_order", "additional_projection_buffer_bytes", "telemetry"):
            data = fixture()
            del data[key]
            with self.assertRaisesRegex(ValueError, "missing required field"):
                analyze(data)
        for key in ("decode_mode", "wired_memory", "generated_token_ids"):
            data = fixture()
            del data["trials"][0][key]
            with self.assertRaisesRegex(ValueError, "missing required field"):
                analyze(data)

    def test_gpu_command_timing_blocks_ratios_but_old_reports_remain_eligible(self):
        for enabled in (None, False, True):
            with self.subTest(enabled=enabled):
                data = fixture()
                if enabled is not None:
                    data["gpu_command_timing"] = {"enabled": enabled}
                result = analyze(data)
                self.assertEqual(result["eligible_for_unprofiled_comparison"], enabled is not True)
                comparisons = result["baseline_comparisons"] + result["adjacent_four_trial_pairs"]
                self.assertTrue(comparisons)
                if enabled is True:
                    self.assertIn("GPU command timing must be disabled for unprofiled comparisons",
                                  result["validation_failures"])
                    for comparison in comparisons:
                        self.assertEqual(comparison["status"], "ineligible_for_unprofiled_comparison")
                        self.assertIsNone(comparison["rate_ratio"])
                        self.assertIsNone(comparison["percent_delta"])
                else:
                    self.assertFalse(result["validation_failures"])
                    self.assertTrue(all(c["rate_ratio"] is not None for c in comparisons))

    def test_invalid_or_incomplete_decode_window_fails(self):
        for value in (0, -1, float("nan"), float("inf"), True):
            data = fixture()
            data["trials"][0]["decode_step_seconds"][0] = value
            with self.assertRaises(ValueError):
                analyze(data)
        for key, value in (("decode_steps", 2), ("generated_token_ids", [7, 8]), ("decode_tokens_per_second", 99)):
            data = fixture()
            data["trials"][0][key] = value
            with self.assertRaises(ValueError):
                analyze(data)

    def test_at_least_one_retained_trial_and_original_order_are_required(self):
        for skip in (-1, 4):
            with self.assertRaises(ValueError):
                analyze(fixture(), skip_first=skip)
        data = fixture(("reference",), rates=[10])
        self.assertEqual(analyze(data)["groups"][0]["trial_count"], 1)
        data = fixture()
        data["trials"][0]["repetition"] = 2
        with self.assertRaises(ValueError):
            analyze(data)
        data = fixture()
        data["decode_order"][1] = "scalar"
        with self.assertRaises(ValueError):
            analyze(data)


if __name__ == "__main__":
    unittest.main()
