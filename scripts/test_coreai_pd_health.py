"""CPU tests of the saved-health contract auditor; no live-service acceptance."""
import unittest

from check_coreai_pd_health import audit, validate_health


def health(**changes):
    return {"backend": "native-coreai", "ready": True, "pd_scheduling": "serial",
            "independent_pd_functions": True, "prefill_chunk_size": 4, "prefill_policy": "chunked",
            "active_requests": 0, "queued_requests": 0, "requests_in_flight": 0, "max_pending_requests": 2,
            "prefill_group_milliseconds": {}, "decode_group_milliseconds": {}, **changes}


class HealthContractTests(unittest.TestCase):
    def test_ready_and_serial_queue_samples(self):
        samples = [("idle", health()), ("queued", health(active_requests=1, queued_requests=1, requests_in_flight=2)),
                   ("completed", health(prefill_group_milliseconds={"prefill.gdn": 10, "decode.qsa": 1},
                                        decode_group_milliseconds={"decode.gdn": 4}))]
        result = audit(samples, require_queued=True, require_work=True)
        self.assertTrue(result["passed"])
        self.assertEqual(result["queued_samples"], 1)

    def test_force_s1_keeps_independent_asset_identity(self):
        validate_health(health(prefill_chunk_size=1, prefill_policy="tokenwise"), expected_chunk=1)

    def test_missing_pd_fields_are_not_accepted_as_legacy_defaults(self):
        value = health()
        del value["pd_scheduling"]
        with self.assertRaisesRegex(AssertionError, "serial"):
            validate_health(value)

    def test_multiple_active_requests_violate_serial_claim(self):
        with self.assertRaisesRegex(AssertionError, "multiple active"):
            validate_health(health(active_requests=2, requests_in_flight=2))

    def test_reservations_and_admission_limit(self):
        for value in [health(active_requests=1),
                      health(active_requests=1, queued_requests=2, requests_in_flight=3)]:
            with self.assertRaises(AssertionError):
                validate_health(value)

    def test_phase_policy_and_selected_chunk_must_agree(self):
        for value in [health(prefill_chunk_size=1), health(prefill_chunk_size=8),
                      health(prefill_chunk_size=True), health(independent_pd_functions=False)]:
            with self.assertRaises(AssertionError):
                validate_health(value)

    def test_timing_groups_must_be_finite_nonnegative_numbers(self):
        for duration in [-1, float("inf"), float("nan"), True, "4"]:
            with self.assertRaisesRegex(AssertionError, "duration"):
                validate_health(health(prefill_group_milliseconds={"prefill.gdn": duration}))

    def test_loading_samples_cannot_establish_ready_phase_configuration(self):
        with self.assertRaisesRegex(AssertionError, "no ready"):
            audit([("loading", health(ready=False, independent_pd_functions=False,
                                        prefill_chunk_size=1, prefill_policy="tokenwise"))])

    def test_required_work_and_queue_evidence_cannot_be_fabricated_by_idle_health(self):
        with self.assertRaisesRegex(AssertionError, "active-plus-queued"):
            audit([("idle", health())], require_queued=True)
        with self.assertRaisesRegex(AssertionError, "both business phases"):
            audit([("idle", health())], require_work=True)


if __name__ == "__main__":
    unittest.main()
