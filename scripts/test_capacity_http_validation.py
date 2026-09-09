"""Small saved-log contract tests; no server, subprocess, GPU, or model.

Run: python3 -B scripts/test_capacity_http_validation.py
Fixtures describe one oracle, two successful requests (including first-token
EOS), and a cancelled request. Logs deliberately arrive in a different order.
"""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from capacity_http_validation import require_capacity_health, validate_capacity_terminals


PID = 42001


def workload():
    clients = [
        {"event": "oracle", "request_id": "o1", "passed": True,
         "usage": {"completion_tokens": 16}, "content": "synthetic oracle"},
        {"event": "request", "request_id": "r1", "passed": True,
         "usage": {"completion_tokens": 5}},
        {"event": "request", "request_id": "r0", "passed": True,
         "usage": {"completion_tokens": 1}, "finish_reason": "stop"},
        {"event": "cancel", "request_id": "c1", "passed": True,
         "client_abort_requested": True},
    ]
    common = {"schema": "qwen-http-lifecycle-v1", "event": "model_terminal", "pid": PID,
              "mtp_depth": 0}
    # Fixed literal counts exercise the public audit contract independently of
    # its implementation, including out-of-order terminal association.
    server = [
        dict(common, request_id="c1", model_kind="cancelled", kv_append_mode=None,
             kv_capacity_token_steps=None, kv_capacity_workspace_fallbacks=None,
             kv_capacity_workspace_peak_bytes=None),
        dict(common, request_id="r0", model_kind="completed", model_finish_reason="eos",
             completion_tokens=1, decoded_tokens=0, kv_append_mode="capacity256",
             kv_capacity_token_steps=0, kv_capacity_workspace_fallbacks=0,
             kv_capacity_workspace_peak_bytes=0),
        dict(common, request_id="o1", model_kind="completed", model_finish_reason="length",
             completion_tokens=16, decoded_tokens=15, kv_append_mode="capacity256",
             kv_capacity_token_steps=15, kv_capacity_workspace_fallbacks=0,
             kv_capacity_workspace_peak_bytes=4096),
        dict(common, request_id="r1", event="output_terminal", model_kind="completed"),
        dict(common, request_id="r1", model_kind="completed", model_finish_reason="length",
             completion_tokens=5, decoded_tokens=4, kv_append_mode="capacity256",
             kv_capacity_token_steps=4, kv_capacity_workspace_fallbacks=0,
             kv_capacity_workspace_peak_bytes=8192),
    ]
    return clients, server


def terminal(server, identity):
    return next(row for row in server if row["request_id"] == identity and row["event"] == "model_terminal")


class CapacityHTTPValidationTests(unittest.TestCase):
    def audit(self, clients, server, transform_server=None, stop_requested=lambda: False):
        with tempfile.TemporaryDirectory(prefix="capacity-http-validation-cpu-") as temporary:
            directory = Path(temporary)
            events, log = directory / "churn.events.ndjson", directory / "server.log"
            events.write_text("".join(json.dumps(row) + "\n" for row in clients))
            # Legacy text uses the same event label and must not count as an
            # extra JSON terminal or satisfy a missing structured record.
            text = "HTTP request id=r1 event=model_terminal legacy summary\n"
            text += "".join(json.dumps(row) + "\n" for row in server)
            log.write_text(transform_server(text) if transform_server else text)
            before = {str(path.resolve()): (path.read_bytes(), path.stat().st_mtime_ns)
                      for path in (events, log)}
            result = validate_capacity_terminals(events, log, PID, stop_requested=stop_requested)
            for path in (events, log):
                data, mtime = before[str(path.resolve())]
                self.assertEqual(path.read_bytes(), data)
                self.assertEqual(path.stat().st_mtime_ns, mtime)
            if result["passed"]:
                for path, (data, _) in before.items():
                    self.assertEqual(result["inputs"][path],
                                     {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
            return result

    def rejected(self, result, reason):
        self.assertIs(result["passed"], False)
        self.assertIs(result["complete"], False)
        self.assertTrue(any(reason in error for error in result["errors"]), result["errors"])

    def test_health_policy_preserves_the_existing_health_object(self):
        health = {"ready": True, "pid": PID, "prefix_cache": {"entries": 2},
                  "kv_append_policy": {"kv_append_mode": "capacity256",
                                       "scope": "autoregressive_decode_only",
                                       "prefill_uses_capacity": False, "mtp_kv_append_mode": "reference"}}
        saved = copy.deepcopy(health)
        self.assertIs(require_capacity_health(health), health)
        self.assertEqual(health, saved)
        for label, alter in (
            ("absent", lambda h: h.pop("kv_append_policy")),
            ("null", lambda h: h.update(kv_append_policy=None)),
            ("reference server", lambda h: h["kv_append_policy"].update(kv_append_mode="reference")),
            ("wrong scope", lambda h: h["kv_append_policy"].update(scope="all_inference")),
            ("prefill enabled", lambda h: h["kv_append_policy"].update(prefill_uses_capacity=True)),
            ("numeric false", lambda h: h["kv_append_policy"].update(prefill_uses_capacity=0)),
            ("MTP capacity", lambda h: h["kv_append_policy"].update(mtp_kv_append_mode="capacity256")),
            ("missing scope", lambda h: h["kv_append_policy"].pop("scope")),
        ):
            with self.subTest(label=label):
                changed = copy.deepcopy(health)
                alter(changed)
                with self.assertRaises(RuntimeError):
                    require_capacity_health(changed)

    def test_completed_cancelled_and_first_token_eos_join_by_id(self):
        result = self.audit(*workload())
        self.assertIs(result["passed"], True, result["errors"])
        self.assertIs(result["complete"], True)
        self.assertEqual(result["counts"], {"oracle": 1, "request": 2, "cancel": 1,
                                          "completed": 3, "cancelled": 1,
                                          "completed_with_decode": 2, "model_terminals": 4})
        self.assertEqual(result["capacity_token_steps"], 19)
        self.assertEqual(result["workspace_fallbacks"], 0)
        self.assertEqual(result["workspace_peak_bytes"], 8192)

    def test_only_first_token_outputs_cannot_prove_capacity_execution(self):
        clients, server = workload()
        for row in clients:
            if row["event"] != "cancel":
                row["usage"]["completion_tokens"] = 1
        for row in server:
            if row["event"] == "model_terminal" and row["model_kind"] == "completed":
                row.update(model_finish_reason="eos", completion_tokens=1, decoded_tokens=0,
                           kv_capacity_token_steps=0, kv_capacity_workspace_peak_bytes=0)
        result = self.audit(clients, server)
        self.rejected(result, "No successfully evaluated capacity decode steps")
        self.assertEqual(result["capacity_token_steps"], 0)
        self.assertEqual(result["counts"]["completed"], 3)

    def test_id_association_rejects_duplicates_missing_and_orphans(self):
        for label, alter, message in (
            ("client duplicate", lambda c, s: c.append(dict(c[0])), "Duplicate client ID"),
            ("terminal duplicate", lambda c, s: s.append(dict(terminal(s, "r1"))), "Duplicate JSON model terminal"),
            ("missing terminal", lambda c, s: s.remove(terminal(s, "r1")), "terminal ID sets differ"),
            ("orphan terminal", lambda c, s: terminal(s, "r1").update(request_id="unknown"), "Orphan/invalid model terminal"),
            ("empty client id", lambda c, s: c[0].update(request_id=""), "Invalid client request ID"),
            ("missing client", lambda c, s: c.pop(0), "Orphan/invalid model terminal"),
        ):
            with self.subTest(label=label):
                clients, server = workload()
                alter(clients, server)
                self.rejected(self.audit(clients, server), message)

    def test_completed_execution_accounting_rejects_bad_metadata(self):
        for field, value, message in (
            ("pid", PID + 1, "Terminal PID mismatch"),
            ("pid", True, "Terminal PID mismatch"),
            ("mtp_depth", 2, "Unexpected MTP terminal"),
            ("model_kind", "failed", "Successful client model did not complete"),
            ("kv_append_mode", "reference", "Completed runner policy mismatch"),
            ("completion_tokens", 6, "AR completion/decoded count mismatch"),
            ("decoded_tokens", 5, "AR completion/decoded count mismatch"),
            ("kv_capacity_token_steps", 3, "Successful capacity step mismatch"),
            ("kv_capacity_token_steps", None, "Successful capacity step mismatch"),
            ("kv_capacity_token_steps", True, "Successful capacity step mismatch"),
            ("kv_capacity_workspace_fallbacks", 1, "Unexpected workspace fallback"),
            ("kv_capacity_workspace_fallbacks", False, "Invalid workspace fallback count"),
            ("kv_capacity_workspace_peak_bytes", 0, "Workspace peak mismatch"),
            ("kv_capacity_workspace_peak_bytes", -1, "Workspace peak mismatch"),
        ):
            with self.subTest(field=field, value=value):
                clients, server = workload()
                terminal(server, "r1")[field] = value
                result = self.audit(clients, server)
                self.rejected(result, message)
                if field == "kv_capacity_workspace_fallbacks" and value == 1:
                    self.assertEqual(result["workspace_fallbacks"], 1)

    def test_first_token_zero_cannot_hide_an_allocation_or_step(self):
        for field in ("kv_capacity_token_steps", "kv_capacity_workspace_peak_bytes"):
            with self.subTest(field=field):
                clients, server = workload()
                terminal(server, "r0")[field] = 1
                result = self.audit(clients, server)
                self.assertIs(result["passed"], False)
                self.assertTrue(result["errors"])

    def test_cancellation_requires_explicit_nulls_not_absent_or_zero(self):
        for field, wrong in (("kv_append_mode", "capacity256"),
                             ("kv_capacity_token_steps", 0),
                             ("kv_capacity_workspace_fallbacks", 0),
                             ("kv_capacity_workspace_peak_bytes", 0)):
            for missing in (False, True):
                with self.subTest(field=field, missing=missing):
                    clients, server = workload()
                    cancelled = terminal(server, "c1")
                    if missing:
                        cancelled.pop(field)
                    else:
                        cancelled[field] = wrong
                    self.rejected(self.audit(clients, server),
                                  "Capacity terminal field absent" if missing else "Cancelled execution counters must be null")

    def test_client_failure_or_unconfirmed_cancel_does_not_pass(self):
        for label, alter, message in (
            ("client failed", lambda c, s: c[1].update(passed=False), "Client did not pass"),
            ("bad client usage", lambda c, s: c[1]["usage"].update(completion_tokens=True), "Invalid client completion count"),
            ("no cancel intent", lambda c, s: c[3].update(client_abort_requested=False), "Missing client cancel intent"),
            ("cancel completed", lambda c, s: terminal(s, "c1").update(model_kind="completed"), "Cancellation not confirmed"),
        ):
            with self.subTest(label=label):
                clients, server = workload()
                alter(clients, server)
                self.rejected(self.audit(clients, server), message)

    def test_saved_logs_must_have_complete_unambiguous_json(self):
        clients, server = workload()
        transforms = (
            ("partial final line", lambda text: text.rstrip("\n"), "Oversized/incomplete line"),
            ("duplicate JSON key", lambda text: text.replace('"pid": 42001', '"pid": 42001, "pid": 42001', 1), "Duplicate JSON key"),
            ("malformed JSON", lambda text: text + '{broken}\n', "ValueError"),
        )
        for label, transform, message in transforms:
            with self.subTest(label=label):
                result = self.audit(clients, server, transform_server=transform)
                if label == "malformed JSON":
                    self.assertIs(result["passed"], False)
                    self.assertTrue(result["errors"])
                else:
                    self.rejected(result, message)

    def test_audit_responds_to_existing_stop_signal_callback(self):
        result = self.audit(*workload(), stop_requested=lambda: True)
        self.rejected(result, "Terminal audit interrupted")


if __name__ == "__main__":
    unittest.main(verbosity=2)
