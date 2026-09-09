"""Saved-evidence CPU controls; no server, model, GPU or subprocess is used."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import analyze_http_cache_churn as analyzer
from cache_churn_oracles import audit_oracle_discrimination


STARTS = {"long_00": 1000, "long_01": 2000, "long_02": 3000,
          "long_03": 4000, "short_00": 65000}


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def evidence(mode="counter-witness"):
    """Literal expected starts, synthetic lifecycle values and bounded output."""
    contents = {name: " ".join(str(start + i) for i in range(8))
                for name, start in STARTS.items()}
    if mode == "legacy":
        contents = {name: "1 2 3 4 5 6 7 8" for name in STARTS}
    clients, terminals, profiles = [], [], {}
    for i, (name, content) in enumerate(contents.items()):
        prompt = 11000 if name.startswith("long_") else 1200
        row = {"event": "oracle", "request_id": f"o{i}", "profile": name,
               "passed": True, "content": content, "content_sha256": digest(content),
               "finish_reason": "length", "wall_seconds": .3,
               "effective_cached_tokens": 0,
               "usage": {"prompt_tokens": prompt, "completion_tokens": 16,
                         "total_tokens": prompt + 16}}
        clients.append(row)
        profiles[name] = {"prompt_tokens": prompt, "completion_tokens": 16,
                          "content_sha256": digest(content), "finish_reason": "length"}
        if mode != "legacy":
            profiles[name]["expected_start"] = STARTS[name]
    request = {**clients[0], "event": "request", "request_id": "r1"}
    request.pop("content")
    clients.append(request)
    clients.append({"event": "cancel", "request_id": "c1", "profile": "long_01",
                    "passed": True, "client_abort_requested": True, "wall_seconds": .1})
    for row in clients:
        terminal = {"schema": "qwen-http-lifecycle-v1", "event": "model_terminal",
                    "request_id": row["request_id"], "pid": 42001, "mtp_depth": 0,
                    "model_kind": "cancelled" if row["event"] == "cancel" else "completed"}
        if row["event"] != "cancel":
            prompt = row["usage"]["prompt_tokens"]
            terminal.update(model_finish_reason="length", prompt_tokens=prompt,
                completion_tokens=16, cached_prompt_tokens=0, computed_prompt_tokens=prompt,
                actual_prefill_tokens=prompt, recomputed_prefill_tokens=0,
                cache_source="cold", prefill_seconds=.2, decode_seconds=.1)
        terminals.append(terminal)
    health = {"pid": 42001, "idle": True,
              **{key: 0 for key in ("active", "active_jobs", "pending_requests", "queued_prefills",
                  "ready_decodes", "resident_sequences", "reserved_tokens", "waiting_prefix_sequences")},
              "prefix_cache": {"entries": 0, "logicalPayloadBytes": 0, "keyTokens": 0, "liveFlights": 0},
              "prefix_cache_limits": {"maxEntries": 1, "maxBytes": 1000, "maxKeyTokens": 30000},
              "prefix_disk_cache": {"entries": 0, "diskBytes": 0, "pendingJobs": 0,
                                    "pendingBytes": 0, "foregroundReadIntents": 0},
              "prefix_disk_cache_limits": {"maxEntries": 1, "maxBytes": 1000,
                                           "maxPendingJobs": 2, "maxPendingBytes": 1000},
              "state_budget": {"maxBytes": 1000, "requestBytes": 0, "cacheBytes": 0,
                  "workspaceBytes": 0, "totalBytes": 0, "peakBytes": 0, "currentLeases": 0},
              "mlx_memory": {"active_bytes": 0, "cache_bytes": 0},
              "memory_pressure_monitor_running": True,
              "memory_pressure": {"observedLevel": "unknown", "lastEventSource": None}}
    summary = {"complete": True, "passed": True, "run_id": "synthetic-oracle-audit",
               "server_pid": 42001, "requests": 2, "completed_requests": 1,
               "client_abort_requests": 1, "profiles": profiles,
               "initial_health": health, "final_health": health}
    if mode != "legacy":
        claim = audit_oracle_discrimination(mode, contents,
            expected_profiles=STARTS, reported_hashes={name: digest(content) for name, content in contents.items()})
        summary.update(fixture_mode=mode, oracle_discrimination=claim)
        clients.append({"event": "oracle_discrimination", **copy.deepcopy(claim)})
    process = {**health, "elapsed_seconds": 1, "fully_idle": True,
               "rss_bytes": 1000, "numeric_fds": 10}
    return summary, clients, terminals, process


class OracleAuditTests(unittest.TestCase):
    def run_audit(self, data):
        summary, clients, terminals, process = data
        with tempfile.TemporaryDirectory(prefix="oracle-audit-cpu-") as temporary:
            directory = Path(temporary)
            paths = {name: directory / name for name in ("summary.json", "events.ndjson", "server.log", "process.ndjson")}
            paths["summary.json"].write_text(json.dumps(summary))
            paths["events.ndjson"].write_text("".join(json.dumps(row) + "\n" for row in clients))
            paths["server.log"].write_text("".join(json.dumps(row) + "\n" for row in terminals))
            paths["process.ndjson"].write_text(json.dumps(process) + "\n")
            before = {name: path.read_bytes() for name, path in paths.items()}
            result = analyzer.analyze_cli(SimpleNamespace(summary=paths["summary.json"],
                events=paths["events.ndjson"], server_log=paths["server.log"],
                process=paths["process.ndjson"], lifecycle=None))
            for name, path in paths.items():
                self.assertEqual(path.read_bytes(), before[name])
            return result

    def assert_rejected(self, data, issue):
        result = self.run_audit(data)
        self.assertFalse(result["passed"], result)
        self.assertIn(issue, result["issues"])
        return result

    def test_strict_complete_join_passes_and_binds_actual_inputs(self):
        result = self.run_audit(evidence())
        self.assertTrue(result["passed"], result["issues"])
        self.assertEqual(result["oracle_discrimination"]["unique_content_hashes"], 5)
        self.assertEqual(result["matched_unique_terminals"], {"oracle": 5, "request": 1, "cancel": 1})
        self.assertEqual(result["unmatched_client_records"], 0)
        self.assertEqual(result["unmatched_server_terminals"], 0)

    def test_legacy_identical_answers_remain_explicitly_nondiscriminating(self):
        result = self.run_audit(evidence("legacy"))
        self.assertTrue(result["passed"], result["issues"])
        self.assertFalse(result["oracle_discrimination"]["required"])
        self.assertFalse(result["oracle_discrimination"]["discriminating"])
        self.assertEqual(result["oracle_discrimination"]["unique_content_hashes"], 1)

    def test_claimed_success_cannot_hide_duplicate_actual_outputs_or_filter_samples(self):
        data = evidence(); summary, clients, _, _ = data
        clients[1]["content"] = clients[0]["content"]
        clients[1]["content_sha256"] = clients[0]["content_sha256"]
        summary["profiles"]["long_01"]["content_sha256"] = clients[0]["content_sha256"]
        result = self.assert_rejected(data, "oracle_discrimination_failed")
        self.assertEqual(result["oracle_discrimination"]["unique_content_hashes"], 4)
        self.assertIn("summary_oracle_discrimination_mismatch", result["issues"])
        self.assertEqual(result["client_counts"]["oracle"], 5)
        self.assertEqual(result["matched_unique_terminals"]["oracle"], 5)

    def test_different_content_with_wrong_start_fails_even_when_claim_changes(self):
        data = evidence(); summary, clients, _, _ = data
        clients[1].update(content="9000 9001", content_sha256=digest("9000 9001"))
        for claim in (summary["oracle_discrimination"], clients[-1]):
            claim["profiles"]["long_01"].update(expected_start=9000,
                content_sha256=digest("9000 9001"), starts_with_expected=True)
        self.assert_rejected(data, "oracle_discrimination_failed")

    def test_same_profile_count_with_wrong_inventory_is_rejected(self):
        data = evidence(); summary = data[0]
        summary["profiles"]["long_04"] = summary["profiles"].pop("long_03")
        result = self.assert_rejected(data, "oracle_discrimination_failed")
        self.assertNotIn("summary_oracle_count_mismatch", result["issues"])

    def test_unknown_mode_and_missing_claims_cannot_fall_back_to_legacy(self):
        for field in ("unknown", "missing_summary", "missing_event"):
            with self.subTest(field=field):
                data = evidence(); summary, clients, _, _ = data
                if field == "unknown": summary["fixture_mode"] = "counter-witnes"
                elif field == "missing_summary": summary.pop("oracle_discrimination")
                else: clients.pop()
                self.assert_rejected(data, {"unknown": "invalid_oracle_discrimination_contract",
                    "missing_summary": "summary_oracle_discrimination_mismatch",
                    "missing_event": "oracle_discrimination_event_count"}[field])

    def test_aggregate_schema_types_and_expected_start_are_recomputed(self):
        for field in ("schema", "expected_start", "passed_type", "duplicate_event"):
            with self.subTest(field=field):
                data = evidence(); summary, clients, _, _ = data
                if field == "schema": clients[-1]["schema"] = "invented-v1"
                elif field == "expected_start": clients[-1]["profiles"]["long_00"]["expected_start"] = 1000.0
                elif field == "passed_type": summary["oracle_discrimination"]["passed"] = 1
                else: clients.append(copy.deepcopy(clients[-1]))
                self.assert_rejected(data, "summary_oracle_discrimination_mismatch" if field == "passed_type"
                    else "oracle_discrimination_event_count" if field == "duplicate_event" else "event_oracle_discrimination_mismatch")

    def test_duplicate_oracle_profile_or_id_keeps_existing_failure(self):
        for duplicate_id in (False, True):
            with self.subTest(duplicate_id=duplicate_id):
                data = evidence(); summary, clients, terminals, _ = data
                extra = copy.deepcopy(clients[0]); term = copy.deepcopy(terminals[0])
                if not duplicate_id:
                    extra["request_id"] = term["request_id"] = "another-oracle-id"
                clients.insert(1, extra); terminals.insert(1, term)
                result = self.assert_rejected(data, "duplicate_profile_oracle")
                self.assertEqual(result["client_counts"]["oracle"], 6)
                if duplicate_id: self.assertIn("duplicate_client_request_id", result["issues"])

    def test_saved_sha_and_summary_profile_start_are_not_trusted(self):
        for field in ("sha", "summary_start"):
            with self.subTest(field=field):
                data = evidence(); summary, clients, _, _ = data
                if field == "sha": clients[1]["content_sha256"] = "0" * 64
                else: summary["profiles"]["long_01"]["expected_start"] = 2001
                self.assert_rejected(data, "oracle_content_hash_mismatch" if field == "sha"
                    else "summary_oracle_start_mismatch")

    def test_incomplete_population_can_be_previewed_but_never_passes(self):
        summary = {"fixture_mode": "counter-witness", "profiles": {}}
        issues = []
        result = analyzer.audit_saved_oracles(summary, {}, [], complete=False,
            issue=lambda code, detail: issues.append(code))
        self.assertTrue(result["validation_pending"])
        self.assertEqual(issues, [])


if __name__ == "__main__":
    unittest.main()
