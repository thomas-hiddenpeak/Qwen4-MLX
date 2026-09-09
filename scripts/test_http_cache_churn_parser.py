"""CPU-only churn controls. Fake transport tests do not exercise a live server."""
from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import probe_http_cache_churn as churn
from cache_churn_oracles import audit_oracle_discrimination, counter_witness_start
from test_http_cache_reliability_parser import frame, nonstream, stream, usage


class CacheChurnParserTests(unittest.TestCase):
    def test_loopback_origins_only(self):
        self.assertEqual(churn.origin("http://localhost:11236"), ("127.0.0.1", 11236))
        self.assertEqual(churn.origin("http://[::1]:11236/"), ("::1", 11236))
        for value in ("https://127.0.0.1", "http://example.com", "http://user@127.0.0.1",
                      "http://127.0.0.1/path", "http://127.0.0.1?x=1", "http://127.0.0.1#x"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                churn.origin(value)

    def test_incremental_sse_split_at_every_byte_including_crlf(self):
        for raw in (stream(usage()), stream(usage()).replace(b"\n", b"\r\n")):
            observer = churn.SSEObservation()
            for index, value in enumerate(raw):
                observer.feed(bytes([value]), index / 1000)
            self.assertTrue(observer.role_seen)
            self.assertTrue(observer.finish_seen)
            self.assertEqual(observer.request_id, "cache-parser-request")
            self.assertIsNotNone(observer.first_content_seconds)
            self.assertEqual(observer.buffer, bytearray())

    def test_sse_metadata_observation_bounds_and_identity(self):
        observer = churn.SSEObservation()
        observer.feed(frame({"role": "assistant"}), .1)
        self.assertIsNone(observer.first_content_seconds)
        with self.assertRaises(ValueError):
            observer.feed(frame({"content": "late"}, id="different"), .2)
        with self.assertRaises(ValueError):
            churn.SSEObservation().feed(b"x" * (churn.MAX_FRAME_BYTES + 1), 1)
        with self.assertRaises(ValueError):
            churn.SSEObservation().feed(b"x" * churn.MAX_FRAME_BYTES + b"\n\n", 1)

    def test_cold_and_hot_completions_compare_content_and_usage_not_hit_count(self):
        client = churn.Client("http://127.0.0.1:11236", 10)
        counts = usage()
        expected = churn.parse_completion(200, {}, nonstream(counts), False)
        for cached in (0, 416):
            current = {**counts, "prompt_tokens_details": {"cached_tokens": cached}}
            with patch.object(client, "raw", return_value=(200, {}, nonstream(current), {"wall_seconds": 1, "response_headers_seconds": .1})):
                row, _ = client.completion(1, "short_00", {}, expected)
            self.assertTrue(row["passed"])
            self.assertEqual(row["effective_cached_tokens"], cached)
            self.assertNotIn("content", row)
            self.assertNotIn("messages", row)
        changed = copy.deepcopy(expected); changed["content"] += " changed"
        with patch.object(client, "raw", return_value=(200, {}, nonstream(counts), {"wall_seconds": 1, "response_headers_seconds": .1})):
            row, result = client.completion(2, "short_00", {}, changed)
        self.assertFalse(row["passed"])
        self.assertIsNone(result)
        self.assertEqual(row["request_id"], "cache-parser-request")

    def test_histogram_storage_is_fixed_and_quantiles_are_labelled(self):
        histogram = churn.LatencyHistogram()
        for _ in range(10000):
            histogram.add(.12)
        summary = histogram.summary()
        self.assertEqual(len(summary["counts"]), len(churn.LATENCY_BUCKETS) + 1)
        self.assertEqual(summary["count"], 10000)
        self.assertEqual(summary["p95_upper_seconds"], .25)
        self.assertEqual(summary["quantile_method"], "fixed_upper_bucket")
        for value in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                histogram.add(value)

    def test_seeded_profiles_diverge_early_and_schedule_covers_hot_cold_cancel(self):
        profiles = churn.make_profiles("long reference\n" * 1000, "model", 7, "fixed", 8, 2)
        self.assertEqual(profiles, churn.make_profiles("long reference\n" * 1000, "model", 7, "fixed", 8, 2))
        starts = {body["messages"][0]["content"][:80] for body in profiles.values()}
        self.assertEqual(len(starts), len(profiles))
        first = churn.ChurnSchedule(list(profiles), 7)
        second = churn.ChurnSchedule(list(profiles), 7)
        steps = [first.next() for _ in range(200)]
        self.assertEqual(steps, [second.next() for _ in range(200)])
        self.assertEqual({step[1] for step in steps}, set(profiles))
        self.assertEqual({step[3] for step in steps if step[2]}, {False, True})
        self.assertEqual(steps[0][1], steps[1][1])

    def test_counter_fixtures_put_distinct_witness_only_in_first_system_header(self):
        source = "Fixture input without a counter.\n" * 1000
        legacy = churn.make_profiles(source, "model", 7, "fixed", 8, 2)
        self.assertEqual(legacy, churn.make_profiles(source, "model", 7, "fixed", 8, 2, mode="legacy"))
        marker = hashlib.sha256(b"fixed:7:long_00").hexdigest()
        self.assertEqual(legacy["long_00"]["messages"][0]["content"], f"Fixture {marker}.\n" + source)
        self.assertEqual(legacy["long_00"]["messages"][1]["content"], "请按顺序输出 1 到 100 的整数，用空格分隔，不解释。")
        profiles = churn.make_profiles(source, "model", 7, "fixed", 8, 2, mode="counter-witness")
        self.assertEqual([counter_witness_start(name) for name in profiles],
                         [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 65000, 66000])
        self.assertEqual(len({body["messages"][1]["content"] for body in profiles.values()}), 1)
        for name, body in profiles.items():
            with self.subTest(name=name):
                witness = str(counter_witness_start(name))
                header, suffix = body["messages"][0]["content"].split("\n", 1)
                self.assertEqual(header, "CACHE_COUNTER_START: " + witness)
                self.assertEqual(suffix, legacy[name]["messages"][0]["content"])
                self.assertNotIn(witness, body["messages"][1]["content"])
                self.assertEqual({key: value for key, value in body.items() if key != "messages"},
                                 {"model": "model", "temperature": 0, "mtp_depth": 0, "max_tokens": 16})

    def test_counter_audit_accepts_distinct_actual_content_and_partial_final_number(self):
        names = [f"long_{index:02d}" for index in range(8)] + ["short_00", "short_01"]
        contents = {name: f" {counter_witness_start(name)} {counter_witness_start(name) + 1} 6" for name in names}
        hashes = {name: hashlib.sha256(content.encode()).hexdigest() for name, content in contents.items()}
        audit = audit_oracle_discrimination("counter-witness", contents,
                                             expected_profiles=names, reported_hashes=hashes)
        self.assertTrue(audit["passed"], audit)
        self.assertTrue(audit["discriminating"])
        self.assertTrue(audit["required"])
        self.assertEqual(audit["profile_count"], 10)
        self.assertEqual(audit["unique_content_hashes"], 10)
        self.assertTrue(all(row["starts_with_expected"] for row in audit["profiles"].values()))

    def test_counter_audit_rejects_wrong_missing_colliding_or_forged_oracles(self):
        baseline = {"long_00": "1000 1001 10", "short_00": "65000 65001 65"}
        cases = [("wrong_integer", {**baseline, "long_00": "10000 10001"}, "wrong_or_missing_start:long_00"),
                 ("word_boundary", {**baseline, "long_00": "1000abc"}, "wrong_or_missing_start:long_00"),
                 ("missing_number", {**baseline, "long_00": ""}, "wrong_or_missing_start:long_00"),
                 ("missing_profile", {"long_00": baseline["long_00"]}, "missing_oracle:short_00"),
                 ("swapped", {"long_00": baseline["short_00"], "short_00": baseline["long_00"]}, "wrong_or_missing_start:long_00"),
                 ("colliding", {"long_00": baseline["long_00"], "short_00": baseline["long_00"]}, "duplicate_oracle_content")]
        for label, contents, issue in cases:
            with self.subTest(label=label):
                audit = audit_oracle_discrimination("counter-witness", contents, expected_profiles=baseline)
                self.assertFalse(audit["passed"])
                self.assertFalse(audit["discriminating"])
                self.assertIn(issue, audit["issues"])
        hashes = {name: hashlib.sha256(content.encode()).hexdigest() for name, content in baseline.items()}
        hashes["long_00"] = "0" * 64
        forged = audit_oracle_discrimination("counter-witness", baseline, reported_hashes=hashes)
        self.assertFalse(forged["passed"])
        self.assertIn("content_hash_mismatch:long_00", forged["issues"])

    def test_legacy_equal_oracles_are_honestly_nondiscriminating(self):
        audit = audit_oracle_discrimination("legacy", {"long_00": "1 2 3 ", "short_00": "1 2 3 "})
        self.assertTrue(audit["passed"])
        self.assertFalse(audit["required"])
        self.assertFalse(audit["discriminating"])
        self.assertEqual(audit["unique_content_hashes"], 1)
        self.assertTrue(all(row["expected_start"] is None and row["starts_with_expected"] is None
                            for row in audit["profiles"].values()))

    def test_counter_main_fails_before_workload_for_wrong_or_colliding_cold_outputs(self):
        for corruption in ("wrong_integer", "colliding"):
            class WitnessClient(FakeClient):
                def raw(self, method, path, body=None, timeout=None, observation=None):
                    status, headers, raw, timing = super().raw(method, path, body, timeout, observation)
                    if path == "/health":
                        return status, headers, raw, timing
                    start = int(body["messages"][0]["content"].split("\n", 1)[0].split(": ")[1])
                    value = start + 1 if corruption == "wrong_integer" else 1000
                    payload = json.loads(raw)
                    payload["choices"][0]["message"]["content"] = f"{value} {value + 1} "
                    return status, headers, json.dumps(payload).encode(), timing

            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "summary.json"
                with patch.object(churn, "Client", WitnessClient), \
                        patch.object(churn, "fixture_text", return_value=("system rule\n" * 1500, {})), \
                        patch.object(churn, "ChurnSchedule") as schedule, \
                        patch.object(churn, "ThreadPoolExecutor") as executor, redirect_stdout(io.StringIO()):
                    result = churn.main(["--output", str(output), "--duration-seconds", "1",
                        "--long-prefixes", "4", "--short-prefixes", "1", "--fixture-mode", "counter-witness"])
                report = json.loads(output.read_text())
                self.assertEqual(result, 1)
                self.assertFalse(report["passed"])
                self.assertEqual(report["error"], "AssertionError: oracle_discrimination")
                self.assertEqual(report["requests"], 0)
                self.assertEqual(report["fixture_mode"], "counter-witness")
                schedule.assert_not_called(); executor.assert_not_called()
                rows = [json.loads(line) for line in Path(report["events_path"]).read_text().splitlines()]
                self.assertEqual(sum(row["event"] == "oracle" for row in rows), 5)
                self.assertFalse(any(row["event"] in ("soak_started", "request", "cancel") for row in rows))
                audits = [row for row in rows if row["event"] == "oracle_discrimination"]
                self.assertEqual(len(audits), 1)
                self.assertEqual({key: value for key, value in audits[0].items() if key != "event"}, report["oracle_discrimination"])
                self.assertFalse(audits[0]["passed"])
                self.assertEqual(audits[0]["profile_count"], 5)
                self.assertTrue(all(row["expected_start"] == counter_witness_start(name)
                                    for name, row in report["profiles"].items()))

    def test_sustained_windows_do_not_accept_only_an_early_total_hit(self):
        before = {"completed_requests": 10, "cached_completions": 5, "cold_completions": 5, "client_abort_requests": 1}
        after = {"completed_requests": 20, "cached_completions": 10, "cold_completions": 10, "client_abort_requests": 2}
        initial = {"bytesRead": 1000, "bytesWritten": 1000, "evictions": 10}
        changed = {"bytesRead": 1500, "bytesWritten": 1500, "evictions": 12}
        self.assertTrue(churn.workload_window(before, after, initial, changed, 300)["continuous_churn_observed"])
        self.assertFalse(churn.workload_window(before, after, initial, initial, 300)["continuous_churn_observed"])
        cached_only = {**after, "cold_completions": before["cold_completions"]}
        self.assertFalse(churn.workload_window(before, cached_only, initial, changed, 300)["continuous_churn_observed"])

    def test_event_log_does_not_write_oversized_partial_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "events.ndjson"
            journal = churn.EventLog(path)
            journal.append("small", sequence=1)
            with self.assertRaises(ValueError):
                journal.append("large", value="x" * churn.MAX_RECORD_BYTES)
            journal.close()
            rows = path.read_text().splitlines()
            self.assertEqual(len(rows), 1)
            self.assertEqual(json.loads(rows[0])["sequence"], 1)

    def test_fake_transport_runs_duration_drains_and_writes_bounded_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "summary.json"
            with patch.object(churn, "Client", FakeClient), patch.object(churn, "fixture_text", return_value=("system rule\n" * 1500, {})), redirect_stdout(io.StringIO()):
                result = churn.main(["--output", str(output), "--duration-seconds", "2.2",
                    "--drain-interval-seconds", "1", "--health-interval-seconds", ".2",
                    "--long-prefixes", "4", "--short-prefixes", "1", "--concurrency", "4"])
            report = json.loads(output.read_text())
            self.assertEqual(result, 0, report.get("error"))
            self.assertTrue(report["passed"])
            self.assertGreater(report["completed_requests"], 10)
            self.assertGreater(report["client_abort_requests"], 0)
            self.assertGreater(report["forced_midrun_drains"], 0)
            self.assertFalse(report["raw_generated_ids_available"])
            self.assertIsInstance(report["requests"], int)
            rows = [json.loads(line) for line in Path(report["events_path"]).read_text().splitlines()]
            boundary = next(row["elapsed_seconds"] for row in rows if row["event"] == "uninterrupted_half_started")
            periodic = [row for row in rows if row["event"] == "drain" and row["label"] == "periodic_first_half"]
            self.assertTrue(all(row["elapsed_seconds"] <= boundary for row in periodic))
            requests = [row for row in rows if row["event"] in ("request", "cancel")]
            self.assertTrue(all("messages" not in row and "content" not in row for row in requests))
            self.assertTrue(all(row.get("server_cancellation_confirmed") is False for row in requests if row["event"] == "cancel"))


def fake_health():
    return {"ready": True, "status": "ready", "pid": 123, "model": "fake-cpu-transport", "idle": True,
        **{key: 0 for key in ("active", "active_jobs", "pending_requests", "queued_prefills", "ready_decodes",
                             "resident_sequences", "reserved_tokens", "waiting_prefix_sequences")},
        "prefix_cache": {"entries": 0, "logicalPayloadBytes": 0, "keyTokens": 0, "evictions": 0, "hits": 0},
        "prefix_cache_limits": {"maxEntries": 1, "maxBytes": 1000, "maxKeyTokens": 30000},
        "prefix_disk_cache": {"entries": 0, "diskBytes": 0, "keyTokens": 0, "pendingJobs": 0, "pendingBytes": 0,
                              **{name: 0 for name in churn.DISK_COUNTERS}},
        "prefix_disk_cache_limits": {"maxEntries": 1, "maxBytes": 1000, "maxKeyTokens": 30000,
                                     "maxPendingJobs": 2, "maxPendingBytes": 1000},
        "state_budget": {"maxBytes": 1000, "requestBytes": 0, "cacheBytes": 0, "workspaceBytes": 0,
                         "totalBytes": 0, "peakBytes": 0, "rejections": 0, "currentLeases": 0},
        "mlx_memory": {"active_bytes": 0, "peak_bytes": 0, "cache_bytes": 0, "limit_bytes": 1000},
        "memory_pressure": {"observedLevel": "unknown", "lastEventSource": None}}


class FakeClient(churn.Client):
    """Deterministic transport-only fixture; the health values are synthetic."""
    def __init__(self, base_url, timeout):
        super().__init__(base_url, timeout)
        self.value, self.seen, self.calls = fake_health(), set(), 0
        self.lock = threading.Lock()

    def raw(self, method, path, body=None, timeout=None, observation=None):
        if path == "/health":
            with self.lock:
                raw = json.dumps(self.value).encode()
            return 200, {}, raw, {"wall_seconds": .001, "response_headers_seconds": .001}
        time.sleep(.006)
        key = body["messages"][0]["content"]
        prompt = 11000 if len(key) > 10000 else 1200
        with self.lock:
            cached = 416 if key in self.seen and self.calls % 3 else 0
            self.seen.add(key); self.calls += 1
            identifier = f"fake-{self.calls}"
            ram, disk = self.value["prefix_cache"], self.value["prefix_disk_cache"]
            if disk["entries"]:
                disk["evictions"] += 1; ram["evictions"] += 1
            disk.update(entries=1, diskBytes=600, keyTokens=832)
            disk["published"] += 1; disk["bytesWritten"] += 600
            disk["bytesRead"] += 600 if cached else 0; disk["hits"] += bool(cached)
            ram.update(entries=1, logicalPayloadBytes=500, keyTokens=832)
            ram["hits"] += bool(cached)
        counts = {"prompt_tokens": prompt, "completion_tokens": 8, "total_tokens": prompt + 8,
                  "prompt_tokens_details": {"cached_tokens": cached}}
        raw = stream(counts) if body.get("stream") else nonstream(counts)
        raw = raw.replace(b"cache-parser-request", identifier.encode())
        if observation is not None:
            observation.feed(raw, .006)
        return 200, {"content-type": "text/event-stream"}, raw, {"wall_seconds": .006, "response_headers_seconds": .001}

    def cancel(self, sequence, profile, body, after_content=False):
        time.sleep(.002)
        return {"sequence": sequence, "profile": profile, "stream": True, "passed": True,
                "client_abort_requested": True, "server_cancellation_confirmed": False,
                "request_id": f"fake-cancel-{sequence}", "wall_seconds": .002,
                "abort_stage": "after_content" if after_content else "after_assistant_role"}


if __name__ == "__main__":
    unittest.main()
