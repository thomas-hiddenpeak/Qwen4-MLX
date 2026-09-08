"""CPU-only churn controls. Fake transport tests do not exercise a live server."""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import probe_http_cache_churn as churn
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
