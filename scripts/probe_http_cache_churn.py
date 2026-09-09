#!/usr/bin/env python3
"""Bounded long-running cache churn against an externally controlled loopback server.

No server/model process is started or stopped. Serial fixture oracles precede a
fixed-duration mixed workload. JSON summary stays bounded; request/health events
are appended to a sibling NDJSON file, and fixture bodies are stored only once.
HTTP exposes text + usage, not raw generated IDs. Client RSTs require independent
server-log confirmation; backend archive byte counters are not physical SSD I/O.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import http.client
import json
import math
from pathlib import Path
import random
import signal
import socket
import struct
import sys
import threading
import time
from urllib.parse import urlsplit
import uuid

sys.dont_write_bytecode = True
from probe_http_cache_reliability import (MAX_RESPONSE_BYTES, digest, events, fixture_text,
                                          fully_idle, health_contract, parse_completion)
from cache_churn_oracles import FIXTURE_MODES, audit_oracle_discrimination, counter_witness_start

MAX_FRAME_BYTES = 65_536
MAX_RECORD_BYTES = 131_072
MAX_TEXT_BYTES = 65_536
LATENCY_BUCKETS = (.01, .05, .1, .25, .5, 1, 2, 4, 8, 16, 30, 60, 120, 240, 480, 900)
COVERAGE_WINDOW_SECONDS = 300
DISK_COUNTERS = ("bytesRead", "bytesWritten", "hits", "published", "evictions", "corruptions",
                 "writeFailures", "rejected", "expired", "spaceRejections", "spaceQueryFailures")


def origin(value):
    url = urlsplit(value)
    if (url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost", "::1") or
            url.username or url.password or url.path not in ("", "/") or url.query or url.fragment):
        raise ValueError("--base-url must be an HTTP loopback origin")
    return ("127.0.0.1" if url.hostname == "localhost" else url.hostname), url.port or 80


class LatencyHistogram:
    """Constant-memory upper-bucket estimates, not exact saved percentiles."""
    def __init__(self):
        self.bins = [0] * (len(LATENCY_BUCKETS) + 1)
        self.count, self.total, self.maximum = 0, 0.0, 0.0

    def add(self, value):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid latency")
        index = next((i for i, upper in enumerate(LATENCY_BUCKETS) if value <= upper), len(LATENCY_BUCKETS))
        self.bins[index] += 1
        self.count += 1; self.total += value; self.maximum = max(self.maximum, value)

    def summary(self):
        def quantile(q):
            if not self.count:
                return None
            target, seen = math.ceil(q * self.count), 0
            for index, count in enumerate(self.bins):
                seen += count
                if seen >= target:
                    return LATENCY_BUCKETS[index] if index < len(LATENCY_BUCKETS) else None
        return {"count": self.count, "sum_seconds": self.total, "max_seconds": self.maximum,
                "bucket_upper_seconds": [*LATENCY_BUCKETS, None], "counts": self.bins.copy(),
                "p50_upper_seconds": quantile(.5), "p95_upper_seconds": quantile(.95),
                "p99_upper_seconds": quantile(.99), "quantile_method": "fixed_upper_bucket"}


class EventLog:
    def __init__(self, path):
        self.file = path.open("x", encoding="utf-8")
        self.lock = threading.Lock()
        self.count = 0

    def append(self, kind, **record):
        raw = json.dumps({"event": kind, **record}, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        if len(raw.encode("utf-8")) > MAX_RECORD_BYTES:
            raise ValueError("NDJSON record exceeds bound")
        with self.lock:
            self.file.write(raw + "\n"); self.file.flush(); self.count += 1

    def close(self):
        with self.lock:
            self.file.close()


class SSEObservation:
    """Bounded incremental observations; legacy parser validates the full body."""
    def __init__(self):
        self.buffer = bytearray()
        self.request_id = None
        self.role_seen = False
        self.finish_seen = False
        self.first_content_seconds = None

    def feed(self, data, elapsed):
        self.buffer.extend(data)
        while True:
            positions = [(self.buffer.find(marker), len(marker)) for marker in (b"\n\n", b"\r\n\r\n")]
            positions = [item for item in positions if item[0] >= 0]
            if not positions:
                if len(self.buffer) > MAX_FRAME_BYTES:
                    raise ValueError("Oversized incomplete SSE frame")
                return
            offset, size = min(positions)
            if offset + size > MAX_FRAME_BYTES:
                raise ValueError("Oversized SSE frame")
            frame = bytes(self.buffer[:offset + size]); del self.buffer[:offset + size]
            for value in events(frame):
                if value == "[DONE]":
                    self.finish_seen = True
                    continue
                chunk = json.loads(value)
                request_id = chunk.get("id")
                if request_id is not None:
                    if not isinstance(request_id, str) or not 0 < len(request_id) <= 256:
                        raise ValueError("Invalid SSE request id")
                    if self.request_id is not None and self.request_id != request_id:
                        raise ValueError("SSE request id changed")
                    self.request_id = request_id
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    self.role_seen |= delta.get("role") == "assistant"
                    if delta.get("content") and self.first_content_seconds is None:
                        self.first_content_seconds = elapsed
                    self.finish_seen |= choice.get("finish_reason") is not None


class Client:
    def __init__(self, base_url, timeout):
        self.host, self.port = origin(base_url)
        self.timeout = timeout

    def raw(self, method, path, body=None, timeout=None, observation=None):
        timeout = self.timeout if timeout is None else timeout
        connection = http.client.HTTPConnection(self.host, self.port, timeout=timeout)
        payload = None if body is None else json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
        started = time.monotonic()
        try:
            connection.connect()
            transport = connection.sock
            transport.settimeout(max(.001, timeout - (time.monotonic() - started)))
            connection.request(method, path, payload, {"Content-Type": "application/json"} if payload else {})
            transport.settimeout(max(.001, timeout - (time.monotonic() - started)))
            response = connection.getresponse()
            header_seconds = time.monotonic() - started
            chunks, received = [], 0
            while True:
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("Absolute HTTP response deadline exceeded")
                # Keep the underlying socket reference: HTTPConnection may
                # detach it for Connection: close while HTTPResponse owns its file.
                transport.settimeout(remaining)
                block = response.read1(min(65_536, MAX_RESPONSE_BYTES + 1 - received))
                if not block:
                    break
                received += len(block)
                if received > MAX_RESPONSE_BYTES:
                    raise ValueError("Response exceeds inherited client bound")
                chunks.append(block)
                if observation is not None and response.status == 200:
                    observation.feed(block, time.monotonic() - started)
            return response.status, {key.lower(): value for key, value in response.getheaders()}, b"".join(chunks), {
                "wall_seconds": time.monotonic() - started, "response_headers_seconds": header_seconds}
        finally:
            connection.close()

    def completion(self, sequence, profile, body, expected=None, stream=False):
        observer = SSEObservation() if stream else None
        row = {"sequence": sequence, "profile": profile, "stream": stream, "passed": False}
        started = time.monotonic()
        try:
            status, headers, raw, timing = self.raw("POST", "/v1/chat/completions", {**body, "stream": stream}, observation=observer)
            row.update(status=status, response_bytes=len(raw), response_sha256=digest(raw), **timing)
            try:
                identity = json.loads(raw).get("id") if not stream else observer.request_id
                if isinstance(identity, str) and len(identity) <= 256:
                    row["request_id"] = identity
            except (ValueError, AttributeError):
                pass
            result = parse_completion(status, headers, raw, stream)
            if len(result["content"].encode("utf-8")) > MAX_TEXT_BYTES:
                raise ValueError("Completion text exceeds oracle bound")
            if expected is not None:
                for name in ("content", "finish_reason"):
                    if result[name] != expected[name]:
                        raise ValueError(f"Text/finish differs from serial oracle: {name}")
                for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    if result["usage"][name] != expected["usage"][name]:
                        raise ValueError(f"Usage differs from serial oracle: {name}")
            row.update(request_id=result["id"], content_sha256=result["content_sha256"],
                       finish_reason=result["finish_reason"], usage=result["usage"],
                       effective_cached_tokens=result["cached_tokens"],
                       first_content_wall_seconds=observer.first_content_seconds if observer else None,
                       passed=True)
            return row, result
        except Exception as error:
            row.update(error=f"{type(error).__name__}: {error}"[:1500], wall_seconds=time.monotonic() - started)
            if observer and observer.request_id:
                row["request_id"] = observer.request_id
            return row, None

    def cancel(self, sequence, profile, body, after_content=False):
        row = {"sequence": sequence, "profile": profile, "stream": True, "passed": False,
               "client_abort_requested": False, "server_cancellation_confirmed": False,
               "abort_stage": "after_content" if after_content else "after_assistant_role"}
        payload = json.dumps({**body, "stream": True}, ensure_ascii=False, separators=(",", ":")).encode()
        sock = response = None
        observer = SSEObservation()
        started = time.monotonic()
        try:
            sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
            hostname = f"[{self.host}]" if ":" in self.host else self.host
            headers = (f"POST /v1/chat/completions HTTP/1.1\r\nHost: {hostname}:{self.port}\r\n"
                       f"Content-Type: application/json\r\nContent-Length: {len(payload)}\r\nConnection: close\r\n\r\n").encode()
            sock.sendall(headers + payload)
            response = http.client.HTTPResponse(sock); response.begin()
            row["status"] = response.status
            if response.status != 200:
                raise ValueError(f"Cancellation HTTP {response.status}")
            received = 0
            while received <= MAX_RESPONSE_BYTES:
                remaining = self.timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("Cancellation response deadline exceeded")
                sock.settimeout(remaining)
                line = response.readline(MAX_FRAME_BYTES + 1)
                if not line:
                    raise ValueError("SSE ended before client cancellation")
                received += len(line)
                if len(line) > MAX_FRAME_BYTES or received > MAX_RESPONSE_BYTES:
                    raise ValueError("Cancellation SSE exceeds bound")
                observer.feed(line, time.monotonic() - started)
                if observer.request_id:
                    row["request_id"] = observer.request_id
                if observer.finish_seen:
                    raise ValueError("Completion finished before requested cancellation point")
                ready = observer.first_content_seconds is not None if after_content else observer.role_seen
                if ready and observer.request_id:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                    row.update(client_abort_requested=True, passed=True, response_prefix_bytes=received)
                    response.close(); response = None; sock.close(); sock = None
                    return row
            raise ValueError("Cancellation byte bound reached")
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"[:1500]
            return row
        finally:
            row["wall_seconds"] = time.monotonic() - started
            if response is not None:
                response.close()
            if sock is not None:
                sock.close()


def make_profiles(system, model, seed, namespace, long_count, short_count, mode="legacy"):
    if mode not in FIXTURE_MODES:
        raise ValueError("Unknown cache churn fixture mode")
    profiles = {}
    short = "Use the supplied reference records.\n" + "".join(
        f"Reference record {i}: preserve the exact ordered input history and measured result.\n" for i in range(72))
    for kind, count, source in (("long", long_count, system), ("short", short_count, short)):
        for index in range(count):
            name = f"{kind}_{index:02d}"
            # Diverge near the first tokens, so the long tails cannot masquerade
            # as multiple fixtures sharing a single 10k cached system prefix.
            marker = hashlib.sha256(f"{namespace}:{seed}:{name}".encode()).hexdigest()
            header = f"Fixture {marker}.\n"
            user = "请按顺序输出 1 到 100 的整数，用空格分隔，不解释。"
            if mode == "counter-witness":
                # The witness occurs only in the first system header, before
                # the first 416-token checkpoint. Every user message is equal.
                header = f"CACHE_COUNTER_START: {counter_witness_start(name)}\n" + header
                user = "请读取系统消息最开头 CACHE_COUNTER_START 指定的整数，从该整数开始按顺序连续输出100个整数，每个整数之间只放一个空格，不解释。"
            profiles[name] = {"model": model, "temperature": 0, "mtp_depth": 0, "max_tokens": 16,
                "messages": [{"role": "system", "content": header + source},
                             {"role": "user", "content": user}]}
    return profiles


class ChurnSchedule:
    """Seeded bounded fixture selection with recurring hot and cyclic cold work."""
    def __init__(self, names, seed):
        self.long = [name for name in names if name.startswith("long_")]
        self.short = [name for name in names if name.startswith("short_")]
        self.random = random.Random(seed)
        self.random.shuffle(self.long); self.random.shuffle(self.short)
        self.sequence, self.cursor, self.last_long = 0, 0, self.long[0]

    def next(self):
        sequence = self.sequence; self.sequence += 1
        if sequence % 4 == 0:
            name = self.long[self.cursor % len(self.long)]
            self.cursor += 1; self.last_long = name
        elif sequence % 4 == 1:
            name = self.last_long
        elif sequence % 4 == 2:
            name = self.short[self.random.randrange(len(self.short))]
        else:
            name = self.long[(self.cursor - 2) % len(self.long)]
        return sequence, name, sequence % 11 == 7, bool((sequence // 11) % 2)


def disk_deltas(before, after):
    return {name: after.get(name, 0) - before.get(name, 0) for name in DISK_COUNTERS
            if type(before.get(name, 0)) is int and type(after.get(name, 0)) is int}


def workload_window(before, after, disk_before, disk_after, seconds):
    counts = {name: after[name] - before[name] for name in
              ("completed_requests", "cached_completions", "cold_completions", "client_abort_requests")}
    disk = disk_deltas(disk_before, disk_after)
    return {"seconds": seconds, "counts": counts, "backend_archive_delta": disk,
            "continuous_churn_observed": all(disk.get(name, 0) > 0 for name in ("bytesRead", "bytesWritten", "evictions"))
              and counts["completed_requests"] > 0 and counts["cold_completions"] > 0 and counts["cached_completions"] > 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:11236")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-seconds", type=float, default=7200)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--long-prefixes", type=int, default=8)
    parser.add_argument("--short-prefixes", type=int, default=2)
    parser.add_argument("--max-inflight-prompt-tokens", type=int, default=30000)
    parser.add_argument("--request-timeout-seconds", type=float, default=240)
    parser.add_argument("--drain-timeout-seconds", type=float, default=180)
    parser.add_argument("--drain-interval-seconds", type=float, default=300)
    parser.add_argument("--health-interval-seconds", type=float, default=2)
    parser.add_argument("--tokens-file", type=Path)
    parser.add_argument("--fixture-namespace", help="Optional replay namespace; default isolates this run with its run ID")
    parser.add_argument("--fixture-mode", choices=FIXTURE_MODES, default="legacy",
                        help="counter-witness requires distinct header-dependent cold outputs before the workload")
    args = parser.parse_args(argv)
    try:
        origin(args.base_url)
    except ValueError as error:
        parser.error(str(error))
    for name, low, high in (("duration_seconds", 1, 86400), ("request_timeout_seconds", 1, 900),
                            ("drain_timeout_seconds", 1, 900), ("drain_interval_seconds", 1, 3600),
                            ("health_interval_seconds", .2, 60)):
        value = getattr(args, name)
        if not math.isfinite(value) or not low <= value <= high:
            parser.error(f"--{name.replace('_', '-')} must be {low}...{high}")
    if not (1 <= args.concurrency <= 8 and 4 <= args.long_prefixes <= 32 and 1 <= args.short_prefixes <= 8):
        parser.error("concurrency must be 1...8, long-prefixes 4...32, short-prefixes 1...8")
    if not 16400 <= args.max_inflight_prompt_tokens <= 262144:
        parser.error("--max-inflight-prompt-tokens must be 16400...262144")
    if args.fixture_namespace is not None and not 1 <= len(args.fixture_namespace.encode()) <= 256:
        parser.error("--fixture-namespace must contain 1...256 bytes")
    output = args.output.resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    detail = output.with_name(output.stem + ".events.ndjson")
    fixture_path = output.with_name(output.stem + ".fixtures.json")
    for path in (output, detail, fixture_path):
        if path.exists():
            parser.error(f"Output path must be new: {path}")
    with output.open("x") as file:
        file.write("{}\n")
    log = EventLog(detail)
    run_id, started = uuid.uuid4().hex, time.monotonic()
    client = Client(args.base_url, args.request_timeout_seconds)
    report = {"schema": "qwen-http-cache-churn-v1", "run_id": run_id, "complete": False, "passed": False,
        "base_url": args.base_url, "seed": args.seed, "planned_soak_seconds": args.duration_seconds,
        "fixture_mode": args.fixture_mode,
        "concurrency": args.concurrency, "max_inflight_prompt_tokens": args.max_inflight_prompt_tokens,
        "script_sha256": digest(Path(__file__).read_bytes()), "events_path": str(detail), "fixtures_path": str(fixture_path),
        "request_timeout_seconds": args.request_timeout_seconds, "drain_timeout_seconds": args.drain_timeout_seconds,
        "requests": 0, "completed_requests": 0, "errors": 0, "client_abort_requests": 0,
        "effective_cached_tokens": 0, "completed_prompt_tokens": 0, "cold_completions": 0, "cached_completions": 0,
        "forced_midrun_drains": 0, "health_samples": 0, "health_errors": 0,
        "full_coverage_windows": 0, "coverage_windows_without_churn": 0,
        "checks": {}, "profiles": {}, "raw_generated_ids_available": False,
        "notes": ["Full text + finish + required usage are compared; raw generated token IDs are unavailable.",
                  "HTTP wall and first-content wall are end-to-end observations, not GPU stage measurements.",
                  "Backend bytesRead/bytesWritten count successful archive operations, not device physical I/O or failed partial reads.",
                  "Every client abort request ID is retained in NDJSON; server terminal cancellation must be checked independently.",
                  "First half has periodic forced drains; second half never deliberately drains before the final drain.",
                  "Seed controls selection; the default unique fixture namespace isolates cache keys. Recorded fixtures allow exact replay."]}
    monitor_stop, fault = threading.Event(), threading.Event()
    health_lock = threading.Lock()
    monitor = None
    first_health_error = []
    latency, first_content = LatencyHistogram(), LatencyHistogram()
    profile_counts = {}

    def save():
        with health_lock:
            report["health_samples"] = health_count[0]
            report["health_errors"] = health_error_count[0]
            report["observed_pressure_sample_counts"] = pressure_counts.copy()
        report["latency"] = latency.summary(); report["first_content_latency"] = first_content.summary()
        report["profile_counts"] = {name: values.copy() for name, values in profile_counts.items()}
        report["elapsed_seconds"] = time.monotonic() - started
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temporary.replace(output)

    health_count, health_error_count = [0], [0]
    pressure_counts = {key: 0 for key in ("unknown", "normal", "warning", "critical", "unavailable")}

    def health(label):
        status, _, raw, timing = client.raw("GET", "/health", timeout=10)
        if status != 200:
            raise ValueError(f"Health HTTP {status}: {raw[:300]!r}")
        value = health_contract(json.loads(raw))
        if report.get("server_pid") is not None and value.get("pid") != report["server_pid"]:
            raise ValueError("Server PID changed during this run")
        pressure = value.get("memory_pressure")
        level = pressure.get("observedLevel", "unavailable") if isinstance(pressure, dict) else "unavailable"
        if level not in pressure_counts:
            raise ValueError("Invalid observed memory pressure level")
        with health_lock:
            health_count[0] += 1; pressure_counts[level] += 1
        log.append("health", label=label, elapsed_seconds=time.monotonic() - started,
                   request_seconds=timing["wall_seconds"], health=value,
                   os_pressure_observed=pressure if isinstance(pressure, dict) and pressure.get("lastEventSource") == "operatingSystem" else None,
                   physical_ssd_io_bytes=None)
        return value

    def drain(label, stop_at=None):
        deadline = time.monotonic() + args.drain_timeout_seconds
        if stop_at is not None:
            deadline = min(deadline, stop_at)
        while time.monotonic() < deadline:
            value = health(label)
            if fully_idle(value):
                log.append("drain", label=label, elapsed_seconds=time.monotonic() - started, health=value)
                return value
            time.sleep(min(.2, max(0, deadline - time.monotonic())))
        if stop_at is not None and time.monotonic() >= stop_at:
            log.append("periodic_drain_cutoff", elapsed_seconds=time.monotonic() - started)
            return None
        raise TimeoutError(f"{label}: request/workspace/SSD work did not drain")

    def observe():
        while not monitor_stop.wait(args.health_interval_seconds):
            try:
                health("continuous")
            except Exception as error:
                message = f"{type(error).__name__}: {error}"[:1500]
                with health_lock:
                    health_error_count[0] += 1
                    if not first_health_error:
                        first_health_error.append(message)
                try:
                    log.append("health_error", elapsed_seconds=time.monotonic() - started, error=message)
                finally:
                    fault.set()
                return

    def check(name, passed, **evidence):
        report["checks"][name] = {"passed": bool(passed), **evidence}
        log.append("check", name=name, passed=bool(passed), **evidence)
        save()
        if not passed:
            raise AssertionError(name)

    def receive(future, kind):
        if kind == "cancel":
            row = future.result()
        else:
            row, _ = future.result()
        row["elapsed_seconds"] = time.monotonic() - started
        log.append(kind, **row)
        report["requests"] += 1
        counts = profile_counts[row["profile"]]
        counts["attempts"] += 1
        if not row["passed"]:
            report["errors"] += 1; report.setdefault("first_request_error", row)
            fault.set(); return
        if kind == "cancel":
            report["client_abort_requests"] += 1; counts["client_aborts"] += 1
            return
        report["completed_requests"] += 1; counts["completed"] += 1
        cached = row["effective_cached_tokens"]
        report["effective_cached_tokens"] += cached
        report["completed_prompt_tokens"] += row["usage"]["prompt_tokens"]
        report["cached_completions" if cached else "cold_completions"] += 1
        counts["cached" if cached else "cold"] += 1
        latency.add(row["wall_seconds"])
        if row["first_content_wall_seconds"] is not None:
            first_content.add(row["first_content_wall_seconds"])

    previous_signals = {}
    def interrupted(number, frame):
        report["interrupted_by_signal"] = number
        fault.set()
    for number in (signal.SIGINT, signal.SIGTERM):
        previous_signals[number] = signal.signal(number, interrupted)

    try:
        initial = drain("initial_idle")
        report.update(server_pid=initial["pid"], model=initial["model"], initial_health=initial)
        system, provenance = fixture_text(args.tokens_file)
        if len(system.encode("utf-8")) > 512 * 1024:
            raise ValueError("Base system fixture exceeds 512 KiB bound")
        namespace = args.fixture_namespace or run_id
        profiles = make_profiles(system, initial["model"], args.seed, namespace, args.long_prefixes, args.short_prefixes,
                                 mode=args.fixture_mode)
        fixture_bytes = json.dumps({"namespace": namespace, "seed": args.seed, "source": provenance,
                                   "fixture_mode": args.fixture_mode, "profiles": profiles}, ensure_ascii=False, indent=2).encode()
        with fixture_path.open("xb") as file:
            file.write(fixture_bytes)
        report["fixture_sha256"] = digest(fixture_bytes)
        profile_counts = {name: {"attempts": 0, "completed": 0, "client_aborts": 0, "cached": 0, "cold": 0} for name in profiles}
        monitor = threading.Thread(target=observe, name="cache-churn-health", daemon=True); monitor.start()
        oracles, published_fixture_count, measured_archive_bytes = {}, 0, 0
        previous = initial
        for name, body in profiles.items():
            row, result = client.completion("oracle-" + name, name, body)
            log.append("oracle", **row, content=result["content"] if result else None)
            check("oracle_" + name, row["passed"], request_id=row.get("request_id"), error=row.get("error"))
            count = result["usage"]["prompt_tokens"]
            check("tokens_" + name, (10_000 <= count < 16_368) if name.startswith("long_") else (833 <= count < 4096),
                  prompt_tokens=count)
            if count + 16 > args.max_inflight_prompt_tokens:
                raise ValueError("A fixture exceeds the client in-flight token budget")
            oracles[name] = result
            after = drain("oracle_drained_" + name)
            delta = disk_deltas(previous["prefix_disk_cache"], after["prefix_disk_cache"])
            check("fresh_publication_" + name, result["cached_tokens"] == 0 and delta.get("published", 0) > 0 and delta.get("bytesWritten", 0) > 0,
                  effective_cached_tokens=result["cached_tokens"], disk_delta=delta)
            published_fixture_count += 1; measured_archive_bytes += delta["bytesWritten"]
            report["profiles"][name] = {"prompt_tokens": count, "completion_tokens": result["usage"]["completion_tokens"],
                "expected_start": counter_witness_start(name) if args.fixture_mode == "counter-witness" else None,
                "content_sha256": result["content_sha256"], "finish_reason": result["finish_reason"],
                "system_sha256": digest(body["messages"][0]["content"].encode()), "new_archive_bytes": delta["bytesWritten"]}
            previous = after
            if fault.is_set():
                raise RuntimeError(first_health_error[0] if first_health_error else "Health fault during oracle population")
        discrimination = audit_oracle_discrimination(args.fixture_mode,
            {name: value["content"] for name, value in oracles.items()}, expected_profiles=profiles,
            reported_hashes={name: value["content_sha256"] for name, value in oracles.items()})
        report["oracle_discrimination"] = discrimination
        log.append("oracle_discrimination", **discrimination)
        check("oracle_discrimination", discrimination["passed"],
              required=discrimination["required"], discriminating=discrimination["discriminating"],
              unique_content_hashes=discrimination["unique_content_hashes"], issues=discrimination["issues"])
        seeded = drain("oracles_complete")
        report["seeded_health"] = seeded
        report["working_set"] = {"distinct_fixture_prefixes": published_fixture_count,
            "measured_distinct_published_archive_bytes": measured_archive_bytes,
            "ssd_limit_bytes": initial["prefix_disk_cache_limits"]["maxBytes"],
            "ram_population_evictions": seeded["prefix_cache"]["evictions"] - initial["prefix_cache"]["evictions"],
            "measurement": "sum of successful new archive writes from serial cold fixtures; no physical SSD or RAM byte claim"}
        check("working_set_exceeds_ssd_capacity", measured_archive_bytes > initial["prefix_disk_cache_limits"]["maxBytes"],
              **report["working_set"])
        check("ram_capacity_churn_observed", seeded["prefix_cache"]["evictions"] > initial["prefix_cache"]["evictions"],
              ram_limit_bytes=initial["prefix_cache_limits"]["maxBytes"])
        # Keep one whole fixture of additional client-side headroom after an
        # RST. The HTTP future can finish before server cancellation releases
        # its prompt reservation. This is conservative workload admission,
        # not a claim that the server has acknowledged that cancellation.
        cancel_headroom = max(value["usage"]["prompt_tokens"] + 16 for value in oracles.values())
        active_token_ceiling = args.max_inflight_prompt_tokens - cancel_headroom
        if active_token_ceiling < cancel_headroom:
            raise ValueError("Client in-flight budget must fit the largest fixture twice, including cancellation headroom")
        report["client_cancel_headroom_tokens"] = cancel_headroom
        report["active_client_token_ceiling"] = active_token_ceiling
        schedule = ChurnSchedule(list(profiles), args.seed)
        soak_start = time.monotonic(); deadline = soak_start + args.duration_seconds
        half = soak_start + args.duration_seconds / 2
        next_drain = soak_start + args.drain_interval_seconds; next_progress = soak_start + 60
        report["soak_started_elapsed_seconds"] = soak_start - started
        window_started, window_health = soak_start, seeded
        window_counts = {name: report[name] for name in
                         ("completed_requests", "cached_completions", "cold_completions", "client_abort_requests")}
        log.append("soak_started", elapsed_seconds=soak_start - started, planned_seconds=args.duration_seconds)
        with ThreadPoolExecutor(max_workers=args.concurrency, thread_name_prefix="cache-churn-request") as workers:
            pending, deferred, forced_drain = {}, None, False
            while pending or (time.monotonic() < deadline and not fault.is_set()):
                now = time.monotonic()
                if now < half and now >= next_drain:
                    forced_drain = True
                if now >= half:
                    forced_drain = False
                if now >= half and not report.get("uninterrupted_half_started"):
                    report["uninterrupted_half_started"] = True
                    log.append("uninterrupted_half_started", elapsed_seconds=now - started)
                while not forced_drain and not fault.is_set() and time.monotonic() < deadline and len(pending) < args.concurrency:
                    if deferred is None:
                        deferred = schedule.next()
                    sequence, name, cancel, after_content = deferred
                    reserved = oracles[name]["usage"]["prompt_tokens"] + 16
                    if sum(item[1] for item in pending.values()) + reserved > active_token_ceiling:
                        break
                    deferred = None
                    if cancel:
                        future = workers.submit(client.cancel, sequence, name, profiles[name], after_content)
                        pending[future] = ("cancel", reserved)
                    else:
                        future = workers.submit(client.completion, sequence, name, profiles[name], oracles[name], sequence % 2 == 1)
                        pending[future] = ("request", reserved)
                if pending:
                    completed, _ = wait(pending, timeout=.2, return_when=FIRST_COMPLETED)
                    for future in completed:
                        kind, _ = pending.pop(future); receive(future, kind)
                elif forced_drain:
                    checkpoint = drain("periodic_first_half", stop_at=half)
                    if checkpoint is not None:
                        report["forced_midrun_drains"] += 1
                        report["last_periodic_drained_health"] = checkpoint
                    forced_drain = False; next_drain = time.monotonic() + args.drain_interval_seconds
                    save()
                elif time.monotonic() < deadline and not fault.is_set():
                    time.sleep(.01)
                if time.monotonic() - window_started >= COVERAGE_WINDOW_SECONDS:
                    current = health("coverage_window")
                    window_end = time.monotonic()
                    window = workload_window(window_counts, report, window_health["prefix_disk_cache"],
                                             current["prefix_disk_cache"], window_end - window_started)
                    log.append("coverage_window", elapsed_seconds=window_end - started, **window)
                    report["full_coverage_windows"] += 1
                    report["coverage_windows_without_churn"] += not window["continuous_churn_observed"]
                    window_started, window_health = window_end, current
                    window_counts = {name: report[name] for name in window_counts}
                if time.monotonic() >= next_progress:
                    save()
                    print(json.dumps({"elapsed_soak_seconds": time.monotonic() - soak_start,
                        "completed": report["completed_requests"], "client_aborts": report["client_abort_requests"],
                        "errors": report["errors"], "forced_drains": report["forced_midrun_drains"]}), flush=True)
                    next_progress = time.monotonic() + 60
        report["actual_soak_seconds_before_final_drain"] = time.monotonic() - soak_start
        final = drain("final_idle"); report["final_health"] = final
        monitor_stop.set(); monitor.join(timeout=12)
        report["soak_disk_delta"] = disk_deltas(seeded["prefix_disk_cache"], final["prefix_disk_cache"])
        check("planned_duration_reached", time.monotonic() >= deadline and not fault.is_set())
        check("all_output_oracles_preserved", report["completed_requests"] > 0 and report["errors"] == 0)
        check("health_and_final_drain", health_error_count[0] == 0 and not monitor.is_alive() and fully_idle(final))
        check("mixed_hot_cold_coverage", report["cached_completions"] > 0 and report["cold_completions"] > 0)
        check("all_profiles_exercised", all(value["completed"] > 0 for value in profile_counts.values()))
        check("client_cancel_and_recovery", report["client_abort_requests"] > 0 and fully_idle(final))
        check("ssd_read_write_churn", all(report["soak_disk_delta"].get(name, 0) > 0 for name in ("bytesRead", "bytesWritten", "evictions")),
              backend_archive_delta=report["soak_disk_delta"])
        if args.duration_seconds >= COVERAGE_WINDOW_SECONDS * 2:
            check("sustained_ssd_churn_windows", report["full_coverage_windows"] > 0 and report["coverage_windows_without_churn"] == 0,
                  complete_windows=report["full_coverage_windows"], window_seconds=COVERAGE_WINDOW_SECONDS,
                  incomplete_tail_window_seconds=time.monotonic() - window_started)
        check("no_new_corruption_or_write_failure", all(final["prefix_disk_cache"][name] == initial["prefix_disk_cache"][name]
              for name in ("corruptions", "writeFailures")))
        report["complete"] = True; report["passed"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"[:1500]
        report["first_health_error"] = first_health_error[0] if first_health_error else None
        log.append("run_error", elapsed_seconds=time.monotonic() - started, error=report["error"])
        report["passed"] = False
    finally:
        monitor_stop.set()
        if monitor is not None:
            monitor.join(timeout=12)
        # Oracle/check failures can leave no client work but outstanding backend
        # publication. Observe its bounded drain; never kill the external server.
        if "final_health" not in report and report.get("server_pid") is not None:
            try:
                report["final_health"] = drain("failure_final_idle")
            except Exception as error:
                report["final_drain_error"] = f"{type(error).__name__}: {error}"[:1500]
        report["event_count"] = log.count
        save(); log.close()
        for number, handler in previous_signals.items():
            signal.signal(number, handler)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
