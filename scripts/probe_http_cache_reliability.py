#!/usr/bin/env python3
"""Exercise an already running loopback AR server and its RAM/SSD prefix cache.

The script never starts, stops or reconfigures a server. Four serial output
oracles precede the requested number of successful mixed concurrent requests.
Periodic SSE resets request cancellation; server logs must independently confirm
the terminal state for the recorded request IDs. HTTP latency is not a GPU phase
benchmark, and logical state accounting is not process RSS.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import http.client
import json
from pathlib import Path
import socket
import struct
import threading
import time
from urllib.parse import urlsplit
import uuid


MAX_RESPONSE_BYTES = 4 * 1024 * 1024
REQUEST_TIMEOUT = 240
IDLE_TIMEOUT = 180
CONCURRENCY = 4


def digest(data):
    return hashlib.sha256(data).hexdigest()


def events(raw):
    parsed = []
    for frame in raw.replace(b"\r\n", b"\n").split(b"\n\n"):
        lines = frame.decode("utf-8", errors="strict").split("\n")
        data = "\n".join(line[5:].lstrip(" ") for line in lines if line.startswith("data:"))
        if data:
            parsed.append(data)
    return parsed


def parse_completion(status, headers, raw, stream):
    if status != 200:
        raise ValueError(f"HTTP {status}: {raw[:500]!r}")
    if not stream:
        result = json.loads(raw)
        if result.get("object") != "chat.completion" or len(result.get("choices", [])) != 1:
            raise ValueError("Malformed nonstream completion")
        choice = result["choices"][0]
        message = choice["message"]
        if choice["index"] != 0 or message.get("role") != "assistant" or message.get("tool_calls"):
            raise ValueError("Unexpected choice/message in AR text probe")
        text, finish, usage = message.get("content") or "", choice["finish_reason"], result["usage"]
        request_id = result["id"]
    else:
        if "text/event-stream" not in headers.get("content-type", ""):
            raise ValueError("Wrong SSE content type")
        frames = events(raw)
        if not frames or frames[-1] != "[DONE]" or frames.count("[DONE]") != 1:
            raise ValueError("Missing or duplicated terminal SSE DONE")
        chunks = [json.loads(frame) for frame in frames[:-1]]
        if not chunks or any("error" in chunk for chunk in chunks):
            raise ValueError(f"SSE error or empty response: {chunks[:2]!r}")
        identities = {(chunk["id"], chunk["created"], chunk["model"]) for chunk in chunks}
        if len(identities) != 1:
            raise ValueError("SSE identity changed")
        first_choices = chunks[0].get("choices", [])
        if len(first_choices) != 1 or first_choices[0].get("delta", {}).get("role") != "assistant":
            raise ValueError("SSE must start with the assistant role")
        text_parts, finishes, usages = [], [], []
        for chunk in chunks:
            if chunk.get("usage") is not None:
                usages.append(chunk["usage"])
            for choice in chunk["choices"]:
                if finishes:
                    raise ValueError("SSE choice appeared after finish")
                if choice["index"] != 0 or choice["delta"].get("tool_calls"):
                    raise ValueError("Unexpected SSE choice/tool call")
                content = choice["delta"].get("content")
                if content is not None:
                    text_parts.append(content)
                if choice.get("finish_reason") is not None:
                    finishes.append(choice["finish_reason"])
        if len(finishes) != 1 or len(usages) != 1:
            raise ValueError("SSE requires one finish and one usage")
        text, finish, usage, request_id = "".join(text_parts), finishes[0], usages[0], chunks[0]["id"]
    if not isinstance(text, str) or finish not in ("stop", "length"):
        raise ValueError("Invalid AR content/finish")
    if not isinstance(usage, dict) or any(name not in usage for name in ("prompt_tokens", "completion_tokens", "total_tokens")):
        raise ValueError("Missing required token usage")
    prompt, completion, total = (usage[name] for name in ("prompt_tokens", "completion_tokens", "total_tokens"))
    # A cold miss legitimately omits prompt_tokens_details. Preserve the raw
    # usage object while normalizing only the comparison's cached-token value.
    if not isinstance(usage.get("prompt_tokens_details", {}), dict):
        raise ValueError("Invalid prompt token details")
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    if any(type(value) is not int for value in (prompt, completion, total, cached)) or not (
        prompt > 0 and 0 < completion <= 16 and total == prompt + completion and 0 <= cached < prompt
    ):
        raise ValueError("Invalid token or cache accounting")
    return {"id": request_id, "content": text, "finish_reason": finish, "usage": usage,
            "content_sha256": digest(text.encode("utf-8")), "cached_tokens": cached}


def health_contract(value):
    """Strictly use published health fields, never infer configured limits."""
    if value.get("ready") is not True or value.get("status") != "ready":
        raise ValueError("Server is not ready")
    required = ("prefix_cache", "prefix_cache_limits", "prefix_disk_cache", "prefix_disk_cache_limits",
                "state_budget", "mlx_memory")
    if any(not isinstance(value.get(name), dict) for name in required):
        raise ValueError("Need enabled RAM+SSD cache and the explicit cache/budget health contract")
    for name in ("active", "active_jobs", "pending_requests", "queued_prefills", "ready_decodes",
                 "resident_sequences", "reserved_tokens", "waiting_prefix_sequences"):
        if type(value.get(name)) is not int or value[name] < 0:
            raise ValueError(f"Missing/invalid health counter: {name}")
    ram, rlim = value["prefix_cache"], value["prefix_cache_limits"]
    disk, dlim = value["prefix_disk_cache"], value["prefix_disk_cache_limits"]
    budget = value["state_budget"]
    bounds = ((ram, "entries", rlim, "maxEntries"), (ram, "logicalPayloadBytes", rlim, "maxBytes"),
              (ram, "keyTokens", rlim, "maxKeyTokens"), (disk, "entries", dlim, "maxEntries"),
              (disk, "diskBytes", dlim, "maxBytes"), (disk, "keyTokens", dlim, "maxKeyTokens"),
              (disk, "pendingJobs", dlim, "maxPendingJobs"), (disk, "pendingBytes", dlim, "maxPendingBytes"),
              (budget, "totalBytes", budget, "maxBytes"), (budget, "peakBytes", budget, "maxBytes"))
    for counters, field, limits, limit in bounds:
        amount, maximum = counters.get(field), limits.get(limit)
        if type(amount) is not int or type(maximum) is not int or not 0 <= amount <= maximum:
            raise ValueError(f"Invalid/exceeded published {field}/{limit} budget: {amount}/{maximum}")
    for field in ("requestBytes", "cacheBytes", "workspaceBytes", "currentLeases", "rejections"):
        if type(budget.get(field)) is not int or budget[field] < 0:
            raise ValueError(f"Invalid state budget counter {field}")
    if sum(budget[field] for field in ("requestBytes", "cacheBytes", "workspaceBytes")) != budget["totalBytes"]:
        raise ValueError("State budget categories do not add to totalBytes")
    for field in ("active_bytes", "peak_bytes", "cache_bytes", "limit_bytes"):
        if type(value["mlx_memory"].get(field)) is not int or value["mlx_memory"][field] < 0:
            raise ValueError(f"Missing MLX memory observation {field}")
    return value


def fully_idle(value):
    names = ("active", "active_jobs", "pending_requests", "queued_prefills", "ready_decodes",
             "resident_sequences", "reserved_tokens", "waiting_prefix_sequences")
    return value.get("idle") is True and all(value[name] == 0 for name in names) and (
        value["state_budget"]["requestBytes"] == value["state_budget"]["workspaceBytes"] == 0
        and value["prefix_disk_cache"]["pendingJobs"] == value["prefix_disk_cache"]["pendingBytes"] == 0
    )


def fixture_text(tokens_file):
    if tokens_file is None:
        path = Path(__file__).resolve().parents[1] / "fixtures/gpu-agent-11k/system-prompt.txt"
        return path.read_text(), {"system_path": str(path), "system_sha256": digest(path.read_bytes())}
    raw = tokens_file.read_bytes()
    ids = json.loads(raw)
    if not isinstance(ids, list) or not ids or any(type(token) is not int or token < 0 for token in ids):
        raise ValueError("--tokens-file must contain token IDs from a local agent fixture")
    # This service exposes chat messages, not token-only inference. Use the
    # fixture's authored system text; never guess a tokenizer or decode IDs with
    # another model. The numeric fixture is retained as provenance only.
    system = tokens_file.parent / "system-prompt.txt"
    if not system.is_file():
        raise ValueError("Token-only HTTP is unsupported; --tokens-file needs adjacent system-prompt.txt")
    return system.read_text(), {"tokens_path": str(tokens_file.resolve()), "tokens_sha256": digest(raw),
                               "token_count": len(ids), "system_path": str(system.resolve()),
                               "system_sha256": digest(system.read_bytes())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:11236")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--tokens-file", type=Path)
    args = parser.parse_args()
    url = urlsplit(args.base_url)
    if (url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost", "::1") or
            url.username or url.password or url.path not in ("", "/") or url.query or url.fragment):
        parser.error("--base-url must be an HTTP loopback origin")
    if not 4 <= args.requests <= 1000:
        parser.error("--requests must be 4...1000 (100 is the standard soak)")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x") as file:
            file.write("{}\n")
    except FileExistsError:
        parser.error("--output must be new")
    run_id, started = uuid.uuid4().hex, time.monotonic()
    report = {"schema": "qwen-http-cache-reliability-v1", "complete": False, "passed": False,
              "run_id": run_id, "base_url": args.base_url, "planned_successful_requests": args.requests,
              "concurrency": CONCURRENCY, "request_timeout_seconds": REQUEST_TIMEOUT,
              "script_sha256": digest(Path(__file__).read_bytes()), "checks": [], "oracles": {},
              "requests": [], "cancelled_requests": [], "health_samples": [], "health_errors": [],
              "notes": ["The service is externally controlled; this script makes HTTP requests only.",
                        "Four output oracles precede the requested successful concurrent request count.",
                        "RST cancellation is client-observed; confirm terminal=cancelled independently in server logs using request IDs.",
                        "All valid requests use temperature=0, mtp_depth=0 and max_tokens=16.",
                        "Cache limits come from health, not assumed command-line defaults.",
                        "No HTTP wall-time measurement is a GPU prefill/decode benchmark or bandwidth claim."]}
    guard = threading.Lock()
    stop_health = threading.Event()
    monitor = None
    host, port = url.hostname, url.port or 80

    def save():
        with guard:
            snapshot = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(snapshot + "\n")
        temporary.replace(output)

    def check(name, condition, **evidence):
        report["checks"].append({"name": name, "passed": bool(condition), **evidence})
        save()
        print(json.dumps({"check": name, "passed": bool(condition)}, ensure_ascii=False), flush=True)
        if not condition:
            raise AssertionError(name)

    def raw_request(method, path, body=None, timeout=REQUEST_TIMEOUT):
        connection = http.client.HTTPConnection(host, port, timeout=timeout)
        payload = None if body is None else json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()
        began = time.monotonic()
        try:
            connection.request(method, path, payload, {"Content-Type": "application/json"} if body is not None else {})
            response = connection.getresponse()
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise ValueError("Response exceeds client bound")
            return response.status, {k.lower(): v for k, v in response.getheaders()}, raw, time.monotonic() - began
        finally:
            connection.close()

    def health(label):
        status, _, raw, wall = raw_request("GET", "/health", timeout=10)
        if status != 200:
            raise ValueError(f"Health HTTP {status}: {raw[:500]!r}")
        value = health_contract(json.loads(raw))
        if report.get("server_pid") is not None and value["pid"] != report["server_pid"]:
            raise ValueError("Server process changed during soak")
        with guard:
            report["health_samples"].append({"label": label, "elapsed_seconds": time.monotonic() - started,
                                              "request_seconds": wall, "health": value})
        return value

    def idle(label):
        deadline = time.monotonic() + IDLE_TIMEOUT
        while time.monotonic() < deadline:
            value = health(label)
            if fully_idle(value):
                return value
            time.sleep(0.1)
        raise TimeoutError("Jobs, request/workspace leases or pending SSD I/O did not drain")

    def observe():
        while not stop_health.wait(0.25):
            try:
                health("continuous")
            except Exception as error:
                with guard:
                    report["health_errors"].append({"elapsed_seconds": time.monotonic() - started,
                                                     "error": f"{type(error).__name__}: {error}"})
                return

    def chat(number, profile, body, expected=None):
        stream = bool(body.get("stream"))
        row = {"number": number, "profile": profile, "stream": stream, "passed": False}
        try:
            status, headers, raw, wall = raw_request("POST", "/v1/chat/completions", body)
            row.update(status=status, wall_seconds=wall, response_body_base64=base64.b64encode(raw).decode())
            result = parse_completion(status, headers, raw, stream)
            row["result"] = result
            if expected is not None:
                for field in ("content", "finish_reason"):
                    if result[field] != expected[field]:
                        raise ValueError(f"Concurrent cached {field} differs from serial oracle")
                for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    if result["usage"][field] != expected["usage"][field]:
                        raise ValueError(f"Concurrent usage {field} differs from serial oracle")
                if result["cached_tokens"] < 416:
                    raise ValueError("Repeated system prefix did not reuse a complete cache boundary")
            row["passed"] = True
            return row
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
            return row

    def abort_stream(number, profile, body):
        payload = json.dumps({**body, "stream": True}, ensure_ascii=False, separators=(",", ":")).encode()
        row = {"number": number, "profile": profile, "passed": False,
               "client_abort_requested": False, "server_cancellation_confirmed": False}
        sock = socket.create_connection((host, port), timeout=REQUEST_TIMEOUT)
        response = None
        began = time.monotonic()
        try:
            hostname = f"[{host}]" if ":" in host else host
            headers = (f"POST /v1/chat/completions HTTP/1.1\r\nHost: {hostname}:{port}\r\n"
                       f"Content-Type: application/json\r\nContent-Length: {len(payload)}\r\nConnection: close\r\n\r\n").encode()
            sock.sendall(headers + payload)
            response = http.client.HTTPResponse(sock)
            response.begin()
            row["status"] = response.status
            if response.status != 200:
                raise ValueError(f"Cancellation request HTTP {response.status}")
            frame, consumed = bytearray(), 0
            while consumed <= MAX_RESPONSE_BYTES and time.monotonic() - began < REQUEST_TIMEOUT:
                line = response.readline(65_537)
                if not line or len(line) > 65_536:
                    raise ValueError("Incomplete or oversized SSE before cancellation")
                frame.extend(line); consumed += len(line)
                if line.strip():
                    continue
                for data in events(bytes(frame)):
                    if data == "[DONE]":
                        raise ValueError("Response finished before cancellation request")
                    chunk = json.loads(data)
                    if "error" in chunk:
                        raise ValueError(f"SSE rejected cancellation request: {chunk['error']}")
                    row["request_id"] = chunk["id"]
                    if any(choice.get("finish_reason") is not None for choice in chunk["choices"]):
                        raise ValueError("Model finish observed before cancellation")
                    if any(choice.get("delta", {}).get("role") == "assistant" for choice in chunk["choices"]):
                        row["prefix_frame_base64"] = base64.b64encode(bytes(frame)).decode()
                        row["abort_stage"] = "after_assistant_role_before_finish"
                        sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                        row["client_abort_requested"] = True
                        response.close(); response = None
                        sock.close()
                        row["passed"] = True
                        return row
                frame.clear()
            raise TimeoutError("No role frame before cancellation deadline")
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
            return row
        finally:
            row["wall_seconds"] = time.monotonic() - began
            if response is not None:
                response.close()
            sock.close()

    try:
        initial = idle("initial_idle")
        report["server_pid"], report["model"] = initial["pid"], initial["model"]
        report["initial_health"] = initial
        long_text, fixture = fixture_text(args.tokens_file)
        report["fixture"] = fixture
        nonce = f"Cache reliability run {run_id}.\n"
        short_text = "Use this reference context consistently.\n" + "".join(
            f"Reference record {index}: cache reuse preserves the exact ordered input history.\n" for index in range(96))
        systems = {"short": nonce + short_text, "long": nonce + long_text}
        profiles = {}
        for name, system in systems.items():
            for variant in (0, 1):
                profiles[f"{name}_{variant}"] = {"model": initial["model"], "temperature": 0,
                    "mtp_depth": 0, "max_tokens": 16, "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": f"请只输出这一行，不加解释：CACHE_RELIABILITY_{variant}_OK。"}]}
        names = list(profiles)
        report["profiles"] = profiles
        monitor = threading.Thread(target=observe, name="cache-health", daemon=True)
        monitor.start()
        for index, (name, body) in enumerate(profiles.items()):
            row = chat(f"oracle-{index}", name, body)
            report["oracles"][name] = row
            check(name + "_serial_oracle", row["passed"], error=row.get("error"))
            count = row["result"]["usage"]["prompt_tokens"]
            check(name + "_meaningful_prefix", 833 <= count < 4096 if name.startswith("short") else 10_000 <= count < 16_384,
                  prompt_tokens=count)
        seeded = idle("oracles_complete")
        check("ssd_publication_after_oracles", seeded["prefix_disk_cache"]["published"] > initial["prefix_disk_cache"]["published"])
        with ThreadPoolExecutor(max_workers=CONCURRENCY + 1, thread_name_prefix="cache-request") as workers:
            for start in range(0, args.requests, CONCURRENCY):
                futures = {}
                for number in range(start, min(start + CONCURRENCY, args.requests)):
                    name = names[number % len(names)]
                    body = {**profiles[name], "stream": number % 2 == 1}
                    future = workers.submit(chat, number, name, body, report["oracles"][name]["result"])
                    futures[future] = "request"
                if start % 12 == 0:
                    # Keep five simultaneous prompt reservations below the service's
                    # 32768-token admission ceiling. Long private-state cancel
                    # correctness is covered by the full-model CLI probe.
                    name = names[(start // 12) % 2]
                    futures[workers.submit(abort_stream, start // 12, name, profiles[name])] = "cancel"
                for future in as_completed(futures):
                    row = future.result()
                    report["cancelled_requests" if futures[future] == "cancel" else "requests"].append(row)
                    check(f"{futures[future]}_{row['number']}", row["passed"], error=row.get("error"))
                check(f"health_through_batch_{start}", not report["health_errors"])
                if start % 12 == 0:
                    report.setdefault("idle_checkpoints", []).append(idle(f"batch_{start}_drained"))
                    save()
        final = idle("final_idle")
        report["final_health"] = final
        stop_health.set()
        if monitor is not None:
            monitor.join(timeout=12)
        check("continuous_health_ready", not report["health_errors"] and monitor is not None and not monitor.is_alive())
        check("all_planned_requests_completed", len(report["requests"]) == args.requests and all(row["passed"] for row in report["requests"]))
        check("mixed_stream_nonstream_coverage", {row["stream"] for row in report["requests"]} == {False, True})
        check("mixed_short_long_coverage", {row["profile"].split("_")[0] for row in report["requests"]} == {"short", "long"})
        check("client_abort_requests_and_recovery", bool(report["cancelled_requests"]) and
              all(row["client_abort_requested"] for row in report["cancelled_requests"]) and fully_idle(final))
        check("all_jobs_and_request_workspace_leases_released", fully_idle(final))
        check("ram_cache_hits_observed", final["prefix_cache"]["hits"] > seeded["prefix_cache"]["hits"])
        check("no_new_ssd_corruption_or_write_failure", all(final["prefix_disk_cache"][field] == initial["prefix_disk_cache"][field]
                                                         for field in ("corruptions", "writeFailures")))
        report["health_sample_count"] = len(report["health_samples"])
        report["elapsed_seconds"] = time.monotonic() - started
        report["complete"] = True
        report["passed"] = all(item["passed"] for item in report["checks"])
        save()
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["elapsed_seconds"] = time.monotonic() - started
        save()
        raise
    finally:
        stop_health.set()
        if monitor is not None:
            monitor.join(timeout=12)
        save()


if __name__ == "__main__":
    main()
