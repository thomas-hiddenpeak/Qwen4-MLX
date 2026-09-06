#!/usr/bin/env python3
"""One controller-owned model server; bounded HTTP lifecycle edge checks.

Run only after the outer experiment controller pauses the reference service.
This follows test_http_server_live.py's request/health/response conventions;
that script's helpers are local to main and cannot be imported independently.
Only this script's Popen child and sockets are closed or signalled, including
on SIGTERM/KeyboardInterrupt. This is a lifecycle gate, not a speed benchmark.
"""
import argparse
import hashlib
import http.client
import json
from pathlib import Path
import signal
import socket
import struct
import subprocess
import time


BODY_LIMIT = 2 * 1024 * 1024
ZERO_COUNTERS = ("active", "active_jobs", "pending_requests", "queued_prefills",
                 "ready_decodes", "resident_sequences", "reserved_tokens")


class HTTPHarness:
    def __init__(self, port, child):
        self.port, self.child = port, child
        self.sockets = set()

    def alive(self):
        if self.child.poll() is not None:
            raise RuntimeError(f"Owned server exited {self.child.returncode}")

    def connect(self, timeout=5):
        self.alive()
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=timeout)
        self.sockets.add(sock)
        return sock

    def close(self, sock, reset=False):
        if sock not in self.sockets:
            return
        self.sockets.remove(sock)
        try:
            if reset:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        finally:
            sock.close()

    def close_all(self):
        for sock in list(self.sockets):
            try:
                self.close(sock)
            except OSError:
                pass

    def request(self, method, path, body=None, timeout=60):
        self.alive()
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        payload = None if body is None else json.dumps(body, ensure_ascii=False).encode()
        started = time.monotonic()
        try:
            connection.request(method, path, body=payload,
                               headers={"Content-Type": "application/json"} if body is not None else {})
            response = connection.getresponse()
            raw = response.read(BODY_LIMIT + 1)
            if len(raw) > BODY_LIMIT:
                raise ValueError("Response exceeds harness byte bound")
            return response.status, dict(response.getheaders()), raw, time.monotonic() - started
        finally:
            connection.close()

    def health(self):
        status, _, raw, _ = self.request("GET", "/health", timeout=3)
        value = json.loads(raw)
        if value.get("pid") != self.child.pid:
            raise RuntimeError("Health responder is not this harness's child")
        return status, value

    def until(self, predicate, timeout=30):
        deadline, last = time.monotonic() + timeout, None
        while time.monotonic() < deadline:
            self.alive()
            try:
                last = self.health()
                if predicate(last):
                    return last[1]
            except (OSError, ValueError, http.client.HTTPException):
                pass
            time.sleep(.1)
        raise TimeoutError(f"Health condition not reached; last={last}")

    def idle(self):
        return self.until(lambda item: item[0] == 200 and item[1].get("idle") is True
                          and all(item[1].get(key) == 0 for key in ZERO_COUNTERS))

    def post_socket(self, body):
        payload = json.dumps(body, ensure_ascii=False).encode()
        sock = self.connect()
        sock.sendall(("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n"
                      "Content-Type: application/json\r\n"
                      f"Content-Length: {len(payload)}\r\n\r\n").encode() + payload)
        return sock


def read_response(sock, timeout=20):
    sock.settimeout(timeout)
    response = http.client.HTTPResponse(sock)
    try:
        response.begin()
        raw = response.read(BODY_LIMIT + 1)
        if len(raw) > BODY_LIMIT:
            raise ValueError("Raw response exceeds harness byte bound")
        return response.status, dict(response.getheaders()), raw
    finally:
        response.close()


def decode_completion(response, stream):
    status, headers, raw, wall = response
    if status != 200:
        raise ValueError(f"HTTP {status}: {raw[:1000]!r}")
    if not stream:
        obj = json.loads(raw)
        assert obj["object"] == "chat.completion"
        choice = obj["choices"][0]
        assert choice["message"]["role"] == "assistant"
        return {"text": choice["message"]["content"], "finish": choice["finish_reason"],
                "usage": obj["usage"], "id": obj["id"], "wall": wall}
    assert "text/event-stream" in {key.lower(): value for key, value in headers.items()}["content-type"]
    parts = [part for part in raw.split(b"\n\n") if part]
    assert parts and parts[-1] == b"data: [DONE]" and parts.count(b"data: [DONE]") == 1
    frames = []
    for part in parts[:-1]:
        assert part.startswith(b"data: ")
        frames.append(json.loads(part[6:].decode("utf-8", errors="strict")))
    assert frames and frames[0]["choices"][0]["delta"].get("role") == "assistant"
    identity = (frames[0]["id"], frames[0]["created"], frames[0]["model"])
    text, finishes, usages = [], [], []
    for frame in frames:
        assert "error" not in frame and (frame["id"], frame["created"], frame["model"]) == identity
        assert len(frame["choices"]) == 1
        for choice in frame["choices"]:
            assert choice["index"] == 0
            text.append(choice["delta"].get("content", ""))
            if choice["finish_reason"] is not None:
                finishes.append(choice["finish_reason"])
        if frame.get("usage") is not None:
            usages.append(frame["usage"])
    assert len(finishes) == len(usages) == 1
    assert frames[-1]["choices"][0]["finish_reason"] is not None and "usage" in frames[-1]
    return {"text": "".join(text), "finish": finishes[0], "usage": usages[0],
            "id": identity[0], "frames": len(frames), "wall": wall}


def content_prefix(sock, count=2, timeout=180):
    """Read complete frames, not arbitrary recv chunks; never repair UTF-8."""
    sock.settimeout(timeout)
    response = http.client.HTTPResponse(sock)
    frames, content = [], []
    size, deadline = 0, time.monotonic() + timeout
    try:
        response.begin()
        if response.status != 200:
            raise ValueError(f"Cancellation precondition got HTTP {response.status}")
        assert "text/event-stream" in response.getheader("Content-Type", "")
        pending = bytearray()
        while len(content) < count:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Timed out waiting for decode content")
            sock.settimeout(remaining)
            line = response.readline(8193)
            if not line or len(line) > 8192:
                raise ValueError("Premature EOF or oversized SSE line")
            pending.extend(line)
            size += len(line)
            if size > 65536 or len(frames) > 256:
                raise ValueError("SSE precondition exceeds harness bounds")
            if line not in (b"\n", b"\r\n"):
                continue
            wire = bytes(pending).rstrip(b"\r\n")
            pending.clear()
            if wire == b"data: [DONE]":
                raise ValueError("Natural completion arrived before decode-cancel precondition")
            assert wire.startswith(b"data: ")
            frame = json.loads(wire[6:].decode("utf-8", errors="strict"))
            assert "error" not in frame and frame["object"] == "chat.completion.chunk"
            frames.append(frame)
            identity = (frames[0]["id"], frames[0]["created"], frames[0]["model"])
            assert (frame["id"], frame["created"], frame["model"]) == identity
            for choice in frame["choices"]:
                assert choice["index"] == 0 and choice["finish_reason"] is None
                value = choice["delta"].get("content")
                if value:
                    content.append(value)
        assert frames[0]["choices"][0]["delta"].get("role") == "assistant"
        return {"id": frames[0]["id"], "content_frames": len(content),
                "text_prefix": "".join(content), "frames": frames, "received_body_bytes": size}
    finally:
        # HTTPResponse owns a makefile reference. Configure RST before dropping
        # that reference; the caller then closes its original socket immediately.
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        finally:
            response.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runner", "model-dir", "fixture", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--port", type=int, default=11236)
    args = parser.parse_args()
    out = args.output.resolve()
    log = out.with_suffix(".server.log")
    if out.exists() or log.exists():
        parser.error("--output and its .server.log must be new")
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be 1024...65535")
    out.parent.mkdir(parents=True, exist_ok=True)
    model = args.model_dir.resolve().name
    child, harness = None, None
    report = {"schema": "qwen-live-http-edge-gate-v1", "complete": False, "passed": False, "checks": [],
              "notes": ["Observed loopback lifecycle only; not a performance or production qualification.",
                        "At least two nonempty content frames precede RST; the first token can come from prefill.",
                        "Decode-stage network cancellation is not evidence of interruption inside MTP verification.",
                        "No paused-reader or live output-buffer overflow claim is made."]}

    def save():
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")

    def check(name, condition, **evidence):
        report["checks"].append({"id": name, "passed": bool(condition), **evidence})
        save()
        print(name, bool(condition), flush=True)
        if not condition:
            raise AssertionError(name)

    def chat(messages, stream=False, depth=0, budget=128):
        return {"model": model, "messages": messages, "max_tokens": budget,
                "stream": stream, "mtp_depth": depth}

    def interrupted(number, _frame):
        raise InterruptedError(f"Harness received signal {number}")

    previous_term = signal.signal(signal.SIGTERM, interrupted)
    previous_int = signal.getsignal(signal.SIGINT)
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", args.port))
        messages = [{"role": "system", "content": (args.fixture / "system-prompt.txt").read_text().strip()},
                    {"role": "user", "content": (args.fixture / "user-prompt.txt").read_text().strip()}]
        command = [str(args.runner.resolve()), "serve-gpu", "--model-dir", str(args.model_dir.resolve()),
                   "--port", str(args.port), "--max-connections", "4", "--output-buffer-bytes", "8192"]
        report.update(command=command, runner_sha256=hashlib.sha256(args.runner.read_bytes()).hexdigest(),
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        with log.open("x") as output:
            child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        report["server_pid"] = child.pid
        harness = HTTPHarness(args.port, child)
        save()
        ready = harness.until(lambda item: item[0] == 200 and item[1].get("ready") is True, timeout=180)
        check("ready_owned_child", ready["pid"] == child.pid, health=ready)
        harness.idle()

        # No health requests occur during saturation. Four incomplete headers
        # occupy four slots; completing all four afterwards proves their admission.
        held = []
        try:
            for _ in range(4):
                sock = harness.connect()
                held.append(sock)
                sock.sendall(b"GET /health HTTP/1.1\r\nHost: localhost\r\nX-Hold: pending")
            time.sleep(.15)
            fifth = harness.connect(timeout=3)
            rejection = None
            try:
                fifth.sendall(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
                received = fifth.recv(1)
                if received == b"":
                    rejection = "eof"
            except (ConnectionResetError, BrokenPipeError):
                rejection = "reset"
            finally:
                harness.close(fifth)
            admitted = []
            for sock in held:
                sock.sendall(b"\r\n\r\n")
                status, _, raw = read_response(sock, timeout=3)
                admitted.append({"status": status, "health": json.loads(raw)})
                harness.close(sock)
            check("max_connections_four_admitted_fifth_closed", rejection in ("eof", "reset")
                  and all(item["status"] == 200 and item["health"].get("pid") == child.pid for item in admitted)
                  and admitted[0]["health"].get("connections") == 4,
                  fifth_observed=rejection, accepted_responses=admitted)
        finally:
            for sock in held:
                harness.close(sock)
        check("connection_slots_recover", True, health=harness.idle())

        # Header and body receive deadlines run together, without any generation.
        pending = []
        try:
            for label, wire in [("header", b"GET /health HTTP/1.1\r\nHost: localhost\r\n"),
                                ("body", b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{")]:
                started = time.monotonic()
                sock = harness.connect()
                pending.append((label, sock, started))
                sock.sendall(wire)
            for label, sock, started in pending:
                status, _, raw = read_response(sock, timeout=22)
                elapsed = time.monotonic() - started
                error = json.loads(raw)
                check("receive_deadline_" + label, status == 408 and 14.5 <= elapsed <= 22
                      and error.get("error", {}).get("code") == "request_timeout",
                      status=status, elapsed_seconds=elapsed, response=error)
                harness.close(sock)
        finally:
            for _, sock, _ in pending:
                harness.close(sock)
        check("receive_deadlines_leave_no_jobs", True, health=harness.idle())

        truncated = harness.connect()
        try:
            truncated.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{")
            truncated.shutdown(socket.SHUT_WR)
            status, _, raw = read_response(truncated, timeout=5)
            error = json.loads(raw)
            check("truncated_content_length_eof_400", status == 400
                  and error.get("error", {}).get("message") == "Incomplete HTTP request",
                  status=status, response=error)
        finally:
            harness.close(truncated)
        check("truncated_body_not_admitted", True, health=harness.idle())

        # Require length rather than natural EOS, and compare AR with MTP's
        # actual emitted budget. The text is also reused after decode cancellation.
        numbers = [{"role": "user", "content": "按顺序写出从1到100的整数，使用英文逗号分隔，不要解释，也不要省略。"}]
        report["budget_prompt"] = numbers
        baseline = None
        for budget in (1, 2, 4):
            ar = decode_completion(harness.request("POST", "/v1/chat/completions", chat(numbers, budget=budget)), False)
            mtp = decode_completion(harness.request("POST", "/v1/chat/completions", chat(numbers, True, 2, budget)), True)
            valid = ar["finish"] == mtp["finish"] == "length" and ar["text"] == mtp["text"] and ar["usage"] == mtp["usage"]
            usage = ar["usage"]
            valid = valid and usage["completion_tokens"] == budget and usage["prompt_tokens"] > 0
            valid = valid and usage["total_tokens"] == usage["prompt_tokens"] + budget
            check("ar_mtp_length_budget_" + str(budget), valid, ar=ar, mtp=mtp)
            baseline = ar
        bad = chat(numbers, depth=2, budget=257)
        status, _, raw, _ = harness.request("POST", "/v1/chat/completions", bad)
        check("mtp_unvalidated_budget_rejected", status == 400, status=status, response=json.loads(raw))
        harness.idle()

        cancellation_request = chat(messages, True, 2, 256)
        report["cancellation_request"] = cancellation_request
        save()
        sock = harness.post_socket(cancellation_request)
        try:
            prefix = content_prefix(sock, count=2)
        finally:
            started = time.monotonic()
            harness.close(sock, reset=True)
        # Retain the observable prefix even if cleanup/log validation fails.
        report["cancellation_prefix"] = prefix
        save()
        idle = harness.idle()
        elapsed = time.monotonic() - started
        matching = [line for line in log.read_text().splitlines() if "HTTP request id=" + prefix["id"] + " " in line]
        check("mtp_decode_rst_cancelled_and_released", prefix["content_frames"] >= 2
              and len(matching) == 1 and "terminal=cancelled stage=decode " in matching[0],
              elapsed_seconds=elapsed, prefix=prefix, matching_server_lines=matching, health=idle)
        fresh = decode_completion(harness.request("POST", "/v1/chat/completions", chat(numbers, budget=4)), False)
        check("fresh_after_mtp_decode_rst_exact", fresh["text"] == baseline["text"]
              and fresh["usage"] == baseline["usage"] and fresh["finish"] == baseline["finish"], result=fresh)
        check("final_idle_all_resources_released", True, health=harness.idle())
        report["complete"] = True
        report["passed"] = all(item["passed"] for item in report["checks"])
    except BaseException as error:
        report["passed"] = False
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        # Repeated controller interrupts must not bypass cleanup of our child.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            if harness is not None:
                harness.close_all()
            if child is not None and child.poll() is None:
                child.send_signal(signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    report["forced_shutdown"] = True
                    child.kill()
                    child.wait(timeout=10)
            if child is not None:
                report["server_exit_code"] = child.returncode
                report["graceful_shutdown"] = not report.get("forced_shutdown", False) and child.returncode == 0
                report["passed"] = report["passed"] and report["graceful_shutdown"]
            save()
        finally:
            signal.signal(signal.SIGTERM, previous_term)
            signal.signal(signal.SIGINT, previous_int)
    print(json.dumps({key: report.get(key) for key in ("passed", "complete", "error", "graceful_shutdown")}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
