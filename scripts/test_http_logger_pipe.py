#!/usr/bin/env python3
"""Owned unread-stderr-pipe liveness gate. Run only under the GPU controller.

stdout goes to a regular file. The parent holds stderr's read end without ever
reading until the owned server has exited. No shared fd flags or other processes
are changed. Real health requests fill the log sink; no fake compute/delay flag.
"""
import argparse
import array
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import socket
import subprocess
import sys
import termios
import time

sys.dont_write_bytecode = True


def find_repo_root(script_path):
    for ancestor in Path(script_path).resolve().parents:
        if (ancestor / "Package.swift").is_file() and (ancestor / "scripts/test_http_server_edges.py").is_file():
            return ancestor
    raise RuntimeError("Cannot locate repository root containing Package.swift and HTTP edge helpers")


REPO = find_repo_root(__file__)
sys.path.insert(0, str(REPO / "scripts"))
from test_http_server_edges import HTTPHarness, decode_completion

PROMPT = [{"role": "user", "content": "按顺序写出从1到100的整数，使用英文逗号分隔，不要解释，也不要省略。"}]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def logging_state(health):
    logs = health.get("logging")
    require(isinstance(logs, dict), "Health lacks bounded logger counters")
    counts = ("max_bytes", "max_events", "max_event_bytes", "buffered_bytes", "buffered_events",
              "queued_events", "in_flight_bytes", "enqueued_events", "written_events",
              "dropped_events", "dropped_bytes", "write_failures")
    require(all(type(logs.get(k)) is int and logs[k] >= 0 for k in counts), "Invalid logger counter")
    require(logs.get("accepting") is True and logs.get("writer_exited") is False, "Writer unexpectedly stopped")
    require(logs["write_failures"] == 0 and logs.get("last_write_errno") is None, "Unread pipe must stay open, without EPIPE")
    require(0 < logs["max_event_bytes"] <= logs["max_bytes"] and logs["max_events"] > 0, "Invalid logger limits")
    require(logs["in_flight_bytes"] <= logs["max_event_bytes"] and
            logs["in_flight_bytes"] <= logs["buffered_bytes"] <= logs["max_bytes"] and
            logs["queued_events"] <= logs["buffered_events"] <= logs["max_events"], "Logger quota exceeded")
    return logs


def backlog_at_limit(logs):
    # Event size can leave < one max-event of byte headroom without fitting the
    # next real record. Require actual drops plus an in-flight write, not merely
    # a pause in reads while the kernel absorbs all output.
    return (logs["dropped_events"] > 0 and logs["in_flight_bytes"] > 0 and
            (logs["buffered_events"] == logs["max_events"] or
             logs["buffered_bytes"] > logs["max_bytes"] - logs["max_event_bytes"]))


def unread_bytes(read_fd):
    value = array.array("i", [0])
    fcntl.ioctl(read_fd, termios.FIONREAD, value, True)
    require(value[0] >= 0, "Invalid owned-pipe byte count")
    return value[0]


def drain_after_exit(read_fd, bound=1_048_576):
    """Only called after child exit. No changes to stderr's write fd flags."""
    result = bytearray()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if not select.select([read_fd], [], [], max(0, deadline - time.monotonic()))[0]:
            break
        chunk = os.read(read_fd, min(65_536, bound + 1 - len(result)))
        if not chunk:
            return bytes(result)
        result.extend(chunk)
        require(len(result) <= bound, "Owned stderr pipe exceeded capture bound")
    raise TimeoutError("Owned pipe did not reach EOF after server exit")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("runner", "model-dir", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--port", type=int, default=11236)
    args = parser.parse_args()
    require(1024 <= args.port <= 65535, "Port must be 1024...65535")
    out = args.output.resolve()
    stdout_path, stderr_path = out.with_suffix(".stdout.log"), out.with_suffix(".stderr.log")
    require(not any(p.exists() for p in (out, stdout_path, stderr_path)), "Outputs must be new")
    out.parent.mkdir(parents=True, exist_ok=True)
    model = args.model_dir.resolve().name
    report = {"schema": "qwen-http-logger-pipe-gate-v1", "complete": False, "passed": False,
              "checks": [], "stderr_read_before_exit": False, "lifecycle_attribution_complete": False,
              "notes": ["One owned real server; stdout is a regular file, stderr is an open unread owned pipe.",
                        "Health-generated logs must reach application quota, retain an in-flight write and report drops.",
                        "Two fixed short AR/MTP outputs use HTTP evidence. Dropped lifecycle logs cannot prove attribution.",
                        "SIGTERM must finish while the pipe remains unread; no logger flush guarantee.",
                        "Closed-pipe EPIPE is a separate Core CPU test under the explicit service SIGPIPE policy; no SSE slow-reader/deadline coverage is claimed."]}
    parent_sigpipe = signal.getsignal(signal.SIGPIPE)
    report["parent_sigpipe_policy_before"] = ("ignored" if parent_sigpipe == signal.SIG_IGN else
                                               "default" if parent_sigpipe == signal.SIG_DFL else "custom")
    child = harness = None
    read_fd = write_fd = None
    term_sent_at = None

    def save():
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    def check(name, condition, **evidence):
        report["checks"].append({"id": name, "passed": bool(condition), **evidence})
        save()
        require(condition, name)

    def interrupted(number, _frame):
        raise InterruptedError(f"Harness interrupted by signal {number}")

    def deadline(_number, _frame):
        raise TimeoutError("Fixed 300-second work budget exhausted")

    previous_term = signal.signal(signal.SIGTERM, interrupted)
    previous_int = signal.signal(signal.SIGINT, interrupted)
    previous_alarm = signal.signal(signal.SIGALRM, deadline)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 300)
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", args.port))
        command = [str(args.runner.resolve()), "serve-gpu", "--model-dir", str(args.model_dir.resolve()),
                   "--port", str(args.port), "--max-connections", "4", "--output-buffer-bytes", "8192"]
        report.update(command=command, runner_sha256=hashlib.sha256(args.runner.read_bytes()).hexdigest(),
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      edge_helper_sha256=hashlib.sha256((REPO / "scripts/test_http_server_edges.py").read_bytes()).hexdigest())
        read_fd, write_fd = os.pipe()
        with stdout_path.open("xb") as stdout:
            # Inherit the outer controller's owned group. Never create a daemon
            # or signal the shared parent group from this inner harness.
            child = subprocess.Popen(command, cwd=REPO, stdout=stdout, stderr=write_fd)
        os.close(write_fd); write_fd = None
        report["server_pid"] = child.pid
        harness = HTTPHarness(args.port, child)
        report["ready"] = harness.until(lambda item: item[0] == 200 and item[1].get("ready") is True, timeout=180)
        logging_state(report["ready"])

        filled = None
        fill_deadline = time.monotonic() + 30
        for number in range(4096):
            require(time.monotonic() < fill_deadline, "Real health logs did not fill bounded logger within 30 seconds")
            status, state = harness.health()
            require(status == 200, "Health failed during owned unread-pipe fill")
            logs = logging_state(state)
            if backlog_at_limit(logs):
                filled = logs
                report["fill_health_requests"] = number + 1
                break
        require(filled is not None, "Real health logs did not reach application log limit")
        # Observe the same blocked writer across additional successful requests.
        # Reading FIONREAD does not consume any bytes or change fd flags.
        stalled = []
        for _ in range(3):
            time.sleep(.1)
            status, state = harness.health()
            logs = logging_state(state)
            stalled.append({"logging": logs, "pipe_unread_bytes": unread_bytes(read_fd)})
            require(status == 200 and backlog_at_limit(logs), "Logger backlog did not persist")
        check("health_responds_with_real_stalled_pipe_and_bounded_drops",
              all(s["logging"]["written_events"] == filled["written_events"] and
                  s["logging"]["in_flight_bytes"] == filled["in_flight_bytes"] and s["pipe_unread_bytes"] > 0 for s in stalled)
              and stalled[-1]["logging"]["dropped_events"] > filled["dropped_events"],
              first=filled, samples=stalled)

        results = []
        for depth, stream in ((0, False), (2, True)):
            body = {"model": model, "messages": PROMPT, "max_tokens": 4, "mtp_depth": depth, "stream": stream}
            result = decode_completion(harness.request("POST", "/v1/chat/completions", body, timeout=60), stream)
            results.append({"mtp_depth": depth, "stream": stream, **result})
        idle = harness.idle()
        logs = logging_state(idle)
        check("real_ar_mtp_and_idle_while_logger_write_is_blocked",
              all(r["text"] == "1,2," and r["finish"] == "length" and r["usage"]["completion_tokens"] == 4 for r in results)
              and results[0]["usage"] == results[1]["usage"]
              and logs["written_events"] == filled["written_events"] and backlog_at_limit(logs),
              requests=results, idle=idle)

        report["before_term"] = {"logging": logs, "pipe_unread_bytes": unread_bytes(read_fd)}
        save()
        start = time.monotonic()
        term_sent_at = start
        child.send_signal(signal.SIGTERM)
        child.wait(timeout=30)  # Pipe stays open and unread through this wait.
        report["term_to_exit_seconds"] = time.monotonic() - start
        report["parent_sigpipe_policy_unchanged"] = signal.getsignal(signal.SIGPIPE) == parent_sigpipe
        check("sigterm_exits_without_reading_or_joining_blocked_writer",
              child.returncode == 0 and report["parent_sigpipe_policy_unchanged"],
              exit_code=child.returncode, elapsed_seconds=report["term_to_exit_seconds"],
              pipe_unread_bytes_after_exit=unread_bytes(read_fd))
        report["complete"] = True
        report["passed"] = all(c["passed"] for c in report["checks"])
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["passed"] = False
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            if harness is not None:
                harness.close_all()
            if child is not None and child.poll() is None:
                if term_sent_at is None:
                    term_sent_at = time.monotonic()
                    child.send_signal(signal.SIGTERM)
                try:
                    # The initial explicit TERM wait and finally share one
                    # 30-second allowance, then at most 10 seconds for KILL.
                    child.wait(timeout=max(0, 30 - (time.monotonic() - term_sent_at)))
                except subprocess.TimeoutExpired:
                    report["forced_shutdown"] = True
                    child.kill(); child.wait(timeout=10)
            if write_fd is not None:
                os.close(write_fd); write_fd = None
            if child is not None:
                report["server_exit_code"] = child.poll()
                report["graceful_shutdown"] = child.returncode == 0 and not report.get("forced_shutdown", False)
                report["passed"] = report["passed"] and report["graceful_shutdown"]
                if child.poll() is not None and read_fd is not None:
                    captured = drain_after_exit(read_fd)
                    stderr_path.write_bytes(captured)
                    report["stderr_captured_after_exit_bytes"] = len(captured)
        except BaseException as error:
            report["cleanup_error"] = f"{type(error).__name__}: {error}"
            report["passed"] = False
        finally:
            for fd in (read_fd, write_fd):
                if fd is not None:
                    os.close(fd)
            try:
                save()
            finally:
                signal.signal(signal.SIGTERM, previous_term)
                signal.signal(signal.SIGINT, previous_int)
                signal.signal(signal.SIGALRM, previous_alarm)
                signal.setitimer(signal.ITIMER_REAL, *previous_timer)
    print(json.dumps({k: report.get(k) for k in ("complete", "passed", "error", "graceful_shutdown")}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
