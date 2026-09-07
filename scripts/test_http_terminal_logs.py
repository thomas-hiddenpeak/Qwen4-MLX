#!/usr/bin/env python3
"""Candidate-only HTTP lifecycle gate; launch live mode only under the GPU controller.

Starts one owned real model server, reuses unchanged edge helpers, then stops
that child in finally. --audit-log only reads an existing log and starts nothing.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode = True


def find_repo_root(script_path):
    for ancestor in Path(script_path).resolve().parents:
        if (ancestor / "Package.swift").is_file() and (ancestor / "scripts/test_http_server_edges.py").is_file():
            return ancestor
    raise RuntimeError("Cannot locate repository root containing Package.swift and scripts/test_http_server_edges.py")


REPO = find_repo_root(__file__)
sys.path.insert(0, str(REPO / "scripts"))
from test_http_server_edges import HTTPHarness, content_prefix, decode_completion

SCHEMA = "qwen-http-lifecycle-v1"
UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
REQUEST = "chatcmpl-" + UUID
NUMBER = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
CLOSE_REASONS = set("connection_limit server_stopping client_initialization_failed transport_failed transport_cancelled receive_failed response_conflict simple_sent simple_send_failed rejected_after_response error_encoding_failed header_send_failed send_failed terminal_sent send_deadline connection_deadline shutdown output_encoding_failed".split())
COMMON = set("schema pid event connection_id request_id reason uptime_seconds".split())
BUFFER = set("stream mtp_depth output_outcome buffered_bytes buffered_events queued_events in_flight_bytes has_in_flight producer_finished transport_closed cancellation_requested output_drained".split())
FIELDS = {
    "model_terminal": set("model_kind stage scheduler_elapsed_seconds model_finish_reason prompt_tokens completion_tokens prefill_seconds decode_seconds".split()),
    "output_terminal": set("text_bytes text_limit_bytes error_code".split()),
    "connection_close": set("connection_age_seconds send_elapsed_seconds lease_id send_kind rejection_status rejection_code output_outcome_before_close".split()),
    "closed_send_released": {"lease_id"},
}
REASONS = {
    "model_terminal": {None, "text_limit", "response_limit", "encoding_failed"},
    "output_terminal": set("completed slow_consumer cancelled generation_failed text_limit response_limit encoding_failed".split()),
    "connection_close": CLOSE_REASONS,
    "closed_send_released": {"send_processed", "send_failed"},
}
ENUMS = {
    "model_kind": {"completed", "failed", "cancelled"}, "stage": {"prefill", "decode"},
    "model_finish_reason": {"eos", "length"}, "send_kind": {"header", "simple", "none", "body"},
    "output_outcome": {None, "completed", "failed", "cancelled", "slowConsumer", "disconnected"},
    "output_outcome_before_close": {None, "completed", "failed", "cancelled", "slowConsumer", "disconnected"},
    "error_code": {None, "slow_consumer", "cancelled", "generation_failed", "output_limit", "encoding_failed"},
    "rejection_code": {None, "queue_full", "invalid_request_error", "request_timeout", "model_unavailable"},
}
BOOLS = set("stream has_in_flight producer_finished transport_closed cancellation_requested output_drained".split())
COUNTS = set("pid mtp_depth buffered_bytes buffered_events queued_events in_flight_bytes prompt_tokens completion_tokens text_bytes text_limit_bytes lease_id rejection_status".split())
TIMES = set("uptime_seconds scheduler_elapsed_seconds prefill_seconds decode_seconds connection_age_seconds send_elapsed_seconds".split())
NUMBERS_PROMPT = [{"role": "user", "content": "按顺序写出从1到100的整数，使用英文逗号分隔，不要解释，也不要省略。"}]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse_record(record, pid):
    require(isinstance(record, dict), "Lifecycle JSON must be an object")
    event = record.get("event")
    require(record.get("schema") == SCHEMA and event in FIELDS, "Unknown lifecycle schema/event")
    require(COMMON <= record.keys() and record.keys() <= COMMON | BUFFER | FIELDS[event], "Missing or unexpected log fields")
    require(record["pid"] == pid and type(record["pid"]) is int, "Unexpected server PID")
    require(re.fullmatch(UUID, record["connection_id"] or "") is not None, "Invalid connection ID")
    rid = record["request_id"]
    require(rid is None or re.fullmatch(REQUEST, rid) is not None, "Invalid request ID")
    require(record["reason"] in REASONS[event], "Unknown lifecycle reason")
    if rid is not None:
        require(BUFFER <= record.keys(), "Request record lacks buffer snapshot")
    for key, value in record.items():
        if key in ENUMS:
            require(value in ENUMS[key], "Unknown finite field: " + key)
        elif key in BOOLS:
            require(type(value) is bool, "Non-boolean field: " + key)
        elif key in COUNTS and value is not None:
            require(type(value) is int and value >= 0, "Invalid counter: " + key)
        elif key in TIMES and value is not None:
            require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "Invalid timing: " + key)
    if rid is not None:
        require(record["mtp_depth"] in (0, 2), "Unknown MTP depth")
        require(record["buffered_bytes"] >= record["in_flight_bytes"], "Invalid retained byte count")
        if event == "output_terminal" and record["stream"]:
            require(record.get("text_bytes") is None and record.get("text_limit_bytes") is None,
                    "SSE must not report nonstream text budget")
    return record


def read_log(path, pid, model):
    """Bounded complete-line reader. Old logs stay readable, but lack new coverage."""
    raw = path.read_bytes()
    require(len(raw) <= 8 * 1024 * 1024, "Server log exceeds 8 MiB gate bound")
    complete = raw[:raw.rfind(b"\n") + 1].decode("utf-8", errors="strict")
    records, legacy = [], {}
    success = (rf"HTTP request id=({REQUEST}) mtp_depth=[02] finish=(?:eos|length) prompt_tokens=\d+ completion_tokens=\d+ "
               rf"prefill_seconds={NUMBER} decode_seconds={NUMBER} scheduler_elapsed_seconds={NUMBER}(?: event=model_terminal)?")
    failed = rf"HTTP request id=({REQUEST}) terminal=(?:failed|cancelled) stage=(?:prefill|decode) scheduler_elapsed_seconds={NUMBER}(?: event=model_terminal)?"
    startup = rf"experimental HTTP pid={pid} address=127\.0\.0\.1:\d+ model={re.escape(model)} state=loading connections=\d+ body_bytes=\d+ output_bytes=\d+"
    for number, line in enumerate(complete.splitlines(), 1):
        if line.startswith("{"):
            records.append(parse_record(json.loads(line), pid))
            continue
        match = re.fullmatch(success, line) or re.fullmatch(failed, line)
        if match:
            legacy.setdefault(match.group(1), []).append(line)
            continue
        allowed = (re.fullmatch(startup, line) or re.fullmatch(r"HTTP model loaded \d+/\d+", line)
                   or line == "HTTP model state=ready default=AR experimental_mtp_depth=2 context=16384 chunk=416")
        require(bool(allowed), f"Unexpected unstructured log line {number}; content withheld")
    return {"records": records, "legacy": legacy, "schema_present": bool(records),
            "ignored_incomplete_tail_bytes": len(raw) - len(complete.encode())}


def validate_request(parsed, expectation):
    rid = expectation["id"]
    records = [r for r in parsed["records"] if r["request_id"] == rid]
    groups = {event: [r for r in records if r["event"] == event] for event in FIELDS}
    require(len(groups["model_terminal"]) == len(groups["connection_close"]) == 1,
            "Need one model terminal and one connection close for " + rid)
    require(len({r["connection_id"] for r in records}) == 1, "Request changed connection identity")
    require(all(r["stream"] == expectation["stream"] and r["mtp_depth"] == expectation["depth"] for r in records),
            "Request configuration mismatch")
    model, close = groups["model_terminal"][0], groups["connection_close"][0]
    outputs, releases = groups["output_terminal"], groups["closed_send_released"]
    require(len(outputs) <= 1 and len(releases) <= 1, "Duplicate output terminal or late lease release")
    require(close["transport_closed"], "Close snapshot must mark transport closed")
    if close["has_in_flight"]:
        require(len(releases) == 1 and releases[0]["lease_id"] == close["lease_id"], "Missing release for closed lease")
        require(releases[0]["buffered_bytes"] == releases[0]["buffered_events"] == releases[0]["in_flight_bytes"] == 0,
                "Closed send did not release retained quota")
    else:
        require(close["buffered_bytes"] == close["buffered_events"] == 0, "Close left unsent quota")
    old = parsed["legacy"].get(rid, [])
    require(len(old) == 1, "Existing legacy request-ID gate is not preserved")
    limit_phase = None
    if expectation["kind"] == "rst":
        require(model["model_kind"] == "cancelled" and model["stage"] == "decode", "RST did not cancel decode")
        require("terminal=cancelled stage=decode " in old[0], "Legacy cancelled/decode gate failed")
        require(close["reason"] in {"transport_failed", "transport_cancelled", "receive_failed", "send_failed", "header_send_failed"},
                "RST was replaced by a deadline/normal close")
        require(close["output_outcome"] in {"disconnected", "cancelled"}, "Unexpected RST output outcome")
        # Disconnect often wins first and emits no separate output_terminal.
        if outputs:
            require(outputs[0]["reason"] == "cancelled" and outputs[0]["output_outcome"] == "cancelled",
                    "Unexpected late output terminal after RST")
    elif expectation["kind"] == "output_limit":
        require(len(outputs) == 1 and outputs[0]["error_code"] == "output_limit", "Missing specific output_limit event")
        require(outputs[0]["reason"] in {"text_limit", "response_limit"} and outputs[0]["output_outcome"] == "failed",
                "Output limit reason/outcome mismatch")
        require(close["reason"] == "terminal_sent" and close["output_outcome"] == "failed", "Limit response did not end normally")
        if outputs[0]["reason"] == "text_limit":
            if model["model_kind"] == "failed":
                require(model["reason"] == "text_limit", "Text limit was lost at scheduler failure")
                limit_phase = "generation_text_limit"
            else:
                require(model["model_kind"] == "completed" and model["reason"] is None,
                        "Unexpected text-limit model terminal")
                # complete(event:) logs model completion before its final
                # utf8.finish() publish; that last replacement can cross quota.
                limit_phase = "completion_utf8_flush_text_limit"
        else:
            require(model["model_kind"] == "completed" and model["reason"] is None,
                    "Final response limit lacks clean model completion")
            limit_phase = "encoded_response_limit"
    else:
        result = expectation["result"]
        require(model["model_kind"] == "completed" and model["model_finish_reason"] == ("eos" if result["finish"] == "stop" else "length"),
                "Successful HTTP response lacks matching model terminal")
        require(model["prompt_tokens"] == result["usage"]["prompt_tokens"] and model["completion_tokens"] == result["usage"]["completion_tokens"],
                "HTTP usage disagrees with model counts")
        require(len(outputs) == 1 and outputs[0]["reason"] == "completed" and outputs[0]["error_code"] is None,
                "Successful response lacks accepted completed output")
        require(close["reason"] == "terminal_sent" and close["output_outcome"] == "completed", "Successful response has wrong close")
        if not expectation["stream"]:
            require(outputs[0]["text_bytes"] == len(result["text"].encode()) and outputs[0]["text_limit_bytes"] == 6144,
                    "Nonstream text-budget accounting mismatch")
    return {"request_id": rid, "connection_id": close["connection_id"], "events": records, "legacy_lines": old,
            "output_limit_phase": limit_phase}


def require_complete_logging(health):
    """Health, not a possibly incomplete log, is the authority for lost records."""
    logs = health.get("logging")
    require(isinstance(logs, dict), "Health lacks bounded logger counters; attribution completeness unknown")
    counts = ("max_bytes", "max_events", "max_event_bytes", "buffered_bytes", "buffered_events",
              "queued_events", "in_flight_bytes", "enqueued_events", "written_events",
              "dropped_events", "dropped_bytes", "write_failures")
    require(all(type(logs.get(k)) is int and logs[k] >= 0 for k in counts), "Invalid bounded logger counter")
    require(logs.get("accepting") is True and logs.get("writer_exited") is False,
            "Logger is not accepting; attribution completeness unknown")
    require(logs["in_flight_bytes"] <= logs["buffered_bytes"] <= logs["max_bytes"]
            and logs["queued_events"] <= logs["buffered_events"] <= logs["max_events"], "Logger quota exceeded")
    require(logs["dropped_events"] == logs["dropped_bytes"] == logs["write_failures"] == 0
            and logs.get("last_write_errno") is None,
            "Lifecycle attribution incomplete: dropped_events=%d dropped_bytes=%d write_failures=%d"
            % (logs["dropped_events"], logs["dropped_bytes"], logs["write_failures"]))
    return logs


def wait_logs(path, child, model, expectations, harness, timeout=10):
    deadline, last = time.monotonic() + timeout, None
    while time.monotonic() < deadline:
        require(child.poll() is None, "Owned server exited while waiting for log records")
        require_complete_logging(harness.health()[1])
        parsed = read_log(path, child.pid, model)
        try:
            evidence = [validate_request(parsed, expected) for expected in expectations]
            return parsed, evidence
        except ValueError as error:
            last = str(error)
        time.sleep(.2)
    raise TimeoutError("Asynchronous lifecycle log gate: " + str(last))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runner", "model-dir", "fixture", "output", "audit-log"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--port", type=int, default=11236)
    parser.add_argument("--model-id")
    parser.add_argument("--pid", type=int)
    parser.add_argument("--try-output-limit", action="store_true")
    args = parser.parse_args()
    if args.audit_log:
        if not args.pid or not args.model_id:
            parser.error("--audit-log requires --pid and --model-id")
        parsed = read_log(args.audit_log, args.pid, args.model_id)
        print(json.dumps({"format_valid": True, "schema_present": parsed["schema_present"],
            "structured_records": len(parsed["records"]), "legacy_requests": len(parsed["legacy"]),
            "candidate_lifecycle_passed": False, "note": "Read-only format audit; no request/HTTP assertions."}))
        return 0
    if any(getattr(args, name) is None for name in ("runner", "model_dir", "fixture", "output")):
        parser.error("Live mode requires --runner --model-dir --fixture --output")
    if not 1024 <= args.port <= 65535:
        parser.error("Port must be 1024...65535")
    out, model = args.output.resolve(), args.model_dir.resolve().name
    log = out.with_suffix(".server.log")
    if out.exists() or log.exists():
        parser.error("Output report and server log must be new")
    out.parent.mkdir(parents=True, exist_ok=True)
    child = harness = None
    expectations = []
    report = {"schema": "qwen-http-terminal-log-gate-v1", "complete": False, "passed": False, "checks": [], "requests": [],
        "output_limit": {"requested": args.try_output_limit, "status": "not_run", "passed": False},
        "notes": ["One real model under the outer GPU controller; no speed claim.",
                  "RST requires two nonempty content frames; first token may come from prefill.",
                  "Structured records may arrive after HTTP/health; correlated by request ID, not line order.",
                  "No SSE overflow or send/connection deadline pass is claimed.",
                  "Live log attribution requires zero health-reported drops/write failures through the final snapshot."]}

    def save():
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")

    def check(name, condition, **evidence):
        report["checks"].append({"id": name, "passed": bool(condition), **evidence})
        save()
        require(condition, name)

    def chat(messages, depth=0, stream=False, budget=4):
        return {"model": model, "messages": messages, "max_tokens": budget, "mtp_depth": depth, "stream": stream}

    def generate(label, depth, stream):
        result = decode_completion(harness.request("POST", "/v1/chat/completions", chat(NUMBERS_PROMPT, depth, stream), timeout=180), stream)
        expected = {"label": label, "kind": "normal", "id": result["id"], "depth": depth, "stream": stream, "result": result}
        expectations.append(expected)
        report["requests"].append(expected)
        save()
        return result

    def equivalent(a, b):
        return all(a[key] == b[key] for key in ("text", "finish", "usage"))

    def interrupted(number, _frame):
        raise InterruptedError(f"Harness received signal {number}")

    def deadline(_number, _frame):
        raise TimeoutError("Fixed 600-second work budget exhausted; no retry")

    previous_term = signal.signal(signal.SIGTERM, interrupted)
    previous_int = signal.signal(signal.SIGINT, interrupted)
    previous_alarm = signal.signal(signal.SIGALRM, deadline)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 600)
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", args.port))
        fixture = [{"role": "system", "content": (args.fixture / "system-prompt.txt").read_text().strip()},
                   {"role": "user", "content": (args.fixture / "user-prompt.txt").read_text().strip()}]
        command = [str(args.runner.resolve()), "serve-gpu", "--model-dir", str(args.model_dir.resolve()), "--port", str(args.port),
                   "--max-connections", "4", "--output-buffer-bytes", "8192"]
        report.update(command=command, runner_sha256=hashlib.sha256(args.runner.read_bytes()).hexdigest(),
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      edge_helper_sha256=hashlib.sha256((REPO / "scripts/test_http_server_edges.py").read_bytes()).hexdigest(),
                      fixture_sha256=hashlib.sha256(json.dumps(fixture, ensure_ascii=False, sort_keys=True).encode()).hexdigest())
        with log.open("x") as output:
            # Inherit the controller-owned process group; do not setsid/daemonize.
            child = subprocess.Popen(command, cwd=REPO, stdout=output, stderr=subprocess.STDOUT)
        report["server_pid"] = child.pid
        harness = HTTPHarness(args.port, child)
        report["ready"] = harness.until(lambda item: item[0] == 200 and item[1].get("ready") is True, timeout=180)
        ar, mtp = generate("baseline_ar", 0, False), generate("baseline_mtp", 2, True)
        check("normal_ar_mtp_fixed_budget", equivalent(ar, mtp) and ar["text"] == "1,2," and ar["finish"] == "length"
              and ar["usage"]["completion_tokens"] == 4, idle=harness.idle())
        sock = harness.post_socket(chat(fixture, 2, True, 256))
        try:
            prefix = content_prefix(sock, count=2, timeout=180)
        finally:
            reset_at = time.monotonic()
            harness.close(sock, reset=True)
        rst = {"label": "mtp_decode_rst", "kind": "rst", "id": prefix["id"], "depth": 2, "stream": True}
        expectations.append(rst)
        report["requests"].append({**rst, "content_frames": prefix["content_frames"], "received_body_bytes": prefix["received_body_bytes"]})
        check("real_rst_after_decode_and_release", prefix["content_frames"] >= 2,
              idle=harness.idle(), cancel_to_idle_observed_seconds=time.monotonic() - reset_at)
        check("fresh_ar_mtp_after_rst", equivalent(generate("fresh_ar", 0, False), ar)
              and equivalent(generate("fresh_mtp", 2, True), mtp), idle=harness.idle())
        parsed, evidence = wait_logs(log, child, model, expectations, harness)
        check("normal_and_rst_structured_and_legacy_paths", True, requests=evidence)
        if args.try_output_limit:
            # 2400 Han characters = 7200 UTF-8 bytes if actually copied. Model
            # compliance is deliberately not assumed; there is no forced output.
            target = "山川日月" * 600
            message = "请逐字抄写下方文本，不要解释、不要代码块、不要省略或使用重复次数说明。只输出原文：\n" + target
            before_ids = {r["request_id"] for r in parsed["records"] if r["request_id"]}
            body = chat([{"role": "user", "content": message}], 0, False, 4096)
            status, headers, raw, wall = harness.request("POST", "/v1/chat/completions", body, timeout=310)
            report["output_limit"].update(http_status=status, response=json.loads(raw), wall_seconds=wall,
                intended_copy_bytes=len(target.encode()), request=body, idle=harness.idle())
            save()
            end = time.monotonic() + 10
            new_ids = set()
            while time.monotonic() < end:
                parsed = read_log(log, child.pid, model)
                new_ids = {r["request_id"] for r in parsed["records"] if r["request_id"]} - before_ids
                if len(new_ids) == 1:
                    break
                time.sleep(.2)
            require(len(new_ids) == 1, "Cannot uniquely correlate optional nonstream response")
            rid = next(iter(new_ids))
            if status == 500 and json.loads(raw).get("error", {}).get("code") == "output_limit":
                expectation = {"label": "output_limit_attempt", "kind": "output_limit", "id": rid, "depth": 0, "stream": False}
                report["output_limit"].update(status="observed", http_code_observed=True, passed=False)
            elif status == 200:
                result = decode_completion((status, headers, raw, wall), False)
                require(result["id"] == rid, "Optional completion ID mismatch")
                expectation = {"label": "output_limit_attempt", "kind": "normal", "id": rid, "depth": 0, "stream": False, "result": result}
                report["output_limit"].update(status="not_triggered", passed=False)
            else:
                raise ValueError("Output-limit candidate returned an unrelated failure")
            expectations.append(expectation)
            report["requests"].append(expectation)
            _, optional_evidence = wait_logs(log, child, model, [expectation], harness)
            report["output_limit"]["log_evidence"] = optional_evidence
            report["output_limit"]["passed"] = report["output_limit"]["status"] == "observed"
            check("fresh_ar_mtp_after_optional_attempt", equivalent(generate("post_limit_ar", 0, False), ar)
                  and equivalent(generate("post_limit_mtp", 2, True), mtp), idle=harness.idle())
        parsed, evidence = wait_logs(log, child, model, expectations, harness)
        expected_ids = {item["id"] for item in expectations}
        observed_ids = {item["request_id"] for item in parsed["records"] if item["request_id"] is not None}
        final_idle = harness.idle()
        report["logging_health"] = require_complete_logging(final_idle)
        check("all_correlated_terminal_paths_and_log_schema", len(expected_ids) == len(expectations) and observed_ids == expected_ids,
              requests=evidence,
              log_format="structured_with_legacy_compatibility", final_idle=final_idle)
        report["complete"] = True
        report["passed"] = all(c["passed"] for c in report["checks"]) and (not args.try_output_limit or report["output_limit"]["passed"])
    except BaseException as error:
        report["passed"] = False
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
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
        except BaseException as error:
            report["passed"] = False
            report["cleanup_error"] = f"{type(error).__name__}: {error}"
        finally:
            if child is not None:
                report["server_exit_code"] = child.poll()
                report["graceful_shutdown"] = child.returncode == 0 and not report.get("forced_shutdown", False)
                report["passed"] = report["passed"] and report["graceful_shutdown"]
            try:
                save()
            finally:
                signal.signal(signal.SIGTERM, previous_term)
                signal.signal(signal.SIGINT, previous_int)
                signal.signal(signal.SIGALRM, previous_alarm)
                signal.setitimer(signal.ITIMER_REAL, *previous_timer)
    print(json.dumps({**{key: report.get(key) for key in ("passed", "complete", "error", "graceful_shutdown")},
                      "output_limit_status": report["output_limit"]["status"],
                      "output_limit_passed": report["output_limit"]["passed"]}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
