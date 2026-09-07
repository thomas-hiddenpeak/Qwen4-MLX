#!/usr/bin/env python3
"""Fixed 12-cycle short HTTP soak, owned by the outer GPU controller.

One model server, no retries or adaptive acceptance. Each cycle cancels MTP
after two nonempty content frames and checks a fresh identical MTP request.
This is not an eight-hour stability test or an inference speed benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import signal
import socket
import subprocess
import time

from test_http_server_edges import HTTPHarness, content_prefix, decode_completion
import test_http_server_edges as edge_helpers


CYCLES = 12
BUDGET = 128
WORK_DEADLINE_SECONDS = 1150
TOTAL_BUDGET_SECONDS = 1200


def process_counts(child):
    """Read only our live child's RSS and numeric FD count, never file names."""
    if child.poll() is not None:
        raise RuntimeError("Cannot sample an exited model child")
    pid = str(child.pid)
    started = time.monotonic()
    rss = subprocess.run(["/bin/ps", "-o", "rss=", "-p", pid],
                         capture_output=True, text=True, timeout=3, check=True)
    rss_kib = int(rss.stdout.strip())
    # -Ff emits PID and descriptor identifiers, excluding file names/paths.
    # cwd/txt/memory mappings are not numeric descriptors and are not counted.
    descriptors = subprocess.run(["/usr/sbin/lsof", "-nP", "-a", "-p", pid, "-Ff"],
                                 capture_output=True, text=True, timeout=5, check=True)
    ids = {int(line[1:]) for line in descriptors.stdout.splitlines() if re.fullmatch(r"f[0-9]+", line)}
    seen_pids = {int(line[1:]) for line in descriptors.stdout.splitlines() if re.fullmatch(r"p[0-9]+", line)}
    if child.poll() is not None or seen_pids != {child.pid} or rss_kib <= 0 or not ids:
        raise RuntimeError("Incomplete or mismatched owned-child RSS/FD observation")
    return {"pid": child.pid, "rss_kib": rss_kib, "numeric_fd_count": len(ids),
            "sampling_seconds": time.monotonic() - started,
            "rss_source": "ps rss (KiB)", "fd_source": "lsof unique numeric descriptor identifiers"}


def equivalent(a, b):
    return all(a[key] == b[key] for key in ("text", "usage", "finish"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runner", "model-dir", "fixture", "golden-report", "output"):
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
    child, harness = None, None
    started = time.monotonic()
    report = {"schema": "qwen-live-http-short-soak-v1", "complete": False, "passed": False,
              "planned_cycles": CYCLES, "checks": [], "cycles": [], "process_samples": [],
              "work_deadline_seconds": WORK_DEADLINE_SECONDS, "total_budget_seconds": TOTAL_BUDGET_SECONDS,
              "notes": ["Exactly 12 cycles are planned before execution; failures stop the run without retries.",
                        "Decode network cancellation does not identify an interruption inside verify or replay.",
                        "RSS is the external ps resident-set observation, not MLX active bytes or physical footprint.",
                        "FD counts include only numeric process descriptors, not cwd/txt/memory-map entries.",
                        "Five idle process samples are descriptive; no leak-free or eight-hour stability claim follows.",
                        "No paused-reader, live overflow or send-deadline case is included."]}

    def save():
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")

    def check(name, condition, **evidence):
        report["checks"].append({"id": name, "passed": bool(condition), **evidence})
        save()
        print(name, bool(condition), flush=True)
        if not condition:
            raise AssertionError(name)

    def chat(messages, depth=2, stream=False, budget=BUDGET):
        return {"model": args.model_dir.resolve().name, "messages": messages,
                "max_tokens": budget, "mtp_depth": depth, "stream": stream}

    def generate(body):
        return decode_completion(harness.request("POST", "/v1/chat/completions", body, timeout=180), body["stream"])

    def sample(cycle, health):
        observation = process_counts(child)
        observation.update(cycle=cycle, elapsed_seconds=time.monotonic() - started, idle_health=health)
        report["process_samples"].append(observation)
        save()

    def interrupted(number, _frame):
        raise InterruptedError(f"Harness received signal {number}")

    def deadline(_number, _frame):
        raise TimeoutError("Fixed 1150-second work deadline reached; remaining cycles are not retried")

    previous_term = signal.signal(signal.SIGTERM, interrupted)
    previous_int = signal.getsignal(signal.SIGINT)
    previous_alarm = signal.signal(signal.SIGALRM, deadline)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, WORK_DEADLINE_SECONDS)
    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", args.port))
        golden = json.loads(args.golden_report.read_text())["trials"][0]
        expected = {"text": golden["text"], "finish": "stop", "usage": {
            "prompt_tokens": len(golden["prompt_tokens"]), "completion_tokens": len(golden["generated_token_ids"]),
            "total_tokens": len(golden["prompt_tokens"]) + len(golden["generated_token_ids"])}}
        if golden["finish_reason"] != "eos" or expected["usage"]["prompt_tokens"] != 11216 or expected["usage"]["completion_tokens"] != 85:
            raise ValueError("Require the frozen tools fixture golden: 11216 prompt tokens and natural 85-token EOS")
        messages = [{"role": "system", "content": (args.fixture / "system-prompt.txt").read_text().strip()},
                    {"role": "user", "content": (args.fixture / "user-prompt.txt").read_text().strip()}]
        short_messages = [{"role": "user", "content": "请只输出下面这行文字，不要解释：中文测试，海浪🌊。"}]
        command = [str(args.runner.resolve()), "serve-gpu", "--model-dir", str(args.model_dir.resolve()),
                   "--port", str(args.port), "--max-connections", "4", "--output-buffer-bytes", "8192"]
        report.update(command=command, request=chat(messages), short_messages=short_messages,
                      expected_frozen_output=expected, golden_report=str(args.golden_report.resolve()),
                      golden_sha256=hashlib.sha256(args.golden_report.read_bytes()).hexdigest(),
                      runner_sha256=hashlib.sha256(args.runner.read_bytes()).hexdigest(),
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      edge_helpers_sha256=hashlib.sha256(Path(edge_helpers.__file__).read_bytes()).hexdigest())
        with log.open("x") as output:
            child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        report["server_pid"] = child.pid
        harness = HTTPHarness(args.port, child)
        save()
        ready = harness.until(lambda item: item[0] == 200 and item[1].get("ready") is True, timeout=180)
        check("ready_owned_child", ready["pid"] == child.pid, health=ready)
        baseline = generate(chat(messages))
        report["baseline_mtp"] = baseline
        check("initial_mtp_matches_frozen_tools_output", equivalent(baseline, expected), result=baseline)
        ar = generate(chat(messages, depth=0))
        report["baseline_ar"] = ar
        check("initial_ar_matches_mtp_and_frozen_output", equivalent(ar, baseline), result=ar)
        short_baseline = generate(chat(short_messages, depth=0, budget=64))
        report["baseline_short"] = short_baseline
        check("initial_short_unicode", short_baseline["text"] == "中文测试，海浪🌊。"
              and short_baseline["finish"] == "stop" and short_baseline["usage"]["completion_tokens"] == 9,
              result=short_baseline)
        sample(0, harness.idle())

        for index in range(1, CYCLES + 1):
            cycle_started = time.monotonic()
            cycle = {"index": index, "complete": False, "started_elapsed_seconds": cycle_started - started}
            report["cycles"].append(cycle)
            save()
            sock = harness.post_socket(chat(messages, stream=True))
            try:
                prefix = content_prefix(sock, count=2, timeout=180)
            finally:
                reset_started = time.monotonic()
                harness.close(sock, reset=True)
            cycle["cancel_prefix"] = prefix
            save()
            idle = harness.idle()
            cycle["cancel_to_idle_observed_seconds"] = time.monotonic() - reset_started
            cycle["idle_after_cancel"] = idle
            lines = harness.terminal_log_lines(log, prefix["id"])
            cycle["cancel_server_lines"] = lines
            check(f"cycle_{index:02d}_decode_rst_released", prefix["content_frames"] >= 2
                  and len(lines) == 1 and "terminal=cancelled stage=decode " in lines[0],
                  request_id=prefix["id"], elapsed_seconds=cycle["cancel_to_idle_observed_seconds"],
                  server_lines=lines, health=idle)

            recovered = generate(chat(messages, stream=True))
            cycle["fresh_mtp"] = recovered
            check(f"cycle_{index:02d}_fresh_mtp_exact", equivalent(recovered, baseline), result=recovered)
            short_depth = 0 if index % 2 else 2
            short = generate(chat(short_messages, depth=short_depth, stream=short_depth == 2, budget=64))
            cycle["short_request"] = {"mtp_depth": short_depth, "result": short}
            check(f"cycle_{index:02d}_short_exact", equivalent(short, short_baseline), mtp_depth=short_depth, result=short)
            if index % 3 == 0:
                full_ar = generate(chat(messages, depth=0))
                cycle["full_ar"] = full_ar
                check(f"cycle_{index:02d}_full_ar_exact", equivalent(full_ar, baseline), result=full_ar)
            idle = harness.idle()
            cycle["idle_at_end"] = idle
            if index % 3 == 0:
                sample(index, idle)
            cycle["wall_seconds"] = time.monotonic() - cycle_started
            cycle["complete"] = True
            save()

        samples = report["process_samples"]
        report["process_observation"] = {
            "sample_cycles": [item["cycle"] for item in samples],
            "rss_kib": [item["rss_kib"] for item in samples],
            "numeric_fd_count": [item["numeric_fd_count"] for item in samples],
            "rss_end_minus_baseline_kib": samples[-1]["rss_kib"] - samples[0]["rss_kib"],
            "fd_end_minus_baseline": samples[-1]["numeric_fd_count"] - samples[0]["numeric_fd_count"],
            "rss_min_kib": min(item["rss_kib"] for item in samples),
            "rss_max_kib": max(item["rss_kib"] for item in samples),
            "fd_min": min(item["numeric_fd_count"] for item in samples),
            "fd_max": max(item["numeric_fd_count"] for item in samples),
            "interpretation": "Five external idle observations only; no MLX memory, leak-free, or long-duration stability inference."}
        check("exactly_twelve_cycles_and_five_idle_samples", len(report["cycles"]) == CYCLES
              and all(item["complete"] for item in report["cycles"])
              and [item["cycle"] for item in samples] == [0, 3, 6, 9, 12])
        final_idle = harness.idle()
        report["logging_health"] = edge_helpers.require_lossless_logging(final_idle)
        check("final_idle_all_resources_released", True, health=final_idle)
        report["complete"] = True
        report["passed"] = all(item["passed"] for item in report["checks"])
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
            if child is not None:
                report["server_exit_code"] = child.returncode
                report["graceful_shutdown"] = not report.get("forced_shutdown", False) and child.returncode == 0
                report["passed"] = report["passed"] and report["graceful_shutdown"]
            report["elapsed_seconds"] = time.monotonic() - started
            if report["elapsed_seconds"] > TOTAL_BUDGET_SECONDS:
                report["passed"] = False
                report["deadline_exceeded"] = True
            save()
        finally:
            signal.signal(signal.SIGTERM, previous_term)
            signal.signal(signal.SIGINT, previous_int)
            signal.signal(signal.SIGALRM, previous_alarm)
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)
    print(json.dumps({key: report.get(key) for key in
                      ("passed", "complete", "error", "elapsed_seconds", "graceful_shutdown")}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
