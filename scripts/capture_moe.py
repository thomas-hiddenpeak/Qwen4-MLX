#!/usr/bin/env python3
"""Pause our baseline, capture one real request, then restore the baseline.

The capture binary is an isolated instrumentation build of the pinned author
runtime. Captured HTTP timing includes forced materialization and is not a
performance baseline. This controller verifies the process identity first.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "experiments/ane-runner"
SSD = ROOT / "experiments/qwen38-ssd"


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def ready(process, port, timeout=300):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited with {process.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=1) as response:
                return json.load(response)
        except (OSError, ValueError):
            time.sleep(0.5)
    raise TimeoutError(f"Server on {port} not ready")


def stop(process):
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-pid", type=int, required=True)
    args = parser.parse_args()
    baseline = SSD / "runtime/mlx-serve/zig-out/bin/mlx-serve"
    capture = RUNNER / "results/moe-capture/runtime-src/zig-out/bin/mlx-serve"
    destination = RUNNER / "fixtures/moe-real"
    destination.mkdir(parents=True, exist_ok=True)
    for filename in ["ARMED", "prefill.safetensors", "decode.safetensors"]:
        if (destination / filename).exists():
            raise FileExistsError(destination / filename)
    command = subprocess.check_output(["ps", "-p", str(args.baseline_pid), "-o", "command="], text=True).strip()
    if not command.startswith(str(baseline) + " ") or "--port 11235" not in command:
        raise RuntimeError("Refusing to stop an unrecognized baseline process")
    parameters = ["--model", str(SSD / "models/Qwen3.8-Flash-Next-MLX-SSD-Stream"),
                  "--serve", "--host", "127.0.0.1", "--port", "11235", "--ctx-size", "4096",
                  "--prefill-chunk", "512", "--max-concurrent", "1", "--prefix-cache-entries", "0",
                  "--no-mtp", "--no-drafter", "--no-pld", "--metrics"]
    if command != " ".join([str(baseline), *parameters]):
        raise RuntimeError("Baseline options differ; refusing to restart with different options")
    record = {"started_utc": utc(), "original_pid": args.baseline_pid, "original_command": command,
              "capture_binary": str(capture), "capture_directory": str(destination)}
    report = RUNNER / "results/moe-capture/session.json"
    save(report, record)
    capture_process = None
    restore_needed = False
    try:
        os.kill(args.baseline_pid, signal.SIGTERM)
        restore_needed = True
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                os.kill(args.baseline_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.25)
        else:
            raise RuntimeError("Baseline did not stop; refusing a second model load")
        print("Baseline stopped; launching capture server", flush=True)
        capture_parameters = parameters.copy()
        capture_parameters[capture_parameters.index("11235")] = "11236"
        env = dict(os.environ, QWEN4_CAPTURE_MOE_DIR=str(destination))
        with (report.parent / "server.log").open("ab") as log:
            capture_process = subprocess.Popen([str(capture), *capture_parameters], env=env,
                                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record["capture_pid"] = capture_process.pid
        save(report, record)
        record["capture_models"] = ready(capture_process, 11236)
        (destination / "ARMED").write_text(utc() + "\n")
        request = {"model": record["capture_models"]["data"][0]["id"],
                   "messages": [{"role": "user", "content": "请用一句中文解释为什么大模型的专家路由需要根据每个词动态选择。"}],
                   "temperature": 0, "max_tokens": 3, "stream": False,
                   "chat_template_kwargs": {"enable_thinking": False}}
        save(destination / "request.json", request)
        print("Capture server ready and armed; sending real Chinese request", flush=True)
        http = urllib.request.Request("http://127.0.0.1:11236/v1/chat/completions",
                                      data=json.dumps(request, ensure_ascii=False).encode(),
                                      headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(http, timeout=300) as response:
            record["response"] = json.load(response)
        save(destination / "response.json", record["response"])
        for filename in ["prefill.safetensors", "decode.safetensors"]:
            path = destination / filename
            if not path.exists() or path.stat().st_size == 0:
                raise RuntimeError(f"Capture missing: {path}")
        record["capture_complete"] = True
        print("Real prefill and decode captured", flush=True)
    except Exception as error:
        record["error"] = repr(error)
        raise
    finally:
        if capture_process is not None and capture_process.poll() is None:
            stop(capture_process)
        (destination / "ARMED").unlink(missing_ok=True)
        if restore_needed:
            # Do not start a second model if the old process has not exited.
            try:
                os.kill(args.baseline_pid, 0)
                record["restoration"] = "Original process still exists; no second server started"
            except ProcessLookupError:
                with (report.parent / "restored-baseline.log").open("ab") as log:
                    restored = subprocess.Popen([str(baseline), *parameters], stdout=log,
                                                stderr=subprocess.STDOUT, start_new_session=True)
                record["restored_pid"] = restored.pid
                save(report, record)
                print(f"Restoring original baseline as PID {restored.pid}", flush=True)
                try:
                    record["restored_models"] = ready(restored, 11235)
                    status_path = SSD / "results/experiment-status.json"
                    status = json.loads(status_path.read_text())
                    status.update(server_pid=restored.pid, server_log=str(report.parent / "restored-baseline.log"),
                                  updated_utc=utc(), phase="ready_for_use", server_retained=True)
                    save(status_path, status)
                    record["restoration"] = "ready"
                    print("Original baseline ready at http://127.0.0.1:11235", flush=True)
                except Exception as error:
                    record["restoration_error"] = repr(error)
                    raise
        record["finished_utc"] = utc()
        save(report, record)


if __name__ == "__main__":
    main()
