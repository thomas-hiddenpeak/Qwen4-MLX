#!/usr/bin/env python3
"""Bounded native-sidecar tests. No GPU/model or third-party Python imports."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import resource
import signal
import struct
import subprocess
import tempfile
import time


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def wait_for(path, record_type, child, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            try:
                data = records(path)
            except json.JSONDecodeError:
                data = []  # Last write may still be in progress.
            if any(row["type"] == record_type for row in data):
                return data
        if child.poll() is not None:
            raise AssertionError(f"Collector exited early: {child.returncode}")
        time.sleep(0.02)
    raise AssertionError(f"No flushed {record_type} record")


def run(executable, path, pid, samples=2, interval=50):
    command = [str(executable), "--output", str(path), "--pid", str(pid),
               "--max-samples", str(samples), "--interval-ms", str(interval)]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert completed.returncode == 0, completed.stderr
    return records(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    exe = args.executable.resolve()
    checks = {}
    with tempfile.TemporaryDirectory(prefix="ane-telemetry-tests-") as temp:
        base = Path(temp)
        live = run(exe, base / "live.jsonl", os.getpid())
        assert [r["type"] for r in live] == ["metadata", "baseline", "sample", "sample", "summary"]
        assert live[-1]["samples_written"] == 2
        assert live[-1]["stop_reason"] == "max_samples"
        assert live[0]["target_start_abstime"] > 0
        assert live[0]["clock"] == "mach_absolute_time_nanoseconds"
        previous = live[1]
        assert previous["process_disk_read_bytes_delta"] is None
        assert previous["system_disk_write_bytes_delta"] is None
        for row in live[2:4]:
            assert row["start_ns"] == previous["end_ns"]
            assert row["start_ns"] <= row["collection_start_ns"] <= row["end_ns"]
            assert row["physical_dram_read_bytes_delta"] is None
            assert row["physical_dram_write_bytes_delta"] is None
            assert row["process"]["error"] is None
            before = previous["process"]["rusage"]
            after = row["process"]["rusage"]
            for direction in ["read", "write"]:
                expected = after[f"disk_{direction}_bytes_cumulative"] - before[f"disk_{direction}_bytes_cumulative"]
                assert row[f"process_disk_{direction}_bytes_delta"] == expected
            io = row["ioreport"]
            if io["kind"] == "bandwidth_residency_histogram" and io["error"] is None:
                assert io["estimated"] and io["unit"] == "events"
                for channel in io["channels"]:
                    assert channel["format"] == 2
                    assert len(channel["state_names"]) == len(channel["residency_delta_raw"])
                    assert "byte_delta" not in channel
            previous = row
        checks["bounded_jsonl_and_clock"] = True
        checks["real_process_endpoints_match_deltas"] = True
        checks["histograms_never_fabricate_dram_bytes"] = True

        # Independent API oracle confirms that CPU accounting uses Mach ticks,
        # rather than silently treating those raw values as nanoseconds.
        proc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        proc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        buffer = ctypes.create_string_buffer(1024)
        before_cpu = resource.getrusage(resource.RUSAGE_SELF)
        assert proc.proc_pid_rusage(os.getpid(), 4, buffer) == 0
        raw_user, raw_system = struct.unpack_from("QQ", buffer.raw, 16)
        after_cpu = resource.getrusage(resource.RUSAGE_SELF)
        timebase = live[0]["timebase"]
        converted = (raw_user + raw_system) * timebase["numer"] / timebase["denom"]
        low = (before_cpu.ru_utime + before_cpu.ru_stime) * 1e9
        high = (after_cpu.ru_utime + after_cpu.ru_stime) * 1e9
        assert low - 2e6 <= converted <= high + 2e6
        checks["cpu_mach_units_checked_against_getrusage"] = True

        existing = base / "existing.jsonl"
        existing.write_bytes(b"preserve me\n")
        bad_commands = [
            ["--output", str(existing), "--pid", str(os.getpid())],
            ["--output", str(base / "bad1"), "--pid", str(os.getpid()), "--interval-ms", "0"],
            ["--output", str(base / "bad2"), "--pid", str(os.getpid()), "--max-samples", "100001"],
            ["--output", str(base / "bad3"), "--pid", str(os.getpid()), "--pid", str(os.getpid())],
        ]
        for command in bad_commands:
            result = subprocess.run([str(exe), *command], capture_output=True, timeout=5)
            assert result.returncode == 2
        assert existing.read_bytes() == b"preserve me\n"
        checks["atomic_no_overwrite_and_invalid_options"] = True

        dead = run(exe, base / "missing.jsonl", 2147483647)
        assert dead[1]["process"]["rusage"] is None
        assert dead[1]["process"]["error"]
        assert dead[1]["process_disk_read_bytes_delta"] is None
        assert dead[-1]["stop_reason"].startswith("target_")
        checks["missing_target_is_error_not_zero"] = True

        path = base / "signal.jsonl"
        collector = subprocess.Popen([str(exe), "--output", str(path), "--pid", str(os.getpid()),
                                      "--interval-ms", "5000", "--max-samples", "100000"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            wait_for(path, "baseline", collector)
            collector.send_signal(signal.SIGTERM)
            _, stderr = collector.communicate(timeout=5)
            assert collector.returncode == 0, stderr
            stopped = records(path)
            assert stopped[-2]["type"] == "sample" and stopped[-2]["is_final_partial"]
            assert stopped[-1]["stop_reason"] == "signal"
            assert stopped[-1]["target_signalled_by_sampler"] is False
        finally:
            if collector.poll() is None:
                collector.kill()
                collector.wait()
        checks["signal_flushes_final_partial_interval"] = True

        target = subprocess.Popen(["/bin/sleep", "30"])
        path = base / "target-exit.jsonl"
        collector = subprocess.Popen([str(exe), "--output", str(path), "--pid", str(target.pid),
                                      "--interval-ms", "50", "--max-samples", "100000"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            wait_for(path, "baseline", collector)
            target.terminate()
            target.wait(timeout=5)
            _, stderr = collector.communicate(timeout=5)
            assert collector.returncode == 0, stderr
            assert records(path)[-1]["stop_reason"].startswith("target_")
        finally:
            for child in [target, collector]:
                if child.poll() is None:
                    child.kill()
                    child.wait()
        checks["collector_exits_when_owned_test_target_disappears"] = True

    report = {"passed": all(checks.values()), "checks": checks,
              "executable": str(exe), "model_started": False,
              "scope": "Small native-sidecar integration checks; private API availability is recorded, not required. No model performance test."}
    content = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(content)
    print(content, end="")


if __name__ == "__main__":
    main()
