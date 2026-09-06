#!/usr/bin/env python3
"""Integration gate for Swift MoE chunking, scatter and bounded expert reloads.

Creates five small real Core ML SwiGLU packages, executes five distinct tokens
through the Swift CPU-only scheduler, and compares with a per-token Torch FP32
oracle. This is synthetic scheduler coverage, not real-model or ANE evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
H, I, E, K, CAPACITY = 4, 8, 4, 2, 2
ATOL, RTOL = 0.002, 0.002


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor(value: np.ndarray, dtype: str = "float32") -> dict:
    return {"shape": list(value.shape), "dtype": dtype, "values": value.reshape(-1).tolist()}


def export_expert(weights: dict, path: Path) -> None:
    # FP16 interfaces exercise the real runner contract; FP32 internals limit
    # numerical ambiguity in this scheduling test. No production exporter used.
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, H, 1, CAPACITY), dtype=types.fp16)],
                opset_version=ct.target.macOS15)
    def program(x):
        x32 = mb.cast(x=x, dtype="fp32")
        gate = mb.conv(x=x32, weight=weights["gate"][:, :, None, None])
        up = mb.conv(x=x32, weight=weights["up"][:, :, None, None])
        silu = mb.mul(x=gate, y=mb.sigmoid(x=gate))
        middle = mb.mul(x=silu, y=up)
        down = mb.conv(x=middle, weight=weights["down"][:, :, None, None])
        return mb.cast(x=down, dtype="fp16", name="y")

    model = ct.convert(program, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                       compute_precision=ct.precision.FLOAT32, compute_units=ct.ComputeUnit.CPU_ONLY,
                       skip_model_load=True)
    model.save(str(path))


def make_case() -> tuple:
    rng = np.random.default_rng(20260905)
    weights = [{name: (rng.integers(-5, 6, size=shape).astype(np.float32) / 8)
                for name, shape in (("gate", (I, H)), ("up", (I, H)), ("down", (H, I)))}
               for _ in range(E + 1)]
    x = np.array([[3, 2, 0, -1], [2, 0, 3, -1], [0, 2, 3, -1],
                  [-1, 0, 2, 3], [3, -1, 0, 2]], dtype=np.float32)
    router = np.eye(E, H, dtype=np.float32)
    shared_gate = np.array([0.125, -0.25, 0.375, -0.5], dtype=np.float32)
    known_ids = np.array([[0, 1], [2, 0], [2, 1], [3, 2], [0, 3]], dtype=np.int32)
    return x, router, shared_gate, weights, known_ids


def oracle(x: np.ndarray, router: np.ndarray, shared_gate: np.ndarray, weights: list,
           round_expert_outputs: bool) -> dict:
    """No grouping, padding, scatter, cache, production Swift, or export helper."""
    rows = {name: [] for name in ("router_logits", "selected_experts", "routing_weights", "routed_sum",
                                  "shared_down", "shared_gate_logits", "shared_gate", "shared_gated", "y")}

    def expert(one_token: torch.Tensor, expert_id: int) -> torch.Tensor:
        w = {name: torch.from_numpy(value) for name, value in weights[expert_id].items()}
        result = F.linear(F.silu(F.linear(one_token, w["gate"])) * F.linear(one_token, w["up"]), w["down"])
        return result.half().float() if round_expert_outputs else result

    with torch.inference_mode():
        for one in torch.from_numpy(x):
            logits = F.linear(one, torch.from_numpy(router))
            _, ids = torch.topk(logits, K, sorted=True)
            # Softmax over selected logits is mathematically equivalent and
            # intentionally differs from Swift's full-E softmax/gather chain.
            probability = torch.softmax(logits[ids], dim=0)
            routed = sum((probability[slot] * expert(one, int(e)) for slot, e in enumerate(ids)),
                         torch.zeros(H, dtype=torch.float32))
            shared = expert(one, E)
            gate_logit = torch.dot(one, torch.from_numpy(shared_gate)).reshape(1)
            gate = torch.sigmoid(gate_logit)
            gated = shared * gate
            values = (logits, ids, probability, routed, shared, gate_logit, gate, gated, routed + gated)
            for name, value in zip(rows, values):
                rows[name].append(value.numpy())
    return {name: np.stack(values)[None, ...] for name, values in rows.items()}


def compare(actual: np.ndarray, expected: np.ndarray, atol: float, rtol: float) -> dict:
    if actual.shape != expected.shape:
        return {"pass": False, "actual_shape": list(actual.shape), "expected_shape": list(expected.shape)}
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    return {"pass": bool(np.allclose(actual, expected, atol=atol, rtol=rtol)),
            "max_abs": float(np.max(np.abs(delta))), "rms_error": float(np.sqrt(np.mean(delta * delta))),
            "exact": bool(np.array_equal(actual, expected)), "atol": atol, "rtol": rtol}


def invoke(runner: Path, directory: Path, manifest: Path, fixture: Path, cache: int, name: str,
           timeout: int, concurrency: int) -> tuple:
    output_path = directory / f"{name}-swift.json"
    output_path.unlink(missing_ok=True)
    command = [str(runner), "probe-moe", "--manifest", str(manifest), "--fixture", str(fixture),
               "--precision", "float32", "--compute-units", "cpuOnly", "--cache-experts", str(cache),
               "--warmups", "0", "--runs", "2", "--output", str(output_path)]
    if concurrency != 1:
        command.extend(["--expert-concurrency", str(concurrency)])
    start = time.monotonic()
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    (directory / f"{name}.stdout.log").write_text(result.stdout)
    (directory / f"{name}.stderr.log").write_text(result.stderr)
    record = {"command": command, "exit_code": result.returncode, "elapsed_seconds": time.monotonic() - start,
              "stdout": result.stdout, "stderr": result.stderr, "report": str(output_path)}
    return record, json.loads(output_path.read_text()) if output_path.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, default=ROOT / ".build/release/ane-runner")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/moe-scheduler-test")
    parser.add_argument("--timeout", type=int, default=120, help="Seconds per Swift invocation")
    parser.add_argument("--expert-concurrency", type=int, default=1, help="Requested degree; cache capacity one must still run serially")
    args = parser.parse_args()
    directory, runner = args.output_dir.resolve(), args.runner.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    report = {"schema": "moe-scheduler-integration-v1", "status": "running", "checks": {},
              "scope": "Synthetic CPU-only Core ML scheduling integration; no real-model or ANE placement claim",
              "dimensions": {"hidden": H, "intermediate": I, "experts": E, "top_k": K,
                             "token_capacity": CAPACITY, "tokens": 5},
              "requested_expert_concurrency": args.expert_concurrency,
              "versions": {"torch": torch.__version__, "coremltools": ct.__version__, "platform": platform.platform()}}
    report_path = directory / "report.json"
    try:
        if not runner.is_file():
            raise ValueError(f"Build the Swift runner first: {runner}")
        if args.expert_concurrency < 1:
            raise ValueError("Expert concurrency must be positive")
        torch.set_num_threads(1)
        x, router, shared_gate, weights, known_ids = make_case()
        reference32 = oracle(x, router, shared_gate, weights, False)
        reference_io = oracle(x, router, shared_gate, weights, True)
        if not np.array_equal(reference32["selected_experts"][0], known_ids):
            raise ValueError("Independent oracle disagrees with hand-derived routing IDs")
        router.astype("<f4").tofile(directory / "router.f32.bin")
        shared_gate.astype("<f4").tofile(directory / "shared_gate.f32.bin")
        fixture_path = directory / "fixture.json"
        write_json(fixture_path, {"inputs": {"x": tensor(x[None])}, "expectedOutputs": {
            name: tensor(value, "int32" if name == "selected_experts" else "float32")
            for name, value in reference32.items()}})
        np.savez(directory / "oracle.npz", x=x, router=router, shared_gate=shared_gate,
                 **{f"fp32_{name}": value for name, value in reference32.items()},
                 **{f"io_{name}": value for name, value in reference_io.items()},
                 **{f"expert_{i}_{name}": value for i, w in enumerate(weights) for name, value in w.items()})
        for i, weight in enumerate(weights):
            export_expert(weight, directory / (f"expert_{i}.mlpackage" if i < E else "shared.mlpackage"))
        manifest = {"schema_version": 1, "layer_index": 0, "hidden_size": H, "expert_count": E, "top_k": K,
                    "token_capacity": CAPACITY, "input_name": "x", "output_name": "y", "dtype": "float16",
                    "routing": {"weights_file": "router.f32.bin", "shared_gate_file": "shared_gate.f32.bin", "dtype": "float32_le"},
                    "experts": {str(i): f"expert_{i}.mlpackage" for i in range(E)},
                    "shared_expert": "shared.mlpackage", "weight_mode": "synthetic_fp32_compute_fp16_io"}
        manifest_path = directory / "manifest.json"
        write_json(manifest_path, manifest)
        counts = np.bincount(known_ids.reshape(-1), minlength=E)
        expected_calls = int(np.sum((counts + CAPACITY - 1) // CAPACITY))
        report["coverage"] = {"known_expert_ids": known_ids.tolist(), "assignments_per_expert": counts.tolist(),
                              "expected_expert_calls_per_iteration": expected_calls, "expected_shared_calls": 3,
                              "expert_chunks_with_padding": int(np.count_nonzero(counts % CAPACITY)),
                              "shared_chunks_with_padding": 1, "all_valid_assignments": int(counts.sum()),
                              "padding_boundary": "Valid outputs and call counts checked; internal padded cells are not observable"}
        reports = {}
        for cache in (E, 1):
            name = f"cache{cache}"
            call, actual = invoke(runner, directory, manifest_path, fixture_path, cache, name, args.timeout, args.expert_concurrency)
            if call["exit_code"] != 0 or actual is None:
                raise RuntimeError(f"{name} failed: {call}")
            checks = {}
            for key, expected in reference32.items():
                item = actual["outputs"][key]
                observed = np.asarray(item["values"]).reshape(item["shape"])
                tolerance = (0.0, 0.0) if key == "selected_experts" else ((1e-6, 1e-6) if key in
                    ("router_logits", "routing_weights", "shared_gate_logits", "shared_gate") else (ATOL, RTOL))
                io_tolerance = (0.0, 0.0) if key == "selected_experts" else (1e-6, 1e-6)
                checks[key] = {"torch_fp32": compare(observed, expected, *tolerance),
                               "torch_fp32_with_fp16_expert_io": compare(observed, reference_io[key], *io_tolerance)}
            timings = [actual["firstIteration"], *actual["iterations"]]
            checks["calls_and_reload"] = {
                "pass": all(t["expertCalls"] == expected_calls and t["sharedCalls"] == 3 for t in timings)
                    and timings[0]["cacheMisses"] == E + 1
                    and all(t["cacheMisses"] == (E if cache == 1 else 0) for t in timings[1:]),
                "observed_timings": timings,
                "expected_cache_misses": [E + 1, E if cache == 1 else 0, E if cache == 1 else 0],
                "cache_scope": "Routed experts are bounded; shared expert has one separate resident slot",
            }
            passed = checks["calls_and_reload"]["pass"] and all(
                value["torch_fp32"]["pass"] and value["torch_fp32_with_fp16_expert_io"]["pass"]
                for key, value in checks.items() if key != "calls_and_reload")
            report["checks"][name] = {"pass": passed, "invocation": call, "outputs": checks}
            reports[name] = actual
        report["checks"]["cache_independence"] = {"pass": reports["cache4"]["outputs"] == reports["cache1"]["outputs"]}
        missing = dict(manifest)
        missing["experts"] = {key: value for key, value in manifest["experts"].items() if key != "2"}
        missing_path = directory / "missing-expert-manifest.json"
        write_json(missing_path, missing)
        call, actual = invoke(runner, directory, missing_path, fixture_path, 1, "missing-expert", args.timeout, args.expert_concurrency)
        report["checks"]["missing_expert"] = {
            "pass": call["exit_code"] != 0 and actual is None
                and "Dynamic routing requires missing experts: 2" in call["stderr"],
            "removed_expert": 2, "invocation": call,
        }
        report["source"] = {"script": str(Path(__file__).resolve()), "script_sha256": digest(Path(__file__)),
                            "runner": str(runner), "runner_sha256": digest(runner),
                            "fixture_sha256": digest(fixture_path), "manifest_sha256": digest(manifest_path)}
        report["status"] = "pass" if all(check["pass"] for check in report["checks"].values()) else "fail"
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.TimeoutExpired) as error:
        report["status"], report["error"] = "fail", str(error)
    write_json(report_path, report)
    print(json.dumps({"status": report["status"], "report": str(report_path),
                      "checks": {name: value["pass"] for name, value in report["checks"].items()},
                      **({"error": report["error"]} if "error" in report else {})}, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
