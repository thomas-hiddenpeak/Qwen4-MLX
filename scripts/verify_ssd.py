#!/usr/bin/env python3
"""Verify the Swift lookup CLI against saved real rows and an independent PyTorch reference.

Reads only the table header and requested rows, never the whole table. Does not
start a model server or install packages. Run with the existing project .venv.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch


class CheckFailure(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise CheckFailure(message)


def read_json(path):
    # Swift may encode negative floating zero as JSON -0; preserve its sign.
    return json.loads(Path(path).read_text(), parse_int=lambda value: -0.0 if value == "-0" else int(value))


class TableReference:
    """Independent safetensors header reader; values are decoded by PyTorch."""

    def __init__(self, path):
        self.path = path
        self.fd = os.open(path, os.O_RDONLY)
        self.bytes_read = 0
        try:
            self.file_bytes = os.fstat(self.fd).st_size
            header_size = int.from_bytes(self.read(0, 8), "little")
            require(0 < header_size <= 1024 * 1024, "Invalid or unexpectedly large safetensors header")
            self.header = json.loads(self.read(8, header_size))
            weight = self.header["weight"]
            require(weight["dtype"] == "F8_E4M3", "Expected FP8 E4M3 table")
            shape = weight["shape"]
            require(len(shape) == 2 and all(type(v) is int and v > 0 for v in shape), "Invalid table shape")
            self.row_count, self.dimension = shape
            begin, end = weight["data_offsets"]
            require(type(begin) is int and type(end) is int and 0 <= begin < end, "Invalid payload offsets")
            self.offset = 8 + header_size + begin
            require(end - begin == self.row_count * self.dimension, "Payload extent differs from shape")
            require(8 + header_size + end <= self.file_bytes, "Table is truncated")
            self.scale = np.float32(self.header["__metadata__"]["scale"])
            require(np.isfinite(self.scale) and self.scale > 0, "Invalid FP8 scale")
        except BaseException:
            os.close(self.fd)
            raise

    def read(self, offset, length):
        body = os.pread(self.fd, length, offset)
        self.bytes_read += len(body)
        require(len(body) == length, f"Short read at offset {offset}")
        return body

    def rows(self, row_ids):
        require(all(0 <= row < self.row_count for row in row_ids), "Reference row outside table")
        payload = b"".join(self.read(self.offset + row * self.dimension, self.dimension) for row in row_ids)
        return np.frombuffer(payload, dtype=np.uint8).copy().reshape(len(row_ids), self.dimension)

    def close(self):
        os.close(self.fd)


def decode_reference(raw, scale):
    # Deliberately use PyTorch's native FP8/BF16 casts, not the Swift bit formula.
    fp32 = torch.from_numpy(raw).view(torch.float8_e4m3fn).to(torch.float32)
    decoded = (fp32 * float(scale)).to(torch.bfloat16).to(torch.float32).numpy()
    require(np.isfinite(decoded).all(), "Reference rows contain nonfinite FP8 values")
    return decoded


def compare_exact(actual, expected):
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    require(actual.shape == expected.shape, f"Shape mismatch: {actual.shape} != {expected.shape}")
    require(np.isfinite(actual).all(), "CLI returned nonfinite values")
    different = actual.view(np.uint32) != expected.view(np.uint32)
    locations = np.argwhere(different)
    return {
        "passed": not bool(locations.size),
        "compared_scalars": int(actual.size),
        "different_float32_bit_patterns": int(different.sum()),
        "max_absolute_difference": float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))),
        "first_mismatch": None if not locations.size else {
            "index": locations[0].tolist(),
            "actual": float(actual[tuple(locations[0])]),
            "expected": float(expected[tuple(locations[0])]),
        },
    }


def invoke(runner, model_dir, rows, directory, case, timeout, expect_rejection=False):
    output_path = directory / f"{case}.json"
    command = [str(runner), "lookup", "--model-dir", str(model_dir), "--rows", ",".join(map(str, rows)), "--output", str(output_path)]
    began = time.monotonic()
    completed = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)
    record = {"case": case, "row_ids": rows, "returncode": completed.returncode,
              "wall_seconds": time.monotonic() - began, "stderr_tail": completed.stderr[-3000:]}
    if expect_rejection:
        record["passed"] = completed.returncode > 0
        return None, record
    require(completed.returncode == 0, f"CLI {case} failed ({completed.returncode}): {completed.stderr[-3000:]}")
    require(output_path.is_file(), f"CLI {case} did not write --output JSON")
    return read_json(output_path), record


def verify_case(runner, model_dir, table, rows, directory, case, timeout, expected=None):
    raw = table.rows(rows)
    reference = decode_reference(raw, table.scale)
    fixture_comparison = None if expected is None else compare_exact(reference, expected)
    if fixture_comparison is not None:
        require(fixture_comparison["passed"], "Saved BF16 sample differs from fresh independent decoding")
    output, record = invoke(runner, model_dir, rows, directory, case, timeout)
    require(output.get("row_count") == table.row_count, f"{case}: wrong complete-table row_count")
    require(output.get("dimension") == table.dimension, f"{case}: wrong dimension")
    require(np.float32(output.get("scale")) == table.scale, f"{case}: wrong scale")
    require(output.get("row_ids") == rows, f"{case}: row order or duplicate IDs changed")
    require(output.get("output_dtype") == "bfloat16_as_float32", f"{case}: unexpected output dtype")
    values = np.asarray(output["values"], dtype=np.float32)
    require(values.shape in ((len(rows), table.dimension), (len(rows) * table.dimension,)), f"{case}: unexpected values shape {values.shape}")
    comparison = compare_exact(values.reshape(reference.shape), reference)
    record.update(comparison)
    if fixture_comparison is not None:
        record["saved_fixture_vs_pytorch"] = fixture_comparison
    return record, raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="Explicit completed local model directory")
    parser.add_argument("--runner", type=Path, required=True, help="Built .build/release/ane-runner executable")
    parser.add_argument("--sample-dir", type=Path, required=True, help="Directory containing rows.npz and metadata.json")
    parser.add_argument("--verification", type=Path, help="Full-file download verification JSON; defaults to sample-dir/../download-verification.json")
    parser.add_argument("--random-rows", type=int, default=32, help="Additional deterministic random rows (default 32; maximum 256)")
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--timeout", type=float, default=120, help="Per-CLI invocation timeout in seconds")
    parser.add_argument("--report", type=Path, help="Also save the complete JSON report to this path")
    args = parser.parse_args()
    if not 0 <= args.random_rows <= 256 or args.timeout <= 0:
        parser.error("--random-rows must be 0..256 and --timeout must be positive")
    runner, model_dir, sample_dir = (path.expanduser().resolve() for path in (args.runner, args.model_dir, args.sample_dir))
    verification_path = (args.verification or sample_dir.parent / "download-verification.json").expanduser().resolve()
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "passed": False,
              "runner": str(runner), "model_directory": str(model_dir), "sample_directory": str(sample_dir),
              "reference": "PyTorch native float8_e4m3fn -> float32 multiply by file scale -> bfloat16 -> float32",
              "versions": {"numpy": np.__version__, "torch": torch.__version__}, "seed": args.seed, "cases": [],
              "scope": "Exact row lookup, order, boundary rejection, and decoded values only; not token/ngram hashing, PLE math, ANE execution, cold SSD latency, or whole-model quality."}
    table = None
    try:
        require(runner.is_file() and os.access(runner, os.X_OK), "Runner is missing or not executable")
        metadata = read_json(sample_dir / "metadata.json")
        verified = read_json(verification_path)
        require(verified.get("phase") == "complete", "Full-file download verification has not completed")
        require(Path(verified["model_directory"]).resolve() == model_dir, "Verification names a different model directory")
        matches = [item for item in verified["files"] if item["path"] == "ngram_table.bin"]
        require(len(matches) == 1, "Verification must contain exactly one ngram_table.bin record")
        verified_table = matches[0]
        require(verified_table["sha256"] == metadata["source_file_expected_sha256"], "Full-file hash record does not match saved sample source")
        table = TableReference(model_dir / "ngram_table.bin")
        require(table.file_bytes == verified_table["size"] == metadata["source_file_bytes"], "Current file size differs from verified source")
        require(table.offset == metadata["payload_offset"] and table.dimension == metadata["row_width"], "Table header differs from saved sample")
        require(table.scale == np.float32(metadata["scale"]), "Table scale differs from saved sample")
        report["verified_source"] = {"verification_file": str(verification_path), **verified_table,
                                     "note": "Prior complete-file SHA256 result reused; this script does not rehash 51.2 GB"}
        report["table"] = {"row_count": table.row_count, "dimension": table.dimension, "scale": float(table.scale), "payload_offset": table.offset}
        with np.load(sample_dir / "rows.npz", allow_pickle=False) as sample:
            sample_rows = [int(row) for row in sample["row_indices"]]
            saved_raw = sample["raw_codes"].copy()
            saved_values = sample["decoded_bf16_as_fp32"].copy()
        require(sample_rows == metadata["row_indices"] and len(sample_rows) == 16, "Expected the recorded 16-row fixture")
        require(saved_raw.shape == saved_values.shape == (16, table.dimension), "Invalid saved fixture shapes")
        require(saved_raw.dtype == np.uint8 and saved_values.dtype == np.float32, "Invalid saved fixture dtypes")
        generator = random.Random(args.seed)
        boundary_rows = sorted({0, 1, table.row_count // 2 - 1, table.row_count // 2, table.row_count - 2, table.row_count - 1})
        random_rows = generator.sample(range(table.row_count), min(args.random_rows, table.row_count))
        extra_rows = list(dict.fromkeys(boundary_rows + random_rows))
        with tempfile.TemporaryDirectory(prefix="ane-runner-ssd-check-") as temporary:
            directory = Path(temporary)
            record, actual_raw = verify_case(runner, model_dir, table, sample_rows, directory, "saved_16_rows", args.timeout, expected=saved_values)
            record["raw_sample_bytes_identical"] = bool(np.array_equal(actual_raw, saved_raw))
            record["passed"] = record["passed"] and record["raw_sample_bytes_identical"]
            report["cases"].append(record)
            record, _ = verify_case(runner, model_dir, table, extra_rows, directory, "random_and_boundaries", args.timeout)
            record["boundary_rows"] = boundary_rows
            record["random_rows"] = random_rows
            report["cases"].append(record)
            record, _ = verify_case(runner, model_dir, table, [table.row_count - 1, 0, table.row_count - 1, 1], directory, "order_and_duplicates", args.timeout)
            report["cases"].append(record)
            for name, invalid in (("negative_row_rejected", -1), ("past_end_rejected", table.row_count)):
                _, record = invoke(runner, model_dir, [invalid], directory, name, args.timeout, expect_rejection=True)
                report["cases"].append(record)
        report["passed"] = all(record["passed"] for record in report["cases"])
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    finally:
        if table is not None:
            report["reference_logical_bytes_read"] = table.bytes_read
            table.close()
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        output = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(output)
        print(output, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
