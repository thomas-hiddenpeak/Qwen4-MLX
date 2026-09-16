#!/usr/bin/env python3
"""Compile the current removable CoreAI request queue and run CPU-only checks."""
import argparse
import hashlib
import json
import pathlib
import platform
import subprocess
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[1]
SOURCES = ["Sources/CoreAIRunnerCLI/CoreAIRequestQueue.swift",
           "scripts/fixtures/CoreAIRequestQueueCheck.swift"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=pathlib.Path,
                        default=REPO / "results/coreai-service/request-queue-cpu/report.json")
    args = parser.parse_args()
    report = {"passed": False, "cases": []}
    try:
        with tempfile.TemporaryDirectory(prefix="coreai-request-queue-") as directory:
            executable = pathlib.Path(directory) / "queue-check"
            subprocess.run(["xcrun", "swiftc", "-swift-version", "6", "-target",
                            platform.machine() + "-apple-macosx26.2", *SOURCES,
                            "-o", str(executable)], cwd=REPO, check=True,
                           capture_output=True, text=True, timeout=60)
            run = subprocess.run([str(executable)], capture_output=True, text=True, timeout=30)
            report = json.loads(run.stdout)
            report["exit_code"] = run.returncode
            report["stderr"] = run.stderr
            report["passed"] = report["passed"] and run.returncode == 0
    except Exception as error:
        report["error"] = str(error)
        if isinstance(error, subprocess.CalledProcessError):
            report["compiler_stderr"] = error.stderr
    report["source_sha256"] = {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in SOURCES}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "cases": len(report["cases"]),
                      "report": str(args.output.resolve()), "error": report.get("error")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
