#!/usr/bin/env python3
"""Foreground reference262K profile; adopt only after CLI and HTTP acceptance."""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--port', type=int, default=11236)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535 or args.port == 11235:
        parser.error('Use a port in1024...65535 other than the reference service11235')
    model = args.model_dir.expanduser().resolve()
    if not model.is_dir() or not (model / 'config.json').is_file():
        parser.error('--model-dir must identify an existing model directory with config.json')
    repo = next((p for p in Path(__file__).resolve().parents
                 if (p / 'Package.swift').is_file() and (p / 'Sources/ANERunnerCLI').is_dir()), None)
    if repo is None:
        parser.error('Place this script inside the ane-runner repository')
    runner = repo / '.build/release/ane-runner'
    if not runner.is_file() or not os.access(runner, os.X_OK):
        parser.error('Build the validated release executable before using this profile')
    # No passthrough tail: callers cannot silently override this profile.
    # SSD and physical paging stay off by omission. The262K HTTP policy rejects MTP.
    command = [str(runner), 'serve-gpu', '--model-dir', str(model), '--port', str(args.port),
               '--context-limit', '262144', '--max-reserved-tokens', '262144',
               '--max-resident-sequences', '1', '--max-body-bytes', '8388608',
               '--connection-deadline-seconds', '3600', '--state-budget-bytes', '25769803776',
               '--prefix-cache-bytes', '8589934592', '--prefix-cache-entries', '2',
               '--prefill-attention', 'reference', '--kv-append-mode', 'reference']
    os.execv(str(runner), command)


if __name__ == '__main__':
    main()
