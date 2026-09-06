#!/usr/bin/env python3
"""Exercise real runner telemetry startup/ownership without loading a model."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--executable', required=True, type=Path)
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args()
binary = args.executable.resolve()
checks = {}
with tempfile.TemporaryDirectory(prefix='ane-telemetry-runner-') as temporary:
    root = Path(temporary)
    directory = root / 'exclusive'
    def attempt(index):
        command = [str(binary), 'probe-telemetry', '--telemetry-dir', str(directory),
                   '--seconds', '0.3', '--output', str(root / f'probe-{index}.json')]
        result = subprocess.run(command, text=True, capture_output=True, timeout=10)
        return index, result.returncode, result.stderr
    with ThreadPoolExecutor(max_workers=8) as pool:
        runs = list(pool.map(attempt, range(8)))
    winners = [index for index, code, _ in runs if code == 0]
    assert len(winners) == 1, runs
    checks['exactly_one_concurrent_directory_owner'] = True
    rows = [json.loads(line) for line in (directory / 'hardware.jsonl').read_text().splitlines()]
    report = json.loads((root / f'probe-{winners[0]}.json').read_text())['telemetry']
    assert rows[0]['type'] == 'metadata' and rows[1]['type'] == 'baseline'
    assert rows[1]['end_ns'] <= report['events'][0]['start_ns']
    assert report['events'][0]['phase'] == 'diagnostic_idle'
    checks['complete_hardware_baseline_precedes_first_phase'] = True
    assert report['collector_status'] == 'completed' and rows[-1]['type'] == 'summary'
    assert rows[-1]['stop_reason'] == 'signal'
    assert all(row.get('target_pid') == report['target_pid'] for row in rows)
    checks['collector_shutdown_and_identity_are_consistent'] = True
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}
    assert attempt(99)[1] != 0
    assert hashes == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}
    checks['retry_preserves_every_existing_evidence_file'] = True
result = {'passed': all(checks.values()), 'checks': checks, 'executable': str(binary),
          'executable_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
          'model_started': False, 'scope': 'Idle runner lifecycle and evidence ownership, not inference throughput.'}
with args.output.open('x') as handle:
    json.dump(result, handle, indent=2)
    handle.write('\n')
print(json.dumps(result))
