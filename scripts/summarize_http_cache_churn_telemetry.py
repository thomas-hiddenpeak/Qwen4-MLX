#!/usr/bin/env python3
"""Read saved files only; summarize separate telemetry and wrapper sample windows."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


def integer(value):
    return type(value) is int and value >= 0


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def stats(values):
    if not values:
        return {'count': 0}
    ordered = sorted(values)
    n = len(values)
    return {'count': n, 'first': values[0], 'last': values[-1],
            'last_minus_first': values[-1] - values[0], 'min': ordered[0], 'max': ordered[-1],
            'p50': ordered[math.ceil(n * .5) - 1], 'p95': ordered[math.ceil(n * .95) - 1]}


def analyze(telemetry_path, process_path, lifecycle_path, expected_pid):
    result = {'schema': 'qwen-partial-process-telemetry-v1', 'target_pid': expected_pid,
              'analyzer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'inputs': {}, 'issues': {}, 'exclusions': {}, 'full_two_hour_footprint_coverage': False,
              'two_hour_acceptance_evaluated': False,
              'definitions': [
                  'Telemetry window is only the valid same-PID/same-start libproc samples actually captured.',
                  'physical_footprint_bytes is ri_phys_footprint; resident_bytes is ri_resident_size. Neither is the logical state ledger.',
                  'Discrete sampled maxima and endpoint changes are descriptive, not true instantaneous peaks or proof of absence of leaks.',
                  'Process disk deltas are libproc accounting, not archive logical bytes, NAND bytes, or physical DRAM bandwidth.',
                  'FD/ps RSS belong to the separate wrapper elapsed clock. No cross-clock partial-window alignment is assumed.',
                  'Wrapper PID ownership comes from the original Popen lifetime and health guard; those records have no libproc start identity.',
                  'IOReport bins, device/global sums and thermal/GPU state data are not converted to process bandwidth.',
              ]}

    def issue(code, detail):
        item = result['issues'].setdefault(code, {'count': 0, 'examples': []})
        item['count'] += 1
        if len(item['examples']) < 5:
            item['examples'].append(str(detail)[:250])

    def rows(path, limit=128 * 1024 * 1024):
        path = Path(path)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            size = os.fstat(stream.fileno()).st_size
            if size > limit:
                raise ValueError(f'{path}: snapshot exceeds {limit} bytes')
            info = {'snapshot_bytes': size, 'complete_lines': 0, 'partial_tail_bytes': 0}
            result['inputs'][str(path)] = info
            left = size
            while left:
                line = stream.readline(min(left, 4 * 1024 * 1024 + 1))
                if not line:
                    raise ValueError(f'{path}: shortened during read')
                left -= len(line)
                digest.update(line)
                if len(line) > 4 * 1024 * 1024:
                    raise ValueError(f'{path}: oversized line')
                if not line.endswith(b'\n'):
                    info['partial_tail_bytes'] = len(line)
                    break
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f'{path}: expected JSON object')
                info['complete_lines'] += 1
                yield value
            info['snapshot_sha256'] = digest.hexdigest()

    lifecycle_path = Path(lifecycle_path)
    with lifecycle_path.open('rb') as stream:
        raw = stream.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError('lifecycle exceeds 1 MiB')
    lifecycle = json.loads(raw)
    result['inputs'][str(lifecycle_path)] = {'snapshot_bytes': len(raw), 'snapshot_sha256': hashlib.sha256(raw).hexdigest()}
    services = lifecycle.get('services', [])
    if len(services) != 1 or services[0].get('pid') != expected_pid:
        issue('wrapper_owned_pid_mismatch', expected_pid)
    result['wrapper_lifecycle'] = {k: lifecycle.get(k) for k in ('complete', 'passed', 'sampling', 'cleanup')}

    metadata = None
    summary = None
    samples = []
    previous = None
    counts = Counter()
    excluded = Counter()
    segment = 0
    expected_id = 0
    for row in rows(telemetry_path):
        kind = row.get('type')
        counts[kind] += 1
        if kind == 'metadata':
            if metadata is not None:
                issue('duplicate_metadata', counts[kind])
            metadata = row
            if row.get('target_pid') != expected_pid or not integer(row.get('target_start_abstime')) or row['target_start_abstime'] == 0 or row.get('target_identity_error') is not None:
                issue('invalid_metadata_identity', expected_pid)
            if row.get('clock') != 'mach_absolute_time_nanoseconds':
                issue('unexpected_clock', row.get('clock'))
            continue
        if kind == 'summary':
            summary = row
            if counts[kind] != 1 or row.get('target_pid') != expected_pid or row.get('target_signalled_by_sampler') is not False:
                issue('invalid_sidecar_summary', row)
            continue
        if kind not in ('baseline', 'sample'):
            issue('unexpected_record_type', kind)
            continue
        if len(samples) >= 2048:
            raise ValueError('valid telemetry sample bound exceeded')
        sample_id = row.get('sample_id')
        if not integer(sample_id) or sample_id != expected_id or (sample_id == 0) != (kind == 'baseline'):
            issue('sample_id_gap_or_duplicate', sample_id)
            previous = None
            segment += 1
        expected_id = sample_id + 1 if integer(sample_id) else expected_id + 1
        if not integer(sample_id):
            continue
        process = row.get('process') or {}
        usage = process.get('rusage')
        valid = (metadata is not None and row.get('target_pid') == expected_pid
                 and row.get('clock') == 'mach_absolute_time_nanoseconds'
                 and process.get('error') is None and isinstance(usage, dict)
                 and usage.get('pid') == expected_pid
                 and usage.get('start_abstime') == metadata.get('target_start_abstime')
                 and usage.get('exit_abstime') == 0)
        if not valid:
            reason = row.get('stop_reason') or 'invalid_identity_or_process_sample'
            excluded[reason] += 1
            if reason not in ('target_unavailable', 'target_exited'):
                issue('invalid_process_sample', [sample_id, reason])
            previous = None
            segment += 1
            continue
        keys = ('physical_footprint_bytes', 'resident_bytes', 'disk_read_bytes_cumulative', 'disk_write_bytes_cumulative')
        if not all(integer(usage.get(k)) for k in keys):
            issue('invalid_process_measurement', sample_id)
            previous = None
            segment += 1
            continue
        lo, hi = process.get('read_start_ns'), process.get('read_end_ns')
        if not integer(lo) or not integer(hi) or lo > hi:
            issue('invalid_process_read_window', sample_id)
            previous = None
            segment += 1
            continue
        current = {'sample_id': sample_id, 'segment': segment, 'read_start_ns': lo,
                   'read_end_ns': hi, **{k: usage[k] for k in keys}}
        if previous is not None:
            if lo <= previous['read_end_ns']:
                issue('non_increasing_process_time', sample_id)
                previous = None
                segment += 1
                continue
            for direction in ('read', 'write'):
                key = f'disk_{direction}_bytes_cumulative'
                delta = usage[key] - previous[key]
                if delta < 0 or row.get(f'process_disk_{direction}_bytes_delta') != delta:
                    issue('process_io_delta_mismatch', [sample_id, direction])
        samples.append(current)
        previous = current
    if metadata is None or counts['baseline'] != 1 or len(samples) < 2:
        issue('insufficient_telemetry_samples', len(samples))
    if summary is not None and summary.get('samples_written') != counts['sample']:
        issue('summary_sample_count_mismatch', summary.get('samples_written'))
    if summary is not None and result['inputs'][str(telemetry_path)].get('partial_tail_bytes'):
        issue('partial_tail_after_summary', telemetry_path)
    result['telemetry_counts'] = dict(counts)
    result['exclusions'] = dict(excluded)
    result['sidecar_complete'] = summary is not None
    result['sidecar_summary'] = summary
    result['telemetry_metadata'] = {k: (metadata or {}).get(k) for k in ('created_utc', 'created_ns', 'target_start_abstime', 'target_start_ns', 'interval_ms', 'max_samples')}
    groups = []
    for group_id in sorted({r['segment'] for r in samples}):
        group = [r for r in samples if r['segment'] == group_id]
        first, last = group[0], group[-1]
        groups.append({'sample_count': len(group), 'first_sample_id': first['sample_id'], 'last_sample_id': last['sample_id'],
                       'read_start_ns': first['read_start_ns'], 'read_end_ns': last['read_end_ns'],
                       'endpoint_span_seconds': (last['read_end_ns'] - first['read_end_ns']) / 1e9,
                       'physical_footprint_bytes': stats([r['physical_footprint_bytes'] for r in group]),
                       'resident_rusage_bytes': stats([r['resident_bytes'] for r in group]),
                       'process_io_endpoint_delta_bytes': {direction: last[f'disk_{direction}_bytes_cumulative'] - first[f'disk_{direction}_bytes_cumulative'] for direction in ('read', 'write')}})
    result['telemetry_valid_contiguous_windows'] = groups
    wrapper = []
    for row in rows(process_path, 16 * 1024 * 1024):
        if len(wrapper) >= 2048:
            raise ValueError('wrapper sample bound exceeded')
        if row.get('pid') != expected_pid or row.get('observation_error') or not number(row.get('elapsed_seconds')) or not integer(row.get('rss_bytes')) or not integer(row.get('numeric_fds')):
            issue('invalid_wrapper_sample', row.get('elapsed_seconds'))
            continue
        if wrapper and row['elapsed_seconds'] <= wrapper[-1]['elapsed_seconds']:
            issue('non_increasing_wrapper_time', row['elapsed_seconds'])
        wrapper.append({k: row[k] for k in ('elapsed_seconds', 'rss_bytes', 'numeric_fds')})
    if len(wrapper) < 2:
        issue('insufficient_wrapper_samples', len(wrapper))
    sampling = lifecycle.get('sampling') or {}
    if sampling.get('errors', 0):
        issue('wrapper_sampling_failed', sampling.get('error'))
    if sampling.get('completed') is True and sampling.get('samples') != len(wrapper):
        issue('wrapper_completed_sample_count_mismatch', sampling.get('samples'))
    result['wrapper_separate_window'] = {
        'clock': 'wrapper sample-thread elapsed_seconds; not aligned to telemetry mach clock',
        'first_elapsed_seconds': wrapper[0]['elapsed_seconds'] if wrapper else None,
        'last_elapsed_seconds': wrapper[-1]['elapsed_seconds'] if wrapper else None,
        'rss_ps_bytes': stats([r['rss_bytes'] for r in wrapper]),
        'numeric_fds': stats([r['numeric_fds'] for r in wrapper]),
    }
    result['valid_sample_summary_available'] = len(samples) >= 2 and len(wrapper) >= 2
    result['sample_contract_passed'] = result['valid_sample_summary_available'] and not result['issues']
    return result


def self_test():
    import copy
    with tempfile.TemporaryDirectory(prefix='partial-telemetry-cpu-') as directory:
        p = Path(directory)
        metadata = {'type': 'metadata', 'clock': 'mach_absolute_time_nanoseconds', 'target_pid': 123, 'target_start_abstime': 456, 'target_identity_error': None}
        samples = []
        for i in range(2):
            usage = {'pid': 123, 'start_abstime': 456, 'exit_abstime': 0, 'physical_footprint_bytes': 900 + i, 'resident_bytes': 300 + i, 'disk_read_bytes_cumulative': 10 + i * 5, 'disk_write_bytes_cumulative': 20 + i * 7}
            samples.append({'type': 'baseline' if i == 0 else 'sample', 'sample_id': i, 'clock': 'mach_absolute_time_nanoseconds', 'target_pid': 123, 'process': {'error': None, 'rusage': usage, 'read_start_ns': 100 + i * 100, 'read_end_ns': 110 + i * 100}, 'process_disk_read_bytes_delta': None if i == 0 else 5, 'process_disk_write_bytes_delta': None if i == 0 else 7})
        process = [{'pid': 123, 'elapsed_seconds': i * 30., 'rss_bytes': 300 + i, 'numeric_fds': 11 + i} for i in range(2)]
        lifecycle = {'services': [{'pid': 123}], 'sampling': {'completed': True, 'samples': 2, 'errors': 0}}
        def run(records, process_rows=process):
            (p/'t').write_text(''.join(json.dumps(r) + '\n' for r in records))
            (p/'p').write_text(''.join(json.dumps(r) + '\n' for r in process_rows))
            (p/'l').write_text(json.dumps(lifecycle))
            return analyze(p/'t', p/'p', p/'l', 123)
        baseline = [metadata] + samples
        r = run(baseline)
        assert r['sample_contract_passed'] and not r['sidecar_complete']
        assert r['telemetry_valid_contiguous_windows'][0]['process_io_endpoint_delta_bytes'] == {'read': 5, 'write': 7}
        assert r['wrapper_separate_window']['numeric_fds']['last_minus_first'] == 1
        for mutate, expected in (
            (lambda x: x[2]['process']['rusage'].__setitem__('start_abstime', 999), 'invalid_process_sample'),
            (lambda x: x[2].__setitem__('process_disk_read_bytes_delta', 99), 'process_io_delta_mismatch'),
            (lambda x: x[2].__setitem__('sample_id', 7), 'sample_id_gap_or_duplicate'),
        ):
            bad = copy.deepcopy(baseline); mutate(bad)
            assert expected in run(bad)['issues']
        stopped = baseline + [{'type': 'sample', 'sample_id': 2, 'clock': 'mach_absolute_time_nanoseconds', 'target_pid': 123, 'stop_reason': 'target_unavailable', 'process': {'error': {'code': 3}, 'rusage': None}}, {'type': 'summary', 'target_pid': 123, 'samples_written': 2, 'target_signalled_by_sampler': False, 'stop_reason': 'target_unavailable'}]
        r = run(stopped)
        assert r['sample_contract_passed'] and r['sidecar_complete'] and r['exclusions']['target_unavailable'] == 1
        bad_process = copy.deepcopy(process); bad_process[1]['pid'] = 999
        assert 'invalid_wrapper_sample' in run(baseline, bad_process)['issues']
    print(json.dumps({'self_test': 'passed', 'controls': 6, 'scope': 'synthetic saved CPU files only'}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--telemetry', type=Path)
    parser.add_argument('--process', type=Path)
    parser.add_argument('--lifecycle', type=Path)
    parser.add_argument('--expected-pid', type=int)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        if any((args.telemetry, args.process, args.lifecycle, args.expected_pid, args.output)):
            parser.error('--self-test cannot be combined with inputs')
        self_test(); return 0
    if not all((args.telemetry, args.process, args.lifecycle, args.expected_pid, args.output)) or args.expected_pid < 1:
        parser.error('all input/output paths and a positive --expected-pid are required')
    result = analyze(args.telemetry, args.process, args.lifecycle, args.expected_pid)
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({k: result[k] for k in ('sample_contract_passed', 'sidecar_complete', 'full_two_hour_footprint_coverage', 'issues')}))
    return 0 if result['sample_contract_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
