"""Align actual MLX command-buffer spans to same-process CPU token windows.

Command buffers can include stalls: span coverage is neither shader utilization
nor measured DRAM bandwidth. CPU forward/evaluate intervals can overlap GPU work.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def union_ns(intervals):
    total = 0
    end = None
    for lo, hi in sorted(intervals):
        if hi < lo:
            raise ValueError('Reversed GPU interval')
        if end is None or lo > end:
            total += hi - lo
            end = hi
        elif hi > end:
            total += hi - end
            end = hi
    return total


def analyze(generation, commands, skip_first_trials=1):
    timing = generation['gpu_command_timing']
    if not timing['enabled'] or not timing['finished']:
        raise ValueError('Generation did not finish command timing')
    if generation['provenance']['process_id'] != commands['process_id']:
        raise ValueError('Process IDs differ; refusing cross-process alignment')
    if timing['hook_version'] != commands['version']:
        raise ValueError('Hook versions differ')
    errors = list(commands['errors'])
    complete = commands['complete'] and not errors and commands['dropped_buffers'] == 0 and commands['pending_buffers'] == 0
    records = commands['records']
    if len(records) != commands['recorded_buffers'] or commands['completed_buffers'] != len(records):
        complete = False
        errors.append('Native record counts do not match completion counts')
    valid = []
    for r in records:
        if r['status'] != 4:
            complete = False
            errors.append(f"Command buffer {r['sequence']} did not complete successfully")
        if r['gpu_timestamp_valid'] and r['host_clock_bracket_valid']:
            lo, hi = r['gpu_start_ns'], r['gpu_end_ns']
            if not (isinstance(lo, int) and isinstance(hi, int) and 0 < lo <= hi):
                raise ValueError('Invalid native GPU timestamp')
            valid.append((lo, hi, r))
        elif r['buffer_ops'] > 0:
            complete = False
            errors.append(f"Nonempty command buffer {r['sequence']} has invalid timing/clock bracket")
    valid.sort(key=lambda x: x[0])
    steps = []
    for s in timing['steps']:
        lo, middle, hi = s['start_ns'], s['forward_end_ns'], s['evaluation_end_ns']
        if not 0 < lo <= middle <= hi:
            raise ValueError('Invalid CPU step window')
        overlapping = [(a, b, r) for a, b, r in valid if a < hi and b > lo]
        clipped = [(max(a, lo), min(b, hi)) for a, b, _ in overlapping]
        observed = union_ns(clipped)
        # Preserve observed spans even for incomplete evidence, but suppress
        # any complete-coverage or gap inference in that case.
        graph_boundary = s.get('graph_boundary_available', True)
        if not isinstance(graph_boundary, bool):
            raise ValueError('Invalid graph-boundary availability flag')
        record = dict(s, step_wall_ms=(hi-lo)/1e6,
                      forward_wall_ms=(middle-lo)/1e6 if graph_boundary else None,
                      evaluation_and_readback_wall_ms=(hi-middle)/1e6 if graph_boundary else None,
                      observed_gpu_buffer_span_union_ms=observed/1e6,
                      overlapping_command_buffer_count=len(overlapping),
                      contained_command_buffer_count=sum(a >= lo and b <= hi for a,b,_ in overlapping),
                      command_buffer_span_coverage_fraction=observed/(hi-lo) if complete and hi > lo else None,
                      outside_observed_buffer_spans_ms=((hi-lo)-observed)/1e6 if complete else None,
                      physical_dram_bandwidth_gbps=None)
        # Compare the submission time of each next buffer with the preceding
        # buffer's end. If submission was already done, the gap is not simply
        # waiting for this host commit; it can still include driver/queue work.
        gaps = []
        cursor = lo
        for a,b,r in overlapping:
            gap_end = min(a, hi)
            if gap_end > cursor:
                gap = {'start_ns': cursor, 'end_ns': gap_end, 'milliseconds': (gap_end-cursor)/1e6,
                       'next_cpu_commit_ns': r['cpu_commit_ns'],
                       'next_buffer_already_committed_at_gap_start': r['cpu_commit_ns'] <= cursor}
                gaps.append(gap)
            cursor = max(cursor, min(b,hi))
        record['observed_gaps_before_next_buffer'] = gaps if complete else None
        steps.append(record)
    retained = [s for s in steps if s['phase'] == 'decode' and s['repetition'] >= skip_first_trials]
    if not retained:
        raise ValueError('No decode steps remain after excluding warmup trials')
    groups = []
    for rep in sorted({s['repetition'] for s in retained}):
        rows = [s for s in retained if s['repetition'] == rep]
        wall = math.fsum(s['step_wall_ms'] for s in rows)
        observed = math.fsum(s['observed_gpu_buffer_span_union_ms'] for s in rows)
        phase_start = min(s['start_ns'] for s in rows)
        phase_end = max(s['evaluation_end_ns'] for s in rows)
        phase_wall = (phase_end-phase_start)/1e6
        phase_observed = union_ns([(max(a,phase_start),min(b,phase_end)) for a,b,_ in valid
                                  if a < phase_end and b > phase_start])/1e6
        gaps = [gap for s in rows for gap in (s['observed_gaps_before_next_buffer'] or [])]
        groups.append({'repetition': rep, 'decode_steps': len(rows),
            'step_wall_sum_ms': wall, 'observed_gpu_buffer_span_union_sum_ms': observed,
            'continuous_decode_window_ms': phase_wall,
            'continuous_observed_gpu_buffer_span_union_ms': phase_observed,
            'continuous_buffer_span_coverage_fraction': phase_observed/phase_wall if complete else None,
            'time_between_recorded_steps_ms': phase_wall-wall,
            'command_buffer_span_coverage_fraction': observed/wall if complete else None,
            'outside_observed_buffer_spans_sum_ms': wall-observed if complete else None,
            'median_forward_wall_ms': statistics.median(s['forward_wall_ms'] for s in rows)
                if all(s['forward_wall_ms'] is not None for s in rows) else None,
            'median_evaluation_and_readback_wall_ms': statistics.median(s['evaluation_and_readback_wall_ms'] for s in rows)
                if all(s['evaluation_and_readback_wall_ms'] is not None for s in rows) else None,
            'median_step_wall_ms': statistics.median(s['step_wall_ms'] for s in rows),
            'median_command_buffer_count': statistics.median(s['overlapping_command_buffer_count'] for s in rows),
            'gap_ms_with_next_buffer_already_committed': math.fsum(g['milliseconds'] for g in gaps if g['next_buffer_already_committed_at_gap_start']) if complete else None,
            'gap_ms_with_next_buffer_not_yet_committed': math.fsum(g['milliseconds'] for g in gaps if not g['next_buffer_already_committed_at_gap_start']) if complete else None})
    return {'schema_version':1, 'complete': bool(complete), 'errors': errors,
            'process_id': commands['process_id'], 'native_buffers': len(records),
            'valid_timestamp_buffers': len(valid), 'skip_first_trials': skip_first_trials,
            'decode_groups': groups, 'steps': steps, 'physical_dram_bandwidth_gbps': None,
            'notes': [
                'GPU intervals come from MLX command buffers; union removes overlap and clips spans to each CPU step.',
                'Command-buffer spans can include memory stalls and execution gaps; coverage is not shader utilization.',
                'Time outside command-buffer spans can include host building/encoding, driver scheduling, completion and unrelated system work; it is not all removable overhead.',
                'CPU forward and evaluation intervals can overlap GPU spans and must not be added to them.',
                'MTP round markers explicitly lack a graph boundary; their forward/evaluation split is null. Whole-round spans remain available.',
                'A gap with the next buffer already committed is not caused by waiting for that particular commit; it is not a unique attribution to the GPU or driver.',
                'buffer_ops is MLX scheduling bookkeeping; buffer_sizes_elements is not bytes or measured traffic.',
                'This diagnostic changes completion handlers and adds clocks; uninstrumented throughput must be measured separately.']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--generation', type=Path, required=True)
    p.add_argument('--commands', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--skip-first-trials', type=int, default=1)
    args=p.parse_args()
    try:
        if args.skip_first_trials < 0 or args.output.resolve() in (args.generation.resolve(),args.commands.resolve()):
            raise ValueError('Invalid warmup count or output path')
        g=args.generation.read_bytes(); c=args.commands.read_bytes()
        result=analyze(json.loads(g),json.loads(c),args.skip_first_trials)
        result['sources']=[{'path':str(path.resolve()),'sha256':hashlib.sha256(raw).hexdigest()} for path,raw in [(args.generation,g),(args.commands,c)]]
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    except (KeyError,TypeError,ValueError,OSError) as e:
        p.exit(2,f'error: {e}\n')


if __name__ == '__main__':
    main()
