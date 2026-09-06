"""Describe isolated real-weight GEMV runs; logical rates are not DRAM counters."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
from analyze_gpu_command_timing import union_ns


def analyze(report, native):
    if not report['passed'] or not native['complete'] or native['errors'] or native['dropped_buffers'] or native['pending_buffers']:
        raise ValueError('Probe correctness or native timing completeness failed')
    if report['provenance']['process_id'] != native['process_id']:
        raise ValueError('Mismatched process IDs')
    if report['gpu_command_timing']['hook_version'] != native['version'] or not report['gpu_command_timing']['finished']:
        raise ValueError('Hook version/lifecycle mismatch')
    records=native['records']
    if len(records) != native['recorded_buffers'] or len(records) != native['completed_buffers']:
        raise ValueError('Native buffer counts differ')
    if any(r['status'] != 4 or (r['buffer_ops'] > 0 and not (r['gpu_timestamp_valid'] and r['host_clock_bracket_valid'])) for r in records):
        raise ValueError('Invalid buffer status or timestamp')
    groups=[]
    for c in report['cases']:
        if not all(x['finite'] and x['bitwise_equal'] for x in c['comparisons']):
            raise ValueError('Matrix repeatability gate failed')
        rows=[]
        for it in c['measured_iterations']:
            lo,hi=it['start_ns'],it['evaluation_end_ns']
            relevant=[r for r in records if r['gpu_timestamp_valid'] and r['gpu_start_ns'] < hi and r['gpu_end_ns'] > lo]
            if not relevant or any(r['gpu_start_ns'] < lo or r['gpu_end_ns'] > hi for r in relevant):
                raise ValueError('Missing or cross-window GPU buffer in matvec probe')
            intervals=[(r['gpu_start_ns'],r['gpu_end_ns']) for r in relevant]
            duration=union_ns(intervals)
            if duration <= 0 or not any(r['buffer_ops'] > 0 for r in relevant):
                raise ValueError('Matvec did not execute a positive GPU workload')
            rows.append(dict(it,gpu_buffer_span_union_ns=duration,buffer_count=len(relevant),
                             logical_weight_bytes_per_gpu_span_gbps=it['logical_weight_bytes']/duration))
        total_bytes=sum(r['logical_weight_bytes'] for r in rows)
        gpu_ns=sum(r['gpu_buffer_span_union_ns'] for r in rows)
        groups.append({'name':c['name'],'dtype':c['dtype'],'iterations':len(rows),
            'unique_weight_slots':c['unique_weight_slots'],'source_weight_bytes':c['source_weight_bytes'],
            'logical_bytes_per_matvec':rows[0]['logical_weight_bytes'],
            'median_gpu_buffer_span_ms':statistics.median(r['gpu_buffer_span_union_ns'] for r in rows)/1e6,
            'median_cpu_wall_ms':statistics.median(r['cpu_wall_seconds'] for r in rows)*1e3,
            'aggregate_logical_weight_bytes_per_gpu_span_gbps':total_bytes/gpu_ns,
            'first_quarter_median_gpu_ms':statistics.median(r['gpu_buffer_span_union_ns'] for r in rows[:max(1,len(rows)//4)])/1e6,
            'last_quarter_median_gpu_ms':statistics.median(r['gpu_buffer_span_union_ns'] for r in rows[-max(1,len(rows)//4):])/1e6,
            'physical_dram_bandwidth_gbps':None,'measured_iterations':rows})
    return {'schema_version':1,'process_id':native['process_id'],'native_complete':True,'cases':groups,
            'notes':['Real BF16 weights with deterministic synthetic inputs, 8 warmups then repeated new GEMV graphs.',
                     'Logical bytes divided by command-buffer GPU span is not measured DRAM bandwidth.',
                     'Four GDN matrices rotate; head repeats one larger matrix. Cache residency, clock state and isolated dispatch gaps can differ from full decode.',
                     'No full-model speedup is predicted by this microbenchmark.']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True);p.add_argument('--commands',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    try:
        if args.output.resolve() in (args.report.resolve(),args.commands.resolve()):raise ValueError('Output would overwrite input')
        a=args.report.read_bytes();b=args.commands.read_bytes();result=analyze(json.loads(a),json.loads(b))
        result['sources']=[{'path':str(path.resolve()),'sha256':hashlib.sha256(raw).hexdigest()} for path,raw in [(args.report,a),(args.commands,b)]]
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    except (KeyError,ValueError,TypeError,OSError) as e:p.exit(2,f'error: {e}\n')


if __name__ == '__main__':main()
