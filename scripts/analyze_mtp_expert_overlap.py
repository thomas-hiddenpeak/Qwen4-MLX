#!/usr/bin/env python3
"""Offline expert overlap for the fixed 11057-prompt/128-output/D2 experiment.

Requires one normal and one captured batchedScalarLinear request, exact original
AR golden IDs, all 48 layers, and complete routing records. File paths are CLI
arguments; no model is loaded. Logical reuse bounds are not physical traffic."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
sys.dont_write_bytecode = True
ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'Package.swift').is_file())
sys.path.insert(0, str(ROOT / 'scripts'))
from check_agent_workload import strict_json

PER_PROJECTION_BYTES = 2560 * 640 // 2 + 2 * (2560 * 640 // 64) * 2
PER_EXPERT_BYTES = 3 * PER_PROJECTION_BYTES
GEOMETRY = dict(expertCount=512, topK=10, hiddenSize=2560, intermediateSize=640, bits=4, groupSize=64)
COUNTERS = ('rounds', 'draftedTokens', 'acceptedDraftTokens', 'verifiedTokens', 'replayedTokens',
            'emittedTokens', 'prefillHistoryTokens', 'historyStartPosition', 'acceptanceHistogram')


def require(value, message):
    if not value:
        raise ValueError(message)


def integer(x):
    return type(x) is int and x >= 0


def finite(x):
    return type(x) in (int, float) and math.isfinite(x) and x >= 0


def near(a, b):
    return finite(a) and finite(b) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-9)


def identity(path):
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def overlap(ids):
    """Slot order is preserved in inputs; each token must select ten distinct experts."""
    require(type(ids) is list and len(ids) in (2, 3), 'Expected S2/S3 expert rows')
    for row in ids:
        require(type(row) is list and len(row) == 10 and all(integer(x) and x < 512 for x in row)
                and len(set(row)) == 10, 'Invalid, duplicate, or out-of-range expert ID')
    counts = Counter(x for row in ids for x in row)
    assignments = 10 * len(ids); unique = len(counts); redundant = assignments - unique
    fanout = {str(f): sum(n == f for n in counts.values()) for f in range(1, len(ids) + 1)}
    pairs = [{'token_rows': [i, j], 'intersection': len(set(ids[i]) & set(ids[j]))}
             for i in range(len(ids)) for j in range(i + 1, len(ids))]
    require(sum(int(f) * n for f, n in fanout.items()) == assignments and
            sum((int(f) - 1) * n for f, n in fanout.items()) == redundant and
            sum(x['intersection'] for x in pairs) == sum(n * (n - 1) // 2 for n in counts.values()),
            'Fanout/intersection accounting differs')
    return {'assignments': assignments, 'unique_experts': unique, 'redundant_assignments': redundant,
            'pair_intersections': pairs, 'fanout_histogram': fanout,
            'assignment_logical_bytes': assignments * PER_EXPERT_BYTES,
            'ideal_unique_logical_bytes': unique * PER_EXPERT_BYTES,
            'removable_logical_bytes_upper_bound': redundant * PER_EXPERT_BYTES,
            'removable_logical_fraction_upper_bound': redundant / assignments,
            'gate_up_only_removable_logical_bytes_upper_bound': redundant * 2 * PER_PROJECTION_BYTES,
            'gate_up_only_removable_fraction_of_all_routed_logical_bytes': (redundant * 2) / (assignments * 3),
            'logical_reuse_ratio': assignments / unique}


def summarize(rows):
    if not rows:
        return {'observed': False, 'records': 0, 'totals': None}
    n = sum(r['assignments'] for r in rows); u = sum(r['unique_experts'] for r in rows)
    f = Counter(); pairs = Counter(); patterns = Counter()
    for row in rows:
        f.update(row['fanout_histogram'])
        pairs.update(str(p['intersection']) for p in row['pair_intersections'])
        patterns[tuple(row['fanout_histogram'].get(str(i), 0) for i in (1, 2, 3))] += 1
    return {'observed': True, 'records': len(rows),
            'forwards': len({(r['repetition'], r['position']) for r in rows}),
            'totals': {'assignments': n, 'unique_expert_occurrences': u,
                'redundant_assignments': n - u, 'assignment_logical_bytes': n * PER_EXPERT_BYTES,
                'ideal_unique_logical_bytes': u * PER_EXPERT_BYTES,
                'removable_logical_bytes_upper_bound': (n - u) * PER_EXPERT_BYTES,
                'removable_logical_fraction_upper_bound': (n - u) / n, 'logical_reuse_ratio': n / u,
                'gate_up_only_removable_logical_bytes_upper_bound': (n - u) * 2 * PER_PROJECTION_BYTES,
                'gate_up_only_removable_fraction_of_all_routed_logical_bytes': ((n - u) * 2) / (n * 3)},
            'unique_expert_count_histogram': dict(sorted(Counter(str(r['unique_experts']) for r in rows).items(), key=lambda x: int(x[0]))),
            'fanout_expert_occurrence_histogram': dict(sorted(f.items())),
            'pair_intersection_count_histogram': dict(sorted(pairs.items(), key=lambda x: int(x[0]))),
            'fanout_patterns': [{'experts_at_fanout_1_2_3': list(pattern), 'records': count}
                               for pattern, count in sorted(patterns.items(), key=lambda x: (-x[1], x[0]))]}


def validate_geometry(data, model_directory, index_sha256):
    require(data['complete'] is True and data['model_directory'] == model_directory and
            data['index_sha256'] == index_sha256 and data['tensor_count'] == 432 and data['layer_count'] == 48 and
            len(data['records']) == 432, 'Actual header geometry identity/count differs')
    seen = set(); totals = Counter(); gate_up = Counter()
    for r in data['records']:
        layer, projection, part = r['layer'], r['projection'], r['part']
        require(integer(layer) and layer < 48 and projection in ('gate_proj', 'up_proj', 'down_proj') and
                part in ('weight', 'scales', 'biases'), 'Unknown actual header tensor')
        key = (layer, projection, part)
        require(key not in seen, 'Duplicate actual header tensor')
        seen.add(key)
        output, input_size = (2560, 640) if projection == 'down_proj' else (640, 2560)
        shape = [512, output, input_size // (8 if part == 'weight' else 64)]
        byte_width = 4 if part == 'weight' else 2
        expected_bytes = output * shape[2] * byte_width
        require(r['shape'] == shape and r['dtype'] == ('U32' if part == 'weight' else 'BF16') and
                type(r['bytes_per_expert']) is int and r['bytes_per_expert'] == expected_bytes,
                'Actual tensor shape/dtype/byte calculation differs')
        totals[layer] += expected_bytes
        if projection != 'down_proj': gate_up[layer] += expected_bytes
    require(all(totals[layer] == PER_EXPERT_BYTES and gate_up[layer] == 2 * PER_PROJECTION_BYTES for layer in range(48)) and
            data['routed_bytes_per_expert_per_layer'] == PER_EXPERT_BYTES and
            data['gate_up_bytes_per_expert_per_layer'] == 2 * PER_PROJECTION_BYTES,
            'Actual header totals differ')
    return {'validated': True, 'layers': 48, 'tensors': 432,
            'logical_bytes_per_expert': PER_EXPERT_BYTES, 'gate_up_bytes_per_expert': 2 * PER_PROJECTION_BYTES}


def analyze(normal, diagnostic, golden, weight_geometry=None):
    gt = golden['trials'][0]; prompt = gt['prompt_tokens']; output = gt['generated_token_ids'][:128]
    require(len(prompt) == 11057 and len(output) == 128, 'Golden is not complete fixed 11k/128')
    trials = []
    for name, report in [('normal', normal), ('diagnostic', diagnostic)]:
        require(report['requested_repetitions'] == 1 and len(report['trials']) == 1 and report['max_tokens'] == 128
                and report['context_limit'] == 16384, name + ': wrong request geometry')
        require(report['mtp_order'] == [2] and report['mtp_verification_order'] == ['batchedScalarLinear']
                and report['mtp_enabled'] is True and report['mtp_weights_loaded'] is True, name + ': wrong MTP mode')
        require(report['experimental_decode_async_every_layers'] == report['experimental_decode_async_submissions'] == 0
                and report['prefill_evaluate_every_layers'] == report['verification_evaluate_every_layers'] == 4
                and report['prefill_attention_mode'] == 'reference' and report['scheduler_environment'] == {}, name + ': wrong policy')
        require(report['profiler']['mode'] == 'disabled' and report['profiler']['stages'] == []
                and report['profiler']['droppedRecords'] == 0 and report['gpu_command_timing']['enabled'] is False,
                name + ': profiler/native trace must be disabled')
        t = report['trials'][0]; st = t['mtp_statistics']; cost = t['mtp_cost_summary']
        require(t['repetition'] == 0 and t['prompt_tokens'] == prompt and t['generated_token_ids'] == output
                and t['finish_reason'] == 'length' and t['final_state_offset'] == len(prompt) + len(output) - 1
                and t['qsa_active_layers'] == 12, name + ': full IDs/offset differ')
        require(t['mtp_depth'] == 2 and t['mtp_verification'] == 'batchedScalarLinear' and t['mtp_draft_history_limit'] == 1024
                and t['decode_mode'] == t['gdn_gemv_mode'] == t['prefill_accumulation'] == 'reference'
                and t['prefill_chunk'] == 416 and t['ssd_prefetch'] == 'nextChunk' and t['ssd_workers'] == 1
                and t['sampling'] == 'greedy' and t['wired_memory']['policy'] == 'disabled', name + ': trial config differs')
        require(all(integer(st[k]) for k in COUNTERS if k != 'acceptanceHistogram') and
                len(st['acceptanceHistogram']) == 5 and all(integer(x) for x in st['acceptanceHistogram']) and
                sum(st['acceptanceHistogram']) == st['rounds'] and
                sum(i * count for i, count in enumerate(st['acceptanceHistogram'])) == st['acceptedDraftTokens'], name + ': histogram/counters invalid')
        require(st['emittedTokens'] == 127 and st['replayedTokens'] == 0 and st['prefillHistoryTokens'] == 1024
                and st['historyStartPosition'] == 10033 and st['rounds'] <= st['draftedTokens'] <= 2 * st['rounds']
                and st['acceptedDraftTokens'] <= st['draftedTokens'], name + ': work counts differ')
        require(integer(t['decode_steps']) and t['decode_steps'] == len(t['decode_step_seconds']) and
                all(finite(x) and x > 0 for x in t['decode_step_seconds']), name + ': decode steps differ')
        secs = math.fsum(t['decode_step_seconds'])
        require(secs > 0 and near(secs, t['phase_metrics']['decode_seconds']) and near(127 / secs, t['decode_tokens_per_second'])
                and all(cost[k] is True for k in ('countersConsistent', 'acceptanceHistogramConsistent', 'componentsFitDecodeWindow'))
                and cost['committedDecodeTokens'] == 127 and cost['decodeSteps'] == t['decode_steps']
                and cost['speculativeRounds'] == st['rounds']
                and cost['targetOnlyDecodeSteps'] == t['decode_steps'] - st['rounds'] in (0, 1), name + ': decode denominator/cost differs')
        require(len(t['prefill_chunk_seconds']) == 28, name + ': prefill chunk count differs')
        trials.append(t)
    nt, dt = trials
    geometry_check = validate_geometry(weight_geometry, diagnostic['model_directory'],
        diagnostic['provenance']['model_metadata_sha256'].get('model.safetensors.index.json')) if weight_geometry is not None else None
    require({k: nt['mtp_statistics'][k] for k in COUNTERS} == {k: dt['mtp_statistics'][k] for k in COUNTERS}, 'Normal/diagnostic counters differ')
    require(normal['model_directory'] == diagnostic['model_directory'] and
            all(normal['provenance'][k] == diagnostic['provenance'][k] for k in ('executable_sha256', 'model_metadata_sha256'))
            and normal['provenance']['process_id'] != diagnostic['provenance']['process_id'], 'Process/model/binary identity differs')
    nc = normal.get('mtp_expert_routes')
    require(nc is None or (nc['enabled'] is False and nc.get('records', []) == [] and nc.get('droppedRecords', 0) == 0), 'Normal unexpectedly captured routes')
    capture = diagnostic['mtp_expert_routes']
    require(capture['enabled'] is True and capture['finished'] is True and capture['maximumRecords'] == 8192
            and capture['droppedRecords'] == 0 and finite(capture['readbackMilliseconds']), 'Incomplete/dropped capture')
    require(all(type(capture[k]) is int and capture[k] == v for k, v in GEOMETRY.items()), 'Logical-byte geometry differs')
    records = capture['records']
    require(0 < len(records) <= capture['maximumRecords'], 'No records or invalid record count')
    groups = defaultdict(list); rows = []
    for r in records:
        require(r['repetition'] == 0 and r['phase'] == 'verification' and integer(r['position'])
                and 11057 <= r['position'] < 11184 and type(r['tokenCount']) is int and r['tokenCount'] in (2, 3)
                and integer(r['layer']) and r['layer'] < 48 and len(r['expertIDs']) == r['tokenCount'], 'Invalid record identity/shape')
        groups[(r['position'], r['tokenCount'])].append(r['layer'])
        rows.append({'repetition': r['repetition'], 'position': r['position'], 'token_count': r['tokenCount'],
                     'layer': r['layer'], 'expert_ids': r['expertIDs'], **overlap(r['expertIDs'])})
    ordered = list(groups)
    require([p for p, s in ordered] == sorted({p for p, s in ordered}) and ordered[0][0] == 11057,
            'Repeated/reordered verification position')
    require(len(ordered) == dt['mtp_statistics']['rounds'] and
            all(layers == list(range(48)) for layers in groups.values()) and
            len(records) == 48 * len(ordered), 'Missing/duplicate/reordered layer or round')
    require([(r['position'], r['tokenCount']) for r in records] == [key for key in ordered for _ in range(48)],
            'Interleaved/reordered forward records')
    require(sum(s for p, s in ordered) == dt['mtp_statistics']['verifiedTokens'] - dt['mtp_cost_summary']['targetOnlyDecodeSteps'],
            'Recorded shapes do not cover speculative verification inputs')
    require(sum(s - 1 for p, s in ordered) == dt['mtp_statistics']['draftedTokens'], 'Recorded proposals differ from draft counter')
    all_layers = [{'layer': layer, **summarize([r for r in rows if r['layer'] == layer])} for layer in range(48)]
    return {'complete': True, 'all_correct': True, 'errors': [], 'geometry': GEOMETRY,
            'actual_header_geometry_check': geometry_check,
            'logical_bytes_per_projection': PER_PROJECTION_BYTES, 'logical_bytes_per_expert': PER_EXPERT_BYTES,
            'readback_ms_after_request_timing': capture['readbackMilliseconds'],
            'counters': {k: dt['mtp_statistics'][k] for k in COUNTERS},
            'target_only_decode_steps_excluded': dt['mtp_cost_summary']['targetOnlyDecodeSteps'],
            'timings_separate_no_speedup_claim': {name: {'phase_metrics': t['phase_metrics'], 'mtp_cost_summary': t['mtp_cost_summary']}
                                                for name, t in [('normal', nt), ('diagnostic', dt)]},
            'overall': summarize(rows),
            'by_shape': [{'token_count': s, **summarize([r for r in rows if r['token_count'] == s])} for s in (2, 3)],
            'layer_ranking': sorted(all_layers, key=lambda x: (-x['totals']['removable_logical_fraction_upper_bound'], x['layer'])),
            'by_layer_shape': [{'layer': layer, 'token_count': s, **summarize([r for r in rows if r['layer'] == layer and r['token_count'] == s])}
                               for layer in range(48) for s in (2, 3)],
            'records': rows,
            'scope': ['Only within one verification forward and layer; no cross-layer or cross-round deduplication.',
                      'Routed experts only. The dense shared expert, router, activations, GDN/attention and head are excluded.',
                      'Logical weight-footprint difference is an ideal reuse upper bound, not actual DRAM traffic or a speedup estimate.',
                      'Caches may already reuse weights; grouping, tiling, refetch and changed arithmetic can offset any opportunity.',
                      'Collector retains route tensors and reads after request timing; timings are reported separately without a benefit ratio.',
                      'A missing shape is unobserved, not zero overlap; all actual layer/forward records are retained.']}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('normal', 'diagnostic', 'golden', 'output'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--weight-geometry', type=Path)
    args = p.parse_args(argv)
    require(not args.output.exists(), 'Refuse to overwrite analysis')
    require(args.output.resolve() not in [getattr(args, k).resolve() for k in ('normal', 'diagnostic', 'golden')], 'Output is input')
    result = {'schema': 'mtp-expert-overlap-v1', 'complete': False, 'all_correct': False, 'errors': [],
              'analysis_source': identity(Path(__file__)), 'inputs': {}}
    try:
        loaded = {}
        for k in ('normal', 'diagnostic', 'golden'):
            path = getattr(args, k); result['inputs'][k] = identity(path); loaded[k] = strict_json(path.read_text())
        if args.weight_geometry is not None:
            result['inputs']['weight_geometry'] = identity(args.weight_geometry)
            loaded['weight_geometry'] = strict_json(args.weight_geometry.read_text())
        result.update(analyze(**loaded))
    except (OSError, ValueError, TypeError, KeyError, IndexError, ZeroDivisionError) as error:
        result['errors'].append(f'{type(error).__name__}: {error}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({'complete': result['complete'], 'errors': result['errors'], 'output': str(args.output)}))
    return 0 if result['complete'] and result['all_correct'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
