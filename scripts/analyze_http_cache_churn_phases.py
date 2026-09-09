#!/usr/bin/env python3
"""Supplement a passed churn audit with phase metrics; saved files only, no server access.

Run from the same repository working directory used by analyze-churn.py, so its
relative input paths resolve identically. The old audit remains the authority for
oracle, resource and lifecycle acceptance. This supplement rechecks its exact
events/server byte hashes, then independently joins IDs and checks phase fields.
Example: python3 -B results/kv-night-churn-prep/analyze-churn-phases.py \
  --audit results/RUN/analysis.json --events results/RUN/churn.events.ndjson \
  --server-log results/RUN/server.log --output results/RUN/phase-analysis.json
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import tempfile

SCHEMA = 'qwen-churn-phase-audit-v1'
MAX_LINE = 1_048_576
MAX_PROFILES = 64
EPSILON_SECONDS = 1e-9
TERMINAL_SECONDS = (
    'prefill_seconds', 'prefill_active_seconds', 'prefill_suspension_seconds',
    'model_first_token_ready_seconds', 'decode_seconds', 'decode_service_seconds',
    'decode_suspension_seconds', 'handoff_wait_seconds', 'handoff_consume_seconds',
    'scheduler_elapsed_seconds',
)
DURATION_NAMES = ('wall_seconds', 'first_content_wall_seconds') + tuple(
    'decode_round_compute_seconds' if name == 'decode_seconds' else name
    for name in TERMINAL_SECONDS)
MEAN_NAMES = ('round_compute', 'service_active')


def integer(value):
    return type(value) is int and value >= 0


def duration(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def fingerprint(data):
    return {'snapshot_bytes': len(data), 'snapshot_sha256': hashlib.sha256(data).hexdigest()}


def analyze(audit_path, events_path, server_path):
    audit_path, events_path, server_path = map(Path, (audit_path, events_path, server_path))
    with audit_path.open('rb') as stream:
        audit_bytes = stream.read(8_388_609)
    if len(audit_bytes) > 8_388_608:
        raise ValueError('Base audit exceeds the 8 MiB document limit')
    audit = json.loads(audit_bytes)
    if not isinstance(audit, dict):
        raise ValueError('Base audit must be an object')
    report = {
        'schema': SCHEMA, 'complete': audit.get('complete') is True, 'passed': False,
        'run_id': audit.get('run_id'), 'server_pid': audit.get('server_pid'),
        'base_audit': {'path': str(audit_path.resolve()), **fingerprint(audit_bytes)},
        'analyzer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'inputs': {}, 'issues': {}, 'groups': [],
        'definitions': {
            'scope': 'Completed oracle and soak requests; cancelled requests are ID-checked but excluded from phase distributions.',
            'base_acceptance': 'Requires an already passed independent audit and identical events/server bytes. Resource/process/lifecycle conclusions are not rederived here.',
            'prefill_seconds': 'Target prefill forward computation, separate from prefill active service and suspension.',
            'model_first_token_ready_seconds': 'Generator request start to first token ready before publication; excludes HTTP admission queue and network delivery. Not client TTFT.',
            'decode_round_compute_seconds': 'Source field decode_seconds: accumulated decode round computation. It is contained within decode_service_seconds, not active plus suspension.',
            'decode_service_seconds': 'Active decode consumer execution including callbacks and first-token publication; excludes handoff and suspension.',
            'decode_suspension_seconds': 'Time outside admitted decode slices; reported separately, never added to round compute as an identity.',
            'handoff_wait_seconds': 'Producer readiness to decode-session admission, ending before payload consumption; subsequent delay to a decode slice belongs to suspension.',
            'handoff_consume_seconds': 'Same-process handoff handle consumption, not network transfer.',
            'decoded_tokens': 'Actual reported tokens generated after the first prefill token; ordinary AR requires decoded_tokens = completion_tokens - 1.',
            'request_mean_seconds_per_decoded_token': 'Each request contributes round compute / decoded_tokens and service active / decoded_tokens only when decoded_tokens > 0. Service includes first-token and callback overhead.',
            'percentiles': 'Exact nearest-rank p50/p95 across requests in one phase/profile/cache_source group. Request-mean distributions are NOT individual-token TPOT percentiles.',
            'totals': 'Each metric contains sum as well as request distribution. No heterogeneous-profile aggregate speedup or token latency percentile is inferred.',
        },
        'limits': {'input_line_bytes': MAX_LINE, 'profiles': MAX_PROFILES,
                   'groups': 2 * MAX_PROFILES * 3, 'issue_examples_per_code': 8,
                   'round_vs_service_tolerance_seconds': EPSILON_SECONDS},
    }

    def issue(code, detail):
        item = report['issues'].setdefault(code, {'count': 0, 'examples': []})
        item['count'] += 1
        if len(item['examples']) < 8:
            item['examples'].append(str(detail)[:500])

    if (audit.get('schema') != 'qwen-churn-independent-audit-v1'
            or audit.get('complete') is not True or audit.get('passed') is not True
            or audit.get('issues') != {}
            or audit.get('unmatched_client_records') != 0
            or audit.get('unmatched_server_terminals') != 0):
        issue('base_audit_not_accepted', audit_path)
    if not integer(audit.get('server_pid')) or audit['server_pid'] == 0:
        issue('invalid_base_server_pid', audit.get('server_pid'))
    if not isinstance(audit.get('run_id'), str) or not 0 < len(audit['run_id']) <= 256:
        issue('invalid_base_run_id', audit.get('run_id'))

    def expected_input(path):
        inputs = audit.get('inputs')
        if not isinstance(inputs, dict):
            issue('base_input_provenance_missing', path)
            return None
        matches = [(key, value) for key, value in inputs.items()
                   if Path(key).resolve() == path.resolve()]
        if len(matches) != 1 or not isinstance(matches[0][1], dict):
            issue('base_input_path_not_unique', path)
            return None
        return matches[0]

    def rows(path, role, mixed=False):
        expected = expected_input(path)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            before = os.fstat(stream.fileno())
            remaining = before.st_size
            consumed = 0
            while remaining:
                line = stream.readline(min(MAX_LINE + 1, remaining))
                if not line:
                    break
                remaining -= len(line)
                consumed += len(line)
                digest.update(line)
                if len(line) > MAX_LINE:
                    issue('oversized_input_line', (role, consumed))
                    break
                if not line.endswith(b'\n'):
                    issue('incomplete_input_line', (role, consumed))
                    break
                if mixed and not line.lstrip().startswith(b'{'):
                    continue
                try:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError('Expected object')
                except (ValueError, UnicodeDecodeError) as error:
                    issue('malformed_input_json', (role, str(error)))
                    continue
                yield value
            after = os.fstat(stream.fileno())
        observed = {'path': str(path.resolve()), 'snapshot_bytes': before.st_size,
                    'consumed_bytes': consumed, 'snapshot_sha256': digest.hexdigest()}
        report['inputs'][role] = observed
        if remaining or (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            issue('input_changed_or_not_fully_read', role)
        if expected is not None:
            observed['base_audit_path'] = expected[0]
            if any(observed[name] != expected[1].get(name)
                   for name in ('snapshot_bytes', 'snapshot_sha256')):
                issue('base_input_fingerprint_mismatch', role)

    with tempfile.TemporaryDirectory(prefix='qwen-churn-phases-') as temporary:
        db = sqlite3.connect(str(Path(temporary) / 'join.sqlite3'))
        db.execute('PRAGMA cache_size=-4096')
        db.execute('CREATE TABLE clients(id TEXT PRIMARY KEY, kind TEXT, profile TEXT, row TEXT, n INTEGER DEFAULT 1)')
        db.execute('CREATE TABLE terminals(id TEXT PRIMARY KEY, row TEXT, n INTEGER DEFAULT 1)')
        db.execute('CREATE TABLE metrics(phase TEXT, profile TEXT, source TEXT, category TEXT, name TEXT, value REAL)')
        counts = Counter()
        profiles = set()
        for row in rows(events_path, 'events'):
            kind = row.get('event')
            if kind not in ('oracle', 'request', 'cancel'):
                continue
            counts[kind] += 1
            identity, profile = row.get('request_id'), row.get('profile')
            if not isinstance(identity, str) or not 0 < len(identity) <= 256:
                issue('invalid_client_id', kind)
                continue
            if not isinstance(profile, str) or not 0 < len(profile) <= 128:
                issue('invalid_profile', identity)
                continue
            if profile not in profiles:
                if len(profiles) >= MAX_PROFILES:
                    issue('profile_bound_exceeded', profile)
                    continue
                profiles.add(profile)
            # Do not persist oracle output text or unneeded client fields in the temporary join.
            minimal = {key: row.get(key) for key in ('passed', 'usage', 'wall_seconds',
                       'first_content_wall_seconds', 'client_abort_requested')}
            db.execute('INSERT INTO clients(id,kind,profile,row) VALUES(?,?,?,?) '
                       'ON CONFLICT(id) DO UPDATE SET n=n+1',
                       (identity, kind, profile, json.dumps(minimal)))
        for row in rows(server_path, 'server_log', mixed=True):
            if row.get('schema') != 'qwen-http-lifecycle-v1' or row.get('event') != 'model_terminal':
                continue
            counts['model_terminal'] += 1
            identity = row.get('request_id')
            if not isinstance(identity, str) or not 0 < len(identity) <= 256:
                issue('invalid_terminal_id', row.get('event'))
                continue
            db.execute('INSERT INTO terminals(id,row) VALUES(?,?) ON CONFLICT(id) DO UPDATE SET n=n+1',
                       (identity, json.dumps(row)))
        for table, code in (('clients', 'duplicate_client_id'), ('terminals', 'duplicate_model_terminal')):
            for identity, count in db.execute(f'SELECT id,n FROM {table} WHERE n != 1'):
                issue(code, (identity, count))
        for identity, in db.execute('SELECT c.id FROM clients c LEFT JOIN terminals t ON c.id=t.id WHERE t.id IS NULL'):
            issue('missing_model_terminal', identity)
        for identity, in db.execute('SELECT t.id FROM terminals t LEFT JOIN clients c ON c.id=t.id WHERE c.id IS NULL'):
            issue('orphan_model_terminal', identity)

        matched = Counter()
        included = Counter()
        for identity, kind, profile, client_json, terminal_json in db.execute(
                'SELECT c.id,c.kind,c.profile,c.row,t.row FROM clients c JOIN terminals t ON c.id=t.id '
                'WHERE c.n=1 AND t.n=1'):
            matched[kind] += 1
            client, terminal = json.loads(client_json), json.loads(terminal_json)
            if client.get('passed') is not True:
                issue('client_not_passed', identity)
            if terminal.get('pid') != audit.get('server_pid'):
                issue('terminal_pid_mismatch', identity)
            if type(terminal.get('mtp_depth')) is not int or terminal['mtp_depth'] != 0:
                issue('non_ar_terminal', identity)
            if kind == 'cancel':
                if terminal.get('model_kind') != 'cancelled' or client.get('client_abort_requested') is not True:
                    issue('cancel_not_confirmed', identity)
                continue
            if terminal.get('model_kind') != 'completed':
                issue('success_not_completed', identity)
                continue
            source = terminal.get('cache_source')
            if source not in ('cold', 'memory', 'disk'):
                issue('invalid_cache_source', identity)
                continue
            completion, decoded = terminal.get('completion_tokens'), terminal.get('decoded_tokens')
            valid_tokens = integer(completion) and completion >= 1 and integer(decoded)
            if not valid_tokens:
                issue('invalid_token_count', (identity, completion, decoded))
            elif decoded != completion - 1:
                issue('ar_decoded_count_mismatch', (identity, completion, decoded))
            usage = client.get('usage')
            if not isinstance(usage, dict) or usage.get('completion_tokens') != completion:
                issue('client_completion_count_mismatch', identity)
            values = {}
            for name in ('wall_seconds',) + TERMINAL_SECONDS:
                value = client.get(name) if name == 'wall_seconds' else terminal.get(name)
                if not duration(value):
                    issue('missing_or_invalid_phase_seconds', (identity, name, value))
                else:
                    values['decode_round_compute_seconds' if name == 'decode_seconds' else name] = value
            first_content = client.get('first_content_wall_seconds')
            if first_content is not None:
                if not duration(first_content):
                    issue('invalid_first_content_seconds', identity)
                else:
                    values['first_content_wall_seconds'] = first_content
            round_seconds, service_seconds = terminal.get('decode_seconds'), terminal.get('decode_service_seconds')
            if duration(round_seconds) and duration(service_seconds):
                if round_seconds > service_seconds + EPSILON_SECONDS:
                    issue('round_compute_exceeds_service', identity)
                if valid_tokens and decoded == 0 and round_seconds > EPSILON_SECONDS:
                    issue('zero_decoded_tokens_with_round_compute', identity)
            phase = 'oracle' if kind == 'oracle' else 'soak_success'
            included[phase] += 1
            for name, value in values.items():
                db.execute('INSERT INTO metrics VALUES(?,?,?,?,?,?)',
                           (phase, profile, source, 'seconds', name, value))
            if valid_tokens:
                for name, value in (('completion_tokens', completion), ('decoded_tokens', decoded),
                                    ('zero_decoded_token_requests', int(decoded == 0))):
                    db.execute('INSERT INTO metrics VALUES(?,?,?,?,?,?)',
                               (phase, profile, source, 'counts', name, value))
                if decoded > 0:
                    for name, value in (('round_compute', round_seconds), ('service_active', service_seconds)):
                        if duration(value):
                            db.execute('INSERT INTO metrics VALUES(?,?,?,?,?,?)',
                                       (phase, profile, source, 'request_means', name, value / decoded))
        report['client_counts'] = dict(counts)
        report['matched_unique_terminals'] = dict(matched)
        report['completed_requests_in_groups'] = dict(included)
        for kind in ('oracle', 'request', 'cancel', 'model_terminal'):
            if counts[kind] != audit.get('client_counts', {}).get(kind, 0):
                issue('base_audit_count_mismatch', (kind, counts[kind]))
        for kind in ('oracle', 'request', 'cancel'):
            if matched[kind] != audit.get('matched_unique_terminals', {}).get(kind, 0):
                issue('base_audit_matched_count_mismatch', (kind, matched[kind]))
        if included['oracle'] == 0 or included['soak_success'] == 0:
            issue('no_completed_phase_requests', dict(included))

        db.execute('CREATE INDEX metrics_groups ON metrics(phase,profile,source,category,name,value)')

        def stats(phase, profile, source, category, name):
            parameters = (phase, profile, source, category, name)
            condition = 'phase=? AND profile=? AND source=? AND category=? AND name=?'
            count, total, low, high, mean = db.execute(
                f'SELECT COUNT(*),SUM(value),MIN(value),MAX(value),AVG(value) FROM metrics WHERE {condition}', parameters).fetchone()
            if count == 0:
                return {'count': 0}
            result = {'count': count, 'sum': total, 'min': low, 'max': high, 'mean': mean}
            for label, q in (('p50', .5), ('p95', .95)):
                result[label] = db.execute(f'SELECT value FROM metrics WHERE {condition} '
                    'ORDER BY value LIMIT 1 OFFSET ?', (*parameters, math.ceil(q * count) - 1)).fetchone()[0]
            for name, value in tuple(result.items()):
                if not duration(value):
                    issue('nonfinite_aggregate', (phase, profile, source, category, name))
                    result[name] = None
            if category == 'counts':
                for name in ('sum', 'min', 'max', 'p50', 'p95'):
                    if result[name] is not None:
                        result[name] = int(result[name])
            return result

        for phase, profile, source in db.execute('SELECT DISTINCT phase,profile,source FROM metrics ORDER BY 1,2,3').fetchall():
            group = {'phase': phase, 'profile': profile, 'cache_source': source}
            group['tokens'] = {name: stats(phase, profile, source, 'counts', name)
                               for name in ('completion_tokens', 'decoded_tokens')}
            group['completed_requests'] = group['tokens']['completion_tokens']['count']
            group['zero_decoded_token_requests'] = stats(phase, profile, source, 'counts', 'zero_decoded_token_requests').get('sum', 0)
            group['seconds'] = {name: stats(phase, profile, source, 'seconds', name) for name in DURATION_NAMES}
            group['request_mean_seconds_per_decoded_token'] = {
                name: stats(phase, profile, source, 'request_means', name) for name in MEAN_NAMES}
            report['groups'].append(group)
        db.close()
    report['passed'] = report['complete'] and not report['issues']
    return report


def self_test():
    """Tiny synthetic evidence only; the fabricated base audits are test fixtures."""
    import copy

    with tempfile.TemporaryDirectory(prefix='qwen-phase-selftest-') as directory:
        directory = Path(directory)
        audit_path, events_path, server_path = (directory / name for name in ('audit.json', 'events.ndjson', 'server.log'))
        client = {'event': 'request', 'request_id': 'r1', 'profile': 'long_00', 'passed': True,
                  'usage': {'completion_tokens': 3}, 'wall_seconds': 20, 'first_content_wall_seconds': None}
        terminal = {'schema': 'qwen-http-lifecycle-v1', 'event': 'model_terminal', 'request_id': 'r1',
                    'pid': 123, 'mtp_depth': 0, 'model_kind': 'completed', 'cache_source': 'disk',
                    'completion_tokens': 3, 'decoded_tokens': 2,
                    **{name: .5 for name in TERMINAL_SECONDS}}
        terminal['decode_seconds'], terminal['decode_service_seconds'] = .2, .6
        # Same profile/source: distinct denominators prove request-mean percentiles.
        clients = [dict(client, event='oracle', request_id='o1'), client,
                   dict(client, request_id='r2', usage={'completion_tokens': 5}),
                   dict(client, request_id='r0', usage={'completion_tokens': 1}),
                   dict(client, event='cancel', request_id='c1', client_abort_requested=True)]
        terminals = [dict(terminal, request_id='o1'), terminal,
                     dict(terminal, request_id='r2', completion_tokens=5, decoded_tokens=4,
                          decode_seconds=.8, decode_service_seconds=1.2),
                     dict(terminal, request_id='r0', completion_tokens=1, decoded_tokens=0,
                          decode_seconds=0),
                     {'schema': 'qwen-http-lifecycle-v1', 'event': 'model_terminal', 'request_id': 'c1',
                      'pid': 123, 'mtp_depth': 0, 'model_kind': 'cancelled'}]

        def check(mutator=None, issue_code=None, mutate_after_hash=False):
            c, t = copy.deepcopy(clients), copy.deepcopy(terminals)
            audit = {'schema': 'qwen-churn-independent-audit-v1', 'complete': True, 'passed': True,
                     'run_id': 'fake-phase-control', 'server_pid': 123, 'issues': {},
                     'unmatched_client_records': 0, 'unmatched_server_terminals': 0,
                     'client_counts': {'oracle': 1, 'request': 3, 'cancel': 1, 'model_terminal': 5},
                     'matched_unique_terminals': {'oracle': 1, 'request': 3, 'cancel': 1}}
            if mutator:
                mutator(c, t, audit)
            events_path.write_text(''.join(json.dumps(row) + '\n' for row in c))
            server_path.write_text('Legacy event=model_terminal duplicate is ignored\n' + ''.join(json.dumps(row) + '\n' for row in t))
            audit['inputs'] = {str(path): fingerprint(path.read_bytes()) for path in (events_path, server_path)}
            audit_path.write_text(json.dumps(audit))
            if mutate_after_hash:
                with server_path.open('a') as stream:
                    stream.write('late diagnostic\n')
            result = analyze(audit_path, events_path, server_path)
            if issue_code is None:
                assert result['passed'], result['issues']
            else:
                assert not result['passed'] and issue_code in result['issues'], result['issues']
            return result

        result = check()
        group = next(row for row in result['groups'] if row['phase'] == 'soak_success')
        assert group['tokens']['decoded_tokens']['sum'] == 6
        assert group['zero_decoded_token_requests'] == 1
        assert group['request_mean_seconds_per_decoded_token']['round_compute']['count'] == 2
        assert group['request_mean_seconds_per_decoded_token']['round_compute']['p50'] == .1
        assert group['request_mean_seconds_per_decoded_token']['round_compute']['p95'] == .2
        assert group['seconds']['decode_round_compute_seconds']['sum'] == 1.0
        checks = 1
        controls = [
            (lambda c, t, a: t[1].pop('decoded_tokens'), 'invalid_token_count'),
            (lambda c, t, a: t[1].__setitem__('decoded_tokens', 3), 'ar_decoded_count_mismatch'),
            (lambda c, t, a: t[1].__setitem__('decode_service_seconds', .1), 'round_compute_exceeds_service'),
            (lambda c, t, a: t[1].pop('handoff_consume_seconds'), 'missing_or_invalid_phase_seconds'),
            (lambda c, t, a: t[1].__setitem__('model_first_token_ready_seconds', float('nan')), 'missing_or_invalid_phase_seconds'),
            (lambda c, t, a: t[1].__setitem__('decode_suspension_seconds', -1), 'missing_or_invalid_phase_seconds'),
            (lambda c, t, a: t.append(dict(t[1])), 'duplicate_model_terminal'),
            (lambda c, t, a: t.pop(1), 'missing_model_terminal'),
            (lambda c, t, a: t[1].__setitem__('request_id', 'orphan'), 'orphan_model_terminal'),
            (lambda c, t, a: t[1].__setitem__('pid', 456), 'terminal_pid_mismatch'),
            (lambda c, t, a: t[-1].__setitem__('model_kind', 'completed'), 'cancel_not_confirmed'),
            (lambda c, t, a: t[1].__setitem__('mtp_depth', 1), 'non_ar_terminal'),
            (lambda c, t, a: a.__setitem__('complete', False), 'base_audit_not_accepted'),
            (lambda c, t, a: t[3].__setitem__('decode_seconds', .1), 'zero_decoded_tokens_with_round_compute'),
        ]
        for mutator, issue_code in controls:
            check(mutator, issue_code)
            checks += 1
        check(issue_code='base_input_fingerprint_mismatch', mutate_after_hash=True)
        checks += 1
    print(json.dumps({'self_test': 'passed', 'controls': checks, 'scope': 'synthetic CPU files only'}))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, help='Passed original independent analysis.json')
    parser.add_argument('--events', type=Path)
    parser.add_argument('--server-log', type=Path)
    parser.add_argument('--output', type=Path, help='New output file; existing files are never overwritten')
    parser.add_argument('--self-test', action='store_true', help='Run small synthetic CPU controls only')
    args = parser.parse_args()
    if args.self_test:
        if any((args.audit, args.events, args.server_log, args.output)):
            parser.error('--self-test cannot be combined with evidence arguments')
        return self_test()
    if not all((args.audit, args.events, args.server_log, args.output)):
        parser.error('--audit, --events, --server-log and --output are required')
    try:
        result = analyze(args.audit, args.events, args.server_log)
        with args.output.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(json.dumps({name: result[name] for name in
          ('complete', 'passed', 'matched_unique_terminals', 'issues')}, separators=(',', ':')))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
