#!/usr/bin/env python3
"""Stream saved churn evidence into a temporary SQLite join; never contact a server."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import tempfile

METRICS = ('wall_seconds', 'first_content_wall_seconds', 'scheduler_elapsed_seconds',
           'prefill_seconds', 'prefill_active_seconds', 'prefill_suspension_seconds',
           'decode_seconds', 'cache_lookup_seconds', 'cache_restore_seconds',
           'cache_save_seconds', 'cache_wait_seconds')
RESOURCE = ('rss_bytes', 'numeric_fds', 'request_bytes', 'cache_bytes', 'workspace_bytes',
            'total_bytes', 'leases', 'mlx_active_bytes', 'mlx_cache_bytes', 'disk_bytes',
            'pending_jobs', 'pending_bytes')


def analyze(events_path, server_path, process_path, summary_path, lifecycle_path=None):
    summary = json.loads(Path(summary_path).read_text())
    complete = summary.get('complete') is True
    report = {'schema': 'qwen-churn-independent-audit-v1', 'complete': complete, 'passed': False,
              'run_id': summary.get('run_id'), 'server_pid': summary.get('server_pid'),
              'issues': {}, 'inputs': {}, 'partial_tails': {}, 'coverage_windows': [],
              'notes': [
                  'Only JSON schema qwen-http-lifecycle-v1 model_terminal rows count; legacy text duplicates do not.',
                  'Client content hashes/finish/usage are independently compared to each profile oracle; raw token IDs are unavailable.',
                  'Percentiles use exact nearest rank over matched requests; prefill and decode total durations remain separate.',
                  'No TPOT is inferred from completion token count, and no physical SSD byte claim is made.',
                  'RSS/FD use the sampler elapsed clock; health uses the churn elapsed clock. They are not assumed to share an epoch.',
                  'Resource trends are descriptive. No numerical leak or performance SLO is invented.',
                  'Disk rejected is aggregate admission rejection; without a per-operation field it cannot identify a rejected read.',
                  'Running snapshots can have unmatched records and can never pass final acceptance.']}
    issue_counts = Counter()

    def issue(code, detail):
        issue_counts[code] += 1
        entry = report['issues'].setdefault(code, {'count': 0, 'examples': []})
        entry['count'] = issue_counts[code]
        if len(entry['examples']) < 8: entry['examples'].append(str(detail)[:500])

    def rows(path, mixed=False):
        path = Path(path)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            size = path.stat().st_size
            report['inputs'][str(path)] = {'snapshot_bytes': size}
            remaining = size
            while remaining:
                line = stream.readline(min(1_048_577, remaining))
                if not line: break
                remaining -= len(line); digest.update(line)
                if len(line) > 1_048_576:
                    issue('oversized_input_line', path); break
                if not line.endswith(b'\n'):
                    report['partial_tails'][str(path)] = len(line)
                    if complete: issue('incomplete_final_line', path)
                    break
                if mixed and not line.lstrip().startswith(b'{'): continue
                try:
                    value = json.loads(line)
                    if not isinstance(value, dict): raise ValueError('Expected object')
                    yield value
                except (ValueError, UnicodeDecodeError) as error:
                    issue('malformed_input_json', f'{path}: {error}')
        report['inputs'][str(path)]['snapshot_sha256'] = digest.hexdigest()

    def integer(value): return type(value) is int and value >= 0
    def number(value): return type(value) in (int, float) and math.isfinite(value) and value >= 0

    with tempfile.TemporaryDirectory(prefix='qwen-churn-audit-') as directory:
        db = sqlite3.connect(str(Path(directory) / 'join.sqlite3'))
        db.execute('PRAGMA cache_size=-4096')
        db.execute('CREATE TABLE clients(id TEXT PRIMARY KEY, kind TEXT, profile TEXT, row TEXT, occurrences INTEGER DEFAULT 1)')
        db.execute('CREATE TABLE terminals(id TEXT PRIMARY KEY, row TEXT, occurrences INTEGER DEFAULT 1)')
        db.execute('CREATE TABLE metrics(phase TEXT, profile TEXT, cache_source TEXT, name TEXT, value REAL)')
        db.execute('CREATE TABLE resources(source TEXT, elapsed REAL, idle INTEGER, name TEXT, value REAL)')
        counts = Counter(); oracles = {}; pressure = Counter(); process_errors = 0
        soak_started = summary.get('soak_started_elapsed_seconds')

        def resource(row, source, elapsed, health=None):
            if not number(elapsed): issue('invalid_sample_time', (source, elapsed)); return
            health = row if health is None else health
            if health.get('pid') != summary.get('server_pid'):
                issue('sample_pid_mismatch', (source, health.get('pid')))
            budget = health.get('state_budget') or {}; disk = health.get('prefix_disk_cache') or {}
            ram = health.get('prefix_cache') or {}; mlx = health.get('mlx_memory') or {}
            values = {'rss_bytes': row.get('rss_bytes'), 'numeric_fds': row.get('numeric_fds'),
                      'request_bytes': budget.get('requestBytes'), 'cache_bytes': budget.get('cacheBytes'),
                      'workspace_bytes': budget.get('workspaceBytes'), 'total_bytes': budget.get('totalBytes'),
                      'leases': budget.get('currentLeases'), 'mlx_active_bytes': mlx.get('active_bytes'),
                      'mlx_cache_bytes': mlx.get('cache_bytes'), 'disk_bytes': disk.get('diskBytes'),
                      'pending_jobs': disk.get('pendingJobs'), 'pending_bytes': disk.get('pendingBytes')}
            for key, value in values.items():
                if value is None and source == 'health' and key in ('rss_bytes', 'numeric_fds'): continue
                if not integer(value): issue('invalid_resource_value', (source, key, value)); continue
                db.execute('INSERT INTO resources VALUES(?,?,?,?,?)',
                           (source, elapsed, int(row.get('fully_idle', False)), key, value))
            if all(integer(budget.get(k)) for k in ('requestBytes', 'cacheBytes', 'workspaceBytes', 'totalBytes', 'maxBytes')):
                if sum(budget[k] for k in ('requestBytes', 'cacheBytes', 'workspaceBytes')) != budget['totalBytes']:
                    issue('ledger_sum_mismatch', (elapsed, budget))
                if budget['totalBytes'] > budget['maxBytes']: issue('ledger_limit_exceeded', (elapsed, budget))
                if integer(budget.get('peakBytes')) and budget['peakBytes'] > budget['maxBytes']:
                    issue('ledger_peak_exceeded', (elapsed, budget))
            for counters, limits, fields in (
                (ram, health.get('prefix_cache_limits', summary.get('initial_health', {}).get('prefix_cache_limits', {})),
                 (('entries', 'maxEntries'), ('logicalPayloadBytes', 'maxBytes'), ('keyTokens', 'maxKeyTokens'))),
                (disk, health.get('prefix_disk_cache_limits', summary.get('initial_health', {}).get('prefix_disk_cache_limits', {})),
                 (('entries', 'maxEntries'), ('diskBytes', 'maxBytes'), ('pendingJobs', 'maxPendingJobs'), ('pendingBytes', 'maxPendingBytes')))):
                for field, limit in fields:
                    if not integer(counters.get(field)) or not integer(limits.get(limit)):
                        issue('missing_resource_bound', (source, field, limit))
                    elif counters[field] > limits[limit]: issue('resource_limit_exceeded', (source, field, counters[field], limits[limit]))
            if health.get('memory_pressure_monitor_running') is not True: issue('pressure_monitor_not_running', source)
            state = health.get('memory_pressure') or {}
            pressure[(source, state.get('observedLevel', 'unavailable'), state.get('lastEventSource', 'none'))] += 1
            logging = health.get('logging') or {}
            if logging.get('dropped_events', 0) or logging.get('write_failures', 0):
                issue('server_log_loss', (elapsed, logging.get('dropped_events'), logging.get('write_failures')))

        for event in rows(events_path):
            kind = event.get('event')
            if kind in ('oracle', 'request', 'cancel'):
                counts[kind] += 1
                identity = event.get('request_id')
                if not isinstance(identity, str) or not 0 < len(identity) <= 256:
                    issue('client_missing_request_id', (kind, event.get('sequence'))); continue
                db.execute('INSERT INTO clients(id,kind,profile,row) VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET occurrences=occurrences+1',
                           (identity, kind, event.get('profile'), json.dumps(event, separators=(',', ':'))))
                if event.get('passed') is not True: issue('client_request_failed', (identity, event.get('error')))
                if kind == 'oracle':
                    profile = event.get('profile')
                    if profile in oracles: issue('duplicate_profile_oracle', profile)
                    oracles[profile] = event
                    if not isinstance(event.get('content'), str) or hashlib.sha256(event['content'].encode()).hexdigest() != event.get('content_sha256'):
                        issue('oracle_content_hash_mismatch', identity)
            elif kind == 'health':
                resource(event.get('health') or {}, 'health', event.get('elapsed_seconds'))
            elif kind == 'soak_started': soak_started = event.get('elapsed_seconds')
            elif kind == 'coverage_window':
                if len(report['coverage_windows']) < 128: report['coverage_windows'].append(event)
                else: issue('coverage_window_bound', 128)
            elif kind in ('health_error', 'run_error'): issue(kind, event.get('error'))

        for terminal in rows(server_path, mixed=True):
            if terminal.get('schema') != 'qwen-http-lifecycle-v1' or terminal.get('event') != 'model_terminal': continue
            counts['model_terminal'] += 1
            identity = terminal.get('request_id')
            if not isinstance(identity, str) or not identity:
                issue('terminal_missing_request_id', terminal); continue
            if terminal.get('pid') != summary.get('server_pid'): issue('terminal_pid_mismatch', identity)
            db.execute('INSERT INTO terminals(id,row) VALUES(?,?) ON CONFLICT(id) DO UPDATE SET occurrences=occurrences+1',
                       (identity, json.dumps(terminal, separators=(',', ':'))))

        for table, code in (('clients', 'duplicate_client_request_id'), ('terminals', 'duplicate_model_terminal')):
            for identity, n in db.execute(f'SELECT id,occurrences FROM {table} WHERE occurrences != 1'):
                issue(code, (identity, n))
        missing = db.execute('SELECT COUNT(*) FROM clients c LEFT JOIN terminals t ON c.id=t.id WHERE t.id IS NULL').fetchone()[0]
        orphan = db.execute('SELECT COUNT(*) FROM terminals t LEFT JOIN clients c ON c.id=t.id WHERE c.id IS NULL').fetchone()[0]
        report['unmatched_client_records'] = missing; report['unmatched_server_terminals'] = orphan
        if complete:
            for identity, in db.execute('SELECT c.id FROM clients c LEFT JOIN terminals t ON c.id=t.id WHERE t.id IS NULL'):
                issue('missing_model_terminal', identity)
            for identity, in db.execute('SELECT t.id FROM terminals t LEFT JOIN clients c ON c.id=t.id WHERE c.id IS NULL'):
                issue('orphan_model_terminal', identity)

        matched = Counter()
        for identity, kind, profile, client_json, terminal_json in db.execute(
                'SELECT c.id,c.kind,c.profile,c.row,t.row FROM clients c JOIN terminals t ON c.id=t.id WHERE c.occurrences=1 AND t.occurrences=1'):
            client, terminal = json.loads(client_json), json.loads(terminal_json)
            matched[kind] += 1
            if terminal.get('mtp_depth') != 0: issue('unexpected_mtp', identity)
            if kind == 'cancel':
                if client.get('client_abort_requested') is not True or terminal.get('model_kind') != 'cancelled':
                    issue('rst_not_cancelled', (identity, terminal.get('model_kind')))
            else:
                if terminal.get('model_kind') != 'completed': issue('success_not_completed', identity)
                usage = client.get('usage') or {}
                for name in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                    if not integer(usage.get(name)): issue('invalid_client_usage', (identity, name))
                for name in ('prompt_tokens', 'completion_tokens', 'cached_prompt_tokens', 'computed_prompt_tokens',
                             'actual_prefill_tokens', 'recomputed_prefill_tokens'):
                    if not integer(terminal.get(name)): issue('invalid_terminal_token_count', (identity, name))
                if all(integer(usage.get(k)) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')):
                    if usage['total_tokens'] != usage['prompt_tokens'] + usage['completion_tokens']:
                        issue('client_total_usage_mismatch', identity)
                for name in ('prompt_tokens', 'completion_tokens'):
                    if usage.get(name) != terminal.get(name): issue('client_server_usage_mismatch', (identity, name))
                if client.get('effective_cached_tokens') != terminal.get('cached_prompt_tokens'):
                    issue('cached_usage_mismatch', identity)
                cache_source = terminal.get('cache_source')
                if cache_source not in ('cold', 'memory', 'disk'):
                    issue('invalid_cache_source', (identity, cache_source))
                elif integer(terminal.get('cached_prompt_tokens')):
                    if (cache_source == 'cold') != (terminal['cached_prompt_tokens'] == 0):
                        issue('cache_source_token_mismatch', identity)
                if all(integer(terminal.get(k)) for k in ('prompt_tokens', 'cached_prompt_tokens', 'computed_prompt_tokens')):
                    if terminal['prompt_tokens'] != terminal['cached_prompt_tokens'] + terminal['computed_prompt_tokens']:
                        issue('prompt_cached_computed_mismatch', identity)
                if all(integer(terminal.get(k)) for k in ('actual_prefill_tokens', 'computed_prompt_tokens', 'recomputed_prefill_tokens')):
                    if terminal['actual_prefill_tokens'] != terminal['computed_prompt_tokens'] + terminal['recomputed_prefill_tokens']:
                        issue('actual_computed_recomputed_mismatch', identity)
                mapped_finish = {'eos': 'stop', 'length': 'length'}.get(terminal.get('model_finish_reason'))
                if mapped_finish != client.get('finish_reason'): issue('finish_reason_mismatch', identity)
                oracle = oracles.get(profile)
                if oracle is None: issue('missing_profile_oracle', (identity, profile))
                elif kind == 'request':
                    for name in ('content_sha256', 'finish_reason'):
                        if client.get(name) != oracle.get(name): issue('oracle_mismatch', (identity, name))
                    for name in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                        if usage.get(name) != oracle.get('usage', {}).get(name): issue('oracle_usage_mismatch', (identity, name))
            phase = 'oracle' if kind == 'oracle' else ('soak_cancel' if kind == 'cancel' else 'soak_success')
            source = terminal.get('cache_source', 'unavailable')
            for metric in METRICS:
                value = client.get(metric) if metric in ('wall_seconds', 'first_content_wall_seconds') else terminal.get(metric)
                if value is None:
                    if kind != 'cancel' and metric in ('wall_seconds', 'prefill_seconds', 'decode_seconds'):
                        issue('missing_stage_duration', (identity, metric))
                    continue
                if not number(value): issue('invalid_duration', (identity, metric)); continue
                db.execute('INSERT INTO metrics VALUES(?,?,?,?,?)', (phase, profile, source, metric, value))

        for sample in rows(process_path):
            counts['process_samples'] += 1
            if sample.get('observation_error'):
                process_errors += 1; issue('process_observation_error', sample['observation_error'])
            resource(sample, 'process', sample.get('elapsed_seconds'))
        if counts['process_samples'] == 0: issue('no_process_samples', process_path)
        db.execute('CREATE INDEX metrics_group ON metrics(phase,profile,cache_source,name,value)')
        db.execute('CREATE INDEX resources_group ON resources(source,name,elapsed,value)')

        def stats(table, condition, parameters):
            count, low, high, mean = db.execute(f'SELECT COUNT(*),MIN(value),MAX(value),AVG(value) FROM {table} WHERE {condition}', parameters).fetchone()
            if not count: return {'count': 0}
            value = {'count': count, 'min': low, 'max': high, 'mean': mean}
            for label, q in (('p50', .5), ('p95', .95)):
                value[label] = db.execute(f'SELECT value FROM {table} WHERE {condition} ORDER BY value LIMIT 1 OFFSET ?',
                                          (*parameters, math.ceil(q * count) - 1)).fetchone()[0]
            return value

        report['durations_by_phase_profile_cache_source'] = []
        groups = list(db.execute('SELECT DISTINCT phase,profile,cache_source FROM metrics ORDER BY 1,2,3'))
        for phase, profile, source in groups:
            report['durations_by_phase_profile_cache_source'].append({'phase': phase, 'profile': profile, 'cache_source': source,
                'seconds': {name: stats('metrics', 'phase=? AND profile=? AND cache_source=? AND name=?', (phase, profile, source, name)) for name in METRICS}})
        report['durations_by_phase_cache_source'] = []
        for phase, source in list(db.execute('SELECT DISTINCT phase,cache_source FROM metrics ORDER BY 1,2')):
            report['durations_by_phase_cache_source'].append({'phase': phase, 'cache_source': source,
                'seconds': {name: stats('metrics', 'phase=? AND cache_source=? AND name=?', (phase, source, name)) for name in METRICS}})
        report['resources'] = {}
        for source in ('health', 'process'):
            start, end = db.execute('SELECT MIN(elapsed),MAX(elapsed) FROM resources WHERE source=?', (source,)).fetchone()
            if start is None: continue
            span = end - start
            windows = [('all', start, end + .000001)]
            windows += [(f'quarter_{i+1}', start + span*i/4, start + span*(i+1)/4 + (.000001 if i == 3 else 0)) for i in range(4)]
            windows += [(f'elapsed_{i*300}_{(i+1)*300}s', i*300, (i+1)*300) for i in range(int(end//300)+1)]
            if source == 'health' and number(soak_started) and end >= soak_started:
                windows += [('soak_all', soak_started, end + .000001)]
                windows += [(f'soak_quarter_{i+1}', soak_started + (end-soak_started)*i/4,
                             soak_started + (end-soak_started)*(i+1)/4 + (.000001 if i == 3 else 0)) for i in range(4)]
            report['resources'][source] = {'elapsed_first': start, 'elapsed_last': end, 'windows': []}
            for label, lo, hi in windows:
                report['resources'][source]['windows'].append({'window': label, 'elapsed_start': lo, 'elapsed_end': hi,
                    'values': {name: stats('resources', 'source=? AND name=? AND elapsed>=? AND elapsed<?', (source, name, lo, hi)) for name in RESOURCE}})
            first, last = report['resources'][source]['windows'][1], report['resources'][source]['windows'][4]
            report['resources'][source]['first_to_last_quarter_p50_delta'] = {
                name: last['values'][name]['p50'] - first['values'][name]['p50']
                for name in RESOURCE if first['values'][name]['count'] and last['values'][name]['count']}
        report['pressure_observations'] = [{'source': source, 'level': level, 'event_source': event, 'samples': count}
            for (source, level, event), count in pressure.items()]
        report['client_counts'] = dict(counts); report['matched_unique_terminals'] = dict(matched)
        report['soak_started_elapsed_seconds'] = soak_started
        disk_counter_names = ('bytesRead', 'bytesWritten', 'hits', 'misses', 'published', 'evictions', 'rejected',
                              'corruptions', 'writeFailures', 'spaceRejections', 'spaceQueryFailures', 'expired')
        disk_snapshots = {stage: (summary.get(key) or {}).get('prefix_disk_cache') or {}
                          for stage, key in (('initial', 'initial_health'), ('seeded', 'seeded_health'), ('final', 'final_health'))}
        report['disk_counters'] = {stage: {key: value.get(key) for key in disk_counter_names} for stage, value in disk_snapshots.items()}
        report['disk_counters']['soak_delta'] = {key: disk_snapshots['final'][key] - disk_snapshots['seeded'][key]
            for key in disk_counter_names if integer(disk_snapshots['final'].get(key)) and integer(disk_snapshots['seeded'].get(key))}
        if complete:
            for field, actual in (('requests', counts['request'] + counts['cancel']), ('completed_requests', counts['request']),
                                  ('client_abort_requests', counts['cancel'])):
                if summary.get(field) != actual: issue('summary_count_mismatch', (field, summary.get(field), actual))
            if counts['oracle'] != len(summary.get('profiles', {})): issue('summary_oracle_count_mismatch', counts['oracle'])
            if summary.get('passed') is not True: issue('churn_summary_not_passed', summary.get('error'))
            final = summary.get('final_health') or {}
            report['final_drain'] = {key: final.get(key) for key in ('idle', 'active', 'active_jobs', 'pending_requests',
                'queued_prefills', 'ready_decodes', 'resident_sequences', 'reserved_tokens', 'waiting_prefix_sequences',
                'state_budget', 'prefix_cache', 'prefix_disk_cache')}
            for name in ('active', 'active_jobs', 'pending_requests', 'queued_prefills', 'ready_decodes',
                         'resident_sequences', 'reserved_tokens', 'waiting_prefix_sequences'):
                if final.get(name) != 0: issue('final_not_drained', (name, final.get(name)))
            for category, names in (('state_budget', ('requestBytes', 'workspaceBytes')),
                                    ('prefix_disk_cache', ('pendingJobs', 'pendingBytes'))):
                for name in names:
                    if final.get(category, {}).get(name) != 0: issue('final_owner_not_drained', (category, name))
            if final.get('idle') is not True: issue('final_not_idle', final.get('idle'))
        lifecycle_ready = True
        if lifecycle_path is not None:
            lifecycle = json.loads(Path(lifecycle_path).read_text())
            lifecycle_ready = lifecycle.get('complete') is True and lifecycle.get('passed') is True
            report['wrapper_complete_passed'] = lifecycle_ready
        report['passed'] = bool(complete and summary.get('passed') is True and lifecycle_ready and not report['issues'])
        db.close()
    return report


def analyze_cli(args):
    """Keep the original analysis pure; bind and validate CLI acceptance inputs."""
    def snapshot(path):
        with path.open('rb') as stream:
            raw = stream.read(8_388_609)
        if len(raw) > 8_388_608:
            raise ValueError(f'{path}: JSON evidence exceeds 8 MiB')
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError(f'{path}: expected JSON object')
        return raw, value

    summary_raw, summary = snapshot(args.summary)
    lifecycle_raw, lifecycle = snapshot(args.lifecycle) if args.lifecycle is not None else (None, None)
    # Analyze these exact captured bytes, rather than hashing one version and
    # rereading a concurrently updated summary/lifecycle later in analyze().
    with tempfile.TemporaryDirectory(prefix='qwen-churn-cli-evidence-') as directory:
        summary_copy = Path(directory) / 'summary.json'
        summary_copy.write_bytes(summary_raw)
        lifecycle_copy = None
        if lifecycle_raw is not None:
            lifecycle_copy = Path(directory) / 'lifecycle.json'
            lifecycle_copy.write_bytes(lifecycle_raw)
        result = analyze(args.events, args.server_log, args.process, summary_copy, lifecycle_copy)
    for path, raw in ((args.summary, summary_raw), (args.lifecycle, lifecycle_raw)):
        if path is not None:
            result['inputs'][str(path)] = {'snapshot_bytes': len(raw), 'snapshot_sha256': hashlib.sha256(raw).hexdigest()}

    def issue(code, detail):
        item = result['issues'].setdefault(code, {'count': 0, 'examples': []})
        item['count'] += 1
        if len(item['examples']) < 8:
            item['examples'].append(str(detail)[:500])

    result['cli_preview_only'] = not result['complete']
    if result['complete']:
        final = summary.get('final_health') or {}
        for category, name in (('prefix_disk_cache', 'foregroundReadIntents'), ('prefix_cache', 'liveFlights')):
            value = (final.get(category) or {}).get(name)
            if type(value) is not int or value != 0:
                issue('final_owner_not_drained', (category, name, value))
        if lifecycle is not None:
            services = lifecycle.get('services') or []
            service = services[0] if len(services) == 1 and isinstance(services[0], dict) else {}
            if service.get('pid') != result['server_pid']:
                issue('lifecycle_service_identity_mismatch', service.get('pid'))
            if service.get('close_completed_logged') is not True:
                issue('lifecycle_close_not_completed', service.get('close_completed_logged'))
            cleanup = lifecycle.get('cleanup') or {}
            for name, value in (
                ('service.exit_code', service.get('exit_code')),
                ('client.exit_code', (lifecycle.get('client') or {}).get('exit_code')),
                ('cleanup.server_exit_code', cleanup.get('server_exit_code')),
                ('cleanup.client_exit_code', cleanup.get('client_exit_code')),
            ):
                if type(value) is not int or value != 0:
                    issue('lifecycle_exit_not_zero', (name, value))
            sampling = lifecycle.get('sampling') or {}
            if cleanup.get('sampler_completed') is not True or sampling.get('completed') is not True:
                issue('lifecycle_sampler_not_completed', (cleanup.get('sampler_completed'), sampling.get('completed')))
            if type(sampling.get('errors')) is not int or sampling['errors'] != 0:
                issue('lifecycle_sampler_errors', sampling.get('errors'))
            count = sampling.get('samples')
            if type(count) is not int or count <= 0 or count != result['client_counts'].get('process_samples'):
                issue('lifecycle_sampler_count_mismatch', count)
    result['passed'] = bool(result['passed'] and not result['issues'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--events', type=Path, required=True)
    parser.add_argument('--server-log', type=Path, required=True)
    parser.add_argument('--process', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--lifecycle', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-incomplete', action='store_true',
                        help='Allow an issue-free incomplete preview to exit 0; complete/passed remain false')
    args = parser.parse_args()
    result = analyze_cli(args)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({key: result.get(key) for key in ('complete', 'passed', 'client_counts', 'matched_unique_terminals',
          'unmatched_client_records', 'unmatched_server_terminals', 'issues')}, separators=(',', ':')))
    return 0 if result['passed'] or (args.allow_incomplete and not result['complete'] and not result['issues']) else 1


if __name__ == '__main__': raise SystemExit(main())
