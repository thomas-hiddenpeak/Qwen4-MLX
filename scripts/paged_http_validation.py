"""Small CPU-only phase/lifetime audit layered on the existing churn workload."""
import math
from pathlib import Path

from capacity_http_validation import integer, require, rows
from probe_http_cache_reliability import fully_idle, idle_workspace_bytes

FIELDS = ('paged_kv_import_seconds', 'paged_kv_token_steps', 'paged_kv_reused_prefix_tokens',
          'paged_kv_imported_suffix_rows', 'paged_kv_capacity_fallbacks', 'paged_kv_reserved_pages_per_layer')


def require_paged_health(value, pages=None):
    pool = value.get('paged_kv_pool') or {}
    require(pool.get('configured') is True and (pages is None or pool.get('maximum_pages_per_layer') == pages),
            'Owned server does not publish its configured physical pool')
    expected = idle_workspace_bytes(value)
    require(expected > 0 and pool.get('fallback_policy') == 'whole_cursor_dense', 'Wrong paged idle/fallback contract')
    return expected


def validate_paged_terminals(events_path, server_path, pid, initial, final, stop_requested=lambda: False):
    report = {'schema': 'qwen-http-paged-terminal-audit-v1', 'complete': False, 'passed': False,
              'server_pid': pid, 'inputs': {}, 'errors': [], 'completed': 0, 'cancelled': 0,
              'paged_completed': 0, 'prefix_reused_completed': 0, 'warm_paged_reused_completed': 0, 'dense_fallback_completed': 0,
              'token_steps': 0, 'reused_prefix_tokens': 0, 'imported_suffix_rows': 0,
              'notes': ['Request IDs join real client completions/cancels to actual runner terminal phases.',
                        'Existing independent churn audit remains responsible for content/oracle/SSD working-set checks.',
                        'Fixed pool reservation includes allocator allowance and static metadata; retained cache pages are permitted at idle.',
                        'Cancelled partial work lacks phase counters and is not invented from configuration.']}
    clients, terminals = {}, set()
    try:
        expected = require_paged_health(initial)
        require(require_paged_health(final, initial['paged_kv_pool']['maximum_pages_per_layer']) == expected,
                'Fixed arena reservation changed')
        require(initial['pid'] == final['pid'] == pid and fully_idle(initial) and fully_idle(final), 'Owned process did not drain exactly')
        for value in rows(Path(events_path), report):
            if stop_requested(): raise InterruptedError('Paged audit interrupted')
            if value.get('event') not in ('oracle', 'request', 'cancel'): continue
            identity = value.get('request_id')
            require(isinstance(identity, str) and identity not in clients and len(clients) < 100000, 'Invalid/duplicate client ID')
            require(value.get('passed') is True, 'Client workload did not pass')
            clients[identity] = {k: value.get(k) for k in ('event', 'usage', 'client_abort_requested')}
        for value in rows(Path(server_path), report, mixed=True):
            if stop_requested(): raise InterruptedError('Paged audit interrupted')
            if value.get('schema') != 'qwen-http-lifecycle-v1' or value.get('event') != 'model_terminal': continue
            identity = value.get('request_id')
            require(identity in clients and identity not in terminals and value.get('pid') == pid, 'Orphan/duplicate/wrong-PID model terminal')
            terminals.add(identity); client = clients[identity]
            require(value.get('mtp_depth') == 0 and all(k in value for k in FIELDS), 'Missing AR/paged phase contract')
            if client['event'] == 'cancel':
                require(client.get('client_abort_requested') is True and value.get('model_kind') == 'cancelled', 'Client RST not confirmed cancelled')
                require(all(value[k] is None for k in FIELDS), 'Cancelled counters must remain unknown')
                report['cancelled'] += 1; continue
            require(value.get('model_kind') == 'completed', 'Successful client did not complete on model')
            steps, reused, imported, fallback, claim = (value[k] for k in FIELDS[1:])
            require(all(integer(x) for x in (steps, reused, imported, fallback, claim)), 'Invalid paged phase integers')
            require(type(value[FIELDS[0]]) in (int, float) and math.isfinite(value[FIELDS[0]]) and value[FIELDS[0]] >= 0, 'Invalid import duration')
            n, prompt = client['usage']['completion_tokens'], client['usage']['prompt_tokens']
            require(value.get('completion_tokens') == n and value.get('prompt_tokens') == prompt and value.get('decoded_tokens') == n-1, 'AR usage mismatch')
            require(fallback in (0, 1), 'Capacity fallback must be whole-cursor/sticky')
            if value.get('kv_append_mode') == 'paged32':
                require(steps == n-1 > 0 and fallback == 0 and reused + imported == prompt and claim > 0 and value[FIELDS[0]] > 0, 'Physical decode phase mismatch')
                # Cold producers can immediately reuse a checkpoint they just
                # published, so reused need not be <= input cached tokens.
                require(reused % 416 == 0 and reused < prompt, 'Invalid committed checkpoint boundary')
                report['paged_completed'] += 1; report['prefix_reused_completed'] += int(reused > 0)
                report['warm_paged_reused_completed'] += int(reused > 0 and value.get('cached_prompt_tokens') == reused and value.get('cache_source') == 'memory')
            else:
                require(value.get('kv_append_mode') == 'reference' and steps == reused == imported == claim == 0 and (fallback == 1 or n == 1), 'Unexpected dense fallback behavior')
                report['dense_fallback_completed'] += fallback
            report['completed'] += 1; report['token_steps'] += steps
            report['reused_prefix_tokens'] += reused; report['imported_suffix_rows'] += imported
        require(terminals == clients.keys() and report['completed'] > 0 and report['cancelled'] > 0, 'Incomplete actual request/cancel coverage')
        require(report['paged_completed'] > 0 and report['prefix_reused_completed'] > 0, 'No actual physical prefix-reuse completion')
        require(report['warm_paged_reused_completed'] > 0, 'No actual RAM-hit physical prefix-reuse completion')
        a, b = initial['paged_kv_pool']['statistics'], final['paged_kv_pool']['statistics']
        require(b['encoded_materializations'] == a['encoded_materializations'] == 0, 'Normal HTTP performed dense page exports')
        require(b['encoded_reads'] - a['encoded_reads'] >= report['token_steps'] * 12,
                'Native twelve-layer reads do not cover completed paged token steps')
        for health, actual in (('terminal_results_with_phases', report['completed']), ('terminal_events_without_phases', report['cancelled']),
                ('completed_token_steps', report['token_steps']), ('completed_capacity_fallbacks', report['dense_fallback_completed']),
                ('completed_reused_prefix_tokens', report['reused_prefix_tokens']), ('completed_imported_suffix_rows', report['imported_suffix_rows'])):
            require(b[health] - a[health] == actual, 'Terminal/health counter mismatch: ' + health)
        report.update(expected_fixed_workspace_bytes=expected, final_live_cache_pages=b['live_pages'],
                      initial_pool=a, final_pool=b, complete=True, passed=True)
    except (OSError, ValueError, TypeError, KeyError, InterruptedError) as error:
        report['errors'].append(f'{type(error).__name__}: {error}'[:1500])
    return report
