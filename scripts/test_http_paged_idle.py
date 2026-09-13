"""CPU-only ownership/counter controls; no HTTP or model is started."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from probe_http_cache_reliability import fully_idle, idle_workspace_bytes
from paged_http_validation import validate_paged_terminals


def health():
    s = dict(vm_page_bytes=16384, layer_count=12, physical_pages=6144,
        arena_logical_bytes=402653184, arena_allocated_bytes=402653184, arena_reserved_bytes=405012480,
        live_pages=0, free_pages=6144, minimum_free_pages_per_layer=512, high_water_pages=0,
        claimed_pages_per_layer=0, active_decode_claims=0, encoded_writes=0, encoded_reads=0,
        encoded_materializations=0, in_flight_operations=0, completed_operations=0,
        failed_operations=0, statistics_error=None, terminal_results_with_phases=0,
        terminal_events_without_phases=0, completed_token_steps=0, completed_capacity_fallbacks=0,
        completed_reused_prefix_tokens=0, completed_imported_suffix_rows=0)
    return dict(pid=123, idle=True, active=0, active_jobs=0, pending_requests=0,
        queued_prefills=0, ready_decodes=0, resident_sequences=0, reserved_tokens=0,
        waiting_prefix_sequences=0, prefix_cache=dict(logicalPayloadBytes=0, liveFlights=0),
        prefix_disk_cache=dict(pendingJobs=0, pendingBytes=0, foregroundReadIntents=0),
        state_budget=dict(requestBytes=0, cacheBytes=0, workspaceBytes=405012480),
        paged_kv_pool=dict(configured=True, page_tokens=32, maximum_pages_per_layer=512,
            fallback_policy='whole_cursor_dense', statistics=s))


class PagedIdleTests(unittest.TestCase):
    def test_exact_arena_plus_metadata_and_legacy_zero_workspace(self):
        h = health(); self.assertEqual(idle_workspace_bytes(h), 405012480); self.assertTrue(fully_idle(h))
        del h['paged_kv_pool']; h['state_budget']['workspaceBytes'] = 0
        self.assertTrue(fully_idle(h))
        h['state_budget']['workspaceBytes'] = 1; self.assertFalse(fully_idle(h))

    def test_extra_workspace_or_unreleased_claim_never_becomes_idle(self):
        h = health(); h['state_budget']['workspaceBytes'] += 1; self.assertFalse(fully_idle(h))
        h = health(); h['paged_kv_pool']['statistics'].update(active_decode_claims=1, claimed_pages_per_layer=9)
        self.assertFalse(fully_idle(h))

    def test_reservation_tamper_poison_or_unknown_statistics_rejected(self):
        for changes in ({'arena_reserved_bytes': 405012481}, {'failed_operations': 1},
                        {'statistics_error': 'unknown'}, {'vm_page_bytes': True}):
            h = health(); h['paged_kv_pool']['statistics'].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError): fully_idle(h)

    def test_empty_cache_cannot_leave_live_pages_but_retained_cache_can(self):
        h = health(); h['paged_kv_pool']['statistics'].update(live_pages=12, free_pages=6132, high_water_pages=12, minimum_free_pages_per_layer=511)
        self.assertFalse(fully_idle(h))
        h['prefix_cache']['logicalPayloadBytes'] = h['state_budget']['cacheBytes'] = 1000
        self.assertTrue(fully_idle(h))
        h['state_budget']['cacheBytes'] += 1; self.assertFalse(fully_idle(h))

    def test_callback_io_and_flight_owners_prevent_idle(self):
        for section, field in (('prefix_disk_cache', 'foregroundReadIntents'), ('prefix_cache', 'liveFlights')):
            h = health(); h[section][field] = 1; self.assertFalse(fully_idle(h))

    def test_live_completion_atomic_overlap_is_valid_but_not_idle(self):
        h = health(); h['paged_kv_pool']['statistics'].update(encoded_reads=1, completed_operations=1, in_flight_operations=1)
        self.assertEqual(idle_workspace_bytes(h), 405012480); self.assertFalse(fully_idle(h))
        h['paged_kv_pool']['statistics']['in_flight_operations'] = 0; self.assertTrue(fully_idle(h))

    def terminal_fixture(self, warm=True):
        initial, final = health(), health(); events, terminals = [], []
        for i, kind in enumerate(('request', 'request', 'cancel')):
            identity = f'id-{i}'
            events.append(dict(event=kind, request_id=identity, passed=True, client_abort_requested=kind == 'cancel',
                               usage=dict(prompt_tokens=11057, completion_tokens=16)))
            row = dict(schema='qwen-http-lifecycle-v1', event='model_terminal', request_id=identity, pid=123,
                mtp_depth=0, model_kind='cancelled' if kind == 'cancel' else 'completed',
                prompt_tokens=11057, completion_tokens=16, decoded_tokens=15, kv_append_mode='paged32',
                cached_prompt_tokens=10816 if i == 1 and warm else 0, cache_source='memory' if i == 1 and warm else 'cold',
                paged_kv_import_seconds=.005, paged_kv_token_steps=15, paged_kv_reused_prefix_tokens=10816,
                paged_kv_imported_suffix_rows=241, paged_kv_capacity_fallbacks=0, paged_kv_reserved_pages_per_layer=9)
            if kind == 'cancel':
                for field in list(row):
                    if field.startswith('paged_kv_'): row[field] = None
            terminals.append(row)
        final['paged_kv_pool']['statistics'].update(terminal_results_with_phases=2, terminal_events_without_phases=1,
            completed_token_steps=30, completed_reused_prefix_tokens=21632, completed_imported_suffix_rows=482,
            encoded_reads=360, encoded_writes=384, completed_operations=744)
        return initial, final, events, terminals

    def audit_fixture(self, fixture):
        initial, final, events, terminals = fixture
        with tempfile.TemporaryDirectory() as directory:
            e, s = Path(directory)/'events.ndjson', Path(directory)/'server.log'
            e.write_text(''.join(json.dumps(r)+'\n' for r in events)); s.write_text(''.join(json.dumps(r)+'\n' for r in terminals))
            return validate_paged_terminals(e, s, 123, initial, final)

    def test_cold_nonoracle_can_reuse_new_checkpoint_and_warm_proves_cross_request(self):
        result = self.audit_fixture(self.terminal_fixture())
        self.assertTrue(result['passed'], result['errors']); self.assertEqual(result['warm_paged_reused_completed'], 1)
        result = self.audit_fixture(self.terminal_fixture(warm=False))
        self.assertFalse(result['passed']); self.assertIn('RAM-hit', result['errors'][0])
        fixture = self.terminal_fixture(); fixture[3][1]['cached_prompt_tokens'] -= 416
        result = self.audit_fixture(fixture); self.assertFalse(result['passed']); self.assertIn('RAM-hit', result['errors'][0])

    def test_terminal_health_drift_and_exports_rejected(self):
        for field in ('completed_token_steps', 'encoded_materializations'):
            fixture = self.terminal_fixture(); fixture[1]['paged_kv_pool']['statistics'][field] += 1
            if field == 'encoded_materializations': fixture[1]['paged_kv_pool']['statistics']['completed_operations'] += 1
            result = self.audit_fixture(fixture); self.assertFalse(result['passed'], field)


if __name__ == '__main__': unittest.main()
