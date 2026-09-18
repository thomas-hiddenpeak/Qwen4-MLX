#!/usr/bin/env python3
"""Bounded CPU checks of the v2 explicit-weight contract; no device execution."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

from export_coreai_pd_shared import (ALIGNMENT, ExternalModule, buffer_signature,
    export_generic, externalizable_buffers, geometry_signature, validate_baseline,
    write_aligned_weights, validate_contiguous_affine, build_decoder_layer, main,
    resolve_qsa_working_sets, qsa_working_set_modules, resolve_moe_transfers,
    validate_gdn_prefill_rows, validate_gdn_prefill_policy, resolve_phases, radix4_tail_chunks)


torch.set_num_threads(2)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('weight', torch.arange(12, dtype=torch.float16).reshape(3, 4) / 16)
        self.register_buffer('bias', torch.tensor([.125, -.25, .5], dtype=torch.float32))
        self.register_buffer('geometry', torch.arange(3, dtype=torch.int32))

    def forward(self, stream, state):
        output = (torch.nn.functional.linear(stream.float(), self.weight.float())
                  + self.bias + self.geometry.float()).half()
        return output, state + stream.float().sum()


class SharedExportTests(unittest.TestCase):
    def test_radix4_tail_family_is_optional_bounded_and_unpadded(self):
        baseline = dict(version=1, backend='native-coreai-pd', status='complete',
            completeModelLayerSet=True, prefillKernels='tensor', q4Kernel='metal',
            stableProjections=False, layers=[{'index': i} for i in range(48)],
            tokenChunk=2048, tailChunks=[4, 16, 32, 64, 128, 256, 512, 1024], capacity=16384)
        self.assertEqual(resolve_phases(baseline), validate_baseline(baseline))
        for primary in (32, 64, 256, 1024, 2048, 4096):
            phases = dict(resolve_phases(baseline, primary, integer_grouping=True, radix4_tails=True))
            self.assertEqual(phases['prefill'], primary)
            for size in (48, 192, 768):
                self.assertEqual(f'prefill_s{size}' in phases, size < primary)
            self.assertEqual(len(phases), len(set(phases.values())))
        phases = dict(resolve_phases(baseline, radix4_tails=True))
        counts = sorted(phases.values(), reverse=True)
        remaining, selected = 817, []
        while remaining:
            size = next(n for n in counts if n <= remaining)
            selected.append(size)
            remaining -= size
        self.assertEqual(selected, [768, 48, 1])
        self.assertEqual(radix4_tail_chunks(48), [])
        for invalid in (True, 1.5, 0):
            with self.assertRaises(ValueError): radix4_tail_chunks(invalid)
        with self.assertRaises(ValueError): resolve_phases(baseline, radix4_tails=1)

    def test_gdn_prefill_rows_validate_before_weight_reads(self):
        for rows in (1, 2, 4):
            validate_gdn_prefill_rows(rows)
        for rows in (0, 3, 8, None, True, 2.0, '4'):
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, '--gdn-prefill-rows'):
                build_decoder_layer(None, {}, 4096, prefill_sdpa_fp16=True,
                    fuse_gateup=True, gdn_prefill_rows=rows)

    def test_gdn_readout_policy_validate_before_weight_reads(self):
        validate_gdn_prefill_policy(4, 'readout-v2')
        for rows in (1, 2, 4):
            validate_gdn_prefill_policy(rows, 'experimental-v1')
        for rows, policy in ((1, 'readout-v2'), (2, 'readout-v2'), (4, 'unknown')):
            with self.assertRaisesRegex(ValueError, '--gdn-prefill-policy'):
                build_decoder_layer(None, {}, 4096, prefill_sdpa_fp16=True,
                    fuse_gateup=True, gdn_prefill_rows=rows, gdn_prefill_policy=policy)

    def test_builder_gdn_ilp_preserves_external_weights_state_and_s1(self):
        import export_coreai_pd_shared as exporter
        from coreai_gdn_chunk_metal import FusedGDNRecurrence
        from coreai_gdn_ilp_probe import PhaseILPRecurrence
        from coreai_moe_chunk import make_synthetic
        from export_coreai_gdn import GDN, GDNConfig, STATE_BINDINGS
        from export_coreai_hybrid import state_metadata
        config = {'hidden_size': 64, 'hc_count': 4, 'hc_lowrank': 8,
                  'rms_norm_eps': 1e-6, 'vocab_size': 32, 'ple_embed_dim': 16,
                  'ple_conv_kernel_size': 4, 'ngram_size': 3,
                  'num_experts': 12, 'num_experts_per_tok': 10}
        hc = {'input_mix_weight_down.weight': np.ones((8, 256), np.float32) * .01,
              'input_mix_weight_up.weight': np.ones((256, 8), np.float32) * .01,
              'hc_norm.weight': np.ones(256, np.float32),
              'block_inject_weight.weight': np.ones((4, 256), np.float32) * .01}
        def prepare(*_):
            c = GDNConfig(64, 1, 2, 128, 7, 4, 1e-6)
            rng = np.random.default_rng(2911)
            weights = {name: (rng.standard_normal(shape) * .1).astype(np.float32)
                       for name, shape in c.weight_shapes.items()}
            states = {'conv_history': torch.zeros(1, 3, c.channels).half(),
                      'recurrent_state': torch.zeros(1, 2, 7, 128)}
            return 'gdn', GDN(c, weights), states, STATE_BINDINGS, 'hidden', 'output'
        results = []
        with mock.patch.object(exporter, 'load_layer', side_effect=lambda *_: make_synthetic()), \
                mock.patch.object(exporter, 'prepare_layer', side_effect=prepare), \
                mock.patch.object(exporter, 'read_hc', return_value=hc):
            for rows in (1, 2, 4):
                results.append(build_decoder_layer(SimpleNamespace(prefix='language_model.model.layers.0.mlp.'),
                    config, 32, prefill_sdpa_fp16=True, fuse_gateup=True, gdn_prefill_rows=rows))
        baseline, _, original_states, original_bindings, original_geometry, _ = results[0]
        self.assertIsInstance(baseline.attention.recurrence, FusedGDNRecurrence)
        original_named, _ = externalizable_buffers(baseline, original_geometry)
        before_bytes = [(n, v.numpy().tobytes()) for n, v in original_named]
        hidden = (torch.randn(1, 1, 64) * .1).half()
        original_s1 = baseline.attention(hidden, *original_states.values())
        for rows, (module, kind, states, bindings, geometry, kernels) in zip((2, 4), results[1:]):
            self.assertEqual(kind, 'gdn')
            self.assertIsInstance(module.attention.recurrence, PhaseILPRecurrence)
            self.assertEqual(module.attention.recurrence.rows, rows)
            self.assertIsInstance(module.attention.recurrence.decode, FusedGDNRecurrence)
            self.assertEqual(len(kernels), len(set(kernels)))
            named, _ = externalizable_buffers(module, geometry)
            self.assertEqual(buffer_signature(named), buffer_signature(original_named))
            self.assertEqual([(n, v.numpy().tobytes()) for n, v in named], before_bytes)
            self.assertEqual(state_metadata(states), state_metadata(original_states))
            self.assertEqual(bindings, original_bindings)
            self.assertEqual(geometry, original_geometry)
            for a, b in zip(module.attention(hidden, *states.values()), original_s1):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_moe_transfers_configuration_and_shared_weight_signature(self):
        from coreai_moe_chunk import ChunkQ4MoE, make_synthetic
        from coreai_moe_transfers import install_moe_transfers
        from coreai_q4_flat import flatten_moe_weights
        self.assertEqual(resolve_moe_transfers(), {'enabled': False, 'tailPrecision': 'native'})
        self.assertEqual(resolve_moe_transfers(True), {'enabled': True, 'tailPrecision': 'float32'})
        for precision in ('float16', 'float32', 'native', 'native-copy'):
            self.assertEqual(resolve_moe_transfers(True, precision)['tailPrecision'], precision)
            with self.assertRaisesRegex(ValueError, 'requires --moe-direct-transfers'):
                resolve_moe_transfers(False, precision)
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            resolve_moe_transfers(True, 'float64')
        with self.assertRaisesRegex(ValueError, 'top10'):
            build_decoder_layer(None, {'num_experts_per_tok': 8, 'hidden_size': 64}, 4096,
                prefill_sdpa_fp16=True, fuse_gateup=True, moe_direct_transfers=True)
        module = ChunkQ4MoE(make_synthetic(), block=16, columns=32, inner=64, fuse_gateup=True)
        flatten_moe_weights(module)
        before, geometry = externalizable_buffers(module, {'decode.expert_ids'})
        original_signature = buffer_signature(before)
        original_geometry = geometry_signature(geometry)
        original_owners = [(name, value.data_ptr()) for name, value in module.named_buffers()]
        install_moe_transfers(module, tail_precision='float32')
        after, geometry = externalizable_buffers(module, {'decode.expert_ids'})
        self.assertEqual(buffer_signature(after), original_signature)
        self.assertEqual(geometry_signature(geometry), original_geometry)
        self.assertEqual([(name, value.data_ptr()) for name, value in module.named_buffers()], original_owners)
        examples = {name: {'x': torch.zeros(1, count, 64).half()} for name, count in [('main', 1), ('prefill', 4)]}
        with tempfile.TemporaryDirectory() as directory:
            for precision in ('float32', 'native-copy'):
                install_moe_transfers(module, tail_precision=precision)
                result = export_generic(module, after, {'decode.expert_ids'}, examples, ('output', 'ids', 'scores'),
                    Path(directory)/f'moe-{precision}.aimodel', module.custom_kernels())
                self.assertEqual(result['weightSignature'], original_signature)
                self.assertTrue(all(set(item['usedCapturedBuffers']) <= {'base.decode.expert_ids'}
                                    for item in result['torchExport'].values()))
                self.assertTrue(all(item['userInputCount'] == len(after)+1 for item in result['torchExport'].values()))

    def test_qsa_working_set_limits_and_shared_tensor_ownership(self):
        from coreai_qsa_chunk import QwenQSAChunk, make_tiny
        from export_coreai_pd import DecoderLayer
        self.assertEqual(resolve_qsa_working_sets(None, 2048, 16384), [])
        entries = resolve_qsa_working_sets([32, 16], 8, 64)
        self.assertEqual(entries, [
            {'tokenCount': 8, 'kvLimit': 16, 'function': 'prefill_s8_kv16'},
            {'tokenCount': 8, 'kvLimit': 32, 'function': 'prefill_s8_kv32'}])
        for limits, count in (([], 8), ([8, 8], 8), ([4], 8), ([10], 8), ([64], 8), ([8], 1)):
            with self.assertRaises(ValueError):resolve_qsa_working_sets(limits, count, 64)
        original = DecoderLayer(QwenQSAChunk(make_tiny(64, 16), prefill_sdpa_fp16=True),
            torch.nn.Identity(), torch.nn.Identity(), torch.nn.Identity(), torch.nn.Identity(), 6)
        base_buffers = dict(original.named_buffers())
        overrides = qsa_working_set_modules(original, entries)
        self.assertEqual(list(overrides), ['prefill_s8_kv16', 'prefill_s8_kv32'])
        for module in overrides.values():
            buffers = dict(module.named_buffers())
            self.assertEqual(list(buffers), list(base_buffers))
            for name, value in buffers.items():
                self.assertIs(value, base_buffers[name])
        with self.assertRaises(ValueError):
            qsa_working_set_modules(original, [{**entries[0], 'function': 'wrong_name'}])

    def test_entry_override_exports_distinct_math_with_same_external_contract(self):
        class ScaledTiny(Tiny):
            def forward(self, stream, state):
                output, next_state = super().forward(stream, state)
                return output * 2, next_state
        module, alternate = Tiny(), ScaledTiny()
        named, geometry = externalizable_buffers(module, {'geometry'})
        examples = {name: {'stream': torch.ones(1, 4, 4).half(), 'state': torch.zeros(1)}
                    for name in ('main', 'specialized')}
        weights = tuple(value for _, value in named)
        base = ExternalModule(module, [name for name, _ in named], 2)(*examples['main'].values(), *weights)
        specialized = ExternalModule(alternate, [name for name, _ in named], 2)(*examples['main'].values(), *weights)
        torch.testing.assert_close(specialized[0], base[0] * 2, rtol=0, atol=0)
        torch.testing.assert_close(specialized[1], base[1], rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as directory:
            asset = export_generic(module, named, geometry, examples, ('output', 'state_out'),
                Path(directory)/'overrides.aimodel', [], entry_modules={'specialized': alternate})
            self.assertEqual(set(asset['torchExport']), {'main', 'specialized'})
            self.assertTrue(all(item['usedCapturedBuffers'] == ['base.geometry']
                                for item in asset['torchExport'].values()))
            with self.assertRaisesRegex(ValueError, 'no corresponding'):
                export_generic(module, named, geometry, examples, ('output', 'state_out'),
                    Path(directory)/'unknown.aimodel', [], entry_modules={'unknown': alternate})
            alternate.weight = alternate.weight.float()
            with self.assertRaisesRegex(ValueError, 'signature'):
                export_generic(module, named, geometry, examples, ('output', 'state_out'),
                    Path(directory)/'bad-shape.aimodel', [], entry_modules={'specialized': alternate})

    def test_contiguous_affine_configuration_rejects_before_weight_reads(self):
        for flat in (False, True):
            validate_contiguous_affine(flat_q4=flat, moe_tile=(16, 64, 128), contiguous_affine=False)
        for block in (16, 32):
            for columns in (32, 64):
                validate_contiguous_affine(flat_q4=True, moe_tile=(block, columns, 64), contiguous_affine=True)
        for flat, tile, message in ((False, (16, 32, 64), '--flat-q4'),
                                    (True, (16, 32, 128), 'BK=64'),
                                    (True, (16, 16, 64), 'BN32/64')):
            with self.subTest(flat=flat, tile=tile), self.assertRaisesRegex(ValueError, message):
                # None cannot read a model source: invalid setup must fail first.
                build_decoder_layer(None, {}, 4096, prefill_sdpa_fp16=True, fuse_gateup=True,
                    flat_q4=flat, moe_tile=tile, contiguous_affine=True)

    def test_actual_cli_contiguous_affine_default_opt_in_and_provenance(self):
        import export_coreai_pd_shared as exporter
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline_dir, model_dir = root/'baseline', root/'model'
            baseline_dir.mkdir()
            model_dir.mkdir()
            config = model_dir/'config.json'
            config.write_text('{"text_config": {}}')
            asset_dir = baseline_dir/'embedding.aimodel'
            asset_dir.mkdir()
            (asset_dir/'tiny.bin').write_bytes(b'CPU CLI fixture')
            baseline = {'version': 1, 'backend': 'native-coreai-pd', 'status': 'complete',
                'completeModelLayerSet': True, 'prefillKernels': 'tensor', 'q4Kernel': 'metal',
                'stableProjections': False, 'layers': [{'index': i} for i in range(48)],
                'tokenChunk': 2048, 'tailChunks': [], 'capacity': 4096, 'moeTile': [16, 32, 64],
                'modelDirectory': str(model_dir), 'configSHA256': hashlib.sha256(config.read_bytes()).hexdigest(),
                'assets': {'embedding': {'path': 'embedding.aimodel', 'modelBytes': 15}}}
            (baseline_dir/'manifest.json').write_text(json.dumps(baseline))
            with mock.patch.object(exporter, 'Source') as source, \
                    mock.patch.object(exporter.DenseConfig, 'from_model'), \
                    mock.patch.object(exporter, 'verify_recorded_asset'), \
                    mock.patch.object(exporter.torch, 'set_num_threads'), \
                    mock.patch.object(exporter.torch, 'set_num_interop_threads'), \
                    mock.patch.object(exporter.shutil, 'disk_usage', return_value=mock.Mock(free=10**12)):
                source.return_value.directory = model_dir
                cases = [(False, False, None), (True, False, None), (False, True, None),
                         (True, True, 'float16'), (False, True, 'float32'), (False, True, 'native'),
                         (False, True, 'native-copy')]
                for index, (enabled, direct, precision) in enumerate(cases):
                    output = root/f'case-{index}'
                    arguments = ['--baseline-pd', str(baseline_dir), '--output', str(output),
                                 '--components', 'embedding', '--flat-q4']
                    if enabled:
                        arguments += ['--contiguous-affine']
                    if direct:
                        arguments += ['--moe-direct-transfers']
                    if precision is not None:
                        arguments += ['--moe-tail-precision', precision]
                    rows = (1, 2, 4)[index % 3]
                    if index:
                        arguments += ['--gdn-prefill-rows', str(rows)]
                    main(arguments)
                    saved = json.loads((output/'manifest.json').read_text())
                    self.assertEqual(saved['contiguousAffine'], enabled)
                    self.assertTrue(saved['flatQ4'])
                    self.assertIn('coreai_q4_flat', saved['authoringSourceSHA256'])
                    self.assertIn('coreai_moe_transfers', saved['authoringSourceSHA256'])
                    self.assertIn('coreai_moe_inverse_copy', saved['authoringSourceSHA256'])
                    self.assertIn('coreai_gdn_ilp_probe', saved['authoringSourceSHA256'])
                    self.assertEqual(saved['gdnPrefillRows'], rows)
                    self.assertIn('S1', saved['gdnPrefillNumerics'])
                    self.assertEqual(saved['moeDirectTransfers'], direct)
                    self.assertEqual(saved['moeTailPrecision'], (precision or 'float32') if direct else 'native')
                    self.assertFalse(saved['completeModelLayerSet'])
                source.reset_mock()
                with self.assertRaisesRegex(ValueError, '--flat-q4'):
                    main(['--baseline-pd', str(baseline_dir), '--output', str(root/'invalid'),
                          '--contiguous-affine'])
                source.assert_not_called()
                with self.assertRaisesRegex(ValueError, 'requires --moe-direct-transfers'):
                    main(['--baseline-pd', str(baseline_dir), '--output', str(root/'invalid-tail'),
                          '--moe-tail-precision', 'float32'])
                source.assert_not_called()

    def test_weight_binary_preserves_dtype_bytes_alignment_and_hashes(self):
        named = [('packed', torch.tensor([[-32768, -1, 0, 32767]], dtype=torch.int16)),
                 ('half', torch.tensor([.125, -1.75], dtype=torch.float16)),
                 ('norm', torch.tensor([1.00001, -.001], dtype=torch.float32))]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'layer.weights.bin'
            record = write_aligned_weights(path, named)
            raw = path.read_bytes()
            self.assertEqual(record['sha256'], hashlib.sha256(raw).hexdigest())
            self.assertEqual(record['byteLength'], len(raw))
            self.assertEqual(len(raw) % ALIGNMENT, 0)
            for item, (_, value) in zip(record['buffers'], named):
                self.assertEqual(item['byteOffset'] % ALIGNMENT, 0)
                actual = raw[item['byteOffset']:item['byteOffset'] + item['byteLength']]
                self.assertEqual(actual, value.numpy().tobytes())
                self.assertEqual(item['sha256'], hashlib.sha256(actual).hexdigest())
            with self.assertRaises(FileExistsError):
                write_aligned_weights(path, named)

    def test_explicit_weights_change_result_and_restore_original_module(self):
        base = Tiny()
        named, geometry = externalizable_buffers(base, {'geometry'})
        wrapper = ExternalModule(base, [name for name, _ in named], 2)
        x, state = torch.ones(1, 2, 4).half(), torch.tensor([.75])
        original = base(x, state)
        supplied = tuple(value for _, value in named)
        actual = wrapper(x, state, *supplied)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(original, actual)))
        changed = wrapper(x, state, supplied[0] * 0, supplied[1] + 2)
        self.assertFalse(torch.equal(changed[0], original[0]))
        self.assertTrue(torch.equal(base(x, state)[0], original[0]))
        self.assertEqual(set(geometry), {'geometry'})
        with self.assertRaises(ValueError):
            wrapper(x, state, supplied[0])

    def test_graph_reuse_signature_distinguishes_layout_from_values(self):
        one = Tiny()
        two = Tiny()
        two.weight *= 2
        first, first_geometry = externalizable_buffers(one, {'geometry'})
        second, second_geometry = externalizable_buffers(two, {'geometry'})
        self.assertEqual(buffer_signature(first), buffer_signature(second))
        self.assertEqual(geometry_signature(first_geometry), geometry_signature(second_geometry))
        two.geometry += 1
        self.assertNotEqual(geometry_signature(first_geometry), geometry_signature(second_geometry))
        second[0] = (second[0][0], second[0][1].float())
        self.assertNotEqual(buffer_signature(first), buffer_signature(second))

    def test_unknown_geometry_exclusion_is_rejected(self):
        with self.assertRaises(ValueError):
            externalizable_buffers(Tiny(), {'missing'})

    def test_two_phase_asset_has_explicit_inputs_and_only_geometry_capture(self):
        module = Tiny()
        named, geometry = externalizable_buffers(module, {'geometry'})
        examples = {name: {'stream': torch.zeros(1, count, 4).half(), 'state': torch.zeros(1)}
                    for name, count in [('main', 1), ('prefill', 4)]}
        with tempfile.TemporaryDirectory() as directory:
            result = export_generic(module, named, geometry, examples, ('output', 'state_out'),
                Path(directory) / 'tiny.aimodel', [], metal_weight_inputs=True)
            self.assertEqual(result['inputNames'], ['stream', 'state', 'weight_000', 'weight_001'])
            self.assertEqual(set(result['torchExport']), {'main', 'prefill'})
            for stats in result['torchExport'].values():
                self.assertEqual(stats['usedCapturedBuffers'], ['base.geometry'])
                self.assertEqual(stats['userInputCount'], 4)
            self.assertTrue(result['metalWeightInputs'])
            self.assertLess(result['modelBytes'], 100000)

    def test_sdpa_externalization_retains_explicit_weight_input(self):
        from coreai_torch.composite_ops import SDPA
        class Attention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('weight', torch.eye(4).half())
                self.sdpa = SDPA(scale=.5, is_causal=False)
            def forward(self, stream):
                x = torch.nn.functional.linear(stream.float(), self.weight.float())[:, None]
                return self.sdpa(x, x, x).half()
        module = Attention()
        named, geometry = externalizable_buffers(module, set())
        examples = {name: {'stream': torch.zeros(1, count, 4).half()}
                    for name, count in [('main', 1), ('prefill', 4)]}
        with tempfile.TemporaryDirectory() as directory:
            result = export_generic(module, named, geometry, examples, ('output',),
                Path(directory) / 'attention.aimodel', [])
            self.assertEqual(result['inputNames'], ['stream', 'weight_000'])
            self.assertTrue(all(not item['usedCapturedBuffers'] for item in result['torchExport'].values()))

    def test_baseline_preserves_phase_contract_and_rejects_invalid_tails(self):
        manifest = {'version': 1, 'backend': 'native-coreai-pd', 'status': 'complete',
            'completeModelLayerSet': True, 'prefillKernels': 'tensor', 'q4Kernel': 'metal',
            'stableProjections': False, 'layers': [{'index': i} for i in range(48)],
            'tokenChunk': 2048, 'tailChunks': [4, 64, 512]}
        self.assertEqual(validate_baseline(manifest), [('main', 1), ('prefill', 2048),
            ('prefill_s4', 4), ('prefill_s64', 64), ('prefill_s512', 512)])
        for tails in ([4, 4], [2048], [7]):
            with self.assertRaises(ValueError):
                validate_baseline({**manifest, 'tailChunks': tails})
        with self.assertRaises(ValueError):
            validate_baseline({**manifest, 'status': 'exporting'})


if __name__ == '__main__':
    unittest.main()
