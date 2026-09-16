#!/usr/bin/env python3
"""Bounded CPU checks of the v2 explicit-weight contract; no device execution."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from export_coreai_pd_shared import (ALIGNMENT, ExternalModule, buffer_signature,
    export_generic, externalizable_buffers, geometry_signature, validate_baseline,
    write_aligned_weights, validate_contiguous_affine, build_decoder_layer, main)


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
                for enabled in (False, True):
                    output = root/('on' if enabled else 'off')
                    arguments = ['--baseline-pd', str(baseline_dir), '--output', str(output),
                                 '--components', 'embedding', '--flat-q4']
                    if enabled:
                        arguments += ['--contiguous-affine']
                    main(arguments)
                    saved = json.loads((output/'manifest.json').read_text())
                    self.assertEqual(saved['contiguousAffine'], enabled)
                    self.assertTrue(saved['flatQ4'])
                    self.assertIn('coreai_q4_flat', saved['authoringSourceSHA256'])
                    self.assertFalse(saved['completeModelLayerSet'])
                source.reset_mock()
                with self.assertRaisesRegex(ValueError, '--flat-q4'):
                    main(['--baseline-pd', str(baseline_dir), '--output', str(root/'invalid'),
                          '--contiguous-affine'])
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
