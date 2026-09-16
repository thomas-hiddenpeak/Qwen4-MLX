"""CPU-only rank-one Q4 storage/phase and authoring contract checks."""
import tempfile
import unittest
from pathlib import Path
import re

import torch

from coreai_moe_chunk import ChunkQ4MoE, make_synthetic
from coreai_q4_flat import (FlatMetalPackedQ4, flatten_moe_weights, _grouped_source,
                           install_contiguous_affine, get_flat_grouped_kernel, get_flat_gateup_kernel)
from coreai_q4_grouped import GEMM_SOURCE
from coreai_q4_gateup import GATEUP_SOURCE
from export_coreai_pd_shared import externalizable_buffers, export_generic, authoring_source_hashes
from export_moe import sha256_file


class FlatQ4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self, fused=True):
        return ChunkQ4MoE(make_synthetic(), block=16, columns=32, inner=64, fuse_gateup=fused).eval()

    def test_storage_and_signature_stay_identical(self):
        model = self.model()
        before = list(model.named_buffers())
        kernels = flatten_moe_weights(model)
        after = list(model.named_buffers())
        self.assertTrue(kernels)
        self.assertEqual([name for name, _ in before], [name for name, _ in after])
        changed = []
        for (name, a), (_, b) in zip(before, after, strict=True):
            self.assertEqual((a.dtype, a.numel(), a.data_ptr()), (b.dtype, b.numel(), b.data_ptr()))
            torch.testing.assert_close(a.flatten(), b.flatten(), atol=0, rtol=0)
            if a.shape != b.shape:
                changed.append(name)
                self.assertEqual(b.ndim, 1)
        self.assertEqual(len(changed), 9)
        self.assertEqual(kernels, flatten_moe_weights(model))
        self.assertTrue(all(isinstance(getattr(model.decode, name), FlatMetalPackedQ4)
                            for name in ('gate_proj', 'up_proj', 'down_proj')))

    def test_all_phases_and_fusion_match_original_cpu(self):
        generator = torch.Generator().manual_seed(7163)
        inputs = [(torch.randn(1, size, 64, generator=generator) * .125).half() for size in (1, 4, 17)]
        inputs[1][:, 0] = 0  # An exact router tie still picks the same experts.
        for fused in (False, True):
            with self.subTest(fused=fused), torch.inference_mode():
                model = self.model(fused)
                expected = [model(x) for x in inputs]
                flatten_moe_weights(model)
                for x, reference in zip(inputs, expected, strict=True):
                    for actual, wanted in zip(model(x), reference, strict=True):
                        torch.testing.assert_close(actual, wanted, atol=0, rtol=0)

    def test_external_shared_export_s1_and_chunk(self):
        model = self.model()
        flatten_moe_weights(model)
        named, _ = externalizable_buffers(model, {'decode.expert_ids'})
        examples = {'main': {'x': torch.zeros(1, 1, 64, dtype=torch.float16)},
                    'prefill': {'x': torch.zeros(1, 4, 64, dtype=torch.float16)}}
        with tempfile.TemporaryDirectory() as directory:
            result = export_generic(model, named, {'decode.expert_ids'}, examples,
                ('output', 'ids', 'scores'), Path(directory) / 'flat.aimodel', model.custom_kernels())
            self.assertLess(result['modelBytes'], 100_000)
            weights = {row['bufferName']: row for row in result['weightSignature']}
            for projection in ('gate_proj', 'up_proj', 'down_proj'):
                for name in ('packed', 'scales', 'biases'):
                    self.assertEqual(len(weights[f'decode.{projection}.{name}']['shape']), 1)
            for phase in ('main', 'prefill'):
                self.assertTrue(set(result['torchExport'][phase]['usedCapturedBuffers']) <= {'base.decode.expert_ids'})
                self.assertEqual(result['torchExport'][phase]['userInputCount'], len(named) + 1)
            # These are actual emitted MSL parameter declarations. The CPU-only
            # oracle's rank-three reshape must not reach the exported shader.
            for kernel in flatten_moe_weights(model):
                for _, source in kernel.kernel_cache.values():
                    if 'packed' in source:
                        self.assertIn('metal::dextents<int, 1>, tensor_handle>', source)
                        for line in source.splitlines():
                            if '[[kernel]]' in line:
                                self.assertNotIn('metal::dextents<int, 3>, tensor_handle> packed', line)

    def test_all_weight_addresses_replaced_or_fail_closed(self):
        for original, prefixes in ((GEMM_SOURCE, ('',)), (GATEUP_SOURCE, ('gate_', 'up_'))):
            source = _grouped_source(original, 512, 640, 2560, 16, 32, 64, prefixes)
            self.assertIn('const int K=2560, N=640;', source)
            for prefix in prefixes:
                self.assertIn(f'{prefix}packed[(expert*N+n)*(K/4)+k/4]', source)
                for name in ('scales', 'biases'):
                    self.assertIn(f'{prefix}{name}[(expert*N+n)*(K/64)+k/64]', source)
            for address in re.findall(r'(?:\w*packed|\w*scales|\w*biases)\[([^\]]*)\]', source):
                self.assertNotIn(',', address)
            with self.assertRaisesRegex(ValueError, 'expected exactly one'):
                _grouped_source(original.replace('k/4,n,expert', 'k / 4,n,expert'),
                                512, 640, 2560, 16, 32, 64, prefixes)

    def test_shared_exporter_records_flat_source(self):
        self.assertEqual(authoring_source_hashes()['coreai_q4_flat'],
                         sha256_file(Path(__file__).with_name('coreai_q4_flat.py')))

    def test_contiguous_affine_is_optional_preserves_storage_and_phase_outputs(self):
        for fused in (False, True):
            model = self.model(fused)
            self.assertFalse(model.contiguous_affine)
            flatten_moe_weights(model)
            before = [(n, v.dtype, tuple(v.shape), v.data_ptr()) for n, v in model.named_buffers()]
            inputs = [torch.zeros(1, size, 64, dtype=torch.float16) for size in (1, 17)]
            inputs[-1][0, 1, 0] = .5
            inputs[-1][0, 2, 0] = -.5
            with torch.inference_mode():
                expected = [model(x) for x in inputs]
                kernels = install_contiguous_affine(model)
                self.assertEqual(kernels, install_contiguous_affine(model))
                self.assertTrue(set(kernels) <= set(model.custom_kernels()))
                for x, reference in zip(inputs, expected, strict=True):
                    for actual, wanted in zip(model(x), reference, strict=True):
                        torch.testing.assert_close(actual, wanted, atol=0, rtol=0)
            self.assertEqual(before, [(n, v.dtype, tuple(v.shape), v.data_ptr()) for n, v in model.named_buffers()])

    def test_contiguous_registration_normalizes_defaults_and_rejects_bk128(self):
        for getter in (get_flat_grouped_kernel, get_flat_gateup_kernel):
            self.assertIs(getter(12, 128, 64), getter(12, 128, 64, 16, 32, 64, False))
            self.assertIs(getter(12, 128, 64, contiguous_affine=True),
                          getter(12, 128, 64, 16, 32, 64, True))
            with self.assertRaisesRegex(ValueError, 'BK=64'):
                getter(12, 128, 64, inner=128, contiguous_affine=True)
        model = self.model()
        with self.assertRaisesRegex(ValueError, 'flattened'):
            install_contiguous_affine(model)


if __name__ == '__main__':
    unittest.main()
