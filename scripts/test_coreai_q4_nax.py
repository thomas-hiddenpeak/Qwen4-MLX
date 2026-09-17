"""CPU integration controls for optional NAX projection dispatch; no GPU."""
import tempfile
import itertools
from pathlib import Path
import unittest

import torch

from coreai_moe_chunk import ChunkQ4MoE, make_synthetic
from coreai_q4_flat import flatten_moe_weights
from coreai_q4_nax import check_fragment_mapping, get_kernel, install_nax_moe
from coreai_q4_nax_gateup import get_kernel as get_gateup_kernel
from export_coreai_pd_shared import buffer_signature, externalizable_buffers, export_generic


class NAXTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self, fused=True):
        return ChunkQ4MoE(make_synthetic(), block=16, columns=32, inner=64, fuse_gateup=fused).eval()

    def test_fragment_coverage_and_canonical_kernel_registration(self):
        check_fragment_mapping()
        self.assertIs(get_kernel(12, 64, 128), get_kernel(12, 64, 128, 1, False))
        self.assertIs(get_kernel(12, 64, 128, pointers=True), get_kernel(12, 64, 128, 1, True))
        self.assertIs(get_gateup_kernel(12, 128, 64), get_gateup_kernel(12, 128, 64, 1))

    def test_optional_install_preserves_views_and_all_cpu_phases(self):
        generator = torch.Generator().manual_seed(32181)
        inputs = [(torch.randn(1, count, 64, generator=generator)*.125).half() for count in (1, 4, 17)]
        inputs[-1][:, 0] = 0
        for fused, projections in itertools.product((False, True), ('down', 'all')):
            with self.subTest(fused=fused, projections=projections), torch.inference_mode():
                model = self.model(fused)
                self.assertFalse(model.nax_moe)
                flatten_moe_weights(model)
                before = [(n, v.dtype, tuple(v.shape), v.data_ptr()) for n, v in model.named_buffers()]
                expected = [model(x) for x in inputs]
                kernels = install_nax_moe(model, projections=projections)
                self.assertEqual(kernels, install_nax_moe(model, projections=projections))
                if projections == 'down':
                    self.assertEqual(len(kernels), 1)
                    self.assertIs(kernels[0], get_kernel(*model.decode.down_proj.geometry, 1, True))
                self.assertTrue(set(kernels) <= set(model.custom_kernels()))
                after = [(n, v.dtype, tuple(v.shape), v.data_ptr()) for n, v in model.named_buffers()]
                self.assertEqual(before, after)
                for x, wanted in zip(inputs, expected, strict=True):
                    for actual, target in zip(model(x), wanted, strict=True):
                        torch.testing.assert_close(actual, target, atol=0, rtol=0)

    def test_nonflat_and_wrong_plan_rejected_before_enabling(self):
        model = self.model()
        with self.assertRaisesRegex(ValueError, 'flattened'):
            install_nax_moe(model)
        self.assertFalse(model.nax_moe)
        flatten_moe_weights(model)
        with self.assertRaisesRegex(ValueError, 'projections'):
            install_nax_moe(model, projections='gate')
        self.assertFalse(model.nax_moe)
        model.block = 32
        with self.assertRaisesRegex(ValueError, 'BM16'):
            install_nax_moe(model)
        self.assertFalse(model.nax_moe)

    def test_default_down_does_not_dispatch_gateup_or_decode(self):
        from unittest.mock import patch
        model = self.model()
        flatten_moe_weights(model)
        install_nax_moe(model)
        with patch('coreai_q4_nax.nax_grouped_gateup', side_effect=AssertionError('gate/up must stay original')):
            model(torch.zeros(1, 4, 64).half())
        with patch('coreai_q4_nax.nax_grouped_linear', side_effect=AssertionError('decode must stay original')):
            model(torch.zeros(1, 1, 64).half())

    def test_shared_export_keeps_signature_and_s1_separate(self):
        model = self.model()
        flatten_moe_weights(model)
        named, geometry = externalizable_buffers(model, {'decode.expert_ids'})
        signature = buffer_signature(named)
        install_nax_moe(model)
        examples = {name: {'x': torch.zeros(1, count, 64).half()}
                    for name, count in (('main', 1), ('prefill', 4))}
        with tempfile.TemporaryDirectory() as directory:
            result = export_generic(model, named, geometry, examples, ('output', 'ids', 'scores'),
                Path(directory)/'nax.aimodel', model.custom_kernels())
            self.assertEqual(result['weightSignature'], signature)
            self.assertTrue(all(set(row['usedCapturedBuffers']) <= {'base.decode.expert_ids'}
                                for row in result['torchExport'].values()))
            self.assertLess(result['modelBytes'], 150000)


if __name__ == '__main__':
    unittest.main()
