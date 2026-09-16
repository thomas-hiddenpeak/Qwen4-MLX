"""CPU checks for stable integer grouping and optional MoE integration."""
from pathlib import Path
import tempfile
import unittest

import torch

from coreai_expert_grouping import (enable_integer_grouping, integer_grouping,
    histogram_reference, prefix_reference, get_integer_grouping_kernels)
from coreai_moe_chunk import ChunkQ4MoE, grouping_permutations, make_synthetic
from coreai_q4_flat import flatten_moe_weights
from export_coreai_pd_shared import (externalizable_buffers, export_generic,
                                    resolve_phases, validate_baseline, authoring_source_hashes)
from export_moe import sha256_file


class IntegerGroupingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def assert_grouping(self, ids, experts):
        actual = integer_grouping(ids, experts)
        permutation = torch.argsort(ids, stable=True).int()
        inverse = torch.empty_like(permutation).scatter(0, permutation.long(), torch.arange(ids.numel(), dtype=torch.int32))
        expected = permutation, inverse, ids[permutation.long()]
        for a, b in zip(actual, expected, strict=True):
            self.assertEqual(a.dtype, torch.int32)
            self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(actual[0][actual[1].long()], torch.arange(ids.numel(), dtype=torch.int32)))

    def test_partial_blocks_duplicates_and_unused_experts(self):
        generator = torch.Generator().manual_seed(8716)
        for count in (1, 17, 255, 256, 257, 529):
            ids = torch.randint(0, 13, (count,), generator=generator, dtype=torch.int32)
            ids[::7] = 511
            self.assert_grouping(ids, 512)
        self.assert_grouping(torch.full((529,), 511, dtype=torch.int32), 512)
        self.assert_grouping(torch.arange(529, dtype=torch.int32).flip(0) % 512, 512)

    def test_large_chunks_do_not_need_float_exact_keys(self):
        generator = torch.Generator().manual_seed(15297)
        for tokens in (4096, 8192):
            ids = torch.randint(0, 512, (tokens * 10,), generator=generator, dtype=torch.int32)
            with self.assertRaisesRegex(ValueError, 'FP32 integer'):
                grouping_permutations(ids, 512)
            self.assert_grouping(ids, 512)

    def test_prefix_offsets_are_exclusive_and_cover_all_assignments(self):
        ids = torch.tensor([2, 2, 0, 4, 2, 0] * 100, dtype=torch.int32)
        counts = histogram_reference(ids, 5, 256)
        offsets = prefix_reference(counts)
        self.assertEqual(int(counts.sum()), ids.numel())
        for expert in range(5):
            prior_experts = int(counts[:, :expert].sum())
            for block in range(counts.shape[0]):
                self.assertEqual(int(offsets[block, expert]), prior_experts + int(counts[:block, expert].sum()))
        with self.assertRaisesRegex(ValueError, 'out of range'):
            integer_grouping(torch.tensor([512], dtype=torch.int32), 512)

    def test_optional_moe_matches_previous_routes_scores_and_outputs(self):
        generator = torch.Generator().manual_seed(675)
        xs = [(torch.randn(1, count, 64, generator=generator) * .125).half() for count in (1, 4, 29)]
        for fused in (False, True):
            for flat in (False, True):
                with self.subTest(fused=fused, flat=flat), torch.inference_mode():
                    model = ChunkQ4MoE(make_synthetic(), columns=32, inner=64, fuse_gateup=fused).eval()
                    if flat:
                        flatten_moe_weights(model)
                    expected = [model(x) for x in xs]
                    enable_integer_grouping(model)
                    for x, outputs in zip(xs, expected, strict=True):
                        for actual, wanted in zip(model(x), outputs, strict=True):
                            torch.testing.assert_close(actual, wanted, atol=0, rtol=0)

    def test_shared_large_chunk_export_captures_no_weights(self):
        model = ChunkQ4MoE(make_synthetic(), columns=32, inner=64, fuse_gateup=True).eval()
        flatten_moe_weights(model)
        enable_integer_grouping(model)
        named, geometry = externalizable_buffers(model, {'decode.expert_ids'})
        examples = {'main': {'x': torch.zeros(1, 1, 64, dtype=torch.float16)},
                    'prefill': {'x': torch.zeros(1, 4096, 64, dtype=torch.float16)}}
        with tempfile.TemporaryDirectory() as directory:
            result = export_generic(model, named, geometry, examples, ('output', 'ids', 'scores'),
                Path(directory) / 'large.aimodel', model.custom_kernels())
            self.assertLess(result['modelBytes'], 150_000)
            for stats in result['torchExport'].values():
                self.assertTrue(set(stats['usedCapturedBuffers']) <= {'base.decode.expert_ids'})
                self.assertEqual(stats['userInputCount'], len(named) + 1)
            for kernel in get_integer_grouping_kernels(model.experts):
                self.assertTrue(kernel.kernel_cache)

    def test_large_shared_phases_inherit_v1_geometry_and_require_integer_flag(self):
        baseline = {'version': 1, 'backend': 'native-coreai-pd', 'status': 'complete',
            'completeModelLayerSet': True, 'prefillKernels': 'tensor', 'q4Kernel': 'metal',
            'stableProjections': False, 'layers': [{'index': i} for i in range(48)],
            'tokenChunk': 2048, 'tailChunks': [4, 16, 64, 512, 1024], 'capacity': 16384}
        self.assertEqual(resolve_phases(baseline), validate_baseline(baseline))
        phases = resolve_phases(baseline, 4096, integer_grouping=True)
        self.assertEqual(dict(phases)['prefill'], 4096)
        self.assertEqual(dict(phases)['prefill_s2048'], 2048)
        self.assertEqual(dict(phases)['prefill_s512'], 512)
        self.assertEqual(dict(resolve_phases(baseline, 8192, integer_grouping=True))['prefill_s4096'], 4096)
        self.assertEqual(baseline['tokenChunk'], 2048)
        for size in (4096, 8192):
            with self.assertRaisesRegex(ValueError, 'integer-grouping'):
                resolve_phases(baseline, size)
        with self.assertRaisesRegex(ValueError, 'capacity'):
            resolve_phases({**baseline, 'capacity': 4096}, 8192, integer_grouping=True)
        with self.assertRaisesRegex(ValueError, 'baseline phase'):
            validate_baseline({**baseline, 'tokenChunk': 4096})
        self.assertEqual(authoring_source_hashes()['coreai_expert_grouping'],
                         sha256_file(Path(__file__).with_name('coreai_expert_grouping.py')))


if __name__ == '__main__':
    unittest.main()
