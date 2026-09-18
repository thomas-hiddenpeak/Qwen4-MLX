"""Bounded CPU checks for generic non-power-of-two prefill tail shapes."""
import copy
import unittest

import torch

from coreai_moe_chunk import ChunkQ4MoE, make_synthetic
from coreai_moe_transfers import install_moe_transfers
from coreai_q4_flat import flatten_moe_weights
from coreai_expert_grouping import enable_integer_grouping
from export_coreai_radix4_tails import for_primary


class Radix4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_s48_s768_native_copy_matches_cpu_native_tail(self):
        baseline = ChunkQ4MoE(make_synthetic(), block=16, columns=32, inner=64, fuse_gateup=True)
        flatten_moe_weights(baseline)
        enable_integer_grouping(baseline)
        candidate = copy.copy(baseline)
        install_moe_transfers(candidate, tail_precision='native-copy')
        generator = torch.Generator().manual_seed(48)
        x = (torch.randn(1, 768, 64, generator=generator) * .125).half()
        x[:, ::31] = 0  # Include exact routing ties at different expert row positions.
        for count in (48, 768):
            with self.subTest(count=count):
                for actual, expected in zip(candidate(x[:, :count]), baseline(x[:, :count])):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_two_primary_manifests_reuse_assets_and_weights(self):
        original = {'tokenChunk': 4096, 'tailChunks': [4, 16, 48, 192, 768, 2048],
            'qsaWorkingSets': [{'tokenCount': 2048}, {'tokenCount': 4096}],
            'assets': {'head': {'path': 'head.aimodel', 'prefillFunction': 'prefill'}},
            'sharedAssets': {'gdn': {'path': 'gdn.aimodel', 'prefillFunction': 'prefill'}},
            'layers': [{'path': 'gdn.aimodel', 'prefillFunction': 'prefill',
                        'weights': {'path': 'same.weights.bin', 'sha256': 'unchanged'}}]}
        derived = for_primary(original, 2048)
        self.assertEqual(derived['assets']['head']['prefillFunction'], 'prefill_s2048')
        self.assertEqual(derived['layers'][0]['prefillFunction'], 'prefill_s2048')
        self.assertEqual(derived['layers'][0]['weights'], original['layers'][0]['weights'])
        self.assertEqual(derived['tailChunks'], [4, 16, 48, 192, 768])
        self.assertEqual(derived['qsaWorkingSets'], [{'tokenCount': 2048}])
        self.assertEqual(original['tokenChunk'], 4096)
        self.assertEqual(for_primary(original, 4096), original)


if __name__ == '__main__': unittest.main()
