"""Bounded CPU checks for the independent native-parity dequant experiment."""
import unittest

import torch

from coreai_expert_grouping import enable_integer_grouping
from coreai_moe_chunk import ChunkQ4MoE, make_synthetic
from coreai_moe_dequant_parity import ParityDequantOnceChunkMoE, parity_source
from coreai_moe_transfers import install_moe_transfers
from coreai_q4_flat import flatten_moe_weights
from coreai_q4_nax import install_nax_moe


def make_pair(threshold=4):
    base = ChunkQ4MoE(make_synthetic(), block=16, columns=32, inner=64, fuse_gateup=True).eval()
    flatten_moe_weights(base)
    enable_integer_grouping(base)
    install_moe_transfers(base, tail_precision='native-copy')
    install_nax_moe(base, projections='all', down_block=32, gateup_policy='native-parity-v2')
    return base, ParityDequantOnceChunkMoE(base, minimum_chunk=threshold, block=32, columns=64).eval()


class ParityDequantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_storage_and_subthreshold_path_unchanged(self):
        old, new = make_pair()
        self.assertEqual([(n, v.dtype, tuple(v.shape), v.data_ptr()) for n, v in old.named_buffers()],
                         [(n, v.dtype, tuple(v.shape), v.data_ptr()) for n, v in new.named_buffers()])
        x = (torch.randn(1, 3, 64, generator=torch.Generator().manual_seed(741))*.125).half()
        for count in (1, 3):
            for actual, expected in zip(new(x[:, :count]), old(x[:, :count]), strict=True):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_whole_moe_matches_measured_parity_cpu_reference(self):
        old, new = make_pair()
        for count in (4, 17):
            x = (torch.randn(1, count, 64, generator=torch.Generator().manual_seed(997))*.5).half()
            x[:, 0] = 0
            actual, expected = new(x), old(x)
            torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)
            torch.testing.assert_close(actual[2], expected[2], atol=0, rtol=0)
            torch.testing.assert_close(actual[0], expected[0], atol=2e-5, rtol=1e-3)
            self.assertTrue(torch.isfinite(actual[0]).all())

    def test_export_orders_three_banks_and_preserves_up_accumulator(self):
        _, model = make_pair()
        # Registration precedes tracing so CPU/export order cannot deregister an
        # identically named custom op through a second default-argument cache key.
        model.custom_kernels()
        ep = torch.export.export(model, (torch.zeros(1, 4, 64, dtype=torch.float16),))
        nodes = [n for n in ep.graph.nodes if n.op == 'call_function']
        dequant = [n for n in nodes if 'qwen_q4_dequant_sequence_' in str(n.target)]
        up = [n for n in nodes if 'qwen_dequant_up_native_parity_' in str(n.target)]
        gemm = [n for n in nodes if 'qwen_dequant_gemm_' in str(n.target)]
        self.assertEqual((len(dequant), len(up), len(gemm)), (3, 1, 2))
        self.assertIn(gemm[0], dequant[1].all_input_nodes)
        self.assertIn(up[0], dequant[2].all_input_nodes)
        self.assertIn(gemm[0], up[0].all_input_nodes)
        source = parity_source()
        self.assertIn('half(g*accum[i])', source)
        self.assertEqual(source.count('volatile thread half'), 3)
        self.assertNotIn('half(accum[i])', source)


if __name__ == '__main__':
    unittest.main()
