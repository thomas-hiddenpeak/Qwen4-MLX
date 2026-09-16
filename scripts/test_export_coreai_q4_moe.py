"""Small independent CPU checks for packed-bit fidelity and exact routing ties."""
import unittest

import numpy as np
import torch

from export_coreai_q4_moe import PackedQ4, Q4MoE, PROJECTIONS


class PackedQ4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_signed_bitpatterns_and_repeated_selected_ids_match_unsigned_oracle(self):
        rng = np.random.default_rng(23)
        packed = rng.integers(0, 2**32, size=(3, 2, 8), dtype=np.uint32)
        packed[0, 0] = [0, 0xffffffff, 0x80000000, 0x7fffffff, 0x01234567, 0x89abcdef, 0xfedcba98, 0x76543210]
        scales = np.array([[[0.125], [0.03125]], [[0.5], [0.25]], [[0.0625], [0.015625]]], dtype=np.float32)
        biases = -scales * 7
        module = PackedQ4(packed, scales, biases)
        ids = torch.tensor([2, 0, 2, 1], dtype=torch.int32)
        actual = module.selected_dense(ids).numpy()
        codes = np.empty((3, 2, 64), dtype=np.float32)
        for expert in range(3):
            for row in range(2):
                for element in range(64):
                    codes[expert, row, element] = (int(packed[expert, row, element // 8]) >> (4 * (element % 8))) & 15
        dense = (codes * scales + biases).astype(np.float16)
        np.testing.assert_array_equal(actual, dense[ids.numpy()])
        self.assertEqual(module.packed.dtype, torch.int16)
        self.assertEqual(module.packed.numel() * 4, codes.size)
        self.assertEqual(module.packed.numpy().tobytes(), packed.tobytes())
        x = rng.standard_normal((4, 1, 64)).astype(np.float16)
        expected = np.matmul(x.astype(np.float32), dense[ids.numpy()].astype(np.float32).transpose(0, 2, 1)).astype(np.float16)
        np.testing.assert_array_equal(module(torch.from_numpy(x), ids).numpy(), expected)

    def test_router_zero_ties_choose_lowest_ids_and_keep_normalized_scores(self):
        experts, hidden, intermediate, top_k = 12, 64, 64, 10
        rng = np.random.default_rng(17)
        router = rng.standard_normal((experts, hidden)).astype(np.float32) * 0.1
        shared_router = np.zeros((1, hidden), np.float32)
        shared = {name: np.zeros((hidden, intermediate), np.float32) for name in PROJECTIONS}
        packed = rng.integers(0, 2**32, size=(experts, hidden, intermediate // 8), dtype=np.uint32)
        scales = np.full((experts, hidden, intermediate // 64), 0.125, np.float32)
        quantized = {name: (packed, scales, -scales * 7) for name in PROJECTIONS}
        model = Q4MoE(router, shared_router, shared, quantized, top_k)
        output, ids, scores = model(torch.zeros(1, 1, hidden, dtype=torch.float16))
        np.testing.assert_array_equal(ids.numpy().reshape(-1), np.arange(top_k))
        np.testing.assert_array_equal(output.numpy(), np.zeros((1, 1, hidden)))
        self.assertAlmostEqual(float(scores.float().sum()), 1.0, delta=0.002)
        # Unequal logits independently sorted with the source's lowest-ID ties.
        x = torch.from_numpy(rng.standard_normal((1, 1, hidden)).astype(np.float16))
        actual_ids, _ = model.routing(x)
        logits = (x.numpy().astype(np.float32) @ router.astype(np.float16).astype(np.float32).T).astype(np.float16).reshape(-1)
        expected_ids = sorted(range(experts), key=lambda index: (-float(logits[index]), index))[:top_k]
        np.testing.assert_array_equal(actual_ids.numpy(), expected_ids)


if __name__ == "__main__":
    unittest.main()
