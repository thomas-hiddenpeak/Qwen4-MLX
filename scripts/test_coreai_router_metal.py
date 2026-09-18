"""CPU-only ordering/policy controls for the isolated router candidate."""
import unittest
from types import SimpleNamespace

import torch

from coreai_moe_chunk import ChunkQ4MoE
from coreai_router_metal import (boundary_logits, fused_routing, get_router_kernel,
                                order_keys, original_routing, ids_only_routing, DiagnosticProbe)
from coreai_tensor_matmul import tensor_linear


class RouterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_all_finite_half_patterns_follow_numeric_order(self):
        values = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.float16)
        values = values[torch.isfinite(values)]
        permutation = torch.randperm(values.numel(), generator=torch.Generator().manual_seed(12729))
        values = values[permutation].reshape(-1, 512)
        ordered = torch.argsort(values.float(), dim=-1, descending=True, stable=True)
        keys = torch.argsort(order_keys(values), dim=-1, descending=True)
        self.assertTrue(torch.equal(ordered, keys))
        self.assertGreater(int(order_keys(values).min()), 0)

    def test_boundaries_and_signed_zero_choose_lowest_ids(self):
        logits = boundary_logits()
        ids, scores = original_routing(logits)
        self.assertTrue(torch.equal(ids, torch.topk(order_keys(logits), 10, dim=-1).indices.int()))
        for row in (0, 1, 6, 7):
            self.assertTrue(torch.equal(ids[row], torch.arange(10, dtype=torch.int32)))
        self.assertTrue(torch.equal(ids[11], torch.tensor([191, 223, 255, 287, 319, 351, 383, 415, 447, 479], dtype=torch.int32)))
        self.assertTrue(bool(torch.isfinite(scores).all()))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip((ids, scores), fused_routing(logits), strict=True)))
        # Algebraically canceling the full softmax changes the FP16 policy.
        selected = torch.gather(logits, -1, ids.long()).float().softmax(-1).half()
        self.assertFalse(torch.equal(scores, selected))

    def test_reference_matches_existing_chunk_routing(self):
        generator = torch.Generator().manual_seed(8846)
        weight = (torch.randn(512, 64, generator=generator)*.1).half()
        x = (torch.randn(1, 9, 64, generator=generator)*.125).half()
        x[:, 0] = 0
        existing = SimpleNamespace(experts=512, top_k=10,
            decode=SimpleNamespace(router=weight, expert_ids=torch.arange(512, dtype=torch.int32)[None]))
        expected = ChunkQ4MoE.routing(existing, x)
        actual = original_routing(tensor_linear(x, weight).reshape(9, 512))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True)))

    def test_ids_only_and_diagnostic_preserve_cpu_score_chain(self):
        logits = boundary_logits()
        expected = original_routing(logits)
        actual = ids_only_routing(logits)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, expected, strict=True)))
        baseline = DiagnosticProbe(False)(logits)
        candidate = DiagnosticProbe(True)(logits)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(baseline, candidate, strict=True)))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(baseline[:2], expected, strict=True)))

    def test_shape_dtype_and_kernel_identity(self):
        self.assertIs(get_router_kernel(), get_router_kernel())
        for bad in (torch.zeros(1, 511).half(), torch.zeros(1, 512), torch.zeros(0, 512).half()):
            with self.assertRaises(ValueError):
                fused_routing(bad)
        with self.assertRaisesRegex(ValueError, 'Finite'):
            order_keys(torch.full((1, 512), float('inf'), dtype=torch.float16))


if __name__ == '__main__':
    unittest.main()
