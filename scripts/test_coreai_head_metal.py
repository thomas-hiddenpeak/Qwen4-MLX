"""CPU-only checks for optional FP32-output Metal vocabulary projection."""
import unittest

import numpy as np
import torch

from coreai_head_metal import MetalHead, MetalHeadProjection, head_linear, head_reference
from export_coreai_dense import DenseConfig, Head
from export_coreai_pd import LastHead


def lane_oracle(x, weight):
    x, weight = x.numpy().astype(np.float32)[0], weight.numpy().astype(np.float32)
    accum = np.zeros((x.shape[0], weight.shape[0], 32), np.float32)
    for start in range(0, x.shape[-1], 32):
        count = min(32, x.shape[-1] - start)
        accum[..., :count] += x[:, None, start:start+count] * weight[None, :, start:start+count]
    while accum.shape[-1] > 1:
        accum = accum[..., ::2] + accum[..., 1::2]
    return accum[..., 0][None]


class HeadMetalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_real_inner_dimension_and_partial_rows_keep_fp32(self):
        rng = np.random.default_rng(8027)
        for k, n in ((1, 1), (31, 7), (129, 9), (2560, 641)):
            with self.subTest(k=k, n=n):
                x = torch.from_numpy(rng.normal(0, .3, (1, 4, k)).astype(np.float16))
                w = torch.from_numpy(rng.normal(0, .04, (n, k)).astype(np.float16))
                expected = head_reference(x, w)
                actual = lane_oracle(x, w)
                self.assertEqual(expected.dtype, torch.float32)
                np.testing.assert_allclose(actual, expected.numpy(), atol=2e-6, rtol=1e-5)
                np.testing.assert_array_equal(actual, np.concatenate([
                    lane_oracle(x[:, t:t+1], w) for t in range(4)], axis=1))
                self.assertTrue(np.any(expected.numpy() != expected.half().float().numpy()))

    def test_hc_and_last_token_semantics_are_unchanged(self):
        c = DenseConfig(8, 4, 3, 1e-6, 19, 2, 3, 2)
        rng = np.random.default_rng(8041)
        weights = {"input_mix_weight_down.weight": rng.normal(0, .1, (c.low_rank, c.width)),
                   "input_mix_weight_up.weight": rng.normal(0, .1, (c.width, c.low_rank)),
                   "hc_norm.weight": np.ones(c.width)}
        old = Head(c, weights, rng.normal(0, .1, (c.vocabulary, c.hidden))).eval()
        new = MetalHead(old).eval()
        self.assertIs(new.mixer, old.mixer)
        self.assertEqual(new.weight.data_ptr(), old.weight.data_ptr())
        x = torch.from_numpy(rng.normal(0, .3, (1, 4, c.width)).astype(np.float16))
        torch.testing.assert_close(LastHead(new)(x), LastHead(old)(x), atol=0, rtol=0)
        torch.testing.assert_close(LastHead(new)(x), new(x[:, -1:].contiguous()), atol=0, rtol=0)

    def test_export_is_opaque_and_has_float_output(self):
        w = torch.randn(7, 129).half()
        module = MetalHeadProjection(w)
        x = torch.randn(1, 1, 129).half()
        exported = torch.export.export(module, (x,))
        ops = [str(node.target) for node in exported.graph.nodes if node.op == "call_function"]
        self.assertEqual(sum("qwen_fp16_head_gemv_fp32_output_v1" in op for op in ops), 1)
        self.assertFalse(any("linear" in op or "matmul" in op or "_to_copy" in op for op in ops))
        y = exported.module()(x)
        self.assertEqual(y.dtype, torch.float32)
        torch.testing.assert_close(y, head_reference(x, w), atol=0, rtol=0)

    def test_invalid_inputs_fail(self):
        x, w = torch.zeros(1, 1, 8).half(), torch.zeros(7, 8).half()
        for bad_x, bad_w in ((x.float(), w), (x, w.float()), (x.expand(2, -1, -1), w), (x[..., :-1], w)):
            with self.assertRaises(ValueError):
                head_linear(bad_x, bad_w)


if __name__ == "__main__":
    unittest.main()
