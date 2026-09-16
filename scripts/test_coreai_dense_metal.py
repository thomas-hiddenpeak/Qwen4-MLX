"""CPU-only semantic and graph checks for optional stable dense Metal GEMV."""
import unittest

import numpy as np
import torch

from coreai_dense_metal import dense_linear, dense_reference, make_smoke


def lane_oracle(x, weight):
    """NumPy FP32 lane accumulation and balanced 32-way reduction."""
    x, weight = x.numpy().astype(np.float32)[0], weight.numpy().astype(np.float32)
    accum = np.zeros((x.shape[0], weight.shape[0], 32), np.float32)
    for start in range(0, x.shape[-1], 32):
        count = min(32, x.shape[-1] - start)
        accum[..., :count] += x[:, None, start:start+count] * weight[None, :, start:start+count]
    while accum.shape[-1] > 1:
        accum = accum[..., ::2] + accum[..., 1::2]
    return accum[..., 0][None].astype(np.float16)


class DenseMetalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_lane_reduction_is_token_independent_and_close_to_fp32_linear(self):
        for inputs, outputs in ((1, 1), (31, 7), (129, 7), (640, 2560), (2560, 641)):
            with self.subTest(inputs=inputs, outputs=outputs):
                model, x = make_smoke(inputs, outputs)
                actual = lane_oracle(x, model.weight)
                singles = np.concatenate([lane_oracle(x[:, t:t+1], model.weight) for t in range(4)], axis=1)
                np.testing.assert_array_equal(actual, singles)
                expected = dense_reference(x, model.weight).numpy()
                delta = actual.astype(np.float64) - expected.astype(np.float64)
                self.assertLessEqual(float(np.abs(delta).max()), 0.002)
                self.assertLessEqual(float(np.linalg.norm(delta) / np.linalg.norm(expected.astype(np.float64))), 0.001)

    def test_export_keeps_opaque_kernel_for_s1_and_s4(self):
        model, x = make_smoke()
        for count in (1, 4):
            sample = x[:, :count].contiguous()
            graph = torch.export.export(model, (sample,))
            ops = [str(node.target) for node in graph.graph.nodes if node.op == 'call_function']
            self.assertEqual(sum('qwen_fp16_dense_gemv_stable_v1' in name for name in ops), 1)
            self.assertFalse(any('matmul' in name or 'linear' in name for name in ops))
            torch.testing.assert_close(graph.module()(sample), dense_reference(sample, model.weight), atol=0, rtol=0)

    def test_weight_is_shared_and_invalid_shapes_or_dtypes_fail(self):
        model, x = make_smoke()
        self.assertEqual(model.weight.dtype, torch.float16)
        self.assertEqual(model.weight.data_ptr(), dict(model.named_buffers())['weight'].data_ptr())
        for invalid_x, invalid_weight in ((x.float(), model.weight), (x, model.weight.float()),
                                           (x.expand(2,-1,-1), model.weight), (x[..., :-1], model.weight)):
            with self.assertRaises(ValueError):
                dense_linear(invalid_x, invalid_weight)
        torch.testing.assert_close(model(torch.zeros_like(x)), torch.zeros(1,4,7,dtype=torch.float16), atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
