"""CPU-only dense-block equations and dilated PLE state checks."""
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from export_coreai_dense import DenseConfig, Embedding, HCRead, HCWrite, Head, PLE


class DenseExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def config(self):
        return DenseConfig(hidden=3, streams=2, low_rank=2, epsilon=1e-6,
                           vocabulary=7, ple_dim=3, ple_kernel=3, ple_dilation=2)

    def hc_weights(self, c):
        rng = np.random.default_rng(94)
        return {name: (0.2 * rng.standard_normal(shape)).astype(np.float32) for name, shape in {
            "input_mix_weight_down.weight": (c.low_rank, c.width),
            "input_mix_weight_up.weight": (c.width, c.low_rank),
            "hc_norm.weight": (c.width,), "block_inject_weight.weight": (c.streams, c.width)}.items()}

    def test_hc_read_write_match_numpy_with_explicit_fp16_boundaries(self):
        c = self.config()
        w = self.hc_weights(c)
        x = np.random.default_rng(37).standard_normal((1, 1, c.width)).astype(np.float16)
        half = lambda v: np.asarray(v).astype(np.float16).astype(np.float32)
        sigmoid = lambda v: half(1 / (1 + np.exp(-np.asarray(v, dtype=np.float32))))
        xx = x.astype(np.float32).reshape(1, 1, c.streams, c.hidden)
        n = half(xx / np.sqrt(np.mean(xx ** 2, axis=-1, keepdims=True) + c.epsilon))
        n = half(n * half(w['hc_norm.weight']).reshape(c.streams, c.hidden))
        flat = n.reshape(1, 1, c.width)
        down = half(flat @ half(half(w['input_mix_weight_down.weight']) / c.streams).T)
        act = half(down * sigmoid(down))
        up = half(act @ half(w['input_mix_weight_up.weight']).T).reshape(1, 1, c.streams, c.hidden)
        expected_mixed = half(half(n * sigmoid(up)).mean(2))
        inject_logits = half(flat @ half(half(w['block_inject_weight.weight']) / c.streams).T)
        expected_injection = half(sigmoid(inject_logits) * 2).reshape(1, 1, c.streams, 1)
        model = HCRead(c, w)
        with torch.inference_mode():
            mixed, injection = model(torch.from_numpy(x))
            written = HCWrite(c)(torch.from_numpy(x), mixed, injection)
        np.testing.assert_allclose(mixed.float().numpy(), expected_mixed, rtol=0, atol=1e-4)
        np.testing.assert_array_equal(injection.float().numpy(), expected_injection)
        expected_write = half(xx + half(expected_mixed[:, :, None] * expected_injection)).reshape(x.shape)
        np.testing.assert_allclose(written.float().numpy(), expected_write, rtol=0, atol=1e-4)

    def test_ple_dilation_and_history_match_independent_grouped_conv(self):
        c = self.config()
        weights = {"key_proj.weight": np.zeros((c.width, c.ple_dim), np.float32),
                   "value_proj.weight": np.eye(c.hidden, dtype=np.float32),
                   "norm_key.weight": np.ones(c.width, np.float32),
                   "norm_query.weight": np.ones(c.width, np.float32),
                   "norm_conv.weight": np.ones(c.width, np.float32),
                   "conv1d.weight": np.arange(c.width * c.ple_kernel, dtype=np.float32).reshape(c.width, c.ple_kernel, 1) / 50}
        model = PLE(c, weights)
        stream = torch.ones(1, 1, c.width, dtype=torch.float16)
        history = torch.arange(c.ple_history * c.width, dtype=torch.float32).reshape(1, c.ple_history, c.width).half() / 20
        for step in range(5):
            embedding = torch.tensor([[[0.5 + step / 5, -0.75, 1.0]]], dtype=torch.float16)
            # Zero key projection fixes sigmoid(gate)=0.5, independent of stream.
            gated = (embedding * 0.5).repeat(1, 1, c.streams).half()
            v = gated.reshape(1, 1, c.streams, c.hidden).float()
            normalized = (v / torch.sqrt(v.square().mean(-1, keepdim=True) + c.epsilon)).half().reshape(1, 1, c.width)
            joined = torch.cat((history, normalized), 1)
            conv = F.conv1d(joined.float().transpose(1, 2), model.conv1d_weight.float().transpose(1, 2),
                            dilation=c.ple_dilation, groups=c.width).transpose(1, 2).half()
            expected = (stream + (gated + conv * conv.sigmoid()).half()).half()
            with torch.inference_mode():
                actual, next_history = model(stream, embedding, history)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0.001)
            torch.testing.assert_close(next_history, joined[:, 1:], rtol=0, atol=0)
            history = next_history

    def test_embedding_and_head_shapes_and_mixer_are_preserved(self):
        c = self.config()
        table = np.arange(c.vocabulary * c.hidden, dtype=np.float32).reshape(c.vocabulary, c.hidden) / 10
        embedding = Embedding(c, table)
        token = torch.tensor([3], dtype=torch.int32)
        stream = embedding(token)
        torch.testing.assert_close(stream, torch.from_numpy(table[3]).half().repeat(c.streams).reshape(1, 1, c.width), rtol=0, atol=0)
        weights = self.hc_weights(c)
        head = Head(c, weights, table)
        expected = F.linear(HCRead(c, weights, with_injection=False)(stream).float(), torch.from_numpy(table).half().float())
        actual = head(stream)
        self.assertEqual(actual.shape, (1, 1, c.vocabulary))
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
