"""CPU-only GDN state/layout checks with tiny asymmetric dimensions."""
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from export_coreai_gdn import GDN, GDNConfig


class GDNExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self):
        config = GDNConfig(hidden=5, key_heads=2, value_heads=6, key_dim=3, value_dim=4, kernel=4, epsilon=1e-6)
        rng = np.random.default_rng(37)
        weights = {name: (rng.standard_normal(shape) * 0.2).astype(np.float32)
                   for name, shape in config.weight_shapes.items()}
        return GDN(config, weights, dtype=torch.float32).eval()

    def inputs(self, model, length=7):
        c = model.config
        generator = torch.Generator().manual_seed(71)
        return (torch.randn(1, length, c.hidden, generator=generator),
                torch.randn(1, c.kernel - 1, c.channels, generator=generator) * 0.1,
                torch.randn(1, c.value_heads, c.value_dim, c.key_dim, generator=generator) * 0.1)

    def independent_reference(self, model, hidden, history, state):
        """Use grouped Conv1d and per-head matrix algebra instead of broadcasts."""
        c = model.config
        qkv = F.linear(hidden, model.in_proj_qkv_weight)
        joined = torch.cat((history, qkv), dim=1)
        conv = F.conv1d(joined.transpose(1, 2), model.conv1d_weight.transpose(1, 2), groups=c.channels).transpose(1, 2)
        conv = F.silu(conv)
        z = F.linear(hidden, model.in_proj_z_weight).reshape(1, -1, c.value_heads, c.value_dim)
        a, b = F.linear(hidden, model.in_proj_a_weight), F.linear(hidden, model.in_proj_b_weight)
        decay = torch.exp(-model.A_log.exp() * F.softplus(a + model.dt_bias))
        beta = b.sigmoid()
        result = []
        state = state.clone()
        width = c.key_heads * c.key_dim
        for t in range(hidden.shape[1]):
            out = []
            for hv in range(c.value_heads):
                hk = hv // (c.value_heads // c.key_heads)
                q = conv[0, t, hk*c.key_dim:(hk+1)*c.key_dim]
                k = conv[0, t, width+hk*c.key_dim:width+(hk+1)*c.key_dim]
                q = q / torch.sqrt(q.square().mean() + c.epsilon) / c.key_dim
                k = k / torch.sqrt(k.square().mean() + c.epsilon) / c.key_dim ** 0.5
                v = conv[0, t, 2*width+hv*c.value_dim:2*width+(hv+1)*c.value_dim]
                matrix = state[0, hv] * decay[0, t, hv]
                matrix = matrix + torch.outer((v - matrix @ k) * beta[0, t, hv], k)
                state[0, hv] = matrix
                y = matrix @ q
                out.append(y / torch.sqrt(y.square().mean() + c.epsilon) * model.norm_weight * z[0, t, hv].sigmoid())
            result.append(F.linear(torch.cat(out), model.out_proj_weight))
        return torch.stack(result)[None], joined[:, -(c.kernel-1):], state

    def test_asymmetric_state_and_contiguous_head_mapping_match_independent_equation(self):
        model = self.model()
        inputs = self.inputs(model)
        actual = model(*inputs)
        expected = self.independent_reference(model, *inputs)
        for value, target in zip(actual, expected):
            torch.testing.assert_close(value, target, rtol=5e-5, atol=1e-6)

    def test_prefill_decode_continuation_matches_one_chunk_and_state_matters(self):
        model = self.model()
        hidden, history, state = self.inputs(model)
        whole = model(hidden, history, state)
        prefix = model(hidden[:, :4], history, state)
        pieces = [prefix[0]]
        next_history, next_state = prefix[1:]
        for position in range(4, 7):
            step = model(hidden[:, position:position+1], next_history, next_state)
            pieces.append(step[0])
            next_history, next_state = step[1:]
        for value, target in zip((torch.cat(pieces, 1), next_history, next_state), whole):
            torch.testing.assert_close(value, target, rtol=5e-5, atol=1e-6)
        cold = model(hidden[:, -1:], torch.zeros_like(history), torch.zeros_like(state))
        self.assertGreater(float((cold[0] - pieces[-1]).abs().max()), 1e-3)

    def test_invalid_grouping_and_weight_shape_fail_before_export(self):
        model = self.model()
        with self.assertRaisesRegex(ValueError, "grouped"):
            GDN(GDNConfig(5, 2, 3, 3, 4, 4, 1e-6), {})
        weights = {name: np.zeros(shape, np.float32) for name, shape in model.config.weight_shapes.items()}
        weights["in_proj_qkv.weight"] = np.zeros((2, 5), np.float32)
        with self.assertRaisesRegex(ValueError, "in_proj_qkv"):
            GDN(model.config, weights)


if __name__ == "__main__":
    unittest.main()
