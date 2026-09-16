"""CPU-only state, causal visibility, sparse budget, and chunk-boundary checks."""
import unittest

import numpy as np
import torch

from export_coreai_qsa import BINDINGS, INPUT_NAMES, OUTPUT_NAMES, QwenQSA, initial_state


class QSAStateExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self, zero_index=False):
        config = {"hidden_size": 8, "num_attention_heads": 2, "num_key_value_heads": 1,
                  "head_dim": 8, "indexer_n_heads": 2, "indexer_head_dim": 8,
                  "indexer_compress_ratio": 4, "indexer_budget": 8,
                  "partial_rotary_factor": 0.5, "rms_norm_eps": 1e-6,
                  "rope_parameters": {"rope_theta": 10000000}}
        rng = np.random.default_rng(123)
        shapes = {"q_proj.weight": (32, 8), "k_proj.weight": (8, 8), "v_proj.weight": (8, 8),
                  "o_proj.weight": (8, 16), "indexer.index_qk_proj.weight": (24, 8)}
        weights = {k: (0.1 * rng.standard_normal(s)).astype(np.float32) for k, s in shapes.items()}
        for name in ("q_norm.weight", "k_norm.weight", "indexer.q_layernorm.weight", "indexer.k_layernorm.weight"):
            weights[name] = np.ones(8, dtype=np.float32)
        if zero_index:
            weights["indexer.index_qk_proj.weight"][:] = 0
        return QwenQSA(config, weights, capacity=16).eval()

    def execute(self, model, x, state):
        values = {"x": x, **state}
        with torch.inference_mode():
            result = dict(zip(OUTPUT_NAMES, model(*(values[n] for n in INPUT_NAMES))))
        return result, {key: result[out] for key, out in BINDINGS.items()}

    def test_chunked_and_single_prefill_agree_across_pool_and_sparse_boundary(self):
        model = self.model()
        x = torch.from_numpy(np.random.default_rng(77).standard_normal((1, 13, 8)).astype(np.float16))
        whole, _ = self.execute(model, x, initial_state(model))
        state, outputs, offset = initial_state(model), [], 0
        for count in (3, 1, 5, 3, 1):
            result, state = self.execute(model, x[:, offset:offset + count], state)
            outputs.append(result["y"])
            offset += count
        torch.testing.assert_close(torch.cat(outputs, dim=1), whole["y"], rtol=0, atol=0.0005)
        for key, out in BINDINGS.items():
            torch.testing.assert_close(state[key], whole[out], rtol=0, atol=0)

    def test_sparse_budget_tie_bias_and_incomplete_tail_are_exact(self):
        model = self.model(zero_index=True)
        x = torch.ones(1, 13, 8, dtype=torch.float16)
        result, _ = self.execute(model, x, initial_state(model))
        mask = result["attention_mask"][0, 0]
        self.assertTrue(torch.equal(mask[10, :11], torch.ones(11, dtype=torch.int32)))
        self.assertEqual(mask[10, 11:].sum().item(), 0)
        # Three complete blocks exceed top2: zero scores and negative index bias
        # pick blocks0/1. Position12 is the incomplete tail and remains visible.
        self.assertTrue(torch.equal(mask[11], torch.tensor([1] * 8 + [0] * 8, dtype=torch.int32)))
        self.assertTrue(torch.equal(mask[12], torch.tensor([1] * 8 + [0] * 4 + [1] + [0] * 3, dtype=torch.int32)))

    def test_append_preserves_prefix_and_unused_storage_is_masked(self):
        model = self.model()
        x = torch.ones(1, 4, 8, dtype=torch.float16)
        first, state = self.execute(model, x, initial_state(model))
        clean = {k: v.clone() for k, v in state.items()}
        state = {k: v.clone() for k, v in state.items()}
        state["key_cache"][:, :, 5:] = 123
        state["value_cache"][:, :, 5:] = -321
        state["raw_cache"][:, 5:] = 19
        old_key = state["key_cache"].clone()
        appended, _ = self.execute(model, x[:, :1], state)
        reference, _ = self.execute(model, x[:, :1], clean)
        torch.testing.assert_close(appended["y"], reference["y"], rtol=0, atol=0)
        torch.testing.assert_close(appended["key_cache_out"][:, :, :4], old_key[:, :, :4], rtol=0, atol=0)
        torch.testing.assert_close(appended["key_cache_out"][:, :, 5:], old_key[:, :, 5:], rtol=0, atol=0)
        self.assertEqual(appended["offset_out"].item(), 5)
        self.assertEqual(appended["pooled_count_out"].item(), 1)


if __name__ == "__main__":
    unittest.main()
