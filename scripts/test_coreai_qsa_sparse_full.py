"""CPU full-wrapper contract tests; device speed/numerics remain separate."""
import unittest

import numpy as np
import torch

from coreai_qsa_chunk import seeded_state
from coreai_qsa_sparse_full import QwenQSASparsePrefill
from export_coreai_qsa import INPUT_NAMES, OUTPUT_NAMES, QwenQSA


def small_source():
    config = {"hidden_size": 16, "num_attention_heads": 4, "num_key_value_heads": 1,
        "head_dim": 32, "indexer_n_heads": 2, "indexer_head_dim": 8,
        "indexer_compress_ratio": 4, "indexer_budget": 32, "partial_rotary_factor": 0.125,
        "rms_norm_eps": 1e-6, "rope_parameters": {"rope_theta": 10000000}}
    generator = np.random.default_rng(61733)
    shapes = {"q_proj.weight": (256, 16), "k_proj.weight": (32, 16), "v_proj.weight": (32, 16),
        "o_proj.weight": (16, 128), "indexer.index_qk_proj.weight": (24, 16)}
    weights = {name: (generator.standard_normal(shape) * 0.1).astype(np.float32) for name, shape in shapes.items()}
    for name, size in (("q_norm.weight", 32), ("k_norm.weight", 32),
                       ("indexer.q_layernorm.weight", 8), ("indexer.k_layernorm.weight", 8)):
        weights[name] = generator.uniform(0.8, 1.2, size).astype(np.float32)
    return QwenQSA(config, weights, 128).eval()


class SparseFullTests(unittest.TestCase):
    def test_sparse_prefill_preserves_every_state_and_mask_and_scalar_decode(self):
        torch.set_num_threads(2)
        source = small_source()
        candidate = QwenQSASparsePrefill(source).eval()
        for count, offset in ((1, 35), (7, 0), (7, 31), (7, 65)):
            with self.subTest(count=count, offset=offset):
                x = torch.randn(1, count, 16, generator=torch.Generator().manual_seed(6100 + count)).half() * 0.2
                values = {"x": x, **seeded_state(source, offset)}
                args = tuple(values[name] for name in INPUT_NAMES)
                with torch.inference_mode():
                    actual, expected = candidate(*args), source(*args)
                for name, a, b in zip(OUTPUT_NAMES, actual, expected):
                    approximate = name == "y" and count > 1
                    torch.testing.assert_close(a, b, rtol=0.005 if approximate else 0, atol=0.0003 if approximate else 0)
                graph = torch.export.export(candidate, args)
                targets = [str(node.target) for node in graph.graph.nodes if node.op == "call_function"]
                self.assertEqual(sum("qwen_qsa_sparse_" in target for target in targets), 0 if count == 1 else 1)


if __name__ == "__main__":
    unittest.main()
