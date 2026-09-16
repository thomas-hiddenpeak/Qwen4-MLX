"""CPU stage decomposition correctness; no device execution."""
import unittest

import torch

from coreai_qsa_chunk import QwenQSAChunk, make_tiny, repeat_activations, replay_state
from coreai_qsa_profile import capture_stages


class QSAProfileTests(unittest.TestCase):
    def test_stages_reconstruct_current_candidate_at_cold_and_sparse_boundaries(self):
        torch.set_num_threads(2)
        source = make_tiny(capacity=2304)
        rows = torch.randn(1, 13, source.hidden, generator=torch.Generator().manual_seed(1027)).half() * 0.2
        candidate = QwenQSAChunk(source)
        for count, offset in ((128, 0), (16, 2047), (128, 2051)):
            with self.subTest(count=count, offset=offset):
                inputs = {"x": repeat_activations(rows, count, offset), **replay_state(source, rows, offset)}
                stages, checks = capture_stages(candidate, inputs)
                self.assertTrue(all(value["exact"] for value in checks.values()))
                self.assertEqual([row[0] for row in stages],
                    ["sdpa", "sdpa_fp32_halfio", "sdpa_fp16", "indexer", "index_scores", "topk_mask", "projection_cache", "output_projection"])
                for name, model, values, expected in stages:
                    with torch.inference_mode():
                        actual = model(*values.values())
                    actual = actual if isinstance(actual, tuple) else (actual,)
                    for a, b in zip(actual, expected.values()):
                        torch.testing.assert_close(a, b, rtol=0.005 if name == "sdpa_fp16" else 0,
                                                   atol=0.0003 if name == "sdpa_fp16" else 0)
                        self.assertTrue(torch.isfinite(b).all())


if __name__ == "__main__":
    unittest.main()
