"""CPU semantic/online recurrence tests for the isolated sparse attention probe."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from coreai_qsa_sparse import (SparseAttention, block_ids_from_mask, export_smoke, make_cases, online_reference,
                               select_blocks, selected_positions, sparse_reference)


def numpy_dense_reference(inputs, ratio=4):
    """Independent dense boolean-mask/F64 softmax oracle, chronological keys."""
    query, keys, values, ids, offset = (value.numpy() for value in inputs.values())
    result = np.zeros_like(query)
    count, width = query.shape[2:]
    group = query.shape[1] // keys.shape[1]
    for token in range(count):
        position = int(offset[0]) + token
        mask = np.zeros(keys.shape[2], dtype=bool)
        if (position + 1) // ratio <= ids.shape[-1]:
            mask[:position + 1] = True
        else:
            for block in ids[0, token]:
                if block >= 0 and (int(block) + 1) * ratio <= position + 1:
                    mask[int(block) * ratio:(int(block) + 1) * ratio] = True
            mask[(position + 1) // ratio * ratio:position + 1] = True
        for head in range(query.shape[1]):
            if not mask.any():
                continue
            q = query[0, head, token].astype(np.float64)
            k = keys[0, head // group, mask].astype(np.float64)
            v = values[0, head // group, mask].astype(np.float64)
            scores = q @ k.T / np.sqrt(width)
            probability = np.exp(scores - scores.max())
            result[0, head, token] = probability @ v / probability.sum()
    return torch.from_numpy(result)


class SparseQSATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_semantic_and_online_oracles_match_independent_dense_mask(self):
        for width in (32, 256):
            for name, values in make_cases(head_dim=width):
                with self.subTest(width=width, case=name):
                    expected = numpy_dense_reference(values)
                    exact = sparse_reference(*values.values())
                    online = online_reference(*values.values())
                    torch.testing.assert_close(exact, expected, rtol=0.001, atol=0.0003)
                    torch.testing.assert_close(online, expected, rtol=0.005, atol=0.0003)
                    self.assertTrue(torch.isfinite(online).all())

    def test_causal_future_slots_are_never_read_and_score_order_preserves_semantics(self):
        cases = dict(make_cases(head_dim=32))
        cold = cases["dense_cold"]
        original = sparse_reference(*cold.values())
        poisoned = {key: value.clone() for key, value in cold.items()}
        count = cold["query"].shape[2]
        poisoned["keys"][:, :, count:] = torch.nan
        poisoned["values"][:, :, count:] = torch.nan
        torch.testing.assert_close(sparse_reference(*poisoned.values()), original, rtol=0, atol=0)
        a = sparse_reference(*cases["sparse_partial"].values())
        b = sparse_reference(*cases["score_order_permuted"].values())
        torch.testing.assert_close(a, b, rtol=0.001, atol=0.0003)

    def test_partial_blocks_dense_threshold_duplicate_and_future_ids(self):
        ids = torch.tensor([7, 1], dtype=torch.int32)
        self.assertEqual(selected_positions(ids, 10, 32), list(range(11)))
        self.assertEqual(selected_positions(ids, 11, 32), [4, 5, 6, 7])
        self.assertEqual(selected_positions(ids, 14, 32), [4, 5, 6, 7, 12, 13, 14])
        self.assertEqual(selected_positions(torch.tensor([-1, -1]), 11, 32), [])
        with self.assertRaisesRegex(ValueError, "unique"):
            selected_positions(torch.tensor([1, 1]), 11, 32)
        # Identical positive scores retain the existing low-block bias; future
        # very-large scores must not displace visible blocks.
        scores = torch.ones(1, 3, 16)
        scores[:, :, 8:] = 1000
        actual = select_blocks(scores, torch.tensor([31], dtype=torch.int32), budget=16)
        self.assertEqual(actual.sort(-1).values.tolist(), [[[0, 1, 2, 3]] * 3])

    def test_export_graph_keeps_runtime_selected_ids_and_offsets(self):
        cases = dict(make_cases(head_dim=32))
        module = SparseAttention()
        graph = torch.export.export(module, tuple(cases["dense_cold"].values()))
        values = cases["invalid_future_sentinels"]
        torch.testing.assert_close(graph.module()(*values.values()), sparse_reference(*values.values()), rtol=0, atol=0)

    def test_mask_recovery_preserves_full_selection_and_tail_and_rejects_partial_blocks(self):
        for name, inputs in make_cases(head_dim=32)[:3]:
            with self.subTest(case=name):
                count, capacity = inputs["query"].shape[2], inputs["keys"].shape[2]
                offset = int(inputs["offset"].item())
                mask = torch.zeros(1, 1, count, capacity, dtype=torch.int32)
                for token in range(count):
                    positions = selected_positions(inputs["block_ids"][0, token], offset + token, capacity)
                    mask[0, 0, token, positions] = 1
                restored = block_ids_from_mask(mask, offset, budget=32)
                actual = sparse_reference(inputs["query"], inputs["keys"], inputs["values"], restored, inputs["offset"])
                torch.testing.assert_close(actual, sparse_reference(*inputs.values()), rtol=0.001, atol=0.0003)
                if name == "sparse_partial":
                    broken = mask.clone()
                    block = int(restored[0, 0, 0].item())
                    broken[0, 0, 0, block * 4] = 0
                    with self.assertRaisesRegex(ValueError, "complete-block"):
                        block_ids_from_mask(broken, offset, budget=32)

    def test_small_asset_and_fixtures_are_complete_without_global_gather(self):
        with tempfile.TemporaryDirectory(prefix="coreai-qsa-sparse-test-") as directory:
            root = Path(directory) / "probe"
            report = export_smoke(root, head_dim=32, capacity=64, budget=16, count=4)
            self.assertEqual(report["globalGatherBuffers"], 0)
            ir = (root / "coreai-main-after.txt").read_text()
            self.assertEqual(ir.count("coreai.metal4_kernel "), 1)
            self.assertNotIn("coreai.gather", ir)
            for case in report["cases"]:
                fixture = json.loads((root / case["fixture"]).read_text())
                self.assertEqual(set(fixture["inputs"]), {"query", "keys", "values", "block_ids", "offset"})
                self.assertEqual(set(fixture["expectedOutputs"]), {"output"})


if __name__ == "__main__":
    unittest.main()
