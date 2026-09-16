"""CPU-only correctness/export checks for the direct packed CoreAI Metal GEMV."""
import unittest

import numpy as np
import torch

from coreai_q4_metal import MetalPackedQ4, make_smoke, selected_q4_reference
from export_coreai_q4_moe import PackedQ4


def unsigned_oracle(model, x, ids, round_weights=True):
    """Independent unsigned shifts, followed by the explicit affine boundary."""
    packed = model.packed.numpy().view(np.uint16)
    shifts = np.arange(0, 16, 4, dtype=np.uint16)
    codes = ((packed[..., None] >> shifts) & 15).reshape(*packed.shape[:2], -1).astype(np.float32)
    scales = np.repeat(model.scales.numpy().astype(np.float32), 64, axis=-1)
    biases = np.repeat(model.biases.numpy().astype(np.float32), 64, axis=-1)
    dense = codes * scales + biases
    if round_weights:
        dense = dense.astype(np.float16).astype(np.float32)
    chosen = dense[ids.numpy()]
    return np.matmul(x.numpy().astype(np.float32), chosen.transpose(0, 2, 1)).astype(np.float16)


class Q4MetalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_signed_bits_and_repeated_ids_independent_oracle(self):
        model, x, _ = make_smoke(batch=4)
        ids = torch.tensor([2, 0, 2, 1], dtype=torch.int32)
        actual = model(x, ids).numpy()
        np.testing.assert_array_equal(actual, unsigned_oracle(model, x, ids))
        self.assertTrue(torch.any(model.packed < 0))
        np.testing.assert_array_equal(model(torch.zeros_like(x), ids).numpy(), np.zeros_like(actual))

    def test_group_boundaries_and_tail_rows_match_existing_packed_q4(self):
        for input_size, output_size, batch in [(64, 1, 1), (128, 7, 4), (640, 5, 10), (2560, 4, 3)]:
            with self.subTest(input_size=input_size, output_size=output_size, batch=batch):
                model, x, ids = make_smoke(batch, input_size, output_size)
                original = PackedQ4(model.packed.numpy().view(np.uint32), model.scales.numpy(), model.biases.numpy())
                torch.testing.assert_close(model(x, ids), original(x, ids), atol=0, rtol=0)

    def test_from_packed_shares_buffers_without_repacking(self):
        model, x, ids = make_smoke()
        original = PackedQ4(model.packed.numpy().view(np.uint32), model.scales.numpy(), model.biases.numpy())
        replacement = MetalPackedQ4.from_packed(original)
        for name in ("packed", "scales", "biases"):
            self.assertEqual(getattr(original, name).data_ptr(), getattr(replacement, name).data_ptr())
        torch.testing.assert_close(replacement(x, ids), original(x, ids), atol=0, rtol=0)

    def test_real_projection_dimensions_s1_s4_and_last_rows(self):
        # Model gate/up: 2560 -> 640; down: 640 -> 2560. The 256 case is
        # supplemental, not this model's expert-intermediate dimension.
        for input_size, output_size in ((2560, 640), (640, 2560), (256, 7)):
            for batch in (10, 40):
                with self.subTest(input_size=input_size, output_size=output_size, batch=batch):
                    model, x, ids = make_smoke(batch, input_size, output_size)
                    result = model(x, ids)
                    self.assertEqual(result.shape, (batch, 1, output_size))
                    # Independently reconstruct first/middle/final rows. Keeping
                    # the oracle bounded avoids a second full selected bank.
                    rows = [0, output_size // 2, output_size - 1]
                    subset = MetalPackedQ4(model.packed[:, rows], model.scales[:, rows], model.biases[:, rows])
                    np.testing.assert_array_equal(result[:, :, rows].numpy(), unsigned_oracle(subset, x, ids))

    def test_fp16_weight_rounding_is_preserved(self):
        model, x, ids = make_smoke(batch=4, input_size=2560, output_size=7)
        actual = model(x, ids).numpy()
        np.testing.assert_array_equal(actual, unsigned_oracle(model, x, ids))
        # This fixture detects an implementation that skips FP16 dequant rounding.
        self.assertTrue(np.any(actual != unsigned_oracle(model, x, ids, round_weights=False)))

    def test_simd_lane_accumulation_error_within_declared_tolerance(self):
        model, x, ids = make_smoke(batch=4, input_size=2560, output_size=7)
        packed = model.packed.numpy().view(np.uint16)
        actual = np.zeros((x.shape[0], 1, packed.shape[1]), np.float16)
        for batch, expert in enumerate(ids.tolist()):
            for row in range(packed.shape[1]):
                accum = np.zeros(32, np.float32)
                for word in range(packed.shape[2]):
                    lane, group = word % 32, word // 16
                    for nibble in range(4):
                        code = (int(packed[expert, row, word]) >> (4 * nibble)) & 15
                        weight = np.float16(np.float32(model.scales[expert, row, group]) * code + np.float32(model.biases[expert, row, group]))
                        accum[lane] += np.float32(weight) * np.float32(x[batch, 0, word * 4 + nibble])
                # Balanced reduction; runtime implementation order remains a GPU check.
                while accum.size > 1:
                    accum = accum[::2] + accum[1::2]
                actual[batch, 0, row] = accum[0]
        expected = model(x, ids).numpy()
        delta = actual.astype(np.float64) - expected.astype(np.float64)
        self.assertLessEqual(float(np.max(np.abs(delta))), 0.002)
        self.assertLessEqual(float(np.linalg.norm(delta) / np.linalg.norm(expected.astype(np.float64))), 0.001)

    def test_torch_export_keeps_one_opaque_kernel_no_dense_materialization(self):
        model, x, ids = make_smoke()
        exported = torch.export.export(model, (x, ids))
        calls = [str(n.target) for n in exported.graph.nodes if n.op == "call_function"]
        kernels = [target for target in calls if "coreai_metal_kernels.qwen_affine_q4_selected_gemv_i16_v1" in target]
        self.assertEqual(len(kernels), 1)
        self.assertFalse(any("matmul" in name or "index_select" in name or "bitwise" in name for name in calls))
        torch.testing.assert_close(exported.module()(x, ids), selected_q4_reference(x, ids, model.packed, model.scales, model.biases), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
