"""Bounded CPU checks of BM32 NAX register mapping and expert row coverage."""
import unittest

import torch

from coreai_q4_metal import make_smoke
from coreai_q4_nax_m32_probe import Projection, check_fragment_mapping, source_text


class BM32Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_fragment_coverage_and_single_weight_stage(self):
        check_fragment_mapping()
        source = source_text(67, 128)
        self.assertEqual(source.count('threadgroup half right_memory[BN*72]'), 1)
        self.assertEqual(source.count('right_memory[int(thread_id)*72+word*4+nibble]='), 1)
        self.assertEqual(source.count('operation.run(a,b,c)'), 1)
        self.assertEqual(source.count('operation.run(a_hi,b,c_hi)'), 1)
        self.assertIn('row+16<count', source)
        self.assertNotIn('left_memory', source)

    def test_two_subtiles_expert_boundaries_and_output_columns(self):
        ids = torch.repeat_interleave(torch.arange(5, dtype=torch.int32), torch.tensor([1, 16, 17, 32, 33]))
        projection, x, _ = make_smoke(len(ids), 128, 67, 5, seed=29117)
        values = (x[:, 0].contiguous(), ids, projection.packed.flatten(),
                  projection.scales.flatten(), projection.biases.flatten())
        outputs = [Projection((5, 67, 128), block)(*values) for block in (16, 32)]
        torch.testing.assert_close(outputs[0][0], outputs[1][0], rtol=0, atol=0)
        for block, (_, plan) in zip((16, 32), outputs):
            coverage = torch.zeros(len(ids), dtype=torch.int32)
            for expert, start, count, _ in plan[1:1 + int(plan[0, 0])].tolist():
                self.assertTrue(0 < count <= block)
                self.assertTrue(bool((ids[start:start + count] == expert).all()))
                coverage[start:start + count] += 1
            self.assertTrue(bool((coverage == 1).all()))
        self.assertLess(int(outputs[1][1][0, 0]), int(outputs[0][1][0, 0]))


if __name__ == '__main__':
    unittest.main()
