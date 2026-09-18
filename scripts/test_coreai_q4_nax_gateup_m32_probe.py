"""CPU contracts for the separate BM32 measured-parity gate/up candidate."""
import unittest

import torch

from coreai_q4_nax_gateup_m32_probe import Projection, source_text, tiny_inputs


class GateUpBM32Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_source_reuses_weight_staging_and_preserves_three_half_boundaries(self):
        source = source_text(67, 128)
        self.assertEqual(source.count('threadgroup half right_memory[BN*72]'), 1)
        self.assertEqual(source.count('right_memory[int(thread_id)*72+word*4+nibble]='), 2)
        for name in ('gate_accum', 'up_accum', 'gate_accum_hi', 'up_accum_hi'):
            self.assertIn(f'operation.run(a_hi,b,{name});' if name.endswith('_hi') else f'operation.run(a,b,{name});', source)
        for name in ('gate_boundary', 'sigmoid_boundary', 'gate_up_boundary'):
            self.assertEqual(source.count('volatile thread half ' + name + '='), 1)
        self.assertEqual(source.count('{'), source.count('}'))

    def test_cpu_policies_match_across_plans_and_expert_boundaries(self):
        values = tiny_inputs()
        expected, plan16 = Projection((5, 67, 128), 16)(*values)
        actual, plan32 = Projection((5, 67, 128), 32)(*values)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for block, plan in ((16, plan16), (32, plan32)):
            coverage = torch.zeros(values[0].shape[0], dtype=torch.int32)
            for expert, start, count, _ in plan[1:1 + int(plan[0, 0])].tolist():
                self.assertTrue(0 < count <= block)
                self.assertTrue(bool((values[1][start:start + count] == expert).all()))
                coverage[start:start + count] += 1
            self.assertTrue(bool((coverage == 1).all()))


if __name__ == '__main__':
    unittest.main()
