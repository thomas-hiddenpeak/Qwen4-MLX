"""CPU validation of independent SG1/2/4 gate/up geometry and numerical policy."""
import unittest

import torch

from coreai_q4_nax_gateup_m32_probe import tiny_inputs
from coreai_q4_nax_gateup_parity import source_text as baseline_source
from coreai_q4_nax_gateup_simd_probe import Projection, get_kernel, source_text


class SIMDGroupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_source_changes_only_bn_not_output_policy(self):
        baseline = baseline_source(67, 128)
        for sg in (1, 2, 4):
            candidate = source_text(67, 128, sg)
            self.assertEqual(candidate.replace(f'BN=32*{sg};', 'BN=32*1;'), baseline)
            self.assertEqual(candidate.count('volatile thread half '), 3)
            self.assertIs(get_kernel(5, 67, 128, sg), get_kernel(5, 67, 128, sg))
        with self.assertRaises(ValueError):
            source_text(67, 128, 3)

    def test_cpu_output_and_bm16_plan_remain_exact(self):
        values = tiny_inputs()
        expected = Projection((5, 67, 128), 1)(*values)
        for sg in (2, 4):
            actual = Projection((5, 67, 128), sg)(*values)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
