"""CPU numerical policy and indexing contracts; device parity is separate."""
import unittest

import torch

from coreai_hc_write_probe import DirectHCWrite, POLICIES, reference
from export_coreai_dense import DenseConfig, HCWrite


class HCWriteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_rounding_policies_and_non_aligned_branch_indexing(self):
        generator = torch.Generator().manual_seed(2813)
        stream = torch.randn(1, 3, 20, generator=generator).half()
        output = torch.randn(1, 3, 5, generator=generator).half()
        injection = torch.rand(1, 3, 4, 1, generator=generator).half()
        c = DenseConfig(5, 4, 1, 1e-6, 1, 1, 2, 1)
        rounded = reference(stream, output, injection, policy='rounded')
        torch.testing.assert_close(rounded, HCWrite(c)(stream, output, injection), rtol=0, atol=0)
        unrounded = reference(stream, output, injection, policy='unrounded')
        self.assertFalse(torch.equal(rounded, unrounded))
        torch.testing.assert_close(unrounded, reference(stream, output, injection, policy='fma'), rtol=0, atol=0)
        manual = torch.empty_like(stream)
        for token in range(3):
            for channel in range(20):
                manual[0, token, channel] = (stream[0, token, channel].float() +
                    output[0, token, channel % 5].float() * injection[0, token, channel // 5, 0].float()).half()
        torch.testing.assert_close(unrounded, manual, rtol=0, atol=0)
        for policy in POLICIES:
            model = DirectHCWrite(policy)
            torch.testing.assert_close(model(stream, output, injection),
                reference(stream, output, injection, policy=policy), rtol=0, atol=0)
            program = torch.export.export(model, (stream, output, injection))
            targets = [str(node.target) for node in program.graph.nodes if node.op == 'call_function']
            self.assertEqual(sum('qwen_experimental_hc_write_' in target for target in targets), 1)

    def test_reject_mismatched_geometry_and_dtype(self):
        model = DirectHCWrite('unrounded')
        values = (torch.zeros(1, 3, 20).half(), torch.zeros(1, 3, 5).half(), torch.zeros(1, 3, 4, 1).half())
        for bad in ((values[0].float(), *values[1:]),
                    (values[0], values[1], values[2][:, :, :3]),
                    (values[0], values[1][:, :2], values[2])):
            with self.assertRaises(ValueError):
                model(*bad)


if __name__ == '__main__':
    unittest.main()
