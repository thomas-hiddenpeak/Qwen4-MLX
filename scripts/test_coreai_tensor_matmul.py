"""CPU-only authoring and interface checks for the MPP GEMM prototype."""
import tempfile
from pathlib import Path
import unittest

import torch

from coreai_tensor_matmul import TensorLinear, export_smoke, get_tensor_kernel, tensor_linear


class TensorMatmulTests(unittest.TestCase):
    def test_default_and_explicit_tiles_share_one_custom_op_registration(self):
        self.assertIs(get_tensor_kernel(), get_tensor_kernel(32, 64))
        self.assertIs(get_tensor_kernel(), get_tensor_kernel(tile_n=64, tile_m=32))

    def test_rectangular_transpose_and_fp16_boundary(self):
        # Exactly representable small operands: independent Float64 oracle also
        # detects a transposition or output-axis mistake without a loose tolerance.
        x = (torch.arange(15).reshape(1, 3, 5).float() / 16).half()
        weight = (torch.arange(35).reshape(7, 5).float() / 32 - 0.5).half()
        actual = tensor_linear(x, weight)
        expected = (x.double() @ weight.double().T).half()
        self.assertEqual(actual.shape, (1, 3, 7))
        self.assertEqual(actual.dtype, torch.float16)
        self.assertTrue(torch.equal(actual, expected))

    def test_rejects_unsupported_io(self):
        w = torch.ones(7, 5, dtype=torch.float16)
        for x in [torch.ones(2, 3, 5, dtype=torch.float16), torch.ones(1, 3, 4, dtype=torch.float16),
                  torch.ones(1, 3, 5), torch.ones(3, 5, dtype=torch.float16),
                  torch.ones(1, 0, 5, dtype=torch.float16)]:
            with self.assertRaises(ValueError): tensor_linear(x, w)
        with self.assertRaises(ValueError): tensor_linear(torch.ones(1, 3, 5).half(), w.float())
        with self.assertRaises(ValueError): get_tensor_kernel(17, 32)

    def test_export_retains_mpp_custom_op_and_int_tensor_signature(self):
        # Three tokens and seven outputs require both M and N edge handling;
        # device execution is intentionally left to the dedicated probe owner.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'small'
            report = export_smoke(path, counts=(3, 33), input_size=16, output_size=7)
            self.assertTrue((path / report['model'] / 'main.mlirb').is_file())
            self.assertFalse(report['deviceValidated'])
            self.assertEqual(len(report['cases']), 4)
            self.assertTrue(report['sourceFiles'])
            for source in report['sourceFiles']:
                code = (path / source).read_text()
                self.assertIn('#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>', code)
                self.assertIn('metal::dextents<int, 2>', code)
                self.assertIn('matmul2d<descriptor, execution_simdgroups<4>>', code)
                self.assertIn('operation.run(left, right, accum)', code)
                self.assertIn('accum.is_valid_element(element)', code)
                self.assertNotIn('accum.get_mask(', code)
                self.assertNotIn('simd_sum(', code)
                self.assertNotIn('accum.store(', code)
            ep = torch.export.export(TensorLinear(torch.ones(7,16).half()),
                                     args=(torch.zeros(1,3,16).half(),))
            targets = [str(n.target) for n in ep.graph.nodes if n.op == 'call_function']
            self.assertEqual(sum('qwen_mpp_fp16_gemm_' in t for t in targets), 1)
            self.assertFalse(any('aten.mm.' in t or 'aten.bmm.' in t for t in targets))


if __name__ == '__main__': unittest.main()
