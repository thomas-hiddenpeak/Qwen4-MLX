"""Bounded CPU checks for experimental native-Q4 grouped arithmetic."""
import unittest

import torch

from coreai_q4_grouped import grouped_reference
from coreai_q4_native_grouped_probe import ExternalNativeProjection, _weights, small_case


class NativeGroupedProbeTests(unittest.TestCase):
    def test_edges_and_affine_rounding_difference(self):
        args=small_case()
        output,plan=ExternalNativeProjection(5,37,128)(*args)
        original=grouped_reference(args[0],plan,*_weights(*args[2:],5,37,128))
        self.assertEqual(output.shape,(23,37))
        self.assertTrue(bool(torch.isfinite(output).all()))
        self.assertEqual(int(plan[1:1+int(plan[0,0]),2].sum()),23)
        self.assertGreater(float((output.float()-original.float()).abs().max()),0)
        zero,zero_plan=ExternalNativeProjection(5,37,128)(torch.zeros_like(args[0]),*args[1:])
        self.assertEqual(int(torch.count_nonzero(zero)),0)
        torch.testing.assert_close(plan,zero_plan,rtol=0,atol=0)

    def test_exactly_representable_affine_weights_agree(self):
        x,ids,packed,scales,biases=small_case()
        # With power-of-two activations and affine metadata, removing the
        # per-weight half round cannot change these exactly representable terms.
        x=torch.round(x.float()*8).half()/8
        scales=torch.full_like(scales,0.0625)
        biases=torch.full_like(biases,-0.25)
        output,plan=ExternalNativeProjection(5,37,128)(x,ids,packed,scales,biases)
        original=grouped_reference(x,plan,*_weights(packed,scales,biases,5,37,128))
        torch.testing.assert_close(output,original,rtol=0,atol=0)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
