"""CPU checks for the experimental native uint4 probe, not GPU acceptance."""
import unittest

import torch

from coreai_q4_native_probe import SOURCE, example, reference, unpack


class NativeUInt4ProbeTests(unittest.TestCase):
    def test_packed_nibble_order_and_unsigned_storage(self):
        x,packed,scales,biases=example()
        codes=unpack(packed)
        self.assertTrue(bool((packed<0).any()))
        torch.testing.assert_close(codes[0],(torch.arange(64)%16).float(),rtol=0,atol=0)
        torch.testing.assert_close(codes[1],(15-torch.arange(64)%16).float(),rtol=0,atol=0)
        raw,_=reference(x,packed,scales,biases)
        for row,column in ((0,0),(1,1),(2,63)):
            torch.testing.assert_close(raw[row],codes[:,column],rtol=0,atol=0)

    def test_affine_reordering_is_explicitly_not_fp16_weight_math(self):
        x,packed,scales,biases=example()
        _,candidate=reference(x,packed,scales,biases)
        weights=unpack(packed)*scales.float()[:,None]+biases.float()[:,None]
        unrounded=x.double()@weights.double().T
        torch.testing.assert_close(candidate.double(),unrounded,rtol=1e-4,atol=2e-6)
        original=x.float()@weights.half().float().T
        self.assertGreater(float((candidate-original).abs().max()),1e-4)
        self.assertIn('array<int,2>{1,256}',SOURCE)
        self.assertIn('alignas(128)',SOURCE)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
