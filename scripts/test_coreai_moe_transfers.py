"""Bounded CPU contracts for optional top10 gather and weighted-reduction kernels."""
import copy
import unittest

import numpy as np
import torch

from coreai_moe_transfers import ordered_gather, inverse_weight_sum, install_moe_transfers


class TransfersTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_gather_repeats_original_token_without_reordering_columns(self):
        for count,hidden in ((1,4),(7,12),(17,64)):
            generator=torch.Generator().manual_seed(count)
            x=torch.randn(1,count,hidden,generator=generator).half()
            permutation=torch.randperm(count*10,generator=generator).int()
            expected=x[0,permutation.long()//10]
            torch.testing.assert_close(ordered_gather(x,permutation),expected,atol=0,rtol=0)

    def test_tail_has_explicit_half_product_and_float_slot_order(self):
        count,hidden=7,12
        generator=torch.Generator().manual_seed(47)
        down=torch.randn(count*10,hidden,generator=generator).half()
        # Varied signed, unnormalized scores exercise rounding and cancellation.
        scores=torch.randn(1,count,10,generator=generator).half()
        inverse=torch.randperm(count*10,generator=generator).int()
        expected=np.zeros((count,hidden),dtype=np.float32)
        for token in range(count):
            for slot in range(10):
                products=(down[inverse[token*10+slot]].numpy().astype(np.float32)*
                          np.float32(scores[0,token,slot])).astype(np.float16)
                expected[token]+=products.astype(np.float32)
        actual=inverse_weight_sum(down,inverse,scores)
        torch.testing.assert_close(actual,torch.from_numpy(expected.astype(np.float16)).unsqueeze(0),atol=0,rtol=0)

    def test_reject_wrong_shape_dtype_and_empty_inputs(self):
        with self.assertRaises(ValueError):ordered_gather(torch.zeros(1,0,4).half(),torch.empty(0,dtype=torch.int32))
        with self.assertRaises(ValueError):ordered_gather(torch.zeros(1,1,5).half(),torch.arange(10,dtype=torch.int32))
        with self.assertRaises(ValueError):ordered_gather(torch.zeros(1,1,4).half(),torch.arange(10,dtype=torch.int64))
        with self.assertRaises(ValueError):inverse_weight_sum(torch.zeros(10,4).half(),torch.arange(10,dtype=torch.int32),torch.zeros(1,1,9).half())

    def test_float_product_mode_is_explicit_and_matches_float_oracle(self):
        generator=torch.Generator().manual_seed(83)
        down=torch.randn(70,12,generator=generator).half()
        scores=torch.randn(1,7,10,generator=generator).half()
        inverse=torch.randperm(70,generator=generator).int()
        expected=torch.zeros(7,12,dtype=torch.float32)
        for slot in range(10):
            expected+=down[inverse[slot::10].long()].float()*scores[0,:,slot,None].float()
        actual=inverse_weight_sum(down,inverse,scores,product_precision='float32')
        torch.testing.assert_close(actual,expected.half().unsqueeze(0),atol=0,rtol=0)
        half=inverse_weight_sum(down,inverse,scores)
        self.assertTrue(torch.any(actual!=half),'The fixture must distinguish the two product boundaries')

    def test_chunk_and_dequant_opt_in_preserves_weights_routing_and_decode(self):
        from coreai_moe_chunk import ChunkQ4MoE,make_synthetic
        from coreai_moe_dequant import DequantOnceChunkMoE
        from coreai_q4_flat import flatten_moe_weights
        from coreai_expert_grouping import enable_integer_grouping
        for integer,dequant in ((False,False),(True,False),(True,True)):
            baseline=ChunkQ4MoE(make_synthetic(),block=16,columns=32,inner=64,fuse_gateup=True).eval()
            flatten_moe_weights(baseline)
            if integer:enable_integer_grouping(baseline)
            if dequant:baseline=DequantOnceChunkMoE(baseline,minimum_chunk=4)
            candidate=copy.copy(baseline)
            kernels=install_moe_transfers(candidate)
            self.assertFalse(baseline.direct_transfers)
            self.assertTrue(candidate.direct_transfers)
            self.assertTrue(all(k in candidate.custom_kernels() for k in kernels))
            self.assertEqual([(n,t.data_ptr()) for n,t in baseline.named_buffers()],
                             [(n,t.data_ptr()) for n,t in candidate.named_buffers()])
            x=torch.randn(1,17,64,generator=torch.Generator().manual_seed(47)).half()*.125
            x[:,0]=0
            for count in (1,4,17):
                expected=baseline(x[:,:count]);actual=candidate(x[:,:count])
                torch.testing.assert_close(actual[1],expected[1],atol=0,rtol=0)
                torch.testing.assert_close(actual[2],expected[2],atol=0,rtol=0)
                torch.testing.assert_close(actual[0],expected[0],atol=0,rtol=0)


if __name__=='__main__':unittest.main()
