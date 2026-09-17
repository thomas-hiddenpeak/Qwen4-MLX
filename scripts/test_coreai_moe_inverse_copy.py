import copy
import unittest

import torch

from coreai_moe_inverse_copy import inverse_copy,InverseCopyChunkMoE
from coreai_moe_chunk import ChunkQ4MoE,make_synthetic
from coreai_q4_flat import flatten_moe_weights
from coreai_expert_grouping import enable_integer_grouping
from coreai_moe_transfers import install_moe_transfers


class InverseCopyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_copy_bits_and_nonidentity_duplicate_rows(self):
        generator=torch.Generator().manual_seed(47)
        down=torch.randn(70,12,generator=generator).half()
        down[0,:4]=torch.tensor([0.,-0.,2**-24,-2**-24]).half()
        for ids in (torch.randperm(70,generator=generator).int(),torch.tensor([0,69,0,1],dtype=torch.int32)):
            actual=inverse_copy(down,ids)
            expected=torch.index_select(down,0,ids.long())
            torch.testing.assert_close(actual.view(torch.int16),expected.view(torch.int16),atol=0,rtol=0)

    def test_full_moe_native_math_and_weight_contract(self):
        for grouping in (False,True):
            baseline=ChunkQ4MoE(make_synthetic(),block=16,columns=32,inner=64,fuse_gateup=True)
            flatten_moe_weights(baseline)
            if grouping:enable_integer_grouping(baseline)
            candidate=InverseCopyChunkMoE(baseline)
            self.assertEqual([(n,t.data_ptr()) for n,t in baseline.named_buffers()],
                             [(n,t.data_ptr()) for n,t in candidate.named_buffers()])
            x=torch.randn(1,17,64,generator=torch.Generator().manual_seed(91)).half()*.125;x[:,0]=0
            for count in (1,4,17):
                for got,want in zip(candidate(x[:,:count]),baseline(x[:,:count])):
                    torch.testing.assert_close(got,want,atol=0,rtol=0)

    def test_formal_selection_matches_wrapper_and_preserves_native_reduction(self):
        from coreai_moe_dequant import DequantOnceChunkMoE
        for dequant in (False,True):
            baseline=ChunkQ4MoE(make_synthetic(),block=16,columns=32,inner=64,fuse_gateup=True)
            flatten_moe_weights(baseline);enable_integer_grouping(baseline)
            if dequant:baseline=DequantOnceChunkMoE(baseline,minimum_chunk=4)
            candidate=copy.copy(baseline)
            kernels=install_moe_transfers(candidate,tail_precision='native-copy')
            self.assertFalse(baseline.direct_transfers)
            self.assertTrue(all(kernel in candidate.custom_kernels() for kernel in kernels))
            self.assertEqual([(n,t.data_ptr()) for n,t in candidate.named_buffers()],
                             [(n,t.data_ptr()) for n,t in baseline.named_buffers()])
            x=torch.randn(1,4,64,generator=torch.Generator().manual_seed(47)).half()*.125
            for count in (1,4):
                for got,want in zip(candidate(x[:,:count]),baseline(x[:,:count])):
                    torch.testing.assert_close(got,want,atol=0,rtol=0)
            graph=torch.export.export(candidate,(x,)).graph_module.code
            self.assertIn('qwen_moe_inverse_copy_h4_v1',graph)
            self.assertIn('aten.sum.dim_IntList',graph)
            self.assertNotIn('qwen_moe_inverse_weight_sum',graph)


if __name__=='__main__':unittest.main()
