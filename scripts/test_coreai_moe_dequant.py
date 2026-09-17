"""Bounded CPU checks for opt-in sequential dequant-once MoE."""
import unittest

import torch

from coreai_moe_chunk import ChunkQ4MoE, make_synthetic
from coreai_moe_dequant import DequantOnceChunkMoE
from coreai_q4_flat import flatten_moe_weights
from coreai_expert_grouping import enable_integer_grouping


def models():
    original=ChunkQ4MoE(make_synthetic(),block=16,columns=32,inner=64,fuse_gateup=True).eval()
    flatten_moe_weights(original)
    enable_integer_grouping(original)
    return original,DequantOnceChunkMoE(original,minimum_chunk=4).eval()


class MoEDequantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_alias_and_threshold_fallback(self):
        old,new=models()
        self.assertEqual([(n,t.data_ptr()) for n,t in old.named_buffers()],[(n,t.data_ptr()) for n,t in new.named_buffers()])
        x=(torch.randn(1,3,64,generator=torch.Generator().manual_seed(42))*.125).half()
        for count in (1,3):
            for actual,expected in zip(new(x[:,:count]),old(x[:,:count])):
                torch.testing.assert_close(actual,expected,atol=0,rtol=0)

    def test_full_moe_ties_routes_scores_and_fp16_outputs(self):
        old,new=models()
        for count in (4,17):
            x=(torch.randn(1,count,64,generator=torch.Generator().manual_seed(91))*.125).half()
            x[:,0]=0
            x[0,1,:2]=torch.tensor([.5,0]).half()
            x[0,2,:2]=torch.tensor([-.5,0]).half()
            actual,expected=new(x),old(x)
            torch.testing.assert_close(actual[1],expected[1],atol=0,rtol=0)
            torch.testing.assert_close(actual[2],expected[2],atol=0,rtol=0)
            torch.testing.assert_close(actual[0],expected[0],atol=.002,rtol=.001)
            self.assertEqual(actual[0].dtype,torch.float16)
            self.assertTrue(torch.isfinite(actual[0]).all())

    def test_export_has_three_dequant_ops_and_ordering_operands(self):
        _,model=models()
        x=torch.zeros(1,4,64,dtype=torch.float16)
        ep=torch.export.export(model,(x,))
        calls=[n for n in ep.graph.nodes if n.op=='call_function']
        dequant=[n for n in calls if 'qwen_q4_dequant_sequence_' in str(n.target)]
        gemm=[n for n in calls if 'qwen_dequant_gemm_' in str(n.target)]
        self.assertEqual(len(dequant),3)
        self.assertEqual(len(gemm),3)
        def ancestors(node):
            result=set(node.all_input_nodes)
            for parent in node.all_input_nodes:result.update(ancestors(parent))
            return result
        self.assertIn(gemm[0], list(dequant[1].all_input_nodes))
        self.assertIn(gemm[0], ancestors(dequant[2]))
        self.assertIn(gemm[1], ancestors(dequant[2]))
        self.assertTrue(any(n is dequant[0] for n in gemm[0].all_input_nodes))
        self.assertTrue(any(n is dequant[2] for n in gemm[2].all_input_nodes))


if __name__=='__main__':unittest.main()
