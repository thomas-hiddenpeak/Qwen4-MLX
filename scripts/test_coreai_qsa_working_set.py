"""CPU equivalence of static work views, including state and sparse boundaries."""
import unittest

import torch

from coreai_qsa_chunk import QwenQSAChunk, make_tiny, seeded_state
from coreai_qsa_working_set import QwenQSAWorkingSet, working_set_function
from export_coreai_qsa import INPUT_NAMES


class QSAWorkingSetTests(unittest.TestCase):
    def compare_case(self,capacity,budget,offset,count,limit,pooled_count=None):
        source=make_tiny(capacity,budget)
        original=QwenQSAChunk(source,prefill_sdpa_fp16=True)
        candidate=QwenQSAWorkingSet(source,limit)
        candidate.validate_bounds(offset,count)
        state=seeded_state(source,offset,pooled_count=pooled_count)
        generator=torch.Generator().manual_seed(712)
        x=(torch.randn(1,count,source.hidden,generator=generator)*0.3).half()
        values={'x':x,**state}
        with torch.inference_mode():
            expected=original(*(values[n] for n in INPUT_NAMES))
            actual=candidate(*(values[n] for n in INPUT_NAMES))
        for a,b in zip(actual[1:7],expected[1:7]):
            torch.testing.assert_close(a,b,atol=0,rtol=0)
        effective=source.capacity if count==1 else limit
        torch.testing.assert_close(actual[-1],expected[-1][...,:effective],atol=0,rtol=0)
        self.assertEqual(int(torch.count_nonzero(expected[-1][...,effective:])),0)
        torch.testing.assert_close(actual[0],expected[0],atol=1e-5,rtol=1e-4)
        return candidate,values

    def test_dense_cold_and_nonzero_history(self):
        for args in ((64,16,0,8,8),(64,16,3,8,12),(64,16,8,8,16)):
            with self.subTest(args=args):self.compare_case(*args)

    def test_sparse_partial_blocks_and_lagging_pool(self):
        for args in ((64,16,15,8,24,None),(64,16,19,4,24,3),
                     (64,16,39,8,48,None),(64,16,59,4,64,None)):
            with self.subTest(args=args):self.compare_case(*args)

    def test_actual_2051_sparse_threshold(self):
        self.compare_case(2112,2048,2051,4,2056,510)

    def test_s1_unchanged_even_past_bucket(self):
        self.compare_case(64,16,35,1,16)

    def test_selection_guards_and_dense_graph_pruning(self):
        module,values=self.compare_case(64,16,0,8,8)
        for args in ((1,8),(0,9),(-1,8)):
            with self.assertRaises(ValueError):module.validate_bounds(*args)
        self.assertEqual(working_set_function(2048,4096),'prefill_s2048_kv4096')
        graph=torch.export.export(module,tuple(values[n] for n in INPUT_NAMES))
        targets=[str(n.target) for n in graph.graph.nodes if n.op=='call_function']
        self.assertFalse(any('topk' in name or 'argsort' in name for name in targets))


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
