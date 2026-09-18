"""CPU wiring checks; whole-layer GPU parity remains an explicit device test."""
import unittest

import torch

from coreai_hc_write_layer import with_direct_hc_write
from coreai_hc_write_probe import DirectHCWrite
from test_export_coreai_pd import CoreAIPrefillTests as PDTests


class WholeHCWriteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_aliases_s1_and_original_output_contract_with_ple(self):
        helper = PDTests()
        for ple in (False,True):
            c, original, history, state = helper.decoder(ple)
            before = [(name,value.data_ptr()) for name,value in original.named_buffers()]
            for scope in ('attention','moe','both'):
                candidate, kernels = with_direct_hc_write(original,scope=scope)
                self.assertEqual([(name,value.data_ptr()) for name,value in candidate.named_buffers()],before)
                self.assertFalse(hasattr(original,'direct_hc_write'))
                self.assertEqual(len(kernels),1)
                stream = helper.tensor((1,4,c.width),seed=149)
                embedding = helper.tensor((1,4,c.ple_dim),seed=151)
                states = (history,state)
                if ple: states += (helper.tensor((1,c.ple_history,c.width),seed=152),)
                single = (stream[:,:1],embedding[:,:1],*states) if ple else (stream[:,:1],*states)
                for actual,expected in zip(candidate(*single),original(*single)):
                    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
                args = (stream,embedding,*states) if ple else (stream,*states)
                outputs = candidate(*args)
                self.assertEqual(len(outputs),4 if ple else 3)
                graph = torch.export.export(candidate,args).graph_module.code
                self.assertEqual(graph.count('torch.ops.coreai_metal_kernels.qwen_experimental_hc_write_unrounded_v1.default('),
                                 2 if scope=='both' else 1)
                self.assertTrue(all(torch.isfinite(x).all() for x in outputs))
            self.assertEqual([(name,value.data_ptr()) for name,value in original.named_buffers()],before)

    def test_invalid_selection_does_not_mutate(self):
        _, original, _, _ = PDTests().decoder(False)
        for options in ({'scope':'wrong'},{'policy':'wrong'}):
            with self.assertRaises(ValueError): with_direct_hc_write(original,**options)
        self.assertFalse(hasattr(original,'direct_hc_write'))


if __name__=='__main__': unittest.main()
