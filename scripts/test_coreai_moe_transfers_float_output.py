import copy
import unittest

import torch

from coreai_moe_chunk import ChunkQ4MoE, make_synthetic
from coreai_moe_transfers import install_moe_transfers
from coreai_moe_transfers_float_output import FloatOutputChunkMoE
from coreai_q4_flat import flatten_moe_weights
from coreai_expert_grouping import enable_integer_grouping


class FloatOutputTests(unittest.TestCase):
    def test_storage_decode_and_eager_float_product_math(self):
        torch.set_num_threads(2)
        baseline=ChunkQ4MoE(make_synthetic(),block=16,columns=32,inner=64,fuse_gateup=True)
        flatten_moe_weights(baseline);enable_integer_grouping(baseline)
        half_output=copy.copy(baseline);install_moe_transfers(half_output,tail_precision='float32')
        candidate=FloatOutputChunkMoE(baseline)
        self.assertEqual([(n,t.data_ptr()) for n,t in baseline.named_buffers()],
                         [(n,t.data_ptr()) for n,t in candidate.named_buffers()])
        self.assertFalse(baseline.direct_transfers)
        x=torch.randn(1,17,64,generator=torch.Generator().manual_seed(47)).half()*.125
        x[:,0]=0
        for count in (1,4,17):
            expected=half_output(x[:,:count]);actual=candidate(x[:,:count])
            for got,want in zip(actual,expected):torch.testing.assert_close(got,want,atol=0,rtol=0)


if __name__=='__main__':unittest.main()
