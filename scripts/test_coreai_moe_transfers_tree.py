import unittest

import numpy as np
import torch

from coreai_moe_transfers_tree import tree_reference,TreeFloatOutputChunkMoE
from coreai_moe_chunk import ChunkQ4MoE,make_synthetic


class TreeTailTests(unittest.TestCase):
    def test_independent_numpy_tree_and_decode_storage(self):
        torch.set_num_threads(2)
        generator=torch.Generator().manual_seed(4701)
        down=torch.randn(170,12,generator=generator).half()
        scores=torch.randn(1,17,10,generator=generator).half()
        inverse=torch.randperm(170,generator=generator).int()
        oracle=np.zeros((17,16,12),dtype=np.float32)
        for token in range(17):
            for slot in range(10):
                oracle[token,slot]=down[inverse[token*10+slot]].numpy().astype(np.float32)*np.float32(scores[0,token,slot])
        for stride in (8,4,2,1):
            for slot in range(stride):oracle[:,slot]=oracle[:,slot]+oracle[:,slot+stride]
        torch.testing.assert_close(tree_reference(down,inverse,scores),torch.from_numpy(oracle[:,0]).unsqueeze(0),atol=0,rtol=0)
        original=ChunkQ4MoE(make_synthetic())
        candidate=TreeFloatOutputChunkMoE(original)
        self.assertEqual([(n,t.data_ptr()) for n,t in original.named_buffers()],
                         [(n,t.data_ptr()) for n,t in candidate.named_buffers()])
        x=torch.zeros(1,1,64,dtype=torch.float16)
        for got,want in zip(candidate(x),original(x)):torch.testing.assert_close(got,want,atol=0,rtol=0)


if __name__=='__main__':unittest.main()
