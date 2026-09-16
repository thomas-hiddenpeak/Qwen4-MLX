"""CPU semantic tests for complete grouped MoE, without device execution."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch

from coreai_moe_chunk import ChunkQ4MoE, make_synthetic, export_model, grouping_permutations


class ChunkMoETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_order_scores_ties_and_weighted_merge_against_token_reference(self):
        original=make_synthetic();reference=deepcopy(original);model=ChunkQ4MoE(original)
        generator=torch.Generator().manual_seed(5146)
        x=(torch.randn(1,35,64,generator=generator)*0.125).half()
        x[:,0]=0;x[0,1,:2]=torch.tensor([0.5,0]);x[0,2,:2]=torch.tensor([-0.5,0])
        with torch.inference_mode():
            actual=model(x)
            expected=[torch.cat([reference(x[:,i:i+1])[j] for i in range(x.shape[1])],dim=1) for j in range(3)]
        self.assertTrue(torch.equal(actual[1],expected[1]))
        self.assertTrue(torch.equal(actual[2],expected[2]))
        self.assertEqual(actual[1][0,0].tolist(),list(range(10)))
        torch.testing.assert_close(actual[0],expected[0],atol=0.000002,rtol=0.002)

    def test_s1_keeps_direct_kernel_and_buffer_storage(self):
        original=make_synthetic()
        pointers={name:getattr(original.gate_proj,name).data_ptr() for name in ('packed','scales','biases')}
        reference=deepcopy(original)
        model=ChunkQ4MoE(original)
        for name,pointer in pointers.items():self.assertEqual(getattr(model.decode.gate_proj,name).data_ptr(),pointer)
        x=torch.zeros(1,1,64).half()
        for actual,expected in zip(model(x),reference(x)):
            self.assertTrue(torch.equal(actual,expected))
        ep=torch.export.export(model,args=(x,))
        targets=[str(n.target) for n in ep.graph.nodes if n.op=='call_function']
        self.assertEqual(sum('qwen_affine_q4_selected_gemv' in t for t in targets),3)
        self.assertFalse(any('qwen_expert_plan' in t or 'qwen_q4_grouped_' in t for t in targets))

    def test_stable_grouping_maximum_chunk_and_exact_integer_keys(self):
        experts = 512
        generator = torch.Generator().manual_seed(2718)
        for count, maximum_key in ((512, 2_621_439), (2048, 10_485_759)):
            rows = count * 10
            ids = torch.randint(experts, (rows,), dtype=torch.int32, generator=generator)
            ids[-1] = experts - 1
            positions = torch.arange(rows, dtype=torch.int32)
            keys = ids * rows + positions
            self.assertEqual(keys.dtype, torch.int32)
            self.assertEqual(int(keys.max()), maximum_key)
            self.assertLess(int(keys.max()), 2**24)
            self.assertTrue(torch.equal(keys.float().int(), keys))
            permutation, inverse = grouping_permutations(ids, experts)
            self.assertTrue(torch.equal(permutation, torch.argsort(ids, stable=True)))
            self.assertTrue(torch.equal(ids[permutation][inverse], ids))
            self.assertTrue(torch.equal(permutation[inverse], positions.long()))
        with self.assertRaises(ValueError):
            grouping_permutations(torch.zeros(32769, dtype=torch.int32), experts)
        with self.assertRaises(ValueError):
            ChunkQ4MoE(make_synthetic())(torch.zeros(1, 2049, 64).half())

    def test_one_plan_shared_across_three_projections_and_authoring(self):
        model=ChunkQ4MoE(make_synthetic())
        ep=torch.export.export(model,args=(torch.zeros(1,33,64).half(),))
        targets=[str(n.target) for n in ep.graph.nodes if n.op=='call_function']
        self.assertEqual(sum('qwen_expert_plan' in t for t in targets),1)
        self.assertEqual(sum('qwen_q4_grouped_' in t for t in targets),3)
        self.assertEqual(sum('qwen_mpp_fp16_gemm_' in t for t in targets),5)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'small'
            report=export_model(path,make_synthetic(),counts=(1,5))
            self.assertTrue((path/report['model']/'main.mlirb').is_file())
            self.assertFalse(report['deviceValidated'])
            self.assertTrue(all(c['idsExact'] for case in report['cases'] for c in case['cpuTokenChecks']))

    def test_optional_fused_gateup_keeps_cpu_outputs_and_s1(self):
        original=make_synthetic()
        plain=ChunkQ4MoE(deepcopy(original),columns=32,inner=64)
        fused=ChunkQ4MoE(original,columns=32,inner=64,fuse_gateup=True)
        generator=torch.Generator().manual_seed(3221)
        x=(torch.randn(1,35,64,generator=generator)*0.125).half();x[:,0]=0
        for count in (1,35):
            for actual,expected in zip(fused(x[:,:count]),plain(x[:,:count])):
                self.assertTrue(torch.equal(actual,expected))
        ep=torch.export.export(fused,args=(x,))
        targets=[str(n.target) for n in ep.graph.nodes if n.op=='call_function']
        self.assertEqual(sum('qwen_q4_grouped_gateup_' in t for t in targets),1)
        self.assertEqual(sum('qwen_q4_grouped_m' in t for t in targets),1)
        with tempfile.TemporaryDirectory() as directory:
            report=export_model(Path(directory)/'fused',make_synthetic(),counts=(1,5),
                                columns=32,inner=64,fuse_gateup=True)
            self.assertTrue(report['fusedGateUp'])
            self.assertEqual(report['tile'],[16,32,64])


if __name__=='__main__':unittest.main()
