"""Independent CPU checks of fused gate/up authoring, without device execution."""
from pathlib import Path
import tempfile
import unittest

import torch

from coreai_q4_gateup import GroupedGateUp,export_smoke,fused_grouped_gateup
from coreai_q4_metal import make_smoke,selected_q4_reference
from coreai_q4_grouped import make_plan


class GateUpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_independent_selected_reference_and_partial_tiles(self):
        gate,x,ids=make_smoke(41,192,37,5,seed=1907)
        up,_,_=make_smoke(41,192,37,5,seed=2399)
        ids,permutation=torch.sort(ids);x=x[permutation]
        g=selected_q4_reference(x,ids,gate.packed,gate.scales,gate.biases)[:,0]
        u=selected_q4_reference(x,ids,up.packed,up.scales,up.biases)[:,0]
        expected=((g*g.sigmoid()).half()*u).half()
        for block in (16,32):
            actual,plan=GroupedGateUp(gate,up,block)(x[:,0],ids)
            torch.testing.assert_close(actual,expected,atol=.001,rtol=.003)
            self.assertEqual(sum(row[2] for row in plan[1:1+int(plan[0,0])].tolist()),41)
            zero,_=GroupedGateUp(gate,up,block)(torch.zeros_like(x[:,0]),ids)
            self.assertTrue(torch.equal(zero,torch.zeros_like(zero)))

    def test_storage_and_exported_graph_are_fused(self):
        gate,x,ids=make_smoke(19,128,67,3)
        up,_,_=make_smoke(19,128,67,3,seed=2399)
        model=GroupedGateUp(gate,up)
        self.assertEqual(model.gate_packed.data_ptr(),gate.packed.data_ptr())
        self.assertEqual(model.up_packed.data_ptr(),up.packed.data_ptr())
        ids,perm=torch.sort(ids)
        ep=torch.export.export(model,args=(x[:,0][perm],ids))
        calls=[str(n.target) for n in ep.graph.nodes if n.op=='call_function']
        self.assertEqual(sum('qwen_q4_grouped_gateup_' in c for c in calls),1)
        self.assertEqual(sum('qwen_expert_plan_' in c for c in calls),1)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'asset'
            report=export_smoke(path,batch=19,inputs=128,outputs=67,experts=3)
            self.assertEqual(report['threadgroupMemoryBytes'],20480)
            self.assertTrue((path/report['model']/'main.mlirb').is_file())
            self.assertFalse(report['deviceValidated'])

    def test_mismatched_projection_shape_rejected(self):
        gate,x,ids=make_smoke(4,128,7,3)
        up,_,_=make_smoke(4,128,8,3)
        ids,perm=torch.sort(ids);plan=make_plan(ids,3)
        with self.assertRaises(ValueError):
            fused_grouped_gateup(x[:,0][perm],plan,gate.packed,gate.scales,gate.biases,
                                up.packed,up.scales,up.biases)


if __name__=='__main__':unittest.main()
