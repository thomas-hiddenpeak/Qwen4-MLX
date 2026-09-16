import unittest
import torch
from coreai_q4_grouped import make_plan, plan_reference, grouped_linear
from coreai_q4_metal import make_smoke, selected_q4_reference


class GroupedQ4Tests(unittest.TestCase):
    def test_every_assignment_is_covered_once(self):
        for block in (16,32):
            for counts in ([1,0,17,32,3],[0,0,79,0,0],[15,16,17,31,32]):
                ids=torch.repeat_interleave(torch.arange(len(counts),dtype=torch.int32),torch.tensor(counts))
                plan=make_plan(ids,len(counts),block)
                seen=[]
                for e,start,count,_ in plan[1:1+int(plan[0,0])].tolist():
                    self.assertLessEqual(count,block)
                    self.assertTrue(bool((ids[start:start+count]==e).all()))
                    seen.extend(range(start,start+count))
                self.assertEqual(seen,list(range(len(ids))))
                self.assertTrue(bool((plan[1+int(plan[0,0]):]==0).all()))

    def test_unsorted_or_out_of_range_ids_rejected_by_oracle(self):
        for ids in ([2,0,1],[-1,0,2],[0,1,5]):
            with self.assertRaises(ValueError):plan_reference(torch.tensor(ids,dtype=torch.int32),5,16)

    def test_grouped_matches_selected_projection(self):
        original,x,ids=make_smoke(41,192,37,5)
        permutation=torch.argsort(ids)
        x=x[permutation];ids=ids[permutation]
        expected=selected_q4_reference(x,ids,original.packed,original.scales,original.biases)[:,0]
        plan=make_plan(ids,5,16)
        actual=grouped_linear(x[:,0],plan,original.packed,original.scales,original.biases)
        torch.testing.assert_close(actual,expected,atol=.001,rtol=.001)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
