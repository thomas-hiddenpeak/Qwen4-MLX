"""CPU checks for register recurrence and the optional WY algebra oracle."""
import unittest

import numpy as np
import torch

from coreai_gdn_chunk import unit_lower_inverse, wy_recurrence
from coreai_gdn_chunk_metal import FusedGDNRecurrence, make_inputs, recurrence_reference


def simd_oracle(inputs):
    q,k,v,decay,beta,state=[x.numpy().astype(np.float32) for x in inputs]
    heads,values=state.shape[1:3]
    cell=state[0].reshape(heads,values,4,32).transpose(0,1,3,2).copy()
    output=np.empty_like(v,dtype=np.float16)
    def reduce_lanes(value):
        while value.shape[-1]>1: value=value[...,::2]+value[...,1::2]
        return value[...,0]
    for token in range(q.shape[1]):
        key=k[0,token].reshape(heads,4,32).transpose(0,2,1)
        query=q[0,token].reshape(heads,4,32).transpose(0,2,1)
        cell*=decay[0,token,:,None,None,None]
        memory=np.zeros((heads,values,32),np.float32)
        for part in range(4): memory+=cell[...,part]*key[:,None,:,part]
        delta=(v[0,token]-reduce_lanes(memory))*beta[0,token,:,None]
        result=np.zeros_like(memory)
        for part in range(4):
            cell[...,part]+=delta[...,None]*key[:,None,:,part]
            result+=cell[...,part]*query[:,None,:,part]
        output[0,token]=reduce_lanes(result).astype(np.float16)
    return torch.from_numpy(output),torch.from_numpy(cell.transpose(0,1,3,2).reshape(1,heads,values,128).copy())


class GDNChunkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def assert_numerics(self,actual,expected,absolute=2e-4,relative=0.002):
        for a,b in zip(actual,expected):
            delta=a.float()-b.float()
            self.assertLessEqual(float(delta.abs().max()),absolute)
            self.assertLessEqual(float(delta.norm()/b.float().norm().clamp_min(1e-12)),relative)

    def test_register_partition_s128_s256_s512_tail_rows(self):
        for tokens in (128,256,512):
            inputs=make_inputs(tokens,heads=2,value_dim=7)
            self.assert_numerics(simd_oracle(inputs),recurrence_reference(*inputs))

    def test_two_chunks_then_decode_preserve_fp32_state(self):
        inputs=make_inputs(259,heads=2,value_dim=5)
        expected=recurrence_reference(*inputs)
        state=inputs[-1]
        rows=[]
        start=0
        for count in (128,128,1,1,1):
            output,state=FusedGDNRecurrence()(*[v[:,start:start+count] for v in inputs[:-1]],state)
            rows.append(output)
            start+=count
        torch.testing.assert_close(torch.cat(rows,1),expected[0],rtol=0,atol=0)
        torch.testing.assert_close(state,expected[1],rtol=0,atol=0)
        self.assertEqual(state.dtype,torch.float32)

    def test_torch_export_contains_one_recurrence_op_at_s128(self):
        model=FusedGDNRecurrence()
        inputs=make_inputs(128,heads=2,value_dim=7)
        program=torch.export.export(model,inputs)
        names=[str(n.target) for n in program.graph.nodes if n.op=='call_function']
        self.assertEqual(sum('qwen_gdn_recurrence_k128_register_v1' in n for n in names),1)
        self.assertFalse(any('aten.sum' in n or 'aten.matmul' in n for n in names))

    def test_wy_and_block_inverse_handle_zero_one_gates_and_partial_block(self):
        inputs=list(make_inputs(131,heads=2,value_dim=5))
        inputs[3][:,2]=0
        inputs[3][:,64:68]=1
        inputs[3][:,100]=0
        inputs[4][:,8]=0
        inputs[4][:,80]=1
        expected=recurrence_reference(*inputs)
        self.assert_numerics(wy_recurrence(*inputs,block_size=64),expected)
        full=tuple(v[:,:128] for v in inputs[:-1])+(inputs[-1],)
        self.assert_numerics(wy_recurrence(*full,block_size=64,solver='block_inverse'),recurrence_reference(*full))

    def test_block_inverse_strongly_correlated_keys_avoids_power_series_cancellation(self):
        # I+tril(ones,-1) has an exactly bidiagonal inverse. A naive finite
        # geometric series creates huge cancelling powers at this dimension.
        lower=torch.tril(torch.ones(2,128,128),diagonal=-1)
        actual=unit_lower_inverse(lower)
        expected=torch.eye(128)-torch.diag(torch.ones(127),diagonal=-1)
        torch.testing.assert_close(actual,expected.expand(2,-1,-1),rtol=0,atol=0)


if __name__=='__main__': unittest.main()
