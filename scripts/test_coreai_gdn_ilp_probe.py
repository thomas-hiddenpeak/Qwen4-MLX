"""CPU contract checks for opt-in GDN ILP; Metal execution is a separate gate."""
import unittest

import numpy as np
import torch

from coreai_gdn_chunk import GDNRegisterPrefill
from coreai_gdn_chunk_metal import make_inputs
from coreai_gdn_ilp_probe import ILPRecurrence, PhaseILPRecurrence, install_gdn_ilp
from export_coreai_gdn import GDN, GDNConfig


def small_gdn():
    config = GDNConfig(16, 1, 2, 128, 7, 4, 1e-6)
    rng = np.random.default_rng(2901)
    weights = {name: (rng.standard_normal(shape) * .1).astype(np.float32)
               for name, shape in config.weight_shapes.items()}
    return GDNRegisterPrefill(GDN(config, weights))


class ILPInstallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_opt_in_keeps_storage_and_s1_object_and_is_idempotent(self):
        model = small_gdn()
        old = model.recurrence
        before = [(n, v.data_ptr(), v.shape) for n, v in model.named_buffers()]
        kernels = install_gdn_ilp(model)
        self.assertIsInstance(model.recurrence, PhaseILPRecurrence)
        self.assertIs(model.recurrence.decode, old)
        self.assertEqual(before, [(n, v.data_ptr(), v.shape) for n, v in model.named_buffers()])
        self.assertEqual(install_gdn_ilp(model), kernels)
        with self.assertRaises(ValueError):
            install_gdn_ilp(model, rows=2)

    def test_export_uses_old_s1_and_new_prefill_op(self):
        for rows in (2, 4):
            model = small_gdn()
            install_gdn_ilp(model, rows)
            for count in (1, 11):
                program = torch.export.export(model.recurrence, make_inputs(count, 2, 7))
                names = [str(node.target) for node in program.graph.nodes if node.op == 'call_function']
                expected = ('qwen_gdn_recurrence_k128_register_v1' if count == 1 else
                            f'qwen_experimental_gdn_recurrence_k128_ilp{rows}_v1')
                self.assertEqual(sum(expected in name for name in names), 1)
                self.assertEqual(sum('qwen_' in name for name in names), 1)

    def test_repaired_policy_preserves_buffers_and_s1_and_export_boundary(self):
        from coreai_gdn_ilp_readout_probe import ReadoutILPRecurrence
        model = small_gdn()
        old = model.recurrence
        before = [(name, value.data_ptr(), value.shape) for name, value in model.named_buffers()]
        kernels = install_gdn_ilp(model, policy='readout-v2')
        self.assertIs(model.recurrence.decode, old)
        self.assertIsInstance(model.recurrence.prefill, ReadoutILPRecurrence)
        self.assertEqual(before, [(name, value.data_ptr(), value.shape) for name, value in model.named_buffers()])
        self.assertEqual(install_gdn_ilp(model, policy='readout-v2'), kernels)
        for count in (1, 11):
            inputs = make_inputs(count, 2, 7)
            program = torch.export.export(model.recurrence, inputs)
            targets = [str(node.target) for node in program.graph.nodes if node.op == 'call_function']
            expected = ('qwen_gdn_recurrence_k128_register_v1' if count == 1 else
                        'qwen_experimental_gdn_recurrence_k128_ilp4_readout_v2')
            self.assertEqual(sum(expected in name for name in targets), 1)
            self.assertEqual(sum('qwen_' in name for name in targets), 1)
            for actual, reference in zip(model.recurrence(*inputs), old(*inputs), strict=True):
                torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            install_gdn_ilp(model, policy='experimental-v1')
        for rows, policy in ((2, 'readout-v2'), (4, 'unknown')):
            other = small_gdn()
            previous = other.recurrence
            with self.assertRaises(ValueError):
                install_gdn_ilp(other, rows, policy=policy)
            self.assertIs(other.recurrence, previous)

    def test_full_attention_chunk_then_s1_matches_original_cpu(self):
        torch.manual_seed(1907)
        for rows in (2, 4):
            baseline, candidate = small_gdn(), small_gdn()
            install_gdn_ilp(candidate, rows)
            config = baseline.config
            old_state = torch.randn(1, 2, 7, 128) * .01
            old_history = (torch.randn(1, 3, config.channels) * .1).half()
            new_state, new_history = old_state.clone(), old_history.clone()
            for count in (4, 4, 1):
                hidden = (torch.randn(1, count, 16) * .1).half()
                expected, old_history, old_state = baseline(hidden, old_history, old_state)
                actual, new_history, new_state = candidate(hidden, new_history, new_state)
                for a, b in ((actual, expected), (new_history, old_history), (new_state, old_state)):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_reject_unknown_recurrence_without_partial_install(self):
        one, two = small_gdn(), small_gdn()
        two.recurrence = torch.nn.Identity()
        whole = torch.nn.ModuleList([one, two])
        old = one.recurrence
        with self.assertRaises(ValueError):
            install_gdn_ilp(whole)
        self.assertIs(one.recurrence, old)
        with self.assertRaises(ValueError):
            install_gdn_ilp(torch.nn.Identity())
        with self.assertRaises(ValueError):
            ILPRecurrence(8)


if __name__ == '__main__':
    unittest.main()
