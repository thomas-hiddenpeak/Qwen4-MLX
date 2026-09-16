"""CPU-only multi-token and state-boundary checks for fused CoreAI PD authoring.

Uses small synthetic weights with the real group64 Q4 packing. No source-model
load, CoreAI runtime execution, server, or accelerator is required. Registration
checks author only tiny temporary assets and remove them when the test finishes.
"""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from export_coreai_dense import DenseConfig, Embedding, HCRead, HCWrite, Head, PLE
from export_coreai_gdn import GDN, GDNConfig
from export_coreai_pd import DecoderLayer, LastHead, export_shared, install_stable_projections
from export_coreai_q4_moe import PROJECTIONS, Q4MoE


class CoreAIPrefillTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def config(self):
        # Three streams, asymmetric rank/PLE/head dimensions, and a nine-token
        # history catch hardcoded real-model or S1 reshape assumptions.
        return DenseConfig(hidden=64, streams=3, low_rank=5, epsilon=1e-6,
                           vocabulary=23, ple_dim=8, ple_kernel=4, ple_dilation=3)

    def weights(self, shapes, seed):
        rng = np.random.default_rng(seed)
        return {name: (rng.standard_normal(shape) * 0.035 + (1 if "norm" in name else 0)).astype(np.float32)
                for name, shape in shapes.items()}

    def hc_weights(self, config, seed):
        return self.weights({"input_mix_weight_down.weight": (config.low_rank, config.width),
                             "input_mix_weight_up.weight": (config.width, config.low_rank),
                             "hc_norm.weight": (config.width,),
                             "block_inject_weight.weight": (config.streams, config.width)}, seed)

    def ple(self, config):
        shapes = {"key_proj.weight": (config.width, config.ple_dim),
                  "value_proj.weight": (config.hidden, config.ple_dim),
                  "norm_key.weight": (config.width,), "norm_query.weight": (config.width,),
                  "norm_conv.weight": (config.width,),
                  "conv1d.weight": (config.width, config.ple_kernel, 1)}
        return PLE(config, self.weights(shapes, 512)).eval()

    def moe(self, hidden=64):
        experts, intermediate, shared_intermediate, top_k = 12, 128, 96, 10
        rng = np.random.default_rng(713)
        # Controlled paired and nonadjacent ties; each prompt token can select a
        # different order. Other neural weights remain dense and asymmetric.
        router = np.zeros((experts, hidden), np.float32)
        router[:, 0] = (np.arange(experts) // 2) * 0.25
        router[:, 1] = (np.arange(experts) % 3) * 0.125
        shared_router = (rng.standard_normal((1, hidden)) * 0.035).astype(np.float32)
        shared = {}
        quantized = {}
        for name in PROJECTIONS:
            output, inputs = (hidden, intermediate) if name == "down_proj" else (intermediate, hidden)
            shared_shape = (hidden, shared_intermediate) if name == "down_proj" else (shared_intermediate, hidden)
            shared[name] = (rng.standard_normal(shared_shape) * 0.035).astype(np.float32)
            packed = rng.integers(0, 2**32, (experts, output, inputs // 8), dtype=np.uint32)
            scales = np.full((experts, output, inputs // 64), 0.0078125, np.float32)
            quantized[name] = (packed, scales, -scales * 7)
        return Q4MoE(router, shared_router, shared, quantized, top_k).eval()

    def tensor(self, shape, seed=91, scale=0.125):
        generator = torch.Generator().manual_seed(seed)
        return (torch.randn(shape, generator=generator) * scale).half()

    def assert_near(self, actual, expected):
        # Declared before execution: tolerate FP16 GEMM boundary rounding, not
        # token mixing, wrong routing, or lost recurrent/convolution history.
        torch.testing.assert_close(actual, expected, rtol=5e-3, atol=3e-4)
        self.assertTrue(bool(torch.isfinite(actual).all()))

    def test_embedding_tiles_each_token_without_mixing_token_and_stream_axes(self):
        c = self.config()
        weights = np.arange(c.vocabulary * c.hidden, dtype=np.float32).reshape(c.vocabulary, c.hidden) / 128
        model = Embedding(c, weights)
        tokens = torch.tensor([17, 2, 17, 22], dtype=torch.int32)
        actual = model(tokens)
        selected = torch.from_numpy(weights).half()[tokens.long()]
        expected = torch.cat([selected] * c.streams, dim=-1).unsqueeze(0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual, torch.cat([model(token[None]) for token in tokens], dim=1), rtol=0, atol=0)

    def test_hc_read_write_multi_token_matches_independent_s1_calls(self):
        c = self.config()
        weights = self.hc_weights(c, 513)
        read, write = HCRead(c, weights), HCWrite(c)
        stream = self.tensor((1, 8, c.width))
        output = self.tensor((1, 8, c.hidden), seed=92)
        mixed, injection = read(stream)
        scalar = [read(stream[:, index:index+1]) for index in range(8)]
        self.assertEqual(list(mixed.shape), [1, 8, c.hidden])
        self.assertEqual(list(injection.shape), [1, 8, c.streams, 1])
        self.assert_near(mixed, torch.cat([value[0] for value in scalar], dim=1))
        self.assert_near(injection, torch.cat([value[1] for value in scalar], dim=1))
        actual = write(stream, output, injection)
        expected = torch.cat([write(stream[:, i:i+1], output[:, i:i+1], scalar[i][1]) for i in range(8)], dim=1)
        self.assert_near(actual, expected)
        self.assert_near(HCRead(c, weights, with_injection=False)(stream), mixed)

    def test_ple_two_chunks_then_decode_preserves_dilated_history(self):
        c = self.config()
        model = self.ple(c)
        stream = self.tensor((1, 9, c.width), seed=93)
        embedding = self.tensor((1, 9, c.ple_dim), seed=94, scale=0.5)
        history = self.tensor((1, c.ple_history, c.width), seed=95)
        saved = history.clone()
        expected, expected_state = model(stream, embedding, history)
        for lengths in ((4, 4, 1), (1,) * 9):
            offset, current, outputs = 0, history, []
            for count in lengths:
                value, current = model(stream[:, offset:offset+count], embedding[:, offset:offset+count], current)
                outputs.append(value)
                offset += count
            self.assert_near(torch.cat(outputs, dim=1), expected)
            self.assert_near(current, expected_state)
        torch.testing.assert_close(history, saved, rtol=0, atol=0)
        cold, _ = model(stream[:, :4], embedding[:, :4], torch.zeros_like(history))
        self.assertGreater(float((cold - expected[:, :4]).abs().max()), 1e-3)

    def test_last_head_uses_only_final_prompt_position(self):
        c = self.config()
        head = Head(c, self.hc_weights(c, 514), self.weights({"head": (c.vocabulary, c.hidden)}, 515)["head"])
        model = LastHead(head)
        stream = self.tensor((1, 4, c.width), seed=96)
        actual = model(stream)
        self.assertEqual(list(actual.shape), [1, 1, c.vocabulary])
        torch.testing.assert_close(actual, head(stream[:, -1:]), rtol=0, atol=0)
        changed = stream.clone()
        changed[:, :-1] = changed[:, :-1] * 5 + 1
        torch.testing.assert_close(model(changed), actual, rtol=0, atol=0)

    def test_moe_prefill_matches_s1_and_routes_each_token_with_lowest_id_ties(self):
        model = self.moe()
        x = torch.zeros(1, 4, model.hidden, dtype=torch.float16)
        x[0, :, 0] = torch.tensor([1, -1, 0, .25])
        x[0, :, 1] = torch.tensor([0, 0, 0, 1])
        actual = model.prefill(x)
        scalar = [model(x[:, index:index+1]) for index in range(4)]
        for index in range(3):
            expected = torch.cat([value[index] for value in scalar], dim=1)
            if index == 1:
                torch.testing.assert_close(actual[index], expected, rtol=0, atol=0)
            else:
                self.assert_near(actual[index], expected)
        # Independently rank exact FP16 router logits rather than asking the
        # implementation's scalar router to define the expected tie policy.
        logits = (x.numpy().astype(np.float32) @ model.router.numpy().astype(np.float32).T).astype(np.float16)[0]
        expected_ids, expected_scores = [], []
        for row in logits:
            ids = sorted(range(model.experts), key=lambda i: (-float(row[i]), i))[:model.top_k]
            exp = np.exp(row.astype(np.float32) - np.max(row.astype(np.float32)))
            probabilities = (exp / exp.sum()).astype(np.float16)
            selected = probabilities[ids]
            denominator = selected[0]
            for value in selected[1:]:
                denominator = np.float16(denominator + value)
            expected_ids.append(ids)
            expected_scores.append((selected / denominator).astype(np.float16))
        np.testing.assert_array_equal(actual[1].numpy()[0], expected_ids)
        np.testing.assert_array_equal(actual[1].numpy()[0, 2], np.arange(model.top_k))
        np.testing.assert_allclose(actual[2].numpy()[0], expected_scores, rtol=1e-3, atol=1e-4)
        self.assertFalse(torch.equal(actual[1][:, 0], actual[1][:, 1]))
        # A permutation must only permute outputs/routing; no cross-token top-k.
        permutation = torch.tensor([2, 0, 3, 1])
        shuffled = model(x[:, permutation])
        for value, target in zip(shuffled, actual):
            if not value.is_floating_point():
                torch.testing.assert_close(value, target[:, permutation], rtol=0, atol=0)
            else:
                self.assert_near(value, target[:, permutation])

    def decoder(self, with_ple):
        c = self.config()
        config = GDNConfig(c.hidden, key_heads=2, value_heads=6, key_dim=3, value_dim=4, kernel=4, epsilon=c.epsilon)
        gdn = GDN(config, self.weights(config.weight_shapes, 516)).eval()
        model = DecoderLayer(gdn, HCRead(c, self.hc_weights(c, 517)), self.moe(),
                             HCRead(c, self.hc_weights(c, 518)), HCWrite(c), state_count=2,
                             ple=self.ple(c) if with_ple else None).eval()
        history = self.tensor((1, config.kernel - 1, config.channels), seed=97)
        state = self.tensor((1, config.value_heads, config.value_dim, config.key_dim), seed=98).float()
        return c, model, history, state

    def test_fused_gdn_decoder_s4_matches_four_s1_calls_with_and_without_ple(self):
        for with_ple in (False, True):
            with self.subTest(ple=with_ple):
                c, model, history, state = self.decoder(with_ple)
                stream = self.tensor((1, 4, c.width), seed=99)
                embedding = self.tensor((1, 4, c.ple_dim), seed=100, scale=0.5)
                states = (history, state)
                if with_ple:
                    states += (self.tensor((1, c.ple_history, c.width), seed=101),)
                saved = tuple(value.clone() for value in states)
                args = (stream, embedding, *states) if with_ple else (stream, *states)
                actual = model(*args)
                current, pieces = states, []
                for index in range(4):
                    row = stream[:, index:index+1]
                    row_args = (row, embedding[:, index:index+1], *current) if with_ple else (row, *current)
                    result = model(*row_args)
                    pieces.append(result[0])
                    current = result[1:]
                expected = (torch.cat(pieces, dim=1), *current)
                self.assertEqual(len(actual), 4 if with_ple else 3)
                for value, target in zip(actual, expected):
                    self.assert_near(value, target)
                for value, target in zip(states, saved):
                    torch.testing.assert_close(value, target, rtol=0, atol=0)
                # Exercise functional capture of varargs/state outputs without
                # calling TorchConverter or any CoreAI runtime.
                captured = torch.export.export(model, args).module()(*args)
                for value, target in zip(captured, actual):
                    torch.testing.assert_close(value, target, rtol=0, atol=0)

    def test_optional_stable_install_preserves_fused_cpu_oracle_and_existing_buffers(self):
        # This covers GDN, both HC readers, MoE router/shared projections, and
        # PLE in their real composition, including nonzero recurrent/history I/O.
        # Exactness here concerns the CPU callback, not Metal reduction parity.
        c, model, history, state = self.decoder(with_ple=True)
        stream = self.tensor((1, 4, c.width), seed=102)
        embedding = self.tensor((1, 4, c.ple_dim), seed=103)
        ple_state = self.tensor((1, c.ple_history, c.width), seed=104)
        examples = [(stream[:, :count], embedding[:, :count], history, state, ple_state)
                    for count in (1, 4)]
        expected = [model(*args) for args in examples]
        buffers = {name: (value.data_ptr(), value.clone()) for name, value in model.named_buffers()}
        baseline = torch.export.export(model, examples[-1])
        linear_count = sum(node.target == torch.ops.aten.linear.default for node in baseline.graph.nodes)
        self.assertGreater(linear_count, 0)
        install_stable_projections(model)
        for args, targets in zip(examples, expected):
            for actual, target in zip(model(*args), targets):
                torch.testing.assert_close(actual, target, rtol=0, atol=0)
        self.assertEqual(set(buffers), set(dict(model.named_buffers())))
        for name, value in model.named_buffers():
            address, saved = buffers[name]
            self.assertEqual(value.data_ptr(), address)
            torch.testing.assert_close(value, saved, rtol=0, atol=0)
        graph = torch.export.export(model, examples[-1])
        operations = [str(node.target) for node in graph.graph.nodes if node.op == "call_function"]
        self.assertEqual(sum("qwen_fp16_dense_gemv_stable_v1" in name for name in operations), linear_count)
        self.assertFalse(any(node.target == torch.ops.aten.linear.default for node in graph.graph.nodes))
        for actual, target in zip(graph.module()(*examples[-1]), expected[-1]):
            torch.testing.assert_close(actual, target, rtol=0, atol=0)
        # Opt-in must not patch classes/global helpers or another model instance.
        _, untouched, _, _ = self.decoder(with_ple=True)
        untouched_graph = torch.export.export(untouched, examples[-1])
        self.assertFalse(any("qwen_fp16_dense_gemv_stable_v1" in str(node.target)
                             for node in untouched_graph.graph.nodes))

    def test_optional_stable_qsa_bound_helper_preserves_cpu_outputs_and_sparse_state(self):
        from check_coreai_qsa_chunks import small_module
        from export_coreai_qsa import BINDINGS, INPUT_NAMES, OUTPUT_NAMES, initial_state
        model = small_module(capacity=32)
        seed = {"x": self.tensor((1, 9, model.hidden), seed=105), **initial_state(model)}
        result = dict(zip(OUTPUT_NAMES, model(*(seed[name] for name in INPUT_NAMES))))
        state = {name: result[output] for name, output in BINDINGS.items()}
        x = self.tensor((1, 4, model.hidden), seed=106)
        examples = [tuple({"x": x[:, :count], **state}[name] for name in INPUT_NAMES) for count in (1, 4)]
        expected = [model(*args) for args in examples]
        install_stable_projections(model)
        for args, targets in zip(examples, expected):
            actual = model(*args)
            for value, target in zip(actual, targets):
                torch.testing.assert_close(value, target, rtol=0, atol=0)
            graph = torch.export.export(model, args)
            # Five weight-name lookups must remain opaque projections; state,
            # rotary and sparse-mask graph operations are intentionally retained.
            self.assertEqual(sum("qwen_fp16_dense_gemv_stable_v1" in str(node.target)
                                 for node in graph.graph.nodes), 5)
            for value, target in zip(graph.module()(*args), targets):
                torch.testing.assert_close(value, target, rtol=0, atol=0)

    def test_optional_stable_small_gdn_and_qsa_register_both_export_functions(self):
        from coreai_dense_metal import get_dense_kernel
        from check_coreai_qsa_chunks import small_module
        from export_coreai_qsa import OUTPUT_NAMES as QSA_OUTPUTS, initial_state
        config = GDNConfig(5, 2, 6, 3, 4, 4, 1e-6)
        gdn = GDN(config, self.weights(config.weight_shapes, 519)).eval()
        qsa = small_module(capacity=32)
        cases = [
            ("gdn", gdn, "hidden", config.hidden,
             {"conv_history": self.tensor((1, config.kernel - 1, config.channels), seed=107),
              "recurrent_state": self.tensor((1, config.value_heads, config.value_dim, config.key_dim), seed=108).float()},
             ("output", "next_conv_history", "next_recurrent_state")),
            ("qsa", qsa, "x", qsa.hidden, initial_state(qsa), QSA_OUTPUTS),
        ]
        with tempfile.TemporaryDirectory(prefix="coreai-stable-pd-test-") as directory:
            for name, model, input_name, width, states, outputs in cases:
                with self.subTest(attention=name):
                    install_stable_projections(model)
                    examples = {function: {input_name: self.tensor((1, count, width), seed=109), **states}
                                for function, count in (("main", 1), ("prefill", 4))}
                    path = Path(directory) / (name + ".aimodel")
                    asset = export_shared(model, examples, outputs, path, [get_dense_kernel()])
                    self.assertGreater(asset["modelBytes"], 0)
                    self.assertLess(asset["modelBytes"], 1_000_000)
                    self.assertTrue((path / "main.mlirb").is_file())
                    self.assertEqual(asset["inputNames"], list(examples["main"]))
                    self.assertEqual(asset["outputNames"], list(outputs))


if __name__ == "__main__":
    unittest.main()
