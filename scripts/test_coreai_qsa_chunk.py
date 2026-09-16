"""Bounded CPU tests; custom Metal execution and device numerics are separate."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from coreai_qsa_chunk import (QwenQSAChunk, export_probe, get_pool_kernel,
                             make_tiny, pool_reference, seeded_state,
                             replay_state, repeat_activations, write_tensor_json)
from export_coreai_qsa import BINDINGS, INPUT_NAMES, OUTPUT_NAMES


class QSAChunkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def values(self, source, count, offset, pooled_count=None):
        generator = torch.Generator().manual_seed(1027 + count)
        x = (torch.randn(1, count, source.hidden, generator=generator) * 0.2).half()
        return {"x": x, **seeded_state(source, offset, pooled_count=pooled_count)}

    def run_model(self, model, values):
        with torch.inference_mode():
            return dict(zip(OUTPUT_NAMES, model(*(values[key] for key in INPUT_NAMES))))

    def test_large_chunks_preserve_source_math_and_all_inputs(self):
        source = make_tiny(capacity=3072)
        candidate = QwenQSAChunk(source).eval()
        for count, offset, pool in ((128, 0, None), (128, 3, None), (256, 2051, None),
                                    (128, 2051, 0), (256, 2816, None), (512, 0, None), (512, 2051, None)):
            with self.subTest(count=count, offset=offset, pooled_count=pool):
                values = self.values(source, count, offset, pool)
                saved = {name: value.clone() for name, value in values.items()}
                expected = self.run_model(source, values)
                actual = self.run_model(candidate, values)
                for name in OUTPUT_NAMES:
                    torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
                for name in values:
                    torch.testing.assert_close(values[name], saved[name], rtol=0, atol=0)
        self.assertIs(candidate.source, source)
        self.assertEqual(candidate.source.q_proj_weight.data_ptr(), source.q_proj_weight.data_ptr())

    def test_projection_only_replay_state_matches_full_layer_calls(self):
        source = make_tiny(capacity=32)
        rows = self.values(source, 7, 0)["x"]
        for offset in (0, 4, 11, 32):
            with self.subTest(offset=offset):
                expected = seeded_state(source, 0)
                for start in range(0, offset, 4):
                    x = repeat_activations(rows, min(4, offset - start), start)
                    output = self.run_model(source, {"x": x, **expected})
                    expected = {name: output[out] for name, out in BINDINGS.items()}
                actual = replay_state(source, rows, offset, projection_chunk=4)
                for name in BINDINGS:
                    torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)

    def test_streaming_json_preserves_tensor_values_and_fixture_contract(self):
        tensors = {"large": torch.linspace(-3, 3, 40000).half().reshape(1, 40000),
                   "cursor": torch.tensor([8192], dtype=torch.int32)}
        with tempfile.TemporaryDirectory(prefix="coreai-qsa-json-test-") as directory:
            path = Path(directory) / "fixture.json"
            for inputs in (None, {"x": tensors["cursor"]}):
                write_tensor_json(path, tensors, inputs=inputs)
                fixture = json.loads(path.read_text())
                if inputs is not None:
                    self.assertEqual(set(fixture), {"inputs", "expectedOutputs"})
                    fixture = fixture["expectedOutputs"]
                for name, expected in tensors.items():
                    self.assertEqual(fixture[name]["shape"], list(expected.shape))
                    self.assertEqual(fixture[name]["values"], expected.flatten().float().tolist())

    def test_s128_and_s256_then_decode_match_s1_state_continuation(self):
        source = make_tiny(capacity=2560)
        candidate = QwenQSAChunk(source).eval()
        for count in (128, 256):
            with self.subTest(count=count):
                values = self.values(source, count + 4, 2051)
                tokens = values.pop("x")
                def execute(lengths):
                    current = values
                    position, ys, masks = 0, [], []
                    for length in lengths:
                        result = self.run_model(candidate, {"x": tokens[:, position:position+length], **current})
                        current = {key: result[out] for key, out in BINDINGS.items()}
                        ys.append(result["y"])
                        masks.append(result["attention_mask"])
                        position += length
                    return torch.cat(ys, 1), torch.cat(masks, 2), current
                chunk = execute((count, 1, 1, 1, 1))
                scalar = execute((1,) * (count + 4))
                torch.testing.assert_close(chunk[0], scalar[0], rtol=0.005, atol=0.0003)
                torch.testing.assert_close(chunk[1], scalar[1], rtol=0, atol=0)
                for name in BINDINGS:
                    torch.testing.assert_close(chunk[2][name], scalar[2][name], rtol=0.005, atol=0.0003)

    def test_pool_commits_only_requested_blocks_and_matches_independent_numpy_math(self):
        source = make_tiny(capacity=32)
        state = seeded_state(source, 19)
        raw = state["raw_cache"]
        previous = state["pooled_cache"]
        for begin, end in ((0, 0), (2, 2), (2, 5), (0, 8)):
            with self.subTest(begin=begin, end=end):
                result = pool_reference(raw, previous, source.indexer_k_layernorm_weight,
                    source.cosine, source.sine, torch.tensor([begin], dtype=torch.int32),
                    torch.tensor([end], dtype=torch.int32), ratio=4, rope_dim=4, epsilon=source.eps)
                expected = previous.numpy().copy()
                for block in range(begin, end):
                    mean = raw.numpy()[0, block*4:block*4+4].astype(np.float32).mean(0).astype(np.float16)
                    norm = (mean.astype(np.float32) / np.sqrt(np.mean(mean.astype(np.float32)**2) + np.float32(source.eps))
                            * source.indexer_k_layernorm_weight.numpy().astype(np.float32)).astype(np.float16)
                    cos = source.cosine.numpy()[block*4].astype(np.float16)
                    sin = source.sine.numpy()[block*4].astype(np.float16)
                    a, b = norm[:2].copy(), norm[2:4].copy()
                    norm[:2] = (a * cos).astype(np.float16) - (b * sin).astype(np.float16)
                    norm[2:4] = (b * cos).astype(np.float16) + (a * sin).astype(np.float16)
                    expected[0, block] = norm
                np.testing.assert_allclose(result.numpy(), expected, rtol=0.001, atol=0.001)
                self.assertTrue(torch.equal(result[:, :begin], previous[:, :begin]))
                self.assertTrue(torch.equal(result[:, end:], previous[:, end:]))

    def test_export_keeps_runtime_cursors_and_opaque_incremental_pool(self):
        source = make_tiny(capacity=2304)
        model = QwenQSAChunk(source).eval()
        values = self.values(source, 128, 0)
        graph = torch.export.export(model, tuple(values[name] for name in INPUT_NAMES))
        targets = [str(node.target) for node in graph.graph.nodes if node.op == "call_function"]
        self.assertEqual(sum("qwen_mpp_fp16_gemm_" in name for name in targets), 5)
        self.assertEqual(sum("qwen_qsa_incremental_pool_" in name for name in targets), 1)
        # The same graph traced at offset0 must consume a different cursor and
        # a lagging pooled_count, without retracing/baking host scalar values.
        moved = self.values(source, 128, 2051, pooled_count=0)
        actual = graph.module()(*(moved[name] for name in INPUT_NAMES))
        expected = self.run_model(source, moved)
        for value, name in zip(actual, OUTPUT_NAMES):
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)

    def test_s1_retains_source_projections_and_shared_incremental_pool(self):
        source = make_tiny(capacity=2304)
        model = QwenQSAChunk(source).eval()
        values = self.values(source, 1, 2055)
        expected = self.run_model(source, values)
        actual = self.run_model(model, values)
        for name in OUTPUT_NAMES:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
        graph = torch.export.export(model, tuple(values[name] for name in INPUT_NAMES))
        targets = [str(node.target) for node in graph.graph.nodes if node.op == "call_function"]
        self.assertFalse(any("qwen_mpp_fp16_gemm_" in name for name in targets))
        self.assertEqual(sum("qwen_qsa_incremental_pool_" in name for name in targets), 1)

    def test_optional_half_sdpa_changes_only_prefill_precision_not_cache_or_decode(self):
        source = make_tiny(capacity=4352)
        candidate = QwenQSAChunk(source, prefill_sdpa_fp16=True).eval()
        for count, offset in ((1, 2055), (128, 2051), (1024, 2051), (2048, 2051)):
            with self.subTest(count=count):
                values = self.values(source, count, offset)
                observed = []
                hook = source.sdpa.register_forward_pre_hook(lambda module, args: observed.append(args[0].dtype))
                actual = self.run_model(candidate, values)
                hook.remove()
                expected = self.run_model(source, values)
                self.assertEqual(observed, [torch.float32 if count == 1 else torch.float16])
                for name in OUTPUT_NAMES[1:]:
                    torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
                torch.testing.assert_close(actual["y"], expected["y"],
                    rtol=0 if count == 1 else 0.005, atol=0 if count == 1 else 0.0003)

    def test_optional_half_sdpa_export_retains_float_decode_and_half_prefill(self):
        with tempfile.TemporaryDirectory(prefix="coreai-qsa-half-test-") as directory:
            path = Path(directory) / "asset"
            report = export_probe(make_tiny(capacity=512), path, chunks=(128,),
                                  offsets=[0], prefill_sdpa_fp16=True, decode_steps=1, verify_candidate_cpu=False)
            self.assertTrue(report["prefillSDPAFP16"])
            self.assertFalse(report["candidateCPUExecuted"])
            for row in json.loads((path / "cpu-checks.json").read_text()):
                self.assertFalse(row["candidateCPUExecuted"])
                self.assertIsNone(row["checks"])
                self.assertTrue(row["referenceOutputsFinite"])
            self.assertEqual(report["sdpaShapeAudit"]["main"]["operandDType"], "float32")
            self.assertEqual(report["sdpaShapeAudit"]["prefill_s128"]["operandDType"], "float16")
            for function, dtype in (("main", "f32"), ("prefill_s128", "f16")):
                lines = (path / f"coreai-{function}-after.txt").read_text().splitlines()
                sdpa = next(line for line in lines if "coreai.invoke" in line and "sdpa" in line)
                self.assertIn(f"x8x{dtype}>", sdpa)
            sequence = json.loads((path / report["sequences"][0]).read_text())
            self.assertEqual(len(sequence["steps"]), 2)

    def test_small_asset_has_shared_s1_s128_functions_and_state_only_in_initial_fixture(self):
        with tempfile.TemporaryDirectory(prefix="coreai-qsa-chunk-test-") as directory:
            path = Path(directory) / "asset"
            report = export_probe(make_tiny(capacity=512), path, chunks=(128,))
            self.assertLess(report["modelBytes"], 1_000_000)
            self.assertTrue((path / "qsa-chunk.aimodel/main.mlirb").is_file())
            self.assertEqual(report["models"]["s1"]["path"], report["models"]["s128"]["path"])
            for sequence_name in report["sequences"]:
                sequence = json.loads((path / sequence_name).read_text())
                state = json.loads((path / sequence["initialState"]).read_text())
                self.assertEqual(set(state), set(BINDINGS))
                for step in sequence["steps"]:
                    fixture = json.loads((path / step["fixture"]).read_text())
                    self.assertEqual(set(fixture["inputs"]), {"x"})
                    self.assertTrue(set(BINDINGS.values()) <= set(fixture["expectedOutputs"]))
            self.assertEqual(report["sdpaShapeAudit"]["prefill_s128"]["keysAndValues"], [1, 1, 512, 8])
            for name in report["sourceFiles"]:
                self.assertTrue((path / name).is_file())


if __name__ == "__main__":
    unittest.main()
