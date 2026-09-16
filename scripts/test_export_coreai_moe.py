"""Small CPU-only numerical/provenance checks; no model weights or runtime calls."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import export_coreai_moe as exporter


class CoreAIExpertExportTests(unittest.TestCase):
    def write_fixture(self, directory, value, layer=0):
        path = directory / "prefill.json"
        fixture = {"inputs": {"x": exporter.tensor_json(value)},
                   "expectedOutputs": {"shared_down": exporter.tensor_json(np.zeros_like(value))}}
        exporter.write_json(path, fixture)
        exporter.write_json(directory / "provenance.json", {"phases": {"prefill": {
            "source_metadata": {"layer_index": str(layer)}, "fixture_sha256": exporter.sha256_file(path)}}})
        return path

    def weights(self):
        result = {}
        for name, value in {
            "gate_proj": np.array([[1, 0], [0, -1]], dtype=np.float32),
            "up_proj": np.array([[0.5, 0], [0, 0.25]], dtype=np.float32),
            "down_proj": np.array([[1, 2], [-1, 0.5]], dtype=np.float32),
        }.items():
            result[name] = {"dense32": value, "dense16": value.astype(np.float16)}
        return result

    def test_padding_and_layer_provenance_prevent_wrong_captured_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            value = np.array([[[1, 2], [3, 4]]], dtype=np.float32)
            self.write_fixture(directory, value)
            inputs, fixture, evidence = exporter.fixture_inputs(directory, 32, 2, weight_layer=1)
            np.testing.assert_array_equal(inputs["actual"][:, :2], value)
            self.assertEqual(np.count_nonzero(inputs["actual"][:, 2:]), 0)
            self.assertEqual(evidence["valid_tokens"], 2)
            self.assertEqual(evidence["padding_tokens"], 30)
            self.assertEqual(evidence["input_capture_layer"], 0)
            self.assertFalse(evidence["matches_weight_layer"])
            self.assertEqual(evidence["input_scope"], "cross-layer activation replay")
            rows = exporter.save_cases(directory, inputs, exporter.reference_models(self.weights()), fixture, evidence, "shared")
            self.assertNotIn("fp32_source_vs_captured_bf16_shared_down", rows[0])
            recorded = json.loads((directory / "actual.json").read_text())
            self.assertEqual(recorded["inputs"]["x"]["dtype"], "float16")
            self.assertEqual(recorded["expectedOutputs"]["y"]["shape"], [1, 32, 2])
            self.assertEqual(rows[1]["fp16_vs_fp32_fp16_weights"]["max_abs"], 0)

    def test_cpu_reference_matches_independent_numpy_equation(self):
        weights = self.weights()
        x = np.array([[[0.5, -0.75], [0, 0]]], dtype=np.float16)
        gate = np.array([[[0.5, 0.75], [0, 0]]], dtype=np.float64)
        up = np.array([[[0.25, -0.1875], [0, 0]]], dtype=np.float64)
        middle = gate / (1 + np.exp(-gate)) * up
        expected = np.stack((middle[..., 0] + 2 * middle[..., 1],
                             -middle[..., 0] + 0.5 * middle[..., 1]), axis=-1)
        refs = exporter.cpu_references(exporter.reference_models(weights), x)
        np.testing.assert_allclose(refs["fp32_source_weights"], expected, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(refs["fp16"], expected, rtol=0.002, atol=1e-4)
        self.assertEqual(exporter.errors(refs["fp16"][:, 1:], refs["fp32_source_weights"][:, 1:])["max_abs"], 0)

    def test_fixture_hash_and_hidden_width_are_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = self.write_fixture(directory, np.zeros((1, 2, 2), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "fixture"):
                exporter.fixture_inputs(directory, 32, 3, 0)
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "hash"):
                exporter.fixture_inputs(directory, 32, 2, 0)

    def test_fp32_projections_preserve_fp16_buffers_and_activation_boundaries(self):
        model = exporter.SwiGLU(self.weights(), fp32_projections=True).eval()
        x = torch.tensor([[[0.431, -1.271]]], dtype=torch.float16)
        with patch.object(exporter.F, "linear", wraps=exporter.F.linear) as linear:
            with patch.object(exporter.F, "silu", wraps=exporter.F.silu) as silu:
                actual = model(x)
        self.assertEqual(linear.call_count, 3)
        for call in linear.call_args_list:
            self.assertEqual(call.args[0].dtype, torch.float32)
            self.assertEqual(call.args[1].dtype, torch.float32)
        self.assertEqual(silu.call_args.args[0].dtype, torch.float16)
        self.assertEqual(actual.dtype, torch.float16)
        self.assertTrue(all(weight.dtype == torch.float16 for weight in model.buffers()))
        self.assertTrue(all(not reference.fp32_projections for reference in exporter.reference_models(self.weights()).values()))
        half = lambda value: value.astype(np.float16).astype(np.float32)
        values = x.numpy().astype(np.float32)
        weights = self.weights()
        gate = half(values @ weights["gate_proj"]["dense32"].T)
        up = half(values @ weights["up_proj"]["dense32"].T)
        middle = half(half(gate / (1 + np.exp(-gate))) * up)
        expected = half(middle @ weights["down_proj"]["dense32"].T)
        np.testing.assert_array_equal(actual.float().numpy(), expected)

    def test_errors_reject_broadcast_and_nonfinite_values(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            exporter.errors(np.zeros((1, 2)), np.zeros((2,)))
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            exporter.errors(np.array([np.nan]), np.zeros(1))
        self.assertIsNone(exporter.errors(np.ones(1), np.zeros(1))["relative_l2"])


if __name__ == "__main__":
    unittest.main()
