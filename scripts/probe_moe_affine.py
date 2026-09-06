#!/usr/bin/env python3
"""One Q4 blockwise affine candidate; its extra offset rounding is explicit."""
from collections import Counter
import json
from pathlib import Path
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import types
import numpy as np
import torch
import export_moe as e


def main():
    torch.set_num_threads(4)
    directory = e.DEFAULT_OUTPUT / "affine-probe"
    directory.mkdir(exist_ok=True)
    expert = 42
    weights = e.Source(0).expert(expert)
    prepared, checks = {}, []
    for name, values in weights.items():
        scale = values["scale32"].astype(np.float16)
        if np.any(scale == 0):
            raise ValueError("Zero scale requires a separate exact representation")
        offset = (-values["bias32"] / values["scale32"]).astype(np.float16)
        materialized = ((values["codes"].reshape(*scale.shape, e.GROUP).astype(np.float16) - offset[..., None]) * scale[..., None]).reshape(values["codes"].shape)
        checks.append({"projection": name, "vs_affine_fp32": e.errors(materialized, values["dense32"]),
                       "vs_fp16_rounded_affine": e.errors(materialized, values["dense16"])})
        prepared[name] = (values["codes"].astype(types.nptype_from_builtin(types.uint4)), scale, offset)
    @mb.program(input_specs=[mb.TensorSpec(shape=(1, e.HIDDEN, 1, 32), dtype=types.fp16)], opset_version=ct.target.macOS15)
    def program(x):
        value = mb.reshape(x=mb.transpose(x=x, perm=[0, 3, 2, 1]), shape=[1, 32, e.HIDDEN])
        def linear(v, name):
            data, scale, offset = prepared[name]
            weight = mb.constexpr_blockwise_shift_scale(data=data, scale=scale, offset=offset, name=name + "_q4")
            return mb.linear(x=v, weight=weight)
        gate, up = linear(value, "gate_proj"), linear(value, "up_proj")
        activation = mb.real_div(x=gate, y=mb.add(x=mb.exp(x=mb.mul(x=gate, y=np.float16(-1))), y=np.float16(1)))
        y = linear(mb.mul(x=activation, y=up), "down_proj")
        return mb.transpose(x=mb.reshape(x=y, shape=[1, 32, 1, e.HIDDEN]), perm=[0, 3, 2, 1], name="y")
    model = ct.convert(program, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS15,
                       compute_units=ct.ComputeUnit.CPU_AND_NE, compute_precision=ct.precision.FLOAT16)
    path = directory / "expert0042_affine_linear_s32.mlpackage"
    model.save(str(path))
    plan = e.compute_plan(model)
    cases = []
    for label in ("prefill", "decode"):
        fixture = e.ROOT / f"ane-runner/fixtures/moe-real/converted/{label}.npz"
        value = np.load(fixture)["x"]
        n = value.shape[1]
        x = np.zeros((1, e.HIDDEN, 1, 32), np.float16)
        x[:, :, 0, :n] = value.transpose(0, 2, 1)
        y = model.predict({"x": x})["y"]
        fp16_model = ct.models.MLModel(str(e.DEFAULT_OUTPUT / "layer_0_fp16_linear_s32/expert_0042.mlpackage"), compute_units=ct.ComputeUnit.CPU_AND_NE)
        y_fp16 = fp16_model.predict({"x": x})["y"]
        cases.append({"fixture": label, "valid_tokens": n, "vs_fp16_linear": e.errors(y, y_fp16),
                      "vs_fp32_affine": e.errors(y[..., :n], e.reference(weights, x[..., :n], "fp32_affine")),
                      "padded_outputs_exact_zero": bool(np.count_nonzero(y[..., n:]) == 0)})
    report = {"expert": expert, "model": str(path), "package_bytes": e.package_bytes(path),
              "weight_reconstruction": checks, "compute_plan": plan,
              "plan_preferred_counts": dict(Counter(p["preferred"] for p in plan)), "actual_cases": cases,
              "limitation": "This affine representation adds FP16 rounding in offset=-bias/scale and is not bit-exact source Q4 preservation. Plan preference is not a hardware trace."}
    e.write_json(directory / "report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "compute_plan"}, indent=2))


if __name__ == "__main__":
    main()
