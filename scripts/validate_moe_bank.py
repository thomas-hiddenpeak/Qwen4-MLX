#!/usr/bin/env python3
"""Validate available expert packages on captured activations and routing.

Routing IDs and probabilities here come from capture, isolating expert numerics.
Swift's independent dynamic routing is validated by its own integration report.
"""
import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import coremltools as ct
import numpy as np
import torch
import export_moe as e


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, default=e.ROOT / "ane-runner/fixtures/moe-real/converted")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    directory = args.manifest.parent
    capacity = manifest["token_capacity"]
    torch.set_num_threads(4)
    source = e.Source(manifest["layer_index"])
    cases = {name: dict(np.load(args.fixtures / f"{name}.npz")) for name in ("prefill", "decode")}
    needed = sorted(set(int(i) for case in cases.values() for i in case["selected_experts"].ravel()))
    missing = [i for i in needed if str(i) not in manifest["experts"]]
    if missing:
        raise ValueError(f"Missing actually routed experts: {missing}")
    sums = {name: {ref: np.zeros_like(case["x"]) for ref in ("coreml", "fp32_affine", "fp32_fp16_weights", "fp16")}
            for name, case in cases.items()}
    shared = {}
    rows = []
    plan_counts = Counter()
    for identifier in [*needed, "shared"]:
        is_shared = identifier == "shared"
        path = directory / (manifest["shared_expert"] if is_shared else manifest["experts"][str(identifier)])
        weights = source.shared() if is_shared else source.expert(identifier)
        model = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
        plan = e.compute_plan(model)
        counts = Counter(p["preferred"] for p in plan if not p["op"].split(".")[-1].startswith("constexpr_"))
        plan_counts.update(counts)
        row = {"expert": identifier, "compute_plan": plan, "plan_preferred_counts": dict(counts), "cases": []}
        for name, case in cases.items():
            ids = case["selected_experts"]
            if not is_shared and identifier not in ids:
                continue
            x = case["x"].astype(np.float16)
            n = x.shape[1]
            if n > capacity:
                raise ValueError("Captured activation exceeds bank capacity")
            padded = np.zeros((1, e.HIDDEN, 1, capacity), np.float16)
            padded[:, :, 0, :n] = x.transpose(0, 2, 1)
            prediction = model.predict({"x": padded})["y"]
            outputs = {"coreml": prediction[:, :, 0, :n].transpose(0, 2, 1)}
            for mode in ("fp32_affine", "fp32_fp16_weights", "fp16"):
                outputs[mode] = e.reference(weights, x.transpose(0, 2, 1)[:, :, None], mode)[:, :, 0].transpose(0, 2, 1)
            record = {"fixture": name, "valid_tokens": n,
                      "padded_outputs_exact_zero": bool(np.count_nonzero(prediction[..., n:]) == 0),
                      "errors": {mode: e.errors(outputs["coreml"], outputs[mode]) for mode in outputs if mode != "coreml"}}
            if is_shared:
                shared[name] = outputs
                record["vs_native_shared_down"] = e.errors(outputs["coreml"], case["shared_down"])
            else:
                native_actual, predicted_selected = [], []
                for token in range(n):
                    for slot in range(e.TOP_K):
                        if int(ids[0, token, slot]) != identifier:
                            continue
                        for mode, output in outputs.items():
                            sums[name][mode][0, token] += output[0, token] * case["routing_weights"][0, token, slot]
                        if "selected_expert_outputs" in case:
                            native_actual.append(case["selected_expert_outputs"][0, token, slot])
                            predicted_selected.append(outputs["coreml"][0, token])
                if native_actual:
                    record["vs_native_selected_expert"] = e.errors(np.stack(predicted_selected), np.stack(native_actual))
            row["cases"].append(record)
        rows.append(row)
        del model, weights
        gc.collect()
    combined = {}
    for name, case in cases.items():
        outputs = {}
        for mode in sums[name]:
            outputs[mode + "_routed_sum"] = sums[name][mode]
            outputs[mode + "_shared_down"] = shared[name][mode]
            outputs[mode + "_y"] = sums[name][mode] + shared[name][mode] * case["shared_gate"]
        np.savez(directory / f"validation_{name}.npz", **outputs)
        y = outputs["coreml_y"]
        combined[name] = {"source_fixture": str(args.fixtures / f"{name}.npz"),
                          "source_fixture_sha256": e.sha256_file(args.fixtures / f"{name}.npz"),
                          "native_routed_sum_error": e.errors(outputs["coreml_routed_sum"], case["routed_sum"]),
                          "native_full_moe_error": e.errors(y, case["y"]),
                          "fp32_affine_full_moe_error": e.errors(y, outputs["fp32_affine_y"]),
                          "fp32_fp16_weights_full_moe_error": e.errors(y, outputs["fp32_fp16_weights_y"]),
                          "fp16_full_moe_error": e.errors(y, outputs["fp16_y"])}
    report = {"manifest": str(args.manifest), "expert_ids": needed, "expert_packages_checked": len(needed),
              "shared_package_checked": True, "plan_preferred_counts": dict(plan_counts), "packages": rows,
              "combined_with_captured_routing_and_shared_gate": combined,
              "scope": "Actual model inputs, but captured routing IDs/probabilities/gates are reused to isolate expert math. Swift routing is a separate check.",
              "hardware_evidence": "All package predictions were actually called with CPU_AND_NE. Device preferences are ComputePlan evidence, not a hardware trace."}
    e.write_json(directory / "validation.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "packages"}, indent=2))


if __name__ == "__main__":
    main()
