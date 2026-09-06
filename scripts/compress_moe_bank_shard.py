#!/usr/bin/env python3
"""Compress disjoint expert shards with output-channel LUT8 for bank assembly.

Run with the repository .venv/bin/python (coremltools 9.0). No environment
changes are needed. CPU clustering uses Core ML Tools' bundled weighted 1-D
KMeans via the public CUSTOM palettization callback; no private ANE APIs.
Workers must receive disjoint expert IDs. Root assembles the final manifest.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import gc
import json
from pathlib import Path
import time

import coremltools as ct
import export_moe as e
import probe_moe_compression_v2 as p


@contextmanager
def claim_expert(output_dir, filename):
    """Atomically exclude another exporter for this package, before any checks."""
    claims_dir = output_dir / ".claims"
    claims_dir.mkdir(exist_ok=True)
    claim = claims_dir / Path(filename).stem
    try:
        claim.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(
            f"Refusing duplicate export of {filename}: claim already exists at {claim}; "
            "another worker may still own it. No package or record was removed."
        ) from error
    try:
        yield
    finally:
        # Remove only this process's empty claim; never touch model artifacts.
        claim.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=p.BASE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experts", default="", help="Disjoint comma-separated expert IDs")
    parser.add_argument("--shared", action="store_true", help="Only one worker may own shared")
    parser.add_argument("--group-size", type=int, choices=(1, 4, 16), default=4)
    parser.add_argument("--worker", required=True, help="Unique name for this worker's fragment report")
    args = parser.parse_args()
    expert_ids = sorted(set(map(int, args.experts.split(",")))) if args.experts else []
    if any(i < 0 or i >= 512 for i in expert_ids) or not (expert_ids or args.shared):
        parser.error("Provide valid expert IDs and/or --shared")
    if not args.worker.replace("_", "").replace("-", "").isalnum():
        parser.error("Worker names must be alphanumeric, with optional _ or -")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_dir = args.output_dir / "reports"
    records_dir.mkdir(exist_ok=True)
    fragment = {"schema_version": 1, "worker": args.worker, "source_directory": str(args.source_dir.resolve()),
                "output_directory": str(args.output_dir.resolve()), "weight_mode": f"lut8_output_group{args.group_size}",
                "graph_layout": "linear", "token_capacity": 32, "experts": {}, "records": [],
                "clustering": "coremltools 9.0 bundled weighted kmeans1d CPU helper, public CUSTOM LUT8 compression API",
                "hardware_evidence": "ComputePlan preferences only. This batch exporter does not run predictions or trace hardware."}
    for identifier in [*expert_ids, *(["shared"] if args.shared else [])]:
        filename = "shared.mlpackage" if identifier == "shared" else f"expert_{identifier:04d}.mlpackage"
        source_path, output_path = args.source_dir / filename, args.output_dir / filename
        record_path = records_dir / ("shared.json" if identifier == "shared" else f"expert_{identifier:04d}.json")
        with claim_expert(args.output_dir, filename):
            if output_path.exists() or record_path.exists():
                raise FileExistsError(f"Refusing concurrent or ambiguous overwrite: {output_path}; use disjoint IDs and a fresh output directory")
            start = time.perf_counter()
            model = ct.models.MLModel(str(source_path), compute_units=ct.ComputeUnit.CPU_AND_NE, skip_model_load=True)
            metadata = p.matrix_metadata(model)
            transform, config, description = p.configuration(f"lut8_group{args.group_size}", [v["name"] for v in metadata.values()])
            compress_start = time.perf_counter()
            compressed = transform(model, config=config)
            compression_seconds = time.perf_counter() - compress_start
            compressed.save(str(output_path))
            load_start = time.perf_counter()
            loaded = ct.models.MLModel(str(output_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
            initial_load_seconds = time.perf_counter() - load_start
            plan = e.compute_plan(loaded)
            runtime_counts = dict(Counter(op["preferred"] for op in plan if "constexpr" not in op["op"]))
            decoded = ct.optimize.coreml.decompress_weights(compressed)
            reconstructed = p.matrix_metadata(decoded)
            record = {"expert": identifier, "source_package": str(source_path.resolve()),
                      "output_package": str(output_path.resolve()), "group_size": args.group_size,
                      "configuration": description, "source_package_bytes": e.package_bytes(source_path),
                      "output_package_bytes": e.package_bytes(output_path),
                      "compression_seconds": compression_seconds,
                      "initial_package_load_including_compile_seconds": initial_load_seconds,
                      "total_export_and_weight_validation_seconds": time.perf_counter() - start,
                      "weight_errors": {name: e.errors(reconstructed[name]["value"], metadata[name]["value"]) for name in metadata},
                      "compute_plan": plan, "runtime_plan_counts": runtime_counts,
                      "source_file_sha256": {str(path.relative_to(source_path)): e.sha256_file(path) for path in source_path.rglob("*") if path.is_file()},
                      "output_file_sha256": {str(path.relative_to(output_path)): e.sha256_file(path) for path in output_path.rglob("*") if path.is_file()},
                      "prediction_validated": False}
            e.write_json(record_path, record)
            if runtime_counts != {"MLNeuralEngineComputeDevice": 12}:
                raise ValueError(f"Unexpected runtime placement for {filename}: {runtime_counts}; package is not added to fragment")
            if identifier == "shared":
                fragment["shared_expert"] = filename
            else:
                fragment["experts"][str(identifier)] = filename
            fragment["records"].append(str(record_path.relative_to(args.output_dir)))
            e.write_json(args.output_dir / f"fragment-{args.worker}.json", fragment)
            print(json.dumps({"worker": args.worker, "expert": identifier, "bytes": record["output_package_bytes"],
                              "seconds": record["total_export_and_weight_validation_seconds"], "plan": runtime_counts}), flush=True)
            del model, compressed, loaded, decoded, metadata, reconstructed
            gc.collect()


if __name__ == "__main__":
    main()
