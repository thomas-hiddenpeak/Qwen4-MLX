#!/usr/bin/env python3
"""Validate unique expert artifacts and assemble a portable compressed MoE bank.

Checks the current per-expert artifacts against their recorded hashes rather
than trusting potentially overlapping worker fragments. Shared/router copies
must match the original FP16 bank. Does not claim runtime numerical validation.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def files(path):
    return {str(p.relative_to(path)): digest(p) for p in sorted(path.rglob("*")) if p.is_file()}


def size(path):
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--experts-file", type=Path, required=True)
    parser.add_argument("--group-size", type=int, choices=(1, 4, 16), default=4)
    args = parser.parse_args()
    source = args.source_manifest.resolve().parent
    bank = args.bank.resolve()
    manifest = json.loads(args.source_manifest.read_text())
    ids = sorted(set(map(int, args.experts_file.read_text().replace(",", " ").split())))
    require(ids and all(0 <= i < manifest["expert_count"] for i in ids), "Invalid expert IDs")
    records = []
    experts = {}
    for expert in ids:
        name = f"expert_{expert:04d}.mlpackage"
        record_path = bank / "reports" / f"expert_{expert:04d}.json"
        record = json.loads(record_path.read_text())
        require(record["expert"] == expert and record["group_size"] == args.group_size, f"Wrong metadata: {name}")
        require(Path(record["source_package"]).resolve() == source / name, f"Wrong source: {name}")
        require(Path(record["output_package"]).resolve() == bank / name, f"Wrong output: {name}")
        require(record["runtime_plan_counts"] == {"MLNeuralEngineComputeDevice": 12}, f"Unexpected plan: {name}")
        require(files(source / name) == record["source_file_sha256"], f"Source hash mismatch: {name}")
        require(files(bank / name) == record["output_file_sha256"], f"Output hash mismatch: {name}")
        experts[str(expert)] = name
        records.append({"expert": expert, "record_sha256": digest(record_path),
                        "source_bytes": size(source / name), "output_bytes": size(bank / name)})
    shared = manifest["shared_expert"]
    require(files(bank / shared) == files(source / shared) and size(bank / shared) > 0, "Shared copy differs")
    copies = {shared: size(bank / shared)}
    for key in ("weights_file", "shared_gate_file"):
        name = manifest["routing"][key]
        require(digest(bank / name) == digest(source / name), f"Routing copy differs: {name}")
        copies[name] = (bank / name).stat().st_size
    manifest.update(experts=experts, weight_mode=f"lut8_output_group{args.group_size}_shared_fp16",
                    weight_semantics=f"Routed affine Q4 weights decoded to FP16, then approximately re-encoded with LUT8 per {args.group_size} output channels. Shared FP16 and FP32 routing copies unchanged. This is not lossless source Q4, nor a claim of native 8-bit ANE arithmetic.",
                    compressed_bank_validation="assembly-validation.json")
    routed_before = sum(r["source_bytes"] for r in records)
    routed_after = sum(r["output_bytes"] for r in records)
    overhead = sum(copies.values())
    report = {"passed": True, "source_manifest": str(args.source_manifest.resolve()),
              "source_manifest_sha256": digest(args.source_manifest), "expert_ids": ids,
              "expert_records": records, "unchanged_copies_bytes": copies,
              "routed_source_bytes": routed_before, "routed_compressed_bytes": routed_after,
              "source_model_assets_bytes": routed_before + overhead,
              "compressed_model_assets_bytes": routed_after + overhead,
              "model_asset_reduction_fraction": 1 - (routed_after + overhead) / (routed_before + overhead),
              "runtime_ops_ane_preferred": 12 * len(ids),
              "limits": "Package/model asset bytes exclude reports, compiled caches and runtime memory. Current artifacts hash-checked; duplicate fragment references are ignored. Numerical and hardware execution gates are separate."}
    for name, value in (("assembly-validation.json", report), ("manifest.json", manifest)):
        temporary = bank / f".{name}.tmp"
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        temporary.replace(bank / name)
    print(json.dumps({k: report[k] for k in ("passed", "source_model_assets_bytes", "compressed_model_assets_bytes", "model_asset_reduction_fraction")}, indent=2))


if __name__ == "__main__":
    main()
