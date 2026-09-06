#!/usr/bin/env python3
"""Audit the fixed Qwen3.8 Swift runner's weight footprint using headers only.

Standard library only: no MLX import, device initialization, weight payload read,
or inference. The explicit selection profile mirrors the full 48-layer text
runner with MTP disabled; it is not a generic checkpoint parameter counter.
"""

import argparse
from collections import Counter
import hashlib
import json
from math import prod
from pathlib import Path
import struct
import sys


EXPECTED_SOURCE_BYTES = 77_843_121_920
EXPECTED_LOGICAL_BYTES = 9_951_107_840
WIDTHS = {"BF16": 2, "U32": 4, "F8_E4M3": 1}
SOURCE_FILES = (
    "QwenModel.swift", "QwenConfiguration.swift", "GPUWeights.swift",
    "GPUGatedDeltaNet.swift", "GPUAttention.swift", "GPUHyperConnection.swift",
    "GPUMoE.swift", "GPUPLE.swift",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def child_path(directory, filename):
    require(isinstance(filename, str) and filename == Path(filename).name
            and not filename.startswith("."), "Unsafe model file name")
    path = (directory / filename).resolve(strict=True)
    require(path.parent == directory and path.is_file(), "Model file escapes its directory")
    return path


def read_header(path, maximum=64 * 1024 * 1024):
    """Read the length prefix and JSON header, never the tensor payload."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        require(len(prefix) == 8, f"Truncated length prefix: {path.name}")
        length = struct.unpack("<Q", prefix)[0]
        require(0 < length <= maximum and 8 + length <= size,
                f"Invalid header length: {path.name}")
        raw = stream.read(length)
        require(len(raw) == length, f"Truncated header: {path.name}")
    header = json.loads(raw)
    require(isinstance(header, dict), f"Header is not an object: {path.name}")
    return header, {
        "file": path.name, "file_bytes": size, "header_bytes_including_prefix": length + 8,
        "header_with_prefix_sha256": hashlib.sha256(prefix + raw).hexdigest(),
        "payload_was_read": False, "payload_sha256_verified": False,
    }


def descriptor(header, provenance, name, shape, dtype):
    item = header.get(name)
    require(isinstance(item, dict), f"Missing header tensor: {name}")
    require(item.get("dtype") == dtype and item.get("shape") == shape,
            f"Unexpected dtype/shape: {name}; expected {dtype} {shape}")
    offsets = item.get("data_offsets")
    require(isinstance(offsets, list) and len(offsets) == 2
            and all(type(x) is int and x >= 0 for x in offsets),
            f"Invalid offsets: {name}")
    count = prod(shape) * WIDTHS[dtype]
    require(offsets[1] - offsets[0] == count
            and offsets[1] <= provenance["file_bytes"] - provenance["header_bytes_including_prefix"],
            f"Tensor length/bounds mismatch: {name}")
    return count, offsets


def validate_config(config):
    require(config.get("model_type") == "qwen4_exp", "Expected qwen4_exp checkpoint")
    text = config.get("text_config", {})
    expected = {
        "num_hidden_layers": 48, "hidden_size": 2560, "vocab_size": 248320,
        "hc_count": 4, "hc_lowrank": 320, "num_experts": 512,
        "num_experts_per_tok": 10, "moe_intermediate_size": 640,
        "shared_expert_intermediate_size": 640, "full_attention_interval": 4,
        "num_attention_heads": 24, "num_key_value_heads": 2, "head_dim": 256,
        "linear_num_key_heads": 16, "linear_num_value_heads": 48,
        "linear_key_head_dim": 128, "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4, "indexer_budget": 2048,
        "indexer_compress_ratio": 4, "indexer_n_heads": 4,
        "indexer_kv_heads": 1, "indexer_head_dim": 128,
        "ple_embed_dim": 2560, "ple_conv_kernel_size": 4,
        "heads_per_ngram": 8, "ngram_size": 3,
    }
    for key, value in expected.items():
        require(type(text.get(key)) is int and text[key] == value,
                f"Unsupported config {key}: expected {value}")
    require(text.get("ple_layer_ids") == [2], "Expected PLE at zero-based layer 1")
    require(text.get("layer_types") == [
        "full_attention" if i % 4 == 3 else "linear_attention" for i in range(48)
    ], "Unexpected layer_types")
    quantization = config.get("quantization", {})
    require(all(quantization.get(k) == v for k, v in
                {"bits": 4, "group_size": 64, "mode": "affine"}.items()),
            "Expected affine Q4, group size 64")
    require(config.get("ngram_table", {}).get("format") == "fp8_e4m3fn",
            "Expected FP8 E4M3FN n-gram table")


def selection_profile():
    """Names/shapes explicitly loaded by the current Swift module initializers."""
    selected = {}

    def add(name, module, shape, dtype="BF16"):
        require(name not in selected, f"Duplicate selection: {name}")
        selected[name] = {"module": module, "shape": shape, "dtype": dtype}

    def hc(prefix, injection=True):
        add(prefix + ".input_mix_weight_down.weight", "hc", [320, 10240])
        add(prefix + ".input_mix_weight_up.weight", "hc", [10240, 320])
        add(prefix + ".hc_norm.weight", "hc", [10240])
        if injection:
            add(prefix + ".block_inject_weight.weight", "hc", [4, 10240])

    add("language_model.model.embed_tokens.weight", "embedding_one_row", [248320, 2560])
    add("language_model.lm_head.weight", "lm_head", [248320, 2560])
    hc("language_model.model.hyper_connection_mixer", injection=False)
    for layer in range(48):
        prefix = f"language_model.model.layers.{layer}"
        hc(prefix + ".attn_hyper_connection")
        hc(prefix + ".mlp_hyper_connection")
        if layer % 4 != 3:
            for suffix, shape in {
                "in_proj_qkv.weight": [10240, 2560], "in_proj_z.weight": [6144, 2560],
                "in_proj_a.weight": [48, 2560], "in_proj_b.weight": [48, 2560],
                "out_proj.weight": [2560, 6144], "conv1d.weight": [10240, 4, 1],
                "A_log": [48], "dt_bias": [48], "norm.weight": [128],
            }.items():
                add(prefix + ".linear_attn." + suffix, "gdn", shape)
        else:
            for suffix, shape in {
                "q_proj.weight": [12288, 2560], "k_proj.weight": [512, 2560],
                "v_proj.weight": [512, 2560], "o_proj.weight": [2560, 6144],
                "q_norm.weight": [256], "k_norm.weight": [256],
                "indexer.index_qk_proj.weight": [640, 2560],
                "indexer.q_layernorm.weight": [128], "indexer.k_layernorm.weight": [128],
            }.items():
                add(prefix + ".self_attn." + suffix, "attention", shape)
        add(prefix + ".mlp.gate.weight", "routers", [512, 2560])
        add(prefix + ".mlp.shared_expert_gate.weight", "routers", [1, 2560])
        for projection, output, input_width in (("gate_proj", 640, 2560),
                                              ("up_proj", 640, 2560),
                                              ("down_proj", 2560, 640)):
            add(prefix + f".mlp.shared_expert.{projection}.weight", "shared",
                [output, input_width])
            name = prefix + f".mlp.switch_mlp.{projection}"
            add(name + ".weight", "selected_q4", [512, output, input_width // 8], "U32")
            for part in ("scales", "biases"):
                add(name + "." + part, "selected_q4", [512, output, input_width // 64])
        if layer == 1:
            for suffix, shape in {
                "key_proj.weight": [10240, 2560], "value_proj.weight": [2560, 2560],
                "norm_key.weight": [10240], "norm_query.weight": [10240],
                "norm_conv.weight": [10240], "conv1d.weight": [10240, 4, 1],
            }.items():
                add(prefix + ".ple." + suffix, "ple_dense", shape)
    return selected


def quantization_hypothesis(records):
    groups = {
        "gdn_three_largest_projections": [r for r in records if r["module"] == "gdn"
            and any(r["name"].endswith("." + n + ".weight")
                    for n in ("in_proj_qkv", "in_proj_z", "out_proj"))],
        "gdn_all_dense_projections": [r for r in records if r["module"] == "gdn"
            and len(r["shape"]) == 2],
        "hc_dense_projections": [r for r in records if r["module"] == "hc"
            and len(r["shape"]) == 2],
        "lm_head": [r for r in records if r["module"] == "lm_head"],
    }
    result = {}
    for name, rows in groups.items():
        require(all(r["dtype"] == "BF16" and r["shape"][-1] % 64 == 0 for r in rows),
                "Counterfactual requires BF16 matrices with complete input groups")
        parameters = sum(prod(r["shape"]) for r in rows)
        original = sum(r["source_bytes"] for r in rows)
        variants = {}
        for bits in (8, 4):
            codes = parameters * bits // 8
            metadata = parameters // 64 * 4  # one BF16 scale and BF16 bias per group
            variants[f"affine_q{bits}_group64"] = {
                "code_bytes": codes, "scale_and_bias_bytes": metadata,
                "total_bytes": codes + metadata, "effective_bits_per_parameter": bits + 0.5,
                "logical_bytes_saved": original - codes - metadata,
                "logical_reduction_fraction": (original - codes - metadata) / original,
            }
        result[name] = {
            "tensor_count": len(rows), "parameter_count": parameters,
            "current_bf16_bytes": original, "variants": variants,
        }
    return {
        "status": "hypothetical_not_converted_not_numerically_validated_not_benchmarked",
        "note": "Requantization changes weight values. Byte savings are not a speed prediction. Groups overlap; do not sum all groups.",
        "group_size": 64, "scale_dtype": "BF16", "bias_dtype": "BF16", "groups": result,
    }


def audit(model_directory):
    directory = model_directory.resolve(strict=True)
    config_path = child_path(directory, "config.json")
    index_path = child_path(directory, "model.safetensors.index.json")
    config = json.loads(config_path.read_bytes())
    validate_config(config)
    index = json.loads(index_path.read_bytes()).get("weight_map")
    require(isinstance(index, dict), "Missing weight_map")
    selected = selection_profile()
    require(set(selected) <= set(index), "Index is missing selected text weights")
    cache, header_sources, records, intervals = {}, {}, [], {}
    for name, spec in sorted(selected.items()):
        shard = index[name]
        require(isinstance(shard, str) and shard.endswith(".safetensors"), "Invalid shard extension")
        if shard not in cache:
            cache[shard], header_sources[shard] = read_header(child_path(directory, shard))
        count, offsets = descriptor(cache[shard], header_sources[shard], name,
                                    spec["shape"], spec["dtype"])
        logical = count
        if spec["module"] == "selected_q4":
            require(count % 512 == 0, "Expert payload not evenly partitioned")
            logical = count // 512 * 10
        elif spec["module"] == "embedding_one_row":
            logical = 2560 * 2
        records.append({"name": name, "shard": shard, **spec, "source_bytes": count,
                        "logical_bytes_per_decode_token": logical, "payload_offsets": offsets})
        intervals.setdefault(shard, []).append((*offsets, name))
    for shard, spans in intervals.items():
        end = 0
        for lower, upper, name in sorted(spans):
            require(lower >= end, f"Overlapping selected tensors in {shard}: {name}")
            end = upper
    source_total = sum(r["source_bytes"] for r in records)
    logical_total = sum(r["logical_bytes_per_decode_token"] for r in records)
    require(source_total == EXPECTED_SOURCE_BYTES, "Source byte total differs from the audited load profile")
    require(logical_total == EXPECTED_LOGICAL_BYTES, "Logical byte total differs from the audited profile")
    require(Counter(r["dtype"] for r in records) == {"BF16": 1355, "U32": 144},
            "Unexpected selected dtype counts")
    modules = {}
    for row in records:
        module = modules.setdefault(row["module"], {"tensor_count": 0, "source_bytes": 0,
                                                    "logical_bytes_per_decode_token": 0})
        module["tensor_count"] += 1
        module["source_bytes"] += row["source_bytes"]
        module["logical_bytes_per_decode_token"] += row["logical_bytes_per_decode_token"]
    for module in modules.values():
        module["logical_gb_decimal_per_decode_token"] = module["logical_bytes_per_decode_token"] / 1e9
        module["logical_fraction"] = module["logical_bytes_per_decode_token"] / logical_total
    top = sorted(modules, key=lambda key: modules[key]["logical_bytes_per_decode_token"], reverse=True)
    norms = [r for r in records if ".indexer." in r["name"]
             and r["name"].endswith("layernorm.weight")]
    short_unused = sum(r["source_bytes"] for r in norms)
    gdn_packed = sum(r["source_bytes"] for r in records if r["module"] == "gdn"
                     and ".in_proj_" in r["name"])
    hc_packed = sum(r["source_bytes"] for r in records if r["module"] == "hc"
                    and ".layers." in r["name"]
                    and any(part in r["name"] for part in ("input_mix_weight_down", "block_inject_weight")))
    hc_scaled = sum(r["source_bytes"] for r in records if r["module"] == "hc"
                    and any(part in r["name"] for part in ("input_mix_weight_down", "block_inject_weight")))
    table_path = child_path(directory, config["ngram_table"]["file"])
    table_header, table_provenance = read_header(table_path, maximum=1024 * 1024)
    table_bytes, table_offsets = descriptor(table_header, table_provenance, "weight", [320001536, 160], "F8_E4M3")
    require(table_offsets == [0, table_bytes]
            and table_provenance["file_bytes"] == table_provenance["header_bytes_including_prefix"] + table_bytes,
            "Unexpected n-gram table file length")
    runner_root = Path(__file__).resolve().parents[1]
    source_fingerprints = [fingerprint(runner_root / "Sources" / "ANERunnerGPU" / name)
                           for name in SOURCE_FILES]
    return {
        "schema_version": 1, "model_directory": str(directory),
        "method": "CPU-only safetensors headers plus explicit current Swift full-text load profile",
        "selection_scope": {"layers": 48, "gdn_layers": 36, "attention_layers": 12,
                            "experts_per_layer": 512, "experts_per_token": 10,
                            "mtp_enabled": False, "mtp_weights_selected": False,
                            "vision_weights_selected": False,
                            "profile_is_automatically_extracted_from_swift": False,
                            "note": "Source hashes are recorded for drift review; update this explicit profile if Swift loading changes."},
        "provenance": {"script": fingerprint(Path(__file__).resolve()),
                       "configuration": fingerprint(config_path), "index": fingerprint(index_path),
                       "runner_sources": source_fingerprints,
                       "safetensors_headers": [header_sources[k] for k in sorted(header_sources)]},
        "totals": {"selected_tensor_count": len(records), "loaded_source_weight_bytes": source_total,
                   "static_logical_decode_weight_bytes_per_token": logical_total,
                   "non_routed_bf16_logical_bytes_per_token": logical_total - modules["selected_q4"]["logical_bytes_per_decode_token"],
                   "selected_dtype_counts": dict(Counter(r["dtype"] for r in records)),
                   "top_three_modules": top[:3]},
        "modules": {key: modules[key] for key in top},
        "routed_quantization": {"bits": 4, "group_size": 64, "codes_storage_dtype": "U32",
                                "scale_dtype": "BF16", "bias_dtype": "BF16",
                                "effective_bits_per_parameter": 4.5,
                                "selected_matrix_parameters_per_token": 48 * 10 * 3 * 2560 * 640,
                                "selected_code_bytes_per_token": sum(r["logical_bytes_per_decode_token"] for r in records if r["module"] == "selected_q4" and r["dtype"] == "U32"),
                                "selected_scale_and_bias_bytes_per_token": sum(r["logical_bytes_per_decode_token"] for r in records if r["module"] == "selected_q4" and r["dtype"] == "BF16"),
                                "formula": "48 * 10 * 3 * 2560 * 640 * (0.5 + 4/64)"},
        "conditional_unused_weights": {
            "context_total_at_most": 2051, "short_context_unused_indexer_norm_bytes": short_unused,
            "short_context_adjusted_logical_bytes": logical_total - short_unused,
            "tensor_names": [r["name"] for r in norms],
            "long_context_note": "Indexer key norm is unused on steps without a new pooled block (3072 B across 12 layers); static ledger does not track this.",
            "possible_short_context_query_projection_elision_bytes": 12 * 512 * 2560 * 2,
            "query_note": "The current matmul computes all 640 rows, so these query weights ARE currently counted and used in compute. Keeping only 128 key rows below the threshold needs a numerical gate and has not been implemented here.",
        },
        "additional_resident_layout_buffers": {
            "excluded_from_logical_weight_ledger": True,
            "gdn_packed_if_prepared_bytes": gdn_packed,
            "hc_packed_if_prepared_bytes": hc_packed,
            "all_packed_if_prepared_bytes": gdn_packed + hc_packed,
            "hc_prescaled_bf16_bytes": hc_scaled,
            "source_plus_prescaled_plus_all_packed_bytes": source_total + hc_scaled + gdn_packed + hc_packed,
            "note": "Analytical live weight-buffer sizes, not measured allocation. Original HC source weights coexist with prescaled BF16 tensors. Prepared packed projections coexist with originals for A/B; a forward selects one projection path, not both. Views do not add another full matrix payload. Reusing storage can reduce residency without equal per-token read savings. Allocator cache, LUTs, runtime scalars, state and activations are excluded.",
        },
        "ngram_table_separate_scope": {
            "header": table_provenance, "row_count": 320001536, "dimension": 160,
            "format": "FP8 E4M3FN", "payload_bytes": table_bytes,
            "requested_rows_per_token": 16, "logical_requested_payload_bytes_per_token": 2560,
            "scale_from_configuration": config["ngram_table"]["scale"],
            "excluded_from_device_weight_ledger": True,
            "note": "SSD demand-read logical payload only; actual disk traffic and page-cache behavior are not measured. No table rows read by this audit.",
        },
        "requantization_hypotheses": quantization_hypothesis(records),
        "measurement_limits": {
            "physical_dram_bytes": None, "physical_dram_bandwidth_gbps": None,
            "physical_disk_bytes": None, "inference_seconds": None,
            "note": "Each selected expert is counted once, each dense source-equivalent tensor once, and one token embedding row. This is not measured traffic: cache reuse/refetches, state, activations and dispatch effects are excluded. No inference or numerical validation was performed.",
        },
        "excluded_index_tensor_count": len(set(index) - set(selected)),
        "selected_tensors": records,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True, help="Existing Qwen3.8 MLX SSD model directory")
    parser.add_argument("--output", type=Path, required=True, help="New JSON file; existing output is never overwritten")
    args = parser.parse_args(argv)
    try:
        require(not args.output.exists(), f"Output already exists: {args.output}")
        report = audit(args.model_dir)
        serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"audit failed: {error}\n")
    print(json.dumps({"output": str(args.output.resolve()), **report["totals"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
