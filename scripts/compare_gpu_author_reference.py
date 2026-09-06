#!/usr/bin/env python3
"""Compare saved Swift generation and a bounded author-server capture, offline.

Exact API-exposed IDs and complete visible text are checked separately from an
unexposed stop ID. Native decode rates retain each implementation's denominator
and timing boundary. This script never loads a model or issues HTTP requests.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def first_difference(actual, expected):
    for index, (a, b) in enumerate(zip(actual, expected)):
        if a != b:
            return {"position": index, "swift_id": a, "author_id": b}
    if len(actual) < len(expected):
        return {"position": len(actual), "swift_id": None, "author_id": expected[len(actual)]}
    return None


def summarize(values):
    return {"samples": values, "median": statistics.median(values) if values else None,
            "minimum": min(values) if values else None, "maximum": max(values) if values else None}


def compare(swift, reference, metadata, request, benchmarks, skip_swift_trials):
    trials = swift["trials"]
    if not 0 <= skip_swift_trials < len(trials):
        raise ValueError("--skip-swift-trials must leave at least one warm trial")
    ids = reference["generated_token_ids"]
    prompt = reference["prompt_token_ids"]
    results = []
    for index, trial in enumerate(trials):
        output = trial["generated_token_ids"]
        duration = trial["decode_step_seconds"]
        if not all(math.isfinite(value) and value > 0 for value in duration):
            raise ValueError("Swift decode durations must be finite and positive")
        if len(duration) != trial["decode_steps"]:
            raise ValueError("Swift decode step count differs from duration samples")
        derived_rate = len(duration) / sum(duration) if duration else None
        recorded_rate = trial.get("decode_tokens_per_second")
        if derived_rate is not None and not math.isclose(derived_rate, recorded_rate, rel_tol=1e-10):
            raise ValueError("Swift recorded decode rate differs from the measured-step denominator")
        divergence = first_difference(output, ids)
        # Never append an ID to the author sequence. An extra Swift suffix is
        # reported separately, including when Swift labels it EOS.
        suffix = output[len(ids):] if divergence is None else None
        results.append({
            "trial_index": index, "repetition": trial.get("repetition"),
            "included_in_warm_statistics": index >= skip_swift_trials,
            "prompt_token_ids_exact": trial["prompt_tokens"] == prompt,
            "all_author_exposed_output_ids_exact": divergence is None,
            "first_exposed_id_difference": divergence,
            "visible_text_exact": trial["text"] == reference["generated_text"],
            "swift_generated_token_count": len(output), "author_exposed_token_count": len(ids),
            "swift_suffix_ids_not_observed_from_author": suffix,
            "complete_generated_id_sequences_directly_observed_equal": output == ids,
            "swift_finish_reason": trial["finish_reason"], "author_finish_reason": reference["finish_reason"],
            "prefill_accumulation": trial.get("prefill_accumulation"),
            "source_reference_accumulation": trial.get("prefill_accumulation") == "reference",
            "mtp_disabled": trial.get("mtp_enabled") is False,
            "sampling_greedy": trial.get("sampling") == "greedy",
            "final_prompt_token_held_back": trial.get("final_prompt_token_held_back"),
            "swift_prefill_chunk_capacity": trial.get("prefill_chunk"),
            "short_prompt_fits_one_prefix_chunk": len(prompt) - 1 <= trial.get("prefill_chunk", 0),
            "decode_steps": len(duration), "decode_seconds": sum(duration),
            "decode_tokens_per_second": derived_rate,
            "request_seconds_excluding_load": trial["request_seconds_excluding_load"],
            "time_to_first_token_seconds_excluding_load": trial["time_to_first_token_seconds_excluding_load"],
        })
    swift_hashes = swift.get("provenance", {}).get("model_metadata_sha256", {})
    author_hashes = metadata.get("source_sha256", {})
    hash_checks = {name: bool(swift_hashes.get(name)) and swift_hashes.get(name) == author_hashes.get(name)
                   for name in ["config.json", "tokenizer.json", "model.safetensors.index.json"]}
    exact_request = all(request.get(key) == expected for key, expected in {
        "temperature": 0, "top_p": 1, "top_k": 1, "repeat_penalty": 1,
        "presence_penalty": 0, "enable_mtp": False, "enable_drafter": False,
        "enable_pld": False, "stream": False, "logprobs": 1,
    }.items())
    warm = results[skip_swift_trials:]
    author_rates = []
    for run in benchmarks:
        timing = run.get("server_log", {})
        parsed = timing.get("parsed_timing")
        if timing.get("unambiguous_request_window") and parsed is not None:
            rate = parsed["decode_tokens_per_second"]
            if not math.isfinite(rate) or rate <= 0:
                raise ValueError("Author decode rate must be finite and positive")
            author_rates.append(rate)
    all_visible = all(row["prompt_token_ids_exact"] and row["all_author_exposed_output_ids_exact"]
                      and row["visible_text_exact"] for row in results)
    source_mode = all(row["source_reference_accumulation"] for row in results)
    modes_off = (swift.get("mtp_enabled") is False and swift.get("mtp_weights_loaded") is False
                 and all(row["mtp_disabled"] for row in results)
                 and all(metadata.get(key) is False for key in ["mtp_enabled", "drafter_enabled", "pld_enabled"]))
    profiler_disabled = swift.get("profiler", {}).get("mode") == "disabled"
    gates = {
        "author_capture_completed": metadata.get("completed") is True,
        "prompt_visible_text_and_all_author_exposed_ids_exact": all_visible,
        "source_reference_prefill_accumulation": source_mode,
        "model_metadata_hashes_match": all(hash_checks.values()),
        "speculation_disabled": modes_off,
        "greedy_request_parameters_match": exact_request and all(row["sampling_greedy"] for row in results),
        "swift_profiler_disabled": profiler_disabled,
        "all_benchmark_texts_match_reference": bool(benchmarks) and all(run["text_matches_reference"] for run in benchmarks),
        "all_author_benchmarks_have_unambiguous_server_timing": bool(benchmarks) and len(author_rates) == len(benchmarks),
    }
    swift_rates = [row["decode_tokens_per_second"] for row in warm]
    ratio = (statistics.median(swift_rates) / statistics.median(author_rates)
             if all(gates.values()) and all(rate is not None for rate in swift_rates) else None)
    return {
        "schema_version": 1, "checks": gates, "model_metadata_hash_checks": hash_checks,
        "visible_output_and_source_mode_gate_passes": all_visible and source_mode and modes_off and all(hash_checks.values()),
        "trials": results, "author_token_id_coverage": reference.get("token_id_coverage"),
        "unobserved_stop_id_note": "A Swift terminal EOS is not compared against an inferred author EOS. Exact author-exposed IDs and visible text do not independently prove the author's unexposed stop-token ID.",
        "warm_timing": {
            "swift_skipped_initial_trials": skip_swift_trials,
            "author_warmup": "One same-prompt logprob reference request, excluded; benchmark requests omit logprobs.",
            "swift_decode_tokens_per_second": summarize(swift_rates),
            "author_native_decode_tokens_per_second": summarize(author_rates),
            "swift_request_seconds_excluding_load": summarize([row["request_seconds_excluding_load"] for row in warm]),
            "author_http_request_wall_seconds": summarize([run["request_wall_seconds"] for run in benchmarks]),
            "swift_ttft_seconds_excluding_load": summarize([row["time_to_first_token_seconds_excluding_load"] for row in warm]),
            "author_ttft_seconds": None,
            "native_reported_rate_ratio_swift_over_author": ratio,
            "ratio_scope": "Operational comparison of native reported rates, not an exactly matched kernel timer or hardware-speedup measurement. Omitted unless correctness, modes, metadata and timing attribution checks pass.",
            "source_rate_log_resolution_tokens_per_second": 0.1,
            "swift_numerator_and_timer": "Number of measured post-first-token forward/sample steps divided by their summed host wall time; includes the step that samples EOS. First token is included in TTFT instead.",
            "author_numerator_and_timer": "Emitted completion_tokens divided by scheduler decode tick wall time. EOS is checked before append/advance, so it is not emitted/counted. The lazy pipeline and final stop tick change the timer boundary relative to synchronous Swift steps.",
            "denominator_caution": "For the current 26-content-token stop case, Swift has 27 sampled IDs including EOS and 26 timed decode steps; author is expected to emit 26 IDs. Equal numerators do not make pipeline/timer boundaries identical. No ad hoc 26/27 scaling is applied.",
            "request_scope_caution": "Swift request wall starts after model loading/tokenization; author HTTP wall also includes transport, request parsing/tokenization and scheduling. Captured HTTP logs do not expose an exact first-token timestamp.",
        },
        "physical_dram_bytes": None, "physical_dram_bandwidth_gbps": None,
        "source_boundaries": {
            "swift": "Sources/ANERunnerCLI/GPUGeneration.swift: prefill includes held-back final prompt and first sample; decode count / sum(decode).",
            "author": ["src/generate.zig:2471 skip_lazy_preforward", "src/generate.zig:7299 Generator.next",
                       "src/generate.zig:7561 checkStop before EOS append", "src/scheduler.zig:3999 decode tick accounting",
                       "src/server.zig:9158 formatPerfBracket", "src/generate.zig:591 tokensPerSec"],
        },
        "limitations": ["One short deterministic prompt is a bounded equivalence test, not a broad quality evaluation.",
                        "Model metadata hashes agree only on metadata; this comparison does not rehash checkpoint payloads.",
                        "Swift max-token budget is absent from this report schema; author request budget is recorded. Complete matching text/stop is checked separately.",
                        "Sequential runs share hardware but are not simultaneous paired counter measurements. No physical bandwidth is inferred."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--swift-report", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--skip-swift-trials", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new report path")
    swift = json.loads(args.swift_report.read_text())
    metadata_path = args.capture_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    files = {"reference": "reference.json", "request": "completion-request.json"}
    if (args.capture_dir / "benchmark-runs.json").exists():
        files["benchmarks"] = "benchmark-runs.json"
    values, evidence = {}, []
    for key, name in files.items():
        path = args.capture_dir / name
        digest = sha(path)
        if metadata.get("fixture_files_sha256", {}).get(name) != digest:
            raise ValueError(f"Capture file no longer matches its provenance hash: {name}")
        values[key] = json.loads(path.read_text())
        evidence.append({"path": str(path.resolve()), "sha256": digest})
    report = compare(swift, values["reference"], metadata, values["request"], values.get("benchmarks", []), args.skip_swift_trials)
    report["source_files"] = [{"path": str(args.swift_report.resolve()), "sha256": sha(args.swift_report)},
                              {"path": str(metadata_path.resolve()), "sha256": sha(metadata_path)}, *evidence]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as destination:
        json.dump(report, destination, ensure_ascii=False, indent=2, allow_nan=False)
        destination.write("\n")
    print(json.dumps({"output": str(args.output), "checks": report["checks"],
                      "warm_timing": report["warm_timing"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
