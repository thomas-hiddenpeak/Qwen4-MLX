#!/usr/bin/env python3
"""Bounded reference/benchmark capture from an already running local author server.

Never loads a model, starts/stops a service, retries a generation, or overwrites
an output directory. One logprob reference warms the model; at most three
subsequent requests measure the same continuation without logprobs. Only
API-exposed per-token boundaries are inverted to IDs; an omitted EOS is never
guessed. Uses only the Python standard library.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = ROOT / "experiments/qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream"
DEFAULT_RUNTIME = ROOT / "experiments/qwen38-ssd/runtime/mlx-serve"
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
PERF = re.compile(
    r"<-\s*(\d+)\+(\d+) tokens \((\d+)ms\) "
    r"\[prefill: ([0-9.]+) tok/s(?: \((\d+) cached / (\d+) total\))?, "
    r"decode: ([0-9.]+) tok/s\] \[([^\]]+)\]"
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for data in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def save(out, name, value):
    (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def vocabulary_decoder(tokenizer):
    require(tokenizer["decoder"]["type"] == "ByteLevel", "Only exact ByteLevel vocabulary inversion is supported")
    allowed = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    mapping = {byte: chr(byte) for byte in allowed}
    extra = 0
    for byte in range(256):
        if byte not in mapping:
            mapping[byte] = chr(256 + extra)
            extra += 1
    inverse = {value: key for key, value in mapping.items()}
    by_id = {identifier: text for text, identifier in tokenizer["model"]["vocab"].items()}
    by_id.update({entry["id"]: entry["content"] for entry in tokenizer.get("added_tokens", [])})
    raw_by_id, exact_ids_by_text = {}, defaultdict(list)
    for identifier, text in by_id.items():
        raw = b"".join(bytes([inverse[char]]) if char in inverse else char.encode() for char in text)
        raw_by_id[identifier] = raw
        try:
            decoded = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            continue
        exact_ids_by_text[decoded].append(identifier)
    return raw_by_id, exact_ids_by_text


def recover_ids(response, raw_by_id, ids_by_text, prompt_ids, max_tokens):
    require(len(response["choices"]) == 1, "Expected one completion choice")
    choice, usage = response["choices"][0], response["usage"]
    logprobs = choice.get("logprobs")
    require(isinstance(logprobs, dict), "Exact output ID recovery needs per-token logprob boundaries")
    texts = logprobs["tokens"]
    require(0 < len(texts) <= max_tokens, "Unexpected number of exposed logprob tokens")
    require(len(logprobs["text_offset"]) == len(texts) == len(logprobs["token_logprobs"]), "Logprob array lengths disagree")
    require(usage["prompt_tokens"] == len(prompt_ids), "API prompt count disagrees with /tokenize")
    require(type(usage["completion_tokens"]) is int and len(texts) <= usage["completion_tokens"] <= max_tokens,
            "Completion usage must include every exposed token and remain within the requested limit")
    require(usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"], "API usage totals disagree")
    generated_ids, evidence, offset = [], [], 0
    for index, text in enumerate(texts):
        require("\ufffd" not in text, "Lossy UTF-8 replacement prevents exact token ID recovery")
        candidates = ids_by_text.get(text, [])
        require(len(candidates) == 1, f"Token {index} has {len(candidates)} lossless inverse candidates")
        token_id = candidates[0]
        require(logprobs["text_offset"][index] == offset, f"Token {index} has an unexpected UTF-8 byte offset")
        raw = raw_by_id[token_id]
        require(raw == text.encode("utf-8"), f"Token {index} does not roundtrip exactly")
        evidence.append({"position": index, "id": token_id, "text": text, "utf8_hex": raw.hex(),
                         "text_offset": offset, "unique_inverse_candidate_count": 1,
                         "logprob": logprobs["token_logprobs"][index]})
        generated_ids.append(token_id)
        offset += len(raw)
    decoded = b"".join(raw_by_id[i] for i in generated_ids).decode("utf-8")
    require(decoded == choice["text"], "Per-token text differs from the full completion")
    coverage = {
        "api_exposed_token_count": len(generated_ids),
        "api_usage_completion_tokens": usage["completion_tokens"],
        "usage_tokens_without_exposed_ids": usage["completion_tokens"] - len(generated_ids),
        "all_usage_tokens_have_exposed_ids": len(generated_ids) == usage["completion_tokens"],
        "inferred_or_appended_token_ids": [],
        "scope": "Exactly the API-exposed per-token logprob entries. No EOS or other unexposed ID is inferred, even when finish_reason is stop.",
    }
    return generated_ids, evidence, decoded, coverage


def verify_process(pid, model, runtime, port):
    command = subprocess.check_output(["ps", "-ww", "-p", str(pid), "-o", "command="], text=True).strip()
    started = subprocess.check_output(["ps", "-p", str(pid), "-o", "lstart="], text=True).strip()
    words = shlex.split(command)
    require(words and Path(words[0]).resolve() == (runtime / "zig-out/bin/mlx-serve").resolve(), "PID is not the requested author binary")
    for flag in ["--serve", "--no-mtp", "--no-drafter", "--no-pld"]:
        require(flag in words, f"Live server process lacks {flag}")
    for flag, expected in [("--model", str(model)), ("--port", str(port))]:
        require(flag in words and words.index(flag) + 1 < len(words), f"Live process lacks {flag} value")
        actual = words[words.index(flag) + 1]
        require(str(Path(actual).resolve()) == expected if flag == "--model" else actual == expected,
                f"Live process {flag} differs from requested endpoint/model")
    return {"pid": pid, "started": started, "command": command}


def verify_loaded(status, model_name=None):
    loaded = [entry for entry in status["data"] if entry.get("loaded") and entry.get("state") == "ready"]
    require(len(loaded) == 1, "Expected exactly one ready model")
    model = loaded[0]
    require(model["meta"].get("mtp_loaded") is False, "Live model has MTP loaded or does not expose the flag")
    require(model["meta"].get("drafter_loaded") is False, "Live model has drafter loaded or does not expose the flag")
    if "pld_enabled" in model["meta"]:
        require(model["meta"]["pld_enabled"] is False, "Live model reports PLD enabled")
    if model_name is not None:
        require(model["id"] == model_name, "Loaded model changed during capture")
    return model


def log_cursor(path):
    if path is None:
        return None
    stat = path.stat()
    return {"device": stat.st_dev, "inode": stat.st_ino, "offset": stat.st_size}


def capture_log(path, before, out, name, usage):
    if path is None:
        return {"available": False, "reason": "No --server-log provided"}
    after = log_cursor(path)
    require((before["device"], before["inode"]) == (after["device"], after["inode"])
            and after["offset"] >= before["offset"], "Server log rotated or truncated during request")
    with path.open("rb") as handle:
        handle.seek(before["offset"])
        raw = handle.read(after["offset"] - before["offset"])
    (out / f"{name}-server-log.bin").write_bytes(raw)
    decoded = raw.decode("utf-8", errors="replace")
    lines = decoded.splitlines()
    timing_lines = [line for line in lines if PERF.search(ANSI.sub("", line))]
    (out / f"{name}-server-timing-lines.txt").write_text("\n".join(timing_lines) + ("\n" if timing_lines else ""))
    generation_requests = [line for line in lines if re.search(r"POST /v1/(?:chat/completions|completions|responses)", ANSI.sub("", line))]
    valid = len(generation_requests) == 1 and len(timing_lines) == 1
    parsed = None
    if valid:
        match = PERF.search(ANSI.sub("", timing_lines[0]))
        parsed = {"prompt_tokens": int(match[1]), "completion_tokens": int(match[2]),
                  "elapsed_ms": int(match[3]), "prefill_tokens_per_second": float(match[4]),
                  "cached_prompt_tokens": int(match[5]) if match[5] is not None else 0,
                  "decode_tokens_per_second": float(match[7]), "finish_reason": match[8]}
        valid = parsed["prompt_tokens"] == usage["prompt_tokens"] and parsed["completion_tokens"] == usage["completion_tokens"]
    return {"available": True, "path": str(path), "start": before, "end": after,
            "raw_segment_sha256": hashlib.sha256(raw).hexdigest(), "request_log_lines": generation_requests,
            "timing_log_lines": timing_lines, "unambiguous_request_window": valid,
            "parsed_timing": parsed if valid else None,
            "scope": "Exact bytes appended during this HTTP request; attribution requires exactly one generation request and one matching timing line. No physical memory counters."}


class Capture:
    def __init__(self, args):
        self.args, self.out = args, args.output
        self.requests, self.generations = [], 0

    def request(self, endpoint, name, body=None):
        generation = endpoint == "/v1/completions"
        if generation:
            require(self.generations < 1 + self.args.benchmark_runs <= 4, "Generation request budget exhausted")
            self.generations += 1
        if body is not None:
            save(self.out, name + "-request.json", body)
        record = {"name": name, "endpoint": endpoint, "generation": generation,
                  "started_at_utc": datetime.now(timezone.utc).isoformat(), "completed": False}
        self.requests.append(record)
        save(self.out, "request-ledger.json", self.requests)
        cursor = log_cursor(self.args.server_log) if generation else None
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode()
        req = urllib.request.Request(self.args.base_url + endpoint, data=data, headers={"Content-Type": "application/json"})
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                raw, status, headers = response.read(), response.status, dict(response.headers)
        except urllib.error.HTTPError as error:
            (self.out / (name + "-response.json")).write_bytes(error.read())
            record.update(status=error.code, error=str(error))
            save(self.out, "request-ledger.json", self.requests)
            raise
        wall = time.perf_counter() - started
        (self.out / (name + "-response.json")).write_bytes(raw)
        record.update(status=status, completed=True, wall_seconds=wall, body_bytes=len(raw))
        save(self.out, "request-ledger.json", self.requests)
        save(self.out, name + "-http.json", {**record, "url": req.full_url, "headers": headers})
        require(status == 200, f"Unexpected HTTP status {status}")
        value = json.loads(raw)
        if generation:
            record["server_log"] = capture_log(self.args.server_log, cursor, self.out, name, value["usage"])
            save(self.out, "request-ledger.json", self.requests)
        return value, record


def run(args):
    # Reserve one new directory atomically; even empty existing directories are
    # refused so an uncertain prior request can never be accidentally replayed.
    args.output.mkdir(parents=True, exist_ok=False)
    capture = Capture(args)
    metadata = {"schema": "author-greedy-no-mtp-capture-v2", "completed": False,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "maximum_generation_requests": 1 + args.benchmark_runs,
                "mtp_enabled": False, "drafter_enabled": False, "pld_enabled": False}
    try:
        process = verify_process(args.pid, args.model_dir, args.runtime_dir, urllib.parse.urlparse(args.base_url).port or 80)
        metadata["service_before"] = process
        save(args.output, "service-process-before.json", process)
        status, _ = capture.request("/v1/models", "models-before")
        loaded = verify_loaded(status)
        prompt = args.prompt_file.read_bytes().decode("utf-8", errors="strict")
        require(prompt and len(prompt.encode()) <= 64 * 1024, "Prompt must be nonempty and at most 64 KiB")
        (args.output / "prompt.txt").write_bytes(prompt.encode("utf-8"))
        tokenized, _ = capture.request("/tokenize", "prompt-tokenize", {"content": prompt})
        prompt_ids = tokenized["tokens"]
        raw_by_id, ids_by_text = vocabulary_decoder(json.loads((args.model_dir / "tokenizer.json").read_text()))
        require(prompt_ids and all(type(i) is int and i in raw_by_id for i in prompt_ids), "Invalid prompt token IDs")
        require(b"".join(raw_by_id[i] for i in prompt_ids).decode("utf-8") == prompt, "Raw prompt tokenization does not roundtrip")
        save(args.output, "prompt-token-ids.json", prompt_ids)
        body = {"model": loaded["id"], "prompt": prompt, "max_tokens": args.max_tokens,
                "temperature": 0, "top_p": 1, "top_k": 1, "repeat_penalty": 1,
                "presence_penalty": 0, "seed": 42, "stream": False, "logprobs": 1,
                "enable_mtp": False, "enable_drafter": False, "enable_pld": False}
        response, reference_timing = capture.request("/v1/completions", "completion", body)
        ids, evidence, text, coverage = recover_ids(response, raw_by_id, ids_by_text, prompt_ids, args.max_tokens)
        detokenized, _ = capture.request("/detokenize", "output-detokenize", {"tokens": ids})
        require(detokenized["content"] == text, "Generated IDs fail server /detokenize roundtrip")
        save(args.output, "generated-token-ids.json", ids)
        save(args.output, "token-id-recovery.json", evidence)
        (args.output / "generated-text.txt").write_text(text)
        reference = {"schema": "author-greedy-no-mtp-full-model-reference-v2", "prompt_token_ids": prompt_ids,
                     "generated_token_ids": ids, "generated_text": text, "max_new_tokens": args.max_tokens,
                     "mtp_enabled": False, "finish_reason": response["choices"][0]["finish_reason"],
                     "temperature": 0, "token_id_coverage": coverage,
                     "reference_generation_request_count": 1, "request_wall_seconds": reference_timing["wall_seconds"]}
        save(args.output, "reference.json", reference)
        benchmarks = []
        benchmark_body = {key: value for key, value in body.items() if key != "logprobs"}
        for index in range(args.benchmark_runs):
            # Inspect the live identity immediately before every generation.
            require(verify_process(args.pid, args.model_dir, args.runtime_dir,
                    urllib.parse.urlparse(args.base_url).port or 80) == process, "Server process identity changed")
            before, _ = capture.request("/v1/models", f"benchmark-{index + 1}-models-before")
            verify_loaded(before, loaded["id"])
            value, timing = capture.request("/v1/completions", f"benchmark-{index + 1}", benchmark_body)
            require(value["usage"]["prompt_tokens"] == len(prompt_ids), "Benchmark prompt count differs")
            require(0 < value["usage"]["completion_tokens"] <= args.max_tokens, "Benchmark exceeded output token budget")
            require(value["choices"][0].get("logprobs") is None, "Benchmark unexpectedly returned logprobs")
            benchmarks.append({"run": index + 1, "request_wall_seconds": timing["wall_seconds"],
                               "usage": value["usage"], "finish_reason": value["choices"][0]["finish_reason"],
                               "text_matches_reference": value["choices"][0]["text"] == text,
                               "server_log": timing["server_log"],
                               "token_ids": None, "token_id_scope": "No per-token boundaries in this benchmark response; IDs are not inferred from whole text."})
            save(args.output, "benchmark-runs.json", benchmarks)
        after, _ = capture.request("/v1/models", "models-after")
        verify_loaded(after, loaded["id"])
        final_process = verify_process(args.pid, args.model_dir, args.runtime_dir, urllib.parse.urlparse(args.base_url).port or 80)
        require(final_process == process, "Server process identity changed during capture")
        save(args.output, "service-process-after.json", final_process)
        server_rates = [run["server_log"]["parsed_timing"]["decode_tokens_per_second"] for run in benchmarks
                        if run["server_log"].get("parsed_timing") is not None]
        save(args.output, "benchmark-summary.json", {
            "runs": len(benchmarks), "warmup": "The preceding same-prompt reference generation; it requests logprobs and is excluded from benchmark statistics.",
            "request_wall_seconds": [run["request_wall_seconds"] for run in benchmarks],
            "median_request_wall_seconds": statistics.median(run["request_wall_seconds"] for run in benchmarks) if benchmarks else None,
            "server_decode_tokens_per_second": server_rates,
            "median_server_decode_tokens_per_second": statistics.median(server_rates) if len(server_rates) == len(benchmarks) and benchmarks else None,
            "all_texts_match_reference": all(run["text_matches_reference"] for run in benchmarks) if benchmarks else None,
            "physical_dram_bytes": None, "physical_dram_bandwidth_gbps": None,
            "timing_scope": "HTTP wall includes transport and server work. Server decode tok/s comes only from unambiguously scoped exact log lines and uses the author's completion-token/time convention. No DRAM counters or inferred physical bandwidth.",
        })
        metadata.update(completed=True, service_after=final_process,
                        source_commit=subprocess.check_output(["git", "-C", str(args.runtime_dir), "rev-parse", "HEAD"], text=True).strip(),
                        source_model_directory=str(args.model_dir), prompt_source=str(args.prompt_file),
                        source_sha256={name: sha(path) for name, path in {
                            "binary": args.runtime_dir / "zig-out/bin/mlx-serve", "server.zig": args.runtime_dir / "src/server.zig",
                            "transformer.zig": args.runtime_dir / "src/transformer.zig", "tokenizer.zig": args.runtime_dir / "src/tokenizer.zig",
                            "tokenizer.json": args.model_dir / "tokenizer.json", "config.json": args.model_dir / "config.json",
                            "model.safetensors.index.json": args.model_dir / "model.safetensors.index.json", "capture_script": Path(__file__)}.items()},
                        token_id_coverage=coverage,
                        prompt_format="Raw prompt-file bytes decoded as UTF-8; /v1/completions adds no chat template.",
                        speculation_evidence=["Live PID --no-mtp --no-drafter --no-pld before/after and before every benchmark",
                            "Live models metadata mtp_loaded=false and drafter_loaded=false before/after and before every benchmark",
                            "Every generation explicitly sets enable_mtp=false, enable_drafter=false, enable_pld=false",
                            "PLD has no loaded-state field in this API; process flag and request flags are the live PLD evidence"],
                        weight_hash_scope="Only source/binary/config/index/tokenizer are hashed; checkpoint payloads are not reread.")
        print(json.dumps({"output": str(args.output), "api_exposed_tokens": len(ids), "token_id_coverage": coverage,
                          "generation_requests": capture.generations, "benchmark_server_decode_tps": server_rates}, ensure_ascii=False))
    except Exception as error:
        metadata["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        metadata["generation_requests_attempted"] = capture.generations
        metadata["http_requests_attempted"] = len(capture.requests)
        metadata["http_requests_completed"] = sum(record["completed"] for record in capture.requests)
        metadata["fixture_files_sha256"] = {path.name: sha(path) for path in args.output.iterdir()
                                           if path.is_file() and path.name != "metadata.json"}
        save(args.output, "metadata.json", metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory; must not exist")
    parser.add_argument("--pid", type=int, required=True, help="Existing author server PID; never started or stopped here")
    parser.add_argument("--prompt-file", type=Path, required=True, help="Exact raw prompt, including any intended template")
    parser.add_argument("--max-tokens", type=int, default=64, choices=range(1, 65), metavar="1...64")
    parser.add_argument("--benchmark-runs", type=int, default=0, choices=range(4), metavar="0...3")
    parser.add_argument("--server-log", type=Path, help="Optional existing log; only newly appended request-scoped bytes are collected")
    parser.add_argument("--base-url", default="http://127.0.0.1:11235")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME)
    args = parser.parse_args()
    endpoint = urllib.parse.urlparse(args.base_url)
    require(endpoint.scheme == "http" and endpoint.hostname in ("127.0.0.1", "localhost", "::1")
            and not endpoint.path.strip("/") and not endpoint.query and not endpoint.fragment
            and endpoint.username is None, "Only a local plain HTTP service root is accepted")
    require(args.pid > 0, "PID must be positive")
    args.base_url = args.base_url.rstrip("/")
    for name in ["output", "prompt_file", "model_dir", "runtime_dir", "server_log"]:
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    require(args.prompt_file.is_file(), "Prompt file is missing")
    if args.server_log is not None:
        require(args.server_log.is_file(), "Supplied server log is missing")
    run(args)


if __name__ == "__main__":
    main()
