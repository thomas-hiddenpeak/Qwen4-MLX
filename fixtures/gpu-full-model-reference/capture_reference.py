#!/usr/bin/env python3
"""Capture exactly one bounded greedy reference from the existing author service.

No model is loaded or service started here. Refuses to overwrite an existing
completion response. Token IDs are reconstructed from per-token logprob text
only when the tokenizer vocabulary gives exactly one lossless candidate.
"""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.request


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
MODEL = ROOT / "experiments/qwen38-ssd/models/Qwen3.8-Flash-Next-MLX-SSD-Stream"
RUNTIME = ROOT / "experiments/qwen38-ssd/runtime/mlx-serve"
BASE = "http://127.0.0.1:11235"
PID = 10591


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for data in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def request(endpoint, name, body=None):
    if body is not None:
        save(name + "-request.json", body)
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode()
    req = urllib.request.Request(BASE + endpoint, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as response:
        raw = response.read()
        assert response.status == 200
        (OUT / (name + "-response.json")).write_bytes(raw)
        save(name + "-http.json", {"url": BASE + endpoint, "status": response.status,
                                   "headers": dict(response.headers), "body_bytes": len(raw)})
    return json.loads(raw)


def vocabulary_decoder(tokenizer):
    assert tokenizer["decoder"]["type"] == "ByteLevel"
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


def main():
    if (OUT / "completion-response.json").exists():
        raise RuntimeError("Reference response already exists; refusing a repeated generation or overwrite")
    process = subprocess.check_output(["ps", "-p", str(PID), "-o", "pid=,lstart=,command="], text=True).strip()
    assert "--port 11235" in process and "--no-mtp" in process and str(MODEL) in process
    (OUT / "service-process.txt").write_text(process + "\n")
    status = request("/v1/models", "models-before")
    loaded = [entry for entry in status["data"] if entry.get("loaded") and entry.get("state") == "ready"]
    assert len(loaded) == 1 and loaded[0]["meta"]["mtp_loaded"] is False
    assert loaded[0]["meta"]["drafter_loaded"] is False
    prompt = ("<|im_start|>user\n请用一句话解释太阳为什么发光。<|im_end|>\n"
              "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    (OUT / "prompt.txt").write_text(prompt)
    prompt_response = request("/tokenize", "prompt-tokenize", {"content": prompt})
    prompt_ids = prompt_response["tokens"]
    assert prompt_ids and all(isinstance(identifier, int) and 0 <= identifier < 248320 for identifier in prompt_ids)
    tokenizer = json.loads((MODEL / "tokenizer.json").read_text())
    raw_by_id, ids_by_text = vocabulary_decoder(tokenizer)
    assert b"".join(raw_by_id[i] for i in prompt_ids).decode("utf-8") == prompt
    body = {"model": loaded[0]["id"], "prompt": prompt, "max_tokens": 16,
            "temperature": 0, "top_p": 1, "top_k": 1, "repeat_penalty": 1,
            "presence_penalty": 0, "seed": 42, "stream": False, "logprobs": 1,
            "enable_mtp": False, "enable_drafter": False, "enable_pld": False}
    started = time.perf_counter()
    response = request("/v1/completions", "completion", body)
    request_seconds = time.perf_counter() - started
    choice = response["choices"][0]
    logprobs = choice["logprobs"]
    assert logprobs is not None, "No per-token logprobs; exact output ID recovery is unavailable"
    texts = logprobs["tokens"]
    assert len(texts) == response["usage"]["completion_tokens"]
    assert response["usage"]["prompt_tokens"] == len(prompt_ids)
    assert len(texts) <= 16 and texts
    generated_ids, evidence = [], []
    offset = 0
    for index, text in enumerate(texts):
        assert "\ufffd" not in text, "Lossy UTF-8 replacement prevents exact output ID recovery"
        candidates = ids_by_text.get(text, [])
        assert len(candidates) == 1, ("Ambiguous/nonexistent vocabulary inverse", index, text, candidates)
        token_id = candidates[0]
        assert logprobs["text_offset"][index] == offset
        raw = raw_by_id[token_id]
        assert raw == text.encode("utf-8")
        evidence.append({"position": index, "id": token_id, "text": text, "utf8_hex": raw.hex(),
                         "text_offset": offset, "unique_inverse_candidate_count": len(candidates),
                         "logprob": logprobs["token_logprobs"][index]})
        generated_ids.append(token_id)
        offset += len(raw)
    decoded = b"".join(raw_by_id[i] for i in generated_ids).decode("utf-8")
    assert decoded == choice["text"]
    detokenized = request("/detokenize", "output-detokenize", {"tokens": generated_ids})
    assert detokenized["content"] == decoded
    after = request("/v1/models", "models-after")
    assert after["data"][0]["meta"]["mtp_loaded"] is False
    save("prompt-token-ids.json", prompt_ids)
    save("generated-token-ids.json", generated_ids)
    save("token-id-recovery.json", evidence)
    (OUT / "generated-text.txt").write_text(decoded)
    save("reference.json", {"schema": "author-greedy-no-mtp-full-model-reference-v1",
                            "prompt_token_ids": prompt_ids, "generated_token_ids": generated_ids,
                            "generated_text": decoded, "max_new_tokens": 16, "mtp_enabled": False,
                            "finish_reason": choice["finish_reason"], "temperature": 0,
                            "generation_request_count": 1})
    save("metadata.json", {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(["git", "-C", str(RUNTIME), "rev-parse", "HEAD"], text=True).strip(),
        "source_model_directory": str(MODEL), "service_process": process,
        "source_binary": str(RUNTIME / "zig-out/bin/mlx-serve"),
        "source_sha256": {name: sha(path) for name, path in {
            "binary": RUNTIME / "zig-out/bin/mlx-serve", "server.zig": RUNTIME / "src/server.zig",
            "transformer.zig": RUNTIME / "src/transformer.zig", "tokenizer.zig": RUNTIME / "src/tokenizer.zig",
            "tokenizer.json": MODEL / "tokenizer.json", "config.json": MODEL / "config.json",
            "model.safetensors.index.json": MODEL / "model.safetensors.index.json",
            "capture_reference.py": Path(__file__)}.items()},
        "prompt_format": "Explicit raw Qwen role markers and closed think prefix; /v1/completions applies no server-side chat template.",
        "generation_request_count": 1, "request_wall_seconds": request_seconds,
        "timing_scope": "One reference capture, not a throughput benchmark; includes HTTP and requested per-token logprobs overhead.",
        "mtp_evidence": ["Live PID command line --no-mtp", "Live models-before/after meta.mtp_loaded=false",
                         "Request enable_mtp=false", "Source handler additionally disables all speculation when logprobs_n>0"],
        "exact_token_id_evidence": "Per-token logprob text boundaries map uniquely to the source ByteLevel vocabulary; UTF-8 replacement rejected; raw-byte offsets, total text and /detokenize roundtrip all checked. Full-text retokenization was not used.",
        "weight_payload_hash_scope": "Config/index/tokenizer/binary/source hashed; this capture does not reread/hash all checkpoint payloads.",
        "source_api_lines": {"raw_prompt_tokenization": "src/server.zig:5735", "speculation_disabled_for_logprobs": "src/server.zig:5844",
                             "per_token_logprob_text": "src/server.zig:9328", "bytelevel_decode": "src/tokenizer.zig:398"},
        "fixture_files_sha256": {path.name: sha(path) for path in OUT.iterdir() if path.is_file() and path.name != "metadata.json"},
    })
    print(json.dumps({"prompt_tokens": len(prompt_ids), "generated_tokens": len(generated_ids),
                      "text": decoded, "token_ids": generated_ids, "seconds": request_seconds}, ensure_ascii=False))


if __name__ == "__main__":
    main()
