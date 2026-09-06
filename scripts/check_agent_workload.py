#!/usr/bin/env python3
"""Check a frozen synthetic agent workload and, optionally, generated JSON.

No inference or tool execution. --retokenize calls only the existing CPU
tokenize command, with argv (never a shell), and compares frozen token IDs.
This functional gate is independent of AR/MTP token-for-token compatibility.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def shape(value, schema, location="$"):
    """The deliberately small schema vocabulary used by these two fixtures."""
    kind = schema["type"]
    kinds = {"object": dict, "array": list, "string": str, "boolean": bool, "integer": int}
    require(type(value) is kinds[kind], f"{location}: expected {kind}")
    if kind == "object":
        properties = schema["properties"]
        require(set(value) == set(properties), f"{location}: missing or extra keys")
        for key, child in properties.items():
            shape(value[key], child, location + "." + key)
    elif kind == "array":
        if "minItems" in schema:
            require(len(value) >= schema["minItems"], f"{location}: too few items")
        if "maxItems" in schema:
            require(len(value) <= schema["maxItems"], f"{location}: too many items")
        for i, child in enumerate(value):
            shape(child, schema["items"], f"{location}[{i}]")
        if schema.get("uniqueItems"):
            require(len({json.dumps(v, sort_keys=True) for v in value}) == len(value),
                    f"{location}: duplicate items")


def check_text(text, expected, schema):
    value = strict_json(text.strip())
    shape(value, schema)
    require(value == expected, "JSON is valid but differs from the frozen expected answer")
    return value


def fixture(directory):
    directory = directory.resolve()
    manifest = strict_json((directory / "manifest.json").read_text())
    require(manifest["schema"] == "qwen38-synthetic-agent-workload-v1", "Unknown fixture schema")
    require(manifest["synthetic"] is True, "Fixture must explicitly declare synthetic data")
    for name, fingerprint in manifest["files"].items():
        path = (directory / name).resolve()
        require(path.parent == directory, f"Fixture path escapes its directory: {name}")
        require(path.is_file() and digest(path) == fingerprint["sha256"], f"Changed fixture file: {name}")
        require(path.stat().st_size == fingerprint["bytes"], f"Changed fixture size: {name}")
    system = (directory / "system-prompt.txt").read_text().strip()
    user = (directory / "user-prompt.txt").read_text().strip()
    prompt = (directory / "prompt.txt").read_text()
    rendered = (f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{user}<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    require(prompt == rendered, "Frozen prompt differs from its system/user no-thinking framing")
    ids = strict_json((directory / "prompt-token-ids.json").read_text())
    require(type(ids) is list and all(type(v) is int and v >= 0 for v in ids), "Invalid frozen token IDs")
    require(10000 <= len(ids) <= 12000 and len(ids) == manifest["input_tokens"], "Input must contain 10k–12k tokens")
    tokenization = strict_json((directory / "tokenization.json").read_text())
    require(tokenization["tokens"] == ids, "Tokenization report differs from frozen token IDs")
    require(tokenization["rendered_prompt"] == prompt and tokenization["decoded"] == prompt,
            "Tokenization roundtrip changed prompt bytes")
    expected = strict_json((directory / "expected.json").read_text())
    schema = strict_json((directory / "response-schema.json").read_text())
    shape(expected, schema)
    source = (directory / "source-data.json").read_text()
    for anchor in manifest["answer_evidence"]:
        require(anchor["source_fragment"] in source and anchor["source_fragment"] in system,
                f"Missing frozen answer evidence: {anchor['id']}")
    checker = Path(__file__).resolve()
    require(digest(checker) == manifest["checker_sha256"], "Checker changed since fixture freeze")
    return directory, manifest, prompt, ids, expected, schema


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--generation", type=Path, action="append", default=[])
    parser.add_argument("--text-file", type=Path, action="append", default=[])
    parser.add_argument("--retokenize", action="store_true")
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--model-dir", type=Path)
    args = parser.parse_args()
    report = {"schema": "qwen38-agent-functional-check-v1", "passed": False,
              "fixture": str(args.fixture.resolve()), "checks": [],
              "scope": "Frozen input integrity and exact structured answer; no inference or AR/MTP equivalence claim"}
    try:
        directory, manifest, prompt, ids, expected, schema = fixture(args.fixture)
        report.update(fixture_id=manifest["fixture_id"], input_tokens=len(ids), fixture_integrity=True)
        if args.retokenize:
            require(args.runner is not None and args.model_dir is not None,
                    "--retokenize requires --runner and --model-dir")
            model = args.model_dir.resolve()
            for name, checksum in manifest["tokenizer_metadata_sha256"].items():
                require(digest(model / name) == checksum, f"Tokenizer metadata changed: {name}")
            with tempfile.TemporaryDirectory(prefix="agent-workload-tokenize-") as temporary:
                destination = Path(temporary) / "tokenization.json"
                command = [str(args.runner.resolve()), "tokenize", "--model-dir", str(model),
                           "--prompt", prompt, "--chat", "false", "--output", str(destination)]
                result = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
                require(result.returncode == 0, f"CPU tokenize failed: {result.stderr[-1500:]}")
                actual = strict_json(destination.read_text())
                require(actual["tokens"] == ids and actual["decoded"] == prompt and actual["rendered_prompt"] == prompt,
                        "CPU tokenize disagrees with frozen IDs/text")
            report["retokenized_exact"] = True
        for path in args.text_file:
            check_text(path.read_text(), expected, schema)
            report["checks"].append({"text_file": str(path.resolve()), "sha256": digest(path), "passed": True})
        for path in args.generation:
            data = strict_json(path.read_text())
            require(data["max_tokens"] in manifest["output_budgets"], "Unspecified output budget")
            trials = data["trials"]
            require(type(trials) is list and len(trials) > 0, "Generation has no trials")
            for index, trial in enumerate(trials):
                actual_prompt = trial["prompt_tokens"]
                require(type(actual_prompt) is list and all(type(v) is int for v in actual_prompt),
                        "Generation prompt_tokens must be the full integer token-ID array")
                require(actual_prompt == ids, "Generation prompt IDs differ from frozen input")
                require(trial["finish_reason"] in ("length", "eos", "stop"), "Unknown finish reason")
                require(0 < len(trial["generated_token_ids"]) <= data["max_tokens"], "Invalid generated token count")
                check_text(trial["text"], expected, schema)
                report["checks"].append({"generation": str(path.resolve()), "trial": index,
                                          "finish_reason": trial["finish_reason"], "passed": True})
            report.setdefault("generation_sources", []).append({"path": str(path.resolve()), "sha256": digest(path)})
            report["generation_input_identity_note"] = (
                "Each trial's complete prompt_tokens integer array matched the frozen input IDs exactly.")
        report["functional_outputs_checked"] = len(report["checks"])
        report["passed"] = True
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        report["error"] = str(error)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
