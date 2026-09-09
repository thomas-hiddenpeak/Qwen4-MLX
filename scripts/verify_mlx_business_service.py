#!/usr/bin/env python3
"""Exclusive, serial local-service capacity/cache probe; Python 3.9 stdlib only.

The caller owns the service lifecycle. Long mode checks a synthetic context
boundary and repeat-cache reuse, not model quality or production endurance.
"""
import argparse
import hashlib
import json
import pathlib
import shutil
import sys
import time
import urllib.error
import urllib.request


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


class Probe:
    def __init__(self, args):
        self.args = args
        self.root = pathlib.Path(args.output).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.base = args.server_url.rstrip("/")
        self.events = self.root.joinpath("progress.ndjson").open("x", encoding="utf-8")
        self.results = []

    def event(self, event_name, **fields):
        record = dict(event=event_name, unix_seconds=time.time(), **fields)
        line = json.dumps(record, ensure_ascii=False)
        self.events.write(line + "\n")
        self.events.flush()
        print(line, flush=True)

    def http(self, path, body=None, timeout=60):
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(self.base + path, data=data,
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            require(response.status == 200, "HTTP status %s for %s" % (response.status, path))
            return json.load(response)

    def tokenize(self, content):
        tokens = self.http("/tokenize", {"content": content})["tokens"]
        require(isinstance(tokens, list) and all(type(x) is int for x in tokens),
                "invalid tokenization result")
        return tokens

    def metrics(self):
        value = self.http("/metrics.json")
        require("counters" in value and "gauges" in value, "missing metrics schema")
        return value

    def run_request(self, name, path, payload, require_cached=False,
                    expected_prompt=None, exact_completion=None, expected_content=None):
        directory = self.root / name
        directory.mkdir()
        write_json(directory / "request.json", payload)
        before = self.metrics()
        write_json(directory / "metrics-before.json", before)
        require(before["gauges"]["requests_running"] == 0 and
                before["gauges"]["requests_waiting"] == 0, "service is not idle")
        self.event("request_start", name=name, endpoint=path,
                   expected_prompt_tokens=expected_prompt, max_tokens=payload["max_tokens"])
        began = time.monotonic()
        record = dict(name=name, endpoint=path, status="failed", http_status=None,
                      elapsed_seconds=None, first_visible_token_seconds=None,
                      usage=None, content="", reasoning_content="", finish_reasons=[],
                      timings=None, done=False)
        error = None
        try:
            request = urllib.request.Request(
                self.base + path, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
            with urllib.request.urlopen(request, timeout=self.args.timeout) as response:
                record["http_status"] = response.status
                require(response.status == 200, "inference HTTP status is not 200")
                require("text/event-stream" in response.headers.get("Content-Type", ""),
                        "inference response is not SSE")
                with (directory / "response.sse").open("xb") as raw, \
                        (directory / "chunks.ndjson").open("x", encoding="utf-8") as chunks:
                    for line in response:
                        raw.write(line)
                        raw.flush()
                        elapsed = time.monotonic() - began
                        require(elapsed <= self.args.timeout, "request exceeded total deadline")
                        if not line.startswith(b"data:"):
                            continue
                        data = line[5:].strip()
                        if data == b"[DONE]":
                            record["done"] = True
                            break
                        chunk = json.loads(data)
                        chunks.write(json.dumps(dict(elapsed_seconds=elapsed, body=chunk),
                                                ensure_ascii=False) + "\n")
                        chunks.flush()
                        require("error" not in chunk, "SSE error: %s" % chunk.get("error"))
                        if chunk.get("usage") is not None:
                            require(record["usage"] is None, "duplicate SSE usage chunks")
                            record["usage"] = chunk["usage"]
                        if chunk.get("timings") is not None:
                            record["timings"] = chunk["timings"]
                        for choice in chunk.get("choices", []):
                            require(choice.get("index", 0) == 0, "unexpected multiple choices")
                            delta = choice.get("delta") or {}
                            content = choice.get("text") or delta.get("content") or ""
                            reasoning = delta.get("reasoning_content") or ""
                            if content or reasoning:
                                if record["first_visible_token_seconds"] is None:
                                    record["first_visible_token_seconds"] = elapsed
                                    self.event("first_visible_token", name=name, elapsed_seconds=elapsed)
                                record["content"] += content
                                record["reasoning_content"] += reasoning
                            if choice.get("finish_reason") is not None:
                                record["finish_reasons"].append(choice["finish_reason"])
            record["elapsed_seconds"] = time.monotonic() - began
            require(record["done"], "SSE missing [DONE]")
            require(len(record["finish_reasons"]) == 1 and
                    record["finish_reasons"][0] in ("stop", "length"), "invalid finish reason")
            usage = record["usage"]
            require(isinstance(usage, dict), "SSE missing usage")
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                require(type(usage.get(field)) is int and usage[field] >= 0,
                        "invalid usage.%s" % field)
            require(usage["prompt_tokens"] + usage["completion_tokens"] == usage["total_tokens"],
                    "usage token conservation failed")
            require(0 < usage["completion_tokens"] <= payload["max_tokens"],
                    "invalid generated token count")
            if expected_prompt is not None:
                require(usage["prompt_tokens"] == expected_prompt, "prompt was truncated or token count differs")
            else:
                require(10000 <= usage["prompt_tokens"] <= 14000, "short phase prompt is outside 10k-14k")
            if exact_completion is not None:
                require(usage["completion_tokens"] == exact_completion, "wrong boundary output count")
            if expected_content is not None:
                record["expected_content"] = expected_content
                record["content_matches_expected"] = record["content"].strip() == expected_content
                require(record["content_matches_expected"], "response does not match requested marker")
        except urllib.error.HTTPError as exc:
            record["http_status"] = exc.code
            record["http_error_body"] = exc.read().decode("utf-8", errors="replace")
            error = exc
        except Exception as exc:
            error = exc
        record["elapsed_seconds"] = time.monotonic() - began
        try:
            # The request must be the sole completion between snapshots. A short
            # poll accommodates the independent gauge sampler after final SSE.
            after = self.metrics()
            deadline = time.monotonic() + 5
            while error is None and time.monotonic() < deadline and (
                    after["counters"]["requests_success_total"] == before["counters"]["requests_success_total"] or
                    after["gauges"]["requests_running"] != 0):
                time.sleep(0.1)
                after = self.metrics()
            write_json(directory / "metrics-after.json", after)
            delta = {key: after["counters"][key] - value
                     for key, value in before["counters"].items()}
            record["metrics_counter_delta"] = delta
            if error is None:
                require(delta["requests_success_total"] == 1, "not exactly one successful request between metrics")
                require(delta["requests_cancelled_total"] == 0, "cancelled request observed")
                require(after["gauges"]["requests_running"] == 0 and
                        after["gauges"]["requests_waiting"] == 0, "service did not return idle")
                require(delta["prompt_tokens_total"] == record["usage"]["prompt_tokens"], "metrics prompt count mismatch")
                require(delta["generation_tokens_total"] == record["usage"]["completion_tokens"], "metrics output count mismatch")
                cached = delta["prefix_cache_tokens_total"]
                require(0 <= cached <= delta["prompt_tokens_total"], "invalid cached token count")
                require(delta["prefill_tokens_total"] + cached == delta["prompt_tokens_total"],
                        "metrics prefill/cache token conservation failed")
                details = record["usage"].get("prompt_tokens_details", {})
                if path == "/v1/chat/completions":
                    require(type(details.get("cached_tokens")) is int, "chat usage missing cached_tokens")
                    require(details["cached_tokens"] == cached, "usage and metrics cache counts differ")
                    record["cached_tokens_source"] = "usage_and_metrics_delta"
                else:
                    record["cached_tokens_source"] = "metrics_delta"
                    record["raw_api_cached_tokens_present"] = "cached_tokens" in details
                    if "cached_tokens" in details:
                        require(details["cached_tokens"] == cached, "usage and metrics cache counts differ")
                record["cached_tokens"] = cached
                record["observed_cache_state"] = "cold" if cached == 0 else "warm"
                record["uncached_prefill_tokens"] = delta["prefill_tokens_total"]
                if require_cached:
                    require(cached > 0 and delta["prefix_cache_hits_total"] == 1,
                            "repeated/appended request did not reuse a cached prefix")
                record["status"] = "passed"
        except Exception as exc:
            if error is None:
                error = exc
        if error is not None:
            record["error"] = "%s: %s" % (type(error).__name__, error)
        write_json(directory / "result.json", record)
        self.results.append(record)
        self.event("request_complete", **{key: value for key, value in record.items()
                                         if key not in ("content", "reasoning_content")})
        if error is not None:
            raise error
        return record

    def short(self):
        fixtures = pathlib.Path(self.args.fixtures).resolve()
        marker = "CACHE_READY_7319"
        system = ("本次缓存测试的固定标记是 %s。用户要求返回固定标记时，只返回该标记，不作解释。\n\n" % marker +
                  (fixtures / "system-prompt.txt").read_text(encoding="utf-8"))
        user = "请只返回系统提示词开头给出的固定标记，不要输出其他内容。"
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        payload = dict(model=self.model, messages=messages, max_tokens=64, temperature=0,
                       enable_thinking=False, stream=True, stream_options={"include_usage": True})
        write_json(self.root / "prompt-provenance.json", dict(
            phase="short", fixture_directory=str(fixtures),
            system_tokens=len(self.tokenize(system)), user_tokens=len(self.tokenize(user)),
            exact_chat_prompt_tokens_source="response_usage", quality_evaluation="bounded_marker_retrieval"))
        first = self.run_request("short-cold", "/v1/chat/completions", payload, expected_content=marker)
        second = self.run_request("short-repeat", "/v1/chat/completions", payload, require_cached=True,
                                  expected_prompt=first["usage"]["prompt_tokens"], expected_content=marker)
        appended = dict(payload)
        appended["messages"] = messages + [
            {"role": "assistant", "content": second["content"]},
            {"role": "user", "content": "现在请只返回新标记 CACHE_BRANCH_8421，不要再返回旧标记或其他内容。"}]
        self.run_request("short-append", "/v1/chat/completions", appended, require_cached=True,
                         expected_content="CACHE_BRANCH_8421")

    def long(self):
        target = 262143
        # ASCII-only token boundaries avoid ending in half of a UTF-8 character.
        # This deliberately synthetic capacity prompt makes no quality claim.
        seed = ("This is a synthetic cache capacity check. Read the following records.\n" +
                "Record: the agent reads a repository, checks a cache, and reports a result.\n" * 1024)
        seed_ids = self.tokenize(seed)
        source = seed * (target // len(seed_ids) + 2)
        ids = self.tokenize(source)
        require(len(ids) >= target, "synthetic source too short")
        selected = ids[:target]
        prompt = self.http("/detokenize", {"tokens": selected})["content"]
        roundtrip = self.tokenize(prompt)
        require(roundtrip == selected and len(roundtrip) == target,
                "exact boundary prompt token roundtrip failed")
        self.root.joinpath("prompt.txt").write_text(prompt, encoding="utf-8")
        write_json(self.root / "prompt-token-ids.json", selected)
        write_json(self.root / "prompt-provenance.json", dict(
            phase="long", prompt_tokens=target, max_tokens=1, total_context_tokens=262144,
            prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            token_roundtrip_exact=True, synthetic_capacity_only=True, quality_evaluation=False))
        self.event("prompt_ready", prompt_tokens=target, total_context_tokens=262144)
        payload = dict(model=self.model, prompt=prompt, max_tokens=1, temperature=0,
                       stream=True, stream_options={"include_usage": True})
        first = self.run_request("long-cold", "/v1/completions", payload,
                                 expected_prompt=target, exact_completion=1)
        second = self.run_request("long-repeat", "/v1/completions", payload, require_cached=True,
                                  expected_prompt=target, exact_completion=1)
        write_json(self.root / "repeat-comparison.json", dict(
            greedy_outputs_equal=first["content"] == second["content"],
            bitwise_model_correctness_verified=False, capacity_only=True))

    def run(self):
        outcome = "failed"
        error = None
        try:
            shutil.copyfile(pathlib.Path(__file__).resolve(), self.root / "service_probe.py")
            models = self.http("/v1/models")
            write_json(self.root / "models-before.json", models)
            require(bool(models.get("data")), "no loaded model")
            self.model = models["data"][0]["id"]
            self.event("probe_start", phase=self.args.phase, server=self.base, model=self.model)
            getattr(self, self.args.phase)()
            outcome = "passed"
        except Exception as exc:
            error = "%s: %s" % (type(exc).__name__, exc)
        finally:
            summary = dict(status=outcome, phase=self.args.phase, server=self.base,
                           requests=self.results, error=error, capacity_only=self.args.phase == "long",
                           quality_evaluation="bounded_marker_retrieval" if self.args.phase == "short" else False,
                           production_endurance_verified=False,
                           ttft_definition="first nonempty content or reasoning SSE payload; null if none")
            write_json(self.root / "summary.json", summary)
            self.event("probe_complete", status=outcome, phase=self.args.phase, error=error)
            self.events.close()
        return 0 if outcome == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("short", "long"))
    parser.add_argument("--output", required=True, help="New evidence directory; refuses overwrite")
    parser.add_argument("--server-url", default="http://127.0.0.1:11235")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Socket stall timeout and checked request deadline, seconds")
    parser.add_argument("--fixtures", default=str(pathlib.Path(__file__).resolve().parents[1] /
                                                 "fixtures/gpu-agent-11k"))
    args = parser.parse_args()
    require(args.timeout > 0, "timeout must be positive")
    return Probe(args).run()


if __name__ == "__main__":
    sys.exit(main())
