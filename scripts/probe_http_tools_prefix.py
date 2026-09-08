#!/usr/bin/env python3
"""Probe an already running loopback server; never starts/stops a model process.

Retains actual request/response bytes and reassembles OpenAI SSE calls. The
client executes a deterministic local test-data function after the model emits
its call. No assistant output, tool selection or tool arguments are fabricated.
Prefix checks use the repository's public 11k system fixture, a per-run nonce,
and actual usage/health counters. HTTP wall time is not a GPU phase benchmark.
"""
from __future__ import annotations

import argparse
import base64
import http.client
import json
from pathlib import Path
import time
from urllib.parse import urlsplit
import uuid


MAX_RESPONSE_BYTES = 4 * 1024 * 1024
CONTROL_MARKERS = ("<tool_call", "</tool_call", "<function=", "<parameter=", "</parameter>")


def parse_response(status, headers, raw, stream):
    if status != 200:
        raise ValueError(f"HTTP {status}: {raw[:500]!r}")
    if not stream:
        result = json.loads(raw)
        assert result["object"] == "chat.completion"
        choice = result["choices"][0]
        assert choice["index"] == 0
        message = choice["message"]
        assert message["role"] == "assistant"
        return {"id": result["id"], "message": message, "finish_reason": choice["finish_reason"], "usage": result["usage"]}
    assert "text/event-stream" in dict(headers).get("content-type", "")
    events = []
    for part in raw.replace(b"\r\n", b"\n").split(b"\n\n"):
        if not part:
            continue
        text = part.decode("utf-8", errors="strict")
        data = "\n".join(line[5:].lstrip(" ") for line in text.split("\n") if line.startswith("data:"))
        if data:
            events.append(data)
    assert events and events[-1] == "[DONE]" and events.count("[DONE]") == 1
    chunks = [json.loads(event) for event in events[:-1]]
    assert chunks and not any("error" in chunk for chunk in chunks), chunks
    assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
    identities = {(chunk["id"], chunk["created"], chunk["model"]) for chunk in chunks}
    assert len(identities) == 1
    text_parts, calls, finishes, usage = [], {}, [], None
    for chunk in chunks:
        if "usage" in chunk:
            usage = chunk["usage"]
        for choice in chunk["choices"]:
            assert choice["index"] == 0
            delta = choice["delta"]
            if delta.get("content") is not None:
                text_parts.append(delta["content"])
            for part in delta.get("tool_calls", []):
                index = part["index"]
                assert isinstance(index, int) and 0 <= index < 16
                call = calls.setdefault(index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                if "id" in part:
                    assert not call["id"] or call["id"] == part["id"]
                    call["id"] = part["id"]
                if "type" in part:
                    assert part["type"] == "function"
                function = part.get("function", {})
                call["function"]["name"] += function.get("name", "")
                call["function"]["arguments"] += function.get("arguments", "")
            if choice.get("finish_reason") is not None:
                finishes.append(choice["finish_reason"])
    assert len(finishes) == 1 and usage is not None
    assert sorted(calls) == list(range(len(calls)))
    message = {"role": "assistant", "content": "".join(text_parts)}
    if calls:
        message["tool_calls"] = [calls[index] for index in sorted(calls)]
        if not message["content"]:
            message["content"] = None
    return {"id": chunks[0]["id"], "message": message, "finish_reason": finishes[0], "usage": usage,
            "sse_chunk_count": len(chunks)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:11236")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=("tools", "prefix", "all"), default="all")
    args = parser.parse_args()
    url = urlsplit(args.base_url)
    if url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost", "::1") or url.username or url.password or url.path not in ("", "/") or url.query or url.fragment:
        parser.error("--base-url must be an HTTP loopback origin")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("x") as file:
            file.write("{}\n")
    except FileExistsError:
        parser.error("--output must be new")
    report = {"schema": "qwen-http-tools-prefix-v1", "complete": False, "passed": False,
              "suite": args.suite, "base_url": args.base_url, "run_id": uuid.uuid4().hex,
              "requests": [], "checks": [], "tool_executions": [],
              "notes": ["Client only: does not start, stop, or reconfigure the server.",
                        "The weather function returns local test data, not real weather; model calls and responses are real.",
                        "HTTP wall seconds include tokenization, queueing, prefill, decode and transport; no phase speed claim.",
                        "cached_tokens is the actual server-reported reused prefix; cold and hit outputs are compared."]}

    def save():
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    def check(name, condition, **fields):
        report["checks"].append({"name": name, "passed": bool(condition), **fields})
        save()
        print(json.dumps({"check": name, "passed": bool(condition)}, ensure_ascii=False), flush=True)
        if not condition:
            raise AssertionError(name)

    def request(name, method, path, body=None, timeout=240):
        payload = None if body is None else json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        connection = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=timeout)
        row = {"name": name, "method": method, "path": path, "request": body,
               "request_body_utf8": payload.decode("utf-8") if payload is not None else None}
        report["requests"].append(row)
        started = time.monotonic()
        try:
            connection.request(method, path, body=payload, headers={"Content-Type": "application/json"} if payload is not None else {})
            response = connection.getresponse()
            row["status"] = response.status
            row["headers"] = [(name.lower(), value) for name, value in response.getheaders()]
            pieces, size = [], 0
            while True:
                part = response.read1(min(65536, MAX_RESPONSE_BYTES + 1 - size))
                if not part:
                    break
                if not pieces:
                    row["first_body_byte_seconds"] = time.monotonic() - started
                pieces.append(part); size += len(part)
                if size > MAX_RESPONSE_BYTES:
                    raise ValueError("Response exceeds harness bound")
            raw = b"".join(pieces)
            row["response_body_base64"] = base64.b64encode(raw).decode("ascii")
            row["response_body_utf8"] = raw.decode("utf-8", errors="strict")
            return row, raw
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            row["wall_seconds"] = time.monotonic() - started
            connection.close(); save()

    def health(name):
        row, raw = request(name, "GET", "/health", timeout=5)
        value = json.loads(raw)
        check(name, row["status"] == 200 and value.get("ready") is True, health=value)
        return value

    def chat(name, body):
        row, raw = request(name, "POST", "/v1/chat/completions", body)
        parsed = parse_response(row["status"], row["headers"], raw, body.get("stream", False))
        row["parsed"] = parsed
        content = parsed["message"].get("content") or ""
        check(name + "_no_control_leak", not any(marker in content for marker in CONTROL_MARKERS))
        return parsed, row

    try:
        initial = health("initial_health")
        model = initial["model"]
        report["server_pid"] = initial["pid"]
        check("initial_idle", initial.get("idle") is True and initial.get("reserved_tokens") == 0)
        models_row, models_raw = request("models", "GET", "/v1/models")
        check("model_identity", models_row["status"] == 200 and [item["id"] for item in json.loads(models_raw)["data"]] == [model])

        if args.suite in ("tools", "all"):
            tools = [{"type": "function", "function": {"name": "lookup_test_weather",
                "description": "查询本地测试数据中的天气观测。观测编号只能通过本函数获得。",
                "parameters": {"type": "object", "properties": {"city": {"type": "string", "enum": ["北京"]}},
                               "required": ["city"], "additionalProperties": False}}}]
            messages = [{"role": "system", "content": "你是测试数据助手。需要天气观测时调用工具，不猜测观测编号。拿到工具结果后，不再次调用工具，用一行报告城市、温度和原始观测编号。"},
                        {"role": "user", "content": "请调用 lookup_test_weather 查询北京的测试天气，并报告观测编号。"}]
            base = {"model": model, "messages": messages, "tools": tools, "tool_choice": "auto", "temperature": 0, "mtp_depth": 0, "max_tokens": 192}
            all_results = {}
            for stream in (False, True):
                mode = "sse" if stream else "nonstream"
                first, _ = chat(mode + "_tool_call", {**base, "stream": stream})
                calls = first["message"].get("tool_calls", [])
                check(mode + "_real_function_call", first["finish_reason"] == "tool_calls" and len(calls) == 1)
                call = calls[0]
                arguments = json.loads(call["function"]["arguments"])
                check(mode + "_tool_arguments", call["type"] == "function" and bool(call["id"]) and
                      call["function"]["name"] == "lookup_test_weather" and arguments == {"city": "北京"}, call=call)
                # This is a real client-side test function invocation. Its
                # deterministic fixture result never substitutes for model output.
                def lookup_test_weather(city):
                    if city != "北京":
                        raise ValueError("No test observation for city")
                    return {"city": city, "temperature_c": 23.5, "condition": "晴", "observation_id": "LOCAL-7319"}
                tool_result = lookup_test_weather(**arguments)
                report["tool_executions"].append({"mode": mode, "call_id": call["id"], "function": call["function"]["name"],
                    "arguments": arguments, "result": tool_result})
                followup = messages + [first["message"], {"role": "tool", "tool_call_id": call["id"],
                    "content": json.dumps(tool_result, ensure_ascii=False)}]
                final, _ = chat(mode + "_tool_result_followup", {**base, "messages": followup, "stream": stream})
                final_text = final["message"].get("content") or ""
                check(mode + "_tool_roundtrip", not final["message"].get("tool_calls") and final["finish_reason"] == "stop" and
                      "LOCAL-7319" in final_text and "23.5" in final_text, final_text=final_text)
                all_results[mode] = {"call": call["function"], "final_text": final_text}
            check("sse_nonstream_semantic_match", all_results["sse"] == all_results["nonstream"])
            none, _ = chat("tool_choice_none", {**base, "tool_choice": "none", "messages": [
                {"role": "user", "content": "只输出 TOOLS_DISABLED，不加解释。"}], "max_tokens": 32})
            check("none_has_no_call", not none["message"].get("tool_calls") and "TOOLS_DISABLED" in (none["message"].get("content") or ""))
            invalid = [
                ("required_choice", {**base, "tool_choice": "required"}),
                ("named_choice", {**base, "tool_choice": {"type": "function", "function": {"name": "lookup_test_weather"}}}),
                ("tools_with_mtp", {**base, "mtp_depth": 2}),
                ("unmatched_tool_result", {**base, "messages": messages + [{"role": "tool", "tool_call_id": "missing", "content": "x"}]}),
                ("strict_tool", {**base, "tools": [{"type": "function", "function": {**tools[0]["function"], "strict": True}}]}),
            ]
            for name, body in invalid:
                row, raw = request("reject_" + name, "POST", "/v1/chat/completions", body)
                check("reject_" + name, row["status"] == 400 and "error" in json.loads(raw))
            after_tools = health("tools_final_health")
            check("tools_released_requests", after_tools.get("idle") is True and after_tools.get("resident_sequences") == 0 and after_tools.get("reserved_tokens") == 0)

        if args.suite in ("prefix", "all"):
            before = health("prefix_initial_health")
            check("cache_enabled", isinstance(before.get("prefix_cache"), dict))
            fixture = Path(__file__).resolve().parents[1] / "fixtures/gpu-agent-11k/system-prompt.txt"
            fixture_text = fixture.read_text()
            nonce = report["run_id"]
            system = "Cache probe identity: " + nonce + "\n" + fixture_text
            messages = [{"role": "system", "content": system}, {"role": "user", "content": "请只输出 PREFIX_CACHE_OK，不要解释。"}]
            body = {"model": model, "messages": messages, "temperature": 0, "mtp_depth": 0, "max_tokens": 32}
            cold, cold_row = chat("prefix_cold", body)
            cold_cached = cold["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
            check("long_prefix_cold", cold_cached == 0 and 10_000 <= cold["usage"]["prompt_tokens"] < 16_384,
                  prompt_tokens=cold["usage"]["prompt_tokens"], cached_tokens=cold_cached)
            after_cold = health("prefix_after_cold")
            warm, warm_row = chat("prefix_hit", {**body, "stream": True})
            warm_cached = warm["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
            check("long_prefix_hit", warm_cached >= 10_000 and warm_cached % 416 == 0,
                  prompt_tokens=warm["usage"]["prompt_tokens"], cached_tokens=warm_cached)
            check("cold_hit_same_generation", cold["message"] == warm["message"] and cold["finish_reason"] == warm["finish_reason"] and
                  cold["usage"]["prompt_tokens"] == warm["usage"]["prompt_tokens"] and cold["usage"]["completion_tokens"] == warm["usage"]["completion_tokens"],
                  cold_wall_seconds=cold_row["wall_seconds"], hit_wall_seconds=warm_row["wall_seconds"])
            changed, _ = chat("prefix_changed_user_hit", {**body, "messages": [messages[0],
                {"role": "user", "content": "请只输出 PREFIX_OTHER_OK，不要解释。"}]})
            check("same_system_different_user_reuses", changed["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0) == warm_cached and
                  "PREFIX_OTHER_OK" in (changed["message"].get("content") or ""))
            different_system = "Cache probe identity: different-" + nonce + "\n" + fixture_text
            isolated, _ = chat("prefix_changed_system_miss", {**body, "messages": [{"role": "system", "content": different_system}, messages[1]]})
            check("different_system_isolated", isolated["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0) == 0)
            after = health("prefix_final_health")
            stats, old = after["prefix_cache"], before["prefix_cache"]
            check("cache_health_counters", stats["hits"] >= old["hits"] + 2 and stats["misses"] >= old["misses"] + 2 and
                  stats["published"] >= old["published"] + 2 and stats["entries"] > 0 and stats["logicalPayloadBytes"] > 0 and
                  stats["restoreFailures"] == old["restoreFailures"], before=old, after_cold=after_cold["prefix_cache"], after=stats)
            check("prefix_released_requests", after.get("idle") is True and after.get("resident_sequences") == 0 and after.get("reserved_tokens") == 0)
        last = health("final_health")
        check("same_server_owner", last["pid"] == report["server_pid"])
        report["complete"] = True
        report["passed"] = all(item["passed"] for item in report["checks"])
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    finally:
        save()
    print(json.dumps({"passed": report["passed"], "complete": report["complete"], "error": report.get("error"), "output": str(output)}, ensure_ascii=False), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
