#!/usr/bin/env python3
"""Small external acceptance check for an already-running CoreAI HTTP service.

Uses only the standard library. This does not start or configure the service.
Every request and raw response is saved beside --output. A pass establishes
only these HTTP/cache/cancellation examples, not general model quality.
"""

from __future__ import annotations

import argparse
import datetime as dt
import http.client
import json
import math
from pathlib import Path
import socket
import struct
import sys
import time
from typing import Any
from urllib.parse import urlsplit
import uuid


MAX_RESPONSE_BYTES = 2 * 1024 * 1024
NUMBERS_ONLY = "只输出最终数字，不要解释、标点或思考过程。"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def parse_sse(raw: bytes) -> dict[str, Any]:
    """Parse event boundaries, including multiline data and CRLF framing."""
    text = raw.decode("utf-8", errors="strict").replace("\r\n", "\n")
    require(text.endswith("\n\n"), "SSE ended with an incomplete event")
    events: list[dict[str, Any]] = []
    done = False
    parts: list[str] = []
    identity = None
    usage = None
    reason = None
    for block in text.split("\n\n"):
        data = []
        for line in block.splitlines():
            if line.startswith("data:"):
                value = line[5:]
                data.append(value[1:] if value.startswith(" ") else value)
        if not data:
            continue
        require(not done, "SSE has data after [DONE]")
        payload = "\n".join(data)
        if payload == "[DONE]":
            done = True
            continue
        event = json.loads(payload)
        require(isinstance(event, dict), "SSE event must be a JSON object")
        require("error" not in event, f"SSE error: {event.get('error')}")
        require(event.get("object") == "chat.completion.chunk", "Unexpected SSE object")
        current = (event.get("id"), event.get("created"), event.get("model"))
        require(all(value is not None for value in current), "Missing SSE request identity")
        if identity is None:
            identity = current
        require(identity == current, "SSE request identity changed between events")
        choices = event.get("choices")
        require(isinstance(choices, list) and len(choices) == 1, "Expected one SSE choice")
        choice = choices[0]
        require(choice.get("index") == 0, "Unexpected SSE choice index")
        delta = choice.get("delta")
        require(isinstance(delta, dict), "Missing SSE delta")
        content = delta.get("content", "")
        require(isinstance(content, str), "Non-text SSE content")
        require(reason is None, "SSE emitted another choice after its terminal event")
        parts.append(content)
        if choice.get("finish_reason") is not None:
            reason = choice["finish_reason"]
        if "usage" in event:
            require(usage is None, "Duplicate SSE usage")
            usage = event["usage"]
        events.append(event)
    require(done, "SSE is missing [DONE]")
    require(reason in ("stop", "length"), "SSE is missing a valid finish reason")
    require(usage is not None, "SSE is missing terminal usage")
    return {"text": "".join(parts), "usage": usage, "finish_reason": reason,
            "events": events, "model": identity[2] if identity else None}


def validate_usage(usage: Any) -> tuple[int, int]:
    require(isinstance(usage, dict), "Missing usage object")
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        require(type(usage.get(key)) is int and usage[key] >= 0, f"Invalid usage.{key}")
    require(usage["prompt_tokens"] > 0, "Expected nonempty prompt")
    require(0 < usage["completion_tokens"] <= 8, "Completion usage is outside 1...8")
    require(usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"],
            "Usage total does not match prompt plus completion")
    details = usage.get("prompt_tokens_details", {})
    require(isinstance(details, dict), "Invalid prompt_tokens_details")
    cached = details.get("cached_tokens", 0)
    require(type(cached) is int and 0 <= cached <= usage["prompt_tokens"], "Invalid cached_tokens")
    return usage["prompt_tokens"], cached


class Acceptance:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.url = urlsplit(args.base_url)
        if (self.url.scheme not in ("http", "https") or not self.url.hostname
                or self.url.query or self.url.fragment or self.url.username or self.url.password):
            raise ValueError("--base-url must be an HTTP(S) origin or path, without credentials/query/fragment")
        self.output = Path(args.output).resolve()
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.artifacts = self.output.parent / (self.output.stem + ".artifacts-" + uuid.uuid4().hex[:8])
        self.artifacts.mkdir()
        self.request_index = 0
        self.model = ""
        self.report: dict[str, Any] = {
            "version": 1, "started_at": utc_now(), "base_url": args.base_url,
            "timeout_seconds": args.timeout, "artifacts": str(self.artifacts),
            "scope": "Small HTTP/cache/isolation/cancellation examples; not general model quality acceptance.",
            "checks": [], "passed": False,
        }
        self.write_report()

    def write_report(self) -> None:
        temporary = self.output.with_suffix(self.output.suffix + ".tmp")
        save_json(temporary, self.report)
        temporary.replace(self.output)

    def check(self, name: str, function) -> Any:
        start = time.monotonic()
        record: dict[str, Any] = {"name": name, "started_at": utc_now()}
        print(f"START {name}", flush=True)
        result = None
        try:
            result = function()
            record.update(status="passed", details=result)
        except Exception as error:
            record.update(status="failed", error=f"{type(error).__name__}: {error}")
        record["seconds"] = time.monotonic() - start
        self.report["checks"].append(record)
        self.write_report()
        print(f"{record['status'].upper()} {name} ({record['seconds']:.2f}s)"
              + (f": {record['error']}" if "error" in record else ""), flush=True)
        return result

    def connect(self, timeout: float | None = None):
        cls = http.client.HTTPSConnection if self.url.scheme == "https" else http.client.HTTPConnection
        return cls(self.url.hostname, self.url.port, timeout=timeout or self.args.timeout)

    def path(self, endpoint: str) -> str:
        return self.url.path.rstrip("/") + endpoint

    def begin(self, name: str, method: str, endpoint: str, body: Any = None,
              timeout: float | None = None, wait_for_headers: bool = True):
        self.request_index += 1
        folder = self.artifacts / f"{self.request_index:03d}-{name}"
        folder.mkdir()
        save_json(folder / "request.json", {"method": method, "url": self.args.base_url.rstrip("/") + endpoint,
                                           "json": body, "started_at": utc_now()})
        connection = self.connect(timeout)
        encoded = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {"Accept": "application/json, text/event-stream", "Connection": "close"}
        if encoded is not None:
            headers["Content-Type"] = "application/json; charset=utf-8"
        transport = None
        try:
            connection.request(method, self.path(endpoint), body=encoded, headers=headers)
            transport = connection.sock  # getresponse may detach close-delimited sockets.
            if not wait_for_headers:
                # HTTPConnection.request also sends over an established TLS
                # socket for HTTPS. Queued requests must not wait for .started.
                save_json(folder / "response-not-read.json", {
                    "reason": "Queued cancellation sends the full request without waiting for response headers."})
                return connection, transport, None, folder
            response = connection.getresponse()
            save_json(folder / "headers.json", {"status": response.status, "reason": response.reason,
                                                "headers": response.getheaders()})
            return connection, transport, response, folder
        except Exception as error:
            save_json(folder / "error.json", {"error": f"{type(error).__name__}: {error}"})
            self.abort_connection(connection, transport)
            raise

    @staticmethod
    def abort_connection(connection, transport, response=None) -> dict[str, Any]:
        """Abort the full TCP connection, including an HTTPResponse-owned socket."""
        result: dict[str, Any] = {"at": utc_now(), "linger_zero": False, "errors": []}
        if transport is not None:
            try:
                transport.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                result["linger_zero"] = True
            except OSError as error:
                result["errors"].append(str(error))
        for item in (response, connection, transport):
            if item is not None:
                try:
                    item.close()
                except OSError as error:
                    result["errors"].append(str(error))
        return result

    def exchange(self, name: str, method: str, endpoint: str, body: Any = None,
                 timeout: float | None = None) -> tuple[int, dict[str, str], bytes, Path]:
        start = time.monotonic()
        connection, transport, response, folder = self.begin(name, method, endpoint, body, timeout)
        raw = bytearray()
        try:
            deadline = start + (timeout or self.args.timeout)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Response exceeded the request deadline")
                if transport is not None:
                    transport.settimeout(remaining)
                block = response.read1(65536)
                if not block:
                    break
                raw.extend(block)
                require(len(raw) <= MAX_RESPONSE_BYTES, "Response exceeds the acceptance tool's 2 MiB limit")
                if response.isclosed():
                    break
            data = bytes(raw)
            data.decode("utf-8", errors="strict")
            headers = {key.lower(): value for key, value in response.getheaders()}
            if "application/json" in headers.get("content-type", ""):
                save_json(folder / "response.json", json.loads(data))
            return response.status, headers, data, folder
        except Exception as error:
            save_json(folder / "error.json", {"error": f"{type(error).__name__}: {error}"})
            raise
        finally:
            suffix = "sse" if "text/event-stream" in response.getheader("Content-Type", "") else "utf8"
            (folder / ("response." + suffix)).write_bytes(raw)
            save_json(folder / "timing.json", {"seconds": time.monotonic() - start, "bytes": len(raw)})
            response.close()
            connection.close()

    def health(self, log_name: str, timeout: float = 10) -> dict[str, Any]:
        # Polls go in one NDJSON file rather than thousands of small folders.
        connection = self.connect(timeout)
        record: dict[str, Any] = {"at": utc_now(), "method": "GET", "path": self.path("/health")}
        try:
            connection.request("GET", self.path("/health"), headers={"Connection": "close"})
            response = connection.getresponse()
            raw = response.read(MAX_RESPONSE_BYTES + 1)
            record.update(status=response.status, headers=response.getheaders(),
                          raw_utf8=raw.decode("utf-8", errors="replace"))
            require(response.status == 200, f"Health returned HTTP {response.status}")
            require(len(raw) <= MAX_RESPONSE_BYTES, "Health response is too large")
            value = json.loads(raw.decode("utf-8", errors="strict"))
            require(isinstance(value, dict), "Health must be a JSON object")
            return value
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            connection.close()
            with (self.artifacts / (log_name + ".ndjson")).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def wait_ready(self) -> dict[str, Any]:
        deadline = time.monotonic() + self.args.timeout
        last: Any = None
        while time.monotonic() < deadline:
            try:
                last = self.health("ready-health", min(10, max(.1, deadline - time.monotonic())))
                if last.get("ready") is True:
                    require(last.get("requests_in_flight") == 0, "Service is busy; use a dedicated idle service")
                    self.report["initial_health"] = last
                    return last
                if last.get("stopping") or last.get("phase") in ("failed", "stopped"):
                    raise RuntimeError(f"Service cannot become ready: {last}")
            except (OSError, http.client.HTTPException) as error:
                last = str(error)
            time.sleep(min(1, max(0, deadline - time.monotonic())))
        raise TimeoutError(f"Service did not become ready: {last}")

    def models(self) -> dict[str, Any]:
        status, _, raw, folder = self.exchange("models", "GET", "/v1/models")
        require(status == 200, f"Models returned HTTP {status}")
        value = json.loads(raw)
        rows = value.get("data", [])
        require(isinstance(rows, list) and len(rows) == 1, "Expected exactly one served model")
        self.model = rows[0].get("id", "")
        require(isinstance(self.model, str) and bool(self.model), "Missing model ID")
        self.report["model"] = self.model
        return {"model": self.model, "artifact": str(folder)}

    def payload(self, messages: list[dict[str, str]], stream: bool = False) -> dict[str, Any]:
        return {"model": self.model, "messages": messages, "max_tokens": 8,
                "temperature": 0, "stream": stream, "mtp_depth": 0}

    def chat(self, name: str, messages: list[dict[str, str]], expected: str,
             stream: bool = False, cache: str | None = None) -> dict[str, Any]:
        status, headers, raw, folder = self.exchange(name, "POST", "/v1/chat/completions",
                                                    self.payload(messages, stream))
        require(status == 200, f"Expected HTTP 200, received {status}: {raw[:1000]!r}")
        if stream:
            require("text/event-stream" in headers.get("content-type", ""), "Expected SSE Content-Type")
            parsed = parse_sse(raw)
            save_json(folder / "parsed-sse.json", parsed)
        else:
            require("application/json" in headers.get("content-type", ""), "Expected JSON Content-Type")
            value = json.loads(raw)
            require(value.get("object") == "chat.completion", "Unexpected completion object")
            choices = value.get("choices", [])
            require(len(choices) == 1 and choices[0].get("index") == 0, "Expected one completion choice")
            choice = choices[0]
            require(choice.get("finish_reason") in ("stop", "length"), "Invalid finish reason")
            message = choice.get("message", {})
            require(message.get("role") == "assistant", "Missing assistant role")
            parsed = {"text": message.get("content"), "usage": value.get("usage"),
                      "finish_reason": choice["finish_reason"], "model": value.get("model")}
        require(parsed["model"] == self.model, "Response names another model")
        require(isinstance(parsed["text"], str), "Response content is not text")
        prompt, cached = validate_usage(parsed["usage"])
        require(parsed["text"].strip() == expected,
                f"Expected exactly {expected!r}, received {parsed['text']!r}; artifact: {folder}")
        if cache == "full":
            require(cached == prompt, f"Exact repeat cached {cached}/{prompt} prompt tokens")
        elif cache == "partial":
            require(0 < cached < prompt, f"Changed user prompt must reuse a partial prefix, got {cached}/{prompt}")
        return {"text": parsed["text"], "expected": expected, "usage": parsed["usage"],
                "finish_reason": parsed["finish_reason"], "artifact": str(folder)}

    def reject(self, name: str, body: dict[str, Any]) -> dict[str, Any]:
        if name == "reject_context_overflow":
            capacity = self.report.get("initial_health", {}).get("capacity")
            require(type(capacity) is int and 0 < capacity <= 4096,
                    "max_tokens=4096 is only an overflow probe for capacity<=4096; request was not sent")
        status, _, raw, folder = self.exchange(name, "POST", "/v1/chat/completions", body)
        require(status == 400, f"Expected HTTP 400, received {status}: {raw[:1000]!r}")
        error = json.loads(raw).get("error")
        require(isinstance(error, dict) and bool(error.get("message")), "Missing structured error")
        return {"status": status, "error": error, "artifact": str(folder)}

    def cancel_prefill(self) -> dict[str, Any]:
        before = self.health("cancel-health")
        idle_deadline = time.monotonic() + self.args.timeout
        while before.get("requests_in_flight") != 0 and time.monotonic() < idle_deadline:
            time.sleep(.25)
            before = self.health("cancel-health", min(10, max(.1, idle_deadline - time.monotonic())))
        require(before.get("requests_in_flight") == 0, "Cancellation test requires an idle service")
        require(type(before.get("cancelled")) is int, "Health is missing its cancellation counter")
        messages = [{"role": "system", "content": NUMBERS_ONLY}, {"role": "user", "content":
                    "取消测试标识 " + uuid.uuid4().hex + "。\n"
                    + "Background text for a cancellation test. " * 64
                    + "\n17*3等于多少？只输出最终数字。"}]
        connection, transport, response, folder = self.begin(
            "cancel-prefill", "POST", "/v1/chat/completions", self.payload(messages, True))
        at_disconnect = None
        progress_error = None
        raw = b""
        try:
            require(response.status == 200, f"Cancellation request received HTTP {response.status}")
            require("text/event-stream" in response.getheader("Content-Type", ""), "Cancellation request is not SSE")
            # Capture the initial role event only: never wait for a content token.
            while not raw.endswith((b"\n\n", b"\r\n\r\n")):
                line = response.readline(65536)
                require(bool(line), "SSE closed before its initial event")
                raw += line
                require(len(raw) <= 65536, "Initial SSE event is too large")
            deadline = time.monotonic() + self.args.timeout
            baseline = None
            while time.monotonic() < deadline:
                value = self.health("cancel-health", min(10, max(.1, deadline - time.monotonic())))
                require(value.get("requests_in_flight") == 1, "Cancellation target is absent or other requests interfered")
                require(value.get("phase") == "prefill", f"Target left prefill before cancellation: {value}")
                completed = value.get("progressCompleted")
                require(type(completed) is int, "Health is missing prefill progressCompleted")
                if baseline is None:
                    baseline = completed
                if completed >= baseline + 2:
                    at_disconnect = value
                    break
                time.sleep(.25)
            require(at_disconnect is not None, "No two actual prefill steps observed before deadline")
        except Exception as error:
            progress_error = error
        finally:
            (folder / "partial-response.sse").write_bytes(raw)
            # A FIN is legal HTTP write-half-close and does not prove the reader
            # disappeared. Abort the full TCP connection after observed prefill.
            if transport is not None:
                try:
                    transport.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                except OSError:
                    pass
            response.close()
            connection.close()
            if transport is not None:
                transport.close()
        disconnected_at = time.monotonic()
        deadline = disconnected_at + self.args.timeout
        after = None
        while time.monotonic() < deadline:
            after = self.health("cancel-health", min(10, max(.1, deadline - time.monotonic())))
            if after.get("requests_in_flight") == 0:
                break
            time.sleep(.5)
        details = {"before": before, "at_disconnect": at_disconnect, "after": after,
                   "drain_seconds": time.monotonic() - disconnected_at, "artifact": str(folder)}
        save_json(folder / "cancellation.json", details)
        if progress_error is not None:
            raise progress_error
        require(after is not None and after.get("requests_in_flight") == 0, "Cancelled request did not drain")
        require(after.get("cancelled") == before["cancelled"] + 1, "Expected exactly one new cancellation")
        require(after.get("ready") is True, "Service was not ready after cancellation")
        return details

    def cancel_queued(self) -> dict[str, Any]:
        """Keep A running while three queued B requests are separately cancelled."""
        log_name = "queued-cancel-health"

        def wait_health(predicate, seconds: float, description: str, guard=None):
            deadline = time.monotonic() + seconds
            last = None
            while time.monotonic() < deadline:
                last = self.health(log_name, min(2, max(.001, deadline - time.monotonic())))
                if guard is not None:
                    guard(last)
                if predicate(last) and time.monotonic() <= deadline:
                    return last
                time.sleep(min(.05, max(0, deadline - time.monotonic())))
            raise TimeoutError(f"{description}; last health: {last}")

        before = wait_health(lambda value: value.get("requests_in_flight") == 0,
                             self.args.timeout, "Service did not become idle before queued cancellation")
        require(before.get("ready") is True, "Queued cancellation requires a ready service")
        require(before.get("max_pending_requests") == 2,
                "Queued cancellation requires health.max_pending_requests=2 (one active plus one queued)")
        for key in ("cancelled", "completed", "failed", "active_requests", "queued_requests"):
            require(type(before.get(key)) is int, f"Health is missing integer {key}")
        require(before["active_requests"] == before["queued_requests"] == 0,
                "Service reports active/queued jobs while idle")

        def require_active(value):
            require(value.get("ready") is True and value.get("phase") == "prefill"
                    and value.get("active_requests") == 1,
                    f"A must remain in prefill while queued B is cancelled: {value}")
            require(value.get("completed") == before["completed"]
                    and value.get("failed") == before["failed"],
                    "A/B unexpectedly completed or failed during queued cancellation")

        messages = [{"role": "system", "content": NUMBERS_ONLY}, {"role": "user", "content":
                    "排队取消测试标识 " + uuid.uuid4().hex + "。\n"
                    + "Background text for a cancellation test. " * 64
                    + "\n17*3等于多少？只输出最终数字。"}]
        connection, transport, response, folder = self.begin(
            "queued-cancel-active-a", "POST", "/v1/chat/completions", self.payload(messages, True))
        details: dict[str, Any] = {"before": before, "artifact": str(folder), "iterations": []}
        raw = b""
        queued = None
        failure = None
        try:
            require(response.status == 200, f"A received HTTP {response.status}")
            require("text/event-stream" in response.getheader("Content-Type", ""), "A did not start SSE")
            while not raw.endswith((b"\n\n", b"\r\n\r\n")):
                line = response.readline(65536)
                require(bool(line), "A closed before its initial SSE event")
                raw += line
                require(len(raw) <= 65536, "A initial SSE event is too large")
            initial = self.health(log_name)
            require_active(initial)
            require(type(initial.get("progressCompleted")) is int, "Missing A prefill progress")
            details["active_prefill"] = wait_health(
                lambda value: value.get("requests_in_flight") == 1 and value.get("queued_requests") == 0
                and value.get("progressCompleted", -1) >= initial["progressCompleted"] + 2,
                self.args.timeout, "A did not execute two new prefill tokens", require_active)

            for index in range(3):
                iteration: dict[str, Any] = {"index": index + 1}
                details["iterations"].append(iteration)
                b_messages = [{"role": "system", "content": NUMBERS_ONLY},
                              {"role": "user", "content": f"排队请求 {index + 1}。17*3等于多少？"}]
                queued = self.begin(f"queued-cancel-b-{index + 1}", "POST", "/v1/chat/completions",
                                    self.payload(b_messages, True), wait_for_headers=False)
                b_connection, b_transport, b_response, b_folder = queued
                iteration["artifact"] = str(b_folder)
                iteration["admitted"] = wait_health(
                    lambda value: value.get("requests_in_flight") == 2 and value.get("queued_requests") == 1,
                    min(3, self.args.timeout), "B did not occupy the one queued slot", require_active)
                require(iteration["admitted"].get("cancelled") == before["cancelled"] + index,
                        "Unexpected cancellation count before B disconnect")

                # A and B occupy both admission slots; C must fail before any
                # success/SSE headers or inference can start.
                status, _, c_raw, c_folder = self.exchange(
                    f"queued-cancel-overflow-c-{index + 1}", "POST", "/v1/chat/completions",
                    self.payload(b_messages), timeout=min(3, self.args.timeout))
                require(status == 429, f"C must be rejected with 429 while A+B are admitted, got {status}")
                error = json.loads(c_raw).get("error")
                require(isinstance(error, dict) and bool(error.get("message")), "C lacks a structured error")
                iteration["overflow"] = {"status": status, "error": error, "artifact": str(c_folder)}

                disconnected = time.monotonic()
                iteration["disconnect"] = self.abort_connection(b_connection, b_transport, b_response)
                queued = None
                save_json(b_folder / "disconnect.json", iteration["disconnect"])
                require(iteration["disconnect"]["linger_zero"], "Could not configure B's abortive TCP close")
                iteration["after"] = wait_health(
                    lambda value: value.get("requests_in_flight") == 1 and value.get("queued_requests") == 0
                    and value.get("cancelled") == before["cancelled"] + index + 1,
                    min(3, self.args.timeout) - (time.monotonic() - disconnected),
                    "Queued B did not release its slot and increment cancelled within 3 seconds", require_active)
                iteration["drain_seconds"] = time.monotonic() - disconnected
                require(iteration["drain_seconds"] <= 3, "Queued cancellation exceeded 3 seconds")
                save_json(b_folder / "queued-cancellation.json", iteration)
                save_json(folder / "queued-cancellation.json", details)
        except Exception as error:
            failure = error
            details["error"] = f"{type(error).__name__}: {error}"
        finally:
            # Also covers admission/protocol failures and user interruption.
            if queued is not None:
                b_connection, b_transport, b_response, b_folder = queued
                cleanup = self.abort_connection(b_connection, b_transport, b_response)
                save_json(b_folder / "cleanup-disconnect.json", cleanup)
            details["active_disconnect"] = self.abort_connection(connection, transport, response)
            (folder / "partial-response.sse").write_bytes(raw)
            save_json(folder / "queued-cancellation.json", details)

        disconnected = time.monotonic()
        try:
            details["after"] = wait_health(
                lambda value: value.get("requests_in_flight") == 0 and value.get("active_requests") == 0
                and value.get("queued_requests") == 0,
                self.args.timeout, "A/B did not drain after cleanup")
            details["active_drain_seconds"] = time.monotonic() - disconnected
            after = details["after"]
            require(after.get("ready") is True and after.get("phase") == "idle", "Service not ready/idle after A cancellation")
            if failure is None:
                require(details["active_disconnect"]["linger_zero"], "Could not configure A's abortive TCP close")
                require(after.get("cancelled") == before["cancelled"] + 4,
                        "Expected exactly four cancellations: three queued B requests and active A")
                require(after.get("completed") == before["completed"] and after.get("failed") == before["failed"],
                        "A/B unexpectedly completed or failed")
        except Exception as error:
            details["cleanup_error"] = f"{type(error).__name__}: {error}"
            if failure is None:
                failure = error
        finally:
            save_json(folder / "queued-cancellation.json", details)
        if failure is not None:
            raise failure
        return details

    def run(self) -> int:
        if self.check("health_ready", self.wait_ready) is None:
            return self.finish()
        if self.check("models", self.models) is None:
            return self.finish()
        arithmetic = [{"role": "system", "content": NUMBERS_ONLY},
                      {"role": "user", "content": "17*3等于多少？"}]
        first = self.check("cold_json_51", lambda: self.chat("cold-json", arithmetic, "51"))
        # Existing caches are not cleared: cold means the first request in this
        # invocation, and its actual cached_tokens remain visible in the report.
        self.check("same_prompt_warm_sse_51_full_cache", lambda: self.chat(
            "warm-sse", arithmetic, "51", stream=True, cache="full"))

        def secret(number: str, question: str):
            return [{"role": "system", "content":
                     f"本次会话的暗号是{number}。用户询问暗号时，只输出{number}，不要输出其他内容。"},
                    {"role": "user", "content": question}]

        q1, q2 = "请告诉我本次会话的暗号。", "这个会话的暗号是多少？"
        self.check("system_47_seed", lambda: self.chat("system-47-seed", secret("47", q1), "47"))
        self.check("system_47_changed_user_cache", lambda: self.chat(
            "system-47-reuse", secret("47", q2), "47", cache="partial"))
        self.check("different_system_83_isolation", lambda: self.chat("system-83", secret("83", q2), "83"))
        self.check("original_system_47_after_switch", lambda: self.chat("system-47-return", secret("47", q1), "47"))
        if first is not None:
            multi = arithmetic + [{"role": "assistant", "content": first["text"]},
                                  {"role": "user", "content": "把刚才的结果加2。只输出最终数字。"}]
            self.check("multiturn_53", lambda: self.chat("multiturn", multi, "53"))
        else:
            self.report["checks"].append({"name": "multiturn_53", "status": "skipped",
                                           "reason": "No validated assistant answer from cold_json_51"})
            self.write_report()
        for name, patch in (("reject_wrong_model", {"model": self.model + "-incorrect"}),
                            ("reject_unknown_field", {"unsupported_acceptance_field": True}),
                            ("reject_context_overflow", {"max_tokens": 4096}),
                            ("reject_mtp", {"mtp_depth": 2})):
            body = self.payload(arithmetic) | patch
            self.check(name, lambda name=name, body=body: self.reject(name, body))
        if self.args.skip_cancel:
            for name in ("cancel_during_prefill", "cancel_queued_while_active"):
                self.report["checks"].append({"name": name, "status": "skipped",
                                               "reason": "Explicit --skip-cancel"})
            self.write_report()
        else:
            self.check("cancel_during_prefill", self.cancel_prefill)
            self.check("cancel_queued_while_active", self.cancel_queued)
            self.check("short_request_after_cancel", lambda: self.chat("after-cancel", arithmetic, "51"))
        self.check("final_health", lambda: self.final_health())
        return self.finish()

    def final_health(self) -> dict[str, Any]:
        # Completed HTTP response can precede worker bookkeeping/reset briefly.
        deadline = time.monotonic() + self.args.timeout
        while time.monotonic() < deadline:
            value = self.health("final-health", min(10, max(.1, deadline - time.monotonic())))
            if value.get("requests_in_flight") == 0:
                require(value.get("ready") is True, "Service is not ready at completion")
                self.report["final_health"] = value
                return value
            time.sleep(.25)
        raise TimeoutError("Service did not become idle at completion")

    def finish(self) -> int:
        checks = self.report["checks"]
        self.report["finished_at"] = utc_now()
        self.report["counts"] = {key: sum(c["status"] == key for c in checks)
                                 for key in ("passed", "failed", "skipped")}
        self.report["passed"] = bool(checks) and all(c["status"] == "passed" for c in checks)
        self.report["selected_checks_passed"] = not any(c["status"] == "failed" for c in checks)
        self.write_report()
        print(json.dumps({"report": str(self.output), "passed": self.report["passed"],
                          "counts": self.report["counts"]}, ensure_ascii=False), flush=True)
        return 0 if self.report["selected_checks_passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:11236")
    parser.add_argument("--output", default="results/coreai-service/http-acceptance.json")
    parser.add_argument("--timeout", type=float, default=1800,
                        help="Per request/wait deadline in seconds (default: 1800)")
    parser.add_argument("--skip-cancel", action="store_true", help="Skip cancellation; full passed remains false")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    try:
        return Acceptance(args).run()
    except KeyboardInterrupt:
        print("Interrupted; the report retains completed checks.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
