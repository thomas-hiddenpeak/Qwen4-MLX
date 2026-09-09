"""Additional capacity-policy checks for the unchanged controlled churn workload.

Saved-file audit only: no server/process management, requests, GPU, or fixtures.
Existing churn/independent audits remain responsible for output and cache safety.
"""
import hashlib
import json
import os
from pathlib import Path


POLICY = {"kv_append_mode": "capacity256", "scope": "autoregressive_decode_only",
          "prefill_uses_capacity": False, "mtp_kv_append_mode": "reference"}
FIELDS = ("kv_append_mode", "kv_capacity_token_steps", "kv_capacity_workspace_fallbacks",
          "kv_capacity_workspace_peak_bytes")
MAX_LINE = 1_048_576
MAX_REQUESTS = 100_000


def require_capacity_health(health):
    """Call after the existing health_contract, without weakening its checks."""
    policy = health.get("kv_append_policy")
    if policy != POLICY or policy.get("prefill_uses_capacity") is not False:
        raise RuntimeError("Owned server does not publish the expected capacity256 AR-only policy")
    return health


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value):
    return type(value) is int and value >= 0


def object_without_duplicates(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def rows(path, report, mixed=False):
    """Bound each line and hash the exact stable post-shutdown input bytes."""
    digest, consumed = hashlib.sha256(), 0
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        while True:
            line = stream.readline(MAX_LINE + 1)
            if not line:
                break
            require(len(line) <= MAX_LINE and line.endswith(b"\n"), f"Oversized/incomplete line in {path.name}")
            digest.update(line)
            consumed += len(line)
            if mixed and not line.lstrip().startswith(b"{"):
                continue  # Legacy text summaries are not structured terminals.
            value = json.loads(line, object_pairs_hook=object_without_duplicates)
            require(isinstance(value, dict), f"Expected JSON object in {path.name}")
            yield value
        after = os.fstat(stream.fileno())
    require((before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
            (after.st_size, after.st_mtime_ns, after.st_ctime_ns) and consumed == before.st_size,
            f"Input changed during final audit: {path.name}")
    report["inputs"][str(path.resolve())] = {"bytes": consumed, "sha256": digest.hexdigest()}


def validate_capacity_terminals(events_path, server_path, pid, stop_requested=lambda: False):
    report = {"schema": "qwen-http-capacity-terminal-audit-v1", "complete": False, "passed": False,
              "server_pid": pid, "inputs": {}, "errors": [],
              "limits": {"line_bytes": MAX_LINE, "request_ids": MAX_REQUESTS},
              "counts": {"oracle": 0, "request": 0, "cancel": 0, "completed": 0,
                         "cancelled": 0, "completed_with_decode": 0, "model_terminals": 0},
              "capacity_token_steps": 0, "workspace_fallbacks": 0, "workspace_peak_bytes": 0,
              "notes": ["Additional policy/step audit; retain the original churn and independent output/cache/lifecycle audit.",
                        "Completed mode and counters are runner phase results. Health configuration alone is not execution evidence.",
                        "Cancelled requests without a result require four explicit null fields; their partial execution is unknown.",
                        "For a first-token EOS, decoded steps and workspace peak may be zero. The run must contain successful capacity decode steps.",
                        "Workspace peak is a logical reservation, not RSS or an allocation observation."]}
    clients, terminals = {}, set()
    try:
        require(type(pid) is int and pid > 0, "Invalid owned server PID")
        for value in rows(Path(events_path), report):
            if stop_requested():
                raise InterruptedError("Terminal audit interrupted")
            kind = value.get("event")
            if kind not in ("oracle", "request", "cancel"):
                continue
            identity = value.get("request_id")
            require(isinstance(identity, str) and 0 < len(identity) <= 256, "Invalid client request ID")
            require(identity not in clients, f"Duplicate client ID: {identity}")
            require(len(clients) < MAX_REQUESTS, "Client request ID bound exhausted")
            require(value.get("passed") is True, f"Client did not pass: {identity}")
            if kind == "cancel":
                require(value.get("client_abort_requested") is True, f"Missing client cancel intent: {identity}")
                completion = None
            else:
                usage = value.get("usage")
                completion = usage.get("completion_tokens") if isinstance(usage, dict) else None
                require(integer(completion) and completion >= 1, f"Invalid client completion count: {identity}")
            # No prompts, output content, or full client rows retained in memory.
            clients[identity] = (kind, completion)
            report["counts"][kind] += 1
        for value in rows(Path(server_path), report, mixed=True):
            if stop_requested():
                raise InterruptedError("Terminal audit interrupted")
            if value.get("schema") != "qwen-http-lifecycle-v1" or value.get("event") != "model_terminal":
                continue
            identity = value.get("request_id")
            require(isinstance(identity, str) and identity in clients, f"Orphan/invalid model terminal: {identity}")
            require(identity not in terminals, f"Duplicate JSON model terminal: {identity}")
            require(len(terminals) < MAX_REQUESTS, "Model terminal ID bound exhausted")
            terminals.add(identity)
            report["counts"]["model_terminals"] += 1
            require(type(value.get("pid")) is int and value["pid"] == pid, f"Terminal PID mismatch: {identity}")
            require(type(value.get("mtp_depth")) is int and value["mtp_depth"] == 0, f"Unexpected MTP terminal: {identity}")
            require(all(field in value for field in FIELDS), f"Capacity terminal field absent: {identity}")
            kind, client_completion = clients[identity]
            if kind == "cancel":
                require(value.get("model_kind") == "cancelled", f"Cancellation not confirmed: {identity}")
                require(all(value[field] is None for field in FIELDS), f"Cancelled execution counters must be null: {identity}")
                report["counts"]["cancelled"] += 1
                continue
            require(value.get("model_kind") == "completed", f"Successful client model did not complete: {identity}")
            require(value["kv_append_mode"] == "capacity256", f"Completed runner policy mismatch: {identity}")
            completion, decoded = value.get("completion_tokens"), value.get("decoded_tokens")
            steps, fallbacks, peak = (value[field] for field in FIELDS[1:])
            require(integer(completion) and completion == client_completion and integer(decoded) and
                    decoded == completion - 1, f"AR completion/decoded count mismatch: {identity}")
            require(integer(steps) and steps == decoded, f"Successful capacity step mismatch: {identity}")
            require(integer(fallbacks), f"Invalid workspace fallback count: {identity}")
            report["workspace_fallbacks"] += fallbacks
            require(fallbacks == 0, f"Unexpected workspace fallback: {identity}: {fallbacks}")
            require(integer(peak) and (peak > 0 if decoded else peak == 0), f"Workspace peak mismatch: {identity}")
            report["counts"]["completed"] += 1
            report["counts"]["completed_with_decode"] += int(decoded > 0)
            report["capacity_token_steps"] += steps
            report["workspace_peak_bytes"] = max(report["workspace_peak_bytes"], peak)
        require(clients.keys() == terminals, "Client/model terminal ID sets differ; at least one terminal is missing")
        require(report["counts"]["oracle"] > 0 and report["counts"]["request"] > 0 and
                report["counts"]["cancel"] > 0, "Missing oracle/success/cancellation workload coverage")
        require(report["capacity_token_steps"] > 0, "No successfully evaluated capacity decode steps")
        report["complete"] = report["passed"] = True
    except (OSError, ValueError, TypeError) as error:
        report["errors"].append(f"{type(error).__name__}: {error}"[:1500])
    return report
