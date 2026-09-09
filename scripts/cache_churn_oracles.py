"""Pure, bounded audit of cold HTTP text oracles (no token-ID claim).

``audit_oracle_discrimination(mode, {profile: content}, expected_profiles=names,
reported_hashes={profile: sha256})`` returns a JSON-compatible report. The two
keyword arguments are optional; callers that know the fixture inventory or
saved hashes should supply them so missing fixtures and stale evidence fail.
Counter witnesses require the first output integer to match the system header
and every full content hash to differ. A trailing partial next number is valid
under the existing 16-token output budget; this is not a full arithmetic audit.
Legacy output equality is recorded, never promoted into witness evidence and
never rejected just for being equal. Invalid input shapes raise ValueError;
bad evidence produces ``passed=False`` with bounded issues.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
from itertools import islice
import re

FIXTURE_MODES = ("legacy", "counter-witness")
ORACLE_DISCRIMINATION_SCHEMA = "qwen-cache-churn-oracle-discrimination-v1"
MAX_PROFILES = 40
MAX_CONTENT_BYTES = 65_536


def counter_witness_start(profile: str) -> int:
    """Stable disjoint starts for the CLI's at-most 32 long and 8 short cases."""
    match = re.fullmatch(r"(long|short)_([0-9]{2})", profile) if isinstance(profile, str) else None
    if match is None:
        raise ValueError("Counter-witness profile must be long_00..31 or short_00..07")
    kind, index = match.group(1), int(match.group(2))
    if index >= (32 if kind == "long" else 8):
        raise ValueError("Counter-witness profile index exceeds fixture bounds")
    return (index + (1 if kind == "long" else 65)) * 1000


def _names(values):
    names = tuple(islice(iter(values), MAX_PROFILES + 1))
    if (len(names) > MAX_PROFILES or any(not isinstance(name, str) or not 1 <= len(name) <= 64
                                         for name in names) or len(set(names)) != len(names)):
        raise ValueError("Oracle profile inventory must contain at most 40 unique bounded names")
    return set(names)


def audit_oracle_discrimination(mode, cold_oracles, *, expected_profiles=None, reported_hashes=None):
    """Recompute evidence from actual strings; never trust a stored passed flag."""
    if mode not in FIXTURE_MODES:
        raise ValueError("Unknown cache churn fixture mode")
    if not isinstance(cold_oracles, Mapping):
        raise ValueError("Cold oracles must map profile names to content strings")
    actual = _names(cold_oracles)
    expected = actual if expected_profiles is None else _names(expected_profiles)
    if reported_hashes is not None and not isinstance(reported_hashes, Mapping):
        raise ValueError("Reported hashes must map profile names to SHA-256 strings")
    reported = None if reported_hashes is None else _names(reported_hashes)
    required = mode == "counter-witness"
    issues = []
    if not expected:
        issues.append("empty_profile_inventory")
    for name in sorted(expected - actual):
        issues.append(f"missing_oracle:{name}")
    for name in sorted(actual - expected):
        issues.append(f"unexpected_oracle:{name}")
    if reported is not None and reported != actual:
        issues.append("reported_hash_inventory_mismatch")
    profiles, hashes = {}, []
    for name in sorted(actual):
        start, starts, content_hash = None, None, None
        if required:
            try:
                start = counter_witness_start(name)
            except ValueError:
                issues.append(f"invalid_counter_profile:{name}")
        content = cold_oracles[name]
        if not isinstance(content, str) or len(content.encode("utf-8")) > MAX_CONTENT_BYTES:
            issues.append(f"invalid_content:{name}")
        else:
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            hashes.append(content_hash)
            if required:
                match = re.match(r"^\s*([0-9]+)\b", content)
                starts = start is not None and match is not None and match.group(1) == str(start)
                if not starts:
                    issues.append(f"wrong_or_missing_start:{name}")
            if reported_hashes is not None and reported_hashes.get(name) != content_hash:
                issues.append(f"content_hash_mismatch:{name}")
        profiles[name] = {"expected_start": start, "content_sha256": content_hash,
                          "starts_with_expected": starts}
    unique = len(set(hashes))
    if required and unique != len(hashes):
        issues.append("duplicate_oracle_content")
    # Legacy uniqueness is just an observation, not proof of prefix dependence.
    discriminating = not issues and len(actual) >= 2 and unique == len(actual)
    return {"schema": ORACLE_DISCRIMINATION_SCHEMA, "fixture_mode": mode,
            "required": required, "passed": not issues, "discriminating": discriminating,
            "profile_count": len(actual), "unique_content_hashes": unique,
            "profiles": profiles, "issues": issues}
