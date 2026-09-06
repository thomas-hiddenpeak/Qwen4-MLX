"""Bounded cleanup of one subprocess session created by this controller.

Children must retain the session/process group; daemonizing is unsupported.
No PID from a report, shared parent group, or name-based process search is used.
"""
from dataclasses import asdict, dataclass
import math
import os
import signal
import subprocess
import time


@dataclass(frozen=True)
class GroupCleanup:
    pid: int
    pgid: int
    cleaned: bool
    term_sent: bool
    kill_sent: bool
    leader_exit_code: object
    elapsed_seconds: float
    error: object = None

    def report(self):
        return asdict(self)


class OwnedProcessGroup:
    """One Popen owner; finish is terminal and safe to call repeatedly."""

    @classmethod
    def start(cls, command, **kwargs):
        if "start_new_session" in kwargs or "preexec_fn" in kwargs or "process_group" in kwargs:
            raise ValueError("OwnedProcessGroup exclusively sets the child session")
        process = subprocess.Popen(command, start_new_session=True, **kwargs)
        # setsid has completed before Popen returns. Do not look up a PID later:
        # a fast leader may have exited while its descendants still own this group.
        return cls(process)

    def __init__(self, process):
        self.process = process
        self.pid = process.pid
        self.pgid = process.pid
        self._cleanup = None
        self._gone = False

    def _exists(self):
        self.process.poll()  # Reap our direct child, including an exited leader.
        if self._gone:
            return False
        if self.pgid <= 1 or self.pgid == os.getpgrp():
            raise RuntimeError("Refuse to signal a shared or invalid process group")
        try:
            os.killpg(self.pgid, 0)
            return True
        except ProcessLookupError:
            self._gone = True
            return False

    def _signal(self, value):
        if not self._exists():
            return False
        try:
            os.killpg(self.pgid, value)
            return True
        except ProcessLookupError:
            self._gone = True
            return False

    def _wait_empty(self, seconds):
        deadline = time.monotonic() + seconds
        while self._exists():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(0.05, remaining))
        return True

    def finish(self, grace_seconds=45, kill_seconds=10):
        if self._cleanup is not None:
            return self._cleanup
        if not all(type(x) in (int, float) and math.isfinite(x) and x >= 0
                   for x in (grace_seconds, kill_seconds)):
            raise ValueError("Cleanup deadlines must be finite nonnegative seconds")
        started = time.monotonic()
        term_sent = kill_sent = cleaned = False
        error = None
        try:
            if self._exists():
                term_sent = self._signal(signal.SIGTERM)
                cleaned = self._wait_empty(grace_seconds)
                if not cleaned:
                    kill_sent = self._signal(signal.SIGKILL)
                    cleaned = self._wait_empty(kill_seconds)
                    if not cleaned:
                        error = "Owned process group remains after cleanup deadline"
            else:
                cleaned = True
        except (OSError, RuntimeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        self._cleanup = GroupCleanup(self.pid, self.pgid, cleaned, term_sent, kill_sent,
                                     self.process.poll(), time.monotonic() - started, error)
        # A failed cleanup is also terminal: never reuse a stale pgid after the
        # controller records recovery_blocked and hands the problem to its owner.
        return self._cleanup
