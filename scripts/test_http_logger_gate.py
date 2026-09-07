"""CPU schema/gate controls. These do not establish any Swift or live coverage."""
import copy
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


edge = load("test_http_server_edges", ROOT / "scripts/test_http_server_edges.py")
terminal = load("test_http_terminal_logs", ROOT / "scripts/test_http_terminal_logs.py")
pipe_gate = load("test_http_logger_pipe", ROOT / "scripts/test_http_logger_pipe.py")


def state():
    return {"logging": {"accepting": True, "writer_exited": False, "max_bytes": 65536, "max_events": 128,
            "max_event_bytes": 4096, "buffered_bytes": 0, "buffered_events": 0, "queued_events": 0,
            "in_flight_bytes": 0, "enqueued_events": 3, "written_events": 3, "dropped_events": 0,
            "dropped_bytes": 0, "write_failures": 0, "last_write_errno": None}}


class LoggerGateTests(unittest.TestCase):
    def testCompleteLoggingRequiresKnownZeroLossCounters(self):
        terminal.require_complete_logging(state())
        for bad in ({}, {"logging": None}, {"logging": {}}):
            with self.assertRaises(ValueError):
                terminal.require_complete_logging(bad)
        for patch in ({"dropped_events": 1, "dropped_bytes": 100}, {"write_failures": 1, "last_write_errno": 32},
                      {"accepting": False}, {"dropped_events": True}, {"last_write_errno": 5}):
            bad = state()
            bad["logging"].update(patch)
            with self.assertRaises(ValueError):
                terminal.require_complete_logging(bad)

    def testPausedReaderWithoutApplicationQuotaCannotPassPipeGate(self):
        logs = state()["logging"]
        self.assertFalse(pipe_gate.backlog_at_limit(logs))
        logs.update(in_flight_bytes=512, buffered_bytes=512, buffered_events=1)
        self.assertFalse(pipe_gate.backlog_at_limit(logs))
        logs.update(dropped_events=1)
        self.assertFalse(pipe_gate.backlog_at_limit(logs))
        logs.update(buffered_bytes=65000, buffered_events=100, queued_events=99)
        self.assertTrue(pipe_gate.backlog_at_limit(logs))
        pipe_gate.logging_state({"logging": logs})
        # Observed drops invalidate complete lifecycle attribution, even though
        # this is the positive precondition for the separate liveness gate.
        with self.assertRaises(ValueError):
            terminal.require_complete_logging({"logging": logs})

    def testPipeGateRejectsFailuresAndOutOfBoundOrBooleanCounters(self):
        for patch in ({"write_failures": 1, "last_write_errno": 32}, {"buffered_bytes": 65537},
                      {"buffered_events": 129}, {"in_flight_bytes": 4097, "buffered_bytes": 4097},
                      {"queued_events": True}, {"writer_exited": True}):
            bad = state()
            bad["logging"].update(patch)
            with self.assertRaises(ValueError):
                pipe_gate.logging_state(bad)

    def testOwnedPipeByteInspectionDoesNotConsumeAndPostExitDrainIsBounded(self):
        read_fd, write_fd = os.pipe()
        try:
            os.write(write_fd, b"owned cpu fixture\n")
            first = pipe_gate.unread_bytes(read_fd)
            self.assertEqual(first, len(b"owned cpu fixture\n"))
            self.assertEqual(pipe_gate.unread_bytes(read_fd), first)
            os.close(write_fd)
            write_fd = None
            self.assertEqual(pipe_gate.drain_after_exit(read_fd), b"owned cpu fixture\n")
        finally:
            os.close(read_fd)
            if write_fd is not None:
                os.close(write_fd)

    def testLegacyAttributionKeepsCompleteLinesAndRejectsUnknownOrLostRecords(self):
        class CPUHarness:
            value = state()
            def alive(self):
                pass
            def health(self):
                return 200, self.value
        harness = CPUHarness()
        # Every temporary fixture belongs to this test process.
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "fixture.log"
            line = "HTTP request id=fixture terminal=cancelled stage=decode scheduler_elapsed_seconds=1"
            log.write_text(line + "\n" + line[:30])
            self.assertEqual(edge.HTTPHarness.terminal_log_lines(harness, log, "fixture"), [line])
            edge.require_lossless_logging(harness.value)
            for bad in ({}, {"logging": {"accepting": True, "dropped_events": 1, "dropped_bytes": 1, "write_failures": 0}}):
                harness.value = bad
                with self.assertRaises(ValueError):
                    edge.HTTPHarness.terminal_log_lines(harness, log, "fixture")
                with self.assertRaises(ValueError):
                    edge.require_lossless_logging(harness.value)


# Re-run the six existing protocol/terminal controls against this same parser.
legacy_path = ROOT / "scripts/test_http_terminal_log_validator.py"
legacy_tests = load("existing_terminal_log_controls", legacy_path)


def load_tests(loader, tests, pattern):
    tests.addTests(loader.loadTestsFromModule(legacy_tests))
    return tests


if __name__ == "__main__":
    unittest.main()
