"""CPU subprocess controls; no model, controller, reference service, or GPU."""
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from owned_process_group import OwnedProcessGroup


class OwnedProcessGroupTests(unittest.TestCase):
    def start(self, source, *args):
        group = OwnedProcessGroup.start([sys.executable, "-u", "-c", source, *map(str, args)],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       stdin=subprocess.DEVNULL, text=True)

        def cleanup():
            result = group.finish(grace_seconds=1, kill_seconds=3)
            group.process.stdout.close()
            group.process.stderr.close()
            self.assertTrue(result.cleaned, result.report())

        self.addCleanup(cleanup)
        ready, _, _ = select.select([group.process.stdout], [], [], 5)
        self.assertTrue(ready, "CPU child did not become ready")
        line = group.process.stdout.readline()
        self.assertTrue(line, "CPU child exited before ready")
        return group, line.strip()

    def test_exited_leader_does_not_hide_live_descendant(self):
        with tempfile.TemporaryDirectory() as temporary:
            ready = Path(temporary) / "ready"
            stopped = Path(temporary) / "stopped"
            descendant = """
import signal, sys, time
from pathlib import Path
def stop(_number, _frame):
    Path(sys.argv[2]).write_text('clean')
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
Path(sys.argv[1]).write_text('ready')
time.sleep(30)
"""
            leader = """
import json, os, subprocess, sys, time
from pathlib import Path
child = subprocess.Popen([sys.executable, '-u', '-c', sys.argv[1], sys.argv[2], sys.argv[3]],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
deadline = time.monotonic()+5
while not Path(sys.argv[2]).exists() and time.monotonic()<deadline: time.sleep(.01)
assert Path(sys.argv[2]).exists()
print(json.dumps({'pid':child.pid, 'pgid':os.getpgid(child.pid)}), flush=True)
"""
            group, line = self.start(leader, descendant, ready, stopped)
            child = json.loads(line)
            self.assertEqual(child["pgid"], group.pgid)
            self.assertEqual(group.process.wait(timeout=3), 0)
            result = group.finish(grace_seconds=3, kill_seconds=2)
            self.assertTrue(result.cleaned, result.report())
            self.assertTrue(result.term_sent)
            self.assertFalse(result.kill_sent)
            self.assertEqual(stopped.read_text(), "clean")

    def test_term_allows_graceful_cleanup_before_force(self):
        group, _ = self.start("""
import signal, time
def stop(_number, _frame):
    time.sleep(.35)
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
print('ready', flush=True)
time.sleep(30)
""")
        result = group.finish(grace_seconds=2, kill_seconds=1)
        self.assertTrue(result.cleaned, result.report())
        self.assertFalse(result.kill_sent)
        self.assertEqual(result.leader_exit_code, 0)
        self.assertGreaterEqual(result.elapsed_seconds, .3)

    def test_force_only_owned_group_and_leave_control_alive(self):
        source = """
import signal, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
print('ready', flush=True)
time.sleep(30)
"""
        target, _ = self.start(source)
        control, _ = self.start(source)
        self.assertNotEqual(target.pgid, control.pgid)
        self.assertNotEqual(target.pgid, os.getpgrp())
        result = target.finish(grace_seconds=.2, kill_seconds=3)
        self.assertTrue(result.cleaned, result.report())
        self.assertTrue(result.kill_sent)
        self.assertIsNone(control.process.poll())
        os.killpg(control.pgid, 0)

    def test_repeat_finish_never_signals_again(self):
        group, _ = self.start("import time; print('ready', flush=True); time.sleep(30)")
        first = group.finish(grace_seconds=2, kill_seconds=1)
        self.assertTrue(first.cleaned)
        with mock.patch("owned_process_group.os.killpg", side_effect=AssertionError("unexpected signal")):
            self.assertIs(group.finish(), first)


if __name__ == "__main__":
    unittest.main()
