"""Offline tests only. No Instruments launch, GPU workload, or model load."""
import importlib.util
import json
from pathlib import Path
import plistlib
import sqlite3
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("gpu_trace", Path(__file__).with_name("gpu_trace.py"))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def table(columns, rows):
    schema = "<schema>" + "".join(f"<col><mnemonic>{c}</mnemonic></col>" for c in columns) + "</schema>"
    return "<trace-query-result><node>" + schema + rows + "</node></trace-query-result>"


class TraceTests(unittest.TestCase):
    def test_shared_nodes_and_nested_pid(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "test.xml"
            p.write_text(table(["process", "timestamp"],
                '<row><process id="1" fmt="runner (42)"><pid>42</pid><session>x</session></process><time id="2">123</time></row>'
                '<row><process ref="1"/><time ref="2"/></row>'))
            rows = M.read_rows(p)
            self.assertEqual(rows[0][0], rows[1][0])
            self.assertEqual(rows[1][0]["process"][0], "42")

    def test_dangling_reference_fails(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "test.xml"
            p.write_text(table(["x"], '<row><time ref="missing"/></row>'))
            with self.assertRaises(ValueError):
                M.read_rows(p)

    def test_exact_time_conversion(self):
        clock = M.time_mapping([({"timebase-info": ["125", "3"], "mabs-epoch": "15474755391589", "update-time": "0"}, {})])
        self.assertEqual(clock["absolute_anchor_ns"], 644781474649541)
        self.assertEqual(M.absolute_time(120847042, clock), 644781595496583)
        self.assertEqual(clock["uncertainty_ns"], 1)

    def test_unknown_time_is_never_wall_clock_estimate(self):
        clock = M.time_mapping([])
        self.assertIsNone(M.absolute_time(100, clock))
        self.assertEqual(clock["alignment_status"], "unknown")

    def test_target_filter_excludes_desktop_and_keeps_raw_state(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "metal-gpu-intervals.xml"
            p.write_text(table(["start", "duration", "process", "state"],
                '<row><time>10</time><duration>20</duration><process fmt="runner (42)"><pid>42</pid></process><state>Active</state></row>'
                '<row><time>10</time><duration>30</duration><process fmt="desktop (99)"><pid>99</pid></process><state>Active</state></row>'))
            clock = M.time_mapping([])
            result = M.gpu_intervals(d, clock, 42)
            self.assertEqual(result["interval_count"], 1)
            self.assertEqual(result["gpu_intervals"][0]["end_ns"], 30)
            self.assertNotIn("start_absolute_ns", result["gpu_intervals"][0])
            self.assertEqual(M.gpu_intervals(d, clock, None)["interval_count"], 0)

    def test_empty_report_keeps_dram_unknown(self):
        with tempfile.TemporaryDirectory() as d:
            result = M.build_report(d, "unknown.trace")
            self.assertEqual(result["bandwidth_sample_count"], 0)
            self.assertFalse(result["gpu_read_write_bandwidth_available"])
            self.assertIsNone(result["physical_dram_bytes"])

    def test_template_copy_does_not_mutate_shared_zero(self):
        with tempfile.TemporaryDirectory() as d:
            source, out = Path(d) / "source", Path(d) / "out"
            objects = ["$null", 0, "counterprofile", "counterprofileinternal", "counterscounterprofile",
                       {"NS.keys": [plistlib.UID(2), plistlib.UID(3), plistlib.UID(4)], "NS.objects": [plistlib.UID(1)] * 3}]
            source.write_bytes(plistlib.dumps({"$objects": objects}, fmt=plistlib.FMT_BINARY))
            original = source.read_bytes()
            M.make_template(source, out, 4)
            data = plistlib.loads(out.read_bytes())
            self.assertEqual(data["$objects"][1], 0)
            self.assertEqual(source.read_bytes(), original)
            with self.assertRaises(FileExistsError):
                M.make_template(source, out, 4)

    def test_coverage_uses_trace_window_not_observed_event_span(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            trace = directory / "test.trace"
            (trace / "Trace1.run").mkdir(parents=True)
            (trace / "form.template").write_bytes(b"fixture")
            connection = sqlite3.connect(trace / "Trace1.run/RunIssues.storedata")
            connection.execute("CREATE TABLE ZISSUE (ZTYPE, ZSUBTYPE, ZMESSAGE, ZCOUNT, ZRELATIVETIMESTAMP)")
            connection.commit()
            (directory / "toc.xml").write_text('<trace-toc><run number="1"><info><summary><duration>5.782251</duration><end-reason>Time limit reached</end-reason></summary></info></run></trace-toc>')
            (directory / "metal-gpu-intervals.xml").write_text(table([], ''))
            coverage = M.trace_coverage(trace, directory, M.time_mapping([]))
            self.assertEqual(coverage["end_ns"], 5782251000)
            self.assertTrue(coverage["complete"])
            self.assertIsNone(coverage["start_absolute_ns"])
            connection.execute("INSERT INTO ZISSUE VALUES (1,0,'dropped events',1,0)")
            connection.commit()
            connection.close()
            self.assertFalse(M.trace_coverage(trace, directory, M.time_mapping([]))["complete"])

    def test_successful_recovery_does_not_make_source_complete(self):
        with tempfile.TemporaryDirectory() as d:
            coverage = {"complete": True, "errors": []}
            M.mark_incomplete_source(coverage, Path(d) / "source.trace")
            self.assertFalse(coverage["complete"])
            self.assertEqual(coverage["source_status"], "recovered_from_incomplete_source_trace")
            self.assertTrue(coverage["errors"])
            self.assertFalse(coverage["incomplete_source_trace"]["finalized_archive_marker_present"])


if __name__ == "__main__":
    unittest.main()
