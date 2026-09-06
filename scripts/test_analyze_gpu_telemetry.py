"""Pure offline contract tests; no model, MLX, HTTP or hardware access."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("analyze_gpu_telemetry", Path(__file__).with_name("analyze_gpu_telemetry.py"))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def event(phase, start, end, rep=None, **extra):
    return {"phase": phase, "start_ns": start, "end_ns": end, "repetition": rep,
            "input_tokens": 1 if phase != "load" else 0,
            "output_tokens": 1 if phase == "decode" else 0, "succeeded": True, **extra}


def generation():
    return {"telemetry": {"clock": M.CLOCK, "events": [
        event("load", 0, 100), event("prefill", 100, 200, 0), event("decode", 200, 300, 0),
        event("prefill", 400, 500, 1),
        event("decode", 500, 540, 1, step_index=0, ssd_wait_seconds=0.002, ssd_requested_row_bytes=2560),
        event("decode", 550, 590, 1, step_index=1, ssd_wait_seconds=0.001, ssd_requested_row_bytes=2560)],
        "request_windows": [
            {"repetition": 0, "start_ns": 100, "end_ns": 300, "prefill_end_ns": 200,
             "decode_start_ns": 200, "decode_end_ns": 300},
            {"repetition": 1, "start_ns": 400, "end_ns": 600, "prefill_end_ns": 500,
             "decode_start_ns": 500, "decode_end_ns": 600}]}}


def sample(start, end, ident=0, **values):
    return {"type": "sample", "sample_id": ident, "target_pid": 123, "start_ns": start, "end_ns": end, **values}


def hardware(*samples):
    return [{"type": "metadata", "clock": M.CLOCK}, *samples]


class IntervalTests(unittest.TestCase):
    def testHalfOpenBoundariesAndPoints(self):
        phases, _ = M.build_phases(generation())
        self.assertEqual(M.classify_interval(sample(500, 600), phases)["phase_id"], "decode:1")
        self.assertEqual(M.classify_interval(sample(500, 500), phases)["phase_id"], "decode:1")
        self.assertEqual(M.classify_interval(sample(600, 600), phases)["status"], "outside")
        self.assertEqual(M.classify_interval(sample(499, 501), phases)["status"], "partial")
        self.assertEqual(M.union_intervals([(0, 4), (2, 6), (6, 7), (9, 9)]), [(0, 7)])

    def testAcrossStepsWithinOnePhaseIsNotPartial(self):
        report = M.analyze(generation(), hardware(sample(520, 580, process_disk_read_bytes_delta=8192)))
        warm = report["warm_decode"]
        self.assertEqual(warm["output_tokens"], 2)
        self.assertAlmostEqual(warm["successful_step_seconds"], 80e-9)
        self.assertAlmostEqual(warm["phase_window_seconds"], 100e-9)
        counter = warm["disk_counters"]["process_disk_read_bytes_delta"]
        self.assertEqual(counter["sum"], 8192)
        self.assertAlmostEqual(counter["warm_window_coverage_fraction"], 0.6)
        self.assertEqual(warm["logical_requested_row_bytes"], 5120)
        self.assertEqual(warm["residual_ssd_wait_seconds"], 0.003)

    def testCrossPhaseBytesAndHistogramsAreNotProrated(self):
        hist = {"kind": "bandwidth_residency_histogram", "source": "IOReport", "estimated": True,
                "unit": "events", "channels": [{"id": "pmp0", "name": "bandwidth", "state_names": ["low", "high"], "residency_delta_raw": [7, 3]}]}
        records = hardware(sample(450, 550, 1, process_disk_read_bytes_delta=100, system_disk_read_bytes_delta=999, ioreport=hist),
                           sample(550, 600, 2, process_disk_read_bytes_delta=0, system_disk_read_bytes_delta=200, ioreport=hist))
        result = M.analyze(generation(), records)
        warm = result["warm_decode"]
        self.assertEqual(warm["disk_counters"]["process_disk_read_bytes_delta"]["sum"], 0)
        self.assertEqual(warm["disk_counters"]["system_disk_read_bytes_delta"]["sum"], 200)
        self.assertEqual(warm["disk_counters"]["process_disk_read_bytes_delta"]["partial_overlap_ns"], 50)
        self.assertEqual(len(warm["pmp_histogram_trend"]), 1)
        self.assertEqual(warm["pmp_histogram_trend"][0]["bins"], [7, 3])
        self.assertEqual(result["hardware"]["raw_records"][1]["process_disk_read_bytes_delta"], 100)
        self.assertIsNone(warm["physical_dram_bytes"])
        self.assertIsNone(warm["physical_dram_bandwidth_gbps"])

    def testMissingIsNotZeroAndFailedLeavesDoNotContribute(self):
        gen = generation()
        gen["telemetry"]["events"][-1]["succeeded"] = False
        report = M.analyze(gen, hardware(sample(500, 600, process_disk_read_bytes_delta=None)))
        warm = report["warm_decode"]
        self.assertEqual(warm["output_tokens"], 1)
        self.assertEqual(warm["logical_requested_row_bytes"], 2560)
        self.assertIsNone(warm["disk_counters"]["process_disk_read_bytes_delta"]["sum"])
        phase = next(p for p in report["phases"] if p["id"] == "decode:1")
        self.assertEqual(phase["failed_or_unknown_events"], 1)

    def testClockMismatchIsUnalignedNotGuessed(self):
        records = hardware(sample(500, 600, process_disk_read_bytes_delta=123))
        records[0]["clock"] = "unix_epoch_nanoseconds"
        report = M.analyze(generation(), records)
        self.assertEqual(report["hardware"]["alignment"], "unknown")
        self.assertEqual(report["hardware"]["assignments"][0]["status"], "unaligned")
        self.assertIsNone(report["warm_decode"]["disk_counters"]["process_disk_read_bytes_delta"]["sum"])

    def testOverlapsAndInvalidEnvelopesAreRejected(self):
        gen = generation()
        gen["telemetry"]["events"][-1]["start_ns"] = 530
        with self.assertRaisesRegex(ValueError, "Overlapping generation"):
            M.analyze(gen)
        gen = generation()
        gen["telemetry"]["request_windows"][-1]["decode_start_ns"] = 499
        with self.assertRaisesRegex(ValueError, "crosses"):
            M.analyze(gen)
        with self.assertRaisesRegex(ValueError, "Overlapping hardware"):
            M.analyze(generation(), hardware(sample(500, 560), sample(550, 600)))

    def testLegacyNoClockDoesNotInventWarmWindows(self):
        report = M.analyze({"trials": [{"repetition": 1, "decode_step_seconds": [0.1]}]})
        self.assertEqual(report["phases"], [])
        self.assertIsNone(report["warm_decode"]["tokens_per_successful_step_second"])
        self.assertTrue(report["warnings"])

    def testExplicitWarmSelectionIsRecorded(self):
        report = M.analyze(generation(), warm_repetitions=[0])
        self.assertEqual(report["warm_decode"]["phase_ids"], ["decode:0"])
        self.assertEqual(report["warm_decode"]["selection"], "explicit repetitions")

    def testProcessIdentityMismatchDoesNotAttributeProcessBytes(self):
        gen = generation()
        gen["telemetry"]["target_pid"] = 999
        report = M.analyze(gen, hardware(sample(500, 600, process_disk_read_bytes_delta=123, system_disk_read_bytes_delta=456)))
        warm = report["warm_decode"]["disk_counters"]
        self.assertIsNone(warm["process_disk_read_bytes_delta"]["sum"])
        self.assertEqual(warm["system_disk_read_bytes_delta"]["sum"], 456)

    def testActualRequestSchemaUsesRequestEndForDecode(self):
        gen = generation()
        for request in gen["telemetry"]["request_windows"]:
            del request["decode_end_ns"]
        self.assertEqual(M.build_phases(gen)[0][-1]["end_ns"], 600)
        self.assertEqual(M.analyze({"telemetry": None})["phases"], [])

    def testCounterOwnReadWindowCanCrossWhenSamplerWindowDoesNot(self):
        baseline = {"type": "baseline", "start_ns": 510, "end_ns": 510,
                    "process": {"read_start_ns": 495, "read_end_ns": 498}}
        records = hardware(baseline, sample(510, 590, process_disk_read_bytes_delta=50,
                                           process={"read_start_ns": 580, "read_end_ns": 585},
                                           ioreport={"kind": "bandwidth_residency_histogram", "channels": None}))
        report = M.analyze(generation(), records)
        self.assertEqual(report["hardware"]["assignments"][0]["status"], "contained")
        process = report["hardware"]["source_assignments"][0]
        self.assertEqual(process["status"], "partial")
        self.assertEqual(process["start_ns"], 495)
        self.assertIsNone(report["warm_decode"]["disk_counters"]["process_disk_read_bytes_delta"]["sum"])


class InstrumentTests(unittest.TestCase):
    def report(self):
        return {"source": "Instruments Metal System Trace", "clock": {"kind": "trace_relative_ns", "absolute_anchor_ns": None, "alignment_status": "unknown"},
                "counters": [{"id": 1, "name": "GPU memory bandwidth", "unit": "GB/s", "scope": "GPU/system interface; may include SLC"}],
                "samples": [{"counter_id": 1, "start_ns": 0, "end_ns": 50, "value": 321}]}

    def testRelativeClockWithoutVerifiedAnchorPreservesRawOnly(self):
        source = self.report()
        result = M.analyze(generation(), instruments=source)["instruments"]
        self.assertEqual(result["alignment"], "unknown")
        self.assertEqual(result["raw_report"], source)
        self.assertEqual(result["assignments"][0]["status"], "unaligned")
        self.assertEqual(result["phases"]["decode:1"]["counter_samples"], [])

    def testVerifiedAnchorAssignsRawGaugeWithoutDRAMConversion(self):
        source = self.report()
        source["clock"].update(absolute_anchor_ns=500, alignment_status="verified")
        result = M.analyze(generation(), instruments=source)
        instrument = result["instruments"]
        self.assertEqual(instrument["assignments"][0]["phase_id"], "decode:1")
        stats = instrument["phases"]["decode:1"]["counter_statistics"]["1"]
        self.assertEqual(stats["unweighted_sample_mean"], 321)
        self.assertEqual(stats["definition"]["unit"], "GB/s")
        self.assertIsNone(result["warm_decode"]["physical_dram_bandwidth_gbps"])
        source["clock"]["alignment_status"] = "approximate"
        self.assertEqual(M.analyze(generation(), instruments=source)["instruments"]["alignment"], "unknown")

    def testCrossPhaseInstrumentNotAssigned(self):
        source = self.report()
        source["clock"].update(absolute_anchor_ns=480, alignment_status="verified")
        result = M.analyze(generation(), instruments=source)["instruments"]
        self.assertEqual(result["assignments"][0]["status"], "partial")
        self.assertEqual(result["phases"]["decode:1"]["counter_statistics"], {})

    def testUncertainAnchorDoesNotAttributeBoundarySample(self):
        source = self.report()
        source["clock"].update(absolute_anchor_ns=500, alignment_status="verified", uncertainty_ns=10)
        result = M.analyze(generation(), instruments=source)["instruments"]
        self.assertEqual(result["assignments"][0]["status"], "alignment_boundary_ambiguous")
        self.assertEqual(result["phases"]["decode:1"]["counter_statistics"], {})

    def testCounterGroupIsPartOfIdentity(self):
        source = self.report()
        source["clock"] = M.CLOCK
        source["counters"] = [{"id": 1, "group_index": 0, "unit": "%"}, {"id": 1, "group_index": 1, "unit": "GB/s"}]
        source["samples"] = [{"counter_id": 1, "group_index": 1, "start_ns": 520, "end_ns": 550, "value": 4}]
        result = M.analyze(generation(), instruments=source)["instruments"]
        self.assertEqual(result["phases"]["decode:1"]["counter_statistics"]["1:1"]["definition"]["unit"], "GB/s")


class GPUTimelineTests(unittest.TestCase):
    def report(self, intervals=(), complete=True):
        report = {"source": "Apple Instruments Metal System Trace", "clock": M.CLOCK,
                  "target_pid": 123, "gpu_intervals": list(intervals), "command_buffer_submissions": [],
                  "status": "target_gpu_intervals_collected" if intervals else "no_target_intervals"}
        if complete:
            report["coverage"] = {"start_ns": 0, "end_ns": 1000, "complete": True, "source": "verified trace bounds"}
        return report

    def active(self, start, end, pid=123):
        return {"start_ns": start, "end_ns": end, "pid": pid, "state": "Active"}

    def analyze(self, report):
        gen = generation()
        gen["telemetry"]["target_pid"] = 123
        return M.analyze(gen, gpu_intervals=report)

    def testOverlappingNestedGPUWorkIsUnioned(self):
        source = self.report([self.active(510, 550), self.active(520, 540), self.active(545, 580)])
        source["command_buffer_submissions"] = [self.active(500, 600)]
        result = self.analyze(source)
        phase = result["gpu_timeline"]["phases"]["decode:1"]
        self.assertAlmostEqual(phase["gpu_active_seconds"], 70e-9)
        self.assertAlmostEqual(phase["gpu_active_fraction"], 0.7)
        self.assertAlmostEqual(phase["largest_idle_gap_seconds"], 20e-9)
        self.assertAlmostEqual(phase["cpu_command_buffer_submission_union_seconds"], 100e-9)

    def testNoTargetIsNullWithoutCoverageButZeroWithCompleteCoverage(self):
        unknown = self.analyze(self.report(complete=False))["gpu_timeline"]["phases"]["decode:1"]
        self.assertIsNone(unknown["gpu_active_fraction"])
        self.assertIsNone(unknown["observed_active_union_seconds"])
        known = self.analyze(self.report())["gpu_timeline"]["phases"]["decode:1"]
        self.assertEqual(known["gpu_active_fraction"], 0)
        self.assertAlmostEqual(known["largest_idle_gap_seconds"], 100e-9)

    def testPartialGPUIntervalIsNotClippedIntoPhase(self):
        phase = self.analyze(self.report([self.active(490, 550)]))["gpu_timeline"]["phases"]["decode:1"]
        self.assertEqual(phase["partial_active_records"], 1)
        self.assertIsNone(phase["gpu_active_seconds"])
        self.assertIsNone(phase["largest_idle_gap_seconds"])

    def testWrongPIDAndCPUOnlySubmissionsAreNotGPUTime(self):
        source = self.report([self.active(500, 600, pid=999)], complete=False)
        source["command_buffer_submissions"] = [self.active(500, 600)]
        result = self.analyze(source)["gpu_timeline"]
        self.assertEqual(result["assignments"]["gpu_intervals"][0]["status"], "pid_mismatch")
        self.assertIsNone(result["phases"]["decode:1"]["gpu_active_seconds"])

    def testUnknownClockDoesNotProduceUtilization(self):
        source = self.report([self.active(510, 550)])
        source["clock"] = {"kind": "trace_relative_ns", "alignment_status": "unknown"}
        phase = self.analyze(source)["gpu_timeline"]["phases"]["decode:1"]
        self.assertIsNone(phase["gpu_active_seconds"])

    def testKnownTraceWindowWithRunIssueIsNotComplete(self):
        source = self.report([self.active(510, 550)])
        source["coverage"].update(complete=False, errors=["Data stream: Time Mapping"])
        phase = self.analyze(source)["gpu_timeline"]["phases"]["decode:1"]
        self.assertEqual(phase["trace_window_coverage_fraction"], 1)
        self.assertIsNone(phase["gpu_active_fraction"])
        self.assertAlmostEqual(phase["observed_active_union_seconds"], 40e-9)

    def testTOCDurationRoundingIsIncludedInCoverageBoundary(self):
        source = self.report()
        source["coverage"].update(start_ns=499, end_ns=601, duration_rounding_uncertainty_ns=2)
        phase = self.analyze(source)["gpu_timeline"]["phases"]["decode:1"]
        self.assertIsNone(phase["gpu_active_fraction"])
        self.assertEqual(phase["coverage_boundary_uncertainty_ns"], 2)

    def testMissingGPUExportTableCannotEstablishZero(self):
        source = self.report()
        source["coverage"]["target_interval_table_available"] = False
        phase = self.analyze(source)["gpu_timeline"]["phases"]["decode:1"]
        self.assertIsNone(phase["gpu_active_fraction"])


class ProcessResourceTests(unittest.TestCase):
    def testCPUDeltaUsesOneCoreFractionAndMemoryIsSampledPeak(self):
        def process(start, end, user, system, footprint):
            return {"read_start_ns": start, "read_end_ns": end,
                    "rusage": {"start_abstime": 12, "cpu_user_time_ns_cumulative": user,
                               "cpu_system_time_ns_cumulative": system,
                               "physical_footprint_bytes": footprint, "resident_bytes": footprint // 2}}
        baseline = {"type": "baseline", "target_pid": 123, "start_ns": 501, "end_ns": 501,
                    "process": process(500, 501, 100, 50, 2000)}
        records = hardware(baseline, sample(501, 600, process=process(599, 600, 300, 100, 3000)))
        report = M.analyze(generation(), records)
        warm = report["warm_decode"]
        self.assertEqual(warm["process_cpu"]["one_core_fraction"], 2.5)
        self.assertEqual(warm["process_sampled_memory_peaks_bytes"]["physical_footprint_bytes"], 3000)

    def testCounterResetAndReplacementPIDAreUnavailable(self):
        records = hardware({"type": "baseline", "target_pid": 123, "process": {"rusage": {"start_abstime": 12, "cpu_user_time_ns_cumulative": 100, "cpu_system_time_ns_cumulative": 50}}},
                           sample(500, 600, process={"rusage": {"start_abstime": 13, "cpu_user_time_ns_cumulative": 110, "cpu_system_time_ns_cumulative": 60}}))
        self.assertIsNone(M.analyze(generation(), records)["warm_decode"]["process_cpu"]["one_core_fraction"])


class CLITests(unittest.TestCase):
    def testActualPhaseJSONLCanBeUsedOnFailedRun(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "phases.jsonl"
            records = [{"type": "metadata", "clock": M.CLOCK, "inference_succeeded": False}]
            records += [{"type": "interval", **e} for e in generation()["telemetry"]["events"]]
            path.write_text("\n".join(json.dumps(r) for r in records))
            telemetry = M.load_phase_file(path)
            phases, _ = M.build_phases({"telemetry": telemetry})
            self.assertEqual(phases[-1]["end_ns"], 590)
            self.assertFalse(telemetry["inference_succeeded"])

    def testFilesProvenanceAndStrictJSON(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            gen, hw, out = (directory / name for name in ("generation.json", "hardware.jsonl", "report.json"))
            gen.write_text(json.dumps(generation()))
            hw.write_text("\n".join(json.dumps(r) for r in hardware(sample(500, 600, process_disk_read_bytes_delta=4))))
            M.main(["--generation", str(gen), "--hardware", str(hw), "--output", str(out)])
            report = json.loads(out.read_text())
            self.assertEqual(report["warm_decode"]["disk_counters"]["process_disk_read_bytes_delta"]["sum"], 4)
            self.assertEqual(len(report["input_files"]["generation"]["sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
