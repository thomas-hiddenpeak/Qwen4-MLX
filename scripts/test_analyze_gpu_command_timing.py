import unittest
from analyze_gpu_command_timing import analyze, union_ns


class CommandTimingTests(unittest.TestCase):
    def test_union_merges_overlapping_buffers(self):
        self.assertEqual(union_ns([(10,20),(15,25),(30,40)]),25)

    def fixture(self):
        g={'provenance':{'process_id':123},'gpu_command_timing':{'enabled':True,'finished':True,'hook_version':'v1',
            'steps':[{'phase':'decode','repetition':1,'step':0,'start_ns':100,'forward_end_ns':120,'evaluation_end_ns':200}]}}
        def r(i,a,b,commit):
            return {'sequence':i,'status':4,'gpu_timestamp_valid':True,'host_clock_bracket_valid':True,
                'gpu_start_ns':a,'gpu_end_ns':b,'cpu_commit_ns':commit,'buffer_ops':2}
        c={'process_id':123,'version':'v1','complete':True,'errors':[],'dropped_buffers':0,'pending_buffers':0,
            'recorded_buffers':2,'completed_buffers':2,'records':[r(0,110,150,105),r(1,160,190,145)]}
        return g,c

    def test_span_and_gap_do_not_double_count_cpu_time(self):
        a=analyze(*self.fixture())
        s=a['steps'][0]
        self.assertAlmostEqual(s['command_buffer_span_coverage_fraction'],0.7)
        self.assertAlmostEqual(s['outside_observed_buffer_spans_ms'],30/1e6)
        self.assertEqual([g['next_buffer_already_committed_at_gap_start'] for g in s['observed_gaps_before_next_buffer']],[False,True])

    def test_incomplete_suppresses_coverage_and_gap_claims(self):
        g,c=self.fixture();c['dropped_buffers']=1
        a=analyze(g,c)
        self.assertFalse(a['complete'])
        self.assertIsNone(a['steps'][0]['command_buffer_span_coverage_fraction'])
        self.assertIsNone(a['steps'][0]['observed_gaps_before_next_buffer'])
        self.assertAlmostEqual(a['steps'][0]['observed_gpu_buffer_span_union_ms'],70/1e6)

    def test_refuses_cross_process_timestamps(self):
        g,c=self.fixture();c['process_id']=124
        with self.assertRaises(ValueError):analyze(g,c)

    def test_mtp_round_retains_span_without_inventing_graph_boundary(self):
        g,c=self.fixture()
        g['gpu_command_timing']['steps'][0]['graph_boundary_available']=False
        a=analyze(g,c)
        s=a['steps'][0]
        self.assertTrue(a['complete'])
        self.assertIsNone(s['forward_wall_ms'])
        self.assertIsNone(s['evaluation_and_readback_wall_ms'])
        self.assertIsNone(a['decode_groups'][0]['median_forward_wall_ms'])
        self.assertIsNone(a['decode_groups'][0]['median_evaluation_and_readback_wall_ms'])
        self.assertAlmostEqual(s['command_buffer_span_coverage_fraction'],0.7)

    def test_graph_boundary_flag_must_be_boolean(self):
        g,c=self.fixture()
        g['gpu_command_timing']['steps'][0]['graph_boundary_available']='false'
        with self.assertRaises(ValueError):analyze(g,c)


if __name__ == '__main__':unittest.main()
