"""CPU-only synthetic controls; no saved model, route or result fixture required."""
import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.dont_write_bytecode = True
import analyze_mtp_expert_overlap as m


def fixture():
    golden = {'trials': [{'prompt_tokens': [7] * 11057, 'generated_token_ids': list(range(128))}]}
    stats = dict(rounds=55, draftedTokens=109, acceptedDraftTokens=72, verifiedTokens=164,
        replayedTokens=0, emittedTokens=127, acceptanceHistogram=[0,38,17,0,0],
        prefillHistoryTokens=1024, historyStartPosition=10033)
    cost = dict(countersConsistent=True, acceptanceHistogramConsistent=True, componentsFitDecodeWindow=True,
        committedDecodeTokens=127, decodeSteps=55, speculativeRounds=55, targetOnlyDecodeSteps=0)
    trial = dict(repetition=0, prompt_tokens=golden['trials'][0]['prompt_tokens'], generated_token_ids=list(range(128)),
        finish_reason='length', final_state_offset=11184, qsa_active_layers=12, mtp_depth=2,
        mtp_verification='batchedScalarLinear', mtp_draft_history_limit=1024, decode_mode='reference',
        gdn_gemv_mode='reference', prefill_accumulation='reference', prefill_chunk=416,
        ssd_prefetch='nextChunk', ssd_workers=1, sampling='greedy', wired_memory={'policy':'disabled'},
        mtp_statistics=stats, mtp_cost_summary=cost, decode_steps=55, decode_step_seconds=[1.0]*55,
        decode_tokens_per_second=127/55, phase_metrics={'decode_seconds':55.0}, prefill_chunk_seconds=[1.0]*28)
    normal = dict(requested_repetitions=1, max_tokens=128, context_limit=16384, mtp_order=[2],
        mtp_verification_order=['batchedScalarLinear'], mtp_enabled=True, mtp_weights_loaded=True,
        experimental_decode_async_every_layers=0, experimental_decode_async_submissions=0,
        prefill_evaluate_every_layers=4, verification_evaluate_every_layers=4,
        prefill_attention_mode='reference', scheduler_environment={}, profiler={'mode':'disabled','stages':[],'droppedRecords':0},
        gpu_command_timing={'enabled':False}, trials=[trial], model_directory='/synthetic-model',
        provenance={'process_id':1,'executable_sha256':'synthetic-executable',
                    'model_metadata_sha256':{'model.safetensors.index.json':'synthetic-index'}}, mtp_expert_routes={'enabled':False})
    diagnostic = copy.deepcopy(normal); diagnostic['provenance']['process_id']=2
    routes=[];position=11057
    for index,accepted in enumerate([2]*17+[1]*38):
        s=2 if index==54 else 3
        for layer in range(48):
            ids = [list(range(10)) for _ in range(s)] if layer%2==0 else [list(range(j*10,(j+1)*10)) for j in range(s)]
            routes.append(dict(repetition=0,phase='verification',position=position,tokenCount=s,layer=layer,expertIDs=ids))
        position+=1+accepted
    assert position==11184
    diagnostic['mtp_expert_routes']=dict(enabled=True,finished=True,maximumRecords=8192,droppedRecords=0,
        readbackMilliseconds=1.0,**m.GEOMETRY,records=routes)
    headers=[]
    for layer in range(48):
        for projection in ('gate_proj','up_proj','down_proj'):
            output,input_size=(2560,640) if projection=='down_proj' else (640,2560)
            for part in ('weight','scales','biases'):
                packed=part=='weight';shape=[512,output,input_size//(8 if packed else 64)]
                headers.append(dict(layer=layer,projection=projection,part=part,shape=shape,dtype='U32' if packed else 'BF16',
                    bytes_per_expert=output*shape[2]*(4 if packed else 2)))
    geometry=dict(complete=True,model_directory='/synthetic-model',index_sha256='synthetic-index',tensor_count=432,
        layer_count=48,routed_bytes_per_expert_per_layer=2764800,gate_up_bytes_per_expert_per_layer=1843200,records=headers)
    return normal,diagnostic,golden,geometry


class ExpertOverlapControls(unittest.TestCase):
    def setUp(self):
        self.normal,self.diagnostic,self.golden,self.geometry=fixture()

    def reject(self, which, mutate, message):
        mutate(self.normal if which=='normal' else self.diagnostic)
        with self.assertRaisesRegex(ValueError,message):
            m.analyze(self.normal,self.diagnostic,self.golden)

    def test_exact_affine_geometry_bytes(self):
        self.assertEqual(m.PER_PROJECTION_BYTES,921600)
        self.assertEqual(m.PER_EXPERT_BYTES,2764800)

    def test_disjoint_s2(self):
        r=m.overlap([list(range(10)),list(range(10,20))])
        self.assertEqual((r['unique_experts'],r['removable_logical_bytes_upper_bound']),(20,0))

    def test_identical_s3_pair_counts_do_not_equal_redundancy(self):
        r=m.overlap([list(range(10))]*3)
        self.assertEqual((r['unique_experts'],r['redundant_assignments']),(10,20))
        self.assertEqual(sum(p['intersection'] for p in r['pair_intersections']),30)
        self.assertEqual(r['fanout_histogram'],{'1':0,'2':0,'3':10})
        self.assertEqual(r['gate_up_only_removable_logical_bytes_upper_bound'],20*1843200)

    def test_mixed_s3_fanout(self):
        r=m.overlap([list(range(10)),list(range(5,15)),list(range(8,18))])
        self.assertEqual((r['unique_experts'],r['redundant_assignments']),(18,12))
        self.assertEqual(r['fanout_histogram'],{'1':8,'2':8,'3':2})
        self.assertEqual([p['intersection'] for p in r['pair_intersections']],[5,2,7])

    def test_slot_permutation_preserves_identity_statistics(self):
        ids=[list(range(10)),list(range(5,15)),list(range(8,18))]
        self.assertEqual(m.overlap(ids),m.overlap([row[::-1] for row in ids]))

    def test_duplicate_within_row_rejected(self):
        with self.assertRaises(ValueError): m.overlap([[0]*10,list(range(10))])

    def test_out_of_range_rejected(self):
        with self.assertRaises(ValueError): m.overlap([list(range(10)),list(range(503,513))])

    def test_boolean_id_rejected(self):
        with self.assertRaises(ValueError): m.overlap([[True]+list(range(1,10)),list(range(10))])

    def test_missing_shape_unknown(self):
        self.assertEqual(m.summarize([]),{'observed':False,'records':0,'totals':None})

    def test_full_synthetic_request_with_minimal_disabled_normal(self):
        r=m.analyze(self.normal,self.diagnostic,self.golden)
        self.assertTrue(r['complete'])
        self.assertEqual(len(r['records']),2640)
        self.assertEqual([s['records'] for s in r['by_shape']],[48,2592])
        self.assertEqual(r['layer_ranking'][0]['layer'],0)

    def test_dropped_capture_rejected(self):
        self.reject('diagnostic',lambda d:d['mtp_expert_routes'].__setitem__('droppedRecords',1),'Incomplete/dropped')

    def test_missing_layer_rejected(self):
        self.reject('diagnostic',lambda d:d['mtp_expert_routes']['records'].pop(0),'Missing/duplicate')

    def test_wrong_verification_phase_rejected(self):
        self.reject('diagnostic',lambda d:d['mtp_expert_routes']['records'][0].__setitem__('phase','decode'),'Invalid record')

    def test_wrong_full_output_rejected(self):
        self.reject('diagnostic',lambda d:d['trials'][0]['generated_token_ids'].__setitem__(50,900),'full IDs')

    def test_normal_capture_rejected(self):
        self.reject('normal',lambda d:d.__setitem__('mtp_expert_routes',self.diagnostic['mtp_expert_routes']),'Normal unexpectedly')

    def test_disabled_with_records_rejected(self):
        self.reject('normal',lambda d:d['mtp_expert_routes'].__setitem__('records',[self.diagnostic['mtp_expert_routes']['records'][0]]),'Normal unexpectedly')

    def test_disabled_with_drops_rejected(self):
        self.reject('normal',lambda d:d['mtp_expert_routes'].__setitem__('droppedRecords',1),'Normal unexpectedly')

    def test_wrong_logical_geometry_rejected(self):
        self.reject('diagnostic',lambda d:d['mtp_expert_routes'].__setitem__('bits',8),'geometry differs')

    def test_all_432_synthetic_headers(self):
        result=m.analyze(self.normal,self.diagnostic,self.golden,self.geometry)
        self.assertEqual(result['actual_header_geometry_check']['tensors'],432)

    def test_header_byte_mismatch_rejected(self):
        self.geometry['records'][0]['bytes_per_expert']+=1
        with self.assertRaisesRegex(ValueError,'byte calculation'):
            m.analyze(self.normal,self.diagnostic,self.golden,self.geometry)

    def run_cli(self, bad_json=False):
        with tempfile.TemporaryDirectory(prefix='mtp-expert-overlap-') as temp:
            td=Path(temp)
            for name,data in [('normal',self.normal),('diagnostic',self.diagnostic),('golden',self.golden),('weight-geometry',self.geometry)]:
                (td/(name+'.json')).write_text(json.dumps(data))
            if bad_json: (td/'diagnostic.json').write_text('{"duplicate":1,"duplicate":2}')
            args=[x for key in ('normal','diagnostic','golden','weight-geometry') for x in ('--'+key,str(td/(key+'.json')))]
            with contextlib.redirect_stdout(io.StringIO()): result=m.main(args+['--output',str(td/'analysis.json')])
            return result,json.loads((td/'analysis.json').read_text())

    def test_actual_cli_valid_synthetic(self):
        code,result=self.run_cli()
        self.assertEqual(code,0);self.assertTrue(result['complete'])
        self.assertTrue(result['actual_header_geometry_check']['validated'])

    def test_actual_cli_duplicate_json_fails(self):
        code,result=self.run_cli(bad_json=True)
        self.assertEqual(code,1);self.assertFalse(result['complete'])
        self.assertIn('Duplicate JSON key',result['errors'][0])


if __name__=='__main__':
    unittest.main()
