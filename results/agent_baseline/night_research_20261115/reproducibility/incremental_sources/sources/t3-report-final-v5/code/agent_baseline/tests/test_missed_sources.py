"""Synthetic arrays/fake workers only; no model, API, Slurm or dataset execution."""
import json
import tempfile
import threading
from io import BytesIO
import unittest
from urllib.error import HTTPError
from pathlib import Path
from unittest.mock import patch

import numpy as np

from agent_baseline.analysis.missed_sources import (
    change_candidates, union, measure, summarize_measures, oracle_add, f1,
    split_components, raw_sources, pack, unpack, save_npz, load_npz, inclusion,
    quotas, ensure_settings, candidate_records, score_answers, S1, SOURCES,
    ledger_unit, rows_for, specs, inventory, output_path, statistics_group)
from agent_baseline.analysis.missed_sources_ask import (
    run_requests, AskStopped, existing_verifier, Budget, RequestGuard)
from agent_baseline.analysis.missed_sources_gpu import feature_sources, generate_one
from agent_baseline.workers.missed_sources_worker import visual_state
from agent_baseline.analysis.io_retry import write_json, save_image, retry_io


def box(x0,y0,x1,y1,shape=(12,16)):
    m=np.zeros(shape,bool);m[y0:y1,x0:x1]=True;return m


class MetricsTests(unittest.TestCase):
    def test_direction_subtraction_and_matching_before_subtraction(self):
        old=box(1,1,4,4);new=box(7,2,11,5);stable=box(7,8,10,11)
        start=box(7,2,8,5)
        a,b=[old,stable],[new,stable]
        np.testing.assert_array_equal(union(change_candidates(a,b,start,'appeared'),start.shape),new & ~start)
        np.testing.assert_array_equal(union(change_candidates(a,b,start,'disappeared'),start.shape),old)
        np.testing.assert_array_equal(union(change_candidates(a,b,start,'any'),start.shape),(old|new)&~start)
        self.assertEqual(change_candidates([stable],[stable],stable,'any'),[])
        self.assertEqual(change_candidates([],[],start,'any'),[])
        # Presence must be evaluated on raw instances, not clipped residuals.
        large=box(0,0,8,8);small=box(0,0,4,4)
        clipped=large & ~small
        self.assertEqual(change_candidates([small],[large],small,'appeared'),[])
        self.assertTrue(clipped.any())

    def test_directional_overlap_is_per_instance_not_opposite_union(self):
        big=box(0,0,10,10);a=box(0,0,2,10);b=box(4,0,6,10)
        empty=np.zeros_like(big)
        # Each overlap .2 < .25 even though the opposite union covers .4.
        found=change_candidates([a,b],[big],empty,'appeared')
        np.testing.assert_array_equal(union(found,big.shape),big)

    def test_recall_precision_residual_and_original_components(self):
        a=box(1,1,5,3);b=box(9,6,13,8)
        truth=a|b;start=box(1,1,3,3)
        s1=[box(3,1,5,2)]  # half of first missed component
        masks=[box(9,6,11,8),box(0,10,2,12)]  # half second + FP
        result=measure(start,truth,masks,s1)
        self.assertEqual(result['missing_pixels'],12)
        self.assertEqual(result['missing_components'],2)
        self.assertAlmostEqual(result['pixel_recall'],4/12)
        self.assertEqual(result['component_recall'],.5)
        self.assertEqual(result['precision'],.5)
        self.assertAlmostEqual(result['marginal_pixel_recall'],4/12)
        self.assertAlmostEqual(result['residual_pixel_recall'],4/10)
        self.assertEqual(result['marginal_component_recall'],.5)
        self.assertEqual(result['residual_component_recall'],.5)
        # Same source overlapping S1 contributes only uncovered TP.
        overlap=measure(start,truth,[truth],s1)
        self.assertEqual(overlap['novel_pixels'],10)
        self.assertEqual(overlap['residual_pixel_recall'],1)

    def test_eight_connectivity_and_null_denominators(self):
        m=np.eye(3,dtype=bool)
        self.assertEqual(len(split_components(m)),1)
        r=measure(m,m,[],[])
        self.assertIsNone(r['pixel_recall'])
        self.assertIsNone(r['precision'])
        self.assertIsNone(r['component_recall'])
        self.assertIsNone(r['residual_pixel_recall'])
        self.assertEqual(f1([0,0,0]),100)

    def test_oracle_keeps_whole_majority_components_and_start(self):
        truth=box(1,1,4,3)|box(10,8,13,10)
        start=box(14,0,16,1) # FP in start must stay
        good=box(1,1,5,3) # TP6 FP2
        tie=box(10,8,16,10) # TP6 FP6, reject
        result=oracle_add(start,[good,tie],truth)
        self.assertEqual(result['kept'],1)
        self.assertEqual(result['counts'],[6,4,6])
        self.assertAlmostEqual(result['f1'],200*6/22)
        empty=oracle_add(start,[],truth)
        self.assertEqual(empty['counts'],[0,2,12])

    def test_aggregate_concentration_and_new_criterion_no_five_point_gate(self):
        start=np.zeros((12,16),bool);truth=box(2,2,4,4)
        rows=[dict(image=f'i{i}',**measure(start,truth,[truth] if i<3 else [],[])) for i in range(5)]
        result=summarize_measures(rows)
        self.assertEqual(result['contributing_images'],3)
        self.assertAlmostEqual(result['top2_share'],2/3)
        self.assertEqual(len(result['per_image']),5)
        self.assertIn('可收录',inclusion(result,.001))
        self.assertIn('不收录',inclusion(result,0))
        self.assertIn('待验证器',inclusion(result,None))
        result['contributing_images']=2
        self.assertIn('单列',inclusion(result,1))

    def test_packed_empty_roundtrip_and_raw_threshold_strict(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            path=Path(tmp)/'x.npz'
            save_npz(path,masks=pack([], (3,5)),shape=[3,5])
            z=load_npz(path)
            self.assertEqual(unpack(z['masks'],tuple(z['shape'])).shape,(0,3,5))
        start=np.zeros((12,16),bool);mask=box(1,1,3,3)
        raw={s:pack([mask],start.shape) for s in SOURCES}
        raw.update(shape=start.shape,S3_scores=np.array([.5]),S4_scores=np.array([.6]))
        found=raw_sources(raw,start,{'S3':.5,'S4':.5})
        self.assertEqual(found['S3'],[])
        self.assertEqual(len(found['S4']),1)

    def test_quotas_favor_precision_and_cap_observed_density(self):
        def m(p,n):
            return dict(precision=p,candidates=n*10,per_image=[dict(candidates=n) for _ in range(10)])
        result=quotas({'S1_02':m(.9,10),'S1_03':m(1,30),'S2':m(.3,10),'S3':m(.2,1)},12)
        self.assertEqual(result['suggested']['S1_03'],0)
        self.assertGreater(result['suggested']['S1_02'],result['suggested']['S2'])
        self.assertLessEqual(result['suggested']['S3'],1)
        self.assertEqual(sum(result['suggested'].values()),12)


class FakeGpuTests(unittest.TestCase):
    def test_feature_gate_difference_and_large_object_area(self):
        start=np.zeros((8,8),bool);start[0,0]=1
        proposal=np.ones((8,8),bool)
        # Gain on left, loss on right; no ground truth involved.
        def dynamic(op,time):
            if op=='segment_everything':
                return [proposal]
            if op=='extract_features':
                a=np.zeros((2,2,2),np.float32);a[:,:,0 if time=='t1' else 1]=1;return a
            building=np.zeros((8,8),np.float32)
            building[:,4:] = 1 if time=='t1' else 0
            building[:,:4] = 1 if time=='t2' else 0
            return np.stack([1-building,building])
        new,scores=feature_sources(dynamic,start,'appeared')
        expected=np.zeros_like(start);expected[:,:4]=1;expected &= ~start
        np.testing.assert_array_equal(union(new['S4'],start.shape),expected)
        np.testing.assert_array_equal(union(new['S3'],start.shape),expected)
        self.assertEqual(scores['S4'],[1.])
        old,_=feature_sources(dynamic,start,'disappeared')
        self.assertTrue(union(old['S4'],start.shape)[:,4:].all())
        self.assertFalse(union(old['S4'],start.shape)[:,:4].any())

    def test_visual_same_date_uses_single_image_and_opposite_uses_mosaic(self):
        calls=[]
        class Processor:
            def set_image(self,im):
                calls.append(('image',np.asarray(im).shape));return {}
            def add_geometric_prompt(self,**kwargs):
                calls.append(('box',kwargs['box']));return kwargs['state']
        a=np.zeros((8,12,3),np.uint8)
        visual_state(Processor(),a,a,[0,0,6,4],True)
        self.assertEqual(calls[0],('image',(8,12,3)))
        self.assertEqual(calls[1][1],[.25,.25,.5,.5])
        calls.clear()
        visual_state(Processor(),a,a,[0,0,6,4],False)
        self.assertEqual(calls[0],('image',(8,24,3)))


class AskTests(unittest.TestCase):
    def test_inflight_answer_is_saved_after_peer_stop(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            barrier=threading.Barrier(2)
            calls=[]
            def backend(record,guard):
                calls.append(record['uid'])
                barrier.wait(timeout=10)
                if record['uid']=='a:0':
                    guard.trip('HTTP 403')
                guard.charge(.01)
                return dict(answer='yes',cost_cny=.01)
            with self.assertRaises(AskStopped):
                run_requests([dict(uid=f'a:{i}') for i in range(4)],backend,tmp,1,concurrency=2)
            saved=json.loads((Path(tmp)/'answers.json').read_text())
            self.assertEqual(set(calls),{'a:0','a:1'})
            self.assertEqual(set(saved),{'a:1'})

    def test_existing_api_stops_on_first_429_403_and_balance_error(self):
        from agent_baseline.ask_eval import api_answer, VARIANTS
        for code,body in ((429,b'rate limit'),(403,b'access denied'),(400,b'insufficient balance')):
            with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
                guard=RequestGuard(Budget(tmp,1,.02),'x:1')
                def http(_payload):
                    raise HTTPError('https://example.invalid',code,'synthetic',{},BytesIO(body))
                with patch('agent_baseline.planner.Planner._http',side_effect=http) as call, \
                     patch('agent_baseline.planner.time.sleep',side_effect=AssertionError('must not retry')):
                    with self.assertRaises(AskStopped):
                        api_answer([],dict(VARIANTS['f37_corr'],strict_stop=True),guard,1)
                    self.assertEqual(call.call_count,1)
                self.assertTrue((Path(tmp)/'STOP.json').exists())

    def test_stop_and_resume_preserves_answer_and_spending(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            records=[dict(uid=f'i:{n}') for n in range(1,4)]
            calls=[]
            def backend(record,guard):
                calls.append(record['uid'])
                if record['uid']=='i:2':
                    guard.trip('HTTP 429')
                guard.charge(.01)
                return dict(answer='yes',cost_cny=.01)
            with self.assertRaises(AskStopped):
                run_requests(records,backend,tmp,1,concurrency=1,allowance=.02)
            self.assertEqual(calls,['i:1','i:2'])
            saved=json.loads((Path(tmp)/'answers.json').read_text())
            self.assertEqual(list(saved),['i:1'])
            self.assertTrue((Path(tmp)/'STOP.json').exists())
            def ok(record,guard):
                calls.append(record['uid']);guard.charge(.01)
                return dict(answer='no',cost_cny=.01)
            with self.assertRaises(AskStopped):
                run_requests(records,ok,tmp,1,resume=True)
            result=run_requests(records,ok,tmp,1,resume=True,clear_stop=True)
            self.assertEqual(calls,['i:1','i:2','i:2','i:3'])
            self.assertEqual(result['answers'],3)
            self.assertAlmostEqual(result['cost_cny'],.03)

    def test_budget_reservation_and_resume_no_repeat(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            records=[dict(uid=f'x:{i}') for i in range(5)]
            calls=[]
            def backend(record,guard):
                calls.append(record['uid']);guard.charge(.01)
                return dict(answer='yes',cost_cny=.01)
            with self.assertRaises(AskStopped):
                run_requests(records,backend,tmp,.025,concurrency=2,allowance=.02)
            self.assertEqual(calls,['x:0'])
            result=run_requests(records,backend,tmp,1,concurrency=2,allowance=.02,resume=True,clear_stop=True)
            self.assertEqual(len(calls),5)
            self.assertEqual(len(set(calls)),5)
            self.assertAlmostEqual(result['cost_cny'],.05)

    def test_http_failures_write_stop_without_retries(self):
        for code in (429,403,402):
            with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
                calls=[]
                class HttpFailure(Exception):
                    pass
                def backend(record,guard):
                    calls.append(record['uid'])
                    exc=HttpFailure();exc.code=code;raise exc
                with self.assertRaises(AskStopped):
                    run_requests([dict(uid='a:1'),dict(uid='a:2')],backend,tmp,1)
                self.assertEqual(calls,['a:1'])
                self.assertEqual(json.loads((Path(tmp)/'STOP.json').read_text())['reason'],f'HTTP {code}')

    def test_empty_pool_and_settings_change(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            result=run_requests([],lambda *_:self.fail('empty calls backend'),tmp,1,settings={'scope':'x'})
            self.assertEqual(result['answers'],0)
            with self.assertRaises(ValueError):
                run_requests([],lambda *_:None,tmp,1,resume=True,settings={'scope':'y'})

    def test_real_adapter_strict_stop_and_scope_without_network(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            a=Path(tmp)/'a.png';b=Path(tmp)/'b.png'
            save_image(a,np.zeros((12,16,3),np.uint8));save_image(b,np.zeros((12,16,3),np.uint8))
            record=dict(uid='a:1',mask=box(1,1,4,4),bbox=[1,1,4,4],row=dict(
                image_t1=str(a),image_t2=str(b),change_direction='appeared',scope='building_rebuild_excluded'))
            with patch('agent_baseline.ask_eval.api_answer',return_value={'answer':'yes'}) as api:
                result=existing_verifier(record,object())
                self.assertEqual(result['answer'],'yes')
                args=api.call_args.args
                self.assertTrue(args[1]['strict_stop'])
                self.assertEqual(args[4].scope,'building_rebuild_excluded')
                self.assertEqual(args[4].change_direction,'appeared')


class IOTests(unittest.TestCase):
    def test_local_io_retries_three_times(self):
        calls=[]
        def read():
            calls.append(1)
            if len(calls)<3:
                raise OSError('synthetic EIO')
            return 7
        with patch('agent_baseline.analysis.io_retry.time.sleep'):
            self.assertEqual(retry_io(read),7)
        self.assertEqual(len(calls),3)

    def test_output_restricted_to_staging(self):
        with self.assertRaises(ValueError):
            output_path('/somewhere/outside/results.json')


class PipelineTests(unittest.TestCase):
    def test_statistics_candidates_estimated_asks_and_rule_f1_with_fake_data(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root=Path(tmp);work=root/'work';spec=dict(group='fake',dataset='levir',names='fake.txt',scope='building')
            rows=[]
            for i in range(3):
                stem=f'i{i}'
                start=box(1,1,3,3);a=box(5,2,7,4);b=box(10,7,12,9)
                truth=start|a|b
                gt=root/(stem+'.png');save_image(gt,truth.astype(np.uint8)*255)
                rows.append(dict(id=stem,gt_mask=str(gt)))
                save_npz(work/'base/fake'/(stem+'.npz'),start=start,yes=pack([start],start.shape),shape=start.shape)
                data={s:pack([a] if s.startswith('S1') else [b] if s in ('S2','S3') else [],start.shape) for s in SOURCES}
                data.update(shape=start.shape,S3_scores=np.array([.8]),S4_scores=np.zeros(0))
                save_npz(work/'raw/fake'/(stem+'.npz'),**data)
            with patch('agent_baseline.analysis.missed_sources.rows_for',return_value=rows), \
                 patch('agent_baseline.analysis.missed_sources.audit_settings',return_value={'scope':'building'}), \
                 patch('agent_baseline.analysis.missed_sources.ledger_unit',return_value={'unit_cost_cny':.001}):
                summary=statistics_group(root,work,spec)
                self.assertEqual(summary['source_metrics']['S2']['residual_pixel_recall'],1)
                self.assertEqual(summary['source_metrics']['S2']['contributing_images'],3)
                self.assertEqual(summary['thresholds']['S3'],0)
                self.assertEqual(summary['upper_all']['f1'],100)
                self.assertEqual(statistics_group(root,work,spec,resume=True),summary)
                records=list(candidate_records(root,work,spec,[S1,'S2'],combined=True))
                self.assertEqual(len(records),6)
                self.assertIn('idx',records[0])
                def backend(record,guard):
                    guard.charge(.001)
                    return dict(answer=None,cost_cny=.001) # unparsed answers are retained
                run_requests(records,backend,work/'ask/fake'/f'{S1}+S2',1)
                scores=score_answers(root,work,spec,[S1,'S2'])
                self.assertEqual(scores['f1'],100)
                self.assertEqual(scores['unparsed'],6)


if __name__=='__main__':
    unittest.main()
