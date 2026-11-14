import json
from pathlib import Path
import tempfile
import unittest
from agent_baseline.analysis.night_t3_report import request_cost,delta_reproduction,optional_patchset,remaining16_diagnostic,optional_positive_endpoints,optional_segearth_dense


class ReportTests(unittest.TestCase):
    def test_accounting_ignores_mirrors_and_historical_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name,cost in [('B0/tasks/x/answer.json',8),('B1/tasks/x/answer.json',.2),('B1/tasks/x/failure_000.json',.1),('B1/tasks/x/attempt_001.json',.2),('B1/summary.json',.3),('t33/retrieval/B0/tasks/x/answer.json',.4),('r1_residual/tune/tasks/x/answer.json',.05),('r1_residual/tune/tasks/y/reused.json',9)]:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(dict(cost_cny=cost)))
            result=request_cost(root)
            self.assertAlmostEqual(result['cost_cny'],.75)
            self.assertEqual(result['request_artifacts'],4)
            self.assertAlmostEqual(result['by_root']['r1_residual'],.05)
            self.assertAlmostEqual(result['by_root']['t33'],.4)

    def test_patchset_partial_does_not_claim_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            stages,rows=optional_patchset(root)
            self.assertEqual(stages['r1_patchset/confirm']['status'],'not_started')
            path=root/'r1_patchset/tune/tasks/example/metrics.json'
            path.parent.mkdir(parents=True);path.write_text(json.dumps(dict(id='example',split='tune')))
            stages,rows=optional_patchset(root)
            self.assertEqual(len(rows),1)
            self.assertEqual(stages['r1_patchset/tune']['status'],'partial')
            self.assertEqual(stages['r1_patchset/tune']['expected'],30)

    def test_remaining16_is_additional_total_pixel_sensitivity(self):
        records=[dict(id=str(n),split='tune',group='目标类',remaining_gt_pixels=n,
            scores=[dict(method='m',k=8,counts=[n,2,0])]) for n in (0,15,16,20)]
        data=dict(endpoint=dict(records=[]),retrieval=dict(stage=dict(records=records)),audit=None)
        before=json.dumps(data,sort_keys=True)
        result=remaining16_diagnostic(data)
        row=next(r for r in result['rows'] if r['split']=='tune' and r['group']=='全部')
        self.assertEqual(row['tasks'],2)
        self.assertEqual(row['tp'],36)
        self.assertEqual(row['available_stage_tasks'],4)
        self.assertEqual([r['id'] for r in row['per_task_counts']],['16','20'])
        self.assertEqual(json.dumps(data,sort_keys=True),before)

    def test_optional_endpoint_missing_and_actual_summary_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);stages,records,value=optional_positive_endpoints(root)
            self.assertEqual(stages['positive_endpoints/B2']['status'],'not_started')
            self.assertIsNone(value)
            folder=root/'positive_endpoints_tune';folder.mkdir()
            (folder/'inventory.json').write_text(json.dumps(dict(unique_missing=28)))
            stages,_,_=optional_positive_endpoints(root)
            self.assertEqual(stages['positive_endpoints/B2']['status'],'pending_cache')
            item=dict(tasks=30,expected=30,status='complete',records=[],summary=[])
            (folder/'summary.json').write_text(json.dumps(dict(variants=dict(B0=item))))
            stages,records,_=optional_positive_endpoints(root)
            self.assertEqual(stages['positive_endpoints/B0']['status'],'complete')
            self.assertEqual(records['positive_endpoints/B0']['summary'],[])

    def test_confirmation_reads_records_not_stale_aggregate(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            stages,_,_=optional_positive_endpoints(root,'confirm')
            self.assertEqual(stages['positive_endpoints_confirm/B0']['status'],'not_started')
            folder=root/'positive_endpoints_confirm';folder.mkdir()
            (folder/'summary.json').write_text(json.dumps(dict(variants=dict(B0=dict(tasks=30)))))
            path=folder/'B0/tasks/a/metrics.json';path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(id='a',split='confirm',group='目标类',remaining_gt_pixels=0,scores=[],verification={})))
            stages,records,value=optional_positive_endpoints(root,'confirm')
            self.assertEqual(stages['positive_endpoints_confirm/B0']['completed'],1)
            self.assertEqual(stages['positive_endpoints_confirm/B0']['status'],'partial')
            self.assertEqual(value['variants']['B0']['tasks'],1)
            self.assertNotIn('positive_endpoints/B0',records)
            self.assertNotIn('B3',value['variants'])

    def test_dense_counts_only_schema_is_separate_and_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);stage,value=optional_segearth_dense(root)
            self.assertEqual(stage['status'],'not_started');self.assertIsNone(value)
            path=root/'segearth_dense_tune/tasks/a/metrics.json';path.parent.mkdir(parents=True)
            row=dict(id='a',split='tune',group='目标类',remaining_gt_pixels=16,
                routeA_reproduced=True,B0_reproduced=True,
                scores=[dict(method=m,k=0,counts=[4,2,12]) for m in ('routeA_historical','B0_historical','B2_dense','B3_dense')])
            path.write_text(json.dumps(row))
            stage,value=optional_segearth_dense(root)
            self.assertEqual(stage['status'],'partial');self.assertEqual(stage['completed'],1)
            self.assertEqual(value['historical_reproduction'],dict(routeA=1,B0=1))
            group=next(r for r in value['summary'] if r['support']=='positive' and r['group']=='全部')
            self.assertEqual(group['methods']['B2_dense']['tp'],4)
            self.assertNotIn('covered_pixels',group['methods']['B2_dense'])

    def test_reproduction_requires_original_counts(self):
        old=dict(per_task=[dict(id='a',records=[dict(method='dino',k=8,counts=[1,2,3])])])
        new=[dict(id='a',scores=[dict(method='delta',k=8,counts=[1,2,3]),dict(method='endpoint',k=8,counts=[9,9,9])])]
        self.assertTrue(delta_reproduction(new,old)['all_available_agree'])
        new[0]['scores'][0]['counts']=[1,3,3]
        self.assertEqual(len(delta_reproduction(new,old)['mismatches']),1)
        self.assertFalse(delta_reproduction([],old)['all_available_agree'])


if __name__=='__main__':unittest.main()
