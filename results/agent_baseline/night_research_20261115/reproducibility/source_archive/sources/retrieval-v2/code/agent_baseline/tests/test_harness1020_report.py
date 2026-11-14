import json
from pathlib import Path
import tempfile
import unittest
from agent_baseline.analysis.harness1020_report import collect


class ReportTests(unittest.TestCase):
    def test_partial_report_resumes_and_records_cost_without_losing_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';out=root/'out';selection=root/'selection.json'
            selection.write_text(json.dumps([dict(pilot=True,manifest=dict(id=n+'.png'),task='T1.1xC0',cohort='hard') for n in ['a','b']]))
            def session(name):
                p=run/'baseline/T1.1xC0/val'/name;p.mkdir(parents=True)
                (p/'input.json').write_text(json.dumps(dict(seed=1,model_parameters={'model':'fixture'},gt_mask_path='fixed')))
                (p/'steps.jsonl').write_text('')
                (p/'final.json').write_text(json.dumps(dict(eval=dict(tp=8,fp=2,fn=2,f1=.8),tool_calls=2,planning_rounds=1,termination_reason='submit',cost_cny=.2)))
                return p
            first=session('a')
            with self.assertRaises(SystemExit):collect(run,selection,out,'baseline')
            self.assertEqual(json.loads((out/'metric.json').read_text())['completed'],1)
            (first/'final.json').unlink()
            session('b');collect(run,selection,out,'baseline')
            result=json.loads((out/'metric.json').read_text())
            self.assertEqual(result['completed'],2)
            self.assertAlmostEqual(result['cost_cny'],.4)
            self.assertAlmostEqual(result['groups']['all']['pooled_f1'],80)

    def test_frozen_task_subset_does_not_expect_other_task_sessions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);selection=root/'selection.json'
            selection.write_text(json.dumps([dict(pilot=True,manifest=dict(id='a.png'),task='T5.1',cohort='simple')]))
            collect(root/'run',selection,root/'out','p2',task='T1.1xC0')
            self.assertEqual(json.loads((root/'out/metric.json').read_text())['expected'],0)

if __name__=='__main__':unittest.main()
