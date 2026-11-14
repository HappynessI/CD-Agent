import json
from pathlib import Path
import tempfile
import unittest
from agent_baseline.analysis.trace_behavior import analyze, discover, summarize

class TraceBehaviorTests(unittest.TestCase):
    def test_failed_retry_state_and_lineage(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)
            def call(name,ok=True,**kw):
                return dict(tool=name,args={'mask':'mask#0'},ok=ok,**kw)
            rows=[dict(step=0,current_mask='mask#0',calls=[]),
                  dict(step=1,calls=[call('view',False,error_code='bad_args')]),
                  dict(step=2,calls=[call('view',False,error_code='bad_args')]),
                  dict(step=3,calls=[call('combine',handles_created=['mask#1'],input_handles=['mask#0'],current_before='mask#0',current_after='mask#1',eval={'delta_f1':-.3})],current_mask='mask#1')]
            (path/'steps.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            (path/'final.json').write_text(json.dumps(dict(task_id='T3.1',eval={'f1':.2},planning_rounds=3,final_handle='mask#1',termination_reason='budget',submitted_by='budget',change_area_px=2)))
            result=analyze(path)
            kinds=[e['kind'] for e in result['events']]
            self.assertIn('retry_unchanged',kinds)
            self.assertIn('answer_regression',kinds)
            self.assertEqual(result['structural_unused_producer_fraction'],0)
            self.assertEqual(result['error_reactions'],{'same_args':1,'changed_tool':1})
            self.assertEqual(result['first_mask_step'],0)
            result['config']='synthetic'
            self.assertEqual(summarize([result])[0]['normal_round_median'],None)

    def test_discovery_excludes_tests(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ['dev','g1test','test','code_w']:
                p=Path(d)/name;p.mkdir()
                (p/'final.json').write_text('{}');(p/'steps.jsonl').write_text('')
            self.assertEqual([p.name for p in discover(Path(d))],['dev'])
