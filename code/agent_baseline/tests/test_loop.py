import ast
import copy
import json
from pathlib import Path
import re
import unittest
from unittest.mock import patch
import numpy as np
from common import read_jsonl
from agent_baseline.tools import SCHEMAS
from agent_baseline.planner import token_cost, clean_messages
from .helpers import run, call, response, tempdir, mask


def assert_no_evidence(test, payload):
    def scan(value):
        if isinstance(value, dict):
            for key, val in value.items():
                test.assertFalse(key.startswith('gt_') or key == 'eval', key)
                if key == 'url' and isinstance(val,str) and val.startswith('data:image/'):
                    continue
                scan(val)
        elif isinstance(value,list):
            for item in value:
                scan(item)
        elif isinstance(value,str):
            test.assertIsNone(re.search(r'levir|\bf1\b|\bprecision\b|\brecall\b|ground truth|/datasets/', value, re.I), value[:200])
    scan(payload)


class LoopTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempdir()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def test_budget_uses_current_last_round_executes(self):
        script=[response([call('remove_region',dict(mask='mask#0',ids=[1]))])]
        final, box, client, recorder = run(self.root,script,init=mask(),budget=1)
        self.assertEqual(final['submitted_by'],'budget')
        self.assertEqual(final['change_area_px'],0)
        self.assertEqual(final['fallback_kind'],'current')
        self.assertEqual(final['planning_rounds'],1)
        self.assertIn('系统兜底',final['summary'])

    def test_last_round_submit_beats_budget(self):
        final,*_=run(self.root,[response([call('submit',dict(mask='mask#0',summary='Retain it.'))])],init=mask(),budget=1)
        self.assertEqual((final['submitted_by'],final['fallback_kind']),('model','none'))

    def test_budget_empty_and_no_tools_twice(self):
        final,_,client,_=run(self.root,[response(),response()],budget=30)
        self.assertEqual(final['termination_reason'],'no_submit')
        self.assertEqual(final['submitted_by'],'empty')
        self.assertEqual(final['planning_rounds'],2)
        self.assertIn('Please call tools',client.requests[1]['messages'][-1]['content'])
        final,*_=run(self.root/'other',[response()],budget=1)
        self.assertEqual((final['submitted_by'],final['termination_reason']),('empty','budget'))

    def test_no_tools_with_initial(self):
        final,*_=run(self.root,[response(content='I have submitted.'),response()],init=mask())
        self.assertEqual(final['submitted_by'],'no_submit')
        self.assertTrue(final['identical_to_init'])

    def test_bad_json_local_and_submit_skip(self):
        malformed=call('inspect',{})
        malformed['function']['arguments']='{"mask":'
        script=[response([malformed,call('submit',dict(mask='mask#0',summary='Keep.'),'c2'),
                          call('remove_region',dict(mask='mask#0',ids=[1]),'c3')])]
        final,box,_,recorder=run(self.root,script,init=mask())
        calls=recorder.steps[0]['calls']
        self.assertEqual(calls[0]['error_code'],'bad_args')
        self.assertEqual(calls[2]['status'],'skipped_due_to_submit')
        self.assertEqual(final['submitted_by'],'model')
        self.assertEqual(len(box.session.handles),1)

    def test_more_than_eight_calls_and_same_round_handle(self):
        script=[response([call('segment_prompt',dict(time='t1',box=[0,0,10,10]),'c1'),
                          call('inspect',dict(mask='mask#1'),'c2')]+[call('inspect',dict(mask='mask#0'),f'c{i}') for i in range(3,10)]),
                response([call('submit',dict(mask='mask#0',summary='Retain.'))])]
        final,_,client,recorder=run(self.root,script,init=mask())
        calls=recorder.steps[0]['calls']
        self.assertEqual(calls[1]['error_code'],'bad_handle')
        self.assertEqual(calls[8]['error_code'],'bad_args')
        self.assertEqual(len([m for m in client.requests[1]['messages'] if m['role']=='tool']),9)

    def test_all_tools_before_image_and_retention(self):
        script=[response([call('view',{},f'v{i}'),call('inspect',dict(mask='mask#0'),f'i{i}')]) for i in range(5)]
        script.append(response([call('submit',dict(mask='mask#0',summary='Retain.'))]))
        final,_,client,_=run(self.root,script,init=mask())
        messages=client.requests[-1]['messages']
        image_count=sum(b['type']=='image_url' for m in messages if isinstance(m.get('content'),list) for b in m['content'])
        self.assertEqual(image_count,8)
        self.assertIn('[image dropped: view#1',json.dumps(messages))
        self.assertIn('[image dropped: view#2',json.dumps(messages))
        first_assistant=next(i for i,m in enumerate(messages) if m['role']=='assistant')
        self.assertEqual([m['role'] for m in messages[first_assistant:first_assistant+4]],['assistant','tool','tool','user'])
        self.assertEqual(final['view_calls'],5)

    def test_gt_replacement_changes_only_recorder(self):
        script=[response([call('ask',dict(regions=[dict(mask='mask#0',ids=[1])]))]),
                response([call('remove_region',dict(mask='mask#0',ids=[1]))]),
                response([call('submit',dict(mask='mask#1',summary='Remove the region.'))])]
        a,_,client_a,ra=run(self.root/'a',script,gt=mask(),init=mask())
        b,_,client_b,rb=run(self.root/'b',script,gt=mask(None),init=mask())
        self.assertEqual(json.dumps(client_a.requests,sort_keys=True),json.dumps(client_b.requests,sort_keys=True))
        self.assertNotEqual(a['eval'],b['eval'])
        self.assertNotEqual(ra.steps[0]['eval'],rb.steps[0]['eval'])
        for payload in client_a.requests:
            assert_no_evidence(self,payload)
        leaked=copy.deepcopy(client_a.requests[0])
        leaked['messages'].append(dict(role='user',content=json.dumps({'eval':ra.initial_eval})))
        with self.assertRaises(AssertionError):
            assert_no_evidence(self,leaked)

    def test_isolation_negative_and_import_ast(self):
        payload={'messages':[{'role':'user','content':'Region area 12; sam_quality 0.9'}], 'tools':SCHEMAS}
        assert_no_evidence(self,payload)
        for bad in ({'eval':{'f1':.9}}, {'content':'ground truth'}, {'gt_mask':'reference'}, {'content':'/datasets/LEVIR-CD/image'}):
            with self.assertRaises(AssertionError):
                assert_no_evidence(self,bad)
        package=Path(__file__).resolve().parents[1]
        allowed={'recorder.py','summarize.py','fixed_routes.py'}
        for path in package.rglob('*.py'):
            tree=ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.ImportFrom):
                    imported={alias.name for alias in node.names}
                    if imported & {'load_gt','evaluate'}:
                        self.assertIn(path.name,allowed)

    def test_eval_cache_object_counts_and_cost(self):
        import agent_baseline.recorder as rec
        real=rec.object_counts
        with patch.object(rec,'object_counts',wraps=real) as counts:
            final,_,_,recorder=run(self.root,[response([call('inspect',dict(mask='mask#0'))]),
                response([call('submit',dict(mask='mask#0',summary='Retain.'))])],init=mask())
        self.assertEqual(counts.call_count,2)
        self.assertEqual(len(recorder.cache),1)
        self.assertEqual(recorder.steps[0]['eval']['delta_f1'],0)
        self.assertEqual(final['tokens'],dict(input=200,cached=40,output=20))
        self.assertAlmostEqual(final['cost_cny'],token_cost(final['tokens']))
        self.assertTrue((self.root/'input.json').exists())
        records=read_jsonl(self.root/'messages.jsonl')
        self.assertNotIn('data:image',json.dumps(records))
        self.assertEqual(len(records),2)

    def test_error_current_fallback(self):
        # Exhausting the scripted client raises before a second response, never a real request.
        final,*_=run(self.root,[response([call('inspect',dict(mask='mask#0'))])],init=mask())
        self.assertEqual(final['submitted_by'],'error')
        self.assertEqual(final['fallback_kind'],'current')
        self.assertEqual(final['planning_rounds'],1)

    def test_missing_initial_is_recorded_before_error(self):
        final,_,client,_=run(self.root,[],task='T5.1')
        self.assertEqual(final['submitted_by'],'error')
        self.assertEqual(final['fallback_kind'],'empty')
        self.assertEqual(client.requests,[])
        self.assertTrue((self.root/'input.json').exists())

    def test_no_tool_counter_resets(self):
        script=[response(),response([call('inspect',dict(mask='mask#0'))]),response(),
                response([call('submit',dict(mask='mask#0',summary='Keep.'))])]
        final,*_=run(self.root,script,init=mask())
        self.assertEqual(final['submitted_by'],'model')
        self.assertEqual(final['planning_rounds'],4)

    def test_first_candidate_delta_null(self):
        script=[response([call('segment_text',dict(time='t1',text='building'))]),
                response([call('to_mask',dict(sets=['inst#0']))])]
        final,_,_,recorder=run(self.root,script,budget=2)
        self.assertIsNone(recorder.steps[0]['eval'])
        self.assertIsNone(recorder.steps[1]['eval']['delta_f1'])


if __name__ == '__main__':
    unittest.main()
