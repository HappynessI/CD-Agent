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
from agent_baseline.tools import Toolbox
from .helpers import run, call, response, tempdir, mask, rgb, segment, prompt


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

    def test_review_opens_once_and_the_same_submit_then_finishes(self):
        submit=call('submit',dict(mask='mask#0',summary='Retain.'))
        final,box,client,recorder=run(self.root,[response([submit]),response([submit])],init=mask(),review=True)
        self.assertEqual(final['submitted_by'],'model')
        self.assertEqual(final['planning_rounds'],2)
        first=recorder.steps[0]['calls'][0]
        self.assertEqual(first['status'],'report')
        self.assertIn('Report before submission',first['result_text'])
        assert_no_evidence(self,json.loads(first['result_text']))
        self.assertEqual(recorder.steps[1]['calls'][0]['status'],'executed')

    def test_review_reports_records_by_outline_and_additions(self):
        init=mask()|mask((30,20,35,25))
        script=[response([call('ask',dict(regions=[dict(mask='mask#0',ids=[1])]))]),
                response([call('segment_prompt',dict(time='t1',box=[12,0,18,3]))]),
                response([call('combine',dict(op='union',a='mask#0',b='mask#1'))]),
                response([call('submit',dict(mask='mask#2',summary='Add.'))]),
                response([call('submit',dict(mask='mask#2',summary='Add.'))])]
        _,box,_,recorder=run(self.root,script,init=init,review=True)
        text=json.loads(recorder.steps[3]['calls'][0]['result_text'])['report']
        self.assertIn('1 yes, 0 no, 0 unparsed, 2 with no answer',text)
        self.assertIn('no answer: id=',text)
        self.assertRegex(text,r'[1-9][0-9]* pixels added')
        self.assertEqual(box.session.submitted,'mask#2')

    def test_review_at_three_replies_left_and_on_the_last_reply(self):
        view=lambda i:response([call('view',dict(region=[0,0,10+i,10+i]),f'v{i}')])
        script=[view(0),view(1),view(2),response([call('submit',dict(mask='mask#0',summary='Retain.'))])]
        final,_,client,recorder=run(self.root,script,init=mask(),budget=5,review=True)
        # Reply 2 leaves 3 replies, so the checklist arrives there and the later submit is not intercepted.
        self.assertEqual(final['planning_rounds'],4)
        self.assertEqual(recorder.steps[3]['calls'][0]['status'],'executed')
        texts=[m['content'] for m in client.requests[2]['messages'] if m['role']=='user' and isinstance(m['content'],str)]
        self.assertTrue(any('Report before submission' in t for t in texts))
        last=[response([call('submit',dict(mask='mask#0',summary='Retain.'))])]
        (self.root/'one').mkdir()
        final,_,_,_=run(self.root/'one',last,init=mask(),budget=1,review=True)
        self.assertEqual(final['submitted_by'],'model')

    def test_verify_stage_checks_all_reports_and_never_edits(self):
        init=mask()|mask((30,20,35,25))
        seen=[]
        def reply(pairs,question):
            seen.append(len(pairs))
            return [dict(answer='no' if i==0 else 'yes',type='no_change',reason='Same roof.',p_yes=None) for i,_ in enumerate(pairs)],dict(tokens=dict(input=10,cached=0,output=2),cost_cny=.5,attempts=[])
        submit=call('submit',dict(mask='mask#0',summary='Keep all.'))
        final,box,_,recorder=run(self.root,[response([submit]),response([submit])],init=init,verify=True,ask_backend=reply)
        self.assertEqual(seen,[2])
        text=json.loads(recorder.steps[0]['calls'][0]['result_text'])['report']
        self.assertIn('1 yes, 1 no',text)
        self.assertIn('IDs answered no',text)
        for word in ('reject','fail','must'):
            self.assertNotIn(word,text.lower())
        self.assertTrue((box.session.get('mask#0').data==init).all())
        self.assertEqual(box.session.submitted,'mask#0')
        self.assertEqual(final['submitted_by'],'model')
        self.assertAlmostEqual(final['ask_cost_cny'],.5)
        ledger=json.loads((self.root/'verification.json').read_text())
        self.assertEqual(sum(row['differs_from_rule'] for row in ledger),1)

    def test_stage_also_checks_given_components_dropped_unasked(self):
        init=mask()|mask((30,20,35,25))
        seen=[]
        def reply(pairs,question):
            seen.append(len(pairs))
            return [dict(answer='yes',type='new_building',reason='New.',p_yes=.9) for _ in pairs],{}
        script=[response([call('remove_region',dict(mask='mask#0',ids=[1]))]),
                response([call('submit',dict(mask='mask#1',summary='One.'))]),response([call('submit',dict(mask='mask#1',summary='One.'))])]
        _,_,_,recorder=run(self.root,script,init=init,verify=True,ask_backend=reply)
        self.assertEqual(seen,[2])
        text=json.loads(recorder.steps[1]['calls'][0]['result_text'])['report']
        self.assertIn('gone and answered yes, IDs on mask#0: [1]',text)

    def test_new_outline_goes_back_to_the_verifier_and_a_checked_one_does_not(self):
        init=mask()|mask((30,20,35,25))
        seen=[]
        def reply(pairs,question):
            seen.append(len(pairs))
            return [dict(answer='yes',type='new_building',reason='New.',p_yes=.9) for _ in pairs],{}
        script=[response([call('submit',dict(mask='mask#0',summary='One.'))]),
                response([call('remove_region',dict(mask='mask#0',ids=[1]))]),
                response([call('submit',dict(mask='mask#1',summary='Two.'))])]
        final,box,_,recorder=run(self.root,script,init=init,verify=True,ask_backend=reply)
        self.assertEqual(seen,[2])
        self.assertEqual(box.session.submitted,'mask#1')
        self.assertEqual(final['planning_rounds'],3)
        self.assertEqual(recorder.steps[2]['calls'][0]['status'],'executed')

    def test_ask_defaults_to_unchecked_components_largest_first_and_caps(self):
        from agent_baseline import tools
        init=np.zeros((32,48),bool)
        init[1,1::2]=True
        init[3:10,20:30]=True
        def reply(pairs,question):
            return [dict(answer='yes',type='x',reason='r',p_yes=.5) for _ in pairs],{}
        box=Toolbox(rgb(),self.root,segment,prompt,reply,init_mask=init)
        old,tools.MAX_ASK=tools.MAX_ASK,3
        self.addCleanup(setattr,tools,'MAX_ASK',old)
        first=box.execute('ask',{}).public
        self.assertEqual(len(first['results']),3)
        self.assertEqual(first['results'][0]['id'],max(box.session.get('mask#0').components,key=lambda c:c['area'])['id'])
        self.assertGreater(first['remaining_unverified'],0)
        second=box.execute('ask',{}).public
        self.assertFalse({r['id'] for r in first['results']}&{r['id'] for r in second['results']})

    def test_proposal_becomes_current_and_status_reports_it(self):
        script=[response([call('segment_prompt',dict(time='t1',box=[0,0,6,6]))]),
                response([call('submit',dict(mask='mask#0',summary='Propose.'))]),
                response([call('view',{})]),
                response([call('submit',dict(mask='mask#0',summary='Done.'))])]
        final,box,client,recorder=run(self.root,script,review=True,status=True)
        text=[m['content'] for m in client.requests[2]['messages'] if m['role']=='user' and isinstance(m['content'],str)][-1]
        self.assertIn('Current answer mask#0',text)
        self.assertEqual(box.session.submitted,'mask#0')

    def test_report_lists_what_left_the_mask_and_all_answers_are_kept(self):
        init=mask()|mask((30,20,35,25))
        replies=iter(['yes','no'])
        def reply(pairs,question):
            return [dict(answer=next(replies),type='x',reason='because',p_yes=None) for _ in pairs],{}
        ask1=call('ask',dict(regions=[dict(mask='mask#0',ids=[1])]))
        ask2=call('ask',dict(regions=[dict(mask='mask#0',ids=[1])]),'b')
        script=[response([ask1]),response([call('view',{})]),response([ask2]),
                response([call('remove_region',dict(mask='mask#0',ids=[2]))]),
                response([call('submit',dict(mask='mask#1',summary='x'))]),response([call('submit',dict(mask='mask#1',summary='x'))])]
        _,box,_,recorder=run(self.root,script,init=init,review=True,ask_backend=reply)
        text=json.loads(recorder.steps[4]['calls'][0]['result_text'])['report']
        self.assertIn('answers disagree across repeats for IDs: [1]',text)
        self.assertIn('1 given components are gone',text)
        self.assertIn('gone with no answer',text)
        entry=box.entry(box.session.get('mask#0'),box.session.get('mask#0').components[0])
        self.assertEqual([r['answer'] for r in entry['records']],['yes','no'])

    def test_skills_are_indexed_and_loaded_only_when_on(self):
        from agent_baseline.tools import SKILLS
        script=[response([call('load_skill',dict(name='mask-correction'))]),response([call('submit',dict(mask='mask#0',summary='Keep.'))])]
        final,box,client,recorder=run(self.root,script,init=mask(),skills=('mask-correction',))
        self.assertEqual(final['submitted_by'],'model')
        system=client.requests[0]['messages'][0]['content']
        self.assertIn('- mask-correction:',system)
        self.assertNotIn('Removing:',system)
        self.assertIn('Removing:',recorder.steps[0]['calls'][0]['result_text'])
        self.assertIn('load_skill',[t['function']['name'] for t in client.requests[0]['tools']])
        (self.root/'off').mkdir()
        _,_,client,_=run(self.root/'off',[response([call('submit',dict(mask='mask#0',summary='Keep.'))])],init=mask())
        self.assertNotIn('load_skill',[t['function']['name'] for t in client.requests[0]['tools']])
        for path in SKILLS.glob('*/SKILL.md'):
            text=path.read_text()
            for word in ('LEVIR','levir','F1','86.','val_','train_'):
                self.assertNotIn(word,text,path)

    def test_tools_and_prompts_are_the_ones_g1_used(self):
        g1=json.loads((Path(__file__).parent/'fixtures'/'g1_prompts.json').read_text(encoding='utf-8'))
        for task,init in (('T1.1xC0',None),('T5.1',mask())):
            script=[response([call('submit',dict(mask='empty',summary='x'))]),response([call('submit',dict(mask='empty',summary='x'))])]
            _,_,client,_=run(self.root,script,init=init,task=task,status=True,verify=True,prompt_version='v4')
            request=client.requests[0]
            self.assertEqual(request['tools'],g1[task]['schemas'])
            self.assertEqual(request['messages'][0]['content'],g1[task]['system_prompt'])
            self.assertTrue(request['messages'][1]['content'][0]['text'].startswith(g1[task]['instruction']))

    def test_submit_empty_answers_nothing_changed(self):
        final,box,_,_=run(self.root,[response([call('submit',dict(mask='empty',summary='No change.'))])])
        self.assertEqual(final['submitted_by'],'model')
        self.assertEqual(final['change_area_px'],0)

    def test_status_line_and_plan_only_when_on(self):
        plan=call('plan',dict(items=[dict(text='Look at both dates',status='done'),dict(text='Check components',status='todo')]))
        script=[response([plan]),response([call('submit',dict(mask='mask#0',summary='Retain.'))])]
        final,box,client,recorder=run(self.root,script,init=mask(),status=True,plan=True)
        self.assertEqual(final['submitted_by'],'model')
        names=[t['function']['name'] for t in client.requests[0]['tools']]
        self.assertIn('plan',names)
        texts=[m['content'] for m in client.requests[1]['messages'] if m['role']=='user' and isinstance(m['content'],str)]
        self.assertEqual(len(texts),1)
        self.assertIn('29 replies left',texts[0])
        self.assertIn('0 yes, 0 no, 0 unparsed, 1 none',texts[0])
        self.assertIn('Plan: 1/2 closed; next: Check components',texts[0])
        assert_no_evidence(self,dict(content=texts[0]))
        (self.root/'off').mkdir()
        _,_,client,_=run(self.root/'off',[response([call('submit',dict(mask='mask#0',summary='Retain.'))])],init=mask())
        self.assertNotIn('plan',[t['function']['name'] for t in client.requests[0]['tools']])

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
        final,_,client,_=run(self.root,[response(content='pseudo call A'),response(content='pseudo call B'),response()],budget=30)
        self.assertEqual(final['termination_reason'],'no_submit')
        self.assertEqual(final['submitted_by'],'empty')
        self.assertEqual(final['planning_rounds'],3)
        self.assertIn('no function call',client.requests[1]['messages'][-1]['content'])
        # The text-only reply leaves the context; the retry forces a call under another seed.
        self.assertNotIn('pseudo call A',json.dumps(client.requests[1]['messages']))
        self.assertEqual(client.requests[1]['tool_choice'],'required')
        self.assertNotEqual(client.requests[0]['seed'],client.requests[1]['seed'])
        self.assertNotEqual(client.requests[1]['seed'],client.requests[2]['seed'])
        final,*_=run(self.root/'other',[response()],budget=1)
        self.assertEqual((final['submitted_by'],final['termination_reason']),('empty','budget'))

    def test_no_tools_with_initial(self):
        final,*_=run(self.root,[response(content='I have submitted.'),response(),response()],init=mask())
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
        script=[response([call('view',dict(region=[0,0,2+i,2+i]),f'v{i}'),call('inspect',dict(mask='mask#0'),f'i{i}')]) for i in range(5)]
        script.append(response([call('submit',dict(mask='mask#0',summary='Retain.'))]))
        final,_,client,_=run(self.root,script,init=mask())
        messages=client.requests[-1]['messages']
        image_count=sum(b['type']=='image_url' for m in messages if isinstance(m.get('content'),list) for b in m['content'])
        self.assertEqual(image_count,12)  # five view pairs + the initial pair: nothing dropped yet
        self.assertNotIn('[image dropped',json.dumps(messages))
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
        allowed={'replay_g1.py','make_levir_tasks.py','recorder.py','summarize.py','fixed_routes.py','make_demo.py','pick_demo.py','diagnose.py','ask_eval.py','compare_runs.py','levir_analysis.py','boundary_eval.py','sam3_soft_eval.py','agent_vs_rule.py','vs_rule.py','verifier_select.py','planner_room.py','ask_pair.py','ask_fp_sample.py','vs_fixed.py','vs_fixed_r7.py','g3_vs_g2_levir.py','make_t41_tasks.py',
                 'test_pyes_band.py'}  # offline p_yes analysis test; its 'evaluate' is the analysis's own scorer
        # Offline analysis may read labels, but nothing the agent runs may import it.
        online=['loop_json.py','tools.py','state.py','view.py','backends.py','planner.py','instances.py','tasks.py','python_sandbox.py','python_worker.py']
        for filename in online:
            source=(package/filename).read_text()
            for module in ('make_demo','pick_demo','diagnose','summarize','fixed_routes','ask_eval','compare_runs','levir_analysis','replay_g1','make_levir_tasks'):
                self.assertNotIn('import '+module,source)
                self.assertNotIn('from .'+module,source)
        for path in package.rglob('*.py'):
            tree=ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.ImportFrom):
                    imported={alias.name for alias in node.names}
                    if imported & {'load_gt','evaluate'}:
                        self.assertIn(path.name,allowed)

    def test_images_leave_in_one_batch_and_prefix_stays_stable(self):
        script=[response([call('view',dict(region=[0,0,2+i,2+i]),f'v{i}')]) for i in range(9)]
        script.append(response([call('submit',dict(mask='mask#0',summary='Retain.'))]))
        _,_,client,_=run(self.root,script,init=mask())
        def images(request):
            return sum(b['type']=='image_url' for m in request['messages'] if isinstance(m.get('content'),list) for b in m['content'])
        counts=[images(r) for r in client.requests]
        self.assertEqual(max(counts),16)
        self.assertEqual(counts[-1],10)
        # Up to the batch drop each request extends the previous one without editing it.
        for before,after in zip(client.requests[:6],client.requests[1:7]):
            self.assertEqual(json.dumps(before['messages']),json.dumps(after['messages'][:len(before['messages'])]))

    def yes_to_all(self, seen=None):
        def reply(pairs,question):
            if seen is not None:
                seen.append(len(pairs))
            return [dict(answer='yes',type='new_building',reason='New.',p_yes=.9) for _ in pairs],{}
        return reply

    def test_gate2_no_current_answer_gets_a_report_on_the_latest_mask(self):
        view=lambda i:response([call('view',dict(region=[0,0,10+i,10+i]),f'v{i}')])
        make=response([call('segment_text',dict(time='t2',text='building',tile=512),'s')])
        script=[make,response([call('to_mask',dict(sets=['inst#0'],set_current=False),'m')]),view(0),view(1),response([call('submit',dict(mask='mask#1',summary='One.'),'x')]),response([call('submit',dict(mask='mask#1',summary='One.'),'y')])]
        final,box,client,recorder=run(self.root,script,verify=True,ask_backend=self.yes_to_all(),gate2=True,budget=6)
        texts=[m['content'] for r in client.requests for m in r['messages'] if m['role']=='user' and isinstance(m['content'],str)]
        self.assertTrue(any('You have no current answer yet' in t and 'Report before submission' in t for t in texts))
        self.assertEqual(final['submitted_by'],'model')

    def test_gate2_last_reply_submit_gets_a_report_and_two_replies_once(self):
        submit=lambda i:response([call('submit',dict(mask='mask#0',summary='Keep.'),f's{i}')])
        final,_,client,recorder=run(self.root,[submit(0),submit(1),submit(2)],init=mask(),verify=True,ask_backend=self.yes_to_all(),gate2=True,budget=1)
        self.assertEqual(recorder.steps[0]['calls'][0]['status'],'report')
        self.assertEqual(recorder.steps[1]['calls'][0]['status'],'executed')
        self.assertEqual(final['submitted_by'],'model')
        gate=json.loads((self.root/'gate.json').read_text())
        self.assertEqual(gate['grant']['kind'],'last_reply')
        self.assertEqual(gate['extra_used'],1)

    def test_gate2_stopping_without_tools_gets_a_report_once(self):
        final,_,client,recorder=run(self.root,[response([],'a'),response([],'b'),response([],'c'),response([],'d'),response([call('submit',dict(mask='mask#0',summary='Keep.'),'s')])],init=mask(),verify=True,ask_backend=self.yes_to_all(),gate2=True)
        gate=json.loads((self.root/'gate.json').read_text())
        self.assertEqual(gate['grant']['kind'],'no_tools')
        self.assertEqual(gate['grant']['at_reply'],3)
        self.assertEqual(final['submitted_by'],'model')
        self.assertIn('Report before submission',json.dumps(client.requests[3]['messages'][-2:]))

    def test_gate2_budget_ending_with_unchecked_components_gets_a_report(self):
        view=lambda i:response([call('view',dict(region=[0,0,10+i,10+i]),f'v{i}')])
        script=[response([call('segment_text',dict(time='t2',text='building',tile=512),'g')]),
                response([call('add_region',dict(mask='mask#0',other='inst#1'),'a')]),view(0),
                response([call('submit',dict(mask='mask#2',summary='Keep.'),'s')]),response([call('submit',dict(mask='mask#2',summary='Keep.'),'t')])]
        final,_,_,_=run(self.root,script,init=mask(),verify=True,ask_backend=self.yes_to_all(),gate2=True,budget=3)
        gate=json.loads((self.root/'gate.json').read_text())
        self.assertEqual(gate['grant']['kind'],'budget')
        self.assertEqual(final['submitted_by'],'model')

    def test_gate2_resubmit_that_drops_unasked_given_components_goes_back_to_the_gate(self):
        init=mask()|mask((30,20,35,25))
        calls=[]
        def flaky(pairs,question):
            calls.append(len(pairs))
            if len(calls)==2:
                raise RuntimeError('verifier down')
            return [dict(answer='yes',type='new_building',reason='New.',p_yes=.9) for _ in pairs],{}
        script=lambda:[response([call('ask',dict(regions=[dict(mask='mask#0',ids=[1])]),'a')]),
                response([call('remove_region',dict(mask='mask#0',ids=[2]),'r')]),
                response([call('submit',dict(mask='mask#1',summary='One.'),'s')]),
                response([call('submit',dict(mask='mask#1',summary='One.'),'t')]),
                response([call('submit',dict(mask='mask#1',summary='One.'),'u')])]
        final,_,_,recorder=run(self.root,script(),init=init,verify=True,ask_backend=flaky,gate2=True)
        self.assertEqual(recorder.steps[3]['calls'][0]['status'],'report')
        self.assertEqual(final['planning_rounds'],5)
        (self.root/'g1').mkdir()
        calls.clear()
        final,_,_,recorder=run(self.root/'g1',script(),init=init,verify=True,ask_backend=flaky)
        self.assertEqual(recorder.steps[3]['calls'][0]['status'],'executed')
        self.assertEqual(final['planning_rounds'],4)

    def test_gate2_lets_the_next_ask_batch_through_and_still_blocks_a_true_repeat(self):
        init=np.zeros((32,48),bool)
        for k in range(3):
            init[2,2+4*k:4+4*k]=True
        ask=lambda i:response([call('ask',{},f'a{i}')])
        submit=lambda i:response([call('submit',dict(mask='mask#0',summary='All.'),f's{i}')])
        import agent_baseline.tools as tools_module
        seen=[]
        with patch.object(tools_module,'MAX_ASK',2):
            final,_,_,recorder=run(self.root,[ask(0),ask(1),ask(2),submit(0),submit(1)],init=init,verify=True,ask_backend=self.yes_to_all(seen),gate2=True)
        self.assertEqual(seen,[2,1])
        self.assertEqual(len(recorder.steps[1]['calls']),1)
        self.assertEqual(recorder.steps[2]['calls'],[])
        self.assertEqual(final['submitted_by'],'model')
        (self.root/'g1').mkdir()
        seen.clear()
        with patch.object(tools_module,'MAX_ASK',2):
            _,_,_,recorder=run(self.root/'g1',[ask(0),ask(1),submit(0),submit(1)],init=init,verify=True,ask_backend=self.yes_to_all(seen))
        self.assertEqual(recorder.steps[1]['calls'],[])

    def test_repeated_call_does_not_run_or_stay_in_context(self):
        same=lambda i:response([call('inspect',dict(mask='mask#0'),f'r{i}')])
        final,_,client,recorder=run(self.root,[same(0),same(1),response([call('submit',dict(mask='mask#0',summary='Retain.'))])],init=mask())
        self.assertEqual(final['submitted_by'],'model')
        self.assertEqual(recorder.steps[1]['calls'],[])
        self.assertEqual(client.requests[2]['tool_choice'],'required')
        self.assertIn('You repeated',json.dumps(client.requests[2]['messages'][-1]))
        self.assertNotIn('r1',[t['id'] for m in client.requests[2]['messages'] for t in m.get('tool_calls') or []])

    def test_quota_error_stops_every_session_without_final(self):
        import io, urllib.error
        from agent_baseline.planner import Planner, RunGuard, RunStopped
        def denied(code,body):
            def client(payload):
                raise urllib.error.HTTPError('u',code,'x',{},io.BytesIO(body))
            return client
        stop=self.root/'STOP.json'
        guard=RunGuard(stop)
        planner=Planner(client=denied(403,b'Access to model denied'),guard=guard,sleep=lambda s:None)
        with self.assertRaises(RunStopped):
            planner.complete([dict(role='user',content='x')],[],1)
        self.assertEqual(planner.attempts,1)
        self.assertTrue(stop.exists())
        # Another session, even another process sharing the file, sends nothing.
        other=Planner(client=lambda payload:self.fail('no request after a stop'),guard=RunGuard(stop))
        with self.assertRaises(RunStopped):
            other.complete([dict(role='user',content='x')],[],1)
        stop.unlink()
        limited=Planner(client=denied(429,b'Allocated quota exceeded'),guard=guard,sleep=lambda s:None)
        with self.assertRaises(RunStopped):
            limited.complete([dict(role='user',content='x')],[],1)
        self.assertEqual(limited.attempts,1)
        stop.unlink()
        slow=[]
        busy=Planner(client=denied(429,b'Too many requests'),guard=guard,sleep=slow.append)
        with self.assertRaises(Exception):
            busy.complete([dict(role='user',content='x')],[],1)
        self.assertEqual(busy.attempts,6)  # an ordinary rate limit still backs off and retries
        self.assertFalse(stop.exists())

    def test_budget_ledger_stops_the_batch(self):
        from agent_baseline.planner import Planner, RunGuard, RunStopped
        guard=RunGuard(self.root/'STOP.json',self.root/'budget.txt',max_cost=0.001)
        reply=lambda payload:dict(id='a',choices=[dict(message=dict(role='assistant',content='x'))],usage=dict(prompt_tokens=1000,completion_tokens=0))
        planner=Planner(client=reply,guard=guard)
        with self.assertRaises(RunStopped):
            planner.complete([dict(role='user',content='x')],[],1)
        # A spent budget stops its own batch and leaves every other batch running.
        self.assertFalse((self.root/'STOP.json').exists())
        self.assertTrue((self.root/'budget'/'STOP.json').exists())
        self.assertTrue(guard.stopped())
        self.assertFalse(RunGuard(self.root/'STOP.json',self.root/'other.txt').stopped())

    def test_budget_ledger_sums_processes_and_survives_a_bad_peer(self):
        from agent_baseline.planner import RunGuard, RunStopped
        one=RunGuard(self.root/'STOP.json',self.root/'batch.txt',max_cost=1)
        two=RunGuard(self.root/'STOP.json',self.root/'batch.txt',max_cost=1)
        two.own=one.own.with_name('peer.txt')
        one.charge(.4)
        two.charge(.4)
        self.assertEqual(float(one.own.read_text()),.4)
        (one.directory/'broken.txt').write_bytes(b'\x00\x00.1')
        one.checked=two.checked=0
        self.assertAlmostEqual(one.total(),.8)
        self.assertFalse(one.stopped())
        one.checked=0
        with self.assertRaises(RunStopped):
            one.charge(.3)
        self.assertIn('budget',(one.directory/'STOP.json').read_text())
        self.assertFalse((self.root/'STOP.json').exists())

    def test_request_cost_by_model_and_tier(self):
        from agent_baseline.planner import request_cost, token_cost
        small=dict(input=1_000_000,cached=0,output=1_000_000)
        self.assertAlmostEqual(request_cost('qwen3.7-plus',small),token_cost(small))
        self.assertAlmostEqual(request_cost('qwen3.7-flash',dict(input=30_000,cached=30_000,output=1000)),(30_000*.2+1000*.8)/1e6)
        self.assertAlmostEqual(request_cost('qwen3.7-flash',dict(input=40_000,cached=0,output=1000)),(40_000*.6+1000*2.4)/1e6)

    def test_budget_ledger_stops_when_it_cannot_be_written(self):
        from agent_baseline.planner import RunGuard, RunStopped
        (self.root/'blocked').write_text('a file where the ledger directory should be')
        guard=RunGuard(self.root/'STOP.json',self.root/'blocked.txt')
        with self.assertRaises(RunStopped):
            for _ in range(guard.MAX_WRITE_FAILURES):
                guard.charge(.1)
        self.assertIn('ledger',(self.root/'STOP.json').read_text())

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
