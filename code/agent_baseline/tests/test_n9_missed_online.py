import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from agent_baseline.n9_missed_online_tool import CurrentMaskProvider, N9Toolbox, schema
from agent_baseline.analysis.n8_missed_assets import SOURCES
from agent_baseline.tools import session_schemas
from agent_baseline.state import Session
from agent_baseline.planner import Planner, RunGuard
from agent_baseline.recorder import Recorder


class OnlineContracts(unittest.TestCase):
    def config(self):
        return dict(profile='any',sources='all',semantic=.2,dino=.4,alpha=0.,verifier='V0',k=3)

    def test_reconstructs_against_changed_current_mask(self):
        provider=CurrentMaskProvider({},self.config(),'any')
        provider.sources={s:np.zeros((32,32),bool) for s in SOURCES}
        provider.sources['S1_02'][8:16,8:16]=True
        provider.probabilities=[np.zeros((32,32)),np.ones((32,32))]
        with patch('agent_baseline.analysis.n8_missed_prepare.feature_difference',return_value=.8):
            first,_=provider.candidates(np.zeros((32,32),bool))
            current=first[0]['mask'].copy()
            second,_=provider.candidates(current)
            again,_=provider.candidates(np.zeros((32,32),bool))
        self.assertEqual(len(first),1); self.assertEqual(second,[])
        np.testing.assert_array_equal(again[0]['mask'],first[0]['mask'])

    def test_baseline_schema_excludes_tool_and_directions_are_checked(self):
        N9Toolbox.runtime=dict(arm='baseline',config=self.config())
        with tempfile.TemporaryDirectory() as tmp:
            tool=N9Toolbox([np.zeros((16,16,3),np.uint8)]*2,tmp,session=Session((16,16),np.zeros((16,16),bool)))
            exposed=session_schemas(tool.session,missed_candidates=tool.missed_provider is not None)
        self.assertNotIn('find_missed',[s['function']['name'] for s in exposed])
        with self.assertRaises(ValueError): CurrentMaskProvider({},self.config(),'disappeared')
        self.assertEqual(schema(3)['function']['parameters']['properties']['limit']['maximum'],3)

    def test_paid_tool_returns_proposals_without_changing_answer_and_records_cost(self):
        mask=np.zeros((32,32),bool); mask[8:16,8:16]=True
        candidate=dict(index=0,mask=mask,area=64,bbox=[8,8,16,16],sources=['S1_02'],
                       semantic_delta=.9,p1=0.,p2=.9,dino=.8)
        raw=dict(id='mock-request',choices=[dict(message=dict(role='assistant',content=json.dumps(dict(change='yes',reason='new roof'))),finish_reason='stop')],
                 usage=dict(prompt_tokens=100,completion_tokens=20))
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp); guard=RunGuard(tmp/'STOP.json',tmp/'budget.txt')
            N9Toolbox.runtime=dict(arm='find_missed',asset={},config=self.config(),session_seed=123)
            session=Session(mask.shape,np.zeros(mask.shape,bool),scope='building')
            tool=N9Toolbox([np.zeros((32,32,3),np.uint8)]*2,tmp,ask=SimpleNamespace(guard=guard),session=session)
            before=session.current_mask
            with patch.object(tool.missed_provider,'candidates',return_value=([candidate],{})), patch(
                'agent_baseline.n9_missed_online_tool.Planner',side_effect=lambda **kw:Planner(client=lambda _:raw,**kw)):
                result=tool.find_missed()
            self.assertEqual(session.current_mask,before)
            self.assertFalse(session.get(before).data.any())
            self.assertEqual(result['accepted_ids'],[1]); self.assertTrue(result['props'].startswith('props#'))
            self.assertGreater(guard.spent,0); self.assertAlmostEqual(tool.meta['cost_cny'],guard.spent)
            recorder=object.__new__(Recorder); recorder.ask_tokens=dict(input=0,cached=0,output=0)
            recorder.ask_cost=0.; recorder.ask_attempts=0; archived=[]
            recorder.append=lambda name,event:archived.append((name,event))
            recorder.ask_spending(tool.meta,'find-call')
            self.assertEqual(recorder.ask_tokens['input'],100); self.assertAlmostEqual(recorder.ask_cost,guard.spent)
            self.assertEqual(len(archived),1)
            self.assertTrue((tmp/'n9_missed/call_001/candidate_01/request.json').exists())
            self.assertTrue((tmp/'n9_missed/call_001/candidate_01/attempt_01.json').exists())



class LoopDispatchContracts(unittest.TestCase):
    config = OnlineContracts.config
    def test_actual_loop_schema_dispatch_metadata_and_restoration(self):
        from agent_baseline import loop_json, tools
        from agent_baseline.analysis.n9_missed_agent_entry import installed
        original_box=loop_json.Toolbox; original_schema=tools.SCHEMA_BY_NAME['find_missed']
        for arm in ('baseline','find_missed'):
            runtime=dict(arm=arm,asset={},config=self.config(),session_seed=123)
            with tempfile.TemporaryDirectory() as tmp, patch.object(loop_json,'run_session',side_effect=lambda *a,**k:k['metadata']):
                with installed(runtime) as loop:
                    tool=loop.Toolbox([np.zeros((32,32,3),np.uint8)]*2,tmp,
                                     session=Session((32,32),np.zeros((32,32),bool),scope='building'))
                    names=[s['function']['name'] for s in tools.session_schemas(tool.session,missed_candidates=tool.missed_provider is not None)]
                    self.assertEqual('find_missed' in names,arm=='find_missed')
                    if arm=='find_missed':
                        with patch.object(tool.missed_provider,'candidates',return_value=([],{})):
                            result=tool.execute('find_missed',{'limit':3})
                        self.assertTrue(result.public['ok'])
                        self.assertFalse(tool.execute('find_missed',{'limit':4}).public['ok'])
                    else: self.assertFalse(tool.execute('find_missed',{}).public['ok'])
                    meta=loop.run_session(metadata=dict(backends=dict(ask_model='qwen3.7-flash')))
                    self.assertEqual(meta['backends']['n9_arm'],arm)
                    self.assertEqual(meta['backends']['ask_model'],'qwen3.7-flash')
                self.assertIs(loop_json.Toolbox,original_box)
                self.assertEqual(tools.SCHEMA_BY_NAME['find_missed'],original_schema)

    def test_partial_paid_usage_survives_tool_dispatch_failure(self):
        from agent_baseline.analysis.n9_missed_agent_entry import installed
        mask=np.zeros((32,32),bool); mask[5:13,5:13]=True
        other=np.zeros((32,32),bool); other[20:28,20:28]=True
        def candidate(index,m,bbox):
            return dict(index=index,mask=m,area=64,bbox=bbox,sources=['S1_02'],semantic_delta=.9,p1=0.,p2=.9,dino=.8)
        raw=dict(id='partial-paid',choices=[dict(message=dict(role='assistant',content='{"change":"yes","reason":"new roof"}'),finish_reason='stop')],
                 usage=dict(prompt_tokens=100,completion_tokens=20))
        calls=[]
        def client(_):
            calls.append(1)
            if len(calls)==1:return raw
            raise ValueError('Simulated second-candidate failure')
        runtime=dict(arm='find_missed',asset={},config=self.config(),session_seed=123)
        with tempfile.TemporaryDirectory() as tmp, installed(runtime) as loop:
            tmp=Path(tmp); guard=RunGuard(tmp/'STOP.json',tmp/'budget.txt')
            tool=loop.Toolbox([np.zeros((32,32,3),np.uint8)]*2,tmp,ask=SimpleNamespace(guard=guard),
                session=Session((32,32),np.zeros((32,32),bool),scope='building'))
            before=tool.session.current_mask
            with patch.object(tool.missed_provider,'candidates',return_value=([
                candidate(0,mask,[5,5,13,13]),candidate(1,other,[20,20,28,28])],{})), patch(
                'agent_baseline.n9_missed_online_tool.Planner',side_effect=lambda **kw:Planner(client=client,**kw)):
                result=tool.execute('find_missed',{})
            self.assertFalse(result.public['ok']); self.assertEqual(tool.session.current_mask,before)
            self.assertGreater(result.internal['cost_cny'],0)
            self.assertAlmostEqual(result.internal['cost_cny'],guard.spent)
            self.assertEqual(len(result.internal['attempts']),2)
            self.assertEqual(result.internal['tokens']['input'],100)
            self.assertEqual(len(list((tmp/'n9_missed').glob('call_*/candidate_*/attempt_*.json'))),2)
            recorder=object.__new__(Recorder); recorder.ask_tokens=dict(input=0,cached=0,output=0)
            recorder.ask_cost=0.; recorder.ask_attempts=0; recorder.append=lambda *a:None
            recorder.ask_spending(result.internal,'failed-tool')
            self.assertAlmostEqual(recorder.ask_cost,guard.spent)

if __name__=='__main__': unittest.main()
