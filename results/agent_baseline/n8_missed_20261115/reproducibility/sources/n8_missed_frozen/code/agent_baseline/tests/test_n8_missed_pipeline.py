import copy
import unittest
import numpy as np
from agent_baseline.analysis.n8_missed_pipeline import residual_state,should_verify,make_request,rank_candidates

class PipelineTest(unittest.TestCase):
    def test_no_physical_evidence_means_no_verifier(self):
        mask=np.ones((8,8),bool)
        c=dict(index=0,semantic_delta=.01,dino=.8)
        state=residual_state([c],[mask],mask.shape,.1)
        self.assertEqual(state['net_candidates'],0);self.assertFalse(should_verify(state,25))
        c['semantic_delta']=.5;c['dino']=.1
        state=residual_state([c],[mask],mask.shape,.1)
        self.assertEqual(state['net_candidates'],1);self.assertFalse(should_verify(state,25))
        c['dino']=.2
        self.assertTrue(should_verify(residual_state([c],[mask],mask.shape,.1),25))
    def test_double_arm_changes_only_evidence_text(self):
        template=dict(messages=[dict(role='system',content='scope'),dict(role='user',content=[
            *[dict(type='image_url',image_url={'url':'old'}) for _ in range(4)],dict(type='text',text='original question')])],
            parameters=dict(model='old',seed=99,temperature=0,max_tokens=256))
        original=copy.deepcopy(template);candidate=dict(area=80,p1=.1,p2=.9,dino=.4)
        left=make_request(template,['0.jpg','1.jpg','2.jpg','3.jpg'],candidate,'V0',3)
        right=make_request(template,['0.jpg','1.jpg','2.jpg','3.jpg'],candidate,'V0_evidence',3)
        self.assertTrue(right['messages'][1]['content'][-1]['text'].startswith('original question\nQuantitative evidence'))
        right['messages'][1]['content'][-1]['text']=left['messages'][1]['content'][-1]['text']
        self.assertEqual(left,right);self.assertEqual(template,original);self.assertEqual(left['parameters']['seed'],1019)
    def test_rank_prefix_and_no_label_or_source_routing(self):
        candidates=[dict(index=i,semantic_delta=.3,dino=.4,area=100+i) for i in range(12)]
        first=[r['index'] for r in rank_candidates(candidates,.1,.5)]
        altered=[dict(r,group='arbitrary',gt_tp=1000-i,task_id='unused') for i,r in enumerate(candidates)]
        second=[r['index'] for r in rank_candidates(altered,.1,.5)]
        self.assertEqual(first[:5],second[:10][:5]);self.assertEqual(first[0],11)
