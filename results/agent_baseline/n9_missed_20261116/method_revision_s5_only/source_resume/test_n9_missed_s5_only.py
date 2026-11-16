import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from agent_baseline.analysis.n9_missed_s5_only import configurations,candidate_indices,equivalent_requests,cumulative_cost
from agent_baseline.analysis.n8_missed_prepare import reconstruct,SOURCES

class S5OnlyContracts(unittest.TestCase):
    def test_low_dino_directed_candidates_retained_any_still_strict(self):
        candidates=[dict(index=0,sources=['S1_02'],semantic_delta=.4,dino=.1,area=64),
                    dict(index=1,sources=['S2'],semantic_delta=.3,dino=.3,area=64)]
        config=configurations('demolish')[0]
        self.assertIsNone(config['dino']);self.assertEqual(candidate_indices(candidates,config),[1,0])
        self.assertEqual(candidate_indices(candidates,configurations('any')[0]),[])
        self.assertEqual([len(configurations(p)) for p in ('any','build','demolish')],[16,24,12])

    def test_s5_raw_source_gate_is_unchanged(self):
        sources={s:np.zeros((32,32),bool) for s in SOURCES}
        sources['S5'][2:10,2:10]=True;sources['S1_02'][20:28,20:28]=True
        with patch('agent_baseline.analysis.n8_missed_prepare.feature_difference',return_value=.1):
            rows,_,audit=reconstruct(sources,np.zeros((32,32),bool),'disappeared',[np.ones((32,32)),np.zeros((32,32))],None)
        self.assertEqual(audit['s5_after_dino020'],0)
        self.assertEqual(len(rows),1)
        self.assertEqual(candidate_indices(rows,configurations('demolish')[0]),[0])

    def test_reuse_checks_actual_bytes_text_and_complete_parameters(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d);a=d/'a';b=d/'b';a.write_bytes(b'image');b.write_bytes(b'image')
            original=dict(parameters=dict(model='qwen3.7-flash',seed=1,temperature=0),messages=[dict(role='user',content=[
                dict(type='image_url',image_url=dict(url=str(a))),dict(type='text',text='same question')])])
            expected=copy.deepcopy(original);expected['messages'][0]['content'][0]['image_url']['url']=str(b)
            self.assertTrue(equivalent_requests(expected,original))
            b.write_bytes(b'different');self.assertFalse(equivalent_requests(expected,original));b.write_bytes(b'image')
            expected['parameters']['seed']=2;self.assertFalse(equivalent_requests(expected,original))
            expected['parameters']['seed']=1;expected['messages'][0]['content'][1]['text']='changed'
            self.assertFalse(equivalent_requests(expected,original))

    def test_cost_parent_plus_new_usage_only(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            parent=Path(d);child=parent/'method_revision_s5_only'
            for root,cost in ((parent,1.),(child,.2)):
                (root/'usage').mkdir(parents=True);(root/'usage/a.json').write_text(json.dumps(dict(cost_cny=cost)))
            (child/'answers').mkdir();(child/'answers/reused.json').write_text(json.dumps(dict(cost_cny=1.)))
            self.assertAlmostEqual(cumulative_cost(child),1.2)

if __name__=='__main__':unittest.main()
