import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from agent_baseline.analysis.trace_extended import behavioral_events,semantic_signature,spatial_verifier


class ExtendedTraceTests(unittest.TestCase):
    def test_equivalence_preserves_coordinates_and_state(self):
        sig=lambda args:semantic_signature(dict(tool='view',args=args))
        self.assertNotEqual(sig(dict(region=[0,1,2,3])),sig(dict(region=[1,0,3,2])))
        self.assertEqual(sig(dict(ids=[2,1])),sig(dict(ids=[1,2])))
        self.assertNotEqual(sig(dict(set_current=True)),sig(dict(set_current=False)))

    def test_cycles_abandoned_candidate_and_coordinate_errors(self):
        rows=[]
        for i,name in enumerate(('view','inspect','view','inspect'),1):
            rows.append(dict(step=i,calls=[dict(tool=name,args={'mask':'mask#0'},ok=True)]))
        rows.append(dict(step=5,calls=[dict(tool='view',args={'region':[0,0,768,768]},ok=False,error_code='bad_args',
            result_text=json.dumps(dict(message='region must be an ordered box inside the full image.')),
            candidate_eval=[dict(handle='mask#1',f1=.8)])]))
        result=behavioral_events(dict(image_shape=[512,512]),dict(eval=dict(f1=.5),final_handle='mask#2'),rows)
        kinds=[e['kind'] for e in result['events']]
        for kind in ('call_cycle','equivalent_repeated_call','coordinate_out_of_bounds','better_candidate_abandoned'):self.assertIn(kind,kinds)
        self.assertAlmostEqual(result['final_vs_best_candidate'],-.3)

    def test_exact_outline_override_preserves_remove_region_ids(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'masks').mkdir()
            original=np.zeros((6,6),bool);original[0,0]=True;original[4:6,4:6]=True
            remaining=original.copy();remaining[0,0]=False
            empty=np.zeros_like(original)
            for name,a in [('mask_0',original),('mask_1',remaining),('mask_2',empty),('gt',remaining)]:
                Image.fromarray(a.astype('uint8')*255).save((p/'masks'/name).with_suffix('.png'))
            steps=[dict(step=1,calls=[dict(tool='remove_region',args=dict(mask='mask#0',ids=[1]),handles_created=['mask#1'])]),
                dict(step=2,calls=[dict(tool='ask',result_text=json.dumps(dict(results=[dict(mask='mask#1',id=2,answer='yes')])))]),
                dict(step=3,calls=[dict(tool='remove_region',args=dict(mask='mask#1',ids=[2]),handles_created=['mask#2'])])]
            result=spatial_verifier(p,dict(gt_mask_path=str(p/'masks/gt.png')),dict(final_handle='mask#2'),steps)
            self.assertEqual(result['decisions']['yes_remove'],1)
            self.assertEqual(result['events'][0]['area'],4)
            self.assertEqual(result['override_pixel_counterfactual_delta'],-1)


if __name__=='__main__':unittest.main()
