import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from agent_baseline.tools import Toolbox
from agent_baseline.tests.helpers import rgb
from agent_baseline.tests.test_tools_t13 import FakeVision, SHAPE

class Gate6Tests(unittest.TestCase):
    def test_view_preserves_old_fields_and_pixels(self):
        with tempfile.TemporaryDirectory() as d:
            old = Toolbox(rgb(), Path(d)/'old')
            new = Toolbox(rgb(), Path(d)/'new', gate6=True)
            for region in (None, [2, 3, 20, 25]):
                a = old.execute('view', {'region':region} if region else {})
                b = new.execute('view', {'region':region} if region else {})
                self.assertTrue(b.public['ok'])
                self.assertEqual(b.public.pop('original_size'), [48,32])
                self.assertIn('original-image', b.public.pop('coordinate_system'))
                self.assertEqual(a.public,b.public)
                for x,y in zip(a.images,b.images):
                    self.assertEqual(Path(x).read_bytes(),Path(y).read_bytes())

    def test_quantiles_and_empty(self):
        with tempfile.TemporaryDirectory() as d:
            outputs=[]
            for enabled in (False,True):
                box=Toolbox([np.zeros((*SHAPE,3),np.uint8)]*2,Path(d),tools_t13=True,vision=FakeVision(),gate6=enabled)
                def run(name,**args):
                    result=box.execute(name,args).public
                    self.assertTrue(result['ok'],result)
                    return result
                props=run('segment_everything',time='t1')['props']
                a=run('extract_features',time='t1')['feat']
                b=run('extract_features',time='t2')['feat']
                outputs.append(run('compare_features',props=props,feat_t1=a,feat_t2=b))
                empty=run('select',set=props,area_min=999)['props']
                out=run('compare_features',props=empty,feat_t1=a,feat_t2=b)
                if enabled:
                    self.assertEqual(list(out['score_quantiles'].values()),[None]*7)
            quantiles=outputs[1].pop('score_quantiles')
            self.assertAlmostEqual(quantiles['max'],1.)
            self.assertEqual(list(quantiles.values()),sorted(quantiles.values()))
            self.assertEqual(outputs[0],outputs[1])
