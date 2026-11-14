import unittest
from tempfile import TemporaryDirectory
import numpy as np
from agent_baseline.state import Session
from agent_baseline.tools import Toolbox, session_schemas


class MissedIntegrationTests(unittest.TestCase):
    def test_tool_is_opt_in_and_keeps_answer(self):
        class Provider:
            def candidates(self, current, limit):
                return [dict(id=1, idx=np.array([5,6]), area=2, score=1., bbox=[1,1,3,2],
                             sources=['S4'], source_rank=1)]
        with TemporaryDirectory() as directory:
            s=Session((4,4),np.zeros((4,4),bool))
            t=Toolbox([np.zeros((4,4,3),np.uint8)]*2,directory,session=s)
            self.assertNotIn('find_missed',[x['function']['name'] for x in session_schemas(s)])
            self.assertFalse(t.execute('find_missed',{}).public['ok'])
            t.missed_provider=Provider();before=s.current_mask
            result=t.execute('find_missed',dict(limit=5))
            self.assertTrue(result.public['ok'])
            self.assertEqual(s.current_mask,before)
            self.assertIsNone(s.submitted)
            self.assertIn('find_missed',[x['function']['name'] for x in session_schemas(s,missed_candidates=True)])

    def test_explicit_renderer_is_used_for_verification(self):
        class Backend:
            context=True
            rendered=0
            def render(self,rgb,mask,bbox):
                self.rendered+=1
                return ['close1','close2','wide1','wide2']
            def __call__(self,pairs,question):
                assert pairs==[['close1','close2','wide1','wide2']]
                return [dict(answer='yes',reason='fixture')],{}
        with TemporaryDirectory() as directory:
            mask=np.zeros((4,4),bool);mask[1:3,1:3]=True
            s=Session((4,4),mask);backend=Backend()
            t=Toolbox([np.zeros((4,4,3),np.uint8)]*2,directory,session=s,ask=backend)
            item=s.get(s.current_mask)
            result=t.verify([(s.current_mask,item,item.components[0])],'ask')
            self.assertEqual(backend.rendered,1)
            self.assertEqual(result[0]['answer'],'yes')
            np.testing.assert_array_equal(s.get(s.current_mask).data,mask)

if __name__=='__main__':unittest.main()
