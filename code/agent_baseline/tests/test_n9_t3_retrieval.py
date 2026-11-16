import inspect
import unittest
import numpy as np
from agent_baseline.tools_t13 import Features
from agent_baseline.analysis.n9_t3_retrieval import clean_candidates,scores,geometry_indices,retrieve
from agent_baseline.analysis.n9_t3_fixed import evaluate

class RetrievalTests(unittest.TestCase):
    def test_no_n8_micro_overlap_size_gate(self):
        ex=np.zeros((10,10),bool);ex[:2,:5]=True
        masks=np.zeros((2,10,10),bool);masks[0]=ex;masks[1,0,0]=True;masks[1,4,1:8]=True
        clean,ids,audit=clean_candidates(masks,ex,True)
        self.assertEqual(ids,[1]);self.assertEqual(int(clean[0].sum()),7)
        self.assertEqual(audit[1]['intersection'],1)
        self.assertFalse(clean[0,0,0]);self.assertTrue(audit[0]['removed'])

    def test_cross_image_never_subtracts_coincident_coordinates(self):
        ex=np.ones((3,3),bool);masks=np.ones((1,3,3),bool)
        clean,ids,audit=clean_candidates(masks,ex,False)
        np.testing.assert_array_equal(clean,masks);self.assertEqual(ids,[0]);self.assertEqual(audit,[])
        result=evaluate(clean,[0],masks[0],np.zeros_like(ex))
        self.assertEqual(result['counts'],[9,0,0])

    def test_net_region_features_are_recomputed(self):
        ex=np.zeros((4,4),bool);ex[0,:]=True
        mask=np.zeros((1,4,4),bool);mask[0,0,0]=True;mask[0,2:4,:]=True
        clean,_,_=clean_candidates(mask,ex,True)
        first=np.zeros((4,4,2),np.float32);last=np.zeros_like(first)
        last[0,:,0]=100;last[2:4,:,1]=1
        features=[Features(first,(4,4),'t1'),Features(last,(4,4),'t2')]
        before=scores(mask,features,features,ex);after=scores(clean,features,features,ex)
        self.assertGreater(before['delta'][0],.9);self.assertAlmostEqual(float(after['delta'][0]),0.)

    def test_geometry_uses_new_reference_support(self):
        images=[np.zeros((12,12,3),np.uint8)]*2
        masks=np.zeros((1,12,12),bool);masks[0,:2,:2]=True
        small=np.zeros((12,12),bool);small[:2,:2]=True
        large=np.ones((12,12),bool)
        self.assertEqual(geometry_indices(masks,images,small,images),[0])
        self.assertEqual(geometry_indices(masks,images,large,images),[])
        self.assertEqual(list(inspect.signature(retrieve).parameters),['query_images','reference_images','example','query_backend','reference_backend','same_image'])

if __name__=='__main__':unittest.main()
