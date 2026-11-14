import unittest
import numpy as np
from agent_baseline.tools_t13 import Features
from agent_baseline.analysis.night_t33_retrieval import external_scores,geometry_indices


class CrossImageRetrievalTests(unittest.TestCase):
    def test_external_reference_coordinates_and_direction(self):
        masks=np.array([[[1,0],[0,0]],[[0,1],[0,0]]],bool)
        q1=Features(np.zeros((2,2,2),np.float32),(2,2),'t1')
        q2=Features(np.array([[[1,0],[0,1]],[[0,0],[0,0]]],np.float32),(2,2),'t2')
        r1=Features(np.zeros((1,1,2),np.float32),(3,3),'t1')
        r2=Features(np.array([[[0,1]]],np.float32),(3,3),'t2')
        support=np.ones((3,3),bool)
        scores=external_scores(masks,[q1,q2],[r1,r2],support)
        self.assertGreater(scores[1],scores[0])
        reversed_scores=external_scores(masks,[q1,q2],[r2,r1],support)
        self.assertLess(reversed_scores[1],reversed_scores[0])

    def test_geometry_compares_different_image_shapes(self):
        masks=np.ones((1,4,4),bool);reference=np.ones((2,2),bool)
        query=[np.full((4,4,3),100,np.uint8)]*2
        ref=[np.full((2,2,3),100,np.uint8)]*2
        self.assertEqual(geometry_indices(masks,query,reference,ref),[0])


if __name__=='__main__':unittest.main()
