import unittest
import numpy as np
from agent_baseline.analysis.night_t3_segearth_dense import directed_dense,aggregate,METHODS
from agent_baseline.analysis.t3_routes import semantic_route


class DenseTests(unittest.TestCase):
    def test_background_direction_unknown_and_example(self):
        maps=[np.array([[0,1],[1,2]]),np.array([[2,2],[1,1]])];ex=np.array([[0,1],[0,0]],bool)
        self.assertFalse(directed_dense(maps,[1,2],ex).any())
        np.testing.assert_array_equal(directed_dense(maps,[2,1],ex),[[False,False],[False,True]])
        self.assertFalse(directed_dense(maps,[1,1],ex).any())
        self.assertFalse(directed_dense(maps,[None,2],ex).any())

    def test_routeA_foreground_example_and_common_argmax(self):
        p1=np.zeros((7,2,2));p2=np.zeros_like(p1);p1[0]=.8;p2[0]=.8
        p1[1]=.1;p2[2]=.1;p1[1,1,1]=.9;p2[2,1,1]=.9
        ex=np.array([[1,0],[0,0]],bool)
        pair,pred=semantic_route([p1,p2],ex)
        self.assertEqual(pair,[1,2])
        np.testing.assert_array_equal(pred,directed_dense([p1.argmax(0),p2.argmax(0)],pair,ex))
        self.assertEqual(int(pred.sum()),1)

    def test_original_positive_and_sensitivity_denominators(self):
        records=[dict(group='目标类',remaining_gt_pixels=n,scores=[dict(method=m,counts=[n,1,0]) for m in METHODS]) for n in (0,1,16)]
        rows=aggregate(records)
        sizes={r['support']:r['tasks'] for r in rows if r['group']=='全部'}
        self.assertEqual(sizes,dict(all=3,positive=2,empty=1,**{'remaining_gt>=16_diagnostic':1}))


if __name__=='__main__':unittest.main()
