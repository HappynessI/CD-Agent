import unittest
import numpy as np
from agent_baseline.analysis.night_t3_r1_joint import joint_order


class JointRankTests(unittest.TestCase):
    def test_endpoint_breaks_same_delta_without_new_candidates(self):
        a=np.array([[11.,10.],[1.,0.],[1.,0.]])
        b=np.array([[10.,11.],[0.,1.],[0.,1.]])
        order,scores=joint_order(a,b,a[1],b[1])
        np.testing.assert_allclose(scores['delta'],[1.,1.,1.])
        np.testing.assert_allclose(scores['equal_joint'],.5*scores['delta']+.5*scores['endpoint'])
        self.assertEqual(order,[1,2,0])
        self.assertEqual(sorted(order),list(range(len(a))))


if __name__=='__main__':unittest.main()
