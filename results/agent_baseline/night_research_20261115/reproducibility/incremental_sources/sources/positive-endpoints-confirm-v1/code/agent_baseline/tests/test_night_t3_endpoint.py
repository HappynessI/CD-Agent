import unittest
import numpy as np
from agent_baseline.analysis.night_t3_endpoint import representations


class EndpointTests(unittest.TestCase):
    def test_same_delta_can_have_different_endpoints(self):
        a=np.array([[1.,0.],[11.,10.]])
        b=np.array([[0.,1.],[10.,11.]])
        result=representations(a,b,a[0],b[0])
        np.testing.assert_allclose(result['delta'],[1.,1.])
        self.assertGreater(result['endpoint'][0],result['endpoint'][1])
        np.testing.assert_allclose(result['equal_joint'],.5*(result['delta']+result['endpoint']))

    def test_only_exemplar_swap_reverses_delta(self):
        a=np.eye(2);b=a[::-1].copy()
        result=representations(a,b,a[0],b[0])
        np.testing.assert_allclose(result['delta_example_swapped'],-result['delta'])
        np.testing.assert_allclose(result['endpoint_example_swapped'],[0.,1.])


if __name__=='__main__':unittest.main()
