import unittest
import numpy as np
from agent_baseline.tools_t13 import Features,pool_features,cosine
from agent_baseline.analysis.night_t3_patchset import support_weights,patch_scores,ranking


class PatchsetTests(unittest.TestCase):
    def test_support_reproduces_original_pooling_and_delta(self):
        rng=np.random.default_rng(7);shape=(7,9);grid=(3,4);idx=np.array([0,1,10,20,33,62]);other=np.array([2,15,16,38])
        a=Features(rng.normal(size=(*grid,5)).astype(np.float32),shape,'t1')
        b=Features(rng.normal(size=(*grid,5)).astype(np.float32),shape,'t2')
        w=support_weights(idx,shape,grid);v=support_weights(other,shape,grid)
        np.testing.assert_allclose(w.sum(),1,atol=1e-7)
        for f in (a,b):np.testing.assert_allclose(w@f.data.reshape(-1,5),pool_features(f,idx),atol=1e-7)
        direct=cosine(pool_features(b,idx)-pool_features(a,idx),pool_features(b,other)-pool_features(a,other))
        delta=(b.data-a.data).reshape(-1,5)
        np.testing.assert_allclose(cosine(w@delta,v@delta),direct,atol=1e-6)
        self.assertEqual(ranking([.5,.5,.2]),[0,1,2])

    def test_single_patch_equals_delta_and_swap(self):
        delta=np.array([[1.,2.],[-2.,1.],[2.,3.]])
        score,swap=patch_scores(delta,[np.array([1.,0.,0.])],np.array([0.,0.,1.]))
        self.assertAlmostEqual(score[0],float(cosine(delta[0],delta[2])),places=6)
        self.assertAlmostEqual(swap[0],-score[0],places=6)

    def test_zero_mass_not_redistributed(self):
        delta=np.array([[1.,0.],[0.,0.]])
        score,swap=patch_scores(delta,[np.array([.25,.75])],np.array([1.,0.]))
        self.assertAlmostEqual(score[0],.625)
        self.assertAlmostEqual(swap[0],-.625)
        score,_=patch_scores(delta,[np.array([0.,1.])],np.array([1.,0.]))
        self.assertEqual(score[0],0)


if __name__=='__main__':unittest.main()
