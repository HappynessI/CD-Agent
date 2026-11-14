"""Verify oracle against exhaustive subsets and single-operation easy cases."""
import itertools
import unittest
import numpy as np
from scipy import ndimage as ndi
from revise_t5_review_v3 import simple_edit_oracles, analyse
from revise_whu_official import counts, metrics
from test_data_revision import DataRevisionTest


class OracleTest(unittest.TestCase):
    def test_deletion_matches_all_subsets(self):
        g=np.zeros((40,40),bool);g[2:8,2:8]=True;g[15:20,15:20]=True
        p=g.copy();p[2:10,2:6]=True;p[28:35,28:35]=True
        lab,n=ndi.label(p,np.ones((3,3)));best=0
        for subset in itertools.product((False,True),repeat=n):
            q=np.isin(lab,[i+1 for i,include in enumerate(subset) if include]);best=max(best,metrics(counts(q,g))['f1'])
        self.assertAlmostEqual(simple_edit_oracles(g,p)['arbitrary_whole_component_deletion']['metrics']['f1'],best)

    def test_one_whole_object_fill_is_easy_even_with_partial_edge(self):
        g=np.zeros((100,100),bool);g[10:80,10:80]=True
        p=np.zeros_like(g);p[10:20,10:20]=True
        oracle=simple_edit_oracles(g,p)
        self.assertEqual(oracle['best_single_edit']['metrics']['f1'],1.)
        self.assertFalse(analyse(g,p)[2]['core_candidate'])


if __name__=='__main__':unittest.main()
