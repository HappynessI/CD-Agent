import unittest
from unittest.mock import patch
import numpy as np
from agent_baseline.analysis.n8_missed_prepare import reconstruct, directional_delta, SOURCES

class ReconstructionTest(unittest.TestCase):
    def test_direction_removes_sources(self):
        sources={s:np.zeros((32,32),bool) for s in SOURCES}
        sources['S4'][1:9,1:9]=True;sources['S1_02'][20:28,20:28]=True
        p=[np.ones((32,32))*.8,np.ones((32,32))*.1]
        with patch('agent_baseline.analysis.n8_missed_prepare.feature_difference',return_value=.3):
            rows,_,audit=reconstruct(sources,np.zeros((32,32),bool),'disappeared',p,None)
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['sources'],['S1_02'])
        self.assertAlmostEqual(rows[0]['semantic_delta'],.7)
        self.assertNotIn('S4',audit['enabled_sources'])
    def test_dilation_and_fragment_cutoff(self):
        sources={s:np.zeros((32,32),bool) for s in SOURCES}
        sources['S1_02'][6:16,6:16]=True;sources['S2'][25:29,25:31]=True
        base=np.zeros((32,32),bool);base[10:12,10:12]=True
        with patch('agent_baseline.analysis.n8_missed_prepare.feature_difference',return_value=.3):
            rows,_,_=reconstruct(sources,base,'any',[np.zeros(base.shape),np.ones(base.shape)],None)
        self.assertEqual(len(rows),1)
        self.assertFalse(rows[0]['mask'][7,10])  # Euclidean distance 3
        self.assertTrue(rows[0]['mask'][6,10])   # Euclidean distance 4
        self.assertTrue(rows[0]['mask'][7,7])    # Diagonal distance >3
    def test_semantic_direction(self):
        a=np.array([.8,.1]);b=np.array([.1,.8])
        np.testing.assert_allclose(directional_delta(a,b,'any'),[.7,.7])
        np.testing.assert_allclose(directional_delta(a,b,'appeared'),[-.7,.7])
