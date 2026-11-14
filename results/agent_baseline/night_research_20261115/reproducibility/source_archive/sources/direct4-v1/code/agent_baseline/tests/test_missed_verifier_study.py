import unittest
import numpy as np
from agent_baseline.analysis.missed_verifier_study import SOURCES, ranked_candidates, metrics, recall

class MissedStudyTest(unittest.TestCase):
    def test_source_round_robin_deduplicates_exact_masks(self):
        a=np.zeros((8,8),bool);a[:2,:2]=True
        b=np.zeros_like(a);b[4:7,4:7]=True
        pool={s:[] for s in SOURCES};pool['S1_02']=[a];pool['S2']=[a,b]
        result=ranked_candidates(pool)
        self.assertEqual(len(result),2)
        self.assertEqual(result[0]['sources'],['S1_02','S2'])
        self.assertEqual(result[1]['source'],'S2')
    def test_recall_union_and_zero_denominator(self):
        missing=np.eye(4,dtype=bool)
        self.assertEqual(recall(missing,missing)['pixel_recall'],1)
        self.assertIsNone(recall(missing,np.zeros_like(missing))['pixel_recall'])
        self.assertAlmostEqual(metrics([2,1,1])['f1'],200/3)

if __name__=='__main__': unittest.main()
