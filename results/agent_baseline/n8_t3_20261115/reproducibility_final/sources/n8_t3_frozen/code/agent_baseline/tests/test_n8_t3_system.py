import unittest
import numpy as np
from agent_baseline.analysis.n8_t3_system import construct_pool,relative_margin,passes_gate,evidence_card


class SystemTests(unittest.TestCase):
    def test_relative_margin_and_empty(self):
        self.assertEqual(relative_margin([4.],[0])['z'],0)
        self.assertEqual(relative_margin([],[])['z'],0)
        self.assertFalse(passes_gate(0,0,0,0,0))
        self.assertAlmostEqual(relative_margin([3.,1.,2.],[0,2,1])['z'],1.5/(.5+1e-6))

    def test_union_reconstructs_candidates_and_deduplicates_exact_masks(self):
        a=np.zeros((3,3),bool);a[0,0]=True;b=np.zeros_like(a);b[2,2]=True
        p=np.zeros((7,3,3));p[1]=1;q=np.zeros_like(p);q[5]=1
        masks,sources=construct_pool(np.stack([a,a]),a|b,[p,q],dict(structural=True,predicted_pair=[1,5]),'hybrid_union_0.7')
        self.assertEqual(len(masks),2);self.assertEqual(len(sources[0]),3)
        self.assertTrue(masks.any(axis=0)[2,2])

    def test_evidence_card_has_two_predicted_lines(self):
        means=[[0,1,0,0,0,0,0],[0,0,0,0,0,1,0]]
        card=evidence_card(dict(area=20,predicted_pair=[1,5],example_mean_probabilities=means),
            dict(area=30,predicted_pair=[1,5],mean_probabilities=means,similarity=.8,ptrans=.9),'delta')
        self.assertEqual(len(card.splitlines()),2);self.assertIn('building(1.0000)',card)


if __name__=='__main__':unittest.main()
