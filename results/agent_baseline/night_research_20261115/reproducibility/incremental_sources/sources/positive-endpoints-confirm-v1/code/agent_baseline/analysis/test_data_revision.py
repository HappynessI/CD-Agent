"""Small offline regression tests for T5 admission and metric conventions (Slurm)."""
import unittest
import numpy as np
from revise_t5_review import analyse, instruction
from revise_whu_official import metrics, counts


class DataRevisionTest(unittest.TestCase):
    def test_no_error_and_empty_are_not_core(self):
        g=np.zeros((80,80),bool);g[10:40,10:40]=True
        self.assertFalse(analyse(g,g)[2]['core_candidate'])
        self.assertFalse(analyse(~g & False,~g & False)[2]['core_candidate'])
        self.assertEqual(metrics(counts(g,g))['f1'],1.)

    def test_single_whole_object_add_or_delete_excluded(self):
        g=np.zeros((80,80),bool);g[10:40,10:40]=True
        p=np.zeros_like(g)
        self.assertFalse(analyse(g,p)[2]['core_candidate'])
        self.assertFalse(analyse(p,g)[2]['core_candidate'])

    def test_shifted_partial_object_mixed_feedback(self):
        g=np.zeros((80,80),bool);g[10:50,10:50]=True
        p=np.zeros_like(g);p[10:50,15:55]=True
        _,feedback,qc=analyse(g,p)
        self.assertTrue(qc['core_candidate'])
        self.assertEqual({e['type'] for e in feedback},{'missed','false_positive'})
        for e in feedback:
            x,y=e['point'];self.assertTrue(e['region'][y,x])
            self.assertTrue(e['partial_object'])
            self.assertTrue(g[y,x] != p[y,x])
        text=instruction('T5.3',feedback,g.shape)
        self.assertIn('adjust its outline',text)
        self.assertNotIn('keep the rest',text)

    def test_identical_pixel_counts_for_perfect_prediction(self):
        g=np.zeros((30,30),bool);g[4:10,4:10]=True
        self.assertEqual(counts(g,g),dict(tp=36,fp=0,fn=0))
        self.assertEqual(metrics(counts(g,g))['iou'],1.)


if __name__=='__main__':unittest.main()
