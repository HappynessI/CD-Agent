import unittest
import numpy as np
from agent_baseline import instances
from agent_baseline.analysis.night_missed_candidates import rematch,gate_features

class CandidateGatesTest(unittest.TestCase):
    def test_overlap_changes_presence_before_subtracting_start(self):
        small=np.zeros((8,8),bool);small[:2,:2]=True
        large=np.zeros_like(small);large[:4,:4]=True
        found=dict(t1=instances.from_dense([small],[1]),t2=instances.from_dense([large],[1]))
        start=np.zeros_like(small);start[3,3]=True
        self.assertFalse(rematch(found,start,'appeared',.25,'overlap').any())
        self.assertEqual(int(rematch(found,start,'appeared',.50,'overlap').sum()),15)

    def test_rgb_features_are_image_only(self):
        mask=np.ones((4,4),bool)
        images=[np.zeros((4,4,3),np.uint8),np.full((4,4,3),255,np.uint8)]
        row=gate_features([mask],images,None)[0]
        self.assertAlmostEqual(row['rgb_mean_rms'],1.)
        self.assertAlmostEqual(row['rgb_pixel_rms'],1.)
        self.assertIsNone(row['dino'])

if __name__=='__main__':unittest.main()
