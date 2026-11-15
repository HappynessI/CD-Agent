import unittest
import numpy as np
from agent_baseline.analysis.night_t3_mask_panel import panels
from agent_baseline.analysis.night_t3_grounding import raw_views


class MaskPanelTests(unittest.TestCase):
    def test_binary_holes_and_rgb_alignment(self):
        image=np.full((64,64,3),170,np.uint8)
        support=np.zeros((64,64),bool);support[10:54,10:54]=True;support[20:44,20:44]=False
        result=panels(image,support);raw=raw_views(image,support)
        self.assertEqual(len(result),4)
        self.assertTrue(np.array_equal(np.asarray(result[0][1]),np.asarray(raw[0][1])))
        self.assertTrue(np.array_equal(np.asarray(result[2][1]),np.asarray(raw[1][1])))
        mask=np.asarray(result[1][1]);self.assertEqual(int(mask[192,192,0]),0)
        self.assertEqual(int(mask[90,90,0]),255)
        self.assertEqual(set(np.unique(mask)),{0,255})
        self.assertTrue((image==170).all())


if __name__=='__main__':unittest.main()
