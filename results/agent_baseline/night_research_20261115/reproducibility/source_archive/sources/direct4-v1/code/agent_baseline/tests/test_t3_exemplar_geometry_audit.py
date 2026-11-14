import unittest
import numpy as np
from render import crop_box
from agent_baseline.analysis.t3_exemplar_geometry_audit import geometry, cross_summary


class GeometryAuditTests(unittest.TestCase):
    def test_hole_and_open_channel_are_distinguished(self):
        mask=np.zeros((80,80),bool); mask[20:40,20:40]=True; mask[25:35,25:35]=False
        result=geometry(mask,mask)
        self.assertEqual((result['hole_count'],result['hole_pixels']),(1,100))
        self.assertEqual(result['remaining_gt_pixels'],0)
        self.assertEqual(result['crops']['close']['box'],crop_box([20,20,40,40],mask.shape))
        self.assertAlmostEqual(result['crops']['close']['support_fraction'],300/6400)
        mask[20:26,29]=False
        self.assertEqual(geometry(mask,mask)['hole_count'],0)

    def test_touching_exterior_is_not_hole(self):
        mask=np.zeros((100,100),bool); mask[:10,:10]=True
        truth=mask.copy(); truth[80,80]=True
        result=geometry(mask,truth)
        self.assertEqual(result['remaining_gt_pixels'],1)
        self.assertTrue(result['support_touches_image_edge']['top'])
        self.assertTrue(result['support_touches_image_edge']['left'])
        self.assertEqual(result['hole_pixels'],0)
        self.assertEqual(result['crops']['close']['exemplar_visible_fraction'],1.)

    def test_unknown_counts_but_missing_parse_does_not(self):
        mask=np.ones((10,10),bool); base=geometry(mask,mask)
        rows=[dict(base,group='目标类',semantic_available=True,pair_label_agreement=False),
              dict(base,group='目标类',semantic_available=False,pair_label_agreement=None),
              dict(base,group='地表类',semantic_available=True,pair_label_agreement=True)]
        total=cross_summary(rows)[0]
        self.assertEqual(total['tasks'],3)
        self.assertEqual(total['semantic_available'],2)
        self.assertEqual(total['pair_label_agreement'],.5)


if __name__=='__main__': unittest.main()
