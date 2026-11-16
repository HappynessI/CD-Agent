import unittest
import numpy as np
from agent_baseline.analysis.n9_t3_data import geometry,example_reasons,support,query_reasons,source_partition,choose,components,component_geometry,FG

class GeometryTests(unittest.TestCase):
    def test_component_grouping_preserves_label_and_pixel_order(self):
        from scipy import ndimage as ndi
        mask=np.zeros((15,16),bool);mask[1:3,1:4]=True;mask[8:11,10:14]=True
        mask[5,5]=True;mask[6,6]=True
        labels,n=ndi.label(mask,FG)
        expected=[np.flatnonzero(labels==i) for i in range(1,n+1)]
        actual=components(mask)
        self.assertEqual(len(actual),len(expected))
        for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)
        self.assertEqual(components(np.zeros((3,3),bool)),[])
        ring=np.zeros((40,40),bool);ring[5:20,7:22]=True;ring[9:13,11:15]=False
        self.assertEqual(component_geometry(np.flatnonzero(ring),ring.shape),geometry(ring))

    def test_exact_limits_and_hole_fraction(self):
        e=np.ones((8,8),bool);self.assertEqual(example_reasons(geometry(e)),[])
        e=np.ones((10,10),bool);e[4:6,3:7]=False
        self.assertAlmostEqual(geometry(e)['hole_ratio'],.08);self.assertEqual(example_reasons(geometry(e)),[])
        e[3:7,3:7]=False;self.assertIn('example_hole_ratio_above0.1',example_reasons(geometry(e)))
        self.assertEqual(example_reasons(geometry(np.ones((64,64),bool))),[])
        self.assertIn('example_area_above4096',example_reasons(geometry(np.ones((65,65),bool))))

    def test_eight_connected_and_thin_bbox(self):
        e=np.eye(64,dtype=bool);g=geometry(e)
        self.assertEqual(g['components'],1);self.assertIn('example_bbox_fill_below0.25',example_reasons(g))

    def test_remaining_support_preserves_tiny_truth(self):
        mask=np.zeros((20,20),bool);mask[:8,:8]=True;mask[12:16,12:20]=True
        self.assertEqual(query_reasons(support(mask)),[])
        mask[:8,:8]=False
        self.assertEqual(support(mask)['largest_component'],32)
        self.assertIn('query_pixels_below64',query_reasons(support(mask)))
        small=np.zeros((30,30),bool);small[::3,::3]=True
        self.assertIn('query_no_component32',query_reasons(support(small)))

    def test_external_sources_are_fixed_disjoint_and_complete(self):
        names=[f'image_{i:03d}.png' for i in range(109)];parts=source_partition(names)
        self.assertEqual(list(map(len,parts.values())),[36,36,37])
        self.assertEqual(set().union(*map(set,parts.values())),set(names))
        self.assertFalse(set(parts['tune'])&set(parts['confirm']))
        self.assertEqual(parts,source_partition(list(reversed(names))))
        self.assertEqual(choose([1,2,3],'task'),choose([1,2,3],'task'))

if __name__=='__main__':unittest.main()
