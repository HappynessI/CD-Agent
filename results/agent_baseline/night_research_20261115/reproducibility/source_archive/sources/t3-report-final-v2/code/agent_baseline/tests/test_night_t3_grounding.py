import unittest
import numpy as np
from agent_baseline.analysis.night_t3_grounding import spotlight,views,fixed_split


class GroundingTests(unittest.TestCase):
    def test_spotlight_preserves_support_and_darkens_holes(self):
        image=np.full((80,80,3),200,np.uint8)
        mask=np.zeros((80,80),bool);mask[20:60,20:60]=True;mask[30:50,30:50]=False
        result=spotlight(image,mask)
        self.assertTrue(np.array_equal(result[mask],image[mask]))
        self.assertTrue((result[~mask]==60).all())
        self.assertEqual(int(result[35,35,0]),60)
        self.assertTrue((image==200).all())

    def test_cross_date_information_is_explicit(self):
        images=[np.full((80,80,3),200,np.uint8),np.full((80,80,3),100,np.uint8)]
        support=np.ones((80,80),bool)
        b1=views(images,support,0,'B1');b2=views(images,support,0,'B2')
        self.assertEqual(len(b1),2);self.assertEqual(len(b2),6)
        self.assertTrue(all((np.asarray(v)==200).all() for _,v in b1))
        self.assertTrue((np.asarray(b2[-1][1])==100).all())
        self.assertIn('T2',b2[-1][0])

    def test_fixed_image_split_does_not_depend_on_row_order(self):
        rows=[dict(id=f'example{i}',image_t1=f'/fixed/image{i}.png') for i in range(60)]
        a=fixed_split(rows);b=fixed_split(list(reversed(rows)))
        self.assertEqual(a,b);self.assertEqual(sum(v=='tune' for v in a.values()),30)
        with self.assertRaises(ValueError):fixed_split(rows+[dict(id='duplicate',image_t1='/fixed/image0.png')])


if __name__=='__main__':unittest.main()
