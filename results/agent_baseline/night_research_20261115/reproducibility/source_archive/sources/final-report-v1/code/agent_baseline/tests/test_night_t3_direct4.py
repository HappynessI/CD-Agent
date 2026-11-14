import unittest
import numpy as np
from agent_baseline.analysis.night_t3_direct4 import four_views,SYSTEM,T3_QUESTION


class DirectFourTests(unittest.TestCase):
    def test_explicit_roles_dates_and_no_class_pair_input(self):
        images=[np.full((80,80,3),70,np.uint8),np.full((80,80,3),140,np.uint8)]
        ex=np.zeros((80,80),bool);ex[10:20,10:20]=True
        candidate=np.zeros_like(ex);candidate[50:60,50:60]=True
        result=four_views(images,ex,candidate)
        self.assertEqual(len(result),4)
        for index,(label,view) in enumerate(result):
            self.assertNotIn('candidate contour',label)
            self.assertIn('support contour',label)
            self.assertTrue(label.startswith('Example' if index<2 else 'Candidate'))
            self.assertIn('T1' if index%2==0 else 'T2',label)
            self.assertEqual(int(np.asarray(view)[0,0,1]),70 if index%2==0 else 140)
        swapped=four_views(images,ex,candidate,swap_example=True)
        self.assertEqual(int(np.asarray(swapped[0][1])[0,0,1]),140)
        for i in (2,3):np.testing.assert_array_equal(np.asarray(swapped[i][1]),np.asarray(result[i][1]))
        self.assertNotIn('building',SYSTEM+T3_QUESTION)
        self.assertIn('example_t1',T3_QUESTION)
        self.assertIn('candidate_t2',T3_QUESTION)


if __name__=='__main__':unittest.main()
