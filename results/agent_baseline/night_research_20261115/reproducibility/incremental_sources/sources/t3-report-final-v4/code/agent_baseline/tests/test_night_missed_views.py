import unittest
import numpy as np
from agent_baseline.analysis.night_missed_views import split_images,render

class NightViewsTest(unittest.TestCase):
    def test_same_image_directions_do_not_cross_splits(self):
        rows=[dict(group=g,stem=str(n)) for g in ('second_appeared','second_disappeared','levir_any') for n in range(10)]
        split=split_images(rows)
        self.assertEqual(len(split),20)
        self.assertEqual(sum(v=='tune' for k,v in split.items() if k.startswith('second:')),5)
        self.assertEqual(split,split_images(list(reversed(rows))))

    def test_spotlight_holes_dark_and_blue_never_covers_candidate(self):
        mask=np.zeros((64,64),bool);mask[16:48,16:48]=True;mask[26:38,26:38]=False
        base=np.ones_like(mask);rgb=[np.full((64,64,3),200,np.uint8)]*2
        first,_=render(rgb,mask,base,[16,16,48,48],'V1')
        two,_=render(rgb,mask,base,[16,16,48,48],'V2')
        a,b=np.asarray(first[0]),np.asarray(two[0])
        self.assertTrue(np.all(a[192,192]<100))
        self.assertTrue(np.all(a[120,120]==200))
        self.assertTrue(np.array_equal(a[120,120],b[120,120]))
        self.assertGreater(int(b[192,192,2]),int(b[192,192,0]))

if __name__=='__main__':unittest.main()
