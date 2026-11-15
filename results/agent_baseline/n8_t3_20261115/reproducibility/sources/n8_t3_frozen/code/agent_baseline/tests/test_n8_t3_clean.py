import unittest
import numpy as np
from agent_baseline.analysis.n8_t3_common import clean_pool


class CleaningTests(unittest.TestCase):
    def test_remove_before_ranking_and_net_support(self):
        ex=np.zeros((10,10),bool);ex[:2,:5]=True
        a=np.zeros_like(ex);a[:2,:]=True # overlap .5 => remove
        b=np.ones_like(ex) # overlap .1 => subtract, retain 90
        c=np.zeros_like(ex);c[5:,:]=True # disjoint => retain
        clean,ids,audit=clean_pool(np.stack([a,b,c]),ex)
        self.assertEqual(ids,[1,2]);self.assertFalse((clean&ex).any())
        self.assertEqual(int(clean[0].sum()),90);self.assertEqual(audit[0]['removed'],'overlap_or_iou')

    def test_exact_threshold_is_removed(self):
        ex=np.zeros((5,5),bool);ex.flat[:3]=True
        mask=np.zeros_like(ex);mask.flat[:20]=True
        clean,ids,_=clean_pool(np.stack([mask]),ex)
        self.assertEqual(ids,[]);self.assertEqual(clean.shape,(0,5,5))


if __name__=='__main__':unittest.main()
