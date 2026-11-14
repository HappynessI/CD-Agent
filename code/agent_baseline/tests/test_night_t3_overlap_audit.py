import unittest
import numpy as np
from agent_baseline.analysis.night_t3_overlap_audit import candidate_geometry


class OverlapTests(unittest.TestCase):
    def test_residual_support_and_counts_exclude_example_only(self):
        mask=np.array([[1,1],[1,0]],bool);ex=np.array([[1,0],[0,0]],bool);truth=np.array([[1,1],[0,1]],bool)
        r=candidate_geometry(mask,ex,truth)
        self.assertEqual((r['candidate_pixels'],r['example_overlap_pixels'],r['remaining_support_pixels']),(3,1,2))
        self.assertEqual(r['raw_counts_including_example'],[2,1,1])
        self.assertEqual(r['remaining_counts'],[1,1,1]);self.assertEqual(r['category'],'overlap_nonempty')
        self.assertEqual(candidate_geometry(ex,ex,truth)['category'],'empty_residual')
        self.assertEqual(candidate_geometry(mask&~ex,ex,truth)['category'],'no_overlap')


if __name__=='__main__':unittest.main()
