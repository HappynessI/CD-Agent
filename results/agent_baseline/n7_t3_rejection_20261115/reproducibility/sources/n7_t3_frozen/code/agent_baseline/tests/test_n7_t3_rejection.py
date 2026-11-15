import unittest
from agent_baseline.analysis.n7_t3_rejection import GRID, threshold_indices, select_threshold


class RejectionSelectionTests(unittest.TestCase):
    def test_no_backfill_and_inclusive_threshold(self):
        self.assertEqual(threshold_indices(list(range(10)), [0.2]*10, 0.2), list(range(8)))
        self.assertEqual(threshold_indices(list(range(10)), [0.1]*8+[1.,1.], 0.2), [])

    def test_tune_only_and_tp_constraint(self):
        row=dict(split='tune',methods={'delta_top8':{'counts':[100,100,0]}},grid={})
        for t in GRID:row['grid']['delta_threshold_'+str(t)]={'counts':[100,100,0]}
        row['grid']['delta_threshold_0.4']={'counts':[95,20,5]}
        row['grid']['delta_threshold_0.6']={'counts':[94,0,6]}
        self.assertEqual(select_threshold([row],'delta')['threshold'],.4)
        row['split']='confirm'
        with self.assertRaises(ValueError):select_threshold([row],'delta')

    def test_ties_keep_lower_threshold(self):
        row=dict(split='tune',methods={'joint_top8':{'counts':[10,10,0]}},
                 grid={'joint_threshold_'+str(t):{'counts':[10,10,0]} for t in GRID})
        self.assertEqual(select_threshold([row],'joint')['threshold'],-1.)


if __name__=='__main__':unittest.main()
