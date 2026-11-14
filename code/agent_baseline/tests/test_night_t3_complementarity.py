import unittest
import numpy as np
from agent_baseline.analysis.night_t3_complementarity import regions,summarize,METHODS
from agent_baseline.analysis.t3_routes import pixel_counts


class ComplementarityTests(unittest.TestCase):
    def test_disjoint_partition_and_fixed_union_counts(self):
        sam=np.array([[1,1],[0,0]],bool);dense=np.array([[0,1],[1,0]],bool)
        truth=np.array([[1,0],[1,1]],bool);ex=np.zeros((2,2),bool);parts=regions(sam,dense)
        np.testing.assert_array_equal(sum(parts[k].astype(int) for k in METHODS[:3]),parts['union'].astype(int))
        counts={k:pixel_counts(v,truth,ex) for k,v in parts.items()}
        self.assertEqual(counts['union'],[2,1,1])
        self.assertEqual(sum(v[0] for k,v in counts.items() if k!='union'),2)
        self.assertEqual(sum(v[1] for k,v in counts.items() if k!='union'),1)
        np.testing.assert_array_equal(sam,[[True,True],[False,False]])

    def test_empty_input_regions_and_grouping(self):
        empty=np.zeros((2,2),bool);parts=regions(empty,empty)
        self.assertTrue(all(not v.any() for v in parts.values()))
        record=dict(group='地表类',remaining_gt_pixels=0,scores=[dict(method=k,counts=[0,0,0]) for k in METHODS])
        rows=summarize([record]);row=next(r for r in rows if r['support']=='empty' and r['group']=='全部')
        self.assertEqual(row['tasks'],1);self.assertEqual(row['methods']['union']['fp'],0)


if __name__=='__main__':unittest.main()
