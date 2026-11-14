import unittest
import numpy as np
from agent_baseline import instances
from agent_baseline.analysis.night_sparse_overlap import sparse_overlaps
from agent_baseline.analysis.night_missed_candidates import rematch_policies,rematch,MATCH_POLICIES


class SparseOverlapTest(unittest.TestCase):
    def test_exact_pairs_and_all_masks_against_original_match(self):
        shape=(24,24)
        def box(y,x,h,w):
            m=np.zeros(shape,bool);m[y:y+h,x:x+w]=True;return m
        # Intersection >255 catches accidental boolean/uint8 multiplication.
        aa=[box(0,0,20,20),box(20,20,2,2),box(22,0,2,2)]
        bb=[box(0,0,20,20),box(0,0,10,10),box(10,10,10,10),box(20,20,2,2)]
        found=dict(t1=instances.from_dense(aa,[1]*len(aa)),t2=instances.from_dense(bb,[1]*len(bb)))
        stats={};pairs=sparse_overlaps(found,shape,stats)
        self.assertEqual(pairs,instances.overlaps(found['t1'],found['t2']))
        self.assertEqual(pairs[0][2],400);self.assertEqual(stats['dtype'],'int32')
        base=box(0,0,1,1)
        for direction in ('any','appeared','disappeared'):
            result=rematch_policies(found,base,direction,'csr')
            for key,threshold,mode in MATCH_POLICIES:
                self.assertTrue(np.array_equal(result[key],rematch(found,base,direction,threshold,mode)))

    def test_irregular_and_empty_sets_match_reference_pairs(self):
        rng=np.random.default_rng(1016);shape=(17,19)
        aa=rng.random((7,*shape))>.85;bb=rng.random((9,*shape))>.75
        found=dict(t1=instances.from_dense(aa,[1]*len(aa)),t2=instances.from_dense(bb,[1]*len(bb)))
        self.assertEqual(sparse_overlaps(found,shape),instances.overlaps(found['t1'],found['t2']))
        for side in ('t1','t2'):
            empty=dict(found);empty[side]=[]
            self.assertEqual(sparse_overlaps(empty,shape),[])


if __name__=='__main__':unittest.main()
