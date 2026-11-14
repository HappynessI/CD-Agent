import unittest
import tempfile
from pathlib import Path
import numpy as np
from agent_baseline import instances
from agent_baseline.analysis.night_missed_candidates import rematch,gate_features,visual_instances,rematch_policies,MATCH_POLICIES
from agent_baseline.analysis.missed_sources import change_candidates,pack,save_npz

class CandidateGatesTest(unittest.TestCase):
    def test_shared_intersections_equal_original_for_all_directions(self):
        def box(y,x,h,w):
            m=np.zeros((12,12),bool);m[y:y+h,x:x+w]=True;return m
        # One-to-many, asymmetric quarter coverage, disjoint and exact matches.
        a=[box(0,0,4,4),box(6,6,2,2),box(9,0,1,1)]
        b=[box(0,0,2,2),box(2,2,2,2),box(6,6,2,2),box(9,9,2,2)]
        base=box(0,0,1,1)
        found=dict(t1=instances.from_dense(a,[1]*len(a)),t2=instances.from_dense(b,[1]*len(b)))
        for direction in ('any','appeared','disappeared'):
            fast=rematch_policies(found,base,direction)
            for key,threshold,mode in MATCH_POLICIES:
                with self.subTest(direction=direction,policy=key):
                    self.assertTrue(np.array_equal(fast[key],rematch(found,base,direction,threshold,mode)))
        for empty in ('t1','t2'):
            sparse=dict(found);sparse[empty]=[]
            fast=rematch_policies(sparse,base,'any')
            for key,threshold,mode in MATCH_POLICIES:
                self.assertTrue(np.array_equal(fast[key],rematch(sparse,base,'any',threshold,mode)))

    def test_all_reference_cache_union_matches_original_s5(self):
        # Original SAM3.visual pools every reference/seed before matching.
        # An opposite-date match can come from a different reference.
        a=np.zeros((8,8),bool);a[:2,:2]=True
        b=np.zeros_like(a);b[5:7,5:7]=True
        empty=[]
        cache={'t1_0001_t1':[a,a], 't1_0001_t2':empty,
               't2_0001_t1':empty, 't2_0001_t2':[a,b]}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);folder=root/'gpu_cache/visual/levir_any/example'
            folder.mkdir(parents=True)
            for name,masks in cache.items():
                save_npz(folder/(name+'.npz'),masks=pack(masks,a.shape),shape=a.shape)
            found,count=visual_instances(root,'levir_any','example',a.shape)
            self.assertEqual(count,4)
            self.assertEqual(len(found['t1']),1)
            base=np.zeros_like(a);base[5,5]=True
            expected=instances.to_mask(instances.from_dense(
                change_candidates([a,a],[a,b],base,'any'),[1]),a.shape)
            self.assertTrue(np.array_equal(rematch(found,base,'any',.25,'overlap'),expected))
            self.assertEqual(int(expected.sum()),3)

    def test_overlap_changes_presence_before_subtracting_start(self):
        small=np.zeros((8,8),bool);small[:2,:2]=True
        large=np.zeros_like(small);large[:4,:4]=True
        found=dict(t1=instances.from_dense([small],[1]),t2=instances.from_dense([large],[1]))
        start=np.zeros_like(small);start[3,3]=True
        self.assertFalse(rematch(found,start,'appeared',.25,'overlap').any())
        self.assertEqual(int(rematch(found,start,'appeared',.50,'overlap').sum()),15)

    def test_rgb_features_are_image_only(self):
        mask=np.ones((4,4),bool)
        images=[np.zeros((4,4,3),np.uint8),np.full((4,4,3),255,np.uint8)]
        row=gate_features([mask],images,None)[0]
        self.assertAlmostEqual(row['rgb_mean_rms'],1.)
        self.assertAlmostEqual(row['rgb_pixel_rms'],1.)
        self.assertIsNone(row['dino'])

if __name__=='__main__':unittest.main()
