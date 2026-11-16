import unittest
import numpy as np
from agent_baseline.analysis.task_reconstruct_t3 import geometry,best_example,select_three

class HomogeneousT3Contracts(unittest.TestCase):
    def test_geometry_fills_holes_without_cropping_instance(self):
        mask=np.zeros((32,32),bool);mask[5:25,5:25]=True;mask[10:12,10:12]=False
        filled,g=geometry(mask)
        self.assertEqual(g['area'],400);self.assertEqual(g['raw_area'],396)
        self.assertAlmostEqual(g['hole_ratio'],.01);self.assertAlmostEqual(g['compactness'],1.)
        self.assertEqual(g['bbox'],[5,5,25,25]);self.assertTrue(filled[10,10])

    def test_full_remaining_keeps_nonmatching_instances(self):
        masks=[];records=[]
        for i in range(4):
            m=np.zeros((20,80),bool);m[:,i*20:i*20+10]=True
            filled,g=geometry(m);masks.append(filled)
            records.append(dict(id=i+1,**g,rgb_mean=[100,100,100] if i<3 else [255,0,0]))
        vectors=np.array([[1.,0.],[1.,0.],[1.,0.],[0.,1.]])
        best,failures,n=best_example(masks,records,vectors)
        self.assertEqual(n,3);self.assertEqual(best['matching_other_instances'],2)
        self.assertEqual(best['remaining_pixels'],600)
        self.assertAlmostEqual(best['cluster_area_fraction'],.75)
        self.assertEqual(len(best['instances']),4)
        self.assertFalse(best['instances'][3]['is_match'])

    def test_selected_images_distinct_and_high_homogeneity_preferred(self):
        rows=[dict(dataset=d,stem=s,category='building',direction='build',cluster_area_fraction=f,
                   matching_other_pixels=600,compactness=.9) for d,s,f in
              [('whu','a',.95),('whu','a',.94),('whu','b',.92),('second','a',.9),('second','z',.8)]]
        picked=select_three(rows)
        self.assertEqual(len(picked),3)
        self.assertEqual(len({(r['dataset'],r['stem']) for r in picked}),3)
        self.assertTrue(all(r['cluster_area_fraction']>=.85 for r in picked))

if __name__=='__main__':unittest.main()
