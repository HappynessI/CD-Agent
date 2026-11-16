import unittest
import numpy as np
from agent_baseline.analysis.task_reconstruct_t3 import geometry,best_example
from agent_baseline.analysis.task_reconstruct_t3_interior import strict_example,touches_border

class InteriorContract(unittest.TestCase):
    def test_border_example_excludes_whole_task_without_deleting_instance(self):
        masks=[];records=[]
        for i,left in enumerate((0,25,50)):
            m=np.zeros((30,75),bool);m[5:25,left:left+10]=True
            filled,g=geometry(m);masks.append(filled);records.append(dict(id=i+1,**g,rgb_mean=[100,100,100]))
        vectors=np.array([[1.,0.]]*3)
        self.assertIsNotNone(best_example(masks,records,vectors)[0])
        best,failures,n,audit=strict_example(masks,records,vectors)
        self.assertIsNone(best);self.assertEqual(n,0)
        self.assertTrue(audit[0]['touches_border']);self.assertFalse(audit[1]['touches_border'])
        self.assertEqual(len(audit),3);self.assertEqual(sum(r['area'] for r in audit),600)
        self.assertEqual(failures['whole_gt_has_border_instances'],1)

if __name__=='__main__':unittest.main()
