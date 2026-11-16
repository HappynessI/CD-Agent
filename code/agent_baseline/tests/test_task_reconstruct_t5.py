import unittest
import numpy as np
from agent_baseline.analysis.task_reconstruct_t5 import lesions,score

class T5Construction(unittest.TestCase):
    def test_complete_instance_hole_filled_without_initial_mutation(self):
        gt=np.zeros((40,40),bool);gt[5:30,5:30]=1;gt[15,15]=0
        pred=np.zeros_like(gt);pred[5:10,5:30]=1;before=pred.copy()
        rows=list(lesions(pred,gt,np.ones_like(gt),'disappeared',400))
        region,kind,operation,error=rows[0]
        self.assertEqual(kind,'FN-Demolish');self.assertTrue(region[15,15]);self.assertEqual(int(region.sum()),625)
        self.assertGreaterEqual(error,400);self.assertTrue(np.array_equal(pred,before))
    def test_fp_uses_entire_predicted_component(self):
        gt=np.zeros((40,40),bool);gt[:10,:10]=1
        pred=gt.copy();pred[15:35,15:35]=1
        rows=list(lesions(pred,gt,np.ones_like(gt),'appeared',400))
        region,kind,operation,error=rows[0]
        self.assertEqual((kind,operation,error),('FP','剔除',400));self.assertEqual(int(region.sum()),400)
        self.assertGreater(score(pred&~region,gt,np.ones_like(gt)),score(pred,gt,np.ones_like(gt)))

if __name__=='__main__':unittest.main()
