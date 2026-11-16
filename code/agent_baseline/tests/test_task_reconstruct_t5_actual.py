import unittest
import numpy as np
from agent_baseline.analysis.task_reconstruct_t5_actual import lesions,ideal_edit

class ActualError(unittest.TestCase):
    def ring(self):
        p=np.zeros((40,40),bool);p[5:35,5:35]=1;p[6:34,6:34]=0;return p
    def test_filled_background_is_not_false_positive(self):
        pred=self.ring();gt=np.zeros_like(pred);before=pred.copy()
        self.assertEqual(int(pred.sum()),116)
        self.assertEqual(list(lesions(pred,gt,np.ones_like(gt),'appeared',400)),[])
        self.assertTrue(np.array_equal(pred,before))
    def test_filled_gt_is_not_existing_true_positive(self):
        pred=self.ring();gt=np.zeros_like(pred);gt[10:30,10:30]=1
        rows=[r for r in lesions(pred,gt,np.ones_like(gt),'appeared',100) if r[1] in ('FP','Coarse-Merge')]
        self.assertEqual(len(rows),1);region,kind,op,error=rows[0]
        self.assertEqual((kind,op,error),('FP','剔除',116));self.assertTrue(region[20,20])
        repaired=ideal_edit(pred,gt,region,op);self.assertFalse(repaired.any())
    def test_addition_does_not_remove_other_false_positives(self):
        pred=np.zeros((10,10),bool);pred[2,2]=1;gt=np.zeros_like(pred);gt[3,3]=1;region=np.ones_like(pred)
        add=ideal_edit(pred,gt,region,'补检');self.assertTrue(add[2,2]);self.assertTrue(add[3,3])
        refine=ideal_edit(pred,gt,region,'修边');self.assertFalse(refine[2,2]);self.assertTrue(refine[3,3])

if __name__=='__main__':unittest.main()
