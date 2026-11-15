import unittest
import numpy as np
from agent_baseline.analysis.night_t3_paired_diagnostic import values, paired_interval, analyze


class PairedTests(unittest.TestCase):
    def test_identical_arms_have_zero_paired_uncertainty(self):
        counts=[[10, 2, 3], [0, 7, 0], [100, 5, 20]]
        interval=paired_interval(counts, counts, replicates=100)
        self.assertEqual(interval, {k:[0., 0.] for k in ('tp','fp','fn','f1')})

    def test_pooled_f1_and_deterministic_resampling(self):
        before=[[1,0,0],[0,99,9]]; after=[[2,0,0],[0,80,8]]
        self.assertAlmostEqual(values(np.asarray(before).sum(axis=0))[-1], 200/110)
        self.assertEqual(paired_interval(before,after,replicates=100), paired_interval(before,after,replicates=100))
        one=paired_interval([[1,3,1]], [[1,1,1]], replicates=20)
        self.assertEqual(one['fp'], [-2.,-2.])
        self.assertAlmostEqual(one['f1'][0], 50-200/6)

    def test_partial_and_empty_false_positive_case_remain_visible(self):
        row=dict(id='a',support='empty',group='目标类',before=[0,10,0],after=[0,4,0],
            delta=dict(tp=0,fp=-6,fn=0,f1=0),metrics_path='/a/metrics.json')
        result=analyze([row])
        self.assertEqual(result['status'],'partial')
        self.assertEqual(result['extremes']['fp']['benefits'][0]['id'],'a')
        self.assertEqual(result['extremes']['f1']['benefits'],[])
        positive=next(r for r in result['strata'] if r['support']=='positive' and r['group']=='全部')
        self.assertIsNone(positive['bootstrap95'])


if __name__=='__main__': unittest.main()
