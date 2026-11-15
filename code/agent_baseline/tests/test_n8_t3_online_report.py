import unittest
from agent_baseline.analysis.n8_t3_online_report import summarize,matches

class ReportTests(unittest.TestCase):
    def test_support_non_submit_and_attempt_vs_return_are_distinct(self):
        rows=[dict(remaining_gt_pixels=16,counts=[0,0,16],state='fallback',submitted_by='fallback',termination_reason='budget',retrieve_calls=1,retrieve_returns=0,cost_cny=.1),
              dict(remaining_gt_pixels=0,counts=[0,3,0],state='successful_submit',submitted_by='model',termination_reason='submit',retrieve_calls=2,retrieve_returns=1,cost_cny=.2),
              dict(remaining_gt_pixels=4,state='missing_final',retrieve_calls=0,retrieve_returns=0,cost_cny=.03)]
        result=summarize(rows)
        self.assertEqual((result['expected'],result['scored'],result['remaining_gt_pixels']),(3,2,20))
        self.assertEqual((result['no_model_submit'],result['positive_fully_empty'],result['empty_fp']),(1,1,3))
        self.assertEqual((result['retrieve_attempted_tasks'],result['retrieve_returned_tasks']),(2,1))
        self.assertTrue(matches(rows[0],'remaining_gt>=16_diagnostic'));self.assertFalse(matches(rows[2],'remaining_gt>=16_diagnostic'))

if __name__=='__main__':unittest.main()
