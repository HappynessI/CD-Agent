import json
import tempfile
import unittest
from pathlib import Path
from agent_baseline.analysis.night_missed_report import collect,render


class NightReportTest(unittest.TestCase):
    def test_missing_stages_and_failure_cost_not_double_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def put(name,data):
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data))
            put('usage/V1/ok.json',dict(cost_cny=.2,raw='yes'))
            put('usage/V1/error.json',dict(cost_cny=.1,error_type='HTTPError'))
            put('failures/V1/error/1.json',dict(cost_cny=.1,error_type='HTTPError'))
            put('rank_V1/answers/old.json',dict(reused_from='/old/answer.json',cost_cny=0,original_cost_cny=.9))
            result=collect(root)
            self.assertEqual(result['views']['status'],'pending')
            self.assertAlmostEqual(result['new_recorded_cost_cny'],.3)
            self.assertEqual(result['views_usage']['failed_attempts'],1)
            self.assertEqual(len(result['exact_cases']),1)
            self.assertIn('pending',render(result))


if __name__=='__main__':unittest.main()
