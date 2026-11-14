import json
from pathlib import Path
import tempfile
import unittest
from agent_baseline.analysis.night_t3_report import request_cost,delta_reproduction


class ReportTests(unittest.TestCase):
    def test_accounting_ignores_mirrors_and_historical_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name,cost in [('B0/tasks/x/answer.json',8),('B1/tasks/x/answer.json',.2),('B1/tasks/x/failure_000.json',.1),('B1/tasks/x/attempt_001.json',.2),('B1/summary.json',.3),('t33/retrieval/B0/tasks/x/answer.json',.4)]:
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(dict(cost_cny=cost)))
            result=request_cost(root)
            self.assertAlmostEqual(result['cost_cny'],.7)
            self.assertEqual(result['request_artifacts'],3)

    def test_reproduction_requires_original_counts(self):
        old=dict(per_task=[dict(id='a',records=[dict(method='dino',k=8,counts=[1,2,3])])])
        new=[dict(id='a',scores=[dict(method='delta',k=8,counts=[1,2,3]),dict(method='endpoint',k=8,counts=[9,9,9])])]
        self.assertTrue(delta_reproduction(new,old)['all_available_agree'])
        new[0]['scores'][0]['counts']=[1,3,3]
        self.assertEqual(len(delta_reproduction(new,old)['mismatches']),1)
        self.assertFalse(delta_reproduction([],old)['all_available_agree'])


if __name__=='__main__':unittest.main()
