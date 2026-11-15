import json
from pathlib import Path
import tempfile
import unittest
from agent_baseline.analysis.n8_t3_cost import request_ledger


class CostTests(unittest.TestCase):
    def test_attempts_include_paid_failure_without_double_aggregate(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            (p/'attempt_001.json').write_text(json.dumps(dict(usage=dict(prompt_tokens=1000,completion_tokens=100),error_type='PlannerError')))
            (p/'attempt_002.json').write_text(json.dumps(dict(usage=dict(prompt_tokens=1000,completion_tokens=100))))
            (p/'attempt_003.json').write_text(json.dumps(dict(usage=None,error_type='ConnectionError')))
            (p/'answer.json').write_text(json.dumps(dict(cost_cny=99)))
            (p/'failure_000.json').write_text(json.dumps(dict(cost_cny=99)))
            value=request_ledger(p)
            self.assertAlmostEqual(value['known_cost_cny'],.00056)
            self.assertAlmostEqual(value['known_failed_cost_cny'],.00028)
            self.assertEqual(value['usage_missing_attempts'],1)
            self.assertEqual(value['failed_attempts'],2)


if __name__=='__main__':unittest.main()
