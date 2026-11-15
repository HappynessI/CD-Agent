import unittest
from agent_baseline.analysis.n7_missed_trigger import trigger, metrics

class TriggerTest(unittest.TestCase):
    def test_gate_uses_only_observable_allowlist(self):
        state={'mask_coverage':.04,'direction':'appeared'}
        self.assertTrue(trigger(state,'coverage',.05))
        for nuisance in ({'dataset':'levir','f1':99,'tp':999}, {'dataset':'second','f1':0,'tp':0}):
            self.assertTrue(trigger(dict(state,**nuisance),'coverage',.05))
        self.assertFalse(trigger({'mask_coverage':.06},'coverage',.05))
    def test_metric_counts(self):
        self.assertEqual(metrics(dict(tp=10,fp=5,fn=5))['f1'],2000/30)
