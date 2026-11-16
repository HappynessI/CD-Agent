import unittest
from agent_baseline.analysis.n9_missed_legacy_abstain import decision

class AbstainContract(unittest.TestCase):
    def test_truncated_yes_prefix_is_not_recovered_or_relabelled_no(self):
        record=dict(answer=None,finish_reason='length',raw='{"change":"yes","reason":"cut off')
        self.assertEqual(decision(record),'invalid_abstain');self.assertIsNone(record['answer'])
        self.assertEqual(decision(dict(answer='unknown')),'invalid_abstain')
        self.assertEqual(decision(dict(answer='no')),'reject')
        self.assertEqual(decision(dict(answer='yes')),'accept')

if __name__=='__main__':unittest.main()
