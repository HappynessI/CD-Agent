import unittest
import numpy as np
from agent_baseline.analysis.n9_missed_score import answer_action

class InvalidAbstain(unittest.TestCase):
    def test_invalid_truncated_yes_never_adds_or_becomes_no(self):
        answer={'answer':'unknown','raw':'{"change":"yes", "reason":"unfinished','finish_reason':'length'}
        action=answer_action(answer,True)
        self.assertIsNone(action)
        base=np.eye(4,dtype=bool);before=base.copy();candidate=np.ones((4,4),bool)
        if action is True:base|=candidate
        self.assertTrue(np.array_equal(base,before))
        self.assertEqual(answer['answer'],'unknown')
        with self.assertRaises(RuntimeError):answer_action(answer)
    def test_valid_binary_decisions_preserved(self):
        self.assertIs(answer_action({'answer':'yes'},True),True)
        self.assertIs(answer_action({'answer':'no'},True),False)

if __name__=='__main__':unittest.main()
