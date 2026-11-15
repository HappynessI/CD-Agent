import copy
import unittest
from agent_baseline.analysis.night_verifier_selection import choose, VARIANTS


class SelectionTest(unittest.TestCase):
    def test_confirmation_cannot_change_choice_and_missing_tune_is_rejected(self):
        groups={}
        for group in ('second_appeared','second_disappeared','levir_any'):
            groups[group+'/tune']={v:{'10':dict(complete=True,metrics=dict(f1=52 if v=='V1' else 50))} for v in VARIANTS}
            groups[group+'/confirm']={v:{'10':dict(complete=True,metrics=dict(f1=100 if v=='V4' else 0))} for v in VARIANTS}
        report=dict(groups=groups)
        self.assertEqual(choose(report)['selected_verifier'],'V1')
        del groups['levir_any/confirm']
        self.assertEqual(choose(report)['selected_verifier'],'V1')
        incomplete=copy.deepcopy(report)
        incomplete['groups']['levir_any/tune']['V4']['10']['complete']=False
        with self.assertRaises(ValueError):choose(incomplete)


if __name__=='__main__':unittest.main()
