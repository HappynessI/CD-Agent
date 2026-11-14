import copy
import unittest
from agent_baseline.analysis.night_verifier_by_task import choose,score,VARIANTS


class TaskVerifierTest(unittest.TestCase):
    def test_confirmation_and_other_k_cannot_change_choice(self):
        report=dict(groups={})
        for group in ('second_appeared','second_disappeared','levir_any'):
            arms={v:{'10':dict(complete=True,metrics=dict(f1=20 if v=='V2' else 10)),
                     '3':dict(complete=True,metrics=dict(f1=100 if v=='V4' else 0))} for v in VARIANTS}
            report['groups'][group+'/tune']=arms
            report['groups'][group+'/confirm']=copy.deepcopy(arms)
        first=choose(report)
        for group in ('second_appeared','second_disappeared','levir_any'):
            report['groups'][group+'/confirm']['V4']['10']['metrics']['f1']=1000
        self.assertEqual(choose(report),first)
        self.assertTrue(all(x['variant']=='V2' for x in first['groups'].values()))
        result=score(first,report)
        self.assertTrue(all(x['variant']=='V2' for x in result['groups'].values()))
        self.assertFalse(result['complete']) # K5/K20 absent; no invented completion.

    def test_ties_preserve_v0_and_missing_tune_fails(self):
        report=dict(groups={g+'/tune':{v:{'10':dict(complete=True,metrics=dict(f1=1))} for v in VARIANTS} for g in ('a','b','c')})
        self.assertTrue(all(x['variant']=='V0' for x in choose(report)['groups'].values()))
        report['groups']['a/tune']['V4']['10']['complete']=False
        with self.assertRaises(ValueError):choose(report)


if __name__=='__main__':unittest.main()
