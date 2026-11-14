import unittest
from agent_baseline.analysis.night_t3_blind_review import choose


class BlindSelectionTests(unittest.TestCase):
    def test_correctness_is_not_consulted_and_groups_are_fixed(self):
        rows=[];split={}
        for part in ('tune','confirm'):
            for transition in ('1to5','1to2'):
                for i in range(5):
                    name=f'{part}_{i}_{transition}';rows.append(dict(id=name,pair_correct=True));split[name]=part
        first=choose(rows,split)
        second=choose([dict(r,pair_correct=False) for r in reversed(rows)],split)
        self.assertEqual([r['id'] for r in first],[r['id'] for r in second]);self.assertEqual(len(first),12)
        self.assertEqual(sum(split[r['id']]=='confirm' for r in first),6)


if __name__=='__main__':unittest.main()
