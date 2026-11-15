import copy
import unittest
from agent_baseline.analysis.night_t3_semantic_strata import audit_semantics,VARIANTS


class SemanticStrataTests(unittest.TestCase):
    def fixture(self):
        rows=[dict(id=str(i),split='tune',group='目标类',remaining_gt_pixels=(16 if i<8 else 1 if i<11 else 0),
            before_agrees=i<11,after_agrees=i<8,pair_agrees=i<8) for i in range(30)]
        return dict(records=rows),{name:dict(records=copy.deepcopy(rows)) for name in VARIANTS}

    def test_identical_cohort_strata_and_pair_ids(self):
        audit,variants=self.fixture();prior=copy.deepcopy(variants)
        result=audit_semantics(audit,variants)
        self.assertEqual(result['cohort_sizes'],{'all':30,'positive':11,'remaining_gt>=16_diagnostic':8,'empty':19})
        row=next(r for r in result['rows'] if r['support']=='positive' and r['group']=='全部' and r['variant']=='B3')
        self.assertEqual(row['correct_counts'],dict(before=11,after=8,pair=8))
        self.assertEqual(set(row['pair_correct_ids']),{str(i) for i in range(8)})
        self.assertEqual(variants,prior)

    def test_missing_variant_case_rejected(self):
        audit,variants=self.fixture();variants['B3']['records'].pop()
        with self.assertRaises(ValueError):audit_semantics(audit,variants)


if __name__=='__main__':unittest.main()
