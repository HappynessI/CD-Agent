import unittest
from agent_baseline.analysis.night_candidate_quality import quality_bin,aggregate,BINS,VARIANTS

class QualityTest(unittest.TestCase):
    def test_inclusive_boundaries(self):
        for counts,expected in [((0,10),BINS[0]),((1,9),BINS[1]),((11,89),BINS[2]),
                ((1,1),BINS[2]),((51,49),BINS[3]),((9,1),BINS[3]),((91,9),BINS[4]),((1,0),BINS[4]),((0,0),'empty_after_subtraction')]:
            self.assertEqual(quality_bin(*counts),expected)

    def test_denominator_and_missing_are_retained(self):
        rows=[]
        for i,decision in enumerate(('yes','no',None,'unparsed')):
            rows.append(dict(group='g',split='confirm',tp=1,fp=9,quality_bin=BINS[1],
                answers={v:dict(answer=decision,missing=i==2) for v in VARIANTS}))
        for bins in aggregate(rows).values():
            cell=bins[BINS[1]]
            self.assertEqual(cell['candidates'],4)
            self.assertEqual(cell['yes']+cell['no']+cell['unknown'],4)
            self.assertEqual(cell['unknown'],2);self.assertEqual(cell['missing'],1)
            self.assertEqual((cell['accepted_candidate_tp_sum'],cell['accepted_candidate_fp_sum']),(1,9))

if __name__=='__main__':unittest.main()
