import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
from agent_baseline.analysis.night_t3_positive_endpoints import construct,needs,inventory,save_summary
from agent_baseline.analysis.t3_diag import directed_proposals
from agent_baseline.analysis.night_t3_retrieval import score,summaries


class PositiveEndpointsTests(unittest.TestCase):
    def test_strict_matches_existing_and_positive_removes_only_exclusions(self):
        source1=np.array([[[1,1,1,0]],[[0,0,1,1]]],bool)
        source2=np.array([[[0,1,0,0]]],bool)
        target1=np.array([[[0,0,1,0]]],bool)
        target2=np.array([[[1,1,1,1]]],bool)
        stacks=[source1,source2,target1,target2]
        strict,positive,ss,ps=construct(stacks,(1,4))
        np.testing.assert_array_equal(strict,directed_proposals(*stacks,(1,4)))
        self.assertEqual(ps.tolist(),[0,1,2])
        self.assertTrue(positive.any(axis=0).all())
        self.assertEqual(strict.any(axis=0).tolist(),[[True,False,False,True]])
        self.assertTrue(np.all(strict<=positive[ss]))

    def test_confirmation_inventory_and_summary_use_only_requested_variants(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);rows=[dict(id='a')]
            parses={v:{'a':dict(interpretation=dict(pair=[1,1]))} for v in ('B0','B2')}
            plan=inventory(rows,parses,root,root,root,('B0','B2'))
            self.assertEqual(plan['variants'],['B0','B2'])
            self.assertEqual(len(plan['cases']),2)
            self.assertEqual(plan['unique_missing'],0)
            save_summary(root,dict(B0=[],B2=[]),'confirm')
            result=json.loads((root/'summary.json').read_text())
            self.assertEqual(set(result['variants']),{'B0','B2'})
            self.assertIn('Full confirm30',result['interpretation'])
            self.assertIn('multiply exposed',result['confirmation_scope'])
            self.assertEqual(result['status'],'partial')

    def test_empty_candidate_scoring_summary_contract(self):
        masks=np.zeros((0,2,2),bool);truth=np.array([[1,0],[0,0]],bool);ex=np.zeros((2,2),bool)
        result=score(masks,[],truth,ex,'strict_all',0)
        self.assertEqual(result['counts'],[0,0,1])
        rows=summaries([dict(split='tune',group='目标类',remaining_gt_pixels=1,scores=[result],verification={})])
        row=next(r for r in rows if r['split']=='tune' and r['support']=='positive' and r['group']=='全部')
        self.assertEqual(row['methods']['strict_all@0']['covered_pixels'],0)
        self.assertEqual(row['methods']['strict_all@0']['truth_pixels'],1)
        self.assertEqual(row['methods']['strict_all@0']['fn'],1)

    def test_empty_endpoints_and_same_unknown_do_not_create_proposals(self):
        empty=np.zeros((0,2,2),bool);full=np.ones((1,2,2),bool)
        strict,positive,ss,ps=construct([empty,empty,empty,full],(2,2))
        self.assertEqual(len(strict),0);self.assertEqual(len(positive),0)
        self.assertEqual(needs([None,2]),[]);self.assertEqual(needs([2,2]),[])
        self.assertEqual(needs([2,1]),[('non-vegetated ground','t1'),('non-vegetated ground','t2'),('low vegetation','t1'),('low vegetation','t2')])


if __name__=='__main__':unittest.main()
