import tempfile
from pathlib import Path
import unittest
import numpy as np
from agent_baseline.analysis.night_t3_oracle import cached_union,constraint_counts
from agent_baseline.analysis.t3_diag import directed_proposals


class OracleCacheTests(unittest.TestCase):
    def test_constraint_counts_match_original_directed_pool(self):
        s1=np.array([[1,1],[1,0]],bool);s2=np.array([[0,1],[0,0]],bool)
        t1=np.array([[0,0],[1,0]],bool);t2=np.ones((2,2),bool);truth=np.ones((2,2),bool)
        counts=constraint_counts([s1,s2,t1,t2],truth)
        original=directed_proposals(s1[None],s2[None],t1[None],t2[None],truth.shape)
        self.assertEqual(counts['strict_loss_and_gain'],int(original.any(axis=0).sum()))
        self.assertEqual((counts['positive_intersection'],counts['after_source_loss'],counts['after_target_gain'],counts['strict_loss_and_gain']),(3,2,2,1))

    def test_missing_cache_stays_missing_without_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            result=cached_union(root,root,dict(id='case',image_t1='image'),'building','t1',(2,2))
            self.assertEqual(result,(None,None))
            self.assertEqual(list(root.iterdir()),[])


if __name__=='__main__':unittest.main()
