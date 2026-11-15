import unittest
import numpy as np
from agent_baseline.analysis.night_candidate_geometry import geometry,classify_answer


class GeometryTest(unittest.TestCase):
    def test_hole_edge_detached_and_disconnected_proposals(self):
        base=np.zeros((32,32),bool);base[2:10,2:10]=True;base[5:7,5:7]=False
        hole=np.zeros_like(base);hole[5:7,5:7]=True
        self.assertEqual(geometry(hole,base)['stratum'],'internal_hole')
        edge=np.zeros_like(base);edge[10,2:10]=True
        self.assertEqual(geometry(edge,base)['stratum'],'adjacent_edge_fragment')
        region=np.zeros_like(base);region[18:29,18:29]=True
        self.assertEqual(geometry(region,base)['stratum'],'detached_region')
        fragments=np.zeros_like(base);fragments[15,1]=True;fragments[20,1]=True
        result=geometry(fragments,base)
        self.assertEqual(result['components'],2)
        self.assertEqual(result['stratum'],'detached_fragment')
        self.assertEqual(result['baseline_overlap_pixels'],0)

    def test_proxy_errors_keep_missing_and_truncation_separate(self):
        self.assertEqual(classify_answer(None,4,0)['missing'],1)
        errors=classify_answer(dict(answer='unknown',finish_reason='length',parse_failure=True),4,0)
        self.assertEqual(errors['proxy_positive'],1)
        self.assertEqual(errors['false_no'],0)
        self.assertEqual(errors['truncated'],1)
        self.assertEqual(errors['parse_failure'],1)
        self.assertEqual(classify_answer(dict(answer='yes'),0,4)['false_yes'],1)


if __name__=='__main__':unittest.main()
