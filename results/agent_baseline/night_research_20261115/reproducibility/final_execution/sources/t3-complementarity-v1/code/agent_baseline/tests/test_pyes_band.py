"""Synthetic checks for pixel accounting and development-only band selection."""
import unittest
import numpy as np
from agent_baseline.analysis.pyes_band import (
    bin_index, component_pixels, evaluate, f1, paired_ci, segments, select_band, vectors,
)


def dataset(rows, truths=None):
    truths = truths or {'a': 100}
    return dict(images=list(truths), truth_pixels=truths, rows=[dict(image='a', **r) for r in rows])


class PYesBandTests(unittest.TestCase):
    def test_segments_missing_endpoints_and_no_only(self):
        rows = [dict(answer='no', p_yes=p, tp=2, fp=3) for p in (0,.1,.9,1,None,float('nan'))]
        rows.append(dict(answer='yes',p_yes=.1,tp=100,fp=0))
        table = segments(rows)
        self.assertEqual([table[i]['components'] for i in (0,1,9,10)],[1,1,2,2])
        self.assertEqual(sum(r['components'] for r in table),6)
        self.assertEqual(sum(r['tp'] for r in table),12)
        self.assertEqual(sum(r['fp'] for r in table),18)
        self.assertEqual(bin_index(-.1),10)

    def test_pixel_delta_includes_fn_outside_candidates(self):
        data = dataset([dict(answer='yes',p_yes=.99,tp=50,fp=10),
                        dict(answer='no',p_yes=.3,tp=30,fp=10),
                        dict(answer='no',p_yes=None,tp=10,fp=20)])
        base = vectors(data)
        changed = vectors(data,[.2,.4])
        np.testing.assert_array_equal(base,[[50,10,50]])
        np.testing.assert_array_equal(changed,[[80,20,20]])
        self.assertAlmostEqual(f1(changed)-f1(base),80-62.5)
        result = evaluate(data,[.2,.4])
        self.assertAlmostEqual(result['band']['delta_f1'],17.5)
        self.assertEqual(result['band']['ci95'],[17.5,17.5])

    def test_selection_uses_only_development(self):
        dev = dataset([dict(answer='yes',p_yes=.99,tp=50,fp=0),
                       dict(answer='no',p_yes=.25,tp=40,fp=0),
                       dict(answer='no',p_yes=.75,tp=0,fp=100)])
        report = dataset([dict(answer='yes',p_yes=.99,tp=50,fp=0),
                          dict(answer='no',p_yes=.25,tp=0,fp=100),
                          dict(answer='no',p_yes=.75,tp=40,fp=0)])
        band = select_band(dev)
        self.assertEqual(band,[.2,.3])
        self.assertEqual(select_band(report),[.7,.8])
        self.assertLess(evaluate(report,band)['band']['delta_f1'],0)
        self.assertEqual(select_band(dev),band)

    def test_tie_narrowest_closed_band_and_no_gain(self):
        dev = dataset([dict(answer='yes',p_yes=1,tp=50,fp=0),
                       dict(answer='no',p_yes=.3,tp=40,fp=0)])
        self.assertEqual(select_band(dev),[.2,.3])
        bad = dataset([dict(answer='yes',p_yes=1,tp=50,fp=0),
                       dict(answer='no',p_yes=.3,tp=0,fp=100)])
        self.assertIsNone(select_band(bad))
        self.assertEqual(evaluate(bad,None)['band']['delta_f1'],0)
        self.assertIsNone(evaluate(bad,None,selection_available=False)['band'])

    def test_perfect_majority_is_not_pixel_filter(self):
        data = dataset([dict(answer='yes',p_yes=1,tp=30,fp=0),
                        dict(answer='no',p_yes=.3,tp=20,fp=10),
                        dict(answer='no',p_yes=.3,tp=20,fp=20),
                        dict(answer='no',p_yes=None,tp=20,fp=0)])
        np.testing.assert_array_equal(vectors(data,[.2,.4],perfect=True),[[50,10,50]])
        np.testing.assert_array_equal(vectors(data,perfect=True,all_no=True),[[70,10,30]])

    def test_unparsed_answer_is_kept_by_frozen_rule(self):
        data = dataset([dict(answer=None,p_yes=.01,tp=40,fp=5),
                        dict(answer='no',p_yes=.25,tp=40,fp=0)])
        np.testing.assert_array_equal(vectors(data),[[40,5,60]])
        self.assertEqual(select_band(data),[.2,.3])

    def test_component_ids_and_diagonal_connectivity(self):
        mask = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1]],bool)
        truth = np.array([[1,0,1,0],[0,0,0,0],[0,0,0,1]],bool)
        self.assertEqual(component_pixels(mask,truth),[(1,1,1),(2,1,0)])

    def test_paired_bootstrap_deterministic_and_keeps_empty_images(self):
        base = np.array([[10,5,20],[0,0,0],[20,10,5]])
        changed = np.array([[20,6,10],[0,0,0],[22,11,3]])
        self.assertEqual(paired_ci(changed,base),paired_ci(changed,base))
        # All-empty resamples occur above 2.5%; they remain in the distribution.
        self.assertEqual(paired_ci(changed,base)[0],0)
        self.assertGreater(paired_ci(changed,base)[1],0)


if __name__ == '__main__':
    unittest.main()
