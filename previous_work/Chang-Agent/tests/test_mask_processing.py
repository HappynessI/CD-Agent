import unittest

import numpy as np

from change_agent.adapters.omniovcd_adapter import MaskPairProcessor, connected_components
from change_agent.perturbations import make_training_target, perturb_mask


class MaskProcessingTest(unittest.TestCase):
    def test_extracts_diagonal_as_one_eight_connected_instance(self):
        mask = np.eye(4, dtype=bool)
        self.assertEqual(len(connected_components(mask)), 1)

    def test_unmatched_instance_becomes_change(self):
        t1 = np.zeros((8, 8), dtype=bool)
        t2 = np.zeros_like(t1)
        t1[1:3, 1:3] = True
        t2[1:3, 1:3] = True
        t2[5:7, 5:7] = True
        update = MaskPairProcessor().rebuild(t1, t2)
        self.assertEqual(update.matching, ((0, 0),))
        self.assertEqual(int(update.change_mask.sum()), 4)

    def test_overlap_presence_tolerates_split_merge_but_greedy_does_not(self):
        t1 = np.zeros((10, 12), dtype=bool)
        t1[2:5, 1:4] = True
        t1[2:5, 7:10] = True
        t2 = t1.copy()
        t2[3, 4:7] = True

        presence = MaskPairProcessor().rebuild(t1, t2)
        greedy = MaskPairProcessor(matching_mode="greedy_one_to_one").rebuild(t1, t2)

        self.assertFalse(presence.change_mask.any())
        self.assertTrue(greedy.change_mask.any())
        self.assertEqual(presence.matching, ((0, 0), (1, 0)))
        self.assertTrue(presence.evidence["matching"]["split_merge_ambiguity"])

    def test_directional_coverage_matches_omniovcd_semantics(self):
        t1 = np.zeros((12, 12), dtype=bool)
        t2 = np.zeros_like(t1)
        t1[4:6, 4:6] = True
        t2[1:11, 1:11] = True

        update = MaskPairProcessor(overlap_threshold=0.25).rebuild(t1, t2)

        # T1 is present in T2 (coverage 1.0), but T2 is not present in T1
        # (coverage 0.04), so the large T2 instance remains a change.
        self.assertEqual(int(update.change_mask.sum()), 100)
        pair = update.evidence["matching"]["candidate_pairs"][0]
        self.assertEqual(pair["t1_coverage"], 1.0)
        self.assertEqual(pair["t2_coverage"], 0.04)

    def test_disappeared_added_and_unrelated_instances_are_changes(self):
        t1 = np.zeros((12, 12), dtype=bool)
        t2 = np.zeros_like(t1)
        t1[1:4, 1:4] = True
        t2[8:11, 8:11] = True

        update = MaskPairProcessor().rebuild(t1, t2)

        self.assertEqual(update.matching, ())
        self.assertEqual(int(update.change_mask.sum()), 18)

    def test_area_filters_can_be_enabled_or_disabled(self):
        t1 = np.zeros((8, 8), dtype=bool)
        t2 = np.zeros_like(t1)
        t1[1, 1] = True
        t1[4:6, 4:6] = True

        no_filter = MaskPairProcessor().rebuild(t1, t2)
        t12_filter = MaskPairProcessor(t12_min_instance_area=2).rebuild(t1, t2)
        cd_filter = MaskPairProcessor(cd_min_instance_area=2).rebuild(t1, t2)

        self.assertEqual(int(no_filter.change_mask.sum()), 5)
        self.assertEqual(int(t12_filter.change_mask.sum()), 4)
        self.assertEqual(int(cd_filter.change_mask.sum()), 4)

    def test_default_matching_configuration_is_auditable(self):
        mask = np.zeros((4, 4), dtype=bool)
        update = MaskPairProcessor().rebuild(mask, mask)
        evidence = update.evidence["matching"]
        self.assertEqual(evidence["matching_mode"], "overlap_presence")
        self.assertEqual(evidence["overlap_threshold"], 0.25)
        self.assertEqual(evidence["t12_min_instance_area"], 0)
        self.assertEqual(evidence["cd_min_instance_area"], 0)

    def test_training_targets_are_offline_and_correct(self):
        gt = np.zeros((10, 10), dtype=bool)
        gt[3:7, 3:7] = True
        candidate = perturb_mask(gt, "dilate", radius=1)
        target = make_training_target(candidate, gt)
        self.assertLess(target.quality, 1)
        self.assertTrue(target.false_positive_map.any())
        self.assertEqual(target.error_type, "false_positive_change")


if __name__ == "__main__":
    unittest.main()
