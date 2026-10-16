import unittest

import numpy as np

from change_agent.state import ChangeState
from change_agent.coordinates import normalized_point_to_pixel
from change_agent.verifier_regions import (
    attach_verifier_regions,
    build_candidate_delta_regions,
    build_empty_mask_scan_regions,
    build_verifier_regions,
)


class VerifierRegionTest(unittest.TestCase):
    def test_empty_dual_masks_trigger_full_frame_scan_grid(self):
        image = np.zeros((12, 15, 3), dtype=np.uint8)
        empty = np.zeros((12, 15), dtype=bool)
        state = ChangeState(image, image, "building", empty, empty, empty)

        proposals = attach_verifier_regions(
            state,
            max_regions=2,
            empty_mask_scan_grid_size=3,
        )

        self.assertEqual(len(proposals), 9)
        self.assertTrue(all(item["audit_kind"] == "scan" for item in proposals))
        self.assertEqual(
            [item["batch_index"] for item in proposals],
            [0, 0, 1, 1, 2, 2, 3, 3, 4],
        )
        coverage = np.zeros_like(empty, dtype=np.uint8)
        for proposal in proposals:
            x1, y1, x2, y2 = proposal["box_pixels"]
            coverage[y1 : y2 + 1, x1 : x2 + 1] += 1
        self.assertTrue(np.all(coverage == 1))
        facts = state.evidence["verifier_mask_facts"]
        self.assertTrue(facts["proposal_config"]["empty_dual_mask_scan"])
        self.assertEqual(facts["initial_audit_pixels"], empty.size)
        self.assertEqual(facts["initial_audit_uncovered_pixels"], 0)
        self.assertEqual(facts["initial_audit_coverage_ratio"], 1.0)

    def test_empty_mask_scan_rejects_nonempty_temporal_masks(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        empty = np.zeros((8, 8), dtype=bool)
        nonempty = empty.copy()
        nonempty[2, 2] = True
        state = ChangeState(
            image, image, "building", empty, nonempty, nonempty
        )

        with self.assertRaisesRegex(ValueError, "both temporal masks"):
            build_empty_mask_scan_regions(state)

    def test_empty_mask_scan_rejects_nonempty_change_mask(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        empty = np.zeros((8, 8), dtype=bool)
        nonempty = empty.copy()
        nonempty[2, 2] = True
        state = ChangeState(
            image,
            image,
            "building",
            empty,
            empty,
            nonempty,
        )

        with self.assertRaisesRegex(ValueError, "change mask to be empty"):
            build_empty_mask_scan_regions(state)

    def test_component_seed_is_distance_transform_interior_point(self):
        image = np.zeros((21, 21, 3), dtype=np.uint8)
        mask = np.zeros((21, 21), dtype=bool)
        mask[3:18, 4:17] = True
        mask[3:10, 10:17] = False
        state = ChangeState(image, image, "building", np.zeros_like(mask), mask, mask)

        proposal = build_verifier_regions(state)[0]
        seed = normalized_point_to_pixel(
            proposal["component_seed_normalized"], state.image_size
        )

        self.assertTrue(mask[seed[1], seed[0]])
        self.assertNotEqual(seed, tuple(np.argwhere(mask)[0][::-1]))
        self.assertGreater(seed[0], 4)
        self.assertGreater(seed[1], 3)

    def test_proposals_merge_change_and_temporal_difference_sources(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        t1 = np.zeros((32, 32), dtype=bool)
        t2 = np.zeros_like(t1)
        t2[10:15, 12:18] = True
        state = ChangeState(image, image, "building", t1, t2, t2)

        proposals = attach_verifier_regions(state)

        self.assertEqual(len(proposals), 1)
        self.assertIn("change_component", proposals[0]["sources"])
        self.assertIn("temporal_difference", proposals[0]["sources"])
        self.assertEqual(state.evidence["verifier_mask_facts"]["change_pixels"], 30)
        self.assertEqual(proposals[0]["change_pixels"], 30)
        self.assertEqual(proposals[0]["component_t1_mask_pixels"], 0)
        self.assertEqual(proposals[0]["component_t2_mask_pixels"], 30)
        self.assertFalse(proposals[0]["component_seed_t1_mask_white"])
        self.assertTrue(proposals[0]["component_seed_t2_mask_white"])
        self.assertEqual(len(proposals[0]["component_seed_normalized"]), 2)
        self.assertEqual(
            state.evidence["verifier_mask_facts"]["initial_audit_coverage_ratio"],
            1.0,
        )

    def test_initial_components_are_not_dropped_by_per_batch_limit(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        previous_mask = np.zeros((32, 32), dtype=bool)
        previous_mask[2:5, 2:5] = True
        current_mask = previous_mask.copy()
        current_mask[20:25, 20:25] = True
        previous = ChangeState(
            image, image, "building", np.zeros_like(previous_mask), previous_mask, previous_mask
        )
        current = ChangeState(
            image, image, "building", np.zeros_like(current_mask), current_mask, current_mask
        )

        proposals = build_verifier_regions(current, max_regions=1)

        self.assertEqual(len(proposals), 2)
        self.assertEqual([item["batch_index"] for item in proposals], [0, 1])

    def test_single_white_pixel_is_never_lost_by_min_area_filter(self):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        mask = np.zeros((16, 16), dtype=bool)
        mask[8, 8] = True
        state = ChangeState(image, image, "building", np.zeros_like(mask), mask, mask)

        proposals = build_verifier_regions(state, min_component_area=4)

        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["component_area"], 1)
        self.assertEqual(proposals[0]["change_pixels"], 1)

    def test_candidate_delta_regions_keep_added_and_removed_polarity_separate(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        previous_mask = np.zeros((32, 32), dtype=bool)
        previous_mask[2:5, 2:5] = True
        current_mask = np.zeros_like(previous_mask)
        current_mask[20:25, 20:25] = True
        previous = ChangeState(
            image, image, "building", np.zeros_like(previous_mask), previous_mask, previous_mask
        )
        current = ChangeState(
            image, image, "building", np.zeros_like(current_mask), current_mask, current_mask
        )

        proposals = build_candidate_delta_regions(current, previous, max_regions=2)

        self.assertEqual({item["effect_kind"] for item in proposals}, {"added", "removed"})
        self.assertTrue(
            all(len(item["sources"]) == 1 for item in proposals)
        )
        self.assertTrue(
            all(
                item["component_t1_mask_pixels"] <= item["component_area"]
                and item["component_t2_mask_pixels"] <= item["component_area"]
                for item in proposals
            )
        )
        removed = next(item for item in proposals if item["effect_kind"] == "removed")
        facts = removed["transition_mask_facts"]
        self.assertGreater(facts["previous_t2_delta_pixels"], 0)
        self.assertEqual(facts["candidate_t2_delta_pixels"], 0)
        self.assertTrue(facts["previous_seed_t2_white"])
        self.assertFalse(facts["candidate_seed_t2_white"])

    def test_initial_batching_covers_exact_components_not_padded_boxes(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        t1 = np.zeros((32, 32), dtype=bool)
        t2 = np.zeros_like(t1)
        t2[10:12, 10:12] = True
        t2[10:12, 14:16] = True
        state = ChangeState(image, image, "building", t1, t2, t2)

        proposals = attach_verifier_regions(
            state, max_regions=1, padding_ratio=1.0
        )
        facts = state.evidence["verifier_mask_facts"]

        self.assertEqual(len(proposals), 2)
        self.assertEqual(facts["initial_audit_pixels"], 8)
        self.assertEqual(facts["initial_audit_covered_pixels"], 8)
        self.assertEqual(facts["initial_audit_uncovered_pixels"], 0)
        self.assertEqual(facts["initial_audit_coverage_ratio"], 1.0)
        self.assertEqual(facts["proposal_config"]["max_regions_per_batch"], 1)
        self.assertEqual(facts["proposal_config"]["batch_count"], 2)

    def test_candidate_attachment_keeps_components_separate_with_full_coverage(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        previous_mask = np.zeros((32, 32), dtype=bool)
        current_mask = np.zeros_like(previous_mask)
        current_mask[2:4, 2:4] = True
        current_mask[10:12, 10:12] = True
        current_mask[20:22, 20:22] = True
        previous = ChangeState(
            image, image, "building", np.zeros_like(previous_mask), previous_mask, previous_mask
        )
        current = ChangeState(
            image, image, "building", np.zeros_like(current_mask), current_mask, current_mask
        )

        proposals = attach_verifier_regions(current, previous, max_regions=6)

        self.assertEqual(len(proposals), 3)
        self.assertTrue(all(item["effect_kind"] == "added" for item in proposals))
        self.assertEqual(sum(item["component_area"] for item in proposals), 12)
        self.assertEqual(
            current.evidence["verifier_mask_facts"]["candidate_delta_uncovered_pixels"],
            0,
        )
        self.assertEqual(
            current.evidence["verifier_mask_facts"]["candidate_delta_coverage_ratio"],
            1.0,
        )

    def test_component_batch_size_preserves_every_delta_component(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        previous_mask = np.zeros((32, 32), dtype=bool)
        current_mask = np.zeros_like(previous_mask)
        for offset in (2, 8, 14, 20):
            current_mask[offset : offset + 2, offset : offset + 2] = True
        previous = ChangeState(
            image, image, "building", np.zeros_like(previous_mask), previous_mask, previous_mask
        )
        current = ChangeState(
            image, image, "building", np.zeros_like(current_mask), current_mask, current_mask
        )

        proposals = attach_verifier_regions(current, previous, max_delta_regions=3)
        facts = current.evidence["verifier_mask_facts"]

        self.assertEqual(len(proposals), 4)
        self.assertEqual([item["batch_index"] for item in proposals], [0, 0, 0, 1])
        self.assertEqual(facts["candidate_delta_covered_pixels"], 16)
        self.assertEqual(facts["candidate_delta_uncovered_pixels"], 0)
        self.assertEqual(facts["candidate_delta_coverage_ratio"], 1.0)
        self.assertEqual(facts["proposal_config"]["max_regions_per_batch"], 3)
        self.assertEqual(facts["proposal_config"]["batch_count"], 2)

    def test_point_action_scope_aggregates_fragmented_delta_with_full_coverage(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        previous_mask = np.zeros((32, 32), dtype=bool)
        previous_mask[8:12, 8:12] = True
        previous_mask[16:18, 16:18] = True
        candidate_mask = np.zeros_like(previous_mask)
        previous = ChangeState(
            image,
            image,
            "building",
            np.zeros_like(previous_mask),
            previous_mask,
            previous_mask,
        )
        candidate = ChangeState(
            image,
            image,
            "building",
            np.zeros_like(candidate_mask),
            candidate_mask,
            candidate_mask,
            evidence={
                "locality": {
                    "composition_mode": (
                        "remove_simpleclick_delta_within_clicked_component"
                    ),
                    "roi_xyxy": [4, 4, 22, 22],
                },
                "tool_input": {"coordinate": [9, 9]},
            },
        )

        proposals = attach_verifier_regions(
            candidate, previous, max_delta_regions=1
        )

        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["component_area"], 20)
        facts = proposals[0]["transition_mask_facts"]
        self.assertTrue(facts["action_scoped_delta"])
        self.assertEqual(facts["delta_component_count"], 2)
        self.assertEqual(proposals[0]["component_seed_pixels"], [9, 9])
        self.assertEqual(
            candidate.evidence["verifier_mask_facts"][
                "candidate_delta_coverage_ratio"
            ],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
