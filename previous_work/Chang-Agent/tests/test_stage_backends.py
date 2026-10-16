import hashlib
import json
import unittest

import numpy as np

from change_agent.adapters.stage_backends import (
    _CompleteStageJsonLogitsProcessor,
    _delta_only_contour,
    _generation_eos_token_ids,
    _hosted_messages,
    _local_messages,
    _normalized_crop_box,
    _stage_visual_input_scope,
    _stage_images,
    _stage_prompt,
)
from change_agent.state import ChangeState
from change_agent.terminal_review import (
    PLAN_NULL_ALTERNATIVE_PROMPT_VERSION,
    PLAN_NULL_UNREVIEWED_PROMPT_VERSION,
    SELECTION_POLICY_VERSION,
)
from change_agent.verifier_regions import EMPTY_MASK_SCAN_PROPOSAL_VERSION
from change_agent.verifier_protocol import AuditChecklist, StageProtocolError


def _terminal_review_audit_payload(
    *,
    branch: str,
    prompt_version: str,
    region_id: str = "r7",
    audit_kind: str = "present",
    proposal_version: str | None = None,
) -> dict:
    region = {"region_id": region_id, "audit_kind": audit_kind}
    if proposal_version is not None:
        region["proposal_version"] = proposal_version
    return {
        "target_class": "building",
        "region": region,
        "proposal_catalog": [region],
        "terminal_review": {
            "round_index": 2,
            "round_kind": "plan_null_alternative",
            "trigger": "plan_null_after_allowed_repair",
            "fresh_backend_context": True,
            "inherits_hidden_state": False,
            "copies_round_one_raw_response": False,
            "selection_policy_version": SELECTION_POLICY_VERSION,
            "preferred_region_id": region_id,
            "prompt_version": prompt_version,
            "selection_branch": branch,
            "review_region_id": region_id,
            "proposal_version": proposal_version,
        },
    }


class NormalizedCropBoxTests(unittest.TestCase):
    def test_json_completion_guard_suppresses_eos_until_outer_object_closes(self):
        class Tokenizer:
            text = '{"selection":{"region_ids":["r0"],"reason":"x"}'

            def decode(self, *args, **kwargs):
                return self.text

        tokenizer = Tokenizer()
        guard = _CompleteStageJsonLogitsProcessor(
            tokenizer=tokenizer,
            prompt_length=1,
            stage="select",
            eos_token_ids=(2, 3),
        )

        guarded = guard(np.array([[9, 1]]), np.zeros((1, 5), dtype=float))

        self.assertTrue(np.isneginf(guarded[0, 2]))
        self.assertTrue(np.isneginf(guarded[0, 3]))
        self.assertEqual(guarded[0, 1], 0.0)

        tokenizer.text += "}"
        completed = guard(np.array([[9, 1]]), np.zeros((1, 5), dtype=float))
        self.assertTrue(np.isneginf(completed[0, 1]))
        self.assertEqual(completed[0, 2], 0.0)
        self.assertEqual(completed[0, 3], 0.0)

    def test_json_completion_guard_ignores_out_of_vocabulary_eos_ids(self):
        class Tokenizer:
            def decode(self, *args, **kwargs):
                return "{}"

        scores = np.zeros((1, 4), dtype=float)
        guarded = _CompleteStageJsonLogitsProcessor(
            tokenizer=Tokenizer(),
            prompt_length=1,
            stage="decision",
            eos_token_ids=(-1, 8),
        )(np.array([[9, 1]]), scores)

        self.assertTrue(np.array_equal(guarded, scores))

    def test_json_completion_guard_blocks_51135_extra_object_shape(self):
        class Tokenizer:
            text = (
                '{"region_id":"r0","visual_judgment":{'
                '"t1_state":"background","t2_state":"building",'
                '"evidence":"[0, 0, 0]"},{"diagnosis":{'
                '"audit_checklist":{"evidence_sufficient":"pass"}}}'
            )

            def decode(self, *args, **kwargs):
                return self.text

        guard = _CompleteStageJsonLogitsProcessor(
            tokenizer=Tokenizer(),
            prompt_length=1,
            stage="audit",
            eos_token_ids=(2,),
        )

        guarded = guard(np.array([[9, 1]]), np.zeros((1, 5), dtype=float))

        self.assertTrue(np.isneginf(guarded[0, 2]))
        self.assertEqual(guarded[0, 1], 0.0)

    def test_generation_eos_ids_include_model_and_tokenizer_terminators(self):
        class GenerationConfig:
            eos_token_id = [2, 3]

        class Model:
            generation_config = GenerationConfig()

        class Tokenizer:
            eos_token_id = 3

        class Processor:
            tokenizer = Tokenizer()

        self.assertEqual(_generation_eos_token_ids(Model(), Processor()), (2, 3))

    def test_delta_only_contour_preserves_delta_rgb_and_blacks_padding(self):
        image = np.zeros((5, 5, 3), dtype=np.uint8)
        image[2, 2] = (12, 34, 56)
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True

        rendered = np.asarray(_delta_only_contour(image, mask))

        self.assertEqual(tuple(rendered[2, 2]), (12, 34, 56))
        self.assertEqual(tuple(rendered[0, 0]), (0, 0, 0))
        self.assertEqual(tuple(rendered[1, 1]), (0, 255, 255))

    def test_tiny_region_is_expanded_for_bailian(self):
        crop = _normalized_crop_box((500, 500, 500, 500), (256, 256))
        self.assertEqual(crop, (128, 128, 139, 139))
        self.assertEqual(crop[2] - crop[0], 11)
        self.assertEqual(crop[3] - crop[1], 11)

    def test_expansion_stays_inside_image_at_edge(self):
        crop = _normalized_crop_box((0, 0, 0, 0), (256, 256))
        self.assertEqual(crop, (0, 0, 11, 11))
        self.assertGreaterEqual(crop[0], 0)
        self.assertGreaterEqual(crop[1], 0)
        self.assertLessEqual(crop[2], 256)
        self.assertLessEqual(crop[3], 256)

    def test_existing_large_region_is_unchanged(self):
        crop = _normalized_crop_box((100, 200, 800, 900), (256, 256))
        self.assertEqual(crop, (26, 51, 205, 231))

    def test_proposal_mode_sends_marked_overview_and_local_crops(self):
        state = ChangeState(
            np.zeros((32, 32, 3), dtype=np.uint8),
            np.zeros((32, 32, 3), dtype=np.uint8),
            "building",
            np.zeros((32, 32), dtype=bool),
            np.zeros((32, 32), dtype=bool),
            np.zeros((32, 32), dtype=bool),
        )
        payload = {
            "visual_context": "proposal",
            "region": {
                "region_id": "r0",
                "audit_kind": "present",
                "component_seed_normalized_1000": [500, 500],
                "box_normalized_1000": [250, 250, 750, 750],
            },
        }
        payload["proposal_catalog"] = [payload["region"]]

        images = _stage_images("diagnosis", state, None, payload)

        self.assertEqual(len(images), 6)
        self.assertIn("Active-region marked global overview", images[0][0])
        self.assertTrue(all("proposal" in label for label, _ in images[1:]))

    def test_hybrid_mode_sends_full_context_and_local_crops(self):
        state = ChangeState(
            np.zeros((32, 32, 3), dtype=np.uint8),
            np.zeros((32, 32, 3), dtype=np.uint8),
            "building",
            np.zeros((32, 32), dtype=bool),
            np.zeros((32, 32), dtype=bool),
            np.zeros((32, 32), dtype=bool),
        )
        payload = {
            "visual_context": "hybrid",
            "region": {
                "region_id": "r0",
                "audit_kind": "present",
                "component_seed_normalized_1000": [500, 500],
                "box_normalized_1000": [250, 250, 750, 750],
            },
        }
        payload["proposal_catalog"] = [payload["region"]]

        images = _stage_images("diagnosis", state, None, payload)

        self.assertEqual(len(images), 6)
        self.assertIn("Active-region marked global overview", images[0][0])
        self.assertEqual(images[1][0], "Exact T1 proposal crop")
        self.assertEqual(images[-1][0], "Exact proposal change-mask crop")
        self.assertEqual(images[0][1].size, (160, 32))
        self.assertEqual(images[1][1].size, (16, 16))

    def test_select_stage_renders_one_numbered_global_overview(self):
        state = ChangeState(
            np.zeros((32, 32, 3), dtype=np.uint8),
            np.zeros((32, 32, 3), dtype=np.uint8),
            "building",
            np.zeros((32, 32), dtype=bool),
            np.pad(np.ones((6, 6), dtype=bool), 13),
            np.pad(np.ones((6, 6), dtype=bool), 13),
        )
        payload = {
            "proposal_catalog": [
                {
                    "region_id": "r0",
                    "audit_kind": "present",
                    "component_seed_normalized_1000": [500, 500],
                    "box_normalized_1000": [400, 400, 600, 600],
                }
            ]
        }

        images = _stage_images("select", state, None, payload)

        self.assertEqual(len(images), 1)
        self.assertIn("Numbered global proposal overview", images[0][0])
        self.assertEqual(images[0][1].size, (96, 32))

    def test_proposal_local_stage_keeps_active_marked_overview(self):
        state = ChangeState(
            np.zeros((32, 32, 3), dtype=np.uint8),
            np.zeros((32, 32, 3), dtype=np.uint8),
            "building",
            np.zeros((32, 32), dtype=bool),
            np.zeros((32, 32), dtype=bool),
            np.zeros((32, 32), dtype=bool),
        )
        region = {
            "region_id": "r0",
            "audit_kind": "present",
            "component_seed_normalized_1000": [500, 500],
            "box_normalized_1000": [250, 250, 750, 750],
        }
        images = _stage_images(
            "diagnosis",
            state,
            None,
            {
                "visual_context": "proposal",
                "region": region,
                "proposal_catalog": [region],
            },
        )

        self.assertEqual(len(images), 6)
        self.assertIn("yellow is r0", images[0][0])
        self.assertEqual(images[1][0], "Exact T1 proposal crop")
        self.assertEqual(images[-1][0], "Exact proposal change-mask crop")

    def test_hybrid_local_stage_uses_full_state_even_with_catalog(self):
        state = ChangeState(
            np.zeros((32, 32, 3), dtype=np.uint8),
            np.zeros((32, 32, 3), dtype=np.uint8),
            "building",
            np.zeros((32, 32), dtype=bool),
            np.zeros((32, 32), dtype=bool),
            np.zeros((32, 32), dtype=bool),
        )
        region = {
            "region_id": "r0",
            "audit_kind": "present",
            "component_seed_normalized_1000": [500, 500],
            "box_normalized_1000": [250, 250, 750, 750],
        }

        images = _stage_images(
            "diagnosis",
            state,
            None,
            {
                "visual_context": "hybrid",
                "region": region,
                "proposal_catalog": [region],
            },
        )

        self.assertEqual(len(images), 6)
        self.assertIn("Active-region marked global overview", images[0][0])
        self.assertEqual(images[0][1].size, (160, 32))
        self.assertEqual(images[-1][0], "Exact proposal change-mask crop")

    def test_evidence_prompt_describes_the_selected_visual_scope(self):
        proposal_prompt = _stage_prompt(
            "evidence", {"visual_context": "proposal"}
        )
        hybrid_prompt = _stage_prompt("evidence", {"visual_context": "hybrid"})

        self.assertIn("active-region marked RGB/change overview", proposal_prompt)
        self.assertIn("active-region marked full-frame state", hybrid_prompt)

    def test_plan_null_terminal_review_prompts_are_frozen_distinct_and_versioned(self):
        unreviewed = _stage_prompt(
            "audit",
            _terminal_review_audit_payload(
                branch="next_unreviewed_environment_region",
                prompt_version=PLAN_NULL_UNREVIEWED_PROMPT_VERSION,
            ),
        )
        alternative = _stage_prompt(
            "audit",
            _terminal_review_audit_payload(
                branch="first_unresolved_environment_region",
                prompt_version=PLAN_NULL_ALTERNATIVE_PROMPT_VERSION,
            ),
        )

        self.assertIn(PLAN_NULL_UNREVIEWED_PROMPT_VERSION, unreviewed)
        self.assertIn("round one did not inspect", unreviewed)
        self.assertNotIn(PLAN_NULL_ALTERNATIVE_PROMPT_VERSION, unreviewed)
        self.assertIn(PLAN_NULL_ALTERNATIVE_PROMPT_VERSION, alternative)
        self.assertIn("seek an alternative diagnosis", alternative)
        self.assertNotIn(PLAN_NULL_UNREVIEWED_PROMPT_VERSION, alternative)
        self.assertNotEqual(
            hashlib.sha256(unreviewed.encode("utf-8")).hexdigest(),
            hashlib.sha256(alternative.encode("utf-8")).hexdigest(),
        )

    def test_plan_null_terminal_review_prompt_rejects_unknown_or_mismatched_version(self):
        payloads = [
            _terminal_review_audit_payload(
                branch="next_unreviewed_environment_region",
                prompt_version=PLAN_NULL_ALTERNATIVE_PROMPT_VERSION,
            ),
            _terminal_review_audit_payload(
                branch="first_unresolved_environment_region",
                prompt_version="qwen3vl_unknown_plan_null_prompt_v1",
            ),
        ]
        missing_key = _terminal_review_audit_payload(
            branch="first_unresolved_environment_region",
            prompt_version=PLAN_NULL_ALTERNATIVE_PROMPT_VERSION,
        )
        missing_key["terminal_review"].pop("prompt_version")
        payloads.append(missing_key)

        for index, payload in enumerate(payloads):
            with self.subTest(index=index), self.assertRaises(StageProtocolError):
                _stage_prompt("audit", payload)

    def test_h1_scan_prompt_binds_frozen_proposal_version_and_h0_ignores_it(self):
        h0_payload = {
            "target_class": "building",
            "region": {"region_id": "s0", "audit_kind": "scan"},
        }
        versioned_h0_payload = {
            "target_class": "building",
            "region": {
                "region_id": "s0",
                "audit_kind": "scan",
                "proposal_version": EMPTY_MASK_SCAN_PROPOSAL_VERSION,
            },
        }
        self.assertEqual(
            _stage_prompt("audit", h0_payload),
            _stage_prompt("audit", versioned_h0_payload),
        )

        h1_payload = _terminal_review_audit_payload(
            branch="next_unreviewed_environment_region",
            prompt_version=PLAN_NULL_UNREVIEWED_PROMPT_VERSION,
            region_id="s0",
            audit_kind="scan",
            proposal_version=EMPTY_MASK_SCAN_PROPOSAL_VERSION,
        )
        prompt = _stage_prompt("audit", h1_payload)
        self.assertIn(EMPTY_MASK_SCAN_PROPOSAL_VERSION, prompt)
        for proposal_version in (None, "qwen3vl_wrong_scan_grid_v1"):
            malformed = _terminal_review_audit_payload(
                branch="next_unreviewed_environment_region",
                prompt_version=PLAN_NULL_UNREVIEWED_PROMPT_VERSION,
                region_id="s0",
                audit_kind="scan",
                proposal_version=proposal_version,
            )
            with self.subTest(proposal_version=proposal_version), self.assertRaises(
                StageProtocolError
            ):
                _stage_prompt("audit", malformed)

    def test_candidate_region_includes_previous_and_candidate_object_mask_crops(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        previous_mask = np.zeros((32, 32), dtype=bool)
        previous_mask[8:12, 8:12] = True
        candidate_mask = np.zeros_like(previous_mask)
        previous = ChangeState(
            image, image, "building", np.zeros_like(previous_mask), previous_mask, previous_mask
        )
        candidate = ChangeState(
            image, image, "building", np.zeros_like(candidate_mask), candidate_mask, candidate_mask
        )
        region = {
            "region_id": "d0",
            "audit_kind": "delta_removed",
            "component_seed_normalized_1000": [300, 300],
            "box_normalized_1000": [200, 200, 450, 450],
        }

        images = _stage_images(
            "candidate_evidence",
            candidate,
            previous,
            {
                "visual_context": "proposal",
                "region": region,
                "proposal_catalog": [region],
            },
        )

        self.assertEqual(len(images), 11)
        labels = [label for label, _ in images]
        self.assertIn("Exact previous accepted T1 object-mask crop", labels)
        self.assertIn("Exact previous accepted T2 object-mask crop", labels)
        self.assertIn("Exact previous accepted change-mask crop", labels)
        self.assertIn("Exact candidate-added pixel crop", labels)
        self.assertIn("Exact candidate-removed pixel crop", labels)
        self.assertIn(
            "Exact candidate-delta-only T2 RGB crop; cyan contour marks delta",
            labels,
        )
        self.assertNotIn("Exact T1 proposal crop", labels)
        self.assertNotIn("Exact T2 proposal crop", labels)

    def test_direct_candidate_includes_delta_masks_and_exact_delta_crops(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        previous_change = np.zeros((32, 32), dtype=bool)
        candidate_change = previous_change.copy()
        candidate_change[8:10, 8:10] = True
        previous = ChangeState(
            image, image, "building", previous_change, previous_change, previous_change
        )
        candidate = ChangeState(
            image, image, "building", previous_change, candidate_change, candidate_change
        )

        images = _stage_images(
            "direct", candidate, previous, {"mode": "candidate"}
        )

        labels = [label for label, _ in images]
        self.assertIn("Candidate-added change pixels", labels)
        self.assertIn("Candidate-removed change pixels", labels)
        self.assertIn(
            "Exact candidate-delta-only T1 RGB crop; cyan contour marks delta",
            labels,
        )
        self.assertIn("Exact candidate-added pixel crop", labels)

    def test_direct_replan_prompt_forbids_repeating_rejected_action(self):
        prompt = _stage_prompt(
            "direct",
            {
                "mode": "replan",
                "rejected_action": {
                    "target_view": "t2",
                    "action": "positive_point",
                    "coordinate": [217, 166],
                },
                "rejection_reasons": ["locality_outside_roi_exceeded"],
                "rejection_history": [
                    {"action": {"action": "positive_point"}, "step_index": 1}
                ],
            },
        )

        self.assertIn("rollback replan", prompt)
        self.assertIn("Do not repeat the rejected action or geometry", prompt)
        self.assertIn("rejection_history", prompt)
        self.assertIn("comparison must be uncertain", prompt)

    def test_atomic_prompt_keeps_target_semantics_and_compact_checklist(self):
        audit_prompt = _stage_prompt(
            "audit",
            {
                "target_class": "building",
                "region": {"region_id": "r7"},
            },
        )

        self.assertNotIn("visual_confidence", audit_prompt)
        self.assertNotIn("<CONFIDENCE_0_TO_1>", audit_prompt)
        self.assertIn('"audit_checklist"', audit_prompt)
        self.assertNotIn('"mask_assessment"', audit_prompt)
        self.assertNotIn('"target_view"', audit_prompt)
        self.assertNotIn('"status":<AUDIT_STATUS>', audit_prompt)
        self.assertIn('"evidence_sufficient":"<AUDIT_STATUS>"', audit_prompt)
        self.assertIn(
            '"t1_state":"<BUILDING_BACKGROUND_MIXED_OR_UNCERTAIN>"',
            audit_prompt,
        )
        self.assertIn(
            '"evidence":"REPLACE_WITH_ONE_SHORT_PAIRED_RGB_FACT"',
            audit_prompt,
        )
        self.assertNotIn(
            '"evidence":"strongest local support or contradiction',
            audit_prompt,
        )
        self.assertIn(
            "semantic target-class labels from the unmodified RGB",
            audit_prompt,
        )
        self.assertIn(
            "Never output black, white, green, or any other color name",
            audit_prompt,
        )
        self.assertIn("trailers, RVs, mobile equipment", audit_prompt)
        self.assertIn("deterministic order", audit_prompt)
        self.assertIn("cannot pass as correct", audit_prompt)
        self.assertIn("At least one substantive checklist item", audit_prompt)
        self.assertIn(
            "proposal or change-mask color/presence is not evidence",
            audit_prompt.casefold(),
        )
        self.assertIn("names the visible physical content", audit_prompt)
        self.assertIn("color alone is not a semantic target", audit_prompt)
        self.assertIn("never copy placeholder or template prose", audit_prompt)
        self.assertNotIn('"change_mask_state":<MASK_STATE>', audit_prompt)
        self.assertIn("Runtime injects it into the trace", audit_prompt)
        self.assertNotIn("screening_resolution", audit_prompt)
        self.assertNotIn("screening_hypothesis", audit_prompt)

    def test_atomic_output_contract_template_is_valid_json(self):
        prompt = _stage_prompt(
            "audit",
            {
                "target_class": "building",
                "region": {"region_id": "r7"},
            },
        )

        template = prompt.split("template:\n", 1)[1].split("\nTASK:", 1)[0]
        parsed = json.loads(template)

        self.assertEqual(
            set(parsed), {"region_id", "visual_judgment", "diagnosis"}
        )
        self.assertEqual(parsed["region_id"], "r7")
        self.assertEqual(
            set(parsed["visual_judgment"]), {"t1_state", "t2_state", "evidence"}
        )
        self.assertEqual(
            parsed["visual_judgment"]["t1_state"],
            "<BUILDING_BACKGROUND_MIXED_OR_UNCERTAIN>",
        )
        self.assertEqual(
            parsed["visual_judgment"]["t2_state"],
            "<BUILDING_BACKGROUND_MIXED_OR_UNCERTAIN>",
        )
        self.assertEqual(
            parsed["visual_judgment"]["evidence"],
            "REPLACE_WITH_ONE_SHORT_PAIRED_RGB_FACT",
        )
        self.assertEqual(set(parsed["diagnosis"]), {"audit_checklist"})
        self.assertEqual(
            parsed["diagnosis"]["audit_checklist"]["evidence_sufficient"],
            "<AUDIT_STATUS>",
        )
        self.assertEqual(
            set(parsed["diagnosis"]["audit_checklist"].values()),
            {"<AUDIT_STATUS>"},
        )
        self.assertEqual(len(parsed["diagnosis"]["audit_checklist"]), 7)

    def test_atomic_rgb_enum_repair_names_only_semantic_values(self):
        prompt = _stage_prompt(
            "audit",
            {
                "target_class": "building",
                "region": {"region_id": "r7"},
            },
            "atomic audit contains an invalid RGB state",
        )

        self.assertIn("Correct t1_state and t2_state", prompt)
        self.assertIn(
            "exactly building, background, mixed, or uncertain",
            prompt,
        )
        self.assertIn(
            "black, white, green, and all other color names are invalid",
            prompt,
        )

    def test_atomic_evidence_boilerplate_repair_requires_rgb_fact(self):
        prompt = _stage_prompt(
            "audit",
            {
                "target_class": "building",
                "region": {"region_id": "r7"},
            },
            "atomic audit evidence copied schema boilerplate",
        )

        self.assertIn("region-specific visible fact", prompt)
        self.assertIn("never copy template prose", prompt)
        self.assertIn("never copy placeholder or template prose", prompt)

    def test_atomic_image_placeholder_repair_requires_rgb_fact(self):
        prompt = _stage_prompt(
            "audit",
            {
                "target_class": "building",
                "region": {"region_id": "r7"},
            },
            "atomic audit evidence copied an image/schema placeholder",
        )

        self.assertIn("image/schema marker", prompt)
        self.assertIn("natural-language visible fact", prompt)

    def test_atomic_mask_color_repair_requires_paired_rgb_fact(self):
        prompt = _stage_prompt(
            "audit",
            {
                "target_class": "building",
                "region": {"region_id": "r7"},
            },
            "atomic audit evidence relies on mask/presentation color",
        )

        self.assertIn("visible target-class content from paired RGB", prompt)
        self.assertIn("Do not cite change_mask_state", prompt)
        self.assertIn("any mask, presentation pixels, or added marks", prompt)

    def test_atomic_json_numeric_evidence_repair_requires_natural_language_rgb_fact(self):
        prompt = _stage_prompt(
            "audit",
            {
                "target_class": "building",
                "region": {"region_id": "r7"},
            },
            "atomic audit evidence is a JSON/numeric/coordinate placeholder",
        )

        self.assertIn("natural-language observation", prompt)
        self.assertIn("paired unmodified RGB", prompt)
        self.assertIn("bare JSON value/list, number, RGB tuple, box", prompt)
        self.assertIn("or coordinate list", prompt)

    def test_scan_audit_requires_rgb_only_evidence(self):
        prompt = _stage_prompt(
            "audit",
            {
                "target_class": "building",
                "region": {"region_id": "s0", "audit_kind": "scan"},
            },
        )

        self.assertIn("Only the paired RGB crops are visual evidence", prompt)
        self.assertIn("never cite predicted-mask state", prompt)
        self.assertIn("as evidence of no RGB change", prompt)
        self.assertIn("spaces between words", prompt)
        self.assertIn("do not join words with underscores", prompt)
        self.assertNotIn("inside this black tile", prompt)
        self.assertNotIn("all-black prediction tile", prompt)

    def test_atomic_stage_includes_exact_component_geometry(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        t2 = np.zeros((32, 32), dtype=bool)
        t2[8:12, 8:12] = True
        state = ChangeState(image, image, "building", np.zeros_like(t2), t2, t2)
        region = {
            "region_id": "r0",
            "audit_kind": "present",
            "component_seed_normalized_1000": [300, 300],
            "box_normalized_1000": [200, 200, 450, 450],
        }

        images = _stage_images(
            "audit",
            state,
            None,
            {
                "visual_context": "proposal",
                "region": region,
                "proposal_catalog": [region],
            },
        )

        self.assertEqual(images[-1][0], "Exact audited component geometry crop")
        self.assertTrue(np.asarray(images[-1][1]).any())

    def test_atomic_evidence_only_repair_uses_authoritative_paired_rgb(self):
        grid_y, grid_x = np.indices((100, 100), dtype=np.uint16)
        t1_image = np.stack(
            (grid_x, grid_y, (grid_x + grid_y) % 256),
            axis=-1,
        ).astype(np.uint8)
        t2_image = np.stack(
            ((grid_x + 7) % 256, (grid_y + 11) % 256, grid_x),
            axis=-1,
        ).astype(np.uint8)
        t2_mask = np.zeros((100, 100), dtype=bool)
        t2_mask[83:100, 30:55] = True
        state = ChangeState(
            t1_image,
            t2_image,
            "building",
            np.zeros_like(t2_mask),
            t2_mask,
            t2_mask,
        )
        region = {
            "region_id": "r3",
            "audit_kind": "present",
            "change_mask_state": "white",
            "component_seed_normalized_1000": [430, 920],
            "box_normalized_1000": [306, 843, 545, 1000],
        }
        overlapping_region = {
            "region_id": "r6",
            "audit_kind": "present",
            "component_seed_normalized_1000": [480, 950],
            "box_normalized_1000": [384, 918, 608, 1000],
        }
        payload = {
            "target_class": "building",
            "visual_context": "hybrid",
            "region": region,
            "proposal_catalog": [region, overlapping_region],
            "repair_context": {
                "mode": "atomic_evidence_only",
                "origin_validation_error": (
                    "atomic audit evidence relies on mask/presentation color"
                ),
            },
        }
        error = (
            "atomic audit evidence relies on mask/presentation color instead "
            "of observable RGB content; rejected text mentioned a red outline"
        )

        local = _local_messages("audit", state, payload, None, error)
        hosted = _hosted_messages("audit", state, payload, None, error)
        local_content = local[0]["content"]
        hosted_content = hosted[-1]["content"]

        self.assertEqual(
            [item["type"] for item in local_content],
            ["text", "image", "text", "image", "text"],
        )
        self.assertEqual(
            [item["type"] for item in hosted_content],
            ["text", "image_url", "text", "image_url", "text"],
        )
        labels = [local_content[0]["text"], local_content[2]["text"]]
        self.assertEqual(
            labels,
            [
                "Authoritative audit region r3: T1 unmodified RGB-only crop",
                "Authoritative audit region r3: T2 unmodified RGB-only crop",
            ],
        )
        images = [local_content[1]["image"], local_content[3]["image"]]
        self.assertEqual([image.mode for image in images], ["RGB", "RGB"])
        self.assertEqual([image.size for image in images], [(25, 17), (25, 17)])
        self.assertTrue(
            np.array_equal(np.asarray(images[0]), t1_image[83:100, 30:55])
        )
        self.assertTrue(
            np.array_equal(np.asarray(images[1]), t2_image[83:100, 30:55])
        )
        self.assertFalse(
            any(
                token in label.casefold()
                for label in labels
                for token in ("overview", "mask", "geometry")
            )
        )
        prompt = local_content[-1]["text"]
        self.assertIn("Repair only the atomic-audit evidence string", prompt)
        self.assertIn("mechanically preserve", prompt)
        self.assertIn("REPAIR:", prompt)
        self.assertNotIn(error, prompt)
        self.assertNotIn("mask/presentation", prompt)
        self.assertNotIn("red outline", prompt)
        self.assertNotIn("change_mask_state", prompt)
        self.assertNotIn("proposal_catalog", prompt)
        self.assertNotIn("<ENVIRONMENT_FACTS>", prompt)
        self.assertNotIn("normalized [0,1000] geometry", prompt)
        self.assertEqual(
            _stage_visual_input_scope("audit", payload, error),
            "authoritative_pair_rgb_only_evidence_repair",
        )
        self.assertEqual(
            _stage_visual_input_scope(
                "audit",
                payload,
                "atomic evidence repair changed the region_id",
            ),
            "authoritative_pair_rgb_only_evidence_repair",
        )

        unauthorized = dict(payload)
        unauthorized.pop("repair_context")
        standard = _local_messages("audit", state, unauthorized, None, error)
        self.assertEqual(
            sum(item["type"] == "image" for item in standard[0]["content"]),
            7,
        )
        self.assertEqual(
            _stage_visual_input_scope("audit", unauthorized, error),
            "standard",
        )

    def test_empty_mask_scan_audit_uses_authoritative_paired_rgb_only(self):
        t1_image = np.full((30, 30, 3), (240, 0, 0), dtype=np.uint8)
        t2_image = np.full((30, 30, 3), (0, 240, 0), dtype=np.uint8)
        t1_image[10:20, 10:20] = (12, 34, 56)
        t2_image[10:20, 10:20] = (65, 43, 21)
        empty = np.zeros((30, 30), dtype=bool)
        state = ChangeState(
            t1_image,
            t2_image,
            "building",
            empty,
            empty,
            empty,
        )
        region = {
            "region_id": "s4",
            "audit_kind": "scan",
            "change_mask_state": "black",
            "component_seed_normalized_1000": [500, 500],
            "box_normalized_1000": [333, 333, 667, 667],
        }
        catalog = [
            region,
            {
                "region_id": "s5",
                "audit_kind": "scan",
                "box_normalized_1000": [667, 333, 1000, 667],
            },
        ]

        images = _stage_images(
            "audit",
            state,
            None,
            {
                "target_class": "building",
                "visual_context": "proposal",
                "region": region,
                "proposal_catalog": catalog,
            },
        )
        prompt = _stage_prompt(
            "audit",
            {
                "target_class": "building",
                "region": region,
                "proposal_catalog": catalog,
            },
        )

        self.assertEqual(
            [label for label, _ in images],
            [
                "Authoritative scan tile s4: T1 RGB-only crop",
                "Authoritative scan tile s4: T2 RGB-only crop",
            ],
        )
        self.assertEqual(
            [image.size for _, image in images],
            [(11, 11), (11, 11)],
        )
        self.assertFalse(any("mask" in label.casefold() for label, _ in images))
        self.assertFalse(any("overview" in label.casefold() for label, _ in images))
        self.assertFalse(any("geometry" in label.casefold() for label, _ in images))
        self.assertTrue(
            np.all(np.asarray(images[0][1]) == np.asarray((12, 34, 56)))
        )
        self.assertTrue(
            np.all(np.asarray(images[1][1]) == np.asarray((65, 43, 21)))
        )
        self.assertIn("both predicted T1/T2 object masks are empty", prompt)
        self.assertIn("authoritative tile box", prompt)
        self.assertNotIn("proposal_catalog", prompt)

    def test_scan_audit_visual_input_fails_closed_on_invalid_environment_state(self):
        image = np.zeros((30, 30, 3), dtype=np.uint8)
        empty = np.zeros((30, 30), dtype=bool)
        base_region = {
            "region_id": "s4",
            "audit_kind": "scan",
            "box_normalized_1000": [333, 333, 667, 667],
        }

        with self.subTest(reason="missing_box"):
            with self.assertRaisesRegex(StageProtocolError, "authoritative tile box"):
                _stage_images(
                    "audit",
                    ChangeState(image, image, "building", empty, empty, empty),
                    None,
                    {
                        "target_class": "building",
                        "region": {"region_id": "s4", "audit_kind": "scan"},
                    },
                )

        with self.subTest(reason="nonempty_object_mask"):
            nonempty = empty.copy()
            nonempty[0, 0] = True
            with self.assertRaisesRegex(StageProtocolError, "all predicted masks"):
                _stage_images(
                    "audit",
                    ChangeState(
                        image,
                        image,
                        "building",
                        nonempty,
                        empty,
                        nonempty,
                    ),
                    None,
                    {"target_class": "building", "region": base_region},
                )

        with self.subTest(reason="nonempty_change_mask"):
            nonempty = empty.copy()
            nonempty[0, 0] = True
            with self.assertRaisesRegex(StageProtocolError, "all predicted masks"):
                _stage_images(
                    "audit",
                    ChangeState(
                        image,
                        image,
                        "building",
                        empty,
                        empty,
                        nonempty,
                    ),
                    None,
                    {"target_class": "building", "region": base_region},
                )

    def test_atomic_machine_label_repair_requires_natural_language_rgb_fact(self):
        prompt = _stage_prompt(
            "audit",
            {
                "target_class": "building",
                "region": {"region_id": "s7", "audit_kind": "scan"},
            },
            "atomic audit evidence is an underscore-delimited label",
        )

        self.assertIn("natural-language sentence", prompt)
        self.assertIn("Use spaces between words", prompt)
        self.assertIn("paired unmodified RGB", prompt)
        self.assertNotIn("snake_case", prompt)
        self.assertNotIn("all_black", prompt)

    def test_audit_checklist_deterministically_derives_error_and_quality(self):
        correct = AuditChecklist(
            evidence_sufficient="pass",
            target_class_only="pass",
            white_pixels_supported="pass",
            boundary_alignment="pass",
            internal_holes_absent="pass",
            changed_object_extent_complete="pass",
            fragment_artifacts_absent="pass",
        )
        mixed = AuditChecklist(
            evidence_sufficient="pass",
            target_class_only="pass",
            white_pixels_supported="fail",
            boundary_alignment="fail",
            internal_holes_absent="pass",
            changed_object_extent_complete="fail",
            fragment_artifacts_absent="pass",
        )
        missing = AuditChecklist(
            evidence_sufficient="pass",
            target_class_only="not_applicable",
            white_pixels_supported="not_applicable",
            boundary_alignment="not_applicable",
            internal_holes_absent="pass",
            changed_object_extent_complete="fail",
            fragment_artifacts_absent="not_applicable",
        )
        uncertain = AuditChecklist(
            evidence_sufficient="uncertain",
            target_class_only="uncertain",
            white_pixels_supported="uncertain",
            boundary_alignment="uncertain",
            internal_holes_absent="uncertain",
            changed_object_extent_complete="uncertain",
            fragment_artifacts_absent="uncertain",
        )

        self.assertEqual(correct.error_type, "none")
        self.assertEqual(correct.quality_score, 1.0)
        self.assertEqual(mixed.error_type, "mixed_error")
        self.assertEqual(mixed.quality_score, 0.5)
        self.assertEqual(missing.error_type, "false_negative")
        self.assertEqual(missing.quality_score, 0.5)
        self.assertEqual(uncertain.error_type, "uncertain_region")
        self.assertEqual(uncertain.quality_score, 0.0)


if __name__ == "__main__":
    unittest.main()
