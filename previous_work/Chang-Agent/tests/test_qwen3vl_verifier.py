import json
import unittest

import numpy as np
from PIL import Image

from change_agent.adapters.qwen3vl_verifier import Qwen3VLZeroShotVerifier
from change_agent.state import AgentAction, ChangeState
from change_agent.verifier_regions import attach_verifier_regions


class FakeInputs(dict):
    def to(self, device):
        return self


class FakeProcessor:
    def __init__(self, payloads):
        self.payloads = payloads if isinstance(payloads, list) else [payloads]
        self.messages_history = []
        self.call_count = 0

    def apply_chat_template(self, messages, **kwargs):
        self.messages_history.append(messages)
        return FakeInputs(input_ids=np.zeros((1, 3), dtype=np.int64))

    def batch_decode(self, generated, **kwargs):
        payload = self.payloads[min(self.call_count, len(self.payloads) - 1)]
        self.call_count += 1
        return [payload if isinstance(payload, str) else json.dumps(payload)]


class FakeModel:
    device = "cpu"

    def __init__(self):
        self.max_new_tokens = []
        self.generation_calls = []

    def generate(self, **kwargs):
        self.max_new_tokens.append(kwargs["max_new_tokens"])
        self.generation_calls.append(dict(kwargs))
        return np.zeros((1, 5), dtype=np.int64)


def make_state(end=8):
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[4:end, 4:end] = 180
    t1 = np.zeros((16, 16), dtype=bool)
    t2 = np.zeros_like(t1)
    t2[4:end, 4:end] = True
    return ChangeState(image, image, "building", t1, t2, t2)


def rich_regions(
    proposals,
    verdict="true_change",
    *,
    target_view=None,
    action=None,
    confidence=0.92,
    severity=0.1,
    feedback="The exact RGB evidence supports this local diagnosis.",
    t1_state="background",
    t2_state="building",
):
    return {
        "regions": [
            {
                "region_id": item["region_id"],
                "change_mask_state": (
                    "white_predicted_change"
                    if item.get("audit_kind") == "present"
                    else "black_predicted_unchanged"
                ),
                "t1_state": t1_state,
                "t2_state": t2_state,
                "verdict": verdict,
                "target_view": target_view,
                "suggested_action": action,
                "confidence": confidence,
                "severity": severity,
                "feedback": feedback,
            }
            for item in proposals
        ]
    }


def rich_effects(
    proposals,
    effect="added_true_change",
    *,
    target_view=None,
    action=None,
    confidence=0.88,
    severity=0.2,
    feedback="The candidate delta improves the local change boundary.",
    t1_state="background",
    t2_state="building",
):
    return {
        "regions": [
            {
                "region_id": item["region_id"],
                "delta_kind": item["effect_kind"],
                "t1_state": t1_state,
                "t2_state": t2_state,
                "effect": effect,
                "target_view": target_view,
                "suggested_action": action,
                "confidence": confidence,
                "severity": severity,
                "feedback": feedback,
            }
            for item in proposals
        ]
    }


def synthesis(
    *,
    comparison="initial",
    quality=0.9,
    progress=0.0,
    error_type="none",
    target_view=None,
    region_id=None,
    action="finish",
    feedback="The current mask is accurate and no further correction is needed.",
):
    return {
        "quality_score": quality,
        "progress_score": progress,
        "comparison": comparison,
        "error_type": error_type,
        "target_view": target_view,
        "region_id": region_id,
        "suggested_action": action,
        "feedback": feedback,
    }


class QwenVerifierTest(unittest.TestCase):
    def test_initial_qwen_owns_scores_diagnosis_and_finish(self):
        state = make_state()
        proposals = attach_verifier_regions(state)
        model = FakeModel()
        processor = FakeProcessor([rich_regions(proposals), synthesis(quality=0.91)])
        verifier = Qwen3VLZeroShotVerifier(model=model, processor=processor)

        output = verifier.verify(state, None, None)

        self.assertTrue(output.verifier_valid)
        self.assertEqual(output.comparison, "initial")
        self.assertEqual(output.quality_score, 0.91)
        self.assertEqual(output.progress_score, 0.0)
        self.assertTrue(output.accept)
        self.assertTrue(output.stop)
        self.assertEqual(processor.call_count, 2)
        self.assertEqual(model.max_new_tokens, [1024, 1024])
        self.assertTrue(all(call["do_sample"] is False for call in model.generation_calls))
        self.assertTrue(
            all(call["repetition_penalty"] == 1.05 for call in model.generation_calls)
        )
        self.assertEqual(
            verifier.last_evidence["decision_mode"],
            "qwen_region_diagnosis_then_global_synthesis",
        )

    def test_qwen_selects_exact_false_positive_correction(self):
        state = make_state()
        proposals = attach_verifier_regions(state)
        selected = proposals[0]
        processor = FakeProcessor(
            [
                rich_regions(
                    proposals,
                    "false_positive",
                    target_view="t2",
                    action="negative_point",
                    severity=0.8,
                    feedback="Both RGB views show background, so this change is spurious.",
                ),
                synthesis(
                    quality=0.45,
                    error_type="false_positive_change",
                    target_view="t2",
                    region_id=selected["region_id"],
                    action="negative_point",
                    feedback="Remove the spurious component from the T2 object mask.",
                ),
            ]
        )
        verifier = Qwen3VLZeroShotVerifier(model=FakeModel(), processor=processor)

        output = verifier.verify(state, None, None)

        seed = tuple(selected["component_seed_normalized"])
        self.assertEqual(output.error_type, "false_positive_change")
        self.assertEqual(output.target_view, "t2")
        self.assertEqual(output.suggested_action, "negative_point")
        self.assertEqual(output.error_region, (*seed, *seed))
        self.assertFalse(output.accept)

    def test_global_false_negative_cannot_target_present_component(self):
        state = make_state()
        proposals = attach_verifier_regions(state)
        local_conflict = rich_regions(
            proposals,
            "false_negative",
            target_view="t1",
            action="positive_point",
        )
        global_conflict = synthesis(
            quality=0.3,
            error_type="false_negative",
            target_view="t1",
            region_id=proposals[0]["region_id"],
            action="positive_point",
        )
        verifier = Qwen3VLZeroShotVerifier(
            model=FakeModel(),
            processor=FakeProcessor([local_conflict, global_conflict]),
            max_retries=1,
        )

        output = verifier.verify(state, None, None)

        self.assertFalse(output.verifier_valid)
        self.assertIn(
            "global false_negative cannot target an already-white region",
            " ".join(verifier.last_evidence["validation_errors"]),
        )

    def test_local_true_change_advisory_action_is_left_for_qwen_synthesis(self):
        state = make_state()
        proposals = attach_verifier_regions(state)
        verifier = Qwen3VLZeroShotVerifier(
            model=FakeModel(),
            processor=FakeProcessor(
                [
                    rich_regions(
                        proposals,
                        "true_change",
                        target_view="t2",
                        action="positive_point",
                    ),
                    synthesis(),
                ]
            ),
        )

        output = verifier.verify(state, None, None)

        self.assertTrue(output.verifier_valid)
        self.assertEqual(output.error_type, "none")
        self.assertTrue(output.stop)

    def test_global_correction_cannot_select_locally_correct_region(self):
        state = make_state()
        proposals = attach_verifier_regions(state)
        selected = proposals[0]
        verifier = Qwen3VLZeroShotVerifier(
            model=FakeModel(),
            processor=FakeProcessor(
                [
                    rich_regions(proposals, "true_change"),
                    synthesis(
                        quality=0.4,
                        error_type="false_positive_change",
                        target_view="t2",
                        region_id=selected["region_id"],
                        action="negative_point",
                    ),
                ]
            ),
            max_retries=1,
        )

        output = verifier.verify(state, None, None)

        self.assertFalse(output.verifier_valid)
        self.assertIn(
            "global correction cannot select a region that the local diagnosis marked correct",
            " ".join(verifier.last_evidence["validation_errors"]),
        )

    def test_global_negative_point_requires_white_seed_in_target_mask(self):
        state = make_state()
        proposals = attach_verifier_regions(state)
        selected = proposals[0]
        verifier = Qwen3VLZeroShotVerifier(
            model=FakeModel(),
            processor=FakeProcessor(
                [
                    rich_regions(proposals, "false_positive"),
                    synthesis(
                        quality=0.4,
                        error_type="false_positive_change",
                        target_view="t1",
                        region_id=selected["region_id"],
                        action="negative_point",
                    ),
                ]
            ),
            max_retries=1,
        )

        output = verifier.verify(state, None, None)

        self.assertFalse(output.verifier_valid)
        self.assertIn(
            "global negative_point targets a black seed in the editable object mask",
            " ".join(verifier.last_evidence["validation_errors"]),
        )

    def test_cosmetic_enum_and_string_null_drift_is_canonicalized(self):
        state = make_state()
        proposals = attach_verifier_regions(state)
        regions = rich_regions(proposals)
        for item in regions["regions"]:
            item["change_mask_state"] = "WHITE"
            item["target_view"] = "T1"
            item["suggested_action"] = "null"
        global_decision = synthesis()
        global_decision["comparison"] = "Initial"
        global_decision["target_view"] = "null"
        global_decision["region_id"] = "None"
        global_decision["suggested_action"] = "FINISH"
        verifier = Qwen3VLZeroShotVerifier(
            model=FakeModel(),
            processor=FakeProcessor([regions, global_decision]),
        )

        output = verifier.verify(state, None, None)

        self.assertTrue(output.verifier_valid)
        self.assertEqual(output.comparison, "initial")
        self.assertTrue(output.stop)
        self.assertIsNone(
            verifier.last_evidence["synthesis_decision"]["target_view"]
        )

    def test_missing_proposal_can_drive_false_negative_correction(self):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        t1 = np.zeros((16, 16), dtype=bool)
        t2 = np.zeros_like(t1)
        t2[4:8, 4:8] = True
        state = ChangeState(image, image, "building", t1, t2, np.zeros_like(t1))
        proposals = attach_verifier_regions(state)
        selected = proposals[0]
        verifier = Qwen3VLZeroShotVerifier(
            model=FakeModel(),
            processor=FakeProcessor(
                [
                    rich_regions(
                        proposals,
                        "false_negative",
                        target_view="t2",
                        action="positive_point",
                    ),
                    synthesis(
                        quality=0.35,
                        error_type="false_negative",
                        target_view="t2",
                        region_id=selected["region_id"],
                        action="positive_point",
                    ),
                ]
            ),
        )

        output = verifier.verify(state, None, None)

        self.assertEqual(output.error_type, "false_negative")
        self.assertEqual(output.suggested_action, "positive_point")

    def test_missing_proposal_can_be_correct_unchanged(self):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        t1 = np.zeros((16, 16), dtype=bool)
        t2 = np.zeros_like(t1)
        t2[4:8, 4:8] = True
        state = ChangeState(image, image, "building", t1, t2, np.zeros_like(t1))
        proposals = attach_verifier_regions(state)
        verifier = Qwen3VLZeroShotVerifier(
            model=FakeModel(),
            processor=FakeProcessor(
                [rich_regions(proposals, "correct_unchanged"), synthesis()]
            ),
        )

        output = verifier.verify(state, None, None)

        self.assertTrue(output.verifier_valid)
        self.assertEqual(output.error_type, "none")
        self.assertTrue(output.stop)

    def test_candidate_mixed_effect_can_be_judged_better_and_accepted(self):
        previous = make_state(7)
        state = make_state(8)
        proposals = attach_verifier_regions(state, previous)
        selected = proposals[0]
        processor = FakeProcessor(
            [
                rich_effects(
                    proposals,
                    "mixed",
                    target_view="t2",
                    action="box",
                    feedback=(
                        "Most added pixels recover the new building, while a thin fringe enters "
                        "background; the recovered core is more important."
                    ),
                ),
                synthesis(
                    comparison="better",
                    quality=0.78,
                    progress=0.16,
                    error_type="mixed_error",
                    target_view="t2",
                    region_id=selected["region_id"],
                    action="box",
                    feedback=(
                        "The candidate is better overall because the recovered building core "
                        "outweighs the small fringe. Refine the same region with a box."
                    ),
                ),
            ]
        )
        verifier = Qwen3VLZeroShotVerifier(model=FakeModel(), processor=processor)

        output = verifier.verify(
            state,
            0.62,
            AgentAction("t2", "positive_point", coordinate=(7, 7)),
            previous,
        )

        self.assertEqual(output.comparison, "better")
        self.assertTrue(output.accept)
        self.assertFalse(output.stop)
        self.assertEqual(output.quality_score, 0.78)
        self.assertAlmostEqual(output.score_delta, 0.16)
        self.assertEqual(
            verifier.last_evidence["synthesis_decision"]["comparison"], "better"
        )

    def test_candidate_qwen_can_reject_harmful_delta(self):
        previous = make_state(7)
        state = make_state(8)
        proposals = attach_verifier_regions(state, previous)
        selected = proposals[0]
        verifier = Qwen3VLZeroShotVerifier(
            model=FakeModel(),
            processor=FakeProcessor(
                [
                    rich_effects(
                        proposals,
                        "added_false_change",
                        target_view="t2",
                        action="negative_point",
                    ),
                    synthesis(
                        comparison="worse",
                        quality=0.4,
                        progress=-0.3,
                        error_type="false_positive_change",
                        target_view="t2",
                        region_id=selected["region_id"],
                        action="negative_point",
                    ),
                ]
            ),
        )

        output = verifier.verify(
            state,
            0.7,
            AgentAction("t2", "positive_point", coordinate=(7, 7)),
            previous,
        )

        self.assertEqual(output.comparison, "worse")
        self.assertFalse(output.accept)
        self.assertAlmostEqual(output.score_delta, -0.3)

    def test_effect_polarity_is_structurally_validated(self):
        previous = make_state(7)
        state = make_state(8)
        proposals = attach_verifier_regions(state, previous)
        verifier = Qwen3VLZeroShotVerifier(
            model=FakeModel(),
            processor=FakeProcessor(rich_effects(proposals, "removed_false_positive")),
            max_retries=1,
        )

        output = verifier.verify(
            state, None, AgentAction("t2", "positive_point", coordinate=(7, 7)), previous
        )

        self.assertFalse(output.verifier_valid)
        self.assertIn(
            "added delta region cannot receive a removed effect label",
            " ".join(verifier.last_evidence["validation_errors"]),
        )

    def test_identical_candidate_skips_qwen_and_is_unchanged(self):
        previous = make_state()
        state = previous.clone()
        attach_verifier_regions(state, previous)
        processor = FakeProcessor({})
        verifier = Qwen3VLZeroShotVerifier(model=FakeModel(), processor=processor)

        output = verifier.verify(state, 0.8, AgentAction("t2", "finish"), previous)

        self.assertEqual(output.comparison, "unchanged")
        self.assertEqual(processor.call_count, 0)
        self.assertEqual(
            verifier.last_evidence["decision_mode"], "programmatic_identical_state"
        )

    def test_identical_candidate_fingerprint_reuses_qwen_decision(self):
        previous = make_state(7)
        state = make_state(8)
        proposals = attach_verifier_regions(state, previous)
        selected = proposals[0]
        processor = FakeProcessor(
            [
                rich_effects(proposals),
                synthesis(
                    comparison="better",
                    quality=0.8,
                    progress=0.2,
                    error_type="mixed_error",
                    target_view="t2",
                    region_id=selected["region_id"],
                    action="box",
                ),
            ]
        )
        verifier = Qwen3VLZeroShotVerifier(model=FakeModel(), processor=processor)
        action = AgentAction("t2", "positive_point", coordinate=(7, 7))

        first = verifier.verify(state, 0.6, action, previous)
        second = verifier.verify(state, 0.6, action, previous)

        self.assertEqual(first, second)
        self.assertEqual(processor.call_count, 2)
        self.assertTrue(verifier.last_evidence["cache_hit"])

    def test_all_initial_components_are_batched_then_globally_synthesized(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        t1 = np.zeros((32, 32), dtype=bool)
        t2 = np.zeros_like(t1)
        t2[2:5, 2:5] = True
        t2[25:28, 25:28] = True
        state = ChangeState(image, image, "building", t1, t2, t2)
        attach_verifier_regions(state, max_regions=1, padding_ratio=0.0)
        proposals = state.evidence["verifier_region_proposals"]
        payloads = [rich_regions([proposal]) for proposal in proposals]
        payloads.append(synthesis())
        processor = FakeProcessor(payloads)
        verifier = Qwen3VLZeroShotVerifier(model=FakeModel(), processor=processor)

        output = verifier.verify(state, None, None)

        self.assertTrue(output.verifier_valid)
        self.assertEqual(processor.call_count, len(proposals) + 1)
        self.assertEqual(
            state.evidence["verifier_mask_facts"]["initial_audit_coverage_ratio"], 1.0
        )

    def test_all_delta_components_are_batched_then_globally_synthesized(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        previous_mask = np.zeros((32, 32), dtype=bool)
        previous_mask[28, 28] = True
        current_mask = np.zeros_like(previous_mask)
        for y, x in ((2, 2), (10, 10), (20, 20)):
            current_mask[y, x] = True
        previous = ChangeState(
            image, image, "building", np.zeros_like(previous_mask), previous_mask, previous_mask
        )
        state = ChangeState(
            image, image, "building", np.zeros_like(current_mask), current_mask, current_mask
        )
        proposals = attach_verifier_regions(state, previous, max_delta_regions=3)
        batches = [proposals[:3], proposals[3:]]
        payloads = []
        for batch in batches:
            items = []
            for proposal in batch:
                effect = (
                    "added_true_change"
                    if proposal["effect_kind"] == "added"
                    else "removed_false_positive"
                )
                items.extend(rich_effects([proposal], effect)["regions"])
            payloads.append({"regions": items})
        payloads.append(
            synthesis(comparison="better", quality=0.86, progress=0.25)
        )
        processor = FakeProcessor(payloads)
        verifier = Qwen3VLZeroShotVerifier(model=FakeModel(), processor=processor)

        output = verifier.verify(
            state, 0.61, AgentAction("t2", "box", box=(0, 0, 31, 31)), previous
        )

        self.assertEqual(len(proposals), 4)
        self.assertEqual(processor.call_count, 3)
        self.assertEqual(output.comparison, "better")
        self.assertEqual(
            state.evidence["verifier_mask_facts"]["candidate_delta_coverage_ratio"], 1.0
        )

    def test_unknown_synthesis_region_is_invalid(self):
        state = make_state()
        proposals = attach_verifier_regions(state)
        verifier = Qwen3VLZeroShotVerifier(
            model=FakeModel(),
            processor=FakeProcessor(
                [
                    rich_regions(proposals),
                    synthesis(
                        quality=0.3,
                        error_type="uncertain_region",
                        target_view="t2",
                        region_id="invented",
                        action="box",
                    ),
                ]
            ),
            max_retries=1,
        )

        output = verifier.verify(state, None, None)

        self.assertFalse(output.verifier_valid)
        self.assertIsNone(output.suggested_action)

    def test_truncated_top_level_json_is_never_read_as_nested_json(self):
        with self.assertRaisesRegex(ValueError, "incomplete JSON object"):
            Qwen3VLZeroShotVerifier._extract_json_object(
                '```json {"regions":[{"region_id":"r0"}'
            )

    def test_invalid_generation_retains_previous_score_but_authorizes_nothing(self):
        state = make_state()
        proposals = attach_verifier_regions(state)
        verifier = Qwen3VLZeroShotVerifier(
            model=FakeModel(),
            processor=FakeProcessor([rich_regions(proposals), synthesis(quality=0.87)]),
            max_retries=1,
        )
        valid = verifier.verify(state, None, None)
        verifier.processor = FakeProcessor("{broken")
        candidate = make_state(7)
        attach_verifier_regions(candidate, state)

        invalid = verifier.verify(
            candidate,
            valid.quality_score,
            AgentAction("t2", "negative_point", coordinate=(7, 7)),
            state,
        )

        self.assertFalse(invalid.verifier_valid)
        self.assertFalse(invalid.accept)
        self.assertIsNone(invalid.suggested_action)
        self.assertEqual(invalid.quality_score, 0.87)

    def test_candidate_key_changes_with_context_action_and_schema(self):
        previous = make_state(7)
        state = make_state(8)
        proposals = attach_verifier_regions(state, previous)
        facts = state.evidence["verifier_mask_facts"]
        verifier = Qwen3VLZeroShotVerifier(model=FakeModel(), processor=FakeProcessor({}))
        action = AgentAction("t2", "positive_point", coordinate=(7, 7))
        baseline = verifier._candidate_fingerprint(state, previous, action, proposals, facts)
        changed = state.clone()
        changed.query = "warehouse"

        self.assertNotEqual(
            baseline,
            verifier._candidate_fingerprint(changed, previous, action, proposals, facts),
        )
        self.assertNotEqual(
            baseline,
            verifier._candidate_fingerprint(
                state,
                previous,
                AgentAction("t2", "positive_point", coordinate=(6, 7)),
                proposals,
                facts,
            ),
        )
        verifier.SCHEMA_VERSION = "rich_region_diagnosis_next"
        self.assertNotEqual(
            baseline,
            verifier._candidate_fingerprint(state, previous, action, proposals, facts),
        )

    def test_prompts_request_long_diagnosis_and_qwen_global_decision(self):
        state = make_state()
        proposals = attach_verifier_regions(state)
        verifier = Qwen3VLZeroShotVerifier(model=FakeModel(), processor=FakeProcessor({}))

        messages = verifier.build_messages(state, None, None)
        prompt = " ".join(
            item["text"]
            for item in messages[0]["content"]
            if item["type"] == "text"
        )
        self.assertIn("error-diagnosis core", prompt)
        self.assertIn("false_positive", prompt)
        self.assertIn("false_negative", prompt)
        self.assertIn("correct_unchanged", prompt)
        self.assertIn("FINAL CURRENT CHANGE MASK", prompt)
        self.assertIn("feedback (one or two concise diagnostic sentences", prompt)
        self.assertIn("Full predicted T1 object mask", prompt)
        self.assertIn("CLEAN T1 RGB", prompt)
        self.assertIn("diagnostic data, not scene colors", prompt)

        local_panels = [
            item["image"]
            for item in messages[0]["content"]
            if item["type"] == "image" and item["image"].size == (576, 384)
        ]
        panel = np.asarray(local_panels[0])
        x1, y1, x2, y2 = proposals[0]["box_pixels"]
        expected_crop = np.asarray(state.t1_image)[y1 : y2 + 1, x1 : x2 + 1]
        expected_resized = np.asarray(
            Qwen3VLZeroShotVerifier._as_image(expected_crop).resize(
                (192, 192), Image.Resampling.BILINEAR
            )
        )
        np.testing.assert_array_equal(panel[:192, :192], expected_resized)
        self.assertFalse(np.any(np.all(panel[:192, :192] == [140, 0, 140], axis=2)))
        component = verifier._proposal_initial_component(state, proposals[0])
        tight_region = verifier._tight_component_region(component)
        expected_tight = np.asarray(
            Qwen3VLZeroShotVerifier._as_image(state.t1_image[tight_region]).resize(
                (192, 192), Image.Resampling.BILINEAR
            )
        )
        np.testing.assert_array_equal(panel[192:, :192], expected_tight)

        judgments = tuple(
            verifier._parse_rich_region_payload(rich_regions(proposals), proposals)
        )
        synthesis_messages = verifier.build_synthesis_messages(
            state,
            None,
            None,
            proposals,
            state.evidence["verifier_mask_facts"],
            judgments,
            initial=True,
        )
        synthesis_prompt = " ".join(
            item["text"]
            for item in synthesis_messages[0]["content"]
            if item["type"] == "text"
        )
        self.assertIn("quality_score", synthesis_prompt)
        self.assertIn("progress_score", synthesis_prompt)
        self.assertIn("better/worse/unchanged", synthesis_prompt.replace(", ", "/"))
        self.assertIn('"component_area": 16', synthesis_prompt)
        self.assertIn('"component_t1_mask_pixels": 0', synthesis_prompt)
        self.assertIn('"component_seed_t2_mask_white": true', synthesis_prompt)
        local_summary = synthesis_prompt.split("Local diagnoses:", 1)[1]
        self.assertNotIn('"suggested_action"', local_summary)
        self.assertNotIn('"target_view"', local_summary)

    def test_component_outline_preserves_audited_rgb_pixels(self):
        rgb = np.zeros((5, 5, 3), dtype=np.uint8)
        rgb[2, 2] = [40, 80, 120]
        component = np.zeros((5, 5), dtype=bool)
        component[2, 2] = True

        outlined = Qwen3VLZeroShotVerifier._outline_component(rgb, component)

        np.testing.assert_array_equal(outlined[2, 2], rgb[2, 2])
        np.testing.assert_array_equal(outlined[1, 2], [255, 255, 0])


if __name__ == "__main__":
    unittest.main()
