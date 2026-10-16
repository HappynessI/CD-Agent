import unittest

import numpy as np

from change_agent.adapters.direct_verifier import DirectQwenVerifier
from change_agent.adapters.stage_backends import _stage_prompt
from change_agent.state import AgentAction, ChangeState, VerifierOutput


RUBRIC_IDS = (
    "evidence_sufficient",
    "target_class_only",
    "change_semantic_precision",
    "change_semantic_recall",
    "changed_object_extent",
    "change_boundary_alignment",
    "change_artifact_control",
)


def make_state():
    image1 = np.zeros((16, 16, 3), dtype=np.uint8)
    image2 = image1.copy()
    image2[4:9, 4:9] = 180
    t1 = np.zeros((16, 16), dtype=bool)
    t2 = np.zeros_like(t1)
    t2[4:9, 4:9] = True
    return ChangeState(image1, image2, "building", t1, t2, t2)


def make_rubric(**overrides):
    return {
        rubric_id: {
            "pass": overrides.get(rubric_id, True),
            "evidence": f"Observable evidence for {rubric_id}.",
        }
        for rubric_id in RUBRIC_IDS
    }


def make_verdict(
    *,
    rubric=None,
    candidate_effect=None,
    error_type="none",
    target_view=None,
    action="finish",
    coordinate=None,
    box=None,
    feedback="Auditable rubric verdict.",
):
    return {
        "rubric": rubric or make_rubric(),
        "candidate_effect": candidate_effect,
        "error_type": error_type,
        "target_view": target_view,
        "suggested_action": action,
        "coordinate_normalized_1000": coordinate,
        "box_normalized_1000": box,
        "feedback": feedback,
    }


def candidate_effect(*, improved, fp=False, fn=False, shape=False):
    return {
        "intended_error_improved": improved,
        "introduced_false_positive": fp,
        "introduced_false_negative": fn,
        "boundary_or_artifact_worsened": shape,
        "evidence": "Candidate versus accepted state evidence.",
    }


class DirectBackend:
    def __init__(self, verdict):
        self.verdicts = iter(verdict if isinstance(verdict, list) else [verdict])
        self.calls = []
        self.payloads = []

    def generate_stage(self, stage, state, payload, previous_state=None):
        self.calls.append((stage, payload["mode"], previous_state is not None))
        self.payloads.append(dict(payload))
        return {"verdict": next(self.verdicts)}


class RepairingDirectBackend(DirectBackend):
    def repair_stage(
        self, stage, state, payload, validation_error, previous_state=None
    ):
        self.calls.append((stage, payload["mode"], previous_state is not None, "repair"))
        self.payloads.append(dict(payload))
        return {"verdict": next(self.verdicts)}


class DirectVerifierTest(unittest.TestCase):
    def test_direct_prompt_locks_target_class_and_forbids_model_scores(self):
        prompt = _stage_prompt(
            "direct",
            {"mode": "initial", "target_class": "building"},
        )

        self.assertIn('"target_class":"building"', prompt)
        self.assertIn("roads, parking areas, vehicles, vegetation", prompt)
        self.assertIn('"change_semantic_precision"', prompt)
        self.assertIn('"candidate_effect":null', prompt)
        contract = prompt.split("TASK:", 1)[0]
        self.assertNotIn("quality_score", contract)
        self.assertNotIn("progress_score", contract)
        self.assertNotIn('"accept"', contract)

    def test_candidate_prompt_requires_binary_paired_effect(self):
        prompt = _stage_prompt(
            "direct",
            {"mode": "candidate", "target_class": "building"},
        )

        self.assertIn('"intended_error_improved":<BOOLEAN>', prompt)
        self.assertIn('"introduced_false_positive":<BOOLEAN>', prompt)
        self.assertIn("negative_point, the selected coordinate must be white", prompt)

    def test_runtime_computes_quality_from_binary_rubric(self):
        rubric = make_rubric(change_semantic_precision=False)
        backend = DirectBackend(
            make_verdict(
                rubric=rubric,
                error_type="false_positive_change",
                target_view="t2",
                action="negative_point",
                coordinate=[500, 500],
                feedback="White change extends beyond supported building change.",
            )
        )

        output = DirectQwenVerifier(backend).verify(make_state(), None, None)

        self.assertTrue(output.verifier_valid)
        self.assertEqual(output.quality_score, 0.7)
        self.assertEqual(output.progress_score, 0.0)
        self.assertEqual(output.comparison, "initial")
        self.assertEqual(output.error_type, "false_positive_change")
        self.assertEqual(output.error_region, (500, 500, 500, 500))
        self.assertFalse(output.accept)
        self.assertEqual(backend.payloads[0]["target_class"], "building")

    def test_clean_initial_state_can_stop_only_when_every_item_passes(self):
        backend = DirectBackend(make_verdict())
        verifier = DirectQwenVerifier(backend)

        output = verifier.verify(make_state(), None, None)

        self.assertEqual(output.quality_score, 1.0)
        self.assertTrue(output.accept)
        self.assertTrue(output.stop)
        self.assertEqual(
            verifier.last_evidence["rubric_aggregation"]["source"],
            "runtime_weighted_binary_rubric",
        )
        self.assertNotIn(
            "quality_score", verifier.last_evidence["direct_verdict"]
        )

    def test_legacy_model_authored_scores_are_rejected(self):
        backend = DirectBackend(
            {
                "comparison": "initial",
                "quality_score": 1.0,
                "progress_score": 0.0,
                "accept": True,
                "error_type": "none",
                "target_view": None,
                "suggested_action": "finish",
                "coordinate_normalized_1000": None,
                "box_normalized_1000": None,
                "feedback": "Legacy score.",
            }
        )

        output = DirectQwenVerifier(backend).verify(make_state(), None, None)

        self.assertFalse(output.verifier_valid)
        self.assertIsNone(output.suggested_action)

    def test_target_scope_diagnostic_does_not_block_actionable_diagnosis(self):
        rubric = make_rubric(target_class_only=False)
        backend = DirectBackend(
            make_verdict(
                rubric=rubric,
                error_type="mixed_error",
                target_view="t2",
                action="negative_point",
                coordinate=[500, 500],
                feedback="Judgment included non-building content.",
            )
        )
        verifier = DirectQwenVerifier(backend)

        output = verifier.verify(make_state(), None, None)

        self.assertTrue(output.verifier_valid)
        self.assertTrue(output.localization_valid)
        self.assertEqual(output.suggested_action, "negative_point")
        self.assertEqual(output.error_region, (500, 500, 500, 500))
        self.assertTrue(
            verifier.last_evidence["rubric_aggregation"]["hard_gates_pass"]
        )

    def test_direct_repair_replaces_black_seed_negative_point(self):
        rubric = make_rubric(change_semantic_precision=False)
        invalid_verdict = make_verdict(
            rubric=rubric,
            error_type="false_positive_change",
            target_view="t2",
            action="negative_point",
            coordinate=[0, 0],
        )
        repaired_verdict = make_verdict(
            rubric=rubric,
            error_type="false_positive_change",
            target_view="t2",
            action="negative_point",
            coordinate=[500, 500],
        )
        backend = RepairingDirectBackend([invalid_verdict, repaired_verdict])

        output = DirectQwenVerifier(backend).verify(make_state(), None, None)

        self.assertTrue(output.verifier_valid)
        self.assertEqual(output.error_region, (500, 500, 500, 500))
        self.assertEqual(len(backend.calls), 2)
        self.assertEqual(backend.calls[-1][-1], "repair")
        self.assertIn(
            "previous_invalid_response",
            backend.payloads[-1]["repair_context"],
        )

    def test_direct_repair_cannot_flip_actionable_diagnosis_to_finish(self):
        rubric = make_rubric(change_semantic_precision=False)
        invalid_geometry = make_verdict(
            rubric=rubric,
            error_type="false_positive_change",
            target_view=None,
            action="negative_point",
            coordinate=[500, 500],
        )
        semantic_flip = make_verdict()
        backend = RepairingDirectBackend([invalid_geometry, semantic_flip])
        verifier = DirectQwenVerifier(backend)

        output = verifier.verify(make_state(), None, None)

        self.assertFalse(output.verifier_valid)
        self.assertIn(
            "changed rubric pass values or semantic error_type",
            verifier.last_evidence["validation_errors"][0],
        )

    def test_direct_positive_point_allows_white_seed(self):
        rubric = make_rubric(change_semantic_recall=False)
        backend = DirectBackend(
            make_verdict(
                rubric=rubric,
                error_type="false_negative",
                target_view="t2",
                action="positive_point",
                coordinate=[500, 500],
            )
        )

        output = DirectQwenVerifier(backend).verify(make_state(), None, None)

        self.assertTrue(output.verifier_valid)
        self.assertEqual(output.suggested_action, "positive_point")

    def test_candidate_comparison_is_derived_from_binary_effect(self):
        rubric = make_rubric(change_artifact_control=False)
        backend = DirectBackend(
            make_verdict(
                rubric=rubric,
                candidate_effect=candidate_effect(improved=True),
                error_type="mixed_error",
                target_view="t2",
                action="positive_point",
                coordinate=[300, 700],
            )
        )

        candidate = make_state()
        previous = make_state()
        candidate.t2_mask[1, 1] = True
        candidate.change_mask[1, 1] = True
        output = DirectQwenVerifier(backend).verify(
            candidate,
            0.8,
            AgentAction("t2", "positive_point", coordinate=(8, 8)),
            previous,
        )

        self.assertEqual(output.quality_score, 0.9)
        self.assertAlmostEqual(output.progress_score, 0.1)
        self.assertEqual(output.comparison, "better")
        self.assertTrue(output.accept)

    def test_candidate_harm_is_programmatically_worse(self):
        rubric = make_rubric(change_semantic_precision=False)
        backend = DirectBackend(
            make_verdict(
                rubric=rubric,
                candidate_effect=candidate_effect(improved=False, fp=True),
                error_type="false_positive_change",
                target_view="t2",
                action="negative_point",
                coordinate=[500, 500],
            )
        )

        candidate = make_state()
        previous = make_state()
        candidate.t2_mask[1, 1] = True
        candidate.change_mask[1, 1] = True
        output = DirectQwenVerifier(backend).verify(
            candidate,
            0.8,
            AgentAction("t2", "positive_point", coordinate=(8, 8)),
            previous,
        )

        self.assertEqual(output.comparison, "worse")
        self.assertFalse(output.accept)

    def test_direct_verifier_replans_after_rejected_candidate(self):
        initial_rubric = make_rubric(change_semantic_recall=False)
        initial_verdict = make_verdict(
            rubric=initial_rubric,
            error_type="false_negative",
            target_view="t2",
            action="positive_point",
            coordinate=[650, 340],
            feedback="T2 misses a building.",
        )
        replan_verdict = make_verdict(
            rubric=initial_rubric,
            error_type="false_negative",
            target_view="t2",
            action="positive_point",
            coordinate=[300, 700],
            feedback="Use a different missing-building point.",
        )
        backend = DirectBackend([initial_verdict, replan_verdict])
        verifier = DirectQwenVerifier(backend)
        accepted_state = make_state()
        accepted_feedback = verifier.verify(accepted_state, None, None)
        rejected_candidate = accepted_state.clone()
        rejected_candidate.t2_mask[1, 1] = True
        rejected_candidate.change_mask[1, 1] = True

        output = verifier.replan_after_rejection(
            accepted_state,
            rejected_candidate,
            accepted_feedback,
            VerifierOutput(
                comparison="worse",
                error_type="false_negative",
                target_view="t2",
                suggested_action="positive_point",
                feedback="Candidate exceeded locality.",
            ),
            AgentAction("t2", "positive_point", coordinate=(10, 5)),
            ["locality_outside_roi_exceeded"],
            [
                {
                    "step_index": 1,
                    "action": {"action": "positive_point"},
                    "rejection_reasons": ["locality_outside_roi_exceeded"],
                }
            ],
        )

        self.assertTrue(output.verifier_valid)
        self.assertEqual(output.quality_score, 0.7)
        self.assertEqual(output.comparison, "uncertain")
        self.assertFalse(output.accept)
        self.assertEqual(output.error_region, (300, 700, 300, 700))
        self.assertEqual(
            backend.calls,
            [("direct", "initial", False), ("direct", "replan", True)],
        )
        self.assertEqual(
            backend.payloads[-1]["rejected_candidate_mask_delta"][
                "change_changed_pixels"
            ],
            1,
        )

    def test_direct_replan_repairs_repeated_action(self):
        rubric = make_rubric(change_semantic_recall=False)
        initial_verdict = make_verdict(
            rubric=rubric,
            error_type="false_negative",
            target_view="t2",
            action="positive_point",
            coordinate=[650, 340],
        )
        repaired_verdict = make_verdict(
            rubric=rubric,
            error_type="false_negative",
            target_view="t2",
            action="positive_point",
            coordinate=[300, 700],
        )
        backend = RepairingDirectBackend(
            [initial_verdict, initial_verdict, repaired_verdict]
        )
        verifier = DirectQwenVerifier(backend)
        accepted_state = make_state()
        accepted_feedback = verifier.verify(accepted_state, None, None)

        output = verifier.replan_after_rejection(
            accepted_state,
            accepted_state.clone(),
            accepted_feedback,
            VerifierOutput(
                comparison="unchanged",
                error_type="false_negative",
                target_view="t2",
                suggested_action="positive_point",
                feedback="Candidate made no progress.",
            ),
            AgentAction("t2", "positive_point", coordinate=(10, 5)),
            ["candidate_effect_not_better"],
            [],
        )

        self.assertTrue(output.verifier_valid)
        self.assertEqual(output.error_region, (300, 700, 300, 700))
        self.assertEqual(len(backend.calls), 3)
        self.assertEqual(backend.calls[-1][-1], "repair")

    def test_direct_replan_rejects_same_pixel_action(self):
        rubric = make_rubric(change_semantic_recall=False)
        initial_verdict = make_verdict(
            rubric=rubric,
            error_type="false_negative",
            target_view="t2",
            action="positive_point",
            coordinate=[667, 333],
        )
        repeated_replan_verdict = dict(initial_verdict)
        backend = DirectBackend([initial_verdict, repeated_replan_verdict])
        verifier = DirectQwenVerifier(backend)
        accepted_state = make_state()
        accepted_feedback = verifier.verify(accepted_state, None, None)

        output = verifier.replan_after_rejection(
            accepted_state,
            accepted_state.clone(),
            accepted_feedback,
            VerifierOutput(
                comparison="worse",
                error_type="false_negative",
                target_view="t2",
                suggested_action="positive_point",
                feedback="Candidate was rejected.",
            ),
            AgentAction("t2", "positive_point", coordinate=(10, 5)),
            ["locality_outside_roi_exceeded"],
            [],
        )

        self.assertFalse(output.verifier_valid)
        self.assertIsNone(output.suggested_action)
        self.assertIn(
            "repeated the rejected action",
            verifier.last_evidence["validation_errors"][0],
        )


if __name__ == "__main__":
    unittest.main()
