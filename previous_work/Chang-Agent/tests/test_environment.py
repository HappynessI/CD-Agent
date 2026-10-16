import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from change_agent.adapters.omniovcd_adapter import InitializationResult, MaskPairProcessor
from change_agent.action_parser import ActionValidationError
from change_agent.environment import ChangeAgentEnvironment
from change_agent.executor import ActionExecutor
from change_agent.state import AgentAction, VerifierOutput


class Backend:
    def __init__(self):
        self.processor = MaskPairProcessor(overlap_threshold=0.5)

    def initialize(self, t1_image, t2_image, query):
        t1 = np.zeros(t1_image.shape[:2], dtype=bool)
        t2 = np.zeros_like(t1)
        t1[2:5, 2:5] = True
        return InitializationResult(t1, t2, self.processor.rebuild(t1, t2))

    def rebuild(self, t1_mask, t2_mask, evidence):
        return self.processor.rebuild(t1_mask, t2_mask, evidence)


class Point:
    def refine(self, image, initial_mask, coordinate, is_positive, click_history=()):
        output = initial_mask.copy()
        x, y = coordinate
        output[y, x] = is_positive
        return output


class Box:
    def __init__(self):
        self.last_box = None

    def segment_box(self, image, box_cxcywh_normalized, query):
        self.last_box = box_cxcywh_normalized
        output = np.zeros(image.shape[:2], dtype=bool)
        output[2:5, 2:5] = True
        return output


class EvidenceVerifier:
    def verify(self, state, previous_score, previous_action, previous_state=None):
        # A matched T1/T2 instance produces an empty change mask and the best score.
        score = 0.9 if not state.change_mask.any() else 0.3
        delta = 0 if previous_score is None else score - previous_score
        return VerifierOutput(
            quality_score=score,
            score_delta=delta,
            error_type="none" if score > 0.8 else "uncertain_region",
            target_view="t2",
            suggested_action="finish" if score > 0.8 else "box",
            feedback="test feedback",
            accept=score > 0.8,
        )


class SequenceVerifier:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.previous_states = []
        self.call_count = 0

    def verify(self, state, previous_score, previous_action, previous_state=None):
        self.call_count += 1
        self.previous_states.append(previous_state.clone() if previous_state else None)
        return next(self.outputs)


class ProposalCaptureVerifier:
    def __init__(self):
        self.states = []

    def verify(self, state, previous_score, previous_action, previous_state=None):
        self.states.append(state.clone())
        return VerifierOutput(
            comparison="initial" if previous_state is None else "better",
            error_type="false_positive_change",
            target_view="t1",
            error_region=(0, 0, 1000, 1000),
            suggested_action="negative_point",
            feedback="proposal capture",
        )


class ReplanningVerifier:
    def __init__(self, initial_feedback, replan_feedback):
        self.initial_feedback = initial_feedback
        self.replan_feedback = replan_feedback
        self.replan_calls = []

    def verify(self, state, previous_score, previous_action, previous_state=None):
        if previous_state is not None:
            raise AssertionError("hard gate should skip candidate verification")
        return self.initial_feedback

    def replan_after_rejection(
        self,
        accepted_state,
        rejected_candidate,
        accepted_feedback,
        rejected_feedback,
        rejected_action,
        rejection_reasons,
        rejection_history,
    ):
        self.replan_calls.append(
            {
                "accepted_state": accepted_state.clone(),
                "rejected_candidate": rejected_candidate.clone(),
                "accepted_feedback": accepted_feedback,
                "rejected_feedback": rejected_feedback,
                "rejected_action": rejected_action,
                "rejection_reasons": tuple(rejection_reasons),
                "rejection_history": tuple(rejection_history),
            }
        )
        return self.replan_feedback


class EnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.box = Box()
        self.environment = ChangeAgentEnvironment(
            Backend(), ActionExecutor(Point(), self.box), EvidenceVerifier(), max_steps=4
        )
        self.image1 = np.zeros((8, 8, 3), dtype=np.uint8)
        self.image2 = np.ones_like(self.image1)

    def test_box_step_rebuilds_state_and_finish_can_be_accepted(self):
        observation = self.environment.reset(self.image1, self.image2, "building")
        self.assertTrue(observation.change_mask.any())
        observation, done = self.environment.step(
            AgentAction("t2", "box", box=(2, 2, 5, 5))
        )
        self.assertFalse(done)
        self.assertFalse(observation.change_mask.any())
        self.assertIsNotNone(self.box.last_box)
        _, done = self.environment.step(AgentAction("t2", "finish"))
        self.assertTrue(done)
        self.assertEqual(self.environment.best_state.step_index, 1)

    def test_observation_exposes_predicted_masks_but_not_gt(self):
        observation = self.environment.reset(self.image1, self.image2, "building")
        public = observation.to_mapping()
        self.assertIn("predicted_t1_mask", public)
        self.assertIn("predicted_t2_mask", public)
        self.assertTrue(np.array_equal(public["predicted_t1_mask"], observation.t1_mask))
        self.assertTrue(np.array_equal(public["predicted_t2_mask"], observation.t2_mask))
        self.assertFalse(any("gt" in key.lower() for key in public))

    def test_environment_attaches_mask_derived_regions_before_verification(self):
        verifier = ProposalCaptureVerifier()
        environment = ChangeAgentEnvironment(
            Backend(), ActionExecutor(Point(), self.box), verifier, max_steps=2
        )

        environment.reset(self.image1, self.image2, "building")

        evidence = verifier.states[0].evidence
        self.assertGreater(len(evidence["verifier_region_proposals"]), 0)
        self.assertEqual(evidence["verifier_mask_facts"]["change_pixels"], 9)
        self.assertIn(
            "change_component",
            evidence["verifier_region_proposals"][0]["sources"],
        )

    def test_authorized_point_uses_live_environment_proposal(self):
        class RecordingPoint:
            def __init__(self):
                self.initial_masks = []

            def refine(
                self, image, initial_mask, coordinate, is_positive, click_history=()
            ):
                self.initial_masks.append(np.array(initial_mask, copy=True))
                return np.array(initial_mask, copy=True)

        point = RecordingPoint()
        verifier = SequenceVerifier(
            [
                VerifierOutput(
                    comparison="initial",
                    error_type="false_positive_change",
                    target_view="t1",
                    suggested_action="negative_point",
                ),
                VerifierOutput(comparison="unchanged", accept=False),
            ]
        )
        environment = ChangeAgentEnvironment(
            Backend(), ActionExecutor(point, self.box), verifier, max_steps=2
        )
        environment.reset(self.image1, self.image2, "building")
        proposal = environment.state.evidence["verifier_region_proposals"][0]
        seed = tuple(proposal["component_seed_pixels"])

        environment.step(
            AgentAction("t1", "negative_point", coordinate=seed),
            authorized_region_id=proposal["region_id"],
        )

        entry = environment.trajectory.entries[1]
        self.assertEqual(
            entry.execution["point_authorization"]["region_id"],
            proposal["region_id"],
        )
        self.assertEqual(
            entry.execution["point_authorization"]["proposal_box_xyxy"],
            proposal["box_pixels"],
        )
        self.assertLessEqual(
            int(point.initial_masks[0].sum()),
            int(environment.trajectory.entries[0].state.t1_mask.sum()),
        )

    def test_authorized_point_rejects_unknown_or_mismatched_region(self):
        verifier = SequenceVerifier(
            [VerifierOutput(comparison="initial", suggested_action="negative_point")]
        )
        environment = ChangeAgentEnvironment(
            Backend(), ActionExecutor(Point(), self.box), verifier, max_steps=2
        )
        environment.reset(self.image1, self.image2, "building")
        proposal = environment.state.evidence["verifier_region_proposals"][0]
        seed_x, seed_y = proposal["component_seed_pixels"]

        with self.assertRaisesRegex(ActionValidationError, "absent from the live state"):
            environment.step(
                AgentAction("t1", "negative_point", coordinate=(seed_x, seed_y)),
                authorized_region_id="missing",
            )
        with self.assertRaisesRegex(ActionValidationError, "does not match"):
            environment.step(
                AgentAction(
                    "t1",
                    "negative_point",
                    coordinate=((seed_x + 1) % 8, seed_y),
                ),
                authorized_region_id=proposal["region_id"],
            )

    def test_combined_point_change_gate_rejects_only_when_all_limits_exceed(self):
        class RemoveAllPoint:
            def refine(
                self, image, initial_mask, coordinate, is_positive, click_history=()
            ):
                return np.zeros_like(initial_mask)

        verifier = SequenceVerifier(
            [
                VerifierOutput(
                    comparison="initial",
                    error_type="false_positive_change",
                    target_view="t1",
                    suggested_action="negative_point",
                )
            ]
        )
        environment = ChangeAgentEnvironment(
            Backend(),
            ActionExecutor(RemoveAllPoint(), self.box),
            verifier,
            max_steps=2,
            max_point_proposal_change_ratio=0.25,
            max_point_proposal_change_pixels=4,
            max_point_image_change_ratio=0.1,
        )
        environment.reset(self.image1, self.image2, "building")
        proposal = environment.state.evidence["verifier_region_proposals"][0]
        seed = tuple(proposal["component_seed_pixels"])

        environment.step(
            AgentAction("t1", "negative_point", coordinate=seed),
            authorized_region_id=proposal["region_id"],
        )

        entry = environment.trajectory.entries[1]
        self.assertFalse(entry.execution["candidate_accepted"])
        self.assertTrue(entry.execution["verifier_skipped_by_hard_gate"])
        self.assertIn(
            "point_proposal_combined_change_gate_exceeded",
            entry.execution["candidate_rejection_reasons"],
        )
        self.assertEqual(verifier.call_count, 1)
        self.assertGreater(
            entry.execution["point_authorization"]["proposal_change_ratio"],
            0.25,
        )
        gate = entry.execution["point_authorization"]["change_gate"]
        self.assertTrue(gate["proposal_ratio_exceeded"])
        self.assertTrue(gate["absolute_pixels_exceeded"])
        self.assertTrue(gate["image_ratio_exceeded"])
        self.assertTrue(gate["hard_reject"])

    def test_combined_point_change_gate_exempts_small_high_ratio_edit(self):
        class RemoveAllPoint:
            def refine(
                self, image, initial_mask, coordinate, is_positive, click_history=()
            ):
                return np.zeros_like(initial_mask)

        verifier = SequenceVerifier(
            [
                VerifierOutput(
                    comparison="initial",
                    error_type="false_positive_change",
                    target_view="t1",
                    suggested_action="negative_point",
                ),
                VerifierOutput(comparison="worse", accept=False),
            ]
        )
        environment = ChangeAgentEnvironment(
            Backend(),
            ActionExecutor(RemoveAllPoint(), self.box),
            verifier,
            max_steps=2,
        )
        environment.reset(self.image1, self.image2, "building")
        proposal = environment.state.evidence["verifier_region_proposals"][0]

        environment.step(
            AgentAction(
                "t1",
                "negative_point",
                coordinate=tuple(proposal["component_seed_pixels"]),
            ),
            authorized_region_id=proposal["region_id"],
        )

        entry = environment.trajectory.entries[1]
        gate = entry.execution["point_authorization"]["change_gate"]
        self.assertGreater(gate["proposal_change_ratio"], 0.25)
        self.assertFalse(gate["absolute_pixels_exceeded"])
        self.assertFalse(gate["hard_reject"])
        self.assertFalse(entry.execution["verifier_skipped_by_hard_gate"])
        self.assertEqual(verifier.call_count, 2)

    def _run_combined_point_gate_case(
        self,
        *,
        removed_pixels,
        max_proposal_ratio,
        max_changed_pixels,
        max_image_ratio,
        gate_enabled=True,
    ):
        class RemovePixelsPoint:
            def refine(
                self, image, initial_mask, coordinate, is_positive, click_history=()
            ):
                output = np.array(initial_mask, copy=True)
                rows, columns = np.nonzero(output)
                output[rows[:removed_pixels], columns[:removed_pixels]] = False
                return output

        verifier = SequenceVerifier(
            [
                VerifierOutput(
                    comparison="initial",
                    error_type="false_positive_change",
                    target_view="t1",
                    suggested_action="negative_point",
                ),
                VerifierOutput(comparison="worse", accept=False),
            ]
        )
        environment = ChangeAgentEnvironment(
            Backend(),
            ActionExecutor(RemovePixelsPoint(), self.box),
            verifier,
            max_steps=2,
            max_point_proposal_change_ratio=max_proposal_ratio,
            max_point_proposal_change_pixels=max_changed_pixels,
            max_point_image_change_ratio=max_image_ratio,
            point_proposal_change_gate_enabled=gate_enabled,
        )
        environment.reset(self.image1, self.image2, "building")
        proposal = environment.state.evidence["verifier_region_proposals"][0]

        environment.step(
            AgentAction(
                "t1",
                "negative_point",
                coordinate=tuple(proposal["component_seed_pixels"]),
            ),
            authorized_region_id=proposal["region_id"],
        )

        entry = environment.trajectory.entries[1]
        gate = entry.execution["point_authorization"]["change_gate"]
        self.assertEqual(gate["changed_pixels"], removed_pixels)
        self.assertFalse(gate["hard_reject"])
        self.assertFalse(entry.execution["verifier_skipped_by_hard_gate"])
        self.assertNotIn(
            "point_proposal_combined_change_gate_exceeded",
            entry.execution["candidate_rejection_reasons"],
        )
        self.assertEqual(verifier.call_count, 2)
        return gate

    def test_combined_point_change_gate_allows_when_proposal_ratio_is_below_limit(self):
        gate = self._run_combined_point_gate_case(
            removed_pixels=3,
            max_proposal_ratio=0.5,
            max_changed_pixels=2,
            max_image_ratio=0.04,
        )

        self.assertFalse(gate["proposal_ratio_exceeded"])
        self.assertTrue(gate["absolute_pixels_exceeded"])
        self.assertTrue(gate["image_ratio_exceeded"])

    def test_combined_point_change_gate_allows_when_image_ratio_is_below_limit(self):
        gate = self._run_combined_point_gate_case(
            removed_pixels=9,
            max_proposal_ratio=0.25,
            max_changed_pixels=4,
            max_image_ratio=0.2,
        )

        self.assertTrue(gate["proposal_ratio_exceeded"])
        self.assertTrue(gate["absolute_pixels_exceeded"])
        self.assertFalse(gate["image_ratio_exceeded"])

    def test_combined_point_change_gate_allows_when_gate_is_disabled(self):
        gate = self._run_combined_point_gate_case(
            removed_pixels=9,
            max_proposal_ratio=0.25,
            max_changed_pixels=4,
            max_image_ratio=0.1,
            gate_enabled=False,
        )

        self.assertFalse(gate["enabled"])
        self.assertTrue(gate["proposal_ratio_exceeded"])
        self.assertTrue(gate["absolute_pixels_exceeded"])
        self.assertTrue(gate["image_ratio_exceeded"])

    def test_combined_point_change_gate_uses_strict_greater_than_thresholds(self):
        gate = self._run_combined_point_gate_case(
            removed_pixels=9,
            max_proposal_ratio=1.0,
            max_changed_pixels=9,
            max_image_ratio=9 / 64,
        )

        self.assertEqual(gate["proposal_change_ratio"], 1.0)
        self.assertEqual(gate["image_change_ratio"], 9 / 64)
        self.assertFalse(gate["proposal_ratio_exceeded"])
        self.assertFalse(gate["absolute_pixels_exceeded"])
        self.assertFalse(gate["image_ratio_exceeded"])

    def test_authorized_box_must_match_live_environment_proposal(self):
        verifier = SequenceVerifier(
            [
                VerifierOutput(comparison="initial", suggested_action="box"),
                VerifierOutput(comparison="worse", accept=False),
            ]
        )
        environment = ChangeAgentEnvironment(
            Backend(), ActionExecutor(Point(), self.box), verifier, max_steps=2
        )
        environment.reset(self.image1, self.image2, "building")
        proposal = environment.state.evidence["verifier_region_proposals"][0]
        box = tuple(proposal["box_pixels"])

        environment.step(
            AgentAction("t2", "box", box=box),
            authorized_region_id=proposal["region_id"],
        )

        authorization = environment.trajectory.entries[1].execution[
            "box_authorization"
        ]
        self.assertEqual(authorization["region_id"], proposal["region_id"])
        self.assertEqual(authorization["proposal_box_xyxy"], proposal["box_pixels"])
        self.assertTrue(authorization["geometry_match"])
        self.assertEqual(
            authorization["scope_policy"], "environment_live_proposal"
        )

    def test_authorized_box_rejects_geometry_mismatch(self):
        verifier = SequenceVerifier(
            [VerifierOutput(comparison="initial", suggested_action="box")]
        )
        environment = ChangeAgentEnvironment(
            Backend(), ActionExecutor(Point(), self.box), verifier, max_steps=2
        )
        environment.reset(self.image1, self.image2, "building")
        proposal = environment.state.evidence["verifier_region_proposals"][0]

        with self.assertRaisesRegex(ActionValidationError, "does not match"):
            environment.step(
                AgentAction("t2", "box", box=(0, 0, 1, 1)),
                authorized_region_id=proposal["region_id"],
            )

    def test_accepted_candidate_refreshes_full_proposals_for_next_action(self):
        verifier = SequenceVerifier(
            [
                VerifierOutput(
                    comparison="initial",
                    error_type="false_positive_change",
                    target_view="t1",
                    suggested_action="negative_point",
                ),
                VerifierOutput(
                    comparison="better",
                    accept=True,
                    error_type="false_positive_change",
                    target_view="t1",
                    suggested_action="negative_point",
                ),
                VerifierOutput(comparison="worse", accept=False),
            ]
        )
        environment = ChangeAgentEnvironment(
            Backend(), ActionExecutor(Point(), self.box), verifier, max_steps=3
        )
        environment.reset(self.image1, self.image2, "building")
        first = environment.state.evidence["verifier_region_proposals"][0]

        environment.step(
            AgentAction(
                "t1",
                "negative_point",
                coordinate=tuple(first["component_seed_pixels"]),
            ),
            authorized_region_id=first["region_id"],
        )

        refresh = environment.trajectory.entries[1].execution[
            "live_proposal_refresh"
        ]
        self.assertEqual(
            refresh["policy"], "full_state_after_accepted_candidate"
        )
        self.assertTrue(refresh["region_ids"])
        self.assertTrue(
            all(region_id.startswith(("r", "s")) for region_id in refresh["region_ids"])
        )
        second = next(
            proposal
            for proposal in environment.state.evidence[
                "verifier_region_proposals"
            ]
            if proposal["component_seed_t1_mask_white"]
        )

        environment.step(
            AgentAction(
                "t1",
                "negative_point",
                coordinate=tuple(second["component_seed_pixels"]),
            ),
            authorized_region_id=second["region_id"],
        )

        self.assertEqual(len(environment.trajectory.entries), 3)

    def test_pairwise_worse_candidate_is_rejected_without_scores(self):
        initial_feedback = VerifierOutput(
            comparison="initial",
            error_type="false_negative",
            target_view="t2",
            error_region=(0, 0, 100, 100),
            suggested_action="positive_point",
            feedback="Initial feedback.",
        )
        candidate_feedback = VerifierOutput(
            comparison="worse",
            error_type="false_positive_change",
            target_view="t2",
            error_region=(0, 0, 100, 100),
            suggested_action="negative_point",
            feedback="Candidate is worse.",
        )
        environment = ChangeAgentEnvironment(
            Backend(),
            ActionExecutor(Point(), self.box),
            SequenceVerifier([initial_feedback, candidate_feedback]),
            max_steps=2,
        )
        environment.reset(self.image1, self.image2, "building")

        environment.step(AgentAction("t2", "positive_point", coordinate=(0, 0)))

        entry = environment.trajectory.entries[1]
        self.assertFalse(entry.execution["candidate_accepted"])
        self.assertEqual(entry.execution["pairwise_comparison"], "worse")
        self.assertIn(
            "candidate_effect_not_better",
            entry.execution["candidate_rejection_reasons"],
        )

    def test_pairwise_better_but_verifier_rejected_candidate_is_rolled_back(self):
        initial_feedback = VerifierOutput(
            comparison="initial",
            error_type="false_negative",
            target_view="t2",
            error_region=(0, 0, 100, 100),
            suggested_action="positive_point",
            feedback="Initial feedback.",
        )
        candidate_feedback = VerifierOutput(
            comparison="better",
            error_type="false_negative",
            target_view="t2",
            error_region=(0, 0, 100, 100),
            suggested_action="positive_point",
            feedback="Relatively better but still insufficient.",
            accept=False,
        )
        environment = ChangeAgentEnvironment(
            Backend(),
            ActionExecutor(Point(), self.box),
            SequenceVerifier([initial_feedback, candidate_feedback]),
            max_steps=2,
        )
        environment.reset(self.image1, self.image2, "building")
        accepted_mask = environment.state.change_mask.copy()

        environment.step(AgentAction("t2", "positive_point", coordinate=(0, 0)))

        entry = environment.trajectory.entries[1]
        self.assertFalse(entry.execution["candidate_accepted"])
        self.assertEqual(entry.execution["pairwise_comparison"], "better")
        self.assertIn(
            "verifier_not_accepted", entry.execution["candidate_rejection_reasons"]
        )
        self.assertTrue(np.array_equal(environment.state.change_mask, accepted_mask))
        self.assertEqual(environment.trajectory.best_entry.step_index, 0)

    def test_locality_gate_rejects_global_point_component(self):
        class GlobalPoint:
            def refine(self, image, initial_mask, coordinate, is_positive, click_history=()):
                return np.ones_like(initial_mask)

        initial_feedback = VerifierOutput(
            quality_score=0.3,
            error_type="false_negative",
            target_view="t2",
            error_region=(0, 0, 100, 100),
            suggested_action="positive_point",
            feedback="Initial feedback.",
        )
        candidate_feedback = VerifierOutput(
            quality_score=0.8,
            score_delta=0.5,
            progress_score=0.5,
            error_type="none",
            feedback="Candidate feedback.",
        )
        verifier = SequenceVerifier([initial_feedback, candidate_feedback])
        environment = ChangeAgentEnvironment(
            Backend(),
            ActionExecutor(GlobalPoint(), self.box),
            verifier,
            max_steps=3,
            max_selection_area_delta=1.0,
            max_locality_outside_ratio=0.0,
            max_target_mask_change_ratio=1.0,
            max_component_count_delta=100,
        )
        environment.reset(self.image1, self.image2, "building")
        environment.step(AgentAction("t2", "positive_point", coordinate=(0, 0)))

        entry = environment.trajectory.entries[1]
        self.assertFalse(entry.execution["candidate_accepted"])
        self.assertIn(
            "locality_outside_roi_exceeded",
            entry.execution["candidate_rejection_reasons"],
        )
        self.assertTrue(entry.execution["verifier_skipped_by_hard_gate"])
        self.assertEqual(verifier.call_count, 1)

    def test_rollback_replans_from_accepted_state_without_reusing_action(self):
        class GlobalPoint:
            def refine(self, image, initial_mask, coordinate, is_positive, click_history=()):
                return np.ones_like(initial_mask)

        initial_feedback = VerifierOutput(
            comparison="initial",
            error_type="false_negative",
            target_view="t2",
            error_region=(0, 0, 100, 100),
            suggested_action="positive_point",
            feedback="Initial feedback.",
        )
        replacement_feedback = VerifierOutput(
            comparison="uncertain",
            error_type="false_negative",
            target_view="t2",
            error_region=(700, 700, 700, 700),
            suggested_action="positive_point",
            feedback="Use a different, local point.",
        )
        verifier = ReplanningVerifier(initial_feedback, replacement_feedback)
        environment = ChangeAgentEnvironment(
            Backend(),
            ActionExecutor(GlobalPoint(), self.box),
            verifier,
            max_steps=3,
            max_selection_area_delta=1.0,
            max_locality_outside_ratio=0.0,
            max_target_mask_change_ratio=1.0,
            max_component_count_delta=100,
        )
        environment.reset(self.image1, self.image2, "building")
        accepted_mask = environment.state.change_mask.copy()

        observation, _ = environment.step(
            AgentAction("t2", "positive_point", coordinate=(0, 0))
        )

        self.assertTrue(np.array_equal(environment.state.change_mask, accepted_mask))
        self.assertEqual(observation.feedback.error_region, (700, 700, 700, 700))
        self.assertEqual(len(verifier.replan_calls), 1)
        self.assertEqual(
            verifier.replan_calls[0]["rejection_reasons"],
            ("locality_outside_roi_exceeded",),
        )
        self.assertEqual(
            verifier.replan_calls[0]["rejected_action"].coordinate, (0, 0)
        )
        self.assertEqual(verifier.replan_calls[0]["rejection_history"][0]["step_index"], 1)
        self.assertEqual(
            verifier.replan_calls[0]["rejection_history"][0]["rejection_reasons"],
            ["locality_outside_roi_exceeded"],
        )
        entry = environment.trajectory.entries[1]
        self.assertIn("rollback_replan", entry.execution)
        self.assertEqual(
            entry.execution["rollback_replan"]["verifier_output"]["error_region"],
            [700, 700, 700, 700],
        )

    def test_verifier_feedback_declares_normalized_public_coordinates(self):
        observation = self.environment.reset(self.image1, self.image2, "building")
        feedback = observation.feedback.to_dict()
        self.assertEqual(feedback["coordinate_space"], "normalized_0_1000")

    def test_raw_action_and_trajectory_artifacts(self):
        self.environment.reset(self.image1, self.image2, "building")
        raw = '{"target_view":"t1","action":"positive_point","coordinate":[0,0]}'
        self.environment.step(raw)
        with tempfile.TemporaryDirectory() as directory:
            path = self.environment.trajectory.save(directory)
            payload = json.loads(path.read_text())
            self.assertEqual(payload["steps"][1]["raw_action"], raw)
            self.assertNotIn("coordinate_frame", payload["steps"][1]["raw_action_payload"])
            self.assertEqual(
                payload["steps"][1]["raw_action_payload"]["coordinate"], [0, 0]
            )
            self.assertEqual(len(payload["steps"][1]["t1_mask_sha256"]), 64)
            self.assertEqual(len(payload["steps"][1]["change_mask_sha256"]), 64)
            self.assertTrue((Path(directory) / "masks" / "step_001.npy").exists())

    def test_repeated_small_public_coordinates_are_warned_not_autocorrected(self):
        self.environment.reset(self.image1, self.image2, "building")
        raw = '{"target_view":"t1","action":"positive_point","coordinate":[100,200]}'
        self.environment.step(raw)
        self.environment.step(
            '{"target_view":"t1","action":"positive_point","coordinate":[240,240]}'
        )
        warning = self.environment.trajectory.entries[2].execution["coordinate_warning"]
        self.assertIn("not auto-corrected to pixels", warning)

    def test_exact_rejected_action_cannot_repeat_on_same_live_state(self):
        initial_feedback = VerifierOutput(
            comparison="initial",
            error_type="false_negative",
            target_view="t2",
            error_region=(0, 0, 100, 100),
            suggested_action="positive_point",
            feedback="Initial feedback.",
        )
        rejected_feedback = VerifierOutput(
            comparison="worse",
            error_type="false_positive_change",
            target_view="t2",
            error_region=(0, 0, 100, 100),
            suggested_action="negative_point",
            feedback="Candidate is worse.",
        )
        environment = ChangeAgentEnvironment(
            Backend(),
            ActionExecutor(Point(), self.box),
            SequenceVerifier([initial_feedback, rejected_feedback]),
            max_steps=3,
        )
        environment.reset(self.image1, self.image2, "building")
        raw = '{"target_view":"t2","action":"positive_point","coordinate":[0,0]}'
        environment.step(raw)

        self.assertIn("rejected_action=", environment.observation().history_summary)
        self.assertIn('"coordinate":[0,0]', environment.observation().history_summary)

        with self.assertRaisesRegex(ActionValidationError, "exactly repeats"):
            environment.step(raw)

    def test_initial_selection_policy_keeps_initial_state(self):
        environment = ChangeAgentEnvironment(
            Backend(),
            ActionExecutor(Point(), self.box),
            EvidenceVerifier(),
            max_steps=4,
            selection_policy="initial",
        )
        environment.reset(self.image1, self.image2, "building")
        environment.step(AgentAction("t2", "box", box=(2, 2, 5, 5)))
        self.assertEqual(environment.trajectory.best_entry.step_index, 0)

    def test_invalid_verifier_candidate_is_recorded_then_rolled_back(self):
        initial_feedback = VerifierOutput(
            quality_score=0.3,
            error_type="uncertain_region",
            target_view="t2",
            error_region=(0, 0, 1000, 1000),
            suggested_action="box",
            feedback="Initial state needs review.",
        )
        invalid_feedback = VerifierOutput(
            quality_score=0.9,
            error_type="none",
            target_view="t2",
            feedback="Malformed verifier result.",
            verifier_valid=False,
            localization_valid=False,
        )
        environment = ChangeAgentEnvironment(
            Backend(),
            ActionExecutor(Point(), self.box),
            SequenceVerifier([initial_feedback, invalid_feedback]),
            max_steps=3,
        )
        environment.reset(self.image1, self.image2, "building")
        accepted_mask = environment.state.change_mask.copy()

        environment.step(AgentAction("t2", "positive_point", coordinate=(0, 0)))

        rejected = environment.trajectory.entries[1]
        self.assertFalse(rejected.execution["candidate_accepted"])
        self.assertIn("verifier_invalid", rejected.execution["candidate_rejection_reasons"])
        self.assertFalse(np.array_equal(rejected.state.change_mask, accepted_mask))
        self.assertTrue(np.array_equal(environment.state.change_mask, accepted_mask))
        self.assertTrue(
            np.array_equal(
                environment.verifier.previous_states[1].change_mask, accepted_mask
            )
        )
        self.assertEqual(environment.state.step_index, 1)
        self.assertIs(environment.feedback, initial_feedback)
        self.assertEqual(environment.trajectory.best_entry.step_index, 0)

    def test_excessive_mask_area_candidate_is_rolled_back(self):
        initial_feedback = VerifierOutput(
            quality_score=0.3,
            error_type="uncertain_region",
            target_view="t2",
            error_region=(0, 0, 1000, 1000),
            suggested_action="box",
            feedback="Initial state needs review.",
        )
        optimistic_feedback = VerifierOutput(
            quality_score=0.95,
            score_delta=0.65,
            error_type="none",
            target_view="t2",
            feedback="Candidate looks good.",
            accept=True,
        )
        verifier = SequenceVerifier([initial_feedback, optimistic_feedback])
        environment = ChangeAgentEnvironment(
            Backend(),
            ActionExecutor(Point(), self.box),
            verifier,
            max_steps=3,
            max_selection_area_delta=0.0,
        )
        environment.reset(self.image1, self.image2, "building")
        accepted_mask = environment.state.change_mask.copy()

        environment.step(AgentAction("t2", "positive_point", coordinate=(0, 0)))

        rejected = environment.trajectory.entries[1]
        self.assertFalse(rejected.execution["candidate_accepted"])
        self.assertIn(
            "mask_area_delta_exceeded",
            rejected.execution["candidate_rejection_reasons"],
        )
        self.assertEqual(rejected.verifier.comparison, "worse")
        self.assertTrue(rejected.execution["verifier_skipped_by_hard_gate"])
        self.assertEqual(verifier.call_count, 1)
        self.assertTrue(np.array_equal(environment.state.change_mask, accepted_mask))
        self.assertEqual(environment.trajectory.best_entry.step_index, 0)

    def test_finish_is_rejected_before_any_tool_action(self):
        self.environment.reset(self.image1, self.image2, "building")
        with self.assertRaises(ActionValidationError):
            self.environment.step(AgentAction("t2", "finish"))

    def test_initial_verified_error_free_state_can_finish_without_tool(self):
        class MatchedBackend(Backend):
            def initialize(self, t1_image, t2_image, query):
                t1 = np.zeros(t1_image.shape[:2], dtype=bool)
                t1[2:5, 2:5] = True
                return InitializationResult(t1, t1.copy(), self.processor.rebuild(t1, t1))

        class InitialClearVerifier:
            def verify(self, state, previous_score, previous_action, previous_state=None):
                return VerifierOutput(
                    comparison="initial" if previous_state is None else "unchanged",
                    error_type="none",
                    feedback="All inspected regions are supported.",
                    suggested_action="finish",
                    accept=True,
                    stop=True,
                )

        environment = ChangeAgentEnvironment(
            MatchedBackend(),
            ActionExecutor(Point(), self.box),
            InitialClearVerifier(),
            max_steps=2,
        )
        environment.reset(self.image1, self.image2, "building")

        _, done = environment.step(AgentAction("t2", "finish"))

        self.assertTrue(done)

    def test_trajectory_can_save_masks_in_a_separate_directory(self):
        self.environment.reset(self.image1, self.image2, "building")
        self.environment.step(AgentAction("t1", "positive_point", coordinate=(0, 0)))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trajectory_path = self.environment.trajectory.save(
                root / "trajectories" / "sample", root / "masks" / "sample"
            )
            payload = json.loads(trajectory_path.read_text())
            mask_path = (trajectory_path.parent / payload["steps"][1]["change_mask_file"]).resolve()
            self.assertTrue(mask_path.exists())
            self.assertEqual(mask_path, (root / "masks" / "sample" / "step_001.npy").resolve())

    def test_runtime_rejects_non_inference_mode(self):
        with self.assertRaises(ValueError):
            ChangeAgentEnvironment(
                Backend(),
                ActionExecutor(Point(), self.box),
                EvidenceVerifier(),
                inference_only=False,
            )

    def test_accepted_point_rebases_session_to_exact_live_mask(self):
        class RecordingPoint:
            def __init__(self):
                self.calls = []

            def refine(
                self,
                image,
                initial_mask,
                coordinate,
                is_positive,
                click_history=(),
            ):
                self.calls.append(
                    (
                        np.array(initial_mask, copy=True),
                        coordinate,
                        is_positive,
                        click_history,
                    )
                )
                output = np.array(initial_mask, copy=True)
                x, y = coordinate
                output[y, x] = is_positive
                return output

        point = RecordingPoint()
        verifier = SequenceVerifier(
            [
                VerifierOutput(quality_score=0.2, suggested_action="positive_point"),
                VerifierOutput(quality_score=0.4, progress_score=0.2),
                VerifierOutput(quality_score=0.1, progress_score=-0.3),
                VerifierOutput(quality_score=0.5, progress_score=0.1),
            ]
        )
        environment = ChangeAgentEnvironment(
            Backend(),
            ActionExecutor(point, self.box),
            verifier,
            max_steps=4,
            max_selection_area_delta=1.0,
            max_locality_outside_ratio=1.0,
            max_target_mask_change_ratio=1.0,
            max_component_count_delta=100,
        )
        environment.reset(self.image1, self.image2, "building")
        environment.step(AgentAction("t2", "positive_point", coordinate=(0, 0)))
        environment.step(AgentAction("t2", "negative_point", coordinate=(1, 1)))
        environment.step(AgentAction("t2", "positive_point", coordinate=(2, 2)))

        self.assertTrue(all(call[3] == () for call in point.calls))
        self.assertFalse(point.calls[0][0].any())
        self.assertTrue(point.calls[1][0][0, 0])
        self.assertTrue(point.calls[2][0][0, 0])
        self.assertFalse(
            environment.trajectory.entries[2].execution["candidate_accepted"]
        )
        first_update = environment.trajectory.entries[1].execution[
            "point_session_update"
        ]
        self.assertEqual(first_update["policy"], "rebase_to_accepted_live_mask")
        self.assertEqual(first_update["target_view"], "t2")
        self.assertEqual(first_update["base_mask_pixels"], 1)
        self.assertNotIn(
            "point_session_update",
            environment.trajectory.entries[2].execution,
        )

    def test_accepted_box_starts_a_new_point_session(self):
        class RecordingPoint:
            def __init__(self):
                self.calls = []

            def refine(
                self,
                image,
                initial_mask,
                coordinate,
                is_positive,
                click_history=(),
            ):
                self.calls.append((np.array(initial_mask, copy=True), click_history))
                output = np.array(initial_mask, copy=True)
                x, y = coordinate
                output[y, x] = is_positive
                return output

        point = RecordingPoint()
        verifier = SequenceVerifier(
            [
                VerifierOutput(quality_score=0.2, suggested_action="positive_point"),
                VerifierOutput(quality_score=0.3, progress_score=0.1),
                VerifierOutput(quality_score=0.4, progress_score=0.1),
                VerifierOutput(quality_score=0.5, progress_score=0.1),
            ]
        )
        environment = ChangeAgentEnvironment(
            Backend(),
            ActionExecutor(point, self.box),
            verifier,
            max_steps=4,
            max_selection_area_delta=1.0,
            max_locality_outside_ratio=1.0,
            max_target_mask_change_ratio=1.0,
            max_component_count_delta=100,
        )
        environment.reset(self.image1, self.image2, "building")
        environment.step(AgentAction("t2", "positive_point", coordinate=(0, 0)))
        environment.step(AgentAction("t2", "box", box=(2, 2, 5, 5)))
        box_mask = np.array(environment.state.t2_mask, copy=True)
        environment.step(AgentAction("t2", "positive_point", coordinate=(6, 6)))

        self.assertEqual(point.calls[1][1], ())
        self.assertTrue(np.array_equal(point.calls[1][0], box_mask))


if __name__ == "__main__":
    unittest.main()
