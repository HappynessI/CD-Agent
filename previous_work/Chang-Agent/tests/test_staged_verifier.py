import unittest
import json

import numpy as np

from change_agent.adapters.omniovcd_adapter import (
    InitializationResult,
    MaskPairProcessor,
)
from change_agent.adapters.staged_verifier import StagedQwenVerifier
from change_agent.adapters.stage_backends import _extract_stage_json, _stage_prompt
from change_agent.coordinates import normalized_point_to_pixel
from change_agent.environment import ChangeAgentEnvironment
from change_agent.executor import ActionExecutor
from change_agent.state import AgentAction, ChangeState, VerifierOutput
from change_agent.verifier_protocol import (
    Diagnosis,
    EvidenceJudgment,
    EvidenceRecord,
    StageProtocolError,
)
from change_agent.verifier_regions import attach_verifier_regions
from tools.run_levir_change_agent import _execute_verifier_action


def make_state():
    image1 = np.zeros((16, 16, 3), dtype=np.uint8)
    image2 = image1.copy()
    image2[4:9, 4:9] = 180
    t1 = np.zeros((16, 16), dtype=bool)
    t2 = np.zeros_like(t1)
    t2[4:9, 4:9] = True
    state = ChangeState(image1, image2, "building", t1, t2, t2)
    attach_verifier_regions(state, max_regions=6, min_component_area=1)
    return state


def make_two_region_state():
    image1 = np.zeros((32, 32, 3), dtype=np.uint8)
    image2 = image1.copy()
    t1 = np.zeros((32, 32), dtype=bool)
    t2 = np.zeros_like(t1)
    t2[3:8, 3:8] = True
    t2[20:26, 20:26] = True
    state = ChangeState(image1, image2, "building", t1, t2, t2)
    attach_verifier_regions(state, max_regions=6, min_component_area=1)
    return state


def make_editability_conflict_state():
    image1 = np.zeros((16, 16, 3), dtype=np.uint8)
    image2 = image1.copy()
    image1[4:9, 4:9] = 180
    t1 = np.zeros((16, 16), dtype=bool)
    t1[4:9, 4:9] = True
    t2 = np.zeros_like(t1)
    change = np.zeros_like(t1)
    state = ChangeState(image1, image2, "building", t1, t2, change)
    attach_verifier_regions(state, max_regions=6, min_component_area=1)
    return state


def attach_candidate_and_live_regions(candidate, previous):
    """Mirror the Environment's candidate-delta plus accepted-live catalogs."""

    attach_verifier_regions(candidate, previous, max_regions=6, min_component_area=1)
    live_snapshot = candidate.clone()
    attach_verifier_regions(live_snapshot, max_regions=6, min_component_area=1)
    candidate.evidence["verifier_next_live_region_proposals"] = (
        live_snapshot.evidence["verifier_region_proposals"]
    )
    candidate.evidence["verifier_next_live_mask_facts"] = live_snapshot.evidence[
        "verifier_mask_facts"
    ]


def audit_checklist_for(error):
    statuses = {
        "evidence_sufficient": "pass",
        "target_class_only": "pass",
        "white_pixels_supported": "pass",
        "boundary_alignment": "pass",
        "internal_holes_absent": "pass",
        "changed_object_extent_complete": "pass",
        "fragment_artifacts_absent": "pass",
    }
    if error == "false_positive_change":
        statuses["white_pixels_supported"] = "fail"
    elif error == "false_negative":
        statuses["changed_object_extent_complete"] = "fail"
    elif error == "mixed_error":
        statuses["white_pixels_supported"] = "fail"
        statuses["changed_object_extent_complete"] = "fail"
    elif error == "uncertain_region":
        statuses = {key: "uncertain" for key in statuses}
    return statuses


def grounded_audit_checklist_for(error):
    return {
        name: {
            "status": status,
            "evidence": f"Observable evidence for {name}: {status}.",
        }
        for name, status in audit_checklist_for(error).items()
    }


class ScriptedBackend:
    def __init__(
        self,
        *,
        t1="background",
        t2="building",
        error="none",
        target=None,
        action="negative_point",
        evidence_quality="clear",
    ):
        self.t1 = t1
        self.t2 = t2
        self.error = error
        self.target = target
        self.action = action
        self.evidence_quality = evidence_quality
        self.calls = []
        self.previous_seen = []
        self.payloads = []

    def generate_stage(self, stage, state, payload, previous_state=None):
        self.calls.append(stage)
        self.previous_seen.append(previous_state is not None)
        self.payloads.append(payload)
        if stage == "select":
            return {
                "selection": {
                    "region_ids": [payload["proposal_catalog"][0]["region_id"]],
                    "reason": "Most material marked proposal.",
                }
            }
        region = payload.get("region", {})
        region_id = region.get("region_id")
        if stage == "audit":
            return {
                "region_id": region_id,
                "visual_judgment": {
                    "t1_state": self.t1,
                    "t2_state": self.t2,
                    "evidence": "Paired RGB and exact component are observable.",
                },
                "diagnosis": {
                    "audit_checklist": audit_checklist_for(self.error),
                },
            }
        if stage in {"evidence", "candidate_evidence"}:
            return {
                "region_id": region_id,
                "visual_judgment": {
                    "t1_state": self.t1,
                    "t2_state": self.t2,
                    "evidence_quality": self.evidence_quality,
                },
            }
        if stage == "diagnosis":
            return {
                "region_id": region_id,
                "diagnosis": {
                    "audit_checklist": audit_checklist_for(self.error),
                    "target_view": self.target,
                },
            }
        if stage == "plan":
            return {
                "region_id": region_id,
                "plan": {
                    "action": self.action,
                    "target_view": self.target,
                    "coordinate_normalized_1000": region["component_seed_normalized_1000"],
                    "box_normalized_1000": None,
                },
            }
        return {
            "decision": {
                "feedback": "Structured staged decision.",
            }
        }


class RepairingBackend(ScriptedBackend):
    def __init__(self):
        super().__init__(error="none", target=None)
        self.repair_errors = []

    def generate_stage(self, stage, state, payload, previous_state=None):
        if stage == "audit":
            self.calls.append(stage)
            return {"region": payload["region"], "schema": payload["schema"]}
        return super().generate_stage(stage, state, payload, previous_state)

    def repair_stage(
        self, stage, state, payload, validation_error, previous_state=None
    ):
        self.calls.append(f"repair:{stage}")
        self.repair_errors.append(validation_error)
        return ScriptedBackend.generate_stage(
            self, stage, state, payload, previous_state
        )


class MissingDiagnosisConfidenceBackend(ScriptedBackend):
    def generate_stage(self, stage, state, payload, previous_state=None):
        if stage == "diagnosis":
            self.calls.append(stage)
            return {
                "region_id": payload["region"]["region_id"],
                "diagnosis": {"error_type": "none", "target_view": None},
            }
        return super().generate_stage(stage, state, payload, previous_state)


class LowQualityInitialBackend(ScriptedBackend):
    def generate_stage(self, stage, state, payload, previous_state=None):
        response = super().generate_stage(stage, state, payload, previous_state)
        if stage == "decision" and payload["mode"] == "initial":
            response["decision"]["quality_score"] = 0.0
        return response


class SelectAllBackend(ScriptedBackend):
    def generate_stage(self, stage, state, payload, previous_state=None):
        if stage == "select":
            self.calls.append(stage)
            self.previous_seen.append(previous_state is not None)
            return {
                "selection": {
                    "region_ids": [
                        item["region_id"] for item in payload["proposal_catalog"]
                    ],
                    "reason": "Audit both marked regions.",
                }
            }
        return super().generate_stage(stage, state, payload, previous_state)


class StagedVerifierTest(unittest.TestCase):
    def test_completion_gate_prefers_next_live_mask_facts(self):
        state = make_state()
        state.evidence["verifier_mask_facts"][
            "initial_audit_uncovered_pixels"
        ] = 0
        state.evidence["verifier_next_live_mask_facts"] = {
            "initial_audit_uncovered_pixels": 1,
        }
        verifier = StagedQwenVerifier(ScriptedBackend())

        passed, reason = verifier._state_completion_gate(
            state, (), (), (), ()
        )

        self.assertFalse(passed)
        self.assertEqual(reason, "environment audit mask has uncovered pixels")

    def test_completion_gate_rejects_missing_coverage_fact(self):
        state = make_state()
        state.evidence["verifier_next_live_mask_facts"] = {}
        verifier = StagedQwenVerifier(ScriptedBackend())

        passed, reason = verifier._state_completion_gate(
            state, (), (), (), ()
        )

        self.assertFalse(passed)
        self.assertEqual(
            reason, "environment audit uncovered-pixel fact is invalid"
        )

    def test_two_accepted_actions_replan_against_promoted_live_catalog(self):
        class TwoComponentBackend:
            def __init__(self):
                self.processor = MaskPairProcessor(overlap_threshold=0.5)

            def initialize(self, t1_image, t2_image, query):
                t1_mask = np.zeros(t1_image.shape[:2], dtype=bool)
                t1_mask[2, 2] = True
                t1_mask[6, 6] = True
                t2_mask = np.zeros_like(t1_mask)
                return InitializationResult(
                    t1_mask,
                    t2_mask,
                    self.processor.rebuild(t1_mask, t2_mask),
                )

            def rebuild(self, t1_mask, t2_mask, evidence):
                return self.processor.rebuild(t1_mask, t2_mask, evidence)

        class RemoveClickedPoint:
            def refine(
                self,
                image,
                initial_mask,
                coordinate,
                is_positive,
                click_history=(),
            ):
                if is_positive:
                    raise AssertionError("integration expects negative points only")
                result = np.array(initial_mask, dtype=bool, copy=True)
                x, y = coordinate
                result[y, x] = False
                return result

        class UnexpectedBox:
            def segment_box(self, image, box_cxcywh_normalized, query):
                raise AssertionError(
                    "two-point integration must not call box backend"
                )

        verifier = StagedQwenVerifier(
            ScriptedBackend(
                t1="background",
                t2="background",
                error="false_positive_change",
                target="t1",
            )
        )
        environment = ChangeAgentEnvironment(
            TwoComponentBackend(),
            ActionExecutor(RemoveClickedPoint(), UnexpectedBox()),
            verifier,
            max_steps=3,
            verifier_min_region_area=1,
        )
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        observation = environment.reset(image, image.copy(), "building")

        observation, errors, executed = _execute_verifier_action(
            environment, observation, loop_index=1
        )

        self.assertEqual(errors, [])
        self.assertTrue(executed)
        self.assertTrue(
            environment.trajectory.entries[-1].execution["candidate_accepted"]
        )
        plan = verifier.last_evidence["stage_trace"]["plan"]
        planned_region_id = plan["region_id"]
        live_proposal = next(
            item
            for item in environment.state.evidence["verifier_region_proposals"]
            if item["region_id"] == planned_region_id
        )
        self.assertEqual(
            tuple(plan["coordinate_normalized_1000"]),
            tuple(live_proposal["component_seed_normalized"]),
        )
        self.assertEqual(
            normalized_point_to_pixel(
                plan["coordinate_normalized_1000"],
                environment.state.image_size,
            ),
            tuple(live_proposal["component_seed_pixels"]),
        )

        observation, errors, executed = _execute_verifier_action(
            environment, observation, loop_index=2
        )

        self.assertEqual(errors, [])
        self.assertTrue(executed)
        final_entry = environment.trajectory.entries[-1]
        self.assertTrue(final_entry.execution["candidate_accepted"])
        self.assertEqual(
            final_entry.execution["action_generation"]["authorized_region_id"],
            planned_region_id,
        )
        self.assertFalse(environment.state.change_mask.any())

    def test_environment_target_view_facts_have_stable_t1_t2_order(self):
        proposal = make_state().evidence["verifier_region_proposals"][0]

        record = EvidenceRecord.from_proposal(proposal)

        self.assertEqual(list(record.editable_seed_white), ["t1", "t2"])

    def test_rollback_replan_uses_distinct_cached_region(self):
        state = make_two_region_state()
        backend = SelectAllBackend(
            error="false_positive_change",
            target="t2",
        )
        verifier = StagedQwenVerifier(backend)
        accepted_feedback = verifier.verify(state, None, None)
        first_normalized = accepted_feedback.error_region[:2]
        first_pixel = normalized_point_to_pixel(first_normalized, state.image_size)
        rejected_action = AgentAction(
            "t2", "negative_point", coordinate=first_pixel
        )

        output = verifier.replan_after_rejection(
            state,
            state.clone(),
            accepted_feedback,
            VerifierOutput(comparison="worse", accept=False),
            rejected_action,
            ["candidate_effect_not_better"],
            [{"action": rejected_action.to_dict()}],
        )

        self.assertTrue(output.verifier_valid)
        self.assertEqual(output.suggested_action, "negative_point")
        self.assertNotEqual(output.error_region[:2], first_normalized)
        self.assertIsNotNone(verifier._planned_diagnosis)
        self.assertEqual(
            verifier._planned_diagnosis.region_id,
            verifier.last_evidence["replan"]["selected_region_id"],
        )
        self.assertEqual(
            verifier._planned_action,
            AgentAction(
                "t2",
                "negative_point",
                coordinate=normalized_point_to_pixel(
                    output.error_region[:2], state.image_size
                ),
            ),
        )

    def test_empty_dual_mask_scan_authorizes_environment_box_and_candidate(self):
        image1 = np.zeros((18, 18, 3), dtype=np.uint8)
        image2 = image1.copy()
        image2[12:17, 12:17] = 180
        empty = np.zeros((18, 18), dtype=bool)
        state = ChangeState(image1, image2, "building", empty, empty, empty)
        attach_verifier_regions(
            state,
            max_regions=3,
            empty_mask_scan_grid_size=3,
        )
        verifier = StagedQwenVerifier(
            ScriptedBackend(
                t1="background",
                t2="building",
                error="false_negative",
                target="t2",
            ),
            max_regions=16,
        )

        output = verifier.verify(state, None, None)

        self.assertTrue(output.verifier_valid)
        self.assertEqual(output.error_type, "false_negative")
        self.assertEqual(output.suggested_action, "box")
        self.assertIsNotNone(output.error_region)
        self.assertIsNotNone(verifier._planned_diagnosis)
        self.assertTrue(
            verifier._planned_diagnosis.region_id.startswith("s")
        )
        box = verifier.last_evidence["stage_trace"]["plan"][
            "box_normalized_1000"
        ]
        x1, y1 = normalized_point_to_pixel(box[:2], state.image_size)
        x2, y2 = normalized_point_to_pixel(box[2:], state.image_size)
        self.assertEqual(
            verifier._planned_action,
            AgentAction("t2", "box", box=(x1, y1, x2, y2)),
        )

        candidate_t2 = state.t2_mask.copy()
        candidate_t2[y1 : y2 + 1, x1 : x2 + 1] = True
        candidate = ChangeState(
            state.t1_image,
            state.t2_image,
            state.query,
            state.t1_mask.copy(),
            candidate_t2,
            np.logical_xor(state.t1_mask, candidate_t2),
        )
        attach_candidate_and_live_regions(candidate, state)
        action = verifier._planned_action
        assert action is not None

        candidate_output = verifier.verify(
            candidate,
            output.quality_score,
            action,
            state,
        )

        self.assertEqual(candidate_output.comparison, "better")
        self.assertTrue(candidate_output.accept)
        self.assertIn("candidate_evidence", verifier.backend.calls)

    def test_box_recovery_candidate_uses_persisted_false_negative_direction(self):
        verifier = StagedQwenVerifier(ScriptedBackend())
        verifier._planned_diagnosis = Diagnosis(
            "s8",
            "false_negative",
            "t2",
            None,
        )
        record = EvidenceRecord(
            "d0",
            "delta_added",
            "white",
            "absent",
            (600, 600, 1000, 1000),
            (800, 800),
        )
        verifier._accepted_records = (
            EvidenceRecord(
                "s8",
                "scan",
                "black",
                "absent",
                (600, 600, 1000, 1000),
                (800, 800),
            ),
        )
        verifier._planned_action = AgentAction(
            "t2", "box", box=(10, 10, 20, 20)
        )
        judgment = EvidenceJudgment(
            "d0",
            "background",
            "building",
            "clear",
        )

        assessment = verifier._assess_candidate_transition(
            (record,),
            (judgment,),
            AgentAction("t2", "box", box=(10, 10, 20, 20)),
        )

        self.assertTrue(assessment.intended_error_improved)
        self.assertFalse(assessment.introduced_harm)
        self.assertEqual(assessment.comparison, "better")

    def test_added_change_opposite_to_target_view_fails_closed(self):
        verifier = StagedQwenVerifier(ScriptedBackend())
        record = EvidenceRecord(
            "d0",
            "delta_added",
            "white",
            "absent",
            (600, 600, 1000, 1000),
            (800, 800),
        )
        cases = (
            ("t1", "background", "building"),
            ("t2", "building", "background"),
        )
        for target_view, t1_state, t2_state in cases:
            with self.subTest(target_view=target_view):
                verifier._planned_diagnosis = Diagnosis(
                    "r0",
                    "false_negative",
                    target_view,
                    None,
                )
                verifier._accepted_records = (
                    EvidenceRecord(
                        "r0",
                        "missing",
                        "black",
                        "present",
                        (600, 600, 1000, 1000),
                        (800, 800),
                    ),
                )
                verifier._planned_action = AgentAction(
                    target_view,
                    "positive_point",
                    coordinate=(10, 10),
                )
                assessment = verifier._assess_candidate_transition(
                    (record,),
                    (
                        EvidenceJudgment(
                            "d0",
                            t1_state,
                            t2_state,
                            "clear",
                        ),
                    ),
                    AgentAction(
                        target_view,
                        "positive_point",
                        coordinate=(10, 10),
                    ),
                )

                self.assertFalse(assessment.intended_error_improved)
                self.assertTrue(assessment.introduced_false_positive)
                self.assertEqual(assessment.comparison, "worse")
                self.assertIn(
                    f"target_transition_opposes_{target_view}",
                    assessment.evidence,
                )

    def test_candidate_target_must_match_persisted_diagnosis(self):
        verifier = StagedQwenVerifier(ScriptedBackend())
        verifier._planned_diagnosis = Diagnosis(
            "s8",
            "false_negative",
            "t2",
            None,
        )
        verifier._accepted_records = (
            EvidenceRecord(
                "s8",
                "scan",
                "black",
                "absent",
                (600, 600, 1000, 1000),
                (800, 800),
            ),
        )
        verifier._planned_action = AgentAction(
            "t2", "box", box=(10, 10, 20, 20)
        )
        assessment = verifier._assess_candidate_transition(
            (
                EvidenceRecord(
                    "d0",
                    "delta_added",
                    "white",
                    "absent",
                    (600, 600, 1000, 1000),
                    (800, 800),
                ),
            ),
            (
                EvidenceJudgment(
                    "d0",
                    "building",
                    "background",
                    "clear",
                ),
            ),
            AgentAction("t1", "box", box=(10, 10, 20, 20)),
        )

        self.assertFalse(assessment.intended_error_improved)
        self.assertFalse(assessment.evidence_sufficient)
        self.assertEqual(assessment.comparison, "uncertain")
        self.assertIn(
            "attempted_action_mismatches_persisted_plan",
            assessment.evidence,
        )

    def test_candidate_without_persisted_diagnosis_fails_closed(self):
        verifier = StagedQwenVerifier(ScriptedBackend())
        assessment = verifier._assess_candidate_transition(
            (
                EvidenceRecord(
                    "d0",
                    "delta_added",
                    "white",
                    "absent",
                    (600, 600, 1000, 1000),
                    (800, 800),
                ),
            ),
            (
                EvidenceJudgment(
                    "d0",
                    "background",
                    "building",
                    "clear",
                ),
            ),
            AgentAction("t2", "positive_point", coordinate=(10, 10)),
        )

        self.assertFalse(assessment.intended_error_improved)
        self.assertFalse(assessment.evidence_sufficient)
        self.assertEqual(assessment.comparison, "uncertain")
        self.assertIn(
            "attempted_action_mismatches_persisted_plan",
            assessment.evidence,
        )

    def test_candidate_without_executable_persisted_plan_fails_closed(self):
        verifier = StagedQwenVerifier(ScriptedBackend())
        verifier._planned_diagnosis = Diagnosis(
            "r0", "false_negative", "t2", None
        )
        assessment = verifier._assess_candidate_transition(
            (
                EvidenceRecord(
                    "d0",
                    "delta_added",
                    "white",
                    "absent",
                    (600, 600, 1000, 1000),
                    (800, 800),
                ),
            ),
            (
                EvidenceJudgment(
                    "d0", "background", "building", "clear"
                ),
            ),
            AgentAction("t2", "positive_point", coordinate=(10, 10)),
        )

        self.assertFalse(assessment.evidence_sufficient)
        self.assertEqual(assessment.comparison, "uncertain")

    def test_candidate_geometry_must_match_persisted_plan(self):
        verifier = StagedQwenVerifier(ScriptedBackend())
        verifier._planned_diagnosis = Diagnosis(
            "r0", "false_negative", "t2", None
        )
        verifier._planned_action = AgentAction(
            "t2", "positive_point", coordinate=(10, 10)
        )
        assessment = verifier._assess_candidate_transition(
            (
                EvidenceRecord(
                    "d0",
                    "delta_added",
                    "white",
                    "absent",
                    (600, 600, 1000, 1000),
                    (800, 800),
                ),
            ),
            (
                EvidenceJudgment(
                    "d0", "background", "building", "clear"
                ),
            ),
            AgentAction("t2", "positive_point", coordinate=(11, 10)),
        )

        self.assertFalse(assessment.intended_error_improved)
        self.assertFalse(assessment.evidence_sufficient)
        self.assertEqual(assessment.comparison, "uncertain")
        self.assertIn(
            "attempted_action_mismatches_persisted_plan",
            assessment.evidence,
        )

    def test_removing_unchanged_target_presence_can_improve_change_mask(self):
        verifier = StagedQwenVerifier(ScriptedBackend())
        verifier._planned_diagnosis = Diagnosis(
            "r0",
            "false_positive_change",
            "t2",
            None,
        )
        verifier._accepted_records = (
            EvidenceRecord(
                "r0",
                "present",
                "white",
                "present",
                (600, 600, 1000, 1000),
                (800, 800),
            ),
        )
        verifier._planned_action = AgentAction(
            "t2", "negative_point", coordinate=(10, 10)
        )
        assessment = verifier._assess_candidate_transition(
            (
                EvidenceRecord(
                    "d0",
                    "delta_removed",
                    "black",
                    "absent",
                    (600, 600, 1000, 1000),
                    (800, 800),
                ),
            ),
            (
                EvidenceJudgment(
                    "d0",
                    "building",
                    "building",
                    "clear",
                ),
            ),
            AgentAction("t2", "negative_point", coordinate=(10, 10)),
        )

        self.assertTrue(assessment.intended_error_improved)
        self.assertFalse(assessment.introduced_harm)
        self.assertTrue(assessment.evidence_sufficient)
        self.assertEqual(assessment.comparison, "better")
        self.assertNotIn("target_transition_opposes_t1", assessment.evidence)

    def test_candidate_action_kind_must_match_persisted_diagnosis(self):
        verifier = StagedQwenVerifier(ScriptedBackend())
        verifier._planned_diagnosis = Diagnosis(
            "r0",
            "false_negative",
            "t2",
            None,
        )
        verifier._accepted_records = (
            EvidenceRecord(
                "r0",
                "missing",
                "black",
                "present",
                (600, 600, 1000, 1000),
                (800, 800),
            ),
        )
        verifier._planned_action = AgentAction(
            "t2", "positive_point", coordinate=(10, 10)
        )
        assessment = verifier._assess_candidate_transition(
            (
                EvidenceRecord(
                    "d0",
                    "delta_removed",
                    "black",
                    "absent",
                    (600, 600, 1000, 1000),
                    (800, 800),
                ),
            ),
            (
                EvidenceJudgment(
                    "d0",
                    "background",
                    "background",
                    "clear",
                ),
            ),
            AgentAction("t2", "negative_point", coordinate=(10, 10)),
        )

        self.assertFalse(assessment.intended_error_improved)
        self.assertFalse(assessment.evidence_sufficient)
        self.assertEqual(assessment.comparison, "uncertain")
        self.assertIn(
            "attempted_action_mismatches_persisted_plan",
            assessment.evidence,
        )

    def test_corrupt_persisted_plan_cannot_self_authorize(self):
        cases = (
            (
                Diagnosis("r0", "false_positive_change", "t2", None),
                AgentAction("t2", "box", box=(10, 10, 20, 20)),
                "delta_removed",
                ("building", "building"),
            ),
            (
                Diagnosis("r0", "false_negative", "t1", None),
                AgentAction("t2", "positive_point", coordinate=(10, 10)),
                "delta_added",
                ("background", "building"),
            ),
            (
                Diagnosis("r0", "false_positive_change", "t2", None),
                AgentAction(
                    "t2",
                    "negative_point",
                    coordinate=(10, 10),
                    box=(10, 10, 20, 20),
                ),
                "delta_removed",
                ("background", "background"),
            ),
        )
        for diagnosis, action, audit_kind, states in cases:
            with self.subTest(diagnosis=diagnosis, action=action):
                verifier = StagedQwenVerifier(ScriptedBackend())
                verifier._planned_diagnosis = diagnosis
                verifier._planned_action = action
                assessment = verifier._assess_candidate_transition(
                    (
                        EvidenceRecord(
                            "d0",
                            audit_kind,
                            "black" if audit_kind == "delta_removed" else "white",
                            "absent",
                            (600, 600, 1000, 1000),
                            (800, 800),
                        ),
                    ),
                    (EvidenceJudgment("d0", *states, "clear"),),
                    action,
                )

                self.assertFalse(assessment.intended_error_improved)
                self.assertFalse(assessment.evidence_sufficient)
                self.assertEqual(assessment.comparison, "uncertain")
                self.assertIn(
                    "attempted_action_mismatches_persisted_plan",
                    assessment.evidence,
                )

    def test_identical_finish_authorization_sets_stop(self):
        state = make_state()
        verifier = StagedQwenVerifier(ScriptedBackend())
        verifier._last_valid_output = VerifierOutput(
            quality_score=0.9,
            comparison="better",
            error_type="none",
            suggested_action="finish",
            accept=True,
            stop=False,
        )

        output = verifier.verify(state.clone(), 0.9, None, state)

        self.assertTrue(output.stop)

    def test_identical_nonfinish_candidate_never_inherits_acceptance(self):
        state = make_state()
        verifier = StagedQwenVerifier(ScriptedBackend())
        verifier._last_valid_output = VerifierOutput(
            quality_score=0.9,
            comparison="better",
            error_type="false_negative",
            target_view="t2",
            suggested_action="positive_point",
            accept=True,
            stop=False,
        )

        output = verifier.verify(
            state.clone(),
            0.9,
            AgentAction("t2", "positive_point", coordinate=(0, 0)),
            state,
        )

        self.assertEqual(output.comparison, "unchanged")
        self.assertFalse(output.accept)
        self.assertFalse(output.stop)
        self.assertFalse(output.localization_valid)
        self.assertIsNone(output.suggested_action)

    def test_stage_parser_selects_atomic_schema_match_instead_of_first_json(self):
        correct = {
            "region_id": "r0",
            "visual_judgment": {
                "t1_state": "background",
                "t2_state": "building",
                "evidence": "Exact white component covers a building appearance.",
            },
            "diagnosis": {
                "audit_checklist": audit_checklist_for("none"),
            },
        }
        raw = (
            json.dumps({"region": {"region_id": "r0"}, "schema": "input"})
            + "\n"
            + json.dumps(correct)
        )

        self.assertEqual(_extract_stage_json(raw, "audit"), correct)

    def test_stage_parser_rejects_copied_environment_context(self):
        raw = json.dumps({"region": {"region_id": "r0"}, "schema": "input"})
        with self.assertRaisesRegex(StageProtocolError, "candidate_keys"):
            _extract_stage_json(raw, "audit")

    def test_run_stage_call_history_requires_typed_parse_and_records_atomic_merge(
        self,
    ):
        class AuditedExtractingBackend:
            def __init__(self, stage, responses):
                self.responses = [
                    json.loads(json.dumps(response)) for response in responses
                ]
                self.call_history = [
                    {
                        "backend": "fixture",
                        "stage": stage,
                        "raw_response": '{"historical": true}',
                        "parsed_output": {"historical": True},
                    }
                ]

            def _complete(self, stage, attempt_kind):
                response = self.responses.pop(0)
                self.call_history.append(
                    {
                        "backend": "fixture",
                        "stage": stage,
                        "network_attempt": 1,
                        "network_error": "transient fixture retry",
                        "retry_scheduled": True,
                    }
                )
                self.call_history.append(
                    {
                        "backend": "fixture",
                        "stage": stage,
                        "attempt_kind": attempt_kind,
                        "raw_response": json.dumps(response, sort_keys=True),
                        "parsed_output": json.loads(json.dumps(response)),
                    }
                )
                return response

            def generate_stage(self, stage, state, payload, previous_state=None):
                return self._complete(stage, "initial")

            def repair_stage(
                self,
                stage,
                state,
                payload,
                validation_error,
                previous_state=None,
            ):
                return self._complete(stage, "repair")

        backend = AuditedExtractingBackend(
            "plan",
            ({"typed": "invalid"}, {"typed": "valid"}),
        )
        verifier = StagedQwenVerifier(backend, max_retries=2)

        def parse_typed(payload):
            if payload.get("typed") != "valid":
                raise StageProtocolError()
            return payload["typed"]

        self.assertEqual(
            verifier._run_stage("plan", make_state(), {}, parse_typed),
            "valid",
        )
        historical, initial_network, initial, repair_network, repair = (
            backend.call_history
        )
        self.assertEqual(historical["parsed_output"], {"historical": True})
        self.assertEqual(initial_network["network_error"], "transient fixture retry")
        self.assertNotIn("parse_error", initial_network)
        self.assertNotIn("parsed_output", initial)
        self.assertTrue(initial["parse_error"].strip())
        self.assertEqual(initial["raw_response"], '{"typed": "invalid"}')
        self.assertEqual(repair_network["network_error"], "transient fixture retry")
        self.assertNotIn("parse_error", repair)
        self.assertEqual(repair["parsed_output"], {"typed": "valid"})

        invalid_anchor = {
            "region_id": "r0",
            "visual_judgment": {
                "t1_state": "background",
                "t2_state": "building",
                "evidence": "short_observable_rgb_evidence",
            },
            "diagnosis": {"audit_checklist": {"evidence_sufficient": "pass"}},
        }
        repaired_evidence = {
            "region_id": "r0",
            "visual_judgment": {
                "evidence": "T2 RGB shows a roofed building where T1 shows bare ground."
            },
        }
        atomic_backend = AuditedExtractingBackend(
            "audit",
            (invalid_anchor, repaired_evidence),
        )
        atomic_verifier = StagedQwenVerifier(atomic_backend, max_retries=2)

        def parse_atomic(payload):
            evidence = payload["visual_judgment"]["evidence"]
            if evidence == "short_observable_rgb_evidence":
                raise StageProtocolError(
                    "atomic audit evidence copied schema boilerplate"
                )
            if "diagnosis" not in payload:
                raise StageProtocolError("atomic merged payload lacks diagnosis")
            return payload

        merged = atomic_verifier._run_stage(
            "audit",
            make_state(),
            {},
            parse_atomic,
        )
        atomic_initial = atomic_backend.call_history[2]
        atomic_repair = atomic_backend.call_history[4]
        self.assertNotIn("parsed_output", atomic_initial)
        self.assertTrue(atomic_initial["parse_error"].strip())
        self.assertEqual(
            atomic_repair["raw_response"],
            json.dumps(repaired_evidence, sort_keys=True),
        )
        self.assertEqual(atomic_repair["parsed_output"], merged)
        self.assertEqual(
            atomic_repair["parsed_output"]["visual_judgment"]["t1_state"],
            "background",
        )
        self.assertIn("diagnosis", atomic_repair["parsed_output"])

    def test_prompt_puts_output_contract_before_wrapped_environment(self):
        prompt = _stage_prompt(
            "audit",
            {
                "target_class": "building",
                "region": {"region_id": "r7"},
                "schema": "atomic_grounded_region_audit_v11",
            },
        )

        self.assertLess(prompt.index("OUTPUT CONTRACT"), prompt.index("<ENVIRONMENT_FACTS>"))
        self.assertIn('"environment_facts"', prompt)
        self.assertIn('"region_id":"r7"', prompt)
        self.assertIn("Do not copy the Environment envelope", prompt)

    def test_atomic_prompt_has_no_default_none_bias_and_keeps_target_semantics(self):
        prompt = _stage_prompt(
            "audit",
            {
                "target_class": "building",
                "region": {"region_id": "r7"},
                "schema": "atomic_grounded_region_audit_v11",
            },
        )

        self.assertNotIn("normally correct and error_type is none", prompt)
        self.assertIn("one compact atomic semantic audit", prompt)
        self.assertIn("trailers, RVs, mobile equipment", prompt)
        self.assertIn("concrete RGB observation specific to this region", prompt)
        self.assertIn('"audit_checklist"', prompt)
        self.assertNotIn('"mask_assessment"', prompt)
        self.assertNotIn('"evidence_quality"', prompt)
        self.assertNotIn('"target_view"', prompt)
        self.assertIn("Runtime derives evidence quality, mask assessment", prompt)
        self.assertNotIn("screening_resolution", prompt)
        self.assertNotIn("screening_hypothesis", prompt)

    def test_atomic_audit_receives_state_target_class(self):
        backend = ScriptedBackend(error="none", target=None)
        output = StagedQwenVerifier(backend).verify(make_state(), None, None)

        self.assertTrue(output.verifier_valid)
        audit_payload = next(
            payload
            for stage, payload in zip(backend.calls, backend.payloads)
            if stage == "audit"
        )
        self.assertEqual(audit_payload["target_class"], "building")
        self.assertEqual(audit_payload["region"]["change_mask_state"], "white")

    def test_environment_mask_state_is_injected_into_trace(self):
        verifier = StagedQwenVerifier(ScriptedBackend(error="none", target=None))
        output = verifier.verify(make_state(), None, None)
        self.assertTrue(output.verifier_valid)
        self.assertEqual(
            verifier.last_evidence["stage_trace"]["judgments"][0][
                "change_mask_state"
            ],
            "white",
        )

    def test_model_cannot_override_environment_mask_state(self):
        class MaskOverrideBackend(ScriptedBackend):
            def generate_stage(self, stage, state, payload, previous_state=None):
                response = super().generate_stage(
                    stage, state, payload, previous_state
                )
                if stage == "audit":
                    response["visual_judgment"]["change_mask_state"] = "black"
                return response

        verifier = StagedQwenVerifier(MaskOverrideBackend(), max_retries=1)

        output = verifier.verify(make_state(), None, None)

        self.assertTrue(output.verifier_valid)
        self.assertFalse(output.localization_valid)
        self.assertIn(
            "visual_judgment must contain exactly",
            verifier.last_evidence["regional_validation_errors"][0]["error"],
        )

    def test_atomic_audit_rejects_copied_evidence_boilerplate(self):
        invalid_values = (
            "strongest local support or contradiction after checking the whole component",
            "strongest local support or contradiction after checking the whole component.",
            "SHORT_OBSERVABLE_RGB_EVIDENCE",
            "<SHORT_OBSERVABLE_RGB_EVIDENCE>",
            "Evidence: <SHORT_OBSERVABLE_RGB_EVIDENCE>.",
            "REPLACE_WITH_ONE_SHORT_PAIRED_RGB_FACT",
            "Evidence: REPLACE_WITH_ONE_SHORT_PAIRED_RGB_FACT.",
        )
        for invalid_value in invalid_values:
            with self.subTest(invalid_value=invalid_value):
                class BoilerplateEvidenceBackend(ScriptedBackend):
                    def generate_stage(
                        self, stage, state, payload, previous_state=None
                    ):
                        response = super().generate_stage(
                            stage, state, payload, previous_state
                        )
                        if stage == "audit":
                            response["visual_judgment"]["evidence"] = invalid_value
                        return response

                verifier = StagedQwenVerifier(
                    BoilerplateEvidenceBackend(),
                    max_retries=1,
                )
                output = verifier.verify(make_state(), None, None)

                self.assertTrue(output.verifier_valid)
                self.assertFalse(output.localization_valid)
                self.assertIsNone(output.suggested_action)
                self.assertIn(
                    "evidence copied schema boilerplate",
                    verifier.last_evidence["regional_validation_errors"][0][
                        "error"
                    ],
                )

    def test_atomic_audit_evidence_must_be_a_string(self):
        for invalid_value in (None, 7, True, {"fact": "rgb"}, ["rgb"]):
            with self.subTest(invalid_value=invalid_value):
                class NonStringEvidenceBackend(ScriptedBackend):
                    def generate_stage(
                        self, stage, state, payload, previous_state=None
                    ):
                        response = super().generate_stage(
                            stage, state, payload, previous_state
                        )
                        if stage == "audit":
                            response["visual_judgment"]["evidence"] = invalid_value
                        return response

                verifier = StagedQwenVerifier(
                    NonStringEvidenceBackend(),
                    max_retries=1,
                )
                output = verifier.verify(make_state(), None, None)

                self.assertFalse(output.localization_valid)
                self.assertIn(
                    "evidence must be a string",
                    verifier.last_evidence["regional_validation_errors"][0][
                        "error"
                    ],
                )

    def test_atomic_audit_rejects_image_or_schema_placeholder_evidence(self):
        invalid_values = (
            "<RGB_IMAGE_1>",
            "<RGB_IMAGE>",
            "<RGB_IMAGE_OBSERVATION>",
            "Evidence: <IMAGE_1>.",
            "<LOCAL_CROP>",
            "<RGB IMAGE 1>",
            "<|image_pad|>",
        )
        for invalid_value in invalid_values:
            with self.subTest(invalid_value=invalid_value):
                class PlaceholderEvidenceBackend(ScriptedBackend):
                    def generate_stage(
                        self, stage, state, payload, previous_state=None
                    ):
                        response = super().generate_stage(
                            stage, state, payload, previous_state
                        )
                        if stage == "audit":
                            response["visual_judgment"]["evidence"] = invalid_value
                        return response

                verifier = StagedQwenVerifier(
                    PlaceholderEvidenceBackend(), max_retries=1
                )
                output = verifier.verify(make_state(), None, None)

                self.assertTrue(output.verifier_valid)
                self.assertFalse(output.localization_valid)
                self.assertIsNone(output.suggested_action)
                self.assertIn(
                    "image/schema placeholder",
                    verifier.last_evidence["regional_validation_errors"][0][
                        "error"
                    ],
                )

    def test_atomic_audit_repairs_image_placeholder_with_rgb_fact(self):
        class RepairingImageMarkerBackend(ScriptedBackend):
            def __init__(self):
                super().__init__()
                self.repair_errors = []

            def generate_stage(self, stage, state, payload, previous_state=None):
                response = super().generate_stage(
                    stage, state, payload, previous_state
                )
                if stage == "audit":
                    response["visual_judgment"]["evidence"] = "<RGB_IMAGE_1>"
                return response

            def repair_stage(
                self, stage, state, payload, validation_error, previous_state=None
            ):
                self.repair_errors.append(validation_error)
                return ScriptedBackend.generate_stage(
                    self, stage, state, payload, previous_state
                )

        backend = RepairingImageMarkerBackend()
        verifier = StagedQwenVerifier(backend, max_retries=2)
        output = verifier.verify(make_state(), None, None)

        self.assertTrue(output.verifier_valid)
        self.assertTrue(output.localization_valid)
        self.assertTrue(backend.repair_errors)
        self.assertIn("image/schema placeholder", backend.repair_errors[0])
        self.assertEqual(verifier.last_evidence["regional_validation_errors"], [])

    def test_atomic_audit_rejects_underscore_delimited_label_evidence(self):
        invalid_values = (
            "all_black",
            "all_black.",
            "no_visible_building_in_both_images",
            "no_visible_building_in_T1_or_T2",
        )
        for invalid_value in invalid_values:
            with self.subTest(invalid_value=invalid_value):
                class MachineLabelEvidenceBackend(ScriptedBackend):
                    def generate_stage(
                        self, stage, state, payload, previous_state=None
                    ):
                        response = super().generate_stage(
                            stage, state, payload, previous_state
                        )
                        if stage == "audit":
                            response["visual_judgment"]["evidence"] = invalid_value
                        return response

                verifier = StagedQwenVerifier(
                    MachineLabelEvidenceBackend(), max_retries=1
                )
                output = verifier.verify(make_state(), None, None)

                self.assertTrue(output.verifier_valid)
                self.assertFalse(output.localization_valid)
                self.assertIsNone(output.suggested_action)
                self.assertIn(
                    "underscore-delimited label",
                    verifier.last_evidence["regional_validation_errors"][0][
                        "error"
                    ],
                )

    def test_atomic_audit_repairs_machine_label_with_rgb_fact(self):
        class RepairingMachineLabelBackend(ScriptedBackend):
            def __init__(self):
                super().__init__()
                self.repair_errors = []

            def generate_stage(self, stage, state, payload, previous_state=None):
                response = super().generate_stage(
                    stage, state, payload, previous_state
                )
                if stage == "audit":
                    response["visual_judgment"]["evidence"] = (
                        "no_visible_building_in_both_images"
                    )
                return response

            def repair_stage(
                self, stage, state, payload, validation_error, previous_state=None
            ):
                self.repair_errors.append(validation_error)
                return ScriptedBackend.generate_stage(
                    self, stage, state, payload, previous_state
                )

        backend = RepairingMachineLabelBackend()
        verifier = StagedQwenVerifier(backend, max_retries=2)
        output = verifier.verify(make_state(), None, None)

        self.assertTrue(output.verifier_valid)
        self.assertTrue(output.localization_valid)
        self.assertEqual(len(backend.repair_errors), 1)
        self.assertIn("underscore-delimited label", backend.repair_errors[0])
        self.assertEqual(verifier.last_evidence["regional_validation_errors"], [])

    def test_atomic_evidence_repair_preserves_validated_semantic_anchor(self):
        class SemanticFlipRepairBackend(ScriptedBackend):
            def __init__(self):
                super().__init__(error="false_positive_change")
                self.repair_payloads = []
                self.initial_response = None

            def generate_stage(self, stage, state, payload, previous_state=None):
                response = super().generate_stage(
                    stage, state, payload, previous_state
                )
                if stage == "audit":
                    response["visual_judgment"]["evidence"] = (
                        "T2 shows a roofed building with a red outline while "
                        "T1 shows bare ground."
                    )
                    self.initial_response = response
                return response

            def repair_stage(
                self,
                stage,
                state,
                payload,
                validation_error,
                previous_state=None,
            ):
                self.repair_payloads.append(dict(payload))
                self.calls.append(f"repair:{stage}")
                response = self.initial_response
                response["visual_judgment"]["evidence"] = (
                    "T2 RGB shows a roofed building where T1 RGB shows bare ground."
                )
                response["diagnosis"]["audit_checklist"] = audit_checklist_for(
                    "none"
                )
                return response

        backend = SemanticFlipRepairBackend()
        verifier = StagedQwenVerifier(backend, max_retries=2)

        output = verifier.verify(make_state(), None, None)

        self.assertTrue(output.verifier_valid)
        self.assertTrue(output.localization_valid)
        self.assertEqual(output.error_type, "false_positive_change")
        self.assertEqual(output.suggested_action, "negative_point")
        self.assertEqual(len(backend.repair_payloads), 1)
        repair_context = backend.repair_payloads[0]["repair_context"]
        self.assertEqual(repair_context["mode"], "atomic_evidence_only")
        self.assertIn("diagnosis.audit_checklist", repair_context["preserve_fields"])
        self.assertEqual(verifier.last_evidence["regional_validation_errors"], [])

    def test_atomic_evidence_repair_never_falls_back_to_semantic_rewrite(self):
        class MultiRepairBackend(ScriptedBackend):
            def __init__(self):
                super().__init__(error="false_positive_change")
                self.repair_payloads = []
                self.repair_errors = []

            def generate_stage(self, stage, state, payload, previous_state=None):
                response = super().generate_stage(
                    stage, state, payload, previous_state
                )
                if stage == "audit":
                    response["visual_judgment"]["evidence"] = (
                        "T2 shows a roofed building with a red outline."
                    )
                return response

            def repair_stage(
                self,
                stage,
                state,
                payload,
                validation_error,
                previous_state=None,
            ):
                self.repair_payloads.append(dict(payload))
                self.repair_errors.append(validation_error)
                if len(self.repair_payloads) == 1:
                    return {
                        "region_id": "wrong-region",
                        "visual_judgment": {
                            "evidence": "T2 RGB shows a roofed structure."
                        },
                        "diagnosis": {},
                    }
                response = ScriptedBackend.generate_stage(
                    self, stage, state, payload, previous_state
                )
                response["visual_judgment"]["evidence"] = (
                    "T2 RGB shows a roofed building where T1 RGB shows bare ground."
                )
                response["diagnosis"]["audit_checklist"] = audit_checklist_for(
                    "none"
                )
                return response

        backend = MultiRepairBackend()
        verifier = StagedQwenVerifier(backend, max_retries=3)

        output = verifier.verify(make_state(), None, None)

        self.assertTrue(output.verifier_valid)
        self.assertEqual(output.error_type, "false_positive_change")
        self.assertEqual(output.suggested_action, "negative_point")
        self.assertEqual(len(backend.repair_payloads), 2)
        self.assertIn("changed the region_id", backend.repair_errors[1])
        self.assertTrue(
            all(
                item["repair_context"]["mode"] == "atomic_evidence_only"
                for item in backend.repair_payloads
            )
        )
        self.assertEqual(
            backend.repair_payloads[0]["repair_context"]["origin_validation_error"],
            backend.repair_payloads[1]["repair_context"]["origin_validation_error"],
        )

    def test_atomic_evidence_repair_requires_a_fully_valid_anchor(self):
        class InvalidAnchorBackend(ScriptedBackend):
            def __init__(self):
                super().__init__()
                self.repair_payloads = []

            def generate_stage(self, stage, state, payload, previous_state=None):
                response = super().generate_stage(
                    stage, state, payload, previous_state
                )
                if stage == "audit":
                    response["visual_judgment"]["evidence"] = (
                        "T2 shows a roofed building with a red outline."
                    )
                    response["diagnosis"]["audit_checklist"] = (
                        audit_checklist_for("false_negative")
                    )
                return response

            def repair_stage(
                self,
                stage,
                state,
                payload,
                validation_error,
                previous_state=None,
            ):
                self.repair_payloads.append(dict(payload))
                return ScriptedBackend.generate_stage(
                    self, stage, state, payload, previous_state
                )

        backend = InvalidAnchorBackend()
        verifier = StagedQwenVerifier(backend, max_retries=2)

        output = verifier.verify(make_state(), None, None)

        self.assertTrue(output.verifier_valid)
        self.assertTrue(output.localization_valid)
        self.assertEqual(output.suggested_action, "finish")
        self.assertEqual(len(backend.repair_payloads), 1)
        self.assertNotIn("repair_context", backend.repair_payloads[0])
        self.assertEqual(verifier.last_evidence["regional_validation_errors"], [])

    def test_atomic_audit_rejects_json_numeric_or_coordinate_placeholder_evidence(self):
        invalid_values = (
            "0",
            "[]",
            "[null]",
            '["roof"]',
            "[0, 0, 0]",
            "[0,212,196,435]",
            "306, 843, 545, 1000",
            "(47, 325)",
            "[1e3, -2.5E-2, .75, 4., 5, 6, 7, 8, 9]",
        )
        for invalid_value in invalid_values:
            with self.subTest(invalid_value=invalid_value):
                class PlaceholderEvidenceBackend(ScriptedBackend):
                    def generate_stage(
                        self, stage, state, payload, previous_state=None
                    ):
                        response = super().generate_stage(
                            stage, state, payload, previous_state
                        )
                        if stage == "audit":
                            response["visual_judgment"]["evidence"] = invalid_value
                        return response

                verifier = StagedQwenVerifier(
                    PlaceholderEvidenceBackend(), max_retries=1
                )
                output = verifier.verify(make_state(), None, None)

                self.assertTrue(output.verifier_valid)
                self.assertFalse(output.localization_valid)
                self.assertIsNone(output.suggested_action)
                self.assertIn(
                    "JSON/numeric/coordinate placeholder",
                    verifier.last_evidence["regional_validation_errors"][0][
                        "error"
                    ],
                )

    def test_atomic_audit_rejects_mask_color_shortcut_evidence(self):
        for invalid_value in (
            "The change mask is black, so there is no building change.",
            "All pixels are black in the change-mask presentation.",
            "All pixels are black in both T1 and T2.",
            "All pixels in both T1/T2 are black.",
            "Both predicted masks are empty, therefore there is no RGB change.",
            "The change map is all black, so no building changed.",
            "T2 shows a white pixel mask that covers a connected component.",
            "White pixels in the exact component mask indicate a building.",
            "A white area appears in the T2 mask but was not present in T1.",
            "T2 shows a white roof and a red outline while T1 has no building.",
            "The yellow contour marks the visible building in T2.",
            "The marker around the T2 roof is cyan.",
            "A magenta border highlights the candidate building.",
            "The T2 component is outlined in red.",
            "all black",
            "all-black.",
        ):
            with self.subTest(invalid_value=invalid_value):
                class MaskShortcutBackend(ScriptedBackend):
                    def generate_stage(
                        self, stage, state, payload, previous_state=None
                    ):
                        response = super().generate_stage(
                            stage, state, payload, previous_state
                        )
                        if stage == "audit":
                            response["visual_judgment"]["evidence"] = invalid_value
                        return response

                verifier = StagedQwenVerifier(
                    MaskShortcutBackend(), max_retries=1
                )
                output = verifier.verify(make_state(), None, None)

                self.assertTrue(output.verifier_valid)
                self.assertFalse(output.localization_valid)
                self.assertIn(
                    "mask/presentation color",
                    verifier.last_evidence["regional_validation_errors"][0][
                        "error"
                    ],
                )

    def test_atomic_audit_allows_rgb_detail_with_non_shortcut_mask_context(self):
        class ObservableRoofEvidenceBackend(ScriptedBackend):
            def __init__(self, evidence):
                super().__init__()
                self.evidence = evidence

            def generate_stage(self, stage, state, payload, previous_state=None):
                response = super().generate_stage(
                    stage, state, payload, previous_state
                )
                if stage == "audit":
                    response["visual_judgment"]["evidence"] = self.evidence
                return response

        valid_values = (
            "White roof pixels are visible in the unmodified RGB.",
            "White roof pixels form a rectilinear roofed structure in "
            "unmodified T2 RGB.",
            "A red metal roof is visible in T2 but absent in T1.",
            "A blue roof stands beside a concrete border wall in T2.",
            "Dark asphalt road surrounds the light roof in both RGB views.",
            "A white border wall surrounds the T2 building.",
            "A black outline traces the physical roof edge in T2 RGB.",
            "A greenhouse is visible and the audited mask covers it.",
            "White roof pixels remain visible inside the audited mask in T2 RGB.",
        )
        for valid_value in valid_values:
            with self.subTest(valid_value=valid_value):
                verifier = StagedQwenVerifier(
                    ObservableRoofEvidenceBackend(valid_value), max_retries=1
                )
                output = verifier.verify(make_state(), None, None)

                self.assertTrue(output.verifier_valid)
                self.assertTrue(output.localization_valid)
                self.assertEqual(
                    verifier.last_evidence["regional_validation_errors"], []
                )

    def test_atomic_audit_repairs_copied_evidence_with_rgb_fact(self):
        class RepairingEvidenceBackend(ScriptedBackend):
            def __init__(self):
                super().__init__()
                self.repair_errors = []

            def generate_stage(self, stage, state, payload, previous_state=None):
                response = super().generate_stage(
                    stage, state, payload, previous_state
                )
                if stage == "audit":
                    response["visual_judgment"]["evidence"] = (
                        "strongest local support or contradiction after checking "
                        "the whole component"
                    )
                return response

            def repair_stage(
                self, stage, state, payload, validation_error, previous_state=None
            ):
                self.repair_errors.append(validation_error)
                return ScriptedBackend.generate_stage(
                    self, stage, state, payload, previous_state
                )

        backend = RepairingEvidenceBackend()
        verifier = StagedQwenVerifier(backend, max_retries=2)
        output = verifier.verify(make_state(), None, None)

        self.assertTrue(output.localization_valid)
        self.assertEqual(len(backend.repair_errors), 1)
        self.assertIn("schema boilerplate", backend.repair_errors[0])
        self.assertEqual(verifier.last_evidence["regional_validation_errors"], [])

    def test_atomic_audit_repair_cannot_copy_placeholder(self):
        class InvalidRepairEvidenceBackend(ScriptedBackend):
            def __init__(self):
                super().__init__()
                self.repair_called = False

            def generate_stage(self, stage, state, payload, previous_state=None):
                response = super().generate_stage(
                    stage, state, payload, previous_state
                )
                if stage == "audit":
                    response["visual_judgment"]["evidence"] = (
                        "strongest local support or contradiction after checking "
                        "the whole component"
                    )
                return response

            def repair_stage(
                self, stage, state, payload, validation_error, previous_state=None
            ):
                self.repair_called = True
                response = ScriptedBackend.generate_stage(
                    self, stage, state, payload, previous_state
                )
                response["visual_judgment"]["evidence"] = (
                    "<SHORT_OBSERVABLE_RGB_EVIDENCE>"
                )
                return response

        backend = InvalidRepairEvidenceBackend()
        verifier = StagedQwenVerifier(backend, max_retries=2)
        output = verifier.verify(make_state(), None, None)

        self.assertTrue(backend.repair_called)
        self.assertFalse(output.localization_valid)
        self.assertIsNone(output.suggested_action)
        self.assertIn(
            "evidence copied schema boilerplate",
            verifier.last_evidence["regional_validation_errors"][0]["error"],
        )

    def test_compact_audit_rejects_nested_per_item_evidence(self):
        class LegacyGroundedBackend(ScriptedBackend):
            def generate_stage(self, stage, state, payload, previous_state=None):
                response = super().generate_stage(
                    stage, state, payload, previous_state
                )
                if stage == "audit":
                    response["diagnosis"]["audit_checklist"] = (
                        grounded_audit_checklist_for("false_positive_change")
                    )
                return response

        verifier = StagedQwenVerifier(LegacyGroundedBackend(), max_retries=1)
        output = verifier.verify(make_state(), None, None)

        self.assertTrue(output.verifier_valid)
        self.assertFalse(output.localization_valid)
        self.assertIsNone(output.suggested_action)
        self.assertIn(
            "values must be status strings",
            verifier.last_evidence["regional_validation_errors"][0]["error"],
        )

    def test_compact_checklist_derives_evidence_quality(self):
        backend = ScriptedBackend(error="uncertain_region", target=None)
        verifier = StagedQwenVerifier(backend)

        output = verifier.verify(make_state(), None, None)

        self.assertTrue(output.verifier_valid)
        self.assertEqual(
            verifier.last_evidence["stage_trace"]["judgments"][0][
                "evidence_quality"
            ],
            "ambiguous",
        )

    def test_white_no_transition_all_pass_fails_closed(self):
        verifier = StagedQwenVerifier(
            ScriptedBackend(t1="background", t2="background", error="none")
        )

        output = verifier.verify(make_state(), None, None)

        self.assertTrue(output.verifier_valid)
        self.assertFalse(output.localization_valid)
        self.assertFalse(output.stop)
        self.assertIsNone(output.suggested_action)
        self.assertIn(
            "white region with no target transition",
            verifier.last_evidence["regional_validation_errors"][0]["error"],
        )

    def test_black_transition_all_pass_fails_closed(self):
        image1 = np.zeros((16, 16, 3), dtype=np.uint8)
        image2 = image1.copy()
        empty = np.zeros((16, 16), dtype=bool)
        state = ChangeState(image1, image2, "building", empty, empty, empty)
        attach_verifier_regions(
            state,
            max_regions=1,
            min_component_area=1,
            empty_mask_scan_grid_size=1,
        )
        verifier = StagedQwenVerifier(
            ScriptedBackend(t1="background", t2="building", error="none")
        )

        output = verifier.verify(state, None, None)

        self.assertTrue(output.verifier_valid)
        self.assertFalse(output.localization_valid)
        self.assertFalse(output.stop)
        self.assertIsNone(output.suggested_action)
        self.assertIn(
            "black region with a target transition",
            verifier.last_evidence["regional_validation_errors"][0]["error"],
        )

    def test_all_substantive_checks_not_applicable_stays_uncertain(self):
        class AllNotApplicableBackend(ScriptedBackend):
            def generate_stage(self, stage, state, payload, previous_state=None):
                response = super().generate_stage(
                    stage, state, payload, previous_state
                )
                if stage == "audit":
                    checklist = response["diagnosis"]["audit_checklist"]
                    for name in checklist:
                        if name != "evidence_sufficient":
                            checklist[name] = "not_applicable"
                return response

        verifier = StagedQwenVerifier(AllNotApplicableBackend(error="none"))

        output = verifier.verify(make_state(), None, None)

        self.assertTrue(output.verifier_valid)
        self.assertFalse(output.localization_valid)
        self.assertFalse(output.stop)
        self.assertIsNone(output.suggested_action)
        self.assertEqual(output.quality_score, 0.0)
        self.assertEqual(
            verifier.last_evidence["stage_trace"]["diagnoses"][0]["error_type"],
            "uncertain_region",
        )

    def test_noncanonical_compact_enum_fails_closed(self):
        class UppercaseStatusBackend(ScriptedBackend):
            def generate_stage(self, stage, state, payload, previous_state=None):
                response = super().generate_stage(
                    stage, state, payload, previous_state
                )
                if stage == "audit":
                    response["diagnosis"]["audit_checklist"][
                        "evidence_sufficient"
                    ] = "Pass"
                return response

        verifier = StagedQwenVerifier(UppercaseStatusBackend(error="none"))

        output = verifier.verify(make_state(), None, None)

        self.assertTrue(output.verifier_valid)
        self.assertFalse(output.localization_valid)
        self.assertIsNone(output.suggested_action)
        self.assertIn(
            "unsupported status",
            verifier.last_evidence["regional_validation_errors"][0]["error"],
        )

    def test_mask_color_rgb_states_still_fail_closed(self):
        verifier = StagedQwenVerifier(
            ScriptedBackend(t1="black", t2="white", error="none"),
            max_retries=1,
        )

        output = verifier.verify(make_state(), None, None)

        self.assertTrue(output.verifier_valid)
        self.assertFalse(output.localization_valid)
        self.assertIsNone(output.suggested_action)
        self.assertIn(
            "invalid RGB state",
            verifier.last_evidence["regional_validation_errors"][0]["error"],
        )

    def test_invalid_stage_output_is_repaired_before_verifier_aborts(self):
        backend = RepairingBackend()
        verifier = StagedQwenVerifier(backend, max_retries=2)

        output = verifier.verify(make_state(), None, None)

        self.assertTrue(output.verifier_valid)
        self.assertTrue(output.stop)
        self.assertIn("repair:audit", backend.calls)
        self.assertIn("must contain exactly", backend.repair_errors[0])

    def test_atomic_diagnosis_persists_compact_checklist_and_one_evidence(self):
        backend = ScriptedBackend(error="none", target=None)
        verifier = StagedQwenVerifier(backend)

        output = verifier.verify(make_state(), None, None)

        self.assertTrue(output.verifier_valid)
        diagnosis = verifier.last_evidence["stage_trace"]["diagnoses"][0]
        self.assertNotIn("confidence", diagnosis)
        self.assertEqual(diagnosis["error_type"], "none")
        self.assertEqual(diagnosis["audit_checklist"]["evidence_sufficient"], "pass")
        self.assertFalse(diagnosis["audit_evidence"]["evidence_sufficient"])
        self.assertIn("Paired RGB", diagnosis["summary"])

    def test_correct_appearance_change_can_finish_without_action_plan(self):
        backend = ScriptedBackend(error="none", target=None)
        verifier = StagedQwenVerifier(backend)

        output = verifier.verify(make_state(), None, None)

        self.assertTrue(output.verifier_valid)
        self.assertTrue(output.accept)
        self.assertTrue(output.stop)
        self.assertEqual(output.suggested_action, "finish")
        self.assertEqual(
            backend.calls, ["audit"]
        )
        self.assertEqual(
            verifier.last_evidence["stage_trace"]["evidence"][0]["change_mask_state"],
            "white",
        )

    def test_initial_finish_is_derived_from_state_not_model_quality(self):
        verifier = StagedQwenVerifier(
            LowQualityInitialBackend(error="none", target=None)
        )

        output = verifier.verify(make_state(), None, None)

        self.assertEqual(output.quality_score, 1.0)
        self.assertTrue(output.accept)
        self.assertTrue(output.stop)

    def test_initial_audits_every_environment_region_across_batches(self):
        backend = ScriptedBackend(error="none", target=None)
        verifier = StagedQwenVerifier(backend)

        output = verifier.verify(make_two_region_state(), None, None)

        self.assertTrue(output.verifier_valid)
        self.assertTrue(output.accept)
        self.assertTrue(output.stop)
        self.assertEqual(output.suggested_action, "finish")
        self.assertEqual(
            backend.calls,
            ["audit", "audit"],
        )
        audit_ids = [
            payload["region"]["region_id"]
            for stage, payload in zip(backend.calls, backend.payloads)
            if stage == "audit"
        ]
        self.assertEqual(audit_ids, ["r0", "r1"])
        self.assertTrue(
            all(
                "screening_hypothesis" not in payload
                for stage, payload in zip(backend.calls, backend.payloads)
                if stage == "audit"
            )
        )
        trace = verifier.last_evidence["stage_trace"]
        self.assertTrue(trace["state_completion_gate_passed"])
        self.assertEqual(set(trace["audited_region_ids"]), {"r0", "r1"})

    def test_invalid_region_fails_closed_without_suppressing_later_action(self):
        class OneAmbiguousRegionBackend(ScriptedBackend):
            def generate_stage(self, stage, state, payload, previous_state=None):
                response = super().generate_stage(
                    stage, state, payload, previous_state
                )
                if stage == "audit" and payload["region"]["region_id"] == "r0":
                    response["visual_judgment"]["mask_assessment"] = "mixed"
                return response

        backend = OneAmbiguousRegionBackend(
            error="false_positive_change",
            target="t2",
        )
        verifier = StagedQwenVerifier(
            backend,
            max_retries=2,
        )

        output = verifier.verify(make_two_region_state(), None, None)

        self.assertTrue(output.verifier_valid)
        self.assertEqual(output.suggested_action, "negative_point")
        self.assertEqual(output.error_type, "false_positive_change")
        regional_errors = verifier.last_evidence["regional_validation_errors"]
        self.assertEqual(regional_errors[0]["region_id"], "r0")
        diagnoses = verifier.last_evidence["stage_trace"]["diagnoses"]
        self.assertEqual(diagnoses[0]["error_type"], "uncertain_region")
        self.assertEqual(diagnoses[1]["error_type"], "false_positive_change")

    def test_clear_appearance_change_can_still_be_false_positive(self):
        backend = ScriptedBackend(
            error="false_positive_change", target="t2", action="negative_point"
        )
        verifier = StagedQwenVerifier(backend)

        output = verifier.verify(make_state(), None, None)

        self.assertTrue(output.verifier_valid)
        self.assertEqual(output.error_type, "false_positive_change")
        self.assertEqual(output.suggested_action, "negative_point")
        self.assertNotIn("plan", backend.calls)

    def test_false_positive_target_is_canonicalized_to_unique_white_view(self):
        backend = ScriptedBackend(
            t1="background",
            t2="background",
            error="false_positive_change",
            target="t1",
            action="negative_point",
        )
        verifier = StagedQwenVerifier(backend)

        output = verifier.verify(make_state(), None, None)

        self.assertTrue(output.verifier_valid)
        self.assertTrue(output.localization_valid)
        self.assertEqual(output.target_view, "t2")
        self.assertEqual(output.suggested_action, "negative_point")
        self.assertNotIn("plan", backend.calls)

    def test_false_positive_without_any_white_target_seed_fails_closed(self):
        source = make_state()
        empty = np.zeros_like(source.t2_mask)
        state = ChangeState(
            source.t1_image,
            source.t2_image,
            source.query,
            empty,
            empty,
            source.change_mask.copy(),
        )
        attach_verifier_regions(state, max_regions=6, min_component_area=1)
        verifier = StagedQwenVerifier(
            ScriptedBackend(
                t1="background",
                t2="background",
                error="false_positive_change",
                target="t1",
            )
        )

        output = verifier.verify(state, None, None)

        self.assertTrue(output.verifier_valid)
        self.assertFalse(output.localization_valid)
        self.assertIsNone(output.suggested_action)

    def test_false_negative_semantic_editability_conflict_fails_closed(self):
        verifier = StagedQwenVerifier(
            ScriptedBackend(
                t1="building",
                t2="background",
                error="false_negative",
            )
        )

        output = verifier.verify(make_editability_conflict_state(), None, None)

        self.assertTrue(output.verifier_valid)
        self.assertFalse(output.localization_valid)
        self.assertIsNone(output.suggested_action)
        diagnosis = verifier.last_evidence["stage_trace"]["diagnoses"][0]
        self.assertEqual(diagnosis["error_type"], "false_negative")
        self.assertIsNone(diagnosis["target_view"])

    def test_false_negative_semantic_editability_intersection_is_executable(self):
        source = make_editability_conflict_state()
        t1 = np.zeros_like(source.t1_mask)
        t2 = source.t1_mask.copy()
        state = ChangeState(
            source.t1_image,
            source.t2_image,
            source.query,
            t1,
            t2,
            np.zeros_like(t1),
        )
        attach_verifier_regions(state, max_regions=6, min_component_area=1)
        verifier = StagedQwenVerifier(
            ScriptedBackend(
                t1="building",
                t2="background",
                error="false_negative",
            )
        )

        output = verifier.verify(state, None, None)

        self.assertTrue(output.verifier_valid)
        self.assertTrue(output.localization_valid)
        self.assertEqual(output.target_view, "t1")
        self.assertEqual(output.suggested_action, "positive_point")
        self.assertEqual(
            verifier._planned_action,
            AgentAction(
                "t1",
                "positive_point",
                coordinate=normalized_point_to_pixel(
                    output.error_region[:2], state.image_size
                ),
            ),
        )

        action = verifier._planned_action
        assert action is not None and action.coordinate is not None
        candidate_t1 = state.t1_mask.copy()
        x, y = action.coordinate
        candidate_t1[y, x] = True
        candidate = ChangeState(
            state.t1_image,
            state.t2_image,
            state.query,
            candidate_t1,
            state.t2_mask.copy(),
            np.logical_xor(candidate_t1, state.t2_mask),
        )
        attach_candidate_and_live_regions(candidate, state)

        candidate_output = verifier.verify(
            candidate,
            output.quality_score,
            action,
            state,
        )

        self.assertEqual(candidate_output.comparison, "better")
        self.assertTrue(candidate_output.accept)
        self.assertIn("candidate_evidence", verifier.backend.calls)

    def test_reset_clears_persisted_executable_plan(self):
        verifier = StagedQwenVerifier(
            ScriptedBackend(
                t1="background",
                t2="background",
                error="false_positive_change",
            )
        )

        verifier.verify(make_state(), None, None)
        self.assertIsNotNone(verifier._planned_action)

        verifier.reset()

        self.assertIsNone(verifier._planned_action)
        self.assertIsNone(verifier._planned_diagnosis)

    def test_invalid_top_diagnosis_falls_back_to_executable_region(self):
        class MixedTargetBackend(SelectAllBackend):
            def generate_stage(self, stage, state, payload, previous_state=None):
                if stage == "audit":
                    self.calls.append(stage)
                    region_id = payload["region"]["region_id"]
                    return {
                        "region_id": region_id,
                        "visual_judgment": {
                            "t1_state": "background",
                            "t2_state": "background",
                            "evidence": "The white component has no building transition.",
                        },
                        "diagnosis": {
                            "audit_checklist": audit_checklist_for(
                                "false_positive_change"
                            ),
                        },
                    }
                return super().generate_stage(stage, state, payload, previous_state)

        source = make_two_region_state()
        t2 = source.t2_mask.copy()
        t2[20:26, 20:26] = False
        state = ChangeState(
            source.t1_image,
            source.t2_image,
            source.query,
            source.t1_mask.copy(),
            t2,
            source.change_mask.copy(),
        )
        attach_verifier_regions(state, max_regions=6, min_component_area=1)
        verifier = StagedQwenVerifier(MixedTargetBackend())

        output = verifier.verify(state, None, None)

        self.assertTrue(output.verifier_valid)
        self.assertTrue(output.localization_valid)
        self.assertEqual(output.suggested_action, "negative_point")
        self.assertEqual(output.target_view, "t2")
        self.assertEqual(
            verifier.last_evidence["stage_trace"]["plan"]["region_id"], "r1"
        )

    def test_valid_false_positive_plan_reuses_environment_seed(self):
        backend = ScriptedBackend(
            t1="background",
            t2="background",
            error="false_positive_change",
            target="t2",
            action="negative_point",
        )
        verifier = StagedQwenVerifier(backend)
        state = make_state()

        output = verifier.verify(state, None, None)

        self.assertTrue(output.verifier_valid)
        self.assertEqual(output.error_type, "false_positive_change")
        self.assertEqual(output.target_view, "t2")
        self.assertEqual(output.suggested_action, "negative_point")
        self.assertEqual(output.error_region[0], output.error_region[2])
        self.assertEqual(output.error_region[1], output.error_region[3])
        proposal_seed = verifier.last_evidence["stage_trace"]["evidence"][0][
            "component_seed_normalized_1000"
        ]
        self.assertEqual(list(output.error_region[:2]), proposal_seed)
        self.assertEqual(
            verifier._planned_action,
            AgentAction(
                "t2",
                "negative_point",
                coordinate=normalized_point_to_pixel(
                    tuple(proposal_seed), state.image_size
                ),
            ),
        )
        self.assertNotIn("plan", backend.calls)

    def test_initial_without_plan_cannot_authorize_external_candidate(self):
        previous = make_state()
        verifier = StagedQwenVerifier(ScriptedBackend(error="mixed_error"))

        initial = verifier.verify(previous, None, None)

        self.assertIsNone(initial.suggested_action)
        self.assertIsNone(verifier._planned_action)
        t2_mask = previous.t2_mask.copy()
        t2_mask[0:2, 0:2] = True
        candidate = ChangeState(
            previous.t1_image,
            previous.t2_image,
            previous.query,
            previous.t1_mask.copy(),
            t2_mask,
            np.logical_xor(previous.t1_mask, t2_mask),
        )
        attach_candidate_and_live_regions(candidate, previous)
        action = AgentAction("t2", "positive_point", coordinate=(0, 0))

        output = verifier.verify(
            candidate,
            initial.quality_score,
            action,
            previous,
        )

        self.assertEqual(output.comparison, "uncertain")
        self.assertFalse(output.accept)
        self.assertNotIn("candidate_evidence", verifier.backend.calls)
        trace = verifier.last_evidence["stage_trace"]
        transition = trace["transition_assessment"]
        self.assertFalse(transition["evidence_sufficient"])
        self.assertEqual(trace["evidence"], [])
        self.assertEqual(trace["audited_region_ids"], [])
        self.assertEqual(trace["judgments"], [])
        self.assertIn(
            "attempted_action_mismatches_persisted_plan",
            transition["evidence"],
        )

    def test_atomic_audit_fails_changed_region_id_closed(self):
        class UnknownRegionBackend(ScriptedBackend):
            def generate_stage(self, stage, state, payload, previous_state=None):
                response = super().generate_stage(
                    stage, state, payload, previous_state
                )
                if stage == "audit":
                    response["region_id"] = "invented"
                return response

        verifier = StagedQwenVerifier(UnknownRegionBackend(), max_retries=1)

        output = verifier.verify(make_state(), None, None)

        self.assertTrue(output.verifier_valid)
        self.assertFalse(output.localization_valid)
        self.assertIsNone(output.suggested_action)
        self.assertIn(
            "wrong region_id",
            verifier.last_evidence["regional_validation_errors"][0]["error"],
        )

    def test_candidate_comparison_receives_previous_state_and_accepts_better(self):
        candidate = make_state()
        previous_mask = np.zeros_like(candidate.change_mask)
        previous = ChangeState(
            candidate.t1_image,
            candidate.t2_image,
            candidate.query,
            candidate.t1_mask,
            previous_mask,
            previous_mask,
        )
        attach_candidate_and_live_regions(candidate, previous)
        backend = ScriptedBackend(error="none", target=None)
        verifier = StagedQwenVerifier(backend)
        verifier._planned_diagnosis = Diagnosis(
            "r0", "false_negative", "t2", None
        )

        action = AgentAction("t2", "positive_point", coordinate=(4, 4))
        verifier._planned_action = action
        output = verifier.verify(candidate, 0.4, action, previous)

        self.assertTrue(output.verifier_valid)
        self.assertEqual(output.comparison, "better")
        self.assertTrue(output.accept)
        self.assertTrue(output.stop)
        transition = verifier.last_evidence["stage_trace"]["transition_assessment"]
        self.assertTrue(transition["intended_error_improved"])
        self.assertFalse(transition["introduced_false_negative"])
        self.assertTrue(transition["evidence_sufficient"])
        self.assertEqual(transition["source"], "runtime_candidate_evidence")
        transition_previous_seen = [
            previous_seen
            for stage, previous_seen in zip(backend.calls, backend.previous_seen)
            if stage == "candidate_evidence"
        ]
        self.assertTrue(transition_previous_seen)
        self.assertTrue(all(transition_previous_seen))
        self.assertEqual(
            verifier.last_evidence["stage_trace"]["mode"], "candidate"
        )

    def test_runtime_rejects_removed_real_change_without_candidate_diagnosis(self):
        previous = make_state()
        empty = np.zeros_like(previous.change_mask)
        candidate = ChangeState(
            previous.t1_image,
            previous.t2_image,
            previous.query,
            previous.t1_mask,
            empty,
            empty,
        )
        attach_candidate_and_live_regions(candidate, previous)
        backend = ScriptedBackend(t1="background", t2="building")
        verifier = StagedQwenVerifier(backend)
        verifier._planned_diagnosis = Diagnosis(
            "r0", "false_positive_change", "t2", None
        )

        action = AgentAction("t2", "negative_point", coordinate=(4, 4))
        verifier._planned_action = action
        output = verifier.verify(candidate, 0.4, action, previous)

        self.assertTrue(output.verifier_valid)
        self.assertEqual(output.comparison, "worse")
        self.assertFalse(output.accept)
        self.assertFalse(output.stop)
        transition = verifier.last_evidence["stage_trace"]["transition_assessment"]
        self.assertTrue(transition["introduced_false_negative"])
        self.assertNotIn("candidate_diagnosis", backend.calls)
        self.assertNotIn("decision", backend.calls)

    def test_runtime_accepts_removed_unsupported_change(self):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        previous_mask = np.zeros((16, 16), dtype=bool)
        previous_mask[4:9, 4:9] = True
        empty = np.zeros_like(previous_mask)
        previous = ChangeState(
            image, image, "building", empty, previous_mask, previous_mask
        )
        candidate = ChangeState(image, image, "building", empty, empty, empty)
        attach_candidate_and_live_regions(candidate, previous)
        backend = ScriptedBackend(t1="background", t2="background")
        verifier = StagedQwenVerifier(backend)
        verifier._planned_diagnosis = Diagnosis(
            "r0", "false_positive_change", "t2", None
        )
        verifier._planned_action = AgentAction(
            "t2", "negative_point", coordinate=(4, 4)
        )

        output = verifier.verify(
            candidate,
            0.4,
            AgentAction("t2", "negative_point", coordinate=(4, 4)),
            previous,
        )

        self.assertEqual(output.comparison, "better")
        self.assertTrue(output.accept)
        self.assertTrue(output.stop)

    def test_empty_replan_catalog_cannot_bypass_uncovered_pixel_gate(self):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        previous_mask = np.zeros((16, 16), dtype=bool)
        previous_mask[4:9, 4:9] = True
        empty = np.zeros_like(previous_mask)
        previous = ChangeState(
            image, image, "building", empty, previous_mask, previous_mask
        )
        candidate = ChangeState(image, image, "building", empty, empty, empty)
        attach_verifier_regions(
            candidate, previous, max_regions=6, min_component_area=1
        )
        candidate.evidence["verifier_next_live_region_proposals"] = []
        candidate.evidence["verifier_next_live_mask_facts"] = {
            "initial_audit_uncovered_pixels": 1,
        }
        verifier = StagedQwenVerifier(
            ScriptedBackend(t1="background", t2="background")
        )
        verifier._planned_diagnosis = Diagnosis(
            "r0", "false_positive_change", "t2", None
        )
        verifier._planned_action = AgentAction(
            "t2", "negative_point", coordinate=(4, 4)
        )

        output = verifier.verify(
            candidate,
            0.4,
            AgentAction("t2", "negative_point", coordinate=(4, 4)),
            previous,
        )

        self.assertEqual(output.comparison, "better")
        self.assertTrue(output.accept)
        self.assertFalse(output.stop)
        self.assertFalse(output.localization_valid)
        self.assertIsNone(output.suggested_action)
        trace = verifier.last_evidence["stage_trace"]
        self.assertFalse(trace["state_completion_gate_passed"])
        self.assertEqual(
            trace["state_completion_gate_reason"],
            "environment audit mask has uncovered pixels",
        )

    def test_ambiguous_candidate_evidence_fails_closed(self):
        previous = make_state()
        empty = np.zeros_like(previous.change_mask)
        candidate = ChangeState(
            previous.t1_image,
            previous.t2_image,
            previous.query,
            previous.t1_mask,
            empty,
            empty,
        )
        attach_verifier_regions(candidate, previous, max_regions=6, min_component_area=1)
        verifier = StagedQwenVerifier(ScriptedBackend(evidence_quality="ambiguous"))
        verifier._planned_diagnosis = Diagnosis(
            "r0", "false_positive_change", "t2", None
        )
        verifier._planned_action = AgentAction(
            "t2", "negative_point", coordinate=(4, 4)
        )

        output = verifier.verify(
            candidate,
            0.4,
            AgentAction("t2", "negative_point", coordinate=(4, 4)),
            previous,
        )

        self.assertEqual(output.comparison, "uncertain")
        self.assertFalse(output.accept)
        transition = verifier.last_evidence["stage_trace"]["transition_assessment"]
        self.assertFalse(transition["evidence_sufficient"])

    def test_candidate_negative_point_rechecks_accepted_target_seed(self):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        target_mask = np.zeros((16, 16), dtype=bool)
        previous_change = np.zeros_like(target_mask)
        previous_change[4:9, 4:9] = True
        candidate_change = np.zeros_like(target_mask)
        previous = ChangeState(
            image,
            image,
            "building",
            target_mask,
            target_mask,
            previous_change,
        )
        candidate = ChangeState(
            image,
            image,
            "building",
            target_mask,
            target_mask,
            candidate_change,
        )
        attach_verifier_regions(
            candidate, previous, max_regions=6, min_component_area=1
        )
        verifier = StagedQwenVerifier(ScriptedBackend())

        output = verifier.verify(
            candidate,
            0.4,
            AgentAction("t2", "negative_point", coordinate=(4, 4)),
            previous,
        )

        self.assertFalse(output.verifier_valid)
        self.assertIn(
            "negative point requires a white seed",
            verifier.last_evidence["validation_errors"][0],
        )

    def test_candidate_evidence_prompt_has_no_mask_error_reclassification(self):
        prompt = _stage_prompt(
            "candidate_evidence",
            {"visual_context": "proposal", "target_class": "building"},
        )

        self.assertIn("judge only the pixels marked white", prompt)
        self.assertIn("cyan contour", prompt)
        self.assertIn("presentation padding", prompt)
        self.assertIn("ignore unedited objects", prompt)
        self.assertIn("Do not diagnose false_positive_change", prompt)
        self.assertIn("Runtime combines this observation", prompt)
        self.assertNotIn('"visual_confidence":0.0', prompt)

    def test_initial_feedback_and_readiness_are_runtime_derived(self):
        backend = ScriptedBackend(error="none", target=None)
        verifier = StagedQwenVerifier(backend)

        output = verifier.verify(make_state(), None, None)

        self.assertNotIn("decision", backend.calls)
        self.assertIn("compact checklist", output.feedback)
        self.assertTrue(output.accept)
        self.assertTrue(output.stop)


if __name__ == "__main__":
    unittest.main()
