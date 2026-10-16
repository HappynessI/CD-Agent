import copy
import hashlib
import json
import tempfile
import unittest

import numpy as np

from change_agent.adapters.omniovcd_adapter import (
    InitializationResult,
    MaskPairProcessor,
)
from change_agent.adapters.staged_verifier import StagedQwenVerifier
from change_agent.adapters.stage_backends import _stage_prompt
from change_agent.adapters.terminal_review_verifier import (
    TerminalReviewVerifier,
    _accepted_state_id,
    _rebuild_plan_null_after_allowed_repair,
    _round_one_call_identities,
    _round_one_counters,
    _round_two_calls,
    _select_plan_null_review_region,
)
from change_agent.environment import ChangeAgentEnvironment
from change_agent.executor import ActionExecutor
from change_agent.terminal_review import (
    PLAN_NULL_ALTERNATIVE_PROMPT_VERSION,
    PLAN_NULL_UNREVIEWED_PROMPT_VERSION,
    SELECTION_POLICY_VERSION,
    TerminalReviewPolicy,
)
from change_agent.state import AgentAction, ChangeState, VerifierOutput
from change_agent.verifier_regions import EMPTY_MASK_SCAN_PROPOSAL_VERSION
from change_agent.verifier_protocol import (
    AuditChecklist,
    Diagnosis,
    EvidenceJudgment,
    EvidenceRecord,
    StageProtocolError,
)
from tools.run_levir_change_agent import _execute_verifier_action


def _audit_checklist(error_type: str) -> dict[str, str]:
    result = {
        "evidence_sufficient": "pass",
        "target_class_only": "pass",
        "white_pixels_supported": "pass",
        "boundary_alignment": "pass",
        "internal_holes_absent": "pass",
        "changed_object_extent_complete": "pass",
        "fragment_artifacts_absent": "pass",
    }
    if error_type == "false_positive_change":
        result["white_pixels_supported"] = "fail"
    elif error_type == "false_negative":
        result["changed_object_extent_complete"] = "fail"
    elif error_type == "mixed_error":
        result["white_pixels_supported"] = "fail"
        result["changed_object_extent_complete"] = "fail"
    elif error_type == "uncertain_region":
        result = {key: "uncertain" for key in result}
    return result


class _AuditedBackend:
    """Deterministic stage backend retaining the production audit contract."""

    def __init__(
        self,
        round_errors,
        *,
        candidate_states=("background", "background"),
        finish_t1_pixel_count=None,
    ):
        self.round_errors = list(round_errors)
        self.candidate_states = candidate_states
        self.finish_t1_pixel_count = finish_t1_pixel_count
        self.call_history = []
        self.payloads = []

    def reset_audit(self):
        self.call_history = []
        self.payloads = []

    def generate_stage(self, stage, state, payload, previous_state=None):
        round_index = int(payload.get("terminal_review", {}).get("round_index", 1))
        error_type = self.round_errors[min(round_index - 1, len(self.round_errors) - 1)]
        region = payload["region"]
        region_id = region["region_id"]
        if (
            self.finish_t1_pixel_count is not None
            and int(state.t1_mask.sum())
            == self.finish_t1_pixel_count
        ):
            error_type = "none"
        elif (
            round_index == 2
            and error_type == "false_positive_change"
            and region_id != "r0"
        ):
            error_type = "none"
        if stage == "candidate_evidence":
            response = {
                "region_id": region_id,
                "visual_judgment": {
                    "t1_state": self.candidate_states[0],
                    "t2_state": self.candidate_states[1],
                    "evidence_quality": "clear",
                },
            }
        else:
            if error_type == "false_negative":
                t1_state, t2_state = "background", "building"
                evidence = (
                    "The later RGB crop contains a compact building absent earlier."
                )
            elif error_type == "none" and region.get(
                "editable_seed_white", {}
            ).get("t1"):
                t1_state, t2_state = "building", "background"
                evidence = (
                    "The earlier RGB crop contains a compact building absent later."
                )
            elif error_type == "none" and region.get(
                "editable_seed_white", {}
            ).get("t2"):
                t1_state, t2_state = "background", "building"
                evidence = (
                    "The later RGB crop contains a compact building absent earlier."
                )
            else:
                t1_state, t2_state = "background", "background"
                evidence = "The paired RGB crops show the same paved background."
            response = {
                "region_id": region_id,
                "visual_judgment": {
                    "t1_state": t1_state,
                    "t2_state": t2_state,
                    "evidence": evidence,
                },
                "diagnosis": {
                    "audit_checklist": _audit_checklist(error_type),
                },
            }
        prompt = _stage_prompt(stage, payload, None)
        self.call_history.append(
            {
                "backend": "fixture",
                "stage": stage,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "attempt_kind": "initial",
                "network_retry_attempt_count": 0,
                "visual_input_scope": "standard",
                "raw_response": json.dumps(response, sort_keys=True),
                "parsed_output": copy.deepcopy(response),
            }
        )
        self.payloads.append(copy.deepcopy(payload))
        return response


class _InitialMaskBackend:
    def __init__(self, t1_mask, t2_mask):
        self.t1_mask = np.asarray(t1_mask, dtype=bool)
        self.t2_mask = np.asarray(t2_mask, dtype=bool)
        self.processor = MaskPairProcessor(overlap_threshold=0.5)

    def initialize(self, t1_image, t2_image, query):
        update = self.processor.rebuild(self.t1_mask, self.t2_mask)
        return InitializationResult(self.t1_mask, self.t2_mask, update)

    def rebuild(self, t1_mask, t2_mask, evidence):
        return self.processor.rebuild(t1_mask, t2_mask, evidence)


class _RemoveClickedComponent:
    def refine(
        self,
        image,
        initial_mask,
        coordinate,
        is_positive,
        click_history=(),
    ):
        if is_positive:
            raise AssertionError("fixture authorizes only negative points")
        result = np.asarray(initial_mask, dtype=bool).copy()
        x, y = coordinate
        result[y, x] = False
        return result


class _UnexpectedBox:
    def segment_box(self, image, box_cxcywh_normalized, query):
        raise AssertionError("fixture must not invoke the box tool")


class _AcceptedFinishReviewVerifier:
    """Minimal hook fixture proving accepted finish is reviewed before done."""

    def __init__(self, reviewed_output, *, initial_output=None):
        self.reviewed_output = reviewed_output
        self.initial_output = initial_output
        self.review_calls = []
        self.last_evidence = {}

    def verify(self, state, previous_score, previous_action, previous_state=None):
        if previous_state is None:
            return self.initial_output or VerifierOutput(
                comparison="initial",
                error_type="none",
                suggested_action="finish",
                accept=True,
                stop=True,
            )
        return VerifierOutput(
            comparison="better",
            error_type="none",
            suggested_action="finish",
            accept=True,
            stop=True,
        )

    def review_after_acceptance(self, state, accepted_feedback):
        self.review_calls.append((state.step_index, accepted_feedback))
        return self.reviewed_output


class TerminalReviewIntegrationTest(unittest.TestCase):
    @staticmethod
    def _raw_plan_null_fixture(
        *,
        region_ids=("r0", "r1"),
        audited_region_ids=("r0", "r1"),
        error_types=("mixed_error", "uncertain_region"),
        regional_validation_errors=(),
        repaired_region_ids=(),
    ):
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        mask = np.zeros((4, 4), dtype=bool)
        proposals = [
            {
                "region_id": region_id,
                "audit_kind": "mixed",
                "component_seed_normalized_1000": [500, 500],
                "box_normalized_1000": [0, 0, 999, 999],
                "component_area": 16,
                "temporal_difference_pixels": 1,
                "t1_mask_pixels": 0,
                "t2_mask_pixels": 0,
                "component_seed_t1_mask_white": False,
                "component_seed_t2_mask_white": False,
            }
            for region_id in region_ids
        ]
        records = tuple(EvidenceRecord.from_proposal(item) for item in proposals)
        state = ChangeState(
            image,
            image.copy(),
            "building",
            mask,
            mask.copy(),
            mask.copy(),
            evidence={
                "verifier_region_proposals": proposals,
                "verifier_mask_facts": {
                    "initial_audit_uncovered_pixels": 0,
                },
            },
        )
        calls = []
        for region_id in audited_region_ids:
            for attempt_kind in (
                ("initial", "repair")
                if region_id in repaired_region_ids
                else ("initial",)
            ):
                call = {
                        "backend": "fixture",
                        "stage": "audit",
                        "attempt_kind": attempt_kind,
                        "prompt_sha256": hashlib.sha256(
                            f"{region_id}:{attempt_kind}".encode("utf-8")
                        ).hexdigest(),
                        "network_retry_attempt_count": 0,
                        "visual_input_scope": "standard",
                        "raw_response": "{}",
                    }
                if attempt_kind == "initial" and region_id in repaired_region_ids:
                    call["raw_response"] = "{"
                    call["parse_error"] = "malformed initial JSON"
                else:
                    call["parsed_output"] = {"region_id": region_id}
                calls.append(call)
        by_region = dict(zip(audited_region_ids, error_types, strict=True))
        diagnoses = []
        judgments = []
        for region_id in audited_region_ids:
            statuses = _audit_checklist(by_region[region_id])
            checklist = AuditChecklist(**statuses)
            diagnoses.append(
                Diagnosis(
                    region_id=region_id,
                    error_type=by_region[region_id],
                    target_view=None,
                    audit_checklist=checklist,
                    summary=f"{region_id} fixture diagnosis",
                ).to_dict()
            )
            error_type = by_region[region_id]
            t1_state = (
                "uncertain"
                if error_type == "uncertain_region"
                else "background"
            )
            t2_state = (
                "uncertain"
                if error_type == "uncertain_region"
                else "background"
            )
            judgments.append(
                EvidenceJudgment(
                    region_id=region_id,
                    t1_state=t1_state,
                    t2_state=t2_state,
                    evidence_quality=(
                        "ambiguous"
                        if error_type == "uncertain_region"
                        else "clear"
                    ),
                    change_mask_state=next(
                        record.change_mask_state
                        for record in records
                        if record.region_id == region_id
                    ),
                    mask_assessment={
                        "none": "correct",
                        "mixed_error": "mixed",
                        "uncertain_region": "uncertain",
                    }[error_type],
                    evidence=f"{region_id} fixture paired RGB evidence",
                ).to_dict()
            )
        if tuple(audited_region_ids) != tuple(region_ids):
            completion_passed = False
            completion_reason = (
                f"audited {len(audited_region_ids)} of {len(region_ids)} "
                "region(s); unaudited regions are not evidence of correctness"
            )
        elif any(
            item["evidence_quality"] != "clear"
            or item["t1_state"] not in {"building", "background"}
            or item["t2_state"] not in {"building", "background"}
            for item in judgments
        ):
            completion_passed = False
            completion_reason = (
                "at least one region lacks sufficient visual evidence"
            )
        elif any(error_type != "none" for error_type in error_types):
            completion_passed = False
            completion_reason = "at least one diagnosed error remains"
        else:
            completion_passed = True
            completion_reason = (
                "all Environment audit regions are covered and diagnosed none"
            )
        evidence = {
            "type": "staged_qwen_verifier",
            "schema_version": StagedQwenVerifier.SCHEMA_VERSION,
            "verifier_valid": True,
            "localization_valid": False,
            "validation_errors": [],
            "regional_validation_errors": list(regional_validation_errors),
            "backend_calls": calls,
            "stage_trace": {
                "mode": "initial",
                "evidence": [record.to_dict() for record in records],
                "audited_region_ids": list(audited_region_ids),
                "judgments": judgments,
                "diagnoses": diagnoses,
                "plan": None,
                "state_completion_gate_passed": completion_passed,
                "state_completion_gate_reason": completion_reason,
            },
        }
        return state, evidence

    def test_plan_null_trigger_is_rebuilt_from_complete_raw_stage_evidence(self):
        state, evidence = self._raw_plan_null_fixture()
        verifier = StagedQwenVerifier(_AuditedBackend(["mixed_error"]))
        output = VerifierOutput(
            comparison="initial",
            error_type="mixed_error",
            target_view="t2",
            suggested_action=None,
            verifier_valid=True,
            localization_valid=False,
        )

        self.assertTrue(
            _rebuild_plan_null_after_allowed_repair(
                output,
                evidence,
                state=state,
                verifier=verifier,
            )
        )

        for mutation in (
            lambda value: value.pop("stage_trace"),
            lambda value: value["stage_trace"].__setitem__("plan", {}),
            lambda value: value["stage_trace"]["evidence"][0].__setitem__(
                "component_area", 15
            ),
            lambda value: value["stage_trace"]["judgments"][1].__setitem__(
                "evidence_quality", "clear"
            ),
            lambda value: value["stage_trace"]["diagnoses"][0][
                "audit_evidence"
            ].__setitem__("evidence_sufficient", "partially forged evidence"),
            lambda value: value["stage_trace"].__setitem__(
                "state_completion_gate_passed", True
            ),
            lambda value: value["stage_trace"].__setitem__(
                "state_completion_gate_reason", "forged completion"
            ),
            lambda value: value["backend_calls"][0].pop(
                "network_retry_attempt_count"
            ),
            lambda value: value["backend_calls"][0].pop("raw_response"),
            lambda value: value["backend_calls"][0].pop("parsed_output"),
            lambda value: value.__setitem__(
                "regional_validation_errors",
                [{"region_id": "r0", "error": "schema mismatch"}],
            ),
        ):
            malformed = copy.deepcopy(evidence)
            mutation(malformed)
            self.assertFalse(
                _rebuild_plan_null_after_allowed_repair(
                    output,
                    malformed,
                    state=state,
                    verifier=verifier,
                )
            )
        self.assertFalse(
            _rebuild_plan_null_after_allowed_repair(
                VerifierOutput(suggested_action="finish", verifier_valid=True),
                evidence,
                state=state,
                verifier=verifier,
            )
        )
        finish_state, forged_plan_null = self._raw_plan_null_fixture(
            error_types=("none", "none"),
        )
        self.assertFalse(
            _rebuild_plan_null_after_allowed_repair(
                VerifierOutput(
                    comparison="initial",
                    error_type="none",
                    target_view="t2",
                    suggested_action=None,
                    verifier_valid=True,
                    localization_valid=False,
                ),
                forged_plan_null,
                state=finish_state,
                verifier=verifier,
            )
        )

        incomplete_state, gate_rejected_finish = self._raw_plan_null_fixture(
            audited_region_ids=("r0",),
            error_types=("none",),
        )
        self.assertTrue(
            _rebuild_plan_null_after_allowed_repair(
                VerifierOutput(
                    comparison="initial",
                    error_type="uncertain_region",
                    target_view="t2",
                    suggested_action=None,
                    verifier_valid=True,
                    localization_valid=False,
                    accept=False,
                    stop=False,
                ),
                gate_rejected_finish,
                state=incomplete_state,
                verifier=verifier,
            )
        )

        repaired_state, repaired = self._raw_plan_null_fixture(
            repaired_region_ids=("r0",),
        )
        self.assertTrue(
            _rebuild_plan_null_after_allowed_repair(
                output,
                repaired,
                state=repaired_state,
                verifier=verifier,
            )
        )

    def test_plan_null_selection_prefers_next_unreviewed_environment_region(self):
        state, evidence = self._raw_plan_null_fixture(
            audited_region_ids=("r0",),
            error_types=("mixed_error",),
        )

        selection = _select_plan_null_review_region(state, evidence)

        self.assertIsNotNone(selection)
        self.assertEqual(selection.region_id, "r1")
        self.assertEqual(
            selection.prompt_version,
            PLAN_NULL_UNREVIEWED_PROMPT_VERSION,
        )
        self.assertEqual(
            selection.selection_branch,
            "next_unreviewed_environment_region",
        )

    def test_plan_null_selection_uses_first_unresolved_in_environment_order(self):
        state, evidence = self._raw_plan_null_fixture(
            error_types=("mixed_error", "uncertain_region"),
        )

        selection = _select_plan_null_review_region(state, evidence)

        self.assertIsNotNone(selection)
        self.assertEqual(selection.region_id, "r0")
        self.assertEqual(
            selection.prompt_version,
            PLAN_NULL_ALTERNATIVE_PROMPT_VERSION,
        )
        self.assertEqual(
            selection.selection_branch,
            "first_unresolved_environment_region",
        )

    def test_plan_null_selection_rejects_reordered_or_incomplete_environment_catalog(self):
        state, evidence = self._raw_plan_null_fixture()
        reordered = copy.deepcopy(evidence)
        trace = reordered["stage_trace"]
        for key in ("evidence", "judgments", "diagnoses"):
            trace[key] = list(reversed(trace[key]))
        trace["audited_region_ids"] = list(
            reversed(trace["audited_region_ids"])
        )
        reordered["backend_calls"] = list(
            reversed(reordered["backend_calls"])
        )

        _, incomplete = self._raw_plan_null_fixture(
            audited_region_ids=("r0",),
            error_types=("mixed_error",),
        )
        incomplete["stage_trace"]["evidence"] = [{"region_id": "r0"}]

        _, foreign = self._raw_plan_null_fixture()
        foreign = copy.deepcopy(foreign)
        for record in foreign["stage_trace"]["evidence"]:
            if record["region_id"] == "r1":
                record["region_id"] = "rx"
        foreign["stage_trace"]["audited_region_ids"][1] = "rx"
        for key in ("judgments", "diagnoses"):
            foreign["stage_trace"][key][1]["region_id"] = "rx"
        foreign["backend_calls"][1]["parsed_output"]["region_id"] = "rx"

        for name, malformed in (
            ("reordered", reordered),
            ("incomplete", incomplete),
            ("foreign", foreign),
        ):
            with self.subTest(name=name):
                self.assertIsNone(
                    _select_plan_null_review_region(state, malformed)
                )

    def test_round_one_call_identities_require_complete_validated_call_groups(self):
        _, evidence = self._raw_plan_null_fixture(
            repaired_region_ids=("r0",),
        )

        identities = _round_one_call_identities(evidence)

        self.assertEqual(
            [(item.stage, item.region_id) for item in identities],
            [("audit", "r0"), ("audit", "r0"), ("audit", "r1")],
        )
        self.assertEqual(
            [item.prompt_sha256 for item in identities],
            [item["prompt_sha256"] for item in evidence["backend_calls"]],
        )
        for mutation in (
            lambda value: value["backend_calls"][0].pop("backend"),
            lambda value: value["backend_calls"][0].pop("raw_response"),
            lambda value: value["backend_calls"][0].pop("visual_input_scope"),
            lambda value: value["backend_calls"][0].__setitem__(
                "parsed_output", {"region_id": "r0"}
            ),
            lambda value: value["backend_calls"][-1].pop("parsed_output"),
            lambda value: value["backend_calls"][1].__setitem__(
                "parse_error", "forged alongside parsed output"
            ),
            lambda value: value["backend_calls"].insert(0, None),
            lambda value: value["backend_calls"].append(
                copy.deepcopy(value["backend_calls"][0])
            ),
        ):
            malformed = copy.deepcopy(evidence)
            mutation(malformed)
            with self.assertRaises(ValueError):
                _round_one_call_identities(malformed)

    def test_round_one_network_retries_are_rebuilt_from_backend_evidence(self):
        evidence = {
            "backend_calls": [
                {
                    "attempt_kind": "initial",
                    "network_retry_attempt_count": 0,
                },
                {
                    "attempt_kind": "repair",
                    "network_retry_attempt_count": 1,
                },
            ]
        }

        counters = _round_one_counters(
            evidence,
            environment_step_count=2,
            max_network_retry_attempts=1,
        )

        self.assertEqual(counters.stage_initial_attempt_count, 1)
        self.assertEqual(counters.schema_repair_attempt_count, 1)
        self.assertEqual(counters.network_retry_attempt_count, 1)
        self.assertEqual(counters.environment_step_count, 2)

    def test_missing_or_malformed_round_one_network_evidence_fails_closed(self):
        with self.assertRaises(ValueError):
            _round_one_counters(
                {"backend_calls": [{"attempt_kind": "initial"}]},
                environment_step_count=0,
            )
        for value in (None, True, -1, "0", 1):
            evidence = {
                "backend_calls": [
                    {
                        "attempt_kind": "initial",
                        "network_retry_attempt_count": value,
                    }
                ]
            }
            with self.subTest(value=value), self.assertRaises(ValueError):
                _round_one_counters(evidence, environment_step_count=0)

    def test_round_two_network_retries_are_not_hardcoded(self):
        call = {
            "backend": "fixture",
            "stage": "audit",
            "attempt_kind": "initial",
            "prompt_sha256": "a" * 64,
            "raw_response": "{}",
            "parsed_output": {"region_id": "r0"},
            "visual_input_scope": "standard",
            "network_retry_attempt_count": 0,
            "terminal_review": {
                "round_index": 2,
                "round_kind": "finish_confirmation",
                "trigger": "provisional_finish",
                "fresh_backend_context": True,
                "inherits_hidden_state": False,
                "copies_round_one_raw_response": False,
                "selection_policy_version": SELECTION_POLICY_VERSION,
                "preferred_region_id": None,
                "prompt_version": None,
                "selection_branch": None,
                "review_region_id": "r0",
                "proposal_version": None,
            },
        }

        calls = _round_two_calls([call])

        self.assertEqual(calls[0].network_retry_attempt_count, 0)
        missing_retry = copy.deepcopy(call)
        missing_retry.pop("network_retry_attempt_count")
        with self.assertRaises(ValueError):
            _round_two_calls([missing_retry])

    def test_round_two_calls_reject_malformed_or_incomplete_raw_audit(self):
        call = {
            "backend": "fixture",
            "stage": "audit",
            "attempt_kind": "initial",
            "prompt_sha256": "a" * 64,
            "raw_response": "{}",
            "parsed_output": {"region_id": "r0"},
            "visual_input_scope": "standard",
            "network_retry_attempt_count": 0,
            "terminal_review": {
                "round_index": 2,
                "round_kind": "plan_null_alternative",
                "trigger": "plan_null_after_allowed_repair",
                "fresh_backend_context": True,
                "inherits_hidden_state": False,
                "copies_round_one_raw_response": False,
                "selection_policy_version": SELECTION_POLICY_VERSION,
                "preferred_region_id": "r0",
                "prompt_version": PLAN_NULL_ALTERNATIVE_PROMPT_VERSION,
                "selection_branch": "first_unresolved_environment_region",
                "review_region_id": "r0",
                "proposal_version": None,
            },
        }
        mutations = (
            lambda value: value.pop("backend"),
            lambda value: value.pop("stage"),
            lambda value: value.pop("raw_response"),
            lambda value: value.pop("parsed_output"),
            lambda value: value.pop("visual_input_scope"),
            lambda value: value.__setitem__("attempt_kind", "repair"),
            lambda value: value.__setitem__(
                "parse_error", "forged alongside parsed output"
            ),
            lambda value: value["terminal_review"].__setitem__(
                "prompt_version", PLAN_NULL_UNREVIEWED_PROMPT_VERSION
            ),
            lambda value: value["terminal_review"].pop("proposal_version"),
            lambda value: value["parsed_output"].__setitem__(
                "region_id", "r1"
            ),
        )

        for index, mutation in enumerate(mutations):
            malformed = copy.deepcopy(call)
            mutation(malformed)
            with self.subTest(index=index), self.assertRaises(ValueError):
                _round_two_calls([malformed])
        with self.assertRaises(ValueError):
            _round_two_calls([None])
        repair = copy.deepcopy(call)
        repair["attempt_kind"] = "repair"
        repair["terminal_review"]["review_region_id"] = "r1"
        repair["terminal_review"]["preferred_region_id"] = "r1"
        repair["parsed_output"]["region_id"] = "r1"
        with self.assertRaises(ValueError):
            _round_two_calls([call, repair])

        same_region_repair = copy.deepcopy(call)
        same_region_repair["attempt_kind"] = "repair"
        same_region_repair["prompt_sha256"] = "b" * 64
        with self.assertRaises(ValueError):
            _round_two_calls([call, same_region_repair])

        failed_initial = copy.deepcopy(call)
        failed_initial.pop("parsed_output")
        failed_initial["parse_error"] = "initial schema mismatch"
        repaired_calls = _round_two_calls(
            [failed_initial, same_region_repair]
        )
        self.assertEqual(
            [item.attempt_kind.value for item in repaired_calls],
            ["initial", "repair"],
        )

        failed_repair = copy.deepcopy(same_region_repair)
        failed_repair.pop("parsed_output")
        failed_repair["parse_error"] = "repair schema mismatch"
        with self.assertRaises(ValueError):
            _round_two_calls([failed_initial, failed_repair])

    @staticmethod
    def _environment(
        *,
        arm: str,
        round_errors,
        image_size=8,
        component_count=1,
        candidate_states=("background", "background"),
        finish_t1_pixel_count=None,
    ):
        image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        t1_mask = np.zeros((image_size, image_size), dtype=bool)
        t1_mask[2, 2] = True
        if component_count == 2:
            t1_mask[image_size - 2, image_size - 2] = True
        t2_mask = np.zeros_like(t1_mask)
        backend = _AuditedBackend(
            round_errors,
            candidate_states=candidate_states,
            finish_t1_pixel_count=finish_t1_pixel_count,
        )
        staged = StagedQwenVerifier(backend, max_regions=6)
        verifier = (
            TerminalReviewVerifier(staged, policy=TerminalReviewPolicy.h1())
            if arm == "h1"
            else TerminalReviewVerifier(
                staged, policy=TerminalReviewPolicy.h0()
            )
            if arm == "h0_decorator"
            else staged
        )
        environment = ChangeAgentEnvironment(
            _InitialMaskBackend(t1_mask, t2_mask),
            ActionExecutor(_RemoveClickedComponent(), _UnexpectedBox()),
            verifier,
            max_steps=3,
            verifier_min_region_area=1,
        )
        return environment, backend, image

    def test_h0_decorator_preserves_legacy_prompt_calls_and_output_exactly(self):
        legacy, legacy_backend, image = self._environment(
            arm="h0", round_errors=("false_positive_change",)
        )
        h0, h0_backend, _ = self._environment(
            arm="h0_decorator", round_errors=("false_positive_change",)
        )

        legacy_output = legacy.reset(
            image, image.copy(), "building"
        ).feedback
        h0_output = h0.reset(image, image.copy(), "building").feedback

        self.assertEqual(h0_output, legacy_output)
        self.assertEqual(h0_backend.payloads, legacy_backend.payloads)
        self.assertEqual(h0_backend.call_history, legacy_backend.call_history)
        self.assertNotIn("terminal_review", h0_backend.payloads[0])
        self.assertFalse(h0.verifier.last_terminal_review_audit["round_two_invoked"])
        self.assertEqual(
            h0.verifier.last_terminal_review_audit["counters_before"],
            h0.verifier.last_terminal_review_audit["counters_after"],
        )

    def test_h1_nonterminal_passthrough_trajectory_is_acyclic(self):
        environment, backend, image = self._environment(
            arm="h1", round_errors=("false_positive_change",)
        )

        observation = environment.reset(image, image.copy(), "building")

        self.assertEqual(observation.feedback.suggested_action, "negative_point")
        audit = environment.verifier.last_terminal_review_audit
        self.assertIsNone(audit["trigger"])
        self.assertFalse(audit["round_two_invoked"])
        self.assertEqual(len(backend.call_history), 1)
        evidence = environment.verifier.last_evidence
        round_one = evidence["terminal_review"]["round_one_evidence"]
        self.assertIsNot(evidence, round_one)
        self.assertNotIn("terminal_review", round_one)

        with tempfile.TemporaryDirectory() as directory:
            trajectory_path = environment.trajectory.save(directory)
            trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))

        serialized_review = trajectory["steps"][0]["execution"][
            "verifier_evidence"
        ]["terminal_review"]
        self.assertIsNone(serialized_review["round_two_evidence"])
        self.assertNotIn(
            "terminal_review", serialized_review["round_one_evidence"]
        )

    def test_h1_initial_finish_is_independent_round_two_and_h0_is_legacy(self):
        h0, h0_backend, image = self._environment(
            arm="h0", round_errors=("none", "false_positive_change")
        )
        h1, h1_backend, _ = self._environment(
            arm="h1", round_errors=("none", "none")
        )

        h0_observation = h0.reset(image, image.copy(), "building")
        h1_observation = h1.reset(image, image.copy(), "building")

        self.assertEqual(h0_observation.feedback.suggested_action, "finish")
        self.assertEqual(len(h0_backend.call_history), 1)
        self.assertNotIn("terminal_review", h0_backend.payloads[0])
        self.assertEqual(h1_observation.feedback.suggested_action, "finish")
        self.assertEqual(len(h1_backend.call_history), 2)
        self.assertNotIn("terminal_review", h1_backend.payloads[0])
        review_context = h1_backend.payloads[1]["terminal_review"]
        self.assertEqual(review_context["round_index"], 2)
        self.assertEqual(review_context["round_kind"], "finish_confirmation")
        self.assertTrue(review_context["fresh_backend_context"])
        self.assertFalse(review_context["inherits_hidden_state"])
        self.assertFalse(review_context["copies_round_one_raw_response"])
        self.assertNotEqual(
            h1_backend.call_history[0]["prompt_sha256"],
            h1_backend.call_history[1]["prompt_sha256"],
        )
        audit = h1.verifier.last_terminal_review_audit
        self.assertTrue(audit["round_two_invoked"])
        self.assertEqual(audit["effective_outcome"], "finish")
        self.assertEqual(
            h1.verifier.terminal_stop_reason,
            "terminate_confirmed_finish",
        )

    def test_h1_plan_null_round_two_safe_plan_still_enters_environment_gates(self):
        environment, backend, image = self._environment(
            arm="h1",
            round_errors=("mixed_error", "false_positive_change"),
            candidate_states=("building", "background"),
        )

        observation = environment.reset(image, image.copy(), "building")

        self.assertEqual(observation.feedback.suggested_action, "negative_point")
        audit = environment.verifier.last_terminal_review_audit
        self.assertEqual(audit["trigger"], "plan_null_after_allowed_repair")
        self.assertEqual(audit["effective_outcome"], "unique_safe_plan")
        self.assertTrue(audit["safe_plan_returned_to_existing_flow"])
        self.assertFalse(audit["tool_executed_by_scheduler"])
        review_context = backend.payloads[1]["terminal_review"]
        self.assertEqual(
            review_context["prompt_version"],
            PLAN_NULL_ALTERNATIVE_PROMPT_VERSION,
        )
        self.assertEqual(
            review_context["selection_branch"],
            "first_unresolved_environment_region",
        )
        self.assertEqual(len(environment.trajectory.entries), 1)

    def test_plan_null_prompt_branch_and_version_must_match(self):
        environment, _, image = self._environment(
            arm="h0",
            round_errors=("mixed_error",),
        )
        environment.reset(image, image.copy(), "building")
        staged = environment.verifier

        for branch, prompt_version in (
            (
                "next_unreviewed_environment_region",
                PLAN_NULL_ALTERNATIVE_PROMPT_VERSION,
            ),
            (
                "first_unresolved_environment_region",
                PLAN_NULL_UNREVIEWED_PROMPT_VERSION,
            ),
            (
                "unregistered_selection_branch",
                PLAN_NULL_ALTERNATIVE_PROMPT_VERSION,
            ),
        ):
            with self.subTest(branch=branch), self.assertRaises(
                StageProtocolError
            ):
                staged.review_accepted_state(
                    environment.state,
                    round_index=2,
                    round_kind="plan_null_alternative",
                    trigger="plan_null_after_allowed_repair",
                    preferred_region_id="r0",
                    prompt_version=prompt_version,
                    selection_branch=branch,
                )

    def test_plan_null_review_cannot_claim_full_state_completion_from_one_region(self):
        environment, _, image = self._environment(
            arm="h1",
            round_errors=("mixed_error", "none"),
            component_count=2,
        )

        observation = environment.reset(image, image.copy(), "building")

        audit = environment.verifier.last_terminal_review_audit
        self.assertEqual(audit["trigger"], "plan_null_after_allowed_repair")
        self.assertEqual(audit["effective_outcome"], "plan_null")
        self.assertIsNone(observation.feedback.suggested_action)
        round_two = environment.verifier.last_evidence["terminal_review"][
            "round_two_evidence"
        ]["terminal_review_round"]
        self.assertFalse(round_two["completion_gate_passed"])
        self.assertIn("audited 1 of 2 region", round_two["completion_gate_reason"])

    def test_real_h1_empty_mask_initial_finish_invokes_independent_round_two(self):
        image = np.zeros((9, 9, 3), dtype=np.uint8)
        empty = np.zeros((9, 9), dtype=bool)
        backend = _AuditedBackend(("none", "none"))
        verifier = TerminalReviewVerifier(
            StagedQwenVerifier(backend, max_regions=9),
            policy=TerminalReviewPolicy.h1(),
        )
        environment = ChangeAgentEnvironment(
            _InitialMaskBackend(empty, empty),
            ActionExecutor(_RemoveClickedComponent(), _UnexpectedBox()),
            verifier,
            max_steps=3,
            verifier_max_regions=9,
            verifier_min_region_area=1,
            empty_mask_scan_grid_size=3,
        )

        observation = environment.reset(image, image.copy(), "building")

        proposals = environment.state.evidence["verifier_region_proposals"]
        self.assertEqual(
            [item["region_id"] for item in proposals],
            [f"s{index}" for index in range(9)],
        )
        self.assertTrue(
            all(
                item["proposal_version"] == EMPTY_MASK_SCAN_PROPOSAL_VERSION
                for item in proposals
            )
        )
        self.assertEqual(observation.feedback.suggested_action, "finish")
        audit = verifier.last_terminal_review_audit
        self.assertTrue(audit["round_two_invoked"])
        self.assertEqual(audit["round_two_kind"], "finish_confirmation")
        self.assertIsNone(audit["contract_violation"])
        self.assertEqual(len(backend.call_history), 18)
        round_one = {
            (item["parsed_output"]["region_id"], item["prompt_sha256"])
            for item in backend.call_history[:9]
        }
        round_two = {
            (item["parsed_output"]["region_id"], item["prompt_sha256"])
            for item in backend.call_history[9:]
        }
        self.assertTrue(round_two - round_one)
        self.assertTrue(
            all(
                item["terminal_review"]["proposal_version"]
                == EMPTY_MASK_SCAN_PROPOSAL_VERSION
                for item in backend.call_history[9:]
            )
        )

    def test_post_accept_plan_null_is_reviewed_on_the_new_accepted_state(self):
        environment, _, image = self._environment(
            arm="h1",
            round_errors=("mixed_error", "false_positive_change"),
            component_count=2,
        )
        observation = environment.reset(image, image.copy(), "building")
        first_state_id = environment.verifier.last_terminal_review_audit[
            "accepted_state_id"
        ]

        observation, errors, executed = _execute_verifier_action(
            environment, observation, loop_index=1
        )

        self.assertEqual(errors, [])
        self.assertTrue(executed)
        entry = environment.trajectory.entries[-1]
        self.assertTrue(entry.execution["candidate_accepted"])
        self.assertIn("post_accept_terminal_review", entry.execution)
        post_review = entry.execution["post_accept_terminal_review"]
        post_audit = post_review["verifier_evidence"]["terminal_review"]
        self.assertEqual(post_audit["trigger"], "plan_null_after_allowed_repair")
        self.assertEqual(post_audit["effective_outcome"], "unique_safe_plan")
        self.assertNotEqual(post_audit["accepted_state_id"], first_state_id)
        self.assertEqual(observation.feedback.suggested_action, "negative_point")
        self.assertFalse(environment.done)

    def test_real_post_accept_finish_confirmation_sets_environment_done(self):
        environment, _, image = self._environment(
            arm="h1",
            round_errors=("mixed_error", "false_positive_change"),
            component_count=2,
            finish_t1_pixel_count=1,
        )
        observation = environment.reset(image, image.copy(), "building")

        observation, errors, executed = _execute_verifier_action(
            environment, observation, loop_index=1
        )

        self.assertEqual(errors, [])
        self.assertTrue(executed)
        entry = environment.trajectory.entries[-1]
        self.assertTrue(entry.execution["candidate_accepted"])
        post_review = entry.execution["post_accept_terminal_review"]
        post_audit = post_review["verifier_evidence"]["terminal_review"]
        self.assertEqual(post_audit["trigger"], "provisional_finish")
        self.assertEqual(post_audit["effective_outcome"], "finish")
        self.assertEqual(observation.feedback.suggested_action, "finish")
        self.assertTrue(observation.feedback.stop)
        self.assertTrue(environment.done)
        self.assertEqual(
            environment.verifier.terminal_stop_reason,
            "terminate_confirmed_finish",
        )

    def test_safe_review_plan_can_be_rejected_by_unchanged_hard_gate(self):
        environment, _, image = self._environment(
            arm="h1",
            round_errors=("mixed_error", "false_positive_change"),
            image_size=8,
        )
        environment.max_selection_area_delta = 0.0
        observation = environment.reset(image, image.copy(), "building")

        _, errors, executed = _execute_verifier_action(
            environment, observation, loop_index=1
        )

        self.assertEqual(errors, [])
        self.assertTrue(executed)
        entry = environment.trajectory.entries[-1]
        self.assertFalse(entry.execution["candidate_accepted"])
        self.assertIn(
            "mask_area_delta_exceeded",
            entry.execution["candidate_rejection_reasons"],
        )
        self.assertTrue(entry.execution["verifier_skipped_by_hard_gate"])

    def test_accepted_state_identity_ignores_step_counter_but_not_masks(self):
        environment, backend, image = self._environment(
            arm="h1", round_errors=("none", "none")
        )
        environment.reset(image, image.copy(), "building")
        state = environment.state
        same_content = state.clone()
        same_content.step_index = state.step_index + 1
        changed_content = state.clone()
        changed_content.t1_mask = changed_content.t1_mask.copy()
        changed_content.t1_mask[0, 0] = ~changed_content.t1_mask[0, 0]

        self.assertEqual(_accepted_state_id(state), _accepted_state_id(same_content))
        self.assertNotEqual(
            _accepted_state_id(state), _accepted_state_id(changed_content)
        )

        call_count_after_first_review = len(backend.call_history)
        second_output = environment.verifier.verify(
            same_content, None, None, None
        )

        self.assertEqual(
            len(backend.call_history), call_count_after_first_review + 1
        )
        self.assertIsNone(second_output.suggested_action)
        second_audit = environment.verifier.last_terminal_review_audit
        self.assertFalse(second_audit["round_two_invoked"])
        self.assertEqual(
            second_audit["contract_violation"],
            "round_two_already_used_for_accepted_state",
        )

    def test_accepted_finish_is_reviewed_and_done_uses_reviewed_feedback(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        empty = np.zeros((8, 8), dtype=bool)
        reviewed = VerifierOutput(
            comparison="initial",
            error_type="false_positive_change",
            target_view="t2",
            error_region=(500, 500, 500, 500),
            suggested_action="negative_point",
            accept=False,
            stop=False,
        )
        verifier = _AcceptedFinishReviewVerifier(reviewed)
        environment = ChangeAgentEnvironment(
            _InitialMaskBackend(empty, empty),
            ActionExecutor(_RemoveClickedComponent(), _UnexpectedBox()),
            verifier,
            max_steps=3,
            verifier_min_region_area=1,
            empty_mask_scan_grid_size=1,
        )
        environment.reset(image, image.copy(), "building")

        observation, done = environment.step(
            '{"target_view":"t2","action":"finish"}'
        )

        self.assertEqual(len(verifier.review_calls), 1)
        self.assertIs(observation.feedback, reviewed)
        self.assertEqual(observation.feedback.suggested_action, "negative_point")
        self.assertFalse(done)
        self.assertFalse(environment.done)
        self.assertIn(
            "post_accept_terminal_review",
            environment.trajectory.entries[-1].execution,
        )

    def test_accepted_finish_terminates_only_after_review_confirms_finish(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        empty = np.zeros((8, 8), dtype=bool)
        confirmed = VerifierOutput(
            comparison="initial",
            error_type="none",
            suggested_action="finish",
            accept=True,
            stop=True,
        )
        verifier = _AcceptedFinishReviewVerifier(confirmed)
        environment = ChangeAgentEnvironment(
            _InitialMaskBackend(empty, empty),
            ActionExecutor(_RemoveClickedComponent(), _UnexpectedBox()),
            verifier,
            max_steps=3,
            verifier_min_region_area=1,
            empty_mask_scan_grid_size=1,
        )
        environment.reset(image, image.copy(), "building")

        observation, done = environment.step(
            '{"target_view":"t2","action":"finish"}'
        )

        self.assertEqual(len(verifier.review_calls), 1)
        self.assertIs(observation.feedback, confirmed)
        self.assertTrue(done)
        self.assertTrue(environment.done)

    def test_confirmed_post_accept_review_terminates_after_nonfinish_tool(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        t1_mask = np.zeros((8, 8), dtype=bool)
        t1_mask[2, 2] = True
        t2_mask = np.zeros_like(t1_mask)
        initial = VerifierOutput(
            comparison="initial",
            error_type="false_positive_change",
            target_view="t1",
            error_region=(286, 286, 286, 286),
            suggested_action="negative_point",
            accept=False,
            stop=False,
        )
        confirmed = VerifierOutput(
            comparison="initial",
            error_type="none",
            suggested_action="finish",
            accept=True,
            stop=True,
        )
        verifier = _AcceptedFinishReviewVerifier(
            confirmed, initial_output=initial
        )
        environment = ChangeAgentEnvironment(
            _InitialMaskBackend(t1_mask, t2_mask),
            ActionExecutor(_RemoveClickedComponent(), _UnexpectedBox()),
            verifier,
            max_steps=3,
            verifier_min_region_area=1,
        )
        environment.reset(image, image.copy(), "building")

        observation, done = environment.step(
            AgentAction("t1", "negative_point", coordinate=(2, 2))
        )

        self.assertEqual(len(verifier.review_calls), 1)
        self.assertIs(observation.feedback, confirmed)
        self.assertTrue(done)
        self.assertTrue(environment.done)
        self.assertEqual(
            environment.trajectory.entries[-1].verifier.suggested_action,
            "finish",
        )
        self.assertIn(
            "post_accept_terminal_review",
            environment.trajectory.entries[-1].execution,
        )


if __name__ == "__main__":
    unittest.main()
