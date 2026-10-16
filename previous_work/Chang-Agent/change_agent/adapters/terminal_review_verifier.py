"""P1 H1 decorator for one independent, GT-free terminal review.

The wrapped :class:`StagedQwenVerifier` remains the sole owner of parsing,
Environment proposal geometry and candidate assessment.  This decorator only
schedules round two and selects the frozen transition; it has no executor or
tool API.
"""

from __future__ import annotations

import copy
import hashlib
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..state import AgentAction, ChangeState, VerifierOutput
from ..terminal_review import (
    ALLOWED_TRIGGER_INPUTS,
    BackendCallIdentity,
    PLAN_NULL_ALTERNATIVE_PROMPT_VERSION,
    PLAN_NULL_UNREVIEWED_PROMPT_VERSION,
    ReviewNextState,
    RoundTwoCallAudit,
    RoundTwoObservation,
    RoundTwoOutcome,
    RoundTwoRequest,
    StageAttemptKind,
    SELECTION_POLICY_VERSION,
    TerminalReviewCounters,
    TerminalReviewPolicy,
    TerminalReviewRequest,
    TerminalReviewScheduler,
    TerminalTrigger,
    select_terminal_trigger,
)
from ..verifier_regions import EMPTY_MASK_SCAN_PROPOSAL_VERSION
from ..verifier_protocol import (
    AUDIT_CHECKS,
    AuditChecklist,
    Diagnosis,
    EvidenceJudgment,
    EvidenceRecord,
    StageProtocolError,
)
from .staged_verifier import (
    StagedQwenVerifier,
    _runtime_target_view,
    _uncertain_audit_checklist,
)


_PLAN_NULL_PROMPT_VERSION_BY_BRANCH = {
    "next_unreviewed_environment_region": (
        PLAN_NULL_UNREVIEWED_PROMPT_VERSION
    ),
    "first_unresolved_environment_region": (
        PLAN_NULL_ALTERNATIVE_PROMPT_VERSION
    ),
}


class TerminalReviewVerifier:
    """Decorate the staged verifier with the frozen H1 terminal scheduler."""

    SCHEMA_VERSION = StagedQwenVerifier.SCHEMA_VERSION

    def __init__(
        self,
        verifier: StagedQwenVerifier,
        *,
        policy: TerminalReviewPolicy | None = None,
    ):
        self.verifier = verifier
        self.policy = policy or TerminalReviewPolicy.h1()
        self.scheduler: TerminalReviewScheduler[VerifierOutput] = (
            TerminalReviewScheduler(self.policy)
        )
        self.backend = verifier.backend
        self.last_evidence: dict[str, Any] = {}
        self.last_terminal_review_audit: dict[str, Any] | None = None
        self._round_two_evidence: dict[str, Any] | None = None
        self._terminal_outcome: str | None = None

    def reset(self) -> None:
        self.verifier.reset()
        self.scheduler.reset()
        self.last_evidence = {}
        self.last_terminal_review_audit = None
        self._round_two_evidence = None
        self._terminal_outcome = None

    def on_candidate_rejected(self, previous_feedback: VerifierOutput) -> None:
        self.verifier.on_candidate_rejected(previous_feedback)

    def replan_after_rejection(self, *args: Any, **kwargs: Any) -> VerifierOutput:
        output = self.verifier.replan_after_rejection(*args, **kwargs)
        self.last_evidence = copy.deepcopy(self.verifier.last_evidence)
        return output

    def verify(
        self,
        state: ChangeState,
        previous_score: float | None,
        previous_action: AgentAction | None,
        previous_state: ChangeState | None = None,
    ) -> VerifierOutput:
        round_one = self.verifier.verify(
            state,
            previous_score,
            previous_action,
            previous_state,
        )
        if previous_state is not None:
            self.last_evidence = copy.deepcopy(self.verifier.last_evidence)
            return round_one

        return self._resolve_terminal(
            state,
            round_one,
            copy.deepcopy(self.verifier.last_evidence),
        )

    def review_after_acceptance(
        self,
        state: ChangeState,
        accepted_feedback: VerifierOutput,
    ) -> VerifierOutput:
        """Review a terminal result only after existing gates accepted its state."""

        if accepted_feedback.suggested_action not in {None, "finish"}:
            self.last_evidence = copy.deepcopy(self.verifier.last_evidence)
            return accepted_feedback
        return self._resolve_terminal(
            state,
            accepted_feedback,
            copy.deepcopy(self.verifier.last_evidence),
        )

    def _resolve_terminal(
        self,
        state: ChangeState,
        round_one: VerifierOutput,
        round_one_evidence: dict[str, Any],
    ) -> VerifierOutput:
        provisional_finish = bool(
            round_one.verifier_valid
            and round_one.accept
            and round_one.stop
            and round_one.suggested_action == "finish"
            and round_one.error_type == "none"
        )
        plan_null = bool(
            not provisional_finish
            and _rebuild_plan_null_after_allowed_repair(
                round_one,
                round_one_evidence,
                state=state,
                verifier=self.verifier,
            )
        )
        trigger = select_terminal_trigger(
            provisional_finish=provisional_finish,
            plan_null_after_allowed_repair=plan_null,
        )
        # H0 remains a byte/behavior identity control.  H1 must retain every
        # strict round-one initial identity before an eligible round two; a
        # malformed call chain may not be made to look independent by omission.
        call_identities = (
            _round_one_call_identities(round_one_evidence)
            if self.policy.terminal_review and trigger is not None
            else ()
        )
        counters = _round_one_counters(
            round_one_evidence,
            state.step_index,
            max_network_retry_attempts=(
                self.policy.effective_network_retry_limit
            ),
        )
        plan_null_selection = (
            _select_plan_null_review_region(state, round_one_evidence)
            if trigger is TerminalTrigger.PLAN_NULL_AFTER_ALLOWED_REPAIR
            else None
        )
        proposals = state.evidence.get("verifier_region_proposals")
        round_two_unavailable_reason = (
            "no_environment_proposal_available_for_round_two"
            if trigger is not None
            and (proposals is None or isinstance(proposals, list) and not proposals)
            else None
        )
        if (
            trigger is TerminalTrigger.PLAN_NULL_AFTER_ALLOWED_REPAIR
            and round_two_unavailable_reason is None
            and plan_null_selection is None
        ):
            round_two_unavailable_reason = "no_eligible_plan_null_review_region"
        self._round_two_evidence = None
        result = self.scheduler.resolve(
            TerminalReviewRequest(
                accepted_state_id=_accepted_state_id(state),
                round_one_value=round_one,
                trigger=trigger,
                counters=counters,
                trigger_input_fields=tuple(sorted(ALLOWED_TRIGGER_INPUTS)),
                round_one_call_identities=call_identities,
                gt_free=True,
                round_two_already_used=False,
                round_two_unavailable_reason=round_two_unavailable_reason,
            ),
            lambda request: self._observe_round_two(
                state,
                request,
                plan_null_selection=plan_null_selection,
            ),
        )
        self.last_terminal_review_audit = result.audit.to_dict()
        transition_next_state = (
            result.transition.next_state if result.transition else None
        )
        self._terminal_outcome = (
            transition_next_state.value
            if transition_next_state
            in {
                ReviewNextState.TERMINATE_CONFIRMED_FINISH,
                ReviewNextState.TERMINATE_NO_EXECUTABLE_ACTION_AFTER_REVIEW,
            }
            else None
        )
        if result.passthrough:
            final_output = round_one
            # Keep the envelope structurally separate from its raw round-one
            # child.  Aliasing them would make terminal_review.round_one_evidence
            # point back to the outer evidence dict and create a serialization
            # cycle for every non-terminal H1 passthrough.
            final_evidence = copy.deepcopy(round_one_evidence)
        elif result.value is not None:
            final_output = result.value
            final_evidence = copy.deepcopy(
                self._round_two_evidence or round_one_evidence
            )
        else:
            self.verifier._planned_diagnosis = None
            self.verifier._planned_action = None
            final_output = _no_action_output(
                round_one,
                effective_outcome=result.audit.effective_outcome,
                next_state=(
                    result.transition.next_state if result.transition else None
                ),
            )
            final_evidence = copy.deepcopy(
                self._round_two_evidence or round_one_evidence
            )
        terminal_review = dict(self.last_terminal_review_audit)
        terminal_review["round_one_evidence"] = round_one_evidence
        terminal_review["round_two_evidence"] = copy.deepcopy(
            self._round_two_evidence
        )
        final_evidence["terminal_review"] = terminal_review
        final_evidence["decision_mode"] = (
            "gtfree_independent_terminal_review_v1"
        )
        self.last_evidence = final_evidence
        return final_output

    @property
    def terminal_stop_reason(self) -> str | None:
        return self._terminal_outcome

    def _observe_round_two(
        self,
        state: ChangeState,
        request: RoundTwoRequest,
        *,
        plan_null_selection: "_PlanNullReviewSelection | None",
    ) -> RoundTwoObservation[VerifierOutput]:
        preferred_region_id = (
            plan_null_selection.region_id
            if plan_null_selection is not None
            else None
        )
        output, evidence = self.verifier.review_accepted_state(
            state,
            round_index=request.round_index,
            round_kind=request.round_kind.value,
            trigger=request.trigger.value,
            preferred_region_id=preferred_region_id,
            prompt_version=(
                plan_null_selection.prompt_version
                if plan_null_selection is not None
                else None
            ),
            selection_branch=(
                plan_null_selection.selection_branch
                if plan_null_selection is not None
                else None
            ),
        )
        calls = _round_two_calls(evidence.get("backend_calls"))
        diagnoses = evidence.get("stage_trace", {}).get("diagnoses", [])
        diagnosis_types = [
            item.get("error_type")
            for item in diagnoses
            if isinstance(item, Mapping)
        ]
        safe_plan_count = _nonnegative_int(
            evidence.get("safe_plan_candidate_count", 0),
            name="safe_plan_candidate_count",
        )
        schema_valid = evidence.get("schema_valid") is True
        completion_passed = evidence.get("completion_gate_passed") is True
        if not schema_valid:
            outcome = RoundTwoOutcome.SCHEMA_INVALID_AFTER_ALLOWED_REPAIR
            value = None
        elif "uncertain_region" in diagnosis_types:
            outcome = RoundTwoOutcome.UNRESOLVED
            value = None
        elif (
            safe_plan_count == 1
            and sum(item != "none" for item in diagnosis_types) == 1
            and output.suggested_action in {
                "positive_point",
                "negative_point",
                "box",
            }
        ):
            outcome = RoundTwoOutcome.UNIQUE_SAFE_PLAN
            value = output
        elif safe_plan_count > 0 or any(
            item in {"false_positive_change", "false_negative", "mixed_error"}
            for item in diagnosis_types
        ):
            outcome = RoundTwoOutcome.UNSAFE_OR_NONUNIQUE_PLAN
            value = None
        elif (
            completion_passed
            and output.suggested_action == "finish"
            and all(item == "none" for item in diagnosis_types)
        ):
            outcome = RoundTwoOutcome.FINISH
            value = output
        else:
            outcome = RoundTwoOutcome.PLAN_NULL
            value = None
        reported_safe_plan_count = safe_plan_count
        if outcome is RoundTwoOutcome.FINISH:
            reported_safe_plan_count = 0
        self._round_two_evidence = {
            "type": "staged_qwen_verifier",
            "schema_version": self.verifier.SCHEMA_VERSION,
            "verifier_valid": bool(schema_valid),
            "localization_valid": bool(output.localization_valid),
            "stage_trace": copy.deepcopy(evidence.get("stage_trace")),
            "validation_errors": [],
            "regional_validation_errors": copy.deepcopy(
                evidence.get("regional_validation_errors", [])
            ),
            "backend_calls": copy.deepcopy(evidence.get("backend_calls", [])),
            "terminal_review_round": {
                key: copy.deepcopy(value)
                for key, value in evidence.items()
                if key not in {"stage_trace", "backend_calls"}
            },
        }
        return RoundTwoObservation(
            outcome=outcome,
            value=value,
            safe_plan_candidate_count=reported_safe_plan_count,
            backend_calls=calls,
            completion_gate_passed=(
                True if outcome is RoundTwoOutcome.FINISH else completion_passed
            ),
            schema_valid=(
                False
                if outcome
                is RoundTwoOutcome.SCHEMA_INVALID_AFTER_ALLOWED_REPAIR
                else True
            ),
            gt_free=True,
            fresh_backend_context=True,
            inherits_hidden_state=False,
            copied_round_one_raw_response=False,
        )


def _accepted_state_id(state: ChangeState) -> str:
    """Fingerprint accepted Environment content, independent of bookkeeping.

    ``step_index`` is an episode counter rather than state content.  Including
    it would let a caller obtain a second H1 review merely by cloning identical
    masks at a later step, which violates the exactly-once token for one
    accepted Environment state.  Query, images, and all three live masks are
    the GT-free identity instead.
    """

    digest = hashlib.sha256()
    digest.update(state.query.encode("utf-8"))
    for value in (
        state.t1_image,
        state.t2_image,
        state.t1_mask,
        state.t2_mask,
        state.change_mask,
    ):
        array = np.ascontiguousarray(value)
        digest.update(str(array.shape).encode("ascii"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class _TraceRegionView:
    mode: str
    evidence_region_ids: tuple[str, ...]
    evidence_records: tuple[Mapping[str, Any], ...]
    audited_region_ids: tuple[str, ...]
    diagnoses: tuple[Mapping[str, Any], ...]
    judgments: tuple[Mapping[str, Any], ...]
    expected_call_groups: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class _PlanNullReviewSelection:
    region_id: str
    prompt_version: str
    selection_branch: str


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _region_ids(value: Any) -> tuple[str, ...] | None:
    if not isinstance(value, list):
        return None
    result: list[str] = []
    for item in value:
        region_id = item.get("region_id") if isinstance(item, Mapping) else None
        if not isinstance(region_id, str) or not region_id or region_id in result:
            return None
        result.append(region_id)
    return tuple(result)


def _string_region_ids(value: Any) -> tuple[str, ...] | None:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) or not item for item in value)
        or len(value) != len(set(value))
    ):
        return None
    return tuple(value)


def _mapped_region_records(
    value: Any,
    *,
    expected_region_ids: tuple[str, ...],
) -> tuple[Mapping[str, Any], ...] | None:
    if not isinstance(value, list) or len(value) != len(expected_region_ids):
        return None
    records: list[Mapping[str, Any]] = []
    for expected, item in zip(expected_region_ids, value, strict=True):
        if not isinstance(item, Mapping) or item.get("region_id") != expected:
            return None
        records.append(item)
    return tuple(records)


def _trace_region_view(evidence: Mapping[str, Any]) -> _TraceRegionView | None:
    trace = evidence.get("stage_trace")
    if not isinstance(trace, Mapping):
        return None
    mode = trace.get("mode")
    if mode == "initial":
        evidence_key = "evidence"
        evidence_region_ids = _region_ids(trace.get(evidence_key))
        audited_region_ids = _string_region_ids(trace.get("audited_region_ids"))
        diagnosis_key = "diagnoses"
        judgment_key = "judgments"
        prefix_groups: tuple[tuple[str, str], ...] = ()
    elif mode == "candidate":
        evidence_key = "replan_evidence"
        evidence_region_ids = _region_ids(trace.get(evidence_key))
        audited_region_ids = _string_region_ids(
            trace.get("replan_audited_region_ids")
        )
        diagnosis_key = "replan_diagnoses"
        judgment_key = "replan_judgments"
        candidate_ids = _string_region_ids(trace.get("audited_region_ids"))
        candidate_evidence_ids = _region_ids(trace.get("evidence"))
        if (
            candidate_ids is None
            or candidate_evidence_ids is None
            or candidate_ids != candidate_evidence_ids
        ):
            return None
        prefix_groups = tuple(
            ("candidate_evidence", region_id) for region_id in candidate_ids
        )
    else:
        return None
    if evidence_region_ids is None or audited_region_ids is None:
        return None
    if not audited_region_ids or any(
        region_id not in evidence_region_ids for region_id in audited_region_ids
    ):
        return None
    if tuple(
        region_id for region_id in evidence_region_ids if region_id in audited_region_ids
    ) != audited_region_ids:
        return None
    evidence_records = _mapped_region_records(
        trace.get(evidence_key), expected_region_ids=evidence_region_ids
    )
    diagnoses = _mapped_region_records(
        trace.get(diagnosis_key), expected_region_ids=audited_region_ids
    )
    judgments = _mapped_region_records(
        trace.get(judgment_key), expected_region_ids=audited_region_ids
    )
    if evidence_records is None or diagnoses is None or judgments is None:
        return None
    return _TraceRegionView(
        mode=mode,
        evidence_region_ids=evidence_region_ids,
        evidence_records=evidence_records,
        audited_region_ids=audited_region_ids,
        diagnoses=diagnoses,
        judgments=judgments,
        expected_call_groups=prefix_groups
        + tuple(("audit", region_id) for region_id in audited_region_ids),
    )


def _validated_round_one_call_groups(
    evidence: Mapping[str, Any],
) -> tuple[tuple[str, str, tuple[Mapping[str, Any], ...]], ...] | None:
    view = _trace_region_view(evidence)
    calls = evidence.get("backend_calls")
    if view is None or not isinstance(calls, list):
        return None
    normalized_calls: list[Mapping[str, Any]] = []
    for call in calls:
        if not isinstance(call, Mapping):
            return None
        if not isinstance(call.get("backend"), str) or not call.get("backend"):
            return None
        if not isinstance(call.get("stage"), str) or not call.get("stage"):
            return None
        if call.get("attempt_kind") not in {"initial", "repair"}:
            return None
        retry_count = call.get("network_retry_attempt_count")
        if (
            isinstance(retry_count, bool)
            or not isinstance(retry_count, int)
            or retry_count < 0
        ):
            return None
        prompt = call.get("prompt_sha256")
        if not isinstance(prompt, str) or _SHA256_RE.fullmatch(prompt) is None:
            return None
        if "raw_response" not in call or not isinstance(
            call.get("raw_response"), str
        ):
            return None
        if (
            not isinstance(call.get("visual_input_scope"), str)
            or not call.get("visual_input_scope")
            or "terminal_review" in call
        ):
            return None
        normalized_calls.append(call)
    groups: list[tuple[str, str, tuple[Mapping[str, Any], ...]]] = []
    call_index = 0
    for expected_stage, expected_region in view.expected_call_groups:
        if call_index >= len(normalized_calls):
            return None
        initial = normalized_calls[call_index]
        if (
            initial.get("stage") != expected_stage
            or initial.get("attempt_kind") != "initial"
        ):
            return None
        group = [initial]
        call_index += 1
        if (
            call_index < len(normalized_calls)
            and normalized_calls[call_index].get("stage") == expected_stage
            and normalized_calls[call_index].get("attempt_kind") == "repair"
        ):
            group.append(normalized_calls[call_index])
            call_index += 1
        if len(group) == 2 and (
            "parsed_output" in group[0]
            or not isinstance(group[0].get("parse_error"), str)
            or not group[0].get("parse_error")
        ):
            return None
        for call_index_in_group, call in enumerate(group):
            parsed = call.get("parsed_output")
            if isinstance(parsed, Mapping) and "parse_error" in call:
                return None
            parsed_region = (
                parsed.get("region_id") if isinstance(parsed, Mapping) else None
            )
            if parsed_region is not None and parsed_region != expected_region:
                return None
            if parsed_region is None and (
                call_index_in_group == len(group) - 1
                or not isinstance(call.get("parse_error"), str)
                or not call.get("parse_error")
            ):
                return None
            if (
                call_index_in_group == len(group) - 1
                and parsed_region != expected_region
            ):
                return None
        groups.append((expected_stage, expected_region, tuple(group)))
    if call_index != len(normalized_calls):
        return None
    return tuple(groups)


def _rebuild_plan_null_after_allowed_repair(
    round_one: VerifierOutput,
    evidence: Mapping[str, Any],
    *,
    state: ChangeState | None = None,
    verifier: StagedQwenVerifier | None = None,
) -> bool:
    """Rebuild the plan-null trigger from raw staged evidence, fail closed."""

    if (
        round_one.verifier_valid is not True
        or round_one.suggested_action is not None
        or evidence.get("type") != "staged_qwen_verifier"
        or evidence.get("schema_version") != StagedQwenVerifier.SCHEMA_VERSION
        or evidence.get("verifier_valid") is not True
        or evidence.get("localization_valid") is not False
        or evidence.get("validation_errors") != []
    ):
        return False
    trace = evidence.get("stage_trace")
    if not isinstance(trace, Mapping):
        return False
    view = _trace_region_view(evidence)
    groups = _validated_round_one_call_groups(evidence)
    if view is None or groups is None or state is None or verifier is None:
        return False
    proposals = state.evidence.get("verifier_region_proposals")
    environment_region_ids = _region_ids(proposals)
    if (
        not isinstance(proposals, list)
        or environment_region_ids is None
        or environment_region_ids != view.evidence_region_ids
    ):
        return False
    try:
        records = tuple(EvidenceRecord.from_proposal(item) for item in proposals)
        if tuple(record.to_dict() for record in records) != tuple(
            dict(item) for item in view.evidence_records
        ):
            return False
        records_by_id = {record.region_id: record for record in records}
        judgments: list[EvidenceJudgment] = []
        diagnoses: list[Diagnosis] = []
        expected_judgment_keys = {
            "region_id",
            "t1_state",
            "t2_state",
            "evidence_quality",
            "change_mask_state",
            "mask_assessment",
            "evidence",
            "screening_hypothesis",
            "screening_resolution",
        }
        expected_diagnosis_keys = {
            "region_id",
            "error_type",
            "target_view",
            "audit_checklist",
            "audit_evidence",
            "summary",
        }
        for judgment_item, diagnosis_item in zip(
            view.judgments, view.diagnoses, strict=True
        ):
            if (
                set(judgment_item) != expected_judgment_keys
                or set(diagnosis_item) != expected_diagnosis_keys
            ):
                return False
            region_id = diagnosis_item.get("region_id")
            if (
                not isinstance(region_id, str)
                or judgment_item.get("region_id") != region_id
                or region_id not in records_by_id
            ):
                return False
            record = records_by_id[region_id]
            checklist_payload = diagnosis_item.get("audit_checklist")
            evidence_payload = diagnosis_item.get("audit_evidence")
            if (
                not isinstance(checklist_payload, Mapping)
                or set(checklist_payload) != set(AUDIT_CHECKS)
                or not isinstance(evidence_payload, Mapping)
                or set(evidence_payload) != set(AUDIT_CHECKS)
                or any(
                    not isinstance(checklist_payload[key], str)
                    or not isinstance(evidence_payload[key], str)
                    for key in AUDIT_CHECKS
                )
            ):
                return False
            if all(evidence_payload[key] == "" for key in AUDIT_CHECKS):
                checklist = AuditChecklist.from_mapping(checklist_payload)
            elif all(
                evidence_payload[key].strip() for key in AUDIT_CHECKS
            ):
                grounded_checklist = {
                    key: {
                        "status": checklist_payload[key],
                        "evidence": evidence_payload[key],
                    }
                    for key in AUDIT_CHECKS
                }
                checklist = AuditChecklist.from_mapping(grounded_checklist)
            else:
                return False
            diagnosis = Diagnosis(
                region_id=region_id,
                error_type=diagnosis_item["error_type"],
                target_view=diagnosis_item["target_view"],
                audit_checklist=checklist,
                summary=diagnosis_item["summary"],
            )
            if (
                not isinstance(diagnosis_item["error_type"], str)
                or (
                    diagnosis_item["target_view"] is not None
                    and not isinstance(diagnosis_item["target_view"], str)
                )
                or not isinstance(diagnosis_item["summary"], str)
                or diagnosis.to_dict() != dict(diagnosis_item)
            ):
                return False
            t1_state = judgment_item["t1_state"]
            t2_state = judgment_item["t2_state"]
            evidence_quality = judgment_item["evidence_quality"]
            judgment_evidence = judgment_item["evidence"]
            if (
                t1_state
                not in {"building", "background", "mixed", "uncertain"}
                or t2_state
                not in {"building", "background", "mixed", "uncertain"}
                or evidence_quality
                not in {"clear", "ambiguous", "insufficient"}
                or judgment_item["change_mask_state"]
                != record.change_mask_state
                or not isinstance(judgment_evidence, str)
                or not judgment_evidence.strip()
                or judgment_item["screening_hypothesis"] is not None
                or judgment_item["screening_resolution"] is not None
            ):
                return False
            judgment = EvidenceJudgment(
                region_id=region_id,
                t1_state=t1_state,
                t2_state=t2_state,
                evidence_quality=evidence_quality,
                change_mask_state=judgment_item["change_mask_state"],
                mask_assessment=judgment_item["mask_assessment"],
                evidence=judgment_evidence,
                screening_hypothesis=None,
                screening_resolution=None,
            )
            expected_assessment = {
                "none": "correct",
                "false_positive_change": "false_positive",
                "false_negative": "false_negative",
                "mixed_error": "mixed",
                "uncertain_region": "uncertain",
            }[diagnosis.error_type]
            expected_quality = {
                "pass": "clear",
                "fail": "insufficient",
                "uncertain": "ambiguous",
            }[checklist.evidence_sufficient]
            fail_closed_uncertain = bool(
                evidence_quality == "insufficient"
                and all(
                    status == "uncertain"
                    for status in checklist.to_dict().values()
                )
            )
            concrete_states = {t1_state, t2_state} <= {
                "building",
                "background",
            }
            target_transition = concrete_states and t1_state != t2_state
            if (
                judgment.to_dict() != dict(judgment_item)
                or judgment.mask_assessment != expected_assessment
                or (
                    evidence_quality != expected_quality
                    and not fail_closed_uncertain
                )
                or (
                    record.change_mask_state == "white"
                    and concrete_states
                    and not target_transition
                    and diagnosis.error_type == "none"
                )
                or (
                    record.change_mask_state == "black"
                    and target_transition
                    and diagnosis.error_type == "none"
                )
                or (
                    diagnosis.error_type == "false_negative"
                    and concrete_states
                    and not target_transition
                )
                or (
                    diagnosis.error_type
                    in {"false_positive_change", "false_negative"}
                    and not concrete_states
                )
                or diagnosis.target_view
                != _runtime_target_view(
                    record,
                    diagnosis.error_type,
                    judgment.t1_state,
                    judgment.t2_state,
                )
            ):
                return False
            verifier._validate_diagnosis(record, diagnosis)
            judgments.append(judgment)
            diagnoses.append(diagnosis)
        selected, rebuilt_plan = verifier._select_action_plan(
            state,
            records,
            tuple(diagnoses),
        )
        audited_records = tuple(
            records_by_id[region_id] for region_id in view.audited_region_ids
        )
        completion_passed, completion_reason = verifier._state_completion_gate(
            state,
            records,
            audited_records,
            tuple(judgments),
            tuple(diagnoses),
        )
        if (
            rebuilt_plan is not None
            and rebuilt_plan.action == "finish"
            and not completion_passed
        ):
            audited_ids = {record.region_id for record in audited_records}
            missing = next(
                (
                    record
                    for record in records
                    if record.region_id not in audited_ids
                ),
                records[0],
            )
            selected = Diagnosis(
                missing.region_id,
                "uncertain_region",
                None,
                _uncertain_audit_checklist(),
            )
            rebuilt_plan = None
    except (KeyError, TypeError, ValueError, StageProtocolError):
        return False
    # The serialized plan and completion fields are consistency receipts only:
    # trigger eligibility comes from the independent reconstruction above.
    if (
        rebuilt_plan is not None
        or selected is None
        or trace.get("plan") is not None
        or trace.get("state_completion_gate_passed") is not completion_passed
        or trace.get("state_completion_gate_reason") != completion_reason
    ):
        return False
    expected_target_view = selected.target_view or "t2"
    expected_accept = view.mode == "candidate"
    expected_comparison = "better" if expected_accept else "initial"
    if (
        round_one.error_type != selected.error_type
        or round_one.target_view != expected_target_view
        or round_one.error_region is not None
        or round_one.comparison != expected_comparison
        or round_one.localization_valid is not False
        or round_one.accept is not expected_accept
        or round_one.stop is not False
    ):
        return False
    regional_errors = evidence.get("regional_validation_errors")
    if not isinstance(regional_errors, list):
        return False
    repair_regions = {
        region_id
        for stage, region_id, calls in groups
        if stage == "audit" and len(calls) == 2
    }
    seen_error_regions: set[str] = set()
    for item in regional_errors:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"region_id", "error"}
            or not isinstance(item.get("region_id"), str)
            or not isinstance(item.get("error"), str)
            or not item.get("error")
        ):
            return False
        region_id = str(item["region_id"])
        if region_id in seen_error_regions or region_id not in repair_regions:
            return False
        seen_error_regions.add(region_id)
    return True


def _select_plan_null_review_region(
    state: ChangeState,
    evidence: Mapping[str, Any],
) -> _PlanNullReviewSelection | None:
    """Select by Environment order: next unreviewed, else first unresolved."""

    view = _trace_region_view(evidence)
    proposals = state.evidence.get("verifier_region_proposals")
    environment_region_ids = _region_ids(proposals)
    if view is None or environment_region_ids is None or not environment_region_ids:
        return None
    if view.evidence_region_ids != environment_region_ids:
        return None
    reviewed = set(view.audited_region_ids)
    for region_id in environment_region_ids:
        if region_id not in reviewed:
            return _PlanNullReviewSelection(
                region_id=region_id,
                prompt_version=PLAN_NULL_UNREVIEWED_PROMPT_VERSION,
                selection_branch="next_unreviewed_environment_region",
            )
    diagnosis_by_region = {
        str(item["region_id"]): item for item in view.diagnoses
    }
    regional_errors = evidence.get("regional_validation_errors")
    malformed_regions = {
        str(item.get("region_id"))
        for item in regional_errors
        if isinstance(item, Mapping) and isinstance(item.get("region_id"), str)
    } if isinstance(regional_errors, list) else set()
    for region_id in environment_region_ids:
        diagnosis = diagnosis_by_region.get(region_id)
        if region_id in malformed_regions or (
            isinstance(diagnosis, Mapping)
            and diagnosis.get("error_type") != "none"
        ):
            return _PlanNullReviewSelection(
                region_id=region_id,
                prompt_version=PLAN_NULL_ALTERNATIVE_PROMPT_VERSION,
                selection_branch="first_unresolved_environment_region",
            )
    return None


def _round_one_call_identities(
    evidence: Mapping[str, Any],
) -> tuple[BackendCallIdentity, ...]:
    groups = _validated_round_one_call_groups(evidence)
    if groups is None:
        raise ValueError("round-one call identities require a complete raw call chain")
    return tuple(
        BackendCallIdentity(
            stage=stage,
            region_id=region_id,
            prompt_sha256=str(call["prompt_sha256"]),
        )
        for stage, region_id, calls in groups
        for call in calls
    )


def _round_one_counters(
    evidence: Mapping[str, Any],
    environment_step_count: int,
    *,
    max_network_retry_attempts: int = 0,
) -> TerminalReviewCounters:
    if (
        isinstance(max_network_retry_attempts, bool)
        or not isinstance(max_network_retry_attempts, int)
        or max_network_retry_attempts < 0
    ):
        raise ValueError("round-one network retry budget is invalid")
    calls = evidence.get("backend_calls")
    if not isinstance(calls, list):
        raise ValueError("round-one evidence lacks an explicit backend_calls list")
    values: list[Mapping[str, Any]] = []
    for index, item in enumerate(calls):
        if not isinstance(item, Mapping):
            raise ValueError(f"round-one backend call {index} is malformed")
        attempt_kind = item.get("attempt_kind")
        if attempt_kind not in {"initial", "repair"}:
            raise ValueError(
                f"round-one backend call {index} lacks a frozen attempt_kind"
            )
        values.append(item)
    network_retry_attempt_count = sum(
        _network_retry_attempt_count(item, index=index)
        for index, item in enumerate(values)
    )
    if network_retry_attempt_count > max_network_retry_attempts:
        raise ValueError("round-one backend calls exceed the network retry budget")
    return TerminalReviewCounters(
        verifier_round_count=1,
        stage_initial_attempt_count=sum(
            item.get("attempt_kind") == "initial"
            for item in values
        ),
        schema_repair_attempt_count=sum(
            item.get("attempt_kind") == "repair"
            for item in values
        ),
        network_retry_attempt_count=network_retry_attempt_count,
        tool_action_attempt_count=0,
        tool_failure_count=0,
        environment_step_count=environment_step_count,
    )


def _round_two_calls(value: Any) -> tuple[RoundTwoCallAudit, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("terminal-review backend_calls must be a non-empty list")
    context_keys = {
        "round_index",
        "round_kind",
        "trigger",
        "fresh_backend_context",
        "inherits_hidden_state",
        "copies_round_one_raw_response",
        "selection_policy_version",
        "preferred_region_id",
        "prompt_version",
        "selection_branch",
        "review_region_id",
        "proposal_version",
    }
    normalized: list[dict[str, Any]] = []
    observed_triggers: set[str] = set()
    for index, call in enumerate(value):
        if not isinstance(call, Mapping):
            raise ValueError(
                f"terminal-review backend call {index} is malformed"
            )
        context = call.get("terminal_review")
        if not isinstance(context, Mapping):
            raise ValueError(
                "terminal-review backend call lacks terminal_review context"
            )
        if set(context) != context_keys:
            raise ValueError(
                "terminal-review backend call context keys differ"
            )
        region_id = context.get("review_region_id")
        if not isinstance(region_id, str) or not region_id:
            raise ValueError("terminal-review backend call lacks review_region_id")
        backend = call.get("backend")
        stage = call.get("stage")
        attempt_kind = call.get("attempt_kind")
        prompt_sha256 = call.get("prompt_sha256")
        raw_response = call.get("raw_response")
        visual_input_scope = call.get("visual_input_scope")
        if not isinstance(backend, str) or not backend:
            raise ValueError("terminal-review backend call lacks backend identity")
        if stage != "audit":
            raise ValueError("terminal-review backend call stage must be audit")
        if attempt_kind not in {"initial", "repair"}:
            raise ValueError("terminal-review backend call lacks attempt_kind")
        if (
            not isinstance(prompt_sha256, str)
            or _SHA256_RE.fullmatch(prompt_sha256) is None
        ):
            raise ValueError("terminal-review backend call prompt hash is malformed")
        if not isinstance(raw_response, str):
            raise ValueError("terminal-review backend call lacks raw_response")
        if not isinstance(visual_input_scope, str) or not visual_input_scope:
            raise ValueError(
                "terminal-review backend call lacks visual_input_scope"
            )
        if (
            type(context.get("round_index")) is not int
            or context.get("round_index") != 2
            or context.get("selection_policy_version")
            != SELECTION_POLICY_VERSION
            or context.get("fresh_backend_context") is not True
            or context.get("inherits_hidden_state") is not False
            or context.get("copies_round_one_raw_response") is not False
        ):
            raise ValueError("terminal-review backend call context is malformed")
        proposal_version = context.get("proposal_version")
        if proposal_version not in {None, EMPTY_MASK_SCAN_PROPOSAL_VERSION}:
            raise ValueError(
                "terminal-review backend call proposal_version is malformed"
            )
        if proposal_version == EMPTY_MASK_SCAN_PROPOSAL_VERSION and not region_id.startswith(
            "s"
        ):
            raise ValueError(
                "terminal-review scan proposal_version has a non-scan region"
            )
        trigger = context.get("trigger")
        if trigger == "plan_null_after_allowed_repair":
            if (
                context.get("round_kind") != "plan_null_alternative"
                or context.get("preferred_region_id") != region_id
                or _PLAN_NULL_PROMPT_VERSION_BY_BRANCH.get(
                    context.get("selection_branch")
                )
                != context.get("prompt_version")
            ):
                raise ValueError(
                    "terminal-review plan-null call context is malformed"
                )
        elif trigger == "provisional_finish":
            if (
                context.get("round_kind") != "finish_confirmation"
                or any(
                    context.get(key) is not None
                    for key in (
                        "preferred_region_id",
                        "prompt_version",
                        "selection_branch",
                    )
                )
            ):
                raise ValueError(
                    "terminal-review finish call context is malformed"
                )
        else:
            raise ValueError("terminal-review backend call trigger is malformed")
        observed_triggers.add(trigger)
        parsed_output: dict[str, Any] | None = None
        if "parsed_output" in call:
            if "parse_error" in call:
                raise ValueError(
                    "terminal-review call records both parsed output and parse error"
                )
            parsed = call.get("parsed_output")
            if (
                not isinstance(parsed, Mapping)
                or parsed.get("region_id") != region_id
            ):
                raise ValueError(
                    "terminal-review parsed output region differs"
                )
            parsed_output = dict(parsed)
        elif (
            not isinstance(call.get("parse_error"), str)
            or not call.get("parse_error")
        ):
            raise ValueError(
                "terminal-review call without parsed output lacks parse_error"
            )
        normalized.append(
            {
                "stage": stage,
                "region_id": region_id,
                "attempt_kind": StageAttemptKind(attempt_kind),
                "prompt_sha256": prompt_sha256,
                "raw_response": raw_response,
                "parsed_output": parsed_output,
                "visual_input_scope": visual_input_scope,
                "network_retry_attempt_count": _network_retry_attempt_count(
                    call, index=index
                ),
            }
        )
    if len(observed_triggers) != 1:
        raise ValueError("terminal-review backend calls mix trigger identities")

    result: list[RoundTwoCallAudit] = []
    seen_groups: set[tuple[str, str]] = set()
    call_index = 0
    while call_index < len(normalized):
        initial = normalized[call_index]
        group_key = (initial["stage"], initial["region_id"])
        if (
            initial["attempt_kind"] is not StageAttemptKind.INITIAL
            or group_key in seen_groups
        ):
            raise ValueError(
                "terminal-review call group must begin with one initial attempt"
            )
        group = [initial]
        call_index += 1
        if call_index < len(normalized):
            candidate_repair = normalized[call_index]
            if candidate_repair["attempt_kind"] is StageAttemptKind.REPAIR:
                if (
                    candidate_repair["stage"],
                    candidate_repair["region_id"],
                ) != group_key:
                    raise ValueError(
                        "terminal-review repair differs from its initial call"
                    )
                group.append(candidate_repair)
                call_index += 1
        if len(group) == 2 and group[0]["parsed_output"] is not None:
            raise ValueError(
                "terminal-review repair followed a parsed initial output"
            )
        if group[-1]["parsed_output"] is None:
            raise ValueError(
                "terminal-review call group lacks a final parsed output"
            )
        seen_groups.add(group_key)
        for item in group:
            result.append(
                RoundTwoCallAudit(
                    stage=item["stage"],
                    region_id=item["region_id"],
                    attempt_kind=item["attempt_kind"],
                    prompt_sha256=item["prompt_sha256"],
                    raw_response=item["raw_response"],
                    parsed_output=item["parsed_output"],
                    visual_input_scope=item["visual_input_scope"],
                    network_retry_attempt_count=item[
                        "network_retry_attempt_count"
                    ],
                )
            )
    if (
        observed_triggers == {"plan_null_after_allowed_repair"}
        and len(seen_groups) != 1
    ):
        raise ValueError("plan-null round two must audit exactly one region")
    return tuple(result)


def _network_retry_attempt_count(
    call: Mapping[str, Any], *, index: int
) -> int:
    """Reconstruct one call's retry count from explicit backend evidence."""

    explicit = call.get("network_retry_attempt_count")
    if (
        isinstance(explicit, bool)
        or not isinstance(explicit, int)
        or explicit < 0
    ):
        raise ValueError(
            f"backend call {index} lacks a valid explicit network retry counter"
        )
    return explicit


def _no_action_output(
    round_one: VerifierOutput,
    *,
    effective_outcome: RoundTwoOutcome | None,
    next_state: ReviewNextState | None,
) -> VerifierOutput:
    schema_valid = effective_outcome is not RoundTwoOutcome.SCHEMA_INVALID_AFTER_ALLOWED_REPAIR
    return VerifierOutput(
        quality_score=round_one.quality_score,
        score_delta=0.0,
        progress_score=0.0,
        comparison="initial",
        error_type=(
            round_one.error_type
            if round_one.error_type != "none"
            else "uncertain_region"
        ),
        target_view=round_one.target_view,
        error_region=None,
        suggested_action=None,
        feedback=(
            "Terminal review failed closed without an executable action: "
            f"outcome={effective_outcome.value if effective_outcome else 'none'}; "
            f"next_state={next_state.value if next_state else 'none'}."
        ),
        accept=False,
        verifier_valid=schema_valid,
        localization_valid=False,
        stop=False,
    )


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


__all__ = ["TerminalReviewVerifier"]
