"""Stage-oriented MLLM verifier.

This adapter deliberately keeps model responses small and typed.  It can be
used with a local Qwen model or a hosted backend implementing ``StageBackend``.
The Environment remains the authority for proposal geometry and mask
editability; the model cannot invent either.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any, Callable, Mapping, Sequence

from ..coordinates import normalized_point_to_pixel
from ..state import AgentAction, ChangeState, VerifierOutput
from ..terminal_review import (
    PLAN_NULL_ALTERNATIVE_PROMPT_VERSION,
    PLAN_NULL_UNREVIEWED_PROMPT_VERSION,
    SELECTION_POLICY_VERSION,
)
from ..verifier_regions import EMPTY_MASK_SCAN_PROPOSAL_VERSION, build_verifier_regions
from ..verifier_protocol import (
    ActionPlan,
    AuditChecklist,
    Decision,
    Diagnosis,
    EvidenceJudgment,
    EvidenceRecord,
    StageBackend,
    StageProtocolError,
    StageTrace,
    TransitionAssessment,
    is_atomic_evidence_grounding_error,
)


class StagedQwenVerifier:
    """Audit Environment proposals and resolve executable geometry in code."""

    SCHEMA_VERSION = "staged_verifier_environment_ordered_compact_audit_v12_1"

    def __init__(
        self,
        backend: StageBackend,
        *,
        max_regions: int = 6,
        max_retries: int = 2,
        visual_context: str = "hybrid",
    ):
        if max_regions < 1:
            raise ValueError("max_regions must be positive")
        if max_retries < 1:
            raise ValueError("max_retries must be positive")
        if visual_context not in {"proposal", "hybrid"}:
            raise ValueError("visual_context must be proposal or hybrid")
        self.backend = backend
        self.max_regions = max_regions
        self.max_retries = max_retries
        self.visual_context = visual_context
        self.last_evidence: dict[str, Any] = {}
        self._last_valid_output: VerifierOutput | None = None
        self._accepted_records: tuple[EvidenceRecord, ...] = ()
        self._accepted_diagnoses: tuple[Diagnosis, ...] = ()
        self._planned_diagnosis: Diagnosis | None = None
        self._planned_action: AgentAction | None = None
        self._regional_validation_errors: list[dict[str, str]] = []

    def reset(self) -> None:
        self.last_evidence = {}
        self._last_valid_output = None
        self._accepted_records = ()
        self._accepted_diagnoses = ()
        self._planned_diagnosis = None
        self._planned_action = None
        self._regional_validation_errors = []
        reset_audit = getattr(self.backend, "reset_audit", None)
        if callable(reset_audit):
            reset_audit()

    def review_accepted_state(
        self,
        state: ChangeState,
        *,
        round_index: int,
        round_kind: str,
        trigger: str,
        preferred_region_id: str | None = None,
        prompt_version: str | None = None,
        selection_branch: str | None = None,
    ) -> tuple[VerifierOutput, dict[str, Any]]:
        """Independently re-audit one accepted state for the P1 terminal review.

        This method never executes an action.  It reuses the unchanged atomic
        parser, completion gate and programmatic plan derivation, then returns a
        candidate output which still has to enter the existing Environment
        geometry/candidate safety flow.
        """

        if type(round_index) is not int or round_index != 2:
            raise StageProtocolError("terminal review must use round_index=2")
        expected_kind = {
            "provisional_finish": "finish_confirmation",
            "plan_null_after_allowed_repair": "plan_null_alternative",
        }.get(trigger)
        if round_kind != expected_kind:
            raise StageProtocolError("terminal-review trigger/kind mismatch")
        proposals = list(state.evidence.get("verifier_region_proposals", []))
        records = tuple(EvidenceRecord.from_proposal(item) for item in proposals)
        if not records:
            raise StageProtocolError("no Environment proposal is available")
        if len(records) > self.max_regions:
            raise StageProtocolError(
                f"proposal count {len(records)} exceeds "
                f"staged verifier max_regions={self.max_regions}"
            )
        selected_records = records
        if preferred_region_id and not any(
            item.region_id == preferred_region_id for item in records
        ):
            raise StageProtocolError(
                "terminal-review preferred region is absent from accepted state"
            )
        if round_kind == "plan_null_alternative":
            if not preferred_region_id or not prompt_version or not selection_branch:
                raise StageProtocolError(
                    "plan-null review requires one versioned Environment selection"
                )
            selected_records = tuple(
                item for item in records if item.region_id == preferred_region_id
            )
            if len(selected_records) != 1:
                raise StageProtocolError(
                    "plan-null review must select exactly one Environment region"
                )
            expected_prompt_version = {
                "next_unreviewed_environment_region": (
                    PLAN_NULL_UNREVIEWED_PROMPT_VERSION
                ),
                "first_unresolved_environment_region": (
                    PLAN_NULL_ALTERNATIVE_PROMPT_VERSION
                ),
            }.get(selection_branch)
            if expected_prompt_version != prompt_version:
                raise StageProtocolError(
                    "plan-null prompt branch/version differs from frozen mapping"
                )
        elif any(
            value is not None
            for value in (
                preferred_region_id,
                prompt_version,
                selection_branch,
            )
        ):
            raise StageProtocolError(
                "finish confirmation cannot carry plan-null prompt metadata"
            )
        audit_start = self._audit_length()
        validation_error_start = len(self._regional_validation_errors)
        context = {
            "round_index": 2,
            "round_kind": round_kind,
            "trigger": trigger,
            "fresh_backend_context": True,
            "inherits_hidden_state": False,
            "copies_round_one_raw_response": False,
            "selection_policy_version": (
                SELECTION_POLICY_VERSION
            ),
            "preferred_region_id": preferred_region_id,
            "prompt_version": prompt_version,
            "selection_branch": selection_branch,
        }
        audited, judgments, diagnoses = self._audit_regions_in_batches(
            state,
            selected_records,
            terminal_review_context=context,
        )
        selected, plan = self._select_action_plan(
            state, selected_records, diagnoses
        )
        completion_passed, completion_reason = self._state_completion_gate(
            state,
            records,
            audited,
            judgments,
            diagnoses,
        )
        if plan is not None and plan.action == "finish" and not completion_passed:
            plan = None
        decision = self._initial_decision(
            diagnoses,
            selected,
            plan,
            runtime_quality=_diagnosis_quality(diagnoses),
        )
        output = self._output(decision, plan, selected, None, initial=True)
        self._accepted_records = selected_records
        self._accepted_diagnoses = diagnoses
        self._planned_diagnosis = selected
        self._planned_action = (
            _plan_agent_action(plan, state.image_size)
            if plan is not None and plan.action != "finish"
            else None
        )
        safe_plan_keys: set[str] = set()
        for diagnosis in diagnoses:
            candidate_plan = self._plan(state, selected_records, diagnosis)
            if candidate_plan is None or candidate_plan.action == "finish":
                continue
            safe_plan_keys.add(
                json.dumps(
                    candidate_plan.__dict__,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        round_validation_errors = list(
            self._regional_validation_errors[validation_error_start:]
        )
        trace = StageTrace(
            mode="initial",
            evidence=selected_records,
            audited_region_ids=tuple(item.region_id for item in audited),
            judgments=judgments,
            diagnoses=diagnoses,
            plan=plan,
            decision=decision,
            transition_assessment=None,
            state_completion_gate_passed=completion_passed,
            state_completion_gate_reason=completion_reason,
        )
        return output, {
            **context,
            "preferred_region_id": preferred_region_id,
            "stage_trace": trace.to_dict(),
            "backend_calls": self._backend_calls_since(audit_start),
            "safe_plan_candidate_count": len(safe_plan_keys),
            "schema_valid": not round_validation_errors,
            "regional_validation_errors": round_validation_errors,
            "completion_gate_passed": completion_passed,
            "completion_gate_reason": completion_reason,
        }

    def on_candidate_rejected(self, previous_feedback: VerifierOutput) -> None:
        self._last_valid_output = (
            previous_feedback if previous_feedback.verifier_valid else None
        )

    def replan_after_rejection(
        self,
        accepted_state: ChangeState,
        rejected_candidate: ChangeState,
        accepted_feedback: VerifierOutput,
        rejected_feedback: VerifierOutput,
        rejected_action: AgentAction,
        rejection_reasons: Sequence[str],
        rejection_history: Sequence[Mapping[str, Any]],
    ) -> VerifierOutput:
        """Choose a different already-diagnosed region after rollback."""

        rejected = [rejected_action]
        for item in rejection_history:
            action = item.get("action")
            if not isinstance(action, Mapping):
                continue
            try:
                rejected.append(
                    AgentAction(
                        str(action["target_view"]),
                        str(action["action"]),
                        coordinate=(
                            tuple(int(value) for value in action["coordinate"])
                            if action.get("coordinate") is not None
                            else None
                        ),
                        box=(
                            tuple(int(value) for value in action["box"])
                            if action.get("box") is not None
                            else None
                        ),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        candidates = tuple(
            diagnosis
            for diagnosis in self._accepted_diagnoses
            if diagnosis.error_type
            in {"false_positive_change", "false_negative"}
        )
        for diagnosis in candidates:
            plan = self._plan(accepted_state, self._accepted_records, diagnosis)
            if plan is None or plan.action == "finish":
                continue
            action = _plan_agent_action(plan, accepted_state.image_size)
            if action in rejected:
                continue
            output = VerifierOutput(
                quality_score=accepted_feedback.quality_score,
                progress_score=0.0,
                score_delta=0.0,
                comparison="uncertain",
                error_type=diagnosis.error_type,
                target_view=plan.target_view or "t2",
                error_region=_plan_region(plan),
                suggested_action=plan.action,
                feedback=(
                    f"Rollback excluded failed geometry; try alternate region "
                    f"{diagnosis.region_id}."
                ),
                accept=False,
                verifier_valid=True,
                localization_valid=True,
                stop=False,
            )
            # Candidate verification must receive the diagnosis that authorized
            # this replacement action, not the rejected region's stale diagnosis.
            self._planned_diagnosis = diagnosis
            self._planned_action = action
            self.last_evidence = {
                "type": "staged_qwen_verifier",
                "schema_version": self.SCHEMA_VERSION,
                "decision_mode": "programmatic_alternate_region_after_rollback",
                "verifier_valid": True,
                "localization_valid": True,
                "replan": {
                    "selected_region_id": diagnosis.region_id,
                    "rejected_action_count": len(rejected),
                    "rejection_reasons": list(rejection_reasons),
                },
                "validation_errors": [],
                "backend_calls": [],
            }
            return output
        self._planned_diagnosis = None
        self._planned_action = None
        output = VerifierOutput(
            quality_score=accepted_feedback.quality_score,
            progress_score=0.0,
            score_delta=0.0,
            comparison="uncertain",
            error_type=accepted_feedback.error_type,
            target_view=accepted_feedback.target_view,
            error_region=None,
            suggested_action=None,
            feedback="No untried safe diagnosed region remains after rollback.",
            accept=False,
            verifier_valid=True,
            localization_valid=False,
            stop=False,
        )
        self.last_evidence = {
            "type": "staged_qwen_verifier",
            "schema_version": self.SCHEMA_VERSION,
            "decision_mode": "programmatic_no_alternate_region_after_rollback",
            "verifier_valid": True,
            "localization_valid": False,
            "replan": {
                "selected_region_id": None,
                "rejected_action_count": len(rejected),
                "rejection_reasons": list(rejection_reasons),
            },
            "validation_errors": [],
            "backend_calls": [],
        }
        return output

    def verify(
        self,
        state: ChangeState,
        previous_score: float | None,
        previous_action: AgentAction | None,
        previous_state: ChangeState | None = None,
    ) -> VerifierOutput:
        audit_start = self._audit_length()
        self._regional_validation_errors = []
        try:
            if previous_state is not None and self._states_identical(
                state, previous_state
            ):
                previous = self._last_valid_output
                finish_authorized = bool(
                    previous
                    and previous.accept
                    and previous.error_type == "none"
                    and previous.suggested_action == "finish"
                )
                output = VerifierOutput(
                    quality_score=previous.quality_score if previous else previous_score,
                    progress_score=0.0,
                    score_delta=0.0,
                    comparison="unchanged",
                    error_type=previous.error_type if previous else "uncertain_region",
                    target_view=previous.target_view if previous else "t2",
                    error_region=previous.error_region if previous else None,
                    suggested_action="finish" if finish_authorized else None,
                    feedback="Candidate masks are identical to the accepted state.",
                    accept=finish_authorized,
                    verifier_valid=True,
                    localization_valid=finish_authorized,
                    stop=finish_authorized,
                )
                self.last_evidence = {
                    "type": "staged_qwen_verifier",
                    "schema_version": self.SCHEMA_VERSION,
                    "decision_mode": "programmatic_identical_state",
                    "verifier_valid": True,
                    "localization_valid": output.localization_valid,
                    "validation_errors": [],
                    "backend_calls": self._backend_calls_since(audit_start),
                }
                return output
            proposals = list(state.evidence.get("verifier_region_proposals", []))
            records = tuple(EvidenceRecord.from_proposal(item) for item in proposals)
            if not records:
                raise StageProtocolError("no Environment proposal is available")
            if len(records) > self.max_regions:
                raise StageProtocolError(
                    f"proposal count {len(records)} exceeds "
                    f"staged verifier max_regions={self.max_regions}"
                )
            if previous_state is None:
                output, trace = self._verify_initial(state, records, previous_score)
            else:
                output, trace = self._verify_candidate(
                    state,
                    previous_state,
                    records,
                    previous_score,
                    previous_action,
                )
            self.last_evidence = {
                "type": "staged_qwen_verifier",
                "schema_version": self.SCHEMA_VERSION,
                "verifier_valid": output.verifier_valid,
                "localization_valid": output.localization_valid,
                "stage_trace": trace.to_dict(),
                "validation_errors": [],
                "regional_validation_errors": list(
                    self._regional_validation_errors
                ),
                "backend_calls": self._backend_calls_since(audit_start),
            }
            if output.verifier_valid:
                self._last_valid_output = output
            return output
        except (KeyError, TypeError, ValueError, StageProtocolError) as error:
            self.last_evidence = {
                "type": "staged_qwen_verifier",
                "schema_version": self.SCHEMA_VERSION,
                "verifier_valid": False,
                "localization_valid": False,
                "validation_errors": [str(error)],
                "backend_calls": self._backend_calls_since(audit_start),
            }
            previous = self._last_valid_output
            return VerifierOutput(
                quality_score=previous.quality_score if previous else None,
                progress_score=0.0,
                comparison="uncertain",
                error_type=previous.error_type if previous else "uncertain_region",
                target_view=(
                    previous.target_view
                    if previous
                    else previous_action.target_view if previous_action else "t2"
                ),
                error_region=previous.error_region if previous else None,
                suggested_action=None,
                feedback="Staged verifier output was invalid; no action is authorized.",
                accept=False,
                verifier_valid=False,
                localization_valid=False,
                stop=False,
            )

    def _audit_length(self) -> int:
        history = getattr(self.backend, "call_history", None)
        return len(history) if isinstance(history, list) else 0

    def _backend_calls_since(self, start: int) -> list[dict[str, Any]]:
        history = getattr(self.backend, "call_history", None)
        if not isinstance(history, list):
            return []
        return [dict(item) for item in history[start:] if isinstance(item, Mapping)]

    def _stage_completion_audit_since(
        self,
        start: int,
        stage: str,
    ) -> dict[str, Any] | None:
        """Return only this attempt's final response-bearing audit entry."""

        history = getattr(self.backend, "call_history", None)
        if not isinstance(history, list):
            return None
        for item in reversed(history[start:]):
            if (
                isinstance(item, dict)
                and item.get("stage") == stage
                and "raw_response" in item
                and "network_error" not in item
                and "provider_error" not in item
            ):
                return item
        return None

    def _record_typed_parse_failure(
        self,
        start: int,
        stage: str,
        error: str,
    ) -> None:
        audit = self._stage_completion_audit_since(start, stage)
        if audit is None:
            return
        audit.pop("parsed_output", None)
        audit["parse_error"] = error.strip() or (
            f"{stage} typed parser rejected the extracted JSON"
        )

    def _record_typed_parse_success(
        self,
        start: int,
        stage: str,
        payload: Any,
    ) -> None:
        audit = self._stage_completion_audit_since(start, stage)
        if audit is None:
            return
        audit.pop("parse_error", None)
        audit["parsed_output"] = copy.deepcopy(
            dict(payload) if isinstance(payload, Mapping) else payload
        )

    def _run_stage(
        self,
        stage: str,
        state: ChangeState,
        payload: Mapping[str, Any],
        parser: Callable[[Mapping[str, Any]], Any],
        previous_state: ChangeState | None = None,
    ) -> Any:
        """Generate, validate, and repair one stage without weakening its schema."""

        errors: list[str] = []
        repair = getattr(self.backend, "repair_stage", None)
        atomic_evidence_anchor: Mapping[str, Any] | None = None
        for attempt in range(self.max_retries):
            raw: Any = None
            backend_returned = False
            attempt_audit_start = self._audit_length()
            try:
                attempt_payload = dict(payload)
                evidence_only_repair = bool(
                    attempt > 0
                    and atomic_evidence_anchor is not None
                )
                if evidence_only_repair:
                    attempt_payload["repair_context"] = {
                        "mode": "atomic_evidence_only",
                        "origin_validation_error": errors[0],
                        "preserve_fields": [
                            "region_id",
                            "visual_judgment.t1_state",
                            "visual_judgment.t2_state",
                            "diagnosis.audit_checklist",
                        ],
                    }
                if attempt == 0:
                    raw = self.backend.generate_stage(
                        stage, state, attempt_payload, previous_state
                    )
                elif callable(repair):
                    raw = repair(
                        stage,
                        state,
                        attempt_payload,
                        errors[-1],
                        previous_state,
                    )
                else:
                    break
                backend_returned = True
                if evidence_only_repair:
                    raw = _merge_atomic_evidence_repair(
                        atomic_evidence_anchor,
                        raw,
                    )
                parsed = parser(raw)
                self._record_typed_parse_success(
                    attempt_audit_start,
                    stage,
                    raw,
                )
                return parsed
            except (KeyError, TypeError, ValueError, StageProtocolError) as error:
                error_text = str(error).strip() or (
                    f"{stage} typed parser rejected the extracted JSON"
                )
                if backend_returned:
                    self._record_typed_parse_failure(
                        attempt_audit_start,
                        stage,
                        error_text,
                    )
                if (
                    attempt == 0
                    and stage == "audit"
                    and is_atomic_evidence_grounding_error(error_text)
                ):
                    atomic_evidence_anchor = _validated_atomic_evidence_anchor(
                        raw,
                        parser,
                    )
                errors.append(error_text)
        if not errors:
            raise StageProtocolError(f"{stage} backend does not support repair retries")
        raise StageProtocolError(
            f"{stage} failed after {len(errors)} attempt(s): {errors[-1]}"
        )

    @staticmethod
    def _states_identical(state: ChangeState, previous_state: ChangeState) -> bool:
        return (
            (state.t1_mask == previous_state.t1_mask).all()
            and (state.t2_mask == previous_state.t2_mask).all()
            and (state.change_mask == previous_state.change_mask).all()
        )

    def _verify_initial(
        self,
        state: ChangeState,
        records: tuple[EvidenceRecord, ...],
        previous_score: float | None,
    ) -> tuple[VerifierOutput, StageTrace]:
        audited_records, judgments, diagnoses = self._audit_regions_in_batches(
            state, records
        )
        selected, plan = self._select_action_plan(state, records, diagnoses)
        completion_passed, completion_reason = self._state_completion_gate(
            state, records, audited_records, judgments, diagnoses
        )
        if plan is not None and plan.action == "finish" and not completion_passed:
            missing = next(
                (
                    item
                    for item in records
                    if item.region_id
                    not in {audited.region_id for audited in audited_records}
                ),
                records[0],
            )
            selected = Diagnosis(
                missing.region_id,
                "uncertain_region",
                None,
                _uncertain_audit_checklist(),
            )
            plan = None
        decision = self._initial_decision(
            diagnoses,
            selected,
            plan,
            runtime_quality=_diagnosis_quality(diagnoses),
        )
        if decision.comparison != "initial":
            raise StageProtocolError("initial decision must use comparison=initial")
        if not completion_passed and plan is None:
            decision = Decision(
                decision.comparison,
                decision.quality_score,
                decision.progress_score,
                decision.accept,
                decision.stop,
                f"{decision.feedback}; completion gate: {completion_reason}",
            )
        output = self._output(decision, plan, selected, previous_score, initial=True)
        self._accepted_records = records
        self._accepted_diagnoses = diagnoses
        self._planned_diagnosis = selected
        self._planned_action = (
            _plan_agent_action(plan, state.image_size)
            if plan is not None and plan.action != "finish"
            else None
        )
        return output, StageTrace(
            mode="initial",
            evidence=records,
            audited_region_ids=tuple(item.region_id for item in audited_records),
            judgments=judgments,
            diagnoses=diagnoses,
            plan=plan,
            decision=decision,
            transition_assessment=None,
            state_completion_gate_passed=completion_passed,
            state_completion_gate_reason=completion_reason,
        )

    def _verify_candidate(
        self,
        state: ChangeState,
        previous_state: ChangeState,
        records: tuple[EvidenceRecord, ...],
        previous_score: float | None,
        previous_action: AgentAction | None,
    ) -> tuple[VerifierOutput, StageTrace]:
        self._validate_attempted_action(previous_state, previous_action)
        candidate_audited_records: tuple[EvidenceRecord, ...] = ()
        if self._candidate_action_matches_persisted_plan(previous_action):
            judgments = self._inspect_candidate_evidence(
                state,
                records,
                previous_state=previous_state,
                proposal_catalog=records,
                previous_action=previous_action,
            )
            candidate_audited_records = records
        else:
            # The persisted exact plan is authoritative.  Reject stale,
            # cross-process, or substituted actions before spending an MLLM call.
            judgments = ()
        transition_assessment = self._assess_candidate_transition(
            records, judgments, previous_action
        )
        decision = self._candidate_decision(transition_assessment, previous_score)
        plan: ActionPlan | None = None
        replan_records: tuple[EvidenceRecord, ...] = ()
        replan_audited_records: tuple[EvidenceRecord, ...] = ()
        replan_judgments: tuple[EvidenceJudgment, ...] = ()
        replan_diagnoses: tuple[Diagnosis, ...] = ()
        completion_passed: bool | None = None
        completion_reason: str | None = None
        output_diagnosis = self._select_diagnosis(self._accepted_diagnoses)
        if decision.accept and decision.comparison == "better":
            full_proposals = state.evidence.get(
                "verifier_next_live_region_proposals"
            )
            if not isinstance(full_proposals, list):
                proposal_config = state.evidence.get(
                    "verifier_mask_facts", {}
                ).get("proposal_config", {})
                full_proposals = build_verifier_regions(
                    state,
                    None,
                    max_regions=self.max_regions,
                    padding_ratio=float(
                        proposal_config.get("padding_ratio", 0.25)
                    ),
                )
            replan_records = tuple(
                EvidenceRecord.from_proposal(item) for item in full_proposals
            )
            if replan_records:
                (
                    replan_audited_records,
                    replan_judgments,
                    replan_diagnoses,
                ) = self._audit_regions_in_batches(
                    state, replan_records
                )
                remaining, plan = self._select_action_plan(
                    state, replan_records, replan_diagnoses
                )
                completion_passed, completion_reason = self._state_completion_gate(
                    state,
                    replan_records,
                    replan_audited_records,
                    replan_judgments,
                    replan_diagnoses,
                )
                if (
                    plan is not None
                    and plan.action == "finish"
                    and not completion_passed
                ):
                    missing = next(
                        (
                            item
                            for item in replan_records
                            if item.region_id
                            not in {
                                audited.region_id
                                for audited in replan_audited_records
                            }
                        ),
                        replan_records[0],
                    )
                    remaining = Diagnosis(
                        missing.region_id,
                        "uncertain_region",
                        None,
                        _uncertain_audit_checklist(),
                    )
                    plan = None
                output_diagnosis = remaining
            else:
                completion_passed, completion_reason = self._state_completion_gate(
                    state, (), (), (), ()
                )
                output_diagnosis = None
                if completion_passed:
                    plan = ActionPlan(None, "finish", None)
            self._accepted_records = replan_records
            self._accepted_diagnoses = replan_diagnoses
            self._planned_diagnosis = output_diagnosis
            self._planned_action = (
                _plan_agent_action(plan, state.image_size)
                if plan is not None and plan.action != "finish"
                else None
            )
        output = self._output(
            decision, plan, output_diagnosis, previous_score, initial=False
        )
        return output, StageTrace(
            mode="candidate",
            evidence=candidate_audited_records,
            audited_region_ids=tuple(
                item.region_id for item in candidate_audited_records
            ),
            judgments=judgments,
            diagnoses=(),
            plan=plan,
            decision=decision,
            transition_assessment=transition_assessment,
            state_completion_gate_passed=completion_passed,
            state_completion_gate_reason=completion_reason,
            replan_evidence=replan_records,
            replan_audited_region_ids=tuple(
                item.region_id for item in replan_audited_records
            ),
            replan_judgments=replan_judgments,
            replan_diagnoses=replan_diagnoses,
        )

    def _audit_regions_in_batches(
        self,
        state: ChangeState,
        records: tuple[EvidenceRecord, ...],
        *,
        terminal_review_context: Mapping[str, Any] | None = None,
    ) -> tuple[
        tuple[EvidenceRecord, ...],
        tuple[EvidenceJudgment, ...],
        tuple[Diagnosis, ...],
    ]:
        """Audit every region atomically in authoritative Environment order.

        Every Environment region is required by the completion gate, so a model
        selection loop can only reorder the same exhaustive audit set.  Keeping
        the Environment order removes that no-op generation stage and its
        derived screening-resolution consistency constraint without reducing
        coverage.
        """

        judgments: list[EvidenceJudgment] = []
        diagnoses: list[Diagnosis] = []
        for record in records:
            record_payload = record.to_dict()
            proposal_version = None
            if terminal_review_context is not None:
                proposal_version = record_payload.get("proposal_version")
                if record.audit_kind == "scan" and (
                    proposal_version != EMPTY_MASK_SCAN_PROPOSAL_VERSION
                ):
                    raise StageProtocolError(
                        "terminal-review scan proposal_version differs from "
                        "frozen contract"
                    )
                if record.audit_kind != "scan" and proposal_version is not None:
                    raise StageProtocolError(
                        "terminal-review proposal_version is registered only "
                        "for synthetic scan regions"
                    )
            else:
                # H0/ordinary verifier prompt bytes are a frozen control.  The
                # proposal version remains available to H1 raw trace replay but
                # must not perturb legacy model input.
                record_payload.pop("proposal_version", None)
            payload: dict[str, Any] = {
                "target_class": state.query,
                "region": record_payload,
                "proposal_catalog": [
                    {
                        key: value
                        for key, value in item.to_dict().items()
                        if terminal_review_context is not None
                        or key != "proposal_version"
                    }
                    for item in records
                ],
                "schema": "environment_ordered_compact_region_audit_v12_1",
                "visual_context": self.visual_context,
            }
            call_start = self._audit_length()
            if terminal_review_context is not None:
                payload["terminal_review"] = {
                    **dict(terminal_review_context),
                    "review_region_id": record.region_id,
                    "proposal_version": proposal_version,
                }
            try:
                judgment, diagnosis = self._run_stage(
                    "audit",
                    state,
                    payload,
                    lambda response, current_record=record: _parse_atomic_audit(
                        response, current_record
                    ),
                    None,
                )
            except StageProtocolError as error:
                error_text = str(error)
                self._regional_validation_errors.append(
                    {"region_id": record.region_id, "error": error_text}
                )
                judgment = EvidenceJudgment(
                    record.region_id,
                    "uncertain",
                    "uncertain",
                    "insufficient",
                    record.change_mask_state,
                    "uncertain",
                    f"Atomic audit failed closed: {error_text}",
                )
                diagnosis = Diagnosis(
                    record.region_id,
                    "uncertain_region",
                    None,
                    _uncertain_audit_checklist(),
                    f"Environment-ordered local audit failed closed: {error_text}",
                )
            finally:
                if terminal_review_context is not None:
                    history = getattr(self.backend, "call_history", None)
                    if isinstance(history, list):
                        for call in history[call_start:]:
                            if isinstance(call, dict):
                                call["terminal_review"] = {
                                    **dict(terminal_review_context),
                                    "review_region_id": record.region_id,
                                    "proposal_version": proposal_version,
                                }
            self._validate_diagnosis(record, diagnosis)
            judgments.append(judgment)
            diagnoses.append(diagnosis)
        return records, tuple(judgments), tuple(diagnoses)

    def _inspect_candidate_evidence(
        self,
        state: ChangeState,
        records: tuple[EvidenceRecord, ...],
        *,
        previous_state: ChangeState,
        proposal_catalog: tuple[EvidenceRecord, ...],
        previous_action: AgentAction | None,
    ) -> tuple[EvidenceJudgment, ...]:
        """Inspect candidate deltas without reclassifying black/white state errors."""

        judgments: list[EvidenceJudgment] = []
        for record in records[: self.max_regions]:
            judgment = self._run_stage(
                "candidate_evidence",
                state,
                {
                    "target_class": state.query,
                    "region": record.to_dict(),
                    "proposal_catalog": [
                        item.to_dict() for item in proposal_catalog
                    ],
                    "schema": "candidate_transition_evidence_v1",
                    "visual_context": self.visual_context,
                    "attempted_action": (
                        previous_action.to_dict() if previous_action else None
                    ),
                    "persisted_initial_diagnosis": (
                        self._planned_diagnosis.to_dict()
                        if self._planned_diagnosis is not None
                        else None
                    ),
                },
                lambda response, region_id=record.region_id: _parse_judgment(
                    response, region_id
                ),
                previous_state,
            )
            judgments.append(judgment)
        return tuple(judgments)

    @staticmethod
    def _validate_attempted_action(
        previous_state: ChangeState,
        previous_action: AgentAction | None,
    ) -> None:
        """Recheck point editability against the accepted pre-action state."""

        if previous_action is None:
            raise StageProtocolError("candidate transition requires an attempted action")
        if previous_action.action not in {"positive_point", "negative_point"}:
            return
        if previous_action.coordinate is None:
            raise StageProtocolError("point candidate requires an attempted coordinate")
        x, y = previous_action.coordinate
        width, height = previous_state.image_size
        if not (0 <= x < width and 0 <= y < height):
            raise StageProtocolError("attempted point is outside the accepted state")
        mask = (
            previous_state.t1_mask
            if previous_action.target_view == "t1"
            else previous_state.t2_mask
        )
        seed_white = bool(mask[y, x])
        if previous_action.action == "negative_point" and not seed_white:
            raise StageProtocolError(
                "negative point requires a white seed in the accepted target mask"
            )
        if previous_action.action == "positive_point" and seed_white:
            raise StageProtocolError(
                "positive point requires a black seed in the accepted target mask"
            )

    def _state_completion_gate(
        self,
        state: ChangeState,
        records: tuple[EvidenceRecord, ...],
        audited_records: tuple[EvidenceRecord, ...],
        judgments: tuple[EvidenceJudgment, ...],
        diagnoses: tuple[Diagnosis, ...],
    ) -> tuple[bool, str]:
        """Authorize finish only after every Environment audit region was inspected."""

        facts = state.evidence.get("verifier_next_live_mask_facts")
        if not isinstance(facts, Mapping):
            facts = state.evidence.get("verifier_mask_facts")
        if not isinstance(facts, Mapping):
            return False, "environment audit coverage facts are missing"
        uncovered_pixels = facts.get("initial_audit_uncovered_pixels")
        if (
            isinstance(uncovered_pixels, bool)
            or not isinstance(uncovered_pixels, int)
            or uncovered_pixels < 0
        ):
            return False, "environment audit uncovered-pixel fact is invalid"
        if uncovered_pixels > 0:
            return False, "environment audit mask has uncovered pixels"
        record_ids = {item.region_id for item in records}
        audited_ids = {item.region_id for item in audited_records}
        if audited_ids != record_ids:
            return False, (
                f"audited {len(audited_ids)} of {len(record_ids)} region(s); "
                "unaudited regions are not evidence of correctness"
            )
        judgment_ids = {item.region_id for item in judgments}
        diagnosis_ids = {item.region_id for item in diagnoses}
        if judgment_ids != record_ids or diagnosis_ids != record_ids:
            return False, "not every audited region has a judgment and diagnosis"
        if any(not self._judgment_is_sufficient(item) for item in judgments):
            return False, "at least one region lacks sufficient visual evidence"
        if any(item.error_type != "none" for item in diagnoses):
            return False, "at least one diagnosed error remains"
        return True, "all Environment audit regions are covered and diagnosed none"

    @staticmethod
    def _validate_diagnosis(
        record: EvidenceRecord,
        diagnosis: Diagnosis,
    ) -> None:
        if diagnosis.error_type == "false_negative" and record.change_mask_state != "black":
            raise StageProtocolError("false_negative requires a black/missing change region")
        if (
            diagnosis.error_type == "false_positive_change"
            and record.change_mask_state != "white"
        ):
            raise StageProtocolError("false_positive_change requires a white change region")
        # White/black proposal polarity is a structural invariant.  It is not a
        # semantic correctness proof: one component can contain both a real
        # temporal change and unsupported boundary/interior pixels.  Qwen must
        # therefore retain authority to emit false_positive_change, mixed_error,
        # or none after inspecting RGB and mask coverage.

    def _judgment_is_sufficient(self, judgment: EvidenceJudgment) -> bool:
        return bool(
            judgment.evidence_quality == "clear"
            and judgment.t1_state in {"building", "background"}
            and judgment.t2_state in {"building", "background"}
        )

    def _candidate_action_matches_persisted_plan(
        self, previous_action: AgentAction | None
    ) -> bool:
        diagnosis = self._planned_diagnosis
        planned_action = self._planned_action
        if diagnosis is None or planned_action is None or previous_action is None:
            return False
        allowed_actions = {
            "false_negative": {"positive_point", "box"},
            "false_positive_change": {"negative_point"},
        }.get(diagnosis.error_type, set())
        geometry_valid = bool(
            (
                planned_action.action in {"positive_point", "negative_point"}
                and planned_action.coordinate is not None
                and planned_action.box is None
            )
            or (
                planned_action.action == "box"
                and planned_action.coordinate is None
                and planned_action.box is not None
            )
        )
        return bool(
            planned_action.action in allowed_actions
            and planned_action.target_view in {"t1", "t2"}
            and planned_action.target_view == diagnosis.target_view
            and geometry_valid
            and previous_action == planned_action
        )

    def _assess_candidate_transition(
        self,
        records: tuple[EvidenceRecord, ...],
        judgments: tuple[EvidenceJudgment, ...],
        previous_action: AgentAction | None,
    ) -> TransitionAssessment:
        """Derive benefit/harm from action direction plus local RGB judgments."""

        attempted_action = previous_action.action if previous_action else None
        expected_kind = {
            "negative_point": "delta_removed",
            "positive_point": "delta_added",
        }.get(attempted_action)
        if attempted_action == "box" and self._planned_diagnosis is not None:
            expected_kind = {
                "false_negative": "delta_added",
                "false_positive_change": "delta_removed",
            }.get(self._planned_diagnosis.error_type)
        planned_kind = (
            {
                "false_negative": "delta_added",
                "false_positive_change": "delta_removed",
            }.get(self._planned_diagnosis.error_type)
            if self._planned_diagnosis is not None
            else None
        )
        planned_action_consistent = bool(
            expected_kind == planned_kind
            and self._candidate_action_matches_persisted_plan(previous_action)
        )
        by_id = {item.region_id: item for item in judgments}
        expected_added_states: tuple[str, str] | None = None
        if expected_kind == "delta_added" and previous_action is not None:
            expected_added_states = (
                ("building", "background")
                if previous_action.target_view == "t1"
                else ("background", "building")
            )
        sufficient = (
            bool(records)
            and expected_kind is not None
            and planned_action_consistent
        )
        intended_improved = False
        introduced_false_positive = False
        introduced_false_negative = False
        boundary_or_artifact_worsened = False
        evidence_parts: list[str] = []
        if not planned_action_consistent:
            evidence_parts.append("attempted_action_mismatches_persisted_plan")
        for record in records:
            judgment = by_id.get(record.region_id)
            if judgment is None or not self._judgment_is_sufficient(judgment):
                sufficient = False
                evidence_parts.append(
                    f"{record.region_id}:{record.audit_kind}:insufficient_visual_evidence"
                )
                continue
            real_target_change = judgment.t1_state != judgment.t2_state
            evidence_parts.append(
                f"{record.region_id}:{record.audit_kind}:"
                f"{judgment.t1_state}->{judgment.t2_state}:"
                f"evidence_quality={judgment.evidence_quality}"
            )
            if record.audit_kind == "delta_removed":
                if real_target_change:
                    introduced_false_negative = True
                elif expected_kind == "delta_removed" and planned_action_consistent:
                    intended_improved = True
                else:
                    boundary_or_artifact_worsened = True
            elif record.audit_kind == "delta_added":
                if not real_target_change:
                    introduced_false_positive = True
                elif (
                    expected_kind == "delta_added"
                    and planned_action_consistent
                    and (judgment.t1_state, judgment.t2_state)
                    == expected_added_states
                ):
                    intended_improved = True
                elif expected_kind == "delta_added":
                    if planned_action_consistent:
                        introduced_false_positive = True
                        evidence_parts.append(
                            f"{record.region_id}:target_transition_opposes_"
                            f"{previous_action.target_view if previous_action else 'unknown'}"
                        )
                else:
                    boundary_or_artifact_worsened = True
            else:
                sufficient = False
                evidence_parts.append(f"{record.region_id}:unsupported_delta_polarity")
            if record.audit_kind != expected_kind:
                sufficient = False
                boundary_or_artifact_worsened = True
                evidence_parts.append(
                    f"{record.region_id}:unexpected_for_{previous_action.action if previous_action else 'none'}"
                )
        if expected_kind is None:
            evidence_parts.append("unsupported_or_missing_attempted_action")
        return TransitionAssessment(
            intended_error_improved=intended_improved,
            introduced_false_positive=introduced_false_positive,
            introduced_false_negative=introduced_false_negative,
            boundary_or_artifact_worsened=boundary_or_artifact_worsened,
            evidence_sufficient=sufficient,
            evidence="; ".join(evidence_parts),
        )

    def _select_diagnosis(
        self, diagnoses: tuple[Diagnosis, ...]
    ) -> Diagnosis | None:
        actionable = [item for item in diagnoses if item.error_type != "none"]
        if not actionable:
            return None
        safe_priority = {
            "false_positive_change": 2,
            "false_negative": 2,
            "mixed_error": 1,
            "uncertain_region": 0,
        }
        return max(
            actionable,
            key=lambda item: safe_priority[item.error_type],
        )

    def _select_action_plan(
        self,
        state: ChangeState,
        records: tuple[EvidenceRecord, ...],
        diagnoses: tuple[Diagnosis, ...],
    ) -> tuple[Diagnosis | None, ActionPlan | None]:
        """Prefer the highest-priority diagnosis that maps to a safe action.

        A model diagnosis can be semantically plausible yet name a target view
        whose Environment seed is not editable.  Do not let that single invalid
        target suppress another independently diagnosed, executable region.
        """

        selected = self._select_diagnosis(diagnoses)
        if selected is None:
            return None, ActionPlan(None, "finish", None)
        safe_priority = {
            "false_positive_change": 2,
            "false_negative": 2,
            "mixed_error": 1,
            "uncertain_region": 0,
        }
        records_by_id = {item.region_id: item for item in records}
        ranked = sorted(
            (item for item in diagnoses if item.error_type != "none"),
            key=lambda item: (
                -safe_priority[item.error_type],
                records_by_id[item.region_id].component_area,
            ),
        )
        for diagnosis in ranked:
            plan = self._plan(state, records, diagnosis)
            if plan is not None:
                return diagnosis, plan
        return selected, None

    def _plan(
        self,
        state: ChangeState,
        records: tuple[EvidenceRecord, ...],
        diagnosis: Diagnosis | None,
    ) -> ActionPlan | None:
        if diagnosis is None or diagnosis.error_type == "none":
            return ActionPlan(None, "finish", None)
        record = next(item for item in records if item.region_id == diagnosis.region_id)
        if diagnosis.error_type not in {
            "false_positive_change",
            "false_negative",
        }:
            return None
        if record.audit_kind == "scan":
            if (
                diagnosis.error_type != "false_negative"
                or diagnosis.target_view not in {"t1", "t2"}
            ):
                return None
            return ActionPlan(
                record.region_id,
                "box",
                diagnosis.target_view,
                box_normalized_1000=record.box_normalized_1000,
            )
        action = (
            "negative_point"
            if diagnosis.error_type == "false_positive_change"
            else "positive_point"
        )
        required_seed_white = action == "negative_point"
        target_view = diagnosis.target_view
        editable_views = [
            view
            for view in ("t1", "t2")
            if record.editable_seed_white.get(view) is required_seed_white
        ]
        if len(editable_views) != 1 or target_view != editable_views[0]:
            return None
        return ActionPlan(
            record.region_id,
            action,
            target_view,
            coordinate_normalized_1000=record.component_seed_normalized_1000,
        )

    @staticmethod
    def _initial_decision(
        diagnoses: tuple[Diagnosis, ...],
        selected: Diagnosis | None,
        plan: ActionPlan | None,
        *,
        runtime_quality: float,
    ) -> Decision:
        if selected is not None and selected.summary:
            feedback = selected.summary
        elif plan is not None and plan.action == "finish":
            feedback = "All Environment audit regions passed the compact checklist."
        elif any(item.error_type == "uncertain_region" for item in diagnoses):
            feedback = "At least one Environment audit region remains uncertain."
        else:
            feedback = "No safe executable correction was derived from the compact audit."
        return Decision("initial", runtime_quality, 0.0, False, False, feedback)

    @staticmethod
    def _candidate_decision(
        assessment: TransitionAssessment,
        previous_score: float | None,
    ) -> Decision:
        comparison = assessment.comparison
        progress = {
            "better": 1.0,
            "worse": -1.0,
            "unchanged": 0.0,
            "uncertain": 0.0,
        }[comparison]
        return Decision(
            comparison,
            float(previous_score) if previous_score is not None else 0.0,
            progress,
            comparison == "better",
            False,
            assessment.evidence,
        )

    def _output(
        self,
        decision: Decision,
        plan: ActionPlan | None,
        diagnosis: Diagnosis | None,
        previous_score: float | None,
        *,
        initial: bool,
    ) -> VerifierOutput:
        if plan is None:
            error_type = diagnosis.error_type if diagnosis else "uncertain_region"
            target_view = diagnosis.target_view if diagnosis and diagnosis.target_view else "t2"
            region = None
            suggested_action = None
        elif plan.action == "finish":
            error_type = "none"
            target_view = "t2"
            region = None
            suggested_action = "finish"
        else:
            error_type = diagnosis.error_type if diagnosis else "uncertain_region"
            target_view = plan.target_view or "t2"
            if plan.coordinate_normalized_1000 is not None:
                x, y = plan.coordinate_normalized_1000
                region = (x, y, x, y)
            else:
                assert plan.box_normalized_1000 is not None
                region = plan.box_normalized_1000
            suggested_action = plan.action
        state_ready = bool(plan is not None and plan.action == "finish")
        accept = bool(
            state_ready and error_type == "none"
            if initial
            else decision.comparison == "better" and decision.accept
        )
        state_feedback = (
            "current accepted state has no diagnosed remaining error"
            if state_ready
            else f"next state action: {suggested_action or 'none'} ({error_type})"
        )
        feedback = (
            f"transition: {decision.feedback}; state: {state_feedback}"
            if not initial
            else decision.feedback
        )
        return VerifierOutput(
            quality_score=decision.quality_score,
            progress_score=decision.progress_score,
            score_delta=(
                0.0
                if previous_score is None
                else decision.quality_score - previous_score
            ),
            comparison=decision.comparison,
            error_type=error_type,
            target_view=target_view,
            error_region=region,
            suggested_action=suggested_action,
            feedback=feedback,
            accept=accept,
            verifier_valid=True,
            localization_valid=(
                plan is not None and (plan.action == "finish" or region is not None)
            ),
            stop=bool(accept and state_ready),
        )


def _normalize_evidence_quality(value: Any) -> str:
    quality = str(value).strip().lower()
    if quality == "high":
        return "clear"
    if quality not in {"clear", "ambiguous", "insufficient"}:
        raise StageProtocolError("visual judgment contains an invalid evidence_quality")
    return quality


_ATOMIC_EVIDENCE_ANCHOR_PROBE = (
    "Paired T1 and T2 RGB crops show visible physical scene content in the "
    "audited region."
)


def _replace_atomic_evidence(
    payload: Any,
    evidence: Any,
) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise StageProtocolError("atomic evidence repair anchor must be an object")
    visual = payload.get("visual_judgment")
    if not isinstance(visual, Mapping):
        raise StageProtocolError(
            "atomic evidence repair requires a visual_judgment object"
        )
    replaced = dict(payload)
    replaced_visual = dict(visual)
    replaced_visual["evidence"] = evidence
    replaced["visual_judgment"] = replaced_visual
    return replaced


def _validated_atomic_evidence_anchor(
    raw: Any,
    parser: Callable[[Mapping[str, Any]], Any],
) -> Mapping[str, Any] | None:
    """Keep an audit anchor only when replacing evidence makes it fully valid."""

    try:
        candidate = _replace_atomic_evidence(
            raw,
            _ATOMIC_EVIDENCE_ANCHOR_PROBE,
        )
        parser(candidate)
    except (KeyError, TypeError, ValueError, StageProtocolError):
        return None
    return copy.deepcopy(dict(raw))


def _merge_atomic_evidence_repair(
    anchor: Mapping[str, Any],
    repaired: Any,
) -> Mapping[str, Any]:
    """Use only repaired evidence; preserve all validated semantic fields."""

    if not isinstance(repaired, Mapping):
        raise StageProtocolError("atomic evidence repair response must be an object")
    if repaired.get("region_id") != anchor.get("region_id"):
        raise StageProtocolError("atomic evidence repair changed the region_id")
    visual = repaired.get("visual_judgment")
    if not isinstance(visual, Mapping) or "evidence" not in visual:
        raise StageProtocolError(
            "atomic evidence repair response lacks visual_judgment.evidence"
        )
    return _replace_atomic_evidence(anchor, visual["evidence"])


def _parse_judgment(payload: Mapping[str, Any], region_id: str) -> EvidenceJudgment:
    _exact_keys(payload, {"region_id", "visual_judgment"}, "evidence")
    if payload["region_id"] != region_id or not isinstance(payload["visual_judgment"], Mapping):
        raise StageProtocolError("evidence response has the wrong region_id or shape")
    body = payload["visual_judgment"]
    keys = set(body)
    current_keys = {"t1_state", "t2_state", "evidence_quality"}
    legacy_keys = current_keys | {"visual_confidence"}
    if keys != current_keys and keys != legacy_keys:
        raise StageProtocolError(
            "visual_judgment must contain t1_state, t2_state, and evidence_quality"
        )
    t1, t2 = body["t1_state"], body["t2_state"]
    if t1 not in {"building", "background", "mixed", "uncertain"} or t2 not in {"building", "background", "mixed", "uncertain"}:
        raise StageProtocolError("visual judgment contains an invalid RGB state")
    quality = _normalize_evidence_quality(body["evidence_quality"])
    return EvidenceJudgment(region_id, t1, t2, quality)


def _parse_atomic_audit(
    payload: Mapping[str, Any], record: EvidenceRecord
) -> tuple[EvidenceJudgment, Diagnosis]:
    """Parse the compact local audit and derive every redundant runtime field."""

    _exact_keys(
        payload,
        {"region_id", "visual_judgment", "diagnosis"},
        "atomic audit",
    )
    if payload["region_id"] != record.region_id:
        raise StageProtocolError("atomic audit response has the wrong region_id")
    visual = payload["visual_judgment"]
    if not isinstance(visual, Mapping):
        raise StageProtocolError("atomic audit visual_judgment must be an object")
    _exact_keys(
        visual,
        {
            "t1_state",
            "t2_state",
            "evidence",
        },
        "atomic audit visual_judgment",
    )
    mask_state = record.change_mask_state
    t1_state = str(visual["t1_state"])
    t2_state = str(visual["t2_state"])
    if t1_state not in {"building", "background", "mixed", "uncertain"} or t2_state not in {
        "building",
        "background",
        "mixed",
        "uncertain",
    }:
        raise StageProtocolError("atomic audit contains an invalid RGB state")
    evidence_value = visual["evidence"]
    if not isinstance(evidence_value, str):
        raise StageProtocolError("atomic audit evidence must be a string")
    evidence = evidence_value.strip()
    if not evidence:
        raise StageProtocolError("atomic audit must provide observable visual evidence")
    normalized_evidence = evidence.casefold()
    if (
        "strongest local support or contradiction after checking the whole component"
        in normalized_evidence
        or "short_observable_rgb_evidence" in normalized_evidence
        or "replace_with_one_short_paired_rgb_fact" in normalized_evidence
    ):
        raise StageProtocolError(
            "atomic audit evidence copied schema boilerplate; provide a concrete "
            "observable RGB fact"
        )
    if re.search(r"<[^<>\r\n]{1,80}>", normalized_evidence):
        raise StageProtocolError(
            "atomic audit evidence copied an image/schema placeholder; provide a "
            "concrete observable RGB fact"
        )
    numeric_value = r"[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    numeric_or_coordinate_placeholder = bool(
        re.fullmatch(
            rf"\s*[\[(]?\s*{numeric_value}"
            rf"(?:\s*,\s*{numeric_value})*\s*[\])]?\s*",
            evidence,
        )
    )
    try:
        json.loads(evidence)
    except (json.JSONDecodeError, TypeError, ValueError):
        json_literal_placeholder = False
    else:
        json_literal_placeholder = True
    if numeric_or_coordinate_placeholder or json_literal_placeholder:
        raise StageProtocolError(
            "atomic audit evidence is a JSON/numeric/coordinate placeholder; "
            "provide a concrete observable RGB fact"
        )
    machine_label_candidate = normalized_evidence.rstrip(".!?")
    if re.fullmatch(
        r"[a-z0-9]+(?:_[a-z0-9]+)+",
        machine_label_candidate,
    ):
        raise StageProtocolError(
            "atomic audit evidence is an underscore-delimited label; provide a "
            "natural-language observable RGB fact"
        )
    mask_or_presentation_shortcut = any(
        re.search(pattern, normalized_evidence)
        for pattern in (
            r"\bchange_mask_state\b",
            r"\b(?:change[-_ ]mask|change map)\b[^.;\n]{0,80}"
            r"\b(?:black|white|green|empty)\b",
            r"\b(?:black|white|green|empty)\b[^.;\n]{0,40}"
            r"\b(?:change[-_ ]mask|change map)\b",
            r"\b(?:black|white|green|empty)\b[^.;\n]{0,60}"
            r"\bcomponent masks?\b",
            r"\b(?:all[- ]?)?(?:black|white|green)[- ]+"
            r"(?:pixels?[- ]+)?"
            r"(?:(?:predicted|prediction|object|change|component)[- ]+)?masks?\b",
            r"\b(?:(?:predicted|prediction|object|change|component)\s+)?masks?\s+"
            r"(?:is|are|looks?|appears?|remains?)\s+"
            r"(?:all\s+)?(?:black|white|green|empty)\b",
            r"\b(?:black|white|green)\s+(?:presentation|padding)"
            r"(?:\s+pixels?)?\b",
            r"\b(?:presentation|padding)(?:\s+pixels?)?\s+"
            r"(?:is|are|looks?|appears?|remains?)\s+"
            r"(?:black|white|green)\b",
            r"\b(?:t1|t2)[-_ ]+(?:(?:predicted|object|change)[-_ ]+)?masks?\b",
        )
    )
    pure_color_state = bool(
        re.fullmatch(
            r"all[-_ ]+(?:black|white|green)[.!?]?",
            normalized_evidence,
        )
    )
    presentation_colors = (
        r"(?:red|green|blue|cyan|yellow|magenta|orange)"
    )
    presentation_marks = (
        r"(?:outline(?:s|d)?|contour(?:s|ed)?|border(?:s|ed)?|markers?|"
        r"mark(?:s|ed)?|dots?|seeds?|highlight(?:s|ed)?|overlays?|"
        r"label(?:s|ed)?)"
    )
    presentation_annotation_shortcut = bool(
        re.search(
            rf"\b{presentation_colors}[-_ ]+{presentation_marks}\b",
            normalized_evidence,
        )
        or re.search(
            rf"\b{presentation_marks}\b[^.;\n]{{0,24}}\b{presentation_colors}\b",
            normalized_evidence,
        )
    )
    both_view_color_shortcut = bool(
        "all pixels" in normalized_evidence
        and "both" in normalized_evidence
        and "t1" in normalized_evidence
        and "t2" in normalized_evidence
        and re.search(r"\b(?:black|white|green)\b", normalized_evidence)
    )
    if (
        mask_or_presentation_shortcut
        or pure_color_state
        or presentation_annotation_shortcut
        or both_view_color_shortcut
    ):
        raise StageProtocolError(
            "atomic audit evidence relies on mask/presentation color instead of "
            "observable RGB content"
        )

    body = payload["diagnosis"]
    if not isinstance(body, Mapping):
        raise StageProtocolError("atomic audit diagnosis must be an object")
    _exact_keys(
        body,
        {"audit_checklist"},
        "atomic audit diagnosis",
    )
    checklist_payload = body["audit_checklist"]
    if not isinstance(checklist_payload, Mapping):
        raise StageProtocolError("atomic audit checklist must be an object")
    if any(isinstance(value, Mapping) for value in checklist_payload.values()):
        raise StageProtocolError(
            "compact audit checklist values must be status strings"
        )
    checklist = AuditChecklist.from_mapping(checklist_payload)
    error_type = checklist.error_type
    assessment = {
        "none": "correct",
        "false_positive_change": "false_positive",
        "false_negative": "false_negative",
        "mixed_error": "mixed",
        "uncertain_region": "uncertain",
    }[error_type]
    quality = {
        "pass": "clear",
        "uncertain": "ambiguous",
        "fail": "insufficient",
    }[checklist.evidence_sufficient]
    if mask_state == "white" and error_type == "false_negative":
        raise StageProtocolError("a white region cannot be a pure false negative")
    if mask_state == "black" and error_type == "false_positive_change":
        raise StageProtocolError("a black region cannot be a false positive")
    concrete_states = {t1_state, t2_state} <= {"building", "background"}
    target_transition = concrete_states and t1_state != t2_state
    if mask_state == "white" and concrete_states and not target_transition and error_type == "none":
        raise StageProtocolError(
            "a white region with no target transition cannot pass as correct"
        )
    if mask_state == "black" and target_transition and error_type == "none":
        raise StageProtocolError(
            "a black region with a target transition cannot pass as correct"
        )
    if error_type == "false_negative" and concrete_states and not target_transition:
        raise StageProtocolError(
            "a false negative requires a concrete target transition"
        )
    if error_type in {"false_positive_change", "false_negative"} and (
        t1_state not in {"building", "background"}
        or t2_state not in {"building", "background"}
    ):
        raise StageProtocolError(
            "actionable compact audit requires concrete T1/T2 RGB states"
        )
    target_view = _runtime_target_view(record, error_type, t1_state, t2_state)
    judgment = EvidenceJudgment(
        record.region_id,
        t1_state,
        t2_state,
        quality,
        mask_state,
        assessment,
        evidence,
    )
    diagnosis = Diagnosis(
        record.region_id,
        error_type,
        target_view,
        checklist,
        evidence,
    )
    return judgment, diagnosis


def _runtime_target_view(
    record: EvidenceRecord,
    error_type: str,
    t1_state: str,
    t2_state: str,
) -> str | None:
    """Resolve a target only when record facts or paired RGB make it unique."""

    if error_type == "false_positive_change":
        white_views = [
            view
            for view in ("t1", "t2")
            if record.editable_seed_white.get(view) is True
        ]
        return white_views[0] if len(white_views) == 1 else None
    if error_type == "false_negative":
        semantic_target: str | None = None
        if t1_state == "building" and t2_state == "background":
            semantic_target = "t1"
        elif t2_state == "building" and t1_state == "background":
            semantic_target = "t2"
        if (
            semantic_target is not None
            and record.editable_seed_white.get(semantic_target) is False
        ):
            return semantic_target
        return None
    return None


def _uncertain_audit_checklist() -> AuditChecklist:
    return AuditChecklist(
        evidence_sufficient="uncertain",
        target_class_only="uncertain",
        white_pixels_supported="uncertain",
        boundary_alignment="uncertain",
        internal_holes_absent="uncertain",
        changed_object_extent_complete="uncertain",
        fragment_artifacts_absent="uncertain",
    )


def _diagnosis_quality(diagnoses: tuple[Diagnosis, ...]) -> float:
    scores = [
        item.audit_checklist.quality_score
        for item in diagnoses
        if item.audit_checklist is not None
    ]
    return sum(scores) / len(scores) if scores else 0.0


def _exact_keys(payload: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(payload) != expected:
        raise StageProtocolError(
            f"{name} must contain exactly {sorted(expected)}; got {sorted(payload)}"
        )


def _plan_dict(plan: ActionPlan | None) -> dict[str, Any] | None:
    if plan is None:
        return None
    return {
        "region_id": plan.region_id,
        "action": plan.action,
        "target_view": plan.target_view,
        "coordinate_normalized_1000": list(plan.coordinate_normalized_1000)
        if plan.coordinate_normalized_1000 is not None
        else None,
        "box_normalized_1000": list(plan.box_normalized_1000)
        if plan.box_normalized_1000 is not None
        else None,
    }


def _plan_region(plan: ActionPlan) -> tuple[int, int, int, int]:
    if plan.coordinate_normalized_1000 is not None:
        x, y = plan.coordinate_normalized_1000
        return x, y, x, y
    if plan.box_normalized_1000 is None:
        raise StageProtocolError("executable plan has no geometry")
    return plan.box_normalized_1000


def _plan_agent_action(
    plan: ActionPlan, image_size: tuple[int, int]
) -> AgentAction:
    if plan.target_view not in {"t1", "t2"}:
        raise StageProtocolError("executable plan has no target view")
    if plan.coordinate_normalized_1000 is not None:
        return AgentAction(
            plan.target_view,
            plan.action,
            coordinate=normalized_point_to_pixel(
                plan.coordinate_normalized_1000, image_size
            ),
        )
    if plan.box_normalized_1000 is not None:
        x1, y1 = normalized_point_to_pixel(
            plan.box_normalized_1000[:2], image_size
        )
        x2, y2 = normalized_point_to_pixel(
            plan.box_normalized_1000[2:], image_size
        )
        return AgentAction(
            plan.target_view,
            plan.action,
            box=(x1, y1, x2, y2),
        )
    raise StageProtocolError("executable plan has no point or box geometry")
