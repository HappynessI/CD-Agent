"""Full-context Qwen verifier with a deterministic change-detection rubric.

Direct mode deliberately avoids mask-derived Proposals: Qwen sees the complete
T1/T2 pair and masks and authors action geometry.  Qwen does not author quality,
progress, comparison, or acceptance scores.  It answers auditable binary rubric
items; runtime aggregates them and retains ownership of every decision gate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Callable, Mapping, Sequence

from ..action_parser import ActionParser, ActionValidationError
from ..state import AgentAction, ChangeState, VerifierOutput
from ..verifier_protocol import ERROR_TYPES, StageBackend, StageProtocolError


_ERROR_TYPE_ALIASES = {
    "false_negative_change": "false_negative",
    "missing_detection": "false_negative",
    "missing_t1_mask": "false_negative",
    "missing_t2_mask": "false_negative",
    "under_detection": "false_negative",
    "false_positive": "false_positive_change",
    "over_detection": "false_positive_change",
    "spurious_detection": "false_positive_change",
}
_ACTIONS = {"positive_point", "negative_point", "box", "finish"}

# Evidence sufficiency is a hard gate; target-scope is retained as an auditable
# diagnostic rather than a stop gate.  The remaining weights express the final
# change-mask objective.  Precision/recall dominate cosmetic boundary and
# artifact judgments, and every weight is owned by runtime rather than Qwen.
# ``target_class_only`` is an auditable scope diagnostic, not a stop gate.  A
# model can correctly identify non-target pixels as false positives and still
# report that evidence.  Only visual judgeability blocks an executable repair.
_RUBRIC_GATES = ("evidence_sufficient",)
_RUBRIC_WEIGHTS = {
    "change_semantic_precision": 3,
    "change_semantic_recall": 3,
    "changed_object_extent": 2,
    "change_boundary_alignment": 1,
    "change_artifact_control": 1,
}
# Schema contains both hard-gated evidence and auditable target scope.  Scope
# remains outside weighted quality, but must still be parsed and persisted.
_RUBRIC_IDS = _RUBRIC_GATES + ("target_class_only",) + tuple(_RUBRIC_WEIGHTS)
_CANDIDATE_EFFECT_KEYS = {
    "intended_error_improved",
    "introduced_false_positive",
    "introduced_false_negative",
    "boundary_or_artifact_worsened",
    "evidence",
}


@dataclass(frozen=True)
class _RubricJudgment:
    rubric_id: str
    passed: bool
    evidence: str


@dataclass(frozen=True)
class _CandidateEffect:
    intended_error_improved: bool
    introduced_false_positive: bool
    introduced_false_negative: bool
    boundary_or_artifact_worsened: bool
    evidence: str


@dataclass(frozen=True)
class _DirectVerdict:
    rubric: tuple[_RubricJudgment, ...]
    candidate_effect: _CandidateEffect | None
    error_type: str
    target_view: str | None
    suggested_action: str
    coordinate_normalized_1000: tuple[int, int] | None
    box_normalized_1000: tuple[int, int, int, int] | None
    feedback: str


class DirectQwenVerifier:
    """Global visual diagnosis/action grounding without Proposal geometry."""

    SCHEMA_VERSION = "direct_change_rubric_v3"

    def __init__(
        self,
        backend: StageBackend,
        *,
        accept_threshold: float = 0.82,
        max_retries: int = 2,
    ):
        if not 0 <= accept_threshold <= 1:
            raise ValueError("accept_threshold must be in [0,1]")
        if max_retries < 1:
            raise ValueError("max_retries must be positive")
        self.backend = backend
        self.accept_threshold = accept_threshold
        self.max_retries = max_retries
        self.last_evidence: dict[str, Any] = {}
        self._last_valid_output: VerifierOutput | None = None

    def reset(self) -> None:
        self.last_evidence = {}
        self._last_valid_output = None
        reset_audit = getattr(self.backend, "reset_audit", None)
        if callable(reset_audit):
            reset_audit()

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
        """Author a different Direct action after Environment rollback."""

        audit_start = self._audit_length()
        payload = {
            "mode": "replan",
            "schema": self.SCHEMA_VERSION,
            "target_class": accepted_state.query,
            "accepted_score": accepted_feedback.quality_score,
            "accepted_feedback": accepted_feedback.to_dict(),
            "rejected_action": rejected_action.to_dict(),
            "rejection_reasons": list(rejection_reasons),
            "rejection_history": [dict(item) for item in rejection_history],
            "rejected_candidate_verdict": rejected_feedback.to_dict(),
            "rejected_candidate_mask_delta": _mask_delta_summary(
                accepted_state, rejected_candidate
            ),
        }
        try:
            def validate_replan(verdict: _DirectVerdict) -> None:
                """Keep retryable Direct replans executable and non-duplicate."""

                _validate_executable_direct_action(accepted_state, verdict)
                if not _rubric_gates_pass(verdict):
                    raise StageProtocolError(
                        "direct rollback replan rubric hard gates failed"
                    )
                if (
                    _verdict_action(verdict, accepted_state.image_size)
                    == rejected_action
                ):
                    raise StageProtocolError(
                        "direct rollback replan repeated the rejected action"
                    )

            verdict = self._run_stage(
                accepted_state,
                payload,
                mode="replan",
                previous_state=rejected_candidate,
                validate=validate_replan,
            )
            output, aggregate = self._output(
                verdict, accepted_feedback.quality_score, mode="replan"
            )
            self.last_evidence = {
                "type": "direct_qwen_verifier",
                "schema_version": self.SCHEMA_VERSION,
                "decision_mode": "qwen_full_context_direct_rubric_replan",
                "proposal_mode": "direct",
                "verifier_valid": True,
                "localization_valid": output.localization_valid,
                "replan": {
                    "rejected_action": rejected_action.to_dict(),
                    "rejection_reasons": list(rejection_reasons),
                    "rejection_history": payload["rejection_history"],
                    "rejected_candidate_mask_delta": payload[
                        "rejected_candidate_mask_delta"
                    ],
                },
                "direct_verdict": _verdict_to_dict(verdict),
                "rubric_aggregation": aggregate,
                "validation_errors": [],
                "backend_calls": self._backend_calls_since(audit_start),
            }
            # Replan contract keeps accept=false in audit output.  If replan
            # nevertheless proves accepted state clean and asks for finish,
            # retain an internal finish authorization for the following
            # programmatic identical-state check.
            self._last_valid_output = (
                replace(output, accept=True, stop=True)
                if output.error_type == "none"
                and output.suggested_action == "finish"
                else output
            )
            return output
        except (KeyError, TypeError, ValueError, StageProtocolError) as error:
            self.last_evidence = {
                "type": "direct_qwen_verifier",
                "schema_version": self.SCHEMA_VERSION,
                "decision_mode": "qwen_full_context_direct_rubric_replan",
                "proposal_mode": "direct",
                "verifier_valid": False,
                "localization_valid": False,
                "replan": {
                    "rejected_action": rejected_action.to_dict(),
                    "rejection_reasons": list(rejection_reasons),
                    "rejection_history": [dict(item) for item in rejection_history],
                },
                "validation_errors": [str(error)],
                "backend_calls": self._backend_calls_since(audit_start),
            }
            return VerifierOutput(
                quality_score=accepted_feedback.quality_score,
                progress_score=0.0,
                comparison="uncertain",
                error_type=accepted_feedback.error_type,
                target_view=accepted_feedback.target_view,
                error_region=None,
                suggested_action=None,
                feedback="Direct rollback replan was invalid; no action is authorized.",
                accept=False,
                verifier_valid=False,
                localization_valid=False,
                stop=False,
            )

    def verify(
        self,
        state: ChangeState,
        previous_score: float | None,
        previous_action: AgentAction | None,
        previous_state: ChangeState | None = None,
    ) -> VerifierOutput:
        audit_start = self._audit_length()
        try:
            if previous_state is not None and self._states_identical(state, previous_state):
                previous = self._last_valid_output
                output = VerifierOutput(
                    quality_score=previous.quality_score if previous else previous_score,
                    progress_score=0.0,
                    score_delta=0.0,
                    comparison="unchanged",
                    error_type=previous.error_type if previous else "uncertain_region",
                    target_view=previous.target_view if previous else "t2",
                    error_region=previous.error_region if previous else None,
                    suggested_action=previous.suggested_action if previous else None,
                    feedback="Candidate masks are identical to the accepted state.",
                    accept=bool(previous and previous.accept),
                    verifier_valid=True,
                    localization_valid=bool(previous and previous.localization_valid),
                    stop=bool(previous and previous.stop),
                )
                self.last_evidence = {
                    "type": "direct_qwen_verifier",
                    "schema_version": self.SCHEMA_VERSION,
                    "decision_mode": "programmatic_identical_state",
                    "proposal_mode": "direct",
                    "verifier_valid": True,
                    "localization_valid": output.localization_valid,
                    "rubric_reused_from_accepted_state": True,
                    "validation_errors": [],
                    "backend_calls": self._backend_calls_since(audit_start),
                }
                return output
            mode = "initial" if previous_state is None else "candidate"
            verdict = self._run_stage(
                state,
                {
                    "mode": mode,
                    "schema": self.SCHEMA_VERSION,
                    "target_class": state.query,
                    "previous_score": previous_score,
                    "previous_action": (
                        previous_action.to_dict() if previous_action else None
                    ),
                },
                mode=mode,
                previous_state=previous_state,
                validate=lambda candidate: _validate_executable_direct_action(
                    state, candidate
                ),
            )
            output, aggregate = self._output(verdict, previous_score, mode=mode)
            self.last_evidence = {
                "type": "direct_qwen_verifier",
                "schema_version": self.SCHEMA_VERSION,
                "decision_mode": "qwen_full_context_direct_rubric_action",
                "proposal_mode": "direct",
                "verifier_valid": True,
                "localization_valid": output.localization_valid,
                "direct_verdict": _verdict_to_dict(verdict),
                "rubric_aggregation": aggregate,
                "validation_errors": [],
                "backend_calls": self._backend_calls_since(audit_start),
            }
            self._last_valid_output = output
            return output
        except (KeyError, TypeError, ValueError, StageProtocolError) as error:
            previous = self._last_valid_output
            self.last_evidence = {
                "type": "direct_qwen_verifier",
                "schema_version": self.SCHEMA_VERSION,
                "proposal_mode": "direct",
                "verifier_valid": False,
                "localization_valid": False,
                "validation_errors": [str(error)],
                "backend_calls": self._backend_calls_since(audit_start),
            }
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
                feedback="Direct verifier output was invalid; no action is authorized.",
                accept=False,
                verifier_valid=False,
                localization_valid=False,
                stop=False,
            )

    def _run_stage(
        self,
        state: ChangeState,
        payload: Mapping[str, Any],
        *,
        mode: str,
        previous_state: ChangeState | None,
        validate: Callable[[_DirectVerdict], None] | None = None,
    ) -> _DirectVerdict:
        errors: list[str] = []
        repair = getattr(self.backend, "repair_stage", None)
        semantic_anchor: tuple[Any, ...] | None = None
        previous_raw: Mapping[str, Any] | None = None
        for attempt in range(self.max_retries):
            try:
                attempt_payload = dict(payload)
                if attempt > 0 and previous_raw is not None:
                    attempt_payload["repair_context"] = {
                        "previous_invalid_response": dict(previous_raw),
                        "preserve_semantic_fields": [
                            "rubric.pass",
                            "error_type",
                            "candidate_effect",
                        ],
                    }
                raw = (
                    self.backend.generate_stage(
                        "direct", state, attempt_payload, previous_state
                    )
                    if attempt == 0
                    else repair(
                        "direct",
                        state,
                        attempt_payload,
                        errors[-1],
                        previous_state,
                    )
                    if callable(repair)
                    else None
                )
                if raw is None:
                    break
                if isinstance(raw, Mapping):
                    previous_raw = dict(raw)
                try:
                    verdict = _parse_direct_verdict(raw, mode=mode)
                except (KeyError, TypeError, ValueError, StageProtocolError):
                    raw_anchor = _raw_semantic_anchor(raw)
                    if raw_anchor is not None and semantic_anchor is None:
                        semantic_anchor = raw_anchor
                    raise
                current_anchor = _semantic_anchor(verdict)
                if semantic_anchor is None:
                    semantic_anchor = current_anchor
                elif attempt > 0 and current_anchor != semantic_anchor:
                    raise StageProtocolError(
                        "direct repair changed rubric pass values or semantic error_type"
                    )
                if validate is not None:
                    validate(verdict)
                return verdict
            except (KeyError, TypeError, ValueError, StageProtocolError) as error:
                errors.append(str(error))
        if not errors:
            raise StageProtocolError("direct backend does not support repair retries")
        raise StageProtocolError(
            f"direct failed after {len(errors)} attempt(s): {errors[-1]}"
        )

    def _output(
        self,
        verdict: _DirectVerdict,
        previous_score: float | None,
        *,
        mode: str,
    ) -> tuple[VerifierOutput, dict[str, Any]]:
        computed_quality = _rubric_quality(verdict)
        quality = (
            previous_score
            if mode == "replan" and previous_score is not None
            else computed_quality
        )
        score_delta = 0.0 if previous_score is None else quality - previous_score
        comparison = _derived_comparison(verdict, mode)
        gates_pass = _rubric_gates_pass(verdict)
        region = (
            (
                verdict.coordinate_normalized_1000[0],
                verdict.coordinate_normalized_1000[1],
                verdict.coordinate_normalized_1000[0],
                verdict.coordinate_normalized_1000[1],
            )
            if verdict.coordinate_normalized_1000 is not None
            else verdict.box_normalized_1000
        )
        accept = (
            gates_pass
            and verdict.error_type == "none"
            and quality >= self.accept_threshold
            if mode == "initial"
            else gates_pass and comparison == "better"
            if mode == "candidate"
            else False
        )
        stop = bool(
            accept
            and verdict.error_type == "none"
            and verdict.suggested_action == "finish"
        )
        suggested_action = verdict.suggested_action if gates_pass else None
        localization_valid = bool(
            gates_pass and (suggested_action == "finish" or region is not None)
        )
        aggregate = {
            "source": "runtime_weighted_binary_rubric",
            "quality_score": quality,
            "computed_rubric_quality": computed_quality,
            "score_delta": score_delta,
            "comparison": comparison,
            "accept": accept,
            "hard_gates": list(_RUBRIC_GATES),
            "hard_gates_pass": gates_pass,
            "weights": dict(_RUBRIC_WEIGHTS),
        }
        return VerifierOutput(
            quality_score=quality,
            progress_score=score_delta,
            score_delta=score_delta,
            comparison=comparison,
            error_type=verdict.error_type,
            target_view=verdict.target_view or "t2",
            error_region=region if gates_pass else None,
            suggested_action=suggested_action,
            feedback=verdict.feedback,
            accept=accept,
            verifier_valid=True,
            localization_valid=localization_valid,
            stop=stop,
        ), aggregate

    def _audit_length(self) -> int:
        history = getattr(self.backend, "call_history", None)
        return len(history) if isinstance(history, list) else 0

    def _backend_calls_since(self, start: int) -> list[dict[str, Any]]:
        history = getattr(self.backend, "call_history", None)
        if not isinstance(history, list):
            return []
        return [dict(item) for item in history[start:] if isinstance(item, Mapping)]

    @staticmethod
    def _states_identical(state: ChangeState, previous_state: ChangeState) -> bool:
        return (
            (state.t1_mask == previous_state.t1_mask).all()
            and (state.t2_mask == previous_state.t2_mask).all()
            and (state.change_mask == previous_state.change_mask).all()
        )


def _parse_direct_verdict(payload: Mapping[str, Any], *, mode: str) -> _DirectVerdict:
    if set(payload) != {"verdict"} or not isinstance(payload["verdict"], Mapping):
        raise StageProtocolError("direct response must contain exactly a verdict object")
    if mode not in {"initial", "candidate", "replan"}:
        raise StageProtocolError(f"unsupported direct verdict mode: {mode!r}")
    body = payload["verdict"]
    expected = {
        "rubric",
        "candidate_effect",
        "error_type",
        "target_view",
        "suggested_action",
        "coordinate_normalized_1000",
        "box_normalized_1000",
        "feedback",
    }
    if set(body) != expected:
        raise StageProtocolError(
            f"direct verdict must contain exactly {sorted(expected)}; got {sorted(body)}"
        )
    rubric = _parse_rubric(body["rubric"])
    candidate_effect = _parse_candidate_effect(body["candidate_effect"], mode)
    error_type = _canonical_error_type(body["error_type"])
    if error_type not in ERROR_TYPES:
        raise StageProtocolError("direct verdict error_type is unsupported")
    action = str(body["suggested_action"])
    if action not in _ACTIONS:
        raise StageProtocolError("direct verdict suggested_action is unsupported")
    target = body["target_view"]
    coordinate = _point(body["coordinate_normalized_1000"])
    box = _box(body["box_normalized_1000"])
    feedback = body["feedback"]
    if not isinstance(feedback, str) or not feedback.strip():
        raise StageProtocolError("direct verdict feedback must be a non-empty string")
    verdict = _DirectVerdict(
        rubric=rubric,
        candidate_effect=candidate_effect,
        error_type=error_type,
        target_view=target,
        suggested_action=action,
        coordinate_normalized_1000=coordinate,
        box_normalized_1000=box,
        feedback=feedback.strip(),
    )
    all_quality_pass = all(
        _rubric_passes(verdict, rubric_id) for rubric_id in _RUBRIC_WEIGHTS
    )
    gates_pass = _rubric_gates_pass(verdict)
    if not gates_pass and error_type != "uncertain_region":
        raise StageProtocolError(
            "failed Direct rubric hard gate requires error_type=uncertain_region"
        )
    target_scope_pass = _rubric_passes(verdict, "target_class_only")
    if error_type == "none" and not (
        gates_pass and target_scope_pass and all_quality_pass
    ):
        raise StageProtocolError("error_type=none requires every Direct rubric item to pass")
    if gates_pass and target_scope_pass and all_quality_pass and error_type != "none":
        raise StageProtocolError("all Direct rubric items pass requires error_type=none")
    if error_type == "none":
        if target is not None or action != "finish" or coordinate is not None or box is not None:
            raise StageProtocolError(
                "none direct verdict must finish with null target and geometry"
            )
    else:
        if target not in {"t1", "t2"} or action == "finish":
            raise StageProtocolError("direct error verdict needs target_view and tool action")
        if action in {"positive_point", "negative_point"}:
            if coordinate is None or box is not None:
                raise StageProtocolError(
                    "direct point action needs only coordinate_normalized_1000"
                )
        elif box is None or coordinate is not None:
            raise StageProtocolError(
                "direct box action needs only box_normalized_1000"
            )
    return verdict


def _parse_rubric(value: Any) -> tuple[_RubricJudgment, ...]:
    if not isinstance(value, Mapping) or set(value) != set(_RUBRIC_IDS):
        got = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise StageProtocolError(
            f"direct rubric must contain exactly {sorted(_RUBRIC_IDS)}; got {got}"
        )
    judgments: list[_RubricJudgment] = []
    for rubric_id in _RUBRIC_IDS:
        item = value[rubric_id]
        if not isinstance(item, Mapping) or set(item) != {"pass", "evidence"}:
            raise StageProtocolError(
                f"direct rubric {rubric_id} must contain exactly pass and evidence"
            )
        if not isinstance(item["pass"], bool):
            raise StageProtocolError(f"direct rubric {rubric_id} pass must be boolean")
        evidence = item["evidence"]
        if not isinstance(evidence, str) or not evidence.strip():
            raise StageProtocolError(
                f"direct rubric {rubric_id} evidence must be non-empty"
            )
        judgments.append(
            _RubricJudgment(rubric_id, bool(item["pass"]), evidence.strip())
        )
    return tuple(judgments)


def _parse_candidate_effect(value: Any, mode: str) -> _CandidateEffect | None:
    if mode != "candidate":
        if value is not None:
            raise StageProtocolError(
                "initial/replan direct candidate_effect must be JSON null"
            )
        return None
    if not isinstance(value, Mapping) or set(value) != _CANDIDATE_EFFECT_KEYS:
        raise StageProtocolError(
            "candidate direct verdict needs exact binary candidate_effect fields"
        )
    boolean_keys = _CANDIDATE_EFFECT_KEYS - {"evidence"}
    if any(not isinstance(value[key], bool) for key in boolean_keys):
        raise StageProtocolError("direct candidate_effect flags must be booleans")
    evidence = value["evidence"]
    if not isinstance(evidence, str) or not evidence.strip():
        raise StageProtocolError("direct candidate_effect evidence must be non-empty")
    return _CandidateEffect(
        intended_error_improved=bool(value["intended_error_improved"]),
        introduced_false_positive=bool(value["introduced_false_positive"]),
        introduced_false_negative=bool(value["introduced_false_negative"]),
        boundary_or_artifact_worsened=bool(
            value["boundary_or_artifact_worsened"]
        ),
        evidence=evidence.strip(),
    )


def _rubric_mapping(verdict: _DirectVerdict) -> dict[str, _RubricJudgment]:
    return {item.rubric_id: item for item in verdict.rubric}


def _rubric_passes(verdict: _DirectVerdict, rubric_id: str) -> bool:
    return _rubric_mapping(verdict)[rubric_id].passed


def _rubric_gates_pass(verdict: _DirectVerdict) -> bool:
    return all(_rubric_passes(verdict, rubric_id) for rubric_id in _RUBRIC_GATES)


def _rubric_quality(verdict: _DirectVerdict) -> float:
    total = sum(_RUBRIC_WEIGHTS.values())
    passed = sum(
        weight
        for rubric_id, weight in _RUBRIC_WEIGHTS.items()
        if _rubric_passes(verdict, rubric_id)
    )
    return round(passed / total, 6)


def _derived_comparison(verdict: _DirectVerdict, mode: str) -> str:
    if mode == "initial":
        return "initial"
    if mode == "replan" or not _rubric_gates_pass(verdict):
        return "uncertain"
    effect = verdict.candidate_effect
    if effect is None:
        raise StageProtocolError("candidate comparison requires candidate_effect")
    harm = (
        effect.introduced_false_positive
        or effect.introduced_false_negative
        or effect.boundary_or_artifact_worsened
    )
    if effect.intended_error_improved and not harm:
        return "better"
    if harm and not effect.intended_error_improved:
        return "worse"
    if not effect.intended_error_improved and not harm:
        return "unchanged"
    return "uncertain"


def _validate_executable_direct_action(
    state: ChangeState, verdict: _DirectVerdict
) -> None:
    """Reject deterministic no-op point plans before a segmentation worker runs.

    This is a geometry/tool-contract check, not a semantic decision: a negative
    click can only subtract SimpleClick-refined pixels from a currently white
    local mask region, so its seed must be white. A positive click is deliberately
    not constrained by current seed occupancy because SimpleClick can expand a
    component from an already-white point. Direct mode has no Environment
    Proposal, but it still owns the same current T1/T2 masks and can reject the
    deterministic negative no-op before launching an expensive worker.
    """

    if verdict.error_type == "none" or verdict.suggested_action == "finish":
        return
    action = _verdict_action(verdict, state.image_size)
    if action.action not in {"positive_point", "negative_point"}:
        return
    if action.coordinate is None:
        raise StageProtocolError("direct point action has no pixel coordinate")
    target_mask = state.t1_mask if action.target_view == "t1" else state.t2_mask
    x, y = action.coordinate
    seed_is_white = bool(target_mask[y, x])
    if action.action == "negative_point" and not seed_is_white:
        raise StageProtocolError(
            "direct negative_point requires a white seed in the editable target mask"
        )


def _verdict_to_dict(verdict: _DirectVerdict) -> dict[str, Any]:
    return asdict(verdict)


def _semantic_anchor(verdict: _DirectVerdict) -> tuple[Any, ...]:
    """Fields a structural repair must not reinterpret."""

    rubric_pass = tuple(
        (item.rubric_id, item.passed) for item in verdict.rubric
    )
    effect = verdict.candidate_effect
    effect_flags = (
        None
        if effect is None
        else (
            effect.intended_error_improved,
            effect.introduced_false_positive,
            effect.introduced_false_negative,
            effect.boundary_or_artifact_worsened,
        )
    )
    return verdict.error_type, rubric_pass, effect_flags


def _raw_semantic_anchor(raw: Any) -> tuple[Any, ...] | None:
    """Recover semantic fields from a response that failed geometry parsing."""

    if not isinstance(raw, Mapping) or not isinstance(raw.get("verdict"), Mapping):
        return None
    body = raw["verdict"]
    rubric_value = body.get("rubric")
    error_type = body.get("error_type")
    if not isinstance(rubric_value, Mapping) or error_type is None:
        return None
    try:
        rubric = _parse_rubric(rubric_value)
    except (KeyError, TypeError, ValueError, StageProtocolError):
        return None
    effect_flags: tuple[bool, bool, bool, bool] | None = None
    effect = body.get("candidate_effect")
    if isinstance(effect, Mapping) and all(
        isinstance(effect.get(key), bool)
        for key in (
            "intended_error_improved",
            "introduced_false_positive",
            "introduced_false_negative",
            "boundary_or_artifact_worsened",
        )
    ):
        effect_flags = (
            bool(effect["intended_error_improved"]),
            bool(effect["introduced_false_positive"]),
            bool(effect["introduced_false_negative"]),
            bool(effect["boundary_or_artifact_worsened"]),
        )
    return (
        _canonical_error_type(error_type),
        tuple((item.rubric_id, item.passed) for item in rubric),
        effect_flags,
    )


def _canonical_error_type(value: Any) -> str:
    raw = str(value).strip().lower().replace("-", " ").replace(" ", "_")
    return _ERROR_TYPE_ALIASES.get(raw, raw)


def _point(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2 or any(
        isinstance(item, bool) or not isinstance(item, int) or not 0 <= item <= 1000
        for item in value
    ):
        raise StageProtocolError(
            "direct coordinate_normalized_1000 must be two [0,1000] integers"
        )
    return int(value[0]), int(value[1])


def _box(value: Any) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4 or any(
        isinstance(item, bool) or not isinstance(item, int) or not 0 <= item <= 1000
        for item in value
    ):
        raise StageProtocolError(
            "direct box_normalized_1000 must contain four [0,1000] integers"
        )
    x1, y1, x2, y2 = (int(item) for item in value)
    if x1 >= x2 or y1 >= y2:
        raise StageProtocolError("direct box_normalized_1000 must be ordered")
    return x1, y1, x2, y2


def _mask_delta_summary(
    accepted_state: ChangeState, rejected_candidate: ChangeState
) -> dict[str, int]:
    """Return GT-free changed-pixel counts for a rejected Direct candidate."""

    return {
        "t1_changed_pixels": int(
            (accepted_state.t1_mask != rejected_candidate.t1_mask).sum()
        ),
        "t2_changed_pixels": int(
            (accepted_state.t2_mask != rejected_candidate.t2_mask).sum()
        ),
        "change_changed_pixels": int(
            (accepted_state.change_mask != rejected_candidate.change_mask).sum()
        ),
    }


def _verdict_action(
    verdict: _DirectVerdict, image_size: tuple[int, int]
) -> AgentAction:
    """Convert Direct public geometry to Environment pixel action."""

    payload: dict[str, Any] = {
        "target_view": verdict.target_view or "t2",
        "action": verdict.suggested_action,
    }
    if verdict.coordinate_normalized_1000 is not None:
        payload["coordinate"] = list(verdict.coordinate_normalized_1000)
    if verdict.box_normalized_1000 is not None:
        payload["box"] = list(verdict.box_normalized_1000)
    try:
        return ActionParser().parse_payload(payload, image_size)
    except ActionValidationError as error:
        raise StageProtocolError(
            f"direct verdict cannot be converted to an executable action: {error}"
        ) from error
