"""Frozen, GT-free terminal-review scheduling for the Qwen3-VL H0/H1 arms.

The scheduler in this module is deliberately independent of the Environment and
the tool executor.  It can request one round-two observation and return either a
confirmed terminal decision or a value that still has to enter the existing
parser/geometry/candidate safety flow.  It has no API capable of executing a
tool action.

H0 is an identity decorator.  H1 recognizes only the two preregistered terminal
triggers and consumes at most one independent review for an accepted-state ID.
The complete two-trigger by seven-outcome transition table is encoded below so
callers cannot invent an unregistered fallback at runtime.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Generic, Mapping, TypeVar


SCHEMA_VERSION = "qwen3vl_terminal_review_audit_v1"
FACTOR_ID = "terminal_review_scheduler_v1"
SELECTION_POLICY_VERSION = "environment_ordered_terminal_review_v1"
# These identifiers are part of the raw round-two evidence.  They make the
# plan-null selection branch auditable: a changed ordering alone cannot be
# presented as an alternative diagnosis prompt.
PLAN_NULL_UNREVIEWED_PROMPT_VERSION = (
    "qwen3vl_plan_null_next_unreviewed_region_v1"
)
PLAN_NULL_ALTERNATIVE_PROMPT_VERSION = (
    "qwen3vl_plan_null_alternative_diagnosis_v1"
)

FORBIDDEN_TRIGGER_INPUTS = frozenset(
    {
        "gt_mask",
        "offline_tp_fp_fn",
        "offline_iou_f1",
        "manual_failure_label",
        "historical_false_finish_identity",
    }
)
ALLOWED_TRIGGER_INPUTS = frozenset(
    {
        "accepted_environment_state",
        "round_one_protocol_result",
        "round_one_runtime_result",
        "budget_counts",
    }
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_NONNEGATIVE_COUNTER_FIELDS = (
    "verifier_round_count",
    "stage_initial_attempt_count",
    "schema_repair_attempt_count",
    "network_retry_attempt_count",
    "tool_action_attempt_count",
    "tool_failure_count",
    "environment_step_count",
)

# The terminal-review audit intentionally combines a per-accepted-state
# semantic-round snapshot with the episode's Environment-step snapshot.  Tool
# attempts cannot occur inside the scheduler and are reported separately by the
# episode harness.  Publishing the scope prevents validators from incorrectly
# equating these local counters with episode totals.
TERMINAL_REVIEW_COUNTER_SCOPE: Mapping[str, str] = MappingProxyType(
    {
        "verifier_round_count": "accepted_environment_state",
        "stage_initial_attempt_count": "accepted_environment_state",
        "schema_repair_attempt_count": "accepted_environment_state",
        "network_retry_attempt_count": "accepted_environment_state",
        "tool_action_attempt_count": "terminal_review_scheduler_only",
        "tool_failure_count": "terminal_review_scheduler_only",
        "environment_step_count": "episode_cumulative_snapshot",
    }
)


class TerminalReviewContractError(ValueError):
    """Raised when scheduler inputs would weaken the frozen P1 contract."""


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class TerminalReviewArm(_StringEnum):
    H0 = "h0"
    H1 = "h1"


class TerminalTrigger(_StringEnum):
    PROVISIONAL_FINISH = "provisional_finish"
    PLAN_NULL_AFTER_ALLOWED_REPAIR = "plan_null_after_allowed_repair"


class RoundTwoKind(_StringEnum):
    FINISH_CONFIRMATION = "finish_confirmation"
    PLAN_NULL_ALTERNATIVE = "plan_null_alternative"


class RoundTwoOutcome(_StringEnum):
    FINISH = "finish"
    UNIQUE_SAFE_PLAN = "unique_safe_plan"
    PLAN_NULL = "plan_null"
    UNRESOLVED = "unresolved"
    UNSAFE_OR_NONUNIQUE_PLAN = "unsafe_or_nonunique_plan"
    SCHEMA_INVALID_AFTER_ALLOWED_REPAIR = (
        "schema_invalid_after_allowed_repair"
    )
    BUDGET_EXHAUSTED = "budget_exhausted"


class ReviewDecision(_StringEnum):
    CONFIRMED_FINISH = "confirmed_finish"
    SAFE_PLAN_ENTERS_EXISTING_FLOW = "safe_plan_enters_existing_flow"
    ROUND_CONFLICT_NO_ACTION = "round_conflict_no_action"
    NO_EXECUTABLE_ACTION_AFTER_REVIEW = "no_executable_action_after_review"


class ReviewNextState(_StringEnum):
    TERMINATE_CONFIRMED_FINISH = "terminate_confirmed_finish"
    EXISTING_ENVIRONMENT_ACTION_AND_CANDIDATE_FLOW = (
        "existing_environment_action_and_candidate_flow"
    )
    TERMINATE_NO_EXECUTABLE_ACTION_AFTER_REVIEW = (
        "terminate_no_executable_action_after_review"
    )


class ToolActionAuthorization(_StringEnum):
    FORBIDDEN = "forbidden"
    EXISTING_SAFETY_FLOW_REQUIRED = (
        "not_yet_authorized_existing_safety_flow_required"
    )


class StageAttemptKind(_StringEnum):
    INITIAL = "initial"
    REPAIR = "repair"


@dataclass(frozen=True)
class TerminalTransition:
    trigger: TerminalTrigger
    outcome: RoundTwoOutcome
    decision: ReviewDecision
    next_state: ReviewNextState
    tool_action_authorization: ToolActionAuthorization

    def to_dict(self) -> dict[str, str]:
        return {
            "round_one_trigger": self.trigger.value,
            "round_two_outcome": self.outcome.value,
            "review_decision": self.decision.value,
            "next_state": self.next_state.value,
            "tool_action_authorization": self.tool_action_authorization.value,
        }


def _transition(
    trigger: TerminalTrigger,
    outcome: RoundTwoOutcome,
    decision: ReviewDecision,
    next_state: ReviewNextState,
    authorization: ToolActionAuthorization,
) -> TerminalTransition:
    return TerminalTransition(trigger, outcome, decision, next_state, authorization)


_NO_ACTION = ReviewNextState.TERMINATE_NO_EXECUTABLE_ACTION_AFTER_REVIEW
_FORBIDDEN = ToolActionAuthorization.FORBIDDEN

_TRANSITIONS = {
    (TerminalTrigger.PROVISIONAL_FINISH, RoundTwoOutcome.FINISH): _transition(
        TerminalTrigger.PROVISIONAL_FINISH,
        RoundTwoOutcome.FINISH,
        ReviewDecision.CONFIRMED_FINISH,
        ReviewNextState.TERMINATE_CONFIRMED_FINISH,
        _FORBIDDEN,
    ),
    (
        TerminalTrigger.PROVISIONAL_FINISH,
        RoundTwoOutcome.UNIQUE_SAFE_PLAN,
    ): _transition(
        TerminalTrigger.PROVISIONAL_FINISH,
        RoundTwoOutcome.UNIQUE_SAFE_PLAN,
        ReviewDecision.SAFE_PLAN_ENTERS_EXISTING_FLOW,
        ReviewNextState.EXISTING_ENVIRONMENT_ACTION_AND_CANDIDATE_FLOW,
        ToolActionAuthorization.EXISTING_SAFETY_FLOW_REQUIRED,
    ),
    (TerminalTrigger.PROVISIONAL_FINISH, RoundTwoOutcome.PLAN_NULL): _transition(
        TerminalTrigger.PROVISIONAL_FINISH,
        RoundTwoOutcome.PLAN_NULL,
        ReviewDecision.ROUND_CONFLICT_NO_ACTION,
        _NO_ACTION,
        _FORBIDDEN,
    ),
    (TerminalTrigger.PROVISIONAL_FINISH, RoundTwoOutcome.UNRESOLVED): _transition(
        TerminalTrigger.PROVISIONAL_FINISH,
        RoundTwoOutcome.UNRESOLVED,
        ReviewDecision.NO_EXECUTABLE_ACTION_AFTER_REVIEW,
        _NO_ACTION,
        _FORBIDDEN,
    ),
    (
        TerminalTrigger.PROVISIONAL_FINISH,
        RoundTwoOutcome.UNSAFE_OR_NONUNIQUE_PLAN,
    ): _transition(
        TerminalTrigger.PROVISIONAL_FINISH,
        RoundTwoOutcome.UNSAFE_OR_NONUNIQUE_PLAN,
        ReviewDecision.NO_EXECUTABLE_ACTION_AFTER_REVIEW,
        _NO_ACTION,
        _FORBIDDEN,
    ),
    (
        TerminalTrigger.PROVISIONAL_FINISH,
        RoundTwoOutcome.SCHEMA_INVALID_AFTER_ALLOWED_REPAIR,
    ): _transition(
        TerminalTrigger.PROVISIONAL_FINISH,
        RoundTwoOutcome.SCHEMA_INVALID_AFTER_ALLOWED_REPAIR,
        ReviewDecision.NO_EXECUTABLE_ACTION_AFTER_REVIEW,
        _NO_ACTION,
        _FORBIDDEN,
    ),
    (
        TerminalTrigger.PROVISIONAL_FINISH,
        RoundTwoOutcome.BUDGET_EXHAUSTED,
    ): _transition(
        TerminalTrigger.PROVISIONAL_FINISH,
        RoundTwoOutcome.BUDGET_EXHAUSTED,
        ReviewDecision.NO_EXECUTABLE_ACTION_AFTER_REVIEW,
        _NO_ACTION,
        _FORBIDDEN,
    ),
    (
        TerminalTrigger.PLAN_NULL_AFTER_ALLOWED_REPAIR,
        RoundTwoOutcome.FINISH,
    ): _transition(
        TerminalTrigger.PLAN_NULL_AFTER_ALLOWED_REPAIR,
        RoundTwoOutcome.FINISH,
        ReviewDecision.ROUND_CONFLICT_NO_ACTION,
        _NO_ACTION,
        _FORBIDDEN,
    ),
    (
        TerminalTrigger.PLAN_NULL_AFTER_ALLOWED_REPAIR,
        RoundTwoOutcome.UNIQUE_SAFE_PLAN,
    ): _transition(
        TerminalTrigger.PLAN_NULL_AFTER_ALLOWED_REPAIR,
        RoundTwoOutcome.UNIQUE_SAFE_PLAN,
        ReviewDecision.SAFE_PLAN_ENTERS_EXISTING_FLOW,
        ReviewNextState.EXISTING_ENVIRONMENT_ACTION_AND_CANDIDATE_FLOW,
        ToolActionAuthorization.EXISTING_SAFETY_FLOW_REQUIRED,
    ),
    (
        TerminalTrigger.PLAN_NULL_AFTER_ALLOWED_REPAIR,
        RoundTwoOutcome.PLAN_NULL,
    ): _transition(
        TerminalTrigger.PLAN_NULL_AFTER_ALLOWED_REPAIR,
        RoundTwoOutcome.PLAN_NULL,
        ReviewDecision.NO_EXECUTABLE_ACTION_AFTER_REVIEW,
        _NO_ACTION,
        _FORBIDDEN,
    ),
    (
        TerminalTrigger.PLAN_NULL_AFTER_ALLOWED_REPAIR,
        RoundTwoOutcome.UNRESOLVED,
    ): _transition(
        TerminalTrigger.PLAN_NULL_AFTER_ALLOWED_REPAIR,
        RoundTwoOutcome.UNRESOLVED,
        ReviewDecision.NO_EXECUTABLE_ACTION_AFTER_REVIEW,
        _NO_ACTION,
        _FORBIDDEN,
    ),
    (
        TerminalTrigger.PLAN_NULL_AFTER_ALLOWED_REPAIR,
        RoundTwoOutcome.UNSAFE_OR_NONUNIQUE_PLAN,
    ): _transition(
        TerminalTrigger.PLAN_NULL_AFTER_ALLOWED_REPAIR,
        RoundTwoOutcome.UNSAFE_OR_NONUNIQUE_PLAN,
        ReviewDecision.NO_EXECUTABLE_ACTION_AFTER_REVIEW,
        _NO_ACTION,
        _FORBIDDEN,
    ),
    (
        TerminalTrigger.PLAN_NULL_AFTER_ALLOWED_REPAIR,
        RoundTwoOutcome.SCHEMA_INVALID_AFTER_ALLOWED_REPAIR,
    ): _transition(
        TerminalTrigger.PLAN_NULL_AFTER_ALLOWED_REPAIR,
        RoundTwoOutcome.SCHEMA_INVALID_AFTER_ALLOWED_REPAIR,
        ReviewDecision.NO_EXECUTABLE_ACTION_AFTER_REVIEW,
        _NO_ACTION,
        _FORBIDDEN,
    ),
    (
        TerminalTrigger.PLAN_NULL_AFTER_ALLOWED_REPAIR,
        RoundTwoOutcome.BUDGET_EXHAUSTED,
    ): _transition(
        TerminalTrigger.PLAN_NULL_AFTER_ALLOWED_REPAIR,
        RoundTwoOutcome.BUDGET_EXHAUSTED,
        ReviewDecision.NO_EXECUTABLE_ACTION_AFTER_REVIEW,
        _NO_ACTION,
        _FORBIDDEN,
    ),
}

TRANSITION_TABLE: Mapping[
    tuple[TerminalTrigger, RoundTwoOutcome], TerminalTransition
] = MappingProxyType(_TRANSITIONS)


def transition_for(
    trigger: TerminalTrigger, outcome: RoundTwoOutcome
) -> TerminalTransition:
    """Return the one frozen transition for a trigger/outcome pair."""

    try:
        return TRANSITION_TABLE[(trigger, outcome)]
    except KeyError as error:
        raise TerminalReviewContractError(
            f"unregistered terminal-review transition: {trigger!r}/{outcome!r}"
        ) from error


def select_terminal_trigger(
    *,
    provisional_finish: bool,
    plan_null_after_allowed_repair: bool,
) -> TerminalTrigger | None:
    """Select only one of the two frozen GT-free triggers."""

    if not isinstance(provisional_finish, bool) or not isinstance(
        plan_null_after_allowed_repair, bool
    ):
        raise TerminalReviewContractError("terminal trigger flags must be booleans")
    if provisional_finish and plan_null_after_allowed_repair:
        raise TerminalReviewContractError(
            "provisional finish and repaired plan-null cannot both be true"
        )
    if provisional_finish:
        return TerminalTrigger.PROVISIONAL_FINISH
    if plan_null_after_allowed_repair:
        return TerminalTrigger.PLAN_NULL_AFTER_ALLOWED_REPAIR
    return None


@dataclass(frozen=True)
class TerminalReviewPolicy:
    arm: TerminalReviewArm
    harness_protocol_id: str
    experiment_arm_id: str
    terminal_review: bool
    max_verifier_rounds: int
    max_stage_attempts_per_round: int
    max_environment_steps: int
    effective_network_retry_limit: int = 0
    effective_tool_attempts_per_authorized_action: int = 1

    def __post_init__(self) -> None:
        expected = {
            TerminalReviewArm.H0: (
                "qwen3vl_harness_h0_legacy_single_round_v1",
                "v12_1-h0-legacy-single-round-base-0802",
                False,
                1,
                2,
                3,
                0,
                1,
            ),
            TerminalReviewArm.H1: (
                "qwen3vl_harness_h1_gtfree_terminal_review_v1",
                "v12_1-h1-gtfree-terminal-review-base-0802",
                True,
                2,
                2,
                3,
                0,
                1,
            ),
        }[self.arm]
        observed = (
            self.harness_protocol_id,
            self.experiment_arm_id,
            self.terminal_review,
            self.max_verifier_rounds,
            self.max_stage_attempts_per_round,
            self.max_environment_steps,
            self.effective_network_retry_limit,
            self.effective_tool_attempts_per_authorized_action,
        )
        if observed != expected:
            raise TerminalReviewContractError(
                f"{self.arm.value} policy differs from the frozen preregistration"
            )

    @classmethod
    def h0(cls) -> "TerminalReviewPolicy":
        return cls(
            arm=TerminalReviewArm.H0,
            harness_protocol_id="qwen3vl_harness_h0_legacy_single_round_v1",
            experiment_arm_id="v12_1-h0-legacy-single-round-base-0802",
            terminal_review=False,
            max_verifier_rounds=1,
            max_stage_attempts_per_round=2,
            max_environment_steps=3,
        )

    @classmethod
    def h1(cls) -> "TerminalReviewPolicy":
        return cls(
            arm=TerminalReviewArm.H1,
            harness_protocol_id="qwen3vl_harness_h1_gtfree_terminal_review_v1",
            experiment_arm_id="v12_1-h1-gtfree-terminal-review-base-0802",
            terminal_review=True,
            max_verifier_rounds=2,
            max_stage_attempts_per_round=2,
            max_environment_steps=3,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_id": FACTOR_ID,
            "arm": self.arm.value,
            "harness_protocol_id": self.harness_protocol_id,
            "experiment_arm_id": self.experiment_arm_id,
            "terminal_review": self.terminal_review,
            "max_verifier_rounds": self.max_verifier_rounds,
            "max_stage_attempts_per_round": self.max_stage_attempts_per_round,
            "max_environment_steps": self.max_environment_steps,
            "effective_network_retry_limit": self.effective_network_retry_limit,
            "effective_tool_attempts_per_authorized_action": (
                self.effective_tool_attempts_per_authorized_action
            ),
        }


@dataclass(frozen=True)
class TerminalReviewCounters:
    verifier_round_count: int
    stage_initial_attempt_count: int
    schema_repair_attempt_count: int
    network_retry_attempt_count: int
    tool_action_attempt_count: int
    tool_failure_count: int
    environment_step_count: int

    def __post_init__(self) -> None:
        for name in _NONNEGATIVE_COUNTER_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TerminalReviewContractError(
                    f"{name} must be a non-negative integer"
                )
        if self.tool_failure_count > self.tool_action_attempt_count:
            raise TerminalReviewContractError(
                "tool failures cannot exceed tool action attempts"
            )

    def to_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in _NONNEGATIVE_COUNTER_FIELDS}


@dataclass(frozen=True)
class BackendCallIdentity:
    stage: str
    region_id: str
    prompt_sha256: str

    def __post_init__(self) -> None:
        if not self.stage or not self.region_id:
            raise TerminalReviewContractError(
                "backend call identity requires a stage and region_id"
            )
        if _SHA256_RE.fullmatch(self.prompt_sha256) is None:
            raise TerminalReviewContractError("backend prompt SHA256 is malformed")


@dataclass(frozen=True)
class RoundTwoCallAudit:
    stage: str
    region_id: str
    attempt_kind: StageAttemptKind
    prompt_sha256: str
    raw_response: str
    parsed_output: Mapping[str, Any] | None = None
    visual_input_scope: str = "standard"
    network_retry_attempt_count: int = 0

    def __post_init__(self) -> None:
        BackendCallIdentity(self.stage, self.region_id, self.prompt_sha256)
        if not isinstance(self.raw_response, str):
            raise TerminalReviewContractError("raw_response must be retained as text")
        if not self.visual_input_scope:
            raise TerminalReviewContractError("visual_input_scope must be non-empty")
        if (
            isinstance(self.network_retry_attempt_count, bool)
            or not isinstance(self.network_retry_attempt_count, int)
            or self.network_retry_attempt_count < 0
        ):
            raise TerminalReviewContractError(
                "network_retry_attempt_count must be a non-negative integer"
            )

    @property
    def identity(self) -> BackendCallIdentity:
        return BackendCallIdentity(self.stage, self.region_id, self.prompt_sha256)

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_index": 2,
            "stage": self.stage,
            "region_id": self.region_id,
            "attempt_kind": self.attempt_kind.value,
            "prompt_sha256": self.prompt_sha256,
            "raw_response": self.raw_response,
            "parsed_output": (
                dict(self.parsed_output) if self.parsed_output is not None else None
            ),
            "visual_input_scope": self.visual_input_scope,
            "network_retry_attempt_count": self.network_retry_attempt_count,
        }


ValueT = TypeVar("ValueT")


@dataclass(frozen=True)
class TerminalReviewRequest(Generic[ValueT]):
    accepted_state_id: str
    round_one_value: ValueT
    trigger: TerminalTrigger | None
    counters: TerminalReviewCounters
    trigger_input_fields: tuple[str, ...] = ()
    round_one_call_identities: tuple[BackendCallIdentity, ...] = ()
    gt_free: bool = True
    round_two_already_used: bool = False
    round_two_unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.accepted_state_id:
            raise TerminalReviewContractError("accepted_state_id must be non-empty")
        if not isinstance(self.gt_free, bool) or not self.gt_free:
            raise TerminalReviewContractError(
                "terminal-review trigger inputs must be explicitly GT-free"
            )
        if not isinstance(self.round_two_already_used, bool):
            raise TerminalReviewContractError(
                "round_two_already_used must be a boolean"
            )
        if self.round_two_unavailable_reason not in {
            None,
            "no_environment_proposal_available_for_round_two",
            "no_eligible_plan_null_review_region",
        }:
            raise TerminalReviewContractError(
                "round_two_unavailable_reason is not a registered fail-closed reason"
            )
        if self.round_two_unavailable_reason is not None and self.trigger is None:
            raise TerminalReviewContractError(
                "round-two unavailability requires a registered terminal trigger"
            )
        if len(self.trigger_input_fields) != len(set(self.trigger_input_fields)):
            raise TerminalReviewContractError("trigger input fields must be unique")
        unknown = set(self.trigger_input_fields) - ALLOWED_TRIGGER_INPUTS
        forbidden = set(self.trigger_input_fields) & FORBIDDEN_TRIGGER_INPUTS
        if unknown or forbidden:
            detail = sorted(unknown | forbidden)
            raise TerminalReviewContractError(
                "terminal-review trigger contains forbidden/unregistered inputs: "
                f"{detail}"
            )


@dataclass(frozen=True)
class RoundTwoRequest:
    accepted_state_id: str
    trigger: TerminalTrigger
    round_index: int
    round_kind: RoundTwoKind
    selection_policy_version: str
    max_stage_attempts_per_stage: int
    remaining_environment_steps: int
    gt_free: bool
    allow_round_three: bool

    def __post_init__(self) -> None:
        if self.round_index != 2:
            raise TerminalReviewContractError("terminal review must be round_index=2")
        if not self.gt_free:
            raise TerminalReviewContractError("round two must be GT-free")
        if self.allow_round_three:
            raise TerminalReviewContractError("a third verifier round is forbidden")
        if self.max_stage_attempts_per_stage != 2:
            raise TerminalReviewContractError(
                "round two must retain initial plus at most one repair"
            )
        if self.remaining_environment_steps < 0:
            raise TerminalReviewContractError(
                "remaining environment step budget cannot be negative"
            )


@dataclass(frozen=True)
class RoundTwoObservation(Generic[ValueT]):
    outcome: RoundTwoOutcome
    value: ValueT | None
    safe_plan_candidate_count: int
    backend_calls: tuple[RoundTwoCallAudit, ...]
    completion_gate_passed: bool | None
    schema_valid: bool | None
    gt_free: bool = True
    fresh_backend_context: bool = True
    inherits_hidden_state: bool = False
    copied_round_one_raw_response: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.safe_plan_candidate_count, bool)
            or not isinstance(self.safe_plan_candidate_count, int)
            or self.safe_plan_candidate_count < 0
        ):
            raise TerminalReviewContractError(
                "safe_plan_candidate_count must be a non-negative integer"
            )
        if not self.gt_free:
            raise TerminalReviewContractError("round-two observation is not GT-free")
        if not self.fresh_backend_context:
            raise TerminalReviewContractError(
                "round two must use a fresh backend context"
            )
        if self.inherits_hidden_state:
            raise TerminalReviewContractError(
                "round two must not inherit model hidden state"
            )
        if self.copied_round_one_raw_response:
            raise TerminalReviewContractError(
                "round two must not copy the round-one raw response"
            )
        if self.outcome is RoundTwoOutcome.UNIQUE_SAFE_PLAN:
            if self.value is None or self.safe_plan_candidate_count != 1:
                raise TerminalReviewContractError(
                    "unique_safe_plan requires exactly one retained plan value"
                )
        elif self.outcome is RoundTwoOutcome.FINISH:
            if self.value is None or self.safe_plan_candidate_count != 0:
                raise TerminalReviewContractError(
                    "finish requires one terminal value and no safe action plan"
                )
            if self.completion_gate_passed is not True:
                raise TerminalReviewContractError(
                    "finish requires the unchanged completion gate"
                )
        elif self.value is not None:
            raise TerminalReviewContractError(
                "only finish or a unique safe plan may retain a review value"
            )
        if self.outcome is RoundTwoOutcome.SCHEMA_INVALID_AFTER_ALLOWED_REPAIR:
            if self.schema_valid is not False:
                raise TerminalReviewContractError(
                    "schema-invalid outcome must set schema_valid=false"
                )
        elif self.outcome is RoundTwoOutcome.BUDGET_EXHAUSTED:
            if self.schema_valid is not None:
                raise TerminalReviewContractError(
                    "budget-exhausted outcome must not claim schema validity"
                )
        elif self.schema_valid is not True:
            raise TerminalReviewContractError(
                "non-budget round-two outcomes require a schema-valid observation"
            )


@dataclass(frozen=True)
class TerminalReviewAudit:
    policy: TerminalReviewPolicy
    accepted_state_id: str
    trigger_input_fields: tuple[str, ...]
    eligible: bool
    trigger: TerminalTrigger | None
    round_two_invoked: bool
    round_two_kind: RoundTwoKind | None
    observed_outcome: RoundTwoOutcome | None
    effective_outcome: RoundTwoOutcome | None
    transition: TerminalTransition | None
    counters_before: TerminalReviewCounters
    counters_after: TerminalReviewCounters
    backend_calls: tuple[RoundTwoCallAudit, ...] = ()
    nonduplicate_prompt_or_region: bool | None = None
    contract_violation: str | None = None
    safe_plan_returned_to_existing_flow: bool = False
    gt_free: bool = True
    round_three_allowed: bool = False
    tool_executed_by_scheduler: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "factor_id": FACTOR_ID,
            "policy": self.policy.to_dict(),
            "accepted_state_id": self.accepted_state_id,
            "gt_free": self.gt_free,
            "trigger_input_fields": list(self.trigger_input_fields),
            "forbidden_trigger_inputs_present": sorted(
                set(self.trigger_input_fields) & FORBIDDEN_TRIGGER_INPUTS
            ),
            "eligible": self.eligible,
            "trigger": self.trigger.value if self.trigger else None,
            "round_two_invoked": self.round_two_invoked,
            "round_two_count": (
                self.counters_after.verifier_round_count
                - self.counters_before.verifier_round_count
            ),
            "round_index": 2 if self.round_two_invoked else None,
            "round_two_kind": (
                self.round_two_kind.value if self.round_two_kind else None
            ),
            "observed_outcome": (
                self.observed_outcome.value if self.observed_outcome else None
            ),
            "effective_outcome": (
                self.effective_outcome.value if self.effective_outcome else None
            ),
            "transition": self.transition.to_dict() if self.transition else None,
            "fresh_backend_context": (
                True if self.round_two_invoked else None
            ),
            "inherits_hidden_state": False,
            "copied_round_one_raw_response": False,
            "nonduplicate_prompt_or_region": self.nonduplicate_prompt_or_region,
            "round_three_allowed": self.round_three_allowed,
            "contract_violation": self.contract_violation,
            "safe_plan_returned_to_existing_flow": (
                self.safe_plan_returned_to_existing_flow
            ),
            "tool_executed_by_scheduler": self.tool_executed_by_scheduler,
            "counter_scope": dict(TERMINAL_REVIEW_COUNTER_SCOPE),
            "counters_before": self.counters_before.to_dict(),
            "counters_after": self.counters_after.to_dict(),
            "backend_calls": [item.to_dict() for item in self.backend_calls],
        }


@dataclass(frozen=True)
class TerminalReviewResult(Generic[ValueT]):
    value: ValueT | None
    passthrough: bool
    transition: TerminalTransition | None
    audit: TerminalReviewAudit


RoundTwoObserver = Callable[[RoundTwoRequest], RoundTwoObservation[ValueT]]


class TerminalReviewScheduler(Generic[ValueT]):
    """Identity H0 decorator and exactly-once H1 terminal-review scheduler."""

    def __init__(self, policy: TerminalReviewPolicy):
        self.policy = policy
        self._reviewed_state_ids: set[str] = set()
        self._active_state_ids: set[str] = set()

    def reset(self) -> None:
        """Start a new episode without retaining accepted-state review tokens."""

        if self._active_state_ids:
            raise TerminalReviewContractError(
                "cannot reset terminal-review tokens during an active review"
            )
        self._reviewed_state_ids.clear()

    def resolve(
        self,
        request: TerminalReviewRequest[ValueT],
        observe_round_two: RoundTwoObserver[ValueT],
    ) -> TerminalReviewResult[ValueT]:
        if self.policy.arm is TerminalReviewArm.H0:
            return self._passthrough(request)
        self._validate_budget_snapshot(request.counters)
        if request.trigger is None:
            return self._passthrough(request)

        trigger = request.trigger
        round_kind = _round_kind(trigger)
        if request.round_two_unavailable_reason is not None:
            return self._budget_exhausted_without_review(
                request,
                round_kind=round_kind,
                reason=request.round_two_unavailable_reason,
            )
        already_used = bool(
            request.round_two_already_used
            or request.accepted_state_id in self._reviewed_state_ids
            or request.accepted_state_id in self._active_state_ids
        )
        no_round_budget = (
            request.counters.verifier_round_count >= self.policy.max_verifier_rounds
        )
        if already_used or no_round_budget:
            reason = (
                "round_two_already_used_for_accepted_state"
                if already_used
                else "verifier_round_budget_exhausted_before_round_two"
            )
            return self._budget_exhausted_without_review(
                request, round_kind=round_kind, reason=reason
            )
        if request.counters.verifier_round_count != 1:
            raise TerminalReviewContractError(
                "an eligible H1 trigger must follow exactly one verifier round"
            )
        if not request.round_one_call_identities:
            raise TerminalReviewContractError(
                "eligible H1 review requires round-one call identities"
            )

        review_request = RoundTwoRequest(
            accepted_state_id=request.accepted_state_id,
            trigger=trigger,
            round_index=2,
            round_kind=round_kind,
            selection_policy_version=SELECTION_POLICY_VERSION,
            max_stage_attempts_per_stage=(
                self.policy.max_stage_attempts_per_round
            ),
            remaining_environment_steps=max(
                0,
                self.policy.max_environment_steps
                - request.counters.environment_step_count,
            ),
            gt_free=True,
            allow_round_three=False,
        )
        # Consume the state token before calling external observation code.  A
        # re-entrant attempt therefore cannot recursively create round three.
        self._reviewed_state_ids.add(request.accepted_state_id)
        self._active_state_ids.add(request.accepted_state_id)
        try:
            observation = observe_round_two(review_request)
        finally:
            self._active_state_ids.discard(request.accepted_state_id)
        if not isinstance(observation, RoundTwoObservation):
            raise TerminalReviewContractError(
                "round-two observer must return RoundTwoObservation"
            )

        self._validate_stage_attempts(observation.backend_calls)
        nonduplicate = self._validate_independence(
            request.round_one_call_identities,
            observation,
        )
        counters_after = _advance_counters(
            request.counters, observation.backend_calls
        )
        effective_outcome = observation.outcome
        if (
            observation.outcome is RoundTwoOutcome.UNIQUE_SAFE_PLAN
            and request.counters.environment_step_count
            >= self.policy.max_environment_steps
        ):
            effective_outcome = RoundTwoOutcome.BUDGET_EXHAUSTED
        transition = transition_for(trigger, effective_outcome)
        return_value = (
            observation.value
            if transition.next_state
            in {
                ReviewNextState.TERMINATE_CONFIRMED_FINISH,
                ReviewNextState.EXISTING_ENVIRONMENT_ACTION_AND_CANDIDATE_FLOW,
            }
            else None
        )
        safe_plan_returned = bool(
            transition.next_state
            is ReviewNextState.EXISTING_ENVIRONMENT_ACTION_AND_CANDIDATE_FLOW
            and return_value is not None
        )
        audit = TerminalReviewAudit(
            policy=self.policy,
            accepted_state_id=request.accepted_state_id,
            trigger_input_fields=request.trigger_input_fields,
            eligible=True,
            trigger=trigger,
            round_two_invoked=True,
            round_two_kind=round_kind,
            observed_outcome=observation.outcome,
            effective_outcome=effective_outcome,
            transition=transition,
            counters_before=request.counters,
            counters_after=counters_after,
            backend_calls=observation.backend_calls,
            nonduplicate_prompt_or_region=nonduplicate,
            contract_violation=(
                "environment_step_budget_exhausted_after_review"
                if effective_outcome is RoundTwoOutcome.BUDGET_EXHAUSTED
                and observation.outcome is RoundTwoOutcome.UNIQUE_SAFE_PLAN
                else None
            ),
            safe_plan_returned_to_existing_flow=safe_plan_returned,
        )
        return TerminalReviewResult(
            value=return_value,
            passthrough=False,
            transition=transition,
            audit=audit,
        )

    def _passthrough(
        self, request: TerminalReviewRequest[ValueT]
    ) -> TerminalReviewResult[ValueT]:
        audit = TerminalReviewAudit(
            policy=self.policy,
            accepted_state_id=request.accepted_state_id,
            trigger_input_fields=request.trigger_input_fields,
            eligible=False,
            trigger=request.trigger,
            round_two_invoked=False,
            round_two_kind=None,
            observed_outcome=None,
            effective_outcome=None,
            transition=None,
            counters_before=request.counters,
            counters_after=request.counters,
        )
        return TerminalReviewResult(
            value=request.round_one_value,
            passthrough=True,
            transition=None,
            audit=audit,
        )

    def _budget_exhausted_without_review(
        self,
        request: TerminalReviewRequest[ValueT],
        *,
        round_kind: RoundTwoKind,
        reason: str,
    ) -> TerminalReviewResult[ValueT]:
        if request.trigger is None:
            raise TerminalReviewContractError(
                "budget exhaustion requires a registered terminal trigger"
            )
        transition = transition_for(
            request.trigger, RoundTwoOutcome.BUDGET_EXHAUSTED
        )
        audit = TerminalReviewAudit(
            policy=self.policy,
            accepted_state_id=request.accepted_state_id,
            trigger_input_fields=request.trigger_input_fields,
            eligible=True,
            trigger=request.trigger,
            round_two_invoked=False,
            round_two_kind=round_kind,
            observed_outcome=None,
            effective_outcome=RoundTwoOutcome.BUDGET_EXHAUSTED,
            transition=transition,
            counters_before=request.counters,
            counters_after=request.counters,
            contract_violation=reason,
        )
        return TerminalReviewResult(
            value=None,
            passthrough=False,
            transition=transition,
            audit=audit,
        )

    def _validate_budget_snapshot(self, counters: TerminalReviewCounters) -> None:
        if counters.verifier_round_count > self.policy.max_verifier_rounds:
            raise TerminalReviewContractError("verifier round counter exceeds budget")
        if counters.environment_step_count > self.policy.max_environment_steps:
            raise TerminalReviewContractError(
                "environment step counter exceeds budget"
            )

    def _validate_stage_attempts(
        self, calls: tuple[RoundTwoCallAudit, ...]
    ) -> None:
        grouped: dict[tuple[str, str], list[StageAttemptKind]] = {}
        for call in calls:
            if (
                call.network_retry_attempt_count
                > self.policy.effective_network_retry_limit
            ):
                raise TerminalReviewContractError(
                    "round two exceeded the frozen network retry limit"
                )
            grouped.setdefault((call.stage, call.region_id), []).append(
                call.attempt_kind
            )
        for key, attempts in grouped.items():
            if len(attempts) > self.policy.max_stage_attempts_per_round:
                raise TerminalReviewContractError(
                    f"round-two stage {key!r} exceeded its attempt budget"
                )
            if not attempts or attempts[0] is not StageAttemptKind.INITIAL:
                raise TerminalReviewContractError(
                    f"round-two stage {key!r} must begin with one initial attempt"
                )
            if attempts.count(StageAttemptKind.INITIAL) != 1:
                raise TerminalReviewContractError(
                    f"round-two stage {key!r} repeated its initial attempt"
                )
            if attempts.count(StageAttemptKind.REPAIR) > 1:
                raise TerminalReviewContractError(
                    f"round-two stage {key!r} used more than one repair"
                )

    @staticmethod
    def _validate_independence(
        round_one_calls: tuple[BackendCallIdentity, ...],
        observation: RoundTwoObservation[Any],
    ) -> bool | None:
        initial_calls = tuple(
            item.identity
            for item in observation.backend_calls
            if item.attempt_kind is StageAttemptKind.INITIAL
        )
        if (
            observation.outcome is RoundTwoOutcome.BUDGET_EXHAUSTED
            and not initial_calls
        ):
            return None
        if not initial_calls:
            raise TerminalReviewContractError(
                "a non-budget round two must retain an initial backend call"
            )
        prior_prompt_regions = {
            (item.region_id, item.prompt_sha256) for item in round_one_calls
        }
        nonduplicate = any(
            (item.region_id, item.prompt_sha256) not in prior_prompt_regions
            for item in initial_calls
        )
        if not nonduplicate:
            raise TerminalReviewContractError(
                "round two duplicates every round-one prompt/region identity"
            )
        if (
            observation.outcome
            is RoundTwoOutcome.SCHEMA_INVALID_AFTER_ALLOWED_REPAIR
            and not any(
                item.attempt_kind is StageAttemptKind.REPAIR
                for item in observation.backend_calls
            )
        ):
            raise TerminalReviewContractError(
                "schema-invalid-after-repair requires one recorded repair attempt"
            )
        return True


def _round_kind(trigger: TerminalTrigger) -> RoundTwoKind:
    return {
        TerminalTrigger.PROVISIONAL_FINISH: RoundTwoKind.FINISH_CONFIRMATION,
        TerminalTrigger.PLAN_NULL_AFTER_ALLOWED_REPAIR: (
            RoundTwoKind.PLAN_NULL_ALTERNATIVE
        ),
    }[trigger]


def _advance_counters(
    before: TerminalReviewCounters,
    calls: tuple[RoundTwoCallAudit, ...],
) -> TerminalReviewCounters:
    return TerminalReviewCounters(
        verifier_round_count=before.verifier_round_count + 1,
        stage_initial_attempt_count=(
            before.stage_initial_attempt_count
            + sum(item.attempt_kind is StageAttemptKind.INITIAL for item in calls)
        ),
        schema_repair_attempt_count=(
            before.schema_repair_attempt_count
            + sum(item.attempt_kind is StageAttemptKind.REPAIR for item in calls)
        ),
        network_retry_attempt_count=(
            before.network_retry_attempt_count
            + sum(item.network_retry_attempt_count for item in calls)
        ),
        tool_action_attempt_count=before.tool_action_attempt_count,
        tool_failure_count=before.tool_failure_count,
        environment_step_count=before.environment_step_count,
    )


__all__ = [
    "ALLOWED_TRIGGER_INPUTS",
    "BackendCallIdentity",
    "FACTOR_ID",
    "FORBIDDEN_TRIGGER_INPUTS",
    "ReviewDecision",
    "ReviewNextState",
    "RoundTwoCallAudit",
    "RoundTwoKind",
    "RoundTwoObservation",
    "RoundTwoOutcome",
    "RoundTwoRequest",
    "SCHEMA_VERSION",
    "SELECTION_POLICY_VERSION",
    "PLAN_NULL_UNREVIEWED_PROMPT_VERSION",
    "PLAN_NULL_ALTERNATIVE_PROMPT_VERSION",
    "StageAttemptKind",
    "TerminalReviewArm",
    "TerminalReviewAudit",
    "TerminalReviewContractError",
    "TerminalReviewCounters",
    "TerminalReviewPolicy",
    "TerminalReviewRequest",
    "TerminalReviewResult",
    "TerminalReviewScheduler",
    "TERMINAL_REVIEW_COUNTER_SCOPE",
    "TerminalTransition",
    "TerminalTrigger",
    "ToolActionAuthorization",
    "TRANSITION_TABLE",
    "select_terminal_trigger",
    "transition_for",
]
