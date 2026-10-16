"""Typed, stage-oriented protocol for GT-free change verification.

The Environment owns geometry and editability.  A model is only asked to select
existing region IDs and make local semantic decisions. Keeping these records
typed prevents a model from silently changing a region id, coordinate frame,
or mask fact between diagnosis and programmatic action resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Protocol, Sequence

from .state import ChangeState

StageName = Literal[
    "audit",
    "select",
    "evidence",
    "diagnosis",
    "candidate_evidence",
    "decision",
    "direct",
]

ERROR_TYPES = {
    "none",
    "false_positive_change",
    "false_negative",
    "mixed_error",
    "uncertain_region",
}
TARGET_VIEWS = ("t1", "t2")
ACTIONS = {"positive_point", "negative_point", "box", "finish"}
COMPARISONS = {"initial", "better", "worse", "unchanged", "uncertain"}
RGB_STATES = {"building", "background", "mixed", "uncertain"}
AUDIT_KINDS = {
    "present",
    "missing",
    "scan",
    "delta_added",
    "delta_removed",
    "mixed",
}
AUDIT_STATUSES = {"pass", "fail", "not_applicable", "uncertain"}
AUDIT_CHECKS = (
    "evidence_sufficient",
    "target_class_only",
    "white_pixels_supported",
    "boundary_alignment",
    "internal_holes_absent",
    "changed_object_extent_complete",
    "fragment_artifacts_absent",
)
FALSE_POSITIVE_AUDIT_CHECKS = {
    "target_class_only",
    "white_pixels_supported",
    "boundary_alignment",
    "fragment_artifacts_absent",
}
FALSE_NEGATIVE_AUDIT_CHECKS = {
    "internal_holes_absent",
    "changed_object_extent_complete",
}
ATOMIC_EVIDENCE_GROUNDING_ERROR_MARKERS = (
    "atomic audit evidence copied schema boilerplate",
    "atomic audit evidence copied an image/schema placeholder",
    "atomic audit evidence is a JSON/numeric/coordinate placeholder",
    "atomic audit evidence is an underscore-delimited label",
    "atomic audit evidence relies on mask/presentation color",
)


class StageProtocolError(ValueError):
    """Raised when a model response cannot be safely used by the runtime."""


def is_atomic_evidence_grounding_error(detail: str) -> bool:
    """Return whether an audit failure is isolated to its evidence string."""

    return any(
        marker in str(detail)
        for marker in ATOMIC_EVIDENCE_GROUNDING_ERROR_MARKERS
    )


@dataclass(frozen=True)
class EvidenceRecord:
    """Authoritative Environment facts for one proposed region."""

    region_id: str
    audit_kind: str
    change_mask_state: Literal["white", "black", "mixed"]
    temporal_difference_state: Literal["present", "absent", "mixed"]
    box_normalized_1000: tuple[int, int, int, int]
    component_seed_normalized_1000: tuple[int, int]
    t1_mask_pixels: int = 0
    t2_mask_pixels: int = 0
    component_area: int = 0
    editable_seed_white: Mapping[str, bool] = field(default_factory=dict)
    allowed_actions: tuple[str, ...] = ()
    transition_mask_facts: Mapping[str, int | bool] = field(default_factory=dict)
    proposal_version: str | None = None

    def __post_init__(self) -> None:
        if not self.region_id:
            raise StageProtocolError("region_id must be non-empty")
        if self.audit_kind not in AUDIT_KINDS:
            raise StageProtocolError(f"unsupported audit_kind: {self.audit_kind!r}")
        if self.change_mask_state not in {"white", "black", "mixed"}:
            raise StageProtocolError("unsupported change_mask_state")
        if self.temporal_difference_state not in {"present", "absent", "mixed"}:
            raise StageProtocolError("unsupported temporal_difference_state")
        _validate_box(self.box_normalized_1000)
        _validate_point(self.component_seed_normalized_1000)
        if any(value < 0 for value in (self.t1_mask_pixels, self.t2_mask_pixels, self.component_area)):
            raise StageProtocolError("mask pixel counts must be non-negative")
        for view, value in self.editable_seed_white.items():
            if view not in TARGET_VIEWS or not isinstance(value, bool):
                raise StageProtocolError("editable_seed_white must map t1/t2 to bool")
        for action in self.allowed_actions:
            if action not in ACTIONS - {"finish"}:
                raise StageProtocolError(f"unsupported allowed action: {action!r}")
        for key, value in self.transition_mask_facts.items():
            valid_value = isinstance(value, bool) or (
                isinstance(value, int) and value >= 0
            )
            if not isinstance(key, str) or not valid_value:
                raise StageProtocolError(
                    "transition_mask_facts must map strings to non-negative integers or booleans"
                )
        if self.proposal_version is not None and not self.proposal_version.strip():
            raise StageProtocolError("proposal_version must be a non-empty string")

    @classmethod
    def from_proposal(cls, proposal: Mapping[str, Any]) -> "EvidenceRecord":
        """Convert an Environment proposal without allowing model-authored facts."""

        region_id = str(proposal["region_id"])
        proposal_version: str | None = None
        if "proposal_version" in proposal:
            raw_proposal_version = proposal["proposal_version"]
            if (
                not isinstance(raw_proposal_version, str)
                or not raw_proposal_version.strip()
            ):
                raise StageProtocolError(
                    "proposal_version must be an explicit non-empty string"
                )
            proposal_version = raw_proposal_version
        seed = proposal.get("component_seed_normalized") or proposal.get(
            "component_seed_normalized_1000"
        )
        box = proposal.get("box_normalized") or proposal.get("box_normalized_1000")
        if seed is None or box is None:
            raise StageProtocolError(f"proposal {region_id} lacks canonical geometry")
        if "audit_kind" in proposal:
            audit_kind = str(proposal["audit_kind"])
        elif proposal.get("effect_kind") == "added":
            audit_kind = "delta_added"
        elif proposal.get("effect_kind") == "removed":
            audit_kind = "delta_removed"
        else:
            audit_kind = "mixed"
        change_state = (
            "white"
            if audit_kind in {"present", "delta_added"}
            else "black"
            if audit_kind in {"missing", "scan", "delta_removed"}
            else "mixed"
        )
        facts = proposal.get("mask_facts", {})
        editable = {
            view: bool(proposal.get(f"component_seed_{view}_mask_white", False))
            for view in TARGET_VIEWS
        }
        allowed = tuple(
            action
            for action in ("positive_point", "negative_point", "box")
            if action == "box"
            or any(
                editable.get(view, False) if action == "negative_point" else not editable.get(view, False)
                for view in TARGET_VIEWS
            )
        )
        return cls(
            region_id=region_id,
            audit_kind=audit_kind,
            change_mask_state=change_state,
            temporal_difference_state=(
                "present"
                if int(proposal.get("temporal_difference_pixels", 0)) > 0
                else "absent"
            ),
            box_normalized_1000=tuple(int(value) for value in box),
            component_seed_normalized_1000=tuple(int(value) for value in seed),
            t1_mask_pixels=int(proposal.get("t1_mask_pixels", facts.get("t1_mask_pixels", 0))),
            t2_mask_pixels=int(proposal.get("t2_mask_pixels", facts.get("t2_mask_pixels", 0))),
            component_area=int(proposal.get("component_area", 0)),
            editable_seed_white=editable,
            allowed_actions=allowed,
            transition_mask_facts={
                str(key): value
                for key, value in proposal.get("transition_mask_facts", {}).items()
                if isinstance(value, (int, bool))
            },
            proposal_version=proposal_version,
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "region_id": self.region_id,
            "audit_kind": self.audit_kind,
            "change_mask_state": self.change_mask_state,
            "temporal_difference_state": self.temporal_difference_state,
            "box_normalized_1000": list(self.box_normalized_1000),
            "component_seed_normalized_1000": list(self.component_seed_normalized_1000),
            "t1_mask_pixels": self.t1_mask_pixels,
            "t2_mask_pixels": self.t2_mask_pixels,
            "component_area": self.component_area,
            "editable_seed_white": dict(self.editable_seed_white),
            "allowed_actions": list(self.allowed_actions),
            "transition_mask_facts": dict(self.transition_mask_facts),
        }
        # Preserve the frozen v12.1 H0 prompt bytes for ordinary proposals.
        # Only the H1 empty-mask scan surface carries a proposal version.
        if self.proposal_version is not None:
            result["proposal_version"] = self.proposal_version
        return result


@dataclass(frozen=True)
class EvidenceJudgment:
    region_id: str
    t1_state: str
    t2_state: str
    evidence_quality: Literal["clear", "ambiguous", "insufficient"]
    change_mask_state: str | None = None
    mask_assessment: str | None = None
    evidence: str = ""
    screening_hypothesis: str | None = None
    screening_resolution: str | None = None

    @property
    def confidence(self) -> float:
        """Compatibility score derived by runtime, never authored by the model."""

        return 1.0 if self.evidence_quality == "clear" else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "t1_state": self.t1_state,
            "t2_state": self.t2_state,
            "evidence_quality": self.evidence_quality,
            "change_mask_state": self.change_mask_state,
            "mask_assessment": self.mask_assessment,
            "evidence": self.evidence,
            "screening_hypothesis": self.screening_hypothesis,
            "screening_resolution": self.screening_resolution,
        }


@dataclass(frozen=True)
class AuditChecklist:
    """Discrete local mask audit used to derive diagnosis and quality."""

    evidence_sufficient: str
    target_class_only: str
    white_pixels_supported: str
    boundary_alignment: str
    internal_holes_absent: str
    changed_object_extent_complete: str
    fragment_artifacts_absent: str
    evidence: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in AUDIT_CHECKS:
            status = getattr(self, name)
            if status not in AUDIT_STATUSES:
                raise StageProtocolError(
                    f"audit checklist {name} has unsupported status: {status!r}"
                )
        if self.evidence_sufficient == "not_applicable":
            raise StageProtocolError(
                "audit checklist evidence_sufficient cannot be not_applicable"
            )
        if self.evidence:
            if set(self.evidence) != set(AUDIT_CHECKS):
                raise StageProtocolError(
                    "grounded audit evidence must cover every checklist item"
                )
            if any(not str(value).strip() for value in self.evidence.values()):
                raise StageProtocolError(
                    "grounded audit evidence must be non-empty for every checklist item"
                )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "AuditChecklist":
        if set(payload) != set(AUDIT_CHECKS):
            raise StageProtocolError(
                "audit checklist must contain exactly: "
                + ", ".join(AUDIT_CHECKS)
            )
        statuses: dict[str, str] = {}
        evidence: dict[str, str] = {}
        grounded = any(isinstance(payload[name], Mapping) for name in AUDIT_CHECKS)
        for name in AUDIT_CHECKS:
            value = payload[name]
            if grounded:
                if not isinstance(value, Mapping) or set(value) != {"status", "evidence"}:
                    raise StageProtocolError(
                        f"grounded audit item {name} must contain status and evidence"
                    )
                statuses[name] = str(value["status"])
                evidence[name] = str(value["evidence"])
            else:
                statuses[name] = str(value)
        return cls(**statuses, evidence=evidence)

    def to_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in AUDIT_CHECKS}

    def evidence_dict(self) -> dict[str, str]:
        return {name: str(self.evidence.get(name, "")) for name in AUDIT_CHECKS}

    @property
    def quality_score(self) -> float:
        """Return a reproducible pass ratio; abstention is scored as zero."""

        if (
            self.evidence_sufficient != "pass"
            or "uncertain" in self.to_dict().values()
        ):
            return 0.0
        scorable = [
            status
            for name, status in self.to_dict().items()
            if name != "evidence_sufficient" and status != "not_applicable"
        ]
        if not scorable:
            return 0.0
        return sum(status == "pass" for status in scorable) / len(scorable)

    @property
    def error_type(self) -> str:
        """Derive the coarse executor diagnosis from failed audit dimensions."""

        values = self.to_dict()
        if self.evidence_sufficient != "pass" or "uncertain" in values.values():
            return "uncertain_region"
        applicable = {
            name: status
            for name, status in values.items()
            if name != "evidence_sufficient" and status != "not_applicable"
        }
        if not applicable:
            return "uncertain_region"
        failed = {name for name, status in applicable.items() if status == "fail"}
        false_positive = bool(failed & FALSE_POSITIVE_AUDIT_CHECKS)
        false_negative = bool(failed & FALSE_NEGATIVE_AUDIT_CHECKS)
        if false_positive and false_negative:
            return "mixed_error"
        if false_positive:
            return "false_positive_change"
        if false_negative:
            return "false_negative"
        return "none"


@dataclass(frozen=True)
class Diagnosis:
    region_id: str
    error_type: str
    target_view: str | None
    audit_checklist: AuditChecklist | None = None
    summary: str = ""

    def __post_init__(self) -> None:
        if self.error_type not in ERROR_TYPES:
            raise StageProtocolError(f"unsupported error_type: {self.error_type!r}")
        if self.target_view is not None and self.target_view not in TARGET_VIEWS:
            raise StageProtocolError("target_view must be t1, t2, or null")
        if self.error_type == "none" and self.target_view is not None:
            raise StageProtocolError("none diagnosis must not target a view")
        if (
            self.audit_checklist is not None
            and self.error_type != self.audit_checklist.error_type
        ):
            raise StageProtocolError(
                "diagnosis error_type must equal the runtime-derived audit checklist result"
            )

    @property
    def confidence(self) -> float:
        """Compatibility score computed from the binary checklist."""

        return self.audit_checklist.quality_score if self.audit_checklist else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "error_type": self.error_type,
            "target_view": self.target_view,
            "audit_checklist": (
                self.audit_checklist.to_dict() if self.audit_checklist else None
            ),
            "audit_evidence": (
                self.audit_checklist.evidence_dict() if self.audit_checklist else None
            ),
            "summary": self.summary,
        }


@dataclass(frozen=True)
class ActionPlan:
    region_id: str | None
    action: str
    target_view: str | None
    coordinate_normalized_1000: tuple[int, int] | None = None
    box_normalized_1000: tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise StageProtocolError(f"unsupported action: {self.action!r}")
        if self.action == "finish":
            if self.region_id is not None or self.target_view is not None:
                raise StageProtocolError("finish plan must not select a region or view")
            if self.coordinate_normalized_1000 is not None or self.box_normalized_1000 is not None:
                raise StageProtocolError("finish plan must not contain geometry")
            return
        if self.region_id is None or self.target_view not in TARGET_VIEWS:
            raise StageProtocolError("tool plan requires region_id and target_view")
        if self.action in {"positive_point", "negative_point"}:
            if self.coordinate_normalized_1000 is None or self.box_normalized_1000 is not None:
                raise StageProtocolError("point plan requires only coordinate_normalized_1000")
            _validate_point(self.coordinate_normalized_1000)
        if self.action == "box":
            if self.box_normalized_1000 is None or self.coordinate_normalized_1000 is not None:
                raise StageProtocolError("box plan requires only box_normalized_1000")
            _validate_box(self.box_normalized_1000)


@dataclass(frozen=True)
class Decision:
    comparison: str
    quality_score: float
    progress_score: float
    accept: bool
    stop: bool
    feedback: str = ""

    def __post_init__(self) -> None:
        if self.comparison not in COMPARISONS:
            raise StageProtocolError(f"unsupported comparison: {self.comparison!r}")
        if not 0 <= self.quality_score <= 1:
            raise StageProtocolError("quality_score must be in [0,1]")
        if not -1 <= self.progress_score <= 1:
            raise StageProtocolError("progress_score must be in [-1,1]")
        if self.stop and not self.accept:
            raise StageProtocolError("stop requires accept=true")


@dataclass(frozen=True)
class TransitionAssessment:
    """Runtime-derived candidate effects from local model-observed RGB evidence."""

    intended_error_improved: bool
    introduced_false_positive: bool
    introduced_false_negative: bool
    boundary_or_artifact_worsened: bool
    evidence_sufficient: bool = True
    evidence: str = ""
    source: str = "runtime_candidate_evidence"

    @property
    def introduced_harm(self) -> bool:
        return bool(
            self.introduced_false_positive
            or self.introduced_false_negative
            or self.boundary_or_artifact_worsened
        )

    @property
    def comparison(self) -> str:
        if not self.evidence_sufficient:
            return "uncertain"
        if self.intended_error_improved and not self.introduced_harm:
            return "better"
        if self.introduced_harm and not self.intended_error_improved:
            return "worse"
        if not self.intended_error_improved and not self.introduced_harm:
            return "unchanged"
        return "uncertain"


@dataclass(frozen=True)
class StageTrace:
    """Serializable intermediate state retained for audit and replay."""

    mode: Literal["initial", "candidate"]
    evidence: tuple[EvidenceRecord, ...]
    audited_region_ids: tuple[str, ...] = ()
    judgments: tuple[EvidenceJudgment, ...] = ()
    diagnoses: tuple[Diagnosis, ...] = ()
    plan: ActionPlan | None = None
    decision: Decision | None = None
    transition_assessment: TransitionAssessment | None = None
    state_completion_gate_passed: bool | None = None
    state_completion_gate_reason: str | None = None
    replan_evidence: tuple[EvidenceRecord, ...] = ()
    replan_audited_region_ids: tuple[str, ...] = ()
    replan_judgments: tuple[EvidenceJudgment, ...] = ()
    replan_diagnoses: tuple[Diagnosis, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "evidence": [item.to_dict() for item in self.evidence],
            "audited_region_ids": list(self.audited_region_ids),
            "judgments": [item.to_dict() for item in self.judgments],
            "diagnoses": [item.to_dict() for item in self.diagnoses],
            "plan": self.plan.__dict__ if self.plan else None,
            "decision": self.decision.__dict__ if self.decision else None,
            "transition_assessment": (
                self.transition_assessment.__dict__
                if self.transition_assessment
                else None
            ),
            "state_completion_gate_passed": self.state_completion_gate_passed,
            "state_completion_gate_reason": self.state_completion_gate_reason,
            "replan_evidence": [item.to_dict() for item in self.replan_evidence],
            "replan_audited_region_ids": list(self.replan_audited_region_ids),
            "replan_judgments": [item.to_dict() for item in self.replan_judgments],
            "replan_diagnoses": [item.to_dict() for item in self.replan_diagnoses],
        }


class StageBackend(Protocol):
    """Model-independent interface used by local and hosted MLLM backends."""

    def generate_stage(
        self,
        stage: StageName,
        state: ChangeState,
        payload: Mapping[str, Any],
        previous_state: ChangeState | None = None,
    ) -> Mapping[str, Any]: ...


def _validate_point(point: Sequence[int]) -> None:
    if len(point) != 2 or any(not isinstance(value, int) for value in point):
        raise StageProtocolError("normalized point must contain two integers")
    if any(value < 0 or value > 1000 for value in point):
        raise StageProtocolError("normalized point must be within [0,1000]")


def _validate_box(box: Sequence[int]) -> None:
    if len(box) != 4 or any(not isinstance(value, int) for value in box):
        raise StageProtocolError("normalized box must contain four integers")
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
        raise StageProtocolError("normalized box must be ordered and within [0,1000]")


def _validate_confidence(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
        raise StageProtocolError("confidence must be in [0,1]")
