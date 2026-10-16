"""Region-grounded rich Qwen diagnosis and candidate verification."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from .omniovcd_adapter import connected_components
from ..coordinates import pixel_box_to_normalized, pixel_point_to_normalized
from ..state import AgentAction, ChangeState, VerifierOutput
from ..verifier_regions import attach_verifier_regions


@dataclass(frozen=True)
class _RegionJudgment:
    region_id: str
    change_mask_state: str
    t1_state: str
    t2_state: str
    verdict: str
    target_view: str | None
    feedback: str
    suggested_action: str | None = None
    confidence: float = 0.0
    severity: float = 0.0


@dataclass(frozen=True)
class _EffectJudgment:
    region_id: str
    delta_kind: str
    t1_state: str
    t2_state: str
    effect: str
    target_view: str | None = None
    suggested_action: str | None = None
    confidence: float = 0.0
    severity: float = 0.0
    feedback: str = ""


@dataclass(frozen=True)
class _SynthesisDecision:
    quality_score: float
    progress_score: float
    comparison: str
    error_type: str
    target_view: str | None
    region_id: str | None
    suggested_action: str
    feedback: str


class Qwen3VLZeroShotVerifier:
    """Let Qwen diagnose local errors and synthesize the corrective decision.

    Runtime code owns proposal geometry, coverage, parsing, locality, rollback, and
    cache identity. Qwen owns FP/FN/mixed diagnosis, quality/progress assessment,
    candidate comparison, and the next corrective action.
    """

    SCHEMA_VERSION = "tight_rgb_mask_occupancy_coherent_synthesis_v14"
    CANDIDATE_EVIDENCE_MODES = ("rich_delta_diagnosis", "global_synthesis")
    ERROR_TYPES = {
        "none",
        "false_positive_change",
        "false_negative",
        "mixed_error",
        "uncertain_region",
    }
    COMPARISONS = {"better", "worse", "unchanged", "uncertain"}
    REGION_VERDICTS = {
        "true_change",
        "correct_unchanged",
        "false_positive",
        "false_negative",
        "mixed",
        "uncertain",
    }
    ACTIONS = {"positive_point", "negative_point", "box", "finish"}
    RGB_STATES = {"building", "background", "mixed", "uncertain"}
    EFFECT_LABELS = {
        "added_true_change",
        "added_false_change",
        "removed_false_positive",
        "removed_true_change",
        "mixed",
        "uncertain",
    }
    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        max_new_tokens: int = 1024,
        accept_threshold: float = 0.82,
        max_retries: int = 2,
        do_sample: bool = False,
        repetition_penalty: float = 1.05,
        **legacy_localization_options: Any,
    ):
        if not 0 <= accept_threshold <= 1:
            raise ValueError("accept_threshold must be in [0, 1]")
        if max_retries < 1:
            raise ValueError("max_retries must be positive")
        if repetition_penalty < 1:
            raise ValueError("repetition_penalty must be at least 1")
        self.model = model
        self.processor = processor
        self.max_new_tokens = max_new_tokens
        # Initial finish requires both Qwen's no-error diagnosis and this quality floor.
        # Candidate acceptance is based on Qwen's direct pairwise comparison.
        self.accept_threshold = accept_threshold
        self.max_retries = max_retries
        self.do_sample = bool(do_sample)
        self.repetition_penalty = float(repetition_penalty)
        self.legacy_localization_options = dict(legacy_localization_options)
        self.last_evidence: dict[str, Any] = {}
        self._last_valid_output: VerifierOutput | None = None
        self._candidate_cache: dict[
            str, tuple[VerifierOutput, dict[str, Any]]
        ] = {}

    def reset(self) -> None:
        self._last_valid_output = None
        self.last_evidence = {}
        self._candidate_cache = {}

    def on_candidate_rejected(self, previous_feedback: VerifierOutput) -> None:
        self._last_valid_output = (
            previous_feedback if previous_feedback.verifier_valid else None
        )

    def verify(
        self,
        state: ChangeState,
        previous_score: float | None,
        previous_action: AgentAction | None,
        previous_state: ChangeState | None = None,
    ) -> VerifierOutput:
        if previous_state is not None:
            return self._verify_candidate(
                state, previous_score, previous_action, previous_state
            )

        proposals = state.evidence.get("verifier_region_proposals")
        if not isinstance(proposals, list):
            # Standalone verifier calls (unit tests/smokes) still get the exact same
            # deterministic proposal builder. Normal runtime attaches them in Env.
            proposals = attach_verifier_regions(state, previous_state)
        mask_facts = dict(state.evidence.get("verifier_mask_facts", {}))

        if not proposals:
            return self._invalid_output(
                ["no mask-derived proposal is available for a reliable diagnosis"],
                region_attempts=[],
                pairwise_attempts=[],
                previous_action=previous_action,
            )

        judgments, region_attempts, region_errors = self._run_region_batches(
            state, proposals, mask_facts
        )
        if judgments is None:
            return self._invalid_output(
                region_errors,
                region_attempts=region_attempts,
                pairwise_attempts=[],
                previous_action=previous_action,
            )

        decision, synthesis_attempts, synthesis_errors = self._run_synthesis(
            state,
            None,
            previous_action,
            proposals,
            mask_facts,
            judgments,
            initial=True,
        )
        if decision is None:
            return self._invalid_output(
                region_errors + synthesis_errors,
                region_attempts=region_attempts,
                pairwise_attempts=synthesis_attempts,
                previous_action=previous_action,
            )
        output = self._output_from_synthesis(
            decision, proposals, previous_score, previous_action
        )
        self._last_valid_output = output
        self.last_evidence = {
            "type": "qwen3vl_rich_region_zero_shot",
            "decision_mode": "qwen_region_diagnosis_then_global_synthesis",
            "gt_available": False,
            "verifier_valid": True,
            "localization_valid": output.localization_valid,
            "mask_facts": mask_facts,
            "region_proposals": proposals,
            "region_judgments": [
                {
                    "region_id": item.region_id,
                    "change_mask_state": item.change_mask_state,
                    "t1_state": item.t1_state,
                    "t2_state": item.t2_state,
                    "verdict": item.verdict,
                    "target_view": item.target_view,
                    "feedback": item.feedback,
                    "suggested_action": item.suggested_action,
                    "confidence": item.confidence,
                    "severity": item.severity,
                }
                for item in judgments
            ],
            "region_attempts": region_attempts,
            "synthesis_attempts": synthesis_attempts,
            "comparison": decision.comparison,
            "synthesis_decision": self._decision_dict(decision),
            "validation_errors": region_errors + synthesis_errors,
        }
        return output

    def _verify_candidate(
        self,
        state: ChangeState,
        previous_score: float | None,
        previous_action: AgentAction | None,
        previous_state: ChangeState,
    ) -> VerifierOutput:
        proposals = list(state.evidence.get("verifier_region_proposals", []))
        facts = dict(state.evidence.get("verifier_mask_facts", {}))
        fingerprint = self._candidate_fingerprint(
            state, previous_state, previous_action, proposals, facts
        )
        cached = self._candidate_cache.get(fingerprint)
        if cached is not None:
            output, evidence = cached
            self.last_evidence = copy.deepcopy(evidence)
            self.last_evidence["cache_hit"] = True
            self.last_evidence["reused_from_step"] = evidence.get("decision_step")
            self._last_valid_output = output if output.verifier_valid else None
            return output

        if self._states_identical(state, previous_state):
            previous = self._last_valid_output
            output = VerifierOutput(
                quality_score=previous.quality_score if previous else previous_score,
                progress_score=0.0,
                score_delta=0.0,
                comparison="unchanged",
                error_type=previous.error_type if previous else "uncertain_region",
                target_view=(
                    previous.target_view
                    if previous
                    else previous_action.target_view if previous_action else "t2"
                ),
                error_region=previous.error_region if previous else None,
                suggested_action=previous.suggested_action if previous else None,
                feedback="Candidate masks are identical to the accepted state.",
                accept=bool(previous and previous.accept),
                verifier_valid=True,
                localization_valid=bool(previous and previous.localization_valid),
                stop=bool(previous and previous.stop),
            )
            self.last_evidence = {
                "type": "qwen3vl_rich_delta_zero_shot",
                "decision_mode": "programmatic_identical_state",
                "candidate_fingerprint": fingerprint,
                "decision_key": fingerprint,
                "decision_step": state.step_index,
                "cache_hit": False,
                "comparison": "unchanged",
                "effect_attempts": [],
                "validation_errors": [],
                "gt_available": False,
                "verifier_valid": True,
                "localization_valid": output.localization_valid,
            }
            self._cache_candidate(fingerprint, output)
            return output

        effect_errors: list[str] = []
        effect_attempts: list[dict[str, Any]] = []
        judgments: tuple[_EffectJudgment, ...] | None = None
        uncovered = int(facts.get("candidate_delta_uncovered_pixels", 0))
        if uncovered:
            effect_errors.append(
                f"candidate delta has {uncovered} pixels outside the compact proposal set"
            )
        elif not proposals:
            effect_errors.append("candidate delta has no auditable proposal")
        else:
            judgments, effect_attempts, batch_errors = self._run_effect_batches(
                state, previous_state, previous_action, proposals, facts
            )
            effect_errors.extend(batch_errors)

        if judgments is None:
            output = self._invalid_output(
                effect_errors,
                region_attempts=[],
                pairwise_attempts=[],
                previous_action=previous_action,
            )
            self.last_evidence.update(
                {
                    "type": "qwen3vl_rich_delta_zero_shot",
                    "decision_mode": "qwen_delta_diagnosis_then_global_synthesis",
                    "candidate_fingerprint": fingerprint,
                    "decision_key": fingerprint,
                    "decision_step": state.step_index,
                    "cache_hit": False,
                    "effect_attempts": effect_attempts,
                    "mask_facts": facts,
                    "region_proposals": proposals,
                }
            )
            self._cache_candidate(fingerprint, output)
            return output

        decision, synthesis_attempts, synthesis_errors = self._run_synthesis(
            state,
            previous_state,
            previous_action,
            proposals,
            facts,
            judgments,
            initial=False,
        )
        if decision is None:
            output = self._invalid_output(
                effect_errors + synthesis_errors,
                region_attempts=effect_attempts,
                pairwise_attempts=synthesis_attempts,
                previous_action=previous_action,
            )
            self.last_evidence.update(
                {
                    "type": "qwen3vl_rich_delta_zero_shot",
                    "candidate_fingerprint": fingerprint,
                    "decision_key": fingerprint,
                    "decision_step": state.step_index,
                    "effect_attempts": effect_attempts,
                    "synthesis_attempts": synthesis_attempts,
                }
            )
            self._cache_candidate(fingerprint, output)
            return output
        output = self._output_from_synthesis(
            decision, proposals, previous_score, previous_action
        )
        self._last_valid_output = output
        self.last_evidence = {
            "type": "qwen3vl_rich_delta_zero_shot",
            "decision_mode": "qwen_delta_diagnosis_then_global_synthesis",
            "candidate_fingerprint": fingerprint,
            "decision_key": fingerprint,
            "decision_step": state.step_index,
            "cache_hit": False,
            "gt_available": False,
            "verifier_valid": True,
            "localization_valid": output.localization_valid,
            "mask_facts": facts,
            "region_proposals": proposals,
            "effect_judgments": [
                {
                    "region_id": item.region_id,
                    "delta_kind": item.delta_kind,
                    "t1_state": item.t1_state,
                    "t2_state": item.t2_state,
                    "effect": item.effect,
                    "target_view": item.target_view,
                    "suggested_action": item.suggested_action,
                    "confidence": item.confidence,
                    "severity": item.severity,
                    "feedback": item.feedback,
                }
                for item in judgments
            ],
            "effect_attempts": effect_attempts,
            "synthesis_attempts": synthesis_attempts,
            "comparison": decision.comparison,
            "synthesis_decision": self._decision_dict(decision),
            "validation_errors": effect_errors + synthesis_errors,
        }
        self._cache_candidate(fingerprint, output)
        return output

    def _run_region_batches(
        self,
        state: ChangeState,
        proposals: list[dict[str, Any]],
        facts: dict[str, Any],
    ) -> tuple[
        tuple[_RegionJudgment, ...] | None,
        list[dict[str, Any]],
        list[str],
    ]:
        batch_size = int(
            facts.get("proposal_config", {}).get("max_regions_per_batch")
            or len(proposals)
        )
        if batch_size < 1:
            return None, [], ["initial audit batch size must be positive"]
        all_judgments: list[_RegionJudgment] = []
        attempts: list[dict[str, Any]] = []
        errors: list[str] = []
        for batch_index, start in enumerate(range(0, len(proposals), batch_size)):
            batch = proposals[start : start + batch_size]
            parsed: tuple[_RegionJudgment, ...] | None = None
            batch_errors: list[str] = []
            for _ in range(self.max_retries):
                raw = self._generate_messages(
                    self.build_messages(
                        state,
                        None,
                        None,
                        None,
                        batch_errors[-1:],
                        proposals_override=batch,
                    )
                )
                try:
                    payload = self._extract_json_object(raw)
                    parsed = self._parse_rich_region_payload(payload, batch)
                    attempts.append(
                        {
                            "batch_index": batch_index,
                            "region_ids": [item["region_id"] for item in batch],
                            "raw": raw,
                            "output": payload,
                        }
                    )
                    break
                except (TypeError, ValueError, KeyError) as error:
                    batch_errors.append(str(error))
                    errors.append(f"initial batch {batch_index}: {error}")
                    attempts.append(
                        {
                            "batch_index": batch_index,
                            "region_ids": [item["region_id"] for item in batch],
                            "raw": raw,
                            "error": str(error),
                        }
                    )
            if parsed is None:
                return None, attempts, errors
            all_judgments.extend(parsed)
        return tuple(all_judgments), attempts, errors

    def _run_effect_batches(
        self,
        state: ChangeState,
        previous_state: ChangeState,
        previous_action: AgentAction | None,
        proposals: list[dict[str, Any]],
        facts: dict[str, Any],
    ) -> tuple[
        tuple[_EffectJudgment, ...] | None,
        list[dict[str, Any]],
        list[str],
    ]:
        batch_size = int(
            facts.get("proposal_config", {}).get("max_regions_per_batch")
            or len(proposals)
        )
        if batch_size < 1:
            return None, [], ["candidate delta batch size must be positive"]
        all_judgments: list[_EffectJudgment] = []
        attempts: list[dict[str, Any]] = []
        errors: list[str] = []
        for batch_index, start in enumerate(range(0, len(proposals), batch_size)):
            batch = proposals[start : start + batch_size]
            batch_facts = dict(facts)
            batch_facts.update(
                {
                    "batch_index": batch_index,
                    "batch_proposal_count": len(batch),
                    "total_proposal_count": len(proposals),
                }
            )
            parsed: tuple[_EffectJudgment, ...] | None = None
            batch_errors: list[str] = []
            for _ in range(self.max_retries):
                raw = self._generate_messages(
                    self.build_effect_messages(
                        state,
                        previous_state,
                        previous_action,
                        batch,
                        batch_facts,
                        batch_errors[-1:],
                    )
                )
                try:
                    payload = self._extract_json_object(raw)
                    parsed = self._parse_rich_effect_payload(payload, batch)
                    attempts.append(
                        {
                            "batch_index": batch_index,
                            "region_ids": [item["region_id"] for item in batch],
                            "raw": raw,
                            "output": payload,
                        }
                    )
                    break
                except (TypeError, ValueError, KeyError) as error:
                    batch_errors.append(str(error))
                    errors.append(f"candidate batch {batch_index}: {error}")
                    attempts.append(
                        {
                            "batch_index": batch_index,
                            "region_ids": [item["region_id"] for item in batch],
                            "raw": raw,
                            "error": str(error),
                        }
                    )
            if parsed is None:
                return None, attempts, errors
            all_judgments.extend(parsed)
        return tuple(all_judgments), attempts, errors

    def _run_synthesis(
        self,
        state: ChangeState,
        previous_state: ChangeState | None,
        previous_action: AgentAction | None,
        proposals: list[dict[str, Any]],
        facts: dict[str, Any],
        judgments: tuple[_RegionJudgment, ...] | tuple[_EffectJudgment, ...],
        *,
        initial: bool,
    ) -> tuple[_SynthesisDecision | None, list[dict[str, Any]], list[str]]:
        attempts: list[dict[str, Any]] = []
        errors: list[str] = []
        for _ in range(self.max_retries):
            raw = self._generate_messages(
                self.build_synthesis_messages(
                    state,
                    previous_state,
                    previous_action,
                    proposals,
                    facts,
                    judgments,
                    initial=initial,
                    previous_errors=errors[-1:],
                )
            )
            try:
                payload = self._extract_json_object(raw)
                decision = self._parse_synthesis_payload(
                    payload, proposals, judgments, initial=initial
                )
                attempts.append({"raw": raw, "output": payload})
                return decision, attempts, errors
            except (TypeError, ValueError, KeyError) as error:
                errors.append(str(error))
                attempts.append({"raw": raw, "error": str(error)})
        return None, attempts, errors

    def _output_from_synthesis(
        self,
        decision: _SynthesisDecision,
        proposals: list[dict[str, Any]],
        previous_score: float | None,
        previous_action: AgentAction | None,
    ) -> VerifierOutput:
        lookup = {item["region_id"]: item for item in proposals}
        region = None
        if decision.region_id is not None:
            proposal = lookup[decision.region_id]
            if decision.suggested_action in {"positive_point", "negative_point"}:
                x, y = proposal["component_seed_normalized"]
                region = (x, y, x, y)
            else:
                region = tuple(proposal["box_normalized"])
        target_view = (
            decision.target_view
            or (previous_action.target_view if previous_action else "t2")
        )
        initial = decision.comparison == "initial"
        accept = (
            decision.error_type == "none"
            and decision.quality_score >= self.accept_threshold
            if initial
            else decision.comparison == "better"
        )
        stop = (
            accept
            and decision.error_type == "none"
            and decision.suggested_action == "finish"
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
            error_type=decision.error_type,
            target_view=target_view,
            error_region=region,
            suggested_action=decision.suggested_action,
            feedback=decision.feedback,
            accept=accept,
            verifier_valid=True,
            localization_valid=(decision.error_type == "none" or region is not None),
            stop=stop,
        )

    @staticmethod
    def _decision_dict(decision: _SynthesisDecision) -> dict[str, Any]:
        return {
            "quality_score": decision.quality_score,
            "progress_score": decision.progress_score,
            "comparison": decision.comparison,
            "error_type": decision.error_type,
            "target_view": decision.target_view,
            "region_id": decision.region_id,
            "suggested_action": decision.suggested_action,
            "feedback": decision.feedback,
        }

    @staticmethod
    def _initial_geometry_consistent(verdict: str, audit_kind: Any) -> bool:
        if audit_kind == "present":
            return verdict not in {"false_negative", "correct_unchanged"}
        if audit_kind == "missing":
            return verdict not in {"true_change", "false_positive"}
        return verdict in {"mixed", "uncertain"}

    def _cache_candidate(
        self, fingerprint: str, output: VerifierOutput
    ) -> None:
        self._candidate_cache[fingerprint] = (
            output,
            copy.deepcopy(self.last_evidence),
        )

    def _invalid_output(
        self,
        errors: list[str],
        *,
        region_attempts: list[dict[str, Any]],
        pairwise_attempts: list[dict[str, Any]],
        previous_action: AgentAction | None,
    ) -> VerifierOutput:
        previous = self._last_valid_output
        retained = previous.feedback if previous is not None else None
        messages = ["Verifier invalid; no action is authorized; recheck required."]
        if retained:
            messages.append(f"Previous valid feedback retained: {retained}")
        self.last_evidence = {
            "type": "qwen3vl_rich_region_zero_shot",
            "decision_mode": "invalid_rich_verifier_output",
            "region_attempts": region_attempts,
            "pairwise_attempts": pairwise_attempts,
            "validation_errors": errors,
            "fallback": True,
            "verifier_valid": False,
            "localization_valid": False,
            "gt_available": False,
        }
        return VerifierOutput(
            quality_score=previous.quality_score if previous else None,
            progress_score=0.0,
            score_delta=0.0,
            comparison="uncertain",
            error_type=(previous.error_type if previous else "uncertain_region"),
            target_view=(
                previous.target_view
                if previous
                else previous_action.target_view if previous_action else "t2"
            ),
            error_region=previous.error_region if previous else None,
            suggested_action=None,
            feedback=" ".join(messages),
            accept=False,
            verifier_valid=False,
            localization_valid=False,
            stop=False,
        )

    def build_messages(
        self,
        state: ChangeState,
        previous_score: float | None,
        previous_action: AgentAction | None,
        previous_state: ChangeState | None = None,
        previous_errors: list[str] | None = None,
        *,
        proposals_override: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        del previous_score, previous_action, previous_state
        proposals = (
            list(proposals_override)
            if proposals_override is not None
            else list(state.evidence.get("verifier_region_proposals", []))
        )
        correction = (
            f"Your previous regional diagnosis was invalid: "
            f"{previous_errors[-1]}. Correct it.\n"
            if previous_errors
            else ""
        )
        prompt = (
            "Act as the error-diagnosis core for building change detection, not as a simple "
            "object classifier. Judge whether the FINAL CURRENT CHANGE MASK is correct at every "
            "exact Environment region. White in the current change mask means predicted change; "
            "black means predicted unchanged. The T1/T2 object masks are auxiliary predicted "
            "building occupancy, not ground truth. A missing object in one temporal object mask "
            "is NOT automatically a false negative; T1=background and T2=building is a normal "
            "true building appearance when RGB supports it. Use: true_change for a white region "
            "supported by a real RGB building appearance/disappearance; correct_unchanged for a "
            "black region correctly showing no RGB building change; false_positive for a white "
            "region without real RGB building change; false_negative only for a black region "
            "that misses real RGB building change; mixed when the exact component contains both "
            "correct and erroneous parts; uncertain only when visual evidence cannot decide. "
            "First copy change_mask_state from the authoritative proposal: audit_kind=present "
            "means white_predicted_change and audit_kind=missing means black_predicted_unchanged. "
            "Then independently record the clean-RGB state at the exact "
            "component pixels in T1 and T2 as building, background, mixed, or uncertain. Same "
            "decisive states normally mean no real building change; different decisive states "
            "normally mean a real appearance/disappearance. Apply these exact error definitions: "
            "white_predicted_change plus same T1/T2 state is false_positive; "
            "white_predicted_change plus different states is true_change; "
            "black_predicted_unchanged plus same states is correct_unchanged; "
            "black_predicted_unchanged plus different states is false_negative. Use mixed or "
            "uncertain when states are not decisive. These fields are supporting evidence, but "
            "you still own the final error verdict. target_view means the predicted "
            "temporal object mask to EDIT so the final change "
            "mask becomes correct, not merely the image where a building is visible. For mixed, "
            "explain the beneficial and harmful subparts and their relative importance. Return exactly "
            "{\"regions\":[...]} with one item per supplied region. Every item must contain only "
            "region_id, change_mask_state, t1_state, t2_state, verdict, target_view "
            "(t1/t2/null), suggested_action "
            "(positive_point/negative_point/box/null), confidence (0..1), severity (0..1), and "
            "feedback (one or two concise diagnostic sentences; do not repeat phrases). "
            "Write lowercase enum values and literal JSON null, never strings such as \"T1\" or "
            "\"null\". For true_change or correct_unchanged use null target/action unless a boundary error "
            "still needs correction. Do not output coordinates, scores for "
            "the whole image, comparison, accept, or GT claims.\n"
            f"{correction}"
        )
        content: list[dict[str, Any]] = [
            {"type": "text", "text": "Full T1 original image:"},
            {"type": "image", "image": self._as_image(state.t1_image)},
            {"type": "text", "text": "Full T2 original image:"},
            {"type": "image", "image": self._as_image(state.t2_image)},
            {"type": "text", "text": "Full predicted T1 object mask:"},
            {"type": "image", "image": self._mask_image(state.t1_mask)},
            {"type": "text", "text": "Full predicted T2 object mask:"},
            {"type": "image", "image": self._mask_image(state.t2_mask)},
            {"type": "text", "text": "Full current change mask:"},
            {"type": "image", "image": self._mask_image(state.change_mask)},
        ]
        for proposal in proposals:
            public_proposal = {
                key: proposal[key]
                for key in (
                    "region_id",
                    "component_area",
                    "box_normalized",
                    "component_t1_mask_pixels",
                    "component_t2_mask_pixels",
                    "component_seed_t1_mask_white",
                    "component_seed_t2_mask_white",
                )
                if key in proposal
            }
            audit_kind = proposal.get("audit_kind")
            public_proposal["audit_kind"] = audit_kind
            public_proposal["geometry_rule"] = (
                "currently white: false_negative and correct_unchanged are impossible"
                if audit_kind == "present"
                else "currently black: true_change and false_positive are impossible"
                if audit_kind == "missing"
                else "geometry is mixed; inspect cautiously"
            )
            content.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            f"Initial audit proposal {proposal['region_id']}: "
                            f"{json.dumps(public_proposal, ensure_ascii=False)}. "
                            "Panel layout: top row is padded CLEAN T1 RGB, padded CLEAN T2 RGB, "
                            "and exact binary geometry; bottom row is tight CLEAN T1 RGB, tight "
                            "CLEAN T2 RGB, and amplified raw RGB difference. Geometry/difference "
                            "tiles are diagnostic data, not scene colors. No RGB pixel has been "
                            "outlined, recolored, masked, brightened, or darkened."
                        ),
                    },
                    {"type": "image", "image": self._rgb_initial_panel(state, proposal)},
                ]
            )
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def build_effect_messages(
        self,
        state: ChangeState,
        previous_state: ChangeState,
        previous_action: AgentAction | None,
        proposals: list[dict[str, Any]],
        facts: dict[str, Any],
        previous_errors: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        correction = (
            f"Your previous delta diagnosis was invalid: {previous_errors[-1]}. Correct it.\n"
            if previous_errors
            else ""
        )
        prompt = (
            "Act as the candidate-error verifier. Diagnose each exact added/removed delta using "
            "the RGB images, previous/candidate masks, local panel, action, and geometry. Do not "
            "First copy delta_kind (added/removed) from the authoritative proposal, then "
            "independently record t1_state and t2_state at the exact delta pixels using only "
            "clean RGB. Same decisive states mean no real building change; different decisive "
            "states mean a real building change. Then diagnose the candidate effect. Do not "
            "collapse mixed evidence "
            "to uncertain: describe which parts improve and which parts "
            "damage the FINAL CHANGE MASK and which side dominates so a later global synthesis "
            "can weigh them. An added region is newly white: added_true_change means RGB supports "
            "the new change and added_false_change means it does not. A removed region is newly "
            "black: removed_false_positive means a spurious change was correctly removed and "
            "removed_true_change means a real change was wrongly removed. Do not reinterpret "
            "object-mask occupancy as candidate-delta polarity. Return exactly "
            "{\"regions\":[...]} with one item per supplied region. Each item contains only "
            "region_id, delta_kind, t1_state, t2_state, effect "
            "(added_true_change/added_false_change/removed_false_positive/"
            "removed_true_change/mixed/uncertain), target_view (t1/t2/null), suggested_action "
            "(positive_point/negative_point/box/finish/null), confidence (0..1), severity (0..1), "
            "and feedback (one or two concise diagnostic sentences; do not repeat phrases). "
            "Write lowercase enum values and literal JSON null, not the string \"null\". "
            "Do not output overall comparison, "
            "quality/progress scores, coordinates, accept, or GT claims.\n"
            f"Exact candidate delta facts: {json.dumps(facts, ensure_ascii=False)}\n"
            f"Action: {json.dumps(self._public_action(previous_action, state.image_size), ensure_ascii=False)}\n"
            f"{correction}"
        )
        content: list[dict[str, Any]] = [
            {"type": "text", "text": "Fixed T1 original image:"},
            {"type": "image", "image": self._as_image(state.t1_image)},
            {"type": "text", "text": "Fixed T2 original image:"},
            {"type": "image", "image": self._as_image(state.t2_image)},
            {"type": "text", "text": "Previous accepted final change mask:"},
            {"type": "image", "image": self._mask_image(previous_state.change_mask)},
            {"type": "text", "text": "Candidate final change mask:"},
            {"type": "image", "image": self._mask_image(state.change_mask)},
            {"type": "text", "text": "Candidate predicted T1 object mask:"},
            {"type": "image", "image": self._mask_image(state.t1_mask)},
            {"type": "text", "text": "Candidate predicted T2 object mask:"},
            {"type": "image", "image": self._mask_image(state.t2_mask)},
        ]
        for proposal in proposals:
            content.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            f"Candidate delta proposal {proposal['region_id']} with exact metadata: "
                            f"{json.dumps(proposal, ensure_ascii=False)}. Panel layout: top row "
                            "is padded CLEAN T1 RGB, padded CLEAN T2 RGB, and exact binary delta "
                            "geometry; bottom row is tight CLEAN T1 RGB, tight CLEAN T2 RGB, "
                            "and amplified raw RGB difference. Diagnostic tiles are not scene colors. No RGB "
                            "pixel has been outlined, recolored, masked, brightened, or darkened."
                        ),
                    },
                    {"type": "image", "image": self._rgb_delta_panel(state, previous_state, proposal)},
                ]
            )
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def build_synthesis_messages(
        self,
        state: ChangeState,
        previous_state: ChangeState | None,
        previous_action: AgentAction | None,
        proposals: list[dict[str, Any]],
        facts: dict[str, Any],
        judgments: tuple[_RegionJudgment, ...] | tuple[_EffectJudgment, ...],
        *,
        initial: bool,
        previous_errors: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        correction = (
            f"Your previous global synthesis was invalid: {previous_errors[-1]}. Correct it.\n"
            if previous_errors
            else ""
        )
        if initial:
            summaries = []
            proposal_by_id = {item["region_id"]: item for item in proposals}
            for item in judgments:
                if not isinstance(item, _RegionJudgment):
                    continue
                proposal = proposal_by_id[item.region_id]
                summaries.append(
                    {
                        "region_id": item.region_id,
                        "change_mask_state": item.change_mask_state,
                        "t1_state": item.t1_state,
                        "t2_state": item.t2_state,
                        "verdict": item.verdict,
                        "audit_kind": proposal.get("audit_kind"),
                        "component_area": proposal.get("component_area"),
                        "component_t1_mask_pixels": proposal.get(
                            "component_t1_mask_pixels"
                        ),
                        "component_t2_mask_pixels": proposal.get(
                            "component_t2_mask_pixels"
                        ),
                        "component_seed_t1_mask_white": proposal.get(
                            "component_seed_t1_mask_white"
                        ),
                        "component_seed_t2_mask_white": proposal.get(
                            "component_seed_t2_mask_white"
                        ),
                        "box_normalized": proposal.get("box_normalized"),
                        "confidence": item.confidence,
                        "severity": item.severity,
                        "feedback": item.feedback,
                        "geometry_consistency": self._initial_geometry_consistent(
                            item.verdict, proposal.get("audit_kind")
                        ),
                    }
                )
            summaries.sort(
                key=lambda value: (value.get("component_area", 0), value["region_id"])
            )
            mode = (
                "This is the initial state: comparison must be initial and progress_score must "
                "be 0.0. Select the most useful next correction, including a mixed/uncertain "
                "region when further correction can resolve it. Independently plan the final "
                "target/action from RGB and geometry; local target/action suggestions are not "
                "part of the synthesis evidence. Prefer a compact, high-confidence "
                "error over a large or uncertain edit. A false geometry_consistency flag means "
                "the local Qwen wording conflicts with authoritative white/black mask state; "
                "resolve that conflict yourself using change_mask_state, T1/T2 states, and "
                "feedback instead of copying the inconsistent verdict. The local diagnoses are "
                "ordered from smallest to largest component for risk-aware planning; do not "
                "default to the first region unless its evidence makes it the best correction. "
                "component_t1_mask_pixels/component_t2_mask_pixels and seed mask values describe "
                "the editable predicted object masks, so choose a target/action that can actually "
                "change the selected error."
            )
        else:
            summaries = []
            proposal_by_id = {item["region_id"]: item for item in proposals}
            for item in judgments:
                if not isinstance(item, _EffectJudgment):
                    continue
                proposal = proposal_by_id[item.region_id]
                summaries.append(
                    {
                        "region_id": item.region_id,
                        "delta_kind": item.delta_kind,
                        "t1_state": item.t1_state,
                        "t2_state": item.t2_state,
                        "effect": item.effect,
                        "effect_kind": proposal.get("effect_kind"),
                        "component_area": proposal.get("component_area"),
                        "component_t1_mask_pixels": proposal.get(
                            "component_t1_mask_pixels"
                        ),
                        "component_t2_mask_pixels": proposal.get(
                            "component_t2_mask_pixels"
                        ),
                        "component_seed_t1_mask_white": proposal.get(
                            "component_seed_t1_mask_white"
                        ),
                        "component_seed_t2_mask_white": proposal.get(
                            "component_seed_t2_mask_white"
                        ),
                        "box_normalized": proposal.get("box_normalized"),
                        "confidence": item.confidence,
                        "severity": item.severity,
                        "feedback": item.feedback,
                    }
                )
            summaries.sort(
                key=lambda value: (value.get("component_area", 0), value["region_id"])
            )
            mode = (
                "Compare the candidate with the previous accepted state. Weigh beneficial and "
                "harmful portions, including mixed regions, and directly decide better, worse, "
                "unchanged, or uncertain. Do not use a rule that mixed automatically means "
                "uncertain. Independently plan the remaining correction; local target/action "
                "suggestions are not part of the synthesis evidence. The local diagnoses are "
                "ordered from smallest to largest component. Exact component and seed mask "
                "occupancy describe the editable predicted object masks; use them to avoid an "
                "action that cannot change the chosen region."
            )
        prompt = (
            "You are the core Change Verifier and correction planner. Synthesize the local "
            "diagnoses into one global judgment. Return exactly one JSON object containing only "
            "quality_score (0..1), progress_score (-1..1), comparison "
            "(initial/better/worse/unchanged/uncertain), error_type "
            "(none/false_positive_change/false_negative/mixed_error/uncertain_region), "
            "target_view (t1/t2/null), region_id (one supplied ID or null), suggested_action "
            "(positive_point/negative_point/box/finish), and feedback (two to four sentences "
            "explaining the tradeoff and correction). If error_type is none, use null region_id "
            "and finish. Write lowercase enum values and literal JSON null, not the string "
            "\"null\". Otherwise select an exact region_id and a non-finish correction. "
            "Calibrate quality_score as the estimated correctness of the complete final change "
            "mask, not confidence in one easy region; do not assign above 0.9 when any credible "
            "error remains. "
            f"{mode}\n"
            f"Authoritative mask/delta facts: {json.dumps(facts, ensure_ascii=False)}\n"
            f"Action: {json.dumps(self._public_action(previous_action, state.image_size), ensure_ascii=False)}\n"
            f"Local diagnoses: {json.dumps(summaries, ensure_ascii=False)}\n"
            f"{correction}"
        )
        content: list[dict[str, Any]] = [
            {"type": "text", "text": "Fixed T1 original image:"},
            {"type": "image", "image": self._as_image(state.t1_image)},
            {"type": "text", "text": "Fixed T2 original image:"},
            {"type": "image", "image": self._as_image(state.t2_image)},
        ]
        if previous_state is not None:
            content.extend(
                [
                    {"type": "text", "text": "Previous accepted final change mask:"},
                    {"type": "image", "image": self._mask_image(previous_state.change_mask)},
                ]
            )
        content.extend(
            [
                {"type": "text", "text": "Candidate/current predicted T1 mask:"},
                {"type": "image", "image": self._mask_image(state.t1_mask)},
                {"type": "text", "text": "Candidate/current predicted T2 mask:"},
                {"type": "image", "image": self._mask_image(state.t2_mask)},
                {"type": "text", "text": "Candidate/current final change mask:"},
                {"type": "image", "image": self._mask_image(state.change_mask)},
                {"type": "text", "text": prompt},
            ]
        )
        return [{"role": "user", "content": content}]

    @staticmethod
    def _bounded_number(value: Any, name: str, lower: float, upper: float) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        number = float(value)
        if not lower <= number <= upper:
            raise ValueError(f"{name} must be in [{lower}, {upper}]")
        return number

    @staticmethod
    def _enum_token(value: Any, allowed: set[str], name: str) -> str:
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string")
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized not in allowed:
            raise ValueError(f"unsupported {name}")
        return normalized

    @classmethod
    def _nullable_enum_token(
        cls, value: Any, allowed: set[str], name: str
    ) -> str | None:
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in {
            "",
            "null",
            "none",
            "n/a",
            "na",
        }:
            return None
        return cls._enum_token(value, allowed, name)

    @classmethod
    def _change_mask_state_token(cls, value: Any) -> str:
        aliases = {
            "white": "white_predicted_change",
            "predicted_change": "white_predicted_change",
            "black": "black_predicted_unchanged",
            "predicted_unchanged": "black_predicted_unchanged",
        }
        if isinstance(value, str):
            normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
            value = aliases.get(normalized, normalized)
        return cls._enum_token(
            value,
            {"white_predicted_change", "black_predicted_unchanged"},
            "change_mask_state",
        )

    def _parse_rich_region_payload(
        self, payload: dict[str, Any], proposals: list[dict[str, Any]]
    ) -> tuple[_RegionJudgment, ...]:
        if set(payload) != {"regions"} or not isinstance(payload["regions"], list):
            raise ValueError("regional diagnosis must contain only a regions list")
        expected = [item["region_id"] for item in proposals]
        expected_fields = {
            "region_id",
            "change_mask_state",
            "t1_state",
            "t2_state",
            "verdict",
            "target_view",
            "suggested_action",
            "confidence",
            "severity",
            "feedback",
        }
        parsed: dict[str, _RegionJudgment] = {}
        by_id = {item["region_id"]: item for item in proposals}
        for item in payload["regions"]:
            if not isinstance(item, dict) or set(item) != expected_fields:
                raise ValueError("each regional diagnosis must use the exact rich schema")
            region_id = item["region_id"]
            if region_id not in by_id or region_id in parsed:
                raise ValueError("regional diagnosis has an unknown or duplicate region_id")
            change_mask_state = self._change_mask_state_token(
                item["change_mask_state"]
            )
            t1_state = self._enum_token(
                item["t1_state"], self.RGB_STATES, "T1 RGB state"
            )
            t2_state = self._enum_token(
                item["t2_state"], self.RGB_STATES, "T2 RGB state"
            )
            verdict = self._enum_token(
                item["verdict"], self.REGION_VERDICTS, "regional verdict"
            )
            target_view = self._nullable_enum_token(
                item["target_view"], {"t1", "t2"}, "target_view"
            )
            action = self._nullable_enum_token(
                item["suggested_action"], self.ACTIONS - {"finish"},
                "regional suggested_action",
            )
            feedback = item["feedback"]
            if not isinstance(feedback, str) or not feedback.strip():
                raise ValueError("regional feedback must be a non-empty string")
            audit_kind = by_id[region_id].get("audit_kind")
            expected_mask_state = (
                "white_predicted_change"
                if audit_kind == "present"
                else "black_predicted_unchanged"
                if audit_kind == "missing"
                else None
            )
            if expected_mask_state is not None and change_mask_state != expected_mask_state:
                raise ValueError("change_mask_state contradicts authoritative audit geometry")
            parsed[region_id] = _RegionJudgment(
                region_id=region_id,
                change_mask_state=change_mask_state,
                t1_state=t1_state,
                t2_state=t2_state,
                verdict=verdict,
                target_view=target_view,
                feedback=feedback.strip(),
                suggested_action=action,
                confidence=self._bounded_number(item["confidence"], "confidence", 0, 1),
                severity=self._bounded_number(item["severity"], "severity", 0, 1),
            )
        if set(parsed) != set(expected):
            raise ValueError("regional diagnosis must cover every supplied region exactly once")
        return tuple(parsed[region_id] for region_id in expected)

    def _parse_rich_effect_payload(
        self, payload: dict[str, Any], proposals: list[dict[str, Any]]
    ) -> tuple[_EffectJudgment, ...]:
        if set(payload) != {"regions"} or not isinstance(payload["regions"], list):
            raise ValueError("delta diagnosis must contain only a regions list")
        expected = [item["region_id"] for item in proposals]
        expected_fields = {
            "region_id",
            "delta_kind",
            "t1_state",
            "t2_state",
            "effect",
            "target_view",
            "suggested_action",
            "confidence",
            "severity",
            "feedback",
        }
        by_id = {item["region_id"]: item for item in proposals}
        parsed: dict[str, _EffectJudgment] = {}
        for item in payload["regions"]:
            if not isinstance(item, dict) or set(item) != expected_fields:
                raise ValueError("each delta diagnosis must use the exact rich schema")
            region_id = item["region_id"]
            if region_id not in by_id or region_id in parsed:
                raise ValueError("delta diagnosis has an unknown or duplicate region_id")
            delta_kind = self._enum_token(
                item["delta_kind"], {"added", "removed"}, "delta_kind"
            )
            t1_state = self._enum_token(
                item["t1_state"], self.RGB_STATES, "T1 RGB state"
            )
            t2_state = self._enum_token(
                item["t2_state"], self.RGB_STATES, "T2 RGB state"
            )
            effect = self._enum_token(
                item["effect"], self.EFFECT_LABELS, "candidate effect label"
            )
            target_view = self._nullable_enum_token(
                item["target_view"], {"t1", "t2"}, "target_view"
            )
            action = self._nullable_enum_token(
                item["suggested_action"], self.ACTIONS,
                "delta suggested_action",
            )
            feedback = item["feedback"]
            effect_kind = by_id[region_id].get("effect_kind")
            if delta_kind != effect_kind:
                raise ValueError("delta_kind contradicts authoritative candidate geometry")
            if effect_kind == "added" and effect.startswith("removed_"):
                raise ValueError("added delta region cannot receive a removed effect label")
            if effect_kind == "removed" and effect.startswith("added_"):
                raise ValueError("removed delta region cannot receive an added effect label")
            if not isinstance(feedback, str) or not feedback.strip():
                raise ValueError("delta feedback must be a non-empty string")
            parsed[region_id] = _EffectJudgment(
                region_id=region_id,
                delta_kind=delta_kind,
                t1_state=t1_state,
                t2_state=t2_state,
                effect=effect,
                target_view=target_view,
                suggested_action=action,
                confidence=self._bounded_number(item["confidence"], "confidence", 0, 1),
                severity=self._bounded_number(item["severity"], "severity", 0, 1),
                feedback=feedback.strip(),
            )
        if set(parsed) != set(expected):
            raise ValueError("delta diagnosis must cover every supplied region exactly once")
        return tuple(parsed[region_id] for region_id in expected)

    def _parse_synthesis_payload(
        self,
        payload: dict[str, Any],
        proposals: list[dict[str, Any]],
        judgments: tuple[_RegionJudgment, ...] | tuple[_EffectJudgment, ...],
        *,
        initial: bool,
    ) -> _SynthesisDecision:
        expected_fields = {
            "quality_score",
            "progress_score",
            "comparison",
            "error_type",
            "target_view",
            "region_id",
            "suggested_action",
            "feedback",
        }
        if set(payload) != expected_fields:
            raise ValueError("global synthesis must use the exact rich schema")
        quality = self._bounded_number(payload["quality_score"], "quality_score", 0, 1)
        progress = self._bounded_number(payload["progress_score"], "progress_score", -1, 1)
        comparison = self._enum_token(
            payload["comparison"], self.COMPARISONS | {"initial"}, "comparison"
        )
        if initial:
            if comparison != "initial" or progress != 0.0:
                raise ValueError("initial synthesis requires comparison=initial and progress_score=0")
        elif comparison not in self.COMPARISONS:
            raise ValueError("candidate comparison is unsupported")
        error_type = self._enum_token(
            payload["error_type"], self.ERROR_TYPES, "global error_type"
        )
        target_view = self._nullable_enum_token(
            payload["target_view"], {"t1", "t2"}, "target_view"
        )
        region_id = payload["region_id"]
        if isinstance(region_id, str) and region_id.strip().lower() in {
            "",
            "null",
            "none",
            "n/a",
            "na",
        }:
            region_id = None
        action = self._enum_token(
            payload["suggested_action"], self.ACTIONS, "global suggested_action"
        )
        feedback = payload["feedback"]
        if region_id is not None and region_id not in {
            item["region_id"] for item in proposals
        }:
            raise ValueError("global synthesis references an unknown region_id")
        if not isinstance(feedback, str) or not feedback.strip():
            raise ValueError("global feedback must be a non-empty string")
        if error_type == "none":
            if region_id is not None or target_view is not None or action != "finish":
                raise ValueError("error_type none requires null target/region and finish")
        elif region_id is None or target_view is None or action == "finish":
            raise ValueError("a global error requires an exact region, target, and correction")
        if initial and region_id is not None:
            proposal = next(
                item for item in proposals if item["region_id"] == region_id
            )
            audit_kind = proposal.get("audit_kind")
            if audit_kind == "present" and error_type == "false_negative":
                raise ValueError("global false_negative cannot target an already-white region")
            if audit_kind == "missing" and error_type == "false_positive_change":
                raise ValueError("global false_positive cannot target an already-black region")
            local = next(
                (
                    item
                    for item in judgments
                    if isinstance(item, _RegionJudgment) and item.region_id == region_id
                ),
                None,
            )
            if (
                local is not None
                and local.verdict in {"true_change", "correct_unchanged"}
                and self._initial_geometry_consistent(local.verdict, audit_kind)
            ):
                raise ValueError(
                    "global correction cannot select a region that the local diagnosis marked correct"
                )
        if region_id is not None and action == "negative_point":
            proposal = next(
                item for item in proposals if item["region_id"] == region_id
            )
            seed_key = f"component_seed_{target_view}_mask_white"
            if proposal.get(seed_key) is False:
                raise ValueError(
                    "global negative_point targets a black seed in the editable object mask"
                )
        return _SynthesisDecision(
            quality_score=quality,
            progress_score=progress,
            comparison=comparison,
            error_type=error_type,
            target_view=target_view,
            region_id=region_id,
            suggested_action=action,
            feedback=feedback.strip(),
        )

    @staticmethod
    def _states_identical(state: ChangeState, previous_state: ChangeState) -> bool:
        return (
            np.array_equal(state.t1_mask, previous_state.t1_mask)
            and np.array_equal(state.t2_mask, previous_state.t2_mask)
            and np.array_equal(state.change_mask, previous_state.change_mask)
        )

    def _candidate_fingerprint(
        self,
        state: ChangeState,
        previous_state: ChangeState,
        previous_action: AgentAction | None,
        proposals: list[dict[str, Any]],
        facts: dict[str, Any],
    ) -> str:
        digest = hashlib.sha256()
        for array in (
            state.t1_image,
            state.t2_image,
            previous_state.t1_mask,
            previous_state.t2_mask,
            previous_state.change_mask,
            state.t1_mask,
            state.t2_mask,
            state.change_mask,
        ):
            value = np.ascontiguousarray(array)
            digest.update(str(value.shape).encode("ascii"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(value.tobytes())
        identity = {
            "schema_version": self.SCHEMA_VERSION,
            "query": state.query,
            "action": previous_action.to_dict() if previous_action else None,
            "max_new_tokens": self.max_new_tokens,
            "max_retries": self.max_retries,
            "accept_threshold": self.accept_threshold,
            "do_sample": self.do_sample,
            "repetition_penalty": self.repetition_penalty,
            "candidate_evidence_modes": self.CANDIDATE_EVIDENCE_MODES,
            "model": getattr(getattr(self.model, "config", None), "_name_or_path", None),
            "proposals": proposals,
            "facts": facts,
        }
        digest.update(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        return digest.hexdigest()

    def _generate_messages(self, messages: list[dict[str, Any]]) -> str:
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        device = getattr(self.model, "device", None)
        if device is not None and hasattr(inputs, "to"):
            inputs = inputs.to(device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            repetition_penalty=self.repetition_penalty,
        )
        input_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
        generated = outputs[:, input_ids.shape[1] :]
        return self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    @staticmethod
    def _extract_json_object(raw: str) -> dict[str, Any]:
        decoder = json.JSONDecoder()
        start = raw.find("{")
        if start < 0:
            raise ValueError("verifier response contains no JSON object")
        try:
            value, _ = decoder.raw_decode(raw[start:])
        except json.JSONDecodeError as error:
            # Do not scan into nested objects: a truncated top-level response
            # must remain invalid instead of being misread as one region.
            raise ValueError("verifier response contains incomplete JSON object") from error
        if not isinstance(value, dict):
            raise ValueError("verifier response JSON must be an object")
        return value

    @staticmethod
    def _public_action(
        action: AgentAction | None, image_size: tuple[int, int]
    ) -> dict[str, Any] | None:
        if action is None:
            return None
        result: dict[str, Any] = {
            "target_view": action.target_view,
            "action": action.action,
            "coordinate_space": "normalized_0_1000",
        }
        if action.coordinate is not None:
            result["coordinate"] = list(
                pixel_point_to_normalized(action.coordinate, image_size)
            )
        if action.box is not None:
            result["box"] = list(pixel_box_to_normalized(action.box, image_size))
        return result

    @staticmethod
    def _rgb_delta_panel(
        state: ChangeState,
        previous_state: ChangeState,
        proposal: dict[str, Any],
        panel_size: int = 192,
    ) -> Image.Image:
        """Show padded/tight clean RGB, exact delta geometry, and raw difference."""

        x1, y1, x2, y2 = (int(value) for value in proposal["box_pixels"])
        region = (slice(y1, y2 + 1), slice(x1, x2 + 1))
        full_delta = Qwen3VLZeroShotVerifier._proposal_delta_component(
            state, previous_state, proposal
        )
        delta = full_delta[region]
        raw_t1 = state.t1_image[region]
        raw_t2 = state.t2_image[region]
        t1 = np.array(raw_t1, dtype=np.uint8, copy=True)
        t2 = np.array(raw_t2, dtype=np.uint8, copy=True)
        delta_rgb = np.repeat((delta.astype(np.uint8) * 255)[..., None], 3, axis=2)
        absolute_difference = np.abs(
            raw_t2.astype(np.int16) - raw_t1.astype(np.int16)
        )
        absolute_difference = np.clip(absolute_difference * 2, 0, 255).astype(np.uint8)
        tight_region = Qwen3VLZeroShotVerifier._tight_component_region(full_delta)
        tiles = [
            Image.fromarray(t1),
            Image.fromarray(t2),
            Image.fromarray(delta_rgb),
            Image.fromarray(np.asarray(state.t1_image[tight_region], dtype=np.uint8)),
            Image.fromarray(np.asarray(state.t2_image[tight_region], dtype=np.uint8)),
            Image.fromarray(absolute_difference),
        ]
        canvas = Image.new("RGB", (panel_size * 3, panel_size * 2))
        for index, tile in enumerate(tiles):
            resample = (
                Image.Resampling.NEAREST if index == 2 else Image.Resampling.BILINEAR
            )
            canvas.paste(
                tile.resize((panel_size, panel_size), resample),
                ((index % 3) * panel_size, (index // 3) * panel_size),
            )
        return canvas

    @staticmethod
    def _rgb_initial_panel(
        state: ChangeState,
        proposal: dict[str, Any],
        panel_size: int = 192,
    ) -> Image.Image:
        """Show padded/tight clean RGB and exact initial component geometry."""

        x1, y1, x2, y2 = (int(value) for value in proposal["box_pixels"])
        region = (slice(y1, y2 + 1), slice(x1, x2 + 1))
        full_component = Qwen3VLZeroShotVerifier._proposal_initial_component(
            state, proposal
        )
        component = full_component[region]
        raw_t1 = state.t1_image[region]
        raw_t2 = state.t2_image[region]
        t1 = np.array(raw_t1, dtype=np.uint8, copy=True)
        t2 = np.array(raw_t2, dtype=np.uint8, copy=True)
        component_rgb = np.repeat(
            (component.astype(np.uint8) * 255)[..., None], 3, axis=2
        )
        absolute_difference = np.abs(
            raw_t2.astype(np.int16) - raw_t1.astype(np.int16)
        )
        absolute_difference = np.clip(absolute_difference * 2, 0, 255).astype(np.uint8)
        tight_region = Qwen3VLZeroShotVerifier._tight_component_region(full_component)
        tiles = [
            Image.fromarray(t1),
            Image.fromarray(t2),
            Image.fromarray(component_rgb),
            Image.fromarray(np.asarray(state.t1_image[tight_region], dtype=np.uint8)),
            Image.fromarray(np.asarray(state.t2_image[tight_region], dtype=np.uint8)),
            Image.fromarray(absolute_difference),
        ]
        canvas = Image.new("RGB", (panel_size * 3, panel_size * 2))
        for index, tile in enumerate(tiles):
            resample = (
                Image.Resampling.NEAREST if index == 2 else Image.Resampling.BILINEAR
            )
            canvas.paste(
                tile.resize((panel_size, panel_size), resample),
                ((index % 3) * panel_size, (index // 3) * panel_size),
            )
        return canvas

    @staticmethod
    def _tight_component_region(component: np.ndarray) -> tuple[slice, slice]:
        ys, xs = np.nonzero(np.asarray(component, dtype=bool))
        if not len(xs):
            raise ValueError("cannot crop an empty verifier component")
        return (
            slice(int(ys.min()), int(ys.max()) + 1),
            slice(int(xs.min()), int(xs.max()) + 1),
        )

    @staticmethod
    def _outline_component(rgb: np.ndarray, component: np.ndarray) -> np.ndarray:
        """Draw an outer yellow ring without covering component RGB pixels."""

        image = np.array(rgb, dtype=np.uint8, copy=True)
        mask = np.asarray(component, dtype=bool)
        padded = np.pad(mask, 1, mode="constant", constant_values=False)
        dilated = np.zeros_like(mask)
        for offset_y in range(3):
            for offset_x in range(3):
                dilated |= padded[
                    offset_y : offset_y + mask.shape[0],
                    offset_x : offset_x + mask.shape[1],
                ]
        outline = np.logical_and(dilated, ~mask)
        image[outline] = np.array([255, 255, 0], dtype=np.uint8)
        return image

    @staticmethod
    def _proposal_initial_component(
        state: ChangeState, proposal: dict[str, Any]
    ) -> np.ndarray:
        audit_kind = proposal.get("audit_kind")
        if audit_kind == "present":
            source = np.asarray(state.change_mask, dtype=bool)
        elif audit_kind == "missing":
            source = np.logical_and(
                np.logical_xor(state.t1_mask, state.t2_mask),
                ~state.change_mask,
            )
        else:
            source = np.logical_or(
                state.change_mask,
                np.logical_and(
                    np.logical_xor(state.t1_mask, state.t2_mask),
                    ~state.change_mask,
                ),
            )
        seed_x, seed_y = proposal.get("component_seed_pixels", (None, None))
        if seed_x is None or seed_y is None:
            raise ValueError("initial proposal has no component seed")
        component = next(
            (
                item
                for item in connected_components(source)
                if item[int(seed_y), int(seed_x)]
            ),
            None,
        )
        if component is None:
            raise ValueError("initial proposal seed is outside its audit component")
        return component

    @staticmethod
    def _proposal_delta_component(
        state: ChangeState,
        previous_state: ChangeState,
        proposal: dict[str, Any],
    ) -> np.ndarray:
        if proposal.get("effect_kind") == "added":
            full_delta = np.logical_and(state.change_mask, ~previous_state.change_mask)
        elif proposal.get("effect_kind") == "removed":
            full_delta = np.logical_and(previous_state.change_mask, ~state.change_mask)
        else:
            raise ValueError("delta proposal has no valid effect_kind")
        seed_x, seed_y = proposal.get("component_seed_pixels", (None, None))
        if seed_x is None or seed_y is None:
            raise ValueError("delta proposal has no component seed")
        component = next(
            (
                item
                for item in connected_components(full_delta)
                if item[int(seed_y), int(seed_x)]
            ),
            None,
        )
        if component is None:
            raise ValueError("delta proposal seed does not identify a candidate component")
        return component

    @staticmethod
    def _as_image(value: Any) -> Image.Image:
        if isinstance(value, Image.Image):
            return value
        array = np.asarray(value)
        if array.dtype != np.uint8:
            if array.max(initial=0) <= 1:
                array = array * 255
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array)

    @staticmethod
    def _mask_image(mask: np.ndarray) -> Image.Image:
        return Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L")
