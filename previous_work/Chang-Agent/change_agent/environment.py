"""GT-free environment reset/step logic for iterative change refinement."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Protocol

import numpy as np

from .action_parser import ActionParser, ActionValidationError
from .adapters.omniovcd_adapter import InitializationResult, PairUpdate
from .executor import ActionExecutor
from .state import AgentAction, AgentObservation, ChangeState, VerifierOutput
from .trajectory import Trajectory, TrajectoryEntry
from .verifier import Verifier
from .verifier_regions import attach_verifier_regions


class StateBackend(Protocol):
    def initialize(
        self, t1_image: np.ndarray, t2_image: np.ndarray, query: str
    ) -> InitializationResult: ...

    def rebuild(
        self, t1_mask: np.ndarray, t2_mask: np.ndarray, evidence: Mapping[str, Any]
    ) -> PairUpdate: ...


class ChangeAgentEnvironment:
    """Owns GT-free state/evidence and exposes model masks needed for editing."""

    def __init__(
        self,
        backend: StateBackend,
        executor: ActionExecutor,
        verifier: Verifier,
        *,
        action_parser: ActionParser | None = None,
        max_steps: int = 8,
        inference_only: bool = True,
        run_metadata: dict[str, Any] | None = None,
        selection_policy: str = "conservative_best",
        selection_epsilon: float = 0.0,
        max_selection_area_delta: float = 0.25,
        max_locality_outside_ratio: float = 0.1,
        max_target_mask_change_ratio: float = 0.25,
        max_component_count_delta: int = 4,
        max_point_proposal_change_ratio: float = 0.25,
        max_point_proposal_change_pixels: int = 256,
        max_point_image_change_ratio: float = 0.004,
        point_proposal_change_gate_enabled: bool = True,
        require_tool_before_finish: bool = True,
        verifier_max_regions: int = 6,
        verifier_max_delta_regions: int = 3,
        verifier_min_region_area: int = 4,
        verifier_region_padding_ratio: float = 0.25,
        empty_mask_scan_grid_size: int = 0,
        enable_verifier_regions: bool = True,
    ):
        if not inference_only:
            raise ValueError("runtime Environment currently supports inference_only=True only")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if not 0 <= max_locality_outside_ratio <= 1:
            raise ValueError("max_locality_outside_ratio must be in [0, 1]")
        if not 0 <= max_target_mask_change_ratio <= 1:
            raise ValueError("max_target_mask_change_ratio must be in [0, 1]")
        if max_component_count_delta < 0:
            raise ValueError("max_component_count_delta must be non-negative")
        if not 0 < max_point_proposal_change_ratio <= 1:
            raise ValueError("max_point_proposal_change_ratio must be in (0, 1]")
        if max_point_proposal_change_pixels < 0:
            raise ValueError(
                "max_point_proposal_change_pixels must be non-negative"
            )
        if not 0 <= max_point_image_change_ratio <= 1:
            raise ValueError("max_point_image_change_ratio must be in [0, 1]")
        if (
            verifier_max_regions < 1
            or verifier_max_delta_regions < 1
            or verifier_min_region_area < 1
        ):
            raise ValueError("Verifier region limits must be positive")
        if verifier_region_padding_ratio < 0:
            raise ValueError("verifier_region_padding_ratio must be non-negative")
        if not 0 <= empty_mask_scan_grid_size <= 16:
            raise ValueError("empty_mask_scan_grid_size must be within [0,16]")
        self.backend = backend
        self.executor = executor
        self.verifier = verifier
        self.action_parser = action_parser or ActionParser()
        self.max_steps = max_steps
        self.inference_only = True
        self.selection_policy = selection_policy
        self.selection_epsilon = selection_epsilon
        self.max_selection_area_delta = max_selection_area_delta
        self.max_locality_outside_ratio = max_locality_outside_ratio
        self.max_target_mask_change_ratio = max_target_mask_change_ratio
        self.max_component_count_delta = max_component_count_delta
        self.max_point_proposal_change_ratio = max_point_proposal_change_ratio
        self.max_point_proposal_change_pixels = (
            max_point_proposal_change_pixels
        )
        self.max_point_image_change_ratio = max_point_image_change_ratio
        self.point_proposal_change_gate_enabled = bool(
            point_proposal_change_gate_enabled
        )
        self.require_tool_before_finish = require_tool_before_finish
        self.verifier_max_regions = verifier_max_regions
        self.verifier_max_delta_regions = verifier_max_delta_regions
        self.verifier_min_region_area = verifier_min_region_area
        self.verifier_region_padding_ratio = verifier_region_padding_ratio
        self.empty_mask_scan_grid_size = empty_mask_scan_grid_size
        self.enable_verifier_regions = bool(enable_verifier_regions)
        self.trajectory = Trajectory(
            run_metadata,
            selection_policy=selection_policy,
            selection_epsilon=selection_epsilon,
            max_area_delta=max_selection_area_delta,
        )
        self.state: ChangeState | None = None
        self.feedback: VerifierOutput | None = None
        self.done = False
        self._small_coordinate_streak = 0
        self._point_session_masks: dict[str, np.ndarray] = {}
        self._accepted_point_clicks: dict[
            str, list[tuple[tuple[int, int], bool]]
        ] = {"t1": [], "t2": []}
        self._rejected_action_signatures: set[str] = set()

    def reset(
        self, t1_image: np.ndarray, t2_image: np.ndarray, query: str
    ) -> AgentObservation:
        if not query.strip():
            raise ValueError("query must not be empty")
        t1_image = np.asarray(t1_image)
        t2_image = np.asarray(t2_image)
        reset_verifier = getattr(self.verifier, "reset", None)
        if callable(reset_verifier):
            reset_verifier()
        initialized = self.backend.initialize(t1_image, t2_image, query)
        self.state = self._state_from_update(
            t1_image,
            t2_image,
            query,
            initialized.t1_mask,
            initialized.t2_mask,
            initialized.update,
            step_index=0,
        )
        if self.enable_verifier_regions:
            self._attach_verifier_regions(self.state, None)
        self.feedback = self.verifier.verify(self.state, None, None, None)
        execution = self._with_verifier_evidence(
            {"event": "reset", "candidate_accepted": True}
        )
        self.done = False
        self._small_coordinate_streak = 0
        self._point_session_masks = {
            "t1": np.array(self.state.t1_mask, dtype=bool, copy=True),
            "t2": np.array(self.state.t2_mask, dtype=bool, copy=True),
        }
        self._accepted_point_clicks = {"t1": [], "t2": []}
        self._rejected_action_signatures = set()
        self.trajectory = Trajectory(
            self.trajectory.run_metadata,
            selection_policy=self.selection_policy,
            selection_epsilon=self.selection_epsilon,
            max_area_delta=self.max_selection_area_delta,
        )
        self.trajectory.append(
            TrajectoryEntry(0, None, None, self.feedback, self.state.clone(), execution)
        )
        return self.observation()

    def observation(self) -> AgentObservation:
        if self.state is None:
            raise RuntimeError("environment must be reset before observation")
        return AgentObservation(
            t1_image=np.array(self.state.t1_image, copy=True),
            t2_image=np.array(self.state.t2_image, copy=True),
            query=self.state.query,
            change_mask=np.array(self.state.change_mask, copy=True),
            feedback=self.feedback,
            history_summary=self.trajectory.history_summary(),
            t1_mask=np.array(self.state.t1_mask, copy=True),
            t2_mask=np.array(self.state.t2_mask, copy=True),
        )

    def step(
        self,
        action_or_raw: AgentAction | str,
        *,
        authorized_region_id: str | None = None,
    ) -> tuple[AgentObservation, bool]:
        if self.state is None or self.feedback is None:
            raise RuntimeError("environment must be reset before step")
        if self.done:
            raise RuntimeError("episode has already finished")
        if self.state.step_index >= self.max_steps:
            self.done = True
            return self.observation(), True

        raw_action = action_or_raw if isinstance(action_or_raw, str) else None
        raw_payload = None
        coordinate_warning = None
        if raw_action is not None:
            raw_payload = self.action_parser.extract_payload(raw_action)
            coordinate_warning = self._coordinate_warning(raw_payload)
        action = (
            self.action_parser.parse_payload(raw_payload, self.state.image_size)
            if isinstance(action_or_raw, str)
            else action_or_raw
        )
        action = self.action_parser.validate_pixel_action(action, self.state.image_size)
        if (
            action.action == "finish"
            and self.require_tool_before_finish
            and not any(entry.execution.get("tool") for entry in self.trajectory.entries)
            and not self._initial_finish_authorized()
        ):
            raise ActionValidationError("finish is forbidden before a segmentation tool action")
        action_signature = self._action_signature(self.state, action)
        if (
            action.action != "finish"
            and action_signature in self._rejected_action_signatures
        ):
            raise ActionValidationError(
                "action exactly repeats a previously rejected action on the same accepted state"
            )
        previous_state = self.state.clone()
        previous_feedback = self.feedback
        next_index = self.state.step_index + 1
        execution: dict[str, Any] = {}
        execution["action_signature"] = action_signature
        if raw_payload is not None:
            execution["raw_action_payload"] = raw_payload
        if coordinate_warning is not None:
            execution["coordinate_warning"] = coordinate_warning

        if action.action == "finish":
            candidate = self.state.clone()
            candidate.step_index = next_index
            execution["event"] = "finish_requested"
        else:
            target_image = (
                self.state.t1_image if action.target_view == "t1" else self.state.t2_image
            )
            target_mask = (
                self.state.t1_mask if action.target_view == "t1" else self.state.t2_mask
            )
            point_proposal_box = None
            point_proposal_component_area = None
            box_proposal_box = None
            if action.action in {"positive_point", "negative_point"}:
                point_proposal = self._authorized_point_proposal(
                    action, authorized_region_id
                )
                if point_proposal is not None:
                    point_proposal_box, point_proposal_component_area = point_proposal
            elif action.action == "box":
                box_proposal_box = self._authorized_box_proposal(
                    action, authorized_region_id
                )
            result = self.executor.execute(
                action,
                target_image,
                target_mask,
                self.state.query,
                point_session_mask=self._point_session_masks[action.target_view],
                point_click_history=tuple(
                    self._accepted_point_clicks[action.target_view]
                ),
                point_region_id=authorized_region_id,
                point_proposal_box=point_proposal_box,
                point_proposal_component_area=point_proposal_component_area,
                box_region_id=(
                    authorized_region_id
                    if box_proposal_box is not None
                    else None
                ),
            )
            t1_mask = result.mask if action.target_view == "t1" else self.state.t1_mask
            t2_mask = result.mask if action.target_view == "t2" else self.state.t2_mask
            evidence = dict(self.state.evidence)
            evidence.update(result.evidence)
            evidence["last_target_view"] = action.target_view
            update = self.backend.rebuild(t1_mask, t2_mask, evidence)
            candidate = self._state_from_update(
                self.state.t1_image,
                self.state.t2_image,
                self.state.query,
                t1_mask,
                t2_mask,
                update,
                step_index=next_index,
            )
            execution.update(result.evidence)

        previous_area_ratio = float(previous_state.change_mask.mean())
        candidate_area_ratio = float(candidate.change_mask.mean())
        area_delta = abs(candidate_area_ratio - previous_area_ratio)
        hard_rejection_reasons: list[str] = []
        if action.action != "finish":
            if area_delta > self.max_selection_area_delta:
                hard_rejection_reasons.append("mask_area_delta_exceeded")
            locality = execution.get("locality", {})
            if locality.get("outside_roi_ratio", 0.0) > self.max_locality_outside_ratio:
                hard_rejection_reasons.append("locality_outside_roi_exceeded")
            if (
                locality.get("target_mask_change_ratio", 0.0)
                > self.max_target_mask_change_ratio
            ):
                hard_rejection_reasons.append("target_mask_change_exceeded")
            if (
                abs(locality.get("component_count_delta", 0))
                > self.max_component_count_delta
            ):
                hard_rejection_reasons.append("component_count_delta_exceeded")
            point_authorization = execution.get("point_authorization", {})
            proposal_change_ratio = point_authorization.get(
                "proposal_change_ratio"
            )
            point_changed_pixels = int(locality.get("changed_pixels", 0))
            point_image_change_ratio = float(
                locality.get("target_mask_change_ratio", 0.0)
            )
            proposal_ratio_exceeded = bool(
                proposal_change_ratio is not None
                and proposal_change_ratio
                > self.max_point_proposal_change_ratio
            )
            absolute_pixels_exceeded = bool(
                point_changed_pixels
                > self.max_point_proposal_change_pixels
            )
            image_ratio_exceeded = bool(
                point_image_change_ratio
                > self.max_point_image_change_ratio
            )
            combined_point_gate_triggered = bool(
                self.point_proposal_change_gate_enabled
                and proposal_ratio_exceeded
                and absolute_pixels_exceeded
                and image_ratio_exceeded
            )
            if proposal_change_ratio is not None:
                point_authorization["change_gate"] = {
                    "enabled": self.point_proposal_change_gate_enabled,
                    "proposal_change_ratio": proposal_change_ratio,
                    "max_proposal_change_ratio": (
                        self.max_point_proposal_change_ratio
                    ),
                    "changed_pixels": point_changed_pixels,
                    "max_changed_pixels": (
                        self.max_point_proposal_change_pixels
                    ),
                    "image_change_ratio": point_image_change_ratio,
                    "max_image_change_ratio": (
                        self.max_point_image_change_ratio
                    ),
                    "proposal_ratio_exceeded": proposal_ratio_exceeded,
                    "absolute_pixels_exceeded": absolute_pixels_exceeded,
                    "image_ratio_exceeded": image_ratio_exceeded,
                    "hard_reject": combined_point_gate_triggered,
                    "policy": (
                        "reject_only_when_ratio_and_absolute_and_image_limits_exceed"
                    ),
                }
            if combined_point_gate_triggered:
                hard_rejection_reasons.append(
                    "point_proposal_combined_change_gate_exceeded"
                )

        verifier_skipped = bool(hard_rejection_reasons)
        if verifier_skipped:
            verifier_output = VerifierOutput(
                quality_score=None,
                progress_score=None,
                comparison="worse",
                error_type=previous_feedback.error_type,
                target_view=previous_feedback.target_view,
                error_region=previous_feedback.error_region,
                suggested_action=previous_feedback.suggested_action,
                feedback=(
                    "Candidate rejected before Qwen verification by hard runtime gates: "
                    + ", ".join(hard_rejection_reasons)
                ),
                accept=False,
                verifier_valid=True,
                localization_valid=previous_feedback.localization_valid,
                stop=False,
            )
            execution["verifier_evidence"] = {
                "type": "runtime_hard_gate_precheck",
                "verifier_skipped": True,
                "rejection_reasons": list(hard_rejection_reasons),
                "gt_available": False,
            }
        else:
            if self.enable_verifier_regions:
                self._attach_verifier_regions(candidate, previous_state)
                if action.action != "finish":
                    live_snapshot = candidate.clone()
                    self._attach_verifier_regions(live_snapshot, None)
                    candidate.evidence = dict(candidate.evidence)
                    candidate.evidence[
                        "verifier_next_live_region_proposals"
                    ] = live_snapshot.evidence.get(
                        "verifier_region_proposals", []
                    )
                    candidate.evidence[
                        "verifier_next_live_mask_facts"
                    ] = live_snapshot.evidence.get(
                        "verifier_mask_facts", {}
                    )
            verifier_output = self.verifier.verify(
                candidate, self.feedback.quality_score, action, previous_state
            )

        rejection_reasons = list(hard_rejection_reasons)
        if not verifier_skipped and not verifier_output.verifier_valid:
            rejection_reasons.append("verifier_invalid")
        if verifier_output.comparison is not None:
            ranking_progress = {
                "better": 1.0,
                "worse": -1.0,
                "initial": 0.0,
                "unchanged": 0.0,
                "uncertain": 0.0,
            }[verifier_output.comparison]
        else:
            ranking_progress = (
                verifier_output.progress_score
                if verifier_output.progress_score is not None
                else verifier_output.score_delta
            )
        if not verifier_skipped:
            if action.action == "finish":
                if not verifier_output.accept:
                    rejection_reasons.append("verifier_not_accepted")
            elif verifier_output.comparison is not None:
                if verifier_output.comparison != "better":
                    rejection_reasons.append("candidate_effect_not_better")
                elif not verifier_output.accept:
                    # A relative ranking may still be an explicit rejection:
                    # runtime safety gates must not convert it into a commit.
                    rejection_reasons.append("verifier_not_accepted")
            elif ranking_progress <= self.selection_epsilon:
                rejection_reasons.append("progress_did_not_improve")
        candidate_accepted = not rejection_reasons
        post_accept_reviewed = False
        execution.update(
            {
                "candidate_accepted": candidate_accepted,
                "candidate_rejection_reasons": rejection_reasons,
                "previous_area_ratio": previous_area_ratio,
                "candidate_area_ratio": candidate_area_ratio,
                "candidate_area_delta": area_delta,
                "ranking_progress": ranking_progress,
                "pairwise_comparison": verifier_output.comparison,
                "verifier_skipped_by_hard_gate": verifier_skipped,
            }
        )
        if not verifier_skipped:
            execution = self._with_verifier_evidence(execution)
        if candidate_accepted:
            if action.action != "finish" and self.enable_verifier_regions:
                # Candidate verification consumes action-scoped delta proposals,
                # while an accepted candidate's next staged plan is built from a
                # fresh full-state catalog. Promote the live state to that same
                # full proposal namespace before the next authorized action, so
                # the planned r*/s* ID cannot be looked up against stale d* deltas.
                candidate.evidence = dict(candidate.evidence)
                live_proposals = candidate.evidence.pop(
                    "verifier_next_live_region_proposals", None
                )
                live_facts = candidate.evidence.pop(
                    "verifier_next_live_mask_facts", None
                )
                if not isinstance(live_proposals, list) or not isinstance(
                    live_facts, Mapping
                ):
                    self._attach_verifier_regions(candidate, None)
                    live_proposals = candidate.evidence.get(
                        "verifier_region_proposals", []
                    )
                else:
                    candidate.evidence["verifier_region_proposals"] = (
                        live_proposals
                    )
                    candidate.evidence["verifier_mask_facts"] = dict(
                        live_facts
                    )
                execution["live_proposal_refresh"] = {
                    "policy": "full_state_after_accepted_candidate",
                    "proposal_count": len(live_proposals),
                    "region_ids": [
                        item.get("region_id")
                        for item in live_proposals
                        if isinstance(item, Mapping)
                    ],
                }
            self.state = candidate
            self.feedback = verifier_output
            post_accept_review = getattr(
                self.verifier, "review_after_acceptance", None
            )
            terminal_feedback_candidate = verifier_output.suggested_action in {
                None,
                "finish",
            }
            if callable(post_accept_review) and (
                action.action == "finish" or terminal_feedback_candidate
            ):
                # A non-terminal next action already belongs to the existing
                # action/candidate flow.  TerminalReviewVerifier itself is
                # called for both accepted finish and plan-null outputs; the
                # explicit finish branch closes the historical gap where a
                # finish candidate skipped the post-accept review entirely.
                reviewed_feedback = post_accept_review(
                    candidate,
                    verifier_output,
                )
                if not isinstance(reviewed_feedback, VerifierOutput):
                    raise TypeError(
                        "review_after_acceptance must return a VerifierOutput"
                    )
                self.feedback = reviewed_feedback
                post_accept_reviewed = True
                execution["post_accept_terminal_review"] = {
                    "verifier_output": reviewed_feedback.to_dict(),
                    "verifier_evidence": self._verifier_evidence(),
                }
            if action.action in {"positive_point", "negative_point"}:
                # The Executor commits only a locally composed subset of the raw
                # SimpleClick prediction.  Replaying the click against the older
                # session base would therefore continue an internal raw-mask state
                # that differs from the accepted live mask.  Rebase after every
                # accepted point so the next edit starts from the exact committed
                # target-view mask.
                accepted_target_mask = (
                    candidate.t1_mask
                    if action.target_view == "t1"
                    else candidate.t2_mask
                )
                self._point_session_masks[action.target_view] = np.array(
                    accepted_target_mask, dtype=bool, copy=True
                )
                self._accepted_point_clicks[action.target_view] = []
                execution["point_session_update"] = {
                    "policy": "rebase_to_accepted_live_mask",
                    "target_view": action.target_view,
                    "base_mask_pixels": int(accepted_target_mask.sum()),
                }
            elif action.action == "box":
                accepted_target_mask = (
                    candidate.t1_mask
                    if action.target_view == "t1"
                    else candidate.t2_mask
                )
                self._point_session_masks[action.target_view] = np.array(
                    accepted_target_mask, dtype=bool, copy=True
                )
                self._accepted_point_clicks[action.target_view] = []
                execution["point_session_update"] = {
                    "policy": "rebase_to_accepted_live_mask",
                    "target_view": action.target_view,
                    "base_mask_pixels": int(accepted_target_mask.sum()),
                }
        else:
            # Keep the rejected candidate in the trajectory, but continue from
            # the last accepted masks.  A verifier may optionally author a
            # replacement action using the rejected candidate as evidence.
            previous_state.step_index = next_index
            self.state = previous_state
            self.feedback = previous_feedback
            reject_callback = getattr(self.verifier, "on_candidate_rejected", None)
            if callable(reject_callback):
                reject_callback(previous_feedback)
            if action.action != "finish":
                self._rejected_action_signatures.add(action_signature)
            replan = getattr(self.verifier, "replan_after_rejection", None)
            if callable(replan):
                replan_history = self.trajectory.rejected_action_history(limit=3)
                replan_history.append(
                    self._rejection_history_record(
                        next_index,
                        action,
                        verifier_output,
                        rejection_reasons,
                        execution,
                    )
                )
                try:
                    replan_output = replan(
                        previous_state,
                        candidate,
                        previous_feedback,
                        verifier_output,
                        action,
                        tuple(rejection_reasons),
                        tuple(replan_history[-4:]),
                    )
                    if not isinstance(replan_output, VerifierOutput):
                        raise TypeError(
                            "replan_after_rejection must return a VerifierOutput"
                        )
                except Exception as error:
                    replan_output = VerifierOutput(
                        quality_score=previous_feedback.quality_score,
                        progress_score=0.0,
                        comparison="uncertain",
                        error_type=previous_feedback.error_type,
                        target_view=previous_feedback.target_view,
                        error_region=None,
                        suggested_action=None,
                        feedback="Rollback replan failed; no action is authorized.",
                        accept=False,
                        verifier_valid=False,
                        localization_valid=False,
                        stop=False,
                    )
                    execution["rollback_replan_error"] = type(error).__name__
                self.feedback = replan_output
                execution["rollback_replan"] = {
                    "verifier_output": replan_output.to_dict(),
                    "verifier_evidence": self._verifier_evidence(),
                }
        self.done = (
            next_index >= self.max_steps
            or (
                candidate_accepted
                and self.feedback is not None
                and self.feedback.stop
                and (
                    action.action == "finish"
                    or post_accept_reviewed
                )
            )
        )
        self.trajectory.append(
            TrajectoryEntry(
                next_index,
                raw_action,
                action,
                verifier_output,
                candidate.clone(),
                execution,
            )
        )
        return self.observation(), self.done

    def _coordinate_warning(self, payload: dict[str, Any]) -> str | None:
        values = payload.get("coordinate") or payload.get("box")
        if values is None:
            self._small_coordinate_streak = 0
            return None
        if all(isinstance(value, (int, float)) and 0 <= value <= 255 for value in values):
            self._small_coordinate_streak += 1
        else:
            self._small_coordinate_streak = 0
        if self._small_coordinate_streak >= 2:
            return (
                "consecutive_public_coordinates_all_le_255; interpreted as normalized_1000_xy, "
                "not auto-corrected to pixels"
            )
        return None

    def _authorized_point_proposal(
        self,
        action: AgentAction,
        region_id: str | None,
    ) -> tuple[tuple[int, int, int, int], int] | None:
        """Resolve a Verifier-selected point region to Environment-owned geometry."""

        if region_id is None:
            return None
        if self.state is None or action.coordinate is None:
            raise ActionValidationError("authorized point action lacks live state geometry")
        proposal = self._authorized_region_proposal(region_id, "point")
        try:
            box = tuple(int(value) for value in proposal["box_pixels"])
            seed = tuple(int(value) for value in proposal["component_seed_pixels"])
            component_area = int(proposal["component_area"])
        except (KeyError, TypeError, ValueError) as error:
            raise ActionValidationError(
                f"authorized point region {region_id!r} has invalid geometry"
            ) from error
        if len(box) != 4 or len(seed) != 2:
            raise ActionValidationError(
                f"authorized point region {region_id!r} has invalid geometry"
            )
        if component_area < 1:
            raise ActionValidationError(
                f"authorized point region {region_id!r} has invalid component area"
            )
        width, height = self.state.image_size
        x1, y1, x2, y2 = box
        if not (0 <= x1 <= x2 < width and 0 <= y1 <= y2 < height):
            raise ActionValidationError(
                f"authorized point region {region_id!r} has out-of-frame geometry"
            )
        if action.coordinate != seed:
            raise ActionValidationError(
                "point coordinate does not match the Environment seed for authorized "
                f"region {region_id!r}"
            )
        x, y = seed
        if not (x1 <= x <= x2 and y1 <= y <= y2):
            raise ActionValidationError(
                f"authorized point region {region_id!r} seed is outside its proposal box"
            )
        return (x1, y1, x2, y2), component_area

    def _authorized_box_proposal(
        self,
        action: AgentAction,
        region_id: str | None,
    ) -> tuple[int, int, int, int] | None:
        """Require a staged box action to match its live Environment proposal."""

        if region_id is None:
            return None
        if self.state is None or action.box is None:
            raise ActionValidationError("authorized box action lacks live state geometry")
        proposal = self._authorized_region_proposal(region_id, "box")
        try:
            box = tuple(int(value) for value in proposal["box_pixels"])
        except (KeyError, TypeError, ValueError) as error:
            raise ActionValidationError(
                f"authorized box region {region_id!r} has invalid geometry"
            ) from error
        width, height = self.state.image_size
        if len(box) != 4:
            raise ActionValidationError(
                f"authorized box region {region_id!r} has invalid geometry"
            )
        x1, y1, x2, y2 = box
        if not (0 <= x1 <= x2 < width and 0 <= y1 <= y2 < height):
            raise ActionValidationError(
                f"authorized box region {region_id!r} has out-of-frame geometry"
            )
        if action.box != box:
            raise ActionValidationError(
                "box geometry does not match the Environment box for authorized "
                f"region {region_id!r}"
            )
        return x1, y1, x2, y2

    def _authorized_region_proposal(
        self, region_id: str, action_kind: str
    ) -> Mapping[str, Any]:
        if self.state is None:
            raise ActionValidationError(
                f"authorized {action_kind} action lacks live state geometry"
            )
        proposals = self.state.evidence.get("verifier_region_proposals", [])
        proposal = next(
            (
                item
                for item in proposals
                if isinstance(item, Mapping) and item.get("region_id") == region_id
            ),
            None,
        )
        if proposal is None:
            raise ActionValidationError(
                f"authorized {action_kind} region {region_id!r} is absent from the live state"
            )
        return proposal

    def _with_verifier_evidence(self, execution: dict[str, Any]) -> dict[str, Any]:
        result = dict(execution)
        evidence = self._verifier_evidence()
        if evidence:
            result["verifier_evidence"] = evidence
        return result

    def _verifier_evidence(self) -> dict[str, Any] | None:
        evidence = getattr(self.verifier, "last_evidence", None)
        return dict(evidence) if isinstance(evidence, Mapping) else None

    @staticmethod
    def _rejection_history_record(
        step_index: int,
        action: AgentAction,
        verifier_output: VerifierOutput,
        rejection_reasons: list[str],
        execution: Mapping[str, Any],
    ) -> dict[str, Any]:
        locality = execution.get("locality", {})
        return {
            "step_index": step_index,
            "action": action.to_dict(),
            "rejection_reasons": list(rejection_reasons),
            "candidate_comparison": verifier_output.comparison,
            "candidate_accept": bool(verifier_output.accept),
            "candidate_feedback": verifier_output.feedback[:400],
            "candidate_area_delta": execution.get("candidate_area_delta"),
            "changed_pixels": locality.get("changed_pixels"),
            "outside_roi_ratio": locality.get("outside_roi_ratio"),
        }

    @staticmethod
    def _action_signature(state: ChangeState, action: AgentAction) -> str:
        digest = hashlib.sha256()
        for mask in (state.t1_mask, state.t2_mask, state.change_mask):
            value = np.ascontiguousarray(mask, dtype=np.uint8)
            digest.update(str(value.shape).encode("ascii"))
            digest.update(value.tobytes())
        digest.update(
            json.dumps(
                action.to_dict(), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
        return digest.hexdigest()

    def _initial_finish_authorized(self) -> bool:
        """Allow a verified, error-free initial state to finish without a no-op tool."""

        feedback = self.feedback
        return bool(
            feedback is not None
            and feedback.verifier_valid
            and feedback.comparison == "initial"
            and feedback.error_type == "none"
            and feedback.stop
        )

    def _attach_verifier_regions(
        self, state: ChangeState, previous_state: ChangeState | None
    ) -> None:
        attach_verifier_regions(
            state,
            previous_state,
            max_regions=self.verifier_max_regions,
            max_delta_regions=self.verifier_max_delta_regions,
            min_component_area=self.verifier_min_region_area,
            padding_ratio=self.verifier_region_padding_ratio,
            empty_mask_scan_grid_size=self.empty_mask_scan_grid_size,
        )

    @property
    def best_state(self) -> ChangeState:
        return self.trajectory.best_entry.state.clone()

    @staticmethod
    def _state_from_update(
        t1_image: np.ndarray,
        t2_image: np.ndarray,
        query: str,
        t1_mask: np.ndarray,
        t2_mask: np.ndarray,
        update: PairUpdate,
        step_index: int,
    ) -> ChangeState:
        return ChangeState(
            t1_image=t1_image,
            t2_image=t2_image,
            query=query,
            t1_mask=t1_mask,
            t2_mask=t2_mask,
            change_mask=update.change_mask,
            t1_instances=update.t1_instances,
            t2_instances=update.t2_instances,
            matching=update.matching,
            evidence=update.evidence,
            step_index=step_index,
        )
