"""Positive and fail-closed fixtures for the frozen summary-v2 contract."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = "test_1_1"
PROFILE = "p1_smoke1"
EVALUATION_ID = "DEV-SMOKE-1-H1-V1"
ARM = "v12_1-h0-legacy-single-round-base-0802"
HARNESS_PROTOCOL = "qwen3vl_harness_h0_legacy_single_round_v1"
RUNTIME_COMMIT = "a" * 40
HARNESS_COMMIT = "b" * 40


def load_tool(name: str) -> ModuleType:
    path = ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load tool: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SUMMARIZER = load_tool("summarize_qwen3vl_e2e_run")
ANALYZER = load_tool("analyze_staged_verifier_run")
VALIDATOR = load_tool("validate_qwen3vl_e2e_summary")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def metric_block(tp: int, fp: int, fn: int, tn: int) -> dict[str, Any]:
    counts = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    rates = SUMMARIZER.rates_from_counts(counts)
    return {
        **{key: round(rates[key], 8) for key in ("iou", "precision", "recall", "f1")},
        **counts,
    }


def make_run(run_dir: Path) -> None:
    counts = metric_block(4, 1, 1, 10)
    backend_call = {
        "backend": "fixture",
        "stage": "audit",
        "attempt_kind": "initial",
        "network_retry_attempt_count": 0,
        "prompt_sha256": "c" * 64,
        "raw_response": "{}",
        "parsed_output": {"region_id": "r0"},
        "visual_input_scope": "standard",
    }
    evidence = {
        "type": "staged_qwen_verifier",
        "schema_version": "staged_verifier_environment_ordered_compact_audit_v12_1",
        "verifier_valid": True,
        "localization_valid": True,
        "validation_errors": [],
        "regional_validation_errors": [],
        "backend_calls": [backend_call],
        "stage_trace": {
            "mode": "initial",
            "evidence": [{"region_id": "r0", "audit_kind": "present"}],
            "audited_region_ids": ["r0"],
            "judgments": [{"region_id": "r0"}],
            "diagnoses": [{"region_id": "r0", "error_type": "none"}],
            "plan": None,
            "replan_evidence": [],
            "replan_audited_region_ids": [],
            "replan_judgments": [],
            "replan_diagnoses": [],
        },
    }
    trajectory = {
        "steps": [
            {
                "step_index": 0,
                "offline_metrics": counts,
                "execution": {
                    "verifier_evidence": evidence,
                },
            }
        ]
    }
    budget = {
        "verifier_round_count": 1,
        "stage_initial_attempt_count": 1,
        "schema_repair_attempt_count": 0,
        "network_retry_attempt_count": 0,
        "tool_action_attempt_count": 0,
        "tool_failure_count": 0,
        "environment_step_count": 0,
    }
    gt_audit = {
        "pre_rollout_gt_access_count": 0,
        "violation_ids": [],
        "rollout_completed_before_offline_scoring": True,
    }
    safety_audit = {"relaxation_count": 0, "relaxed_boundaries": []}
    p1_sample = {
        "terminal_review": None,
        "terminal_reviews": [],
        "budget_accounting": budget,
        "gt_boundary_audit": gt_audit,
        "safety_boundary_audit": safety_audit,
        "final_region_protocol_valid": True,
    }
    episode = {
        "sample": SAMPLE,
        "execution_status": "completed",
        "episode_outcome": "no_executable_action",
        "converged": False,
        "stop_reason": "verifier_no_executable_action",
        "elapsed_seconds": 1.0,
        "loop_count": 0,
        "candidate_count": 0,
        "accepted_candidate_count": 0,
        "rejected_candidate_count": 0,
        "tool_action_count": 0,
        "empty_mask_scan_fallback_action_count": 0,
        "action_attempt_count": 0,
        "action_attempt_error_count": 0,
        "harness_protocol_id": HARNESS_PROTOCOL,
        "experiment_arm_id": ARM,
        "terminal_review_factor_id": "terminal_review_scheduler_v1",
        "terminal_review_arm": "h0",
        **p1_sample,
    }
    write_json(
        run_dir / "rollout_complete.json",
        {"execution_status": "completed", "gt_loaded": False, "samples": [SAMPLE]},
    )
    write_json(
        run_dir / "per_sample_metrics.json",
        {
            "samples": {SAMPLE: {"initial": counts, "verifier_selected": counts}},
            "aggregate": counts,
        },
    )
    write_json(run_dir / "trajectories" / SAMPLE / "trajectory.json", trajectory)
    write_json(run_dir / "trajectories" / SAMPLE / "episode_summary.json", episode)
    write_json(
        run_dir / "p1_harness_audit.json",
        {
            "schema_id": "qwen3vl_p1_harness_audit_v1",
            "factor_id": "terminal_review_scheduler_v1",
            "harness_protocol_id": HARNESS_PROTOCOL,
            "experiment_arm_id": ARM,
            "arm": "h0",
            "requested_sample_ids": [SAMPLE],
            "per_sample": {SAMPLE: p1_sample},
            "totals": budget,
            "gt_boundary_violation_count": 0,
            "safety_boundary_relaxation_count": 0,
            "tool_failure_count": 0,
            "written_before_rollout_complete": True,
            "gt_loaded": False,
        },
    )


def summarize(run_dir: Path) -> dict[str, Any]:
    return SUMMARIZER.summarize(
        argparse.Namespace(
            run_dir=run_dir,
            output=run_dir / "unused.json",
            evaluation_profile=PROFILE,
            evaluation_id=EVALUATION_ID,
            model_state="base",
            job_id="123",
            experiment_arm=ARM,
            protocol_version="v12.1",
            harness_protocol_id=HARNESS_PROTOCOL,
            runtime_commit=RUNTIME_COMMIT,
            harness_commit=HARNESS_COMMIT,
            p1_harness_audit=None,
            expected_samples=[SAMPLE],
        )
    )


def validator_args(run_dir: Path, summary_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        summary=summary_path,
        run_dir=run_dir,
        evaluation_profile=PROFILE,
        evaluation_id=EVALUATION_ID,
        expected_samples=[SAMPLE],
        experiment_arm=ARM,
        harness_protocol_id=HARNESS_PROTOCOL,
        runtime_commit=RUNTIME_COMMIT,
        harness_commit=HARNESS_COMMIT,
        job_id="123",
    )


class Qwen3VLSummaryValidatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temporary.name) / "run"
        make_run(self.run_dir)
        self.summary = summarize(self.run_dir)
        self.summary_path = Path(self.temporary.name) / "summary.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def validate(self, payload: dict[str, Any]) -> dict[str, Any]:
        write_json(self.summary_path, payload)
        return VALIDATOR.validate_summary(validator_args(self.run_dir, self.summary_path))

    def assert_rejected(self, mutation: Callable[[dict[str, Any]], None]) -> None:
        payload = copy.deepcopy(self.summary)
        mutation(payload)
        with self.assertRaises(ValueError):
            self.validate(payload)

    def test_accepts_exact_hash_bound_p1_summary(self) -> None:
        result = self.validate(self.summary)
        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["scored_samples"], 1)
        self.assertEqual(
            self.summary["p1_contract"]["budget_accounting_totals"][
                "verifier_round_count"
            ],
            1,
        )

    def test_terminal_trigger_rebuild_matches_production_stage_decisions(self) -> None:
        def raw_call(stage: str, region_id: str, prompt: str) -> dict[str, Any]:
            return {
                "backend": "fixture",
                "stage": stage,
                "attempt_kind": "initial",
                "network_retry_attempt_count": 0,
                "prompt_sha256": prompt * 64,
                "raw_response": "{}",
                "parsed_output": {"region_id": region_id},
                "visual_input_scope": "standard",
            }

        initial = {
            "type": "staged_qwen_verifier",
            "schema_version": (
                "staged_verifier_environment_ordered_compact_audit_v12_1"
            ),
            "verifier_valid": True,
            "validation_errors": [],
            "regional_validation_errors": [],
            "stage_trace": {
                "mode": "initial",
                "evidence": [{"region_id": "r0", "audit_kind": "present"}],
                "audited_region_ids": ["r0"],
                "judgments": [{"region_id": "r0"}],
                "diagnoses": [{"region_id": "r0", "error_type": "none"}],
                "plan": {"action": "finish"},
                "decision": {
                    "comparison": "initial",
                    "accept": False,
                    "stop": False,
                },
                "state_completion_gate_passed": True,
            },
            "backend_calls": [raw_call("audit", "r0", "a")],
        }
        candidate = {
            **copy.deepcopy(initial),
            "stage_trace": {
                "mode": "candidate",
                "evidence": [
                    {"region_id": "d0", "audit_kind": "delta_added"}
                ],
                "audited_region_ids": ["d0"],
                "judgments": [{"region_id": "d0"}],
                "diagnoses": [],
                "replan_evidence": [
                    {"region_id": "r0", "audit_kind": "present"}
                ],
                "replan_audited_region_ids": ["r0"],
                "replan_judgments": [{"region_id": "r0"}],
                "replan_diagnoses": [
                    {"region_id": "r0", "error_type": "none"}
                ],
                "plan": {"action": "finish"},
                "decision": {
                    "comparison": "better",
                    "accept": True,
                    "stop": False,
                },
                "state_completion_gate_passed": True,
            },
            "backend_calls": [
                raw_call("candidate_evidence", "d0", "b"),
                raw_call("audit", "r0", "c"),
            ],
        }

        for evidence in (initial, candidate):
            with self.subTest(mode=evidence["stage_trace"]["mode"]):
                self.assertEqual(
                    SUMMARIZER.rebuild_terminal_trigger_from_round_one(evidence),
                    "provisional_finish",
                )
                self.assertEqual(
                    VALIDATOR.independently_rebuild_terminal_trigger(evidence),
                    "provisional_finish",
                )

        invalid_mutations = (
            lambda item: item["stage_trace"]["decision"].update(
                {"accept": True, "stop": True}
            ),
            lambda item: item["stage_trace"].__setitem__(
                "state_completion_gate_passed", False
            ),
            lambda item: item["stage_trace"].__setitem__(
                "plan", {"action": "box"}
            ),
            lambda item: item["stage_trace"]["diagnoses"][0].__setitem__(
                "error_type", "mixed_error"
            ),
        )
        for index, mutation in enumerate(invalid_mutations):
            with self.subTest(initial_invalid=index):
                malformed = copy.deepcopy(initial)
                mutation(malformed)
                self.assertIsNone(
                    SUMMARIZER.rebuild_terminal_trigger_from_round_one(malformed)
                )
                self.assertIsNone(
                    VALIDATOR.independently_rebuild_terminal_trigger(malformed)
                )

        for field, value in (
            ("accept", False),
            ("stop", True),
            ("comparison", "unchanged"),
        ):
            with self.subTest(candidate_invalid=field):
                malformed_candidate = copy.deepcopy(candidate)
                malformed_candidate["stage_trace"]["decision"][field] = value
                self.assertIsNone(
                    SUMMARIZER.rebuild_terminal_trigger_from_round_one(
                        malformed_candidate
                    )
                )
                self.assertIsNone(
                    VALIDATOR.independently_rebuild_terminal_trigger(
                        malformed_candidate
                    )
                )

        malformed_initial = copy.deepcopy(initial)
        malformed_initial["stage_trace"]["decision"]["comparison"] = "better"
        self.assertIsNone(
            SUMMARIZER.rebuild_terminal_trigger_from_round_one(malformed_initial)
        )
        self.assertIsNone(
            VALIDATOR.independently_rebuild_terminal_trigger(malformed_initial)
        )

    def test_legacy_smoke_profile_remains_distinct(self) -> None:
        for implementation in (SUMMARIZER, ANALYZER, VALIDATOR):
            self.assertEqual(
                implementation.EVALUATION_PROFILES["smoke1"],
                ("DEV-SMOKE-1-V1", ("test_20_15",)),
            )
            self.assertEqual(
                implementation.EVALUATION_PROFILES["p1_smoke1"],
                ("DEV-SMOKE-1-H1-V1", ("test_1_1",)),
            )
        self.assertIn("confirmed_finish", SUMMARIZER.FINISH_REASONS)
        self.assertIn("confirmed_finish", VALIDATOR.FINISH_REASONS)

    def test_schema_profile_order_arm_protocol_and_count_tampering_fail_closed(self) -> None:
        mutations = (
            lambda payload: payload.__setitem__(
                "summary_schema_version", "qwen3vl_e2e_summary_v1"
            ),
            lambda payload: payload.__setitem__("evaluation_profile", "smoke1"),
            lambda payload: payload.__setitem__("evaluation_id", "DEV-SMOKE-1-V1"),
            lambda payload: payload.__setitem__("sample_order", []),
            lambda payload: payload.__setitem__("requested_samples", 2),
            lambda payload: payload.__setitem__("scored_samples", 0),
            lambda payload: payload.__setitem__(
                "experiment_arm", "v12_1-h1-gtfree-terminal-review-base-0802"
            ),
            lambda payload: payload.__setitem__(
                "harness_protocol_id",
                "qwen3vl_harness_h1_gtfree_terminal_review_v1",
            ),
            lambda payload: payload["per_sample"][SAMPLE]["selected"].__setitem__(
                "tp", 5
            ),
            lambda payload: payload["aggregate"]["selected"].__setitem__("iou", 0.0),
            lambda payload: payload["macro"]["selected"].__setitem__("f1", 0.0),
        )
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                self.assert_rejected(mutation)

    def test_budget_gt_safety_tool_and_artifact_tampering_fail_closed(self) -> None:
        mutations = (
            lambda payload: payload["p1_contract"]["budget_accounting_totals"].__setitem__(
                "verifier_round_count", 3
            ),
            lambda payload: payload["p1_contract"]["per_sample"][SAMPLE][
                "budget_accounting"
            ].pop("schema_repair_attempt_count"),
            lambda payload: payload["p1_contract"]["per_sample"][SAMPLE][
                "budget_accounting"
            ].__setitem__("network_retry_attempt_count", -1),
            lambda payload: payload["p1_contract"]["hard_gates"].__setitem__(
                "pre_rollout_gt_violation_count", 1
            ),
            lambda payload: payload["p1_contract"]["hard_gates"].__setitem__(
                "safety_boundary_relaxation_count", 1
            ),
            lambda payload: payload["actions"].__setitem__("tool_failure_count", 1),
            lambda payload: payload["artifact_bindings"]["source_metrics"].__setitem__(
                "sha256", "0" * 64
            ),
            lambda payload: payload["artifact_bindings"]["rollout_marker"].__setitem__(
                "path", "/tmp/not-the-bound-rollout-marker.json"
            ),
        )
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                self.assert_rejected(mutation)

    def test_raw_artifact_mutation_invalidates_hash_binding(self) -> None:
        metrics_path = self.run_dir / "per_sample_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["samples"][SAMPLE]["initial"]["tp"] += 1
        write_json(metrics_path, metrics)
        with self.assertRaises(ValueError):
            self.validate(self.summary)

    def test_h0_raw_audit_cannot_claim_terminal_review(self) -> None:
        audit_path = self.run_dir / "p1_harness_audit.json"
        episode_path = self.run_dir / "trajectories" / SAMPLE / "episode_summary.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        episode = json.loads(episode_path.read_text(encoding="utf-8"))
        audit["per_sample"][SAMPLE]["terminal_review"] = {}
        episode["terminal_review"] = {}
        write_json(audit_path, audit)
        write_json(episode_path, episode)
        with self.assertRaises(ValueError):
            summarize(self.run_dir)

    def test_final_region_protocol_gate_is_rebuilt_from_raw_trajectory(self) -> None:
        audit_path = self.run_dir / "p1_harness_audit.json"
        episode_path = self.run_dir / "trajectories" / SAMPLE / "episode_summary.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        episode = json.loads(episode_path.read_text(encoding="utf-8"))
        audit["per_sample"][SAMPLE]["final_region_protocol_valid"] = False
        episode["final_region_protocol_valid"] = False
        write_json(audit_path, audit)
        write_json(episode_path, episode)
        with self.assertRaises(ValueError):
            summarize(self.run_dir)

        make_run(self.run_dir)
        trajectory_path = self.run_dir / "trajectories" / SAMPLE / "trajectory.json"
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        trajectory["steps"][0]["execution"]["verifier_evidence"][
            "backend_calls"
        ][0].pop("parsed_output")
        write_json(trajectory_path, trajectory)
        with self.assertRaises(ValueError):
            summarize(self.run_dir)

    def test_h0_final_programmatic_or_malformed_evidence_does_not_fallback(self) -> None:
        trajectory_path = self.run_dir / "trajectories" / SAMPLE / "trajectory.json"
        valid_trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        for final_evidence in (
            {
                "type": "staged_qwen_verifier",
                "schema_version": (
                    "staged_verifier_environment_ordered_compact_audit_v12_1"
                ),
                "decision_mode": "programmatic_no_alternate_region_after_rollback",
                "verifier_valid": True,
                "validation_errors": [],
                "backend_calls": [],
            },
            {
                "type": "pre_qwen_hard_gate",
                "verifier_valid": False,
                "validation_errors": ["candidate rejected before regional review"],
                "backend_calls": [],
            },
        ):
            with self.subTest(final_evidence=final_evidence):
                trajectory = copy.deepcopy(valid_trajectory)
                trajectory["steps"].append(
                    {
                        "step_index": 1,
                        "execution": {"verifier_evidence": final_evidence},
                    }
                )
                self.assertFalse(
                    SUMMARIZER.rebuild_final_region_protocol_valid(
                        trajectory,
                        arm="h0",
                        review_records=[],
                    )
                )
                rebuilt, reviews = (
                    VALIDATOR.independently_rebuild_final_region_protocol_valid(
                        trajectory,
                        arm="h0",
                    )
                )
                self.assertFalse(rebuilt)
                self.assertEqual(reviews, [])

    def test_final_region_protocol_requires_complete_raw_call_groups(self) -> None:
        trajectory_path = self.run_dir / "trajectories" / SAMPLE / "trajectory.json"
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        evidence = trajectory["steps"][0]["execution"]["verifier_evidence"]
        repaired = copy.deepcopy(evidence)
        repaired["backend_calls"][0].pop("parsed_output")
        repaired["backend_calls"][0]["parse_error"] = "schema parse failed"
        repair_call = copy.deepcopy(evidence["backend_calls"][0])
        repair_call["attempt_kind"] = "repair"
        repaired["backend_calls"].append(repair_call)
        self.assertTrue(
            SUMMARIZER.staged_region_protocol_valid_from_raw_evidence(
                repaired,
                require_round_two=False,
            )
        )
        self.assertTrue(
            VALIDATOR.independently_validate_staged_region_evidence(
                repaired,
                require_round_two=False,
            )
        )
        excessive_repairs = copy.deepcopy(repaired)
        excessive_repairs["backend_calls"].append(copy.deepcopy(repair_call))
        self.assertFalse(
            SUMMARIZER.staged_region_protocol_valid_from_raw_evidence(
                excessive_repairs,
                require_round_two=False,
            )
        )
        self.assertFalse(
            VALIDATOR.independently_validate_staged_region_evidence(
                excessive_repairs,
                require_round_two=False,
            )
        )
        for mutation in (
            lambda item: item["backend_calls"][0].pop("raw_response"),
            lambda item: item["backend_calls"][0].pop("stage"),
            lambda item: item["backend_calls"].append(
                copy.deepcopy(item["backend_calls"][0])
            ),
            lambda item: item.__setitem__(
                "regional_validation_errors",
                [{"region_id": "r0", "error": "invalid"}],
            ),
        ):
            with self.subTest(mutation=mutation):
                mutated = copy.deepcopy(evidence)
                mutation(mutated)
                self.assertFalse(
                    SUMMARIZER.staged_region_protocol_valid_from_raw_evidence(
                        mutated,
                        require_round_two=False,
                    )
                )
                self.assertFalse(
                    VALIDATOR.independently_validate_staged_region_evidence(
                        mutated,
                        require_round_two=False,
                    )
                )

    def test_proposal_version_raw_replay_and_declared_tamper_fail_closed(
        self,
    ) -> None:
        scan_version = "qwen3vl_empty_dual_mask_scan_grid_v1"

        def staged_evidence(
            region_id: str,
            *,
            audit_kind: str,
            proposal_version: str | None,
            visual_input_scope: str,
        ) -> dict[str, Any]:
            proposal: dict[str, Any] = {
                "region_id": region_id,
                "audit_kind": audit_kind,
            }
            if proposal_version is not None:
                proposal["proposal_version"] = proposal_version
            context = {
                "round_index": 2,
                "round_kind": "finish_confirmation",
                "trigger": "provisional_finish",
                "fresh_backend_context": True,
                "inherits_hidden_state": False,
                "copies_round_one_raw_response": False,
                "selection_policy_version": (
                    "environment_ordered_terminal_review_v1"
                ),
                "preferred_region_id": None,
                "prompt_version": None,
                "selection_branch": None,
                "review_region_id": region_id,
                "proposal_version": proposal_version,
            }
            return {
                "type": "staged_qwen_verifier",
                "schema_version": (
                    "staged_verifier_environment_ordered_compact_audit_v12_1"
                ),
                "verifier_valid": True,
                "validation_errors": [],
                "regional_validation_errors": [],
                "stage_trace": {
                    "mode": "initial",
                    "evidence": [proposal],
                    "audited_region_ids": [region_id],
                    "judgments": [{"region_id": region_id}],
                    "diagnoses": [
                        {"region_id": region_id, "error_type": "none"}
                    ],
                    "plan": {"action": "finish"},
                },
                "backend_calls": [
                    {
                        "backend": "fixture",
                        "stage": "audit",
                        "attempt_kind": "initial",
                        "network_retry_attempt_count": 0,
                        "prompt_sha256": "d" * 64,
                        "raw_response": "{}",
                        "parsed_output": {"region_id": region_id},
                        "visual_input_scope": visual_input_scope,
                        "terminal_review": context,
                    }
                ],
                "terminal_review_round": {
                    "schema_valid": True,
                    "completion_gate_passed": True,
                },
            }

        def assert_raw_replay(value: dict[str, Any], expected: bool) -> None:
            self.assertIs(
                SUMMARIZER.staged_region_protocol_valid_from_raw_evidence(
                    value,
                    require_round_two=True,
                ),
                expected,
            )
            self.assertIs(
                VALIDATOR.independently_validate_staged_region_evidence(
                    value,
                    require_round_two=True,
                ),
                expected,
            )

        scan = staged_evidence(
            "s1",
            audit_kind="scan",
            proposal_version=scan_version,
            visual_input_scope="authoritative_pair_rgb_only_scan",
        )
        ordinary = staged_evidence(
            "r0",
            audit_kind="present",
            proposal_version=None,
            visual_input_scope="standard",
        )
        assert_raw_replay(scan, True)
        assert_raw_replay(ordinary, True)
        explicit_null = copy.deepcopy(ordinary)
        explicit_null["stage_trace"]["evidence"][0]["proposal_version"] = None
        assert_raw_replay(explicit_null, True)

        for region_id in ("s", "s-1", "sx"):
            with self.subTest(ordinary_shape=region_id):
                shaped_ordinary = staged_evidence(
                    region_id,
                    audit_kind="present",
                    proposal_version=None,
                    visual_input_scope="standard",
                )
                assert_raw_replay(shaped_ordinary, True)
                false_scan = staged_evidence(
                    region_id,
                    audit_kind="scan",
                    proposal_version=scan_version,
                    visual_input_scope="authoritative_pair_rgb_only_scan",
                )
                assert_raw_replay(false_scan, False)

        scan_mutations = (
            lambda item: item["stage_trace"]["evidence"][0].pop(
                "proposal_version"
            ),
            lambda item: item["stage_trace"]["evidence"][0].__setitem__(
                "proposal_version", "qwen3vl_wrong_scan_grid_v1"
            ),
            lambda item: item["stage_trace"]["evidence"][0].__setitem__(
                "audit_kind", "present"
            ),
            lambda item: item["backend_calls"][0]["terminal_review"].pop(
                "proposal_version"
            ),
            lambda item: item["backend_calls"][0]["terminal_review"].__setitem__(
                "proposal_version", "qwen3vl_wrong_scan_grid_v1"
            ),
            lambda item: item["backend_calls"][0].__setitem__(
                "visual_input_scope", "standard"
            ),
        )
        for index, mutation in enumerate(scan_mutations):
            with self.subTest(scan_mutation=index):
                mutated = copy.deepcopy(scan)
                mutation(mutated)
                assert_raw_replay(mutated, False)

        ordinary_mutations = (
            lambda item: item["stage_trace"]["evidence"][0].__setitem__(
                "proposal_version", scan_version
            ),
            lambda item: item["stage_trace"]["evidence"][0].__setitem__(
                "audit_kind", "scan"
            ),
            lambda item: item["backend_calls"][0].__setitem__(
                "visual_input_scope", "authoritative_pair_rgb_only_scan"
            ),
        )
        for index, mutation in enumerate(ordinary_mutations):
            with self.subTest(ordinary_mutation=index):
                mutated = copy.deepcopy(ordinary)
                mutation(mutated)
                assert_raw_replay(mutated, False)

        raw_call = scan["backend_calls"][0]
        normalized_call = {
            "round_index": 2,
            "stage": raw_call["stage"],
            "region_id": "s1",
            "attempt_kind": raw_call["attempt_kind"],
            "prompt_sha256": raw_call["prompt_sha256"],
            "raw_response": raw_call["raw_response"],
            "parsed_output": raw_call["parsed_output"],
            "visual_input_scope": raw_call["visual_input_scope"],
            "network_retry_attempt_count": raw_call[
                "network_retry_attempt_count"
            ],
        }
        round_one = copy.deepcopy(ordinary)
        round_one["backend_calls"][0].pop("terminal_review")
        round_one.pop("terminal_review_round")
        raw_record = {
            "round_one_evidence": round_one,
            "round_two_evidence": scan,
        }
        normalized = {
            "round_two_invoked": True,
            "trigger": "provisional_finish",
            "backend_calls": [normalized_call],
        }
        SUMMARIZER.validate_terminal_review_independence(
            raw_record,
            normalized,
            source="fixture",
        )
        VALIDATOR.independently_validate_review_identity_and_selection(
            raw_record,
            normalized,
            name="fixture",
        )
        normalized_mutations = (
            lambda item: item["backend_calls"][0].__setitem__(
                "proposal_version", scan_version
            ),
            lambda item: item["backend_calls"][0].__setitem__(
                "region_id", "r0"
            ),
            lambda item: item["backend_calls"][0].__setitem__(
                "parsed_output", {"region_id": "r0"}
            ),
            lambda item: item["backend_calls"][0].__setitem__(
                "visual_input_scope", "standard"
            ),
        )
        for index, mutation in enumerate(normalized_mutations):
            with self.subTest(normalized_mutation=index):
                mutated = copy.deepcopy(normalized)
                mutation(mutated)
                with self.assertRaises(ValueError):
                    SUMMARIZER.validate_terminal_review_independence(
                        raw_record,
                        mutated,
                        source="fixture",
                    )
                with self.assertRaises(ValueError):
                    VALIDATOR.independently_validate_review_identity_and_selection(
                        raw_record,
                        mutated,
                        name="fixture",
                    )

        repaired_round_one = copy.deepcopy(scan)
        repaired_round_one.pop("terminal_review_round")
        failed_initial = repaired_round_one["backend_calls"][0]
        failed_initial.pop("terminal_review")
        failed_initial.pop("parsed_output")
        failed_initial["parse_error"] = "schema parse failed"
        failed_initial["prompt_sha256"] = "e" * 64
        successful_repair = copy.deepcopy(failed_initial)
        successful_repair["attempt_kind"] = "repair"
        successful_repair["prompt_sha256"] = "f" * 64
        successful_repair.pop("parse_error")
        successful_repair["parsed_output"] = {"region_id": "s1"}
        repaired_round_one["backend_calls"].append(successful_repair)
        for reused_kind, reused_prompt in (
            ("failed_initial", "e" * 64),
            ("repair", "f" * 64),
        ):
            with self.subTest(reused_round_one_identity=reused_kind):
                duplicate_round_two = copy.deepcopy(scan)
                duplicate_round_two["backend_calls"][0][
                    "prompt_sha256"
                ] = reused_prompt
                duplicate_record = {
                    "round_one_evidence": repaired_round_one,
                    "round_two_evidence": duplicate_round_two,
                }
                duplicate_normalized = copy.deepcopy(normalized)
                duplicate_normalized["backend_calls"][0][
                    "prompt_sha256"
                ] = reused_prompt
                with self.assertRaises(ValueError):
                    SUMMARIZER.validate_terminal_review_independence(
                        duplicate_record,
                        duplicate_normalized,
                        source="fixture",
                    )
                with self.assertRaises(ValueError):
                    VALIDATOR.independently_validate_review_identity_and_selection(
                        duplicate_record,
                        duplicate_normalized,
                        name="fixture",
                    )

        raw_review = {
            "eligible": True,
            "round_two_invoked": True,
            "round_two_count": 1,
            "round_index": 2,
            "round_one_evidence": {"backend_calls": []},
            "round_two_evidence": scan,
        }
        trajectory = {
            "steps": [
                {
                    "execution": {
                        "verifier_evidence": {"terminal_review": raw_review}
                    }
                }
            ]
        }
        records = SUMMARIZER.terminal_review_records_from_trajectory(
            trajectory,
            source="fixture",
        )
        self.assertTrue(
            SUMMARIZER.rebuild_final_region_protocol_valid(
                trajectory,
                arm="h1",
                review_records=records,
            )
        )
        rebuilt, _ = VALIDATOR.independently_rebuild_final_region_protocol_valid(
            trajectory,
            arm="h1",
        )
        self.assertTrue(rebuilt)

        later_unreviewed = copy.deepcopy(ordinary)
        later_unreviewed["backend_calls"][0].pop("terminal_review")
        later_unreviewed.pop("terminal_review_round")
        trajectory["steps"].append(
            {"execution": {"verifier_evidence": later_unreviewed}}
        )
        stale_records = SUMMARIZER.terminal_review_records_from_trajectory(
            trajectory,
            source="fixture",
        )
        self.assertFalse(
            SUMMARIZER.rebuild_final_region_protocol_valid(
                trajectory,
                arm="h1",
                review_records=stale_records,
            )
        )
        rebuilt, _ = VALIDATOR.independently_rebuild_final_region_protocol_valid(
            trajectory,
            arm="h1",
        )
        self.assertFalse(rebuilt)

    def test_h1_schema_valid_round_two_is_required_for_final_region(self) -> None:
        context = {
            "round_index": 2,
            "round_kind": "finish_confirmation",
            "trigger": "provisional_finish",
            "fresh_backend_context": True,
            "inherits_hidden_state": False,
            "copies_round_one_raw_response": False,
            "selection_policy_version": "environment_ordered_terminal_review_v1",
            "preferred_region_id": None,
            "prompt_version": None,
            "selection_branch": None,
            "review_region_id": "r0",
            "proposal_version": None,
        }
        round_two = {
            "type": "staged_qwen_verifier",
            "schema_version": (
                "staged_verifier_environment_ordered_compact_audit_v12_1"
            ),
            "verifier_valid": True,
            "validation_errors": [],
            "regional_validation_errors": [],
            "stage_trace": {
                "mode": "initial",
                "evidence": [{"region_id": "r0", "audit_kind": "present"}],
                "audited_region_ids": ["r0"],
                "judgments": [{"region_id": "r0"}],
                "diagnoses": [{"region_id": "r0", "error_type": "none"}],
                "plan": {"action": "finish"},
            },
            "backend_calls": [
                {
                    "backend": "fixture",
                    "stage": "audit",
                    "attempt_kind": "initial",
                    "network_retry_attempt_count": 0,
                    "prompt_sha256": "d" * 64,
                    "raw_response": "{}",
                    "parsed_output": {"region_id": "r0"},
                    "visual_input_scope": "standard",
                    "terminal_review": context,
                }
            ],
            "terminal_review_round": {
                "schema_valid": True,
                "completion_gate_passed": True,
            },
        }
        audit = {
            "eligible": True,
            "round_two_invoked": True,
            "round_two_count": 1,
            "round_index": 2,
        }
        record = {
            "audit": audit,
            "round_one_evidence": {},
            "round_two_evidence": round_two,
        }
        trajectory = {
            "steps": [
                {
                    "execution": {
                        "verifier_evidence": {
                            "terminal_review": {
                                **audit,
                                "round_one_evidence": {},
                                "round_two_evidence": round_two,
                            }
                        }
                    }
                }
            ]
        }
        self.assertTrue(
            SUMMARIZER.rebuild_final_region_protocol_valid(
                trajectory,
                arm="h1",
                review_records=[record],
            )
        )
        rebuilt, _ = VALIDATOR.independently_rebuild_final_region_protocol_valid(
            trajectory,
            arm="h1",
        )
        self.assertTrue(rebuilt)
        for mutation in (
            lambda item: item.pop("terminal_review_round"),
            lambda item: item["terminal_review_round"].__setitem__(
                "schema_valid", False
            ),
            lambda item: item.pop("backend_calls"),
        ):
            with self.subTest(mutation=mutation):
                invalid_round_two = copy.deepcopy(round_two)
                mutation(invalid_round_two)
                invalid_record = {**record, "round_two_evidence": invalid_round_two}
                invalid_trajectory = copy.deepcopy(trajectory)
                invalid_trajectory["steps"][0]["execution"]["verifier_evidence"][
                    "terminal_review"
                ]["round_two_evidence"] = invalid_round_two
                self.assertFalse(
                    SUMMARIZER.rebuild_final_region_protocol_valid(
                        invalid_trajectory,
                        arm="h1",
                        review_records=[invalid_record],
                    )
                )
                invalid_rebuilt, _ = (
                    VALIDATOR.independently_rebuild_final_region_protocol_valid(
                        invalid_trajectory,
                        arm="h1",
                    )
                )
                self.assertFalse(invalid_rebuilt)

    def test_malformed_terminal_review_tail_fails_closed(self) -> None:
        trajectory = {
            "steps": [
                {
                    "execution": {
                        "verifier_evidence": {"terminal_review": "malformed"}
                    }
                }
            ]
        }
        with self.assertRaises(ValueError):
            SUMMARIZER.terminal_review_records_from_trajectory(
                trajectory,
                source="fixture",
            )
        with self.assertRaises(ValueError):
            VALIDATOR.independent_terminal_review_records(trajectory)

    def test_h1_budget_exhausted_without_round_two_cannot_claim_final_region(self) -> None:
        audit = {
            "eligible": True,
            "round_two_invoked": False,
            "round_two_count": 0,
            "round_index": None,
        }
        trajectory = {
            "steps": [
                {
                    "execution": {
                        "verifier_evidence": {
                            "terminal_review": {
                                **audit,
                                "round_one_evidence": {},
                                "round_two_evidence": None,
                            }
                        }
                    }
                }
            ]
        }
        records = SUMMARIZER.terminal_review_records_from_trajectory(
            trajectory,
            source="fixture",
        )
        self.assertFalse(
            SUMMARIZER.rebuild_final_region_protocol_valid(
                trajectory,
                arm="h1",
                review_records=records,
            )
        )
        rebuilt, independent_records = (
            VALIDATOR.independently_rebuild_final_region_protocol_valid(
                trajectory,
                arm="h1",
            )
        )
        self.assertFalse(rebuilt)
        self.assertEqual(len(independent_records), 1)


if __name__ == "__main__":
    unittest.main()
