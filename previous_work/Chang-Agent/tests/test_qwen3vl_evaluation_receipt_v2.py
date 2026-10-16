from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_tool(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = load_tool(
    "build_qwen3vl_evaluation_receipt",
    "tools/build_qwen3vl_evaluation_receipt.py",
)
VALIDATOR = load_tool(
    "validate_qwen3vl_evaluation_receipt",
    "tools/validate_qwen3vl_evaluation_receipt.py",
)


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binding(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def replace_key_value(path: Path, key: str, value: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    matches = [index for index, line in enumerate(lines) if line.startswith(f"{key}=")]
    if len(matches) != 1:
        raise AssertionError(f"expected exactly one {key!r} field in {path}")
    lines[matches[0]] = f"{key}={value}"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def refresh_receipt_bindings(
    receipt: Path,
    artifacts: dict[str, Path],
) -> None:
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    for name, path in artifacts.items():
        record = payload["artifacts"][name]
        record["sha256"] = sha256(path)
        record["size_bytes"] = path.stat().st_size
    write_json(receipt, payload)


class CompleteReceiptFixture:
    sample = "test_1_1"
    profile = "p1_smoke1"
    evaluation_id = "DEV-SMOKE-1-H1-V1"
    arm = "h0"
    experiment_arm = "v12_1-h0-legacy-single-round-base-0802"
    harness_protocol = "qwen3vl_harness_h0_legacy_single_round_v1"
    job_id = "123"
    worker_hash = hashlib.sha256(b"worker").hexdigest()
    cpu_worker_hash = hashlib.sha256(b"cpu-worker").hexdigest()
    dependency_guard_worker_hash = hashlib.sha256(
        b"dependency-guard-worker"
    ).hexdigest()

    def __init__(self, root: Path) -> None:
        self.root = root
        self.output = root / "output"
        self.run_dir = self.output / "run"
        self.input_root = root / "inputs"
        self.model = root / "model"
        self.worktree = root / "worktree"
        self.output.mkdir()
        self.run_dir.mkdir()
        (self.output / "logs").mkdir()
        self.model.mkdir()
        self.worktree.mkdir()
        self.worker_file = self.root / "evaluation-worker.slurm"
        self.worker_file.write_bytes(b"worker")
        self.freeze_commit = self._make_clean_worktree()
        self.commit = self.freeze_commit
        self._make_inputs()
        self._make_model()
        self._make_run_artifacts()
        self._make_static_manifests()
        self._make_summaries()
        self._make_runtime_manifest()
        self._make_text_receipts()

    def _run_git(self, *arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(self.worktree), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    def _make_clean_worktree(self) -> str:
        self._run_git("init", "-q")
        self._run_git("config", "user.email", "fixture@example.invalid")
        self._run_git("config", "user.name", "P1 receipt fixture")
        (self.worktree / "tracked.txt").write_text(
            "implementation\n", encoding="utf-8"
        )
        self._run_git("add", "tracked.txt")
        self._run_git("commit", "-q", "-m", "implementation fixture")
        self.implementation_commit = self._run_git("rev-parse", "HEAD")
        (self.worktree / "tracked.txt").write_text("frozen\n", encoding="utf-8")
        self._run_git("add", "tracked.txt")
        self._run_git("commit", "-q", "-m", "freeze fixture")
        return self._run_git("rev-parse", "HEAD")

    def _make_inputs(self) -> None:
        for directory, payload in (
            ("A", b"online-a"),
            ("B", b"online-b"),
            ("label_cvt", b"offline-label"),
        ):
            path = self.input_root / directory / f"{self.sample}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)

    def _make_model(self) -> None:
        (self.model / "model.safetensors").write_bytes(b"tiny-model")
        write_json(self.model / "config.json", {"model_type": "qwen3_vl"})

    def _make_run_artifacts(self) -> None:
        self.counters = {
            "verifier_round_count": 1,
            "stage_initial_attempt_count": 1,
            "schema_repair_attempt_count": 0,
            "network_retry_attempt_count": 0,
            "tool_action_attempt_count": 0,
            "tool_failure_count": 0,
            "environment_step_count": 0,
        }
        self.rollout = write_json(
            self.run_dir / "rollout_complete.json",
            {
                "execution_status": "completed",
                "gt_loaded": False,
                "samples": [self.sample],
            },
        )
        self.metrics = write_json(
            self.run_dir / "per_sample_metrics.json",
            {"samples": {self.sample: {}}},
        )
        trajectory_dir = self.run_dir / "trajectories" / self.sample
        staged_evidence = {
            "type": "staged_qwen_verifier",
            "schema_version": (
                "staged_verifier_environment_ordered_compact_audit_v12_1"
            ),
            "verifier_valid": True,
            "localization_valid": True,
            "validation_errors": [],
            "regional_validation_errors": [],
            "stage_trace": {
                "mode": "initial",
                "evidence": [{"region_id": "r0", "audit_kind": "present"}],
                "audited_region_ids": ["r0"],
                "judgments": [{"region_id": "r0"}],
                "diagnoses": [{"region_id": "r0", "error_type": "none"}],
                "plan": None,
            },
            "backend_calls": [
                {
                    "backend": "fixture",
                    "stage": "audit",
                    "attempt_kind": "initial",
                    "visual_input_scope": "standard",
                    "network_retry_attempt_count": 0,
                    "prompt_sha256": "c" * 64,
                    "raw_response": "{}",
                    "parsed_output": {"region_id": "r0"},
                }
            ],
        }
        self.trajectory = write_json(
            trajectory_dir / "trajectory.json",
            {
                "steps": [
                    {
                        "step_index": 0,
                        "execution": {"verifier_evidence": staged_evidence},
                    }
                ]
            },
        )
        self.per_sample_audit = {
            "terminal_review": None,
            "terminal_reviews": [],
            "budget_accounting": self.counters,
            "gt_boundary_audit": {
                "pre_rollout_gt_access_count": 0,
                "violation_ids": [],
                "rollout_completed_before_offline_scoring": True,
            },
            "safety_boundary_audit": {
                "relaxation_count": 0,
                "relaxed_boundaries": [],
            },
            "final_region_protocol_valid": True,
        }
        self.episode = write_json(
            trajectory_dir / "episode_summary.json",
            {
                "sample": self.sample,
                "execution_status": "completed",
                **self.per_sample_audit,
            },
        )
        self.audit = write_json(
            self.run_dir / "p1_harness_audit.json",
            {
                "schema_id": "qwen3vl_p1_harness_audit_v1",
                "factor_id": "terminal_review_scheduler_v1",
                "harness_protocol_id": self.harness_protocol,
                "experiment_arm_id": self.experiment_arm,
                "arm": self.arm,
                "requested_sample_ids": [self.sample],
                "per_sample": {self.sample: self.per_sample_audit},
                "totals": self.counters,
                "gt_boundary_violation_count": 0,
                "safety_boundary_relaxation_count": 0,
                "tool_failure_count": 0,
                "written_before_rollout_complete": True,
                "gt_loaded": False,
            },
        )
        self.run_manifest = self.run_dir / "run_manifest.md"
        self.run_manifest.write_text(
            "\n".join(
                (
                    "# fixture",
                    "",
                    "- terminal_review_factor_id: `terminal_review_scheduler_v1`",
                    f"- terminal_review_arm: `{self.arm}`",
                    f"- harness_protocol_id: `{self.harness_protocol}`",
                    f"- experiment_arm_id: `{self.experiment_arm}`",
                    f"- git_commit: `{self.commit}`",
                    "- git_dirty: `False`",
                    "- seed: `42`",
                    "",
                )
            ),
            encoding="utf-8",
        )

    def _make_static_manifests(self) -> None:
        amendment_relative = (
            "doc/training_pipeline/0802_third_try/amendment.json"
        )
        self.config = write_json(
            self.root
            / "doc/training_pipeline/0802_third_try/resolved_config.json",
            {
                "schema_id": "qwen3vl_p1_resolved_run_config_v1",
                "configuration_status": "frozen",
                "execution_readiness": "execution_ready",
                "p1_execution_authorized": True,
                "paths": {
                    "clean_run_worktree": str(self.worktree.resolve()),
                    "base_model": str(self.model.resolve()),
                    "p1_amendment": amendment_relative,
                },
                "frozen_identity": {
                    "factor_id": "terminal_review_scheduler_v1",
                    "model_state": "base",
                    "model_io_protocol": "v12.1",
                    "seed": 42,
                    self.arm: {
                        "terminal_review_arm": self.arm,
                        "experiment_arm_id": self.experiment_arm,
                        "harness_protocol_id": self.harness_protocol,
                        "max_verifier_rounds": 1,
                    },
                },
                "profiles": {
                    self.profile: {
                        "evaluation_id": self.evaluation_id,
                        "ordered_samples": [self.sample],
                    }
                },
                "slurm": {
                    "partition": "gre",
                    "fixed_node": "gpu48",
                    "maximum_ca0802_concurrent_gpus": 3,
                    "maximum_p1_concurrent_gpus_by_dependency_design": 1,
                },
                "allocations": [
                    {
                        "allocation_id": "CA_0802(3)",
                        "arm": self.arm,
                        "evaluation_profile": self.profile,
                        "resources": {"gpus": 1},
                    }
                ],
                "freeze_commit_hash_bindings": {
                    "status": "frozen",
                    "model_files_sha256": {
                        "model.safetensors": sha256(
                            self.model / "model.safetensors"
                        )
                    },
                    "tool_files_sha256": {
                        "harness_worker": self.worker_hash
                    },
                    "worker_sha256": {
                        "cpu_fixtures": self.cpu_worker_hash,
                        "evaluation": self.worker_hash,
                        "dependency_guard": self.dependency_guard_worker_hash,
                    },
                },
            },
        )
        self.amendment_payload = {
            "schema_id": "qwen3vl_p1_execution_amendment_v1",
            "record_status": "frozen",
            "execution_readiness": "execution_ready",
            "p1_execution_authorized": True,
            "implementation_commit": self.implementation_commit,
            "clean_run_worktree": str(self.worktree.resolve()),
            "resolved_run_config_sha256": sha256(self.config),
            "model_hash": {
                "model.safetensors": sha256(
                    self.model / "model.safetensors"
                )
            },
            "tool_hashes": {"harness_worker": self.worker_hash},
            "worker_sha256": {
                "cpu_fixtures": self.cpu_worker_hash,
                "evaluation": self.worker_hash,
                "dependency_guard": self.dependency_guard_worker_hash,
            },
        }
        self.amendment = write_json(
            self.root / amendment_relative,
            self.amendment_payload,
        )
        self.amendment_canonical_hash = BUILDER.canonical_sha256(
            self.amendment_payload
        )
        self.evaluation_payload = {
            "schema_id": "qwen3vl_evaluation_input_manifest_v1",
            "design_status": "frozen",
            "evaluation_profiles": [
                {
                    "phase": "smoke1",
                    "evaluation_id": self.evaluation_id,
                    "ordered_sample_ids": [self.sample],
                    "arm_order": ["h0", "h1"],
                }
            ],
            "source_contract": {"input_root": str(self.input_root.resolve())},
            "gt_boundary": {
                "a_and_b_online_inputs": True,
                "label_cvt_online_input": False,
                "label_cvt_offline_scorer_only": True,
            },
            "samples": {
                self.sample: {
                    "a_sha256": sha256(
                        self.input_root / "A" / f"{self.sample}.png"
                    ),
                    "b_sha256": sha256(
                        self.input_root / "B" / f"{self.sample}.png"
                    ),
                    "label_sha256": sha256(
                        self.input_root / "label_cvt" / f"{self.sample}.png"
                    ),
                    "label_access": "offline_scorer_only_after_rollout_complete",
                }
            },
        }
        self.evaluation = write_json(
            self.root / "evaluation_input_manifest.json", self.evaluation_payload
        )
        self.evaluation_canonical_hash = BUILDER.canonical_sha256(
            self.evaluation_payload
        )

    def _artifact_bindings(self) -> dict[str, Any]:
        return {
            "algorithm": "sha256",
            "rollout_marker": binding(self.rollout),
            "source_metrics": binding(self.metrics),
            "p1_harness_audit": binding(self.audit),
            "per_sample": {
                self.sample: {
                    "trajectory": binding(self.trajectory),
                    "episode_summary": binding(self.episode),
                    "tool_reports": [],
                }
            },
        }

    def _make_summaries(self) -> None:
        bindings = self._artifact_bindings()
        configured = {
            "max_stage_attempts_per_round": 2,
            "max_environment_steps": 3,
            "effective_network_retry_limit": 0,
            "effective_tool_attempts_per_authorized_action": 1,
            "max_verifier_rounds": 1,
        }
        summary = {
            "summary_schema_version": "qwen3vl_e2e_summary_v2",
            "status": "completed",
            "evaluation_profile": self.profile,
            "evaluation_id": self.evaluation_id,
            "model_state": "base",
            "job_id": self.job_id,
            "experiment_arm": self.experiment_arm,
            "protocol_version": "v12.1",
            "harness_protocol_id": self.harness_protocol,
            "runtime_commit": self.commit,
            "harness_commit": self.commit,
            "requested_samples": 1,
            "scored_samples": 1,
            "sample_order": [self.sample],
            "gt_boundary": {
                "rollout_marker_claims_gt_free": True,
                "offline_metrics_are_post_rollout": True,
            },
            "actions": {
                "candidates": 0,
                "accepted": 0,
                "rejected": 0,
                "action_attempts": 0,
                "action_attempt_errors": 0,
                "fallback_actions": 0,
                "tool_actions": 0,
                "successful_tool_actions": 0,
                "tool_failure_count": 0,
                "finish_count": 1,
            },
            "artifact_bindings": bindings,
            "p1_contract": {
                "schema_version": "qwen3vl_e2e_summary_p1_contract_v1",
                "source_audit_schema_id": "qwen3vl_p1_harness_audit_v1",
                "factor_id": "terminal_review_scheduler_v1",
                "arm": self.arm,
                "harness_protocol_id": self.harness_protocol,
                "experiment_arm": self.experiment_arm,
                "terminal_review": False,
                "model_io_protocol": "v12.1",
                "gt_loaded_before_rollout_complete": False,
                "frozen_safety_boundaries": [
                    "parser",
                    "geometry_authorization",
                    "locality_hard_gate",
                    "candidate_benefit_without_harm",
                    "rollback",
                ],
                "hard_gates": {
                    "requested_scored_mismatch_count": 0,
                    "pre_rollout_gt_violation_count": 0,
                    "safety_boundary_relaxation_count": 0,
                    "tool_failure_count": 0,
                    "all_passed": True,
                },
                "configured_budgets": configured,
                "budget_accounting_totals": self.counters,
                "per_sample": {self.sample: self.per_sample_audit},
                "final_region_protocol_valid_count": 1,
                "round_two_invoked_count": 0,
                "provenance": {
                    "model_state": "base",
                    "runtime_commit": self.commit,
                    "harness_commit": self.commit,
                    "job_id": self.job_id,
                    "artifact_bindings": bindings,
                },
            },
        }
        self.summary = write_json(self.output / "baseline_summary.json", summary)
        self.protocol_summary = write_json(
            self.output / "protocol_summary.json",
            {
                "status": "completed",
                "evaluation_profile": self.profile,
                "evaluation_id": self.evaluation_id,
                "experiment_arm": self.experiment_arm,
                "protocol_version": "v12.1",
                "harness_protocol_id": self.harness_protocol,
                "job_id": self.job_id,
                "runtime_commit": self.commit,
                "harness_commit": self.commit,
                "sample_order": [self.sample],
            },
        )

    def _make_runtime_manifest(self) -> None:
        model_hash = sha256(self.model / "model.safetensors")
        tool_hashes = {"harness_worker": self.worker_hash}
        sample_files = self.evaluation_payload["samples"][self.sample]
        self.runtime = write_json(
            self.output / "input_manifest.json",
            {
                "schema_id": "qwen3vl_runtime_input_manifest_v1",
                "phase": "post_rollout_complete",
                "evaluation_profile": self.profile,
                "evaluation_id": self.evaluation_id,
                "sample_order": [self.sample],
                "terminal_review_arm": self.arm,
                "experiment_arm": self.experiment_arm,
                "harness_protocol_id": self.harness_protocol,
                "protocol_version": "v12.1",
                "input_root": str(self.input_root.resolve()),
                "files_sha256": {
                    self.sample: {
                        "A": sample_files["a_sha256"],
                        "B": sample_files["b_sha256"],
                        "label_cvt": sample_files["label_sha256"],
                    }
                },
                "gt_hashed_after_rollout": True,
                "model_path": str(self.model.resolve()),
                "model_files_sha256": {"model.safetensors": model_hash},
                "tool_files_sha256": tool_hashes,
                "tool_file_paths": {
                    "harness_worker": str(self.worker_file.resolve())
                },
                "runtime_source_sha256": {
                    "runner": hashlib.sha256(b"runtime-source").hexdigest()
                },
                "runtime_commits": {
                    "chang_agent": self.commit,
                    "expected_chang_agent": self.commit,
                    "harness": self.commit,
                    "expected_harness": self.commit,
                },
                "executed_worker_sha256": self.worker_hash,
                "evaluation_input_manifest": {
                    "path": str(self.evaluation.resolve()),
                    "canonical_sha256": self.evaluation_canonical_hash,
                },
                "resolved_run_config": {
                    "path": str(self.config.resolve()),
                    "sha256": sha256(self.config),
                },
                "slurm_job_id": self.job_id,
                "runtime_environment": {"device": "NVIDIA A800-SXM4-80GB"},
            },
        )

    def _make_text_receipts(self) -> None:
        self.launch = self.output / "launch_command.txt"
        self.launch.write_text(
            f"launch=python --terminal-review-arm {self.arm} --seed 42\n",
            encoding="utf-8",
        )
        self.submission = self.output / "submission.txt"
        submission_values = (
            ("submission_status", "submitted"),
            ("job_id", self.job_id),
            ("job_name", "CA0802P1SmokeH0"),
            ("allocation_id", "CA_0802(3)"),
            ("evaluation_profile", self.profile),
            ("evaluation_id", self.evaluation_id),
            ("samples", self.sample),
            ("arm", self.arm),
            ("experiment_arm_id", self.experiment_arm),
            ("harness_protocol_id", self.harness_protocol),
            ("expected_commit", self.commit),
            ("implementation_commit", self.implementation_commit),
            ("worker_sha256", self.worker_hash),
            ("config_sha256", sha256(self.config)),
            ("amendment_canonical_sha256", self.amendment_canonical_hash),
            ("partition", "gre"),
            ("node", "gpu48"),
            ("resources", "gpu:1,cpu:4,mem:48G,time:01:00:00"),
            ("dependency", "afterok:122"),
            ("output", str(self.output.resolve())),
        )
        self.submission.write_text(
            "".join(f"{key}={value}\n" for key, value in submission_values),
            encoding="utf-8",
        )
        self.status = self.output / "logs" / f"status-{self.job_id}.txt"
        status_values = (
            ("status", "success"),
            ("exit_code", "0"),
            ("elapsed_seconds", "1"),
            ("job_id", self.job_id),
            ("job_name", "CA0802P1SmokeH0"),
            ("allocation_id", "CA_0802(3)"),
            ("evaluation_profile", self.profile),
            ("evaluation_id", self.evaluation_id),
            ("samples", self.sample),
            ("arm", self.arm),
            ("experiment_arm_id", self.experiment_arm),
            ("harness_protocol_id", self.harness_protocol),
            ("commit", self.commit),
            ("worker_sha256", self.worker_hash),
            ("result", "all_rollout_and_summary_gates_passed"),
            ("error", ""),
            ("path", str(self.output.resolve())),
        )
        self.status.write_text(
            "".join(f"{key}={value}\n" for key, value in status_values),
            encoding="utf-8",
        )
        self.summary_validation = self.output / "summary_validation.txt"
        self.summary_validation.write_text(
            "status=valid;profile=p1_smoke1;"
            f"arm={self.experiment_arm};scored=1/1\n",
            encoding="utf-8",
        )

    def build(self, *, implementation_commit: str | None = None) -> Path:
        implementation_commit = (
            self.implementation_commit
            if implementation_commit is None
            else implementation_commit
        )
        receipt = self.output / "evaluation_receipt.json"
        args = BUILDER.parse_args(
            [
                "--output",
                str(receipt),
                "--output-parent",
                str(self.output),
                "--run-dir",
                str(self.run_dir),
                "--evaluation-profile",
                self.profile,
                "--evaluation-id",
                self.evaluation_id,
                "--expected-samples",
                self.sample,
                "--arm",
                self.arm,
                "--experiment-arm",
                self.experiment_arm,
                "--harness-protocol-id",
                self.harness_protocol,
                "--runtime-commit",
                self.commit,
                "--harness-commit",
                self.commit,
                "--implementation-commit",
                implementation_commit,
                "--job-id",
                self.job_id,
                "--run-worktree",
                str(self.worktree),
                "--model-path",
                str(self.model),
                "--worker-sha256",
                self.worker_hash,
                "--evaluation-input-manifest",
                str(self.evaluation),
                "--evaluation-input-manifest-canonical-sha256",
                self.evaluation_canonical_hash,
                "--runtime-input-manifest",
                str(self.runtime),
                "--resolved-run-config",
                str(self.config),
                "--resolved-run-config-sha256",
                sha256(self.config),
                "--run-manifest",
                str(self.run_manifest),
                "--launch-command",
                str(self.launch),
                "--submission-receipt",
                str(self.submission),
                "--status-receipt",
                str(self.status),
                "--summary-validation",
                str(self.summary_validation),
            ]
        )
        BUILDER.atomic_publish(receipt, BUILDER.build_payload(args))
        return receipt


class Qwen3VLEvaluationReceiptV2Tests(unittest.TestCase):
    def test_complete_receipt_passes_independent_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CompleteReceiptFixture(Path(temporary))
            receipt = fixture.build()
            payload = VALIDATOR.validate_receipt(receipt)
            self.assertEqual(payload["schema_id"], "qwen3vl_evaluation_receipt_v2")
            self.assertEqual(payload["status"], "complete")
            self.assertEqual(payload["identity"]["sample_order"], [fixture.sample])
            self.assertNotEqual(
                fixture.implementation_commit,
                fixture.freeze_commit,
            )
            self.assertEqual(
                fixture._run_git("rev-parse", f"{fixture.freeze_commit}^"),
                fixture.implementation_commit,
            )
            self.assertEqual(
                payload["identity"]["runtime_commit"],
                fixture.freeze_commit,
            )
            self.assertEqual(
                payload["provenance"]["implementation_commit"],
                fixture.implementation_commit,
            )
            self.assertEqual(
                payload["provenance"]["freeze_commit"],
                fixture.freeze_commit,
            )
            self.assertEqual(
                payload["artifacts"]["p1_amendment"]["sha256"],
                sha256(fixture.amendment),
            )

    def test_builder_and_validator_reject_external_model_or_tool_drift(
        self,
    ) -> None:
        for target_name in ("model", "tool"):
            for validation_stage in ("builder", "validator"):
                with self.subTest(
                    target=target_name,
                    stage=validation_stage,
                ), tempfile.TemporaryDirectory() as temporary:
                    fixture = CompleteReceiptFixture(Path(temporary))
                    receipt = (
                        fixture.build()
                        if validation_stage == "validator"
                        else None
                    )
                    target = (
                        fixture.model / "model.safetensors"
                        if target_name == "model"
                        else fixture.worker_file
                    )
                    target.write_bytes(b"externally-tampered")
                    if validation_stage == "builder":
                        with self.assertRaises(BUILDER.ReceiptBuildError):
                            fixture.build()
                    else:
                        assert receipt is not None
                        with self.assertRaises(
                            VALIDATOR.ReceiptValidationError
                        ):
                            VALIDATOR.validate_receipt(receipt)

    def test_validator_rejects_rebound_config_authorization_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CompleteReceiptFixture(Path(temporary))
            receipt = fixture.build()
            config = json.loads(fixture.config.read_text(encoding="utf-8"))
            config["configuration_status"] = "draft"
            write_json(fixture.config, config)
            amendment = json.loads(
                fixture.amendment.read_text(encoding="utf-8")
            )
            amendment["resolved_run_config_sha256"] = sha256(fixture.config)
            write_json(fixture.amendment, amendment)
            replace_key_value(
                fixture.submission,
                "config_sha256",
                sha256(fixture.config),
            )
            replace_key_value(
                fixture.submission,
                "amendment_canonical_sha256",
                BUILDER.canonical_sha256(amendment),
            )
            runtime = json.loads(fixture.runtime.read_text(encoding="utf-8"))
            runtime["resolved_run_config"]["sha256"] = sha256(fixture.config)
            write_json(fixture.runtime, runtime)
            refresh_receipt_bindings(
                receipt,
                {
                    "resolved_run_config": fixture.config,
                    "p1_amendment": fixture.amendment,
                    "runtime_input_manifest": fixture.runtime,
                    "submission_receipt": fixture.submission,
                },
            )
            with self.assertRaises(VALIDATOR.ReceiptValidationError):
                VALIDATOR.validate_receipt(receipt)

    def test_status_or_runtime_manifest_mutation_invalidates_receipt(self) -> None:
        for target_name in ("status", "runtime"):
            with self.subTest(target=target_name), tempfile.TemporaryDirectory() as temporary:
                fixture = CompleteReceiptFixture(Path(temporary))
                receipt = fixture.build()
                target = fixture.status if target_name == "status" else fixture.runtime
                with target.open("a", encoding="utf-8") as handle:
                    handle.write("\n")
                with self.assertRaises(VALIDATOR.ReceiptValidationError):
                    VALIDATOR.validate_receipt(receipt)

    def test_trajectory_or_episode_mutation_invalidates_receipt(self) -> None:
        for target_name in ("trajectory", "episode"):
            with self.subTest(target=target_name), tempfile.TemporaryDirectory() as temporary:
                fixture = CompleteReceiptFixture(Path(temporary))
                receipt = fixture.build()
                target = (
                    fixture.trajectory
                    if target_name == "trajectory"
                    else fixture.episode
                )
                payload = json.loads(target.read_text(encoding="utf-8"))
                if target_name == "trajectory":
                    payload["steps"][0]["execution"]["verifier_evidence"].pop(
                        "stage_trace"
                    )
                else:
                    payload["final_region_protocol_valid"] = False
                write_json(target, payload)
                with self.assertRaises(VALIDATOR.ReceiptValidationError):
                    VALIDATOR.validate_receipt(receipt)

    def test_builder_rejects_wrong_or_non_parent_implementation_commit(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CompleteReceiptFixture(Path(temporary))
            with self.assertRaises(BUILDER.ReceiptBuildError):
                fixture.build(implementation_commit=fixture.freeze_commit)

            non_parent = "f" * 40
            amendment = dict(fixture.amendment_payload)
            amendment["implementation_commit"] = non_parent
            write_json(fixture.amendment, amendment)
            amendment_hash = BUILDER.canonical_sha256(amendment)
            replace_key_value(
                fixture.submission,
                "implementation_commit",
                non_parent,
            )
            replace_key_value(
                fixture.submission,
                "amendment_canonical_sha256",
                amendment_hash,
            )
            with self.assertRaisesRegex(
                BUILDER.ReceiptBuildError,
                "direct child",
            ):
                fixture.build(implementation_commit=non_parent)

    def test_builder_rejects_amendment_or_submission_a_b_binding_drift(
        self,
    ) -> None:
        for mutation in (
            "amendment_implementation",
            "submission_freeze",
            "submission_implementation",
            "submission_amendment_digest",
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                fixture = CompleteReceiptFixture(Path(temporary))
                if mutation == "amendment_implementation":
                    amendment = dict(fixture.amendment_payload)
                    amendment["implementation_commit"] = fixture.freeze_commit
                    write_json(fixture.amendment, amendment)
                elif mutation == "submission_freeze":
                    replace_key_value(
                        fixture.submission,
                        "expected_commit",
                        fixture.implementation_commit,
                    )
                elif mutation == "submission_implementation":
                    replace_key_value(
                        fixture.submission,
                        "implementation_commit",
                        fixture.freeze_commit,
                    )
                else:
                    replace_key_value(
                        fixture.submission,
                        "amendment_canonical_sha256",
                        "0" * 64,
                    )
                with self.assertRaises(BUILDER.ReceiptBuildError):
                    fixture.build()

    def test_builder_rejects_config_or_amendment_hash_binding_tamper(
        self,
    ) -> None:
        for mutation in (
            "config_authorization",
            "amendment_model_hash",
            "amendment_tool_hash",
            "amendment_worker_hash",
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                fixture = CompleteReceiptFixture(Path(temporary))
                if mutation == "config_authorization":
                    config = json.loads(
                        fixture.config.read_text(encoding="utf-8")
                    )
                    config["configuration_status"] = "draft"
                    write_json(fixture.config, config)
                    amendment = json.loads(
                        fixture.amendment.read_text(encoding="utf-8")
                    )
                    amendment["resolved_run_config_sha256"] = sha256(
                        fixture.config
                    )
                else:
                    amendment = json.loads(
                        fixture.amendment.read_text(encoding="utf-8")
                    )
                    if mutation == "amendment_model_hash":
                        amendment["model_hash"] = {
                            "model.safetensors": "0" * 64
                        }
                    elif mutation == "amendment_tool_hash":
                        amendment["tool_hashes"] = {
                            "harness_worker": "0" * 64
                        }
                    else:
                        amendment["worker_sha256"] = {
                            **amendment["worker_sha256"],
                            "evaluation": "0" * 64,
                        }
                write_json(fixture.amendment, amendment)
                replace_key_value(
                    fixture.submission,
                    "config_sha256",
                    sha256(fixture.config),
                )
                replace_key_value(
                    fixture.submission,
                    "amendment_canonical_sha256",
                    BUILDER.canonical_sha256(amendment),
                )
                with self.assertRaises(BUILDER.ReceiptBuildError):
                    fixture.build()

    def test_validator_rejects_receipt_a_b_provenance_tampering(self) -> None:
        for key, replacement in (
            ("implementation_commit", "freeze"),
            ("freeze_commit", "implementation"),
        ):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temporary:
                fixture = CompleteReceiptFixture(Path(temporary))
                receipt = fixture.build()
                payload = json.loads(receipt.read_text(encoding="utf-8"))
                payload["provenance"][key] = (
                    fixture.freeze_commit
                    if replacement == "freeze"
                    else fixture.implementation_commit
                )
                write_json(receipt, payload)
                with self.assertRaises(VALIDATOR.ReceiptValidationError):
                    VALIDATOR.validate_receipt(receipt)

    def test_validator_rejects_rebound_amendment_or_submission_a_b_tampering(
        self,
    ) -> None:
        for mutation in (
            "amendment_schema",
            "amendment_implementation",
            "amendment_model_hash",
            "amendment_tool_hash",
            "amendment_worker_hash",
            "submission_freeze",
            "submission_implementation",
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                fixture = CompleteReceiptFixture(Path(temporary))
                receipt = fixture.build()
                rebound: dict[str, Path] = {}
                if mutation.startswith("amendment_"):
                    amendment = json.loads(
                        fixture.amendment.read_text(encoding="utf-8")
                    )
                    if mutation == "amendment_schema":
                        amendment["schema_id"] = "tampered_amendment_schema"
                    elif mutation == "amendment_model_hash":
                        amendment["model_hash"] = {
                            "model.safetensors": "0" * 64
                        }
                    elif mutation == "amendment_tool_hash":
                        amendment["tool_hashes"] = {
                            "harness_worker": "0" * 64
                        }
                    elif mutation == "amendment_worker_hash":
                        amendment["worker_sha256"] = {
                            **amendment["worker_sha256"],
                            "evaluation": "0" * 64,
                        }
                    else:
                        amendment["implementation_commit"] = fixture.freeze_commit
                    write_json(fixture.amendment, amendment)
                    replace_key_value(
                        fixture.submission,
                        "amendment_canonical_sha256",
                        BUILDER.canonical_sha256(amendment),
                    )
                    rebound["p1_amendment"] = fixture.amendment
                    rebound["submission_receipt"] = fixture.submission
                elif mutation == "submission_freeze":
                    replace_key_value(
                        fixture.submission,
                        "expected_commit",
                        fixture.implementation_commit,
                    )
                    rebound["submission_receipt"] = fixture.submission
                else:
                    replace_key_value(
                        fixture.submission,
                        "implementation_commit",
                        fixture.freeze_commit,
                    )
                    rebound["submission_receipt"] = fixture.submission
                refresh_receipt_bindings(receipt, rebound)
                with self.assertRaises(VALIDATOR.ReceiptValidationError):
                    VALIDATOR.validate_receipt(receipt)


if __name__ == "__main__":
    unittest.main()
