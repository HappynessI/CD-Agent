from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_tool(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PAIR = load_tool(
    "build_qwen3vl_p1_pair_comparison",
    "tools/build_qwen3vl_p1_pair_comparison.py",
)
RUNTIME = load_tool(
    "build_qwen3vl_runtime_input_manifest",
    "tools/build_qwen3vl_runtime_input_manifest.py",
)
RUNNER = load_tool(
    "run_levir_change_agent_p1_boundary",
    "tools/run_levir_change_agent.py",
)
SUMMARY_VALIDATOR = load_tool(
    "validate_qwen3vl_e2e_summary_pair_fixture",
    "tools/validate_qwen3vl_e2e_summary.py",
)


def write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


class P1ResolvedExecutionContractTests(unittest.TestCase):
    _FORMAL_JOB_IDS = ("101", "102", "103", "104", "105")
    _ALLOCATION_IDS = (
        "CA_0802(2)",
        "CA_0802(3)",
        "CA_0802(4)",
        "CA_0802(5)",
        "CA_0802(6)",
    )
    _JOB_NAMES = (
        "CA0802P1CPUFixtures",
        "CA0802P1SmokeH0",
        "CA0802P1SmokeH1",
        "CA0802P1CoreH0",
        "CA0802P1CoreH1",
    )
    _OUTPUT_NAMES = (
        "CA_0802(2)-qwen3vl-h0-h1-cpu-fixtures-retry3",
        "CA_0802(3)-qwen3vl-h0-base-smoke1-retry3",
        "CA_0802(4)-qwen3vl-h1-base-smoke1-retry3",
        "CA_0802(5)-qwen3vl-h0-base-core3-retry3",
        "CA_0802(6)-qwen3vl-h1-base-core3-retry3",
    )

    def _run_dependency_guard_fixture(
        self,
        fixture_root: Path,
        *,
        accounting: tuple[tuple[str, str, str, str, str], ...],
        existing_status_job_ids: set[str],
        notifier_exit_code: int = 0,
        expected_implementation_commit: str | None = None,
        control_records: dict[str, str] | None = None,
        submission_dependency_overrides: dict[str, str] | None = None,
    ) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
        self.assertEqual(len(accounting), 5)
        workspace_root = fixture_root / "workspace"
        workspace_root.mkdir()
        artifact_root = fixture_root / "artifact"
        output_root = artifact_root / "outputs"
        output_root.mkdir(parents=True)
        run_root = fixture_root / "run-worktree"
        tools_dir = run_root / "tools"
        config_path = (
            run_root
            / "doc/training_pipeline/0802_third_try/"
            "QWEN3VL_P1_RESOLVED_RUN_CONFIG_0802.json"
        )
        tools_dir.mkdir(parents=True)
        config_path.parent.mkdir(parents=True)
        for worker_name in (
            "run_qwen3vl_p1_cpu_fixtures.slurm",
            "run_qwen3vl_p1_evaluation.slurm",
            "run_qwen3vl_p1_dependency_guard.slurm",
        ):
            (tools_dir / worker_name).write_bytes(
                (ROOT / "tools" / worker_name).read_bytes()
            )
        config = json.loads(
            (
                ROOT
                / "doc/training_pipeline/0802_third_try/"
                "QWEN3VL_P1_RESOLVED_RUN_CONFIG_0802.json"
            ).read_text(encoding="utf-8")
        )
        config["configuration_status"] = "frozen"
        config["execution_readiness"] = "execution_ready"
        config["p1_execution_authorized"] = True
        config["paths"].update(
            {
                "workspace_root": str(workspace_root.resolve()),
                "clean_run_worktree": str(run_root.resolve()),
                "artifact_repository": str(artifact_root.resolve()),
                "output_root": str(output_root.resolve()),
                "omniovcd_python": str(Path(sys.executable).resolve()),
            }
        )
        for allocation, output_name in zip(
            config["allocations"], self._OUTPUT_NAMES, strict=True
        ):
            allocation["output_path"] = str((output_root / output_name).resolve())
        write_json(config_path, config)
        for command in (
            ["git", "init", str(run_root)],
            ["git", "-C", str(run_root), "config", "user.name", "P1 Fixture"],
            ["git", "-C", str(run_root), "config", "user.email", "fixture@example.invalid"],
            ["git", "-C", str(run_root), "add", "."],
            ["git", "-C", str(run_root), "commit", "-m", "fixture implementation"],
            ["git", "-C", str(run_root), "commit", "--allow-empty", "-m", "fixture freeze"],
        ):
            subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        commit = subprocess.check_output(
            ["git", "-C", str(run_root), "rev-parse", "HEAD"], text=True
        ).strip()
        direct_parent = subprocess.check_output(
            ["git", "-C", str(run_root), "rev-parse", "HEAD^"], text=True
        ).strip()
        self.assertNotEqual(commit, direct_parent)
        implementation_commit = (
            direct_parent
            if expected_implementation_commit is None
            else expected_implementation_commit
        )
        cpu_worker_hash = sha256(
            tools_dir / "run_qwen3vl_p1_cpu_fixtures.slurm"
        )
        gpu_worker_hash = sha256(
            tools_dir / "run_qwen3vl_p1_evaluation.slurm"
        )
        guard_worker = tools_dir / "run_qwen3vl_p1_dependency_guard.slurm"
        guard_worker_hash = sha256(guard_worker)
        for index, output_name in enumerate(self._OUTPUT_NAMES):
            output = output_root / output_name
            (output / "logs").mkdir(parents=True)
            worker_hash = cpu_worker_hash if index == 0 else gpu_worker_hash
            expected_dependency = (
                "none"
                if index == 0
                else f"afterok:{self._FORMAL_JOB_IDS[index - 1]}"
            )
            dependency = (submission_dependency_overrides or {}).get(
                self._FORMAL_JOB_IDS[index], expected_dependency
            )
            (output / "submission.txt").write_text(
                "submission_status=submitted\n"
                f"job_id={self._FORMAL_JOB_IDS[index]}\n"
                f"job_name={self._JOB_NAMES[index]}\n"
                f"allocation_id={self._ALLOCATION_IDS[index]}\n"
                f"expected_commit={commit}\n"
                f"implementation_commit={implementation_commit}\n"
                f"worker_sha256={worker_hash}\n"
                + f"dependency={dependency}\n",
                encoding="utf-8",
            )
            if self._FORMAL_JOB_IDS[index] in existing_status_job_ids:
                (output / "logs" / f"status-{self._FORMAL_JOB_IDS[index]}.txt").write_text(
                    f"worker-owned-status={self._FORMAL_JOB_IDS[index]}\n",
                    encoding="utf-8",
                )

        cpu_output = output_root / self._OUTPUT_NAMES[0]
        (cpu_output / "dependency_guard_submission.txt").write_text(
            "submission_status=submitted\n"
            "job_id=999\n"
            "job_name=CA0802P1DependencyGuard\n"
            f"expected_commit={commit}\n"
            f"implementation_commit={implementation_commit}\n"
            f"worker_sha256={guard_worker_hash}\n",
            encoding="utf-8",
        )
        accounting_data = fixture_root / "fake-sacct.txt"
        accounting_data.write_text(
            "".join(
                f"{job_id}|{state}|{exit_code}|{start}|{end}|{reason}|\n"
                for job_id, (state, exit_code, start, end, reason) in zip(
                    self._FORMAL_JOB_IDS, accounting
                )
            ),
            encoding="utf-8",
        )
        fake_sacct = fixture_root / "fake-sacct"
        fake_sacct.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "job_id=''\n"
            "while [[ $# -gt 0 ]]; do\n"
            "  case \"$1\" in\n"
            "    -j) job_id=\"$2\"; shift 2 ;;\n"
            "    *) shift ;;\n"
            "  esac\n"
            "done\n"
            "while IFS= read -r row; do\n"
            "  case \"$row\" in\n"
            "    \"${job_id}|\"*) printf '%s\\n' \"$row\" ;;\n"
            "  esac\n"
            "done <\"${FAKE_SACCT_DATA:?}\"\n",
            encoding="utf-8",
        )
        fake_sacct.chmod(0o750)
        control_data = fixture_root / "fake-scontrol.txt"
        control_data.write_text(
            "".join(
                f"{job_id}|{record}\n"
                for job_id, record in (control_records or {}).items()
            ),
            encoding="utf-8",
        )
        fake_scontrol = fixture_root / "scontrol"
        fake_scontrol.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "[[ \"${1:-}\" == show && \"${2:-}\" == job ]] || exit 2\n"
            "job_id=\"${3:-}\"\n"
            "while IFS='|' read -r candidate record; do\n"
            "  if [[ \"$candidate\" == \"$job_id\" ]]; then\n"
            "    printf '%s\\n' \"$record\"\n"
            "    exit 0\n"
            "  fi\n"
            "done <\"${FAKE_SCONTROL_DATA:?}\"\n"
            "exit 1\n",
            encoding="utf-8",
        )
        fake_scontrol.chmod(0o750)
        notifier_log = fixture_root / "fake-notifier.log"
        fake_notifier = fixture_root / "fake-notifier"
        fake_notifier.write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "printf '%s\\t%s\\n' \"${QQ_SUB:?}\" \"${QQ_BODY:?}\" "
            ">>\"${FAKE_NOTIFIER_LOG:?}\"\n"
            "exit \"${FAKE_NOTIFIER_EXIT_CODE:-0}\"\n",
            encoding="utf-8",
        )
        fake_notifier.chmod(0o750)
        env = {
            **os.environ,
            "REPO_ROOT": str(run_root),
            "WORKSPACE_ROOT": str(workspace_root),
            "ARTIFACT_ROOT": str(artifact_root),
            "PYTHON": str(Path(sys.executable).resolve()),
            "EXPECTED_COMMIT": commit,
            "EXPECTED_IMPLEMENTATION_COMMIT": implementation_commit,
            "EXPECTED_WORKER_SHA256": guard_worker_hash,
            "EXPECTED_CPU_WORKER_SHA256": cpu_worker_hash,
            "EXPECTED_GPU_WORKER_SHA256": gpu_worker_hash,
            "EXPECTED_CONFIG_SHA256": sha256(config_path),
            "SACCT_BIN": str(fake_sacct),
            "SCONTROL_BIN": str(fake_scontrol),
            "QQ_NOTIFIER": str(fake_notifier),
            "FAKE_SACCT_DATA": str(accounting_data),
            "FAKE_SCONTROL_DATA": str(control_data),
            "FAKE_NOTIFIER_LOG": str(notifier_log),
            "FAKE_NOTIFIER_EXIT_CODE": str(notifier_exit_code),
            "SLURM_JOB_ID": "999",
            "SLURM_JOB_NAME": "CA0802P1DependencyGuard",
            "SLURM_JOB_PARTITION": "gre",
            "SLURM_CPUS_PER_TASK": "1",
            "SLURMD_NODENAME": "cpu-fixture-node",
            "CUDA_VISIBLE_DEVICES": "",
        }
        completed = subprocess.run(
            ["bash", str(guard_worker), *self._FORMAL_JOB_IDS],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        return completed, output_root, notifier_log

    def test_runner_pre_rollout_validation_does_not_stat_labels(self) -> None:
        root = Path("/synthetic/p1-boundary")
        args = SimpleNamespace(
            proposal_mode="proposal",
            verifier="qwen_staged",
            empty_mask_scan_grid_size=3,
            bailian_network_retries=0,
            bailian_network_backoff_seconds=0.0,
            max_point_proposal_change_ratio=0.25,
            max_point_proposal_change_pixels=256,
            max_point_image_change_ratio=0.004,
            staged_verifier_backend="local",
            simpleclick_checkpoint=str(root / "simpleclick.pth"),
            sam3_checkpoint=str(root / "sam3.pt"),
            sam3_bpe=str(root / "sam3.bpe"),
            model_path=str(root / "model"),
            input_root=root / "inputs",
            samples=["sample"],
        )
        observed: list[Path] = []

        def record_exists(path: Path) -> bool:
            observed.append(path)
            return True

        with mock.patch.object(Path, "exists", new=record_exists):
            RUNNER._validate_inputs(args)

        self.assertTrue(observed)
        self.assertTrue(
            all("label_cvt" not in path.parts for path in observed),
            observed,
        )

    def test_offline_label_validation_requires_marker_before_label_access(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "inputs"
            label = input_root / "label_cvt" / "sample.png"
            label.parent.mkdir(parents=True)
            label.write_bytes(b"fixture-content-is-opened-through-a-mock")
            marker = root / "run" / "rollout_complete.json"

            with mock.patch.object(RUNNER.Image, "open") as image_open:
                with self.assertRaises(RuntimeError):
                    RUNNER._validate_offline_labels_after_rollout(
                        input_root=input_root,
                        samples=("sample",),
                        rollout_marker=marker,
                    )
                image_open.assert_not_called()

            write_json(
                marker,
                {
                    "completed_at": "2026-08-02T00:00:00+00:00",
                    "samples": ["sample"],
                    "execution_status": "completed",
                    "gt_loaded": False,
                },
            )
            verified_image = mock.MagicMock()
            verified_image.__enter__.return_value = verified_image
            with mock.patch.object(
                RUNNER.Image, "open", return_value=verified_image
            ) as image_open:
                RUNNER._validate_offline_labels_after_rollout(
                    input_root=input_root,
                    samples=("sample",),
                    rollout_marker=marker,
                )
            image_open.assert_called_once_with(label)
            verified_image.verify.assert_called_once_with()

    def test_dependency_chain_and_resource_cap_are_frozen(self) -> None:
        config = json.loads(
            (
                ROOT
                / "doc/training_pipeline/0802_third_try/"
                "QWEN3VL_P1_RESOLVED_RUN_CONFIG_0802.json"
            ).read_text(encoding="utf-8")
        )
        allocations = config["allocations"]
        self.assertEqual(
            [item["allocation_id"] for item in allocations],
            [
                "CA_0802(2)",
                "CA_0802(3)",
                "CA_0802(4)",
                "CA_0802(5)",
                "CA_0802(6)",
            ],
        )
        self.assertEqual(allocations[0]["resources"]["gpus"], 0)
        self.assertTrue(
            all(item["resources"]["gpus"] == 1 for item in allocations[1:])
        )
        self.assertEqual(
            [item["dependency"] for item in allocations],
            [
                None,
                "afterok:CA_0802(2)",
                "afterok:CA_0802(3)",
                "afterok:CA_0802(4)",
                "afterok:CA_0802(5)",
            ],
        )
        self.assertEqual(config["slurm"]["fixed_node"], "gpu48")
        self.assertEqual(config["slurm"]["partition"], "gre")
        self.assertEqual(
            config["slurm"]["maximum_p1_concurrent_gpus_by_dependency_design"],
            1,
        )
        self.assertLessEqual(
            config["slurm"]["maximum_ca0802_concurrent_gpus"], 3
        )
        submitter = (ROOT / "tools/submit_qwen3vl_p1.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("--hold", submitter)
        self.assertIn("scontrol release", submitter)
        self.assertIn('SLURM_USER="guisongxia"', submitter)
        self.assertNotIn("rm -rf", submitter)

    def test_submitter_and_workers_enforce_a_b_and_readiness_contract(
        self,
    ) -> None:
        config = json.loads(
            (
                ROOT
                / "doc/training_pipeline/0802_third_try/"
                "QWEN3VL_P1_RESOLVED_RUN_CONFIG_0802.json"
            ).read_text(encoding="utf-8")
        )
        commit_contract = config["implementation_and_freeze_commit_contract"]
        self.assertEqual(commit_contract["implementation_commit_symbol"], "A")
        self.assertEqual(commit_contract["freeze_commit_symbol"], "B")
        self.assertEqual(commit_contract["required_parent_relation"], "B^ == A")
        self.assertTrue(commit_contract["freeze_files_must_not_self_reference_their_own_commit_hash"])

        submitter = (ROOT / "tools/submit_qwen3vl_p1.sh").read_text(
            encoding="utf-8"
        )
        for required in (
            'EXPECTED_COMMIT="$(git -C "$ROOT" rev-parse HEAD)"',
            'EXPECTED_IMPLEMENTATION_COMMIT="$(git -C "$ROOT" rev-parse "${EXPECTED_COMMIT}^")"',
            'config.get("configuration_status") != "frozen"',
            'config.get("execution_readiness") != "execution_ready"',
            'config.get("p1_execution_authorized") is not True',
            'fixture.get("execution_readiness") != "execution_ready"',
            'fixture.get("p1_execution_authorized") is not True',
            "implementation_commit=%s",
            "amendment_canonical_sha256=%s",
        ):
            self.assertIn(required, submitter)

        worker_sources = {
            name: (ROOT / relative).read_text(encoding="utf-8")
            for name, relative in (
                ("cpu", "tools/run_qwen3vl_p1_cpu_fixtures.slurm"),
                ("evaluation", "tools/run_qwen3vl_p1_evaluation.slurm"),
                ("guard", "tools/run_qwen3vl_p1_dependency_guard.slurm"),
            )
        }
        for name, source in worker_sources.items():
            with self.subTest(worker=name):
                self.assertIn("EXPECTED_IMPLEMENTATION_COMMIT", source)
                self.assertIn('rev-parse "${EXPECTED_COMMIT}^"', source)
        self.assertIn(
            '--implementation-commit "$EXPECTED_IMPLEMENTATION_COMMIT"',
            worker_sources["evaluation"],
        )
        self.assertIn(
            '"implementation_commit": implementation_commit',
            worker_sources["cpu"],
        )
        self.assertIn(
            '"freeze_commit": freeze_commit',
            worker_sources["cpu"],
        )

    def test_worker_and_guard_submit_blocks_are_statically_well_formed(
        self,
    ) -> None:
        worker = (ROOT / "tools/run_qwen3vl_p1_evaluation.slurm").read_text(
            encoding="utf-8"
        )
        marker = 'RUNTIME_DEVICE="$("$PYTHON" - "$MODEL_PATH" <<\'PY\'\n'
        self.assertIn(marker, worker)
        python_body = worker.split(marker, 1)[1].split("\nPY\n", 1)[0]
        ast.parse(python_body)
        self.assertEqual(python_body.count('if "A800" not in name:'), 1)

        submitter = (ROOT / "tools/submit_qwen3vl_p1.sh").read_text(
            encoding="utf-8"
        )
        guard_block = submitter.split('guard_output="$(sbatch', 1)[1].split(
            'guard_submit_rc="$?"', 1
        )[0]
        self.assertEqual(guard_block.count("--output="), 1)
        self.assertEqual(guard_block.count("--error="), 1)

    def test_dependency_guard_is_auxiliary_afterany_and_zero_gpu(self) -> None:
        config = json.loads(
            (
                ROOT
                / "doc/training_pipeline/0802_third_try/"
                "QWEN3VL_P1_RESOLVED_RUN_CONFIG_0802.json"
            ).read_text(encoding="utf-8")
        )
        guard = config["dependency_guard"]
        self.assertFalse(guard["included_in_formal_allocations"])
        self.assertEqual(guard["resources"]["gpus"], 0)
        self.assertEqual(
            guard["dependency"], "afterany:<all_five_formal_job_ids>"
        )
        self.assertEqual(len(config["allocations"]), 5)
        worker = (
            ROOT / "tools/run_qwen3vl_p1_dependency_guard.slurm"
        ).read_text(encoding="utf-8")
        self.assertNotIn("#SBATCH --gres", worker)
        submitter = (ROOT / "tools/submit_qwen3vl_p1.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("GUARD_DEPENDENCY=", submitter)
        self.assertIn("afterany:%s", submitter)
        self.assertIn('--dependency="$GUARD_DEPENDENCY"', submitter)
        self.assertIn('dependency_value="afterok:${JOB_IDS[-1]}"', submitter)

    def test_dependency_guard_synthesizes_only_never_started_downstream(
        self,
    ) -> None:
        accounting = (
            ("COMPLETED", "0:0", "2026-08-02T00:00:00", "2026-08-02T00:01:00", "None"),
            ("FAILED", "1:0", "2026-08-02T00:01:00", "2026-08-02T00:02:00", "NonZeroExitCode"),
            ("CANCELLED", "0:0", "Unknown", "2026-08-02T00:02:00", "DependencyNeverSatisfied"),
            ("CANCELLED", "0:0", "Unknown", "2026-08-02T00:02:00", "DependencyNeverSatisfied"),
            ("CANCELLED", "0:0", "Unknown", "2026-08-02T00:02:00", "DependencyNeverSatisfied"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            completed, output_root, notifier_log = self._run_dependency_guard_fixture(
                Path(temporary),
                accounting=accounting,
                existing_status_job_ids={"101", "102"},
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            smoke_h0_status = (
                output_root / self._OUTPUT_NAMES[1] / "logs/status-102.txt"
            )
            self.assertEqual(
                smoke_h0_status.read_text(encoding="utf-8"),
                "worker-owned-status=102\n",
            )
            for index in (2, 3, 4):
                job_id = self._FORMAL_JOB_IDS[index]
                log_dir = output_root / self._OUTPUT_NAMES[index] / "logs"
                status = (log_dir / f"status-{job_id}.txt").read_text(
                    encoding="utf-8"
                )
                self.assertIn("status=cancelled\n", status)
                self.assertIn("result=not_started_due_to_failed_dependency\n", status)
                self.assertIn("status_writer=qwen3vl_p1_dependency_guard_v1\n", status)
                attempt = (log_dir / f"qq-attempt-{job_id}.txt").read_text(
                    encoding="utf-8"
                )
                self.assertIn("attempted=true\n", attempt)
                self.assertIn("notifier_exit_code=0\n", attempt)
                self.assertIn("delivery_claimed=false\n", attempt)
            guard_receipt = (
                output_root
                / self._OUTPUT_NAMES[0]
                / "dependency_guard_receipt.txt"
            ).read_text(encoding="utf-8")
            self.assertIn("status=success\n", guard_receipt)
            self.assertIn(
                "synthesized_dependency_status_job_ids=103,104,105\n",
                guard_receipt,
            )
            self.assertEqual(
                len(notifier_log.read_text(encoding="utf-8").splitlines()),
                4,
            )

    def test_dependency_guard_fails_closed_for_started_job_missing_status(
        self,
    ) -> None:
        accounting = (
            ("COMPLETED", "0:0", "2026-08-02T00:00:00", "2026-08-02T00:01:00", "None"),
            ("FAILED", "1:0", "2026-08-02T00:01:00", "2026-08-02T00:02:00", "NonZeroExitCode"),
            ("CANCELLED", "0:0", "2026-08-02T00:02:00", "2026-08-02T00:03:00", "DependencyNeverSatisfied"),
            ("CANCELLED", "0:0", "Unknown", "2026-08-02T00:03:00", "DependencyNeverSatisfied"),
            ("CANCELLED", "0:0", "Unknown", "2026-08-02T00:03:00", "DependencyNeverSatisfied"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            completed, output_root, _ = self._run_dependency_guard_fixture(
                Path(temporary),
                accounting=accounting,
                existing_status_job_ids={"101", "102", "104", "105"},
            )
            self.assertNotEqual(completed.returncode, 0)
            log_dir = output_root / self._OUTPUT_NAMES[2] / "logs"
            self.assertFalse((log_dir / "status-103.txt").exists())
            self.assertFalse((log_dir / "qq-attempt-103.txt").exists())
            guard_receipt = (
                output_root
                / self._OUTPUT_NAMES[0]
                / "dependency_guard_receipt.txt"
            ).read_text(encoding="utf-8")
            self.assertIn("status=failed\n", guard_receipt)
            self.assertIn("unresolved_missing_status_job_ids=103\n", guard_receipt)

    def test_dependency_guard_requires_live_dependency_evidence_when_sacct_reason_is_none(
        self,
    ) -> None:
        accounting = (
            ("COMPLETED", "0:0", "2026-08-02T00:00:00", "2026-08-02T00:01:00", "None"),
            ("FAILED", "3:0", "2026-08-02T00:01:00", "2026-08-02T00:02:00", "Dependency"),
            ("CANCELLED", "0:0", "None", "2026-08-02T00:02:00", "None"),
            ("CANCELLED", "0:0", "None", "2026-08-02T00:02:00", "None"),
            ("CANCELLED", "0:0", "None", "2026-08-02T00:02:00", "None"),
        )
        control_records = {
            "103": "JobId=103 JobState=CANCELLED Reason=DependencyNeverSatisfied Dependency=afterok:102(failed) NodeList=",
            "104": "JobId=104 JobState=CANCELLED Reason=DependencyNeverSatisfied Dependency=afterok:103(cancelled) NodeList=",
            "105": "JobId=105 JobState=CANCELLED Reason=DependencyNeverSatisfied Dependency=afterok:104(cancelled) NodeList=",
        }
        with tempfile.TemporaryDirectory() as temporary:
            completed, output_root, _ = self._run_dependency_guard_fixture(
                Path(temporary),
                accounting=accounting,
                existing_status_job_ids={"101", "102"},
                control_records=control_records,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            receipt = (
                output_root / self._OUTPUT_NAMES[0] / "dependency_guard_receipt.txt"
            ).read_text(encoding="utf-8")
            self.assertIn("synthesized_dependency_status_job_ids=103,104,105\n", receipt)
            status = (
                output_root / self._OUTPUT_NAMES[2] / "logs/status-103.txt"
            ).read_text(encoding="utf-8")
            self.assertIn("expected_dependency=afterok:102\n", status)
            self.assertIn("dependency_evidence_source=scontrol_exact\n", status)
            self.assertIn("controller_job_state=CANCELLED\n", status)
            self.assertIn("controller_reason=DependencyNeverSatisfied\n", status)
            self.assertIn("controller_dependency=afterok:102(failed)\n", status)

            tampered_records = {
                "wrong_reason": (
                    "JobId=103 JobState=CANCELLED Reason=None "
                    "Dependency=afterok:102(failed) NodeList="
                ),
                "wrong_job": (
                    "JobId=999 JobState=CANCELLED Reason=DependencyNeverSatisfied "
                    "Dependency=afterok:102(failed) NodeList="
                ),
                "wrong_state": (
                    "JobId=103 JobState=RUNNING Reason=DependencyNeverSatisfied "
                    "Dependency=afterok:102(failed) NodeList=gpu48"
                ),
                "missing_dependency": (
                    "JobId=103 JobState=CANCELLED "
                    "Reason=DependencyNeverSatisfied NodeList="
                ),
                "wrong_predecessor": (
                    "JobId=103 JobState=CANCELLED Reason=DependencyNeverSatisfied "
                    "Dependency=afterok:101(failed) NodeList="
                ),
                "extra_edge": (
                    "JobId=103 JobState=CANCELLED Reason=DependencyNeverSatisfied "
                    "Dependency=afterok:102(failed),afterok:999(failed) NodeList="
                ),
                "duplicate_dependency_field": (
                    "JobId=103 JobState=CANCELLED Reason=DependencyNeverSatisfied "
                    "Dependency=afterok:102(failed) "
                    "Dependency=afterok:999(failed) NodeList="
                ),
            }
            for label, bad_record in tampered_records.items():
                with self.subTest(label=label):
                    tampered = dict(control_records)
                    tampered["103"] = bad_record
                    tampered_root = Path(temporary) / label
                    tampered_root.mkdir()
                    completed, _, _ = self._run_dependency_guard_fixture(
                        tampered_root,
                        accounting=accounting,
                        existing_status_job_ids={"101", "102"},
                        control_records=tampered,
                    )
                    self.assertNotEqual(completed.returncode, 0)

            missing_root = Path(temporary) / "missing_scontrol_record"
            missing_root.mkdir()
            missing_record = dict(control_records)
            missing_record.pop("103")
            completed, _, _ = self._run_dependency_guard_fixture(
                missing_root,
                accounting=accounting,
                existing_status_job_ids={"101", "102"},
                control_records=missing_record,
            )
            self.assertNotEqual(completed.returncode, 0)

            drift_root = Path(temporary) / "submission_dependency_drift"
            drift_root.mkdir()
            completed, _, _ = self._run_dependency_guard_fixture(
                drift_root,
                accounting=accounting,
                existing_status_job_ids={"101", "102"},
                control_records=control_records,
                submission_dependency_overrides={"103": "afterok:101"},
            )
            self.assertNotEqual(completed.returncode, 0)

    def test_submitter_arm_identity_is_exact_and_non_concatenating(self) -> None:
        source = (ROOT / "tools/submit_qwen3vl_p1.sh").read_text(encoding="utf-8")
        self.assertNotIn(
            '"$([[ "$arm" == h0 && "$gpus" == 1 ]]',
            source,
        )
        resolver_anchor = source.index("mapfile -t RESOLVED")
        resolver_start = source.index("<<'PY'\n", resolver_anchor) + len("<<'PY'\n")
        resolver_end = source.index('\nPY\n); [[ "${#RESOLVED[@]}"', resolver_start)
        resolver = source[resolver_start:resolver_end]
        function_start = source.index("write_submission_receipt() {")
        function_end = source.index("\n}\n\nfor row", function_start) + len("\n}")
        receipt_function = source[function_start:function_end]

        def canonical(payload: object) -> str:
            raw = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            return hashlib.sha256(raw).hexdigest()

        with tempfile.TemporaryDirectory() as temporary:
            fixture_root = Path(temporary)
            run_root = fixture_root / "run"
            workspace_root = fixture_root / "workspace"
            artifact_root = fixture_root / "artifact"
            output_root = artifact_root / "outputs"
            model = fixture_root / "model"
            input_root = fixture_root / "input"
            simpleclick = fixture_root / "simpleclick.pth"
            sam3 = fixture_root / "sam3.pt"
            sam3_bpe = fixture_root / "sam3.bpe"
            for directory in (
                run_root,
                workspace_root,
                output_root,
                model,
                input_root,
            ):
                directory.mkdir(parents=True)
            for path in (simpleclick, sam3, sam3_bpe):
                path.write_bytes(b"identity-fixture")

            relative_root = Path("doc/training_pipeline/0802_third_try")
            config_path = run_root / relative_root / "QWEN3VL_P1_RESOLVED_RUN_CONFIG_0802.json"
            amendment_path = run_root / relative_root / "QWEN3VL_P1_EXECUTION_AMENDMENT_0802.json"
            input_path = run_root / relative_root / "QWEN3VL_EVALUATION_INPUT_MANIFEST_0802.json"
            fixture_path = run_root / relative_root / "QWEN3VL_H0_H1_CPU_FIXTURE_MANIFEST_0802.json"

            input_manifest = json.loads(
                (ROOT / relative_root / input_path.name).read_text(encoding="utf-8")
            )
            fixture = json.loads(
                (ROOT / relative_root / fixture_path.name).read_text(encoding="utf-8")
            )
            fixture["execution_readiness"] = "execution_ready"
            fixture["p1_execution_authorized"] = True
            write_json(input_path, input_manifest)
            write_json(fixture_path, fixture)

            config = json.loads(
                (ROOT / relative_root / config_path.name).read_text(encoding="utf-8")
            )
            config["configuration_status"] = "frozen"
            config["execution_readiness"] = "execution_ready"
            config["p1_execution_authorized"] = True
            python = str(Path(sys.executable).resolve())
            config["paths"].update(
                {
                    "workspace_root": str(workspace_root.resolve()),
                    "clean_run_worktree": str(run_root.resolve()),
                    "artifact_repository": str(artifact_root.resolve()),
                    "output_root": str(output_root.resolve()),
                    "input_root": str(input_root.resolve()),
                    "base_model": str(model.resolve()),
                    "simpleclick_checkpoint": str(simpleclick.resolve()),
                    "sam3_checkpoint": str(sam3.resolve()),
                    "sam3_bpe": str(sam3_bpe.resolve()),
                    "omniovcd_python": python,
                    "segagent_python": python,
                }
            )
            for item, output_name in zip(
                config["allocations"], self._OUTPUT_NAMES, strict=True
            ):
                item["output_path"] = str((output_root / output_name).resolve())

            cpu_hash = "a" * 64
            gpu_hash = "b" * 64
            guard_hash = "c" * 64
            submitter_hash = "d" * 64
            bindings = config["freeze_commit_hash_bindings"]
            bindings["status"] = "frozen"
            bindings["worker_sha256"] = {
                "cpu_fixtures": cpu_hash,
                "evaluation": gpu_hash,
                "dependency_guard": guard_hash,
            }
            bindings["tool_files_sha256"]["harness_worker"] = gpu_hash
            bindings["tool_files_sha256"]["submitter"] = submitter_hash
            # A draft may deliberately null any tool whose bytes changed
            # pending the clean implementation hash job.  This fixture turns
            # its private clone into a fully frozen synthetic contract, so it
            # must resolve every such placeholder instead of assuming only
            # the worker and submitter hashes can be null.
            for key, value in bindings["tool_files_sha256"].items():
                if value is None:
                    bindings["tool_files_sha256"][key] = hashlib.sha256(
                        f"submitter-identity-fixture:{key}".encode()
                    ).hexdigest()
            write_json(config_path, config)

            amendment = json.loads(
                (ROOT / relative_root / amendment_path.name).read_text(encoding="utf-8")
            )
            implementation_commit = "e" * 40
            amendment["record_status"] = "frozen"
            amendment["execution_readiness"] = "execution_ready"
            amendment["p1_execution_authorized"] = True
            amendment["implementation_commit"] = implementation_commit
            amendment["clean_run_worktree"] = str(run_root.resolve())
            amendment["cpu_fixture_manifest_sha256"] = canonical(fixture)
            amendment["worker_sha256"] = dict(bindings["worker_sha256"])
            amendment["model_hash"] = dict(bindings["model_files_sha256"])
            amendment["tool_hashes"] = dict(bindings["tool_files_sha256"])
            amendment["summary_schema_definition_and_validator_hashes"].update(
                {
                    "definition_sha256": bindings["tool_files_sha256"][
                        "summarizer"
                    ],
                    "validator_sha256": bindings["tool_files_sha256"][
                        "summary_validator"
                    ],
                }
            )
            amendment["input_manifest_canonical_sha256"] = canonical(input_manifest)
            amendment["resolved_run_config_sha256"] = sha256(config_path)
            amendment["unique_output_paths"] = [
                item["output_path"] for item in config["allocations"]
            ]
            amendment["submission_script_path_and_sha256"] = {
                "path": "tools/submit_qwen3vl_p1.sh",
                "sha256": submitter_hash,
            }
            write_json(amendment_path, amendment)

            completed = subprocess.run(
                [
                    sys.executable,
                    "-",
                    str(config_path),
                    str(amendment_path),
                    str(input_path),
                    str(fixture_path),
                    str(run_root),
                    str(workspace_root),
                    str(artifact_root),
                    str(output_root),
                    python,
                    python,
                    python,
                    str(model),
                    str(input_root),
                    str(simpleclick),
                    str(sam3),
                    str(sam3_bpe),
                    "f" * 40,
                    implementation_commit,
                    sha256(config_path),
                    cpu_hash,
                    gpu_hash,
                    guard_hash,
                    submitter_hash,
                    canonical(input_manifest),
                    canonical(fixture),
                    canonical(amendment),
                ],
                input=resolver,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            rows = completed.stdout.splitlines()
            self.assertEqual(len(rows), 5)
            fields = [row.split("|") for row in rows]
            self.assertTrue(all(len(row) == 16 for row in fields))
            expected_identities = {
                "h0_h1": ("not_applicable", "not_applicable"),
                "h0": (
                    "v12_1-h0-legacy-single-round-base-0802",
                    "qwen3vl_harness_h0_legacy_single_round_v1",
                ),
                "h1": (
                    "v12_1-h1-gtfree-terminal-review-base-0802",
                    "qwen3vl_harness_h1_gtfree_terminal_review_v1",
                ),
            }
            for row in fields:
                self.assertEqual((row[14], row[15]), expected_identities[row[3]])

            rows_path = fixture_root / "resolved-rows.txt"
            rows_path.write_text(completed.stdout, encoding="utf-8")
            bash_program = f"""set -euo pipefail
{receipt_function}
EXPECTED_COMMIT={'f' * 40}
EXPECTED_IMPLEMENTATION_COMMIT={implementation_commit}
CPU_WORKER_SHA256={cpu_hash}
GPU_WORKER_SHA256={gpu_hash}
CONFIG_SHA256={sha256(config_path)}
AMENDMENT_CANONICAL_SHA256={canonical(amendment)}
PARTITION=gre
NODE=gpu48
while IFS= read -r row; do
  IFS='|' read -r allocation_id job_name phase arm profile evaluation_id output qq_title gpus cpus memory time_limit dependency samples experiment_arm_id harness_protocol_id <<<"$row"
  mkdir -p -- "$output"
  write_submission_receipt "$output" "$allocation_id" "$job_name" "$profile" "$evaluation_id" "$arm" \
    "$experiment_arm_id" "$harness_protocol_id" "$gpus" "$cpus" "$memory" "$time_limit" \
    "$dependency" 777 "$samples"
done <{rows_path}
"""
            receipt_run = subprocess.run(
                ["bash", "-c", bash_program],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(receipt_run.returncode, 0, receipt_run.stderr)
            for row in fields:
                receipt = (Path(row[6]) / "submission.txt").read_text(
                    encoding="utf-8"
                )
                expected_arm, expected_protocol = expected_identities[row[3]]
                self.assertIn(f"experiment_arm_id={expected_arm}\n", receipt)
                self.assertIn(f"harness_protocol_id={expected_protocol}\n", receipt)
                self.assertEqual(receipt.count("experiment_arm_id="), 1)
                self.assertEqual(receipt.count("harness_protocol_id="), 1)

    def test_dependency_guard_records_failed_best_effort_notifier(self) -> None:
        accounting = (
            ("COMPLETED", "0:0", "2026-08-02T00:00:00", "2026-08-02T00:01:00", "None"),
            ("COMPLETED", "0:0", "2026-08-02T00:01:00", "2026-08-02T00:02:00", "None"),
            ("COMPLETED", "0:0", "2026-08-02T00:02:00", "2026-08-02T00:03:00", "None"),
            ("FAILED", "1:0", "2026-08-02T00:03:00", "2026-08-02T00:04:00", "NonZeroExitCode"),
            ("CANCELLED", "0:0", "Unknown", "2026-08-02T00:04:00", "DependencyNeverSatisfied"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            completed, output_root, _ = self._run_dependency_guard_fixture(
                Path(temporary),
                accounting=accounting,
                existing_status_job_ids={"101", "102", "103", "104"},
                notifier_exit_code=42,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            attempt = (
                output_root
                / self._OUTPUT_NAMES[4]
                / "logs/qq-attempt-105.txt"
            ).read_text(encoding="utf-8")
            self.assertIn("notifier_exit_code=42\n", attempt)
            self.assertIn("delivery_claimed=false\n", attempt)

    def test_dependency_guard_rejects_non_parent_implementation_commit(
        self,
    ) -> None:
        accounting = tuple(
            (
                "COMPLETED",
                "0:0",
                f"2026-08-02T00:0{index}:00",
                f"2026-08-02T00:0{index + 1}:00",
                "None",
            )
            for index in range(5)
        )
        with tempfile.TemporaryDirectory() as temporary:
            completed, output_root, _ = self._run_dependency_guard_fixture(
                Path(temporary),
                accounting=accounting,
                existing_status_job_ids=set(self._FORMAL_JOB_IDS),
                expected_implementation_commit="f" * 40,
            )
            self.assertNotEqual(completed.returncode, 0)
            guard_receipt = (
                output_root
                / self._OUTPUT_NAMES[0]
                / "dependency_guard_receipt.txt"
            ).read_text(encoding="utf-8")
            self.assertIn("status=failed\n", guard_receipt)
            self.assertIn("not the direct child", guard_receipt)

    def test_runtime_manifest_pre_phase_does_not_hash_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_root = root / "inputs"
            for directory, content in (
                ("A", b"online-a"),
                ("B", b"online-b"),
                ("label_cvt", b"offline-label"),
            ):
                path = input_root / directory / "sample.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            model = root / "model"
            model.mkdir()
            (model / "model.safetensors").write_bytes(b"tiny-model")
            tool = root / "tool.bin"
            tool.write_bytes(b"tool")
            worker_file = root / "evaluation-worker.slurm"
            worker_file.write_bytes(b"worker")
            source = root / "runtime.py"
            source.write_text("pass\n", encoding="utf-8")
            tool_hash = sha256(tool)
            worker_hash = sha256(worker_file)
            config = write_json(
                root / "config.json",
                {
                    "schema_id": "qwen3vl_p1_resolved_run_config_v1",
                    "configuration_status": "frozen",
                    "execution_readiness": "execution_ready",
                    "p1_execution_authorized": True,
                    "paths": {"base_model": str(model.resolve())},
                    "freeze_commit_hash_bindings": {
                        "status": "frozen",
                        "model_files_sha256": {
                            "model.safetensors": sha256(
                                model / "model.safetensors"
                            )
                        },
                        "tool_files_sha256": {
                            "harness_worker": worker_hash,
                            "tool": tool_hash,
                        },
                        "worker_sha256": {"evaluation": worker_hash},
                    },
                },
            )
            evaluation = {
                "evaluation_profiles": [
                    {
                        "evaluation_id": "DEV-SMOKE-1-H1-V1",
                        "ordered_sample_ids": ["sample"],
                    }
                ],
                "samples": {
                    "sample": {
                        "a_path_relative_to_input_root": "A/sample.png",
                        "a_sha256": sha256(input_root / "A/sample.png"),
                        "b_path_relative_to_input_root": "B/sample.png",
                        "b_sha256": sha256(input_root / "B/sample.png"),
                        "label_path_relative_to_input_root": "label_cvt/sample.png",
                        "label_sha256": sha256(
                            input_root / "label_cvt/sample.png"
                        ),
                    }
                },
            }
            evaluation_path = write_json(root / "evaluation.json", evaluation)
            output = root / "input_manifest.json"
            original_sha256_file = RUNTIME.sha256_file

            def forbid_label(path: Path) -> str:
                self.assertNotIn("label_cvt", path.parts)
                return original_sha256_file(path)

            argv = [
                "--phase",
                "pre_rollout",
                "--output",
                str(output),
                "--run-dir",
                str(root / "run"),
                "--input-root",
                str(input_root),
                "--model-path",
                str(model),
                "--evaluation-input-manifest",
                str(evaluation_path),
                "--evaluation-input-manifest-canonical-sha256",
                RUNTIME.canonical_sha256(evaluation),
                "--resolved-run-config",
                str(config),
                "--resolved-run-config-sha256",
                sha256(config),
                "--evaluation-profile",
                "p1_smoke1",
                "--evaluation-id",
                "DEV-SMOKE-1-H1-V1",
                "--arm",
                "h1",
                "--experiment-arm-id",
                PAIR.ARM_IDS["h1"],
                "--harness-protocol-id",
                PAIR.PROTOCOL_IDS["h1"],
                "--model-io-protocol",
                "v12.1",
                "--runtime-commit",
                "a" * 40,
                "--worker-sha256",
                worker_hash,
                "--job-id",
                "123",
                "--runtime-device",
                "NVIDIA A800-SXM4-80GB",
                "--samples",
                "sample",
                "--tool-file",
                f"tool={tool}",
                "--tool-file",
                f"harness_worker={worker_file}",
                "--runtime-source",
                f"source={source}",
            ]
            with mock.patch.object(RUNTIME, "git_head", return_value="a" * 40), mock.patch.object(
                RUNTIME, "sha256_file", side_effect=forbid_label
            ), mock.patch.dict(
                os.environ,
                {
                    "SLURM_JOB_NAME": "fixture",
                    "SLURMD_NODENAME": "gpu48",
                    "SLURM_JOB_PARTITION": "gre",
                },
                clear=False,
            ):
                self.assertEqual(RUNTIME.main(argv), 0)
                for target in (
                    model / "model.safetensors",
                    tool,
                    worker_file,
                ):
                    original = target.read_bytes()
                    target.write_bytes(b"external-drift")
                    with self.subTest(target=target.name), self.assertRaises(
                        RUNTIME.RuntimeManifestError
                    ):
                        RUNTIME.build_payload(
                            RUNTIME.parse_args(argv), include_gt=False
                        )
                    target.write_bytes(original)
                missing_tool_args = RUNTIME.parse_args(argv)
                missing_tool_args.tool_file = [f"harness_worker={worker_file}"]
                with self.assertRaisesRegex(
                    RUNTIME.RuntimeManifestError,
                    "exact set",
                ):
                    RUNTIME.build_payload(
                        missing_tool_args, include_gt=False
                    )
                original_config = json.loads(config.read_text(encoding="utf-8"))
                unauthorized_config = dict(original_config)
                unauthorized_config["configuration_status"] = "draft"
                write_json(config, unauthorized_config)
                unauthorized_args = RUNTIME.parse_args(argv)
                unauthorized_args.resolved_run_config_sha256 = sha256(config)
                with self.assertRaisesRegex(
                    RUNTIME.RuntimeManifestError,
                    "not frozen",
                ):
                    RUNTIME.build_payload(
                        unauthorized_args, include_gt=False
                    )
                write_json(config, original_config)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["phase"], "pre_rollout_gt_free")
            self.assertIsNone(payload["files_sha256"]["sample"]["label_cvt"])
            self.assertFalse(payload["gt_hashed_after_rollout"])
            self.assertFalse(payload["gt_boundary"]["label_content_accessed"])

    def test_evaluation_worker_tool_files_match_frozen_exact_key_set(self) -> None:
        import re

        config = json.loads(
            (
                ROOT
                / "doc/training_pipeline/0802_third_try/"
                "QWEN3VL_P1_RESOLVED_RUN_CONFIG_0802.json"
            ).read_text(encoding="utf-8")
        )
        frozen_keys = set(
            config["freeze_commit_hash_bindings"]["tool_files_sha256"]
        )
        worker = (ROOT / "tools/run_qwen3vl_p1_evaluation.slurm").read_text(
            encoding="utf-8"
        )
        worker_keys = re.findall(r'--tool-file "([^=]+)=', worker)
        self.assertEqual(len(worker_keys), len(set(worker_keys)))
        self.assertEqual(set(worker_keys), frozen_keys)

    def test_submitter_and_workers_bind_resolved_paths_and_python_identities(
        self,
    ) -> None:
        config = json.loads(
            (
                ROOT
                / "doc/training_pipeline/0802_third_try/"
                "QWEN3VL_P1_RESOLVED_RUN_CONFIG_0802.json"
            ).read_text(encoding="utf-8")
        )
        paths = config["paths"]
        required_paths = {
            "workspace_root",
            "clean_run_worktree",
            "artifact_repository",
            "output_root",
            "input_root",
            "base_model",
            "simpleclick_checkpoint",
            "sam3_checkpoint",
            "sam3_bpe",
            "omniovcd_python",
            "segagent_python",
        }
        self.assertTrue(required_paths <= set(paths))
        for key in required_paths:
            self.assertTrue(Path(paths[key]).is_absolute(), key)
        self.assertTrue(
            all(
                Path(item["output_path"]).parent == Path(paths["output_root"])
                for item in config["allocations"]
            )
        )

        sources = {
            name: (ROOT / relative).read_text(encoding="utf-8")
            for name, relative in (
                ("submitter", "tools/submit_qwen3vl_p1.sh"),
                ("cpu", "tools/run_qwen3vl_p1_cpu_fixtures.slurm"),
                ("evaluation", "tools/run_qwen3vl_p1_evaluation.slurm"),
                ("guard", "tools/run_qwen3vl_p1_dependency_guard.slurm"),
            )
        }
        for name, source in sources.items():
            with self.subTest(component=name):
                self.assertIn('config.get("configuration_status") != "frozen"', source)
                self.assertIn('config.get("execution_readiness") != "execution_ready"', source)
                self.assertIn('config.get("p1_execution_authorized") is not True', source)
                self.assertIn("resolved(sys.executable)", source)
        for token in (
            '"clean_run_worktree": root_text',
            '"artifact_repository": artifact_text',
            '"output_root": output_root_text',
            '"omniovcd_python": omniovcd_python_text',
        ):
            self.assertIn(token, sources["submitter"])
        for name in ("cpu", "evaluation", "guard"):
            self.assertIn('("clean_run_worktree", root', sources[name])
            self.assertIn('("artifact_repository", artifact', sources[name])
            self.assertIn('("output_root", output_root', sources[name])
            self.assertIn('("omniovcd_python",', sources[name])
        self.assertIn('"input_root": input_root_text', sources["submitter"])
        self.assertIn('"base_model": model_text', sources["submitter"])
        self.assertIn(
            'str(resolved(item["output_path"], strict=False))',
            sources["submitter"],
        )
        self.assertIn(
            'resolved(declared_output) != resolved(output_text)',
            sources["cpu"],
        )
        self.assertIn('("input_root", input_root)', sources["evaluation"])
        self.assertIn('("base_model", model)', sources["evaluation"])
        self.assertIn(
            'raise SystemExit("paired H0 receipt path differs from the resolved config")',
            sources["evaluation"],
        )
        self.assertIn(
            'observed_output = resolved(pathlib.Path(output_root_text) / output_name)',
            sources["guard"],
        )
        self.assertNotIn('MODEL_PATH="${MODEL_PATH:-', sources["submitter"])
        self.assertNotIn('INPUT_ROOT="${INPUT_ROOT:-', sources["evaluation"])

    def test_pair_gate_requires_smoke_round_two(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            implementation_commit = "a" * 40
            freeze_commit = "b" * 40
            evaluation = write_json(root / "evaluation.json", {"frozen": True})
            run_manifest = root / "run_manifest.md"
            run_manifest.write_text("frozen\n", encoding="utf-8")
            summary_validation = root / "summary_validation.txt"
            summary_validation.write_text("status=valid\n", encoding="utf-8")
            runtime_common = {
                "evaluation_profile": "p1_smoke1",
                "evaluation_id": "DEV-SMOKE-1-H1-V1",
                "sample_order": ["test_1_1"],
                "input_root": "/fixed/input",
                "files_sha256": {
                    "test_1_1": {"A": "a", "B": "b", "label_cvt": "g"}
                },
                "model_path": "/fixed/model",
                "model_files_sha256": {"model.safetensors": "m"},
                "tool_files_sha256": {"harness_worker": "w"},
                "runtime_source_sha256": {"runner": "r"},
            }

            def make_arm(arm: str, round_two: int) -> tuple[Path, dict]:
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
                    "review_region_id": "r0",
                    "proposal_version": None,
                }
                staged_evidence = {
                    "type": "staged_qwen_verifier",
                    "schema_version": (
                        "staged_verifier_environment_ordered_compact_audit_v12_1"
                    ),
                    "verifier_valid": True,
                    "validation_errors": [],
                    "regional_validation_errors": [],
                    "stage_trace": {
                        "mode": "initial",
                        "evidence": [
                            {"region_id": "r0", "audit_kind": "present"}
                        ],
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
                            "visual_input_scope": "standard",
                            "network_retry_attempt_count": 0,
                            "prompt_sha256": "f" * 64,
                            "raw_response": "{}",
                            "parsed_output": {"region_id": "r0"},
                        }
                    ],
                }
                if arm == "h1" and round_two:
                    staged_evidence["backend_calls"][0]["terminal_review"] = context
                    staged_evidence["terminal_review_round"] = {
                        "schema_valid": True,
                        "completion_gate_passed": True,
                    }
                if arm == "h1":
                    final_region_valid = bool(
                        round_two
                        and SUMMARY_VALIDATOR.independently_validate_staged_region_evidence(
                            staged_evidence,
                            require_round_two=True,
                        )
                    )
                else:
                    final_region_valid = (
                        SUMMARY_VALIDATOR.independently_validate_staged_region_evidence(
                            staged_evidence,
                            require_round_two=False,
                        )
                    )
                terminal = {
                    "round_two_invoked": round_two == 1,
                    "round_two_count": round_two,
                    "round_index": 2 if round_two else None,
                    "observed_outcome": "finish" if round_two else None,
                    "contract_violation": None,
                    "fresh_backend_context": True if round_two else None,
                    "nonduplicate_prompt_or_region": True if round_two else None,
                    "gt_free": True,
                    "forbidden_trigger_inputs_present": [],
                    "backend_calls": [{"round_index": 2}] if round_two else [],
                }
                summary = write_json(
                    root / f"{arm}_summary.json",
                    {
                        "aggregate": {
                            "initial": {"iou": 0.5, "f1": 0.6},
                            "selected": {"iou": 0.5, "f1": 0.6},
                            "delta_iou": 0.0,
                            "delta_f1": 0.0,
                        },
                        "requested_samples": 1,
                        "scored_samples": 1,
                        "p1_contract": {
                            "final_region_protocol_valid_count": int(final_region_valid),
                            "round_two_invoked_count": round_two,
                            "per_sample": {
                                "test_1_1": {
                                    "terminal_review": terminal,
                                    "final_region_protocol_valid": final_region_valid,
                                }
                            },
                        },
                    },
                )
                runtime = write_json(
                    root / f"{arm}_runtime.json",
                    {
                        **runtime_common,
                        "terminal_review_arm": arm,
                        "experiment_arm": PAIR.ARM_IDS[arm],
                    },
                )
                receipt_path = write_json(root / f"{arm}_receipt.json", {})
                receipt = {
                    "identity": {
                        "arm": arm,
                        "experiment_arm_id": PAIR.ARM_IDS[arm],
                        "harness_protocol_id": PAIR.PROTOCOL_IDS[arm],
                        "evaluation_profile": "p1_smoke1",
                        "evaluation_id": "DEV-SMOKE-1-H1-V1",
                        "sample_order": ["test_1_1"],
                        "model_state": "base",
                        "runtime_commit": freeze_commit,
                        "job_id": "1" if arm == "h0" else "2",
                    },
                    "provenance": {
                        "implementation_commit": implementation_commit,
                        "freeze_commit": freeze_commit,
                        "run_worktree": "/fixed/worktree",
                        "git_dirty": False,
                        "git_worktree_sha256": "c" * 64,
                        "model_path": "/fixed/model",
                        "model_sha256": "d" * 64,
                        "tool_sha256": {"worker": "e" * 64},
                        "worker_sha256": "e" * 64,
                        "evaluation_input_manifest_canonical_sha256": "f" * 64,
                    },
                    "gates": {
                        "gt_loaded_before_rollout_complete": False,
                        "pre_rollout_gt_access_count": 0,
                        "safety_boundary_relaxation_count": 0,
                        "requested_sample_count": 1,
                        "scored_sample_count": 1,
                        "requested_scored_mismatch_count": 0,
                        "tool_failure_count": 0,
                        "offline_scoring_after_rollout_complete": True,
                        "summary_validator_passed": True,
                    },
                    "artifacts": {
                        "baseline_summary": artifact(summary),
                        "evaluation_input_manifest": artifact(evaluation),
                        "runtime_input_manifest": artifact(runtime),
                        "run_manifest": artifact(run_manifest),
                        "summary_validation": artifact(summary_validation),
                    },
                }
                return receipt_path, receipt

            h0_path, h0 = make_arm("h0", 0)
            h1_path, h1_without_review = make_arm("h1", 0)
            with self.assertRaises(PAIR.PairComparisonError):
                PAIR.build_payload(
                    h0_path=h0_path,
                    h1_path=h1_path,
                    h0=h0,
                    h1=h1_without_review,
                    profile="p1_smoke1",
                )
            h1_path, h1 = make_arm("h1", 1)
            payload = PAIR.build_payload(
                h0_path=h0_path,
                h1_path=h1_path,
                h0=h0,
                h1=h1,
                profile="p1_smoke1",
            )
            self.assertEqual(payload["status"], "complete")
            self.assertTrue(payload["hard_gates"]["h1_smoke_round_two_schema_valid"])
            self.assertEqual(payload["implementation_commit"], implementation_commit)
            self.assertEqual(payload["freeze_commit"], freeze_commit)
            self.assertNotEqual(
                payload["implementation_commit"],
                payload["freeze_commit"],
            )

            tampered_h1 = json.loads(json.dumps(h1))
            tampered_h1["provenance"]["freeze_commit"] = "c" * 40
            with self.assertRaises(PAIR.PairComparisonError):
                PAIR.build_payload(
                    h0_path=h0_path,
                    h1_path=h1_path,
                    h0=h0,
                    h1=tampered_h1,
                    profile="p1_smoke1",
                )


if __name__ == "__main__":
    unittest.main()
