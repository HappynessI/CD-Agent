from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = (
    "test_20_15",
    "test_78_13",
    "test_85_16",
    "test_1_1",
    "test_50_8",
)
E2E3_SAMPLES = ("test_20_15", "test_1_1", "test_50_8")
EVALUATION_PROFILE = "e2e5"
EVALUATION_ID = "DEV-EXTENDED-5-V1"
E2E3_EVALUATION_PROFILE = "e2e3"
E2E3_EVALUATION_ID = "DEV-CORE-3-V1"
CONTROL_SCHEMA = "staged_verifier_environment_ordered_audit_v11_3"
TREATMENT_SCHEMA = "staged_verifier_environment_ordered_compact_audit_v12_1"
SUMMARY_SCHEMA_VERSION = "qwen3vl_e2e_summary_v2"
V1_ARCHIVE_RUN = (
    ROOT
    / "outputs"
    / "CA_0801_archive"
    / "CA_0801(10)-qwen3vl-v121-post-sft-e2e3-v1"
    / "run"
)


def _load_tool(name: str) -> ModuleType:
    path = ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load tool module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ANALYZER = _load_tool("analyze_staged_verifier_run")
COMPARATOR = _load_tool("compare_local_verifier_ab")
BASELINE_SUMMARIZER = _load_tool("summarize_qwen3vl_e2e_run")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _initial_evidence(
    *, schema: str = TREATMENT_SCHEMA, region_ids: tuple[str, ...] = ("r0", "r1")
) -> dict[str, Any]:
    records = [{"region_id": region_id} for region_id in region_ids]
    compact_schema = schema == TREATMENT_SCHEMA
    primary_ids_key = "audited_region_ids" if compact_schema else "selected_region_ids"
    replan_ids_key = (
        "replan_audited_region_ids"
        if compact_schema
        else "replan_selected_region_ids"
    )
    trace = {
        "mode": "initial",
        "evidence": records,
        primary_ids_key: list(reversed(region_ids)),
        "judgments": [
            {"region_id": region_id} for region_id in reversed(region_ids)
        ],
        "diagnoses": [
            {"region_id": region_id, "error_type": "none"}
            for region_id in reversed(region_ids)
        ],
        "plan": None,
        "replan_evidence": [],
        replan_ids_key: [],
        "replan_judgments": [],
        "replan_diagnoses": [],
    }
    return {
        "type": "staged_qwen_verifier",
        "schema_version": schema,
        "verifier_valid": True,
        "localization_valid": True,
        "validation_errors": [],
        "regional_validation_errors": [],
        "backend_calls": [
            {
                "stage": "audit",
                "attempt_kind": "initial",
                "latency_seconds": 1.0,
                "raw_response": "{}",
            }
            for _ in region_ids
        ],
        "stage_trace": trace,
    }


def _programmatic_evidence(
    mode: str = "programmatic_identical_state",
    *,
    schema: str = TREATMENT_SCHEMA,
) -> dict[str, Any]:
    evidence = {
        "type": "staged_qwen_verifier",
        "schema_version": schema,
        "decision_mode": mode,
        "verifier_valid": True,
        "localization_valid": True,
        "validation_errors": [],
        "backend_calls": [],
    }
    if mode != "programmatic_identical_state":
        evidence["localization_valid"] = mode == "programmatic_alternate_region_after_rollback"
        evidence["replan"] = {
            "selected_region_id": (
                "r1"
                if mode == "programmatic_alternate_region_after_rollback"
                else None
            ),
            "rejected_action_count": 1,
            "rejection_reasons": ["tool rejected prior geometry"],
        }
    return evidence


def _invalid_evidence() -> dict[str, Any]:
    return {
        "type": "staged_qwen_verifier",
        "schema_version": TREATMENT_SCHEMA,
        "verifier_valid": False,
        "localization_valid": False,
        "validation_errors": ["invalid staged response"],
        "backend_calls": [],
    }


def _hard_gate_evidence() -> dict[str, Any]:
    return {
        "type": "runtime_hard_gate_precheck",
        "verifier_skipped": True,
        "rejection_reasons": ["point_proposal_combined_change_gate_exceeded"],
        "gt_available": False,
    }


def _step(
    evidence: dict[str, Any],
    *,
    step_index: int,
    candidate_accepted: bool,
    offline_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    step = {
        "step_index": step_index,
        "candidate_accepted": candidate_accepted,
        "execution": {
            "candidate_accepted": candidate_accepted,
            "verifier_evidence": evidence,
        },
    }
    if offline_metrics is not None:
        step["offline_metrics"] = offline_metrics
    return step


def _episode(sample: str, step_count: int) -> dict[str, Any]:
    candidate_count = step_count - 1
    return {
        "sample": sample,
        "execution_status": "completed",
        "episode_outcome": "no_executable_action",
        "converged": False,
        "stop_reason": "verifier_no_executable_action",
        "elapsed_seconds": float(step_count + 2),
        "loop_count": max(1, candidate_count),
        "candidate_count": candidate_count,
        "accepted_candidate_count": 0,
        "rejected_candidate_count": candidate_count,
        "tool_action_count": 0,
        "empty_mask_scan_fallback_action_count": 0,
        "action_attempt_count": candidate_count,
        "action_attempt_error_count": 0,
    }


def _write_run(
    run_dir: Path,
    *,
    samples: tuple[str, ...] = SAMPLES,
    schema: str = TREATMENT_SCHEMA,
    region_ids: tuple[str, ...] = ("r0", "r1"),
    extra_evidence: tuple[dict[str, Any], ...] = (),
    mutate_first: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    _write_json(
        run_dir / "rollout_complete.json",
        {
            "execution_status": "completed",
            "gt_loaded": False,
            "samples": sorted(samples),
        },
    )
    for index, sample in enumerate(samples):
        initial = _initial_evidence(schema=schema, region_ids=region_ids)
        if index == 0 and mutate_first is not None:
            mutate_first(initial)
        evidence_items = (initial, *copy.deepcopy(extra_evidence))
        steps = [
            _step(
                evidence,
                step_index=step_index,
                candidate_accepted=step_index == 0,
            )
            for step_index, evidence in enumerate(evidence_items)
        ]
        _write_json(
            run_dir / "trajectories" / sample / "trajectory.json",
            {"steps": steps},
        )
        episode = _episode(sample, len(steps))
        episode["elapsed_seconds"] = max(
            episode["elapsed_seconds"], float(len(region_ids) + 1)
        )
        _write_json(
            run_dir / "trajectories" / sample / "episode_summary.json",
            episode,
        )


def _analyze(
    run_dir: Path,
    *,
    role: str | None = None,
    job_id: str = "",
    samples: tuple[str, ...] = SAMPLES,
    evaluation_profile: str = EVALUATION_PROFILE,
    evaluation_id: str = EVALUATION_ID,
) -> dict[str, Any]:
    control = role == "control"
    return ANALYZER.summarize(
        argparse.Namespace(
            run_dir=run_dir,
            output=run_dir / "unused.json",
            evaluation_profile=evaluation_profile,
            evaluation_id=evaluation_id,
            expected_samples=list(samples),
            experiment_arm=("v11_3-control" if control else "v12_1-compact")
            if role
            else "",
            protocol_version=("v11.3" if control else "v12.1") if role else "",
            job_id=job_id,
            runtime_commit=_commit("a" if control else "b") if role else "",
            harness_commit=_commit("c") if role else "",
        )
    )


def _digest(character: str) -> str:
    return character * 64


def _commit(character: str) -> str:
    return character * 40


def _manifest(role: str) -> dict[str, Any]:
    control = role == "control"
    source_digest = _digest("6" if control else "7")
    # The control pin is intentionally literal: changing the comparator's
    # constant must not silently make the fixture self-consistent.
    arm_commit = (
        "f80e2702793d265a78b9539d438b60dd812d5fd1"
        if control
        else _commit("c")
    )
    harness_commit = _commit("c")
    return {
        "evaluation_profile": EVALUATION_PROFILE,
        "evaluation_id": EVALUATION_ID,
        "experiment_arm": "v11_3-control" if control else "v12_1-compact",
        "protocol_version": "v11.3" if control else "v12.1",
        "sample_order": list(SAMPLES),
        "input_root": "/guisongxia01/pangchao/wangyihan/datasets/LEVIR-CD/test_256",
        "files_sha256": {
            sample: {
                "A": _digest("1"),
                "B": _digest("2"),
                "label_cvt": _digest("3"),
            }
            for sample in SAMPLES
        },
        "gt_hashed_after_rollout": True,
        "model_path": "/guisongxia01/pangchao/wangyihan/models/Qwen3-VL-2B-Instruct",
        "model_files_sha256": {"model.safetensors": _digest("4")},
        "tool_files_sha256": {
            key: _digest("5") for key in COMPARATOR.REQUIRED_TOOL_KEYS
        },
        "slurm_node": "gpu25",
        "slurm_partition": "gre",
        "slurm_job_id": "100" if control else "101",
        "runtime_source_sha256": {
            key: source_digest for key in COMPARATOR.REQUIRED_RUNTIME_SOURCE_KEYS
        },
        "runtime_commits": {
            "chang_agent": arm_commit,
            "expected_chang_agent": arm_commit,
            "harness": harness_commit,
            "expected_harness": harness_commit,
            "omniovcd": _commit("d"),
        },
        "runtime_environment": {
            "python": "3.12.13",
            "torch": "2.10.0+cu128",
            "cuda": "12.8",
            "device": "NVIDIA A800",
            "transformers": "4.57.0",
            "accelerate": "1.10.1",
            "numpy": "2.3.2",
            "pillow": "11.3.0",
        },
        "executed_worker_sha256": _digest("5"),
    }


def _write_receipt(path: Path, values: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{key}={value}\n" for key, value in values.items()),
        encoding="utf-8",
    )


def _read_receipt(path: Path) -> dict[str, str]:
    return {
        key: value
        for key, value in (
            line.split("=", 1)
            for line in path.read_text(encoding="utf-8").splitlines()
            if "=" in line
        )
    }


def _metric_block(tp: int, fp: int, fn: int, tn: int) -> dict[str, Any]:
    counts = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return {
        **BASELINE_SUMMARIZER.metrics_from_counts(counts),
    }


def _write_metrics(
    run_dir: Path, *, samples_in_profile: tuple[str, ...] = SAMPLES
) -> None:
    """Write post-rollout metrics with nontrivial initial/selected deltas."""
    samples: dict[str, Any] = {}
    for index, sample in enumerate(samples_in_profile):
        initial = (4 + index, 1, 1, 10 + index)
        # One degraded sample, one improved sample, and the remainder unchanged.
        if index == 0:
            selected = (3, 2, 2, 9)
        elif index == 1:
            selected = (6, 0, 0, 12)
        else:
            selected = initial
        samples[sample] = {
            "initial": _metric_block(*initial),
            "verifier_selected": _metric_block(*selected),
        }
    selected_counts = {key: 0 for key in ("tp", "fp", "fn", "tn")}
    for sample in samples.values():
        selected = sample["verifier_selected"]
        for key in selected_counts:
            selected_counts[key] += selected[key]
    _write_json(
        run_dir / "per_sample_metrics.json",
        {"samples": samples, "aggregate": _metric_block(**selected_counts)},
    )


def _write_custom_metrics(
    run_dir: Path,
    pairs: dict[str, tuple[tuple[int, int, int, int], tuple[int, int, int, int]]],
) -> None:
    samples = {
        sample: {
            "initial": _metric_block(*initial),
            "verifier_selected": _metric_block(*selected),
        }
        for sample, (initial, selected) in pairs.items()
    }
    aggregate_counts = {key: 0 for key in ("tp", "fp", "fn", "tn")}
    for sample in samples.values():
        for key in aggregate_counts:
            aggregate_counts[key] += sample["verifier_selected"][key]
    _write_json(
        run_dir / "per_sample_metrics.json",
        {"samples": samples, "aggregate": _metric_block(**aggregate_counts)},
    )


def _replace_metric_pair(
    run_dir: Path,
    sample: str,
    *,
    initial: tuple[int, int, int, int],
    selected: tuple[int, int, int, int],
) -> None:
    path = run_dir / "per_sample_metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["samples"][sample] = {
        "initial": _metric_block(*initial),
        "verifier_selected": _metric_block(*selected),
    }
    aggregate_counts = {key: 0 for key in ("tp", "fp", "fn", "tn")}
    for metrics in payload["samples"].values():
        for key in aggregate_counts:
            aggregate_counts[key] += metrics["verifier_selected"][key]
    payload["aggregate"] = _metric_block(**aggregate_counts)
    _write_json(path, payload)


def _prepare_action_artifacts(run_dir: Path) -> None:
    """Add episode/tool evidence so all baseline action counters are exercised."""
    episode_updates = (
        {
            "candidate_count": 2,
            "accepted_candidate_count": 1,
            "rejected_candidate_count": 1,
            "tool_action_count": 1,
            "action_attempt_count": 2,
            "action_attempt_error_count": 0,
            "episode_outcome": "action_executed",
            "stop_reason": "verifier_no_executable_action",
        },
        {
            "candidate_count": 1,
            "accepted_candidate_count": 0,
            "rejected_candidate_count": 1,
            "tool_action_count": 0,
            "action_attempt_count": 2,
            "action_attempt_error_count": 1,
            "empty_mask_scan_fallback_action_count": 1,
            "episode_outcome": "no_executable_action",
        },
        {
            "candidate_count": 0,
            "accepted_candidate_count": 0,
            "rejected_candidate_count": 0,
            "tool_action_count": 0,
            "action_attempt_count": 0,
            "action_attempt_error_count": 0,
            "episode_outcome": "no_executable_action",
        },
        {
            "candidate_count": 0,
            "accepted_candidate_count": 0,
            "rejected_candidate_count": 0,
            "tool_action_count": 0,
            "action_attempt_count": 0,
            "action_attempt_error_count": 0,
            "episode_outcome": "finished",
            "stop_reason": "initial_verifier_authorized_finish",
        },
        {
            "candidate_count": 0,
            "accepted_candidate_count": 0,
            "rejected_candidate_count": 0,
            "tool_action_count": 0,
            "action_attempt_count": 0,
            "action_attempt_error_count": 0,
            "episode_outcome": "verifier_invalid",
            "stop_reason": "verifier_invalid_response",
        },
    )
    for sample, update in zip(SAMPLES, episode_updates):
        path = run_dir / "trajectories" / sample / "episode_summary.json"
        episode = json.loads(path.read_text(encoding="utf-8"))
        episode.update(update)
        _write_json(path, episode)

        decisions = [True] * update["accepted_candidate_count"] + [
            False
        ] * update["rejected_candidate_count"]
        trajectory_path = run_dir / "trajectories" / sample / "trajectory.json"
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
        initial_step = trajectory["steps"][0]
        initial_schema = initial_step["execution"]["verifier_evidence"].get(
            "schema_version", TREATMENT_SCHEMA
        )
        trajectory["steps"] = [
            initial_step,
            *[
                _step(
                    _programmatic_evidence(schema=initial_schema),
                    step_index=index,
                    candidate_accepted=decision,
                )
                for index, decision in enumerate(decisions, start=1)
            ],
        ]
        _write_json(trajectory_path, trajectory)

    _write_json(
        run_dir / "tool_runs" / SAMPLES[0] / "point_000" / "report.json",
        {"status": "success", "sample": SAMPLES[0]},
    )
    # A failed report is counted even when no action was ultimately accepted.
    _write_json(
        run_dir / "tool_runs" / SAMPLES[2] / "box_000" / "report.json",
        {"status": "failed", "sample": SAMPLES[2]},
    )


def _baseline(
    run_dir: Path,
    *,
    role: str,
    job_id: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    control = role == "control"
    return BASELINE_SUMMARIZER.summarize(
        argparse.Namespace(
            run_dir=run_dir,
            output=run_dir / "unused-baseline.json",
            evaluation_profile=EVALUATION_PROFILE,
            evaluation_id=EVALUATION_ID,
            model_state="base",
            job_id=job_id,
            experiment_arm="v11_3-control" if control else "v12_1-compact",
            protocol_version="v11.3" if control else "v12.1",
            runtime_commit=manifest["runtime_commits"]["chang_agent"],
            harness_commit=manifest["runtime_commits"]["harness"],
            expected_samples=list(SAMPLES),
        )
    )


def _standalone_baseline(
    run_dir: Path,
    *,
    samples: tuple[str, ...],
    evaluation_profile: str,
    evaluation_id: str,
    model_state: str = "base",
) -> dict[str, Any]:
    return BASELINE_SUMMARIZER.summarize(
        argparse.Namespace(
            run_dir=run_dir,
            output=run_dir / "unused-baseline.json",
            evaluation_profile=evaluation_profile,
            evaluation_id=evaluation_id,
            model_state=model_state,
            job_id="123",
            experiment_arm="contract-fixture",
            protocol_version="contract-v1",
            runtime_commit=_commit("a"),
            harness_commit=_commit("b"),
            expected_samples=list(samples),
        )
    )


def _comparison_protocol(
    run_dir: Path, *, role: str, job_id: str, manifest: dict[str, Any]
) -> dict[str, Any]:
    control = role == "control"
    return ANALYZER.summarize(
        argparse.Namespace(
            run_dir=run_dir,
            output=run_dir / "unused-protocol.json",
            evaluation_profile=EVALUATION_PROFILE,
            evaluation_id=EVALUATION_ID,
            expected_samples=list(SAMPLES),
            experiment_arm="v11_3-control" if control else "v12_1-compact",
            protocol_version="v11.3" if control else "v12.1",
            job_id=job_id,
            runtime_commit=manifest["runtime_commits"]["chang_agent"],
            harness_commit=manifest["runtime_commits"]["harness"],
        )
    )


def _write_comparison_fixture(root: Path) -> tuple[Path, Path, Path]:
    control = root / "control"
    treatment = root / "treatment"
    for arm_root, role, schema in (
        (control, "control", CONTROL_SCHEMA),
        (treatment, "treatment", TREATMENT_SCHEMA),
    ):
        run_dir = arm_root / "run"
        _write_run(run_dir, schema=schema)
        _write_metrics(run_dir)
        _prepare_action_artifacts(run_dir)
        manifest = _manifest(role)
        job_id = "100" if role == "control" else "101"
        _write_json(arm_root / "input_manifest.json", manifest)
        _write_json(
            arm_root / "protocol_summary.json",
            _comparison_protocol(run_dir, role=role, job_id=job_id, manifest=manifest),
        )
        _write_json(
            arm_root / "baseline_summary.json",
            _baseline(run_dir, role=role, job_id=job_id, manifest=manifest),
        )
        _write_receipt(
            arm_root / "submission.txt",
            {
                "job_id": job_id,
                "job_name": f"local-verifier-{role}",
                "partition": "gre",
                "node": "gpu25",
                "dependency": "none" if role == "control" else "afterany:100",
                "arm": manifest["experiment_arm"],
                "protocol": manifest["protocol_version"],
                "arm_repo": str(ROOT),
                "expected_commit": manifest["runtime_commits"]["expected_chang_agent"],
                "expected_harness_commit": manifest["runtime_commits"]["expected_harness"],
                "worker": str(
                    (ROOT / "tools/run_qwen3vl_base_e2e5.slurm").resolve()
                ),
                "worker_sha256": _digest("5"),
                "resources": "gpu:1,cpu:2,mem:32G,time=03:00:00",
                "evaluation_profile": EVALUATION_PROFILE,
                "evaluation_id": EVALUATION_ID,
                "samples": " ".join(SAMPLES),
                "output": str(arm_root.resolve()),
            },
        )
        _write_receipt(
            arm_root / "logs" / f"status-{job_id}.txt",
            {
                "status": "success",
                "exit_code": "0",
                "elapsed_seconds": "30",
                "job_id": job_id,
                "run_dir": str(run_dir.resolve()),
                "worker_sha256": _digest("5"),
                "evaluation_profile": EVALUATION_PROFILE,
                "evaluation_id": EVALUATION_ID,
                "samples": " ".join(SAMPLES),
                "key_result": "protocol=completed;baseline=completed",
                "error": "",
            },
        )
    return control, treatment, root / "comparison"


def _compare(control: Path, treatment: Path, output: Path) -> dict[str, Any]:
    return COMPARATOR.summarize(
        argparse.Namespace(
            control=control,
            treatment=treatment,
            output=output,
            expected_harness_commit=_commit("c"),
        )
    )


def _mutate_json(path: Path, mutation: Callable[[dict[str, Any]], None]) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutation(payload)
    _write_json(path, payload)


class Qwen3VLE2ESummarizerContractTest(unittest.TestCase):
    def _summarize_smoke(
        self,
        *,
        initial: tuple[int, int, int, int],
        selected: tuple[int, int, int, int],
    ) -> dict[str, Any]:
        run_dir = Path(self.temporary.name) / "run"
        _write_run(run_dir, samples=(SAMPLES[0],))
        _write_custom_metrics(run_dir, {SAMPLES[0]: (initial, selected)})
        return _standalone_baseline(
            run_dir,
            samples=(SAMPLES[0],),
            evaluation_profile="smoke1",
            evaluation_id="DEV-SMOKE-1-V1",
        )

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_summary_v2_reports_count_deltas_macro_and_empty_mask_contract(self) -> None:
        result = self._summarize_smoke(
            initial=(0, 0, 0, 16), selected=(0, 4, 0, 12)
        )

        self.assertEqual(result["summary_schema_version"], SUMMARY_SCHEMA_VERSION)
        sample = result["per_sample"][SAMPLES[0]]
        self.assertEqual(sample["initial"]["iou"], 1.0)
        self.assertEqual(sample["initial"]["f1"], 1.0)
        self.assertEqual(sample["delta_counts"], {"tp": 0, "fp": 4, "fn": 0})
        self.assertTrue(sample["degraded"])
        self.assertEqual(result["aggregate"]["aggregation"], "micro_count_sum")
        self.assertEqual(result["aggregate"]["delta_counts"], sample["delta_counts"])
        self.assertEqual(result["macro"]["aggregation"], "sample_arithmetic_mean")
        self.assertEqual(result["macro"]["initial"]["iou"], 1.0)
        self.assertEqual(result["macro"]["selected"]["iou"], 0.0)
        self.assertEqual(sample["selected"]["precision"], 0.0)
        self.assertEqual(sample["selected"]["recall"], 1.0)
        self.assertEqual(sample["selected"]["f1"], 0.0)
        self.assertFalse(
            BASELINE_SUMMARIZER.is_degraded(
                {"tp": 10, "fp": 10, "fn": 10, "tn": 70},
                {"tp": 20, "fp": 11, "fn": 9, "tn": 60},
            )
        )

    def test_post_sft_summary_reports_only_its_own_selected_state(self) -> None:
        run_dir = Path(self.temporary.name) / "run"
        _write_run(run_dir, samples=(SAMPLES[0],))
        _write_custom_metrics(
            run_dir, {SAMPLES[0]: ((2, 1, 1, 12), (3, 1, 0, 12))}
        )
        result = _standalone_baseline(
            run_dir,
            samples=(SAMPLES[0],),
            evaluation_profile="smoke1",
            evaluation_id="DEV-SMOKE-1-V1",
            model_state="post_sft",
        )

        self.assertEqual(result["model_state"], "post_sft")
        self.assertEqual(
            set(result["aggregate"]) & {"initial", "selected"},
            {"initial", "selected"},
        )
        self.assertNotIn("base_selected", result)
        self.assertNotIn("base_selected", result["aggregate"])
        self.assertNotIn("base_selected", result["per_sample"][SAMPLES[0]])

    def test_metric_first_degradation_uses_count_ties_only_as_tiebreakers(self) -> None:
        cases = (
            (
                "unchanged",
                {"tp": 10, "fp": 5, "fn": 5, "tn": 80},
                {"tp": 10, "fp": 5, "fn": 5, "tn": 80},
                False,
            ),
            (
                "tied_rates_tp_down",
                {"tp": 10, "fp": 5, "fn": 5, "tn": 80},
                {"tp": 8, "fp": 4, "fn": 4, "tn": 84},
                True,
            ),
            (
                "zero_rates_fp_up",
                {"tp": 0, "fp": 0, "fn": 166, "tn": 65370},
                {"tp": 0, "fp": 6518, "fn": 166, "tn": 58852},
                True,
            ),
            (
                "zero_rates_fn_up",
                {"tp": 0, "fp": 0, "fn": 5, "tn": 95},
                {"tp": 0, "fp": 0, "fn": 6, "tn": 94},
                True,
            ),
        )
        for name, initial, selected, expected in cases:
            with self.subTest(name=name):
                self.assertEqual(
                    BASELINE_SUMMARIZER.is_degraded(initial, selected), expected
                )

    def test_empty_prediction_and_v1_zero_iou_fp_regression_are_degraded(self) -> None:
        empty_prediction = self._summarize_smoke(
            initial=(3, 0, 0, 13), selected=(0, 0, 3, 13)
        )
        self.assertEqual(
            empty_prediction["per_sample"][SAMPLES[0]]["selected"]["precision"],
            0.0,
        )
        self.assertEqual(
            empty_prediction["per_sample"][SAMPLES[0]]["selected"]["recall"],
            0.0,
        )
        self.assertEqual(
            empty_prediction["per_sample"][SAMPLES[0]]["selected"]["f1"],
            0.0,
        )
        self.assertTrue(empty_prediction["per_sample"][SAMPLES[0]]["degraded"])

        self.temporary.cleanup()
        self.temporary = tempfile.TemporaryDirectory()
        v1 = self._summarize_smoke(
            initial=(0, 0, 166, 65370), selected=(0, 6518, 166, 58852)
        )
        sample = v1["per_sample"][SAMPLES[0]]
        self.assertEqual(sample["delta_iou"], 0.0)
        self.assertEqual(sample["delta_f1"], 0.0)
        self.assertEqual(sample["delta_counts"], {"tp": 0, "fp": 6518, "fn": 0})
        self.assertTrue(sample["degraded"])

    @unittest.skipUnless(V1_ARCHIVE_RUN.is_dir(), "0801 v1 archive is not present")
    def test_read_only_v1_archive_summary_marks_test_50_8_degraded(self) -> None:
        result = _standalone_baseline(
            V1_ARCHIVE_RUN,
            samples=E2E3_SAMPLES,
            evaluation_profile=E2E3_EVALUATION_PROFILE,
            evaluation_id=E2E3_EVALUATION_ID,
            model_state="post_sft",
        )

        sample = result["per_sample"]["test_50_8"]
        self.assertEqual(result["model_state"], "post_sft")
        self.assertNotIn("base_selected", result)
        self.assertNotIn("base_selected", sample)
        self.assertEqual(sample["initial"]["iou"], 0.0)
        self.assertEqual(sample["selected"]["iou"], 0.0)
        self.assertEqual(sample["delta_counts"], {"tp": 0, "fp": 6518, "fn": 0})
        self.assertTrue(sample["degraded"])
        self.assertEqual(
            sample["candidate_oracle"]["candidates"][0]["classification"],
            "harmful",
        )
        self.assertEqual(
            sample["candidate_oracle"]["candidates"][0]["leaf_class"],
            "false_accept",
        )
        self.assertEqual(sample["candidate_oracle"]["false_accept_count"], 1)

    def test_macro_is_equal_weighted_while_micro_uses_pixel_sums(self) -> None:
        run_dir = Path(self.temporary.name) / "run"
        _write_run(run_dir, samples=E2E3_SAMPLES)
        _write_custom_metrics(
            run_dir,
            {
                E2E3_SAMPLES[0]: ((1, 0, 0, 9), (1, 0, 0, 9)),
                E2E3_SAMPLES[1]: ((0, 9, 1, 0), (0, 9, 1, 0)),
                E2E3_SAMPLES[2]: ((1, 0, 0, 999), (1, 0, 0, 999)),
            },
        )
        result = _standalone_baseline(
            run_dir,
            samples=E2E3_SAMPLES,
            evaluation_profile=E2E3_EVALUATION_PROFILE,
            evaluation_id=E2E3_EVALUATION_ID,
        )

        self.assertEqual(result["macro"]["initial"]["iou"], 0.66666667)
        self.assertNotEqual(
            result["macro"]["initial"]["iou"],
            result["aggregate"]["initial"]["iou"],
        )

    def test_summary_rejects_rollout_sample_reordering(self) -> None:
        run_dir = Path(self.temporary.name) / "run"
        _write_run(run_dir, samples=E2E3_SAMPLES)
        _write_metrics(run_dir, samples_in_profile=E2E3_SAMPLES)
        marker_path = run_dir / "rollout_complete.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["samples"] = list(E2E3_SAMPLES)
        _write_json(marker_path, marker)

        with self.assertRaisesRegex(
            ValueError, "rollout sample identity/canonical order"
        ):
            _standalone_baseline(
                run_dir,
                samples=E2E3_SAMPLES,
                evaluation_profile=E2E3_EVALUATION_PROFILE,
                evaluation_id=E2E3_EVALUATION_ID,
            )

    def test_candidate_leaf_mapping_matches_preregistered_3_by_3_contract(
        self,
    ) -> None:
        cases = (
            ("beneficial", "accepted", "qwen", "true_accept"),
            ("beneficial", "rejected", "qwen", "qwen_false_reject"),
            (
                "beneficial",
                "rejected",
                "pre_qwen_hard_gate",
                "hard_gate_false_reject",
            ),
            ("harmful", "accepted", "qwen", "false_accept"),
            ("harmful", "rejected", "qwen", "true_reject"),
            (
                "harmful",
                "rejected",
                "pre_qwen_hard_gate",
                "hard_gate_correct_reject",
            ),
            ("indeterminate", "accepted", "qwen", "indeterminate_accepted"),
            (
                "indeterminate",
                "rejected",
                "qwen",
                "indeterminate_qwen_rejected",
            ),
            (
                "indeterminate",
                "rejected",
                "pre_qwen_hard_gate",
                "indeterminate_hard_gate_rejected",
            ),
        )
        expected_leaf_classes = {case[3] for case in cases} | {"unavailable"}

        for implementation in (BASELINE_SUMMARIZER, COMPARATOR):
            with self.subTest(implementation=implementation.__name__):
                self.assertEqual(
                    set(implementation.CANDIDATE_LEAF_CLASSES),
                    expected_leaf_classes,
                )
                for classification, decision, decision_path, expected in cases:
                    self.assertEqual(
                        implementation.candidate_leaf_class(
                            classification=classification,
                            oracle_status="available",
                            decision=decision,
                            decision_path=decision_path,
                        ),
                        expected,
                    )
                for decision in ("accepted", "rejected"):
                    self.assertEqual(
                        implementation.candidate_leaf_class(
                            classification="harmful",
                            oracle_status="available",
                            decision=decision,
                            decision_path="unavailable",
                        ),
                        "unavailable",
                    )
                self.assertEqual(
                    implementation.candidate_leaf_class(
                        classification="harmful",
                        oracle_status="available",
                        decision="accepted",
                        decision_path="pre_qwen_hard_gate",
                    ),
                    "unavailable",
                )
                contradictory = {
                    "execution": {
                        "verifier_skipped_by_hard_gate": True,
                        "verifier_evidence": {"type": "staged_qwen_verifier"},
                    }
                }
                self.assertEqual(
                    implementation.explicit_candidate_decision_path(
                        contradictory, hard_gate=True
                    ),
                    "unavailable",
                )

    def test_candidate_oracle_classifies_confusion_and_tracks_accepted_state(self) -> None:
        trajectory = {
            "steps": [
                {
                    "step_index": 0,
                    "offline_metrics": _metric_block(10, 5, 5, 80),
                },
                {
                    "step_index": 1,
                    "candidate_accepted": True,
                    "execution": {"candidate_accepted": True},
                    "offline_metrics": _metric_block(11, 4, 4, 81),
                },
                {
                    "step_index": 2,
                    "candidate_accepted": False,
                    "execution": {"candidate_accepted": False},
                    "offline_metrics": _metric_block(10, 5, 5, 80),
                },
                {
                    "step_index": 3,
                    "candidate_accepted": False,
                    "execution": {
                        "candidate_accepted": False,
                        "verifier_skipped_by_hard_gate": True,
                    },
                    "offline_metrics": _metric_block(12, 3, 3, 82),
                },
                {
                    "step_index": 4,
                    "candidate_accepted": False,
                    "execution": {
                        "candidate_accepted": False,
                        "verifier_skipped_by_hard_gate": False,
                    },
                    "offline_metrics": _metric_block(12, 3, 3, 82),
                },
                {
                    "step_index": 5,
                    "candidate_accepted": False,
                    "execution": {
                        "candidate_accepted": False,
                        "feedback": "rejected before Qwen by a hard gate",
                    },
                    "offline_metrics": _metric_block(11, 4, 4, 81),
                },
                {
                    "step_index": 6,
                    "candidate_accepted": False,
                    "execution": {"candidate_accepted": False},
                },
            ]
        }
        oracle = BASELINE_SUMMARIZER.candidate_oracle_summary(
            trajectory, candidate_count=6, source="fixture"
        )
        candidates = oracle["candidates"]

        self.assertEqual(
            [item["classification"] for item in candidates],
            [
                "beneficial",
                "harmful",
                "beneficial",
                "beneficial",
                "indeterminate",
                "indeterminate",
            ],
        )
        self.assertEqual(candidates[1]["reference_step_index"], 1)
        self.assertFalse(candidates[1]["false_reject"])
        self.assertTrue(candidates[2]["false_reject"])
        self.assertTrue(candidates[2]["hard_gate_false_reject"])
        self.assertFalse(candidates[2]["qwen_false_reject"])
        self.assertEqual(candidates[2]["decision_path"], "pre_qwen_hard_gate")
        self.assertEqual(candidates[2]["leaf_class"], "hard_gate_false_reject")
        self.assertTrue(candidates[3]["qwen_false_reject"])
        self.assertFalse(candidates[3]["hard_gate_false_reject"])
        self.assertEqual(candidates[3]["decision_path"], "qwen")
        self.assertEqual(candidates[3]["leaf_class"], "qwen_false_reject")
        self.assertFalse(candidates[4]["hard_gate_before_qwen"])
        self.assertEqual(candidates[5]["oracle_status"], "unavailable")
        self.assertEqual(
            candidates[5]["unavailable_reason"],
            "candidate_offline_counts_unavailable",
        )
        self.assertEqual(oracle["false_accept_count"], 0)
        self.assertEqual(oracle["qwen_false_reject_count"], 1)
        self.assertEqual(oracle["hard_gate_false_reject_count"], 1)
        self.assertEqual(oracle["all_false_reject_count"], 2)
        self.assertEqual(oracle["false_reject_count"], 2)
        self.assertEqual(
            oracle["false_reject_count_semantics"],
            "alias_of_all_false_reject_count",
        )
        self.assertEqual(
            oracle["all_false_reject_count"],
            oracle["qwen_false_reject_count"]
            + oracle["hard_gate_false_reject_count"],
        )
        self.assertEqual(oracle["hard_gate_before_qwen_count"], 1)
        self.assertEqual(
            oracle["classification_counts"],
            {"beneficial": 3, "harmful": 1, "indeterminate": 2},
        )
        self.assertEqual(
            oracle["oracle_status_counts"], {"available": 5, "unavailable": 1}
        )
        self.assertTrue(
            BASELINE_SUMMARIZER.explicit_hard_gate_before_qwen(
                {
                    "execution": {
                        "verifier_evidence": {
                            "type": "runtime_hard_gate_precheck",
                            "verifier_skipped": True,
                        }
                    }
                }
            )
        )
        self.assertEqual(
            BASELINE_SUMMARIZER.classify_candidate_counts(
                {"tp": 10, "fp": 10, "fn": 10},
                {"tp": 20, "fp": 11, "fn": 9},
            ),
            "beneficial",
        )

    def test_candidate_oracle_counts_false_accept_and_rejects_bad_structure(self) -> None:
        harmful_accept = {
            "steps": [
                {
                    "step_index": 0,
                    "offline_metrics": _metric_block(0, 0, 166, 65370),
                },
                {
                    "step_index": 1,
                    "candidate_accepted": True,
                    "execution": {
                        "candidate_accepted": True,
                        "verifier_skipped_by_hard_gate": False,
                    },
                    "offline_metrics": _metric_block(0, 6518, 166, 58852),
                },
            ]
        }
        oracle = BASELINE_SUMMARIZER.candidate_oracle_summary(
            harmful_accept, candidate_count=1, source="fixture"
        )
        self.assertEqual(oracle["false_accept_count"], 1)

        for name, trajectory, count in (
            ("missing_initial", {"steps": []}, 0),
            (
                "wrong_index",
                {
                    "steps": [
                        {
                            "step_index": 1,
                            "offline_metrics": _metric_block(1, 0, 0, 1),
                        }
                    ]
                },
                0,
            ),
            (
                "count_mismatch",
                {
                    "steps": [
                        {
                            "step_index": 0,
                            "offline_metrics": _metric_block(1, 0, 0, 1),
                        }
                    ]
                },
                1,
            ),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                BASELINE_SUMMARIZER.candidate_oracle_summary(
                    trajectory, candidate_count=count, source="fixture"
                )


class LocalVerifierAnalyzerTest(unittest.TestCase):
    def test_terminal_review_backend_calls_expand_both_rounds_without_double_counting(
        self,
    ) -> None:
        def wrap_terminal_review(evidence: dict[str, Any]) -> None:
            round_one = copy.deepcopy(evidence)
            round_two = copy.deepcopy(evidence)
            for call in round_two["backend_calls"]:
                call["latency_seconds"] = 0.0
                call["raw_response"] = '{"round":2}'
            evidence.clear()
            evidence.update(copy.deepcopy(round_two))
            evidence["decision_mode"] = "gtfree_independent_terminal_review_v1"
            evidence["terminal_review"] = {
                "round_two_invoked": True,
                "round_one_evidence": round_one,
                "round_two_evidence": round_two,
            }

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            _write_run(run_dir, mutate_first=wrap_terminal_review)
            result = _analyze(run_dir)

        first = result["per_sample"][SAMPLES[0]]
        self.assertEqual(first["backend_call_count"], 4)
        self.assertEqual(first["stage_calls"]["audit"], 4)
        self.assertEqual(first["attempt_kinds"]["initial"], 4)

    def test_accepts_valid_v12_1_initial_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            _write_run(run_dir)
            result = _analyze(run_dir)

        self.assertEqual(result["schema_versions"], [TREATMENT_SCHEMA])
        self.assertEqual(result["evaluation_profile"], EVALUATION_PROFILE)
        self.assertEqual(result["evaluation_id"], EVALUATION_ID)
        self.assertEqual(result["totals"]["initial_atomic_audit_records"], 10)
        self.assertEqual(
            result["per_sample"][SAMPLES[0]]["initial_audited_region_ids"],
            ["r0", "r1"],
        )

    def test_accepts_fixed_e2e3_identity_in_both_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            _write_run(run_dir, samples=E2E3_SAMPLES)
            _write_metrics(run_dir, samples_in_profile=E2E3_SAMPLES)
            protocol = _analyze(
                run_dir,
                samples=E2E3_SAMPLES,
                evaluation_profile=E2E3_EVALUATION_PROFILE,
                evaluation_id=E2E3_EVALUATION_ID,
            )
            baseline = BASELINE_SUMMARIZER.summarize(
                argparse.Namespace(
                    run_dir=run_dir,
                    output=run_dir / "unused-baseline.json",
                    evaluation_profile=E2E3_EVALUATION_PROFILE,
                    evaluation_id=E2E3_EVALUATION_ID,
                    model_state="base",
                    job_id="123",
                    experiment_arm="v12_1-compact",
                    protocol_version="v12.1",
                    runtime_commit=_commit("a"),
                    harness_commit=_commit("b"),
                    expected_samples=list(E2E3_SAMPLES),
                )
            )

            marker = json.loads(
                (run_dir / "rollout_complete.json").read_text(encoding="utf-8")
            )
            metrics = json.loads(
                (run_dir / "per_sample_metrics.json").read_text(encoding="utf-8")
            )

        self.assertEqual(marker["samples"], sorted(E2E3_SAMPLES))
        self.assertEqual(list(metrics["samples"]), list(E2E3_SAMPLES))
        for summary in (protocol, baseline):
            self.assertEqual(
                summary["evaluation_profile"], E2E3_EVALUATION_PROFILE
            )
            self.assertEqual(summary["evaluation_id"], E2E3_EVALUATION_ID)
            self.assertEqual(summary["sample_order"], list(E2E3_SAMPLES))
            self.assertEqual(list(summary["per_sample"]), list(E2E3_SAMPLES))

    def test_analyzer_rejects_requested_order_in_legacy_rollout_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            _write_run(run_dir, samples=E2E3_SAMPLES)
            marker_path = run_dir / "rollout_complete.json"
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            marker["samples"] = list(E2E3_SAMPLES)
            _write_json(marker_path, marker)

            with self.assertRaisesRegex(ValueError, "canonical order"):
                _analyze(
                    run_dir,
                    samples=E2E3_SAMPLES,
                    evaluation_profile=E2E3_EVALUATION_PROFILE,
                    evaluation_id=E2E3_EVALUATION_ID,
                )

    def test_e2e3_summary_identity_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            _write_run(run_dir, samples=E2E3_SAMPLES)
            with self.assertRaisesRegex(ValueError, "evaluation ID"):
                _analyze(
                    run_dir,
                    samples=E2E3_SAMPLES,
                    evaluation_profile=E2E3_EVALUATION_PROFILE,
                    evaluation_id=EVALUATION_ID,
                )

    def test_accepts_v11_selected_region_schema_and_variable_coverage(self) -> None:
        for schema in (CONTROL_SCHEMA, TREATMENT_SCHEMA):
            with self.subTest(schema=schema), tempfile.TemporaryDirectory() as temporary:
                run_dir = Path(temporary) / "run"
                _write_run(
                    run_dir,
                    schema=schema,
                    region_ids=("r0", "r1", "r2", "r3"),
                )
                result = _analyze(run_dir)

            self.assertEqual(
                result["totals"]["initial_atomic_audit_records"],
                len(SAMPLES) * 4,
            )
            self.assertEqual(
                result["per_sample"][SAMPLES[0]]["initial_audited_region_ids"],
                ["r0", "r1", "r2", "r3"],
            )

    def test_accepts_staged_programmatic_and_invalid_no_trace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            _write_run(
                run_dir,
                extra_evidence=(
                    _programmatic_evidence(),
                    _programmatic_evidence(
                        "programmatic_alternate_region_after_rollback"
                    ),
                    _programmatic_evidence(
                        "programmatic_no_alternate_region_after_rollback"
                    ),
                    _invalid_evidence(),
                ),
            )
            result = _analyze(run_dir)

        self.assertEqual(result["totals"]["verifier_invalid_steps"], len(SAMPLES))
        self.assertEqual(result["totals"]["validation_error_count"], len(SAMPLES))
        self.assertEqual(result["totals"]["episode_elapsed_seconds"], 35.0)

    def test_rejects_unknown_backend_stage_or_attempt(self) -> None:
        def unknown_stage(evidence: dict[str, Any]) -> None:
            evidence["backend_calls"][0]["stage"] = "audti"

        def unknown_attempt(evidence: dict[str, Any]) -> None:
            evidence["backend_calls"][0]["attempt_kind"] = "retry"

        for name, mutation in (("stage", unknown_stage), ("attempt", unknown_attempt)):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                run_dir = Path(temporary) / "run"
                _write_run(run_dir, mutate_first=mutation)
                with self.assertRaises(ValueError):
                    _analyze(run_dir)

    def test_reports_and_validates_visual_input_scope(self) -> None:
        def rgb_only_repair(evidence: dict[str, Any]) -> None:
            repair_call = copy.deepcopy(evidence["backend_calls"][0])
            repair_call["attempt_kind"] = "repair"
            repair_call["visual_input_scope"] = (
                "authoritative_pair_rgb_only_evidence_repair"
            )
            evidence["backend_calls"].append(repair_call)

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            _write_run(run_dir, mutate_first=rgb_only_repair)
            result = _analyze(run_dir)

        self.assertEqual(
            result["totals"]["visual_input_scopes"],
            {
                "authoritative_pair_rgb_only_evidence_repair": 1,
                "unreported": len(SAMPLES) * 2,
            },
        )

        def unknown_scope(evidence: dict[str, Any]) -> None:
            evidence["backend_calls"][0]["visual_input_scope"] = "rgb-ish"

        def malformed_scope(evidence: dict[str, Any]) -> None:
            evidence["backend_calls"][0]["visual_input_scope"] = []

        def impossible_initial_repair_scope(evidence: dict[str, Any]) -> None:
            evidence["backend_calls"][0]["visual_input_scope"] = (
                "authoritative_pair_rgb_only_evidence_repair"
            )

        def impossible_non_audit_scan_scope(evidence: dict[str, Any]) -> None:
            evidence["backend_calls"][0]["stage"] = "candidate_evidence"
            evidence["backend_calls"][0]["visual_input_scope"] = (
                "authoritative_pair_rgb_only_scan"
            )

        for name, mutation in (
            ("unknown", unknown_scope),
            ("malformed", malformed_scope),
            ("initial_repair", impossible_initial_repair_scope),
            ("non_audit_scan", impossible_non_audit_scan_scope),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                run_dir = Path(temporary) / "run"
                _write_run(run_dir, mutate_first=mutation)
                with self.assertRaisesRegex(ValueError, "visual input scope"):
                    _analyze(run_dir)

    def test_rejects_fewer_audit_calls_than_atomic_records(self) -> None:
        def remove_call(evidence: dict[str, Any]) -> None:
            evidence["backend_calls"].pop()

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            _write_run(run_dir, mutate_first=remove_call)
            with self.assertRaises(ValueError):
                _analyze(run_dir)

    def test_rejects_judgment_and_replan_coverage_gaps(self) -> None:
        def primary_gap(evidence: dict[str, Any]) -> None:
            evidence["stage_trace"]["judgments"].pop()

        def replan_gap(evidence: dict[str, Any]) -> None:
            trace = evidence["stage_trace"]
            trace["replan_evidence"] = [{"region_id": "r2"}]
            trace["replan_audited_region_ids"] = ["r2"]
            trace["replan_diagnoses"] = [
                {"region_id": "r2", "error_type": "none"}
            ]

        for name, mutation in (("primary", primary_gap), ("replan", replan_gap)):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                run_dir = Path(temporary) / "run"
                _write_run(run_dir, mutate_first=mutation)
                with self.assertRaises(ValueError):
                    _analyze(run_dir)

    def test_rejects_episode_elapsed_below_backend_latency(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            _write_run(run_dir)
            episode_path = (
                run_dir / "trajectories" / SAMPLES[0] / "episode_summary.json"
            )
            episode = json.loads(episode_path.read_text(encoding="utf-8"))
            episode["elapsed_seconds"] = 0.5
            _write_json(episode_path, episode)
            with self.assertRaises(ValueError):
                _analyze(run_dir)

    def test_accepts_runtime_hard_gate_precheck(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            _write_run(run_dir, extra_evidence=(_hard_gate_evidence(),))
            result = _analyze(run_dir)

        self.assertEqual(result["totals"]["hard_gate_skipped_steps"], len(SAMPLES))

    def test_rejects_duplicate_or_missing_region_id_coverage(self) -> None:
        def duplicate(evidence: dict[str, Any]) -> None:
            evidence["stage_trace"]["evidence"].append({"region_id": "r0"})

        def missing(evidence: dict[str, Any]) -> None:
            evidence["stage_trace"]["audited_region_ids"] = ["r0"]

        for name, mutation in (("duplicate", duplicate), ("missing", missing)):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                run_dir = Path(temporary) / "run"
                _write_run(run_dir, mutate_first=mutation)
                with self.assertRaises(ValueError):
                    _analyze(run_dir)


class LocalVerifierComparatorTest(unittest.TestCase):
    def test_e2e_submitter_pins_harness_worker_and_clean_pythonpath(self) -> None:
        submitter = (ROOT / "tools/submit_local_verifier_e2e5.sh").read_text(
            encoding="utf-8"
        )
        worker = (ROOT / "tools/run_qwen3vl_base_e2e5.slurm").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            'WORKER="$ROOT/tools/run_qwen3vl_base_e2e5.slurm"',
            submitter,
        )
        self.assertNotIn('WORKER="${WORKER:-', submitter)
        self.assertIn("EXPECTED_WORKER_SHA256", submitter)
        self.assertIn("executed Slurm worker differs", worker)
        self.assertIn(
            'ACTUAL_WORKER_SHA256="$(sha256sum -- "${BASH_SOURCE[0]}"',
            worker,
        )
        self.assertIn(
            '[[ "$ACTUAL_WORKER_SHA256" == "$EXPECTED_WORKER_SHA256" ]]',
            worker,
        )
        self.assertIn("actual_worker_sha256 = sys.argv[12]", worker)
        self.assertIn(
            '"executed_worker_sha256": actual_worker_sha256',
            worker,
        )
        self.assertIn(
            'export PYTHONPATH="$ROOT:$WORKSPACE/Chang-Agent/third_party/OmniOVCD:'
            '$WORKSPACE/Chang-Agent/third_party/SegAgent:'
            '$WORKSPACE/Chang-Agent/third_party/SegAgent/third_party/SimpleClick"',
            worker,
        )
        self.assertNotIn("${PYTHONPATH:+", worker)

    def test_e2e_evaluation_profiles_are_closed_and_persisted(self) -> None:
        submitter = (ROOT / "tools/submit_local_verifier_e2e5.sh").read_text(
            encoding="utf-8"
        )
        worker = (ROOT / "tools/run_qwen3vl_base_e2e5.slurm").read_text(
            encoding="utf-8"
        )
        summarizer = (ROOT / "tools/summarize_qwen3vl_e2e_run.py").read_text(
            encoding="utf-8"
        )
        analyzer = (ROOT / "tools/analyze_staged_verifier_run.py").read_text(
            encoding="utf-8"
        )

        for source in (submitter, worker):
            self.assertIn(
                'EVALUATION_PROFILE="${EVALUATION_PROFILE-e2e5}"', source
            )
            self.assertNotIn(
                'EVALUATION_PROFILE="${EVALUATION_PROFILE:-e2e5}"', source
            )
            self.assertIn('case "$EVALUATION_PROFILE" in', source)
            self.assertIn('EVALUATION_ID="DEV-EXTENDED-5-V1"', source)
            self.assertIn(
                "SAMPLES=(test_20_15 test_78_13 test_85_16 test_1_1 test_50_8)",
                source,
            )
            self.assertIn('EVALUATION_ID="DEV-CORE-3-V1"', source)
            self.assertIn(
                "SAMPLES=(test_20_15 test_1_1 test_50_8)", source
            )
            self.assertIn('EVALUATION_ID="DEV-SMOKE-1-V1"', source)
            self.assertIn("SAMPLES=(test_20_15)", source)
            self.assertIn(
                "EVALUATION_PROFILE must be exactly e2e5, e2e3, or smoke1",
                source,
            )

        self.assertIn(
            ",EVALUATION_PROFILE=$EVALUATION_PROFILE", submitter
        )
        self.assertIn(
            "evaluation_profile=%s\\nevaluation_id=%s\\nsamples=%s\\noutput=%s",
            submitter,
        )
        self.assertIn(
            'if [[ "$EVALUATION_PROFILE" != e2e5 && -z "${OUTPUT:-}" ]]',
            submitter,
        )
        self.assertIn(
            '$EVALUATION_PROFILE requires an explicit non-empty OUTPUT', submitter
        )
        self.assertIn(
            "worker_sha256=%s\\nevaluation_profile=%s\\nevaluation_id=%s\\n"
            "samples=%s\\nkey_result=%s",
            worker,
        )
        self.assertIn(
            '"$ACTUAL_WORKER_SHA256" "$EVALUATION_PROFILE" "$EVALUATION_ID"',
            worker,
        )
        self.assertIn("evaluation_profile = sys.argv[13]", worker)
        self.assertIn("evaluation_id = sys.argv[14]", worker)
        self.assertIn("samples = sys.argv[15:]", worker)
        self.assertIn('"evaluation_profile": evaluation_profile', worker)
        self.assertIn('"evaluation_id": evaluation_id', worker)
        self.assertNotIn('"evaluation_id": "DEV-EXTENDED-5-V1"', worker)
        self.assertEqual(
            worker.count('--evaluation-profile "$EVALUATION_PROFILE"'), 2
        )
        self.assertEqual(worker.count('--evaluation-id "$EVALUATION_ID"'), 2)
        for source in (summarizer, analyzer):
            self.assertIn('"--evaluation-profile"', source)
            self.assertIn('"DEV-CORE-3-V1"', source)
            self.assertIn(
                '("test_20_15", "test_1_1", "test_50_8")', source
            )
            self.assertIn("choices=tuple(EVALUATION_PROFILES)", source)
            self.assertIn(
                'parser.add_argument("--evaluation-id", required=True)', source
            )
            self.assertIn('"evaluation_profile":', source)
            self.assertIn('"evaluation_id":', source)

    def test_non_e2e5_profiles_reject_missing_or_empty_output(self) -> None:
        submitter = ROOT / "tools/submit_local_verifier_e2e5.sh"
        for profile in ("e2e3", "smoke1"):
            for output in (None, ""):
                with self.subTest(profile=profile, output=output):
                    environment = os.environ.copy()
                    environment["EVALUATION_PROFILE"] = profile
                    environment["NODE"] = "unused-before-output-gate"
                    if output is None:
                        environment.pop("OUTPUT", None)
                    else:
                        environment["OUTPUT"] = output
                    result = subprocess.run(
                        ["bash", str(submitter), "compact"],
                        cwd=ROOT,
                        env=environment,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertEqual(result.returncode, 2)
                    self.assertIn(
                        f"{profile} requires an explicit non-empty OUTPUT",
                        result.stderr,
                    )

    def test_e2e_worker_signal_receipt_is_fail_closed_and_atomic(self) -> None:
        worker = (ROOT / "tools/run_qwen3vl_base_e2e5.slurm").read_text(
            encoding="utf-8"
        )

        self.assertIn("TERMINATION_SIGNAL=\"\"", worker)
        self.assertIn("TERMINATION_EXIT_CODE=\"\"", worker)
        self.assertIn("FINISHING=0", worker)
        self.assertIn("FINISHING=1", worker)
        self.assertIn("STATUS=\"cancelled\"", worker)
        self.assertIn('exit "$exit_code"', worker)
        self.assertIn("trap 'handle_termination TERM 143' TERM", worker)
        self.assertIn("trap 'handle_termination INT 130' INT", worker)
        self.assertLess(
            worker.index('if [[ -n "$TERMINATION_SIGNAL" ]]'),
            worker.index('elif [[ "$rc" -eq 0 ]]'),
        )
        self.assertIn(
            'elif [[ "$rc" -eq 0 ]]; then\n    STATUS="success"', worker
        )
        self.assertIn(
            'if [[ "$FINISHING" -eq 1 ]]; then',
            worker,
        )
        self.assertIn(
            'if [[ "$FINISHING" -eq 1 ]]; then\n'
            '    trap - ERR EXIT\n'
            '    set +e',
            worker,
        )
        self.assertIn(
            'finish() {\n'
            "  trap '' TERM INT\n"
            '  trap - ERR EXIT\n'
            '  local rc="$1"\n'
            '  FINISHING=1',
            worker,
        )
        self.assertIn('trap \'finish "$?"\' EXIT', worker)
        self.assertNotIn('finish() {\n  local rc="$?"', worker)
        self.assertIn(
            'write_status_receipt "$exit_code" "$elapsed"', worker
        )
        self.assertIn(
            'if [[ "$STATUS" != "success" ]] && ! write_status_receipt '
            '"$rc" "$elapsed"; then',
            worker,
        )
        self.assertLess(
            worker.index('if [[ "$STATUS" != "success" ]]'),
            worker.index('QQ_SUB="Qwen3-VL verifier ${EVALUATION_PROFILE}'),
        )
        self.assertIn(
            'local temporary_path="${receipt_path}.tmp-${BASHPID}"', worker
        )
        self.assertIn('mv -f -- "$temporary_path" "$receipt_path"', worker)
        self.assertNotIn('>"$LOG_DIR/status-${JOB_ID}.txt"', worker)

    def test_accepts_fixed_control_and_treatment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            control, treatment, output = _write_comparison_fixture(Path(temporary))
            result = _compare(control, treatment, output)
            treatment_baseline = json.loads(
                (treatment / "baseline_summary.json").read_text(encoding="utf-8")
            )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["evaluation_profile"], EVALUATION_PROFILE)
        self.assertEqual(result["evaluation_id"], EVALUATION_ID)
        self.assertTrue(result["comparability"]["all"])
        self.assertTrue(result["no_selected_metric_regression"])
        self.assertIn("selected_f1", result["per_sample"][SAMPLES[0]])
        self.assertEqual(
            treatment_baseline["summary_schema_version"], SUMMARY_SCHEMA_VERSION
        )
        self.assertEqual(treatment_baseline["model_state"], "base")
        self.assertEqual(treatment_baseline["candidate_oracle"]["candidate_count"], 3)
        self.assertEqual(
            treatment_baseline["candidate_oracle"]["classification_counts"],
            {"beneficial": 0, "harmful": 0, "indeterminate": 3},
        )
        self.assertEqual(
            treatment_baseline["candidate_oracle"]["leaf_class_counts"][
                "unavailable"
            ],
            3,
        )

    def test_rejects_baseline_schema_model_macro_delta_and_oracle_tampering(self) -> None:
        def wrong_schema(payload: dict[str, Any]) -> None:
            payload["summary_schema_version"] = "qwen3vl_e2e_summary_v1"

        def wrong_model_state(payload: dict[str, Any]) -> None:
            payload["model_state"] = "post_sft"

        def wrong_delta_counts(payload: dict[str, Any]) -> None:
            payload["per_sample"][SAMPLES[0]]["delta_counts"]["fp"] += 1

        def wrong_macro(payload: dict[str, Any]) -> None:
            payload["macro"]["selected"]["iou"] += 0.01

        def wrong_candidate_oracle(payload: dict[str, Any]) -> None:
            payload["per_sample"][SAMPLES[0]]["candidate_oracle"][
                "classification_counts"
            ]["indeterminate"] += 1

        def wrong_candidate_leaf(payload: dict[str, Any]) -> None:
            payload["candidate_oracle"]["leaf_class_counts"][
                "qwen_false_reject"
            ] += 1

        for name, mutation in (
            ("schema", wrong_schema),
            ("model_state", wrong_model_state),
            ("delta_counts", wrong_delta_counts),
            ("macro", wrong_macro),
            ("candidate_oracle", wrong_candidate_oracle),
            ("candidate_leaf", wrong_candidate_leaf),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                control, treatment, output = _write_comparison_fixture(Path(temporary))
                _mutate_json(treatment / "baseline_summary.json", mutation)
                with self.assertRaises(ValueError):
                    _compare(control, treatment, output)

    def test_comparator_enforces_v1_tied_metric_count_degradation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            control, treatment, output = _write_comparison_fixture(Path(temporary))
            for root, role, job_id in (
                (control, "control", "100"),
                (treatment, "treatment", "101"),
            ):
                _replace_metric_pair(
                    root / "run",
                    SAMPLES[-1],
                    initial=(0, 0, 166, 65370),
                    selected=(0, 6518, 166, 58852),
                )
                manifest = json.loads(
                    (root / "input_manifest.json").read_text(encoding="utf-8")
                )
                _write_json(
                    root / "baseline_summary.json",
                    _baseline(
                        root / "run", role=role, job_id=job_id, manifest=manifest
                    ),
                )

            result = _compare(control, treatment, output)
            self.assertTrue(result["per_sample"][SAMPLES[-1]]["degraded"]["control"])
            self.assertTrue(result["per_sample"][SAMPLES[-1]]["degraded"]["treatment"])

            def hide_degradation(payload: dict[str, Any]) -> None:
                payload["per_sample"][SAMPLES[-1]]["degraded"] = False
                payload["degraded_sample_count"] -= 1

            _mutate_json(treatment / "baseline_summary.json", hide_degradation)
            with self.assertRaises(ValueError):
                _compare(control, treatment, output)

    def test_comparator_rejects_e2e3_as_freeze_evidence(self) -> None:
        manifest = _manifest("treatment")
        manifest.update(
            evaluation_profile=E2E3_EVALUATION_PROFILE,
            evaluation_id=E2E3_EVALUATION_ID,
            sample_order=list(E2E3_SAMPLES),
        )

        self.assertEqual(COMPARATOR.EXPECTED_EVALUATION_PROFILE, "e2e5")
        self.assertEqual(COMPARATOR.EXPECTED_EVALUATION_ID, EVALUATION_ID)
        with self.assertRaisesRegex(ValueError, "evaluation_profile must be 'e2e5'"):
            COMPARATOR.require_manifest(
                manifest,
                name="treatment manifest",
                arm_role="treatment",
            )

    def test_rejects_exchanged_arms(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            control, treatment, output = _write_comparison_fixture(Path(temporary))
            control_manifest = _manifest("treatment")
            treatment_manifest = _manifest("control")
            _write_json(control / "input_manifest.json", control_manifest)
            _write_json(treatment / "input_manifest.json", treatment_manifest)
            with self.assertRaises(ValueError):
                _compare(control, treatment, output)

    def test_rejects_malformed_sha(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            control, treatment, output = _write_comparison_fixture(Path(temporary))
            manifest = _manifest("treatment")
            manifest["files_sha256"][SAMPLES[0]]["A"] = "not-a-sha256"
            _write_json(treatment / "input_manifest.json", manifest)
            with self.assertRaises(ValueError):
                _compare(control, treatment, output)

    def test_rejects_different_slurm_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            control, treatment, output = _write_comparison_fixture(Path(temporary))
            manifest = _manifest("treatment")
            manifest["slurm_node"] = "gpu26"
            _write_json(treatment / "input_manifest.json", manifest)
            with self.assertRaises(ValueError):
                _compare(control, treatment, output)

    def test_rejects_inconsistent_coverage_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            control, treatment, output = _write_comparison_fixture(Path(temporary))
            protocol_path = treatment / "protocol_summary.json"
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            protocol["per_sample"][SAMPLES[0]][
                "initial_environment_region_records"
            ] += 1
            _write_json(protocol_path, protocol)
            with self.assertRaises(ValueError):
                _compare(control, treatment, output)

    def test_rejects_swapped_baseline_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            control, treatment, output = _write_comparison_fixture(Path(temporary))
            control_path = control / "baseline_summary.json"
            treatment_path = treatment / "baseline_summary.json"
            control_summary = json.loads(control_path.read_text(encoding="utf-8"))
            treatment_summary = json.loads(treatment_path.read_text(encoding="utf-8"))
            _write_json(control_path, treatment_summary)
            _write_json(treatment_path, control_summary)
            with self.assertRaises(ValueError):
                _compare(control, treatment, output)

    def test_rejects_summary_run_job_and_identity_mismatches(self) -> None:
        def stale_protocol_run(root: Path) -> None:
            _mutate_json(
                root / "protocol_summary.json",
                lambda payload: payload.__setitem__(
                    "run_dir", str((root / "stale-run").resolve())
                ),
            )

        def stale_baseline_metrics(root: Path) -> None:
            _mutate_json(
                root / "baseline_summary.json",
                lambda payload: payload.__setitem__(
                    "source_metrics", str((root / "stale-metrics.json").resolve())
                ),
            )

        def wrong_job(root: Path) -> None:
            _mutate_json(
                root / "baseline_summary.json",
                lambda payload: payload.__setitem__("job_id", "999"),
            )

        def wrong_protocol_arm(root: Path) -> None:
            _mutate_json(
                root / "protocol_summary.json",
                lambda payload: payload.__setitem__(
                    "experiment_arm", "v11_3-control"
                ),
            )

        for name, mutation in (
            ("protocol_run", stale_protocol_run),
            ("baseline_metrics", stale_baseline_metrics),
            ("job_id", wrong_job),
            ("protocol_arm", wrong_protocol_arm),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                control, treatment, output = _write_comparison_fixture(Path(temporary))
                mutation(treatment)
                with self.assertRaises(ValueError):
                    _compare(control, treatment, output)

    def test_rejects_submission_and_terminal_receipt_mismatches(self) -> None:
        def wrong_submission_job(root: Path) -> None:
            path = root / "submission.txt"
            receipt = _read_receipt(path)
            receipt["job_id"] = "999"
            _write_receipt(path, receipt)

        def failed_status(root: Path) -> None:
            path = root / "logs" / "status-101.txt"
            receipt = _read_receipt(path)
            receipt.update(status="failed", exit_code="1", error="worker failed")
            _write_receipt(path, receipt)

        def wrong_dependency(root: Path) -> None:
            path = root / "submission.txt"
            receipt = _read_receipt(path)
            receipt["dependency"] = "none"
            _write_receipt(path, receipt)

        def stale_status_run(root: Path) -> None:
            path = root / "logs" / "status-101.txt"
            receipt = _read_receipt(path)
            receipt["run_dir"] = str((root / "stale-run").resolve())
            _write_receipt(path, receipt)

        def wrong_submission_worker(root: Path) -> None:
            path = root / "submission.txt"
            receipt = _read_receipt(path)
            receipt["worker_sha256"] = _digest("9")
            _write_receipt(path, receipt)

        def wrong_submission_worker_path(root: Path) -> None:
            path = root / "submission.txt"
            receipt = _read_receipt(path)
            receipt["worker"] = str((root / "old-worker.slurm").resolve())
            _write_receipt(path, receipt)

        def wrong_status_worker(root: Path) -> None:
            path = root / "logs" / "status-101.txt"
            receipt = _read_receipt(path)
            receipt["worker_sha256"] = _digest("9")
            _write_receipt(path, receipt)

        def wrong_manifest_executed_worker(root: Path) -> None:
            path = root / "input_manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["executed_worker_sha256"] = _digest("9")
            _write_json(path, manifest)

        for name, mutation in (
            ("submission_job", wrong_submission_job),
            ("submission_dependency", wrong_dependency),
            ("failed_status", failed_status),
            ("status_run", stale_status_run),
            ("submission_worker", wrong_submission_worker),
            ("submission_worker_path", wrong_submission_worker_path),
            ("status_worker", wrong_status_worker),
            ("manifest_executed_worker", wrong_manifest_executed_worker),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                control, treatment, output = _write_comparison_fixture(Path(temporary))
                mutation(treatment)
                with self.assertRaises(ValueError):
                    _compare(control, treatment, output)

    def test_rejects_evaluation_identity_drift_across_all_artifacts(self) -> None:
        json_artifacts = (
            "input_manifest.json",
            "baseline_summary.json",
            "protocol_summary.json",
        )
        json_drifts: tuple[tuple[str, Any], ...] = (
            ("evaluation_profile", "smoke1"),
            ("evaluation_id", "DEV-SMOKE-1-V1"),
            ("sample_order", ["test_20_15"]),
        )
        for artifact in json_artifacts:
            for field, value in json_drifts:
                with (
                    self.subTest(artifact=artifact, field=field),
                    tempfile.TemporaryDirectory() as temporary,
                ):
                    control, treatment, output = _write_comparison_fixture(
                        Path(temporary)
                    )
                    _mutate_json(
                        treatment / artifact,
                        lambda payload, key=field, drift=value: payload.__setitem__(
                            key, drift
                        ),
                    )
                    with self.assertRaises(ValueError):
                        _compare(control, treatment, output)

        receipt_artifacts = ("submission.txt", "logs/status-101.txt")
        receipt_drifts = (
            ("evaluation_profile", "smoke1"),
            ("evaluation_id", "DEV-SMOKE-1-V1"),
            ("samples", "test_20_15"),
        )
        for artifact in receipt_artifacts:
            for field, value in receipt_drifts:
                with (
                    self.subTest(artifact=artifact, field=field),
                    tempfile.TemporaryDirectory() as temporary,
                ):
                    control, treatment, output = _write_comparison_fixture(
                        Path(temporary)
                    )
                    path = treatment / artifact
                    receipt = _read_receipt(path)
                    receipt[field] = value
                    _write_receipt(path, receipt)
                    with self.assertRaises(ValueError):
                        _compare(control, treatment, output)

    def test_rejects_environment_key_drift_even_when_shared(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            control, treatment, output = _write_comparison_fixture(Path(temporary))
            for root in (control, treatment):
                path = root / "input_manifest.json"
                manifest = json.loads(path.read_text(encoding="utf-8"))
                del manifest["runtime_environment"]["pillow"]
                _write_json(path, manifest)
            with self.assertRaises(ValueError):
                _compare(control, treatment, output)

        with tempfile.TemporaryDirectory() as temporary:
            control, treatment, output = _write_comparison_fixture(Path(temporary))
            path = treatment / "input_manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest["runtime_environment"]["numpy"] = "different-version"
            _write_json(path, manifest)
            with self.assertRaises(ValueError):
                _compare(control, treatment, output)

    def test_rejects_consistent_but_wrong_control_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            control, treatment, output = _write_comparison_fixture(Path(temporary))
            wrong_commit = _commit("a")
            manifest_path = control / "input_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["runtime_commits"]["chang_agent"] = wrong_commit
            manifest["runtime_commits"]["expected_chang_agent"] = wrong_commit
            _write_json(manifest_path, manifest)
            for name in ("protocol_summary.json", "baseline_summary.json"):
                _mutate_json(
                    control / name,
                    lambda payload: payload.__setitem__("runtime_commit", wrong_commit),
                )
            submission_path = control / "submission.txt"
            submission = _read_receipt(submission_path)
            submission["expected_commit"] = wrong_commit
            _write_receipt(submission_path, submission)
            with self.assertRaises(ValueError):
                _compare(control, treatment, output)

    def test_rejects_consistent_compact_commit_that_differs_from_harness(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            control, treatment, output = _write_comparison_fixture(Path(temporary))
            wrong_commit = _commit("b")
            manifest_path = treatment / "input_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["runtime_commits"]["chang_agent"] = wrong_commit
            manifest["runtime_commits"]["expected_chang_agent"] = wrong_commit
            _write_json(manifest_path, manifest)
            for name in ("protocol_summary.json", "baseline_summary.json"):
                _mutate_json(
                    treatment / name,
                    lambda payload: payload.__setitem__("runtime_commit", wrong_commit),
                )
            submission_path = treatment / "submission.txt"
            submission = _read_receipt(submission_path)
            submission["expected_commit"] = wrong_commit
            _write_receipt(submission_path, submission)
            with self.assertRaises(ValueError):
                _compare(control, treatment, output)

    def test_rejects_noncanonical_samples_and_incomplete_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            control, treatment, output = _write_comparison_fixture(Path(temporary))
            for root in (control, treatment):
                _mutate_json(
                    root / "input_manifest.json",
                    lambda payload: payload.__setitem__(
                        "sample_order", list(reversed(SAMPLES))
                    ),
                )
            with self.assertRaises(ValueError):
                _compare(control, treatment, output)

        with tempfile.TemporaryDirectory() as temporary:
            control, treatment, output = _write_comparison_fixture(Path(temporary))
            _mutate_json(
                treatment / "run" / "rollout_complete.json",
                lambda payload: payload.__setitem__(
                    "samples", list(SAMPLES)
                ),
            )
            with self.assertRaises(ValueError):
                _compare(control, treatment, output)

        with tempfile.TemporaryDirectory() as temporary:
            control, treatment, output = _write_comparison_fixture(Path(temporary))
            _mutate_json(
                treatment / "baseline_summary.json",
                lambda payload: payload.__setitem__("status", "partial"),
            )
            with self.assertRaises(ValueError):
                _compare(control, treatment, output)

    def test_rejects_protocol_scalar_and_elapsed_total_tampering(self) -> None:
        for field, increment in (
            ("backend_call_count", 1),
            ("episode_elapsed_seconds", 0.25),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                control, treatment, output = _write_comparison_fixture(Path(temporary))
                path = treatment / "protocol_summary.json"
                protocol = json.loads(path.read_text(encoding="utf-8"))
                protocol["totals"][field] += increment
                _write_json(path, protocol)
                with self.assertRaises(ValueError):
                    _compare(control, treatment, output)

    def test_rejects_protocol_visual_input_scope_tampering(self) -> None:
        for name in ("unknown", "missing_call"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                control, treatment, output = _write_comparison_fixture(
                    Path(temporary)
                )
                path = treatment / "protocol_summary.json"
                protocol = json.loads(path.read_text(encoding="utf-8"))
                sample_scopes = protocol["per_sample"][SAMPLES[0]][
                    "visual_input_scopes"
                ]
                total_scopes = protocol["totals"]["visual_input_scopes"]
                sample_scopes["unreported"] -= 1
                total_scopes["unreported"] -= 1
                if name == "unknown":
                    sample_scopes["invented_scope"] = 1
                    total_scopes["invented_scope"] = 1
                _write_json(path, protocol)
                with self.assertRaises(ValueError):
                    _compare(control, treatment, output)

    def test_rejects_baseline_metric_and_aggregate_tampering(self) -> None:
        def per_sample_metric(payload: dict[str, Any]) -> None:
            sample = payload["per_sample"][SAMPLES[0]]
            replacement = _metric_block(2, 3, 3, 8)
            sample["selected"] = replacement
            sample["delta_iou"] = round(
                replacement["iou"] - sample["initial"]["iou"], 8
            )
            sample["delta_f1"] = round(
                replacement["f1"] - sample["initial"]["f1"], 8
            )
            sample["degraded"] = replacement["iou"] < sample["initial"]["iou"]

        def aggregate_metric(payload: dict[str, Any]) -> None:
            replacement = _metric_block(99, 1, 1, 99)
            payload["aggregate"]["selected"] = replacement
            payload["aggregate"]["delta_iou"] = round(
                replacement["iou"] - payload["aggregate"]["initial"]["iou"], 8
            )
            payload["aggregate"]["delta_f1"] = round(
                replacement["f1"] - payload["aggregate"]["initial"]["f1"], 8
            )

        for name, mutation in (
            ("per_sample", per_sample_metric),
            ("aggregate", aggregate_metric),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                control, treatment, output = _write_comparison_fixture(Path(temporary))
                _mutate_json(treatment / "baseline_summary.json", mutation)
                with self.assertRaises(ValueError):
                    _compare(control, treatment, output)

    def test_rejects_baseline_action_outcome_and_tool_total_tampering(self) -> None:
        def per_sample_action(payload: dict[str, Any]) -> None:
            payload["per_sample"][SAMPLES[0]]["candidate_count"] += 1

        def aggregate_action(payload: dict[str, Any]) -> None:
            payload["actions"]["accepted"] += 1

        def outcome_count(payload: dict[str, Any]) -> None:
            payload["episode_outcome_counts"]["no_executable_action"] += 1

        def invalid_count(payload: dict[str, Any]) -> None:
            payload["invalid_or_no_action_samples"] += 1

        def degraded_count(payload: dict[str, Any]) -> None:
            payload["degraded_sample_count"] += 1

        def tool_total(payload: dict[str, Any]) -> None:
            payload["actions"]["tool_failure_count"] += 1

        for name, mutation in (
            ("per_sample_action", per_sample_action),
            ("aggregate_action", aggregate_action),
            ("outcome_count", outcome_count),
            ("invalid_count", invalid_count),
            ("degraded_count", degraded_count),
            ("tool_total", tool_total),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                control, treatment, output = _write_comparison_fixture(Path(temporary))
                _mutate_json(treatment / "baseline_summary.json", mutation)
                with self.assertRaises(ValueError):
                    _compare(control, treatment, output)

    def test_rejects_each_baseline_action_counter_tampering(self) -> None:
        aggregate_fields = (
            "candidates",
            "accepted",
            "rejected",
            "tool_actions",
            "action_attempts",
            "action_attempt_errors",
            "fallback_actions",
            "successful_tool_actions",
            "tool_failure_count",
            "finish_count",
        )
        per_sample_fields = (
            "candidate_count",
            "accepted_candidate_count",
            "rejected_candidate_count",
            "tool_action_count",
            "action_attempt_error_count",
        )
        with tempfile.TemporaryDirectory() as temporary:
            control, treatment, output = _write_comparison_fixture(Path(temporary))
            baseline_path = treatment / "baseline_summary.json"
            original = json.loads(baseline_path.read_text(encoding="utf-8"))
            for field in aggregate_fields:
                with self.subTest(scope="aggregate", field=field):
                    tampered = copy.deepcopy(original)
                    tampered["actions"][field] += 1
                    _write_json(baseline_path, tampered)
                    with self.assertRaises(ValueError):
                        _compare(control, treatment, output)
            for field in per_sample_fields:
                with self.subTest(scope="per_sample", field=field):
                    tampered = copy.deepcopy(original)
                    tampered["per_sample"][SAMPLES[0]][field] += 1
                    _write_json(baseline_path, tampered)
                    with self.assertRaises(ValueError):
                        _compare(control, treatment, output)
            tampered = copy.deepcopy(original)
            tampered["actions"]["tool_success_rate"] = 0.25
            _write_json(baseline_path, tampered)
            with self.subTest(scope="aggregate", field="tool_success_rate"):
                with self.assertRaises(ValueError):
                    _compare(control, treatment, output)

    def test_rejects_stale_baseline_after_raw_episode_or_tool_change(self) -> None:
        def wrong_episode_identity(root: Path) -> None:
            path = root / "run" / "trajectories" / SAMPLES[0] / "episode_summary.json"
            _mutate_json(path, lambda payload: payload.__setitem__("sample", "other"))

        def changed_tool_status(root: Path) -> None:
            path = (
                root
                / "run"
                / "tool_runs"
                / SAMPLES[2]
                / "box_000"
                / "report.json"
            )
            _mutate_json(path, lambda payload: payload.__setitem__("status", "success"))

        for name, mutation in (
            ("episode_identity", wrong_episode_identity),
            ("tool_status", changed_tool_status),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                control, treatment, output = _write_comparison_fixture(Path(temporary))
                mutation(treatment)
                with self.assertRaises(ValueError):
                    _compare(control, treatment, output)


if __name__ == "__main__":
    unittest.main()
