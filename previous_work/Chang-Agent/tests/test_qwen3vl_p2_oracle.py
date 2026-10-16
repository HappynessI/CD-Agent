from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from change_agent.adapters.omniovcd_adapter import MaskPairProcessor
from change_agent.state import ChangeState
from change_agent.verifier_regions import attach_verifier_regions


ROOT = Path(__file__).resolve().parents[1]


def load_tool(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GENERATOR = load_tool("run_qwen3vl_p2_oracle_test", "tools/run_qwen3vl_p2_oracle.py")
SCORER = load_tool("score_qwen3vl_p2_oracle_test", "tools/score_qwen3vl_p2_oracle.py")


def write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mask_spec(root: Path, relative: str, value: np.ndarray) -> dict[str, object]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(value, dtype=np.uint8))
    return {
        "path": relative,
        "sha256": sha256(path),
        "mask_sha256": SCORER.mask_sha256(value),
        "shape": list(value.shape),
    }


def present_proposal(index: int) -> dict[str, object]:
    x = 10 + index
    return {
        "region_id": f"r{index}",
        "audit_kind": "present",
        "component_area": 1,
        "component_seed_pixels": [x, 10],
        "component_seed_normalized": [100 + index, 100],
        "box_pixels": [x - 1, 9, x + 1, 11],
        "box_normalized": [90 + index, 90, 110 + index, 110],
        "temporal_difference_pixels": 1,
        "component_seed_t1_mask_white": False,
        "component_seed_t2_mask_white": True,
    }


def scan_proposal(index: int) -> dict[str, object]:
    row, column = divmod(index, 3)
    x1, y1 = column * 10, row * 10
    return {
        "region_id": f"s{index}",
        "proposal_version": "qwen3vl_empty_dual_mask_scan_grid_v1",
        "audit_kind": "scan",
        "component_area": 100,
        "component_seed_pixels": [x1 + 4, y1 + 4],
        "component_seed_normalized": [column * 300 + 150, row * 300 + 150],
        "box_pixels": [x1, y1, x1 + 9, y1 + 9],
        "box_normalized": [column * 300, row * 300, column * 300 + 299, row * 300 + 299],
        "temporal_difference_pixels": 0,
        "component_seed_t1_mask_white": False,
        "component_seed_t2_mask_white": False,
    }


class FakePointBackend:
    def __init__(self) -> None:
        self.inputs: list[np.ndarray] = []
        self.last_evidence = {"backend": "fake_point"}

    def refine(self, image, initial_mask, coordinate, is_positive, click_history=()):
        del image, click_history
        self.inputs.append(np.array(initial_mask, copy=True))
        result = np.array(initial_mask, dtype=bool, copy=True)
        x, y = coordinate
        result[y, x] = bool(is_positive)
        return result


class FakeBoxBackend:
    def __init__(self) -> None:
        self.inputs: list[tuple[float, float, float, float]] = []
        self.last_evidence = {"backend": "fake_box"}

    def segment_box(self, image, box_cxcywh_normalized, query):
        del query
        self.inputs.append(tuple(box_cxcywh_normalized))
        return np.zeros(image.shape[:2], dtype=bool)


def minimal_runtime_manifest() -> dict[str, object]:
    return {
        "query": "building",
        "matching": {
            "overlap_threshold": 0.25,
            "mode": "overlap_presence",
            "t12_min_instance_area": 0,
            "cd_min_instance_area": 0,
        },
        "proposal_config": {
            "max_regions": 1,
            "max_delta_regions": 1,
            "min_component_area": 4,
            "padding_ratio": 0.25,
            "empty_mask_scan_grid_size": 3,
        },
        "gate_config": {
            "max_selection_area_delta": 0.25,
            "max_locality_outside_ratio": 0.1,
            "max_target_mask_change_ratio": 0.25,
            "max_component_count_delta": 4,
            "max_point_proposal_change_ratio": 0.25,
            "max_point_proposal_change_pixels": 256,
            "max_point_image_change_ratio": 0.004,
            "point_proposal_change_gate_enabled": True,
        },
        "authorized_negative_point_input_scope": "full_session",
    }


class P2ActionEnumerationTests(unittest.TestCase):
    def test_runtime_commit_requires_a_full_git_sha(self):
        self.assertEqual(GENERATOR.validate_runtime_commit("A" * 40), "a" * 40)
        for invalid in (None, "a" * 39, "g" * 40):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, "40-character Git SHA"
            ):
                GENERATOR.validate_runtime_commit(invalid)

    def test_frozen_inventory_is_27_primary_plus_9_diagnostic(self):
        present = [present_proposal(index) for index in range(9)]
        scans = [scan_proposal(index) for index in range(9)]
        actions = GENERATOR.enumerate_oracle_actions("core3", present + scans)
        primary = [item for item in actions if item["tier"] == "primary"]
        diagnostic = [
            item for item in actions if item["tier"] == "diagnostic_executor_only"
        ]
        self.assertEqual(len(actions), 36)
        self.assertEqual(len(primary), 27)
        self.assertEqual(len(diagnostic), 9)
        self.assertEqual(
            {(item["action"], item["target_view"]) for item in primary},
            {("negative_point", "t2"), ("box", "t1"), ("box", "t2")},
        )
        self.assertTrue(
            all(item["action"] == "positive_point" and item["target_view"] == "t1" for item in diagnostic)
        )

    def test_scan_never_leaks_nominal_positive_point_permission(self):
        actions = GENERATOR.enumerate_oracle_actions("empty", [scan_proposal(0)])
        self.assertEqual([item["action"] for item in actions], ["box", "box"])

    def test_execution_manifest_rejects_gt_references(self):
        with self.assertRaisesRegex(ValueError, "GT reference"):
            GENERATOR.assert_execution_manifest_is_gt_free({"label_path": "/hidden/x.png"})
        with self.assertRaisesRegex(ValueError, "GT path"):
            GENERATOR.assert_execution_manifest_is_gt_free(
                {"some_input": "/dataset/label_cvt/x.png"}
            )


class P2FreshEnvironmentTests(unittest.TestCase):
    def _sample(self) -> GENERATOR.LoadedSample:
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        t1 = np.zeros((16, 16), dtype=bool)
        t2 = np.zeros((16, 16), dtype=bool)
        t2[8, 8] = True
        processor = MaskPairProcessor(
            overlap_threshold=0.25,
            matching_mode="overlap_presence",
            t12_min_instance_area=0,
            cd_min_instance_area=0,
        )
        update = processor.rebuild(t1, t2, {})
        state = ChangeState(
            image,
            image,
            "building",
            t1,
            t2,
            update.change_mask,
            update.t1_instances,
            update.t2_instances,
            update.matching,
            update.evidence,
            0,
        )
        attach_verifier_regions(
            state,
            None,
            max_regions=1,
            max_delta_regions=1,
            min_component_area=4,
            padding_ratio=0.25,
            empty_mask_scan_grid_size=3,
        )
        proposals = tuple(state.evidence["verifier_region_proposals"])
        return GENERATOR.LoadedSample(
            "sample", image, image, t1, t2, "building", proposals, state
        )

    def test_each_candidate_starts_from_the_exact_same_initial_masks(self):
        sample = self._sample()
        action = GENERATOR.enumerate_oracle_actions("sample", sample.proposals)[0]
        point = FakePointBackend()
        box = FakeBoxBackend()
        first, first_verifier = GENERATOR.execute_one_candidate(
            sample, action, point, box, minimal_runtime_manifest()
        )
        second, second_verifier = GENERATOR.execute_one_candidate(
            sample, action, point, box, minimal_runtime_manifest()
        )
        self.assertEqual([int(item.sum()) for item in point.inputs], [1, 1])
        self.assertTrue(np.array_equal(first.state.change_mask, second.state.change_mask))
        self.assertTrue(
            all(entry == 1 for entry in (first_verifier.initial_call_count, second_verifier.initial_call_count))
        )
        # If the real hard gate fires, the candidate stub is not called. Otherwise it
        # catches the gate-pass candidate; either outcome proves Qwen is absent.
        for entry, verifier in ((first, first_verifier), (second, second_verifier)):
            expected = 0 if entry.execution["verifier_skipped_by_hard_gate"] else 1
            self.assertEqual(verifier.candidate_call_count, expected)
            self.assertFalse(verifier.last_evidence.get("qwen_invoked", True))
            self.assertFalse(entry.execution["candidate_accepted"])
            self.assertFalse(
                np.array_equal(
                    entry.state.change_mask, sample.initial_state.change_mask
                )
            )


class P2MetricAndGateTests(unittest.TestCase):
    def test_classification_covers_beneficial_harmful_and_indeterminate(self):
        initial = {"iou": 0.5, "f1": 0.6, "tp": 5, "fp": 3, "fn": 2}
        beneficial = {"iou": 0.6, "f1": 0.7, "tp": 6, "fp": 2, "fn": 1}
        harmful = {"iou": 0.4, "f1": 0.6, "tp": 5, "fp": 4, "fn": 2}
        indeterminate = {"iou": 0.5, "f1": 0.61, "tp": 5, "fp": 3, "fn": 2}
        self.assertEqual(SCORER.classify_candidate(initial, beneficial)[0], "beneficial")
        self.assertEqual(SCORER.classify_candidate(initial, harmful)[0], "harmful")
        self.assertEqual(SCORER.classify_candidate(initial, indeterminate)[0], "indeterminate")

    def test_gate_leaf_denominators_and_wilson_zero_denominator(self):
        self.assertEqual(
            SCORER.gate_leaf("beneficial", True), "hard_gate_false_reject"
        )
        interval = SCORER.wilson_interval(1, 2)
        self.assertEqual(interval["numerator"], 1)
        self.assertEqual(interval["denominator"], 2)
        self.assertLess(interval["wilson95_lower"], 0.5)
        self.assertGreater(interval["wilson95_upper"], 0.5)
        empty = SCORER.wilson_interval(0, 0)
        self.assertIsNone(empty["rate"])
        self.assertIsNone(empty["wilson95_lower"])

    def test_primary_and_diagnostic_confusion_are_summarized_separately(self):
        rows = [
            {
                "tier": "primary",
                "oracle_label": "beneficial",
                "gate_leaf": "hard_gate_false_reject",
                "hard_gate": {"rejected": True},
            },
            {
                "tier": "diagnostic_executor_only",
                "oracle_label": "harmful",
                "gate_leaf": "harmful_gate_pass",
                "hard_gate": {"rejected": False},
            },
        ]
        primary = SCORER.summarize_gate_confusion(
            [item for item in rows if item["tier"] == "primary"]
        )
        diagnostic = SCORER.summarize_gate_confusion(
            [item for item in rows if item["tier"] == "diagnostic_executor_only"]
        )
        self.assertEqual(primary["oracle_counts"]["beneficial"], 1)
        self.assertEqual(primary["oracle_counts"]["harmful"], 0)
        self.assertEqual(diagnostic["oracle_counts"]["harmful"], 1)
        self.assertEqual(
            primary["gate_rates_with_wilson95"]["hard_gate_false_reject_rate"]["denominator"],
            1,
        )


class P2ScoringBoundaryTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
        candidate_root = root / "candidates"
        initial = np.zeros((3, 3), dtype=bool)
        candidate = np.array(initial, copy=True)
        candidate[1, 1] = True
        initial_states = {
            sample: {
                "t1_mask": mask_spec(
                    candidate_root, f"initial_states/{sample}/t1.npy", initial
                ),
                "t2_mask": mask_spec(
                    candidate_root, f"initial_states/{sample}/t2.npy", initial
                ),
                "change_mask": mask_spec(
                    candidate_root, f"initial_states/{sample}/change.npy", initial
                ),
            }
            for sample in SCORER.CORE3_ORDERED_SAMPLES
        }
        inventory = [
            *[("test_20_15", "primary") for _ in range(8)],
            *[("test_20_15", "diagnostic_executor_only") for _ in range(8)],
            ("test_1_1", "primary"),
            ("test_1_1", "diagnostic_executor_only"),
            *[("test_50_8", "primary") for _ in range(18)],
        ]
        candidate_identities = []
        for number, (sample, tier) in enumerate(inventory):
            candidate_id = f"c{number:02d}"
            candidate_specs = {
                "t1": mask_spec(
                    candidate_root,
                    f"candidates/{candidate_id}/masks/t1.npy",
                    initial,
                ),
                "t2": mask_spec(
                    candidate_root,
                    f"candidates/{candidate_id}/masks/t2.npy",
                    candidate,
                ),
                "change": mask_spec(
                    candidate_root,
                    f"candidates/{candidate_id}/masks/change.npy",
                    candidate,
                ),
            }
            state_hashes = {
                name: candidate_specs[name]["mask_sha256"]
                for name in ("t1", "t2", "change")
            }
            record = {
                "candidate_id": candidate_id,
                "sample": sample,
                "tier": tier,
                "region_id": f"r{number}",
                "action": "positive_point",
                "target_view": "t2",
                "candidate_masks": candidate_specs,
                "candidate_state_mask_sha256": state_hashes,
                "hard_gate": {
                    "passed": True,
                    "rejected": False,
                    "rejection_reasons": [],
                    "verifier_skipped_by_hard_gate": False,
                },
                "no_qwen_audit": {
                    "qwen_invoked": False,
                    "initial_stub_calls": 1,
                    "candidate_stub_calls": 1,
                    "decision_path": "eligible_for_qwen_not_evaluated",
                },
                "tool_evidence": {
                    "tool": "simpleclick",
                    "tool_input": {"coordinate": [1, 1]},
                    "tool_result": {
                        "worker_mode": "point",
                        "worker_report": {"status": "success", "mode": "point"},
                    },
                    "verifier_skipped_by_hard_gate": False,
                    "candidate_accepted": False,
                    "verifier_evidence": {
                        "type": "p2_no_qwen_reject_verifier",
                        "qwen_invoked": False,
                    },
                },
            }
            record_path = write_json(
                candidate_root / f"candidates/{candidate_id}/candidate.json", record
            )
            candidate_identities.append(
                {
                    "candidate_id": candidate_id,
                    "sample": sample,
                    "tier": tier,
                    "candidate_state_mask_sha256": state_hashes,
                    "record_path": f"candidates/{candidate_id}/candidate.json",
                    "record_sha256": sha256(record_path),
                }
            )
        execution_hash = "e" * 64
        preregistration_hash = "p" * 64
        runtime_commit = "1" * 40
        index = {
            "schema_id": SCORER.CANDIDATE_INDEX_SCHEMA_ID,
            "execution_manifest": {"path": "/frozen/no-gt.json", "sha256": execution_hash},
            "preregistration_canonical_sha256": preregistration_hash,
            "p2_runtime_commit": runtime_commit,
            "gt_loaded": False,
            "ordered_samples": list(SCORER.CORE3_ORDERED_SAMPLES),
            "counts": dict(SCORER.CORE3_CANDIDATE_COUNTS),
            "initial_states": initial_states,
            "candidates": candidate_identities,
        }
        index_path = write_json(candidate_root / "candidates.json", index)
        marker = {
            "schema_id": SCORER.COMPLETE_SCHEMA_ID,
            "execution_status": "completed",
            "gt_loaded": False,
            "ordered_samples": list(SCORER.CORE3_ORDERED_SAMPLES),
            "counts": index["counts"],
            "execution_manifest": {"sha256": execution_hash},
            "preregistration_canonical_sha256": preregistration_hash,
            "p2_runtime_commit": runtime_commit,
            "candidate_index": {"path": "candidates.json", "sha256": sha256(index_path)},
        }
        marker_path = write_json(
            candidate_root / "candidate_generation_complete.json", marker
        )
        scoring = {
            "schema_id": SCORER.SCHEMA_ID,
            "status": "frozen",
            "ordered_samples": list(SCORER.CORE3_ORDERED_SAMPLES),
            "target_samples": list(SCORER.CORE3_TARGET_SAMPLES),
            "labels": {
                sample: {
                    "path": str(root / "labels" / f"{sample}.png"),
                    "sha256": "f" * 64,
                }
                for sample in SCORER.CORE3_ORDERED_SAMPLES
            },
            "candidate_generation_complete_sha256": sha256(marker_path),
            "candidate_index_sha256": sha256(index_path),
            "execution_manifest_sha256": execution_hash,
            "preregistration_canonical_sha256": preregistration_hash,
            "p2_runtime_commit": runtime_commit,
            "expected_candidate_counts": index["counts"],
            "candidate_flow_counts": {
                "requested": 36,
                "generated": 36,
                "tool_success": 36,
            },
        }
        scoring_path = write_json(root / "scoring.json", scoring)
        return (
            scoring_path,
            candidate_root,
            candidate_root / "candidates/c00/masks/change.npy",
        )

    def _refresh_hash_chain(self, scoring_path: Path, candidate_root: Path) -> None:
        index_path = candidate_root / "candidates.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        for identity in index["candidates"]:
            record_path = candidate_root / identity["record_path"]
            identity["record_sha256"] = sha256(record_path)
        write_json(index_path, index)

        marker_path = candidate_root / "candidate_generation_complete.json"
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["candidate_index"]["sha256"] = sha256(index_path)
        write_json(marker_path, marker)

        scoring = json.loads(scoring_path.read_text(encoding="utf-8"))
        scoring["candidate_index_sha256"] = sha256(index_path)
        scoring["candidate_generation_complete_sha256"] = sha256(marker_path)
        write_json(scoring_path, scoring)

    def _assert_fails_before_gt(
        self, scoring_path: Path, candidate_root: Path, message: str
    ) -> None:
        calls: list[str] = []
        with self.assertRaisesRegex(ValueError, message):
            SCORER.score(
                scoring_path,
                candidate_root,
                label_loader=lambda spec, sample: calls.append(sample),
            )
        self.assertEqual(calls, [])

    def test_label_loader_runs_only_after_every_candidate_hash_is_verified(self):
        with tempfile.TemporaryDirectory() as temporary:
            scoring_path, candidate_root, _ = self._fixture(Path(temporary))
            calls: list[str] = []

            def loader(spec, sample):
                del spec
                calls.append(sample)
                value = np.zeros((3, 3), dtype=bool)
                value[1, 1] = True
                return value

            report = SCORER.score(scoring_path, candidate_root, label_loader=loader)
            self.assertEqual(calls, list(SCORER.CORE3_ORDERED_SAMPLES))
            self.assertEqual(
                report["target_primary_actionability"]["test_1_1"]["decision"],
                "primary_actionable_and_gate_pass",
            )
            self.assertEqual(report["p2_pass_stop_decision_scope"], "primary_only")
            self.assertEqual(report["by_tier"]["primary"]["candidate_count"], 27)
            self.assertEqual(report["action_spec_count"], 36)
            self.assertEqual(report["unique_candidate_state_count"], 3)
            target = report["target_primary_actionability"]["test_1_1"]
            self.assertEqual(target["best_primary_candidate_id"], "c16")
            self.assertEqual(target["best_primary_candidate_oracle_label"], "beneficial")

    def test_candidate_mask_tamper_fails_before_any_label_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            scoring_path, candidate_root, candidate_mask = self._fixture(Path(temporary))
            candidate_mask.write_bytes(candidate_mask.read_bytes() + b"tamper")
            calls: list[str] = []

            def loader(spec, sample):
                del spec
                calls.append(sample)
                return np.zeros((3, 3), dtype=bool)

            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                SCORER.score(scoring_path, candidate_root, label_loader=loader)
            self.assertEqual(calls, [])

    def test_target_keeps_best_primary_even_when_none_is_beneficial(self):
        with tempfile.TemporaryDirectory() as temporary:
            scoring_path, candidate_root, _ = self._fixture(Path(temporary))
            report = SCORER.score(
                scoring_path,
                candidate_root,
                label_loader=lambda spec, sample: np.zeros((3, 3), dtype=bool),
            )
            target = report["target_primary_actionability"]["test_1_1"]
            self.assertEqual(
                target["decision"],
                "no_primary_beneficial_candidate_in_frozen_enumeration",
            )
            self.assertEqual(target["best_primary_candidate_id"], "c16")
            self.assertEqual(target["best_primary_candidate_oracle_label"], "harmful")
            self.assertIsNone(target["best_primary_beneficial_candidate_id"])

    def test_distinct_actions_preserve_count_when_candidate_state_is_identical(self):
        hashes = {"t1": "a", "t2": "b", "change": "c"}
        summary = SCORER.summarize_candidate_states(
            [
                {"sample": "test_1_1", "candidate_state_mask_sha256": hashes},
                {"sample": "test_1_1", "candidate_state_mask_sha256": hashes},
            ]
        )
        self.assertEqual(summary["action_spec_count"], 2)
        self.assertEqual(summary["unique_candidate_state_count"], 1)
        self.assertEqual(summary["candidate_state_duplicate_count"], 1)

    def test_best_beneficial_candidate_is_restricted_to_gate_pass(self):
        initial = np.zeros((2, 2), dtype=bool)
        lower = np.array([[True, False], [False, False]])
        higher = np.array([[True, True], [False, False]])

        def record(candidate_id, rejected, mask):
            mask_hash = SCORER.mask_sha256(mask)
            return {
                "candidate_id": candidate_id,
                "sample": "test_1_1",
                "tier": "primary",
                "region_id": candidate_id,
                "action": "positive_point",
                "target_view": "t2",
                "candidate_state_mask_sha256": {
                    "t1": "a" * 64,
                    "t2": mask_hash,
                    "change": mask_hash,
                },
                "hard_gate": {"passed": not rejected, "rejected": rejected},
                "no_qwen_audit": {"decision_path": "unused_in_direct_score"},
            }

        verified = SCORER.VerifiedGeneration(
            marker={},
            index={},
            initial_masks={name: initial for name in SCORER.CORE3_ORDERED_SAMPLES},
            candidates=(
                (record("higher_but_gated", True, higher), higher),
                (record("lower_gate_pass", False, lower), lower),
            ),
        )
        scoring = {
            "ordered_samples": list(SCORER.CORE3_ORDERED_SAMPLES),
            "labels": {name: {} for name in SCORER.CORE3_ORDERED_SAMPLES},
            "target_samples": ["test_1_1"],
        }
        report = SCORER.score_verified_generation(
            scoring,
            verified,
            label_loader=lambda spec, sample: np.ones((2, 2), dtype=bool),
        )
        target = report["target_primary_actionability"]["test_1_1"]
        self.assertEqual(target["decision"], "primary_actionable_and_gate_pass")
        self.assertEqual(
            target["best_primary_beneficial_candidate_id"], "lower_gate_pass"
        )

    def test_label_shape_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            scoring_path, candidate_root, _ = self._fixture(Path(temporary))
            calls: list[str] = []

            def loader(spec, sample):
                del spec
                calls.append(sample)
                return np.zeros((2, 2), dtype=bool)

            with self.assertRaisesRegex(ValueError, "shapes differ"):
                SCORER.score(scoring_path, candidate_root, label_loader=loader)
            self.assertEqual(calls, [SCORER.CORE3_ORDERED_SAMPLES[0]])

    def test_execution_binding_tamper_fails_before_any_label_access(self):
        for target in ("marker", "index"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                scoring_path, candidate_root, _ = self._fixture(Path(temporary))
                path = candidate_root / (
                    "candidate_generation_complete.json"
                    if target == "marker"
                    else "candidates.json"
                )
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["execution_manifest"]["sha256"] = "x" * 64
                write_json(path, payload)
                self._refresh_hash_chain(scoring_path, candidate_root)
                self._assert_fails_before_gt(
                    scoring_path, candidate_root, "execution manifest binding mismatch"
                )

    def test_preregistration_binding_tamper_fails_before_any_label_access(self):
        for target in ("marker", "index"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                scoring_path, candidate_root, _ = self._fixture(Path(temporary))
                path = candidate_root / (
                    "candidate_generation_complete.json"
                    if target == "marker"
                    else "candidates.json"
                )
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["preregistration_canonical_sha256"] = "x" * 64
                write_json(path, payload)
                self._refresh_hash_chain(scoring_path, candidate_root)
                self._assert_fails_before_gt(
                    scoring_path,
                    candidate_root,
                    "preregistration canonical binding mismatch",
                )

    def test_runtime_commit_binding_tamper_fails_before_any_label_access(self):
        for target in ("marker", "index", "scoring"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                scoring_path, candidate_root, _ = self._fixture(Path(temporary))
                if target == "marker":
                    path = candidate_root / "candidate_generation_complete.json"
                elif target == "index":
                    path = candidate_root / "candidates.json"
                else:
                    path = scoring_path
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["p2_runtime_commit"] = "2" * 40
                write_json(path, payload)
                self._refresh_hash_chain(scoring_path, candidate_root)
                self._assert_fails_before_gt(
                    scoring_path, candidate_root, "P2 runtime commit binding mismatch"
                )

    def test_tool_and_no_qwen_audit_tamper_fails_before_label_access(self):
        for target in ("tool_status", "stub_calls", "decision_path"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                scoring_path, candidate_root, _ = self._fixture(Path(temporary))
                record_path = candidate_root / "candidates/c00/candidate.json"
                record = json.loads(record_path.read_text(encoding="utf-8"))
                if target == "tool_status":
                    record["tool_evidence"]["tool_result"]["worker_report"][
                        "status"
                    ] = "failed"
                    message = "tool success evidence is malformed"
                elif target == "stub_calls":
                    record["no_qwen_audit"]["candidate_stub_calls"] = 0
                    message = "gate/no-Qwen audit is malformed"
                else:
                    record["no_qwen_audit"]["decision_path"] = "pre_qwen_hard_gate"
                    message = "gate/no-Qwen audit is malformed"
                write_json(record_path, record)
                self._refresh_hash_chain(scoring_path, candidate_root)
                self._assert_fails_before_gt(scoring_path, candidate_root, message)

    def test_self_consistent_non_core_counts_fail_before_label_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            scoring_path, candidate_root, _ = self._fixture(Path(temporary))
            wrong = {"total": 35, "primary": 26, "diagnostic_executor_only": 9}
            index_path = candidate_root / "candidates.json"
            index = json.loads(index_path.read_text(encoding="utf-8"))
            index["counts"] = wrong
            write_json(index_path, index)
            marker_path = candidate_root / "candidate_generation_complete.json"
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            marker["counts"] = wrong
            write_json(marker_path, marker)
            scoring = json.loads(scoring_path.read_text(encoding="utf-8"))
            scoring["expected_candidate_counts"] = wrong
            scoring["candidate_flow_counts"] = {
                "requested": 35,
                "generated": 35,
                "tool_success": 35,
            }
            write_json(scoring_path, scoring)
            self._refresh_hash_chain(scoring_path, candidate_root)
            self._assert_fails_before_gt(
                scoring_path, candidate_root, "exactly 27 primary plus 9 diagnostic"
            )

    def test_each_count_binding_mismatch_fails_before_any_label_access(self):
        for target in ("marker", "index", "scoring"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                scoring_path, candidate_root, _ = self._fixture(Path(temporary))
                if target == "marker":
                    path = candidate_root / "candidate_generation_complete.json"
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    payload["counts"]["total"] = 2
                elif target == "index":
                    path = candidate_root / "candidates.json"
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    payload["counts"]["total"] = 2
                else:
                    path = scoring_path
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    payload["expected_candidate_counts"]["total"] = 2
                write_json(path, payload)
                self._refresh_hash_chain(scoring_path, candidate_root)
                self._assert_fails_before_gt(
                    scoring_path, candidate_root, "candidate"
                )

    def test_record_semantic_state_hash_mismatch_fails_before_label_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            scoring_path, candidate_root, _ = self._fixture(Path(temporary))
            record_path = candidate_root / "candidates/c00/candidate.json"
            record = json.loads(record_path.read_text(encoding="utf-8"))
            record["candidate_state_mask_sha256"]["change"] = "x" * 64
            write_json(record_path, record)
            self._refresh_hash_chain(scoring_path, candidate_root)
            self._assert_fails_before_gt(
                scoring_path, candidate_root, "semantic state hash mismatch"
            )

    def test_marker_tamper_fails_before_any_label_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            scoring_path, candidate_root, _ = self._fixture(Path(temporary))
            marker = candidate_root / "candidate_generation_complete.json"
            marker.write_text(marker.read_text(encoding="utf-8") + " ", encoding="utf-8")
            calls: list[str] = []
            with self.assertRaisesRegex(ValueError, "marker SHA256 mismatch"):
                SCORER.score(
                    scoring_path,
                    candidate_root,
                    label_loader=lambda spec, sample: calls.append(sample),
                )
            self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
