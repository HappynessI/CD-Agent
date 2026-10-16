from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FORMAL_ATTEMPT = Path(
    "/guisongxia01/pangchao/wangyihan/Chang-Agent/outputs/"
    "CA_0803(1)-qwen3vl-p2-oracle-actionability/attempt-0001"
)
RESULT = ROOT / (
    "doc/training_pipeline/0803_fourth_try/"
    "QWEN3VL_P2_ACTIONABILITY_RESULT_0803.json"
)
ACCEPTANCE = ROOT / (
    "doc/training_pipeline/0803_fourth_try/"
    "QWEN3VL_P2_ACTIONABILITY_ACCEPTANCE_0803.json"
)


def load_tool():
    path = ROOT / "tools/validate_qwen3vl_p2_result.py"
    spec = importlib.util.spec_from_file_location("validate_qwen3vl_p2_result_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VALIDATOR = load_tool()


class StrictJsonTests(unittest.TestCase):
    def test_duplicate_key_and_nonfinite_numbers_are_rejected(self):
        cases = (
            ('{"schema_id":"x","schema_id":"y"}\n', "duplicate JSON key"),
            ('{"value":NaN}\n', "non-finite JSON constant"),
            ('{"value":1e999}\n', "non-finite JSON number"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "value.json"
            for payload, message in cases:
                with self.subTest(payload=payload):
                    path.write_text(payload, encoding="utf-8")
                    with self.assertRaisesRegex(VALIDATOR.ValidationError, message):
                        VALIDATOR.read_json(path)

    def test_binding_rejects_hash_and_size_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.txt"
            path.write_text("evidence\n", encoding="utf-8")
            valid = {
                "path": str(path.resolve()),
                "sha256": VALIDATOR.sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            self.assertEqual(VALIDATOR.verify_binding(valid, path, "artifact"), path)
            for key, value in (("sha256", "0" * 64), ("size_bytes", path.stat().st_size + 1)):
                tampered = dict(valid)
                tampered[key] = value
                with self.subTest(key=key), self.assertRaisesRegex(
                    VALIDATOR.ValidationError, "differs"
                ):
                    VALIDATOR.verify_binding(tampered, path, "artifact")

    def test_removed_implementation_worktree_uses_only_exact_commit_a_blobs(self):
        historical_path = (
            VALIDATOR.REMOVED_IMPLEMENTATION_A_WORKTREE
            / "tools/build_qwen3vl_p2_preregistration.py"
        )
        binding = {
            "path": str(historical_path),
            "sha256": "<hash omitted>",
            "size_bytes": 27663,
        }
        self.assertEqual(
            VALIDATOR.verify_binding(binding, None, "historical builder"),
            historical_path,
        )

        for key, value in (("sha256", "0" * 64), ("size_bytes", 27664)):
            tampered = dict(binding)
            tampered[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(
                VALIDATOR.ValidationError, "differs"
            ):
                VALIDATOR.verify_binding(tampered, None, "historical builder")

        basename_only = dict(binding)
        basename_only["path"] = str(
            VALIDATOR.REMOVED_IMPLEMENTATION_A_WORKTREE.parent
            / "another-worktree/tools/build_qwen3vl_p2_preregistration.py"
        )
        with self.assertRaisesRegex(VALIDATOR.ValidationError, "does not exist"):
            VALIDATOR.verify_binding(basename_only, None, "wrong worktree builder")

    def test_exact_commit_blob_rejects_commit_and_path_tampering(self):
        payload = VALIDATOR.exact_commit_blob(
            ROOT,
            VALIDATOR.IMPLEMENTATION_A,
            "tools/build_qwen3vl_p2_preregistration.py",
            "historical builder",
        )
        self.assertEqual(
            VALIDATOR.hashlib.sha256(payload).hexdigest(),
            "<hash omitted>",
        )
        self.assertEqual(
            len(payload),
            27663,
        )
        with self.assertRaisesRegex(VALIDATOR.ValidationError, "commit object"):
            VALIDATOR.exact_commit_blob(
                ROOT,
                "0" * 40,
                "tools/build_qwen3vl_p2_preregistration.py",
                "wrong commit",
            )
        with self.assertRaisesRegex(VALIDATOR.ValidationError, "relative path"):
            VALIDATOR.exact_commit_blob(
                ROOT,
                VALIDATOR.IMPLEMENTATION_A,
                "../tools/build_qwen3vl_p2_preregistration.py",
                "escaping path",
            )

    def test_fixed_acceptance_parent_descendant_and_blobs(self):
        parents = VALIDATOR.run_git(
            ROOT,
            "rev-list",
            "--parents",
            "-n",
            "1",
            VALIDATOR.ACCEPTANCE_D,
        ).stdout.split()
        self.assertEqual(
            parents,
            [VALIDATOR.ACCEPTANCE_D, VALIDATOR.ACCEPTANCE_PARENT_C],
        )
        head = VALIDATOR.run_git(ROOT, "rev-parse", "HEAD").stdout.strip()
        descendant = VALIDATOR.run_git(
            ROOT,
            "merge-base",
            "--is-ancestor",
            VALIDATOR.ACCEPTANCE_D,
            head,
            check=False,
        )
        self.assertEqual(descendant.returncode, 0)
        for relative, (expected_sha256, expected_size) in (
            VALIDATOR.ACCEPTANCE_D_BLOBS.items()
        ):
            with self.subTest(relative=str(relative)):
                payload = VALIDATOR.exact_commit_blob(
                    ROOT,
                    VALIDATOR.ACCEPTANCE_D,
                    relative.as_posix(),
                    "fixed acceptance blob",
                )
                self.assertEqual(len(payload), expected_size)
                self.assertEqual(
                    VALIDATOR.hashlib.sha256(payload).hexdigest(),
                    expected_sha256,
                )
                current = ROOT / relative
                self.assertEqual(current.stat().st_size, expected_size)
                self.assertEqual(
                    VALIDATOR.sha256_file(current), expected_sha256
                )

    def test_acceptance_lineage_rejects_a_nonfixed_parent(self):
        with self.assertRaisesRegex(
            VALIDATOR.ValidationError, "fixed P2 acceptance parent C"
        ):
            VALIDATOR.verify_git_lineage(ROOT, "0" * 40)


class IndependentMetricTests(unittest.TestCase):
    def test_metrics_and_classification_do_not_depend_on_the_scorer(self):
        reference = np.array([[1, 0], [0, 0]], dtype=bool)
        initial = np.zeros((2, 2), dtype=bool)
        candidate = np.array([[1, 1], [0, 0]], dtype=bool)
        initial_metrics = VALIDATOR.metrics(initial, reference)
        candidate_metrics = VALIDATOR.metrics(candidate, reference)
        label, deltas = VALIDATOR.classify_candidate(
            initial_metrics, candidate_metrics, VALIDATOR.COMPARISON_EPSILON
        )
        self.assertEqual(label, "beneficial")
        self.assertEqual((deltas["tp"], deltas["fp"], deltas["fn"]), (1, 1, -1))
        self.assertEqual(VALIDATOR.gate_leaf(label, False), "beneficial_gate_pass")

    def test_zero_intersection_fp_reduction_is_indeterminate(self):
        initial = {"iou": 0.0, "f1": 0.0, "tp": 0, "fp": 626, "fn": 0}
        candidate = {"iou": 0.0, "f1": 0.0, "tp": 0, "fp": 566, "fn": 0}
        label, deltas = VALIDATOR.classify_candidate(
            initial, candidate, VALIDATOR.COMPARISON_EPSILON
        )
        self.assertEqual(label, "indeterminate")
        self.assertEqual(deltas["fp"], -60)


@unittest.skipUnless(RESULT.is_file() and FORMAL_ATTEMPT.is_dir(), "canonical P2 artifacts unavailable")
class CanonicalResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = VALIDATOR.read_json(RESULT, "canonical result")

    def test_result_and_all_formal_evidence_validate_in_commit_context(self):
        arguments = {}
        expected_mode = "candidate"
        head = VALIDATOR.run_git(ROOT, "rev-parse", "HEAD").stdout.strip()
        if ACCEPTANCE.is_file():
            acceptance = VALIDATOR.read_json(ACCEPTANCE, "canonical acceptance")
            closeout = acceptance.get("candidate_closeout")
            if not isinstance(closeout, dict):
                raise AssertionError("canonical acceptance lacks candidate_closeout")
            receipt = closeout.get("receipt")
            if not isinstance(receipt, dict) or not isinstance(receipt.get("path"), str):
                raise AssertionError("canonical acceptance lacks candidate receipt path")
            validated_commit = closeout.get("validated_commit")
            if not isinstance(validated_commit, str):
                raise AssertionError("canonical acceptance lacks validated commit")
            self.assertEqual(
                validated_commit, VALIDATOR.ACCEPTANCE_PARENT_C
            )
            arguments = {
                "parent_closeout_receipt": Path(receipt["path"]),
                "expected_parent_commit": VALIDATOR.ACCEPTANCE_PARENT_C,
            }
            expected_mode = "acceptance"
            acceptance_parents = VALIDATOR.run_git(
                ROOT,
                "rev-list",
                "--parents",
                "-n",
                "1",
                VALIDATOR.ACCEPTANCE_D,
            ).stdout.split()
            self.assertEqual(
                acceptance_parents,
                [VALIDATOR.ACCEPTANCE_D, VALIDATOR.ACCEPTANCE_PARENT_C],
            )
            descendant = VALIDATOR.run_git(
                ROOT,
                "merge-base",
                "--is-ancestor",
                VALIDATOR.ACCEPTANCE_D,
                head,
                check=False,
            )
            self.assertEqual(descendant.returncode, 0)
        elif head == VALIDATOR.FREEZE_B:
            self.skipTest(
                "precommit registry preparation runs from dirty freeze B; "
                "exact candidate validation starts after C exists"
            )
        else:
            parent = VALIDATOR.run_git(ROOT, "rev-parse", "HEAD^").stdout.strip()
            self.assertEqual(parent, VALIDATOR.FREEZE_B)
        summary = VALIDATOR.validate(RESULT, ROOT, FORMAL_ATTEMPT, **arguments)
        self.assertEqual(summary["mode"], expected_mode)
        self.assertEqual(summary["candidate_count"], 36)
        self.assertEqual(summary["unique_candidate_state_count"], 36)
        self.assertFalse(summary["proceed"])

    def test_result_count_and_gate_tampering_is_rejected(self):
        mutations = (
            ("candidate count", lambda value: value["oracle_scoring"].__setitem__("candidate_count", 35)),
            ("duplicate states", lambda value: value["oracle_scoring"].__setitem__("unique_candidate_state_count", 35)),
            ("tier count", lambda value: value["candidate_generation"].__setitem__("primary", 26)),
            ("Qwen", lambda value: value["oracle_scoring"].__setitem__("qwen_invoked", True)),
            ("proceed gate", lambda value: value["decision"].__setitem__("proceed_to_c_iso_and_t_traj_data_construction", True)),
            ("closeout preclaim", lambda value: value.__setitem__("closeout_status", "accepted")),
        )
        for name, mutate in mutations:
            value = copy.deepcopy(self.result)
            mutate(value)
            with self.subTest(name=name), self.assertRaises(VALIDATOR.ValidationError):
                VALIDATOR.verify_result_claims(value)

    def test_target_and_scope_tampering_is_rejected(self):
        value = copy.deepcopy(self.result)
        value["target_results"]["test_1_1"]["primary_beneficial_count"] = 1
        initial = {
            sample: dict(self.result["target_results"][sample]["initial_metrics"])
            for sample in VALIDATOR.TARGET_SAMPLES
        }
        targets = {
            "test_1_1": {
                "primary_candidate_count": 1,
                "primary_beneficial_count": 0,
                "primary_beneficial_gate_pass_count": 0,
                "decision": "no_primary_beneficial_candidate_in_frozen_enumeration",
                "best_primary_candidate_id": None,
                "best_primary_candidate_oracle_label": None,
                "best_primary_candidate_gate_passed": None,
                "best_primary_candidate_deltas": None,
            },
            "test_50_8": {
                "primary_candidate_count": 18,
                "primary_beneficial_count": 2,
                "primary_beneficial_gate_pass_count": 1,
                "decision": "primary_actionable_and_gate_pass",
                "best_primary_beneficial_candidate_id": None,
                "best_primary_beneficial_deltas": None,
            },
        }
        with self.assertRaisesRegex(VALIDATOR.ValidationError, "primary_beneficial_count"):
            VALIDATOR.verify_result_targets(value, initial, targets, [])

    def test_acceptance_arguments_must_be_paired(self):
        with self.assertRaisesRegex(VALIDATOR.ValidationError, "supplied together"):
            VALIDATOR.validate(
                RESULT,
                ROOT,
                FORMAL_ATTEMPT,
                expected_parent_commit="a" * 40,
            )


class ParentCloseoutShapeTests(unittest.TestCase):
    def test_parent_receipt_must_be_candidate_mode(self):
        receipt = {
            "schema_id": "qwen3vl_p2_closeout_validation_receipt_v1",
            "status": "complete",
            "validation_mode": "acceptance",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "closeout_validation_receipt.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "inventory differs"):
                VALIDATOR.verify_parent_closeout_receipt(
                    path,
                    "a" * 40,
                    path,
                    Path(directory),
                    Path(directory),
                )

    def test_acceptance_record_rejects_a_self_reference_field(self):
        acceptance = {
            "schema_id": "qwen3vl_p2_actionability_acceptance_v1",
            "status": "accepted",
            "scientific_decision_unchanged": True,
            "proceed_to_c_iso_and_t_traj_data_construction": False,
            "candidate_closeout": {},
            "next_boundary": "stop_before_c_iso_and_t_traj_data_construction",
            "acceptance_commit": "a" * 40,
        }
        self.assertNotEqual(
            set(acceptance),
            {
                "schema_id",
                "status",
                "scientific_decision_unchanged",
                "proceed_to_c_iso_and_t_traj_data_construction",
                "candidate_closeout",
                "next_boundary",
            },
        )

    def test_closeout_checks_cover_the_frozen_execution_surface(self):
        self.assertIs(
            VALIDATOR.CLOSEOUT_CHECKS["frozen_execution_surface_unchanged"], True
        )

    def test_acceptance_schema_requires_unchanged_science_and_candidate_binding(self):
        required_top_level = {
            "schema_id",
            "status",
            "scientific_decision_unchanged",
            "proceed_to_c_iso_and_t_traj_data_construction",
            "candidate_closeout",
            "next_boundary",
        }
        required_candidate = {
            "validated_commit",
            "validation_mode",
            "job_id",
            "receipt",
        }
        valid_shape = {
            "schema_id": "qwen3vl_p2_actionability_acceptance_v1",
            "status": "accepted",
            "scientific_decision_unchanged": True,
            "proceed_to_c_iso_and_t_traj_data_construction": False,
            "candidate_closeout": {
                "validated_commit": "a" * 40,
                "validation_mode": "candidate",
                "job_id": "1",
                "receipt": {},
            },
            "next_boundary": "stop_before_c_iso_and_t_traj_data_construction",
        }
        self.assertEqual(set(valid_shape), required_top_level)
        self.assertEqual(set(valid_shape["candidate_closeout"]), required_candidate)
        self.assertTrue(valid_shape["scientific_decision_unchanged"])
        self.assertFalse(
            valid_shape["proceed_to_c_iso_and_t_traj_data_construction"]
        )
        self.assertEqual(
            valid_shape["candidate_closeout"]["validation_mode"], "candidate"
        )


if __name__ == "__main__":
    unittest.main()
