from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from change_agent.adapters.staged_verifier import StagedQwenVerifier
from change_agent.verifier_protocol import Diagnosis, EvidenceRecord, StageProtocolError
from tools import build_qwen3vl_p2_preregistration as builder
from tools import validate_qwen3vl_p2_preregistration as validator


ROOT = Path(__file__).resolve().parents[1]
DOC_ROOT = ROOT / "doc/training_pipeline/0803_fourth_try"
PREREG_PATH = DOC_ROOT / "QWEN3VL_P2_ACTIONABILITY_PREREGISTRATION_0803.json"
EXECUTION_PATH = DOC_ROOT / "QWEN3VL_P2_ORACLE_EXECUTION_MANIFEST_0803.json"

EXPECTED_SAMPLE_HASHES = {
    "test_20_15": (
        "<hash omitted>",
        "<hash omitted>",
    ),
    "test_1_1": (
        "<hash omitted>",
        "<hash omitted>",
    ),
    "test_50_8": (
        "<hash omitted>",
        "<hash omitted>",
    ),
}
EXPECTED_GLOBAL_HASHES = {
    "all_proposals_canonical_sha256": "<hash omitted>",
    "all_actions_canonical_sha256": "<hash omitted>",
    "combined_inventory_canonical_sha256": "<hash omitted>",
}


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


class FrozenManifestTestCase(unittest.TestCase):
    def frozen(self) -> tuple[dict[str, object], dict[str, object]]:
        if not PREREG_PATH.is_file() or not EXECUTION_PATH.is_file():
            self.skipTest("the two manifests are materialized only in freeze commit B")
        return load_json(PREREG_PATH), load_json(EXECUTION_PATH)


class P2FrozenEnumerationTests(FrozenManifestTestCase):
    def test_exact_36_action_inventory_and_canonical_hashes(self):
        prereg, _ = self.frozen()
        samples = prereg["samples"]
        actions = [
            action
            for sample_name in validator.SAMPLES
            for action in samples[sample_name]["actions"]
        ]
        proposals = [
            proposal
            for sample_name in validator.SAMPLES
            for proposal in samples[sample_name]["proposals"]
        ]
        self.assertEqual(len(actions), 36)
        self.assertEqual(sum(item["tier"] == "primary" for item in actions), 27)
        self.assertEqual(
            sum(item["tier"] == "diagnostic_executor_only" for item in actions), 9
        )
        self.assertEqual(len({item["candidate_id"] for item in actions}), 36)
        for sample_name, expected in EXPECTED_SAMPLE_HASHES.items():
            sample = samples[sample_name]
            self.assertEqual(
                (sample["proposals_canonical_sha256"], sample["actions_canonical_sha256"]),
                expected,
            )
            self.assertEqual(
                validator.canonical_sha256(sample["proposals"]), expected[0]
            )
            self.assertEqual(validator.canonical_sha256(sample["actions"]), expected[1])
        contract = prereg["action_enumeration_contract"]
        for name, expected in EXPECTED_GLOBAL_HASHES.items():
            self.assertEqual(contract[name], expected)
        self.assertEqual(
            validator.canonical_sha256(proposals),
            EXPECTED_GLOBAL_HASHES["all_proposals_canonical_sha256"],
        )
        self.assertEqual(
            validator.canonical_sha256(actions),
            EXPECTED_GLOBAL_HASHES["all_actions_canonical_sha256"],
        )
        self.assertEqual(
            validator.canonical_sha256({"proposals": proposals, "actions": actions}),
            EXPECTED_GLOBAL_HASHES["combined_inventory_canonical_sha256"],
        )

    def test_seed_proposal_limits_and_source_anchors_are_exact(self):
        prereg, _ = self.frozen()
        protocol = prereg["execution_protocol"]
        self.assertEqual(protocol["seed"], 42)
        self.assertEqual(
            protocol["proposal_config"],
            {
                "max_regions": 1,
                "max_delta_regions": 1,
                "min_component_area": 4,
                "padding_ratio": 0.25,
                "empty_mask_scan_grid_size": 3,
            },
        )
        self.assertEqual(
            prereg["anchors"]["p1_runtime_source_freeze_commit_b4"],
            validator.SOURCE_COMMIT,
        )
        self.assertEqual(
            prereg["anchors"]["p1_acceptance_commit"], validator.ACCEPTANCE_COMMIT
        )
        h0_source = prereg["source_h0_retry3"]
        self.assertEqual(h0_source["job_id"], "53158")
        self.assertEqual(h0_source["state"], "COMPLETED")
        self.assertEqual(h0_source["arm"], "h0")
        self.assertEqual(h0_source["output_root"], str(validator.P1_H0_OUTPUT))
        self.assertEqual(
            h0_source["evaluation_receipt"]["sha256"],
            validator.P1_H0_RECEIPT_SHA256,
        )
        self.assertEqual(
            h0_source["rollout_complete"]["sha256"],
            validator.P1_H0_ROLLOUT_SHA256,
        )
        source = prereg["source_h1_retry3"]
        self.assertEqual(source["job_id"], "53159")
        self.assertEqual(source["state"], "COMPLETED")
        self.assertEqual(source["arm"], "h1")
        self.assertEqual(source["output_root"], str(validator.P1_H1_OUTPUT))
        self.assertEqual(
            source["evaluation_receipt"]["sha256"],
            validator.P1_H1_RECEIPT_SHA256,
        )
        self.assertEqual(
            source["rollout_complete"]["sha256"], validator.P1_H1_ROLLOUT_SHA256
        )
        self.assertEqual(
            source["pair_comparison_receipt"]["sha256"], validator.P1_PAIR_SHA256
        )
        for sample_name in validator.SAMPLES:
            frozen_mask_root = (
                validator.P1_H0_OUTPUT
                / "run/tool_runs"
                / sample_name
                / "sam3_initialization/initialize_000"
            )
            sample = prereg["samples"][sample_name]
            self.assertEqual(
                sample["t1_mask"]["path"], str(frozen_mask_root / "t1_mask.npy")
            )
            self.assertEqual(
                sample["t2_mask"]["path"], str(frozen_mask_root / "t2_mask.npy")
            )

    def test_h1_and_pair_provenance_validates_independently(self):
        prereg, _ = self.frozen()
        validator.validate_h1_pair_source(prereg)

    def test_h1_or_pair_identity_tampering_fails_closed(self):
        prereg, _ = self.frozen()
        wrong_source = copy.deepcopy(prereg)
        wrong_source["source_h1_retry3"]["job_id"] = "53158"
        with self.assertRaisesRegex(validator.ValidationError, "H1 retry3 source"):
            validator.validate_structure(wrong_source)

        original_read = validator.read_json

        def tampered_pair(path: Path):
            payload = original_read(path)
            if path.name == "pair_comparison_receipt.json":
                payload["comparison"]["h1"]["selected_iou"] = 0.1
            return payload

        with (
            mock.patch.object(validator, "read_json", side_effect=tampered_pair),
            self.assertRaisesRegex(validator.ValidationError, "selected_iou"),
        ):
            validator.validate_h1_pair_source(prereg)

        wrong_h0 = copy.deepcopy(prereg)
        wrong_h0["source_h0_retry3"]["job_id"] = "53159"
        with self.assertRaisesRegex(validator.ValidationError, "H0 retry3 source"):
            validator.validate_structure(wrong_h0)

        wrong_mask_source = copy.deepcopy(prereg)
        wrong_mask_source["samples"]["test_1_1"]["t1_mask"]["path"] = str(
            validator.P1_H1_OUTPUT
            / "run/tool_runs/test_1_1/sam3_initialization/initialize_000/t1_mask.npy"
        )
        with self.assertRaisesRegex(validator.ValidationError, "canonical H0 retry3"):
            validator.validate_inventory(wrong_mask_source, ROOT, verify_files=False)

    def test_present_positive_point_is_executor_only_not_planner_reachable(self):
        proposal = {
            "region_id": "r0",
            "audit_kind": "present",
            "component_seed_normalized": [500, 500],
            "box_normalized": [400, 400, 600, 600],
            "component_area": 10,
            "temporal_difference_pixels": 10,
            "component_seed_t1_mask_white": False,
            "component_seed_t2_mask_white": True,
        }
        record = EvidenceRecord.from_proposal(proposal)
        self.assertEqual(record.change_mask_state, "white")
        self.assertIn("positive_point", record.allowed_actions)
        diagnosis = Diagnosis("r0", "false_negative", "t1")
        with self.assertRaisesRegex(
            StageProtocolError, "false_negative requires a black"
        ):
            StagedQwenVerifier._validate_diagnosis(record, diagnosis)

    def test_wrong_acceptance_commit_is_rejected(self):
        prereg, _ = self.frozen()
        tampered = copy.deepcopy(prereg)
        tampered["anchors"]["p1_acceptance_commit"] = "0" * 40
        with self.assertRaisesRegex(validator.ValidationError, "acceptance commit"):
            validator.validate_structure(tampered)

    def test_execution_is_the_exact_gt_free_projection(self):
        prereg, execution = self.frozen()
        validator.validate_execution(prereg, execution)
        validator.assert_gt_free(execution)
        encoded = json.dumps(execution, ensure_ascii=False).lower()
        for forbidden in ("offline_label", "label_cvt", "scoring_manifest", "input_root"):
            self.assertNotIn(forbidden, encoded)

    def test_manifest_tampering_fails_closed(self):
        prereg, execution = self.frozen()
        tampered_prereg = copy.deepcopy(prereg)
        tampered_prereg["samples"]["test_1_1"]["actions"][0]["target_view"] = "t1"
        with self.assertRaisesRegex(validator.ValidationError, "canonical hash"):
            validator.validate_inventory(tampered_prereg, ROOT, verify_files=False)
        tampered_execution = copy.deepcopy(execution)
        tampered_execution["samples"]["test_1_1"]["offline_label"] = {
            "path": "/forbidden/label.png"
        }
        with self.assertRaises(validator.ValidationError):
            validator.validate_execution(prereg, tampered_execution)


class P2GroundTruthBoundaryTests(FrozenManifestTestCase):
    def test_builder_sample_does_not_open_stat_hash_or_resolve_label(self):
        label = builder.DATASET / "label_cvt/test_1_1.png"
        original_open = Path.open
        original_read_bytes = Path.read_bytes
        original_stat = Path.stat
        original_resolve = Path.resolve

        def reject_label(operation, original):
            def guarded(path, *args, **kwargs):
                if str(path) == str(label):
                    raise AssertionError(f"builder attempted label {operation}")
                return original(path, *args, **kwargs)

            return guarded

        with (
            mock.patch.object(Path, "open", reject_label("open", original_open)),
            mock.patch.object(
                Path, "read_bytes", reject_label("read_bytes", original_read_bytes)
            ),
            mock.patch.object(Path, "stat", reject_label("stat", original_stat)),
            mock.patch.object(
                Path, "resolve", reject_label("resolve", original_resolve)
            ),
        ):
            sample = builder._sample(ROOT, "test_1_1")
        self.assertEqual(sample["offline_label"]["path"], str(label))
        self.assertEqual(
            sample["offline_label"]["sha256"], builder.IMAGE_HASHES["test_1_1"]["label"]
        )

    def test_frozen_validator_never_routes_an_offline_label_to_file_io(self):
        prereg, _ = self.frozen()
        visited: list[str] = []
        original_resolve = validator.resolve_identity

        def fake_resolve(value, repo_root, location, *, verify_hash):
            visited.append(location)
            if "offline_label" in location:
                raise AssertionError("frozen validation touched an offline label")
            return original_resolve(
                value, repo_root, location, verify_hash=verify_hash
            )

        with mock.patch.object(validator, "resolve_identity", side_effect=fake_resolve):
            validator.validate_file_bindings(prereg, ROOT, labels=False)
        self.assertTrue(visited)
        self.assertFalse(any("offline_label" in item for item in visited))

    def test_p1_archive_fallback_is_exact_and_preserves_frozen_path(self):
        logical = (
            validator.P1_H0_OUTPUT
            / "run/tool_runs/test_1_1/sam3_initialization/initialize_000/t1_mask.npy"
        )
        materialized = validator.materialize_p1_file(logical, "test mask")
        self.assertEqual(
            materialized,
            (
                validator.P1_H0_ARCHIVE_OUTPUT
                / "run/tool_runs/test_1_1/sam3_initialization/initialize_000/t1_mask.npy"
            ).resolve(strict=True),
        )
        identity = {
            "path": str(logical),
            "sha256": builder.MASK_FILE_HASHES["test_1_1"][0],
        }
        self.assertEqual(
            validator.resolve_identity(identity, ROOT, "test mask", verify_hash=True),
            materialized,
        )

        missing = validator.P1_H0_OUTPUT / "run/not-a-frozen-evidence-file.json"
        with self.assertRaisesRegex(
            validator.ValidationError, "exact archived P1 evidence is unavailable"
        ):
            validator.materialize_p1_file(missing, "missing test evidence")

        tampered = dict(identity)
        tampered["sha256"] = "0" * 64
        with self.assertRaisesRegex(validator.ValidationError, "SHA256 mismatch"):
            validator.resolve_identity(tampered, ROOT, "test mask", verify_hash=True)

    def test_post_generation_validation_precedes_first_label_access(self):
        order: list[str] = []
        prereg = {"samples": {name: {"offline_label": {}} for name in validator.SAMPLES}}
        execution: dict[str, object] = {}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prereg_path = root / "prereg.json"
            execution_path = root / "execution.json"
            candidate_root = root / "generation"
            output = root / "runtime.json"
            candidate_root.mkdir()
            prereg_path.write_text("{}\n", encoding="utf-8")
            execution_path.write_text("{}\n", encoding="utf-8")

            def fake_read(path):
                return prereg if path == prereg_path else execution

            def fake_bindings(_prereg, _root, *, labels):
                order.append("labels" if labels else "nonlabels")

            with (
                mock.patch.object(validator, "read_json", side_effect=fake_read),
                mock.patch.object(validator, "validate_structure"),
                mock.patch.object(validator, "validate_inventory"),
                mock.patch.object(validator, "validate_execution"),
                mock.patch.object(
                    validator,
                    "verify_generation_before_labels",
                    side_effect=lambda *args: (
                        order.append("all_candidates_verified")
                        or (root / "marker.json", root / "index.json")
                    ),
                ),
                mock.patch.object(
                    validator, "validate_file_bindings", side_effect=fake_bindings
                ),
                mock.patch.object(validator, "runtime_scoring_manifest", return_value={}),
                mock.patch.object(validator, "write_new"),
            ):
                rc = validator.main(
                    [
                        "--preregistration",
                        str(prereg_path),
                        "--execution-manifest",
                        str(execution_path),
                        "--repo-root",
                        str(root),
                        "--phase",
                        "post_generation",
                        "--candidate-root",
                        str(candidate_root),
                        "--expected-runtime-commit",
                        "a" * 40,
                        "--runtime-scoring-output",
                        str(output),
                    ]
                )
        self.assertEqual(rc, 0)
        self.assertLess(order.index("all_candidates_verified"), order.index("labels"))


class P2ABAndSubmissionSourceTests(unittest.TestCase):
    def test_a_b_parent_and_two_file_only_assertions_are_present(self):
        submitter = (ROOT / "tools/submit_qwen3vl_p2_oracle.sh").read_text(
            encoding="utf-8"
        )
        worker = (ROOT / "tools/run_qwen3vl_p2_contract_validation.slurm").read_text(
            encoding="utf-8"
        )
        helper = (ROOT / "tools/verify_qwen3vl_p2_release.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("rev-parse HEAD^", submitter)
        self.assertIn("rev-list", helper)
        self.assertIn("--name-status", helper)
        self.assertIn('expected = tuple(f"A\\t{path}"', helper)
        for source in (submitter, worker, helper):
            self.assertIn("QWEN3VL_P2_ACTIONABILITY_PREREGISTRATION_0803.json", source)
            self.assertIn("QWEN3VL_P2_ORACLE_EXECUTION_MANIFEST_0803.json", source)

    def test_default_submitter_can_only_submit_zero_gpu_validation(self):
        source = (ROOT / "tools/submit_qwen3vl_p2_oracle.sh").read_text(
            encoding="utf-8"
        )
        default_mode = source.index('MODE="validation"')
        validation_branch = source.index('if [[ "$MODE" == "validation" ]]')
        formal_guard = source.index('[[ "$CONTRACT_JOB_ARG" =~ ^[0-9]+$ ]]')
        gpu_submission = source.index("--gres=gpu:1")
        self.assertLess(default_mode, validation_branch)
        self.assertLess(validation_branch, formal_guard)
        self.assertLess(formal_guard, gpu_submission)
        self.assertIn('[[ "$1" == "--formal" ]]', source)

    def test_formal_chain_is_afterok_and_manifest_staging(self):
        submitter = (ROOT / "tools/submit_qwen3vl_p2_oracle.sh").read_text(
            encoding="utf-8"
        )
        materializer = (ROOT / "tools/run_qwen3vl_p2_manifest_build.slurm").read_text(
            encoding="utf-8"
        )
        self.assertIn('--dependency="afterok:$GENERATION_JOB"', submitter)
        self.assertIn("MANIFEST_ROOT_RAW", materializer)
        self.assertIn("STAGING_ROOT", materializer)
        self.assertIn("flock -n", materializer)

    def test_release_jobs_do_not_export_caller_environment_or_use_assert_gates(self):
        paths = [
            ROOT / "tools/submit_qwen3vl_p2_oracle.sh",
            ROOT / "tools/submit_qwen3vl_p2_manifest_build.sh",
            ROOT / "tools/run_qwen3vl_p2_manifest_build.slurm",
            ROOT / "tools/run_qwen3vl_p2_contract_validation.slurm",
            ROOT / "tools/run_qwen3vl_p2_oracle_generation.slurm",
            ROOT / "tools/run_qwen3vl_p2_oracle_scoring.slurm",
            ROOT / "tools/verify_qwen3vl_p2_release.py",
        ]
        sources = [path.read_text(encoding="utf-8") for path in paths]
        self.assertIn("--export=NIL", sources[0])
        self.assertIn("--export=NIL", sources[1])
        for source in sources:
            self.assertNotIn("--export=ALL", source)
            self.assertNotRegex(source, r"(?m)^\s*assert\s+")


if __name__ == "__main__":
    unittest.main()
