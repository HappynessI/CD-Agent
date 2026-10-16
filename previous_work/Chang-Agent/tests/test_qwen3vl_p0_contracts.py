import hashlib
import json
import math
import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
P0_DOC_ROOT = REPO_ROOT / "doc" / "training_pipeline" / "0802_third_try"
PREREGISTRATION_PATH = P0_DOC_ROOT / "QWEN3VL_H0_H1_PREREGISTRATION_0802.json"
INPUT_MANIFEST_PATH = P0_DOC_ROOT / "QWEN3VL_EVALUATION_INPUT_MANIFEST_0802.json"
P0_R1_AMENDMENT_PATH = P0_DOC_ROOT / "QWEN3VL_P0_VALIDATION_AMENDMENT_0802.json"
P0_R2_AMENDMENT_PATH = (
    P0_DOC_ROOT / "QWEN3VL_P0_VALIDATION_AMENDMENT_R2_0802.json"
)
P0_R2_AMENDMENT_CANONICAL_SHA256 = (
    "<hash omitted>"
)
RECONSTRUCTED_CONFIG_PATH = (
    P0_DOC_ROOT / "QWEN3VL_V2_RECONSTRUCTED_RESOLVED_CONFIG_0802.json"
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")
GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
CA0802_ARCHIVE_ROOT = REPO_ROOT / "outputs" / "CA_0802_archive"
P0_HISTORICAL_OUTPUT_BASENAMES = frozenset(
    {
        "CA_0802(1)-qwen3vl-p0-contract-validation",
        "CA_0802(1R1)-qwen3vl-p0-contract-validation",
    }
)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise AssertionError(f"expected a JSON object in {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(payload: dict) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _resolve_p0_historical_output(path_value: str) -> Path:
    """Resolve a frozen P0 output, allowing only its exact archive relocation."""

    direct = Path(path_value)
    if direct.exists():
        return direct.resolve(strict=True)

    if direct.name not in P0_HISTORICAL_OUTPUT_BASENAMES:
        raise FileNotFoundError(
            f"frozen P0 output does not exist and is not archive-eligible: {direct}"
        )

    archive_root = CA0802_ARCHIVE_ROOT.resolve(strict=True)
    archived = (archive_root / direct.name).resolve(strict=True)
    if not archived.is_relative_to(archive_root) or not archived.is_dir():
        raise AssertionError(
            f"archived P0 output is not a contained directory: {archived}"
        )
    return archived


def _is_degraded(contract: dict, before: dict, after: dict) -> bool:
    """Independently exercise every declarative degradation predicate."""

    epsilon = contract["comparison_epsilon"]
    delta_iou = after["iou"] - before["iou"]
    delta_f1 = after["f1"] - before["f1"]
    metric_tie = abs(delta_iou) <= epsilon and abs(delta_f1) <= epsilon
    predicates = {
        "delta_iou_below_negative_epsilon": delta_iou < -epsilon,
        "delta_f1_below_negative_epsilon": delta_f1 < -epsilon,
        "metric_tie_and_tp_decreases": metric_tie and after["tp"] < before["tp"],
        "metric_tie_and_fp_increases": metric_tie and after["fp"] > before["fp"],
        "metric_tie_and_fn_increases": metric_tie and after["fn"] > before["fn"],
    }
    unknown = set(contract["degraded_if"]) - set(predicates)
    if unknown:
        raise AssertionError(f"unknown degradation predicates: {sorted(unknown)}")
    return any(predicates[name] for name in contract["degraded_if"])


class Qwen3VLP0PreregistrationContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = _load_json(PREREGISTRATION_PATH)
        cls.input_manifest = _load_json(INPUT_MANIFEST_PATH)
        cls.p0_r1_amendment = _load_json(P0_R1_AMENDMENT_PATH)
        cls.p0_r2_amendment = _load_json(P0_R2_AMENDMENT_PATH)

    def test_design_status_is_separate_from_acceptance_and_execution(self):
        payload = self.payload
        self.assertEqual(payload["schema_id"], "qwen3vl_harness_preregistration_v1")
        self.assertEqual(payload["design_status"], "frozen")
        self.assertEqual(payload["execution_readiness"], "not_execution_ready")
        self.assertIn("never backfill this manifest", payload["state_interpretation"])
        acceptance = payload["p0_acceptance_receipt"]
        self.assertEqual(acceptance["status_at_design_freeze"], "pending")
        self.assertIsNone(acceptance["sha256_at_design_freeze"])
        self.assertTrue(acceptance["observation_results_are_not_backfilled"])
        self.assertTrue(acceptance["final_receipt_hash_is_bound_by_p1_amendment"])
        self.assertFalse(payload["p1_execution_authorized"])
        self.assertTrue(payload["scope"]["p0_design_only"])
        self.assertFalse(payload["scope"]["h1_code_implemented"])
        self.assertFalse(payload["scope"]["cpu_fixture_execution_authorized"])
        self.assertFalse(payload["scope"]["gpu_job_authorized"])
        self.assertIsNone(payload["anchors"]["implementation_commit"])
        self.assertIn("p0_acceptance_receipt_is_pending", payload["not_execution_ready_reasons"])
        self.assertIn("receipt_v2_definition_and_validator_are_not_implemented", payload["not_execution_ready_reasons"])

        # Commit anchors are provenance fields, not assertions about the checkout
        # running this test. This deliberately does not compare either value to HEAD.
        self.assertIsNotNone(
            GIT_SHA_RE.fullmatch(payload["anchors"]["design_base_commit"])
        )
        self.assertIsNotNone(
            GIT_SHA_RE.fullmatch(payload["anchors"]["frozen_runtime_commit"])
        )
        self.assertEqual(
            payload["anchors"]["summary_schema_id"], "qwen3vl_e2e_summary_v2"
        )
        self.assertEqual(
            payload["anchors"]["receipt_schema_id"],
            "qwen3vl_evaluation_receipt_v2",
        )
        self.assertIn(
            "p1_execution_authorized", payload["required_p1_amendment_fields"]
        )
        self.assertIn("implementation_commit", payload["required_p1_amendment_fields"])
        self.assertTrue(
            {
                "amends_preregistration_canonical_sha256",
                "p0_acceptance_receipt_sha256",
                "cpu_fixture_manifest_sha256",
                "input_manifest_canonical_sha256",
                "summary_schema_definition_and_validator_hashes",
                "receipt_schema_definition_and_validator_hashes",
                "submission_script_path_and_sha256",
                "qq_notification_titles_verified",
                "user_approval_reference",
            }.issubset(self.payload["required_p1_amendment_fields"])
        )

    def test_artifacts_are_independent_while_schemas_are_shared(self):
        schemas = self.payload["schema_implementation_contract"]
        policy = schemas["artifact_policy"]
        self.assertTrue(policy["independent_harness_protocol_per_arm"])
        self.assertTrue(policy["independent_experiment_arm_id_per_arm"])
        self.assertTrue(policy["independent_summary_artifact_per_arm"])
        self.assertTrue(policy["independent_receipt_artifact_per_arm"])
        self.assertEqual(
            policy["shared_summary_schema_id"], "qwen3vl_e2e_summary_v2"
        )
        self.assertEqual(
            policy["shared_receipt_schema_id"], "qwen3vl_evaluation_receipt_v2"
        )
        self.assertEqual(
            schemas["summary"]["validator_status"], "implementation_in_progress"
        )
        self.assertEqual(schemas["receipt"]["definition_status"], "not_implemented")
        self.assertEqual(schemas["receipt"]["validator_status"], "not_implemented")
        self.assertTrue(
            schemas["receipt"]["execution_blocker_until_definition_and_validator_pass"]
        )

    def test_h0_and_h1_have_independent_ids_and_frozen_budgets(self):
        h0 = self.payload["arms"]["h0"]
        h1 = self.payload["arms"]["h1"]
        self.assertNotEqual(h0["harness_protocol_id"], h1["harness_protocol_id"])
        self.assertNotEqual(h0["experiment_arm_id"], h1["experiment_arm_id"])
        self.assertEqual(
            h0["harness_protocol_id"],
            "qwen3vl_harness_h0_legacy_single_round_v1",
        )
        self.assertEqual(
            h1["harness_protocol_id"],
            "qwen3vl_harness_h1_gtfree_terminal_review_v1",
        )
        self.assertEqual(
            h0["experiment_arm_id"], "v12_1-h0-legacy-single-round-base-0802"
        )
        self.assertEqual(
            h1["experiment_arm_id"], "v12_1-h1-gtfree-terminal-review-base-0802"
        )
        self.assertEqual(
            h0["budgets"],
            {
                "max_verifier_rounds": 1,
                "max_stage_attempts_per_round": 2,
                "max_environment_steps": 3,
            },
        )
        self.assertEqual(
            h1["budgets"],
            {
                "max_verifier_rounds": 2,
                "max_stage_attempts_per_round": 2,
                "max_environment_steps": 3,
            },
        )
        self.assertFalse(h0["terminal_review"])
        self.assertTrue(h1["terminal_review"])
        self.assertEqual(h1["independence_contract"]["round_index"], 2)
        self.assertFalse(h1["independence_contract"]["third_round_allowed"])

        resources = self.payload["resource_contract"]
        self.assertEqual(resources["max_total_gpus_for_ca0802"], 3)
        self.assertEqual(resources["p1_job_gpus"], 1)
        self.assertFalse(resources["h0_h1_concurrent"])
        self.assertEqual(resources["p0_contract_validation_gpus"], 0)
        self.assertEqual(resources["p1_cpu_fixture_gpus"], 0)
        self.assertEqual(self.payload["evaluation_order"][0]["gpu"], 0)

    def test_unique_treatment_factor_and_controls_are_explicit(self):
        factor = self.payload["treatment_factor"]
        self.assertEqual(factor["factor_id"], "terminal_review_scheduler_v1")
        self.assertTrue(factor["is_unique_causal_variable"])
        self.assertEqual(factor["h0_value"], "legacy_single_round_terminal_policy")
        self.assertEqual(
            factor["h1_value"], "one_gtfree_independent_terminal_review_round"
        )
        self.assertTrue(factor["all_other_runtime_variables_must_be_identical"])
        allowed = set(factor["allowed_arm_differences"])
        self.assertIn("max_verifier_rounds", allowed)
        self.assertIn("terminal_review", allowed)

        controls = {record["name"]: record for record in self.payload["controlled_variables"]}
        self.assertEqual(
            set(controls),
            {
                "model_weights_and_model_io_protocol",
                "tools_and_safety_boundaries",
                "ordered_inputs",
                "generation_and_seed",
                "accepted_environment_initial_state_and_nonround_budgets",
            },
        )
        self.assertTrue(all("identical" in record["relation"] for record in controls.values()))

    def test_round_transition_table_is_a_complete_trigger_outcome_cross_product(self):
        h1 = self.payload["arms"]["h1"]
        triggers = set(h1["round_two_triggers"])
        outcomes = set(h1["round_two_outcomes"])
        rows = h1["round_transition_table"]
        indexed = {
            (row["round_one_trigger"], row["round_two_outcome"]): row
            for row in rows
        }
        expected_pairs = {(trigger, outcome) for trigger in triggers for outcome in outcomes}
        self.assertTrue(h1["transition_table_complete_cross_product"])
        self.assertEqual(len(rows), len(expected_pairs))
        self.assertEqual(set(indexed), expected_pairs)

        plan_null_finish = indexed[("plan_null_after_allowed_repair", "finish")]
        self.assertEqual(plan_null_finish["review_decision"], "round_conflict_no_action")
        self.assertEqual(plan_null_finish["tool_action_authorization"], "forbidden")
        for trigger in triggers:
            safe_plan = indexed[(trigger, "unique_safe_plan")]
            self.assertEqual(
                safe_plan["next_state"],
                "existing_environment_action_and_candidate_flow",
            )
            self.assertEqual(
                safe_plan["tool_action_authorization"],
                "not_yet_authorized_existing_safety_flow_required",
            )

    def test_h1_round_two_trigger_is_gt_free(self):
        invariants = self.payload["shared_invariants"]
        h1 = self.payload["arms"]["h1"]
        self.assertTrue(invariants["online_gt_free"])
        self.assertTrue(invariants["gt_allowed_only_after_rollout_complete"])
        forbidden_inputs = h1["forbidden_trigger_inputs"]
        self.assertEqual(
            set(forbidden_inputs),
            {
                "gt_mask",
                "offline_tp_fp_fn",
                "offline_iou_f1",
                "manual_failure_label",
                "historical_false_finish_identity",
            },
        )
        self.assertEqual(len(forbidden_inputs), len(set(forbidden_inputs)))
        self.assertEqual(
            h1["round_two_triggers"],
            ["provisional_finish", "plan_null_after_allowed_repair"],
        )
        self.assertTrue(h1["independence_contract"]["fresh_backend_context"])
        self.assertFalse(h1["independence_contract"]["inherits_hidden_state"])
        self.assertFalse(
            h1["independence_contract"]["copies_round_one_raw_response"]
        )
        self.assertFalse(h1["decision_contract"]["force_action_to_meet_candidate_quota"])

    def test_metric_empty_mask_truth_table_is_frozen(self):
        contract = self.payload["metric_contract"]
        self.assertEqual(
            contract["empty_gt_and_empty_prediction"],
            {"iou": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0},
        )
        self.assertEqual(
            contract["empty_prediction_nonempty_gt"],
            {"precision": 0.0, "recall": 0.0, "iou": 0.0, "f1": 0.0},
        )
        self.assertEqual(
            contract["empty_gt_nonempty_prediction"],
            {"precision": 0.0, "recall": 1.0, "iou": 0.0, "f1": 0.0},
        )

    def test_degradation_contract_catches_metric_and_count_regressions(self):
        contract = self.payload["metric_contract"]
        self.assertEqual(contract["comparison_epsilon"], 1e-12)
        degraded_if = contract["degraded_if"]
        self.assertEqual(
            set(degraded_if),
            {
                "delta_iou_below_negative_epsilon",
                "delta_f1_below_negative_epsilon",
                "metric_tie_and_tp_decreases",
                "metric_tie_and_fp_increases",
                "metric_tie_and_fn_increases",
            },
        )
        self.assertEqual(len(degraded_if), len(set(degraded_if)))
        self.assertEqual(
            contract["oracle_beneficial_if"],
            "delta_iou_above_epsilon_and_delta_f1_above_epsilon",
        )
        self.assertEqual(
            contract["oracle_harmful_if"],
            "either_metric_decreases_or_metric_tie_with_worse_counts",
        )

        baseline = {"iou": 0.5, "f1": 0.6, "tp": 10, "fp": 2, "fn": 3}
        cases = {
            "iou decline": ({**baseline, "iou": 0.4}, True),
            "f1 decline": ({**baseline, "f1": 0.5}, True),
            "tied metrics with lower tp": ({**baseline, "tp": 9}, True),
            "tied metrics with higher fp": ({**baseline, "fp": 3}, True),
            "tied metrics with higher fn": ({**baseline, "fn": 4}, True),
            "unchanged": (baseline.copy(), False),
            "strict metric improvement": (
                {**baseline, "iou": 0.6, "f1": 0.7},
                False,
            ),
        }
        for label, (candidate, expected) in cases.items():
            with self.subTest(label=label):
                self.assertEqual(_is_degraded(contract, baseline, candidate), expected)

        # Historical v1 failure mode: IoU/F1 stay at zero while 6,518 false
        # positives are added. Count-sensitive degradation must still reject it.
        zero_before = {"iou": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 10}
        zero_after = {**zero_before, "fp": 6518}
        self.assertTrue(_is_degraded(contract, zero_before, zero_after))

    def test_candidate_confusion_mapping_is_complete_and_has_denominators(self):
        contract = self.payload["candidate_confusion_contract"]
        oracle_labels = set(contract["oracle_labels"])
        decision_paths = set(contract["decision_paths"])
        mapping = {
            (record["oracle_label"], record["decision_path"]): record["leaf_class"]
            for record in contract["mapping"]
        }
        self.assertTrue(contract["leaf_classes_are_mutually_exclusive"])
        self.assertEqual(len(contract["mapping"]), len(mapping))
        self.assertEqual(
            set(mapping),
            {
                (oracle_label, decision_path)
                for oracle_label in oracle_labels
                for decision_path in decision_paths
            },
        )
        self.assertEqual(
            mapping[("harmful", "accepted_after_qwen")], "false_accept"
        )
        self.assertEqual(
            mapping[("beneficial", "rejected_after_qwen")],
            "qwen_false_reject",
        )
        self.assertEqual(
            mapping[("beneficial", "rejected_by_hard_gate_before_qwen")],
            "hard_gate_false_reject",
        )
        self.assertTrue(contract["indeterminate_never_forced_into_binary_confusion"])

        denominators = contract["rollups_and_denominators"]
        self.assertEqual(
            denominators["false_accept_rate"]["denominator"],
            "all_oracle_harmful_candidates",
        )
        self.assertEqual(
            denominators["qwen_false_reject_rate"]["denominator"],
            "oracle_beneficial_candidates_reaching_qwen",
        )
        self.assertEqual(
            denominators["hard_gate_false_reject_rate"]["denominator"],
            "all_oracle_beneficial_candidates",
        )

    def test_protocol_and_safety_primary_endpoints_are_machine_readable(self):
        contract = self.payload["primary_endpoint_contract"]
        self.assertTrue(contract["all_endpoints_must_pass"])
        endpoints = {
            endpoint["endpoint_id"]: endpoint for endpoint in contract["endpoints"]
        }
        self.assertEqual(
            set(endpoints),
            {
                "h0_behavior_parity_rate",
                "eligible_trigger_exactly_one_round_two_rate",
                "h1_final_region_protocol_validity_delta_vs_h0",
                "pre_rollout_gt_boundary_violation_rate",
                "safety_boundary_relaxation_rate",
                "requested_scored_mismatch_rate",
                "tool_failure_rate",
            },
        )
        for endpoint_id, endpoint in endpoints.items():
            with self.subTest(endpoint=endpoint_id):
                for field in (
                    "numerator",
                    "denominator",
                    "comparator",
                    "operator",
                    "threshold",
                ):
                    self.assertIn(field, endpoint)
                self.assertIn(endpoint["operator"], {"==", ">="})

        self.assertEqual(
            endpoints["h1_final_region_protocol_validity_delta_vs_h0"]["operator"],
            ">=",
        )
        self.assertEqual(
            endpoints["h1_final_region_protocol_validity_delta_vs_h0"]["threshold"],
            0.0,
        )
        diagnostics = self.payload["diagnostic_and_promotion_metrics"]
        self.assertTrue(diagnostics["not_harness_freeze_primary"])
        self.assertFalse(
            diagnostics["candidate_novelty_required_for_harness_safety_freeze"]
        )
        self.assertFalse(diagnostics["may_drive_online_trigger_region_or_action_selection"])
        self.assertFalse(diagnostics["force_action_to_meet_candidate_quota"])

    def test_retry_accounting_separates_schema_network_and_tool_attempts(self):
        contract = self.payload["retry_accounting_contract"]
        self.assertTrue(contract["max_stage_attempts_per_round_includes_initial"])
        self.assertEqual(contract["max_schema_repairs_per_stage_per_round"], 1)
        self.assertEqual(contract["effective_network_retry_limit_for_local_backend"], 0)
        self.assertEqual(
            contract["effective_tool_action_attempts_per_authorized_action"], 1
        )
        self.assertTrue(
            contract["action_retries_cli"]["deprecated_compatibility_option"]
        )
        self.assertFalse(
            contract["action_retries_cli"]["changes_effective_tool_attempt_limit"]
        )
        self.assertEqual(
            set(contract["required_separate_counters"]),
            {
                "verifier_round_count",
                "stage_initial_attempt_count",
                "schema_repair_attempt_count",
                "network_retry_attempt_count",
                "tool_action_attempt_count",
                "tool_failure_count",
            },
        )

    def test_input_manifest_hash_orders_and_offline_label_boundary(self):
        reference = self.payload["input_manifest"]
        relative_path = Path(reference["path"])
        self.assertFalse(relative_path.is_absolute())
        self.assertEqual((REPO_ROOT / relative_path).resolve(), INPUT_MANIFEST_PATH)
        self.assertEqual(reference["schema_id"], self.input_manifest["schema_id"])
        self.assertIsNotNone(SHA256_RE.fullmatch(reference["canonical_sha256"]))
        self.assertEqual(
            reference["canonical_sha256"],
            "<hash omitted>",
        )
        self.assertEqual(
            reference["canonicalization"],
            self.input_manifest["canonical_json_contract"]["serialization"],
        )
        self.assertEqual(
            _canonical_json_sha256(self.input_manifest),
            reference["canonical_sha256"],
        )

        profiles = {
            profile["phase"]: profile
            for profile in self.input_manifest["evaluation_profiles"]
        }
        self.assertEqual(profiles["smoke1"]["ordered_sample_ids"], ["test_1_1"])
        self.assertEqual(
            profiles["core3"]["ordered_sample_ids"],
            ["test_20_15", "test_1_1", "test_50_8"],
        )
        prereg_profiles = {
            record["phase"]: record
            for record in self.payload["evaluation_order"]
            if "samples" in record
        }
        self.assertEqual(
            profiles["smoke1"]["ordered_sample_ids"],
            prereg_profiles["smoke1"]["samples"],
        )
        self.assertEqual(
            profiles["core3"]["ordered_sample_ids"],
            prereg_profiles["core3"]["samples"],
        )

        gt_boundary = self.input_manifest["gt_boundary"]
        self.assertFalse(gt_boundary["label_cvt_online_input"])
        self.assertTrue(gt_boundary["label_cvt_offline_scorer_only"])
        self.assertTrue(gt_boundary["label_may_be_opened_only_after_rollout_complete"])
        input_root = Path(self.input_manifest["source_contract"]["input_root"])
        self.assertEqual(input_root, Path(self.payload["pinned_paths"]["input_root"]))
        self.assertEqual(
            set(self.input_manifest["samples"]),
            {"test_20_15", "test_1_1", "test_50_8"},
        )
        for sample_id, sample in self.input_manifest["samples"].items():
            with self.subTest(sample=sample_id):
                for prefix in ("a", "b", "label"):
                    relative_source = Path(sample[f"{prefix}_path_relative_to_input_root"])
                    self.assertFalse(relative_source.is_absolute())
                    source_path = (input_root / relative_source).resolve()
                    self.assertTrue(source_path.is_relative_to(input_root.resolve()))
                    self.assertTrue(source_path.is_file())
                    self.assertEqual(_sha256(source_path), sample[f"{prefix}_sha256"])
                self.assertEqual(
                    sample["label_access"],
                    "offline_scorer_only_after_rollout_complete",
                )

    def test_cpu_fixture_manifest_is_an_explicit_p1_blocker(self):
        fixture = self.payload["cpu_fixture_manifest"]
        self.assertEqual(fixture["status"], "p1_blocker_not_created")
        self.assertIsNone(fixture["canonical_sha256"])
        self.assertTrue(fixture["required_before_cpu_fixture_submission"])
        self.assertTrue(fixture["required_before_any_gpu_submission"])
        fixture_phase = self.payload["evaluation_order"][0]
        self.assertEqual(fixture_phase["fixture_manifest_path"], fixture["path"])
        self.assertIsNone(fixture_phase["fixture_manifest_sha256"])

    def test_reserved_allocations_and_notification_titles_are_exact(self):
        allocations = self.payload["reserved_run_allocations"]
        expected_names = [
            "CA_0802(1)-qwen3vl-p0-contract-validation",
            "CA_0802(2)-qwen3vl-h0-h1-cpu-fixtures",
            "CA_0802(3)-qwen3vl-h0-base-smoke1",
            "CA_0802(4)-qwen3vl-h1-base-smoke1",
            "CA_0802(5)-qwen3vl-h0-base-core3",
            "CA_0802(6)-qwen3vl-h1-base-core3",
        ]
        expected_titles = [
            "CA0802 P0 contract validation",
            "CA0802 P1 H0-H1 CPU fixtures",
            "CA0802 P1 H0 base smoke1",
            "CA0802 P1 H1 base smoke1",
            "CA0802 P1 H0 base CORE3",
            "CA0802 P1 H1 base CORE3",
        ]
        self.assertEqual(
            [Path(record["output_path"]).name for record in allocations],
            expected_names,
        )
        self.assertEqual([record["qq_title"] for record in allocations], expected_titles)
        self.assertEqual(
            len({record["output_path"] for record in allocations}), len(allocations)
        )
        for record in allocations:
            with self.subTest(output=record["output_path"]):
                output_path = Path(record["output_path"])
                self.assertTrue(record["must_not_exist_before_submission"])
                self.assertEqual(
                    record["reservation_status"],
                    "reserved_absent_at_design_freeze",
                )
                self.assertEqual(Path(record["stdout_path"]).parent, output_path / "logs")
                self.assertEqual(Path(record["stderr_path"]).parent, output_path / "logs")

        receipt_path = Path(self.payload["p0_acceptance_receipt"]["path"])
        self.assertFalse(receipt_path.is_absolute())
        self.assertEqual(
            receipt_path,
            Path(
                "doc/training_pipeline/0802_third_try/"
                "QWEN3VL_P0_ACCEPTANCE_RECEIPT_0802.json"
            ),
        )
        notifications = self.payload["notification_contract"]
        self.assertEqual(notifications["channel"], "QQ")
        self.assertEqual(
            set(notifications["required_body_fields"]),
            {
                "job_name",
                "job_id",
                "terminal_status",
                "elapsed_seconds",
                "key_result_or_error",
                "artifact_path",
            },
        )
        self.assertTrue(notifications["credentials_or_proxy_values_forbidden"])

    def test_p0_retry_amendment_preserves_failure_and_changes_only_validation(self):
        amendment = self.p0_r1_amendment
        self.assertEqual(
            amendment["schema_id"], "qwen3vl_p0_validation_amendment_v1"
        )
        self.assertEqual(amendment["design_status"], "frozen")
        self.assertEqual(
            amendment["amends"]["path"],
            PREREGISTRATION_PATH.relative_to(REPO_ROOT).as_posix(),
        )
        self.assertEqual(
            amendment["amends"]["canonical_sha256"],
            _canonical_json_sha256(self.payload),
        )
        self.assertEqual(
            amendment["amends"]["scope"],
            "p0_contract_validation_allocation_and_worker_identity_only",
        )

        reason = amendment["reason_for_amendment"]
        self.assertEqual(reason["category"], "p0_harness_preflight_bug")
        self.assertFalse(reason["scientific_result_available"])
        self.assertFalse(reason["h0_or_h1_behavior_observed"])
        self.assertFalse(reason["gpu_job_started"])

        failed = amendment["failed_attempt"]
        self.assertEqual(Path(failed["output_path"]).name, "CA_0802(1)-qwen3vl-p0-contract-validation")
        self.assertEqual(
            failed["launch_commit"],
            "383521f72e846b459697dc24208b8245243975ec",
        )
        self.assertEqual(failed["execution_status"], "failed")
        self.assertFalse(failed["tests_started"])
        self.assertEqual(
            [(job["job_id"], job["job_name"], job["state"]) for job in failed["jobs"]],
            [
                ("52553", "CA0802P0Target", "FAILED"),
                ("52554", "CA0802P0Regress", "FAILED"),
            ],
        )
        self.assertTrue(all(job["allocated_gpus"] == 0 for job in failed["jobs"]))

        failed_root = _resolve_p0_historical_output(failed["output_path"])
        evidence_records = failed["preservation"]["evidence_sha256"]
        self.assertEqual(len(evidence_records), 9)
        self.assertEqual(
            len({record["path_relative_to_output"] for record in evidence_records}),
            len(evidence_records),
        )
        for record in evidence_records:
            with self.subTest(path=record["path_relative_to_output"]):
                relative = Path(record["path_relative_to_output"])
                self.assertFalse(relative.is_absolute())
                self.assertNotIn("..", relative.parts)
                source = (failed_root / relative).resolve(strict=True)
                self.assertTrue(source.is_relative_to(failed_root))
                self.assertEqual(_sha256(source), record["sha256"])

        identity = amendment["amended_worker_identity_contract"]
        self.assertTrue(identity["slurm_spool_copy_expected"])
        self.assertTrue(identity["forbid_spool_path_equality_with_worktree"])
        self.assertTrue(identity["path_identity_is_not_used_as_byte_identity"])
        self.assertEqual(
            set(identity["required_checks"]),
            {
                "exact_clean_repo_commit_matches_submission",
                "sha256_of_slurm_spool_copy_matches_submitter_pinned_worker_sha256",
                "sha256_of_repo_worker_matches_submitter_pinned_worker_sha256",
            },
        )

        retry = amendment["retry_allocation"]
        retry_root = Path(retry["output_path"])
        self.assertEqual(retry_root.name, "CA_0802(1R1)-qwen3vl-p0-contract-validation")
        self.assertNotEqual(retry_root, Path(failed["output_path"]))
        self.assertEqual(retry["targeted_job_name"], "CA0802P0TargetR1")
        self.assertEqual(retry["regression_job_name"], "CA0802P0RegressR1")
        self.assertEqual(retry["qq_title"], "CA0802 P0 contract validation R1")
        self.assertEqual(retry["gpus_per_job"], 0)
        self.assertEqual(Path(retry["stdout_path"]).parent, retry_root / "logs")
        self.assertEqual(Path(retry["stderr_path"]).parent, retry_root / "logs")
        self.assertTrue(retry["must_not_exist_before_submission"])

        unchanged = amendment["unchanged_contract"]
        self.assertTrue(unchanged["h0_h1_design"])
        self.assertEqual(unchanged["treatment_factor"], "terminal_review_scheduler_v1")
        self.assertTrue(unchanged["p1_allocations_ca0802_2_through_6"])
        self.assertFalse(unchanged["p1_execution_authorized"])
        self.assertFalse(unchanged["h1_code_implemented"])
        self.assertFalse(unchanged["gpu_execution_authorized"])
        authorization = amendment["authorization"]
        self.assertTrue(authorization["p0_retry_execution_authorized"])
        self.assertEqual(authorization["maximum_gpus"], 0)
        self.assertFalse(authorization["p1_authorized"])

        worker = (REPO_ROOT / "tools/run_qwen3vl_p0_validation.slurm").read_text(
            encoding="utf-8"
        )
        self.assertIn("EXECUTED_WORKER_COPY", worker)
        self.assertIn("observed_copy_sha256", worker)
        self.assertIn("observed_pinned_sha256", worker)
        self.assertNotIn('[[ "$WORKER_SELF" ==', worker)

    def test_p0_r2_amendment_freezes_both_failures_and_exact_order_contract(self):
        amendment = self.p0_r2_amendment
        self.assertEqual(
            amendment["schema_id"], "qwen3vl_p0_validation_amendment_r2_v1"
        )
        self.assertEqual(amendment["design_status"], "frozen")
        self.assertEqual(
            amendment["digest_contract"]["canonical_scope"],
            "the complete parsed JSON object",
        )
        self.assertIn(
            "raw file bytes",
            amendment["digest_contract"]["evidence_sha256_semantics"],
        )
        self.assertEqual(
            _canonical_json_sha256(amendment),
            P0_R2_AMENDMENT_CANONICAL_SHA256,
        )

        chain = amendment["amendment_chain"]
        self.assertEqual(
            chain["preregistration"]["path"],
            PREREGISTRATION_PATH.relative_to(REPO_ROOT).as_posix(),
        )
        self.assertEqual(
            chain["preregistration"]["canonical_sha256"],
            _canonical_json_sha256(self.payload),
        )
        self.assertEqual(
            chain["r1_amendment"]["path"],
            P0_R1_AMENDMENT_PATH.relative_to(REPO_ROOT).as_posix(),
        )
        self.assertEqual(
            chain["r1_amendment"]["canonical_sha256"],
            _canonical_json_sha256(self.p0_r1_amendment),
        )
        self.assertEqual(
            chain["r1_amendment"]["raw_file_sha256"],
            _sha256(P0_R1_AMENDMENT_PATH),
        )

        reason = amendment["reason_for_amendment"]
        self.assertEqual(
            reason["category"], "canonical_sample_order_contract_conflict"
        )
        self.assertEqual(reason["failure_class"], "contract_failed")
        self.assertTrue(reason["systematic_not_flaky"])
        self.assertFalse(reason["scientific_result_available"])
        self.assertFalse(reason["gpu_job_started"])

        attempts = amendment["failed_attempts"]
        self.assertEqual(
            [attempt["attempt_id"] for attempt in attempts],
            ["p0_initial", "p0_r1"],
        )
        self.assertEqual(
            [len(attempt["evidence_sha256"]) for attempt in attempts], [9, 11]
        )
        for attempt in attempts:
            attempt_root = _resolve_p0_historical_output(attempt["output_path"])
            records = attempt["evidence_sha256"]
            self.assertEqual(
                len({record["path_relative_to_output"] for record in records}),
                len(records),
            )
            for record in records:
                with self.subTest(
                    attempt=attempt["attempt_id"],
                    path=record["path_relative_to_output"],
                ):
                    relative = Path(record["path_relative_to_output"])
                    self.assertFalse(relative.is_absolute())
                    self.assertNotIn("..", relative.parts)
                    source = (attempt_root / relative).resolve(strict=True)
                    self.assertTrue(source.is_relative_to(attempt_root))
                    self.assertEqual(_sha256(source), record["sha256"])
            self.assertTrue(
                all(job["allocated_gpus"] == 0 for job in attempt["jobs"])
            )

        r1 = attempts[1]
        self.assertTrue(r1["tests_started"])
        self.assertEqual(r1["scientific_outcome"], "not_evaluated")
        self.assertEqual(r1["failure_stage"], "sample_identity_order_contract")
        self.assertEqual(
            r1["test_observations"]["targeted"]["combined"],
            "77 passed, 3 errors, 80 total",
        )
        self.assertEqual(
            r1["test_observations"]["full_regression"],
            "352 passed, 3 errors, 3 skipped, 358 total",
        )
        self.assertEqual(
            r1["test_observations"]["expected_summary_artifact_status"],
            "not_created",
        )
        archive_window = r1["archive_hash_window"]
        self.assertEqual(archive_window["scope"], "8_named_summarizer_inputs")
        self.assertEqual(archive_window["file_count"], 8)
        self.assertEqual(archive_window["archive_run_total_files_observed"], 89)
        self.assertEqual(archive_window["uncovered_files"], 81)
        self.assertEqual(
            archive_window["before_manifest_sha256"],
            archive_window["after_manifest_sha256"],
        )
        self.assertFalse(
            archive_window["persistent_filesystem_immutability_proven"]
        )

        order = amendment["corrected_legacy_order_contract"]
        self.assertEqual(
            order["rollout_complete_samples"]["required_value"],
            "sorted(requested_samples)",
        )
        self.assertEqual(
            order["rollout_complete_samples"]["comparison"],
            "exact_list_equality",
        )
        self.assertTrue(
            order["rollout_complete_samples"]["set_equality_forbidden"]
        )
        self.assertEqual(
            order["per_sample_metrics_samples"]["required_value"],
            "requested_samples",
        )
        self.assertEqual(
            set(order["required_consumers"]),
            {
                "tools/analyze_staged_verifier_run.py",
                "tools/summarize_qwen3vl_e2e_run.py",
                "tools/compare_local_verifier_ab.py",
            },
        )

        retry = amendment["retry_allocation"]
        retry_root = Path(retry["output_path"])
        self.assertEqual(
            retry_root.name, "CA_0802(1R2)-qwen3vl-p0-contract-validation"
        )
        self.assertEqual(retry["targeted_job_name"], "CA0802P0TargetR2")
        self.assertEqual(retry["regression_job_name"], "CA0802P0RegressR2")
        self.assertEqual(retry["qq_title"], "CA0802 P0 contract validation R2")
        self.assertEqual(retry["gpus_per_job"], 0)
        self.assertEqual(retry["retry_of_output"], attempts[1]["output_path"])
        self.assertEqual(
            retry["regression_dependency"], "afterany:<targeted_job_id>"
        )
        self.assertEqual(
            retry["targeted_resources"],
            {"cpus": 2, "memory": "8G", "time": "00:20:00"},
        )
        self.assertEqual(
            retry["regression_resources"],
            {"cpus": 2, "memory": "8G", "time": "00:45:00"},
        )
        self.assertTrue(retry["must_verify_all_failed_attempts"])
        self.assertTrue(retry["must_not_exist_before_submission"])
        self.assertEqual(Path(retry["stdout_path"]).parent, retry_root / "logs")
        self.assertEqual(Path(retry["stderr_path"]).parent, retry_root / "logs")

        unchanged = amendment["unchanged_contract"]
        self.assertTrue(unchanged["h0_h1_design"])
        self.assertEqual(
            unchanged["input_manifest_canonical_sha256"],
            _canonical_json_sha256(self.input_manifest),
        )
        self.assertEqual(
            unchanged["workbook_raw_sha256"],
            "<hash omitted>",
        )
        self.assertNotEqual(
            unchanged["workbook_raw_sha256"],
            _sha256(REPO_ROOT / "experiments" / "experiment_registry.xlsx"),
            "the P0 amendment must bind its historical workbook snapshot, not "
            "prevent the maintained registry from receiving later append-only rows",
        )
        self.assertFalse(unchanged["p1_execution_authorized"])
        self.assertFalse(unchanged["h1_code_implemented"])
        self.assertFalse(unchanged["gpu_execution_authorized"])
        authorization = amendment["authorization"]
        self.assertTrue(authorization["p0_retry_execution_authorized"])
        self.assertEqual(authorization["maximum_gpus"], 0)
        self.assertFalse(authorization["p1_authorized"])

        receipt = amendment["acceptance_receipt"]
        self.assertEqual(receipt["schema_id"], "qwen3vl_p0_acceptance_receipt_v1")
        self.assertEqual(
            Path(receipt["path"]),
            Path(
                "doc/training_pipeline/0802_third_try/"
                "QWEN3VL_P0_ACCEPTANCE_RECEIPT_0802.json"
            ),
        )
        self.assertEqual(
            receipt["must_reference_attempt_ids"],
            ["p0_initial", "p0_r1", "p0_r2"],
        )
        self.assertEqual(receipt["status_at_amendment"], "pending")
        self.assertEqual(
            set(receipt["required_top_level_fields"]),
            {
                "schema_id",
                "status",
                "created_utc",
                "validation_commit",
                "validation_worktree",
                "frozen_inputs",
                "attempts",
                "gate_matrix",
                "archive_hash_window",
                "notification_contract",
                "p0_accepted",
                "failed_gates",
                "p1_execution_authorized",
                "canonical_json_contract",
            },
        )
        self.assertEqual(
            set(receipt["required_attempt_fields"]),
            {
                "attempt_id",
                "retry_of_attempt_id",
                "terminal_status",
                "scientific_result_available",
                "output_path",
                "launch_commit",
                "worker_sha256",
                "jobs",
                "evidence_sha256",
            },
        )
        self.assertEqual(
            set(receipt["required_job_fields"]),
            {
                "job_id",
                "job_name",
                "mode",
                "state",
                "exit_code",
                "elapsed",
                "allocated_cpus",
                "allocated_gpus",
                "requested_memory",
                "node",
            },
        )
        self.assertEqual(
            set(receipt["required_frozen_input_digests"]),
            {
                "preregistration_canonical_sha256",
                "r1_amendment_canonical_sha256",
                "r2_amendment_canonical_sha256",
                "input_manifest_canonical_sha256",
                "workbook_raw_sha256",
            },
        )
        complete_if = " ".join(receipt["complete_iff"])
        self.assertIn("all required gate IDs", complete_if)
        self.assertIn("Slurm state COMPLETED", complete_if)
        self.assertIn("status success and exit_code 0", complete_if)
        self.assertIn("p1_execution_authorized is false", complete_if)
        self.assertEqual(
            set(receipt["required_gate_ids"]),
            {
                "amendment_chain_hashes_match",
                "failed_attempt_evidence_20_of_20_match",
                "r2_submission_receipt_valid",
                "r2_exact_clean_commit_and_worker_identity",
                "r2_zero_gpu_resources",
                "p0_contract_tests_pass",
                "experiment_registry_tests_pass",
                "local_verifier_analysis_tests_pass",
                "shell_syntax_check_pass",
                "workbook_deterministic_check_pass",
                "v1_archive_summary_generated",
                "v1_false_accept_and_degradation_assertion_pass",
                "v1_archive_8_file_hash_window_unchanged",
                "full_unittest_regression_pass",
                "r2_targeted_job_success",
                "r2_regression_job_success",
            },
        )
        self.assertEqual(
            receipt["notification_claim_contract"][
                "allowed_claim_without_external_delivery_receipt"
            ],
            "worker_attempted_notification",
        )
        self.assertEqual(
            receipt["notification_claim_contract"][
                "forbidden_claim_without_external_delivery_receipt"
            ],
            "recipient_delivery_proven",
        )

        submitter = (REPO_ROOT / "tools/submit_qwen3vl_p0_validation.sh").read_text(
            encoding="utf-8"
        )
        worker = (REPO_ROOT / "tools/run_qwen3vl_p0_validation.slurm").read_text(
            encoding="utf-8"
        )
        for token in (
            retry_root.name,
            retry["targeted_job_name"],
            retry["regression_job_name"],
            P0_R2_AMENDMENT_PATH.name,
            P0_R2_AMENDMENT_CANONICAL_SHA256,
        ):
            self.assertIn(token, submitter)
            self.assertIn(token, worker)
        for script in (submitter, worker):
            self.assertIn(
                '"$AMENDMENT" "$R2_AMENDMENT_CANONICAL_SHA256"', script
            )
            self.assertIn(
                '"$INPUT_MANIFEST" "$INPUT_MANIFEST_CANONICAL_SHA256"', script
            )
            self.assertIn(
                '"experiments/experiment_registry.xlsx" "$WORKBOOK_SHA256"',
                script,
            )
        for attempt in attempts:
            for record in attempt["evidence_sha256"]:
                pair_pattern = re.compile(
                    rf'"{re.escape(record["path_relative_to_output"])}"'
                    rf'\s*\\\s*"{record["sha256"]}"'
                )
                for script in (submitter, worker):
                    self.assertRegex(script, pair_pattern)
        self.assertIn("verify_failed_attempts", worker)


class Qwen3VLV2ReconstructedConfigContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = _load_json(RECONSTRUCTED_CONFIG_PATH)

    def test_schema_status_warning_and_read_only_boundary(self):
        payload = self.payload
        self.assertEqual(
            payload["schema_id"],
            "qwen3vl_reconstructed_resolved_training_config_evidence_map_v1",
        )
        self.assertEqual(payload["record_type"], "reconstructed_resolved_config")
        self.assertEqual(
            payload["record_status"],
            "reconstructed_read_only_evidence_not_original_runtime_receipt",
        )
        warning = payload["warning"]
        self.assertTrue(warning["original_config_used_is_not_resolved"])
        self.assertEqual(
            warning["allowed_use"],
            "Audit and interpretation of the completed historical run only.",
        )
        for recovered_fact in (
            "max_samples=39",
            "learning_rate=1e-4",
            "num_train_epochs=4",
            "48 examples",
            "2 epochs",
            "12 optimizer steps",
            "3e-5 peak learning rate",
        ):
            with self.subTest(recovered_fact=recovered_fact):
                self.assertIn(recovered_fact, warning["reason"])
        self.assertIn("original runtime-emitted receipt", warning["forbidden_use"])
        self.assertIn("do not use archived data paths", warning["forbidden_use"])
        unrecoverable = " ".join(payload["known_unrecoverable_or_non_persisted_fields"])
        self.assertIn("submit-shell spelling/order", unrecoverable)
        self.assertIn("transient output directory no longer exists", unrecoverable)
        self.assertIn("not valid rerun inputs", unrecoverable)
        self.assertFalse(payload["archive_mutation"])

    def test_key_recovered_values_are_cross_consistent(self):
        payload = self.payload
        config = payload["reconstructed_resolved_config"]
        outcome = payload["observed_training_outcome"]
        historical = payload["historical_experiment"]

        self.assertEqual(historical["job_id"], "52011")
        self.assertEqual(historical["data_job_id"], "52009")
        self.assertIsNotNone(GIT_SHA_RE.fullmatch(historical["code_commit"]))
        self.assertIsNotNone(GIT_SHA_RE.fullmatch(historical["llamafactory_commit"]))
        self.assertEqual(config["max_samples"], outcome["train_examples"])
        self.assertEqual(outcome["train_examples"], 48)
        self.assertEqual(outcome["validation_examples_not_used_by_trainer"], 24)
        self.assertFalse(config["do_eval"])
        self.assertEqual(config["learning_rate"], 3e-5)
        self.assertEqual(config["num_train_epochs"], outcome["actual_epochs"])
        self.assertEqual(outcome["actual_epochs"], 2.0)
        self.assertEqual(outcome["optimizer_steps"], 12)
        self.assertEqual(config["effective_train_batch_size"], 8)
        self.assertEqual(config["seed"], 801)
        self.assertEqual(config["data_seed"], 801)
        self.assertEqual(outcome["gpus"], config["world_size"])
        self.assertEqual(outcome["gpus"], 2)
        self.assertEqual(
            config["effective_train_batch_size"],
            config["per_device_train_batch_size"]
            * config["gradient_accumulation_steps"]
            * config["world_size"],
        )
        expected_steps = (
            outcome["train_examples"]
            * config["num_train_epochs"]
            / config["effective_train_batch_size"]
        )
        self.assertTrue(math.isclose(expected_steps, outcome["optimizer_steps"]))

    def test_all_repo_relative_evidence_files_exist_and_match_sha256(self):
        evidence_records = []
        for field_name, field_evidence in self.payload["field_evidence"].items():
            for record in field_evidence.get("evidence_files", []):
                evidence_records.append((field_name, record))

        expected_paths = {
            "outputs/CA_0801_archive/CA_0801(13)-qwen3vl-v121-hardcase-sft-v2/submission.txt",
            "outputs/CA_0801_archive/CA_0801(13)-qwen3vl-v121-hardcase-sft-v2/logs/train-52011.log",
            "outputs/CA_0801_archive/CA_0801(13)-qwen3vl-v121-hardcase-sft-v2/trainer_state.json",
            "outputs/CA_0801_archive/CA_0801(13)-qwen3vl-v121-hardcase-sft-v2/train_results.json",
            "outputs/CA_0801_archive/CA_0801(13)-qwen3vl-v121-hardcase-sft-v2/training_summary.json",
            "outputs/CA_0801_archive/CA_0801(12)-qwen3vl-v121-hardcase-data-v2-rerun1/data/manifest.json",
            "outputs/CA_0801_archive/CA_0801(12)-qwen3vl-v121-hardcase-data-v2-rerun1/data/validation.json",
        }
        self.assertEqual(len(evidence_records), len(expected_paths))
        self.assertEqual({record["path"] for _, record in evidence_records}, expected_paths)

        manifest_path = next(
            record
            for _, record in evidence_records
            if record["path"].endswith("/data/manifest.json")
        )
        self.assertEqual(
            manifest_path["sha256"],
            self.payload["observed_training_outcome"]["dataset_manifest_sha256"],
        )

        repo_root = REPO_ROOT.resolve()
        # A formal clean run worktree exposes the ignored historical artifacts
        # through a submitter-pinned ``outputs`` symlink.  Constrain resolved
        # evidence to that exact output root instead of incorrectly requiring
        # the symlink target to remain physically below the code worktree.
        output_root = (repo_root / "outputs").resolve(strict=True)
        for field_name, record in evidence_records:
            with self.subTest(field=field_name, path=record["path"]):
                relative_path = Path(record["path"])
                self.assertFalse(relative_path.is_absolute())
                self.assertEqual(relative_path.parts[0], "outputs")
                self.assertNotIn("..", relative_path.parts)
                evidence_path = (repo_root / relative_path).resolve()
                self.assertTrue(evidence_path.is_relative_to(output_root))
                self.assertTrue(evidence_path.is_file())
                self.assertIsNotNone(SHA256_RE.fullmatch(record["sha256"]))
                self.assertEqual(_sha256(evidence_path), record["sha256"])


if __name__ == "__main__":
    unittest.main()
