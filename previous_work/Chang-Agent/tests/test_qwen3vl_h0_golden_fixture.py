import copy
import hashlib
import json
import types
import unittest
from pathlib import Path

import numpy as np

from change_agent.adapters.stage_backends import _stage_prompt
from change_agent.adapters.staged_verifier import StagedQwenVerifier
from change_agent.adapters.terminal_review_verifier import TerminalReviewVerifier
from change_agent.state import ChangeState
from change_agent.terminal_review import TerminalReviewPolicy
from change_agent.verifier_regions import attach_verifier_regions
from tools.run_levir_change_agent import (
    _episode_outcome,
    _execute_verifier_action,
    _initial_verifier_stop_reason,
)
from tools.validate_qwen3vl_h0_golden_fixture import (
    GOLDEN_RECEIPT_RELPATH,
    GoldenFixtureValidationError,
    canonical_sha256,
    sha256_bytes,
    validate_receipt,
    validate_receipt_payload,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RECEIPT_PATH = REPO_ROOT / GOLDEN_RECEIPT_RELPATH
MANIFEST_PATH = (
    REPO_ROOT
    / "doc/training_pipeline/0802_third_try/"
    "QWEN3VL_H0_H1_CPU_FIXTURE_MANIFEST_0802.json"
)


def _json_value(value):
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _fill_rectangles(mask, rectangles):
    for x1, y1, x2, y2 in rectangles:
        mask[int(y1) : int(y2), int(x1) : int(x2)] = True


def _state_from_receipt(fixture):
    spec = fixture["state"]
    height, width, channels = spec["image_shape_hwc"]
    t1_image = np.full(
        (height, width, channels), spec["t1_image_fill"], dtype=np.uint8
    )
    t2_image = np.full(
        (height, width, channels), spec["t2_image_fill"], dtype=np.uint8
    )
    x1, y1, x2, y2 = spec["t2_image_patch_xyxy_exclusive"]
    t2_image[y1:y2, x1:x2] = spec["t2_image_patch_value"]
    t1_mask = np.zeros((height, width), dtype=bool)
    t2_mask = np.zeros_like(t1_mask)
    _fill_rectangles(t1_mask, spec["t1_mask_rectangles_xyxy_exclusive"])
    _fill_rectangles(t2_mask, spec["t2_mask_rectangles_xyxy_exclusive"])
    if spec["change_mask_rule"] != "logical_xor_t1_t2":
        raise AssertionError("unsupported frozen change-mask rule")
    state = ChangeState(
        t1_image,
        t2_image,
        spec["query"],
        t1_mask,
        t2_mask,
        np.logical_xor(t1_mask, t2_mask),
    )
    region = fixture["region_builder"]
    if region["factory"] != "attach_verifier_regions":
        raise AssertionError("unsupported frozen region factory")
    attach_verifier_regions(
        state,
        max_regions=region["max_regions"],
        min_component_area=region["min_component_area"],
        padding_ratio=region["padding_ratio"],
        empty_mask_scan_grid_size=region["empty_mask_scan_grid_size"],
    )
    return state


class _ReceiptAuditBackend:
    """Replay only receipt-defined model input; retain production-like audit."""

    def __init__(self, response):
        self.response = copy.deepcopy(response)
        self.call_history = []
        self.payloads = []

    def reset_audit(self):
        self.call_history = []
        self.payloads = []

    def generate_stage(self, stage, state, payload, previous_state=None):
        if stage != "audit":
            raise AssertionError(f"golden H0 fixture forbids stage {stage!r}")
        if previous_state is not None:
            raise AssertionError("golden H0 fixture is an initial-state audit")
        response = {
            "region_id": payload["region"]["region_id"],
            **copy.deepcopy(self.response),
        }
        prompt = _stage_prompt(stage, payload, None)
        self.payloads.append(copy.deepcopy(payload))
        self.call_history.append(
            {
                "backend": "frozen_h0_receipt_fixture",
                "stage": stage,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "attempt_kind": "initial",
                "network_retry_attempt_count": 0,
                "visual_input_scope": "standard",
                "raw_response": json.dumps(response, sort_keys=True),
                "parsed_output": copy.deepcopy(response),
            }
        )
        return response


def _verifier_from_receipt(fixture, backend):
    spec = fixture["verifier"]
    if spec["factory"] != "StagedQwenVerifier":
        raise AssertionError("unsupported frozen verifier factory")
    return StagedQwenVerifier(
        backend,
        max_regions=spec["max_regions"],
        max_retries=spec["max_stage_attempts_including_initial"],
        visual_context=spec["visual_context"],
    )


def _model_call_projection(backend):
    payload_contracts = []
    for call, payload in zip(backend.call_history, backend.payloads):
        region = payload["region"]
        payload_contracts.append(
            {
                "stage": call["stage"],
                "region_id": region["region_id"],
                "proposal_catalog_region_ids": [
                    item["region_id"] for item in payload["proposal_catalog"]
                ],
                "schema": payload["schema"],
                "target_class": payload["target_class"],
                "visual_context": payload["visual_context"],
                "terminal_review_present": "terminal_review" in payload,
            }
        )
    return {
        "count": len(backend.call_history),
        "stages": [item["stage"] for item in backend.call_history],
        "region_ids": [
            item["parsed_output"]["region_id"] for item in backend.call_history
        ],
        "attempt_kinds": [
            item["attempt_kind"] for item in backend.call_history
        ],
        "schema_repair_attempt_count": sum(
            item["attempt_kind"] == "repair" for item in backend.call_history
        ),
        "network_retry_attempt_count": sum(
            item["network_retry_attempt_count"] for item in backend.call_history
        ),
        "payload_contracts": payload_contracts,
    }


def _diagnosis_projection(trace):
    return [
        {
            "region_id": item["region_id"],
            "error_type": item["error_type"],
            "target_view": item["target_view"],
            "audit_checklist": item["audit_checklist"],
            "summary": item["summary"],
        }
        for item in trace["diagnoses"]
    ]


def _legacy_termination_projection(output):
    observation = types.SimpleNamespace(feedback=output)
    stop_reason = _initial_verifier_stop_reason(observation)
    if stop_reason is None:
        returned, errors, action_executed = _execute_verifier_action(
            object(),
            observation,
            loop_index=1,
        )
        if returned is not observation or not errors or action_executed:
            raise AssertionError("plan-null fixture unexpectedly executed an action")
        stop_reason = "verifier_no_executable_action"
    return stop_reason, _episode_outcome(stop_reason)


def _behavior_projection(verifier, backend, output, h0_audit, h0_policy):
    evidence = verifier.last_evidence
    trace = evidence["stage_trace"]
    stop_reason, episode_outcome = _legacy_termination_projection(output)
    planned_action = verifier._planned_action
    counters_after = h0_audit["counters_after"]
    return {
        "region_order": trace["audited_region_ids"],
        "region_records": trace["evidence"],
        "model_calls": _model_call_projection(backend),
        "runtime_diagnoses": _diagnosis_projection(trace),
        "plan": trace["plan"],
        "verifier_output": output.to_dict(),
        "safety_gates": {
            "completion_gate_passed": trace["state_completion_gate_passed"],
            "completion_gate_reason": trace["state_completion_gate_reason"],
            "regional_validation_errors": evidence["regional_validation_errors"],
        },
        "candidate_and_termination": {
            "planned_tool_action": (
                planned_action.to_dict() if planned_action is not None else None
            ),
            "tool_action_attempt_count": 0,
            "candidate_state_count": 0,
            "legacy_stop_reason": stop_reason,
            "episode_outcome": episode_outcome,
        },
        "h0_rounds": {
            "max_verifier_rounds": h0_policy.max_verifier_rounds,
            "verifier_round_count": counters_after["verifier_round_count"],
            "round_two_invocations": int(bool(h0_audit["round_two_invoked"])),
            "round_three_invocations": 0,
        },
    }


class Qwen3VLH0GoldenFixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.receipt = _load_json(RECEIPT_PATH)
        cls.manifest = _load_json(MANIFEST_PATH)

    def test_frozen_h0_cases_match_independent_golden_receipt(self):
        validate_receipt(RECEIPT_PATH, MANIFEST_PATH, repo_root=REPO_ROOT)
        fixture = self.receipt["fixture_definition"]
        for case in self.receipt["fixture_cases"]:
            with self.subTest(case_id=case["case_id"]):
                legacy_backend = _ReceiptAuditBackend(case["backend_audit_response"])
                legacy = _verifier_from_receipt(fixture, legacy_backend)
                legacy_output = legacy.verify(
                    _state_from_receipt(fixture), None, None
                )

                h0_backend = _ReceiptAuditBackend(case["backend_audit_response"])
                h0_staged = _verifier_from_receipt(fixture, h0_backend)
                h0_policy = TerminalReviewPolicy.h0()
                h0 = TerminalReviewVerifier(h0_staged, policy=h0_policy)
                h0_output = h0.verify(_state_from_receipt(fixture), None, None)
                self.assertIsNotNone(h0.last_terminal_review_audit)

                # Each implementation is compared directly with the frozen
                # receipt. Equality between two current implementations is not
                # accepted as an oracle.
                self.assertEqual(
                    _json_value(legacy_output.to_dict()),
                    case["expected_behavior"]["verifier_output"],
                )
                self.assertEqual(
                    _json_value(h0_output.to_dict()),
                    case["expected_behavior"]["verifier_output"],
                )
                self.assertEqual(
                    _json_value(_model_call_projection(h0_backend)),
                    case["expected_behavior"]["model_calls"],
                )
                observed = _behavior_projection(
                    legacy,
                    legacy_backend,
                    legacy_output,
                    h0.last_terminal_review_audit,
                    h0_policy,
                )
                self.assertEqual(
                    _json_value(observed), case["expected_behavior"]
                )
                h0_observed = _behavior_projection(
                    h0_staged,
                    h0_backend,
                    h0_output,
                    h0.last_terminal_review_audit,
                    h0_policy,
                )
                self.assertEqual(
                    _json_value(h0_observed), case["expected_behavior"]
                )

    def test_receipt_or_manifest_tampering_fails_closed(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["golden_fixture_contract"]["receipt_raw_sha256"] = "0" * 64
        with self.assertRaises(GoldenFixtureValidationError):
            validate_receipt_payload(
                self.receipt,
                manifest,
                observed_receipt_raw_sha256=sha256_bytes(RECEIPT_PATH.read_bytes()),
                repo_root=REPO_ROOT,
            )

        manifest = copy.deepcopy(self.manifest)
        manifest["golden_fixture_contract"]["validator_sha256"] = "0" * 64
        with self.assertRaises(GoldenFixtureValidationError):
            validate_receipt_payload(
                self.receipt,
                manifest,
                observed_receipt_raw_sha256=sha256_bytes(RECEIPT_PATH.read_bytes()),
                repo_root=REPO_ROOT,
            )

    def test_coordinated_oracle_and_hash_tampering_fails_closed(self):
        receipt = copy.deepcopy(self.receipt)
        receipt["fixture_cases"][0]["expected_behavior"]["verifier_output"][
            "suggested_action"
        ] = None
        tampered_raw = json.dumps(
            receipt, ensure_ascii=False, indent=2, allow_nan=False
        ).encode("utf-8")
        manifest = copy.deepcopy(self.manifest)
        contract = manifest["golden_fixture_contract"]
        contract["receipt_raw_sha256"] = sha256_bytes(tampered_raw)
        contract["receipt_canonical_sha256"] = canonical_sha256(receipt)
        with self.assertRaises(GoldenFixtureValidationError):
            validate_receipt_payload(
                receipt,
                manifest,
                observed_receipt_raw_sha256=sha256_bytes(tampered_raw),
                repo_root=REPO_ROOT,
            )

        receipt = copy.deepcopy(self.receipt)
        receipt["source_anchor"]["commit"] = "0" * 40
        tampered_raw = json.dumps(
            receipt, ensure_ascii=False, indent=2, allow_nan=False
        ).encode("utf-8")
        manifest = copy.deepcopy(self.manifest)
        contract = manifest["golden_fixture_contract"]
        contract["receipt_raw_sha256"] = sha256_bytes(tampered_raw)
        contract["receipt_canonical_sha256"] = canonical_sha256(receipt)
        with self.assertRaises(GoldenFixtureValidationError):
            validate_receipt_payload(
                receipt,
                manifest,
                observed_receipt_raw_sha256=sha256_bytes(tampered_raw),
                repo_root=REPO_ROOT,
            )


if __name__ == "__main__":
    unittest.main()
