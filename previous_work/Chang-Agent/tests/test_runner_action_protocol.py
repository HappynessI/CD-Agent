import json
import re
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools.run_levir_change_agent import (
    _episode_outcome,
    _execute_direct_verifier_action,
    _execute_verifier_action,
    _final_region_protocol_valid,
    _initial_verifier_stop_reason,
    _outcome_counts,
    _raw_staged_region_protocol_valid,
    _record_run_failure,
)


class DirectEnvironment:
    def __init__(self):
        self.calls = []
        self.trajectory = SimpleNamespace(
            entries=[SimpleNamespace(execution={})]
        )

    def step(self, raw, **kwargs):
        self.calls.append((raw, kwargs) if kwargs else raw)
        return "next_observation", False


class RunnerActionProtocolTest(unittest.TestCase):
    def test_raw_staged_region_replay_binds_scan_proposal_version(self):
        scan_version = "qwen3vl_empty_dual_mask_scan_grid_v1"

        def context(
            region_id: str,
            proposal_version: str | None,
            *,
            trigger: str = "provisional_finish",
        ) -> dict[str, object]:
            plan_null = trigger == "plan_null_after_allowed_repair"
            return {
                "round_index": 2,
                "round_kind": (
                    "plan_null_alternative" if plan_null else "finish_confirmation"
                ),
                "trigger": trigger,
                "fresh_backend_context": True,
                "inherits_hidden_state": False,
                "copies_round_one_raw_response": False,
                "selection_policy_version": (
                    "environment_ordered_terminal_review_v1"
                ),
                "preferred_region_id": region_id if plan_null else None,
                "prompt_version": (
                    "qwen3vl_plan_null_next_unreviewed_region_v1"
                    if plan_null
                    else None
                ),
                "selection_branch": (
                    "next_unreviewed_environment_region" if plan_null else None
                ),
                "review_region_id": region_id,
                "proposal_version": proposal_version,
            }

        def evidence(
            region_id: str = "s0",
            proposal_version: str | None = scan_version,
            *,
            trigger: str = "provisional_finish",
            audit_kind: str | None = None,
            visual_input_scope: str | None = None,
        ) -> dict[str, object]:
            is_scan = re.fullmatch(r"s[0-9]+", region_id) is not None
            proposal = {
                "region_id": region_id,
                "audit_kind": audit_kind or ("scan" if is_scan else "present"),
            }
            if proposal_version is not None:
                proposal["proposal_version"] = proposal_version
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
                    "diagnoses": [{"region_id": region_id}],
                    "plan": {"action": "finish"},
                },
                "backend_calls": [
                    {
                        "backend": "fixture",
                        "stage": "audit",
                        "attempt_kind": "initial",
                        "network_retry_attempt_count": 0,
                        "prompt_sha256": "c" * 64,
                        "raw_response": "{}",
                        "parsed_output": {"region_id": region_id},
                        "visual_input_scope": visual_input_scope or (
                            "authoritative_pair_rgb_only_scan"
                            if is_scan
                            else "standard"
                        ),
                        "terminal_review": context(
                            region_id,
                            proposal_version,
                            trigger=trigger,
                        ),
                    }
                ],
                "terminal_review_round": {"schema_valid": True},
            }

        scan = evidence()
        self.assertTrue(
            _raw_staged_region_protocol_valid(scan, require_round_two=True)
        )
        self.assertTrue(
            _raw_staged_region_protocol_valid(
                evidence("s1", scan_version),
                require_round_two=True,
            )
        )
        ordinary = evidence("r0", None)
        self.assertTrue(
            _raw_staged_region_protocol_valid(ordinary, require_round_two=True)
        )
        for ordinary_region_id in ("s", "s-1", "sx"):
            with self.subTest(ordinary_region_id=ordinary_region_id):
                self.assertTrue(
                    _raw_staged_region_protocol_valid(
                        evidence(ordinary_region_id, None),
                        require_round_two=True,
                    )
                )

        nonnumeric_scan = evidence("sx", scan_version)
        self.assertFalse(
            _raw_staged_region_protocol_valid(
                nonnumeric_scan,
                require_round_two=True,
            )
        )
        for mutation in (
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
                "proposal_version", None
            ),
            lambda item: item["backend_calls"][0].pop("backend"),
            lambda item: item["backend_calls"][0].pop("visual_input_scope"),
            lambda item: item["backend_calls"][0].pop("raw_response"),
            lambda item: item["backend_calls"][0].__setitem__(
                "visual_input_scope", "standard"
            ),
            lambda item: item["backend_calls"][0].__setitem__(
                "parse_error", "cannot coexist with parsed output"
            ),
        ):
            with self.subTest(mutation=mutation):
                malformed = json.loads(json.dumps(scan))
                mutation(malformed)
                self.assertFalse(
                    _raw_staged_region_protocol_valid(
                        malformed,
                        require_round_two=True,
                    )
                )

        for mutation in (
            lambda item: item["stage_trace"]["evidence"][0].__setitem__(
                "audit_kind", "scan"
            ),
            lambda item: item["stage_trace"]["evidence"][0].__setitem__(
                "proposal_version", scan_version
            ),
            lambda item: item["backend_calls"][0].__setitem__(
                "visual_input_scope", "authoritative_pair_rgb_only_scan"
            ),
        ):
            with self.subTest(ordinary_mutation=mutation):
                malformed = json.loads(json.dumps(ordinary))
                mutation(malformed)
                self.assertFalse(
                    _raw_staged_region_protocol_valid(
                        malformed,
                        require_round_two=True,
                    )
                )

        repaired = json.loads(json.dumps(scan))
        initial = repaired["backend_calls"][0]
        initial.pop("parsed_output")
        initial["parse_error"] = "schema parse failed"
        repair = json.loads(json.dumps(initial))
        repair["attempt_kind"] = "repair"
        repair["prompt_sha256"] = "d" * 64
        repair.pop("parse_error")
        repair["parsed_output"] = {"region_id": "s0"}
        repaired["backend_calls"].append(repair)
        self.assertTrue(
            _raw_staged_region_protocol_valid(repaired, require_round_two=True)
        )

        success_then_repair = json.loads(json.dumps(scan))
        appended = json.loads(json.dumps(success_then_repair["backend_calls"][0]))
        appended["attempt_kind"] = "repair"
        appended["prompt_sha256"] = "d" * 64
        success_then_repair["backend_calls"].append(appended)
        self.assertFalse(
            _raw_staged_region_protocol_valid(
                success_then_repair,
                require_round_two=True,
            )
        )

        h0_with_context = json.loads(json.dumps(ordinary))
        self.assertFalse(
            _raw_staged_region_protocol_valid(
                h0_with_context,
                require_round_two=False,
            )
        )
        h0_with_context["backend_calls"][0].pop("terminal_review")
        self.assertTrue(
            _raw_staged_region_protocol_valid(
                h0_with_context,
                require_round_two=False,
            )
        )

        plan_null = evidence(trigger="plan_null_after_allowed_repair")
        self.assertTrue(
            _raw_staged_region_protocol_valid(plan_null, require_round_two=True)
        )
        plan_null["backend_calls"][0]["terminal_review"][
            "selection_branch"
        ] = "first_unresolved_environment_region"
        self.assertFalse(
            _raw_staged_region_protocol_valid(plan_null, require_round_two=True)
        )

        reviewed_block = {
            "terminal_review": {
                "eligible": True,
                "round_two_invoked": True,
                "round_two_count": 1,
                "round_index": 2,
                "round_one_evidence": {"backend_calls": []},
                "round_two_evidence": scan,
            }
        }
        environment = SimpleNamespace(
            trajectory=SimpleNamespace(
                entries=[
                    SimpleNamespace(
                        execution={"verifier_evidence": reviewed_block}
                    )
                ]
            )
        )
        policy = SimpleNamespace(terminal_review=True)
        self.assertTrue(_final_region_protocol_valid(environment, policy))

        later_unreviewed = json.loads(json.dumps(ordinary))
        later_unreviewed["backend_calls"][0].pop("terminal_review")
        later_unreviewed.pop("terminal_review_round")
        environment.trajectory.entries.append(
            SimpleNamespace(
                execution={"verifier_evidence": later_unreviewed}
            )
        )
        self.assertFalse(_final_region_protocol_valid(environment, policy))

    def test_initial_invalid_verifier_stops_before_agent_actions(self):
        observation = SimpleNamespace(
            feedback=SimpleNamespace(verifier_valid=False)
        )

        self.assertEqual(
            _initial_verifier_stop_reason(observation), "initial_verifier_invalid"
        )
        self.assertIsNone(
            _initial_verifier_stop_reason(
                SimpleNamespace(feedback=SimpleNamespace(verifier_valid=True))
            )
        )

    def test_initial_authorized_finish_stops_before_agent_actions(self):
        observation = SimpleNamespace(
            feedback=SimpleNamespace(
                verifier_valid=True,
                accept=True,
                stop=True,
                comparison="initial",
                error_type="none",
            )
        )

        self.assertEqual(
            _initial_verifier_stop_reason(observation),
            "initial_verifier_authorized_finish",
        )

    def test_stop_reasons_are_separate_from_execution_status(self):
        self.assertEqual(
            _episode_outcome("verifier_authorized_finish"), "converged"
        )
        self.assertEqual(
            _episode_outcome("initial_verifier_authorized_finish"), "converged"
        )
        self.assertEqual(
            _episode_outcome("max_steps_reached"), "budget_exhausted"
        )
        self.assertEqual(
            _episode_outcome("initial_verifier_invalid"), "verifier_invalid"
        )
        self.assertEqual(
            _episode_outcome("verifier_no_executable_action"),
            "no_executable_action",
        )
        counts = _outcome_counts(
            {
                "a": {"episode_outcome": "converged"},
                "b": {"episode_outcome": "budget_exhausted"},
                "c": {"episode_outcome": "budget_exhausted"},
            }
        )
        self.assertEqual(
            counts, {"budget_exhausted": 2, "converged": 1}
        )

    def test_failed_run_manifest_is_explicit_and_redacted(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = {"status": "running"}

            _record_run_failure(
                output,
                manifest,
                RuntimeError("provider rejected sk-should-not-leak"),
                time.monotonic(),
            )

            failure = json.loads(
                (output / "failure.json").read_text(encoding="utf-8")
            )
            rendered = (output / "run_manifest.md").read_text(
                encoding="utf-8"
            )
        self.assertEqual(failure["execution_status"], "failed")
        self.assertEqual(manifest["status"], "failed")
        self.assertNotIn("sk-should-not-leak", json.dumps(failure))
        self.assertNotIn("sk-should-not-leak", rendered)

    def test_direct_verifier_executes_its_normalized_point_plan(self):
        environment = DirectEnvironment()
        observation = SimpleNamespace(
            feedback=SimpleNamespace(
                verifier_valid=True,
                target_view="t2",
                suggested_action="negative_point",
                error_region=(250, 300, 250, 300),
            )
        )

        returned, errors, executed = _execute_direct_verifier_action(
            environment, observation, loop_index=2
        )

        self.assertEqual(returned, "next_observation")
        self.assertEqual(errors, [])
        self.assertTrue(executed)
        self.assertEqual(
            environment.calls,
            ['{"target_view":"t2","action":"negative_point","coordinate":[250,300]}'],
        )
        self.assertEqual(
            environment.trajectory.entries[-1].execution["action_generation"]["source"],
            "direct_verifier_full_context",
        )

    def test_empty_mask_scan_box_is_marked_as_fallback(self):
        environment = DirectEnvironment()
        environment.verifier = SimpleNamespace(
            last_evidence={
                "stage_trace": {"plan": {"region_id": "s3", "action": "box"}}
            }
        )
        environment.state = SimpleNamespace(
            evidence={
                "verifier_mask_facts": {
                    "proposal_config": {"empty_dual_mask_scan": True}
                }
            }
        )
        observation = SimpleNamespace(
            feedback=SimpleNamespace(
                verifier_valid=True,
                target_view="t2",
                suggested_action="box",
                error_region=(667, 667, 1000, 1000),
            )
        )

        returned, errors, executed = _execute_verifier_action(
            environment, observation, loop_index=1
        )

        self.assertEqual(returned, "next_observation")
        self.assertEqual(errors, [])
        self.assertTrue(executed)
        generation = environment.trajectory.entries[-1].execution[
            "action_generation"
        ]
        self.assertTrue(generation["empty_mask_scan_fallback"])
        self.assertEqual(
            generation["source"],
            "verifier_empty_mask_scan_environment_box",
        )

    def test_staged_point_passes_authorized_region_outside_raw_action(self):
        environment = DirectEnvironment()
        environment.verifier = SimpleNamespace(
            last_evidence={
                "stage_trace": {
                    "plan": {
                        "region_id": "r4",
                        "action": "positive_point",
                    }
                }
            }
        )
        observation = SimpleNamespace(
            feedback=SimpleNamespace(
                verifier_valid=True,
                target_view="t2",
                suggested_action="positive_point",
                error_region=(300, 400, 300, 400),
            )
        )

        returned, errors, executed = _execute_verifier_action(
            environment, observation, loop_index=3
        )

        self.assertEqual(returned, "next_observation")
        self.assertEqual(errors, [])
        self.assertTrue(executed)
        raw, kwargs = environment.calls[0]
        self.assertEqual(
            raw,
            '{"target_view":"t2","action":"positive_point","coordinate":[300,400]}',
        )
        self.assertNotIn("region", raw)
        self.assertEqual(kwargs, {"authorized_region_id": "r4"})
        self.assertEqual(
            environment.trajectory.entries[-1].execution["action_generation"][
                "authorized_region_id"
            ],
            "r4",
        )

    def test_staged_box_passes_authorized_region_outside_raw_action(self):
        environment = DirectEnvironment()
        environment.verifier = SimpleNamespace(
            last_evidence={
                "stage_trace": {
                    "plan": {"region_id": "s3", "action": "box"}
                }
            }
        )
        observation = SimpleNamespace(
            feedback=SimpleNamespace(
                verifier_valid=True,
                target_view="t2",
                suggested_action="box",
                error_region=(100, 200, 400, 500),
            )
        )

        returned, errors, executed = _execute_verifier_action(
            environment, observation, loop_index=5
        )

        self.assertEqual(returned, "next_observation")
        self.assertEqual(errors, [])
        self.assertTrue(executed)
        raw, kwargs = environment.calls[0]
        self.assertEqual(
            raw,
            '{"target_view":"t2","action":"box","box":[100,200,400,500]}',
        )
        self.assertNotIn("region", raw)
        self.assertEqual(kwargs, {"authorized_region_id": "s3"})
        self.assertEqual(
            environment.trajectory.entries[-1].execution["action_generation"][
                "authorized_region_id"
            ],
            "s3",
        )

    def test_staged_action_without_region_id_fails_closed(self):
        environment = DirectEnvironment()
        environment.verifier = SimpleNamespace(last_evidence={})
        observation = SimpleNamespace(
            feedback=SimpleNamespace(
                verifier_valid=True,
                target_view="t2",
                suggested_action="negative_point",
                error_region=(300, 400, 300, 400),
            )
        )

        returned, errors, executed = _execute_verifier_action(
            environment, observation, loop_index=6
        )

        self.assertIs(returned, observation)
        self.assertFalse(executed)
        self.assertEqual(environment.calls, [])
        self.assertIn("lacks an authorized", errors[0]["error"])

    def test_rollback_replan_region_takes_precedence(self):
        environment = DirectEnvironment()
        environment.verifier = SimpleNamespace(
            last_evidence={
                "stage_trace": {"plan": {"region_id": "r0"}},
                "replan": {"selected_region_id": "r2"},
            }
        )
        observation = SimpleNamespace(
            feedback=SimpleNamespace(
                verifier_valid=True,
                target_view="t1",
                suggested_action="negative_point",
                error_region=(100, 200, 100, 200),
            )
        )

        _execute_verifier_action(environment, observation, loop_index=4)

        self.assertEqual(
            environment.calls[0][1], {"authorized_region_id": "r2"}
        )


if __name__ == "__main__":
    unittest.main()
