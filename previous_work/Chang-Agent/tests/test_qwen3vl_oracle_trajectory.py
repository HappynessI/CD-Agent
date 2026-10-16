from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from change_agent.adapters.omniovcd_adapter import MaskPairProcessor, OmniOVCDAdapter
from change_agent.environment import ChangeAgentEnvironment
from change_agent.executor import ActionExecutor
from change_agent.state import AgentAction, ChangeState


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_qwen3vl_oracle_trajectory",
    ROOT / "tools" / "run_qwen3vl_oracle_trajectory.py",
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


def proposal(
    region_id: str,
    *,
    kind: str = "present",
    seed: tuple[int, int] = (10, 10),
    t1_white: bool = True,
    t2_white: bool = False,
    box: tuple[int, int, int, int] = (6, 6, 14, 14),
) -> dict:
    return {
        "region_id": region_id,
        "audit_kind": kind,
        "component_seed_pixels": list(seed),
        "box_pixels": list(box),
        "component_area": 1,
        "component_seed_t1_mask_white": t1_white,
        "component_seed_t2_mask_white": t2_white,
    }


def state(change: np.ndarray) -> ChangeState:
    image = np.zeros((*change.shape, 3), dtype=np.uint8)
    empty = np.zeros_like(change, dtype=bool)
    return ChangeState(image, image, "building", empty, empty, change)


def runtime_episode(t1_mask: np.ndarray) -> RUNNER.RuntimeEpisode:
    image = np.zeros((*t1_mask.shape, 3), dtype=np.uint8)
    identity = RUNNER.FileIdentity(Path("/unused.png"), "0" * 64)
    return RUNNER.RuntimeEpisode(
        episode_id="sample",
        sample_id="sample",
        split="train",
        parent_id="parent",
        t1_image=image,
        t2_image=image,
        t1_source=identity,
        t2_source=identity,
        crop_xyxy_exclusive=(0, 0, 256, 256),
        query="building",
        rgb_states={"t1": "background", "t2": "building"},
        rgb_semantic_evidence="T1 is background and T2 contains the building.",
        rgb_semantic_source="unit fixture",
        rgb_semantic_status="provisional_non_blinded",
        rgb_states_by_region={},
        rgb_states_by_step={},
    )


class FakePointBackend:
    def __init__(self, *, no_op: bool = False):
        self.no_op = no_op
        self.calls: list[dict] = []
        self.last_evidence: dict = {}

    def refine(self, image, initial_mask, coordinate, is_positive, click_history=()):
        self.calls.append(
            {
                "image_shape": tuple(np.asarray(image).shape),
                "mask_shape": tuple(np.asarray(initial_mask).shape),
                "mask_sha256": RUNNER.mask_sha256(
                    np.asarray(initial_mask, dtype=bool)
                ),
                "coordinate": coordinate,
                "is_positive": is_positive,
                "history": tuple(click_history),
            }
        )
        result = np.array(initial_mask, dtype=bool, copy=True)
        if not self.no_op:
            x, y = coordinate
            result[y, x] = bool(is_positive)
        self.last_evidence = {
            "worker_mode": "point",
            "worker_report": {"status": "success", "mode": "point"},
        }
        return result


class FakeBoxBackend:
    def __init__(self):
        self.last_evidence: dict = {}

    def segment_box(self, image, box_cxcywh_normalized, query):
        self.last_evidence = {
            "worker_mode": "box",
            "worker_report": {"status": "success", "mode": "box"},
        }
        return np.zeros(np.asarray(image).shape[:2], dtype=bool)


class FakeRuntimeFactory:
    """Uses the real Environment/Executor while keeping unit tests model-free."""

    def __init__(self, initial_t1: np.ndarray, *, canonical_no_op: bool = False):
        self.initial_t1 = np.array(initial_t1, dtype=bool, copy=True)
        self.initial_t2 = np.zeros_like(self.initial_t1)
        self.canonical_no_op = canonical_no_op
        self.verifiers: list[RUNNER.NoQwenAcceptVerifier] = []
        self.runtime_inputs: list[RUNNER.RuntimeEpisode] = []
        self.point_calls: list[dict] = []

    @property
    def qwen_calls(self) -> int:
        return sum(item.qwen_calls for item in self.verifiers)

    def _environment(self, initializer, *, max_steps: int, no_op: bool):
        point = FakePointBackend(no_op=no_op)
        box = FakeBoxBackend()
        verifier = RUNNER.NoQwenAcceptVerifier()
        self.verifiers.append(verifier)
        backend = OmniOVCDAdapter(initializer, box.segment_box, MaskPairProcessor())
        environment = ChangeAgentEnvironment(
            backend,
            ActionExecutor(point, box),
            verifier,
            max_steps=max_steps,
            require_tool_before_finish=False,
            empty_mask_scan_grid_size=3,
            enable_verifier_regions=True,
        )
        original_refine = point.refine

        def recording_refine(*args, **kwargs):
            result = original_refine(*args, **kwargs)
            self.point_calls.extend(point.calls[-1:])
            return result

        point.refine = recording_refine
        return environment

    def create_canonical_environment(self, episode, artifact_dir, max_steps):
        del artifact_dir
        self.runtime_inputs.append(episode)

        def initializer(t1_image, t2_image, query):
            del t1_image, t2_image, query
            return self.initial_t1, self.initial_t2, {"initializer": "fake_fresh_sam3"}

        return self._environment(
            initializer, max_steps=max_steps, no_op=self.canonical_no_op
        )

    def create_branch_environment(self, episode, accepted_state, artifact_dir):
        del artifact_dir
        self.runtime_inputs.append(episode)
        t1 = np.array(accepted_state.t1_mask, copy=True)
        t2 = np.array(accepted_state.t2_mask, copy=True)

        def initializer(t1_image, t2_image, query):
            del t1_image, t2_image, query
            return t1, t2, {"initializer": "fake_accepted_branch"}

        return self._environment(initializer, max_steps=1, no_op=False)


class OracleTrajectoryUnitTests(unittest.TestCase):
    def test_production_runtime_config_is_an_explicit_gt_free_allowlist(self):
        manifest = {
            "samples": {
                "sample": {
                    "source_parent": {"label": {"path": "/secret-label.png"}},
                    "crop": {"gt_pixels": 123},
                }
            },
            "tools_and_models": {
                "workspace_root": "/workspace",
                "segagent_python": {"path": "/seg-python", "sha256": "a" * 64},
                "omniovcd_python": {"path": "/sam-python", "sha256": "b" * 64},
                "simpleclick_checkpoint": {"path": "/click.pth", "sha256": "c" * 64},
                "sam3_checkpoint": {"path": "/sam3.pt", "sha256": "d" * 64},
                "sam3_bpe": {"path": "/bpe.gz", "sha256": "e" * 64},
                "qwen_base_model_disabled": {"path": "/qwen"},
                "external_source_repositories": [{"path": "/external"}],
            },
            "execution_protocol": {
                "seed": 42,
                "query": "building",
                "sam3_resolution": 1008,
                "qwen_policy": {"enabled": False},
            },
            "matching": {"mode": "overlap_presence"},
            "proposal_config": {"max_regions": 1},
            "hard_gate_config": {"max_selection_area_delta": 0.25},
            "metric_and_success_contract": {"endpoint_success": {"minimum_iou": 0.9}},
        }
        runtime = RUNNER.build_runtime_factory_config(manifest)
        self.assertEqual(
            set(runtime),
            {
                "tools_and_models",
                "execution_protocol",
                "matching",
                "proposal_config",
                "hard_gate_config",
                "authorized_negative_point_input_scope",
            },
        )
        self.assertNotIn("query", runtime["execution_protocol"])
        self.assertNotIn("qwen_policy", runtime["execution_protocol"])
        self.assertNotIn("qwen_base_model_disabled", runtime["tools_and_models"])
        self.assertNotIn("external_source_repositories", runtime["tools_and_models"])
        encoded = json.dumps(runtime, sort_keys=True)
        self.assertNotIn("secret-label", encoded)
        self.assertNotIn("gt_pixels", encoded)
        self.assertNotIn("endpoint_success", encoded)

    def test_action_enumeration_covers_useful_executor_surface(self):
        actions = RUNNER.enumerate_live_actions(
            "episode",
            1,
            [
                proposal("r0", kind="present"),
                proposal("r1", kind="mixed", seed=(30, 30), box=(26, 26, 34, 34)),
                proposal(
                    "s0",
                    kind="scan",
                    seed=(50, 50),
                    t1_white=False,
                    t2_white=False,
                    box=(40, 40, 60, 60),
                ),
            ],
        )
        self.assertEqual(
            [(item.action.action, item.action.target_view) for item in actions],
            [
                ("negative_point", "t1"),
                ("positive_point", "t1"),
                ("box", "t1"),
                ("positive_point", "t2"),
                ("box", "t2"),
                ("negative_point", "t1"),
                ("positive_point", "t1"),
                ("box", "t1"),
                ("positive_point", "t2"),
                ("box", "t2"),
                ("positive_point", "t1"),
                ("box", "t1"),
                ("positive_point", "t2"),
                ("box", "t2"),
            ],
        )
        self.assertEqual(actions[0].reachability_tier, "runtime_planner_primary")
        self.assertEqual(actions[3].reachability_tier, "executor_reachable_complement")
        self.assertEqual(actions[2].reachability_tier, "executor_reachable_box")
        self.assertEqual(actions[-4].reachability_tier, "executor_reachable_scan_point")
        self.assertTrue(
            all(item.action.to_dict()["coordinate_space"] == "pixel" for item in actions)
        )

    def test_non_unique_non_scan_proposal_still_has_executor_actions(self):
        actions = RUNNER.enumerate_live_actions(
            "episode",
            1,
            [
                proposal("r0", t1_white=True, t2_white=True),
                proposal("r1", t1_white=False, t2_white=False),
            ],
        )
        self.assertEqual(
            [(item.action.action, item.action.target_view) for item in actions],
            [
                ("negative_point", "t1"),
                ("positive_point", "t1"),
                ("box", "t1"),
                ("negative_point", "t2"),
                ("positive_point", "t2"),
                ("box", "t2"),
                ("positive_point", "t1"),
                ("box", "t1"),
                ("positive_point", "t2"),
                ("box", "t2"),
            ],
        )

    def test_scan_never_enumerates_negative_even_if_white_flag_drifts(self):
        actions = RUNNER.enumerate_live_actions(
            "episode",
            1,
            [
                proposal(
                    "scan",
                    kind="scan",
                    t1_white=True,
                    t2_white=True,
                )
            ],
        )
        self.assertEqual(
            [(item.action.action, item.action.target_view) for item in actions],
            [
                ("positive_point", "t1"),
                ("box", "t1"),
                ("positive_point", "t2"),
                ("box", "t2"),
            ],
        )

    def test_action_inventory_rejects_unknown_audit_kind(self):
        with self.assertRaisesRegex(ValueError, "unsupported audit_kind"):
            RUNNER.enumerate_live_actions(
                "episode",
                1,
                [proposal("unknown", kind="future_kind")],
            )

    def test_runner_frozen_contract_rejects_missing_and_wrong_consumed_fields(self):
        config_path = (
            ROOT / "configs" / "qwen3vl_oracle_trajectory_pilot_0803.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        RUNNER._validate_frozen_runtime_contract(config)
        mutations = (
            lambda value: value["candidate_and_replay_budget"].pop(
                "max_candidate_actions_per_step"
            ),
            lambda value: value["execution_protocol"].update({"max_steps": 2}),
            lambda value: value["hard_gate_config"].update(
                {"max_point_image_change_ratio": 0.01}
            ),
            lambda value: value["proposal_config"].update({"max_regions": 2}),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                changed = copy.deepcopy(config)
                mutate(changed)
                with self.assertRaises(ValueError):
                    RUNNER._validate_frozen_runtime_contract(changed)

    def test_reachability_tiers_match_current_deterministic_planner(self):
        actions = RUNNER.enumerate_live_actions(
            "episode",
            1,
            [
                proposal("present", kind="present"),
                proposal("missing", kind="missing"),
                proposal("mixed", kind="mixed"),
                proposal(
                    "scan",
                    kind="scan",
                    t1_white=False,
                    t2_white=False,
                ),
            ],
        )
        tiers = {
            (item.region_id, item.action.action, item.action.target_view):
            item.reachability_tier
            for item in actions
        }
        self.assertEqual(
            tiers[("present", "negative_point", "t1")],
            "runtime_planner_primary",
        )
        self.assertEqual(
            tiers[("present", "positive_point", "t2")],
            "executor_reachable_complement",
        )
        self.assertEqual(
            tiers[("missing", "positive_point", "t2")],
            "runtime_planner_primary",
        )
        self.assertEqual(
            tiers[("missing", "negative_point", "t1")],
            "executor_reachable_extended_negative",
        )
        self.assertTrue(
            all(
                tier != "runtime_planner_primary"
                for (region_id, _action, _view), tier in tiers.items()
                if region_id == "mixed"
            )
        )
        self.assertEqual(
            tiers[("scan", "box", "t1")], "runtime_planner_primary"
        )
        self.assertEqual(
            tiers[("scan", "positive_point", "t1")],
            "executor_reachable_scan_point",
        )

    def test_empty_gt_metrics_use_exact_empty_scores(self):
        empty = np.zeros((4, 4), dtype=bool)
        metrics = RUNNER.mask_metrics(empty, empty)
        self.assertEqual(metrics["pixel_error"], 0)
        self.assertEqual(metrics["iou"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)

    def test_endpoint_gate_separates_local_progress_from_target_recovery(self):
        nonempty_partial = {
            "gt_pixels": 100,
            "iou": 0.89,
            "f1": 0.95,
            "fp": 0,
        }
        nonempty_pass = dict(nonempty_partial, iou=0.90, f1=18.0 / 19.0)
        empty_fail = {"gt_pixels": 0, "iou": 0.0, "f1": 0.0, "fp": 65}
        empty_pass = dict(empty_fail, fp=64)
        self.assertFalse(RUNNER.endpoint_gate(nonempty_partial)["passed"])
        self.assertTrue(RUNNER.endpoint_gate(nonempty_pass)["passed"])
        self.assertFalse(RUNNER.endpoint_gate(empty_fail)["passed"])
        self.assertTrue(RUNNER.endpoint_gate(empty_pass)["passed"])
        prediction = np.zeros((10, 10), dtype=bool)
        reference = np.zeros_like(prediction)
        prediction.flat[:64] = True
        self.assertTrue(
            RUNNER.endpoint_gate(RUNNER.mask_metrics(prediction, reference))["passed"]
        )

    def test_rgb_semantic_gate(self):
        negative = AgentAction("t1", "negative_point", coordinate=(1, 1))
        positive = AgentAction("t2", "positive_point", coordinate=(1, 1))
        box = AgentAction("t2", "box", box=(0, 0, 2, 2))
        self.assertTrue(RUNNER.rgb_semantic_gate(negative, "background")["passed"])
        self.assertFalse(RUNNER.rgb_semantic_gate(negative, "building")["passed"])
        self.assertTrue(RUNNER.rgb_semantic_gate(positive, "building")["passed"])
        self.assertTrue(RUNNER.rgb_semantic_gate(box, "building")["passed"])

    def test_offline_top4_uses_error_pixels_then_candidate_id(self):
        mask = np.zeros((256, 256), dtype=bool)
        ground_truth = np.zeros_like(mask)
        actions = []
        for index, pixels in enumerate((1, 5, 3, 5, 2)):
            x1 = index * 20
            box = (x1, 0, x1 + 9, 9)
            mask[0, x1 : x1 + pixels] = True
            item = proposal(
                f"s{index}",
                kind="scan",
                seed=(x1 + 4, 4),
                t1_white=False,
                t2_white=False,
                box=box,
            )
            actions.append(
                RUNNER.CandidateAction(
                    f"candidate-{index}",
                    f"s{index}",
                    "runtime_planner_primary",
                    AgentAction("t2", "box", box=box),
                    item,
                )
            )
        source = state(mask)
        episode = runtime_episode(mask)
        catalog, selected = RUNNER.freeze_candidate_shortlist(
            episode, 1, source, ground_truth, actions, 4
        )
        self.assertEqual(
            [item.candidate_id for item in selected],
            ["candidate-1", "candidate-3", "candidate-2", "candidate-4"],
        )
        self.assertEqual(sum(item["selected_for_execution"] for item in catalog), 4)

    def test_strict_step_gate_rejects_lower_error_when_iou_falls(self):
        gt = np.zeros((3, 3), dtype=bool)
        gt[0, :2] = True
        current = np.zeros_like(gt)
        current[0, 0] = True
        current[1, :3] = True
        candidate = np.zeros_like(gt)
        records = [
            {
                "candidate_id": "candidate",
                "hard_gate": {"passed": True},
                "rgb_semantic_gate": {"passed": True},
            }
        ]
        _, ranked = RUNNER.score_completed_candidates(
            state(current), records, [state(candidate)], gt
        )
        self.assertGreater(records[0]["metric_delta"]["pixel_error_reduction"], 0)
        self.assertFalse(records[0]["per_step_metric_gate"]["passed"])
        self.assertEqual(ranked, [])

    def test_canonical_replay_hash_mismatch_fails_closed(self):
        initial = np.zeros((256, 256), dtype=bool)
        # The frozen empty-GT endpoint permits up to 64 false positives.  Start
        # just outside that endpoint so the branch's one-pixel correction is
        # selected and the deliberately no-op canonical replay is exercised.
        initial[10, 10:75] = True
        episode = runtime_episode(initial)
        pilot = RUNNER.PilotEpisode(
            episode,
            RUNNER.LabelSource(
                RUNNER.FileIdentity(Path("/unused-label.png"), "1" * 64),
                (0, 0, 256, 256),
            ),
        )
        factory = FakeRuntimeFactory(initial, canonical_no_op=True)
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "canonical replay masks differ"):
                RUNNER.run_episode(
                    pilot,
                    factory,
                    Path(temporary),
                    label_loader=lambda source: np.zeros((256, 256), dtype=bool),
                )

    def test_episode_budget_contract_rejects_drift(self):
        initial = np.zeros((256, 256), dtype=bool)
        episode = runtime_episode(initial)
        pilot = RUNNER.PilotEpisode(
            episode,
            RUNNER.LabelSource(
                RUNNER.FileIdentity(Path("/unused-label.png"), "1" * 64),
                (0, 0, 256, 256),
            ),
        )
        cases = (
            ({"max_steps": 2}, "max_steps must equal 3"),
            ({"max_candidates_per_step": 3}, "max_candidates_per_step must equal 4"),
            ({"max_tool_executions": 23}, "max_tool_executions must equal 24"),
        )
        for kwargs, pattern in cases:
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as temporary:
                with self.assertRaisesRegex(ValueError, pattern):
                    RUNNER.run_episode(
                        pilot,
                        FakeRuntimeFactory(initial),
                        Path(temporary),
                        label_loader=lambda source: np.zeros((256, 256), dtype=bool),
                        **kwargs,
                    )

    def test_three_step_cap_and_gt_free_runtime_surface(self):
        initial = np.zeros((256, 256), dtype=bool)
        for y, x in ((20, 20), (70, 70), (120, 120), (190, 190)):
            initial[y, x] = True
        episode = runtime_episode(initial)
        pilot = RUNNER.PilotEpisode(
            episode,
            RUNNER.LabelSource(
                RUNNER.FileIdentity(Path("/unused-label.png"), "2" * 64),
                (0, 0, 256, 256),
            ),
        )
        factory = FakeRuntimeFactory(initial)
        label_calls = []

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)

            def label_loader(source):
                freeze_path = (
                    output
                    / "episodes/sample/catalog_freeze_before_gt.json"
                )
                self.assertTrue(freeze_path.is_file())
                frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
                self.assertEqual(frozen["gt_access_count_before"], 0)
                self.assertFalse(frozen["contains_gt_derived_fields"])
                self.assertTrue(frozen["actions"])
                label_calls.append(source)
                return np.zeros((256, 256), dtype=bool)

            record = RUNNER.run_episode(
                pilot,
                factory,
                output,
                endpoint_empty_gt_max_fp=0,
                label_loader=label_loader,
            )
            freeze = json.loads(
                (output / "episodes/sample/catalog_freeze_before_gt.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertEqual(len(record["steps"]), 3)
        self.assertEqual(record["termination"]["reason"], "max_steps")
        self.assertEqual(record["episode_status"], "locally_reachable_only")
        self.assertTrue(record["local_reachability"]["passed"])
        self.assertFalse(record["endpoint_reachability"]["final_gate"]["passed"])
        self.assertEqual(len(label_calls), 1)
        self.assertEqual(freeze["gt_access_count_before"], 0)
        self.assertFalse(freeze["contains_gt_derived_fields"])
        self.assertEqual(record["gt_access_audit"]["access_count"], 1)
        self.assertFalse(record["gt_access_audit"]["label_passed_to_runtime_factory"])
        self.assertEqual(factory.qwen_calls, 0)
        self.assertTrue(factory.runtime_inputs)
        self.assertTrue(
            all("label" not in vars(item) and "ground_truth" not in vars(item) for item in factory.runtime_inputs)
        )
        self.assertTrue(factory.point_calls)
        self.assertTrue(
            all(
                set(call)
                == {
                    "image_shape",
                    "mask_shape",
                    "mask_sha256",
                    "coordinate",
                    "is_positive",
                    "history",
                }
                for call in factory.point_calls
            )
        )
        self.assertTrue(all(call["history"] == () for call in factory.point_calls))
        negative_calls = [
            call for call in factory.point_calls if not call["is_positive"]
        ]
        self.assertEqual(len(negative_calls), 6)
        replay_sources = []
        for index in range(0, len(negative_calls), 2):
            branch_call, canonical_call = negative_calls[index : index + 2]
            self.assertEqual(branch_call["coordinate"], canonical_call["coordinate"])
            self.assertEqual(branch_call["mask_sha256"], canonical_call["mask_sha256"])
            replay_sources.append(branch_call["mask_sha256"])
        self.assertEqual(len(set(replay_sources)), 3)

    def test_episode_stops_when_absolute_empty_gt_endpoint_is_reached(self):
        initial = np.zeros((256, 256), dtype=bool)
        initial[10:15, 10:23] = True  # 65 FP; one real executor click reaches 64.
        episode = runtime_episode(initial)
        pilot = RUNNER.PilotEpisode(
            episode,
            RUNNER.LabelSource(
                RUNNER.FileIdentity(Path("/unused-label.png"), "3" * 64),
                (0, 0, 256, 256),
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            record = RUNNER.run_episode(
                pilot,
                FakeRuntimeFactory(initial),
                Path(temporary),
                label_loader=lambda source: np.zeros((256, 256), dtype=bool),
            )
        self.assertEqual(record["episode_status"], "endpoint_success")
        self.assertEqual(record["termination"]["reason"], "endpoint_reached")
        self.assertEqual(len(record["steps"]), 1)
        self.assertEqual(record["final_metrics"]["fp"], 64)
        self.assertEqual(record["endpoint_reachability"]["success_step"], 1)
        self.assertFalse(record["endpoint_reachability"]["exact_recovery"])


class SlurmSubmissionLifecycleContractTests(unittest.TestCase):
    def test_submitter_freezes_exact_two_file_B_and_uses_retryable_attempts(self):
        text = (ROOT / "tools" / "submit_qwen3vl_oracle_trajectory.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("verify_freeze_delta", text)
        self.assertIn("freeze_B_delta_must_be_exact_config_and_0803_doc", text)
        self.assertIn("reserve_next_attempt", text)
        self.assertNotIn('ATTEMPT="$CAMPAIGN_ROOT/attempt-0001"', text)
        self.assertIn("flock -n", text)

        function_body = text.split("verify_freeze_delta() {", 1)[1].split(
            "\n}\n\nverify_exact_freeze_state()", 1
        )[0]
        self.assertEqual(function_body.count("if git -C \"$ROOT\" diff --quiet"), 2)
        self.assertIn("return 0", function_body)
        self.assertNotIn(
            "&& { ERROR_TEXT=freeze_B_0803_doc_delta_missing",
            function_body,
        )

    def test_submitter_cancels_orphans_and_relocates_unsubmitted_formal_attempt(self):
        text = (ROOT / "tools" / "submit_qwen3vl_oracle_trajectory.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("cancel_submitted_jobs", text)
        self.assertIn("cancel_and_wait_one", text)
        self.assertIn("recover_ambiguous_submission_by_name", text)
        self.assertEqual(
            text.count('recover_ambiguous_submission_by_name "$'),
            3,
        )
        self.assertIn('"ambiguous_sbatch": {', text)
        self.assertIn('&& "$AMBIGUOUS_SUBMISSION" == false', text)
        self.assertIn("relocate_unsubmitted_formal_attempt", text)
        self.assertIn("trap submission_finish EXIT", text)

    def test_validation_worker_binds_exact_b_and_reports_scientific_result(self):
        text = (
            ROOT / "tools" / "run_qwen3vl_oracle_trajectory_validate.slurm"
        ).read_text(encoding="utf-8")
        self.assertIn('marker.get("runtime_commit") != freeze_b', text)
        self.assertIn('index.get("runtime_commit") != freeze_b', text)
        self.assertIn(
            'conclusion not in {"global_release_passed", "scientific_no_go"}',
            text,
        )
        self.assertIn('"scientific_conclusion": (', text)
        self.assertIn('"global_release_passed": (', text)
        self.assertIn(
            'RESULT="validation_and_derivation_complete;scientific_conclusion=',
            text,
        )

    def test_all_contract_consumers_require_v3_preflight_hash_binding(self):
        consumers = (
            ROOT / "tools" / "submit_qwen3vl_oracle_trajectory.sh",
            ROOT / "tools" / "run_qwen3vl_oracle_trajectory_pilot.slurm",
            ROOT / "tools" / "run_qwen3vl_oracle_trajectory_validate.slurm",
        )
        for consumer in consumers:
            with self.subTest(consumer=consumer.name):
                text = consumer.read_text(encoding="utf-8")
                self.assertIn("qwen3vl_oracle_trajectory_contract_receipt_v3", text)
                self.assertIn("qwen3vl_oracle_trajectory_preflight_receipt_v1", text)
                self.assertIn('preflight.get("passed") is not True', text)
                self.assertIn('entry.get("sha256")', text)
                self.assertIn('entry.get("size_bytes")', text)

    def test_pilot_and_validation_workers_emit_bootstrap_terminal_receipts(self):
        workers = {
            "pilot": ROOT / "tools" / "run_qwen3vl_oracle_trajectory_pilot.slurm",
            "validation": ROOT / "tools" / "run_qwen3vl_oracle_trajectory_validate.slurm",
        }
        for name, worker in workers.items():
            with self.subTest(worker=name), tempfile.TemporaryDirectory() as directory:
                attempt = Path(directory) / "attempt-0001"
                (attempt / "status").mkdir(parents=True)
                completed = subprocess.run(
                    [
                        "bash",
                        str(worker),
                        "/definitely/missing/runtime-root",
                        str(attempt),
                        "/definitely/missing/config.json",
                        sys.executable,
                    ],
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=10,
                )
                self.assertNotEqual(completed.returncode, 0)
                receipts = list((attempt / "status").glob("bootstrap-*.json"))
                self.assertEqual(len(receipts), 1)
                receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
                self.assertEqual(
                    receipt["schema_id"],
                    "qwen3vl_oracle_trajectory_worker_bootstrap_terminal_v1",
                )
                self.assertEqual(receipt["status"], "failed")


if __name__ == "__main__":
    unittest.main()
