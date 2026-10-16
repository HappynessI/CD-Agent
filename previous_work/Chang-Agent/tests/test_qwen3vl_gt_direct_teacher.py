from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from change_agent.adapters.omniovcd_adapter import MaskPairProcessor, OmniOVCDAdapter
from change_agent.environment import ChangeAgentEnvironment
from change_agent.executor import ActionExecutor


ROOT = Path(__file__).resolve().parents[1]
BASE_SPEC = importlib.util.spec_from_file_location(
    "run_qwen3vl_oracle_trajectory_teacher_test_base",
    ROOT / "tools" / "run_qwen3vl_oracle_trajectory.py",
)
assert BASE_SPEC is not None and BASE_SPEC.loader is not None
BASE = importlib.util.module_from_spec(BASE_SPEC)
sys.modules[BASE_SPEC.name] = BASE
BASE_SPEC.loader.exec_module(BASE)
# The teacher runner uses the direct-script import name first.  Bind the exact
# already-loaded module so dataclass/type identities remain stable in tests.
sys.modules.setdefault("run_qwen3vl_oracle_trajectory", BASE)
SPEC = importlib.util.spec_from_file_location(
    "run_qwen3vl_gt_direct_teacher",
    ROOT / "tools" / "run_qwen3vl_gt_direct_teacher.py",
)
assert SPEC is not None and SPEC.loader is not None
TEACHER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TEACHER
SPEC.loader.exec_module(TEACHER)


class FakePointBackend:
    def __init__(self, transform=None) -> None:
        self.calls: list[dict] = []
        self.last_evidence: dict = {}
        self.transform = transform

    def refine(self, image, initial_mask, coordinate, is_positive, click_history=()):
        self.calls.append(
            {
                "image_shape": tuple(np.asarray(image).shape),
                "mask_sha256": BASE.mask_sha256(initial_mask),
                "coordinate": tuple(coordinate),
                "is_positive": bool(is_positive),
                "history": tuple(click_history),
            }
        )
        if self.transform is None:
            value = np.array(initial_mask, dtype=bool, copy=True)
            x, y = coordinate
            value[y, x] = bool(is_positive)
        else:
            value = np.asarray(
                self.transform(initial_mask, tuple(coordinate), bool(is_positive)),
                dtype=bool,
            )
        self.last_evidence = {
            "worker_mode": "point",
            "worker_report": {"status": "success", "mode": "point"},
        }
        return value


class FakeBoxBackend:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.last_evidence: dict = {}

    def segment_box(self, image, box_cxcywh_normalized, query):
        self.calls.append(
            {
                "image_shape": tuple(np.asarray(image).shape),
                "box": tuple(box_cxcywh_normalized),
                "query": query,
            }
        )
        self.last_evidence = {
            "worker_mode": "box",
            "worker_report": {"status": "success", "mode": "box"},
        }
        return np.zeros(np.asarray(image).shape[:2], dtype=bool)


class FakeRuntimeFactory:
    """Real Environment/Executor with deterministic no-model tool doubles."""

    def __init__(self, initial_t1: np.ndarray, initial_t2: np.ndarray, *, point_transform=None) -> None:
        self.initial_t1 = np.array(initial_t1, dtype=bool, copy=True)
        self.initial_t2 = np.array(initial_t2, dtype=bool, copy=True)
        self.runtime_inputs: list[BASE.RuntimeEpisode] = []
        self.verifiers: list[BASE.NoQwenAcceptVerifier] = []
        self.point_calls: list[dict] = []
        self.box_calls: list[dict] = []
        self.point_transform = point_transform

    @property
    def qwen_calls(self) -> int:
        return sum(verifier.qwen_calls for verifier in self.verifiers)

    def _environment(self, initializer, max_steps: int) -> ChangeAgentEnvironment:
        point = FakePointBackend(self.point_transform)
        box = FakeBoxBackend()
        verifier = BASE.NoQwenAcceptVerifier()
        self.verifiers.append(verifier)
        backend = OmniOVCDAdapter(initializer, box.segment_box, MaskPairProcessor())
        environment = ChangeAgentEnvironment(
            backend,
            ActionExecutor(point, box),
            verifier,
            max_steps=max_steps,
            require_tool_before_finish=False,
            enable_verifier_regions=True,
            empty_mask_scan_grid_size=3,
        )
        original_refine = point.refine
        original_box = box.segment_box

        def record_point(*args, **kwargs):
            result = original_refine(*args, **kwargs)
            self.point_calls.append(dict(point.calls[-1]))
            return result

        def record_box(*args, **kwargs):
            result = original_box(*args, **kwargs)
            self.box_calls.append(dict(box.calls[-1]))
            return result

        point.refine = record_point
        box.segment_box = record_box
        backend.segment_box_callback = box.segment_box
        return environment

    def create_canonical_environment(self, episode, artifact_dir, max_steps):
        del artifact_dir
        self.runtime_inputs.append(episode)

        def initialize(t1_image, t2_image, query):
            del t1_image, t2_image, query
            return (
                np.array(self.initial_t1, copy=True),
                np.array(self.initial_t2, copy=True),
                {"initializer": "fake_fresh_sam3"},
            )

        return self._environment(initialize, max_steps)

    def create_branch_environment(self, episode, accepted_state, artifact_dir):
        del artifact_dir
        self.runtime_inputs.append(episode)
        t1 = np.array(accepted_state.t1_mask, dtype=bool, copy=True)
        t2 = np.array(accepted_state.t2_mask, dtype=bool, copy=True)

        def initialize(t1_image, t2_image, query):
            del t1_image, t2_image, query
            return t1, t2, {"initializer": "fake_accepted_branch"}

        return self._environment(initialize, 1)


def runtime_episode(
    *,
    t1_state: str = "background",
    t2_state: str = "building",
) -> BASE.RuntimeEpisode:
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    identity = BASE.FileIdentity(Path("/unused.png"), "0" * 64)
    return BASE.RuntimeEpisode(
        episode_id="train_407_r3c2",
        sample_id="train_407_r3c2",
        split="train",
        parent_id="train_407",
        t1_image=image,
        t2_image=image,
        t1_source=identity,
        t2_source=identity,
        crop_xyxy_exclusive=(0, 0, 256, 256),
        query="building",
        rgb_states={"t1": t1_state, "t2": t2_state},
        rgb_semantic_evidence="unit fixture",
        rgb_semantic_source="unit fixture",
        rgb_semantic_status="test",
        rgb_states_by_region={},
        rgb_states_by_step={},
    )


def contract(initial_hashes: dict[str, str]) -> TEACHER.TeacherContract:
    config_path = ROOT / "configs" / "qwen3vl_oracle_trajectory_pilot_0803.json"
    return TEACHER.TeacherContract(
        base_config_path=config_path,
        base_manifest={},
        ordered_samples=("train_407_r3c2",),
        expected_initial_hashes={"train_407_r3c2": initial_hashes},
        max_steps=3,
        matched_max_candidates=4,
        wide_max_candidates=96,
        max_error_components=4,
        matched_padding_ratio=0.25,
        wide_padding_ratios=(0.0, 0.1, 0.25, 0.5),
        endpoint_iou=0.9,
        endpoint_f1=18.0 / 19.0,
        endpoint_empty_fp=64,
        comparison_epsilon=1e-12,
    )


def point_candidate(
    episode_id: str,
    step_index: int,
    coordinate: tuple[int, int],
    box: tuple[int, int, int, int],
) -> BASE.CandidateAction:
    region_id = f"teacher-step-{step_index:03d}-point"
    proposal = {
        "region_id": region_id,
        "component_seed_pixels": list(coordinate),
        "box_pixels": list(box),
        "component_area": 1,
        "component_seed_t1_mask_white": False,
        "component_seed_t2_mask_white": False,
        "teacher_only": True,
        "gt_derived_geometry": True,
    }
    return BASE.CandidateAction(
        candidate_id=f"{episode_id}--step-{step_index:03d}--point",
        region_id=region_id,
        reachability_tier="teacher_direct_gt_change_error",
        action=BASE.AgentAction("t2", "positive_point", coordinate=coordinate),
        proposal=proposal,
    )


class TeacherGeometryUnitTests(unittest.TestCase):
    def test_config_contract_binds_0803_and_four_formal_nontrivial_samples(self):
        manifest, frozen = TEACHER.load_contract(
            ROOT / "configs" / "qwen3vl_gt_direct_teacher_0804.json"
        )
        self.assertEqual(manifest["scope_and_access_audit"]["teacher_only"], True)
        self.assertFalse(manifest["scope_and_access_audit"]["training_authorized"])
        self.assertEqual(frozen.ordered_samples, TEACHER.FORMAL_SAMPLE_IDS)
        self.assertEqual(frozen.max_steps, 3)
        self.assertEqual(frozen.matched_max_candidates, 4)
        self.assertEqual(frozen.wide_max_candidates, 96)

    def test_gt_error_components_create_interior_points_multiscale_boxes_and_live_polarity(self):
        t1 = np.zeros((256, 256), dtype=bool)
        t2 = np.zeros_like(t1)
        t1[100:109, 100:109] = True
        change = np.logical_xor(t1, t2)
        state = BASE.ChangeState(
            np.zeros((256, 256, 3), dtype=np.uint8),
            np.zeros((256, 256, 3), dtype=np.uint8),
            "building",
            t1,
            t2,
            change,
        )
        ground_truth = np.zeros_like(change)
        ground_truth[20:29, 20:29] = True  # FN; the existing T1 block is FP.
        episode = runtime_episode(t1_state="background", t2_state="building")

        wide, catalog = TEACHER.build_teacher_candidates(
            episode,
            state,
            ground_truth,
            1,
            geometry_mode="wide",
            candidate_limit=96,
        )
        self.assertEqual(catalog["error_inventory"]["selected_component_count"], 2)
        self.assertEqual(catalog["padding_ratios"], [0.0, 0.1, 0.25, 0.5])
        self.assertEqual(catalog["action_count_before_cap"], 36)
        self.assertEqual(len(wide), 36)
        self.assertTrue(all(item.proposal["teacher_only"] for item in wide))
        self.assertTrue(all(item.proposal["geometry_source"] == "offline_gt_direct_teacher" for item in wide))
        self.assertTrue(all("component" not in item.proposal for item in wide))
        fp_points = [
            item
            for item in wide
            if item.proposal["error_kind"] == "fp"
            and item.action.target_view == "t1"
            and item.action.action.endswith("point")
        ]
        self.assertTrue(fp_points)
        self.assertEqual(
            {item.action.action for item in fp_points},
            {"negative_point", "positive_point"},
        )
        fn_points_t2 = [
            item
            for item in wide
            if item.proposal["error_kind"] == "fn"
            and item.action.target_view == "t2"
            and item.action.action.endswith("point")
        ]
        self.assertTrue(fn_points_t2)
        self.assertTrue(all(item.action.action == "positive_point" for item in fn_points_t2))
        fn_boxes = {
            tuple(item.action.box)
            for item in wide
            if item.proposal["error_kind"] == "fn"
            and item.action.target_view == "t2"
            and item.action.action == "box"
        }
        self.assertGreaterEqual(len(fn_boxes), 3)
        # Distance-transform interior seed of a 9x9 square is its center.
        self.assertTrue(
            all(item.action.coordinate == (24, 24) for item in fn_points_t2)
        )

    def test_wide_catalog_materializes_all_96_candidates_including_tail_half(self):
        t1 = np.zeros((256, 256), dtype=bool)
        t2 = np.zeros_like(t1)
        ground_truth = np.zeros_like(t1)
        for y, x in ((20, 20), (20, 80), (80, 20), (80, 80)):
            # Four equal disconnected FN components.  Both live masks are white at
            # every teacher seed, so each padding has all six reachable routes:
            # negative/positive/box on T1 and on T2.
            t1[y : y + 5, x : x + 5] = True
            t2[y : y + 5, x : x + 5] = True
            ground_truth[y : y + 5, x : x + 5] = True
        state = BASE.ChangeState(
            np.zeros((256, 256, 3), dtype=np.uint8),
            np.zeros((256, 256, 3), dtype=np.uint8),
            "building",
            t1,
            t2,
            np.zeros_like(t1),
        )

        actions, catalog = TEACHER.build_teacher_candidates(
            runtime_episode(),
            state,
            ground_truth,
            1,
            geometry_mode="wide",
            candidate_limit=96,
        )

        self.assertEqual(len(catalog["selected_components"]), 4)
        self.assertEqual(catalog["proposal_geometry_count"], 16)
        self.assertEqual(catalog["action_count_before_cap"], 96)
        self.assertEqual(catalog["execution_candidate_count"], 96)
        self.assertEqual(len(actions), 96)
        self.assertEqual(len({item.candidate_id for item in actions}), 96)
        self.assertEqual(catalog["candidate_ids"], [item.candidate_id for item in actions])
        self.assertEqual(
            {item.proposal["padding_ratio"] for item in actions[48:]},
            {0.25, 0.5},
        )
        self.assertEqual(
            sum(item.proposal["padding_ratio"] == 0.5 for item in actions),
            24,
        )

    def test_matched_safe_real_environment_reaches_exact_and_replay_hashes_match(self):
        initial_t1 = np.zeros((256, 256), dtype=bool)
        initial_t2 = np.zeros_like(initial_t1)
        factory = FakeRuntimeFactory(initial_t1, initial_t2)
        episode = runtime_episode()
        expected = {
            "t1": BASE.mask_sha256(initial_t1),
            "t2": BASE.mask_sha256(initial_t2),
            "change": BASE.mask_sha256(initial_t1),
        }
        ground_truth = np.zeros_like(initial_t1)
        ground_truth[50, 60] = True
        with tempfile.TemporaryDirectory() as directory:
            record = TEACHER.run_track(
                "matched_safe",
                episode,
                ground_truth,
                factory,
                Path(directory),
                Path(directory) / "episodes" / episode.episode_id,
                contract(expected),
                expected,
            )
        self.assertEqual(record["track_status"], "endpoint_success")
        self.assertEqual(record["termination"]["reason"], "exact_recovery")
        self.assertEqual(record["termination"]["selected_step_count"], 1)
        self.assertEqual(record["final_metrics"]["pixel_error"], 0)
        replay = record["steps"][0]["canonical_replay"]
        self.assertTrue(replay["all_three_mask_hashes_match"])
        self.assertFalse(replay["counterfactual_gate_bypass"])
        self.assertEqual(replay["mode"], "persistent_canonical_environment_replay")
        self.assertEqual(record["steps"][0]["action"]["action"], "positive_point")
        self.assertEqual(record["steps"][0]["action"]["target_view"], "t2")
        self.assertEqual(factory.qwen_calls, 0)
        self.assertGreaterEqual(len(factory.point_calls), 2)  # candidate + replay
        self.assertTrue(
            all(
                "label" not in vars(item) and "ground_truth" not in vars(item)
                for item in factory.runtime_inputs
            )
        )

    def test_diagnostic_smoke_is_step_one_two_real_tools_and_not_scientific(self):
        initial = np.zeros((256, 256), dtype=bool)
        factory = FakeRuntimeFactory(initial, initial)
        episode = runtime_episode()
        expected = {
            "t1": BASE.mask_sha256(initial),
            "t2": BASE.mask_sha256(initial),
            "change": BASE.mask_sha256(initial),
        }
        ground_truth = np.zeros_like(initial)
        ground_truth[50, 60] = True
        pilot = BASE.PilotEpisode(
            episode,
            BASE.LabelSource(
                BASE.FileIdentity(Path("/unused-label.png"), "4" * 64),
                (0, 0, 256, 256),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            record = TEACHER.run_diagnostic_smoke_episode(
                pilot,
                factory,
                Path(directory),
                contract(expected),
                label_loader=lambda _: ground_truth,
                expected_gt_pixels=1,
            )

        self.assertTrue(record["diagnostic_smoke"])
        self.assertFalse(record["scientific_result_authoritative"])
        self.assertTrue(record["teacher_only"])
        self.assertFalse(record["training_authorized"])
        self.assertEqual(record["qwen_calls"], 0)
        self.assertEqual(
            record["diagnostic_contract"]["step_indices_executed"], [1]
        )
        self.assertEqual(record["diagnostic_contract"]["max_real_candidates"], 2)
        self.assertFalse(
            record["diagnostic_contract"]["endpoint_search_performed"]
        )
        self.assertEqual(record["actual_execution"]["candidate_count"], 2)
        self.assertEqual(
            record["actual_execution"]["tool_types"], ["sam3", "simpleclick"]
        )
        self.assertEqual(
            record["actual_execution"]["action_types"],
            ["box", "positive_point"],
        )
        self.assertTrue(record["actual_execution"]["all_gt_geometry_authorized"])
        self.assertTrue(
            record["actual_execution"]["all_real_tool_receipts_present"]
        )
        self.assertEqual(len(factory.box_calls), 1)
        self.assertEqual(len(factory.point_calls), 1)
        self.assertTrue(
            all(
                "label" not in vars(item) and "ground_truth" not in vars(item)
                for item in factory.runtime_inputs
            )
        )

    def test_fresh_hash_drift_fails_before_label_access(self):
        initial = np.zeros((256, 256), dtype=bool)
        factory = FakeRuntimeFactory(initial, initial)
        episode = runtime_episode()
        wrong = {
            "t1": "1" * 64,
            "t2": "2" * 64,
            "change": "3" * 64,
        }
        label_calls: list[object] = []
        pilot = BASE.PilotEpisode(
            episode,
            BASE.LabelSource(
                BASE.FileIdentity(Path("/unused-label.png"), "4" * 64),
                (0, 0, 256, 256),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "initial hashes differ"):
                TEACHER.run_episode(
                    pilot,
                    factory,
                    Path(directory),
                    contract(wrong),
                    label_loader=lambda source: label_calls.append(source),
                    expected_gt_pixels=1,
                )
        self.assertEqual(label_calls, [])

    def test_wide_raw_counterfactual_then_accepted_branches_can_reach_third_step(self):
        initial = np.zeros((256, 256), dtype=bool)

        def transform(mask, coordinate, is_positive):
            value = np.array(mask, dtype=bool, copy=True)
            if coordinate == (10, 10):
                value[0:20, 0:20] = True  # 400 pixels: point combined gate rejects.
            else:
                x, y = coordinate
                value[y, x] = is_positive
            return value

        factory = FakeRuntimeFactory(initial, initial, point_transform=transform)
        episode = runtime_episode()
        ground_truth = np.zeros_like(initial)
        ground_truth[0:20, 0:20] = True
        ground_truth[100, 100] = True
        ground_truth[150, 150] = True
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            canonical = factory.create_canonical_environment(
                episode, output / "fresh", 3
            )
            canonical.reset(episode.t1_image, episode.t2_image, episode.query)
            assert canonical.state is not None
            source = canonical.state.clone()

            first = point_candidate(episode.episode_id, 1, (10, 10), (0, 0, 30, 30))
            first_record, first_state = TEACHER._execute_candidate(
                episode, source, first, factory, output / "candidate-1", output,
                ground_truth, 1, 0, 1e-12,
            )
            self.assertFalse(first_record["hard_gate"]["passed"])
            first_replay, source, canonical = TEACHER._replay_selected(
                episode, source, first, first_state, factory, output / "replay-1",
                output, ground_truth, "wide_raw", 1, canonical,
            )
            self.assertTrue(first_replay["counterfactual_gate_bypass"])
            self.assertIsNone(canonical)

            second = point_candidate(
                episode.episode_id, 2, (100, 100), (96, 96, 104, 104)
            )
            _, second_state = TEACHER._execute_candidate(
                episode, source, second, factory, output / "candidate-2", output,
                ground_truth, 2, 0, 1e-12,
            )
            second_replay, source, canonical = TEACHER._replay_selected(
                episode, source, second, second_state, factory, output / "replay-2",
                output, ground_truth, "wide_raw", 2, canonical,
            )
            self.assertFalse(second_replay["counterfactual_gate_bypass"])
            self.assertEqual(
                second_replay["mode"], "second_branch_canonical_environment_replay"
            )
            self.assertIsNone(canonical)  # max_steps=1 branch must never be reused.

            third = point_candidate(
                episode.episode_id, 3, (150, 150), (146, 146, 154, 154)
            )
            _, third_state = TEACHER._execute_candidate(
                episode, source, third, factory, output / "candidate-3", output,
                ground_truth, 3, 0, 1e-12,
            )
            third_replay, source, canonical = TEACHER._replay_selected(
                episode, source, third, third_state, factory, output / "replay-3",
                output, ground_truth, "wide_raw", 3, canonical,
            )
        self.assertFalse(third_replay["counterfactual_gate_bypass"])
        self.assertIsNone(canonical)
        self.assertEqual(BASE.mask_metrics(source.change_mask, ground_truth)["pixel_error"], 0)

    def test_adaptive_track_order_stops_after_first_success(self):
        winner = TEACHER._episode_conclusion("wide_safe")
        self.assertTrue(winner["proposal_bottleneck_supported"])
        self.assertFalse(winner["proposal_only_claim_allowed"])
        self.assertEqual(
            winner["classification"],
            "proposal_geometry_or_coverage_bottleneck_supported",
        )
        skipped = TEACHER._skipped_track("wide_hard", "wide_safe")
        self.assertFalse(skipped["executed"])
        self.assertEqual(skipped["skip_reason"], "prior_track_endpoint_success")
        self.assertFalse(skipped["training_authorized"])


if __name__ == "__main__":
    unittest.main()
