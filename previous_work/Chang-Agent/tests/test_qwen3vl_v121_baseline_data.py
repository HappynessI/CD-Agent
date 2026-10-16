import json
import os
import unittest
from collections import Counter
from pathlib import Path

from change_agent.adapters.stage_backends import _stage_prompt, _stage_visual_input_scope
from change_agent.adapters.staged_verifier import (
    StagedQwenVerifier,
    _parse_atomic_audit,
    _parse_judgment,
    _plan_agent_action,
)
from change_agent.state import ChangeState
from change_agent.verifier_regions import attach_verifier_regions
from change_agent.verifier_protocol import EvidenceRecord
from tools.build_qwen3vl_v121_baseline_data import (
    audit_target,
    canonical_core_content_sha256,
    checklist,
    component_sha256,
    environment_record,
    residual_component,
    sampling_content_key,
    scan_grid_geometry,
    unchanged_transition_case,
    with_action_locality,
    V6_FROZEN_CORE_CANONICAL_SHA256,
    V6_FROZEN_CORE_HANDOFF_SHA256,
    WHITE_NO_TRANSITION_VALIDATION_ERROR,
)

import numpy as np


class Qwen3VLV121BaselineDataTest(unittest.TestCase):
    V5_CONTRAST_CONSTRUCTION = "production_contrast_repair_v5"
    V5_SEMANTIC_IDS = {
        "train_train_118_r1c1_changed",
        "train_train_114_r3c1_changed",
    }
    V5_ACTIONABLE_IDS = {
        "train_train_364_r0c2_unchanged",
        "train_train_365_r3c2_unchanged",
    }
    V6_RESIDUAL_VIEWS = {
        "train_train_51_r3c0_changed": "t2",
        "train_train_85_r0c0_changed": "t1",
    }
    V6_TRUE_TRANSITION_VIEWS = {
        "train_train_407_r3c2_changed": "t2",
        "train_train_231_r3c2_changed": "t1",
    }

    @staticmethod
    def _environment_record_from_row(row):
        """Recover the typed Environment region embedded in one ShareGPT row."""
        prompt = row["conversations"][0]["value"]
        start = prompt.index("<ENVIRONMENT_FACTS>") + len("<ENVIRONMENT_FACTS>")
        end = prompt.index("</ENVIRONMENT_FACTS>", start)
        envelope = json.loads(prompt[start:end].strip())
        region = envelope["environment_facts"]["region"]
        # ``from_proposal`` intentionally ignores serialized authoritative
        # editable facts and only trusts the pre-envelope proposal shape.  A
        # ShareGPT row already contains the frozen Environment envelope, so
        # reconstruct the typed record directly for this delivery audit.
        return EvidenceRecord(
            region_id=str(region["region_id"]),
            audit_kind=str(region["audit_kind"]),
            change_mask_state=str(region["change_mask_state"]),
            temporal_difference_state=str(region["temporal_difference_state"]),
            box_normalized_1000=tuple(int(value) for value in region["box_normalized_1000"]),
            component_seed_normalized_1000=tuple(
                int(value) for value in region["component_seed_normalized_1000"]
            ),
            t1_mask_pixels=int(region["t1_mask_pixels"]),
            t2_mask_pixels=int(region["t2_mask_pixels"]),
            component_area=int(region["component_area"]),
            editable_seed_white={
                str(view): bool(value)
                for view, value in region["editable_seed_white"].items()
            },
            allowed_actions=tuple(str(value) for value in region["allowed_actions"]),
            transition_mask_facts={
                str(key): value
                for key, value in region["transition_mask_facts"].items()
            },
        )

    def test_v6_curation_config_freezes_matched_contrast_recipe(self):
        path = Path(__file__).resolve().parents[1] / "configs/qwen3vl_v121_curation_0801.json"
        payload = json.loads(path.read_text())
        v6 = payload["hardcase_v6"]
        self.assertEqual(v6["base_round"], "v3-action")
        self.assertFalse(v6["include_background_fp_pairs"])
        core = v6["frozen_v3_core"]
        self.assertEqual(core["expected_rows"], 53)
        self.assertEqual(core["source_job_id"], "52132")
        self.assertEqual(
            core["handoff_content_sha256"],
            V6_FROZEN_CORE_HANDOFF_SHA256,
        )
        self.assertEqual(
            core["canonical_content_sha256"],
            V6_FROZEN_CORE_CANONICAL_SHA256,
        )
        self.assertEqual(
            core["canonical_content_fields"],
            ["id", "prompt_sha256", "target_sha256", "image_sha256"],
        )
        residual = {
            item["candidate_id"]: item["source_view"]
            for item in v6["residual_false_positive_cases"]
        }
        self.assertEqual(residual, self.V6_RESIDUAL_VIEWS)
        true_transition = {
            item["candidate_id"]: item["target_view"]
            for item in v6["true_transition_cases"]
        }
        self.assertEqual(true_transition, self.V6_TRUE_TRANSITION_VIEWS)
        self.assertEqual(
            {item["variant"] for item in v6["true_transition_cases"]},
            {"forward", "time_reverse"},
        )
        self.assertEqual(
            {int(item["expected_component_area"]) for item in v6["true_transition_cases"]},
            {763, 1630},
        )

    def test_v7_curation_config_declares_only_sampling_repetition(self):
        path = Path(__file__).resolve().parents[1] / "configs/qwen3vl_v121_curation_0801.json"
        payload = json.loads(path.read_text())
        self.assertEqual(
            payload["hardcase_v7"],
            {
                "base_round":"v6-matched-contrast",
                "repeat_selector_field":"v6_extension",
                "expected_unique_train_rows":63,
                "expected_val_rows":24,
                "expected_weighted_source_rows":10,
                "sampling_repeat_copies":2,
                "sampling_weight_factor":3,
                "expected_train_exposures":83,
            },
        )

    def test_v6_true_transition_repairs_finish_and_removals_are_worse(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        empty = np.zeros((32, 32), dtype=bool)
        blob = empty.copy()
        blob[8:24, 8:24] = True
        cases = (
            ("t2", "background", "building"),
            ("t1", "building", "background"),
        )
        for target_view, t1_state, t2_state in cases:
            with self.subTest(target_view=target_view):
                t1_mask = blob if target_view == "t1" else empty
                t2_mask = blob if target_view == "t2" else empty
                correct = ChangeState(
                    image, image, "building", t1_mask, t2_mask, blob
                )
                record, proposal = environment_record(correct, expected_kind="present")
                annotation = {
                    "t1_state": t1_state,
                    "t2_state": t2_state,
                    "evidence": (
                        "T1 and T2 show a real roofed-building transition at "
                        "the same location."
                    ),
                }
                repair_target = audit_target(record.region_id, annotation, "correct")
                _, repair_diagnosis = _parse_atomic_audit(repair_target, record)
                repair_plan = StagedQwenVerifier._plan(
                    None, correct, (record,), repair_diagnosis
                )
                self.assertEqual(repair_diagnosis.error_type, "none")
                self.assertEqual(
                    (repair_plan.action, repair_plan.target_view),
                    ("finish", None),
                )

                wrong_target = audit_target(
                    record.region_id, annotation, "false_positive_transition"
                )
                _, wrong_diagnosis = _parse_atomic_audit(wrong_target, record)
                wrong_plan = StagedQwenVerifier._plan(
                    None, correct, (record,), wrong_diagnosis
                )
                self.assertEqual(wrong_diagnosis.error_type, "false_positive_change")
                self.assertEqual(wrong_diagnosis.target_view, target_view)
                self.assertEqual(
                    (wrong_plan.action, wrong_plan.target_view),
                    ("negative_point", target_view),
                )

                harmful = ChangeState(
                    image, image, "building", empty, empty, empty
                )
                harmful = with_action_locality(
                    harmful, proposal, action="negative_point"
                )
                removed, _ = environment_record(
                    harmful, correct, expected_kind="delta_removed"
                )
                judgment = _parse_judgment(
                    {
                        "region_id": removed.region_id,
                        "visual_judgment": {
                            "t1_state": t1_state,
                            "t2_state": t2_state,
                            "evidence_quality": "clear",
                        },
                    },
                    removed.region_id,
                )
                action = _plan_agent_action(wrong_plan, correct.image_size)
                verifier = object.__new__(StagedQwenVerifier)
                verifier._planned_diagnosis = wrong_diagnosis
                verifier._planned_action = action
                assessment = verifier._assess_candidate_transition(
                    (removed,), (judgment,), action
                )
                self.assertTrue(assessment.evidence_sufficient)
                self.assertFalse(assessment.intended_error_improved)
                self.assertTrue(assessment.introduced_false_negative)
                self.assertEqual(assessment.comparison, "worse")

    def test_v6_manifest_contract_if_available(self):
        """Run as an explicit post-build gate with the v6 manifest env set."""
        configured = os.environ.get("QWEN3VL_V121_V6_MANIFEST")
        if not configured:
            self.skipTest("set QWEN3VL_V121_V6_MANIFEST for the post-build contract")
        manifest_path = Path(configured)
        self.assertTrue(
            manifest_path.is_file(),
            f"missing explicitly requested v6 manifest: {manifest_path}",
        )

        data_root = manifest_path.parent
        manifest = json.loads(manifest_path.read_text())
        validation = json.loads((data_root / "validation.json").read_text())
        self.assertEqual(manifest["data_round"], "v6-matched-contrast")
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["data_round"], "v6-matched-contrast")
        self.assertEqual(
            (validation["train_rows"], validation["val_rows"]),
            (63, 24),
        )
        self.assertEqual(
            (validation["train_parents"], validation["val_parents"]),
            (11, 6),
        )
        self.assertEqual(
            validation["distribution"],
            {
                "train|audit|standard": 25,
                "train|audit|authoritative_pair_rgb_only_evidence_repair": 8,
                "train|candidate_evidence|standard": 21,
                "train|audit|authoritative_pair_rgb_only_scan": 9,
                "val|audit|standard": 9,
                "val|audit|authoritative_pair_rgb_only_scan": 6,
                "val|audit|authoritative_pair_rgb_only_evidence_repair": 3,
                "val|candidate_evidence|standard": 6,
            },
        )
        self.assertEqual(
            validation["semantic_state_distribution"],
            {
                "train|background->building": 32,
                "train|background->background": 13,
                "train|building->building": 8,
                "train|building->background": 10,
                "val|background->building": 10,
                "val|building->background": 5,
                "val|background->background": 9,
            },
        )
        self.assertEqual(
            validation["candidate_outcome_distribution"],
            {
                "train|improvement": 13,
                "train|harm": 8,
                "val|unspecified": 6,
            },
        )
        self.assertFalse(manifest["test_data_used"])
        self.assertFalse(manifest["raw_gt_model_visible"])

        receipt = manifest["v6_frozen_core"]
        self.assertEqual(receipt, validation["v6_frozen_core"])
        self.assertEqual(receipt["rows"], 53)
        self.assertEqual(
            receipt["handoff_content_sha256"],
            V6_FROZEN_CORE_HANDOFF_SHA256,
        )
        self.assertEqual(
            receipt["canonical_content_sha256"],
            V6_FROZEN_CORE_CANONICAL_SHA256,
        )

        samples = manifest["samples"]
        train_samples = [item for item in samples if item["split"] == "train"]
        core = [item for item in train_samples if item.get("v6_core") is True]
        self.assertEqual(len(core), 53)
        self.assertEqual(
            canonical_core_content_sha256(core),
            V6_FROZEN_CORE_CANONICAL_SHA256,
        )
        self.assertFalse(
            any(
                item.get("construction")
                == "production_background_false_transition_v3"
                for item in train_samples
            )
        )
        grid_harm = [
            item for item in train_samples
            if "_candidate_grid_harm_d" in item["id"]
        ]
        self.assertEqual(len(grid_harm), 3)
        self.assertTrue(
            all(item.get("candidate_outcome") == "harm" for item in grid_harm)
        )

        rows = {
            row["id"]: row
            for row in json.loads((data_root / "train.json").read_text())
        }
        blank = ChangeState(
            np.zeros((256, 256, 3), dtype=np.uint8),
            np.zeros((256, 256, 3), dtype=np.uint8),
            "building",
            np.zeros((256, 256), dtype=bool),
            np.zeros((256, 256), dtype=bool),
            np.zeros((256, 256), dtype=bool),
        )

        residual = [
            item for item in train_samples
            if item.get("construction")
            == "production_residual_fp_contract_repair_v4"
        ]
        self.assertEqual(len(residual), 6)
        self.assertTrue(
            all(
                item.get("v6_extension") is True
                and item.get("v6_residual_label_correction") is True
                and item.get("residual_label_kind")
                == "false_positive_background"
                for item in residual
            )
        )
        for candidate_id, target_view in self.V6_RESIDUAL_VIEWS.items():
            grouped = [
                item for item in residual
                if item["candidate_id"] == candidate_id
            ]
            repair = next(
                item for item in grouped
                if item["v4_example_kind"] == "audit_repair_replay"
            )
            candidate = next(
                item for item in grouped
                if item["v4_example_kind"] == "candidate_improvement"
            )
            repair_row = rows[repair["id"]]
            repair_record = self._environment_record_from_row(repair_row)
            repair_target = json.loads(repair_row["conversations"][-1]["value"])
            self.assertEqual(
                repair_target["diagnosis"]["audit_checklist"]["target_class_only"],
                "fail",
            )
            _, diagnosis = _parse_atomic_audit(repair_target, repair_record)
            plan = StagedQwenVerifier._plan(
                None, blank, (repair_record,), diagnosis
            )
            self.assertEqual(diagnosis.error_type, "false_positive_change")
            self.assertEqual(diagnosis.target_view, target_view)
            self.assertEqual(
                (plan.action, plan.target_view),
                ("negative_point", target_view),
            )
            self.assertEqual(
                candidate["attempted_action"],
                {"target_view": target_view, "action": "negative_point"},
            )
            candidate_row = rows[candidate["id"]]
            candidate_record = self._environment_record_from_row(candidate_row)
            candidate_target = json.loads(
                candidate_row["conversations"][-1]["value"]
            )
            judgment = _parse_judgment(
                candidate_target, candidate_record.region_id
            )
            action = _plan_agent_action(plan, blank.image_size)
            verifier = object.__new__(StagedQwenVerifier)
            verifier._planned_diagnosis = diagnosis
            verifier._planned_action = action
            assessment = verifier._assess_candidate_transition(
                (candidate_record,), (judgment,), action
            )
            self.assertTrue(assessment.evidence_sufficient)
            self.assertTrue(assessment.intended_error_improved)
            self.assertFalse(assessment.introduced_harm)
            self.assertEqual(assessment.comparison, "better")

        matched = [
            item for item in train_samples
            if item.get("construction")
            == "production_matched_true_transition_v6"
        ]
        self.assertEqual(len(matched), 4)
        self.assertEqual(
            Counter(item["v6_example_kind"] for item in matched),
            Counter(
                {
                    "true_transition_repair": 2,
                    "true_transition_harmful_removal": 2,
                }
            ),
        )
        repairs = {
            item["candidate_id"]: item
            for item in matched
            if item["v6_example_kind"] == "true_transition_repair"
        }
        self.assertEqual(set(repairs), set(self.V6_TRUE_TRANSITION_VIEWS))
        for candidate_id, target_view in self.V6_TRUE_TRANSITION_VIEWS.items():
            repair = repairs[candidate_id]
            repair_row = rows[repair["id"]]
            repair_record = self._environment_record_from_row(repair_row)
            repair_target = json.loads(repair_row["conversations"][-1]["value"])
            _, repair_diagnosis = _parse_atomic_audit(
                repair_target, repair_record
            )
            repair_plan = StagedQwenVerifier._plan(
                None, blank, (repair_record,), repair_diagnosis
            )
            self.assertEqual(repair_diagnosis.error_type, "none")
            self.assertEqual(
                (repair_plan.action, repair_plan.target_view),
                ("finish", None),
            )
            candidate = next(
                item for item in matched
                if item["id"] == repair["paired_candidate_id"]
            )
            self.assertEqual(candidate["paired_repair_id"], repair["id"])
            self.assertEqual(
                (
                    candidate["attempted_action"]["action"],
                    candidate["attempted_action"]["target_view"],
                    candidate["candidate_outcome"],
                    candidate["expected_candidate_comparison"],
                ),
                ("negative_point", target_view, "harm", "worse"),
            )

            wrong_target = audit_target(
                repair_record.region_id,
                repair_target["visual_judgment"],
                "false_positive_transition",
            )
            _, wrong_diagnosis = _parse_atomic_audit(
                wrong_target, repair_record
            )
            wrong_plan = StagedQwenVerifier._plan(
                None, blank, (repair_record,), wrong_diagnosis
            )
            self.assertEqual(wrong_diagnosis.target_view, target_view)
            self.assertEqual(
                (wrong_plan.action, wrong_plan.target_view),
                ("negative_point", target_view),
            )
            action = _plan_agent_action(wrong_plan, blank.image_size)
            self.assertEqual(candidate["attempted_action"], action.to_dict())
            candidate_row = rows[candidate["id"]]
            candidate_record = self._environment_record_from_row(candidate_row)
            candidate_target = json.loads(
                candidate_row["conversations"][-1]["value"]
            )
            judgment = _parse_judgment(
                candidate_target, candidate_record.region_id
            )
            verifier = object.__new__(StagedQwenVerifier)
            verifier._planned_diagnosis = wrong_diagnosis
            verifier._planned_action = action
            assessment = verifier._assess_candidate_transition(
                (candidate_record,), (judgment,), action
            )
            self.assertTrue(assessment.evidence_sufficient)
            self.assertFalse(assessment.intended_error_improved)
            self.assertTrue(assessment.introduced_false_negative)
            self.assertEqual(assessment.comparison, "worse")

    def test_v7_manifest_contract_if_available(self):
        """Validate 83 exposures over exactly the 63 unique v6 contents."""
        configured = os.environ.get("QWEN3VL_V121_V7_MANIFEST")
        if not configured:
            self.skipTest("set QWEN3VL_V121_V7_MANIFEST for the post-build contract")
        manifest_path = Path(configured)
        self.assertTrue(
            manifest_path.is_file(),
            f"missing explicitly requested v7 manifest: {manifest_path}",
        )

        data_root = manifest_path.parent
        manifest = json.loads(manifest_path.read_text())
        validation = json.loads((data_root / "validation.json").read_text())
        self.assertEqual(manifest["data_round"], "v7-weighted-contrast")
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["data_round"], "v7-weighted-contrast")
        self.assertEqual(
            (validation["train_rows"], validation["val_rows"]),
            (83, 24),
        )
        self.assertEqual(
            (validation["train_parents"], validation["val_parents"]),
            (11, 6),
        )
        self.assertEqual(
            validation["distribution"],
            {
                "train|audit|standard":37,
                "train|audit|authoritative_pair_rgb_only_evidence_repair":8,
                "train|candidate_evidence|standard":29,
                "train|audit|authoritative_pair_rgb_only_scan":9,
                "val|audit|standard":9,
                "val|audit|authoritative_pair_rgb_only_scan":6,
                "val|audit|authoritative_pair_rgb_only_evidence_repair":3,
                "val|candidate_evidence|standard":6,
            },
        )
        self.assertEqual(
            validation["semantic_state_distribution"],
            {
                "train|background->building":36,
                "train|background->background":25,
                "train|building->building":8,
                "train|building->background":14,
                "val|background->building":10,
                "val|building->background":5,
                "val|background->background":9,
            },
        )
        self.assertEqual(
            validation["candidate_outcome_distribution"],
            {
                "train|improvement":17,
                "train|harm":12,
                "val|unspecified":6,
            },
        )
        self.assertFalse(manifest["test_data_used"])
        self.assertFalse(manifest["raw_gt_model_visible"])

        receipt = manifest["v7_sampling_receipt"]
        self.assertEqual(receipt, validation["v7_sampling_receipt"])
        self.assertEqual(
            {
                key:receipt[key]
                for key in (
                    "schema",
                    "strategy",
                    "base_round",
                    "unique_train_rows",
                    "train_exposures",
                    "val_rows",
                    "weighted_source_rows",
                    "repeat_rows",
                    "sampling_repeat_copies",
                    "sampling_weight_factor",
                    "unique_content_count",
                    "repeat_content_exact",
                    "test_data_used",
                )
            },
            {
                "schema":"qwen3vl_v121_sampling_receipt_v1",
                "strategy":"explicit_row_repetition",
                "base_round":"v6-matched-contrast",
                "unique_train_rows":63,
                "train_exposures":83,
                "val_rows":24,
                "weighted_source_rows":10,
                "repeat_rows":20,
                "sampling_repeat_copies":2,
                "sampling_weight_factor":3,
                "unique_content_count":63,
                "repeat_content_exact":True,
                "test_data_used":False,
            },
        )
        self.assertEqual(
            manifest["v6_frozen_core"]["canonical_content_sha256"],
            V6_FROZEN_CORE_CANONICAL_SHA256,
        )

        samples = manifest["samples"]
        train_samples = [item for item in samples if item["split"] == "train"]
        val_samples = [item for item in samples if item["split"] == "val"]
        self.assertEqual((len(train_samples),len(val_samples)),(83,24))
        self.assertEqual(len({item["id"] for item in train_samples}),83)
        repeats = [
            item for item in train_samples
            if item.get("construction") == "sampling_weight_repeat_v7"
        ]
        unique = [
            item for item in train_samples
            if item.get("construction") != "sampling_weight_repeat_v7"
        ]
        weighted = [
            item for item in unique
            if item.get("v7_weighted_source") is True
        ]
        self.assertEqual((len(unique),len(weighted),len(repeats)),(63,10,20))
        self.assertEqual(
            Counter(item["construction"] for item in weighted),
            Counter({
                "production_residual_fp_contract_repair_v4":6,
                "production_matched_true_transition_v6":4,
            }),
        )
        self.assertEqual(
            Counter(item["stage"] for item in train_samples),
            Counter({"audit":54,"candidate_evidence":29}),
        )
        self.assertEqual(
            len({sampling_content_key(item) for item in train_samples}),
            63,
        )
        self.assertEqual(
            canonical_core_content_sha256(unique),
            receipt["unique_train_content_sha256"],
        )
        self.assertEqual(
            canonical_core_content_sha256(weighted),
            receipt["weighted_source_content_sha256"],
        )
        self.assertEqual(
            [item["id"] for item in weighted],
            receipt["weighted_source_ids"],
        )
        self.assertTrue(
            all(
                item.get("sampling_weight_factor") == 3
                and item.get("sampling_repeat_count") == 2
                for item in weighted
            )
        )
        self.assertFalse(
            any(
                item.get("construction") == "sampling_weight_repeat_v7"
                for item in val_samples
            )
        )

        train_rows = {
            row["id"]:row
            for row in json.loads((data_root / "train.json").read_text())
        }
        self.assertEqual(len(train_rows),83)
        manifest_by_id={item["id"]:item for item in train_samples}
        self.assertEqual(
            Counter(item["repeated_from_id"] for item in repeats),
            Counter({item["id"]:2 for item in weighted}),
        )
        for source in weighted:
            self.assertEqual(
                {
                    item["sampling_repeat_index"]
                    for item in repeats
                    if item["repeated_from_id"] == source["id"]
                },
                {1,2},
            )
        exact_manifest_fields=(
            "split",
            "stage",
            "attempt_kind",
            "visual_input_scope",
            "prompt_sha256",
            "target_sha256",
            "image_sha256",
            "parent_id",
            "candidate_id",
            "source_kind",
            "source_sha256",
            "t1_state",
            "t2_state",
        )
        for repeat in repeats:
            source = manifest_by_id[repeat["repeated_from_id"]]
            source_row = train_rows[source["id"]]
            repeat_row = train_rows[repeat["id"]]
            self.assertNotEqual(repeat["id"],source["id"])
            self.assertEqual(
                repeat["repeated_source_construction"],
                source["construction"],
            )
            self.assertIn(repeat["sampling_repeat_index"],{1,2})
            self.assertEqual(repeat["sampling_weight_factor"],3)
            self.assertTrue(repeat["sampling_repeat_only"])
            self.assertFalse(repeat["unique_content"])
            self.assertNotIn("v6_extension",repeat)
            self.assertEqual(
                {key:repeat[key] for key in exact_manifest_fields},
                {key:source[key] for key in exact_manifest_fields},
            )
            self.assertEqual(
                sampling_content_key(repeat),
                sampling_content_key(source),
            )
            self.assertEqual(
                repeat_row["conversations"],
                source_row["conversations"],
            )
            self.assertEqual(repeat_row["images"],source_row["images"])

    def test_v5_curation_config_declares_two_contrast_pairs(self):
        """The v5 recipe must use existing train-only components, not new test data."""
        path = Path(__file__).resolve().parents[1] / "configs/qwen3vl_v121_curation_0801.json"
        payload = json.loads(path.read_text())
        v5 = payload["hardcase_v5"]
        semantic = v5["semantic_anchor_cases"]
        actionable = v5["actionable_repair_cases"]
        self.assertEqual(len(semantic), 2)
        self.assertEqual(len(actionable), 2)
        self.assertEqual(
            {item["candidate_id"] for item in semantic}, self.V5_SEMANTIC_IDS
        )
        self.assertEqual(
            {item["candidate_id"] for item in actionable}, self.V5_ACTIONABLE_IDS
        )
        for item in semantic + actionable:
            self.assertTrue(item["candidate_id"].startswith("train_"))
            self.assertIn("paired_candidate_id", item)
            area_key = (
                "expected_component_area"
                if item in semantic
                else "expected_area"
            )
            self.assertIn(area_key, item)
            self.assertGreater(int(item[area_key]), 0)
        self.assertEqual(
            {item["source_view"] for item in actionable}, {"t2"}
        )
        self.assertEqual(
            {int(item["component_rank"]) for item in actionable}, {0, 1}
        )

    def test_v5_semantic_anchor_parser_finishes_without_negative_action(self):
        """A true transition must not be repaired into a negative point."""
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        empty = np.zeros((32, 32), dtype=bool)
        blob = empty.copy()
        blob[8:24, 8:24] = True
        state = ChangeState(image, image, "building", empty, blob, blob)
        record, _ = environment_record(state, expected_kind="present")
        annotation = {
            "t1_state": "background",
            "t2_state": "building",
            "evidence": "T1 shows bare ground while T2 shows a roofed building at the same location.",
        }
        target = audit_target(record.region_id, annotation, "correct")
        _, diagnosis = _parse_atomic_audit(target, record)
        plan = StagedQwenVerifier._plan(None, state, (record,), diagnosis)
        self.assertEqual(diagnosis.error_type, "none")
        self.assertIsNotNone(plan)
        self.assertEqual((plan.action, plan.target_view), ("finish", None))

    def test_v5_actionable_repair_derives_negative_point_and_better_delta_removed(self):
        """A same-class false transition must localize to t2 and improve on removal."""
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        empty = np.zeros((32, 32), dtype=bool)
        blob = empty.copy()
        blob[8:24, 8:24] = True
        case = unchanged_transition_case(image, image, blob, source_view="t2")
        wrong, corrected = case["false_transition"], case["corrected"]
        record, proposal = environment_record(wrong, expected_kind="present")
        annotation = {
            "t1_state": "building",
            "t2_state": "building",
            "evidence": "The same roofed building remains visible in both T1 and T2.",
        }
        target = audit_target(record.region_id, annotation, "false_positive_transition")
        _, diagnosis = _parse_atomic_audit(target, record)
        plan = StagedQwenVerifier._plan(None, wrong, (record,), diagnosis)
        self.assertEqual(diagnosis.error_type, "false_positive_change")
        self.assertEqual(diagnosis.target_view, "t2")
        self.assertIsNotNone(plan)
        self.assertEqual((plan.action, plan.target_view), ("negative_point", "t2"))

        corrected = with_action_locality(corrected, proposal, action="negative_point")
        delta, _ = environment_record(corrected, wrong, expected_kind="delta_removed")
        self.assertEqual(delta.audit_kind, "delta_removed")
        candidate_target = {
            "region_id": delta.region_id,
            "visual_judgment": {
                "t1_state": "building",
                "t2_state": "building",
                "evidence_quality": "clear",
            },
        }
        judgment = _parse_judgment(candidate_target, delta.region_id)
        verifier = object.__new__(StagedQwenVerifier)
        verifier._planned_diagnosis = diagnosis
        verifier._planned_action = _plan_agent_action(plan, wrong.image_size)
        assessment = verifier._assess_candidate_transition(
            (delta,), (judgment,), verifier._planned_action
        )
        self.assertTrue(assessment.evidence_sufficient)
        self.assertTrue(assessment.intended_error_improved)
        self.assertFalse(assessment.introduced_harm)
        self.assertEqual(assessment.comparison, "better")

    def test_v5_manifest_contract_if_available(self):
        """Validate a produced v5 manifest without making unit tests depend on output numbering."""
        configured = os.environ.get("QWEN3VL_V121_V5_MANIFEST")
        if not configured:
            self.skipTest("set QWEN3VL_V121_V5_MANIFEST for the post-build contract")
        manifest_path = Path(configured)
        self.assertTrue(
            manifest_path.is_file(),
            f"missing explicitly requested v5 manifest: {manifest_path}",
        )

        data_root = manifest_path.parent
        manifest = json.loads(manifest_path.read_text())
        validation = json.loads((data_root / "validation.json").read_text())
        self.assertTrue(validation["valid"])
        self.assertEqual((validation["train_rows"], validation["val_rows"]), (69, 24))
        self.assertEqual(
            (validation["train_parents"], validation["val_parents"]), (11, 6)
        )
        self.assertEqual(
            validation["distribution"],
            {
                "train|audit|standard": 30,
                "train|audit|authoritative_pair_rgb_only_evidence_repair": 8,
                "train|candidate_evidence|standard": 22,
                "train|audit|authoritative_pair_rgb_only_scan": 9,
                "val|audit|standard": 9,
                "val|audit|authoritative_pair_rgb_only_scan": 6,
                "val|audit|authoritative_pair_rgb_only_evidence_repair": 3,
                "val|candidate_evidence|standard": 6,
            },
        )
        self.assertEqual(
            validation["semantic_state_distribution"],
            {
                "train|background->building": 32,
                "train|background->background": 19,
                "train|building->building": 10,
                "train|building->background": 8,
                "val|background->building": 10,
                "val|building->background": 5,
                "val|background->background": 9,
            },
        )
        self.assertFalse(manifest["test_data_used"])
        self.assertFalse(manifest["raw_gt_model_visible"])

        samples = manifest["samples"]
        residual = [
            item for item in samples
            if item.get("construction") == "production_residual_fp_contract_repair_v4"
        ]
        self.assertEqual(len(residual), 6)
        self.assertTrue(all(item.get("v5_residual_label_correction") is True for item in residual))
        self.assertTrue(all(item.get("residual_label_kind") == "false_positive_background" for item in residual))
        v5 = [
            item for item in samples
            if item.get("construction") == self.V5_CONTRAST_CONSTRUCTION
        ]
        self.assertEqual(len(v5), 4)
        self.assertEqual(
            Counter(item["v5_example_kind"] for item in v5),
            Counter({"semantic_anchor_repair": 2, "actionable_transition_repair": 2}),
        )
        self.assertTrue(
            all(
                (item["stage"], item["attempt_kind"], item["visual_input_scope"], item["region_audit_kind"])
                == ("audit", "repair", "standard", "present")
                for item in v5
            )
        )
        self.assertTrue(
            all(
                item.get("origin_validation_error") == WHITE_NO_TRANSITION_VALIDATION_ERROR
                for item in v5
            )
        )

        rows = {row["id"]: row for row in json.loads((data_root / "train.json").read_text())}
        sample_by_id = {item["id"]: item for item in samples}
        for item in residual:
            if item["stage"] != "audit":
                continue
            target = json.loads(rows[item["id"]]["conversations"][-1]["value"])
            self.assertEqual(
                target["diagnosis"]["audit_checklist"]["target_class_only"],
                "fail",
            )
        repair_states = Counter(
            (item["t1_state"], item["t2_state"])
            for item in samples
            if item.get("split") == "train"
            and item.get("attempt_kind") == "repair"
            and (
                item.get("origin_validation_error") == WHITE_NO_TRANSITION_VALIDATION_ERROR
                or item.get("rejected_validation_error") == WHITE_NO_TRANSITION_VALIDATION_ERROR
            )
        )
        self.assertEqual(
            repair_states,
            Counter(
                {
                    ("background", "background"): 2,
                    ("background", "building"): 2,
                    ("building", "building"): 2,
                }
            ),
        )

        for item in v5:
            row = rows[item["id"]]
            record = self._environment_record_from_row(row)
            target = json.loads(row["conversations"][-1]["value"])
            _, diagnosis = _parse_atomic_audit(target, record)
            plan = StagedQwenVerifier._plan(None, ChangeState(
                np.zeros((32, 32, 3), dtype=np.uint8),
                np.zeros((32, 32, 3), dtype=np.uint8),
                "building",
                np.zeros((32, 32), dtype=bool),
                np.zeros((32, 32), dtype=bool),
                np.zeros((32, 32), dtype=bool),
            ), (record,), diagnosis)
            paired_id = item["paired_candidate_id"]
            self.assertIn(paired_id, sample_by_id)
            paired = sample_by_id[paired_id]
            self.assertEqual(paired["stage"], "candidate_evidence")
            self.assertEqual(paired["region_audit_kind"], "delta_removed")
            self.assertEqual(paired["candidate_outcome"],
                             "harm" if item["v5_example_kind"] == "semantic_anchor_repair" else "improvement")
            if item["v5_example_kind"] == "semantic_anchor_repair":
                self.assertEqual(diagnosis.error_type, "none")
                self.assertEqual((plan.action, plan.target_view), ("finish", None))
            else:
                self.assertEqual(diagnosis.error_type, "false_positive_change")
                self.assertEqual(diagnosis.target_view, "t2")
                self.assertEqual((plan.action, plan.target_view), ("negative_point", "t2"))

    def test_v4_residual_contract_derives_negative_point_and_improvement(self):
        image=np.zeros((32,32,3),dtype=np.uint8);empty=np.zeros((32,32),dtype=bool)
        blob=empty.copy();blob[8:24,8:24]=True
        component,facts=residual_component(
            empty,blob,empty,source_view="t2",rank=0,
            expected_area=256,expected_gt_overlap=0,
        )
        component_again,facts_again=residual_component(
            empty,blob,empty,source_view="t2",rank=0,
            expected_area=256,expected_gt_overlap=0,
        )
        self.assertTrue(np.array_equal(component,component_again))
        self.assertEqual(facts,facts_again)
        self.assertEqual(facts["component_sha256"],component_sha256(blob))
        self.assertEqual(facts["gt_change_overlap_pixels"],0)

        case=unchanged_transition_case(image,image,component,source_view="t2")
        wrong=case["false_transition"];correct=case["corrected"]
        record,proposal=environment_record(wrong,expected_kind="present")
        annotation={"t1_state":"background","t2_state":"background","evidence":"T1 and T2 show tree canopy without a roofed building at this location."}
        target=audit_target(record.region_id,annotation,"false_positive_transition")
        payload={"target_class":"building","region":record.to_dict(),"proposal_catalog":[record.to_dict()],"schema":"environment_ordered_compact_region_audit_v12_1","visual_context":"proposal"}
        repair_prompt=_stage_prompt("audit",payload,WHITE_NO_TRANSITION_VALIDATION_ERROR)
        self.assertIn(
            "REPAIR: The previous response was rejected for this exact reason: "
            + WHITE_NO_TRANSITION_VALIDATION_ERROR,
            repair_prompt,
        )
        self.assertEqual(
            _stage_visual_input_scope("audit",payload,WHITE_NO_TRANSITION_VALIDATION_ERROR),
            "standard",
        )
        _,diagnosis=_parse_atomic_audit(target,record)
        plan=StagedQwenVerifier._plan(None,wrong,(record,),diagnosis)
        self.assertEqual(diagnosis.error_type,"false_positive_change")
        self.assertEqual(diagnosis.target_view,"t2")
        self.assertIsNotNone(plan)
        self.assertEqual((plan.action,plan.target_view),("negative_point","t2"))
        self.assertEqual(record.editable_seed_white,{"t1":False,"t2":True})
        self.assertEqual(
            target["diagnosis"]["audit_checklist"],
            {
                "evidence_sufficient":"pass",
                "target_class_only":"pass",
                "white_pixels_supported":"fail",
                "boundary_alignment":"fail",
                "internal_holes_absent":"not_applicable",
                "changed_object_extent_complete":"not_applicable",
                "fragment_artifacts_absent":"pass",
            },
        )
        corrected=with_action_locality(correct,proposal,action="negative_point")
        delta,_=environment_record(corrected,wrong,expected_kind="delta_removed")
        self.assertEqual(delta.audit_kind,"delta_removed")
        self.assertEqual(delta.component_area,256)
        self.assertEqual(delta.transition_mask_facts["previous_t2_delta_pixels"],256)
        self.assertEqual(delta.transition_mask_facts["candidate_t2_delta_pixels"],0)
        candidate_target={"region_id":delta.region_id,"visual_judgment":{"t1_state":"background","t2_state":"background","evidence_quality":"clear"}}
        judgment=_parse_judgment(candidate_target,delta.region_id)
        verifier=object.__new__(StagedQwenVerifier)
        verifier._planned_diagnosis=diagnosis
        verifier._planned_action=_plan_agent_action(plan,wrong.image_size)
        assessment=verifier._assess_candidate_transition(
            (delta,),(judgment,),verifier._planned_action
        )
        self.assertTrue(assessment.evidence_sufficient)
        self.assertTrue(assessment.intended_error_improved)
        self.assertFalse(assessment.introduced_harm)
        self.assertEqual(assessment.comparison,"better")

    def test_v3_transition_cases_are_one_sided_and_reversible(self):
        image=np.zeros((32,32,3),dtype=np.uint8);empty=np.zeros((32,32),dtype=bool)
        blob=empty.copy();blob[8:24,8:24]=True
        case=unchanged_transition_case(image,image,blob,source_view="t2")
        wrong=case["false_transition"];correct=case["corrected"]
        self.assertTrue(np.array_equal(np.logical_xor(wrong.t1_mask,wrong.t2_mask),blob))
        self.assertFalse(np.logical_xor(correct.t1_mask,correct.t2_mask)[blob].any())
        initial,proposal=environment_record(wrong,expected_kind="present")
        self.assertEqual(initial.audit_kind,"present")
        self.assertEqual(initial.component_area,blob.sum())
        self.assertNotEqual(initial.component_area,48*48)
        self.assertEqual(initial.editable_seed_white, {"t1":False,"t2":True})
        annotation={"t1_state":"building","t2_state":"building","evidence":"The same roofed building remains visible in both T1 and T2."}
        _,diagnosis=_parse_atomic_audit(audit_target(initial.region_id,annotation,"false_positive_transition"),initial)
        self.assertEqual(diagnosis.error_type,"false_positive_change")
        corrected=with_action_locality(correct,proposal,action="negative_point")
        delta,_=environment_record(corrected,wrong,expected_kind="delta_removed")
        self.assertEqual(delta.audit_kind,"delta_removed")
        self.assertEqual(delta.component_area,blob.sum())
        facts=delta.transition_mask_facts
        self.assertEqual(facts["previous_t2_delta_pixels"],int(blob.sum()))
        self.assertEqual(facts["candidate_t2_delta_pixels"],0)

    def test_hardcase_grid_matches_frozen_runtime_geometry(self):
        self.assertEqual(scan_grid_geometry(0), ((0, 0, 329, 329), (165, 165)))
        self.assertEqual(scan_grid_geometry(4), ((333, 333, 663, 663), (498, 498)))
        self.assertEqual(scan_grid_geometry(8), ((667, 667, 1000, 1000), (831, 831)))

    def test_manual_curation_is_parent_split_safe_and_balanced(self):
        path = Path(__file__).resolve().parents[1] / "configs/qwen3vl_v121_curation_0801.json"
        payload = json.loads(path.read_text())
        used = [row for row in payload["samples"] if row.get("use") is True]
        self.assertEqual(len(used), 15)
        train = [row for row in used if row["candidate_id"].startswith("train_")]
        val = [row for row in used if row["candidate_id"].startswith("val_")]
        self.assertEqual((len(train), len(val)), (9, 6))
        self.assertTrue(any(row["t1_state"] == "building" and row["t2_state"] == "background" for row in val))
        for row in used:
            self.assertIn(row["t1_state"], {"building", "background"})
            self.assertIn(row["t2_state"], {"building", "background"})
            self.assertIn("T1", row["evidence"])
            self.assertIn("T2", row["evidence"])
            self.assertNotIn("mask", row["evidence"].lower())
        v3 = payload["hardcase_v3"]
        self.assertEqual(len(v3["unchanged_transition_cases"]), 3)
        self.assertEqual(
            sorted(item["expected_area"] for item in v3["unchanged_transition_cases"]),
            [236, 464, 1205],
        )
        self.assertEqual(v3["grid_scan_tile_ids"], [0, 4, 8])
        v4 = payload["hardcase_v4"]
        cases = v4["residual_false_positive_cases"]
        self.assertEqual(len(cases), 2)
        self.assertEqual(len({item["candidate_id"] for item in cases}), 2)
        for item in cases:
            self.assertTrue(item["candidate_id"].startswith("train_"))
            self.assertIn(item["source_view"], {"t1", "t2"})
            self.assertGreaterEqual(item["expected_area"], 200)
            self.assertLessEqual(item["expected_area"], 2000)
            self.assertEqual(item["expected_gt_overlap"], 0)
            self.assertEqual((item["t1_state"], item["t2_state"]), ("background", "background"))
            self.assertIn("T1", item["evidence"])
            self.assertIn("T2", item["evidence"])

    def test_all_audit_label_templates_pass_runtime_parser(self):
        mask = np.zeros((32, 32), dtype=bool)
        mask[8:24, 8:24] = True
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        annotation = {"t1_state":"background", "t2_state":"building", "evidence":"T1 shows bare ground while T2 shows a roofed building at the same location."}
        for audit_kind, label_kind in (("present","correct"),("missing","false_negative")):
            change = mask if audit_kind == "present" else np.zeros_like(mask)
            state = ChangeState(image,image,"building",np.zeros_like(mask),mask,change)
            record = EvidenceRecord("r0",audit_kind,"white" if audit_kind=="present" else "black","present",(100,100,900,900),(500,500),0,256,256,{"t1":False,"t2":True},("positive_point","box"),{})
            target = audit_target("r0", annotation, label_kind)
            judgment, diagnosis = _parse_atomic_audit(target, record)
            self.assertEqual(judgment.region_id, "r0")
            self.assertEqual(diagnosis.error_type, "none" if label_kind=="correct" else "false_negative")
        self.assertEqual(checklist("false_positive")["white_pixels_supported"], "fail")
        self.assertEqual(checklist("false_positive_background")["target_class_only"], "fail")
        for states,label_kind in (("background","false_positive_background"),("building","false_positive_transition")):
            annotation={"t1_state":states,"t2_state":states,"evidence":"The same visible scene content remains in T1 and T2."}
            state=ChangeState(image,image,"building",np.zeros_like(mask),mask,mask)
            record=EvidenceRecord("r0","present","white","present",(100,100,900,900),(500,500),0,256,256,{"t1":False,"t2":True},("negative_point",),{})
            _,diagnosis=_parse_atomic_audit(audit_target("r0",annotation,label_kind),record)
            self.assertEqual(diagnosis.error_type,"false_positive_change")


if __name__ == "__main__":
    unittest.main()
