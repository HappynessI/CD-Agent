import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tools.build_qwen3vl_sft_smoke import (
    DATASET_NAMES,
    PROPOSALS_PER_SAMPLE,
    SAMPLES_PER_CLASS,
    TOTAL_SAMPLES,
    PatchRecord,
    _assistant_target,
    _build_one,
    _proposal_catalog,
    _stage_payload,
    choose_smoke_records,
    dataset_info_payload,
    validate_output,
)
from change_agent.adapters.stage_backends import _stage_prompt


class Qwen3VLSFTSmokeDataTest(unittest.TestCase):
    def test_selection_is_balanced_and_parent_grouped(self):
        dummy = Path("/unused")
        records = []
        for index in range(80):
            records.append(
                PatchRecord(
                    parent_id=f"positive_{index}",
                    parent_index=index,
                    row=index % 4,
                    column=(index // 4) % 4,
                    gt_pixels=64 + index * 11,
                    t1_path=dummy,
                    t2_path=dummy,
                    label_path=dummy,
                )
            )
            records.append(
                PatchRecord(
                    parent_id=f"empty_{index}",
                    parent_index=1000 + index,
                    row=index % 4,
                    column=(index // 4) % 4,
                    gt_pixels=0,
                    t1_path=dummy,
                    t2_path=dummy,
                    label_path=dummy,
                )
            )
        positives, empties = choose_smoke_records(records, seed=729)
        self.assertEqual(len(positives), SAMPLES_PER_CLASS)
        self.assertEqual(len(empties), SAMPLES_PER_CLASS)
        parents = {item.parent_id for item in positives + empties}
        self.assertEqual(len(parents), TOTAL_SAMPLES)
        self.assertTrue(all(item.gt_pixels > 0 for item in positives))
        self.assertTrue(all(item.gt_pixels == 0 for item in empties))

    def test_select_contract_has_four_shuffled_regions_and_no_target_view(self):
        gt = np.zeros((256, 256), dtype=bool)
        gt[64:192, 80:176] = True
        candidate = np.array(gt, copy=True)
        candidate[96:128, 112:144] = False
        target_ids = set()
        for index in range(12):
            sample_id = f"train_{index}_r0c0"
            catalog, target_id = _proposal_catalog(
                candidate,
                gt,
                "false_negative",
                sample_id,
                t1_mask=gt,
                t2_mask=np.zeros_like(gt),
            )
            target_ids.add(target_id)
            self.assertEqual(len(catalog), PROPOSALS_PER_SAMPLE)
            self.assertEqual([item["region_id"] for item in catalog], ["r0", "r1", "r2", "r3"])
            prompt = _stage_prompt("select", _stage_payload(catalog))
            target = _assistant_target("false_negative", target_id)
            parsed = json.loads(target)
            self.assertEqual(set(parsed), {"selection"})
            self.assertEqual(set(parsed["selection"]), {"region_ids", "reason"})
            self.assertEqual(parsed["selection"]["region_ids"], [target_id])
            self.assertNotIn("target_view", prompt + target)
            self.assertNotIn('"action"', target)
            self.assertNotIn('"coordinate"', target)
        self.assertGreaterEqual(len(target_ids), 2)

    def test_build_one_writes_one_absolute_runtime_style_overview(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            t1_path = root / "train_1.png"
            t2_path = root / "train_1_t2.png"
            label_path = root / "train_1_label.png"
            Image.fromarray(np.full((256, 256, 3), 90, dtype=np.uint8)).save(t1_path)
            Image.fromarray(np.full((256, 256, 3), 110, dtype=np.uint8)).save(t2_path)
            label = np.zeros((256, 256), dtype=np.uint8)
            label[64:192, 80:176] = 255
            Image.fromarray(label).save(label_path)
            record = PatchRecord(
                parent_id="train_1",
                parent_index=1,
                row=0,
                column=0,
                gt_pixels=int((label > 0).sum()),
                t1_path=t1_path,
                t2_path=t2_path,
                label_path=label_path,
            )
            images_dir = root / "images"
            images_dir.mkdir()
            example, manifest = _build_one(record, "false_negative", images_dir)
            self.assertEqual(example["conversations"][0]["value"].count("<image>"), 1)
            self.assertEqual(len(example["images"]), 1)
            image_path = Path(example["images"][0])
            self.assertTrue(image_path.is_absolute())
            self.assertTrue(image_path.is_file())
            with Image.open(image_path) as overview:
                self.assertEqual(overview.size, (256 * 3, 256))
            self.assertEqual(manifest["proposal_count"], PROPOSALS_PER_SAMPLE)
            self.assertGreater(manifest["synthetic_error_pixels"], 0)
            self.assertLess(manifest["candidate_change_pixels"], manifest["gt_change_pixels"])

    def test_validator_enforces_exact_32_example_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            images_dir = output / "images"
            images_dir.mkdir()
            examples = []
            samples = []
            for index in range(TOTAL_SAMPLES):
                image_path = (images_dir / f"{index}.png").resolve()
                Image.new("RGB", (256 * 3, 256), (index, 0, 0)).save(image_path)
                error_type = (
                    "false_negative"
                    if index < SAMPLES_PER_CLASS
                    else "false_positive_change"
                )
                target_id = f"r{index % PROPOSALS_PER_SAMPLE}"
                sample_id = f"sample_{index}"
                examples.append(
                    {
                        "id": sample_id,
                        "conversations": [
                            {"from": "human", "value": "<image>\nselect one region"},
                            {"from": "gpt", "value": _assistant_target(error_type, target_id)},
                        ],
                        "images": [str(image_path)],
                    }
                )
                samples.append(
                    {
                        "id": sample_id,
                        "parent_id": f"train_{index}",
                        "synthetic_error_type": error_type,
                        "proposal_count": PROPOSALS_PER_SAMPLE,
                        "selected_region_ids": [target_id],
                    }
                )
            payloads = {
                "sharegpt.json": examples,
                "dataset_info.json": dataset_info_payload(),
                "manifest.json": {
                    "dataset_names": list(DATASET_NAMES),
                    "source_split": "train_original",
                    "test_data_used": False,
                    "raw_gt_label_image_model_visible": False,
                    "gt_derived_synthetic_facts_model_visible": True,
                    "samples": samples,
                },
            }
            for name, payload in payloads.items():
                with (output / name).open("w", encoding="utf-8") as handle:
                    json.dump(payload, handle)
            report = validate_output(output)
            self.assertTrue(report["valid"], report["errors"])
            self.assertEqual(report["examples"], TOTAL_SAMPLES)
            self.assertEqual(report["unique_parent_scenes"], TOTAL_SAMPLES)

            samples[0]["selected_region_ids"] = [
                f"r{(int(samples[0]['selected_region_ids'][0][1:]) + 1) % PROPOSALS_PER_SAMPLE}"
            ]
            with (output / "manifest.json").open("w", encoding="utf-8") as handle:
                json.dump(payloads["manifest.json"], handle)
            mismatch_report = validate_output(output)
            self.assertFalse(mismatch_report["valid"])
            self.assertTrue(
                any("ShareGPT target and manifest target disagree" in error for error in mismatch_report["errors"])
            )


if __name__ == "__main__":
    unittest.main()
