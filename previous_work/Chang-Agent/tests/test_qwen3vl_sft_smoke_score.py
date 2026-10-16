import json
import tempfile
import unittest
from pathlib import Path

from tools.score_qwen3vl_sft_smoke import (
    EXIT_ACCEPTANCE_FAILED,
    EXIT_PIPELINE_INCOMPLETE,
    EXPECTED_ROWS,
    main,
)


def _response(region_id: str, reason: str = "select the synthetic error region") -> str:
    return json.dumps(
        {"selection": {"region_ids": [region_id], "reason": reason}},
        separators=(",", ":"),
    )


class Qwen3VLSFTSmokeScoreTest(unittest.TestCase):
    def _paths(self, root: Path) -> dict[str, Path]:
        return {
            "predictions": root / "generated_predictions.jsonl",
            "train": root / "train_results.json",
            "eval": root / "eval_results.json",
            "predict": root / "predict_results.json",
            "metrics": root / "smoke_metrics.json",
            "manifest": root / "smoke_manifest.json",
            "summary": root / "completion_summary.json",
        }

    def _rows(self) -> list[dict[str, str]]:
        rows = []
        for index in range(EXPECTED_ROWS):
            target = _response(f"r{index % 4}")
            rows.append({"predict": target, "label": target})
        return rows

    def _write_inputs(self, paths: dict[str, Path], rows: list[object]) -> None:
        paths["predictions"].write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
        paths["train"].write_text(json.dumps({"train_loss": 1.25}), encoding="utf-8")
        paths["eval"].write_text(
            json.dumps({"eval_loss": 0.75, "eval_rouge-l": 88.0}), encoding="utf-8"
        )
        paths["predict"].write_text(
            json.dumps({"predict_loss": 0.5, "predict_bleu-4": 77.0}), encoding="utf-8"
        )

    def _run(self, root: Path, paths: dict[str, Path]) -> int:
        return main(
            [
                "--generated-predictions",
                str(paths["predictions"]),
                "--train-metrics",
                str(paths["train"]),
                "--eval-metrics",
                str(paths["eval"]),
                "--predict-metrics",
                str(paths["predict"]),
                "--smoke-metrics-output",
                str(paths["metrics"]),
                "--smoke-manifest-output",
                str(paths["manifest"]),
                "--completion-summary-output",
                str(paths["summary"]),
                "--artifact-dir",
                str(root),
                "--local-output-dir",
                str(root),
                "--dataset",
                "levir_qwen3vl_sft32_train",
                "--eval-dataset",
                "levir_qwen3vl_sft32_eval",
                "--job-id",
                "unit-test",
                "--llamafactory-commit",
                "a" * 40,
            ]
        )

    def _load_reports(self, paths: dict[str, Path]) -> tuple[dict, dict, dict]:
        return tuple(
            json.loads(paths[name].read_text(encoding="utf-8"))
            for name in ("metrics", "manifest", "summary")
        )

    def test_complete_exact_predictions_pass_both_statuses(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._paths(root)
            self._write_inputs(paths, self._rows())

            self.assertEqual(self._run(root, paths), 0)
            metrics, manifest, summary = self._load_reports(paths)
            self.assertEqual(metrics["pipeline_status"], "success")
            self.assertEqual(metrics["acceptance_status"], "passed")
            self.assertEqual(metrics["raw_nonempty_line_count"], EXPECTED_ROWS)
            self.assertEqual(metrics["json_parsed_line_count"], EXPECTED_ROWS)
            self.assertEqual(metrics["json_dict_row_count"], EXPECTED_ROWS)
            self.assertEqual(metrics["strict_json_count"], EXPECTED_ROWS)
            self.assertEqual(metrics["schema_valid_count"], EXPECTED_ROWS)
            self.assertEqual(metrics["unknown_region_id_count"], 0)
            self.assertEqual(metrics["region_id_accuracy"], 1.0)
            self.assertEqual(metrics["exact_match_rate"], 1.0)
            self.assertTrue(metrics["protocol_gate"]["passed"])
            self.assertEqual(manifest["exit_code"], 0)
            self.assertEqual(summary["acceptance_status"], "passed")

    def test_outer_non_dict_row_is_pipeline_incomplete_and_reports_survive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._paths(root)
            rows: list[object] = self._rows()
            rows[-1] = []
            self._write_inputs(paths, rows)

            self.assertEqual(self._run(root, paths), EXIT_PIPELINE_INCOMPLETE)
            metrics, manifest, summary = self._load_reports(paths)
            self.assertEqual(metrics["pipeline_status"], "failed")
            self.assertEqual(metrics["acceptance_status"], "not_evaluated")
            self.assertEqual(metrics["raw_nonempty_line_count"], EXPECTED_ROWS)
            self.assertEqual(metrics["json_parsed_line_count"], EXPECTED_ROWS)
            self.assertEqual(metrics["json_dict_row_count"], EXPECTED_ROWS - 1)
            self.assertEqual(metrics["strict_json_rate"], (EXPECTED_ROWS - 1) / EXPECTED_ROWS)
            self.assertEqual(metrics["rate_denominator"], EXPECTED_ROWS)
            self.assertTrue(metrics["pipeline_errors"])
            self.assertEqual(manifest["exit_code"], EXIT_PIPELINE_INCOMPLETE)
            self.assertEqual(summary["pipeline_status"], "failed")

    def test_fence_and_json_wrapper_are_not_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._paths(root)
            rows = self._rows()
            rows[0]["predict"] = f"```json\n{rows[0]['predict']}\n```"
            rows[1]["predict"] = json.dumps({"answer": json.loads(rows[1]["predict"])})
            self._write_inputs(paths, rows)

            self.assertEqual(self._run(root, paths), EXIT_ACCEPTANCE_FAILED)
            metrics, manifest, summary = self._load_reports(paths)
            self.assertEqual(metrics["pipeline_status"], "success")
            self.assertEqual(metrics["acceptance_status"], "failed")
            self.assertEqual(metrics["strict_json_count"], EXPECTED_ROWS - 1)
            self.assertEqual(metrics["schema_valid_count"], EXPECTED_ROWS - 2)
            self.assertEqual(metrics["unknown_region_id_count"], 0)
            self.assertEqual(metrics["strict_json_rate"], (EXPECTED_ROWS - 1) / EXPECTED_ROWS)
            self.assertEqual(metrics["schema_valid_rate"], (EXPECTED_ROWS - 2) / EXPECTED_ROWS)
            self.assertFalse(metrics["protocol_gate"]["passed"])
            self.assertEqual(manifest["exit_code"], EXIT_ACCEPTANCE_FAILED)
            self.assertEqual(summary["acceptance_status"], "failed")

    def test_unknown_region_is_a_separate_protocol_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._paths(root)
            rows = self._rows()
            rows[0]["predict"] = _response("r9")
            self._write_inputs(paths, rows)

            self.assertEqual(self._run(root, paths), EXIT_ACCEPTANCE_FAILED)
            metrics, _, summary = self._load_reports(paths)
            self.assertEqual(metrics["strict_json_count"], EXPECTED_ROWS)
            self.assertEqual(metrics["schema_valid_count"], EXPECTED_ROWS)
            self.assertEqual(metrics["unknown_region_id_count"], 1)
            self.assertEqual(metrics["unknown_region_id_rate"], 1 / EXPECTED_ROWS)
            self.assertEqual(metrics["region_id_correct_count"], EXPECTED_ROWS - 1)
            self.assertEqual(metrics["exact_match_count"], EXPECTED_ROWS - 1)
            self.assertEqual(summary["pipeline_status"], "success")
            self.assertEqual(summary["acceptance_status"], "failed")


if __name__ == "__main__":
    unittest.main()
