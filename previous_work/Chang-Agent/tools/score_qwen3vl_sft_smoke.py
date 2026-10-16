#!/usr/bin/env python3
"""Score the fixed 32-row Qwen3-VL SFT overfit smoke contract.

Exit codes are part of the worker contract:

* 0: the pipeline is complete and the response protocol gate passes;
* 3: an input artifact or the outer 32-row JSONL contract is incomplete;
* 4: all 32 JSONL rows are present dictionaries, but the protocol gate fails.

The three JSON reports are written before returning either scoring failure.
Response parsing deliberately does not remove Markdown fences or extract JSON
substrings, so wrapped output is never silently accepted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Sequence


EXPECTED_ROWS = 32
ALLOWED_REGION_IDS = frozenset({"r0", "r1", "r2", "r3"})
EXIT_PIPELINE_INCOMPLETE = 3
EXIT_ACCEPTANCE_FAILED = 4


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generated-predictions",
        "--predictions",
        dest="generated_predictions",
        type=Path,
        required=True,
    )
    parser.add_argument("--train-metrics", type=Path, required=True)
    parser.add_argument("--eval-metrics", type=Path, required=True)
    parser.add_argument("--predict-metrics", type=Path, required=True)
    parser.add_argument("--smoke-metrics-output", type=Path, required=True)
    parser.add_argument("--smoke-manifest-output", type=Path, required=True)
    parser.add_argument("--completion-summary-output", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--local-output-dir", type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--eval-dataset", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--model-path", default="")
    parser.add_argument("--train-config", default="")
    parser.add_argument("--llamafactory-commit", default="")
    return parser.parse_args(argv)


def _read_metrics(path: Path, name: str, errors: list[str]) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        errors.append(f"{name} metrics file is missing: {path}")
        return {}
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        errors.append(f"{name} metrics file is unreadable JSON: {path}: {error}")
        return {}
    if not isinstance(payload, dict):
        errors.append(f"{name} metrics must be a JSON object: {path}")
        return {}
    return payload


def _finite_metrics(payload: dict[str, Any]) -> dict[str, int | float]:
    result: dict[str, int | float] = {}
    for key, value in payload.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if math.isfinite(value):
            result[str(key)] = value
    return result


def _read_prediction_rows(
    path: Path, errors: list[str]
) -> tuple[int, int, list[tuple[int, dict[str, Any]]]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        errors.append(f"generated predictions file is missing: {path}")
        return 0, 0, []
    except (OSError, UnicodeError) as error:
        errors.append(f"generated predictions file is unreadable: {path}: {error}")
        return 0, 0, []

    nonempty = [(line_number, line) for line_number, line in enumerate(lines, 1) if line.strip()]
    parsed_count = 0
    dict_rows: list[tuple[int, dict[str, Any]]] = []
    for line_number, line in nonempty:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            errors.append(f"prediction JSONL line {line_number} is invalid JSON: {error.msg}")
            continue
        parsed_count += 1
        if not isinstance(payload, dict):
            errors.append(
                f"prediction JSONL line {line_number} must decode to an object, "
                f"found {type(payload).__name__}"
            )
            continue
        dict_rows.append((line_number, payload))

    if len(nonempty) != EXPECTED_ROWS:
        errors.append(
            f"expected {EXPECTED_ROWS} raw nonempty prediction lines, found {len(nonempty)}"
        )
    if parsed_count != EXPECTED_ROWS:
        errors.append(
            f"expected {EXPECTED_ROWS} JSON-parsed prediction lines, found {parsed_count}"
        )
    if len(dict_rows) != EXPECTED_ROWS:
        errors.append(
            f"expected {EXPECTED_ROWS} JSON object prediction rows, found {len(dict_rows)}"
        )
    return len(nonempty), parsed_count, dict_rows


def _parse_response(text: Any) -> dict[str, Any] | None:
    """Parse only a complete JSON object; never strip fences or text wrappers."""

    if not isinstance(text, str) or not text.strip():
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _schema_region_ids(payload: dict[str, Any]) -> list[str] | None:
    """Validate structure; the allowed-ID check is a separate protocol gate."""

    if set(payload) != {"selection"}:
        return None
    selection = payload.get("selection")
    if not isinstance(selection, dict) or set(selection) != {"region_ids", "reason"}:
        return None
    region_ids = selection.get("region_ids")
    reason = selection.get("reason")
    if (
        not isinstance(region_ids, list)
        or len(region_ids) != 1
        or not all(isinstance(item, str) for item in region_ids)
        or not isinstance(reason, str)
        or not reason.strip()
    ):
        return None
    return region_ids


def _label_region_ids(label_text: Any) -> list[str] | None:
    payload = _parse_response(label_text)
    if payload is None:
        return None
    region_ids = _schema_region_ids(payload)
    if region_ids is None or any(item not in ALLOWED_REGION_IDS for item in region_ids):
        return None
    return region_ids


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _artifact_names(
    artifact_dir: Path | None, output_paths: Sequence[Path]
) -> list[str]:
    names: set[str] = set()
    if artifact_dir is not None and artifact_dir.is_dir():
        names.update(path.name for path in artifact_dir.iterdir() if path.is_file())
    if artifact_dir is not None:
        resolved_dir = artifact_dir.resolve(strict=False)
        for path in output_paths:
            if path.parent.resolve(strict=False) == resolved_dir:
                names.add(path.name)
    return sorted(names)


def _local_file_names(local_output_dir: Path | None) -> list[str]:
    if local_output_dir is None or not local_output_dir.is_dir():
        return []
    return sorted(path.name for path in local_output_dir.iterdir() if path.is_file())


def score(args: argparse.Namespace) -> tuple[int, str]:
    pipeline_errors: list[str] = []
    train_metrics = _read_metrics(args.train_metrics, "train", pipeline_errors)
    eval_metrics = _read_metrics(args.eval_metrics, "eval", pipeline_errors)
    predict_metrics = _read_metrics(args.predict_metrics, "predict", pipeline_errors)
    raw_line_count, parsed_count, rows = _read_prediction_rows(
        args.generated_predictions, pipeline_errors
    )

    strict_json_count = 0
    schema_valid_count = 0
    unknown_region_id_count = 0
    region_id_correct_count = 0
    exact_match_count = 0

    for line_number, row in rows:
        prediction_text = row.get("predict")
        label_text = row.get("label")
        expected_ids = _label_region_ids(label_text)
        if expected_ids is None:
            pipeline_errors.append(
                f"prediction JSONL line {line_number} has no valid select-stage label"
            )

        if isinstance(prediction_text, str) and isinstance(label_text, str):
            if prediction_text.strip() == label_text.strip():
                exact_match_count += 1

        prediction = _parse_response(prediction_text)
        if prediction is None:
            continue
        strict_json_count += 1
        region_ids = _schema_region_ids(prediction)
        if region_ids is None:
            continue
        schema_valid_count += 1
        has_unknown = any(item not in ALLOWED_REGION_IDS for item in region_ids)
        if has_unknown:
            unknown_region_id_count += 1
        if not has_unknown and expected_ids is not None and region_ids == expected_ids:
            region_id_correct_count += 1

    pipeline_status = "success" if not pipeline_errors else "failed"
    gate_observed = {
        "strict_json_count": strict_json_count,
        "schema_valid_count": schema_valid_count,
        "unknown_region_id_count": unknown_region_id_count,
    }
    gate_required = {
        "strict_json_count": EXPECTED_ROWS,
        "schema_valid_count": EXPECTED_ROWS,
        "unknown_region_id_count": 0,
    }
    gate_passed = gate_observed == gate_required
    if pipeline_status != "success":
        acceptance_status = "not_evaluated"
        exit_code = EXIT_PIPELINE_INCOMPLETE
    elif gate_passed:
        acceptance_status = "passed"
        exit_code = 0
    else:
        acceptance_status = "failed"
        exit_code = EXIT_ACCEPTANCE_FAILED

    denominator = EXPECTED_ROWS
    protocol_failures = [
        f"{name}: expected {gate_required[name]}, found {gate_observed[name]}"
        for name in gate_required
        if gate_observed[name] != gate_required[name]
    ]
    status = "success" if exit_code == 0 else "failed"
    smoke_metrics: dict[str, Any] = {
        "status": status,
        "pipeline_status": pipeline_status,
        "acceptance_status": acceptance_status,
        "exit_code": exit_code,
        "dataset": args.dataset,
        "eval_dataset": args.eval_dataset,
        "rows": EXPECTED_ROWS,
        "rate_denominator": EXPECTED_ROWS,
        "expected_prediction_count": EXPECTED_ROWS,
        "raw_nonempty_line_count": raw_line_count,
        "json_parsed_line_count": parsed_count,
        "json_dict_row_count": len(rows),
        "prediction_count": len(rows),
        "strict_json_count": strict_json_count,
        "strict_json_rate": strict_json_count / denominator,
        "schema_valid_count": schema_valid_count,
        "schema_valid_rate": schema_valid_count / denominator,
        "unknown_region_id_count": unknown_region_id_count,
        "unknown_region_id_rate": unknown_region_id_count / denominator,
        "region_id_correct_count": region_id_correct_count,
        "region_id_accuracy": region_id_correct_count / denominator,
        "exact_match_count": exact_match_count,
        "exact_match_rate": exact_match_count / denominator,
        "protocol_gate": {
            "passed": gate_passed if pipeline_status == "success" else False,
            "required": gate_required,
            "observed": gate_observed,
            "failures": protocol_failures,
        },
        "pipeline_errors": pipeline_errors,
        "train_loss": train_metrics.get("train_loss"),
        "eval_loss": eval_metrics.get("eval_loss"),
        "predict_loss": predict_metrics.get("predict_loss"),
        "train_metrics": _finite_metrics(train_metrics),
        "eval_metrics": _finite_metrics(eval_metrics),
        "predict_metrics": _finite_metrics(predict_metrics),
    }

    key_result = (
        f"pipeline={pipeline_status};acceptance={acceptance_status};"
        f"rows={len(rows)}/{EXPECTED_ROWS};"
        f"strict_json={strict_json_count}/{EXPECTED_ROWS};"
        f"schema={schema_valid_count}/{EXPECTED_ROWS};"
        f"unknown={unknown_region_id_count};"
        f"region_id={region_id_correct_count}/{EXPECTED_ROWS};"
        f"exact={exact_match_count}/{EXPECTED_ROWS}"
    )
    output_paths = (
        args.smoke_metrics_output,
        args.smoke_manifest_output,
        args.completion_summary_output,
    )
    summary: dict[str, Any] = {
        "status": status,
        "pipeline_status": pipeline_status,
        "acceptance_status": acceptance_status,
        "exit_code": exit_code,
        "job_id": args.job_id,
        "rows": EXPECTED_ROWS,
        "raw_nonempty_line_count": raw_line_count,
        "json_dict_row_count": len(rows),
        "dataset": args.dataset,
        "eval_dataset": args.eval_dataset,
        "overfit_smoke": True,
        "adapter_retained": False,
        "key_result": key_result,
        "protocol_gate": smoke_metrics["protocol_gate"],
        "pipeline_errors": pipeline_errors,
        "model_path": args.model_path,
        "train_config": args.train_config,
        "llamafactory_commit": args.llamafactory_commit,
        "artifacts": _artifact_names(args.artifact_dir, output_paths),
        "local_adapter_files": _local_file_names(args.local_output_dir),
    }

    _atomic_write_json(args.smoke_metrics_output, smoke_metrics)
    _atomic_write_json(args.smoke_manifest_output, summary)
    _atomic_write_json(args.completion_summary_output, summary)
    return exit_code, key_result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    exit_code, key_result = score(args)
    print(key_result)
    if exit_code != 0:
        print(
            f"qwen3vl_sft_smoke_score: failed with exit code {exit_code}; "
            f"reports retained at {args.smoke_metrics_output.parent}",
            file=sys.stderr,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
