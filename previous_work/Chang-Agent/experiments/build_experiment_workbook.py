#!/usr/bin/env python3
"""Build and audit the maintained Change-Agent experiment workbook.

The implementation intentionally uses only the Python standard library so the
registry can be rebuilt on the management node without installing packages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "experiments"
SOURCE = EXPERIMENT_DIR / "experiment_registry.json"
OUTPUT = EXPERIMENT_DIR / "experiment_registry.xlsx"
SMOKE_METRICS = (
    ROOT / "outputs/CA_0729(2)-qwen3vl-sft32-smoke/smoke_metrics.json"
)
SMOKE_ROOT = ROOT / "outputs/CA_0729(2)-qwen3vl-sft32-smoke"
TEST_ROOT = Path(
    "/guisongxia01/pangchao/wangyihan/datasets/LEVIR-CD/test_256"
)
V2_RECONSTRUCTED_CONFIG = (
    ROOT
    / "doc/training_pipeline/0802_third_try/"
    "QWEN3VL_V2_RECONSTRUCTED_RESOLVED_CONFIG_0802.json"
)
P1_RETRY3_RESULT = (
    ROOT
    / "doc/training_pipeline/0802_third_try/"
    "QWEN3VL_P1_RETRY3_RESULT_0802.json"
)
P1_RETRY3_TRAJECTORY_ANALYSIS = Path(
    "/guisongxia01/pangchao/wangyihan/wyh/diagnostics/"
    "qwen3vl_p1_retry3_result_0802/trajectory_analysis.json"
)
HISTORICAL_REPOSITORY_ROOT = Path(
    "/guisongxia01/pangchao/wangyihan/Chang-Agent-local-qwen3vl"
)
SUPPORTED_ARCHIVE_DATES = {"0801", "0802"}
ARCHIVED_EXPERIMENT_DATES = {
    "0802(P1-H0)": "0802",
    "0802(P1-H1)": "0802",
}
P2_ACTIONABILITY_RESULT = (
    ROOT
    / "doc/training_pipeline/0803_fourth_try/"
    "QWEN3VL_P2_ACTIONABILITY_RESULT_0803.json"
)
P2_FORMAL_ATTEMPT = (
    ROOT / "outputs/CA_0803(1)-qwen3vl-p2-oracle-actionability/attempt-0001"
)
P2_IMPLEMENTATION_COMMIT = "8252a57b191a2a7c4aa176a3f82cca5218119533"
P2_FREEZE_COMMIT = "cda314ce9caeb6e9dd69ba0d08fe5254a8689fdd"
P2_GENERATION_JOB = "53272"
P2_SCORING_JOB = "53273"
P2_BINDING_SHA256 = {
    "submission_intent.json": "<hash omitted>",
    "submission_receipt.json": "<hash omitted>",
    "generation_receipt.json": "<hash omitted>",
    "generation/candidate_generation_complete.json": "<hash omitted>",
    "generation/candidates.json": "<hash omitted>",
    "scoring_receipt.json": "<hash omitted>",
    "scoring/runtime_scoring_manifest.json": "<hash omitted>",
    "scoring/score_report.json": "<hash omitted>",
    "scoring/actionability_decision.json": "<hash omitted>",
}

FIXED_EVALUATION_ORDERS = {
    "DEV-FIXED-3-V1": ["test_20_15", "test_78_13", "test_85_16"],
    "DEV-SMOKE-1-H1-V1": ["test_1_1"],
    "DEV-CORE-3-V1": ["test_20_15", "test_1_1", "test_50_8"],
    "DEV-EXTENDED-5-V1": [
        "test_20_15",
        "test_78_13",
        "test_85_16",
        "test_1_1",
        "test_50_8",
    ],
}

EXPECTED_SHEETS = [
    "01_实验记录",
    "02_字段字典",
    "03_数据登记",
    "04_运行尝试",
]
EXPECTED_HEADERS = {
    "01_实验记录": (
        "实验ID",
        "运行记录ID",
        "实际执行日期UTC",
        "记录阶段",
        "execution_status",
        "scientific_outcome",
        "lineage_status",
        "状态说明",
        "实验目的",
        "唯一变量",
        "控制变量/可比性",
        "Verifier功能焦点",
        "schema/protocol",
        "训练数据ID",
        "功能评测ID",
        "端到端评测ID",
        "基座模型",
        "模型状态",
        "训练方法",
        "LoRA rank",
        "LoRA alpha",
        "LoRA dropout",
        "LoRA target",
        "冻结模块",
        "batch/train×accum=effective",
        "learning_rate",
        "scheduler/warmup",
        "configured epochs",
        "max optimizer steps",
        "actual epochs",
        "precision/checkpointing",
        "seed/template/cutoff",
        "generation config",
        "trainable params",
        "train loss",
        "eval/predict loss",
        "adapter保留",
        "功能主指标",
        "训练前功能状态",
        "训练前主指标",
        "训练前附加指标JSON",
        "训练后功能状态",
        "训练后主指标",
        "训练后附加指标JSON",
        "功能Δ",
        "E2E请求样本",
        "E2E实际评分样本",
        "训练前E2E状态",
        "前IoU",
        "前Precision",
        "前Recall",
        "前F1",
        "训练后E2E状态",
        "后IoU",
        "后Precision",
        "后Recall",
        "后F1",
        "ΔIoU",
        "ΔF1",
        "动作/accepted/rejected/finish",
        "数据泄漏与捷径",
        "资源与耗时",
        "版本记录",
        "历史运行路径",
        "当前证据路径",
        "权威证据",
        "结论/限制",
        "登记完整性",
    ),
    "02_字段字典": (
        "类别",
        "功能/阶段",
        "字段/合同",
        "合法值/类型",
        "精确定义",
        "责任方/来源",
        "模型关系",
        "GT关系",
        "当前注意",
        "代码证据",
    ),
    "03_数据登记": (
        "记录类型",
        "数据ID",
        "用途",
        "标准状态",
        "LEVIR-CD根目录",
        "source split",
        "sample_id/顺序",
        "parent scene/tile",
        "crop_xyxy_exclusive",
        "样本数",
        "GT正像素",
        "选择/构造规则",
        "简明构造或评测pipeline",
        "GT访问策略",
        "A_SHA256",
        "B_SHA256",
        "label_SHA256",
        "关联实验",
        "历史运行路径",
        "当前证据路径",
        "权威证据",
        "限制/备注",
    ),
    "04_运行尝试": (
        "attempt_id",
        "实验ID",
        "retry_of_attempt_id",
        "depends_on_attempt_ids",
        "stage",
        "Job ID",
        "execution_status",
        "scientific_outcome",
        "lineage_status",
        "failure_class",
        "实际执行日期UTC",
        "代码commit",
        "历史运行路径",
        "当前证据路径",
        "配置/模型/工具哈希",
        "核心指标/结果",
        "资源与耗时",
        "权威证据",
        "限制/备注",
    ),
}
MISSING_MARKERS = {"未运行", "未记录", "未测量", "不适用", "null"}
EXECUTION_STATUSES = {"planned", "running", "completed", "failed", "cancelled"}
SCIENTIFIC_OUTCOMES = {
    "not_evaluated",
    "baseline_reference",
    "pass",
    "negative",
    "inconclusive",
}
LINEAGE_STATUSES = {"canonical", "superseded"}
FAILURE_CLASSES = {
    "not_applicable",
    "infra_failed",
    "contract_failed",
    "test_failed",
}
FAILURE_EXECUTION_CLASSES = FAILURE_CLASSES - {"not_applicable"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
INVALID_XML_RE = re.compile(
    "[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]"
)

CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
  {sheet_overrides}
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
"""

ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""

STYLES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="4">
    <font><sz val="10"/><name val="Aptos"/></font>
    <font><b/><color rgb="FFFFFFFF"/><sz val="15"/><name val="Aptos Display"/></font>
    <font><b/><color rgb="FFFFFFFF"/><sz val="10"/><name val="Aptos"/></font>
    <font><i/><color rgb="FF404040"/><sz val="10"/><name val="Aptos"/></font>
  </fonts>
  <fills count="6">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF17365D"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FF2F75B5"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFD9EAF7"/><bgColor indexed="64"/></patternFill></fill>
    <fill><patternFill patternType="solid"><fgColor rgb="FFFFF2CC"/><bgColor indexed="64"/></patternFill></fill>
  </fills>
  <borders count="2">
    <border><left/><right/><top/><bottom/><diagonal/></border>
    <border><left style="thin"><color rgb="FFD9E1F2"/></left><right style="thin"><color rgb="FFD9E1F2"/></right><top style="thin"><color rgb="FFD9E1F2"/></top><bottom style="thin"><color rgb="FFD9E1F2"/></bottom><diagonal/></border>
  </borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="7">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0"><alignment vertical="center"/></xf>
    <xf numFmtId="0" fontId="3" fillId="4" borderId="0" xfId="0"><alignment wrapText="1" vertical="center"/></xf>
    <xf numFmtId="0" fontId="2" fillId="3" borderId="1" xfId="0"><alignment wrapText="1" vertical="center" horizontal="center"/></xf>
    <xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0"><alignment wrapText="1" vertical="top"/></xf>
    <xf numFmtId="0" fontId="0" fillId="5" borderId="1" xfId="0"><alignment wrapText="1" vertical="top"/></xf>
    <xf numFmtId="0" fontId="0" fillId="0" borderId="1" xfId="0"><alignment vertical="top"/><protection locked="1"/></xf>
  </cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate source, artifacts, and workbook without rewriting it",
    )
    return parser.parse_args()


def load_registry() -> dict[str, Any]:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "chang_agent_experiment_registry_v2":
        raise ValueError("unsupported experiment registry schema")
    sheets = payload.get("sheets")
    if not isinstance(sheets, list) or [item.get("name") for item in sheets] != EXPECTED_SHEETS:
        raise ValueError(f"registry must contain exactly {EXPECTED_SHEETS}")
    for sheet in sheets:
        headers = sheet.get("headers")
        rows = sheet.get("rows")
        expected_headers = EXPECTED_HEADERS[sheet["name"]]
        if not isinstance(headers, list) or tuple(headers) != expected_headers:
            raise ValueError(
                f"{sheet['name']} headers changed: {headers!r} != {expected_headers!r}"
            )
        if not isinstance(rows, list):
            raise ValueError(f"{sheet['name']} rows must be a list")
        expected = set(headers)
        for index, row in enumerate(rows, start=1):
            if set(row) != expected:
                missing = sorted(expected - set(row))
                extra = sorted(set(row) - expected)
                raise ValueError(
                    f"{sheet['name']} row {index} key mismatch: missing={missing}, extra={extra}"
                )
            for header, value in row.items():
                if value is None or value == "":
                    raise ValueError(
                        f"{sheet['name']} row {index} column {header!r} is blank; use an explicit missing marker"
                    )
    run_ids = [row["运行记录ID"] for row in sheets[0]["rows"]]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("运行记录ID must be unique")
    experiment_ids = [row["实验ID"] for row in sheets[0]["rows"]]
    if len(experiment_ids) != len(set(experiment_ids)):
        raise ValueError("实验ID must be unique")
    for row in sheets[0]["rows"]:
        validate_status_axes(row, context=f"experiment {row['实验ID']}")
        validate_path_columns(
            row,
            context=f"experiment {row['实验ID']}",
            archive_date=archive_date_for_experiment(row["实验ID"]),
            exact_archive_mapping=row["实验ID"] in ARCHIVED_EXPERIMENT_DATES,
            allow_missing_current=row["实验ID"] in {"0729(1)", "0730(4)"},
        )

    for row in sheets[2]["rows"]:
        data_id = row["数据ID"]
        validate_path_columns(
            row,
            context=f"data {data_id}",
            archive_date="0801" if data_id.startswith("TRAIN-0801-") else None,
            execution_status="completed",
            allow_missing_current=(
                row["记录类型"] != "训练数据版本"
                or data_id == "TRAIN-0729-SFT32-V1"
            ),
        )

    attempt_rows = sheets[3]["rows"]
    attempt_ids = [row["attempt_id"] for row in attempt_rows]
    if len(attempt_ids) != len(set(attempt_ids)):
        raise ValueError("attempt_id must be unique")
    job_ids = [row["Job ID"] for row in attempt_rows]
    if len(job_ids) != len(set(job_ids)):
        raise ValueError("attempt ledger Job ID must be unique")
    known_attempts = set(attempt_ids)
    known_experiments = set(experiment_ids)
    retry_successors: dict[str, list[str]] = {}
    attempt_edges: dict[str, list[str]] = {attempt_id: [] for attempt_id in attempt_ids}
    for row in attempt_rows:
        attempt_id = row["attempt_id"]
        validate_status_axes(row, context=f"attempt {attempt_id}")
        if row["实验ID"] not in known_experiments:
            raise ValueError(
                f"attempt {attempt_id} references unknown experiment {row['实验ID']}"
            )
        retry_of = row["retry_of_attempt_id"]
        if retry_of != "不适用":
            if retry_of not in known_attempts:
                raise ValueError(
                    f"attempt {attempt_id} retries unknown attempt {retry_of}"
                )
            if retry_of == attempt_id:
                raise ValueError(f"attempt {attempt_id} cannot retry itself")
            retry_target = next(
                candidate
                for candidate in attempt_rows
                if candidate["attempt_id"] == retry_of
            )
            if (
                retry_target["实验ID"] != row["实验ID"]
                or retry_target["stage"] != row["stage"]
            ):
                raise ValueError(
                    f"attempt {attempt_id} retry target must share experiment and stage"
                )
            retry_successors.setdefault(retry_of, []).append(attempt_id)
            attempt_edges[attempt_id].append(retry_of)
        dependencies = parse_id_array(
            row["depends_on_attempt_ids"],
            context=f"attempt {attempt_id} depends_on_attempt_ids",
        )
        unknown_dependencies = set(dependencies) - known_attempts
        if unknown_dependencies:
            raise ValueError(
                f"attempt {attempt_id} has unknown dependencies "
                f"{sorted(unknown_dependencies)}"
            )
        if attempt_id in dependencies:
            raise ValueError(f"attempt {attempt_id} cannot depend on itself")
        if retry_of in dependencies:
            raise ValueError(
                f"attempt {attempt_id} must not duplicate retry lineage as a dependency"
            )
        attempt_edges[attempt_id].extend(dependencies)
        failure_class = row["failure_class"]
        if failure_class not in FAILURE_CLASSES:
            raise ValueError(
                f"attempt {attempt_id} has invalid failure_class {failure_class!r}"
            )
        if failure_class in FAILURE_EXECUTION_CLASSES and not (
            row["execution_status"] == "failed"
            and row["scientific_outcome"] == "not_evaluated"
        ):
            raise ValueError(
                f"attempt {attempt_id} {failure_class} must be failed/not_evaluated"
            )
        if row["execution_status"] == "failed" and failure_class not in FAILURE_EXECUTION_CLASSES:
            raise ValueError(
                f"failed attempt {attempt_id} must have a concrete failure_class"
            )
        if row["execution_status"] == "completed" and failure_class != "not_applicable":
            raise ValueError(
                f"completed attempt {attempt_id} must use failure_class=not_applicable"
            )
        validate_path_columns(
            row,
            context=f"attempt {attempt_id}",
            archive_date=archive_date_for_experiment(row["实验ID"]),
            exact_archive_mapping=row["实验ID"] in ARCHIVED_EXPERIMENT_DATES,
        )
    validate_attempt_graph(attempt_edges)
    for row in attempt_rows:
        successors = retry_successors.get(row["attempt_id"], [])
        if row["lineage_status"] == "superseded" and len(successors) != 1:
            raise ValueError(
                f"superseded attempt {row['attempt_id']} must have exactly one "
                f"registered retry successor; observed={successors}"
            )
        if row["lineage_status"] == "canonical" and successors:
            raise ValueError(
                f"canonical attempt {row['attempt_id']} has retry successor {successors}"
            )
    contracts = {
        (row["功能/阶段"], row["字段/合同"])
        for row in sheets[1]["rows"]
        if row["类别"] == "JSON合同"
    }
    required_contracts = {
        (stage, direction)
        for stage in ("select", "audit", "candidate_evidence", "decision")
        for direction in ("标准输入", "标准输出")
    }
    if not required_contracts.issubset(contracts):
        raise ValueError(
            f"stage contracts missing: {sorted(required_contracts - contracts)}"
        )
    return payload


def validate_status_axes(row: dict[str, Any], *, context: str) -> None:
    execution_status = row["execution_status"]
    scientific_outcome = row["scientific_outcome"]
    lineage_status = row["lineage_status"]
    if execution_status not in EXECUTION_STATUSES:
        raise ValueError(f"{context} has invalid execution_status {execution_status!r}")
    if scientific_outcome not in SCIENTIFIC_OUTCOMES:
        raise ValueError(
            f"{context} has invalid scientific_outcome {scientific_outcome!r}"
        )
    if lineage_status not in LINEAGE_STATUSES:
        raise ValueError(f"{context} has invalid lineage_status {lineage_status!r}")
    if scientific_outcome in {"baseline_reference", "pass", "negative", "inconclusive"}:
        if execution_status != "completed":
            raise ValueError(
                f"{context} cannot have scientific outcome {scientific_outcome!r} "
                f"with execution_status {execution_status!r}"
            )


def parse_id_array(value: Any, *, context: str) -> list[str]:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a JSON array string")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"{context} is not valid JSON: {error}") from error
    if not isinstance(parsed, list) or any(
        not isinstance(item, str) or not item for item in parsed
    ):
        raise ValueError(f"{context} must contain only non-empty string IDs")
    if len(parsed) != len(set(parsed)):
        raise ValueError(f"{context} contains duplicate IDs")
    return parsed


def validate_attempt_graph(edges: dict[str, list[str]]) -> None:
    """Reject retry/dependency cycles, including mixed-edge cycles."""

    visited: set[str] = set()
    active: set[str] = set()

    def visit(attempt_id: str) -> None:
        if attempt_id in active:
            raise ValueError(f"attempt lineage/dependency cycle includes {attempt_id}")
        if attempt_id in visited:
            return
        active.add(attempt_id)
        for predecessor in edges[attempt_id]:
            visit(predecessor)
        active.remove(attempt_id)
        visited.add(attempt_id)

    for attempt_id in edges:
        visit(attempt_id)


def parse_output_paths(value: Any, *, context: str) -> list[str]:
    if isinstance(value, str) and value in MISSING_MARKERS:
        return []
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a path or JSON array string")
    if value.startswith("["):
        paths = parse_id_array(value, context=context)
    else:
        paths = [value]
    for path_text in paths:
        path = PurePosixPath(path_text)
        if (
            path.is_absolute()
            or not path.parts
            or path.parts[0] != "outputs"
            or ".." in path.parts
            or "\\" in path_text
            or path.as_posix() != path_text
        ):
            raise ValueError(
                f"{context} must contain safe repo-relative outputs/... paths: "
                f"{path_text!r}"
            )
    return paths


def archive_date_for_experiment(experiment_id: str) -> str | None:
    """Return an explicit archive namespace for maintained historical runs."""

    if experiment_id.startswith("0801"):
        return "0801"
    return ARCHIVED_EXPERIMENT_DATES.get(experiment_id)


def archive_output_path(historical_path: str, *, archive_date: str) -> str:
    """Translate one validated historical output path to its archive location."""

    if archive_date not in SUPPORTED_ARCHIVE_DATES:
        raise ValueError(f"unsupported output archive date: {archive_date!r}")
    path = PurePosixPath(historical_path)
    expected_prefix = f"CA_{archive_date}("
    if (
        path.is_absolute()
        or len(path.parts) < 2
        or path.parts[0] != "outputs"
        or not path.parts[1].startswith(expected_prefix)
        or ".." in path.parts
        or "\\" in historical_path
        or path.as_posix() != historical_path
    ):
        raise ValueError(
            f"historical {archive_date} path must start with "
            f"outputs/{expected_prefix}...: {historical_path!r}"
        )
    return PurePosixPath(
        "outputs",
        f"CA_{archive_date}_archive",
        *path.parts[1:],
    ).as_posix()


def resolve_archived_binding_path(
    historical_path: Path,
    *,
    archive_date: str,
) -> Path:
    """Resolve a frozen absolute binding through one declared archive mapping.

    Runtime receipts remain immutable and therefore retain their original absolute
    paths.  This function never probes that stale location and never falls back: it
    validates the historical repository root and date namespace, then resolves the
    corresponding path below this checkout's declared archive.
    """

    if not historical_path.is_absolute():
        raise ValueError(
            f"historical binding must be absolute: {historical_path}"
        )
    try:
        relative = historical_path.relative_to(HISTORICAL_REPOSITORY_ROOT)
    except ValueError as error:
        raise ValueError(
            "historical binding is outside the frozen repository root: "
            f"{historical_path}"
        ) from error
    current_relative = archive_output_path(
        PurePosixPath(*relative.parts).as_posix(),
        archive_date=archive_date,
    )
    return ROOT / current_relative


def validate_path_columns(
    row: dict[str, Any],
    *,
    context: str,
    archive_date: str | None,
    exact_archive_mapping: bool = False,
    execution_status: str | None = None,
    allow_missing_current: bool = False,
) -> None:
    historical_paths = parse_output_paths(
        row["历史运行路径"], context=f"{context} historical path"
    )
    current_paths = parse_output_paths(
        row["当前证据路径"], context=f"{context} current evidence path"
    )
    status = execution_status or row["execution_status"]
    if status != "planned" and not current_paths and not allow_missing_current:
        raise ValueError(f"{context} must record a current evidence path")
    if archive_date is not None:
        if not historical_paths or not current_paths:
            raise ValueError(f"{context} must record historical and archive paths")
        expected_run_prefix = f"CA_{archive_date}("
        if any(
            len(PurePosixPath(path).parts) < 2
            or not PurePosixPath(path).parts[1].startswith(expected_run_prefix)
            for path in historical_paths
        ):
            raise ValueError(
                f"{context} historical paths must be outputs/{expected_run_prefix}..."
            )
        expected_archive_root = ("outputs", f"CA_{archive_date}_archive")
        if any(
            PurePosixPath(path).parts[:2] != expected_archive_root
            for path in current_paths
        ):
            raise ValueError(
                f"{context} current evidence must be under "
                f"outputs/CA_{archive_date}_archive"
            )
        if exact_archive_mapping:
            expected_current_paths = [
                archive_output_path(path, archive_date=archive_date)
                for path in historical_paths
            ]
            if current_paths != expected_current_paths:
                raise ValueError(
                    f"{context} current evidence must exactly match its declared "
                    f"{archive_date} archive mapping: "
                    f"{current_paths} != {expected_current_paths}"
                )
    if status != "planned":
        for path in current_paths:
            resolved = ROOT / path
            if not resolved.exists():
                raise FileNotFoundError(
                    f"{context} current evidence path missing: {resolved}"
                )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_key_value_receipt(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"missing receipt: {path}")
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        if key in result:
            raise ValueError(f"duplicate receipt key {key!r}: {path}")
        result[key] = value
    return result


def validate_legacy_0729_evidence(registry: dict[str, Any]) -> None:
    # The legacy 0729 smoke output was moved out of this repository and is no
    # longer present at either historical output root.  Keep schema validation
    # strict, but do not make rebuilding the maintained workbook depend on an
    # irretrievable external artifact; the registry row records the gap.
    if not SMOKE_METRICS.is_file():
        return
    experiment_rows = registry["sheets"][0]["rows"]
    success = next(
        row
        for row in experiment_rows
        if row["运行记录ID"] == "CA_0729(2)-qwen3vl-sft32-smoke"
    )
    metrics = json.loads(SMOKE_METRICS.read_text(encoding="utf-8"))
    expected_metrics = {
        "rows": metrics["rows"],
        "strict_json_rate": metrics["strict_json_rate"],
        "schema_valid_rate": metrics["schema_valid_rate"],
        "unknown_region_id_rate": metrics["unknown_region_id_rate"],
        "region_id_accuracy": metrics["region_id_accuracy"],
        "exact_match_rate": metrics["exact_match_rate"],
        "bleu4": metrics["predict_metrics"]["predict_bleu-4"],
        "rouge1": metrics["predict_metrics"]["predict_rouge-1"],
        "rouge2": metrics["predict_metrics"]["predict_rouge-2"],
        "rougeL": metrics["predict_metrics"]["predict_rouge-l"],
        "protocol_gate": "passed" if metrics["protocol_gate"]["passed"] else "failed",
    }
    registered_metrics = json.loads(success["训练后附加指标JSON"])
    if set(registered_metrics) != set(expected_metrics):
        raise ValueError("registered 0729 post-SFT metric keys differ from evidence")
    for key, expected in expected_metrics.items():
        observed = registered_metrics[key]
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            matches = isinstance(observed, (int, float)) and math.isclose(
                float(observed), float(expected), rel_tol=0.0, abs_tol=1e-12
            )
        else:
            matches = observed == expected
        if not matches:
            raise ValueError(
                f"registered 0729 metric {key} differs: {observed!r} != {expected!r}"
            )
    if not math.isclose(
        float(success["训练后主指标"]), float(metrics["region_id_accuracy"])
    ):
        raise ValueError("registered 0729 primary metric differs from evidence")
    for field in ("训练前功能状态", "训练前主指标", "训练前E2E状态", "训练后E2E状态"):
        if not str(success[field]).startswith(("未运行", "未测量")):
            raise ValueError(f"0729 incomplete result must explicitly mark {field}")

    completion_path = SMOKE_ROOT / "completion_summary.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("adapter_retained") is not False:
        raise ValueError("completion evidence no longer says adapter_retained=false")
    if "false" not in str(success["adapter保留"]).lower():
        raise ValueError("registry must record that the 0729 adapter was not retained")

    data_rows = registry["sheets"][2]["rows"]
    training_data = next(
        row for row in data_rows if row["数据ID"] == "TRAIN-0729-SFT32-V1"
    )
    manifest_path = SMOKE_ROOT / "data/manifest.json"
    validation_path = SMOKE_ROOT / "data/validation.json"
    manifest_hash = sha256_file(manifest_path)
    validation_hash = sha256_file(validation_path)
    evidence = str(training_data["权威证据"])
    if manifest_hash not in evidence or validation_hash not in evidence:
        raise ValueError("registered SFT32 manifest/validation SHA256 is stale")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if not (
        validation.get("valid") is True
        and validation.get("examples") == 32
        and validation.get("unique_parent_scenes") == 32
        and validation.get("class_distribution")
        == {"false_negative": 16, "false_positive_change": 16}
    ):
        raise ValueError("SFT32 validation evidence no longer matches registry contract")


def validate_fixed_evaluation_evidence(registry: dict[str, Any]) -> None:
    """Audit the fixed diagnostic set even when unrelated 0729 files are absent."""

    data_rows = registry["sheets"][2]["rows"]
    summaries = {
        row["数据ID"]: row
        for row in data_rows
        if row["记录类型"] == "数据集概要"
    }
    for data_id, expected_order in FIXED_EVALUATION_ORDERS.items():
        if data_id not in summaries:
            raise ValueError(f"fixed evaluation summary missing: {data_id}")
        tokens = [
            item.strip().split(maxsplit=1)
            for item in str(summaries[data_id]["sample_id/顺序"]).split(";")
        ]
        try:
            observed_order = [
                sample for position, sample in sorted((int(position), sample) for position, sample in tokens)
            ]
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid ordered sample summary for {data_id}") from error
        if observed_order != expected_order:
            raise ValueError(
                f"{data_id} order changed: {observed_order} != {expected_order}"
            )

    member_rows = [row for row in data_rows if row["记录类型"] == "评测样本成员"]
    if len(member_rows) != 5:
        raise ValueError("data registry must contain exactly five evaluation members")
    member_orders = {data_id: {} for data_id in FIXED_EVALUATION_ORDERS}
    registered_samples: dict[str, dict[str, int]] = {}
    for row in member_rows:
        try:
            member = json.loads(row["sample_id/顺序"])
        except json.JSONDecodeError as error:
            raise ValueError("evaluation member order must be JSON") from error
        if not isinstance(member, dict) or not isinstance(member.get("sample_id"), str):
            raise ValueError("evaluation member order must contain string sample_id")
        sample = member.pop("sample_id")
        if sample in registered_samples:
            raise ValueError(f"duplicate evaluation member {sample}")
        memberships = {item.strip() for item in row["数据ID"].split(";")}
        if set(member) != memberships:
            raise ValueError(
                f"evaluation member {sample} membership/order mismatch: "
                f"{set(member)} != {memberships}"
            )
        if any(not isinstance(position, int) or position < 1 for position in member.values()):
            raise ValueError(f"evaluation member {sample} has invalid set position")
        registered_samples[sample] = member
        for data_id, position in member.items():
            if position in member_orders[data_id]:
                raise ValueError(f"duplicate {data_id} position {position}")
            member_orders[data_id][position] = sample
        for directory, field in (
            ("A", "A_SHA256"),
            ("B", "B_SHA256"),
            ("label_cvt", "label_SHA256"),
        ):
            expected_hash = str(row[field])
            if not SHA256_RE.fullmatch(expected_hash):
                raise ValueError(f"invalid registered SHA256 for {sample} {field}")
            path = TEST_ROOT / directory / f"{sample}.png"
            if not path.is_file():
                raise FileNotFoundError(f"missing fixed evaluation file: {path}")
            actual_hash = sha256_file(path)
            if actual_hash != expected_hash:
                raise ValueError(
                    f"fixed evaluation file changed: {path}; {actual_hash} != {expected_hash}"
                )
    for data_id, expected_order in FIXED_EVALUATION_ORDERS.items():
        observed_order = [
            sample for _, sample in sorted(member_orders[data_id].items())
        ]
        if observed_order != expected_order:
            raise ValueError(
                f"{data_id} member order changed: {observed_order} != {expected_order}"
            )


def validate_0801_archive_evidence(registry: dict[str, Any]) -> None:
    """Bind selected 0801 attempt-ledger backfills to read-only archive evidence."""

    attempts = {row["attempt_id"]: row for row in registry["sheets"][3]["rows"]}
    required_attempts = {
        "BASE-E2E3-51727",
        "BASE-E2E5-51731",
        "V1-CURATION-51847",
        "V1-DATA-51876",
        "V1-DATA-51883",
        "V1-SFT-51891",
        "V1-SFT-51896",
        "V1-E2E3-51903",
        "V3-DATA-52129",
        "V3-DATA-52132",
        "V5-DATA-52307",
        "V5-MANIFEST-52311",
        "V5-MANIFEST-52315",
        "V5-MANIFEST-52319",
    }
    missing_attempts = required_attempts - set(attempts)
    if missing_attempts:
        raise ValueError(f"required 0801 attempts missing: {sorted(missing_attempts)}")
    for attempt_id in required_attempts:
        row = attempts[attempt_id]
        evidence_path = Path(row["当前证据路径"])
        if evidence_path.is_absolute() or evidence_path.parts[:2] != (
            "outputs",
            "CA_0801_archive",
        ):
            raise ValueError(
                f"attempt {attempt_id} current evidence path must be archive-relative"
            )
        evidence_root = ROOT / evidence_path
        if not evidence_root.is_dir():
            raise FileNotFoundError(
                f"attempt {attempt_id} current evidence path missing: {evidence_root}"
            )

    base_e2e3_root = ROOT / attempts["BASE-E2E3-51727"]["当前证据路径"]
    base_e2e5_root = ROOT / attempts["BASE-E2E5-51731"]["当前证据路径"]
    v1_data_root = ROOT / attempts["V1-DATA-51883"]["当前证据路径"]
    v1_sft_root = ROOT / attempts["V1-SFT-51896"]["当前证据路径"]
    v1_e2e3_root = ROOT / attempts["V1-E2E3-51903"]["当前证据路径"]

    base_e2e3 = json.loads(
        (base_e2e3_root / "baseline_summary.json").read_text(encoding="utf-8")
    )
    base_e2e5 = json.loads(
        (base_e2e5_root / "baseline_summary.json").read_text(encoding="utf-8")
    )
    v1_summary = json.loads(
        (v1_e2e3_root / "post_sft_summary.json").read_text(encoding="utf-8")
    )
    experiment_rows = {row["实验ID"]: row for row in registry["sheets"][0]["rows"]}
    base_row = experiment_rows["0801(BASE)"]
    v1_row = experiment_rows["0801(V1)"]
    if not (
        base_row["execution_status"] == "completed"
        and base_row["scientific_outcome"] == "baseline_reference"
        and base_row["lineage_status"] == "canonical"
        and v1_row["execution_status"] == "completed"
        and v1_row["scientific_outcome"] == "negative"
        and v1_row["lineage_status"] == "canonical"
    ):
        raise ValueError("base/v1 experiment status axes changed")
    expected_summaries = (
        (
            base_e2e3,
            "51727",
            "DEV-CORE-3-V1",
            3,
            0.7422312,
            0.85204673,
            FIXED_EVALUATION_ORDERS["DEV-CORE-3-V1"],
        ),
        (
            base_e2e5,
            "51731",
            "DEV-EXTENDED-5-V1",
            5,
            0.6762904,
            0.80688931,
            FIXED_EVALUATION_ORDERS["DEV-EXTENDED-5-V1"],
        ),
        (
            v1_summary,
            "51903",
            "DEV-CORE-3-V1",
            3,
            0.3687664,
            0.5388303,
            FIXED_EVALUATION_ORDERS["DEV-CORE-3-V1"],
        ),
    )
    for (
        summary,
        job_id,
        evaluation_id,
        count,
        selected_iou,
        selected_f1,
        sample_order,
    ) in expected_summaries:
        selected = summary.get("aggregate", {}).get("selected", {})
        if not (
            summary.get("status") == "completed"
            and summary.get("job_id") == job_id
            and summary.get("evaluation_id") == evaluation_id
            and summary.get("requested_samples") == count
            and summary.get("scored_samples") == count
            and summary.get("sample_order") == sample_order
            and math.isclose(float(selected.get("iou")), selected_iou, abs_tol=1e-12)
            and math.isclose(float(selected.get("f1")), selected_f1, abs_tol=1e-12)
        ):
            raise ValueError(f"0801 summary evidence mismatch for Job {job_id}")

    if not (
        attempts["BASE-E2E3-51727"]["Job ID"] == base_e2e3["job_id"]
        and attempts["BASE-E2E5-51731"]["Job ID"] == base_e2e5["job_id"]
        and attempts["V1-E2E3-51903"]["Job ID"] == v1_summary["job_id"]
        and math.isclose(float(base_row["后IoU"]), 0.7422312, abs_tol=1e-12)
        and math.isclose(float(base_row["后F1"]), 0.85204673, abs_tol=1e-12)
        and math.isclose(float(v1_row["后IoU"]), 0.3687664, abs_tol=1e-12)
        and math.isclose(float(v1_row["后F1"]), 0.5388303, abs_tol=1e-12)
        and math.isclose(float(v1_row["ΔIoU"]), -0.3734648, abs_tol=1e-12)
        and math.isclose(float(v1_row["ΔF1"]), -0.31321643, abs_tol=1e-12)
    ):
        raise ValueError("base/v1 registered metrics or attempt Job IDs changed")

    v1_test_50 = v1_summary.get("per_sample", {}).get("test_50_8", {})
    initial_counts = v1_test_50.get("initial", {})
    selected_counts = v1_test_50.get("selected", {})
    if not (
        initial_counts.get("fp") == 0
        and selected_counts.get("fp") == 6518
        and selected_counts.get("fn") == 166
    ):
        raise ValueError("v1 test_50_8 +6518 FP negative evidence changed")

    manifest_hash = sha256_file(v1_data_root / "data/manifest.json")
    v1_manifest = json.loads(
        (v1_data_root / "data/manifest.json").read_text(encoding="utf-8")
    )
    training_summary = json.loads(
        (v1_sft_root / "training_summary.json").read_text(encoding="utf-8")
    )
    if not (
        manifest_hash
        == training_summary.get("dataset_manifest_sha256")
        == "<hash omitted>"
        and v1_manifest.get("runtime_commit")
        == attempts["V1-DATA-51876"]["代码commit"]
        == "41a2bd193ebb431c049cf91a9145b830a7cafe89"
        and training_summary.get("status") == "success"
        and training_summary.get("adapter_sha256")
        == "<hash omitted>"
        and training_summary.get("merged_model_sha256")
        == "<hash omitted>"
    ):
        raise ValueError("v1 data/model hash chain differs from registered evidence")

    receipt_contracts = (
        (
            "BASE-E2E3-51727",
            "submission.txt",
            "logs/status-51727.txt",
            "job_id",
            "expected_commit",
            "success",
        ),
        (
            "BASE-E2E5-51731",
            "submission.txt",
            "logs/status-51731.txt",
            "job_id",
            "expected_commit",
            "success",
        ),
        (
            "V1-CURATION-51847",
            "submission.txt",
            "logs/status-51847.txt",
            "job_id",
            "code_commit",
            "success",
        ),
        (
            "V1-DATA-51876",
            "data-build-submission.txt",
            "logs/data-build-status-51876.txt",
            "data_build_job",
            "code_commit",
            "failed",
        ),
        (
            "V1-DATA-51883",
            "data-build-submission-51883.txt",
            "logs/data-build-status-51883.txt",
            "data_build_job",
            "code_commit",
            "success",
        ),
        (
            "V1-SFT-51891",
            "submission.txt",
            "logs/status-51891.txt",
            "job_id",
            "code_commit",
            "failed",
        ),
        (
            "V1-SFT-51896",
            "submission.txt",
            "logs/status-51896.txt",
            "job_id",
            "code_commit",
            "success",
        ),
        (
            "V1-E2E3-51903",
            "submission.txt",
            "logs/status-51903.txt",
            "job_id",
            "commit",
            "success",
        ),
        (
            "V3-DATA-52129",
            "data-build-submission-52129.txt",
            "logs/data-build-status-52129.txt",
            "data_build_job",
            "code_commit",
            "failed",
        ),
        (
            "V3-DATA-52132",
            "data-build-submission-52132.txt",
            "logs/data-build-status-52132.txt",
            "data_build_job",
            "code_commit",
            "success",
        ),
        (
            "V5-DATA-52307",
            "data-build-submission-52307.txt",
            "logs/data-build-status-52307.txt",
            "data_build_job",
            "code_commit",
            "success",
        ),
        (
            "V5-MANIFEST-52311",
            "data-build-submission-52311.txt",
            "logs/data-build-status-52311.txt",
            "data_build_job",
            "code_commit",
            "failed",
        ),
        (
            "V5-MANIFEST-52315",
            "data-build-submission-52315.txt",
            "logs/data-build-status-52315.txt",
            "data_build_job",
            "code_commit",
            "success",
        ),
        (
            "V5-MANIFEST-52319",
            "data-build-submission-52319.txt",
            "logs/data-build-status-52319.txt",
            "data_build_job",
            "code_commit",
            "success",
        ),
    )
    receipt_attempt_ids = {contract[0] for contract in receipt_contracts}
    if receipt_attempt_ids != required_attempts:
        raise ValueError(
            "receipt contracts must cover every required 0801 attempt exactly: "
            f"{sorted(receipt_attempt_ids)} != {sorted(required_attempts)}"
        )
    for (
        attempt_id,
        submission_name,
        status_name,
        job_key,
        commit_key,
        expected_status,
    ) in receipt_contracts:
        row = attempts[attempt_id]
        evidence_root = ROOT / row["当前证据路径"]
        submission = read_key_value_receipt(evidence_root / submission_name)
        status = read_key_value_receipt(evidence_root / status_name)
        expected_exit = "0" if expected_status == "success" else "1"
        if not (
            submission.get(job_key) == row["Job ID"]
            and submission.get(commit_key) == row["代码commit"]
            and status.get("status") == expected_status
            and status.get("exit_code") == expected_exit
            and status.get("job_id", row["Job ID"]) == row["Job ID"]
        ):
            raise ValueError(f"Job/commit/status receipt mismatch for {attempt_id}")

    # Bind the failure taxonomy to immutable traceback content.  A generic
    # status=failed receipt is insufficient to distinguish infra, contract,
    # and test failures during independent review.
    failure_diagnostics = {
        "V1-DATA-51876": (
            "logs/data-build-51876.err",
            "<hash omitted>",
            "SyntaxError: unexpected character after line continuation character",
        ),
        "V1-SFT-51891": (
            "logs/train-51891.log",
            "<hash omitted>",
            "FileNotFoundError: [Errno 2] No such file or directory: 'torchrun'",
        ),
        "V3-DATA-52129": (
            "logs/data-build-52129.err",
            "<hash omitted>",
            "ValueError: v3 scope distribution mismatch",
        ),
        "V5-MANIFEST-52311": (
            "logs/data-build-52311.err",
            "<hash omitted>",
            "AssertionError: None != 't2'",
        ),
    }
    for attempt_id, (relative_path, expected_hash, signature) in failure_diagnostics.items():
        diagnostic = ROOT / attempts[attempt_id]["当前证据路径"] / relative_path
        if (
            sha256_file(diagnostic) != expected_hash
            or signature not in diagnostic.read_text(encoding="utf-8")
        ):
            raise ValueError(f"failure-class diagnostic changed for {attempt_id}")

    for attempt_id, expected_hash, expected_runtime, expected_round, train_rows, val_rows in (
        (
            "V3-DATA-52132",
            "<hash omitted>",
            "b40c651c2279dac2010d1bda0090520f02674563",
            "v3-action",
            53,
            24,
        ),
        (
            "V5-DATA-52307",
            "<hash omitted>",
            "1589dd7f90aa5787ef1ed62bacf0edbcc54283e0",
            "v5-contrast-repair",
            69,
            24,
        ),
    ):
        data_root = ROOT / attempts[attempt_id]["当前证据路径"] / "data"
        manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
        validation = json.loads((data_root / "validation.json").read_text(encoding="utf-8"))
        if not (
            sha256_file(data_root / "manifest.json") == expected_hash
            and manifest.get("runtime_commit") == expected_runtime
            and validation.get("valid") is True
            and validation.get("data_round") == expected_round
            and validation.get("train_rows") == train_rows
            and validation.get("val_rows") == val_rows
            and validation.get("test_data_used") is False
            and (data_root / ".data_complete").read_text(encoding="utf-8").strip()
            == "complete"
        ):
            raise ValueError(f"canonical data artifact changed for {attempt_id}")

    expected_attempt_axes = {
        "BASE-E2E3-51727": ("completed", "baseline_reference", "canonical", "not_applicable"),
        "BASE-E2E5-51731": ("completed", "baseline_reference", "canonical", "not_applicable"),
        "V1-CURATION-51847": ("completed", "not_evaluated", "canonical", "not_applicable"),
        "V1-DATA-51876": ("failed", "not_evaluated", "superseded", "infra_failed"),
        "V1-DATA-51883": ("completed", "not_evaluated", "canonical", "not_applicable"),
        "V1-SFT-51891": ("failed", "not_evaluated", "superseded", "infra_failed"),
        "V1-SFT-51896": ("completed", "not_evaluated", "canonical", "not_applicable"),
        "V1-E2E3-51903": ("completed", "negative", "canonical", "not_applicable"),
        "V3-DATA-52129": ("failed", "not_evaluated", "superseded", "contract_failed"),
        "V3-DATA-52132": ("completed", "not_evaluated", "canonical", "not_applicable"),
        "V5-DATA-52307": ("completed", "not_evaluated", "canonical", "not_applicable"),
        "V5-MANIFEST-52311": ("failed", "not_evaluated", "superseded", "test_failed"),
        "V5-MANIFEST-52315": ("completed", "not_evaluated", "superseded", "not_applicable"),
        "V5-MANIFEST-52319": ("completed", "not_evaluated", "canonical", "not_applicable"),
    }
    if set(expected_attempt_axes) != required_attempts:
        raise ValueError("attempt-axis contracts must cover all required 0801 attempts")
    for attempt_id, expected_axes in expected_attempt_axes.items():
        row = attempts[attempt_id]
        observed_axes = (
            row["execution_status"],
            row["scientific_outcome"],
            row["lineage_status"],
            row["failure_class"],
        )
        if observed_axes != expected_axes:
            raise ValueError(
                f"attempt axes changed for {attempt_id}: "
                f"{observed_axes} != {expected_axes}"
            )

    failed_receipts = (
        ROOT
        / attempts["V1-DATA-51876"]["当前证据路径"]
        / "logs/data-build-status-51876.txt",
        ROOT
        / attempts["V1-SFT-51891"]["当前证据路径"]
        / "logs/status-51891.txt",
    )
    for receipt in failed_receipts:
        status = read_key_value_receipt(receipt)
        if status.get("status") != "failed" or status.get("exit_code") != "1":
            raise ValueError(f"infra-failed receipt changed: {receipt}")

    if attempts["V1-SFT-51896"]["代码commit"] != (
        "3020d8df83b15e6062508784cea63f32638683d5"
    ):
        raise ValueError("canonical v1 SFT commit changed")
    if (v1_sft_root / ".training_complete").read_text(encoding="utf-8").strip() != "complete":
        raise ValueError("canonical v1 SFT completion marker changed")


def validate_v2_reconstructed_config(registry: dict[str, Any]) -> None:
    """Keep the v2 audit reconstruction distinct from a runtime-emitted receipt."""

    payload = json.loads(V2_RECONSTRUCTED_CONFIG.read_text(encoding="utf-8"))
    warning = payload.get("warning", {})
    if not (
        payload.get("schema_id")
        == "qwen3vl_reconstructed_resolved_training_config_evidence_map_v1"
        and payload.get("record_type") == "reconstructed_resolved_config"
        and payload.get("record_status")
        == "reconstructed_read_only_evidence_not_original_runtime_receipt"
        and warning.get("original_config_used_is_not_resolved") is True
        and "Audit and interpretation" in warning.get("allowed_use", "")
        and "Do not present" in warning.get("forbidden_use", "")
        and payload.get("archive_mutation") is False
    ):
        raise ValueError("v2 reconstructed-config provenance contract changed")
    v2_row = next(
        row
        for row in registry["sheets"][0]["rows"]
        if row["实验ID"] == "0801(1)"
    )
    registered = " ".join(
        str(v2_row[field])
        for field in ("权威证据", "结论/限制", "登记完整性")
    )
    if not (
        V2_RECONSTRUCTED_CONFIG.relative_to(ROOT).as_posix() in registered
        and "reconstructed" in registered
        and "不是原始 runtime-emitted resolved config" in registered
        and "不能作为重训输入" in registered
    ):
        raise ValueError("v2 row does not preserve reconstructed-config limitations")


def validate_0802_p1_evidence(registry: dict[str, Any]) -> None:
    """Bind the append-only P1 retry lineage to the successful retry3 pair."""

    experiment_rows = {
        row["实验ID"]: row for row in registry["sheets"][0]["rows"]
    }
    expected_experiment_axes = {
        "0802(P1-H0)": ("completed", "baseline_reference", "canonical"),
        "0802(P1-H1)": ("completed", "pass", "canonical"),
    }
    for experiment_id, expected_axes in expected_experiment_axes.items():
        if experiment_id not in experiment_rows:
            raise ValueError(f"P1 experiment row missing: {experiment_id}")
        row = experiment_rows[experiment_id]
        observed_axes = (
            row["execution_status"],
            row["scientific_outcome"],
            row["lineage_status"],
        )
        if observed_axes != expected_axes:
            raise ValueError(
                f"{experiment_id} axes changed: {observed_axes} != {expected_axes}"
            )
        if not (
            math.isclose(float(row["后IoU"]), 0.7422312, abs_tol=1e-12)
            and math.isclose(float(row["后F1"]), 0.85204673, abs_tol=1e-12)
            and math.isclose(float(row["ΔIoU"]), 0.0, abs_tol=1e-12)
            and math.isclose(float(row["ΔF1"]), 0.0, abs_tol=1e-12)
            and P1_RETRY3_RESULT.relative_to(ROOT).as_posix()
            in str(row["权威证据"])
        ):
            raise ValueError(f"{experiment_id} registered P1 metrics/evidence changed")
        historical_paths = parse_output_paths(
            row["历史运行路径"],
            context=f"{experiment_id} historical path",
        )
        current_paths = parse_output_paths(
            row["当前证据路径"],
            context=f"{experiment_id} current evidence path",
        )
        expected_current_paths = [
            archive_output_path(path, archive_date="0802")
            for path in historical_paths
        ]
        if current_paths != expected_current_paths or any(
            path not in str(row["权威证据"]) for path in current_paths
        ):
            raise ValueError(
                f"{experiment_id} must explicitly bind authoritative evidence "
                "to the 0802 archive"
            )
    h1_limits = " ".join(
        str(experiment_rows["0802(P1-H1)"][field])
        for field in ("状态说明", "结论/限制", "登记完整性")
    )
    if not (
        "协议" in h1_limits
        and "安全" in h1_limits
        and "不能声称分割提升" in h1_limits
        and "不能声称 LEVIR-CD 泛化" in h1_limits
    ):
        raise ValueError("P1 H1 pass limitations were weakened")

    data_rows = registry["sheets"][2]["rows"]
    smoke = next(
        (row for row in data_rows if row["数据ID"] == "DEV-SMOKE-1-H1-V1"),
        None,
    )
    if smoke is None or not (
        smoke["记录类型"] == "数据集概要"
        and smoke["sample_id/顺序"] == "1 test_1_1"
        and smoke["样本数"] == 1
        and smoke["GT正像素"] == 0
        and smoke["A_SHA256"]
        == "<hash omitted>"
        and smoke["B_SHA256"]
        == "<hash omitted>"
        and smoke["label_SHA256"]
        == "<hash omitted>"
    ):
        raise ValueError("DEV-SMOKE-1-H1-V1 registry contract changed")

    attempts = {row["attempt_id"]: row for row in registry["sheets"][3]["rows"]}
    roles = (
        ("P1-CPU", "0802(P1-H1)", "p1_cpu_fixtures", 2, "h0-h1-cpu-fixtures"),
        ("P1-SMOKE-H0", "0802(P1-H0)", "p1_smoke_h0", 3, "h0-base-smoke1"),
        ("P1-SMOKE-H1", "0802(P1-H1)", "p1_smoke_h1", 4, "h1-base-smoke1"),
        ("P1-CORE-H0", "0802(P1-H0)", "p1_core_h0", 5, "h0-base-core3"),
        ("P1-CORE-H1", "0802(P1-H1)", "p1_core_h1", 6, "h1-base-core3"),
        ("P1-GUARD", "0802(P1-H1)", "p1_dependency_guard", 2, "h0-h1-cpu-fixtures"),
    )
    chains = (
        {
            "suffix": "",
            "jobs": ("53064", "53065", "53066", "53067", "53068", "53069"),
            "commit": "1dfbdbeda9f8b1ace5e3074a4dee9d52b8f86efe",
            "implementation": "f97f9f090369bc5d97f2a60b3d247aae6166b16e",
            "config": "<hash omitted>",
            "amendment": "<hash omitted>",
            "statuses": ("completed", "failed", "cancelled", "cancelled", "cancelled", "failed"),
        },
        {
            "suffix": "-retry1",
            "jobs": ("53109", "53110", "53111", "53112", "53113", "53114"),
            "commit": "a59382656ad6cdd4ee62b76a655f84c2279f023c",
            "implementation": "15cdca7ddc4611ad773576dd582e5b4513188d8e",
            "config": "<hash omitted>",
            "amendment": "<hash omitted>",
            "statuses": ("completed", "completed", "failed", "cancelled", "cancelled", "completed"),
        },
        {
            "suffix": "-retry2",
            "jobs": ("53141", "53142", "53143", "53144", "53145", "53146"),
            "commit": "64871db9f03d19d6ddb9ab3edf502c3ecb698ee5",
            "implementation": "e31fcb01d34cc28924e9c4189f319d7f44af4911",
            "config": "<hash omitted>",
            "amendment": "<hash omitted>",
            "statuses": ("completed", "completed", "completed", "completed", "failed", "completed"),
        },
        {
            "suffix": "-retry3",
            "jobs": ("53155", "53156", "53157", "53158", "53159", "53160"),
            "commit": "5210f89f5ca13ee1e636f5aa6a44631206a80123",
            "implementation": "a88404ac6a014f42692d1119d54de42c980fb310",
            "config": "<hash omitted>",
            "amendment": "<hash omitted>",
            "statuses": ("completed", "completed", "completed", "completed", "completed", "completed"),
        },
    )
    input_manifest_hash = (
        "<hash omitted>"
    )
    expected_p1_ids: set[str] = set()
    for chain_index, chain in enumerate(chains):
        current_ids = [
            f"{role[0]}-{job}" for role, job in zip(roles, chain["jobs"])
        ]
        expected_p1_ids.update(current_ids)
        for role_index, (role, job, execution_status) in enumerate(
            zip(roles, chain["jobs"], chain["statuses"])
        ):
            prefix, experiment_id, stage, allocation, slug = role
            attempt_id = f"{prefix}-{job}"
            if attempt_id not in attempts:
                raise ValueError(f"required P1 attempt missing: {attempt_id}")
            row = attempts[attempt_id]
            expected_retry = (
                "不适用"
                if chain_index == 0
                else f"{prefix}-{chains[chain_index - 1]['jobs'][role_index]}"
            )
            dependencies: list[str] = []
            if role_index in (1, 2, 3, 4):
                dependencies = [current_ids[role_index - 1]]
            elif role_index == 5:
                dependencies = current_ids[:5]
            expected_historical_path = (
                f"outputs/CA_0802({allocation})-qwen3vl-{slug}{chain['suffix']}"
            )
            expected_current_path = archive_output_path(
                expected_historical_path,
                archive_date="0802",
            )
            expected_scientific = "not_evaluated"
            if job in {"53144", "53158"}:
                expected_scientific = "baseline_reference"
            elif job == "53159":
                expected_scientific = "pass"
            expected_axes = (
                execution_status,
                expected_scientific,
                "canonical" if chain_index == len(chains) - 1 else "superseded",
                "contract_failed" if execution_status == "failed" else "not_applicable",
            )
            observed_axes = (
                row["execution_status"],
                row["scientific_outcome"],
                row["lineage_status"],
                row["failure_class"],
            )
            if not (
                row["实验ID"] == experiment_id
                and row["stage"] == stage
                and row["Job ID"] == job
                and row["retry_of_attempt_id"] == expected_retry
                and json.loads(row["depends_on_attempt_ids"]) == dependencies
                and row["代码commit"] == chain["commit"]
                and row["历史运行路径"] == expected_historical_path
                and row["当前证据路径"] == expected_current_path
                and expected_current_path in str(row["权威证据"])
                and observed_axes == expected_axes
            ):
                raise ValueError(f"P1 attempt registry contract changed: {attempt_id}")
            hashes = str(row["配置/模型/工具哈希"])
            for required_hash in (
                chain["implementation"],
                chain["config"],
                chain["amendment"],
                input_manifest_hash,
            ):
                if required_hash not in hashes:
                    raise ValueError(
                        f"P1 attempt {attempt_id} is missing frozen hash {required_hash}"
                    )

            evidence_root = ROOT / expected_current_path
            if prefix == "P1-GUARD":
                submission = read_key_value_receipt(
                    evidence_root / "dependency_guard_submission.txt"
                )
                status_path = (
                    evidence_root / "logs" / f"dependency-guard-status-{job}.txt"
                )
            else:
                submission = read_key_value_receipt(evidence_root / "submission.txt")
                status_path = evidence_root / "logs" / f"status-{job}.txt"
            if not (
                submission.get("job_id") == job
                and submission.get("expected_commit") == chain["commit"]
            ):
                raise ValueError(f"P1 submission identity changed: {attempt_id}")

            initial_dependency_cancel = (
                chain_index == 0 and role_index in (2, 3, 4)
            )
            if initial_dependency_cancel:
                if status_path.exists():
                    raise ValueError(
                        f"initial cancelled attempt unexpectedly gained status: {attempt_id}"
                    )
            else:
                status = read_key_value_receipt(status_path)
                expected_worker_status = {
                    "completed": "success",
                    "failed": "failed",
                    "cancelled": "cancelled",
                }[execution_status]
                if not (
                    status.get("job_id") == job
                    and status.get("commit") == chain["commit"]
                    and status.get("status") == expected_worker_status
                ):
                    raise ValueError(f"P1 worker status changed: {attempt_id}")

    observed_p1_ids = {
        attempt_id for attempt_id in attempts if attempt_id.startswith("P1-")
    }
    if observed_p1_ids != expected_p1_ids:
        raise ValueError(
            "P1 attempt membership changed: "
            f"{sorted(observed_p1_ids)} != {sorted(expected_p1_ids)}"
        )

    failure_diagnostics = {
        "P1-SMOKE-H0-53065": (
            "logs/slurm-53065.err",
            "<hash omitted>",
            "submission receipt experiment_arm_id drifted",
        ),
        "P1-GUARD-53069": (
            "logs/dependency-guard-53069.err",
            "<hash omitted>",
            "missing worker status was not eligible for dependency synthesis",
        ),
        "P1-SMOKE-H1-53111": (
            "logs/slurm-53111.err",
            "<hash omitted>",
            "trigger differs from raw round-one evidence",
        ),
        "P1-CORE-H1-53145": (
            "logs/slurm-53145.err",
            "<hash omitted>",
            "RecursionError: maximum recursion depth exceeded",
        ),
    }
    for attempt_id, (relative_path, expected_hash, signature) in failure_diagnostics.items():
        row = attempts[attempt_id]
        diagnostic = ROOT / row["当前证据路径"] / relative_path
        if not (
            diagnostic.is_file()
            and sha256_file(diagnostic) == expected_hash
            and signature in diagnostic.read_text(encoding="utf-8", errors="replace")
            and expected_hash in str(row["配置/模型/工具哈希"])
        ):
            raise ValueError(f"P1 failure diagnostic changed: {attempt_id}")

    retry3_specs = (
        (3, "h0-base-smoke1", "53156", "h0", "DEV-SMOKE-1-H1-V1", ["test_1_1"], 1),
        (4, "h1-base-smoke1", "53157", "h1", "DEV-SMOKE-1-H1-V1", ["test_1_1"], 1),
        (
            5,
            "h0-base-core3",
            "53158",
            "h0",
            "DEV-CORE-3-V1",
            FIXED_EVALUATION_ORDERS["DEV-CORE-3-V1"],
            3,
        ),
        (
            6,
            "h1-base-core3",
            "53159",
            "h1",
            "DEV-CORE-3-V1",
            FIXED_EVALUATION_ORDERS["DEV-CORE-3-V1"],
            3,
        ),
    )
    retry3_roots: dict[str, Path] = {}
    for allocation, slug, job, arm, evaluation_id, sample_order, count in retry3_specs:
        historical_root = (
            f"outputs/CA_0802({allocation})-qwen3vl-{slug}-retry3"
        )
        root = ROOT / archive_output_path(
            historical_root,
            archive_date="0802",
        )
        retry3_roots[job] = root
        summary = json.loads((root / "baseline_summary.json").read_text(encoding="utf-8"))
        receipt = json.loads((root / "evaluation_receipt.json").read_text(encoding="utf-8"))
        identity = receipt.get("identity", {})
        gates = receipt.get("gates", {})
        selected = summary.get("aggregate", {}).get("selected", {})
        expected_iou = 0.0 if count == 1 else 0.7422312
        expected_f1 = 0.0 if count == 1 else 0.85204673
        if not (
            summary.get("status") == "completed"
            and summary.get("job_id") == job
            and summary.get("evaluation_id") == evaluation_id
            and summary.get("sample_order") == sample_order
            and summary.get("requested_samples") == count
            and summary.get("scored_samples") == count
            and math.isclose(float(selected.get("iou")), expected_iou, abs_tol=1e-12)
            and math.isclose(float(selected.get("f1")), expected_f1, abs_tol=1e-12)
            and receipt.get("schema_id") == "qwen3vl_evaluation_receipt_v2"
            and receipt.get("status") == "complete"
            and identity.get("job_id") == job
            and identity.get("arm") == arm
            and identity.get("runtime_commit") == chains[-1]["commit"]
            and identity.get("sample_order") == sample_order
            and gates.get("gt_loaded_before_rollout_complete") is False
            and gates.get("pre_rollout_gt_access_count") == 0
            and gates.get("safety_boundary_relaxation_count") == 0
            and gates.get("requested_sample_count") == count
            and gates.get("scored_sample_count") == count
            and gates.get("requested_scored_mismatch_count") == 0
            and gates.get("tool_failure_count") == 0
            and gates.get("offline_scoring_after_rollout_complete") is True
            and gates.get("summary_validator_passed") is True
        ):
            raise ValueError(f"retry3 summary/evaluation receipt changed for Job {job}")

    for job, expected_hard_gate_count, expected_round_two in (
        ("53157", 34, 1),
        ("53159", 25, 2),
    ):
        pair = json.loads(
            (retry3_roots[job] / "pair_comparison_receipt.json").read_text(
                encoding="utf-8"
            )
        )
        controlled = pair.get("controlled_variable_checks", {})
        hard_gates = pair.get("hard_gates", {})
        comparison = pair.get("comparison", {})
        if not (
            pair.get("schema_id") == "qwen3vl_h0_h1_pair_comparison_v1"
            and pair.get("status") == "complete"
            and pair.get("implementation_commit") == chains[-1]["implementation"]
            and pair.get("freeze_commit") == chains[-1]["commit"]
            and len(controlled) == 18
            and all(value is True for value in controlled.values())
            and len(hard_gates) == expected_hard_gate_count
            and all(value is True for value in hard_gates.values())
            and comparison.get("h0", {}).get("round_two_invoked_count") == 0
            and comparison.get("h1", {}).get("round_two_invoked_count")
            == expected_round_two
            and math.isclose(
                float(comparison.get("diagnostic_delta_selected_iou_h1_minus_h0")),
                0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                float(comparison.get("diagnostic_delta_selected_f1_h1_minus_h0")),
                0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(f"retry3 pair receipt changed for Job {job}")

    if not P1_RETRY3_RESULT.is_file():
        raise FileNotFoundError(f"missing P1 retry3 result: {P1_RETRY3_RESULT}")
    result = json.loads(P1_RETRY3_RESULT.read_text(encoding="utf-8"))
    core = result.get("core3_comparison", {})
    independent = result.get("independent_validation", {})
    analysis_attempts = result.get("trajectory_failure_analysis", {}).get(
        "analysis_job_attempts", []
    )
    if not (
        result.get("schema_id") == "qwen3vl_p1_retry3_result_v1"
        and result.get("result_status") == "complete"
        and result.get("engineering_outcome") == "pass"
        and result.get("scientific_outcome") == "diagnostic_no_gain"
        and result.get("git_lineage", {}).get("implementation_commit_a4")
        == chains[-1]["implementation"]
        and result.get("git_lineage", {}).get("final_freeze_commit_b4")
        == chains[-1]["commit"]
        and independent.get("cross_attempt_pair_composition") is False
        and core.get("sample_order") == FIXED_EVALUATION_ORDERS["DEV-CORE-3-V1"]
        and core.get("h0", {}).get("job_id") == "53158"
        and core.get("h1", {}).get("job_id") == "53159"
        and math.isclose(
            float(core.get("delta_h1_minus_h0", {}).get("micro_iou")),
            0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            float(core.get("delta_h1_minus_h0", {}).get("micro_f1")),
            0.0,
            abs_tol=1e-12,
        )
        and result.get("registry_closeout_target")
        == {
            "experiment_rows": 14,
            "field_dictionary_rows": 52,
            "data_rows": 21,
            "attempt_rows": 38,
        }
        and [(item.get("job_id"), item.get("state")) for item in analysis_attempts]
        == [("53164", "FAILED"), ("53165", "COMPLETED")]
        and analysis_attempts[-1].get("result_sha256")
        == "<hash omitted>"
    ):
        raise ValueError("P1 retry3 result receipt changed")

    if not P1_RETRY3_TRAJECTORY_ANALYSIS.is_file():
        raise FileNotFoundError(
            f"missing P1 retry3 trajectory analysis: {P1_RETRY3_TRAJECTORY_ANALYSIS}"
        )
    if (
        sha256_file(P1_RETRY3_TRAJECTORY_ANALYSIS)
        != "<hash omitted>"
    ):
        raise ValueError("P1 retry3 trajectory analysis SHA256 changed")
    analysis = json.loads(
        P1_RETRY3_TRAJECTORY_ANALYSIS.read_text(encoding="utf-8")
    )
    failure_analysis = analysis.get("per_sample_failure_analysis", {})
    if not (
        analysis.get("schema_id") == "qwen3vl_p1_retry3_trajectory_analysis_v1"
        and analysis.get("status") == "complete"
        and analysis.get("slurm_job_id") == "53165"
        and analysis.get("freeze", {}).get("implementation_commit")
        == chains[-1]["implementation"]
        and analysis.get("freeze", {}).get("freeze_commit") == chains[-1]["commit"]
        and analysis.get("independent_gates", {}).get(
            "cross_attempt_pair_composition"
        )
        is False
        and failure_analysis.get("test_20_15", {}).get("trajectory_class")
        == "safe_rejection_of_three_harmful_candidates_no_gain"
        and failure_analysis.get("test_1_1", {}).get("trajectory_class")
        == "false_finish_complete_false_positive"
        and failure_analysis.get("test_50_8", {}).get("trajectory_class")
        == "false_finish_complete_false_negative"
    ):
        raise ValueError("P1 retry3 trajectory analysis contract changed")

    for group in analysis.get("source_bindings", {}).values():
        for binding in group.values():
            historical_source = Path(binding["path"])
            source = resolve_archived_binding_path(
                historical_source,
                archive_date="0802",
            )
            if not (
                source.is_file()
                and source.stat().st_size == binding["size_bytes"]
                and sha256_file(source) == binding["sha256"]
            ):
                raise ValueError(
                    "P1 result source binding changed after explicit 0802 archive "
                    f"relocation: {historical_source} -> {source}"
                )


def validate_0803_p2_evidence(registry: dict[str, Any]) -> None:
    """Bind the P2 no-go registry row to the immutable formal attempt."""

    experiment_rows = {
        row["实验ID"]: row for row in registry["sheets"][0]["rows"]
    }
    experiment = experiment_rows.get("0803(P2-ACTIONABILITY)")
    if experiment is None:
        raise ValueError("P2 actionability experiment row missing")
    if not (
        (
            experiment["execution_status"],
            experiment["scientific_outcome"],
            experiment["lineage_status"],
        )
        == ("completed", "negative", "canonical")
        and experiment["当前证据路径"]
        == "outputs/CA_0803(1)-qwen3vl-p2-oracle-actionability/attempt-0001"
        and "test_1_1" in str(experiment["状态说明"])
        and "gate=false" in str(experiment["训练后功能状态"])
        and "不得构造 C-ISO/T-TRAJ" in str(experiment["结论/限制"])
        and "不能解释为更大或多步动作空间无解"
        in str(experiment["结论/限制"])
        and "不能声称 LEVIR-CD 泛化" in str(experiment["结论/限制"])
    ):
        raise ValueError("P2 actionability experiment contract changed")

    attempts = {row["attempt_id"]: row for row in registry["sheets"][3]["rows"]}
    expected_attempts = {
        "P2-GENERATION-53272": {
            "stage": "p2_candidate_generation",
            "job": P2_GENERATION_JOB,
            "scientific": "not_evaluated",
            "dependencies": ["P1-CORE-H0-53158", "P1-CORE-H1-53159"],
        },
        "P2-SCORING-53273": {
            "stage": "p2_offline_scoring",
            "job": P2_SCORING_JOB,
            "scientific": "negative",
            "dependencies": ["P2-GENERATION-53272"],
        },
    }
    observed_p2 = {
        attempt_id for attempt_id in attempts if attempt_id.startswith("P2-")
    }
    if observed_p2 != set(expected_attempts):
        raise ValueError(f"P2 attempt membership changed: {sorted(observed_p2)}")
    for attempt_id, expected in expected_attempts.items():
        row = attempts[attempt_id]
        if not (
            row["实验ID"] == "0803(P2-ACTIONABILITY)"
            and row["stage"] == expected["stage"]
            and row["Job ID"] == expected["job"]
            and row["代码commit"] == P2_FREEZE_COMMIT
            and row["retry_of_attempt_id"] == "不适用"
            and json.loads(row["depends_on_attempt_ids"])
            == expected["dependencies"]
            and (
                row["execution_status"],
                row["scientific_outcome"],
                row["lineage_status"],
                row["failure_class"],
            )
            == ("completed", expected["scientific"], "canonical", "not_applicable")
            and row["当前证据路径"] == experiment["当前证据路径"]
        ):
            raise ValueError(f"P2 attempt registry contract changed: {attempt_id}")
        hashes = str(row["配置/模型/工具哈希"])
        for required in (P2_IMPLEMENTATION_COMMIT, P2_FREEZE_COMMIT):
            if required not in hashes:
                raise ValueError(f"P2 attempt {attempt_id} is missing {required}")

    if not P2_ACTIONABILITY_RESULT.is_file():
        raise FileNotFoundError(f"missing P2 result: {P2_ACTIONABILITY_RESULT}")
    result = json.loads(P2_ACTIONABILITY_RESULT.read_text(encoding="utf-8"))
    if not (
        result.get("schema_id") == "qwen3vl_p2_actionability_result_v1"
        and result.get("result_status") == "complete"
        and result.get("engineering_outcome") == "pass"
        and result.get("scientific_outcome") == "negative"
        and result.get("git_lineage", {}).get("implementation_commit_a")
        == P2_IMPLEMENTATION_COMMIT
        and result.get("git_lineage", {}).get("freeze_commit_b")
        == P2_FREEZE_COMMIT
        and result.get("decision", {}).get(
            "proceed_to_c_iso_and_t_traj_data_construction"
        )
        is False
        and result.get("registry_closeout_target")
        == {
            "experiment_rows": 15,
            "field_dictionary_rows": 52,
            "data_rows": 21,
            "attempt_rows": 40,
        }
    ):
        raise ValueError("P2 result receipt changed")

    for relative_path, expected_sha256 in P2_BINDING_SHA256.items():
        artifact = P2_FORMAL_ATTEMPT / relative_path
        if not artifact.is_file() or sha256_file(artifact) != expected_sha256:
            raise ValueError(f"P2 formal artifact changed: {artifact}")

    submission = json.loads(
        (P2_FORMAL_ATTEMPT / "submission_receipt.json").read_text(encoding="utf-8")
    )
    generation = json.loads(
        (P2_FORMAL_ATTEMPT / "generation_receipt.json").read_text(encoding="utf-8")
    )
    scoring = json.loads(
        (P2_FORMAL_ATTEMPT / "scoring_receipt.json").read_text(encoding="utf-8")
    )
    report = json.loads(
        (P2_FORMAL_ATTEMPT / "scoring/score_report.json").read_text(
            encoding="utf-8"
        )
    )
    decision = json.loads(
        (P2_FORMAL_ATTEMPT / "scoring/actionability_decision.json").read_text(
            encoding="utf-8"
        )
    )
    if not (
        submission.get("jobs", {}).get("generation") == P2_GENERATION_JOB
        and submission.get("jobs", {}).get("scoring") == P2_SCORING_JOB
        and submission.get("dependency") == f"afterok:{P2_GENERATION_JOB}"
        and generation.get("status") == "complete"
        and generation.get("job_id") == P2_GENERATION_JOB
        and generation.get("gt_loaded") is False
        and generation.get("counts")
        == {"total": 36, "primary": 27, "diagnostic_executor_only": 9}
        and scoring.get("status") == "complete"
        and scoring.get("job_id") == P2_SCORING_JOB
        and scoring.get("generation_job_id") == P2_GENERATION_JOB
        and scoring.get("gt_access_after_complete_candidate_validation") is True
        and scoring.get("proceed_to_c_iso_and_t_traj_data_construction") is False
        and report.get("candidate_count") == 36
        and report.get("unique_candidate_state_count") == 36
        and report.get("candidate_state_duplicate_count") == 0
        and report.get("oracle_counts")
        == {"beneficial": 4, "harmful": 31, "indeterminate": 1}
        and report.get("qwen_evaluation", {}).get("qwen_invoked") is False
        and decision.get("proceed_to_c_iso_and_t_traj_data_construction") is False
        and decision.get("no_primary_beneficial_candidate_samples") == ["test_1_1"]
        and decision.get("gate_modification_authorized") is False
    ):
        raise ValueError("P2 formal result contract changed")

    targets = decision["target_results"]
    if not (
        targets["test_1_1"]["primary_candidate_count"] == 1
        and targets["test_1_1"]["primary_beneficial_count"] == 0
        and targets["test_1_1"]["best_primary_candidate_deltas"]
        == {"iou": 0.0, "f1": 0.0, "tp": 0, "fp": -60, "fn": 0}
        and targets["test_50_8"]["primary_candidate_count"] == 18
        and targets["test_50_8"]["primary_beneficial_count"] == 2
        and targets["test_50_8"]["primary_beneficial_gate_pass_count"] == 1
        and targets["test_50_8"]["best_primary_candidate_id"]
        == "test_50_8--primary--s8--box--t2"
    ):
        raise ValueError("P2 target actionability decision changed")


def validate_external_evidence(registry: dict[str, Any]) -> None:
    validate_legacy_0729_evidence(registry)
    validate_fixed_evaluation_evidence(registry)
    validate_0801_archive_evidence(registry)
    validate_v2_reconstructed_config(registry)
    validate_0802_p1_evidence(registry)
    validate_0803_p2_evidence(registry)


def xml_text(value: Any) -> str:
    text = INVALID_XML_RE.sub("", str(value))
    return escape(text, {'"': "&quot;"})


def column_name(index: int) -> str:
    if index < 1:
        raise ValueError("column index must be positive")
    result = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def column_width(header: str) -> int:
    normalized = header.lower()
    if any(token in normalized for token in ("sha256", "证据", "目录")):
        return 48
    if any(
        token in normalized
        for token in (
            "json",
            "定义",
            "pipeline",
            "限制",
            "结论",
            "配置",
            "变量",
            "目的",
            "规则",
            "关系",
            "注意",
            "状态",
        )
    ):
        return 38
    if any(token in normalized for token in ("id", "模型", "schema", "protocol")):
        return 28
    return 18


def cell_xml(reference: str, value: Any, style: int) -> str:
    if isinstance(value, bool):
        return f'<c r="{reference}" s="{style}" t="b"><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f'<c r="{reference}" s="{style}" t="n"><v>{value}</v></c>'
    text = xml_text(value)
    preserve = ' xml:space="preserve"' if text != text.strip() else ""
    return f'<c r="{reference}" s="{style}" t="inlineStr"><is><t{preserve}>{text}</t></is></c>'


def worksheet_xml(sheet: dict[str, Any]) -> str:
    headers = sheet["headers"]
    rows = sheet["rows"]
    last_col = column_name(len(headers))
    last_row = len(rows) + 3
    cols = "".join(
        f'<col min="{index}" max="{index}" width="{column_width(header)}" customWidth="1"/>'
        for index, header in enumerate(headers, start=1)
    )
    title = cell_xml("A1", sheet["title"], 1)
    note = cell_xml("A2", sheet["note"], 2)
    header_cells = "".join(
        cell_xml(f"{column_name(index)}3", header, 3)
        for index, header in enumerate(headers, start=1)
    )
    row_xml: list[str] = [
        f'<row r="1" ht="27" customHeight="1">{title}</row>',
        f'<row r="2" ht="60" customHeight="1">{note}</row>',
        f'<row r="3" ht="42" customHeight="1">{header_cells}</row>',
    ]
    for row_number, row in enumerate(rows, start=4):
        cells = []
        for column_index, header in enumerate(headers, start=1):
            value = row[header]
            style = 5 if isinstance(value, str) and value in MISSING_MARKERS else 4
            cells.append(
                cell_xml(f"{column_name(column_index)}{row_number}", value, style)
            )
        row_xml.append(f'<row r="{row_number}">{"".join(cells)}</row>')

    freeze_cell = sheet.get("freeze_cell", "A4")
    match = re.fullmatch(r"([A-Z]+)([0-9]+)", freeze_cell)
    if not match:
        raise ValueError(f"invalid freeze_cell: {freeze_cell}")
    x_split = 0
    for char in match.group(1):
        x_split = x_split * 26 + ord(char) - 64
    x_split -= 1
    y_split = int(match.group(2)) - 1
    pane = (
        f'<pane xSplit="{x_split}" ySplit="{y_split}" topLeftCell="{freeze_cell}" '
        'activePane="bottomRight" state="frozen"/>'
    )
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <dimension ref="A1:{last_col}{last_row}"/>
  <sheetViews><sheetView workbookViewId="0"><pane_placeholder/></sheetView></sheetViews>
  <sheetFormatPr defaultRowHeight="15"/>
  <cols>{cols}</cols>
  <sheetData>{''.join(row_xml)}</sheetData>
  <autoFilter ref="A3:{last_col}{last_row}"/>
  <mergeCells count="2"><mergeCell ref="A1:{last_col}1"/><mergeCell ref="A2:{last_col}2"/></mergeCells>
</worksheet>
'''.replace("<pane_placeholder/>", pane)


def workbook_xml(sheets: list[dict[str, Any]]) -> str:
    sheet_nodes = "".join(
        f'<sheet name="{xml_text(sheet["name"])}" sheetId="{index}" r:id="rId{index}"/>'
        for index, sheet in enumerate(sheets, start=1)
    )
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <bookViews><workbookView activeTab="0"/></bookViews>
  <sheets>{sheet_nodes}</sheets>
  <calcPr calcId="0" fullCalcOnLoad="1"/>
</workbook>
'''


def workbook_rels(sheets: list[dict[str, Any]]) -> str:
    relations = [
        f'<Relationship Id="rId{index}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{index}.xml"/>'
        for index, _ in enumerate(sheets, start=1)
    ]
    relations.append(
        f'<Relationship Id="rId{len(sheets) + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    )
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  {''.join(relations)}
</Relationships>
'''


def app_xml(sheets: list[dict[str, Any]]) -> str:
    names = "".join(f"<vt:lpstr>{xml_text(sheet['name'])}</vt:lpstr>" for sheet in sheets)
    count = len(sheets)
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Change-Agent experiment registry builder</Application>
  <HeadingPairs><vt:vector size="2" baseType="variant"><vt:variant><vt:lpstr>Worksheets</vt:lpstr></vt:variant><vt:variant><vt:i4>{count}</vt:i4></vt:variant></vt:vector></HeadingPairs>
  <TitlesOfParts><vt:vector size="{count}" baseType="lpstr">{names}</vt:vector></TitlesOfParts>
</Properties>
'''


def core_xml(updated_at: str) -> str:
    timestamp = updated_at if "T" in updated_at else f"{updated_at}T00:00:00Z"
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Change-Agent experiment registry</dc:title>
  <dc:creator>Change-Agent maintainers</dc:creator>
  <cp:lastModifiedBy>Change-Agent maintainers</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{timestamp}</dcterms:modified>
</cp:coreProperties>
'''


def build_workbook(registry: dict[str, Any], output: Path) -> None:
    sheets = registry["sheets"]
    overrides = "".join(
        f'<Override PartName="/xl/worksheets/sheet{index}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        for index, _ in enumerate(sheets, start=1)
    )
    parts: dict[str, str] = {
        "[Content_Types].xml": CONTENT_TYPES.format(sheet_overrides=overrides),
        "_rels/.rels": ROOT_RELS,
        "xl/workbook.xml": workbook_xml(sheets),
        "xl/_rels/workbook.xml.rels": workbook_rels(sheets),
        "xl/styles.xml": STYLES,
        "docProps/app.xml": app_xml(sheets),
        "docProps/core.xml": core_xml(registry["updated_at_utc"]),
    }
    for index, sheet in enumerate(sheets, start=1):
        parts[f"xl/worksheets/sheet{index}.xml"] = worksheet_xml(sheet)

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="change-agent-workbook-") as directory:
        temporary = Path(directory) / output.name
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(parts):
                info = zipfile.ZipInfo(name, date_time=(2026, 7, 30, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, parts[name].encode("utf-8"))
        with tempfile.NamedTemporaryFile(
            prefix=".experiment_registry.",
            suffix=".xlsx.tmp",
            dir=output.parent,
            delete=False,
        ) as handle:
            staging = Path(handle.name)
        try:
            shutil.copyfile(temporary, staging)
            staging.replace(output)
        finally:
            staging.unlink(missing_ok=True)


def validate_workbook(path: Path, registry: dict[str, Any]) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing workbook: {path}")
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"corrupt workbook member: {bad}")
        required = {
            "[Content_Types].xml",
            "_rels/.rels",
            "xl/workbook.xml",
            "xl/_rels/workbook.xml.rels",
            "xl/styles.xml",
            "docProps/app.xml",
            "docProps/core.xml",
            *(
                f"xl/worksheets/sheet{index}.xml"
                for index in range(1, len(EXPECTED_SHEETS) + 1)
            ),
        }
        missing = required - set(archive.namelist())
        if missing:
            raise ValueError(f"workbook members missing: {sorted(missing)}")
        for name in required:
            if name.endswith(".xml") or name.endswith(".rels"):
                ET.fromstring(archive.read(name))
        namespace = {
            "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
        }
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheet_names = [
            node.attrib["name"] for node in workbook.findall("m:sheets/m:sheet", namespace)
        ]
        if sheet_names != EXPECTED_SHEETS:
            raise ValueError(f"unexpected workbook sheets: {sheet_names}")
        for index, sheet in enumerate(registry["sheets"], start=1):
            tree = ET.fromstring(archive.read(f"xl/worksheets/sheet{index}.xml"))
            rows = tree.findall("m:sheetData/m:row", namespace)
            expected_count = len(sheet["rows"]) + 3
            if len(rows) != expected_count:
                raise ValueError(
                    f"{sheet['name']} workbook row count {len(rows)} != {expected_count}"
                )
            auto_filter = tree.find("m:autoFilter", namespace)
            if auto_filter is None:
                raise ValueError(f"{sheet['name']} lacks autoFilter")


def validate_workbook_matches_source(path: Path, registry: dict[str, Any]) -> None:
    """Reject manual workbook drift from the versionable JSON authority."""

    with tempfile.TemporaryDirectory(prefix="change-agent-workbook-check-") as directory:
        expected = Path(directory) / path.name
        build_workbook(registry, expected)
        expected_hash = sha256_file(expected)
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError(
            "workbook differs from experiment_registry.json; rebuild it before use: "
            f"{actual_hash} != {expected_hash}"
        )


def main() -> None:
    args = parse_args()
    registry = load_registry()
    validate_external_evidence(registry)
    if not args.check:
        build_workbook(registry, OUTPUT)
    validate_workbook(OUTPUT, registry)
    validate_workbook_matches_source(OUTPUT, registry)
    print(
        json.dumps(
            {
                "status": "valid",
                "mode": "check" if args.check else "build",
                "workbook": str(OUTPUT),
                "sheets": [
                    {"name": sheet["name"], "data_rows": len(sheet["rows"])}
                    for sheet in registry["sheets"]
                ],
                "sha256": sha256_file(OUTPUT),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
