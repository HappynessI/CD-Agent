from __future__ import annotations

import copy
import importlib.util
import json
import re
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = REPO_ROOT / "experiments" / "build_experiment_workbook.py"
WORKBOOK_PATH = REPO_ROOT / "experiments" / "experiment_registry.xlsx"

EXPECTED_SHEETS = (
    "01_实验记录",
    "02_字段字典",
    "03_数据登记",
    "04_运行尝试",
)
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
EXPECTED_ROW_COUNTS = {
    "01_实验记录": 15,
    "02_字段字典": 52,
    "03_数据登记": 21,
    "04_运行尝试": 40,
}

EXPECTED_EXECUTION_STATUSES = {
    "planned",
    "running",
    "completed",
    "failed",
    "cancelled",
}
EXPECTED_SCIENTIFIC_OUTCOMES = {
    "not_evaluated",
    "baseline_reference",
    "pass",
    "negative",
    "inconclusive",
}
EXPECTED_LINEAGE_STATUSES = {"canonical", "superseded"}
EXPECTED_FAILURE_CLASSES = {
    "not_applicable",
    "infra_failed",
    "contract_failed",
    "test_failed",
}

EXPECTED_ATTEMPT_IDS = {
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
    "P1-CPU-53064",
    "P1-SMOKE-H0-53065",
    "P1-SMOKE-H1-53066",
    "P1-CORE-H0-53067",
    "P1-CORE-H1-53068",
    "P1-GUARD-53069",
    "P1-CPU-53109",
    "P1-SMOKE-H0-53110",
    "P1-SMOKE-H1-53111",
    "P1-CORE-H0-53112",
    "P1-CORE-H1-53113",
    "P1-GUARD-53114",
    "P1-CPU-53141",
    "P1-SMOKE-H0-53142",
    "P1-SMOKE-H1-53143",
    "P1-CORE-H0-53144",
    "P1-CORE-H1-53145",
    "P1-GUARD-53146",
    "P1-CPU-53155",
    "P1-SMOKE-H0-53156",
    "P1-SMOKE-H1-53157",
    "P1-CORE-H0-53158",
    "P1-CORE-H1-53159",
    "P1-GUARD-53160",
    "P2-GENERATION-53272",
    "P2-SCORING-53273",
}
EXPECTED_RETRY_OF = {
    "V1-DATA-51883": "V1-DATA-51876",
    "V1-SFT-51896": "V1-SFT-51891",
    "V3-DATA-52132": "V3-DATA-52129",
    "V5-MANIFEST-52315": "V5-MANIFEST-52311",
    "V5-MANIFEST-52319": "V5-MANIFEST-52315",
    "P1-CPU-53109": "P1-CPU-53064",
    "P1-SMOKE-H0-53110": "P1-SMOKE-H0-53065",
    "P1-SMOKE-H1-53111": "P1-SMOKE-H1-53066",
    "P1-CORE-H0-53112": "P1-CORE-H0-53067",
    "P1-CORE-H1-53113": "P1-CORE-H1-53068",
    "P1-GUARD-53114": "P1-GUARD-53069",
    "P1-CPU-53141": "P1-CPU-53109",
    "P1-SMOKE-H0-53142": "P1-SMOKE-H0-53110",
    "P1-SMOKE-H1-53143": "P1-SMOKE-H1-53111",
    "P1-CORE-H0-53144": "P1-CORE-H0-53112",
    "P1-CORE-H1-53145": "P1-CORE-H1-53113",
    "P1-GUARD-53146": "P1-GUARD-53114",
    "P1-CPU-53155": "P1-CPU-53141",
    "P1-SMOKE-H0-53156": "P1-SMOKE-H0-53142",
    "P1-SMOKE-H1-53157": "P1-SMOKE-H1-53143",
    "P1-CORE-H0-53158": "P1-CORE-H0-53144",
    "P1-CORE-H1-53159": "P1-CORE-H1-53145",
    "P1-GUARD-53160": "P1-GUARD-53146",
}
EXPECTED_DEPENDENCIES = {
    "BASE-E2E5-51731": ["BASE-E2E3-51727"],
    "V1-DATA-51876": ["V1-CURATION-51847"],
    "V1-DATA-51883": ["V1-CURATION-51847"],
    "V1-SFT-51891": ["V1-DATA-51883"],
    "V1-SFT-51896": ["V1-DATA-51883"],
    "V1-E2E3-51903": ["V1-SFT-51896"],
    "V5-MANIFEST-52311": ["V5-DATA-52307"],
    "V5-MANIFEST-52315": ["V5-DATA-52307"],
    "V5-MANIFEST-52319": ["V5-DATA-52307"],
    "P1-SMOKE-H0-53065": ["P1-CPU-53064"],
    "P1-SMOKE-H1-53066": ["P1-SMOKE-H0-53065"],
    "P1-CORE-H0-53067": ["P1-SMOKE-H1-53066"],
    "P1-CORE-H1-53068": ["P1-CORE-H0-53067"],
    "P1-GUARD-53069": ["P1-CPU-53064","P1-SMOKE-H0-53065","P1-SMOKE-H1-53066","P1-CORE-H0-53067","P1-CORE-H1-53068"],
    "P1-SMOKE-H0-53110": ["P1-CPU-53109"],
    "P1-SMOKE-H1-53111": ["P1-SMOKE-H0-53110"],
    "P1-CORE-H0-53112": ["P1-SMOKE-H1-53111"],
    "P1-CORE-H1-53113": ["P1-CORE-H0-53112"],
    "P1-GUARD-53114": ["P1-CPU-53109","P1-SMOKE-H0-53110","P1-SMOKE-H1-53111","P1-CORE-H0-53112","P1-CORE-H1-53113"],
    "P1-SMOKE-H0-53142": ["P1-CPU-53141"],
    "P1-SMOKE-H1-53143": ["P1-SMOKE-H0-53142"],
    "P1-CORE-H0-53144": ["P1-SMOKE-H1-53143"],
    "P1-CORE-H1-53145": ["P1-CORE-H0-53144"],
    "P1-GUARD-53146": ["P1-CPU-53141","P1-SMOKE-H0-53142","P1-SMOKE-H1-53143","P1-CORE-H0-53144","P1-CORE-H1-53145"],
    "P1-SMOKE-H0-53156": ["P1-CPU-53155"],
    "P1-SMOKE-H1-53157": ["P1-SMOKE-H0-53156"],
    "P1-CORE-H0-53158": ["P1-SMOKE-H1-53157"],
    "P1-CORE-H1-53159": ["P1-CORE-H0-53158"],
    "P1-GUARD-53160": ["P1-CPU-53155","P1-SMOKE-H0-53156","P1-SMOKE-H1-53157","P1-CORE-H0-53158","P1-CORE-H1-53159"],
    "P2-GENERATION-53272": ["P1-CORE-H0-53158","P1-CORE-H1-53159"],
    "P2-SCORING-53273": ["P2-GENERATION-53272"],
}
EXPECTED_CONCRETE_FAILURES = {
    "V1-DATA-51876": "infra_failed",
    "V1-SFT-51891": "infra_failed",
    "V3-DATA-52129": "contract_failed",
    "V5-MANIFEST-52311": "test_failed",
    "P1-SMOKE-H0-53065": "contract_failed",
    "P1-GUARD-53069": "contract_failed",
    "P1-SMOKE-H1-53111": "contract_failed",
    "P1-CORE-H1-53145": "contract_failed",
}
EXPECTED_FAILURE_DIAGNOSTIC_SHA256 = {
    "V1-DATA-51876": "<hash omitted>",
    "V1-SFT-51891": "<hash omitted>",
    "V3-DATA-52129": "<hash omitted>",
    "V5-MANIFEST-52311": "<hash omitted>",
    "P1-SMOKE-H0-53065": "<hash omitted>",
    "P1-GUARD-53069": "<hash omitted>",
    "P1-SMOKE-H1-53111": "<hash omitted>",
    "P1-CORE-H1-53145": "<hash omitted>",
}

P1_EXPERIMENT_IDS = {"0802(P1-H0)", "0802(P1-H1)"}
P1_ATTEMPT_IDS = {
    "P1-CPU-53064",
    "P1-SMOKE-H0-53065",
    "P1-SMOKE-H1-53066",
    "P1-CORE-H0-53067",
    "P1-CORE-H1-53068",
    "P1-GUARD-53069",
    "P1-CPU-53109",
    "P1-SMOKE-H0-53110",
    "P1-SMOKE-H1-53111",
    "P1-CORE-H0-53112",
    "P1-CORE-H1-53113",
    "P1-GUARD-53114",
    "P1-CPU-53141",
    "P1-SMOKE-H0-53142",
    "P1-SMOKE-H1-53143",
    "P1-CORE-H0-53144",
    "P1-CORE-H1-53145",
    "P1-GUARD-53146",
    "P1-CPU-53155",
    "P1-SMOKE-H0-53156",
    "P1-SMOKE-H1-53157",
    "P1-CORE-H0-53158",
    "P1-CORE-H1-53159",
    "P1-GUARD-53160",
}
P2_EXPERIMENT_ID = "0803(P2-ACTIONABILITY)"
P2_ATTEMPT_IDS = {"P2-GENERATION-53272", "P2-SCORING-53273"}

POST_V1_EXPERIMENT_IDS = {f"0801({index})" for index in range(1, 9)}
POST_V1_DATA_IDS = {
    "TRAIN-0801-V121-HARDCASE-V2",
    "TRAIN-0801-V121-ACTION-V3",
    "TRAIN-0801-V121-ACTION-V3-BGFP",
    "TRAIN-0801-V121-ACTION-V3-BGFP-SCOPE",
    "TRAIN-0801-V121-ACTION-V4-RESIDUAL",
    "TRAIN-0801-V121-ACTION-V5-CONTRAST-REPAIR",
    "TRAIN-0801-V121-ACTION-V6-MATCHED-CONTRAST",
    "TRAIN-0801-V121-ACTION-V7-WEIGHTED-CONTRAST",
}

XML_NAMESPACE = {
    "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
}
CELL_REFERENCE_RE = re.compile(r"^([A-Z]+)[1-9][0-9]*$")


def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "experiment_registry_builder_under_test", BUILDER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load registry builder from {BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sheet(payload: dict, name: str) -> dict:
    return next(sheet for sheet in payload["sheets"] if sheet["name"] == name)


def _attempt(payload: dict, attempt_id: str) -> dict:
    rows = _sheet(payload, "04_运行尝试")["rows"]
    return next(row for row in rows if row["attempt_id"] == attempt_id)


def _decode_paths(value: str) -> list[str]:
    if value in {"未运行", "未记录", "未测量", "不适用", "null"}:
        return []
    decoded = json.loads(value) if value.startswith("[") else [value]
    if not isinstance(decoded, list) or not all(
        isinstance(item, str) for item in decoded
    ):
        raise AssertionError(f"invalid path representation: {value!r}")
    return decoded


def _column_index(reference: str) -> int:
    match = CELL_REFERENCE_RE.fullmatch(reference)
    if match is None:
        raise AssertionError(f"invalid XLSX cell reference: {reference!r}")
    result = 0
    for character in match.group(1):
        result = result * 26 + ord(character) - ord("A") + 1
    return result - 1


def _cell_value(cell: ET.Element):
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(
            node.text or "" for node in cell.findall(".//m:t", XML_NAMESPACE)
        )
    value_node = cell.find("m:v", XML_NAMESPACE)
    if value_node is None or value_node.text is None:
        return None
    if cell_type == "b":
        return value_node.text == "1"
    if cell_type == "n":
        if re.fullmatch(r"-?[0-9]+", value_node.text):
            return int(value_node.text)
        return float(value_node.text)
    raise AssertionError(f"unsupported XLSX cell type: {cell_type!r}")


def _worksheet_matrix(archive: zipfile.ZipFile, index: int) -> list[list]:
    root = ET.fromstring(archive.read(f"xl/worksheets/sheet{index}.xml"))
    matrix = []
    for row in root.findall("m:sheetData/m:row", XML_NAMESPACE):
        values = []
        for cell in row.findall("m:c", XML_NAMESPACE):
            column = _column_index(cell.attrib["r"])
            if len(values) <= column:
                values.extend([None] * (column + 1 - len(values)))
            values[column] = _cell_value(cell)
        matrix.append(values)
    return matrix


class ExperimentRegistryContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.builder = _load_builder()
        cls.registry = cls.builder.load_registry()

    def _assert_registry_rejected(self, payload: dict, message_pattern: str) -> None:
        with tempfile.TemporaryDirectory(prefix="registry-invalid-") as directory:
            source = Path(directory) / "experiment_registry.json"
            source.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
            with patch.object(self.builder, "SOURCE", source):
                with self.assertRaisesRegex(ValueError, message_pattern):
                    self.builder.load_registry()

    def _future_registry(self, execution_status: str, current_path: str) -> dict:
        payload = copy.deepcopy(self.registry)
        experiment = copy.deepcopy(_sheet(payload, "01_实验记录")["rows"][0])
        experiment.update(
            {
                "实验ID": "0802(P0-SYNTHETIC)",
                "运行记录ID": "CA_0802-p0-registry-contract-synthetic",
                "实际执行日期UTC": "未运行",
                "记录阶段": "P0 synthetic registry contract",
                "execution_status": execution_status,
                "scientific_outcome": "not_evaluated",
                "lineage_status": "canonical",
                "历史运行路径": "[]",
                "当前证据路径": current_path,
            }
        )
        attempt = copy.deepcopy(_sheet(payload, "04_运行尝试")["rows"][0])
        attempt.update(
            {
                "attempt_id": "P0-SYNTHETIC-0802",
                "实验ID": experiment["实验ID"],
                "retry_of_attempt_id": "不适用",
                "depends_on_attempt_ids": "[]",
                "stage": "p0_registry_contract",
                "Job ID": "P0-SYNTHETIC-NOT-SUBMITTED",
                "execution_status": execution_status,
                "scientific_outcome": "not_evaluated",
                "lineage_status": "canonical",
                "failure_class": "not_applicable",
                "实际执行日期UTC": "未运行",
                "历史运行路径": "[]",
                "当前证据路径": current_path,
            }
        )
        _sheet(payload, "01_实验记录")["rows"] = [experiment]
        _sheet(payload, "03_数据登记")["rows"] = []
        _sheet(payload, "04_运行尝试")["rows"] = [attempt]
        return payload

    def _load_future_registry(self, payload: dict, root: Path) -> dict:
        source = root / "experiment_registry.json"
        source.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        with (
            patch.object(self.builder, "ROOT", root),
            patch.object(self.builder, "SOURCE", source),
        ):
            return self.builder.load_registry()

    def test_schema_sheet_headers_and_row_counts_are_frozen(self):
        self.assertEqual(
            self.registry["schema_version"], "chang_agent_experiment_registry_v2"
        )
        self.assertEqual(
            tuple(sheet["name"] for sheet in self.registry["sheets"]),
            EXPECTED_SHEETS,
        )
        self.assertEqual(tuple(self.builder.EXPECTED_SHEETS), EXPECTED_SHEETS)
        self.assertEqual(self.builder.EXPECTED_HEADERS, EXPECTED_HEADERS)
        for name in EXPECTED_SHEETS:
            with self.subTest(sheet=name):
                sheet = _sheet(self.registry, name)
                self.assertEqual(tuple(sheet["headers"]), EXPECTED_HEADERS[name])
                self.assertEqual(len(sheet["rows"]), EXPECTED_ROW_COUNTS[name])
                self.assertTrue(
                    all(set(row) == set(EXPECTED_HEADERS[name]) for row in sheet["rows"])
                )

    def test_status_axes_and_failure_class_are_closed_enums(self):
        expected_axes = (
            ("EXECUTION_STATUSES", EXPECTED_EXECUTION_STATUSES),
            ("SCIENTIFIC_OUTCOMES", EXPECTED_SCIENTIFIC_OUTCOMES),
            ("LINEAGE_STATUSES", EXPECTED_LINEAGE_STATUSES),
            ("FAILURE_CLASSES", EXPECTED_FAILURE_CLASSES),
        )
        for attribute, expected in expected_axes:
            with self.subTest(attribute=attribute):
                self.assertEqual(set(getattr(self.builder, attribute)), expected)

        for sheet_name in ("01_实验记录", "04_运行尝试"):
            rows = _sheet(self.registry, sheet_name)["rows"]
            self.assertLessEqual(
                {row["execution_status"] for row in rows},
                EXPECTED_EXECUTION_STATUSES,
            )
            self.assertLessEqual(
                {row["scientific_outcome"] for row in rows},
                EXPECTED_SCIENTIFIC_OUTCOMES,
            )
            self.assertLessEqual(
                {row["lineage_status"] for row in rows},
                EXPECTED_LINEAGE_STATUSES,
            )
        attempts = _sheet(self.registry, "04_运行尝试")["rows"]
        self.assertLessEqual(
            {row["failure_class"] for row in attempts}, EXPECTED_FAILURE_CLASSES
        )

    def test_attempt_ledger_membership_retry_chains_and_dependencies_are_frozen(self):
        attempts = {
            row["attempt_id"]: row
            for row in _sheet(self.registry, "04_运行尝试")["rows"]
        }
        self.assertEqual(set(attempts), EXPECTED_ATTEMPT_IDS)
        self.assertIn("V1-CURATION-51847", attempts)

        retry_of = {
            attempt_id: row["retry_of_attempt_id"]
            for attempt_id, row in attempts.items()
            if row["retry_of_attempt_id"] != "不适用"
        }
        self.assertEqual(retry_of, EXPECTED_RETRY_OF)

        dependencies = {
            attempt_id: json.loads(row["depends_on_attempt_ids"])
            for attempt_id, row in attempts.items()
            if json.loads(row["depends_on_attempt_ids"])
        }
        self.assertEqual(dependencies, EXPECTED_DEPENDENCIES)
        for attempt_id, predecessor in retry_of.items():
            self.assertNotIn(predecessor, dependencies.get(attempt_id, []))

        concrete_failures = {
            attempt_id: row["failure_class"]
            for attempt_id, row in attempts.items()
            if row["failure_class"] != "not_applicable"
        }
        self.assertEqual(concrete_failures, EXPECTED_CONCRETE_FAILURES)
        for attempt_id, failure_class in concrete_failures.items():
            row = attempts[attempt_id]
            self.assertEqual(row["execution_status"], "failed")
            self.assertEqual(row["scientific_outcome"], "not_evaluated")
            self.assertEqual(row["lineage_status"], "superseded")
            self.assertIn(failure_class, EXPECTED_FAILURE_CLASSES)
            self.assertIn(
                EXPECTED_FAILURE_DIAGNOSTIC_SHA256[attempt_id],
                row["配置/模型/工具哈希"],
            )

    def test_p1_arm_rows_smoke_identity_and_conservative_scientific_axes_are_frozen(self):
        experiments = {
            row["实验ID"]: row
            for row in _sheet(self.registry, "01_实验记录")["rows"]
        }
        self.assertTrue(P1_EXPERIMENT_IDS.issubset(experiments))
        self.assertEqual(
            (
                experiments["0802(P1-H0)"]["execution_status"],
                experiments["0802(P1-H0)"]["scientific_outcome"],
                experiments["0802(P1-H0)"]["lineage_status"],
            ),
            ("completed", "baseline_reference", "canonical"),
        )
        self.assertEqual(
            (
                experiments["0802(P1-H1)"]["execution_status"],
                experiments["0802(P1-H1)"]["scientific_outcome"],
                experiments["0802(P1-H1)"]["lineage_status"],
            ),
            ("completed", "pass", "canonical"),
        )
        for experiment_id in P1_EXPERIMENT_IDS:
            row = experiments[experiment_id]
            self.assertAlmostEqual(float(row["后IoU"]), 0.7422312)
            self.assertAlmostEqual(float(row["后F1"]), 0.85204673)
            self.assertEqual(float(row["ΔIoU"]), 0.0)
            self.assertEqual(float(row["ΔF1"]), 0.0)
        self.assertIn(
            "不能声称分割提升", experiments["0802(P1-H1)"]["结论/限制"]
        )
        self.assertIn(
            "不能声称 LEVIR-CD 泛化",
            experiments["0802(P1-H1)"]["结论/限制"],
        )

        data_rows = _sheet(self.registry, "03_数据登记")["rows"]
        smoke = next(
            row for row in data_rows if row["数据ID"] == "DEV-SMOKE-1-H1-V1"
        )
        self.assertEqual(smoke["sample_id/顺序"], "1 test_1_1")
        self.assertEqual(smoke["样本数"], 1)
        self.assertEqual(smoke["GT正像素"], 0)
        test_1 = next(
            row
            for row in data_rows
            if row["记录类型"] == "评测样本成员"
            and '"sample_id":"test_1_1"' in row["sample_id/顺序"]
        )
        membership = json.loads(test_1["sample_id/顺序"])
        self.assertEqual(membership["DEV-SMOKE-1-H1-V1"], 1)
        self.assertIn("DEV-SMOKE-1-H1-V1", test_1["数据ID"])

        attempts = {
            row["attempt_id"]: row
            for row in _sheet(self.registry, "04_运行尝试")["rows"]
        }
        self.assertEqual(
            {attempt_id for attempt_id in attempts if attempt_id.startswith("P1-")},
            P1_ATTEMPT_IDS,
        )
        canonical = {
            attempt_id
            for attempt_id in P1_ATTEMPT_IDS
            if attempts[attempt_id]["lineage_status"] == "canonical"
        }
        self.assertEqual(
            canonical,
            {
                "P1-CPU-53155",
                "P1-SMOKE-H0-53156",
                "P1-SMOKE-H1-53157",
                "P1-CORE-H0-53158",
                "P1-CORE-H1-53159",
                "P1-GUARD-53160",
            },
        )
        self.assertEqual(
            attempts["P1-CORE-H0-53158"]["scientific_outcome"],
            "baseline_reference",
        )
        self.assertEqual(
            attempts["P1-CORE-H0-53144"]["scientific_outcome"],
            "baseline_reference",
        )
        self.assertEqual(
            attempts["P1-CORE-H1-53159"]["scientific_outcome"], "pass"
        )
        for attempt_id in P1_ATTEMPT_IDS - {
            "P1-CORE-H0-53144",
            "P1-CORE-H0-53158",
            "P1-CORE-H1-53159",
        }:
            self.assertEqual(
                attempts[attempt_id]["scientific_outcome"], "not_evaluated"
            )

    def test_p2_actionability_no_go_and_attempts_are_frozen(self):
        experiments = {
            row["实验ID"]: row
            for row in _sheet(self.registry, "01_实验记录")["rows"]
        }
        row = experiments[P2_EXPERIMENT_ID]
        self.assertEqual(
            (row["execution_status"], row["scientific_outcome"], row["lineage_status"]),
            ("completed", "negative", "canonical"),
        )
        self.assertIn("test_1_1", row["状态说明"])
        self.assertIn("gate=false", row["训练后功能状态"])
        self.assertIn("不得构造 C-ISO/T-TRAJ", row["结论/限制"])
        self.assertIn("不能解释为更大或多步动作空间无解", row["结论/限制"])
        self.assertIn("不能声称 LEVIR-CD 泛化", row["结论/限制"])

        attempts = {
            item["attempt_id"]: item
            for item in _sheet(self.registry, "04_运行尝试")["rows"]
        }
        self.assertEqual(
            {attempt_id for attempt_id in attempts if attempt_id.startswith("P2-")},
            P2_ATTEMPT_IDS,
        )
        generation = attempts["P2-GENERATION-53272"]
        scoring = attempts["P2-SCORING-53273"]
        self.assertEqual(generation["scientific_outcome"], "not_evaluated")
        self.assertEqual(scoring["scientific_outcome"], "negative")
        self.assertEqual(generation["代码commit"], self.builder.P2_FREEZE_COMMIT)
        self.assertEqual(scoring["代码commit"], self.builder.P2_FREEZE_COMMIT)
        self.assertIn("36/36", generation["核心指标/结果"])
        self.assertIn("proceed=false", scoring["核心指标/结果"])
        self.assertEqual(
            json.loads(scoring["depends_on_attempt_ids"]),
            ["P2-GENERATION-53272"],
        )

    def test_v1_data_producer_and_revalidation_receipt_are_distinct(self):
        producer_commit = "41a2bd193ebb431c049cf91a9145b830a7cafe89"
        receipt_commit = "9fad7fbacdd1da5dce9ff6d4559758c3e41f25b3"
        producer = _attempt(self.registry, "V1-DATA-51876")
        revalidation = _attempt(self.registry, "V1-DATA-51883")
        self.assertEqual(producer["代码commit"], producer_commit)
        self.assertEqual(revalidation["代码commit"], receipt_commit)
        self.assertNotEqual(producer["代码commit"], revalidation["代码commit"])

        evidence_root = REPO_ROOT / producer["当前证据路径"]
        self.assertEqual(
            evidence_root,
            REPO_ROOT / revalidation["当前证据路径"],
        )
        manifest_path = evidence_root / "data" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        producer_submission = self.builder.read_key_value_receipt(
            evidence_root / "data-build-submission.txt"
        )
        producer_status = self.builder.read_key_value_receipt(
            evidence_root / "logs" / "data-build-status-51876.txt"
        )
        revalidation_submission = self.builder.read_key_value_receipt(
            evidence_root / "data-build-submission-51883.txt"
        )
        revalidation_status = self.builder.read_key_value_receipt(
            evidence_root / "logs" / "data-build-status-51883.txt"
        )
        self.assertEqual(manifest["runtime_commit"], producer_commit)
        self.assertEqual(
            self.builder.sha256_file(manifest_path),
            "<hash omitted>",
        )
        self.assertEqual(
            set(producer_submission), {"data_build_job", "code_commit", "run_dir"}
        )
        self.assertEqual(producer_submission["data_build_job"], "51876")
        self.assertEqual(producer_submission["code_commit"], producer_commit)
        self.assertEqual(
            producer_status,
            {
                "status": "failed",
                "exit_code": "1",
                "elapsed_seconds": "3",
                "result": "",
            },
        )
        self.assertEqual(
            set(revalidation_submission),
            {"data_build_job", "code_commit", "run_dir"},
        )
        self.assertEqual(revalidation_submission["data_build_job"], "51883")
        self.assertEqual(revalidation_submission["code_commit"], receipt_commit)
        self.assertEqual(revalidation_status["status"], "success")
        self.assertEqual(revalidation_status["exit_code"], "0")
        self.assertEqual(revalidation_status["result"], "train=39;val=24;valid=True")
        self.assertNotEqual(manifest["runtime_commit"], receipt_commit)

    def test_post_v1_main_and_data_rows_preserve_historical_and_archive_paths(self):
        experiments = {
            row["实验ID"]: row
            for row in _sheet(self.registry, "01_实验记录")["rows"]
        }
        data_rows = {
            row["数据ID"]: row for row in _sheet(self.registry, "03_数据登记")["rows"]
        }
        self.assertTrue(POST_V1_EXPERIMENT_IDS.issubset(experiments))
        self.assertTrue(POST_V1_DATA_IDS.issubset(data_rows))

        selected_rows = [experiments[item] for item in sorted(POST_V1_EXPERIMENT_IDS)]
        selected_rows.extend(data_rows[item] for item in sorted(POST_V1_DATA_IDS))
        for row in selected_rows:
            identity = row.get("实验ID", row.get("数据ID"))
            with self.subTest(identity=identity):
                historical = _decode_paths(row["历史运行路径"])
                current = _decode_paths(row["当前证据路径"])
                self.assertTrue(historical)
                self.assertEqual(len(current), len(historical))
                self.assertEqual(
                    current,
                    [
                        "outputs/CA_0801_archive/" + path.removeprefix("outputs/")
                        for path in historical
                    ],
                )
                self.assertTrue(
                    all(path.startswith("outputs/CA_0801(") for path in historical)
                )
                self.assertTrue(
                    all(
                        path.startswith("outputs/CA_0801_archive/CA_0801(")
                        for path in current
                    )
                )

    def test_p1_paths_preserve_history_and_explicit_0802_archive_evidence(self):
        experiments = {
            row["实验ID"]: row
            for row in _sheet(self.registry, "01_实验记录")["rows"]
        }
        attempts = {
            row["attempt_id"]: row
            for row in _sheet(self.registry, "04_运行尝试")["rows"]
        }
        selected_rows = [experiments[item] for item in sorted(P1_EXPERIMENT_IDS)]
        selected_rows.extend(attempts[item] for item in sorted(P1_ATTEMPT_IDS))

        for row in selected_rows:
            identity = row.get("实验ID", row.get("attempt_id"))
            with self.subTest(identity=identity):
                historical = _decode_paths(row["历史运行路径"])
                current = _decode_paths(row["当前证据路径"])
                self.assertTrue(historical)
                self.assertEqual(
                    current,
                    [
                        self.builder.archive_output_path(
                            path,
                            archive_date="0802",
                        )
                        for path in historical
                    ],
                )
                for path in current:
                    self.assertIn(path, row["权威证据"])

        historical_binding = (
            self.builder.HISTORICAL_REPOSITORY_ROOT
            / "outputs/CA_0802(5)-qwen3vl-h0-base-core3-retry3/"
            "baseline_summary.json"
        )
        self.assertEqual(
            self.builder.resolve_archived_binding_path(
                historical_binding,
                archive_date="0802",
            ),
            REPO_ROOT
            / "outputs/CA_0802_archive/"
            "CA_0802(5)-qwen3vl-h0-base-core3-retry3/baseline_summary.json",
        )

    def test_p1_archive_policy_rejects_stale_or_guessed_current_paths(self):
        for current_path in (
            "outputs/CA_0802(5)-qwen3vl-h0-base-core3-retry3",
            "outputs/CA_0801_archive/CA_0802(5)-qwen3vl-h0-base-core3-retry3",
        ):
            with self.subTest(current_path=current_path):
                payload = copy.deepcopy(self.registry)
                _attempt(payload, "P1-CORE-H0-53158")[
                    "当前证据路径"
                ] = current_path
                self._assert_registry_rejected(
                    payload,
                    "current evidence must",
                )

        with self.assertRaisesRegex(ValueError, "historical 0802 path"):
            self.builder.resolve_archived_binding_path(
                self.builder.HISTORICAL_REPOSITORY_ROOT
                / "outputs/CA_0802_archive/"
                "CA_0802(5)-qwen3vl-h0-base-core3-retry3/baseline_summary.json",
                archive_date="0802",
            )

    def test_missing_legacy_0729_does_not_skip_independent_evidence_audits(self):
        with tempfile.TemporaryDirectory(prefix="registry-missing-0729-") as directory:
            missing = Path(directory) / "smoke_metrics.json"
            self.assertFalse(missing.exists())
            with (
                patch.object(self.builder, "SMOKE_METRICS", missing),
                patch.object(
                    self.builder,
                    "validate_legacy_0729_evidence",
                    wraps=self.builder.validate_legacy_0729_evidence,
                ) as legacy,
                patch.object(
                    self.builder, "validate_fixed_evaluation_evidence"
                ) as fixed,
                patch.object(
                    self.builder, "validate_0801_archive_evidence"
                ) as archive,
                patch.object(
                    self.builder, "validate_v2_reconstructed_config"
                ) as reconstructed,
                patch.object(self.builder, "validate_0802_p1_evidence") as p1,
                patch.object(self.builder, "validate_0803_p2_evidence") as p2,
            ):
                self.builder.validate_external_evidence(self.registry)
            legacy.assert_called_once_with(self.registry)
            fixed.assert_called_once_with(self.registry)
            archive.assert_called_once_with(self.registry)
            reconstructed.assert_called_once_with(self.registry)
            p1.assert_called_once_with(self.registry)
            p2.assert_called_once_with(self.registry)

    def test_future_0802_paths_distinguish_planned_missing_and_terminal_evidence(self):
        relative_path = "outputs/CA_0802-p0-registry-contract-synthetic"
        with tempfile.TemporaryDirectory(prefix="registry-future-") as directory:
            root = Path(directory)
            planned = self._future_registry("planned", relative_path)
            loaded = self._load_future_registry(planned, root)
            self.assertEqual(
                _sheet(loaded, "01_实验记录")["rows"][0]["execution_status"],
                "planned",
            )

            terminal = self._future_registry("completed", relative_path)
            with self.assertRaisesRegex(
                FileNotFoundError, "current evidence path missing"
            ):
                self._load_future_registry(terminal, root)

            (root / relative_path).mkdir(parents=True)
            loaded = self._load_future_registry(terminal, root)
            self.assertEqual(
                _sheet(loaded, "04_运行尝试")["rows"][0]["execution_status"],
                "completed",
            )

    def test_future_0802_paths_reject_absolute_parent_and_backslash_forms(self):
        invalid_paths = (
            "/tmp/CA_0802-absolute",
            "outputs/../CA_0802-parent",
            r"outputs\CA_0802-backslash",
        )
        for current_path in invalid_paths:
            with self.subTest(current_path=current_path):
                payload = self._future_registry("planned", current_path)
                with tempfile.TemporaryDirectory(
                    prefix="registry-future-invalid-"
                ) as directory:
                    with self.assertRaisesRegex(
                        ValueError, "safe repo-relative outputs/"
                    ):
                        self._load_future_registry(payload, Path(directory))

    def test_invalid_axis_and_failure_class_contracts_are_rejected(self):
        invalid_enums = (
            ("execution_status", "finished", "invalid execution_status"),
            ("scientific_outcome", "improved", "invalid scientific_outcome"),
            ("lineage_status", "latest", "invalid lineage_status"),
            ("failure_class", "model_failed", "invalid failure_class"),
        )
        for field, value, message in invalid_enums:
            with self.subTest(field=field):
                payload = copy.deepcopy(self.registry)
                _attempt(payload, "BASE-E2E3-51727")[field] = value
                self._assert_registry_rejected(payload, message)

        payload = copy.deepcopy(self.registry)
        failed = _attempt(payload, "BASE-E2E3-51727")
        failed["execution_status"] = "failed"
        failed["scientific_outcome"] = "not_evaluated"
        failed["failure_class"] = "not_applicable"
        self._assert_registry_rejected(payload, "must have a concrete failure_class")

        payload = copy.deepcopy(self.registry)
        _attempt(payload, "BASE-E2E3-51727")["failure_class"] = "infra_failed"
        self._assert_registry_rejected(payload, "must be failed/not_evaluated")

    def test_invalid_retry_and_dependency_references_are_rejected(self):
        invalid_references = (
            (
                "V1-DATA-51883",
                "retry_of_attempt_id",
                "UNKNOWN-ATTEMPT",
                "retries unknown attempt",
            ),
            (
                "BASE-E2E3-51727",
                "depends_on_attempt_ids",
                '["UNKNOWN-ATTEMPT"]',
                "has unknown dependencies",
            ),
            (
                "V1-DATA-51883",
                "retry_of_attempt_id",
                "V3-DATA-52129",
                "must share experiment and stage",
            ),
            (
                "V1-DATA-51883",
                "retry_of_attempt_id",
                "V1-SFT-51891",
                "must share experiment and stage",
            ),
        )
        for attempt_id, field, value, message in invalid_references:
            with self.subTest(attempt_id=attempt_id, field=field, value=value):
                payload = copy.deepcopy(self.registry)
                _attempt(payload, attempt_id)[field] = value
                self._assert_registry_rejected(payload, message)

    def test_retry_successor_cardinality_contract_is_enforced(self):
        payload = copy.deepcopy(self.registry)
        _attempt(payload, "V1-DATA-51876")["lineage_status"] = "canonical"
        self._assert_registry_rejected(payload, "canonical attempt.*retry successor")

        payload = copy.deepcopy(self.registry)
        _attempt(payload, "BASE-E2E5-51731")["lineage_status"] = "superseded"
        self._assert_registry_rejected(
            payload, "superseded attempt.*exactly one registered retry successor"
        )

        payload = copy.deepcopy(self.registry)
        duplicate_successor = copy.deepcopy(_attempt(payload, "V1-DATA-51883"))
        duplicate_successor["attempt_id"] = "V1-DATA-51884"
        duplicate_successor["Job ID"] = "51884"
        _sheet(payload, "04_运行尝试")["rows"].append(duplicate_successor)
        self._assert_registry_rejected(
            payload, "superseded attempt.*exactly one registered retry successor"
        )

    def test_dependency_retry_and_mixed_cycles_are_rejected(self):
        payloads = []

        dependency_cycle = copy.deepcopy(self.registry)
        _attempt(dependency_cycle, "BASE-E2E3-51727")[
            "depends_on_attempt_ids"
        ] = '["BASE-E2E5-51731"]'
        payloads.append(("dependency", dependency_cycle))

        retry_cycle = copy.deepcopy(self.registry)
        _attempt(retry_cycle, "V1-DATA-51876")[
            "retry_of_attempt_id"
        ] = "V1-DATA-51883"
        payloads.append(("retry", retry_cycle))

        mixed_cycle = copy.deepcopy(self.registry)
        _attempt(mixed_cycle, "V1-DATA-51876")[
            "depends_on_attempt_ids"
        ] = '["V1-DATA-51883"]'
        payloads.append(("mixed", mixed_cycle))

        for edge_type, payload in payloads:
            with self.subTest(edge_type=edge_type):
                self._assert_registry_rejected(
                    payload, "attempt lineage/dependency cycle"
                )

    def test_real_registry_and_external_evidence_validate(self):
        self.assertEqual(self.builder.load_registry(), self.registry)
        self.builder.validate_external_evidence(self.registry)

    def test_tracked_workbook_exactly_matches_the_registry(self):
        self.assertTrue(WORKBOOK_PATH.is_file(), f"missing tracked {WORKBOOK_PATH}")
        with zipfile.ZipFile(WORKBOOK_PATH) as archive:
            workbook = ET.fromstring(archive.read("xl/workbook.xml"))
            workbook_sheet_names = tuple(
                node.attrib["name"]
                for node in workbook.findall("m:sheets/m:sheet", XML_NAMESPACE)
            )
            self.assertEqual(workbook_sheet_names, EXPECTED_SHEETS)

            for index, name in enumerate(EXPECTED_SHEETS, start=1):
                with self.subTest(sheet=name):
                    matrix = _worksheet_matrix(archive, index)
                    self.assertEqual(tuple(matrix[2]), EXPECTED_HEADERS[name])
                    headers = EXPECTED_HEADERS[name]
                    expected_rows = [
                        [row[header] for header in headers]
                        for row in _sheet(self.registry, name)["rows"]
                    ]
                    self.assertEqual(matrix[3:], expected_rows)

        self.builder.validate_workbook(WORKBOOK_PATH, self.registry)
        self.builder.validate_workbook_matches_source(WORKBOOK_PATH, self.registry)


if __name__ == "__main__":
    unittest.main()
