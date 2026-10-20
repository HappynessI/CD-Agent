"""Collect completed V1 runs into Chinese Markdown tables; no model inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from exp6b_common import BACKBONES, ROOT

DATASET_NAMES = {"second_cv": "SECOND 验证集", "second": "SECOND 测试集",
                 "sysu": "SYSU 测试集", "levir": "LEVIR 测试集", "whu": "WHU 测试集"}
METHOD_NAMES = {
    "baseline": "原始检测器", "recall/B+G0.05": "补漏 B+G0.05",
    "recall/B+G0.05+H1": "补漏 B+G0.05+H1", "recall/all": "补漏全部来源",
    "remove/verifier": "验证器去误检", "remove/detector_mean": "检测器均值去误检",
    "combined/verifier": "补漏+验证器去误检", "combined/detector_mean": "补漏+检测器均值去误检",
}


def number(value, percent=False, signed=False):
    if value is None:
        return "—"
    value *= 100 if percent else 1
    return f"{value:+.2f}" if signed else f"{value:.2f}" if percent else f"{value:.4f}"


def table(headers, rows):
    return (["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
            + ["| " + " | ".join(map(str, row)) + " |" for row in rows] + [""])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+")
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASET_NAMES),
                        default=list(DATASET_NAMES))
    parser.add_argument("--output", type=Path, default=ROOT / "verifier/tables_6b_6e.md")
    args = parser.parse_args()
    base = ROOT / "verifier"
    candidates = [b + suffix for suffix in ("", "_pred") for b in BACKBONES]
    runs = args.runs or [r for r in candidates if (base / r / "val_metrics.json").exists()]
    if not runs:
        raise FileNotFoundError(f"No completed training reports under {base}")
    validations, evaluations, calibrations = {}, {}, {}
    for run in runs:
        validations[run] = json.loads((base / run / "val_metrics.json").read_text())
        path = base / run / "thresholds_second_cv.json"
        if path.exists():
            calibrations[run] = json.loads(path.read_text())
            if calibrations[run]["run_id"] != validations[run]["run_id"]:
                raise ValueError(f"{run}: threshold and training reports belong to different runs")
        for dataset in args.datasets:
            path = base / run / f"eval_{dataset}.json"
            if path.exists():
                result = json.loads(path.read_text())
                if result["run_id"] != validations[run]["run_id"]:
                    raise ValueError(f"{run}/{dataset}: evaluation belongs to a different run")
                evaluations[(run, dataset)] = result
    lines = [
        "# 实验 6b / 6e 汇总",
        "",
        "数值来自已完成的训练、验证阈值和评估 JSON；缺失结果标为“待运行”。"
        "F1、精度、召回、误检率、像素精度均为百分数，ΔF1 为百分点；AUC/AP 为 0–1。"
        "端到端阈值来自 SECOND 交叉拟合验证图像，像素指标按 micro 汇总。",
        "",
        "## 完成状态",
        "",
    ]
    lines += table(["模型", "阈值", *[DATASET_NAMES[d] for d in args.datasets]], [
        [run, "已冻结" if run in calibrations else "待运行",
         *["已完成" if (run, d) in evaluations else "待运行" for d in args.datasets]]
        for run in runs])
    lines += ["## 训练与验证", ""]
    lines += table(
        ["模型", "训练块数", "正例率", "正例权重", "最佳轮次", "早停用 AP", "补漏验证 AUC", "补漏验证 AP"],
        [[run, v["train"]["n"], number(v["train"]["positive_rate"], True),
          number(v["train"]["pos_weight"]), v["best_epoch"],
          number(v["selection_population"]["ap"]), number(v["recall"]["overall"]["auc"]),
          number(v["recall"]["overall"]["ap"])] for run, v in validations.items()])
    eligible = [r for r in runs if r in BACKBONES and r != "none"]
    if eligible:
        best = max(eligible, key=lambda r: validations[r]["recall"]["overall"]["ap"])
        lines += [f"已完成的未加入 pred 训练版本中，按补漏验证 AP 选择的骨干为 `{best}`。"
                  "缺失骨干完成后应重新比较；测试集结果不参与选择。", ""]
    lines += ["## 切块分类与召回 80% 的像素精度", ""]
    rows = []
    for (run, dataset), result in evaluations.items():
        metric = result["recall"]["overall"]
        precision = result["pixels"]["pixel_precision_at_recall_80"]
        rows.append([run, DATASET_NAMES[dataset], metric["n"], number(metric["auc"]),
                     number(metric["ap"]), number(metric["recall_80"]["fpr"], True),
                     number(metric["recall_90"]["fpr"], True),
                     number(precision["precision"], True), number(precision["f1_half"], True)])
    lines += table(["模型", "数据集", "有标签补漏块", "AUC", "AP", "80% 召回误检率",
                    "90% 召回误检率", "补入像素精度", "F1/2 门槛"], rows)
    lines += ["本表的召回 80% / 90% 阈值按当前数据集有标签块定义，仅用于诊断；"
              "像素并集包含通过阈值的全部补漏块，包括 label=null 的块。"
              "端到端结果使用下一表所对应的验证集固定阈值。", ""]
    lines += ["## 按来源的补漏分类", ""]
    rows = []
    for (run, dataset), result in evaluations.items():
        for source, metric in result["recall"]["by_source"].items():
            rows.append([run, DATASET_NAMES[dataset], source, metric["n"],
                         number(metric["auc"]), number(metric["ap"]),
                         number(metric["recall_80"]["fpr"], True),
                         number(metric["recall_90"]["fpr"], True)])
    lines += table(["模型", "数据集", "来源", "块数", "AUC", "AP",
                    "80% 召回误检率", "90% 召回误检率"], rows)
    lines += ["## 端到端补漏、去误检与合并", ""]
    rows = []
    object_rows = []
    for (run, dataset), result in evaluations.items():
        for key, label in METHOD_NAMES.items():
            metric = result["pixels"][key]
            rows.append([run, DATASET_NAMES[dataset], label, number(metric["f1"], True),
                         number(metric["precision"], True), number(metric["recall"], True),
                         number(metric["delta_f1"], True, True), metric["added_tp"],
                         metric["added_fp"], metric["removed_tp"], metric["removed_fp"]])
            if key == "baseline" or key.startswith(("recall/", "combined/")):
                recall = metric["target_recall"]
                object_rows.append([run, DATASET_NAMES[dataset], label,
                                    *[f'{v["hit"]}/{v["total"]} ({number(v["recall"], True)}%)'
                                      for v in (recall[k] for k in ("all", "<100", "100-1000", ">1000"))]])
    lines += table(["模型", "数据集", "方法", "F1", "精度", "召回", "ΔF1",
                    "新增 TP", "新增 FP", "删除 TP", "删除 FP"], rows)
    lines += ["合并使用验证集选择的补漏来源组合，补入范围限定为原预测外；"
              "删除与补漏阈值分别冻结。加入 pred 重训的版本以 `_pred` 标识。", ""]
    lines += table(["模型", "数据集", "合并使用来源"], [
        [run, DATASET_NAMES[dataset], result["pixels"]["combined_sources"]]
        for (run, dataset), result in evaluations.items()])
    lines += ["## 目标级召回", ""]
    lines += table(["模型", "数据集", "方法", "全部", "<100 像素", "100–999 像素", "≥1000 像素"],
                   object_rows)
    lines += ["按 `objects.gt_label` 拆分全部真值目标，预测覆盖至少 50% 视为召回；"
              "每格为命中数/目标总数（召回百分数）。面积桶沿用 `common.area_bucket`，"
              "旧字段 `>1000` 实际包含 1000。", ""]
    lines += ["## 验证集阈值与平台", ""]
    rows = []
    for run, calibration in calibrations.items():
        for operation in ("recall", "remove"):
            for key, curve in calibration[operation].items():
                interval = curve["contiguous_near_best_interval"]
                rows.append([run, f"{operation}/{key}", f'{curve["threshold"]:.10g}',
                             number(curve["delta_f1"], True, True),
                             f"[{interval[0]:.10g}, {interval[1]:.10g}]", curve["points"],
                             f'[数据]({curve["curve"]}) / [图]({curve["plot"]})'])
    lines += table(["模型", "操作", "阈值", "验证 ΔF1", "连续近最优区间", "曲线点数", "完整曲线"], rows)
    lines += ["近最优区间为包含最优阈值且 ΔF1 距最优不超过 0.1 个百分点的连续区间。"
              "曲线保留全部分数变化点；同一分数的切块同时通过或删除。", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp.md")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(args.output)
    print(f"saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
