"""Print the markdown tables used in results/*.md from the saved experiment outputs.

Usage: python report_tables.py exp1|exp2|exp3 [--tag qwen35_4b]
"""
from __future__ import annotations

import argparse
import json

from common import DATASETS, DISPLAY as DS, OUT, PROB_DETECTORS, detectors

NAMES = {"schanger_small": "SChanger-small", "schanger_base": "SChanger-base", "sam3_omniovcd": "SAM3+OmniOVCD",
         "mambascd_base": "MambaSCD-Base"}


def ready(exp, ds, file):
    return (OUT / exp / ds / file).exists()


def exp1_done(ds):
    return all((OUT / "exp1" / ds / det / "summary.json").exists() for det in detectors(ds))


def pct(x, digits=1):
    return "–" if x is None else f"{100 * x:.{digits}f}"


def exp1():
    print("### 像素指标与误差像素构成（阈值 0.5）\n")
    print("| 数据集 | 检测器 | F1 | P | R | 整目标漏检 | 部分命中 | 整目标误检 | 边界误差 |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for ds in [d for d in DATASETS if exp1_done(d)]:
        for det in detectors(ds):
            s = json.load(open(OUT / "exp1" / ds / det / "summary.json"))
            p, e = s["pixel"], s["error_share"]
            print(f"| {DS[ds]} | {NAMES[det]} | {pct(p['f1'], 2)} | {pct(p['precision'], 2)} | {pct(p['recall'], 2)} | "
                  f"{pct(e['miss'])}% | {pct(e['partial'])}% | {pct(e['fp'])}% | {pct(e['boundary'])}% |")
    print("\n### 目标级计数\n")
    print("| 数据集 | 检测器 | 真值目标 | 命中 | 部分命中 | 整目标漏检 | 漏检中零覆盖占比 | 部分命中中覆盖≥50%占比 | 预测目标 | 整目标误检 | 无信号漏检（覆盖<10%）目标数（占真值像素） |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for ds in [d for d in DATASETS if exp1_done(d)]:
        for det in detectors(ds):
            s = json.load(open(OUT / "exp1" / ds / det / "summary.json"))
            c = s["gt_object_counts"]
            ns = s.get("no_signal_miss")
            nstxt = f" | {ns['objects']}（{pct(ns['pixel_share_of_gt'])}%）" if ns else " | –"
            print(f"| {DS[ds]} | {NAMES[det]} | {s['gt_objects']} | {c['hit']} | {c['partial']} | {c['miss']} | "
                  f"{pct(s['miss_without_any_overlap'], 0)}% | {pct(s['partial_gt_coverage_ge_0.5'], 0)}% | "
                  f"{s['pred_objects']} | {s['fp_objects']}{nstxt} |")
    print("\n### 按真值目标面积（像素）分桶\n")
    print("| 数据集 | 检测器 | <100：目标数 / 漏检率 / 部分命中率 | 100–1000 | >1000 |")
    print("|---|---|---|---|---|")
    for ds in [d for d in DATASETS if exp1_done(d)]:
        for det in detectors(ds):
            b = json.load(open(OUT / "exp1" / ds / det / "summary.json"))["area_buckets"]
            cells = [f"{v['gt_objects']} / {pct(v['miss_rate'])}% / {pct(v['partial_rate'])}%" for v in b.values()]
            print(f"| {DS[ds]} | {NAMES[det]} | " + " | ".join(cells) + " |")
    print("\n### 阈值扫描（有概率图的检测器）\n")
    print("| 数据集 | 检测器 | 阈值 | F1 | P | R | 整目标漏检 | 0.5 时漏检被找回数（其中成为命中） | 整目标误检 |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for ds in [d for d in DATASETS if exp1_done(d)]:
        for det in [d for d in detectors(ds) if d in PROB_DETECTORS]:
            for r in json.load(open(OUT / "exp1" / ds / det / "sweep.json")):
                p = r["pixel"]
                print(f"| {DS[ds]} | {NAMES[det]} | {r['threshold']} | {pct(p['f1'], 2)} | {pct(p['precision'], 2)} | "
                      f"{pct(p['recall'], 2)} | {r['gt_object_counts']['miss']} | "
                      f"{r['recovered_misses']}/{r['base_misses']} ({r['recovered_misses_as_hit']}) | {r['fp_objects']} |")
    for ds in [d for d in DATASETS if exp1_done(d)]:
        for det in detectors(ds):
            t = json.load(open(OUT / "exp1" / ds / det / "summary.json")).get("by_transition")
            if not t:
                continue
            print(f"\n### {DS[ds]} {NAMES[det]}：按变化类型（t1 类别 -> t2 类别），按像素数排序\n")
            print("| 变化类型 | 真值目标 | 像素 | IoU 口径漏检率（目标） | 部分命中率 | 无信号漏检像素占该类型（覆盖<10%） |")
            print("|---|---:|---:|---:|---:|---:|")
            for key, v in sorted(t.items(), key=lambda kv: -kv[1]["pixels"]):
                print(f"| {key} | {v['gt_objects']} | {v['pixels']} | {pct(v['miss'] / v['gt_objects'])}% | "
                      f"{pct(v['partial'] / v['gt_objects'])}% | {pct(v.get('no_signal_pixels', 0) / v['pixels'])}% |")


def exp2():
    settings = ("original", "2a", "2b", "both", "2a_select", "both_select", "2c", "2a_strict")
    for variant, title in (("box", "框提示"), ("boxpt", "框 + 1 个正点")):
        print(f"\n### F1（%），SAM3 {title}\n")
        print("| 数据集 | 检测器 | " + " | ".join(settings) + " |")
        print("|---|---|" + "---:|" * len(settings))
        for ds in [d for d in DATASETS if ready("exp2", d, "metrics.json")]:
            m = json.load(open(OUT / "exp2" / ds / "metrics.json"))["pixel"]
            for det in detectors(ds):
                cells = [pct(m[f"{det}|{variant}|{s}"]["f1"], 2) for s in settings]
                print(f"| {DS[ds]} | {NAMES[det]} | " + " | ".join(cells) + " |")
    print("\n### SAM3 oracle 掩码质量（框提示）\n")
    print("| 数据集 | 全部真值目标平均 IoU | <100 | 100–1000 | >1000 | 检测器 | 命中目标：检测器 IoU / SAM3 IoU / SAM3 更好占比 | 漏检目标：SAM3 平均 IoU / IoU≥0.5 占比 |")
    print("|---|---:|---:|---:|---:|---|---|---|")
    for ds in [d for d in DATASETS if ready("exp2", d, "metrics.json")]:
        q = json.load(open(OUT / "exp2" / ds / "metrics.json"))["sam_quality"]["box"]
        for i, det in enumerate(detectors(ds)):
            d = q[det]
            head = (f"| {DS[ds]} | {q['all_mean_iou']:.3f} | {q['mean_iou_<100']:.3f} | {q['mean_iou_100-1000']:.3f} | "
                    f"{q['mean_iou_>1000']:.3f} ") if i == 0 else "| | | | | "
            print(head + f"| {NAMES[det]} | {d['hit_det_mean_iou']:.3f} / {d['hit_sam_mean_iou']:.3f} / "
                  f"{pct(d['hit_sam_better_frac'], 0)}% | {d['miss_sam_mean_iou']:.3f} / {pct(d['miss_sam_iou_ge_0.5_frac'], 0)}% |")


def exp3(tag):
    for ds in DATASETS:
        path = OUT / "exp3" / ds / f"analysis_{tag}.json"
        if not path.exists():
            continue
        a = json.load(open(path))
        print(f"\n### {DS[ds]}（{a['n']} 个候选，{a['parsed']} 个可解析，yes/no 词元平均概率质量 {'–' if a['mean_yes_no_mass'] is None else format(a['mean_yes_no_mass'], '.3f')}）\n")
        print("| 子集 | n（正样本） | MLLM P(yes) AUC [95% CI] | MLLM 口头置信度 AUC | 检测器平均概率 AUC [95% CI] | MLLM 回答准确率 | 检测器 >0.5 准确率 | 各来源 MLLM 答“是”比例 |")
        print("|---|---|---|---|---|---:|---:|---|")
        for name, e in a["subsets"].items():
            def cell(key):
                v = e[key]
                if v["auc"] is None:
                    return "–"
                ci = v["auc_ci95"]
                return f"{v['auc']:.3f}" + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "")
            rates = ", ".join(f"{k} {pct(v, 0)}%" for k, v in e["per_source_yes_rate"].items())
            print(f"| {name} | {e['n']} ({e['positives']}) | {cell('mllm_p_yes')} | {e['mllm_verbal']['auc']:.3f} | "
                  f"{cell('detector')} | {pct(e['acc_mllm_answer'])}% | {pct(e['acc_detector_0.5'])}% | {rates} |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("which", choices=("exp1", "exp2", "exp3"))
    parser.add_argument("--tag", default="qwen35_4b")
    args = parser.parse_args()
    {"exp1": exp1, "exp2": exp2, "exp3": lambda: exp3(args.tag)}[args.which]()
