#!/usr/bin/env python3
"""Print every session's showcase criteria and failure candidates; never select one."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True
if __package__:
    from .make_demo import read_json, read_steps, metric_f1, pct
else:
    from make_demo import read_json, read_steps, metric_f1, pct

GROUPS = {
    '看': {'view', 'inspect'},
    '分割': {'segment_text', 'segment_prompt'},
    '比较或语义': {'match_instances', 'ask'},
    # Match the repository's summarize.MODIFY definition.
    '修改': {'combine', 'morph', 'remove_region', 'add_region', 'refine_region'},
}


def mark(value):
    return '未知' if value is None else '是' if value else '否'


def md(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def delta_text(value):
    return '未知' if value is None else f'{value * 100:+.2f} pp'


def inspect_session(directory, summary, init_f1_from_final=False):
    rows = read_steps(directory)
    final_path = directory / 'final.json'
    final = read_json(final_path) if final_path.exists() else {}
    input_path = directory / 'input.json'
    info = read_json(input_path) if input_path.exists() else {}
    task = final.get('task_id', info.get('task_id', directory.parent.parent.name))
    all_calls = [c for row in rows for c in row.get('calls', [])]
    calls = [c for c in all_calls if c.get('ok') and c.get('status') != 'skipped_due_to_submit']
    used = {c['tool'] for c in calls}
    categories = [label for label, tools in GROUPS.items() if used & tools]
    seen_view = False
    improved_after_view = False
    for call in calls:
        if call['tool'] in GROUPS['修改'] and seen_view:
            delta = next((c.get('delta_f1_from_parent') for c in call.get('candidate_eval') or [] if c.get('delta_f1_from_parent') is not None), (call.get('eval') or {}).get('delta_f1'))
            if delta is not None and delta > 0:
                improved_after_view = True
        if call['tool'] == 'view':
            seen_view = True
    normal = (final.get('submitted_by') == 'model' and final.get('termination_reason') == 'submit'
              and any(c['tool'] == 'submit' for c in calls)) if final else None
    final_metric = final.get('eval') or {}
    f1 = metric_f1(final_metric)
    count = final.get('planning_rounds', sum(row['step'] > 0 for row in rows))
    initial = next((metric_f1(row.get('eval')) for row in rows if row['step'] == 0), None)
    gt_pixels = final_metric.get('tp', 0) + final_metric.get('fn', 0) if 'tp' in final_metric and 'fn' in final_metric else None
    if task in ('T5.1', 'T5.1xC1'):
        if init_f1_from_final:
            initial = metric_f1(final.get('initial_eval'))
    else:
        initial = (1.0 if gt_pixels == 0 else 0.0) if gt_pixels is not None else None
    delta = f1 - initial if f1 is not None and initial is not None else None
    aggregate = summary.get('f1') if summary else None
    above = f1 >= aggregate if f1 is not None and aggregate is not None else None
    checks = [normal, len(categories) >= 3, improved_after_view, above, 8 <= count <= 20]
    qualifies = all(c is True for c in checks)
    reason_names = ['正常 submit', '至少三类工具', '看图后修改 ΔF1>0', '达到整集 F1ᶜ', '8–20 步']
    missing = [name + ('未知' if check is None else '未满足') for name, check in zip(reason_names, checks) if check is not True]
    return dict(name=directory.name, task=task, normal=normal, categories=categories,
                improved=improved_after_view, above=above, f1=f1, initial=initial, delta=delta,
                steps=count, in_range=8 <= count <= 20, qualifies=qualifies, reasons=missing,
                gt_pixels=gt_pixels, errors=sum(not c.get('ok', False) for c in all_calls))


def quartile(values, fraction=.25):
    """Linear interpolation, same convention as numpy's default percentile."""
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def report(config_dir, init_f1_from_final=False):
    directory = Path(config_dir)
    path = directory / 'summary.json'
    summary = read_json(path) if path.exists() else None
    paths = sorted(p for p in directory.iterdir() if p.is_dir() and any((p / f).exists() for f in ('input.json', 'steps.jsonl', 'final.json')))
    if not paths:
        raise ValueError('没有发现轨迹目录。')
    records = [inspect_session(p, summary, init_f1_from_final) for p in paths]
    records.sort(key=lambda r: (not r['qualifies'], -(r['delta'] if r['delta'] is not None else -float('inf')), r['name']))
    lines = ['# 展示候选检查（不自动选定）', '',
             '这是挑出来的展示样例，不代表平均水平。以下是所有已发现图像的筛选依据。', '',
             f'发现 {len(records)} 张轨迹；目标为完整 64 张。' + ('已覆盖 64 张目录。' if len(records) == 64 else '当前输入不完整，未补造缺失图像。')]
    if summary:
        lines += [f'汇总记录 {summary.get("images", "未知")} 张，整集 F1ᶜ {pct(summary.get("f1"))}，逐图平均 F1 {pct(summary.get("mean_image_f1"))}。']
        if summary.get('images') != len(records):
            lines += ['汇总图数与当前目录数不同，门槛按提供的 summary.json，完整性需另行核对。']
    else:
        lines += ['缺少 summary.json：整集 F1ᶜ 门槛未知，无法确认符合全部条件。']
    lines += ['', '类别按成功且实际执行的调用统计：看 / 分割 / 比较或语义 / 修改；修改含 combine、morph、remove_region、add_region、refine_region。',
              '“看图后修改”是调用先后顺序与记录器 ΔF1 的可核验证据，不推断主观意图。',
              'ΔF1：T5.1 为最终减初始；T1.1 为最终减空预测起点（空 GT 起点为 1，其余为 0）。pp 为百分点。',
              f'T5.1 初始 F1 来源：{"final.initial_eval" if init_f1_from_final else "第 0 步 eval"}；缺失时标未知。', '',
              '| 图像 | 符合全部 | 正常 submit | 类别数及名称 | 看后修改且提升 | 最终 F1 | ≥整集 F1ᶜ | 步数 | 8–20 | 初始/空预测 F1 | ΔF1 | 失败调用 | 未满足/未知 |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for row in records:
        cells = [row['name'], mark(row['qualifies']), mark(row['normal']),
                 f'{len(row["categories"])}：' + '、'.join(row['categories']), mark(row['improved']),
                 pct(row['f1']), mark(row['above']), row['steps'], mark(row['in_range']),
                 pct(row['initial']), delta_text(row['delta']), row['errors'], '；'.join(row['reasons']) or '全部满足']
        lines.append('| ' + ' | '.join(md(cell) for cell in cells) + ' |')
    lines += ['', '## 典型失败候选（供人工检查，不自动选定）', '']
    t1 = [r for r in records if r['task'].startswith('T1.1') and r['f1'] is not None]
    t5 = [r for r in records if r['task'] in ('T5.1', 'T5.1xC1') and r['delta'] is not None]
    if t1:
        values = list((summary or {}).get('per_image_f1', {}).values()) or [r['f1'] for r in t1]
        q1 = quartile(values)
        population = 'summary 逐图指标' if (summary or {}).get('per_image_f1') else '当前可用轨迹（非完整整集）'
        lines += [f'T1.1：{population}的 F1 下四分位为 {pct(q1)}；排除空 GT 后按距下四分位的绝对差取最多 3 张。', '']
        candidates = sorted((r for r in t1 if r['gt_pixels'] is not None and r['gt_pixels'] > 0), key=lambda r: (abs(r['f1'] - q1), r['name']))[:3]
        lines += [f'- {r["name"]}：F1 {pct(r["f1"])}，GT {r["gt_pixels"]} 像素，正常 submit：{mark(r["normal"])}。' for r in candidates] or ['- 没有可核验的非空 GT 候选。']
    if any(r['task'] in ('T5.1', 'T5.1xC1') for r in records):
        lines += ['T5.1：最终比初始变差最多的最多 3 张（仅列 ΔF1<0）。', '']
        worse = sorted((r for r in t5 if r['delta'] < 0), key=lambda r: (r['delta'], r['name']))[:3]
        lines += [f'- {r["name"]}：{pct(r["initial"])} → {pct(r["f1"])}，ΔF1 {delta_text(r["delta"])}。' for r in worse] or ['- 当前有初始指标的轨迹中没有变差案例。']
        unknown = sum(r['task'] in ('T5.1', 'T5.1xC1') and r['delta'] is None for r in records)
        if unknown:
            lines += [f'- {unknown} 张缺初始或最终指标，无法判断是否变差。']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-dir', type=Path, required=True)
    parser.add_argument('--init-f1-from-final', action='store_true', help='Use final.initial_eval instead of step 0 eval.')
    parser.add_argument('--out', type=Path, help='Also write the exact printed Markdown report.')
    args = parser.parse_args()
    result = report(args.config_dir, args.init_f1_from_final)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result, encoding='utf-8')
    print(result, end='')


if __name__ == '__main__':
    main()
