"""T3.1 dev-only routes. report is metadata-only; A/prepare/B belong in Slurm.

No command calls an API except the explicit `verify` command. See README_t3_routes.md.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .io_retry import read_text, read_image, write_json, write_text, retry_io, exists
from .probe_t13_common import CLASSES
from .second_labels import ENCODING
from .t3_subset import pick

GROUPS = ('目标类', '地表类')
GRID = (4, 8, 16, 32)
TASKS = Path('results/agent_baseline/data/t3_tasks_dev_0of1/tasks.jsonl')


def pair_id(row):
    pair = tuple(map(int, row['id'].rsplit('_', 1)[-1].split('to')))
    if len(pair) != 2 or any(c not in CLASSES for c in pair) or pair[0] == pair[1]:
        raise ValueError('Require a directed SECOND class transition')
    return pair


def group(row):
    return GROUPS[0] if set(pair_id(row)) & {5, 6} else GROUPS[1]


def load_dev(run):
    rows = [json.loads(line) for line in read_text(run / TASKS).splitlines() if line.strip()]
    rows, _ = pick(rows)
    names = {Path(r['image_t1']).name for r in rows}
    expected = set(read_text(run / 'second_dev/names.txt').split())
    if len(rows) != 60 or names != expected or any(r['task_type'] != 'T3.1' for r in rows):
        raise ValueError('Only the fixed 60-image T3.1 development subset is allowed')
    dataset = Path(rows[0]['image_t1']).parent.parent
    if json.loads(read_text(dataset / 'cls_gray.json')) != ENCODING:
        raise ValueError('SECOND label encoding differs')
    return rows


def pooled(rows):
    tp, fp, fn = map(int, np.asarray([r['counts'] for r in rows], dtype=np.int64).reshape(-1, 3).sum(axis=0))
    return dict(f1_c=200*tp/max(1,2*tp+fp+fn), precision=100*tp/max(1,tp+fp),
                recall=100*tp/max(1,tp+fn), tp=tp, fp=fp, fn=fn)


def pixel_counts(pred, truth, exemplar):
    from .second_labels import pixel_counts as counts
    return counts(np.asarray(pred, bool) & ~exemplar, np.asarray(truth, bool) & ~exemplar)


def breakdown(rows, diagnostic):
    by_id = {r['id']: r for r in diagnostic['tasks']}
    if len(by_id) != len(rows) or set(by_id) != {r['id'] for r in rows}:
        raise ValueError('Diagnostic must cover exactly the fixed dev subset')
    result = {}
    for name in ('全部', *GROUPS):
        chosen = [r for r in rows if name == '全部' or group(r) == name]
        evidence = [by_id[r['id']]['methods']['sam_dino'] for r in chosen]
        totals = {k: sum(e['proposals'][k] for e in evidence) for k in
                  ('gt_instances', 'aligned_instances', 'truth_pixels', 'covered_pixels', 'proposals')}
        totals.update(instance_recall_upper=totals['aligned_instances']/totals['gt_instances'] if totals['gt_instances'] else None,
                      pixel_coverage_upper=totals['covered_pixels']/totals['truth_pixels'] if totals['truth_pixels'] else None)
        result[name] = dict(tasks=len(chosen), fixed=pooled([e['scores']['0.6'] for e in evidence]), proposals=totals)
    return result


def semantic_route(probabilities, exemplar):
    """Closed six-class exemplar decision, with background retained for dense maps.

    Argmax of mean probability over exemplar pixels (not majority pixel labels).
    Class 0 is a worker background channel, not a candidate SECOND transition.
    No ground truth or transition metadata enters this function.
    """
    exemplar = np.asarray(exemplar, bool)
    if not exemplar.any():
        raise ValueError('Empty exemplar')
    maps, pair = [], []
    for p in probabilities:
        p = np.asarray(p)
        if p.shape != (7, *exemplar.shape) or not np.isfinite(p).all():
            raise ValueError('Expected background plus six aligned probability maps')
        pair.append(int(np.argmax(p[1:, exemplar].mean(axis=1))) + 1)
        maps.append(p.argmax(axis=0))
    source, target = pair
    prediction = (maps[0] == source) & (maps[1] == target)
    if source == target:
        prediction[:] = False
    return pair, prediction & ~exemplar


def evaluate_a(row, backend):
    exemplar = read_image(row['exemplar_mask']) > 0
    pair, pred = semantic_route([backend('classify', time=t, classes=list(CLASSES.values())) for t in ('t1', 't2')], exemplar)
    # Truth is used only after predictions are fixed.
    truth = read_image(row['gt_mask']) > 0
    return dict(id=row['id'], predicted_pair=pair, true_pair=list(pair_id(row)),
                pair_correct=tuple(pair) == pair_id(row), counts=pixel_counts(pred, truth, exemplar))


def summarize_a(rows, tasks):
    by_id = {t['id']: t for t in tasks}
    result = {}
    for name in ('全部', *GROUPS):
        selected = [by_id[r['id']] for r in rows if r['id'] in by_id and (name == '全部' or group(r) == name)]
        result[name] = dict(tasks=len(selected), correct=sum(t['pair_correct'] for t in selected),
            pair_accuracy=sum(t['pair_correct'] for t in selected)/len(selected) if selected else None,
            score=pooled(selected))
    return result


def ledger_estimate(path):
    """Four 384px crops in f37_corr; route B uses eight crops and a new prompt."""
    from agent_baseline.planner import request_cost, FLASH_TIERS
    rows = [json.loads(line) for line in read_text(path).splitlines() if line.strip()]
    tokens = [r['tokens'] for r in rows if r.get('tokens') and r.get('answer') in ('yes', 'no')]
    if not tokens:
        raise ValueError('No usable ask ledger token evidence')
    mean_in = float(np.mean([t['input'] for t in tokens]))
    mean_out = float(np.mean([t['output'] for t in tokens]))
    # Twice the four-view input, plus 512 tokens for the changed question.
    inp = int(np.ceil(2*mean_in + 512))
    output = max(128, int(np.ceil(mean_out)))
    nominal = request_cost('qwen3.7-flash', dict(input=inp, cached=0, output=output))
    upper = request_cost('qwen3.7-flash', dict(input=2*inp, cached=0, output=256))
    return dict(samples=len(tokens), mean_ledger_input=mean_in, mean_ledger_output=mean_out,
        model='qwen3.7-flash', prices_cny_per_million=[list(t) for t in FLASH_TIERS[:2]],
        estimated_input=inp, estimated_output=output, per_request_cny=nominal,
        conservative_per_request_cny=upper,
        assumption='八张384像素图；输入=四图ledger均值×2+512；保守输入再×2，输出256；缓存按全价；不是账单或硬上限')


def candidate_estimate(rows, diagnostic, unit, grid=GRID):
    by_id = {t['id']: t for t in diagnostic['tasks']}
    result = {}
    for k in grid:
        counts = {r['id']: min(k, by_id[r['id']]['methods']['sam_dino']['proposals']['proposals']) for r in rows}
        total = sum(counts.values())
        result[str(k)] = dict(candidates=total, requests=total, per_task=counts,
            estimated_cost_cny=total*unit['per_request_cny'],
            conservative_cost_cny=total*unit['conservative_per_request_cny'], score=None,
            groups={g: sum(counts[r['id']] for r in rows if group(r) == g) for g in GROUPS})
    maximum = max(counts['requests'] for counts in result.values())
    return dict(grid=result, unique_requests_all_k=maximum,
        unique_cost_cny=maximum*unit['per_request_cny'],
        unique_conservative_cost_cny=maximum*unit['conservative_per_request_cny'], selected_k=None,
        counting='每任务两期全部SAM提议；含示例提议和重复提议；按分数稳定排序取前k，无0.6过滤；跨k复用同一候选答案')


def fmt(value):
    return '待运行' if value is None else f'{value:.2f}'


def render(report):
    lines = ['# T3 开发诊断', '', '仅开发60任务。F1/P/R为累计像素计数的百分数。示例从预测、真值、提议上限中移除。', '',
             '| 分组 | 任务 | 固定F1 | P | R | 实例对齐召回上限 | 像素覆盖上限 |', '|---|---:|---:|---:|---:|---:|---:|']
    for g, r in report['breakdown'].items():
        s, p = r['fixed'], r['proposals']
        lines.append(f"| {g} | {r['tasks']} | {s['f1_c']:.2f} | {s['precision']:.2f} | {s['recall']:.2f} | {fmt(None if p['instance_recall_upper'] is None else 100*p['instance_recall_upper'])} | {fmt(None if p['pixel_coverage_upper'] is None else 100*p['pixel_coverage_upper'])} |")
    lines += ['', report['conclusion'], '', '路线A：示例区域六类平均概率取最大，再求两期有向语义交集。',
              '这是Agent可自主选择的已有工具组合，可把有名称的示例转成T1任务。未将它写入T3的Agent流程。', '',
              '| 路线A分组 | 已完成任务 | 类别对准确率 | F1 | P | R |', '|---|---:|---:|---:|---:|---:|']
    for g in ('全部', *GROUPS):
        r = report['route_a']['groups'].get(g)
        cells = [str(r['tasks']), fmt(100*r['pair_accuracy']) if r['pair_accuracy'] is not None else '—', *(fmt(r['score'][k]) for k in ('f1_c','precision','recall'))] if r else ['0', '待GPU', '待GPU', '待GPU', '待GPU']
        lines.append('| ' + ' | '.join([g, *cells]) + ' |')
    lines += ['', f"路线A状态：{report['route_a']['status']}。", '',
              '| k | 候选数 | 请求数 | 估计元 | 保守估计元 | 路线B F1/P/R |', '|---|---:|---:|---:|---:|---|']
    for k, r in report['route_b']['grid'].items():
        score = r['score']
        text = '待批准后运行' if score is None else '/'.join(fmt(score[x]) for x in ('f1_c','precision','recall'))
        lines.append(f"| {k} | {r['candidates']} | {r['requests']} | {r['estimated_cost_cny']:.4f} | {r['conservative_cost_cny']:.4f} | {text} |")
    b = report['route_b']
    lines += ['', f"跨k复用答案：共{b['unique_requests_all_k']}次；估计{b['unique_cost_cny']:.4f}元，保守{b['unique_conservative_cost_cny']:.4f}元。",
              report['pricing']['assumption'], 'k只在开发集按F1选，平局取小k。路线B分数待完整核验后计算。' if b['selected_k'] is None else f"开发集选择k={b['selected_k']}。", '']
    agent = report.get('agent10', {})
    lines += ['补充Agent试跑：开发60任务中按种子1016抽10任务，清单为results/agent10_tasks.jsonl，不改提示。',
              '同时开放match_exemplar与classify/segment_semantic。调用率、示例类别判断证据和排除示例的F1由离线脚本统计。',
              f"Agent10状态：{agent.get('status', '待Claude试跑')}。", '']
    return '\n'.join(lines)


def make_report(run, out, ledger):
    rows = load_dev(run)
    diagnostic = json.loads(read_text(run / 't3_diag/report.json'))
    groups = breakdown(rows, diagnostic)
    unit = ledger_estimate(ledger)
    a_path = out / 'route_a/progress.json'
    a = json.loads(read_text(a_path)) if exists(a_path) else None
    if a is not None and a['policy']['task_ids'] != [r['id'] for r in rows]:
        raise ValueError('Route A subset differs')
    report = dict(status='等待GPU与API决定', tasks=len(rows), label_encoding=ENCODING,
        inputs=dict(tasks=str(TASKS), diagnostic='t3_diag/report.json', cache='t3_diag/cache', ledger=str(ledger.relative_to(run))),
        breakdown=groups, route_a=dict(status='待GPU', groups={}) if a is None else
            dict(status='完成' if len(a['tasks']) == len(rows) else '部分完成', groups=summarize_a(rows, a['tasks'])),
        route_b=candidate_estimate(rows, diagnostic, unit), pricing=unit)
    target, surface = groups[GROUPS[0]], groups[GROUPS[1]]
    report['conclusion'] = (f"目标类固定F1={target['fixed']['f1_c']:.2f}，地表类={surface['fixed']['f1_c']:.2f}。"
        '两组都未解决示例匹配，不能说只在地表类失效。提议覆盖与实例对齐的差距也含提议粒度和误检，不能全部归因于DINO。没有运行Agent，不能断言其路线选择的效果。')
    b_path = out / 'route_b/scores.json'
    if exists(b_path):
        b = json.loads(read_text(b_path))
        if b['task_ids'] != [r['id'] for r in rows]:
            raise ValueError('Route B subset differs')
        for k, score in b['scores'].items():
            report['route_b']['grid'][k].update(score=score['全部']['score'], score_groups=score)
        report['route_b']['selected_k'] = b['selected_k']
    agent_path = out / 'agent10_metrics.json'
    report['agent10'] = dict(status='待Claude试跑', tasks=10, seed=1016, manifest='results/agent10_tasks.jsonl', metrics=None)
    if exists(agent_path):
        report['agent10'].update(status='已有轨迹统计，待复核', metrics=json.loads(read_text(agent_path)))
    if a is not None and len(a['tasks']) == 60 and report['route_b']['selected_k'] is not None:
        report['status'] = '开发两路线完成，待审核'
    write_json(out / 't3_routes.json', report)
    write_text(out / 't3_routes.md', render(report))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['report', 'route-a', 'prepare-b', 'verify', 'score-b'])
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, default=Path('results'))
    p.add_argument('--ledger', type=Path, help='Existing flash four-view ask ledger, relative to R')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--concurrency', type=int, default=2)
    p.add_argument('--max-cost-cny', type=float, default=5.)
    p.add_argument('--ask-variant', choices=['default', 't3_transition'], default='default')
    a = p.parse_args()
    a.run_root = a.run_root.resolve()
    a.output_dir = a.output_dir.resolve()
    if a.command == 'report':
        make_report(a.run_root, a.output_dir, a.run_root / (a.ledger or Path('ask_eval/dev2/f37_corr.jsonl')))
    else:
        from .t3_routes_runtime import run_command
        run_command(a)


if __name__ == '__main__':
    main()
