"""Offline T5.2 pooled pixel scores, error repair, off-feedback edits and paired CIs."""
import argparse
import json
import os
from pathlib import Path
if __package__ in (None, ''):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import agent_baseline
import numpy as np
from scipy import ndimage as ndi
from common import components, confusion, load_mask_png, scores, write_json
from agent_baseline.tasks import read_tasks
from agent_baseline.analysis.make_t52_feedback import region_mask


def counts(pred, truth):
    c = confusion(pred, truth)
    return [c['tp'], c['fp'], c['fn']]


def score(rows):
    tp, fp, fn = np.asarray(rows, np.int64).sum(axis=0)
    c = dict(tp=int(tp), fp=int(fp), fn=int(fn))
    s = scores(c)
    return dict(c, f1_c=100 * s['f1'], precision=100 * s['precision'], recall=100 * s['recall'])


def repair_and_outside(final, initial, truth, feedback, radius=32):
    """Repair uses the full source component; the exclusion uses its 32px disk dilation."""
    final, initial, truth = [np.asarray(m, bool) for m in (final, initial, truth)]
    if not (final.shape == initial.shape == truth.shape):
        raise ValueError('Prediction, truth and given mask shapes differ')
    gl, _ = components(truth)
    il, _ = components(initial)
    union = np.zeros_like(initial)
    repaired = dict(missed=0, false_positive=0)
    total = dict(missed=0, false_positive=0)
    for item in feedback:
        region = region_mask(item, gl, il, initial, full=True)
        area = int(region.sum())
        if area == 0:
            raise ValueError(f'Feedback source component is absent: {item["region_id"]}')
        union |= region
        kept = int(final[region].sum())
        kind = item['type']
        repaired[kind] += int(2 * kept >= area if kind == 'missed' else 2 * kept < area)
        total[kind] += 1
    outside = ndi.distance_transform_edt(~union) > radius if union.any() else np.ones_like(union)
    return dict(repaired=repaired, pointed=total,
        outside_added_px=int((final & ~initial & outside).sum()),
        outside_removed_px=int((initial & ~final & outside).sum()),
        outside_pixels=int(outside.sum()),
        outside_final=counts(final[outside], truth[outside]),
        outside_given=counts(initial[outside], truth[outside]))


def summarize(image_counts, local):
    result = score(image_counts)
    repair = {}
    for kind in ('missed', 'false_positive', 'all'):
        keys = ('missed', 'false_positive') if kind == 'all' else (kind,)
        fixed = sum(r['repaired'][k] for r in local for k in keys)
        total = sum(r['pointed'][k] for r in local for k in keys)
        repair[kind] = dict(repaired=fixed, pointed=total, rate=fixed / total if total else None)
    result['repair'] = repair
    result['outside'] = dict(
        added_px=sum(r['outside_added_px'] for r in local),
        removed_px=sum(r['outside_removed_px'] for r in local),
        pixels=sum(r['outside_pixels'] for r in local),
        final=score([r['outside_final'] for r in local]),
        given=score([r['outside_given'] for r in local]))
    result['outside']['delta_f1'] = result['outside']['final']['f1_c'] - result['outside']['given']['f1_c']
    return result


def mean_scores(per_run):
    """Each run first pools pixels/regions, then the three run scores are averaged."""
    mean = {key: float(np.mean([r[key] for r in per_run])) for key in ('f1_c', 'precision', 'recall', 'delta_f1_given')}
    mean['repair'] = {}
    for kind in ('missed', 'false_positive', 'all'):
        values = [r['repair'][kind] for r in per_run]
        mean['repair'][kind] = {key: (float(np.mean([v[key] for v in values])) if values[0][key] is not None else None)
                                for key in ('repaired', 'pointed', 'rate')}
    mean['outside'] = {key: float(np.mean([r['outside'][key] for r in per_run]))
                       for key in ('added_px', 'removed_px', 'pixels', 'delta_f1')}
    return mean


def paired_bootstrap(agent, baseline, samples=4000, seed=1016):
    """Keep repeats fixed; share each paired image sample across all three repeats."""
    agent, baseline = np.asarray(agent), np.asarray(baseline)
    if agent.ndim != 3 or baseline.shape != agent.shape[1:]:
        raise ValueError('Expected (3, images, 3 counts) versus (images, 3 counts)')
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        idx = rng.integers(0, agent.shape[1], agent.shape[1])
        values.append(float(np.mean([score(a[idx])['f1_c'] for a in agent])) - score(baseline[idx])['f1_c'])
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def session_prediction(root, stem, shape):
    directory = Path(root) / stem
    final = json.loads((directory / 'final.json').read_text())
    if final.get('mask_path'):
        pred = load_mask_png(directory / final['mask_path'])
    elif final.get('submitted_by') == 'error' and final.get('fallback_kind') == 'empty':
        # loop_json's input-error final explicitly records this empty prediction.
        pred = np.zeros(shape, bool)
    else:
        raise ValueError(f'Missing final mask: {directory}')
    if pred.shape != shape:
        raise ValueError(f'Final mask shape differs: {directory}')
    return pred, dict(error=int(final['submitted_by'] == 'error' or bool(final.get('error'))),
        fallback=int(final['submitted_by'] != 'model'))


def evaluate_group(spec, samples=4000, seed=1016):
    rows = read_tasks(spec['manifest'], 'T5.2')
    if not rows or len({r['meta']['k'] for r in rows}) != 1:
        raise ValueError('Use one nonempty k manifest per group')
    if len({Path(r['id']).stem for r in rows}) != len(rows):
        raise ValueError('Duplicate image output stems')
    for key in ('agent_runs', 'g2_runs'):
        if len(spec[key]) != 3 or len(set(spec[key])) != 3:
            raise ValueError(f'{key} requires three distinct run directories')
    image_data = []
    for row in rows:
        truth = load_mask_png(row['gt_mask'])
        initial = load_mask_png(row['init_mask'])
        image_data.append((row, truth, initial))
    reports, arrays = {}, {}
    given_f1 = score([counts(i, g) for _, g, i in image_data])['f1_c']
    for method, roots in dict(given=[None], fixed=[spec['fixed_dir']],
                              g2_no_feedback=spec['g2_runs'], agent=spec['agent_runs']).items():
        per_run, repeated_counts, diagnostics = [], [], []
        for root in roots:
            image_counts, local = [], []
            stats = dict(error_sessions=0, fallback_sessions=0)
            for row, truth, initial in image_data:
                stem = Path(row['id']).stem
                if method == 'given':
                    pred = initial
                elif method == 'fixed':
                    pred = load_mask_png(Path(root) / (stem + '.png'))
                else:
                    pred, diag = session_prediction(root, stem, truth.shape)
                    stats['error_sessions'] += diag['error']
                    stats['fallback_sessions'] += diag['fallback']
                if pred.shape != truth.shape:
                    raise ValueError(f'Mask shape differs: {method} {stem}')
                image_counts.append(counts(pred, truth))
                local.append(repair_and_outside(pred, initial, truth, row['meta']['feedback']))
            summary = summarize(image_counts, local)
            summary['delta_f1_given'] = summary['f1_c'] - given_f1
            per_run.append(summary)
            repeated_counts.append(image_counts)
            diagnostics.append(stats)
        reports[method] = dict(mean=mean_scores(per_run), per_run=per_run, diagnostics=diagnostics)
        arrays[method] = np.asarray(repeated_counts, np.int64)
    comparisons = {}
    for baseline in ('fixed', 'given'):
        comparisons[baseline] = dict(delta_f1=reports['agent']['mean']['f1_c'] - reports[baseline]['mean']['f1_c'],
            ci95=paired_bootstrap(arrays['agent'], arrays[baseline][0], samples, seed))
    return dict(images=len(rows), k=rows[0]['meta']['k'], methods=reports, comparisons=comparisons,
        bootstrap=dict(samples=samples, seed=seed, unit='paired image; same indices across all three fixed repeats'),
        definitions=dict(connectivity=8, outside='outside Euclidean dilation (radius 32) of full pointed source components',
            repair_rate='fraction of pointed source components repaired; null when no pointed errors',
            units='F1/P/R percent; delta F1 percentage points; outside pixels dataset totals per run'))


def markdown(report):
    lines = ['| k | 方法 / 重复 | F1^c | P | R | ΔF1 给定 | 修复 漏检 / 误检 / 合计 | 区域外 增 / 删 | 区域外 ΔF1 | 错误 / 兜底 | Δ固定 [95% CI] | Δ给定 [95% CI] |',
             '|---|---|---:|---:|---:|---:|---|---|---:|---|---|---|']
    def rate(value):
        return '—' if value is None else f'{100 * value:.2f}%'
    def ci(value):
        return f"{value['delta_f1']:.2f} [{value['ci95'][0]:.2f}, {value['ci95'][1]:.2f}]"
    for group in report.values():
        for method, entry in group['methods'].items():
            table_rows = [('mean', entry['mean'], entry['diagnostics'])]
            if len(entry['per_run']) == 3:
                table_rows += [(f's{n}', s, [d]) for n, (s, d) in enumerate(zip(entry['per_run'], entry['diagnostics']), 1)]
            for label, s, diag in table_rows:
                repairs = ' / '.join(rate(s['repair'][k]['rate']) for k in ('missed', 'false_positive', 'all'))
                outside = s['outside']
                intervals = [ci(group['comparisons'][b]) if method == 'agent' and label == 'mean' else '—' for b in ('fixed', 'given')]
                lines.append(f"| {group['k']} | {method} {label} | {s['f1_c']:.2f} | {s['precision']:.2f} | {s['recall']:.2f} | {s['delta_f1_given']:+.2f} | {repairs} | {outside['added_px']:g} / {outside['removed_px']:g} | {outside['delta_f1']:+.2f} | {sum(d['error_sessions'] for d in diag)} / {sum(d['fallback_sessions'] for d in diag)} | {intervals[0]} | {intervals[1]} |")
    return '\n'.join(lines) + '\n\n三次重复先分别累计整集像素再取均值。修复率按被点组件汇总，区域外像素数为整集总量的逐次值或均值。均值行的错误/兜底为三次总数，错误可同时计入兜底。无被点错误时修复率为 —。区域外以完整源组件的欧氏距离外扩 32 像素为界，空反馈时为整图。CI 按图配对抽样 4000 次（种子 1016），三次重复共享抽样、不抽重复。缺会话或已有路径指向的掩码缺失时失败。\n'


def default_spec(run):
    run = Path(run).resolve()
    return dict(groups={f'k{k}': dict(manifest=str(run / f't52/tasks/T5.2_k{k}.jsonl'),
        fixed_dir=str(run / f't52/fixed/k{k}'),
        agent_runs=[str(run / f'json_qwen37flash_g2_t52k{k}_s{n}/T5.2/val') for n in (1, 2, 3)],
        g2_runs=[str(run / f'json_qwen37flash_g2_s{n}/T5.1/val') for n in (1, 2, 3)]) for k in (1, 3)})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path)
    p.add_argument('--write-spec', action='store_true')
    p.add_argument('--run-root', type=Path, default=Path(os.environ.get('AGENT_RUN', '/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run')))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--bootstrap-samples', type=int, default=4000)
    p.add_argument('--seed', type=int, default=1016)
    a = p.parse_args()
    if a.write_spec:
        write_json(a.output, default_spec(a.run_root))
        return
    if not a.spec or a.bootstrap_samples < 1:
        p.error('--spec and positive --bootstrap-samples are required')
    spec = json.loads(a.spec.read_text())
    report = {name: evaluate_group(group, a.bootstrap_samples, a.seed) for name, group in spec['groups'].items()}
    write_json(a.output, report)
    a.output.with_suffix('.md').write_text(markdown(report))


if __name__ == '__main__':
    main()
