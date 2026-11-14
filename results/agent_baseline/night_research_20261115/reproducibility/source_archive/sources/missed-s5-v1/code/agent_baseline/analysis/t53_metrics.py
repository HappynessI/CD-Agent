"""T5.3 paired image evaluation, reusing T5.2 scores, repair and outside metrics."""
import argparse
import json
from pathlib import Path
if __package__ in (None, ''):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import agent_baseline
import numpy as np
from common import components, load_mask_png, write_json
from agent_baseline.analysis.make_t52_feedback import region_mask, RUN
from agent_baseline.analysis.t52_metrics import counts, score, repair_and_outside, summarize, mean_scores, session_prediction
from agent_baseline.tasks import read_tasks


def localization(final, initial, truth, feedback):
    """Changed components = 8-connected components of XOR(final, given).

    Targets are the saved described error regions (uncovered pixels for a miss,
    full given component for a false positive). No edits => precision null;
    no feedback => target hit rate null. Unprompted edits still count.
    """
    changed = np.asarray(final, bool) ^ initial
    labels, comps = components(changed)
    gl, _ = components(truth)
    il, _ = components(initial)
    targets = [region_mask(e, gl, il, initial) for e in feedback]
    union = np.zeros_like(initial)
    for region in targets:
        union |= region
    hit = sum(bool(union[labels == c['id']].any()) for c in comps)
    return dict(changed_components=len(comps), matching_components=hit,
                described_regions=len(targets), edited_regions=sum(bool(changed[r].any()) for r in targets))


def localization_summary(rows):
    out = {k: sum(r[k] for r in rows) for k in
           ('changed_components', 'matching_components', 'described_regions', 'edited_regions')}
    out['component_precision'] = out['matching_components'] / out['changed_components'] if out['changed_components'] else None
    out['described_region_edit_rate'] = out['edited_regions'] / out['described_regions'] if out['described_regions'] else None
    return out


def paired_delta(a, b, samples=4000, seed=1016):
    """Keep repeats fixed and share paired image draws across both methods."""
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim != 3 or b.ndim != 3 or a.shape[1:] != b.shape[1:]:
        raise ValueError('Expected repeated per-image TP/FP/FN arrays with matching images')
    rng = np.random.default_rng(seed)
    def difference(idx):
        return float(np.mean([score(r[idx])['f1_c'] for r in a]) - np.mean([score(r[idx])['f1_c'] for r in b]))
    values = [difference(rng.integers(0, a.shape[1], a.shape[1])) for _ in range(samples)]
    return dict(delta_f1=difference(np.arange(a.shape[1])),
                ci95=[float(x) for x in np.percentile(values, [2.5, 97.5])])


def evaluate(spec, samples=4000, seed=1016):
    rows = read_tasks(spec['manifest'], 'T5.3')
    if not rows:
        raise ValueError('Empty T5.3 manifest')
    if len({Path(r['id']).stem for r in rows}) != len(rows):
        raise ValueError('Duplicate image output stems')
    methods = dict(given=dict(kind='given', roots=[None]),
        largest=dict(kind='fixed', roots=[spec['largest_dir']]),
        verifier=dict(kind='fixed', roots=[spec['verifier_dir']]),
        g2_t51=dict(kind='session', roots=spec['g2_runs']),
        agent=dict(kind='session', roots=spec['agent_runs']))
    if spec.get('t52_runs'):
        methods['t52'] = dict(kind='session', roots=spec['t52_runs'])
    for method in methods.values():
        if method['kind'] == 'session' and (len(method['roots']) != 3 or len(set(method['roots'])) != 3):
            raise ValueError('Session methods require three distinct repeat directories')
    data = [(r, load_mask_png(r['gt_mask']), load_mask_png(r['init_mask'])) for r in rows]
    given_f1 = score([counts(i, g) for _, g, i in data])['f1_c']
    reports, arrays = {}, {}
    for name, method in methods.items():
        per_run, repeated, diagnostics, per_image = [], [], [], []
        for root in method['roots']:
            image_counts, local, located, images = [], [], [], {}
            stats = dict(error_sessions=0, fallback_sessions=0)
            for row, truth, initial in data:
                stem = Path(row['id']).stem
                if method['kind'] == 'given':
                    pred = initial
                elif method['kind'] == 'fixed':
                    pred = load_mask_png(Path(root) / (stem + '.png'))
                else:
                    pred, diag = session_prediction(root, stem, truth.shape)
                    stats['error_sessions'] += diag['error']
                    stats['fallback_sessions'] += diag['fallback']
                feedback = row['meta']['feedback']
                image_counts.append(counts(pred, truth))
                local.append(repair_and_outside(pred, initial, truth, feedback))
                located.append(localization(pred, initial, truth, feedback))
                images[row['id']] = dict(counts=image_counts[-1], local=local[-1], localization=located[-1])
            summary = summarize(image_counts, local)
            summary['delta_f1_given'] = summary['f1_c'] - given_f1
            summary['localization'] = localization_summary(located)
            per_run.append(summary)
            repeated.append(image_counts)
            diagnostics.append(stats)
            per_image.append(images)
        mean = mean_scores(per_run)
        mean['localization'] = {k: (float(np.mean([s['localization'][k] for s in per_run
                                                   if s['localization'][k] is not None]))
                                  if any(s['localization'][k] is not None for s in per_run) else None)
                                for k in per_run[0]['localization']}
        reports[name] = dict(mean=mean, per_run=per_run, diagnostics=diagnostics, per_image=per_image)
        arrays[name] = np.asarray(repeated, np.int64)
    return dict(images=len(rows), methods=reports,
        comparisons={name: paired_delta(arrays['agent'], a, samples, seed) for name, a in arrays.items() if name != 'agent'},
        bootstrap=dict(samples=samples, seed=seed, unit='paired image; shared image sample across fixed repeats'),
        definitions=dict(connectivity=8, outside='T5.2: outside radius-32 Euclidean dilation of full described source components',
            localization='8-connected XOR edits intersecting the described error region; misses use uncovered pixels',
            no_edits='component precision null', no_feedback='described region edit rate null',
            t52='if supplied, re-evaluate the T5.2 masks on the T5.3 described errors for a paired reference'))


def markdown(report):
    def rate(value):
        return '—' if value is None else f'{100*value:.2f}%'
    lines = ['| 方法 / 重复 | F1^c | P | R | ΔF1 给定 | 修复 漏检 / 误检 | 区域外 增 / 删 | 区域外 ΔF1 | 定位组件比例 | 描述区域被改动 | 错误 / 兜底 |',
             '|---|---:|---:|---:|---:|---|---|---:|---:|---:|---|']
    for method, entry in report['methods'].items():
        table = [('mean', entry['mean'], entry['diagnostics'])]
        if len(entry['per_run']) > 1:
            table.extend((f's{n}', s, [d]) for n, (s, d) in enumerate(zip(entry['per_run'], entry['diagnostics']), 1))
        for label, s, diag in table:
            repair, outside, loc = s['repair'], s['outside'], s['localization']
            lines.append(f"| {method} {label} | {s['f1_c']:.2f} | {s['precision']:.2f} | {s['recall']:.2f} | {s['delta_f1_given']:+.2f} | {rate(repair['missed']['rate'])} / {rate(repair['false_positive']['rate'])} | {outside['added_px']:g} / {outside['removed_px']:g} | {outside['delta_f1']:+.2f} | {rate(loc['component_precision'])} | {rate(loc['described_region_edit_rate'])} | {sum(d['error_sessions'] for d in diag)} / {sum(d['fallback_sessions'] for d in diag)} |")
    lines += ['', '| Agent − 参照 | ΔF1 [95% CI] |', '|---|---|']
    for method, c in report['comparisons'].items():
        lines.append(f"| {method} | {c['delta_f1']:+.2f} [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}] |")
    lines += ['', 'F1/P/R 为百分数，ΔF1 为百分点。逐次累计像素后取均值；定位比例逐次汇总组件后取均值。',
              '定位组件为最终与给定掩码异或后的 8 连通域。无改动时比例为 —，有反馈但未改动时描述区域被改动率为 0。',
              '修复和区域外指标沿用 T5.2 完整源组件口径，区域外为欧氏距离外扩 32 像素之外。',
              f"按图配对 bootstrap {report['bootstrap']['samples']} 次，种子 {report['bootstrap']['seed']}，重复固定并共享图像抽样。缺会话或掩码即报错。", '']
    return '\n'.join(lines)


def default_spec(run):
    run = Path(run).resolve()
    spec = dict(manifest=str(run / 't53/tasks/T5.3.jsonl'),
        largest_dir=str(run / 't53/fixed/largest/masks'), verifier_dir=str(run / 't53/fixed/verifier/masks'),
        agent_runs=[str(run / f'json_qwen37flash_g3_t53_s{n}/T5.3/val') for n in (1, 2, 3)],
        g2_runs=[str(run / f'json_qwen37flash_g2_s{n}/T5.1/val') for n in (1, 2, 3)])
    t52 = [run / f'json_qwen37flash_g2_t52_k1_s{n}/T5.2/val' for n in (1, 2, 3)]
    if any(p.exists() for p in t52):
        spec['t52_runs'] = [str(p) for p in t52]
    return spec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path)
    p.add_argument('--write-spec', action='store_true')
    p.add_argument('--run-root', type=Path, default=RUN)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--bootstrap-samples', type=int, default=4000)
    p.add_argument('--seed', type=int, default=1016)
    a = p.parse_args()
    if a.write_spec:
        write_json(a.output, default_spec(a.run_root))
        return
    if not a.spec or a.bootstrap_samples < 1:
        p.error('--spec and positive --bootstrap-samples are required')
    report = evaluate(json.loads(a.spec.read_text()), a.bootstrap_samples, a.seed)
    write_json(a.output, report)
    a.output.with_suffix('.md').write_text(markdown(report))


if __name__ == '__main__':
    main()
