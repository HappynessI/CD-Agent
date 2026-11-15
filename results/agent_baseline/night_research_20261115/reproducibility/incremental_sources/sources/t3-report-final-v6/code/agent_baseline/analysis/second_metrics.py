"""Offline SECOND report: pooled pixel scores, direction errors and paired image CIs.

Input is a JSON spec with a `groups` mapping. Each group has manifest, three
agent_runs (directories containing <stem>/final.json), fixed_dir, rule_dir,
init_dir, undirected_dir and rule_answers. C0 may omit rule_dir/rule_answers.
All counts include empty/error fallbacks. Missing sessions/masks are errors,
never silently dropped. See README_second.md for a complete spec generator.
"""
import argparse
import json
from pathlib import Path
if __package__ in (None, ''):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import agent_baseline
import numpy as np
from PIL import Image
from common import write_json
from agent_baseline.tasks import read_tasks, needs_init
from agent_baseline.analysis.second_labels import DIRECTION_GT, pixel_counts


def direction_violation(pred, opposite):
    pred, opposite = np.asarray(pred, bool), np.asarray(opposite, bool)
    if pred.shape != opposite.shape:
        raise ValueError('Direction truth shape differs')
    return float((pred & opposite).sum() / pred.sum()) if pred.any() else 0.


def counts(pred, truth, opposite=None, valid=None):
    if valid is not None:
        pred, truth = pred & valid, truth & valid
    return pixel_counts(pred, truth) + [int((pred & opposite).sum()) if opposite is not None else 0, int(pred.sum())]


def score(c):
    tp, fp, fn, wrong, predicted = np.asarray(c, float).sum(axis=0)
    return dict(f1_c=200 * tp / max(1, 2*tp+fp+fn), precision=100 * tp / max(1, tp+fp),
                recall=100 * tp / max(1, tp+fn), direction_violation=wrong / max(1, predicted),
                tp=int(tp), fp=int(fp), fn=int(fn), opposite_pixels=int(wrong), predicted_pixels=int(predicted))


def paired_bootstrap(agent, baseline, seed=1016, samples=4000):
    """Resample paired images, sharing indices across all three repeats and baseline.

    Each replicate is the mean of three pooled Agent F1 values minus the pooled
    baseline F1. Repeats are kept fixed; the CI measures image sampling variation.
    """
    agent, baseline = np.asarray(agent), np.asarray(baseline)
    if agent.ndim != 3 or baseline.shape != agent.shape[1:]:
        raise ValueError('Expected (repeats, images, counts) vs (images, counts)')
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        idx = rng.integers(0, agent.shape[1], agent.shape[1])
        values.append(np.mean([score(a[idx])['f1_c'] for a in agent]) - score(baseline[idx])['f1_c'])
    return [float(v) for v in np.percentile(values, [2.5, 97.5])]


def mask(path, shape=None):
    with Image.open(path) as image:
        data = np.asarray(image) > 0
    if data.ndim != 2 or (shape is not None and data.shape != shape):
        raise ValueError(f'Invalid binary mask shape: {path}')
    return data


def rebuilt_online(path):
    if not path.exists():
        return dict(answers=0, regions=0, ledger_missing=1)
    ledger = json.loads(path.read_text())
    return dict(answers=sum(r.get('type') == 'rebuilt' for e in ledger for r in e['records']),
                regions=sum(any(r.get('type') == 'rebuilt' for r in e['records']) for e in ledger), ledger_missing=0)


def evaluate_group(spec, samples=4000, seed=1016):
    rows = read_tasks(spec['manifest'])
    if not rows:
        raise ValueError('Empty task manifest')
    if len({(r['task_type'], r['change_direction']) for r in rows}) != 1:
        raise ValueError('Each group must use one task and direction')
    task, direction = rows[0]['task_type'], rows[0]['change_direction']
    if len(spec['agent_runs']) != 3 or len(set(spec['agent_runs'])) != 3:
        raise ValueError('Exactly three distinct repeat directories are required')
    reference = 'fixed_unintersected' if task == 'T1.1xC1t' else 'fixed_any'
    baselines = dict(fixed_direction=spec['fixed_dir'], **{reference: spec['undirected_dir']})
    if spec.get('rule_dir'):
        baselines['rule_direction'] = spec['rule_dir']
    if direction != 'any' and not spec.get('rule_dir'):
        raise ValueError('C1 requires the directed deletion-rule baseline')
    if needs_init(task):
        baselines['given_unchanged'] = spec['init_dir']
        baselines['given_direction_only'] = spec['fixed_dir']
    base = {name: [] for name in baselines}
    repeats, diagnostics = [], []
    truths = []
    stems = [Path(r['id']).stem for r in rows]
    if len(stems) != len(set(stems)):
        raise ValueError('Duplicate image IDs')
    for row, stem in zip(rows, stems):
        from agent_baseline.evaluation import load_reference
        gt, valid = load_reference(row['gt_mask'], row.get('gt_ignore_mask'))
        opposite = None
        if direction != 'any':
            opposite_kind = 'demolished' if direction == 'appeared' else 'new'
            opposite = mask(Path(row['gt_mask']).parent.parent / opposite_kind / Path(row['gt_mask']).name, gt.shape)
        truths.append((gt, opposite, valid))
        for name, directory in baselines.items():
            base[name].append(counts(mask(Path(directory) / (stem + '.png'), gt.shape), gt, opposite, valid))
    for run in spec['agent_runs']:
        predictions = []
        stats = dict(error_sessions=0, fallback_sessions=0, submitted_by={}, rebuilt_answers=0,
                     rebuilt_regions=0, missing_verification_ledgers=0)
        for row, stem, (gt, opposite, valid) in zip(rows, stems, truths):
            directory = Path(run) / stem
            final = json.loads((directory / 'final.json').read_text())
            by = final['submitted_by']
            stats['submitted_by'][by] = stats['submitted_by'].get(by, 0) + 1
            stats['error_sessions'] += by == 'error' or bool(final.get('error'))
            stats['fallback_sessions'] += by != 'model'
            if final.get('mask_path'):
                pred = mask(directory / final['mask_path'], gt.shape)
            elif by == 'error' or final.get('fallback_kind') == 'empty':
                pred = np.zeros_like(gt)
            else:
                raise ValueError(f'Missing final mask: {directory}')
            predictions.append(counts(pred, gt, opposite, valid))
            r = rebuilt_online(directory / 'verification.json')
            stats['rebuilt_answers'] += r['answers']
            stats['rebuilt_regions'] += r['regions']
            stats['missing_verification_ledgers'] += r['ledger_missing']
        repeats.append(predictions)
        diagnostics.append(stats)
    A = np.asarray(repeats, np.int64)
    per_run = [score(a) for a in A]
    average = {k: float(np.mean([r[k] for r in per_run])) for k in ('f1_c', 'precision', 'recall', 'direction_violation')}
    references = {name: score(np.asarray(c)) for name, c in base.items()}
    comparisons = {}
    for name in ('fixed_direction', 'rule_direction'):
        if name in base:
            comparisons[name] = dict(delta_f1=average['f1_c'] - references[name]['f1_c'],
                ci95=paired_bootstrap(A, base[name], seed, samples))
    offline_rebuilt = None
    if spec.get('rule_answers'):
        answers = [json.loads(line) for line in Path(spec['rule_answers']).read_text().splitlines() if line.strip()]
        matched = {r['uid']: r for r in answers if r['uid'].rsplit(':', 1)[0] in stems}
        offline_rebuilt = sum(r.get('type') == 'rebuilt' for r in matched.values())
    elif direction != 'any':
        raise ValueError('C1 requires rule_answers to count offline rebuilt answers')
    return dict(task=task, direction=direction, images=len(rows), repeats=3,
        agent_mean=average, agent_per_run=per_run, baselines=references, comparisons=comparisons,
        delta_f1_given=average['f1_c'] - references['given_unchanged']['f1_c'] if needs_init(task) else None,
        run_diagnostics=diagnostics, offline_rebuilt=offline_rebuilt,
        bootstrap=dict(seed=seed, samples=samples, unit='paired image; same resampled images across fixed three repeats'),
        units='F1/P/R percent; delta F1 percentage points; direction violation fraction; empty prediction violation=0')


def markdown(report):
    lines = ['| Group | Row | F1^c | P | R | Direction violation | ΔF1 given | Δ fixed [95% CI] | Δ rule [95% CI] | Rebuilt | Error / fallback |',
             '|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|']
    def interval(row, name):
        comp = row['comparisons'].get(name)
        return f"{comp['delta_f1']:.2f} [{comp['ci95'][0]:.2f}, {comp['ci95'][1]:.2f}]" if comp else '—'
    for group, row in report.items():
        s = row['agent_mean']
        diag = row['run_diagnostics']
        delta = f"{row['delta_f1_given']:.2f}" if row['delta_f1_given'] is not None else '—'
        lines.append(f"| {group} | Agent mean (3) | {s['f1_c']:.2f} | {s['precision']:.2f} | {s['recall']:.2f} | {s['direction_violation']:.4f} | {delta} | {interval(row, 'fixed_direction')} | {interval(row, 'rule_direction')} | {sum(d['rebuilt_answers'] for d in diag)} | {sum(d['error_sessions'] for d in diag)} / {sum(d['fallback_sessions'] for d in diag)} |")
        for name, s in row['baselines'].items():
            rebuilt = row['offline_rebuilt'] if name == 'rule_direction' else '—'
            lines.append(f"| {group} | {name} | {s['f1_c']:.2f} | {s['precision']:.2f} | {s['recall']:.2f} | {s['direction_violation']:.4f} | — | — | — | {rebuilt} | — |")
    return '\n'.join(lines) + '\n\nF1/P/R 按每次重复累计像素后计算，Agent 行报三次均值。方向违反率分母为预测像素数，空预测记 0。CI 按图配对抽样，三次重复固定。Rebuilt：在线为账本回答次数（含重复核验），离线为去重区域回答次数。错误会话计入兜底，缺失会话拒绝汇总。\n'


def default_spec(run, prefix='second_g2'):
    from agent_baseline.analysis.make_second_tasks import SUBTASKS
    run = Path(run).resolve()
    groups = {}
    for task, direction in SUBTASKS:
        key = f'{task}_{direction}'
        group = dict(manifest=str(run / 'second/tasks' / (key + '.jsonl')),
            agent_runs=[str(run / f'{prefix}_{direction}_s{n}' / task / 'val') for n in (1, 2, 3)],
            fixed_dir=str(run / 'second/fixed' / direction),
            undirected_dir=str(run / 'second/fixed/any'),
            init_dir=str(run / 'init/second_sam3_imc_p85/val'))
        if direction != 'any':
            group.update(rule_dir=str(run / 'second/rule' / direction / 'masks'),
                         rule_answers=str(run / 'second/rule' / direction / 'answers.jsonl'))
        groups[key] = group
    return dict(groups=groups)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path)
    p.add_argument('--write-spec', action='store_true')
    p.add_argument('--run-root', type=Path)
    p.add_argument('--agent-prefix', default='second_g2')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--bootstrap-samples', type=int, default=4000)
    p.add_argument('--seed', type=int, default=1016)
    a = p.parse_args()
    if a.write_spec:
        if not a.run_root:
            p.error('--write-spec requires --run-root')
        write_json(a.output, default_spec(a.run_root, a.agent_prefix))
        return
    if not a.spec or a.bootstrap_samples < 1:
        p.error('--spec and a positive bootstrap sample count are required')
    spec = json.loads(a.spec.read_text())
    report = {name: evaluate_group(group, a.bootstrap_samples, a.seed) for name, group in spec['groups'].items()}
    write_json(a.output, report)
    a.output.with_suffix('.md').write_text(markdown(report))


if __name__ == '__main__':
    main()
