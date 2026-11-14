"""C2/C4 pooled pixel scores, region/direction violations and paired image CIs."""
import argparse
import json
from pathlib import Path
import numpy as np
from common import write_json
from agent_baseline.analysis.second_metrics import mask, counts as direction_counts, score as direction_score
from agent_baseline.analysis.second_box_baselines import GROUPS, final_prediction
from agent_baseline.tasks import read_tasks


def outside_violation(pred, region):
    pred, region = np.asarray(pred, bool), np.asarray(region, bool)
    if pred.shape != region.shape:
        raise ValueError('Region and prediction shapes differ')
    return float((pred & ~region).sum() / pred.sum()) if pred.any() else 0.


def counts(pred, truth, region, opposite=None):
    outside_violation(pred, region)
    return direction_counts(pred, truth, opposite) + [int((pred & ~region).sum())]


def score(rows):
    rows = np.asarray(rows)
    result = direction_score(rows[:, :5])
    result['outside_pixels'] = int(rows[:, 5].sum())
    result['outside_violation'] = result['outside_pixels'] / max(1, result['predicted_pixels'])
    return result


def paired_bootstrap(agent, baseline, samples=4000, seed=1016):
    """Share each paired image sample across the three fixed repeats on both sides."""
    agent, baseline = np.asarray(agent), np.asarray(baseline)
    if baseline.ndim == 2:
        baseline = baseline[None, ...]
    if (agent.ndim != 3 or agent.shape[0] != 3 or baseline.ndim != 3
            or baseline.shape[0] not in (1, 3) or baseline.shape[1:] != agent.shape[1:]):
        raise ValueError('Expected three Agent repeats and one or three paired baseline repeats')
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        idx = rng.integers(0, agent.shape[1], agent.shape[1])
        values.append(float(np.mean([score(a[idx])['f1_c'] for a in agent])
                            - np.mean([score(b[idx])['f1_c'] for b in baseline])))
    return [float(v) for v in np.percentile(values, [2.5, 97.5])]


def evaluate_group(spec, samples=4000, seed=1016):
    rows = read_tasks(spec['manifest'])
    if not rows or len({(r['task_type'], r['change_direction']) for r in rows}) != 1:
        raise ValueError('Use one nonempty task/direction manifest per group')
    task = rows[0]['task_type']
    if task not in ('T1.1xC2', 'T1.1xC4'):
        raise ValueError('Expected C2/C4 manifest')
    if len(spec['agent_runs']) != 3 or len(set(spec['agent_runs'])) != 3:
        raise ValueError('Exactly three distinct Agent repeat directories are required')
    required = {'fixed_any', 'rule', 'posthoc'} | ({'fixed_direction'} if task == 'T1.1xC4' else set())
    if not required <= spec['baselines'].keys():
        raise ValueError(f'Missing baseline groups: {sorted(required - spec["baselines"].keys())}')
    stems = [Path(r['id']).stem for r in rows]
    if len(set(stems)) != len(stems):
        raise ValueError('Duplicate output stems')
    truths = []
    for row in rows:
        truth = mask(row['gt_mask'])
        region = mask(row['region_mask'], truth.shape)
        opposite = mask(Path(spec['demolished_dir']) / Path(row['id']).name, truth.shape) if task == 'T1.1xC4' else None
        truths.append((truth, region, opposite))
    A, diagnostics = [], []
    for run in spec['agent_runs']:
        per, diag = [], dict(error_sessions=0, fallback_sessions=0)
        for stem, (truth, region, opposite) in zip(stems, truths):
            pred, session_diag = final_prediction(Path(run) / stem, truth.shape)
            per.append(counts(pred, truth, region, opposite))
            for key, value in session_diag.items():
                diag[key] += value
        A.append(per)
        diagnostics.append(diag)
    A = np.asarray(A, np.int64)
    per_run = [score(a) for a in A]
    fields = ('f1_c', 'precision', 'recall', 'outside_violation', 'direction_violation')
    average = {key: float(np.mean([r[key] for r in per_run])) for key in fields}
    baselines, comparisons = {}, {}
    for name, directories in spec['baselines'].items():
        if isinstance(directories, str):
            directories = [directories]
        if len(directories) != (3 if name == 'posthoc' else 1) or len(set(directories)) != len(directories):
            raise ValueError(f'Incorrect baseline repeat directories: {name}')
        B = np.asarray([[counts(mask(Path(directory) / (stem + '.png'), truth.shape), truth, region, opposite)
                         for stem, (truth, region, opposite) in zip(stems, truths)] for directory in directories])
        per = [score(b) for b in B]
        mean = {key: float(np.mean([r[key] for r in per])) for key in fields}
        baselines[name] = dict(mean=mean, per_run=per)
        comparisons[name] = dict(delta_f1=average['f1_c'] - mean['f1_c'], ci95=paired_bootstrap(A, B, samples, seed))
    preparation = json.loads(Path(spec['baseline_preparation']).read_text())
    kind = 'c2' if task == 'T1.1xC2' else 'c4'
    prep = preparation['groups'][kind]
    if prep['ids'] != [r['id'] for r in rows]:
        raise ValueError('Baseline preparation image order differs from manifest')
    return dict(task=task, images=len(rows), repeats=3, agent_mean=average, agent_per_run=per_run,
                baselines=baselines, comparisons=comparisons, run_diagnostics=diagnostics,
                posthoc_diagnostics=prep['run_diagnostics'],
                bootstrap=dict(samples=samples, seed=seed, unit='paired image, shared indices across all three repeats'),
                units='F1/P/R percent; delta F1 percentage points; violations are pooled predicted-pixel fractions, empty=0')


def default_spec(run, baseline_root, agent_prefix='second_g2'):
    run, baseline_root = Path(run).resolve(), Path(baseline_root).resolve()
    groups = {}
    for kind, task, direction, _, _ in GROUPS:
        base = {key: str(baseline_root / kind / key) for key in ('fixed_any', 'rule')}
        if kind == 'c4':
            base['fixed_direction'] = str(baseline_root / kind / 'fixed_direction')
        base['posthoc'] = [str(baseline_root / kind / f'posthoc_s{n}') for n in (1, 2, 3)]
        groups[kind] = dict(manifest=str(run / 'second/tasks' / f'{task}_{direction}.jsonl'),
            agent_runs=[str(run / f'{agent_prefix}_{kind}_s{n}' / task / 'val') for n in (1, 2, 3)],
            baselines=base, baseline_preparation=str(baseline_root / 'preparation.json'),
            demolished_dir=str(run / 'second/gt/demolished'))
    return dict(groups=groups)


def markdown(report):
    lines = ['| 任务 | 方法 / 重复 | F1^c | P | R | 框外违反率 | 方向违反率 | Agent − 对照 [95% CI] | 错误 / 兜底会话 |',
             '|---|---|---:|---:|---:|---:|---:|---|---:|']
    def line(group, name, s, diag=None, comp=None):
        direction = f"{s['direction_violation']:.6f}" if group == 'c4' else '—'
        delta = f"{comp['delta_f1']:.2f} [{comp['ci95'][0]:.2f}, {comp['ci95'][1]:.2f}]" if comp else '—'
        errors = f"{diag['error_sessions']} / {diag['fallback_sessions']}" if diag else '—'
        lines.append(f"| {group} | {name} | {s['f1_c']:.2f} | {s['precision']:.2f} | {s['recall']:.2f} | {s['outside_violation']:.6f} | {direction} | {delta} | {errors} |")
    for group, row in report.items():
        total = {k: sum(d[k] for d in row['run_diagnostics']) for k in ('error_sessions', 'fallback_sessions')}
        line(group, 'Agent 三次均值', row['agent_mean'], total)
        for n, (s, diag) in enumerate(zip(row['agent_per_run'], row['run_diagnostics']), 1):
            line(group, f'Agent s{n}', s, diag)
        for name, base in row['baselines'].items():
            diag = None
            if name == 'posthoc':
                diag = {k: sum(d[k] for d in row['posthoc_diagnostics'].values()) for k in ('error_sessions', 'fallback_sessions')}
            line(group, name, base['mean'], diag, row['comparisons'][name])
            if name == 'posthoc':
                for n, s in enumerate(base['per_run'], 1):
                    line(group, f'posthoc s{n}', s, row['posthoc_diagnostics'][f'posthoc_s{n}'])
    return '\n'.join(lines) + '\n\n逐次累计像素计算 F1/P/R，再取三次均值。违反率分母为全部预测像素，空预测记 0。C4 方向违反使用完整拆除真值。按图配对 bootstrap 4000 次、种子 1016，三次重复共享抽样；区间只含图像抽样变化。错误会话计入兜底，缺会话报错。\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--write-spec', action='store_true')
    p.add_argument('--run-root', type=Path)
    p.add_argument('--baseline-root', type=Path)
    p.add_argument('--agent-prefix', default='second_g2')
    p.add_argument('--spec', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.write_spec:
        if not a.run_root or not a.baseline_root:
            p.error('--write-spec requires --run-root and --baseline-root')
        write_json(a.output, default_spec(a.run_root, a.baseline_root, a.agent_prefix))
        return
    if not a.spec:
        p.error('--spec is required')
    spec = json.loads(a.spec.read_text())
    report = {name: evaluate_group(group) for name, group in spec['groups'].items()}
    write_json(a.output, report)
    a.output.with_suffix('.md').write_text(markdown(report))


if __name__ == '__main__':
    main()
