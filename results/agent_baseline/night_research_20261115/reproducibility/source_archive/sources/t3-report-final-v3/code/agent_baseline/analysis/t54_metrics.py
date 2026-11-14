"""T5.4 pooled pixel F1/P/R and paired-image intervals, using SECOND metrics."""
import argparse
import json
from pathlib import Path
if __package__ in (None, ''):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import agent_baseline
import numpy as np
from common import write_json
from agent_baseline.analysis.second_metrics import counts, mask, score, paired_bootstrap
from agent_baseline.tasks import read_tasks


def evaluate(spec, samples=4000, seed=1016):
    rows = read_tasks(spec['manifest'])
    if not rows or any(r['task_type'] != 'T5.4' for r in rows):
        raise ValueError('Use a nonempty T5.4 manifest')
    runs = spec.get('agent_runs', [])
    if runs and (len(runs) != 3 or len(set(runs)) != 3):
        raise ValueError('Exactly three distinct Agent repeats are required')
    if not {'fixed_xor', 't2_buildings'} <= spec['baselines'].keys():
        raise ValueError('Both fixed_xor and t2_buildings references are required')
    if samples < 1:
        raise ValueError('bootstrap samples must be positive')
    stems = [Path(r['id']).stem for r in rows]
    if len(set(stems)) != len(stems):
        raise ValueError('Duplicate output stems')
    truths = [mask(r['gt_mask']) for r in rows]
    baselines = {name: np.asarray([counts(mask(Path(directory) / (stem + '.png'), truth.shape), truth)
                                  for stem, truth in zip(stems, truths)], np.int64)
                 for name, directory in spec['baselines'].items()}
    methods = {name: score(c) for name, c in baselines.items()}
    comparisons = {}
    # Also useful before Agent runs: quantify XOR vs the requested un-XORed reference.
    for name, baseline in baselines.items():
        if name != 'fixed_xor':
            fixed = baselines['fixed_xor']
            comparisons['fixed_xor_minus_' + name] = dict(
                delta_f1=methods['fixed_xor']['f1_c'] - methods[name]['f1_c'],
                ci95=paired_bootstrap(fixed[None, ...], baseline, seed, samples))
    repeats, diagnostics = [], []
    for run in runs:
        per, diag = [], dict(error_sessions=0, fallback_sessions=0)
        for stem, truth in zip(stems, truths):
            directory = Path(run) / stem
            final = json.loads((directory / 'final.json').read_text())
            by = final['submitted_by']
            diag['error_sessions'] += by == 'error' or bool(final.get('error'))
            diag['fallback_sessions'] += by != 'model'
            if final.get('mask_path'):
                pred = mask(directory / final['mask_path'], truth.shape)
            elif by == 'error' or final.get('fallback_kind') == 'empty':
                pred = np.zeros_like(truth)
            else:
                raise ValueError(f'Missing final mask: {directory}')
            per.append(counts(pred, truth))
        repeats.append(per)
        diagnostics.append(diag)
    per_run = []
    if repeats:
        agent = np.asarray(repeats, np.int64)
        per_run = [score(r) for r in agent]
        methods['agent_mean'] = {key: float(np.mean([r[key] for r in per_run]))
                                 for key in ('f1_c', 'precision', 'recall')}
        for name, baseline in baselines.items():
            comparisons['agent_minus_' + name] = dict(
                delta_f1=methods['agent_mean']['f1_c'] - methods[name]['f1_c'],
                ci95=paired_bootstrap(agent, baseline, seed, samples))
    return dict(task='T5.4', images=len(rows), methods=methods, agent_per_run=per_run,
                comparisons=comparisons, run_diagnostics=diagnostics,
                bootstrap=dict(samples=samples, seed=seed,
                    unit='paired image; same sampled images across three fixed Agent repeats'),
                units='F1/P/R percent; delta F1 percentage points; all empty/error predictions retained')


def markdown(report):
    lines = ['| 方法 | F1 | P | R |', '|---|---:|---:|---:|']
    for name, values in report['methods'].items():
        lines.append(f"| {name} | {values['f1_c']:.2f} | {values['precision']:.2f} | {values['recall']:.2f} |")
    lines += ['', '| 比较 | ΔF1 | 95% 区间 |', '|---|---:|---|']
    for name, values in report['comparisons'].items():
        lo, hi = values['ci95']
        lines.append(f"| {name} | {values['delta_f1']:.2f} | [{lo:.2f}, {hi:.2f}] |")
    lines += ['', '累计像素后计算 F1/P/R。Agent 报三次均值。按图配对抽样，三次重复固定；区间只含图像抽样变化。',
              '错误与空预测均纳入，缺失会话或掩码直接报错。', '']
    return '\n'.join(lines)


def default_spec(run, split='val', agent_prefix=None):
    run = Path(run).resolve()
    spec = dict(manifest=str(run / f't54/tasks/T5.4_{split}.jsonl'),
                baselines=dict(fixed_xor=str(run / f't54/fixed/{split}/xor'),
                               t2_buildings=str(run / f't54/fixed/{split}/buildings'),
                               rule=str(run / f't54/rule/{split}/any/masks')))
    if agent_prefix:
        spec['agent_runs'] = [str(run / f'{agent_prefix}_s{n}/T5.4/val') for n in (1, 2, 3)]
    return spec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path)
    p.add_argument('--write-spec', action='store_true')
    p.add_argument('--run-root', type=Path)
    p.add_argument('--split', choices=['dev', 'val'], default='val')
    p.add_argument('--agent-prefix')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--bootstrap-samples', type=int, default=4000)
    p.add_argument('--seed', type=int, default=1016)
    a = p.parse_args()
    if a.write_spec:
        if not a.run_root:
            p.error('--write-spec requires --run-root')
        write_json(a.output, default_spec(a.run_root, a.split, a.agent_prefix))
        return
    if not a.spec:
        p.error('--spec is required')
    report = evaluate(json.loads(a.spec.read_text()), a.bootstrap_samples, a.seed)
    write_json(a.output, report)
    a.output.with_suffix('.md').write_text(markdown(report))


if __name__ == '__main__':
    main()
