"""Clip frozen routes, rules and three historical Agent repeats to C2/C4 regions."""
import argparse
import json
from pathlib import Path
import numpy as np
from agent_baseline.analysis.make_second_boxes import frozen_write, png_bytes
from agent_baseline.analysis.second_metrics import mask
from agent_baseline.fixed_routes import RUN
from agent_baseline.tasks import read_tasks

GROUPS = [('c2', 'T1.1xC2', 'any', 'T1.1xC0', 'any'),
          ('c4', 'T1.1xC4', 'appeared', 'T1.1xC1', 'appeared')]


def final_prediction(directory, shape):
    directory = Path(directory)
    final = json.loads((directory / 'final.json').read_text())
    by = final['submitted_by']
    if final.get('mask_path'):
        pred = mask(directory / final['mask_path'], shape)
    elif by == 'error' or final.get('fallback_kind') == 'empty':
        pred = np.zeros(shape, bool)
    else:
        raise ValueError(f'Missing final mask: {directory}')
    return pred, dict(error_sessions=int(by == 'error' or bool(final.get('error'))),
                      fallback_sessions=int(by != 'model'))


def generate(run, output, fixed_any, fixed_appeared, rule_any, rule_appeared, agent_prefix='second_g2'):
    run, output = Path(run).resolve(), Path(output).resolve()
    writes, groups = [], {}
    for kind, task, direction, old_task, old_direction in GROUPS:
        rows = read_tasks(run / 'second/tasks' / f'{task}_{direction}.jsonl')
        if not rows:
            raise ValueError('Empty box manifest')
        stems = [Path(r['id']).stem for r in rows]
        if len(set(stems)) != len(stems):
            raise ValueError('Duplicate output stems')
        sources = dict(fixed_any=Path(fixed_any), rule=Path(rule_any if kind == 'c2' else rule_appeared))
        if kind == 'c4':
            sources['fixed_direction'] = Path(fixed_appeared)
        sources.update({f'posthoc_s{n}': run / f'{agent_prefix}_{old_direction}_s{n}' / old_task / 'val' for n in (1, 2, 3)})
        diagnostics = {name: dict(error_sessions=0, fallback_sessions=0) for name in sources if name.startswith('posthoc')}
        for row, stem in zip(rows, stems):
            region = mask(row['region_mask'])
            for name, source in sources.items():
                if name.startswith('posthoc'):
                    pred, diag = final_prediction(source / stem, region.shape)
                    for key, value in diag.items():
                        diagnostics[name][key] += value
                else:
                    pred = mask(source / (stem + '.png'), region.shape)
                writes.append((output / kind / name / (stem + '.png'), png_bytes(pred & region)))
        groups[kind] = dict(ids=[r['id'] for r in rows], sources={k: str(v.resolve()) for k, v in sources.items()},
                            run_diagnostics=diagnostics)
    report = dict(groups=groups, operation='pixel intersection; boundary buildings are clipped', agent_prefix=agent_prefix)
    writes.append((output / 'preparation.json', (json.dumps(report, indent=2) + '\n').encode()))
    for path, content in writes:
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f'Existing output differs: {path}; choose a fresh output directory')
    for path, content in writes:
        frozen_write(path, content)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, default=RUN)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--fixed-any-dir', type=Path, required=True)
    p.add_argument('--fixed-appeared-dir', type=Path, required=True)
    p.add_argument('--rule-any-dir', type=Path, required=True)
    p.add_argument('--rule-appeared-dir', type=Path, required=True)
    p.add_argument('--agent-prefix', default='second_g2', help='Historical C0/C1 config prefix, including a path if needed')
    a = p.parse_args()
    report = generate(a.run_root, a.output_dir, a.fixed_any_dir, a.fixed_appeared_dir,
                      a.rule_any_dir, a.rule_appeared_dir, a.agent_prefix)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
