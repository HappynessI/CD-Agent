"""Metrics-only paired image bootstrap for fixed R1 joint and residual studies (Slurm)."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from .io_retry import read_text, write_json, write_text

SEED = 20261115
REPLICATES = 5000
PAIRS = {
    'r1_endpoint': ('R1_geometry_dino', 'R1_equal_joint'),
    'r1_residual': ('R1_direct4_verified', 'R1_residual_verified'),
}


def values(counts):
    """Same zero-denominator convention as t3_routes.pooled; F1 in percent."""
    counts = np.asarray(counts)
    tp, fp, fn = np.moveaxis(counts, -1, 0)
    f1 = 200 * tp / np.maximum(1, 2 * tp + fp + fn)
    return np.concatenate((counts, np.asarray(f1)[..., None]), axis=-1)


def paired_interval(before, after, seed=SEED, replicates=REPLICATES):
    """Resample paired images, then recompute pooled counts/F1, never mean task F1."""
    before, after = np.asarray(before, dtype=np.int64), np.asarray(after, dtype=np.int64)
    if len(before) == 0:
        return None
    draws = np.random.default_rng(seed).integers(0, len(before), size=(replicates, len(before)))
    differences = values(after[draws].sum(axis=1)) - values(before[draws].sum(axis=1))
    bounds = np.percentile(differences, [2.5, 97.5], axis=0)
    return {name: bounds[:, index].tolist() for index, name in enumerate(('tp', 'fp', 'fn', 'f1'))}


def read_pairs(folder, methods):
    records = []
    for path in sorted((folder / 'tasks').glob('*/metrics.json')):
        row = json.loads(read_text(path))
        scores = {r['method']: r['counts'] for r in row['scores'] if r['k'] == 8}
        before, after = [scores[name] for name in methods]
        records.append(dict(id=row['id'], split=row['split'], group=row['group'],
            support='positive' if row['remaining_gt_pixels'] > 0 else 'empty',
            metrics_path=str(path.resolve()), before=before, after=after,
            delta=dict(zip(('tp', 'fp', 'fn', 'f1'), (values(after)-values(before)).tolist()))))
    return records


def analyze(records):
    strata = []
    for support in ('all', 'positive', 'empty'):
        for group in ('全部', '目标类', '地表类'):
            rows = [r for r in records if (support == 'all' or r['support'] == support)
                    and (group == '全部' or r['group'] == group)]
            before = np.asarray([r['before'] for r in rows], dtype=np.int64).reshape(-1, 3)
            after = np.asarray([r['after'] for r in rows], dtype=np.int64).reshape(-1, 3)
            strata.append(dict(support=support, group=group, images=len(rows),
                before=dict(zip(('tp', 'fp', 'fn', 'f1'), values(before.sum(axis=0)).tolist())),
                after=dict(zip(('tp', 'fp', 'fn', 'f1'), values(after.sum(axis=0)).tolist())),
                delta=dict(zip(('tp', 'fp', 'fn', 'f1'), (values(after.sum(axis=0))-values(before.sum(axis=0))).tolist())),
                bootstrap95=paired_interval(before, after)))
    # Empty-GT images have F1=0 under both arms: include FP ranking explicitly.
    extremes = {}
    for metric, beneficial_sign in [('tp', 1), ('fp', -1), ('f1', 1)]:
        ordered = sorted(records, key=lambda r: (-beneficial_sign*r['delta'][metric], r['id']))
        extremes[metric] = dict(benefits=[r for r in ordered if beneficial_sign*r['delta'][metric] > 0][:5],
            losses=[r for r in reversed(ordered) if beneficial_sign*r['delta'][metric] < 0][:5])
    return dict(tasks=len(records), expected=30, status='complete' if len(records) == 30 else 'partial',
        records=records, strata=strata, extremes=extremes)


def markdown(result):
    lines = ['# Fixed paired image diagnostics', '', result['interpretation'], '',
        f"Bootstrap: paired images with replacement, seed {SEED}, {REPLICATES} replicates, percentile 95% intervals. Each fixed dev60 task is one unique image (enforced by the original split). Tune and confirmation are never pooled. F1 differences are percentage points; counts are pooled pixels. Empty-GT F1 is zero by the original scoring convention, so inspect FP. Small strata may have degenerate intervals; intervals are descriptive, not selection tests.", '']
    for name, item in result['experiments'].items():
        lines.extend([f'## {name}: {item["status"]} ({item["tasks"]}/30)', '',
            f"Strict pair: {item['baseline']} → {item['intervention']}, both Top8. Positive means remaining GT > 0.", '',
            '| GT | Group | Images | ΔTP | ΔFP | ΔFN | ΔF1 [95%] |', '|---|---|---:|---:|---:|---:|---|'])
        for row in item['strata']:
            d=row['delta']; ci=row['bootstrap95']; interval=ci['f1'] if ci else None
            lines.append(f"| {row['support']} | {row['group']} | {row['images']} | {d['tp']:.0f} | {d['fp']:.0f} | {d['fn']:.0f} | {d['f1']:.3f} {interval} |")
        lines.extend(['', 'Full TP/FP/FN/F1 intervals and per-image deltas are in summary.json. Extremes are descriptive cases, not newly selected evaluation subsets.', ''])
        for metric, categories in item['extremes'].items():
            for category, rows in categories.items():
                lines.append(f'{metric} {category}:')
                for row in rows:
                    lines.append(f"- [{row['id']}]({row['metrics_path']}): {row['support']}, {row['group']}; ΔTP/FP/FN/F1={row['delta']}")
                if not rows: lines.append('- None observed.')
                lines.append('')
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--night', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Run diagnostic computation through Slurm')
    result=dict(code=str(Path(__file__).resolve()), seed=SEED, replicates=REPLICATES,
        interpretation='Read-only metrics comparison. No images, features, labels or API are loaded. Partial cohorts cannot justify promotion. Bootstrap uncertainty does not select methods or weights; historical exposure of dev60 remains unchanged.',
        experiments={})
    for study, methods in PAIRS.items():
        for split in (('tune', 'confirm') if study == 'r1_endpoint' else ('tune',)):
            records=read_pairs(args.night/study/split, methods)
            result['experiments'][f'{study}/{split}']=dict(analyze(records), baseline=methods[0], intervention=methods[1])
    write_json(args.output/'summary.json', result)
    write_text(args.output/'report.md', markdown(result))


if __name__ == '__main__': main()
