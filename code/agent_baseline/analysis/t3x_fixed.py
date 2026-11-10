"""External-example fixed route: SAM proposals -> exemplar score >= threshold -> union.

T3.2 uses T2 appearance. T3.3 uses T2-T1 vectors. T3.4 subtracts negative
cosine from positive cosine. Tune one threshold per type on complete dev only.
Report uses those thresholds once and saves every task mask; no API.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from common import save_mask_png, write_json
from agent_baseline.backends_t13 import WorkerPool
from agent_baseline.examples_t3x import load_examples, example_vector, combine_scores, INSTRUCTIONS
from agent_baseline.instances import from_dense
from agent_baseline.tasks import read_tasks, load_pair
from agent_baseline.tools_t13 import Features, pooled, cosine
from .make_t3x_tasks import split_names
from .probe_t13_common import RUN
from .t3_fixed import f1
from .t3_metrics import load, counts

THRESHOLDS = {k: [round(float(v), 2) for v in np.linspace(-2 if k == 'T3.4' else -1,
    2 if k == 'T3.4' else 1, 81 if k == 'T3.4' else 41)] for k in INSTRUCTIONS}


def proposal_scores(masks, a, b, examples, mode):
    rows = from_dense(masks, np.ones(len(masks)))
    candidate = pooled(b, rows) - pooled(a, rows) if mode == 'change' else pooled(b if mode == 't2' else a, rows)
    positive = cosine(candidate, example_vector(examples[0], mode))
    negative = cosine(candidate, example_vector(examples[1], mode)) if len(examples) == 2 else None
    return combine_scores(positive, negative)


def predict(masks, scores, threshold, shape):
    keep = np.asarray(scores) >= threshold
    return masks[keep].any(axis=0) if keep.any() else np.zeros(shape, bool)


def task_counts(masks, scores, truth, thresholds):
    return {t: counts(predict(masks, scores, t, truth.shape), truth) for t in thresholds}


def validate_inputs(args, rows):
    inventory = json.loads((args.tasks.parent / 'inventory.json').read_text())
    groups = split_names(args.run_root)
    if (inventory['phase'] != args.phase or sorted(inventory['names']) != sorted(groups[args.phase])
            or inventory['example_names'] != groups['rest'] or inventory['total'] != len(rows)):
        raise ValueError('Require the complete manifest and matching dev/report/rest inventories')
    if not rows or any(r['task_type'] not in INSTRUCTIONS for r in rows):
        raise ValueError('Require external-example tasks')
    if any(Path(r['image_t1']).name not in groups[args.phase] or
           Path(r['image_t2']).name != Path(r['image_t1']).name for r in rows):
        raise ValueError('Evaluation image must belong to the selected split')
    generation = {k: inventory[k] for k in ('seed', 'min_area', 'max_area', 'example_names', 'rule')}
    config = json.loads(args.vision_config.read_text())
    frozen = json.loads(args.settings.read_text()) if args.settings else None
    if args.phase == 'val' and (frozen is None or args.mask_dir is None):
        raise ValueError('Report requires development --settings and --mask-dir')
    if frozen is not None:
        if (frozen['phase'] != 'dev' or not frozen['complete'] or
                sorted(frozen['names']) != sorted(groups['dev']) or
                frozen['generation'] != generation or frozen['backend_config'] != config):
            raise ValueError('Require complete development settings with identical generation and backends')
        if any(r['task_type'] not in frozen['selected'] for r in rows):
            raise ValueError('A report task type has no development threshold')
    return inventory, generation, config, frozen


def evaluate(args):
    rows = read_tasks(args.tasks)
    inventory, generation, config, frozen = validate_inputs(args, rows)
    kinds = sorted({r['task_type'] for r in rows})
    thresholds = {k: [frozen['selected'][k]] if frozen else THRESHOLDS[k] for k in kinds}
    totals = {k: {t: np.zeros(3, np.int64) for t in thresholds[k]} for k in kinds}
    per_task = {}
    by_image = defaultdict(list)
    for row in rows:
        by_image[(row['image_t1'], row['image_t2'])].append(row)
    pool = WorkerPool(config, args.cache)
    try:
        for paths, tasks in sorted(by_image.items()):
            shape = load_pair(tasks[0])[0].shape[:2]
            backend = pool.for_pair(paths)
            masks = np.concatenate([backend('segment_everything', time=t) for t in ('t1', 't2')], axis=0)
            needed = ('t1', 't2') if any(r['task_type'] != 'T3.2' for r in tasks) else ('t2',)
            features = {t: Features(backend('extract_features', time=t), shape, t) for t in needed}
            for row in tasks:
                kind = row['task_type']
                mode = 't2' if kind == 'T3.2' else 'change'
                examples = load_examples(row['examples'], pool)
                scores = proposal_scores(masks, features.get('t1'), features['t2'], examples, mode)
                truth = load(row['gt_mask'])  # scoring only, never a feature input
                if truth.shape != shape:
                    raise ValueError('Evaluation truth shape differs from image pair')
                result = task_counts(masks, scores, truth, thresholds[kind])
                for t, c in result.items():
                    totals[kind][t] += c
                per_task[row['id']] = dict(task_type=kind, counts={str(t): c for t, c in result.items()})
                if frozen and args.mask_dir is not None:
                    save_mask_png(args.mask_dir / (row['id'] + '.png'), predict(masks, scores, thresholds[kind][0], shape))
            print(Path(paths[0]).name, flush=True)
    finally:
        pool.close()
    scores = {k: {str(t): f1(c) for t, c in values.items()} for k, values in totals.items()}
    selected = {k: (thresholds[k][0] if frozen else max(thresholds[k],
        key=lambda t: (round(scores[k][str(t)]['f1_c'], 10), -t))) for k in kinds}
    report = dict(phase=args.phase, complete=True, names=inventory['names'], tasks=len(rows),
        generation=generation, backend_config=config, scores=scores, selected=selected, per_task=per_task,
        rule='Pooled pixel F1 per task type; dev only selects thresholds, ties choose smaller; all evaluation pixels retained; T3.4 positive cosine minus negative cosine',
        settings=str(args.settings.resolve()) if args.settings else None)
    write_json(args.output, report)
    print(json.dumps(dict(phase=args.phase, tasks=len(rows), selected=selected)))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', type=Path, required=True)
    p.add_argument('--phase', choices=['dev', 'val'], required=True)
    p.add_argument('--run-root', type=Path, default=RUN)
    p.add_argument('--vision-config', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--settings', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mask-dir', type=Path)
    evaluate(p.parse_args())


if __name__ == '__main__':
    main()
