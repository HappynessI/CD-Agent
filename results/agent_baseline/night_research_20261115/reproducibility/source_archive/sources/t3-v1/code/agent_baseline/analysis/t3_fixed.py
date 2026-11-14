"""T3.1 fixed route: union of proposals whose change-vector similarity to the exemplar >= threshold.

Same proposals, features and similarity as probe_t3 (SAM ViT-H both dates, DINOv2 T2 - T1 pooled
difference, cosine to the exemplar's difference). Pixel F1 pools TP/FP/FN over all tasks; the
exemplar's own pixels are removed from both prediction and truth, so self-retrieval earns nothing.
The threshold is chosen on the 60 development images by pooled F1 (ties -> smaller threshold) and
applied once to the 128 report images. GPU, no API; reuses the T1.3 vision cache.
"""
import json
from pathlib import Path
import numpy as np
from common import write_json, save_mask_png
from agent_baseline.backends_t13 import WorkerPool
from agent_baseline.tools_t13 import Features
from .make_t3_tasks import make_tasks
from .probe_t3 import task_scores
from .probe_t13_common import parser, selected, pair_backend, backend_config, inventories
from .second_labels import load_labels

THRESHOLDS = [round(x, 2) for x in np.arange(0.0, 1.0, 0.05)]


def f1(c):
    tp, fp, fn = c
    return dict(f1_c=200 * tp / max(1, 2 * tp + fp + fn), precision=100 * tp / max(1, tp + fp),
                recall=100 * tp / max(1, tp + fn), tp=int(tp), fp=int(fp), fn=int(fn))


def task_counts(masks, scores, task, shape, thresholds, save=None):
    def dense(idx_list):
        m = np.zeros(int(np.prod(shape)), bool)
        for idx in idx_list:
            m[np.asarray(idx, np.int64)] = True
        return m
    exemplar = dense([task['instances'][task['exemplar_index']]])
    truth = dense(task['instances']) & ~exemplar
    flat = masks.reshape(len(masks), -1)
    out = {}
    for t in thresholds:
        keep = np.asarray(scores) >= t
        pred = (flat[keep].any(axis=0) if keep.any() else np.zeros(flat.shape[1], bool)) & ~exemplar
        out[t] = [int((pred & truth).sum()), int((pred & ~truth).sum()), int((~pred & truth).sum())]
        if save is not None and len(thresholds) == 1:
            save_mask_png(save, pred.reshape(shape))
    return out


def main():
    p = parser(__doc__)
    p.add_argument('--settings', type=Path, help='Development report; required for val')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mask-dir', type=Path, help='With a frozen threshold: save each task prediction (exemplar removed) as <task id>.png')
    p.add_argument('--task-ids', type=Path, help='Manifest JSONL; score only these task ids (threshold still frozen from dev)')
    a = p.parse_args()
    names = selected(a)
    only = {json.loads(l)['id'] for l in a.task_ids.read_text().splitlines() if l.strip()} if a.task_ids else None
    frozen = json.loads(a.settings.read_text()) if a.settings else None
    if a.phase == 'val' and (frozen is None or frozen['phase'] != 'dev' or frozen['names'] != inventories(a.run_root)['dev']):
        p.error('val requires the complete development report')
    thresholds = [frozen['selected']] if frozen else THRESHOLDS
    config = backend_config(a, frozen)
    pool = WorkerPool(config, a.cache)
    totals = {t: np.zeros(3, np.int64) for t in thresholds}
    tasks_n = 0
    try:
        for name in names:
            labels = load_labels(a.dataset_root, name)
            tasks = [t for t in make_tasks(*labels, name) if only is None or t['id'] in only]
            if not tasks:
                continue
            backend = pair_backend(pool, a, name)
            masks = np.concatenate([backend('segment_everything', time=t) for t in ('t1', 't2')], axis=0)
            f_1, f_2 = [Features(backend('extract_features', time=t), labels[0].shape, t) for t in ('t1', 't2')]
            for task in tasks:
                counts = task_counts(masks, task_scores(masks, f_1, f_2, task), task, labels[0].shape, thresholds,
                                     a.mask_dir / (task['id'] + '.png') if a.mask_dir else None)
                for t, c in counts.items():
                    totals[t] += c
                tasks_n += 1
            print(name, flush=True)
    finally:
        pool.close()
    scores = {str(t): f1(c) for t, c in totals.items()}
    best = max(thresholds, key=lambda t: (round(scores[str(t)]['f1_c'], 10), -t)) if not frozen else thresholds[0]
    report = dict(phase=a.phase, names=names, tasks=tasks_n, backend_config=config, scores=scores, selected=best,
                  rule='pooled pixel F1 over tasks, exemplar pixels removed from prediction and truth; dev picks threshold')
    write_json(a.output, report)
    print(json.dumps(dict(phase=a.phase, tasks=tasks_n, selected=best, score=scores[str(best)])))


if __name__ == '__main__':
    main()
