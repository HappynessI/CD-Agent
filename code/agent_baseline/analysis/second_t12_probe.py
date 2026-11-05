"""Compute-node SAM3 class segmentation probe; no Agent or paid API.

Scores only pixels with nonzero semantic labels, separately for each date.
These are changed regions. Unchanged land cover has no semantic ground truth;
this probe cannot estimate full-image precision, recall or IoU.
"""
import argparse
from pathlib import Path
if __package__ in (None, ''):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import agent_baseline
import numpy as np
from common import write_json
from agent_baseline.backends import SegmentBackend
from agent_baseline.instances import to_mask
from agent_baseline.fixed_routes import RUN
from agent_baseline.analysis.second_labels import DATASET, check_encoding, load_labels, pixel_counts
from agent_baseline.analysis.second_prepare import local_segmenter

CLASSES = {4: 'water', 3: 'tree', 1: 'low vegetation', 2: 'non-vegetated ground', 6: 'playground', 5: 'building'}
LIMITATION = '只在非零语义标签（变化区域）评测；未变化区域没有地类真值，不能视作全图分割指标。'


def annotated_counts(pred, label, class_id):
    annotated = label != 0
    return pixel_counts(pred & annotated, (label == class_id) & annotated)


def probe(dataset, names, run, worker=None, segmenter=None):
    check_encoding(dataset)
    total = {time: {text: np.zeros(3, np.int64) for text in CLASSES.values()} for time in ('t1', 't2')}
    per_image = {}
    for name in names:
        paths = [str(Path(dataset).resolve() / part / name) for part in ('img1', 'img2')]
        labels = load_labels(dataset, name)
        backend = SegmentBackend('val', name, worker, .85, image_paths=paths)
        per_image[name] = {}
        for index, time in enumerate(('t1', 't2')):
            if labels[index].shape != (512, 512):
                raise ValueError('This probe expects 512 x 512 labels')
            per_image[name][time] = {}
            for class_id, text in CLASSES.items():
                if segmenter is not None:
                    segmenter.infer('val', name, time, text, 512, conf=.1, pixel_thr=.85, image_path=paths[index])
                rows, _ = backend(time, text, 512, .5)
                pred = to_mask(rows, labels[index].shape)
                c = annotated_counts(pred, labels[index], class_id)
                total[time][text] += c
                per_image[name][time][text] = dict(zip(('tp', 'fp', 'fn'), c))
        print(name, flush=True)
    scores = {}
    for time, classes in total.items():
        scores[time] = {}
        for text, c in classes.items():
            tp, fp, fn = map(int, c)
            ratio = lambda n, d: n / d if d else None
            scores[time][text] = dict(tp=tp, fp=fp, fn=fn, precision=ratio(tp, tp+fp),
                                     recall=ratio(tp, tp+fn), iou=ratio(tp, tp+fp+fn))
    return dict(limitation=LIMITATION, count=len(names), tile=512, pixel_thr=.85, score_thr=.5,
                prompts={str(k): v for k, v in CLASSES.items()}, scores=scores, per_image=per_image)


def main():
    import os
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root', type=Path, default=DATASET)
    p.add_argument('--names-file', type=Path, required=True)
    p.add_argument('--run-root', type=Path, default=RUN)
    p.add_argument('--output', type=Path, required=True)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--sam3-worker')
    mode.add_argument('--local', action='store_true')
    a = p.parse_args()
    os.environ['AGENT_RUN'] = str(a.run_root)
    report = probe(a.dataset_root, a.names_file.read_text().split(), a.run_root,
                   a.sam3_worker, local_segmenter(a.run_root) if a.local else None)
    write_json(a.output, report)
    lines = [LIMITATION, '', '| 时期 | 类别 | P | R | IoU |', '|---|---|---:|---:|---:|']
    for time, classes in report['scores'].items():
        for text, s in classes.items():
            values = ['—' if s[k] is None else f'{100*s[k]:.2f}' for k in ('precision', 'recall', 'iou')]
            lines.append('| ' + ' | '.join([time, text, *values]) + ' |')
    a.output.with_suffix('.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
