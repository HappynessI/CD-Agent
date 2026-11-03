"""Per-image error attribution of a validation run against the fixed SAM3 I-M-C route, with what-if and bootstrap rows.

Reads trajectories and labels only; for the outer loop (README 10.1), never for the agent.
    python diagnose.py --config json_qwen37plus_r0 --task T1.1xC0 --split val
"""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = 'agent_baseline'
    import agent_baseline

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import numpy as np
from common import load_gt, load_mask_png, names, confusion, image_f1, scores, read_jsonl, write_json

RUN = Path(os.environ.get('AGENT_RUN', '/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run'))
EDIT = {'remove_region', 'add_region', 'refine_region', 'morph', 'combine'}


def total_f1(rows, key):
    tot = {k: sum(r[key][k] for r in rows) for k in ('tp', 'fp', 'fn')}
    return scores(tot)['f1'] * 100


def bootstrap(rows, a, b, n=2000, seed=0):
    """Paired image-level bootstrap of F1^c(a) - F1^c(b)."""
    rng = np.random.default_rng(seed)
    arr = {k: np.array([[r[a][x], r[b][x]] for r in rows]) for k, x in (('tp', 'tp'), ('fp', 'fp'), ('fn', 'fn'))}
    out = []
    for _ in range(n):
        idx = rng.integers(0, len(rows), len(rows))
        t, p, f = (arr[k][idx].sum(0) for k in ('tp', 'fp', 'fn'))
        d = 2 * t / np.maximum(2 * t + p + f, 1)
        out.append(100 * (d[0] - d[1]))
    return [float(np.percentile(out, q)) for q in (2.5, 50, 97.5)]


def tag(row):
    """Primary cause, first match wins; rules are about what the agent did, not about scores it saw."""
    if row['submitted_by'] == 'empty':
        return 'no_answer'
    if row['submitted_by'] in ('budget', 'no_submit', 'error'):
        return 'stopped_by_budget'
    if row['route'] != 'imc_like':
        return 'skipped_matching'
    if row['thresholds'] and row['thresholds'] != [0.5]:
        return 'changed_threshold'
    if row['edits']:
        return 'edited'
    return 'plain_route'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--task', required=True)
    p.add_argument('--split', default='val')
    a = p.parse_args()
    base = RUN / a.config / a.task / a.split
    rows = []
    for name in names(a.split):
        stem = Path(name).stem
        d = base / stem
        final = json.loads((d / 'final.json').read_text())
        gt = load_gt(a.split, name)
        pred = load_mask_png(d / final['mask_path'])
        imc = load_mask_png(RUN / 'init' / 'sam3_imc' / a.split / name)
        calls = [c for s in read_jsonl(d / 'steps.jsonl') for c in s['calls'] if c['ok']]
        used = [c['tool'] for c in calls]
        route = 'imc_like' if used.count('segment_text') >= 2 and 'match_instances' in used else 'other'
        thr = sorted({(c['args'] or {}).get('score_thr', 0.5) for c in calls if c['tool'] == 'segment_text'})
        texts = sorted({(c['args'] or {}).get('text') for c in calls if c['tool'] == 'segment_text'})
        tiles = sorted({(c['args'] or {}).get('tile', 512) for c in calls if c['tool'] == 'segment_text'})
        c_pred, c_imc = confusion(pred, gt), confusion(imc, gt)
        rows.append(dict(image=stem, submitted_by=final['submitted_by'], rounds=final['planning_rounds'],
            route=route, thresholds=thr, texts=texts, tiles=tiles, views=used.count('view'), asks=used.count('ask'),
            edits=sum(t in EDIT for t in used), removes=used.count('remove_region'),
            gt_px=int(gt.sum()), pred=c_pred, imc=c_imc, f1=image_f1(pred, gt), f1_imc=image_f1(imc, gt),
            same_as_imc=bool(np.array_equal(pred, imc))))
    for r in rows:
        r['cause'] = tag(r)
    out = dict(config=a.config, task=a.task, f1=total_f1(rows, 'pred'), f1_imc=total_f1(rows, 'imc'),
               bootstrap_vs_imc=bootstrap(rows, 'pred', 'imc'), causes={})
    for cause, n in Counter(r['cause'] for r in rows).items():
        sub = [r for r in rows if r['cause'] == cause]
        # What-if: these images keep the fixed route's mask, the rest stay as they are.
        swapped = [dict(r, pred=r['imc']) if r['cause'] == cause else r for r in rows]
        out['causes'][cause] = dict(images=n, f1_if_fixed_route_here=total_f1(swapped, 'pred'),
            loss=total_f1(swapped, 'pred') - out['f1'], dtp=sum(r['pred']['tp'] - r['imc']['tp'] for r in sub),
            dfp=sum(r['pred']['fp'] - r['imc']['fp'] for r in sub), dfn=sum(r['pred']['fn'] - r['imc']['fn'] for r in sub),
            examples=[r['image'] for r in sorted(sub, key=lambda r: r['f1'] - r['f1_imc'])[:5]])
    out['same_as_imc'] = sum(r['same_as_imc'] for r in rows)
    out['texts'] = Counter(t for r in rows for t in r['texts']).most_common(10)
    out['thresholds'] = Counter(str(t) for r in rows for t in r['thresholds']).most_common(10)
    out['tiles'] = Counter(str(t) for r in rows for t in r['tiles']).most_common()
    out['images'] = rows
    write_json(base / 'diagnosis.json', out)
    print(json.dumps({k: v for k, v in out.items() if k != 'images'}, ensure_ascii=False, indent=1))
    worst = sorted(rows, key=lambda r: (r['pred']['tp'] - r['imc']['tp']) - (r['pred']['fp'] - r['imc']['fp']) * 0.5)[:12]
    for r in worst:
        print(r['image'], r['cause'], r['submitted_by'], 'r', r['rounds'], 'F1', round(r['f1'], 3), 'imc', round(r['f1_imc'], 3),
              'dTP', r['pred']['tp'] - r['imc']['tp'], 'dFP', r['pred']['fp'] - r['imc']['fp'], 'dFN', r['pred']['fn'] - r['imc']['fn'], r['thresholds'], r['texts'])


if __name__ == '__main__':
    main()
