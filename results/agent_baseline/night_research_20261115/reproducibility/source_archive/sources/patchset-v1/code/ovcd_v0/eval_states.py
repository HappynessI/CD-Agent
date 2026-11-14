"""Score the four states (M0, M0+A, M0+A+B, M0+A+B+C) of one agent run.

Reads RUN/agent/<name>/<split>/{masks,traj}; writes RUN/eval/<name>_<split>.json with, per state,
cumulative F1^c / IoU^c / P / R, whole-object miss and false-alarm rates, mean per-image F1, and the
MLLM calls and seconds actually charged by the loop.

    python eval_states.py --name r1_k20 --split test
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np

from common import RUN, evaluate, load_gt, load_mask_png, names, read_json, write_json

STATES = ('M0', 'A', 'B', 'C')


def main(a):
    base = RUN / 'agent' / a.name / a.split
    gts = {n: load_gt(a.split, n) for n in names(a.split)}
    out = dict(name=a.name, split=a.split, states={}, per_image={})
    for st in STATES:
        preds = {n: load_mask_png(base / 'masks' / st / n) for n in gts}
        res, per = evaluate(preds, a.split, gts)
        out['states'][st] = res
        out['per_image'][st] = per
    calls, secs, cfg, stage_calls = [], [], None, {'A': 0, 'B': 0}
    stage_mllm = {'A': 0.0, 'B': 0.0}
    stage_cpu = {'A': 0.0, 'B': 0.0, 'C': 0.0}
    box_try = box_fail = 0
    for n in gts:
        t = read_json(base / 'traj' / (Path(n).stem + '.json'))
        cfg = t['config']
        calls.append(t['stats']['calls'])
        box_try += t['stats']['box_try']
        box_fail += t['stats']['box_fail']
        secs.append(sum(s.get('mllm_seconds', 0) for s in t['steps']))
        for k_, v in t['stats'].get('seconds', {}).items():
            stage_cpu[k_] += v
        for s in t['steps']:
            if s['stage'] in stage_calls:
                stage_calls[s['stage']] += 1 + ('T5' in s['action'])
                stage_mllm[s['stage']] += s.get('mllm_seconds', 0)
    out['config'] = cfg
    nimg = len(gts)
    # Per state, cumulative: MLLM calls and seconds (batched cache time / batch size), plus the
    # CPU replay time of the loop (mask ops, CPU SAM decoding in B and C).
    acc_c = acc_m = acc_cpu = 0.0
    for st, key in (('M0', None), ('A', 'A'), ('B', 'B'), ('C', 'C')):
        if key:
            acc_c += stage_calls.get(key, 0)
            acc_m += stage_mllm.get(key, 0)
            acc_cpu += stage_cpu[key]
        out['states'][st].update(mean_calls=acc_c / nimg, mean_mllm_seconds=acc_m / nimg,
                                 mean_cpu_seconds=acc_cpu / nimg)
    out['cost'] = dict(mean_calls=float(np.mean(calls)), max_calls=int(max(calls)),
                       mean_mllm_seconds=float(np.mean(secs)), total_mllm_seconds=float(sum(secs)),
                       calls_by_stage=stage_calls, mllm_seconds_by_stage=stage_mllm,
                       cpu_seconds_by_stage=stage_cpu, box_try=box_try, box_fail=box_fail,
                       box_fail_rate=box_fail / max(box_try, 1),
                       note='mllm seconds = batched (16) cache wall time / batch size; M-C-I init ~7.2 s/image GPU not included')
    write_json(RUN / 'eval' / f'{a.name}_{a.split}.json', out)
    for st in STATES:
        r = out['states'][st]
        print(st, *(f"{k}={r[k]:.4f}" for k in ('f1', 'iou', 'precision', 'recall', 'miss_rate',
                                                 'false_alarm_rate', 'mean_image_f1')))
    print(out['cost'])


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--split', choices=['val', 'test'], required=True)
    main(p.parse_args())
