"""Validation-only selection, stage by stage (M2 -> M3 -> M4), for each budget k.

  A: tau_a over a grid (tau_a = 0 keeps M0); a_share and a_order are fixed inputs (from M1)
  B: with A fixed, tau_b over a grid (tau_b > 1 adds nothing)
  C: with A, B fixed, c_mode none vs box with IoU gates (gate 0 = always replace)
Every grid point is scored by cumulative F1^c on val and saved, so "no gain" or "worse" outcomes
stay visible.  Writes RUN/select/<round>/val.json.

    python select_val.py --round r1 --tag r1 --a-share 0.5 --a-order score_asc --procs 8
"""
from __future__ import annotations
import argparse
import json
from dataclasses import asdict, replace
from multiprocessing import Pool

import numpy as np

from common import RUN, load_gt, load_m0, names, scores, write_json
from agent import Config, Image, run

TAU = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
GATES = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9]
CLIP_R = [0, 1, 2, 4]
CLIP_COVER = [0.0, 0.3, 0.5]
_P = None


def work(args):
    split, name, cfgs = args
    import torch
    torch.set_num_threads(1)
    global _P
    if _P is None:
        from sam_tools import Prompter
        _P = Prompter('cpu')
    m0 = load_m0(cfgs[0].m0, split, name)
    gt = load_gt(split, name)
    img = Image(split, name, m0, cfgs[0].tag, gt=gt, prompter=_P)
    out = []
    for cfg in cfgs:
        snap, _, stats = run(img, cfg, log=False)
        c = {}
        for st, m in snap.items():
            c[st] = [int((m & gt).sum()), int((m & ~gt).sum()), int((~m & gt).sum())]
        out.append(dict(conf=c, stats=stats))
    return out


def grid(pool, split, cfgs, key):
    res = pool.map(work, [(split, n, cfgs) for n in names(split)])
    rows = []
    for i, cfg in enumerate(cfgs):
        tp = sum(r[i]['conf'][key][0] for r in res)
        fp = sum(r[i]['conf'][key][1] for r in res)
        fn = sum(r[i]['conf'][key][2] for r in res)
        calls = [r[i]['stats']['calls'] for r in res]
        bt = sum(r[i]['stats']['box_try'] for r in res)
        bf = sum(r[i]['stats']['box_fail'] for r in res)
        rows.append(dict(config=asdict(cfg), state=key, tp=tp, fp=fp, fn=fn, **scores(dict(tp=tp, fp=fp, fn=fn)),
                         mean_calls=float(np.mean(calls)), box_try=bt, box_fail=bf))
    return rows


def best(rows):
    return max(rows, key=lambda r: (round(r['f1'], 6), -r['mean_calls']))


def main(a):
    global _P
    from sam_tools import Prompter
    _P = Prompter('cpu')  # decoder only; forked workers share it
    import gc; gc.collect()
    out = dict(args=vars(a), budgets={})
    with Pool(a.procs) as pool:
        for k in a.ks:
            base = Config(tag=a.tag, k=k, a_share=a.a_share, a_order=a.a_order, b_order='queue',
                          use_t5=bool(a.use_t5), b_max=a.b_max, m0=a.m0)
            orders = a.a_orders or [a.a_order]
            if a.c_fixed:  # C-aware selection: A and B are scored after the fixed C (round 9)
                cf = json.loads(a.c_fixed)
                base = replace(base, **cf)
                ra = grid(pool, 'val', [replace(base, a_order=o, tau_a=t, stages='AC') for o in orders for t in TAU], 'C')
                ba = best(ra)['config']
                base = replace(base, a_order=ba['a_order'])
                rb = grid(pool, 'val', [replace(base, tau_a=ba['tau_a'], tau_b=t, stages='ABC') for t in TAU[1:] + [1.01]], 'C')
                bb = best(rb)['config']
                cc = [replace(base, tau_a=bb['tau_a'], tau_b=bb['tau_b'], stages='ABC', c_mode='none')]
                cc += [replace(base, tau_a=bb['tau_a'], tau_b=bb['tau_b'], stages='ABC', c_mode='sam3clip', c_radius=r, c_cover=c, c_tile=a.c_tile, c_conf=q)
                       for q in a.c_confs for r in CLIP_R for c in CLIP_COVER]
                rc = grid(pool, 'val', cc, 'C')
                out['budgets'][str(k)] = dict(A=ra, B=rb, C=rc, chosen=best(rc)['config'])
                print(k, 'A', best(ra)['f1'], 'B', best(rb)['f1'], 'C', best(rc)['f1'], best(rc)['config'], flush=True)
                write_json(RUN / 'select' / a.round / 'val.json', out)
                continue
            ra = grid(pool, 'val', [replace(base, a_order=o, tau_a=t, stages='A') for o in orders for t in TAU], 'A')
            ba = best(ra)['config']
            base = replace(base, a_order=ba['a_order'])
            rb = grid(pool, 'val', [replace(base, tau_a=ba['tau_a'], tau_b=t, stages='AB') for t in TAU[1:] + [1.01]], 'B')
            bb = best(rb)['config']
            cc = [replace(base, tau_a=bb['tau_a'], tau_b=bb['tau_b'], stages='ABC', c_mode='none')]
            cc += [replace(base, tau_a=bb['tau_a'], tau_b=bb['tau_b'], stages='ABC', c_mode='box', c_gate=g) for g in GATES]
            cc += [replace(base, tau_a=bb['tau_a'], tau_b=bb['tau_b'], stages='ABC', c_mode='sam3clip', c_radius=r, c_cover=c, c_tile=a.c_tile, c_conf=q)
                   for q in a.c_confs for r in CLIP_R for c in CLIP_COVER] if a.c_tile >= 0 else []
            rc = grid(pool, 'val', cc, 'C')
            if a.c_scores:  # second C step on top of the best first step: initializer-score filter
                bc = best(rc)['config']
                rc += grid(pool, 'val', [replace(base, **{**bc, 'c_score': q}) for q in a.c_scores], 'C')
            out['budgets'][str(k)] = dict(A=ra, B=rb, C=rc, chosen=best(rc)['config'])
            print(k, 'A', best(ra)['f1'], 'B', best(rb)['f1'], 'C', best(rc)['f1'], best(rc)['config'], flush=True)
            write_json(RUN / 'select' / a.round / 'val.json', out)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--round', required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--a-share', type=float, required=True)
    p.add_argument('--a-order', required=True)
    p.add_argument('--ks', type=int, nargs='+', default=[10, 20, 50])
    p.add_argument('--procs', type=int, default=8)
    p.add_argument('--use-t5', type=int, default=1)
    p.add_argument('--b-max', type=int, default=20)
    p.add_argument('--m0', default='mci', choices=['mci', 'imc', 'rea'])
    p.add_argument('--c-tile', type=int, default=-1, help='also try C = clip to SAM3 support from this SAM3 run (-1: off)')
    p.add_argument('--c-confs', type=float, nargs='+', default=[0.5], help='SAM3 score thresholds for the clip support')
    p.add_argument('--c-fixed', help='JSON of C fields; if given, A and B are selected on the state after this C')
    p.add_argument('--c-scores', type=float, nargs='*', default=[], help='also try keeping M0 pixels only above these initializer scores')
    p.add_argument('--a-orders', nargs='+', help='also select the stage-A order on val among these')
    main(p.parse_args())
