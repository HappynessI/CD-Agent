"""Verifier selection on a listed pool (plan v2, part B). Labels only score.

For each variant against L1: the F1 of the rule "delete what is answered no" and the AUC of its score over pure-change
against pure-false components, each with an image-paired bootstrap. Votes are combined from answer files.
Usage: verifier_select.py SPLIT NAMESFILE INIT TAG OUT
"""
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from PIL import Image
sys.path[:0] = [str(Path(__file__).resolve().parents[2] / 'ovcd_v0'), str(Path(__file__).resolve().parents[2])]
import agent_baseline
from common import load_gt, components
from agent_baseline.ask_eval import auc

R = Path(os.environ.get('AGENT_RUN', '/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run'))
SPLIT, NAMES, INIT, TAG, OUT = sys.argv[1:6]
todo = Path(NAMES).read_text().split()


def table(name):
    stem = Path(name).stem
    gt = load_gt(SPLIT, name)
    m = np.asarray(Image.open(R / 'init' / INIT / SPLIT / f'{stem}.png')) > 0
    labels, comps = components(m)
    rows = [(f"{stem}:{c['id']}", int(((labels == c['id']) & gt).sum()), int(c['area'])) for c in comps]
    return stem, int(gt.sum()), rows


def load(variant):
    out = {}
    path = R / 'ask_eval' / TAG / f'{variant}.jsonl'
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        row = json.loads(line)
        out[row['uid']] = row
    return out


CHANGE_TYPES = ('new_building', 'demolished', 'rebuilt')


def single(rows):
    """A reply that opens with a type instead of yes or no still says what it means."""
    out = {}
    for u, r in rows.items():
        answer = r.get('answer')
        if answer in ('yes', 'no'):
            out[u] = (answer == 'yes', r.get('p_yes'))
        elif answer in CHANGE_TYPES or answer in ('no_change', 'shadow', 'seasonal_or_color', 'misregistration', 'vehicle_or_temporary', 'non_building_change'):
            out[u] = (answer in CHANGE_TYPES, None)
    return out


def vote(*sets):
    """Majority of the answers; the score is the mean over the answers (of p_yes where read, else 0/1)."""
    uids = set.intersection(*(set(s) for s in sets))
    out = {}
    for u in uids:
        says = [s[u][0] for s in sets]
        vals = [s[u][1] if s[u][1] is not None else float(s[u][0]) for s in sets]
        out[u] = (sum(says) * 2 > len(says), float(np.mean(vals)))
    return out


def any_no(a, b):
    return {u: (a[u][0] and b[u][0], None) for u in set(a) & set(b)}


def any_yes(a, b):
    return {u: (a[u][0] or b[u][0], None) for u in set(a) & set(b)}


def main():
    with Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', 8))) as pool:
        images = pool.map(table, todo, chunksize=4)
    files = {p.stem: single(load(p.stem)) for p in sorted((R / 'ask_eval' / TAG).glob('*.jsonl'))}
    variants = dict(files)
    if all(k in files for k in ('v2_s1', 'v2_s2', 'v2_s3')):
        variants['V2 vote of 3 samples'] = vote(files['v2_s1'], files['v2_s2'], files['v2_s3'])
    if all(k in files for k in ('v2_close', 'v2_wide', 'f37_corr')):
        variants['V2 vote of close, wide, both'] = vote(files['v2_close'], files['v2_wide'], files['f37_corr'])
    if 'v3' in files and 'f37_corr' in files:
        variants['V3 delete if either says no'] = any_no(files['f37_corr'], files['v3'])
        variants['V3 delete if both say no'] = any_yes(files['f37_corr'], files['v3'])
        both = set(files['f37_corr']) & set(files['v3'])
        agree = sum(files['f37_corr'][u][0] == files['v3'][u][0] for u in both)
        print('flash and plus agree on', agree, 'of', len(both), flush=True)
    L1 = variants['f37_corr']
    stems = [i[0] for i in images]
    def vectors(answers):
        """Per image rows of (tp kept, fp kept, tp lost); components without an answer stay."""
        out = np.zeros((len(images), 3))
        for k, (stem, g, rows) in enumerate(images):
            tp = fp = 0
            for u, t, a in rows:
                if answers.get(u, (True, None))[0]:
                    tp += t
                    fp += a - t
            out[k] = tp, fp, g - tp
        return out
    def f1(v):
        s = v.sum(0)
        return 200 * s[0] / max(1, 2 * s[0] + s[1] + s[2])
    def scored(answers):
        """Per image: score, purity class, area for the components this variant answered."""
        out = []
        for stem, g, rows in images:
            S, Y = [], []
            for u, t, a in rows:
                if u in answers:
                    says, p = answers[u]
                    S.append(p if p is not None else float(says))
                    Y.append(t / max(a, 1))
            out.append((np.array(S), np.array(Y)))
        return out
    def auc_of(parts, chosen):
        S = np.concatenate([parts[i][0] for i in chosen])
        Y = np.concatenate([parts[i][1] for i in chosen])
        return auc(S[Y >= .5], S[Y < .5])
    rng = np.random.default_rng(0)
    base = vectors(dict())
    report = dict(images=len(images), components=sum(len(i[2]) for i in images), base_f1=round(float(f1(base)), 2), variants={})
    picks = [rng.integers(0, len(images), len(images)) for _ in range(2000)]
    # A pool is scored only where every image of it has answers, so a partly finished variant is reported on what it covers.
    for name, answers in variants.items():
        covered = [k for k, (_, _, rows) in enumerate(images) if rows and all(u in answers for u, _, _ in rows)] + [k for k, (_, _, rows) in enumerate(images) if not rows]
        if len(covered) < .9 * len(images):
            sets = {u.split(':')[0] for u in answers}
            print(name, 'skipped: covers', len(sets), 'images', flush=True)
            continue
        v, l, p, q = vectors(answers), vectors(L1), scored(answers), scored(L1)
        share = float(np.mean([a[0] for a in answers.values()]))
        read = float(np.mean([a[1] is not None for a in answers.values()]))
        row = dict(n=len(answers), yes_rate=round(share, 3), p_yes_read=round(read, 3), rule_f1=round(float(f1(v)), 2),
                   auc=round(auc_of(p, range(len(images))), 4), lost_tp=int(base[:, 0].sum() - v[:, 0].sum()), removed_fp=int(base[:, 1].sum() - v[:, 1].sum()))
        # The best any fixed threshold on the score can do, with the threshold chosen on this same pool.
        sweep = {}
        for thr in (.02, .05, .1, .2, .3, .5, .7, .9):
            kept = {u: (((p if p is not None else float(says)) >= thr), None) for u, (says, p) in answers.items()}
            sweep[str(thr)] = round(float(f1(vectors(kept))), 2)
        row['threshold_f1'] = sweep
        if name != 'f37_corr':
            df, da = [], []
            for chosen in picks:
                df.append(f1(v[chosen]) - f1(l[chosen]))
                da.append(auc_of(p, chosen) - auc_of(q, chosen))
            row.update(rule_f1_minus_L1=round(float(f1(v) - f1(l)), 2), rule_ci=[round(float(x), 2) for x in np.percentile(df, [2.5, 97.5])],
                       auc_minus_L1=round(float(row['auc'] - auc_of(q, range(len(images)))), 4), auc_ci=[round(float(x), 4) for x in np.percentile(da, [2.5, 97.5])])
            row['keep'] = bool(row['rule_ci'][0] > 0 and row['auc_ci'][0] > 0)
        report['variants'][name] = row
        print(name, json.dumps(row, ensure_ascii=False), flush=True)
    Path(OUT).write_text(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
