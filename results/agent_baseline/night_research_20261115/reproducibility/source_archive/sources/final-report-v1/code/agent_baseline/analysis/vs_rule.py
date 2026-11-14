"""Plan v2 report: the Agent against the deletion rule, per configuration. Labels only score.

The rule keeps the components of the fixed-route mask that the offline verifier pool does not answer no to. A deviation
is a pixel where the Agent's mask and the rule's mask differ; its net gain is the F1 difference it causes.
Usage: vs_rule.py SPLIT INIT ASKDIR RULE CONFIG_PREFIX... (each prefix has _s1.._s3 runs)
"""
import json
import os
import re
import sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from PIL import Image
sys.path[:0] = [str(Path(__file__).resolve().parents[2] / 'ovcd_v0'), str(Path(__file__).resolve().parents[2])]
import agent_baseline
from common import names, load_gt, components

R = Path(os.environ.get('AGENT_RUN', '/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run'))
SPLIT, INIT, ASK, RULE = sys.argv[1:5]
PREFIXES = sys.argv[5:]
TASKS = ('T1.1xC0', 'T5.1')
REJECT = re.compile(r'\b(reject|rejected|rejection|was not accepted|not accepted|refused)\b', re.I)
rows = {json.loads(l)['uid']: json.loads(l) for l in open(R / 'ask_eval' / ASK / f'{RULE}.jsonl')}
todo = names(SPLIT)


def counts(m, gt):
    return np.array([(m & gt).sum(), (m & ~gt).sum(), (~m & gt).sum()], float)


def base(name):
    stem = Path(name).stem
    gt = load_gt(SPLIT, name)
    init = np.asarray(Image.open(R / 'init' / INIT / SPLIT / f'{stem}.png')) > 0
    labels, comps = components(init)
    no = [c['id'] for c in comps if rows.get(f"{stem}:{c['id']}", {}).get('answer') == 'no']
    rule = init & ~np.isin(labels, no)
    return stem, gt, init, labels, comps, set(no), rule


def session(arg):
    prefix, n, task, name = arg
    stem, gt, init, labels, comps, no, rule = base(name)
    d = R / f'{prefix}_s{n}' / task / SPLIT / stem
    if not (d / 'final.json').exists():
        return None
    fin = json.load(open(d / 'final.json'))
    agent = np.asarray(Image.open(d / fin['mask_path'])) > 0
    kept_no = removed_yes = 0
    for c in comps:
        region = labels == c['id']
        kept = agent[region].mean() >= .5
        kept_no += bool(c['id'] in no and kept)
        removed_yes += bool(c['id'] not in no and not kept)
    words = 0
    for line in open(d / 'steps.jsonl'):
        step = json.loads(line)
        words += bool(step.get('planner') and REJECT.search(step['planner'].get('thought') or ''))
    ledger = json.load(open(d / 'verification.json')) if (d / 'verification.json').exists() else []
    return dict(prefix=prefix, n=n, task=task, stem=stem, agent=counts(agent, gt), rule=counts(rule, gt), init=counts(init, gt),
        by=fin['submitted_by'], rounds=fin['planning_rounds'], cost=fin['cost_cny'], kept_no=kept_no, removed_yes=removed_yes,
        kept_no_px=counts(agent & init & ~rule, gt)[:2], removed_yes_px=counts(rule & ~agent, gt)[:2], added_px=counts(agent & ~init, gt)[:2],
        same_as_rule=bool((agent == rule).all()), reject_word=words, ledger_differs=sum(bool(r['differs_from_rule']) for r in ledger),
        answered=sum(r['latest'] is not None for r in ledger), agent_mask_empty=not agent.any())


def f1(c):
    return 200 * c[..., 0].sum(-1) / np.maximum(1, 2 * c[..., 0].sum(-1) + c[..., 1].sum(-1) + c[..., 2].sum(-1))


def ci(A, B, rng, samples=4000):
    """A: (runs, images, 3), B: (images, 3) or (runs2, images, 3); images and runs resampled independently."""
    out = []
    B = B if B.ndim == 3 else B[None]
    for _ in range(samples):
        i = rng.integers(0, A.shape[1], A.shape[1])
        a = np.mean([f1(A[j][i]) for j in rng.integers(0, len(A), len(A))])
        b = np.mean([f1(B[j][i]) for j in rng.integers(0, len(B), len(B))])
        out.append(a - b)
    return [round(float(v), 2) for v in np.percentile(out, [2.5, 97.5])]


def main():
    args = [(p, n, t, x) for p in PREFIXES for n in (1, 2, 3) for t in TASKS for x in todo]
    with Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', 8))) as pool:
        res = [r for r in pool.map(session, args, chunksize=8) if r]
    rng = np.random.default_rng(0)
    stems = [Path(x).stem for x in todo]
    report = {}
    runs = {}
    for prefix in PREFIXES:
        for task in TASKS:
            g = [r for r in res if r['prefix'] == prefix and r['task'] == task]
            if not g:
                continue
            present = sorted({r['n'] for r in g})
            by = {(r['n'], r['stem']): r for r in g}
            complete = [n for n in present if all((n, s) in by for s in stems)]
            A = np.array([[by[(n, s)]['agent'] for s in stems] for n in complete])
            Ru = np.array([by[(complete[0], s)]['rule'] for s in stems])
            In = np.array([by[(complete[0], s)]['init'] for s in stems])
            runs[(prefix, task)] = A
            per = [round(float(f1(a)), 2) for a in A]
            sum3 = lambda k: [round(float(sum(r[k][i] for r in g)) / len(complete)) for i in range(2)]
            row = dict(sessions=len(g), complete_runs=len(complete), per_run=per, mean=round(float(np.mean(per)), 2),
                sd=round(float(np.std(per, ddof=1)), 2) if len(per) > 1 else None, rule=round(float(f1(Ru)), 2), fixed=round(float(f1(In)), 2),
                minus_rule=round(float(np.mean(per) - f1(Ru)), 2), minus_rule_ci=ci(A, Ru, rng),
                minus_fixed=round(float(np.mean(per) - f1(In)), 2), minus_fixed_ci=ci(A, In, rng),
                kept_no_components=sum(r['kept_no'] for r in g) / len(complete), removed_yes_components=sum(r['removed_yes'] for r in g) / len(complete),
                kept_no_px_tp_fp=sum3('kept_no_px'), removed_yes_px_tp_fp=sum3('removed_yes_px'), added_px_tp_fp=sum3('added_px'),
                same_as_rule_sessions=sum(r['same_as_rule'] for r in g), by={k: sum(r['by'] == k for r in g) for k in sorted({r['by'] for r in g})},
                empty=sum(r['agent_mask_empty'] for r in g), rounds=round(float(np.mean([r['rounds'] for r in g])), 1),
                reject_word_sessions=sum(r['reject_word'] > 0 for r in g), cost=round(sum(r['cost'] for r in g), 2),
                answered_components_mean=round(float(np.mean([r['answered'] for r in g])), 1), ledger_differs_mean=round(float(np.mean([r['ledger_differs'] for r in g])), 1))
            # Net gain of each kind of deviation: put only that kind back on top of the rule.
            T = Ru.sum(0)
            gain = lambda d_tp, d_fp: round(float(200 * (T[0] + d_tp) / (2 * (T[0] + d_tp) + T[1] + d_fp + T[2] - d_tp) - f1(Ru)), 2)
            kt, kf = row['kept_no_px_tp_fp']; rt, rf = row['removed_yes_px_tp_fp']; at, af = row['added_px_tp_fp']
            row['net_gain'] = dict(kept_no=gain(kt, kf), removed_yes=gain(-rt, -rf), added=gain(at, af), note='one kind at a time, on the rule mask; not additive')
            report[f'{prefix} {task}'] = row
            print(prefix, task, json.dumps(row, ensure_ascii=False), flush=True)
    for a, b in zip(PREFIXES, PREFIXES[1:]):
        for task in TASKS:
            if (a, task) in runs and (b, task) in runs:
                A, B = runs[(a, task)], runs[(b, task)]
                d = round(float(np.mean(f1(B)) - np.mean(f1(A))), 2)
                report[f'{b} - {a} {task}'] = dict(diff=d, ci=ci(B, A, rng))
                print(b, '-', a, task, d, report[f'{b} - {a} {task}']['ci'], flush=True)
    if os.environ.get('OUT'):
        Path(os.environ['OUT']).write_text(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
