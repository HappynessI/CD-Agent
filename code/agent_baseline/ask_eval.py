"""Offline quality card for the ask tool: every verifier sees the same fixed component pool.

The pool is every connected component of the fixed-route mask, so it does not depend on what a planner
chose to ask. Labels are used only by `report`; the prompts and crops never see them.

  answers (GPU for local, API for the rest):
    python -m agent_baseline.ask_eval answers --split val --variant q37 --concurrency 12 --max-cost-cny 20
  report (CPU):
    python -m agent_baseline.ask_eval report --split val --out probe/ask_eval.json

Variants: local = Qwen3.5-4B worker (the BF ask); q37 and vl = API with the BF crop;
q37_ms = API with close-up and wider context crops; q37_think = q37 with thinking enabled.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import threading
import numpy as np
from PIL import Image
from common import components, load_rgb, names
from mllm import parse_json
from prompts import SYSTEM, VERIFY
from .backends import encode_image, post, yes_probability
from .planner import Planner, RunGuard, RunStopped, token_cost, usage_tokens
from .view import ask_images

RUN = Path(os.environ.get('AGENT_RUN', '/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run'))
# The ledger prices every variant at planner.PRICE (qwen3.7-plus). The price of qwen3-vl-plus is not
# confirmed here, so its cost_cny is only an estimate; the console bill is authoritative.
VARIANTS = dict(q37=dict(model='qwen3.7-plus'), q37_ms=dict(model='qwen3.7-plus', context=True),
                q37_think=dict(model='qwen3.7-plus', think=True), vl=dict(model='qwen3-vl-plus'))
CONTEXT_QUESTION = ('The first two images are close-ups and the last two show the same place with more '
                    'surroundings. ' + VERIFY)


def load_init(split, stem):
    return np.asarray(Image.open(RUN / 'init' / 'sam3_imc' / split / f'{stem}.png')) > 0


def pool(split):
    """All components of the fixed-route masks, largest first within an image."""
    rows = []
    for name in names(split):
        stem = Path(name).stem
        _, comps = components(load_init(split, stem))
        rows.extend(dict(uid=f"{stem}:{c['id']}", name=name, stem=stem, id=c['id'], area=int(c['area']),
                         bbox=[int(v) for v in c['bbox']]) for c in sorted(comps, key=lambda c: -c['area']))
    return rows


def crops(rgb, labels, row, context):
    return ask_images(rgb, labels == row['id'], row['bbox'], context)


def api_answer(images, spec, guard, seed):
    words = ['T1 close-up', 'T2 close-up', 'T1 wider view', 'T2 wider view']
    content = []
    for word, image in zip(words, images):
        content += [dict(type='text', text=word), dict(type='image_url',
                    image_url=dict(url='data:image/jpeg;base64,' + encode_image(image)))]
    content.append(dict(type='text', text=CONTEXT_QUESTION if spec.get('context') else VERIFY))
    parameters = dict(model=spec['model'], max_tokens=256, parallel_tool_calls=False)
    if spec.get('think'):
        # Thinking endpoints reject forced JSON and logprobs, and need room for the reasoning.
        parameters.update(enable_thinking=True, max_tokens=4096, stream=False)
    else:
        parameters.update(logprobs=True, top_logprobs=5, response_format={'type': 'json_object'})
    # The guard stops every thread on a quota error or when the batch budget is spent.
    planner = Planner(seed=seed, guard=guard, parameters=parameters, max_attempts=12)
    result = planner.complete([dict(role='system', content=SYSTEM), dict(role='user', content=content)], [], 1)
    tokens = usage_tokens(result['usage'])
    raw = result['message'].get('content') or ''
    parsed = parse_json(raw)
    parsed = parsed if isinstance(parsed, dict) else {}
    return dict(answer=parsed.get('change'), type=parsed.get('type'), reason=parsed.get('reason'), raw=raw,
                p_yes=yes_probability(result['raw']['choices'][0]), tokens=tokens, cost_cny=token_cost(tokens))


def answers(args):
    out = RUN / 'ask_eval' / args.split / f'{args.variant}.jsonl'
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)['uid'] for line in out.read_text().splitlines() if line.strip()}
    rows = [row for row in pool(args.split) if row['uid'] not in done]
    if args.limit:
        rows = rows[:args.limit]
    guard = RunGuard(RUN / 'STOP.json', RUN / 'budget' / f'ask_eval_{args.variant}.txt', args.max_cost_cny)
    guard.check()
    lock, images = threading.Lock(), {}

    def image(row):
        with lock:
            if row['stem'] not in images:
                labels, _ = components(load_init(args.split, row['stem']))
                images.clear()
                images[row['stem']] = (load_rgb(args.split, row['name']), labels)
            return images[row['stem']]

    spec = VARIANTS.get(args.variant, dict(local=True))

    def one(row):
        rgb, labels = image(row)
        pair = crops(rgb, labels, row, spec.get('context'))
        if spec.get('local'):
            result = post(args.ask_worker, dict(pairs=[[encode_image(im) for im in pair]], question=VERIFY))['results'][0]
        else:
            result = api_answer(pair, spec, guard, sum(map(ord, row['uid'])) % (2**31))
        return dict(uid=row['uid'], variant=args.variant, **{k: result.get(k) for k in
                    ('answer', 'p_yes', 'type', 'reason', 'tokens', 'cost_cny', 'raw')})

    failures = 0
    with ThreadPoolExecutor(args.concurrency) as pool_, out.open('a') as file:
        for start in range(0, len(rows), args.concurrency * 4):
            futures = [(row, pool_.submit(one, row)) for row in rows[start:start + args.concurrency * 4]]
            for row, future in futures:
                try:
                    file.write(json.dumps(future.result(), ensure_ascii=False) + '\n')
                except RunStopped:
                    raise
                except Exception as exc:
                    failures += 1
                    print('failed', row['uid'], type(exc).__name__, str(exc)[:200], flush=True)
            file.flush()
            print(args.variant, min(start + args.concurrency * 4, len(rows)), '/', len(rows), 'failures', failures, flush=True)


def auc(positive, negative, weights_positive=None, weights_negative=None):
    """Rank AUC; with weights, each pair counts in proportion to the product of its weights."""
    wp = np.ones(len(positive)) if weights_positive is None else np.asarray(weights_positive, float)
    wn = np.ones(len(negative)) if weights_negative is None else np.asarray(weights_negative, float)
    order = np.argsort(negative)
    sorted_values, cumulative = np.asarray(negative)[order], np.cumsum(wn[order])
    total = wn.sum()
    below = np.searchsorted(sorted_values, positive, side='left')
    same = np.searchsorted(sorted_values, positive, side='right')
    wins = np.where(below > 0, cumulative[below - 1], 0.)
    upto = np.where(same > 0, cumulative[same - 1], 0.)
    return float((wp * (wins + .5 * (upto - wins))).sum() / (wp.sum() * total)) if len(positive) and total else None


def f1(tp, fp, fn):
    return 200 * tp / max(1, 2 * tp + fp + fn)


def truth(split):
    from common import load_gt
    table, totals = {}, dict(fp=0, tp=0, fn=0)
    for name in names(split):
        stem = Path(name).stem
        gt = load_gt(split, name)
        data = load_init(split, stem)
        labels, comps = components(data)
        totals['tp'] += int((data & gt).sum())
        totals['fp'] += int((data & ~gt).sum())
        totals['fn'] += int((~data & gt).sum())
        for c in comps:
            region = labels == c['id']
            table[f"{stem}:{c['id']}"] = dict(tp=int((region & gt).sum()), fp=int((region & ~gt).sum()), area=int(c['area']))
    return table, totals


def report(args):
    table, base = truth(args.split)
    result = dict(pool=len(table), base_f1=f1(base['tp'], base['fp'], base['fn']), variants={})
    for path in sorted((RUN / 'ask_eval' / args.split).glob('*.jsonl')):
        rows = {}
        for line in path.read_text().splitlines():
            row = json.loads(line)
            rows[row['uid']] = row
        usable = [(uid, row) for uid, row in rows.items() if row.get('answer') in ('yes', 'no') and uid in table]
        purity = np.array([table[u]['tp'] / max(1, table[u]['area']) for u, _ in usable])
        area = np.array([table[u]['area'] for u, _ in usable])
        says_yes = np.array([r['answer'] == 'yes' for _, r in usable])
        score = np.array([r['p_yes'] if r.get('p_yes') is not None else float(r['answer'] == 'yes') for _, r in usable])
        real = purity >= .5
        tp = np.array([table[u]['tp'] for u, _ in usable])
        fp = np.array([table[u]['fp'] for u, _ in usable])
        removed_tp, removed_fp = int(tp[~says_yes].sum()), int(fp[~says_yes].sum())
        row = dict(answered=len(usable), asked=len(rows), yes_rate=float(says_yes.mean()),
                   has_logprob=float(np.mean([r.get('p_yes') is not None for _, r in usable])),
                   auc=auc(score[real], score[~real]), auc_area=auc(score[real], score[~real], area[real], area[~real]),
                   yes_purity=float(tp[says_yes].sum() / max(1, area[says_yes].sum())),
                   no_purity=float(tp[~says_yes].sum() / max(1, area[~says_yes].sum())),
                   no_removes_tp=removed_tp, no_removes_fp=removed_fp,
                   delete_no_f1=f1(base['tp'] - removed_tp, base['fp'] - removed_fp, base['fn'] + removed_tp))
        bins = {}
        for low, high in ((0, .1), (.1, .5), (.5, .9), (.9, 1.01)):
            chosen = (purity >= low) & (purity < high)
            bins[f'{low}-{high}'] = dict(n=int(chosen.sum()), area=int(area[chosen].sum()),
                                         no_rate=float((~says_yes[chosen]).mean()) if chosen.any() else None,
                                         no_area=float(area[chosen & ~says_yes].sum() / max(1, area[chosen].sum())))
        row['by_purity'] = bins
        # Thresholds on p_yes, for variants with logprobs: the best score any fixed threshold can reach.
        if row['has_logprob'] > .9:
            sweep = {}
            for thr in (.05, .1, .2, .3, .5, .7, .9):
                gone = score < thr
                sweep[str(thr)] = f1(base['tp'] - int(tp[gone].sum()), base['fp'] - int(fp[gone].sum()), base['fn'] + int(tp[gone].sum()))
            row['threshold_f1'] = sweep
        costs = [r.get('cost_cny') or 0 for r in rows.values()]
        row['cost_cny'] = float(sum(costs))
        result['variants'][path.stem] = row
    text = json.dumps(result, indent=1, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).write_text(text)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    one = sub.add_parser('answers')
    one.add_argument('--split', default='val')
    one.add_argument('--variant', required=True, choices=['local', *VARIANTS])
    one.add_argument('--concurrency', type=int, default=8)
    one.add_argument('--limit', type=int)
    one.add_argument('--ask-worker')
    one.add_argument('--max-cost-cny', type=float, default=20)
    two = sub.add_parser('report')
    two.add_argument('--split', default='val')
    two.add_argument('--out')
    args = parser.parse_args()
    (answers if args.command == 'answers' else report)(args)


if __name__ == '__main__':
    main()
