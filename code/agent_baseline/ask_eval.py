"""Offline quality card for the ask tool: every verifier sees the same fixed component pool.

The pool is every connected component of the fixed-route mask, so it does not depend on what a planner
chose to ask. Labels are used only by `report`; the prompts and crops never see them.

  answers (GPU for local, API for the rest):
    python -m agent_baseline.ask_eval answers --split val --variant q37 --concurrency 12 --max-cost-cny 20
  report (CPU):
    python -m agent_baseline.ask_eval report --split val --out probe/ask_eval.json

Variants: local = Qwen3.5-4B worker (the BF ask); q37 and vl = API with the BF crop;
q37_ms = API with close-up and wider context crops; q37_think = q37 with thinking enabled;
q37_pp = one question per date ("is there a building inside the contour?"), the answers are compared;
q37_corr = q37_ms with a question about the correspondence of the object across the dates;
f37_ms and f37_corr = q37_ms and q37_corr on qwen3.7-flash.
Plan v2, on the dev2 pool: f37_corr is L1. v1_ex adds fixed solved examples before the question; v2_s1..3 sample L1
three times; v2_close and v2_wide ask from one crop pair only; v3 is L1 on qwen3.7-plus; v4 makes the answer the
first token of the reply, so that its probability can be read.
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
from prompts import PSEUDO_TYPES, SYSTEM, VERIFY, verifier_prompts
from .backends import CORRESPONDENCE, encode_image, example_blocks, post, yes_probability
from .planner import Planner, RunGuard, RunStopped, request_cost, token_cost, usage_tokens
from .view import ask_images

RUN = Path(os.environ.get('AGENT_RUN', '/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run'))
# The ledger prices every variant at planner.PRICE (qwen3.7-plus). The price of qwen3-vl-plus is not
# confirmed here, so its cost_cny is only an estimate; the console bill is authoritative.
VARIANTS = dict(q37=dict(model='qwen3.7-plus'), q37_ms=dict(model='qwen3.7-plus', context=True),
                q37_think=dict(model='qwen3.7-plus', think=True), vl=dict(model='qwen3-vl-plus'),
                q37_pp=dict(model='qwen3.7-plus', presence=True),
                q37_corr=dict(model='qwen3.7-plus', context=True, correspondence=True),
                f37_ms=dict(model='qwen3.7-flash', context=True),
                f37_corr=dict(model='qwen3.7-flash', context=True, correspondence=True),
                v1_ex=dict(model='qwen3.7-flash', context=True, correspondence=True, examples=True),
                v2_s1=dict(model='qwen3.7-flash', context=True, correspondence=True, temperature=.7, sample=1),
                v2_s2=dict(model='qwen3.7-flash', context=True, correspondence=True, temperature=.7, sample=2),
                v2_s3=dict(model='qwen3.7-flash', context=True, correspondence=True, temperature=.7, sample=3),
                v2_close=dict(model='qwen3.7-flash', correspondence=True, only='close'),
                v2_wide=dict(model='qwen3.7-flash', context=True, correspondence=True, only='wide'),
                v3=dict(model='qwen3.7-plus', context=True, correspondence=True),
                v4=dict(model='qwen3.7-flash', context=True, correspondence=True, first_token=True))
PRESENCE_SYSTEM = 'You inspect one remote sensing image of a place, shown as a close-up and as a wider view.'
PRESENCE = ('The red contour marks the same region in both views. Is there a building inside the red contour? '
            'Reply with JSON only: {"building": "yes" or "no", "reason": one short sentence}')
CONTEXT_QUESTION = ('The first two images are close-ups and the last two show the same place with more '
                    'surroundings. ' + VERIFY)


INIT = 'sam3_imc'
LISTED = None
TAG = None


def image_names(split):
    return LISTED or names(split)


FIRST_TOKEN = CORRESPONDENCE.split('Reply with JSON only')[0] + ('Reply on one line in this exact format, starting with the answer: '
    'yes or no, then |, then one of ' + str(PSEUDO_TYPES).replace("'", '') + ', then |, then one short sentence. Example: no|no_change|Same roof outline in both dates.')
ONE_PAIR = dict(close='The two images show the same place at T1 and T2, close up. ', wide='The two images show the same place at T1 and T2, with surroundings. ')
ONE_QUESTION = CORRESPONDENCE.split('The red contour')[1]


def load_init(split, stem):
    return np.asarray(Image.open(RUN / 'init' / INIT / split / f'{stem}.png')) > 0


def folder(split):
    return RUN / 'ask_eval' / (TAG or (split if INIT == 'sam3_imc' else f'{split}_{INIT}'))


def pool(split):
    """All components of the fixed-route masks, largest first within an image."""
    rows = []
    for name in image_names(split):
        stem = Path(name).stem
        _, comps = components(load_init(split, stem))
        rows.extend(dict(uid=f"{stem}:{c['id']}", name=name, stem=stem, id=c['id'], area=int(c['area']),
                         bbox=[int(v) for v in c['bbox']]) for c in sorted(comps, key=lambda c: -c['area']))
    return rows


def crops(rgb, labels, row, context):
    return ask_images(rgb, labels == row['id'], row['bbox'], context)


def api_answer(images, spec, guard, seed, session=None):
    words = ['T1 close-up', 'T2 close-up', 'T1 wider view', 'T2 wider view']
    question = CORRESPONDENCE if spec.get('correspondence') else CONTEXT_QUESTION if spec.get('context') else VERIFY
    system = SYSTEM
    if session is not None:
        settings = verifier_prompts(session.target, session.change_direction, session.scope)
        system = settings['SYSTEM']
        question = settings['CORRESPONDENCE'] if spec.get('correspondence') else settings['VERIFY']
        if spec.get('context') and not spec.get('correspondence'):
            question = 'The first two images are close-ups and the last two show the same place with more surroundings. ' + question
    if spec.get('only'):
        keep = (0, 1) if spec['only'] == 'close' else (2, 3)
        words = [words[i] for i in keep]
        images = [images[i] for i in keep]
        question = ONE_PAIR[spec['only']] + 'The red contour' + ONE_QUESTION
    if spec.get('first_token'):
        question = FIRST_TOKEN
    content = list(example_blocks()) if spec.get('examples') else []
    for word, image in zip(words, images):
        content += [dict(type='text', text=word), dict(type='image_url',
                    image_url=dict(url='data:image/jpeg;base64,' + encode_image(image)))]
    content.append(dict(type='text', text=question))
    parameters = dict(model=spec['model'], max_tokens=256, parallel_tool_calls=False)
    if spec.get('temperature') is not None:
        parameters['temperature'] = spec['temperature']
    if spec.get('think'):
        # Thinking endpoints reject forced JSON and logprobs, and need room for the reasoning.
        parameters.update(enable_thinking=True, max_tokens=4096, stream=False)
    else:
        parameters.update(logprobs=True, top_logprobs=5)
        if not spec.get('first_token'):
            parameters['response_format'] = {'type': 'json_object'}
    # The guard stops every thread on a quota error or when the batch budget is spent.
    planner = Planner(seed=seed, guard=guard, parameters=parameters, max_attempts=12)
    result = planner.complete([dict(role='system', content=system), dict(role='user', content=content)], [], 1)
    tokens = usage_tokens(result['usage'])
    raw = result['message'].get('content') or ''
    if spec.get('first_token'):
        parts = [part.strip() for part in raw.split('|', 2)]
        parsed = dict(change=parts[0].lower(), type=parts[1] if len(parts) > 1 else None, reason=parts[2] if len(parts) > 2 else None)
    else:
        parsed = parse_json(raw)
        parsed = parsed if isinstance(parsed, dict) else {}
    return dict(answer=parsed.get('change'), type=parsed.get('type'), reason=parsed.get('reason'), raw=raw,
                p_yes=yes_probability(result['raw']['choices'][0]), tokens=tokens, cost_cny=request_cost(spec['model'], tokens))


def presence_answer(rgb, labels, row, spec, guard, seed):
    """Each date alone: a persistent building gives yes twice, however different the two dates look."""
    images = crops(rgb, labels, row, True)
    readings, tokens = [], dict(input=0, cached=0, output=0)
    for t in (0, 1):
        content = []
        for word, image in (('Close-up', images[t]), ('Wider view', images[t + 2])):
            content += [dict(type='text', text=word), dict(type='image_url',
                        image_url=dict(url='data:image/jpeg;base64,' + encode_image(image)))]
        content.append(dict(type='text', text=PRESENCE))
        planner = Planner(seed=seed, guard=guard, max_attempts=12, parameters=dict(model=spec['model'], max_tokens=128,
            parallel_tool_calls=False, logprobs=True, top_logprobs=5, response_format={'type': 'json_object'}))
        result = planner.complete([dict(role='system', content=PRESENCE_SYSTEM), dict(role='user', content=content)], [], 1)
        for key, value in usage_tokens(result['usage']).items():
            tokens[key] += value
        raw = result['message'].get('content') or ''
        parsed = parse_json(raw)
        parsed = parsed if isinstance(parsed, dict) else {}
        readings.append(dict(answer=parsed.get('building'), p=yes_probability(result['raw']['choices'][0]), reason=parsed.get('reason')))
    a1, a2 = readings[0]['answer'], readings[1]['answer']
    valid = a1 in ('yes', 'no') and a2 in ('yes', 'no')
    p1, p2 = readings[0]['p'], readings[1]['p']
    return dict(answer=('yes' if a1 != a2 else 'no') if valid else None, p_yes=None if p1 is None or p2 is None else p1 * (1 - p2) + p2 * (1 - p1),
                type=f'{a1}->{a2}', reason=json.dumps(readings, ensure_ascii=False), raw=None, tokens=tokens, cost_cny=token_cost(tokens))


def answers(args):
    out = folder(args.split) / f'{args.variant}.jsonl'
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)['uid'] for line in out.read_text().splitlines() if line.strip()}
    rows = [row for row in pool(args.split) if row['uid'] not in done]
    if args.limit:
        rows = rows[:args.limit]
    guard = RunGuard(RUN / 'STOP.json', RUN / 'budget' / f'ask_eval_{TAG or INIT}_{args.variant}.txt', args.max_cost_cny)
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
        pair = None if spec.get('presence') else crops(rgb, labels, row, spec.get('context'))
        seed = (sum(map(ord, row['uid'])) + 1000 * spec.get('sample', 0)) % (2**31)
        if spec.get('presence'):
            result = presence_answer(rgb, labels, row, spec, guard, seed)
        elif spec.get('local'):
            result = post(args.ask_worker, dict(pairs=[[encode_image(im) for im in pair]], question=VERIFY))['results'][0]
        else:
            result = api_answer(pair, spec, guard, seed)
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
    for name in image_names(split):
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
    for path in sorted(folder(args.split).glob('*.jsonl')):
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
                   auc_binary=auc(says_yes[real].astype(float), says_yes[~real].astype(float)),
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
        # An endpoint may return logprobs for the first token only; mixed scores are then not a ranking.
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
    one.add_argument('--init-name', default='sam3_imc')
    one.add_argument('--variant', required=True, choices=['local', *VARIANTS])
    one.add_argument('--concurrency', type=int, default=8)
    one.add_argument('--limit', type=int)
    one.add_argument('--ask-worker')
    one.add_argument('--max-cost-cny', type=float, default=20)
    one.add_argument('--names-file')
    one.add_argument('--tag', help='Folder under ask_eval, when the pool is a listed set of images.')
    two = sub.add_parser('report')
    two.add_argument('--split', default='val')
    two.add_argument('--init-name', default='sam3_imc')
    two.add_argument('--out')
    two.add_argument('--names-file')
    two.add_argument('--tag')
    args = parser.parse_args()
    global INIT, LISTED, TAG
    INIT = args.init_name
    LISTED = Path(args.names_file).read_text().split() if args.names_file else None
    TAG = args.tag
    (answers if args.command == 'answers' else report)(args)


if __name__ == '__main__':
    main()
