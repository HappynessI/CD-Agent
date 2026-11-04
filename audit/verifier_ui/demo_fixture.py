"""Build a synthetic store (fake images, made-up labels and scores) to try the UI without
the server data. Mirrors the exp6k layout: demo pools from train, dev sets from val, test
sets from test, per dataset; a local model with token probabilities and qwen3-vl-plus with
verbal confidence, forms A / B / A+B, batched asking by image, API usage and cost.

  python demo_fixture.py /tmp/verifier_demo     -> /tmp/verifier_demo/verifier.db
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

import store
from store import RunRecorder, icl_template, zero_shot_template

SYSTEM = {
    'second': ('You judge whether a candidate region in a SECOND image pair is a real semantic change. In SECOND, '
               'change means the land-cover class differs between T1 and T2 among: water, ground, low vegetation, '
               'tree, building, playground. Not change: seasonal colour of the same vegetation, lighting and '
               'shadows, imaging tone, a roof repainted or renovated, vehicles, slight misregistration.\n'
               'Answer with JSON only: {"change": "yes" or "no", "confidence": integer 0-100}'),
    'sysu': ('You judge whether a candidate region in a SYSU-CD image pair is a real change. SYSU-CD marks new '
             'buildings, suburban expansion, groundwork before construction, vegetation change, road expansion and '
             'sea construction. Not change: seasonal colour, lighting and shadows, imaging tone, a roof repainted, '
             'vehicles or ships, slight misregistration.\n'
             'Answer with JSON only: {"change": "yes" or "no", "confidence": integer 0-100}'),
}
CLASSES = ['low vegetation', 'ground', 'building', 'tree', 'water', 'playground']
PSEUDO = ['季节 / 物候', '光照与阴影', '色调 / 成像差异', '屋顶换色或翻新', '车辆等临时物体', '配准错位']
SPLIT_OF = {'demo_pool': 'train', 'dev': 'val', 'test': 'test'}


def render(root, s, rng):
    """Whole images (256 px), binary mask and 128 px near crops with a red contour."""
    x0, y0 = rng.randrange(20, 180), rng.randrange(20, 180)
    side = int(math.sqrt(s['area']) / 3) + 12
    box = [x0, y0, x0 + side, y0 + side]
    base = (85 + 25 * (s['dataset'] == 'sysu'), 115 + rng.randrange(-15, 15), 75)
    paths = {}
    for t in ('t1', 't2'):
        im = Image.new('RGB', (256, 256), base)
        d = ImageDraw.Draw(im)
        for _ in range(6):
            a, b = rng.randrange(0, 230), rng.randrange(0, 230)
            d.rectangle((a, b, a + 26, b + 18), fill=(150, 150, 140))
        if t == 't2' and s['label']:
            d.rectangle(box, fill=(195, 190, 180))
        if t == 't2' and not s['label']:
            d.rectangle(box, fill=(base[0] + 30, base[1] + 25, base[2] + 10))  # pseudo-change
        paths[f'full_{t}'] = root / 'img' / f"{s['cid']}_full_{t}.png"
        im.save(paths[f'full_{t}'])
        crop = im.crop((x0 - 32, y0 - 32, x0 + side + 32, y0 + side + 32)).resize((128, 128))
        ImageDraw.Draw(crop).rectangle((24, 24, 104, 104), outline=(230, 30, 30), width=2)
        paths[t] = root / 'img' / f"{s['cid']}_{t}.png"
        crop.save(paths[t])
    mask = Image.new('L', (256, 256), 0)
    ImageDraw.Draw(mask).rectangle(box, fill=255)
    paths['mask'] = root / 'img' / f"{s['cid']}_mask.png"
    mask.save(paths['mask'])
    return {k: str(v) for k, v in paths.items()}, box


def make(root, dataset, purpose, n, rng, image_offset=0):
    out = []
    for i in range(n):
        if purpose == 'demo_pool':
            stratum = ['pos2', 'pos_other', 'hard_neg', 'rand_neg'][[0, 0, 1, 2, 2, 3][i % 6]]
        else:
            stratum = rng.choices(['pos2', 'pos_other', 'hard_neg', 'rand_neg'], [3, 2, 3, 2])[0]
        label = int(stratum.startswith('pos'))
        s = dict(dataset=dataset, split=SPLIT_OF[purpose], cid=f'{dataset}_{purpose}_{3 * image_offset + i}', label=label,
                 image=f'{dataset}_{SPLIT_OF[purpose]}_{image_offset + i // 3}', source=rng.choice(['U5_x4']),
                 area=int(rng.lognormvariate(7.2, 0.9)) + 200, stratum=stratum,
                 weight=round({'pos2': 4.0, 'pos_other': 6.0, 'hard_neg': 12.0, 'rand_neg': 60.0}[stratum], 1))
        s['det_mean'] = min(.99, max(.01, rng.gauss(.42 if label else (.38 if stratum == 'hard_neg' else .2), .12)))
        s['m1_score'] = min(.99, max(.01, rng.gauss(.45 if label else (.36 if stratum == 'hard_neg' else .18), .12)))
        if label:
            a, b = rng.sample(CLASSES, 2)
            s['change_kind'] = f'{a} → {b}' if dataset == 'second' else None
            s['rationale'] = f'{a} 变为 {b}' if dataset == 'second' else '新建建筑'
        else:
            s['pseudo_type'] = rng.choice(PSEUDO) if stratum == 'hard_neg' else None
            s['rationale'] = f"{s['pseudo_type'] or '无明显变化'}，属于无变化"
        if dataset == 'second':
            t1 = s['change_kind'].split(' → ')[0] if label else rng.choice(CLASSES)
            t2 = s['change_kind'].split(' → ')[1] if label else t1
            f1, f2 = rng.choice([.74, .82, .9, 1.0]), rng.choice([.74, .82, .9, 1.0])
            s['semantic_t1'] = [{'name': t1 if label else '无变化（未提供地类）', 'fraction': f1}, {'name': 'tree', 'fraction': 1 - f1}]
            s['semantic_t2'] = [{'name': t2 if label else '无变化（未提供地类）', 'fraction': f2}, {'name': 'tree', 'fraction': 1 - f2}]
            s['semantic_full'] = [[{'name': '无变化（未提供地类）', 'fraction': .81}, {'name': t1, 'fraction': .12}, {'name': 'tree', 'fraction': .07}],
                                  [{'name': '无变化（未提供地类）', 'fraction': .81}, {'name': t2, 'fraction': .12}, {'name': 'tree', 'fraction': .07}]]
            s['semantic_text'] = f'候选区域 T1：{t1} …；T2：{t2} …；整图两期主要类别……（长版，前端会改写成一句）'
        s['inputs'], s['bbox'] = render(root, s, rng)
        out.append(s)
    return out


def fake_answer(q, skill, rng):
    hard = 0.6 if q['stratum'] == 'hard_neg' else 0
    logit = (1.2 if q['label'] else -1.2) * skill + 0.9 + hard + rng.gauss(0, 1.3)  # yes-biased like exp6j
    return 1 / (1 + math.exp(-logit))


def main():
    root = Path(sys.argv[1])
    (root / 'img').mkdir(parents=True, exist_ok=True)
    db_path = root / 'verifier.db'
    db_path.unlink(missing_ok=True)
    rng = random.Random(0)
    data = {}
    conn = store.connect(db_path)
    with conn:
        for ds in ('second', 'sysu'):
            pool = make(root, ds, 'demo_pool', 36, rng)
            pool[5] = {**pool[5], 'split': 'val', 'cid': f'{ds}_pool_from_val'}  # planted violation
            dev = make(root, ds, 'dev', 120, rng)
            test = make(root, ds, 'test', 150, rng)
            dev[0] = {**dev[0], 'image': pool[0]['image']}  # planted image overlap
            pool256 = pool + make(root, ds, 'demo_pool', 24, rng, image_offset=100)  # superset, shares reviews
            for name, purpose, items in ((f'exp6k_demos_{ds}', 'demo_pool', pool), (f'exp6k_demos256_{ds}', 'demo_pool', pool256),
                                         (f'exp6k_dev_{ds}', 'dev', dev),
                                         (f'exp6k_test_{ds}', 'test', test)):
                store.add_sample_set(conn, name, purpose, ds, items, note='synthetic fixture')
            data[ds] = pool, dev, test
        for i, s in enumerate(data['second'][0][:8]):
            sid = conn.execute('SELECT id FROM sample WHERE cid=?', (s['cid'],)).fetchone()[0]
            store.set_review(conn, sid, 'reject' if i == 3 else 'pass', '理由写得不准' if i == 3 else None)
    conn.close()

    def run(ds, purpose, model, mode, form, k, skill, readout, batch_by=None, prompt_v=1):
        pool, dev, test = data[ds]
        queries = dev if purpose == 'dev' else test
        demos = [d for d in pool if d['split'] == 'train'][:k]
        prompt = SYSTEM[ds] if prompt_v == 1 else SYSTEM[ds].replace('Answer with', 'Judge only the red contour.\nAnswer with')
        name = f'{ds}_{purpose}_{model}_{mode}_{form}_k{k}' + ('' if prompt_v == 1 else f'_p{prompt_v}')
        template = zero_shot_template(prompt) if mode == 'zero' else icl_template(k, prompt, form=form)
        api = model == 'qwen3-vl-plus'
        with RunRecorder(db_path, name=name, model=model, prompt_text=prompt, prompt_name=f'{ds}_sys_v{prompt_v}',
                         prompt_parent=store.digest(SYSTEM[ds]) if prompt_v > 1 else None, template=template,
                         demos=demos, mode=mode, form=form, pool=f'exp6k_demos_{ds}', eval_set=f'exp6k_{purpose}_{ds}',
                         strategy='balanced random (synthetic)', seed=0, experiment='fixture', readout=readout,
                         batch_by=batch_by) as rec:
            if batch_by:
                by_image = {}
                for q in queries:
                    by_image.setdefault(q['image'], []).append(q)
                for image, qs in by_image.items():
                    ps = [fake_answer(q, skill, rng) for q in qs]
                    ans = ['yes' if p > .5 else 'no' for p in ps]
                    conf = [int(50 + abs(p - .5) * 90) // 5 * 5 for p in ps]
                    raw = '\n'.join(f'{j + 1}. {{"change": "{a}", "confidence": {c}}}' for j, (a, c) in enumerate(zip(ans, conf)))
                    rec.record_batch(qs, raw, [dict(answer=a, confidence=c) for a, c in zip(ans, conf)], group_key=image,
                                     usage=dict(input_tokens=900 * k + 700 * len(qs), output_tokens=18 * len(qs),
                                                cost=(900 * k + 700 * len(qs)) * 8e-7 + 18 * len(qs) * 8e-6))
                return
            for q in queries:
                p = fake_answer(q, skill, rng)
                a = 'yes' if p > .5 else 'no'
                c = int(50 + abs(p - .5) * 90) // 5 * 5
                usage = dict(input_tokens=900 * k + 700, output_tokens=18, cost=(900 * k + 700) * 8e-7 + 18 * 8e-6) if api else None
                rec.record(q, p_yes=None if api else p, yes_no_mass=None if api else .97, answer=a, confidence=c,
                           raw_text=f'{{"change": "{a}", "confidence": {c}}}', usage=usage)

    for ds in ('second', 'sysu'):
        for purpose in ('dev', 'test'):
            run(ds, purpose, 'qwen3.5-4b', 'zero', 'A', 0, .5, 'token_prob')
            for k in (1, 2, 4, 8):
                run(ds, purpose, 'qwen3.5-4b', 'few', 'A', k, .5 + .12 * math.log2(k + 1), 'token_prob')
            for k in (16, 32):
                run(ds, purpose, 'qwen3.5-4b', 'many', 'A', k, .55 + .1 * math.log2(k), 'token_prob')
            run(ds, purpose, 'qwen3-vl-plus', 'zero', 'A', 0, .8, 'verbal_confidence')
            for form, bonus in (('A', 0), ('B', -.15), ('AB', .1)):
                for k in (4, 8, 16):
                    run(ds, purpose, 'qwen3-vl-plus', 'few' if k <= 8 else 'many', form, k,
                        .8 + bonus + .12 * math.log2(k), 'verbal_confidence')
            run(ds, purpose, 'qwen3-vl-plus', 'batched_many', 'A', 16, 1.1, 'verbal_confidence', batch_by='image')
        run(ds, 'dev', 'qwen3.5-4b', 'zero', 'A', 0, .6, 'token_prob', prompt_v=2)
    print(db_path)


if __name__ == '__main__':
    main()
