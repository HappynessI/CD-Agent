"""Audit UI for MLLM verifier runs. Reads everything; writes only demo review results
(and a metric cache table).

  VERIFIER_DB=/path/verifier.db uvicorn app:app --port 8765    (from this directory)
  then on the laptop: ssh -L 8765:localhost:8765 <server>, open http://localhost:8765
"""
from __future__ import annotations

import io
import json
import os
from collections import Counter, defaultdict
from contextlib import closing
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

import store

DB = Path(os.environ.get('VERIFIER_DB', Path(__file__).with_name('verifier.db')))
STATIC = Path(__file__).with_name('static')
WALKTHROUGH_DIR = Path(os.environ['WALKTHROUGH_DIR']).resolve() if os.environ.get('WALKTHROUGH_DIR') else None
AREA_BINS = ((0, 500, '<500'), (500, 2000, '500–2k'), (2000, 8000, '2k–8k'), (8000, None, '≥8k'))
SPLIT_FOR = {'demo_pool': 'train', 'dev': 'val', 'test': 'test'}
METRIC_VERSION = 2  # bump when store.metrics changes so cached values are recomputed
app = FastAPI(title='MLLM verifier audit')


def db():
    conn = store.connect(DB)
    conn.execute('CREATE TABLE IF NOT EXISTS metric_cache (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
    return conn


def rows(sql, *args):
    with closing(db()) as conn:
        return [dict(r) for r in conn.execute(sql, args)]


def cached(key, compute):
    with closing(db()) as conn:
        hit = conn.execute('SELECT value FROM metric_cache WHERE key = ?', (key,)).fetchone()
        if hit:
            return json.loads(hit[0])
        value = compute()
        with conn:
            conn.execute('INSERT OR REPLACE INTO metric_cache VALUES (?, ?)', (key, json.dumps(value)))
        return value


def area_bin(area):
    if area is None:
        return '未知'
    return next(label for low, high, label in AREA_BINS if area >= low and (high is None or area < high))


def composition(items):
    """Counts shown for demo pools, demo sets and dev/test sets."""
    pos = [i for i in items if i['label'] == 1]
    neg = [i for i in items if i['label'] == 0]
    count = lambda xs, key: dict(Counter(key(x) or '未填' for x in xs).most_common())
    return {'n': len(items), 'n_pos': len(pos), 'n_neg': len(neg),
            'strata': count(items, lambda x: x.get('stratum')),
            'pseudo_type': count(neg, lambda x: x['pseudo_type']),
            'change_kind': count(pos, lambda x: x['change_kind']),
            'area': {label: sum(area_bin(i['area']) == label for i in items) for *_, label in AREA_BINS},
            'source': count(items, lambda x: x['source']),
            'split': count(items, lambda x: x['split']),
            'images': len({(i['dataset'], i['image']) for i in items}),
            'max_per_image': max(Counter((i['dataset'], i['image']) for i in items).values(), default=0)}


def set_items(set_id):
    return [store.with_semantic(r) for r in rows(
        'SELECT i.stratum, i.weight, v.status AS review_status, v.note AS review_note, v.reviewed_at, s.* '
        'FROM sample_set_item i JOIN sample s ON s.id = i.sample_id LEFT JOIN sample_review v ON v.sample_id = s.id '
        'WHERE i.set_id = ? ORDER BY s.id', set_id)]


def demo_items(demo_set_id):
    return [store.with_semantic(r) for r in rows(
        'SELECT i.position, i.shown_label, s.* FROM demo_set_item i JOIN sample s ON s.id = i.sample_id '
        'WHERE i.demo_set_id = ? ORDER BY i.position', demo_set_id)]


# ---- split checks ----

def split_violations():
    """Demo pools from train, dev sets from val, test sets from test; no source image shared
    between a dataset's pools / demo sets, dev sets and test sets."""
    sets = rows('SELECT * FROM sample_set')
    out, owners = [], defaultdict(set)
    for s in sets:
        items = set_items(s['id'])
        bad = [i for i in items if i['split'] != SPLIT_FOR[s['purpose']]]
        if bad:
            out.append({'kind': 'wrong_split', 'set': s['name'], 'n': len(bad),
                        'text': f"{s['name']}（{s['purpose']}）有 {len(bad)} 个样本来自 {sorted({i['split'] for i in bad})}，应为 {SPLIT_FOR[s['purpose']]}"})
        for i in items:
            owners[(i['dataset'], i['image'])].add('demo' if s['purpose'] == 'demo_pool' else s['purpose'])
    for d in rows('SELECT DISTINCT s.dataset, s.image, s.split, d.name FROM demo_set_item i '
                  'JOIN sample s ON s.id = i.sample_id JOIN demo_set d ON d.id = i.demo_set_id'):
        owners[(d['dataset'], d['image'])].add('demo')
        if d['split'] != 'train':
            out.append({'kind': 'wrong_split', 'set': d['name'], 'n': 1,
                        'text': f"示例集 {d['name']} 用了 {d['split']} 划分的图 {d['image']}"})
    shared = Counter(tuple(sorted(v)) for k, v in owners.items() if len(v) > 1 and k[1])
    for roles, n in shared.items():
        out.append({'kind': 'image_overlap', 'n': n, 'text': f"{n} 张原图同时出现在 {' 和 '.join(roles)} 中"})
    return out


# ---- metrics ----

def eval_rows(run_id, dataset, score_key=None):
    """Rows for store.metrics: a run's outputs (score_key None) or a reference score
    (det_mean / m1_score) over the same samples, weighted by the run's eval set."""
    out = rows('SELECT s.label, s.image, s.det_mean, s.m1_score, o.p_yes, o.answer, o.confidence, '
               'i.weight, i.stratum FROM output o JOIN run r ON r.id = o.run_id JOIN sample s ON s.id = o.sample_id '
               'LEFT JOIN sample_set_item i ON i.set_id = r.eval_set_id AND i.sample_id = s.id '
               'WHERE o.run_id = ? AND s.dataset = ?', run_id, dataset)
    for r in out:
        if score_key:
            r['score'] = r[score_key]
            r['pred'] = None if r['score'] is None else int(r['score'] > 0.5)
        else:
            r['score'], r['pred'] = store.score(r), store.prediction(r)
        r['group'] = r['image']
    return out


def run_metrics(run_id, dataset, n_outputs, score_key=None):
    key = f'v{METRIC_VERSION}|run{run_id}|{dataset}|{n_outputs}|{score_key}'
    return cached(key, lambda: store.metrics(eval_rows(run_id, dataset, score_key)))


def domain(demos, dataset):
    if not demos:
        return 'none'
    same = sum(d['dataset'] == dataset for d in demos)
    return 'same' if same == len(demos) else 'cross' if same == 0 else 'mixed'


def run_summaries(purpose=''):
    runs = rows('SELECT r.*, p.name AS prompt_name, d.mode, d.k, d.form, d.name AS demo_set_name, '
                's.name AS eval_set_name, s.purpose AS eval_purpose FROM run r '
                'JOIN prompt_version p ON p.id = r.prompt_id JOIN demo_set d ON d.id = r.demo_set_id '
                'LEFT JOIN sample_set s ON s.id = r.eval_set_id ORDER BY r.id')
    if purpose:
        runs = [r for r in runs if (r['eval_purpose'] or 'other') == purpose]
    counts = defaultdict(dict)
    for r in rows('SELECT o.run_id, s.dataset, COUNT(*) AS n FROM output o JOIN sample s ON s.id = o.sample_id '
                  'GROUP BY o.run_id, s.dataset'):
        counts[r['run_id']][r['dataset']] = r['n']
    demos = {}
    for r in runs:
        r.pop('template')
        if r['demo_set_id'] not in demos:
            demos[r['demo_set_id']] = demo_items(r['demo_set_id'])
        r['datasets'] = {ds: {**run_metrics(r['id'], ds, n), 'demo_domain': domain(demos[r['demo_set_id']], ds)}
                         for ds, n in sorted(counts[r['id']].items())}
    return runs


def references(runs):
    """D and M1-ign3 on exactly the samples of one run per (eval set, dataset)."""
    seen, out = set(), []
    for r in runs:
        for ds, m in r['datasets'].items():
            key = (r['eval_set_id'] or f"run{r['id']}", ds)
            if key in seen or r['eval_set_id'] is None:
                continue
            seen.add(key)
            n = m['n']
            for score_key, label in (('det_mean', 'D（检测器均值）'), ('m1_score', 'M1-ign3')):
                ref = run_metrics(r['id'], ds, n, score_key)
                if ref.get('auc') is not None:
                    out.append({'eval_set': r['eval_set_name'], 'eval_purpose': r['eval_purpose'], 'dataset': ds,
                                'name': label, 'score_key': score_key, 'n_samples': n, **ref})
    return out


# ---- API ----

@app.get('/api/prompts')
def prompts():
    items = rows('SELECT p.*, COUNT(r.id) AS runs, GROUP_CONCAT(DISTINCT r.model) AS models FROM prompt_version p '
                 'LEFT JOIN run r ON r.prompt_id = p.id GROUP BY p.id ORDER BY p.created_at, p.name')
    rendered = defaultdict(list)
    for p in items:
        if p['kind'] == 'rendered':
            rendered[p['parent_id']].append(p['id'])
    for p in items:
        p['rendered_ids'] = rendered[p['id']] if p['kind'] != 'rendered' else []
    return items


@app.get('/api/checks')
def checks():
    return split_violations()


@app.get('/api/sets')
def sample_sets():
    out = []
    for s in rows('SELECT * FROM sample_set ORDER BY dataset, purpose, name'):
        items = set_items(s['id'])
        out.append({**s, 'n': len(items), 'n_pos': sum(i['label'] == 1 for i in items),
                    'reviewed': dict(Counter(i['review_status'] or 'todo' for i in items))})
    return out


@app.get('/api/sets/{set_id}')
def sample_set(set_id: int):
    head = rows('SELECT * FROM sample_set WHERE id = ?', set_id)
    if not head:
        raise HTTPException(404)
    items = set_items(set_id)
    kept = [i for i in items if i['review_status'] != 'reject']
    want = SPLIT_FOR[head[0]['purpose']]
    for i in items:
        i['split_ok'] = i['split'] == want
        i['inputs'] = list(json.loads(i['inputs']))
    return {**head[0], 'items': items, 'composition': composition(items), 'composition_kept': composition(kept),
            'violations': [v for v in split_violations() if v.get('set') in (None, head[0]['name'])]}


class Review(BaseModel):
    sample_id: int
    status: str | None = None
    note: str | None = None


@app.post('/api/sets/{set_id}/review')
def review(set_id: int, body: Review):
    if body.status not in (None, '', 'pass', 'reject'):
        raise HTTPException(400, 'status must be pass, reject or empty')
    if not rows('SELECT 1 FROM sample_set_item WHERE set_id = ? AND sample_id = ?', set_id, body.sample_id):
        raise HTTPException(404)
    with closing(db()) as conn, conn:
        store.set_review(conn, body.sample_id, body.status, body.note)
    return rows('SELECT status AS review_status, note AS review_note, reviewed_at FROM sample_review '
                'WHERE sample_id = ?', body.sample_id)[0]


@app.get('/api/demo_sets')
def demo_sets():
    sets = rows('SELECT d.*, p.name AS pool_name, COUNT(DISTINCT r.id) AS runs FROM demo_set d '
                'LEFT JOIN sample_set p ON p.id = d.pool_id LEFT JOIN run r ON r.demo_set_id = d.id GROUP BY d.id '
                "ORDER BY CASE d.mode WHEN 'zero' THEN 0 WHEN 'few' THEN 1 WHEN 'many' THEN 2 ELSE 3 END, d.form, d.k")
    for d in sets:
        items = demo_items(d['id'])
        d['sources'] = dict(Counter(f"{i['dataset']}/{i['split']}" for i in items))
        d['n_pos'] = sum(i['label'] == 1 for i in items)
        d['n_not_train'] = sum(i['split'] != 'train' for i in items)
    return sets


@app.get('/api/demo_sets/{demo_set_id}')
def demo_set(demo_set_id: str):
    head = rows('SELECT d.*, p.name AS pool_name FROM demo_set d LEFT JOIN sample_set p ON p.id = d.pool_id '
                'WHERE d.id = ?', demo_set_id)
    if not head:
        raise HTTPException(404)
    items = demo_items(demo_set_id)
    for i in items:
        i['inputs'] = list(json.loads(i['inputs']))
    return {**head[0], 'items': items, 'composition': composition(items)}


@app.get('/api/runs')
def runs(purpose: str = ''):
    found = run_summaries(purpose)
    return {'runs': found, 'references': references(found)}


@app.get('/api/runs/{run_id}')
def run(run_id: int):
    found = [r for r in run_summaries() if r['id'] == run_id]
    if not found:
        raise HTTPException(404)
    r = found[0]
    r['template'] = json.loads(rows('SELECT template FROM run WHERE id = ?', run_id)[0]['template'])
    r['prompt_text'] = rows('SELECT text FROM prompt_version WHERE id = ?', r['prompt_id'])[0]['text']
    r['demos'] = demo_items(r['demo_set_id'])
    want = SPLIT_FOR.get(r['eval_purpose'])
    demo_images = {(d['dataset'], d['image']) for d in r['demos'] if d['image']}
    queries = rows('SELECT s.id, s.dataset, s.image, s.split FROM output o JOIN sample s ON s.id = o.sample_id '
                   'WHERE o.run_id = ?', run_id)
    r['leaks'] = {'demos_not_train': sum(d['split'] != 'train' for d in r['demos']),
                  'queries_wrong_split': sum(q['split'] != want for q in queries) if want else 0,
                  'queries_sharing_demo_image': sum((q['dataset'], q['image']) in demo_images for q in queries),
                  'queries_that_are_demos': len({q['id'] for q in queries} & {d['id'] for d in r['demos']})}
    r['want_split'] = want
    r['calls'] = rows('SELECT COUNT(*) AS n FROM call WHERE run_id = ?', run_id)[0]['n']
    return r


@app.get('/api/runs/{run_id}/outputs')
def outputs(run_id: int, dataset: str = '', source: str = '', stratum: str = '', verdict: str = '',
            limit: int = 100, offset: int = 0):
    """verdict: '' | wrong | right | fp | fn. Wrong answers sort first, most confident first."""
    out = rows('SELECT o.*, s.dataset, s.split, s.cid, s.image, s.source, s.label, s.inputs, s.det_mean, s.m1_score, '
               'i.stratum, i.weight, c.raw_text AS call_raw FROM output o JOIN run r ON r.id = o.run_id '
               'JOIN sample s ON s.id = o.sample_id LEFT JOIN call c ON c.id = o.call_id '
               'LEFT JOIN sample_set_item i ON i.set_id = r.eval_set_id AND i.sample_id = s.id WHERE o.run_id = ?', run_id)
    for o in out:
        o['pred'], o['score'] = store.prediction(o), store.score(o)
        o['correct'] = None if o['pred'] is None or o['label'] is None else o['pred'] == o['label']
        o['inputs'] = list(json.loads(o['inputs']))
    keep = {'wrong': lambda o: o['correct'] is False, 'right': lambda o: o['correct'] is True,
            'fp': lambda o: o['pred'] == 1 and o['label'] == 0, 'fn': lambda o: o['pred'] == 0 and o['label'] == 1}
    sources = sorted({o['source'] for o in out if o['source']})
    out = [o for o in out if (not dataset or o['dataset'] == dataset) and (not source or o['source'] == source)
           and (not stratum or o['stratum'] == stratum) and (not verdict or keep[verdict](o))]
    out.sort(key=lambda o: (o['correct'] is not False, -abs((o['score'] if o['score'] is not None else .5) - .5),
                            o['dataset'], o['cid']))
    return {'total': len(out), 'sources': sources, 'rows': out[offset: offset + limit]}


@app.get('/api/calls/{call_id}')
def call_members(call_id: int):
    """Queries answered together in one batched call, in prompt order."""
    return rows('SELECT o.position, o.answer, s.* FROM output o JOIN sample s ON s.id = o.sample_id '
                'WHERE o.call_id = ? ORDER BY o.position', call_id)


@app.get('/api/shots')
def shots(purpose: str = ''):
    """One point per (run, query dataset): x = number of demos; plus D / M1-ign3 references."""
    found = run_summaries(purpose)
    points = []
    for r in found:
        for ds, m in r['datasets'].items():
            points.append({'run': r['name'], 'run_id': r['id'], 'model': r['model'], 'prompt_id': r['prompt_id'],
                           'prompt_name': r['prompt_name'], 'mode': r['mode'], 'form': r['form'], 'k': r['k'],
                           'dataset': ds, 'demo_domain': m['demo_domain'], 'readout': r['readout'],
                           'batch_size': r['batch_size'], 'eval_set': r['eval_set_name'],
                           **{k: m.get(k) for k in ('n', 'acc', 'acc_ci', 'auc', 'auc_ci', 'ap',
                                                    'primary_auc', 'primary_auc_ci', 'primary_ap')}})
    return {'points': points, 'references': references(found)}


@app.get('/img/{sample_id}/{key}')
def image(sample_id: int, key: str):
    # Only paths registered for a sample are served.
    found = rows('SELECT inputs FROM sample WHERE id = ?', sample_id)
    path = found and json.loads(found[0]['inputs']).get(key)
    if not path or not Path(path).is_file():
        raise HTTPException(404)
    return FileResponse(path)


@lru_cache(maxsize=2048)
def location_thumb(path, bbox_json, size=192):
    from PIL import Image, ImageDraw
    with Image.open(path) as im:
        im = im.convert('RGB')
        scale = size / max(im.size)
        im = im.resize((round(im.width * scale), round(im.height * scale)))
    x0, y0, x1, y1 = (v * scale for v in json.loads(bbox_json))
    pad = 3
    ImageDraw.Draw(im).rectangle((x0 - pad, y0 - pad, x1 + pad, y1 + pad), outline=(255, 214, 0), width=2)
    buf = io.BytesIO()
    im.save(buf, 'JPEG', quality=85)
    return buf.getvalue()


@app.get('/thumb/{sample_id}')
def thumb(sample_id: int):
    """Whole-image thumbnail with the candidate box: a pre-rendered `thumb` input if given,
    else drawn from full_t2 (or full_t1) and bbox."""
    found = rows('SELECT inputs, bbox FROM sample WHERE id = ?', sample_id)
    if not found:
        raise HTTPException(404)
    inputs = json.loads(found[0]['inputs'])
    if inputs.get('thumb') and Path(inputs['thumb']).is_file():
        return FileResponse(inputs['thumb'])
    full = inputs.get('full_t2') or inputs.get('full_t1')
    if not full or not found[0]['bbox'] or not Path(full).is_file():
        raise HTTPException(404)
    return Response(location_thumb(full, found[0]['bbox']), media_type='image/jpeg')


@app.get('/')
def index():
    return FileResponse(STATIC / 'index.html')


@app.get('/api/walkthrough')
def walkthrough_catalog():
    """Read offline indexes only; this mode never opens the audit database."""
    scopes = []
    for dataset, splits in [('second', ('val', 'test')), ('sysu', ('val', 'test')),
                            ('levir', ('val', 'test')), ('whu', ('test',))]:
        for split in splits:
            path = WALKTHROUGH_DIR / dataset / split / 'index.json' if WALKTHROUGH_DIR else None
            row = json.loads(path.read_text()) if path and path.is_file() else None
            stats_path = path.with_name('stats.json') if path else None
            statistics = json.loads(stats_path.read_text()) if stats_path and stats_path.is_file() else None
            if row and statistics and row.get('version') == statistics.get('version') == 2 and row.get('pool') == statistics.get('pool'):
                row['statistics'] = {k: statistics[k] for k in ('complete', 'analyzed', 'expected', 'averages', 'aggregates', 'seed')}
            else:
                row = None
            scopes.append(dict(dataset=dataset, split=split, ready=bool(row and row['images']), index=row))
    return dict(configured=WALKTHROUGH_DIR is not None, scopes=scopes,
                message='未生成：请先构建离线图像并设置 WALKTHROUGH_DIR。')


def walkthrough_path(relative):
    if WALKTHROUGH_DIR is None:
        raise HTTPException(404, '未生成：未设置 WALKTHROUGH_DIR')
    path = (WALKTHROUGH_DIR / relative).resolve()
    if not path.is_relative_to(WALKTHROUGH_DIR) or path.suffix not in ('.json', '.png') or not path.is_file():
        raise HTTPException(404, '离线产物不存在')
    return path


@app.get('/api/walkthrough/{dataset}/{split}/{image}')
def walkthrough_image(dataset: str, split: str, image: str):
    inventory = json.loads(walkthrough_path(f'{dataset}/{split}/index.json').read_text())
    if inventory.get('version') != 2 or image not in [Path(name).stem for name in inventory['images']]:
        raise HTTPException(404, '此图不在已生成的小图池内')
    return json.loads(walkthrough_path(f'{dataset}/{split}/{image}/walkthrough.json').read_text())


@app.get('/walkthrough-assets/{relative:path}')
def walkthrough_asset(relative: str):
    return FileResponse(walkthrough_path(relative))
