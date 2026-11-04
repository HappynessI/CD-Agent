"""SQLite store for MLLM verifier runs: prompts, demo pools, dev/test sets, ICL demo sets,
raw outputs, review results and metrics.

Preparation scripts register samples and sets; inference scripts write through RunRecorder;
app.py reads everything and writes only the review fields. A minimal ICL run:

    from store import RunRecorder, icl_template
    with RunRecorder('verifier.db', name='second_dev_A_k8_s0', model='qwen3-vl-plus',
                     prompt_name='second_sys_v1', prompt_text=SYSTEM, demos=demos, mode='few',
                     form='A', pool='exp6k_demos_second', eval_set='exp6k_dev_second',
                     readout='verbal_confidence', strategy='balanced random', seed=0,
                     template=icl_template(len(demos), SYSTEM, form='A')) as rec:
        for q in queries:
            rec.record(q, answer='no', confidence=80, raw_text=text,
                       usage=dict(input_tokens=..., output_tokens=..., cost=...))

Batched asking (all candidates of one image in one call):
        rec.record_batch(queries, raw_text, per_query=[...], group_key=image, usage={...})

Samples are dicts with dataset, split, cid and inputs; the optional keys are the sample
columns in schema.sql (image, source, label, rationale, pseudo_type, change_kind,
semantic_text, area, det_mean, m1_score, bbox). Anything else is kept in meta.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = Path(__file__).with_name('schema.sql')
FRAMING = {'t1': 'T1, earlier', 't2': 'T2, later',
           'context_t1': 'T1, earlier, wider context', 'context_t2': 'T2, later, wider context',
           'full_t1': 'T1, whole image', 'full_t2': 'T2, whole image', 'mask': 'binary change mask'}
COLUMNS = ('image', 'source', 'label', 'rationale', 'pseudo_type', 'change_kind', 'semantic_text',
           'area', 'det_mean', 'm1_score', 'bbox')
ALIASES = {'det_mean_prob': 'det_mean'}
SET_KEYS = ('stratum', 'weight')
STRATA = ('pos2', 'pos_other', 'hard_neg', 'rand_neg')
FORM_KEYS = {'A': ('t1', 't2'), 'B': ('full_t1', 'full_t2', 'mask'), 'AB': ('t1', 't2', 'full_t1', 'full_t2', 'mask')}


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def ensure_prompt_columns(db):
    columns = {r[1] for r in db.execute('PRAGMA table_info(prompt_version)')}
    for name in ('grp', 'title', 'kind'):
        if name not in columns:
            constraint = " CHECK (kind IN ('template', 'rendered'))" if name == 'kind' else ''
            db.execute(f'ALTER TABLE prompt_version ADD COLUMN {name} TEXT{constraint}')


def connect(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA.read_text())
    with db:
        ensure_prompt_columns(db)
    if db.execute('PRAGMA user_version').fetchone()[0] < 1:
        # Reviews used to live on sample_set_item (per pool); move them to sample_review once.
        # A sample reviewed in several pools keeps its most recent review.
        with db:
            db.execute('INSERT OR IGNORE INTO sample_review (sample_id, status, note, reviewed_at) '
                       'SELECT sample_id, review_status, review_note, reviewed_at FROM sample_set_item '
                       'WHERE review_status IS NOT NULL OR review_note IS NOT NULL ORDER BY reviewed_at DESC')
            db.execute('PRAGMA user_version = 1')
    return db


def digest(value):
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def add_prompt(db, name, text, parent_id=None, note=None, grp=None, title=None, kind=None):
    if kind not in (None, 'template', 'rendered'):
        raise ValueError('kind must be template, rendered or None')
    pid = digest(text)
    db.execute('INSERT OR IGNORE INTO prompt_version '
               '(id, name, text, parent_id, note, created_at, grp, title, kind) VALUES (?,?,?,?,?,?,?,?,?)',
               (pid, name, text, parent_id, note, now(), grp, title, kind))
    return pid


def add_sample(db, s):
    """Insert or update one sample (keyed by dataset, split, cid); returns its id."""
    s = {ALIASES.get(k, k): v for k, v in s.items()}
    if s.get('semantic_t1') and s.get('semantic_t2'):
        s['semantic_text_long'] = s.get('semantic_text')
        s['semantic_text'] = semantic_line(s['semantic_t1'], s['semantic_t2'], s.get('semantic_full'))
    fixed = ('dataset', 'split', 'cid', 'inputs') + COLUMNS + SET_KEYS
    meta = {k: v for k, v in s.items() if k not in fixed}
    values = {k: s.get(k) for k in COLUMNS}
    values['label'] = None if values['label'] is None else int(values['label'])
    values['bbox'] = None if values['bbox'] is None else json.dumps([float(v) for v in values['bbox']])
    cols = ('dataset', 'split', 'cid', *COLUMNS, 'inputs', 'meta')
    row = (s['dataset'], s['split'], str(s['cid']), *values.values(), json.dumps(s['inputs']),
           json.dumps(meta, default=str, ensure_ascii=False) if meta else None)
    db.execute(f'INSERT INTO sample ({", ".join(cols)}) VALUES ({", ".join("?" * len(cols))}) '
               f'ON CONFLICT (dataset, split, cid) DO UPDATE SET '
               + ', '.join(f'{c} = COALESCE(excluded.{c}, {c})' for c in cols[3:]), row)
    return db.execute('SELECT id FROM sample WHERE dataset=? AND split=? AND cid=?',
                      (s['dataset'], s['split'], str(s['cid']))).fetchone()[0]


def add_sample_set(db, name, purpose, dataset, samples, note=None):
    """Register a demo pool / dev set / test set. Each sample may carry stratum and weight.
    Re-registering updates strata and weights; reviews are per sample and are never touched."""
    db.execute('INSERT INTO sample_set (name, purpose, dataset, note, created_at) VALUES (?,?,?,?,?) '
               'ON CONFLICT (name) DO UPDATE SET purpose = excluded.purpose, dataset = excluded.dataset, '
               'note = COALESCE(excluded.note, note)', (name, purpose, dataset, note, now()))
    set_id = set_id_of(db, name)
    for s in samples:
        db.execute('INSERT INTO sample_set_item (set_id, sample_id, stratum, weight) VALUES (?,?,?,?) '
                   'ON CONFLICT (set_id, sample_id) DO UPDATE SET stratum = excluded.stratum, weight = excluded.weight',
                   (set_id, add_sample(db, s), s.get('stratum'), float(s.get('weight', 1.0))))
    return set_id


def set_id_of(db, name):
    if name is None:
        return None
    row = db.execute('SELECT id FROM sample_set WHERE name = ?', (name,)).fetchone()
    if row is None:
        raise KeyError(f'unknown sample set {name!r}; register it with add_sample_set first')
    return row[0]


def add_demo_set(db, demos, mode, form='A', pool=None, name=None, strategy=None, seed=None, note=None,
                 shown_labels=None):
    """demos: ordered sample dicts. shown_labels defaults to yes/no from each demo's label."""
    ids = [add_sample(db, d) for d in demos]
    shown = shown_labels or [None if d.get('label') is None else ('yes' if d['label'] else 'no') for d in demos]
    did = digest({'mode': mode, 'form': form, 'items': ids, 'shown': shown})
    db.execute('INSERT OR IGNORE INTO demo_set VALUES (?,?,?,?,?,?,?,?,?,?)',
               (did, name or f'{mode}_{form}_k{len(ids)}_{did[:6]}', mode, form, len(ids), set_id_of(db, pool),
                strategy, seed, note, now()))
    db.executemany('INSERT OR IGNORE INTO demo_set_item VALUES (?,?,?,?)',
                   [(did, i, sid, lab) for i, (sid, lab) in enumerate(zip(ids, shown))])
    return did


# ---- message templates (images and answers as refs, resolved by the UI) ----

def image_parts(ref, keys):
    out = []
    for key in keys:
        out += [{'type': 'text', 'text': f'{ref} ({FRAMING[key]}):'}, {'type': 'image', 'ref': f'{ref}.{key}'}]
    return out


def zero_shot_template(prompt, view='near'):
    """exp3 / exp6j layout: Image 1 (T1), Image 2 (T2) [, context pair], then the prompt."""
    content = []
    for i, key in enumerate(('t1', 't2', 'context_t1', 'context_t2')[: 4 if view == 'context' else 2]):
        content += [{'type': 'text', 'text': f'Image {i + 1} ({FRAMING[key]}):'},
                    {'type': 'image', 'ref': f'query[0].{key}'}]
    return [{'role': 'user', 'content': content + [{'type': 'text', 'text': prompt}]}]


def icl_template(k, system, form='A', batch=1, query_form='A'):
    """Suggested ICL layout: system prompt; k demos in the given form, each followed by its
    answer and rationale (and SECOND semantics for form B); then the query/queries.
    Scripts that lay out differently should pass their own template."""
    content = []
    for i in range(k):
        ref = f'demo[{i}]'
        content += image_parts(ref, FORM_KEYS[form])
        if form != 'A':
            content.append({'type': 'text', 'ref': f'{ref}.semantic'})
        content.append({'type': 'text', 'ref': f'{ref}.answer'})
    for j in range(batch):
        content += image_parts(f'query[{j}]', FORM_KEYS[query_form])
    return [{'role': 'system', 'content': [{'type': 'text', 'text': system}]},
            {'role': 'user', 'content': content}]


def git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=Path(__file__).parent,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class RunRecorder:
    """readout: token_prob (P(yes) from logits) | verbal_confidence | answer_only."""

    def __init__(self, db_path, name, model, prompt_text, template, prompt_name=None, demos=(),
                 mode='zero', form='A', pool=None, eval_set=None, strategy=None, seed=None, experiment=None,
                 readout=None, view='near', packing='multi', batch_size=1, batch_by=None, currency='CNY',
                 config=None, prompt_parent=None, prompt_note=None, replace=False):
        self.db = connect(db_path)
        with self.db:
            if replace:
                delete_run(self.db, name)
            pid = add_prompt(self.db, prompt_name or name, prompt_text, prompt_parent, prompt_note)
            did = add_demo_set(self.db, list(demos), mode if demos else 'zero', form, pool,
                               strategy=strategy, seed=seed)
            cur = self.db.execute(
                'INSERT INTO run (name, experiment, model, prompt_id, demo_set_id, eval_set_id, readout, view, '
                'packing, batch_size, batch_by, currency, template, git_commit, config, created_at) '
                'VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (name, experiment, model, pid, did, set_id_of(self.db, eval_set), readout, view, packing,
                 batch_size, batch_by, currency, json.dumps(template, ensure_ascii=False), git_commit(),
                 json.dumps(config, default=str) if config else None, now()))
            self.run_id = cur.lastrowid
        self.calls = 0
        self.usage = {'api_calls': 0, 'input_tokens': 0, 'output_tokens': 0, 'cost': 0.0}
        self.has_usage = False

    def _count(self, usage):
        self.usage['api_calls'] += 1
        if usage:
            self.has_usage = True
            for k in ('input_tokens', 'output_tokens', 'cost'):
                self.usage[k] += usage.get(k) or 0

    def record(self, sample, call_id=None, position=None, p_yes=None, logit_diff=None, yes_no_mass=None,
               answer=None, confidence=None, raw_text=None, usage=None, **extra):
        """One query. usage (input_tokens, output_tokens, cost) counts one API call."""
        sid = add_sample(self.db, sample)
        if call_id is None:
            self._count(usage)
        self.db.execute('INSERT OR REPLACE INTO output VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                        (self.run_id, sid, call_id, position, p_yes, logit_diff, yes_no_mass, answer,
                         confidence, raw_text, json.dumps(extra, default=str) if extra else None))

    def record_batch(self, queries, raw_text, per_query, group_key=None, usage=None, **meta):
        """One call answering several queries; per_query[i] holds record() kwargs for queries[i]."""
        usage = usage or {}
        cur = self.db.execute(
            'INSERT INTO call (run_id, idx, group_key, raw_text, input_tokens, output_tokens, cost, meta) '
            'VALUES (?,?,?,?,?,?,?,?)',
            (self.run_id, self.calls, group_key, raw_text, usage.get('input_tokens'), usage.get('output_tokens'),
             usage.get('cost'), json.dumps(meta, default=str) if meta else None))
        self.calls += 1
        self._count(usage)
        for pos, (q, kw) in enumerate(zip(queries, per_query, strict=True)):
            self.record(q, call_id=cur.lastrowid, position=pos, **kw)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if exc[0] is None:
            u = self.usage
            self.db.execute('UPDATE run SET api_calls=?, input_tokens=?, output_tokens=?, cost=? WHERE id=?',
                            (u['api_calls'], u['input_tokens'] if self.has_usage else None,
                             u['output_tokens'] if self.has_usage else None,
                             u['cost'] if self.has_usage else None, self.run_id))
            self.db.commit()
        else:
            self.db.rollback()
        self.db.close()


def delete_run(db, name):
    row = db.execute('SELECT id FROM run WHERE name=?', (name,)).fetchone()
    if row:
        for table in ('output', 'call'):
            db.execute(f'DELETE FROM {table} WHERE run_id=?', (row[0],))
        db.execute('DELETE FROM run WHERE id=?', (row[0],))


def set_review(db, sample_id, status, note):
    """Review is per sample, so it shows in every pool that contains the sample."""
    db.execute('INSERT INTO sample_review VALUES (?,?,?,?) ON CONFLICT (sample_id) DO UPDATE SET '
               'status = excluded.status, note = excluded.note, reviewed_at = excluded.reviewed_at',
               (sample_id, status or None, note or None, now()))


def semantic_line(t1, t2, full=None):
    """Form B text: one line for the candidate, e.g. "T1 非植被地表 → T2 树木（占候选 74%）",
    from its per-date class histograms ([{name, fraction}, ...], largest first); plus, when
    `full` (the whole image's two histograms) is given, a second line with the whole-image
    main classes. SECOND labels classes only inside changed areas; that caveat is shown
    as a tooltip in the UI, not in the text."""
    name = lambda r: '无标注地类' if r['name'].startswith('无变化') else r['name']
    f1, f2 = round(t1[0]['fraction'] * 100), round(t2[0]['fraction'] * 100)
    if f1 == f2:
        line = f'T1 {name(t1[0])} → T2 {name(t2[0])}（占候选 {f1}%）'
    else:
        line = f'T1 {name(t1[0])}（占候选 {f1}%）→ T2 {name(t2[0])}（占候选 {f2}%）'
    if full:
        top = lambda h: '、'.join(f"{name(r)} {r['fraction']:.0%}" for r in h[:3])
        line += f'\n整图两期主要类别 T1：{top(full[0])}；T2：{top(full[1])}'
    return line


def with_semantic(row):
    """Replace semantic_text by semantic_line when the sample's meta has the histograms
    (exp6k semantic_t1 / semantic_t2), keeping the long text as semantic_text_long."""
    meta = json.loads(row['meta']) if isinstance(row.get('meta'), str) else (row.get('meta') or {})
    if meta.get('semantic_t1') and meta.get('semantic_t2'):
        row['semantic_text_long'] = meta.get('semantic_text_long', row.get('semantic_text'))
        row['semantic_text'] = semantic_line(meta['semantic_t1'], meta['semantic_t2'], meta.get('semantic_full'))
    return row


# ---- metrics (computed on read) ----

def prediction(row):
    if row['answer'] in ('yes', 'no'):
        return int(row['answer'] == 'yes')
    return None if row['p_yes'] is None else int(row['p_yes'] > 0.5)


def score(row):
    """Ranking score for AUC / AP: P(yes) when the readout has it, else the verbal
    confidence turned into P(yes), else the bare answer."""
    if row['p_yes'] is not None:
        return row['p_yes']
    if row['answer'] in ('yes', 'no'):
        c = row['confidence']
        if c is None:
            return float(row['answer'] == 'yes')
        c = min(max(c, 0), 100) / 100
        return c if row['answer'] == 'yes' else 1 - c
    return None


def auc(y, s, w=None):
    """Weighted ROC AUC with ties counted as one half."""
    import numpy as np
    y, s = np.asarray(y) == 1, np.asarray(s, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    if not y.any() or y.all():
        return None
    _, inv = np.unique(s, return_inverse=True)
    wp, wn = np.bincount(inv, w * y), np.bincount(inv, w * ~y)
    below = np.cumsum(wn) - wn
    return float((wp * (below + wn / 2)).sum() / (wp.sum() * wn.sum()))


def ap(y, s, w=None):
    """Weighted average precision (step-wise over distinct thresholds, as sklearn)."""
    import numpy as np
    y, s = np.asarray(y) == 1, np.asarray(s, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)
    if not y.any():
        return None
    _, inv = np.unique(-s, return_inverse=True)
    wp, wa = np.bincount(inv, w * y), np.bincount(inv, w)
    precision = np.cumsum(wp) / np.cumsum(wa)
    return float((wp * precision).sum() / wp.sum())


def metrics(rows, n_boot=2000, seed=0):
    """rows: dicts with label, pred, score, weight, stratum, group (source image).
    acc / auc / ap use the sampling weights. primary_* is pos2 vs all negatives.
    CIs: 2000 bootstrap resamples of whole images (as exp8s3)."""
    import numpy as np
    rows = [r for r in rows if r['label'] is not None]
    out = {'n': len(rows), 'n_pos': sum(r['label'] for r in rows)}
    if not rows:
        return out
    y = np.array([r['label'] for r in rows])
    w = np.array([r.get('weight') or 1.0 for r in rows], float)
    pred = np.array([np.nan if r['pred'] is None else r['pred'] for r in rows], float)
    sc = np.array([np.nan if r['score'] is None else r['score'] for r in rows], float)
    strata = np.array([r.get('stratum') or '' for r in rows])
    has_p, has_s = ~np.isnan(pred), ~np.isnan(sc)
    primary = has_s & ((strata == 'pos2') | (y == 0)) if (strata == 'pos2').any() else None

    def acc(idx):
        m = idx[has_p[idx]]
        return float(np.average(pred[m] == y[m], weights=w[m])) if len(m) else None

    def scored(fn, idx, mask):
        m = idx[mask[idx]]
        return fn(y[m], sc[m], w[m]) if len(m) else None

    every = np.arange(len(rows))
    out.update(acc=acc(every),
               yes_rate=float(np.average(pred[has_p], weights=w[has_p])) if has_p.any() else None,
               auc=scored(auc, every, has_s), ap=scored(ap, every, has_s),
               primary_auc=scored(auc, every, primary) if primary is not None else None,
               primary_ap=scored(ap, every, primary) if primary is not None else None,
               strata={k: {'n': int((strata == k).sum()),
                           'acc': acc(every[strata == k]),
                           'yes_rate': float(pred[(strata == k) & has_p].mean()) if ((strata == k) & has_p).any() else None}
                       for k in STRATA if (strata == k).any()})
    # Image-cluster bootstrap.
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault(r.get('group') or f'#{i}', []).append(i)
    members = [np.array(v) for v in groups.values()]
    rng = np.random.default_rng(seed)
    boots = {'acc': [], 'auc': [], 'primary_auc': []}
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(members), len(members))])
        boots['acc'].append(acc(idx))
        boots['auc'].append(scored(auc, idx, has_s))
        if primary is not None:
            boots['primary_auc'].append(scored(auc, idx, primary))
    for k, vals in boots.items():
        vals = [v for v in vals if v is not None]
        out[f'{k}_ci'] = [float(np.quantile(vals, .025)), float(np.quantile(vals, .975))] if vals else None
    return out
