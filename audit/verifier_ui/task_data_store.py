"""Persistence for task data; never migrate or modify legacy tables."""
from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

COLORS = ['蓝', '红或橙', '白或浅灰', '深灰或黑', '绿', '其他或混色', '看不清']
SUBTYPES = ['临时工棚或彩钢板房', '厂房或仓库', '住宅或一般建筑', '看不清']
CLASS_NAMES = ['未变化 / 无地类', '低矮植被', '非植被地表', '树', '水', '建筑', '运动场']
PALETTE = [(255, 255, 255), (0, 128, 0), (128, 128, 128), (0, 255, 0),
           (0, 0, 255), (128, 0, 0), (255, 0, 0)]
PURPOSE_NAMES = {'ecology': '生态保护核查', 'urban': '城市扩张监测', 'water_loss': '河湖侵占排查',
                 'water_gain': '新增水面', 'bare_soil': '扬尘治理', 'greening': '绿化成效'}
_anchors = {}


def retry_io(fn):
    """Three attempts for transient OSError on the shared disk."""
    for attempt in range(3):
        try:
            return fn()
        except OSError:
            if attempt == 2:
                raise
            time.sleep(.1 * (attempt + 1))


def read_text(path):
    return retry_io(lambda: Path(path).read_text(encoding='utf-8'))


def write_text(path, value):
    def write():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(value, encoding='utf-8')
    retry_io(write)


def dumps(value):
    return json.dumps(value, ensure_ascii=False)


def now():
    return datetime.now(timezone.utc).isoformat(timespec='microseconds')


def connect(path=None):
    path = str(path or os.environ.get('TASK_DATA_DB') or os.environ.get('VERIFIER_DB') or
               Path(__file__).with_name('verifier.db'))
    uri = path.startswith('file:')
    if uri and 'mode=memory' in path and path not in _anchors:
        _anchors[path] = sqlite3.connect(path, uri=True, check_same_thread=False)
    db = sqlite3.connect(path, uri=uri)
    db.row_factory = sqlite3.Row
    schema = read_text(Path(__file__).with_name('schema.sql')).split('-- TASK_DATA_SCHEMA\n', 1)[1]
    db.executescript(schema)
    return db


def metadata(db):
    return {r['key']: json.loads(r['value']) for r in db.execute('SELECT * FROM task_data_meta')}


def set_meta(db, key, value):
    db.execute('INSERT INTO task_data_meta VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value',
               (key, dumps(value)))


def mappings(db):
    return [dict(purpose=r['purpose'], name=PURPOSE_NAMES.get(r['purpose'], r['purpose']),
                 value=json.loads(r['value']), generated_value=json.loads(r['generated_value']))
            for r in db.execute('SELECT * FROM task_mapping ORDER BY rowid')]


def save_mapping(db, purpose, value, annotator):
    old = db.execute('SELECT value FROM task_mapping WHERE purpose=?', (purpose,)).fetchone()
    if not old:
        raise KeyError(purpose)
    normalized = {k: sorted(set(value[k])) for k in ('t1', 't2')}
    if any(type(v) is not int or v not in range(1, 7) for vs in normalized.values() for v in vs):
        raise ValueError('类别只能为 1–6')
    if json.loads(old['value']) != normalized:
        db.execute('INSERT INTO task_mapping_history(purpose,before_value,after_value,annotator,modified_at) '
                   'VALUES (?,?,?,?,?)', (purpose, old['value'], dumps(normalized), annotator, now()))
        db.execute('UPDATE task_mapping SET value=? WHERE purpose=?', (dumps(normalized), purpose))


def examples(db):
    return [dict(**{k: v for k, v in json.loads(r['payload']).items() if k != 'assets'},
                 verdict=r['verdict'], note=r['note'], annotator=r['annotator'], reviewed_at=r['reviewed_at'])
            for r in db.execute('SELECT * FROM task_example ORDER BY rowid')]


def components(db, pass_number=1, image='', direction=''):
    """Blind-pass responses contain neither first-pass labels nor prefill."""
    result = []
    for r in db.execute('SELECT * FROM task_component ORDER BY id'):
        if pass_number == 2 and not r['second_pass']:
            continue
        p = json.loads(r['payload'])
        if (image and p['image'] != image) or (direction and p['direction'] != direction):
            continue
        p.pop('assets', None)
        if pass_number == 2:
            p.pop('prefill_color', None)
        prefix = 'second_' if pass_number == 2 else ''
        p.update({k: r[prefix + k] for k in ('color', 'subtype', 'annotator', 'labeled_at')})
        p['second_pass'] = bool(r['second_pass'])
        result.append(p)
    return result


def save_review(db, ident, verdict, note, annotator):
    r = db.execute('SELECT * FROM task_example WHERE id=?', (ident,)).fetchone()
    if not r:
        raise KeyError(ident)
    if verdict not in ('reasonable', 'unreasonable', None):
        raise ValueError('无效审查结果')
    before = {k: r[k] for k in ('verdict', 'note', 'annotator', 'reviewed_at')}
    stamp = now()
    db.execute('INSERT INTO task_edit(kind,target,pass_number,before_value,annotator,modified_at) '
               'VALUES (?,?,?,?,?,?)', ('review', ident, 1, dumps(before), annotator, stamp))
    db.execute('UPDATE task_example SET verdict=?,note=?,annotator=?,reviewed_at=? WHERE id=?',
               (verdict, note, annotator, stamp, ident))


def save_label(db, ident, pass_number, color, subtype, annotator):
    r = db.execute('SELECT * FROM task_component WHERE id=?', (ident,)).fetchone()
    if pass_number not in (1, 2):
        raise ValueError('pass_number 必须为 1 或 2')
    if not r or (pass_number == 2 and not r['second_pass']):
        raise KeyError(ident)
    if color not in COLORS or subtype not in SUBTYPES:
        raise ValueError('请同时选择屋顶颜色和子类')
    prefix = 'second_' if pass_number == 2 else ''
    keys = [prefix + k for k in ('color', 'subtype', 'annotator', 'labeled_at')]
    stamp = now()
    db.execute('INSERT INTO task_edit(kind,target,pass_number,before_value,annotator,modified_at) '
               'VALUES (?,?,?,?,?,?)', ('component', ident, pass_number,
                                       dumps({k: r[k] for k in keys}), annotator, stamp))
    db.execute(f'UPDATE task_component SET {",".join(k + "=?" for k in keys)} WHERE id=?',
               (color, subtype, annotator, stamp, ident))


def undo(db, kind, annotator, pass_number=1, target=None):
    sql = 'SELECT * FROM task_edit WHERE kind=? AND annotator=? AND pass_number=? AND undone_at IS NULL'
    args = [kind, annotator, pass_number]
    if target:
        sql += ' AND target=?'
        args.append(target)
    r = db.execute(sql + ' ORDER BY id DESC LIMIT 1', args).fetchone()
    if not r:
        return None
    before = json.loads(r['before_value'])
    table = 'task_example' if kind == 'review' else 'task_component'
    db.execute(f'UPDATE {table} SET {",".join(k + "=?" for k in before)} WHERE id=?',
               (*before.values(), r['target']))
    db.execute('UPDATE task_edit SET undone_at=? WHERE id=?', (now(), r['id']))
    return r['target']


def exports(db):
    history = [dict(id=r['id'], purpose=r['purpose'], before=json.loads(r['before_value']),
                    after=json.loads(r['after_value']), annotator=r['annotator'], modified_at=r['modified_at'])
               for r in db.execute('SELECT * FROM task_mapping_history ORDER BY id')]
    review = dict(examples=examples(db), mappings=mappings(db), mapping_history=history,
                  metadata=metadata(db), warning='映射改动需要重新生成真值后再审')
    labels, paired = [], []
    for r in db.execute('SELECT * FROM task_component ORDER BY id'):
        p = json.loads(r['payload'])
        p.pop('assets', None)
        p.update({k: r[k] for k in ('color', 'subtype', 'annotator', 'labeled_at')})
        p['prefill_changed'] = None if r['color'] is None else r['color'] != p['prefill_color']
        p['second_pass'] = bool(r['second_pass'])
        p['second_result'] = ({k: r['second_' + k] for k in ('color', 'subtype', 'annotator', 'labeled_at')}
                              if r['second_labeled_at'] else None)
        p['ignore'] = r['color'] in ('其他或混色', '看不清') or r['subtype'] == '看不清'
        labels.append(p)
        if r['labeled_at'] and r['second_labeled_at']:
            paired.append(r)
    agreement = {'paired': len(paired), 'sampled': sum(p['second_pass'] for p in labels)}
    for key in ('color', 'subtype', 'both'):
        agreement[key] = (sum((r['color'] == r['second_color'] if key == 'color' else
                              r['subtype'] == r['second_subtype'] if key == 'subtype' else
                              r['color'] == r['second_color'] and r['subtype'] == r['second_subtype'])
                             for r in paired) / len(paired)) if paired else None
    return review, dict(components=labels, agreement=agreement,
                        ignore_rule='其他或混色、看不清：整个组件设为忽略区',
                        ignored=sum(p['ignore'] for p in labels),
                        labeled=sum(p['labeled_at'] is not None for p in labels))
