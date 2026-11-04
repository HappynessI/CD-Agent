"""按既定名称规则补充提示词元数据；默认只读，--apply 备份后写入。"""
from __future__ import annotations

import argparse
import re
import sqlite3
from contextlib import closing
from pathlib import Path

from store import ensure_prompt_columns


GROUPS = {'sysu': 'SYSU', 'second': 'SECOND', 'sysu_second': 'SYSU + SECOND 共用',
          'levir': 'LEVIR', 'whu': 'WHU', 'levir_whu': 'LEVIR + WHU 共用'}
MODELS = {'qwen': 'Qwen3.5-4B', 'dars1b': 'DA-RS-1B', 'dars4b': 'DA-RS-4B'}
VIEWS = {'near': '近景', 'context': '近景 + 整图'}


def metadata(name):
    if m := re.fullmatch(r'exp3_(sysu_second|levir_whu)', name):
        return m[1], '零样本 · exp3 · 原版', 'template'
    if m := re.fullmatch(r'exp6j_sysu_second_step2_(near|context)', name):
        title = '近景（exp6l 的 Z0）' if m[1] == 'near' else VIEWS[m[1]]
        return 'sysu_second', f'零样本 · exp6j · {title}', 'template'
    if m := re.fullmatch(r'exp6v2_([abc])', name):
        return 'sysu_second', f'零样本 · V2 · 写法 {m[1]}', 'template'
    if m := re.fullmatch(r'(sysu|second|levir|whu)_sys_v1_(en|zh)', name):
        return m[1], '口径系统提示 · v1 · ' + ('英文' if m[2] == 'en' else '中文'), 'template'
    if m := re.fullmatch(r'exp6j_step([12])_(sysu_second|levir_whu)_(qwen|dars1b|dars4b)_'
                         r'(near|context)_(multi|sheet)(_expanded)?', name):
        step, grp, model, view, packing, expanded = m.groups()
        parts = ['渲染', 'exp6j ' + ('第一步' if step == '1' else '第二步'),
                 MODELS[model], VIEWS[view], '多图' if packing == 'multi' else '拼成一张图']
        if expanded:
            parts.append('已展开图像占位')
        return grp, ' · '.join(parts), 'rendered'
    if m := re.fullmatch(r'exp6v2_([abc])_qwen_rendered', name):
        return 'sysu_second', f'渲染 · V2 写法 {m[1]} · Qwen3.5-4B', 'rendered'
    return None


def plan(rows):
    metas = {r['id']: metadata(r['name']) for r in rows}
    unknown = [r['name'] for r in rows if metas[r['id']] is None]
    if unknown:
        raise ValueError('未匹配规则的 name：\n' + '\n'.join(unknown))
    errors = []
    for r in rows:
        grp, _, kind = metas[r['id']]
        if kind != 'rendered':
            continue
        parent = metas.get(r['parent_id'])
        if parent is None or parent[2] != 'template':
            errors.append(f"{r['name']}：父版本不存在或不是 template")
        elif grp != parent[0]:
            errors.append(f"{r['name']}：名称分组与父模板分组不一致")
    if errors:
        raise ValueError('\n'.join(errors))
    return metas


def migrate(path, apply=False):
    path = Path(path).resolve(strict=True)
    with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)) as source:
        source.row_factory = sqlite3.Row
        rows = [dict(r) for r in source.execute('SELECT * FROM prompt_version ORDER BY created_at, name')]
        metas = plan(rows)
        columns = {r[1] for r in source.execute('PRAGMA table_info(prompt_version)')}
        changes = [r for r in rows if tuple(r.get(k) for k in ('grp', 'title', 'kind')) != metas[r['id']]]
        print('name | grp（显示分组） | kind | title')
        for r in rows:
            grp, title, kind = metas[r['id']]
            print(f"{r['name']} | {grp}（{GROUPS[grp]}） | {kind} | {title}")
        print(f'共 {len(rows)} 条，待更新 {len(changes)} 条。')
        if not apply:
            print('dry-run：未写入数据库，未创建备份。')
            return
        if not changes and {'grp', 'title', 'kind'} <= columns:
            print('元数据已一致，无需写入。')
            return
        backup = Path(str(path) + '.bak-prompt-meta')
        # Exclusive creation refuses to overwrite an existing backup.
        with backup.open('xb'):
            pass
        with closing(sqlite3.connect(backup)) as target:
            source.backup(target)
        print(f'备份：{backup}')
    with closing(sqlite3.connect(path)) as db, db:
        db.execute('BEGIN IMMEDIATE')
        ensure_prompt_columns(db)
        db.executemany('UPDATE prompt_version SET grp=?, title=?, kind=? WHERE id=?',
                       [(*metas[r['id']], r['id']) for r in changes])
    print(f'已更新 {len(changes)} 条。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, required=True, help='待检查或迁移的 SQLite 库')
    parser.add_argument('--apply', action='store_true', help='备份后写入；默认 dry-run')
    args = parser.parse_args()
    migrate(args.db, args.apply)


if __name__ == '__main__':
    main()
