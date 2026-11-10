"""Prepare task review examples and SECOND components from labels and RGB only.

No inference modules are imported. T4.1 reads only the manifest's gt_mask, never
init_mask, fixed, rule, answers, or any model output. See README for CLI usage.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import defaultdict
from contextlib import closing
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

import task_data_store as store

SEED = 1016
MIN_COMPONENT_AREA = 64
MIN_CROP_SIZE = 96
SECOND_PASS_SHARE = .20
# Pillow HSV channels: H in [0,255] converted to degrees, S/V to [0,1].
HSV_DARK_V = .25
HSV_UNREADABLE_V = .04
HSV_LIGHT_S = .20
HSV_LIGHT_V = .65
HSV_COLOR_S = .25
HSV_BLUE = (180, 260)
HSV_RED_MAX = 45
HSV_RED_MIN = 330
HSV_GREEN = (70, 175)
EIGHT = np.ones((3, 3), dtype=bool)
DEFAULT_MAPPING = {
    'ecology': {'t1': [1, 3], 't2_not': [1, 3]},
    'urban': {'t1_not': [5, 6], 't2': [5, 6]},
    'water_loss': {'t1': [4], 't2_not': [4]},
    'water_gain': {'t1_not': [4], 't2': [4]},
    'bare_soil': {'t1': [1, 3], 't2': [2]},
    'greening': {'t1': [2, 5, 6], 't2': [1, 3]},
}


def read_image(path, mode=None):
    def read():
        with Image.open(path) as image:
            return image.convert(mode).copy() if mode else image.copy()
    return store.retry_io(read)


def save_image(path, image):
    def write():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
    store.retry_io(write)


def normalize_mapping(spec):
    return {k: sorted(spec[k] if k in spec else set(range(1, 7)) - set(spec[k + '_not']))
            for k in ('t1', 't2')}


def select_examples(rows, seed=SEED):
    """Rank quartiles over nonempty truth; pick small, large, then random, no repeats."""
    eligible = sorted((dict(r) for r in rows if r['area'] > 0), key=lambda r: (r['area'], r['id']))
    rng = random.Random(seed)
    if len(eligible) < 3:
        return [dict(r, selection='数量不足，全部展示') for r in eligible]
    quarter = max(1, math.ceil(len(eligible) / 4))
    small = rng.choice(eligible[:quarter])
    large = rng.choice(eligible[-quarter:])
    remaining = [r for r in eligible if r['id'] not in {small['id'], large['id']}]
    return [dict(small, selection='后四分之一（小）'), dict(large, selection='前四分之一（大）'),
            dict(rng.choice(remaining), selection='随机')]


def prefill_color(rgb, mask):
    if not mask.any():
        return '看不清'
    hsv = np.asarray(Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert('HSV'))
    h, s, v = np.median(hsv[mask], axis=0) / np.array([255 / 360, 255, 255])
    if v < HSV_UNREADABLE_V:
        return '看不清'
    if v < HSV_DARK_V:
        return '深灰或黑'
    if s <= HSV_LIGHT_S and v >= HSV_LIGHT_V:
        return '白或浅灰'
    if s >= HSV_COLOR_S:
        if HSV_BLUE[0] <= h <= HSV_BLUE[1]:
            return '蓝'
        if h <= HSV_RED_MAX or h >= HSV_RED_MIN:
            return '红或橙'
        if HSV_GREEN[0] <= h <= HSV_GREEN[1]:
            return '绿'
    return '其他或混色'


def blind_sample(ids, seed=SEED):
    ids = sorted(ids)
    return set(random.Random(seed).sample(ids, round(len(ids) * SECOND_PASS_SHARE)))


def crop_box(bbox, size):
    width, height = size
    x0, y0, x1, y1 = bbox
    w, h = min(width, max(MIN_CROP_SIZE, x1 - x0 + 24)), min(height, max(MIN_CROP_SIZE, y1 - y0 + 24))
    x = min(max(0, (x0 + x1 - w) // 2), width - w)
    y = min(max(0, (y0 + y1 - h) // 2), height - h)
    return [x, y, x + w, y + h]


def mask_runs(mask):
    runs = []
    for y in np.flatnonzero(mask.any(axis=1)):
        edges = np.diff(np.r_[False, mask[y], False].astype(np.int8))
        runs.extend([int(y), int(x0), int(x1)] for x0, x1 in
                    zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    return runs


def component_records(name, a, b, rgb1, rgb2, paths, pool):
    records = []
    for direction, truth, rgb in [('new', (a != 5) & (b == 5), rgb2),
                                   ('demolished', (a == 5) & (b != 5), rgb1)]:
        labels, _ = ndi.label(truth, EIGHT)
        counts = np.bincount(labels.ravel())
        for number, slices in enumerate(ndi.find_objects(labels), 1):
            if slices is None or counts[number] < MIN_COMPONENT_AREA:
                continue
            ys, xs = slices
            local = labels[slices] == number
            bbox = [xs.start, ys.start, xs.stop, ys.stop]
            records.append(dict(id=f'{Path(name).stem}:{direction}:{number}', image=name,
                                direction=direction, pool=pool, bbox=bbox, area=int(counts[number]),
                                prefill_color=prefill_color(rgb[slices], local),
                                assets=dict(**paths, size=[a.shape[1], a.shape[0]],
                                            crop=crop_box(bbox, (a.shape[1], a.shape[0])),
                                            runs=[[y + ys.start, x0 + xs.start, x1 + xs.start]
                                                  for y, x0, x1 in mask_runs(local)])))
    return records


def put_dataset(db, examples, components, mapping, metadata):
    """Repeat imports preserve edits. A changed selection needs a fresh database."""
    for table, records in [('task_example', examples), ('task_component', components)]:
        existing = {r[0] for r in db.execute(f'SELECT id FROM {table}')}
        incoming = {r['id'] for r in records}
        if existing and existing != incoming:
            raise ValueError(f'{table} 的数据清单已变化，请使用新的 TASK_DATA_DB，避免覆盖已有审核')
    with db:
        for r in examples:
            db.execute('INSERT INTO task_example(id,payload) VALUES (?,?) ON CONFLICT(id) '
                       'DO UPDATE SET payload=excluded.payload', (r['id'], store.dumps(r)))
        second = blind_sample([r['id'] for r in components])
        for r in components:
            db.execute('INSERT INTO task_component(id,payload,second_pass) VALUES (?,?,?) '
                       'ON CONFLICT(id) DO UPDATE SET payload=excluded.payload,second_pass=excluded.second_pass',
                       (r['id'], store.dumps(r), int(r['id'] in second)))
        for purpose, spec in mapping.items():
            value = store.dumps(normalize_mapping(spec))
            db.execute('INSERT INTO task_mapping VALUES (?,?,?) ON CONFLICT(purpose) '
                       'DO UPDATE SET generated_value=excluded.generated_value', (purpose, value, value))
        for k, v in metadata.items():
            store.set_meta(db, k, v)
    return dict(examples=len(examples), components=len(components), second_pass=len(second), **metadata)


def jsonl(path):
    return [json.loads(line) for line in store.read_text(path).splitlines() if line.strip()]


def prepare_real(db, run_root, second_root, components_enabled=True, limit_images=None):
    run, second = Path(run_root), Path(second_root)
    stats = json.loads(store.read_text(run / 'construct/stats.json'))
    groups, notes = defaultdict(list), []
    manifests = [('R1', run / 'construct/r1/tasks/T4.3_report.jsonl'),
                 ('R2', run / 'construct/r2/tasks/T1.3xC3_report.jsonl'),
                 ('T4.1', run / 't41/tasks/T4.1_any.jsonl')]
    r3 = stats.get('r3', {})
    if r3.get('decision') == 'do':
        manifests.append(('R3', run / 'construct/r3/tasks/T4.4_report.jsonl'))
    else:
        notes.append(f"R3 未出卡：{r3.get('decision', '未生成统计')}；非空图数 {r3.get('nonempty_images', {})}")
    for task, path in manifests:
        if not store.retry_io(path.exists):
            notes.append(f'{task} 任务文件尚未生成；暂不出卡')
            continue
        for row in jsonl(path):
            meta = row.get('meta') or {}
            group = meta['purpose'] if task == 'R1' else meta['reference'] if task == 'R3' else ''
            # Actual ground truth only: do not use meta scores, init masks or model answers.
            mask = np.asarray(read_image(row['gt_mask'], 'L')) > 0
            assets = dict(t1=row['image_t1'], t2=row['image_t2'], gt=row['gt_mask'])
            if task != 'T4.1':
                name = Path(row['image_t1']).name
                assets.update(s1=str(second / 'slabel1' / name), s2=str(second / 'slabel2' / name))
            groups[(task, group)].append(dict(id=f'{task}:{group}:{row["id"]}', task=task,
                group=group, task_type=row['task_type'], instruction=row['instruction'],
                image=Path(row['image_t1']).name, area=int(mask.sum()), assets=assets,
                semantic_available=task != 'T4.1', min_m2=meta.get('min_m2')))
    examples, sampling = [], []
    # Include empty groups in the page, rather than silently omitting a purpose.
    expected = [('R1', p) for p in stats['mapping']] + [('R2', ''), ('T4.1', '')]
    if r3.get('decision') == 'do':
        expected += [('R3', 'water'), ('R3', 'playground')]
    for key in expected:
        rows = groups[key]
        selected = select_examples(rows)
        examples.extend(selected)
        sampling.append(dict(task=key[0], group=key[1], available=sum(r['area'] > 0 for r in rows),
                             selected=len(selected), insufficient=len(selected) < 3))
    components = []
    counts = {}
    if components_enabled:
        pools = [('second_dev', run / 'second_dev/names.txt', 60),
                 ('second_val128', run / 'second/second_val128.txt', 128)]
        seen = set()
        for pool, path, expected_count in pools:
            names = store.read_text(path).split()
            counts[pool] = len(names)
            if len(names) != expected_count:
                raise ValueError(f'{pool} 应有 {expected_count} 张，实际 {len(names)}')
            if seen.intersection(names):
                raise ValueError('开发集与报告集有重复图名')
            seen.update(names)
            if limit_images is not None:
                names = names[:limit_images]
            for name in names:
                a, b = [np.asarray(read_image(second / part / name)) for part in ('slabel1', 'slabel2')]
                paths = dict(t1=str(second / 'img1' / name), t2=str(second / 'img2' / name))
                rgb1, rgb2 = [np.asarray(read_image(paths[k], 'RGB')) for k in ('t1', 't2')]
                components.extend(component_records(name, a, b, rgb1, rgb2, paths, pool))
    if limit_images is not None:
        notes.append(f'轻量真实数据预览：每个图池仅前 {limit_images} 张；不是完整标注库')
    return put_dataset(db, examples, components, stats['mapping'],
                       dict(source='real', seed=SEED, sampling=sampling, notes=notes,
                            image_strategy='按登记路径从原图读取，轮廓与裁剪由页面接口生成',
                            image_pools=counts, r3=r3, preview=limit_images is not None,
                            components_enabled=components_enabled))


def prepare_synthetic(db, directory):
    directory = Path(directory).resolve()
    examples, components, sampling = [], [], []
    tasks = [('R1', 'ecology'), ('R2', ''), ('T4.1', '')]
    for task, group in tasks:
        candidates = []
        for i, side in enumerate((12, 30, 55)):
            size = 256
            a = np.zeros((size, size), np.uint8)
            b = np.zeros_like(a)
            a[40:40 + side, 50:50 + side] = 3 if task == 'R1' else 2
            b[40:40 + side, 50:50 + side] = 2 if task == 'R1' else 5
            a[145:210, 160:220], b[145:210, 160:220] = 2, 1
            rgb1, rgb2 = [], []
            for label, dest in [(a, rgb1), (b, rgb2)]:
                image = Image.new('RGB', (size, size), '#7d8974')
                draw = ImageDraw.Draw(image)
                for x in range(0, size, 48):
                    draw.rectangle((x, 0, x + 10, size), fill='#b9b6a4')
                draw.rectangle((35, 32, 120, 125), fill='#31534a' if label is a else '#59788c')
                draw.rectangle((155, 140, 225, 215), fill='#b6b3a6')
                dest.append(image)
            prefix = directory / f'{task}_{i}'
            assets = {}
            for key, image in [('t1', rgb1[0]), ('t2', rgb2[0]), ('s1', Image.fromarray(a)),
                               ('s2', Image.fromarray(b)), ('gt', Image.fromarray(((a != b) & (a != 0)).astype(np.uint8) * 255))]:
                path = prefix.with_name(prefix.name + '_' + key + '.png')
                save_image(path, image)
                assets[key] = str(path)
            # Synthetic truth follows the displayed task, not all colored regions.
            truth = (a == 3) & (b == 2) if task == 'R1' else (a != b) & (a != 0)
            if task == 'T4.1':
                truth = (a != 5) & (b == 5)
            save_image(assets['gt'], Image.fromarray(truth.astype(np.uint8) * 255))
            area = int(truth.sum())
            instruction = {'R1': 'For an ecological protection check, find vegetation destroyed between T1 and T2.',
                           'R2': 'Find all land-cover changes except changes between low vegetation and trees.',
                           'T4.1': 'Find building changes whose footprint is larger than 260 square meters.'}[task]
            candidates.append(dict(id=f'{task}:{group}:synthetic_{i}', task=task, group=group,
                                   task_type={'R1': 'T4.3', 'R2': 'T1.3xC3', 'T4.1': 'T4.1'}[task],
                                   instruction=instruction + ' Output a binary change mask.',
                                   image=f'synthetic_{task}_{i}.png', area=area, assets=assets,
                                   semantic_available=task != 'T4.1'))
        examples.extend(select_examples(candidates))
        sampling.append(dict(task=task, group=group, available=3, selected=3, insufficient=False))
    roof_colors = ['#2459bc', '#c54727', '#deded9', '#26282b', '#37935d']
    for number in range(4):
        name = f'synthetic_components_{number}.png'
        a, b = np.zeros((256, 256), np.uint8), np.zeros((256, 256), np.uint8)
        im1, im2 = Image.new('RGB', (256, 256), '#87977f'), Image.new('RGB', (256, 256), '#89977f')
        for j in range(5):
            x, y = 20 + (j % 3) * 77, 35 + (j // 3) * 125
            box = [x, y, x + 32, y + 26]
            if (j + number) % 2:
                a[y:y + 26, x:x + 32] = 5
                ImageDraw.Draw(im1).rectangle(box, fill=roof_colors[j], outline='#343735', width=2)
            else:
                b[y:y + 26, x:x + 32] = 5
                ImageDraw.Draw(im2).rectangle(box, fill=roof_colors[j], outline='#343735', width=2)
        paths = {}
        for key, image in [('t1', im1), ('t2', im2), ('s1', Image.fromarray(a)), ('s2', Image.fromarray(b))]:
            path = directory / (Path(name).stem + '_' + key + '.png')
            save_image(path, image)
            paths[key] = str(path)
        components.extend(component_records(name, a, b, np.asarray(im1), np.asarray(im2),
                                             {k: paths[k] for k in ('t1', 't2')}, 'synthetic'))
    return put_dataset(db, examples, components, DEFAULT_MAPPING,
                       dict(source='synthetic', seed=SEED, sampling=sampling, preview=False,
                            notes=['合成演示数据：3 个任务各 3 例，20 个组件；不用于研究结论'],
                            image_strategy='合成图片位于指定目录', components_enabled=True))


def export_files(db, directory):
    review, labels = store.exports(db)
    for name, value in [('task_review.json', review), ('second_component_labels.json', labels)]:
        store.write_text(Path(directory) / name, json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    return dict(examples=len(review['examples']), components=len(labels['components']), agreement=labels['agreement'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--synthetic', type=Path)
    mode.add_argument('--run-root', type=Path)
    mode.add_argument('--export', type=Path, dest='export_dir')
    parser.add_argument('--second-root', type=Path)
    parser.add_argument('--db', default=os.environ.get('TASK_DATA_DB'))
    parser.add_argument('--review-only', action='store_true')
    parser.add_argument('--preview-images', type=int, help='仅用于轻量真实截图；每个图池取前 N 张，明确标为预览')
    args = parser.parse_args()
    if args.run_root and not args.second_root:
        parser.error('--run-root 需要 --second-root')
    if args.preview_images is not None and args.preview_images < 1:
        parser.error('--preview-images 必须为正数')
    with closing(store.connect(args.db)) as db:
        if args.synthetic:
            result = prepare_synthetic(db, args.synthetic)
        elif args.run_root:
            result = prepare_real(db, args.run_root, args.second_root, not args.review_only, args.preview_images)
        else:
            result = export_files(db, args.export_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
