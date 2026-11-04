"""Compute full-split statistics, then render a deterministic small image pool on CPU."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
HERE = Path(__file__).resolve().parent
STAGING = HERE.parents[2]
ASSETS = STAGING / 'batch54_assets'
sys.path.insert(0, str(STAGING / 'batch48'))
import exp8s7_common as s7
import numpy as np
from scipy import ndimage as ndi
from PIL import Image, ImageDraw
from common import load_rgb

s5, s6, c = s7.s5, s7.s6, s7.c
VERSION = 2
POOL_SEED = 20261028
PNG_BUDGET = 1_000_000_000
SCOPE_BUDGET = PNG_BUDGET // 7
SCOPES = [('second', 'val'), ('second', 'test'), ('sysu', 'val'), ('sysu', 'test'),
          ('levir', 'val'), ('levir', 'test'), ('whu', 'test')]
COLORS = dict(detector='#3487ed', candidate='#f5ce35', novel='#30d878',
              cell1='#f2784b', cell2='#dc62df', cell3='#50d5db', cell5='#a9a9a9', ring='#ffffff')
SOURCE_COLORS = {'B': '#f5ce35', 'G0.05': '#ff8f55', 'H1': '#cd95ff'}
TITLES = ['原始数据', '各来源初始化候选', '去重', '切块与新增像素', '分流', '碎片过滤', '最终候选']


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def adapter(ds, split):
    if (ds, split) not in SCOPES:
        raise ValueError(f'无冻结缓存：{ds}/{split}（WHU 没有验证划分）')
    batch = 'batch48' if ds in ('second', 'sysu') else 'batch50'
    path = STAGING / batch / 'run' / ds / 'units' / f'{split}.npz'
    with np.load(path, allow_pickle=False) as z:
        frozen = {k: z[k] for k in ('images', 'image_id', 'id', 'bbox', 'mask_data', 'mask_offsets', 'novel', 'majority', 'scheme')}
    if batch == 'batch48':
        loader, pool, raw, kind = s5.load, c.pool_dir(ds, split), c.raw_dataset(ds, split), 'U5_x4'
    else:
        # The historical adapter installs dataset-specific geometry readers; each CLI run handles one dataset.
        sys.path.insert(0, str(STAGING / 'batch50'))
        import exp8s8_common as s8
        loader, pool, raw, kind = s8.load, s8.pool_dir(ds, split), s8.raw_dataset(ds, split), s8.pool_kind(ds)
    return dict(frozen=frozen, frozen_path=path, loader=loader, pool=pool, raw=raw, kind=kind)


def full(u, shape):
    out = np.zeros(shape, bool)
    s5.add_mask(out, u)
    return out


def union(units, shape):
    out = np.zeros(shape, bool)
    for u in units:
        s5.add_mask(out, u)
    return out


def novel(u, base):
    x0, y0, x1, y1 = u['bbox']
    return dict(u, mask=u['mask'] & ~base[y0:y1, x0:x1])


def rgb(color):
    return np.array(tuple(bytes.fromhex(color.lstrip('#'))), dtype=float)


def overlay(base, layers):
    out = np.asarray(base).copy()
    for mask, color, outline in layers:
        if outline:
            mask = mask & ~ndi.binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)
        out[mask] = (.18 * out[mask] + .82 * rgb(color)).astype(np.uint8)
    return Image.fromarray(out)


def layers(units, shape, color, outline=True):
    return [(full(u, shape), color, outline) for u in units]


def crop_box(box, shape):
    x0, y0, x1, y1 = map(int, box)
    side = min(max(shape), max(64, int(max(x1-x0, y1-y0) * 1.35)))
    x = max(0, min(shape[1]-side, (x0+x1-side)//2))
    y = max(0, min(shape[0]-side, (y0+y1-side)//2))
    return [x, y, min(shape[1], x+side), min(shape[0], y+side)]


def verify(d, kept, z, image):
    i = z['images'].tolist().index(image)
    indices = s7.image_rows(z, i)
    by_id = {int(z['id'][j]): int(j) for j in indices}
    if [u['id'] for u in kept] != z['id'][indices].tolist():
        raise ValueError(f'{image}: 第5步单元集合与冻结表不同')
    pixels = 0
    for u in kept:
        j = by_id[u['id']]
        ref = s7.unpack(z, j)
        if not np.array_equal(full(u, d['gt'].shape), full(ref, d['gt'].shape)):
            raise ValueError(f'{image}/{u["id"]}: 第5步新增像素与冻结表不同')
        if int(u['mask'].sum()) != int(z['novel'][j]):
            raise ValueError(f'{image}/{u["id"]}: 第5步新增面积与冻结表不同')
        pixels += int(u['mask'].sum())
    return dict(status='exact', units=len(kept), novel_pixels_sum=pixels,
                novel_pixels_union=int(union(kept, d['gt'].shape).sum()), reference_ids=z['id'][indices].tolist())


def input_stamp(a, ctx, name):
    pool = ctx['pool']
    dependencies = [ctx['frozen_path'], pool / 'images' / f'{Path(name).stem}.json', Path(__file__)]
    dependencies += [Path(module.__file__) for module in (s5, s6, c)]
    if a.dataset in ('levir', 'whu'):
        dependencies.append(STAGING / 'batch50' / 'exp8s8_common.py')
    dependencies += [pool / k / f'{Path(name).stem}.npz' for k in ('objects_none', 'objects_direct_pairs', ctx['kind'])]
    return {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in dependencies}


def load_scene(a, ctx, name):
    d = ctx['loader'](a.dataset, a.split, name)
    shape, base = d['gt'].shape, d['base']
    pool = ctx['pool']
    doc = read(pool / 'images' / f'{Path(name).stem}.json')
    raw, raw_shape = c.b40.load_units(pool / 'objects_none' / f'{Path(name).stem}.npz')
    dedup, dedup_shape = c.b40.load_units(pool / 'objects_direct_pairs' / f'{Path(name).stem}.npz')
    if tuple(raw_shape) != shape or tuple(dedup_shape) != shape:
        raise ValueError('来源/去重缓存图像尺寸不匹配')
    for key, units in [('objects_none', raw), ('objects_direct_pairs', dedup), (ctx['kind'], d['objects'])]:
        details = doc['units'][key]['details']
        if len(details) != len(units):
            raise ValueError(f'{name}/{key}: 元数据数量与掩码不符')
        for u, meta in zip(units, details):
            if list(u['bbox']) != meta['bbox'] or int(u['mask'].sum()) != meta['area']:
                raise ValueError(f'{name}/{key}: 元数据几何与掩码不符')
            u.update({k: meta[k] for k in ('sources', 'members', 'parent_object') if k in meta})
    scheme = str(ctx['frozen']['scheme'])
    kept = s6.units(d, scheme)
    try:
        verification = verify(d, kept, ctx['frozen'], name)
    except ValueError as exc:
        write(a.out/a.dataset/a.split/'failures'/(Path(name).stem+'.json'), dict(image=name,error=str(exc)))
        raise
    return d, doc, raw, dedup, kept, scheme, verification


def stage_stats(d, units):
    added = union([novel(u, d['base']) for u in units], d['gt'].shape)
    v = s5.coverage(d, added)
    return dict(count=len(units), hits2=v['hits2'], objects2=v['objects2'],
                coverage2=v['hits2']/v['objects2'] if v['objects2'] else None)


def analyze(a, ctx, name):
    started = time.monotonic()
    d, doc, raw, dedup, kept, scheme, verification = load_scene(a, ctx, name)
    shape, base, objects = d['gt'].shape, d['base'], d['objects']
    counts = [stage_stats(d, group) for group in ([], raw, dedup, objects, d['candidates'], kept, kept)]
    counts[0]['count'] = int(ndi.label(base, s5.EIGHT)[1])
    threshold = None if a.dataset == 'whu' else doc['config']['geometry']['median_area'] * (16 if a.dataset == 'levir' else 4)
    merged = [i for i,u in enumerate(dedup) if len(u['members']) > 1]
    large = [] if threshold is None else [i for i,u in enumerate(dedup) if u['area'] > threshold and sum(v.get('parent_object') == i for v in objects) > 1]
    small = [i for i,u in enumerate(dedup) if threshold is None or u['area'] <= threshold]
    regions = {}
    if merged:
        regions['merge'] = min(merged, key=lambda i:(-dedup[i]['area'], dedup[i]['members']))
    for tag, ids in [('large', large), ('small', small)]:
        if not ids:
            continue
        def score(i):
            areas = sorted((v['area'] for v in objects if v.get('parent_object') == i), reverse=True)
            return (areas[1] if tag == 'large' and len(areas) > 1 else 0, dedup[i]['area'], -i)
        regions[tag] = max(ids, key=score)
    for k in (1,2,3):
        rows = [r for r in d['rows'] if r['cell'] == k]
        if rows:
            regions[f'cell{k}'] = max(rows, key=lambda r:(r['area'], -r['id']))['id']
    routed = []
    for i,u in enumerate(objects):
        x0,y0,x1,y1 = u['bbox']
        routed.append((i, float(base[y0:y1,x0:x1][u['mask']].mean())))
    for tag, fill in [('fill',True),('refine',False)]:
        ids = [i for i,r in routed if (r < .1) == fill]
        if ids:
            regions[tag] = max(ids, key=lambda i:(objects[i]['area'], -i))
    kept_map = {u['id']:u for u in kept}
    filters = []
    for u in d['candidates']:
        radius = float(ndi.distance_transform_edt(np.pad(u['mask'],1)).max())
        retained = u['id'] in kept_map
        v = kept_map[u['id']] if retained else u
        x0,y0,x1,y1 = v['bbox']
        filters.append(dict(id=u['id'], area=int(u['mask'].sum()), radius=radius, retained=retained,
                            cell2=bool((v['mask'] & (d['cell'][y0:y1,x0:x1] == 2)).any())))
    for tag, group in [('thin',[u for u in filters if not u['retained'] and u['radius']<2]),
                       ('tiny',[u for u in filters if not u['retained'] and u['radius']>=2]),
                       ('keep',[u for u in filters if u['retained']])]:
        if group:
            regions[tag] = max(group,key=lambda u:(u['cell2'] if tag=='keep' else False,u['area'],-u['id']))['id']
    features = {k:k in regions for k in ('merge','large','small','cell1','cell2','cell3','fill','refine','thin','tiny')}
    features['kept2'] = any(u['cell2'] and u['retained'] for u in filters)
    boxes = {}
    for key, ident in regions.items():
        if key in ('merge','large','small'): box = dedup[ident]['bbox']
        elif key.startswith('cell'): box = next(r['bbox'] for r in d['rows'] if r['id']==ident)
        elif key in ('fill','refine'): box = objects[ident]['bbox']
        else: box = next(u['bbox'] for u in d['candidates'] if u['id']==ident)
        boxes[key] = crop_box(box, shape)
    record = dict(version=VERSION, dataset=a.dataset, split=a.split, image=name, shape=list(shape),
                  steps=counts, features=features, regions=regions, region_boxes=boxes,
                  sources={source:stage_stats(d,[u for u in raw if source in u['sources']]) for source in doc['config']['sources']},
                  verification=verification, inputs=input_stamp(a,ctx,name), seconds=time.monotonic()-started)
    if 'torch' in sys.modules:
        raise RuntimeError('CPU统计不应导入torch')
    return record


def build_image(a, ctx, name, record):
    started = time.monotonic()
    dest = a.out / a.dataset / a.split / Path(name).stem
    marker = dest / 'walkthrough.json'
    stamp = input_stamp(a, ctx, name)
    if record['inputs'] != stamp:
        raise ValueError(f'{name}: 输入改变，请重新stats和merge')
    if marker.exists() and not a.force:
        old = read(marker)
        if old['version'] != VERSION or old['inputs'] != stamp:
            raise ValueError(f'{marker}: 输入或版本变化；请用 --force 重新渲染')
        if all((dest/f).is_file() and (dest/f).stat().st_size==old['asset_sizes'][f] for f in old['assets']):
            print(f'resume render {a.dataset}/{a.split}/{name}', flush=True)
            return
    dest.mkdir(parents=True, exist_ok=True)
    marker.unlink(missing_ok=True)
    scope_bytes = sum(p.stat().st_size for p in dest.parent.glob('*/*.png'))
    d, doc, raw, dedup, kept, scheme, verification = load_scene(a, ctx, name)
    shape, base, pool = d['gt'].shape, d['base'], ctx['pool']
    t1, t2 = [load_rgb(ctx['raw'], name, v) for v in ('t1', 't2')]
    assets = []
    backgrounds = {}

    def save(filename, im, box=None):
        nonlocal scope_bytes
        if not isinstance(im, Image.Image):
            im = Image.fromarray(im)
        if box:
            im = im.crop(tuple(box))
        # Store only changed pixels over the shared RGB background, losslessly.
        # Crop backgrounds use the original full-resolution T1/T2 PNG via CSS.
        plain = {'t1.png','t2.png','gt.png','gt_cells.png','detector.png','probability.png','pick.png'}
        if filename not in plain:
            background = t1 if filename.endswith('_t1.png') else t2
            background_file = 't1.png' if filename.endswith('_t1.png') else 't2.png'
            box = box or [0,0,shape[1],shape[0]]
            x0,y0,x1,y1 = box
            pixels = np.asarray(im.convert('RGB'))
            changed = np.any(pixels != background[y0:y1,x0:x1], axis=2)
            rgba = np.zeros((*pixels.shape[:2],4),np.uint8)
            rgba[changed,:3] = pixels[changed]
            rgba[changed,3] = 255
            im = Image.fromarray(rgba)
            backgrounds[filename] = dict(file=background_file, box=box)
        temp = dest / f'{filename}.{os.getpid()}.tmp'
        im.save(temp, format='PNG', optimize=True, compress_level=9)
        previous = dest / filename
        new_size = scope_bytes + temp.stat().st_size - (previous.stat().st_size if previous.exists() else 0)
        if new_size > SCOPE_BUDGET:
            temp.unlink()
            raise ValueError(f'{a.dataset}/{a.split}: PNG将超过划分预算{SCOPE_BUDGET}字节，已停止写入')
        temp.replace(dest / filename)
        scope_bytes = new_size
        assets.append(filename)
        return filename

    def panel(filename, units=(), color='candidate', box=None, detector=False, background=None):
        ls = [(base, COLORS['detector'], False)] if detector else []
        ls += layers(units, shape, COLORS.get(color, color), color != 'novel')
        return save(filename, overlay(t2 if background is None else background, ls), box)

    def frame(label, filename):
        return dict(label=label, file=filename)

    def stats(units):
        added = union([novel(u, base) for u in units], shape)
        v = s5.coverage(d, added)
        return dict(count=len(units), hits2=v['hits2'], objects2=v['objects2'],
                    coverage2=v['hits2']/v['objects2'] if v['objects2'] else None)

    steps = [dict(number=i, title=t, frames=[], examples=[], parameters={}) for i, t in enumerate(TITLES)]
    detector_count = int(ndi.label(base, np.ones((3, 3)))[1])
    steps[0].update(stats=dict(stats([]), count=detector_count), count_label='检测器连通域')
    steps[0]['parameters'] = {'检测器阈值': 'probability > 0.5', 'GT 最小评估目标 / 漏检块': '20 像素',
                              '②覆盖': '按GT目标去重，D与候选新增并集覆盖目标≥50%'}
    save('t1.png', t1); save('t2.png', t2)
    save('gt.png', np.repeat((d['gt']*255).astype(np.uint8)[..., None], 3, axis=2))
    cell_layers = [(d['cell'] == k, COLORS[f'cell{k}'], False) for k in (1, 2, 3, 5)]
    save('gt_cells.png', overlay(np.repeat((d['gt']*100).astype(np.uint8)[..., None], 3, axis=2), cell_layers))
    save('detector.png', overlay(np.zeros_like(t2), [(base, COLORS['detector'], False)]))
    p = np.clip(d['prob'], 0, 1)
    heat = np.stack([p*255, (1-np.abs(p-.5)*2)*180, (1-p)*255], axis=-1).astype(np.uint8)
    save('probability.png', heat)
    steps[0]['frames'] = [frame(l, f) for l, f in [('T1','t1.png'),('T2','t2.png'),('GT','gt.png'),('检测器二值','detector.png')]]

    sources = doc['config']['sources']
    source_info = []
    explanations = {'B': '两期语义类别不同的区域。' if a.dataset in ('second','sysu') else '两期建筑类别有无不同（building XOR）。',
                    'G0.05': '寻找检测器低概率响应。冻结代码先对 p > 0.05 取连通域，再保留检测器已报比例≤0.5 的区域；新增部分为 0.05 < p ≤ 0.5。',
                    'H1': 'AnyChange：SAM 物体两期特征差异产生的候选。'}
    all_layers = []
    for j, source in enumerate(sources):
        group = [u for u in raw if source in u['sources']]
        filename = panel(f'source_{j}.png', group)
        source_info.append(dict(name=source, description=explanations[source], file=filename, **stats(group)))
        all_layers += layers(group, shape, SOURCE_COLORS[source])
    save('sources_all.png', overlay(t2, all_layers))
    steps[1].update(stats=stats(raw), sources=source_info, parameters={'来源': sources, '最小候选': '10 像素', '已报比例': '≤0.5'})
    steps[2].update(stats=stats(dedup), parameters={'合并规则': '跨来源 IoU≥0.5 或交集/较小面积≥0.8；按IoU降序直接对合并，每个原候选最多参与一对'})
    merged = sorted([u for u in dedup if len(u['members']) > 1], key=lambda u: (-int(u['mask'].sum()), u['members']))
    if merged:
        u = dedup[record['regions']['merge']]; box = record['region_boxes']['merge']
        original = [raw[i] for i in u['members']]
        ls = [(full(v, shape), SOURCE_COLORS[v['sources'][0]], True) for v in original]
        steps[2]['examples'].append(dict(title='重叠候选直接合并', bbox=box, note=f"原候选 {u['members']} → 1 个对象", frames=[
            frame('合并前：来源轮廓', save('dedup_before.png', overlay(t2, ls), box)),
            frame('合并后：一个对象', panel('dedup_after.png', [u], box=box))]))
    steps[2]['frames'] = [frame('全图合并前', 'sources_all.png'), frame('全图合并后', panel('dedup.png', dedup))]

    threshold = None if a.dataset == 'whu' else doc['config']['geometry']['median_area'] * (16 if a.dataset == 'levir' else 4)
    objects = d['objects']
    steps[3].update(stats=stats(objects), parameters={'面积阈值': threshold, '单元': ctx['kind'],
        '切块规则': '超过阈值与缓存SAM自动掩码分区求交，每块≥10像素；普通对象完整保留。' if threshold else 'WHU冻结版本使用去重对象，未启用U5切块。',
        '新增像素': '候选掩码减去检测器已报部分'})
    large = [] if threshold is None else [i for i,u in enumerate(dedup) if int(u['mask'].sum()) > threshold and sum(v.get('parent_object') == i for v in objects) > 1]
    small = [i for i,u in enumerate(dedup) if threshold is None or int(u['mask'].sum()) <= threshold]
    for tag, ids, title in [('large', large, '超过阈值的大对象'), ('small', small, '普通对象对照')]:
        if not ids:
            continue
        i = record['regions'][tag]
        u = dedup[i]; box = crop_box(u['bbox'], shape)
        children = [v for v in objects if v.get('parent_object') == i] if threshold else [objects[i]]
        steps[3]['examples'].append(dict(title=title, bbox=box, note=f"对象面积 {int(u['mask'].sum())}，输出 {len(children)} 块", frames=[
            frame('原对象与检测器', panel(f'{tag}_parent.png', [u], box=box, detector=True)),
            frame('先叠检测器', panel(f'{tag}_detector.png', box=box, detector=True)),
            frame('SAM切块与检测器', panel(f'{tag}_parts.png', children, box=box, detector=True)),
            frame('只保留新增像素', panel(f'{tag}_novel.png', [novel(v, base) for v in children], color='novel', box=box))]))
    steps[3]['frames'] = [frame('全图候选块', panel('units.png', objects, detector=True)), frame('全图新增像素', panel('novel.png', [novel(u,base) for u in objects], 'novel'))]

    steps[4].update(stats=stats(d['candidates']), parameters={'实际分流': '冻结库存中的单元自身 r=|候选∩D|/|候选|；r<0.1补漏，否则精修',
        '①②③': 'GT评估分类，仅用于解释；实际流程不读取GT。①②为GT目标检测覆盖<10%；③为覆盖≥10%的目标剩余漏检。',
        '① / ②': '漏检块到D的棋盘距离≤1 / >1；图中白环是D的1像素膨胀环',
        '⑤': 'GT目标或漏检连通块<20像素'}, fill_count=len(d['candidates']), refine_count=len(objects)-len(d['candidates']))
    ring = ndi.binary_dilation(base, structure=np.ones((3,3))) & ~base
    evaluation = overlay(t2, cell_layers + [(base,COLORS['detector'],False), (ring,COLORS['ring'],False)] + layers(objects,shape,COLORS['candidate']))
    steps[4]['frames'] = [frame('GT分类、检测器与候选', save('routing.png', evaluation))]
    routed = []
    for i,u in enumerate(objects):
        x0,y0,x1,y1=u['bbox']; r=float(base[y0:y1,x0:x1][u['mask']].mean())
        routed.append(dict(u, id=i, r=r))
    for k, label in [(1,'① 无信号、贴着检测结果'), (2,'② 无信号、孤立'), (3,'③ 有信号')]:
        rows = [r for r in d['rows'] if r['cell'] == k]
        if rows:
            r = next(r for r in rows if r['id'] == record['regions'][f'cell{k}'])
            box = crop_box(r['bbox'], shape)
            target = d['labels'] == r['id']
            matches = [(int((full(u,shape)&target).sum()), u) for u in routed]
            best = max(matches, key=lambda item:(item[0],-item[1]['id'])) if matches else (0,None)
            selected = [best[1]] if best[0] else []
            route_note = (f"；单元 {best[1]['id']}，r={best[1]['r']:.4f}，" + ('补漏组' if best[1]['r']<.1 else '精修组')) if selected else '；此漏检块没有相交候选，r无定义'
            example = overlay(t2, [(target,COLORS[f'cell{k}'],False),(base,COLORS['detector'],False),(ring,COLORS['ring'],False)] + layers(selected,shape,COLORS['candidate']))
            steps[4]['examples'].append(dict(title=label, bbox=box,
                note=f"GT目标检测覆盖 {r['detector_coverage']:.1%}；距D {r['distance']} 像素；漏检面积 {r['area']}"+route_note,
                frames=[frame(label,save(f'cell{k}.png',example,box))]))
    for fill_group, tag in [(True,'fill'), (False,'refine')]:
        group=[u for u in routed if (u['r'] < .1) == fill_group]
        if group:
            u=next(u for u in group if u['id']==record['regions'][tag]);box=crop_box(u['bbox'],shape)
            steps[4]['examples'].append(dict(title='补漏组' if fill_group else '精修组', bbox=box, r=u['r'],
                note=f"单元 {u['id']}，r={u['r']:.4f}，候选面积 {int(u['mask'].sum())}", frames=[frame('GT、检测器与该候选',save(f'route_{tag}.png',overlay(t2,cell_layers+[(base,COLORS['detector'],False),(ring,COLORS['ring'],False)]+layers([u],shape,COLORS['candidate'])),box))]))

    steps[5].update(stats=stats(kept), parameters={'冻结过滤':scheme, '规则':'新增掩码3×3开运算后≥100像素' if scheme=='F2' else '新增掩码最大内切半径≥2且新增面积≥200像素', '冻结核对': verification})
    kept_map={u['id']:u for u in kept}
    filters=[]
    for u in d['candidates']:
        radius=float(ndi.distance_transform_edt(np.pad(u['mask'],1)).max())
        area=int(u['mask'].sum()); retained=u['id'] in kept_map
        filters.append(dict(u, radius=radius, area=area, retained=retained,
            cell2=bool((full(kept_map[u['id']] if retained else u,shape)&(d['cell']==2)).any())))
    groups=[('thin','被去掉的细条',[u for u in filters if not u['retained'] and u['radius']<2]),
            ('tiny','被去掉的小块',[u for u in filters if not u['retained'] and u['radius']>=2]),
            ('keep','保留下来的块',[u for u in filters if u['retained']])]
    for tag,title,group in groups:
        if not group: continue
        u=next(u for u in group if u['id']==record['regions'][tag])
        box=crop_box(u['bbox'],shape); after=kept_map.get(u['id'])
        after_radius=float(ndi.distance_transform_edt(np.pad(after['mask'],1)).max()) if after else 0
        opened_area=int(ndi.binary_opening(u['mask'],structure=s5.EIGHT).sum()) if scheme=='F2' else None
        detail=f"；开运算后面积 {opened_area}" if opened_area is not None else ''
        steps[5]['examples'].append(dict(title=title,bbox=box, note=f"单元 {u['id']}；面积 {u['area']} → {int(after['mask'].sum()) if after else 0}；内切半径 {u['radius']:.2f} → {after_radius:.2f}；含②：{'是' if u['cell2'] else '否'}"+detail,frames=[
            frame('过滤前新增像素',panel(f'filter_{tag}_before.png',[u],'novel',box)),
            frame('过滤后新增像素',panel(f'filter_{tag}_after.png',[after] if after else [],'novel',box))]))
    steps[5]['frames']=[frame('过滤前',panel('filter_before.png',d['candidates'],'novel')),frame('过滤后',panel('filter_after.png',kept,'novel'))]
    final_union=union(kept,shape)
    targets=[]
    for target in sorted(s6.targets2(d)):
        mask=d['gl']==target; added=int((final_union&mask).sum()); total=int(mask.sum()); detected=int((base&mask).sum())
        status='covered' if 2*(added+detected)>=total else 'partial' if added else 'absent'
        yy,xx=np.nonzero(mask)
        targets.append(dict(id=target,area=total,added=added,status=status,bbox=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]))
    final_layers=layers(kept,shape,COLORS['candidate'])
    colored=overlay(t2,[(base,COLORS['detector'],False)]+[(full(u,shape),COLORS['cell2'] if (full(u,shape)&(d['cell']==2)).any() else COLORS['candidate'],True) for u in kept])
    normal=overlay(t2,[(base,COLORS['detector'],False)]+final_layers)
    for im in (normal,colored):
        draw=ImageDraw.Draw(im)
        for r in targets:
            if r['status'] != 'absent': continue
            x0,y0,x1,y1=r['bbox']
            for x in range(x0,x1,8):
                draw.line((x,y0,min(x+4,x1),y0),fill=COLORS['cell2'],width=2);draw.line((x,y1,min(x+4,x1),y1),fill=COLORS['cell2'],width=2)
            for y in range(y0,y1,8):
                draw.line((x0,y,x0,min(y+4,y1)),fill=COLORS['cell2'],width=2);draw.line((x1,y,x1,min(y+4,y1)),fill=COLORS['cell2'],width=2)
    save('final.png',normal);save('final_cells.png',colored)
    pick=np.zeros(shape,np.uint32); candidates=[]
    for ordinal,u in enumerate(kept,1):
        mask=full(u,shape);pick[mask]=ordinal
        box=crop_box(u['bbox'],shape)
        prefix=f'candidate_{u["id"]}'
        candidates.append(dict(id=u['id'],pick=ordinal,bbox=list(map(int,u['bbox'])),area=int(u['mask'].sum()),cell2=bool((mask&(d['cell']==2)).any()),
            frames=[frame('T1',save(prefix+'_t1.png',t1,box)),frame('T2',save(prefix+'_t2.png',t2,box)),frame('新增像素',panel(prefix+'_novel.png',[u],'novel',box))]))
    save('pick.png',np.stack([pick&255,(pick>>8)&255,(pick>>16)&255],axis=-1).astype(np.uint8))
    steps[6].update(stats=stats(kept),parameters={'选择候选':'点击图中轮廓内的像素，或从候选列表选择；重叠处显示最后一个候选',
        '②目标覆盖':'D与候选新增并集覆盖≥50%；紫色虚线框表示新增覆盖为0的②目标'},frames=[frame('最终补漏候选','final.png')],candidates=candidates,targets2=targets)
    features={'merge':bool(merged),'large':bool(large),'small':bool(small),'cell1':bool((d['cell']==1).any()),'cell2':bool(targets),'cell3':bool((d['cell']==3).any()),
              'fill':bool(d['candidates']),'refine':len(objects)>len(d['candidates']),'thin':bool(groups[0][2]),'tiny':bool(groups[1][2]),'kept2':any(u['cell2'] for u in filters if u['retained'])}
    if [step['stats'] for step in steps] != record['steps'] or features != record['features']:
        raise ValueError(f'{name}: 渲染数量/覆盖/特征与stats不一致')
    for step in steps:
        if step['number'] in (2,3,4,5) and not step['examples']:
            step['notice']='本图没有符合规则的典型区域；可切换典型图。'
    result=dict(version=VERSION,dataset=a.dataset,split=a.split,image=name,shape=list(shape),scheme=scheme,steps=steps,
        verification=verification,features=features,colors=COLORS,source_colors=SOURCE_COLORS,assets=assets,
        asset_sizes={f:(dest/f).stat().st_size for f in assets}, backgrounds=backgrounds, inputs=stamp,
        seconds=time.monotonic()-started,provenance=dict(pool=str(pool),frozen=str(ctx['frozen_path']),geometry='batch46 exp8s5_common.geometry',filter='batch47 exp8s6_filter.units'))
    if 'torch' in sys.modules: raise RuntimeError('CPU渲染不应导入torch')
    write(marker,result)
    (dest / 'failure.json').unlink(missing_ok=True)
    print(f"built {a.dataset}/{a.split}/{name}: {len(kept)} units, exact, {result['seconds']:.2f}s",flush=True)


def part_path(a, name, shard):
    return a.out/a.dataset/a.split/'stats_parts'/f'{shard:03d}-of-{a.shards:03d}'/(Path(name).stem+'.json')


def run_stats(a, ctx, selected):
    for i, name in enumerate(ctx['frozen']['images'].tolist()):
        if name not in selected or i % a.shards != a.shard:
            continue
        path = part_path(a,name,a.shard)
        if path.exists() and not a.force:
            record = read(path)
            if record['version'] != VERSION or record['inputs'] != input_stamp(a,ctx,name):
                raise ValueError(f'{path}: 统计输入变化，请用 --force 重算')
            print(f'resume stats {a.dataset}/{a.split}/{name}',flush=True)
            continue
        record = analyze(a,ctx,name)
        write(path,record)
        print(f"stats {a.dataset}/{a.split}/{name}: exact, {record['seconds']:.3f}s",flush=True)


def choose_pool(records):
    remaining = sorted(records, key=lambda d:d['image'])
    chosen, covered = [], set()
    while remaining and len(chosen)<5:
        best = min(remaining,key=lambda d:(-sum(v and k not in covered for k,v in d['features'].items()),
            -int(d['features']['kept2']),-sum(d['features'].values()),-d['steps'][6]['hits2'],d['image']))
        chosen.append(best['image'])
        covered.update(k for k,v in best['features'].items() if v)
        remaining.remove(best)
    # Draw from the complement so a complete scope contains ten different images.
    random_names = random.Random(POOL_SEED).sample([d['image'] for d in remaining], min(5,len(remaining)))
    return chosen, random_names


def merge_stats(a, ctx):
    names = ctx['frozen']['images'].tolist()
    records, missing = [], []
    for i,name in enumerate(names):
        path = part_path(a,name,i % a.shards)
        if not path.exists():
            missing.append(name)
            continue
        d = read(path)
        if (d['version']!=VERSION or d['image']!=name or d['dataset']!=a.dataset or d['split']!=a.split
                or d['verification']['status']!='exact' or d['inputs']!=input_stamp(a,ctx,name)):
            raise ValueError(f'{path}: 统计不匹配或输入改变，请重算该分片')
        records.append(d)
    if missing and not a.allow_partial:
        raise ValueError(f'统计缺少 {len(missing)}/{len(names)} 张；验收可显式 --allow-partial')
    if not records:
        raise ValueError('没有统计结果')
    typical, random_names = choose_pool(records)
    pool = typical + random_names
    aggregates = []
    for i in range(7):
        count = sum(d['steps'][i]['count'] for d in records)
        hits = sum(d['steps'][i]['hits2'] for d in records)
        targets = sum(d['steps'][i]['objects2'] for d in records)
        aggregates.append(dict(count=count, mean_count=count/len(records), hits2=hits, objects2=targets,
                               coverage2=hits/targets if targets else None))
    examples = {k:next((d['image'] for d in sorted(records,key=lambda d:d['image'])
                       if d['image'] in pool and d['features'][k]),None) for k in records[0]['features']}
    result = dict(version=VERSION,dataset=a.dataset,split=a.split,expected=len(names),analyzed=len(records),
        complete=not missing,missing_count=len(missing),seed=POOL_SEED,typical=typical,random=random_names,pool=pool,
        examples=examples,averages=[s['mean_count'] for s in aggregates] if not missing else None,
        aggregates=aggregates,seconds=sum(d['seconds'] for d in records),records=records,
        verification=dict(status='exact',images=len(records),units=sum(d['verification']['units'] for d in records)))
    write(a.out/a.dataset/a.split/'stats.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('records','examples','aggregates')},ensure_ascii=False),flush=True)


def render_pool(a, ctx, selected):
    folder = a.out/a.dataset/a.split
    statistics = read(folder/'stats.json')
    if statistics['version']!=VERSION:
        raise ValueError('统计版本改变，请重新stats和merge')
    if not statistics['complete'] and not a.allow_partial:
        raise ValueError('统计未覆盖完整划分；验收可显式 --allow-partial')
    pool = statistics['pool']
    if a.images and not set(a.images)<=set(pool):
        raise ValueError('--images 中有图名不在统计选出的图池内')
    if not a.images:
        selected = set(pool[:a.limit or None])
    # Remove only this renderer's declared files for images excluded by the new pool.
    for marker in folder.glob('*/walkthrough.json'):
        old = read(marker)
        if old.get('version') not in (1,VERSION) or old.get('dataset')!=a.dataset or old.get('split')!=a.split:
            continue
        if old['image'] not in pool:
            for filename in old['assets']:
                if Path(filename).name != filename:
                    raise ValueError(f'{marker}: 非法产物文件名')
                (marker.parent/filename).unlink(missing_ok=True)
            marker.unlink()
    records = {d['image']:d for d in statistics['records']}
    for name in pool:
        if name in selected:
            build_image(a,ctx,name,records[name])
    ready = []
    size = 0
    for name in pool:
        path = folder/Path(name).stem/'walkthrough.json'
        if not path.is_file():
            continue
        doc = read(path)
        if doc['version']!=VERSION or doc['inputs']!=records[name]['inputs']:
            raise ValueError(f'{name}: 图池里有过期渲染，请重建')
        ready.append(name)
        size += sum(doc['asset_sizes'].values())
    if size > SCOPE_BUDGET:
        raise ValueError(f'本划分PNG为{size}字节，超过七等分的1 GB预算{SCOPE_BUDGET}；未发布索引')
    result = dict(version=VERSION,dataset=a.dataset,split=a.split,images=ready,generated=len(ready),
        pool=pool,pool_size=len(pool),pool_complete=len(ready)==len(pool),png_bytes=size,
        typical=[n for n in statistics['typical'] if n in ready],
        random=[n for n in statistics['random'] if n in ready],stats_file='stats.json',
        # Statistics stay in stats.json; do not put pool averages in this index.
        examples={k:n if n in ready else None for k,n in statistics['examples'].items()})
    write(folder/'index.json',result)
    print(f'render pool {a.dataset}/{a.split}: {len(ready)}/{len(pool)} images, {size/1e6:.2f} MB PNG',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('stats','merge','render'))
    p.add_argument('--dataset',choices=('second','sysu','levir','whu'),required=True)
    p.add_argument('--split',choices=('val','test'),required=True)
    p.add_argument('--out',type=Path,default=ASSETS)
    p.add_argument('--images',nargs='+',help='Exact names for small checks; render only accepts pool members')
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--shards',type=int,default=1)
    p.add_argument('--allow-partial',action='store_true')
    p.add_argument('--force',action='store_true')
    a=p.parse_args();a.out=a.out.resolve()
    if not a.out.is_relative_to(ASSETS.resolve()):p.error('PNG/JSON必须写入 batch54_assets')
    if not 0<=a.shard<a.shards or a.limit<0:p.error('需要0≤shard<shards，limit≥0')
    if a.stage=='render' and a.shards!=1:p.error('render按完整小图池串行运行，不接受分片')
    if a.stage!='merge' and not os.environ.get('SLURM_JOB_ID') and not (0<a.limit<=2 or a.images and len(a.images)<=2):
        p.error('登录节点仅允许每次1–2张真实图片；正式stats/render请交Slurm')
    ctx=adapter(a.dataset,a.split)
    names=ctx['frozen']['images'].tolist()
    if a.images:
        if len(set(a.images))!=len(a.images) or not set(a.images)<=set(names):p.error('图名重复或不属于冻结划分')
        names=[n for n in names if n in a.images]
    if a.limit:names=names[:a.limit]
    selected=set(names)
    if a.stage=='stats':run_stats(a,ctx,selected)
    elif a.stage=='merge':merge_stats(a,ctx)
    else:render_pool(a,ctx,selected)


if __name__=='__main__':
    main()
