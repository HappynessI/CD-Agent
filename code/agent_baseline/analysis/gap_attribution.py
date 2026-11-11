"""Offline pixel ancestry for A minus fixed-route F; never imports online tools.

Run from code/: python -m agent_baseline.analysis.gap_attribution --run-root R
See the task workspace NOTES.md and results/gap_attribution.md for conventions.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
from collections import Counter, defaultdict
import io
import json
from pathlib import Path
import re
import time
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

R_DEFAULT = Path('/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run')
HANDLE = re.compile(r'(?:mask|inst|region|props)#\d+')
RETRIES = Counter()


def retry_read(path, reader):
    """Retry the complete read/decode, including lazy PNG and NPZ I/O."""
    for attempt in range(4):
        try:
            return reader(Path(path))
        except OSError:
            if attempt == 3:
                raise
            RETRIES['retries'] += 1
            time.sleep(.1 * (attempt + 1))


def read_json(path):
    return retry_read(path, lambda p: json.loads(p.read_text()))


def read_jsonl(path):
    return retry_read(path, lambda p: [json.loads(x) for x in p.read_text().splitlines() if x.strip()])


def read_mask(path):
    def read(p):
        with Image.open(p) as im:
            a = np.array(im) > 0
        if a.ndim != 2:
            raise ValueError(f'Expected 2D mask: {p.name}')
        return a
    return retry_read(path, read)


def read_npz(path):
    def read(p):
        with np.load(io.BytesIO(p.read_bytes()), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    return retry_read(path, read)


def handles(value):
    if isinstance(value, str):
        return HANDLE.findall(value)
    if isinstance(value, dict):
        return [h for v in value.values() for h in handles(v)]
    if isinstance(value, list):
        return [h for v in value for h in handles(v)]
    return []


def pixel_counts(a, g):
    return [int((a & g).sum()), int((a & ~g).sum()), int((~a & g).sum())]


def f1(c):
    c = np.asarray(c, dtype=float)
    den = 2*c[..., 0]+c[..., 1]+c[..., 2]
    return np.divide(200*c[..., 0], den, out=np.zeros_like(den), where=den != 0)


def parameter_bin(tool, args, shape):
    a = args or {}
    if tool == 'segment_text':
        query = ' '.join(a.get('text', a.get('query', '')).lower().split())
        category = 'building' if query == 'building' else ('同义词' if query in
            ('buildings', 'house', 'houses', 'roof', 'roofs', 'structure', 'structures') else '其他')
        return dict(query_class=category, query=query, time=a.get('time'),
                    score_thr=a.get('score_thr', .5), tile=a.get('tile', 512))
    if tool in ('segment_prompt', 'refine_region'):
        out = dict(time=a.get('time'), mode='box' if a.get('box') is not None else
                   ('points' if a.get('points') is not None else '默认组件框'))
        if a.get('box') is not None:
            x0, y0, x1, y1 = a['box']
            share = (np.ceil(x1)-np.floor(x0))*(np.ceil(y1)-np.floor(y0))/np.prod(shape)
            out['box_size'] = '<1%' if share < .01 else ('1–10%' if share <= .1 else '>10%')
        return out
    if tool == 'add_region':
        return dict(method='已有区域合入', selected='all' if a.get('ids') is None else 'ids')
    if tool == 'morph':
        return dict(op=a.get('op'), radius=a.get('radius'))
    if tool == 'python':
        return dict(method='数组运算')
    return {k: a[k] for k in ('time', 'text', 'query', 'score_thr') if k in a}


class Trace:
    """Evaluate only queried final-gap pixel positions through immutable handles.

    Origins are call ordinals, -1 means unknown. Inst archives are sampled directly
    from packbits; no model or dense per-instance image stack is needed.
    """
    def __init__(self, directory, indices, shape):
        self.directory, self.indices, self.shape = Path(directory), np.asarray(indices), tuple(shape)
        self.calls, self.producer, self.cache, self.assets = [], {}, {}, {}
        self.nodes = {}
        self.ambiguity = np.zeros(len(indices), bool)
        for row in read_jsonl(self.directory / 'steps.jsonl'):
            for within, c in enumerate(row.get('calls', []), 1):
                c = dict(c, step=row['step'], call_in_step=within, ordinal=len(self.calls))
                self.calls.append(c)
                for h in c.get('handles_created', c.get('output_handles', [])):
                    self.producer[h] = c['ordinal']

    def asset(self, h):
        if h not in self.assets:
            if h.startswith(('inst#', 'props#')):
                self.assets[h] = read_npz(self.directory / 'insts' / ('inst_'+h.split('#')[1]+'.npz'))
            else:
                self.assets[h] = read_mask(self.directory / 'masks' / (h.replace('#', '_')+'.png'))
        return self.assets[h]

    def presence(self, h, ids=None):
        asset = self.asset(h)
        if isinstance(asset, dict):
            if tuple(asset['shape']) != self.shape:
                raise ValueError('Instance shape mismatch')
            bits = asset['masks']
            if ids is not None:
                bits = bits[np.isin(asset['ids'], ids)]
            # Chunk instances to keep temporary arrays small.
            result = np.zeros(len(self.indices), bool)
            for start in range(0, len(bits), 16):
                sampled = bits[start:start+16, self.indices // 8]
                result |= np.any((sampled >> (7-self.indices % 8)) & 1, axis=0)
            return result
        if asset.shape != self.shape:
            raise ValueError('Mask shape mismatch')
        if ids is None:
            return asset.ravel()[self.indices]
        labels = self.component_labels(h)
        return np.isin(labels.ravel()[self.indices], ids)

    def component_labels(self, h):
        # remove_region preserves IDs; its PNG alone does not encode those IDs.
        asset = self.asset(h)
        ordinal = self.producer.get(h)
        c = self.calls[ordinal] if ordinal is not None else None
        if c and c['tool'] == 'remove_region':
            return np.where(asset, self.component_labels(c['args']['mask']), 0)
        return ndi.label(asset, np.ones((3, 3), int))[0]

    def parents(self, h, c):
        a, tool = c.get('args') or {}, c['tool']
        if tool == 'combine':
            return [a['a']] if a['op'] == 'subtract' else [a['a'], a['b']]
        if tool == 'match_instances':
            time_out = str(self.asset(h)['time'])
            return [a['inst_'+time_out]]
        if tool in ('segment_text', 'segment_prompt', 'segment_everything', 'segment_semantic'):
            return []
        return list(dict.fromkeys(handles(a)))

    def origins(self, h):
        if h in self.cache:
            return self.cache[h]
        present = self.presence(h)
        origin = np.full(len(self.indices), -1, dtype=np.int32)
        ordinal = self.producer.get(h)
        if ordinal is None:
            self.nodes[h] = dict(producer=None, parents=[])
            self.cache[h] = (present, origin)
            return present, origin
        c = self.calls[ordinal]
        parents = self.parents(h, c)
        self.nodes[h] = dict(producer=ordinal, parents=parents)
        covered = np.zeros(len(self.indices), bool)
        for parent in parents:
            pp, po = self.origins(parent)
            eligible = present & pp
            if c['tool'] == 'add_region' and parent == c['args']['other']:
                eligible &= self.presence(parent, c['args'].get('ids'))
            self.ambiguity |= eligible & covered & (origin != po) & (origin >= 0) & (po >= 0)
            # Unknown ancestral evidence takes precedence over a later known origin.
            replace = eligible & (~covered | ((origin >= 0) & ((po < 0) | (po < origin))))
            origin[replace] = po[replace]
            covered |= eligible
        novel = present & ~covered
        if c['ok'] and c['tool'] in ('segment_text', 'segment_prompt', 'segment_everything',
                'segment_semantic', 'add_region', 'refine_region', 'python', 'morph'):
            origin[novel] = ordinal
        self.cache[h] = (present, origin)
        return present, origin

    def source(self, ordinal):
        if ordinal == -1:
            return dict(ordinal=-1, tool='未知', parameter_bin={})
        c = self.calls[ordinal]
        a = c.get('args') or {}
        params = {k: v for k, v in a.items() if k not in ('code', 'summary')}
        if c['tool'] == 'python':
            # Exact executable text remains in the read-only source trace.
            params['code_reference'] = 'steps.jsonl 中本调用 args.code'
        out = dict(ordinal=ordinal, step=c['step'], call_in_step=c['call_in_step'],
                   tool=c['tool'], args=params, handles_created=c.get('handles_created', []),
                   parameter_bin=parameter_bin(c['tool'], a, self.shape))
        if a.get('box') is not None:
            x0, y0, x1, y1 = a['box']
            out['box_image_fraction'] = float((np.ceil(x1)-np.floor(x0))*(np.ceil(y1)-np.floor(y0))/np.prod(self.shape))
        elif c['tool'] == 'refine_region' and a.get('points') is None:
            old_labels = self.component_labels(a['mask'])
            yy, xx = np.nonzero(old_labels == a['id'])
            box = [int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1]
            out['effective_box'] = box
            out['box_image_fraction'] = (box[2]-box[0])*(box[3]-box[1])/float(np.prod(self.shape))
            out['parameter_bin']['box_size'] = parameter_bin('segment_prompt', dict(box=box), self.shape)['box_size']
        return out


def analyze_session(directory, fixed_path, root):
    directory, root = Path(directory), Path(root)
    meta, final = read_json(directory/'input.json'), read_json(directory/'final.json')
    g = read_mask(meta['gt_mask_path'])
    a, fixed = read_mask(directory/final['mask_path']), read_mask(fixed_path)
    if a.shape != g.shape or fixed.shape != g.shape:
        raise ValueError('A/F/G shape mismatch')
    counts = pixel_counts(a, g)
    if counts != [final['eval'][k] for k in ('tp', 'fp', 'fn')]:
        raise ValueError(f'Final counts differ: {directory.name}')
    gap = a & ~fixed
    labels, n = ndi.label(gap, np.ones((3, 3), int))
    indices = np.flatnonzero(gap)
    trace = Trace(directory, indices, g.shape)
    present, origins = trace.origins(final['final_handle'])
    if not present.all():
        raise ValueError('Final handle and mask_path differ')
    truth, comp_ids = g.ravel()[indices], labels.ravel()[indices]
    components = []
    for cid, sl in enumerate(ndi.find_objects(labels), 1):
        members = comp_ids == cid
        sources = []
        for ordinal in np.unique(origins[members]):
            assigned = members & (origins == ordinal)
            tp = int(truth[assigned].sum())
            sources.append(dict(source=int(ordinal), tp_pixels=tp, fp_pixels=int(assigned.sum())-tp))
        components.append(dict(id=cid, bbox=[sl[1].start, sl[0].start, sl[1].stop, sl[0].stop],
            tp_pixels=int(truth[members].sum()), fp_pixels=int(members.sum()-truth[members].sum()), sources=sources))
    source_rows = {str(i): trace.source(int(i)) for i in np.unique(origins)}
    removed = pixel_counts(a & fixed, g)
    unknown = origins == -1
    return dict(session=str(directory.relative_to(root)), image=directory.name,
        seed=directory.parents[2].name.rsplit('_s', 1)[-1], agent_counts=counts,
        without_gap_counts=removed, fixed_counts=pixel_counts(fixed, g),
        gap_tp_pixels=int(truth.sum()), gap_fp_pixels=int(len(truth)-truth.sum()),
        component_count=n, unknown_pixels=int(unknown.sum()), unknown_tp_pixels=int(truth[unknown].sum()),
        ambiguous_ancestry_pixels=int(trace.ambiguity.sum()), sources=source_rows,
        ancestry=trace.nodes,
        ancestry_calls={str(i): trace.source(i) for i in sorted({n['producer'] for n in trace.nodes.values() if n['producer'] is not None})},
        components=components)


def bootstrap_delta(agent, removed, samples=5000, seed=1016):
    """Paired image resampling, identical image indices across the three seeds.

    Score each seed by its pooled confusion, then average seed F1 values.
    """
    a, b = np.asarray(agent, dtype=np.int64), np.asarray(removed, dtype=np.int64)
    rng = np.random.default_rng(seed)
    values = np.empty(samples)
    for start in range(0, samples, 100):
        idx = rng.integers(0, a.shape[1], (min(100, samples-start), a.shape[1]))
        values[start:start+len(idx)] = (f1(a[:, idx].sum(axis=2)) - f1(b[:, idx].sum(axis=2))).mean(axis=0)
    return dict(actual_f1=float(f1(a.sum(axis=1)).mean()), without_gap_f1=float(f1(b.sum(axis=1)).mean()),
        delta_f1_pp=float((f1(a.sum(axis=1))-f1(b.sum(axis=1))).mean()),
        ci95_pp=np.percentile(values, [2.5, 97.5]).tolist(), samples=samples, seed=seed)


def summarize(rows):
    bins = {}
    for row in rows:
        for component in row['components']:
            touched = set()
            for s in component['sources']:
                source = row['sources'][str(s['source'])]
                key = json.dumps([source['tool'], source['parameter_bin']], ensure_ascii=False, sort_keys=True)
                b = bins.setdefault(key, dict(tool=source['tool'], parameter_bin=source['parameter_bin'],
                    component_count=0, tp_pixels=0, fp_pixels=0))
                b['tp_pixels'] += s['tp_pixels']
                b['fp_pixels'] += s['fp_pixels']
                touched.add(key)
            for key in touched:
                bins[key]['component_count'] += 1
    tp, fp = sum(r['gap_tp_pixels'] for r in rows), sum(r['gap_fp_pixels'] for r in rows)
    for b in bins.values():
        b['tp_share'] = b['tp_pixels']/tp if tp else 0.
        b['precision'] = b['tp_pixels']/max(1, b['tp_pixels']+b['fp_pixels'])
    seeds = sorted({r['seed'] for r in rows})
    per_seed = [[r for r in rows if r['seed'] == seed] for seed in seeds]
    if any([r['image'] for r in rs] != [r['image'] for r in per_seed[0]] for rs in per_seed):
        raise ValueError('Different image order/coverage across seeds')
    a = [[r['agent_counts'] for r in rs] for rs in per_seed]
    b = [[r['without_gap_counts'] for r in rs] for rs in per_seed]
    return dict(sessions=len(rows), images=len(per_seed[0]), seeds=seeds,
        component_count=sum(r['component_count'] for r in rows), gap_tp_pixels=tp, gap_fp_pixels=fp,
        gap_tp_fraction_of_agent_tp=tp/max(1, sum(r['agent_counts'][0] for r in rows)),
        unknown_pixel_fraction=sum(r['unknown_pixels'] for r in rows)/max(1, tp+fp),
        unknown_tp_fraction=sum(r['unknown_tp_pixels'] for r in rows)/max(1, tp),
        ambiguous_ancestry_pixels=sum(r['ambiguous_ancestry_pixels'] for r in rows),
        f1=bootstrap_delta(a, b),
        per_seed=[dict(seed=seed, sessions=len(rs), gap_tp_pixels=sum(r['gap_tp_pixels'] for r in rs),
            gap_fp_pixels=sum(r['gap_fp_pixels'] for r in rs),
            actual_f1=float(f1(np.sum([r['agent_counts'] for r in rs], axis=0))),
            without_gap_f1=float(f1(np.sum([r['without_gap_counts'] for r in rs], axis=0)))) for seed, rs in zip(seeds, per_seed)],
        tool_bins=sorted(bins.values(), key=lambda b: -b['tp_pixels']),
        examples=sorted(rows, key=lambda r: (-r['gap_tp_pixels'], r['session']))[:2], session_details=rows)


def groups(root):
    spec = read_json(root/'second/metrics_spec.json')['groups']
    for dataset, config, direction in [('second', c, d) for c in ('g2', 'g3') for d in ('appeared', 'disappeared')] + [('whu', 'g3', d) for d in ('any', 'appeared', 'disappeared')]:
        task = 'T1.1xC0' if direction == 'any' else 'T1.1xC1'
        manifest = root/dataset/'tasks'/f'{task}_{direction}.jsonl'
        tasks = read_jsonl(manifest)
        names = sorted(Path(r['id']).stem for r in tasks)
        if len(names) != len(set(names)) or len(names) != 128:
            raise ValueError('Expected 128 unique report images')
        fixed = Path(spec[f'{task}_{direction}']['fixed_dir']) if dataset == 'second' else root/'whu/fixed'/direction
        yield f'{dataset}_{direction}_{config}', dataset, config, direction, task, names, fixed


def verify_fixed(root, dataset, direction, fixed, names):
    settings, counts = read_json(fixed/'settings.json'), read_json(fixed/'counts.json')
    expected = dict(direction=direction, pixel_thr=.85, score_thr=.5, tile=512, overlap_thr=.25,
                    scope='building_rebuild_excluded')
    if settings != expected:
        raise ValueError('Unexpected fixed settings')
    kind = dict(any='c0', appeared='new', disappeared='demolished')[direction]
    total = np.zeros(3, np.int64)
    for name in names:
        c = pixel_counts(read_mask(fixed/(name+'.png')), read_mask(root/dataset/'gt'/kind/(name+'.png')))
        if c != [counts['per_image'][name][k] for k in ('tp', 'fp', 'fn')]:
            raise ValueError(f'Fixed stored counts mismatch: {dataset}/{name}')
        total += c
    if total.tolist() != [counts[k] for k in ('tp', 'fp', 'fn')]:
        raise ValueError('Fixed total differs')
    return dict(path=str(fixed.relative_to(root)), settings=settings, counts=total.tolist(), f1=float(f1(total)),
                validation='全部128图逐像素重算，与 counts.json 每图及总数一致')


def fmt_bin(b):
    return b['tool']+' '+', '.join(f'{k}={v}' for k, v in b['parameter_bin'].items())


def report(result):
    lines = ['# 补漏归因', '', '问题：Agent 补到了哪些固定路线没有的真变化，来自哪次工具调用？', '',
        '三个种子的像素数和组件数合并；F1 为各种子整组像素汇总后 F1 的均值。区间为配对图级 bootstrap：128 图有放回抽样，三个种子共享抽样索引，种子 1016，5000 次，95% 百分位区间。', '',
        '补漏 = A & ~F，扣除 F 后再取 8 连通，无最小面积。同一原组件与 F 重叠时仅保留剩余像素，可能拆为多个补漏组件。TP/FP 是像素口径，不能当作完整新发现建筑数。', '',
        '“去掉补漏”预测是 A & F。Δ = 实际 F1 − 去掉补漏 F1，单位百分点；这是固定最终答案上的消融估计，不是重跑 Agent 或工具因果效应。', '',
        '|组|会话|补漏组件|补漏 TP|补漏 FP|补漏 TP / 全部 TP|实际 F1|去掉补漏 F1|Δ [95% CI]|未知像素|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|---:|']
    for name, g in result['groups'].items():
        f = g['f1']; lo, hi = f['ci95_pp']
        lines.append(f"|{name}|{g['sessions']}|{g['component_count']}|{g['gap_tp_pixels']}|{g['gap_fp_pixels']}|{g['gap_tp_fraction_of_agent_tp']:.1%}|{f['actual_f1']:.2f}|{f['without_gap_f1']:.2f}|{f['delta_f1_pp']:+.2f} [{lo:+.2f}, {hi:+.2f}]|{g['unknown_pixel_fraction']:.2%}|")
    lines += ['', '## 工具与参数', '', '组件数为该档涉及的补漏组件数。同一组件跨档会重复计数；同档多个调用只计一次。TP/FP 每像素只归一个来源，可相加。完整调用与每组件分摊见 JSON 的 session_details。', '',
        '来源限定在最终句柄的祖先。subtract 沿 a，match_instances 沿同一时相实例；多个有效祖先包含同一像素时选最早来源调用。morph 新造像素归 morph。未知祖先证据不被后续调用覆盖。这里归的是像素产生来源，不代表后续选择、匹配、验证没有作用。', '',
        '本版本 add_region 只复制 other 中已存在的区域，因此像素来源继续追到 other；没有单列来源不表示该操作未被使用。refine_region 的内部 prompt 新增像素才归 refine_region。']
    for name, g in result['groups'].items():
        lines += ['', f'### {name}', '', '|工具 + 参数档|组件涉及数|TP 像素|FP 像素|占补漏 TP|精确率|', '|---|---:|---:|---:|---:|---:|']
        for b in g['tool_bins']:
            lines.append(f"|{fmt_bin(b)}|{b['component_count']}|{b['tp_pixels']}|{b['fp_pixels']}|{b['tp_share']:.1%}|{b['precision']:.1%}|")
        lines += ['', '核对例子：按会话补漏 TP 排名前二；同图不同种子可同时入选。每例列 TP 最大的前三个组件；其余组件及来源链见 JSON。', '']
        for e in g['examples']:
            lines += [f"- `{e['session']}`：图 `{e['image']}`，补漏 TP {e['gap_tp_pixels']}，FP {e['gap_fp_pixels']}。"]
            for c in sorted(e['components'], key=lambda c: -c['tp_pixels'])[:3]:
                parts = []
                for s in c['sources']:
                    src = e['sources'][str(s['source'])]
                    call = f"第 {src.get('step', '?')} 步第 {src.get('call_in_step', '?')} 调用 {src['tool']}"
                    parts.append(f"{call}，参数 {json.dumps(src.get('args', {}), ensure_ascii=False)}，TP/FP={s['tp_pixels']}/{s['fp_pixels']}")
                lines.append(f"  - 组件 {c['id']}，框 {c['bbox']}，TP/FP={c['tp_pixels']}/{c['fp_pixels']}；"+'；'.join(parts)+'。')
    lines += ['', '## 结论', '']
    # Distinguish actual useful parameter settings from broad tool names.
    second_low = [b for name, g in result['groups'].items() if name.startswith('second_')
        for b in g['tool_bins'] if b['tool'] == 'segment_text'
        and b['parameter_bin'].get('query') == 'building' and b['parameter_bin'].get('score_thr') == .3]
    tp = sum(b['tp_pixels'] for b in second_low); fp = sum(b['fp_pixels'] for b in second_low)
    lines.append(f'- SECOND 四组中，building 文本门限 0.3 的来源合计补回 TP {tp}、带入 FP {fp}，精确率 {tp/max(1,tp+fp):.1%}。固定路线门限为 0.5；放宽门限确实提供了补漏来源，也需要筛除 FP。')
    points = [b for name, g in result['groups'].items() if name.startswith('second_') and name.endswith('_g3')
        for b in g['tool_bins'] if b['tool'] == 'segment_prompt' and b['parameter_bin'].get('mode') == 'points']
    tp = sum(b['tp_pixels'] for b in points); fp = sum(b['fp_pixels'] for b in points)
    lines.append(f'- SECOND G3 点提示合计补回 TP {tp}、FP {fp}，精确率 {tp/max(1,tp+fp):.1%}。这是实际最终答案保留的像素来源；不能据此把所有点提示调用都视为有效。')
    for dataset in ('whu',):
        totals = defaultdict(lambda: [0, 0])
        for name, g in result['groups'].items():
            if name.startswith(dataset+'_'):
                for b in g['tool_bins']:
                    totals[b['tool']][0] += b['tp_pixels']; totals[b['tool']][1] += b['fp_pixels']
        tool, (tp, fp) = max(totals.items(), key=lambda kv: kv[1][0])
        alltp = sum(v[0] for v in totals.values())
        lines.append(f'- {dataset.upper()} 各方向与配置的描述性合计：{tool} 提供最多补漏 TP（{tp}，占 {tp/max(1, alltp):.1%}），同时带入 FP {fp}，精确率 {tp/max(1,tp+fp):.1%}。不同组不是独立图样本，不据此做显著性检验。')
    bad_bins = [(name, b) for name, g in result['groups'].items() for b in g['tool_bins'] if b['fp_pixels'] > b['tp_pixels']]
    worst = sorted(bad_bins, key=lambda item: -item[1]['fp_pixels'])[:3]
    lines.append('- '+('主要带入 FP 的参数档（按 FP 量取前三）：'+'；'.join(
        f"{name} 的 {fmt_bin(b)}，TP/FP={b['tp_pixels']}/{b['fp_pixels']}，精确率 {b['precision']:.1%}" for name, b in worst)+'。' if worst else '没有 FP 多于 TP 的参数档。'))
    uncertain = [name for name, g in result['groups'].items() if g['f1']['ci95_pp'][0] <= 0 <= g['f1']['ci95_pp'][1]]
    lines.append('- '+('补漏 F1 差的区间跨零：'+ '、'.join(uncertain)+'。这些组仍有真实补回的 TP，但整体 F1 收益尚未确认。' if uncertain else '各组补漏 F1 差的区间均未跨零；具体方向与幅度见总表。'))
    unknown = sum(r['unknown_pixels'] for g in result['groups'].values() for r in g['session_details'])
    lines += ['', f'未知来源共 {unknown} 像素。归因是祖先内最早像素来源；相同 building 查询也可能通过不同匹配、选择和组合把固定路线丢掉的像素带回，不能把这类收益都理解成换查询。']
    lines += ['', '## 核对与限制', '',
        '固定路线均核对 settings.json、生成代码，并逐图重算 TP/FP/FN 对照 counts.json。SECOND 路径另对照 second/metrics_spec.json。所有会话的最终掩码计数对照 final.json；未跳过错误兜底或空答案。', '',
        '多祖先不同来源的歧义数、三个种子的分开结果、每图每组件分摊均写入 JSON。未运行模型、API、Slurm、GPU；未做可选 LEVIR 参照。未重建已执行工具，也未用验证器回答替代真值。', '',
        f"运行时间 {result['elapsed_seconds']:.1f} 秒；读取 OSError 重试 {result['io_retries']} 次。所有轨迹路径相对 R。"]
    return '\n'.join(lines)+'\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, default=R_DEFAULT)
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[3]/'results')
    p.add_argument('--checkpoint', type=Path, help='Optional local directory for per-session resume files')
    args = p.parse_args()
    started = time.monotonic()
    result = dict(method=dict(gap='A & ~F; 8-connectivity', f1='mean of seed pooled F1; pp',
        bootstrap='paired image resampling; same indices across fixed three seeds', samples=5000,
        seed=1016, ancestry='earliest eligible ancestral call per pixel; unknown evidence preserved'), groups={})
    for name, dataset, config, direction, task, names, fixed in groups(args.run_root):
        validation = verify_fixed(args.run_root, dataset, direction, fixed, names)
        rows = []
        cached = {}
        cache_path = None
        if args.checkpoint:
            args.checkpoint.mkdir(parents=True, exist_ok=True)
            cache_path = args.checkpoint/(name+'.jsonl')
            if cache_path.exists():
                cached = {r['session']: r for r in read_jsonl(cache_path)}
        for seed in ('1', '2', '3'):
            base = args.run_root/f'{dataset}_{config}_{direction}_s{seed}'/task/'val'
            actual = retry_read(base, lambda p: sorted(x.name for x in p.iterdir() if x.is_dir()))
            if actual != names:
                raise ValueError(f'Session coverage differs: {base.name}')
            for image in names:
                session = str((base/image).relative_to(args.run_root))
                row = cached.get(session)
                if row is None:
                    row = analyze_session(base/image, fixed/(image+'.png'), args.run_root)
                    if cache_path:
                        with cache_path.open('a') as stream:
                            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
                rows.append(row)
                if len(rows) % 32 == 0:
                    print(f'{name}: {len(rows)}/384 sessions', flush=True)
            print(f'{name} s{seed}: {len(names)} sessions', flush=True)
        group = summarize(rows)
        group['fixed_validation'] = validation
        result['groups'][name] = group
        if args.checkpoint:
            (args.checkpoint/(name+'_summary.json')).write_text(json.dumps(group, ensure_ascii=False))
        print(f"{name}: TP={group['gap_tp_pixels']} FP={group['gap_fp_pixels']} unknown={group['unknown_pixel_fraction']:.4%}", flush=True)
    result['elapsed_seconds'] = time.monotonic()-started
    result['io_retries'] = RETRIES['retries']
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'gap_attribution.json').write_text(json.dumps(result, ensure_ascii=False, indent=1)+'\n')
    (args.output/'gap_attribution.md').write_text(report(result))
    print(f"Finished in {result['elapsed_seconds']:.1f}s", flush=True)


if __name__ == '__main__':
    main()
