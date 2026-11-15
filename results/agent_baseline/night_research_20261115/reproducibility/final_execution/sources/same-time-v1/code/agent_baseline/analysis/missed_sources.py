"""Development-only multi-source missed-change ablation; no inference on import.

Commands inventory/fixed/baseline/generate/statistics/estimate/ask/report separate
GPU work, statistics and paid verification. See README_missed_sources.md.
"""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from agent_baseline import instances
from agent_baseline.segment_cache import manifest_cache_path
from .io_retry import read_text, read_image, write_json, write_text, retry_io

VERSION = 1
SOURCES = ('S1_03', 'S1_02', 'S2', 'S3', 'S4', 'S5')
S1 = 'S1_02'  # 0.3 remains a separately reported control; freeze before scoring.
GRID = tuple(round(x / 20, 2) for x in range(41))  # 1-cosine in [0,2]
STAGING = Path(__file__).resolve().parents[3]
GROUPS = ('second_appeared', 'second_disappeared', 'whu_any',
          'whu_appeared', 'whu_disappeared', 'levir_any')


def exists(path):
    return retry_io(lambda: Path(path).exists())


def load_json(path):
    return json.loads(read_text(path))


def jsonl(path):
    return [json.loads(x) for x in read_text(path).splitlines() if x.strip()]


def save_npz(path, **arrays):
    path = Path(path)
    def save():
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix('.tmp.npz')
        np.savez_compressed(temp, **arrays)
        temp.replace(path)
    retry_io(save)


def load_npz(path):
    def load():
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k].copy() for k in z.files}
    return retry_io(load)


def stack(masks, shape):
    return np.asarray(masks, bool).reshape(-1, *shape)


def union(masks, shape):
    return np.any(stack(masks, shape), axis=0)


def split_components(mask):
    from scipy import ndimage as ndi
    labels, n = ndi.label(np.asarray(mask, bool), np.ones((3, 3), int))
    return [labels == i for i in range(1, n + 1)]


def change_candidates(t1, t2, start, direction, overlap=.25):
    """Match raw single-date instances BEFORE subtracting the starting mask.

    Presence uses any single opposite-date instance covering >=25% of the
    queried instance, independently in each direction (instances.match).
    """
    if direction not in ('any', 'appeared', 'disappeared'):
        raise ValueError('Unknown change direction')
    a, b = (stack(x, start.shape) for x in (t1, t2))
    appeared, disappeared, _ = instances.match(
        instances.from_dense(a, np.ones(len(a))),
        instances.from_dense(b, np.ones(len(b))), overlap, 'overlap')
    chosen = (appeared + disappeared if direction == 'any' else
              appeared if direction == 'appeared' else disappeared)
    return split_components(instances.to_mask(chosen, start.shape) & ~start)


def pixel_counts(pred, truth):
    return np.array([np.count_nonzero(pred & truth), np.count_nonzero(pred & ~truth),
                     np.count_nonzero(~pred & truth)], np.int64)


def f1(counts):
    tp, fp, fn = np.asarray(counts).sum(axis=0) if np.asarray(counts).ndim == 2 else counts
    return float(200 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 100.


def ratio(a, b):
    return float(a / b) if b else None


def oracle_add(start, masks, truth):
    """Fixed TP-majority component oracle, not unrestricted optimal subset F1.

    Re-componentise the source union after removing start. Keep an entire
    8-connected component iff TP>FP; never trim components by truth pixels.
    """
    candidates = split_components(union(masks, start.shape) & ~start)
    kept = [m for m in candidates if np.count_nonzero(m & truth) > np.count_nonzero(m & ~truth)]
    pred = start | union(kept, start.shape)
    return dict(counts=pixel_counts(pred, truth).tolist(), f1=f1(pixel_counts(pred, truth)),
                candidates=len(candidates), kept=len(kept))


def measure(start, truth, masks, s1_masks):
    """Coverage >=50% of each ORIGINAL missed 8-component by candidate union.

    Also report residual components after removing S1-covered pixels. Component
    delta on original objects and residual-fragment recall are distinct fields.
    """
    proposed = union(masks, start.shape) & ~start
    base = union(s1_masks, start.shape) & ~start
    missing = truth & ~start
    residual = missing & ~base
    novel = proposed & residual
    comps, left = split_components(missing), split_components(residual)
    hit = lambda m, p: np.count_nonzero(m & p) >= .5 * np.count_nonzero(m)
    baseline_hit = sum(hit(m, base) for m in comps)
    together_hit = sum(hit(m, base | proposed) for m in comps)
    return dict(missing_pixels=int(missing.sum()), missing_components=len(comps),
        covered_pixels=int((proposed & missing).sum()), covered_components=sum(hit(m, proposed) for m in comps),
        candidate_pixels=int(proposed.sum()), candidate_tp=int((proposed & truth).sum()),
        candidates=len(split_components(proposed)), novel_pixels=int(novel.sum()),
        residual_pixels=int(residual.sum()), residual_components=len(left),
        covered_residual_components=sum(hit(m, proposed) for m in left),
        added_original_components=together_hit-baseline_hit,
        pixel_recall=ratio((proposed & missing).sum(), missing.sum()),
        component_recall=ratio(sum(hit(m, proposed) for m in comps), len(comps)),
        precision=ratio((proposed & truth).sum(), proposed.sum()),
        marginal_pixel_recall=ratio(novel.sum(), missing.sum()),
        marginal_component_recall=ratio(together_hit-baseline_hit, len(comps)),
        residual_pixel_recall=ratio(novel.sum(), residual.sum()),
        residual_component_recall=ratio(sum(hit(m, proposed) for m in left), len(left)))


def summarize_measures(rows):
    fields = ('missing_pixels', 'missing_components', 'covered_pixels', 'covered_components',
              'candidate_pixels', 'candidate_tp', 'candidates', 'novel_pixels', 'residual_pixels',
              'residual_components', 'covered_residual_components', 'added_original_components')
    out = {k: int(sum(r[k] for r in rows)) for k in fields}
    for key, numerator, denominator in (
        ('pixel_recall','covered_pixels','missing_pixels'),
        ('component_recall','covered_components','missing_components'),
        ('precision','candidate_tp','candidate_pixels'),
        ('marginal_pixel_recall','novel_pixels','missing_pixels'),
        ('marginal_component_recall','added_original_components','missing_components'),
        ('residual_pixel_recall','novel_pixels','residual_pixels'),
        ('residual_component_recall','covered_residual_components','residual_components')):
        out[key] = ratio(out[numerator], out[denominator])
    contributors = sorted((dict(image=r['image'], pixels=r['novel_pixels'])
                           for r in rows if r['novel_pixels']), key=lambda r: (-r['pixels'], r['image']))
    out.update(images=len(rows), candidates_per_image=ratio(out['candidates'], len(rows)),
        per_image=[dict(image=r['image'], candidates=r['candidates'], novel_pixels=r['novel_pixels']) for r in rows],
        contributing_images=len(contributors), top_images=[dict(r, share=ratio(r['pixels'],out['novel_pixels']))
                                                         for r in contributors[:5]],
        top2_share=ratio(sum(r['pixels'] for r in contributors[:2]), out['novel_pixels']))
    return out


def inclusion(metrics, delta):
    if not metrics['novel_pixels']:
        return '无新增真变化，不收录'
    if delta is None:
        return '待验证器；尚不能判断收录'
    if delta <= 0:
        return '验证器后净增益不为正，不收录'
    if metrics['contributing_images'] <= 2 or (metrics['top2_share'] or 0) >= .8:
        return '少数图贡献集中，单列分析，不作为普遍来源收录'
    return '有新增真变化、净增益为正且分布较广，可收录'


def quotas(source_metrics, total=32):
    """Highest-averages precision allocation, capped by observed per-image p95.

    This is a return-count suggestion, not a new tool or a tuned policy.
    S1_03 is the sensitivity control and is not allocated a second quota.
    """
    def usable(k,r):
        if k=='S1_03' or r.get('precision') is None or r['precision']<=0 or r['candidates']==0:
            return False
        if k!=S1 and 'novel_pixels' in r:
            if not r['novel_pixels'] or r['contributing_images']<=2 or (r['top2_share'] or 0)>=.8:
                return False
            if r.get('net_delta_after_s1') is not None and r['net_delta_after_s1']<=0:
                return False
        return True
    eligible = {k:r for k,r in source_metrics.items() if usable(k,r)}
    limits = {k: int(math.ceil(np.percentile([x['candidates'] for x in r['per_image']],95)))
              for k,r in eligible.items()}
    count = {k: 0 for k in source_metrics}
    for _ in range(total):
        active = [k for k in eligible if count[k] < limits[k]]
        if not active:
            break
        # Proportional fair rounding; deterministic source-name tie break.
        k = max(active, key=lambda k: (eligible[k]['precision']/(count[k]+1), k))
        count[k] += 1
    return dict(total_budget=total, suggested=count, observed_p95_cap=limits,
                provisional=any(k!=S1 and r.get('net_delta_after_s1') is None for k,r in eligible.items()),
                basis='按像素精确率比例分配，逐名额最高平均数法；各来源不超过开发候选数p95；S1_03仅作敏感性对照')


def output_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(STAGING):
        raise ValueError('Outputs must stay inside this staging directory')
    return path


def reference(path, run):
    return os.path.relpath(Path(path).resolve(), Path(run).resolve())


def specs(run, work):
    out = []
    for group in GROUPS:
        dataset, direction = group.split('_', 1)
        spec = dict(group=group, dataset=dataset, direction=direction,
                    scope='building' if dataset == 'levir' else 'building_rebuild_excluded')
        if dataset == 'second':
            root = f'second_dev/rule_card_defined/{direction}/{direction}'
            spec.update(names='second_dev/names.txt', expected=60,
                tasks=f'second_dev/second/tasks/T1.1xC1_{direction}.jsonl',
                fixed=f'second_dev/second/fixed/{direction}', saved_rule=f'{root}/masks',
                settings=f'{root}/settings.json', answers=f'{root}/answers.jsonl',
                cache_roots=[run/'second_dev', run])
        elif dataset == 'whu':
            task = 'T1.1xC0' if direction == 'any' else 'T1.1xC1'
            spec.update(names='whu/names_dev60.txt', expected=60,
                tasks=f'whu/tasks/{task}_{direction}_dev.jsonl',
                fixed=work/'whu_fixed'/group, answers=work/'whu_rule'/group/'answers.json',
                cache_roots=[run])
        else:
            spec.update(names='dev2/names.txt', expected=200, fixed='init/sam3_imc_p85/train',
                        answers='ask_eval/dev2/f37_corr.jsonl', cache_roots=[run])
        out.append(spec)
    return out


def names_for(run, spec):
    names = read_text(run/spec['names']).split()
    if len(names) != spec['expected'] or len({Path(x).stem for x in names}) != len(names):
        raise ValueError(f"Unexpected development list: {spec['group']}")
    return names


def rows_for(run, spec):
    listed = names_for(run,spec)
    if spec['dataset'] == 'levir':
        from common import LEVIR
        return [dict(id=Path(n).stem, image_t1=str(LEVIR/'train_original/A'/n),
            image_t2=str(LEVIR/'train_original/B'/n), gt_mask=str(LEVIR/'train_original/label'/n),
            target='building', change_direction='any', scope='building') for n in listed]
    rows = jsonl(run/spec['tasks'])
    by_name = {Path(r['image_t1']).stem: r for r in rows}
    wanted = {Path(n).stem for n in listed}
    if set(by_name) != wanted or len(rows) != len(wanted):
        raise ValueError('Manifest must contain exactly the frozen development images')
    return [dict(by_name[Path(n).stem], scope=spec['scope']) for n in listed]


def audit_settings(run, spec):
    if spec['dataset'] != 'second':
        return dict(scope=spec['scope'], variant='f37_corr')
    conf = load_json(run/spec['settings'])
    actual = (conf['direction'], conf['scope'], conf['variant'])
    if actual != (spec['direction'], 'dev_card_defined', 'f37_corr'):
        raise ValueError('SECOND starting rule settings differ from frozen development policy')
    if {Path(x).stem for x in conf['ids']} != {Path(x).stem for x in names_for(run,spec)}:
        raise ValueError('SECOND settings do not match development list')
    if Path(conf['mask_dir']).resolve() != (run/spec['fixed']).resolve():
        raise ValueError('SECOND settings mask_dir differs')
    fixed = load_json(run/spec['fixed']/'settings.json')
    if any(fixed[k] != v for k,v in dict(direction=spec['direction'],pixel_thr=.85,
                score_thr=.5,tile=512,overlap_thr=.25).items()):
        raise ValueError('Unexpected fixed-route configuration')
    return dict(scope=spec['scope'], original_scope='dev_card_defined', variant='f37_corr',
                evidence='batch97已核对旧卡与当前building_rebuild_excluded内容一致',
                settings=spec['settings'])


def answer_map(path):
    if not exists(path):
        return {}
    rows = list(load_json(path).values()) if Path(path).suffix == '.json' else jsonl(path)
    return {r['uid']:r for r in rows}


def ledger_unit(run, spec):
    # Only same-pool development answers are inspected; no report-set fallback.
    if spec['dataset'] == 'whu':
        own = answer_map(spec['answers'])
        if own:
            records = list(own.values())
            basis = reference(spec['answers'],run)
        else:
            peer='second_disappeared' if spec['direction']=='disappeared' else 'second_appeared'
            other = next(s for s in specs(run, run/'unused') if s['group'] == peer)
            unit = ledger_unit(run,other)
            return dict(unit, basis='WHU开发ask缺失；借用SECOND开发同模型/同范围卡单价：'+unit['basis'], proxy=True)
    else:
        records = list(answer_map(run/spec['answers']).values())
        wanted = {Path(n).stem for n in names_for(run,spec)}
        records = [r for r in records if r['uid'].rsplit(':',1)[0] in wanted]
        basis = spec['answers']
    priced = [r for r in records if r.get('cost_cny') is not None and r['cost_cny'] > 0]
    cost = sum(r['cost_cny'] for r in priced)
    return dict(unit_cost_cny=ratio(cost,len(priced)), priced_requests=len(priced),
                recorded_cost_cny=cost, basis=basis, proxy=False,
                price_note='沿用已有ledger的cost_cny/有价格请求数；仅估算，以实际账单为准')


def cache_probe(run, spec, row):
    result = []
    for time in ('t1','t2'):
        found = []
        for root in spec['cache_roots']:
            p = manifest_cache_path(root,row['image_'+time],'building',512,.1,.85)
            if exists(p):
                found.append(reference(p,run))
        result.append(dict(time=time, low_threshold_manifest=found,
                           supports_02=bool(found), soft_cache_min_score=.3))
    return result


def inventory(run, work):
    report = dict(status='待Claude运行GPU与ask；无真实来源召回结果', s1_reference=S1,
                  criterion='新增真变化；验证器后净ΔF1>0；不只靠1–2张图。旧5个百分点门槛已取消。',
                  groups={}, executed=dict(gpu=False, api=False, slurm=False),
                  method=dict(s3_s4_direction='SegEarth建筑/背景概率>.5的T2增/T1减，SAM提议裁该门控及起点',
                    s3_area='原SAM提议面积>图像10%',dino='区域均值1-cosine，开发扫描0–2步长.05',
                    s5='明确yes起点组件；同图同期示例框；另一日期参考/查询拼图；原实例overlap .25匹配'))
    for spec in specs(run,work):
        rows = rows_for(run,spec)
        spec_audit = audit_settings(run,spec)
        report['groups'][spec['group']] = dict(images=len(rows),names=spec['names'],
            policy=spec_audit, ledger=ledger_unit(run,spec),
            cache_first_image_probe=cache_probe(run,spec,rows[0]),
            source_metrics=None, whu_fixed_candidates=None if spec['dataset']=='whu' else '不适用',
            costs_pending='固定路线或来源候选未生成；候选数×ledger单价，未知不记0',
            starting_fixed=reference(run/spec['fixed'],run))
    report['cost_estimates']={g:dict(status='待来源候选',requests=None,cost_cny=None,
        unit_cost_cny=r['ledger']['unit_cost_cny'],cost_per_1000_cny=r['ledger']['unit_cost_cny']*1000
        if r['ledger']['unit_cost_cny'] is not None else None) for g,r in report['groups'].items()}
    report['whu_fixed_cost_estimates']={g:dict(status='待WHU开发固定路线点数',requests=None,cost_cny=None,
        unit_cost_cny=r['ledger']['unit_cost_cny']) for g,r in report['groups'].items() if g.startswith('whu_')}
    return report


def selected_rows(run, spec, shard, shards):
    if shards < 1 or not 0 <= shard < shards:
        raise ValueError('Require 0 <= shard < shards')
    return rows_for(run,spec)[shard::shards]


def ensure_settings(path, value):
    if exists(path) and load_json(path) != value:
        raise ValueError(f'Input/configuration changed; use a fresh output: {path.name}')
    write_json(path,value)


def build_baseline(run, work, spec, shard=0, shards=1, resume=False):
    audit = audit_settings(run,spec)
    folder = work/'base'/spec['group']
    ensure_settings(folder/'settings.json',dict(version=VERSION,policy=audit,
        fixed=reference(run/spec['fixed'],run),answers=reference(run/spec['answers'],run),names=spec['names']))
    answers = answer_map(run/spec['answers'])
    for row in selected_rows(run,spec,shard,shards):
        stem = Path(row['id']).stem
        dest = folder/(stem+'.npz')
        if resume and exists(dest):
            continue
        mask = read_image(run/spec['fixed']/(stem+'.png')) > 0
        # Existing rule IDs use the same 8-component convention as common.components.
        components = split_components(mask)
        required = [f'{stem}:{i+1}' for i in range(len(components))]
        missing = [uid for uid in required if uid not in answers]
        if missing:
            raise ValueError(f'Missing starting-rule answers: {spec["group"]}/{stem} ({len(missing)})')
        kept = [m for i,m in enumerate(components) if answers[required[i]].get('answer') != 'no']
        yes = [m for i,m in enumerate(components) if answers[required[i]].get('answer') == 'yes']
        start = union(kept,mask.shape)
        if spec.get('saved_rule'):
            saved = read_image(run/spec['saved_rule']/(stem+'.png')) > 0
            if not np.array_equal(saved,start):
                raise ValueError('Reconstructed SECOND rule differs from saved rule')
        save_npz(dest,start=start,yes=pack(yes,mask.shape),shape=mask.shape)


def pack(masks, shape):
    data = stack(masks,shape)
    return np.packbits(data.reshape(len(data),int(np.prod(shape))),axis=1)


def unpack(bits, shape):
    return np.unpackbits(bits,axis=1,count=int(np.prod(shape))).reshape(-1,*shape).astype(bool)


def raw_sources(raw, start, thresholds):
    shape = tuple(map(int,raw['shape']))
    out = {}
    for source in SOURCES:
        masks = unpack(raw[source],shape)
        if source in ('S3','S4'):
            masks = masks[raw[source+'_scores'] > thresholds[source]]
        # All proposals of one source form a union; ask unit is its 8-components.
        out[source] = split_components(union(masks,shape) & ~start)
    return out


def scan_thresholds(run,work,spec,rows):
    counts = {source:{t:np.zeros(3,np.int64) for t in GRID} for source in ('S3','S4')}
    for row in rows:
        stem = Path(row['id']).stem
        start = load_npz(work/'base'/spec['group']/(stem+'.npz'))['start']
        truth = read_image(row['gt_mask']) > (127 if spec['dataset']=='levir' else 0)
        raw = load_npz(work/'raw'/spec['group']/(stem+'.npz'))
        for source in counts:
            masks = unpack(raw[source],start.shape)
            for t in GRID:
                pred = start | union(masks[raw[source+'_scores'] > t],start.shape)
                counts[source][t] += pixel_counts(pred,truth)
    selected = {s:min(GRID,key=lambda t:(-f1(counts[s][t]), t)) for s in counts}
    return selected, {s:[dict(threshold=t,counts=counts[s][t].tolist(),f1=f1(counts[s][t])) for t in GRID]
                      for s in counts}


def statistics_group(run,work,spec,resume=False):
    rows = rows_for(run,spec)
    folder=work/'statistics'/spec['group']
    inputs=[]
    for row in rows:
        stem=Path(row['id']).stem
        for path in (work/'base'/spec['group']/(stem+'.npz'),
                     work/'raw'/spec['group']/(stem+'.npz'),Path(row['gt_mask'])):
            stat=retry_io(lambda path=path:path.stat())
            inputs.append(dict(path=reference(path,run),size=stat.st_size,mtime_ns=stat.st_mtime_ns))
    ensure_settings(folder/'settings.json',dict(version=VERSION,inputs=inputs,s1=S1,grid=list(GRID)))
    checkpoint=folder/'summary.json'
    if resume and exists(checkpoint):
        return load_json(checkpoint)
    thresholds, scan = scan_thresholds(run,work,spec,rows)
    measurement = {s:[] for s in SOURCES}
    baseline_counts, upper_s1, upper_all = [], [], []
    for row in rows:
        stem = Path(row['id']).stem
        start = load_npz(work/'base'/spec['group']/(stem+'.npz'))['start']
        truth = read_image(row['gt_mask']) > (127 if spec['dataset']=='levir' else 0)
        raw = load_npz(work/'raw'/spec['group']/(stem+'.npz'))
        masks = raw_sources(raw,start,thresholds)
        baseline_counts.append(pixel_counts(start,truth))
        upper_s1.append(oracle_add(start,masks[S1],truth)['counts'])
        # Include both S1 thresholds: matching presence can be non-monotone.
        upper_all.append(oracle_add(start,[m for s in SOURCES for m in masks[s]],truth)['counts'])
        for source in SOURCES:
            measurement[source].append(dict(image=stem,**measure(start,truth,masks[source],masks[S1])))
            dest = work/'candidates'/spec['group']/source/(stem+'.npz')
            save_npz(dest,masks=pack(masks[source],start.shape),shape=start.shape)
    summary = {s:summarize_measures(v) for s,v in measurement.items()}
    ranked = sorted(('S2','S3','S4','S5'),key=lambda s:(-summary[s]['novel_pixels'],s))
    priority = [S1]+[s for s in ranked[:2] if summary[s]['novel_pixels'] > 0]
    # The task supplement asks for per-source verified deltas. Estimate all
    # sources, but mark the original step-12 top-two plan as the first priority.
    chosen = [S1]+ranked
    result = dict(images=len(rows),policy=audit_settings(run,spec),baseline_f1=f1(baseline_counts),
                baseline_counts=np.sum(baseline_counts,axis=0).tolist(),
                thresholds=thresholds, threshold_scan=scan, source_metrics=summary,
                upper_s1=dict(f1=f1(upper_s1),counts=np.sum(upper_s1,axis=0).tolist()),
                upper_all=dict(f1=f1(upper_all),counts=np.sum(upper_all,axis=0).tolist()),
                selected_for_ask=chosen, priority_for_ask=priority, quota_suggestion=quotas(summary),
                ledger=ledger_unit(run,spec), names=spec['names'])
    write_json(checkpoint,result)
    return result


def candidate_records(run,work,spec,sources,combined=False,fixed=False):
    """Yield mask components for costs/ask, with stable IDs within each pool."""
    for row in rows_for(run,spec):
        stem = Path(row['id']).stem
        if fixed:
            mask = read_image(work/'whu_fixed'/spec['group']/(stem+'.png')) > 0
            by_source = {'fixed':split_components(mask)}
        else:
            base = load_npz(work/'base'/spec['group']/(stem+'.npz'))['start']
            by_source = {}
            for source in sources:
                z = load_npz(work/'candidates'/spec['group']/source/(stem+'.npz'))
                by_source[source] = list(unpack(z['masks'],tuple(z['shape'])))
            if combined:
                by_source = {'+'.join(sources):split_components(union(
                    [m for masks in by_source.values() for m in masks],base.shape))}
        for source,masks in by_source.items():
            for i,mask in enumerate(masks,1):
                y,x = np.nonzero(mask)
                yield dict(uid=f'{stem}:{i}',pool=source,row=row,
                    idx=np.flatnonzero(mask).astype(np.int32),shape=mask.shape,
                    bbox=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)])


def estimate(run,work,report,whu_fixed=False):
    result = {}
    for spec in specs(run,work):
        group = spec['group']
        if whu_fixed and spec['dataset'] != 'whu':
            continue
        item = report.get('groups',{}).get(group,{})
        if not whu_fixed and not item.get('source_metrics'):
            result[group] = dict(status='待来源统计',requests=None,cost_cny=None)
            continue
        source_sets = [('fixed',)] if whu_fixed else [(s,) for s in item['selected_for_ask']]
        if not whu_fixed:
            # Individual-source deltas and incremental net gain beyond verified S1.
            source_sets += [(S1,s) for s in item['selected_for_ask'] if s != S1]
        ledger = ledger_unit(run,spec)
        pools = {}
        for sources in source_sets:
            pool = '+'.join(sources)
            answer_file = (work/'whu_rule'/group/'answers.json' if whu_fixed else
                           work/'ask'/group/pool/'answers.json')
            done = answer_map(answer_file)
            n = pending = 0
            for record in candidate_records(run,work,spec,sources,combined=len(sources)>1,fixed=whu_fixed):
                n += 1
                pending += record['uid'] not in done
            unit = ledger['unit_cost_cny']
            pools[pool] = dict(candidates=n, remaining_requests=pending,
                priority=whu_fixed or all(s in item.get('priority_for_ask',[]) for s in sources),
                cost_cny=n*unit if unit is not None else None,
                remaining_cost_cny=pending*unit if unit is not None else None,
                max_cost_suggestion_cny=max(.1,math.ceil(pending*unit*1.5*100)/100) if unit else None)
        result[group] = dict(ledger=ledger,pools=pools)
    return result


def score_answers(run,work,spec,sources,fixed=False):
    pool = '+'.join(sources)
    path = work/'whu_rule'/spec['group']/'answers.json' if fixed else work/'ask'/spec['group']/pool/'answers.json'
    done = answer_map(path)
    totals, no, unparsed, required = [], 0, 0, 0
    for row in rows_for(run,spec):
        stem = Path(row['id']).stem
        if fixed:
            start = read_image(work/'whu_fixed'/spec['group']/(stem+'.png')) > 0
            candidates = split_components(start)
            pred = np.zeros_like(start)
        else:
            start = load_npz(work/'base'/spec['group']/(stem+'.npz'))['start']
            masks = []
            for source in sources:
                z=load_npz(work/'candidates'/spec['group']/source/(stem+'.npz'))
                masks.extend(unpack(z['masks'],start.shape))
            candidates = split_components(union(masks,start.shape))
            pred = start.copy()
        for i,m in enumerate(candidates,1):
            uid=f'{stem}:{i}'
            required += 1
            if uid not in done:
                return None
            answer = done[uid].get('answer')
            no += answer == 'no'
            unparsed += answer not in ('yes','no')
            if answer != 'no':
                pred |= m
        if fixed:
            from .io_retry import save_image
            save_image(work/'whu_rule'/spec['group']/'masks'/(stem+'.png'),pred.astype(np.uint8)*255)
        truth = read_image(row['gt_mask']) > (127 if spec['dataset']=='levir' else 0)
        totals.append(pixel_counts(pred,truth))
    return dict(f1=f1(totals),counts=np.sum(totals,axis=0).tolist(),candidates=required,
                no=int(no),unparsed=int(unparsed),cost_cny=sum(x.get('cost_cny',0) for x in done.values()))


def update_rules(run,work,report):
    for spec in specs(run,work):
        group=report['groups'][spec['group']]
        if not group.get('source_metrics'):
            continue
        rules={}
        for source in group['selected_for_ask']:
            rules[source]=score_answers(run,work,spec,[source])
            if source != S1:
                rules[S1+'+'+source]=score_answers(run,work,spec,[S1,source])
        control=rules.get(S1)
        for source,metrics in group['source_metrics'].items():
            individual=rules.get(source)
            combined=rules.get(S1+'+'+source)
            metrics['rule_f1']=individual['f1'] if individual else None
            metrics['delta_from_start']=individual['f1']-group['baseline_f1'] if individual else None
            metrics['delta_vs_s1']=individual['f1']-control['f1'] if individual and control else None
            metrics['net_delta_after_s1']=combined['f1']-control['f1'] if combined and control else None
            metrics['decision']=inclusion(metrics,metrics['net_delta_after_s1']) if source != S1 else '低门限基准'
        group['verified_rules']=rules
        group['quota_suggestion']=quotas(group['source_metrics'])


def markdown(report):
    lines=['# 多来源补漏前置实验','',report['status'],'',
        '仅开发集。S1 主对照固定为 0.2；0.3 单列敏感性。未运行的数字写“待运行”，不写零。',
        '旧 5 个百分点门槛已取消。收录需有 S1 以外真变化、验证器后净 ΔF1>0，并检查是否只靠 1–2 张图。',
        '组件口径：8 连通漏检组件，被来源并集覆盖≥50%；残余组件在扣除 S1 像素后重新连通。',
        '边际百分点以原漏检为分母；残余召回以 S1 后剩余漏检为分母。精确率按像素并集。','']
    for name,g in report['groups'].items():
        lines += [f'## {name}（{g["images"]} 张）','',
                  f'ledger 单价：{g["ledger"]["unit_cost_cny"]} 元/组件。来源：`{g["ledger"]["basis"]}`。','']
        if not g.get('source_metrics'):
            lines += ['| 来源 | 漏检召回/组件 | 精确率 | 候选/图 | 剩余漏检召回 | 净ΔF1 | 判断 |',
                      '|---|---|---|---|---|---|---|']
            lines += [f'| {s} | 待运行 | 待运行 | 待点数 | 待运行 | 待ask | 待判断 |' for s in SOURCES]
            lines += ['', ('WHU 开发起点须先生成。' if name.startswith('whu_') else '来源候选尚未生成。')+
                      '候选数未知，费用为“候选数 × 上述单价”。','']
            continue
        fmt=lambda v: '—' if v is None else f'{100*v:.2f}%'
        num=lambda v: '待ask' if v is None else f'{v:+.3f}'
        lines += [f'起点 F1={g["baseline_f1"]:.3f}；TP为主组件选择参照：S1={g["upper_s1"]["f1"]:.3f}，全部来源={g["upper_all"]["f1"]:.3f}。',
                  '这两个“上限”保留 TP>FP 的整组件；不是对任意候选子集求最优 F1，全部来源连通后可能合并组件。',
                  f'S3/S4 门限：{g["thresholds"]}；开发集0–2步长0.05扫描，起点+来源累计像素F1最高，并列取小。','',
                  '| 来源 | 像素/组件召回 | 精确率 | 候选/图 | 边际像素/组件 | 剩余像素/组件召回 | 图数/前2图占比 | Δ起点 / 净ΔS1 | 判断 |',
                  '|---|---|---|---|---|---|---|---|---|']
        for s,r in g['source_metrics'].items():
            lines += [f'| {s} | {fmt(r["pixel_recall"])}/{fmt(r["component_recall"])} | {fmt(r["precision"])} | {r["candidates_per_image"]:.2f} | {fmt(r["marginal_pixel_recall"])}/{fmt(r["marginal_component_recall"])} | {fmt(r["residual_pixel_recall"])}/{fmt(r["residual_component_recall"])} | {r["contributing_images"]}/{fmt(r["top2_share"])} | {num(r.get("delta_from_start"))} / {num(r.get("net_delta_after_s1"))} | {r.get("decision","待ask")} |']
        lines += ['', '贡献最大的图（新增 TP / 全部新增 TP）：']
        for s,r in g['source_metrics'].items():
            lines += [f'- {s}：'+('；'.join(f'{x["image"]} {x["pixels"]}像素/{fmt(x["share"])}' for x in r['top_images']) or '无新增TP')]
        lines += ['',f'返回配额建议（总预算32；待核验初步={g["quota_suggestion"]["provisional"]}）：`{g["quota_suggestion"]["suggested"]}`。已观察到无新增/净增益不为正或过于集中的来源不分配；其余按开发像素精确率和候选密度建议，未经在线验证。','']
    lines += ['## 费用估计','',
              '| 阶段/组/池 | 候选 | 剩余请求 | 估价（元） | 优先 |',
              '|---|---:|---:|---:|---|']
    for stage in ('whu_fixed_cost_estimates','cost_estimates'):
        for group,item in report.get(stage,{}).items():
            if 'pools' not in item:
                lines += [f'| {stage}/{group} | 待点数 | 未知 | 未知 | 待运行 |']
                continue
            for pool,r in item['pools'].items():
                cost='未知' if r['cost_cny'] is None else f'{r["cost_cny"]:.4f}'
                lines += [f'| {stage}/{group}/{pool} | {r["candidates"]} | {r["remaining_requests"]} | {cost} | {r["priority"]} |']
    lines += ['', '## 未验证项','',
              '精确费用见 JSON 的 cost_estimates；GPU 候选未产生前不估造候选数。WHU 自身 ledger 缺失时借用 SECOND 开发同卡 flash 单价，明确标记代理估计。',
              '实现交付阶段未执行 GPU、API 或 Slurm。实际执行状态以运行日志和已填统计/ask字段为准。',
              'S3/S4通过SegEarth建筑增减限制目标和方向；该门控可能遗漏建筑，计入实际召回。S5另一日期使用参考/查询拼图，分辨率影响待验证。',
              '不接入 find_missed 工具。没有真实统计与 ask 证据时，不宣称有效或无效。','']
    return '\n'.join(lines)


def write_report(run,work,dest,report):
    write_json(dest,report)
    write_text(dest.with_suffix('.md'),markdown(report))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['inventory','fixed','baseline','generate','statistics','estimate','ask','report'])
    p.add_argument('--run',type=Path,required=True,help='Read-only existing experiment root R')
    p.add_argument('--work',type=Path,default=STAGING/'artifacts')
    p.add_argument('--report',type=Path,default=STAGING/'results/missed_sources.json')
    p.add_argument('--group',choices=GROUPS)
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--shards',type=int,default=1)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--whu-fixed',action='store_true')
    p.add_argument('--pool',help='One source, or S1_02+Sx; from cost estimate')
    p.add_argument('--concurrency',type=int,default=1)
    p.add_argument('--max-cost-cny',type=float)
    p.add_argument('--request-cost-cny',type=float,default=.02,help='Hard per-request cost reservation; conservative bound, not average')
    p.add_argument('--clear-stop',action='store_true',help='Explicitly acknowledge prior STOP and resume')
    p.add_argument('--vision-config',type=Path)
    p.add_argument('--sam3-python',type=Path)
    p.add_argument('--sam3-root',type=Path)
    p.add_argument('--sam3-checkpoint',type=Path)
    p.add_argument('--sam3-bpe',type=Path)
    a=p.parse_args()
    run=a.run.resolve(); work=output_path(a.work); dest=output_path(a.report)
    chosen=[s for s in specs(run,work) if not a.group or s['group']==a.group]
    if a.command=='inventory':
        write_report(run,work,dest,inventory(run,work)); return
    if a.command in ('fixed','generate'):
        from .missed_sources_gpu import run_gpu
        run_gpu(a,run,work,chosen); return
    if a.command=='baseline':
        for spec in chosen:
            build_baseline(run,work,spec,a.shard,a.shards,a.resume)
        return
    report=load_json(dest) if exists(dest) else inventory(run,work)
    if a.command=='statistics':
        for spec in chosen:
            report['groups'][spec['group']]=statistics_group(run,work,spec,a.resume)
            report.setdefault('executed',{})['gpu']=True
            report['executed']['slurm']=bool(os.environ.get('SLURM_JOB_ID'))
            report['status']='开发来源统计已生成；验证器结果逐池更新'
            write_report(run,work,dest,report)
        report['cost_estimates']=estimate(run,work,report)
    elif a.command=='estimate':
        key='whu_fixed_cost_estimates' if a.whu_fixed else 'cost_estimates'
        report[key]=estimate(run,work,report,a.whu_fixed)
    elif a.command=='ask':
        if not a.group or not a.max_cost_cny:
            p.error('ask requires --group and --max-cost-cny')
        from .missed_sources_ask import run_ask_cli
        result=run_ask_cli(a,run,work,chosen[0],report)
        report.setdefault('executed',{})['api']=report.get('executed',{}).get('api',False) or result['answers']>0
        report['executed']['slurm']=bool(os.environ.get('SLURM_JOB_ID'))
    if a.command in ('report','ask'):
        update_rules(run,work,report)
    write_report(run,work,dest,report)


if __name__=='__main__':
    main()
