"""Offline error taxonomy. No model, API, scheduler or GPU imports.

Run from code/: python -m agent_baseline.analysis.error_taxonomy --help
Conventions, commands and limitations are maintained in workspace NOTES.md.
"""
import os
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_var] = '1'
import argparse
from collections import Counter, defaultdict
import io
import json
from pathlib import Path
import random
import re
import time
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from agent_baseline.analysis import gap_attribution as gap

ROOT = Path(__file__).resolve().parents[3]
R_DEFAULT = ROOT.parent / 'agent_baseline_run'
RETRIES = Counter()
STRUCTURE = np.ones((3, 3), dtype=np.uint8)
FP_KEYS = ('粘连误检', '整目标误检')
FN_KEYS = ('整目标漏检', '部分漏检或边界', '小目标漏检', '大目标漏检')
TRACE_KEYS = ('兜底提交', '大框失败', '验证器误删 TP', '规划器补进 FP')
DEFINITIONS = {
    '粘连误检': 'FP 所在预测 8 连通域含至少一个 TP；不等同于边界误差。',
    '整目标误检': 'FP 所在预测 8 连通域不含 TP；与粘连误检互斥并覆盖全部 FP。',
    '方向错': 'C1 的 FP 与反方向真值相交；附加标签，分母全部 FP。',
    '原址重建': 'SECOND 的 FP 中，slabel1=5 且 slabel2=5；0 为未变化无地类，两期非零即有变化标注；附加标签。',
    '整目标漏检': '真值 8 连通域没有任何预测像素，该域全部 FN。',
    '部分漏检或边界': '真值 8 连通域含 TP，其中未覆盖像素；与整目标漏检互斥并覆盖全部 FN。',
    '小目标漏检': '面积严格小于 64 的真值 8 连通域，且整体漏检；附加标签。',
    '大目标漏检': '面积严格大于图面积 10% 的真值 8 连通域中所有 FN；附加标签。',
    '兜底提交': 'submitted_by != model；损失为该会话最终全部 FP/FN 的理想修复。另计当前句柄不等于最新工具生成 mask 的兜底。',
    '大框失败': 'segment_prompt 纯框面积严格超过图面积 10%，且被 box_too_large 拦截或返回掩码 FP>TP。另列全部大框暴露、拦截和返回次数；损失只修复失败大框后代最终保留的 FP。',
    '验证器误删 TP': '先看到 no，后有成功编辑把该轮廓完整删除；轮廓 TP 占比严格超过 50%。损失只恢复最终仍缺失的 TP；同一像素只计一次。',
    '规划器补进 FP': '最终预测减去固定路线和给定掩码，按 8 连通域分组，FP 严格多于 TP 的残差组件；只修复其中 FP。该口径包含重新分割扩出的残差。',
    '主路线选错': 'T1.2 无成功 segment_semantic，或最终非空像素没有语义工具的保留祖先；T1.3 无成功整图 segment_everything + compare_features，或最终没有来自同一提议链的保留祖先。分开报告未调用、调用失败、调用后未使用。损失为这些会话最终全部 FP/FN 的理想修复。',
}


def retry_io(call):
    """Initial attempt plus up to three OSError retries; count every retry."""
    for attempt in range(4):
        try:
            return call()
        except OSError:
            if attempt == 3:
                raise
            RETRIES['io_retries'] += 1
            time.sleep(.1 * (attempt + 1))


def read_json(path):
    return json.loads(retry_io(lambda: Path(path).read_text()))


def read_jsonl(path):
    return [json.loads(s) for s in retry_io(lambda: Path(path).read_text()).splitlines() if s.strip()]


def read_image(path, rgb=False):
    def read():
        with Image.open(path) as im:
            return np.array(im.convert('RGB') if rgb else im)
    return retry_io(read)


def mask(path):
    a = read_image(path)
    if a.ndim != 2:
        raise ValueError('Expected two-dimensional mask')
    return a > 0


def atomic_bytes(path, data):
    path = Path(path)
    def write():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + '.tmp')
        tmp.write_bytes(data)
        tmp.replace(path)
    retry_io(write)


def write_json(path, data):
    atomic_bytes(path, (json.dumps(data, ensure_ascii=False, indent=2) + '\n').encode())


def relative(path, root):
    # os.path.relpath preserves ../../ references for datasets outside R.
    return os.path.relpath(path, root)


def resolve(directory, path):
    p = Path(path)
    return p if p.is_absolute() else directory / p


def counts(pred, gt):
    return [int((pred & gt).sum()), int((pred & ~gt).sum()), int((~pred & gt).sum())]


def f1(c):
    tp, fp, fn = c
    return 200. * tp / max(1, 2*tp + fp + fn)


def pixel_taxonomy(pred, gt, opposite=None, rebuilt=None):
    """8-connected primary partitions plus independent overlapping tags.

    FP primary partition uses prediction components with/without TP.
    FN primary partition uses truth components with/without TP. Small means
    area <64 AND no TP; large means area >10% of image (partial or complete).
    """
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    if pred.shape != gt.shape or pred.ndim != 2:
        raise ValueError('Prediction/truth shape mismatch')
    fp, fn = pred & ~gt, gt & ~pred
    pl, _ = ndi.label(pred, STRUCTURE)
    gl, _ = ndi.label(gt, STRUCTURE)
    tp_pred = np.unique(pl[pred & gt])
    tp_gt = np.unique(gl[pred & gt])
    attached = fp & np.isin(pl, tp_pred)
    partial = fn & np.isin(gl, tp_gt)
    areas = np.bincount(gl.ravel())
    small_ids = np.flatnonzero((areas < 64) & (np.arange(len(areas)) > 0))
    large_ids = np.flatnonzero((areas > gt.size * .1) & (np.arange(len(areas)) > 0))
    result = {'粘连误检': attached, '整目标误检': fp & ~attached,
              '整目标漏检': fn & ~partial, '部分漏检或边界': partial,
              '小目标漏检': fn & ~partial & np.isin(gl, small_ids),
              '大目标漏检': fn & np.isin(gl, large_ids)}
    for key, label in (('方向错', opposite), ('原址重建', rebuilt)):
        if label is not None:
            if label.shape != gt.shape:
                raise ValueError('Additional label shape mismatch')
            result[key] = fp & label
    assert sum(int(result[k].sum()) for k in FP_KEYS) == int(fp.sum())
    assert sum(int(result[k].sum()) for k in FN_KEYS[:2]) == int(fn.sum())
    return result


class Archive:
    """Reuse gap_attribution's component-ID preservation and packed archives.

    Trace only loads local arrays. Retry all its complete read/decode operations
    through this module so retry counts include those reads as well.
    """
    def __init__(self, directory, shape, steps):
        self.directory, self.shape, self.steps = Path(directory), shape, steps
        self.calls, self.producer, self.assets, self.labels_cache = [], {}, {}, {}
        self._dense = {}
        for s in steps:
            for c in s.get('calls', []):
                c = dict(c, step=s['step'], ordinal=len(self.calls))
                self.calls.append(c)
                for h in c.get('handles_created') or []:
                    self.producer[h] = c['ordinal']

    def asset(self, h):
        if h not in self.assets:
            if h.startswith(('inst#', 'props#')):
                p = self.directory / 'insts' / ('inst_' + h.split('#')[1] + '.npz')
                def read():
                    with np.load(io.BytesIO(p.read_bytes()), allow_pickle=False) as z:
                        return {k: z[k] for k in z.files}
                self.assets[h] = retry_io(read)
            else:
                self.assets[h] = mask(self.directory / 'masks' / (h.replace('#', '_') + '.png'))
        return self.assets[h]

    def dense(self, h):
        if h not in self._dense:
            a = self.asset(h)
            self._dense[h] = (np.unpackbits(np.bitwise_or.reduce(a['masks'], axis=0),
                count=int(np.prod(self.shape))).reshape(self.shape).astype(bool)
                if isinstance(a, dict) and len(a['masks']) else
                np.zeros(self.shape, bool) if isinstance(a, dict) else a)
        return self._dense[h]

    def component_labels(self, h):
        if h not in self.labels_cache:
            self.labels_cache[h] = gap.Trace.component_labels(self, h)
        return self.labels_cache[h]

    def parents(self, h):
        ordinal = self.producer.get(h)
        if ordinal is None:
            return []
        return gap.Trace.parents(self, h, self.calls[ordinal])

    def final_support(self, h, targets):
        """Propagate target lineage only through retained pixel intersections.

        Subtraction uses only its first operand, and add_region honors ids.
        This measures retained lineage, not a causal contribution of the tool.
        """
        cache = {}
        def visit(k):
            if k in cache:
                return cache[k]
            present = self.dense(k)
            if k in targets:
                out = present.copy()
            else:
                out = np.zeros(self.shape, bool)
                for p in self.parents(k):
                    part = visit(p)
                    c = self.calls[self.producer[k]]
                    if c['tool'] == 'add_region' and p == c['args']['other'] and c['args'].get('ids') is not None:
                        ids = c['args']['ids']
                        a = self.asset(p)
                        if isinstance(a, dict):
                            bits = a['masks'][np.isin(a['ids'], ids)]
                            selected = (np.unpackbits(np.bitwise_or.reduce(bits, axis=0), count=present.size)
                                        .reshape(self.shape).astype(bool)) if len(bits) else np.zeros_like(present)
                        else:
                            selected = np.isin(self.component_labels(p), ids)
                        part = part & selected
                    out |= present & part
            cache[k] = out
            return out
        return visit(h)


def report_answers(text):
    """Recover no/yes outline IDs actually delivered in stage reports."""
    m = re.search(r'Verifier answers on (mask#\d+):', text)
    if not m:
        return []
    answers = []
    for answer in ('yes', 'no'):
        ids = re.search(r'IDs answered ' + answer + r': (\[[\d, ]*\])', text)
        if ids:
            answers.extend(dict(mask=m[1], id=i, answer=answer) for i in json.loads(ids[1]))
    return answers


def trajectory_taxonomy(directory, meta, final, pred, gt, fixed=None, route=None):
    steps = read_jsonl(Path(directory) / 'steps.jsonl')
    arc = Archive(directory, gt.shape, steps)
    empty = np.zeros_like(gt)
    # Each class stores final error pixels to repair; overlaps are intentional.
    repair = {k: empty.copy() for k in (('主路线选错',) if route else TRACE_KEYS)}
    flags, detail = {}, dict(bigbox_calls=0, bigbox_blocked=0, bigbox_returned=0,
                            bigbox_fp_majority=0, verifier_removed_components=0)
    if route:
        required = ('segment_semantic',) if route == 'T1.2' else ('segment_everything', 'compare_features')
        called = {c['tool'] for c in arc.calls}
        successful = {c['tool'] for c in arc.calls if c.get('ok')}
        targets = set()
        for c in arc.calls:
            if not c.get('ok'):
                continue
            if route == 'T1.2' and c['tool'] == 'segment_semantic':
                targets.update(c.get('handles_created') or [])
            elif route == 'T1.3' and c['tool'] == 'compare_features':
                # The proposal ancestry must contain a whole-image proposal call.
                todo = [c['args']['props']]
                seen, whole = set(), False
                while todo:
                    h = todo.pop()
                    if h in seen:
                        continue
                    seen.add(h)
                    pc = arc.calls[arc.producer[h]] if h in arc.producer else None
                    if pc and pc['tool'] == 'segment_everything' and not pc['args'].get('region'):
                        whole = True
                    todo.extend(arc.parents(h))
                if whole:
                    targets.update(c.get('handles_created') or [])
        used = int(arc.final_support(final['final_handle'], targets).sum()) if targets and pred.any() else 0
        wrong = not all(k in successful for k in required) or used == 0
        flags['主路线选错'] = wrong
        if wrong:
            repair['主路线选错'] = pred ^ gt
        detail.update(required_tools=list(required), tools_called=sorted(called & set(required)),
                      tools_successful=sorted(successful & set(required)), retained_main_pixels=used,
                      route_reason='未调用' if not all(k in called for k in required) else
                      '调用失败' if not all(k in successful for k in required) else
                      '调用后未使用' if wrong else '使用主路线')
        return repair, flags, detail

    latest, current = None, None
    initial = 'mask#0' if meta.get('init_mask_path') else None
    big_targets = set()
    # Answers are tracked by exact outline across handles and component renumbering.
    active_answers = {}
    removed = empty.copy()

    def receive(rows):
        for a in rows:
            if a.get('answer') not in ('yes', 'no') or not a.get('mask') or a.get('id') is None:
                continue
            h, cid = a['mask'], a['id']
            labels = arc.component_labels(h)
            region = labels == cid
            area = int(region.sum())
            if area and int((region & gt).sum()) > area / 2:
                # Exact outline identity, shared across renumbered handles. A later
                # yes supersedes an earlier no for the identical outline.
                active_answers[np.flatnonzero(region).tobytes()] = (a['answer'], region)

    for s in steps:
        if s.get('report'):
            receive(report_answers(s['report']))
        for c in s.get('calls', []):
            a = c.get('args') or {}
            outputs = c.get('handles_created') or []
            for h in outputs:
                if h.startswith('mask#') and h != initial:
                    latest = h
            if c['tool'] == 'segment_prompt' and a.get('box') is not None and not a.get('points'):
                x0, y0, x1, y1 = a['box']
                area = max(0, np.ceil(x1)-np.floor(x0))*max(0, np.ceil(y1)-np.floor(y0))
                if area > gt.size * .1:
                    detail['bigbox_calls'] += 1
                    detail['bigbox_blocked'] += c.get('error_code') == 'box_too_large'
                    if c.get('ok'):
                        detail['bigbox_returned'] += 1
                        for h in outputs:
                            if h.startswith('mask#'):
                                cc = counts(arc.dense(h), gt)
                                detail['bigbox_fp_majority'] += cc[1] > cc[0]
                                if cc[1] > cc[0]:
                                    big_targets.add(h)
            # A successful mask edit after an answer must remove the entire outline.
            if c.get('ok') and c['tool'] not in ('segment_prompt', 'segment_semantic'):
                for h in outputs:
                    if not h.startswith('mask#'):
                        continue
                    after = arc.dense(h)
                    parents = arc.parents(h)
                    for _, (answer, region) in active_answers.items():
                        if answer == 'no' and not (after & region).any() and any(np.all(arc.dense(p)[region]) for p in parents):
                            newly = region & ~removed
                            if newly.any():
                                detail['verifier_removed_components'] += 1
                            removed |= region
            text = c.get('result_text') or '{}'
            try:
                public = json.loads(text)
            except json.JSONDecodeError:
                public = {}
            receive(public.get('results', []))
            receive(report_answers(public.get('report', '')))
            if 'current_after' in c:
                current = c['current_after']
        if 'current_mask' in s:
            current = s['current_mask']
    flags['兜底提交'] = final.get('submitted_by') != 'model'
    detail.update(fallback_current=current, fallback_latest=latest,
                  fallback_current_differs_latest=bool(flags['兜底提交'] and latest is not None and current != latest))
    if flags['兜底提交']:
        repair['兜底提交'] = pred ^ gt
    flags['大框失败'] = detail['bigbox_blocked'] > 0 or detail['bigbox_fp_majority'] > 0
    if big_targets:
        repair['大框失败'] = arc.final_support(final['final_handle'], big_targets) & pred & ~gt
    repair['验证器误删 TP'] = removed & gt & ~pred
    flags['验证器误删 TP'] = detail['verifier_removed_components'] > 0
    baseline = fixed.copy() if fixed is not None else empty.copy()
    if initial:
        baseline |= arc.dense(initial)
    novel = pred & ~baseline
    labels, n = ndi.label(novel, STRUCTURE)
    area = np.bincount(labels.ravel(), minlength=n+1)
    tp = np.bincount(labels[gt], minlength=n+1)
    bad_ids = np.flatnonzero((area > 2*tp) & (np.arange(n+1) > 0))
    repair['规划器补进 FP'] = np.isin(labels, bad_ids) & ~gt
    flags['规划器补进 FP'] = bool(len(bad_ids))
    detail['added_fp_majority_components'] = len(bad_ids)
    return repair, flags, detail


def groups(root):
    out = [dict(dataset='LEVIR_C0', task='T1.1xC0', direction='any',
                prefix='json_qwen37flash_g3', fixed='init/sam3_imc_p85/val', pixel=True)]
    for dataset in ('SECOND', 'WHU'):
        for direction, suffix in (('any', 'C0'), ('appeared', '新建'), ('disappeared', '拆除')):
            out.append(dict(dataset=dataset+'_'+suffix, task='T1.1xC0' if direction == 'any' else 'T1.1xC1',
                direction=direction, prefix=dataset.lower() + ('_g2' if dataset == 'SECOND' and direction == 'any' else '_g3')+'_'+direction,
                fixed=dataset.lower()+'/fixed/'+direction, pixel=True))
    for route in ('T1.2', 'T1.3'):
        for generation in ('g3', 'g4'):
            out.append(dict(dataset='SECOND_'+route, task=route+'xC0' if route == 'T1.2' else route,
                direction='any', prefix='json_qwen37flash_'+generation+'_'+route.replace('.', '').lower()+'_any',
                route=route, method='Agent_'+generation.upper(), pixel=False))
    for g in out:
        g['runs'] = [g['prefix']+f'_s{s}/'+g['task']+'/val' for s in (1, 2, 3)]
        g.setdefault('method', 'Agent_G2' if g['prefix'].startswith('second_g2') else 'Agent_G3')
    return out


def truth(meta, directory):
    p = meta.get('gt_mask_path')
    if p:
        return mask(resolve(directory, p)), resolve(directory, p)
    source = resolve(directory, meta['source_images'][0])
    p = source.parent.parent / 'label' / (directory.name+'.png')
    return mask(p), p


def add_record(bucket, group, method, name, directory, meta, final, pred, gt, gt_path,
               prediction_path, categories, flags=None, details=None, root=None):
    c = counts(pred, gt)
    row = dict(image=name, session=relative(directory, root) if final else None,
               run=relative(directory.parents[2], root) if final else None,
               seed=directory.parents[2].name.rsplit('_s', 1)[-1] if final else None,
               prediction=relative(prediction_path, root) if prediction_path else None,
               gt=relative(gt_path, root), source_images=[relative(resolve(directory, p), root) for p in meta['source_images']],
               counts=c, categories={}, details=details or {})
    for key, pixels in categories.items():
        n = int(pixels.sum())
        fp = int((pixels & pred & ~gt).sum())
        fn = int((pixels & gt & ~pred).sum())
        repaired = [c[0]+fn, c[1]-fp, c[2]-fn]
        row['categories'][key] = dict(pixels=n, fp=fp, fn=fn,
            affected=bool(flags.get(key)) if flags is not None else n > 0,
            repaired_counts=repaired, f1_repair_pp=f1(repaired)-f1(c),
            denominator='FP' if key in FP_KEYS+('方向错', '原址重建') else 'FN' if key in FN_KEYS else 'FP+FN')
    bucket.setdefault(group['dataset'], {}).setdefault(method, []).append(row)


def summarize(records):
    output = {}
    for dataset, methods in records.items():
        output[dataset] = {}
        for method, rows in methods.items():
            total = np.sum([r['counts'] for r in rows], axis=0).tolist()
            categories = {}
            keys = list(dict.fromkeys(k for r in rows for k in r['categories']))
            for key in keys:
                rr = [r for r in rows if key in r['categories']]
                cat = [r['categories'][key] for r in rr]
                cc = np.sum([r['counts'] for r in rr], axis=0).tolist()
                corrected = np.sum([r['categories'][key]['repaired_counts'] for r in rr], axis=0).tolist()
                den = cat[0]['denominator']
                denom = cc[1] if den == 'FP' else cc[2] if den == 'FN' else cc[1]+cc[2]
                pixels = sum(x['pixels'] for x in cat)
                categories[key] = dict(pixels=pixels, pixel_share=pixels/denom if denom else 0.,
                    denominator=den, denominator_pixels=denom,
                    images=len({r['image'] for r in rr if r['categories'][key]['affected']}),
                    sessions=sum(x['affected'] for x in cat), error_sessions=sum(x['pixels'] > 0 for x in cat),
                    tp_recovered=sum(x['fn'] for x in cat), fp_removed=sum(x['fp'] for x in cat),
                    f1_repair_pp=f1(corrected)-f1(cc),
                    affected_session_mean_f1_repair_pp=float(np.mean([x['f1_repair_pp'] for x in cat if x['affected']])) if any(x['affected'] for x in cat) else 0.,
                    examples=[])
            details = dict(fallback_current_differs_latest=sum(r['details'].get('fallback_current_differs_latest', False) for r in rows))
            for key in ('bigbox_calls', 'bigbox_blocked', 'bigbox_returned', 'bigbox_fp_majority', 'verifier_removed_components', 'added_fp_majority_components'):
                details[key] = sum(r['details'].get(key, 0) for r in rows)
            details['route_reasons'] = dict(Counter(r['details']['route_reason'] for r in rows if 'route_reason' in r['details']))
            output[dataset][method] = dict(images=len({r['image'] for r in rows}), sessions=len(rows),
                counts=dict(zip(('tp', 'fp', 'fn'), total)), pooled_f1=f1(total),
                per_seed_sessions=dict(Counter(r['seed'] for r in rows if r['seed'])),
                categories=categories, diagnostics=details)
    return output


def pick_examples(rows, key):
    candidates = [r for r in rows if key in r['categories'] and r['categories'][key]['affected']]
    if not candidates:
        return []
    candidates.sort(key=lambda r: (-r['categories'][key]['pixels'], r['image'], r['session'] or ''))
    # Three distinct images where possible; seeds never overwrite one another.
    first = candidates[0]
    remaining = [r for r in candidates[1:] if r['image'] != first['image']]
    by_image = {}
    for row in remaining:
        by_image.setdefault(row['image'], []).append(row)
    rng = random.Random(1016)
    image_names = sorted(by_image)
    chosen = rng.sample(image_names, min(2, len(image_names)))
    return [first] + [rng.choice(by_image[n]) for n in chosen]


def jpeg_bytes(im, quality=45):
    im.thumbnail((384, 384), Image.Resampling.LANCZOS)
    b = io.BytesIO()
    im.convert('RGB').save(b, format='JPEG', quality=quality, optimize=True)
    return b.getvalue()


def export_examples(records, summary, output, root):
    byte_count, n_examples = 0, 0
    for dataset, methods in records.items():
        for method, rows in methods.items():
            for key, category in summary[dataset][method]['categories'].items():
                selected = pick_examples(rows, key)
                for index, row in enumerate(selected):
                    sub = Path('error_examples')/dataset/method/key/row['image']
                    images = [read_image(root/p, rgb=True) for p in row['source_images']]
                    gt = mask(root/row['gt'])
                    pred = mask(root/row['prediction']) if row['prediction'] else np.zeros_like(gt)
                    overlay = (images[1].astype(float)*.5).astype(np.uint8)
                    overlay[pred & gt] = (0, 255, 0)
                    overlay[pred & ~gt] = (255, 0, 0)
                    overlay[~pred & gt] = (0, 0, 255)
                    arrays = dict(t1=images[0], t2=images[1], gt=(gt.astype(np.uint8)*255), overlay=overlay)
                    files = {}
                    for label, array in arrays.items():
                        p = output/sub/(label+'.jpg')
                        data = jpeg_bytes(Image.fromarray(array))
                        byte_count += len(data)
                        atomic_bytes(p, data)
                        files[label] = str(sub/(label+'.jpg'))
                    info = dict(image=row['image'], category=key, loss_pixels=row['categories'][key]['pixels'],
                        f1_repair_pp=row['categories'][key]['f1_repair_pp'], run=row['run'], session=row['session'],
                        seed=row['seed'], prediction=row['prediction'], gt=row['gt'], source_images=row['source_images'],
                        selection='最大损失像素' if index == 0 else '种子 1016 随机', files=files)
                    encoded = (json.dumps(info, ensure_ascii=False, indent=2)+'\n').encode()
                    byte_count += len(encoded)
                    atomic_bytes(output/sub/'example.json', encoded)
                    category['examples'].append(dict(image=row['image'], **files,
                        metadata=str(sub/'example.json'), session=row['session']))
                    n_examples += 1
                category['example_shortfall'] = max(0, 3-len(selected))
    if byte_count > 30_000_000:
        raise ValueError('Example assets exceed 30 MB; reduce JPEG quality')
    return dict(examples=n_examples, bytes=byte_count, limit_bytes=30_000_000)


def markdown(report):
    lines = ['# 典型错误统计', '',
        'Agent 合并三个固定种子，固定路线只计一次。像素占比先累计像素再相除。图数按图名去重，会话数保留种子。',
        'SECOND C0 只有 G2；其余建筑任务用 G3。T1.2/T1.3 的 G3/G4 只统计主路线，不加入建筑像素表。',
        'F1 单位为百分点，按全部会话累计 TP/FP/FN。损失 = 仅理想修复该类最终错误像素后的 F1 − 实际 F1。各类重叠，不可相加，也不能解释为改 harness 后的预期收益。',
        '零分母占比记 0。附加标签不参与主类加和。零损失的大框/兜底样例仅代表触发事件。', '']
    for key, text in DEFINITIONS.items():
        lines.append(f'- **{key}**：{text}')
    lines += ['', '| 数据集 | 方法 | 图 / 会话 | F1 | 类 | 像素 | 占全部 FP/FN | 涉及图 / 会话 | 理想修复 ΔF1 |',
              '|---|---|---:|---:|---|---:|---:|---:|---:|']
    for dataset, methods in report['summary'].items():
        for method, s in methods.items():
            for key, c in s['categories'].items():
                lines.append(f"| {dataset} | {method} | {s['images']} / {s['sessions']} | {s['pooled_f1']:.2f} | {key} | {c['pixels']:,} | {100*c['pixel_share']:.2f}% ({c['denominator']}) | {c['images']} / {c['sessions']} | {c['f1_repair_pp']:.3f} |")
    lines += ['', '## 轨迹补充', '', '| 数据集 | 方法 | 兜底当前≠最新 | 大框调用 / 拦截 / 返回 / FP为主 | 误删组件 | 主路线结果 |',
              '|---|---|---:|---|---:|---|']
    for dataset, methods in report['summary'].items():
        for method, s in methods.items():
            if method == '固定路线':
                continue
            d = s['diagnostics']
            lines.append(f"| {dataset} | {method} | {d['fallback_current_differs_latest']} | {d['bigbox_calls']} / {d['bigbox_blocked']} / {d['bigbox_returned']} / {d['bigbox_fp_majority']} | {d['verifier_removed_components']} | {d['route_reasons']} |")
    lines += ['', '## r4 对照', '', report['r4_comparison']['explanation'], '', '## Agent 的典型错误', '']
    # Conclusions are generated from measured error pixels, not hand-picked examples.
    for dataset, methods in report['summary'].items():
        for method, s in methods.items():
            if method == '固定路线':
                continue
            trace = [(k, c) for k, c in s['categories'].items() if k in TRACE_KEYS+('主路线选错',)]
            if trace and any(c['pixels'] for _, c in trace):
                k, c = max(trace, key=lambda x: x[1]['f1_repair_pp'])
                lines.append(f"- {dataset} {method}：轨迹标签中理想修复差最大为「{k}」，{c['sessions']}/{s['sessions']} 会话，{c['f1_repair_pp']:.2f} 分。优先回看对应样例和触发链；不作因果归因。")
            elif trace:
                lines.append(f"- {dataset} {method}：当前覆盖内未观察到这几类轨迹标签的剩余错误像素。")
    lines += ['', '## 固定路线的典型错误', '']
    for dataset, methods in report['summary'].items():
        if '固定路线' not in methods:
            continue
        s = methods['固定路线']
        cc = s['categories']
        lines.append(f"- {dataset}：粘连占 FP {100*cc['粘连误检']['pixel_share']:.1f}%，整目标漏检占 FN {100*cc['整目标漏检']['pixel_share']:.1f}%。前者需区分真实边界偏差和相连伪变化；后者是新增候选的机会。")
    lines += ['', '## 覆盖与限制', '',
        f"状态：{report['status']}。耗时 {report['elapsed_seconds']:.1f} 秒；I/O 重试 {report['io_retries']} 次。",
        '主路线的使用按保留像素祖先判断；即使只保留少量主路线像素，也不归为完全未使用。验证器只计已送给规划器的 no，后续成功编辑完整移除的轮廓；部分削减和未显式形成编辑链的重建不计，可能低估。',
        '大框拦截后可能通过其他路线补救；该类零损失不能说明阈值没有代价。兜底和主路线标签的全部错误修复只描述对应会话错误量，不能归因于停止机制或工具选择。',
        '样例每类优先最大损失像素，再按种子 1016 抽两张不同图；不足三张保留全部，并在 JSON 标出缺额。原图只在选样后读取。']
    if report['missing']:
        lines.extend('- 缺失：'+m for m in report['missing'])
    return '\n'.join(lines)+'\n'


def merge_reports(paths, root, output, no_examples=False):
    """Merge disjoint shards, recompute ratios/F1 and select global examples."""
    parts = [read_json(path) for path in paths]
    records, coverage, missing = {}, [], []
    identities = set()
    for part in parts:
        coverage.extend(part['coverage'])
        missing.extend(part['missing'])
        for dataset, methods in part['records'].items():
            for method, rows in methods.items():
                for row in rows:
                    identity = (dataset, method, row['session'] or row['image'])
                    if identity in identities:
                        raise ValueError('Overlapping shard records: '+str(identity))
                    identities.add(identity)
                    records.setdefault(dataset, {}).setdefault(method, []).append(row)
    for methods in records.values():
        for rows in methods.values():
            rows.sort(key=lambda r: (r['image'], r['seed'] or ''))
    summary = summarize(records)
    old_fp = sum(p['r4_comparison']['older_fp'] for p in parts)
    old_attached = sum(p['r4_comparison']['older_attached_fp'] for p in parts)
    current = summary.get('LEVIR_C0', {}).get('固定路线', {}).get('categories', {}).get('粘连误检', {}).get('pixel_share')
    comparison = dict(reference_percent=54, current_pixel_threshold=.85, older_pixel_threshold=.5,
        older_fp=old_fp, older_attached_fp=old_attached, older_share=old_attached/old_fp if old_fp else None,
        current_share=current, missing=[m for p in parts for m in p['r4_comparison']['missing']])
    comparison['explanation'] = (f"r4 是像素门限 0.5→0.85 的调整轮。并行复算旧门限 {100*old_attached/max(1,old_fp):.2f}%，现用门限 {100*(current or 0):.2f}%。54% 是历史整数表述；检查完整 64 张及两种门限，避免将历史动机误归为现用掩码结果。")
    if len(records.get('LEVIR_C0', {}).get('固定路线', [])) == 64:
        comparison['older_matches_rounded_54'] = round(100*old_attached/max(1,old_fp)) == 54
        comparison['current_matches_rounded_54'] = round(100*(current or 0)) == 54
        comparison['explanation'] += (f" 旧门限是否复现 54%：{comparison['older_matches_rounded_54']}；现用门限是否复现：{comparison['current_matches_rounded_54']}。两者都不复现时，历史实例/组件口径可能不同，未找到原始计数，原因不能确定。")
    complete = not missing
    for group in groups(root):
        expected = 64 if group['dataset'] == 'LEVIR_C0' else 128
        rr = records.get(group['dataset'], {}).get(group['method'], [])
        complete &= len(rr) == expected * 3 and len({r['image'] for r in rr}) == expected
        if group['pixel']:
            complete &= len(records.get(group['dataset'], {}).get('固定路线', [])) == expected
    report = dict(status='complete' if complete else 'merged_partial', conventions=DEFINITIONS,
        records=records, summary=summary, coverage=coverage, missing=missing, r4_comparison=comparison,
        elapsed_seconds=sum(p['elapsed_seconds'] for p in parts), io_retries=sum(p['io_retries'] for p in parts),
        merged_shards=len(parts))
    report['assets'] = export_examples(records, summary, output, root) if not no_examples else dict(examples=0, bytes=0)
    site = {d: {m: {k: dict(pixel_share=c['pixel_share'], images=c['images'], sessions=c['sessions'],
        examples=c['examples']) for k, c in s['categories'].items()} for m, s in methods.items()} for d, methods in summary.items()}
    write_json(output/'error_taxonomy.json', report)
    write_json(output/'error_taxonomy_site.json', site)
    atomic_bytes(output/'error_taxonomy.md', markdown(report).encode())
    print(json.dumps(dict(status=report['status'], shards=len(parts), assets=report['assets']), ensure_ascii=False), flush=True)


def finalize_progress(checkpoint, supplement, root, output):
    """Publish available observations; checkpoint wins over smoke duplicates.

    This is an explicit partial delivery, never presented as a full experiment.
    """
    checkpoint = read_json(checkpoint)
    small = read_json(supplement)
    records = small['records']
    for dataset, methods in checkpoint['records'].items():
        for method, rows in methods.items():
            existing = records.setdefault(dataset, {}).setdefault(method, [])
            incoming = {r['session'] or r['image']: r for r in rows}
            combined = {r['session'] or r['image']: r for r in existing}
            combined.update(incoming)
            records[dataset][method] = sorted(combined.values(), key=lambda r: (r['image'], r['seed'] or ''))
    # Reapply the tightened event definition to old smoke records.
    for methods in records.values():
        for rows in methods.values():
            for row in rows:
                if '大框失败' in row['categories']:
                    row['categories']['大框失败']['affected'] = bool(row['details']['bigbox_blocked'] or row['details']['bigbox_fp_majority'])
    summary = summarize(records)
    old_fp, old_attached = 0, 0
    old_missing = []
    for row in records['LEVIR_C0']['固定路线']:
        path = root/'init/sam3_imc/val'/(row['image']+'.png')
        try:
            gt, pred = mask(root/row['gt']), mask(path)
            old_fp += counts(pred, gt)[1]
            old_attached += int(pixel_taxonomy(pred, gt)['粘连误检'].sum())
        except OSError as exc:
            old_missing.append(relative(path, root)+' errno='+str(exc.errno))
    current = summary['LEVIR_C0']['固定路线']['categories']['粘连误检']['pixel_share']
    comparison = dict(reference_percent=54, current_pixel_threshold=.85, older_pixel_threshold=.5,
        older_fp=old_fp, older_attached_fp=old_attached, older_share=old_attached/old_fp if old_fp else None,
        current_share=current, missing=old_missing,
        older_matches_rounded_54=round(100*old_attached/max(1,old_fp)) == 54 if not old_missing else None,
        current_matches_rounded_54=round(100*current) == 54)
    comparison['explanation'] = (f"LEVIR 64 张固定路线：现用像素门限 0.85 的粘连 FP 占比 {100*current:.2f}%；旧门限 0.5 为 {100*old_attached/max(1,old_fp):.2f}%。README 的 r4 是 0.5→0.85 调整轮。旧门限按整数是否复现 54%：{comparison['older_matches_rounded_54']}；现用门限是否复现：{comparison['current_matches_rounded_54']}。若只有旧门限一致，说明 54% 来自调整前的诊断；若两者都不一致，历史组件/实例口径可能不同，未找到原始计数，不能确定原因。")
    report = dict(status='partial_full_run_with_smoke', conventions=DEFINITIONS, records=records,
        summary=summary, coverage=checkpoint['coverage'], missing=checkpoint['missing'],
        r4_comparison=comparison, elapsed_seconds=checkpoint['elapsed_seconds']+small['elapsed_seconds'],
        io_retries=checkpoint['io_retries']+small['io_retries'],
        limitations=['LEVIR 全量，其余组每组五张图；全量预估超过登录节点十分钟，交由 Claude 按 NOTES 的分片命令运行。',
                    '全量尝试在 SECOND C0 train_1150 种子 2 的 final.json 重试后仍 EIO；进度外未完成的工作未计为零错误。'])
    report['assets'] = export_examples(records, summary, output, root)
    site = {d: {m: {k: dict(pixel_share=c['pixel_share'], images=c['images'], sessions=c['sessions'],
        examples=c['examples']) for k, c in s['categories'].items()} for m, s in methods.items()} for d, methods in summary.items()}
    write_json(output/'error_taxonomy.json', report)
    write_json(output/'error_taxonomy_site.json', site)
    text = markdown(report)
    warning = ('**部分结果：LEVIR 为完整 64 张；其余每组仅 5 张。不得用子集占比替代全量结论。**\n\n'
               '全量尝试遇到持续 EIO，且按已测耗时预估超出登录节点十分钟；全量分片与合并命令见 NOTES.md。\n\n')
    atomic_bytes(output/'error_taxonomy.md', (warning+text).encode())
    print(json.dumps(dict(status=report['status'], assets=report['assets'], r4=comparison), ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-root', type=Path, default=R_DEFAULT)
    p.add_argument('--output', type=Path, default=ROOT/'results')
    p.add_argument('--names', help='Comma-separated image stems, or a text file')
    p.add_argument('--shard', help='Zero-based i/n, applied to each sorted image inventory')
    p.add_argument('--limit', type=int, help='First N images per group (smoke only)')
    p.add_argument('--groups', help='Comma-separated dataset keys')
    p.add_argument('--max-seconds', type=float, default=590)
    p.add_argument('--no-examples', action='store_true')
    p.add_argument('--merge', type=Path, nargs='+', help='Merge disjoint shard error_taxonomy.json files')
    p.add_argument('--resume', action='store_true', help='Resume this output directory progress.json')
    p.add_argument('--finalize-progress', type=Path, help='Publish available checkpoint plus five-image smoke')
    p.add_argument('--supplement', type=Path, help='Smoke error_taxonomy.json for partial publication')
    a = p.parse_args()
    root, output = a.run_root.absolute(), a.output.absolute()
    if not output.is_relative_to(ROOT):
        p.error('--output must be inside the task directory')
    if a.merge:
        merge_reports(a.merge, root, output, a.no_examples)
        return
    if a.finalize_progress:
        if not a.supplement:
            p.error('--finalize-progress requires --supplement')
        finalize_progress(a.finalize_progress, a.supplement, root, output)
        return
    started = time.monotonic()
    records, missing, coverage = {}, [], []
    previous_elapsed, previous_retries = 0., 0
    if a.resume:
        checkpoint = read_json(output/'progress.json')
        records, missing = checkpoint['records'], checkpoint['missing']
        previous_elapsed = checkpoint['elapsed_seconds']
        previous_retries = checkpoint['io_retries']

    def save_progress():
        write_json(output/'progress.json', dict(records=records, missing=missing,
            coverage=coverage, elapsed_seconds=previous_elapsed+time.monotonic()-started,
            io_retries=previous_retries+RETRIES['io_retries']))
    selected_names = None
    if a.names:
        path = Path(a.names)
        seq = retry_io(path.read_text).split() if retry_io(path.is_file) else a.names.split(',')
        selected_names = {Path(n).stem for n in seq}
    shard_i, shard_n = map(int, a.shard.split('/')) if a.shard else (0, 1)
    if not 0 <= shard_i < shard_n:
        p.error('--shard requires 0 <= i < n')
    spec = [g for g in groups(root) if not a.groups or g['dataset'] in a.groups.split(',')]
    # The source encoding is read once, not inferred from binary GT masks.
    second_dataset = Path('/guisongxia01/pangchao/haoxiangyu/rs_datasets/SECOND')
    encoding = read_json(second_dataset/'cls_gray.json')
    if encoding.get('building') != 5 or encoding.get('non-change') != 0:
        raise ValueError('SECOND encoding differs from checked 0/5 convention')
    timed_out = False
    for group in spec:
        run0 = root/group['runs'][0]
        try:
            names = sorted(x.name for x in retry_io(lambda: list(run0.iterdir())) if retry_io(x.is_dir))
        except FileNotFoundError:
            missing.append('运行目录 '+group['runs'][0])
            continue
        inventory = len(names)
        names = [n for i, n in enumerate(names) if i % shard_n == shard_i and (selected_names is None or n in selected_names)]
        if a.limit:
            names = names[:a.limit]
        done = 0
        for name in names:
            if time.monotonic()-started >= a.max_seconds:
                timed_out = True
                break
            d0 = run0/name
            existing = records.get(group['dataset'], {}).get(group['method'], [])
            if sum(r['image'] == name for r in existing) == 3:
                done += 1
                continue
            # An interrupted/missing image can leave partial in-memory records.
            # Replace those before retrying the image, preserving other groups.
            for method in (group['method'], '固定路线') if group['pixel'] else (group['method'],):
                if method in records.get(group['dataset'], {}):
                    records[group['dataset']][method] = [r for r in records[group['dataset']][method] if r['image'] != name]
            try:
                meta0 = read_json(d0/'input.json')
                gt, gt_path = truth(meta0, d0)
                fixed = mask(root/group['fixed']/(name+'.png')) if group['pixel'] else None
                opposite, rebuilt = None, None
                if group['pixel'] and group['direction'] != 'any':
                    opposite = mask(gt_path.parent.parent/('demolished' if group['direction'] == 'appeared' else 'new')/gt_path.name)
                if group['pixel'] and group['dataset'].startswith('SECOND'):
                    t1, t2 = [read_image(second_dataset/part/(name+'.png')) for part in ('slabel1', 'slabel2')]
                    rebuilt = (t1 == 5) & (t2 == 5)
                if group['pixel']:
                    add_record(records, group, '固定路线', name, d0, meta0, None, fixed, gt, gt_path,
                        root/group['fixed']/(name+'.png'), pixel_taxonomy(fixed, gt, opposite, rebuilt), root=root)
                for run in group['runs']:
                    d = root/run/name
                    meta, final = read_json(d/'input.json'), read_json(d/'final.json')
                    pred_path = resolve(d, final['mask_path']) if final.get('mask_path') else None
                    pred = mask(pred_path) if pred_path else np.zeros_like(gt)
                    if counts(pred, gt) != [final['eval'][k] for k in ('tp', 'fp', 'fn')]:
                        raise ValueError('Final metrics mismatch: '+relative(d, root))
                    categories = pixel_taxonomy(pred, gt, opposite, rebuilt) if group['pixel'] else {}
                    repairs, flags, details = trajectory_taxonomy(d, meta, final, pred, gt, fixed, group.get('route'))
                    pixel_flags = {k: bool(v.any()) for k, v in categories.items()}
                    categories.update(repairs)
                    add_record(records, group, group['method'], name, d, meta, final, pred, gt, gt_path,
                        pred_path, categories, dict(pixel_flags, **flags), details, root)
                done += 1
                if done % 5 == 0:
                    save_progress()
            except FileNotFoundError as exc:
                missing.append(relative(exc.filename, root))
        coverage.append(dict(dataset=group['dataset'], method=group['method'], inventory=inventory,
                             requested_images=len(names), completed_images=done, runs=group['runs']))
        save_progress()
        print(f"{group['dataset']} {group['method']}: {done}/{len(names)} images; elapsed {time.monotonic()-started:.1f}s", flush=True)
        if timed_out:
            break
    summary = summarize(records)
    # Recompute the older pixel-threshold route on exactly the selected LEVIR images.
    comparison = dict(reference_percent=54, current_pixel_threshold=.85, older_pixel_threshold=.5)
    levir_rows = records.get('LEVIR_C0', {}).get('固定路线', [])
    old_total, old_attached, old_missing = 0, 0, []
    for row in levir_rows:
        old_path = root/'init/sam3_imc/val'/(row['image']+'.png')
        try:
            gt = mask(root/row['gt'])
            old = mask(old_path)
            cc = pixel_taxonomy(old, gt)
            old_total += counts(old, gt)[1]
            old_attached += int(cc['粘连误检'].sum())
        except FileNotFoundError:
            old_missing.append(relative(old_path, root))
    comparison.update(older_fp=old_total, older_attached_fp=old_attached,
        older_share=old_attached/old_total if old_total else None, missing=old_missing)
    current = summary.get('LEVIR_C0', {}).get('固定路线', {}).get('categories', {}).get('粘连误检', {}).get('pixel_share')
    comparison['current_share'] = current
    comparison['explanation'] = (f"README r4 行描述的是把像素门限 0.5 改为 0.85 的一轮；54% 作为改边界的动机，不能只凭该行断定统计掩码门限。这里同时复算旧路线 init/sam3_imc/val（0.5）和现用 init/sam3_imc_p85/val（0.85）。选中图上旧路线粘连 FP 占比 {100*old_attached/max(1,old_total):.2f}%；现用路线 {100*(current or 0):.2f}%。完整 64 张时再判断与四舍五入的 54% 是否一致；子集不能验证历史比例。")
    if len(levir_rows) == 64 and not old_missing:
        comparison['older_matches_rounded_54'] = round(100*old_attached/max(1,old_total)) == 54
        comparison['current_matches_rounded_54'] = round(100*(current or 0)) == 54
        comparison['explanation'] += (' 旧门限与 54% 的整数表述一致。' if comparison['older_matches_rounded_54'] else ' 旧门限也不复现 54%；历史统计的组件/实例口径可能不同，未找到原始计数证据，不能确定原因。')
        comparison['explanation'] += (' 现用门限也与 54% 一致。' if comparison['current_matches_rounded_54'] else ' 现用门限与 54% 不一致。')
    report = dict(status='time_limit_partial' if timed_out else 'subset' if a.names or a.limit or a.shard or a.groups else 'complete' if not missing else 'missing_inputs',
        conventions=DEFINITIONS, summary=summary, records=records, coverage=coverage, missing=missing,
        r4_comparison=comparison, io_retries=previous_retries+RETRIES['io_retries'], elapsed_seconds=previous_elapsed+time.monotonic()-started)
    report['assets'] = export_examples(records, summary, output, root) if not a.no_examples else dict(examples=0, bytes=0)
    report['elapsed_seconds'] = previous_elapsed+time.monotonic()-started
    report['io_retries'] = previous_retries+RETRIES['io_retries']
    site = {d: {m: {k: dict(pixel_share=c['pixel_share'], images=c['images'], sessions=c['sessions'],
              examples=c['examples']) for k, c in s['categories'].items()} for m, s in methods.items()} for d, methods in summary.items()}
    write_json(output/'error_taxonomy.json', report)
    write_json(output/'error_taxonomy_site.json', site)
    atomic_bytes(output/'error_taxonomy.md', markdown(report).encode())
    print(json.dumps(dict(status=report['status'], elapsed_seconds=report['elapsed_seconds'],
                         retries=report['io_retries'], assets=report['assets']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
