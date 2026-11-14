"""Offline T4.1 trace review; pooled pixel metrics, no model or API calls.

Paths supplied to the CLI are relative to --run-root unless absolute. Reports
store paths relative to that root. The T2 and morph flags are review candidates,
not ground-truth judgements of the planner's intent.
"""
import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import re
if __package__ in (None, ''):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import agent_baseline

import numpy as np
from common import components
from agent_baseline.analysis.io_retry import exists, read_image, read_text, retry_io, write_json, write_text
from agent_baseline.analysis.t3_metrics import counts, score


def resolve(path, root):
    path = Path(path)
    return path if path.is_absolute() else Path(root) / path


def relative(path, root):
    return os.path.relpath(path, root)


def read_rows(path):
    return [json.loads(line) for line in read_text(path).splitlines() if line.strip()]


def area_request(args):
    return (args.get('area_min') is not None or args.get('area_max') is not None
            or bool(re.search(r'\barea\b', args.get('where') or '')))


class Assets:
    """Lazy archived masks/packed instances. Reconstruct IDs for whole-region edits.

    PNGs do not retain component IDs. remove_region and mask select inherit their
    parent's labels; every other mask producer renumbers via common.components.
    """
    def __init__(self, directory, steps):
        self.directory = Path(directory)
        self.origins = {key: call for step in steps for call in step.get('calls', [])
                        for key in call.get('handles_created', [])}
        self.mask = lru_cache(maxsize=8)(self._mask)
        self.labels = lru_cache(maxsize=8)(self._labels)
        self.packed = lru_cache(maxsize=4)(self._packed)
        self.union = lru_cache(maxsize=4)(self._union)

    def _mask(self, handle):
        return read_image(self.directory / 'masks' / (handle.replace('#', '_') + '.png')) > 0

    def _labels(self, handle):
        call = self.origins.get(handle, {})
        args = call.get('args') or {}
        parent = (args.get('mask') if call.get('tool') == 'remove_region' else
                  args.get('set') if call.get('tool') == 'select' else None)
        if parent and parent.startswith(('mask#', 'region#', 'map#')):
            labels, rows = self.labels(parent)
            data = self.mask(handle)
            keep = set(int(i) for i in np.unique(labels[data]))
            return np.where(data, labels, 0), [r for r in rows if r['id'] in keep]
        return components(self.mask(handle))

    def _packed(self, handle):
        path = self.directory / 'insts' / (handle.replace('#', '_') + '.npz')
        def load():
            with np.load(path) as archive:
                return archive['masks'].copy(), tuple(int(v) for v in archive['shape'])
        return retry_io(load)

    def _union(self, handle):
        packed, shape = self.packed(handle)
        flat = np.bitwise_or.reduce(packed, axis=0) if len(packed) else np.zeros((np.prod(shape) + 7) // 8, np.uint8)
        return np.unpackbits(flat, count=int(np.prod(shape))).reshape(shape).astype(bool)


def t2_evidence(assets, t2, t1, answer):
    """Broad adoption candidate: >=80% T2 coverage and >=80% answer from T2.

    Requiring majority T1 overlap would miss train_293, where the T1 segmentation
    itself misses most buildings. Report that overlap independently for review.
    """
    union = assets.union(t2)
    area = int(union.sum())
    intersection = int((union & answer).sum())
    retained = intersection / max(1, area)
    share = intersection / max(1, int(answer.sum()))
    if not area or retained < .8 or share < .8:
        return None
    result = dict(t2_handle=t2, t2_area_px=area, t2_retained_fraction=retained, answer_from_t2_fraction=share,
                  t1_handle=t1, t1_overlap_fraction=None, t2_instances_overlapping_t1=None,
                  t2_instances=None, majority_t1_overlap=None)
    if t1:
        old = assets.union(t1)
        result['t1_overlap_fraction'] = int((union & old).sum()) / area
        packed, shape = assets.packed(t2)
        n = 0
        for row in packed:
            instance = np.unpackbits(row, count=int(np.prod(shape))).reshape(shape).astype(bool)
            n += bool(instance.any() and (instance & old).sum() / instance.sum() >= .25)
        result.update(t2_instances=len(packed), t2_instances_overlapping_t1=n,
                      majority_t1_overlap=n > len(packed) / 2)
    return result


def analyze_trace(directory, task, root):
    steps = read_rows(Path(directory) / 'steps.jsonl')
    assets = Assets(directory, steps)
    cutoff = task['area_min_m2'] / task['gsd_m'] ** 2
    events = dict(mask_select=[], instance_select=[], remove_small_after_report=[],
                  t2_as_change_candidates=[], morph_merges=[])
    saw_report = False
    t1, t2 = None, []
    flagged_t2 = set()
    for row in steps:
        # Harness-generated reports are separate rows; tool reports appear in result_text.
        if 'Area constraint (' in (row.get('report') or ''):
            saw_report = True
        for index, call in enumerate(row.get('calls', []), 1):
            tool, args = call.get('tool'), call.get('args') or {}
            event = dict(step=row['step'], call=index, tool=tool)
            ok = bool(call.get('ok'))
            if tool == 'select' and area_request(args):
                kind = ('mask_select' if str(args.get('set', '')).startswith('mask#') else
                        'instance_select' if str(args.get('set', '')).startswith(('inst#', 'props#')) else None)
                if kind:
                    events[kind].append(dict(event, args=args, ok=ok))
            if tool == 'remove_region' and saw_report and ok:
                _, regions = assets.labels(args['mask'])
                small = [r['id'] for r in regions if r['id'] in args['ids'] and r['area'] <= cutoff]
                if small:
                    events['remove_small_after_report'].append(dict(event, mask=args['mask'], ids=small,
                        removed_small_px=sum(r['area'] for r in regions if r['id'] in small), ok=True))
            if tool == 'segment_text' and ok and re.search(r'\b(buildings?|houses?|roofs?)\b', args.get('text', ''), re.I):
                handles = [h for h in call.get('handles_created', []) if h.startswith('inst#')]
                if handles and args.get('time') == 't1':
                    t1 = handles[0]
                if handles and args.get('time') == 't2':
                    t2.append((handles[0], t1))
            before, after = call.get('current_before'), call.get('current_after')
            changed = ok and after and after != before
            if changed and tool not in ('submit', 'morph', 'remove_region'):
                answer = assets.mask(after)
                for source, old in t2:
                    if source in flagged_t2:
                        continue
                    evidence = t2_evidence(assets, source, old, answer)
                    if evidence:
                        events['t2_as_change_candidates'].append(dict(event, current_mask=after, **evidence))
                        flagged_t2.add(source)
            if changed and tool == 'morph' and args.get('op') != 'fill_holes':
                source = assets.mask(args['mask'])
                answer = assets.mask(after)
                prior = assets.mask(before) if before else source
                n_before, n_after = len(components(prior)[1]), len(components(answer)[1])
                added = int((answer & ~source).sum())
                # Detect closing or another observed expansion. A pure erosion/open is not a merge.
                if (args.get('op') == 'close' or added) and n_after < n_before:
                    events['morph_merges'].append(dict(event, op=args.get('op'), radius=args.get('radius'),
                        input_mask=args['mask'], current_before=before, current_after=after,
                        components_before=n_before, components_after=n_after,
                        input_components=len(components(source)[1]), added_px=added))
            # The report becomes visible only after this call's result.
            if 'Area constraint (' in (call.get('result_text') or ''):
                saw_report = True
    events['area_report_seen'] = saw_report
    return events


def filtered(mask, cutoff):
    labels, regions = components(mask)
    return np.isin(labels, [r['id'] for r in regions if r['area'] > cutoff])


def discover(roots):
    found = set()
    for root in roots:
        root = Path(root)
        if exists(root / 'final.json'):
            found.add(root)
        else:
            found.update(p.parent for p in retry_io(lambda: list(root.rglob('final.json'))))
    if not found:
        raise ValueError('No completed session directories found')
    return sorted(found)


def summarize(session_roots, tasks_path, root=Path('.'), fixed_dir=None):
    root = Path(root).resolve()
    tasks_path = resolve(tasks_path, root)
    tasks = {Path(r['id']).stem: r for r in read_rows(tasks_path)}
    per_session, all_agent, all_fixed = [], [], []
    for directory in discover([resolve(p, root) for p in session_roots]):
        metadata = json.loads(read_text(directory / 'input.json'))
        if metadata.get('task_id') != 'T4.1':
            continue
        name = directory.name
        task = tasks[name]
        truth = read_image(resolve(task['gt_mask'], root)) > 0
        final = json.loads(read_text(directory / 'final.json'))
        prediction = read_image(directory / final['mask_path']) > 0 if final.get('mask_path') else np.zeros_like(truth)
        if fixed_dir is None:
            # Same route and strict >1040 px rule as probe/t41v2_trial.py.
            split = 'train' if 'train' in Path(task['image_t1']).parts[-3] else 'val'
            fixed_path = root / 'init/sam3_imc_p85' / split / (name + '.png')
        else:
            fixed_path = resolve(fixed_dir, root) / (name + '.png')
        fixed = filtered(read_image(fixed_path) > 0, task['area_min_m2'] / task['gsd_m'] ** 2)
        if prediction.shape != truth.shape or fixed.shape != truth.shape:
            raise ValueError(f'Mask shape differs for {name}')
        agent_counts, fixed_counts = counts(prediction, truth), counts(fixed, truth)
        all_agent.append(agent_counts)
        all_fixed.append(fixed_counts)
        def per_image(value):
            metric = score([value])
            if sum(value) == 0:
                metric['f1_c'] = 100.
            return dict(counts=value, **metric)
        per_session.append(dict(session=relative(directory, root), id=name,
            agent=per_image(agent_counts), fixed_area=per_image(fixed_counts),
            submitted_by=final.get('submitted_by'), trace=analyze_trace(directory, task, root)))
    if not per_session:
        raise ValueError('No T4.1 sessions matched')
    categories = ('mask_select', 'instance_select', 'remove_small_after_report')
    summary = {}
    for kind in categories:
        entries = [s['trace'][kind] for s in per_session]
        summary[kind] = dict(calls=sum(map(len, entries)), sessions=sum(bool(e) for e in entries),
            successful_calls=sum(bool(e['ok']) for rows in entries for e in rows),
            successful_sessions=sum(any(e['ok'] for e in rows) for rows in entries))
    no_requests = [s['session'] for s in per_session if not any(s['trace'][k] for k in categories)]
    no_success = [s['session'] for s in per_session if not any(e['ok'] for k in categories for e in s['trace'][k])]
    summary.update(no_area_filter_sessions=len(no_requests), no_area_filter_session_paths=no_requests,
        no_successful_area_filter_sessions=len(no_success), no_successful_area_filter_session_paths=no_success,
        area_report_sessions=sum(s['trace']['area_report_seen'] for s in per_session))
    return dict(sessions=len(per_session), tasks=relative(tasks_path, root),
        missing_task_ids=sorted(set(tasks) - {s['id'] for s in per_session}),
        rules=dict(area='8 邻域。select 统计请求次数，另列成功次数。remove_region 仅计看到面积报告后实际删除面积不超过任务阈值的组件。',
            t2='候选：某次更新当前答案保留至少 80% 的 T2 建筑分割并且答案至少 80% 来自该分割。每个 T2 分割只列首次。另报与 T1 的像素重叠和实例重叠比例；不据此断言建筑没有变化。需人工复核。',
            morph='当前答案更新后组件数减少，且操作是 close 或实际增加像素；fill_holes 不计。列出输入与此前当前答案，供核对。',
            metrics='沿用 t3_metrics.counts/score 与 probe/t41v2_trial.py。全图像素累计 F1/P/R，百分数；固定 SAM3 I-M-C p85 后按严格大于面积阈值筛 8 邻域组件。逐图空空 F1 按 recorder 取 100。'),
        filtering=summary, agent=dict(counts=np.sum(all_agent, axis=0).tolist(), **score(all_agent)),
        fixed_area=dict(counts=np.sum(all_fixed, axis=0).tolist(), **score(all_fixed)), per_session=per_session)


def markdown(report):
    lines = [f'会话 {report["sessions"]} 个。',
        f'累计像素 F1：Agent {report["agent"]["f1_c"]:.2f}；固定路线 + 面积 {report["fixed_area"]["f1_c"]:.2f}。', '']
    for kind, label in (('mask_select', 'mask 面积过滤'), ('instance_select', '实例面积过滤'),
                        ('remove_small_after_report', '面积报告后删除小组件')):
        value = report['filtering'][kind]
        lines.append(f'- {label}：{value["calls"]} 次、{value["sessions"]} 个会话；成功 {value["successful_calls"]} 次、{value["successful_sessions"]} 个会话。')
    lines += [f'- 无面积过滤请求：{report["filtering"]["no_area_filter_sessions"]} 个会话。',
              f'- 无成功面积过滤：{report["filtering"]["no_successful_area_filter_sessions"]} 个会话。', '',
              '| 会话 | Agent F1 / P / R | 固定 + 面积 F1 / P / R |', '|---|---|---|']
    for row in report['per_session']:
        fmt = lambda v: f'{v["f1_c"]:.2f} / {v["precision"]:.2f} / {v["recall"]:.2f}'
        lines.append(f'| {row["session"]} | {fmt(row["agent"])} | {fmt(row["fixed_area"])} |')
    lines += ['', 'T2 建筑整体纳入候选。需人工复核。', report['rules']['t2'], '']
    flags = 0
    for row in report['per_session']:
        for event in row['trace']['t2_as_change_candidates']:
            flags += 1
            overlap = event['t1_overlap_fraction']
            text = '无 T1 分割' if overlap is None else f'T1 像素重叠 {overlap:.3f}，达 0.25 重叠的实例 {event["t2_instances_overlapping_t1"]}/{event["t2_instances"]}'
            lines.append(f'- {row["session"]}，步骤 {event["step"]}（调用 {event["call"]}），{event["tool"]} → {event["current_mask"]}：T2 保留 {event["t2_retained_fraction"]:.3f}，答案来自 T2 {event["answer_from_t2_fraction"]:.3f}；{text}。')
    if not flags:
        lines.append('无候选。')
    lines += ['', '形态学合并候选。需人工复核。', '']
    flags = 0
    for row in report['per_session']:
        for event in row['trace']['morph_merges']:
            flags += 1
            lines.append(f'- {row["session"]}，步骤 {event["step"]}（调用 {event["call"]}），{event["op"]} 半径 {event["radius"]}：组件 {event["components_before"]} → {event["components_after"]}，新增 {event["added_px"]} 像素。')
    if not flags:
        lines.append('无候选。')
    lines += ['', '口径：', report['rules']['area'], report['rules']['morph'], report['rules']['metrics'],
              f'未找到完成会话的任务：{report["missing_task_ids"]}。']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, default=Path('.'))
    parser.add_argument('--sessions', type=Path, nargs='+', required=True,
                        help='Config roots or individual completed session directories.')
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--fixed-dir', type=Path, help='Optional fixed masks before the same strict area filter.')
    parser.add_argument('--output', type=Path, required=True, help='Output directory, relative to run-root if not absolute.')
    args = parser.parse_args()
    report = summarize(args.sessions, args.tasks, args.run_root, args.fixed_dir)
    output = resolve(args.output, args.run_root)
    write_json(output / 't41_trace_stats.json', report)
    write_text(output / 't41_trace_stats.md', markdown(report))
    print(json.dumps(dict(sessions=report['sessions'], filtering=report['filtering'],
                         agent=report['agent'], fixed_area=report['fixed_area']), ensure_ascii=False))


if __name__ == '__main__':
    main()
