"""xBD label-only sampling and four-class oracle-footprint verification.

No API on import, sampling, estimate or report. See workspace NOTES.md.
"""
import argparse
from collections import Counter, defaultdict
from functools import partial
import json
import math
from pathlib import Path
import random
import re

import numpy as np
from PIL import Image, ImageDraw

from .io_retry import exists, read_image, read_text, retry_io, write_json, write_text

GRADES = ('no-damage', 'minor-damage', 'major-damage', 'destroyed')
UNKNOWN = 'un-classified'
POSITIVE = frozenset(GRADES[2:])
SEED = 1016
QUESTION_VARIANT = 'xbd-jds-v1'
JDS_SOURCE = 'https://arxiv.org/abs/1911.09296 (xBD, Joint Damage Scale); https://xview2.org/'
# Short structural descriptions of the official JDS, not new pixel/area thresholds.
JDS_QUESTION = '''Classify the post-disaster damage of the ONE building outlined in red.
T1 is pre-disaster; T2 is post-disaster. Use T1 to identify the original building.
Use the xBD Joint Damage Scale:
no-damage: no visible structural damage; the building appears undisturbed.
minor-damage: minor damage, such as missing roof shingles/elements; the main structure remains intact.
major-damage: substantial structural damage, such as partial wall or roof collapse.
destroyed: complete destruction, such as a collapsed or leveled building.
Judge structural damage, not just color, shadows, viewing angle or seasonal appearance.
Return ONLY one JSON object with exactly one field: {"damage":"LEVEL"}.
LEVEL must be no-damage, minor-damage, major-damage, or destroyed.'''
RULE = ('seed=1016; nonempty post-disaster building labels; exactly half severe images '
        '(>=1 major/destroyed), half clean images (only no/minor; no un-classified); '
        'within each stratum greedily balance disaster type then event, seeded random ties '
        'and seeded shuffled names; positive stratum first; no model outputs read')


def jsonl(path):
    return [json.loads(x) for x in read_text(path).splitlines() if x.strip()]


def save_jsonl(path, rows):
    write_text(path, ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True)+'\n' for r in rows))


def features(path):
    data = json.loads(read_text(path))
    return data, [f for f in data['features']['xy'] if f['properties']['feature_type'] == 'building']


def label_row(path, data_root):
    data, buildings = features(path)
    counts = Counter(f['properties']['subtype'] for f in buildings)
    if set(counts) - set(GRADES) - {UNKNOWN}:
        raise ValueError('Unknown xBD subtype')
    name = path.name.removesuffix('_post_disaster.json')
    split = path.parent.parent.name
    return dict(split=split, name=name, event=name.rsplit('_', 1)[0],
                disaster_type=data['metadata']['disaster_type'],
                pre=f'{split}/images/{name}_pre_disaster.png',
                post=f'{split}/images/{name}_post_disaster.png',
                label=str(path.relative_to(data_root)),
                counts={g: counts[g] for g in (*GRADES, UNKNOWN)}, buildings=len(buildings))


def inventory(data_root, split, output, resume=False):
    """Atomic label inventory checkpoint after each small group; no image reads."""
    data_root, output = Path(data_root), Path(output)
    rows = {r['name']: r for r in jsonl(output)} if resume and exists(output) else {}
    paths = sorted(retry_io(lambda: list((data_root/split/'labels').glob('*_post_disaster.json'))))
    if not paths:
        raise ValueError(f'No post-disaster labels for {split}')
    for i, path in enumerate(paths):
        name = path.name.removesuffix('_post_disaster.json')
        if name not in rows:
            rows[name] = label_row(path, data_root)
        if i % 50 == 49:
            save_jsonl(output, sorted(rows.values(), key=lambda r: r['name']))
    result = sorted(rows.values(), key=lambda r: r['name'])
    save_jsonl(output, result)
    return result


def severe(row):
    return any(row['counts'][g] for g in POSITIVE)


def sample(rows, size, seed=SEED):
    """Fixed label-only rule: 50% severe, 50% purely no/minor."""
    if size <= 0 or size % 2:
        raise ValueError('Require positive even sample size')
    if len({r['name'] for r in rows}) != len(rows):
        raise ValueError('Duplicate image names')
    rng = random.Random(seed)
    selected, type_n, event_n = [], Counter(), Counter()
    for positive in (True, False):
        groups = defaultdict(list)
        for row in sorted(rows, key=lambda r: r['name']):
            if row['buildings'] > 0 and severe(row) == positive and (positive or not row['counts'][UNKNOWN]):
                groups[row['event']].append(row)
        if sum(map(len, groups.values())) < size//2:
            raise ValueError('Insufficient eligible images for frozen 50/50 rule')
        for group in groups.values():
            rng.shuffle(group)
        for _ in range(size//2):
            available = sorted(e for e, group in groups.items() if group)
            rng.shuffle(available)
            event = min(available, key=lambda e: (type_n[groups[e][-1]['disaster_type']], event_n[e]))
            row = groups[event].pop()
            selected.append(row)
            type_n[row['disaster_type']] += 1
            event_n[event] += 1
    return sorted(selected, key=lambda r: r['name'])


def sample_all(data_root, output, resume=False):
    output = Path(output)
    dev = sample(inventory(data_root, 'hold', output/'xbd_inventory_hold.jsonl', resume), 60)
    report = sample(inventory(data_root, 'test', output/'xbd_inventory_test.jsonl', resume), 100)
    if {r['name'] for r in dev} & {r['name'] for r in report}:
        raise ValueError('Hold/test names overlap')
    probe = sample(dev, 10)
    sets = dict(dev60=dev, report100=report, probe10=probe)
    for name, rows in sets.items():
        path = output/f'xbd_{name}.jsonl'
        if exists(path) and jsonl(path) != rows:
            raise ValueError('Frozen manifest differs; do not resample after model evaluation')
        save_jsonl(path, rows)
    lines = ['# xBD 抽样', '', RULE, '',
             '数据路径相对数据根。hold→dev60，test→report100；probe10 是 dev60 子集。',
             '只读灾后 JSON；没有读取模型输出。负图严格仅含 no/minor。',
             'un-classified 单列，不是负类；忽略其像素。本轮指标按建筑计算。', '']
    audit = {}
    for name, rows in sets.items():
        counts = {g: sum(r['counts'][g] for r in rows) for g in (*GRADES, UNKNOWN)}
        audit[name] = dict(images=len(rows), severe_images=sum(map(severe, rows)),
                           events=dict(sorted(Counter(r['event'] for r in rows).items())),
                           disaster_types=dict(sorted(Counter(r['disaster_type'] for r in rows).items())),
                           buildings=sum(counts.values()), counts=counts)
        lines += [f'## {name}', '', f'图数 {len(rows)}；严重图 {sum(map(severe, rows))}。', '',
                  '| 事件 | 灾种 | 张数 |', '| --- | --- | ---: |']
        for event, n in audit[name]['events'].items():
            kind = next(r['disaster_type'] for r in rows if r['event'] == event)
            lines += [f'| {event} | {kind} | {n} |']
        lines += ['', '| 等级 | 建筑数 |', '| --- | ---: |']
        lines += [f'| {g} | {n} |' for g, n in counts.items()]
        lines += [f'| 合计 | {sum(counts.values())} |', '']
    write_json(output/'xbd_sampling.json', dict(seed=SEED, rule=RULE, sets=audit))
    write_text(output/'xbd_sampling.md', '\n'.join(lines)+'\n')
    return audit


# xBD WKT contains pixel-coordinate POLYGONs; support holes and MULTIPOLYGONs too.
def polygon_rings(text):
    if not re.match(r'^\s*(POLYGON|MULTIPOLYGON)\s*\(', text, re.I):
        raise ValueError('Expected WKT POLYGON or MULTIPOLYGON')
    depth, parts, start, ring_text = 0, [], None, []
    for i, char in enumerate(text):
        if char == '(':
            depth += 1
            start = i+1
        elif char == ')':
            if start is not None:
                ring_text.append((depth, text[start:i]))
                start = None
            depth -= 1
    if depth or not ring_text:
        raise ValueError('Malformed polygon WKT')
    # Every outer polygon starts at (( for POLYGON or ((( for MULTIPOLYGON.
    polygons, current = [], []
    ring_iter = iter(ring_text)
    # Parse ring grouping through the nesting rather than splitting coordinates.
    polygon_depth = 1 if text.lstrip().upper().startswith('POLYGON') else 2
    depth = 0
    for char in text:
        if char == '(':
            depth += 1
            if depth == polygon_depth:
                current = []
            if depth == polygon_depth+1:
                _, coords = next(ring_iter)
                points = [tuple(map(float, p.strip().split())) for p in coords.split(',')]
                if len(points) < 4 or any(len(p) != 2 or not all(math.isfinite(v) for v in p) for p in points):
                    raise ValueError('Invalid polygon ring')
                current.append(points)
        elif char == ')':
            if depth == polygon_depth:
                polygons.append(current)
            depth -= 1
    return polygons


def polygon_mask(text, shape):
    canvas = Image.new('L', (shape[1], shape[0]))
    for rings in polygon_rings(text):
        part = Image.new('L', canvas.size)
        draw = ImageDraw.Draw(part)
        draw.polygon(rings[0], fill=1)
        for hole in rings[1:]:
            draw.polygon(hole, fill=0)
        canvas = Image.fromarray(np.maximum(np.asarray(canvas), np.asarray(part)))
    return np.asarray(canvas, dtype=bool)


def building_crops(rgb, text):
    from render import crop_box, render_pair
    if len(rgb) != 2 or rgb[0].shape != rgb[1].shape:
        raise ValueError('Require aligned pre/post RGB arrays')
    mask = polygon_mask(text, rgb[0].shape[:2])
    yy, xx = np.nonzero(mask)
    if not len(xx):
        raise ValueError('Empty rasterized building')
    bbox = [int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1]
    box = crop_box(bbox, mask.shape)  # frozen ask close-up: 4x bbox, >=64, 384px output
    return render_pair(rgb, mask, box), box


def parse_answer(raw):
    try:
        value = json.loads(raw.strip())
    except (ValueError, AttributeError):
        return None
    return value['damage'] if isinstance(value, dict) and set(value) == {'damage'} and value['damage'] in GRADES else None


def building_records(rows, data_root):
    records, skipped = [], 0
    for row in rows:
        _, buildings = features(Path(data_root)/row['label'])
        actual = Counter(f['properties']['subtype'] for f in buildings)
        if any(actual[g] != row['counts'][g] for g in (*GRADES, UNKNOWN)):
            raise ValueError('Labels changed since frozen manifest')
        for i, feature in enumerate(buildings):
            grade = feature['properties']['subtype']
            if grade == UNKNOWN:
                skipped += 1
                continue
            if grade not in GRADES:
                raise ValueError('Unexpected grade')
            records.append(dict(uid=f"{row['name']}:{i}", name=row['name'], event=row['event'],
                                pre=row['pre'], post=row['post'], wkt=feature['wkt'], truth=grade))
    return records, skipped


def damage_answer(images, guard, seed, question_variant, client=None, snapshot=None, attempt_record=None):
    """ApiAsk image blocks/JPEG settings and Planner transport/pricing; one attempt."""
    from agent_baseline.backends import ApiAsk
    from agent_baseline.planner import request_cost, usage_tokens
    if question_variant != QUESTION_VARIANT:
        raise ValueError('Explicit supported damage question required')
    def event_record(event):
        if attempt_record:
            attempt_record(dict(status='response' if event.get('raw') else 'error',
                http_status=event.get('http_status'), error_type=event.get('error_type'),
                request_id=event.get('request_id'), usage=event.get('usage'),
                seconds=event.get('seconds')))
        body = event.get('error_body', '').lower()
        if event.get('http_status') in (429, 403, 402) or any(w in body for w in (
                'insufficient balance', 'arrearage', 'payment required', '欠费', 'insufficientbalance', 'quota')):
            guard.trip(f"HTTP {event.get('http_status')} / quota or balance")
        if event.get('raw'):
            raw = event['raw']
            tokens = usage_tokens(raw.get('usage'))
            answer = raw.get('choices', [{}])[0].get('message', {}).get('content') or ''
            # Persist the paid reply before Planner charges the ledger; repairable on resume.
            if snapshot:
                snapshot(dict(raw=answer, prediction=parse_answer(answer), tokens=tokens,
                              cost_cny=request_cost('qwen3.7-flash', tokens), request_id=event.get('request_id')))
    verifier = ApiAsk(client=client, guard=guard, model='qwen3.7-flash', max_attempts=1)
    answer, _, tokens, cost = verifier.one(images, '', 1,
        question_variant=question_variant, record=event_record, seed=seed)
    return dict(raw=answer['raw'], prediction=answer['answer'], tokens=tokens, cost_cny=cost)


def api_backend(record, guard, data_root, question_variant, request_dir):
    rgb = [read_image(Path(data_root)/record[k], rgb=True) for k in ('pre','post')]
    images, box = building_crops(rgb, record['wkt'])
    # Only geometry/images enter the prompt; truth and event are not passed.
    seed = SEED + sum(map(ord, record['uid']))
    guard.check()
    attempts = Path(request_dir)/'attempts'
    safe_uid = record['uid'].replace(':','__')
    old = retry_io(lambda: list(attempts.glob(safe_uid+'_*.json')))
    attempt_path = attempts/f'{safe_uid}_{len(old)+1:03d}.json'
    write_json(attempt_path, dict(uid=record['uid'], status='dispatched'))
    log_attempt = lambda value: write_json(attempt_path, dict(uid=record['uid'], **value))
    snapshot = lambda result: write_json(Path(request_dir)/'answer_parts'/
        (record['uid'].replace(':','__')+'.json'), dict(uid=record['uid'], box=box, **result))
    return dict(box=box, **damage_answer(images, guard, seed, question_variant,
                                       snapshot=snapshot, attempt_record=log_attempt))


def reconcile_snapshots(directory):
    """Repair a crash after per-request reply write but before budget write."""
    directory = Path(directory)
    budget_file = directory/'budget.json'
    state = json.loads(read_text(budget_file)) if exists(budget_file) else dict(charges={})
    for path in retry_io(lambda: list((directory/'answer_parts').glob('*.json'))):
        result = json.loads(read_text(path))
        if 'cost_cny' in result:
            state['charges'][result['uid']] = max(state['charges'].get(result['uid'], 0), result['cost_cny'])
    state['spent_cny'] = sum(state['charges'].values())
    write_json(budget_file, state)


def run_probe(records, backend, directory, max_cost, concurrency=1, allowance=.02,
              resume=False, clear_stop=False, question_variant=QUESTION_VARIANT):
    from .missed_sources_ask import run_requests
    directory = Path(directory)
    if question_variant != QUESTION_VARIANT:
        raise ValueError('Unsupported question')
    if any(not math.isfinite(v) or v <= 0 for v in (max_cost, allowance)) or concurrency < 1:
        raise ValueError('Require finite positive cost cap/allowance and positive concurrency')
    if clear_stop and not resume:
        raise ValueError('--clear-stop requires --resume')
    previous = retry_io(lambda: list((directory/'answer_parts').glob('*.json')))
    if not resume and (previous or exists(directory/'settings.json') or exists(directory/'budget.json')):
        raise ValueError('Existing run; use --resume')
    settings = dict(version=1, question_variant=question_variant, question=JDS_QUESTION,
                    model='qwen3.7-flash', crop='render.crop_box(scale=4,min_side=64);render_pair(size=384)',
                    records=records, seed=SEED)
    reconcile_snapshots(directory)
    return run_requests(records, backend, directory, max_cost, concurrency, allowance,
                        resume, clear_stop, settings)


def divide(a, b):
    return a/b if b else None


def metrics(records, answers):
    matrix = [[0]*4 for _ in GRADES]
    unparsed, pending = Counter(), Counter()
    tp = fp = fn = tn = abstain_pos = abstain_neg = 0
    for row in records:
        truth = row['truth']; uid = row['uid']
        if uid not in answers:
            pending[truth] += 1
            continue
        pred = answers[uid].get('prediction')
        if pred not in GRADES:
            unparsed[truth] += 1
            if truth in POSITIVE:
                abstain_pos += 1
            else:
                abstain_neg += 1
            continue
        matrix[GRADES.index(truth)][GRADES.index(pred)] += 1
        positive, real = pred in POSITIVE, truth in POSITIVE
        tp += int(positive and real); fp += int(positive and not real)
        fn += int(not positive and real); tn += int(not positive and not real)
    classes = {}
    for i, grade in enumerate(GRADES):
        support = sum(r['truth'] == grade for r in records)
        attempted = sum(matrix[i])+unparsed[grade]
        classes[grade] = dict(support=support, attempted=attempted,
            precision=divide(matrix[i][i], sum(row[i] for row in matrix)),
            recall=divide(matrix[i][i], attempted), unparsed=unparsed[grade], pending=pending[grade])
    # Failed-to-parse severe buildings count as misses in the primary recall.
    binary = dict(tp=tp, fp=fp, fn=fn+abstain_pos, tn=tn,
        unparsed_positive=abstain_pos, unparsed_negative=abstain_neg,
        precision=divide(tp, tp+fp), recall=divide(tp, tp+fn+abstain_pos),
        f1=divide(2*tp, 2*tp+fp+fn+abstain_pos))
    return dict(confusion_matrix=matrix, unparsed=[unparsed[g] for g in GRADES],
                pending=[pending[g] for g in GRADES], per_class=classes, binary=binary)


def make_report(records, skipped, request_dir, output):
    request_dir, output = Path(request_dir), Path(output)
    answers = {}
    for path in retry_io(lambda: list((request_dir/'answer_parts').glob('*.json'))):
        result = json.loads(read_text(path)); answers[result['uid']] = result
    wanted = {r['uid'] for r in records}
    if set(answers) - wanted:
        raise ValueError('Unexpected answers; use matching manifest')
    cost = sum(a.get('cost_cny', 0) for a in answers.values())
    state_file = request_dir/'budget.json'
    ledger = json.loads(read_text(state_file)).get('spent_cny', 0) if exists(state_file) else 0
    attempts = [json.loads(read_text(p)) for p in retry_io(lambda: list((request_dir/'attempts').glob('*.json')))]
    report = dict(status='complete' if len(answers) == len(records) else 'pending_or_incomplete',
        grades=GRADES, matrix_axes='rows=truth; columns=prediction; unparsed and pending separate',
        scoring_unit='building; oracle post-disaster footprint',
        unknown_policy='un-classified skipped; neither positive nor negative; ignore their pixels',
        unparsed_policy='class recall includes unparsed; binary severe unparsed counts as FN; negative unparsed shown separately',
        pending_policy='unrequested buildings excluded from metric denominators; incomplete results are provisional',
        expected_requests=len(records), requests=len(answers), request_attempts=len(attempts),
        failed_attempts=sum(a['status'] == 'error' for a in attempts),
        uncertain_attempts=sum(a['status'] == 'dispatched' for a in attempts),
        skipped_unclassified=skipped,
        cost_cny=cost, ledger_cost_cny=ledger,
        cost_note='Usage-based existing Planner ledger estimate; provider console bill is authoritative',
        jds_source=JDS_SOURCE, question_variant=QUESTION_VARIANT,
        **metrics(records, answers), by_event={})
    for event in sorted({r['event'] for r in records}):
        subset = [r for r in records if r['event'] == event]
        report['by_event'][event] = dict(expected=len(subset),
            requests=sum(r['uid'] in answers for r in subset), **metrics(subset, answers)['binary'])
    write_json(output/'xbd_probe10_report.json', report)
    fmt = lambda x: '—' if x is None else f'{x:.4f}'
    lines = ['# xBD probe10 验证器报告', '', f"状态：{report['status']}。已保存请求 {len(answers)}/{len(records)}；跳过 un-classified {skipped} 栋。",
        f"请求尝试 {report['request_attempts']} 次；失败 {report['failed_attempts']} 次；已派发但未记录响应 {report['uncertain_attempts']} 次（崩溃或发送前 STOP，计费状态未知）。",
        f'响应费用 {cost:.6f} 元；ledger {ledger:.6f} 元。按现有单价与实际 usage 记账，最终以账单为准。',
        '已知灾后建筑多边形的分类试探；不是检测结果。un-classified 及其像素不算正负。',
        '无法解析的正例计入二分类漏检；负例无法解析数另列。未请求项另列，未完成指标仅供检查。', '',
        '| 真值 / 预测 | '+' | '.join(GRADES)+' | 无法解析 | 未请求 |', '| --- | '+' ---: |'*6]
    for i, grade in enumerate(GRADES):
        lines += ['| '+grade+' | '+' | '.join(map(str, report['confusion_matrix'][i]+[report['unparsed'][i], report['pending'][i]]))+' |']
    lines += ['', '| 等级 | Precision | Recall |', '| --- | ---: | ---: |']
    lines += [f"| {g} | {fmt(v['precision'])} | {fmt(v['recall'])} |" for g,v in report['per_class'].items()]
    b = report['binary']
    lines += ['', f"二分类 P/R/F1：{fmt(b['precision'])} / {fmt(b['recall'])} / {fmt(b['f1'])}。", '',
        '| 事件 | 请求/应请求 | P | R | F1 | 无法解析正/负 |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for event, v in report['by_event'].items():
        lines += [f"| {event} | {v['requests']}/{v['expected']} | {fmt(v['precision'])} | {fmt(v['recall'])} | {fmt(v['f1'])} | {v['unparsed_positive']}/{v['unparsed_negative']} |"]
    lines += ['', f'JDS 来源：{JDS_SOURCE}。', '']
    write_text(output/'xbd_probe10_report.md', '\n'.join(lines))
    return report


def estimate(records, skipped, ledger_path, run_root, output):
    """Use same flash ledger mean as a proxy, not labels/model outputs for selection."""
    rows = jsonl(ledger_path)
    priced = [r for r in rows if r.get('cost_cny', 0) > 0]
    if not priced:
        raise ValueError('No priced requests in reference ledger')
    unit = sum(r['cost_cny'] for r in priced)/len(priced)
    value = dict(requests=len(records), skipped_unclassified=skipped,
        total_buildings=len(records)+skipped, model='qwen3.7-flash',
        reference_ledger=str(Path(ledger_path).relative_to(run_root)),
        priced_reference_requests=len(priced), unit_cost_cny=unit,
        estimated_cost_cny=unit*len(records), proxy=True,
        note='Existing same-model f37_corr ledger mean; old prompt has four images, new prompt two. Not a guaranteed bound.',
        by_image=dict(sorted(Counter(r['name'] for r in records).items())))
    write_json(Path(output)/'xbd_probe10_estimate.json', value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('sample','estimate','ask','report'))
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('results'))
    parser.add_argument('--manifest', type=Path, default=Path('results/xbd_probe10.jsonl'))
    parser.add_argument('--request-dir', type=Path, default=Path('results/xbd_probe10_requests'))
    parser.add_argument('--run-root', type=Path)
    parser.add_argument('--ledger', type=Path, help='Path relative to run-root')
    parser.add_argument('--question-variant', choices=(QUESTION_VARIANT,), help='ask requires explicit xbd-jds-v1; default ask unchanged')
    parser.add_argument('--concurrency', type=int, default=1)
    parser.add_argument('--max-cost-cny', type=float)
    parser.add_argument('--request-cost-cny', type=float, default=.02)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--clear-stop', action='store_true')
    args = parser.parse_args()
    if args.command == 'sample':
        value = sample_all(args.data_root, args.output_dir, args.resume)
    else:
        rows = jsonl(args.manifest)
        if len(rows) != 10 or len({r['name'] for r in rows}) != 10 or any(r['split'] != 'hold' for r in rows):
            parser.error('Require frozen 10 unique hold images')
        if sum(map(severe, rows)) < 5:
            parser.error('Probe must contain >=5 severe images')
        records, skipped = building_records(rows, args.data_root)
        if args.command == 'estimate':
            if not args.run_root or not args.ledger:
                parser.error('estimate requires --run-root and relative --ledger')
            value = estimate(records, skipped, args.run_root/args.ledger, args.run_root, args.output_dir)
        elif args.command == 'report':
            value = make_report(records, skipped, args.request_dir, args.output_dir)
        else:
            if args.question_variant is None or args.max_cost_cny is None:
                parser.error('ask requires --question-variant xbd-jds-v1 and --max-cost-cny')
            backend = partial(api_backend, data_root=args.data_root,
                              question_variant=args.question_variant, request_dir=args.request_dir)
            try:
                value = run_probe(records, backend, args.request_dir, args.max_cost_cny,
                    args.concurrency, args.request_cost_cny, args.resume, args.clear_stop, args.question_variant)
            finally:
                make_report(records, skipped, args.request_dir, args.output_dir)
    print(json.dumps(value, ensure_ascii=False))


if __name__ == '__main__':
    main()
