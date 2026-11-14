"""Ten dev tasks and offline exemplar-classification trace statistics."""
import argparse
import json
from pathlib import Path
import random
import re
import numpy as np
from .io_retry import read_text, read_image, write_json, write_text, exists
from .t3_routes import load_dev, pooled, pixel_counts
from .t3_routes_cache import read_npz


def sample_ten(rows):
    return random.Random(1016).sample(sorted(rows, key=lambda r: r['id']), 10)


def proposal_matches(directory, handle, exemplar):
    if not handle or not handle.startswith(('props#', 'inst#')):
        return False
    path = directory / 'insts' / f"inst_{handle.split('#')[1]}.npz"
    if not exists(path):
        return None
    z = read_npz(path)
    shape = tuple(map(int, z['shape']))
    if shape != exemplar.shape:
        raise ValueError('Archived proposal shape differs from exemplar')
    masks = np.unpackbits(z['masks'], axis=1, count=exemplar.size).reshape(-1, *shape).astype(bool)
    inter = (masks & exemplar).sum(axis=(1, 2))
    union = masks.sum(axis=(1, 2)) + exemplar.sum() - inter
    return bool(np.any(inter / np.maximum(1, union) >= .5))


def inspect_trace(directory, exemplar):
    steps = [json.loads(l) for l in read_text(directory / 'steps.jsonl').splitlines() if l.strip()]
    calls = [(s, c) for s in steps for c in s.get('calls', []) if c.get('ok')]
    used = {n: any(c['tool'] == n for _, c in calls) for n in ('classify', 'segment_semantic', 'match_exemplar')}
    evidence, review, semantic, dates = [], [], {}, set()
    for step, c in calls:
        args, tool = c.get('args') or {}, c['tool']
        if tool == 'classify':
            hit = proposal_matches(directory, args.get('props'), exemplar)
            if hit:
                evidence.append(dict(step=step['step'], tool=tool, time=args.get('time'),
                    evidence='classified proposal IoU with exemplar >= 0.5'))
                dates.add(args.get('time'))
            else:
                review.append(dict(step=step['step'], tool=tool,
                    reason='archive missing' if hit is None else 'no proposal aligned with exemplar'))
        if tool == 'segment_semantic':
            for h in c.get('output_handles', c.get('handles_created', [])):
                semantic[h] = dict(step=step['step'], tool=tool, time=args.get('time'))
            review.append(dict(step=step['step'], tool=tool, reason='full-image semantic call alone is not exemplar category judgement'))
    # T3.1 initial mask#0 is the supplied exemplar. Require both data use and
    # explicit planner class-recognition purpose for full-image semantic calls.
    for step, c in calls:
        args = c.get('args') or {}
        pair = [args.get('a'), args.get('b')]
        h = next((h for h in pair if h in semantic), None)
        planner = step.get('planner') or {}
        text = str((planner.get('thought') or planner.get('content') or '') if isinstance(planner, dict) else planner)
        purpose = re.search(r'exempl|example|示例', text, re.I) and re.search(r'class|categor|cover|veget|building|water|tree|类别|地类|建筑|植被|水', text, re.I)
        if c['tool'] == 'combine' and args.get('op') in ('intersect', 'intersection') and 'mask#0' in pair and h and purpose:
            evidence.append(dict(**semantic[h], downstream_step=step['step'],
                evidence='semantic/exemplar intersection and stated exemplar class purpose'))
            dates.add(semantic[h]['time'])
    return dict(called=used, exemplar_category_evidence=evidence, exemplar_category_judgement=bool(evidence),
        both_dates_judged={'t1', 't2'} <= dates, manual_review=review)


def audit(rows, sessions, output, resume=False):
    from .t3_routes_runtime import bind
    policy = dict(version=1, task_ids=[r['id'] for r in rows], sessions=str(sessions.resolve()), seed=1016)
    bind(output / 'agent10_settings.json', policy, resume)
    progress = output / 'agent10_progress.json'
    tasks = json.loads(read_text(progress))['tasks'] if exists(progress) else []
    done = {t['id'] for t in tasks}
    for row in rows:
        if row['id'] in done:
            continue
        d = sessions / row['id']
        final = json.loads(read_text(d / 'final.json'))
        ex, truth = [read_image(row[k]) > 0 for k in ('exemplar_mask', 'gt_mask')]
        pred = read_image(d / final['mask_path']) > 0 if final.get('mask_path') else np.zeros_like(ex)
        tasks.append(dict(id=row['id'], counts=pixel_counts(pred, truth, ex),
            submitted_by=final.get('submitted_by'), error=bool(final.get('error')), **inspect_trace(d, ex)))
        write_json(progress, dict(tasks=tasks))
    report = dict(tasks=len(tasks), per_task=tasks, score=pooled(tasks),
        calls={n: sum(t['called'][n] for t in tasks) for n in ('classify', 'segment_semantic', 'match_exemplar')},
        exemplar_category_sessions=sum(t['exemplar_category_judgement'] for t in tasks),
        both_dates_sessions=sum(t['both_dates_judged'] for t in tasks),
        review_sessions=[t['id'] for t in tasks if t['manual_review']],
        rule='classify proposal IoU>=.5 with exemplar, or semantic/exemplar intersection with stated class purpose; evidence indicates geometric relevance, not proof of intent; review unresolved traces; exemplar excluded from F1')
    write_json(output / 'agent10_metrics.json', report)
    write_text(output / 'agent10_metrics.md',
        f"开发{len(tasks)}任务。示例类别判断证据={report['exemplar_category_sessions']}，两期证据={report['both_dates_sessions']}。\n"
        f"F1/P/R={report['score']['f1_c']:.2f}/{report['score']['precision']:.2f}/{report['score']['recall']:.2f}。\n"
        '全图语义调用本身不计为示例判断。调用次数与待人工复核项见JSON。\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['sample', 'metrics'])
    p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--sessions', type=Path, help='Directory immediately containing task-id sessions')
    p.add_argument('--output-dir', type=Path, default=Path('results'))
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    rows = sample_ten(load_dev(a.run_root))
    if a.command == 'sample':
        write_text(a.output_dir / 'agent10_tasks.jsonl', ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        write_json(a.output_dir / 'agent10_selection.json', dict(seed=1016, task_ids=[r['id'] for r in rows],
            rule='random.Random(1016).sample(sorted(fixed_60, id), 10); original rows preserved'))
    else:
        if not a.sessions:
            p.error('--sessions required')
        audit(rows, a.sessions, a.output_dir, a.resume)


if __name__ == '__main__':
    main()
