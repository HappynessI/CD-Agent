"""Fixed SECOND dev60 semantic bridge; caches read-only, all execution in Slurm.

Single-date exemplar interpretation is independent of labels and the other date.
Both rankings use identical proposals and share eight-view verification answers.
"""
import argparse
import base64
from io import BytesIO
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
from agent_baseline.planner import Planner, usage_tokens, request_cost
from agent_baseline.backends import encode_image, CROP_WORDS
from agent_baseline.view import ask_images
from mllm import parse_json
from .io_retry import read_image, read_text, write_json, exists
from .t3_routes import load_dev, CLASSES, GROUPS, GRID, group, pair_id, pixel_counts, pooled
from .t3_routes_cache import CachedVision, read_npz
from .t3_routes_match import bbox, T3_SYSTEM, T3_QUESTION
from .t3_diag import proposal_metrics

from agent_baseline.exemplar_semantics import (
    MODEL, VOCAB, PARSE_SYSTEM, PARSE_QUESTION, canonical, single_date_views, request, interpret)


def bridge_scores(masks, probabilities, pair):
    if None in pair or pair[0] == pair[1]:
        return None
    likelihood = probabilities[0][pair[0]] * probabilities[1][pair[1]]
    return np.asarray([float(likelihood[m].mean()) if m.any() else 0. for m in masks])


def rank(scores):
    return [] if scores is None else sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))


def cached_dino_ranking(path, masks):
    """Reuse historical top32 after checking stable proposal identity in Slurm."""
    saved = read_npz(path)
    indices = saved['indices'].astype(int).tolist()
    shape = tuple(map(int, saved['shape']))
    if shape != tuple(masks.shape[1:]) or len(indices) != min(max(GRID), len(masks)):
        raise ValueError('Historical DINO candidate shape/count differs')
    archived = np.unpackbits(saved['masks'], axis=1, count=int(np.prod(shape))).reshape(-1,*shape).astype(bool)
    if not np.array_equal(archived, masks[indices]):
        raise ValueError('Historical DINO candidate masks differ from current SAM proposals')
    return indices


def verification_views(images, exemplar, candidate):
    return [(f'{region} {label}', image) for region, mask in [('Example', exemplar), ('Candidate', candidate)]
        for label, image in zip(CROP_WORDS, ask_images(images, mask, bbox(mask), True))]


def union(masks, selected, exemplar):
    return (masks[selected].any(axis=0) if selected else np.zeros_like(exemplar)) & ~exemplar


def evaluate_task(row, run, historical, out, verify_k, seed):
    folder = out / 'tasks' / row['id']
    final = folder / 'metrics.json'
    if exists(final):
        return json.loads(read_text(final))
    ex = read_image(row['exemplar_mask']) > 0
    images = [read_image(row[k], rgb=True) for k in ('image_t1','image_t2')]
    request_folder = folder
    smoke_settings = out.parent / 'smoke/settings.json'
    smoke_folder = out.parent / 'smoke/tasks' / row['id']
    if out.name == 'dev60' and exists(smoke_settings) and exists(smoke_folder):
        old = json.loads(read_text(smoke_settings))
        if old['seed'] == seed and old['model'] == MODEL:
            request_folder = smoke_folder
    interpretation = interpret(images, ex, request_folder / 'interpretation', seed)
    backend = CachedVision(run / 't3_diag/cache/vision').for_row(row)
    masks = np.concatenate([backend('segment_everything', time=t) for t in ('t1','t2')])
    dino_ranking = cached_dino_ranking(historical / 'route_b/candidates' / f"{row['id']}.npz", masks)
    probabilities = [read_npz(historical / 'route_a/semantic' / f"{row['id']}_{t}.npz")['data'] for t in ('t1','t2')]
    rankings = dict(dino=dino_ranking, bridge=rank(bridge_scores(masks, probabilities, interpretation['pair'])))
    # Actual rankings and verification inputs are fixed before reading GT.
    write_json(folder / 'prediction.json', dict(interpretation=interpretation,
        rankings=rankings, verify_k=verify_k, proposal_count=len(masks)))
    answers = {}
    for i in sorted(set(rankings['dino'][:verify_k]) | set(rankings['bridge'][:verify_k])):
        answer = request(request_folder / 'verification' / str(i), T3_SYSTEM, T3_QUESTION,
            verification_views(images, ex, masks[i]), seed)
        label = answer['parsed'].get('change')
        answers[i] = dict(answer=label if label in ('yes','no') else 'unparsed',
                          cost_cny=answer['cost_cny'])
    truth = read_image(row['gt_mask']) > 0
    actual_pair = pair_id(row)
    verification_audit = [dict(index=i, answer=answer['answer'],
        counts=pixel_counts(masks[i], truth, ex)) for i,answer in answers.items()]
    rankings['oracle_class_diagnostic'] = rank(bridge_scores(masks, probabilities, actual_pair))
    records = []
    for method, ranking in rankings.items():
        for k in GRID:
            chosen = ranking[:k]
            records.append(dict(method=method, k=k, counts=pixel_counts(union(masks, chosen, ex), truth, ex),
                candidates=chosen, recall=proposal_metrics(masks[chosen], truth, ex)))
        if method != 'oracle_class_diagnostic' and verify_k:
            chosen = [i for i in ranking[:verify_k] if answers[i]['answer'] != 'no']
            records.append(dict(method=method + '_verified', k=verify_k,
                counts=pixel_counts(union(masks, chosen, ex), truth, ex), candidates=chosen,
                recall=proposal_metrics(masks[chosen], truth, ex)))
    dense = np.zeros_like(ex)
    pair = interpretation['pair']
    if None not in pair and pair[0] != pair[1]:
        dense = (probabilities[0].argmax(axis=0) == pair[0]) & (probabilities[1].argmax(axis=0) == pair[1])
    records.append(dict(method='bridge_dense', k=0, counts=pixel_counts(dense, truth, ex)))
    failures = []
    if pair != list(actual_pair): failures.append('exemplar_gt_class_disagreement_requires_review')
    if None in pair: failures.append('unknown_exemplar_class')
    if any(a['answer'] == 'unparsed' for a in answers.values()): failures.append('unparsed_verification')
    if any(a['answer']=='no' and a['counts'][0] for a in verification_audit): failures.append('verification_removed_true_pixels')
    if any(a['answer']!='no' and a['counts'][1] for a in verification_audit): failures.append('verification_retained_false_pixels')
    selected_bridge = next(r for r in records if r['method']=='bridge' and r['k']==8)
    upper = proposal_metrics(masks, truth, ex)
    if upper['covered_pixels'] < upper['truth_pixels']: failures.append('proposal_coverage_gap')
    if selected_bridge['recall']['covered_pixels'] < upper['covered_pixels']: failures.append('bridge_top8_ranking_gap')
    record = dict(id=row['id'], group=group(row), interpretation=interpretation,
        true_pair=list(actual_pair), before_correct=pair[0] == actual_pair[0],
        after_correct=pair[1] == actual_pair[1], pair_correct=pair == list(actual_pair),
        semantic_review_required=pair != list(actual_pair),
        semantic_review_note=('GT class agreement is not an adjudicated visual-understanding accuracy; '
            'review both dates for exemplar recognizability, class ambiguity and label discrepancy before assigning a model error.'),
        proposal_upper=upper, records=records,
        verification_requests=len(answers), verification_unparsed=sum(a['answer']=='unparsed' for a in answers.values()),
        verification_audit=verification_audit,
        request_archive=str(request_folder),
        cost_cny=interpretation['cost_cny'] + sum(a['cost_cny'] for a in answers.values()), failures=failures)
    write_json(final, record)
    return record


def summarize(tasks):
    output = {}
    for name in ('全部', *GROUPS):
        selected = [t for t in tasks if name == '全部' or t['group'] == name]
        metrics = {}
        for method, k in sorted({(r['method'], r['k']) for t in selected for r in t['records']}):
            records = [r for t in selected for r in t['records'] if r['method'] == method and r['k'] == k]
            score = pooled(records)
            recalls = [r['recall'] for r in records if 'recall' in r]
            if recalls:
                gt = sum(r['gt_instances'] for r in recalls)
                pixels = sum(r['truth_pixels'] for r in recalls)
                score.update(instance_recall=sum(r['aligned_instances'] for r in recalls)/gt if gt else None,
                    pixel_coverage=sum(r['covered_pixels'] for r in recalls)/pixels if pixels else None)
            metrics[f'{method}@{k}'] = score
        output[name] = dict(tasks=len(selected), semantic={key: sum(t[key] for t in selected)/len(selected)
            if selected else None for key in ('before_correct','after_correct','pair_correct')}, metrics=metrics)
    return dict(tasks=len(tasks), groups=output,
        semantic_metric='agreement with provided SECOND class labels; mismatches require visual/label review, not automatic model-error attribution',
        cost_cny=sum(t['cost_cny'] for t in tasks),
        verification_requests=sum(t['verification_requests'] for t in tasks), per_task=tasks)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--historical', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int, default=60)
    p.add_argument('--verify-k', type=int, default=8, choices=(0,4,8,16,32))
    p.add_argument('--seed', type=int, default=1016)
    p.add_argument('--metric', type=Path)
    p.add_argument('--stage', default='dev60')
    args = p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Run through Slurm')
    rows = load_dev(args.run)[:args.limit]
    policy = dict(version=1, task_ids=[r['id'] for r in rows], model=MODEL, seed=args.seed,
        verify_k=args.verify_k, classes=list(VOCAB), ranking='mean region product of predicted source/target probability',
        verification='shared original eight-image transition question; no class predictions supplied',
        dino='historical top32 original proposal indices; exact packed-mask identity checked before reuse',
        unknown='empty bridge ranking and dense prediction; no GT correction',
        run=str(args.run), historical=str(args.historical))
    settings = args.output / 'settings.json'
    if exists(settings) and json.loads(read_text(settings)) != policy: raise ValueError('Output settings differ')
    write_json(settings, policy)
    tasks = []
    def metric(failed=0):
        if args.metric:
            # Include completed and partial requests from smoke and dev60, without
            # counting response and attempt archives twice.
            bills = list(args.metric.parent.glob('*/tasks/*/**/answer.json'))
            bills += list(args.metric.parent.glob('*/tasks/*/**/failure_*.json'))
            spent = sum(json.loads(read_text(path)).get('cost_cny',0.) for path in bills)
            write_json(args.metric, dict(stage=args.stage, completed=len(tasks), failed=failed,
                failed_tasks=failed, cost_cny=spent, cumulative=True,
                result_path=str(args.output / 'summary.json')))
    try:
        for row in rows:
            tasks.append(evaluate_task(row, args.run, args.historical, args.output, args.verify_k, args.seed))
            write_json(args.output / 'summary.json', summarize(tasks))
            metric()
            print(f"completed {len(tasks)}/{len(rows)} {row['id']}", flush=True)
    except Exception:
        metric(failed=1)
        raise


if __name__ == '__main__':
    main()
