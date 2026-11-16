"""N9 offline scoring: each dataset/task line is a separate pixel population."""
import argparse
import os
from pathlib import Path
import numpy as np
from agent_baseline.evaluation import load_reference, pixel_counts
from .n8_missed_assets import read, write
from .n8_missed_prepare import load_masks
from .n8_missed_pipeline import cost_total
from .n9_missed_assets import OUT
from .n9_missed_protocol import choose_config
from .n9_missed_pipeline import configs_for, config_key
from .missed_verifier_study import metrics


def metric(pred, truth, valid):
    counts = pixel_counts(pred, truth, valid)
    return metrics([counts[k] for k in ('tp', 'fp', 'fn')])


def pooled(records, key):
    return metrics([sum(r[key][k] for r in records) for k in ('tp', 'fp', 'fn')])


def summarize_line(rows):
    if len({r['group'] for r in rows}) != 1:
        raise ValueError('Cross-dataset or cross-task pooled metrics are prohibited')
    before = pooled(rows, 'baseline'); after = pooled(rows, 'after')
    empty = [r for r in rows if r['baseline']['tp']+r['baseline']['fn'] == 0]
    wins = sum(r['delta_f1'] > 1e-9 for r in rows); losses = sum(r['delta_f1'] < -1e-9 for r in rows)
    return dict(tasks=len(rows), baseline=before, after=after, delta_f1=after['f1']-before['f1'],
        macro_baseline={k: float(np.mean([r['baseline'][k] for r in rows])) for k in ('f1', 'precision', 'recall')},
        macro_after={k: float(np.mean([r['after'][k] for r in rows])) for k in ('f1', 'precision', 'recall')},
        macro_delta=float(np.mean([r['delta_f1'] for r in rows])),
        added_tp=sum(r['added_tp'] for r in rows), added_fp=sum(r['added_fp'] for r in rows),
        wins=wins, losses=losses, ties=len(rows)-wins-losses, harm_rate=losses/len(rows),
        calls=sum(r['calls'] for r in rows), vlm_trigger_rate=sum(r['calls']>0 for r in rows)/len(rows),
        candidate_generation_rate=1., counterfactual_answer_cost_cny=sum(r['answer_cost_cny'] for r in rows),
        empty_gt_tasks=len(empty), empty_gt_fp=sum(r['after']['fp'] for r in empty),
        empty_gt_added_fp=sum(r['added_fp'] for r in empty))


def sensitivity(rows):
    # Exactly one task per physical image within this line: bootstrap image units.
    rng = np.random.default_rng(20261116); samples = []
    for _ in range(2000):
        selected = [rows[i] for i in rng.integers(0, len(rows), len(rows))]
        samples.append([pooled(selected, 'after')['f1']-pooled(selected, 'baseline')['f1'],
                        float(np.mean([r['delta_f1'] for r in selected]))])
    interval = np.percentile(samples, [2.5, 97.5], axis=0)
    loo = []
    for row in rows:
        other = [r for r in rows if r['stem'] != row['stem']]
        loo.append(dict(removed=row['stem'], pooled_delta=pooled(other, 'after')['f1']-pooled(other, 'baseline')['f1'],
                        macro_delta=float(np.mean([r['delta_f1'] for r in other]))))
    return dict(unit='physical image within one dataset/task line', bootstrap_replicates=2000,
        pooled_delta_ci=interval[:,0].tolist(), macro_delta_ci=interval[:,1].tolist(), leave_one_image_out=loo)


def score(a):
    refs = {(r['group'], r['stem']): r['task'] for r in read(a.out/'evaluation_manifest.json')}
    cases = {}; metadata = {}
    states = {(r['group'],r['stem']):r for r in read(a.out/f'states_{a.split}.json')}
    for item in read(a.out/'fm1_manifest.json'):
        if item['split'] != a.split: continue
        reference = refs[item['group'],item['stem']]
        truth, valid = load_reference(reference['gt_mask'], reference.get('gt_ignore_mask'))
        with np.load(item['base'], allow_pickle=False) as z: base = z['start'].astype(bool)
        masks = load_masks(item['masks']); before = metric(base, truth, valid)
        state = states[item['group'],item['stem']]
        for config in configs_for(a.out, item['profile'], a.split):
            key = config_key(config); metadata[key] = config
            indexes = state['rankings'][key]; pred = base.copy(); covered = np.zeros_like(base)
            accepted = []; charge = 0.
            for i in indexes:
                uid = f"{item['group']}__{item['stem']}__n9_{i:04d}"
                answer = read(a.out/'answers'/config['verifier']/(uid+'.json'))
                if answer.get('answer') not in ('yes','no'):
                    raise RuntimeError('Unresolved response: '+uid+' '+config['verifier'])
                covered |= masks[i]; charge += answer.get('cost_cny',0.)
                if answer['answer'] == 'yes': pred |= masks[i]; accepted.append(i)
            after = metric(pred, truth, valid)
            cases.setdefault(key, []).append(dict(group=item['group'], stem=item['stem'], profile=item['profile'], split=a.split,
                config=config, baseline=before, after=after, delta_f1=after['f1']-before['f1'],
                added_tp=after['tp']-before['tp'], added_fp=after['fp']-before['fp'],
                calls=len(indexes), submitted_indexes=indexes, accepted_indexes=accepted, answer_cost_cny=charge,
                topk_candidate_tp=int((covered & truth & ~base & valid).sum()),
                topk_candidate_fp=int((covered & ~truth & ~base & valid).sum()),
                baseline_path=item['base'], candidate_masks=item['masks'], gt_path=reference['gt_mask'],
                gt_ignore_path=reference.get('gt_ignore_mask'), image_t1=item['image_t1'], image_t2=item['image_t2']))
    summaries = {}
    for key, rows in cases.items():
        lines = {g: summarize_line([r for r in rows if r['group']==g]) for g in sorted({r['group'] for r in rows})}
        summaries[key] = dict(config=metadata[key], split=a.split, lines=lines)
        if a.split == 'tune':
            summaries[key]['equal_line_delta_f1'] = sum(v['delta_f1'] for v in lines.values())/len(lines)
    write(a.out/f'scores_{a.split}.json', dict(cases=cases, summaries=summaries,
        metric_contract='Every pooled metric is confined to one dataset/task line; no cross-line pixel aggregation'))
    if a.split == 'tune':
        selected = {p: choose_config([r for r in summaries.values() if r['config']['profile']==p])
                    for p in ('any', 'build', 'demolish')}
        frozen = dict(profiles=selected, criterion='Each profile independently maximizes equal mean of its two tune-line delta F1 values; ties fewer calls, smaller K, fixed enumeration')
        target = a.out/'frozen_profiles.json'
        if target.exists() and read(target) != frozen: raise ValueError('Frozen profiles already differ')
        write(target, frozen)
    else:
        decisions = {}; intervals = {}; online = read(a.out/'frozen_protocol.json')['online']
        for key, record in summaries.items():
            for line, result in record['lines'].items():
                selected = [r for r in cases[key] if r['group']==line]
                intervals[line] = sensitivity(selected)
                decisions[line] = dict(profile=record['config']['profile'], config=record['config'],
                    delta_f1=result['delta_f1'], advance=result['delta_f1']>0,
                    reason='Positive within-line confirm pooled delta' if result['delta_f1']>0 else 'Nonpositive within-line confirm pooled delta',
                    online_stems=online[line.split('_')[0]], tasks=10, arms=2, seeds=[1016,1017,1018],
                    online_status='not_started')
        write(a.out/'image_sensitivity.json', intervals)
        write(a.out/'offline_decision.json', decisions)
    write(a.out/'metric.json', dict(stage='scored_'+a.split, status='complete', cost_cny=cost_total(a.out),
        cumulative=True, configurations=len(summaries), line_count=6))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--split', choices=('tune','confirm'), required=True)
    p.add_argument('--out', type=Path, default=OUT); a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Use Slurm')
    score(a)

if __name__ == '__main__': main()
