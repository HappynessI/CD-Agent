"""Audit fixed agent10 sessions, partial-attempt costs, and excluded-exemplar scores."""
import argparse
import json
from pathlib import Path
from agent_baseline.planner import request_cost, usage_tokens
from .io_retry import read_text, read_image, write_json, exists
from .t3_routes import pooled, pixel_counts, group, GROUPS

SEEDS = (1016,2027,3038)


def session_cost(folder):
    """Provider response IDs deduplicate planner/ask; semantic files own their bills."""
    seen, cost, requests, incomplete_lines = set(), 0., 0, 0
    path = folder / 'messages.jsonl'
    events = []
    if exists(path):
        for line in read_text(path).splitlines():
            if not line.strip(): continue
            try: events.append(json.loads(line))
            except json.JSONDecodeError: incomplete_lines += 1
    # Semantic attempts also appear in Recorder messages. Count their original
    # files only, so a request completed before Recorder.step is still included.
    events = [e for e in events if not ('requested_model' in e and 'request' not in e)]
    events += [json.loads(read_text(p)) for p in sorted(folder.glob('semantic_bridge/*/*/attempt_*.json'))]
    for index,event in enumerate(events):
        raw = event.get('raw') if isinstance(event.get('raw'),dict) else {}
        usage = event.get('usage') or raw.get('usage')
        if not usage: continue
        identity = event.get('request_id') or raw.get('id') or ('local-event',index)
        if identity in seen: continue
        seen.add(identity)
        model = (event.get('request') or {}).get('model') or event.get('requested_model') or raw.get('model') or 'qwen3.7-flash'
        cost += request_cost(model,usage_tokens(usage)); requests += 1
    final_path = folder / 'final.json'
    final_cost = json.loads(read_text(final_path)).get('cost_cny',0.) if exists(final_path) else None
    return dict(cost_cny=cost if requests else (final_cost or 0.), known_requests=requests,
        recorder_cost_cny=final_cost, incomplete_message_lines=incomplete_lines,
        basis='deduplicated recorded response usage' if requests else 'final record or no recorded usage')


def completion(final):
    if final is None: return 'missing_final'
    if final.get('error') or final.get('termination_reason') in ('error','input_error') or final.get('submitted_by')=='error': return 'error'
    if final.get('termination_reason')=='submit' and final.get('submitted_by')=='model': return 'successful_submit'
    return 'fallback'


def archive_incomplete(root, config, rows):
    """Preserve interrupted traces before Recorder would truncate them on restart."""
    for row in rows:
        folder = root / config / 'T3.1/val' / row['id']
        if not exists(folder) or exists(folder/'final.json'): continue
        destination = root / 'interrupted_attempts' / config / row['id']
        destination.mkdir(parents=True,exist_ok=True)
        index = len(list(destination.glob('attempt_*')))
        folder.rename(destination / f'attempt_{index:03d}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tasks', type=Path, required=True)
    p.add_argument('--expect-config')
    p.add_argument('--prepare-config')
    args = p.parse_args()
    rows = [json.loads(line) for line in read_text(args.tasks).splitlines() if line.strip()]
    root = args.output/'online'
    if args.prepare_config:
        archive_incomplete(root,args.prepare_config,rows)
        return
    records, costs = [], []
    for arm in ('baseline','bridge'):
        for seed in SEEDS:
            for row in rows:
                folder = root / f'{arm}_s{seed}' / 'T3.1/val' / row['id']
                bill = session_cost(folder)
                costs.append(dict(path=str(folder), **bill))
                final = json.loads(read_text(folder/'final.json')) if exists(folder/'final.json') else None
                record = dict(id=row['id'], arm=arm, seed_offset=seed, group=group(row),
                    state=completion(final), cost_cny=bill['cost_cny'], final_path=str(folder/'final.json'))
                if final is not None:
                    ex,truth = [read_image(row[k])>0 for k in ('exemplar_mask','gt_mask')]
                    pred = read_image(folder/final['mask_path'])>0
                    steps = [json.loads(line) for line in read_text(folder/'steps.jsonl').splitlines() if line.strip()] if exists(folder/'steps.jsonl') else []
                    calls = [c for step in steps for c in step.get('calls',[])]
                    record.update(counts=pixel_counts(pred,truth,ex), submitted_by=final.get('submitted_by'),
                        termination_reason=final.get('termination_reason'),
                        tool_calls=len(calls),
                        tool_errors=[dict(tool=c.get('tool'), code=c.get('error_code'))
                            for c in calls if not c.get('ok', True)],
                        semantic_calls={tool:sum(c.get('tool')==tool for c in calls) for tool in
                            ('interpret_exemplar','rank_semantic_transition','match_exemplar','classify','segment_semantic')})
                records.append(record)
    for folder in sorted((root/'interrupted_attempts').glob('*/*/attempt_*')):
        costs.append(dict(path=str(folder), interrupted=True, **session_cost(folder)))
    summaries = {}
    for arm in ('baseline','bridge'):
        summaries[arm] = {}
        for seed in SEEDS:
            chosen = [r for r in records if r['arm']==arm and r['seed_offset']==seed]
            states = {state:sum(r['state']==state for r in chosen) for state in ('successful_submit','fallback','error','missing_final')}
            summaries[arm][str(seed)] = dict(expected=len(rows), ended=len(rows)-states['missing_final'],
                **states, complete=states['successful_submit']==len(rows),
                groups={g:pooled([r for r in chosen if 'counts' in r and (g=='全部' or r['group']==g)]) for g in ('全部',*GROUPS)})
    spent_online = sum(c['cost_cny'] for c in costs)
    write_json(args.output/'online_summary.json',dict(records=records,summaries=summaries,cost_audit=costs,
        cost_cny=spent_online, complete=all(r['state']=='successful_submit' for r in records),
        scoring='all ended sessions scored including error/fallback; successful-submit counts reported separately'))
    bills = list(args.output.glob('*/tasks/*/**/answer.json'))+list(args.output.glob('*/tasks/*/**/failure_*.json'))
    spent = sum(json.loads(read_text(path)).get('cost_cny',0.) for path in bills)
    selected = records
    if args.expect_config:
        arm,seed = args.expect_config.rsplit('_s',1)
        selected = [r for r in records if r['arm']==arm and r['seed_offset']==int(seed)]
        if not selected: raise ValueError('Unknown expected config')
    missing = sum(r['state']=='missing_final' for r in selected)
    errors = sum(r['state']=='error' for r in selected)
    fallback = sum(r['state']=='fallback' for r in selected)
    write_json(args.output/'metric.json',dict(stage='online',config=args.expect_config,
        completed=sum(r['state']!='missing_final' for r in records), expected=6*len(rows),
        successful_submit=sum(r['state']=='successful_submit' for r in records),
        failed=errors, missing_current_config=missing, fallback_current_config=fallback,
        cumulative=True,cost_cny=spent+spent_online,result_path=str(args.output/'online_summary.json')))
    if args.expect_config and (missing or errors or fallback):
        raise SystemExit(f'Incomplete valid submissions: missing={missing}, errors={errors}, fallback={fallback}')


if __name__ == '__main__': main()
