"""N7 paired trigger replay. Decisions consume an explicit label-free state only."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np

THRESHOLDS = (0.01, 0.05, 0.10)
ENDPOINTS = ('balanced_v0_k10', 'gated_evidence_v0_k10')

def read(path):
    return json.loads(Path(path).read_text())

def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')

def metrics(counts):
    tp, fp, fn = (counts[k] for k in ('tp', 'fp', 'fn'))
    return dict(tp=tp, fp=fp, fn=fn, f1=200*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                precision=100*tp/(tp+fp) if tp+fp else 0., recall=100*tp/(tp+fn) if tp+fn else 0.)

def trigger(state, policy, threshold):
    if policy == 'never': return False
    if policy == 'always': return True
    if policy == 'coverage': return state['mask_coverage'] <= threshold
    if policy == 'direction': return state['direction'] in ('appeared', 'disappeared')
    raise ValueError(policy)

def pooled(rows):
    return metrics({k: sum(r[k] for r in rows) for k in ('tp', 'fp', 'fn')})

def cost(answer):
    tokens = answer.get('tokens') or {}
    usage = answer.get('usage') or {}
    ni = int(tokens.get('input', usage.get('prompt_tokens', 0)))
    no = int(tokens.get('output', usage.get('completion_tokens', 0)))
    pi, po = (.2,.8) if ni <= 32000 else ((.6,2.4) if ni <= 256000 else (1.2,4.8))
    return (ni*pi + no*po)/1e6, ni, no

def load_rows(a):
    rows = []
    for item in read(a.history/'rank_V0/manifest.json'):
        group, stem = item['group'], item['stem']
        record = read(a.history/'rank_V0/scores'/group/(stem+'.json'))
        with np.load(a.base/group/(stem+'.npz'), allow_pickle=False) as z:
            mask = z['start'].astype(bool)
        # No GT-based counts feed the trigger: prediction coverage comes from the actual mask.
        state = dict(mask_coverage=float(mask.mean()), mask_pixels=int(mask.sum()),
                     image_pixels=int(mask.size), direction=item['direction'])
        baseline = record['baseline']
        assert state['mask_pixels'] == baseline['tp'] + baseline['fp'], (group, stem)
        endpoints = dict(balanced_v0_k10=record['reference']['10'],
                         gated_evidence_v0_k10=record['rankings']['evidence']['10']['metrics'])
        assert all(v is not None for v in endpoints.values())
        accounting = {}
        old = item['original_candidate_reference']['candidates'][:10]
        ranked = [item['candidates'][i] for i in item['rankings']['evidence'][:10]]
        for endpoint, candidates in zip(ENDPOINTS, (old, ranked)):
            directory = a.pilot/'answers/four' if endpoint == ENDPOINTS[0] else a.history/'rank_V0/answers'
            answers = [read(directory/(c['uid'].replace(':','__')+'.json')) for c in candidates]
            assert all(r.get('answer') in ('yes','no') for r in answers)
            values = [cost(r) for r in answers]
            accounting[endpoint] = dict(verifier_requests=len(answers),
                token_usage_cost_cny=sum(v[0] for v in values), input_tokens=sum(v[1] for v in values),
                output_tokens=sum(v[2] for v in values))
        rows.append(dict(group=group, stem=stem, split=item['split'], state=state, baseline=baseline,
                         endpoints=endpoints, accounting=accounting,
                         post_generation_diagnostic=dict(candidate_count=item['pre_rank_count'],
                             candidate_area=item['pre_rank_area'],
                             top10_evidence=[c['evidence_score'] for c in ranked])))
    assert len(rows) == 30 and sum(r['split']=='tune' for r in rows) == 15
    return rows

def evaluate(rows, endpoint, policy, threshold):
    output=[]
    for row in rows:
        call = trigger(row['state'], policy, threshold)
        pred = row['endpoints'][endpoint] if call else row['baseline']
        output.append(dict(group=row['group'],stem=row['stem'],split=row['split'],state=row['state'],
            called=call,metrics=pred,baseline=row['baseline'],fixed=row['endpoints'][endpoint],
            delta_f1=pred['f1']-row['baseline']['f1'],delta_vs_fixed=pred['f1']-row['endpoints'][endpoint]['f1'],
            added_tp=pred['tp']-row['baseline']['tp'],added_fp=pred['fp']-row['baseline']['fp'],
            verifier_requests=row['accounting'][endpoint]['verifier_requests'] if call else 0,
            token_usage_cost_cny=row['accounting'][endpoint]['token_usage_cost_cny'] if call else 0,
            failure=('called_harm' if pred['f1']<row['baseline']['f1']-1e-9 else
                     'skipped_benefit' if not call and row['endpoints'][endpoint]['f1']>row['baseline']['f1']+1e-9 else None)))
    return output

def summarize(rows):
    n=len(rows); result=dict(images=n,metrics=pooled([r['metrics'] for r in rows]),
        macro_f1=float(np.mean([r['metrics']['f1'] for r in rows])),
        macro_delta=float(np.mean([r['delta_f1'] for r in rows])),
        macro_delta_vs_fixed=float(np.mean([r['delta_vs_fixed'] for r in rows])),
        calls=sum(r['called'] for r in rows),call_rate=sum(r['called'] for r in rows)/n,
        wins=sum(r['delta_f1']>1e-9 for r in rows),losses=sum(r['delta_f1']< -1e-9 for r in rows),
        added_tp=sum(r['added_tp'] for r in rows),added_fp=sum(r['added_fp'] for r in rows),
        verifier_requests=sum(r['verifier_requests'] for r in rows),
        token_usage_cost_cny=sum(r['token_usage_cost_cny'] for r in rows))
    result['ties']=n-result['wins']-result['losses'];result['harm_rate']=result['losses']/n
    empty=[r for r in rows if r['baseline']['tp']+r['baseline']['fn']==0]
    result['empty_gt_images']=len(empty)
    result['empty_gt_fp_pixels']=sum(r['metrics']['fp'] for r in empty)
    result['empty_gt_added_fp_pixels']=sum(r['added_fp'] for r in empty)
    result['pooled_delta']=result['metrics']['f1']-pooled([r['baseline'] for r in rows])['f1']
    result['pooled_delta_vs_fixed']=result['metrics']['f1']-pooled([r['fixed'] for r in rows])['f1']
    rng=np.random.default_rng(20261115)
    samples=rng.integers(0,n,size=(2000,n))
    for name, field in [('macro_delta_ci95','delta_f1'),('macro_delta_vs_fixed_ci95','delta_vs_fixed')]:
        values=np.array([r[field] for r in rows]); result[name]=np.percentile(values[samples].mean(axis=1),[2.5,97.5]).tolist()
    result['leave_one_out_macro_delta_range']=[min((sum(r['delta_f1'] for r in rows)-r['delta_f1'])/(n-1) for r in rows),max((sum(r['delta_f1'] for r in rows)-r['delta_f1'])/(n-1) for r in rows)] if n>1 else None
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--history',type=Path,required=True);p.add_argument('--pilot',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Run through Slurm')
    start=time.monotonic();a.out.mkdir(parents=True,exist_ok=True);rows=load_rows(a)
    tune=[r for r in rows if r['split']=='tune']
    # One threshold shared across both endpoints; select using the primary balanced endpoint only.
    candidates=[dict(threshold=t,**summarize(evaluate(tune,ENDPOINTS[0],'coverage',t))) for t in THRESHOLDS]
    chosen=max(candidates,key=lambda r:(r['macro_delta'],-r['call_rate'],-r['threshold']))['threshold']
    write(a.out/'frozen_selection.json',dict(threshold=chosen,candidates=candidates,criterion='tune macro delta; ties fewer calls then smaller threshold'))
    write(a.out/'observable_states.json',[{k:r[k] for k in ('group','stem','split','state','post_generation_diagnostic')} for r in rows])
    summaries={};all_cases=[]
    for endpoint in ENDPOINTS:
        for policy in ('never','always','coverage','direction'):
            cases=evaluate(rows,endpoint,policy,chosen)
            all_cases.extend([dict(endpoint=endpoint,policy=policy,**r) for r in cases])
            for split in ('tune','confirm'):
                selected=[r for r in cases if r['split']==split]
                for group in ('all',*sorted({r['group'] for r in selected})):
                    subset=[r for r in selected if group=='all' or r['group']==group]
                    summaries['/'.join((endpoint,policy,split,group))]=summarize(subset)
    write(a.out/'paired_cases.json',all_cases);write(a.out/'summary.json',summaries)
    lines=['# N7 find_missed trigger replay','',f'Frozen coverage threshold: {chosen}. New API requests: 0.',
        'Historical exposed development pool only; no independent generalization or online Agent claim.',
        'Verifier token costs are usage-based replay estimates, not new spending; candidate generation time/cost is unavailable and excluded.',
        'Candidate signals are post-generation diagnostics, not inputs to the call gate. C1 direction is confounded with source in this pool.','',
        '| endpoint | policy | split | group | F1/P/R | macro Δ | pooled Δ | calls | W/L/T | added TP/FP | verifier requests | replay CNY |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for key,s in summaries.items():
        m=s['metrics']; lines.append('| '+' | '.join(key.split('/')+[f"{m['f1']:.4f}/{m['precision']:.4f}/{m['recall']:.4f}",f"{s['macro_delta']:.4f}",f"{s['pooled_delta']:.4f}",f"{s['calls']}/{s['images']}",f"{s['wins']}/{s['losses']}/{s['ties']}",f"{s['added_tp']}/{s['added_fp']}",str(s['verifier_requests']),f"{s['token_usage_cost_cny']:.6f}"])+' |')
    lines += ['', 'Per-image failures and skipped benefits: paired_cases.json. Confidence intervals and leave-one-out stability: summary.json.',
              f'Elapsed replay seconds: {time.monotonic()-start:.3f}. Slurm job: {os.environ["SLURM_JOB_ID"]}.']
    (a.out/'report.md').write_text('\n'.join(lines)+'\n')
    write(a.out/'execution.json',dict(status='complete',slurm_job=os.environ['SLURM_JOB_ID'],elapsed_seconds=time.monotonic()-start,new_api_requests=0))
    print(json.dumps({'status':'complete','threshold':chosen,'out':str(a.out)}))

if __name__=='__main__':main()
