"""Compile actual night T3 evidence, keeping incomplete stages explicit (Slurm only)."""
import argparse
import json
import os
from pathlib import Path
from .io_retry import read_text,write_json,write_text,exists
from .night_t3_retrieval import summaries


def load(path):
    return json.loads(read_text(path)) if exists(path) else None


def request_cost(night):
    # Disjoint roots; B0 is reused historical evidence, never rebilled here.
    entries={}
    for name in ('B1','B2','B3','t33','retrieval','r1_direct4'):
        for pattern in ('**/answer.json','**/failure_*.json'):
            for path in (night/name).glob(pattern):
                entries[str(path.resolve())]=float(load(path).get('cost_cny',0))
    return dict(cost_cny=sum(entries.values()),request_artifacts=len(entries),entries=entries,
        rule='sum each unique successful answer or billed failed attempt artifact once; exclude B0 reuse and all summary/attempt mirrors')


def delta_reproduction(records,old):
    prior={r['id']:r for r in old['per_task']};result=[]
    for row in records:
        historical={(r['method'],r['k']):r['counts'] for r in prior[row['id']]['records']}
        for score in row['scores']:
            if score['method']=='delta':
                original=historical.get(('dino',score['k']))
                result.append(dict(id=row['id'],k=score['k'],new=score['counts'],historical=original,
                    agrees=original is not None and score['counts']==original))
    return dict(tasks=len(records),comparisons=len(result),all_available_agree=bool(result) and all(r['agrees'] for r in result),
        mismatches=[r for r in result if not r['agrees']],records=result)


def collect(night,old):
    audit=load(night/'audit.json');stages={};semantics={}
    for name,expected in [('B0',60),('B1',60),('B2',60),('B3',30),('t33/B0',12),('t33/B1',12),('t33/B2',12)]:
        value=load(night/name/'summary.json');count=value['tasks'] if value else 0
        stages[name]=dict(completed=count,expected=expected,status='complete' if count>=expected else 'pending')
        if value:semantics[name]=value
    stages['B3/confirm']=dict(status='not_promoted',reason='tune pair agreement6/30 below B0 8/30; no confirm run justified')
    retrieval={}
    for prefix in ['retrieval','t33/retrieval']:
        for variant in ['B0','B1','B2','B0_local_raw']:
            folder=night/prefix/variant
            records=[load(p) for p in sorted((folder/'tasks').glob('*/metrics.json'))]
            if records:retrieval[f'{prefix}/{variant}']=dict(tasks=len(records),records=records,summary=summaries(records))
            if variant=='B0':
                for split,expected in ([('tune',30),('confirm',30)] if prefix=='retrieval' else [('tune',6),('confirm',6)]):
                    count=sum(r['split']==split for r in records)
                    stages[f'{prefix}/{variant}/{split}']=dict(completed=count,expected=expected,status='complete' if count==expected else 'pending')
    direct=[]
    for split in ('tune','confirm'):
        current=[load(p) for p in sorted((night/'r1_direct4'/split/'tasks').glob('*/metrics.json'))]
        direct.extend(current)
        stages['r1_direct4/'+split]=dict(completed=len(current),expected=30,status='complete' if len(current)==30 else ('pending' if split=='tune' else 'not_promoted_or_partial'))
    if direct:retrieval['r1_direct4']=dict(tasks=len(direct),records=direct,summary=summaries(direct))
    raw_rows=retrieval.get('retrieval/B0_local_raw',{}).get('records',[])
    stages['local_raw/tune']=dict(completed=sum(r['split']=='tune' for r in raw_rows),expected=30,status='not_submitted_zero_TP_headroom' if not raw_rows else ('complete' if sum(r['split']=='tune' for r in raw_rows)==30 else 'partial'))
    stages['local_raw/confirm']=dict(completed=sum(r['split']=='confirm' for r in raw_rows),expected=30,status='not_requested_without_tune_evidence' if not any(r['split']=='confirm' for r in raw_rows) else 'started')
    oracle=load(night/'oracle_cache/summary.json')
    stages['oracle_cache']=dict(completed=oracle['tasks'] if oracle else 0,expected=60,status='complete' if oracle and oracle['tasks']==60 else 'pending')
    preparation=[]
    for base,pattern in [('retrieval','*/prepare_progress_*.json'),('t33/retrieval','*/prepare_progress.json')]:
        for p in (night/base).glob(pattern):preparation.append(dict(path=str(p),**load(p)))
    endpoint=[load(p) for p in sorted((night/'endpoint/tasks').glob('*/metrics.json'))]
    stages['endpoint']=dict(completed=len(endpoint),expected=60,status='complete' if len(endpoint)==60 else 'pending')
    return dict(stages=stages,audit=audit,semantic_label_agreement=semantics,retrieval=retrieval,preparation=preparation,
        endpoint=dict(records=endpoint,summary=summaries(endpoint)),delta_reproduction=delta_reproduction(endpoint,old),
        manual_review=load(night/'grounding_review.json'),oracle_cache_diagnostic=load(night/'oracle_cache/summary.json'),cost=request_cost(night))


def pool_bounds(data):
    output=[]
    sets={'endpoint':data['endpoint']['records']}
    sets.update({k:v['records'] for k,v in data['retrieval'].items()})
    if data['audit']:sets['old_dev60']=[dict(r,split=data['audit']['split'][r['id']]) for r in data['audit']['records']]
    for name,records in sets.items():
        for split in ('all','tune','confirm'):
            for support in ('all','positive','empty'):
                for group in ('全部','目标类','地表类'):
                    selected=[r for r in records if (split=='all' or r['split']==split) and (group=='全部' or r['group']==group)
                        and (support=='all' or (r['remaining_gt_pixels']>0)==(support=='positive'))]
                    for key in ('fixed_pool_upper','proposal_upper','text_proposal_upper','geometry_proposal_upper'):
                        bounds=[r[key] for r in selected if key in r]
                        if not bounds:continue
                        counts={k:sum(v[k] for v in bounds) for k in ('truth_pixels','covered_pixels','gt_instances','aligned_instances')}
                        output.append(dict(stage=name,pool=key,split=split,support=support,group=group,tasks=len(bounds),**counts))
    return output


def number(value):
    return 'NA' if value is None else f'{value:.4f}'


def readable_stage_results(data):
    lines=['','## Other completed diagnostics and pending evidence','']
    choose=lambda rows,support:next(r for r in rows if r['split']=='tune' and r['support']==support and r['group']=='全部')
    cross=data['retrieval'].get('t33/retrieval/B0')
    if cross and cross['tasks']==12:
        tune=choose(cross['summary'],'all');methods=tune['methods']
        lines.append('T3.3 fixed12 retrieval is complete. Tuning6 Top8 F1: '+', '.join(f"{name}={methods[name]['f1_c']:.3f}" for name in ['R1_geometry_dino@8','R2_native_order@8','R3_verified@8'])+'. These use different proposal pools; only R3 before/after verification shares its fixed text pool. Cross-image label agreement and ambiguous source crops remain separate from achieved pixel scores.')
    else:lines.append(f"T3.3 actual retrieval remains pending/partial ({cross['tasks'] if cross else 0}/12); GPU preparation is not completed evaluation.")
    endpoint=data['endpoint']
    if len(endpoint['records'])==60:
        for support in ('all','positive','empty'):
            row=choose(endpoint['summary'],support)
            lines.append('Endpoint fixed-pool tune '+support+' Top8 F1: '+', '.join(f"{name}={row['methods'][name+'@8']['f1_c']:.3f}" for name in ['delta','endpoint','equal_joint'])+'.')
        lines.append('Endpoint comparisons alone share the full original SAM pool; R1 geometry-filtered values are not a same-pool scoring contrast. Example-date swaps report ranking sensitivity, not new truth or a causal mechanism claim. No coefficient was fitted on confirmation.')
    else:lines.append(f"Endpoint result remains pending/partial ({len(endpoint['records'])}/60); do not select a scoring formula from partial outputs.")
    oracle=data.get('oracle_cache_diagnostic')
    if oracle and oracle['tasks']==60:
        row=choose(oracle['summary'],'positive');old=row['methods']['historical_oracle'];pred=row['methods']['predicted']
        records=[r for r in oracle['records'] if r['split']=='tune' and r['remaining_gt_pixels']>0]
        complete=[r['constraint_diagnostics']['historical_oracle'] for r in records if r['constraint_diagnostics']['historical_oracle']['status']=='complete']
        intersection=sum(r['positive_intersection'] for r in complete);strict=sum(r['strict_loss_and_gain'] for r in complete)
        missing=sum(r['positive_intersection']==0 for r in complete)
        lines.append(f"Historical oracle tune positives cover {old['covered_pixels']}/{old['truth_pixels']} pixels (F1 {old['f1_c']:.4f}), versus predicted text {pred['covered_pixels']}/{pred['truth_pixels']}. Even GT classes do not resolve coverage. Of {len(complete)} cache-complete positive cases, {missing} have zero sourceT1/targetT2 intersection; intersection coverage {intersection} drops to {strict} after strict exclusions. Per-task evidence identifies missing endpoint proposals separately from loss/gain suppression. These oracle masks are historical, not new GT-fed API calls.")
    else:lines.append('Historical oracle/cache constraint diagnosis is pending; no model fill-in is allowed.')
    direct=data['retrieval'].get('r1_direct4')
    if direct and sum(r['split']=='tune' for r in direct['records'])==30:
        row=choose(direct['summary'],'positive');m=row['methods'];base=m['R1_geometry_dino@8'];new=m['R1_direct4_verified@8']
        lines.append(f"R1-direct4 tune positive F1 {base['f1_c']:.3f} → {new['f1_c']:.3f}; TP {base['tp']} → {new['tp']}, FP {base['fp']} → {new['fp']}. Judge positive-target preservation, empty-task behavior and manual response evidence together before any confirmation promotion; no single mechanical F1 cutoff decides it.")
    else:lines.append(f"R1-direct4 remains pending/partial ({sum(r['split']=='tune' for r in direct['records']) if direct else 0}/30 tune). Unlike the cancelled text-pool raw arm, it keeps the R1 pool with positive-target coverage.")
    return lines


def markdown(data):
    lines=['# T3 night actual evidence','',
        'Label agreement is not manual semantic or support-grounding accuracy. The existing dev60 was historically inspected; the new image split isolates new method selection, not unseen data. All original tasks remain, including empty remaining GT.','',
        'B1/B2 change support rendering and prompts jointly; B2 also adds both dates. B3 uses four aligned RGB/binary-support panels. B3 tune did not improve B0, so confirmation was not promoted.','',
        'B0 R1/R2/R3 were predeclared fixed full60 diagnostics, reported by tune/confirm; confirmation was not triggered by a tuning win. No B1/B2/B3 promotion or confirm-based winner selection is implied.','',
        'R1 geometry-DINO changes the pool relative to unfiltered historical DINO. R2 uses predicted classes, including wrong/unknown predictions; native order is filtered SAM query order with source proposals before target, not confidence ranking or random ranking. R3 local verification is a joint input/prompt change. Unparsed answers follow historical keep policy; drop-unparsed results are sensitivity diagnostics.','',
        'Raw-contour tune30 was prepared but not submitted: the completed B0 R2 pool covers zero GT pixels in fixed tune positives, so a verifier cannot recover TP from this pool. Historical GT-text oracle is reused offline in oracle_cache/report.md to separate semantic and proposal-constraint limits, without new GT-fed model requests.','',
        'Observed local-verifier negative case: train_1136_5to1 candidates0/1 are rejected because the model says the same rectangular building shape persists. Actual inputs show a blue roof then green vegetation; the fixed spotlight support creates the same bright silhouette on both dates. This is evidence of a possible support-shape/object-shape confusion, not proof of causal attribution. This task has only3 remaining GT pixels, so a beneficial rejection score would not establish correct visual reasoning. Inputs/responses are archived under retrieval/B0/tasks/train_1136_5to1/verification.','',
        'R1-direct4 is a distinct planned tune-first experiment: same fixed R1 Top8 versus no verification, four local raw-RGB Example/Candidate date views, no predicted class pair. R1 eligibility retains candidate overlap with the example; only scoring removes exemplar pixels. No hidden overlap filter is introduced. Old eight-view original-DINO and new two-view text-pool routes are non-paired contextual references.','',
        'Endpoint methods share the original SAM pool with fixed equal weights. Swapping exemplar dates measures ranking sensitivity only; it does not define new GT.','',
        'T3.3 manual uncertainty: the fixed first tune example train_2130_T33_2to4 is clipped at the source crop bottom; close/wide views nearly coincide. Its post-date tan/gray-green support can resemble bare ground or turbid water. Treat the water-label mismatch as requiring visual/label review, not automatically a pure model error. No labels were changed.','',
        '| Stage | Completed/expected | Status |','|---|---:|---|']
    for name,row in data['stages'].items():lines.append(f"| {name} | {row.get('completed','—')}/{row.get('expected','—')} | {row['status']} |")
    actual=data['retrieval'].get('retrieval/B0')
    if actual and actual['tasks']==60 and data['audit']:
        chosen=lambda rows,support:next(r for r in rows if r['split']=='tune' and r['support']==support and r['group']=='全部')
        prior=chosen(data['audit']['strata'],'all')['methods']['dino@8']
        current=chosen(actual['summary'],'all')['methods'];geometry=current['R1_geometry_dino@8'];text=current['R2_all@0'];verified=current['R3_verified@8']
        positive=chosen(actual['summary'],'positive');empty=chosen(actual['summary'],'empty')
        lines+=['','## Decisions from tuning evidence only','',
            f"Predeclared geometry filtering changes tune Top8 F1 from {prior['f1_c']:.3f} to {geometry['f1_c']:.3f}, TP from {prior['tp']} to {geometry['tp']}, and FP from {prior['fp']} to {geometry['fp']}. Recall remains {geometry['recall']:.3f}%; this supports keeping it as a low-cost baseline, not a solved-task claim.",
            '',f"Predicted text proposals cover {text['tp']} tuning GT pixels. Local verification changes FP from {text['fp']} to {verified['fp']}, but its empty-prediction rate on positive tasks is {positive['methods']['R3_verified@8']['empty_prediction_rate']:.3f}, versus negative accuracy {empty['methods']['R3_verified@8']['empty_prediction_rate']:.3f} on empty tasks. Rejection alone is not reliable recovery.",
            '', 'Do not promote the text bridge or run the planned whole tune raw-view arm when its proposal pool has zero TP headroom. First reuse historical oracle/cache evidence to separate parsing from proposal constraints; evaluate the fixed endpoint alternatives using tune positive-target evidence before considering any method choice. Never select weights or methods from confirmation outcomes.']
    lines+=readable_stage_results(data)
    lines+=['','## Label agreement only','','| Variant | Split | Group | N | Before | After | Pair |','|---|---|---|---:|---:|---:|---:|']
    for name,value in data['semantic_label_agreement'].items():
        for row in value['summary']:
            if row['holes']=='all':lines.append('| '+' | '.join(map(str,[name,row['split'],row['group'],row['tasks'],number(row['before_label_agreement']),number(row['after_label_agreement']),number(row['pair_label_agreement'])]))+' |')
    sets={'old_dev60':data['audit']['strata'] if data['audit'] else [],'endpoint':data['endpoint']['summary']}
    sets.update({name:value['summary'] for name,value in data['retrieval'].items()})
    lines+=['','## Top8 pixel results','','Only completed task artifacts are pooled below; stage counts above identify partial runs. Full TopK and per-task counts remain in report.json. Empty-GT F1 is not a rejection measure; empty prediction rate equals negative accuracy on that stratum.','',
        '| Stage | Split | GT | Group | N | Method | F1 | P | R | Empty prediction | Pixel coverage | Instance coverage |','|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for name,rows in sets.items():
        for row in rows:
            if row['support'] not in ('all','positive','empty'):continue
            for method,value in row['methods'].items():
                if not method.endswith('@8'):continue
                coverage=value.get('covered_pixels',0)/value['truth_pixels'] if value.get('truth_pixels') else None
                instances=value.get('aligned_instances',0)/value['gt_instances'] if value.get('gt_instances') else None
                cells=[name,row['split'],row['support'],row['group'],row['tasks'],method,*[number(value.get(k)) for k in ['f1_c','precision','recall','empty_prediction_rate']],number(coverage),number(instances)]
                lines.append('| '+' | '.join(map(str,cells))+' |')
    lines+=['','## Candidate-pool upper bounds','','| Stage/pool | Split | GT | Group | N | Pixel coverage | Instance coverage |','|---|---|---|---|---:|---:|---:|']
    for row in pool_bounds(data):
        pixel=row['covered_pixels']/row['truth_pixels'] if row['truth_pixels'] else None
        instance=row['aligned_instances']/row['gt_instances'] if row['gt_instances'] else None
        lines.append('| '+' | '.join(map(str,[row['stage']+'/'+row['pool'],row['split'],row['support'],row['group'],row['tasks'],number(pixel),number(instance)]))+' |')
    oracle=data.get('oracle_cache_diagnostic')
    if oracle:
        lines+=['','## Historical GT-text oracle (offline only)','','| Split | GT | Group | N | Method | F1 | P | R | Covered/GT pixels |','|---|---|---|---:|---|---:|---:|---:|---|']
        for row in oracle['summary']:
            for method,v in row['methods'].items():
                lines.append('| '+' | '.join(map(str,[row['split'],row['support'],row['group'],row['tasks'],method,number(v['f1_c']),number(v['precision']),number(v['recall']),str(v['covered_pixels'])+'/'+str(v['truth_pixels'])]))+' |')
    reproduction=data['delta_reproduction'];cost=data['cost']
    lines+=['','## Reproduction, manual evidence and accounting','',
        f"Historical DINO reproduction: {reproduction['tasks']} completed tasks, {reproduction['comparisons']} TopK comparisons; all available counts agree: {reproduction['all_available_agree']}; mismatches: {len(reproduction['mismatches'])}. This is not a full60 claim until all60 finish.",
        '',f"New cumulative API cost: CNY {cost['cost_cny']:.7f}, from {cost['request_artifacts']} unique request artifacts. Historical B0 is reused without rebilling; summary totals are not added.",
        '', 'The complete human review is embedded in report.json and remains in grounding_review.json/md. It distinguishes wrong support localization, uncertain visual states and class-label disagreement. Fixed-pool upper bounds, per-task coverage, counterfactual rank overlap and unparsed diagnostics remain separately recorded; none are actual achieved Agent performance.']
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--night',type=Path,required=True);p.add_argument('--old',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Generate aggregate report through Slurm')
    data=collect(a.night,load(a.old));data['pool_upper_bounds']=pool_bounds(data);write_json(a.output/'report.json',data);write_text(a.output/'report.md',markdown(data))


if __name__=='__main__':main()
