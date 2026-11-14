"""Predicted B0/B2/B3: strict versus positive-endpoint proposals, fixed tune30."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from agent_baseline.exemplar_semantics import CLASSES
from agent_baseline.tools_t13 import Features
from .io_retry import read_text, read_image, write_json, write_text, exists
from .t3_routes import load_dev, group
from .t3_routes_cache import CachedVision, write_npz
from .t3_diag import Sam3Backend, directed_proposals
from .night_t3_retrieval import unpack, ranked, score, summaries

VARIANTS=('B0','B2','B3')


def needs(pair):
    return [] if None in pair or pair[0]==pair[1] else [(CLASSES[c],t) for c in pair for t in ('t1','t2')]


def find_cache(run, night, output, row, category, time):
    keyed=row['id']+'_'+category.replace(' ','_')
    paths=sorted((run/'t3_diag/cache/t3_sam3/sam3_text'/row['id']).glob('*.json'))
    paths += [night/'text_cache/t3_sam3/sam3_text'/keyed/f't1_{time}.json',output/'cache/t3_sam3/sam3_text'/keyed/f't1_{time}.json']
    for path in paths:
        if not exists(path) or not exists(path.with_suffix('.npz')):continue
        meta=json.loads(read_text(path))
        if (meta['request'].get('text')==category and meta['request']['query']==row['image_'+time]
            and meta['pixel_thr']==.85 and meta['score_thr']==.5 and meta['resolution']==1008):return path.with_suffix('.npz')
    return None


def inventory(rows, parses, run, night, output, variants=VARIANTS):
    cases=[];missing={}
    for row in rows:
        for variant in variants:
            pair=parses[variant][row['id']]['interpretation']['pair'];queries=[]
            for category,time in needs(pair):
                path=find_cache(run,night,output,row,category,time)
                query=dict(id=row['id'],category=category,time=time,path=str(path) if path else None)
                queries.append(query)
                if path is None:missing[(row['id'],category,time)]=query
            cases.append(dict(id=row['id'],variant=variant,predicted_pair=pair,queries=queries,
                status='unknown_or_same_class_empty' if not queries else ('missing_cache' if any(q['path'] is None for q in queries) else 'ready')))
    return dict(cases=cases,missing=list(missing.values()),unique_missing=len(missing),tasks=len(rows),variants=list(variants))


def construct(stacks, shape):
    source1,source2,target1,target2=[np.asarray(x,bool).reshape(-1,*shape) for x in stacks]
    union=lambda x:x.any(axis=0) if len(x) else np.zeros(shape,bool)
    # Stable slots retain original source-before-target identities through filtering.
    positive=np.concatenate([source1&union(target2),target2&union(source1)],axis=0)
    strict=positive&~union(source2)&~union(target1)
    ps=np.flatnonzero(positive.any(axis=(1,2)));ss=np.flatnonzero(strict.any(axis=(1,2)))
    return strict[ss],positive[ps],ss,ps


def save_summary(output, records, split='tune'):
    variants={v:dict(tasks=len(records[v]),expected=30,status='complete' if len(records[v])==30 else 'partial',
        records=records[v],summary=summaries(records[v])) for v in records}
    data=dict(variants=variants,status='complete' if all(len(records[v])==30 for v in records) else 'partial',api_calls=0,
        interpretation=f'Fixed predicted classes; strict versus positive-only candidate support. Full {split}30 denominators; no GT class prompts. No verification. Cache generation is local SAM3 only.')
    if split=='confirm':
        data['confirmation_scope']='Exploratory confirmation on multiply exposed images; B2 preselected from positive-stratum label agreement. Tune TP gain was concentrated in train1808, not established stability. No online Agent evaluation.'
    write_json(output/'summary.json',data)
    lines=['# Positive endpoint mechanism pairing','',data['interpretation'],data.get('confirmation_scope',''),'',
        '| Variant | GT | Group | N | Method | F1 | P | R | TP | FP | FN | Covered/truth |',
        '|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|']
    for variant,item in variants.items():
        for row in item['summary']:
            if row['split']!=split:continue
            for name,m in row['methods'].items():
                lines.append('| '+' | '.join(map(str,[variant,row['support'],row['group'],row['tasks'],name,
                    m['f1_c'],m['precision'],m['recall'],m['tp'],m['fp'],m['fn'],f"{m['covered_pixels']}/{m['truth_pixels']}"]))+' |')
    lines+=['','Incomplete stages are not full-cohort evidence. Positive-only can retain unchanged or ambiguous support and raise FP. All-candidate union isolates support exclusions; Top8 additionally reflects fixed delta ranking on changed supports. B2 has cross-date input information and cannot be called a pure spotlight intervention.']
    write_text(output/'report.md','\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--night',type=Path,required=True)
    p.add_argument('--mode',choices=['inventory','prepare','score'],required=True)
    p.add_argument('--split',choices=['tune','confirm'],default='tune')
    p.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS));a=p.parse_args()
    selected_variants=tuple(a.variants)
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Run through Slurm')
    out=a.night/f'positive_endpoints_{a.split}';audit=json.loads(read_text(a.night/'audit.json'))
    rows=[r for r in load_dev(a.run) if audit['split'][r['id']]==a.split];by_id={r['id']:r for r in rows}
    if len(rows)!=30:raise ValueError('Require fixed selected split30')
    parses={v:{r['id']:r for r in json.loads(read_text(a.night/v/'summary.json'))['records'] if r['split']==a.split} for v in selected_variants}
    if any(set(parses[v])!=set(by_id) for v in selected_variants):raise ValueError('Predicted parse cohorts differ')
    settings=dict(code=str(Path(__file__).resolve()),variants=list(selected_variants),split=f'fixed {a.split}30',k=8,
        strict='sourceT1_i AND union(targetT2) AND NOT union(sourceT2) AND NOT union(targetT1), plus symmetric targetT2_j candidates',
        positive_only='sourceT1_i AND union(targetT2), plus targetT2_j AND union(sourceT1)',
        selection='nonempty masks only; unknown/same predicted class empty; source-before-target stable slots; DINO delta score descending then slot order',
        primary='all-candidate union, plus fixed delta Top8; coverage and full/positive/empty target/surface',
        missing='never empty/fill by GT/omit denominator; prepare only missing exact predicted queries; score requires all caches',
        cache='original historical and night caches read-only; new SAM3 masks only in this experiment cache/',
        purpose='New mechanism test prompted by corrected semantic stratification; not confirmation-based selection',api_calls=0)
    if a.split=='confirm':
        settings['confirmation_scope']='Multiply exposed exploratory confirmation; B2 primary preselected from positive tune pair agreement and full60 availability, not B3 marginal F1. Tune TP gains all occur in train1808: motivation for validation, not stable benefit or Agent promotion.'
    if exists(out/'settings.json') and json.loads(read_text(out/'settings.json'))!=settings:raise ValueError('Settings differ')
    write_json(out/'settings.json',settings)
    plan=inventory(rows,parses,a.run,a.night,out,selected_variants)
    if not exists(out/'initial_inventory.json'):write_json(out/'initial_inventory.json',plan)
    write_json(out/'inventory.json',plan)
    if a.mode=='inventory':return
    if a.mode=='prepare':
        root=Path('/guisongxia01/pangchao/wangyihan')
        backend=Sam3Backend(SimpleNamespace(cache=out/'cache',device='cuda',sam3_python=root/'conda_envs/omniovcd-env/bin/python',
            sam3_root=root/'Chang-Agent/third_party/OmniOVCD',sam3_checkpoint=root/'models/sam3/sam3.pt',sam3_bpe=root/'Chang-Agent/third_party/OmniOVCD/sam3/assets/bpe_simple_vocab_16e6.txt.gz'))
        try:
            for index,q in enumerate(plan['missing']):
                row=by_id[q['id']];keyed=dict(row,id=row['id']+'_'+q['category'].replace(' ','_'))
                backend.for_row(keyed)(mode='sam3_text',time=q['time'],category=q['category'],reference_time='t1',box=[])
                write_json(out/'prepare_progress.json',dict(completed_this_run=index+1,requested_this_run=len(plan['missing']),last=q,api_calls=0))
                print('prepared',index+1,len(plan['missing']),q['id'],q['category'],q['time'],flush=True)
        finally:backend.close()
        write_json(out/'inventory.json',inventory(rows,parses,a.run,a.night,out,selected_variants));return
    if plan['missing']:
        write_json(out/'score_pending.json',dict(status='pending_missing_cache',missing=plan['missing'],expected_tasks_per_variant=30))
        raise RuntimeError('Missing caches: inventory.json lists exact requirements; no partial denominator scoring')
    cache=CachedVision(a.run/'t3_diag/cache/vision');records={v:[] for v in selected_variants}
    for row in rows:
        exemplar=read_image(row['exemplar_mask'])>0;shape=exemplar.shape;features=None;loaded={}
        for variant in selected_variants:
            folder=out/variant/'tasks'/row['id'];final=folder/'metrics.json'
            if exists(final):records[variant].append(json.loads(read_text(final)));continue
            pair=parses[variant][row['id']]['interpretation']['pair'];paths=[]
            if needs(pair):
                stacks=[]
                for category,time in needs(pair):
                    path=find_cache(a.run,a.night,out,row,category,time);paths.append(str(path))
                    if str(path) not in loaded:loaded[str(path)]=unpack(path)
                    stack=loaded[str(path)]
                    if stack.shape[1:]!=shape:raise ValueError('Cached mask shape differs')
                    stacks.append(stack)
                strict,positive,ss,ps=construct(stacks,shape)
            else:strict=positive=np.zeros((0,*shape),bool);ss=ps=np.zeros(0,int)
            if features is None and (len(strict) or len(positive)):
                backend=cache.for_row(row);features=[Features(backend('extract_features',time=t),shape,t) for t in ('t1','t2')]
            order_s=ranked(strict,features,exemplar) if len(strict) else [];order_p=ranked(positive,features,exemplar) if len(positive) else []
            write_npz(folder/'proposals.npz',shape=np.asarray(shape),strict=np.packbits(strict.reshape(len(strict),-1) if len(strict) else np.zeros((0,exemplar.size),bool),axis=1),
                positive_only=np.packbits(positive.reshape(len(positive),-1) if len(positive) else np.zeros((0,exemplar.size),bool),axis=1),strict_slots=ss,positive_slots=ps,
                strict_rank=np.asarray(order_s,int),positive_rank=np.asarray(order_p,int))
            # GT is loaded only after proposals and ordering are fixed and persisted.
            truth=read_image(row['gt_mask'])>0
            scores=[score(strict,list(range(len(strict))),truth,exemplar,'strict_all',0),score(positive,list(range(len(positive))),truth,exemplar,'positive_only_all',0),
                score(strict,order_s[:8],truth,exemplar,'strict_delta',8),score(positive,order_p[:8],truth,exemplar,'positive_only_delta',8)]
            reproduction=None
            if variant=='B0':
                old=json.loads(read_text(a.night/'retrieval/B0/tasks'/row['id']/'metrics.json'))
                old_counts={r['method']:r['counts'] for r in old['scores'] if r['k'] in (0,8)}
                reproduction=scores[0]['counts']==old_counts['R2_all'] and scores[2]['counts']==old_counts['R3_dino_noverify']
                write_json(folder/'baseline_reproduction.json',dict(agrees=reproduction,new=[scores[0]['counts'],scores[2]['counts']],old=[old_counts['R2_all'],old_counts['R3_dino_noverify']]))
                if not reproduction:raise ValueError('B0 strict baseline reproduction failed')
            record=dict(id=row['id'],split=a.split,group=group(row),remaining_gt_pixels=int((truth&~exemplar).sum()),variant=variant,predicted_pair=pair,
                status='unknown_or_same_class_empty' if not needs(pair) else 'complete',scores=scores,verification={},cache_sources=paths,
                original_exemplar=row['exemplar_mask'],original_truth_for_scoring_only=row['gt_mask'],b0_strict_reproduced=reproduction,api_cost_cny=0)
            write_json(final,record);records[variant].append(record);save_summary(out,records,a.split)
            print('scored',variant,len(records[variant]),row['id'],flush=True)
    save_summary(out,records,a.split)


if __name__=='__main__':main()
