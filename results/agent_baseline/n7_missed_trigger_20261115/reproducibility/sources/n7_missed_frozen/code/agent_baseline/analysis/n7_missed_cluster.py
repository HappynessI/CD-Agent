"""Supplementary image-cluster uncertainty; preserves first replay artifacts."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from .n7_missed_trigger import read, write, pooled

def cluster_id(row):
    return row['group'].split('_')[0] + '/' + row['stem']

def estimate(rows):
    result=dict(macro_delta=float(np.mean([r['delta_f1'] for r in rows])),
        macro_vs_fixed=float(np.mean([r['delta_vs_fixed'] for r in rows])),
        pooled_delta=pooled([r['metrics'] for r in rows])['f1']-pooled([r['baseline'] for r in rows])['f1'],
        pooled_vs_fixed=pooled([r['metrics'] for r in rows])['f1']-pooled([r['fixed'] for r in rows])['f1'])
    if all('coverage_metrics' in r for r in rows):
        result['macro_vs_coverage']=float(np.mean([r['delta_vs_coverage'] for r in rows]))
        result['pooled_vs_coverage']=pooled([r['metrics'] for r in rows])['f1']-pooled([r['coverage_metrics'] for r in rows])['f1']
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Use Slurm')
    rows=read(a.out/'paired_cases.json');result={}
    for endpoint in sorted({r['endpoint'] for r in rows}):
        for policy in sorted({r['policy'] for r in rows}):
            for split in ('tune','confirm'):
                selected=[r for r in rows if r['endpoint']==endpoint and r['policy']==policy and r['split']==split]
                ids=sorted({cluster_id(r) for r in selected})
                clusters={i:[r for r in selected if cluster_id(r)==i] for i in ids}
                rng=np.random.default_rng(20261115);samples=[]
                for indexes in rng.integers(0,len(ids),size=(2000,len(ids))):
                    samples.append(estimate([r for i in indexes for r in clusters[ids[i]]]))
                point=estimate(selected)
                result['/'.join((endpoint,policy,split))]=dict(tasks=len(selected),images=len(ids),point=point,
                    cluster_bootstrap95={k:np.percentile([s[k] for s in samples],[2.5,97.5]).tolist() for k in point},
                    leave_one_image_out=[dict(omitted=i,**estimate([r for r in selected if cluster_id(r)!=i])) for i in ids])
    write(a.out/'image_cluster_sensitivity.json',result)
    write(a.out/'cluster_execution.json',dict(status='complete',slurm_job=os.environ['SLURM_JOB_ID']))
    print(json.dumps({'status':'complete','out':str(a.out/'image_cluster_sensitivity.json')}))

if __name__=='__main__':main()
