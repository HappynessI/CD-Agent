"""Fixed-pool cached DINO endpoint/delta ablation, with exemplar-date swap control."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from agent_baseline.tools_t13 import Features,pooled,pool_features,cosine
from agent_baseline.instances import from_dense
from .t3_routes import load_dev,group,GRID
from .t3_routes_cache import CachedVision
from .t3_diag import proposal_metrics
from .io_retry import read_text,read_image,write_json,write_text,exists
from .night_t3_retrieval import score,summaries


def representations(candidate1,candidate2,example1,example2):
    result={}
    for suffix,e1,e2 in [('',example1,example2),('_example_swapped',example2,example1)]:
        delta=cosine(candidate2-candidate1,e2-e1)
        endpoint=.5*(cosine(candidate1,e1)+cosine(candidate2,e2))
        result['delta'+suffix]=delta
        result['endpoint'+suffix]=endpoint
        result['equal_joint'+suffix]=.5*(delta+endpoint)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--night',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Run cached feature computation through Slurm')
    audit=json.loads(read_text(a.night/'audit.json'));out=a.night/'endpoint';records=[]
    write_json(out/'settings.json',dict(code=str(Path(__file__).resolve()),pool='original cached SAM T1 followed by T2, no geometry or semantic gate',
        endpoint='0.5*(cos(candidateT1,exampleT1)+cos(candidateT2,exampleT2))',joint='0.5*(delta+endpoint)',
        counterfactual='swap exemplar dates only; candidates and query dates unchanged; diagnostic ranking sensitivity, not new ground truth',
        selection='fixed coefficients and TopK; no confirm-based choice; original split and all60 retained',api_cost_cny=0))
    cache=CachedVision(a.run/'t3_diag/cache/vision')
    for row in load_dev(a.run):
        target=out/'tasks'/row['id']/'metrics.json'
        if exists(target):records.append(json.loads(read_text(target)));continue
        ex=read_image(row['exemplar_mask'])>0;backend=cache.for_row(row)
        masks=np.concatenate([backend('segment_everything',time=t) for t in ('t1','t2')])
        features=[Features(backend('extract_features',time=t),ex.shape,t) for t in ('t1','t2')]
        items=from_dense(masks,np.ones(len(masks)));indices=np.flatnonzero(ex)
        candidates=[pooled(f,items) for f in features];examples=[pool_features(f,indices) for f in features]
        values=representations(*candidates,*examples)
        orders={name:sorted(range(len(masks)),key=lambda i:(-float(v[i]),i)) for name,v in values.items()}
        truth=read_image(row['gt_mask'])>0
        scores=[score(masks,order[:k],truth,ex,name,k) for name,order in orders.items() for k in GRID]
        upper=proposal_metrics(masks,truth,ex)
        record=dict(id=row['id'],split=audit['split'][row['id']],group=group(row),remaining_gt_pixels=int((truth&~ex).sum()),
            fixed_pool_upper=upper,scores=scores,verification={},
            swap_topk_overlap={name:{str(k):len(set(orders[name][:k])&set(orders[name+'_example_swapped'][:k])) for k in GRID} for name in ['delta','endpoint','equal_joint']})
        write_json(target,record);records.append(record)
        write_json(out/'progress.json',dict(tasks=len(records),api_cost_cny=0));print('endpoint',len(records),row['id'],flush=True)
    summary=summaries(records)
    write_json(out/'summary.json',dict(tasks=len(records),records=records,summary=summary,api_cost_cny=0))
    lines=['# Fixed-pool DINO representation diagnostic','',
        'All methods use the original SAM pool. Endpoint and delta weights are fixed; no confirmation-based selection. Swapping only exemplar dates tests ranking sensitivity, not correctness under a new task. Label/GT is used only for offline scoring.','',
        '| Split | GT stratum | Group | N | Method@K | F1 |','|---|---|---|---:|---|---:|']
    for row in summary:
        for method,metrics in row['methods'].items():
            if method.endswith('@8'):lines.append(f"| {row['split']} | {row['support']} | {row['group']} | {row['tasks']} | {method} | {metrics['f1_c']:.4f} |")
    lines+=['','Full TopK, P/R/counts, fixed-pool instance/pixel upper bounds and per-task swap overlap are in summary.json. API cost: 0.']
    write_text(out/'report.md','\n'.join(lines)+'\n')


if __name__=='__main__':main()
