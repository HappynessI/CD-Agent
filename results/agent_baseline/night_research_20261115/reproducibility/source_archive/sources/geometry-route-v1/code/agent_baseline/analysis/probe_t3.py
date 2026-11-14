"""GPU T3.1 proposal retrieval with change-vector exemplars, no API.

4 CPUs, 24 GB RAM, >=16 GB GPU; 0.5-2 h/split estimate, reuses T1.3 cache.
IoU >= .5, one-to-one greedy match in descending score order. Duplicate hits
are false positives. AP = sum precision at each first true hit / #GT instances.
Also reports retrieval excluding exemplar to expose trivial self retrieval.
"""
import json
from pathlib import Path
import numpy as np
from agent_baseline.backends_t13 import WorkerPool
from agent_baseline.tools_t13 import Features, pooled, pool_features, cosine
from agent_baseline.instances import from_dense
from .make_t3_tasks import make_tasks, MIN_AREA, SEED
from .probe_t13_common import backend_config
from .second_labels import load_labels
from .probe_t13_common import parser, selected, pair_backend, emit, shard_path, inventories, collect, fmt, CLASSES


K_VALUES = (1,5,10,20,50)
IOU_THRESHOLD = .5


def retrieval(masks,scores,truths,iou_thr=IOU_THRESHOLD,exclude=None):
    truths = [np.asarray(t,np.int64) for i,t in enumerate(truths) if i!=exclude]
    order = sorted(range(len(scores)),key=lambda i:(-float(scores[i]),i))
    matched = set()
    hits,precision = [],[]
    for rank,i in enumerate(order,1):
        idx = np.flatnonzero(masks[i])
        overlaps = [(len(np.intersect1d(idx,t,assume_unique=True)),j) for j,t in enumerate(truths) if j not in matched]
        candidates = [(inter/(len(idx)+len(truths[j])-inter),j) for inter,j in overlaps if inter]
        best = max(candidates,default=(0.,-1))
        hit = best[0] >= iou_thr
        if hit:
            matched.add(best[1])
            precision.append(len(matched)/rank)
        hits.append(int(hit))
    n = len(truths)
    result = dict(gt_instances=n,proposals=len(masks),recall_all=len(matched)/n if n else None,
                  ap=sum(precision)/n if n else None,
                  recall_at_k={str(k):sum(hits[:k])/n if n else None for k in K_VALUES})
    return result


def task_scores(masks,a,b,task):
    rows = from_dense(masks,np.ones(len(masks)))
    difference = pooled(b,rows)-pooled(a,rows)
    example = task['instances'][task['exemplar_index']]
    vector = pool_features(b,example)-pool_features(a,example)
    return cosine(difference,vector)


def summary(rows):
    tasks = [task for row in rows.values() for task in row['tasks']]
    counts = {}
    for task in tasks:
        counts[task['transition']] = counts.get(task['transition'],0)+1
    def average(items):
        def mean(values):
            values = [v for v in values if v is not None]
            return float(np.mean(values)) if values else None
        return dict(ap=mean([r['ap'] for r in items]),recall_all=mean([r['recall_all'] for r in items]),
            recall_at_k={str(k):mean([r['recall_at_k'][str(k)] for r in items]) for k in K_VALUES})
    return dict(tasks=len(tasks),counts=counts,
        including_exemplar=average([t['including_exemplar'] for t in tasks]),
        excluding_exemplar=average([t['excluding_exemplar'] for t in tasks]),
        per_transition={transition:average([t['excluding_exemplar'] for t in tasks if t['transition']==transition]) for transition in counts})


def main():
    p = parser(__doc__)
    p.add_argument('--settings',type=Path,help='Complete development probe; required for val')
    p.add_argument('--merge',type=Path,nargs='+')
    a = p.parse_args()
    names = selected(a)
    settings = dict(seed=SEED,min_area=MIN_AREA,iou_threshold=IOU_THRESHOLD,k_values=list(K_VALUES))
    frozen = None
    if a.phase=='val':
        if not a.settings:
            p.error('val requires complete development --settings')
        frozen = json.loads(a.settings.read_text())
        if frozen['phase']!='dev' or not frozen['complete'] or frozen['names']!=inventories(a.run_root)['dev'] or frozen['settings']!=settings:
            raise ValueError('Wrong/incomplete development settings')
    config = backend_config(a,frozen)
    if a.merge:
        reports,rows = collect(a.merge,a.phase,inventories(a.run_root)[a.phase])
        if any(r['settings']!=settings for r in reports):
            raise ValueError('Shard settings differ')
        if any(r['backend_config']!=config for r in reports):
            raise ValueError('Shard backend configurations differ')
        names,complete = inventories(a.run_root)[a.phase],True
    else:
        pool = WorkerPool(config,a.cache)
        rows = {}
        try:
            for name in names:
                labels = load_labels(a.dataset_root,name)
                tasks = make_tasks(*labels,name)
                backend = pair_backend(pool,a,name)
                masks = np.concatenate([backend('segment_everything',time=t) for t in ('t1','t2')],axis=0)
                f1,f2 = [Features(backend('extract_features',time=t),labels[0].shape,t) for t in ('t1','t2')]
                result = []
                for task in tasks:
                    scores = task_scores(masks,f1,f2,task)
                    result.append(dict(id=task['id'],transition=f"{CLASSES[task['source']]} -> {CLASSES[task['target']]}",
                        exemplar_index=task['exemplar_index'],
                        including_exemplar=retrieval(masks,scores,task['instances']),
                        excluding_exemplar=retrieval(masks,scores,task['instances'],exclude=task['exemplar_index'])))
                rows[name] = dict(tasks=result)
                print(name,flush=True)
        finally:
            pool.close()
        complete = a.shards==1
    report = dict(backend_config=config,phase=a.phase,names=names,complete=complete,settings=settings,per_image=rows,summary=summary(rows),
        matching='score descending, stable proposal-index ties; greedy one-to-one IoU >= .5; duplicates are FP; AP sum precision at true hit / GT count; macro average per task; report with and without exemplar')
    lines = [report['matching'],'','| 范围 | AP | R@1 | R@5 | R@10 | R@20 | R@50 | R全部 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for key in ('including_exemplar','excluding_exemplar'):
        r = report['summary'][key]
        lines.append('| '+' | '.join([key,fmt(r['ap']),*(fmt(r['recall_at_k'][str(k)]) for k in K_VALUES),fmt(r['recall_all'])])+' |')
    lines += ['', '| 转换 | 任务数 |','|---|---:|']+[f'| {k} | {v} |' for k,v in sorted(report['summary']['counts'].items())]
    emit(shard_path(a,'probe_t3'),report,'\n'.join(lines))


if __name__=='__main__':
    main()
