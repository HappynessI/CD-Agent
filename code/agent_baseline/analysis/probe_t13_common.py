"""Shared offline probe I/O and metrics; never imported by planner tools."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .second_labels import DATASET, load_labels, building_truth, pixel_counts

RUN = Path('/guisongxia01/pangchao/wangyihan/wyh/exp6_staging/agent_baseline_run')
RESULTS = Path(__file__).resolve().parents[3]/'results/agent_baseline/data'
CLASSES = {1:'low vegetation',2:'non-vegetated ground',3:'tree',4:'water',5:'building',6:'playground'}


def binary_truth(a,b):
    building_truth(a,b)  # common encoding/support validation
    return (a != b) & (a != 0) & (b != 0)


def metrics(counts):
    tp,fp,fn = map(int,counts)
    ratio = lambda a,b: a/b if b else None
    return dict(tp=tp,fp=fp,fn=fn,precision=ratio(tp,tp+fp),recall=ratio(tp,tp+fn),
                f1=ratio(2*tp,2*tp+fp+fn),iou=ratio(tp,tp+fp+fn))


def annotated_counts(pred,label,class_id):
    valid = label != 0
    return pixel_counts(np.asarray(pred,bool)&valid,(label==class_id)&valid)


def proposal_union(masks,shape):
    return np.any(masks,axis=0) if len(masks) else np.zeros(shape,bool)


def coverage_upper(masks,truth):
    """Pixel oracle: allow cropping proposals to truth; optimistic for whole-mask selection."""
    covered = int((proposal_union(masks,truth.shape)&truth).sum())
    total = int(truth.sum())
    return dict(covered=covered,truth_pixels=total,coverage=covered/total if total else None,
                pixel_oracle_f1_upper=2*covered/(total+covered) if total else None)


def selection_upper(masks,truth,time_limit=60):
    """Whole-proposal union oracle, MILP fractional optimisation with certified bounds.

    Pixel membership patterns are atoms. z selects masks, y is the exact OR on
    each atom. Dinkelbach solves 2TP-r*(predicted+truth). A time-limited dual
    residual gives a valid F1 upper bound; no heuristic is labelled an upper bound.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix
    import time
    relaxed = coverage_upper(masks,truth)['pixel_oracle_f1_upper']
    total = int(truth.sum())
    if not total:
        return dict(lower=1.,upper=1.,exact=True,status='empty truth: select empty')
    if not len(masks):
        return dict(lower=0.,upper=0.,exact=True,status='no proposals')
    n = len(masks)
    bits = np.packbits(np.asarray(masks,bool).reshape(n,-1).T,axis=1)
    atoms,inverse,area = np.unique(bits,axis=0,return_inverse=True,return_counts=True)
    membership = np.unpackbits(atoms,axis=1,count=n).astype(bool)
    tp = np.bincount(inverse,weights=truth.ravel(),minlength=len(atoms))
    used = membership.any(axis=1)
    membership,area,tp = membership[used],area[used],tp[used]
    k = len(area)
    ai,pj = np.nonzero(membership)
    edges = len(ai)
    # z_j - y_i <= 0 for each edge; y_i - sum z_j <= 0 for each atom.
    rr = np.concatenate([np.arange(edges),np.arange(edges),edges+np.arange(k),edges+ai])
    cc = np.concatenate([pj,n+ai,n+np.arange(k),pj])
    vv = np.concatenate([np.ones(edges),-np.ones(edges),np.ones(k),-np.ones(edges)])
    matrix = coo_matrix((vv,(rr,cc)),shape=(edges+k,n+k)).tocsc()
    constraint = LinearConstraint(matrix,-np.inf,0)
    lower,upper = 0.,float(relaxed)
    started = time.monotonic()
    status = 'time limit'
    while upper-lower > 1e-6:
        remaining = time_limit-(time.monotonic()-started)
        if remaining <= 0:
            break
        r = lower
        objective = np.r_[np.zeros(n),r*area-2*tp]
        solved = milp(objective,integrality=np.r_[np.ones(n),np.zeros(k)],bounds=Bounds(0,1),
                      constraints=constraint,options=dict(time_limit=remaining,mip_rel_gap=0.))
        if solved.x is not None:
            selected = solved.x[:n] > .5
            union = membership[:,selected].any(axis=1)
            lower = max(lower,float(2*tp[union].sum()/(total+area[union].sum())))
        dual_bound = getattr(solved,'mip_dual_bound',None)
        if dual_bound is not None and np.isfinite(dual_bound):
            residual_upper = -float(dual_bound)-r*total
            upper = max(lower,min(upper,r+max(0.,residual_upper)/total))
        if solved.status not in (0,1):
            raise RuntimeError('Proposal oracle solver failed: '+solved.message)
        if solved.status == 1:
            break
        status = 'optimal'
    return dict(lower=lower,upper=upper,exact=upper-lower <= 1e-6,status=status)


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--dataset-root',type=Path,default=DATASET)
    p.add_argument('--run-root',type=Path,default=RUN)
    p.add_argument('--output-root',type=Path,default=RESULTS)
    p.add_argument('--phase',choices=['dev','val'],default='dev')
    p.add_argument('--shard',type=int,default=0)
    p.add_argument('--shards',type=int,default=1)
    p.add_argument('--vision-config',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    return p


def inventories(run):
    dev = (run/'second_dev/names.txt').read_text().split()
    val = (run/'second/second_val128.txt').read_text().split()
    if len(dev)!=60 or len(set(dev))!=60 or len(val)!=128 or len(set(val))!=128 or set(dev)&set(val):
        raise ValueError('Require disjoint, unique 60-image development and 128-image report inventories')
    return dict(dev=dev,val=val)


def selected(args):
    from .second_labels import check_encoding
    check_encoding(args.dataset_root)
    if not 0 <= args.shard < args.shards:
        raise ValueError('Require 0 <= shard < shards')
    return inventories(args.run_root)[args.phase][args.shard::args.shards]


def pair_backend(pool,args,name):
    return pool.for_pair([args.dataset_root/part/name for part in ('img1','img2')])


def emit(path,report,markdown):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    tmp.replace(path)
    path.with_suffix('.md').write_text(markdown+'\n')


def fmt(value):
    return '—' if value is None else f'{100*value:.2f}'


def shard_path(args,stem):
    return args.output_root/f'{stem}_{args.phase}_{args.shard}of{args.shards}.json'


def collect(paths,phase,expected_names):
    """Merge per-image evidence, rejecting incomplete, mixed or duplicate shards."""
    reports = [json.loads(Path(p).read_text()) for p in paths]
    rows = {}
    for report in reports:
        if report['phase'] != phase:
            raise ValueError('Wrong report phase')
        if set(rows)&set(report['per_image']):
            raise ValueError('Duplicate image across shards')
        rows.update(report['per_image'])
    if set(rows)!=set(expected_names):
        raise ValueError('Shards do not cover the complete inventory')
    return reports,rows


def backend_config(args,frozen=None):
    config = json.loads(args.vision_config.read_text())
    if frozen is not None and frozen['backend_config'] != config:
        raise ValueError('Backend configuration differs from frozen development settings')
    return config
