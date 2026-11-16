"""N9 fixed retrieval inference: RGB/E/features only, no GT or transition labels."""
import numpy as np
from agent_baseline.instances import from_dense
from agent_baseline.tools_t13 import Features,pooled,pool_features,cosine
from .t3_geometry import descriptor

RANKS=('delta','joint')
KS=(5,8)


def geometry_indices(masks,query_images,example,reference_images):
    ref=descriptor(example,reference_images);kept=[]
    if ref is None:raise ValueError('Example must be nonempty')
    for i,mask in enumerate(masks):
        c=descriptor(mask,query_images)
        if c is not None and .25<=c[0]/ref[0]<=4 and .333333<=c[1]/ref[1]<=3 and np.linalg.norm(c[2]-ref[2])/np.sqrt(6)<=.25:
            kept.append(i)
    return kept


def clean_candidates(masks,example,same_image):
    if not same_image:return masks.copy(),list(range(len(masks))),[]
    kept=[];ids=[];audit=[];earea=int(example.sum())
    for i,mask in enumerate(masks):
        area=int(mask.sum());intersection=int((mask&example).sum())
        overlap=intersection/max(1,area);iou=intersection/max(1,area+earea-intersection)
        removed=overlap>=.15 or iou>=.15
        audit.append(dict(original_id=i,raw_area=area,intersection=intersection,overlap=overlap,iou=iou,
                          removed=removed,net_area=int((mask&~example).sum())))
        if not removed:kept.append(mask&~example);ids.append(i)
    # N9 has no N8 minimum net-area filter, no semantic route and no relative gate.
    return np.asarray(kept,bool).reshape(-1,*example.shape),ids,audit


def scores(masks,query_features,reference_features,example):
    indices=np.flatnonzero(example)
    e1,e2=[pool_features(f,indices) for f in reference_features]
    if not len(masks):return {r:np.zeros(0,np.float32) for r in RANKS}
    items=from_dense(masks,np.ones(len(masks)))
    c1,c2=[pooled(f,items) for f in query_features]
    delta=cosine(c2-c1,e2-e1)
    endpoint=.5*(cosine(c1,e1)+cosine(c2,e2))
    return dict(delta=delta,joint=.5*(delta+endpoint))


def order(values):return sorted(range(len(values)),key=lambda i:(-float(values[i]),i))


def retrieve(query_images,reference_images,example,query_backend,reference_backend,same_image):
    raw=np.concatenate([query_backend('segment_everything',time=t) for t in ('t1','t2')])
    ids=geometry_indices(raw,query_images,example,reference_images);original=raw[ids]
    query_features=[Features(query_backend('extract_features',time=t),query_images[0].shape[:2],t) for t in ('t1','t2')]
    reference_features=query_features if same_image else [Features(reference_backend('extract_features',time=t),example.shape,t) for t in ('t1','t2')]
    original_scores=scores(original,query_features,reference_features,example)
    clean,clean_ids,audit=clean_candidates(original,example,same_image)
    clean_scores=scores(clean,query_features,reference_features,example)
    original_order=order(original_scores['delta']);orders={r:order(clean_scores[r]) for r in RANKS}
    inference=dict(raw_candidates=len(raw),r1_candidates=len(original),r1_raw_ids=ids,clean_candidates=len(clean),
                   clean_original_ids=clean_ids,cleaning=audit,same_image=same_image,
                   original_delta_scores=original_scores['delta'].tolist(),original_delta_order=original_order,
                   clean_scores={k:v.tolist() for k,v in clean_scores.items()},clean_orders=orders,
                   input_policy='RGB query pair, RGB reference pair, E, SAM proposals and DINO features only; no GT/classes/task identifiers',
                   fresh_example_features=True,net_region_features_recomputed=True,global_gate=False,semantic_route=False,
                   micro_overlap_policy='C minus E; no minimum net area or score threshold')
    return original,clean,inference
