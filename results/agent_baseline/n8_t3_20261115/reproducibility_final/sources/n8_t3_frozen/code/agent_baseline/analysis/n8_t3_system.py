"""Inference-only N8 T3 pool construction, relative gate and evidence cards (no GT)."""
import numpy as np
from .n8_t3_common import connected_masks,ranked_features,CHANNELS
from .n8_t3_route import dense_high

Z_GRID=(0.,1.,2.,3.)
P_GRID=(0.,.1,.25,.5)
VERIFY_K=(1,3,5)


def construct_pool(clean,dense,probabilities,observable,system_method):
    structural=observable['structural'];masks=[];sources=[]
    if not structural:
        for mask in connected_masks(dense):masks.append(mask);sources.append(dict(branch='dense_component'))
    elif system_method=='route':
        for i,mask in enumerate(clean):masks.append(mask);sources.append(dict(branch='clean_instance',clean_id=i))
    else:
        _,op,tau=system_method.split('_');high=dense_high(probabilities,observable['predicted_pair'],dense,float(tau))
        for i,mask in enumerate(clean):
            net=mask&high if op=='intersection' else mask
            if net.any():masks.append(net);sources.append(dict(branch='hybrid_instance_'+op,clean_id=i))
        if op=='union':
            for mask in connected_masks(high):masks.append(mask);sources.append(dict(branch='hybrid_dense_component'))
    # Only exact duplicate masks are merged; no score/GT-driven removal or component-area pruning.
    unique=[];provenance=[];seen={}
    for mask,source in zip(masks,sources):
        key=np.packbits(mask).tobytes()
        if key in seen:provenance[seen[key]].append(source)
        else:seen[key]=len(unique);unique.append(mask);provenance.append([source])
    return np.asarray(unique,bool).reshape(-1,*dense.shape),provenance


def relative_margin(scores,order):
    if len(order)<=1:return dict(z=0.,background_mean=None,background_std=None,background_n=max(0,len(order)-1))
    bg=np.asarray([scores[i] for i in order[1:]],float);mean=float(bg.mean());std=float(bg.std())
    return dict(z=(float(scores[order[0]])-mean)/(std+1e-6),background_mean=mean,background_std=std,background_n=len(bg))


def feature_evidence(masks,example,probabilities,observable,backend,rank):
    values,orders=ranked_features(masks,example,backend);order=orders[rank]
    c1,c2=observable['predicted_pair'];transition=probabilities[0][c1]*probabilities[1][c2]
    candidates=[]
    for i,mask in enumerate(masks):
        means=[p[:,mask].mean(axis=1).tolist() for p in probabilities]
        own=[int(np.argmax(m[1:]))+1 for m in means]
        candidates.append(dict(candidate_id=i,area=int(mask.sum()),similarity=float(values[rank][i]),
            delta_similarity=float(values['delta'][i]),joint_similarity=float(values['joint'][i]),
            mean_probabilities=means,predicted_pair=own,ptrans=float(transition[mask].mean())))
    return values,order,candidates,relative_margin(values[rank],order)


def passes_gate(count,z,ptrans,z_threshold,p_threshold):
    return bool(count>0 and z>=z_threshold and ptrans>=p_threshold)


def evidence_card(observable,candidate,rank):
    def transition(pair,means):
        return f"{CHANNELS[pair[0]]}({means[0][pair[0]]:.4f}) -> {CHANNELS[pair[1]]}({means[1][pair[1]]:.4f})"
    example=transition(observable['predicted_pair'],observable['example_mean_probabilities'])
    current=transition(candidate['predicted_pair'],candidate['mean_probabilities'])
    # Exactly two numeric evidence lines; categories come exclusively from predicted probabilities.
    return (f"Example evidence: area={observable['area']} native pixels; SegEarth T1 -> T2 dominant probabilities: {example}.\n"
            f"Candidate evidence: area={candidate['area']} native pixels; {rank} exemplar similarity={candidate['similarity']:.4f}; "
            f"SegEarth T1 -> T2 dominant probabilities: {current}; example-directed transition consistency={candidate['ptrans']:.4f}.")
