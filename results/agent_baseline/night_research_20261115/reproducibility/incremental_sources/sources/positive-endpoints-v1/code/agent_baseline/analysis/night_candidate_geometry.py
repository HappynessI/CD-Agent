"""Fixed tune candidate geometry audit. Labels enter offline scoring only."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from .io_retry import read_image,write_json
from .missed_sources import load_npz,unpack,specs,rows_for,pixel_counts
from .missed_verifier_study import metrics
from .night_missed_views import VARIANTS,answer_path


def load(path):return json.loads(Path(path).read_text())


def base_geometry(base):
    structure=np.ones((3,3),bool)
    return ndi.binary_fill_holes(base)&~base,ndi.binary_dilation(base,structure=structure)&~base


def geometry(mask,base,context=None):
    holes,adjacent=context if context is not None else base_geometry(base)
    structure=np.ones((3,3),bool);area=int(mask.sum());_,components=ndi.label(mask,structure)
    core=ndi.binary_erosion(mask,structure=structure);edge_fraction=1-float(core.sum())/area if area else 0
    hole_fraction=float((mask&holes).sum())/area if area else 0
    adjacency_fraction=float((mask&adjacent).sum())/area if area else 0
    y,x=np.nonzero(mask);bbox_area=int((x.max()-x.min()+1)*(y.max()-y.min()+1)) if area else 0
    if hole_fraction>=.5:kind='internal_hole'
    elif adjacency_fraction>0:kind='adjacent_edge_fragment' if edge_fraction>=.5 else 'adjacent_region'
    else:kind='detached_fragment' if edge_fraction>=.5 else 'detached_region'
    return dict(stratum=kind,area=area,components=int(components),baseline_overlap_pixels=int((mask&base).sum()),
        hole_fraction=hole_fraction,baseline_adjacency_fraction=adjacency_fraction,
        edge_fraction=edge_fraction,bbox_fill=area/bbox_area if bbox_area else None)


def classify_answer(answer,tp,fp):
    positive=tp>fp;result=dict(proxy_positive=int(positive),proxy_negative=int(not positive),
        true_yes=0,false_yes=0,true_no=0,false_no=0,missing=int(answer is None),unknown=0,truncated=0,parse_failure=0)
    if answer is None:return result
    result['truncated']=int(answer.get('finish_reason')=='length')
    result['parse_failure']=int(bool(answer.get('parse_failure')))
    choice=answer.get('answer');result['unknown']=int(choice not in ('yes','no'))
    if choice in ('yes','no'):result[('true' if positive==(choice=='yes') else 'false')+'_'+choice]=1
    return result


def run(a):
    items=[r for r in load(a.views/'manifest.json') if r['split']=='tune']
    lookup={s['group']:{Path(r['id']).stem:r for r in rows_for(a.run,s)} for s in specs(a.run,a.work) if not s['group'].startswith('whu')}
    policy=dict(split='frozen tune only',candidate_pool='all prepared Top20; also fixed Top10 prefix',api_requests=0,
        geometry='Native pixels: 8-connected components; scipy fill_holes default4 background; baseline adjacency Chebyshev1; edge=mask minus 3x3 erosion',
        strata='hole_fraction>=0.5 first; else any baseline adjacency; edge_fraction>=0.5 fragment; remaining detached_region',
        interpretation='Geometry proxies, not whole-object/semantic ground truth; fixed thresholds are descriptive, never online filtering',
        errors='TP>FP proxy, not semantic annotation; parse/truncation/unknown retained separately',
        denominator='Candidate confusion counts include overlapping candidates; pixel TP/FP use per-image union within each stratum; different strata may overlap and must not be summed as a global pixel count',
        no_source_regeneration=True)
    summary={};records=[]
    for item in items:
        base=load_npz(a.work/'base'/item['group']/(item['stem']+'.npz'))['start'];context=base_geometry(base)
        data=load_npz(Path(item['masks']));masks=list(unpack(data['masks'],tuple(data['shape'])))
        truth=read_image(lookup[item['group']][item['stem']]['gt_mask'])>(127 if item['group'].startswith('levir') else 0)
        candidates=[]
        for index,(candidate,mask) in enumerate(zip(item['candidates'],masks),1):
            tp,fp,_=pixel_counts(mask,truth);answers={}
            for variant in VARIANTS:
                path=answer_path(argparse.Namespace(pilot=a.pilot,out=a.views),variant,candidate['uid'])
                answer=load(path) if path.exists() else None
                answers[variant]=dict(answer=answer.get('answer') if answer else None,errors=classify_answer(answer,tp,fp),
                    answer_path=str(path),input_path=answer.get('input_path') if answer else None)
            candidates.append(dict(uid=candidate['uid'],rank=index,source=candidate.get('source'),sources=candidate.get('sources'),
                geometry=geometry(mask,base,context),tp=int(tp),fp=int(fp),answers=answers))
        row=dict(group=item['group'],stem=item['stem'],baseline=metrics(pixel_counts(base,truth)),candidates=candidates,strata={})
        for k in (10,20):
            for kind in sorted({c['geometry']['stratum'] for c in candidates[:k]}):
                indexes=[i for i,c in enumerate(candidates[:k]) if c['geometry']['stratum']==kind]
                covered=np.zeros_like(base)
                for i in indexes:covered|=masks[i]
                for variant in VARIANTS:
                    accepted=np.zeros_like(base);errors={key:0 for key in candidates[indexes[0]]['answers'][variant]['errors']}
                    for i in indexes:
                        answer=candidates[i]['answers'][variant]
                        for key,value in answer['errors'].items():errors[key]+=value
                        if answer['answer']=='yes':accepted|=masks[i]
                    tp,fp,_=pixel_counts(accepted,truth);ct,cf,_=pixel_counts(covered,truth)
                    key='/'.join((item['group'],str(k),kind,variant));stats=summary.setdefault(key,dict(images=0,candidates=0,candidate_components=0,
                        candidate_union_tp=0,candidate_union_fp=0,accepted_union_tp=0,accepted_union_fp=0,errors={key:0 for key in errors}))
                    stats['images']+=1;stats['candidates']+=len(indexes);stats['candidate_components']+=sum(candidates[i]['geometry']['components'] for i in indexes)
                    for name,value in (('candidate_union_tp',ct),('candidate_union_fp',cf),('accepted_union_tp',tp),('accepted_union_fp',fp)):stats[name]+=int(value)
                    for name,value in errors.items():stats['errors'][name]+=value
                    row['strata']['/'.join((str(k),kind,variant))]=dict(candidates=len(indexes),accepted_tp=int(tp),accepted_fp=int(fp),
                        delta_f1=metrics(pixel_counts(base|accepted,truth))['f1']-row['baseline']['f1'],errors=errors)
        write_json(a.out/'images'/item['group']/(item['stem']+'.json'),row);records.append(row)
        write_json(a.out/'progress.json',dict(completed=len(records),expected=len(items),last=item['stem'],api_requests=0))
    write_json(a.out/'summary.json',dict(policy=policy,groups=summary,images=records))
    lines=['# 固定 tune 候选几何诊断','',policy['interpretation'],policy['denominator'],'',
        '|组/K/几何/臂|图数|候选/组件|候选union TP/FP|接受union TP/FP|proxy TP/FP/TN/FN|missing/unknown/truncated/parse|',
        '|---|---|---|---|---|---|---|']
    for key,v in summary.items():
        e=v['errors'];lines.append('|'+ '|'.join((key,str(v['images']),f"{v['candidates']}/{v['candidate_components']}",
            f"{v['candidate_union_tp']}/{v['candidate_union_fp']}",f"{v['accepted_union_tp']}/{v['accepted_union_fp']}",
            '/'.join(str(e[n]) for n in ('true_yes','false_yes','true_no','false_no')),
            '/'.join(str(e[n]) for n in ('missing','unknown','truncated','parse_failure'))))+'|')
    lines+=['','内部洞/邻接碎片不自动等于正确修补；独立区域不自动等于整栋缺失。这里只检验几何分层是否解释差异，不据GT改候选或选择新问法。逐例图像/答案路径与净ΔF1见summary.json。']
    (a.out/'report.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser()
    for name in ('views','pilot','work','run','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Run diagnosis via Slurm')
    run(a)


if __name__=='__main__':main()
