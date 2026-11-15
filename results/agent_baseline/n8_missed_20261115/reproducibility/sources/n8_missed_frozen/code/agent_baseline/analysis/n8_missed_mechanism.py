"""Post-confirm diagnostic only: fixed outcomes, paired prompts and candidate errors."""
import argparse
import copy
import os
from pathlib import Path
import numpy as np
from .n8_missed_assets import read,write,OUT
from .n8_missed_prepare import load_masks,truth
from .n8_missed_score import config_id

STAGES=('original_all','s5_dino020_all','direction_routed','dilation_only','area25','semantic','residual_evidence')

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use Slurm')
    prompt_audit=[]
    for split in ('tune','confirm'):
        for row in read(a.out/f'requests_{split}.json'):
            paths=[a.out/'inputs'/v/row['uid']/'request.json' for v in ('V0','V0_evidence')]
            v,e=map(read,paths);em=copy.deepcopy(e['messages']);vm=v['messages']
            base_text=vm[1]['content'][-1]['text'];evidence_text=em[1]['content'][-1]['text']
            prefix=evidence_text.startswith(base_text+'\nQuantitative evidence for the red-contour region only:')
            em[1]['content'][-1]['text']=base_text
            images=[[b['image_url']['url'] for b in r['messages'][1]['content'] if b.get('type')=='image_url'] for r in (v,e)]
            same_images=images[0]==images[1] and all(Path(path).is_file() for path in images[0])
            passed=prefix and em==vm and v['parameters']==e['parameters'] and same_images
            prompt_audit.append(dict(uid=row['uid'],split=split,same_seed=v['parameters']['seed']==e['parameters']['seed'],
                same_parameters=v['parameters']==e['parameters'],same_shared_image_files=same_images,only_evidence_text_differs=prefix and em==vm,passed=passed))
    assert all(r['passed'] for r in prompt_audit),'Paired verifier input mismatch'
    write(a.out/'paired_input_audit.json',dict(pairs=len(prompt_audit),passed=sum(r['passed'] for r in prompt_audit),rows=prompt_audit))
    data=read(a.out/'scores_confirm.json');frozen=read(a.out/'frozen_config.json');primary=data['cases'][config_id(frozen)]
    lookup={(r['group'],r['stem']):r for r in primary}
    manifests={(r['group'],r['stem']):r for r in read(a.out/'fm1_manifest.json')}
    stages=read(a.out/'candidate_diagnostics_confirm.json');stage_groups={}
    for group in ('all',*sorted({r['group'] for r in stages})):
        selected=[r for r in stages if group=='all' or r['group']==group]
        stage_groups[group]={stage:{key:sum(r['stages'][stage][key] for r in selected) for key in ('tp','fp')} for stage in STAGES}
        stage_groups[group]['remaining_fn']=sum(r['remaining_fn'] for r in selected)
    write(a.out/'stage_group_diagnostics.json',stage_groups)
    comparison=[];candidate_rows=[]
    for task in primary:
        group,stem=task['group'],task['stem'];item=manifests[group,stem];masks=load_masks(item['masks']);gt=truth(task['gt_path'],group)
        with np.load(task['baseline_path'],allow_pickle=False) as z:base=z['start'].astype(bool)
        arms={}
        for key,rows in data['cases'].items():
            row=next(r for r in rows if r['group']==group and r['stem']==stem)
            arms[key]={k:row[k] for k in ('metrics','delta_f1','added_tp','added_fp','topk_candidate_tp','topk_candidate_fp','accepted_indexes','submitted_indexes')}
        comparison.append(dict(group=group,stem=stem,baseline=task['baseline'],residual_area=task['residual_area'],trigger=task['trigger'],arms=arms))
        requests=[r for r in read(a.out/'requests_confirm.json') if r['group']==group and r['stem']==stem]
        for request in requests:
            index=request['mask_index'];c=item['candidates'][index];mask=masks[index]&~base
            tp=int((mask&gt).sum());fp=int((mask&~gt).sum())
            v,e=[read(a.out/'answers'/variant/(request['uid']+'.json')) for variant in ('V0','V0_evidence')]
            candidate_rows.append(dict(uid=request['uid'],group=group,stem=stem,index=index,primary_top5=index in task['submitted_indexes'],
                tp=tp,fp=fp,zero_tp=tp==0,pixel_beneficial_proxy=tp>fp,area=c['area'],sources=c['sources'],
                p1=c['p1'],p2=c['p2'],semantic_delta=c['semantic_delta'],dino=c['dino'],both_building_probability_majority=c['p1']>=.5 and c['p2']>=.5,
                V0=v['answer'],V0_evidence=e['answer'],V0_reason=v.get('parsed',{}).get('reason',''),evidence_reason=e.get('parsed',{}).get('reason','')))
    stats={}
    for scope in ('top10_union','primary_top5'):
        rows=[r for r in candidate_rows if scope=='top10_union' or r['primary_top5']]
        stats[scope]=dict(candidates=len(rows),no_to_yes=sum(r['V0']=='no' and r['V0_evidence']=='yes' for r in rows),
            yes_to_no=sum(r['V0']=='yes' and r['V0_evidence']=='no' for r in rows),
            no_to_yes_zero_tp=sum(r['V0']=='no' and r['V0_evidence']=='yes' and r['zero_tp'] for r in rows),
            yes_to_no_zero_tp=sum(r['V0']=='yes' and r['V0_evidence']=='no' and r['zero_tp'] for r in rows),
            V0_accepted_zero_tp=sum(r['V0']=='yes' and r['zero_tp'] for r in rows),
            evidence_accepted_zero_tp=sum(r['V0_evidence']=='yes' and r['zero_tp'] for r in rows))
    write(a.out/'candidate_verifier_audit.json',dict(stats=stats,candidates=candidate_rows,interpretation='TP/FP are pixel-ground-truth overlap diagnostics, not manual semantic labels; do not sum overlapping candidate pixels as final output counts'))
    write(a.out/'case_arm_comparison.json',comparison)
    lines=['# Fixed-confirm mechanism audit','',f"Paired request inputs: {len(prompt_audit)}/{len(prompt_audit)} identical images, seeds and parameters; only evidence text differs.",'',
        '| group | stage | candidate TP | candidate FP |','|---|---|---|---|']
    for group,values in stage_groups.items():
        for stage in STAGES:lines.append(f"| {group} | {stage} | {values[stage]['tp']} | {values[stage]['fp']} |")
    lines+=['','## Verifier response switches',str(stats),'','## Per-case fixed arms',
        '| group/image | arm | F1 | Δ initial | added TP/FP | TopK candidate TP/FP |','|---|---|---|---|---|---|']
    for row in comparison:
        for key,r in row['arms'].items():
            lines.append(f"| {row['group']}/{row['stem']} | {key} | {r['metrics']['f1']:.4f} | {r['delta_f1']:+.4f} | {r['added_tp']}/{r['added_fp']} | {r['topk_candidate_tp']}/{r['topk_candidate_fp']} |")
    lines+=['','## Candidate answer disagreements','',
        '| candidate | TP/FP | ΔP / DINO | V0 → evidence | V0 reason | evidence reason |','|---|---|---|---|---|---|']
    for r in candidate_rows:
        if r['V0']!=r['V0_evidence']:
            clean=lambda x:str(x).replace('|','/').replace('\n',' ')
            lines.append(f"| {r['uid']} | {r['tp']}/{r['fp']} | {r['semantic_delta']:.4f}/{r['dino']:.4f} | {r['V0']} → {r['V0_evidence']} | {clean(r['V0_reason'])} | {clean(r['evidence_reason'])} |")
    (a.out/'mechanism_audit.md').write_text('\n'.join(lines)+'\n')
    # Check actual generated local links and image dimensions, without regenerating pages.
    from html.parser import HTMLParser
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):
            if tag=='a':self.links += [v for k,v in attrs if k=='href']
    pages=[]
    for page in (a.out/'diagnostics').glob('*.html'):
        parser=Links();parser.feed(page.read_text())
        missing=[link for link in parser.links if not link.startswith(('#','http')) and not (page.parent/link).exists()]
        assert not missing,(page,missing)
        pages.append(dict(file=page.name,local_links=len(parser.links),missing=missing))
    write(a.out/'diagnostic_link_audit.json',dict(pages=pages,expected_pages=11,complete=len(pages)==11))
    print('Mechanism audit complete; no predictions or parameters changed.')

if __name__=='__main__':main()
