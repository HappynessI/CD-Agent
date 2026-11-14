"""T5 v3 candidate admission adds offline oracle evidence against trivial edits.

The oracle sees GT only during construction. It is not an Agent/SAM score.
Original v2 scripts remain reproducible and unchanged.
"""
import argparse
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
import revise_t5_review as base
from revise_whu_official import counts, metrics, write

BASE_ANALYSE=base.analyse


def simple_edit_oracles(g,p):
    initial=counts(p,g);total_gt=int(g.sum());initial_error=initial['fp']+initial['fn']
    pl,pn=ndi.label(p,base.S8);gl,gn=ndi.label(g,base.S8)
    pa=np.bincount(pl.ravel(),minlength=pn+1);pt=np.bincount(pl[g].ravel(),minlength=pn+1)
    ga=np.bincount(gl.ravel(),minlength=gn+1);covered=np.bincount(gl[p].ravel(),minlength=gn+1)
    # For any optimum F1 subset, components above its precision threshold are
    # retained. Checking all precision-sorted prefixes gives the exact optimum
    # over arbitrary whole-initial-component deletions (including none/all).
    order=sorted(range(1,pn+1),key=lambda i:(-pt[i]/pa[i],i))
    best_delete=dict(tp=0,fp=0,fn=total_gt);best_delete_f1=metrics(best_delete)['f1'];tp=fp=0;kept=0;best_kept=0
    for i in order:
        tp+=int(pt[i]);fp+=int(pa[i]-pt[i]);kept+=1;c=dict(tp=tp,fp=fp,fn=total_gt-tp);f1=metrics(c)['f1']
        if f1>best_delete_f1:best_delete=c;best_delete_f1=f1;best_kept=kept
    edits=[dict(kind='no_edit',counts=initial)]
    for i in range(1,pn+1):
        t=int(pt[i]);f=int(pa[i]-pt[i]);edits.append(dict(kind='delete_initial_component',component_id=i,counts=dict(tp=initial['tp']-t,fp=initial['fp']-f,fn=initial['fn']+t)))
    for i in range(1,gn+1):
        n=int(ga[i]-covered[i]);edits.append(dict(kind='add_complete_gt_component_oracle',component_id=i,counts=dict(tp=initial['tp']+n,fp=initial['fp'],fn=initial['fn']-n)))
    best_f1=max(edits,key=lambda e:metrics(e['counts'])['f1'])
    best_error=min(edits,key=lambda e:e['counts']['fp']+e['counts']['fn'])
    remaining=best_error['counts']['fp']+best_error['counts']['fn']
    return dict(initial=metrics(initial),initial_components=pn,gt_components=gn,
        arbitrary_whole_component_deletion=dict(metrics=metrics(best_delete),kept_components=best_kept),
        best_single_edit=dict(operation={k:v for k,v in best_f1.items() if k!='counts'},metrics=metrics(best_f1['counts'])),
        best_single_edit_error=dict(operation={k:v for k,v in best_error.items() if k!='counts'},remaining_error_px=remaining,
            error_reduction_fraction=(initial_error-remaining)/initial_error if initial_error else 0.),
        interpretation='GT-assisted best-case edits, not model outcomes. Whole-GT addition is optimistic; this cannot establish difficulty for SAM or other shape tools.')


def analyse(g,p):
    errors,selected,qc=BASE_ANALYSE(g,p);oracle=simple_edit_oracles(g,p)
    easy_single=oracle['best_single_edit_error']['error_reduction_fraction']>=.8 or oracle['best_single_edit']['metrics']['f1']>=.98
    easy_deletion=oracle['arbitrary_whole_component_deletion']['metrics']['f1']>=.98
    qc['geometry_candidate']=qc['core_candidate'];qc['simple_edit_oracle']=oracle
    qc['core_candidate']=bool(qc['core_candidate'] and not easy_single and not easy_deletion and oracle['best_single_edit_error']['remaining_error_px']>=128)
    if qc['geometry_candidate'] and not qc['core_candidate']:qc['exclusion_reason']='simple_oracle_edit_nearly_solves_or_leaves_too_little_error'
    return errors,selected,qc


def build(a):
    base.analyse=analyse
    base.build(a)
    audit=base.json.loads((a.output/'t5/audit.json').read_text());summary=base.json.loads((a.output/'t5/summary.json').read_text())
    summary['protocol']['simple_edit_oracle_gate']='exclude best single whole-component delete/GT-add removes >=80% error or reaches F1>=.98; exclude arbitrary component deletion reaches F1>=.98; require >=128 residual error pixels after best one-edit'
    summary['protocol']['oracle_scope']='offline GT-assisted upper bound for specified edits, not inference; actual SAM and multi-step difficulty remains pending review'
    summary['geometry_candidates_before_oracle']=sum(r['qc']['geometry_candidate'] for r in audit)
    selected=[r['qc']['simple_edit_oracle'] for r in audit if r['qc']['core_candidate']]
    summary['selected_oracle_evidence']=dict(n=len(selected),
        best_single_edit_f1_range=[min(o['best_single_edit']['metrics']['f1'] for o in selected),max(o['best_single_edit']['metrics']['f1'] for o in selected)] if selected else None,
        maximum_single_edit_error_reduction=max((o['best_single_edit_error']['error_reduction_fraction'] for o in selected),default=None),
        maximum_arbitrary_deletion_f1=max((o['arbitrary_whole_component_deletion']['metrics']['f1'] for o in selected),default=None))
    summary['revision']='v3_simple_edit_oracle_screen'
    write(a.output/'t5/summary.json',summary);print(base.json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--review',type=Path,required=True);p.add_argument('--whu',type=Path,required=True);build(p.parse_args())
