"""Post-round-one mechanism audit; no new policy and no confirm-based fit."""
import argparse
import os
import numpy as np
from scipy import ndimage
from .n7_missed_trigger import read, write

def main():
    from pathlib import Path
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--base',type=Path,required=True);a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Use Slurm')
    cases=read(a.out/'paired_cases.json')
    rows=[r for r in cases if r['endpoint']=='balanced_v0_k10' and r['policy']=='coverage']
    states=[];lines=['# Initial-mask mechanism audit','','No gate is selected in this audit. Shapes below are obtainable from current Mask components; no candidate generation or API call.','',
        '| split | group | image | coverage | components | largest fraction | tiny area fraction <=16px | baseline F1 | fixed ΔF1 | first gate failure |',
        '|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        with np.load(a.base/r['group']/(r['stem']+'.npz'),allow_pickle=False) as z: mask=z['start'].astype(bool)
        labels,n=ndimage.label(mask,structure=np.ones((3,3)))
        areas=np.bincount(labels.ravel())[1:];total=int(mask.sum())
        features=dict(component_count=n,largest_component_fraction=float(areas.max()/total) if total else 0.,
            tiny_area_fraction=float(areas[areas<=16].sum()/total) if total else 0.,
            median_component_area=float(np.median(areas)) if n else 0.,
            components_per_megapixel=n*1e6/mask.size)
        states.append(dict(group=r['group'],stem=r['stem'],split=r['split'],state=r['state'],shape_features=features))
        # Tune outcomes only in feature-selection table. Confirm failures remain in the immutable first-round record.
        if r['split']=='tune':
            delta=r['fixed']['f1']-r['baseline']['f1']
            lines.append(f"| tune | {r['group']} | {r['stem']} | {r['state']['mask_coverage']:.4f} | {n} | {features['largest_component_fraction']:.4f} | {features['tiny_area_fraction']:.4f} | {r['baseline']['f1']:.4f} | {delta:.4f} | {r['failure']} |")
    write(a.out/'initial_shape_states.json',states)
    (a.out/'tune_shape_audit.md').write_text('\n'.join(lines)+'\n')
    failure_lines=['# First-round primary coverage failure cases','','| split | group | image | coverage | called | ΔF1 | extra TP | extra FP | failure |','|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        if r['failure']:
            failure_lines.append(f"| {r['split']} | {r['group']} | {r['stem']} | {r['state']['mask_coverage']:.5f} | {r['called']} | {r['delta_f1']:.4f} | {r['added_tp']} | {r['added_fp']} | {r['failure']} |")
    (a.out/'first_round_failures.md').write_text('\n'.join(failure_lines)+'\n')
    print('Initial-state audit complete; no new policy selected.')

if __name__=='__main__': main()
