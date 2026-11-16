"""Separate provenance recovery candidate; never replaces or claims to read legacy mask."""
import argparse,json,os
from pathlib import Path
import numpy as np
from .n9_t3_data import load_labels,components,geometry,DATASET
from .io_retry import read_image,save_image,write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--task-id',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--dataset',type=Path,default=DATASET);p.add_argument('--compare-original',action='store_true');a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Slurm required')
    row=next(json.loads(s) for s in a.manifest.read_text().splitlines() if s and json.loads(s)['id']==a.task_id)
    provenance=row['meta']['example_provenance'][0];x,y=load_labels(a.dataset,provenance['name'])
    items=components((x==provenance['source'])&(y==provenance['target']));mask=np.zeros(x.shape,bool);mask.ravel()[items[provenance['component']-1]]=1
    g=geometry(mask)
    if g['area']!=provenance['area'] or g['bbox']!=provenance['bbox']:raise ValueError('Source component does not match original provenance')
    x0,y0,x1,y1=provenance['crop_box'];crop=mask[y0:y1,x0:x1]
    if int(crop.sum())!=g['area']:raise ValueError('Crop loses original support')
    save_image(a.output/'recovered_candidate.png',crop.astype(np.uint8)*255)
    result=dict(task_id=a.task_id,status='source_reconstruction_not_original_read',provenance=provenance,geometry=g,
        original_mask=row['examples'][0]['mask'],candidate='recovered_candidate.png',original_unchanged=True,comparison='not_attempted')
    write_json(a.output/'recovery.json',result)
    if a.compare_original:
        original=read_image(row['examples'][0]['mask'])>0
        result['comparison']='pixel_equal' if original.shape==crop.shape and np.array_equal(original,crop) else 'different'
        write_json(a.output/'recovery.json',result)

if __name__=='__main__':main()
